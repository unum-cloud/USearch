/**
 *  @file src_stub/sequence.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_sequence_* C ABI, scoring every stored sequence
 *      against every query so a binding can be exercised without the engine.
 *
 *  Variable-length payloads cannot ride a strided run, so this modality's batches are tapes - the
 *  bytes end to end around Apache Arrow offsets - the shape most worth testing here, because a
 *  binding that handles strides correctly may still mishandle an offset off by one. The costs are
 *  the other half: a containment takes none and no costs go without similarities, so a binding that
 *  pairs the wrong fields learns at once.
 *
 *  @section sequence_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order. The
 *  alignment kernels are textbook dynamic programming with no band, no vectorization, and quadratic
 *  cost, so a corpus of long sequences is slow rather than wrong. Alignments report the unrounded
 *  costs their similarities give, which rank correctly but are not the engine's scale. Refinement
 *  keeps no edges to re-select, so its @c changed is a constant zero rather than a settling signal,
 *  and a caller must not read it as one. An overlap charges the unrounded costs its similarities
 *  give, where the engine rounds them to integers of at most 15. A containment reports the share of
 *  the query's windows the stored side lacks alone, and folds the case of ASCII letters where the
 *  engine folds Unicode's. A stored sequence reads back as its raw bytes, never as class codes or
 *  folded text. A checkpoint holds its own costs, so a load or view takes none, and a view copies
 *  its checkpoint rather than mapping it. Traversal statistics are plausible fiction: shaped
 *  correctly, invented numerically. A checkpoint is a binary blob of this file's own memory image,
 *  readable by nothing else.
 *
 *  Keeping no edges, the stand-in links every one of its entries to its exact nearest at the frozen
 *  connectivity, the lists a perfect graph would hold where the engine's are the ones it stored. A
 *  cursor walks those lists breadth-first from its start keys, a reshard's split grows its parts
 *  over them, each target keeping the source's storage order, and a sample walks them as the
 *  library's plan walks its graph, placing its draws as the library does.
 */

#include <cmath> // `std::floor`, `std::ceil`

#include <algorithm> // `std::stable_sort`
#include <array>     // `std::array`
#include <limits>    // `std::numeric_limits`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/sequence_index.h"
#include "usearch/sequence/alphabet.hpp" // `protein_similarities`, `dna_similarities`, `dna_complement`

#include "lib.hpp"

namespace util = unum::usearch::utilities;

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Storage

/** The alphabet of a byte-level similarity matrix, one row and column per byte. */
constexpr std::size_t alphabet_k = 256;

/** Classes the similarities of an alphabet fold bytes into at most. */
constexpr std::size_t classes_limit_k = 32;

/** What a handle froze at construction, or adopted from a checkpoint: the metric, the costs it was
 *  given or its metric's own, and what they fold into. */
struct sequence_index_options_t {

    /** What entries are compared by. */
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity = 0;

    /** Bytes of the sketch the handle was built with; zero for the metric's default. */
    std::size_t sketch_bytes = 0;

    /** Upper bound on entries sharing one key. */
    std::size_t values_per_key = 0;

    /** Row-major similarities, @c alphabet_k rows of @c alphabet_k; empty for a containment. */
    std::vector<util::i8_t> similarities;

    /** Cost of one unaligned symbol; zero for a containment. */
    util::i8_t gap = 0;

    /** The class the similarities fold every byte into; empty for a containment. */
    std::vector<util::u8_t> byte_classes;

    /** Classes the similarities fold bytes into; zero for a containment. */
    std::size_t classes = 0;
};

/** The windows a containment and an overlap compare, as the header says. */
constexpr std::size_t containment_widths_k[] = {5, 8};
constexpr std::size_t overlap_widths_k[] = {12, 19};

using sequence_payload_t = std::vector<std::byte>;
using sequence_entry_t = entry<sequence_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<sequence_entry_t>,
              "reallocation must move entries, never copy them");

using sequence_index_t = handle<sequence_payload_t, sequence_index_options_t>;

using sequence_cursor_t = graph_cursor<sequence_index_t>;

/** Whether a handle frozen at @p other takes entries of an index frozen at @p index: the same
 *  kind, costs and connectivity. */
static bool same_kind(sequence_index_options_t const& index, sequence_index_options_t const& other) {
    return index.metric == other.metric && index.connectivity == other.connectivity &&
           index.similarities == other.similarities && index.gap == other.gap;
}

/** Whether the kind is one this build implements at all. */
static bool known_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_protein_alignment_k || metric == usearch_sequence_metric_dna_overlap_k ||
           metric == usearch_sequence_metric_utf8_containment_k;
}

/**
 *  @brief Folds the similarities @p options holds into the class of every byte; false once
 *      @p outcome records why the costs are refused.
 *
 *  Bytes sharing their rows and columns share a class, numbered in the order their first bytes
 *  come. A DNA overlap's must fold into the four letters' classes, one apiece, with the letters'
 *  complements kept apart.
 */
static bool fold_costs(sequence_index_options_t& options, usearch_outcome_t* outcome) {
    std::vector<util::i8_t> const& similarities = options.similarities;
    options.byte_classes.clear(), options.classes = 0;
    std::vector<std::size_t> firsts;
    auto const score = [&](std::size_t first, std::size_t second) { return similarities[first * alphabet_k + second]; };
    auto const alike = [&](std::size_t first, std::size_t second) {
        for (std::size_t other = 0; other != alphabet_k; ++other)
            if (score(first, other) != score(second, other) || score(other, first) != score(other, second))
                return false;
        return true;
    };
    std::size_t const folded_bytes = similarities.empty() ? 0 : alphabet_k;
    for (std::size_t byte = 0; byte != folded_bytes; ++byte) {
        std::size_t code = 0;
        while (code != firsts.size() && !alike(firsts[code], byte)) ++code;
        if (code == firsts.size()) firsts.push_back(byte);
        if (reject_if(firsts.size() > classes_limit_k, usearch_status_config_mismatch_k,
                      "The similarities fold into more than 32 classes", outcome))
            return false;
        options.byte_classes.push_back(static_cast<util::u8_t>(code));
    }
    options.classes = firsts.size();
    if (reject_if(options.classes == 1, usearch_status_config_mismatch_k, "The similarities fold into one class",
                  outcome) ||
        reject_if(options.classes && options.gap < 0, usearch_status_config_mismatch_k, "The gap cost is negative",
                  outcome))
        return false;
    for (std::size_t const first : firsts)
        for (std::size_t const second : firsts)
            if (reject_if(score(first, first) + score(second, second) < 2 * score(first, second),
                          usearch_status_config_mismatch_k, "A pair scores above its mean self-score", outcome))
                return false;
    if (options.metric != usearch_sequence_metric_dna_overlap_k) return true;
    constexpr std::string_view letters_k = "ACGT";
    auto const class_of = [&](char letter) { return options.byte_classes[static_cast<unsigned char>(letter)]; };
    util::u32_t taken = 0;
    std::array<util::u8_t, classes_limit_k> complements {};
    for (char const letter : letters_k)
        taken |= 1u << class_of(letter), complements[class_of(letter)] = class_of(us::sequence::dna_complement(letter));
    bool split = options.classes != letters_k.size() || taken != (1u << letters_k.size()) - 1u;
    for (std::size_t byte = 0; byte != alphabet_k && !split; ++byte)
        if (char const complement = us::sequence::dna_complement(static_cast<char>(byte)))
            split = complements[options.byte_classes[byte]] != class_of(complement);
    return !reject_if(split, usearch_status_config_mismatch_k, "Costs must fold into the four letters' classes",
                      outcome);
}

#pragma endregion Storage

#pragma region Scoring

/** Global alignment with a linear gap under costs: the least cost turning @p first into @p second,
 *  a mismatch costing the mean of the two symbols' self-similarities less their shared similarity
 *  and every unaligned symbol @c gap, as the header defines the score. */
static double costed_alignment(util::bytes_view_t first, util::bytes_view_t second,
                               sequence_index_options_t const& parameters, std::vector<double>& row) {
    auto const score = [&](std::byte a, std::byte b) {
        return static_cast<double>(
            parameters.similarities[std::to_integer<std::size_t>(a) * alphabet_k + std::to_integer<std::size_t>(b)]);
    };
    double const gap = parameters.gap;
    row.resize(second.size() + 1);
    for (std::size_t column = 0; column != row.size(); ++column) row[column] = static_cast<double>(column) * gap;
    for (std::size_t above = 0; above != first.size(); ++above) {
        double diagonal = row[0];
        row[0] = static_cast<double>(above + 1) * gap;
        for (std::size_t column = 0; column != second.size(); ++column) {
            std::byte const a = first[above], b = second[column];
            double const cost = (score(a, a) + score(b, b)) / 2 - score(a, b);
            double const aligned = diagonal + cost;
            diagonal = row[column + 1];
            row[column + 1] = util::min_of(aligned, util::min_of(row[column], row[column + 1]) + gap);
        }
    }
    return row[second.size()];
}

/** One window of a sequence and the place it starts at. */
struct sequence_placed_window_t {
    std::string_view window;
    std::size_t place;
};

/** Reusable buffers, so scoring a whole corpus allocates once rather than per pair. */
struct sequence_search_scratch_t {
    std::vector<double> cost_row;
    std::vector<std::string_view> query_windows;
    std::vector<std::string_view> stored_windows;
    std::vector<sequence_placed_window_t> placed_windows;
    std::vector<std::ptrdiff_t> shifts;
    std::vector<std::byte> query_classes;
    std::vector<std::byte> stored_classes;
    std::vector<std::byte> complement;
};

/** Windows of @p width in a sequence of @p length bytes. */
static std::size_t windows_in(std::size_t length, std::size_t width) { return length < width ? 0 : length - width + 1; }

/** Containment: the query's distinct windows the stored side lacks, over its distinct windows. */
static double window_containment(util::bytes_view_t query, util::bytes_view_t stored,
                                 std::span<std::size_t const> widths, std::vector<std::string_view>& query_windows,
                                 std::vector<std::string_view>& stored_windows) {
    std::string_view const query_text = util::chars_of(query), stored_text = util::chars_of(stored);
    std::size_t distinct = 0, held = 0;
    for (std::size_t const width : widths) {
        query_windows.clear();
        stored_windows.clear();
        for (std::size_t start = 0; start != windows_in(query_text.size(), width); ++start)
            query_windows.push_back(query_text.substr(start, width));
        for (std::size_t start = 0; start != windows_in(stored_text.size(), width); ++start)
            stored_windows.push_back(stored_text.substr(start, width));
        std::sort(query_windows.begin(), query_windows.end());
        query_windows.erase(std::unique(query_windows.begin(), query_windows.end()), query_windows.end());
        std::sort(stored_windows.begin(), stored_windows.end());
        for (std::string_view const window : query_windows)
            held += std::binary_search(stored_windows.begin(), stored_windows.end(), window);
        distinct += query_windows.size();
    }
    return distinct ? static_cast<double>(distinct - held) / static_cast<double>(distinct) : 1.0;
}

/**
 *  @brief Shifted overlap, as the header's reference computes it: the windows of @p query_text
 *      found once in it vote for a shift against @p stored_text, and the symbols the winning shift
 *      lays over each other are aligned inside a band under the costs of @p parameters.
 *
 *  The two texts are what windows compare, the class of every byte, and @p query and @p stored are
 *  the bytes the costs read.
 */
static double shifted_overlap(util::bytes_view_t query, util::bytes_view_t stored, std::string_view query_text,
                              std::string_view stored_text, sequence_index_options_t const& parameters,
                              sequence_search_scratch_t& scratch) {
    constexpr double scale_k = static_cast<double>(1u << 23), ceiling_k = 32767.0 * 256 + 255;
    std::vector<sequence_placed_window_t>& placed = scratch.placed_windows;
    std::vector<std::ptrdiff_t>& shifts = scratch.shifts;
    shifts.clear();
    for (std::size_t const width : overlap_widths_k) {
        placed.clear();
        for (std::size_t place = 0; place != windows_in(query_text.size(), width); ++place)
            placed.push_back({query_text.substr(place, width), place});
        std::sort(placed.begin(), placed.end(),
                  [](sequence_placed_window_t const& first, sequence_placed_window_t const& second) {
                      return first.window < second.window;
                  });
        // A window the query repeats names no single shift, so only the ones it holds once vote.
        for (std::size_t place = 0; place != windows_in(stored_text.size(), width); ++place) {
            std::string_view const window = stored_text.substr(place, width);
            std::size_t const found = static_cast<std::size_t>(
                std::lower_bound(placed.begin(), placed.end(), window,
                                 [](sequence_placed_window_t const& lower, std::string_view higher) {
                                     return lower.window < higher;
                                 }) -
                placed.begin());
            if (found == placed.size() || placed[found].window != window) continue;
            if (found + 1 != placed.size() && placed[found + 1].window == window) continue;
            shifts.push_back(static_cast<std::ptrdiff_t>(place) - static_cast<std::ptrdiff_t>(placed[found].place));
        }
    }
    // The most voted shift, the least one winning a tie.
    std::sort(shifts.begin(), shifts.end());
    std::ptrdiff_t shift = 0;
    std::size_t held = 0;
    for (std::size_t first = 0, last = 0; first != shifts.size(); first = last) {
        while (last != shifts.size() && shifts[last] == shifts[first]) ++last;
        if (last - first > held) held = last - first, shift = shifts[first];
    }
    if (held < 2) return ceiling_k / scale_k;

    std::ptrdiff_t const query_size = static_cast<std::ptrdiff_t>(query.size());
    std::ptrdiff_t const stored_size = static_cast<std::ptrdiff_t>(stored.size());
    std::size_t const low = static_cast<std::size_t>(util::max_of<std::ptrdiff_t>(0, -shift));
    std::size_t const high = static_cast<std::size_t>(util::min_of(query_size, stored_size - shift));
    std::size_t const span = high - low, band = util::min_of<std::size_t>(64, util::max_of<std::size_t>(4, span / 16));
    util::bytes_view_t const aligned_query = query.subspan(low, span);
    util::bytes_view_t const aligned_stored = stored.subspan(
        static_cast<std::size_t>(static_cast<std::ptrdiff_t>(low) + shift), span);
    double const gap = parameters.gap;
    auto const score = [&](std::byte a, std::byte b) {
        return static_cast<double>(
            parameters.similarities[std::to_integer<std::size_t>(a) * alphabet_k + std::to_integer<std::size_t>(b)]);
    };
    auto const cost = [&](std::byte a, std::byte b) { return (score(a, a) + score(b, b)) / 2 - score(a, b); };
    // One row of the table, the cells further than the band from the diagonal left unreachable.
    std::vector<double>& row = scratch.cost_row;
    row.assign(span + 1, std::numeric_limits<double>::infinity());
    for (std::size_t column = 0; column <= util::min_of(band, span); ++column)
        row[column] = static_cast<double>(column) * gap;
    for (std::size_t above = 1; above <= span; ++above) {
        std::size_t const first = above > band ? above - band : 0, last = util::min_of(span, above + band);
        double diagonal = row[first ? first - 1 : 0];
        if (first) row[first - 1] = std::numeric_limits<double>::infinity();
        else row[0] = static_cast<double>(above) * gap;
        for (std::size_t column = util::max_of<std::size_t>(first, 1); column <= last; ++column) {
            double const aligned = diagonal + cost(aligned_query[above - 1], aligned_stored[column - 1]);
            diagonal = row[column];
            row[column] = util::min_of(aligned, util::min_of(row[column - 1], row[column]) + gap);
        }
    }
    std::size_t const shorter = util::min_of(query.size(), stored.size()),
                      longer = util::max_of(query.size(), stored.size());
    double const charged = row[span] + gap * static_cast<double>(shorter - span);
    double const most = 2 * gap * static_cast<double>(shorter);
    double const primary = most > 0      ? util::min_of(32767.0, std::floor(charged * 32767 / most))
                           : charged > 0 ? 32767.0
                                         : 0.0;
    double const secondary = std::ceil(static_cast<double>(longer - span) * 255 / static_cast<double>(longer));
    return (primary * 256 + secondary) / scale_k;
}

/** The class of every byte of @p sequence in @p classes, as an overlap's windows compare them. */
static util::bytes_view_t window_classes(sequence_index_options_t const& parameters, util::bytes_view_t sequence,
                                         std::vector<std::byte>& classes) {
    classes.resize(sequence.size());
    for (std::size_t symbol = 0; symbol != sequence.size(); ++symbol)
        classes[symbol] = static_cast<std::byte>(
            parameters.byte_classes[std::to_integer<std::size_t>(sequence[symbol])]);
    return classes;
}

/** @p sequence with the case of its ASCII letters folded, into @p folded. */
static util::bytes_view_t folded_text(util::bytes_view_t sequence, std::vector<std::byte>& folded) {
    folded.assign(sequence.begin(), sequence.end());
    for (std::byte& symbol : folded)
        if (symbol >= std::byte {'A'} && symbol <= std::byte {'Z'}) symbol |= std::byte {'a' - 'A'};
    return folded;
}

/** Scores @p query as it stands against @p stored under the frozen kind. The constructor and
 *  @c adopt refuse a kind this build does not know, so every enumerator here is one of three. */
static double image_distance(sequence_index_options_t const& parameters, util::bytes_view_t query,
                             util::bytes_view_t stored, sequence_search_scratch_t& scratch) {
    switch (parameters.metric) {
    case usearch_sequence_metric_utf8_containment_k:
        return window_containment(folded_text(query, scratch.query_classes),
                                  folded_text(stored, scratch.stored_classes), containment_widths_k,
                                  scratch.query_windows, scratch.stored_windows);
    case usearch_sequence_metric_dna_overlap_k:
        return shifted_overlap(query, stored, util::chars_of(window_classes(parameters, query, scratch.query_classes)),
                               util::chars_of(window_classes(parameters, stored, scratch.stored_classes)), parameters,
                               scratch);
    default: return costed_alignment(query, stored, parameters, scratch.cost_row);
    }
}

/** Scores one pair under the frozen kind: the query as it stands and, for a DNA overlap, as its
 *  reverse complement, the lower distance counting. */
static double distance(sequence_index_t const& index, util::bytes_view_t query, util::bytes_view_t stored,
                       sequence_search_scratch_t& scratch) {
    sequence_index_options_t const& parameters = index.options;
    double const forward = image_distance(parameters, query, stored, scratch);
    if (parameters.metric != usearch_sequence_metric_dna_overlap_k) return forward;
    scratch.complement.resize(query.size());
    for (std::size_t symbol = 0; symbol != query.size(); ++symbol) {
        // A byte that is no letter reads as T, whose complement is A.
        char const letter = us::sequence::dna_complement(std::to_integer<char>(query[query.size() - 1 - symbol]));
        scratch.complement[symbol] = static_cast<std::byte>(letter ? letter : 'A');
    }
    return util::min_of(forward, image_distance(parameters, scratch.complement, stored, scratch));
}

/** Refuses a tape of anything but bytes, the one element a sequence is made of. */
static bool reject_bytes(usearch_tape_view_t tape, usearch_outcome_t* outcome) {
    return reject_if(tape.scalar != usearch_scalar_u8_k, usearch_status_unknown_kind_k, "A sequence tape holds u8",
                     outcome);
}

/** Entry @p entry of a tape, as the bytes its offsets bound. */
static util::bytes_view_t viewed(usearch_tape_view_t tape, std::size_t entry) {
    std::size_t const begin = offset_at(tape, entry), end = offset_at(tape, entry + 1);
    return util::bytes_of(static_cast<std::uint8_t const*>(tape.data) + begin, end - begin);
}

/** The output twin of @c viewed: writes the entries at @p entries into @p arena as one tape,
 *  writing @p tape only once every byte is read, and false once @p outcome records why not. */
static bool copy_entries(sequence_index_t const& index, std::span<std::size_t const> entries,
                         usearch_arena_handle_t& arena, usearch_tape_view_t& tape, usearch_outcome_t* outcome) {
    std::size_t total_bytes = 0;
    for (std::size_t entry : entries) total_bytes += index.entries[entry].payload.size();
    std::size_t* const offsets = carve<std::size_t>(arena, entries.size() + 1, outcome);
    std::byte* const bytes = offsets ? carve<std::byte>(arena, total_bytes, outcome) : nullptr;
    if (!bytes) return false;
    offsets[0] = 0;
    for (std::size_t found = 0; found != entries.size(); ++found) {
        sequence_payload_t const& payload = index.entries[entries[found]].payload;
        std::copy(payload.begin(), payload.end(), bytes + offsets[found]);
        offsets[found + 1] = offsets[found] + payload.size();
    }
    write_output_tape(bytes, offsets, usearch_scalar_u8_k, tape);
    return true;
}

/**
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
static std::size_t search_tape(sequence_index_t& index, usearch_tape_view_t queries, std::size_t count,
                               matches_outputs_t const& outputs, std::size_t expansion_factor,
                               usearch_traversal_stats_t* stats, usearch_predicate_t predicate,
                               usearch_progress_t progress) {
    std::size_t const entries = index.entries.size();
    if (stats && expansion_factor) stats->computed_distances += expansion_factor * count;

    sequence_search_scratch_t scratch;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != count; ++query) {
        util::bytes_view_t const query_view = viewed(queries, query);
        rank<sequence_payload_t>(
            std::span<sequence_entry_t const>(index.entries).first(entries), predicate,
            [&](sequence_payload_t const& payload) { return distance(index, query_view, payload, scratch); }, ranked);
        outputs.write(query, ranked);
        if (stats) {
            stats->visited_members += entries;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(progress, query + 1, count)) return query + 1;
    }
    return count;
}

#pragma endregion Scoring

#pragma region Serialization

/**
 *  @brief What @c load needs before it can read an entry. The blob is this file's own memory
 *      image rather than a format: no magic, no version, and nothing else reads it back.
 *
 *  Entries are ragged, so the payload is a run of `{key, length, bytes}` records rather than
 *  fixed-width ones, and the prefix holds their summed extent to size the read.
 */
struct sequence_checkpoint_prefix_t {

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Bytes of the sketch the handle was built with. */
    std::size_t sketch_bytes;

    /** How many entries follow the prefix. */
    std::size_t entries;

    /** Bytes of entry payload that follow, keys and lengths included. */
    std::size_t payload_bytes;

    /** What entries are compared by. */
    usearch_sequence_metric_t metric;

    /** Classes the similarities fold bytes into; zero for a containment. */
    std::size_t classes;

    /** Whether a similarity matrix follows. */
    bool costed;

    /** Cost of one unaligned symbol. */
    util::i8_t gap;
};

/** The per-entry overhead of the blob: the key, then the byte length that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(std::size_t);

static std::size_t payload_extent(sequence_index_t const& index, std::size_t entries) {
    std::size_t extent = 0;
    for (std::size_t entry = 0; entry != entries; ++entry)
        extent += record_overhead_k + index.entries[entry].payload.size();
    return extent;
}

static std::size_t parameter_bytes(sequence_index_t const& index) { return index.options.similarities.size(); }

static std::optional<std::size_t> serialized_bytes(sequence_index_t const& index, std::size_t entries) {
    return blob_bytes(sizeof(sequence_checkpoint_prefix_t) + parameter_bytes(index), payload_extent(index, entries), 1);
}

static void write_into(sequence_index_t const& index, std::size_t entries, util::bytes_span_t target) {
    sequence_checkpoint_prefix_t prefix {};
    prefix.connectivity = index.options.connectivity;
    prefix.sketch_bytes = index.options.sketch_bytes;
    prefix.entries = entries;
    prefix.payload_bytes = payload_extent(index, entries);
    prefix.metric = index.options.metric;
    prefix.classes = index.options.classes;
    prefix.costed = !index.options.similarities.empty();
    prefix.gap = index.options.gap;
    write_pod(target, prefix);
    util::bytes_span_t cursor = target.subspan(sizeof(prefix));

    if (!index.options.similarities.empty()) {
        [[maybe_unused]] util::status_t const copied = util::copy_bytes(
            cursor.first(index.options.similarities.size()),
            util::bytes_of(index.options.similarities.data(), index.options.similarities.size()));
        cursor = cursor.subspan(index.options.similarities.size());
    }
    for (std::size_t position = 0; position != entries; ++position) {
        sequence_entry_t const& entry = index.entries[position];
        std::size_t const length = entry.payload.size();
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, length);
        cursor = cursor.subspan(sizeof(length));
        [[maybe_unused]] util::status_t const copied = util::copy_bytes(cursor.first(entry.payload.size()),
                                                                        entry.payload);
        cursor = cursor.subspan(entry.payload.size());
    }
}

/** Reads the prefix and checks the blob is long enough for what it claims. */
static bool parse_prefix(sequence_checkpoint_prefix_t& prefix, util::bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(sequence_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<sequence_checkpoint_prefix_t>(blob);
    std::size_t const table = prefix.costed ? alphabet_k * alphabet_k : 0;
    return fits_blob(blob, sizeof(sequence_checkpoint_prefix_t) + table, prefix.payload_bytes, 1, outcome);
}

/** Writes what a checkpoint froze into @p metadata, answering whether its metric is one we know. */
static bool describe(usearch_sequence_metadata_t& metadata, sequence_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    metadata = {prefix.metric, prefix.connectivity, prefix.classes, prefix.sketch_bytes};
    return true;
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we know. The bound
 *  on entries per key is the loader's to choose, since no checkpoint records it. */
static bool adopt(sequence_index_options_t& options, sequence_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.metric = prefix.metric;
    options.connectivity = prefix.connectivity;
    options.sketch_bytes = prefix.sketch_bytes;
    return true;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
static bool read_from(sequence_index_t& index, util::bytes_view_t blob, usearch_outcome_t* outcome) {
    sequence_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return false;
    std::size_t const entries = prefix.entries;
    index.reset();
    if (!index.accommodate(entries)) {
        propagate_error(usearch_status_out_of_memory_k, outcome, "The checkpoint's entries could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(usearch_status_unknown_kind_k, outcome, "The checkpoint names a metric this build lacks");
        return false;
    }
    index.options.gap = prefix.gap;

    util::bytes_view_t cursor = blob.subspan(sizeof(sequence_checkpoint_prefix_t));
    index.options.similarities.clear();
    if (prefix.costed) {
        index.options.similarities.resize(alphabet_k * alphabet_k);
        [[maybe_unused]] util::status_t const copied = util::copy_bytes(
            util::bytes_of(index.options.similarities.data(), index.options.similarities.size()),
            cursor.first(index.options.similarities.size()));
        cursor = cursor.subspan(index.options.similarities.size());
    }
    if (!fold_costs(index.options, outcome)) return index.reset(), false;

    sequence_payload_t payload;
    for (std::size_t position = 0; position != entries; ++position) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        usearch_key_t key;
        std::size_t length = 0;
        key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        length = read_pod<std::size_t>(cursor);
        cursor = cursor.subspan(sizeof(length));
        if (cursor.size() < length) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        payload.assign(cursor.begin(), cursor.begin() + static_cast<std::ptrdiff_t>(length));
        cursor = cursor.subspan(length);
        index.append(key, payload);
    }
    return true;
}

/** Refuses sketch bytes off a multiple of four, and any for a DNA overlap, which has no sketch. */
static bool reject_sketch_bytes(usearch_sequence_metric_t metric, std::size_t sketch_bytes,
                                usearch_outcome_t* outcome) {
    return reject_if(sketch_bytes % 4, usearch_status_config_mismatch_k, "Sketch bytes come in fours", outcome) ||
           reject_if(sketch_bytes && metric == usearch_sequence_metric_dna_overlap_k, usearch_status_config_mismatch_k,
                     "A DNA overlap keeps no sketch", outcome);
}

/** Refuses the costs no fold reads, so a binding pairing the wrong fields learns at once. */
static bool reject_costs(usearch_sequence_metric_t metric, usearch_sequence_costs_t const* costs,
                         usearch_outcome_t* outcome) {
    if (!costs) return false;
    if (reject_if(!costs->similarities, usearch_status_invalid_argument_k, "Costs need similarities", outcome))
        return true;
    return reject_if(metric == usearch_sequence_metric_utf8_containment_k, usearch_status_config_mismatch_k,
                     "A containment takes no costs", outcome);
}

/** Writes @p shard's whole image into @p blob; false once @p outcome records why not. */
static bool serialize_shard(sequence_index_t const& shard, std::vector<std::byte>& blob, usearch_outcome_t* outcome) {
    std::size_t const entries = shard.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(shard, entries);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return false;
    blob.resize(*length);
    write_into(shard, entries, blob);
    return true;
}

/** A new handle holding the checkpoint in @p blob, or null once the outcome records why not. */
template <typename index_type_>
static index_type_* open_checkpoint(util::bytes_view_t blob, usearch_outcome_t* outcome) {
    index_type_* const index = new (std::nothrow) index_type_();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return nullptr;
    if (read_from(*index, blob, outcome)) return index;
    delete index;
    return nullptr;
}

#pragma endregion Serialization

} // namespace stub
} // namespace usearch
} // namespace unum

struct usearch_sequence_index_handle_t : sequence_index_t {};
struct usearch_sequence_cursor_handle_t : sequence_cursor_t {};
struct usearch_sequence_refine_plan_handle_t : long_verb_plan<usearch_sequence_index_handle_t> {};
struct usearch_sequence_reshard_plan_handle_t : long_verb_plan<usearch_sequence_index_handle_t> {};

extern "C" {

char const* usearch_sequence_metric_name(usearch_sequence_metric_t metric) {
    switch (metric) {
    case usearch_sequence_metric_protein_alignment_k: return "protein-alignment";
    case usearch_sequence_metric_dna_overlap_k: return "dna-overlap";
    case usearch_sequence_metric_utf8_containment_k: return "utf8-containment";
    case usearch_sequence_metric_unknown_k: return "unknown";
    default: return "unrecognized";
    }
}

usearch_status_t usearch_sequence_metric_from_name(usearch_bytes_view_t name, usearch_sequence_metric_t* metric) {
    if (named(name, "unknown")) return *metric = usearch_sequence_metric_unknown_k, usearch_status_success_k;
    for (usearch_sequence_metric_t const known :
         {usearch_sequence_metric_protein_alignment_k, usearch_sequence_metric_dna_overlap_k,
          usearch_sequence_metric_utf8_containment_k})
        if (named(name, usearch_sequence_metric_name(known))) return *metric = known, usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->ram.cpu_class_ceiling, outcome)) return;
    usearch_sequence_metric_t const metric = arguments->metric;
    usearch_sequence_costs_t const* const costs = arguments->costs;
    if (reject_if(!known_metric(metric), usearch_status_unknown_kind_k, "The stand-in does not implement that metric",
                  outcome))
        return;
    if (reject_costs(metric, costs, outcome) || reject_sketch_bytes(metric, arguments->sketch_bytes, outcome)) return;
    usearch_sequence_index_handle_t* const index = new (std::nothrow) usearch_sequence_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options.metric = metric;
    index->options.connectivity = arguments->connectivity ? arguments->connectivity : util::connectivity_16_k;
    index->options.sketch_bytes = arguments->sketch_bytes;
    index->options.values_per_key = arguments->values_per_key;
    // Copied rather than viewed, so a binding whose tables die after this call still scores.
    std::vector<util::i8_t>& similarities = index->options.similarities;
    if (costs) {
        similarities.assign(costs->similarities, costs->similarities + alphabet_k * alphabet_k);
        index->options.gap = costs->gap;
    }
    else if (metric != usearch_sequence_metric_utf8_containment_k) {
        similarities.resize(alphabet_k * alphabet_k);
        us::sequence::byte_similarities_span_t const own {similarities.data(), alphabet_k * alphabet_k};
        bool const dna = metric == usearch_sequence_metric_dna_overlap_k;
        if (dna) us::sequence::dna_similarities(own);
        else us::sequence::protein_similarities(own);
        index->options.gap = dna ? us::sequence::dna_gap_k : us::sequence::protein_gap_k;
    }
    if (!fold_costs(index->options, outcome)) return delete index;
    index->shard_capacity = arguments->ram.shard_capacity;
    *arguments->index = index;
}

void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->vram.gpu_class_ceiling, outcome)) return;
    propagate_error(usearch_status_device_k, outcome, "This build has no GPU backends");
}

void usearch_sequence_load_vram(usearch_sequence_load_vram_t* arguments) {
    if (through_executor<usearch_sequence_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->vram.gpu_class_ceiling, outcome)) return;
    propagate_error(usearch_status_device_k, outcome, "This build has no GPU backends");
}

void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    if (through_executor<usearch_sequence_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->ram.cpu_class_ceiling, outcome)) return;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_sequence_index_handle_t* const index = open_parts<usearch_sequence_index_handle_t>(
        blob, arguments->path,
        [&](util::bytes_view_t part) { return open_checkpoint<usearch_sequence_index_handle_t>(part, outcome); },
        outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, 1, 1);
    *arguments->index = index;
}

void usearch_sequence_load_ram_buffer(usearch_sequence_load_ram_buffer_t* arguments) {
    load_ram_buffer<usearch_sequence_load_ram_buffer>(
        arguments, [&](util::bytes_view_t blob, usearch_outcome_t* outcome) {
            return open_checkpoint<usearch_sequence_index_handle_t>(blob, outcome);
        });
}

void usearch_sequence_load_vram_buffer(usearch_sequence_load_vram_buffer_t* arguments) {
    if (through_executor<usearch_sequence_load_vram_buffer>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->vram.gpu_class_ceiling, outcome)) return;
    propagate_error(usearch_status_device_k, outcome, "This build has no GPU backends");
}

void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->ram.cpu_class_ceiling, outcome)) return;
    util::mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Entries are still copied out of the mapping, so only the view's refusals are mirrored, not
    // its zero cost.
    usearch_sequence_index_handle_t* const index = open_parts<usearch_sequence_index_handle_t>(
        served.view(), arguments->path,
        [&](util::bytes_view_t part) { return open_checkpoint<usearch_sequence_index_handle_t>(part, outcome); },
        outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments) {
    if (through_executor<usearch_sequence_view_nvme>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_ceiling(arguments->nvme.cpu_class_ceiling, outcome)) return;
    util::mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Serves the drives' checkpoint as the RAM view does, mirroring only the residency's refusals.
    usearch_sequence_index_handle_t* const index = open_parts<usearch_sequence_index_handle_t>(
        served.view(), arguments->path,
        [&](util::bytes_view_t part) { return open_checkpoint<usearch_sequence_index_handle_t>(part, outcome); },
        outcome);
    if (!index) return;
    index->served = std::move(served);
    index->residency = usearch_residency_nvme_k;
    *arguments->index = index;
}

void usearch_sequence_free(usearch_sequence_index_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments) {
    if (through_executor<usearch_sequence_reserve>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    reject_if(!index.accommodate(arguments->entries), usearch_status_out_of_memory_k,
              "The capacity could not be reserved", outcome);
}

void usearch_sequence_clear(usearch_sequence_clear_t* arguments) {
    if (through_executor<usearch_sequence_clear>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    index.reset();
}

void usearch_sequence_describe(usearch_sequence_describe_t* arguments) {
    if (through_executor<usearch_sequence_describe>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->description = {};
    usearch_sequence_index_handle_t& index = *arguments->index;
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    *arguments->description = {
        {index.options.metric, index.options.connectivity, index.options.classes, index.options.sketch_bytes},
        index.entries.size(),
        index.entries.capacity(),
        payload_extent(index, index.entries.size()) + index.options.similarities.size(),
        usearch_cpu_class_serial_k,
        usearch_gpu_class_unknown_k,
        index.shards(),
        usearch_load_path_none_k,
    };
}

void usearch_sequence_insert(usearch_sequence_insert_t* arguments) {
    if (through_executor<usearch_sequence_insert>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_read_only(index, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;
    if (reject_tape(arguments->sequences, count, outcome)) return;
    if (reject_bytes(arguments->sequences, outcome)) return;

    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    sequence_payload_t payload;
    std::size_t published = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        util::bytes_view_t const sequence = viewed(arguments->sequences, offset);
        payload.assign(sequence.begin(), sequence.end());
        index.append(key_at(arguments->keys, offset), payload);
        ++published;
        if (!keep_going(arguments->progress, published, count)) break;
    }
    // This modality's batch is all-or-nothing, a cancellation included, so a stopped tape unwinds
    // to the watermark rather than leaving a prefix published.
    if (published != count) {
        index.unwind(watermark);
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the batch");
    }
    if (stats) {
        stats->visited_members = published;
        stats->computed_distances = published * index.entries.size();
        stats->restarts = 1;
    }
}

/** The body of @c usearch_sequence_refine, which @ref through_plan_executor runs. */
static void refine_body(usearch_sequence_refine_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    refine_and_report<usearch_sequence_refine_plan_handle_t>(*arguments, *arguments->index, outcome);
}

void usearch_sequence_refine(usearch_sequence_refine_t* arguments) { through_plan_executor<refine_body>(arguments); }

void usearch_sequence_refine_plan_finish(usearch_sequence_refine_plan_finish_t* arguments) {
    through_plan_executor<finish_plan<usearch_sequence_refine_plan_handle_t, usearch_sequence_refine_plan_finish_t>>(
        arguments);
}

void usearch_sequence_health(usearch_sequence_health_t* arguments) {
    if (through_executor<usearch_sequence_health>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->reachability = {};
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    // An exact scan reaches every entry it holds in one level.
    std::size_t const entries = index.entries.size();
    *arguments->reachability = {entries, entries, entries ? 1u : 0u};
}

void usearch_sequence_search(usearch_sequence_search_t* arguments) {
    if (through_executor<usearch_sequence_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->matches = {};
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_tape_view_t const& queries = arguments->queries;
    if (reject_tape(queries, count, outcome)) return;
    if (reject_bytes(queries, outcome)) return;
    if (reject_traversal(arguments->traversal, outcome)) return;
    usearch_matches_view_t const* const hints = arguments->hints;
    if (reject_if(hints && count && (!hints->counts || (hints->columns && !hints->keys)),
                  usearch_status_invalid_argument_k, "The hints must name keys and counts", outcome))
        return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                      {queries.data, offsets_address(queries)}, outcome);
    matches_outputs_t outputs {};
    if (!arena || !carve_ranked(*arena, count, arguments->wanted, outputs, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_tape(index, queries, count, outputs, arguments->expansion_factor,
                                             arguments->stats, arguments->predicate, arguments->progress);
    if (reject_cancelled(answered, count, outcome)) return;
    *arguments->matches = outputs.view();
}

void usearch_sequence_count(usearch_sequence_count_t* arguments) {
    if (through_executor<usearch_sequence_count>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    auto const range = index.by_key.equal_range(uuid_of(arguments->key));
    std::size_t entries = 0, total_bytes = 0, largest_bytes = 0;
    for (auto position = range.first; position != range.second; ++position, ++entries) {
        std::size_t const bytes = index.entries[position->second].payload.size();
        total_bytes += bytes;
        largest_bytes = util::max_of(largest_bytes, bytes);
    }
    *arguments->key_stats = {entries, total_bytes, largest_bytes};
}

void usearch_sequence_get(usearch_sequence_get_t* arguments) {
    if (through_executor<usearch_sequence_get>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->sequences = {};
    if (arguments->written) *arguments->written = 0;
    usearch_sequence_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const entries = nodes_of(index, arguments->key);
    if (entries.empty()) return;
    if (!copy_entries(index, entries, *arena, *arguments->sequences, outcome)) return;
    if (arguments->written) *arguments->written = entries.size();
}

/** The body of @c usearch_sequence_reshard, which @ref through_plan_executor runs. */
static void reshard_body(usearch_sequence_reshard_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    sequence_search_scratch_t scratch;
    auto const score = [&](util::bytes_view_t first, util::bytes_view_t second) {
        return distance(*arguments->sources[0], first, second, scratch);
    };
    auto const recut = [](usearch_sequence_index_handle_t&, usearch_outcome_t* refusal) {
        return !reject_if(true, usearch_status_unsupported_k, "A handle of one shard has nothing to recut", refusal);
    };
    reshard_and_report<usearch_sequence_reshard_plan_handle_t>(*arguments, score, same_kind, recut, outcome);
}

void usearch_sequence_reshard(usearch_sequence_reshard_t* arguments) { through_plan_executor<reshard_body>(arguments); }

void usearch_sequence_reshard_plan_finish(usearch_sequence_reshard_plan_finish_t* arguments) {
    through_plan_executor<finish_plan<usearch_sequence_reshard_plan_handle_t, usearch_sequence_reshard_plan_finish_t>>(
        arguments);
}

void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (reject_if(arguments->starts_count && !arguments->starts, usearch_status_invalid_argument_k,
                  "The starts must name keys", outcome))
        return;
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_sequence_cursor_handle_t* const cursor = new (std::nothrow) usearch_sequence_cursor_handle_t {};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    open_walk(*cursor, index, arguments->starts, arguments->starts_count);
    *arguments->cursor = cursor;
}

void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments) {
    if (through_executor<usearch_sequence_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->written = 0;
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    if (reject_if(!arguments->cursor, usearch_status_invalid_argument_k, "The cursor must not be null", outcome))
        return;
    usearch_sequence_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_sequence_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    if (reject_if(serial_atomic_ref<util::u32_t>(cursor.pulling).fetch_test_and_set_bit(0, std::memory_order_acquire),
                  usearch_status_contended_k, "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    sequence_search_scratch_t scratch;
    auto const score = [&](util::bytes_view_t first, util::bytes_view_t second) {
        return distance(index, first, second, scratch);
    };
    pull_walk(*arguments, cursor, score, *arena, outcome);
    serial_atomic_ref<util::u32_t>(cursor.pulling).clear_bits(1, std::memory_order_release);
}

void usearch_sequence_cursor_free(usearch_sequence_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_sequence_sample(usearch_sequence_sample_t* arguments) {
    if (through_executor<usearch_sequence_sample>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_sequence_index_handle_t& index = *arguments->index;
    sequence_search_scratch_t scratch;
    auto const score = [&](util::bytes_view_t first, util::bytes_view_t second) {
        return distance(index, first, second, scratch);
    };
    draw_sample(*arguments, index, score, no_fixed_coordinates_t {}, outcome);
}

void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    if (through_executor<usearch_sequence_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_if(index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view already is its checkpoint", outcome))
        return;
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    save_parts(index, arguments->path, arguments->progress, serialize_shard, outcome);
}

void usearch_sequence_save_buffer(usearch_sequence_save_buffer_t* arguments) {
    save_buffer<usearch_sequence_save_buffer>(arguments, serialize_shard);
}

void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments) {
    if (through_executor<usearch_sequence_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome) || !read_first_part(blob, arguments->path, outcome)) return;
    sequence_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    reject_if(!describe(*arguments->metadata, prefix), usearch_status_unknown_kind_k,
              "The checkpoint names a metric this build lacks", outcome);
}

void usearch_sequence_metadata_read_buffer(usearch_sequence_metadata_read_buffer_t* arguments) {
    metadata_read_buffer<usearch_sequence_metadata_read_buffer>(arguments, parse_prefix, describe);
}

} // extern "C"
