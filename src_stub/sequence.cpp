/**
 *  @file src_stub/sequence.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_sequence_* C ABI, scoring every stored sequence
 *      against every query so a binding can be exercised without the engine.
 *
 *  Variable-length payloads cannot ride a strided run, so this modality's batches are tapes - the
 *  bytes end to end around Apache Arrow offsets - the shape most worth testing here, because a
 *  binding that handles strides correctly may still mishandle an offset off by one. The weights
 *  are the other half: a Levenshtein kind refuses any and an alignment kind refuses to go without
 *  them, so a binding pairing the wrong two learns immediately.
 *
 *  @section sequence_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  The alignment kernels are textbook dynamic programming with no band, no vectorization, and
 *  quadratic cost, so a corpus of long sequences is slow rather than wrong. Weighted scores are
 *  similarities and reported as their negation, which ranks correctly but is not the engine's
 *  scale. Refinement keeps no edges to re-select, so its @c changed is a constant zero rather than
 *  a settling signal, and a caller must not read it as one. A checkpoint carries its own weights,
 *  so a load or view checks only that weights are named where the kind needs them, and a view
 *  copies its checkpoint rather than mapping it. Traversal statistics are plausible fiction:
 *  shaped correctly, invented numerically. A checkpoint is a binary blob of this file's own memory
 *  image, readable by nothing else.
 *
 *  Keeping no edges, the stand-in links every one of its entries to its exact nearest at the frozen
 *  connectivity, the lists a perfect graph would hold where the engine's are the ones it stored. A
 *  cursor walks those lists breadth-first from its start keys, a split grows its parts over them,
 *  each receiver keeping the source's storage order, and a sample links each draw to its exact
 *  nearest earlier draws.
 */

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <limits>    // `std::numeric_limits`
#include <memory>    // `std::unique_ptr` - a finished refinement's handle
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <utility>   // `std::exchange` - a landed donor's slot
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/sequence_index.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Storage

/** The alphabet of a byte-level substitution matrix, one row and column per byte. */
constexpr std::size_t alphabet_k = 256;

/** What a handle froze at construction, or adopted from a checkpoint, with the scoring parameters
 *  only an alignment kind carries. */
struct sequence_index_options_t {

    /** What entries are compared by. */
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity = 0;

    /** Upper bound on entries sharing one key. */
    std::size_t values_per_key = 0;

    /** Row-major @c alphabet_k × @c alphabet_k scores; empty under a uniform metric. */
    std::vector<us::utilities::i8_t> substitution;

    /** Cost of one unaligned symbol. */
    us::utilities::i8_t gap = 0;

    /** Window widths in bytes the overlap kind compares at; empty under every other kind. */
    std::vector<std::size_t> window_widths;
};

using sequence_payload_t = std::vector<std::byte>;
using sequence_entry_t = entry<sequence_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<sequence_entry_t>, "reallocation must move rows, never copy them");

using sequence_index_t = handle<sequence_payload_t, sequence_index_options_t>;

using sequence_cursor_t = graph_cursor<sequence_index_t>;

/** Whether a receiver frozen at @p receiver takes a part of an index frozen at @p index: the same
 *  kind, weights, window widths and connectivity. */
static bool same_kind(sequence_index_options_t const& index, sequence_index_options_t const& receiver) {
    return index.metric == receiver.metric && index.connectivity == receiver.connectivity &&
           index.substitution == receiver.substitution && index.gap == receiver.gap &&
           index.window_widths == receiver.window_widths;
}

/** Whether the kind scores without scoring parameters, and so refuses any weights. */
static bool levenshtein_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_levenshtein_bytes_k ||
           metric == usearch_sequence_metric_levenshtein_utf8_k;
}

/** Whether the kind reads a substitution matrix and gap costs. */
static bool alignment_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_needleman_wunsch_k;
}

/** Whether the kind is one this build implements at all. */
static bool known_metric(usearch_sequence_metric_t metric) {
    return levenshtein_metric(metric) || alignment_metric(metric) || metric == usearch_sequence_metric_overlap_k;
}

#pragma endregion Storage

#pragma region Scoring

/** Decodes UTF-8 into code points, treating a malformed byte as its own code point. */
static void decode_utf8(bytes_view_t bytes, std::vector<us::utilities::u32_t>& points) {
    points.clear();
    for (std::size_t position = 0; position != bytes.size();) {
        us::utilities::u8_t const leading = std::to_integer<us::utilities::u8_t>(bytes[position]);
        std::size_t width = 1;
        us::utilities::u32_t point = leading;
        if ((leading & 0xE0) == 0xC0) width = 2, point = leading & 0x1F;
        else if ((leading & 0xF0) == 0xE0) width = 3, point = leading & 0x0F;
        else if ((leading & 0xF8) == 0xF0) width = 4, point = leading & 0x07;
        if (position + width > bytes.size()) width = 1, point = leading;
        for (std::size_t continuation = 1; continuation != width; ++continuation) {
            us::utilities::u8_t const following = std::to_integer<us::utilities::u8_t>(bytes[position + continuation]);
            if ((following & 0xC0) != 0x80) {
                width = 1, point = leading;
                break;
            }
            point = (point << 6) | (following & 0x3F);
        }
        points.push_back(point);
        position += width;
    }
}

/** Levenshtein distance with unit substitution, insertion, and deletion costs. */
template <typename unit_>
static double levenshtein(std::span<unit_ const> first, std::span<unit_ const> second, std::vector<std::size_t>& row) {
    row.resize(second.size() + 1);
    for (std::size_t column = 0; column != row.size(); ++column) row[column] = column;
    for (std::size_t above = 0; above != first.size(); ++above) {
        std::size_t diagonal = row[0];
        row[0] = above + 1;
        for (std::size_t column = 0; column != second.size(); ++column) {
            std::size_t const substitution = diagonal + (first[above] == second[column] ? 0 : 1);
            diagonal = row[column + 1];
            row[column + 1] = std::min(substitution, std::min(row[column], row[column + 1]) + 1);
        }
    }
    return static_cast<double>(row[second.size()]);
}

/** Needleman-Wunsch with a linear gap: the least cost turning @p first into @p second, a
 *  substitution costing the mean of the two symbols' self-scores less their shared score and every
 *  unaligned symbol @c gap, as the header defines the kind. */
static double global_alignment(bytes_view_t first, bytes_view_t second, sequence_index_options_t const& parameters,
                               std::vector<double>& row) {
    auto const score = [&](std::byte a, std::byte b) {
        return static_cast<double>(
            parameters.substitution[std::to_integer<std::size_t>(a) * alphabet_k + std::to_integer<std::size_t>(b)]);
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
            row[column + 1] = std::min(aligned, std::min(row[column], row[column + 1]) + gap);
        }
    }
    return row[second.size()];
}

/** Reusable buffers, so scoring a whole corpus allocates once rather than per pair. */
struct sequence_search_scratch_t {
    std::vector<us::utilities::u32_t> query_points;
    std::vector<us::utilities::u32_t> entry_points;
    std::vector<std::size_t> unsigned_row;
    std::vector<double> cost_row;
    std::vector<std::string_view> query_windows;
};

/** Windows of @p width in a sequence of @p length bytes. */
static std::size_t windows_in(std::size_t length, std::size_t width) { return length < width ? 0 : length - width + 1; }

/** The overlap kind: the stored windows the query lacks, over whichever side has more windows. */
static double window_overlap(bytes_view_t query, bytes_view_t stored, std::span<std::size_t const> widths,
                             std::vector<std::string_view>& query_windows) {
    std::string_view const query_text = chars_of(query), stored_text = chars_of(stored);
    std::size_t query_count = 0, stored_count = 0, hits = 0;
    for (std::size_t const width : widths) {
        query_windows.clear();
        for (std::size_t start = 0; start != windows_in(query_text.size(), width); ++start)
            query_windows.push_back(query_text.substr(start, width));
        std::sort(query_windows.begin(), query_windows.end());
        for (std::size_t start = 0; start != windows_in(stored_text.size(), width); ++start)
            hits += std::binary_search(query_windows.begin(), query_windows.end(), stored_text.substr(start, width));
        query_count += query_windows.size();
        stored_count += windows_in(stored_text.size(), width);
    }
    std::size_t const longer = std::max(query_count, stored_count);
    return longer ? static_cast<double>(longer - hits) / static_cast<double>(longer) : 1.0;
}

/** Scores one pair under the frozen kind. The constructor and @c adopt refuse a kind
 *  this build does not know, so every enumerator reaching here is one of the four. */
static double distance(sequence_index_t const& index, bytes_view_t query, bytes_view_t stored,
                       sequence_search_scratch_t& scratch) {
    sequence_index_options_t const& parameters = index.options;
    switch (parameters.metric) {
    case usearch_sequence_metric_levenshtein_utf8_k:
        decode_utf8(query, scratch.query_points);
        decode_utf8(stored, scratch.entry_points);
        return levenshtein<us::utilities::u32_t>(scratch.query_points, scratch.entry_points, scratch.unsigned_row);
    case usearch_sequence_metric_needleman_wunsch_k:
        return global_alignment(query, stored, parameters, scratch.cost_row);
    case usearch_sequence_metric_overlap_k:
        return window_overlap(query, stored, parameters.window_widths, scratch.query_windows);
    default: return levenshtein<std::byte>(query, stored, scratch.unsigned_row);
    }
}

/** Refuses a tape of anything but bytes, the one element a sequence is made of. */
static bool reject_bytes(usearch_tape_view_t tape, usearch_outcome_t* outcome) {
    return reject_if(tape.scalar != usearch_scalar_u8_k, usearch_status_unknown_kind_k, "A sequence tape holds u8",
                     outcome);
}

/** Entry @p entry of a tape, as the bytes its offsets bound. */
static bytes_view_t viewed(usearch_tape_view_t tape, std::size_t entry) {
    std::size_t const begin = tape.offsets[entry], end = tape.offsets[entry + 1];
    return bytes_of(static_cast<std::uint8_t const*>(tape.data) + begin, end - begin);
}

/** The output twin of @c viewed: lands the entries at @p rows into @p arena as one tape, writing
 *  @p tape only once every byte landed, and answers false once @p outcome records why not. */
static bool land_entries(sequence_index_t const& index, std::span<std::size_t const> rows,
                         usearch_arena_handle_t& arena, usearch_tape_view_t& tape, usearch_outcome_t* outcome) {
    std::size_t total_bytes = 0;
    for (std::size_t row : rows) total_bytes += index.entries[row].payload.size();
    std::size_t* const offsets = carve<std::size_t>(arena, rows.size() + 1, outcome);
    std::byte* const bytes = offsets ? carve<std::byte>(arena, total_bytes, outcome) : nullptr;
    if (!bytes) return false;
    offsets[0] = 0;
    for (std::size_t found = 0; found != rows.size(); ++found) {
        sequence_payload_t const& payload = index.entries[rows[found]].payload;
        std::copy(payload.begin(), payload.end(), bytes + offsets[found]);
        offsets[found + 1] = offsets[found] + payload.size();
    }
    tape = {bytes, offsets, usearch_scalar_u8_k};
    return true;
}

/**
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
static std::size_t search_tape(sequence_index_t& index, usearch_tape_view_t queries, std::size_t count,
                               ranked_landing_t const& landing, std::size_t expansion_factor,
                               usearch_traversal_stats_t* stats, usearch_predicate_t predicate,
                               usearch_progress_t progress) {
    std::size_t const rows = index.entries.size();
    if (stats && expansion_factor) stats->computed_distances += expansion_factor * count;

    sequence_search_scratch_t scratch;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != count; ++query) {
        bytes_view_t const query_view = viewed(queries, query);
        rank<sequence_payload_t>(
            std::span<sequence_entry_t const>(index.entries).first(rows), predicate,
            [&](sequence_payload_t const& payload) { return distance(index, query_view, payload, scratch); }, ranked);
        landing.write(query, ranked);
        if (stats) {
            stats->visited_members += rows;
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
 *  fixed-width rows, and the prefix carries their summed extent to size the read.
 */
struct sequence_checkpoint_prefix_t {

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many entries follow the prefix. */
    std::size_t entries;

    /** Bytes of entry payload that follow, keys and lengths included. */
    std::size_t payload_bytes;

    /** What entries are compared by; it alone says whether a matrix follows. */
    usearch_sequence_metric_t metric;

    /** Cost of one unaligned symbol. */
    us::utilities::i8_t gap;

    /** How many window widths follow the matrix, each a @c std::size_t. */
    std::size_t window_widths_count;
};

/** The per-entry overhead of the blob: the key, then the byte length that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(std::size_t);

static std::size_t payload_extent(sequence_index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row) extent += record_overhead_k + index.entries[row].payload.size();
    return extent;
}

static std::size_t parameter_bytes(sequence_index_t const& index) {
    return index.options.substitution.size() + index.options.window_widths.size() * sizeof(std::size_t);
}

static std::optional<std::size_t> serialized_bytes(sequence_index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(sequence_checkpoint_prefix_t) + parameter_bytes(index), payload_extent(index, rows), 1);
}

static void write_into(sequence_index_t const& index, std::size_t rows, bytes_span_t target) {
    sequence_checkpoint_prefix_t prefix {};
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.metric = index.options.metric;
    prefix.gap = index.options.gap;
    prefix.window_widths_count = index.options.window_widths.size();
    write_pod(target, prefix);
    bytes_span_t cursor = target.subspan(sizeof(prefix));

    if (!index.options.substitution.empty()) {
        [[maybe_unused]] status_t const copied = copy_bytes(
            cursor.first(index.options.substitution.size()),
            bytes_of(index.options.substitution.data(), index.options.substitution.size()));
        cursor = cursor.subspan(index.options.substitution.size());
    }
    for (std::size_t const width : index.options.window_widths) {
        write_pod(cursor, width);
        cursor = cursor.subspan(sizeof(width));
    }
    for (std::size_t row = 0; row != rows; ++row) {
        sequence_entry_t const& entry = index.entries[row];
        std::size_t const length = entry.payload.size();
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, length);
        cursor = cursor.subspan(sizeof(length));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(entry.payload.size()), entry.payload);
        cursor = cursor.subspan(entry.payload.size());
    }
}

/** Reads the prefix and checks the blob is long enough for what it claims. */
static bool parse_prefix(sequence_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(sequence_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<sequence_checkpoint_prefix_t>(blob);
    std::size_t const table = alignment_metric(prefix.metric) ? alphabet_k * alphabet_k : 0;
    if (reject_if(prefix.window_widths_count > blob.size() / sizeof(std::size_t), usearch_status_malformed_k,
                  "The checkpoint lists more widths than it holds", outcome))
        return false;
    std::size_t const widths = prefix.window_widths_count * sizeof(std::size_t);
    return fits_blob(blob, sizeof(sequence_checkpoint_prefix_t) + table + widths, prefix.payload_bytes, 1, outcome);
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we know. The bound
 *  on entries per key is the loader's to choose, since no checkpoint records it. */
static bool adopt(sequence_index_options_t& options, sequence_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.metric = prefix.metric;
    options.connectivity = prefix.connectivity;
    return true;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
static bool read_from(sequence_index_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    sequence_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return false;
    std::size_t const rows = prefix.entries;
    index.reset();
    if (!index.accommodate(rows)) {
        propagate_error(usearch_status_out_of_memory_k, outcome, "The checkpoint's entries could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(usearch_status_unknown_kind_k, outcome, "The checkpoint names a metric this build lacks");
        return false;
    }
    index.options.gap = prefix.gap;

    bytes_view_t cursor = blob.subspan(sizeof(sequence_checkpoint_prefix_t));
    index.options.substitution.clear();
    if (alignment_metric(prefix.metric)) {
        index.options.substitution.resize(alphabet_k * alphabet_k);
        [[maybe_unused]] status_t const copied = copy_bytes(
            bytes_of(index.options.substitution.data(), index.options.substitution.size()),
            cursor.first(index.options.substitution.size()));
        cursor = cursor.subspan(index.options.substitution.size());
    }
    index.options.window_widths.resize(prefix.window_widths_count);
    for (std::size_t& width : index.options.window_widths) {
        width = read_pod<std::size_t>(cursor);
        cursor = cursor.subspan(sizeof(width));
    }

    sequence_payload_t payload;
    for (std::size_t row = 0; row != rows; ++row) {
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

/** Refuses weights a Levenshtein kind would ignore, and their absence where an alignment kind
 *  needs them, so a binding pairing the wrong two learns at once. */
static bool reject_weights(usearch_sequence_metric_t metric, usearch_sequence_weights_t const* weights,
                           usearch_outcome_t* outcome) {
    if (reject_if(!alignment_metric(metric) && weights, usearch_status_config_mismatch_k,
                  "Only an alignment kind takes weights", outcome))
        return true;
    return reject_if(alignment_metric(metric) && (!weights || !weights->substitution_matrix),
                     usearch_status_config_mismatch_k, "An alignment kind needs a substitution matrix", outcome);
}

/** A new handle holding the checkpoint in @p blob, or null once the outcome records why not. */
template <typename index_type_>
static index_type_* open_checkpoint(bytes_view_t blob, usearch_sequence_weights_t const* weights,
                                    usearch_outcome_t* outcome) {
    index_type_* const index = new (std::nothrow) index_type_();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return nullptr;
    if (read_from(*index, blob, outcome) && !reject_weights(index->options.metric, weights, outcome)) return index;
    delete index;
    return nullptr;
}

#pragma endregion Serialization

} // namespace stub
} // namespace usearch
} // namespace unum

struct usearch_sequence_index_handle_t : sequence_index_t {};
struct usearch_sequence_cursor_handle_t : sequence_cursor_t {};

extern "C" {

void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_sequence_metric_t const metric = arguments->metric;
    usearch_sequence_weights_t const* const weights = arguments->weights;
    if (reject_if(!known_metric(metric), usearch_status_unknown_kind_k, "The stand-in does not implement that metric",
                  outcome))
        return;
    if (reject_weights(metric, weights, outcome)) return;
    std::span<std::size_t const> const widths {arguments->window_widths,
                                               arguments->window_widths ? arguments->window_widths_count : 0};
    if (reject_if((metric == usearch_sequence_metric_overlap_k) == widths.empty(), usearch_status_config_mismatch_k,
                  "The overlap kind needs window widths, and only it takes any", outcome))
        return;
    if (reject_if(std::find(widths.begin(), widths.end(), 0u) != widths.end(), usearch_status_invalid_argument_k,
                  "A window is empty", outcome))
        return;
    usearch_sequence_index_handle_t* const index = new (std::nothrow) usearch_sequence_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options.metric = metric;
    index->options.window_widths.assign(widths.begin(), widths.end());
    index->options.connectivity = arguments->connectivity ? arguments->connectivity : us::utilities::connectivity_16_k;
    index->options.values_per_key = arguments->values_per_key;
    if (weights) {
        // Copied rather than borrowed, so a binding whose matrix dies after this call still scores.
        index->options.substitution.assign(weights->substitution_matrix,
                                           weights->substitution_matrix + alphabet_k * alphabet_k);
        index->options.gap = weights->gap;
    }
    *arguments->index = index;
}

void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_sequence_load_vram(usearch_sequence_load_vram_t* arguments) {
    if (through_executor<usearch_sequence_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    if (through_executor<usearch_sequence_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_sequence_index_handle_t* const index = open_checkpoint<usearch_sequence_index_handle_t>(
        blob, arguments->weights, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, 1, 1);
    *arguments->index = index;
}

void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Rows are still copied out of the mapping, so only the view's refusals are mirrored, not its zero cost.
    usearch_sequence_index_handle_t* const index = open_checkpoint<usearch_sequence_index_handle_t>(
        served.view(), arguments->weights, outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments) {
    if (through_executor<usearch_sequence_view_nvme>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Serves the drives' checkpoint as the RAM view does, mirroring only the residency's refusals.
    usearch_sequence_index_handle_t* const index = open_checkpoint<usearch_sequence_index_handle_t>(
        served.view(), arguments->weights, outcome);
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
        {index.options.metric, index.options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        payload_extent(index, index.entries.size()) + index.options.substitution.size(),
        "serial",
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
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    sequence_payload_t payload;
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        bytes_view_t const sequence = viewed(arguments->sequences, row);
        payload.assign(sequence.begin(), sequence.end());
        index.append(key_at(arguments->keys, row), payload);
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

void usearch_sequence_refine(usearch_sequence_refine_t* arguments) {
    if (through_executor<usearch_sequence_refine>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    bool cancelled = false;
    for (std::size_t row = 0; row != rows && !cancelled; ++row)
        cancelled = !keep_going(arguments->progress, row + 1, rows);
    if (arguments->stats) arguments->stats->visited_members = rows;
    // An exact scan reaches every entry it holds in one level.
    if (arguments->reachability) *arguments->reachability = {rows, rows, rows ? 1u : 0u};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
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
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                      {queries.data, queries.offsets}, outcome);
    ranked_landing_t landing {};
    if (!arena || !carve_ranked(*arena, count, arguments->wanted, landing, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_tape(index, queries, count, landing, arguments->expansion_factor,
                                             arguments->stats, arguments->predicate, arguments->progress);
    if (reject_cancelled(answered, count, outcome)) return;
    *arguments->matches = landing.view();
}

void usearch_sequence_count(usearch_sequence_count_t* arguments) {
    if (through_executor<usearch_sequence_count>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    auto const range = index.by_key.equal_range(keyed(arguments->key));
    std::size_t entries = 0, total_bytes = 0, largest_bytes = 0;
    for (auto position = range.first; position != range.second; ++position, ++entries) {
        std::size_t const bytes = index.entries[position->second].payload.size();
        total_bytes += bytes;
        largest_bytes = std::max(largest_bytes, bytes);
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
    std::vector<std::size_t> const rows = rows_under(index, arguments->key);
    if (rows.empty()) return;
    if (!land_entries(index, rows, *arena, *arguments->sequences, outcome)) return;
    if (arguments->written) *arguments->written = rows.size();
}

void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments) {
    if (through_executor<usearch_sequence_absorb>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_sequence_index_t* const donors = arguments->donors;
    bool accommodated = true;
    std::unique_lock writers(index.mutex);
    // Each donor lands whole and is consumed at once, its slot nulled, so a failure leaves it and
    // every later one live handles the caller still owns and must free.
    for (std::size_t position = 0; accommodated && position != donors_count; ++position) {
        usearch_sequence_index_handle_t& donor = *donors[position];
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (sequence_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        delete std::exchange(donors[position], nullptr);
    }
    if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The donor's entries could not be accommodated",
                  outcome))
        return;
    // An exact scan keeps no edges, so the closing refine re-selects nothing; a cancel stops only
    // it.
    std::size_t const rows = index.entries.size();
    bool cancelled = false;
    for (std::size_t row = 0; row != rows && !cancelled; ++row)
        cancelled = !keep_going(arguments->progress, row + 1, rows);
    if (arguments->stats) arguments->stats->visited_members = rows;
    if (arguments->reachability) *arguments->reachability = {rows, rows, rows ? 1u : 0u};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

/** The stand-in's refinement: the donors it still lands, and how far its sweep got. It closes
 *  nothing, so insertions and saves meanwhile proceed. */
struct usearch_sequence_refinement_handle_t {

    /** The handle the refinement was opened on, only ever compared with the one each call
     *  passes. */
    usearch_sequence_index_handle_t const* opened_on;

    /** The caller's donor slots, each nulled once landed. */
    usearch_sequence_index_t* donors;
    std::size_t count;

    /** Donors landed so far, and entries the sweep visited so far. */
    std::size_t landed;
    std::size_t swept;

    /** The receivers a split lands in, the seed picking its roots, and the parts landed so far. */
    usearch_sequence_index_t const* receivers;
    std::size_t receivers_count;
    us::utilities::u64_t seed;
    std::size_t parted;
};

void usearch_sequence_refinement_init(usearch_sequence_refinement_init_t* arguments) {
    if (through_executor<usearch_sequence_refinement_init>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->refinement = nullptr;
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    if (reject_if(arguments->count && !arguments->donors, usearch_status_invalid_argument_k,
                  "The donors must name handles", outcome))
        return;
    for (std::size_t position = 0; position != arguments->count; ++position) {
        if (reject_if(!arguments->donors[position], usearch_status_invalid_argument_k, "A donor must not be null",
                      outcome) ||
            reject_if(arguments->donors[position] == arguments->index, usearch_status_config_mismatch_k,
                      "An index cannot absorb itself", outcome))
            return;
    }
    if (reject_receivers(*arguments, outcome)) return;
    usearch_sequence_refinement_t const refinement = new (std::nothrow)
        usearch_sequence_refinement_handle_t {&index, arguments->donors,    arguments->count,           0,
                                              0,      arguments->receivers, arguments->receivers_count, arguments->seed,
                                              0};
    if (reject_if(!refinement, usearch_status_out_of_memory_k, "The refinement could not be allocated", outcome))
        return;
    *arguments->refinement = refinement;
}

void usearch_sequence_refinement_next(usearch_sequence_refinement_next_t* arguments) {
    if (through_executor<usearch_sequence_refinement_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->visited) *arguments->visited = 0;
    if (arguments->changed) *arguments->changed = 0;
    usearch_sequence_refinement_handle_t& refinement = *arguments->refinement;
    if (reject_if(arguments->index != refinement.opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    for (; refinement.landed != refinement.count; ++refinement.landed) {
        usearch_sequence_index_handle_t& donor = *refinement.donors[refinement.landed];
        if (reject_if(!index.accommodate(index.entries.size() + donor.entries.size()), usearch_status_out_of_memory_k,
                      "The donor's entries could not be accommodated", outcome))
            return;
        for (sequence_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        delete std::exchange(refinement.donors[refinement.landed], nullptr);
    }
    if (refinement.parted != refinement.receivers_count) {
        sequence_search_scratch_t scratch;
        auto const score = [&](bytes_view_t first, bytes_view_t second) {
            return distance(index, first, second, scratch);
        };
        writers.unlock();
        std::span<usearch_sequence_index_handle_t* const> const receivers {refinement.receivers,
                                                                           refinement.receivers_count};
        if (!split_into(index, receivers, refinement.seed, score, same_kind, outcome)) return;
        refinement.parted = refinement.receivers_count;
        writers.lock();
    }
    // Each call continues the sweep where the last one stopped.
    std::size_t const rows = index.entries.size();
    bool cancelled = false;
    while (refinement.swept < rows && !cancelled)
        cancelled = !keep_going(arguments->progress, ++refinement.swept, rows);
    // The stand-in re-selects nothing, so its sweep visits every entry and changes none.
    if (arguments->visited) *arguments->visited = refinement.swept;
    if (arguments->stats) arguments->stats->visited_members = refinement.swept;
    reject_if(cancelled && refinement.swept != rows, usearch_status_cancelled_k,
              "The caller's handler stopped the sweep", outcome);
}

void usearch_sequence_refinement_finish(usearch_sequence_refinement_finish_t* arguments) {
    if (through_executor<usearch_sequence_refinement_finish>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    std::unique_ptr<usearch_sequence_refinement_handle_t> const refinement {arguments->refinement};
    if (reject_if(arguments->index != refinement->opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    if (arguments->stats) arguments->stats->visited_members = rows;
    if (arguments->reachability) *arguments->reachability = {rows, rows, rows ? 1u : 0u};
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
    usearch_sequence_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_sequence_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    if (reject_if(cursor.pulling.test_and_set(std::memory_order_acquire), usearch_status_contended_k,
                  "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    sequence_search_scratch_t scratch;
    auto const score = [&](bytes_view_t first, bytes_view_t second) { return distance(index, first, second, scratch); };
    pull_walk(*arguments, cursor, score, *arena, outcome);
    cursor.pulling.clear(std::memory_order_release);
}

void usearch_sequence_cursor_free(usearch_sequence_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_sequence_sample(usearch_sequence_sample_t* arguments) {
    if (through_executor<usearch_sequence_sample>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->keys = {};
    *arguments->links = {};
    *arguments->written = 0;
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::size_t const anchors_count = arguments->anchors_count;
    usearch_tape_view_t const& anchors = arguments->anchors;
    if (anchors_count && (reject_tape(anchors, anchors_count, outcome) || reject_bytes(anchors, outcome))) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                      {anchors.data, anchors.offsets}, outcome);
    if (!arena) return;

    std::vector<bytes_view_t> anchor_views(anchors_count);
    for (std::size_t anchor = 0; anchor != anchors_count; ++anchor) anchor_views[anchor] = viewed(anchors, anchor);
    sequence_search_scratch_t scratch;
    auto const score = [&](bytes_view_t first, bytes_view_t second) { return distance(index, first, second, scratch); };
    std::shared_lock readers(index.mutex);
    land_sample(*arguments, std::span<sequence_entry_t const>(index.entries),
                std::span<bytes_view_t const>(anchor_views), score, *arena, outcome);
}

void usearch_sequence_split(usearch_sequence_split_t* arguments) {
    if (through_executor<usearch_sequence_split>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->reachability) *arguments->reachability = {};
    usearch_sequence_index_handle_t& index = *arguments->index;
    sequence_search_scratch_t scratch;
    auto const score = [&](bytes_view_t first, bytes_view_t second) { return distance(index, first, second, scratch); };
    land_split(*arguments, index, score, same_kind, outcome);
}

void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    if (through_executor<usearch_sequence_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_if(index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view already is its checkpoint", outcome))
        return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(arguments->path, blob, outcome)) return;
    keep_going(arguments->progress, 1, 1);
}

void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments) {
    if (through_executor<usearch_sequence_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    sequence_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    sequence_index_options_t options {};
    if (reject_if(!adopt(options, prefix), usearch_status_unknown_kind_k,
                  "The checkpoint names a metric this build lacks", outcome))
        return;
    *arguments->metadata = {options.metric, options.connectivity};
}

} // extern "C"
