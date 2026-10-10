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
 */

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <limits>    // `std::numeric_limits`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
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
    std::vector<us::i8_t> substitution;

    /** Cost of opening a gap, and the flat gap cost of the linear kind. */
    us::i8_t gap_open = 0;

    /** Cost of extending an open gap; the linear kind ignores it. */
    us::i8_t gap_extend = 0;
};

using sequence_payload_t = std::vector<std::byte>;
using sequence_entry_t = entry<sequence_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<sequence_entry_t>, "reallocation must move rows, never copy them");

using sequence_index_t = handle<sequence_payload_t, sequence_index_options_t>;

struct sequence_cursor_t {

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    sequence_index_t const* opened_on;

    /** The next entry to hand out. */
    std::size_t position;

    /** The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag pulling;
};

/** Whether the kind scores without scoring parameters, and so refuses any weights. */
static bool levenshtein_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_levenshtein_bytes_k ||
           metric == usearch_sequence_metric_levenshtein_utf8_k;
}

/** Whether the kind reads a substitution matrix and gap costs. */
static bool alignment_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_needleman_wunsch_k ||
           metric == usearch_sequence_metric_needleman_wunsch_gotoh_k ||
           metric == usearch_sequence_metric_smith_waterman_k ||
           metric == usearch_sequence_metric_smith_waterman_gotoh_k;
}

/** Whether the kind is one this build implements at all. */
static bool known_metric(usearch_sequence_metric_t metric) {
    return levenshtein_metric(metric) || alignment_metric(metric);
}

#pragma endregion Storage

#pragma region Scoring

/** Decodes UTF-8 into code points, treating a malformed byte as its own code point. */
static void decode_utf8(bytes_view_t bytes, std::vector<us::u32_t>& points) {
    points.clear();
    for (std::size_t position = 0; position != bytes.size();) {
        us::u8_t const leading = std::to_integer<us::u8_t>(bytes[position]);
        std::size_t width = 1;
        us::u32_t point = leading;
        if ((leading & 0xE0) == 0xC0) width = 2, point = leading & 0x1F;
        else if ((leading & 0xF0) == 0xE0) width = 3, point = leading & 0x0F;
        else if ((leading & 0xF8) == 0xF0) width = 4, point = leading & 0x07;
        if (position + width > bytes.size()) width = 1, point = leading;
        for (std::size_t continuation = 1; continuation != width; ++continuation) {
            us::u8_t const following = std::to_integer<us::u8_t>(bytes[position + continuation]);
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

/** Global alignment with affine gaps - Needleman-Wunsch-Gotoh - as a negated similarity.
 *  Three recurrences run together: an aligned pair, a gap in the first, a gap in the second.
 *  An @p extend equal to @c gap_open collapses it to the flat-gap Needleman-Wunsch kind. */
static double global_alignment(bytes_view_t first, bytes_view_t second, sequence_index_options_t const& parameters,
                               us::i64_t extend, std::vector<us::i64_t>& scratch) {
    us::i64_t const open = parameters.gap_open;
    us::i64_t const unreachable = std::numeric_limits<us::i64_t>::min() / 4;
    std::size_t const width = second.size() + 1;
    scratch.assign(width * 3, 0);
    std::span<us::i64_t> const rows {scratch};
    std::span<us::i64_t> aligned = rows.subspan(0, width);
    std::span<us::i64_t> gap_in_first = rows.subspan(width, width);
    std::span<us::i64_t> gap_in_second = rows.subspan(width * 2, width);

    for (std::size_t column = 0; column != width; ++column) {
        aligned[column] = column ? -(open + static_cast<us::i64_t>(column - 1) * extend) : 0;
        gap_in_first[column] = column ? aligned[column] : unreachable;
        gap_in_second[column] = unreachable;
    }
    for (std::size_t above = 0; above != first.size(); ++above) {
        us::i64_t diagonal = aligned[0];
        aligned[0] = -(open + static_cast<us::i64_t>(above) * extend);
        gap_in_first[0] = unreachable;
        gap_in_second[0] = aligned[0];
        for (std::size_t column = 0; column != second.size(); ++column) {
            std::size_t const cell = std::to_integer<std::size_t>(first[above]) * alphabet_k +
                                     std::to_integer<std::size_t>(second[column]);
            us::i64_t const carried = diagonal + parameters.substitution[cell];
            diagonal = aligned[column + 1];

            gap_in_first[column + 1] = std::max(aligned[column] - open, gap_in_first[column] - extend);
            gap_in_second[column + 1] = std::max(aligned[column + 1] - open, gap_in_second[column + 1] - extend);
            aligned[column + 1] = std::max(carried, std::max(gap_in_first[column + 1], gap_in_second[column + 1]));
        }
    }
    return -static_cast<double>(aligned[second.size()]);
}

/** Local alignment with affine gaps - Smith-Waterman-Gotoh - as a negated similarity.
 *  The same three recurrences under a zero floor, reporting the best cell anywhere in the
 *  table. An @p extend equal to @c gap_open collapses it to the flat-gap Smith-Waterman kind. */
static double local_alignment(bytes_view_t first, bytes_view_t second, sequence_index_options_t const& parameters,
                              us::i64_t extend, std::vector<us::i64_t>& scratch) {
    us::i64_t const open = parameters.gap_open;
    std::size_t const width = second.size() + 1;
    scratch.assign(width * 3, 0);
    std::span<us::i64_t> const rows {scratch};
    std::span<us::i64_t> aligned = rows.subspan(0, width);
    std::span<us::i64_t> gap_in_first = rows.subspan(width, width);
    std::span<us::i64_t> gap_in_second = rows.subspan(width * 2, width);

    us::i64_t best = 0;
    for (std::size_t above = 0; above != first.size(); ++above) {
        us::i64_t diagonal = 0;
        aligned[0] = 0;
        gap_in_first[0] = 0;
        for (std::size_t column = 0; column != second.size(); ++column) {
            std::size_t const cell = std::to_integer<std::size_t>(first[above]) * alphabet_k +
                                     std::to_integer<std::size_t>(second[column]);
            us::i64_t const carried = diagonal + parameters.substitution[cell];
            diagonal = aligned[column + 1];

            gap_in_first[column + 1] = std::max(aligned[column] - open, gap_in_first[column] - extend);
            gap_in_second[column + 1] = std::max(aligned[column + 1] - open, gap_in_second[column + 1] - extend);
            us::i64_t const cell_best = std::max<us::i64_t>(
                0, std::max(carried, std::max(gap_in_first[column + 1], gap_in_second[column + 1])));
            aligned[column + 1] = cell_best;
            best = std::max(best, cell_best);
        }
    }
    return -static_cast<double>(best);
}

/** Reusable buffers, so scoring a whole corpus allocates once rather than per pair. */
struct sequence_search_scratch_t {
    std::vector<us::u32_t> query_points;
    std::vector<us::u32_t> entry_points;
    std::vector<std::size_t> unsigned_row;
    std::vector<us::i64_t> signed_row;
};

/** Scores one pair under the frozen kind. The constructor and @c adopt refuse a kind
 *  this build does not know, so every enumerator reaching here is one of the six. */
static double distance(sequence_index_t const& index, bytes_view_t query, bytes_view_t stored,
                       sequence_search_scratch_t& scratch) {
    sequence_index_options_t const& parameters = index.options;
    switch (parameters.metric) {
    case usearch_sequence_metric_levenshtein_utf8_k:
        decode_utf8(query, scratch.query_points);
        decode_utf8(stored, scratch.entry_points);
        return levenshtein<us::u32_t>(scratch.query_points, scratch.entry_points, scratch.unsigned_row);
    case usearch_sequence_metric_needleman_wunsch_k:
        return global_alignment(query, stored, parameters, parameters.gap_open, scratch.signed_row);
    case usearch_sequence_metric_needleman_wunsch_gotoh_k:
        return global_alignment(query, stored, parameters, parameters.gap_extend, scratch.signed_row);
    case usearch_sequence_metric_smith_waterman_k:
        return local_alignment(query, stored, parameters, parameters.gap_open, scratch.signed_row);
    case usearch_sequence_metric_smith_waterman_gotoh_k:
        return local_alignment(query, stored, parameters, parameters.gap_extend, scratch.signed_row);
    default: return levenshtein<std::byte>(query, stored, scratch.unsigned_row);
    }
}

/** Entry @p entry of a tape, as the bytes its offsets bound. */
static bytes_view_t viewed(usearch_tape_view_t tape, std::size_t entry) {
    std::size_t const begin = tape.offsets[entry], end = tape.offsets[entry + 1];
    return bytes_of(static_cast<std::uint8_t const*>(tape.data) + begin, end - begin);
}

/**
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
static std::size_t search_tape(sequence_index_t& index, usearch_tape_view_t queries, std::size_t count,
                               usearch_matches_span_t const& matches, std::size_t expansion_factor,
                               usearch_sequence_traversal_stats_t* stats, usearch_predicate_t predicate,
                               void* predicate_state, usearch_progress_callback_t progress, void* progress_state) {
    std::size_t const rows = index.entries.size();
    std::size_t const wanted = matches.wanted;
    if (stats && expansion_factor) stats->computed_distances += expansion_factor * count;

    std::span<std::size_t> const counts {matches.counts, count};
    std::fill(counts.begin(), counts.end(), 0);
    sequence_search_scratch_t scratch;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        bytes_view_t const query_view = viewed(queries, query);
        rank<sequence_payload_t>(
            std::span<sequence_entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
            [&](sequence_payload_t const& payload) { return distance(index, query_view, payload, scratch); }, ranked);
        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(matches.keys, query, matches.keys_stride,
                                                                        wanted);
        std::span<usearch_distance_t> const distances_row = row_at<usearch_distance_t>(matches.scores, query,
                                                                                       matches.scores_stride, wanted);
        std::size_t const found = std::min(wanted, ranked.size());
        for (std::size_t match = 0; match != found; ++match)
            keys_row[match] = ranked[match].key, distances_row[match] = ranked[match].distance;
        counts[query] = found;
        if (stats) {
            stats->visited_members += rows;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(progress, progress_state, query + 1, counts.size())) return query + 1;
    }
    return counts.size();
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

    /** Cost of opening a gap, and the flat gap cost of the linear kinds. */
    us::i8_t gap_open;

    /** Cost of extending an open gap. */
    us::i8_t gap_extend;
};

/** The per-entry overhead of the blob: the key, then the byte length that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(std::size_t);

static std::size_t payload_extent(sequence_index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row) extent += record_overhead_k + index.entries[row].payload.size();
    return extent;
}

static std::size_t substitution_bytes(sequence_index_t const& index) { return index.options.substitution.size(); }

static std::optional<std::size_t> serialized_bytes(sequence_index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(sequence_checkpoint_prefix_t) + substitution_bytes(index), payload_extent(index, rows), 1);
}

static void write_into(sequence_index_t const& index, std::size_t rows, bytes_span_t target) {
    sequence_checkpoint_prefix_t prefix {};
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.metric = index.options.metric;
    prefix.gap_open = index.options.gap_open;
    prefix.gap_extend = index.options.gap_extend;
    write_pod(target, prefix);
    bytes_span_t cursor = target.subspan(sizeof(prefix));

    if (!index.options.substitution.empty()) {
        [[maybe_unused]] status_t const copied = copy_bytes(
            cursor.first(index.options.substitution.size()),
            bytes_of(index.options.substitution.data(), index.options.substitution.size()));
        cursor = cursor.subspan(index.options.substitution.size());
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
    return fits_blob(blob, sizeof(sequence_checkpoint_prefix_t) + table, prefix.payload_bytes, 1, outcome);
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
    index.options.gap_open = prefix.gap_open;
    index.options.gap_extend = prefix.gap_extend;

    bytes_view_t cursor = blob.subspan(sizeof(sequence_checkpoint_prefix_t));
    index.options.substitution.clear();
    if (alignment_metric(prefix.metric)) {
        index.options.substitution.resize(alphabet_k * alphabet_k);
        [[maybe_unused]] status_t const copied = copy_bytes(
            bytes_of(index.options.substitution.data(), index.options.substitution.size()),
            cursor.first(index.options.substitution.size()));
        cursor = cursor.subspan(index.options.substitution.size());
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
    if (reject_if(levenshtein_metric(metric) && weights, usearch_status_config_mismatch_k,
                  "A Levenshtein kind takes no weights", outcome))
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
    usearch_sequence_index_handle_t* const index = new (std::nothrow) usearch_sequence_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options.metric = metric;
    index->options.connectivity = arguments->connectivity ? arguments->connectivity : us::connectivity_16_k;
    index->options.values_per_key = arguments->values_per_key;
    if (weights) {
        // Copied rather than borrowed, so a binding whose matrix dies after this call still scores.
        index->options.substitution.assign(weights->substitution_matrix,
                                           weights->substitution_matrix + alphabet_k * alphabet_k);
        index->options.gap_open = weights->gap_open;
        index->options.gap_extend = weights->gap_extend;
    }
    *arguments->index = index;
}

void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    if (through_executor<usearch_sequence_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_sequence_index_handle_t* const index = open_checkpoint<usearch_sequence_index_handle_t>(
        blob, arguments->weights, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
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
    usearch_sequence_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_read_only(index, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;
    if (reject_tape(arguments->sequences, count, outcome)) return;

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
        if (!keep_going(arguments->progress, arguments->progress_state, published, count)) break;
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
    // Cancelling is not an error here, the header having made a stopped sweep merely less improved.
    for (std::size_t row = 0; row != rows; ++row)
        if (!keep_going(arguments->progress, arguments->progress_state, row + 1, rows)) break;
    if (arguments->stats) arguments->stats->visited_members = rows;
}

void usearch_sequence_search(usearch_sequence_search_t* arguments) {
    if (through_executor<usearch_sequence_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_tape(arguments->queries, count, outcome)) return;
    if (reject_matches(arguments->matches, count, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_tape(
        index, arguments->queries, count, arguments->matches, arguments->expansion_factor, arguments->stats,
        arguments->predicate, arguments->predicate_state, arguments->progress, arguments->progress_state);
    reject_cancelled(answered, count, outcome);
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
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->sequences = nullptr;
    *arguments->offsets = nullptr;
    if (arguments->written) *arguments->written = 0;
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const rows = rows_under(index, arguments->key);
    if (rows.empty()) return;
    std::size_t total_bytes = 0;
    for (std::size_t row : rows) total_bytes += index.entries[row].payload.size();
    // The offsets lead, so they start on a word boundary however long the bytes behind them run.
    std::size_t const offsets_bytes = (rows.size() + 1) * sizeof(std::size_t);
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, nullptr, outcome);
    if (!arena) return;
    std::unique_lock const held {arena->serving, std::adopt_lock};
    std::byte* const landing = arena->memory.allocate(offsets_bytes + total_bytes, default_alignment_k);
    if (!landing) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    std::size_t* const offsets = reinterpret_cast<std::size_t*>(landing);
    std::byte* const tape = landing + offsets_bytes;
    offsets[0] = 0;
    for (std::size_t found = 0; found != rows.size(); ++found) {
        sequence_payload_t const& payload = index.entries[rows[found]].payload;
        std::copy(payload.begin(), payload.end(), tape + offsets[found]);
        offsets[found + 1] = offsets[found] + payload.size();
    }
    *arguments->sequences = reinterpret_cast<std::uint8_t*>(tape);
    *arguments->offsets = offsets;
    if (arguments->written) *arguments->written = rows.size();
}

void usearch_sequence_merge(usearch_sequence_merge_t* arguments) {
    if (through_executor<usearch_sequence_merge>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_sequence_index_t* const donors = arguments->donors;
    usearch_progress_callback_t const progress = arguments->progress;
    void* const progress_state = arguments->progress_state;
    std::size_t merged = 0;
    bool accommodated = true;
    while (accommodated && merged != donors_count) {
        usearch_sequence_index_handle_t& donor = *donors[merged];
        std::unique_lock writers(index.mutex);
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (sequence_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        ++merged;
        if (!keep_going(progress, progress_state, merged, donors_count)) break;
    }
    // Only an absorbed donor is consumed, and its slot is nulled as it goes, so a stopped merge
    // leaves the rest live handles the caller still owns and must free.
    for (std::size_t position = 0; position != merged; ++position) {
        delete donors[position];
        donors[position] = nullptr;
    }
    if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The donor's entries could not be accommodated",
                  outcome))
        return;
    reject_cancelled(merged, donors_count, outcome);
}

void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_sequence_cursor_handle_t* const cursor = new (std::nothrow)
        usearch_sequence_cursor_handle_t {&index, 0, index.entries.size(), {}};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    *arguments->cursor = cursor;
}

void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments) {
    if (through_executor<usearch_sequence_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->written = 0;
    usearch_sequence_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_sequence_index_handle_t& index = *arguments->index;
    usearch_keys_span_t const& keys = arguments->keys;
    if (reject_keys_span(keys, outcome)) return;
    if (reject_if(cursor.pulling.test_and_set(std::memory_order_acquire), usearch_status_contended_k,
                  "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    std::size_t const written = std::min(keys.capacity, cursor.limit - cursor.position);
    for (std::size_t row = 0; row != written; ++row) {
        sequence_entry_t const& entry = index.entries[cursor.position + row];
        row_at<usearch_key_t>(keys.data, row, keys.stride, 1).front() = entry.key;
        // Views borrow the handle's own bytes, which growth relocates the row of but never the
        // payload buffer itself, so they survive concurrent insertion.
        if (arguments->sequences) arguments->sequences[row] = as_bytes(entry.payload);
    }
    cursor.position += written;
    *arguments->written = written;
    cursor.pulling.clear(std::memory_order_release);
}

void usearch_sequence_cursor_free(usearch_sequence_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    if (through_executor<usearch_sequence_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(arguments->path, blob, outcome)) return;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
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
