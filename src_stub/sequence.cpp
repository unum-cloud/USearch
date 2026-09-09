/**
 *  @brief A brute-force stand-in for the @c usearch_sequence_* C ABI, scoring every stored
 *      sequence against every query so a binding can be exercised without the engine.
 *
 *  @file src_stub/sequence.cpp
 *
 *  Variable-length payloads cannot ride a strided matrix, so this modality's batches are tapes
 *  of @c usearch_bytes_view_t views - the shape most worth testing here, because a binding that
 *  handles strides correctly may still mishandle a borrowed pointer and a length. The two
 *  constructors are the other half: a handle built for a uniform metric rejects a weighted one
 *  and the reverse, so a binding calling the wrong one learns immediately.
 *
 *  @section sequence_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  The alignment kernels are textbook dynamic programming with no band, no vectorization, and
 *  quadratic cost, so a corpus of long sequences is slow rather than wrong. Weighted scores are
 *  similarities and reported as their negation, which ranks correctly but is not the engine's
 *  scale. Refinement keeps no edges to re-select, so its return is a constant zero rather than a
 *  settling signal, and a caller must not read it as one. Traversal statistics are plausible
 *  fiction: shaped correctly, invented numerically. A checkpoint is a binary blob of this file's
 *  own memory image, readable by nothing else.
 */

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <limits>    // `std::numeric_limits`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/sequence_index.h"

#include "lib.hpp"

namespace us = unum::usearch;
using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace {

#pragma region Storage

/** @brief The alphabet of a byte-level substitution matrix, one row and column per byte. */
constexpr std::size_t alphabet_k = 256;

/**
 *  @brief What a handle froze at construction: the ABI's options, plus the scoring parameters
 *      only the weighted constructor supplies.
 */
struct sequence_options_t {
    /** @brief The options the caller passed, or a checkpoint later adopted. */
    usearch_sequence_options_t options {};
    /** @brief Row-major @c alphabet_k × @c alphabet_k scores; empty under a uniform metric. */
    std::vector<us::i8_t> substitution;
    /** @brief Cost of opening a gap, and the flat gap cost of the linear kind. */
    us::i8_t gap_open = 0;
    /** @brief Cost of extending an open gap; the linear kind ignores it. */
    us::i8_t gap_extend = 0;
};

using payload_t = std::vector<std::byte>;
using entry_t = entry<payload_t>;

static_assert(std::is_nothrow_move_constructible_v<entry_t>, "reallocation must move rows, never copy them");

using index_t = handle<payload_t, sequence_options_t>;

struct cursor_t {
    /** @brief The index being walked; it must outlive the cursor. */
    index_t* index;
    /** @brief The next entry to hand out. */
    std::size_t position;
    /** @brief The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;
};

/** @brief Whether the kind scores without scoring parameters, which decides the constructor. */
bool levenshtein_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_levenshtein_bytes_k ||
           metric == usearch_sequence_metric_levenshtein_utf8_k;
}

/** @brief Whether the kind reads a substitution matrix and gap costs. */
bool alignment_metric(usearch_sequence_metric_t metric) {
    return metric == usearch_sequence_metric_needleman_wunsch_k ||
           metric == usearch_sequence_metric_needleman_wunsch_gotoh_k ||
           metric == usearch_sequence_metric_smith_waterman_k ||
           metric == usearch_sequence_metric_smith_waterman_gotoh_k;
}

/** @brief Whether the kind is one this build implements at all. */
bool known_metric(usearch_sequence_metric_t metric) { return levenshtein_metric(metric) || alignment_metric(metric); }

#pragma endregion Storage

#pragma region Scoring

/** @brief Decodes UTF-8 into code points, treating a malformed byte as its own code point. */
void decode_utf8(bytes_view_t bytes, std::vector<us::u32_t>& points) {
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

/** @brief Levenshtein distance with unit substitution, insertion, and deletion costs. */
template <typename unit_>
double levenshtein(std::span<unit_ const> first, std::span<unit_ const> second, std::vector<std::size_t>& row) {
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

/**
 *  @brief Global alignment with affine gaps - Needleman-Wunsch-Gotoh - as a negated similarity.
 *      Three recurrences run together: an aligned pair, a gap in the first, a gap in the second.
 *      An @p extend equal to @c gap_open collapses it to the flat-gap Needleman-Wunsch kind.
 */
double global_alignment(bytes_view_t first, bytes_view_t second, sequence_options_t const& parameters, us::i64_t extend,
                        std::vector<us::i64_t>& scratch) {
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

/**
 *  @brief Local alignment with affine gaps - Smith-Waterman-Gotoh - as a negated similarity.
 *      The same three recurrences under a zero floor, reporting the best cell anywhere in the
 *      table. An @p extend equal to @c gap_open collapses it to the flat-gap Smith-Waterman kind.
 */
double local_alignment(bytes_view_t first, bytes_view_t second, sequence_options_t const& parameters, us::i64_t extend,
                       std::vector<us::i64_t>& scratch) {
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

/** @brief Reusable buffers, so scoring a whole corpus allocates once rather than per pair. */
struct scratch_t {
    std::vector<us::u32_t> query_points;
    std::vector<us::u32_t> entry_points;
    std::vector<std::size_t> unsigned_row;
    std::vector<us::i64_t> signed_row;
};

/**
 *  @brief Scores one pair under the frozen kind. Both constructors and @c adopt refuse a kind
 *      this build does not know, so every enumerator reaching here is one of the six.
 */
double distance(index_t const& index, bytes_view_t query, bytes_view_t stored, scratch_t& scratch) {
    sequence_options_t const& parameters = index.options;
    switch (parameters.options.metric) {
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

bytes_view_t viewed(usearch_bytes_view_t sequence) { return bytes_of(sequence.data, sequence.length); }

/**
 *  @brief The shared body of both search forms; the filtered one passes a predicate.
 *  @return Queries answered, short of @p counts.size() exactly when the handler cancelled.
 */
std::size_t search_tape(index_t& index, usearch_bytes_view_t const* queries, std::size_t wanted,
                        usearch_sequence_search_options_t const* options, usearch_sequence_traversal_stats_t* stats,
                        usearch_predicate_t predicate, void* predicate_state, usearch_progress_callback_t progress,
                        void* progress_state, usearch_key_t* keys, std::size_t keys_stride,
                        usearch_distance_t* distances, std::size_t distances_stride, std::span<std::size_t> counts) {
    std::size_t const rows = index.entries.size();
    if (stats && options && options->expansion_factor)
        stats->computed_distances += options->expansion_factor * counts.size();

    std::fill(counts.begin(), counts.end(), 0);
    scratch_t scratch;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        bytes_view_t const query_view = viewed(queries[query]);
        rank<payload_t>(
            std::span<entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
            [&](payload_t const& payload) { return distance(index, query_view, payload, scratch); }, ranked);
        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(keys, query, keys_stride, wanted);
        std::span<usearch_distance_t> const distances_row = row_at<usearch_distance_t>(distances, query,
                                                                                       distances_stride, wanted);
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
struct prefix_t {
    /** @brief The connections-per-node bound the handle was built with. */
    std::size_t connectivity;
    /** @brief How many entries follow the prefix. */
    std::size_t entries;
    /** @brief Bytes of entry payload that follow, keys and lengths included. */
    std::size_t payload_bytes;
    /** @brief What entries are compared by; it alone says whether a matrix follows. */
    usearch_sequence_metric_t metric;
    /** @brief Cost of opening a gap, and the flat gap cost of the linear kinds. */
    us::i8_t gap_open;
    /** @brief Cost of extending an open gap. */
    us::i8_t gap_extend;
};

/** @brief The per-entry overhead of the blob: the key, then the byte length that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(std::size_t);

std::size_t payload_extent(index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row) extent += record_overhead_k + index.entries[row].payload.size();
    return extent;
}

std::size_t substitution_bytes(index_t const& index) { return index.options.substitution.size(); }

std::optional<std::size_t> serialized_bytes(index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(prefix_t) + substitution_bytes(index), payload_extent(index, rows), 1);
}

void write_into(index_t const& index, std::size_t rows, bytes_span_t target) {
    prefix_t prefix {};
    prefix.connectivity = index.options.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.metric = index.options.options.metric;
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
        entry_t const& entry = index.entries[row];
        std::size_t const length = entry.payload.size();
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, length);
        cursor = cursor.subspan(sizeof(length));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(entry.payload.size()), entry.payload);
        cursor = cursor.subspan(entry.payload.size());
    }
}

/** @brief Reads the prefix and checks the blob is long enough for what it claims. */
bool parse_prefix(prefix_t& prefix, bytes_view_t blob, usearch_error_t* error) {
    if (reject_if(blob.size() < sizeof(prefix_t), usearch_error_malformed_input_k, "The checkpoint ends early", error))
        return false;
    prefix = read_pod<prefix_t>(blob);
    std::size_t const table = alignment_metric(prefix.metric) ? alphabet_k * alphabet_k : 0;
    return fits_blob(blob, sizeof(prefix_t) + table, prefix.payload_bytes, 1, error);
}

/** @brief Adopts a checkpoint's configuration, answering whether its metric is one we know. */
bool adopt(usearch_sequence_options_t& options, prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.metric = prefix.metric;
    options.connectivity = prefix.connectivity;
    return true;
}

/** @brief Replaces the contents of @p index; the caller holds both locks exclusively. */
bool read_from(index_t& index, bytes_view_t blob, usearch_error_t* error) {
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return false;
    std::size_t const rows = prefix.entries;
    index.reset();
    if (!index.accommodate(rows)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The checkpoint's entries could not be allocated");
        return false;
    }
    if (!adopt(index.options.options, prefix)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
        return false;
    }
    index.options.gap_open = prefix.gap_open;
    index.options.gap_extend = prefix.gap_extend;

    bytes_view_t cursor = blob.subspan(sizeof(prefix_t));
    index.options.substitution.clear();
    if (alignment_metric(prefix.metric)) {
        index.options.substitution.resize(alphabet_k * alphabet_k);
        [[maybe_unused]] status_t const copied = copy_bytes(
            bytes_of(index.options.substitution.data(), index.options.substitution.size()),
            cursor.first(index.options.substitution.size()));
        cursor = cursor.subspan(index.options.substitution.size());
    }

    payload_t payload;
    for (std::size_t row = 0; row != rows; ++row) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_error_malformed_input_k, error, "The checkpoint ends inside an entry");
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
            propagate_error(usearch_error_malformed_input_k, error, "The checkpoint ends inside an entry");
            return false;
        }
        payload.assign(cursor.begin(), cursor.begin() + static_cast<std::ptrdiff_t>(length));
        cursor = cursor.subspan(length);
        index.append(key, payload);
    }
    return true;
}

#pragma endregion Serialization

} // namespace

extern "C" {

usearch_sequence_index_t usearch_sequence_init_uniform(usearch_sequence_options_t const* options,
                                                       usearch_error_t* error) {
    propagate_success(error);
    if (!options) {
        propagate_error(usearch_error_malformed_input_k, error, "The options are null");
        return nullptr;
    }
    if (!known_metric(options->metric)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The stand-in does not implement that metric");
        return nullptr;
    }
    if (!levenshtein_metric(options->metric)) {
        propagate_error(usearch_error_config_mismatch_k, error, "The uniform constructor takes a Levenshtein kind");
        return nullptr;
    }
    index_t* index = new (std::nothrow) index_t();
    if (!index) {
        propagate_error(usearch_error_out_of_memory_k, error, "The handle could not be allocated");
        return nullptr;
    }
    index->options.options = *options;
    if (!index->options.options.connectivity) index->options.options.connectivity = us::connectivity_16_k;
    return index;
}

usearch_sequence_index_t usearch_sequence_init_weighted(usearch_sequence_options_t const* options,
                                                        us::i8_t const* substitution_matrix, us::i8_t gap_open,
                                                        us::i8_t gap_extend, usearch_error_t* error) {
    propagate_success(error);
    if (!options || !substitution_matrix) {
        propagate_error(usearch_error_malformed_input_k, error, "The options or the substitution matrix are null");
        return nullptr;
    }
    if (!known_metric(options->metric)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The stand-in does not implement that metric");
        return nullptr;
    }
    if (!alignment_metric(options->metric)) {
        propagate_error(usearch_error_config_mismatch_k, error, "The weighted constructor takes an alignment kind");
        return nullptr;
    }
    index_t* index = new (std::nothrow) index_t();
    if (!index) {
        propagate_error(usearch_error_out_of_memory_k, error, "The handle could not be allocated");
        return nullptr;
    }
    index->options.options = *options;
    if (!index->options.options.connectivity) index->options.options.connectivity = us::connectivity_16_k;
    // Copied rather than borrowed, so a binding whose matrix dies after this call still scores.
    index->options.substitution.assign(substitution_matrix, substitution_matrix + alphabet_k * alphabet_k);
    index->options.gap_open = gap_open;
    index->options.gap_extend = gap_extend;
    return index;
}

void usearch_sequence_free(usearch_sequence_index_t handle) { delete static_cast<index_t*>(handle); }

void usearch_sequence_reserve(usearch_sequence_index_t handle, std::size_t entries,
                              [[maybe_unused]] std::size_t payload_bytes, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(entries))
        propagate_error(usearch_error_out_of_memory_k, error, "The capacity could not be reserved");
}

void usearch_sequence_clear(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    index.reset();
}

std::size_t usearch_sequence_size(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    return index.entries.size();
}
std::size_t usearch_sequence_capacity(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return index.entries.capacity();
}
std::size_t usearch_sequence_connectivity(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).options.options.connectivity;
}
std::size_t usearch_sequence_memory_usage(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    return payload_extent(index, rows) + index.options.substitution.size();
}
char const* usearch_sequence_capabilities([[maybe_unused]] usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return "serial";
}

std::size_t usearch_sequence_add(usearch_sequence_index_t handle, usearch_executor_t executor, usearch_keys_view_t keys,
                                 usearch_bytes_view_t const* sequences,
                                 [[maybe_unused]] usearch_sequence_insertion_options_t const* options,
                                 usearch_progress_callback_t progress, void* progress_state,
                                 usearch_sequence_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    if (reject_keys(keys, error)) return 0;
    std::size_t const count = keys.count;
    if (reject_if(!sequences, usearch_error_malformed_input_k, "The tape must name storage", error)) return 0;

    std::unique_lock writers(index.mutex);
    if (reject_oversubscribed(index, keys, index.options.options.values_per_key, error)) return 0;
    if (!index.accommodate(index.entries.size() + count)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The batch could not be accommodated");
        return 0;
    }

    std::size_t const watermark = index.entries.size();
    payload_t payload;
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        bytes_view_t const sequence = viewed(sequences[row]);
        payload.assign(sequence.begin(), sequence.end());
        index.append(keys.data[row], payload);
        ++published;
        if (!keep_going(progress, progress_state, published, count)) break;
    }
    // This modality's batch is all-or-nothing, a cancellation included, so a stopped tape unwinds
    // to the watermark rather than leaving a prefix published.
    if (published != count) {
        index.unwind(watermark);
        propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the batch");
        return 0;
    }
    if (stats) {
        stats->visited_members = published;
        stats->computed_distances = published * index.entries.size();
        stats->restarts = 1;
    }
    return count;
}

std::size_t usearch_sequence_refine(usearch_sequence_index_t handle, usearch_executor_t executor,
                                    [[maybe_unused]] usearch_sequence_insertion_options_t const* options,
                                    usearch_progress_callback_t progress, void* progress_state,
                                    usearch_sequence_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    // Cancelling is not an error here, the header having made a stopped sweep merely less improved.
    for (std::size_t row = 0; row != rows; ++row)
        if (!keep_going(progress, progress_state, row + 1, rows)) break;
    if (stats) stats->visited_members = rows;
    return 0;
}

/** @brief The shared body of both search doors; the unfiltered one admits with a null predicate. */
void search_sequence_tape(usearch_sequence_index_t handle, usearch_executor_t executor,
                          usearch_bytes_view_t const* queries, std::size_t queries_count,
                          usearch_matches_span_t const* matches, usearch_sequence_search_options_t const* options,
                          usearch_predicate_t predicate, void* predicate_state, usearch_progress_callback_t progress,
                          void* progress_state, usearch_sequence_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_if(!queries, usearch_error_malformed_input_k, "The tape must name storage", error)) return;
    std::size_t keys_stride = 0, scores_stride = 0;
    if (reject_matches(matches, keys_stride, scores_stride, error)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_tape(index, queries, matches->wanted, options, stats, predicate,
                                             predicate_state, progress, progress_state, matches->keys, keys_stride,
                                             matches->scores, scores_stride, {matches->counts, queries_count});
    reject_cancelled(answered, queries_count, error);
}

void usearch_sequence_search(usearch_sequence_index_t handle, usearch_executor_t executor,
                             usearch_bytes_view_t const* queries, std::size_t queries_count,
                             usearch_matches_span_t const* matches, usearch_sequence_search_options_t const* options,
                             usearch_progress_callback_t progress, void* progress_state,
                             usearch_sequence_traversal_stats_t* stats, usearch_error_t* error) {
    search_sequence_tape(handle, executor, queries, queries_count, matches, options, nullptr, nullptr, progress,
                         progress_state, stats, error);
}

void usearch_sequence_search_if(usearch_sequence_index_t handle, usearch_executor_t executor,
                                usearch_bytes_view_t const* queries, std::size_t queries_count,
                                usearch_matches_span_t const* matches, usearch_sequence_search_options_t const* options,
                                usearch_predicate_t predicate, void* predicate_state,
                                usearch_progress_callback_t progress, void* progress_state,
                                usearch_sequence_traversal_stats_t* stats, usearch_error_t* error) {
    search_sequence_tape(handle, executor, queries, queries_count, matches, options, predicate, predicate_state,
                         progress, progress_state, stats, error);
}

std::size_t usearch_sequence_count(usearch_sequence_index_t handle, usearch_key_t key,
                                   usearch_sequence_key_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const rows = rows_under(index, key);
    std::size_t total_bytes = 0, largest_bytes = 0;
    for (std::size_t row : rows) {
        total_bytes += index.entries[row].payload.size();
        largest_bytes = std::max(largest_bytes, index.entries[row].payload.size());
    }
    if (stats) {
        stats->entries = rows.size();
        stats->total_bytes = total_bytes;
        stats->largest_bytes = largest_bytes;
    }
    return rows.size();
}

std::size_t usearch_sequence_get(usearch_sequence_index_t handle, usearch_key_t key, char* sequences,
                                 std::size_t wanted, std::size_t stride, std::size_t* lengths, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const rows = rows_under(index, key);
    for (std::size_t found = 0; found != rows.size() && found < wanted; ++found) {
        payload_t const& payload = index.entries[rows[found]].payload;
        std::size_t const copied_bytes = std::min(stride, payload.size());
        if (sequences) {
            [[maybe_unused]] status_t const copied = copy_bytes(bytes_of(sequences + found * stride, copied_bytes),
                                                                bytes_view_t(payload).first(copied_bytes));
        }
        if (lengths) lengths[found] = payload.size();
    }
    return rows.size();
}

void usearch_sequence_merge(usearch_sequence_index_t handle, usearch_executor_t executor,
                            usearch_sequence_index_t const* donors, std::size_t donors_count,
                            usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;

    std::size_t merged = 0;
    bool accommodated = true;
    while (accommodated && merged != donors_count) {
        index_t& donor = borrowed<index_t>(donors[merged]);
        std::unique_lock writers(index.mutex);
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        ++merged;
        if (!keep_going(progress, progress_state, merged, donors_count)) break;
    }
    // Only an absorbed donor is consumed: a stopped merge leaves the rest live handles the caller
    // still owns and must free, as the header promises.
    for (std::size_t position = 0; position != merged; ++position) delete static_cast<index_t*>(donors[position]);
    if (!accommodated)
        return propagate_error(usearch_error_out_of_memory_k, error, "The donor's entries could not be accommodated");
    reject_cancelled(merged, donors_count, error);
}

usearch_sequence_cursor_t usearch_sequence_cursor(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    cursor_t* cursor = new (std::nothrow) cursor_t {&index, 0, index.entries.size()};
    if (!cursor) propagate_error(usearch_error_out_of_memory_k, error, "The cursor could not be allocated");
    return cursor;
}

std::size_t usearch_sequence_cursor_next(usearch_sequence_cursor_t handle, usearch_keys_span_t keys,
                                         usearch_bytes_view_t* sequences, usearch_error_t* error) {
    propagate_success(error);
    cursor_t& cursor = borrowed<cursor_t>(handle);
    index_t& index = *cursor.index;
    if (reject_keys_span(keys, error)) return 0;
    std::shared_lock readers(index.mutex);
    std::size_t const written = std::min(keys.capacity, cursor.limit - cursor.position);
    std::span<usearch_key_t> const keys_row {keys.data, written};
    for (std::size_t row = 0; row != written; ++row) {
        entry_t const& entry = index.entries[cursor.position + row];
        keys_row[row] = entry.key;
        // Views borrow the arena's own bytes, which growth relocates the row of but never the
        // payload buffer itself, so they survive concurrent insertion.
        if (sequences) sequences[row] = as_bytes(entry.payload);
    }
    cursor.position += written;
    return written;
}

void usearch_sequence_cursor_free(usearch_sequence_cursor_t handle) { delete static_cast<cursor_t*>(handle); }

std::size_t usearch_sequence_storage_requirements(usearch_sequence_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::optional<std::size_t> const length = serialized_bytes(index, index.entries.size());
    if (!length) {
        propagate_error(usearch_error_overflow_k, error, "The checkpoint would not fit an addressable buffer");
        return 0;
    }
    return *length;
}

void usearch_sequence_save(usearch_sequence_index_t handle, usearch_executor_t executor, usearch_bytes_view_t path,
                           usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (!length)
        return propagate_error(usearch_error_overflow_k, error, "The checkpoint would not fit an addressable buffer");
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(path, blob, error)) return;
    keep_going(progress, progress_state, 1, 1);
}

void usearch_sequence_load(usearch_sequence_index_t handle, usearch_executor_t executor, usearch_bytes_view_t path,
                           usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    std::unique_lock writers(index.mutex);
    if (!read_from(index, blob, error)) return;
    keep_going(progress, progress_state, 1, 1);
}

void usearch_sequence_metadata(usearch_bytes_view_t path, usearch_sequence_options_t* options, usearch_error_t* error) {
    propagate_success(error);
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return;
    if (!adopt(*options, prefix))
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
}

} // extern "C"
