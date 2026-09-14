/**
 *  @file src_stub/dense.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_dense_* C ABI, letting a language binding link
 *      and run its suite in seconds instead of building the engine.
 *
 *  Every SDK in this repository talks to the same C ABI, and most of what breaks in one is the
 *  binding rather than the engine: a leaked handle, a stride counted in scalars instead of
 *  bytes, an unread error, an output row written past its capacity. Finding those should not
 *  cost a build of the instantiation grid, so this file answers the dense verbs with the C++
 *  standard library alone - no submodules, no SIMD, no CUDA - over the store and locks of
 *  @c lib.hpp. A binding suite runs unchanged against this library or the engine's.
 *
 *  Search here is an exact scan, so recall is trivially perfect and ties fall in insertion
 *  order. What this proves is the binding: handle lifetime, strided input and output, the
 *  callback ABI, the truncation and watermark contracts, and error propagation. It is not an
 *  engine and never becomes one.
 *
 *  @section dense_stub_blind_spots What It Cannot Tell You
 *
 *  A test double this lossy invites false confidence, so its limits are worth stating plainly.
 *  Entries are stored as @c double and @c quantization is ignored, so quantization loss is
 *  invisible and an @c i8 index round-trips exactly. Every scalar kind the ABI names is decoded,
 *  the mini-floats and the packed nibble and bit kinds among them, so a row's byte extent is
 *  computed here exactly as the engine computes it and a binding sizing rows wrongly for a narrow
 *  float or a sub-byte kind fails on the shape instead of passing quietly. The metric and the
 *  quantization are never checked against each other, so the bit-set kinds read whatever rows
 *  they are handed where the engine admits @c u1x8 alone. A custom metric callback is ignored
 *  entirely, and @c values_per_key bounds nothing - a key holds as many entries as it is handed,
 *  the header naming no failure for exceeding it. Refinement keeps no edges to re-select, so its
 *  return is a constant zero rather than a settling signal, and a caller must not read it as one.
 *  Residency and huge pages are inert. Traversal statistics are plausible fiction: shaped
 *  correctly, invented numerically, so assert that they move, never what they equal. A checkpoint
 *  is a binary blob of this file's own memory image - not SafeGraphs, not portable across builds,
 *  and readable by nothing but the stand-in that wrote it.
 */
#include <cmath> // `std::sqrt`, `std::exp`

#include <algorithm>     // `std::stable_sort`, `std::min`, `std::max`
#include <optional>      // `std::optional`
#include <span>          // `std::span`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include "usearch/types.h"
#include "usearch/dense_index.h"

#include "lib.hpp"

namespace us = unum::usearch;
using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace {

#pragma region Storage

using payload_t = std::vector<double>;
using entry_t = entry<payload_t>;

static_assert(std::is_nothrow_move_constructible_v<entry_t>, "reallocation must move rows, never copy them");

using index_t = handle<payload_t, usearch_dense_options_t>;

struct cursor_t {

    /** The index being walked; it must outlive the cursor. */
    index_t* index;

    /** The next row to hand out. */
    std::size_t position;

    /** The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;
};

struct finder_t {

    /** Every scored match, nearest first - a snapshot, so it needs no further locking. */
    std::vector<match_t> ranked;

    /** The next match to hand out. */
    std::size_t position;
};

/**
 *  @brief Resolves an input matrix against the frozen geometry, answering whether to refuse it.
 *
 *  The width the caller believes a row has is the one thing the loose parameters never carried,
 *  so this is where a narrower row becomes an error rather than an over-read. A zero stride
 *  resolves to the natural pitch, as the descriptor's docblock promises.
 */
bool reject_view(usearch_matrix_view_t const* matrix, index_t const& index, std::size_t& stride,
                 usearch_error_t* error) {
    if (reject_if(!matrix || !matrix->data, usearch_error_malformed_input_k, "The matrix must name storage", error))
        return true;
    if (reject_scalar(matrix->scalar_kind, error)) return true;
    if (reject_if(matrix->columns != index.options.dimensions, usearch_error_shape_mismatch_k,
                  "The matrix rows are not as wide as the index", error))
        return true;
    std::size_t const natural = row_extent(matrix->scalar_kind, matrix->columns);
    if (reject_if(matrix->stride_bytes && matrix->stride_bytes < natural, usearch_error_unsupported_layout_k,
                  "The stride is shorter than one row", error))
        return true;
    stride = matrix->stride_bytes ? matrix->stride_bytes : natural;
    return false;
}

/** The output twin of @c reject_view, whose width the caller chooses rather than the index. */
bool reject_span(usearch_matrix_span_t const* matrix, std::size_t columns, std::size_t& stride,
                 usearch_error_t* error) {
    if (reject_if(!matrix || !matrix->data, usearch_error_malformed_input_k, "The matrix must name storage", error))
        return true;
    if (reject_scalar(matrix->scalar_kind, error)) return true;
    if (reject_if(matrix->columns != columns, usearch_error_shape_mismatch_k,
                  "The output rows are not as wide as the rows they receive", error))
        return true;
    std::size_t const natural = row_extent(matrix->scalar_kind, matrix->columns);
    if (reject_if(matrix->stride_bytes && matrix->stride_bytes < natural, usearch_error_unsupported_layout_k,
                  "The stride is shorter than one row", error))
        return true;
    stride = matrix->stride_bytes ? matrix->stride_bytes : natural;
    return false;
}

/** Whether the geometry is frozen; a zeroed width makes every derived extent zero. */
bool reject_unfrozen(index_t const& index, usearch_error_t* error) {
    return reject_if(!index.options.dimensions, usearch_error_uninitialized_k, "The geometry was never frozen", error);
}

#pragma endregion Storage

#pragma region Scoring

/** Scores one pair under the frozen kind. @c usearch_dense_init refuses a kind this
 *  build does not know, so every enumerator reaching here is one of the five. */
double distance(usearch_dense_metric_t metric, std::span<double const> first, std::span<double const> second) {
    double dot = 0, first_norm = 0, second_norm = 0, squared = 0;
    std::size_t differing = 0, shared = 0, either = 0;
    for (std::size_t position = 0; position != first.size(); ++position) {
        dot += first[position] * second[position];
        first_norm += first[position] * first[position];
        second_norm += second[position] * second[position];
        double const delta = first[position] - second[position];
        squared += delta * delta;
        differing += first[position] != second[position];
        shared += first[position] != 0 && second[position] != 0;
        either += first[position] != 0 || second[position] != 0;
    }
    switch (metric) {
    case usearch_dense_metric_inner_k: return 1.0 - dot;
    // The kind names L2 itself rather than its square, so the root is part of the answer.
    case usearch_dense_metric_euclidean_k: return std::sqrt(squared);
    case usearch_dense_metric_hamming_k: return static_cast<double>(differing);
    case usearch_dense_metric_jaccard_k:
        return either ? 1.0 - static_cast<double>(shared) / static_cast<double>(either) : 0.0;
    default: {
        double const denominator = std::sqrt(first_norm * second_norm);
        return denominator > 0 ? 1.0 - dot / denominator : 1.0;
    }
    }
}

/** Whether the kind is one this build implements at all. */
bool known_metric(usearch_dense_metric_t metric) {
    return metric == usearch_dense_metric_inner_k || metric == usearch_dense_metric_angular_k ||
           metric == usearch_dense_metric_euclidean_k || metric == usearch_dense_metric_hamming_k ||
           metric == usearch_dense_metric_jaccard_k;
}

double dot_product(std::span<double const> first, std::span<double const> second) {
    double dot = 0;
    for (std::size_t position = 0; position != first.size(); ++position) dot += first[position] * second[position];
    return dot;
}

/**
 *  @brief The shared body of both search forms; the filtered one passes a predicate.
 *  @return Queries answered, short of @p counts.size() exactly when the handler cancelled.
 */
std::size_t search_rows(index_t& index, void const* queries, std::size_t queries_count, std::size_t queries_stride,
                        usearch_scalar_t scalar_kind, std::size_t wanted, usearch_dense_search_options_t const* options,
                        usearch_dense_traversal_stats_t* stats, usearch_predicate_t predicate, void* predicate_state,
                        usearch_progress_callback_t progress, void* progress_state, usearch_key_t* keys,
                        std::size_t keys_stride, usearch_distance_t* distances, std::size_t distances_stride,
                        std::span<std::size_t> counts) {
    std::size_t const rows = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const query_bytes = row_extent(scalar_kind, dimensions);
    // Let the expansion factor move the reported work, so a sweep can see it change.
    if (stats && options && options->expansion_factor)
        stats->computed_distances += options->expansion_factor * queries_count;

    // A cancelled sweep still leaves every count written, so a caller reads the whole array.
    std::fill(counts.begin(), counts.end(), 0);
    std::vector<double> query_row(dimensions);
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        read_row(row_at<std::byte>(queries, query, queries_stride, query_bytes), scalar_kind, query_row);
        rank<payload_t>(
            std::span<entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
            [&](payload_t const& payload) { return distance(index.options.metric, query_row, payload); }, ranked);
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

/** What @c load needs before it can read a row. The blob is this file's own memory image
 *  rather than a format: no magic, no version, and nothing but the stand-in reads it back. */
struct prefix_t {

    /** The frozen row width. */
    std::size_t dimensions;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many rows follow the prefix. */
    std::size_t entries;

    /** What entries are compared by. */
    usearch_dense_metric_t metric;

    /** The scalar kind the rows were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;

    /** Entries that may share one key, which late interaction needs above one. */
    std::size_t values_per_key;
};

std::size_t stored_row_bytes(std::size_t dimensions) { return sizeof(usearch_key_t) + dimensions * sizeof(double); }

/** The whole blob's extent, empty when that many rows could not be addressed. */
std::optional<std::size_t> serialized_bytes(index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(prefix_t), rows, stored_row_bytes(index.options.dimensions));
}

void write_into(index_t const& index, std::size_t rows, bytes_span_t target) {
    prefix_t prefix {};
    prefix.dimensions = index.options.dimensions;
    prefix.quantization = index.options.quantization;
    prefix.values_per_key = index.options.values_per_key;
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.metric = index.options.metric;
    write_pod(target, prefix);

    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        entry_t const& entry = index.entries[row];
        std::size_t const payload_bytes = entry.payload.size() * sizeof(double);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(payload_bytes),
                                                            bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** Reads the prefix and checks the blob is long enough for the rows it claims. */
bool parse_prefix(prefix_t& prefix, bytes_view_t blob, usearch_error_t* error) {
    if (reject_if(blob.size() < sizeof(prefix_t), usearch_error_malformed_input_k, "The checkpoint ends early", error))
        return false;
    prefix = read_pod<prefix_t>(blob);
    return fits_blob(blob, sizeof(prefix_t), prefix.entries, stored_row_bytes(prefix.dimensions), error);
}

/** @brief Describes a checkpoint, answering whether its metric is one we know.
 *
 *  Narrower than @c adopt: this fills only what a checkpoint froze, where that restores a handle's
 *  whole configuration including the bounds its mutable state carries.
 */
bool describe(usearch_dense_metadata_t& metadata, prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    metadata.dimensions = prefix.dimensions;
    metadata.metric = prefix.metric;
    metadata.connectivity = prefix.connectivity;
    metadata.quantization = prefix.quantization;
    return true;
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we know. */
bool adopt(usearch_dense_options_t& options, prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.dimensions = prefix.dimensions;
    options.metric = prefix.metric;
    options.connectivity = prefix.connectivity;
    options.quantization = prefix.quantization;
    options.values_per_key = prefix.values_per_key;
    return true;
}

/** Replaces the contents of @p index; the caller holds both locks exclusively. */
bool read_from(index_t& index, bytes_view_t blob, usearch_error_t* error) {
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return false;
    std::size_t const rows = prefix.entries;
    std::size_t const dimensions = prefix.dimensions;
    index.reset();
    if (!index.accommodate(rows)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The checkpoint's rows could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
        return false;
    }

    bytes_view_t cursor = blob.subspan(sizeof(prefix_t));
    std::vector<double> payload(dimensions);
    for (std::size_t row = 0; row != rows; ++row) {
        usearch_key_t const key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        [[maybe_unused]] status_t const copied = copy_bytes(bytes_of(payload.data(), payload.size()),
                                                            cursor.first(dimensions * sizeof(double)));
        cursor = cursor.subspan(dimensions * sizeof(double));
        index.append(key, payload);
    }
    return true;
}

#pragma endregion Serialization

#pragma region Attention

/** The value matrix a key may address, past which the entry is skipped as unmapped. */
constexpr std::size_t value_rows_limit_k = 1u << 30;

/** An entry's key is its row offset into the value matrix, parked in the low bytes. */
std::size_t value_offset(usearch_key_t const& key) {
    std::size_t offset = 0;
    for (unsigned byte_index = 8; byte_index != 16; ++byte_index) offset = (offset << 8) | key.bytes[byte_index];
    return offset;
}

#pragma endregion Attention

} // namespace

extern "C" {

char const* usearch_dense_metric_name(usearch_dense_metric_t metric) {
    switch (metric) {
    case usearch_dense_metric_inner_k: return "inner";
    case usearch_dense_metric_angular_k: return "angular";
    case usearch_dense_metric_euclidean_k: return "euclidean";
    case usearch_dense_metric_hamming_k: return "hamming";
    case usearch_dense_metric_jaccard_k: return "jaccard";
    default: return "unknown";
    }
}

usearch_dense_metric_t usearch_dense_metric_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_dense_metric_inner_k; code <= usearch_dense_metric_jaccard_k; ++code) {
        usearch_dense_metric_t const metric = static_cast<usearch_dense_metric_t>(code);
        if (named(name, usearch_dense_metric_name(metric))) return metric;
    }
    return usearch_dense_metric_unknown_k;
}

usearch_dense_index_t usearch_dense_init(usearch_dense_options_t const* options, usearch_error_t* error) {
    propagate_success(error);
    if (!options) {
        propagate_error(usearch_error_malformed_input_k, error, "The options are null");
        return nullptr;
    }
    if (!known_metric(options->metric)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The stand-in does not implement that metric");
        return nullptr;
    }
    index_t* index = new (std::nothrow) index_t();
    if (!index) {
        propagate_error(usearch_error_out_of_memory_k, error, "The handle could not be allocated");
        return nullptr;
    }
    index->options = *options;
    if (!index->options.connectivity) index->options.connectivity = us::connectivity_16_k;
    return index;
}

void usearch_dense_free(usearch_dense_index_t handle) { delete static_cast<index_t*>(handle); }

void usearch_dense_reserve(usearch_dense_index_t handle, std::size_t capacity, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    // Sizing the arena moves published rows, so no reader may be walking them.
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(capacity))
        propagate_error(usearch_error_out_of_memory_k, error, "The capacity could not be reserved");
}

void usearch_dense_clear(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    index.reset();
}

std::size_t usearch_dense_size(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    return index.entries.size();
}
std::size_t usearch_dense_capacity(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return index.entries.capacity();
}
std::size_t usearch_dense_dimensions(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).options.dimensions;
}
std::size_t usearch_dense_connectivity(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).options.connectivity;
}
std::size_t usearch_dense_memory_usage(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return index.entries.capacity() * stored_row_bytes(index.options.dimensions);
}
char const* usearch_dense_capabilities([[maybe_unused]] usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return "serial";
}

std::size_t usearch_dense_add(usearch_dense_index_t handle, usearch_executor_t executor, usearch_keys_view_t keys,
                              usearch_matrix_view_t const* vectors,
                              [[maybe_unused]] usearch_dense_insertion_options_t const* options,
                              usearch_progress_callback_t progress, void* progress_state,
                              usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    if (reject_unfrozen(index, error)) return 0;
    std::size_t stride = 0;
    if (reject_view(vectors, index, stride, error)) return 0;
    std::size_t const count = vectors->rows;
    if (reject_keys_pairing(keys, count, error)) return 0;
    usearch_scalar_t const scalar_kind = vectors->scalar_kind;

    // Sizing precedes the first append, so a batch that cannot be accommodated adds nothing -
    // the all-or-nothing insertion this modality alone promises.
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(index.entries.size() + count)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The batch could not be accommodated");
        return 0;
    }

    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(scalar_kind, dimensions);
    std::vector<double> payload(dimensions);
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        read_row(row_at<std::byte>(vectors->data, row, stride, row_bytes), scalar_kind, payload);
        index.append(keys.data[row], payload);
        ++published;
        if (!keep_going(progress, progress_state, published, count)) break;
    }
    if (stats) {
        stats->visited_members = published;
        stats->computed_distances = published * index.entries.size();
        stats->restarts = 1;
    }
    reject_cancelled(published, count, error);
    return published;
}

std::size_t usearch_dense_refine(usearch_dense_index_t handle, usearch_executor_t executor,
                                 [[maybe_unused]] usearch_dense_insertion_options_t const* options,
                                 usearch_progress_callback_t progress, void* progress_state,
                                 usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
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

/** The shared body of both search doors; the unfiltered one admits with a null predicate. */
void search_matrix(usearch_dense_index_t handle, usearch_executor_t executor, usearch_matrix_view_t const* queries,
                   usearch_matches_span_t const* matches, usearch_dense_search_options_t const* options,
                   usearch_predicate_t predicate, void* predicate_state, usearch_progress_callback_t progress,
                   void* progress_state, usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    std::size_t queries_stride = 0, keys_stride = 0, scores_stride = 0;
    if (reject_view(queries, index, queries_stride, error)) return;
    if (reject_matches(matches, keys_stride, scores_stride, error)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_rows(index, queries->data, queries->rows, queries_stride, queries->scalar_kind,
                                             matches->wanted, options, stats, predicate, predicate_state, progress,
                                             progress_state, matches->keys, keys_stride, matches->scores, scores_stride,
                                             {matches->counts, queries->rows});
    reject_cancelled(answered, queries->rows, error);
}

void usearch_dense_search(usearch_dense_index_t handle, usearch_executor_t executor,
                          usearch_matrix_view_t const* queries, usearch_matches_span_t const* matches,
                          usearch_dense_search_options_t const* options, usearch_progress_callback_t progress,
                          void* progress_state, usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    search_matrix(handle, executor, queries, matches, options, nullptr, nullptr, progress, progress_state, stats,
                  error);
}

void usearch_dense_search_if(usearch_dense_index_t handle, usearch_executor_t executor,
                             usearch_matrix_view_t const* queries, usearch_matches_span_t const* matches,
                             usearch_dense_search_options_t const* options, usearch_predicate_t predicate,
                             void* predicate_state, usearch_progress_callback_t progress, void* progress_state,
                             usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    search_matrix(handle, executor, queries, matches, options, predicate, predicate_state, progress, progress_state,
                  stats, error);
}

std::size_t usearch_dense_count(usearch_dense_index_t handle, usearch_key_t key, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return index.by_key.count(keyed(key));
}

std::size_t usearch_dense_get(usearch_dense_index_t handle, usearch_key_t key, usearch_matrix_span_t const* vectors,
                              usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::size_t stride = 0;
    if (vectors && reject_span(vectors, index.options.dimensions, stride, error)) return 0;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const rows = rows_under(index, key);
    if (vectors) {
        std::size_t const row_bytes = row_extent(vectors->scalar_kind, vectors->columns);
        for (std::size_t found = 0; found != rows.size() && found < vectors->capacity; ++found)
            write_row(row_at<std::byte>(vectors->data, found, stride, row_bytes), vectors->scalar_kind,
                      index.entries[rows[found]].payload);
    }
    return rows.size();
}

void usearch_dense_merge(usearch_dense_index_t handle, usearch_executor_t executor, usearch_dense_index_t* donors,
                         std::size_t donors_count, usearch_progress_callback_t progress, void* progress_state,
                         usearch_error_t* error) {
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
    // Only an absorbed donor is consumed, and its slot is nulled as it goes, so a stopped merge
    // leaves the rest live handles the caller still owns and must free.
    for (std::size_t position = 0; position != merged; ++position) {
        delete static_cast<index_t*>(donors[position]);
        donors[position] = nullptr;
    }
    if (!accommodated)
        return propagate_error(usearch_error_out_of_memory_k, error, "The donor's rows could not be accommodated");
    reject_cancelled(merged, donors_count, error);
}

usearch_dense_cursor_t usearch_dense_cursor(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    cursor_t* cursor = new (std::nothrow) cursor_t {&index, 0, index.entries.size()};
    if (!cursor) propagate_error(usearch_error_out_of_memory_k, error, "The cursor could not be allocated");
    return cursor;
}

std::size_t usearch_dense_cursor_next(usearch_dense_cursor_t handle, usearch_keys_span_t keys,
                                      usearch_matrix_span_t const* vectors, usearch_error_t* error) {
    propagate_success(error);
    cursor_t& cursor = borrowed<cursor_t>(handle);
    index_t& index = *cursor.index;
    std::size_t stride = 0;
    if (reject_keys_span(keys, error)) return 0;
    if (vectors && reject_span(vectors, index.options.dimensions, stride, error)) return 0;
    if (vectors && reject_if(vectors->capacity < keys.capacity, usearch_error_shape_mismatch_k,
                             "The vector output holds fewer rows than the key run", error))
        return 0;
    std::shared_lock readers(index.mutex);
    std::size_t const written = std::min(keys.capacity, cursor.limit - cursor.position);
    std::span<usearch_key_t> const keys_row {keys.data, written};
    for (std::size_t row = 0; row != written; ++row) {
        entry_t const& entry = index.entries[cursor.position + row];
        keys_row[row] = entry.key;
        if (vectors) {
            std::size_t const row_bytes = row_extent(vectors->scalar_kind, vectors->columns);
            write_row(row_at<std::byte>(vectors->data, row, stride, row_bytes), vectors->scalar_kind, entry.payload);
        }
    }
    cursor.position += written;
    return written;
}

void usearch_dense_cursor_free(usearch_dense_cursor_t handle) { delete static_cast<cursor_t*>(handle); }

usearch_dense_finder_t usearch_dense_find(usearch_dense_index_t handle, usearch_executor_t executor,
                                          usearch_matrix_view_t const* query, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return nullptr;
    if (reject_unfrozen(index, error)) return nullptr;
    std::size_t stride = 0;
    if (reject_view(query, index, stride, error)) return nullptr;
    if (reject_if(query->rows != 1, usearch_error_shape_mismatch_k, "A finder takes exactly one query row", error))
        return nullptr;
    finder_t* finder = new (std::nothrow) finder_t();
    if (!finder) {
        propagate_error(usearch_error_out_of_memory_k, error, "The finder could not be allocated");
        return nullptr;
    }
    usearch_scalar_t const scalar_kind = query->scalar_kind;
    std::size_t const dimensions = index.options.dimensions;
    std::vector<double> query_row(dimensions);
    std::shared_lock readers(index.mutex);
    read_row(row_at<std::byte>(query->data, 0, stride, row_extent(scalar_kind, dimensions)), scalar_kind, query_row);
    rank<payload_t>(
        index.entries, admitting(nullptr, nullptr),
        [&](payload_t const& payload) { return distance(index.options.metric, query_row, payload); }, finder->ranked);
    finder->position = 0;
    return finder;
}

std::size_t usearch_dense_find_next(usearch_dense_finder_t handle, usearch_keys_span_t keys,
                                    usearch_distance_t* distances, usearch_error_t* error) {
    propagate_success(error);
    finder_t& finder = borrowed<finder_t>(handle);
    if (reject_keys_span(keys, error)) return 0;
    if (reject_if(!distances, usearch_error_malformed_input_k, "The distance output must name storage", error))
        return 0;
    std::size_t const written = std::min(keys.capacity, finder.ranked.size() - finder.position);
    std::span<usearch_key_t> const keys_row {keys.data, written};
    std::span<usearch_distance_t> const distances_row {distances, written};
    for (std::size_t match = 0; match != written; ++match) {
        keys_row[match] = finder.ranked[finder.position + match].key;
        distances_row[match] = finder.ranked[finder.position + match].distance;
    }
    finder.position += written;
    return written;
}

void usearch_dense_find_free(usearch_dense_finder_t handle) { delete static_cast<finder_t*>(handle); }

std::size_t usearch_dense_storage_requirements(usearch_dense_index_t handle, usearch_error_t* error) {
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

void usearch_dense_save(usearch_dense_index_t handle, usearch_executor_t executor, usearch_bytes_view_t path,
                        usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    // Saving is a reader: it demands only that nothing move or erase rows under it.
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (!length)
        return propagate_error(usearch_error_overflow_k, error, "The checkpoint would not fit an addressable buffer");
    // The blob is one chunk, so the handler's only say is whether it is written at all.
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(path, blob, error)) return;
    keep_going(progress, progress_state, 1, 1);
}

void usearch_dense_load(usearch_dense_index_t handle, usearch_executor_t executor, usearch_bytes_view_t path,
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

void usearch_dense_metadata(usearch_bytes_view_t path, usearch_dense_metadata_t* metadata, usearch_error_t* error) {
    propagate_success(error);
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return;
    if (!describe(*metadata, prefix))
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
}

void usearch_dense_save_buffer(usearch_dense_index_t handle, usearch_executor_t executor, usearch_bytes_span_t buffer,
                               usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const needed = serialized_bytes(index, rows);
    if (!needed)
        return propagate_error(usearch_error_overflow_k, error, "The checkpoint would not fit an addressable buffer");
    if (buffer.length < *needed)
        return propagate_error(usearch_error_shape_mismatch_k, error,
                               "The buffer is smaller than the serialized length");
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    write_into(index, rows, bytes_of(buffer.data, buffer.length));
    keep_going(progress, progress_state, 1, 1);
}

void usearch_dense_load_buffer(usearch_dense_index_t handle, usearch_executor_t executor, usearch_bytes_view_t buffer,
                               usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    std::unique_lock writers(index.mutex);
    if (!read_from(index, bytes_of(buffer.data, buffer.length), error)) return;
    keep_going(progress, progress_state, 1, 1);
}

void usearch_dense_metadata_buffer(usearch_bytes_view_t buffer, usearch_dense_metadata_t* metadata,
                                   usearch_error_t* error) {
    propagate_success(error);
    prefix_t prefix {};
    if (!parse_prefix(prefix, bytes_of(buffer.data, buffer.length), error)) return;
    if (!describe(*metadata, prefix))
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
}

void usearch_dense_multiply(usearch_dense_index_t handle, usearch_executor_t executor,
                            usearch_matrix_view_t const* left, usearch_matches_span_t const* products,
                            [[maybe_unused]] usearch_dense_search_options_t const* options,
                            usearch_progress_callback_t progress, void* progress_state,
                            usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_error_config_mismatch_k,
                  "A product is a logit, so the kind must be inner product", error))
        return;
    std::size_t left_stride = 0, keys_stride = 0, scores_stride = 0;
    if (reject_view(left, index, left_stride, error)) return;
    if (reject_matches(products, keys_stride, scores_stride, error)) return;
    usearch_scalar_t const scalar_kind = left->scalar_kind;
    std::size_t const rows_count = left->rows, wanted = products->wanted;

    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(scalar_kind, dimensions);
    std::span<std::size_t> const counts_column {products->counts, rows_count};

    std::fill(counts_column.begin(), counts_column.end(), 0);
    std::vector<double> left_row(dimensions);
    std::vector<match_t> scored;
    std::size_t multiplied = 0;
    for (std::size_t row = 0; row != rows_count; ++row) {
        read_row(row_at<std::byte>(left->data, row, left_stride, row_bytes), scalar_kind, left_row);
        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(products->keys, row, keys_stride, wanted);
        std::span<usearch_distance_t> const products_row = row_at<usearch_distance_t>(products->scores, row,
                                                                                      scores_stride, wanted);
        scored.clear();
        scored.reserve(indexed);
        for (entry_t const& entry : index.entries) scored.push_back({dot_product(left_row, entry.payload), entry.key});
        // A truncated row keeps its nearest pairs, and a product is a similarity, so the largest
        // rank first - the pairs dropped are the small-magnitude ones the header names.
        std::stable_sort(scored.begin(), scored.end(),
                         [](match_t const& first, match_t const& second) { return first.distance > second.distance; });
        std::size_t const written = std::min(wanted, indexed);
        for (std::size_t column = 0; column != written; ++column)
            keys_row[column] = scored[column].key, products_row[column] = scored[column].distance;
        // The true count, which the header allows to exceed `wanted` to signal truncation.
        counts_column[row] = indexed;
        if (stats) stats->computed_distances += indexed;
        ++multiplied;
        if (!keep_going(progress, progress_state, multiplied, rows_count)) break;
    }
    reject_cancelled(multiplied, rows_count, error);
}

void usearch_dense_attend(usearch_dense_index_t handle, usearch_executor_t executor,
                          usearch_matrix_view_t const* queries, usearch_matrix_view_t const* values,
                          usearch_matrix_span_t const* outputs, usearch_dense_softmax_state_t* carry,
                          usearch_dense_attention_options_t const* options, usearch_progress_callback_t progress,
                          void* progress_state, usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_error_config_mismatch_k,
                  "A softmax weighs logits, so the kind must be inner product", error))
        return;

    // The value table answers to no frozen width - its rows are addressed by key, and its own
    // width is the value dimensionality - so it is checked for storage and pitch, not geometry.
    std::size_t queries_stride = 0, values_stride = 0, outputs_stride = 0;
    if (reject_view(queries, index, queries_stride, error)) return;
    if (reject_if(!values || !values->data, usearch_error_malformed_input_k, "The values must name storage", error))
        return;
    if (reject_scalar(values->scalar_kind, error)) return;
    std::size_t const value_dimensions = values->columns;
    if (reject_span(outputs, value_dimensions, outputs_stride, error)) return;
    if (reject_if(outputs->capacity < queries->rows, usearch_error_shape_mismatch_k,
                  "The outputs hold fewer rows than there are queries", error))
        return;
    std::size_t const values_natural = row_extent(values->scalar_kind, value_dimensions);
    if (reject_if(values->stride_bytes && values->stride_bytes < values_natural, usearch_error_unsupported_layout_k,
                  "The values stride is shorter than one row", error))
        return;
    values_stride = values->stride_bytes ? values->stride_bytes : values_natural;

    usearch_scalar_t const queries_scalar_kind = queries->scalar_kind;
    usearch_scalar_t const values_scalar_kind = values->scalar_kind;
    usearch_scalar_t const outputs_scalar_kind = outputs->scalar_kind;
    std::size_t const queries_count = queries->rows;

    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const query_bytes = row_extent(queries_scalar_kind, dimensions);
    std::size_t const value_bytes = row_extent(values_scalar_kind, value_dimensions);
    std::size_t const output_bytes = row_extent(outputs_scalar_kind, value_dimensions);
    double const scale = options && options->score_scale > 0 ? options->score_scale
                                                             : 1.0 / std::sqrt(std::max<double>(dimensions, 1));

    std::vector<double> query_row(dimensions), value(value_dimensions), accumulated(value_dimensions);
    std::size_t attended = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        read_row(row_at<std::byte>(queries->data, query, queries_stride, query_bytes), queries_scalar_kind, query_row);
        // The online-softmax recurrence: a running maximum, a partition, and a weighted sum.
        double maximum = -INFINITY, partition = 0;
        std::fill(accumulated.begin(), accumulated.end(), 0.0);
        for (std::size_t member = 0; member != indexed; ++member) {
            entry_t const& entry = index.entries[member];
            // The table now carries its own height, so a key past it is a wrong softmax rather
            // than a smaller one - refuse it instead of skipping the row.
            std::size_t const offset = value_offset(entry.key);
            if (reject_if(offset >= values->rows, usearch_error_shape_mismatch_k,
                          "An entry's key addresses a row past the value table", error))
                return;
            double const score = dot_product(query_row, entry.payload) * scale;
            read_row(row_at<std::byte>(values->data, offset, values_stride, value_bytes), values_scalar_kind, value);

            double const shifted = std::max(maximum, score);
            double const rescale = std::exp(maximum - shifted);
            double const weight = std::exp(score - shifted);
            partition = partition * rescale + weight;
            for (std::size_t position = 0; position != value_dimensions; ++position)
                accumulated[position] = accumulated[position] * rescale + weight * value[position];
            maximum = shifted;
        }
        for (std::size_t position = 0; position != value_dimensions; ++position)
            accumulated[position] = partition > 0 ? accumulated[position] / partition : 0.0;
        write_row(row_at<std::byte>(outputs->data, query, outputs_stride, output_bytes), outputs_scalar_kind,
                  accumulated);
        if (carry) carry[query] = {static_cast<float>(maximum), static_cast<float>(partition)};
        if (stats) stats->computed_distances += indexed;
        ++attended;
        if (!keep_going(progress, progress_state, attended, queries_count)) break;
    }
    reject_cancelled(attended, queries_count, error);
}

void usearch_dense_maxsim(usearch_dense_index_t handle, usearch_executor_t executor,
                          usearch_matrix_view_t const* queries, std::size_t rows_per_query,
                          usearch_matches_span_t const* matches,
                          [[maybe_unused]] usearch_dense_search_options_t const* options,
                          usearch_progress_callback_t progress, void* progress_state,
                          usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_error_config_mismatch_k,
                  "Late interaction sums maxima over logits, so the kind must be inner product", error))
        return;
    std::size_t queries_stride = 0, keys_stride = 0, scores_stride = 0;
    if (reject_view(queries, index, queries_stride, error)) return;
    if (reject_matches(matches, keys_stride, scores_stride, error)) return;
    if (reject_if(!rows_per_query || queries->rows % rows_per_query, usearch_error_shape_mismatch_k,
                  "The query rows do not divide into whole queries", error))
        return;
    usearch_scalar_t const scalar_kind = queries->scalar_kind;
    std::size_t const queries_count = queries->rows / rows_per_query, wanted = matches->wanted;

    std::shared_lock readers(index.mutex);

    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(scalar_kind, dimensions);
    std::span<std::size_t> const counts_column {matches->counts, queries_count};
    std::fill(counts_column.begin(), counts_column.end(), 0);

    // A document is one key holding its token rows, so the grouping happens here rather than in
    // the walk; an entry's rows need no ordering and two documents need not share a row count.
    std::vector<usearch_key_t> documents;
    std::vector<std::vector<std::size_t>> token_rows;
    std::unordered_map<us::uuid_t, std::size_t> position_of;
    for (std::size_t row = 0; row != index.entries.size(); ++row) {
        auto const [slot, emplace_status] = position_of.emplace(keyed(index.entries[row].key), documents.size());
        if (emplace_status) documents.push_back(index.entries[row].key), token_rows.emplace_back();
        token_rows[slot->second].push_back(row);
    }

    std::vector<double> query_rows(rows_per_query * dimensions);
    std::vector<match_t> ranked;
    std::size_t answered = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        for (std::size_t row = 0; row != rows_per_query; ++row)
            read_row(row_at<std::byte>(queries->data, query * rows_per_query + row, queries_stride, row_bytes),
                     scalar_kind, std::span<double>(query_rows).subspan(row * dimensions, dimensions));

        ranked.clear();
        for (std::size_t document = 0; document != documents.size(); ++document) {
            double summed = 0;
            for (std::size_t row = 0; row != rows_per_query; ++row) {
                std::span<double const> const query_row {query_rows.data() + row * dimensions, dimensions};
                double best = -INFINITY;
                for (std::size_t stored : token_rows[document])
                    best = std::max(best, dot_product(query_row, index.entries[stored].payload));
                summed += best;
            }
            ranked.push_back({summed, documents[document]});
        }
        // Summed maxima are a similarity, so the best key is the largest rather than the nearest.
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](match_t const& first, match_t const& second) { return first.distance > second.distance; });

        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(matches->keys, query, keys_stride, wanted);
        std::span<usearch_distance_t> const scores_row = row_at<usearch_distance_t>(matches->scores, query,
                                                                                    scores_stride, wanted);
        std::size_t const found = std::min(wanted, ranked.size());
        for (std::size_t match = 0; match != found; ++match)
            keys_row[match] = ranked[match].key, scores_row[match] = ranked[match].distance;
        counts_column[query] = found;
        if (stats) {
            stats->visited_members += index.entries.size();
            stats->computed_distances += documents.size() * rows_per_query;
            stats->restarts += 1;
        }
        ++answered;
        if (!keep_going(progress, progress_state, answered, queries_count)) break;
    }
    reject_cancelled(answered, queries_count, error);
}

} // extern "C"
