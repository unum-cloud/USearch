/**
 *  @file src_stub/dense.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_dense_* C ABI, letting a language binding link
 *      and run its suite in seconds instead of building the engine.
 *
 *  Every SDK in this repository talks to the same C ABI, and most of what breaks in one is the
 *  binding rather than the engine: a leaked handle, a stride counted in scalars instead of
 *  bytes, an unread outcome, an output row written past its capacity. Finding those should not
 *  cost a build of the instantiation grid, so this file answers the dense verbs with the C++
 *  standard library alone - no submodules, no SIMD, no CUDA - over the store and locks of
 *  @c lib.hpp. A binding suite runs unchanged against this library or the engine's.
 *
 *  Search here is an exact scan, so recall is trivially perfect and ties fall in insertion
 *  order. What this proves is the binding: handle lifetime, strided input and output, the
 *  callback ABI, the truncation and watermark contracts, and outcome propagation. It is not an
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
 *  they are handed where the engine admits @c u1x8 alone. @c values_per_key bounds nothing - a
 *  key holds as many entries as it is handed, the header naming no failure for exceeding it.
 *  Refinement keeps no edges to re-select, so its @c changed is a constant zero rather than a
 *  settling signal, and a caller must not read it as one. Residency and huge pages are inert, a
 *  view copies its checkpoint rather than mapping it, and only the RAM verbs answer - every VRAM
 *  and NVMe lifecycle verb refuses. Traversal statistics are plausible fiction: shaped
 *  correctly, invented numerically, so assert that they move, never what they equal. A checkpoint
 *  is a binary blob of this file's own memory image - not SafeGraphs, not portable across builds,
 *  and readable by nothing but the stand-in that wrote it.
 */
#include <cmath> // `std::sqrt`, `std::exp`

#include <algorithm>     // `std::stable_sort`, `std::min`, `std::max`, `std::find`
#include <limits>        // `std::numeric_limits`
#include <optional>      // `std::optional`
#include <span>          // `std::span`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include "usearch/types.h"
#include "usearch/dense_index.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace {

#pragma region Storage

using dense_payload_t = std::vector<double>;
using dense_entry_t = entry<dense_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<dense_entry_t>, "reallocation must move rows, never copy them");

/** What a handle froze at construction, or adopted from a checkpoint. */
struct dense_index_options_t {

    /** What entries are compared by. */
    usearch_dense_metric_t metric;

    /** The scalar kind rows were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;

    /** The frozen row width. */
    std::size_t dimensions;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Entries that may share one key, which late interaction needs above one. */
    std::size_t values_per_key;
};

} // namespace

struct usearch_dense_index_handle_t : handle<dense_payload_t, dense_index_options_t> {};

struct usearch_dense_cursor_handle_t {

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    usearch_dense_index_handle_t const* opened_on;

    /** The next row to hand out. */
    std::size_t position;

    /** The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

struct usearch_dense_search_stream_handle_t {

    /** The handle the stream was opened on, only ever compared with the one each pull passes. */
    usearch_dense_index_handle_t const* opened_on;

    /** Every scored match, nearest first - a snapshot, so it needs no further locking. */
    std::vector<match_t> ranked;

    /** The next match to hand out. */
    std::size_t position;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

namespace {

/**
 *  @brief Checks a run of input rows against the frozen geometry, answering whether to refuse it.
 *
 *  The view's width must be the index's own, and a stride narrower than it is where an over-read
 *  becomes an error. A zero stride repeats the first row.
 */
bool reject_view(usearch_rows_view_t const& rows, std::size_t count, usearch_dense_index_handle_t const& index,
                 usearch_outcome_t* outcome) {
    if (reject_if(!rows.data && count, usearch_status_invalid_argument_k, "The rows must name storage", outcome))
        return true;
    if (reject_if(count && rows.columns != index.options.dimensions, usearch_status_shape_mismatch_k,
                  "The rows' width differs from the frozen geometry", outcome))
        return true;
    if (reject_scalar(rows.scalar, outcome)) return true;
    std::size_t const natural = row_extent(rows.scalar, index.options.dimensions);
    return reject_if(rows.stride && rows.stride < natural, usearch_status_unsupported_layout_k,
                     "The stride is shorter than one row", outcome);
}

/** The output twin of @c reject_view, whose width the verb names rather than the index, and
 *  whose zero stride is refused since every row would land in one place. */
bool reject_span(usearch_rows_span_t const& rows, std::size_t columns, std::size_t count, usearch_outcome_t* outcome) {
    if (reject_if(!rows.data, usearch_status_invalid_argument_k, "The rows must name storage", outcome)) return true;
    if (reject_if(rows.columns != columns, usearch_status_shape_mismatch_k,
                  "The rows' width differs from the one the call writes", outcome))
        return true;
    if (reject_scalar(rows.scalar, outcome)) return true;
    if (reject_if(rows.stride < row_extent(rows.scalar, columns), usearch_status_unsupported_layout_k,
                  "The stride is shorter than one row", outcome))
        return true;
    return reject_if(rows.capacity < count, usearch_status_shape_mismatch_k,
                     "The output holds fewer rows than the call writes", outcome);
}

/** Whether the geometry is frozen; a zeroed width makes every derived extent zero. */
bool reject_unfrozen(usearch_dense_index_handle_t const& index, usearch_outcome_t* outcome) {
    return reject_if(!index.options.dimensions, usearch_status_uninitialized_k, "The geometry was never frozen",
                     outcome);
}

#pragma endregion Storage

#pragma region Scoring

/** Scores one pair under the frozen kind. @c usearch_dense_init_ram refuses a kind this
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
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
std::size_t search_rows(usearch_dense_index_handle_t& index, usearch_rows_view_t const& queries, std::size_t count,
                        usearch_matches_span_t const& matches, std::size_t expansion_factor,
                        usearch_dense_traversal_stats_t* stats, usearch_predicate_t predicate, void* predicate_state,
                        usearch_progress_callback_t progress, void* progress_state) {
    std::size_t const rows = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const query_bytes = row_extent(queries.scalar, dimensions);
    std::size_t const wanted = matches.wanted;
    // Let the expansion factor move the reported work, so a sweep can see it change.
    if (stats && expansion_factor) stats->computed_distances += expansion_factor * count;

    // A cancelled sweep still leaves every count written, so a caller reads the whole array.
    std::span<std::size_t> const counts {matches.counts, count};
    std::fill(counts.begin(), counts.end(), 0);
    std::vector<double> query_row(dimensions);
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        read_row(row_at<std::byte>(queries.data, query, queries.stride, query_bytes), queries.scalar, query_row);
        rank<dense_payload_t>(
            std::span<dense_entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
            [&](dense_payload_t const& payload) { return distance(index.options.metric, query_row, payload); }, ranked);
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

/** What @c load needs before it can read a row. The blob is this file's own memory image
 *  rather than a format: no magic, no version, and nothing but the stand-in reads it back. */
struct dense_checkpoint_prefix_t {

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
};

std::size_t stored_row_bytes(std::size_t dimensions) { return sizeof(usearch_key_t) + dimensions * sizeof(double); }

/** The whole blob's extent, empty when that many rows could not be addressed. */
std::optional<std::size_t> serialized_bytes(usearch_dense_index_handle_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(dense_checkpoint_prefix_t), rows, stored_row_bytes(index.options.dimensions));
}

void write_into(usearch_dense_index_handle_t const& index, std::size_t rows, bytes_span_t target) {
    dense_checkpoint_prefix_t prefix {};
    prefix.dimensions = index.options.dimensions;
    prefix.quantization = index.options.quantization;
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.metric = index.options.metric;
    write_pod(target, prefix);

    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        dense_entry_t const& entry = index.entries[row];
        std::size_t const payload_bytes = entry.payload.size() * sizeof(double);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(payload_bytes),
                                                            bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** Reads the prefix and checks the blob is long enough for the rows it claims. */
bool parse_prefix(dense_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(dense_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<dense_checkpoint_prefix_t>(blob);
    return fits_blob(blob, sizeof(dense_checkpoint_prefix_t), prefix.entries, stored_row_bytes(prefix.dimensions),
                     outcome);
}

/** @brief Describes a checkpoint, answering whether its metric is one we know.
 *
 *  Narrower than @c adopt: this fills only what a checkpoint froze, where that restores a handle's
 *  whole configuration including the bounds its mutable state carries.
 */
bool describe(usearch_dense_metadata_t& metadata, dense_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    metadata.dimensions = prefix.dimensions;
    metadata.metric = prefix.metric;
    metadata.connectivity = prefix.connectivity;
    metadata.quantization = prefix.quantization;
    return true;
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we know. The bound
 *  on entries per key is the loader's to choose, since no checkpoint records it. */
bool adopt(dense_index_options_t& options, dense_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.dimensions = prefix.dimensions;
    options.metric = prefix.metric;
    options.connectivity = prefix.connectivity;
    options.quantization = prefix.quantization;
    return true;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
bool read_from(usearch_dense_index_handle_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    dense_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return false;
    std::size_t const rows = prefix.entries;
    std::size_t const dimensions = prefix.dimensions;
    index.reset();
    if (!index.accommodate(rows)) {
        propagate_error(usearch_status_out_of_memory_k, outcome, "The checkpoint's rows could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(usearch_status_unknown_kind_k, outcome, "The checkpoint names a metric this build lacks");
        return false;
    }

    bytes_view_t cursor = blob.subspan(sizeof(dense_checkpoint_prefix_t));
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

/** A new handle holding the checkpoint in @p blob, or null once the outcome records why not. */
usearch_dense_index_handle_t* open_checkpoint(bytes_view_t blob, usearch_outcome_t* outcome) {
    usearch_dense_index_handle_t* const index = new (std::nothrow) usearch_dense_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return nullptr;
    if (read_from(*index, blob, outcome)) return index;
    delete index;
    return nullptr;
}

#pragma endregion Serialization

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

void usearch_dense_init_ram(usearch_dense_init_ram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_if(!known_metric(arguments->metric), usearch_status_unknown_kind_k,
                  "The stand-in does not implement that metric", outcome))
        return;
    usearch_dense_index_handle_t* const index = new (std::nothrow) usearch_dense_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->metric, arguments->quantization, arguments->dimensions,
                      arguments->connectivity ? arguments->connectivity : us::connectivity_16_k,
                      arguments->values_per_key};
    *arguments->index = index;
}

void usearch_dense_init_vram(usearch_dense_init_vram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends");
}

void usearch_dense_load_ram(usearch_dense_load_ram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_dense_index_handle_t* const index = open_checkpoint(blob, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
    *arguments->index = index;
}

void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends");
}

void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    usearch_dense_index_handle_t* const index = open_checkpoint(as_readable_bytes(arguments->buffer), outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
    *arguments->index = index;
}

void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends");
}

void usearch_dense_view_ram(usearch_dense_view_ram_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Rows are still copied out of the mapping, so only the view's refusals are mirrored, not its zero cost.
    usearch_dense_index_handle_t* const index = open_checkpoint(served.view(), outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_dense_view_nvme(usearch_dense_view_nvme_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "NVMe checkpoints carry no keys and no shard wrapper yet");
}

void usearch_dense_free(usearch_dense_index_t handle) { delete handle; }

void usearch_dense_reserve(usearch_dense_reserve_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    // Sizing the arena moves published rows, so no reader may be walking them.
    std::unique_lock writers(index.mutex);
    reject_if(!index.accommodate(arguments->capacity), usearch_status_out_of_memory_k,
              "The capacity could not be reserved", outcome);
}

void usearch_dense_clear(usearch_dense_clear_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    index.reset();
}

void usearch_dense_describe(usearch_dense_describe_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->description = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    dense_index_options_t const& options = index.options;
    *arguments->description = {
        {options.metric, options.quantization, options.dimensions, options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        index.entries.capacity() * stored_row_bytes(options.dimensions),
        "serial",
    };
}

void usearch_dense_add(usearch_dense_add_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    if (reject_unfrozen(index, outcome)) return;
    std::size_t const count = arguments->count;
    usearch_rows_view_t const& vectors = arguments->vectors;
    if (reject_view(vectors, count, index, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;

    // Sizing precedes the first append, so a batch that cannot be accommodated adds nothing.
    std::unique_lock writers(index.mutex);
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(vectors.scalar, dimensions);
    std::vector<double> payload(dimensions);
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        read_row(row_at<std::byte>(vectors.data, row, vectors.stride, row_bytes), vectors.scalar, payload);
        index.append(key_at(arguments->keys, row), payload);
        ++published;
        if (!keep_going(arguments->progress, arguments->progress_state, published, count)) break;
    }
    // The batch is all-or-nothing, a cancellation included, so a stopped run unwinds to the
    // watermark rather than leaving a prefix published.
    if (published != count) {
        index.unwind(watermark);
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the batch");
    }
    if (arguments->stats) {
        arguments->stats->visited_members = published;
        arguments->stats->computed_distances = published * index.entries.size();
        arguments->stats->restarts = 1;
    }
}

void usearch_dense_refine(usearch_dense_refine_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    // Cancelling is not an error here, the header having made a stopped sweep merely less improved.
    for (std::size_t row = 0; row != rows; ++row)
        if (!keep_going(arguments->progress, arguments->progress_state, row + 1, rows)) break;
    if (arguments->stats) arguments->stats->visited_members = rows;
}

void usearch_dense_health(usearch_dense_health_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->reachability = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    // An exact scan reaches every entry it holds in one level.
    std::size_t const entries = index.entries.size();
    *arguments->reachability = {entries, entries, entries ? 1u : 0u};
}

void usearch_dense_search(usearch_dense_search_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_unfrozen(index, outcome)) return;
    std::size_t const count = arguments->count;
    if (reject_view(arguments->queries, count, index, outcome)) return;
    if (reject_matches(arguments->matches, count, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_rows(
        index, arguments->queries, count, arguments->matches, arguments->expansion_factor, arguments->stats,
        arguments->predicate, arguments->predicate_state, arguments->progress, arguments->progress_state);
    reject_cancelled(answered, count, outcome);
}

void usearch_dense_count(usearch_dense_count_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->entries = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    *arguments->entries = index.by_key.count(keyed(arguments->key));
}

void usearch_dense_get(usearch_dense_get_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->vectors = nullptr;
    if (arguments->written) *arguments->written = 0;
    if (arguments->stride) *arguments->stride = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    usearch_scalar_t const scalar = arguments->scalar;
    if (reject_scalar(scalar, outcome)) return;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const rows = rows_under(index, arguments->key);
    if (rows.empty()) return;
    std::size_t const row_bytes = row_extent(scalar, index.options.dimensions);
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, nullptr, outcome);
    if (!arena) return;
    std::unique_lock const held {arena->busy, std::adopt_lock};
    std::byte* const landing = arena->memory.allocate(rows.size() * row_bytes, default_alignment_k);
    if (!landing) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    for (std::size_t found = 0; found != rows.size(); ++found)
        write_row(row_at<std::byte>(landing, found, row_bytes, row_bytes), scalar, index.entries[rows[found]].payload);
    *arguments->vectors = landing;
    if (arguments->written) *arguments->written = rows.size();
    if (arguments->stride) *arguments->stride = row_bytes;
}

void usearch_dense_merge(usearch_dense_merge_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_dense_index_t* const donors = arguments->donors;
    std::size_t merged = 0;
    bool accommodated = true;
    while (accommodated && merged != donors_count) {
        usearch_dense_index_handle_t& donor = *donors[merged];
        std::unique_lock writers(index.mutex);
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (dense_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        ++merged;
        if (!keep_going(arguments->progress, arguments->progress_state, merged, donors_count)) break;
    }
    // Only an absorbed donor is consumed, and its slot is nulled as it goes, so a stopped merge
    // leaves the rest live handles the caller still owns and must free.
    for (std::size_t position = 0; position != merged; ++position) {
        delete donors[position];
        donors[position] = nullptr;
    }
    if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The donor's rows could not be accommodated", outcome))
        return;
    reject_cancelled(merged, donors_count, outcome);
}

void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_dense_cursor_handle_t* const cursor = new (std::nothrow)
        usearch_dense_cursor_handle_t {&index, 0, index.entries.size(), {}};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    *arguments->cursor = cursor;
}

void usearch_dense_cursor_next(usearch_dense_cursor_next_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->written = 0;
    usearch_dense_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_dense_index_handle_t& index = *arguments->index;
    usearch_keys_span_t const& keys = arguments->keys;
    usearch_rows_span_t const& vectors = arguments->vectors;
    if (reject_keys_span(keys, outcome)) return;
    if (vectors.data && reject_span(vectors, index.options.dimensions, keys.capacity, outcome)) return;
    if (reject_if(cursor.busy.test_and_set(std::memory_order_acquire), usearch_status_contended_k,
                  "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    std::size_t const written = std::min(keys.capacity, cursor.limit - cursor.position);
    std::size_t const row_bytes = row_extent(vectors.scalar, index.options.dimensions);
    for (std::size_t row = 0; row != written; ++row) {
        dense_entry_t const& entry = index.entries[cursor.position + row];
        row_at<usearch_key_t>(keys.data, row, keys.stride, 1).front() = entry.key;
        if (vectors.data)
            write_row(row_at<std::byte>(vectors.data, row, vectors.stride, row_bytes), vectors.scalar, entry.payload);
    }
    cursor.position += written;
    *arguments->written = written;
    cursor.busy.clear(std::memory_order_release);
}

void usearch_dense_cursor_free(usearch_dense_cursor_t handle) { delete handle; }

void usearch_dense_search_stream_init(usearch_dense_search_stream_init_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->stream = nullptr;
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_unfrozen(index, outcome)) return;
    usearch_rows_view_t const& query = arguments->query;
    if (reject_view(query, 1, index, outcome)) return;
    usearch_dense_search_stream_handle_t* const stream = new (std::nothrow) usearch_dense_search_stream_handle_t();
    if (reject_if(!stream, usearch_status_out_of_memory_k, "The search stream could not be allocated", outcome)) return;
    std::size_t const dimensions = index.options.dimensions;
    std::vector<double> query_row(dimensions);
    read_row(row_at<std::byte>(query.data, 0, query.stride, row_extent(query.scalar, dimensions)), query.scalar,
             query_row);
    {
        std::shared_lock readers(index.mutex);
        rank<dense_payload_t>(
            index.entries, admitting(nullptr, nullptr),
            [&](dense_payload_t const& payload) { return distance(index.options.metric, query_row, payload); },
            stream->ranked);
    }
    stream->opened_on = &index;
    stream->position = 0;
    *arguments->stream = stream;
}

void usearch_dense_search_stream_next(usearch_dense_search_stream_next_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_search_stream_handle_t& stream = *arguments->stream;
    if (reject_if(arguments->index != stream.opened_on, usearch_status_invalid_argument_k,
                  "The stream was opened on another index", outcome))
        return;
    usearch_matches_span_t const& matches = arguments->matches;
    if (reject_matches(matches, 1, outcome)) return;
    if (reject_if(stream.busy.test_and_set(std::memory_order_acquire), usearch_status_contended_k,
                  "Another pull is still running on the stream", outcome))
        return;
    std::size_t const written = std::min(matches.wanted, stream.ranked.size() - stream.position);
    for (std::size_t match = 0; match != written; ++match) {
        matches.keys[match] = stream.ranked[stream.position + match].key;
        matches.scores[match] = stream.ranked[stream.position + match].distance;
    }
    stream.position += written;
    matches.counts[0] = written;
    stream.busy.clear(std::memory_order_release);
}

void usearch_dense_search_stream_free(usearch_dense_search_stream_t handle) { delete handle; }

void usearch_dense_save(usearch_dense_save_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    // Saving is a reader: it demands only that nothing move or erase rows under it.
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return;
    // The blob is one chunk, so the handler's only say is whether it is written at all.
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(arguments->path, blob, outcome)) return;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
}

void usearch_dense_save_buffer(usearch_dense_save_buffer_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->bytes = nullptr;
    *arguments->length = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, nullptr, outcome);
    if (!arena) return;
    std::unique_lock const held {arena->busy, std::adopt_lock};
    std::byte* const landing = arena->memory.allocate(*length, default_alignment_k);
    if (!landing) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    write_into(index, rows, bytes_span_t {landing, *length});
    *arguments->bytes = reinterpret_cast<std::uint8_t*>(landing);
    *arguments->length = *length;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
}

void usearch_dense_metadata_read(usearch_dense_metadata_read_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    dense_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    reject_if(!describe(*arguments->metadata, prefix), usearch_status_unknown_kind_k,
              "The checkpoint names a metric this build lacks", outcome);
}

void usearch_dense_metadata_read_buffer(usearch_dense_metadata_read_buffer_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    dense_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, as_readable_bytes(arguments->buffer), outcome)) return;
    reject_if(!describe(*arguments->metadata, prefix), usearch_status_unknown_kind_k,
              "The checkpoint names a metric this build lacks", outcome);
}

void usearch_dense_multiply(usearch_dense_multiply_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    *arguments->keys = nullptr;
    *arguments->scores = nullptr;
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_status_config_mismatch_k,
                  "A product is a logit, so the kind must be inner product", outcome))
        return;
    std::size_t const rows_count = arguments->count;
    usearch_rows_view_t const& left = arguments->left;
    if (reject_view(left, rows_count, index, outcome)) return;
    if (reject_if(!arguments->offsets, usearch_status_invalid_argument_k, "The offsets must name storage", outcome))
        return;

    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(left.scalar, dimensions);
    // A kept row holds the first pairs its walk scored - here, the first in storage order.
    std::size_t const kept = arguments->wanted ? std::min(arguments->wanted, indexed) : indexed;

    // Every row is scored before anything lands, so a cancelled call leaves the arena untouched.
    std::vector<double> left_row(dimensions);
    std::vector<usearch_distance_t> products(rows_count * kept);
    std::size_t multiplied = 0;
    for (std::size_t row = 0; row != rows_count; ++row) {
        read_row(row_at<std::byte>(left.data, row, left.stride, row_bytes), left.scalar, left_row);
        for (std::size_t column = 0; column != kept; ++column)
            products[row * kept + column] = dot_product(left_row, index.entries[column].payload);
        if (arguments->stats) arguments->stats->computed_distances += indexed;
        ++multiplied;
        if (!keep_going(arguments->progress, arguments->progress_state, multiplied, rows_count)) break;
    }
    if (reject_cancelled(multiplied, rows_count, outcome)) return;

    std::size_t const pairs = rows_count * kept;
    for (std::size_t row = 0; row <= rows_count; ++row) arguments->offsets[row] = row * kept;
    if (!pairs) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, left.data, outcome);
    if (!arena) return;
    std::unique_lock const held {arena->busy, std::adopt_lock};
    std::byte* const landing = arena->memory.allocate(pairs * (sizeof(usearch_key_t) + sizeof(usearch_distance_t)),
                                                      default_alignment_k);
    if (!landing) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    usearch_key_t* const keys = reinterpret_cast<usearch_key_t*>(landing);
    usearch_distance_t* const scores = reinterpret_cast<usearch_distance_t*>(landing + pairs * sizeof(usearch_key_t));
    for (std::size_t pair = 0; pair != pairs; ++pair)
        keys[pair] = index.entries[pair % kept].key, scores[pair] = products[pair];
    *arguments->keys = keys;
    *arguments->scores = scores;
    if (arguments->written) *arguments->written = pairs;
    if (arguments->required) *arguments->required = rows_count * indexed;
}

void usearch_dense_attend(usearch_dense_attend_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_status_config_mismatch_k,
                  "A softmax weighs logits, so the kind must be inner product", outcome))
        return;

    std::size_t const queries_count = arguments->count, head_count = std::max<std::size_t>(arguments->head_count, 1);
    std::size_t const dimensions = index.options.dimensions;
    usearch_rows_view_t const& queries = arguments->queries;
    usearch_rows_span_t const& outputs = arguments->outputs;
    if (reject_view(queries, queries_count, index, outcome)) return;
    if (reject_span(outputs, dimensions, queries_count, outcome)) return;
    if (reject_if(!arguments->value_lookup || arguments->value_row, usearch_status_invalid_argument_k,
                  "A RAM handle takes only a value lookup, and a GPU handle only a value row", outcome))
        return;
    if (reject_if(queries_count % head_count, usearch_status_shape_mismatch_k, "The heads must divide the rows",
                  outcome))
        return;

    // Exact attention weighs every entry, so every value row resolves once, up front.
    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::vector<usearch_key_t> keys(indexed);
    std::vector<void const*> value_rows(indexed);
    for (std::size_t entry = 0; entry != indexed; ++entry) keys[entry] = index.entries[entry].key;
    if (indexed) arguments->value_lookup(arguments->value_lookup_state, keys.data(), indexed, value_rows.data());
    if (reject_if(std::find(value_rows.begin(), value_rows.end(), nullptr) != value_rows.end(),
                  usearch_status_shape_mismatch_k, "A scored key resolved to no value row", outcome))
        return;

    std::size_t const query_bytes = row_extent(queries.scalar, dimensions);
    std::size_t const value_bytes = row_extent(index.options.quantization, dimensions);
    std::size_t const output_bytes = row_extent(outputs.scalar, dimensions);
    double const scale = arguments->score_scale > 0 ? arguments->score_scale
                                                    : 1.0 / std::sqrt(std::max<double>(dimensions, 1));
    std::vector<double> query_row(dimensions), value(dimensions), accumulated(dimensions), logits(indexed);
    std::size_t const tokens = queries_count / head_count;
    std::size_t answered = 0;
    for (std::size_t token = 0; token != tokens; ++token) {
        for (std::size_t row = token * head_count; row != (token + 1) * head_count; ++row) {
            read_row(row_at<std::byte>(queries.data, row, queries.stride, query_bytes), queries.scalar, query_row);
            double maximum = std::numeric_limits<double>::lowest(), partition = 0;
            for (std::size_t entry = 0; entry != indexed; ++entry) {
                logits[entry] = scale * dot_product(query_row, index.entries[entry].payload);
                maximum = std::max(maximum, logits[entry]);
            }
            std::fill(accumulated.begin(), accumulated.end(), 0.0);
            for (std::size_t entry = 0; entry != indexed; ++entry) {
                double const weight = std::exp(logits[entry] - maximum);
                read_row(row_at<std::byte>(value_rows[entry], 0, 0, value_bytes), index.options.quantization, value);
                partition += weight;
                for (std::size_t position = 0; position != dimensions; ++position)
                    accumulated[position] += weight * value[position];
            }
            for (double& position : accumulated) position = partition > 0 ? position / partition : 0.0;
            write_row(row_at<std::byte>(outputs.data, row, outputs.stride, output_bytes), outputs.scalar, accumulated);
            if (arguments->carry) arguments->carry[row] = {static_cast<float>(maximum), static_cast<float>(partition)};
            if (stats) stats->computed_distances += indexed;
        }
        ++answered;
        if (!keep_going(arguments->progress, arguments->progress_state, answered, tokens)) break;
    }
    reject_cancelled(answered, tokens, outcome);
}

void usearch_dense_maxsim(usearch_dense_maxsim_t* arguments) {
    pending_outcome_t outcome(arguments->outcome);
    usearch_dense_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_status_config_mismatch_k,
                  "Late interaction sums maxima over logits, so the kind must be inner product", outcome))
        return;
    std::size_t const queries_count = arguments->count, rows_per_query = arguments->rows_per_query;
    usearch_rows_view_t const& queries = arguments->queries;
    usearch_matches_span_t const& matches = arguments->matches;
    if (reject_if(!rows_per_query, usearch_status_shape_mismatch_k, "A query must hold at least one row", outcome))
        return;
    if (reject_view(queries, queries_count * rows_per_query, index, outcome)) return;
    if (reject_matches(matches, queries_count, outcome)) return;
    usearch_scalar_t const scalar_kind = queries.scalar;
    std::size_t const wanted = matches.wanted;

    std::shared_lock readers(index.mutex);

    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(scalar_kind, dimensions);
    std::span<std::size_t> const counts_column {matches.counts, queries_count};
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
            read_row(row_at<std::byte>(queries.data, query * rows_per_query + row, queries.stride, row_bytes),
                     scalar_kind, std::span<double>(query_rows).subspan(row * dimensions, dimensions));

        ranked.clear();
        for (std::size_t document = 0; document != documents.size(); ++document) {
            double summed = 0;
            for (std::size_t row = 0; row != rows_per_query; ++row) {
                std::span<double const> const query_row {query_rows.data() + row * dimensions, dimensions};
                double best = std::numeric_limits<double>::lowest();
                for (std::size_t stored : token_rows[document])
                    best = std::max(best, dot_product(query_row, index.entries[stored].payload));
                summed += best;
            }
            ranked.push_back({summed, documents[document]});
        }
        // Summed maxima are a similarity, so the best key is the largest rather than the nearest.
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](match_t const& first, match_t const& second) { return first.distance > second.distance; });

        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(matches.keys, query, matches.keys_stride,
                                                                        wanted);
        std::span<usearch_distance_t> const scores_row = row_at<usearch_distance_t>(matches.scores, query,
                                                                                    matches.scores_stride, wanted);
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
        if (!keep_going(arguments->progress, arguments->progress_state, answered, queries_count)) break;
    }
    reject_cancelled(answered, queries_count, outcome);
}

} // extern "C"
