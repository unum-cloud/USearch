/**
 *  @file src_stub/dense.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_dense_* C ABI, letting a language binding link
 *      and run its suite in seconds instead of building the engine.
 *
 *  Every SDK in this repository talks to the same C ABI, and most of what breaks in one is the
 *  binding rather than the engine: a leaked handle, a stride counted in scalars instead of bytes,
 *  an unread outcome, an output read past the count the call wrote. Finding those should not cost a
 *  build of the instantiation grid, so this file answers the dense verbs with the C++ standard
 *  library alone - no submodules, no SIMD, no CUDA - over the store and locks of @c lib.hpp. A
 *  binding suite runs unchanged against this library or the engine's.
 *
 *  Search here is an exact scan, so recall is trivially perfect and ties fall in insertion
 *  order. What this proves is the binding: handle lifetime, strided input, arena output, the
 *  callback ABI, the truncation and watermark contracts, and outcome propagation. It is not an
 *  engine and never becomes one.
 *
 *  @section dense_stub_blind_spots What It Cannot Tell You
 *
 *  A test double this lossy invites false confidence, so its limits are worth stating plainly.
 *  Entries are stored as @c double whatever the @c quantization, so quantization loss is invisible
 *  and an @c i8 index round-trips exactly, though the quantization itself is refused as the engine
 *  refuses it: @c f32, @c f64 and @c i32 are never stored, and a dtype the build's
 *  @c USEARCH_DENSE_DTYPES leaves out is unsupported. Every scalar kind the engine stores is
 *  decoded, the mini-floats and the packed nibble and bit kinds among them, so a row's byte extent
 *  is computed here exactly as the engine computes it and a binding sizing rows wrongly for a
 *  narrow float or a sub-byte kind fails on the shape instead of passing quietly, while @c i32, the
 *  geospatial coordinate, is refused as the engine refuses it. The metric and the quantization are
 *  never checked against each other, so the bit-set kinds read whatever rows they are handed where
 *  the engine admits @c u1x8 alone. @c values_per_key bounds nothing - a key holds as many entries
 *  as it is handed, the header naming no failure for exceeding it. A walk and a scan rank alike,
 *  both being exact here, and both ask the predicate about every entry a quad at a time, where the
 *  engine's scan asks only about entries closer than the worst match a query already holds.
 *  Refinement keeps no edges to re-select, so its @c changed is a constant zero rather than a
 *  settling signal, and a caller must not read it as one. Residency and huge pages are inert, a
 *  view copies its checkpoint rather than mapping it, and only the RAM verbs answer - every VRAM
 *  and NVMe lifecycle verb refuses. Traversal statistics are plausible fiction: shaped
 *  correctly, invented numerically, so assert that they move, never what they equal. A checkpoint
 *  is a binary blob of this file's own memory image - not SafeGraphs, not portable across builds,
 *  and readable by nothing but the stand-in that wrote it.
 *
 *  Keeping no edges, the stand-in links every one of its entries to its exact nearest at the frozen
 *  connectivity, the lists a perfect graph would hold where the engine's are the ones it stored. A
 *  cursor walks those lists breadth-first from its start keys, a split grows its parts over them,
 *  each receiver keeping the source's storage order, and a sample walks them as the library's plan
 *  walks its graph, placing its draws as the library does.
 */
#include <cmath> // `std::sqrt`, `std::exp`

#include <algorithm>     // `std::stable_sort`, `std::min`, `std::max`, `std::find`
#include <bit>           // `std::has_single_bit` - the connectivity init accepts
#include <limits>        // `std::numeric_limits`
#include <memory>        // `std::unique_ptr` - a finished refinement's handle
#include <optional>      // `std::optional`
#include <span>          // `std::span`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include "usearch/types.h"
#include "usearch/dense_index.h"

#include "lib.hpp"

/** How many storage dtypes the engine's build keeps, from the front of its ladder; CMake passes the
 *  engine's own count, so only builds that bypass CMake take this default. */
#if !defined(USEARCH_DENSE_DTYPES)
#define USEARCH_DENSE_DTYPES 4
#endif

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Storage

using dense_payload_t = std::vector<double>;
using dense_entry_t = entry<dense_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<dense_entry_t>, "reallocation must move entries, never copy them");

/** What a handle froze at construction, or adopted from a checkpoint. */
struct dense_index_options_t {

    /** What entries are compared by. */
    usearch_dense_metric_t metric;

    /** The scalar kind vectors were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;

    /** The frozen row width. */
    std::size_t dimensions;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Entries that may share one key, which late interaction needs above one. */
    std::size_t values_per_key;
};

using dense_index_t = handle<dense_payload_t, dense_index_options_t>;

using dense_cursor_t = graph_cursor<dense_index_t>;

struct dense_search_stream_t {

    /** The handle the stream was opened on, only ever compared with the one each pull passes. */
    dense_index_t const* opened_on;

    /** Every scored match, nearest first - a snapshot, so it needs no further locking. */
    std::vector<match_t> ranked;

    /** The next match to hand out. */
    std::size_t position;

    /** Bit zero is set while a pull runs, so an overlapping pull is refused instead of racing. */
    u32_t pulling = 0;
};

/** Rejects a kind naming no scalar, @c i32, which only geospatial points take, and a sub-byte kind
 *  whose row of @p dimensions ends inside a byte, as every rows view promises whole bytes. */
static bool reject_vector_scalar(usearch_scalar_t kind, std::size_t dimensions, usearch_outcome_t* outcome) {
    if (reject_scalar(kind, outcome)) return true;
    if (reject_if(kind == usearch_scalar_i32_k, usearch_status_unknown_kind_k, "Vectors hold no i32", outcome))
        return true;
    return reject_if(dimensions % dimensions_per_value(kind), usearch_status_shape_mismatch_k,
                     "A sub-byte row must fill whole bytes", outcome);
}

/**
 *  @brief Checks a run of input rows against the frozen geometry, answering whether to refuse it.
 *
 *  The view's width must be the index's own, and a stride narrower than it is where an over-read
 *  becomes an error. A zero stride repeats the first row.
 */
static bool reject_view(usearch_rows_view_t const& rows, std::size_t count, dense_index_t const& index,
                        usearch_outcome_t* outcome) {
    if (reject_if(!rows.data && count, usearch_status_invalid_argument_k, "The rows must name storage", outcome))
        return true;
    if (reject_if(count && rows.columns != index.options.dimensions, usearch_status_shape_mismatch_k,
                  "The rows' width differs from the frozen geometry", outcome))
        return true;
    if (reject_vector_scalar(rows.scalar, index.options.dimensions, outcome)) return true;
    std::size_t const natural = row_extent(rows.scalar, index.options.dimensions);
    return reject_if(rows.stride && rows.stride < natural, usearch_status_unsupported_layout_k,
                     "The stride is shorter than one row", outcome);
}

/** Refuses a quantization the engine never stores - f32, f64, i32 or no scalar at all - and one
 *  its build's @c USEARCH_DENSE_DTYPES leaves off the front of the ladder, as the engine does. */
static bool reject_unstored(usearch_scalar_t quantization, usearch_outcome_t* outcome) {
    static constexpr usearch_scalar_t ladder_k[] = {usearch_scalar_bf16_k, usearch_scalar_u1x8_k, usearch_scalar_i8_k,
                                                    usearch_scalar_e4m3_k, usearch_scalar_f16_k,  usearch_scalar_u8_k,
                                                    usearch_scalar_e5m2_k, usearch_scalar_i4x2_k, usearch_scalar_u4x2_k,
                                                    usearch_scalar_e3m2_k, usearch_scalar_e2m3_k};
    for (std::size_t rung = 0; rung != std::size(ladder_k); ++rung)
        if (ladder_k[rung] == quantization)
            return reject_if(rung >= USEARCH_DENSE_DTYPES, usearch_status_unsupported_k,
                             "This build's USEARCH_DENSE_DTYPES leaves the quantization out", outcome);
    return reject_if(true, usearch_status_unknown_kind_k, "Unsupported quantization", outcome);
}

/** Whether a receiver frozen at @p receiver takes a part of an index frozen at @p index: the same
 *  metric, quantization, width and connectivity. */
static bool same_kind(dense_index_options_t const& index, dense_index_options_t const& receiver) {
    return index.metric == receiver.metric && index.quantization == receiver.quantization &&
           index.dimensions == receiver.dimensions && index.connectivity == receiver.connectivity;
}

#pragma endregion Storage

#pragma region Scoring

/** Scores one pair under the frozen kind. @c usearch_dense_init_ram refuses a kind this
 *  build does not know, so every enumerator reaching here is one of the five. */
static double distance(usearch_dense_metric_t metric, std::span<double const> first, std::span<double const> second) {
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
static bool known_metric(usearch_dense_metric_t metric) {
    return metric == usearch_dense_metric_inner_k || metric == usearch_dense_metric_angular_k ||
           metric == usearch_dense_metric_euclidean_k || metric == usearch_dense_metric_hamming_k ||
           metric == usearch_dense_metric_jaccard_k;
}

static double dot_product(std::span<double const> first, std::span<double const> second) {
    double dot = 0;
    for (std::size_t position = 0; position != first.size(); ++position) dot += first[position] * second[position];
    return dot;
}

/**
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
static std::size_t search_queries(dense_index_t& index, usearch_rows_view_t const& queries, std::size_t count,
                                  matches_outputs_t const& outputs, std::size_t expansion_factor,
                                  usearch_traversal_t traversal, usearch_traversal_stats_t* stats,
                                  usearch_predicate_t predicate, usearch_progress_t progress) {
    std::size_t const entries = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const query_bytes = row_extent(queries.scalar, dimensions);
    // A scan scores every entry, as the engine's does; the expansion factor moves a walk's work.
    bool const scanning = traversal == usearch_traversal_scan_k;
    if (stats) stats->computed_distances += scanning ? entries * count : expansion_factor * count;

    std::vector<double> query_row(dimensions);
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != count; ++query) {
        read_row(row_at<std::byte>(queries.data, query, queries.stride, query_bytes), queries.scalar, query_row);
        rank<dense_payload_t>(
            std::span<dense_entry_t const>(index.entries).first(entries), predicate,
            [&](dense_payload_t const& payload) { return distance(index.options.metric, query_row, payload); }, ranked);
        outputs.write(query, ranked);
        if (stats) {
            stats->visited_members += entries;
            if (!scanning) stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(progress, query + 1, count)) return query + 1;
    }
    return count;
}

#pragma endregion Scoring

#pragma region Serialization

/** What @c load needs before it can read an entry. The blob is this file's own memory image
 *  rather than a format: no magic, no version, and nothing but the stand-in reads it back. */
struct dense_checkpoint_prefix_t {

    /** The frozen row width. */
    std::size_t dimensions;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many entries follow the prefix. */
    std::size_t entries;

    /** What entries are compared by. */
    usearch_dense_metric_t metric;

    /** The scalar kind the vectors were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;
};

static std::size_t stored_entry_bytes(std::size_t dimensions) {
    return sizeof(usearch_key_t) + dimensions * sizeof(double);
}

/** The whole blob's extent, empty when that many entries could not be addressed. */
static std::optional<std::size_t> serialized_bytes(dense_index_t const& index, std::size_t entries) {
    return blob_bytes(sizeof(dense_checkpoint_prefix_t), entries, stored_entry_bytes(index.options.dimensions));
}

static void write_into(dense_index_t const& index, std::size_t entries, bytes_span_t target) {
    dense_checkpoint_prefix_t prefix {};
    prefix.dimensions = index.options.dimensions;
    prefix.quantization = index.options.quantization;
    prefix.connectivity = index.options.connectivity;
    prefix.entries = entries;
    prefix.metric = index.options.metric;
    write_pod(target, prefix);

    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t position = 0; position != entries; ++position) {
        dense_entry_t const& entry = index.entries[position];
        std::size_t const payload_bytes = entry.payload.size() * sizeof(double);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(payload_bytes),
                                                            bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** Reads the prefix and checks the blob is long enough for the entries it claims. */
static bool parse_prefix(dense_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(dense_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<dense_checkpoint_prefix_t>(blob);
    return fits_blob(blob, sizeof(dense_checkpoint_prefix_t), prefix.entries, stored_entry_bytes(prefix.dimensions),
                     outcome);
}

/** @brief Describes a checkpoint, answering whether its metric is one we know.
 *
 *  Narrower than @c adopt: this fills only what a checkpoint froze, where that restores a handle's
 *  whole configuration including the bounds its mutable state carries.
 */
static bool describe(usearch_dense_metadata_t& metadata, dense_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    metadata.dimensions = prefix.dimensions;
    metadata.metric = prefix.metric;
    metadata.connectivity = prefix.connectivity;
    metadata.quantization = prefix.quantization;
    return true;
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we know. The bound
 *  on entries per key is the loader's to choose, since no checkpoint records it. */
static bool adopt(dense_index_options_t& options, dense_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.dimensions = prefix.dimensions;
    options.metric = prefix.metric;
    options.connectivity = prefix.connectivity;
    options.quantization = prefix.quantization;
    return true;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
static bool read_from(dense_index_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    dense_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome) || reject_unstored(prefix.quantization, outcome)) return false;
    std::size_t const entries = prefix.entries;
    std::size_t const dimensions = prefix.dimensions;
    index.reset();
    if (!index.accommodate(entries)) {
        propagate_error(usearch_status_out_of_memory_k, outcome, "The checkpoint's entries could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(usearch_status_unknown_kind_k, outcome, "The checkpoint names a metric this build lacks");
        return false;
    }

    bytes_view_t cursor = blob.subspan(sizeof(dense_checkpoint_prefix_t));
    std::vector<double> payload(dimensions);
    for (std::size_t position = 0; position != entries; ++position) {
        usearch_key_t const key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        [[maybe_unused]] status_t const copied = copy_bytes(bytes_of(payload.data(), payload.size()),
                                                            cursor.first(dimensions * sizeof(double)));
        cursor = cursor.subspan(dimensions * sizeof(double));
        index.append(key, payload);
    }
    return true;
}

/** Writes @p shard's whole image into @p blob; false once @p outcome records why not. */
static bool serialize_shard(dense_index_t const& shard, std::vector<std::byte>& blob, usearch_outcome_t* outcome) {
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
static index_type_* open_checkpoint(bytes_view_t blob, usearch_outcome_t* outcome) {
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

struct usearch_dense_index_handle_t : dense_index_t {};
struct usearch_dense_cursor_handle_t : dense_cursor_t {};
struct usearch_dense_search_stream_handle_t : dense_search_stream_t {};

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
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    // Init freezes the geometry, so it refuses one the engine cannot store: no width, a packed
    // width leaving a storage value partly filled, or a connectivity off the powers of two.
    if (reject_if(!known_metric(arguments->metric), usearch_status_unknown_kind_k,
                  "The stand-in does not implement that metric", outcome) ||
        reject_unstored(arguments->quantization, outcome) ||
        reject_if(!arguments->dimensions || arguments->dimensions % dimensions_per_value(arguments->quantization) ||
                      (arguments->connectivity && !std::has_single_bit(arguments->connectivity)),
                  usearch_status_config_mismatch_k, "The geometry cannot be stored", outcome))
        return;
    usearch_dense_index_handle_t* const index = new (std::nothrow) usearch_dense_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->metric, arguments->quantization, arguments->dimensions,
                      arguments->connectivity ? arguments->connectivity : us::utilities::connectivity_16_k,
                      arguments->values_per_key};
    index->shard_capacity = arguments->ram.shard_capacity;
    *arguments->index = index;
}

void usearch_dense_init_vram(usearch_dense_init_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_dense_load_ram(usearch_dense_load_ram_t* arguments) {
    if (through_executor<usearch_dense_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_dense_index_handle_t* const index = open_parts<usearch_dense_index_handle_t>(
        blob, arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_dense_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, 1, 1);
    *arguments->index = index;
}

void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments) {
    if (through_executor<usearch_dense_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments) {
    load_ram_buffer<usearch_dense_load_ram_buffer>(arguments, open_checkpoint<usearch_dense_index_handle_t>);
}

void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments) {
    if (through_executor<usearch_dense_load_vram_buffer>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_dense_view_ram(usearch_dense_view_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Entries are still copied out of the mapping, so only the view's refusals are mirrored, not
    // its zero cost.
    usearch_dense_index_handle_t* const index = open_parts<usearch_dense_index_handle_t>(
        served.view(), arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_dense_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_dense_view_nvme(usearch_dense_view_nvme_t* arguments) {
    if (through_executor<usearch_dense_view_nvme>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Serves the drives' checkpoint as the RAM view does, mirroring only the residency's refusals.
    usearch_dense_index_handle_t* const index = open_parts<usearch_dense_index_handle_t>(
        served.view(), arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_dense_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->served = std::move(served);
    index->residency = usearch_residency_nvme_k;
    *arguments->index = index;
}

void usearch_dense_free(usearch_dense_index_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_dense_reserve(usearch_dense_reserve_t* arguments) {
    if (through_executor<usearch_dense_reserve>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    // Sizing the arena moves published entries, so no reader may be walking them.
    std::unique_lock writers(index.mutex);
    reject_if(!index.accommodate(arguments->capacity), usearch_status_out_of_memory_k,
              "The capacity could not be reserved", outcome);
}

void usearch_dense_clear(usearch_dense_clear_t* arguments) {
    if (through_executor<usearch_dense_clear>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    index.reset();
}

void usearch_dense_describe(usearch_dense_describe_t* arguments) {
    if (through_executor<usearch_dense_describe>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->description = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    dense_index_options_t const& options = index.options;
    *arguments->description = {
        {options.metric, options.quantization, options.dimensions, options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        index.entries.capacity() * stored_entry_bytes(options.dimensions),
        usearch_cpu_class_serial_k,
        usearch_gpu_class_unknown_k,
        index.shards(),
        usearch_load_path_none_k,
    };
}

void usearch_dense_insert(usearch_dense_insert_t* arguments) {
    if (through_executor<usearch_dense_insert>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::size_t const count = arguments->count;
    usearch_rows_view_t const& vectors = arguments->vectors;
    if (reject_view(vectors, count, index, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;

    // Sizing precedes the first append, so a batch that cannot be accommodated adds nothing.
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(vectors.scalar, dimensions);
    std::vector<double> payload(dimensions);
    std::size_t published = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        read_row(row_at<std::byte>(vectors.data, offset, vectors.stride, row_bytes), vectors.scalar, payload);
        index.append(key_at(arguments->keys, offset), payload);
        ++published;
        // Every entry is claimed before the first report, so the handler only reports.
        keep_going(arguments->progress, published, count);
    }
    if (arguments->stats) {
        arguments->stats->visited_members = published;
        arguments->stats->computed_distances = published * index.entries.size();
        arguments->stats->restarts = 1;
    }
}

void usearch_dense_refine(usearch_dense_refine_t* arguments) {
    if (through_executor<usearch_dense_refine>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    std::size_t const entries = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    bool cancelled = false;
    for (std::size_t entry = 0; entry != entries && !cancelled; ++entry)
        cancelled = !keep_going(arguments->progress, entry + 1, entries);
    if (arguments->stats) arguments->stats->visited_members = entries;
    // An exact scan reaches every entry it holds in one level, as `usearch_dense_health` reports.
    if (arguments->report) arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

void usearch_dense_health(usearch_dense_health_t* arguments) {
    if (through_executor<usearch_dense_health>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->reachability = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    // An exact scan reaches every entry it holds in one level.
    std::size_t const entries = index.entries.size();
    *arguments->reachability = {entries, entries, entries ? 1u : 0u};
}

void usearch_dense_search(usearch_dense_search_t* arguments) {
    if (through_executor<usearch_dense_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->matches = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_view(arguments->queries, count, index, outcome)) return;
    usearch_traversal_t const traversal = arguments->traversal;
    if (reject_traversal(traversal, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries.data},
                                                      outcome);
    matches_outputs_t outputs {};
    if (!arena || !carve_ranked(*arena, count, arguments->wanted, outputs, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_queries(index, arguments->queries, count, outputs, arguments->expansion_factor,
                                                traversal, arguments->stats, arguments->predicate, arguments->progress);
    if (reject_cancelled(answered, count, outcome)) return;
    *arguments->matches = outputs.view();
}

void usearch_dense_count(usearch_dense_count_t* arguments) {
    if (through_executor<usearch_dense_count>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->entries = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    *arguments->entries = index.by_key.count(uuid_of(arguments->key));
}

void usearch_dense_get(usearch_dense_get_t* arguments) {
    if (through_executor<usearch_dense_get>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->vectors = {};
    if (arguments->written) *arguments->written = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    usearch_scalar_t const scalar = arguments->outputs_scalar;
    if (reject_vector_scalar(scalar, index.options.dimensions, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const entries = nodes_of(index, arguments->key);
    if (entries.empty()) return;
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(scalar, dimensions);
    std::byte* const vectors = carve<std::byte>(*arena, entries.size() * row_bytes, outcome);
    if (!vectors) return;
    for (std::size_t found = 0; found != entries.size(); ++found)
        write_row(row_at<std::byte>(vectors, found, row_bytes, row_bytes), scalar,
                  index.entries[entries[found]].payload);
    *arguments->vectors = {vectors, row_bytes, dimensions, scalar};
    if (arguments->written) *arguments->written = entries.size();
}

void usearch_dense_absorb(usearch_dense_absorb_t* arguments) {
    if (through_executor<usearch_dense_absorb>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_dense_index_t* const donors = arguments->donors;
    for (std::size_t position = 0; position != donors_count; ++position)
        if (reject_if(!donors[position], usearch_status_invalid_argument_k, "A donor must not be null", outcome) ||
            reject_if(!same_kind(index.options, donors[position]->options), usearch_status_config_mismatch_k,
                      "A donor of another kind or geometry is refused", outcome))
            return;
    bool accommodated = true;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    // Each donor moves in whole and is consumed at once, its slot nulled, so a failure leaves it
    // and every later one live handles the caller still owns and must free.
    for (std::size_t position = 0; accommodated && position != donors_count; ++position) {
        usearch_dense_index_handle_t& donor = *donors[position];
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (dense_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        delete std::exchange(donors[position], nullptr);
    }
    if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The donor's entries could not be accommodated",
                  outcome))
        return;
    // An exact scan keeps no edges, so the closing refine re-selects nothing; a cancel stops only
    // it.
    std::size_t const entries = index.entries.size();
    bool cancelled = false;
    for (std::size_t entry = 0; entry != entries && !cancelled; ++entry)
        cancelled = !keep_going(arguments->progress, entry + 1, entries);
    if (arguments->stats) arguments->stats->visited_members = entries;
    if (arguments->report) arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

void usearch_dense_compact(usearch_dense_compact_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    if (reject_if(arguments->count && !arguments->sources, usearch_status_invalid_argument_k,
                  "The sources must name handles", outcome))
        return;
    usearch_dense_index_handle_t& target = *arguments->target;
    if (reject_read_only(target, outcome)) return;
    std::size_t const sources_count = arguments->count;
    usearch_dense_index_t* const sources = arguments->sources;
    std::size_t merged_size = target.entries.size();
    for (std::size_t position = 0; position != sources_count; ++position) {
        if (reject_if(!sources[position], usearch_status_invalid_argument_k, "A source must not be null", outcome))
            return;
        for (std::size_t earlier = 0; earlier != position; ++earlier)
            if (reject_if(sources[earlier] == sources[position], usearch_status_config_mismatch_k,
                          "A source must not repeat", outcome))
                return;
        if (sources[position] != arguments->target) merged_size += sources[position]->entries.size();
    }
    // One flat list holds no shards, so in place nothing moves and every other source appends.
    std::unique_lock writers(target.mutex);
    if (arguments->shard_capacity) target.shard_capacity = arguments->shard_capacity;
    if (reject_if(!target.accommodate(merged_size), usearch_status_out_of_memory_k,
                  "The sources' entries could not be accommodated", outcome))
        return;
    for (std::size_t position = 0; position != sources_count; ++position) {
        if (sources[position] == arguments->target) continue;
        for (dense_entry_t const& entry : sources[position]->entries) target.append(entry.key, entry.payload);
    }
    for (std::size_t position = 0; position != sources_count; ++position)
        if (sources[position] != arguments->target) delete std::exchange(sources[position], nullptr);
    std::size_t const entries = target.entries.size();
    if (arguments->stats) arguments->stats->visited_members = entries;
    if (arguments->report) arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
}

/** The stand-in's refinement: the donors it still absorbs, and how far its sweep got. Its index
 *  admits no insertion, save or other refinement until it finishes. */
struct usearch_dense_refinement_handle_t {

    /** The handle the refinement was opened on, only ever compared with the one each call
     *  passes. */
    dense_index_t const* opened_on;

    /** The caller's donor slots, each nulled once consumed. */
    usearch_dense_index_t* donors;
    std::size_t count;

    /** Donors consumed so far, and entries the sweep visited so far. */
    std::size_t consumed;
    std::size_t swept;

    /** The receivers a split fills, the seed picking its roots, and those split off so far. */
    usearch_dense_index_t const* receivers;
    std::size_t receivers_count;
    us::utilities::u64_t seed;
    std::size_t split_receivers_count;

    /** Entries the absorbed donors moved in, each one a node whose links changed. */
    std::size_t moved = 0;
};

void usearch_dense_refinement_init(usearch_dense_refinement_init_t* arguments) {
    if (through_executor<usearch_dense_refinement_init>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->refinement = nullptr;
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    if (reject_if(arguments->count && !arguments->donors, usearch_status_invalid_argument_k,
                  "The donors must name handles", outcome))
        return;
    for (std::size_t position = 0; position != arguments->count; ++position) {
        if (reject_if(!arguments->donors[position], usearch_status_invalid_argument_k, "A donor must not be null",
                      outcome) ||
            reject_if(arguments->donors[position] == arguments->index, usearch_status_config_mismatch_k,
                      "An index cannot absorb itself", outcome) ||
            reject_if(!same_kind(index.options, arguments->donors[position]->options), usearch_status_config_mismatch_k,
                      "A donor of another kind or geometry is refused", outcome))
            return;
    }
    if (reject_receivers(*arguments, outcome)) return;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    usearch_dense_refinement_t const refinement = new (std::nothrow)
        usearch_dense_refinement_handle_t {&index, arguments->donors,    arguments->count,           0,
                                           0,      arguments->receivers, arguments->receivers_count, arguments->seed,
                                           0};
    if (reject_if(!refinement, usearch_status_out_of_memory_k, "The refinement could not be allocated", outcome))
        return;
    index.refinement = refinement;
    *arguments->refinement = refinement;
}

void usearch_dense_refinement_next(usearch_dense_refinement_next_t* arguments) {
    if (through_executor<usearch_dense_refinement_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->reselected) *arguments->reselected = 0;
    if (arguments->changed) *arguments->changed = 0;
    usearch_dense_refinement_handle_t& refinement = *arguments->refinement;
    if (reject_if(arguments->index != refinement.opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    for (; refinement.consumed != refinement.count; ++refinement.consumed) {
        usearch_dense_index_handle_t& donor = *refinement.donors[refinement.consumed];
        if (reject_if(!index.accommodate(index.entries.size() + donor.entries.size()), usearch_status_out_of_memory_k,
                      "The donor's entries could not be accommodated", outcome))
            return;
        for (dense_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        refinement.moved += donor.entries.size();
        delete std::exchange(refinement.donors[refinement.consumed], nullptr);
    }
    if (refinement.split_receivers_count != refinement.receivers_count) {
        auto const score = [&](dense_payload_t const& first, dense_payload_t const& second) {
            return distance(index.options.metric, first, second);
        };
        writers.unlock();
        std::span<usearch_dense_index_handle_t* const> const receivers {refinement.receivers,
                                                                        refinement.receivers_count};
        if (!split_into(index, receivers, refinement.seed, score, same_kind, outcome)) return;
        refinement.split_receivers_count = refinement.receivers_count;
        writers.lock();
    }
    // Each call continues the sweep where the last one stopped.
    std::size_t const entries = index.entries.size();
    bool cancelled = false;
    while (refinement.swept < entries && !cancelled)
        cancelled = !keep_going(arguments->progress, ++refinement.swept, entries);
    // The stand-in re-selects nothing, so its sweep visits every entry and changes none.
    if (arguments->reselected) *arguments->reselected = refinement.swept;
    if (arguments->changed) *arguments->changed = std::min(refinement.moved, refinement.swept);
    if (arguments->stats)
        arguments->stats->visited_members = refinement.swept, arguments->stats->computed_distances = refinement.swept;
    reject_if(cancelled && refinement.swept != entries, usearch_status_cancelled_k,
              "The caller's handler stopped the sweep", outcome);
}

void usearch_dense_refinement_finish(usearch_dense_refinement_finish_t* arguments) {
    if (through_executor<usearch_dense_refinement_finish>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    std::unique_ptr<usearch_dense_refinement_handle_t> const refinement {arguments->refinement};
    if (reject_if(arguments->index != refinement->opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    index.refinement = nullptr;
    std::size_t const entries = index.entries.size();
    if (arguments->stats)
        arguments->stats->visited_members = entries, arguments->stats->computed_distances = refinement->swept;
    if (arguments->report) {
        arguments->report->reselected = refinement->swept;
        arguments->report->changed = refinement->moved;
        arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    }
}

void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (reject_if(arguments->starts_count && !arguments->starts, usearch_status_invalid_argument_k,
                  "The starts must name keys", outcome))
        return;
    usearch_dense_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_dense_cursor_handle_t* const cursor = new (std::nothrow) usearch_dense_cursor_handle_t {};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    open_walk(*cursor, index, arguments->starts, arguments->starts_count);
    *arguments->cursor = cursor;
}

void usearch_dense_cursor_next(usearch_dense_cursor_next_t* arguments) {
    if (through_executor<usearch_dense_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->written = 0;
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    usearch_dense_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_dense_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    if (reject_if(serial_atomic_ref<u32_t>(cursor.pulling).fetch_test_and_set_bit(0, std::memory_order_acquire),
                  usearch_status_contended_k, "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    auto const score = [&](dense_payload_t const& first, dense_payload_t const& second) {
        return distance(index.options.metric, first, second);
    };
    pull_walk(*arguments, cursor, score, *arena, outcome);
    serial_atomic_ref<u32_t>(cursor.pulling).clear_bits(1, std::memory_order_release);
}

void usearch_dense_cursor_free(usearch_dense_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

/** Places a dense draw without shown points by projecting its row. */
struct projected_coordinates_t {
    static std::vector<f32_t> vector_of(dense_payload_t const& payload) {
        return std::vector<f32_t>(payload.begin(), payload.end());
    }
};

void usearch_dense_sample(usearch_dense_sample_t* arguments) {
    if (through_executor<usearch_dense_sample>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_dense_index_handle_t& index = *arguments->index;
    auto const score = [&](std::span<double const> first, std::span<double const> second) {
        return distance(index.options.metric, first, second);
    };
    draw_sample(*arguments, index, score, projected_coordinates_t {}, outcome);
}

void usearch_dense_split(usearch_dense_split_t* arguments) {
    if (through_executor<usearch_dense_split>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    auto const score = [&](dense_payload_t const& first, dense_payload_t const& second) {
        return distance(index.options.metric, first, second);
    };
    split_and_report(*arguments, index, score, same_kind, outcome);
}

void usearch_dense_search_stream_init(usearch_dense_search_stream_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->stream = nullptr;
    usearch_dense_index_handle_t& index = *arguments->index;
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
            index.entries, usearch_predicate_t {},
            [&](dense_payload_t const& payload) { return distance(index.options.metric, query_row, payload); },
            stream->ranked);
    }
    stream->opened_on = &index;
    stream->position = 0;
    *arguments->stream = stream;
}

void usearch_dense_search_stream_next(usearch_dense_search_stream_next_t* arguments) {
    if (through_executor<usearch_dense_search_stream_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->matches = {};
    usearch_dense_search_stream_handle_t& stream = *arguments->stream;
    if (reject_if(arguments->index != stream.opened_on, usearch_status_invalid_argument_k,
                  "The stream was opened on another index", outcome))
        return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    matches_outputs_t outputs {};
    if (!arena || !carve_ranked(*arena, 1, arguments->wanted, outputs, outcome)) return;
    if (reject_if(serial_atomic_ref<u32_t>(stream.pulling).fetch_test_and_set_bit(0, std::memory_order_acquire),
                  usearch_status_contended_k, "Another pull is still running on the stream", outcome))
        return;
    std::span<match_t const> const remaining = std::span<match_t const>(stream.ranked).subspan(stream.position);
    outputs.write(0, remaining);
    stream.position += outputs.counts[0];
    *arguments->matches = outputs.view();
    serial_atomic_ref<u32_t>(stream.pulling).clear_bits(1, std::memory_order_release);
}

void usearch_dense_search_stream_free(usearch_dense_search_stream_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_dense_save(usearch_dense_save_t* arguments) {
    if (through_executor<usearch_dense_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_if(index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view already is its checkpoint", outcome))
        return;
    // Saving is a reader: it demands only that nothing move or erase entries under it.
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    save_parts(index, arguments->path, arguments->progress, serialize_shard, outcome);
}

void usearch_dense_save_buffer(usearch_dense_save_buffer_t* arguments) {
    save_buffer<usearch_dense_save_buffer>(arguments, serialize_shard);
}

void usearch_dense_metadata_read(usearch_dense_metadata_read_t* arguments) {
    if (through_executor<usearch_dense_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome) || !read_first_part(blob, arguments->path, outcome)) return;
    dense_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    reject_if(!describe(*arguments->metadata, prefix), usearch_status_unknown_kind_k,
              "The checkpoint names a metric this build lacks", outcome);
}

void usearch_dense_metadata_read_buffer(usearch_dense_metadata_read_buffer_t* arguments) {
    metadata_read_buffer<usearch_dense_metadata_read_buffer>(arguments, parse_prefix, describe);
}

void usearch_dense_multiply(usearch_dense_multiply_t* arguments) {
    if (through_executor<usearch_dense_multiply>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->products = {};
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_status_config_mismatch_k,
                  "A product is a dot, so the kind must be inner product", outcome))
        return;
    std::size_t const rows_count = arguments->count;
    usearch_rows_view_t const& left = arguments->left;
    if (reject_view(left, rows_count, index, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {left.data}, outcome);
    if (!arena) return;

    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(left.scalar, dimensions);
    // A kept row holds the first pairs its walk scored - here, the first in storage order.
    std::size_t const kept = arguments->wanted ? std::min(arguments->wanted, indexed) : indexed;

    // Every row is scored before anything is written, so a cancelled call leaves its outputs null.
    std::vector<double> left_row(dimensions);
    std::vector<usearch_distance_t> products(rows_count * kept);
    std::size_t multiplied = 0;
    for (std::size_t row = 0; row != rows_count; ++row) {
        read_row(row_at<std::byte>(left.data, row, left.stride, row_bytes), left.scalar, left_row);
        for (std::size_t column = 0; column != kept; ++column)
            products[row * kept + column] = dot_product(left_row, index.entries[column].payload);
        if (arguments->stats) arguments->stats->computed_distances += indexed;
        ++multiplied;
        if (!keep_going(arguments->progress, multiplied, rows_count)) break;
    }
    if (reject_cancelled(multiplied, rows_count, outcome)) return;

    std::size_t const pairs = rows_count * kept;
    std::size_t* const offsets = carve<std::size_t>(*arena, rows_count + 1, outcome);
    usearch_key_t* const keys = offsets ? carve<usearch_key_t>(*arena, pairs, outcome) : nullptr;
    usearch_distance_t* const scores = keys ? carve<usearch_distance_t>(*arena, pairs, outcome) : nullptr;
    if (!scores) return;
    for (std::size_t row = 0; row <= rows_count; ++row) offsets[row] = row * kept;
    for (std::size_t pair = 0; pair != pairs; ++pair)
        keys[pair] = index.entries[pair % kept].key, scores[pair] = products[pair];
    *arguments->products = {keys, scores, offsets};
    if (arguments->written) *arguments->written = pairs;
    if (arguments->required) *arguments->required = rows_count * indexed;
}

void usearch_dense_attend(usearch_dense_attend_t* arguments) {
    if (through_executor<usearch_dense_attend>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    *arguments->outputs = {};
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_status_config_mismatch_k,
                  "A softmax weighs logits, so the kind must be inner product", outcome))
        return;

    std::size_t const queries_count = arguments->count, head_count = std::max<std::size_t>(arguments->head_count, 1);
    std::size_t const dimensions = index.options.dimensions;
    usearch_rows_view_t const& queries = arguments->queries;
    usearch_scalar_t const output_scalar = arguments->outputs_scalar ? arguments->outputs_scalar : queries.scalar;
    if (reject_view(queries, queries_count, index, outcome)) return;
    if (reject_vector_scalar(output_scalar, dimensions, outcome)) return;
    if (reject_if(!arguments->value_lookup, usearch_status_invalid_argument_k, "The value lookup must be set", outcome))
        return;
    if (reject_if(queries_count % head_count, usearch_status_shape_mismatch_k, "The heads must divide the rows",
                  outcome))
        return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                      {queries.data, arguments->carry}, outcome);
    if (!arena) return;
    std::size_t const output_bytes = row_extent(output_scalar, dimensions);
    std::byte* const outputs = carve<std::byte>(*arena, queries_count * output_bytes, outcome);
    if (!outputs) return;

    // Exact attention weighs every entry, so every value row resolves once, up front.
    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::vector<usearch_key_t> keys(indexed);
    std::vector<void const*> value_rows(indexed);
    for (std::size_t entry = 0; entry != indexed; ++entry) keys[entry] = index.entries[entry].key;
    if (indexed)
        arguments->value_lookup(arguments->value_lookup_state, keys.data(), indexed, index.options.quantization,
                                dimensions, value_rows.data());
    if (reject_if(std::find(value_rows.begin(), value_rows.end(), nullptr) != value_rows.end(),
                  usearch_status_shape_mismatch_k, "A scored key resolved to no value row", outcome))
        return;

    std::size_t const query_bytes = row_extent(queries.scalar, dimensions);
    std::size_t const value_bytes = row_extent(index.options.quantization, dimensions);
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
            write_row(row_at<std::byte>(outputs, row, output_bytes, output_bytes), output_scalar, accumulated);
            if (arguments->carry) arguments->carry[row] = {static_cast<float>(maximum), static_cast<float>(partition)};
            if (stats) stats->computed_distances += indexed;
        }
        ++answered;
        if (!keep_going(arguments->progress, answered, tokens)) break;
    }
    if (reject_cancelled(answered, tokens, outcome)) return;
    *arguments->outputs = {outputs, output_bytes, dimensions, output_scalar};
}

void usearch_dense_maxsim(usearch_dense_maxsim_t* arguments) {
    if (through_executor<usearch_dense_maxsim>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    usearch_dense_index_handle_t& index = *arguments->index;
    if (reject_if(index.options.metric != usearch_dense_metric_inner_k, usearch_status_config_mismatch_k,
                  "Late interaction sums maxima over logits, so the kind must be inner product", outcome))
        return;
    std::size_t const queries_count = arguments->count, rows_per_query = arguments->rows_per_query;
    std::size_t const* const query_offsets = arguments->query_offsets;
    usearch_rows_view_t const& queries = arguments->queries;
    auto const first_row = [&](std::size_t query) noexcept {
        return query_offsets ? query_offsets[query] : query * rows_per_query;
    };
    bool hollow = !query_offsets && !rows_per_query;
    for (std::size_t query = 0; query_offsets && query != queries_count; ++query)
        hollow |= query_offsets[query + 1] <= query_offsets[query];
    if (reject_if(hollow, usearch_status_shape_mismatch_k, "A query must hold at least one row", outcome)) return;
    if (reject_view(queries, first_row(queries_count), index, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {queries.data}, outcome);
    if (!arena) return;
    usearch_scalar_t const scalar_kind = queries.scalar;

    std::shared_lock readers(index.mutex);

    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = row_extent(scalar_kind, dimensions);

    // A document is one key holding its token entries, so the grouping happens here rather than in
    // the walk; a document's entries need no ordering and two documents need not share a token
    // count.
    std::vector<usearch_key_t> documents;
    std::vector<std::vector<std::size_t>> token_entries;
    std::unordered_map<us::utilities::uuid_t, std::size_t> document_of;
    for (std::size_t entry = 0; entry != index.entries.size(); ++entry) {
        auto const [slot, emplace_status] = document_of.emplace(uuid_of(index.entries[entry].key), documents.size());
        if (emplace_status) documents.push_back(index.entries[entry].key), token_entries.emplace_back();
        token_entries[slot->second].push_back(entry);
    }

    // The predicate is asked once per document key, before any of its rows is reduced.
    std::vector<usearch_admission_t> admissions(documents.size(), usearch_admit_k);
    usearch_predicate_t const& predicate = arguments->predicate;
    if (predicate.callback)
        for (std::size_t first = 0; first < documents.size(); first += 4)
            predicate.callback(documents.data() + first, admissions.data() + first,
                               std::min<std::size_t>(4, documents.size() - first), predicate.state);

    std::vector<double> query_rows;
    std::vector<match_t> ranked, kept_matches;
    std::vector<std::size_t> kept_offsets {0};
    std::size_t answered = 0, required = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::size_t const begin = first_row(query), rows = first_row(query + 1) - begin;
        query_rows.resize(rows * dimensions);
        for (std::size_t row = 0; row != rows; ++row)
            read_row(row_at<std::byte>(queries.data, begin + row, queries.stride, row_bytes), scalar_kind,
                     std::span<double>(query_rows).subspan(row * dimensions, dimensions));

        ranked.clear();
        for (std::size_t document = 0; document != documents.size(); ++document) {
            if (admissions[document] != usearch_admit_k) continue;
            double summed = 0;
            for (std::size_t row = 0; row != rows; ++row) {
                std::span<double const> const query_row {query_rows.data() + row * dimensions, dimensions};
                double best = std::numeric_limits<double>::lowest();
                for (std::size_t stored : token_entries[document])
                    best = std::max(best, dot_product(query_row, index.entries[stored].payload));
                summed += best;
            }
            ranked.push_back({summed, documents[document]});
        }
        // Summed maxima are a similarity, so the best key is the largest rather than the nearest.
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](match_t const& first, match_t const& second) { return first.distance > second.distance; });

        // A zero `wanted` keeps every scored document.
        std::size_t const kept = arguments->wanted ? std::min(arguments->wanted, ranked.size()) : ranked.size();
        kept_matches.insert(kept_matches.end(), ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(kept));
        kept_offsets.push_back(kept_matches.size());
        required += ranked.size();
        if (stats) {
            stats->visited_members += index.entries.size();
            stats->computed_distances += documents.size() * rows;
            stats->restarts += 1;
        }
        ++answered;
        if (!keep_going(arguments->progress, answered, queries_count)) break;
    }
    if (reject_cancelled(answered, queries_count, outcome)) return;
    std::size_t* const offsets = carve<std::size_t>(*arena, queries_count + 1, outcome);
    usearch_key_t* const keys = offsets ? carve<usearch_key_t>(*arena, kept_matches.size(), outcome) : nullptr;
    usearch_distance_t* const scores = keys ? carve<usearch_distance_t>(*arena, kept_matches.size(), outcome) : nullptr;
    if (!scores) return;
    std::copy(kept_offsets.begin(), kept_offsets.end(), offsets);
    for (std::size_t match = 0; match != kept_matches.size(); ++match)
        keys[match] = kept_matches[match].key, scores[match] = kept_matches[match].distance;
    *arguments->matches = {keys, scores, offsets};
    if (arguments->written) *arguments->written = kept_matches.size();
    if (arguments->required) *arguments->required = required;
}

} // extern "C"
