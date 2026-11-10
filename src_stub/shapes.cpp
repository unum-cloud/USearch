/**
 *  @file src_stub/shapes.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_shapes_* C ABI, scoring every stored shape
 *      against every query so a binding can be exercised without the engine.
 *
 *  One entry is a 3D point cloud whose point count varies, so batches arrive as a tape - every
 *  entry's points end to end, with Apache Arrow offsets counting points around them - rather than
 *  a strided run. That pairing is what this stand-in is for: a binding that counts its offsets in
 *  scalars or bytes instead of points, or that derives a point count from a stride, fails here
 *  immediately.
 *
 *  @section shapes_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  Only the two point-to-point kinds are implemented - Chamfer and Hausdorff - and both are scored
 *  in the frame the points arrived in, with no principal axes and no flips, so a rotated copy
 *  scores apart here where the engine scores it zero. The transport kinds are @b rejected rather
 *  than approximated, so a binding asking for Gromov-Wasserstein learns that here instead of
 *  receiving a plausible wrong number. Points are stored as @c double and @c quantization is
 *  checked but not applied, so @c get and @c cursor_next convert from @c f64 rather than from
 *  whatever the index froze. Refinement keeps no edges to re-select, so its @c changed is a
 *  constant zero rather than a settling signal, and a caller must not read it as one. A view
 *  copies its checkpoint rather than mapping it, and every VRAM lifecycle verb refuses. Traversal
 *  statistics are plausible fiction: shaped correctly, invented numerically. A checkpoint is a
 *  binary blob of this file's own memory image, readable by nothing else.
 *
 *  Keeping no edges, the stand-in links every one of its entries to its exact nearest at the frozen
 *  connectivity, the lists a perfect graph would hold where the engine's are the ones it stored. A
 *  cursor walks those lists breadth-first from its start keys, a split grows its parts over them,
 *  each receiver keeping the source's storage order, and a sample walks them as the library's plan
 *  walks its graph, placing its draws as the library does.
 */
#include <cmath> // `std::sqrt`

#include <algorithm> // `std::stable_sort`
#include <limits>    // `std::numeric_limits`
#include <memory>    // `std::unique_ptr` - a finished refinement's handle
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <utility>   // `std::exchange` - a consumed donor's slot
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/shapes_index.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Storage

constexpr std::size_t coordinates_k = 3;

using shapes_payload_t = std::vector<double>;
using shapes_entry_t = entry<shapes_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<shapes_entry_t>, "reallocation must move entries, never copy them");

/** What a handle froze at construction, or adopted from a checkpoint. */
struct shapes_index_options_t {

    /** What entries are compared by. */
    usearch_shapes_metric_t metric;

    /** The scalar kind points were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;

    /** The frozen point count, or zero for any count. */
    std::size_t points;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Upper bound on entries sharing one key. */
    std::size_t values_per_key;
};

using shapes_index_t = handle<shapes_payload_t, shapes_index_options_t>;

using shapes_cursor_t = graph_cursor<shapes_index_t>;

/** Whether a receiver frozen at @p receiver takes a part of an index frozen at @p index: the same
 *  kind, quantization, point count and connectivity. */
static bool same_kind(shapes_index_options_t const& index, shapes_index_options_t const& receiver) {
    return index.metric == receiver.metric && index.quantization == receiver.quantization &&
           index.points == receiver.points && index.connectivity == receiver.connectivity;
}

/** Whether this stand-in scores the kind, or rejects it rather than approximating it. */
static bool reject_metric(shapes_index_t const& index, usearch_outcome_t* outcome) {
    bool const scored = index.options.metric == usearch_shapes_metric_chamfer_k ||
                        index.options.metric == usearch_shapes_metric_hausdorff_k;
    return reject_if(!scored, usearch_status_unknown_kind_k, "The stand-in implements chamfer and hausdorff only",
                     outcome);
}

/** The point count of a stored entry, which its payload's extent fixes. */
static std::size_t points_of(shapes_entry_t const& entry) { return entry.payload.size() / coordinates_k; }

#pragma endregion Storage

#pragma region Scoring

static double point_distance(std::span<double const> first, std::span<double const> second) {
    double squared = 0;
    for (std::size_t axis = 0; axis != coordinates_k; ++axis) {
        double const delta = first[axis] - second[axis];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

/** Every point of @p from finds its nearest partner in @p onto; the two kinds differ only
 *  in how those distances are aggregated, so both come off this one sweep. */
static void nearest_partners(std::span<double const> from, std::span<double const> onto, double& summed,
                             double& worst) {
    summed = 0, worst = 0;
    std::size_t const from_points = from.size() / coordinates_k, onto_points = onto.size() / coordinates_k;
    if (!onto_points) return;
    for (std::size_t point = 0; point != from_points; ++point) {
        double nearest = std::numeric_limits<double>::infinity();
        for (std::size_t partner = 0; partner != onto_points; ++partner)
            nearest = min_of(nearest, point_distance(from.subspan(point * coordinates_k, coordinates_k),
                                                     onto.subspan(partner * coordinates_k, coordinates_k)));
        summed += nearest;
        worst = max_of(worst, nearest);
    }
    if (from_points) summed /= static_cast<double>(from_points);
}

/** Scores one pair under the frozen kind. @c usearch_shapes_init_ram refuses a kind this
 *  build does not implement, so only the two best-partner kinds reach here. */
static double distance(shapes_index_t const& index, std::span<double const> query, std::span<double const> stored) {
    double forward_mean = 0, forward_worst = 0, reverse_mean = 0, reverse_worst = 0;
    nearest_partners(query, stored, forward_mean, forward_worst);
    nearest_partners(stored, query, reverse_mean, reverse_worst);
    if (index.options.metric == usearch_shapes_metric_chamfer_k) return (forward_mean + reverse_mean) / 2;
    return max_of(forward_worst, reverse_worst);
}

/** Whether the kind is one this build implements. The transport kinds need a solver and
 *  an SVD, which a brute-force stand-in deliberately does not carry. */
static bool known_metric(usearch_shapes_metric_t metric) {
    return metric == usearch_shapes_metric_chamfer_k || metric == usearch_shapes_metric_hausdorff_k;
}

/** Widens entry @p entry of a tape into doubles, sized by its own point count. */
static void read_shape(usearch_tape_view_t tape, std::size_t entry, std::vector<double>& values) {
    std::size_t const first_point = offset_at(tape, entry), point_count = offset_at(tape, entry + 1) - first_point;
    values.resize(point_count * coordinates_k);
    std::size_t const point_bytes = row_extent(tape.scalar, coordinates_k);
    for (std::size_t point = 0; point != point_count; ++point)
        read_row(row_at<std::byte>(tape.data, first_point + point, point_bytes, point_bytes), tape.scalar,
                 std::span<double>(values).subspan(point * coordinates_k, coordinates_k));
}

/** The output twin of @c read_shape: writes the entries at @p entries into @p arena as one tape of
 *  @p scalar_kind points, writing @p tape only once every point is read, and answers false once
 *  @p outcome records why not. */
static bool copy_entries(shapes_index_t const& index, std::span<std::size_t const> entries,
                         usearch_scalar_t scalar_kind, usearch_arena_handle_t& arena, usearch_tape_view_t& tape,
                         usearch_outcome_t* outcome) {
    std::size_t const point_bytes = row_extent(scalar_kind, coordinates_k);
    std::size_t total_points = 0;
    for (std::size_t entry : entries) total_points += points_of(index.entries[entry]);
    std::size_t* const offsets = carve<std::size_t>(arena, entries.size() + 1, outcome);
    std::byte* const written = offsets ? carve<std::byte>(arena, total_points * point_bytes, outcome) : nullptr;
    if (!written) return false;
    offsets[0] = 0;
    for (std::size_t found = 0; found != entries.size(); ++found) {
        shapes_entry_t const& entry = index.entries[entries[found]];
        std::size_t const points = points_of(entry);
        for (std::size_t point = 0; point != points; ++point)
            write_row(row_at<std::byte>(written, offsets[found] + point, point_bytes, point_bytes), scalar_kind,
                      std::span<double const>(entry.payload).subspan(point * coordinates_k, coordinates_k));
        offsets[found + 1] = offsets[found] + points;
    }
    write_output_tape(written, offsets, scalar_kind, tape);
    return true;
}

/**
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
static std::size_t search_tape(shapes_index_t& index, usearch_tape_view_t queries, std::size_t count,
                               matches_outputs_t const& outputs, std::size_t expansion_factor,
                               usearch_traversal_stats_t* stats, usearch_predicate_t predicate,
                               usearch_progress_t progress) {
    std::size_t const entries = index.entries.size();
    if (stats && expansion_factor) stats->computed_distances += expansion_factor * count;

    std::vector<double> query_shape;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != count; ++query) {
        read_shape(queries, query, query_shape);
        rank<shapes_payload_t>(
            std::span<shapes_entry_t const>(index.entries).first(entries), predicate,
            [&](shapes_payload_t const& payload) { return distance(index, query_shape, payload); }, ranked);
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

/** What @c load needs before it can read an entry. The blob is this file's own memory
 *  image rather than a format: no magic, no version, and nothing else reads it back. */
struct shapes_checkpoint_prefix_t {

    /** The frozen point count, or zero for any count. */
    std::size_t points;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many entries follow the prefix. */
    std::size_t entries;

    /** Bytes of entry payload that follow, keys and point counts included. */
    std::size_t payload_bytes;

    /** What entries are compared by. */
    usearch_shapes_metric_t metric;

    /** The scalar kind the points were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;
};

/** The per-entry overhead of the blob: the key, then the point count that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(us::utilities::u64_t);

static std::size_t payload_extent(shapes_index_t const& index, std::size_t entries) {
    std::size_t extent = 0;
    for (std::size_t entry = 0; entry != entries; ++entry)
        extent += record_overhead_k + index.entries[entry].payload.size() * sizeof(double);
    return extent;
}

static std::optional<std::size_t> serialized_bytes(shapes_index_t const& index, std::size_t entries) {
    return blob_bytes(sizeof(shapes_checkpoint_prefix_t), payload_extent(index, entries), 1);
}

static void write_into(shapes_index_t const& index, std::size_t entries, bytes_span_t target) {
    shapes_checkpoint_prefix_t prefix {};
    prefix.points = index.options.points;
    prefix.connectivity = index.options.connectivity;
    prefix.entries = entries;
    prefix.payload_bytes = payload_extent(index, entries);
    prefix.metric = index.options.metric;
    prefix.quantization = index.options.quantization;
    write_pod(target, prefix);

    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t position = 0; position != entries; ++position) {
        shapes_entry_t const& entry = index.entries[position];
        std::size_t const point_count = points_of(entry);
        std::size_t const payload_bytes = entry.payload.size() * sizeof(double);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, point_count);
        cursor = cursor.subspan(sizeof(point_count));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(payload_bytes),
                                                            bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** Reads the prefix and checks the blob is long enough for what it claims. */
static bool parse_prefix(shapes_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(shapes_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<shapes_checkpoint_prefix_t>(blob);
    return fits_blob(blob, sizeof(shapes_checkpoint_prefix_t), prefix.payload_bytes, 1, outcome);
}

/** Writes what a checkpoint froze into @p metadata, answering whether we implement its metric. */
static bool describe(usearch_shapes_metadata_t& metadata, shapes_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    metadata = {prefix.metric, prefix.quantization, prefix.points, prefix.connectivity};
    return true;
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we implement. The
 *  bound on entries per key is the loader's to choose, since no checkpoint records it. */
static bool adopt(shapes_index_options_t& options, shapes_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.points = prefix.points;
    options.connectivity = prefix.connectivity;
    options.metric = prefix.metric;
    options.quantization = prefix.quantization;
    return true;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
static bool read_from(shapes_index_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    shapes_checkpoint_prefix_t prefix {};
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

    bytes_view_t cursor = blob.subspan(sizeof(shapes_checkpoint_prefix_t));
    shapes_payload_t payload;
    for (std::size_t entry_index = 0; entry_index != entries; ++entry_index) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        usearch_key_t key;
        std::size_t point_count = 0;
        key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        point_count = read_pod<std::size_t>(cursor);
        cursor = cursor.subspan(sizeof(point_count));
        std::size_t const scalars = point_count * coordinates_k;
        if (cursor.size() < scalars * sizeof(double)) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        payload.resize(scalars);
        [[maybe_unused]] status_t const copied = copy_bytes(bytes_of(payload.data(), payload.size()),
                                                            cursor.first(scalars * sizeof(double)));
        cursor = cursor.subspan(scalars * sizeof(double));
        index.append(key, payload);
    }
    return true;
}

/** Writes @p shard's whole image into @p blob; false once @p outcome records why not. */
static bool serialize_shard(shapes_index_t const& shard, std::vector<std::byte>& blob, usearch_outcome_t* outcome) {
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

struct usearch_shapes_index_handle_t : shapes_index_t {};
struct usearch_shapes_cursor_handle_t : shapes_cursor_t {};

extern "C" {

char const* usearch_shapes_metric_name(usearch_shapes_metric_t metric) {
    switch (metric) {
    case usearch_shapes_metric_chamfer_k: return "chamfer";
    case usearch_shapes_metric_hausdorff_k: return "hausdorff";
    case usearch_shapes_metric_gromov_wasserstein_k: return "gromov_wasserstein";
    case usearch_shapes_metric_wasserstein_procrustes_k: return "wasserstein_procrustes";
    default: return "unknown";
    }
}

usearch_shapes_metric_t usearch_shapes_metric_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_shapes_metric_chamfer_k; code <= usearch_shapes_metric_wasserstein_procrustes_k; ++code) {
        usearch_shapes_metric_t const metric = static_cast<usearch_shapes_metric_t>(code);
        if (named(name, usearch_shapes_metric_name(metric))) return metric;
    }
    return usearch_shapes_metric_unknown_k;
}

void usearch_shapes_init_ram(usearch_shapes_init_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_if(!known_metric(arguments->metric), usearch_status_unknown_kind_k,
                  "The stand-in implements chamfer and hausdorff only", outcome))
        return;
    if (reject_if(arguments->quantization != usearch_scalar_f16_k && arguments->quantization != usearch_scalar_f32_k,
                  usearch_status_unsupported_k, "Points are stored as f16 or f32 only", outcome))
        return;
    usearch_shapes_index_handle_t* const index = new (std::nothrow) usearch_shapes_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->metric, arguments->quantization, arguments->points,
                      arguments->connectivity ? arguments->connectivity : us::utilities::connectivity_16_k,
                      arguments->values_per_key};
    index->shard_capacity = arguments->ram.shard_capacity;
    *arguments->index = index;
}

void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_shapes_load_ram(usearch_shapes_load_ram_t* arguments) {
    if (through_executor<usearch_shapes_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_shapes_index_handle_t* const index = open_parts<usearch_shapes_index_handle_t>(
        blob, arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_shapes_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, 1, 1);
    *arguments->index = index;
}

void usearch_shapes_load_ram_buffer(usearch_shapes_load_ram_buffer_t* arguments) {
    load_ram_buffer<usearch_shapes_load_ram_buffer>(arguments, open_checkpoint<usearch_shapes_index_handle_t>);
}

void usearch_shapes_load_vram(usearch_shapes_load_vram_t* arguments) {
    if (through_executor<usearch_shapes_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_shapes_load_vram_buffer(usearch_shapes_load_vram_buffer_t* arguments) {
    if (through_executor<usearch_shapes_load_vram_buffer>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_shapes_view_ram(usearch_shapes_view_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Entries are still copied out of the mapping, so only the view's refusals are mirrored, not
    // its zero cost.
    usearch_shapes_index_handle_t* const index = open_parts<usearch_shapes_index_handle_t>(
        served.view(), arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_shapes_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_shapes_view_nvme(usearch_shapes_view_nvme_t* arguments) {
    if (through_executor<usearch_shapes_view_nvme>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Serves the drives' checkpoint as the RAM view does, mirroring only the residency's refusals.
    usearch_shapes_index_handle_t* const index = open_parts<usearch_shapes_index_handle_t>(
        served.view(), arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_shapes_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->served = std::move(served);
    index->residency = usearch_residency_nvme_k;
    *arguments->index = index;
}

void usearch_shapes_free(usearch_shapes_index_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_shapes_reserve(usearch_shapes_reserve_t* arguments) {
    if (through_executor<usearch_shapes_reserve>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    reject_if(!index.accommodate(arguments->entries), usearch_status_out_of_memory_k,
              "The capacity could not be reserved", outcome);
}

void usearch_shapes_clear(usearch_shapes_clear_t* arguments) {
    if (through_executor<usearch_shapes_clear>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    index.reset();
}

void usearch_shapes_describe(usearch_shapes_describe_t* arguments) {
    if (through_executor<usearch_shapes_describe>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->description = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    std::size_t total_points = 0;
    for (shapes_entry_t const& entry : index.entries) total_points += points_of(entry);
    shapes_index_options_t const& options = index.options;
    *arguments->description = {
        {options.metric, options.quantization, options.points, options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        total_points,
        payload_extent(index, index.entries.size()),
        usearch_cpu_class_serial_k,
        usearch_gpu_class_unknown_k,
        index.shards(),
        usearch_load_path_none_k,
    };
}

void usearch_shapes_insert(usearch_shapes_insert_t* arguments) {
    if (through_executor<usearch_shapes_insert>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_shapes_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_tape_view_t const& shapes = arguments->shapes;
    if (reject_read_only(index, outcome)) return;
    if (reject_scalar(shapes.scalar, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;
    if (reject_tape(shapes, count, outcome)) return;
    if (index.options.points)
        for (std::size_t offset = 0; offset != count; ++offset)
            if (reject_if(offset_at(shapes, offset + 1) - offset_at(shapes, offset) != index.options.points,
                          usearch_status_shape_mismatch_k, "The index froze a point count this entry does not match",
                          outcome))
                return;

    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    shapes_payload_t payload;
    std::size_t published = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        read_shape(shapes, offset, payload);
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

void usearch_shapes_refine(usearch_shapes_refine_t* arguments) {
    if (through_executor<usearch_shapes_refine>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    std::size_t const entries = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    bool cancelled = false;
    for (std::size_t entry = 0; entry != entries && !cancelled; ++entry)
        cancelled = !keep_going(arguments->progress, entry + 1, entries);
    if (arguments->stats) arguments->stats->visited_members = entries;
    // An exact scan reaches every entry it holds in one level.
    if (arguments->report) arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

void usearch_shapes_health(usearch_shapes_health_t* arguments) {
    if (through_executor<usearch_shapes_health>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->reachability = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    // An exact scan reaches every entry it holds in one level.
    std::size_t const entries = index.entries.size();
    *arguments->reachability = {entries, entries, entries ? 1u : 0u};
}

void usearch_shapes_search(usearch_shapes_search_t* arguments) {
    if (through_executor<usearch_shapes_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->matches = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_tape_view_t const& queries = arguments->queries;
    if (reject_scalar(queries.scalar, outcome)) return;
    if (reject_metric(index, outcome)) return;
    if (reject_tape(queries, count, outcome)) return;
    if (reject_traversal(arguments->traversal, outcome)) return;
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

void usearch_shapes_count(usearch_shapes_count_t* arguments) {
    if (through_executor<usearch_shapes_count>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    auto const range = index.by_key.equal_range(uuid_of(arguments->key));
    std::size_t entries = 0, total_points = 0, largest_points = 0;
    for (auto position = range.first; position != range.second; ++position, ++entries) {
        std::size_t const points = points_of(index.entries[position->second]);
        total_points += points;
        largest_points = max_of(largest_points, points);
    }
    *arguments->key_stats = {entries, total_points, largest_points};
}

void usearch_shapes_get(usearch_shapes_get_t* arguments) {
    if (through_executor<usearch_shapes_get>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->shapes = {};
    if (arguments->written) *arguments->written = 0;
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_scalar(arguments->outputs_scalar, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const entries = nodes_of(index, arguments->key);
    if (entries.empty()) return;
    if (!copy_entries(index, entries, arguments->outputs_scalar, *arena, *arguments->shapes, outcome)) return;
    if (arguments->written) *arguments->written = entries.size();
}

void usearch_shapes_absorb(usearch_shapes_absorb_t* arguments) {
    if (through_executor<usearch_shapes_absorb>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_shapes_index_t* const donors = arguments->donors;
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
        usearch_shapes_index_handle_t& donor = *donors[position];
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (shapes_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
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

/** The stand-in's refinement: the donors it still absorbs, and how far its sweep got. Its index
 *  admits no insertion, save or other refinement until it finishes. */
struct usearch_shapes_refinement_handle_t {

    /** The handle the refinement was opened on, only ever compared with the one each call
     *  passes. */
    usearch_shapes_index_handle_t const* opened_on;

    /** The caller's donor slots, each nulled once consumed. */
    usearch_shapes_index_t* donors;
    std::size_t count;

    /** Donors consumed so far, and entries the sweep visited so far. */
    std::size_t consumed;
    std::size_t swept;

    /** The receivers a split fills, the seed picking its roots, and those split off so far. */
    usearch_shapes_index_t const* receivers;
    std::size_t receivers_count;
    us::utilities::u64_t seed;
    std::size_t split_receivers_count;

    /** Entries the absorbed donors moved in, each one a node whose links changed. */
    std::size_t moved = 0;
};

void usearch_shapes_refinement_init(usearch_shapes_refinement_init_t* arguments) {
    if (through_executor<usearch_shapes_refinement_init>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->refinement = nullptr;
    usearch_shapes_index_handle_t& index = *arguments->index;
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
    usearch_shapes_refinement_t const refinement = new (std::nothrow)
        usearch_shapes_refinement_handle_t {&index, arguments->donors,    arguments->count,           0,
                                            0,      arguments->receivers, arguments->receivers_count, arguments->seed,
                                            0};
    if (reject_if(!refinement, usearch_status_out_of_memory_k, "The refinement could not be allocated", outcome))
        return;
    index.refinement = refinement;
    *arguments->refinement = refinement;
}

void usearch_shapes_refinement_next(usearch_shapes_refinement_next_t* arguments) {
    if (through_executor<usearch_shapes_refinement_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->reselected) *arguments->reselected = 0;
    if (arguments->changed) *arguments->changed = 0;
    usearch_shapes_refinement_handle_t& refinement = *arguments->refinement;
    if (reject_if(arguments->index != refinement.opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_shapes_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    for (; refinement.consumed != refinement.count; ++refinement.consumed) {
        usearch_shapes_index_handle_t& donor = *refinement.donors[refinement.consumed];
        if (reject_if(!index.accommodate(index.entries.size() + donor.entries.size()), usearch_status_out_of_memory_k,
                      "The donor's entries could not be accommodated", outcome))
            return;
        for (shapes_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        refinement.moved += donor.entries.size();
        delete std::exchange(refinement.donors[refinement.consumed], nullptr);
    }
    if (refinement.split_receivers_count != refinement.receivers_count) {
        auto const score = [&](shapes_payload_t const& first, shapes_payload_t const& second) {
            return distance(index, first, second);
        };
        writers.unlock();
        std::span<usearch_shapes_index_handle_t* const> const receivers {refinement.receivers,
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
    if (arguments->changed) *arguments->changed = min_of(refinement.moved, refinement.swept);
    if (arguments->stats)
        arguments->stats->visited_members = refinement.swept, arguments->stats->computed_distances = refinement.swept;
    reject_if(cancelled && refinement.swept != entries, usearch_status_cancelled_k,
              "The caller's handler stopped the sweep", outcome);
}

void usearch_shapes_refinement_finish(usearch_shapes_refinement_finish_t* arguments) {
    if (through_executor<usearch_shapes_refinement_finish>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    std::unique_ptr<usearch_shapes_refinement_handle_t> const refinement {arguments->refinement};
    if (reject_if(arguments->index != refinement->opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_shapes_index_handle_t& index = *arguments->index;
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

void usearch_shapes_cursor_init(usearch_shapes_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (reject_if(arguments->starts_count && !arguments->starts, usearch_status_invalid_argument_k,
                  "The starts must name keys", outcome))
        return;
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (arguments->starts_count && reject_metric(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    usearch_shapes_cursor_handle_t* const cursor = new (std::nothrow) usearch_shapes_cursor_handle_t {};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    open_walk(*cursor, index, arguments->starts, arguments->starts_count);
    *arguments->cursor = cursor;
}

void usearch_shapes_cursor_next(usearch_shapes_cursor_next_t* arguments) {
    if (through_executor<usearch_shapes_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->written = 0;
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    usearch_shapes_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (arguments->links && reject_metric(index, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    if (reject_if(serial_atomic_ref<u32_t>(cursor.pulling).fetch_test_and_set_bit(0, std::memory_order_acquire),
                  usearch_status_contended_k, "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    auto const score = [&](shapes_payload_t const& first, shapes_payload_t const& second) {
        return distance(index, first, second);
    };
    pull_walk(*arguments, cursor, score, *arena, outcome);
    serial_atomic_ref<u32_t>(cursor.pulling).clear_bits(1, std::memory_order_release);
}

void usearch_shapes_cursor_free(usearch_shapes_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_shapes_sample(usearch_shapes_sample_t* arguments) {
    if (through_executor<usearch_shapes_sample>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_metric(index, outcome)) return;
    auto const score = [&](shapes_payload_t const& first, shapes_payload_t const& second) {
        return distance(index, first, second);
    };
    draw_sample(*arguments, index, score, no_fixed_coordinates_t {}, outcome);
}

void usearch_shapes_split(usearch_shapes_split_t* arguments) {
    if (through_executor<usearch_shapes_split>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_metric(index, outcome)) return;
    auto const score = [&](shapes_payload_t const& first, shapes_payload_t const& second) {
        return distance(index, first, second);
    };
    split_and_report(*arguments, index, score, same_kind, outcome);
}

void usearch_shapes_save(usearch_shapes_save_t* arguments) {
    if (through_executor<usearch_shapes_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_shapes_index_handle_t& index = *arguments->index;
    if (reject_if(index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view already is its checkpoint", outcome))
        return;
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    save_parts(index, arguments->path, arguments->progress, serialize_shard, outcome);
}

void usearch_shapes_save_buffer(usearch_shapes_save_buffer_t* arguments) {
    save_buffer<usearch_shapes_save_buffer>(arguments, serialize_shard);
}

void usearch_shapes_metadata_read(usearch_shapes_metadata_read_t* arguments) {
    if (through_executor<usearch_shapes_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome) || !read_first_part(blob, arguments->path, outcome)) return;
    shapes_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    reject_if(!describe(*arguments->metadata, prefix), usearch_status_unknown_kind_k,
              "The checkpoint names a metric this build lacks", outcome);
}

void usearch_shapes_metadata_read_buffer(usearch_shapes_metadata_read_buffer_t* arguments) {
    metadata_read_buffer<usearch_shapes_metadata_read_buffer>(arguments, parse_prefix, describe);
}

} // extern "C"
