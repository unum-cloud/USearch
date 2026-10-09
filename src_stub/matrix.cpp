/**
 *  @file src_stub/matrix.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_matrix_* C ABI, scoring every stored matrix
 *      against every query so a binding can be exercised without the engine.
 *
 *  One entry is a rank-2 tensor whose height varies and whose width is frozen index-wide, so
 *  batches arrive as a tape - every entry's rows end to end, with Apache Arrow offsets counting
 *  rows around them - rather than a strided run. That pairing is what this stand-in is for: a
 *  binding that counts its offsets in scalars or bytes instead of rows, or that derives a height
 *  from a stride, fails here immediately.
 *
 *  @section matrix_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  Only the two row-to-row kinds are implemented - Chamfer and Hausdorff - and the transport and
 *  superposition kinds are @b rejected rather than approximated by them, so a binding asking for
 *  Gromov-Wasserstein learns that here instead of receiving a plausible wrong number. Rows are
 *  stored as @c double and @c quantization is ignored, so the matrices a cursor borrows are
 *  @c f64 whatever the index froze. Refinement keeps no edges to re-select, so its @c changed is
 *  a constant zero rather than a settling signal, and a caller must not read it as one. A view
 *  copies its checkpoint rather than mapping it, and every VRAM lifecycle verb refuses. Traversal
 *  statistics are plausible fiction: shaped correctly, invented numerically. A checkpoint is a
 *  binary blob of this file's own memory image, readable by nothing else.
 */
#include <cmath> // `std::sqrt`

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <limits>    // `std::numeric_limits`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/matrix_index.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Storage

using matrix_payload_t = std::vector<double>;
using matrix_entry_t = entry<matrix_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<matrix_entry_t>, "reallocation must move rows, never copy them");

/** What a handle froze at construction, or adopted from a checkpoint. */
struct matrix_index_options_t {

    /** What entries are compared by. */
    usearch_matrix_metric_t metric;

    /** The scalar kind rows were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;

    /** The frozen row width shared by every entry. */
    std::size_t columns;

    /** The frozen row count, or zero where heights are ragged. */
    std::size_t rows;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Upper bound on entries sharing one key. */
    std::size_t values_per_key;
};

using matrix_index_t = handle<matrix_payload_t, matrix_index_options_t>;

struct matrix_cursor_t {

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    matrix_index_t const* opened_on;

    /** The next entry to hand out. */
    std::size_t position;

    /** The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag pulling;
};

/** Whether the width is frozen; a zeroed one makes every derived extent zero. */
static bool reject_unfrozen(matrix_index_t const& index, usearch_outcome_t* outcome) {
    return reject_if(!index.options.columns, usearch_status_uninitialized_k, "The geometry was never frozen", outcome);
}

/** Whether this stand-in scores the kind, or rejects it rather than approximating it. */
static bool reject_metric(matrix_index_t const& index, usearch_outcome_t* outcome) {
    bool const scored = index.options.metric == usearch_matrix_metric_chamfer_k ||
                        index.options.metric == usearch_matrix_metric_hausdorff_k;
    return reject_if(!scored, usearch_status_unknown_kind_k, "The stand-in implements chamfer and hausdorff only",
                     outcome);
}

/** The height of a stored entry, which its payload's extent and the frozen width fix. */
static std::size_t rows_of(matrix_index_t const& index, matrix_entry_t const& entry) {
    return entry.payload.size() / index.options.columns;
}

#pragma endregion Storage

#pragma region Scoring

static double row_distance(std::span<double const> first, std::span<double const> second) {
    double squared = 0;
    for (std::size_t column = 0; column != first.size(); ++column) {
        double const delta = first[column] - second[column];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

/** Every row of @p from finds its nearest partner in @p onto; the two kinds differ only
 *  in how those distances are aggregated, so both come off this one sweep. */
static void nearest_partners(std::span<double const> from, std::span<double const> onto, std::size_t columns,
                             double& summed, double& worst) {
    summed = 0, worst = 0;
    std::size_t const from_rows = from.size() / columns, onto_rows = onto.size() / columns;
    if (!onto_rows) return;
    for (std::size_t row = 0; row != from_rows; ++row) {
        double nearest = std::numeric_limits<double>::infinity();
        for (std::size_t partner = 0; partner != onto_rows; ++partner)
            nearest = std::min(
                nearest, row_distance(from.subspan(row * columns, columns), onto.subspan(partner * columns, columns)));
        summed += nearest;
        worst = std::max(worst, nearest);
    }
    if (from_rows) summed /= static_cast<double>(from_rows);
}

/** Scores one pair under the frozen kind. @c usearch_matrix_init_ram refuses a kind this
 *  build does not implement, so only the two best-partner kinds reach here. */
static double distance(matrix_index_t const& index, std::span<double const> query, std::span<double const> stored) {
    std::size_t const columns = index.options.columns;
    double forward_mean = 0, forward_worst = 0, reverse_mean = 0, reverse_worst = 0;
    nearest_partners(query, stored, columns, forward_mean, forward_worst);
    nearest_partners(stored, query, columns, reverse_mean, reverse_worst);
    if (index.options.metric == usearch_matrix_metric_chamfer_k) return forward_mean + reverse_mean;
    return std::max(forward_worst, reverse_worst);
}

/** Whether the kind is one this build implements. The transport and superposition kinds
 *  need a solver and an SVD, which a brute-force stand-in deliberately does not carry. */
static bool known_metric(usearch_matrix_metric_t metric) {
    return metric == usearch_matrix_metric_chamfer_k || metric == usearch_matrix_metric_hausdorff_k;
}

/** Widens entry @p entry of a tape into doubles, sized by its own row count. */
static void read_matrix(usearch_tape_view_t tape, std::size_t entry, std::size_t columns, usearch_scalar_t scalar_kind,
                        std::vector<double>& values) {
    std::size_t const first_row = tape.offsets[entry], row_count = tape.offsets[entry + 1] - first_row;
    values.resize(row_count * columns);
    std::size_t const row_bytes = row_extent(scalar_kind, columns);
    for (std::size_t row = 0; row != row_count; ++row)
        read_row(row_at<std::byte>(tape.data, first_row + row, row_bytes, row_bytes), scalar_kind,
                 std::span<double>(values).subspan(row * columns, columns));
}

/**
 *  @brief The body of a search, filtered when @p predicate is set.
 *  @return Queries answered, short of @p count exactly when the handler cancelled.
 */
static std::size_t search_tape(matrix_index_t& index, usearch_tape_view_t queries, std::size_t count,
                               usearch_scalar_t scalar_kind, usearch_matches_span_t const& matches,
                               std::size_t expansion_factor, usearch_matrix_traversal_stats_t* stats,
                               usearch_predicate_t predicate, void* predicate_state,
                               usearch_progress_callback_t progress, void* progress_state) {
    std::size_t const rows = index.entries.size();
    std::size_t const columns = index.options.columns;
    std::size_t const wanted = matches.wanted;
    if (stats && expansion_factor) stats->computed_distances += expansion_factor * count;

    std::span<std::size_t> const counts {matches.counts, count};
    std::fill(counts.begin(), counts.end(), 0);
    std::vector<double> query_matrix;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        read_matrix(queries, query, columns, scalar_kind, query_matrix);
        rank<matrix_payload_t>(
            std::span<matrix_entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
            [&](matrix_payload_t const& payload) { return distance(index, query_matrix, payload); }, ranked);
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

/** What @c load needs before it can read an entry. The blob is this file's own memory
 *  image rather than a format: no magic, no version, and nothing else reads it back. */
struct matrix_checkpoint_prefix_t {

    /** The frozen row width shared by every entry. */
    std::size_t columns;

    /** The frozen row count, or zero where heights are ragged. */
    std::size_t rows;

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many entries follow the prefix. */
    std::size_t entries;

    /** Bytes of entry payload that follow, keys and row counts included. */
    std::size_t payload_bytes;

    /** What entries are compared by. */
    usearch_matrix_metric_t metric;

    /** The scalar kind the rows were frozen at, which metadata reports back. */
    usearch_scalar_t quantization;
};

/** The per-entry overhead of the blob: the key, then the row count that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(us::u64_t);

static std::size_t payload_extent(matrix_index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row)
        extent += record_overhead_k + index.entries[row].payload.size() * sizeof(double);
    return extent;
}

static std::optional<std::size_t> serialized_bytes(matrix_index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(matrix_checkpoint_prefix_t), payload_extent(index, rows), 1);
}

static void write_into(matrix_index_t const& index, std::size_t rows, bytes_span_t target) {
    matrix_checkpoint_prefix_t prefix {};
    prefix.columns = index.options.columns;
    prefix.rows = index.options.rows;
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.metric = index.options.metric;
    prefix.quantization = index.options.quantization;
    write_pod(target, prefix);

    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        matrix_entry_t const& entry = index.entries[row];
        std::size_t const row_count = rows_of(index, entry);
        std::size_t const payload_bytes = entry.payload.size() * sizeof(double);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, row_count);
        cursor = cursor.subspan(sizeof(row_count));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(payload_bytes),
                                                            bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** Reads the prefix and checks the blob is long enough for what it claims. */
static bool parse_prefix(matrix_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(matrix_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<matrix_checkpoint_prefix_t>(blob);
    return fits_blob(blob, sizeof(matrix_checkpoint_prefix_t), prefix.payload_bytes, 1, outcome);
}

/** Adopts a checkpoint's configuration, answering whether its metric is one we implement. The
 *  bound on entries per key is the loader's to choose, since no checkpoint records it. */
static bool adopt(matrix_index_options_t& options, matrix_checkpoint_prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.columns = prefix.columns;
    options.rows = prefix.rows;
    options.connectivity = prefix.connectivity;
    options.metric = prefix.metric;
    options.quantization = prefix.quantization;
    return true;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
static bool read_from(matrix_index_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    matrix_checkpoint_prefix_t prefix {};
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

    bytes_view_t cursor = blob.subspan(sizeof(matrix_checkpoint_prefix_t));
    matrix_payload_t payload;
    for (std::size_t entry_index = 0; entry_index != entries; ++entry_index) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        usearch_key_t key;
        std::size_t row_count = 0;
        key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        row_count = read_pod<std::size_t>(cursor);
        cursor = cursor.subspan(sizeof(row_count));
        std::size_t const scalars = row_count * prefix.columns;
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

struct usearch_matrix_index_handle_t : matrix_index_t {};
struct usearch_matrix_cursor_handle_t : matrix_cursor_t {};

extern "C" {

void usearch_matrix_init_ram(usearch_matrix_init_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_if(!known_metric(arguments->metric), usearch_status_unknown_kind_k,
                  "The stand-in implements chamfer and hausdorff only", outcome))
        return;
    usearch_matrix_index_handle_t* const index = new (std::nothrow) usearch_matrix_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->metric,
                      arguments->quantization,
                      arguments->columns,
                      arguments->rows,
                      arguments->connectivity ? arguments->connectivity : us::connectivity_16_k,
                      arguments->values_per_key};
    *arguments->index = index;
}

void usearch_matrix_init_vram(usearch_matrix_init_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends");
}

void usearch_matrix_load_ram(usearch_matrix_load_ram_t* arguments) {
    if (deferred<usearch_matrix_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_matrix_index_handle_t* const index = open_checkpoint<usearch_matrix_index_handle_t>(blob, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
    *arguments->index = index;
}

void usearch_matrix_load_vram(usearch_matrix_load_vram_t* arguments) {
    if (deferred<usearch_matrix_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends");
}

void usearch_matrix_view_ram(usearch_matrix_view_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Rows are still copied out of the mapping, so only the view's refusals are mirrored, not its zero cost.
    usearch_matrix_index_handle_t* const index = open_checkpoint<usearch_matrix_index_handle_t>(served.view(), outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_matrix_free(usearch_matrix_index_t handle) { delete handle; }

void usearch_matrix_reserve(usearch_matrix_reserve_t* arguments) {
    if (deferred<usearch_matrix_reserve>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_matrix_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    reject_if(!index.accommodate(arguments->entries), usearch_status_out_of_memory_k,
              "The capacity could not be reserved", outcome);
}

void usearch_matrix_clear(usearch_matrix_clear_t* arguments) {
    if (deferred<usearch_matrix_clear>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_matrix_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    index.reset();
}

void usearch_matrix_describe(usearch_matrix_describe_t* arguments) {
    if (deferred<usearch_matrix_describe>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->description = {};
    usearch_matrix_index_handle_t& index = *arguments->index;
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    std::size_t total_rows = 0;
    for (matrix_entry_t const& entry : index.entries) total_rows += rows_of(index, entry);
    matrix_index_options_t const& options = index.options;
    *arguments->description = {
        {options.metric, options.quantization, options.columns, options.rows, options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        total_rows,
        payload_extent(index, index.entries.size()),
        "serial",
    };
}

void usearch_matrix_add(usearch_matrix_add_t* arguments) {
    if (deferred<usearch_matrix_add>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_matrix_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_matrix_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_tape_view_t const& matrices = arguments->matrices;
    if (reject_read_only(index, outcome)) return;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_scalar(arguments->scalar, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;
    if (reject_tape(matrices, count, outcome)) return;
    // A frozen height is what lets row `i` of one entry pair with row `i` of another, so an
    // entry of any other height is rejected rather than stored.
    if (index.options.rows)
        for (std::size_t row = 0; row != count; ++row)
            if (reject_if(matrices.offsets[row + 1] - matrices.offsets[row] != index.options.rows,
                          usearch_status_shape_mismatch_k, "The index froze a row count this entry does not match",
                          outcome))
                return;

    std::unique_lock writers(index.mutex);
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    matrix_payload_t payload;
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        read_matrix(matrices, row, index.options.columns, arguments->scalar, payload);
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

void usearch_matrix_refine(usearch_matrix_refine_t* arguments) {
    if (deferred<usearch_matrix_refine>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    usearch_matrix_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    // Cancelling is not an error here, the header having made a stopped sweep merely less improved.
    for (std::size_t row = 0; row != rows; ++row)
        if (!keep_going(arguments->progress, arguments->progress_state, row + 1, rows)) break;
    if (arguments->stats) arguments->stats->visited_members = rows;
}

void usearch_matrix_search(usearch_matrix_search_t* arguments) {
    if (deferred<usearch_matrix_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    usearch_matrix_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_scalar(arguments->scalar, outcome)) return;
    if (reject_metric(index, outcome)) return;
    if (reject_tape(arguments->queries, count, outcome)) return;
    if (reject_matches(arguments->matches, count, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_tape(index, arguments->queries, count, arguments->scalar, arguments->matches,
                                             arguments->expansion_factor, arguments->stats, arguments->predicate,
                                             arguments->predicate_state, arguments->progress,
                                             arguments->progress_state);
    reject_cancelled(answered, count, outcome);
}

void usearch_matrix_count(usearch_matrix_count_t* arguments) {
    if (deferred<usearch_matrix_count>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    usearch_matrix_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    auto const range = index.by_key.equal_range(keyed(arguments->key));
    std::size_t entries = 0, total_rows = 0, largest_rows = 0;
    for (auto position = range.first; position != range.second; ++position, ++entries) {
        std::size_t const height = rows_of(index, index.entries[position->second]);
        total_rows += height;
        largest_rows = std::max(largest_rows, height);
    }
    *arguments->key_stats = {entries, total_rows, largest_rows};
}

void usearch_matrix_get(usearch_matrix_get_t* arguments) {
    if (deferred<usearch_matrix_get>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->matrices = nullptr;
    *arguments->offsets = nullptr;
    if (arguments->written) *arguments->written = 0;
    if (arguments->stride) *arguments->stride = 0;
    usearch_matrix_index_handle_t& index = *arguments->index;
    usearch_scalar_t const scalar = arguments->scalar;
    if (reject_scalar(scalar, outcome)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const columns = index.options.columns;
    std::size_t const row_bytes = row_extent(scalar, columns);
    std::vector<std::size_t> const rows = rows_under(index, arguments->key);
    if (rows.empty()) return;
    std::size_t total_rows = 0;
    for (std::size_t row : rows) total_rows += rows_of(index, index.entries[row]);
    // The offsets lead, so the rows behind them start on a word boundary whatever the kind.
    std::size_t const offsets_bytes = (rows.size() + 1) * sizeof(std::size_t);
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, nullptr, outcome);
    if (!arena) return;
    std::unique_lock const held {arena->serving, std::adopt_lock};
    std::byte* const landing = arena->memory.allocate(offsets_bytes + total_rows * row_bytes, default_alignment_k);
    if (!landing) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    std::size_t* const offsets = reinterpret_cast<std::size_t*>(landing);
    std::byte* const tape = landing + offsets_bytes;
    offsets[0] = 0;
    for (std::size_t found = 0; found != rows.size(); ++found) {
        matrix_entry_t const& entry = index.entries[rows[found]];
        std::size_t const height = rows_of(index, entry);
        for (std::size_t line = 0; line != height; ++line)
            write_row(row_at<std::byte>(tape, offsets[found] + line, row_bytes, row_bytes), scalar,
                      std::span<double const>(entry.payload).subspan(line * columns, columns));
        offsets[found + 1] = offsets[found] + height;
    }
    *arguments->matrices = tape;
    *arguments->offsets = offsets;
    if (arguments->written) *arguments->written = rows.size();
    if (arguments->stride) *arguments->stride = row_bytes;
}

void usearch_matrix_merge(usearch_matrix_merge_t* arguments) {
    if (deferred<usearch_matrix_merge>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_matrix_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_matrix_index_t* const donors = arguments->donors;
    usearch_progress_callback_t const progress = arguments->progress;
    void* const progress_state = arguments->progress_state;
    std::size_t merged = 0;
    bool accommodated = true;
    while (accommodated && merged != donors_count) {
        usearch_matrix_index_handle_t& donor = *donors[merged];
        std::unique_lock writers(index.mutex);
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (matrix_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
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

void usearch_matrix_cursor_init(usearch_matrix_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    usearch_matrix_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_matrix_cursor_handle_t* const cursor = new (std::nothrow)
        usearch_matrix_cursor_handle_t {&index, 0, index.entries.size(), {}};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    *arguments->cursor = cursor;
}

void usearch_matrix_cursor_next(usearch_matrix_cursor_next_t* arguments) {
    if (deferred<usearch_matrix_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->written = 0;
    usearch_matrix_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_matrix_index_handle_t& index = *arguments->index;
    usearch_keys_span_t const& keys = arguments->keys;
    if (reject_keys_span(keys, outcome)) return;
    if (reject_if(!arguments->row_counts, usearch_status_invalid_argument_k, "The row counts must name storage",
                  outcome))
        return;
    if (reject_if(cursor.pulling.test_and_set(std::memory_order_acquire), usearch_status_contended_k,
                  "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    std::size_t const written = std::min(keys.capacity, cursor.limit - cursor.position);
    for (std::size_t row = 0; row != written; ++row) {
        matrix_entry_t const& entry = index.entries[cursor.position + row];
        row_at<usearch_key_t>(keys.data, row, keys.stride, 1).front() = entry.key;
        // Borrowed from the handle's own storage, so these are `f64` whatever the index froze.
        if (arguments->matrices) arguments->matrices[row] = entry.payload.data();
        arguments->row_counts[row] = rows_of(index, entry);
    }
    cursor.position += written;
    *arguments->written = written;
    cursor.pulling.clear(std::memory_order_release);
}

void usearch_matrix_cursor_free(usearch_matrix_cursor_t handle) { delete handle; }

void usearch_matrix_save(usearch_matrix_save_t* arguments) {
    if (deferred<usearch_matrix_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_matrix_index_handle_t& index = *arguments->index;
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

void usearch_matrix_metadata_read(usearch_matrix_metadata_read_t* arguments) {
    if (deferred<usearch_matrix_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    matrix_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    matrix_index_options_t options {};
    if (reject_if(!adopt(options, prefix), usearch_status_unknown_kind_k,
                  "The checkpoint names a metric this build lacks", outcome))
        return;
    *arguments->metadata = {options.metric, options.quantization, options.columns, options.rows, options.connectivity};
}

} // extern "C"
