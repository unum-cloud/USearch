/**
 *  @brief A brute-force stand-in for the @c usearch_matrix_* C ABI, scoring every stored matrix
 *      against every query so a binding can be exercised without the engine.
 *
 *  @file src_stub/matrix.cpp
 *
 *  One entry is a rank-2 tensor whose height varies and whose width is frozen index-wide, so
 *  batches arrive as a tape - one array of base pointers beside one of row counts - rather than
 *  a strided matrix. That pairing is what this stand-in is for: a binding that flattens the two
 *  into a single buffer, or that derives a height from a stride, fails here immediately.
 *
 *  @section matrix_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  Only the two row-to-row kinds are implemented - Chamfer and Hausdorff - and the transport and
 *  superposition kinds are @b rejected rather than approximated by them, so a binding asking for
 *  Gromov-Wasserstein learns that here instead of receiving a plausible wrong number. Rows are
 *  stored as @c double and @c quantization is ignored, so the matrices a cursor borrows are
 *  @c f64 whatever the index froze. Refinement keeps no edges to re-select, so its return is a
 *  constant zero rather than a settling signal, and a caller must not read it as one. Traversal
 *  statistics are plausible fiction: shaped correctly, invented numerically. A checkpoint is a
 *  binary blob of this file's own memory image, readable by nothing else.
 */
#include <cmath> // `std::sqrt`

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/matrix_index.h"

#include "lib.hpp"

namespace us = unum::usearch;
using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace {

#pragma region Storage

using payload_t = std::vector<double>;
using entry_t = entry<payload_t>;

static_assert(std::is_nothrow_move_constructible_v<entry_t>, "reallocation must move rows, never copy them");

using index_t = handle<payload_t, usearch_matrix_options_t>;

struct cursor_t {
    /** @brief The index being walked; it must outlive the cursor. */
    index_t* index;
    /** @brief The next entry to hand out. */
    std::size_t position;
    /** @brief The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;
};

/** @brief Whether the width is frozen; a zeroed one makes every derived extent zero. */
bool reject_unfrozen(index_t const& index, usearch_error_t* error) {
    return reject_if(!index.options.columns, usearch_error_uninitialized_k, "The geometry was never frozen", error);
}

/** @brief Whether this stand-in scores the kind, or rejects it rather than approximating it. */
bool reject_metric(index_t const& index, usearch_error_t* error) {
    bool const scored = index.options.metric == usearch_matrix_metric_chamfer_k ||
                        index.options.metric == usearch_matrix_metric_hausdorff_k;
    return reject_if(!scored, usearch_error_unknown_kind_k, "The stand-in implements chamfer and hausdorff only",
                     error);
}

/** @brief The height of a stored entry, which its payload's extent and the frozen width fix. */
std::size_t rows_of(index_t const& index, entry_t const& entry) { return entry.payload.size() / index.options.columns; }

#pragma endregion Storage

#pragma region Scoring

double row_distance(std::span<double const> first, std::span<double const> second) {
    double squared = 0;
    for (std::size_t column = 0; column != first.size(); ++column) {
        double const delta = first[column] - second[column];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

/**
 *  @brief Every row of @p from finds its nearest partner in @p onto; the two kinds differ only
 *      in how those distances are aggregated, so both come off this one sweep.
 */
void nearest_partners(std::span<double const> from, std::span<double const> onto, std::size_t columns, double& summed,
                      double& worst) {
    summed = 0, worst = 0;
    std::size_t const from_rows = from.size() / columns, onto_rows = onto.size() / columns;
    for (std::size_t row = 0; row != from_rows; ++row) {
        double nearest = INFINITY;
        for (std::size_t partner = 0; partner != onto_rows; ++partner)
            nearest = std::min(
                nearest, row_distance(from.subspan(row * columns, columns), onto.subspan(partner * columns, columns)));
        if (nearest == INFINITY) nearest = 0;
        summed += nearest;
        worst = std::max(worst, nearest);
    }
    if (from_rows) summed /= static_cast<double>(from_rows);
}

/**
 *  @brief Scores one pair under the frozen kind. @c usearch_matrix_init refuses a kind this
 *      build does not implement, so only the two best-partner kinds reach here.
 */
double distance(index_t const& index, std::span<double const> query, std::span<double const> stored) {
    std::size_t const columns = index.options.columns;
    double forward_mean = 0, forward_worst = 0, reverse_mean = 0, reverse_worst = 0;
    nearest_partners(query, stored, columns, forward_mean, forward_worst);
    nearest_partners(stored, query, columns, reverse_mean, reverse_worst);
    if (index.options.metric == usearch_matrix_metric_chamfer_k) return forward_mean + reverse_mean;
    return std::max(forward_worst, reverse_worst);
}

/**
 *  @brief Whether the kind is one this build implements. The transport and superposition kinds
 *      need a solver and an SVD, which a brute-force stand-in deliberately does not carry.
 */
bool known_metric(usearch_matrix_metric_t metric) {
    return metric == usearch_matrix_metric_chamfer_k || metric == usearch_matrix_metric_hausdorff_k;
}

/** @brief Widens one tape entry into doubles, sized by its own row count. */
void read_matrix(void const* base, std::size_t row_count, std::size_t columns, usearch_scalar_t scalar_kind,
                 std::vector<double>& values) {
    values.resize(row_count * columns);
    std::size_t const row_bytes = row_extent(scalar_kind, columns);
    for (std::size_t row = 0; row != row_count; ++row)
        read_row(row_at<std::byte>(base, row, row_bytes, row_bytes), scalar_kind,
                 std::span<double>(values).subspan(row * columns, columns));
}

/**
 *  @brief The shared body of both search forms; the filtered one passes a predicate.
 *  @return Queries answered, short of @p counts.size() exactly when the handler cancelled.
 */
std::size_t search_tape(index_t& index, void const* const* queries, std::size_t const* row_counts,
                        usearch_scalar_t scalar_kind, std::size_t wanted,
                        usearch_matrix_search_options_t const* options, usearch_matrix_traversal_stats_t* stats,
                        usearch_predicate_t predicate, void* predicate_state, usearch_progress_callback_t progress,
                        void* progress_state, usearch_key_t* keys, std::size_t keys_stride,
                        usearch_distance_t* distances, std::size_t distances_stride, std::span<std::size_t> counts) {
    std::size_t const rows = index.entries.size();
    std::size_t const columns = index.options.columns;
    if (stats && options && options->expansion_factor)
        stats->computed_distances += options->expansion_factor * counts.size();

    std::fill(counts.begin(), counts.end(), 0);
    std::vector<double> query_matrix;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        read_matrix(queries[query], row_counts[query], columns, scalar_kind, query_matrix);
        rank<payload_t>(
            std::span<entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
            [&](payload_t const& payload) { return distance(index, query_matrix, payload); }, ranked);
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
 */
struct prefix_t {
    /** @brief The frozen row width shared by every entry. */
    std::size_t columns;
    /** @brief The frozen row count, or zero where heights are ragged. */
    std::size_t rows;
    /** @brief The connections-per-node bound the handle was built with. */
    std::size_t connectivity;
    /** @brief How many entries follow the prefix. */
    std::size_t entries;
    /** @brief Bytes of entry payload that follow, keys and row counts included. */
    std::size_t payload_bytes;
    /** @brief What entries are compared by. */
    usearch_matrix_metric_t metric;
};

/** @brief The per-entry overhead of the blob: the key, then the row count that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(us::u64_t);

std::size_t payload_extent(index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row)
        extent += record_overhead_k + index.entries[row].payload.size() * sizeof(double);
    return extent;
}

std::optional<std::size_t> serialized_bytes(index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(prefix_t), payload_extent(index, rows), 1);
}

void write_into(index_t const& index, std::size_t rows, bytes_span_t target) {
    prefix_t prefix {};
    prefix.columns = index.options.columns;
    prefix.rows = index.options.rows;
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.metric = index.options.metric;
    write_pod(target, prefix);

    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        entry_t const& entry = index.entries[row];
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

/** @brief Reads the prefix and checks the blob is long enough for what it claims. */
bool parse_prefix(prefix_t& prefix, bytes_view_t blob, usearch_error_t* error) {
    if (reject_if(blob.size() < sizeof(prefix_t), usearch_error_malformed_input_k, "The checkpoint ends early", error))
        return false;
    prefix = read_pod<prefix_t>(blob);
    return fits_blob(blob, sizeof(prefix_t), prefix.payload_bytes, 1, error);
}

/** @brief Adopts a checkpoint's configuration, answering whether its metric is one we implement. */
bool adopt(usearch_matrix_options_t& options, prefix_t const& prefix) {
    if (!known_metric(prefix.metric)) return false;
    options.columns = prefix.columns;
    options.rows = prefix.rows;
    options.connectivity = prefix.connectivity;
    options.metric = prefix.metric;
    return true;
}

/** @brief Replaces the contents of @p index; the caller holds both locks exclusively. */
bool read_from(index_t& index, bytes_view_t blob, usearch_error_t* error) {
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return false;
    std::size_t const entries = prefix.entries;
    index.reset();
    if (!index.accommodate(entries)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The checkpoint's entries could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
        return false;
    }

    bytes_view_t cursor = blob.subspan(sizeof(prefix_t));
    payload_t payload;
    for (std::size_t entry_index = 0; entry_index != entries; ++entry_index) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_error_malformed_input_k, error, "The checkpoint ends inside an entry");
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
            propagate_error(usearch_error_malformed_input_k, error, "The checkpoint ends inside an entry");
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

#pragma endregion Serialization

} // namespace

extern "C" {

usearch_matrix_index_t usearch_matrix_init(usearch_matrix_options_t const* options, usearch_error_t* error) {
    propagate_success(error);
    if (!options) {
        propagate_error(usearch_error_malformed_input_k, error, "The options are null");
        return nullptr;
    }
    if (!known_metric(options->metric)) {
        propagate_error(usearch_error_unknown_kind_k, error, "The stand-in implements chamfer and hausdorff only");
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

void usearch_matrix_free(usearch_matrix_index_t handle) { delete static_cast<index_t*>(handle); }

void usearch_matrix_reserve(usearch_matrix_index_t handle, std::size_t entries, [[maybe_unused]] std::size_t rows,
                            usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(entries))
        propagate_error(usearch_error_out_of_memory_k, error, "The capacity could not be reserved");
}

void usearch_matrix_clear(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    index.reset();
}

std::size_t usearch_matrix_size(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    return index.entries.size();
}
std::size_t usearch_matrix_capacity(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return index.entries.capacity();
}
std::size_t usearch_matrix_columns(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).options.columns;
}
std::size_t usearch_matrix_rows(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    // The header asks for the rows stored rather than the height frozen, which a ragged index has
    // none of - the two coincide only where every entry shares one height.
    std::shared_lock readers(index.mutex);
    std::size_t stored = 0;
    for (entry_t const& entry : index.entries) stored += rows_of(index, entry);
    return stored;
}
std::size_t usearch_matrix_connectivity(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).options.connectivity;
}
std::size_t usearch_matrix_memory_usage(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return payload_extent(index, index.entries.size());
}
char const* usearch_matrix_capabilities([[maybe_unused]] usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return "serial";
}

std::size_t usearch_matrix_add(usearch_matrix_index_t handle, usearch_executor_t executor, usearch_keys_view_t keys,
                               void const* const* matrices, std::size_t const* row_counts, usearch_scalar_t scalar_kind,
                               [[maybe_unused]] usearch_matrix_insertion_options_t const* options,
                               usearch_progress_callback_t progress, void* progress_state,
                               usearch_matrix_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    if (reject_unfrozen(index, error)) return 0;
    if (reject_scalar(scalar_kind, error)) return 0;
    if (reject_keys(keys, error)) return 0;
    if (reject_if(!matrices || !row_counts, usearch_error_malformed_input_k, "The tape must name storage", error))
        return 0;
    std::size_t const count = keys.count;
    // A frozen height is what lets row `i` of one entry pair with row `i` of another, so an
    // entry of any other height is rejected rather than stored.
    if (index.options.rows)
        for (std::size_t row = 0; row != count; ++row)
            if (row_counts[row] != index.options.rows) {
                propagate_error(usearch_error_shape_mismatch_k, error,
                                "The index froze a row count this entry does not match");
                return 0;
            }

    std::unique_lock writers(index.mutex);
    if (reject_oversubscribed(index, keys, index.options.values_per_key, error)) return 0;
    if (!index.accommodate(index.entries.size() + count)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The batch could not be accommodated");
        return 0;
    }

    std::size_t const watermark = index.entries.size();
    payload_t payload;
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        read_matrix(matrices[row], row_counts[row], index.options.columns, scalar_kind, payload);
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

std::size_t usearch_matrix_refine(usearch_matrix_index_t handle, usearch_executor_t executor,
                                  [[maybe_unused]] usearch_matrix_insertion_options_t const* options,
                                  usearch_progress_callback_t progress, void* progress_state,
                                  usearch_matrix_traversal_stats_t* stats, usearch_error_t* error) {
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
void search_matrix_tape(usearch_matrix_index_t handle, usearch_executor_t executor, void const* const* queries,
                        std::size_t const* row_counts, std::size_t queries_count, usearch_scalar_t scalar_kind,
                        usearch_matches_span_t const* matches, usearch_matrix_search_options_t const* options,
                        usearch_predicate_t predicate, void* predicate_state, usearch_progress_callback_t progress,
                        void* progress_state, usearch_matrix_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    if (reject_scalar(scalar_kind, error)) return;
    if (reject_metric(index, error)) return;
    if (reject_if(!queries || !row_counts, usearch_error_malformed_input_k, "The tape must name storage", error))
        return;
    std::size_t keys_stride = 0, scores_stride = 0;
    if (reject_matches(matches, keys_stride, scores_stride, error)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_tape(
        index, queries, row_counts, scalar_kind, matches->wanted, options, stats, predicate, predicate_state, progress,
        progress_state, matches->keys, keys_stride, matches->scores, scores_stride, {matches->counts, queries_count});
    reject_cancelled(answered, queries_count, error);
}

void usearch_matrix_search(usearch_matrix_index_t handle, usearch_executor_t executor, void const* const* queries,
                           std::size_t const* row_counts, std::size_t queries_count, usearch_scalar_t scalar_kind,
                           usearch_matches_span_t const* matches, usearch_matrix_search_options_t const* options,
                           usearch_progress_callback_t progress, void* progress_state,
                           usearch_matrix_traversal_stats_t* stats, usearch_error_t* error) {
    search_matrix_tape(handle, executor, queries, row_counts, queries_count, scalar_kind, matches, options, nullptr,
                       nullptr, progress, progress_state, stats, error);
}

void usearch_matrix_search_if(usearch_matrix_index_t handle, usearch_executor_t executor, void const* const* queries,
                              std::size_t const* row_counts, std::size_t queries_count, usearch_scalar_t scalar_kind,
                              usearch_matches_span_t const* matches, usearch_matrix_search_options_t const* options,
                              usearch_predicate_t predicate, void* predicate_state,
                              usearch_progress_callback_t progress, void* progress_state,
                              usearch_matrix_traversal_stats_t* stats, usearch_error_t* error) {
    search_matrix_tape(handle, executor, queries, row_counts, queries_count, scalar_kind, matches, options, predicate,
                       predicate_state, progress, progress_state, stats, error);
}

std::size_t usearch_matrix_count(usearch_matrix_index_t handle, usearch_key_t key, usearch_matrix_key_stats_t* stats,
                                 usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const rows = rows_under(index, key);
    std::size_t total_rows = 0, largest_rows = 0;
    for (std::size_t row : rows) {
        std::size_t const height = rows_of(index, index.entries[row]);
        total_rows += height;
        largest_rows = std::max(largest_rows, height);
    }
    if (stats) {
        stats->entries = rows.size();
        stats->total_rows = total_rows;
        stats->largest_rows = largest_rows;
    }
    return rows.size();
}

std::size_t usearch_matrix_get(usearch_matrix_index_t handle, usearch_key_t key, void* matrices, std::size_t wanted,
                               std::size_t stride, std::size_t* row_counts, usearch_scalar_t scalar_kind,
                               usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (matrices && reject_scalar(scalar_kind, error)) return 0;
    std::shared_lock readers(index.mutex);
    std::size_t const columns = index.options.columns;
    std::size_t const row_bytes = row_extent(scalar_kind, columns);
    std::vector<std::size_t> const rows = rows_under(index, key);
    for (std::size_t found = 0; found != rows.size() && found < wanted; ++found) {
        entry_t const& entry = index.entries[rows[found]];
        std::size_t const height = rows_of(index, entry);
        if (matrices) {
            void* target = static_cast<std::byte*>(matrices) + found * stride;
            for (std::size_t line = 0; line != height; ++line)
                write_row(row_at<std::byte>(target, line, row_bytes, row_bytes), scalar_kind,
                          std::span<double const>(entry.payload).subspan(line * columns, columns));
        }
        if (row_counts) row_counts[found] = height;
    }
    return rows.size();
}

void usearch_matrix_merge(usearch_matrix_index_t handle, usearch_executor_t executor,
                          usearch_matrix_index_t const* donors, std::size_t donors_count,
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

usearch_matrix_cursor_t usearch_matrix_cursor(usearch_matrix_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    cursor_t* cursor = new (std::nothrow) cursor_t {&index, 0, index.entries.size()};
    if (!cursor) propagate_error(usearch_error_out_of_memory_k, error, "The cursor could not be allocated");
    return cursor;
}

std::size_t usearch_matrix_cursor_next(usearch_matrix_cursor_t handle, usearch_keys_span_t keys, void const** matrices,
                                       std::size_t* row_counts, usearch_error_t* error) {
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
        // Borrowed from the arena's own storage, so these are `f64` whatever the index froze.
        if (matrices) matrices[row] = entry.payload.data();
        if (row_counts) row_counts[row] = rows_of(index, entry);
    }
    cursor.position += written;
    return written;
}

void usearch_matrix_cursor_free(usearch_matrix_cursor_t handle) { delete static_cast<cursor_t*>(handle); }

std::size_t usearch_matrix_storage_requirements(usearch_matrix_index_t handle, usearch_error_t* error) {
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

void usearch_matrix_save(usearch_matrix_index_t handle, usearch_executor_t executor, usearch_bytes_view_t path,
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

void usearch_matrix_load(usearch_matrix_index_t handle, usearch_executor_t executor, usearch_bytes_view_t path,
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

void usearch_matrix_metadata(usearch_bytes_view_t path, usearch_matrix_options_t* options, usearch_error_t* error) {
    propagate_success(error);
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return;
    if (!adopt(*options, prefix))
        propagate_error(usearch_error_unknown_kind_k, error, "The checkpoint names a metric this build lacks");
}

} // extern "C"
