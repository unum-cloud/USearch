/**
 *  @brief A brute-force stand-in for the @c usearch_dense_* C ABI, letting a language binding
 *      link and run its suite in seconds instead of building the engine.
 *
 *  @file test/stub/dense.cpp
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
 *  invisible and an @c i8 index round-trips exactly. Only six scalar kinds are implemented -
 *  @c f64, @c f32, @c bf16, @c f16, @c i8, @c u8 - and every other kind is @b rejected rather
 *  than guessed at, so a binding computing row bytes wrongly for a narrow float or a packed
 *  sub-byte kind fails here loudly instead of passing quietly. Hamming and Jaccard fall back
 *  to the angular kernel, and a custom metric callback is ignored entirely. Residency and huge
 *  pages are inert. Traversal statistics are plausible fiction: shaped correctly, invented
 *  numerically, so assert that they move, never what they equal. A checkpoint is a binary blob
 *  of this file's own memory image - not SafeGraphs, not portable across builds, and readable
 *  by nothing but the stand-in that wrote it.
 */
#include <cmath> // `std::sqrt`, `std::exp`

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/dense_index.h"

#include "lib.hpp"

namespace us = unum::usearch;
using namespace unum::usearch::stub;

namespace {

#pragma region Storage

using payload_t = std::vector<double>;
using entry_t = entry<payload_t>;

static_assert(std::is_nothrow_move_constructible_v<entry_t>, "reallocation must move rows, never copy them");

using index_t = handle<payload_t, usearch_dense_options_t>;

struct cursor_t {
    /** @brief The index being walked; it must outlive the cursor. */
    index_t* index;
    /** @brief The next row to hand out. */
    std::size_t position;
    /** @brief The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;
};

struct finder_t {
    /** @brief Every scored match, nearest first - a snapshot, so it needs no further locking. */
    std::vector<match_t> ranked;
    /** @brief The next match to hand out. */
    std::size_t position;
};

/** @brief Whether the geometry is frozen; a zeroed width makes every derived extent zero. */
bool reject_unfrozen(index_t const& index, usearch_error_t* error) {
    return reject_if(!index.options.dimensions, usearch_error_uninitialized_k, "the geometry was never frozen", error);
}

#pragma endregion Storage

#pragma region Scoring

/**
 *  @brief Scores one pair under the frozen kind. @c usearch_dense_init refuses a kind this
 *      build does not know, so every enumerator reaching here is one of the five.
 */
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
    case usearch_dense_metric_euclidean_k: return squared;
    case usearch_dense_metric_hamming_k: return (double)differing;
    case usearch_dense_metric_jaccard_k: return either ? 1.0 - (double)shared / (double)either : 0.0;
    default: {
        double const denominator = std::sqrt(first_norm * second_norm);
        return denominator > 0 ? 1.0 - dot / denominator : 1.0;
    }
    }
}

/** @brief Whether the kind is one this build implements at all. */
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

/** @brief The shared body of both search forms; the filtered one passes a predicate. */
void search_rows(index_t& index, void const* queries, std::size_t queries_count, std::size_t queries_stride,
                 usearch_scalar_t scalar_kind, std::size_t wanted, usearch_dense_expansion_config_t const* config,
                 usearch_dense_traversal_stats_t* stats, usearch_predicate_t predicate, void* predicate_state,
                 usearch_key_t* keys, std::size_t keys_stride, usearch_distance_t* distances,
                 std::size_t distances_stride, std::span<std::size_t> counts) {
    std::size_t const rows = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const query_bytes = dimensions * scalar_bytes(scalar_kind);
    // Let the expansion factor move the reported work, so a sweep can see it change.
    if (stats && config && config->expansion_factor)
        stats->computed_distances += config->expansion_factor * queries_count;

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
    }
}

#pragma endregion Scoring

#pragma region Serialization

/**
 *  @brief What @c load needs before it can read a row. The blob is this file's own memory image
 *      rather than a format: no magic, no version, and nothing but the stand-in reads it back.
 */
struct prefix_t {
    us::u64_t dimensions;
    us::u64_t connectivity;
    us::u64_t entries;
    us::u32_t metric;
    us::u32_t quantization;
    us::u32_t pad_dimensions;
};

std::size_t stored_row_bytes(std::size_t dimensions) { return sizeof(usearch_key_t) + dimensions * sizeof(double); }

/** @brief The whole blob's extent, empty when that many rows could not be addressed. */
std::optional<std::size_t> serialized_bytes(index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(prefix_t), rows, stored_row_bytes(index.options.dimensions));
}

void write_into(index_t const& index, std::size_t rows, std::span<std::byte> target) {
    prefix_t prefix {};
    prefix.dimensions = (us::u64_t)index.options.dimensions;
    prefix.connectivity = (us::u64_t)index.options.connectivity;
    prefix.entries = (us::u64_t)rows;
    prefix.metric = (us::u32_t)index.options.metric;
    prefix.quantization = (us::u32_t)index.options.quantization;
    prefix.pad_dimensions = index.options.pad_dimensions ? 1u : 0u;
    write_pod(target, prefix);

    std::span<std::byte> cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        entry_t const& entry = index.entries[row];
        std::size_t const payload_bytes = entry.payload.size() * sizeof(double);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        copy_bytes(cursor, std::as_bytes(std::span<double const>(entry.payload)));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** @brief Reads the prefix and checks the blob is long enough for the rows it claims. */
bool parse_prefix(prefix_t& prefix, std::span<std::byte const> blob, usearch_error_t* error) {
    if (blob.size() < sizeof(prefix_t)) {
        propagate_error(error, usearch_error_malformed_input_k, "the checkpoint ends early");
        return false;
    }
    prefix = read_pod<prefix_t>(blob);
    std::optional<std::size_t> const needed = blob_bytes(sizeof(prefix_t), (std::size_t)prefix.entries,
                                                         stored_row_bytes((std::size_t)prefix.dimensions));
    if (!needed) {
        propagate_error(error, usearch_error_overflow_k, "the checkpoint claims more rows than can be addressed");
        return false;
    }
    if (blob.size() < *needed) {
        propagate_error(error, usearch_error_malformed_input_k, "the checkpoint is shorter than the rows it claims");
        return false;
    }
    return true;
}

/** @brief Adopts a checkpoint's configuration, answering whether its metric is one we know. */
bool adopt(usearch_dense_options_t& options, prefix_t const& prefix) {
    usearch_dense_metric_t const metric = (usearch_dense_metric_t)prefix.metric;
    if (!known_metric(metric)) return false;
    options.dimensions = (std::size_t)prefix.dimensions;
    options.metric = metric;
    options.quantization = (usearch_scalar_t)prefix.quantization;
    options.connectivity = (std::size_t)prefix.connectivity;
    options.pad_dimensions = prefix.pad_dimensions != 0;
    return true;
}

/** @brief Replaces the contents of @p index; the caller holds both locks exclusively. */
bool read_from(index_t& index, std::span<std::byte const> blob, usearch_error_t* error) {
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return false;
    std::size_t const rows = (std::size_t)prefix.entries;
    std::size_t const dimensions = (std::size_t)prefix.dimensions;
    index.reset();
    if (!index.accommodate(rows)) {
        propagate_error(error, usearch_error_out_of_memory_k, "the checkpoint's rows could not be allocated");
        return false;
    }
    if (!adopt(index.options, prefix)) {
        propagate_error(error, usearch_error_unknown_kind_k, "the checkpoint names a metric this build lacks");
        return false;
    }

    std::span<std::byte const> cursor = blob.subspan(sizeof(prefix_t));
    std::vector<double> payload(dimensions);
    for (std::size_t row = 0; row != rows; ++row) {
        usearch_key_t const key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        copy_bytes(std::as_writable_bytes(std::span<double>(payload)), cursor.first(dimensions * sizeof(double)));
        cursor = cursor.subspan(dimensions * sizeof(double));
        index.append(key, payload);
    }
    return true;
}

#pragma endregion Serialization

#pragma region Attention

/** @brief The value matrix a key may address, past which the entry is skipped as unmapped. */
constexpr std::size_t value_rows_limit_k = 1u << 30;

/** @brief An entry's key is its row offset into the value matrix, parked in the low bytes. */
std::size_t value_offset(usearch_key_t const& key) {
    std::size_t offset = 0;
    for (unsigned byte_index = 8; byte_index != 16; ++byte_index) offset = (offset << 8) | key.bytes[byte_index];
    return offset;
}

#pragma endregion Attention

} // namespace

extern "C" {

usearch_dense_index_t usearch_dense_init(usearch_dense_options_t const* options, usearch_error_t* error) {
    propagate_success(error);
    if (!options) {
        propagate_error(error, usearch_error_malformed_input_k, "the options are null");
        return nullptr;
    }
    if (!known_metric(options->metric)) {
        propagate_error(error, usearch_error_unknown_kind_k, "the stand-in does not implement that metric");
        return nullptr;
    }
    index_t* index = new (std::nothrow) index_t();
    if (!index) {
        propagate_error(error, usearch_error_out_of_memory_k, "the handle could not be allocated");
        return nullptr;
    }
    index->options = *options;
    if (!index->options.connectivity) index->options.connectivity = 16;
    return index;
}

void usearch_dense_free(usearch_dense_index_t handle) { delete (index_t*)handle; }

void usearch_dense_reserve(usearch_dense_index_t handle, std::size_t capacity, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    // Sizing the arena moves published rows, so no reader may be walking them.
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(capacity))
        propagate_error(error, usearch_error_out_of_memory_k, "the capacity could not be reserved");
}

void usearch_dense_clear(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::unique_lock writers(index.mutex);
    index.reset();
}

std::size_t usearch_dense_size(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return ((index_t*)handle)->entries.size();
}
std::size_t usearch_dense_capacity(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::shared_lock readers(index.mutex);
    return index.entries.capacity();
}
std::size_t usearch_dense_dimensions(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return ((index_t*)handle)->options.dimensions;
}
std::size_t usearch_dense_connectivity(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return ((index_t*)handle)->options.connectivity;
}
std::size_t usearch_dense_memory_usage(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::shared_lock readers(index.mutex);
    return index.entries.capacity() * stored_row_bytes(index.options.dimensions);
}
char const* usearch_dense_capabilities([[maybe_unused]] usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return "serial";
}

std::size_t usearch_dense_add(usearch_dense_index_t handle, usearch_key_t const* keys, void const* vectors,
                              std::size_t count, std::size_t stride, usearch_scalar_t scalar_kind,
                              [[maybe_unused]] usearch_dense_expansion_config_t const* config,
                              usearch_dense_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = *(index_t*)handle;
    if (reject_unfrozen(index, error)) return 0;
    if (reject_scalar(scalar_kind, error)) return 0;

    // Sizing precedes the first append, so a batch that cannot be accommodated adds nothing -
    // the all-or-nothing insertion this modality alone promises.
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(index.entries.size() + count)) {
        propagate_error(error, usearch_error_out_of_memory_k, "the batch could not be accommodated");
        return 0;
    }

    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = dimensions * scalar_bytes(scalar_kind);
    std::vector<double> payload(dimensions);
    for (std::size_t row = 0; row != count; ++row) {
        read_row(row_at<std::byte>(vectors, row, stride, row_bytes), scalar_kind, payload);
        index.append(keys[row], payload);
    }
    if (stats) {
        stats->visited_members = count;
        stats->computed_distances = count * index.entries.size();
        stats->restarts = 1;
    }
    return count;
}

void usearch_dense_search(usearch_dense_index_t handle, void const* queries, std::size_t queries_count,
                          std::size_t queries_stride, usearch_scalar_t scalar_kind, std::size_t wanted,
                          usearch_dense_expansion_config_t const* config, usearch_dense_traversal_stats_t* stats,
                          usearch_key_t* keys, std::size_t keys_stride, usearch_distance_t* distances,
                          std::size_t distances_stride, std::size_t* counts, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = *(index_t*)handle;
    if (reject_unfrozen(index, error)) return;
    if (reject_scalar(scalar_kind, error)) return;
    std::shared_lock readers(index.mutex);
    search_rows(index, queries, queries_count, queries_stride, scalar_kind, wanted, config, stats, nullptr, nullptr,
                keys, keys_stride, distances, distances_stride, {counts, queries_count});
}

void usearch_dense_search_if(usearch_dense_index_t handle, void const* queries, std::size_t queries_count,
                             std::size_t queries_stride, usearch_scalar_t scalar_kind, std::size_t wanted,
                             usearch_dense_expansion_config_t const* config, usearch_dense_traversal_stats_t* stats,
                             usearch_predicate_t predicate, void* predicate_state, usearch_key_t* keys,
                             std::size_t keys_stride, usearch_distance_t* distances, std::size_t distances_stride,
                             std::size_t* counts, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = *(index_t*)handle;
    if (reject_unfrozen(index, error)) return;
    if (reject_scalar(scalar_kind, error)) return;
    std::shared_lock readers(index.mutex);
    search_rows(index, queries, queries_count, queries_stride, scalar_kind, wanted, config, stats, predicate,
                predicate_state, keys, keys_stride, distances, distances_stride, {counts, queries_count});
}

std::size_t usearch_dense_count(usearch_dense_index_t handle, usearch_key_t key, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::shared_lock readers(index.mutex);
    return index.by_key.count(key);
}

std::size_t usearch_dense_get(usearch_dense_index_t handle, usearch_key_t key, void* vectors, std::size_t wanted,
                              std::size_t stride, usearch_scalar_t scalar_kind, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    if (vectors && reject_scalar(scalar_kind, error)) return 0;
    std::shared_lock readers(index.mutex);
    std::size_t const row_bytes = index.options.dimensions * scalar_bytes(scalar_kind);
    std::vector<std::size_t> const rows = rows_under(index, key);
    for (std::size_t found = 0; found != rows.size() && vectors && found < wanted; ++found)
        write_row(row_at<std::byte>(vectors, found, stride, row_bytes), scalar_kind,
                  index.entries[rows[found]].payload);
    return rows.size();
}

void usearch_dense_merge(usearch_dense_index_t handle, usearch_dense_index_t donor_handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    index_t* donor = (index_t*)donor_handle;
    std::size_t const donated = donor->entries.size();
    {
        // The donor's rows land in one batch, which almost always grows the arena.
        std::unique_lock writers(index.mutex);
        if (!index.accommodate(index.entries.size() + donated)) {
            propagate_error(error, usearch_error_out_of_memory_k, "the donor's rows could not be accommodated");
            delete donor; // consumed either way - the ABI forbids the caller freeing it
            return;
        }
        for (std::size_t row = 0; row != donated; ++row) {
            entry_t const& entry = donor->entries[row];
            index.append(entry.key, entry.payload);
        }
    }
    delete donor;
}

usearch_dense_cursor_t usearch_dense_cursor(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::shared_lock readers(index.mutex);
    cursor_t* cursor = new (std::nothrow) cursor_t {&index, 0, index.entries.size()};
    if (!cursor) propagate_error(error, usearch_error_out_of_memory_k, "the cursor could not be allocated");
    return cursor;
}

std::size_t usearch_dense_cursor_next(usearch_dense_cursor_t handle, usearch_key_t* keys, void* vectors,
                                      std::size_t wanted, std::size_t stride, usearch_scalar_t scalar_kind,
                                      usearch_error_t* error) {
    propagate_success(error);
    cursor_t& cursor = *(cursor_t*)handle;
    if (vectors && reject_scalar(scalar_kind, error)) return 0;
    index_t& index = *cursor.index;
    std::shared_lock readers(index.mutex);
    std::size_t const row_bytes = index.options.dimensions * scalar_bytes(scalar_kind);
    std::size_t const written = std::min(wanted, cursor.limit - cursor.position);
    std::span<usearch_key_t> const keys_row {keys, written};
    for (std::size_t row = 0; row != written; ++row) {
        entry_t const& entry = index.entries[cursor.position + row];
        keys_row[row] = entry.key;
        if (vectors) write_row(row_at<std::byte>(vectors, row, stride, row_bytes), scalar_kind, entry.payload);
    }
    cursor.position += written;
    return written;
}

void usearch_dense_cursor_free(usearch_dense_cursor_t handle) { delete (cursor_t*)handle; }

usearch_dense_finder_t usearch_dense_find(usearch_dense_index_t handle, void const* query, usearch_scalar_t scalar_kind,
                                          usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    if (reject_unfrozen(index, error)) return nullptr;
    if (reject_scalar(scalar_kind, error)) return nullptr;
    finder_t* finder = new (std::nothrow) finder_t();
    if (!finder) {
        propagate_error(error, usearch_error_out_of_memory_k, "the finder could not be allocated");
        return nullptr;
    }
    std::size_t const dimensions = index.options.dimensions;
    std::vector<double> query_row(dimensions);
    std::shared_lock readers(index.mutex);
    read_row(row_at<std::byte>(query, 0, 0, dimensions * scalar_bytes(scalar_kind)), scalar_kind, query_row);
    rank<payload_t>(
        index.entries, admitting(nullptr, nullptr),
        [&](payload_t const& payload) { return distance(index.options.metric, query_row, payload); }, finder->ranked);
    finder->position = 0;
    return finder;
}

std::size_t usearch_dense_find_next(usearch_dense_finder_t handle, usearch_key_t* keys, usearch_distance_t* distances,
                                    std::size_t wanted, usearch_error_t* error) {
    propagate_success(error);
    finder_t& finder = *(finder_t*)handle;
    std::size_t const written = std::min(wanted, finder.ranked.size() - finder.position);
    std::span<usearch_key_t> const keys_row {keys, written};
    std::span<usearch_distance_t> const distances_row {distances, written};
    for (std::size_t match = 0; match != written; ++match) {
        keys_row[match] = finder.ranked[finder.position + match].key;
        distances_row[match] = finder.ranked[finder.position + match].distance;
    }
    finder.position += written;
    return written;
}

void usearch_dense_find_free(usearch_dense_finder_t handle) { delete (finder_t*)handle; }

std::size_t usearch_dense_serialized_length(usearch_dense_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::shared_lock readers(index.mutex);
    std::optional<std::size_t> const length = serialized_bytes(index, index.entries.size());
    if (!length) {
        propagate_error(error, usearch_error_overflow_k, "the checkpoint would not fit an addressable buffer");
        return 0;
    }
    return *length;
}

void usearch_dense_save(usearch_dense_index_t handle, usearch_string_t path, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    // Saving is a reader: it demands only that nothing move or erase rows under it.
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (!length)
        return propagate_error(error, usearch_error_overflow_k, "the checkpoint would not fit an addressable buffer");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    write_file(path, blob, error);
}

void usearch_dense_load(usearch_dense_index_t handle, usearch_string_t path, usearch_error_t* error) {
    propagate_success(error);
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    index_t& index = *(index_t*)handle;
    std::unique_lock writers(index.mutex);
    read_from(index, blob, error);
}

void usearch_dense_metadata(usearch_string_t path, usearch_dense_options_t* options, usearch_error_t* error) {
    propagate_success(error);
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return;
    if (!adopt(*options, prefix))
        propagate_error(error, usearch_error_unknown_kind_k, "the checkpoint names a metric this build lacks");
}

void usearch_dense_save_buffer(usearch_dense_index_t handle, void* buffer, std::size_t length, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const needed = serialized_bytes(index, rows);
    if (!needed)
        return propagate_error(error, usearch_error_overflow_k, "the checkpoint would not fit an addressable buffer");
    if (length < *needed)
        return propagate_error(error, usearch_error_shape_mismatch_k,
                               "the buffer is smaller than the serialized length");
    write_into(index, rows, {(std::byte*)buffer, length});
}

void usearch_dense_load_buffer(usearch_dense_index_t handle, void const* buffer, std::size_t length,
                               usearch_error_t* error) {
    propagate_success(error);
    index_t& index = *(index_t*)handle;
    std::unique_lock writers(index.mutex);
    read_from(index, {(std::byte const*)buffer, length}, error);
}

void usearch_dense_metadata_buffer(void const* buffer, std::size_t length, usearch_dense_options_t* options,
                                   usearch_error_t* error) {
    propagate_success(error);
    prefix_t prefix {};
    if (!parse_prefix(prefix, {(std::byte const*)buffer, length}, error)) return;
    if (!adopt(*options, prefix))
        propagate_error(error, usearch_error_unknown_kind_k, "the checkpoint names a metric this build lacks");
}

void usearch_dense_multiply(usearch_dense_index_t handle, void const* rows, std::size_t rows_count,
                            std::size_t rows_stride, usearch_scalar_t scalar_kind, std::size_t wanted,
                            [[maybe_unused]] usearch_dense_expansion_config_t const* config,
                            usearch_dense_traversal_stats_t* stats, usearch_key_t* columns, std::size_t columns_stride,
                            usearch_distance_t* products, std::size_t products_stride, std::size_t* counts,
                            usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = *(index_t*)handle;
    if (reject_unfrozen(index, error)) return;
    if (reject_scalar(scalar_kind, error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const row_bytes = dimensions * scalar_bytes(scalar_kind);
    std::span<std::size_t> const counts_column {counts, rows_count};

    std::vector<double> left(dimensions);
    for (std::size_t row = 0; row != rows_count; ++row) {
        read_row(row_at<std::byte>(rows, row, rows_stride, row_bytes), scalar_kind, left);
        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(columns, row, columns_stride, wanted);
        std::span<usearch_distance_t> const products_row = row_at<usearch_distance_t>(products, row, products_stride,
                                                                                      wanted);
        std::size_t const written = std::min(wanted, indexed);
        for (std::size_t column = 0; column != written; ++column) {
            entry_t const& entry = index.entries[column];
            keys_row[column] = entry.key;
            products_row[column] = dot_product(left, entry.payload);
        }
        // The true count, which the header allows to exceed `wanted` to signal truncation.
        counts_column[row] = indexed;
        if (stats) stats->computed_distances += indexed;
    }
}

void usearch_dense_attend(usearch_dense_index_t handle, void const* queries, std::size_t queries_count,
                          std::size_t queries_stride, usearch_scalar_t queries_scalar_kind, void const* values,
                          std::size_t value_dimensions, std::size_t values_stride, usearch_scalar_t values_scalar_kind,
                          usearch_dense_attention_config_t const* config, usearch_dense_traversal_stats_t* stats,
                          void* outputs, std::size_t outputs_stride, usearch_scalar_t outputs_scalar_kind,
                          float* maximum_scores, float* partitions, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = *(index_t*)handle;
    if (reject_unfrozen(index, error)) return;
    if (reject_scalar(queries_scalar_kind, error)) return;
    if (reject_scalar(values_scalar_kind, error)) return;
    if (reject_scalar(outputs_scalar_kind, error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const indexed = index.entries.size();
    std::size_t const dimensions = index.options.dimensions;
    std::size_t const query_bytes = dimensions * scalar_bytes(queries_scalar_kind);
    std::size_t const value_bytes = value_dimensions * scalar_bytes(values_scalar_kind);
    std::size_t const output_bytes = value_dimensions * scalar_bytes(outputs_scalar_kind);
    double const scale = config && config->score_scale > 0 ? config->score_scale
                                                           : 1.0 / std::sqrt(std::max<double>(dimensions, 1));

    std::vector<double> query_row(dimensions), value(value_dimensions), accumulated(value_dimensions);
    for (std::size_t query = 0; query != queries_count; ++query) {
        read_row(row_at<std::byte>(queries, query, queries_stride, query_bytes), queries_scalar_kind, query_row);
        // The online-softmax recurrence: a running maximum, a partition, and a weighted sum.
        double maximum = -INFINITY, partition = 0;
        std::fill(accumulated.begin(), accumulated.end(), 0.0);
        for (std::size_t member = 0; member != indexed; ++member) {
            entry_t const& entry = index.entries[member];
            std::size_t const offset = value_offset(entry.key);
            if (offset >= value_rows_limit_k) continue;
            double const score = dot_product(query_row, entry.payload) * scale;
            read_row(row_at<std::byte>(values, offset, values_stride, value_bytes), values_scalar_kind, value);

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
        write_row(row_at<std::byte>(outputs, query, outputs_stride, output_bytes), outputs_scalar_kind, accumulated);
        if (maximum_scores) maximum_scores[query] = (float)maximum;
        if (partitions) partitions[query] = (float)partition;
        if (stats) stats->computed_distances += indexed;
    }
}

} // extern "C"
