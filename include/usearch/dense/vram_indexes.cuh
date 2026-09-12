/**
 *  @brief The sharded index over GPU-resident dense shards - one per device - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/vram_indexes.cuh
 *  @date July 23, 2026
 *
 *  The multi-shard layer over @c dense_vram_index: same-geometry shards, each owned and pinned
 *  to one CUDA device, viewed as one index. Insertions rotate across the shards so they fill
 *  evenly and each holds a representative sample; searches scatter over every shard and gather
 *  through @ref merge_search_results, resolving matches into the caller's keys, so shards with
 *  colliding ID spaces land in one shared ranking. Checkpoints ride the SafeGraphs interchange -
 *  one consolidated monolith, or one per shard beside a manifest - written positionally with
 *  every header last, so a failed save reads as absent.
 *
 *  Beyond the executor's claim, two mechanisms guard the devices: the insertion gate admits one
 *  mutating batch at a time, and a shard's kept failure refuses every later mutation.
 */
#pragma once

#include <cstring> // `std::memcmp` - the monolith's magic

#include <array>       // `std::array` - the monolith magic
#include <atomic>      // `std::atomic` - the admission counter
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <mutex>       // `std::lock_guard`, `std::adopt_lock` - the one-batch-at-a-time hold
#include <span>        // `std::span` - the caller's keys & the merged ranking
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move` - handing a mapping to the buffer that outlives the load

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the shard table & key planes
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the host planes' placement
#include "usearch/utilities/ram_tables.hpp"        // `growing_multimap` - the reverse key plane
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths & manifests - the checkpoints
#include "usearch/dense/vram_index.cuh"

namespace unum {
namespace usearch {

#if defined(__CUDACC__)

/**
 *  @brief The sharded index: same-geometry GPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are
 *  expressible. Shards arrive one at a time through @ref emplace, so the index grows as devices
 *  are engaged rather than being declared whole. The shard's configs, views and results are
 *  re-exported verbatim, so a batched call and a single one read the same.
 */
template <                                               //
    typename key_type_ = uuid_t,                         //
    nk_dtype_t vector_dtype_ = nk_i8_k,                  //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k, //
    cuda_arch_t arch_ceiling_ = cuda_arch_blackwell_k,   //
    typename key_hash_type_ = std::hash<key_type_>       //
    >
struct dense_vram_indexes {
    /** @brief The caller-visible identity, opaque to every shard - indexes store no keys, so
     *      the whole key vocabulary begins here. */
    using key_t = key_type_;

    /** @brief Where one entry lives: the shard, and its shard-local ID - the index's only
     *      address for an entry. */
    struct key_locator_t {
        /** @brief The shard holding the entry, positional in placement order. */
        std::size_t shard = 0;
        /** @brief The entry's ID inside that shard's own space. */
        dense_node_id_t node = 0;
    };

    /** @brief Keys of one shard's rows, indexed by that shard's own node IDs; host-resident. */
    using keys_plane_t = growing_buffer<key_t, forkunion_allocator_t>;
    /** @brief Key to every entry stored under it; insert-only, wait-free for readers. */
    using locators_plane_t = growing_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t>;

    /** @brief The shard type - geometry parameters in the shard's own order, so the two can
     *      never disagree. */
    using index_t = dense_vram_index<vector_dtype_, kernel_kind_, arch_ceiling_>;

    /** @brief One shard: the graph on its device, the keys of its rows on the host, and the
     *      batch in flight on it. Placement is not repeated - the index reports it through
     *      @c device_specs. */
    struct shard_t {
        /** @brief The graph; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;
        /** @brief The key of each of this shard's rows, indexed by node ID. */
        keys_plane_t keys;
        /** @brief The batch submitted on this shard, live until the fan-out drains - a fan-out
         *      submits on every device before awaiting any. */
        typename index_t::insert_result_t inserted;
    };

    /** @brief The streams and their scratch every bulk verb runs on; it names no index, so one
     *      executor serves every instantiation. */
    using executor_t = cuda_executor_t;
    using traversal_t = typename index_t::dense_traversal_context_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;

    /** @brief The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    /** @brief The distance kernel this instantiation walks with. */
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    /** @brief The generation ceiling the kernel kits compiled up to. */
    inline static constexpr cuda_arch_t arch_ceiling_k = arch_ceiling_;

#pragma region Placement

    /** @brief Every index names the host allocator its shard table, key planes, reverse plane and
     *      checkpoint catalogs sit on; nothing is placed by default. */
    dense_vram_indexes() = delete;
    explicit dense_vram_indexes(forkunion_allocator_t allocator) noexcept
        : shards_(allocator), locators_of_key_(allocator), allocator_(allocator) {}
    dense_vram_indexes(dense_vram_indexes&& other) noexcept
        : shards_(other.allocator_), locators_of_key_(other.allocator_), allocator_(other.allocator_) {
        swap(other);
    }
    dense_vram_indexes& operator=(dense_vram_indexes&& other) noexcept { return swap(other), *this; }
    dense_vram_indexes(dense_vram_indexes const&) = delete;
    dense_vram_indexes& operator=(dense_vram_indexes const&) = delete;
    /** @brief Every member is RAII, so the shards die with the index and nothing leaks. */
    ~dense_vram_indexes() noexcept = default;

    /** @brief Exchanges two indexes whole; a move implies the caller holds both exclusively, so
     *      the insertion gate is not swapped along with the state it guards. */
    void swap(dense_vram_indexes& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::size_t const admitted = admitted_rows_.load(std::memory_order_relaxed);
        admitted_rows_.store(other.admitted_rows_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.admitted_rows_.store(admitted, std::memory_order_relaxed);
    }

    /** @brief Appends one shard built here on @p device, for callers with none to hand over. */
    status_t emplace(int device) noexcept {
        std::unique_ptr<index_t> shard = build_(device);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** @brief Appends one shard the caller already built, consuming it; refuses any whose frozen
     *      geometry differs from the shards already here, any already reserved, and any whose
     *      device already holds one - the index shards, never replicates. Refused once storage
     *      exists - @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (shard->capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        for (shard_t const& one : shards_)
            if (one.index->device_specs().ordinal == shard->device_specs().ordinal) return status_t::config_mismatch_k;
        return shards_.push_back(shard_t {std::move(shard), keys_plane_t {allocator_}, {}});
    }

    /** @brief Empties every shard in place, keeping its device, geometry and capacity; a kept
     *      failure drops with the rows. Demands the same quiescence as @ref reserve. */
    status_t clear() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = build_(one.index->device_specs().ordinal);
            if (!fresh) return status_t::bad_alloc_k;
            one.index = std::move(fresh);
            one.keys.clear();
        }
        locators_of_key_.clear();
        admitted_rows_.store(0, std::memory_order_relaxed);
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

#pragma endregion Placement

#pragma region Measures

    /** @brief Shards in the index, one per engaged device. */
    std::size_t shards_count() const noexcept { return shards_.size(); }
    /** @brief The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }
    /** @brief The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** @brief The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }
    /** @brief Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** @brief Entries across every shard - each shard's own @c size, summed. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->size();
        return total;
    }
    /** @brief Rows the admission gate can let in - the key planes' total, sized once at @ref reserve. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.keys.size();
        return total;
    }
    /** @brief The first shard's kept failure, or success; asked of the shards, never cached. */
    status_t status() const noexcept {
        for (shard_t const& one : shards_)
            if (status_t const kept = one.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }
    /** @brief Bytes the index occupies - every shard's storage, its key plane, and the shared
     *      reverse plane; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(dense_vram_indexes) + locators_of_key_.memory_usage();
        for (shard_t const& one : shards_) bytes += one.index->memory_usage() + one.keys.memory_usage();
        return bytes;
    }

#pragma endregion Measures

#pragma region Reservation

    /**
     *  @brief Reserves @p config.capacity rows, split evenly over the shards.
     *
     *  Sizes once. A repeat that already fits is a no-op, and a larger one is refused: the key
     *  planes and the reverse plane both demand a mutation-free phase to grow, and a search
     *  resolves a ranking through them while rows are still landing.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (shards_.empty()) return {0, status_t::uninitialized_k};
        if (std::size_t const held = capacity()) {
            if (config.dimensions != dimensions() || config.connectivity != connectivity())
                return {held, status_t::config_mismatch_k};
            return {held, config.capacity <= held ? status_t::success_k : status_t::config_mismatch_k};
        }
        reserve_capacity_t per_shard = config;
        per_shard.capacity = divide_round_up(config.capacity, shards_.size());

        // Key planes never grow: a search resolves through them while rows land.
        for (shard_t& one : shards_) {
            if (status_t const reserved = one.index->reserve(per_shard).status; failed(reserved)) return {0, reserved};
            if (failed(one.keys.resize(per_shard.capacity))) return {0, status_t::bad_alloc_k};
        }
        // The quota covers every admissible row, so recording a locator is infallible by construction.
        if (status_t const indexed = locators_of_key_.reserve(capacity()); failed(indexed)) return {0, indexed};
        return {capacity(), status_t::success_k};
    }

#pragma endregion Reservation

#pragma region Insertion

    /**
     *  @brief Inserts a batch under @p keys, dealing rows round-robin across the shards from a
     *      rotating start, so they fill evenly and each sees the whole distribution.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row landed in
     *  is the index's business. The keys are what the caller gets its rows back by.
     *
     *  A shard failing past its claim keeps that failure, since the shards already in flight
     *  cannot be un-claimed; a first ID other than the predicted one names a claimer outside
     *  this index, and refuses the call as @c contended_k.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_vectors The batch, one row per entry, in the shards' stored scalar type.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    insert_result_t add(std::span<key_t const> keys, vectors_view_t new_vectors, executor_t& executor,
                        insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (status_t const kept = status(); failed(kept)) return result.status = kept, result;
        if (keys.size() != new_vectors.extent(0)) return result.status = status_t::config_mismatch_k, result;
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k)
            return result.status = paired, result;
        if (!new_vectors.extent(0)) return result;
        // Two holds: the executor's guards the scratch, the index's keeps batches from interleaving claims.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return result.status = claimed, result;
        if (!mutating_.lock_if_free()) return result.status = status_t::contended_k, result;
        std::lock_guard const mutating {mutating_, std::adopt_lock};
        return add_gated_(keys, new_vectors, executor, config);
    }

    /** @brief Inserts one row under @p key into whichever shard the rotation names next - the
     *      one-row sibling of the batched form, priced identically: one launch on one device. */
    insert_result_t add(key_t const& key, vector_view_t new_vector, executor_t& executor,
                        insert_config_t config = {}) noexcept {
        if (!new_vector.is_contiguous()) return {status_t::unsupported_layout_k};
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        return add(std::span<key_t const> {&key, 1u}, row, executor, config);
    }

#pragma endregion Insertion

#pragma region Search

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending
     *      by distance - the end-to-end multi-shard lookup.
     *
     *  Takes the executor rather than one context, unlike its CPU counterpart: a context is
     *  pinned to the device its stream was opened on, so no single one can walk every shard.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[inout] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' key planes, so colliding ID spaces still land in one ranking.
     *  @param[inout] distances Distances of the same ranking, parallel to @p keys.
     *
     *  A failing shard stops the scatter but keeps whatever earlier shards already merged, so
     *  the reported count and the ranking always agree.
     */
    search_result_t search(vector_view_t query_vector, executor_t& executor, search_config_t config,
                           std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        search_result_t merged;
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k)
            return merged.status = paired, merged;
        // The executor serializes, never the index: a second caller with its own executor searches concurrently.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return merged.status = claimed, merged;
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const readied = reserve_traversals_(traversals, executor, 1, config); failed(readied))
            return merged.status = readied, merged;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            traversal_t& traversal = traversals[shard];
            search_result_t const found = shards_[shard].index->search(query_vector, traversal, config);
            if (!found) {
                merged.status = found.status;
                break;
            }
            merged.visited_members += found.visited_members;
            merged.computed_distances += found.computed_distances;
            merged.restarts += found.restarts;
            merged.tail_drops += found.tail_drops;
            merged.visited_saturations += found.visited_saturations;
            merged.count = merge_search_results(to_span(traversal.results), shards_[shard].keys, keys, distances,
                                                merged.count);
        }
        return merged;
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard and gathers the rankings into
     *      one per query - the executor-driven sibling of @c search, one launch per shard
     *      rather than one thread per query: every shard walks the full batch, lands its
     *      matches in its own columns of one host landing, and the drained columns merge
     *      query by query through the key planes.
     *
     *  @param[in] queries The batch, one query per row, in the shards' stored scalar type.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in] wanted Matches wanted per query - the merged rankings' width.
     *  @param[out] keys Merged rankings, @b [queries,wanted] row-major.
     *  @param[out] distances Distances of the same rankings, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *  @return The merged hits summed over queries, and the telemetry summed over shards.
     *
     *  A refused shard stops the fan-out but keeps every shard already landed, so the
     *  rankings and the counts always agree; every stream that was queued is drained before
     *  returning, since the caller's queries and the call's own landing are DMA targets
     *  until then.
     */
    search_result_t search(vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted,
                           std::span<key_t> keys, std::span<distance_t> distances,
                           std::span<std::size_t> counts) noexcept {
        search_result_t merged;
        std::size_t const queries_count = queries.extent(0);
        std::size_t const shards = shards_.size();
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k)
            return merged.status = paired, merged;
        if (counts.size() != queries_count || keys.size() < queries_count * wanted ||
            distances.size() < queries_count * wanted)
            return merged.status = status_t::config_mismatch_k, merged;
        if (!queries_count) return merged;
        // Vetted once, before anything is claimed or carved - the shards would refuse both,
        // but only after the landing and the runs had been sized for a doomed call.
        if (queries.extent(1) != dimensions() || config.count > dense_cuda_sorted_pool<16>::capacity_k)
            return merged.status = status_t::config_mismatch_k, merged;

        // The executor serializes, never the index: a second caller with its own executor
        // searches concurrently, a second on this one is refused before any scratch is carved.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return merged.status = claimed, merged;

        // The host landing: every shard's matches, one wide matrix per plane, shard columns
        // side by side - the exact destination the pitched writeback was built for. Function
        // local, freed by the destructors on every return path, all of which pass the drain
        // below first; never zeroed, since only columns a drained shard wrote are ever read.
        std::size_t const landing_entries = queries_count * shards * config.count;
        growing_buffer<dense_node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<dense_distance_t, forkunion_allocator_t> distances_landing {allocator_};
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(distances_landing.resize_uninitialized(landing_entries)))
            return merged.status = status_t::bad_alloc_k, merged;
        nk::matrix_span<dense_node_id_t> const ids_wide {ids_landing.data(), queries_count, shards * config.count};
        nk::matrix_span<dense_distance_t> const distances_wide {distances_landing.data(), queries_count,
                                                                shards * config.count};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const readied = reserve_traversals_(traversals, executor, queries_count, config); failed(readied))
            return merged.status = readied, merged;

        // Queue every shard before awaiting any, so the batch costs the wall clock of the
        // slowest device rather than the sum of them. From here to the drain there must be
        // no return: the queries and the landing are DMA targets on every queued stream.
        status_t first_refusal = status_t::success_k;
        std::size_t queued = 0;
        for (; queued != shards; ++queued) {
            nk::matrix_span<dense_node_id_t> const shard_ids = ids_wide(
                nk::all, nk::range {queued * config.count, (queued + 1) * config.count}, nk::slice);
            nk::matrix_span<dense_distance_t> const shard_distances = distances_wide(
                nk::all, nk::range {queued * config.count, (queued + 1) * config.count}, nk::slice);
            search_result_t const landed = shards_[queued].index->search(
                queries, shard_ids, shard_distances, traversals[queued], config, completion_t::deferred_k);
            if (!landed) {
                first_refusal = landed.status;
                break;
            }
        }

        // Await every stream that may carry queued transfers - the refuser included, since
        // its upload may already be reading the caller's rows - and merge each drained
        // shard's columns while the later shards are still walking.
        for (std::size_t& count : counts) count = 0;
        dense_scored_node_t frame[dense_cuda_sorted_pool<16>::capacity_k];
        for (std::size_t shard = 0; shard != shards && shard <= queued; ++shard) {
            if (status_t const drained = traversals[shard].drain(); failed(drained)) {
                if (succeeded(first_refusal)) first_refusal = drained;
                continue; // An undrained landing is unreadable - this shard merges nothing.
            }
            if (shard == queued) continue; // The refuser drained for the caller's sake, landed nothing.
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const landed_offset = (query * shards + shard) * config.count;
                std::size_t const live = repack_landed_matches_(
                    ids_landing.data() + landed_offset, distances_landing.data() + landed_offset, config.count, frame);
                std::span<key_t> const keys_row {keys.data() + query * wanted, wanted};
                std::span<distance_t> const distances_row {distances.data() + query * wanted, wanted};
                counts[query] = merge_search_results(std::span<dense_scored_node_t const> {frame, live},
                                                     shards_[shard].keys, keys_row, distances_row, counts[query]);
            }
        }

        // Every drained context's sink holds this call's spend on its shard.
        merged.status = first_refusal;
        for (std::size_t shard = 0; shard != shards && shard <= queued; ++shard) {
            dense_cuda_telemetry_block_t const spent = traversals[shard].spent();
            merged.visited_members += spent.hops;
            merged.computed_distances += spent.scored;
            merged.restarts += spent.restarts;
            merged.tail_drops += spent.tail_drops;
            merged.visited_saturations += spent.visited_saturations;
        }
        for (std::size_t const count : counts) merged.count += count;
        return merged;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** @brief Bytes a @ref save under @p config writes: one monolith, or every part file and the
     *      manifest measured over a placeholder stem; zero for an invalid config or an unplaced
     *      index. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        if (shards_.empty() || !dimensions()) return {0, status_t::uninitialized_k};
        if (status_t const valid = config.validate(); failed(valid)) return {0, valid};
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (status_t const named = custom_fields_(fields, config); failed(named)) return {0, named};
        std::span<safegraphs_custom_field_t const> const fields_span = to_span(fields);

        // Every part is framed to be measured, and a header that cannot be formatted is an
        // allocation failure the caller must see rather than a zero it would size a file by.
        status_t framing = status_t::success_k;
        auto part_bytes = [&](std::size_t nodes) noexcept {
            auto const [header, framed] = monolith_header_(nodes, config, fields_span);
            if (failed(framed)) framing = framed;
            return failed(framed) ? 0 : header.monolith_bytes();
        };
        if (config.layout == checkpoint_layout_t::monolith_k) {
            std::size_t const monolith = part_bytes(size());
            return {monolith, framing};
        }
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += part_bytes(one.index->size());
        growing_buffer<char, forkunion_allocator_t> entry_path {allocator_};
        auto part_at = [&](std::size_t part) noexcept {
            std::size_t const nodes = shards_[part].index->size();
            auto const [name, formatted] = safegraphs_part_path(entry_path, "index", part, shards_.size());
            if (failed(formatted)) framing = formatted;
            return safegraphs_part_t {name, part_bytes(nodes), nodes};
        };
        auto const [manifest, measured] = safegraphs_manifest_bytes(shards_.size(), part_at);
        if (failed(measured)) return {0, measured};
        if (failed(framing)) return {0, framing};
        return {total + manifest, status_t::success_k};
    }

    /**
     *  @brief Serializes the index under @p path in the layout @p config names: one monolith of
     *      every shard's rows in placement order with ids rebased, so it reads as one graph, or
     *      one monolith per shard beside a manifest. Every header lands last and the manifest
     *      after every part, so a failed or cancelled save leaves a file that reads as absent.
     *
     *  @param[in] path The monolith's path, or the manifest's - its parts land beside it as
     *      @c <stem>-00001-of-0000S.safegraphs.
     *  @param[in] config The layout, alignment, certified contract and provenance; the device
     *      planes cannot be settlement-probed, so @c concurrent_k refuses.
     *  @param[inout] executor The claimed streams; the shards stage sequentially through one
     *      host thread, each export riding its own device's stream.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        if (status_t const ready = saveable_(config, executor); failed(ready)) return ready;
        if (config.layout == checkpoint_layout_t::distributed_k)
            return save_distributed_(path, config, executor, on_saved);
        auto const [wanted, measured] = storage_requirements(config);
        if (failed(measured)) return measured;
        auto [mapping, mapped] = mapped_file_t::write(path, wanted);
        if (failed(mapped)) return mapped;
        if (status_t const saved = save_monolith_(mapping.span(), config, executor, on_saved); failed(saved))
            return saved;
        return mapping.sync();
    }

    /** @brief Serializes the index into @p buffer as one monolith - the in-memory sibling of the
     *      path form; a manifest has nowhere to resolve its parts, so @c distributed_k refuses.
     *      Size the buffer by @ref storage_requirements; a shorter one answers @c config_mismatch_k. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        if (status_t const ready = saveable_(config, executor); failed(ready)) return ready;
        if (config.layout != checkpoint_layout_t::monolith_k) return status_t::config_mismatch_k;
        return save_monolith_(monolith, config, executor, on_saved);
    }

    /**
     *  @brief Restores a checkpoint into the placed shards: the content says monolith or
     *      manifest, every part's header is vetted before a byte lands, the reservation comes
     *      from @p config, and the parts land per its placement - as written, part @c p on
     *      shard @c p mod S, or cut into one stripe per shard.
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] config The placement and the rows to reserve; zero fits the checkpoint.
     *  @param[inout] executor The claimed streams; the shards land sequentially through one
     *      host thread, each import riding its own device's stream.
     *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k stops the rest.
     *
     *  The index must be empty, and it takes the hold every batch takes, so a batch in flight
     *  refuses the load as @c contended_k and the load refuses a batch the same way. A cancelled
     *  or failed load clears it and reports why, so nothing is published and the next load
     *  starts clean.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        if (status_t const ready = loadable_(executor); failed(ready)) return ready;
        // The mappings outlive the load, since every part's bytes are read straight out of them.
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings {allocator_};
        growing_buffer<part_t, forkunion_allocator_t> parts {allocator_};
        if (status_t const listed = catalog_(path, mappings, parts); failed(listed)) return listed;
        return load_parts_(parts, config, executor, on_loaded);
    }

    /** @brief Restores one monolith held in @p monolith - the implementation the path form maps
     *      for. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t monolith, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        if (status_t const ready = loadable_(executor); failed(ready)) return ready;
        growing_buffer<part_t, forkunion_allocator_t> parts {allocator_};
        if (status_t const listed = catalog_monolith_(monolith, parts); failed(listed)) return listed;
        return load_parts_(parts, config, executor, on_loaded);
    }

    /** @brief The checkpoint's metadata without loading it - @ref safegraphs_read_metadata. */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
    }

    /** @brief The metadata of one monolith held in memory. */
    static expected<safegraphs_metadata_t> metadata(bytes_view_t buffer) noexcept {
        return safegraphs_read_metadata(buffer);
    }

#pragma endregion Checkpoints

#pragma region Key Planes

    /** @brief Entries stored under @p key; a concurrent insertion can only make it under-report. */
    std::size_t count(key_t const& key) const noexcept { return locators_of_key_.count(key); }

    /** @brief Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return locators_of_key_.contains(key); }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  Safe against concurrent insertions, which only ever add; each row rides one blocking
     *  device-to-host copy, and a device fault leaves that row unwritten with the total honest.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept {
        std::size_t total = 0;
        locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < vectors.extent(0)) [[maybe_unused]]
                status_t const copied = shards_[locator.shard].index->vector(locator.node,
                                                                             vectors.row(total).as_vector());
            ++total;
        });
        return total;
    }

#pragma endregion Key Planes

  private:
#pragma region Placement

    /** @brief One unreserved shard placed on @p device; null when the allocation fails. */
    static std::unique_ptr<index_t> build_(int device) noexcept {
        typename index_t::vectors_allocator_t const allocator {device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(allocator, allocator));
    }

    /** @brief Whether @p executor drives exactly this index's devices, in this index's order -
     *      the pairing every fan-out indexes by, checked once rather than searched per shard. */
    status_t pairs_with_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (executor.size() != shards_.size()) return status_t::config_mismatch_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (executor.device(shard) != shards_[shard].index->device_specs().ordinal)
                return status_t::config_mismatch_k;
        return status_t::success_k;
    }

#pragma endregion Placement

#pragma region Insertion

    /** @brief The first row of a batch that @p shard takes under @p rotation - a bijection over
     *      the shards, so the residue classes it names partition the batch exactly. */
    static std::size_t first_row_of_shard_(std::size_t shard, std::size_t shards, std::size_t rotation) noexcept {
        return (shard + shards - rotation) % shards;
    }

    /** @brief How many of @p rows a shard starting at @p first takes, stepping by @p shards. */
    static std::size_t rows_from_(std::size_t rows, std::size_t first, std::size_t shards) noexcept {
        return first < rows ? divide_round_up(rows - first, shards) : 0;
    }

    /** @brief Builds one traversal context per shard over its own stream and run, so a fan-out
     *      never allocates and never shares scratch; @p rows sizes them for the widest walk. */
    template <typename traversal_config_type_>
    status_t reserve_traversals_(growing_buffer<traversal_t, forkunion_allocator_t>& traversals, executor_t& executor,
                                 std::size_t rows, traversal_config_type_ config) noexcept {
        std::size_t bytes_per_device = 0;
        for (shard_t const& one : shards_)
            bytes_per_device = max_of(bytes_per_device, one.index->traversal_memory_usage(rows, config));
        if (status_t const sized = executor.reserve(bytes_per_device); failed(sized)) return sized;
        if (failed(traversals.reserve_exactly(shards_.size()))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            typename executor_t::stream_t& stream = executor.stream(shard);
            traversals.push_back_reserved(
                traversal_t {typename traversal_t::allocator_t {stream.arena, stream.handle, stream.device}});
            if (!traversals.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** @brief The body of @c add, run with the insertion gate held. */
    insert_result_t add_gated_(std::span<key_t const> keys, vectors_view_t new_vectors, executor_t& executor,
                               insert_config_t config) noexcept {
        insert_result_t result;
        std::size_t const rows = new_vectors.extent(0);
        std::size_t const shards = shards_.size();
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const readied = reserve_traversals_(traversals, executor, divide_round_up(rows, shards), config);
            failed(readied))
            return result.status = readied, result.new_size = size(), result;

        // Admission before anything is taken: past the ceiling, an ID has no key-plane slot.
        std::size_t const rows_ceiling = capacity();
        std::size_t routing_base = admitted_rows_.load(std::memory_order_relaxed);
        do {
            if (rows_ceiling - routing_base < rows)
                return result.status = status_t::overflow_k, result.new_size = size(), result;
        } while (!admitted_rows_.compare_exchange_weak(routing_base, routing_base + rows, std::memory_order_relaxed));
        // Rotating by rows admitted keeps a stream of small batches from drifting onto shard zero.
        std::size_t const rotation = routing_base % shards;

        // Claim and launch on every device before awaiting any. Keys are written before the
        // claim publishes, so a search that can rank a row reads its key fully written.
        std::size_t launched = 0;
        for (; launched != shards; ++launched) {
            std::size_t const first = first_row_of_shard_(launched, shards, rotation);
            std::size_t const taking = rows_from_(rows, first, shards);
            shard_t& one = shards_[launched];
            one.inserted = {status_t::success_k};
            if (!taking) continue;
            dense_node_id_t const first_node = static_cast<dense_node_id_t>(one.index->size());
            // The reverse plane's quota covers the reservation, so recording cannot fail.
            for (std::size_t taken = 0; taken != taking; ++taken) {
                std::size_t const row = first + taken * shards;
                dense_node_id_t const node = first_node + static_cast<dense_node_id_t>(taken);
                one.keys[node] = keys[row];
                [[maybe_unused]] status_t const recorded = locators_of_key_.insert(keys[row],
                                                                                   key_locator_t {launched, node});
            }
            one.inserted = one.index->insert(new_vectors(nk::range {first, rows, shards}, nk::slice), {},
                                             traversals[launched], config, completion_t::deferred_k);
            // A mismatched first ID means a claimer outside the gate, which no key written here can name.
            status_t const launched_status = !one.inserted                         ? one.inserted.status
                                             : one.inserted.first_id != first_node ? status_t::contended_k
                                                                                   : status_t::success_k;
            if (failed(launched_status)) {
                result.status = launched_status;
                break;
            }
        }

        // Drain every stream, the refused launch's included - its upload may still be reading the caller's rows.
        if (status_t const drained = executor.drain(); failed(drained) && succeeded(result.status))
            result.status = drained;
        // Every context's sink holds this call's spend on its shard; an unlaunched one holds zeros.
        for (traversal_t& traversal : traversals) {
            dense_cuda_telemetry_block_t const spent = traversal.spent();
            result.visited_members += spent.hops;
            result.computed_distances += spent.scored;
            result.restarts += spent.restarts;
            result.abandoned_back_links += spent.lock_skips;
            result.tail_drops += spent.tail_drops;
            result.visited_saturations += spent.visited_saturations;
        }
        result.new_size = size();
        return result;
    }

#pragma endregion Insertion

#pragma region Search

    /** @brief Zips one (query, shard) pair of parallel landing rows into the scored shape
     *      @c merge_search_results consumes, stopping at the first sentinel id - an absent
     *      match must never index a key plane. @return The live matches repacked. */
    static std::size_t repack_landed_matches_(dense_node_id_t const* landed_ids,
                                              dense_distance_t const* landed_distances, std::size_t count,
                                              dense_scored_node_t* frame) noexcept {
        std::size_t live = 0;
        for (; live != count && landed_ids[live] != dense_node_id_unknown_k; ++live)
            frame[live] = dense_scored_node_t {landed_ids[live], landed_distances[live]};
        return live;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** @brief One monolith of a checkpoint: where it is, what its header says, and the tokens
     *      its metadata carries. */
    struct part_t {
        /** @brief Where the relative path starts inside the paths arena. */
        std::size_t path_offset = 0;
        /** @brief The relative path's length; zero names the checkpoint path itself. */
        std::size_t path_length = 0;
        /** @brief The part's whole bytes - a mapping's view, or the caller's own buffer. */
        bytes_view_t monolith;
        /** @brief Rows the monolith holds, per its own header and columns. */
        std::size_t nodes = 0;
        /** @brief The monolith's header. */
        safegraphs_header_t header;
        /** @brief The metric token the monolith recorded; empty when it omitted one. */
        safegraphs_name_t metric;
        /** @brief The stored scalar the monolith recorded; empty when it omitted one. */
        safegraphs_name_t quantization;
    };

    /** @brief The interchange token of one key - the monolith's key column dtype. */
    inline static constexpr dtype_t key_dtype_k = safegraphs_key_dtype<key_t>();
    /** @brief The value column's dtype - the stored scalar's own token, so a packed word's
     *      suffix carries the packing and the column's shape counts logical dimensions. */
    inline static constexpr dtype_t value_column_dtype_k = safegraphs_value_dtype<scalar_t>();

    /** @brief Whether the index can write @p config through @p executor: placed, reserved,
     *      tokenized, legal, paired, unbroken, and certified quiescent. */
    status_t saveable_(save_config_t const& config, executor_t const& executor) const noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        if (config.synchronization == synchronization_t::concurrent_k) return status_t::config_mismatch_k;
        if (status_t const valid = config.validate(); failed(valid)) return valid;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(executor);
    }

    /** @brief Whether the index can take a checkpoint through @p executor: placed, empty,
     *      unbroken, and paired. */
    status_t loadable_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(executor);
    }

    /** @brief Storage values per stored row - the copy width of every column plane. */
    std::size_t row_values_() const noexcept {
        return divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** @brief The custom fields every monolith carries: the metric, the stored scalar by its own
     *      name - what a packed @c u8 column cannot say - and the caller's provenance. */
    status_t custom_fields_(growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields,
                            save_config_t const& config) const noexcept {
        if (failed(fields.reserve_exactly(2 + config.provenance.size()))) return status_t::bad_alloc_k;
        fields.push_back_reserved({"metric", nk_kernel_name(kernel_kind_)});
        fields.push_back_reserved({"quantization", nk_dtype_name(vector_dtype_)});
        for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
        return status_t::success_k;
    }

    /** @brief The monolith header for @p nodes rows under @p config 's alignment. */
    expected<safegraphs_header_t> monolith_header_(std::size_t nodes, save_config_t const& config,
                                                   std::span<safegraphs_custom_field_t const> fields) const noexcept {
        safegraphs_regular_shape_t const shape {nodes,        key_dtype_k,    value_column_dtype_k,
                                                dimensions(), dtype_t::u32_k, connectivity()};
        return safegraphs_regular_header(shape, config.alignment, fields);
    }

    /** @brief Walks every plan's steps from this one host thread, taking step @c s of every
     *      shard before step @c s+1, so the devices' streams overlap instead of draining one
     *      device at a time; the first failing step ends the walk.
     *
     *  @param[in] plans One plan per shard, each naming its own step count.
     *  @param[in] per_step Runs one shard's step and answers its status.
     */
    template <typename plan_type_, typename per_step_type_>
    status_t round_robin_steps_(growing_buffer<plan_type_, forkunion_allocator_t>& plans,
                                per_step_type_& per_step) noexcept {
        std::size_t deepest = 0;
        for (plan_type_ const& plan : plans) deepest = max_of(deepest, plan.steps().size());
        for (std::size_t position = 0; position != deepest; ++position)
            for (std::size_t shard = 0; shard != plans.size(); ++shard) {
                auto const steps = plans[shard].steps();
                if (position >= steps.size()) continue;
                if (status_t const landed = per_step(shard, steps[position]); failed(landed)) return landed;
            }
        return status_t::success_k;
    }

    /** @brief Writes every shard into one monolith at @p target: one plan per shard is the cut,
     *      the batches land round-robin over the devices, the plans close in placement order with
     *      their keys, and the header last. */
    template <typename progress_type_>
    status_t save_monolith_(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                            progress_type_& on_saved) noexcept {
        std::size_t const shards = shards_.size();
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;

        // One plan per shard is the cut, and the row base each shard writes from follows.
        growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> bases {allocator_};
        if (failed(plans.reserve_exactly(shards)) || failed(bases.resize(shards + 1))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
            auto [plan, planned] = one.plan_save(config.synchronization, executor.stream(shard).handle);
            if (failed(planned)) return planned;
            bases[shard + 1] = bases[shard] + plan.visible;
            plans.push_back_reserved(std::move(plan));
        }

        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (failed(custom_fields_(fields, config))) return status_t::bad_alloc_k;
        auto const [header, framed] = monolith_header_(bases[shards], config, to_span(fields));
        if (failed(framed)) return framed;
        if (monolith.size() < header.monolith_bytes()) return status_t::overflow_k;

        // Every step is one slice of one column, so a device writes where no other does.
        auto save_step = [&](std::size_t shard, auto const& step) noexcept {
            iota_view<dense_node_id_t> const rows = step.index_rows;
            bytes_span_t const vectors = header.rows_of(monolith, header.value, bases[shard] + rows.first, rows.size());
            bytes_span_t const neighbors = header.rows_of(monolith, header.neighbors, bases[shard] + rows.first,
                                                          rows.size());
            if (status_t const saved = plans[shard].save_vectors(step, vectors); failed(saved)) return saved;
            return plans[shard].save_neighbors(step, neighbors, static_cast<dense_node_id_t>(bases[shard]),
                                               nk::matrix_span<dense_node_id_t> {});
        };
        if (status_t const landed = round_robin_steps_(plans, save_step); failed(landed)) return landed;

        for (std::size_t shard = 0; shard != shards; ++shard) {
            if (status_t const finished = plans[shard].finish(); failed(finished)) return finished;
            std::size_t const nodes = plans[shard].visible;
            if (status_t const wrote = copy_bytes(header.rows_of(monolith, header.key, bases[shard], nodes),
                                                  bytes_of(shards_[shard].keys.data(), nodes));
                failed(wrote))
                return wrote;
            if (on_saved(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
        }
        return safegraphs_write_header(monolith, header, to_span(fields));
    }

    /** @brief Writes one monolith per shard beside @p path, rows verbatim, the batches landing
     *      round-robin over the devices, and the manifest last - the commit record, so a failed or
     *      cancelled save leaves parts and no checkpoint. */
    template <typename progress_type_>
    status_t save_distributed_(std::string_view path, save_config_t const& config, executor_t& executor,
                               progress_type_& on_saved) noexcept {
        std::size_t const shards = shards_.size();
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;

        growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<safegraphs_header_t, forkunion_allocator_t> headers {allocator_};
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings {allocator_};
        if (failed(plans.reserve_exactly(shards)) || failed(headers.reserve_exactly(shards)) ||
            failed(mappings.reserve_exactly(shards)))
            return status_t::bad_alloc_k;
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (failed(custom_fields_(fields, config))) return status_t::bad_alloc_k;
        std::span<safegraphs_custom_field_t const> const named = to_span(fields);

        // Every part is planned, sized and mapped before a byte lands.
        std::string_view const stem = safegraphs_stem(path);
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
            auto [plan, planned] = one.plan_save(config.synchronization, executor.stream(shard).handle);
            if (failed(planned)) return planned;
            auto const [header, framed] = monolith_header_(plan.visible, config, named);
            if (failed(framed)) return framed;
            auto const [file_path, formatted] = safegraphs_part_path(part_path, stem, shard, shards);
            if (failed(formatted)) return formatted;
            auto [mapping, mapped] = mapped_file_t::write(file_path, header.monolith_bytes());
            if (failed(mapped)) return mapped;
            plans.push_back_reserved(std::move(plan));
            headers.push_back_reserved(header);
            mappings.push_back_reserved(std::move(mapping));
        }

        // A part's ids stay shard-local, so its rows travel verbatim under the sentinel convention.
        auto save_step = [&](std::size_t shard, auto const& step) noexcept {
            safegraphs_header_t const& header = headers[shard];
            bytes_span_t const monolith = mappings[shard].span();
            iota_view<dense_node_id_t> const rows = step.index_rows;
            bytes_span_t const vectors = header.rows_of(monolith, header.value, rows.first, rows.size());
            bytes_span_t const neighbors = header.rows_of(monolith, header.neighbors, rows.first, rows.size());
            if (status_t const saved = plans[shard].save_vectors(step, vectors); failed(saved)) return saved;
            return plans[shard].save_neighbors(step, neighbors, dense_node_id_t {0},
                                               nk::matrix_span<dense_node_id_t> {});
        };
        if (status_t const landed = round_robin_steps_(plans, save_step); failed(landed)) return landed;

        std::size_t nodes = 0;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            if (status_t const finished = plans[shard].finish(); failed(finished)) return finished;
            safegraphs_header_t const& header = headers[shard];
            bytes_span_t const monolith = mappings[shard].span();
            std::size_t const part_nodes = plans[shard].visible;
            if (status_t const wrote = copy_bytes(header.rows_of(monolith, header.key, 0, part_nodes),
                                                  bytes_of(shards_[shard].keys.data(), part_nodes));
                failed(wrote))
                return wrote;
            if (status_t const wrote = safegraphs_write_header(monolith, header, named); failed(wrote)) return wrote;
            if (status_t const flushed = mappings[shard].sync(); failed(flushed)) return flushed;
            nodes += part_nodes;
            if (on_saved(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
        }

        // The manifest commits the checkpoint; entries name their parts relative to it.
        std::string_view const base = safegraphs_basename(stem);
        growing_buffer<char, forkunion_allocator_t> entry_path {allocator_};
        auto part_at = [&](std::size_t shard) noexcept {
            auto const [name, formatted] = safegraphs_part_path(entry_path, base, shard, shards);
            return safegraphs_part_t {name, headers[shard].monolith_bytes(), plans[shard].visible};
        };
        return safegraphs_write_manifest(path, shards, part_at);
    }

    /** @brief Reads one part's header and tokens out of the bytes it carries. */
    status_t describe_part_(part_t& part) const noexcept {
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
            if (field.key == "quantization") part.quantization = safegraphs_name_t {field.value};
        };
        auto const [header, parsed] = safegraphs_read_header(part.monolith, allocator_, per_custom_field);
        if (failed(parsed)) return parsed;
        part.header = header;
        part.nodes = header.nodes();
        return status_t::success_k;
    }

    /** @brief Lists @p monolith as the checkpoint's only part. */
    status_t catalog_monolith_(bytes_view_t monolith,
                               growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        part_t part;
        part.monolith = monolith;
        if (status_t const described = describe_part_(part); failed(described)) return described;
        return parts.push_back(part);
    }

    /**
     *  @brief Lists the parts at @p path: the magic makes a lone monolith, anything else a
     *      manifest whose every part is mapped and its header read now, so no entry is taken on
     *      trust. The mappings land in @p mappings, which must outlive the load, since every
     *      part's rows are read straight out of them.
     */
    status_t catalog_(std::string_view path, growing_buffer<mapped_file_t, forkunion_allocator_t>& mappings,
                      growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        auto [mapping, mapped] = mapped_file_t::read(path);
        if (failed(mapped)) return mapped;
        bytes_view_t const checkpoint = mapping.view();
        if (checkpoint.size() >= safegraphs_prefix_t::magic_k.size() &&
            std::memcmp(checkpoint.data(), safegraphs_prefix_t::magic_k.data(), safegraphs_prefix_t::magic_k.size()) ==
                0) {
            if (status_t const listed = catalog_monolith_(checkpoint, parts); failed(listed)) return listed;
            return mappings.push_back(std::move(mapping));
        }

        growing_buffer<char, forkunion_allocator_t> paths_arena {allocator_};
        status_t kept = status_t::success_k;
        auto per_part = [&](safegraphs_part_t entry) noexcept {
            std::size_t const offset = paths_arena.size();
            for (char const character : entry.path)
                if (failed(paths_arena.push_back(character))) return kept = status_t::bad_alloc_k, void();
            part_t part;
            part.path_offset = offset;
            part.path_length = entry.path.size();
            if (failed(parts.push_back(part))) kept = status_t::bad_alloc_k;
        };
        auto const [summary, listed] = safegraphs_read_manifest(path, allocator_, per_part);
        if (failed(listed)) return listed;
        if (failed(kept)) return kept;
        if (parts.empty()) return status_t::malformed_k;
        if (failed(mappings.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (part_t& part : parts) {
            auto const [resolved, joined] = safegraphs_sibling_path(
                part_path, path, std::string_view {paths_arena.data() + part.path_offset, part.path_length});
            if (failed(joined)) return joined;
            auto [part_mapping, part_mapped] = mapped_file_t::read(resolved);
            if (failed(part_mapped)) return part_mapped;
            part.monolith = part_mapping.view();
            if (status_t const described = describe_part_(part); failed(described)) return described;
            mappings.push_back_reserved(std::move(part_mapping));
        }
        return status_t::success_k;
    }

    /** @brief Whether @p part can land here: three regular columns of the index's own dtypes,
     *      the tokens it names agreeing with this instantiation, the geometry @p first froze,
     *      and every column describing the same rows. */
    status_t vetted_part_(part_t const& part, safegraphs_header_t const& first) const noexcept {
        safegraphs_header_t const& header = part.header;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (header.key.dtype != key_dtype_k || header.value.dtype != value_column_dtype_k ||
            header.neighbors.dtype != dtype_t::u32_k)
            return status_t::config_mismatch_k;
        if (header.key.shape != column_shape_t {scalar_shape_t {}} || header.value.shape != first.value.shape ||
            header.neighbors.shape != first.neighbors.shape)
            return status_t::config_mismatch_k;
        if (!part.metric.empty() && nk_kernel_named(part.metric.characters, part.metric.view().size()) != kernel_kind_)
            return status_t::config_mismatch_k;
        if (!part.quantization.empty() &&
            nk_dtype_named(part.quantization.characters, part.quantization.view().size()) != vector_dtype_)
            return status_t::config_mismatch_k;
        if (header.key.nodes() != part.nodes || header.value.nodes() != part.nodes ||
            header.neighbors.nodes() != part.nodes)
            return status_t::malformed_k;
        return status_t::success_k;
    }

    /** @brief The @p round -th assignment of @p shard, or one with no rows when the shard has
     *      fewer; assignments arrive grouped by shard and in part order within one. */
    static checkpoint_assignment_t assignment_of_(
        growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> const& assignments, std::size_t shard,
        std::size_t round) noexcept {
        std::size_t seen = 0;
        for (checkpoint_assignment_t const& assignment : assignments)
            if (assignment.shard == shard && seen++ == round) return assignment;
        return checkpoint_assignment_t {shard, 0, {0, 0}};
    }

    /**
     *  @brief Lands the assignments round by round - the @c r -th of every shard together, one
     *      plan per shard on its own stream, the batches landing round-robin over the devices -
     *      then publishes each plan, lands its keys at the append base and records them in the
     *      reverse plane.
     */
    status_t load_rounds_(growing_buffer<part_t, forkunion_allocator_t> const& parts,
                          growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> const& assignments,
                          executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        growing_buffer<std::size_t, forkunion_allocator_t> rounds_per_shard {allocator_};
        if (failed(rounds_per_shard.resize(shards))) return status_t::bad_alloc_k;
        std::size_t rounds = 0;
        for (checkpoint_assignment_t const& assignment : assignments)
            rounds = max_of(rounds, ++rounds_per_shard[assignment.shard]);

        for (std::size_t round = 0; round != rounds; ++round) {
            growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {allocator_};
            growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> taken {allocator_};
            if (failed(plans.reserve_exactly(shards)) || failed(taken.reserve_exactly(shards)))
                return status_t::bad_alloc_k;
            for (std::size_t shard = 0; shard != shards; ++shard) {
                checkpoint_assignment_t const assignment = assignment_of_(assignments, shard, round);
                index_t& one = *shards_[shard].index;
                if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
                auto [plan, planned] = one.plan_load(assignment.rows.size(), executor.stream(shard).handle);
                if (failed(planned)) return planned;
                plans.push_back_reserved(std::move(plan));
                taken.push_back_reserved(assignment);
            }

            // Every step is one slice of one part's column, so a device reads where none writes.
            auto load_step = [&](std::size_t shard, auto const& step) noexcept {
                checkpoint_assignment_t const& assignment = taken[shard];
                part_t const& part = parts[assignment.part];
                iota_view<dense_node_id_t> const range = step.index_rows;
                std::size_t const first_row = assignment.rows.first + step.source_rows.first;
                bytes_view_t const vectors = part.header.rows_of(part.monolith, part.header.value, first_row,
                                                                 range.size());
                bytes_view_t const neighbors = part.header.rows_of(part.monolith, part.header.neighbors, first_row,
                                                                   range.size());
                if (status_t const loaded = plans[shard].load_vectors(step, vectors); failed(loaded)) return loaded;
                return plans[shard].load_neighbors(step, neighbors,
                                                   {static_cast<dense_node_id_t>(assignment.rows.first),
                                                    static_cast<dense_node_id_t>(assignment.rows.bound)});
            };
            if (status_t const landed = round_robin_steps_(plans, load_step); failed(landed)) return landed;

            // The keys at the append base and the reverse plane first, then the publication - the
            // order an insert keeps, so a search never reaches a published row without its key.
            // Every plan finishes, an empty one included, since finishing hands the shard's claim back.
            for (std::size_t shard = 0; shard != shards; ++shard) {
                checkpoint_assignment_t const& assignment = taken[shard];
                keys_plane_t& keys = shards_[shard].keys;
                std::size_t const base = plans[shard].base;
                std::size_t const rows = assignment.rows.size();
                if (base + rows > keys.size()) return status_t::config_mismatch_k;
                if (rows) {
                    part_t const& part = parts[assignment.part];
                    if (status_t const read = copy_bytes(
                            bytes_of(keys.data() + base, rows),
                            part.header.rows_of(part.monolith, part.header.key, assignment.rows.first, rows));
                        failed(read))
                        return read;
                    for (std::size_t row = 0; row != rows; ++row) [[maybe_unused]]
                        status_t const recorded = locators_of_key_.insert(
                            keys[base + row], key_locator_t {shard, static_cast<dense_node_id_t>(base + row)});
                }
                if (cudaSetDevice(shards_[shard].index->device_specs().ordinal) != cudaSuccess)
                    return status_t::device_error_k;
                if (status_t const finished = plans[shard].finish(); failed(finished)) return finished;
            }
        }
        return status_t::success_k;
    }

    /** @brief Lands @p parts on the shards: vetted, assigned per the placement, reserved from the
     *      config, loaded round by round over the devices, and cleared whole on any failure. */
    template <typename progress_type_>
    status_t load_parts_(growing_buffer<part_t, forkunion_allocator_t>& parts, load_config_t const& config,
                         executor_t& executor, progress_type_& on_loaded) noexcept {
        std::size_t const shards = shards_.size();
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;
        // The same two holds `add` takes: a batch through another executor must see `contended_k`, not the load.
        if (!mutating_.lock_if_free()) return status_t::contended_k;
        std::lock_guard const mutating {mutating_, std::adopt_lock};
        // Empty under the hold, not only at the entry check: a batch admitted between the two
        // would be overwritten by the total this load stores at its end.
        if (size()) return status_t::config_mismatch_k;

        safegraphs_header_t const& first = parts[0].header;
        for (part_t const& part : parts)
            if (status_t const vetted = vetted_part_(part, first); failed(vetted)) return vetted;
        connectivity_t const connectivity = static_cast<connectivity_t>(
            regular_width(first.neighbors.shape).value_or(0));
        if (connectivity != connectivity_16_k && connectivity != connectivity_32_k &&
            connectivity != connectivity_128_k)
            return status_t::malformed_k;
        std::size_t const dimensions = regular_width(first.value.shape).value_or(0);
        if (!dimensions) return status_t::malformed_k;

        // Parts onto shards per the placement, and the rows each shard ends up with.
        growing_buffer<std::size_t, forkunion_allocator_t> part_rows {allocator_};
        if (failed(part_rows.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
        for (part_t const& part : parts) part_rows.push_back_reserved(part.nodes);
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {allocator_};
        if (failed(shard_rows.resize(shards))) return status_t::bad_alloc_k;
        growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> assignments {allocator_};
        status_t kept = status_t::success_k;
        assign_parts(config.placement, to_span(part_rows), shards, [&](checkpoint_assignment_t assignment) noexcept {
            if (failed(assignments.push_back(assignment))) kept = status_t::bad_alloc_k;
            shard_rows[assignment.shard] += assignment.rows.size();
        });
        if (failed(kept)) return kept;
        std::size_t widest = 0;
        std::size_t total = 0;
        for (std::size_t const rows : shard_rows) {
            widest = max_of(widest, rows);
            total += rows;
        }

        // An unreserved index takes the config's rows or the widest shard times the shard count;
        // a reserved one must already fit every shard.
        std::size_t const wanted = max_of(config.capacity, widest * shards);
        if (!capacity()) {
            if (!wanted) return status_t::success_k;
            if (status_t const reserved = reserve({wanted, connectivity, dimensions}).status; failed(reserved))
                return reserved;
        }
        else {
            if (dimensions != this->dimensions() || connectivity != this->connectivity() ||
                capacity() < config.capacity)
                return status_t::config_mismatch_k;
            for (std::size_t shard = 0; shard != shards; ++shard)
                if (shard_rows[shard] > shards_[shard].keys.size()) return status_t::config_mismatch_k;
        }

        status_t settled = load_rounds_(parts, assignments, executor);
        for (std::size_t shard = 0; shard != shards && succeeded(settled); ++shard)
            if (on_loaded(shard) == continuation_t::cancel_k) settled = status_t::cancelled_k;
        // Nothing is published by a failed or cancelled load: the shards go back to empty.
        if (failed(settled)) {
            [[maybe_unused]] status_t const cleared = clear();
            return settled;
        }
        admitted_rows_.store(total, std::memory_order_relaxed);
        return status_t::success_k;
    }

#pragma endregion Checkpoints

    /** @brief The owned shards, graph and keys together. A shard's key plane is sized at
     *      @c reserve and never grown, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;
    /** @brief Every key's entries. Duplicates coexist as separate slots, which is exactly what
     *      several values under one key means, and costs one slot per row rather than per key. */
    locators_plane_t locators_of_key_;
    /** @brief The host allocator behind the shard table, the key and reverse planes and every
     *      checkpoint catalog; the shards' planes live on their devices. */
    forkunion_allocator_t allocator_;
    /** @brief Rows admitted so far - the gate against @ref capacity and the next batch's
     *      rotation base. */
    std::atomic<std::size_t> admitted_rows_ {0};
    /** @brief Held by the one batch of mutating work at a time; a refused hold is @c contended_k. */
    fu::spin_mutex<fu::preferred_yield_t> mutating_;
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
