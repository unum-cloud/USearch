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
 *  one monolith per shard, the manifest committed last.
 *
 *  Beyond the executor's claim, two mechanisms guard the devices: the insertion gate admits one
 *  mutating batch at a time, and a shard's kept failure refuses every later mutation.
 */
#pragma once

#include <atomic>      // `std::atomic` - the admission counter
#include <mutex>       // `std::lock_guard`, `std::adopt_lock` - the one-batch-at-a-time hold
#include <array>       // `std::array` - the monoliths' custom fields
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the caller's keys & the merged ranking
#include <string_view> // `std::string_view` - the checkpoint paths

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"   // `growing_buffer` - the shard table & key planes
#include "usearch/utilities/ram_tables.hpp"    // `growing_multimap` - the reverse key plane
#include "usearch/utilities/serialization.hpp" // SafeGraphs monoliths & manifests - the checkpoints
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
    using keys_plane_t = growing_buffer<key_t>;
    /** @brief Key to every entry stored under it; insert-only, wait-free for readers. */
    using locators_plane_t = growing_multimap<key_t, key_locator_t, key_hash_type_>;

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

    dense_vram_indexes() = default;
    dense_vram_indexes(dense_vram_indexes&& other) noexcept { swap(other); }
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
        return shards_.push_back(shard_t {std::move(shard), keys_plane_t {}, {}});
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
        return geometry.capacity ? reserve(geometry) : status_t::success_k;
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
    status_t reserve(reserve_capacity_t config) noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (capacity()) {
            if (config.dimensions != dimensions() || config.connectivity != connectivity())
                return status_t::config_mismatch_k;
            return config.capacity <= capacity() ? status_t::success_k : status_t::config_mismatch_k;
        }
        reserve_capacity_t per_shard = config;
        per_shard.capacity = divide_round_up(config.capacity, shards_.size());

        // Key planes never grow: a search resolves through them while rows land.
        for (shard_t& one : shards_) {
            if (auto const reserved = one.index->reserve(per_shard); reserved.status != status_t::success_k)
                return reserved.status;
            if (failed(one.keys.resize(per_shard.capacity))) return status_t::bad_alloc_k;
        }
        // The quota covers every admissible row, so recording a locator is infallible by construction.
        return locators_of_key_.reserve(capacity());
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
        growing_buffer<traversal_t> traversals;
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
        growing_buffer<dense_node_id_t> ids_landing;
        growing_buffer<dense_distance_t> distances_landing;
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(distances_landing.resize_uninitialized(landing_entries)))
            return merged.status = status_t::bad_alloc_k, merged;
        nk::matrix_span<dense_node_id_t> const ids_wide {ids_landing.data(), queries_count, shards * config.count};
        nk::matrix_span<dense_distance_t> const distances_wide {distances_landing.data(), queries_count,
                                                                shards * config.count};

        growing_buffer<traversal_t> traversals;
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
        }
        for (std::size_t const count : counts) merged.count += count;
        return merged;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** @brief Bytes the shard monoliths of a @ref save occupy; the manifest's own bytes are not
     *      counted. */
    std::size_t serialized_length() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += monolith_bytes_(one.index->size());
        return total;
    }

    /**
     *  @brief Serializes every shard as a SafeGraphs monolith and, past one shard, the manifest
     *      last - the commit record, so a failed or cancelled save leaves orphan shard files and
     *      a checkpoint that reads as absent rather than half-valid.
     *
     *  @param[in] path A @c .safegraphs.json manifest path, or a bare @c .safegraphs monolith
     *      path for a single-shard index - several shards refuse the monolith form.
     *  @param[inout] executor The claimed streams; the shards stage sequentially through one
     *      host thread, each export a blocking device-to-host copy on its own device.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     *
     *  Demands a mutation-free window; a batch overlapping a save is the caller's contract violation.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, executor_t& executor, progress_type_&& on_saved = {}) noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        std::size_t const shards = shards_.size();
        if (!shards || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        bool const manifested = path.ends_with(".json");
        if (!manifested && shards != 1) return status_t::config_mismatch_k;
        // A failed shard would serialize claimed rows whose payloads may never have landed.
        if (status_t const kept = status(); failed(kept)) return kept;
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return paired;

        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;

        // One cut for the whole checkpoint; the header, the columns, and the manifest all speak it.
        growing_buffer<std::size_t> saved_nodes;
        if (failed(saved_nodes.resize(shards))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards; ++shard) saved_nodes[shard] = shards_[shard].index->size();

        std::string_view const stem = safegraphs_stem(path);
        growing_buffer<char> shard_path;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto const [file_path, formatted] = manifested ? safegraphs_shard_path(shard_path, stem, shard)
                                                           : expected<std::string_view> {path, status_t::success_k};
            if (failed(formatted)) return formatted;
            if (status_t const saved = save_monolith_(shard, file_path, saved_nodes[shard]); failed(saved))
                return saved;
            if (on_saved(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
        }
        if (!manifested) return status_t::success_k;

        // The manifest commits the checkpoint; entries name their monoliths relative to it.
        safegraphs_metadata_t metadata;
        metadata.metric = safegraphs_name_t {nk_kernel_name(kernel_kind_)};
        metadata.quantization = safegraphs_name_t {nk_dtype_name(vector_dtype_)};
        metadata.value_dtype = value_column_dtype_k;
        metadata.key_dtype = key_dtype_k;
        metadata.value_length = monolith_header_(0).value.shape;
        for (std::size_t const nodes : saved_nodes) metadata.nodes += nodes;
        metadata.files = shards;
        std::string_view const base = safegraphs_basename(stem);
        growing_buffer<char> entry_path;
        // Sized once for the widest shard index, so the entries below never allocate again.
        if (auto const [widest, formatted] = safegraphs_shard_path(entry_path, base, shards - 1); failed(formatted))
            return formatted;
        auto shard_at = [&](std::size_t shard) noexcept {
            auto const [name, formatted] = safegraphs_shard_path(entry_path, base, shard);
            return safegraphs_shard_t {name, monolith_bytes_(saved_nodes[shard]), saved_nodes[shard]};
        };
        return safegraphs_write_manifest(path, metadata, shard_at);
    }

    /**
     *  @brief Restores a checkpoint into freshly placed shards: the metadata sizes the
     *      reservation first - the reverse plane never rehashes - then every shard's columns land
     *      and the reverse plane rebuilds by re-insertion.
     *
     *  @param[in] path The manifest a @ref save wrote, or a lone monolith for one shard.
     *  @param[inout] executor The claimed streams; the shards land sequentially through one
     *      host thread, each import a blocking host-to-device copy on its own device.
     *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k stops the rest.
     *
     *  Shards must already be placed, one per manifest entry in order, and the index empty; a
     *  checkpoint whose dtype, kernel or shard count disagrees refuses before any byte lands.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, executor_t& executor, progress_type_&& on_loaded = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        if (status_t const kept = status(); failed(kept)) return kept;
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return paired;

        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;

        // The metadata decides the reservation; the entries decide what each shard reads.
        growing_buffer<char> paths_arena;
        growing_buffer<manifest_entry_t> entries;
        safegraphs_metadata_t metadata;
        std::size_t widest = 0;
        if (path.ends_with(".json")) {
            status_t kept = status_t::success_k;
            auto per_shard = [&](safegraphs_shard_t shard) noexcept {
                std::size_t const offset = paths_arena.size();
                for (char const character : shard.path)
                    if (failed(paths_arena.push_back(character))) return kept = status_t::bad_alloc_k, void();
                if (failed(entries.push_back(manifest_entry_t {offset, shard.path.size(), shard.nodes})))
                    kept = status_t::bad_alloc_k;
                widest = max_of(widest, shard.nodes);
            };
            auto const [read, listed] = safegraphs_read_manifest(path, {}, per_shard);
            if (failed(listed)) return listed;
            if (failed(kept)) return kept;
            metadata = read;
            if (entries.size() != metadata.files) return status_t::malformed_k;
        }
        else {
            auto const [read, parsed] = safegraphs_read_metadata(path);
            if (failed(parsed)) return parsed;
            metadata = read;
            widest = metadata.nodes;
        }
        if (metadata.files != shards) return status_t::config_mismatch_k;
        if (metadata.value_dtype != value_column_dtype_k || metadata.key_dtype != key_dtype_k)
            return status_t::config_mismatch_k;
        if (!metadata.quantization.empty() &&
            nk_dtype_named(metadata.quantization.characters, metadata.quantization.view().size()) != vector_dtype_)
            return status_t::config_mismatch_k;
        if (nk_kernel_named(metadata.metric.characters, metadata.metric.view().size()) != kernel_kind_)
            return status_t::config_mismatch_k;
        connectivity_t const connectivity = static_cast<connectivity_t>(metadata.connectivity);
        if (connectivity != connectivity_16_k && connectivity != connectivity_32_k &&
            connectivity != connectivity_128_k)
            return status_t::malformed_k;
        // Reserved to the widest entry: every shard's key plane must cover the entry it reads.
        if (status_t const reserved = reserve({widest * shards, connectivity, metadata.dimensions}); failed(reserved))
            return reserved;

        growing_buffer<char> shard_path;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto const [file_path, resolved] = entries.empty()
                                                   ? expected<std::string_view> {path, status_t::success_k}
                                                   : safegraphs_sibling_path(
                                                         shard_path, path,
                                                         std::string_view {paths_arena.data() + entries[shard].offset,
                                                                           entries[shard].length});
            if (failed(resolved)) return resolved;
            std::size_t const nodes = entries.empty() ? metadata.nodes : entries[shard].nodes;
            if (status_t const loaded = load_monolith_(shard, file_path, nodes); failed(loaded)) return loaded;
            if (on_loaded(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
        }
        // The rotation resumes past the loaded fills, so the next batch lands where a save left off.
        admitted_rows_.store(widest * shards, std::memory_order_relaxed);
        return status_t::success_k;
    }

    /** @brief The checkpoint's metadata without opening a shard - @ref safegraphs_read_metadata. */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
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
    status_t reserve_traversals_(growing_buffer<traversal_t>& traversals, executor_t& executor, std::size_t rows,
                                 traversal_config_type_ config) noexcept {
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
        growing_buffer<traversal_t> traversals;
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

    /** @brief One parsed manifest entry, its path parked in the shared arena by offset. */
    struct manifest_entry_t {
        /** @brief Where this entry's relative path starts inside the arena. */
        std::size_t offset = 0;
        /** @brief The relative path's length. */
        std::size_t length = 0;
        /** @brief Nodes the entry's monolith declares. */
        std::size_t nodes = 0;
    };

    /** @brief The interchange token of one key - the monolith's key column dtype. */
    inline static constexpr dtype_t key_dtype_k = sizeof(key_t) == 4    ? dtype_t::u32_k
                                                  : sizeof(key_t) == 8  ? dtype_t::u64_k
                                                  : sizeof(key_t) == 16 ? dtype_t::bin16_k
                                                                        : dtype_t::unknown_k;

    /** @brief The interchange token of one stored scalar; the sub-byte families carry none. */
    inline static constexpr dtype_t value_dtype_k = dtype_named(scalar_t::dtype_name());
    /** @brief Whether one storage word packs several dimensions - the sub-byte families. */
    inline static constexpr bool packed_scalars_k = nk::dimensions_per_value<scalar_t>() > 1;
    /** @brief The value column's dtype: packed bytes ride as @c u8 with the logical width in
     *      @c shape_effective, every byte-wide scalar as its own token. */
    inline static constexpr dtype_t value_column_dtype_k = packed_scalars_k ? dtype_t::u8_k : value_dtype_k;

    /** @brief The custom fields every monolith carries: the metric, and the stored scalar by
     *      its own name - what a packed @c u8 column cannot say. */
    static std::array<safegraphs_custom_field_t, 2> custom_fields_() noexcept {
        return {{{"metric", nk_kernel_name(kernel_kind_)}, {"quantization", nk_dtype_name(vector_dtype_)}}};
    }

    /** @brief Storage values per stored row - the copy width of every column plane. */
    std::size_t row_values_() const noexcept {
        return divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** @brief The monolith header for @p nodes rows: three regular columns at aligned starts, and
     *      the data offset they follow. */
    safegraphs_header_t monolith_header_(std::size_t nodes) const noexcept {
        safegraphs_header_t header;
        header.nodes = nodes;
        header.key = {key_dtype_k, dtype_t::unknown_k, 1, 1, 0, nodes * sizeof(key_t), 0, 0};
        std::size_t const value_start = divide_round_up<std::size_t>(header.key.values_length, header.alignment) * //
                                        header.alignment;
        std::size_t const value_shape = packed_scalars_k ? row_values_() * sizeof(scalar_t) : dimensions();
        header.value = {value_column_dtype_k,
                        dtype_t::unknown_k,
                        value_shape,
                        dimensions(),
                        value_start,
                        nodes * row_values_() * sizeof(scalar_t),
                        0,
                        0};
        std::size_t const neighbors_start =
            divide_round_up<std::size_t>(value_start + header.value.values_length, header.alignment) * header.alignment;
        header.neighbors = {dtype_t::u32_k,
                            dtype_t::unknown_k,
                            connectivity(),
                            connectivity(),
                            neighbors_start,
                            nodes * connectivity() * sizeof(dense_node_id_t),
                            0,
                            0};
        header.data_offset = safegraphs_data_offset(header, custom_fields_());
        return header;
    }

    /** @brief Bytes a monolith of @p nodes rows occupies: the header segment and the three columns. */
    std::size_t monolith_bytes_(std::size_t nodes) const noexcept {
        safegraphs_header_t const header = monolith_header_(nodes);
        return header.data_offset + header.neighbors.values_start + header.neighbors.values_length;
    }

    /** @brief Writes shard @p shard 's monolith at @p file_path: the header, then the key, value
     *      and neighbors columns, staged by one blocking device-to-host export on the shard's
     *      own device. @p nodes is the cut every column speaks. */
    status_t save_monolith_(std::size_t shard, std::string_view file_path, std::size_t nodes) noexcept {
        index_t& one = *shards_[shard].index;
        if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
        nk::matrix<scalar_t> vectors = nk::matrix<scalar_t>::try_zeros({nodes, dimensions()});
        nk::matrix<dense_node_id_t> neighbors = nk::matrix<dense_node_id_t>::try_zeros({nodes, connectivity()});
        if (nodes && (vectors.empty() || neighbors.empty())) return status_t::bad_alloc_k;
        // The device planes cannot be settlement-probed, so the caller certifies quiescence.
        if (status_t const staged = one.save(vectors.span(), neighbors.span(), synchronization_t::immutable_k);
            failed(staged))
            return staged;

        safegraphs_header_t const header = monolith_header_(nodes);
        auto [file, opened] = file_t::open(file_path, file_mode_t::write_k);
        if (failed(opened)) return opened;
        if (status_t const wrote = safegraphs_write_header(file, header, custom_fields_()); failed(wrote)) return wrote;
        if (status_t const wrote = safegraphs_write_columns(file, header,                                    //
                                                            bytes_of(shards_[shard].keys.data(), nodes),     //
                                                            bytes_of(vectors.data(), nodes * row_values_()), //
                                                            bytes_of(neighbors.data(), nodes * connectivity()));
            failed(wrote))
            return wrote;
        return file.close();
    }

    /** @brief Reads one monolith into shard @p shard: the columns land in the index on its own
     *      device, the key plane in place, and the reverse plane rebuilds by re-insertion. */
    status_t load_monolith_(std::size_t shard, std::string_view file_path, std::size_t nodes) noexcept {
        auto [file, opened] = file_t::open(file_path, file_mode_t::read_k);
        if (failed(opened)) return opened;
        auto const [header, parsed] = safegraphs_read_header(file);
        if (failed(parsed)) return parsed;
        // Declared counts re-derive before a byte lands, so a crafted monolith never outruns its buffers.
        if (header.nodes != nodes) return status_t::malformed_k;
        if (header.value.shape_effective != dimensions() || header.neighbors.shape != connectivity())
            return status_t::config_mismatch_k;
        if (header.key.dtype != key_dtype_k || header.value.dtype != value_column_dtype_k)
            return status_t::config_mismatch_k;
        if (nodes > shards_[shard].keys.size()) return status_t::malformed_k;

        index_t& one = *shards_[shard].index;
        if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
        nk::matrix<scalar_t> vectors = nk::matrix<scalar_t>::try_zeros({nodes, dimensions()});
        nk::matrix<dense_node_id_t> neighbors = nk::matrix<dense_node_id_t>::try_zeros({nodes, connectivity()});
        if (nodes && (vectors.empty() || neighbors.empty())) return status_t::bad_alloc_k;
        if (status_t const read = safegraphs_read_columns(file, header,                                    //
                                                          bytes_of(shards_[shard].keys.data(), nodes),     //
                                                          bytes_of(vectors.data(), nodes * row_values_()), //
                                                          bytes_of(neighbors.data(), nodes * connectivity()));
            failed(read))
            return read;
        // Any out-of-range ID is "no edge" in the interchange; the traversals know one sentinel.
        safegraphs_canonicalize_neighbors(std::span {neighbors.data(), nodes * connectivity()}, nodes,
                                          dense_node_id_unknown_k);

        if (status_t const landed = one.load(vectors.view(), neighbors.view(), synchronization_t::exclusive_k);
            failed(landed))
            return landed;
        // The reverse plane's quota covers the reservation, so recording cannot fail.
        for (std::size_t row = 0; row != nodes; ++row) [[maybe_unused]]
            status_t const recorded = locators_of_key_.insert(shards_[shard].keys[row],
                                                              key_locator_t {shard, static_cast<dense_node_id_t>(row)});
        return status_t::success_k;
    }

#pragma endregion Checkpoints

    /** @brief The owned shards, graph and keys together. A shard's key plane is sized at
     *      @c reserve and never grown, since a search reads it while rows are still landing. */
    growing_buffer<shard_t> shards_;
    /** @brief Every key's entries. Duplicates coexist as separate slots, which is exactly what
     *      several values under one key means, and costs one slot per row rather than per key. */
    locators_plane_t locators_of_key_;
    /** @brief Rows admitted so far - the gate against @ref capacity and the next batch's
     *      rotation base. */
    std::atomic<std::size_t> admitted_rows_ {0};
    /** @brief Held by the one batch of mutating work at a time; a refused hold is @c contended_k. */
    fu::spin_mutex<fu::preferred_yield_t> mutating_;
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
