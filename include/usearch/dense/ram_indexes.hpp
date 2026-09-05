/**
 *  @brief The sharded index over CPU-resident dense shards - one per memory domain - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/ram_indexes.hpp
 *  @date July 23, 2026
 *
 *  The plural layer over @c dense_ram_index: same-geometry shards, each owned and pinned to
 *  one ForkUnion memory domain, viewed as one index. Insertions rotate across the shards so
 *  they fill evenly and each holds a representative sample; searches scatter over every
 *  shard and gather through @ref merge_search_results, resolving matches into the caller's
 *  keys, so shards with colliding id spaces land in one shared ranking. Checkpoints ride the
 *  SafeGraphs interchange - one consolidated monolith, or one per shard beside a manifest -
 *  written positionally with every header last, so a failed save reads as absent.
 */
#pragma once

#include <cstring> // `std::memcpy`, `std::memcmp` - row copies & the monolith's magic

#include <array>       // `std::array` - the monolith magic
#include <atomic>      // `std::atomic` - the admission word, the chunk cursors & fan-out outcomes
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the callers' key batches & merged rankings
#include <string_view> // `std::string_view` - the checkpoint paths
#include <type_traits> // `std::is_same_v` - detecting the compiled-out default predicate
#include <utility>     // `std::move` - handing a mapping to the buffer that outlives the load

#include <forkunion.h> // `fu_memory_domain_id_t` - the placement argument of `emplace`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the shard table & key planes
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`, `forkunion_executor_t`
#include "usearch/utilities/ram_tables.hpp"        // `growing_multimap` - the reverse key plane
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths & manifests - the checkpoints
#include "usearch/dense/ram_index.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief The sharded index: same-geometry CPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are
 *  expressible. Shards arrive one at a time through @ref emplace, so the index grows as domains
 *  are engaged rather than being declared whole. The shard's configs, views and results are
 *  re-exported verbatim, so a batched call and a single one read the same.
 */
template <                                               //
    typename key_type_ = uuid_t,                         //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k, //
    nk_capability_t capability_ = nk_cap_icelake_k,      //
    typename wait_type_ = fu::preferred_yield_t,         //
    typename key_hash_type_ = std::hash<key_type_>       //
    >
struct dense_ram_indexes {
    /** @brief The caller-visible identity, opaque to every shard - indexes store no keys, so the
     *      whole key vocabulary begins here. */
    using key_t = key_type_;

    /** @brief Where one entry lives: the shard, and its shard-local ID - the index's only
     *      address for an entry. */
    struct key_locator_t {
        /** @brief The shard holding the entry, positional in placement order. */
        std::size_t shard = 0;
        /** @brief The entry's ID inside that shard's own space. */
        dense_node_id_t node = 0;
    };

    /** @brief Keys of one shard's rows, indexed by that shard's own node IDs. */
    using keys_plane_t = growing_buffer<key_t, forkunion_allocator_t>;
    /** @brief Key to every entry stored under it; insert-only, wait-free for readers. */
    using locators_plane_t = growing_multimap<key_t, key_locator_t, key_hash_type_, //
                                              forkunion_allocator_t>;

    /** @brief The shard type - geometry parameters in the shard's own order, so the two can
     *      never disagree. */
    using index_t = dense_ram_index<vector_dtype_, kernel_kind_, capability_, wait_type_, //
                                    forkunion_allocator_t,                                //
                                    forkunion_allocator_t,                                //
                                    arena_allocator<byte_t, forkunion_allocator_t>>;

    /** @brief One shard: the graph, and the keys of its rows. Placement is not repeated - the
     *      index's own allocator carries it. */
    struct shard_t {
        /** @brief The graph; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;
        /** @brief The key of each of this shard's rows, indexed by node ID. */
        keys_plane_t keys;
    };

    /** @brief The threads and their scratch every bulk verb runs on; it names no index, so one
     *      executor serves every instantiation. */
    using executor_t = forkunion_executor_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;
    using traversal_t = typename index_t::dense_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;

    /** @brief The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    /** @brief The distance kernel this instantiation walks with. */
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    /** @brief The compute-capability mask the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

#pragma region Placement

    /** @brief Every index names the host allocator its shard table, reverse plane and checkpoint
     *      catalogs sit on; nothing is placed by default. */
    dense_ram_indexes() = delete;
    explicit dense_ram_indexes(forkunion_allocator_t allocator) noexcept
        : shards_(allocator), locators_of_key_(allocator), allocator_(allocator) {}
    dense_ram_indexes(dense_ram_indexes&& other) noexcept
        : shards_(other.allocator_), locators_of_key_(other.allocator_), allocator_(other.allocator_) {
        swap(other);
    }
    dense_ram_indexes& operator=(dense_ram_indexes&& other) noexcept { return swap(other), *this; }
    dense_ram_indexes(dense_ram_indexes const&) = delete;
    dense_ram_indexes& operator=(dense_ram_indexes const&) = delete;
    /** @brief Every member is RAII, so the shards die with the index and nothing leaks. */
    ~dense_ram_indexes() noexcept = default;

    void swap(dense_ram_indexes& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::size_t const admitted = admitted_rows_.load(std::memory_order_relaxed);
        admitted_rows_.store(other.admitted_rows_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.admitted_rows_.store(admitted, std::memory_order_relaxed);
    }

    /** @brief Appends one shard built here on @p memory_domain, for callers with none to hand over. */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(memory_domain);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** @brief Appends one shard the caller already built, consuming it; refuses any whose frozen
     *      geometry differs from the shards already here, and any already reserved. Refused once
     *      storage exists - @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (shard->capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        keys_plane_t keys {forkunion_allocator_t(shard->vectors_allocator_.memory_domain)};
        return shards_.push_back(shard_t {std::move(shard), std::move(keys)});
    }

    /** @brief Empties every shard in place, keeping its memory domain, geometry and capacity.
     *      Demands the same quiescence as @ref reserve. */
    status_t clear() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = build_(one.index->vectors_allocator_.memory_domain);
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

    /** @brief Shards in the index, one per engaged memory domain. */
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

    /** @brief Bytes the index occupies - every shard's storage, its key plane, and the shared
     *      reverse plane; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(dense_ram_indexes) + locators_of_key_.memory_usage();
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
     *  @brief Inserts one row under @p key into whichever shard the rotation names next - the
     *      single-threaded sibling of the batched form, taking a context where it takes an executor.
     *
     *  Safe to call concurrently with itself, with the batched form, and with searches: the shard
     *  is chosen by one relaxed read-modify-write, and everything past the capacity claim is a
     *  step that cannot fail.
     *
     *  @param[in] key The caller's identity for the row; several rows may share one.
     *  @param[in] new_vector The row, in the shards' stored scalar type.
     *  @param[inout] traversal The calling thread's scratch & telemetry.
     *  @param[in] config The insertion configuration handed to the receiving shard.
     */
    insert_result_t add(key_t const& key, vector_view_t new_vector, traversal_t& traversal,
                        insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (shards_.empty()) return result.status = status_t::uninitialized_k, result;

        // Admission before anything is taken: past the ceiling, an ID has no key-plane slot.
        std::size_t const rows_ceiling = capacity();
        std::size_t routing_base = admitted_rows_.load(std::memory_order_relaxed);
        do {
            if (routing_base == rows_ceiling) return result.status = status_t::overflow_k, result;
        } while (!admitted_rows_.compare_exchange_weak(routing_base, routing_base + 1, std::memory_order_relaxed));

        std::size_t const shard = routing_base % shards_.size();
        shard_t& one = shards_[shard];
        if (status_t const sized = one.index->reserve_traversal(traversal, config); failed(sized))
            return result.status = sized, result;
        auto [claim, granted] = one.index->claim(1);
        if (failed(granted)) return result.status = granted, result;

        // The context's counters run across calls, so the report is the difference this one made.
        std::size_t const visited_before = traversal.visited_members;
        std::size_t const computed_before = traversal.computed_distances;
        std::size_t const refines_before = traversal.computed_distances_in_refines;
        std::size_t const reverse_refines_before = traversal.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = traversal.restarts;

        // The key lands before publication, so a reachable node always reads one.
        dense_node_id_t assigned = 0;
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        one.index->insert_reserved(row, {}, traversal, std::move(claim), config,
                                   [&](dense_node_id_t assigned_id) noexcept {
                                       assigned = assigned_id;
                                       one.keys[assigned_id] = key;
                                   });
        // The reverse plane's quota covers the reservation, so recording cannot fail.
        [[maybe_unused]] status_t const recorded = locators_of_key_.insert(key, key_locator_t {shard, assigned});

        result.visited_members = traversal.visited_members - visited_before;
        result.computed_distances = traversal.computed_distances - computed_before;
        result.computed_distances_in_refines = traversal.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = traversal.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = traversal.restarts - restarts_before;
        result.new_size = size();
        return result;
    }

    /**
     *  @brief Inserts a whole matrix of rows under @p keys, split across the shards from a rotating
     *      start so they fill evenly - the executor-driven sibling of one insertion.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row landed in
     *  is the index's business. The keys are what the caller gets its rows back by.
     *
     *  All-or-nothing across shards: admission is checked and every claim taken before any row
     *  lands, and past that point nothing can fail. One executor drives one call at a time.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_vectors The batch, one row per entry, in the shards' stored scalar type.
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    insert_result_t add(std::span<key_t const> keys, vectors_view_t new_vectors, executor_t& executor,
                        insert_config_t config = {}) noexcept {
        using claim_t = typename index_t::capacity_claim_t;
        insert_result_t result;
        result.new_size = size();
        std::size_t const rows = new_vectors.extent(0);
        std::size_t const shards = shards_.size();
        if (!shards) return result.status = status_t::uninitialized_k, result;
        if (keys.size() != rows) return result.status = status_t::config_mismatch_k, result;
        if (!rows) return result;

        // One caller per executor, and the arenas grow to the widest geometry ever asked of them -
        // no caller guesses a byte count, the frozen geometry already knows it.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return result.status = claimed, result;
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage(config)); failed(sized))
            return result.status = sized, result;

        // Rotating by rows admitted keeps a stream of small batches from drifting onto shard zero;
        // a snapshot serves, since the split only has to agree with the claims it sizes.
        std::size_t const rotation = admitted_rows_.load(std::memory_order_relaxed) % shards;
        growing_buffer<claim_t, forkunion_allocator_t> claims {allocator_};
        if (failed(claims.reserve_exactly(shards))) return result.status = status_t::bad_alloc_k, result;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const taking = rows_from_(rows, first_row_of_shard_(shard, shards, rotation), shards);
            auto [claim, granted] = shards_[shard].index->claim(taking);
            if (failed(granted)) return result.status = granted, result;
            claims.push_back_reserved(std::move(claim));
        }

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, config); failed(ready))
            return result.status = ready, result;

        // Admission is the ceiling and the next batch's rotation base, one CAS for the whole batch;
        // a refusal here drops the claims above unconsumed, so the rows credit themselves back.
        std::size_t const rows_ceiling = capacity();
        std::size_t admitted = admitted_rows_.load(std::memory_order_relaxed);
        do {
            if (rows_ceiling - admitted < rows) return result.status = status_t::overflow_k, result;
        } while (!admitted_rows_.compare_exchange_weak(admitted, admitted + rows, std::memory_order_relaxed));

        // The key lands before publication; the reverse plane's quota covers every claimed row.
        auto insert_row = [&](std::size_t task, std::size_t thread, std::size_t) noexcept {
            std::size_t const shard = (rotation + task) % shards;
            shard_t& one = shards_[shard];
            dense_node_id_t assigned = 0;
            one.index->insert_reserved(new_vectors(nk::range {task, task + 1}, nk::slice), {}, traversals[thread],
                                       claim_t {}, config, [&](dense_node_id_t assigned_id) noexcept {
                                           assigned = assigned_id;
                                           one.keys[assigned_id] = keys[task];
                                       });
            [[maybe_unused]] status_t const recorded = locators_of_key_.insert(keys[task],
                                                                               key_locator_t {shard, assigned});
        };
        // ForkUnion runs every task or none, so a refused fan-out hands the admission back whole.
        if (status_t const fanned = executor.for_tasks(rows, insert_row); failed(fanned)) {
            admitted_rows_.fetch_sub(rows, std::memory_order_relaxed);
            return result.status = fanned, result;
        }

        // Every row dispensed an id, so the claims are spent rather than credited back.
        for (claim_t& claimed : claims) claimed.consume();

        for (traversal_t const& traversal : traversals) {
            result.visited_members += traversal.visited_members;
            result.computed_distances += traversal.computed_distances;
            result.computed_distances_in_refines += traversal.computed_distances_in_refines;
            result.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
            result.restarts += traversal.restarts;
            result.saturated_walks += traversal.saturated_walks;
        }
        result.new_size = size();
        return result;
    }

#pragma endregion Insertion

#pragma region Search

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending by
     *      distance - the end-to-end multi-shard lookup.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[inout] traversal The calling thread's traversal context, reused per shard.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[inout] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' key planes, so colliding ID spaces still land in one ranking.
     *  @param[inout] distances Distances of the same ranking, parallel to @p keys.
     *
     *  A failing shard stops the scatter but keeps whatever earlier shards already merged, so the
     *  reported count and the ranking always agree.
     */
    search_result_t search(vector_view_t query_vector, traversal_t& traversal, search_config_t config,
                           std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        search_result_t merged;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            auto const found = shards_[shard].index->search(query_vector, traversal, config);
            if (!found) {
                merged.status = found.status;
                break;
            }
            merged.visited_members += found.visited_members;
            merged.computed_distances += found.computed_distances;
            merged.restarts += found.restarts;
            merged.saturated_walks += found.saturated_walks;
            merged.count = merge_search_results(to_span(traversal.results), shards_[shard].keys, keys, distances,
                                                merged.count);
        }
        return merged;
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard, one thread per query in flight -
     *      the executor-driven sibling of @ref search, taking the executor exactly where the
     *      single form takes a context.
     *
     *  @param[in] queries The batch, one query per row, in the shards' stored scalar type.
     *  @param[inout] executor The threads and their scratch; one query occupies one at a time.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in] wanted Matches kept per query - the row width of @p keys and @p distances.
     *  @param[out] keys Merged rankings, @b [queries,wanted] row-major.
     *  @param[out] distances Distances of the same rankings, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *  @param[in] predicate Optional per-key admission filter; the default admits everything
     *      and compiles the filtered machinery out. Shards test node ids, so each one's keys
     *      plane translates before the predicate sees a candidate.
     *
     *  A thread that fails stops its own query and leaves the rest to finish; the first failure is
     *  what the merged result reports, and every other query's outputs stand.
     */
    template <typename predicate_type_ = dummy_predicate_t>
    search_result_t search(vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted,
                           std::span<key_t> keys, std::span<distance_t> distances, std::span<std::size_t> counts,
                           predicate_type_&& predicate = {}) noexcept {
        search_result_t merged;
        std::size_t const queries_count = queries.extent(0);
        if (shards_.empty()) return merged.status = status_t::uninitialized_k, merged;
        if (counts.size() != queries_count || keys.size() < queries_count * wanted ||
            distances.size() < queries_count * wanted)
            return merged.status = status_t::config_mismatch_k, merged;
        if (!queries_count) return merged;

        // One caller per executor, and the arenas grow to the widest geometry ever asked of them.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return merged.status = claimed, merged;
        if (status_t const sized = executor.reserve(
                shards_[0].index->traversal_memory_usage(insert_config_t {config.expansion_factor}));
            failed(sized))
            return merged.status = sized, merged;

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, insert_config_t {config.expansion_factor});
            failed(ready))
            return merged.status = ready, merged;

        std::atomic<status_t> first_refusal {status_t::success_k};
        auto scatter_query = [&](std::size_t query, std::size_t thread, std::size_t) noexcept {
            traversal_t& traversal = traversals[thread];
            std::span<key_t> const keys_row {keys.data() + query * wanted, wanted};
            std::span<distance_t> const distances_row {distances.data() + query * wanted, wanted};
            std::size_t found_count = 0;
            for (shard_t const& one : shards_) {
                search_result_t found;
                if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    found = one.index->search(queries.row(query).as_vector(), traversal, config);
                else {
                    auto keyed = [&](dense_node_id_t node) noexcept { return predicate(one.keys[node]); };
                    found = one.index->search_if(queries.row(query).as_vector(), keyed, traversal, config);
                }
                if (!found) {
                    status_t expected_status = status_t::success_k;
                    first_refusal.compare_exchange_strong(expected_status, found.status, std::memory_order_relaxed);
                    break;
                }
                found_count = merge_search_results(to_span(traversal.results), one.keys, keys_row, distances_row, //
                                                   found_count);
            }
            counts[query] = found_count;
        };
        if (status_t const fanned = executor.for_tasks(queries_count, scatter_query); failed(fanned))
            return merged.status = fanned, merged;

        merged.status = first_refusal.load(std::memory_order_relaxed);
        for (traversal_t const& traversal : traversals) {
            merged.visited_members += traversal.visited_members;
            merged.computed_distances += traversal.computed_distances;
            merged.restarts += traversal.restarts;
            merged.saturated_walks += traversal.saturated_walks;
        }
        for (std::size_t const count : counts) merged.count += count;
        return merged;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** @brief Bytes a @ref save under @p config writes: one monolith, or every part file and the
     *      manifest measured over a placeholder stem; zero for an invalid config or an unplaced
     *      index. The row count is read now, so a buffer sized here fits a save that no row
     *      lands before. */
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
     *  @param[in] config The layout, alignment, synchronization contract and provenance.
     *  @param[inout] executor The threads the shard writes fan out over, one shard per task.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     *
     *  Under @c concurrent_k contention is detected, not excluded: a row landing mid-save
     *  reports @c contended_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        if (status_t const ready = saveable_(config); failed(ready)) return ready;
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

    /** @brief Serializes the index into @p monolith as one graph - the implementation the path
     *      form maps for; a manifest has nowhere to resolve its parts, so @c distributed_k
     *      refuses. Size the bytes by @ref storage_requirements; fewer answer @c overflow_k. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        if (status_t const ready = saveable_(config); failed(ready)) return ready;
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
     *  @param[inout] executor The threads the shard reads fan out over, one shard per task.
     *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k stops the rest.
     *
     *  The index must be empty. A cancelled or failed load clears it and reports why, so
     *  nothing is published and the next load starts clean.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        if (status_t const ready = loadable_(); failed(ready)) return ready;
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
        if (status_t const ready = loadable_(); failed(ready)) return ready;
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
     *  Safe against concurrent insertions, which only ever add.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept {
        std::size_t const row_bytes = row_values_() * sizeof(scalar_t);
        std::size_t total = 0;
        locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < vectors.extent(0)) {
                vector_view_t const stored = shards_[locator.shard].index->vector(locator.node);
                std::memcpy(vectors.row(total).as_vector().byte_data(), stored.byte_data(), row_bytes);
            }
            ++total;
        });
        return total;
    }

#pragma endregion Key Planes

  private:
#pragma region Placement

    /** @brief One unreserved shard placed on @p memory_domain; null when the allocation fails. */
    static std::unique_ptr<index_t> build_(fu_memory_domain_id_t memory_domain) noexcept {
        forkunion_allocator_t const allocator {memory_domain};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(allocator, allocator));
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

    /** @brief Builds one traversal context per thread over its own arena, so a fan-out never
     *      allocates and never shares scratch. Sized to the executor, since a dynamic fan-out
     *      picks its threads. */
    status_t reserve_traversals_(growing_buffer<traversal_t, forkunion_allocator_t>& traversals, executor_t& executor,
                                 insert_config_t config) noexcept {
        std::size_t const threads = executor.size();
        if (!threads) return status_t::uninitialized_k;
        if (failed(traversals.reserve_exactly(threads))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != threads; ++thread) {
            traversal_t traversal {arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(thread))};
            // Apart, and apart across calls: identically seeded contexts would send every
            // concurrent inserter through the same entry points.
            traversal.random_generator.state = admitted_rows_.load(std::memory_order_relaxed) + thread;
            if (failed(traversals.push_back(std::move(traversal)))) return status_t::bad_alloc_k;
            if (status_t const sized = shards_[0].index->reserve_traversal(traversals[thread], config); failed(sized))
                return sized;
        }
        return status_t::success_k;
    }

#pragma endregion Insertion

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

    /** @brief Whether the index can write @p config: placed, reserved, tokenized, and legal. */
    status_t saveable_(save_config_t const& config) const noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        return config.validate();
    }

    /** @brief Whether the index can take a checkpoint: placed, and holding nothing yet. */
    status_t loadable_() const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        return size() ? status_t::config_mismatch_k : status_t::success_k;
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

    /** @brief One slab of neighbor ids from @p arena - the scratch a save chunk rebases through;
     *      empty when the arena is short. */
    nk::matrix_span<dense_node_id_t> scratch_of_(typename executor_t::arena_t& arena,
                                                 std::size_t scratch_rows) const noexcept {
        std::size_t const connectivity = this->connectivity();
        std::byte* bytes = arena.allocate(scratch_rows * connectivity * sizeof(dense_node_id_t),
                                          alignof(dense_node_id_t));
        if (!bytes) return {};
        return {reinterpret_cast<dense_node_id_t*>(bytes), scratch_rows, connectivity};
    }

    /** @brief The shard placed on @p memory_domain - a thread's home - or the first shard when
     *      none sits there. */
    std::size_t home_shard_(fu_memory_domain_id_t memory_domain) const noexcept {
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (shards_[shard].index->vectors_allocator_.memory_domain == memory_domain) return shard;
        return 0;
    }

    /**
     *  @brief Fans every plan's chunks out over the executor's threads: each thread drains its
     *      home shard first, then helps the rest, one atomic cursor per shard being the whole
     *      schedule. @p per_step lands one step with the thread's scratch - @p scratch_rows of
     *      neighbor ids, none when zero - and answers a status; the first failure wins.
     */
    template <typename plan_type_, typename per_step_type_>
    status_t drain_steps_(growing_buffer<plan_type_, forkunion_allocator_t>& plans, executor_t& executor,
                          std::size_t scratch_rows, per_step_type_& per_step) noexcept {
        std::size_t const shards = plans.size();
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        std::atomic<status_t> outcome {status_t::success_k};
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            nk::matrix_span<dense_node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    outcome.compare_exchange_strong(expected_status, status_t::bad_alloc_k, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = home_shard_(executor.memory_domain(thread));
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (home + visited) % shards;
                std::atomic_ref<std::size_t> const cursor {cursors[shard]};
                auto const steps = plans[shard].steps();
                for (std::size_t position = cursor.fetch_add(1, std::memory_order_relaxed); position < steps.size();
                     position = cursor.fetch_add(1, std::memory_order_relaxed)) {
                    if (failed(outcome.load(std::memory_order_relaxed))) return;
                    status_t const landed = per_step(shard, steps[position], scratch);
                    if (succeeded(landed)) continue;
                    status_t expected_status = status_t::success_k;
                    outcome.compare_exchange_strong(expected_status, landed, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (status_t const fanned = executor.for_threads(drain); failed(fanned)) return fanned;
        return outcome.load(std::memory_order_relaxed);
    }

    /** @brief Writes every shard into one monolith at @p target: one plan per shard is the cut,
     *      every thread lands chunks from its home shard first, the plans close in placement
     *      order with their keys, and the header last. */
    template <typename progress_type_>
    status_t save_monolith_(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                            progress_type_& on_saved) noexcept {
        std::size_t const shards = shards_.size();
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;
        std::size_t const scratch_rows = power_of_two<std::size_t>(index_t::slab_rows_log2_k);
        if (status_t const sized = executor.reserve(scratch_rows * connectivity() * sizeof(dense_node_id_t));
            failed(sized))
            return sized;

        // One plan per shard is the cut, and the row base each shard writes from follows.
        growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> bases {allocator_};
        if (failed(plans.reserve_exactly(shards)) || failed(bases.resize(shards + 1))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto const [plan, planned] = shards_[shard].index->plan_save(config.synchronization);
            if (failed(planned)) return planned;
            plans.push_back_reserved(plan);
            bases[shard + 1] = bases[shard] + plan.visible;
        }

        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (failed(custom_fields_(fields, config))) return status_t::bad_alloc_k;
        auto const [header, framed] = monolith_header_(bases[shards], config, to_span(fields));
        if (failed(framed)) return framed;
        if (monolith.size() < header.monolith_bytes()) return status_t::overflow_k;

        // Every step is one slice of one column, so a thread writes where nothing else does.
        auto save_step = [&](std::size_t shard, auto const& step, nk::matrix_span<dense_node_id_t> scratch) noexcept {
            iota_view<dense_node_id_t> const rows = step.index_rows;
            bytes_span_t const vectors = header.rows_of(monolith, header.value, bases[shard] + rows.first, rows.size());
            bytes_span_t const neighbors = header.rows_of(monolith, header.neighbors, bases[shard] + rows.first,
                                                          rows.size());
            if (status_t const saved = plans[shard].save_vectors(step, vectors); failed(saved)) return saved;
            return plans[shard].save_neighbors(step, neighbors, static_cast<dense_node_id_t>(bases[shard]), scratch);
        };
        if (status_t const drained = drain_steps_(plans, executor, scratch_rows, save_step); failed(drained))
            return drained;

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

    /** @brief Writes one monolith per shard beside @p path, rows verbatim, every thread landing
     *      chunks from its home shard first, and the manifest last - the commit record, so a
     *      failed or cancelled save leaves parts and no checkpoint. */
    template <typename progress_type_>
    status_t save_distributed_(std::string_view path, save_config_t const& config, executor_t& executor,
                               progress_type_& on_saved) noexcept {
        std::size_t const shards = shards_.size();
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;
        std::size_t const scratch_rows = power_of_two<std::size_t>(index_t::slab_rows_log2_k);
        if (status_t const sized = executor.reserve(scratch_rows * connectivity() * sizeof(dense_node_id_t));
            failed(sized))
            return sized;

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
            auto const [plan, planned] = shards_[shard].index->plan_save(config.synchronization);
            if (failed(planned)) return planned;
            auto const [header, framed] = monolith_header_(plan.visible, config, named);
            if (failed(framed)) return framed;
            auto const [file_path, formatted] = safegraphs_part_path(part_path, stem, shard, shards);
            if (failed(formatted)) return formatted;
            auto [mapping, mapped] = mapped_file_t::write(file_path, header.monolith_bytes());
            if (failed(mapped)) return mapped;
            plans.push_back_reserved(plan);
            headers.push_back_reserved(header);
            mappings.push_back_reserved(std::move(mapping));
        }

        // A part's ids stay shard-local, so its rows travel verbatim under the sentinel convention.
        auto save_step = [&](std::size_t shard, auto const& step, nk::matrix_span<dense_node_id_t> scratch) noexcept {
            safegraphs_header_t const& header = headers[shard];
            bytes_span_t const monolith = mappings[shard].span();
            iota_view<dense_node_id_t> const rows = step.index_rows;
            bytes_span_t const vectors = header.rows_of(monolith, header.value, rows.first, rows.size());
            bytes_span_t const neighbors = header.rows_of(monolith, header.neighbors, rows.first, rows.size());
            if (status_t const saved = plans[shard].save_vectors(step, vectors); failed(saved)) return saved;
            return plans[shard].save_neighbors(step, neighbors, dense_node_id_t {0}, scratch);
        };
        if (status_t const drained = drain_steps_(plans, executor, scratch_rows, save_step); failed(drained))
            return drained;

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
     *      plan per shard, its chunks fanned out over every thread - then publishes each plan,
     *      lands its keys at the append base and records them in the reverse plane.
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
                auto const [plan, planned] = shards_[shard].index->plan_load(assignment.rows.size());
                if (failed(planned)) return planned;
                plans.push_back_reserved(plan);
                taken.push_back_reserved(assignment);
            }

            // Every step is one slice of one part's column, so a thread reads where nothing writes.
            auto load_step = [&](std::size_t shard, auto const& step, nk::matrix_span<dense_node_id_t>) noexcept {
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
            if (status_t const drained = drain_steps_(plans, executor, 0, load_step); failed(drained)) return drained;

            // Publish, then the keys at the append base, then the reverse plane - the quota covers them.
            for (std::size_t shard = 0; shard != shards; ++shard) {
                if (status_t const finished = plans[shard].finish(); failed(finished)) return finished;
                checkpoint_assignment_t const& assignment = taken[shard];
                if (assignment.rows.empty()) continue;
                part_t const& part = parts[assignment.part];
                keys_plane_t& keys = shards_[shard].keys;
                std::size_t const base = plans[shard].base;
                std::size_t const rows = assignment.rows.size();
                if (base + rows > keys.size()) return status_t::config_mismatch_k;
                if (status_t const read = copy_bytes(
                        bytes_of(keys.data() + base, rows),
                        part.header.rows_of(part.monolith, part.header.key, assignment.rows.first, rows));
                    failed(read))
                    return read;
                for (std::size_t row = 0; row != rows; ++row) [[maybe_unused]]
                    status_t const recorded = locators_of_key_.insert(
                        keys[base + row], key_locator_t {shard, static_cast<dense_node_id_t>(base + row)});
            }
        }
        return status_t::success_k;
    }

    /** @brief Lands @p parts on the shards: vetted, assigned per the placement, reserved from the
     *      config, loaded round by round over every thread, and cleared whole on any failure. */
    template <typename progress_type_>
    status_t load_parts_(growing_buffer<part_t, forkunion_allocator_t>& parts, load_config_t const& config,
                         executor_t& executor, progress_type_& on_loaded) noexcept {
        std::size_t const shards = shards_.size();
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;

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
            if (status_t const reserved = reserve({wanted, connectivity, dimensions}); failed(reserved))
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
     *      @ref reserve and never grown, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;
    /** @brief Every key's entries. Duplicates coexist as separate slots, which is exactly what
     *      several values under one key means, and costs one slot per row rather than per key. */
    locators_plane_t locators_of_key_;
    /** @brief The host allocator behind the shard table, the reverse plane and every checkpoint
     *      catalog; the shards carry their own. */
    forkunion_allocator_t allocator_;
    /** @brief Rows admitted so far - the gate against @ref capacity and the next batch's
     *      rotation base. */
    std::atomic<std::size_t> admitted_rows_ {0};
};

} // namespace usearch
} // namespace unum
