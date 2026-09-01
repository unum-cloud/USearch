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
 *  SafeGraphs interchange - one monolith per shard, the manifest committed last.
 */
#pragma once

#include <cstdio>  // `std::snprintf` - stamping shard indexes into monolith names
#include <cstring> // `std::memcpy`, `std::strlen` - row copies & the kernel token's length

#include <memory> // `std::unique_ptr` - an index holds atomics and never moves
#include <span>   // `std::span` - the callers' key batches & merged rankings

#include <forkunion.h> // `fu_memory_domain_id_t` - the placement argument of `emplace`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"   // `growing_buffer` - the shard table & key planes
#include "usearch/utilities/ram_device.hpp"    // `forkunion_allocator`, `forkunion_executor_t`
#include "usearch/utilities/ram_tables.hpp"    // `growing_multimap` - the reverse key plane
#include "usearch/utilities/serialization.hpp" // SafeGraphs monoliths & manifests - the checkpoints
#include "usearch/dense/ram_index.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief The sharded index: same-geometry CPU-resident shards viewed as one.
 *
 *  It owns its shards outright, which is what makes a destructor, a donor-consuming merge,
 *  and clearing expressible at all. Shards arrive one at a time through @ref emplace, so the
 *  index grows as domains are engaged rather than being declared whole. The shard's configs,
 *  views and results are re-exported verbatim, so a batched call and a single one read the
 *  same and no second insertion or search vocabulary exists.
 *
 *  The split between template parameters and runtime arguments follows what costs anything. The
 *  quantization, metric, capability tier, the ForkUnion waiter, and the key hash resolve at
 *  compile time because each sits inside a hot loop - a runtime waiter would be an indirect
 *  call per contended word, a runtime hash one per reverse-plane probe. Topology and placement
 *  stay runtime arguments because they are read once per shard and once per fan-out, so nothing
 *  is bought by burning them into the type, and a great deal is lost: a domain count in the
 *  type multiplies every instantiation the C surface compiles.
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

    /** @brief The shard type this index is a plural of - the geometry parameters in the shard's
     *      own order, so the plural and its shards can never disagree on either. Placement is not a
     *      choice: graph storage lands on the shard's memory domain, and traversal scratch is
     *      carved from the walking thread's own run. */
    using index_t = dense_ram_index<vector_dtype_, kernel_kind_, capability_, wait_type_, //
                                    forkunion_allocator<std::byte>,                       //
                                    forkunion_allocator<std::byte>,                       //
                                    arena_allocator<std::byte, forkunion_allocator<std::byte>>>;

    /** @brief The threads and per-thread scratch every bulk verb runs on; it knows nothing of what
     *      walks, so one executor serves every instantiation the surface compiles. */
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

    dense_ram_indexes() = default;
    dense_ram_indexes(dense_ram_indexes&& other) noexcept { swap(other); }
    dense_ram_indexes& operator=(dense_ram_indexes&& other) noexcept { return swap(other), *this; }
    dense_ram_indexes(dense_ram_indexes const&) = delete;
    dense_ram_indexes& operator=(dense_ram_indexes const&) = delete;
    /** @brief Every member is RAII, so the shards die with the index and nothing leaks. */
    ~dense_ram_indexes() noexcept = default;

    void swap(dense_ram_indexes& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::size_t const admitted = admitted_rows_.load(std::memory_order_relaxed);
        admitted_rows_.store(other.admitted_rows_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.admitted_rows_.store(admitted, std::memory_order_relaxed);
    }

    std::size_t shards_count() const noexcept { return shards_.size(); }
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** @brief The frozen geometry, taken from the first shard - @c emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** @brief Published entries across every shard. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->size();
        return total;
    }

    /** @brief Rows the admission gate can let in - the key planes' total, split at @ref reserve
     *      and never grown, so no second word can disagree with what an id can actually record.
     *      Non-zero exactly once @ref reserve has run, which is what closes the shard set. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.keys.size();
        return total;
    }

    /** @brief Entries stored under @p key; zero means the index holds none. A row lands here only
     *      after it published, so a concurrent insertion can make a stale count under-report,
     *      never claim a row a search cannot reach. */
    std::size_t count(key_t const& key) const noexcept { return locators_of_key_.count(key); }

    /** @brief Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return locators_of_key_.contains(key); }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  Safe against concurrent insertions, which can only make it under-report a row still in
     *  flight - the same benign direction @ref count documents.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept {
        std::size_t const row_bytes = divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>()) *
                                      sizeof(scalar_t);
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

    /** @brief Bytes the whole sharded index occupies - every shard's graph storage, its key
     *      plane, and the shared reverse plane; traversal scratch belongs to executors and
     *      contexts, so their owners count it. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(dense_ram_indexes) + locators_of_key_.memory_usage();
        for (shard_t const& one : shards_) bytes += one.index->memory_usage() + one.keys.memory_usage();
        return bytes;
    }

    /** @brief Drops every shard, so the index is placeable again. */
    void clear() noexcept {
        shards_.clear();
        locators_of_key_.clear();
        admitted_rows_.store(0, std::memory_order_relaxed);
    }

    /** @brief Appends one shard built here on @p memory_domain, for callers with none to hand over. */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        forkunion_allocator<std::byte> const allocator {memory_domain};
        std::unique_ptr<index_t> shard(new (std::nothrow) index_t(allocator, allocator));
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** @brief Appends one shard the caller already built, consuming it; refuses any whose frozen
     *      geometry differs from the shards already here. Its key plane lands wherever the shard's
     *      own allocator places its graph, so the two never end up on different nodes.
     *
     *      Refused once storage exists: @ref reserve closes the shard set, so every batched verb
     *      walks a directory that cannot move under it and needs no synchronization to read. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        keys_plane_t keys {forkunion_allocator<std::byte>(shard->vectors_allocator_.memory_domain)};
        return shards_.push_back(shard_t {std::move(shard), std::move(keys)});
    }

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

        // A shard's key plane sizes to the same split as its graph and is never grown afterwards:
        // a search resolves a ranking through it while rows are still landing, so moving it would
        // strand a reader mid-translation. The reverse plane is sized never to rehash.
        for (shard_t& one : shards_) {
            if (auto const reserved = one.index->reserve(per_shard); reserved.status != status_t::success_k)
                return reserved.status;
            // Sized to the logical split, not to the shard's chunk-rounded grant: the plane is
            // what the admission gate measures against, so its length IS the shard's ceiling.
            if (failed(one.keys.resize(per_shard.capacity))) return status_t::bad_alloc_k;
        }

        // The quota covers every row the admission gate can let in - the key planes' total - so
        // recording an inserted row's locator is infallible by construction, not by checking.
        return locators_of_key_.reserve(capacity());
    }

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

        // Admission before anything is taken: past the ceiling, an id would land beyond its
        // shard's key plane, so the row is refused here rather than recorded nowhere.
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

        // The key lands where the id is assigned, ahead of publication, so a search that reaches
        // the node always reads it; the reverse plane follows publication, so a key only becomes
        // countable for a node that is already searchable - stale counts under-report, never lie.
        dense_node_id_t assigned = 0;
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        one.index->insert_reserved(row, {}, traversal, std::move(claim), config,
                                   [&](dense_node_id_t assigned_id) noexcept {
                                       assigned = assigned_id;
                                       one.keys[assigned_id] = key;
                                   });
        // The reverse plane's quota covers the whole reservation, so a row that won a claim has a
        // slot waiting for it and this insert cannot fail.
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
     *  Node ids never cross this boundary: they are shard-local, and which shard a row landed in
     *  is the index's own business. The keys are what the caller gets its rows back by.
     *
     *  The batch is all-or-nothing across every shard, and @ref reserve is the ceiling: a batch
     *  the remaining admission cannot house is refused whole with @c overflow_k before any claim
     *  is taken. Routing is decided at admission, so every shard's demand is known and claimed up
     *  front, and refused claims credit themselves back on the way out. Past that point nothing
     *  can fail - the walk itself has no failure exit - so two executors calling here
     *  concurrently interleave their rows without either one tearing.
     *
     *  One executor drives one call at a time: the arenas rewind here, so sharing an executor
     *  between two concurrent calls would let one call carve over the other's scratch.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_vectors The batch, one row per entry, in the shards' stored scalar type.
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config The per-shard insertion configuration.
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

        // Rotating the split by the rows admitted so far is what keeps the shards even: the
        // remainder rows land on a different shard every batch, so a stream of small batches
        // cannot drift onto shard zero. Admission is also the ceiling - the key planes split at
        // reserve() and never grow, so a row past them would have nowhere to record its key. One
        // relaxed read-modify-write for the whole batch.
        std::size_t const rows_ceiling = capacity();
        std::size_t routing_base = admitted_rows_.load(std::memory_order_relaxed);
        do {
            if (rows_ceiling - routing_base < rows) return result.status = status_t::overflow_k, result;
        } while (!admitted_rows_.compare_exchange_weak(routing_base, routing_base + rows, std::memory_order_relaxed));
        std::size_t const rotation = routing_base % shards;
        std::size_t const rows_per_shard = rows / shards;
        std::size_t const remainder = rows % shards;

        growing_buffer<claim_t> claims;
        if (failed(claims.reserve_exactly(shards))) return result.status = status_t::bad_alloc_k, result;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const turn = (shard + shards - rotation) % shards;
            auto [claim, granted] = shards_[shard].index->claim(rows_per_shard + (turn < remainder));
            if (failed(granted)) return result.status = granted, result;
            if (failed(claims.push_back(std::move(claim)))) return result.status = status_t::bad_alloc_k, result;
        }

        growing_buffer<traversal_t> traversals;
        if (status_t const ready = reserve_traversals(traversals, executor, config); failed(ready))
            return result.status = ready, result;

        // The key lands where the id is assigned, ahead of publication, so a search that reaches
        // the node always reads it; the reverse plane follows publication, so a key only becomes
        // countable for a node that is already searchable - stale counts under-report, never lie.
        // The reverse plane's quota covers the whole reservation, so a row that won a claim has a
        // slot waiting for it and recording cannot fail.
        auto insert_row = [&](std::size_t task, std::size_t thread, std::size_t) noexcept {
            std::size_t const shard = (routing_base + task) % shards;
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
        if (status_t const fanned = executor.for_tasks(rows, insert_row); failed(fanned))
            return result.status = fanned, result;

        // Every row dispensed an id, so the claims are spent rather than credited back.
        for (claim_t& claimed : claims) claimed.consume();

        for (traversal_t const& traversal : traversals) {
            result.visited_members += traversal.visited_members;
            result.computed_distances += traversal.computed_distances;
            result.computed_distances_in_refines += traversal.computed_distances_in_refines;
            result.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
            result.restarts += traversal.restarts;
        }
        result.new_size = size();
        return result;
    }

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending by
     *      distance - the honest end-to-end multi-domain lookup.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[inout] traversal The calling thread's traversal context, reused per shard.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[inout] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the index's own forward plane, which is why shards with colliding id spaces still land
     *      in one shared ranking.
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
     *  @param[out] keys Merged rankings, @b [queries,wanted] row-major.
     *  @param[out] distances Distances of the same rankings, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *
     *  A thread that fails stops its own query and leaves the rest to finish; the first failure is
     *  what the merged result reports, and every other query's outputs stand.
     */
    search_result_t search(vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted,
                           std::span<key_t> keys, std::span<distance_t> distances,
                           std::span<std::size_t> counts) noexcept {
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

        growing_buffer<traversal_t> traversals;
        if (status_t const ready = reserve_traversals(traversals, executor, insert_config_t {config.expansion_factor});
            failed(ready))
            return merged.status = ready, merged;

        std::atomic<status_t> first_refusal {status_t::success_k};
        auto scatter_query = [&](std::size_t query, std::size_t thread, std::size_t) noexcept {
            traversal_t& traversal = traversals[thread];
            std::span<key_t> const keys_row {keys.data() + query * wanted, wanted};
            std::span<distance_t> const distances_row {distances.data() + query * wanted, wanted};
            std::size_t found_count = 0;
            for (shard_t const& one : shards_) {
                search_result_t const found = one.index->search(queries.row(query).as_vector(), traversal, config);
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
        }
        for (std::size_t const count : counts) merged.count += count;
        return merged;
    }

  private:
    /** @brief Where one entry lives: the shard, and its shard-local id. A shard-local id alone
     *      names nothing, so this pair is the index's only address for an entry. */
    struct key_locator_t {
        std::size_t shard = 0;
        dense_node_id_t node = 0;
    };

    /** @brief Keys of one shard's rows, indexed by that shard's own node ids. */
    using keys_plane_t = growing_buffer<key_t, forkunion_allocator<std::byte>>;
    /** @brief Key to every entry stored under it; insert-only, wait-free for readers. */
    using locators_plane_t = growing_multimap<key_t, key_locator_t, key_hash_type_, //
                                              forkunion_allocator<std::byte>>;

    /** @brief One shard of the plural: the graph, and the keys of its rows. An index holds atomics
     *      and never moves, so the indirection is what keeps it address-stable; the memory domain
     *      is not repeated here, since the index's own allocator already carries it. */
    struct shard_t {
        std::unique_ptr<index_t> index;
        keys_plane_t keys;
    };

    /**
     *  @brief Builds one traversal context per worker over that worker's own arena, so a fan-out
     *      never allocates, never shares scratch and never draws the same landmarks twice.
     *
     *  Sized to the whole executor rather than to the task count: which workers a dynamic
     *  fan-out hands tasks to is not knowable here. The executor's @c reserve already rewound the
     *  arenas, so the previous call's carves are reusable rather than cumulative.
     */
    status_t reserve_traversals(growing_buffer<traversal_t>& traversals, executor_t& executor,
                                              insert_config_t config) noexcept {
        std::size_t const threads = executor.size();
        if (!threads) return status_t::uninitialized_k;
        if (failed(traversals.reserve_exactly(threads))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != threads; ++thread) {
            traversal_t traversal {arena_allocator<std::byte, forkunion_allocator<std::byte>>(executor.arena(thread))};
            // Apart, and apart across calls: identically seeded contexts would send every
            // concurrent inserter through the same entry points.
            traversal.random_generator.state = admitted_rows_.load(std::memory_order_relaxed) + thread;
            if (failed(traversals.push_back(std::move(traversal)))) return status_t::bad_alloc_k;
            if (status_t const sized = shards_[0].index->reserve_traversal(traversals[thread], config); failed(sized))
                return sized;
        }
        return status_t::success_k;
    }

    /** @brief The owned shards, graph and keys together. A shard's key plane is sized at
     *      @ref reserve and never grown, since a search reads it while rows are still landing. */
    growing_buffer<shard_t> shards_;
    /** @brief Every key's entries. Duplicates coexist as separate slots, which is exactly what
     *      several values under one key means, and costs one slot per row rather than per key. */
    locators_plane_t locators_of_key_;
    /** @brief Admission grants so far - the gate against @ref capacity and the rotation base. A
     *      batch refused at the gate takes nothing; one that fails past it leaves its grant
     *      spent, so the word stays monotone and the round-robin split can never route a shard
     *      past its key plane. Nothing publishes through this word - one relaxed read-modify-write
     *      per batch, never per row. */
    std::atomic<std::size_t> admitted_rows_ {0};
};

} // namespace usearch
} // namespace unum
