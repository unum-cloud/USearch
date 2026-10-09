/**
 *  @file include/usearch/dense/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The sharded index over CPU-resident dense shards - one per memory domain - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The plural layer over @c dense_ram_index: same-geometry shards, each owned and pinned to one
 *  ForkUnion memory domain, viewed as one index. Insertions rotate across the shards so they fill
 *  evenly; searches scatter over every shard and gather through @ref merge_search_results,
 *  resolving matches into the caller's keys, so shards with colliding ID spaces land in one
 *  ranking. Checkpoints ride the SafeGraphs interchange - one consolidated monolith, or one per
 *  shard beside a manifest - written positionally with every header last, so a failed save reads
 *  as absent.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memcpy`, `std::memcmp`, `std::memset`

#include <array>       // `std::array`
#include <atomic>      // `std::atomic`
#include <memory>      // `std::unique_ptr`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::move`

#include <forkunion.h> // `fu_memory_domain_id_t`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`, `forkunion_executor_t`
#include "usearch/utilities/ram_tables.hpp"        // `chunked_multimap`
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths & manifests
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_index.hpp"

namespace unum {
namespace usearch {

template <typename indexes_type_>
struct dense_ram_indexes_cursor;

/** Resolves a walked node ID to its key through one shard's key column, rebound per shard. */
template <typename keys_type_>
struct dense_keyed_ids {

    /** The key column of the shard being walked. */
    keys_type_ const* keys = nullptr;

    auto operator()(dense_node_id_t node_id) const noexcept { return (*keys)[node_id]; }
};

/**
 *  @brief The sharded index: same-geometry CPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are
 *  expressible. Shards arrive one at a time through @ref emplace, so the index grows as domains
 *  are engaged rather than being declared whole. The shard's configs, views and results are
 *  re-exported verbatim, so a batched call and a single one read the same.
 */
template <                                                                    //
    typename key_type_ = uuid_t,                                              //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                     //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                      //
    nk_capability_t capability_ = nk_cap_icelake_k,                           //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = fu::standard_cache_hints_t,                  //
    typename key_hash_type_ = std::hash<key_type_>                            //
    >
struct dense_ram_indexes {
    friend struct dense_ram_indexes_cursor<dense_ram_indexes>;

    /** A reader's walk over every shard's published rows in placement order. */
    using cursor_t = dense_ram_indexes_cursor<dense_ram_indexes>;

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;

    /** Where one entry lives: the shard, and its shard-local ID - the index's only address for
     *  an entry. */
    struct key_locator_t {

        /** The shard holding the entry, positional in placement order. */
        std::size_t shard = 0;

        /** The entry's ID inside that shard's own space. */
        dense_node_id_t node = 0;
    };

    /** Key to every entry stored under it; insert-only, wait-free for readers, growing in chunks. */
    using locators_map_t =
        chunked_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t, atomic_ref_template_>;

    /** The atomic reference owning the plain synchronization words - the shards' and this index's
     *  own routing tally alike. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** The shard type - geometry parameters in the shard's own order, so the two can never
     *  disagree. */
    using index_t = dense_ram_index<vector_dtype_, kernel_kind_, capability_, wait_type_, atomic_ref_template_, //
                                    cache_hints_type_,                                                          //
                                    forkunion_allocator_t,                                                      //
                                    forkunion_allocator_t,                                                      //
                                    arena_allocator<byte_t, forkunion_allocator_t>>;

    /** Keys of one shard's rows, indexed by that shard's own node IDs, growing on the shard's own schedule. */
    using keys_column_t = chunked_column<key_t, typename index_t::chunk_schedule_t, index_t::max_node_count_k,
                                         forkunion_allocator_t, atomic_ref_template_>;

    /** One shard: the graph, and the keys of its rows. Placement is not repeated - the index's
     *  own allocator carries it. */
    struct shard_t {

        /** The graph, growing on demand. An index holds atomics and never moves. */
        std::unique_ptr<index_t> index;

        /** The key of each of this shard's rows, indexed by node ID; every claimer covers what the shard hosts
         *  after its claim, so any row a claim can dispense has a key slot before its key lands. */
        keys_column_t keys;
    };

    /** The threads and their scratch every bulk verb runs on; it names no index, so one executor
     *  serves every instantiation. */
    using executor_t = forkunion_executor_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;
    using traversal_t = typename index_t::dense_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using refine_result_t = typename index_t::refine_result_t;
    using health_result_t = typename index_t::health_result_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;
    static_assert(index_family<index_t> && index_claim<shard_capacity_claim_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The distance kernel this instantiation walks with. */
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;

    /** The compute-capability mask the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

#pragma region Placement

    /** Every index names the host allocator its shard table, locator map and checkpoint
     *  catalogs sit on; nothing is placed by default. */
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

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~dense_ram_indexes() noexcept = default;

    void swap(dense_ram_indexes& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::size_t const routed = atomic_ref<std::size_t>(routed_rows_).load(std::memory_order_relaxed);
        atomic_ref<std::size_t>(routed_rows_)
            .store(atomic_ref<std::size_t>(other.routed_rows_).load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        atomic_ref<std::size_t>(other.routed_rows_).store(routed, std::memory_order_relaxed);
    }

    /** Appends one shard built here on @p memory_domain, for callers with none to hand over. */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(memory_domain);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** Appends one shard the caller already built, consuming it; refuses any whose frozen geometry
     *  differs from the shards already here, and any already reserved. Refused once storage
     *  exists - @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (shard->capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        keys_column_t keys {forkunion_allocator_t(shard->vectors_allocator_.memory_domain)};
        return shards_.push_back(shard_t {std::move(shard), std::move(keys)});
    }

    /** Empties every shard in place, keeping its memory domain, geometry and capacity. Demands
     *  the same quiescence as @ref reserve. */
    status_t clear() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = build_(one.index->vectors_allocator_.memory_domain);
            if (!fresh) return status_t::bad_alloc_k;
            one.index = std::move(fresh);
        }
        locators_of_key_.clear();
        atomic_ref<std::size_t>(routed_rows_).store(0, std::memory_order_relaxed);
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

#pragma endregion Placement

#pragma region Measures

    /** Shards in the index, one per engaged memory domain. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }

    /** The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** Entries across every shard - each shard's own @c size, summed. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->size();
        return total;
    }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->capacity();
        return total;
    }

    /** Bytes the index occupies - every shard's storage, its key column, and the shared locator
     *  map; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(dense_ram_indexes) + locators_of_key_.memory_usage();
        for (shard_t const& one : shards_) bytes += one.index->memory_usage() + one.keys.memory_usage();
        return bytes;
    }

#pragma endregion Measures

#pragma region Reservation

    /**
     *  @brief Grows every shard ahead of @p config.capacity rows, split evenly over them - a hint, since the shards,
     *      their key columns and the locator map all grow on demand; the first call freezes the geometry.
     *
     *  Callable any time, alongside insertions and searches: nothing it grows ever moves.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (shards_.empty()) return {0, status_t::uninitialized_k};
        reserve_capacity_t per_shard = config;
        per_shard.capacity = divide_round_up(config.capacity, shards_.size());
        for (shard_t& one : shards_) {
            auto const [shard_capacity, status] = one.index->reserve(per_shard);
            if (failed(status)) return {capacity(), status};
            if (status_t const covered = one.keys.cover(shard_capacity); failed(covered)) return {capacity(), covered};
        }
        if (status_t const indexed = locators_of_key_.reserve(config.capacity); failed(indexed))
            return {capacity(), indexed};
        return {capacity(), status_t::success_k};
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
    expected<insert_result_t> insert(key_t const& key, vector_view_t new_vector, traversal_t& traversal,
                                     insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (shards_.empty()) return {result, status_t::uninitialized_k};

        // The routing tally only names the first shard to try; a full one passes the row on to the next,
        // and any other refusal stops the probe.
        std::size_t const shards = shards_.size();
        std::size_t const routing_base = atomic_ref<std::size_t>(routed_rows_).fetch_add(1, std::memory_order_relaxed);
        shard_capacity_claim_t claim;
        std::size_t probe = 0;
        for (; probe != shards; ++probe) {
            auto [shard_claim, claim_status] = shards_[(routing_base + probe) % shards].index->claim(1);
            if (succeeded(claim_status)) {
                claim = std::move(shard_claim);
                break;
            }
            if (claim_status != status_t::overflow_k) return {result, claim_status};
        }
        if (probe == shards) return {result, status_t::overflow_k};
        std::size_t const shard = (routing_base + probe) % shards;

        shard_t& one = shards_[shard];
        // Covering what the shard hosts after this claim is what lets any later dispense in a fresh chunk find its
        // keys.
        if (status_t const covered = one.keys.cover(one.index->capacity()); failed(covered)) return {result, covered};
        auto [seats, seat_status] = locators_of_key_.claim(1);
        if (failed(seat_status)) return {result, seat_status};
        if (status_t const sized = one.index->reserve_traversal(traversal, config); failed(sized))
            return {result, sized};

        // The context's counters run across calls, so the report is the difference this one made.
        std::size_t const visited_before = traversal.visited_members;
        std::size_t const computed_before = traversal.computed_distances;
        std::size_t const refines_before = traversal.computed_distances_in_refines;
        std::size_t const reverse_refines_before = traversal.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = traversal.restarts;

        // The key lands before publication, so a reachable node always reads one.
        dense_node_id_t assigned = 0;
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        one.index->insert(std::move(claim), row, {}, traversal, config, [&](dense_node_id_t assigned_id) noexcept {
            assigned = assigned_id;
            one.keys[assigned_id] = key;
        });
        locators_of_key_.insert(seats, key, key_locator_t {shard, assigned});

        result.visited_members = traversal.visited_members - visited_before;
        result.computed_distances = traversal.computed_distances - computed_before;
        result.computed_distances_in_refines = traversal.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = traversal.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = traversal.restarts - restarts_before;
        result.new_size = size();
        return {result, status_t::success_k};
    }

    /**
     *  @brief Inserts a whole matrix of rows under @p keys, split across the shards from a rotating
     *      start so they fill evenly - the executor-driven sibling of one insertion.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row landed in
     *  is the index's business. The keys are what the caller gets its rows back by.
     *
     *  All-or-nothing across shards: every claim is taken before any row
     *  lands, and past that point nothing can fail. One executor drives one call at a time.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_vectors The batch, one row per entry, in the shards' stored scalar type.
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, vectors_view_t new_vectors, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (keys.size() != new_vectors.extent(0)) return {result, status_t::config_mismatch_k};
        auto [shard_claim, status] = claim(new_vectors.extent(0));
        if (failed(status)) return {result, status};
        return insert(std::move(shard_claim), keys, new_vectors, executor, config);
    }

    /**
     *  @brief Storage claimed across every shard and not yet inserted against - the fallible half
     *      of a batched insertion, so a caller commits or is refused as a whole.
     *
     *  Owns one shard claim per shard and the rotation they were cut under; an abandoned claim
     *  releases every one of them, and a consumed one releases none.
     */
    struct [[nodiscard]] capacity_claim_t {
        capacity_claim_t() noexcept : capacity_claim_t(forkunion_allocator_t {}) {}
        /** @brief An empty claim over @p allocator; only @ref claim ever fills it. */
        explicit capacity_claim_t(forkunion_allocator_t allocator) noexcept : claims_(allocator) {}
        capacity_claim_t(capacity_claim_t&& other) noexcept
            : claims_(std::move(other.claims_)), rotation_(other.rotation_) {}
        capacity_claim_t& operator=(capacity_claim_t&&) = delete;
        capacity_claim_t(capacity_claim_t const&) = delete;
        capacity_claim_t& operator=(capacity_claim_t const&) = delete;
        /** @brief Every shard claim releases whatever it never dispensed against. */
        ~capacity_claim_t() noexcept = default;

        /** @brief Rows this claim still holds; zero once an insertion has taken it. */
        std::size_t rows() const noexcept {
            std::size_t held = 0;
            for (shard_capacity_claim_t const& one : claims_) held += one.rows();
            return held;
        }
        explicit operator bool() const noexcept { return rows() != 0; }

        /** @brief Marks every shard's rows as taken, so destruction releases nothing. */
        void consume() noexcept {
            for (shard_capacity_claim_t& one : claims_) one.consume();
        }

      private:
        /** @brief One claim per shard, in placement order. */
        growing_buffer<shard_capacity_claim_t, forkunion_allocator_t> claims_;
        /** @brief The rotation the shares were cut under - which shard the batch's first row takes. */
        std::size_t rotation_ = 0;

        friend dense_ram_indexes;
    };

    /**
     *  @brief Claims storage for @p rows across every shard without assigning any identity - the
     *      step a batch runs before touching any shard.
     *
     *  Rotating by rows routed keeps a stream of small batches from drifting onto shard zero. Each shard's key
     *  column is covered to what the shard hosts after its share, and a refusal releases whatever was already
     *  granted.
     */
    expected<capacity_claim_t> claim(std::size_t rows) noexcept {
        std::size_t const shards = shards_.size();
        capacity_claim_t claimed {allocator_};
        if (!shards) return {{}, status_t::uninitialized_k};
        if (!rows) return {std::move(claimed), status_t::success_k};
        if (failed(claimed.claims_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        claimed.rotation_ = atomic_ref<std::size_t>(routed_rows_).fetch_add(rows, std::memory_order_relaxed) % shards;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_t& one = shards_[shard];
            auto [shard_claim, status] = one.index->claim(share_of_(shard, rows, claimed.rotation_));
            if (failed(status)) return {{}, status};
            claimed.claims_.push_back_reserved(std::move(shard_claim));
            if (status_t const covered = one.keys.cover(one.index->capacity()); failed(covered)) return {{}, covered};
        }
        return {std::move(claimed), status_t::success_k};
    }

    /**
     *  @brief Inserts @p new_vectors under @p keys against @p claim already obtained - the half of
     *      a batched insertion that cannot fail once the threads and their scratch are in hand.
     *
     *  @param[in] claim The rows, claimed across the shards under their rotation; consumed here.
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_vectors The batch, one row per entry, in the shards' stored scalar type.
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_result_t> insert(capacity_claim_t&& claim, std::span<key_t const> keys, vectors_view_t new_vectors,
                                     executor_t& executor, insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        std::size_t const rows = new_vectors.extent(0);
        std::size_t const shards = shards_.size();
        if (!shards) return {result, status_t::uninitialized_k};
        if (keys.size() != rows || claim.rows() != rows) return {result, status_t::config_mismatch_k};
        if (!rows) return {result, status_t::success_k};

        // One caller per executor; the arenas grow to the widest geometry ever asked of them.
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {result, claim_status};
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage(config)); failed(sized))
            return {result, sized};

        std::size_t const rotation = claim.rotation_;
        auto [seats, seat_status] = locators_of_key_.claim(rows);
        if (failed(seat_status)) return {result, seat_status};
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, config); failed(ready))
            return {result, ready};

        // The key lands before publication, and each row records its locator in its own claimed seat.
        auto insert_row = [&](std::size_t task, std::size_t thread, std::size_t) noexcept {
            std::size_t const shard = (rotation + task) % shards;
            shard_t& one = shards_[shard];
            dense_node_id_t assigned = 0;
            one.index->insert(shard_capacity_claim_t {}, new_vectors(nk::range(task, task + 1), nk::slice), {},
                              traversals[thread], config, [&](dense_node_id_t assigned_id) noexcept {
                                  assigned = assigned_id;
                                  one.keys[assigned_id] = keys[task];
                              });
            typename locators_map_t::seats_t seat {seats.next + task, seats.next + task + 1};
            locators_of_key_.insert(seat, keys[task], key_locator_t {shard, assigned});
        };
        // ForkUnion runs every task or none, so a refused fan-out releases the claims whole.
        if (status_t const fanned = executor.for_tasks(rows, insert_row); failed(fanned)) return {result, fanned};

        // Every row dispensed an ID, so the claim is spent rather than released.
        claim.consume();

        for (traversal_t const& traversal : traversals) {
            result.visited_members += traversal.visited_members;
            result.computed_distances += traversal.computed_distances;
            result.computed_distances_in_refines += traversal.computed_distances_in_refines;
            result.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
            result.restarts += traversal.restarts;
            result.saturated_walks += traversal.saturated_walks;
        }
        result.new_size = size();
        return {result, status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Refinement

    /**
     *  @brief Re-selects the neighbors of every published node across the shards, then heals whatever that strands -
     *      every shard's @c refine_plan_t drained over the threads, phase by phase.
     *
     *  Every thread pulls steps from its home shard first and steals from the rest in the tail, so a pass costs one
     *  sweep of the graph however the shards are filled. Re-selection claims no identity and moves no watermark, so
     *  searches run throughout and a stopped pass leaves the graph correct. Insertions may not overlap.
     *
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the population; a
     *      @c cancel_k answer skips the rest of the re-selection, still heals, and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};

        // A re-selection walks like any other query, so it sizes like one; the arenas rewind before the bitsets land.
        insert_config_t const walk_config {config.expansion_factor};
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage(walk_config));
            failed(sized))
            return {{}, sized};
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, walk_config); failed(ready))
            return {{}, ready};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> plans {allocator_};
        if (failed(cursors.resize(shards)) || failed(plans.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [plan, status] = shards_[shard].index->plan_refine({0, dense_node_id_unknown_k}, traversals[0],
                                                                    config);
            if (failed(status)) return {{}, status};
            plans.push_back_reserved(std::move(plan));
        }

        std::size_t const published = size();
        // Every step bumps it, so it takes its own line, away from the locals every worker reads.
        alignas(default_alignment_k) std::size_t swept = 0;
        bool cancelled = false;
        auto run_step = [&](std::size_t shard, typename index_t::refine_plan_t::step_t const& step, std::size_t thread,
                            nk::matrix_span<dense_node_id_t>) noexcept -> status_t {
            bool const reselecting = plans[shard].reselecting();
            if (status_t const ran = plans[shard].run(step, traversals[thread]); failed(ran)) return ran;
            if (!reselecting) return status_t::success_k;
            std::size_t const done = atomic_ref<std::size_t>(swept).fetch_add(step.nodes.size(),
                                                                              std::memory_order_relaxed) +
                                     step.nodes.size();
            return on_visited(done, published) == continuation_t::cancel_k ? status_t::cancelled_k
                                                                           : status_t::success_k;
        };
        std::span<std::size_t> const cursors_span = to_writable_span(cursors);
        status_t outcome = status_t::success_k;
        bool pending = true;
        while (pending) {
            std::memset(cursors_span.data(), 0, cursors_span.size_bytes());
            status_t const drained = drain_steps_(plans, cursors_span, executor, 0, run_step);
            // A cancelled re-selection still advances into the heal, so a stopped pass never leaves islands.
            if (drained == status_t::cancelled_k) cancelled = true;
            else if (failed(drained)) {
                outcome = drained;
                break;
            }
            pending = false;
            for (typename index_t::refine_plan_t& plan : plans) pending |= plan.advance();
        }

        refine_result_t result;
        for (typename index_t::refine_plan_t const& plan : plans) combine_refinement_(result, plan.finish().value);
        return {result, failed(outcome) ? outcome : cancelled ? status_t::cancelled_k : status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Health

    /**
     *  @brief Reachability of every shard from as many landmarks as one search draws, swept level
     *      by level over the threads. Read-only, so concurrent insertions leave fresh nodes uncounted.
     *
     *  @param[inout] executor The threads and their scratch; the bitsets land on the first thread's
     *      arena, which the reservation has already grown.
     */
    expected<health_result_t> health(executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (status_t const ready = reserve_audit_(traversals, cursors, executor); failed(ready)) return {{}, ready};

        growing_buffer<typename index_t::health_plan_t, forkunion_allocator_t> plans {allocator_};
        if (failed(plans.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [plan, status] = shards_[shard].index->plan_health(traversals[0]);
            if (failed(status)) return {{}, status};
            plans.push_back_reserved(std::move(plan));
        }
        if (status_t const audited = sweep_reachability_(plans, to_writable_span(cursors), executor); failed(audited))
            return {{}, audited};

        health_result_t result;
        for (typename index_t::health_plan_t const& plan : plans) {
            auto const [shard_health, status] = plan.finish();
            if (failed(status)) return {{}, status};
            combine_health_(result, shard_health);
        }
        return {result, status_t::success_k};
    }

#pragma endregion Health

#pragma region Search

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending by
     *      distance - the end-to-end multi-shard lookup.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[inout] traversal The calling thread's traversal context, reused per shard.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[inout] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' key columns, so colliding ID spaces still land in one ranking.
     *  @param[inout] distances Distances of the same ranking, parallel to @p keys.
     *
     *  A failing shard stops the scatter but keeps whatever earlier shards already merged, so the
     *  reported count and the ranking always agree.
     */
    expected<search_result_t> search(vector_view_t query_vector, traversal_t& traversal, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        search_result_t merged;
        status_t status = status_t::success_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            auto const [search_result, search_status] = shards_[shard].index->search(query_vector, traversal, config);
            if (failed(search_status)) {
                status = search_status;
                break;
            }
            merged.visited_members += search_result.visited_members;
            merged.computed_distances += search_result.computed_distances;
            merged.restarts += search_result.restarts;
            merged.saturated_walks += search_result.saturated_walks;
            merged.count = merge_search_results(to_span(traversal.results), shards_[shard].keys, keys, distances,
                                                merged.count);
        }
        return {merged, status};
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
     *
     *  A thread that fails stops its own query and leaves the rest to finish; the first failure is
     *  what the merged result reports, and every other query's outputs stand.
     */
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     std::size_t wanted, std::span<key_t> keys, std::span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept {
        return search_if(queries, executor, config, wanted, keys, distances, counts, dummy_predicate_t {});
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its key's indexer score against the heads of its query row.
     *
     *  @param[in] heads The indexer heads of every query row.
     */
    expected<search_result_t> search(vectors_view_t queries, dense_indexer_heads<scalar_t> const& heads,
                                     executor_t& executor, search_config_t config, std::size_t wanted,
                                     std::span<key_t> keys, std::span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k && !nk::is_integral_dtype<scalar_t>())
    {
        return search_rows_(queries, heads, executor, config, wanted, keys, distances, counts, dummy_predicate_t {});
    }

    /**
     *  @brief The filtered sibling of the batched @ref search, sharing its whole contract.
     *
     *  @param[in] predicate The per-key admission filter; @c dummy_predicate_t admits everything and
     *      compiles the filtered machinery out. Shards test node IDs, so each one's keys column
     *      translates before the predicate sees a candidate.
     */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vectors_view_t queries, executor_t& executor, search_config_t config,
                                        std::size_t wanted, std::span<key_t> keys, std::span<distance_t> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        return search_rows_(queries, dummy_heads_t {}, executor, config, wanted, keys, distances, counts, predicate);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row keeps
     *      the pairs its walks scored, keys resolved and dots at their native width.
     *
     *  @param[in] left The left-hand rows, one walk per row, in the shards' stored scalar type.
     *  @param[inout] executor The threads and their scratch; one row occupies one at a time.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] wanted Pairs kept per row - the row width of @p keys and @p products.
     *  @param[out] keys The key of each kept pair, @b [rows,wanted] row-major.
     *  @param[out] products The dot of each kept pair, parallel to @p keys.
     *  @param[out] counts Pairs scored per row across every shard, above @p wanted when truncated.
     *
     *  A full row keeps an arbitrary subset of its pairs. A thread that fails stops its own row and
     *  leaves the rest to finish; the first failure is what the merged result reports.
     */
    expected<search_result_t> multiply(vectors_view_t left, executor_t& executor, search_config_t config,
                                       std::size_t wanted, std::span<key_t> keys, std::span<dot_result_t> products,
                                       std::span<std::size_t> counts) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        std::size_t const rows_count = left.extent(0);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != rows_count || keys.size() < rows_count * wanted || products.size() < rows_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};

        auto multiply_row = [&](std::size_t row, std::size_t, traversal_t& traversal) noexcept {
            dense_multiply_fold<key_t, dot_result_t, dense_keyed_ids<keys_column_t>> fold {
                {keys.data() + row * wanted, wanted},
                {products.data() + row * wanted, wanted}};
            status_t refusal = status_t::success_k;
            for (shard_t const& one : shards_) {
                fold.resolve.keys = &one.keys;
                expected<search_result_t> const walked = one.index->search(left.row(row).as_vector(), fold, traversal,
                                                                           config);
                if (!walked) {
                    refusal = walked.status;
                    break;
                }
            }
            counts[row] = fold.scored;
            return refusal;
        };
        return for_each_row_(rows_count, executor, config, multiply_row, counts);
    }

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key among the tokens it ranked, and those maxima sum to that key's score.
     *
     *  @param[in] queries The token rows of every query, @p rows_per_query consecutive rows each.
     *  @param[in] rows_per_query Rows in every query; the row count must be a whole multiple of it.
     *  @param[inout] executor The threads and their scratch; one token row occupies one at a time.
     *  @param[in] config The per-shard walk configuration, applied uniformly; each row ranks as many
     *      tokens as its expansion factor.
     *  @param[in] wanted Keys kept per query - the row width of @p keys and @p scores.
     *  @param[out] keys The best-scoring keys of each query, @b [queries,wanted] row-major.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, at most @p wanted.
     *
     *  A key some row never ranked misses that row's maximum, so with non-negative per-row maxima
     *  a score lands below its exact value, never above.
     */
    expected<search_result_t> maxsim(vectors_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, std::size_t wanted, std::span<key_t> keys,
                                     std::span<f64_t> scores, std::span<std::size_t> counts) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        std::size_t const rows_count = queries.extent(0);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
        std::size_t const groups_count = rows_count / rows_per_query;
        if (counts.size() != groups_count || keys.size() < groups_count * wanted ||
            scores.size() < groups_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!groups_count) return {{}, status_t::success_k};

        // Every token row keeps its whole ranked pool, its keys resolved and merged across the shards.
        std::size_t const width = shards_[0].index->non_zero_expansion_factor(config);
        growing_buffer<key_t, forkunion_allocator_t> row_keys {allocator_};
        growing_buffer<distance_t, forkunion_allocator_t> row_distances {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> row_counts {allocator_};
        dense_maxsim_reducer<key_t, forkunion_allocator_t> reduce {allocator_};
        if (failed(row_keys.resize_uninitialized(rows_count * width)) ||
            failed(row_distances.resize_uninitialized(rows_count * width)) ||
            failed(row_counts.resize_uninitialized(rows_count)) ||
            failed(reduce.reserve(rows_per_query, width, wanted)))
            return {{}, status_t::bad_alloc_k};
        expected<search_result_t> ranked = search(queries, executor, config, width, to_writable_span(row_keys),
                                                  to_writable_span(row_distances), to_writable_span(row_counts));
        if (!ranked) return ranked;
        ranked.value.count = reduce(to_span(row_keys), to_span(row_distances), to_span(row_counts), keys, scores,
                                    counts);
        return ranked;
    }

    /**
     *  @brief Softmax attention of query heads over every key their walks score across the shards: each token
     *      walks every shard in turn, folding into one running state per head, since the shards are disjoint
     *      and online softmax is order-free, and its rows normalize once after the last shard.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t; one walk per token and shard.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's keys to their value rows, as wide as the keys and stored
     *      like them: `void(std::span<key_t const>, std::span<vector_view_t>)`, one row per key. Keys come from
     *      each shard's key column, so only scored nodes resolve; one without a row refuses with
     *      @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys.
     *  @param[out] carry Every query row's softmax state, which merges this call with other branches.
     *  @param[inout] executor The threads and their scratch; one token occupies one at a time.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @return The query rows attended in @c count.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, std::span<f32_t> outputs,
                                     std::span<dense_softmax_state_t> carry, executor_t& executor,
                                     search_config_t config = {}) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        std::size_t const rows_count = queries.extent(0), columns = queries.extent(1);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.size() != rows_count * columns || carry.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};

        auto attend_token = [&](std::size_t token, std::size_t, traversal_t& traversal) noexcept {
            std::size_t const first = token * head_count;
            dense_keyed_ids<keys_column_t> keyed;
            auto shard_lookup = [&](std::span<dense_node_id_t const> nodes, std::span<vector_view_t> rows) noexcept {
                std::array<key_t, 4> keys;
                for (std::size_t lane = 0; lane != nodes.size(); ++lane) keys[lane] = keyed(nodes[lane]);
                value_lookup(std::span<key_t const>(keys).first(nodes.size()), rows);
            };
            status_t status = status_t::success_k;
            typename index_t::template attention_rescoring<decltype(shard_lookup)> const rescoring {
                queries(nk::range(first, first + head_count), nk::slice),
                shard_lookup,
                score_scale,
                outputs.subspan(first * columns, head_count * columns),
                carry.subspan(first, head_count),
                status};
            rescoring.clear();
            for (shard_t const& one : shards_) {
                keyed.keys = &one.keys;
                expected<search_result_t> const walked = one.index->walk_(
                    rescoring.queries.row(0).as_vector(), dummy_fold_t {}, rescoring, traversal, config);
                if (!walked) return walked.status;
                if (failed(status)) return status;
            }
            rescoring.finish();
            return status_t::success_k;
        };
        expected<search_result_t> attended = for_each_row_(rows_count / head_count, executor, config, attend_token, {});
        if (attended) attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** Bytes a @ref save under @p config writes: one monolith, or every part file and the
     *  manifest measured over a placeholder stem; zero for an invalid config or an unplaced
     *  index. The row count is read now, so a buffer sized here fits a save no row lands before. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        if (shards_.empty() || !dimensions()) return {0, status_t::uninitialized_k};
        if (status_t const valid = config.validate(); failed(valid)) return {0, valid};
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (status_t const named = custom_fields_(fields, config); failed(named)) return {0, named};
        std::span<safegraphs_custom_field_t const> const fields_span = to_span(fields);

        // A header that cannot be formatted is an allocation failure the caller must see, never
        // a zero it would size a file by.
        status_t framing = status_t::success_k;
        auto part_bytes = [&](std::size_t nodes) noexcept {
            auto const [header, status] = monolith_header_(nodes, config, fields_span);
            if (failed(status)) framing = status;
            return failed(status) ? 0 : header.monolith_bytes();
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
            auto const [name, status] = safegraphs_part_path(entry_path, "index", part, shards_.size());
            if (failed(status)) framing = status;
            return safegraphs_part_t {name, part_bytes(nodes), nodes};
        };
        auto const [manifest, status] = safegraphs_manifest_bytes(shards_.size(), part_at);
        if (failed(status)) return {0, status};
        if (failed(framing)) return {0, framing};
        return {total + manifest, status_t::success_k};
    }

    /** @brief One monolith of a checkpoint: where it is, what its header says, and the tokens
     *      its metadata carries. */
    struct part_t {
        /** @brief The relative path's characters inside the paths arena; empty names the
         *      checkpoint path itself. */
        iota_view<std::size_t> path;
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

    /**
     *  @brief One save in flight across every shard: one shard plan each, the parts they write
     *      into, and the tail @ref finish commits - the keys, the headers, and the manifest.
     *
     *  Steps are flat and each names its shard, so any number of threads take them from one
     *  cursor; @ref steps_of names one shard's own run, which is what a home-first drain follows.
     *  A dying plan drops its mappings and its shard plans, leaving parts no header commits.
     */
    struct [[nodiscard]] save_plan_t {
        /** @brief One step of one shard: its shard, and that shard's own step. */
        struct step_t {
            /** @brief The shard the rows come from. */
            std::size_t shard = 0;
            /** @brief The shard plan's own step. */
            typename index_t::save_plan_t::step_t inner;
        };

        save_plan_t() noexcept : save_plan_t(forkunion_allocator_t {}) {}
        /** @brief An empty plan over @p allocator; only @ref plan_save ever fills it. */
        explicit save_plan_t(forkunion_allocator_t allocator) noexcept
            : plans_(allocator), headers_(allocator), mappings_(allocator), fields_(allocator), path_(allocator) {}
        save_plan_t(save_plan_t&& other) noexcept
            : layout_(other.layout_), monolith_(other.monolith_), plans_(std::move(other.plans_)),
              headers_(std::move(other.headers_)), mappings_(std::move(other.mappings_)),
              fields_(std::move(other.fields_)), path_(std::move(other.path_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;
        /** @brief Every member is RAII: the mappings and the shard plans die with the plan. */
        ~save_plan_t() noexcept = default;

        /** @brief Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t shard = 0;
            std::size_t first = 0;
            for (; first + plans_[shard].steps().size() <= position; ++shard) first += plans_[shard].steps().size();
            step.shard = shard;
            step.inner = plans_[shard][position - first];
            return step;
        }

        /** @brief The steps the save spans - every shard's own, in placement order; none once spent. */
        indexed_range<save_plan_t> steps() const noexcept {
            std::size_t count = 0;
            if (owner_)
                for (typename index_t::save_plan_t const& plan : plans_) count += plan.steps().size();
            return {this, count};
        }

        /** @brief The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
        iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
            if (!owner_ || shard >= plans_.size()) return {};
            std::size_t first = 0;
            for (std::size_t earlier = 0; earlier != shard; ++earlier) first += plans_[earlier].steps().size();
            return {first, first + plans_[shard].steps().size()};
        }

        /** @brief Lands step @p step through its own shard's plan, rebasing through @p scratch. */
        status_t save(step_t const& step, nk::matrix_span<dense_node_id_t> scratch) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (step.shard >= plans_.size()) return status_t::config_mismatch_k;
            return plans_[step.shard].save(step.inner, scratch);
        }

        /**
         *  @brief Pronounces the checkpoint complete: every shard plan closes, its keys land at
         *      its base, the headers follow, and a distributed save commits with its manifest -
         *      the commit record, so a failed save leaves parts and no checkpoint.
         *
         *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k
         *      stops the remaining shards and reports @c cancelled_k.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t finish(progress_type_&& on_saved = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            bool const distributed = layout_ == checkpoint_layout_t::distributed_k;
            std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
            for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
                if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
                safegraphs_header_t const& header = headers_[distributed ? shard : 0];
                bytes_span_t const target = distributed ? mappings_[shard].span() : monolith_;
                std::size_t const nodes = plans_[shard].visible;
                std::size_t const base = plans_[shard].id_shift;
                // The key column is chunked, so its rows land run by run.
                bytes_span_t const key_rows = header.rows_of(target, header.key, base, nodes);
                std::size_t written = 0;
                status_t wrote = status_t::success_k;
                owner_->shards_[shard].keys.for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
                    if (succeeded(wrote))
                        wrote = copy_bytes(key_rows.subspan(written, run.size_bytes()),
                                           bytes_of(run.data(), run.size()));
                    written += run.size_bytes();
                });
                if (failed(wrote)) return wrote;
                if (distributed) {
                    if (status_t const framed = safegraphs_write_header(target, header, fields); failed(framed))
                        return framed;
                    if (status_t const flushed = mappings_[shard].sync(); failed(flushed)) return flushed;
                }
                if (on_saved(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
            }
            if (distributed) {
                if (status_t const committed = write_manifest_(); failed(committed)) return committed;
            }
            else {
                if (status_t const framed = safegraphs_write_header(monolith_, headers_[0], fields); failed(framed))
                    return framed;
                if (!mappings_.empty())
                    if (status_t const flushed = mappings_[0].sync(); failed(flushed)) return flushed;
            }
            owner_ = nullptr;
            return status_t::success_k;
        }

      private:
        /** @brief The manifest that commits a distributed save; entries name their parts
         *      relative to it. */
        status_t write_manifest_() noexcept {
            std::size_t const shards = plans_.size();
            std::string_view const path = to_string_view(path_);
            std::string_view const base = safegraphs_basename(safegraphs_stem(path));
            growing_buffer<char, forkunion_allocator_t> entry_path {owner_->allocator_};
            auto part_at = [&](std::size_t shard) noexcept {
                auto const [name, status] = safegraphs_part_path(entry_path, base, shard, shards);
                return safegraphs_part_t {name, headers_[shard].monolith_bytes(), plans_[shard].visible};
            };
            return safegraphs_write_manifest(path, shards, part_at);
        }

        /** @brief The layout the parts were framed in. */
        checkpoint_layout_t layout_ = checkpoint_layout_t::monolith_k;
        /** @brief The one graph every shard writes into; empty for a distributed save. */
        bytes_span_t monolith_;
        /** @brief One plan per shard, in placement order. */
        growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans_;
        /** @brief One header per part - one in all for a monolith. */
        growing_buffer<safegraphs_header_t, forkunion_allocator_t> headers_;
        /** @brief The files the parts were mapped into; empty when the caller owns the bytes. */
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;
        /** @brief The custom fields every header carries, outliving the plan's steps. */
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields_;
        /** @brief The checkpoint's own path, copied so the plan outlives the caller's view. */
        growing_buffer<char, forkunion_allocator_t> path_;
        /** @brief The index being checkpointed; null in the empty plan a refused @ref plan_save
         *      returns, and once @c finish committed. */
        dense_ram_indexes* owner_ = nullptr;

        friend dense_ram_indexes;
    };

    /**
     *  @brief Opens a save under @p path in the layout @p config names: one monolith of every
     *      shard's rows in placement order with IDs rebased, so it reads as one graph, or one
     *      monolith per shard beside a manifest.
     *
     *  Every part is framed, sized and mapped here, and every header lands in @c finish, so a
     *  failed or cancelled save reads as absent.
     *
     *  @param[in] path The monolith's path, or the manifest's - its parts land beside it as
     *      @c <stem>-00001-of-0000S.safegraphs.
     *  @param[in] config The layout, alignment, synchronization contract and provenance.
     *
     *  Under @c concurrent_k contention is detected, not excluded: a row landing mid-save
     *  reports @c contended_k.
     */
    expected<save_plan_t> plan_save(std::string_view path, save_config_t const& config) noexcept {
        if (status_t const ready = saveable_(config); failed(ready)) return {{}, ready};
        save_plan_t plan {allocator_};
        if (status_t const kept = keep_path_(plan.path_, path); failed(kept)) return {{}, kept};
        if (config.layout == checkpoint_layout_t::distributed_k) {
            if (status_t const opened = open_save_parts_(plan, config); failed(opened)) return {{}, opened};
            return {std::move(plan), status_t::success_k};
        }
        auto const [bytes, measure_status] = storage_requirements(config);
        if (failed(measure_status)) return {{}, measure_status};
        auto [mapping, map_status] = mapped_file_t::write(path, bytes);
        if (failed(map_status)) return {{}, map_status};
        if (failed(plan.mappings_.push_back(std::move(mapping)))) return {{}, status_t::bad_alloc_k};
        if (status_t const opened = open_save_monolith_(plan, plan.mappings_[0].span(), config); failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** @brief Opens a save into @p monolith as one graph - the implementation the path form maps
     *      for; a manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size the
     *      bytes by @ref storage_requirements; fewer answer @c overflow_k. */
    expected<save_plan_t> plan_save(bytes_span_t monolith, save_config_t const& config) noexcept {
        if (status_t const ready = saveable_(config); failed(ready)) return {{}, ready};
        if (config.layout != checkpoint_layout_t::monolith_k) return {{}, status_t::config_mismatch_k};
        save_plan_t plan {allocator_};
        if (status_t const opened = open_save_monolith_(plan, monolith, config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief One load in flight across every shard: the mapped parts, one shard plan each, and
     *      the flat steps their rows arrive in - one per part's share of one storage chunk.
     *
     *  A shard takes its parts in order, so a plan appends them at one base and @ref finish
     *  publishes every shard at once. A dying plan releases every claim and publishes nothing.
     */
    struct [[nodiscard]] load_plan_t {
        /** @brief One step of one shard: where it lands, which part it reads, and that part's own
         *      ID space alongside the shard plan's step. */
        struct step_t {
            /** @brief The shard the rows land in. */
            std::size_t shard = 0;
            /** @brief The part the rows come from. */
            std::size_t part = 0;
            /** @brief The rows of that part the shard takes - the ID space they speak. */
            iota_view<dense_node_id_t> source_range;
            /** @brief The shard plan's own step, cut to one part's share of one chunk. */
            typename index_t::load_plan_t::step_t inner;
        };

        load_plan_t() noexcept : load_plan_t(forkunion_allocator_t {}) {}
        /** @brief An empty plan over @p allocator; only @ref plan_load ever fills it. */
        explicit load_plan_t(forkunion_allocator_t allocator) noexcept
            : mappings_(allocator), parts_(allocator), landings_(allocator), plans_(allocator), steps_(allocator) {}
        load_plan_t(load_plan_t&& other) noexcept
            : mappings_(std::move(other.mappings_)), parts_(std::move(other.parts_)),
              landings_(std::move(other.landings_)), plans_(std::move(other.plans_)), steps_(std::move(other.steps_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;
        /** @brief Every member is RAII: the mappings die with the plan, and the shard plans
         *      release the rows they claimed unless @c finish published them. */
        ~load_plan_t() noexcept = default;

        /** @brief Rows the checkpoint lands across every shard - what its headers said. */
        std::size_t rows() const noexcept {
            std::size_t landing_rows = 0;
            for (typename index_t::load_plan_t const& plan : plans_) landing_rows += plan.rows;
            return landing_rows;
        }

        /** @brief Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            if (position >= steps().size()) return {};
            return steps_[position];
        }

        /** @brief The steps the load spans - every shard's own, in placement order; none once spent. */
        indexed_range<load_plan_t> steps() const noexcept { return {this, owner_ ? steps_.size() : 0}; }

        /** @brief The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
        iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
            if (!owner_) return {};
            std::size_t first = 0;
            while (first != steps_.size() && steps_[first].shard != shard) ++first;
            std::size_t bound = first;
            while (bound != steps_.size() && steps_[bound].shard == shard) ++bound;
            return {first, bound};
        }

        /** @brief Lands step @p step: its part's own columns, under that part's ID space, through
         *      the shard's own plan. */
        status_t load(step_t const& step) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (step.shard >= plans_.size() || step.part >= parts_.size()) return status_t::config_mismatch_k;
            part_t const& part = parts_[step.part];
            typename index_t::columns_view_t const columns {part.header.data_of(part.monolith, part.header.value),
                                                            part.header.data_of(part.monolith, part.header.neighbors)};
            return plans_[step.shard].load(step.inner, columns, step.source_range);
        }

        /**
         *  @brief Publishes the restored graph: every shard's keys land at its base and enter the
         *      locator map, then every shard plan publishes - the order an insert keeps, so a
         *      search never reaches a published row without its key.
         *
         *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k reports
         *      @c cancelled_k.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t finish(progress_type_&& on_loaded = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
                std::size_t first = plans_[shard].base;
                for (checkpoint_assignment_t const& landing : landings_) {
                    std::size_t const rows = landing.rows.size();
                    if (landing.shard != shard || !rows) continue;
                    part_t const& part = parts_[landing.part];
                    keys_column_t& keys = owner_->shards_[shard].keys;
                    if (status_t const covered = keys.cover(first + rows); failed(covered)) return covered;
                    auto [seats, seat_status] = owner_->locators_of_key_.claim(rows);
                    if (failed(seat_status)) return seat_status;
                    // The key column is chunked, so the part's keys land run by run.
                    bytes_view_t const part_keys = part.header.rows_of(part.monolith, part.header.key,
                                                                       landing.rows.first, rows);
                    std::size_t read_bytes = 0;
                    status_t read = status_t::success_k;
                    keys.for_each_span({first, first + rows}, [&](std::span<key_t> run) noexcept {
                        if (succeeded(read))
                            read = copy_bytes(bytes_of(run.data(), run.size()),
                                              part_keys.subspan(read_bytes, run.size_bytes()));
                        read_bytes += run.size_bytes();
                    });
                    if (failed(read)) return read;
                    for (std::size_t row = 0; row != rows; ++row)
                        owner_->locators_of_key_.insert(
                            seats, keys[first + row], key_locator_t {shard, static_cast<dense_node_id_t>(first + row)});
                    first += rows;
                }
                if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
            }
            for (std::size_t shard = 0; shard != plans_.size(); ++shard)
                if (on_loaded(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
            atomic_ref<std::size_t>(owner_->routed_rows_).store(rows(), std::memory_order_relaxed);
            owner_ = nullptr;
            return status_t::success_k;
        }

      private:
        /** @brief The files the parts were mapped from; every part's rows are read out of them. */
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;
        /** @brief The checkpoint's parts, vetted at planning. */
        growing_buffer<part_t, forkunion_allocator_t> parts_;
        /** @brief Which part lands where, in the order each shard appends them - the sequence
         *      @c finish walks to place the keys. */
        growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> landings_;
        /** @brief One plan per shard, in placement order. */
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans_;
        /** @brief The flat steps, grouped by shard. */
        growing_buffer<step_t, forkunion_allocator_t> steps_;
        /** @brief The index being restored; null in the empty plan a refused @ref plan_load hands
         *      back, and once @c finish published. */
        dense_ram_indexes* owner_ = nullptr;

        friend dense_ram_indexes;
    };

    /**
     *  @brief Opens a load of the checkpoint at @p path: the content says monolith or manifest,
     *      every part's header is vetted before a byte lands, the reservation comes from
     *      @p config, and the parts land per its placement - as written, part @c p on shard
     *      @c p mod S, or cut into one stripe per shard.
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] config The placement and the rows to reserve; zero fits the checkpoint.
     *
     *  The index must be empty; the plan reports the rows it will land through @c rows().
     */
    expected<load_plan_t> plan_load(std::string_view path, load_config_t const& config) noexcept {
        if (status_t const ready = loadable_(); failed(ready)) return {{}, ready};
        load_plan_t plan {allocator_};
        if (status_t const listed = catalog_(path, plan.mappings_, plan.parts_); failed(listed)) return {{}, listed};
        if (status_t const opened = open_load_(plan, config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** @brief Opens a load of one monolith held in @p monolith - the implementation the path form
     *      maps for. */
    expected<load_plan_t> plan_load(bytes_view_t monolith, load_config_t const& config) noexcept {
        if (status_t const ready = loadable_(); failed(ready)) return {{}, ready};
        load_plan_t plan {allocator_};
        if (status_t const listed = catalog_monolith_(monolith, plan.parts_); failed(listed)) return {{}, listed};
        if (status_t const opened = open_load_(plan, config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Serializes the index under @p path - the one-shot over @ref plan_save, its steps
     *      fanned out over @p executor.
     *
     *  @param[in] path The monolith's path, or the manifest's.
     *  @param[in] config The layout, alignment, synchronization contract and provenance.
     *  @param[inout] executor The threads the shard writes fan out over.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(path, config);
        if (failed(status)) return status;
        if (status_t const drained = drain_steps_(plan, executor); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /** Serializes the index into @p monolith as one graph - the one-shot over the bytes form of
     *  @ref plan_save. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(monolith, config);
        if (failed(status)) return status;
        if (status_t const drained = drain_steps_(plan, executor); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /**
     *  @brief Restores the checkpoint at @p path into the placed shards - the one-shot over
     *      @ref plan_load, its steps fanned out over @p executor.
     *
     *  The index must be empty. A cancelled or failed load clears it and reports why, so nothing
     *  is published and the next load starts clean.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(path, config);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        // Nothing is published by a failed or cancelled load: the shards go back to empty, once
        // the plan has died and released whatever it claimed.
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /** Restores one monolith held in @p monolith - the one-shot over the bytes form of
     *  @ref plan_load. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t monolith, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(monolith, config);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /** The checkpoint's metadata without loading it - @ref safegraphs_read_metadata. */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
    }

    /** The metadata of one monolith held in memory. */
    static expected<safegraphs_metadata_t> metadata(bytes_view_t buffer) noexcept {
        return safegraphs_read_metadata(buffer);
    }

#pragma endregion Checkpoints

#pragma region Key Columns

    /** Entries stored under @p key; a concurrent insertion can only make it under-report. */
    std::size_t count(key_t const& key) const noexcept { return locators_of_key_.count(key); }

    /** Whether at least one entry is stored under @p key. */
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

#pragma endregion Key Columns

  private:
#pragma region Placement

    /** One unreserved shard placed on @p memory_domain; null when the allocation fails. */
    static std::unique_ptr<index_t> build_(fu_memory_domain_id_t memory_domain) noexcept {
        forkunion_allocator_t const allocator {memory_domain};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(allocator, allocator));
    }

#pragma endregion Placement

#pragma region Insertion

    /** The first row of a batch that @p shard takes under @p rotation - a bijection over the
     *  shards, so the residue classes it names partition the batch exactly. */
    static std::size_t first_row_of_shard_(std::size_t shard, std::size_t shards, std::size_t rotation) noexcept {
        return (shard + shards - rotation) % shards;
    }

    /** How many of @p rows a shard starting at @p first takes, stepping by @p shards. */
    static std::size_t rows_from_(std::size_t rows, std::size_t first, std::size_t shards) noexcept {
        return first < rows ? divide_round_up(rows - first, shards) : 0;
    }

    /** Shard @p shard's share of a @p rows batch under @p rotation. */
    std::size_t share_of_(std::size_t shard, std::size_t rows, std::size_t rotation) const noexcept {
        return rows_from_(rows, first_row_of_shard_(shard, shards_.size(), rotation), shards_.size());
    }

    /** Builds one traversal context per thread over its own arena, so a fan-out never allocates
     *  and never shares scratch. Sized to the executor, since a dynamic fan-out picks its
     *  threads. */
    status_t reserve_traversals_(growing_buffer<traversal_t, forkunion_allocator_t>& traversals, executor_t& executor,
                                 insert_config_t config) noexcept {
        std::size_t const threads = executor.size();
        if (!threads) return status_t::uninitialized_k;
        if (failed(traversals.reserve_exactly(threads))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != threads; ++thread) {
            traversal_t traversal {arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(thread))};
            // Apart, and apart across calls: identically seeded contexts would send every
            // concurrent inserter through the same entry points.
            traversal.random_generator.state = atomic_ref<std::size_t>(routed_rows_).load(std::memory_order_relaxed) +
                                               thread;
            if (failed(traversals.push_back(std::move(traversal)))) return status_t::bad_alloc_k;
            if (status_t const sized = shards_[0].index->reserve_traversal(traversals[thread], config); failed(sized))
                return sized;
        }
        return status_t::success_k;
    }

#pragma endregion Insertion

#pragma region Health

    /** Folds one shard's pass into the index's: counts add, and the depth is the deepest shard's. */
    static void combine_refinement_(refine_result_t& into, refine_result_t const& one) noexcept {
        into.visited_nodes += one.visited_nodes;
        into.changed_nodes += one.changed_nodes;
        into.islands_found += one.islands_found;
        into.islands_reconnected += one.islands_reconnected;
        into.reachable_nodes += one.reachable_nodes;
        into.sweeps = max_of(into.sweeps, one.sweeps);
        into.visited_members += one.visited_members;
        into.computed_distances += one.computed_distances;
        into.computed_distances_in_refines += one.computed_distances_in_refines;
        into.computed_distances_in_reverse_refines += one.computed_distances_in_reverse_refines;
        into.restarts += one.restarts;
        into.saturated_walks += one.saturated_walks;
    }

    /** Sizes what an audit runs on: one context per thread, one cursor per shard. The reservation
     *  comes first, since it rewinds every arena, and the plans opened after it keep their
     *  bitsets. */
    status_t reserve_audit_(growing_buffer<traversal_t, forkunion_allocator_t>& traversals,
                            growing_buffer<std::size_t, forkunion_allocator_t>& cursors,
                            executor_t& executor) noexcept {
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage()); failed(sized))
            return sized;
        if (status_t const ready = reserve_traversals_(traversals, executor, {}); failed(ready)) return ready;
        return cursors.resize(shards_.size());
    }

    /** One reachability audit over every shard's plan: the bitsets restart, then every level
     *  sweeps every step and advances each plan once, until no level expands a node. */
    template <typename plan_type_>
    status_t sweep_reachability_(growing_buffer<plan_type_, forkunion_allocator_t>& plans,
                                 std::span<std::size_t> cursors, executor_t& executor) noexcept {
        for (plan_type_& plan : plans) plan.restart_sweep();
        auto sweep_step = [&](std::size_t shard, auto const& step, std::size_t,
                              nk::matrix_span<dense_node_id_t>) noexcept {
            return plans[shard].sweep(step), status_t::success_k;
        };
        for (std::size_t expanded = 1; expanded;) {
            std::memset(cursors.data(), 0, cursors.size_bytes());
            if (status_t const drained = drain_steps_(plans, cursors, executor, 0, sweep_step); failed(drained))
                return drained;
            expanded = 0;
            for (plan_type_& plan : plans) expanded += plan.advance_level();
        }
        return status_t::success_k;
    }

    /** Folds one shard's audit into the index's: populations add, and the depth is the deepest
     *  shard's, since a searcher waits for the last one. */
    static void combine_health_(health_result_t& into, health_result_t const& one) noexcept {
        into.audited_nodes += one.audited_nodes;
        into.reachable_nodes += one.reachable_nodes;
        into.sweeps = max_of(into.sweeps, one.sweeps);
    }

#pragma endregion Health

#pragma region Search

    /** The batched @ref search_if under @p heads, which only a @c dummy_predicate_t accompanies. */
    template <typename heads_type_, typename predicate_type_>
    expected<search_result_t> search_rows_(vectors_view_t queries, heads_type_ const& heads, executor_t& executor,
                                           search_config_t config, std::size_t wanted, std::span<key_t> keys,
                                           std::span<distance_t> distances, std::span<std::size_t> counts,
                                           predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count || keys.size() < queries_count * wanted ||
            distances.size() < queries_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!queries_count) return {{}, status_t::success_k};

        auto search_row = [&](std::size_t query, std::size_t, traversal_t& traversal) noexcept {
            std::span<key_t> const keys_row {keys.data() + query * wanted, wanted};
            std::span<distance_t> const distances_row {distances.data() + query * wanted, wanted};
            std::size_t found_count = 0;
            status_t refusal = status_t::success_k;
            for (shard_t const& one : shards_) {
                expected<search_result_t> found {};
                if constexpr (!std::is_same_v<heads_type_, dummy_heads_t>)
                    found = one.index->search(queries.row(query).as_vector(), heads.row(query), traversal, config);
                else if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    found = one.index->search(queries.row(query).as_vector(), traversal, config);
                else {
                    auto keyed = [&](dense_node_id_t node) noexcept { return predicate(one.keys[node]); };
                    found = one.index->search_if(queries.row(query).as_vector(), keyed, traversal, config);
                }
                if (!found) {
                    refusal = found.status;
                    break;
                }
                found_count = merge_search_results(to_span(traversal.results), one.keys, keys_row, distances_row, //
                                                   found_count);
            }
            counts[query] = found_count;
            return refusal;
        };
        return for_each_row_(queries_count, executor, config, search_row, counts);
    }

    /** Claims @p executor, sizes one traversal per thread, runs @p per_row over every row with its
     *  thread and that thread's traversal, and sums the threads' telemetry & the rows' @p counts; a row
     *  that refuses stops itself, and the first refusal is the one kept. */
    template <typename per_row_type_>
    expected<search_result_t> for_each_row_(std::size_t rows_count, executor_t& executor, search_config_t config,
                                            per_row_type_&& per_row, std::span<std::size_t const> counts) noexcept {
        // One caller per executor, and the arenas grow to the widest geometry ever asked of them.
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (status_t const sized = executor.reserve(
                shards_[0].index->traversal_memory_usage(insert_config_t {config.expansion_factor}));
            failed(sized))
            return {{}, sized};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, insert_config_t {config.expansion_factor});
            failed(ready))
            return {{}, ready};

        status_t first_refusal = status_t::success_k;
        auto each_row = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            status_t const refusal = per_row(row, thread, traversals[thread]);
            if (succeeded(refusal)) return;
            status_t expected_status = status_t::success_k;
            atomic_ref<status_t>(first_refusal)
                .compare_exchange_strong(expected_status, refusal, std::memory_order_relaxed,
                                         std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(rows_count, each_row); failed(fanned)) return {{}, fanned};

        search_result_t merged;
        for (std::size_t const count : counts) merged.count += count;
        for (traversal_t const& traversal : traversals) {
            merged.visited_members += traversal.visited_members;
            merged.computed_distances += traversal.computed_distances;
            merged.restarts += traversal.restarts;
            merged.saturated_walks += traversal.saturated_walks;
        }
        return {merged, first_refusal};
    }

#pragma endregion Search

#pragma region Checkpoints

    /** The interchange token of one key - the monolith's key column dtype. */
    inline static constexpr dtype_t key_dtype_k = safegraphs_key_dtype<key_t>();

    /** The value column's dtype - the stored scalar's own token, so a packed word's suffix
     *  carries the packing and the column's shape counts logical dimensions. */
    inline static constexpr dtype_t value_column_dtype_k = safegraphs_value_dtype<scalar_t>();

    /** Whether the index can write @p config: placed, reserved, tokenized, and legal. */
    status_t saveable_(save_config_t const& config) const noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        return config.validate();
    }

    /** Whether the index can take a checkpoint: placed, and holding nothing yet. */
    status_t loadable_() const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        return size() ? status_t::config_mismatch_k : status_t::success_k;
    }

    /** Storage values per stored row - the copy width of every column. */
    std::size_t row_values_() const noexcept {
        return divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** The custom fields every monolith carries: the metric, the stored scalar by its own name -
     *  what a packed @c u8 column cannot say - and the caller's provenance. */
    status_t custom_fields_(growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields,
                            save_config_t const& config) const noexcept {
        if (failed(fields.reserve_exactly(2 + config.provenance.size()))) return status_t::bad_alloc_k;
        fields.push_back_reserved({"metric", nk_kernel_name(kernel_kind_)});
        fields.push_back_reserved({"quantization", nk_dtype_name(vector_dtype_)});
        for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
        return status_t::success_k;
    }

    /** The monolith header for @p nodes rows under @p config 's alignment. */
    expected<safegraphs_header_t> monolith_header_(std::size_t nodes, save_config_t const& config,
                                                   std::span<safegraphs_custom_field_t const> fields) const noexcept {
        safegraphs_regular_shape_t const shape {nodes,        key_dtype_k,    value_column_dtype_k,
                                                dimensions(), dtype_t::u32_k, connectivity()};
        return safegraphs_regular_header(shape, config.alignment, fields);
    }

    /** One slab of neighbor IDs from @p arena - the scratch a save chunk rebases through; empty
     *  when the arena is short. */
    nk::matrix_span<dense_node_id_t> scratch_of_(typename executor_t::arena_t& arena,
                                                 std::size_t scratch_rows) const noexcept {
        std::size_t const connectivity = this->connectivity();
        std::byte* bytes = arena.allocate(scratch_rows * connectivity * sizeof(dense_node_id_t),
                                          alignof(dense_node_id_t));
        if (!bytes) return {};
        return {reinterpret_cast<dense_node_id_t*>(bytes), scratch_rows, connectivity};
    }

    /** The shard placed on @p memory_domain - a thread's home - or the first shard when none sits
     *  there. */
    std::size_t home_shard_(fu_memory_domain_id_t memory_domain) const noexcept {
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (shards_[shard].index->vectors_allocator_.memory_domain == memory_domain) return shard;
        return 0;
    }

    /**
     *  @brief Fans every plan's chunks out over the executor's threads: each thread drains its home
     *      shard first, then helps the rest, one atomic cursor per shard being the whole schedule.
     *
     *  @p per_step lands one step on the thread that pulled it, with that thread's scratch -
     *  @p scratch_rows of neighbor IDs, none when zero - and answers a status; the first failure
     *  wins.
     *
     *  @p cursors is the caller's, one entry per plan, so an audit reuses one across its levels
     *  rather than allocating per level.
     */
    template <typename plan_type_, typename per_step_type_>
    status_t drain_steps_(growing_buffer<plan_type_, forkunion_allocator_t>& plans, std::span<std::size_t> cursors,
                          executor_t& executor, std::size_t scratch_rows, per_step_type_& per_step) noexcept {
        std::size_t const shards = plans.size();
        if (cursors.size() != shards) return status_t::config_mismatch_k;
        status_t outcome = status_t::success_k;
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            nk::matrix_span<dense_node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, status_t::bad_alloc_k, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = home_shard_(executor.memory_domain(thread));
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (home + visited) % shards;
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                auto const steps = plans[shard].steps();
                for (std::size_t position = cursor.fetch_add(1, std::memory_order_relaxed); position < steps.size();
                     position = cursor.fetch_add(1, std::memory_order_relaxed)) {
                    if (failed(atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) return;
                    status_t const landed = per_step(shard, steps[position], thread, scratch);
                    if (succeeded(landed)) continue;
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, landed, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (status_t const fanned = executor.for_threads(drain); failed(fanned)) return fanned;
        return atomic_ref<status_t>(outcome).load(std::memory_order_relaxed);
    }

    /** Fans @p plan 's steps out over the executor's threads: each thread drains its home shard's
     *  own run first, then helps the rest, one atomic cursor per shard being the whole schedule. A
     *  save lands each step with the thread's scratch of neighbor IDs; a load reads straight out of
     *  its part. The first failure wins and stops the rest. */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan, executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        auto const steps = plan.steps();
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step,
                                           nk::matrix_span<dense_node_id_t> scratch) { planned.save(step, scratch); };
        std::size_t const scratch_rows = saving_k
                                             ? power_of_two<std::size_t>(index_t::chunk_schedule_t::slab_rows_log2_k)
                                             : 0;

        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (scratch_rows)
            if (status_t const sized = executor.reserve(scratch_rows * connectivity() * sizeof(dense_node_id_t));
                failed(sized))
                return sized;
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        status_t outcome = status_t::success_k;
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            nk::matrix_span<dense_node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, status_t::bad_alloc_k, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = home_shard_(executor.memory_domain(thread));
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (home + visited) % shards;
                iota_view<std::size_t> const own = plan.steps_of(shard);
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                for (std::size_t taken = cursor.fetch_add(1, std::memory_order_relaxed); taken < own.size();
                     taken = cursor.fetch_add(1, std::memory_order_relaxed)) {
                    if (failed(atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) return;
                    status_t landed = status_t::success_k;
                    if constexpr (saving_k) landed = plan.save(steps[own.first + taken], scratch);
                    else landed = plan.load(steps[own.first + taken]);
                    if (succeeded(landed)) continue;
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, landed, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (status_t const fanned = executor.for_threads(drain); failed(fanned)) return fanned;
        return atomic_ref<status_t>(outcome).load(std::memory_order_relaxed);
    }

    /** @brief Copies @p path into @p buffer, so a plan outlives the caller's view of it. */
    static status_t keep_path_(growing_buffer<char, forkunion_allocator_t>& buffer, std::string_view path) noexcept {
        if (failed(buffer.reserve_exactly(path.size()))) return status_t::bad_alloc_k;
        for (char const character : path) buffer.push_back_reserved(character);
        return status_t::success_k;
    }

    /** Opens shard @p shard 's own save over the @p nodes rows it writes at @p first_row of
     *  @p target; the plan's own @c id_shift is where its keys land in @c finish. */
    status_t open_save_shard_(save_plan_t& plan, std::size_t shard, bytes_span_t target,
                              safegraphs_header_t const& header, std::size_t first_row, std::size_t nodes,
                              save_config_t const& config) noexcept {
        typename index_t::columns_t const columns {header.rows_of(target, header.value, first_row, nodes),
                                                   header.rows_of(target, header.neighbors, first_row, nodes)};
        auto [inner, status] = shards_[shard].index->plan_save(columns, config.synchronization,
                                                               static_cast<dense_node_id_t>(first_row));
        if (failed(status)) return status;
        plan.plans_.push_back_reserved(std::move(inner));
        return status_t::success_k;
    }

    /** @brief Frames @p monolith for one graph, every shard writing at its own base with its IDs
     *      rebased onto it, and opens one save plan per shard over the rows it owns. */
    status_t open_save_monolith_(save_plan_t& plan, bytes_span_t monolith, save_config_t const& config) noexcept {
        std::size_t const shards = shards_.size();
        plan.layout_ = checkpoint_layout_t::monolith_k;
        plan.monolith_ = monolith;
        if (status_t const named = custom_fields_(plan.fields_, config); failed(named)) return named;
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {allocator_};
        if (failed(plan.plans_.reserve_exactly(shards)) || failed(plan.headers_.reserve_exactly(1)) ||
            failed(shard_rows.reserve_exactly(shards)))
            return status_t::bad_alloc_k;

        // The rows are read once, so a shard that grows between here and its watermark is refused there.
        std::size_t rows = 0;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_rows.push_back_reserved(shards_[shard].index->size());
            rows += shard_rows[shard];
        }
        auto const [header, status] = monolith_header_(rows, config, to_span(plan.fields_));
        if (failed(status)) return status;
        if (monolith.size() < header.monolith_bytes()) return status_t::overflow_k;
        plan.headers_.push_back_reserved(header);
        for (std::size_t shard = 0, base = 0; shard != shards; base += shard_rows[shard], ++shard)
            if (status_t const opened = open_save_shard_(plan, shard, monolith, plan.headers_[0], base,
                                                         shard_rows[shard], config);
                failed(opened))
                return opened;
        plan.owner_ = this;
        return status_t::success_k;
    }

    /** Frames, sizes and maps one part per shard beside the plan's path, and opens one save plan
     *  over each - a part's IDs stay shard-local, so its rows travel verbatim. */
    status_t open_save_parts_(save_plan_t& plan, save_config_t const& config) noexcept {
        std::size_t const shards = shards_.size();
        plan.layout_ = checkpoint_layout_t::distributed_k;
        if (status_t const named = custom_fields_(plan.fields_, config); failed(named)) return named;
        if (failed(plan.plans_.reserve_exactly(shards)) || failed(plan.headers_.reserve_exactly(shards)) ||
            failed(plan.mappings_.reserve_exactly(shards)))
            return status_t::bad_alloc_k;

        std::span<safegraphs_custom_field_t const> const fields = to_span(plan.fields_);
        std::string_view const stem = safegraphs_stem(to_string_view(plan.path_));
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const nodes = shards_[shard].index->size();
            auto const [header, frame_status] = monolith_header_(nodes, config, fields);
            if (failed(frame_status)) return frame_status;
            auto const [file_path, path_status] = safegraphs_part_path(part_path, stem, shard, shards);
            if (failed(path_status)) return path_status;
            auto [mapping, map_status] = mapped_file_t::write(file_path, header.monolith_bytes());
            if (failed(map_status)) return map_status;
            plan.headers_.push_back_reserved(header);
            plan.mappings_.push_back_reserved(std::move(mapping));
            if (status_t const opened = open_save_shard_(plan, shard, plan.mappings_[shard].span(),
                                                         plan.headers_[shard], 0, nodes, config);
                failed(opened))
                return opened;
        }
        plan.owner_ = this;
        return status_t::success_k;
    }

    /** Reads one part's header and tokens out of the bytes it carries. */
    status_t describe_part_(part_t& part) const noexcept {
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
            if (field.key == "quantization") part.quantization = safegraphs_name_t {field.value};
        };
        auto const [header, status] = safegraphs_read_header(part.monolith, allocator_, per_custom_field);
        if (failed(status)) return status;
        part.header = header;
        part.nodes = header.nodes();
        return status_t::success_k;
    }

    /** Lists @p monolith as the checkpoint's only part. */
    status_t catalog_monolith_(bytes_view_t monolith,
                               growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        part_t part;
        part.monolith = monolith;
        if (status_t const described = describe_part_(part); failed(described)) return described;
        return parts.push_back(part);
    }

    /** Lists the parts at @p path: the magic makes a lone monolith, anything else a manifest
     *  whose every part is mapped and its header read now, so no entry is taken on trust. The
     *  mappings land in @p mappings, which must outlive the load. */
    status_t catalog_(std::string_view path, growing_buffer<mapped_file_t, forkunion_allocator_t>& mappings,
                      growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        auto [mapping, map_status] = mapped_file_t::read(path);
        if (failed(map_status)) return map_status;
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
            part.path = {offset, paths_arena.size()};
            if (failed(parts.push_back(part))) kept = status_t::bad_alloc_k;
        };
        auto const [summary, summary_status] = safegraphs_read_manifest(path, allocator_, per_part);
        if (failed(summary_status)) return summary_status;
        if (failed(kept)) return kept;
        if (parts.empty()) return status_t::malformed_k;
        if (failed(mappings.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (part_t& part : parts) {
            auto const [sibling, sibling_status] = safegraphs_sibling_path(
                part_path, path, std::string_view {paths_arena.data() + part.path.first, part.path.size()});
            if (failed(sibling_status)) return sibling_status;
            auto [part_mapping, part_status] = mapped_file_t::read(sibling);
            if (failed(part_status)) return part_status;
            part.monolith = part_mapping.view();
            if (status_t const described = describe_part_(part); failed(described)) return described;
            mappings.push_back_reserved(std::move(part_mapping));
        }
        return status_t::success_k;
    }

    /** Whether @p part can land here: three regular columns of the index's own dtypes, the tokens
     *  it names agreeing with this instantiation, the geometry @p first froze, and every column
     *  describing the same rows. */
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

    /**
     *  @brief Vets the plan's parts, assigns them onto the shards per @p config 's placement,
     *      reserves from it, and opens one load plan per shard over the rows it ends up with.
     *
     *  A shard takes its parts in order and appends them at one base, so the flat steps are the
     *  intersections of each part's rows with the storage chunks they land in - a part that spans
     *  two chunks answers two steps, and two parts sharing a chunk answer one each.
     */
    status_t open_load_(load_plan_t& plan, load_config_t const& config) noexcept {
        std::size_t const shards = shards_.size();
        if (plan.parts_.empty()) return status_t::malformed_k;
        safegraphs_header_t const& first = plan.parts_[0].header;
        for (part_t const& part : plan.parts_)
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
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {allocator_};
        if (failed(part_rows.reserve_exactly(plan.parts_.size())) || failed(shard_rows.resize(shards)))
            return status_t::bad_alloc_k;
        for (part_t const& part : plan.parts_) part_rows.push_back_reserved(part.nodes);
        status_t kept = status_t::success_k;
        assign_parts(config.placement, to_span(part_rows), shards, [&](checkpoint_assignment_t assignment) noexcept {
            if (failed(plan.landings_.push_back(assignment))) kept = status_t::bad_alloc_k;
            shard_rows[assignment.shard] += assignment.rows.size();
        });
        if (failed(kept)) return kept;
        std::size_t widest = 0;
        for (std::size_t const rows : shard_rows) widest = max_of(widest, rows);

        // Growth ahead of the rows the parts carry - the config's or the widest shard times the shard count - and
        // the frozen geometry, when there is one, must match the parts'.
        std::size_t const wanted = max_of(config.capacity, widest * shards);
        if (capacity() && (dimensions != this->dimensions() || connectivity != this->connectivity()))
            return status_t::config_mismatch_k;
        if (!capacity() && !wanted) {
            plan.owner_ = this;
            return status_t::success_k;
        }
        if (status_t const reserved = reserve({wanted, connectivity, dimensions}).status; failed(reserved))
            return reserved;

        if (failed(plan.plans_.reserve_exactly(shards))) return status_t::bad_alloc_k;
        std::size_t const vectors_row_bytes = row_values_() * sizeof(scalar_t);
        std::size_t const neighbors_row_bytes = connectivity * sizeof(dense_node_id_t);
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [inner, status] = shards_[shard].index->plan_load(shard_rows[shard]);
            if (failed(status)) return status;
            plan.plans_.push_back_reserved(std::move(inner));
            typename index_t::load_plan_t const& opened = plan.plans_[shard];
            std::size_t first_row = opened.base;
            for (checkpoint_assignment_t const& assignment : plan.landings_) {
                if (assignment.shard != shard) continue;
                std::size_t const bound_row = first_row + assignment.rows.size();
                for (auto const chunk : opened.steps()) {
                    std::size_t const low = max_of<std::size_t>(chunk.index_rows.first, first_row);
                    std::size_t const high = min_of<std::size_t>(chunk.index_rows.bound, bound_row);
                    if (low >= high) continue;
                    typename load_plan_t::step_t step;
                    step.shard = shard;
                    step.part = assignment.part;
                    step.source_range = {static_cast<dense_node_id_t>(assignment.rows.first),
                                         static_cast<dense_node_id_t>(assignment.rows.bound)};
                    step.inner.position = chunk.position;
                    step.inner.index_rows = {static_cast<dense_node_id_t>(low), static_cast<dense_node_id_t>(high)};
                    step.inner.source_rows = {static_cast<dense_node_id_t>(assignment.rows.first + low - first_row),
                                              static_cast<dense_node_id_t>(assignment.rows.first + high - first_row)};
                    step.inner.vectors_bytes = (high - low) * vectors_row_bytes;
                    step.inner.neighbors_bytes = (high - low) * neighbors_row_bytes;
                    if (failed(plan.steps_.push_back(step))) return status_t::bad_alloc_k;
                }
                first_row = bound_row;
            }
        }
        plan.owner_ = this;
        return status_t::success_k;
    }

#pragma endregion Checkpoints

    /** The owned shards, graph and keys together. A shard's key column is sized at @ref reserve
     *  to what the shard holds, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;

    /** Every key's entries. Duplicates coexist as separate slots, costing one slot per row rather
     *  than per key. */
    locators_map_t locators_of_key_;

    /** The host allocator behind the shard table, the locator map and every checkpoint catalog;
     *  the shards carry their own. */
    forkunion_allocator_t allocator_;

    /** Rows routed so far - the next row's shard and the next batch's rotation base. @c mutable
     *  since the pre-P3323 @c std::atomic_ref binds no const word, not even to load it. */
    mutable std::size_t routed_rows_ {0};
};

/**
 *  @brief External reader's cursor over the published rows of one @c dense_ram_indexes: keys
 *      and vectors in placement order, shard by shard.
 *
 *  Pure position - every read names the index it reads, which must be the one the cursor was
 *  opened on, so the cursor keeps no reference that could outlive it. Strictly read-only - the
 *  index is never written through, so readers never stall writers. A shard's watermark is
 *  snapshotted as the walk enters it, so a row once passed is never revisited, while a later
 *  shard may carry rows published after the walk began.
 */
template <typename indexes_type_>
struct dense_ram_indexes_cursor {

    using indexes_t = indexes_type_;
    using key_t = typename indexes_t::key_t;
    using scalar_t = typename indexes_t::scalar_t;
    using vectors_span_t = typename indexes_t::vectors_span_t;

    /** Enters the first shard of @p indexes, the one every later @ref read must name. */
    explicit dense_ram_indexes_cursor(indexes_t const& indexes) noexcept { enter_(indexes); }

    /** Copies up to @p keys.size() rows of @p indexes into @p keys - and their vectors in the stored scalar type
     *  into @p vectors, unless it holds no rows - advancing past them and reporting the rows written. An export
     *  wanting vectors must offer at least as many rows as @p keys, each wide enough for a stored row, or nothing
     *  is read. */
    std::size_t read(indexes_t const& indexes, std::span<key_t> keys, vectors_span_t vectors) noexcept {
        bool const with_vectors = vectors.extent(0) != 0;
        std::size_t const row_values = indexes.row_values_();
        if (with_vectors && (vectors.extent(0) < keys.size() || vectors.extent(1) < row_values)) return 0;
        std::size_t const row_bytes = row_values * sizeof(scalar_t);
        std::size_t written = 0;
        for (; written != keys.size() && shard_ < indexes.shards_.size(); ++written) {
            auto const& shard = indexes.shards_[shard_];
            keys[written] = shard.keys[node_];
            if (with_vectors)
                std::memcpy(vectors.row(written).as_vector().byte_data(), shard.index->vector(node_).byte_data(),
                            row_bytes);
            if (++node_ != visible_) continue;
            ++shard_;
            enter_(indexes);
        }
        return written;
    }

  private:
    /** @brief Snapshots the current shard's watermark, skipping shards with nothing published. */
    void enter_(indexes_t const& indexes) noexcept {
        for (node_ = 0; shard_ < indexes.shards_.size(); ++shard_)
            if ((visible_ = static_cast<dense_node_id_t>(indexes.shards_[shard_].index->size())) != 0) return;
    }

    /** @brief The shard being read, positional in placement order; the shard count past the end. */
    std::size_t shard_ = 0;
    /** @brief The current row inside that shard. */
    dense_node_id_t node_ = 0;
    /** @brief That shard's watermark snapshot: rows strictly below it are readable. */
    dense_node_id_t visible_ = 0;
};

} // namespace usearch
} // namespace unum
