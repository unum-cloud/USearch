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
#include <atomic>      // `std::atomic` - the routing tally, the chunk cursors & fan-out outcomes
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the callers' key batches & merged rankings
#include <string_view> // `std::string_view` - the checkpoint paths
#include <type_traits> // `std::is_same_v` - detecting the compiled-out default predicate
#include <utility>     // `std::move` - handing a mapping to the buffer that outlives the load

#include <forkunion.h> // `fu_memory_domain_id_t` - the placement argument of `emplace`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the shard table & key columns
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`, `forkunion_executor_t`
#include "usearch/utilities/ram_tables.hpp"        // `growing_multimap` - the locator map
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
template <                                                                    //
    typename key_type_ = uuid_t,                                              //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                     //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                      //
    nk_capability_t capability_ = nk_cap_icelake_k,                           //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename key_hash_type_ = std::hash<key_type_>                            //
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
    using keys_column_t = growing_buffer<key_t, forkunion_allocator_t>;
    /** @brief Key to every entry stored under it; insert-only, wait-free for readers. */
    using locators_map_t = growing_multimap<key_t, key_locator_t, key_hash_type_, //
                                            forkunion_allocator_t>;

    /** @brief The atomic reference owning the plain synchronization words - the shards' and this
     *      index's own admission word alike. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** @brief The shard type - geometry parameters in the shard's own order, so the two can
     *      never disagree. */
    using index_t = dense_ram_index<vector_dtype_, kernel_kind_, capability_, wait_type_, atomic_ref_template_, //
                                    forkunion_allocator_t,                                                      //
                                    forkunion_allocator_t,                                                      //
                                    arena_allocator<byte_t, forkunion_allocator_t>>;

    /** @brief One shard: the graph, and the keys of its rows. Placement is not repeated - the
     *      index's own allocator carries it. */
    struct shard_t {
        /** @brief The graph; its capacity counter is bounded by the key column, so its claims are
         *      the only admission a row needs. An index holds atomics and never moves. */
        std::unique_ptr<index_t> index;
        /** @brief The key of each of this shard's rows, indexed by node ID. */
        keys_column_t keys;
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
    using refine_config_t = typename index_t::refine_config_t;
    using refine_result_t = typename index_t::refine_result_t;
    using health_result_t = typename index_t::health_result_t;
    using heal_result_t = typename index_t::heal_result_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;
    static_assert(index_family<index_t> && index_claim<shard_capacity_claim_t>);

    /** @brief The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    /** @brief The distance kernel this instantiation walks with. */
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    /** @brief The compute-capability mask the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

#pragma region Placement

    /** @brief Every index names the host allocator its shard table, locator map and checkpoint
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
        std::size_t const routed = atomic_ref<std::size_t>(routed_rows_).load(std::memory_order_relaxed);
        atomic_ref<std::size_t>(routed_rows_)
            .store(atomic_ref<std::size_t>(other.routed_rows_).load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        atomic_ref<std::size_t>(other.routed_rows_).store(routed, std::memory_order_relaxed);
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
        keys_column_t keys {forkunion_allocator_t(shard->vectors_allocator_.memory_domain)};
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
        atomic_ref<std::size_t>(routed_rows_).store(0, std::memory_order_relaxed);
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
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

    /** @brief Rows the shards can host - the key columns' total, sized once at @ref reserve. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.keys.size();
        return total;
    }

    /** @brief Bytes the index occupies - every shard's storage, its key column, and the shared
     *      locator map; traversal scratch is the executor's. */
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
     *  columns and the locator map both demand a mutation-free phase to grow, and a search
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

        // The locator map first: its ceiling is what makes recording a locator infallible, and a
        // refusal here leaves nothing reserved, so the call stays retryable.
        if (status_t const indexed = locators_of_key_.reserve(per_shard.capacity * shards_.size()); failed(indexed))
            return {0, indexed};
        // A shard rounds up to whole chunks; its key column is sized to what it really holds.
        for (shard_t& one : shards_) {
            auto const [shard_capacity, reserved] = one.index->reserve(per_shard);
            if (failed(reserved)) return {0, reserved};
            if (failed(one.keys.resize(shard_capacity))) return {0, status_t::bad_alloc_k};
        }
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

        // The claim is all a row needs: a shard's counter is bounded by its key column, so a granted
        // row always has a key slot. The routing tally only names the first shard to try; a full
        // one passes the row on to the next, and any other refusal stops the probe.
        std::size_t const shards = shards_.size();
        std::size_t const routing_base = atomic_ref<std::size_t>(routed_rows_).fetch_add(1, std::memory_order_relaxed);
        shard_capacity_claim_t claim;
        std::size_t probe = 0;
        for (; probe != shards; ++probe) {
            auto [attempt, granted] = shards_[(routing_base + probe) % shards].index->claim(1);
            if (succeeded(granted)) {
                claim = std::move(attempt);
                break;
            }
            if (granted != status_t::overflow_k) return {result, granted};
        }
        if (probe == shards) return {result, status_t::overflow_k};
        std::size_t const shard = (routing_base + probe) % shards;

        shard_t& one = shards_[shard];
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
        // The locator map's ceiling covers the reservation, so recording cannot fail.
        [[maybe_unused]] status_t const recorded = locators_of_key_.insert(key, key_locator_t {shard, assigned});

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
        auto [claim, granted] = this->claim(new_vectors.extent(0));
        if (failed(granted)) return {result, granted};
        return insert(std::move(claim), keys, new_vectors, executor, config);
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
     *  Rotating by rows routed keeps a stream of small batches from drifting onto shard zero.
     *  Each shard's counter is bounded by its key column, so a batch never lands more rows on a shard
     *  than its key column holds, and a refusal releases whatever was already granted.
     */
    expected<capacity_claim_t> claim(std::size_t rows) noexcept {
        std::size_t const shards = shards_.size();
        capacity_claim_t claimed {allocator_};
        if (!shards) return {{}, status_t::uninitialized_k};
        if (!rows) return {std::move(claimed), status_t::success_k};
        if (failed(claimed.claims_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        claimed.rotation_ = atomic_ref<std::size_t>(routed_rows_).fetch_add(rows, std::memory_order_relaxed) % shards;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [one, granted] = shards_[shard].index->claim(share_of_(shard, rows, claimed.rotation_));
            if (failed(granted)) return {{}, granted};
            claimed.claims_.push_back_reserved(std::move(one));
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

        // One caller per executor, and the arenas grow to the widest geometry ever asked of them -
        // no caller guesses a byte count, the frozen geometry already knows it.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return {result, claimed};
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage(config)); failed(sized))
            return {result, sized};

        std::size_t const rotation = claim.rotation_;
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, config); failed(ready))
            return {result, ready};

        // The key lands before publication; the locator map's ceiling covers every claimed row.
        auto insert_row = [&](std::size_t task, std::size_t thread, std::size_t) noexcept {
            std::size_t const shard = (rotation + task) % shards;
            shard_t& one = shards_[shard];
            dense_node_id_t assigned = 0;
            one.index->insert(shard_capacity_claim_t {}, new_vectors(nk::range {task, task + 1}, nk::slice), {},
                              traversals[thread], config, [&](dense_node_id_t assigned_id) noexcept {
                                  assigned = assigned_id;
                                  one.keys[assigned_id] = keys[task];
                              });
            [[maybe_unused]] status_t const recorded = locators_of_key_.insert(keys[task],
                                                                               key_locator_t {shard, assigned});
        };
        // ForkUnion runs every task or none, so a refused fan-out releases the claims whole.
        if (status_t const fanned = executor.for_tasks(rows, insert_row); failed(fanned)) return {result, fanned};

        // Every row dispensed an id, so the claim is spent rather than released.
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

    /** @brief One shard's share of a refinement sweep: its published ids cut into steps a fan-out
     *      pulls, the same shape the checkpoint plans take. */
    struct refine_plan_t {

        /** @brief One unit of the sweep's fan-out. */
        struct step_t {
            /** @brief The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;
            /** @brief The ids this step re-selects. */
            iota_view<dense_node_id_t> nodes;
        };

        /** @brief The swept shard; null on a default plan, which holds no steps. */
        index_t* index_ = nullptr;

        /** @brief Ids one step covers - the first chunk's rows, so a step is one growth unit. */
        inline static constexpr std::size_t step_nodes_k = power_of_two<std::size_t>(index_t::first_chunk_rows_log2_k);

        /** @brief Step @p position of the plan; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * step_nodes_k;
            std::size_t const bound = min_of(first + step_nodes_k, index_->size());
            step.position = position;
            step.nodes = {static_cast<dense_node_id_t>(first), static_cast<dense_node_id_t>(bound)};
            return step;
        }

        /** @brief The plan's steps, positionally. */
        indexed_range<refine_plan_t> steps() const noexcept {
            return {this, index_ ? divide_round_up(index_->size(), step_nodes_k) : 0u};
        }
    };

    /**
     *  @brief Re-selects the neighbors of every published node across the shards - an insertion's
     *      wiring, replayed over stored nodes.
     *
     *  Every thread pulls steps from its home shard first and steals from the rest in the tail,
     *  so a sweep costs one pass over the graph however the shards are filled. Re-selection
     *  claims no identity and moves no watermark, so searches run throughout, a stopped sweep
     *  leaves the graph correct and merely less improved, and the next call simply sweeps again -
     *  re-selecting a settled node changes nothing, so nothing needs to remember where one
     *  stopped. Insertions may not overlap.
     *
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity"
     *  @param[in] on_visited Fires per step with the nodes visited so far and the population; a
     *      @c cancel_k answer stops the sweep at that step's boundary, reported as @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};

        // A re-selection walks like any other query, so it sizes like one.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return {{}, claimed};
        insert_config_t const walk_config {config.expansion_factor};
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage(walk_config));
            failed(sized))
            return {{}, sized};
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, walk_config); failed(ready))
            return {{}, ready};

        growing_buffer<refine_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (failed(plans.resize(shards)) || failed(cursors.resize(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) plans[shard].index_ = shards_[shard].index.get();

        // One tally per thread keeps the sweep's counters off every shared line.
        growing_buffer<refine_result_t, forkunion_allocator_t> tallies {allocator_};
        if (failed(tallies.resize(executor.size()))) return {{}, status_t::bad_alloc_k};

        std::size_t const published = size();
        std::atomic<std::size_t> swept {0};
        auto sweep_step = [&](std::size_t shard, typename refine_plan_t::step_t const& step, std::size_t thread,
                              nk::matrix_span<dense_node_id_t>) noexcept -> status_t {
            auto const [part, refined] = shards_[shard].index->refine(step.nodes, traversals[thread], config);
            if (failed(refined)) return refined;
            tallies[thread].visited_nodes += part.visited_nodes;
            tallies[thread].changed_nodes += part.changed_nodes;
            std::size_t const done = swept.fetch_add(part.visited_nodes, std::memory_order_relaxed) +
                                     part.visited_nodes;
            return on_visited(done, published) == continuation_t::cancel_k ? status_t::cancelled_k
                                                                           : status_t::success_k;
        };
        status_t const drained = drain_steps_(plans, to_writable_span(cursors), executor, 0, sweep_step);

        refine_result_t result;
        for (refine_result_t const& tally : tallies) {
            result.visited_nodes += tally.visited_nodes;
            result.changed_nodes += tally.changed_nodes;
        }
        for (traversal_t const& traversal : traversals) {
            result.visited_members += traversal.visited_members;
            result.computed_distances += traversal.computed_distances;
            result.computed_distances_in_refines += traversal.computed_distances_in_refines;
            result.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
            result.restarts += traversal.restarts;
            result.saturated_walks += traversal.saturated_walks;
        }
        return {result, drained};
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
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return {{}, claimed};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (status_t const ready = reserve_audit_(traversals, cursors, executor); failed(ready)) return {{}, ready};

        growing_buffer<typename index_t::health_plan_t, forkunion_allocator_t> plans {allocator_};
        if (failed(plans.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [plan, planned] = shards_[shard].index->plan_health(traversals[0]);
            if (failed(planned)) return {{}, planned};
            plans.push_back_reserved(std::move(plan));
        }
        if (status_t const audited = sweep_reachability_(plans, to_writable_span(cursors), executor); failed(audited))
            return {{}, audited};

        health_result_t result;
        for (typename index_t::health_plan_t const& plan : plans) {
            expected<health_result_t> const audited = plan.finish();
            if (failed(audited.status)) return {{}, audited.status};
            combine_health_(result, audited.value);
        }
        return {result, status_t::success_k};
    }

    /**
     *  @brief Re-links whatever no search can reach: the audit's islands are re-selected, the graph
     *      is re-swept, and whatever the diversity judgment still drops is reconnected into a
     *      reached pick - round after round while the island population shrinks, so a graph no
     *      round can improve ends the pass rather than spinning. Insertions may not overlap.
     *
     *  @param[inout] executor The threads and their scratch.
     */
    expected<heal_result_t> heal(executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return {{}, claimed};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (status_t const ready = reserve_audit_(traversals, cursors, executor); failed(ready)) return {{}, ready};

        growing_buffer<typename index_t::heal_plan_t, forkunion_allocator_t> plans {allocator_};
        if (failed(plans.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [plan, planned] = shards_[shard].index->plan_heal(traversals[0]);
            if (failed(planned)) return {{}, planned};
            plans.push_back_reserved(std::move(plan));
        }

        auto reselect_step = [&](std::size_t shard, auto const& step, std::size_t thread,
                                 nk::matrix_span<dense_node_id_t>) noexcept {
            return plans[shard].reselect_islands(step, traversals[thread]).status;
        };
        auto reconnect_step = [&](std::size_t shard, auto const& step, std::size_t thread,
                                  nk::matrix_span<dense_node_id_t>) noexcept {
            return plans[shard].reconnect_islands(step, traversals[thread]), status_t::success_k;
        };
        auto islands_left = [&]() noexcept {
            std::size_t left = 0;
            for (typename index_t::heal_plan_t const& plan : plans) left += plan.islands_left();
            return left;
        };

        std::span<std::size_t> const cursors_span = to_writable_span(cursors);
        if (status_t const audited = sweep_reachability_(plans, cursors_span, executor); failed(audited))
            return {{}, audited};
        for (std::size_t islands_before = islands_left(); islands_before;) {
            std::memset(cursors_span.data(), 0, cursors_span.size_bytes());
            if (status_t const reselected = drain_steps_(plans, cursors_span, executor, 0, reselect_step);
                failed(reselected))
                return {{}, reselected};
            if (status_t const audited = sweep_reachability_(plans, cursors_span, executor); failed(audited))
                return {{}, audited};
            if (!islands_left()) break;
            std::memset(cursors_span.data(), 0, cursors_span.size_bytes());
            if (status_t const forced = drain_steps_(plans, cursors_span, executor, 0, reconnect_step); failed(forced))
                return {{}, forced};
            if (status_t const audited = sweep_reachability_(plans, cursors_span, executor); failed(audited))
                return {{}, audited};
            // A round that frees no island frees none on the next either - the picks it needs are
            // themselves unreached - so the pass ends here and reports what still stands.
            std::size_t const islands_after = islands_left();
            if (islands_after >= islands_before) break;
            islands_before = islands_after;
        }

        heal_result_t result;
        for (typename index_t::heal_plan_t const& plan : plans) {
            expected<heal_result_t> const healed = plan.finish();
            if (failed(healed.status)) return {{}, healed.status};
            combine_health_(result.health, healed.value.health);
            result.islands_found += healed.value.islands_found;
            result.islands_reconnected += healed.value.islands_reconnected;
            result.visited_members += healed.value.visited_members;
            result.computed_distances += healed.value.computed_distances;
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
            auto const found = shards_[shard].index->search(query_vector, traversal, config);
            if (!found) {
                status = found.status;
                break;
            }
            merged.visited_members += found.value.visited_members;
            merged.computed_distances += found.value.computed_distances;
            merged.restarts += found.value.restarts;
            merged.saturated_walks += found.value.saturated_walks;
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
     *  @param[in] predicate Optional per-key admission filter; the default admits everything
     *      and compiles the filtered machinery out. Shards test node ids, so each one's keys
     *      column translates before the predicate sees a candidate.
     *
     *  A thread that fails stops its own query and leaves the rest to finish; the first failure is
     *  what the merged result reports, and every other query's outputs stand.
     */
    template <typename predicate_type_ = dummy_predicate_t>
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     std::size_t wanted, std::span<key_t> keys, std::span<distance_t> distances,
                                     std::span<std::size_t> counts, predicate_type_&& predicate = {}) noexcept {
        search_result_t merged;
        std::size_t const queries_count = queries.extent(0);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count || keys.size() < queries_count * wanted ||
            distances.size() < queries_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!queries_count) return {merged, status_t::success_k};

        // One caller per executor, and the arenas grow to the widest geometry ever asked of them.
        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return {{}, claimed};
        if (status_t const sized = executor.reserve(
                shards_[0].index->traversal_memory_usage(insert_config_t {config.expansion_factor}));
            failed(sized))
            return {{}, sized};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, insert_config_t {config.expansion_factor});
            failed(ready))
            return {{}, ready};

        std::atomic<status_t> first_refusal {status_t::success_k};
        auto scatter_query = [&](std::size_t query, std::size_t thread, std::size_t) noexcept {
            traversal_t& traversal = traversals[thread];
            std::span<key_t> const keys_row {keys.data() + query * wanted, wanted};
            std::span<distance_t> const distances_row {distances.data() + query * wanted, wanted};
            std::size_t found_count = 0;
            for (shard_t const& one : shards_) {
                expected<search_result_t> found {};
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
            return {{}, fanned};

        for (traversal_t const& traversal : traversals) {
            merged.visited_members += traversal.visited_members;
            merged.computed_distances += traversal.computed_distances;
            merged.restarts += traversal.restarts;
            merged.saturated_walks += traversal.saturated_walks;
        }
        for (std::size_t const count : counts) merged.count += count;
        return {merged, first_refusal.load(std::memory_order_relaxed)};
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
                if (status_t const wrote = copy_bytes(header.rows_of(target, header.key, base, nodes),
                                                      bytes_of(owner_->shards_[shard].keys.data(), nodes));
                    failed(wrote))
                    return wrote;
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
                auto const [name, formatted] = safegraphs_part_path(entry_path, base, shard, shards);
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
     *      monolith per shard beside a manifest. Every part is framed, sized and mapped here, and
     *      every header lands in @c finish, so a failed or cancelled save reads as absent.
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
        auto const [bytes, measured] = storage_requirements(config);
        if (failed(measured)) return {{}, measured};
        auto [mapping, mapped] = mapped_file_t::write(path, bytes);
        if (failed(mapped)) return {{}, mapped};
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
                    if (first + rows > keys.size()) return status_t::config_mismatch_k;
                    if (status_t const read = copy_bytes(
                            bytes_of(keys.data() + first, rows),
                            part.header.rows_of(part.monolith, part.header.key, landing.rows.first, rows));
                        failed(read))
                        return read;
                    for (std::size_t row = 0; row != rows; ++row) [[maybe_unused]]
                        status_t const recorded = owner_->locators_of_key_.insert(
                            keys[first + row], key_locator_t {shard, static_cast<dense_node_id_t>(first + row)});
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
        auto [plan, planned] = plan_save(path, config);
        if (failed(planned)) return planned;
        if (status_t const drained = drain_steps_(plan, executor); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /** @brief Serializes the index into @p monolith as one graph - the one-shot over the bytes
     *      form of @ref plan_save. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, planned] = plan_save(monolith, config);
        if (failed(planned)) return planned;
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
            auto [plan, planned] = plan_load(path, config);
            if (failed(planned)) return planned; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        // Nothing is published by a failed or cancelled load: the shards go back to empty, once
        // the plan has died and released whatever it claimed.
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /** @brief Restores one monolith held in @p monolith - the one-shot over the bytes form of
     *      @ref plan_load. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t monolith, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, planned] = plan_load(monolith, config);
            if (failed(planned)) return planned; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
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

#pragma region Key Columns

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

#pragma endregion Key Columns

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
    /** @brief Shard @p shard's share of a @p rows batch under @p rotation. */
    std::size_t share_of_(std::size_t shard, std::size_t rows, std::size_t rotation) const noexcept {
        return rows_from_(rows, first_row_of_shard_(shard, shards_.size(), rotation), shards_.size());
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

    /** @brief Sizes what an audit runs on: one context per thread, one cursor per shard. The
     *      reservation comes first, since it rewinds every arena, and the plans opened after it
     *      keep their bitsets. */
    status_t reserve_audit_(growing_buffer<traversal_t, forkunion_allocator_t>& traversals,
                            growing_buffer<std::size_t, forkunion_allocator_t>& cursors,
                            executor_t& executor) noexcept {
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage()); failed(sized))
            return sized;
        if (status_t const ready = reserve_traversals_(traversals, executor, {}); failed(ready)) return ready;
        return cursors.resize(shards_.size());
    }

    /** @brief One reachability audit over every shard's plan: the bitsets restart, then every level
     *      sweeps every step and advances each plan once, until no level expands a node - what
     *      @ref health reports and what each round of @ref heal re-runs. */
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

    /** @brief Folds one shard's audit into the index's: populations add, and the depth is the
     *      deepest shard's, since a searcher waits for the last one. */
    static void combine_health_(health_result_t& into, health_result_t const& one) noexcept {
        into.reachable_nodes += one.reachable_nodes;
        into.sweeps = max_of(into.sweeps, one.sweeps);
    }

#pragma endregion Health

#pragma region Checkpoints

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

    /** @brief Storage values per stored row - the copy width of every column. */
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
     *      schedule. @p per_step lands one step on the thread that pulled it, with that thread's
     *      scratch - @p scratch_rows of neighbor ids, none when zero - and answers a status; the
     *      first failure wins.
     *
     *  @p cursors is the caller's, one entry per plan, so an audit reuses one across its levels
     *  rather than allocating per level.
     */
    template <typename plan_type_, typename per_step_type_>
    status_t drain_steps_(growing_buffer<plan_type_, forkunion_allocator_t>& plans, std::span<std::size_t> cursors,
                          executor_t& executor, std::size_t scratch_rows, per_step_type_& per_step) noexcept {
        std::size_t const shards = plans.size();
        if (cursors.size() != shards) return status_t::config_mismatch_k;
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
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                auto const steps = plans[shard].steps();
                for (std::size_t position = cursor.fetch_add(1, std::memory_order_relaxed); position < steps.size();
                     position = cursor.fetch_add(1, std::memory_order_relaxed)) {
                    if (failed(outcome.load(std::memory_order_relaxed))) return;
                    status_t const landed = per_step(shard, steps[position], thread, scratch);
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

    /**
     *  @brief Fans @p plan 's steps out over the executor's threads: each thread drains its home
     *      shard's own run first, then helps the rest, one atomic cursor per shard being the whole
     *      schedule. A save lands each step with the thread's scratch of neighbor IDs; a load
     *      reads straight out of its part. The first failure wins and stops the rest.
     */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan, executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        auto const steps = plan.steps();
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step,
                                           nk::matrix_span<dense_node_id_t> scratch) { planned.save(step, scratch); };
        std::size_t const scratch_rows = saving_k ? power_of_two<std::size_t>(index_t::slab_rows_log2_k) : 0;

        auto [held, claimed] = executor.claim();
        if (failed(claimed)) return claimed;
        if (scratch_rows)
            if (status_t const sized = executor.reserve(scratch_rows * connectivity() * sizeof(dense_node_id_t));
                failed(sized))
                return sized;
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
                iota_view<std::size_t> const own = plan.steps_of(shard);
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                for (std::size_t taken = cursor.fetch_add(1, std::memory_order_relaxed); taken < own.size();
                     taken = cursor.fetch_add(1, std::memory_order_relaxed)) {
                    if (failed(outcome.load(std::memory_order_relaxed))) return;
                    status_t landed = status_t::success_k;
                    if constexpr (saving_k) landed = plan.save(steps[own.first + taken], scratch);
                    else landed = plan.load(steps[own.first + taken]);
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

    /** @brief Copies @p path into @p buffer, so a plan outlives the caller's view of it. */
    static status_t keep_path_(growing_buffer<char, forkunion_allocator_t>& buffer, std::string_view path) noexcept {
        if (failed(buffer.reserve_exactly(path.size()))) return status_t::bad_alloc_k;
        for (char const character : path) buffer.push_back_reserved(character);
        return status_t::success_k;
    }

    /** @brief Opens shard @p shard 's own save over the @p nodes rows it writes at @p first_row of
     *      @p target; the plan's own @c id_shift is where its keys land in @c finish. */
    status_t open_save_shard_(save_plan_t& plan, std::size_t shard, bytes_span_t target,
                              safegraphs_header_t const& header, std::size_t first_row, std::size_t nodes,
                              save_config_t const& config) noexcept {
        typename index_t::columns_t const columns {header.rows_of(target, header.value, first_row, nodes),
                                                   header.rows_of(target, header.neighbors, first_row, nodes)};
        auto [inner, planned] = shards_[shard].index->plan_save(columns, config.synchronization,
                                                                static_cast<dense_node_id_t>(first_row));
        if (failed(planned)) return planned;
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
        auto const [header, framed] = monolith_header_(rows, config, to_span(plan.fields_));
        if (failed(framed)) return framed;
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

    /** @brief Frames, sizes and maps one part per shard beside the plan's path, and opens one save plan
     *      over each - a part's IDs stay shard-local, so its rows travel verbatim. */
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
            auto const [header, framed] = monolith_header_(nodes, config, fields);
            if (failed(framed)) return framed;
            auto const [file_path, formatted] = safegraphs_part_path(part_path, stem, shard, shards);
            if (failed(formatted)) return formatted;
            auto [mapping, mapped] = mapped_file_t::write(file_path, header.monolith_bytes());
            if (failed(mapped)) return mapped;
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
            part.path = {offset, paths_arena.size()};
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
                part_path, path, std::string_view {paths_arena.data() + part.path.first, part.path.size()});
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

        // An unreserved index takes the config's rows or the widest shard times the shard count;
        // a reserved one must already fit every shard.
        std::size_t const wanted = max_of(config.capacity, widest * shards);
        if (!capacity()) {
            if (!wanted) {
                plan.owner_ = this;
                return status_t::success_k;
            }
            if (status_t const reserved = reserve({wanted, connectivity, dimensions}).status; failed(reserved))
                return reserved;
        }
        else {
            if (dimensions != this->dimensions() || connectivity != this->connectivity() ||
                capacity() < config.capacity)
                return status_t::config_mismatch_k;
            for (std::size_t shard = 0; shard != shards; ++shard)
                if (shard_rows[shard] > shards_[shard].keys.size()) return status_t::config_mismatch_k;
            // The load holds the mutation gate, which is the quiet the map's rehash asks for.
            if (status_t const indexed = locators_of_key_.reserve(wanted); failed(indexed)) return indexed;
        }

        if (failed(plan.plans_.reserve_exactly(shards))) return status_t::bad_alloc_k;
        std::size_t const vectors_row_bytes = row_values_() * sizeof(scalar_t);
        std::size_t const neighbors_row_bytes = connectivity * sizeof(dense_node_id_t);
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [inner, planned] = shards_[shard].index->plan_load(shard_rows[shard]);
            if (failed(planned)) return planned;
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

    /** @brief The owned shards, graph and keys together. A shard's key column is sized at
     *      @ref reserve and never grown, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;
    /** @brief Every key's entries. Duplicates coexist as separate slots, which is exactly what
     *      several values under one key means, and costs one slot per row rather than per key. */
    locators_map_t locators_of_key_;
    /** @brief The host allocator behind the shard table, the locator map and every checkpoint
     *      catalog; the shards carry their own. */
    forkunion_allocator_t allocator_;
    /** @brief Rows routed so far - the next row's shard and the next batch's rotation base.
     *      @c mutable since the pre-P3323 @c std::atomic_ref binds no const word, not even to load it. */
    mutable std::size_t routed_rows_ {0};
};

} // namespace usearch
} // namespace unum
