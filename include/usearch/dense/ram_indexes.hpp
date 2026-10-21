/**
 *  @file include/usearch/dense/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The sharded index over CPU-resident dense shards - one per memory domain - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The plural layer over @c ram_index: same-geometry shards, each owned and pinned to one
 *  ForkUnion memory domain, viewed as one index. Insertions rotate across the shards so they fill
 *  evenly; searches scatter over every shard and gather through @ref merge_search_results,
 *  resolving matches into the caller's keys, so shards with colliding ID spaces land in one
 *  ranking. Checkpoints ride @c shard_save_plan and @c shard_load_plan, every step fanned out over
 *  the executor's threads.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memset`, `std::memcpy`, `std::memmove`, `std::memcmp`

#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits` - the open threshold of a short exact ranking
#include <memory>      // `std::unique_ptr`
#include <mutex>       // `std::lock_guard` - the dispensing mutex
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::move`, `std::swap`

#include <forkunion.h> // `fu_memory_domain_id_t`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the plans' per-thread state
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`, `forkunion_executor_t`
#include "usearch/utilities/safegraphs_format.hpp" // `save_config_t`, `load_config_t`, `safegraphs_read_metadata`
#include "usearch/dense/types.hpp"
#include "usearch/dense/sharded_indexes.hpp"
#include "usearch/dense/shard_checkpoints.hpp"
#include "usearch/dense/ram_index.hpp"

namespace unum::usearch::dense {

template <typename indexes_type_>
struct ram_indexes_cursor;

/** Resolves a walked node ID to its key through one shard's key column, rebound per shard. */
template <typename keys_type_>
struct keyed_ids {

    /** The key column of the shard being walked. */
    keys_type_ const* keys = nullptr;

    auto operator()(node_id_t node_id) const noexcept { return (*keys)[node_id]; }
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
    typename quads_kit_type_ = bf16_serial_quads_kit_t,                       //
    typename tiles_kit_type_ = no_tiles_kit_t,                                //
    typename elementwise_kit_type_ = serial_elementwise_kit_t,                //
    typename rmscast_kit_type_ = no_rmscast_kit_t,                            //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,                 //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = dummy_cache_hints_t,                         //
    typename key_hash_type_ = std::hash<key_type_>                            //
    >
struct ram_indexes {
    friend struct ram_indexes_cursor<ram_indexes>;

    /** A reader's walk over every shard's published rows in placement order. */
    using cursor_t = ram_indexes_cursor<ram_indexes>;

    /** The shard type - geometry parameters in the shard's own order, so the two can never
     *  disagree. */
    using index_t =
        ram_index<quads_kit_type_, tiles_kit_type_, elementwise_kit_type_, rmscast_kit_type_, visits_kit_type_,
                  wait_type_, atomic_ref_template_, cache_hints_type_, forkunion_allocator_t, forkunion_allocator_t,
                  arena_allocator<byte_t, forkunion_allocator_t>>;

    /** The shards, the keys of their rows, the rotation that routes new rows, and the commit word a
     *  reader spanning the shards snapshots by. */
    using shards_t = shard_set<index_t, key_type_, key_hash_type_, atomic_ref_template_, wait_type_>;

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;
    using key_locator_t = typename shards_t::key_locator_t;
    using locators_map_t = typename shards_t::locators_map_t;
    using keys_column_t = typename shards_t::keys_column_t;
    using shard_t = typename shards_t::shard_t;

    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** The threads and their scratch every bulk verb runs on; it names no index, so one executor
     *  serves every instantiation. */
    using executor_t = forkunion_executor_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vector_norms_t = typename index_t::vector_norms_t;
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using refine_result_t = typename index_t::refine_result_t;
    using health_result_t = typename index_t::health_result_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;
    using walk_payload_t = typename index_t::walk_payload_t;
    using scan_payload_t = dense::scan_payload_t;
    static_assert(index_family<index_t> && index_claim<shard_capacity_claim_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = index_t::vector_dtype_k;

    /** The kits every shard scores and imports through, as @c src/ picked them. */
    using quads_kit_t = quads_kit_type_;
    using tiles_kit_t = tiles_kit_type_;
    using elementwise_kit_t = elementwise_kit_type_;

#pragma region Placement

    /** Every index names the distance kind its shards walk by, the host allocator its shard
     *  table, locator map and checkpoint catalogs sit on, and the line its shards prefetch by;
     *  nothing is placed by default. */
    ram_indexes() = delete;
    explicit ram_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator,
                         std::size_t line_bytes = default_alignment_k) noexcept
        : shards_(kernel, allocator), line_bytes_(line_bytes) {}
    ram_indexes(ram_indexes&& other) noexcept : shards_(std::move(other.shards_)), line_bytes_(other.line_bytes_) {}
    ram_indexes& operator=(ram_indexes&& other) noexcept { return swap(other), *this; }
    ram_indexes(ram_indexes const&) = delete;
    ram_indexes& operator=(ram_indexes const&) = delete;

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~ram_indexes() noexcept = default;

    /** Exchanges two indexes whole; a move implies the caller holds both exclusively. */
    void swap(ram_indexes& other) noexcept {
        shards_.swap(other.shards_);
        std::swap(line_bytes_, other.line_bytes_);
    }

    /** Appends one shard built here on @p memory_domain, for callers with none to hand over. */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(memory_domain);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** Appends one shard the caller already built, consuming it, with its key column on the
     *  shard's own memory domain; refuses any walking by another kernel or whose frozen geometry
     *  differs from the shards already here, and any already reserved. Refused once storage exists
     *  - @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard) return status_t::config_mismatch_k;
        forkunion_allocator_t const placed {shard->vectors_allocator_.memory_domain};
        return shards_.emplace(std::move(shard), placed);
    }

    /** Empties every shard in place, keeping its memory domain, geometry and capacity. Closes every
     *  shard's dispenser first, so a batch in flight refuses it with @c contended_k; the shards are
     *  rebuilt, so no insertion, search or other call may arrive meanwhile. */
    status_t clear() noexcept {
        if (status_t const closed = shards_.close(); failed(closed)) return closed;
        status_t const cleared = clear_();
        // A shard the rebuild failed to replace is still closed.
        shards_.reopen();
        return cleared;
    }

#pragma endregion Placement

#pragma region Measures and Reservation

    /** The distance kind every shard walks by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return shards_.kernel(); }

    /** Shards in the index, one per engaged memory domain. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }

    /** The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.connectivity(); }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.rows(); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** Bytes the index occupies - every shard's storage, its key column, and the shared locator
     *  map; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept { return sizeof(ram_indexes) + shards_.memory_usage(); }

    /**
     *  @brief Grows every shard ahead of @p config.capacity rows, split evenly over them - a hint,
     *      since the shards, their key columns and the locator map all grow on demand; the first
     *      call freezes the geometry.
     *
     *  Callable any time, alongside insertions and searches: nothing it grows ever moves.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return shards_.reserve(config); }

#pragma endregion Measures and Reservation

#pragma region Insertion

    /**
     *  @brief Inserts one row under @p key into whichever shard the rotation names next - the
     *      single-threaded sibling of the batched form, taking a context where it takes an executor.
     *
     *  Safe to call concurrently with itself, with the batched form, and with searches: the shard
     *  is chosen by one relaxed read-modify-write, and everything past the dispense is a step that
     *  cannot fail. A refine, an absorb or a load holding the shard's dispenser closed answers
     *  @c contended_k.
     *
     *  @param[in] key The caller's identity for the row; several rows may share one.
     *  @param[in] new_vector The row, in the shards' stored scalar type.
     *  @param[inout] traversal The calling thread's scratch & telemetry.
     *  @param[in] config The insertion configuration handed to the receiving shard.
     */
    expected<insert_result_t> insert(key_t const& key, vector_view_t new_vector, traversal_context_t& traversal,
                                     insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (shards_.empty()) return {result, status_t::uninitialized_k};

        // The routing tally only names the first shard to try; a full one passes the row on to the next,
        // and any other refusal stops the probe.
        std::size_t const shards = shards_.size();
        std::size_t const routing_base = shards_.route(1);
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
        auto [seats, seat_status] = shards_.locators().claim(1);
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
        node_id_t assigned = 0;
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        status_t const inserted = one.index->insert(std::move(claim), row, {}, traversal, config,
                                                    [&](node_id_t assigned_id) noexcept {
                                                        assigned = assigned_id;
                                                        one.keys[assigned_id] = key;
                                                    });
        if (failed(inserted)) return {result, inserted};
        shards_.locators().insert(seats, key, key_locator_t {shard, assigned});

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
     *  All-or-nothing across shards: every claim is taken before any row lands, past that point
     *  nothing can fail, and a search sees the whole batch or none of it. One executor drives one
     *  call at a time.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_rows The batch, one row per entry, in any scalar NumKong casts into the
     *      stored one.
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, rows_view_t new_rows, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (keys.size() != new_rows.rows) return {result, status_t::config_mismatch_k};
        // Rows land in slots as wide as the geometry, and a row refused past the dispense would
        // stall its shard.
        if (dimensions() && new_rows.columns != dimensions()) return {result, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(new_rows.rows, executor, config);
        if (failed(status)) return {result, status};
        carved_traversals_t traversals {arena_allocator<traversal_context_t, forkunion_allocator_t>(executor.arena(0))};
        if (status_t const carved = carve_traversals_(traversals, executor); failed(carved)) return {result, carved};
        if (status_t const landed = plan.insert(0, keys, new_rows, to_writable_span(traversals)); failed(landed))
            return {result, landed};
        return plan.finish();
    }

    /**
     *  @brief One batched insertion in flight: every fallible step taken, so its rows can only land,
     *      and @ref finish publishes them all at once.
     *
     *  Rows deal out round-robin over the shards from the rotation's shard on, each taking the next
     *  ID of its shard's range. The plan holds only reversible claims until its first step dispenses
     *  the ranges, so a plan dropped before then releases everything, and a caller spanning several
     *  indexes plans on all of them before stepping any, committing or refusing as a whole. Once
     *  stepped, its ranges hold every later turn on their shards until it finishes, so it must finish.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : claims_(allocator), ranges_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : claims_(std::move(other.claims_)), ranges_(std::move(other.ranges_)),
              executor_claim_(std::move(other.executor_claim_)), executor_(other.executor_), seats_(other.seats_),
              config_(other.config_), rotation_(other.rotation_), rows_(other.rows_), landed_rows_(other.landed_rows_),
              spent_(other.spent_), owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /** An unstepped plan gives back the executor and every shard's claim as its members die; a
         *  stepped one holds its shards' turns, so it must finish. */
        ~insert_plan_t() noexcept { assert((!owner_ || ranges_.empty()) && "A dispensed plan must finish"); }

        /**
         *  @brief Lands rows @p first_row onward of the plan, one per row of @p rows under its key
         *      in @p keys, fanned out over the executor's threads, each walking its context of the
         *      driver's @p traversals and imported into the stored scalar as it lands; fails only
         *      before its rows are dispensed.
         *
         *  The first step dispenses every shard's range under the dispensing mutex, so no two
         *  batches' ranges cross order between the shards. Rows outside the plan or of another
         *  width land nothing, and @ref finish refuses until every row landed. One caller at a
         *  time, like the executor.
         *
         *  @retval contended_k A refine, an absorb or a load holds a shard's dispenser closed;
         *      nothing was dispensed, so a later step may retry and a dropped plan releases its
         *      claims.
         *  @retval config_mismatch_k Rows outside the plan or of another width, or not one context
         *      per executor thread.
         */
        status_t insert(std::size_t first_row, nk::vector_view<key_t> keys, rows_view_t rows,
                        std::span<traversal_context_t> traversals) noexcept {
            std::size_t const count = rows.rows;
            if (!owner_) return status_t::uninitialized_k;
            if (!count) return status_t::success_k;
            if (keys.size() != count || first_row > rows_ || count > rows_ - first_row ||
                rows.columns != owner_->dimensions() || traversals.size() != executor_->size())
                return status_t::config_mismatch_k;
            if (status_t const sized = owner_->size_traversals_(traversals, config_, 0); failed(sized)) return sized;
            if (ranges_.empty()) {
                std::lock_guard<fu::spin_mutex_t> const dispensing {owner_->dispensing_};
                for (std::size_t shard = 0; shard != claims_.size(); ++shard) {
                    index_t& one = *owner_->shards_[shard].index;
                    auto const [range, dispense_status] = one.dispense_(std::move(claims_[shard]));
                    if (failed(dispense_status)) {
                        // Closers go in shard order too, so every earlier range is empty.
                        for ([[maybe_unused]] iota_view<node_id_t> const before : ranges_)
                            assert(before.empty() && "A refused batch dispensed rows on an earlier shard");
                        ranges_.clear();
                        return dispense_status;
                    }
                    ranges_.push_back_reserved(range);
                }
            }
            // The key lands before the walk, so a row reachable once published always reads one.
            auto land_row = [&](std::size_t task, std::size_t thread) noexcept {
                key_locator_t const locator = locator_of_(first_row + task);
                shard_t& one = owner_->shards_[locator.shard];
                one.keys[locator.node] = keys[task];
                one.index->wire_(locator.node, rows, task, traversals[thread], config_);
            };
            drain_in_order_(count, land_row);
            landed_rows_ += count;
            for (traversal_context_t const& traversal : traversals) {
                spent_.visited_members += traversal.visited_members;
                spent_.computed_distances += traversal.computed_distances;
                spent_.computed_distances_in_refines += traversal.computed_distances_in_refines;
                spent_.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
                spent_.restarts += traversal.restarts;
                spent_.saturated_walks += traversal.saturated_walks;
            }
            return status_t::success_k;
        }

        /**
         *  @brief Publishes every row at once and reports the batch's telemetry. A plan with rows
         *      still to land refuses with @c config_mismatch_k and stays whole; a success spends it.
         *
         *  Every touched shard's turn is awaited first; a batch spanning several shards then stores
         *  their watermarks inside a bracket of the commit sequence word, which readers retry around.
         *  The keys enter the locator map last, so a lookup never names a row a search cannot reach.
         */
        expected<insert_result_t> finish() noexcept {
            insert_result_t result = spent_;
            if (!owner_) return {result, status_t::uninitialized_k};
            result.new_size = owner_->size();
            if (landed_rows_ != rows_) return {result, status_t::config_mismatch_k};
            std::size_t touched_shards = 0;
            for (iota_view<node_id_t> const range : ranges_) touched_shards += !range.empty();
            bool const spans_shards = touched_shards > 1;

            // No later range on a shard can take its turn ahead of this one, so the waits hold no bracket open.
            for (std::size_t shard = 0; shard != ranges_.size(); ++shard)
                if (!ranges_[shard].empty()) owner_->shards_[shard].index->await_turn_(ranges_[shard]);
            if (spans_shards) owner_->shards_.begin_commit();
            for (std::size_t shard = 0; shard != ranges_.size(); ++shard)
                if (!ranges_[shard].empty()) owner_->shards_[shard].index->publish_(ranges_[shard]);
            if (spans_shards) owner_->shards_.end_commit();

            auto locate_row = [&](std::size_t row, std::size_t) noexcept {
                key_locator_t const locator = locator_of_(row);
                typename locators_map_t::seats_t seat {seats_.next + row, seats_.next + row + 1};
                owner_->shards_.locators().insert(seat, owner_->shards_[locator.shard].keys[locator.node], locator);
            };
            drain_in_order_(rows_, locate_row);

            result.new_size = owner_->size();
            owner_ = nullptr;
            return {result, status_t::success_k};
        }

      private:

        /** Where row @p row of the plan lands: its shard, and its ID inside that shard's range. */
        key_locator_t locator_of_(std::size_t row) const noexcept {
            std::size_t const shard = (rotation_ + row) % ranges_.size();
            return {shard, static_cast<node_id_t>(ranges_[shard].first + row / ranges_.size())};
        }

        /** Runs @p per_row over @p rows rows taken in ascending order by the executor's threads, so a
         *  row's walk finds the rows of its range dispensed ahead of it mostly published; a refused
         *  fan-out runs them all on the calling thread, since a dispensed row must still land. */
        template <typename per_row_type_>
        void drain_in_order_(std::size_t rows, per_row_type_& per_row) noexcept {
            if (!rows) return;
            // Every row bumps it, so it takes its own line, away from the locals every thread reads.
            alignas(default_alignment_k) std::size_t cursor = 0;
            auto drain = [&](std::size_t thread, std::size_t) noexcept {
                atomic_ref<std::size_t> const next {cursor};
                for (std::size_t row = next.fetch_add(1, std::memory_order_relaxed); row < rows;
                     row = next.fetch_add(1, std::memory_order_relaxed))
                    per_row(row, thread);
            };
            if (failed(executor_->for_threads(drain))) drain(0, 0);
        }

        /** Every shard's capacity claim in placement order, released by its own death until dispensed. */
        growing_buffer<shard_capacity_claim_t, forkunion_allocator_t> claims_;

        /** Every shard's range in placement order, empty where no row lands; none before the first step. */
        growing_buffer<iota_view<node_id_t>, forkunion_allocator_t> ranges_;

        /** The executor's one-caller hold, kept while the plan's threads may run. */
        typename executor_t::claim_t executor_claim_;

        /** The executor the rows fan out over. */
        executor_t* executor_ = nullptr;

        /** One locator seat per row, in row order. */
        typename locators_map_t::seats_t seats_ {};

        /** The per-shard insertion configuration, applied uniformly. */
        insert_config_t config_ {};

        /** The shard row zero lands on. */
        std::size_t rotation_ = 0;

        /** Rows the plan lands. */
        std::size_t rows_ = 0;

        /** Rows landed so far. */
        std::size_t landed_rows_ = 0;

        /** What every step's walks spent, which @ref finish reports. */
        insert_result_t spent_ {};

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once @c finish published. */
        ram_indexes* owner_ = nullptr;

        friend ram_indexes;
    };

    /**
     *  @brief Opens a batched insertion of @p rows: takes the executor, every shard's share of
     *      capacity and key column, and the locator seats, all of them reversible.
     *
     *  Every fallible step happens here but the sizing of the contexts each step is handed, so the
     *  plan's rows can only land, and nothing is dispensed until the plan's first step. The
     *  rotation comes from the rows routed so far, which keeps a stream of small batches from
     *  drifting onto shard zero.
     *
     *  @param[in] rows Rows the plan lands.
     *  @param[inout] executor The threads, held until the plan dies.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        insert_plan_t plan {shards_.allocator()};
        plan.executor_ = &executor;
        plan.config_ = config;
        plan.rows_ = rows;
        plan.owner_ = this;
        if (!rows) return {std::move(plan), status_t::success_k};

        // One caller per executor; the steps walk the driver's contexts, so nothing here rewinds.
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.executor_claim_ = std::move(executor_claim);

        // Each shard's key column is covered to what the shard hosts after its share, and a refusal
        // releases every share already granted as the claims die.
        if (failed(plan.claims_.reserve_exactly(shards)) || failed(plan.ranges_.reserve_exactly(shards)))
            return {{}, status_t::bad_alloc_k};
        plan.rotation_ = shards_.route(rows) % shards;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_t& one = shards_[shard];
            auto [shard_claim, status] = one.index->claim(
                shards_t::rows_from(rows, shards_t::first_row_of_shard(shard, shards, plan.rotation_), shards));
            if (failed(status)) return {{}, status};
            plan.claims_.push_back_reserved(std::move(shard_claim));
            if (status_t const covered = one.keys.cover(one.index->capacity()); failed(covered)) return {{}, covered};
        }
        auto [seats, seat_status] = shards_.locators().claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        return {std::move(plan), status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Refinement

    /**
     *  @brief One refine in flight over every shard, holding every dispenser closed meanwhile.
     *
     *  It owns its shards' passes and part starts, so nothing about it lives in the index, and it
     *  walks on the contexts each call hands it; @ref run continues it after a cancel, and
     *  @ref finish heals what a stopped pass leaves and reopens. Every shard's @c refine_plan_t
     *  drains over the threads phase by phase, each thread pulling steps from its home shard first
     *  and stealing from the rest in the tail, so a pass costs one sweep of the graph however the
     *  shards are filled. An absorb's plan first lands its donors, one after another, and only then
     *  opens the shards' passes over the parts they appended. A split's plan opens every shard's
     *  split at planning, and the phase boundary that publishes the parts moves the keys' locators
     *  into the receivers and empties this index.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb and @ref plan_split fill it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : shard_plans_(allocator), cursors_(allocator), part_bases_(allocator), part_offsets_(allocator),
              receiver_shards_(allocator), receiver_seats_(allocator), emptied_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : shard_plans_(std::move(other.shard_plans_)), cursors_(std::move(other.cursors_)),
              part_bases_(std::move(other.part_bases_)), part_offsets_(std::move(other.part_offsets_)),
              donors_(other.donors_), landed_donors_(other.landed_donors_), receivers_(other.receivers_),
              closed_receivers_(std::exchange(other.closed_receivers_, 0)),
              receiver_shards_(std::move(other.receiver_shards_)), receiver_seats_(std::move(other.receiver_seats_)),
              emptied_(std::move(other.emptied_)), config_(other.config_), visited_(other.visited_),
              population_(other.population_), owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** An unfinished plan heals nothing: it reopens, and the graph is as the last step left
         *  it. */
        ~refine_plan_t() noexcept { release_(); }

        /**
         *  @brief Runs the pending phases on @p executor until the pass ends or @p on_visited
         *      cancels; @c cancelled_k means a later call continues exactly where this one stopped.
         *
         *  Lands the donors still pending first, as @ref plan_absorb describes; a failing one stays
         *  pending, with every later one, for the next call. The executor is held only for the
         *  call.
         *
         *  @param[inout] executor The executor the plan was opened over.
         *  @param[inout] traversals The caller's context of every executor thread, which the walks
         *      run on; the call sizes them.
         *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
         *      population; a @c cancel_k answer stops the drain where its steps ran.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, std::span<traversal_context_t> traversals,
                     progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (executor.size() != traversals.size()) return status_t::config_mismatch_k;
            auto [executor_claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; landed_donors_ != donors_.size(); ++landed_donors_) {
                ram_indexes*& donor = donors_[landed_donors_];
                if (status_t const closed = donor->shards_.close(); failed(closed)) return closed;
                status_t const landed = owner_->absorb_(*donor, executor, *this);
                // Emptied donor shards are fresh and open, so only a failure leaves them closed.
                donor->shards_.reopen();
                if (failed(landed)) return landed;
                donor = nullptr;
            }
            if (shard_plans_.empty())
                if (status_t const opened = open_shards_(); failed(opened)) return opened;
            // Only once the donors landed does every shard have the geometry the walks size by.
            if (status_t const sized = size_walks_(traversals); failed(sized)) return sized;
            return drain_(executor, traversals, on_visited);
        }

        /** Stops any re-selection, audits and heals on the caller's @p traversals, one per
         *  executor thread, reopens the dispensers, and reports. A plan whose donors have not all
         *  landed heals nothing and reports nothing. */
        expected<refine_result_t> finish(executor_t& executor, std::span<traversal_context_t> traversals) noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (executor.size() != traversals.size()) return {{}, status_t::config_mismatch_k};
            auto [executor_claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};
            status_t healed = status_t::success_k;
            if (!shard_plans_.empty()) {
                // The steps a stopped re-selection left go unrun; a stopped heal phase continues.
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                    if (!shard_plans_[shard].reselecting()) continue;
                    shard_plans_[shard].stop();
                    static_cast<void>(shard_plans_[shard].advance());
                    cursors_[shard] = 0;
                }
                dummy_progress_t heal_alone;
                healed = size_walks_(traversals);
                if (succeeded(healed)) healed = drain_(executor, traversals, heal_alone);
            }
            refine_result_t const result = running();
            release_();
            return {result, healed};
        }

        /** The totals so far, summed over the shards' passes between calls: what @ref finish
         *  reports once it healed, and zeros while donors are pending. */
        refine_result_t running() const noexcept {
            refine_result_t result;
            for (typename index_t::refine_plan_t const& plan : shard_plans_) result += plan.finish().value;
            return result;
        }

      private:
        friend ram_indexes;

        /** Reopens this index and every receiver it closed; the plan is unbound after. */
        void release_() noexcept {
            if (owner_) std::exchange(owner_, nullptr)->shards_.reopen();
            for (; closed_receivers_; --closed_receivers_) receivers_[closed_receivers_ - 1]->shards_.reopen();
        }

        /** Sizes the caller's @p traversals for the passes' walks. */
        status_t size_walks_(std::span<traversal_context_t> traversals) noexcept {
            // A split walks its receivers, which keep the geometry once this index empties.
            ram_indexes& walked = receivers_.empty() ? *owner_ : *receivers_[0];
            return walked.size_traversals_(traversals, insert_config_t {config_.expansion_factor}, 0);
        }

        /** Opens every shard's pass over its parts, once no donor is pending. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != owner_->shards_.size(); ++shard) {
                std::span<node_id_t const> parts;
                if (!part_offsets_.empty())
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
                auto [plan, status] = owner_->shards_[shard].index->plan_refine({0, node_id_unknown_k}, config_, parts);
                if (failed(status)) return shard_plans_.clear(), status;
                shard_plans_.push_back_reserved(std::move(plan));
            }
            population_ = owner_->size();
            return status_t::success_k;
        }

        /** Drains every shard's pending phases from where the cursors stand, advancing each plan
         *  between them, until none follows or a step fails or cancels. */
        template <typename progress_type_>
        status_t drain_(executor_t& executor, std::span<traversal_context_t> traversals,
                        progress_type_& on_visited) noexcept {
            // Every step bumps it, so it takes its own line, away from the locals workers read.
            alignas(default_alignment_k) std::size_t visited = visited_;
            auto run_step = [&](std::size_t shard, typename index_t::refine_plan_t::step_t const& step,
                                std::size_t thread, nk::matrix_span<node_id_t>) noexcept -> status_t {
                typename index_t::refine_plan_t& plan = shard_plans_[shard];
                bool const reselecting = plan.reselecting();
                // A split's scatter lands each key beside its row, in the receiver's same shard.
                auto land_key = [&](node_id_t row, std::size_t receiver, node_id_t landed) noexcept {
                    receivers_[receiver]->shards_[shard].keys[landed] = owner_->shards_[shard].keys[row];
                };
                if (status_t const ran = plan.run(step, traversals[thread], land_key); failed(ran)) return ran;
                if (!reselecting) return status_t::success_k;
                std::size_t const done = atomic_ref<std::size_t>(visited).fetch_add(step.nodes.size(),
                                                                                    std::memory_order_relaxed) +
                                         step.nodes.size();
                return on_visited(done, population_) == continuation_t::cancel_k ? status_t::cancelled_k
                                                                                 : status_t::success_k;
            };
            std::span<std::size_t> const cursors = to_writable_span(cursors_);
            status_t drained = status_t::success_k;
            std::size_t advancing = 0;
            do {
                // Every claimed step ran, so a stopped drain's cursors say where the next starts.
                drained = owner_->drain_steps_(shard_plans_, cursors, executor, 0, run_step);
                if (failed(drained)) break;
                // The boundary publishing a split's parts shows a search every receiver shard's
                // rows or none, and only then do their keys resolve.
                bool const landing = !receivers_.empty() && shard_plans_[0].landing();
                if (landing)
                    for (ram_indexes* receiver : receivers_) receiver->shards_.begin_commit();
                advancing = 0;
                for (typename index_t::refine_plan_t& plan : shard_plans_) advancing += plan.advance();
                if (landing) {
                    for (ram_indexes* receiver : receivers_) receiver->shards_.end_commit();
                    if (!shard_plans_[0].landing()) owner_->publish_split_(*this);
                }
                std::memset(cursors.data(), 0, cursors.size_bytes());
            } while (advancing);
            visited_ = visited;
            return drained;
        }

        /** Every shard's own pass in placement order; none until the donors landed. */
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** One step cursor per shard into its current phase. */
        growing_buffer<std::size_t, forkunion_allocator_t> cursors_;

        /** The parts the landings appended, ascending per shard, @c node_id_unknown_k past the
         *  last. */
        growing_buffer<node_id_t, forkunion_allocator_t> part_bases_;

        /** Where each shard's run of @ref part_bases_ starts, and one past the last; empty with no
         *  donors. */
        growing_buffer<std::size_t, forkunion_allocator_t> part_offsets_;

        /** The caller's donor slots, each nulled once landed. */
        std::span<ram_indexes*> donors_;

        /** Donors landed so far. */
        std::size_t landed_donors_ = 0;

        /** The caller's receiver slots of a split, one part each; empty otherwise. */
        std::span<ram_indexes*> receivers_;

        /** Receivers whose dispensers the split closed, in receiver order. */
        std::size_t closed_receivers_ = 0;

        /** Per shard, the receivers' same shards in receiver order - what each shard's split
         *  scatters into. */
        growing_buffer<index_t*, forkunion_allocator_t> receiver_shards_;

        /** One run of locator seats per receiver, as many as its shards' part capacities sum to. */
        growing_buffer<typename locators_map_t::seats_t, forkunion_allocator_t> receiver_seats_;

        /** One fresh shard per shard of this index, built at planning, which it empties into. */
        growing_buffer<std::unique_ptr<index_t>, forkunion_allocator_t> emptied_;

        /** The passes' knobs. */
        refine_config_t config_ {};

        /** Nodes re-selected across every call so far. */
        std::size_t visited_ = 0;

        /** The rows published once the donors landed - the population @c on_visited reports. */
        std::size_t population_ = 0;

        /** The index refined, closed while the plan lives; null in an empty plan and once
         *  finished. */
        ram_indexes* owner_ = nullptr;
    };

    /** Opens a refine of every shard: closes the dispensers and opens the shards' passes, as
     *  @ref plan_absorb does with no donors. Re-selection claims no identity and moves no
     *  watermark, so searches run throughout. */
    expected<refine_plan_t> plan_refine(executor_t const& executor, refine_config_t config = {}) noexcept {
        return plan_absorb({}, executor, config);
    }

    /**
     *  @brief Re-selects the neighbors of every published node across the shards, then heals
     *      whatever that strands: @ref plan_refine, then @c run and @c finish.
     *
     *  Re-selection claims no identity and moves no watermark, so searches run throughout and a
     *  stopped pass leaves the graph correct. Every shard's dispenser stays closed for the pass, so
     *  a batch in flight refuses it with @c contended_k, and an insertion arriving meanwhile is
     *  refused.
     *
     *  @param[inout] executor The threads and their scratch.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
     *      population; a @c cancel_k answer skips the rest of the re-selection, still heals, and
     *      reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_refine(executor, config);
        if (failed(status)) return {{}, status};
        return drive_refine_(plan, executor, on_visited);
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every row of each into
     *      this index, one donor after another: donor shard @c s appends to shard @c s modulo the
     *      shard count with its edges rebased and its keys beside it, as a part of its own.
     *
     *  Each donor lands through one plan that does every fallible thing first: a load claiming all
     *  the rows each receiving shard takes, the key slots, the locator seats and the donor's fresh
     *  empty shards. Its chunk steps then drain over the threads, each copying the donor's chunks
     *  straight into the shard's own, keys beside them, and cannot fail; every shard's rows publish
     *  inside one commit bracket, so a search sees all of them or none, and their keys resolve
     *  after. The donor then empties into its fresh shards, which keep their placement but no
     *  capacity, its slot in @p donors nulls, and its shards join the receiving shards' parts in
     *  the plan. A shard empty beforehand keeps the first part landing in it as its resident part.
     *
     *  Every fallible step of every landing happens here or in the landing's own plan per donor,
     *  never in a step. Once the last donor landed, the plan's re-selection walks every shard
     *  across the parts the landings appended, round by round, so a group costs one refine rather
     *  than one per donor; a cancel pauses only that walk.
     *
     *  Searches of this index run throughout. The dispensers of this index stay closed while the
     *  plan lives and those of the donor landing while it lands, so a batch in flight on either
     *  refuses it with @c contended_k, and an insertion into this index arriving meanwhile is
     *  refused. The donors must stay quiescent besides: the copy reads their chunks and keys
     *  without locks, so nothing may search them or start an insertion into them. An index never
     *  reserved takes the first donor's geometry.
     *
     *  @param[inout] donors The indexes giving their rows up, each nulled once consumed; the slots
     *      must outlive the landings.
     *  @param[in] executor The executor every @c run and @c finish names.
     *  @param[in] config The refine's knobs; zeros mean "derive from connectivity".
     *  @retval config_mismatch_k A null donor, another metric or geometry, or this index itself.
     *  @retval contended_k A batch in flight on this index, or another refine, absorb or load.
     */
    expected<refine_plan_t> plan_absorb(std::span<ram_indexes*> donors, executor_t const& executor,
                                        refine_config_t config = {}) noexcept {
        std::size_t const shards = shards_.size(), threads = executor.size();
        if (!shards || !threads) return {{}, status_t::uninitialized_k};
        ram_indexes const* shaped = capacity() ? this : nullptr;
        for (ram_indexes const* donor : donors) {
            if (!donor || donor == this || donor->kernel() != kernel()) return {{}, status_t::config_mismatch_k};
            if (!donor->capacity()) continue;
            if (!shaped) shaped = donor;
            else if (donor->dimensions() != shaped->dimensions() || donor->connectivity() != shaped->connectivity())
                return {{}, status_t::config_mismatch_k};
        }
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.donors_ = donors;
        plan.config_ = config;
        if (failed(plan.shard_plans_.reserve_exactly(shards)) || failed(plan.cursors_.resize(shards)))
            return {{}, status_t::bad_alloc_k};

        // A shard's parts: its resident one, and one per donor shard landing in it.
        if (!donors.empty()) {
            if (failed(plan.part_offsets_.resize(shards + 1))) return {{}, status_t::bad_alloc_k};
            std::size_t parts = 0;
            for (std::size_t into = 0; into != shards; ++into) {
                plan.part_offsets_[into] = parts++;
                for (ram_indexes const* donor : donors)
                    for (std::size_t from = into; from < donor->shards_.size(); from += shards) ++parts;
            }
            plan.part_offsets_[shards] = parts;
            if (failed(plan.part_bases_.resize(parts))) return {{}, status_t::bad_alloc_k};
            for (node_id_t& base : plan.part_bases_) base = node_id_unknown_k;
        }

        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Moves every row of each of @p donors into this index and links them in: @ref plan_absorb,
     *  then @c run and @c finish. @p on_visited answers as in @ref refine, for the re-selection
     *  alone; a cancel still heals and reports @c cancelled_k. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> absorb(std::span<ram_indexes*> donors, executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_absorb(donors, executor, config);
        if (failed(status)) return {{}, status};
        return drive_refine_(plan, executor, on_visited);
    }

    /** Moves every row of @p donor into this index, as the plural @ref absorb does with one
     *  donor. */
    expected<refine_result_t> absorb(ram_indexes& donor, executor_t& executor, refine_config_t config = {}) noexcept {
        ram_indexes* donors[1] = {&donor};
        return absorb(std::span<ram_indexes*>(donors), executor, config);
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @ref plan_absorb:
     *      shard @c s of this index splits into shard @c s of every receiver, one graph-local part
     *      per receiver, with its keys beside its rows, and this index empties.
     *
     *  Every shard's split opens here, as @c index_t::plan_split describes, and so does every other
     *  fallible thing: every dispenser of this index and of the receivers closed, the receivers'
     *  key slots covered and locator seats claimed for full parts, and this index's fresh empty
     *  shards. The plan's first phases then partition and scatter every shard over the threads,
     *  each key landing beside its row, and the boundary publishing the parts shows a search every
     *  receiver shard's rows or none; their keys resolve after, and this index empties into its
     *  fresh shards, which keep their placement but no capacity. Every receiver shard then
     *  re-selects the rows that lost a link and heals; a cancel pauses only that walk.
     *
     *  Searches of the receivers run throughout, while nothing may search this index or start an
     *  insertion into it; an insertion into a receiver arriving meanwhile is refused.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two, all distinct, each
     *      with as many shards as this index; the slots must outlive the plan.
     *  @param[in] executor The executor every @c run and @c finish names.
     *  @param[in] config The refine's knobs; zeros mean "derive from connectivity".
     *  @param[in] seed Picks every shard's part roots: equal seeds over equal graphs split alike.
     *  @retval config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *      index itself, another metric, geometry or shard count.
     *  @retval contended_k A batch in flight on this index or a receiver, or another refine,
     *      absorb, split or load.
     */
    expected<refine_plan_t> plan_split(std::span<ram_indexes*> receivers, executor_t const& executor,
                                       refine_config_t config = {}, u64_t seed = 0) noexcept {
        std::size_t const shards = shards_.size(), threads = executor.size(), parts = receivers.size();
        if (!shards || !threads) return {{}, status_t::uninitialized_k};
        if (parts < 2) return {{}, status_t::config_mismatch_k};
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            ram_indexes const* into = receivers[receiver];
            if (!into || into == this || into->kernel() != kernel() || into->shards_.size() != shards || into->size())
                return {{}, status_t::config_mismatch_k};
            if (into->capacity() && capacity() &&
                (into->dimensions() != dimensions() || into->connectivity() != connectivity()))
                return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == into) return {{}, status_t::config_mismatch_k};
        }
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.receivers_ = receivers;
        plan.config_ = config;
        for (ram_indexes* into : receivers) {
            if (status_t const closed = into->shards_.close(); failed(closed)) return {{}, closed};
            ++plan.closed_receivers_;
        }
        if (failed(plan.shard_plans_.reserve_exactly(shards)) || failed(plan.cursors_.resize(shards)) ||
            failed(plan.receiver_shards_.reserve_exactly(shards * parts)) ||
            failed(plan.receiver_seats_.reserve_exactly(parts)) || failed(plan.emptied_.reserve_exactly(shards)))
            return {{}, status_t::bad_alloc_k};

        // A full part of every shard in every receiver: its key slots, and the locator seats.
        for (ram_indexes* into : receivers) {
            std::size_t seats = 0;
            for (std::size_t shard = 0; shard != shards; ++shard) {
                std::size_t const capacity = divide_round_up<std::size_t>(shards_[shard].index->size(), parts);
                if (status_t const covered = into->shards_[shard].keys.cover(capacity); failed(covered))
                    return {{}, covered};
                seats += capacity;
            }
            auto [claimed, seat_status] = into->shards_.locators().claim(seats);
            if (failed(seat_status)) return {{}, seat_status};
            plan.receiver_seats_.push_back_reserved(claimed);
        }
        for (shard_t const& part : shards_) {
            plan.emptied_.push_back_reserved(rebuild_(*part.index));
            if (!plan.emptied_.back()) return {{}, status_t::bad_alloc_k};
        }
        for (std::size_t shard = 0; shard != shards; ++shard)
            for (ram_indexes* into : receivers)
                plan.receiver_shards_.push_back_reserved(into->shards_[shard].index.get());
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::span<index_t*> const into {plan.receiver_shards_.data() + shard * parts, parts};
            auto [split, status] = shards_[shard].index->plan_split(into, config, seed + shard);
            if (failed(status)) return {{}, status};
            plan.shard_plans_.push_back_reserved(std::move(split));
        }
        plan.population_ = size();
        return {std::move(plan), status_t::success_k};
    }

    /** Moves every row of this index into one graph-local part per receiver and links each in:
     *  @ref plan_split, then @c run and @c finish. @p on_visited answers as in @ref refine, for the
     *  re-selection alone; a cancel still heals and reports @c cancelled_k. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> split(std::span<ram_indexes*> receivers, executor_t& executor,
                                    refine_config_t config = {}, u64_t seed = 0,
                                    progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_split(receivers, executor, config, seed);
        if (failed(status)) return {{}, status};
        return drive_refine_(plan, executor, on_visited);
    }

#pragma endregion Refinement

#pragma region Health

    /**
     *  @brief Reachability of every shard from as many landmarks as one search draws, swept level
     *      by level over the threads. Read-only, so concurrent insertions leave fresh nodes uncounted.
     *
     *  @param[inout] executor The threads the levels sweep on; each shard's bitsets come from the
     *      shard's own allocator.
     */
    expected<health_result_t> health(executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};

        growing_buffer<std::size_t, forkunion_allocator_t> cursors {shards_.allocator()};
        growing_buffer<typename index_t::health_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        if (failed(cursors.resize(shards)) || failed(plans.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (shard_t const& shard : shards_) {
            auto [plan, status] = shard.index->plan_health();
            if (failed(status)) return {{}, status};
            plans.push_back_reserved(std::move(plan));
        }
        if (status_t const audited = sweep_reachability_(plans, to_writable_span(cursors), executor); failed(audited))
            return {{}, audited};

        health_result_t result;
        for (typename index_t::health_plan_t const& plan : plans) {
            auto const [shard_health, status] = plan.finish();
            if (failed(status)) return {{}, status};
            result += shard_health;
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
     *  reported count and the ranking always agree. Every shard is walked below one consistent
     *  snapshot of the watermarks, so a batch is ranked whole or not at all.
     */
    expected<search_result_t> search(vector_view_t query_vector, traversal_context_t& traversal, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        search_result_t merged;
        std::span<node_id_t const> const watermarks = snapshot_(traversal);
        if (watermarks.size() != shards_.size()) return {merged, status_t::bad_alloc_k};
        status_t status = status_t::success_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            auto const [search_result, search_status] = shards_[shard].index->walk_(
                query_vector, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config, watermarks[shard]);
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
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts into the stored
     *      one; each thread imports its own.
     *  @param[inout] executor The threads; one query occupies one at a time.
     *  @param[inout] traversals One context per executor thread, carved by the caller; the call
     *      reseeds and sizes each, and reports only what it spent.
     *  @param[in] config The per-shard search configuration, applied uniformly; a
     *      @c traversal_t::scan_k scan scores every published row exactly instead, ignoring the
     *      breadth.
     *  @param[out] keys Merged rankings, one query per row, as many matches as its width.
     *  @param[out] distances The same rankings' distances as @p score_type_, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *
     *  A thread that fails stops its own query and leaves the rest to finish; the first failure is
     *  what the merged result reports, and every other query's outputs stand.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, executor_t& executor,
                                     std::span<traversal_context_t> traversals, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts) noexcept {
        return search_if(queries, executor, traversals, config, keys, distances, counts, dummy_predicate_t {});
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its key's indexer score against the heads of its query row.
     *
     *  @param[in] heads The indexer heads of every query row.
     *  @return @c config_mismatch_k unless the shards walk by dot.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     std::span<traversal_context_t> traversals, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, heads, executor, traversals, config, keys, distances, counts,
                            dummy_predicate_t {});
    }

    /**
     *  @brief The filtered sibling of the batched @ref search, sharing its whole contract.
     *
     *  @param[in] predicate The per-key admission filter; @c dummy_predicate_t admits everything
     *      and compiles the filtered machinery out. Shards test node IDs, so each one's keys column
     *      translates before the predicate sees a candidate. A scan asks it only about rows closer
     *      than the worst match their query already holds, so a selective filter is called far less
     *      often than rows are scored.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor,
                                        std::span<traversal_context_t> traversals, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        static_assert((!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> &&
                       !aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t>) ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (config.traversal == traversal_t::scan_k)
            return scan_rows_(queries, executor, traversals, config, keys, distances, counts, predicate);
        return search_rows_(queries, dummy_heads_t {}, executor, traversals, config, keys, distances, counts,
                            predicate);
    }

    /** The filtered sibling for one aggregating predicate per thread, each gathering a walk's
     *  @c walk_payload_t, so it serves walks alone and a scan answers @c config_mismatch_k. */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       walk_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor,
                                        std::span<traversal_context_t> traversals, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        if (config.traversal == traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, dummy_heads_t {}, executor, traversals, config, keys, distances, counts,
                            predicate);
    }

    /** The filtered sibling for one aggregating predicate per thread, each gathering a scan's
     *  @c scan_payload_t, so it serves scans alone and a walk answers @c config_mismatch_k. */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       scan_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor,
                                        std::span<traversal_context_t> traversals, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        if (config.traversal != traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return scan_rows_(queries, executor, traversals, config, keys, distances, counts, predicate);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row keeps
     *      the pairs its walks scored, keys resolved and dots cast into the products' scalar.
     *
     *  @param[in] left The left-hand rows, one walk per row, in any scalar NumKong casts into the
     *      stored one; each thread imports its own.
     *  @param[inout] executor The threads; one row occupies one at a time.
     *  @param[inout] traversals One context per executor thread, as the batched @ref search takes.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[out] keys The key of each kept pair, one left-hand row per row, as many pairs
     *      as its width.
     *  @param[out] products The dot of each kept pair as @p product_type_, parallel to @p keys.
     *  @param[out] counts Pairs scored per row across every shard, above the row width when
     *      truncated.
     *
     *  A full row keeps an arbitrary subset of its pairs. A thread that fails stops its own row and
     *  leaves the rest to finish; the first failure is what the merged result reports. Refused with
     *  @c config_mismatch_k unless the shards walk by dot.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor,
                                       std::span<traversal_context_t> traversals, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                       std::span<std::size_t> counts) noexcept {
        return multiply_if(left, executor, traversals, config, keys, products, counts, dummy_predicate_t {});
    }

    /**
     *  @brief The filtered sibling of @ref multiply, sharing its whole contract: only pairs whose
     *      key @p predicate admits are written and counted, while the walks route through the rest.
     *
     *  @param[in] predicate The per-key admission filter, asked about a scored quad of keys at a
     *      time.
     *  @return @c unsupported_k for a @c traversal_t::scan_k scan, since no exact product scan
     *      exists.
     */
    template <typename product_type_, typename predicate_type_>
    expected<search_result_t> multiply_if(rows_view_t left, executor_t& executor,
                                          std::span<traversal_context_t> traversals, search_config_t config,
                                          nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                          std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        using fold_t =
            multiply_fold<key_t, product_type_, keyed_ids<keys_column_t>, std::remove_reference_t<predicate_type_>&>;
        std::size_t const rows_count = left.rows;
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(rows_count, keys, products); failed(refusal))
            return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};

        auto multiply_row = [&](std::size_t row, std::size_t, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks) noexcept {
            vector_view_t const left_row = shards_[0].index->import_rows_(left, row, 1, traversal).row(0).as_vector();
            fold_t fold {to_span(keys.row(row).as_vector()), to_span(products.row(row).as_vector()), {}, predicate};
            status_t refusal = status_t::success_k;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                fold.resolve.keys = &shards_[shard].keys;
                expected<search_result_t> const walked = shards_[shard].index->walk_(
                    left_row, fold, dummy_rescoring_t {}, traversal, config, watermarks[shard]);
                if (!walked) {
                    refusal = walked.status;
                    break;
                }
            }
            counts[row] = fold.scored;
            return refusal;
        };
        return for_each_row_(rows_count, executor, traversals, config,
                             staged_rows_bytes(vector_dtype_k, 1, left.columns), multiply_row, counts);
    }

    /** NumKong's aggregation over the stored scalar's dots, what a MaxSim total sums in. */
    using total_t = typename scalar_t::reduce_moments_sum_t;

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key, and those maxima sum to that key's score, exact for every key the
     *      call lands; a walk never folds the keys its own dots bound below the kept ones.
     *
     *  @param[in] queries The token rows of every query, laid end to end.
     *  @param[in] query_offsets Where every query's rows start in @p queries, then where the last
     *      ends: one more than the queries.
     *  @param[inout] executor The threads; one query occupies one at a time, end to end.
     *  @param[inout] traversals One context per executor thread, as the batched @ref search takes;
     *      a query keeps its rows, maxima, ranking and candidates in its thread's staged queries.
     *  @param[in] config The per-shard walk configuration, applied uniformly; each row nominates as
     *      many keys as its expansion factor, and a @c traversal_t::scan_k scan nominates every
     *      key.
     *  @param[in] wanted Keys each query keeps, the best ones; zero keeps every key it scored, up
     *      to as many as the shards held when the call began under a scan.
     *  @param[out] scored Keys each query scored, before @p wanted cut any, one per query.
     *  @param[in] land Takes each kept key of a query in rank order, largest total first, on the
     *      thread that ranked it, as `land(thread, query, key, total) noexcept → status_t`; the
     *      first failure stops that query and is what the call answers.
     *
     *  Refused with @c shape_mismatch_k for a query without rows, and with @c config_mismatch_k
     *  for offsets past the rows, a @p scored of another length, or unless the shards walk by dot.
     */
    template <typename land_type_>
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     executor_t& executor, std::span<traversal_context_t> traversals,
                                     search_config_t config, std::size_t wanted, std::span<std::size_t> scored,
                                     land_type_&& land) noexcept {
        return maxsim_if(queries, query_offsets, executor, traversals, config, wanted, scored, land,
                         dummy_predicate_t {});
    }

    /**
     *  @brief The filtered sibling of @ref maxsim, sharing its whole contract: a key @p predicate
     *      rejects is never nominated, so it adds to no score.
     *
     *  @param[in] predicate The per-key admission filter: a walk asks it as the filtered
     *      @ref search_if asks it, a scan once per key and block of queries.
     */
    template <typename land_type_, typename predicate_type_>
    expected<search_result_t> maxsim_if(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                        executor_t& executor, std::span<traversal_context_t> traversals,
                                        search_config_t config, std::size_t wanted, std::span<std::size_t> scored,
                                        land_type_&& land, predicate_type_&& predicate) noexcept {
        static_assert((!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> &&
                       !aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t>) ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel() != nk_kernel_dot_k || query_offsets.empty()) return {{}, status_t::config_mismatch_k};
        std::size_t const queries_count = query_offsets.size() - 1;
        for (std::size_t query = 0; query != queries_count; ++query)
            if (query_offsets[query + 1] <= query_offsets[query]) return {{}, status_t::shape_mismatch_k};
        if (query_offsets.back() > queries.rows || scored.size() != queries_count)
            return {{}, status_t::config_mismatch_k};
        if (!queries_count) return {{}, status_t::success_k};

        // Every query walks its rows on one thread, so the driver sizes walks whatever nominates.
        traversal_t const nominating = config.traversal;
        config.traversal = traversal_t::walk_k;
        config.expansion_factor = static_cast<typename index_t::small_size_t>(
            shards_[0].index->non_zero_expansion_factor(config));
        // Without a bound, a query keeps every candidate it can have: its walks', or every row.
        std::size_t const visible = size();
        auto capacity_of = [&](std::size_t rows) noexcept {
            return wanted ? wanted : nominating == traversal_t::walk_k ? rows * config.expansion_factor : visible;
        };
        // A scan with a matrix engine reads the shards once per block of queries, a block a thread.
        std::size_t const threads = max_of<std::size_t>(traversals.size(), 1);
        std::size_t const block_queries = index_t::has_tiles_k && nominating == traversal_t::scan_k
                                              ? divide_round_up(queries_count, threads)
                                              : 1;
        std::size_t staged_bytes = 0;
        for (std::size_t first = 0; first < queries_count; first += block_queries) {
            std::size_t const taking = min_of(block_queries, queries_count - first);
            std::size_t const rows = query_offsets[first + taking] - query_offsets[first];
            staged_bytes = max_of(staged_bytes, maxsim_staged_bytes_(queries.columns, rows, taking, nominating, config,
                                                                     capacity_of(rows)));
        }
        auto maxsim_block = [&](std::size_t block, std::size_t thread, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks) noexcept {
            std::size_t const first = block * block_queries, taking = min_of(block_queries, queries_count - first);
            std::size_t const rows = query_offsets[first + taking] - query_offsets[first];
            return maxsim_block_(queries, query_offsets.subspan(first, taking + 1), first, nominating, config,
                                 capacity_of(rows), scored.subspan(first, taking), traversal, watermarks,
                                 thread_predicate(predicate, thread), thread, land);
        };
        std::size_t const candidates = max_of(index_t::round_capacity(connectivity()),
                                              index_t::chunk_schedule_t::rows_in(0));
        return for_each_row_(divide_round_up(queries_count, block_queries), executor, traversals, config, staged_bytes,
                             maxsim_block, {}, predicate, candidates);
    }

    /**
     *  @brief Softmax attention of query heads over every key their walks score across the shards: each token
     *      walks every shard in turn, folding into one running state per head, since the shards are disjoint
     *      and online softmax is order-free, and its rows normalize once after the last shard.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t; one walk per token and
     *      shard. Any scalar NumKong casts into the stored one; each thread imports its own
     *      token's heads.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's keys to their value rows, as wide as the keys and stored
     *      like them: `void(std::span<key_t const>, std::span<vector_view_t>)`, one row per key. Keys come from
     *      each shard's key column, so only scored nodes resolve; one without a row refuses with
     *      @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys.
     *      Packed @c f32 rows accumulate in place, any others in the thread's staged queries
     *      until their token exports them.
     *  @param[out] carry Every query row's softmax state, which merges this call with other
     *      branches, or empty to keep none.
     *  @param[inout] executor The threads; one token occupies one at a time.
     *  @param[inout] traversals One context per executor thread, as the batched @ref search takes.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @return The query rows attended in @c count, or @c config_mismatch_k unless the shards walk
     *      by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, std::span<traversal_context_t> traversals,
                                     search_config_t config = {}) noexcept {
        return attend_if(queries, head_count, value_lookup, score_scale, outputs, carry, executor, traversals, config,
                         dummy_predicate_t {});
    }

    /**
     *  @brief The filtered sibling of @ref attend, sharing its whole contract: a key @p predicate
     *      rejects still routes the walks, but folds into no head and resolves no value row.
     *
     *  @param[in] predicate The per-key admission filter, asked about a scored quad of keys at a
     *      time.
     *  @return @c unsupported_k for a @c traversal_t::scan_k scan, since no exact attention scan
     *      exists.
     */
    template <typename value_lookup_type_, typename predicate_type_>
    expected<search_result_t> attend_if(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                        f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                        executor_t& executor, std::span<traversal_context_t> traversals,
                                        search_config_t config, predicate_type_&& predicate) noexcept {
        using admits_t = node_predicate<std::remove_reference_t<predicate_type_>, keyed_ids<keys_column_t>>;
        using lookup_t = shard_value_lookup<value_lookup_type_>;
        std::size_t const rows_count = queries.rows, columns = queries.columns;
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.rows != rows_count || outputs.columns != columns || (!carry.empty() && carry.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};

        // Past the token's imported heads, their `f32` rows, then their softmax states.
        using slice_t = arena_allocator<byte_t>;
        bool const in_place = outputs.template holds<f32_t>() && outputs.stride == columns * sizeof(f32_t);
        std::size_t const heads_bytes = staged_rows_bytes(vector_dtype_k, head_count, columns);
        std::size_t const accumulators_bytes = slice_t::reserved_bytes(head_count * columns * sizeof(f32_t));
        auto attend_token = [&](std::size_t token, std::size_t, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks) noexcept {
            std::size_t const first = token * head_count;
            byte_t* const accumulators = traversal.staged_queries.data() + heads_bytes;
            byte_t* const states = accumulators + accumulators_bytes;
            status_t status = status_t::success_k;
            lookup_t shard_lookup {nullptr, value_lookup};
            admits_t admits {predicate, {}};
            attention_rescoring<quads_kit_t, elementwise_kit_t, lookup_t, admits_t&> const attention {
                shards_[0].index->import_rows_(queries, first, head_count, traversal),
                shard_lookup,
                score_scale,
                {reinterpret_cast<f32_t*>(in_place ? outputs.row(first) : accumulators), head_count, columns},
                carry.empty() ? std::span {reinterpret_cast<softmax_state_t*>(states), head_count}
                              : carry.subspan(first, head_count),
                status,
                admits};
            attention.clear();
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                shard_lookup.keys = admits.key_of.keys = &shards_[shard].keys;
                expected<search_result_t> const walked = shards_[shard].index->walk_(
                    attention.queries.row(0).as_vector(), dummy_fold_t {}, attention, traversal, config,
                    watermarks[shard]);
                if (!walked) return walked.status;
                if (failed(status)) return status;
            }
            attention.finish();
            if (!in_place)
                for (std::size_t head = 0; head != head_count; ++head)
                    export_row<elementwise_kit_t>(attention.outputs.row(head).data(), nk_f32_k, outputs, first + head);
            return status_t::success_k;
        };
        std::size_t const staged_bytes = heads_bytes + accumulators_bytes + head_count * sizeof(softmax_state_t);
        expected<search_result_t> attended = for_each_row_(rows_count / head_count, executor, traversals, config,
                                                           staged_bytes, attend_token, {});
        if (attended) attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** What a host checkpoint does beyond the shared plans: every shard opens its own plan on the
     *  calling thread, a save settles instead of closing the dispensers, and the keys live in the
     *  host columns alone, so nothing else lands or ends. */
    struct checkpoint_residency_t {
        static expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                       typename index_t::columns_t const& columns,
                                                                       synchronization_t synchronization,
                                                                       std::size_t first_row) noexcept {
            return shards[shard].index->plan_save(columns, synchronization, static_cast<node_id_t>(first_row));
        }
        static expected<typename index_t::load_plan_t> open_shard_load(shards_t& shards, std::size_t shard,
                                                                       std::size_t rows) noexcept {
            return shards[shard].index->plan_load(rows);
        }
        static status_t land_keys(shards_t&, std::size_t, iota_view<std::size_t>,
                                  typename index_t::load_plan_t const&) noexcept {
            return status_t::success_k;
        }
        static void end_save(shards_t&) noexcept {}
    };

    /** One save in flight across every shard: one shard plan each, the parts they write into,
     *  and the tail @c finish commits - the keys, the headers, and the manifest. */
    using save_plan_t = shard_save_plan<shards_t, checkpoint_residency_t>;

    /** One load in flight across every shard: the mapped parts, one shard plan each, and the flat
     *  steps their rows arrive in - one per part's share of one storage chunk. */
    using load_plan_t = shard_load_plan<shards_t, checkpoint_residency_t>;

    /** Bytes a @ref save under @p config writes: one monolith, or every part file and the manifest
     *  measured over a placeholder stem; zero for an invalid config or an unplaced index. The row
     *  count is read now, so a buffer sized here fits a save no row lands before. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        return shard_storage_requirements(shards_, config);
    }

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
     *  reports @c contended_k. A settled save holds every batch on all of its shards or on none:
     *  a batch dispenses on every shard before it publishes on any, so a shard still awaiting its
     *  publication refuses its plan, and every shard's plan opens here, before any @c finish
     *  re-checks.
     */
    expected<save_plan_t> plan_save(std::string_view path, save_config_t const& config) noexcept {
        return save_plan_t::open(shards_, {}, path, config);
    }

    /** Opens a save into @p monolith as one graph - the implementation the path form maps for; a
     *  manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size the bytes by
     *  @ref storage_requirements; fewer answer @c overflow_k. */
    expected<save_plan_t> plan_save(bytes_span_t monolith, save_config_t const& config) noexcept {
        return save_plan_t::open(shards_, {}, monolith, config);
    }

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
        return load_plan_t::open(shards_, {}, path, config);
    }

    /** Opens a load of one monolith held in @p monolith - the implementation the path form maps
     *  for. */
    expected<load_plan_t> plan_load(bytes_view_t monolith, load_config_t const& config) noexcept {
        return load_plan_t::open(shards_, {}, monolith, config);
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

    /** Entries stored under @p key; a concurrent insertion can only make it under-report. */
    std::size_t count(key_t const& key) const noexcept { return shards_.locators().count(key); }

    /** Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return shards_.locators().contains(key); }

    /**
     *  @brief Exports up to @p wanted rows stored under @p key into @p vectors in their scalar, and
     *      reports the @b true total under that key - a return above @p wanted means the copy
     *      truncated, so the caller resizes and asks again; zero means absent.
     *
     *  Rows at least as wide as the index take its columns and keep their tails. Safe against
     *  concurrent insertions, which only ever add. The executor rides in the signature unused,
     *  since one key's rows copy on the calling thread, so a caller drives a host index exactly as
     *  it drives a device one.
     */
    std::size_t get(key_t const& key, rows_span_t vectors, std::size_t wanted, executor_t&) const noexcept {
        rows_span_t const target {vectors.data, vectors.rows, dimensions(), vectors.stride, vectors.dtype};
        std::size_t total = 0;
        shards_.locators().find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < target.rows) {
                vector_view_t const stored = shards_[locator.shard].index->vector(locator.node);
                export_row<elementwise_kit_t>(stored.byte_data(), vector_dtype_k, target, total);
            }
            ++total;
        });
        return total;
    }

    /**
     *  @brief The batched @ref count: the entries stored under every key of @p keys, one key per
     *      task over the executor's threads.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[inout] executor The threads the lookups fan out over.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(keys.size(), [&](std::size_t position, std::size_t, std::size_t) noexcept {
            entries[position] = count(keys[position]);
        });
    }

    /** The batched @ref contains: one in @p entries where a key of @p keys holds any entry, zero
     *  where it holds none, one key per task over the executor's threads. */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(keys.size(), [&](std::size_t position, std::size_t, std::size_t) noexcept {
            entries[position] = contains(keys[position]) ? 1 : 0;
        });
    }

    /**
     *  @brief The batched @ref get: up to @p wanted rows of every key of @p keys, and every key's
     *      true total, one key per task over the executor's threads.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] vectors @b [keys·wanted,dimensions] in any scalar NumKong casts into, where row
     *      q · @p wanted + i is the i-th row of key q; refused with @c config_mismatch_k when the
     *      rows are another count or narrower than the index.
     *  @param[in] wanted Rows kept per key.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted
     *      when its rows truncated, zero when absent.
     *  @param[inout] executor The threads the exports fan out over.
     */
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) const noexcept {
        std::size_t const rows = keys.size() * wanted;
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (vectors.rows != rows || (rows && vectors.columns < dimensions())) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(keys.size(), [&](std::size_t position, std::size_t, std::size_t) noexcept {
            entries[position] = get(keys[position], vectors.slice(position * wanted, wanted), wanted, executor);
        });
    }

  private:

    /** The caller's key-taking value lookup, fed a walk's node IDs through one shard's keys. */
    template <typename value_lookup_type_>
    struct shard_value_lookup {

        /** The key column of the shard being walked, rebound per shard. */
        keys_column_t const* keys = nullptr;

        /** The caller's lookup, which takes keys. */
        value_lookup_type_& value_lookup;

        template <std::size_t extent_>
        void operator()(std::span<node_id_t const, extent_> nodes,
                        std::span<vector_view_t, extent_> rows) const noexcept {
            std::array<key_t, extent_> node_keys;
            for (std::size_t lane = 0; lane != extent_; ++lane) node_keys[lane] = (*keys)[nodes[lane]];
            value_lookup(std::span<key_t const, extent_>(node_keys), rows);
        }
    };

    /** Bytes one pack of up to @c pack_columns_k query rows takes, each pack on its own line. */
    std::size_t pack_stride_() const noexcept
        requires index_t::has_tiles_k
    {
        nk_size_t bytes = 0;
        if (tiles_kit_t::pack_size(tiles_kit_t::pack_columns_k, dimensions(), &bytes) != nk_success_k) return 0;
        return round_up_to_multiple<std::size_t>(bytes, default_alignment_k);
    }

    /** Stored rows one gather block holds: as many as share @p cache_bytes with one pack, never
     *  fewer than a kit step, and one step when the cache is unknown. */
    std::size_t block_rows_(std::size_t cache_bytes) const noexcept
        requires index_t::has_tiles_k
    {
        std::size_t const pack_stride = pack_stride_(), row_bytes = packed_bytes(vector_dtype_k, dimensions());
        return max_of((cache_bytes - min_of(cache_bytes, pack_stride)) / row_bytes, tiles_kit_t::stream_rows_k);
    }

    /** Takes @p count elements off the front of @p staged, which then starts on the next line. */
    template <typename element_type_>
    static std::span<element_type_> carve_(bytes_span_t& staged, std::size_t count) noexcept {
        std::size_t const bytes = count * sizeof(element_type_);
        std::span<element_type_> const carved = elements_of<element_type_>(staged.first(bytes));
        staged = staged.subspan(arena_allocator<byte_t>::reserved_bytes(bytes));
        return carved;
    }

    /**
     *  @brief Folds keys through the tiles kit for a block of query rows, every key's visible rows
     *      gathered into a block of stored rows, one segment per key.
     *
     *  The block folds against every pack of the query rows once full, and the @c rank_key each
     *  call passes takes each key with its maxima row, one maximum per query row, once all of its
     *  rows folded; a key with more rows than the block carries its maxima row into the next fold.
     *  Carved from @c staged as @ref maxsim_staged_bytes_ counts it.
     */
    struct tiled_fold_t {
        tiled_fold_t(ram_indexes const& owner, vectors_view_t query_rows, bytes_span_t staged, std::size_t cache_bytes,
                     std::span<node_id_t const> watermarks, traversal_context_t& traversal) noexcept
            : owner_(owner), watermarks_(watermarks), traversal_(traversal), columns_(query_rows.extent(0)),
              pack_bytes_(owner.pack_stride_()), block_capacity_(owner.block_rows_(cache_bytes)),
              row_bytes_(packed_bytes(vector_dtype_k, owner.dimensions())) {
            packs_ =
                carve_<byte_t>(staged, divide_round_up(columns_, tiles_kit_t::pack_columns_k) * pack_bytes_).data();
            block_ = carve_<byte_t>(staged, block_capacity_ * row_bytes_).data();
            ends_ = carve_<std::size_t>(staged, block_capacity_ + 1);
            keys_ = carve_<key_t>(staged, block_capacity_ + 1);
            maxima_ = {carve_<dot_result_t>(staged, (block_capacity_ + 1) * columns_).data(), block_capacity_ + 1,
                       columns_};
            for (std::size_t first = 0; first < columns_; first += tiles_kit_t::pack_columns_k) {
                std::size_t const width = min_of(tiles_kit_t::pack_columns_k, columns_ - first);
                vectors_view_t const pack_rows = query_rows(nk::range(first, first + width), nk::slice);
                [[maybe_unused]] nk_status_t const packing = tiles_kit_t::pack(
                    pack_rows.raw_values_data(), width, owner.dimensions(),
                    static_cast<std::size_t>(pack_rows.stride_bytes(0)),
                    packs_ + first / tiles_kit_t::pack_columns_k * pack_bytes_, 0, width, nullptr);
                assert(packing == nk_success_k && "A matrix engine packs any block");
            }
        }

        /** Gathers every visible row of @p key as its own segment, folding the block whenever it
         *  fills; a key without visible rows folds nothing and is never ranked. */
        template <typename rank_key_type_>
        void add(key_t const& key, rank_key_type_& rank_key) noexcept {
            keys_[segments_] = key;
            for (dot_result_t& maximum : to_span(maxima_.row(segments_).as_vector()))
                maximum = dot_result_t::finite_min();
            std::size_t folded = 0;
            owner_.shards_.locators().find_each(key, [&](key_locator_t locator) noexcept {
                if (locator.node >= watermarks_[locator.shard]) return;
                if (gathered_ == block_capacity_) carry_(rank_key);
                vector_view_t const row = owner_.shards_[locator.shard].index->vector(locator.node);
                std::memcpy(block_ + gathered_++ * row_bytes_, row.byte_data(), row_bytes_);
                ++folded;
            });
            ends_[segments_] = gathered_;
            if (folded) ++segments_;
        }

        /** Folds and ranks whatever the block still holds. */
        template <typename rank_key_type_>
        void finish(rank_key_type_& rank_key) noexcept {
            if (segments_) fold_(segments_, rank_key);
        }

      private:
        ram_indexes const& owner_;
        std::span<node_id_t const> watermarks_;
        traversal_context_t& traversal_;
        std::size_t columns_, pack_bytes_, block_capacity_, row_bytes_;
        byte_t* packs_ = nullptr;
        byte_t* block_ = nullptr;
        std::span<std::size_t> ends_;
        std::span<key_t> keys_;
        nk::matrix_span<dot_result_t> maxima_;
        std::size_t gathered_ = 0, segments_ = 0;

        /** Folds the block's rows into its first @p segments maxima rows against every pack, then
         *  ranks the closed segments. */
        template <typename rank_key_type_>
        void fold_(std::size_t segments, rank_key_type_& rank_key) noexcept {
            vectors_view_t const rows {reinterpret_cast<scalar_t const*>(block_), gathered_, owner_.dimensions()};
            for (std::size_t first = 0; first < columns_; first += tiles_kit_t::pack_columns_k) {
                std::size_t const width = min_of(tiles_kit_t::pack_columns_k, columns_ - first);
                fold_tiles<tiles_kit_t>(
                    rows, packs_ + first / tiles_kit_t::pack_columns_k * pack_bytes_, ends_.first(segments),
                    maxima_(nk::range(std::size_t {0}, segments), nk::range(first, first + width), nk::slice),
                    traversal_.step_dots.span());
            }
            traversal_.computed_distances += gathered_ * columns_;
            for (std::size_t segment = 0; segment != segments_; ++segment)
                rank_key(keys_[segment], to_span(maxima_.row(segment).as_vector()));
        }

        /** Folds a block that filled inside a key: the closed segments rank, and the open one moves
         *  to the front with its maxima to keep accumulating. */
        template <typename rank_key_type_>
        void carry_(rank_key_type_& rank_key) noexcept {
            ends_[segments_] = gathered_;
            fold_(segments_ + 1, rank_key);
            std::memmove(maxima_.row(0).data(), maxima_.row(segments_).data(), columns_ * sizeof(dot_result_t));
            keys_[0] = keys_[segments_];
            gathered_ = segments_ = 0;
        }
    };

    /** One key a MaxSim walk row kept, grouped by key to bound what the key can total. */
    struct alignas(2 * sizeof(key_t)) maxsim_hit_t {

        /** The key the row's walks ranked. */
        key_t key;

        /** The query row whose walks ranked it, counted within the block. */
        u32_t row;

        /** How far its dot rises above the row's worst kept one, in walk distances. */
        distance_t gain;
    };

    /** Larger totals first, ties by key bytes, so every thread count ranks alike. */
    static bool better_(total_t first_total, key_t const& first_key, total_t second_total,
                        key_t const& second_key) noexcept {
        if (first_total != second_total) return first_total > second_total;
        return std::memcmp(&first_key, &second_key, sizeof(key_t)) < 0;
    }

    /** Keeps @p key at @p total among @p best and its keys, @p kept of them so far: appended while
     *  there is room, else in place of the @p worst slot it beats, which a full ranking keeps
     *  current, so a rejected key costs one comparison. */
    static void rank_into_(std::span<maxsim_ranked<total_t>> best, std::span<key_t> best_keys, std::size_t& kept,
                           std::size_t& worst, key_t const& key, total_t total) noexcept {
        if (best.empty()) return;
        if (kept != best.size()) {
            best[kept] = {static_cast<u32_t>(kept), total};
            best_keys[kept++] = key;
            if (kept != best.size()) return;
        }
        else {
            if (!better_(total, key, best[worst].total, best_keys[best[worst].candidate])) return;
            best[worst].total = total;
            best_keys[best[worst].candidate] = key;
        }
        worst = 0;
        for (std::size_t slot = 1; slot != kept; ++slot)
            if (better_(best[worst].total, best_keys[best[worst].candidate], best[slot].total,
                        best_keys[best[slot].candidate]))
                worst = slot;
    }

    /** Hands each query's kept keys of @p best to @p land, largest first, the q-th query as query
     *  @p first_query + q of thread @p thread; answers the first refusal @p land gives. */
    template <typename land_type_>
    static status_t land_ranked_(std::span<maxsim_ranked<total_t>> best, std::span<key_t const> best_keys,
                                 std::span<std::size_t const> kept, std::size_t thread, std::size_t first_query,
                                 land_type_& land) noexcept {
        std::size_t const capacity = best.size() / kept.size();
        for (std::size_t query = 0; query != kept.size(); ++query) {
            std::span<maxsim_ranked<total_t>> const ranked = best.subspan(query * capacity, kept[query]);
            std::span<key_t const> const keys = best_keys.subspan(query * capacity, capacity);
            partial_sort(ranked, ranked.size(),
                         [&](maxsim_ranked<total_t> const& first, maxsim_ranked<total_t> const& second) noexcept {
                             return better_(first.total, keys[first.candidate], second.total, keys[second.candidate]);
                         });
            for (maxsim_ranked<total_t> const& entry : ranked)
                if (status_t const landed = land(thread, first_query + query, keys[entry.candidate], entry.total);
                    failed(landed))
                    return landed;
        }
        return status_t::success_k;
    }

    /** Bytes a MaxSim block of @p queries_count queries over @p rows rows of @p columns stages,
     *  each ranking up to @p capacity keys, laid out as @ref maxsim_block_ carves them, and from a
     *  tile's columns of rows on, the @ref tiled_fold_t past them. */
    std::size_t maxsim_staged_bytes_(std::size_t columns, std::size_t rows, std::size_t queries_count,
                                     traversal_t nominating, search_config_t config,
                                     std::size_t capacity) const noexcept {
        using slice_t = arena_allocator<byte_t>;
        std::size_t const width = nominating == traversal_t::walk_k ? config.expansion_factor : 0;
        std::size_t const bytes = staged_rows_bytes(vector_dtype_k, rows, columns) +
                                  slice_t::reserved_bytes(rows * sizeof(vector_norms_t)) +
                                  slice_t::reserved_bytes(rows * sizeof(dot_result_t)) +
                                  slice_t::reserved_bytes(queries_count * capacity * sizeof(maxsim_ranked<total_t>)) +
                                  slice_t::reserved_bytes(queries_count * capacity * sizeof(key_t)) +
                                  slice_t::reserved_bytes(queries_count * sizeof(std::size_t)) +
                                  slice_t::reserved_bytes(queries_count * sizeof(std::size_t)) +
                                  slice_t::reserved_bytes(width * sizeof(key_t)) +
                                  slice_t::reserved_bytes(width * sizeof(distance_t)) +
                                  slice_t::reserved_bytes(rows * width * sizeof(maxsim_hit_t)) +
                                  slice_t::reserved_bytes(rows * width * sizeof(maxsim_ranked<total_t>)) +
                                  slice_t::reserved_bytes(rows * sizeof(distance_t));
        if constexpr (index_t::has_tiles_k)
            if (rows >= tiles_kit_t::tile_columns_k) {
                std::size_t const block_rows = block_rows_(config.cache_bytes);
                return bytes +
                       slice_t::reserved_bytes(divide_round_up(rows, tiles_kit_t::pack_columns_k) * pack_stride_()) +
                       slice_t::reserved_bytes(block_rows * packed_bytes(vector_dtype_k, dimensions())) +
                       slice_t::reserved_bytes((block_rows + 1) * sizeof(std::size_t)) +
                       slice_t::reserved_bytes((block_rows + 1) * sizeof(key_t)) +
                       slice_t::reserved_bytes((block_rows + 1) * rows * sizeof(dot_result_t));
            }
        return bytes;
    }

    /**
     *  @brief Late interaction of the queries @p query_offsets bounds on @p traversal: every
     *      candidate folds each of its rows below @p watermarks into each query's ranking.
     *
     *  Each query hands its best @p capacity keys to @p land, largest first, the first query as
     *  query @p first_query of thread @p thread, and counts every key it ranked in @p scored. A
     *  walk serves one query, each of its rows ranking @p config's breadth of keys, and visits each
     *  key once in descending bound: its best walk dot per row, and the worst kept dot of every row
     *  that missed it, until a full ranking's worst total beats the next bound; under a
     *  @c traversal_t::scan_k scan every key whose first locator is a visible row is a candidate
     *  once for every query. @p predicate hears each candidate before it folds. A tile's columns of
     *  rows or more fold through the tiles kit, fewer a quad of rows at a time. Everything lives in
     *  the traversal's staged queries, laid out as @ref maxsim_staged_bytes_ counts it: the rows,
     *  their norms, one maximum per row, each query's ranking, its keys, its count and its worst
     *  slot, under walks one row's merged keys and distances, every row's hits, the keys' bounds
     *  and one gain per row, then the tiled fold.
     */
    template <typename predicate_type_, typename land_type_>
    status_t maxsim_block_(rows_view_t queries, std::span<std::size_t const> query_offsets, std::size_t first_query,
                           traversal_t nominating, search_config_t config, std::size_t capacity,
                           std::span<std::size_t> scored, traversal_context_t& traversal,
                           std::span<node_id_t const> watermarks, predicate_type_& predicate, std::size_t thread,
                           land_type_& land) noexcept {
        constexpr bool filtered_k = !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;
        std::size_t const queries_count = query_offsets.size() - 1, first_row = query_offsets.front();
        std::size_t const rows = query_offsets.back() - first_row;
        std::size_t const width = nominating == traversal_t::walk_k ? config.expansion_factor : 0;
        vectors_view_t const query_rows = shards_[0].index->import_rows_(queries, first_row, rows, traversal);
        bytes_span_t staged = to_writable_span(traversal.staged_queries)
                                  .subspan(staged_rows_bytes(vector_dtype_k, rows, queries.columns));
        std::span<vector_norms_t> const norms = carve_<vector_norms_t>(staged, rows);
        std::span<dot_result_t> const maxima = carve_<dot_result_t>(staged, rows);
        std::span<maxsim_ranked<total_t>> const best = carve_<maxsim_ranked<total_t>>(staged, queries_count * capacity);
        std::span<key_t> const best_keys = carve_<key_t>(staged, queries_count * capacity);
        std::span<std::size_t> const kept = carve_<std::size_t>(staged, queries_count);
        std::span<std::size_t> const worst = carve_<std::size_t>(staged, queries_count);
        std::span<key_t> const ranked = carve_<key_t>(staged, width);
        std::span<distance_t> const merged = carve_<distance_t>(staged, width);
        std::span<maxsim_hit_t> const hits = carve_<maxsim_hit_t>(staged, rows * width);
        std::span<maxsim_ranked<total_t>> const bounds = carve_<maxsim_ranked<total_t>>(staged, rows * width);
        std::span<distance_t> const row_gains = carve_<distance_t>(staged, rows);
        for (std::size_t row = 0; row != rows; ++row)
            norms[row] = shards_[0].index->norms(query_rows.row(row).as_vector());
        std::memset(kept.data(), 0, kept.size_bytes());
        std::memset(scored.data(), 0, scored.size_bytes());

        // Each query sums its own rows of a key's maxima.
        auto rank_key = [&](key_t const& key, std::span<dot_result_t const> key_maxima) noexcept {
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const begin = query_offsets[query] - first_row, end = query_offsets[query + 1] - first_row;
                total_t total {};
                for (dot_result_t const maximum : key_maxima.subspan(begin, end - begin)) total += total_t(maximum);
                rank_into_(best.subspan(query * capacity, capacity), best_keys.subspan(query * capacity, capacity),
                           kept[query], worst[query], key, total);
                ++scored[query];
            }
        };
        auto score_key = [&](key_t const& key) noexcept {
            for (dot_result_t& maximum : maxima) maximum = dot_result_t::finite_min();
            std::size_t folded = 0;
            shards_.locators().find_each(key, [&](key_locator_t locator) noexcept {
                if (locator.node >= watermarks[locator.shard]) return;
                vector_view_t const row = shards_[locator.shard].index->vector(locator.node);
                fold_quads<quads_kit_t>(
                    row, shards_[0].index->norms(row), query_rows, norms, maxima,
                    [](dot_result_t& maximum, dot_result_t dot) noexcept { maximum = max_of(maximum, dot); });
                ++folded;
            });
            if (!folded) return;
            traversal.computed_distances += folded * rows;
            rank_key(key, maxima);
        };

        // Hands `visit` each candidate once: every key at its first locator, asked a quad of rows
        // at a time and settled once per first chunk's width of rows, or a walk's keys by bytes.
        auto nominate = [&]<typename visit_type_>(visit_type_&& visit) noexcept {
            if (nominating == traversal_t::scan_k) {
                std::size_t const settled_rows = index_t::chunk_schedule_t::rows_in(0);
                for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                    shard_t const& one = shards_[shard];
                    auto const key_of = [&one](node_id_t node) noexcept { return one.keys[node]; };
                    node_predicate admits {predicate, key_of};
                    auto keep = [&](scan_payload_t node) noexcept { visit(one.keys[node]); };
                    for (std::size_t start = 0; start < watermarks[shard]; start += settled_rows) {
                        std::size_t const end = min_of<std::size_t>(start + settled_rows, watermarks[shard]);
                        quad<node_id_t> nodes;
                        quad<scan_payload_t> payloads;
                        std::size_t filled = 0;
                        for (std::size_t node = start; node != end; ++node) {
                            auto const [leader, leader_status] = shards_.locators().find_first(one.keys[node]);
                            if (failed(leader_status) || leader.shard != shard || leader.node != node) continue;
                            nodes[filled] = static_cast<node_id_t>(node);
                            payloads[filled] = static_cast<scan_payload_t>(node);
                            if (++filled != quad_lanes_k) continue;
                            ask_contenders(admits, nodes, payloads, filled, keep);
                            filled = 0;
                        }
                        ask_contenders(admits, nodes, payloads, filled, keep);
                        settle_predicate(predicate, keep);
                    }
                }
                return status_t::success_k;
            }
            // Each row's walks merge into its best `width` keys, which land among the hits with
            // how far each rose above the row's worst kept dot, the most a missed key can reach.
            std::size_t nominated = 0;
            total_t floors {};
            bool bounded = true;
            for (std::size_t row = 0; row != rows; ++row) {
                vector_view_t const query = query_rows.row(row).as_vector();
                std::size_t found = 0;
                for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                    shard_t const& one = shards_[shard];
                    expected<search_result_t> walked {};
                    if constexpr (!filtered_k)
                        walked = one.index->walk_(query, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config,
                                                  watermarks[shard]);
                    else {
                        auto const key_of = [&one](node_id_t node) noexcept { return one.keys[node]; };
                        node_predicate admits {predicate, key_of};
                        walked = one.index->walk_if_(query, admits, traversal, config, watermarks[shard]);
                    }
                    if (!walked) return walked.status;
                    found = merge_search_results(to_span(traversal.results), one.keys, ranked, merged, found);
                }
                if (!found) {
                    bounded = false;
                    continue;
                }
                distance_t const farthest = merged[found - 1];
                floors += total_t(1 - farthest);
                for (std::size_t hit = 0; hit != found; ++hit)
                    hits[nominated++] = {ranked[hit], static_cast<u32_t>(row), farthest - merged[hit]};
            }

            // Grouped by key, a key's bound adds its best gain per row to each row's floor dot.
            for (distance_t& gain : row_gains) gain = 0;
            radix_group(hits.first(nominated), &maxsim_hit_t::key);
            std::size_t unique = 0;
            for (std::size_t run = 0, end = 0; run != nominated; run = end) {
                while (end != nominated && hits[end].key == hits[run].key) ++end;
                for (std::size_t hit = run; hit != end; ++hit)
                    row_gains[hits[hit].row] = max_of(row_gains[hits[hit].row], hits[hit].gain);
                total_t bound = floors;
                for (std::size_t hit = run; hit != end; ++hit)
                    bound += total_t(row_gains[hits[hit].row]), row_gains[hits[hit].row] = 0;
                hits[unique].key = hits[run].key;
                bounds[unique] = {static_cast<u32_t>(unique), bound};
                ++unique;
            }
            partial_sort(bounds.first(unique), unique,
                         [](maxsim_ranked<total_t> const& first, maxsim_ranked<total_t> const& second) noexcept {
                             return first.total > second.total;
                         });
            // The walk serves one query; once its ranking fills, bounds only fall, so the first
            // below the worst kept total ends the visits. A tiled fold ranks late, so stops later.
            for (std::size_t position = 0; position != unique; ++position) {
                if (bounded && kept[0] == capacity && best[worst[0]].total > bounds[position].total) {
                    // Skipped keys still count as scored, so `scored` sizes a call that keeps all.
                    scored[0] += unique - position;
                    break;
                }
                visit(hits[bounds[position].candidate].key);
            }
            return status_t::success_k;
        };

        if constexpr (index_t::has_tiles_k)
            if (rows >= tiles_kit_t::tile_columns_k) {
                if (failed(traversal.reserve_tiles(tiles_kit_t::stream_rows_k, tiles_kit_t::pack_columns_k, 0)))
                    return status_t::bad_alloc_k;
                tiled_fold_t fold {*this, query_rows, staged, config.cache_bytes, watermarks, traversal};
                if (status_t const nominated = nominate([&](key_t const& key) noexcept { fold.add(key, rank_key); });
                    failed(nominated))
                    return nominated;
                fold.finish(rank_key);
                return land_ranked_(best, best_keys, kept, thread, first_query, land);
            }
        if (status_t const nominated = nominate(score_key); failed(nominated)) return nominated;
        return land_ranked_(best, best_keys, kept, thread, first_query, land);
    }

    /** Rebuilds every shard empty on its memory domain and regrows it to the capacity it had. */
    status_t clear_() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = rebuild_(*one.index);
            if (!fresh) return status_t::bad_alloc_k;
            one.index = std::move(fresh);
        }
        shards_.forget_rows();
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

    /** One unreserved shard placed on @p memory_domain; null when the allocation fails. */
    std::unique_ptr<index_t> build_(fu_memory_domain_id_t memory_domain) const noexcept {
        forkunion_allocator_t const allocator {memory_domain};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel(), allocator, allocator, line_bytes_));
    }

    /** One unreserved shard on the memory domain @p like sits on. */
    std::unique_ptr<index_t> rebuild_(index_t const& like) const noexcept {
        return build_(like.vectors_allocator_.memory_domain);
    }

    /** Sizes the caller's context of every executor thread for a walk of @p config staging
     *  @p staged_bytes of queries, seeded apart and with its counters zeroed, so the call reports
     *  only what it spent. */
    status_t size_traversals_(std::span<traversal_context_t> traversals, insert_config_t config,
                              std::size_t staged_bytes) noexcept {
        for (std::size_t thread = 0; thread != traversals.size(); ++thread) {
            traversal_context_t& traversal = traversals[thread];
            // Apart, and apart across calls: identically seeded contexts would send every
            // concurrent inserter through the same entry points.
            traversal.random_generator.state = shards_.routed() + thread;
            traversal.computed_distances = traversal.computed_distances_in_refines = 0;
            traversal.computed_distances_in_reverse_refines = traversal.visited_members = 0;
            traversal.restarts = traversal.saturated_walks = 0;
            if (status_t const sized = shards_[0].index->reserve_traversal(traversal, config); failed(sized))
                return sized;
            if (failed(traversal.reserve_queries(staged_bytes))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    using carved_traversals_t =
        growing_buffer<traversal_context_t, arena_allocator<traversal_context_t, forkunion_allocator_t>>;

    /** Rewinds every arena of @p executor once and carves one context per thread into
     *  @p traversals, its array in the first arena: what the one-shot drivers walk on, as no caller
     *  hands them contexts. The plans they drive size them. */
    status_t carve_traversals_(carved_traversals_t& traversals, executor_t& executor) noexcept {
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        if (failed(traversals.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            traversals.push_back_reserved(
                traversal_context_t {arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(thread))});
        return status_t::success_k;
    }

    /** Runs and finishes @p plan on contexts carved for the call: the one-shot refine, absorb and
     *  split. */
    template <typename progress_type_>
    expected<refine_result_t> drive_refine_(refine_plan_t& plan, executor_t& executor,
                                            progress_type_& on_visited) noexcept {
        carved_traversals_t traversals {arena_allocator<traversal_context_t, forkunion_allocator_t>(executor.arena(0))};
        if (status_t const carved = carve_traversals_(traversals, executor); failed(carved)) return {{}, carved};
        status_t const ran = plan.run(executor, to_writable_span(traversals), on_visited);
        auto const [result, finish_status] = plan.finish(executor, to_writable_span(traversals));
        return {result, failed(finish_status) ? finish_status : ran};
    }

#pragma region Health

    /** Lands every row of @p donor and consumes it, under the closed dispensers of both indexes
     *  and the executor's claim @p refining holds: every fallible step first, then the chunk
     *  steps, which cannot fail, then one commit bracket publishes, and the donor's shards join
     *  the plan's parts. */
    status_t absorb_(ram_indexes& donor, executor_t& executor, refine_plan_t& refining) noexcept {
        std::size_t const shards = shards_.size();
        if (!donor.size()) return status_t::success_k;
        if (!capacity())
            if (status_t const reserved = reserve({donor.size(), static_cast<connectivity_t>(donor.connectivity()),
                                                   donor.dimensions()})
                                              .status;
                failed(reserved))
                return reserved;

        // Every receiving shard claims all the rows it takes up front, and the donor's fresh shards
        // exist before any lands, so nothing past the claims can fail and a failure releases them.
        std::size_t const donors = donor.shards_.size();
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {shards_.allocator()};
        growing_buffer<std::unique_ptr<index_t>, forkunion_allocator_t> emptied {shards_.allocator()};
        if (failed(plans.reserve_exactly(shards)) || failed(cursors.resize(shards)) ||
            failed(emptied.reserve_exactly(donors)))
            return status_t::bad_alloc_k;
        std::size_t landing_rows = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = into; from < donors; from += shards) rows += donor.shards_[from].index->size();
            auto [plan, status] = shards_[into].index->plan_load(rows);
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
            if (status_t const covered = shards_[into].keys.cover(plans[into].base + rows); failed(covered))
                return covered;
            landing_rows += rows;
        }
        auto [seats, seat_status] = shards_.locators().claim(landing_rows);
        if (failed(seat_status)) return seat_status;
        for (shard_t const& part : donor.shards_) {
            emptied.push_back_reserved(donor.rebuild_(*part.index));
            if (!emptied.back()) return status_t::bad_alloc_k;
        }

        // A step is one storage chunk of a receiving shard, which the donor parts landing there
        // fill in turn, each with its keys; a chunk may straddle several parts.
        auto land = [&](std::size_t into, typename index_t::load_plan_t::step_t const& chunk, std::size_t,
                        nk::matrix_span<node_id_t>) noexcept -> status_t {
            iota_view<node_id_t> landing {plans[into].base, plans[into].base};
            for (std::size_t from = into; from < donors && landing.bound < chunk.index_rows.bound; from += shards) {
                shard_t const& part = donor.shards_[from];
                iota_view<node_id_t> const source_range {0, static_cast<node_id_t>(part.index->size())};
                landing = {landing.bound, landing.bound + source_range.bound};
                typename index_t::load_plan_t::step_t step = chunk;
                step.index_rows = {max_of(chunk.index_rows.first, landing.first),
                                   min_of(chunk.index_rows.bound, landing.bound)};
                if (step.index_rows.first >= step.index_rows.bound) continue;
                step.source_rows = {step.index_rows.first - landing.first, step.index_rows.bound - landing.first};
                if (status_t const loaded = plans[into].load(step, *part.index, source_range); failed(loaded))
                    return loaded;
                for (node_id_t row : step.index_rows) shards_[into].keys[row] = part.keys[row - landing.first];
            }
            return status_t::success_k;
        };
        if (status_t const landed = drain_steps_(plans, to_writable_span(cursors), executor, 0, land); failed(landed))
            return landed;

        // Every step landed, so no plan refuses; the bracket shows a search every shard's rows or
        // none, and the keys resolve only once their rows are visible.
        status_t published = status_t::success_k;
        shards_.begin_commit();
        for (typename index_t::load_plan_t& plan : plans)
            if (succeeded(published)) published = plan.finish();
        shards_.end_commit();
        if (failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                shards_.locators().insert(seats, shards_[into].keys[row], key_locator_t {into, row});

        // Each donor shard becomes a part of its own in the plan, which sized every shard's run.
        shards_.record_parts(donor.shards_, plans, refining.part_bases_.data(), to_span(refining.part_offsets_));
        for (std::size_t from = 0; from != donors; ++from) {
            shard_t& part = donor.shards_[from];
            forkunion_allocator_t const placed {part.index->vectors_allocator_.memory_domain};
            part.index = std::move(emptied[from]);
            part.keys = keys_column_t {placed};
        }
        donor.shards_.forget_rows();
        return status_t::success_k;
    }

    /** Once every shard's split published its parts: every receiver's keys resolve in the seats
     *  @p splitting claimed, and this index empties into the fresh shards built at planning. */
    void publish_split_(refine_plan_t& splitting) noexcept {
        for (std::size_t receiver = 0; receiver != splitting.receivers_.size(); ++receiver) {
            ram_indexes& into = *splitting.receivers_[receiver];
            for (std::size_t shard = 0; shard != into.shards_.size(); ++shard) {
                shard_t const& part = into.shards_[shard];
                for (node_id_t row = 0; row != static_cast<node_id_t>(part.index->size()); ++row)
                    into.shards_.locators().insert(splitting.receiver_seats_[receiver], part.keys[row],
                                                   key_locator_t {shard, row});
            }
        }
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            shard_t& part = shards_[shard];
            forkunion_allocator_t const placed {part.index->vectors_allocator_.memory_domain};
            part.index = std::move(splitting.emptied_[shard]);
            part.keys = keys_column_t {placed};
        }
        shards_.forget_rows();
    }

    /** One reachability audit over every shard's plan: the bitsets restart, then every level
     *  sweeps every step and advances each plan once, until no level expands a node. */
    template <typename plan_type_>
    status_t sweep_reachability_(growing_buffer<plan_type_, forkunion_allocator_t>& plans,
                                 std::span<std::size_t> cursors, executor_t& executor) noexcept {
        for (plan_type_& plan : plans) plan.restart_sweep();
        auto sweep_step = [&](std::size_t shard, auto const& step, std::size_t, nk::matrix_span<node_id_t>) noexcept {
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

#pragma endregion Health

#pragma region Search

    /** Takes the shards' snapshot into @p traversal 's staged IDs, idle during any search and sized
     *  once per context as an insertion sizes them, so a read allocates nothing per call; empty
     *  when they cannot hold one watermark per shard. */
    std::span<node_id_t const> snapshot_(traversal_context_t& traversal) const noexcept {
        if (failed(traversal.reserve_neighbors(max_of(connectivity(), shards_.size()))) ||
            nk::failed(traversal.staged_ids.resize({shards_.size()})))
            return {};
        std::span<node_id_t> const watermarks {traversal.staged_ids.data(), shards_.size()};
        shards_.snapshot(watermarks);
        return watermarks;
    }

    /**
     *  @brief The scan behind @ref search_if: every row published when a query's block starts is
     *      scored, so the rankings are the ones a graph walk approximates.
     *
     *  Each block of queries takes a thread and its traversal, scans every shard below one snapshot
     *  of the watermarks, and lands each row beating a query's worst match in its output row
     *  through the shard's key column. A block at least one tile wide, on a platform with a matrix
     *  engine for the scalar, streams through it a pack at a time, in blocks of rows sized by
     *  @p config's cache bytes; a single query scores four rows at a time. Either way the shards'
     *  own kit finalizes the distances, so they match the walk's.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> scan_rows_(rows_view_t queries, executor_t& executor,
                                         std::span<traversal_context_t> traversals, search_config_t config,
                                         nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                         std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows, wanted = keys.extent(1);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        std::memset(counts.data(), 0, counts.size_bytes());
        if (!queries_count || !wanted) return {{}, status_t::success_k};

        std::size_t const width = index_t::scan_width(queries_count, executor.size());
        auto scan_block = [&](std::size_t block, std::size_t thread, traversal_context_t& traversal,
                              std::span<node_id_t const> watermarks) noexcept {
            std::size_t const first = block * width;
            rows_view_t const block_queries = queries.slice(first, min_of(width, queries_count - first));
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                shard_t const& one = shards_[shard];
                // Each scanned match merges into its query's row as it beats the worst one there.
                auto land = [&](std::size_t column, scored_node_t const candidate) noexcept {
                    std::size_t const query = first + column;
                    std::span<score_type_> const distances_row = to_span(distances.row(query).as_vector());
                    counts[query] = merge_search_results(std::span(&candidate, 1), one.keys,
                                                         to_span(keys.row(query).as_vector()), distances_row,
                                                         counts[query]);
                    return counts[query] == wanted ? static_cast<distance_t>(distances_row.back())
                                                   : std::numeric_limits<distance_t>::max();
                };
                status_t scanned = status_t::success_k;
                if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    scanned = one.index->scan_(block_queries, dummy_predicate_t {}, land, traversal, config,
                                               watermarks[shard]);
                else {
                    auto const key_of = [&one](node_id_t node) noexcept { return one.keys[node]; };
                    scanned = one.index->scan_(block_queries,
                                               node_predicate {thread_predicate(predicate, thread), key_of}, land,
                                               traversal, config, watermarks[shard]);
                }
                if (failed(scanned)) return scanned;
            }
            return status_t::success_k;
        };
        // Every shard's scan sizes the staging its block imports and packs into.
        return for_each_row_(divide_round_up(queries_count, width), executor, traversals, search_config_t {}, 0,
                             scan_block, counts, predicate, scan_round_capacity<typename index_t::tiles_kit_t>());
    }

    /** The batched @ref search_if under @p heads, which only a @c dummy_predicate_t accompanies. */
    template <typename score_type_, typename heads_type_, typename predicate_type_>
    expected<search_result_t> search_rows_(rows_view_t queries, heads_type_ const& heads, executor_t& executor,
                                           std::span<traversal_context_t> traversals, search_config_t config,
                                           nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                           std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows;
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};

        auto search_row = [&](std::size_t query, std::size_t thread, traversal_context_t& traversal,
                              std::span<node_id_t const> watermarks) noexcept {
            std::span<key_t> const keys_row = to_span(keys.row(query).as_vector());
            std::span<score_type_> const distances_row = to_span(distances.row(query).as_vector());
            vector_view_t const query_vector =
                shards_[0].index->import_rows_(queries, query, 1, traversal).row(0).as_vector();
            std::size_t found_count = 0;
            status_t refusal = status_t::success_k;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                shard_t const& one = shards_[shard];
                expected<search_result_t> found {};
                if constexpr (!std::is_same_v<heads_type_, dummy_heads_t>)
                    found = one.index->walk_(query_vector, dummy_fold_t {},
                                             indexer_rescoring<quads_kit_t> {heads.row(query)}, traversal, config,
                                             watermarks[shard]);
                else if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    found = one.index->walk_(query_vector, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config,
                                             watermarks[shard]);
                else {
                    auto const key_of = [&one](node_id_t node) noexcept { return one.keys[node]; };
                    node_predicate admits {thread_predicate(predicate, thread), key_of};
                    found = one.index->walk_if_(query_vector, admits, traversal, config, watermarks[shard]);
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
        return for_each_row_(queries_count, executor, traversals, config,
                             staged_rows_bytes(vector_dtype_k, 1, queries.columns), search_row, counts, predicate,
                             index_t::round_capacity(shards_[0].index->connectivity()));
    }

    /** Claims @p executor, sizes the caller's @p traversals, one per thread, with @p staged_bytes
     *  of staged queries, runs @p per_row over every row with its thread, that thread's traversal
     *  and a fresh snapshot of the watermarks, and sums the threads' telemetry & the rows'
     *  @p counts; a row that refuses stops itself, and the first refusal is the one kept. An
     *  aggregating @p predicate reserves each thread's buffers for @p candidates. Nothing here
     *  rewinds an arena, so whatever the caller carved there lives through the call. */
    template <typename per_row_type_, typename predicate_type_ = dummy_predicate_t>
    expected<search_result_t> for_each_row_(std::size_t rows_count, executor_t& executor,
                                            std::span<traversal_context_t> traversals, search_config_t config,
                                            std::size_t staged_bytes, per_row_type_&& per_row,
                                            std::span<std::size_t const> counts, predicate_type_&& predicate = {},
                                            std::size_t candidates = 0) noexcept {
        // Only `scan_rows_` scans, and it drives the rows under a walk's sizing.
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (traversals.size() != executor.size()) return {{}, status_t::config_mismatch_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (status_t const sized = size_traversals_(traversals, insert_config_t {config.expansion_factor},
                                                    staged_bytes);
            failed(sized))
            return {{}, sized};
        if (status_t const opened = open_predicate(predicate, executor); failed(opened)) return {{}, opened};
        for (std::size_t thread = 0; thread != traversals.size(); ++thread)
            if (failed(reserve_predicate(thread_predicate(predicate, thread), candidates)))
                return {{}, status_t::bad_alloc_k};

        status_t first_refusal = status_t::success_k;
        alignas(default_alignment_k) std::size_t live_rows = 0;
        std::size_t const filling = filling_rows<std::size_t>(shards_[0].index->non_zero_expansion_factor(config),
                                                              shards_[0].index->connectivity());
        auto each_row = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            traversal_context_t& traversal = traversals[thread];
            std::size_t const live = atomic_ref<std::size_t>(live_rows).fetch_add(1, std::memory_order_relaxed) + 1;
            traversal.walk_width = walk_width(executor.size(), live, rows_count - row, filling);
            std::span<node_id_t const> const watermarks = snapshot_(traversal);
            status_t const refusal = watermarks.size() == shards_.size() ? per_row(row, thread, traversal, watermarks)
                                                                         : status_t::bad_alloc_k;
            atomic_ref<std::size_t>(live_rows).fetch_sub(1, std::memory_order_relaxed);
            if (succeeded(refusal)) return;
            status_t expected_status = status_t::success_k;
            atomic_ref<status_t>(first_refusal)
                .compare_exchange_strong(expected_status, refusal, std::memory_order_relaxed,
                                         std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(rows_count, each_row); failed(fanned)) return {{}, fanned};

        search_result_t merged;
        for (std::size_t const count : counts) merged.count += count;
        for (traversal_context_t const& traversal : traversals) {
            merged.visited_members += traversal.visited_members;
            merged.computed_distances += traversal.computed_distances;
            merged.restarts += traversal.restarts;
            merged.saturated_walks += traversal.saturated_walks;
        }
        return {merged, first_refusal};
    }

#pragma endregion Search

#pragma region Checkpoints

    /** One slab of neighbor IDs from @p arena - the scratch a save chunk rebases through; empty
     *  when the arena is short. */
    nk::matrix_span<node_id_t> scratch_of_(typename executor_t::arena_t& arena,
                                           std::size_t scratch_rows) const noexcept {
        std::size_t const connectivity = this->connectivity();
        std::byte* bytes = arena.allocate(scratch_rows * connectivity * sizeof(node_id_t), alignof(node_id_t));
        if (!bytes) return {};
        return {reinterpret_cast<node_id_t*>(bytes), scratch_rows, connectivity};
    }

    /** The home shard of @p thread : the first shard on its memory domain, so its drains start
     *  there, or a spread over the shards when none sits there. */
    std::size_t pick_home_shard_(std::size_t thread, executor_t const& executor) const noexcept {
        std::size_t const shards = shards_.size();
        for (std::size_t shard = 0; shard != shards; ++shard)
            if (shards_[shard].index->vectors_allocator_.memory_domain == executor.memory_domain(thread)) return shard;
        return thread % shards;
    }

    /**
     *  @brief Fans every plan's chunks out over the executor's threads: each thread drains its
     *      home shard first, then helps the rest, one atomic cursor per shard being the whole
     *      schedule.
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
            nk::matrix_span<node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, status_t::bad_alloc_k, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = pick_home_shard_(thread, executor);
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (home + visited) % shards;
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                auto const steps = plans[shard].steps();
                // The failure check comes before the claim, so every claimed step runs and the
                // cursor says where a stopped plan stopped.
                while (!failed(atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) {
                    std::size_t const position = cursor.fetch_add(1, std::memory_order_relaxed);
                    if (position >= steps.size()) break;
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

    /** Fans @p plan 's steps out over the executor's threads: each thread drains the run of the
     *  shard it is home to first, then helps the rest, one atomic cursor per shard being the whole
     *  schedule. A save lands each step with the thread's scratch of neighbor IDs; a load reads
     *  straight out of its part. The first failure wins and stops the rest. */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan, executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        auto const steps = plan.steps();
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step,
                                           nk::matrix_span<node_id_t> scratch) { planned.save(step, scratch); };
        std::size_t const scratch_rows = saving_k
                                             ? power_of_two<std::size_t>(index_t::chunk_schedule_t::slab_rows_log2_k)
                                             : 0;

        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (scratch_rows)
            if (status_t const sized = executor.reserve(scratch_rows * connectivity() * sizeof(node_id_t));
                failed(sized))
                return sized;
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {shards_.allocator()};
        if (failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        status_t outcome = status_t::success_k;
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            nk::matrix_span<node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, status_t::bad_alloc_k, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = pick_home_shard_(thread, executor);
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

#pragma endregion Checkpoints

    /** The shards, the keys of their rows, their routing and the commit word every reader
     *  snapshots by. */
    shards_t shards_;

    /** Held while a planned batch dispenses its shards' ranges, so no two batches' ranges cross
     *  order between the shards and each can wait for its turns outside the commit bracket. */
    fu::spin_mutex_t dispensing_;

    /** The line every shard this index builds prefetches by. */
    std::size_t line_bytes_;
};

/**
 *  @brief External reader's cursor over the published rows of one @c ram_indexes: keys
 *      and vectors in placement order, shard by shard.
 *
 *  Pure position - every read names the index it reads, which must be the one the cursor was
 *  opened on, so the cursor keeps no reference that could outlive it. Strictly read-only - the
 *  index is never written through, so readers never stall writers. A shard's watermark is
 *  snapshotted as the walk enters it, so a row once passed is never revisited, while a later
 *  shard may carry rows published after the walk began.
 */
template <typename indexes_type_>
struct ram_indexes_cursor {

    using indexes_t = indexes_type_;
    using key_t = typename indexes_t::key_t;
    using scalar_t = typename indexes_t::scalar_t;

    /** Enters the first shard of @p indexes, the one every later @ref read must name. */
    explicit ram_indexes_cursor(indexes_t const& indexes) noexcept { enter_(indexes); }

    /** Copies up to @p keys.size() rows of @p indexes into @p keys, and exports their vectors into
     *  @p vectors in its scalar unless it holds no rows, advancing past them and reporting the rows
     *  written. An export wanting vectors must offer at least as many rows as @p keys, each at
     *  least as wide as the index, or nothing is read. */
    std::size_t read(indexes_t const& indexes, std::span<key_t> keys, rows_span_t vectors) noexcept {
        bool const with_vectors = vectors.rows != 0;
        std::size_t const dimensions = indexes.dimensions();
        if (with_vectors && (vectors.rows < keys.size() || vectors.columns < dimensions)) return 0;
        rows_span_t const target {vectors.data, vectors.rows, dimensions, vectors.stride, vectors.dtype};
        std::size_t written = 0;
        for (; written != keys.size() && shard_ < indexes.shards_.size(); ++written) {
            auto const& shard = indexes.shards_[shard_];
            keys[written] = shard.keys[node_];
            if (with_vectors)
                export_row<typename indexes_t::elementwise_kit_t>(shard.index->vector(node_).byte_data(),
                                                                  indexes_t::vector_dtype_k, target, written);
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
            if ((visible_ = static_cast<node_id_t>(indexes.shards_[shard_].index->size())) != 0) return;
    }

    /** @brief The shard being read, positional in placement order; the shard count past the end. */
    std::size_t shard_ = 0;
    /** @brief The current row inside that shard. */
    node_id_t node_ = 0;
    /** @brief That shard's watermark snapshot: rows strictly below it are readable. */
    node_id_t visible_ = 0;
};

} // namespace unum::usearch::dense
