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
 *  ranking. Checkpoints are @c sharded_indexes', every step fanned out over the executor's threads.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memset`

#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits` - the open threshold of a short exact ranking
#include <memory>      // `std::unique_ptr`
#include <mutex>       // `std::lock_guard` - the dispensing mutex
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::move`

#include <forkunion.h> // `fu_memory_domain_id_t`

#include "usearch/dense/sharded_indexes.hpp"
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
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                     //
    nk_capability_t capability_ = nk_cap_icelake_k,                           //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = fu::standard_cache_hints_t,                  //
    typename key_hash_type_ = std::hash<key_type_>                            //
    >
struct ram_indexes
    : sharded_indexes<ram_indexes<key_type_, vector_dtype_, capability_, wait_type_, atomic_ref_template_, //
                                  cache_hints_type_, key_hash_type_>,                                      //
                      ram_index<vector_dtype_, capability_, wait_type_, atomic_ref_template_,              //
                                cache_hints_type_,                                                         //
                                forkunion_allocator_t,                                                     //
                                forkunion_allocator_t,                                                     //
                                arena_allocator<byte_t, forkunion_allocator_t>>,                           //
                      forkunion_executor_t, key_type_, key_hash_type_, atomic_ref_template_> {
    friend struct ram_indexes_cursor<ram_indexes>;

    using base_t = sharded_indexes<
        ram_indexes,
        ram_index<vector_dtype_, capability_, wait_type_, atomic_ref_template_, cache_hints_type_,
                  forkunion_allocator_t, forkunion_allocator_t, arena_allocator<byte_t, forkunion_allocator_t>>,
        forkunion_executor_t, key_type_, key_hash_type_, atomic_ref_template_>;

    /** A reader's walk over every shard's published rows in placement order. */
    using cursor_t = ram_indexes_cursor<ram_indexes>;

    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    using typename base_t::distance_t;
    using typename base_t::executor_t;
    using typename base_t::health_result_t;
    using typename base_t::index_t;
    using typename base_t::insert_config_t;
    using typename base_t::insert_result_t;
    using typename base_t::key_locator_t;
    using typename base_t::key_t;
    using typename base_t::keys_column_t;
    using typename base_t::load_plan_t;
    using typename base_t::locators_map_t;
    using typename base_t::refine_config_t;
    using typename base_t::refine_result_t;
    using typename base_t::reserve_capacity_t;
    using typename base_t::save_plan_t;
    using typename base_t::scalar_t;
    using typename base_t::search_config_t;
    using typename base_t::search_result_t;
    using typename base_t::shard_t;
    using typename base_t::traversal_context_t;
    using typename base_t::vector_view_t;
    using typename base_t::vectors_view_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;
    static_assert(index_family<index_t> && index_claim<shard_capacity_claim_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The compute-capability mask the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

#pragma region Placement

    /** Every index names the distance kind its shards walk by and the host allocator its shard
     *  table, locator map and checkpoint catalogs sit on; nothing is placed by default. */
    ram_indexes() = delete;
    explicit ram_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : base_t(kernel, allocator) {}
    ram_indexes(ram_indexes&& other) noexcept : base_t(other.kernel_, other.allocator_) { this->swap(other); }
    ram_indexes& operator=(ram_indexes&& other) noexcept { return this->swap(other), *this; }

    using base_t::capacity;
    using base_t::connectivity;
    using base_t::dimensions;
    using base_t::emplace;
    using base_t::reserve;
    using base_t::size;

    /** Appends one shard built here on @p memory_domain, for callers with none to hand over. */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(memory_domain);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** Empties every shard in place, keeping its memory domain, geometry and capacity. Closes every
     *  shard's dispenser first, so a batch in flight refuses it with @c contended_k; the shards are
     *  rebuilt, so no insertion, search or other call may arrive meanwhile. */
    status_t clear() noexcept {
        if (status_t const closed = close_shards_(); failed(closed)) return closed;
        status_t const cleared = base_t::clear();
        // A shard the rebuild failed to replace is still closed.
        reopen_shards_();
        return cleared;
    }

#pragma endregion Placement

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
        node_id_t assigned = 0;
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        status_t const inserted = one.index->insert(std::move(claim), row, {}, traversal, config,
                                                    [&](node_id_t assigned_id) noexcept {
                                                        assigned = assigned_id;
                                                        one.keys[assigned_id] = key;
                                                    });
        if (failed(inserted)) return {result, inserted};
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
        if (status_t const landed = plan.insert(0, keys, new_rows); failed(landed)) return {result, landed};
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
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept
            : claims_(allocator), ranges_(allocator), traversals_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : claims_(std::move(other.claims_)), ranges_(std::move(other.ranges_)),
              traversals_(std::move(other.traversals_)), executor_claim_(std::move(other.executor_claim_)),
              executor_(other.executor_), seats_(other.seats_), config_(other.config_), rotation_(other.rotation_),
              rows_(other.rows_), landed_rows_(other.landed_rows_), owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /**
         *  @brief Lands rows @p first_row onward of the plan, one per row of @p rows under its key
         *      in @p keys, fanned out over the executor's threads, each imported into the stored
         *      scalar as it lands; fails only before its rows are dispensed.
         *
         *  The first step dispenses every shard's range under the dispensing mutex, so no two
         *  batches' ranges cross order between the shards. Rows outside the plan or of another
         *  width land nothing, and @ref finish refuses until every row landed. One caller at a
         *  time, like the executor.
         *
         *  @retval contended_k A refine, an absorb or a load holds a shard's dispenser closed;
         *      nothing was dispensed, so a later step may retry and a dropped plan releases its
         *      claims.
         *  @retval config_mismatch_k Rows outside the plan or of another width.
         */
        status_t insert(std::size_t first_row, nk::vector_view<key_t> keys, rows_view_t rows) noexcept {
            std::size_t const count = rows.rows;
            if (!owner_) return status_t::uninitialized_k;
            if (!count) return status_t::success_k;
            if (keys.size() != count || first_row > rows_ || count > rows_ - first_row ||
                rows.columns != owner_->dimensions())
                return status_t::config_mismatch_k;
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
                one.index->wire_(locator.node, rows, task, traversals_[thread], config_);
            };
            drain_in_order_(count, land_row);
            landed_rows_ += count;
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
            insert_result_t result;
            if (!owner_) return {result, status_t::uninitialized_k};
            result.new_size = owner_->size();
            if (landed_rows_ != rows_) return {result, status_t::config_mismatch_k};
            std::size_t touched_shards = 0;
            for (iota_view<node_id_t> const range : ranges_) touched_shards += !range.empty();
            bool const spans_shards = touched_shards > 1;

            // No later range on a shard can take its turn ahead of this one, so the waits hold no bracket open.
            for (std::size_t shard = 0; shard != ranges_.size(); ++shard)
                if (!ranges_[shard].empty()) owner_->shards_[shard].index->await_turn_(ranges_[shard]);
            // The low half counts the brackets open, and closing one carries into the high half, so a reader
            // sees the word move even when two brackets overlap.
            atomic_ref<u64_t> const sequence {owner_->commit_sequence_};
            if (spans_shards) sequence.add(1, std::memory_order_release);
            for (std::size_t shard = 0; shard != ranges_.size(); ++shard)
                if (!ranges_[shard].empty()) owner_->shards_[shard].index->publish_(ranges_[shard]);
            if (spans_shards) sequence.add((u64_t {1} << 32) - 1, std::memory_order_release);

            auto locate_row = [&](std::size_t row, std::size_t) noexcept {
                key_locator_t const locator = locator_of_(row);
                typename locators_map_t::seats_t seat {seats_.next + row, seats_.next + row + 1};
                owner_->locators_of_key_.insert(seat, owner_->shards_[locator.shard].keys[locator.node], locator);
            };
            drain_in_order_(rows_, locate_row);

            result.new_size = owner_->size();
            for (traversal_context_t const& traversal : traversals_) {
                result.visited_members += traversal.visited_members;
                result.computed_distances += traversal.computed_distances;
                result.computed_distances_in_refines += traversal.computed_distances_in_refines;
                result.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
                result.restarts += traversal.restarts;
                result.saturated_walks += traversal.saturated_walks;
            }
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

        /** One traversal per executor thread, over that thread's arena. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

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

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once @c finish published. */
        ram_indexes* owner_ = nullptr;

        friend ram_indexes;
    };

    /**
     *  @brief Opens a batched insertion of @p rows: takes the executor and its scratch, every shard's
     *      share of capacity and key column, and the locator seats, all of them reversible.
     *
     *  Every fallible step happens here, so the plan's rows can only land, and nothing is dispensed
     *  until the plan's first step. The rotation comes from the rows routed so far, which keeps a
     *  stream of small batches from drifting onto shard zero.
     *
     *  @param[in] rows Rows the plan lands.
     *  @param[inout] executor The threads and their scratch, held until the plan dies.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        insert_plan_t plan {allocator_};
        plan.executor_ = &executor;
        plan.config_ = config;
        plan.rows_ = rows;
        plan.owner_ = this;
        if (!rows) return {std::move(plan), status_t::success_k};

        // One caller per executor; the arenas grow to the widest geometry ever asked of them.
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.executor_claim_ = std::move(executor_claim);
        if (status_t const sized = executor.reserve(shards_[0].index->traversal_memory_usage(config)); failed(sized))
            return {{}, sized};

        // Each shard's key column is covered to what the shard hosts after its share, and a refusal
        // releases every share already granted as the claims die.
        if (failed(plan.claims_.reserve_exactly(shards)) || failed(plan.ranges_.reserve_exactly(shards)))
            return {{}, status_t::bad_alloc_k};
        plan.rotation_ = atomic_ref<std::size_t>(routed_rows_).fetch_add(rows, std::memory_order_relaxed) % shards;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_t& one = shards_[shard];
            auto [shard_claim, status] = one.index->claim(
                rows_from_(rows, first_row_of_shard_(shard, shards, plan.rotation_), shards));
            if (failed(status)) return {{}, status};
            plan.claims_.push_back_reserved(std::move(shard_claim));
            if (status_t const covered = one.keys.cover(one.index->capacity()); failed(covered)) return {{}, covered};
        }
        auto [seats, seat_status] = locators_of_key_.claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        if (status_t const ready = reserve_traversals_(plan.traversals_, executor, config); failed(ready))
            return {{}, ready};
        return {std::move(plan), status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Refinement

    /**
     *  @brief One refine in flight over every shard, holding every dispenser closed meanwhile.
     *
     *  It owns its audits, walk contexts and part starts, so nothing about it lives in the index;
     *  @ref run continues it after a cancel, and @ref finish heals what a stopped pass leaves and
     *  reopens. Every shard's @c refine_plan_t drains over the threads phase by phase, each thread
     *  pulling steps from its home shard first and stealing from the rest in the tail, so a pass
     *  costs one sweep of the graph however the shards are filled. An absorb's plan first lands its
     *  donors, one after another, and only then opens the shards' passes over the parts they
     *  appended.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb ever fills it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : arenas_(allocator), traversals_(allocator), homes_(allocator), shard_plans_(allocator),
              cursors_(allocator), part_bases_(allocator), part_offsets_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : arenas_(std::move(other.arenas_)), traversals_(std::move(other.traversals_)),
              homes_(std::move(other.homes_)), shard_plans_(std::move(other.shard_plans_)),
              cursors_(std::move(other.cursors_)), part_bases_(std::move(other.part_bases_)),
              part_offsets_(std::move(other.part_offsets_)), donors_(other.donors_),
              landed_donors_(other.landed_donors_), config_(other.config_), visited_(other.visited_),
              population_(other.population_), owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** An unfinished plan heals nothing: it reopens, and the graph is as the last step left
         *  it. */
        ~refine_plan_t() noexcept {
            if (owner_) owner_->reopen_shards_();
        }

        /**
         *  @brief Runs the pending phases on @p executor until the pass ends or @p on_visited
         *      cancels; @c cancelled_k means a later call continues exactly where this one stopped.
         *
         *  Lands the donors still pending first, as @ref plan_absorb describes; a failing one stays
         *  pending, with every later one, for the next call. The executor is held only for the
         *  call.
         *
         *  @param[inout] executor The executor the plan was sized for.
         *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
         *      population; a @c cancel_k answer stops the drain where its steps ran.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (executor.size() != traversals_.size()) return status_t::config_mismatch_k;
            auto [executor_claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; landed_donors_ != donors_.size(); ++landed_donors_) {
                ram_indexes*& donor = donors_[landed_donors_];
                if (status_t const closed = donor->close_shards_(); failed(closed)) return closed;
                status_t const landed = owner_->absorb_(*donor, executor, *this);
                // Emptied donor shards are fresh and open, so only a failure leaves them closed.
                donor->reopen_shards_();
                if (failed(landed)) return landed;
                donor = nullptr;
            }
            if (shard_plans_.empty())
                if (status_t const opened = open_shards_(); failed(opened)) return opened;
            return drain_(executor, on_visited);
        }

        /** Stops any re-selection, audits and heals, reopens the dispensers, and reports. A plan
         *  whose donors have not all landed heals nothing and reports nothing. */
        expected<refine_result_t> finish(executor_t& executor) noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (executor.size() != traversals_.size()) return {{}, status_t::config_mismatch_k};
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
                healed = drain_(executor, heal_alone);
            }
            refine_result_t const result = running();
            std::exchange(owner_, nullptr)->reopen_shards_();
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

        /** Opens every shard's pass over its parts, its audit on the arena of its home thread, once
         *  no donor is pending. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != owner_->shards_.size(); ++shard) {
                std::span<node_id_t const> parts;
                if (!part_offsets_.empty())
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
                // The first thread homed on the shard holds its audit, on the shard's own domain.
                std::size_t thread = 0;
                while (thread != homes_.size() && homes_[thread] != shard) ++thread;
                auto [plan, status] = owner_->shards_[shard].index->plan_refine(
                    {0, node_id_unknown_k}, traversals_[thread != homes_.size() ? thread : 0], config_, parts);
                if (failed(status)) return shard_plans_.clear(), status;
                shard_plans_.push_back_reserved(std::move(plan));
            }
            population_ = owner_->size();
            return status_t::success_k;
        }

        /** Drains every shard's pending phases from where the cursors stand, advancing each plan
         *  between them, until none follows or a step fails or cancels. */
        template <typename progress_type_>
        status_t drain_(executor_t& executor, progress_type_& on_visited) noexcept {
            // Every step bumps it, so it takes its own line, away from the locals workers read.
            alignas(default_alignment_k) std::size_t visited = visited_;
            auto run_step = [&](std::size_t shard, typename index_t::refine_plan_t::step_t const& step,
                                std::size_t thread, nk::matrix_span<node_id_t>) noexcept -> status_t {
                typename index_t::refine_plan_t& plan = shard_plans_[shard];
                bool const reselecting = plan.reselecting();
                if (status_t const ran = plan.run(step, traversals_[thread]); failed(ran)) return ran;
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
                drained = owner_->drain_steps_(shard_plans_, cursors, to_span(homes_), executor, 0, run_step);
                if (failed(drained)) break;
                advancing = 0;
                for (typename index_t::refine_plan_t& plan : shard_plans_) advancing += plan.advance();
                std::memset(cursors.data(), 0, cursors.size_bytes());
            } while (advancing);
            visited_ = visited;
            return drained;
        }

        /** One arena per executor thread, on that thread's memory domain, so no other call on the
         *  executor rewinds what the plan keeps between its calls. */
        growing_buffer<typename executor_t::arena_t, forkunion_allocator_t> arenas_;

        /** One walk context per executor thread, over that thread's arena of the plan. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

        /** Every executor thread's home shard, picked once: every drain starts the thread there,
         *  and the first thread homed on a shard holds that shard's audit. */
        growing_buffer<std::size_t, forkunion_allocator_t> homes_;

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

    /** Opens a refine of every shard: closes the dispensers and sizes everything the plan runs on,
     *  as @ref plan_absorb does with no donors, then opens the shards' passes. Re-selection claims
     *  no identity and moves no watermark, so searches run throughout. */
    expected<refine_plan_t> plan_refine(executor_t const& executor, refine_config_t config = {}) noexcept {
        return plan_absorb({}, executor, config);
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
     *  @param[in] executor The executor every @c run and @c finish names; its threads size the
     *      plan.
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
            if (!donor || donor == this || donor->kernel_ != kernel_) return {{}, status_t::config_mismatch_k};
            if (!donor->capacity()) continue;
            if (!shaped) shaped = donor;
            else if (donor->dimensions() != shaped->dimensions() || donor->connectivity() != shaped->connectivity())
                return {{}, status_t::config_mismatch_k};
        }
        if (status_t const closed = close_shards_(); failed(closed)) return {{}, closed};
        refine_plan_t plan {allocator_};
        plan.owner_ = this;
        plan.donors_ = donors;
        plan.config_ = config;
        if (failed(plan.arenas_.reserve_exactly(threads)) || failed(plan.traversals_.reserve_exactly(threads)) ||
            failed(plan.shard_plans_.reserve_exactly(shards)) || failed(plan.cursors_.resize(shards)) ||
            failed(pick_home_shards_(plan.homes_, executor)))
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

        // A re-selection walks like any other query, so it sizes like one; contexts seeded apart,
        // and apart across calls, so concurrent walks start from different entry points.
        insert_config_t const walk_config {config.expansion_factor};
        index_t& shape = *(shaped ? shaped : this)->shards_[0].index;
        std::size_t const walk_bytes = shape.traversal_memory_usage(walk_config);
        std::size_t const seed = atomic_ref<std::size_t>(routed_rows_).load(std::memory_order_relaxed);
        for (std::size_t thread = 0; thread != threads; ++thread) {
            plan.arenas_.push_back_reserved(
                typename executor_t::arena_t {forkunion_allocator_t {executor.memory_domain(thread)}});
            if (failed(plan.arenas_[thread].reserve_exactly(walk_bytes))) return {{}, status_t::bad_alloc_k};
            traversal_context_t traversal {arena_allocator<byte_t, forkunion_allocator_t>(plan.arenas_[thread])};
            traversal.random_generator.state = seed + thread;
            plan.traversals_.push_back_reserved(std::move(traversal));
            if (status_t const sized = shape.reserve_traversal(plan.traversals_[thread], walk_config); failed(sized))
                return {{}, sized};
        }
        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Health

    /**
     *  @brief Reachability of every shard from as many landmarks as one search draws, swept level
     *      by level over the threads. Read-only, so concurrent insertions leave fresh nodes uncounted.
     *
     *  @param[inout] executor The threads and their scratch; each shard's bitsets land on the arena
     *      of a thread living on its memory domain, growing it while planning.
     */
    expected<health_result_t> health(executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};

        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        if (status_t const ready = reserve_audit_(traversals, cursors, executor); failed(ready)) return {{}, ready};

        growing_buffer<typename index_t::health_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> homes {allocator_};
        if (failed(plans.reserve_exactly(shards)) || failed(pick_home_shards_(homes, executor)))
            return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            // The shard's first homed thread holds its audit, so the bitsets land on its domain.
            std::size_t thread = 0;
            while (thread != homes.size() && homes[thread] != shard) ++thread;
            auto [plan, status] = shards_[shard].index->plan_health(traversals[thread != homes.size() ? thread : 0]);
            if (failed(status)) return {{}, status};
            plans.push_back_reserved(std::move(plan));
        }
        if (status_t const audited = sweep_reachability_(plans, to_writable_span(cursors), to_span(homes), executor);
            failed(audited))
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
     *  @param[inout] executor The threads and their scratch; one query occupies one at a time.
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
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts) noexcept {
        return search_if(queries, executor, config, keys, distances, counts, dummy_predicate_t {});
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its key's indexer score against the heads of its query row.
     *
     *  @param[in] heads The indexer heads of every query row.
     *  @return @c config_mismatch_k unless the shards walk by dot.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, heads, executor, config, keys, distances, counts, dummy_predicate_t {});
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
    expected<search_result_t> search_if(rows_view_t queries, executor_t& executor, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        if (config.traversal == traversal_t::scan_k)
            return scan_rows_(queries, executor, keys, distances, counts, predicate);
        return search_rows_(queries, dummy_heads_t {}, executor, config, keys, distances, counts, predicate);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row keeps
     *      the pairs its walks scored, keys resolved and dots cast into the products' scalar.
     *
     *  @param[in] left The left-hand rows, one walk per row, in any scalar NumKong casts into the
     *      stored one; each thread imports its own.
     *  @param[inout] executor The threads and their scratch; one row occupies one at a time.
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
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                       std::span<std::size_t> counts) noexcept {
        std::size_t const rows_count = left.rows;
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(rows_count, keys, products); failed(refusal))
            return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};

        auto multiply_row = [&](std::size_t row, std::size_t, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks) noexcept {
            vector_view_t const left_row = shards_[0].index->import_rows_(left, row, 1, traversal).row(0).as_vector();
            multiply_fold<key_t, product_type_, keyed_ids<keys_column_t>> fold {to_span(keys.row(row).as_vector()),
                                                                                to_span(products.row(row).as_vector())};
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
        return for_each_row_(rows_count, executor, config, staged_rows_bytes(vector_dtype_, 1, left.columns),
                             multiply_row, counts);
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
     *  @param[out] keys The best-scoring keys of each query, one query per row, as many keys
     *      as its width.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, at most the width of @p keys.
     *
     *  A key some row never ranked misses that row's maximum, so with non-negative per-row maxima a
     *  score lands below its exact value, never above. Refused with @c config_mismatch_k unless the
     *  shards walk by dot.
     */
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys, nk::matrix_span<f64_t> scores,
                                     std::span<std::size_t> counts) noexcept {
        std::size_t const rows_count = queries.rows;
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
        std::size_t const groups_count = rows_count / rows_per_query;
        if (counts.size() != groups_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(groups_count, keys, scores); failed(refusal))
            return {{}, refusal};
        if (!groups_count) return {{}, status_t::success_k};

        // Every token row keeps its whole ranked pool, its keys resolved and merged across the shards.
        std::size_t const width = shards_[0].index->non_zero_expansion_factor(config);
        growing_buffer<key_t, forkunion_allocator_t> row_keys {allocator_};
        growing_buffer<distance_t, forkunion_allocator_t> row_distances {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> row_counts {allocator_};
        maxsim_reducer<key_t, distance_t, forkunion_allocator_t> reduce {allocator_};
        if (failed(row_keys.resize_uninitialized(rows_count * width)) ||
            failed(row_distances.resize_uninitialized(rows_count * width)) ||
            failed(row_counts.resize_uninitialized(rows_count)) || failed(reduce.reserve(rows_per_query, width)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<key_t> const ranked_keys {row_keys.data(), rows_count, width};
        nk::matrix_span<distance_t> const ranked_distances {row_distances.data(), rows_count, width};
        expected<search_result_t> ranked = search(queries, executor, config, ranked_keys, ranked_distances,
                                                  to_writable_span(row_counts));
        if (!ranked) return ranked;
        ranked.value.count = reduce(ranked_keys, ranked_distances, to_span(row_counts), keys, scores, counts);
        return ranked;
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
     *  @param[inout] executor The threads and their scratch; one token occupies one at a time.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @return The query rows attended in @c count, or @c config_mismatch_k unless the shards walk
     *      by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, search_config_t config = {}) noexcept {
        std::size_t const rows_count = queries.rows, columns = queries.columns;
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.rows != rows_count || outputs.columns != columns || (!carry.empty() && carry.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};

        // Past the token's imported heads, their `f32` rows, then their softmax states.
        using slice_t = arena_allocator<byte_t>;
        bool const in_place = outputs.template holds<f32_t>() && outputs.stride == columns * sizeof(f32_t);
        std::size_t const heads_bytes = staged_rows_bytes(vector_dtype_, head_count, columns);
        std::size_t const accumulators_bytes = slice_t::reserved_bytes(head_count * columns * sizeof(f32_t));
        auto attend_token = [&](std::size_t token, std::size_t, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks) noexcept {
            std::size_t const first = token * head_count;
            byte_t* const accumulators = traversal.staged_queries.data() + heads_bytes;
            byte_t* const states = accumulators + accumulators_bytes;
            status_t status = status_t::success_k;
            shard_value_lookup<value_lookup_type_> shard_lookup {nullptr, value_lookup};
            attention_rescoring<capability_, scalar_t, decltype(shard_lookup)> const attention {
                shards_[0].index->import_rows_(queries, first, head_count, traversal),
                shard_lookup,
                score_scale,
                {reinterpret_cast<f32_t*>(in_place ? outputs.row(first) : accumulators), head_count, columns},
                carry.empty() ? std::span {reinterpret_cast<softmax_state_t*>(states), head_count}
                              : carry.subspan(first, head_count),
                status};
            attention.clear();
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                shard_lookup.keys = &shards_[shard].keys;
                expected<search_result_t> const walked = shards_[shard].index->walk_(
                    attention.queries.row(0).as_vector(), dummy_fold_t {}, attention, traversal, config,
                    watermarks[shard]);
                if (!walked) return walked.status;
                if (failed(status)) return status;
            }
            attention.finish();
            if (!in_place)
                for (std::size_t head = 0; head != head_count; ++head)
                    export_row<capability_>(attention.outputs.row(head).data(), nk_f32_k, outputs, first + head);
            return status_t::success_k;
        };
        std::size_t const staged_bytes = heads_bytes + accumulators_bytes + head_count * sizeof(softmax_state_t);
        expected<search_result_t> attended = for_each_row_(rows_count / head_count, executor, config, staged_bytes,
                                                           attend_token, {});
        if (attended) attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Checkpoints

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
        return this->plan_save_(path, config, nullptr);
    }

    /** Opens a save into @p monolith as one graph - the implementation the path form maps for; a
     *  manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size the bytes by
     *  @ref storage_requirements; fewer answer @c overflow_k. */
    expected<save_plan_t> plan_save(bytes_span_t monolith, save_config_t const& config) noexcept {
        return this->plan_save_(monolith, config, nullptr);
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
        return this->plan_load_(path, config, nullptr);
    }

    /** Opens a load of one monolith held in @p monolith - the implementation the path form maps
     *  for. */
    expected<load_plan_t> plan_load(bytes_view_t monolith, load_config_t const& config) noexcept {
        return this->plan_load_(monolith, config, nullptr);
    }

#pragma endregion Checkpoints

    using base_t::contains;
    using base_t::count;

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
        locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < target.rows) {
                vector_view_t const stored = shards_[locator.shard].index->vector(locator.node);
                export_row<capability_>(stored.byte_data(), vector_dtype_, target, total);
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

        void operator()(std::span<node_id_t const> nodes, std::span<vector_view_t> rows) const noexcept {
            std::array<key_t, 4> node_keys;
            for (std::size_t lane = 0; lane != nodes.size(); ++lane) node_keys[lane] = (*keys)[nodes[lane]];
            value_lookup(std::span<key_t const>(node_keys).first(nodes.size()), rows);
        }
    };

    friend base_t;
    using base_t::allocator_;
    using base_t::close_shards_;
    using base_t::commit_sequence_;
    using base_t::first_row_of_shard_;
    using base_t::kernel_;
    using base_t::locators_of_key_;
    using base_t::reopen_shards_;
    using base_t::routed_rows_;
    using base_t::rows_from_;
    using base_t::shards_;

    /** One unreserved shard placed on @p memory_domain; null when the allocation fails. */
    std::unique_ptr<index_t> build_(fu_memory_domain_id_t memory_domain) const noexcept {
        forkunion_allocator_t const allocator {memory_domain};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel_, allocator, allocator));
    }

    /** One unreserved shard on the memory domain @p like sits on. */
    std::unique_ptr<index_t> rebuild_(index_t const& like) const noexcept {
        return build_(like.vectors_allocator_.memory_domain);
    }

    /** Appends @p shard with its key column on the shard's own memory domain. */
    status_t place_(std::unique_ptr<index_t>&& shard) noexcept {
        forkunion_allocator_t const placed {shard->vectors_allocator_.memory_domain};
        return shards_.push_back(shard_t {std::move(shard), keys_column_t {placed}});
    }

    /** Builds one traversal context per thread over its own arena, so a fan-out never allocates
     *  and never shares scratch. Sized to the executor, since a dynamic fan-out picks its
     *  threads. */
    status_t reserve_traversals_(growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals,
                                 executor_t& executor, insert_config_t config) noexcept {
        std::size_t const threads = executor.size();
        if (!threads) return status_t::uninitialized_k;
        if (failed(traversals.reserve_exactly(threads))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != threads; ++thread) {
            traversal_context_t traversal {arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(thread))};
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
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        growing_buffer<std::unique_ptr<index_t>, forkunion_allocator_t> emptied {allocator_};
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
        auto [seats, seat_status] = locators_of_key_.claim(landing_rows);
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
        if (status_t const landed = drain_steps_(plans, to_writable_span(cursors), to_span(refining.homes_), executor,
                                                 0, land);
            failed(landed))
            return landed;

        // Every step landed, so no plan refuses; the bracket shows a search every shard's rows or
        // none, and the keys resolve only once their rows are visible.
        status_t published = status_t::success_k;
        atomic_ref<u64_t> const sequence {commit_sequence_};
        sequence.add(1, std::memory_order_release);
        for (typename index_t::load_plan_t& plan : plans)
            if (succeeded(published)) published = plan.finish();
        sequence.add((u64_t {1} << 32) - 1, std::memory_order_release);
        if (failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                locators_of_key_.insert(seats, shards_[into].keys[row], key_locator_t {into, row});

        // Each donor shard becomes a part of its own in the plan, which sized every shard's run.
        this->record_parts_(donor, plans, refining.part_bases_.data(), to_span(refining.part_offsets_));
        for (std::size_t from = 0; from != donors; ++from) {
            shard_t& part = donor.shards_[from];
            forkunion_allocator_t const placed {part.index->vectors_allocator_.memory_domain};
            part.index = std::move(emptied[from]);
            part.keys = keys_column_t {placed};
        }
        donor.locators_of_key_.clear();
        atomic_ref<std::size_t>(donor.routed_rows_).store(0, std::memory_order_relaxed);
        return status_t::success_k;
    }

    /** Sizes what an audit runs on: one context per thread, one cursor per shard. The reservation
     *  comes first, since it rewinds every arena, and the plans opened after it keep their
     *  bitsets. */
    status_t reserve_audit_(growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals,
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
                                 std::span<std::size_t> cursors, std::span<std::size_t const> homes,
                                 executor_t& executor) noexcept {
        for (plan_type_& plan : plans) plan.restart_sweep();
        auto sweep_step = [&](std::size_t shard, auto const& step, std::size_t, nk::matrix_span<node_id_t>) noexcept {
            return plans[shard].sweep(step), status_t::success_k;
        };
        for (std::size_t expanded = 1; expanded;) {
            std::memset(cursors.data(), 0, cursors.size_bytes());
            if (status_t const drained = drain_steps_(plans, cursors, homes, executor, 0, sweep_step); failed(drained))
                return drained;
            expanded = 0;
            for (plan_type_& plan : plans) expanded += plan.advance_level();
        }
        return status_t::success_k;
    }

#pragma endregion Health

#pragma region Search

    /** Takes every shard's watermark into @p watermarks, unless it is empty, at one instant no batch
     *  spanning shards publishes through, so a walk of each shard below its own sees every batch whole
     *  or not at all; answers their sum. */
    std::size_t snapshot_(std::span<node_id_t> watermarks) const noexcept {
        atomic_ref<u64_t const> const sequence {commit_sequence_};
        while (true) {
            u64_t const before = sequence.load(std::memory_order_acquire);
            // A set low half is a publication between its shards' stores.
            if (static_cast<u32_t>(before)) {
                wait_type_ {}(&commit_sequence_, before, 0u, fu::wait_capped_t {});
                continue;
            }
            std::size_t total = 0;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                std::size_t const visible = shards_[shard].index->size();
                if (!watermarks.empty()) watermarks[shard] = static_cast<node_id_t>(visible);
                total += visible;
            }
            if (sequence.load(std::memory_order_relaxed) == before) return total;
        }
    }

    /** Takes the snapshot into @p traversal 's staged IDs, idle during any search and sized once per
     *  context as an insertion sizes them, so a read allocates nothing per call; empty when they cannot
     *  hold one watermark per shard. */
    std::span<node_id_t const> snapshot_(traversal_context_t& traversal) const noexcept {
        if (failed(traversal.reserve_neighbors(max_of(connectivity(), shards_.size()))) ||
            nk::failed(traversal.staged_ids.resize({shards_.size()})))
            return {};
        std::span<node_id_t> const watermarks {traversal.staged_ids.data(), shards_.size()};
        snapshot_(watermarks);
        return watermarks;
    }

    /**
     *  @brief The scan behind @ref search_if: every row published when a query's block starts is
     *      scored, so the rankings are the ones a graph walk approximates.
     *
     *  Each block of queries takes a thread and its traversal, scans every shard below one snapshot
     *  of the watermarks, and lands each row beating a query's worst match in its output row
     *  through the shard's key column. A block as wide as one tile, on a platform with a matrix
     *  engine for the scalar, streams through it; a single query scores four rows at a time. Either
     *  way the shards' own kit finalizes the distances, so they match the walk's.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> scan_rows_(rows_view_t queries, executor_t& executor, nk::matrix_span<key_t> keys,
                                         nk::matrix_span<score_type_> distances, std::span<std::size_t> counts,
                                         predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows, wanted = keys.extent(1);
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        std::memset(counts.data(), 0, counts.size_bytes());
        if (!queries_count || !wanted) return {{}, status_t::success_k};

        std::size_t const width = index_t::scan_width(queries_count, executor.size());
        auto scan_block = [&](std::size_t block, std::size_t, traversal_context_t& traversal,
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
                    scanned = one.index->scan_(block_queries, dummy_predicate_t {}, land, traversal, watermarks[shard]);
                else
                    scanned = one.index->scan_(block_queries, keyed_admission {predicate, one.keys}, land, traversal,
                                               watermarks[shard]);
                if (failed(scanned)) return scanned;
            }
            return status_t::success_k;
        };
        // Every shard's scan sizes the staging its block imports and packs into.
        return for_each_row_(divide_round_up(queries_count, width), executor, search_config_t {}, 0, scan_block,
                             counts);
    }

    /** The batched @ref search_if under @p heads, which only a @c dummy_predicate_t accompanies. */
    template <typename score_type_, typename heads_type_, typename predicate_type_>
    expected<search_result_t> search_rows_(rows_view_t queries, heads_type_ const& heads, executor_t& executor,
                                           search_config_t config, nk::matrix_span<key_t> keys,
                                           nk::matrix_span<score_type_> distances, std::span<std::size_t> counts,
                                           predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows;
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};

        auto search_row = [&](std::size_t query, std::size_t, traversal_context_t& traversal,
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
                                             indexer_rescoring<capability_, scalar_t> {heads.row(query)}, traversal,
                                             config, watermarks[shard]);
                else if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    found = one.index->walk_(query_vector, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config,
                                             watermarks[shard]);
                else {
                    keyed_admission keyed {predicate, one.keys};
                    found = one.index->walk_if_(query_vector, keyed, traversal, config, watermarks[shard]);
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
        return for_each_row_(queries_count, executor, config, staged_rows_bytes(vector_dtype_, 1, queries.columns),
                             search_row, counts);
    }

    /** Claims @p executor, sizes one traversal per thread with @p staged_bytes of staged queries,
     *  runs @p per_row over every row with its thread, that thread's traversal and a fresh snapshot
     *  of the watermarks, and sums the threads' telemetry & the rows' @p counts; a row that refuses
     *  stops itself, and the first refusal is the one kept. */
    template <typename per_row_type_>
    expected<search_result_t> for_each_row_(std::size_t rows_count, executor_t& executor, search_config_t config,
                                            std::size_t staged_bytes, per_row_type_&& per_row,
                                            std::span<std::size_t const> counts) noexcept {
        // One caller per executor, and the arenas grow to the widest geometry ever asked of them.
        // Only `scan_rows_` scans, and it drives the rows under a walk's sizing.
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (status_t const sized = executor.reserve(
                shards_[0].index->traversal_memory_usage(insert_config_t {config.expansion_factor}));
            failed(sized))
            return {{}, sized};

        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, insert_config_t {config.expansion_factor});
            failed(ready))
            return {{}, ready};
        for (traversal_context_t& traversal : traversals)
            if (failed(traversal.reserve_queries(staged_bytes))) return {{}, status_t::bad_alloc_k};

        status_t first_refusal = status_t::success_k;
        auto each_row = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            traversal_context_t& traversal = traversals[thread];
            std::span<node_id_t const> const watermarks = snapshot_(traversal);
            status_t const refusal = watermarks.size() == shards_.size() ? per_row(row, thread, traversal, watermarks)
                                                                         : status_t::bad_alloc_k;
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

    /** Whether the index can write @p config: placed, reserved, tokenized, and legal; the steps
     *  claim the executor as they fan out, so none rides here. */
    status_t saveable_(save_config_t const& config, executor_t*) const noexcept {
        static_assert(base_t::key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (base_t::value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        return config.validate();
    }

    /** A host save settles instead of closing the dispensers: a row landing mid-save under
     *  @c concurrent_k reports @c contended_k from its shard. */
    static status_t gate_save_(save_plan_t&, executor_t*) noexcept { return status_t::success_k; }

    /** Ends a save @ref gate_save_ opened, which closed nothing. */
    static void end_save_() noexcept {}

    /** Whether the index can take a checkpoint: placed, and holding nothing yet once every shard's
     *  dispenser is closed for @p plan, whose death reopens them. */
    status_t gate_load_(load_plan_t& plan, executor_t*) noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (status_t const closed = close_shards_(); failed(closed)) return closed;
        plan.owner_ = this;
        return size() ? status_t::config_mismatch_k : status_t::success_k;
    }

    /** One slab of neighbor IDs from @p arena - the scratch a save chunk rebases through; empty
     *  when the arena is short. */
    nk::matrix_span<node_id_t> scratch_of_(typename executor_t::arena_t& arena,
                                           std::size_t scratch_rows) const noexcept {
        std::size_t const connectivity = this->connectivity();
        std::byte* bytes = arena.allocate(scratch_rows * connectivity * sizeof(node_id_t), alignof(node_id_t));
        if (!bytes) return {};
        return {reinterpret_cast<node_id_t*>(bytes), scratch_rows, connectivity};
    }

    /** Picks every thread's home shard into @p homes, once per plan: the first shard on the
     *  thread's memory domain, so its drains start there, or a spread over the shards when none
     *  sits there. */
    status_t pick_home_shards_(growing_buffer<std::size_t, forkunion_allocator_t>& homes,
                               executor_t const& executor) const noexcept {
        std::size_t const shards = shards_.size();
        if (failed(homes.resize(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread) {
            std::size_t shard = 0;
            while (shard != shards &&
                   shards_[shard].index->vectors_allocator_.memory_domain != executor.memory_domain(thread))
                ++shard;
            homes[thread] = shard != shards ? shard : thread % shards;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Fans every plan's chunks out over the executor's threads: each thread drains its
     *      home shard per @p homes first, then helps the rest, one atomic cursor per shard being
     *      the whole schedule.
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
                          std::span<std::size_t const> homes, executor_t& executor, std::size_t scratch_rows,
                          per_step_type_& per_step) noexcept {
        std::size_t const shards = plans.size();
        if (cursors.size() != shards || homes.size() != executor.size()) return status_t::config_mismatch_k;
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
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (homes[thread] + visited) % shards;
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
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> homes {allocator_};
        if (failed(cursors.resize(shards)) || failed(pick_home_shards_(homes, executor))) return status_t::bad_alloc_k;
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
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (homes[thread] + visited) % shards;
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

    /** Held while a planned batch dispenses its shards' ranges, so no two batches' ranges cross
     *  order between the shards and each can wait for its turns outside the commit bracket. */
    fu::spin_mutex_t dispensing_;
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
                export_row<indexes_t::capability_k>(shard.index->vector(node_).byte_data(), indexes_t::vector_dtype_k,
                                                    target, written);
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
