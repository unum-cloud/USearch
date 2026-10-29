/**
 *  @file include/usearch/dense/ram_compaction.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The freestanding compaction of @c ram_indexes sources into a RAM target under a shard
 *      capacity, in place when the target is one of the sources.
 *
 *  A compaction packs the source shards largest first into target shards of at most a shard
 *  capacity of rows, each into the first with room, so it ends with as few shards as fit; an
 *  absorb packs the same way into the shard with the fewest rows. The shard a bin keeps stays
 *  resident with its IDs, every other source shard lands after it with its edges rebased and its
 *  keys beside it, and only the appended nodes walk the target across the parts and link in, then
 *  each target shard heals. The landings and the walk are the absorb plan's own phases.
 */
#pragma once

#include <span>    // `std::span`
#include <utility> // `std::exchange`

#include "usearch/dense/types.hpp" // `compact_config_t`, `compact_result_t`
#include "usearch/dense/ram_indexes.hpp"

namespace unum::usearch::dense {

/**
 *  @brief Opens a compaction of @p sources into @p target, each target shard holding at most
 *      @p config.shard_capacity rows.
 *
 *  When @p target is one of @p sources, it compacts in place: each bin's largest shard keeps its
 *  IDs and the rest land after it as parts, so no shard's rows exist twice; otherwise every row
 *  moves and the sources end empty.
 *
 *  The target and every other source close here, each under its own gate, so the packing never
 *  goes stale: the target stays closed while the plan lives and each source until it lands, and
 *  insertions into any of them answer @c contended_k meanwhile. Searches and key lookups of every
 *  index run throughout, retrying once around the windows that move rows: in place, every moved
 *  row's one locator swaps to its landing inside the window that forgets its old shard. Emptied
 *  shards stay as idle slots on their placement for the target's next shard generation, and the
 *  source slots must outlive the plan.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k A null or repeated source, another metric or geometry.
 *  - @c contended_k A call or batch in flight on the target or a source, or another refine,
 *    absorb, split, compaction or load holding either.
 *  - @c unsupported_k A source shard past the capacity, which would have to be cut first.
 */
template <typename indexes_type_>
expected<ram_compact_plan<indexes_type_>> plan_compact(std::span<indexes_type_* const> sources, indexes_type_& target,
                                                       typename indexes_type_::executor_t& executor,
                                                       compact_config_t const& config = {}) noexcept;

/**
 *  The one-shot @ref unum::usearch::dense::plan_compact, run and finished over @p executor on
 *  contexts carved for the call; @p on_visited reports and cancels the re-selection alone, as in
 *  @c ram_compact_plan::run.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source, no target shard, or @p executor holds no threads, pool or
 *    arenas.
 *  - @c config_mismatch_k A null or repeated source, another kernel or geometry, or a landing
 *    step meeting rows outside its plan.
 *  - @c contended_k A call or plan holds the target or a source, a source stopped emptying,
 *    another call holds @p executor, or its pool runs another dispatch.
 *  - @c unknown_kind_k An empty target reserving for the sources' geometry cannot finalize the
 *    kernel.
 *  - @c unsupported_k A source shard is past the capacity.
 *  - @c overflow_k No shard takes a landing, a landing passes a shard's rows, key column or the
 *    locator map, or a queue capacity wraps.
 */
template <typename indexes_type_, typename progress_type_ = dummy_progress_t>
expected<compact_result_t> compact(std::span<indexes_type_* const> sources, indexes_type_& target,
                                   typename indexes_type_::executor_t& executor, compact_config_t const& config = {},
                                   progress_type_&& on_visited = {}) noexcept;

/**
 *  @brief One compaction in flight into a RAM target, holding its dispensers closed meanwhile.
 *
 *  @ref run lands the sources still pending, the target's own shards first when it compacts in
 *  place, each source staying empty, closed and sealed once it landed, then drains the re-selection
 *  of the appended nodes and the heal; a cancel pauses only that walk, and a later call continues
 *  it.
 *  @ref finish heals what a stopped walk leaves, reopens the target and every source still pending,
 *  and reports.
 */
template <typename indexes_type_>
struct [[nodiscard]] ram_compact_plan {
    using indexes_t = indexes_type_;
    using executor_t = typename indexes_t::executor_t;
    using traversal_context_t = typename indexes_t::traversal_context_t;
    using refine_plan_t = typename indexes_t::refine_plan_t;
    using result_t = compact_result_t;

    ram_compact_plan() noexcept = default;
    ram_compact_plan(ram_compact_plan&& other) noexcept
        : refining_(std::move(other.refining_)), sources_(std::exchange(other.sources_, {})),
          resident_shards_(other.resident_shards_), landed_(other.landed_), moved_entries_(other.moved_entries_) {}
    ram_compact_plan& operator=(ram_compact_plan&&) = delete;
    ram_compact_plan(ram_compact_plan const&) = delete;
    ram_compact_plan& operator=(ram_compact_plan const&) = delete;

    /** Reopens every source still pending; the refine plan reopens the target. */
    ~ram_compact_plan() noexcept { release_sources_(); }

    /**
     *  @brief Lands the pending sources on @p executor, then runs the re-selection until it ends
     *      or @p on_visited cancels; @c cancelled_k means a later call continues exactly there.
     *
     *  A source failing to land stays pending and closed, with every later one, for the next call.
     *
     *  @param[in,out] executor The executor the plan was opened over.
     *  @param[in,out] traversals The caller's context of every executor thread, which the walks run
     *      on; the call sizes them.
     *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
     *      population; a @c cancel_k answer stops the drain where its steps ran.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is bound to no target, or @p executor holds no pool or no
     *    workers.
     *  - @c config_mismatch_k @p traversals holds other than one context per thread, or a landing
     *    step meets rows outside its plan.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or a
     *    source stopped emptying under the landing.
     *  - @c overflow_k A landing passes a shard's rows, its key column or the locator map, or a
     *    queue capacity wraps when rounded.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t run(executor_t& executor, std::span<traversal_context_t> traversals,
                 progress_type_&& on_visited = {}) noexcept {
        indexes_t* const target = refining_.owner_;
        if (!target) return status_t::uninitialized_k;
        if (executor.size() != traversals.size()) return status_t::config_mismatch_k;
        if (status_t const landed = land_(*target, executor); failed(landed)) return landed;
        return refining_.run(executor, traversals, on_visited);
    }

    /**
     *  Stops any re-selection, heals on the caller's @p traversals, one per executor thread,
     *  reopens the target and reports. A plan whose sources have not all landed heals nothing.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is bound to no target, or @p executor holds no pool or no
     *    workers.
     *  - @c config_mismatch_k @p traversals holds other than one context per thread.
     *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    expected<result_t> finish(executor_t& executor, std::span<traversal_context_t> traversals) noexcept {
        result_t result;
        indexes_t* const target = refining_.owner_;
        if (!target) return {result, status_t::uninitialized_k};
        result.source_shards = refining_.landings_.size();
        result.moved_entries = moved_entries_;
        // Before the refine plan unbinds the target, which tells the target among the sources.
        release_sources_();
        auto [graph, status] = refining_.finish(executor, traversals);
        result.graph = graph;
        for (std::size_t shard = 0; shard != target->shards_count(); ++shard)
            if (target->shard(shard).size()) ++result.target_shards;
        return {result, status};
    }

  private:
    template <typename other_type_>
    friend expected<ram_compact_plan<other_type_>> plan_compact(std::span<other_type_* const>, other_type_&,
                                                                typename other_type_::executor_t&,
                                                                compact_config_t const&) noexcept;
    template <typename other_type_, typename progress_type_>
    friend expected<compact_result_t> compact(std::span<other_type_* const>, other_type_&,
                                              typename other_type_::executor_t&, compact_config_t const&,
                                              progress_type_&&) noexcept;

    explicit ram_compact_plan(forkunion_allocator_t allocator) noexcept : refining_(allocator) {}

    /** Validates @p sources against @p target, closes the target and every other source, each
     *  under its own gate, counts their rows and checks their geometry once no batch can land, and
     *  packs every source shard into the target under the tighter of @p config.shard_capacity and
     *  the target's own bound. */
    static expected<ram_compact_plan> open_(std::span<indexes_t* const> sources, indexes_t& target,
                                            executor_t const& executor, compact_config_t const& config) noexcept {
        if (sources.empty() || !target.shards_count() || !executor.size()) return {{}, status_t::uninitialized_k};
        for (std::size_t position = 0; position != sources.size(); ++position) {
            indexes_t const* source = sources[position];
            if (!source || source->kernel() != target.kernel()) return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != position; ++earlier)
                if (sources[earlier] == source) return {{}, status_t::config_mismatch_k};
        }
        // Zero, or anything past the target's own bound, means that bound.
        std::size_t const capacity = config.shard_capacity && config.shard_capacity < target.shard_capacity()
                                         ? config.shard_capacity
                                         : target.shard_capacity();

        ram_compact_plan plan {target.shards_.allocator()};
        refine_plan_t& refining = plan.refining_;
        if (status_t const closed = target.close_gated_(); failed(closed)) return {{}, closed};
        refining.owner_ = &target;
        refining.config_.expansion_factor = config.expansion_factor;
        for (std::size_t closing = 0; closing != sources.size(); ++closing) {
            if (sources[closing] == &target) continue;
            if (status_t const closed = sources[closing]->close_gated_(); failed(closed)) {
                while (closing)
                    if (sources[--closing] != &target) sources[closing]->shards_.reopen();
                return {{}, closed};
            }
        }
        plan.sources_ = sources;

        // Closed, no batch lands, so the rows counted here are the rows that move; a refusal from
        // here reopens everything as the plan dies.
        indexes_t const* shaped = target.capacity() ? &target : nullptr;
        std::size_t resident_shards = 0, moved_entries = 0;
        for (indexes_t const* source : sources) {
            if (source == &target) resident_shards = target.shards_count();
            moved_entries += source->size();
            if (!source->capacity()) continue;
            if (!shaped) shaped = source;
            else if (!same_geometry(*source, *shaped)) return {{}, status_t::config_mismatch_k};
        }
        plan.resident_shards_ = resident_shards;
        if (status_t const packed = target.plan_landings_(sources, capacity, rank_first_fit, refining); failed(packed))
            return {{}, packed};
        std::size_t const shards = refining.part_offsets_.size() - 1;
        if (failed(refining.shard_plans_.reserve_exactly(shards)) || failed(refining.cursors_.resize(shards)))
            return {{}, status_t::bad_alloc_k};
        // Every source row moves but those of the target's own shards that stay where they are.
        for (std::size_t shard = 0; shard != resident_shards; ++shard)
            if (refining.landings_[shard].into == shard) moved_entries -= target.shard(shard).size();
        plan.moved_entries_ = moved_entries;
        return {std::move(plan), status_t::success_k};
    }

    /** Lands every pending source into @p target under the executor's claim: in place, the target's
     *  own shards first, vacating the shards other sources may land in. */
    status_t land_(indexes_t& target, executor_t& executor) noexcept {
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        std::size_t const lead = resident_shards_ ? 1 : 0;
        if (lead && !landed_) {
            std::span<landing_t const> const own = to_span(refining_.landings_).first(resident_shards_);
            if (status_t const landed = target.absorb_(target, own, executor, refining_); failed(landed)) return landed;
            landed_ = 1;
        }
        for (; landed_ - lead != sources_.size(); ++landed_) {
            indexes_t* const source = sources_[landed_ - lead];
            if (source == &target) continue;
            if (status_t const landed = target.absorb_(
                    *source, refining_.landings_of_(sources_.subspan(landed_ - lead)), executor, refining_);
                failed(landed))
                return landed;
        }
        return status_t::success_k;
    }

    /** Reopens every other source still pending, each closed at planning; the plan holds none
     *  after. */
    void release_sources_() noexcept {
        std::size_t const lead = resident_shards_ ? 1 : 0;
        indexes_t* const target = refining_.owner_;
        for (std::size_t pending = landed_ > lead ? landed_ - lead : 0; pending < sources_.size(); ++pending)
            if (sources_[pending] != target) sources_[pending]->shards_.reopen();
        sources_ = {};
    }

    /** Runs and finishes the plan on contexts carved for the call: the one-shot @ref compact. */
    template <typename progress_type_>
    expected<result_t> drive_(executor_t& executor, progress_type_& on_visited) noexcept {
        typename indexes_t::carved_traversals_t traversals {
            arena_allocator<traversal_context_t, forkunion_allocator_t>(executor.arena(0))};
        if (status_t const carved = refining_.owner_->carve_traversals_(traversals, executor); failed(carved))
            return {{}, carved};
        status_t const ran = run(executor, to_writable_span(traversals), on_visited);
        auto const [result, finish_status] = finish(executor, to_writable_span(traversals));
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /** The absorb plan over the target: its landings, part runs and per-shard walks. */
    refine_plan_t refining_;

    /** The caller's source slots, borrowed until the plan dies. */
    std::span<indexes_t* const> sources_;

    /** The target's shards when it is one of the sources, landing first; zero otherwise. */
    std::size_t resident_shards_ = 0;

    /** Sources landed so far, the target's own shards counting first when in place. */
    std::size_t landed_ = 0;

    /** Source rows that change shard. */
    std::size_t moved_entries_ = 0;
};

template <typename indexes_type_>
expected<ram_compact_plan<indexes_type_>> plan_compact(std::span<indexes_type_* const> sources, indexes_type_& target,
                                                       typename indexes_type_::executor_t& executor,
                                                       compact_config_t const& config) noexcept {
    return ram_compact_plan<indexes_type_>::open_(sources, target, executor, config);
}

template <typename indexes_type_, typename progress_type_>
expected<compact_result_t> compact(std::span<indexes_type_* const> sources, indexes_type_& target,
                                   typename indexes_type_::executor_t& executor, compact_config_t const& config,
                                   progress_type_&& on_visited) noexcept {
    auto [plan, status] = plan_compact(sources, target, executor, config);
    if (failed(status)) return {{}, status};
    return plan.drive_(executor, on_visited);
}

} // namespace unum::usearch::dense
