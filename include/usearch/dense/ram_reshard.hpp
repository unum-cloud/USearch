/**
 *  @file include/usearch/dense/ram_reshard.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The freestanding reshard of @c ram_indexes sources into RAM targets: many sources packed
 *      into one target, or one source split into many targets.
 *
 *  A reshard into one target packs every moved source shard into a target shard: a zero shard
 *  capacity keeps the target's shards and appends each moved shard to the one with the fewest rows,
 *  while a capacity repacks every source shard, the target's own among them when it is a source,
 *  largest first into the first shard with room, so it ends with as few shards as fit. The shard a
 *  bin keeps stays resident with its IDs, every other source shard moves in after it with its edges
 *  rebased and its keys beside it, and then one round per target shard links them in. A reshard
 *  into many targets splits shard @c s of its one source into shard @c s of every target, one
 *  graph-local part each, and then every target shard runs one round. The moves and the rounds are
 *  the plan's own phases.
 */
#pragma once

#include <cstring> // `std::memset`

#include <span>    // `std::span`
#include <utility> // `std::exchange`

#include "usearch/dense/types.hpp" // `reshard_config_t`, `reshard_report_t`
#include "usearch/dense/ram_indexes.hpp"

namespace unum::usearch::dense {

/**
 *  @brief Opens a reshard of @p sources into @p targets: several sources into one target, or one
 *      source into several targets.
 *
 *  Into one target, a zero @p config.shard_capacity keeps the target's shards, the target's own
 *  rows staying where they are when it is a source, and every other source's shards move in beside
 *  them; a capacity repacks every source shard into target shards of at most that many rows, in
 *  place when the target is a source, so no shard's rows exist twice. Into several targets, the
 *  one source splits shard by shard into the same shard of every target, which must hold as many
 *  shards and no rows, and the source ends empty.
 *
 *  The plan copies both lists, and every index it reads or moves closes here, each under its own
 *  gate, so the packing never goes stale: the target or targets stay closed while the plan lives
 *  and each source until it moves out, and insertions into any of them answer @c contended_k
 *  meanwhile. Searches and key lookups of every index run throughout, retrying once around the
 *  windows that move rows. Emptied shards stay as idle slots on their placement.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No source, no target, or a null or repeated index in either list.
 *  - @c unsupported_k Several sources and several targets, or a source shard past the capacity,
 *    which would have to be cut first.
 *  - @c uninitialized_k No target shard, or @p executor has no threads.
 *  - @c config_mismatch_k Another metric or geometry, a split's source among its targets, or a
 *    split target holding rows or another shard count.
 *  - @c contended_k A call, batch, plan or load in flight on a target or on a source.
 */
template <typename indexes_type_>
expected<ram_reshard_plan<indexes_type_>> plan_reshard(std::span<indexes_type_* const> sources,
                                                       std::span<indexes_type_* const> targets,
                                                       typename indexes_type_::executor_t& executor,
                                                       reshard_config_t const& config = {}) noexcept;

/**
 *  @brief The one-shot @ref unum::usearch::dense::plan_reshard, run and finished over @p executor
 *      on contexts carved for the call; @p on_visited hears and cancels the rounds, as in
 *      @c ram_reshard_plan::run.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k As @ref plan_reshard refuses.
 *  - @c unsupported_k As @ref plan_reshard refuses.
 *  - @c uninitialized_k No target shard, or @p executor holds no threads, pool or arenas.
 *  - @c config_mismatch_k As @ref plan_reshard refuses, or a move step meeting rows outside its
 *    plan.
 *  - @c contended_k A call or plan holds a target or a source, a source stopped emptying, another
 *    call holds @p executor, or its pool runs another dispatch.
 *  - @c overflow_k No shard takes a move, a move passes a shard's rows, or a queue capacity
 *    wraps.
 */
template <typename indexes_type_, typename progress_type_ = dummy_progress_t>
expected<reshard_report_t> reshard(std::span<indexes_type_* const> sources, std::span<indexes_type_* const> targets,
                                   typename indexes_type_::executor_t& executor, reshard_config_t const& config = {},
                                   progress_type_&& on_visited = {}) noexcept;

/**
 *  @brief One reshard in flight between RAM indexes, holding their dispensers closed meanwhile.
 *
 *  @ref run moves out the sources still pending, then drains one round per target shard; a cancel
 *  pauses the rounds, and a later call continues them. @ref finish runs the rounds to their end,
 *  reopens every index the plan closed, and reports.
 */
template <typename indexes_type_>
struct [[nodiscard]] ram_reshard_plan {
    using indexes_t = indexes_type_;
    using index_t = typename indexes_t::index_t;
    using executor_t = typename indexes_t::executor_t;
    using traversal_context_t = typename indexes_t::traversal_context_t;
    using refine_plan_t = typename indexes_t::refine_plan_t;
    using shard_plan_t = typename index_t::reshard_plan_t;

    ram_reshard_plan() noexcept : ram_reshard_plan(forkunion_allocator_t {}) {}
    ram_reshard_plan(ram_reshard_plan&& other) noexcept
        : sources_(std::move(other.sources_)), targets_(std::move(other.targets_)), rounds_(std::move(other.rounds_)),
          moves_(std::move(other.moves_)), resident_shards_(other.resident_shards_),
          moved_sources_(other.moved_sources_), shard_plans_(std::move(other.shard_plans_)),
          cursors_(std::move(other.cursors_)), closed_targets_(std::exchange(other.closed_targets_, 0)),
          splitting_(std::exchange(other.splitting_, nullptr)), config_(other.config_),
          source_shards_(other.source_shards_), moved_entries_(other.moved_entries_) {}
    ram_reshard_plan& operator=(ram_reshard_plan&&) = delete;
    ram_reshard_plan(ram_reshard_plan const&) = delete;
    ram_reshard_plan& operator=(ram_reshard_plan const&) = delete;

    /** Reopens every index still closed; the graph is as the last task left it. */
    ~ram_reshard_plan() noexcept { release_(); }

    /**
     *  @brief Moves out the pending sources on @p executor, then runs the rounds until they end or
     *      @p on_visited cancels; @c cancelled_k means a later call continues exactly there.
     *
     *  A source failing to move in stays pending and closed, and so does every later one, for the
     *  next call to retry.
     *
     *  @param[in,out] executor The executor the plan was opened over.
     *  @param[in,out] traversals The caller's context of every executor thread, which the tasks run
     *      on; the call sizes them.
     *  @param[in] on_visited Fires at every barrier of the rounds with the units of the phase it
     *      closed, done and in all; a @c cancel_k answer stops the drain where its tasks ran.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is bound to nothing, or @p executor holds no pool or no
     *    workers.
     *  - @c config_mismatch_k @p traversals holds other than one context per thread, or a move
     *    step meets rows outside its plan.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or a
     *    source stopped emptying under the move.
     *  - @c overflow_k A move passes a shard's rows, or a queue capacity wraps when rounded.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t run(executor_t& executor, std::span<traversal_context_t> traversals,
                 progress_type_&& on_visited = {}) noexcept {
        if (splits_()) return run_split_(executor, traversals, on_visited);
        indexes_t* const target = rounds_.owner_;
        if (!target) return status_t::uninitialized_k;
        if (executor.size() != traversals.size()) return status_t::config_mismatch_k;
        if (status_t const moved = move_in_(*target, executor); failed(moved)) return moved;
        return rounds_.run(executor, traversals, on_visited);
    }

    /**
     *  @brief Runs the rounds to their end on the caller's @p traversals, one per executor thread,
     *      reopens every index the plan closed and reports; a plan whose sources have not all
     *      moved in runs no round.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is bound to nothing, or @p executor holds no pool or no
     *    workers.
     *  - @c config_mismatch_k @p traversals holds other than one context per thread.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or the
     *    split's source stayed held under its emptying window.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    expected<reshard_report_t> finish(executor_t& executor, std::span<traversal_context_t> traversals) noexcept {
        reshard_report_t result;
        result.source_shards = source_shards_;
        result.moved_entries = moved_entries_;
        if (splits_()) {
            if (!splitting_) return {result, status_t::uninitialized_k};
            if (executor.size() != traversals.size()) return {result, status_t::config_mismatch_k};
            status_t closed = status_t::success_k;
            {
                auto [executor_claim, claim_status] = executor.claim();
                if (failed(claim_status)) return {result, claim_status};
                dummy_progress_t closing_alone;
                closed = reserve(traversals);
                if (succeeded(closed)) closed = drain_split_(executor, traversals, closing_alone);
            }
            result.graph = running();
            result.target_shards = count_target_shards_();
            release_();
            return {result, closed};
        }
        indexes_t* const target = rounds_.owner_;
        if (!target) return {result, status_t::uninitialized_k};
        // Before the rounds unbind the target, which tells the target among the sources.
        release_sources_();
        auto [graph, status] = rounds_.close_(executor, traversals);
        result.graph = graph;
        result.target_shards = count_target_shards_();
        return {result, status};
    }

    /** The totals so far, summed over every target shard's rounds. */
    refine_report_t running() const noexcept {
        if (!splits_()) return rounds_.running();
        refine_report_t result;
        for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
            if (shard) result += shard_plans_[shard].finish().value;
            else result = shard_plans_[shard].finish().value;
        return result;
    }

    /** The phase the plan is in: the moves', then the one the first shard still rounding is in. */
    phase_t phase() const noexcept {
        if (!splits_()) return pending_sources_() ? phase_t::copy_k : rounds_.phase();
        for (shard_plan_t const& plan : shard_plans_)
            if (plan.moving() || plan.rounding()) return plan.phase();
        return shard_plans_.empty() ? phase_t::none_k : shard_plans_[0].phase();
    }

    /** Where the rounds' phase sits among a round's six; zero in the moves and once every target
     *  shard's round ended. */
    std::size_t ordinal() const noexcept {
        if (!splits_()) return rounds_.ordinal();
        for (shard_plan_t const& plan : shard_plans_)
            if (plan.rounding()) return plan.ordinal();
        return 0;
    }

    /** Units of that phase done and in all, summed over the target shards. */
    std::size_t completed() const noexcept {
        if (!splits_()) return rounds_.completed();
        std::size_t sum = 0;
        for (shard_plan_t const& plan : shard_plans_) sum += plan.completed();
        return sum;
    }
    std::size_t total() const noexcept {
        if (!splits_()) return rounds_.total();
        std::size_t sum = 0;
        for (shard_plan_t const& plan : shard_plans_) sum += plan.total();
        return sum;
    }

    /** Opens every target shard's next round once each one's ended; a no-op in the moves and while
     *  any target shard is mid-round. */
    void next_round() noexcept {
        if (!splits_()) return rounds_.next_round();
        for (shard_plan_t const& plan : shard_plans_)
            if (plan.moving() || plan.rounding()) return;
        for (shard_plan_t& plan : shard_plans_) plan.next_round();
        std::memset(cursors_.data(), 0, cursors_.size() * sizeof(std::size_t));
    }

    /** Sizes the caller's @p traversals for every target shard's round. */
    status_t reserve(std::span<traversal_context_t> traversals) const noexcept {
        if (!splits_()) return rounds_.reserve(traversals);
        // The targets keep the geometry once the source empties.
        if (status_t const sized = targets_[0]->size_traversals_(traversals, insert_config_t {config_.expansion_factor},
                                                                 0);
            failed(sized))
            return sized;
        for (shard_plan_t const& plan : shard_plans_)
            for (traversal_context_t& traversal : traversals)
                if (status_t const reserved = plan.reserve(traversal); failed(reserved)) return reserved;
        return status_t::success_k;
    }

    /** Whether the rows of @p index moved out and left it empty: a source other than the target
     *  once it moved in, or a split's source once its parts published. */
    bool moved(indexes_t const* index) const noexcept {
        if (splits_()) return index == sources_[0] && !shard_plans_.empty() && !shard_plans_[0].moving();
        std::size_t const lead = resident_shards_ ? 1 : 0, moved = moved_sources_ > lead ? moved_sources_ - lead : 0;
        for (std::size_t position = 0; position != moved && position != sources_.size(); ++position)
            if (sources_[position] == index) return true;
        return false;
    }

  private:
    using insert_config_t = typename indexes_t::insert_config_t;
    using refine_config_t = typename indexes_t::refine_config_t;
    using unique_lock_t = typename indexes_t::unique_lock_t;

    template <typename other_type_>
    friend expected<ram_reshard_plan<other_type_>> plan_reshard(std::span<other_type_* const>,
                                                                std::span<other_type_* const>,
                                                                typename other_type_::executor_t&,
                                                                reshard_config_t const&) noexcept;
    template <typename other_type_, typename progress_type_>
    friend expected<reshard_report_t> reshard(std::span<other_type_* const>, std::span<other_type_* const>,
                                              typename other_type_::executor_t&, reshard_config_t const&,
                                              progress_type_&&) noexcept;

    explicit ram_reshard_plan(forkunion_allocator_t allocator) noexcept
        : sources_(allocator), targets_(allocator), rounds_(allocator), moves_(allocator), shard_plans_(allocator),
          cursors_(allocator) {}

    /** Validates both lists and opens the reshard they describe. */
    static expected<ram_reshard_plan> open_(std::span<indexes_t* const> sources, std::span<indexes_t* const> targets,
                                            executor_t const& executor, reshard_config_t const& config) noexcept {
        if (sources.empty() || targets.empty()) return {{}, status_t::invalid_argument_k};
        for (std::span<indexes_t* const> const listed : {sources, targets})
            for (std::size_t position = 0; position != listed.size(); ++position) {
                if (!listed[position]) return {{}, status_t::invalid_argument_k};
                for (std::size_t earlier = 0; earlier != position; ++earlier)
                    if (listed[earlier] == listed[position]) return {{}, status_t::invalid_argument_k};
            }
        if (sources.size() > 1 && targets.size() > 1) return {{}, status_t::unsupported_k};
        if (!executor.size()) return {{}, status_t::uninitialized_k};
        indexes_t const& shaped = *targets.front();
        for (std::span<indexes_t* const> const listed : {sources, targets})
            for (indexes_t const* index : listed)
                if (index->kernel() != shaped.kernel() || index->geometry() != shaped.geometry())
                    return {{}, status_t::config_mismatch_k};
        if (targets.size() > 1) return open_split_(*sources.front(), targets, executor, config);
        return open_pack_(sources, *targets.front(), executor, config);
    }

    /** Closes @p target and every other source, each under its own gate, counts their rows once
     *  no batch can insert, and packs the source shards into the target's, as @ref plan_reshard
     *  describes for a pack. */
    static expected<ram_reshard_plan> open_pack_(std::span<indexes_t* const> sources, indexes_t& target,
                                                 executor_t const& executor, reshard_config_t const& config) noexcept {
        if (!target.shards_count()) return {{}, status_t::uninitialized_k};
        // A capacity repacks every source shard, the target's own among them; none spreads the
        // other sources across the target's shards, which stay as they are.
        bool const repacking = config.shard_capacity != 0;
        std::size_t const capacity = repacking && config.shard_capacity < target.shard_capacity()
                                         ? config.shard_capacity
                                         : target.shard_capacity();

        ram_reshard_plan plan {target.shards_.allocator()};
        if (failed(plan.sources_.reserve_exactly(sources.size())) || failed(plan.targets_.reserve_exactly(1)))
            return {{}, status_t::bad_alloc_k};
        plan.targets_.push_back_reserved(&target);
        refine_plan_t& rounds = plan.rounds_;
        if (status_t const closed = target.close_gated_(); failed(closed)) return {{}, closed};
        rounds.owner_ = &target;
        rounds.config_ = config.refine;
        // Tiles follow the cores the call runs on unless the caller overrides them.
        if (!rounds.config_.cache_bytes) rounds.config_.cache_bytes = executor.cache_bytes_per_core();
        plan.config_ = rounds.config_;
        for (indexes_t* source : sources) {
            if (source == &target) {
                if (repacking) plan.resident_shards_ = target.shards_count();
                continue;
            }
            if (status_t const closed = source->close_gated_(); failed(closed)) return {{}, closed};
            plan.sources_.push_back_reserved(source);
        }
        // The target leads the moves when it repacks, so the list it packs from names it first.
        growing_buffer<indexes_t*, forkunion_allocator_t> packed {target.shards_.allocator()};
        if (failed(packed.reserve_exactly(plan.sources_.size() + 1))) return {{}, status_t::bad_alloc_k};
        if (plan.resident_shards_) packed.push_back_reserved(&target);
        for (indexes_t* source : plan.sources_) packed.push_back_reserved(source);

        // Closed, no batch inserts, so the rows counted here are the rows that move; a refusal from
        // here reopens everything as the plan dies.
        std::size_t moved_entries = 0;
        for (indexes_t const* source : packed) moved_entries += source->size();
        status_t const packing = repacking
                                     ? target.plan_moves_(to_span(packed), capacity, rank_first_fit, plan.moves_)
                                     : target.plan_moves_(to_span(packed), capacity, rank_fewest_entries, plan.moves_);
        if (failed(packing)) return {{}, packing};
        // The moves may have opened a shard generation, whose shards take part like the others.
        std::size_t const shards = target.shards_count();
        if (failed(rounds.shard_plans_.reserve_exactly(shards)) || failed(rounds.cursors_.resize(shards)))
            return {{}, status_t::bad_alloc_k};
        // Every source row moves but those of the target's own shards that stay where they are.
        for (std::size_t shard = 0; shard != plan.resident_shards_; ++shard)
            if (plan.moves_[shard].into == shard) moved_entries -= target.shard(shard).size();
        plan.moved_entries_ = moved_entries;
        plan.source_shards_ = plan.moves_.size();
        return {std::move(plan), status_t::success_k};
    }

    /** Closes @p source and every target, each under its own gate, and opens every source shard's
     *  split into the same shard of every target. */
    static expected<ram_reshard_plan> open_split_(indexes_t& source, std::span<indexes_t* const> targets,
                                                  executor_t const& executor, reshard_config_t config) noexcept {
        std::size_t const shards = source.shards_.size(), parts = targets.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        for (indexes_t const* into : targets)
            if (into == &source || into->shards_.size() != shards) return {{}, status_t::config_mismatch_k};
        ram_reshard_plan plan {source.shards_.allocator()};
        if (failed(plan.sources_.reserve_exactly(1)) || failed(plan.targets_.reserve_exactly(parts)))
            return {{}, status_t::bad_alloc_k};
        for (indexes_t* into : targets) plan.targets_.push_back_reserved(into);
        if (status_t const closed = source.close_gated_(); failed(closed)) return {{}, closed};
        plan.sources_.push_back_reserved(&source);
        plan.splitting_ = &source;
        if (!config.refine.cache_bytes) config.refine.cache_bytes = executor.cache_bytes_per_core();
        plan.config_ = config.refine;
        // A shard generation opened before the close would hold rows no shard's split moves.
        if (source.shards_.size() != shards) return {{}, status_t::contended_k};
        // Closed, a target can take no row, so its emptiness holds for the plan's life.
        for (indexes_t* into : targets) {
            if (status_t const closed = into->close_gated_(); failed(closed)) return {{}, closed};
            ++plan.closed_targets_;
            if (into->size()) return {{}, status_t::config_mismatch_k};
        }
        growing_buffer<index_t*, forkunion_allocator_t> into_shards {source.shards_.allocator()};
        if (failed(plan.shard_plans_.reserve_exactly(shards)) || failed(plan.cursors_.resize(shards)) ||
            failed(into_shards.reserve_exactly(parts)))
            return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            into_shards.clear();
            for (indexes_t* into : targets) into_shards.push_back_reserved(into->shards_[shard].index.get());
            index_t* const from = source.shards_[shard].index.get();
            auto [split, status] = from->plan_reshard(std::span<index_t* const> {&from, 1}, to_span(into_shards),
                                                      {config.refine, 0, config.seed + shard});
            if (failed(status)) return {{}, status};
            plan.shard_plans_.push_back_reserved(std::move(split));
        }
        plan.source_shards_ = shards;
        plan.moved_entries_ = source.size();
        return {std::move(plan), status_t::success_k};
    }

    /** Whether the plan splits one source into several targets, rather than packing into one. */
    bool splits_() const noexcept { return targets_.size() > 1; }

    /** Sources still to move in, the target's own shards counting first when it repacks. */
    bool pending_sources_() const noexcept {
        std::size_t const lead = resident_shards_ ? 1 : 0;
        return moved_sources_ != lead + sources_.size();
    }

    /** The moves of the first of @p pending sources: sources move in order, so the shards of every
     *  later one close the moves. */
    std::span<shard_move_t const> moves_of_(std::span<indexes_t* const> pending) const noexcept {
        std::size_t const count = pending.front()->shards_.size();
        std::size_t later = 0;
        for (indexes_t const* source : pending.subspan(1)) later += source->shards_.size();
        return to_span(moves_).subspan(moves_.size() - later - count, count);
    }

    /** Moves every pending source into @p target under the executor's claim: the target's own
     *  shards first when it repacks, vacating the shards other sources may move into. */
    status_t move_in_(indexes_t& target, executor_t& executor) noexcept {
        if (!pending_sources_()) return status_t::success_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        std::size_t const shards = rounds_.cursors_.size(), lead = resident_shards_ ? 1 : 0;
        if (lead && !moved_sources_) {
            std::span<shard_move_t const> const own = to_span(moves_).first(resident_shards_);
            if (status_t const moved = target.move_in_(target, own, shards, executor); failed(moved)) return moved;
            moved_sources_ = 1;
        }
        for (; moved_sources_ - lead != sources_.size(); ++moved_sources_) {
            std::span<indexes_t* const> const pending = to_span(sources_).subspan(moved_sources_ - lead);
            if (status_t const moved = target.move_in_(*pending.front(), moves_of_(pending), shards, executor);
                failed(moved))
                return moved;
            // Emptied, the source takes insertions again at once.
            pending.front()->shards_.reopen();
        }
        return status_t::success_k;
    }

    /** Runs a split's pending phases on @p executor, as @ref run describes. */
    template <typename progress_type_>
    status_t run_split_(executor_t& executor, std::span<traversal_context_t> traversals,
                        progress_type_& on_visited) noexcept {
        if (!splitting_) return status_t::uninitialized_k;
        if (executor.size() != traversals.size()) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (status_t const sized = reserve(traversals); failed(sized)) return sized;
        if (status_t const drained = drain_split_(executor, traversals, on_visited); failed(drained)) return drained;
        for (shard_plan_t const& plan : shard_plans_)
            if (failed(plan.status())) return plan.status();
        return status_t::success_k;
    }

    /** Drains every source shard's split from where the cursors stand: the barriers of the moves
     *  run inside the source's window and every target's commit bracket, so a search sees every
     *  target shard's rows and keys or none, and the barrier publishing the parts empties the
     *  source inside that window. */
    template <typename progress_type_>
    status_t drain_split_(executor_t& executor, std::span<traversal_context_t> traversals,
                          progress_type_& on_visited) noexcept {
        indexes_t& source = *splitting_;
        auto run_task = [&](std::size_t shard, typename shard_plan_t::task_t const& task,
                            std::size_t thread) noexcept -> status_t {
            return shard_plans_[shard].run(task, traversals[thread]);
        };
        std::span<std::size_t> const cursors = to_writable_span(cursors_);
        std::size_t advancing = 0;
        do {
            if (status_t const drained = source.drain_tasks_(shard_plans_, cursors, executor, run_task);
                failed(drained))
                return drained;
            bool const moving = shard_plans_[0].moving();
            unique_lock_t emptying;
            if (moving) {
                emptying = unique_lock_t {source.gate_, std::try_to_lock};
                if (!emptying) return status_t::contended_k;
                source.shards_.begin_window();
                for (indexes_t* target : targets_) target->shards_.begin_commit();
            }
            advancing = 0;
            std::size_t completed = 0, total = 0;
            for (shard_plan_t& plan : shard_plans_) {
                completed += plan.completed();
                total += plan.total();
                advancing += plan.advance();
            }
            if (moving) {
                for (indexes_t* target : targets_) target->shards_.end_commit();
                if (!shard_plans_[0].moving()) source.clear_();
                source.shards_.end_window();
            }
            std::memset(cursors.data(), 0, cursors.size_bytes());
            // A round hears every barrier with the phase it closed, in that phase's own units; the
            // moves hear none.
            if (total && on_visited(completed, total) == continuation_t::cancel_k) return status_t::cancelled_k;
        } while (advancing);
        return status_t::success_k;
    }

    /** Target shards holding rows. */
    std::size_t count_target_shards_() const noexcept {
        std::size_t shards = 0;
        for (indexes_t const* target : targets_)
            for (std::size_t shard = 0; shard != target->shards_count(); ++shard)
                if (target->shard(shard).size()) ++shards;
        return shards;
    }

    /** Reopens every other source still pending, each closed at planning, the moved ones having
     *  reopened as they emptied; the plan holds none after. */
    void release_sources_() noexcept {
        std::size_t const lead = resident_shards_ ? 1 : 0;
        for (std::size_t pending = moved_sources_ > lead ? moved_sources_ - lead : 0; pending < sources_.size();
             ++pending)
            sources_[pending]->shards_.reopen();
        // The list keeps only the sources that moved, so nothing reopens twice.
        sources_.resize_reserved(moved_sources_ > lead ? moved_sources_ - lead : 0);
    }

    /** Reopens every index the plan still holds closed; the rounds reopen a pack's target. */
    void release_() noexcept {
        if (splitting_) std::exchange(splitting_, nullptr)->shards_.reopen();
        for (; closed_targets_; --closed_targets_) targets_[closed_targets_ - 1]->shards_.reopen();
        if (!splits_()) release_sources_();
    }

    /** Runs and finishes the plan on contexts carved for the call: the one-shot @ref reshard. */
    template <typename progress_type_>
    expected<reshard_report_t> drive_(executor_t& executor, progress_type_& on_visited) noexcept {
        typename indexes_t::carved_traversals_t traversals {
            arena_allocator<traversal_context_t, forkunion_allocator_t>(executor.arena(0))};
        if (status_t const carved = targets_[0]->carve_traversals_(traversals, executor); failed(carved))
            return {{}, carved};
        status_t const ran = run(executor, to_writable_span(traversals), on_visited);
        auto const [result, finish_status] = finish(executor, to_writable_span(traversals));
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /** The sources still pending and moved, copied at planning: the ones that move out, the target
     *  aside, or a split's one source. */
    growing_buffer<indexes_t*, forkunion_allocator_t> sources_;

    /** The one target a pack fills, or every target a split fills, copied at planning. */
    growing_buffer<indexes_t*, forkunion_allocator_t> targets_;

    /** A pack's rounds over its target, which they reopen. */
    refine_plan_t rounds_;

    /** A pack's moves, one per source shard in move order. */
    growing_buffer<shard_move_t, forkunion_allocator_t> moves_;

    /** The target's shards when it repacks, moving first; zero otherwise. */
    std::size_t resident_shards_ = 0;

    /** Sources moved in so far, the target's own shards counting first when it repacks. */
    std::size_t moved_sources_ = 0;

    /** A split's per-shard plans, one per source shard. */
    growing_buffer<shard_plan_t, forkunion_allocator_t> shard_plans_;

    /** One task cursor per source shard into its current phase. */
    growing_buffer<std::size_t, forkunion_allocator_t> cursors_;

    /** Targets whose dispensers a split closed, in target order. */
    std::size_t closed_targets_ = 0;

    /** A split's source, closed while the plan lives; null for a pack and once finished. */
    indexes_t* splitting_ = nullptr;

    /** The rounds' knobs. */
    refine_config_t config_ {};

    /** Shards the sources held, and source rows that change shard. */
    std::size_t source_shards_ = 0;
    std::size_t moved_entries_ = 0;
};

template <typename indexes_type_>
expected<ram_reshard_plan<indexes_type_>> plan_reshard(std::span<indexes_type_* const> sources,
                                                       std::span<indexes_type_* const> targets,
                                                       typename indexes_type_::executor_t& executor,
                                                       reshard_config_t const& config) noexcept {
    return ram_reshard_plan<indexes_type_>::open_(sources, targets, executor, config);
}

template <typename indexes_type_, typename progress_type_>
expected<reshard_report_t> reshard(std::span<indexes_type_* const> sources, std::span<indexes_type_* const> targets,
                                   typename indexes_type_::executor_t& executor, reshard_config_t const& config,
                                   progress_type_&& on_visited) noexcept {
    auto [plan, status] = plan_reshard(sources, targets, executor, config);
    if (failed(status)) return {{}, status};
    return plan.drive_(executor, on_visited);
}

} // namespace unum::usearch::dense
