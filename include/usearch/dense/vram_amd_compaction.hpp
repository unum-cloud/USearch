/**
 *  @file include/usearch/dense/vram_amd_compaction.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The freestanding compaction of @c vram_amd_indexes sources into a RAM target under a
 *      shard capacity.
 *
 *  A compaction packs the source shards largest first into target shards of at most a shard
 *  capacity of rows, each into the first with room, so it ends with as few shards as fit. Each
 *  source shard crosses to the host through its own save plan and lands in its target shard
 *  through that shard's load plan, its edges rebased and its keys beside it: the first part of a
 *  bin keeps its IDs, every later one lands after it. Only the appended rows then walk the target
 *  across the parts and link in, and each target shard heals; the landings feed the RAM absorb
 *  plan's own walk.
 *
 *  The one header that sees both residencies: it includes the AMD tower and the portable RAM
 *  one, which never include each other.
 */
#pragma once

#include <cstring> // `std::memset` - the load cursors between source shards

#include <span>        // `std::span`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::exchange`

#include "usearch/dense/types.hpp" // `compact_config_t`, `compact_result_t`
#include "usearch/dense/ram_indexes.hpp"
#include "usearch/dense/vram_amd_indexes.hpp"

namespace unum::usearch::dense {

/**
 *  @brief Opens a compaction of the device @p sources into the host @p target, each target shard
 *      holding at most @p config.shard_capacity rows.
 *
 *  The target and every source close here, each under its own gate, so the packing never goes
 *  stale: the target stays closed while the plan lives and each source until it lands, and
 *  insertions into any of them answer @c contended_k meanwhile. Searches of the sources run
 *  throughout, retrying once around the window that empties each; searches of the target retry
 *  around the commits that publish each source's rows. Every row moves, so the sources end empty,
 *  closed and sealed, keeping their shards and storage, and their slots must outlive the plan.
 *
 *  @param[in] sources The device indexes giving their rows up, on the devices @p device_executor
 *      drives.
 *  @param[inout] target The host index receiving them; one never reserved takes the sources'
 *      geometry.
 *  @param[inout] device_executor The streams every source shard saves through.
 *  @param[inout] host_executor The threads every target shard loads and walks on.
 *  @retval config_mismatch_k A null or repeated source, another metric or geometry, or a source
 *      device @p device_executor drives no stream on.
 *  @retval contended_k A call or batch in flight on the target or a source, or another refine,
 *      absorb, split, compaction, save or load holding either.
 *  @retval unsupported_k A source shard past the capacity, which would have to be cut first.
 */
template <typename vram_indexes_type_, typename ram_indexes_type_>
expected<vram_amd_compact_plan<vram_indexes_type_, ram_indexes_type_>> plan_compact(
    std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target, hip_executor_t& device_executor,
    typename ram_indexes_type_::executor_t& host_executor, compact_config_t const& config = {}) noexcept;

/** The one-shot @ref plan_compact, run and finished on contexts carved for the call; @p on_visited
 *  reports and cancels the re-selection alone, as in @c vram_amd_compact_plan::run. */
template <typename vram_indexes_type_, typename ram_indexes_type_, typename progress_type_ = dummy_progress_t>
expected<compact_result_t> compact(std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target,
                                   hip_executor_t& device_executor,
                                   typename ram_indexes_type_::executor_t& host_executor,
                                   compact_config_t const& config = {}, progress_type_&& on_visited = {}) noexcept;

/**
 *  @brief One compaction in flight from AMD device shards into a RAM target, holding every
 *      index it moves closed meanwhile.
 *
 *  @ref run lands the sources still pending, each source shard staged on the host once through
 *  columns the plan sized at planning for the widest, then drains the re-selection of the appended
 *  rows and the heal; a cancel pauses only that walk, and a later call continues it. @ref finish
 *  heals what a stopped walk leaves, reopens the target and every source still pending, and
 *  reports.
 */
template <typename vram_indexes_type_, typename ram_indexes_type_>
struct [[nodiscard]] vram_amd_compact_plan {
    using vram_indexes_t = vram_indexes_type_;
    using ram_indexes_t = ram_indexes_type_;
    using device_executor_t = typename vram_indexes_t::executor_t;
    using host_executor_t = typename ram_indexes_t::executor_t;
    using traversal_context_t = typename ram_indexes_t::traversal_context_t;
    using refine_plan_t = typename ram_indexes_t::refine_plan_t;
    using result_t = compact_result_t;
    static_assert(vram_indexes_t::vector_dtype_k == ram_indexes_t::vector_dtype_k,
                  "Rows cross as stored, so both residencies keep one scalar");
    static_assert(std::is_same_v<typename vram_indexes_t::key_t, typename ram_indexes_t::key_t>,
                  "Keys cross as stored, so both residencies keep one key");

    vram_amd_compact_plan() noexcept = default;
    vram_amd_compact_plan(vram_amd_compact_plan&& other) noexcept
        : refining_(std::move(other.refining_)), columns_(std::move(other.columns_)),
          sources_(std::exchange(other.sources_, {})), landed_(other.landed_), moved_rows_(other.moved_rows_) {}
    vram_amd_compact_plan& operator=(vram_amd_compact_plan&&) = delete;
    vram_amd_compact_plan(vram_amd_compact_plan const&) = delete;
    vram_amd_compact_plan& operator=(vram_amd_compact_plan const&) = delete;

    /** Reopens every source still pending; the refine plan reopens the target. */
    ~vram_amd_compact_plan() noexcept { release_sources_(); }

    /**
     *  @brief Lands the pending sources, then runs the re-selection until it ends or @p on_visited
     *      cancels; @c cancelled_k means a later call continues exactly there.
     *
     *  A source failing to land stays pending and closed, with every later one, for the next call.
     *
     *  @param[inout] device_executor The streams the plan was opened over.
     *  @param[inout] host_executor The threads the plan was opened over.
     *  @param[inout] traversals The caller's context of every host thread, which the walks run on;
     *      the call sizes them.
     *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
     *      population; a @c cancel_k answer stops the drain where its steps ran.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t run(device_executor_t& device_executor, host_executor_t& host_executor,
                 std::span<traversal_context_t> traversals, progress_type_&& on_visited = {}) noexcept {
        ram_indexes_t* const target = refining_.owner_;
        if (!target) return status_t::uninitialized_k;
        if (host_executor.size() != traversals.size()) return status_t::config_mismatch_k;
        if (status_t const landed = land_pending_(*target, device_executor, host_executor); failed(landed))
            return landed;
        return refining_.run(host_executor, traversals, on_visited);
    }

    /** Stops any re-selection, heals on the caller's @p traversals, one per host thread, reopens
     *  the target and reports. A plan whose sources have not all landed heals nothing. */
    expected<result_t> finish(host_executor_t& host_executor, std::span<traversal_context_t> traversals) noexcept {
        result_t result;
        ram_indexes_t* const target = refining_.owner_;
        if (!target) return {result, status_t::uninitialized_k};
        result.source_shards = refining_.landings_.size();
        result.moved_rows = moved_rows_;
        release_sources_();
        auto [graph, status] = refining_.finish(host_executor, traversals);
        result.graph = graph;
        for (std::size_t shard = 0; shard != target->shards_count(); ++shard)
            if (target->shard(shard).size()) ++result.target_shards;
        return {result, status};
    }

  private:
    template <typename other_vram_type_, typename other_ram_type_>
    friend expected<vram_amd_compact_plan<other_vram_type_, other_ram_type_>> plan_compact(
        std::span<other_vram_type_* const>, other_ram_type_&, hip_executor_t&, typename other_ram_type_::executor_t&,
        compact_config_t const&) noexcept;
    template <typename other_vram_type_, typename other_ram_type_, typename progress_type_>
    friend expected<compact_result_t> compact(std::span<other_vram_type_* const>, other_ram_type_&, hip_executor_t&,
                                              typename other_ram_type_::executor_t&, compact_config_t const&,
                                              progress_type_&&) noexcept;

    using vram_index_t = typename vram_indexes_t::index_t;
    using vram_shard_t = typename vram_indexes_t::shard_t;
    using fanout_t = typename vram_indexes_t::fanout_t;
    using ram_index_t = typename ram_indexes_t::index_t;
    using ram_shard_t = typename ram_indexes_t::shard_t;
    using ram_load_plan_t = typename ram_index_t::load_plan_t;

    /** A host shard and a device shard never share a placement, so every target shard ranks as
     *  another placement's. */
    struct placement_of_t {
        bool operator()(ram_shard_t const&) const noexcept { return false; }
        bool operator()(vram_shard_t const&) const noexcept { return true; }
    };

    explicit vram_amd_compact_plan(forkunion_allocator_t allocator) noexcept
        : refining_(allocator), columns_(allocator) {}

    /** Bytes one row takes in the host columns: its packed vector, then its neighbors. */
    static std::size_t staged_row_bytes_(vram_indexes_t const& source) noexcept {
        return packed_bytes(vram_indexes_t::vector_dtype_k, source.dimensions()) +
               source.connectivity() * sizeof(node_id_t);
    }

    /** Validates @p sources against @p target, closes the target and every source, each under its
     *  own gate, packs every source shard into the target under the tighter of
     *  @p config.shard_capacity and the target's own bound, and sizes the host columns for the
     *  widest source shard. */
    static expected<vram_amd_compact_plan> open_(std::span<vram_indexes_t* const> sources, ram_indexes_t& target,
                                                 device_executor_t const& device_executor,
                                                 host_executor_t const& host_executor,
                                                 compact_config_t const& config) noexcept {
        if (sources.empty() || !target.shards_count() || !host_executor.size()) return {{}, status_t::uninitialized_k};
        vram_indexes_t const* shaped = sources.front();
        for (std::size_t position = 0; position != sources.size(); ++position) {
            vram_indexes_t const* source = sources[position];
            if (!source || source->kernel() != target.kernel()) return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != position; ++earlier)
                if (sources[earlier] == source) return {{}, status_t::config_mismatch_k};
            if (status_t const kept = source->status(); failed(kept)) return {{}, kept};
            if (status_t const paired = fanout_t::pairs_with(source->shards_, device_executor); failed(paired))
                return {{}, paired};
            if (!source->capacity()) continue;
            if (!shaped->capacity()) shaped = source;
            if (source->dimensions() != shaped->dimensions() || source->connectivity() != shaped->connectivity() ||
                (target.capacity() &&
                 (source->dimensions() != target.dimensions() || source->connectivity() != target.connectivity())))
                return {{}, status_t::config_mismatch_k};
        }
        // Zero, or anything past the target's own bound, means that bound.
        std::size_t const capacity = config.shard_capacity && config.shard_capacity < target.shard_capacity()
                                         ? config.shard_capacity
                                         : target.shard_capacity();

        vram_amd_compact_plan plan {target.shards_.allocator()};
        refine_plan_t& refining = plan.refining_;
        if (status_t const closed = target.close_gated_(); failed(closed)) return {{}, closed};
        refining.owner_ = &target;
        refining.config_.expansion_factor = config.expansion_factor;
        for (std::size_t closing = 0; closing != sources.size(); ++closing)
            if (status_t const closed = sources[closing]->close_gated_(); failed(closed)) {
                while (closing) sources[--closing]->shards_.reopen();
                return {{}, closed};
            }
        plan.sources_ = sources;

        // Closed, the sources hold still, so their shards are what the landings pack.
        std::size_t count = 0, staged_bytes = 0;
        for (vram_indexes_t const* source : sources) {
            count += source->shards_.size();
            plan.moved_rows_ += source->size();
            for (vram_shard_t const& one : source->shards_)
                staged_bytes = max_of(staged_bytes, one.index->size() * staged_row_bytes_(*source));
        }
        if (failed(refining.landings_.resize(count)) || failed(plan.columns_.resize_uninitialized(staged_bytes)))
            return {{}, status_t::bad_alloc_k};
        auto source_shard = [&](std::size_t from) noexcept -> vram_shard_t const& {
            std::size_t source = 0;
            for (; from >= sources[source]->shards_.size(); ++source) from -= sources[source]->shards_.size();
            return sources[source]->shards_[from];
        };
        // Only a source with rows needs a generation, so the geometry to take exists.
        auto open = [&]() noexcept {
            if (!target.capacity())
                if (status_t const reserved =
                        target.reserve({0, static_cast<connectivity_t>(shaped->connectivity()), shaped->dimensions()})
                            .status;
                    failed(reserved))
                    return reserved;
            // The target is sealed, so the generation opens closed like the rest, past the seal.
            return target.shards_.open_generation_past_seal_(
                target.shards_.open_generation(),
                [&target](ram_shard_t const& built) noexcept { return target.build_shard_(built); });
        };
        if (status_t const packed = target.shards_.pack(to_writable_span(refining.landings_), 0, capacity, source_shard,
                                                        placement_of_t {}, rank_first_fit, open);
            failed(packed))
            return {{}, packed};
        if (status_t const sized = ram_indexes_t::shards_t::size_parts(
                to_span(refining.landings_), target.shards_.size(), refining.part_offsets_, refining.part_bases_);
            failed(sized))
            return {{}, sized};
        std::size_t const shards = refining.part_offsets_.size() - 1;
        if (failed(refining.shard_plans_.reserve_exactly(shards)) || failed(refining.cursors_.resize(shards)))
            return {{}, status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Lands every pending source into @p target under both executors' claims, each source's
     *  landings the run of them its shards number. */
    status_t land_pending_(ram_indexes_t& target, device_executor_t& device_executor,
                           host_executor_t& host_executor) noexcept {
        auto [device_claim, device_status] = device_executor.claim();
        if (failed(device_status)) return device_status;
        auto [host_claim, host_status] = host_executor.claim();
        if (failed(host_status)) return host_status;
        std::size_t first = 0;
        for (std::size_t source = 0; source != landed_; ++source) first += sources_[source]->shards_.size();
        for (; landed_ != sources_.size(); ++landed_) {
            vram_indexes_t& source = *sources_[landed_];
            std::size_t const count = source.shards_.size();
            if (status_t const landed = land_(target, source, to_span(refining_.landings_).subspan(first, count),
                                              device_executor, host_executor);
                failed(landed))
                return landed;
            first += count;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Lands every row of @p source where @p landings, one per source shard, put it and
     *      consumes it: the source's gate and every other fallible step first, then the copies,
     *      then one commit bracket publishes and the source shards join the plan's parts.
     *
     *  Every target shard claims all the rows it takes up front, so a failure before publication
     *  releases them. Each source shard then saves into the host columns on its device's stream
     *  and drains, and every target shard's load step covering its landing copies out of the
     *  columns on the host threads, its keys beside it. Once every part landed, the target
     *  publishes, the keys resolve, and the source forgets its rows in place inside its window.
     */
    status_t land_(ram_indexes_t& target, vram_indexes_t& source, std::span<landing_t const> landings,
                   device_executor_t& device_executor, host_executor_t& host_executor) noexcept {
        std::size_t const shards = refining_.cursors_.size();
        if (!source.size()) return status_t::success_k;
        // The source empties once its rows land, so its gate precedes anything a refusal undoes.
        typename vram_indexes_t::unique_lock_t const emptying {source.gate_, std::try_to_lock};
        if (!emptying) return status_t::contended_k;
        if (!target.capacity())
            if (status_t const reserved = target
                                              .reserve({source.size(),
                                                        static_cast<connectivity_t>(source.connectivity()),
                                                        source.dimensions()})
                                              .status;
                failed(reserved))
                return reserved;

        std::size_t const sources = landings.size();
        growing_buffer<ram_load_plan_t, forkunion_allocator_t> plans {target.shards_.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {target.shards_.allocator()};
        if (failed(plans.reserve_exactly(shards)) || failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        std::size_t landing_rows = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = 0; from != sources; ++from)
                if (landings[from].into == into) rows += source.shards_[from].index->size();
            auto [plan, status] = target.shards_[into].index->plan_load(rows);
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
            if (status_t const covered = target.shards_[into].keys.cover(plans[into].base + rows); failed(covered))
                return covered;
            landing_rows += rows;
        }
        auto [seats, seat_status] = target.shards_.locators().claim(landing_rows);
        if (failed(seat_status)) return seat_status;

        std::size_t const vectors_row_bytes = packed_bytes(vram_indexes_t::vector_dtype_k, source.dimensions());
        std::size_t const neighbors_row_bytes = source.connectivity() * sizeof(node_id_t);
        for (std::size_t from = 0; from != sources; ++from) {
            vram_shard_t const& part = source.shards_[from];
            std::size_t const into = landings[from].into;
            node_id_t const rows = static_cast<node_id_t>(part.index->size());
            if (!rows) continue;
            node_id_t first = plans[into].base;
            for (std::size_t earlier = 0; earlier != from; ++earlier)
                if (landings[earlier].into == into)
                    first += static_cast<node_id_t>(source.shards_[earlier].index->size());
            iota_view<node_id_t> const landing {first, static_cast<node_id_t>(first + rows)};

            byte_t* const vectors = columns_.data();
            byte_t* const neighbors = vectors + rows * vectors_row_bytes;
            typename vram_index_t::columns_t const staged {bytes_of(vectors, rows * vectors_row_bytes),
                                                           bytes_of(neighbors, rows * neighbors_row_bytes),
                                                           {}};
            if (status_t const entered = fanout_t::enter(source.shards_, from); failed(entered)) return entered;
            auto [saving, save_status] = part.index->plan_save(
                staged, synchronization_t::immutable_k,
                device_executor.stream(fanout_t::stream_of(source.shards_, device_executor, from)));
            if (failed(save_status)) return save_status;
            for (typename vram_index_t::save_plan_t::step_t const step : saving.steps())
                if (status_t const saved = saving.save(step, {}); failed(saved)) return saved;
            if (status_t const saved = saving.finish(); failed(saved)) return saved;

            // A target step is one storage chunk, which may straddle this part and its neighbors.
            typename ram_index_t::columns_view_t const view {
                bytes_of(static_cast<byte_t const*>(vectors), rows * vectors_row_bytes),
                bytes_of(static_cast<byte_t const*>(neighbors), rows * neighbors_row_bytes),
                {}};
            auto land = [&](std::size_t shard, typename ram_load_plan_t::step_t const& chunk, std::size_t,
                            nk::matrix_span<node_id_t>) noexcept -> status_t {
                if (shard != into) return status_t::success_k;
                typename ram_load_plan_t::step_t step = chunk;
                step.index_rows = {max_of(chunk.index_rows.first, landing.first),
                                   min_of(chunk.index_rows.bound, landing.bound)};
                if (step.index_rows.first >= step.index_rows.bound) return status_t::success_k;
                step.source_rows = {static_cast<node_id_t>(step.index_rows.first - landing.first),
                                    static_cast<node_id_t>(step.index_rows.bound - landing.first)};
                if (status_t const loaded = plans[into].load(step, view, {0, rows}); failed(loaded)) return loaded;
                for (node_id_t row : step.index_rows)
                    target.shards_[into].keys[row] = part.keys.host[row - landing.first];
                return status_t::success_k;
            };
            std::memset(cursors.data(), 0, shards * sizeof(std::size_t));
            if (status_t const loaded = target.drain_steps_(plans, to_writable_span(cursors), host_executor, 0, land);
                failed(loaded))
                return loaded;
        }

        // Every part landed, so no plan refuses; the bracket shows a search every shard's rows or
        // none, and the keys resolve only once their rows are visible.
        status_t published = status_t::success_k;
        target.shards_.begin_commit();
        for (ram_load_plan_t& plan : plans)
            if (succeeded(published)) published = plan.finish();
        if (succeeded(published))
            target.shards_.record_parts(source.shards_, landings, plans, refining_.part_bases_.data(),
                                        to_span(refining_.part_offsets_));
        target.shards_.end_commit();
        if (failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                target.shards_.locators().insert(seats, target.shards_[into].keys[row],
                                                 typename ram_indexes_t::key_locator_t {static_cast<u32_t>(into), row});
        source.shards_.begin_window();
        source.clear_();
        source.shards_.end_window();
        return status_t::success_k;
    }

    /** Reopens every source still pending, each closed at planning; the plan holds none after. */
    void release_sources_() noexcept {
        for (std::size_t pending = landed_; pending < sources_.size(); ++pending) sources_[pending]->shards_.reopen();
        sources_ = {};
    }

    /** Runs and finishes the plan on host contexts carved for the call: the one-shot
     *  @ref compact. */
    template <typename progress_type_>
    expected<result_t> drive_(device_executor_t& device_executor, host_executor_t& host_executor,
                              progress_type_& on_visited) noexcept {
        typename ram_indexes_t::carved_traversals_t traversals {
            arena_allocator<traversal_context_t, forkunion_allocator_t>(host_executor.arena(0))};
        if (status_t const carved = refining_.owner_->carve_traversals_(traversals, host_executor); failed(carved))
            return {{}, carved};
        status_t const ran = run(device_executor, host_executor, to_writable_span(traversals), on_visited);
        auto const [result, finish_status] = finish(host_executor, to_writable_span(traversals));
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /** The absorb plan over the target: its landings, part runs and per-shard walks. */
    refine_plan_t refining_;

    /** One source shard's packed vectors, then its neighbors, sized at planning for the widest,
     *  which every source shard crosses to the host through. */
    growing_buffer<byte_t, forkunion_allocator_t> columns_ {forkunion_allocator_t {}};

    /** The caller's source slots, borrowed until the plan dies. */
    std::span<vram_indexes_t* const> sources_;

    /** Sources landed so far. */
    std::size_t landed_ = 0;

    /** Source rows, every one of which changes shard. */
    std::size_t moved_rows_ = 0;
};

template <typename vram_indexes_type_, typename ram_indexes_type_>
expected<vram_amd_compact_plan<vram_indexes_type_, ram_indexes_type_>> plan_compact(
    std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target, hip_executor_t& device_executor,
    typename ram_indexes_type_::executor_t& host_executor, compact_config_t const& config) noexcept {
    return vram_amd_compact_plan<vram_indexes_type_, ram_indexes_type_>::open_(sources, target, device_executor,
                                                                               host_executor, config);
}

template <typename vram_indexes_type_, typename ram_indexes_type_, typename progress_type_>
expected<compact_result_t> compact(std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target,
                                   hip_executor_t& device_executor,
                                   typename ram_indexes_type_::executor_t& host_executor,
                                   compact_config_t const& config, progress_type_&& on_visited) noexcept {
    auto [plan, status] = plan_compact(sources, target, device_executor, host_executor, config);
    if (failed(status)) return {{}, status};
    return plan.drive_(device_executor, host_executor, on_visited);
}

} // namespace unum::usearch::dense
