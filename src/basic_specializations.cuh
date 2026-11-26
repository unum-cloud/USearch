/**
 *  @file src/basic_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What every basic modality's variants share, whatever they store: the fan-outs over the
 *      executor's threads, each residency's lookups and cursors, the GPU verbs, and the bodies
 *      each variant's verb table points at.
 *
 *  Each modality's @c <modality>_specializations.cuh spells its variants on top of this, and its
 *  class units in @c src/<modality>/ instantiate their class tables; @c basic_index.cuh reaches
 *  them only through @c basic_punned.cuh. The basic index is the same type in every class, so a
 *  class differs only in the kits its metrics score with, and the variants keep the index's
 *  default waiter and atomic reference. Only class units include it, each compiled with its
 *  class's instruction set, so every body is a static member of a template over a variant.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - an entry in an NVMe read's buffer, kept past the read

#include <array>       // `std::array` - the one key a view scans for
#include <bit>         // `std::bit_cast` - the C key struct & `us::util::uuid128_t` share sixteen bytes
#include <limits>      // `std::numeric_limits` - a GPU index's sentinel node ID
#include <memory>      // `std::make_obj_using_allocator` - values built from the index's allocator
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`, `std::invoke_result_t` - the predicate a thread asks

#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a key's entries in insertion order
#include "usearch/utilities/safegraphs_format.hpp"
#include "usearch/basic/ram_indexes.hpp"       // `basic::ram_indexes` - every RAM variant's shards
#include "usearch/basic/shard_checkpoints.hpp" // `basic::save`, `basic::load` - every RAM variant's checkpoints
#include "usearch/basic/nvme_indexes.hpp" // `basic::nvme_indexes` and SmashTable's row kits, through its directories
#if USEARCH_ARCH_CUDA_
#include "usearch/basic/vram_nvidia_indexes.hpp" // `basic::vram_nvidia_indexes` - every GPU variant's shards
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/basic/vram_amd_indexes.hpp" // `basic::vram_amd_indexes` - every AMD GPU variant's shards
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/basic/vram_apple_indexes.hpp" // `basic::vram_apple_indexes` - every Apple GPU variant's shards
#endif

#include "basic_punned.cuh"

namespace util = unum::usearch::utilities;

using basic::traversal_stats_t;

/** The SmashTable row kit one CPU class pairs for the scans its NVMe directories run: picked here
 *  per class, never by a unit's own flags, so every class unit spells one index type with one kit
 *  - the serial floor, as is any class whose @c SMASHTABLE_TARGET_<KIT> is off. The monoversioned
 *  class alone has one unit per modality, so it takes the newest kit its flags compile. */
template <cpu_class_t cpu_class_>
struct row_kit_for_cpu_class {
    using type = st::serial_row_kit_t;
};
#if SMASHTABLE_TARGET_HASWELL
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_haswell_k> {
    using type = st::haswell_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_sierra_k> {
    using type = st::haswell_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_SKYLAKE
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_skylake_k> {
    using type = st::skylake_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_ICELAKE
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_icelake_k> {
    using type = st::icelake_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::amd_genoa_k> {
    using type = st::icelake_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_sapphire_k> {
    using type = st::icelake_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_granite_k> {
    using type = st::icelake_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::intel_diamond_k> {
    using type = st::icelake_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_NEON
template <>
struct row_kit_for_cpu_class<cpu_class_t::aws_g2_k> {
    using type = st::neon_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::apple_m2_k> {
    using type = st::neon_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::apple_m4_k> {
    using type = st::neon_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_SVE
template <>
struct row_kit_for_cpu_class<cpu_class_t::aws_g5_k> {
    using type = st::sve_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    using type = st::sve_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_RVV
template <>
struct row_kit_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    using type = st::rvv_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_POWERVSX
template <>
struct row_kit_for_cpu_class<cpu_class_t::ibm_power10_k> {
    using type = st::powervsx_row_kit_t;
};
#endif
#if SMASHTABLE_TARGET_V128
template <>
struct row_kit_for_cpu_class<cpu_class_t::wasm32_v128_k> {
    using type = st::v128_row_kit_t;
};
template <>
struct row_kit_for_cpu_class<cpu_class_t::wasm64_extended_k> {
    using type = st::v128_row_kit_t;
};
#endif
#if USEARCH_TARGET_MONOVERSIONED
template <>
struct row_kit_for_cpu_class<cpu_class_t::monoversioned_k> {
    using type = st::native_row_kit_t;
};
#endif

/** The row kit a CPU class pairs, spelled short for the variants' explicit instantiations. */
template <cpu_class_t cpu_class_>
using row_kit_of = typename row_kit_for_cpu_class<cpu_class_>::type;

#pragma region Fan Outs

/** Runs @p per_task over @p tasks taken in ascending order by the executor's threads, answering the
 *  fan-out's refusal rather than running them on one thread, which would hide a held executor. */
template <template <typename> class atomic_ref_template_, typename per_task_type_>
util::status_t drain_in_order(util::ram_executor_t& executor, std::size_t tasks, per_task_type_&& per_task) noexcept {
    if (!tasks) return util::status_code_t::success_k;
    // Every task bumps it, so it takes its own line, away from the locals every thread reads.
    alignas(util::default_alignment_k) std::size_t cursor = 0;
    auto drain = [&](std::size_t thread, std::size_t) noexcept {
        atomic_ref_template_<std::size_t> const next {cursor};
        for (std::size_t task = next.fetch_add(1, std::memory_order_relaxed); task < tasks;
             task = next.fetch_add(1, std::memory_order_relaxed))
            per_task(task, thread);
    };
    return executor.for_threads(drain);
}

/** The metric a factory of @p make_metric_type_ builds over a thread's arena. */
template <typename make_metric_type_>
using made_metric = typename std::invoke_result_t<make_metric_type_&, thread_allocator_t const&, std::size_t>::value_t;

/** One call's per-thread scratch: a traversal context and a metric per executor thread, both over
 *  that thread's arena, so the fan-out never allocates and never shares scratch, and the round
 *  capacity every metric stages, the most candidates one round hands a predicate. */
template <typename index_type_, typename metric_type_>
struct thread_scratch {
    using traversal_context_t = typename index_type_::template traversal_context_over<thread_allocator_t>;

    util::growing_buffer<traversal_context_t, thread_allocator_t> traversals;
    util::growing_buffer<metric_type_, thread_allocator_t> metrics;
    std::size_t round_capacity = 0;

    /** Places the per-thread objects in the first arena of @p executor, which @ref prepare
     *  rewinds before it builds them. */
    explicit thread_scratch(util::ram_executor_t& executor) noexcept
        : traversals(thread_allocator_t {executor.arena(0)}), metrics(thread_allocator_t {executor.arena(0)}) {}

    /** Rewinds every arena, keeping the high-water mark of earlier calls, and builds one context
     *  sized for @p connectivity and @p expansion plus one metric from @p make_metric for each
     *  of the first @p threads threads. */
    template <typename make_metric_type_>
    util::status_t prepare(util::ram_executor_t& executor, std::size_t threads, std::size_t connectivity,
                           std::size_t expansion, util::u64_t seed, make_metric_type_&& make_metric) noexcept {
        if (util::status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        if (failed(traversals.reserve_exactly(threads)) || failed(metrics.reserve_exactly(threads)))
            return util::status_code_t::out_of_memory_k;
        typename index_type_::insert_config_t const walk {expansion};
        round_capacity = index_type_::round_capacity(connectivity, walk);
        for (std::size_t thread = 0; thread != threads; ++thread) {
            thread_allocator_t const allocator {executor.arena(thread)};
            traversals.push_back_reserved(traversal_context_t {seed + thread, allocator});
            if (util::status_t const sized = index_type_::reserve_traversal(traversals[thread], connectivity, walk);
                failed(sized))
                return sized;
            auto [metric, make_status] = make_metric(allocator, round_capacity);
            if (failed(make_status)) return make_status;
            metrics.push_back_reserved(std::move(metric));
        }
        return util::status_code_t::success_k;
    }

    /** The counters every thread's walks accumulated. */
    traversal_stats_t telemetry() const noexcept {
        traversal_stats_t total;
        for (traversal_context_t const& context : traversals) total += context.telemetry;
        return total;
    }
};

/** One call's seeding metrics beside its @ref thread_scratch : one per executor thread over that
 *  thread's arena, writing the sketch of each query the thread walks first. */
template <typename seeding_metric_type_>
struct seeding_scratch {
    using seeding_walk_t = util::seeding_walk<typename seeding_metric_type_::query_view_t, seeding_metric_type_>;

    util::growing_buffer<seeding_metric_type_, thread_allocator_t> metrics;

    explicit seeding_scratch(util::ram_executor_t& executor) noexcept
        : metrics(thread_allocator_t {executor.arena(0)}) {}

    /** Builds one metric from @p make_seeding for each of the first @p threads threads, staging
     *  rounds of @p round_capacity, in the arenas the call's one rewind left. */
    template <typename make_seeding_type_>
    util::status_t prepare(util::ram_executor_t& executor, std::size_t threads, std::size_t round_capacity,
                           make_seeding_type_&& make_seeding) noexcept {
        if (failed(metrics.reserve_exactly(threads))) return util::status_code_t::out_of_memory_k;
        for (std::size_t thread = 0; thread != threads; ++thread) {
            auto [metric, make_status] = make_seeding(thread_allocator_t {executor.arena(thread)}, round_capacity);
            if (failed(make_status)) return make_status;
            metrics.push_back_reserved(std::move(metric));
        }
        return util::status_code_t::success_k;
    }

    /** The seeding walk of @p query on @p thread : the sketch that thread's metric writes. */
    template <typename query_type_>
    seeding_walk_t seeding_of(std::size_t thread, query_type_ const& query) noexcept {
        seeding_metric_type_& metric = metrics[thread];
        return {metric.sketch_of(query), metric};
    }

    /** The seeding metric of @p thread, reading the sketches of stored values it walks between. */
    seeding_metric_type_* metric_of(std::size_t thread) noexcept { return &metrics[thread]; }
};

/** No seeding metrics: every search walks once. */
template <>
struct seeding_scratch<util::unseeded_t> {
    explicit seeding_scratch(util::ram_executor_t&) noexcept {}

    template <typename make_seeding_type_>
    util::status_t prepare(util::ram_executor_t&, std::size_t, std::size_t, make_seeding_type_&&) noexcept {
        return util::status_code_t::success_k;
    }

    template <typename query_type_>
    util::unseeded_t seeding_of(std::size_t, query_type_ const&) noexcept {
        return {};
    }

    util::unseeded_t* metric_of(std::size_t) noexcept { return nullptr; }
};

/**
 *  @brief Inserts the entries of @p tape under their keys as one batch: every thread builds its
 *      share of the values from the index's allocator, then one plan writes them on every thread
 *      and publishes once.
 *
 *  Everything fallible runs before the plan dispenses: the tape's offsets, lowered into the first
 *  arena after its one rewind, the keys and values, the traversals and the metrics. A refused build
 *  frees every value built so far and leaves the index untouched. Once the plan dispensed, every
 *  position is written and the progress handler cannot cancel. The plan routes the entries over the
 *  shards of @p indexes, and their walks run on each shard's own scratch, never on a shared one.
 */
template <typename indexes_type_, typename make_metric_type_, typename build_type_>
util::expected<traversal_stats_t> insert_entries(indexes_type_& index, usearch_keys_view_t keys, caller_tape_t tape,
                                                 std::size_t connectivity, std::size_t expansion,
                                                 make_metric_type_&& make_metric, build_type_&& build,
                                                 progress_ticks_t& ticks, util::ram_executor_t& executor) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using value_t = typename indexes_type_::value_t;
    using key_t = typename indexes_type_::key_t;
    using metric_t = made_metric<make_metric_type_>;
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (util::status_t const prepared = scratch.prepare(executor, executor.size(), connectivity, expansion,
                                                        index.size(), make_metric);
        failed(prepared))
        return {{}, prepared};

    auto const [entries, lower_status] = counted_tape_of(tape.tape, tape.count, executor.arena(0));
    if (failed(lower_status)) return {{}, lower_status};
    std::size_t const count = tape.count;
    thread_allocator_t const caller {executor.arena(0)};
    util::growing_buffer<key_t, thread_allocator_t> identities {caller};
    util::growing_buffer<value_t, thread_allocator_t> values {caller};
    if (failed(identities.reserve_exactly(count)) || failed(values.reserve_exactly(count)))
        return {{}, util::status_code_t::out_of_memory_k};
    for (std::size_t offset = 0; offset != count; ++offset) {
        identities.push_back_reserved(uuid_of(key_at(keys, offset)));
        values.push_back_reserved(std::make_obj_using_allocator<value_t>(index.get_allocator()));
    }
    util::status_t built = util::status_code_t::success_k;
    util::status_t const forked = drain_in_order<indexes_type_::template atomic_ref>(
        executor, count, [&](std::size_t offset, std::size_t thread) noexcept {
            if (succeeded(typename indexes_type_::template atomic_ref<util::status_code_t>(built.code)
                              .load(std::memory_order_relaxed)))
                keep_first_failure<indexes_type_::template atomic_ref>(built,
                                                                       build(entries, offset, values[offset], thread));
        });
    if (failed(forked)) return {{}, forked};
    if (failed(built)) return {{}, built};

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    basic::keep_diverse_t const selector {connectivity};
    typename indexes_type_::insert_config_t const config {expansion};
    if (util::status_t const inserted = drain_in_order<indexes_type_::template atomic_ref>(
            executor, count,
            [&](std::size_t offset, std::size_t thread) noexcept {
                plan.insert(offset, to_writable_span(values).subspan(offset, 1), to_span(identities).subspan(offset, 1),
                            scratch.metrics[thread], scratch.traversals[thread], selector, config);
                [[maybe_unused]] util::continuation_t const heard = ticks(offset);
            });
        util::failed(inserted))
        return {{}, inserted};
    if (util::status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    return {scratch.telemetry(), util::status_code_t::success_k};
}

/** The scratch of one call driving a sharded index's @p plan over shards of @p shard_type_, a
 *  metric from @p make_metric per thread, each context grown to what the plan's tasks touch beside
 *  @p widest_list, the widest neighbor list a restitch copies. */
template <typename shard_type_, typename plan_type_, typename make_metric_type_>
util::expected<thread_scratch<shard_type_, made_metric<make_metric_type_>>> prepare_refinement(
    plan_type_ const& plan, std::size_t widest_list, make_metric_type_&& make_metric,
    util::ram_executor_t& executor) noexcept {
    thread_scratch<shard_type_, made_metric<make_metric_type_>> scratch {executor};
    std::size_t const connectivity = util::max_of(widest_list, plan.connectivity());
    util::status_t prepared = scratch.prepare(executor, executor.size(), connectivity,
                                              util::max_of(connectivity, plan.reach_expansion()), plan.health().visible,
                                              make_metric);
    for (std::size_t thread = 0; succeeded(prepared) && thread != scratch.traversals.size(); ++thread)
        prepared = plan.reserve(scratch.traversals[thread]);
    return {std::move(scratch), prepared};
}

/**
 *  @brief Runs @p plan 's pending phases on the executor's threads, each with its metric and
 *      context from @p scratch, until no phase follows or @p ticks cancels; @c cancelled_k means a
 *      later call has more to do.
 *
 *  Every task of a phase runs before the barrier, and every barrier of a round hears the phase it
 *  closed in that phase's own units, so a cancel leaves the plan at the next phase, where the next
 *  call starts; a move no stop skips runs to its end first, and a task's failure ends the call.
 *  The scratch is this call's, from @ref prepare_refinement, while the plan keeps whatever the
 *  later calls need.
 */
template <typename plan_type_, typename shard_type_, typename metric_type_, typename selector_type_>
util::status_t drive_refinement(plan_type_& plan, thread_scratch<shard_type_, metric_type_>& scratch,
                                selector_type_ const& selector, progress_ticks_t& ticks,
                                util::ram_executor_t& executor) noexcept {
    using count_ref_t = typename shard_type_::template atomic_ref<std::size_t>;
    // Every task bumps it, so it takes its own line, away from the locals every thread reads.
    alignas(util::default_alignment_k) std::size_t taken = plan.taken;
    while (true) {
        auto const tasks = plan.tasks();
        util::status_t refusal = util::status_code_t::success_k;
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            count_ref_t const next {taken};
            for (std::size_t position = next.fetch_add(1, std::memory_order_relaxed); position < tasks.size();
                 position = next.fetch_add(1, std::memory_order_relaxed))
                keep_first_failure<shard_type_::template atomic_ref>(
                    refusal, plan.run(tasks[position], scratch.metrics[thread], scratch.traversals[thread], selector));
        };
        if (util::status_t const forked = executor.for_threads(drain); failed(forked)) return forked;
        if (failed(refusal)) return refusal;
        plan.taken = taken = 0;
        std::size_t const completed = plan.completed(), total = plan.total();
        bool const following = plan.advance();
        bool moving = false;
        if constexpr (requires { plan.moving(); }) moving = plan.moving();
        if (total && ticks(completed, total) == util::continuation_t::cancel_k && !moving)
            return util::status_code_t::cancelled_k;
        if (!following) return util::status_code_t::success_k;
    }
}

/** What one unit of @p metric_type_'s distance is worth on the C scale: one, unless a modality's
 *  fixed-point metric, or a GPU index scoring with one, specializes it so that its full scale
 *  reaches the value one. */
template <typename metric_type_>
inline constexpr usearch_distance_t distance_unit = 1;

/** Hands @p body a callable building the key predicate an executor thread asks, which a walk asks
 *  about its nodes through their records: the dummy without a predicate, a cheap one asked a quad
 *  at a time, and an expensive one over that thread's own buffers, reserved for rounds of
 *  @p round_capacity candidates gathered beside @p payload_type_. */
template <typename payload_type_, typename body_type_>
util::status_t with_key_predicates(predicate_t const& predicate, std::size_t round_capacity,
                                   util::ram_executor_t& executor, body_type_&& body) noexcept {
    if (std::holds_alternative<no_predicate_t>(predicate)) {
        body([](std::size_t) noexcept { return util::dummy_predicate_t {}; });
        return util::status_code_t::success_k;
    }
    if (cheap_predicate_t const* const cheap = std::get_if<cheap_predicate_t>(&predicate)) {
        body([&](std::size_t) noexcept { return *cheap; });
        return util::status_code_t::success_k;
    }
    expensive_predicates<payload_type_> expensive {std::get<expensive_predicate_t>(predicate).predicate};
    if (util::status_t const opened = open_predicate(expensive, executor); failed(opened)) return opened;
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        if (util::status_t const reserved = util::reserve_predicate(expensive.of_thread(thread), round_capacity);
            failed(reserved))
            return reserved;
    body([&](std::size_t thread) noexcept -> expensive_predicate<payload_type_>& {
        return expensive.of_thread(thread);
    });
    return util::status_code_t::success_k;
}

/** One snapshot of every shard's watermark of @p indexes in @p watermarks, sized again for a shard
 *  generation opened meanwhile, so every walk of a call sees each batch on all shards or none. */
template <typename indexes_type_, typename watermarks_type_>
util::status_t snapshot_shards(indexes_type_ const& indexes, watermarks_type_& watermarks) noexcept {
    while (true) {
        if (failed(watermarks.resize_uninitialized(indexes.shards_count())))
            return util::status_code_t::out_of_memory_k;
        util::status_t const taken = indexes.snapshot(to_writable_span(watermarks)).status;
        if (taken != util::status_code_t::shape_mismatch_k) return taken;
    }
}

/**
 *  @brief Searches @p count queries on the executor's threads, each built by @p query_of, which
 *      hands the query it prepared to a continuation, and writes every query's matches in
 *      @p outputs.
 *
 *  A predicate narrows each query to the entries it admits: a walk routes past the rest until the
 *  admitted ones fill its breadth, and a scan asks only about entries that would enter the ranking.
 *  Every query walks each shard of @p indexes below one snapshot taken for the whole call, so a
 *  batch counts in every query on all its shards or none, and writes the merged ranking. A handler
 *  answering cancel stops the queries no thread has taken yet. Each query's walks start from the
 *  entries under its row of the optional @p hints, which a scan ignores. A walk by the metric
 *  @p make_seeding builds, where it builds one, goes first on every shard and seeds the walk that
 *  follows by the canonical metric.
 */
template <typename indexes_type_, typename make_metric_type_, typename query_of_type_,
          typename make_seeding_type_ = util::unseeded_factory_t>
util::expected<traversal_stats_t> search_entries(indexes_type_& indexes, std::size_t count, std::size_t connectivity,
                                                 make_metric_type_&& make_metric, query_of_type_&& query_of,
                                                 matches_outputs_t& outputs, search_config_t search,
                                                 predicate_t const& predicate, util::ram_executor_t& executor,
                                                 progress_ticks_t& ticks, usearch_matches_view_t const* hints = nullptr,
                                                 make_seeding_type_&& make_seeding = {}) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using distance_t = typename indexes_type_::distance_t;
    using metric_t = made_metric<make_metric_type_>;
    using traversal_context_t = typename thread_scratch<shard_t, metric_t>::traversal_context_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the merges resolve them");
    std::size_t const wanted = outputs.wanted, expansion = search.expansion_factor;
    // Pools narrower than the ranking they fill would short every query.
    std::size_t const breadth = util::max_of(expansion ? expansion : connectivity * 2, wanted);
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (util::status_t const prepared = scratch.prepare(executor, executor.size(), connectivity, breadth,
                                                        indexes.size(), make_metric);
        failed(prepared))
        return {{}, prepared};
    seeding_scratch<made_metric<make_seeding_type_>> seeding {executor};
    if (util::status_t const prepared = seeding.prepare(executor, executor.size(), scratch.round_capacity,
                                                        make_seeding);
        failed(prepared))
        return {{}, prepared};
    typename indexes_type_::search_config_t const config {breadth};
    thread_allocator_t const caller {executor.arena(0)};
    util::growing_buffer<node_id_t, thread_allocator_t> watermarks {caller};
    util::growing_buffer<distance_t, thread_allocator_t> distances {caller};
    if (failed(distances.resize_uninitialized(executor.size() * wanted)))
        return {{}, util::status_code_t::out_of_memory_k};
    if (util::status_t const taken = snapshot_shards(indexes, watermarks); failed(taken)) return {{}, taken};
    bool const scanning = search.traversal == util::traversal_t::scan_k;
    util::status_t searched = util::status_code_t::success_k;
    util::status_t const asked = with_key_predicates<typename shard_t::scored_node_t>(
        predicate, scratch.round_capacity, executor,
        [&]<typename admits_of_type_>(admits_of_type_&& admits_of) noexcept {
            using admits_t = std::invoke_result_t<admits_of_type_&, std::size_t>;
            util::status_t const forked = drain_in_order<indexes_type_::template atomic_ref>(
                executor, count, [&](std::size_t query_index, std::size_t thread) noexcept {
                    if (failed(typename indexes_type_::template atomic_ref<util::status_code_t>(searched.code)
                                   .load(std::memory_order_relaxed)))
                        return;
                    traversal_context_t& context = scratch.traversals[thread];
                    // Landmarks follow the query, not its thread, so walks repeat run to run.
                    context.random_generator = util::seeded_stream(util::u64_t(query_index) + 1);
                    metric_t& metric = scratch.metrics[thread];
                    admits_t admits = admits_of(thread);
                    std::span<key_t> const keys {reinterpret_cast<key_t*>(outputs.keys) + query_index * wanted, wanted};
                    std::span<distance_t> const scores {distances.data() + thread * wanted, wanted};
                    std::span<key_t const> const query_hints =
                        hints ? std::span<key_t const> {reinterpret_cast<key_t const*>(hints->keys) +
                                                            query_index * hints->columns,
                                                        hints->counts[query_index]}
                              : std::span<key_t const> {};
                    util::status_t const answered = query_of(query_index, thread, [&](auto const& query) noexcept {
                        auto const [found, status] =
                            scanning
                                ? indexes.scan(query, to_span(watermarks), keys, scores, admits, metric, context)
                                : indexes.search_if(query, to_span(watermarks), keys, scores, config, admits, metric,
                                                    context, query_hints, seeding.seeding_of(thread, query));
                        if (failed(status)) return status;
                        for (std::size_t match = 0; match != found; ++match)
                            outputs.scores[query_index * wanted + match] =
                                static_cast<usearch_distance_t>(scores[match]) * distance_unit<metric_t>;
                        outputs.counts[query_index] = found;
                        return status;
                    });
                    keep_first_failure<indexes_type_::template atomic_ref>(searched, answered);
                    if (ticks(query_index) == util::continuation_t::cancel_k)
                        keep_first_failure<indexes_type_::template atomic_ref>(searched,
                                                                               util::status_code_t::cancelled_k);
                });
            keep_first_failure<indexes_type_::template atomic_ref>(searched, forked);
        });
    if (failed(asked)) return {{}, asked};
    if (failed(searched)) return {{}, searched};
    return {scratch.telemetry(), util::status_code_t::success_k};
}

/** The entries @p indexes stores under @p key into @p holders and @p ids, parallel: shard by shard
 *  in placement order, and ascending within each shard, which is insertion order there. */
template <typename indexes_type_>
void entries_under(indexes_type_ const& indexes, usearch_key_t key,
                   util::growing_buffer<typename indexes_type_::index_t const*, thread_allocator_t>& holders,
                   util::growing_buffer<typename indexes_type_::node_id_t, thread_allocator_t>& ids) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using node_id_t = typename indexes_type_::node_id_t;
    std::size_t const entries = indexes.count(uuid_of(key));
    if (!entries || failed(holders.reserve_exactly(entries)) || failed(ids.reserve_exactly(entries))) return;
    indexes.find_each(uuid_of(key), [&](shard_t const& shard, node_id_t node_id) noexcept {
        if (ids.size() == ids.capacity()) return;
        holders.push_back_reserved(&shard);
        ids.push_back_reserved(node_id);
    });
    // Each shard's entries arrive together, so sorting every run restores its insertion order.
    for (std::size_t first = 0, last = 0; first != ids.size(); first = last) {
        while (last != ids.size() && holders[last] == holders[first]) ++last;
        util::partial_sort(to_writable_span(ids).subspan(first, last - first), last - first);
    }
}

/**
 *  @brief Ranks @p count queries on an NVMe view a progress step at a time, each step one call of
 *      the view with the walks of many queries in flight on every thread of @p executor, and writes
 *      every query's matches in @p outputs; a handler answering cancel stops the steps not begun.
 *
 *  Query @c i is built by @p query_of, and @p relation tests the stored values a query would admit;
 *  a per-thread relation family is rebuilt every step, as every step rewinds the arenas. A
 *  predicate narrows each query to the entries it admits, while every entry still routes the walks:
 *  a cheap one is asked a quad at a time, and an expensive one once per completion batch of each
 *  executor thread's walks, or once per staged round of each thread's scan. Walks by the metric
 *  @p make_seeding builds, where it builds one, seed every walk, and a scan ignores them.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_, typename relation_type_,
          typename make_seeding_type_ = util::unseeded_factory_t>
util::expected<traversal_stats_t> rank_nvme_queries(index_type_& index, std::size_t count,
                                                    make_metric_type_& make_metric, query_of_type_& query_of,
                                                    relation_type_& relation, matches_outputs_t& outputs,
                                                    search_config_t search, predicate_t const& predicate,
                                                    nvme_executor_t& executor, progress_ticks_t& ticks,
                                                    make_seeding_type_&& make_seeding = {}) noexcept {
    using key_t = typename index_type_::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walks resolve them");
    std::size_t const wanted = outputs.wanted;
    // Pools narrower than the ranking they fill would short every query.
    typename index_type_::search_config_t const config {
        util::max_of(index.non_zero_expansion_factor({search.expansion_factor}), wanted)};
    bool const scanning = search.traversal == util::traversal_t::scan_k;
    util::dummy_predicate_t no_predicate;
    traversal_stats_t telemetry;
    std::size_t const step = ticks.slice();
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = util::min_of(step, count - first);
        std::span<key_t> const keys {reinterpret_cast<key_t*>(outputs.keys) + first * wanted, taking * wanted};
        std::span<usearch_distance_t> const scores {outputs.scores + first * wanted, taking * wanted};
        std::span<std::size_t> const counts {outputs.counts + first, taking};
        auto step_query_of = [&](std::size_t query_index, thread_allocator_t const& allocator) noexcept {
            return query_of(first + query_index, allocator);
        };
        if constexpr (requires { relation.prepare(executor); })
            if (util::status_t const prepared = relation.prepare(executor); failed(prepared))
                return {telemetry, prepared};
        auto rank = [&]<typename predicate_type_>(predicate_type_& admits) noexcept {
            if (scanning)
                return index.scan(taking, make_metric, step_query_of, keys, scores, counts, admits, relation, executor);
            return index.search(taking, make_metric, step_query_of, keys, scores, counts, config, admits, relation,
                                executor, make_seeding);
        };
        cheap_predicate_t cheap {};
        auto rank_admitted = [&]() noexcept {
            if (std::holds_alternative<no_predicate_t>(predicate)) return rank(no_predicate);
            if (std::holds_alternative<cheap_predicate_t>(predicate))
                return rank(cheap = std::get<cheap_predicate_t>(predicate));
            if (scanning) {
                // The scan opens the family after its rewind, reserving it for its widest batch.
                expensive_predicates<typename index_type_::scan_payload_t> expensive {
                    std::get<expensive_predicate_t>(predicate).predicate};
                return index.scan(taking, make_metric, step_query_of, keys, scores, counts, expensive, relation,
                                  executor);
            }
            expensive_predicates<typename index_type_::walk_payload_t> expensive {
                std::get<expensive_predicate_t>(predicate).predicate};
            return index.search(taking, make_metric, step_query_of, keys, scores, counts, config, expensive, relation,
                                executor, make_seeding);
        };
        auto const [ranked, ranked_status] = rank_admitted();
        if (failed(ranked_status)) return {telemetry, ranked_status};
        telemetry.computed_distances += ranked.computed_distances;
        telemetry.visited_members += ranked.visited_members;
        telemetry.restarts += ranked.restarts;
        telemetry.saturated_walks += ranked.saturated_walks;
        telemetry.computed_distances_in_seeding += ranked.computed_distances_in_seeding;
        first += taking;
        if (ticks(first, count) == util::continuation_t::cancel_k) return {telemetry, util::status_code_t::cancelled_k};
    }
    return {telemetry, util::status_code_t::success_k};
}

/**
 *  @brief Searches @p count queries on an NVMe view, the walks of many in flight on every thread of
 *      @p executor, each query built by @p query_of as the view takes it, and writes every query's
 *      matches in @p outputs; progress reports a step at a time, as @ref rank_nvme_queries ranks
 *      them.
 *
 *  A predicate narrows each query to the entries it admits, while every entry still routes the
 *  walks. Walks by the metric @p make_seeding builds, where it builds one, seed every walk.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_,
          typename make_seeding_type_ = util::unseeded_factory_t>
util::expected<traversal_stats_t> search_entries(index_type_& index, std::size_t count, make_metric_type_&& make_metric,
                                                 query_of_type_&& query_of, matches_outputs_t& outputs,
                                                 search_config_t search, predicate_t const& predicate,
                                                 nvme_executor_t& executor, progress_ticks_t& ticks,
                                                 make_seeding_type_&& make_seeding = {}) noexcept {
    using metric_t = made_metric<make_metric_type_>;
    std::size_t const wanted = outputs.wanted;
    util::dummy_predicate_t relation;
    auto const [ranked, ranked_status] = rank_nvme_queries(index, count, make_metric, query_of, relation, outputs,
                                                           search, predicate, executor, ticks, make_seeding);
    if (failed(ranked_status)) return {ranked, ranked_status};
    for (std::size_t query = 0; query != count; ++query)
        for (std::size_t match = 0; match != outputs.counts[query]; ++match)
            outputs.scores[query * wanted + match] *= distance_unit<metric_t>;
    return {ranked, ranked_status};
}

/** Hands every value an NVMe view stores under @p key to @p on_value, as <tt>on_value(position,
 *  rank, thread, view)</tt> with @c rank in insertion order: one scan of the key columns counts
 *  them, handing the count to @p on_count, as <tt>on_count(entries) → status</tt>, before one more
 *  reads them. */
template <typename index_type_, typename on_count_type_, typename on_value_type_>
util::status_t values_under(index_type_ const& index, usearch_key_t key, nvme_executor_t& executor,
                            on_count_type_&& on_count, on_value_type_&& on_value) noexcept {
    std::array<util::uuid128_t, 1> const identities {uuid_of(key)};
    std::array<std::size_t, 1> entries {};
    if (util::status_t const counted = index.count(identities, entries, executor); failed(counted)) return counted;
    std::size_t const held = entries[0];
    if (util::status_t const prepared = on_count(held); failed(prepared) || !held) return prepared;
    return index.get(identities, entries, held, executor, on_value);
}

/** Copies @p entry, valid only during one read, into thread @p thread 's arena as @p kept, which
 *  outlives the read until the executor's next reservation. */
inline util::status_t keep_entry(nvme_executor_t& executor, std::size_t thread, util::bytes_view_t entry,
                                 util::bytes_view_t& kept) noexcept {
    kept = {};
    if (entry.empty()) return util::status_code_t::success_k;
    util::byte_t* const copy = reinterpret_cast<util::byte_t*>(
        executor.arena(thread).allocate(entry.size(), util::default_alignment_k));
    if (!copy) return util::status_code_t::out_of_memory_k;
    std::memcpy(copy, entry.data(), entry.size());
    kept = {copy, entry.size()};
    return util::status_code_t::success_k;
}

/** Where @p variant computes, its first shard's device or the host, where a call's arena must
 *  live. */
template <typename variant_type_>
arena_residency_t arena_residency_of(variant_type_ const& variant) noexcept {
    if constexpr (is_vram_index<variant_type_>)
        return vram_residency(static_cast<std::size_t>(variant.index.first_device()));
    else return ram_residency_k;
}

/**
 *  @brief Samples @p indexes below one snapshot on the calling thread, scored by the metric
 *      @p make_metric builds over the executor's first arena, placed by @p coordinates_of, and
 *      writes each parent's distance in the units search reports.
 *
 *  The walks are as wide as @p connectivity asks; the plan itself never scans. Where
 *  @p make_seeding builds a metric, every greedy walk follows its sketches first, as the walks of a
 *  seeded search do.
 */
template <typename indexes_type_, typename make_metric_type_, typename coordinates_type_ = util::no_fixed_coordinates_t,
          typename make_seeding_type_ = util::unseeded_factory_t>
util::expected<util::sample_result_t> sample_entries(indexes_type_& indexes, std::size_t connectivity,
                                                     make_metric_type_&& make_metric, basic_sample_inputs_t inputs,
                                                     basic_sample_outputs_t outputs, util::sample_config_t config,
                                                     util::ram_executor_t& executor,
                                                     coordinates_type_ const& coordinates_of = {},
                                                     make_seeding_type_&& make_seeding = {}) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using metric_t = made_metric<make_metric_type_>;
    using seeding_metric_t = made_metric<make_seeding_type_>;
    thread_scratch<shard_t, metric_t> scratch {executor};
    // The draw runs on the calling thread alone, so only its context and metrics are built.
    if (util::status_t const prepared = scratch.prepare(executor, 1, connectivity, connectivity * 2, config.seed,
                                                        make_metric);
        failed(prepared))
        return {{}, prepared};
    seeding_scratch<seeding_metric_t> seeding {executor};
    if (util::status_t const prepared = seeding.prepare(executor, 1, scratch.round_capacity, make_seeding);
        failed(prepared))
        return {{}, prepared};
    util::growing_buffer<node_id_t, thread_allocator_t> watermarks {thread_allocator_t {executor.arena(0)}};
    if (util::status_t const taken = snapshot_shards(indexes, watermarks); failed(taken)) return {{}, taken};
    auto const [sampled, sampled_status] = indexes.sample(inputs, to_span(watermarks), outputs, config,
                                                          scratch.metrics[0], scratch.traversals[0], coordinates_of,
                                                          seeding.metric_of(0));
    if (failed(sampled_status)) return {sampled, sampled_status};
    for (util::f32_t& distance : outputs.parent_distances.first(sampled.written))
        distance = static_cast<util::f32_t>(distance * distance_unit<metric_t>);
    return {sampled, sampled_status};
}

/** Samples an NVMe view through its engines, scored by the metric @p make_metric builds over a
 *  thread's arena against the stored values @p query_of turns into queries, placed by
 *  @p coordinates_of, and writes each parent's distance in the units search reports. Where
 *  @p make_seeding builds a metric, every greedy walk follows its sketches first. */
template <typename variant_type_, typename make_metric_type_, typename query_of_type_,
          typename coordinates_type_ = util::no_fixed_coordinates_t,
          typename make_seeding_type_ = util::unseeded_factory_t>
    requires is_nvme_index<variant_type_>
util::expected<util::sample_result_t> sample_entries(variant_type_& variant, make_metric_type_&& make_metric,
                                                     query_of_type_&& query_of, basic_sample_inputs_t inputs,
                                                     basic_sample_outputs_t outputs, util::sample_config_t config,
                                                     nvme_executor_t& executor,
                                                     coordinates_type_ const& coordinates_of = {},
                                                     make_seeding_type_&& make_seeding = {}) noexcept {
    using metric_t = made_metric<make_metric_type_>;
    auto const [sampled, sampled_status] = variant.index.sample(inputs, outputs, config, make_metric, query_of,
                                                                executor, coordinates_of, make_seeding);
    if (failed(sampled_status)) return {sampled, sampled_status};
    for (util::f32_t& distance : outputs.parent_distances.first(sampled.written))
        distance = static_cast<util::f32_t>(distance * distance_unit<metric_t>);
    return {sampled, sampled_status};
}

#pragma endregion Fan Outs

#pragma region Lookups and Cursors

/** Whether @p traits_type_ measures a key's entries from the values of @p value_type_ a variant of
 *  @p variant_type_ holds, which a count then reads. */
template <typename traits_type_, typename variant_type_, typename stats_type_, typename value_type_>
concept a_tallying_traits = requires(variant_type_ const& variant, stats_type_& stats, value_type_ const& value) {
    variant_bodies<traits_type_>::tally(variant, stats, value);
};

/** Whether @p traits_type_ reads a stored value of a device variant of @p variant_type_ back as it
 *  writes one. */
template <typename traits_type_, typename variant_type_>
concept a_variant_keeping_stored = requires(variant_type_ const& variant,
                                            typename variant_bodies<traits_type_>::kept_t& kept) {
    variant_bodies<traits_type_>::keep_stored(variant, util::bytes_view_t {}, kept);
};

/** Adds @p value of @p variant to @p stats, from any thread, where @p traits_type_ measures such
 *  values. */
template <typename traits_type_, typename variant_type_, typename stats_type_, typename value_type_>
void tally_entry(variant_type_ const& variant, stats_type_& stats, value_type_ const& value) noexcept {
    if constexpr (a_tallying_traits<traits_type_, variant_type_, stats_type_, value_type_>)
        variant_bodies<traits_type_>::tally(variant, stats, value);
}

/** Counts the entries a RAM variant stores under @p key into @p stats, tallying their sizes where
 *  @p traits_type_ measures any. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_ram_index<variant_type_>
util::status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats,
                             util::ram_executor_t&) noexcept {
    using shard_t = typename variant_type_::shard_t;
    variant.index.find_each(uuid_of(key), [&](shard_t const& shard, typename shard_t::node_id_t node) noexcept {
        ++stats.entries;
        tally_entry<traits_type_>(variant, stats, shard.published_value_of(node));
    });
    return util::status_code_t::success_k;
}

/** Counts the entries an NVMe view stores under @p key into @p stats, reading them only where
 *  @p traits_type_ measures their sizes. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_nvme_index<variant_type_>
util::status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats,
                             nvme_executor_t& executor) noexcept {
    using view_t = typename variant_type_::index_t::codec_t::view_t;
    if constexpr (a_tallying_traits<traits_type_, variant_type_, stats_type_, view_t>) {
        auto take_count = [&](std::size_t entries) noexcept {
            return stats.entries = entries, util::status_code_t::success_k;
        };
        auto measure = [&](std::size_t, std::size_t, std::size_t, view_t const& value) noexcept {
            tally_entry<traits_type_>(variant, stats, value);
        };
        return values_under(variant.index, key, executor, take_count, measure);
    }
    else {
        std::array<util::uuid128_t, 1> const identities {uuid_of(key)};
        std::array<std::size_t, 1> entries {};
        util::status_t const counted = variant.index.count(identities, entries, executor);
        stats.entries = entries[0];
        return counted;
    }
}

/** Writes the entries a RAM variant stores under @p key to @p outputs, carved from @p memory, in
 *  insertion order, as @p traits_type_ writes values; answers how many. */
template <typename traits_type_, typename variant_type_>
    requires is_ram_index<variant_type_>
util::expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                        typename traits_type_::get_outputs_t outputs,
                                        util::growing_arena<util::forkunion_allocator_t>& memory,
                                        util::ram_executor_t& executor) noexcept {
    if (util::status_t const rewound = executor.reserve(0); failed(rewound)) return {0, rewound};
    using shard_t = typename variant_type_::shard_t;
    thread_allocator_t const allocator {executor.arena(0)};
    util::growing_buffer<shard_t const*, thread_allocator_t> holders {allocator};
    util::growing_buffer<typename shard_t::node_id_t, thread_allocator_t> ids {allocator};
    entries_under(variant.index, key, holders, ids);
    if (ids.empty()) return {0, util::status_code_t::success_k};
    auto value_at = [&](std::size_t entry) noexcept ->
        typename shard_t::value_t const& { return holders[entry]->published_value_of(ids[entry]); };
    return {ids.size(), variant_bodies<traits_type_>::write_values(variant, ids.size(), value_at, memory, outputs)};
}

/** Writes the entries an NVMe view stores under @p key to @p outputs, carved from @p memory, in
 *  insertion order; the values land in its threads' arenas first, as their sizes are unknown until
 *  read. */
template <typename traits_type_, typename variant_type_>
    requires is_nvme_index<variant_type_>
util::expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                        typename traits_type_::get_outputs_t outputs,
                                        util::growing_arena<util::forkunion_allocator_t>& memory,
                                        nvme_executor_t& executor) noexcept {
    using kept_t = typename variant_bodies<traits_type_>::kept_t;
    std::size_t held = 0;
    kept_t* kept = nullptr;
    auto carve_entries = [&](std::size_t entries) noexcept {
        held = entries;
        if (!held) return util::status_code_t::success_k;
        kept = carve<kept_t>(memory, held);
        return kept ? util::status_code_t::success_k : util::status_code_t::out_of_memory_k;
    };
    util::status_t keeping = util::status_code_t::success_k;
    auto keep = [&](std::size_t, std::size_t rank, std::size_t thread, auto const& value) noexcept {
        keep_first_failure<variant_type_::index_t::template atomic_ref>(
            keeping, variant_bodies<traits_type_>::keep(executor, thread, value, kept[rank]));
    };
    if (util::status_t const read = values_under(variant.index, key, executor, carve_entries, keep); util::failed(read))
        return {0, read};
    if (failed(keeping)) return {0, keeping};
    if (!held) return {0, util::status_code_t::success_k};
    auto kept_at = [&](std::size_t entry) noexcept -> kept_t const& { return kept[entry]; };
    return {held, variant_bodies<traits_type_>::write_values(variant, held, kept_at, memory, outputs)};
}

/** Pulls the next keys of a variant's walk into @p memory, gathering their links into @p links
 *  where named; an NVMe walk reads through the one @p executor, the others take none. */
template <typename variant_type_, typename... executor_types_>
util::expected<std::span<usearch_key_t const>> pull_entries(variant_type_& variant,
                                                            typename variant_type_::index_t::cursor_t& walk,
                                                            links_outputs_t* links, std::size_t wanted,
                                                            util::growing_arena<util::forkunion_allocator_t>& memory,
                                                            executor_types_&... executor) noexcept {
    using key_t = typename variant_type_::index_t::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk reads them");
    // No more entries remain than every shard publishes now, so the carve never outgrows the read.
    std::size_t const taking = util::min_of(variant.index.size(), wanted);
    usearch_key_t* const keys = carve<usearch_key_t>(memory, taking);
    if (taking && !keys) return {{}, util::status_code_t::out_of_memory_k};
    auto const [written, status] = walk.pull(variant.index, std::span<key_t> {reinterpret_cast<key_t*>(keys), taking},
                                             links ? &links->keys : nullptr, links ? &links->offsets : nullptr,
                                             executor...);
    return {{keys, written}, status};
}

#pragma endregion Lookups and Cursors

#if USEARCH_ARCH_CUDA_
#pragma region GPU Variants

/** Reserves @p entries nodes of @p connectivity neighbors and @p bytes of values over every shard
 *  of a GPU variant's @p index, freezing the connectivity. */
template <typename indexes_type_>
util::status_t vram_nvidia_reserve(indexes_type_& index, std::size_t entries, std::size_t connectivity,
                                   std::size_t bytes) noexcept {
    typename indexes_type_::reserve_capacity_t config {};
    config.capacity = entries;
    config.connectivity = static_cast<util::connectivity_t>(connectivity);
    config.bytes = bytes;
    return index.reserve(config).status;
}

/** Writes the live shards of @p index under @p path with @p config 's provenance, through @p codec,
 *  the wire format the RAM variants share: a lone live shard's monolith, or one per live shard
 *  beside the manifest @p path names. */
template <typename indexes_type_, typename codec_type_>
util::status_t vram_nvidia_save(indexes_type_& index, std::string_view path, codec_type_ const& codec,
                                util::save_config_t const& config, std::string_view metric,
                                util::cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
    return index.save(path, codec, executor, config, metric, ticks);
}

/** Inserts the entries of @p tape under the keys of @p keys_view as one batch over the shards of
 *  @p index, freezing @p connectivity first unless a reservation did. */
template <typename indexes_type_>
util::expected<traversal_stats_t> vram_nvidia_insert(indexes_type_& index, usearch_keys_view_t keys_view,
                                                     util::tape_view<util::byte_t const> tape, std::size_t connectivity,
                                                     std::size_t expansion, progress_ticks_t& ticks,
                                                     util::cuda_executor_t& executor) noexcept {
    std::size_t const count = tape.size();
    if (!index.capacity())
        if (util::status_t const reserved = vram_nvidia_reserve(index, count, connectivity,
                                                                tape.offsets[count] - tape.offsets[0]);
            failed(reserved))
            return {{}, reserved};
    util::growing_buffer<util::uuid128_t, util::forkunion_allocator_t> keys {
        util::forkunion_allocator_t {util::no_memory_domain_k}};
    if (failed(keys.reserve_exactly(count))) return {{}, util::status_code_t::out_of_memory_k};
    for (std::size_t offset = 0; offset != count; ++offset) keys.push_back_reserved(uuid_of(key_at(keys_view, offset)));
    typename indexes_type_::insert_config_t const config {static_cast<util::u32_t>(expansion)};
    auto const [inserted, status] = index.insert(to_span(keys), tape, executor, config);
    if (failed(status)) return {{}, status};
    [[maybe_unused]] util::continuation_t const heard = ticks(count - 1);
    return {{.computed_distances = inserted.computed_distances,
             .visited_members = inserted.visited_members,
             .restarts = inserted.restarts},
            util::status_code_t::success_k};
}

/** Searches the queries of @p queries a progress step at a time, each step one merged search over
 *  every shard of @p index, outputs each query's matches in @p outputs on host pages; a handler
 *  answering cancel stops the steps not begun. A search wanting no matches answers before reading
 *  a query. */
template <typename indexes_type_>
util::expected<traversal_stats_t> vram_nvidia_search(indexes_type_& index, util::tape_view<util::byte_t const> queries,
                                                     std::size_t wanted, search_config_t search,
                                                     matches_outputs_t& outputs, progress_ticks_t& ticks,
                                                     util::cuda_executor_t& executor) noexcept {
    using face_t = typename indexes_type_::index_t;
    std::size_t const count = queries.size();
    if (!wanted) {
        for (std::size_t query = 0; query != count; ++query) outputs.counts[query] = 0;
        return {{}, util::status_code_t::success_k};
    }
    using key_t = util::uuid128_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land whole in the caller's outputs");
    typename face_t::search_config_t const config {static_cast<util::u32_t>(search.expansion_factor),
                                                   static_cast<util::u32_t>(wanted), search.traversal};
    std::size_t const step = ticks.slice();
    traversal_stats_t total;
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = util::min_of(step, count - first);
        std::size_t const first_match = first * wanted, matches = taking * wanted;
        usearch_distance_t* const scores = outputs.scores + first_match;
        auto const [found, status] = index.search(
            queries.subview(first, taking),
            nk::matrix_span<key_t> {reinterpret_cast<key_t*>(outputs.keys + first_match), {taking, wanted}},
            nk::matrix_span<usearch_distance_t> {scores, {taking, wanted}},
            std::span<std::size_t> {outputs.counts + first, taking}, config, executor);
        if (failed(status)) return {total, status};
        total.visited_members += found.visited_members;
        total.computed_distances += found.computed_distances;
        total.computed_distances_in_seeding += found.computed_distances_in_seeding;
        total.restarts += found.restarts;
        for (std::size_t match = 0; match != matches; ++match) scores[match] *= distance_unit<face_t>;
        first += taking;
        if (ticks(first, count) == util::continuation_t::cancel_k) return {total, util::status_code_t::cancelled_k};
    }
    return {total, util::status_code_t::success_k};
}

/** Counts the entries a GPU variant stores under @p key on every shard's host key map; their
 *  values stay on the devices, so the extents stay zero. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_vram_index<variant_type_>
util::status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats,
                             util::cuda_executor_t&) noexcept {
    stats.entries = variant.index.count(uuid_of(key));
    return util::status_code_t::success_k;
}

/** Whether a graph of @p graph_type_ launches the gather of its stored values into a tape. */
template <typename graph_type_>
concept a_graph_gathering_tapes = requires(graph_type_& graph,
                                           util::cuda_scratch_t::traversal_context_t const& context) {
    graph.launch_gather_tape_(std::span<typename graph_type_::node_id_t const> {}, nullptr, std::span<util::u64_t> {},
                              context);
};

/**
 *  @brief Writes the entries a GPU variant stores under @p key to @p outputs, carved from
 *      @p memory: each shard's node list crosses down at its node width, @c gather_tape_k packs
 *      its values back to back in the shard's output staging, and one copy brings the tape back,
 *      where the modality reads each stored value back as it writes it.
 *
 *  A modality whose bodies spell no @c keep_stored cannot read its stored bytes back, and a face
 *  that keeps its gather to its sharded index passes none, so both answer @c unsupported_k; shards
 *  on a device @p executor drives no stream on refuse with @c config_mismatch_k.
 */
template <typename traits_type_, typename variant_type_>
    requires is_vram_index<variant_type_>
util::expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                        typename traits_type_::get_outputs_t outputs,
                                        util::growing_arena<util::forkunion_allocator_t>& memory,
                                        util::cuda_executor_t& executor) noexcept {
    using bodies_t = variant_bodies<traits_type_>;
    using kept_t = typename bodies_t::kept_t;
    using graph_t = typename variant_type_::index_t::index_t;
    using node_id_t = typename graph_t::node_id_t;
    if constexpr (!a_variant_keeping_stored<traits_type_, variant_type_> || !a_graph_gathering_tapes<graph_t>)
        return {0, util::status_code_t::unsupported_k};
    else {
        util::uuid128_t const identity = uuid_of(key);
        std::size_t const held = variant.index.count(identity);
        if (!held) return {0, util::status_code_t::success_k};
        node_id_t* const listed = carve<node_id_t>(memory, held);
        kept_t* const kept = carve<kept_t>(memory, held);
        if (!listed || !kept) return {0, util::status_code_t::out_of_memory_k};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {0, claim_status};
        std::size_t found = 0;
        for (graph_t& graph : variant.index.live_shards()) {
            std::size_t const first = found;
            graph.find_each(identity, [&](node_id_t node) noexcept {
                if (found != held) listed[found++] = node;
            });
            std::size_t const count = found - first;
            if (!count) continue;
            std::size_t const stream = util::stream_on(executor,
                                                       static_cast<std::size_t>(graph.device_specs().ordinal));
            if (stream == executor.size()) return {0, util::status_code_t::config_mismatch_k};
            util::cuda_api_t const& api = graph.api();
            // Each node's extent comes off the device's offsets column, so the tape is sized first.
            util::u64_t const* const stored_offsets = graph.storage().offsets;
            std::size_t tape_bytes = 0, sized = first;
            for (std::array<util::u64_t, 2> extent {};
                 sized != found &&
                 cuda_copy(api, extent.data(), stored_offsets + listed[sized], sizeof(extent)) == util::cuda_success_k;
                 ++sized)
                tape_bytes += static_cast<std::size_t>(extent[1] - extent[0]);
            if (sized != found) return {0, util::status_code_t::device_k};
            std::size_t const listed_bytes = util::round_up_to_multiple(count * sizeof(node_id_t),
                                                                        util::default_alignment_k);
            std::size_t const offsets_bytes = util::round_up_to_multiple((count + 1) * sizeof(util::u64_t),
                                                                         util::default_alignment_k);
            // The last gather was awaited, so its run rewinds for this carve, which grows it at
            // most once per call; a stream's first context is born on its device.
            std::size_t const staging_bytes = listed_bytes + offsets_bytes + tape_bytes;
            if (util::status_t const reserved = util::reserve_scratch(executor.stream(stream), staging_bytes, 1);
                failed(reserved))
                return {0, reserved};
            if (cuda_set_device(api, graph.device_specs().ordinal) != util::cuda_success_k)
                return {0, util::status_code_t::device_k};
            util::cuda_scratch_t::traversal_context_t& context = executor.stream(stream).memory.contexts[0];
            util::byte_t* const staging = context.allocator().allocate(staging_bytes);
            util::u64_t* const host_offsets = carve<util::u64_t>(memory, count + 1);
            util::byte_t* const host_tape = carve<util::byte_t>(memory, tape_bytes);
            if (!staging || !host_offsets || (tape_bytes && !host_tape))
                return {0, util::status_code_t::out_of_memory_k};
            node_id_t* const device_listed = reinterpret_cast<node_id_t*>(staging);
            util::u64_t* const device_offsets = reinterpret_cast<util::u64_t*>(staging + listed_bytes);
            util::byte_t* const device_tape = staging + listed_bytes + offsets_bytes;
            host_offsets[0] = 0;
            if (cuda_copy(api, device_listed, listed + first, count * sizeof(node_id_t)) != util::cuda_success_k ||
                cuda_copy(api, device_offsets, host_offsets, sizeof(util::u64_t)) != util::cuda_success_k)
                return {0, util::status_code_t::device_k};
            if (util::status_t const gathered = graph.launch_gather_tape_({device_listed, count}, device_tape,
                                                                          {device_offsets, count + 1}, context);
                failed(gathered))
                return {0, gathered};
            if (util::status_t const waited = context.wait(); failed(waited)) return {0, waited};
            if (cuda_copy(api, host_offsets, device_offsets, (count + 1) * sizeof(util::u64_t)) !=
                    util::cuda_success_k ||
                (tape_bytes && cuda_copy(api, host_tape, device_tape, tape_bytes) != util::cuda_success_k))
                return {0, util::status_code_t::device_k};
            for (std::size_t entry = 0; entry != count; ++entry) {
                util::bytes_view_t const stored {
                    host_tape + host_offsets[entry],
                    static_cast<std::size_t>(host_offsets[entry + 1] - host_offsets[entry])};
                if (util::status_t const read = bodies_t::keep_stored(variant, stored, kept[first + entry]);
                    failed(read))
                    return {0, read};
            }
        }
        auto kept_at = [&](std::size_t entry) noexcept -> kept_t const& { return kept[entry]; };
        return {found, bodies_t::write_values(variant, found, kept_at, memory, outputs)};
    }
}

#pragma endregion GPU Variants
#endif

#if USEARCH_ARCH_ROCM_
#pragma region AMD GPU Variants

/** Reserves @p entries nodes of @p connectivity neighbors and @p bytes of values over every shard
 *  of a GPU variant's @p index, freezing the connectivity. */
template <typename indexes_type_>
util::status_t vram_amd_reserve(indexes_type_& index, std::size_t entries, std::size_t connectivity,
                                std::size_t bytes) noexcept {
    typename indexes_type_::reserve_capacity_t config {};
    config.capacity = entries;
    config.connectivity = static_cast<util::connectivity_t>(connectivity);
    config.bytes = bytes;
    return index.reserve(config).status;
}

/** Writes the live shards of @p index under @p path with @p config 's provenance, through @p codec,
 *  the wire format the RAM variants share: a lone live shard's monolith, or one per live shard
 *  beside the manifest @p path names. */
template <typename indexes_type_, typename codec_type_>
util::status_t vram_amd_save(indexes_type_& index, std::string_view path, codec_type_ const& codec,
                             util::save_config_t const& config, std::string_view metric, util::hip_executor_t& executor,
                             progress_ticks_t& ticks) noexcept {
    return index.save(path, codec, executor, config, metric, ticks);
}

/** Inserts the entries of @p tape under the keys of @p keys_view as one batch over the shards of
 *  @p index, freezing @p connectivity first unless a reservation did. */
template <typename indexes_type_>
util::expected<traversal_stats_t> vram_amd_insert(indexes_type_& index, usearch_keys_view_t keys_view,
                                                  util::tape_view<util::byte_t const> tape, std::size_t connectivity,
                                                  std::size_t expansion, progress_ticks_t& ticks,
                                                  util::hip_executor_t& executor) noexcept {
    std::size_t const count = tape.size();
    if (!index.capacity())
        if (util::status_t const reserved = vram_amd_reserve(index, count, connectivity,
                                                             tape.offsets[count] - tape.offsets[0]);
            failed(reserved))
            return {{}, reserved};
    util::growing_buffer<util::uuid128_t, util::forkunion_allocator_t> keys {
        util::forkunion_allocator_t {util::no_memory_domain_k}};
    if (failed(keys.reserve_exactly(count))) return {{}, util::status_code_t::out_of_memory_k};
    for (std::size_t offset = 0; offset != count; ++offset) keys.push_back_reserved(uuid_of(key_at(keys_view, offset)));
    typename indexes_type_::insert_config_t const config {static_cast<util::u32_t>(expansion)};
    auto const [inserted, status] = index.insert(to_span(keys), tape, executor, config);
    if (failed(status)) return {{}, status};
    [[maybe_unused]] util::continuation_t const heard = ticks(count - 1);
    return {{.computed_distances = inserted.computed_distances,
             .visited_members = inserted.visited_members,
             .restarts = inserted.restarts},
            util::status_code_t::success_k};
}

/** Searches the queries of @p queries a progress step at a time, each step one merged search over
 *  every shard of @p index, outputs each query's matches in @p outputs on host pages; a handler
 *  answering cancel stops the steps not begun. A search wanting no matches answers before it reads
 *  any of the queries. */
template <typename indexes_type_>
util::expected<traversal_stats_t> vram_amd_search(indexes_type_& index, util::tape_view<util::byte_t const> queries,
                                                  std::size_t wanted, search_config_t search,
                                                  matches_outputs_t& outputs, progress_ticks_t& ticks,
                                                  util::hip_executor_t& executor) noexcept {
    using face_t = typename indexes_type_::index_t;
    std::size_t const count = queries.size();
    if (!wanted) {
        for (std::size_t query = 0; query != count; ++query) outputs.counts[query] = 0;
        return {{}, util::status_code_t::success_k};
    }
    using key_t = util::uuid128_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land whole in the caller's outputs");
    typename face_t::search_config_t const config {static_cast<util::u32_t>(search.expansion_factor),
                                                   static_cast<util::u32_t>(wanted), search.traversal};
    std::size_t const step = ticks.slice();
    traversal_stats_t total;
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = util::min_of(step, count - first);
        std::size_t const first_match = first * wanted, matches = taking * wanted;
        usearch_distance_t* const scores = outputs.scores + first_match;
        auto const [found, status] = index.search(
            queries.subview(first, taking),
            nk::matrix_span<key_t> {reinterpret_cast<key_t*>(outputs.keys + first_match), {taking, wanted}},
            nk::matrix_span<usearch_distance_t> {scores, {taking, wanted}},
            std::span<std::size_t> {outputs.counts + first, taking}, config, executor);
        if (failed(status)) return {total, status};
        total.visited_members += found.visited_members;
        total.computed_distances += found.computed_distances;
        total.computed_distances_in_seeding += found.computed_distances_in_seeding;
        total.restarts += found.restarts;
        for (std::size_t match = 0; match != matches; ++match) scores[match] *= distance_unit<face_t>;
        first += taking;
        if (ticks(first, count) == util::continuation_t::cancel_k) return {total, util::status_code_t::cancelled_k};
    }
    return {total, util::status_code_t::success_k};
}

/** Counts the entries a GPU variant stores under @p key on every shard's host key map; their
 *  values stay on the devices, so the extents stay zero. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_vram_index<variant_type_>
util::status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats,
                             util::hip_executor_t&) noexcept {
    stats.entries = variant.index.count(uuid_of(key));
    return util::status_code_t::success_k;
}

/**
 *  @brief Writes the entries a GPU variant stores under @p key to @p outputs, carved from
 *      @p memory: each shard's node list crosses down at its node width, @c gather_tape_k packs
 *      its values back to back in the shard's output staging, and one copy brings the tape back,
 *      where the modality reads each stored value back as it writes it.
 *
 *  Shards on a device @p executor drives no stream on refuse with @c config_mismatch_k.
 */
template <typename traits_type_, typename variant_type_>
    requires is_vram_index<variant_type_> && a_variant_keeping_stored<traits_type_, variant_type_>
util::expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                        typename traits_type_::get_outputs_t outputs,
                                        util::growing_arena<util::forkunion_allocator_t>& memory,
                                        util::hip_executor_t& executor) noexcept {
    using bodies_t = variant_bodies<traits_type_>;
    using kept_t = typename bodies_t::kept_t;
    using graph_t = typename variant_type_::index_t::index_t;
    using node_id_t = typename graph_t::node_id_t;
    util::uuid128_t const identity = uuid_of(key);
    std::size_t const held = variant.index.count(identity);
    if (!held) return {0, util::status_code_t::success_k};
    node_id_t* const listed = carve<node_id_t>(memory, held);
    kept_t* const kept = carve<kept_t>(memory, held);
    if (!listed || !kept) return {0, util::status_code_t::out_of_memory_k};
    auto [claim, claim_status] = executor.claim();
    if (failed(claim_status)) return {0, claim_status};
    std::size_t found = 0;
    for (graph_t& graph : variant.index.live_shards()) {
        std::size_t const first = found;
        graph.find_each(identity, [&](node_id_t node) noexcept {
            if (found != held) listed[found++] = node;
        });
        std::size_t const count = found - first;
        if (!count) continue;
        std::size_t const stream = util::stream_on(executor, static_cast<std::size_t>(graph.device_specs().ordinal));
        if (stream == executor.size()) return {0, util::status_code_t::config_mismatch_k};
        util::hip_api_t const& api = graph.api();
        // Each node's extent comes off the device's offsets column, so the tape is sized first.
        util::u64_t const* const stored_offsets = graph.storage().offsets;
        std::size_t tape_bytes = 0;
        for (std::size_t entry = first; entry != found; ++entry) {
            std::array<util::u64_t, 2> extent {};
            if (api.copy(extent.data(), stored_offsets + listed[entry], sizeof(extent), util::hip_copy_default_k) !=
                util::hip_success_k)
                return {0, util::status_code_t::device_k};
            tape_bytes += static_cast<std::size_t>(extent[1] - extent[0]);
        }
        std::size_t const listed_bytes = util::round_up_to_multiple(count * sizeof(node_id_t),
                                                                    util::default_alignment_k);
        std::size_t const offsets_bytes = util::round_up_to_multiple((count + 1) * sizeof(util::u64_t),
                                                                     util::default_alignment_k);
        // The last gather was awaited, so its run rewinds for this carve, which grows it at
        // most once per call; a stream's first context is born on its device.
        std::size_t const staging_bytes = listed_bytes + offsets_bytes + tape_bytes;
        if (util::status_t const reserved = util::reserve_scratch(executor.stream(stream), staging_bytes, 1);
            failed(reserved))
            return {0, reserved};
        if (hip_set_device(api, graph.device_specs().ordinal) != util::hip_success_k)
            return {0, util::status_code_t::device_k};
        std::span<util::hip_scratch_t::traversal_context_t> const stream_traversals = util::to_writable_span(
            executor.stream(stream).memory.contexts);
        util::hip_scratch_t::traversal_context_t& context = stream_traversals[0];
        util::byte_t* const staging = context.allocator().allocate(staging_bytes);
        util::u64_t* const host_offsets = carve<util::u64_t>(memory, count + 1);
        util::byte_t* const host_tape = carve<util::byte_t>(memory, tape_bytes);
        if (!staging || !host_offsets || (tape_bytes && !host_tape)) return {0, util::status_code_t::out_of_memory_k};
        node_id_t* const device_listed = reinterpret_cast<node_id_t*>(staging);
        util::u64_t* const device_offsets = reinterpret_cast<util::u64_t*>(staging + listed_bytes);
        util::byte_t* const device_tape = staging + listed_bytes + offsets_bytes;
        host_offsets[0] = 0;
        if (api.copy(device_listed, listed + first, count * sizeof(node_id_t), util::hip_copy_default_k) !=
                util::hip_success_k ||
            api.copy(device_offsets, host_offsets, sizeof(util::u64_t), util::hip_copy_default_k) !=
                util::hip_success_k)
            return {0, util::status_code_t::device_k};
        if (util::status_t const gathered = graph.launch_gather_tape_({device_listed, count}, device_tape,
                                                                      {device_offsets, count + 1}, context);
            failed(gathered))
            return {0, gathered};
        if (util::status_t const waited = context.wait(); failed(waited)) return {0, waited};
        if (api.copy(host_offsets, device_offsets, (count + 1) * sizeof(util::u64_t), util::hip_copy_default_k) !=
                util::hip_success_k ||
            (tape_bytes &&
             api.copy(host_tape, device_tape, tape_bytes, util::hip_copy_default_k) != util::hip_success_k))
            return {0, util::status_code_t::device_k};
        for (std::size_t entry = 0; entry != count; ++entry) {
            util::bytes_view_t const stored {host_tape + host_offsets[entry],
                                             static_cast<std::size_t>(host_offsets[entry + 1] - host_offsets[entry])};
            if (util::status_t const read = bodies_t::keep_stored(variant, stored, kept[first + entry]); failed(read))
                return {0, read};
        }
    }
    auto kept_at = [&](std::size_t entry) noexcept -> kept_t const& { return kept[entry]; };
    return {found, bodies_t::write_values(variant, found, kept_at, memory, outputs)};
}

/** A modality whose bodies spell no @c keep_stored cannot read its stored bytes back from an AMD
 *  GPU variant, so it answers @c unsupported_k, as on NVIDIA. */
template <typename traits_type_, typename variant_type_>
    requires is_vram_index<variant_type_> && (!a_variant_keeping_stored<traits_type_, variant_type_>)
util::expected<std::size_t> get_entries(variant_type_&, usearch_key_t, typename traits_type_::get_outputs_t,
                                        util::growing_arena<util::forkunion_allocator_t>&,
                                        util::hip_executor_t&) noexcept {
    return {0, util::status_code_t::unsupported_k};
}

#pragma endregion AMD GPU Variants
#endif

#if USEARCH_ARCH_METAL_
#pragma region Apple GPU Variants

/** Reserves @p entries nodes of @p connectivity neighbors and @p bytes of values over every shard
 *  of an Apple GPU variant's @p index, freezing the connectivity, as @ref vram_nvidia_reserve. */
template <typename indexes_type_>
util::status_t vram_apple_reserve(indexes_type_& index, std::size_t entries, std::size_t connectivity,
                                  std::size_t bytes) noexcept {
    typename indexes_type_::reserve_capacity_t config {};
    config.capacity = entries;
    config.connectivity = static_cast<util::connectivity_t>(connectivity);
    config.bytes = bytes;
    return index.reserve(config).status;
}

/** Writes the live shards of @p index under @p path through @p codec, as @ref vram_nvidia_save. */
template <typename indexes_type_, typename codec_type_>
util::status_t vram_apple_save(indexes_type_& index, std::string_view path, codec_type_ const& codec,
                               util::save_config_t const& config, std::string_view metric,
                               util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
    return index.save(path, codec, executor, config, metric, ticks);
}

/** Inserts the entries of @p tape under the keys of @p keys_view as one batch over the shards of
 *  @p index, freezing @p connectivity first unless a reservation did, as
 *  @ref vram_nvidia_insert. */
template <typename indexes_type_>
util::expected<traversal_stats_t> vram_apple_insert(indexes_type_& index, usearch_keys_view_t keys_view,
                                                    util::tape_view<util::byte_t const> tape, std::size_t connectivity,
                                                    std::size_t expansion, progress_ticks_t& ticks,
                                                    util::metal_executor_t& executor) noexcept {
    std::size_t const count = tape.size();
    if (!index.capacity())
        if (util::status_t const reserved = vram_apple_reserve(index, count, connectivity,
                                                               tape.offsets[count] - tape.offsets[0]);
            failed(reserved))
            return {{}, reserved};
    util::growing_buffer<util::uuid128_t, util::forkunion_allocator_t> keys {
        util::forkunion_allocator_t {util::no_memory_domain_k}};
    if (failed(keys.reserve_exactly(count))) return {{}, util::status_code_t::out_of_memory_k};
    for (std::size_t offset = 0; offset != count; ++offset) keys.push_back_reserved(uuid_of(key_at(keys_view, offset)));
    typename indexes_type_::insert_config_t const config {static_cast<util::u32_t>(expansion)};
    auto const [inserted, status] = index.insert(to_span(keys), tape, executor, config);
    if (failed(status)) return {{}, status};
    [[maybe_unused]] util::continuation_t const heard = ticks(count - 1);
    return {{.computed_distances = inserted.computed_distances,
             .visited_members = inserted.visited_members,
             .restarts = inserted.restarts},
            util::status_code_t::success_k};
}

/** Searches the queries of @p queries a progress step at a time over every shard of @p index, the
 *  matches landing in @p outputs in place, as memory is unified; as @ref vram_nvidia_search. */
template <typename indexes_type_>
util::expected<traversal_stats_t> vram_apple_search(indexes_type_& index, util::tape_view<util::byte_t const> queries,
                                                    std::size_t wanted, search_config_t search,
                                                    matches_outputs_t& outputs, progress_ticks_t& ticks,
                                                    util::metal_executor_t& executor) noexcept {
    using face_t = typename indexes_type_::index_t;
    std::size_t const count = queries.size();
    if (!wanted) {
        for (std::size_t query = 0; query != count; ++query) outputs.counts[query] = 0;
        return {{}, util::status_code_t::success_k};
    }
    using key_t = util::uuid128_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land whole in the caller's outputs");
    typename face_t::search_config_t const config {static_cast<util::u32_t>(search.expansion_factor),
                                                   static_cast<util::u32_t>(wanted), search.traversal};
    std::size_t const step = ticks.slice();
    traversal_stats_t total;
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = util::min_of(step, count - first);
        std::size_t const first_match = first * wanted, matches = taking * wanted;
        usearch_distance_t* const scores = outputs.scores + first_match;
        auto const [found, status] = index.search(
            queries.subview(first, taking),
            nk::matrix_span<key_t> {reinterpret_cast<key_t*>(outputs.keys + first_match), {taking, wanted}},
            nk::matrix_span<usearch_distance_t> {scores, {taking, wanted}},
            std::span<std::size_t> {outputs.counts + first, taking}, config, executor);
        if (failed(status)) return {total, status};
        total.visited_members += found.visited_members;
        total.computed_distances += found.computed_distances;
        total.computed_distances_in_seeding += found.computed_distances_in_seeding;
        total.restarts += found.restarts;
        for (std::size_t match = 0; match != matches; ++match) scores[match] *= distance_unit<face_t>;
        first += taking;
        if (ticks(first, count) == util::continuation_t::cancel_k) return {total, util::status_code_t::cancelled_k};
    }
    return {total, util::status_code_t::success_k};
}

/** Counts the entries an Apple GPU variant stores under @p key on every shard's host key map. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_vram_index<variant_type_>
util::status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats,
                             util::metal_executor_t&) noexcept {
    stats.entries = variant.index.count(uuid_of(key));
    return util::status_code_t::success_k;
}

/** Writes the entries an Apple GPU variant stores under @p key to @p outputs, carved from
 *  @p memory: memory is unified, so each shard's values copy back to back on the host straight from
 *  its tape, where the modality reads each stored value back as it writes it. */
template <typename traits_type_, typename variant_type_>
    requires is_vram_index<variant_type_> && a_variant_keeping_stored<traits_type_, variant_type_>
util::expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                        typename traits_type_::get_outputs_t outputs,
                                        util::growing_arena<util::forkunion_allocator_t>& memory,
                                        util::metal_executor_t&) noexcept {
    using bodies_t = variant_bodies<traits_type_>;
    using kept_t = typename bodies_t::kept_t;
    using graph_t = typename variant_type_::index_t::index_t;
    using node_id_t = typename graph_t::node_id_t;
    util::uuid128_t const identity = uuid_of(key);
    std::size_t const held = variant.index.count(identity);
    if (!held) return {0, util::status_code_t::success_k};
    node_id_t* const listed = carve<node_id_t>(memory, held);
    kept_t* const kept = carve<kept_t>(memory, held);
    if (!listed || !kept) return {0, util::status_code_t::out_of_memory_k};
    std::size_t found = 0;
    for (graph_t const& graph : variant.index.live_shards()) {
        std::size_t const first = found;
        graph.find_each(identity, [&](node_id_t node) noexcept {
            if (found != held) listed[found++] = node;
        });
        std::size_t const count = found - first;
        if (!count) continue;
        util::u64_t const* const stored_offsets = graph.storage().offsets;
        std::size_t tape_bytes = 0;
        for (std::size_t entry = first; entry != found; ++entry)
            tape_bytes += static_cast<std::size_t>(stored_offsets[listed[entry] + 1] - stored_offsets[listed[entry]]);
        util::u64_t* const host_offsets = carve<util::u64_t>(memory, count + 1);
        util::byte_t* const host_tape = carve<util::byte_t>(memory, tape_bytes);
        if (!host_offsets || (tape_bytes && !host_tape)) return {0, util::status_code_t::out_of_memory_k};
        host_offsets[0] = 0;
        if (util::status_t const gathered = graph.gather_tape_({listed + first, count}, host_tape,
                                                               {host_offsets, count + 1});
            failed(gathered))
            return {0, gathered};
        for (std::size_t entry = 0; entry != count; ++entry) {
            util::bytes_view_t const stored {host_tape + host_offsets[entry],
                                             static_cast<std::size_t>(host_offsets[entry + 1] - host_offsets[entry])};
            if (util::status_t const read = bodies_t::keep_stored(variant, stored, kept[first + entry]); failed(read))
                return {0, read};
        }
    }
    auto kept_at = [&](std::size_t entry) noexcept -> kept_t const& { return kept[entry]; };
    return {found, bodies_t::write_values(variant, found, kept_at, memory, outputs)};
}

/** A modality whose bodies spell no @c keep_stored cannot read its stored bytes back from an Apple
 *  GPU variant, so it answers @c unsupported_k, as on NVIDIA. */
template <typename traits_type_, typename variant_type_>
    requires is_vram_index<variant_type_> && (!a_variant_keeping_stored<traits_type_, variant_type_>)
util::expected<std::size_t> get_entries(variant_type_&, usearch_key_t, typename traits_type_::get_outputs_t,
                                        util::growing_arena<util::forkunion_allocator_t>&,
                                        util::metal_executor_t&) noexcept {
    return {0, util::status_code_t::unsupported_k};
}

#pragma endregion Apple GPU Variants
#endif

#pragma region Variant Bodies

/** What a basic reshard moves before its rounds: other sources packed into one target, or the
 *  first source, with every other gathered into it, split into several targets. */
enum class reshard_shape_t : util::u8_t { pack_k, split_k };

/** Whether a reshard plan of @p plan_type_ tells which variant of @p variant_type_ it emptied. */
template <typename plan_type_, typename variant_type_>
concept an_emptied_naming_plan = requires(plan_type_ const& plan, variant_type_ const& variant) {
    plan.moved(&variant.index);
};

/** Whether a variant of @p variant_type_ folds the counts of a source a pack emptied into it. */
template <typename variant_type_>
concept a_pack_folding_variant = requires(variant_type_& driven, variant_type_& source) {
    driven.count_packed_source(source);
};

/** Whether a variant of @p variant_type_ names the case folding of its class. */
template <typename variant_type_>
concept a_variant_folding_texts = requires { variant_type_::fold_texts; };

/** Whether a geometry of @p geometry_type_ names the metric to score by. */
template <typename geometry_type_>
concept a_geometry_naming_metric = requires(geometry_type_ const& geometry) { geometry.metric; };

/** Whether a variant of @p variant_type_ counts the part each target of its split took. */
template <typename variant_type_>
concept a_split_counting_variant = requires(variant_type_& driven, variant_type_& target) {
    driven.count_split_target(target);
};

/** Whether a variant of @p variant_type_ pins one scalar type, which its verb table reports. */
template <typename variant_type_>
concept a_dtype_pinning_variant = requires { variant_type_::dtype_k; };

/** Whether a variant of @p variant_type_ pins one metric, which its verb table reports. */
template <typename variant_type_>
concept a_metric_pinning_variant = requires { variant_type_::metric_k; };

/** A refine plan in flight: the variant whose rounds it drives, and the engine plan, which also
 *  says where the calls stand. */
template <typename variant_type_, typename plan_type_>
struct basic_refine_plan {
    variant_type_* driven = nullptr;
    plan_type_ plan;
};

/**
 *  @brief A reshard plan in flight: the variant whose rounds it drives, the variants whose entries
 *      move into it or out of it, and the engine plan, which also says where the calls stand.
 *
 *  A pack moves every other source into the driven target, and a split moves every other source
 *  into the driven one and it into every target; once the plan closes, every variant's own counts
 *  follow the entries it holds.
 */
template <typename variant_type_, typename plan_type_>
struct basic_reshard_plan {
    variant_type_* driven = nullptr;
    util::growing_buffer<variant_type_*, util::forkunion_allocator_t> moved_variants;
    reshard_shape_t shape = reshard_shape_t::pack_k;
    plan_type_ plan;
};

/**
 *  @brief One basic variant's bodies, every one a static member its verb table points at, taking
 *      the variant as @c void* and casting it back to @p variant_type_.
 *
 *  An NVMe view serves its checkpoint in place, so its mutating bodies answer @c read_only_k; a
 *  device variant answers @c unsupported_k for what only a host walk takes.
 */
template <typename traits_type_, typename variant_type_>
struct specialized_variant {
    using traits_t = traits_type_;
    using bodies_t = variant_bodies<traits_type_>;
    using variant_t = variant_type_;
    using executor_t = typename variant_type_::executor_t;
    using verbs_t = basic_verbs<traits_type_, vendored_residency_of<executor_t>>;
    using geometry_t = typename traits_type_::geometry_t;
    using values_t = typename verbs_t::values_t;
    using queries_t = typename verbs_t::queries_t;
    using within_queries_t = typename verbs_t::within_queries_t;
    using get_outputs_t = typename verbs_t::get_outputs_t;

    static constexpr bool viewed_k = is_nvme_index<variant_type_>;

    static variant_t& of(void* variant) noexcept { return *static_cast<variant_t*>(variant); }
    static variant_t const& of(void const* variant) noexcept { return *static_cast<variant_t const*>(variant); }

    /** The variant of @p geometry, which adopts @p adopted, a default connectivity taking its
     *  place; a host one keeps its tables through @p tables, and a device one sits on the device
     *  of @p residency. */
    static util::expected<void*> create(geometry_t geometry, typename traits_t::geometry_extras_t const& adopted,
                                        arena_residency_t residency, util::forkunion_allocator_t tables) noexcept {
        variant_t* created = nullptr;
        if constexpr (is_vram_index<variant_t>)
            created = new (std::nothrow) variant_t(static_cast<int>(residency.device), geometry);
        else created = new (std::nothrow) variant_t(tables);
        if (!created) return {nullptr, util::status_code_t::out_of_memory_k};
        if (util::status_t const adopting = bodies_t::adopt(*created, geometry, adopted); failed(adopting))
            return delete created, util::expected<void*> {nullptr, adopting};
        return {created, util::status_code_t::success_k};
    }
    static void destroy(void* variant) noexcept { delete static_cast<variant_t*>(variant); }
    static void bound_shards(void* variant, std::size_t nodes) noexcept {
        if constexpr (requires(variant_t& typed) { typed.index.bound_shards(nodes); })
            of(variant).index.bound_shards(nodes);
    }
    static util::status_t emplace(void* variant, arena_residency_t residency,
                                  fu_memory_domain_id_t memory_domain) noexcept {
        if constexpr (viewed_k) return util::status_code_t::read_only_k;
        else if constexpr (is_vram_index<variant_t>)
            return of(variant).index.emplace(static_cast<int>(residency.device));
        else return of(variant).index.emplace(util::forkunion_allocator_t {memory_domain});
    }
    static util::status_t view(void* variant, std::string_view path, std::span<std::string_view const> paths,
                               executor_t* executor) noexcept {
        if constexpr (viewed_k) return of(variant).view(path, paths, *executor);
        else if constexpr (requires(variant_t& typed) { typed.view(path); }) return of(variant).view(path);
        else return util::status_code_t::unsupported_k;
    }
    static util::status_t load(void* variant, std::string_view path, executor_t& executor,
                               progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return util::status_code_t::read_only_k;
        else return of(variant).load(path, executor, ticks);
    }

    /** Reads a checkpoint from host memory into a RAM or GPU variant. */
    static util::status_t load_buffer(void* variant, util::bytes_view_t buffer, executor_t& executor,
                                      progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return util::status_code_t::read_only_k;
        else return of(variant).load(buffer, executor, ticks);
    }
    static util::status_t save(void* variant, std::string_view path, executor_t& executor,
                               progress_ticks_t& ticks) noexcept {
        return of(variant).save(path, executor, ticks);
    }

    /** Writes a RAM variant's checkpoint into memory; a view already is its checkpoint, and a GPU
     *  variant writes none to memory yet. */
    static util::status_t save_buffer(void* variant, util::bytes_span_t buffer, executor_t& executor,
                                      progress_ticks_t& ticks) noexcept {
        if constexpr (is_ram_index<variant_t>) return of(variant).save(buffer, executor, ticks);
        else return util::status_code_t::unsupported_k;
    }
    static util::expected<std::size_t> storage_requirements(void const* variant) noexcept {
        if constexpr (is_ram_index<variant_t>) return of(variant).storage_requirements();
        else return {0, util::status_code_t::unsupported_k};
    }
    static arena_residency_t residency(void const* variant) noexcept { return arena_residency_of(of(variant)); }
    static std::size_t devices(void const* variant, std::span<arena_residency_t> listed) noexcept {
        std::size_t count = 0;
        if constexpr (is_vram_index<variant_t>)
            of(variant).index.for_each_device([&](int device) noexcept {
                arena_residency_t const placed = vram_residency(static_cast<std::size_t>(device));
                for (std::size_t earlier = 0; earlier != count; ++earlier)
                    if (listed[earlier] == placed) return;
                if (count != listed.size()) listed[count++] = placed;
            });
        return count;
    }
    static util::status_t serves_writes(void const*) noexcept {
        return viewed_k ? util::status_code_t::read_only_k : util::status_code_t::success_k;
    }
    static util::status_t reserve(void* variant, reserve_capacity_t capacity) noexcept {
        if constexpr (viewed_k) return util::status_code_t::read_only_k;
        else return of(variant).reserve(capacity);
    }
    static util::status_t clear(void* variant) noexcept {
        if constexpr (viewed_k) return util::status_code_t::read_only_k;
        else return of(variant).clear();
    }
    static void describe(void const* variant, typename traits_t::description_t& description) noexcept {
        variant_t const& typed = of(variant);
        description.size = typed.index.size();
        description.capacity = typed.index.capacity();
        description.memory_usage = typed.index.memory_usage();
        description.cpu_class = usearch_cpu_class_unknown_k;
        description.gpu_class = usearch_gpu_class_unknown_k;
        if constexpr (requires { variant_t::cpu_class_k; })
            description.cpu_class = static_cast<usearch_cpu_class_t>(variant_t::cpu_class_k);
        if constexpr (requires { variant_t::gpu_class_k; })
            description.gpu_class = static_cast<usearch_gpu_class_t>(variant_t::gpu_class_k);
        if constexpr (requires { typed.index.live_shards_count(); })
            description.shards_count = typed.index.live_shards_count();
        else description.shards_count = typed.index.shards_count();
        description.load_path = usearch_load_path_none_k;
        if constexpr (requires { typed.index.loaded_through(); })
            description.load_path = usearch_load_path_of(typed.index.loaded_through());
        bodies_t::describe(typed, description);
    }
    static util::status_t count(void* variant, usearch_key_t key, typename traits_t::key_stats_t& stats,
                                executor_t& executor) noexcept {
        return count_entries<traits_t>(of(variant), key, stats, executor);
    }
    static util::expected<std::size_t> get(void* variant, usearch_key_t key, get_outputs_t outputs,
                                           util::growing_arena<util::forkunion_allocator_t>& memory,
                                           executor_t& executor) noexcept {
        return get_entries<traits_t>(of(variant), key, outputs, memory, executor);
    }
    static util::expected<traversal_stats_t> insert(void* variant, usearch_keys_view_t keys, values_t values,
                                                    insert_config_t config, executor_t& executor,
                                                    progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, util::status_code_t::read_only_k};
        else return of(variant).insert(keys, values, config, executor, ticks);
    }
    static util::status_t serves_search(void const* variant, queries_t const& queries) noexcept {
        if constexpr (requires { bodies_t::serves_search(of(variant), queries); })
            return bodies_t::serves_search(of(variant), queries);
        else return util::status_code_t::success_k;
    }
    static util::expected<traversal_stats_t> search(void* variant, queries_t queries,
                                                    usearch_matches_view_t const* hints, matches_outputs_t& outputs,
                                                    search_config_t config, predicate_t predicate,
                                                    util::growing_arena<util::forkunion_allocator_t>& pages,
                                                    executor_t& executor, progress_ticks_t& ticks) noexcept {
        variant_t& typed = of(variant);
        if constexpr (requires { typed.search(queries, hints, outputs, config, predicate, executor, ticks); })
            return typed.search(queries, hints, outputs, config, predicate, executor, ticks);
        else if (hints) return {{}, util::status_code_t::unsupported_k};
        else if constexpr (requires { typed.search(queries, outputs, config, pages, executor, ticks); })
            return typed.search(queries, outputs, config, pages, executor, ticks);
        else if constexpr (is_vram_index<variant_t>) return typed.search(queries, outputs, config, executor, ticks);
        else return typed.search(queries, outputs, config, predicate, executor, ticks);
    }
    static util::status_t serves_within(void const* variant, within_queries_t const& queries,
                                        predicate_t const& predicate) noexcept {
        if constexpr (requires { bodies_t::serves_within(of(variant), queries, predicate); })
            return bodies_t::serves_within(of(variant), queries, predicate);
        else return util::status_code_t::unsupported_k;
    }
    static util::expected<traversal_stats_t> within(void* variant, within_queries_t queries, within_outputs_t& outputs,
                                                    search_config_t config, predicate_t predicate,
                                                    util::growing_arena<util::forkunion_allocator_t>& memory,
                                                    executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (requires { of(variant).within(queries, outputs, config, predicate, memory, executor, ticks); })
            return of(variant).within(queries, outputs, config, predicate, memory, executor, ticks);
        else return {{}, util::status_code_t::unsupported_k};
    }
    static util::expected<util::sample_result_t> sample(void* variant, basic_sample_inputs_t inputs,
                                                        basic_sample_outputs_t outputs, util::sample_config_t config,
                                                        executor_t& executor) noexcept {
        return of(variant).sample(inputs, outputs, config, executor);
    }

    /** Whether the variant at @p source shares this one's geometry, so its nodes land here. */
    static bool matches(void const* variant, void const* source) noexcept {
        return bodies_t::matches(of(variant), of(source));
    }

    /** Opens one round over every shard of @p driven: a RAM variant's own plan, or a GPU one's
     *  sharded index's on the streams of @p executor. */
    static util::expected<void*> open_refine(variant_t& driven, refine_config_t config, executor_t& executor) noexcept {
        using index_t = typename variant_t::index_t;
        using held_t = basic_refine_plan<variant_t, typename index_t::refine_plan_t>;
        auto [plan, status] = [&]() noexcept -> util::expected<typename index_t::refine_plan_t> {
            if constexpr (is_vram_index<variant_t>) {
                typename index_t::refine_config_t const refine_config {config.cache_bytes, config.expansion_factor};
                return driven.index.plan_refine(refine_config, executor);
            }
            else {
                // A round's tiles follow the cores it runs on unless the caller overrides them.
                if (!config.cache_bytes) config.cache_bytes = executor.cache_bytes_per_core();
                return driven.plan_refine(config);
            }
        }();
        if (failed(status)) return {nullptr, status};
        held_t* const opened = new (std::nothrow) held_t {&driven, std::move(plan)};
        if (!opened) return {nullptr, util::status_code_t::out_of_memory_k};
        return {opened, util::status_code_t::success_k};
    }

    /** Opens the engine reshard of @p shape over @p driven, moving the variants of @p moved in or
     *  out of it: a RAM variant's own plan over the whole lists @p sources and @p targets, or a GPU
     *  one's sharded index's on the streams of @p executor. */
    static util::expected<void*> open_reshard(variant_t& driven, std::span<variant_t* const> moved,
                                              std::span<variant_t* const> sources, std::span<variant_t* const> targets,
                                              reshard_shape_t shape, reshard_options_t options,
                                              executor_t& executor) noexcept {
        using index_t = typename variant_t::index_t;
        using held_t = basic_reshard_plan<variant_t, typename index_t::reshard_plan_t>;
        util::forkunion_allocator_t const allocator {util::no_memory_domain_k};
        util::growing_buffer<variant_t*, util::forkunion_allocator_t> moved_variants {allocator};
        util::growing_buffer<index_t*, util::forkunion_allocator_t> source_indexes {allocator};
        util::growing_buffer<index_t*, util::forkunion_allocator_t> target_indexes {allocator};
        if (failed(moved_variants.reserve_exactly(moved.size())) ||
            failed(source_indexes.reserve_exactly(sources.size())) ||
            failed(target_indexes.reserve_exactly(targets.size())))
            return {nullptr, util::status_code_t::out_of_memory_k};
        for (variant_t* const variant : moved) moved_variants.push_back_reserved(variant);
        for (variant_t* const variant : sources) source_indexes.push_back_reserved(&variant->index);
        for (variant_t* const variant : targets) target_indexes.push_back_reserved(&variant->index);
        auto [plan, status] = [&]() noexcept -> util::expected<typename index_t::reshard_plan_t> {
            if constexpr (is_vram_index<variant_t>) {
                typename index_t::refine_config_t const refine_config {options.refine.cache_bytes,
                                                                       options.refine.expansion_factor};
                return driven.index.plan_reshard(to_span(source_indexes), to_span(target_indexes),
                                                 {refine_config, 0, options.seed}, executor);
            }
            else {
                // A round's tiles follow the cores it runs on unless the caller overrides them.
                refine_config_t sized = options.refine;
                if (!sized.cache_bytes) sized.cache_bytes = executor.cache_bytes_per_core();
                return driven.plan_reshard(to_span(source_indexes), to_span(target_indexes), sized, options.seed);
            }
        }();
        if (failed(status)) return {nullptr, status};
        held_t* const opened = new (std::nothrow) held_t {&driven, std::move(moved_variants), shape, std::move(plan)};
        if (!opened) return {nullptr, util::status_code_t::out_of_memory_k};
        return {opened, util::status_code_t::success_k};
    }

    /** Opens one round over every shard of the variant. */
    static util::expected<void*> refine_plan_open(void* variant, refine_config_t config,
                                                  executor_t& executor) noexcept {
        if constexpr (viewed_k) return {nullptr, util::status_code_t::read_only_k};
        else return open_refine(of(variant), config, executor);
    }

    /** Opens a pack of every other source into the one target, or a split of the first source,
     *  every other gathered into it, into every target; one shard has nothing to recut in place. */
    static util::expected<void*> reshard_plan_open(variants_view_t sources, variants_view_t targets,
                                                   reshard_options_t options, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {nullptr, util::status_code_t::read_only_k};
        else {
            util::forkunion_allocator_t const allocator {util::no_memory_domain_k};
            util::growing_buffer<variant_t*, util::forkunion_allocator_t> listed_sources {allocator};
            util::growing_buffer<variant_t*, util::forkunion_allocator_t> listed_targets {allocator};
            util::growing_buffer<variant_t*, util::forkunion_allocator_t> moved {allocator};
            if (failed(listed_sources.reserve_exactly(sources.size())) ||
                failed(listed_targets.reserve_exactly(targets.size())) ||
                failed(moved.reserve_exactly(sources.size() + targets.size())))
                return {nullptr, util::status_code_t::out_of_memory_k};
            for (std::size_t source = 0; source != sources.size(); ++source)
                listed_sources.push_back_reserved(&of(sources[source]));
            for (std::size_t target = 0; target != targets.size(); ++target)
                listed_targets.push_back_reserved(&of(targets[target]));
            if (targets.size() == 1) {
                variant_t& target = *listed_targets[0];
                for (variant_t* const source : listed_sources)
                    if (source != &target) moved.push_back_reserved(source);
                if (moved.empty()) return {nullptr, util::status_code_t::unsupported_k};
                return open_reshard(target, to_span(moved), to_span(listed_sources), to_span(listed_targets),
                                    reshard_shape_t::pack_k, options, executor);
            }
            variant_t& source = *listed_sources[0];
            // The gathered sources lead, so their counts fold into the first before the split's.
            for (variant_t* const gathered : to_span(listed_sources).subspan(1)) moved.push_back_reserved(gathered);
            if (util::lists_share(to_span(listed_sources), to_span(listed_targets)))
                return {nullptr, util::status_code_t::config_mismatch_k};
            for (variant_t* const target : listed_targets) moved.push_back_reserved(target);
            return open_reshard(source, to_span(moved), to_span(listed_sources), to_span(listed_targets),
                                reshard_shape_t::split_k, options, executor);
        }
    }

    template <typename held_type_>
    static util::status_t run_plan(void* opened, executor_t& executor, progress_ticks_t& ticks) noexcept {
        held_type_& held = *static_cast<held_type_*>(opened);
        if constexpr (is_vram_index<variant_t>) return held.plan.run(executor, ticks);
        else return held.driven->drive(held.plan, executor, ticks);
    }

    /** Hands each variant a reshard moved in or out what its entries count: the driven index folds
     *  every emptied source's, and every target of a split counts its part. */
    template <typename plan_type_>
    static void count_moved(basic_reshard_plan<variant_t, plan_type_>& held) noexcept {
        for (variant_t* const moved : held.moved_variants) {
            if (emptied_by_(held.plan, *moved)) fold_packed_(*held.driven, *moved);
            else if (held.shape == reshard_shape_t::split_k) count_split_(*held.driven, *moved);
        }
    }

    /** Whether @p plan moved @p variant 's entries out and left it empty; never, for a plan that
     *  empties none it could name. */
    template <typename plan_type_>
    static bool emptied_by_(plan_type_ const&, variant_t const&) noexcept {
        return false;
    }
    template <typename plan_type_>
        requires an_emptied_naming_plan<plan_type_, variant_t>
    static bool emptied_by_(plan_type_ const& plan, variant_t const& variant) noexcept {
        return plan.moved(&variant.index);
    }

    /** Folds the counts of @p source, which @p driven took in, into @p driven 's own. */
    static void fold_packed_(variant_t&, variant_t&) noexcept {}
    static void fold_packed_(variant_t& driven, variant_t& source) noexcept
        requires a_pack_folding_variant<variant_t>
    {
        driven.count_packed_source(source);
    }

    /** Counts the part @p target took from @p driven 's split. */
    static void count_split_(variant_t&, variant_t&) noexcept {}
    static void count_split_(variant_t& driven, variant_t& target) noexcept
        requires a_split_counting_variant<variant_t>
    {
        driven.count_split_target(target);
    }

    /** Stops the round at the barrier it stands at, runs the closing health check, or a round after
     *  moves to its end, counts what a reshard moved, and reports; a reshard whose entries have not
     *  all moved reports zero in every count. */
    template <typename held_type_>
    static util::expected<util::refine_report_t> finish_plan(void* opened, executor_t& executor) noexcept {
        held_type_& held = *static_cast<held_type_*>(opened);
        if constexpr (requires { held.plan.moving(); })
            if (held.plan.moving()) return {{}, util::status_code_t::success_k};
        if constexpr (requires { held.moved_variants; }) count_moved(held);
        if constexpr (is_vram_index<variant_t>) return held.plan.finish(executor);
        else {
            held.plan.stop();
            progress_ticks_t closing_alone {{}, 0};
            util::status_t const closed = held.driven->drive(held.plan, executor, closing_alone);
            return {held.plan.finish().value, closed};
        }
    }
    template <typename held_type_>
    static void next_round(void* opened) noexcept {
        static_cast<held_type_*>(opened)->plan.next_round();
    }
    template <typename held_type_>
    static usearch_phase_progress_t plan_phase(void const* opened) noexcept {
        return phase_of(static_cast<held_type_ const*>(opened)->plan);
    }
    template <typename held_type_>
    static util::refine_report_t plan_running(void const* opened) noexcept {
        held_type_ const& held = *static_cast<held_type_ const*>(opened);
        if constexpr (is_vram_index<variant_t>) return held.plan.running();
        else return held.plan.finish().value;
    }
    template <typename held_type_>
    static void free_plan(void* opened) noexcept {
        delete static_cast<held_type_*>(opened);
    }

    /** The entries every plan of @p operation_ runs through; none on a view, whose openings refuse
     *  first. */
    template <plan_operation_t operation_>
    static constexpr plan_verbs<executor_t> plan_verbs_for() noexcept {
        if constexpr (viewed_k) return {};
        else {
            using index_t = typename variant_t::index_t;
            using held_t = std::conditional_t<operation_ == plan_operation_t::refine_k,
                                              basic_refine_plan<variant_t, typename index_t::refine_plan_t>,
                                              basic_reshard_plan<variant_t, typename index_t::reshard_plan_t>>;
            return {&run_plan<held_t>,     &next_round<held_t>,  &plan_phase<held_t>,
                    &plan_running<held_t>, &finish_plan<held_t>, &free_plan<held_t>};
        }
    }

    /** Audits every shard of the variant: a RAM one on the calling thread, a GPU one on its
     *  shards' streams, an NVMe view through its engines, sweeping forward alone. */
    static util::expected<util::reachability_t> health(void* variant, executor_t& executor) noexcept {
        if constexpr (is_ram_index<variant_t>) {
            typename variant_t::index_t& index = of(variant).index;
            if (!index.shards_count()) return {{}, util::status_code_t::uninitialized_k};
            if (util::status_t const rewound = executor.reserve(0); failed(rewound)) return {{}, rewound};
            typename variant_t::index_t::template traversal_context_over<thread_allocator_t> context {
                0, thread_allocator_t {executor.arena(0)}};
            return index.health(context);
        }
        else return of(variant).index.health(executor);
    }

    /** Opens a walk over the variant's shards, breadth-first from @p starts when it names any, its
     *  state placed through @p allocator; an NVMe view resolves them on its first pull. */
    static util::expected<void*> cursor_init(void* variant, std::span<usearch_key_t const> starts,
                                             util::forkunion_allocator_t allocator) noexcept {
        using cursor_t = typename variant_t::index_t::cursor_t;
        using key_t = typename variant_t::index_t::key_t;
        cursor_t* const opened = new (std::nothrow) cursor_t(allocator);
        if (!opened) return {nullptr, util::status_code_t::out_of_memory_k};
        std::span<key_t const> const keys {reinterpret_cast<key_t const*>(starts.data()), starts.size()};
        if (util::status_t const walking = opened->open(of(variant).index, keys); failed(walking))
            return delete opened, util::expected<void*> {nullptr, walking};
        return {opened, util::status_code_t::success_k};
    }
    static util::expected<std::span<usearch_key_t const>> cursor_next(
        void* variant, void* cursor, links_outputs_t* links, std::size_t wanted,
        util::growing_arena<util::forkunion_allocator_t>& memory, executor_t& executor) noexcept {
        using cursor_t = typename variant_t::index_t::cursor_t;
        if constexpr (viewed_k)
            return pull_entries(of(variant), *static_cast<cursor_t*>(cursor), links, wanted, memory, executor);
        else return pull_entries(of(variant), *static_cast<cursor_t*>(cursor), links, wanted, memory);
    }
    static void cursor_free(void* cursor) noexcept {
        delete static_cast<typename variant_t::index_t::cursor_t*>(cursor);
    }

    /** The verb table of the variant, every entry one of the bodies above. */
    static constexpr verbs_t verbs() noexcept {
        nk_dtype_t dtype = nk_dtype_unknown_k;
        util::u32_t metric = 0;
        if constexpr (a_dtype_pinning_variant<variant_t>) dtype = variant_t::dtype_k;
        if constexpr (a_metric_pinning_variant<variant_t>) metric = static_cast<util::u32_t>(variant_t::metric_k);
        fold_texts_t fold_texts = nullptr;
        if constexpr (a_variant_folding_texts<variant_t>) fold_texts = variant_t::fold_texts;
        return {
            .dtype = dtype,
            .metric = metric,
            // Lifetime
            .create = &create,
            .destroy = &destroy,
            .bound_shards = &bound_shards,
            .emplace = &emplace,
            .view = &view,
            // Shape
            .residency = &residency,
            .devices = &devices,
            .describe = &describe,
            .matches = &matches,
            // Writes
            .fold_texts = fold_texts,
            .serves_writes = &serves_writes,
            .reserve = &reserve,
            .clear = &clear,
            .insert = &insert,
            // Refining
            .health = &health,
            // Plans
            .refine_plan_open = &refine_plan_open,
            .reshard_plan_open = &reshard_plan_open,
            .refine_plan = plan_verbs_for<plan_operation_t::refine_k>(),
            .reshard_plan = plan_verbs_for<plan_operation_t::reshard_k>(),
            // Lookups
            .count = &count,
            .get = &get,
            // Walks
            .serves_search = &serves_search,
            .search = &search,
            .serves_within = &serves_within,
            .within = &within,
            // Exploration
            .cursor_init = &cursor_init,
            .cursor_next = &cursor_next,
            .cursor_free = &cursor_free,
            .sample = &sample,
            // Persistence
            .load = &load,
            .load_buffer = &load_buffer,
            .save = &save,
            .save_buffer = &save_buffer,
            .storage_requirements = &storage_requirements,
        };
    }
};

/** The verb tables of the variants in the list, in its order. */
template <typename traits_type_, vendored_residency_t residency_, typename... variant_types_>
static consteval std::array<basic_verbs<traits_type_, residency_>, sizeof...(variant_types_)> basic_tables_for(
    type_list<variant_types_...>) noexcept {
    return {specialized_variant<traits_type_, variant_types_>::verbs()...};
}

/** Whether the variant @p verbs serves @p geometry: it stores values in the scalar the geometry
 *  names, if it fixes one, and scores by the metric the geometry names, if it fixes one. */
template <typename traits_type_, vendored_residency_t residency_>
bool serves_geometry(basic_verbs<traits_type_, residency_> const& verbs,
                     typename traits_type_::geometry_t const& geometry) noexcept {
    if constexpr (requires { geometry.quantization; })
        if (verbs.dtype != nk_dtype_unknown_k && verbs.dtype != nk_dtype_of(geometry.quantization)) return false;
    if constexpr (a_geometry_naming_metric<typename traits_type_::geometry_t>)
        if (verbs.metric && verbs.metric != static_cast<util::u32_t>(geometry.metric)) return false;
    return true;
}

template <typename traits_type_, vendored_residency_t residency_, auto class_>
basic_verbs<traits_type_, residency_> const* basic_class_tables<traits_type_, residency_, class_>::find(
    typename traits_type_::geometry_t const& geometry) noexcept {
    using variants_t = typename traits_type_::template variants<residency_, class_>;
    static constexpr std::array<basic_verbs<traits_type_, residency_>, variants_t::size_k> tables_k =
        basic_tables_for<traits_type_, residency_>(variants_t {});
    for (basic_verbs<traits_type_, residency_> const& verbs : tables_k)
        if (serves_geometry(verbs, geometry)) return &verbs;
    return nullptr;
}

#pragma endregion Variant Bodies
