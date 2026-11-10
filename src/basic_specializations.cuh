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
#include <cstring> // `std::memcpy` - an entry an NVMe read borrowed, kept past it

#include <array>       // `std::array` - the one key a view scans for
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid128_t` share sixteen bytes
#include <limits>      // `std::numeric_limits` - a GPU index's sentinel node ID
#include <memory>      // `std::make_obj_using_allocator` - values built from the index's allocator
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`, `std::invoke_result_t` - the predicate a thread asks

#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a key's entries in insertion order
#include "usearch/utilities/safegraphs_format.hpp"
#include "usearch/basic/ram_indexes.hpp"  // `basic::ram_indexes` - every RAM variant's shards
#include "usearch/basic/nvme_indexes.hpp" // `basic::nvme_indexes` and SmashTable's row kits, through its directories
#if USEARCH_ARCH_CUDA_
#include "usearch/basic/vram_nvidia_indexes.hpp" // `basic::vram_nvidia_indexes` - every GPU variant's shards
#endif

#include "basic_punned.cuh"

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
status_t drain_in_order(forkunion_executor_t& executor, std::size_t tasks, per_task_type_&& per_task) noexcept {
    if (!tasks) return status_t::success_k;
    // Every task bumps it, so it takes its own line, away from the locals every thread reads.
    alignas(default_alignment_k) std::size_t cursor = 0;
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

    growing_buffer<traversal_context_t, thread_allocator_t> traversals;
    growing_buffer<metric_type_, thread_allocator_t> metrics;
    std::size_t round_capacity = 0;

    /** Places the per-thread objects in the first arena of @p executor, which @ref prepare
     *  rewinds before it builds them. */
    explicit thread_scratch(forkunion_executor_t& executor) noexcept
        : traversals(thread_allocator_t {executor.arena(0)}), metrics(thread_allocator_t {executor.arena(0)}) {}

    /** Rewinds every arena, keeping the high-water mark of earlier calls, and builds one context
     *  sized for @p connectivity and @p expansion plus one metric from @p make_metric for each
     *  of the first @p threads threads. */
    template <typename make_metric_type_>
    status_t prepare(forkunion_executor_t& executor, std::size_t threads, std::size_t connectivity,
                     std::size_t expansion, u64_t seed, make_metric_type_&& make_metric) noexcept {
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        if (failed(traversals.reserve_exactly(threads)) || failed(metrics.reserve_exactly(threads)))
            return status_t::bad_alloc_k;
        typename index_type_::insert_config_t const walk {expansion};
        round_capacity = index_type_::round_capacity(connectivity, walk);
        for (std::size_t thread = 0; thread != threads; ++thread) {
            thread_allocator_t const allocator {executor.arena(thread)};
            traversals.push_back_reserved(traversal_context_t {seed + thread, allocator});
            if (status_t const sized = index_type_::reserve_traversal(traversals[thread], connectivity, walk);
                failed(sized))
                return sized;
            auto [metric, make_status] = make_metric(allocator, round_capacity);
            if (failed(make_status)) return make_status;
            metrics.push_back_reserved(std::move(metric));
        }
        return status_t::success_k;
    }

    /** The counters every thread's walks accumulated. */
    traversal_stats_t telemetry() const noexcept {
        traversal_stats_t total;
        for (traversal_context_t const& context : traversals) total += context.telemetry;
        return total;
    }
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
expected<traversal_stats_t> insert_entries(indexes_type_& index, usearch_keys_view_t keys, caller_tape_t tape,
                                           std::size_t connectivity, std::size_t expansion,
                                           make_metric_type_&& make_metric, build_type_&& build,
                                           progress_ticks_t& ticks, forkunion_executor_t& executor) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using value_t = typename indexes_type_::value_t;
    using key_t = typename indexes_type_::key_t;
    using metric_t = made_metric<make_metric_type_>;
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, executor.size(), connectivity, expansion, index.size(),
                                                  make_metric);
        failed(prepared))
        return {{}, prepared};

    auto const [entries, lower_status] = counted_tape_of(tape.tape, tape.count, executor.arena(0));
    if (failed(lower_status)) return {{}, lower_status};
    std::size_t const count = tape.count;
    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<key_t, thread_allocator_t> identities {caller};
    growing_buffer<value_t, thread_allocator_t> values {caller};
    if (failed(identities.reserve_exactly(count)) || failed(values.reserve_exactly(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t offset = 0; offset != count; ++offset) {
        identities.push_back_reserved(uuid_of(key_at(keys, offset)));
        values.push_back_reserved(std::make_obj_using_allocator<value_t>(index.get_allocator()));
    }
    status_t built = status_t::success_k;
    status_t const forked = drain_in_order<indexes_type_::template atomic_ref>(
        executor, count, [&](std::size_t offset, std::size_t thread) noexcept {
            if (succeeded(typename indexes_type_::template atomic_ref<status_t>(built).load(std::memory_order_relaxed)))
                keep_first_failure<indexes_type_::template atomic_ref>(built,
                                                                       build(entries, offset, values[offset], thread));
        });
    if (failed(forked)) return {{}, forked};
    if (failed(built)) return {{}, built};

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    basic::keep_diverse_t const selector {connectivity};
    typename indexes_type_::insert_config_t const config {expansion};
    if (status_t const inserted = drain_in_order<indexes_type_::template atomic_ref>(
            executor, count,
            [&](std::size_t offset, std::size_t thread) noexcept {
                plan.insert(offset, to_writable_span(values).subspan(offset, 1), to_span(identities).subspan(offset, 1),
                            scratch.metrics[thread], scratch.traversals[thread], selector, config);
                [[maybe_unused]] continuation_t const heard = ticks(offset);
            });
        failed(inserted))
        return {{}, inserted};
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    return {scratch.telemetry(), status_t::success_k};
}

/** The scratch of one call driving a sharded index's @p plan over shards of @p shard_type_, a
 *  metric from
 *  @p make_metric per thread, sized for @p widest_list, the widest neighbor list a walk copies, or
 *  the plan's connectivity when that is wider. */
template <typename shard_type_, typename plan_type_, typename make_metric_type_>
expected<thread_scratch<shard_type_, made_metric<make_metric_type_>>> prepare_refinement(
    plan_type_ const& plan, std::size_t widest_list, make_metric_type_&& make_metric,
    forkunion_executor_t& executor) noexcept {
    thread_scratch<shard_type_, made_metric<make_metric_type_>> scratch {executor};
    status_t const prepared = scratch.prepare(executor, executor.size(), max_of(widest_list, plan.connectivity()),
                                              plan.walk().expansion_factor, plan.audit().visible + plan.ticked,
                                              make_metric);
    return {std::move(scratch), prepared};
}

/**
 *  @brief Runs @p plan 's pending phases on the executor's threads, each with its metric and walk
 *      context from @p scratch, until no phase follows or @p ticks cancels; @c cancelled_k means a
 *      later call has more to do.
 *
 *  Every step of a phase runs before the barrier, and the ticks move once per re-selected tile
 *  after it, so a cancel leaves the plan at the next phase, where the next call starts. The scratch
 *  is this call's, from @ref prepare_refinement, and the plan holds what later calls need.
 */
template <typename plan_type_, typename shard_type_, typename metric_type_, typename selector_type_>
status_t drive_refinement(plan_type_& plan, thread_scratch<shard_type_, metric_type_>& scratch,
                          selector_type_ const& selector, progress_ticks_t& ticks,
                          forkunion_executor_t& executor) noexcept {
    using count_ref_t = typename shard_type_::template atomic_ref<std::size_t>;
    std::size_t const population = plan.planned_nodes();
    // Every step bumps it, so it takes its own line, away from the locals every thread reads.
    alignas(default_alignment_k) std::size_t taken = plan.taken;
    bool cancelling = false;
    while (true) {
        std::size_t const walked = plan.tile_nodes();
        auto const steps = plan.steps();
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            count_ref_t const next {taken};
            for (std::size_t position = next.fetch_add(1, std::memory_order_relaxed); position < steps.size();
                 position = next.fetch_add(1, std::memory_order_relaxed)) [[maybe_unused]]
                status_t const ran = plan.run(steps[position], scratch.metrics[thread], scratch.traversals[thread],
                                              selector);
        };
        if (status_t const forked = executor.for_threads(drain); failed(forked)) return forked;
        plan.taken = taken = 0;
        bool const following = plan.advance();
        // Progress moves once per tile, past the barrier, so a cancel resumes at the tile's merge;
        // a copy no stop skips runs to its end first.
        if (walked) {
            plan.ticked += walked;
            cancelling = cancelling || ticks(plan.ticked, population) == continuation_t::cancel_k;
        }
        if (cancelling && !plan.moving()) return status_t::cancelled_k;
        if (!following) return status_t::success_k;
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
status_t with_key_predicates(predicate_t const& predicate, std::size_t round_capacity, forkunion_executor_t& executor,
                             body_type_&& body) noexcept {
    if (std::holds_alternative<no_predicate_t>(predicate)) {
        body([](std::size_t) noexcept { return dummy_predicate_t {}; });
        return status_t::success_k;
    }
    if (cheap_predicate_t const* const cheap = std::get_if<cheap_predicate_t>(&predicate)) {
        body([&](std::size_t) noexcept { return *cheap; });
        return status_t::success_k;
    }
    expensive_predicates<payload_type_> expensive {std::get<expensive_predicate_t>(predicate).predicate};
    if (status_t const opened = open_predicate(expensive, executor); failed(opened)) return opened;
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        if (status_t const reserved = reserve_predicate(expensive.of_thread(thread), round_capacity); failed(reserved))
            return reserved;
    body([&](std::size_t thread) noexcept -> expensive_predicate<payload_type_>& {
        return expensive.of_thread(thread);
    });
    return status_t::success_k;
}

/** One snapshot of every shard's watermark of @p indexes in @p watermarks, sized again for a shard
 *  generation opened meanwhile, so every walk of a call sees each batch on all shards or none. */
template <typename indexes_type_, typename watermarks_type_>
status_t snapshot_shards(indexes_type_ const& indexes, watermarks_type_& watermarks) noexcept {
    while (true) {
        if (failed(watermarks.resize_uninitialized(indexes.shards_count()))) return status_t::bad_alloc_k;
        status_t const taken = indexes.snapshot(to_writable_span(watermarks)).status;
        if (taken != status_t::shape_mismatch_k) return taken;
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
 *  entries under its row of the optional @p hints, which a scan ignores.
 */
template <typename indexes_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_stats_t> search_entries(indexes_type_& indexes, std::size_t count, std::size_t connectivity,
                                           make_metric_type_&& make_metric, query_of_type_&& query_of,
                                           matches_outputs_t& outputs, search_config_t search,
                                           predicate_t const& predicate, forkunion_executor_t& executor,
                                           progress_ticks_t& ticks,
                                           usearch_matches_view_t const* hints = nullptr) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using distance_t = typename indexes_type_::distance_t;
    using metric_t = made_metric<make_metric_type_>;
    using traversal_context_t = typename thread_scratch<shard_t, metric_t>::traversal_context_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the merges resolve them");
    std::size_t const wanted = outputs.wanted, expansion = search.expansion_factor;
    // Pools narrower than the ranking they fill would short every query.
    std::size_t const breadth = max_of(expansion ? expansion : std::size_t(connectivity_32_k) * 2, wanted);
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, executor.size(), connectivity, breadth, indexes.size(),
                                                  make_metric);
        failed(prepared))
        return {{}, prepared};
    typename indexes_type_::search_config_t const config {breadth};
    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<node_id_t, thread_allocator_t> watermarks {caller};
    growing_buffer<distance_t, thread_allocator_t> distances {caller};
    if (failed(distances.resize_uninitialized(executor.size() * wanted))) return {{}, status_t::bad_alloc_k};
    if (status_t const taken = snapshot_shards(indexes, watermarks); failed(taken)) return {{}, taken};
    bool const scanning = search.traversal == traversal_t::scan_k;
    status_t searched = status_t::success_k;
    status_t const asked = with_key_predicates<typename shard_t::scored_node_t>(
        predicate, scratch.round_capacity, executor,
        [&]<typename admits_of_type_>(admits_of_type_&& admits_of) noexcept {
            using admits_t = std::invoke_result_t<admits_of_type_&, std::size_t>;
            status_t const forked = drain_in_order<indexes_type_::template atomic_ref>(
                executor, count, [&](std::size_t query_index, std::size_t thread) noexcept {
                    if (failed(typename indexes_type_::template atomic_ref<status_t>(searched).load(
                            std::memory_order_relaxed)))
                        return;
                    traversal_context_t& context = scratch.traversals[thread];
                    metric_t& metric = scratch.metrics[thread];
                    admits_t admits = admits_of(thread);
                    std::span<key_t> const keys {reinterpret_cast<key_t*>(outputs.keys) + query_index * wanted, wanted};
                    std::span<distance_t> const scores {distances.data() + thread * wanted, wanted};
                    std::span<key_t const> const query_hints =
                        hints ? std::span<key_t const> {reinterpret_cast<key_t const*>(hints->keys) +
                                                            query_index * hints->columns,
                                                        hints->counts[query_index]}
                              : std::span<key_t const> {};
                    status_t const answered = query_of(query_index, thread, [&](auto const& query) noexcept {
                        auto const [found, status] =
                            scanning ? indexes.scan(query, to_span(watermarks), keys, scores, admits, metric, context)
                                     : indexes.search_if(query, to_span(watermarks), keys, scores, config, admits,
                                                         metric, context, query_hints);
                        if (failed(status)) return status;
                        for (std::size_t match = 0; match != found; ++match)
                            outputs.scores[query_index * wanted + match] =
                                static_cast<usearch_distance_t>(scores[match]) * distance_unit<metric_t>;
                        outputs.counts[query_index] = found;
                        return status;
                    });
                    keep_first_failure<indexes_type_::template atomic_ref>(searched, answered);
                    if (ticks(query_index) == continuation_t::cancel_k)
                        keep_first_failure<indexes_type_::template atomic_ref>(searched, status_t::cancelled_k);
                });
            keep_first_failure<indexes_type_::template atomic_ref>(searched, forked);
        });
    if (failed(asked)) return {{}, asked};
    if (failed(searched)) return {{}, searched};
    return {scratch.telemetry(), status_t::success_k};
}

/** The entries @p indexes stores under @p key into @p holders and @p ids, parallel: shard by shard
 *  in placement order, and ascending within each shard, which is insertion order there. */
template <typename indexes_type_>
void entries_under(indexes_type_ const& indexes, usearch_key_t key,
                   growing_buffer<typename indexes_type_::index_t const*, thread_allocator_t>& holders,
                   growing_buffer<typename indexes_type_::node_id_t, thread_allocator_t>& ids) noexcept {
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
        partial_sort(to_writable_span(ids).subspan(first, last - first), last - first);
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
 *  executor thread's walks, or once per staged round of each thread's scan.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_, typename relation_type_>
expected<traversal_stats_t> rank_nvme_queries(index_type_& index, std::size_t count, make_metric_type_& make_metric,
                                              query_of_type_& query_of, relation_type_& relation,
                                              matches_outputs_t& outputs, search_config_t search,
                                              predicate_t const& predicate, nvme_executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
    using key_t = typename index_type_::key_t;
    using search_result_t = typename index_type_::search_result_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walks resolve them");
    std::size_t const wanted = outputs.wanted;
    // Pools narrower than the ranking they fill would short every query.
    typename index_type_::search_config_t const config {
        max_of(index.non_zero_expansion_factor({search.expansion_factor}), wanted)};
    bool const scanning = search.traversal == traversal_t::scan_k;
    dummy_predicate_t no_predicate;
    traversal_stats_t telemetry;
    std::size_t const step = ticks.slice();
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = min_of(step, count - first);
        std::span<key_t> const keys {reinterpret_cast<key_t*>(outputs.keys) + first * wanted, taking * wanted};
        std::span<usearch_distance_t> const scores {outputs.scores + first * wanted, taking * wanted};
        std::span<std::size_t> const counts {outputs.counts + first, taking};
        auto step_query_of = [&](std::size_t query_index, thread_allocator_t const& allocator) noexcept {
            return query_of(first + query_index, allocator);
        };
        if constexpr (requires { relation.prepare(executor); })
            if (status_t const prepared = relation.prepare(executor); failed(prepared)) return {telemetry, prepared};
        auto rank = [&]<typename predicate_type_>(predicate_type_& admits) noexcept {
            if (scanning)
                return index.scan(taking, make_metric, step_query_of, keys, scores, counts, admits, relation, executor);
            return index.search(taking, make_metric, step_query_of, keys, scores, counts, config, admits, relation,
                                executor);
        };
        expected<search_result_t> ranked {};
        cheap_predicate_t cheap {};
        if (std::holds_alternative<no_predicate_t>(predicate)) ranked = rank(no_predicate);
        else if (std::holds_alternative<cheap_predicate_t>(predicate))
            ranked = rank(cheap = std::get<cheap_predicate_t>(predicate));
        else if (scanning) {
            // The scan opens the family after its rewind and reserves it for the widest batch it
            // settles.
            expensive_predicates<typename index_type_::scan_payload_t> expensive {
                std::get<expensive_predicate_t>(predicate).predicate};
            ranked = index.scan(taking, make_metric, step_query_of, keys, scores, counts, expensive, relation,
                                executor);
        }
        else {
            expensive_predicates<typename index_type_::walk_payload_t> expensive {
                std::get<expensive_predicate_t>(predicate).predicate};
            ranked = index.search(taking, make_metric, step_query_of, keys, scores, counts, config, expensive, relation,
                                  executor);
        }
        if (failed(ranked.status)) return {telemetry, ranked.status};
        telemetry.computed_distances += ranked.value.computed_distances;
        telemetry.visited_members += ranked.value.visited_members;
        telemetry.restarts += ranked.value.restarts;
        telemetry.saturated_walks += ranked.value.saturated_walks;
        first += taking;
        if (ticks(first, count) == continuation_t::cancel_k) return {telemetry, status_t::cancelled_k};
    }
    return {telemetry, status_t::success_k};
}

/**
 *  @brief Searches @p count queries on an NVMe view, the walks of many in flight on every thread of
 *      @p executor, each query built by @p query_of as the view takes it, and writes every query's
 *      matches in @p outputs; progress reports a step at a time, as @ref rank_nvme_queries ranks
 *      them.
 *
 *  A predicate narrows each query to the entries it admits, while every entry still routes the
 *  walks.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_stats_t> search_entries(index_type_& index, std::size_t count, make_metric_type_&& make_metric,
                                           query_of_type_&& query_of, matches_outputs_t& outputs,
                                           search_config_t search, predicate_t const& predicate,
                                           nvme_executor_t& executor, progress_ticks_t& ticks) noexcept {
    using metric_t = made_metric<make_metric_type_>;
    std::size_t const wanted = outputs.wanted;
    dummy_predicate_t relation;
    expected<traversal_stats_t> const ranked = rank_nvme_queries(index, count, make_metric, query_of, relation, outputs,
                                                                 search, predicate, executor, ticks);
    if (failed(ranked.status)) return ranked;
    for (std::size_t query = 0; query != count; ++query)
        for (std::size_t match = 0; match != outputs.counts[query]; ++match)
            outputs.scores[query * wanted + match] *= distance_unit<metric_t>;
    return ranked;
}

/** Hands every value an NVMe view stores under @p key to @p on_value, as <tt>on_value(position,
 *  rank, thread, view)</tt> with @c rank in insertion order: one scan of the key columns counts
 *  them, handing the count to @p on_count, as <tt>on_count(entries) → status</tt>, before one more
 *  reads them. */
template <typename index_type_, typename on_count_type_, typename on_value_type_>
status_t values_under(index_type_ const& index, usearch_key_t key, nvme_executor_t& executor, on_count_type_&& on_count,
                      on_value_type_&& on_value) noexcept {
    std::array<us::utilities::uuid128_t, 1> const identities {uuid_of(key)};
    std::array<std::size_t, 1> entries {};
    if (status_t const counted = index.count(identities, entries, executor); failed(counted)) return counted;
    std::size_t const held = entries[0];
    if (status_t const prepared = on_count(held); failed(prepared) || !held) return prepared;
    return index.get(identities, entries, held, executor, on_value);
}

/** Copies @p entry, borrowed for one read, into thread @p thread 's arena as @p kept, which
 *  outlives the read until the executor's next reservation. */
inline status_t keep_entry(nvme_executor_t& executor, std::size_t thread, bytes_view_t entry,
                           bytes_view_t& kept) noexcept {
    kept = {};
    if (entry.empty()) return status_t::success_k;
    byte_t* const copy = reinterpret_cast<byte_t*>(executor.arena(thread).allocate(entry.size(), default_alignment_k));
    if (!copy) return status_t::bad_alloc_k;
    std::memcpy(copy, entry.data(), entry.size());
    kept = {copy, entry.size()};
    return status_t::success_k;
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
 *  The walks are as wide as @p connectivity asks; the plan itself never scans.
 */
template <typename indexes_type_, typename make_metric_type_, typename coordinates_type_ = no_fixed_coordinates_t>
expected<sample_result_t> sample_entries(indexes_type_& indexes, std::size_t connectivity,
                                         make_metric_type_&& make_metric, basic_sample_inputs_t inputs,
                                         basic_sample_outputs_t outputs, sample_config_t config,
                                         forkunion_executor_t& executor,
                                         coordinates_type_ const& coordinates_of = {}) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using metric_t = made_metric<make_metric_type_>;
    thread_scratch<shard_t, metric_t> scratch {executor};
    // The draw runs on the calling thread alone, so only its context and metric are built.
    if (status_t const prepared = scratch.prepare(executor, 1, connectivity, connectivity * 2, config.seed,
                                                  make_metric);
        failed(prepared))
        return {{}, prepared};
    growing_buffer<node_id_t, thread_allocator_t> watermarks {thread_allocator_t {executor.arena(0)}};
    if (status_t const taken = snapshot_shards(indexes, watermarks); failed(taken)) return {{}, taken};
    expected<sample_result_t> const sampled = indexes.sample(inputs, to_span(watermarks), outputs, config,
                                                             scratch.metrics[0], scratch.traversals[0], coordinates_of);
    if (failed(sampled.status)) return sampled;
    for (f32_t& distance : outputs.parent_distances.first(sampled.value.written))
        distance = static_cast<f32_t>(distance * distance_unit<metric_t>);
    return sampled;
}

/** Samples an NVMe view through its engines, scored by the metric @p make_metric builds over a
 *  thread's arena against the stored values @p query_of turns into queries, placed by
 *  @p coordinates_of, and writes each parent's distance in the units search reports. */
template <typename variant_type_, typename make_metric_type_, typename query_of_type_,
          typename coordinates_type_ = no_fixed_coordinates_t>
    requires is_nvme_index<variant_type_>
expected<sample_result_t> sample_entries(variant_type_& variant, make_metric_type_&& make_metric,
                                         query_of_type_&& query_of, basic_sample_inputs_t inputs,
                                         basic_sample_outputs_t outputs, sample_config_t config,
                                         nvme_executor_t& executor,
                                         coordinates_type_ const& coordinates_of = {}) noexcept {
    using metric_t = made_metric<make_metric_type_>;
    expected<sample_result_t> const sampled = variant.index.sample(inputs, outputs, config, make_metric, query_of,
                                                                   executor, coordinates_of);
    if (failed(sampled.status)) return sampled;
    for (f32_t& distance : outputs.parent_distances.first(sampled.value.written))
        distance = static_cast<f32_t>(distance * distance_unit<metric_t>);
    return sampled;
}

#pragma endregion Fan Outs

#pragma region Lookups and Cursors

/** Whether @p traits_type_ measures a key's entries from their values of @p value_type_, which a
 *  count then reads. */
template <typename traits_type_, typename stats_type_, typename value_type_>
concept tallies = requires(stats_type_& stats, value_type_ const& value) {
    variant_bodies<traits_type_>::tally(stats, value);
};

/** Adds @p value to @p stats, from any thread, where @p traits_type_ measures such values. */
template <typename traits_type_, typename stats_type_, typename value_type_>
void tally_entry(stats_type_& stats, value_type_ const& value) noexcept {
    if constexpr (tallies<traits_type_, stats_type_, value_type_>) variant_bodies<traits_type_>::tally(stats, value);
}

/** Counts the entries a RAM variant stores under @p key into @p stats, tallying their sizes where
 *  @p traits_type_ measures any. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_ram_index<variant_type_>
status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats, forkunion_executor_t&) noexcept {
    using shard_t = typename variant_type_::shard_t;
    variant.index.find_each(uuid_of(key), [&](shard_t const& shard, typename shard_t::node_id_t node) noexcept {
        ++stats.entries;
        tally_entry<traits_type_>(stats, shard.published_value_of(node));
    });
    return status_t::success_k;
}

/** Counts the entries an NVMe view stores under @p key into @p stats, reading them only where
 *  @p traits_type_ measures their sizes. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_nvme_index<variant_type_>
status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats,
                       nvme_executor_t& executor) noexcept {
    using view_t = typename variant_type_::index_t::codec_t::view_t;
    if constexpr (tallies<traits_type_, stats_type_, view_t>) {
        auto take_count = [&](std::size_t entries) noexcept { return stats.entries = entries, status_t::success_k; };
        auto measure = [&](std::size_t, std::size_t, std::size_t, view_t const& value) noexcept {
            tally_entry<traits_type_>(stats, value);
        };
        return values_under(variant.index, key, executor, take_count, measure);
    }
    else {
        std::array<us::utilities::uuid128_t, 1> const identities {uuid_of(key)};
        std::array<std::size_t, 1> entries {};
        status_t const counted = variant.index.count(identities, entries, executor);
        stats.entries = entries[0];
        return counted;
    }
}

/** Writes the entries a RAM variant stores under @p key to @p outputs, carved from @p memory, in
 *  insertion order, as @p traits_type_ writes values; answers how many. */
template <typename traits_type_, typename variant_type_>
    requires is_ram_index<variant_type_>
expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                  typename traits_type_::get_outputs_t outputs,
                                  growing_arena<forkunion_allocator_t>& memory,
                                  forkunion_executor_t& executor) noexcept {
    if (status_t const rewound = executor.reserve(0); failed(rewound)) return {0, rewound};
    using shard_t = typename variant_type_::shard_t;
    thread_allocator_t const allocator {executor.arena(0)};
    growing_buffer<shard_t const*, thread_allocator_t> holders {allocator};
    growing_buffer<typename shard_t::node_id_t, thread_allocator_t> ids {allocator};
    entries_under(variant.index, key, holders, ids);
    if (ids.empty()) return {0, status_t::success_k};
    auto value_at = [&](std::size_t entry) noexcept ->
        typename shard_t::value_t const& { return holders[entry]->published_value_of(ids[entry]); };
    return {ids.size(), variant_bodies<traits_type_>::write_values(variant, ids.size(), value_at, memory, outputs)};
}

/** Writes the entries an NVMe view stores under @p key to @p outputs, carved from @p memory, in
 *  insertion order; the values land in its threads' arenas first, as their sizes are unknown until
 *  read. */
template <typename traits_type_, typename variant_type_>
    requires is_nvme_index<variant_type_>
expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                  typename traits_type_::get_outputs_t outputs,
                                  growing_arena<forkunion_allocator_t>& memory, nvme_executor_t& executor) noexcept {
    using kept_t = typename variant_bodies<traits_type_>::kept_t;
    std::size_t held = 0;
    kept_t* kept = nullptr;
    auto carve_entries = [&](std::size_t entries) noexcept {
        held = entries;
        if (!held) return status_t::success_k;
        kept = carve<kept_t>(memory, held);
        return kept ? status_t::success_k : status_t::bad_alloc_k;
    };
    status_t keeping = status_t::success_k;
    auto keep = [&](std::size_t, std::size_t rank, std::size_t thread, auto const& value) noexcept {
        keep_first_failure<variant_type_::index_t::template atomic_ref>(
            keeping, variant_bodies<traits_type_>::keep(executor, thread, value, kept[rank]));
    };
    if (status_t const read = values_under(variant.index, key, executor, carve_entries, keep); failed(read))
        return {0, read};
    if (failed(keeping)) return {0, keeping};
    if (!held) return {0, status_t::success_k};
    auto kept_at = [&](std::size_t entry) noexcept -> kept_t const& { return kept[entry]; };
    return {held, variant_bodies<traits_type_>::write_values(variant, held, kept_at, memory, outputs)};
}

/** Pulls the next keys of a variant's walk into @p memory, gathering their links into @p links
 *  where named; an NVMe walk reads through the one @p executor, the others take none. */
template <typename variant_type_, typename... executor_types_>
expected<std::span<usearch_key_t const>> pull_entries(variant_type_& variant,
                                                      typename variant_type_::index_t::cursor_t& walk,
                                                      links_outputs_t* links, std::size_t wanted,
                                                      growing_arena<forkunion_allocator_t>& memory,
                                                      executor_types_&... executor) noexcept {
    using key_t = typename variant_type_::index_t::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk reads them");
    // No more entries remain than every shard publishes now, so the carve never outgrows the read.
    std::size_t const taking = min_of(variant.index.size(), wanted);
    usearch_key_t* const keys = carve<usearch_key_t>(memory, taking);
    if (taking && !keys) return {{}, status_t::bad_alloc_k};
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
status_t vram_nvidia_reserve(indexes_type_& index, std::size_t entries, std::size_t connectivity,
                             std::size_t bytes) noexcept {
    typename indexes_type_::reserve_capacity_t config {};
    config.capacity = entries;
    config.connectivity = static_cast<connectivity_t>(connectivity);
    config.bytes = bytes;
    return index.reserve(config).status;
}

/** Writes the live shards of @p index under @p path with @p config 's provenance, through @p codec,
 *  the wire format the RAM variants share: a lone live shard's monolith, or one per live shard
 *  beside the manifest @p path names. */
template <typename indexes_type_, typename codec_type_>
status_t vram_nvidia_save(indexes_type_& index, std::string_view path, codec_type_ const& codec,
                          save_config_t const& config, std::string_view metric, cuda_executor_t& executor,
                          progress_ticks_t& ticks) noexcept {
    return index.save(path, codec, executor, config, metric, ticks);
}

/** Inserts the entries of @p tape under the keys of @p keys_view as one batch over the shards of
 *  @p index, freezing @p connectivity first unless a reservation did. */
template <typename indexes_type_>
expected<traversal_stats_t> vram_nvidia_insert(indexes_type_& index, usearch_keys_view_t keys_view,
                                               tape_view<byte_t const> tape, std::size_t connectivity,
                                               std::size_t expansion, progress_ticks_t& ticks,
                                               cuda_executor_t& executor) noexcept {
    std::size_t const count = tape.size();
    if (!index.capacity())
        if (status_t const reserved = vram_nvidia_reserve(index, count, connectivity,
                                                          tape.offsets[count] - tape.offsets[0]);
            failed(reserved))
            return {{}, reserved};
    growing_buffer<us::utilities::uuid128_t, forkunion_allocator_t> keys {forkunion_allocator_t {no_memory_domain_k}};
    if (failed(keys.reserve_exactly(count))) return {{}, status_t::bad_alloc_k};
    for (std::size_t offset = 0; offset != count; ++offset) keys.push_back_reserved(uuid_of(key_at(keys_view, offset)));
    typename indexes_type_::insert_config_t const config {static_cast<u32_t>(expansion)};
    auto const [inserted, status] = index.insert(to_span(keys), tape, executor, config);
    if (failed(status)) return {{}, status};
    [[maybe_unused]] continuation_t const heard = ticks(count - 1);
    return {{.computed_distances = inserted.computed_distances,
             .visited_members = inserted.visited_members,
             .restarts = inserted.restarts},
            status_t::success_k};
}

/** Searches the queries of @p queries a progress step at a time, each step one merged search over
 *  every shard of @p index, outputs each query's matches in @p outputs on host pages; a handler
 *  answering cancel stops the steps not begun. A search wanting no matches answers before reading
 *  a query. */
template <typename indexes_type_>
expected<traversal_stats_t> vram_nvidia_search(indexes_type_& index, tape_view<byte_t const> queries,
                                               std::size_t wanted, std::size_t expansion, matches_outputs_t& outputs,
                                               progress_ticks_t& ticks, cuda_executor_t& executor) noexcept {
    using face_t = typename indexes_type_::index_t;
    std::size_t const count = queries.size();
    if (!wanted) {
        for (std::size_t query = 0; query != count; ++query) outputs.counts[query] = 0;
        return {{}, status_t::success_k};
    }
    using key_t = us::utilities::uuid128_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land whole in the caller's outputs");
    typename face_t::search_config_t const config {static_cast<u32_t>(expansion), static_cast<u32_t>(wanted)};
    std::size_t const step = ticks.slice();
    traversal_stats_t total;
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = min_of(step, count - first);
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
        total.restarts += found.restarts;
        for (std::size_t match = 0; match != matches; ++match) scores[match] *= distance_unit<face_t>;
        first += taking;
        if (ticks(first, count) == continuation_t::cancel_k) return {total, status_t::cancelled_k};
    }
    return {total, status_t::success_k};
}

/** Counts the entries a GPU variant stores under @p key on every shard's host key map; their
 *  values stay on the devices, so the extents stay zero. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_vram_index<variant_type_>
status_t count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats, cuda_executor_t&) noexcept {
    stats.entries = variant.index.count(uuid_of(key));
    return status_t::success_k;
}

/**
 *  @brief Writes the entries a GPU variant stores under @p key to @p outputs, carved from
 *      @p memory: each shard's node list crosses down at its node width, @c gather_tape_k packs
 *      its values back to back in the shard's output staging, and one copy brings the tape back,
 *      where the modality reads each stored value back as it writes it.
 *
 *  A modality whose bodies spell no @c keep_stored cannot read its stored bytes back, and a face
 *  that keeps its gather to its sharded index lends none, so both answer @c unsupported_k; shards
 *  on a device @p executor drives no stream on refuse with @c config_mismatch_k.
 */
template <typename traits_type_, typename variant_type_>
    requires is_vram_index<variant_type_>
expected<std::size_t> get_entries(variant_type_& variant, usearch_key_t key,
                                  typename traits_type_::get_outputs_t outputs,
                                  growing_arena<forkunion_allocator_t>& memory, cuda_executor_t& executor) noexcept {
    using bodies_t = variant_bodies<traits_type_>;
    using kept_t = typename bodies_t::kept_t;
    using graph_t = typename variant_type_::index_t::index_t;
    using node_id_t = typename graph_t::node_id_t;
    if constexpr (!requires(kept_t& kept, graph_t& graph, cuda_executor_t::traversal_context_t const& context) {
                      bodies_t::keep_stored(bytes_view_t {}, kept);
                      graph.launch_gather_tape_(std::span<node_id_t const> {}, nullptr, std::span<u64_t> {}, context);
                  })
        return {0, status_t::unsupported_k};
    else {
        us::utilities::uuid128_t const identity = uuid_of(key);
        std::size_t const held = variant.index.count(identity);
        if (!held) return {0, status_t::success_k};
        node_id_t* const listed = carve<node_id_t>(memory, held);
        kept_t* const kept = carve<kept_t>(memory, held);
        if (!listed || !kept) return {0, status_t::bad_alloc_k};
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
            std::size_t stream = 0;
            while (stream != executor.size() && executor.device(stream) != graph.device_specs().ordinal) ++stream;
            if (stream == executor.size()) return {0, status_t::config_mismatch_k};
            cuda_api_t const& api = graph.api();
            // Each node's extent comes off the device's offsets column, so the tape is sized first.
            u64_t const* const stored_offsets = graph.storage().offsets;
            std::size_t tape_bytes = 0;
            for (std::size_t entry = first; entry != found; ++entry) {
                std::array<u64_t, 2> extent {};
                if (cuda_copy(api, extent.data(), stored_offsets + listed[entry], sizeof(extent)) != cuda_success_k)
                    return {0, status_t::device_error_k};
                tape_bytes += static_cast<std::size_t>(extent[1] - extent[0]);
            }
            std::size_t const listed_bytes = round_up_to_multiple(count * sizeof(node_id_t), default_alignment_k);
            std::size_t const offsets_bytes = round_up_to_multiple((count + 1) * sizeof(u64_t), default_alignment_k);
            // The last gather was awaited, so its run rewinds for this carve, which grows it at
            // most once per call; a stream's first context is born on its device.
            std::size_t const staging_bytes = listed_bytes + offsets_bytes + tape_bytes;
            if (status_t const reserved = executor.reserve(staging_bytes, 1); failed(reserved)) return {0, reserved};
            if (cuda_set_device(api, graph.device_specs().ordinal) != cuda_success_k)
                return {0, status_t::device_error_k};
            std::span<cuda_executor_t::traversal_context_t> const stream_traversals = executor.traversals(stream);
            cuda_executor_t::traversal_context_t& context = stream_traversals[0];
            byte_t* const staging = context.allocator().allocate(staging_bytes);
            u64_t* const host_offsets = carve<u64_t>(memory, count + 1);
            byte_t* const host_tape = carve<byte_t>(memory, tape_bytes);
            if (!staging || !host_offsets || (tape_bytes && !host_tape)) return {0, status_t::bad_alloc_k};
            node_id_t* const device_listed = reinterpret_cast<node_id_t*>(staging);
            u64_t* const device_offsets = reinterpret_cast<u64_t*>(staging + listed_bytes);
            byte_t* const device_tape = staging + listed_bytes + offsets_bytes;
            host_offsets[0] = 0;
            if (cuda_copy(api, device_listed, listed + first, count * sizeof(node_id_t)) != cuda_success_k ||
                cuda_copy(api, device_offsets, host_offsets, sizeof(u64_t)) != cuda_success_k)
                return {0, status_t::device_error_k};
            if (status_t const gathered = graph.launch_gather_tape_({device_listed, count}, device_tape,
                                                                    {device_offsets, count + 1}, context);
                failed(gathered))
                return {0, gathered};
            if (status_t const waited = context.wait(); failed(waited)) return {0, waited};
            if (cuda_copy(api, host_offsets, device_offsets, (count + 1) * sizeof(u64_t)) != cuda_success_k ||
                (tape_bytes && cuda_copy(api, host_tape, device_tape, tape_bytes) != cuda_success_k))
                return {0, status_t::device_error_k};
            for (std::size_t entry = 0; entry != count; ++entry) {
                bytes_view_t const stored {host_tape + host_offsets[entry],
                                           static_cast<std::size_t>(host_offsets[entry + 1] - host_offsets[entry])};
                if (status_t const read = bodies_t::keep_stored(stored, kept[first + entry]); failed(read))
                    return {0, read};
            }
        }
        auto kept_at = [&](std::size_t entry) noexcept -> kept_t const& { return kept[entry]; };
        return {found, bodies_t::write_values(variant, found, kept_at, memory, outputs)};
    }
}

#pragma endregion GPU Variants
#endif

#pragma region Variant Bodies

/** A refinement in flight: each donor's or receiver's index, which the plan nulls as it consumes
 *  it whole, and the plan itself, which also says where the calls stand. */
template <typename variant_type_>
struct basic_refinement_state {
    growing_buffer<typename variant_type_::index_t*, forkunion_allocator_t> others;
    typename variant_type_::refine_plan_t plan;
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
    static expected<void*> create(geometry_t geometry, typename traits_t::geometry_extras_t const& adopted,
                                  arena_residency_t residency, forkunion_allocator_t tables) noexcept {
        variant_t* created = nullptr;
        if constexpr (is_vram_index<variant_t>)
            created = new (std::nothrow) variant_t(static_cast<int>(residency.device), geometry);
        else created = new (std::nothrow) variant_t(tables);
        if (!created) return {nullptr, status_t::bad_alloc_k};
        if (status_t const adopting = bodies_t::adopt(*created, geometry, adopted); failed(adopting))
            return delete created, expected<void*> {nullptr, adopting};
        return {created, status_t::success_k};
    }
    static void destroy(void* variant) noexcept { delete static_cast<variant_t*>(variant); }
    static void bound_shards(void* variant, std::size_t nodes) noexcept {
        if constexpr (requires(variant_t& typed) { typed.index.bound_shards(nodes); })
            of(variant).index.bound_shards(nodes);
    }
    static status_t emplace(void* variant, arena_residency_t residency, fu_memory_domain_id_t memory_domain) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else if constexpr (is_vram_index<variant_t>)
            return of(variant).index.emplace(static_cast<int>(residency.device));
        else return of(variant).index.emplace(memory_domain);
    }
    static status_t view(void* variant, std::string_view path, std::span<std::string_view const> paths,
                         executor_t* executor) noexcept {
        if constexpr (viewed_k) return of(variant).view(path, paths, *executor);
        else if constexpr (requires(variant_t& typed) { typed.view(path); }) return of(variant).view(path);
        else return status_t::unsupported_k;
    }
    static status_t load(void* variant, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(variant).load(path, executor, ticks);
    }

    /** Reads a checkpoint from host memory into a RAM or GPU variant. */
    static status_t load_buffer(void* variant, bytes_view_t buffer, executor_t& executor,
                                progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(variant).load(buffer, executor, ticks);
    }
    static status_t save(void* variant, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
        return of(variant).save(path, executor, ticks);
    }

    /** Writes a RAM variant's checkpoint into memory; a view already is its checkpoint, and a GPU
     *  variant writes none to memory yet. */
    static status_t save_buffer(void* variant, bytes_span_t buffer, executor_t& executor,
                                progress_ticks_t& ticks) noexcept {
        if constexpr (is_ram_index<variant_t>) return of(variant).save(buffer, executor, ticks);
        else return status_t::unsupported_k;
    }
    static expected<std::size_t> storage_requirements(void const* variant) noexcept {
        if constexpr (is_ram_index<variant_t>) return of(variant).storage_requirements();
        else return {0, status_t::unsupported_k};
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
    static status_t serves_writes(void const*) noexcept {
        return viewed_k ? status_t::read_only_k : status_t::success_k;
    }
    static status_t reserve(void* variant, reserve_capacity_t capacity) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(variant).reserve(capacity);
    }
    static status_t clear(void* variant) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
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
    static status_t count(void* variant, usearch_key_t key, typename traits_t::key_stats_t& stats,
                          executor_t& executor) noexcept {
        return count_entries<traits_t>(of(variant), key, stats, executor);
    }
    static expected<std::size_t> get(void* variant, usearch_key_t key, get_outputs_t outputs,
                                     growing_arena<forkunion_allocator_t>& memory, executor_t& executor) noexcept {
        return get_entries<traits_t>(of(variant), key, outputs, memory, executor);
    }
    static expected<traversal_stats_t> insert(void* variant, usearch_keys_view_t keys, values_t values,
                                              insert_config_t config, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else return of(variant).insert(keys, values, config, executor, ticks);
    }
    static status_t serves_search(void const* variant, queries_t const& queries) noexcept {
        if constexpr (requires { bodies_t::serves_search(of(variant), queries); })
            return bodies_t::serves_search(of(variant), queries);
        else return status_t::success_k;
    }
    static expected<traversal_stats_t> search(void* variant, queries_t queries, usearch_matches_view_t const* hints,
                                              matches_outputs_t& outputs, search_config_t config, predicate_t predicate,
                                              growing_arena<forkunion_allocator_t>& pages, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
        variant_t& typed = of(variant);
        if constexpr (requires { typed.search(queries, hints, outputs, config, predicate, executor, ticks); })
            return typed.search(queries, hints, outputs, config, predicate, executor, ticks);
        else if (hints) return {{}, status_t::unsupported_k};
        else if constexpr (requires { typed.search(queries, outputs, config, pages, executor, ticks); })
            return typed.search(queries, outputs, config, pages, executor, ticks);
        else if constexpr (is_vram_index<variant_t>) return typed.search(queries, outputs, config, executor, ticks);
        else return typed.search(queries, outputs, config, predicate, executor, ticks);
    }
    static status_t serves_within(void const* variant, within_queries_t const& queries,
                                  predicate_t const& predicate) noexcept {
        if constexpr (requires { bodies_t::serves_within(of(variant), queries, predicate); })
            return bodies_t::serves_within(of(variant), queries, predicate);
        else return status_t::unsupported_k;
    }
    static expected<traversal_stats_t> within(void* variant, within_queries_t queries, within_outputs_t& outputs,
                                              search_config_t config, predicate_t predicate,
                                              growing_arena<forkunion_allocator_t>& memory, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
        if constexpr (requires { of(variant).within(queries, outputs, config, predicate, memory, executor, ticks); })
            return of(variant).within(queries, outputs, config, predicate, memory, executor, ticks);
        else return {{}, status_t::unsupported_k};
    }
    static expected<sample_result_t> sample(void* variant, basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                            sample_config_t config, executor_t& executor) noexcept {
        return of(variant).sample(inputs, outputs, config, executor);
    }

    /** Whether the variant at @p donor shares this one's geometry, so its nodes land here. */
    static bool matches(void const* variant, void const* donor) noexcept {
        return bodies_t::matches(of(variant), of(donor));
    }

    /** Opens a refinement of the variant, an absorb of @p donors or a split into @p receivers: a
     *  RAM variant's own plan, or a GPU one's sharded index's on the streams of @p executor. */
    static expected<void*> refinement_init(void* variant, variants_view_t donors, variants_view_t receivers,
                                           split_config_t config, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {nullptr, status_t::read_only_k};
        else {
            using refining_t = basic_refinement_state<variant_t>;
            using index_t = typename variant_t::index_t;
            using refine_plan_t = typename variant_t::refine_plan_t;
            variants_view_t const others = receivers.empty() ? donors : receivers;
            growing_buffer<index_t*, forkunion_allocator_t> typed_indexes {forkunion_allocator_t {no_memory_domain_k}};
            if (failed(typed_indexes.reserve_exactly(others.size()))) return {nullptr, status_t::bad_alloc_k};
            for (std::size_t position = 0; position != others.size(); ++position)
                typed_indexes.push_back_reserved(&of(others[position]).index);
            std::span<index_t*> const typed = to_writable_span(typed_indexes);
            variant_t& refined = of(variant);
            // A RAM plan borrows these slots, whose storage moves along with the buffer.
            auto [plan, status] = [&]() noexcept -> expected<refine_plan_t> {
                if constexpr (is_vram_index<variant_t>) {
                    typename index_t::refine_config_t const refine_config {config.refine.expansion_factor,
                                                                           config.refine.memory_limit};
                    if (!receivers.empty())
                        return refined.index.plan_split(typed, {refine_config, config.seed}, executor);
                    if (!donors.empty()) return refined.index.plan_absorb(typed, refine_config, executor);
                    return refined.index.plan_refine(refine_config, executor);
                }
                else {
                    if (!receivers.empty()) return refined.plan_split(typed, config);
                    if (!donors.empty()) return refined.plan_absorb(typed, config.refine);
                    return refined.plan_refine(config.refine);
                }
            }();
            if (failed(status)) return {nullptr, status};
            refining_t* const opened = new (std::nothrow) refining_t {std::move(typed_indexes), std::move(plan)};
            if (!opened) return {nullptr, status_t::bad_alloc_k};
            return {opened, status_t::success_k};
        }
    }
    static std::size_t refinement_population(void const* refinement) noexcept {
        if constexpr (viewed_k) return 0;
        else if constexpr (is_vram_index<variant_t>)
            return static_cast<basic_refinement_state<variant_t> const*>(refinement)->plan.population();
        else return static_cast<basic_refinement_state<variant_t> const*>(refinement)->plan.audit().visible;
    }
    static status_t refinement_next(void* variant, void* refinement, executor_t& executor,
                                    progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else {
            typename variant_t::refine_plan_t& plan = static_cast<basic_refinement_state<variant_t>*>(refinement)->plan;
            if constexpr (is_vram_index<variant_t>) return plan.run(executor, ticks);
            else return of(variant).refine(plan, executor, ticks);
        }
    }
    static bool refinement_consumed(void const* refinement, std::size_t donor) noexcept {
        if constexpr (viewed_k) return false;
        else return static_cast<basic_refinement_state<variant_t> const*>(refinement)->plan.consumed(donor);
    }

    /** Folds what a consumed @p donor counts into the variant, as the points of the clouds it
     *  held. */
    static void count_absorbed_donor(void* variant, void const* donor) noexcept {
        if constexpr (requires(variant_t& typed) { typed.count_absorbed_donor(typed); })
            of(variant).count_absorbed_donor(of(donor));
    }
    static void count_split_receiver(void* variant, void const* refinement, void* receiver) noexcept {
        if constexpr (requires(variant_t& typed) { typed.count_split_receiver(typed); })
            if (!static_cast<basic_refinement_state<variant_t> const*>(refinement)->plan.moving())
                of(variant).count_split_receiver(of(receiver));
    }
    static refine_report_t refinement_running(void* refinement) noexcept {
        if constexpr (viewed_k) return {};
        else if constexpr (is_vram_index<variant_t>)
            return static_cast<basic_refinement_state<variant_t> const*>(refinement)->plan.running();
        else return static_cast<basic_refinement_state<variant_t>*>(refinement)->plan.finish().value;
    }

    /** Skips the re-selection the refinement left, heals the islands it leaves, and reports; one
     *  whose donors have not all moved in heals nothing and reports zeros. */
    static expected<refine_report_t> refinement_finish(void* variant, void* refinement, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            typename variant_t::refine_plan_t& plan = static_cast<basic_refinement_state<variant_t>*>(refinement)->plan;
            if (plan.moving()) return {{}, status_t::success_k};
            if constexpr (is_vram_index<variant_t>) return plan.finish(executor);
            else {
                // The re-selection a cancel left is skipped rather than finished.
                plan.stop();
                progress_ticks_t heal_alone {{}, 0};
                if (status_t const healed = of(variant).refine(plan, executor, heal_alone); failed(healed))
                    return {{}, healed};
                return plan.finish();
            }
        }
    }
    static void refinement_free(void* refinement) noexcept {
        if constexpr (!viewed_k) delete static_cast<basic_refinement_state<variant_t>*>(refinement);
    }

    /** Audits every shard of the variant: a RAM one on the calling thread, a GPU one on its
     *  shards' streams, an NVMe view through its engines, sweeping forward alone. */
    static expected<reachability_t> health(void* variant, executor_t& executor) noexcept {
        if constexpr (is_ram_index<variant_t>) {
            typename variant_t::index_t& index = of(variant).index;
            if (!index.shards_count()) return {{}, status_t::uninitialized_k};
            if (status_t const rewound = executor.reserve(0); failed(rewound)) return {{}, rewound};
            typename variant_t::index_t::template traversal_context_over<thread_allocator_t> context {
                0, thread_allocator_t {executor.arena(0)}};
            return index.health(context);
        }
        else return of(variant).index.health(executor);
    }

    /** Opens a walk over the variant's shards, breadth-first from @p starts when it names any; an
     *  NVMe view resolves them on its first pull. */
    static expected<void*> cursor_init(void* variant, std::span<usearch_key_t const> starts) noexcept {
        using cursor_t = typename variant_t::index_t::cursor_t;
        using key_t = typename variant_t::index_t::key_t;
        cursor_t* const opened = new (std::nothrow) cursor_t(forkunion_allocator_t {no_memory_domain_k});
        if (!opened) return {nullptr, status_t::bad_alloc_k};
        std::span<key_t const> const keys {reinterpret_cast<key_t const*>(starts.data()), starts.size()};
        if (status_t const walking = opened->open(of(variant).index, keys); failed(walking))
            return delete opened, expected<void*> {nullptr, walking};
        return {opened, status_t::success_k};
    }
    static expected<std::span<usearch_key_t const>> cursor_next(void* variant, void* cursor, links_outputs_t* links,
                                                                std::size_t wanted,
                                                                growing_arena<forkunion_allocator_t>& memory,
                                                                executor_t& executor) noexcept {
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
        u32_t metric = 0;
        if constexpr (requires { variant_t::dtype_k; }) dtype = variant_t::dtype_k;
        if constexpr (requires { variant_t::metric_k; }) metric = static_cast<u32_t>(variant_t::metric_k);
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
            .serves_writes = &serves_writes,
            .reserve = &reserve,
            .clear = &clear,
            .insert = &insert,
            // Refining
            .health = &health,
            // Refinements
            .refinement_init = &refinement_init,
            .refinement_population = &refinement_population,
            .refinement_next = &refinement_next,
            .refinement_consumed = &refinement_consumed,
            .count_absorbed_donor = &count_absorbed_donor,
            .count_split_receiver = &count_split_receiver,
            .refinement_running = &refinement_running,
            .refinement_finish = &refinement_finish,
            .refinement_free = &refinement_free,
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
 *  names, if it fixes one, and scores by the kind the geometry names, if its kernels fix one. */
template <typename traits_type_, vendored_residency_t residency_>
bool serves_geometry(basic_verbs<traits_type_, residency_> const& verbs,
                     typename traits_type_::geometry_t const& geometry) noexcept {
    if constexpr (requires { geometry.quantization; })
        if (verbs.dtype != nk_dtype_unknown_k && verbs.dtype != nk_dtype_of(geometry.quantization)) return false;
    if constexpr (requires { geometry.metric; })
        if (verbs.metric && verbs.metric != static_cast<u32_t>(geometry.metric)) return false;
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
