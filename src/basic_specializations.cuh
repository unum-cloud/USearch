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
#include <cstring> // `std::memcpy` - a row an NVMe read borrowed, kept past it

#include <array>       // `std::array` - the one key a view scans for
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
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

using basic::traversal_telemetry_t;

/** The SmashTable row kit one CPU class pairs for the scans its NVMe directories run: picked here
 *  per class, never by a unit's own flags, so every class unit spells one index type with one kit
 *  - the serial floor, as is any class whose @c SMASHTABLE_TARGET_<KIT> is off. */
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

/** The row kit a CPU class pairs, spelled short for the variants' explicit instantiations. */
template <cpu_class_t cpu_class_>
using row_kit_of = typename row_kit_for_cpu_class<cpu_class_>::type;

#pragma region Fan Outs

/** Runs @p per_task over @p tasks taken in ascending order by the executor's threads; a refused
 *  fan-out runs them all on the calling thread, since a dispensed row must still land. */
template <template <typename> class atomic_ref_template_, typename per_task_type_>
void drain_in_order(forkunion_executor_t& executor, std::size_t tasks, per_task_type_&& per_task) noexcept {
    if (!tasks) return;
    // Every task bumps it, so it takes its own line, away from the locals every thread reads.
    alignas(default_alignment_k) std::size_t cursor = 0;
    auto drain = [&](std::size_t thread, std::size_t) noexcept {
        atomic_ref_template_<std::size_t> const next {cursor};
        for (std::size_t task = next.fetch_add(1, std::memory_order_relaxed); task < tasks;
             task = next.fetch_add(1, std::memory_order_relaxed))
            per_task(task, thread);
    };
    if (failed(executor.for_threads(drain))) drain(0, 0);
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
     *  sized for @p connectivity and @p expansion plus one metric from @p make_metric per
     *  thread. */
    template <typename make_metric_type_>
    status_t prepare(forkunion_executor_t& executor, std::size_t connectivity, std::size_t expansion, u64_t seed,
                     make_metric_type_&& make_metric) noexcept {
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        std::size_t const threads = executor.size();
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
    traversal_telemetry_t telemetry() const noexcept {
        traversal_telemetry_t total;
        for (traversal_context_t const& context : traversals) total += context.telemetry;
        return total;
    }
};

/**
 *  @brief Inserts @p count entries under their keys as one batch: every thread builds its share of
 *      the values from the index's allocator, then one plan lands them on every thread and
 *      publishes once.
 *
 *  Everything fallible runs before the plan dispenses: the keys and values, the traversals and the
 *  metrics. A refused build frees every value built so far and leaves the index untouched. Once the
 *  plan dispensed, every position lands and the progress handler can no longer cancel. The plan
 *  routes the rows over the shards of @p indexes, and their walks run on each shard's own scratch.
 */
template <typename indexes_type_, typename make_metric_type_, typename build_type_>
expected<traversal_telemetry_t> insert_entries(indexes_type_& index, usearch_keys_view_t keys, std::size_t count,
                                               std::size_t connectivity, std::size_t expansion,
                                               make_metric_type_&& make_metric, build_type_&& build,
                                               progress_ticks_t& ticks, forkunion_executor_t& executor) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using value_t = typename indexes_type_::value_t;
    using key_t = typename indexes_type_::key_t;
    using metric_t = made_metric<make_metric_type_>;
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, connectivity, expansion, index.size(), make_metric);
        failed(prepared))
        return {{}, prepared};

    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<key_t, thread_allocator_t> identities {caller};
    growing_buffer<value_t, thread_allocator_t> values {caller};
    if (failed(identities.reserve_exactly(count)) || failed(values.reserve_exactly(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) {
        identities.push_back_reserved(uuid_of(key_at(keys, row)));
        values.push_back_reserved(std::make_obj_using_allocator<value_t>(index.get_allocator()));
    }
    status_t built = status_t::success_k;
    drain_in_order<indexes_type_::template atomic_ref>(
        executor, count, [&](std::size_t row, std::size_t thread) noexcept {
            if (succeeded(standard_atomic_ref<status_t>(built).load(std::memory_order_relaxed)))
                keep_first_failure(built, build(row, values[row], thread));
        });
    if (failed(built)) return {{}, built};

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    basic::keep_diverse_t const selector {connectivity};
    typename indexes_type_::insert_config_t const config {expansion};
    drain_in_order<indexes_type_::template atomic_ref>(
        executor, count, [&](std::size_t row, std::size_t thread) noexcept {
            plan.insert(row, to_writable_span(values).subspan(row, 1), to_span(identities).subspan(row, 1),
                        scratch.metrics[thread], scratch.traversals[thread], selector, config);
            [[maybe_unused]] continuation_t const heard = ticks(row);
        });
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    return {scratch.telemetry(), status_t::success_k};
}

/** The scratch of one call driving the plural @p plan over shards of @p shard_type_, a metric from
 *  @p make_metric per thread, sized for @p row_width, the widest row a walk copies, or the plan's
 *  connectivity when that is wider. */
template <typename shard_type_, typename plan_type_, typename make_metric_type_>
expected<thread_scratch<shard_type_, made_metric<make_metric_type_>>> prepare_refinement(
    plan_type_ const& plan, std::size_t row_width, make_metric_type_&& make_metric,
    forkunion_executor_t& executor) noexcept {
    thread_scratch<shard_type_, made_metric<make_metric_type_>> scratch {executor};
    status_t const prepared = scratch.prepare(executor, max_of(row_width, plan.connectivity()),
                                              plan.walk().expansion_factor, plan.audit().visible + plan.ticked,
                                              make_metric);
    return {std::move(scratch), prepared};
}

/**
 *  @brief Runs @p plan 's pending phases on the executor's threads, each with its metric and walk
 *      context from @p scratch, until no phase follows or @p ticks cancels; @c cancelled_k means a
 *      later call has more to do.
 *
 *  Every step a thread takes runs, and a thread takes none once a tick cancelled, so the plan's
 *  @c taken says exactly where the next call starts. The scratch is this call's, from
 *  @ref prepare_refinement; the plan keeps everything the index needs between calls.
 */
template <typename plan_type_, typename shard_type_, typename metric_type_, typename selector_type_>
status_t drive_refinement(plan_type_& plan, thread_scratch<shard_type_, metric_type_>& scratch,
                          selector_type_ const& selector, progress_ticks_t& ticks,
                          forkunion_executor_t& executor) noexcept {
    std::size_t const population = plan.audit().visible;
    // Every step bumps them, so they take their own lines, away from the locals every thread reads.
    alignas(default_alignment_k) std::size_t ticked = plan.ticked;
    alignas(default_alignment_k) std::size_t taken = plan.taken;
    alignas(default_alignment_k) bool cancelled = false;
    while (true) {
        bool const reselecting = plan.reselecting();
        auto const steps = plan.steps();
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            standard_atomic_ref<std::size_t> const next {taken};
            while (!standard_atomic_ref<bool>(cancelled).load(std::memory_order_relaxed)) {
                std::size_t const position = next.fetch_add(1, std::memory_order_relaxed);
                if (position >= steps.size()) break;
                auto const step = steps[position];
                [[maybe_unused]] status_t const ran = plan.run(step, scratch.metrics[thread],
                                                               scratch.traversals[thread], selector);
                if (!reselecting) continue;
                std::size_t const nodes = plan.nodes_in(step);
                std::size_t const done =
                    standard_atomic_ref<std::size_t>(ticked).fetch_add(nodes, std::memory_order_relaxed) + nodes;
                if (ticks(done, population) == continuation_t::cancel_k)
                    standard_atomic_ref<bool>(cancelled).store(true, std::memory_order_relaxed);
            }
        };
        if (failed(executor.for_threads(drain))) drain(0, 0);
        plan.ticked = ticked;
        plan.taken = min_of(taken, steps.size());
        if (cancelled) return status_t::cancelled_k;
        plan.taken = taken = 0;
        if (!plan.advance()) return status_t::success_k;
    }
}

/** What one unit of @p metric_type_'s distance is worth on the C scale: one, unless a modality's
 *  fixed-point metric, or a GPU index scoring with one, specializes it so that its full scale lands
 *  as one. */
template <typename metric_type_>
inline constexpr usearch_distance_t distance_unit = 1;

/** Hands @p body a callable building the key predicate an executor thread asks, which a walk asks
 *  about its nodes through their records: the dummy without a handler, @p predicate asked a quad at
 *  a time when cheap, and when expensive, over that thread's own buffers, reserved for rounds of
 *  @p round_capacity candidates gathered beside @p payload_type_. */
template <typename payload_type_, typename body_type_>
status_t with_key_predicates(usearch_predicate_t predicate, std::size_t round_capacity, forkunion_executor_t& executor,
                             body_type_&& body) noexcept {
    if (!predicate.callback) {
        body([](std::size_t) noexcept { return dummy_predicate_t {}; });
        return status_t::success_k;
    }
    if (predicate.cost != usearch_predicate_cost_expensive_k) {
        predicate_t const cheap {predicate};
        body([&](std::size_t) noexcept { return cheap; });
        return status_t::success_k;
    }
    expensive_predicates<payload_type_> expensive {predicate};
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
 *      hands the query it prepared to a continuation, and lands every row in @p landing.
 *
 *  A predicate narrows each row to the entries it admits: a walk routes past the rest until the
 *  admitted ones fill its breadth, and a scan asks only about rows that would enter the row. Every
 *  row walks each shard of @p indexes below one snapshot taken for the whole call, so a batch
 *  counts in every row on all its shards or none, and lands the merged ranking. A handler
 *  answering cancel stops the queries no thread has taken yet.
 */
template <typename indexes_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_telemetry_t> search_entries(indexes_type_& indexes, std::size_t count, std::size_t wanted,
                                               std::size_t connectivity, std::size_t expansion,
                                               usearch_traversal_t traversal, usearch_predicate_t predicate,
                                               make_metric_type_&& make_metric, query_of_type_&& query_of,
                                               matches_landing_t& landing, progress_ticks_t& ticks,
                                               forkunion_executor_t& executor) noexcept {
    using shard_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using distance_t = typename indexes_type_::distance_t;
    using metric_t = made_metric<make_metric_type_>;
    using traversal_context_t = typename thread_scratch<shard_t, metric_t>::traversal_context_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the merges resolve them");
    // Pools narrower than the row they fill would short every row.
    std::size_t const breadth = max_of(expansion ? expansion : std::size_t(connectivity_32_k) * 2, wanted);
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, connectivity, breadth, indexes.size(), make_metric);
        failed(prepared))
        return {{}, prepared};
    typename indexes_type_::search_config_t const config {breadth};
    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<node_id_t, thread_allocator_t> watermarks {caller};
    growing_buffer<distance_t, thread_allocator_t> distances {caller};
    if (failed(distances.resize_uninitialized(executor.size() * wanted))) return {{}, status_t::bad_alloc_k};
    if (status_t const taken = snapshot_shards(indexes, watermarks); failed(taken)) return {{}, taken};
    bool const scanning = traversal == usearch_traversal_scan_k;
    status_t searched = status_t::success_k;
    status_t const asked = with_key_predicates<typename shard_t::scored_node_t>(
        predicate, scratch.round_capacity, executor,
        [&]<typename admits_of_type_>(admits_of_type_&& admits_of) noexcept {
            using admits_t = std::invoke_result_t<admits_of_type_&, std::size_t>;
            drain_in_order<indexes_type_::template atomic_ref>(
                executor, count, [&](std::size_t row, std::size_t thread) noexcept {
                    if (failed(standard_atomic_ref<status_t>(searched).load(std::memory_order_relaxed))) return;
                    traversal_context_t& context = scratch.traversals[thread];
                    metric_t& metric = scratch.metrics[thread];
                    admits_t admits = admits_of(thread);
                    std::span<key_t> const keys {reinterpret_cast<key_t*>(landing.keys) + row * wanted, wanted};
                    std::span<distance_t> const scores {distances.data() + thread * wanted, wanted};
                    status_t const answered = query_of(row, thread, [&](auto const& query) noexcept {
                        auto const [found, status] = scanning
                                                         ? indexes.scan(query, metric, admits, context,
                                                                        to_span(watermarks), keys, scores)
                                                         : indexes.search_if(query, metric, admits, context, config,
                                                                             to_span(watermarks), keys, scores);
                        if (failed(status)) return status;
                        for (std::size_t match = 0; match != found; ++match)
                            landing.scores[row * wanted + match] = static_cast<usearch_distance_t>(scores[match]) *
                                                                   distance_unit<metric_t>;
                        landing.counts[row] = found;
                        return status;
                    });
                    keep_first_failure(searched, answered);
                    if (ticks(row) == continuation_t::cancel_k) keep_first_failure(searched, status_t::cancelled_k);
                });
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
 *      the view with the walks of many queries in flight on every thread of @p executor, and lands
 *      every row in @p landing; a handler answering cancel stops the steps not begun.
 *
 *  Query @c i is built by @p query_of, and @p relation tests the stored values a row would admit;
 *  a per-thread relation family is rebuilt every step, as every step rewinds the arenas. A
 *  predicate narrows each row to the entries it admits, while every row still routes the walks: a
 *  cheap one is asked a quad at a time, and an expensive one once per completion batch of each
 *  executor thread's walks, or once per staged round of each thread's scan.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_, typename relation_type_>
expected<traversal_telemetry_t> rank_nvme_queries(index_type_& index, std::size_t count, std::size_t wanted,
                                                  std::size_t expansion, usearch_traversal_t traversal,
                                                  usearch_predicate_t predicate, make_metric_type_& make_metric,
                                                  query_of_type_& query_of, relation_type_& relation,
                                                  matches_landing_t& landing, progress_ticks_t& ticks,
                                                  nvme_executor_t& executor) noexcept {
    using key_t = typename index_type_::key_t;
    using search_result_t = typename index_type_::search_result_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walks resolve them");
    // Pools narrower than the row they fill would short every row.
    typename index_type_::search_config_t const config {max_of(index.non_zero_expansion_factor({expansion}), wanted)};
    bool const scanning = traversal == usearch_traversal_scan_k;
    dummy_predicate_t unfiltered;
    predicate_t cheap {predicate};
    traversal_telemetry_t telemetry;
    std::size_t const step = ticks.slice();
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = min_of(step, count - first);
        std::span<key_t> const keys {reinterpret_cast<key_t*>(landing.keys) + first * wanted, taking * wanted};
        std::span<usearch_distance_t> const scores {landing.scores + first * wanted, taking * wanted};
        std::span<std::size_t> const counts {landing.counts + first, taking};
        auto step_query_of = [&](std::size_t row, thread_allocator_t const& allocator) noexcept {
            return query_of(first + row, allocator);
        };
        if constexpr (requires { relation.prepare(executor); })
            if (status_t const prepared = relation.prepare(executor); failed(prepared)) return {telemetry, prepared};
        auto rank = [&]<typename predicate_type_>(predicate_type_& admits) noexcept {
            if (scanning)
                return index.scan(taking, make_metric, step_query_of, executor, keys, scores, counts, admits, relation);
            return index.search(taking, make_metric, step_query_of, executor, config, keys, scores, counts, admits,
                                relation);
        };
        expected<search_result_t> ranked {};
        if (!predicate.callback) ranked = rank(unfiltered);
        else if (predicate.cost != usearch_predicate_cost_expensive_k) ranked = rank(cheap);
        else if (scanning) {
            // The scan opens the family after its rewind and reserves it for the widest batch it
            // settles.
            expensive_predicates<typename index_type_::scan_payload_t> expensive {predicate};
            ranked = index.scan(taking, make_metric, step_query_of, executor, keys, scores, counts, expensive,
                                relation);
        }
        else {
            expensive_predicates<typename index_type_::walk_payload_t> expensive {predicate};
            ranked = index.search(taking, make_metric, step_query_of, executor, config, keys, scores, counts, expensive,
                                  relation);
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
 *      @p executor, each query built by @p query_of as the view takes it, and lands every row in
 *      @p landing; progress lands a step at a time, as @ref rank_nvme_queries ranks them.
 *
 *  A predicate narrows each row to the entries it admits, while every row still routes the walks.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_telemetry_t> search_entries(index_type_& index, std::size_t count, std::size_t wanted,
                                               std::size_t expansion, usearch_traversal_t traversal,
                                               usearch_predicate_t predicate, make_metric_type_&& make_metric,
                                               query_of_type_&& query_of, matches_landing_t& landing,
                                               progress_ticks_t& ticks, nvme_executor_t& executor) noexcept {
    using metric_t = made_metric<make_metric_type_>;
    dummy_predicate_t relation;
    expected<traversal_telemetry_t> const ranked = rank_nvme_queries(index, count, wanted, expansion, traversal,
                                                                     predicate, make_metric, query_of, relation,
                                                                     landing, ticks, executor);
    if (failed(ranked.status)) return ranked;
    for (std::size_t row = 0; row != count; ++row)
        for (std::size_t match = 0; match != landing.counts[row]; ++match)
            landing.scores[row * wanted + match] *= distance_unit<metric_t>;
    return ranked;
}

/** Hands every value an NVMe view stores under @p key to @p on_value, as
 *  `on_value(position, rank, thread, view)` with @c rank in insertion order: one scan of the key
 *  columns counts them, handing the count to @p on_count, as `on_count(entries) → status`, before
 *  one more reads them. */
template <typename index_type_, typename on_count_type_, typename on_value_type_>
status_t values_under(index_type_ const& index, usearch_key_t key, nvme_executor_t& executor, on_count_type_&& on_count,
                      on_value_type_&& on_value) noexcept {
    std::array<us::utilities::uuid_t, 1> const identities {uuid_of(key)};
    std::array<std::size_t, 1> entries {};
    if (status_t const counted = index.count(identities, entries, executor); failed(counted)) return counted;
    std::size_t const held = entries[0];
    if (status_t const prepared = on_count(held); failed(prepared) || !held) return prepared;
    return index.get(identities, held, entries, executor, on_value);
}

/** Copies @p row, borrowed for one read, into thread @p thread 's arena as @p kept, which outlives
 *  the read until the executor's next reservation. */
inline status_t keep_row(nvme_executor_t& executor, std::size_t thread, bytes_view_t row, bytes_view_t& kept) noexcept {
    kept = {};
    if (row.empty()) return status_t::success_k;
    byte_t* const copy = reinterpret_cast<byte_t*>(executor.arena(thread).allocate(row.size(), default_alignment_k));
    if (!copy) return status_t::bad_alloc_k;
    std::memcpy(copy, row.data(), row.size());
    kept = {copy, row.size()};
    return status_t::success_k;
}

/** The GPU ordinal @p variant computes on, its first shard's, where a call's arena must live, or
 *  the host's. */
template <typename variant_type_>
int device_of(variant_type_ const& variant) noexcept {
    if constexpr (is_vram_index<variant_type_>) return variant.index.first_device();
    else return host_device_k;
}

#pragma endregion Fan Outs

#pragma region Lookups and Cursors

/** The cursor over every shard of the RAM variant @p variant_type_, in placement order. */
template <typename variant_type_>
using ram_cursor_of = typename variant_type_::index_t::cursor_t;

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
void count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats, usearch_executor_t,
                   usearch_outcome_t* outcome) noexcept {
    using shard_t = typename variant_type_::index_t::index_t;
    variant.index.find_each(uuid_of(key), [&](shard_t const& shard, typename shard_t::node_id_t node) noexcept {
        ++stats.entries;
        tally_entry<traits_type_>(stats, shard.published_value_of(node));
    });
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Counts the entries an NVMe view stores under @p key into @p stats, reading them only where
 *  @p traits_type_ measures their sizes. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_nvme_index<variant_type_>
void count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats, usearch_executor_t executor_handle,
                   usearch_outcome_t* outcome) noexcept {
    usearch_executor_handle_t call_executor;
    nvme_executor_t* const executor = executor_of(variant, executor_handle, call_executor, outcome);
    if (!executor) return;
    using view_t = typename variant_type_::index_t::codec_t::view_t;
    if constexpr (tallies<traits_type_, stats_type_, view_t>) {
        auto take_count = [&](std::size_t entries) noexcept { return stats.entries = entries, status_t::success_k; };
        auto measure = [&](std::size_t, std::size_t, std::size_t, view_t const& value) noexcept {
            tally_entry<traits_type_>(stats, value);
        };
        propagate_error(values_under(variant.index, key, *executor, take_count, measure), outcome);
    }
    else {
        std::array<us::utilities::uuid_t, 1> const identities {uuid_of(key)};
        std::array<std::size_t, 1> entries {};
        status_t const counted = variant.index.count(identities, entries, *executor);
        stats.entries = entries[0];
        propagate_error(counted, outcome);
    }
}

/** Lands the entries a RAM variant stores under the key @p arguments name into @p memory, in
 *  insertion order, as @p traits_type_ lands values. */
template <typename traits_type_, typename variant_type_, typename arguments_type_>
    requires is_ram_index<variant_type_>
void get_entries(variant_type_& variant, arguments_type_ const& arguments, growing_arena<forkunion_allocator_t>& memory,
                 usearch_outcome_t* outcome) noexcept {
    usearch_executor_handle_t call_executor;
    forkunion_executor_t* const executor = executor_of(variant, arguments.executor, call_executor, outcome);
    if (!executor) return;
    if (status_t const rewound = executor->reserve(0); failed(rewound))
        return propagate_error(rewound, outcome), void();
    using shard_t = typename variant_type_::index_t::index_t;
    thread_allocator_t const allocator {executor->arena(0)};
    growing_buffer<shard_t const*, thread_allocator_t> holders {allocator};
    growing_buffer<typename shard_t::node_id_t, thread_allocator_t> ids {allocator};
    entries_under(variant.index, arguments.key, holders, ids);
    if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    auto value_at = [&](std::size_t entry) noexcept ->
        typename shard_t::value_t const& { return holders[entry]->published_value_of(ids[entry]); };
    if (!variant_bodies<traits_type_>::land(variant, ids.size(), value_at, memory, arguments, outcome)) return;
    if (arguments.written) *arguments.written = ids.size();
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Lands the entries an NVMe view stores under the key @p arguments name into @p memory, in
 *  insertion order; the values land in its threads' arenas first, as their sizes are unknown until
 *  read. */
template <typename traits_type_, typename variant_type_, typename arguments_type_>
    requires is_nvme_index<variant_type_>
void get_entries(variant_type_& variant, arguments_type_ const& arguments, growing_arena<forkunion_allocator_t>& memory,
                 usearch_outcome_t* outcome) noexcept {
    using kept_t = typename variant_bodies<traits_type_>::kept_t;
    usearch_executor_handle_t call_executor;
    nvme_executor_t* const executor = executor_of(variant, arguments.executor, call_executor, outcome);
    if (!executor) return;
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
        keep_first_failure(keeping, variant_bodies<traits_type_>::keep(*executor, thread, value, kept[rank]));
    };
    if (status_t const read = values_under(variant.index, arguments.key, *executor, carve_entries, keep); failed(read))
        return propagate_error(read, outcome), void();
    if (failed(keeping)) return propagate_error(keeping, outcome), void();
    if (!held) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    auto kept_at = [&](std::size_t entry) noexcept -> decltype(auto) { return kept[entry]; };
    if (!variant_bodies<traits_type_>::land(variant, held, kept_at, memory, arguments, outcome)) return;
    if (arguments.written) *arguments.written = held;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys of a RAM variant's shards into @p memory, shard after shard; a destructive
 *  window since the cursor opened answers @c contended_k and lands none. */
template <typename traits_type_, typename variant_type_, typename arguments_type_>
    requires is_ram_index<variant_type_>
void pull_entries(variant_type_& variant, ram_cursor_of<variant_type_>& cursor, arguments_type_ const& arguments,
                  growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept {
    using key_t = typename variant_type_::index_t::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the cursor reads them");
    // No more rows remain than every shard publishes now, so the carve never outgrows the read.
    std::size_t const taking = min_of(variant.index.size(), arguments.wanted);
    usearch_key_t* const keys = carve<usearch_key_t>(memory, taking);
    if (taking && !keys)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
    auto const [written, status] = cursor.read(variant.index,
                                               std::span<key_t> {reinterpret_cast<key_t*>(keys), taking});
    if (failed(status)) return propagate_error(status, outcome), void();
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.written = written;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys of an NVMe view into @p memory through the executor. */
template <typename traits_type_, typename variant_type_, typename arguments_type_>
    requires is_nvme_index<variant_type_>
void pull_entries(variant_type_& variant, std::size_t& row, arguments_type_ const& arguments,
                  growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept {
    usearch_executor_handle_t call_executor;
    nvme_executor_t* const executor = executor_of(variant, arguments.executor, call_executor, outcome);
    if (!executor) return;
    std::size_t const taking = min_of(variant.index.size() - row, arguments.wanted);
    if (!taking) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    usearch_key_t* const keys = carve<usearch_key_t>(memory, taking);
    if (!keys) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
    // The values the read passes along stay unread, as a pull lands keys alone.
    auto skip = [](std::size_t, std::size_t, auto const&) noexcept {};
    std::span<us::utilities::uuid_t> const identities {reinterpret_cast<us::utilities::uuid_t*>(keys), taking};
    if (status_t const read = variant.index.read_rows(row, identities, *executor, skip); failed(read))
        return propagate_error(read, outcome), void();
    row += taking;
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.written = taking;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Lookups and Cursors

#if USEARCH_ARCH_CUDA_
#pragma region GPU Variants

/** The plan of a GPU variant's refinement, which no GPU variant opens yet: each one's
 *  @c plan_refine answers @c unsupported_k, so no call ever runs one. */
struct vram_nvidia_refine_plan_t {
    struct audit_t {
        std::size_t visible = 0;
    };
    audit_t audit() const noexcept { return {}; }
    bool landing() const noexcept { return false; }
    bool consumed(std::size_t) const noexcept { return false; }
    bool reselecting() const noexcept { return false; }
    void stop() noexcept {}
    std::span<std::size_t const> steps() const noexcept { return {}; }
    expected<basic::refine_result_t> finish() const noexcept { return {{}, status_t::unsupported_k}; }
};

/** Reserves @p entries rows of @p connectivity neighbors and @p bytes of values over every shard of
 *  a GPU variant's @p index, freezing the connectivity. */
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

/** Inserts the rows of @p tape under the keys of @p keys_view as one batch over the shards of
 *  @p index, freezing @p connectivity first unless a reservation did. */
template <typename indexes_type_>
expected<traversal_telemetry_t> vram_nvidia_insert(indexes_type_& index, usearch_keys_view_t keys_view,
                                                   tape_view<byte_t const> tape, std::size_t connectivity,
                                                   std::size_t expansion, progress_ticks_t& ticks,
                                                   cuda_executor_t& executor) noexcept {
    std::size_t const count = tape.size();
    if (!index.capacity())
        if (status_t const reserved = vram_nvidia_reserve(index, count, connectivity,
                                                          tape.offsets[count] - tape.offsets[0]);
            failed(reserved))
            return {{}, reserved};
    growing_buffer<us::utilities::uuid_t, forkunion_allocator_t> keys {forkunion_allocator_t {no_memory_domain_k}};
    if (failed(keys.reserve_exactly(count))) return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) keys.push_back_reserved(uuid_of(key_at(keys_view, row)));
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
 *  every shard of @p index, landing each row's matches in @p landing on host pages; a handler
 *  answering cancel stops the steps not begun. A search wanting no matches answers before reading
 *  a query. */
template <typename indexes_type_>
expected<traversal_telemetry_t> vram_nvidia_search(indexes_type_& index, tape_view<byte_t const> queries,
                                                   std::size_t wanted, std::size_t expansion,
                                                   matches_landing_t& landing, progress_ticks_t& ticks,
                                                   cuda_executor_t& executor) noexcept {
    using face_t = typename indexes_type_::index_t;
    std::size_t const count = queries.size();
    if (!wanted) {
        for (std::size_t row = 0; row != count; ++row) landing.counts[row] = 0;
        return {{}, status_t::success_k};
    }
    using key_t = us::utilities::uuid_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land whole in the caller's landing");
    typename face_t::search_config_t const config {static_cast<u32_t>(expansion), static_cast<u32_t>(wanted)};
    std::size_t const step = ticks.slice();
    traversal_telemetry_t total;
    for (std::size_t first = 0; first != count;) {
        std::size_t const taking = min_of(step, count - first);
        std::size_t const first_match = first * wanted, matches = taking * wanted;
        usearch_distance_t* const scores = landing.scores + first_match;
        auto const [found, status] = index.search(
            queries.subview(first, taking), executor, config,
            nk::matrix_span<key_t> {reinterpret_cast<key_t*>(landing.keys + first_match), {taking, wanted}},
            nk::matrix_span<usearch_distance_t> {scores, {taking, wanted}},
            std::span<std::size_t> {landing.counts + first, taking});
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

/** A GPU variant counts no key's entries yet. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_vram_index<variant_type_>
void count_entries(variant_type_&, usearch_key_t, stats_type_&, usearch_executor_t,
                   usearch_outcome_t* outcome) noexcept {
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
}

/** A GPU variant exports no key's entries yet. */
template <typename traits_type_, typename variant_type_, typename arguments_type_>
    requires is_vram_index<variant_type_>
void get_entries(variant_type_&, arguments_type_ const&, growing_arena<forkunion_allocator_t>&,
                 usearch_outcome_t* outcome) noexcept {
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
}

#pragma endregion GPU Variants
#endif

#pragma region Variant Bodies

/** Whether @p variant_type_ serves @p geometry: it stores values in the scalar the geometry names,
 *  if it stores one, and scores by the metric the geometry names, if its kernels fix one. */
template <typename variant_type_, typename geometry_type_>
bool serves_geometry(geometry_type_ const& geometry) noexcept {
    if constexpr (requires { variant_type_::dtype_k; })
        if (nk_dtype_of(geometry.quantization) != variant_type_::dtype_k) return false;
    if constexpr (requires { variant_type_::metric_k; })
        if (geometry.metric != variant_type_::metric_k) return false;
    return true;
}

/** A refinement in flight: each donor's index, which the plan nulls as it lands it whole, and the
 *  plan itself, which also says where the calls stand. */
template <typename variant_type_>
struct basic_refinement_state {
    growing_buffer<typename variant_type_::index_t*, forkunion_allocator_t> landing;
    typename variant_type_::refine_plan_t plan;
};

/** The C report of one walk's @p walked telemetry, or its failure; static, as each class unit
 *  compiles its own under its flags. */
static expected<usearch_traversal_stats_t> walked_stats(expected<traversal_telemetry_t> const& walked) noexcept {
    usearch_traversal_stats_t stats {};
    accumulate_stats(&stats, walked.value);
    return {stats, walked.status};
}

/** The C report of a refinement's @p refined result. */
static basic_refined_t refined_report(basic::refine_result_t const& refined) noexcept {
    basic_refined_t report;
    accumulate_stats(&report.stats, refined.telemetry);
    report.visited = refined.visited_nodes;
    report.changed = refined.changed_nodes;
    report.reachability = {refined.health.audited_nodes, refined.health.reachable_nodes, refined.health.sweeps,
                           refined.health.back_sweeps};
    return report;
}

/**
 *  @brief One basic variant's bodies, every one a static member its verb table points at, taking
 *      the variant as @c void* and casting it back to @p variant_type_.
 *
 *  An NVMe view serves its checkpoint in place, so its mutating bodies answer @c read_only_k; a
 *  device variant answers @c unsupported_k for what only a host walk takes, as cursors.
 */
template <typename traits_type_, typename variant_type_>
struct specialized_variant {
    using traits_t = traits_type_;
    using bodies_t = variant_bodies<traits_type_>;
    using variant_t = variant_type_;
    using executor_t = typename variant_type_::executor_t;
    using verbs_t = basic_verbs<traits_type_, home_of<executor_t>>;
    using geometry_t = typename traits_type_::geometry_t;
    using walked_t = typename verbs_t::walked_t;

    static constexpr bool viewed_k = is_nvme_index<variant_type_>;

    static variant_t& of(void* variant) noexcept { return *static_cast<variant_t*>(variant); }
    static variant_t const& of(void const* variant) noexcept { return *static_cast<variant_t const*>(variant); }

    static bool serves(geometry_t const& geometry) noexcept { return serves_geometry<variant_t>(geometry); }
    static void* create(int device, geometry_t const& geometry) noexcept {
        if constexpr (is_vram_index<variant_t>) return new (std::nothrow) variant_t(device, geometry);
        else return new (std::nothrow) variant_t(forkunion_allocator_t {no_memory_domain_k});
    }
    static void destroy(void* variant) noexcept { delete static_cast<variant_t*>(variant); }
    static bool freeze(void* variant, geometry_t& geometry, typename traits_t::freeze_t const& input,
                       usearch_outcome_t* outcome) noexcept {
        return bodies_t::freeze(of(variant), geometry, input, outcome);
    }
    static void bound_shards(void* variant, std::size_t rows) noexcept {
        if constexpr (requires(variant_t& typed) { typed.index.bound_shards(rows); })
            of(variant).index.bound_shards(rows);
    }
    static status_t emplace(void* variant, int placement) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else if constexpr (is_vram_index<variant_t>) return of(variant).index.emplace(placement);
        else return of(variant).index.emplace(static_cast<fu_memory_domain_id_t>(placement));
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
    static status_t save(void* variant, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
        return of(variant).save(path, executor, ticks);
    }
    static int device(void const* variant) noexcept { return device_of(of(variant)); }
    static std::size_t devices(void const* variant, std::span<int> listed) noexcept {
        std::size_t count = 0;
        if constexpr (is_vram_index<variant_t>)
            of(variant).index.for_each_device([&](int device) noexcept {
                for (std::size_t earlier = 0; earlier != count; ++earlier)
                    if (listed[earlier] == device) return;
                if (count != listed.size()) listed[count++] = device;
            });
        return count;
    }
    static status_t reserve(void* variant, typename traits_t::reserve_t const& arguments) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(variant).reserve(arguments);
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
        bodies_t::describe(typed, description);
    }
    static void count(void* variant, usearch_key_t key, typename traits_t::key_stats_t& stats,
                      usearch_executor_t executor, usearch_outcome_t* outcome) noexcept {
        count_entries<traits_t>(of(variant), key, stats, executor, outcome);
    }
    static void get(void* variant, typename traits_t::get_t const& arguments,
                    growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept {
        get_entries<traits_t>(of(variant), arguments, memory, outcome);
    }
    static walked_t insert(void* variant, typename traits_t::insert_t const& arguments, progress_ticks_t& ticks,
                           executor_t& executor) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else return walked_stats(of(variant).insert(arguments, ticks, executor));
    }
    static bool accepts_search(void const* variant, typename traits_t::search_t const& arguments,
                               usearch_outcome_t* outcome) noexcept {
        return bodies_t::accepts(of(variant), arguments, outcome);
    }

    /** A device walk stages its queries on the arena's host @p pages, where its matches land. */
    static walked_t search(void* variant, typename traits_t::search_t const& arguments,
                           growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing,
                           progress_ticks_t& ticks, executor_t& executor) noexcept {
        variant_t& typed = of(variant);
        if constexpr (requires { typed.search(arguments, pages, landing, ticks, executor); })
            return walked_stats(typed.search(arguments, pages, landing, ticks, executor));
        else return walked_stats(typed.search(arguments, landing, ticks, executor));
    }
    static bool accepts_within(void const* variant, typename traits_t::within_t const& arguments,
                               usearch_outcome_t* outcome) noexcept {
        if constexpr (requires { bodies_t::accepts_within(of(variant), arguments, outcome); })
            return bodies_t::accepts_within(of(variant), arguments, outcome);
        else return propagate_error(usearch_status_unsupported_k, outcome, "This modality walks no radius");
    }
    static walked_t within(void* variant, typename traits_t::within_t const& arguments,
                           growing_arena<forkunion_allocator_t>& memory, usearch_matches_tape_view_t& matches,
                           std::size_t& written, progress_ticks_t& ticks, executor_t& executor) noexcept {
        if constexpr (requires { bodies_t::within(of(variant), arguments, memory, matches, written, ticks, executor); })
            return walked_stats(bodies_t::within(of(variant), arguments, memory, matches, written, ticks, executor));
        else return {{}, status_t::unsupported_k};
    }
    static status_t sample(void* variant, typename traits_t::sample_t const& arguments) noexcept {
        return of(variant).sample(arguments);
    }

    /** Whether the variant at @p donor shares this one's geometry, so its rows land here. */
    static bool matches(void const* variant, void const* donor) noexcept {
        return bodies_t::matches(of(variant), of(donor));
    }
    static expected<void*> refinement_open(void* variant, std::span<void* const> donors,
                                           std::size_t expansion) noexcept {
        if constexpr (viewed_k) return {nullptr, status_t::read_only_k};
        else {
            using refining_t = basic_refinement_state<variant_t>;
            growing_buffer<typename variant_t::index_t*, forkunion_allocator_t> landing {
                forkunion_allocator_t {no_memory_domain_k}};
            if (failed(landing.reserve_exactly(donors.size()))) return {nullptr, status_t::bad_alloc_k};
            for (void* const donor : donors) landing.push_back_reserved(&of(donor).index);
            // The plan borrows the landing slots, whose storage moves along with the buffer.
            auto [plan, status] = donors.empty() ? of(variant).plan_refine(expansion)
                                                 : of(variant).plan_absorb(to_writable_span(landing), expansion);
            if (failed(status)) return {nullptr, status};
            refining_t* const opened = new (std::nothrow) refining_t {std::move(landing), std::move(plan)};
            if (!opened) return {nullptr, status_t::bad_alloc_k};
            return {opened, status_t::success_k};
        }
    }
    static std::size_t refinement_population(void const* refining) noexcept {
        if constexpr (viewed_k) return 0;
        else return static_cast<basic_refinement_state<variant_t> const*>(refining)->plan.audit().visible;
    }
    static status_t refinement_run(void* variant, void* refining, progress_ticks_t& ticks,
                                   executor_t& executor) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else
            return of(variant).refine(static_cast<basic_refinement_state<variant_t>*>(refining)->plan, ticks, executor);
    }
    static bool refinement_landed(void const* refining, std::size_t donor) noexcept {
        if constexpr (viewed_k) return false;
        else return static_cast<basic_refinement_state<variant_t> const*>(refining)->plan.consumed(donor);
    }

    /** Folds what a landed @p donor counts into the variant, as the points of the clouds it
     *  held. */
    static void absorbed(void* variant, void const* donor) noexcept {
        if constexpr (requires(variant_t& typed) { typed.absorbed(typed); }) of(variant).absorbed(of(donor));
    }
    static basic_refined_t refinement_running(void* refining) noexcept {
        if constexpr (viewed_k) return {};
        else return refined_report(static_cast<basic_refinement_state<variant_t>*>(refining)->plan.finish().value);
    }

    /** Skips the re-selection the refinement left, heals what it strands, and reports; one whose
     *  donors have not all landed heals nothing and reports zeros. */
    static expected<basic_refined_t> refinement_finish(void* variant, void* refining, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            typename variant_t::refine_plan_t& plan = static_cast<basic_refinement_state<variant_t>*>(refining)->plan;
            if (plan.landing()) return {{}, status_t::success_k};
            // The re-selection a cancel left is skipped rather than finished.
            plan.stop();
            progress_ticks_t heal_alone {{}, 0};
            if (status_t const healed = of(variant).refine(plan, heal_alone, executor); failed(healed))
                return {{}, healed};
            auto const [refined, status] = plan.finish();
            return {refined_report(refined), status};
        }
    }
    static void refinement_free(void* refining) noexcept {
        if constexpr (!viewed_k) delete static_cast<basic_refinement_state<variant_t>*>(refining);
    }

    /** Opens a cursor on a RAM variant's snapshot or an NVMe view's first row; null once @p outcome
     *  records why not. */
    static void* cursor_open(void* variant, usearch_outcome_t* outcome) noexcept {
        void* opened = nullptr;
        if constexpr (is_ram_index<variant_t>) opened = new (std::nothrow) ram_cursor_of<variant_t>(of(variant).index);
        else if constexpr (viewed_k) opened = new (std::nothrow) std::size_t {0};
        else
            return propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet"),
                   nullptr;
        if (!opened) propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor");
        return opened;
    }
    static void cursor_next(void* variant, void* cursor, typename traits_t::cursor_next_t const& arguments,
                            growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept {
        if constexpr (is_ram_index<variant_t>)
            pull_entries<traits_t>(of(variant), *static_cast<ram_cursor_of<variant_t>*>(cursor), arguments, memory,
                                   outcome);
        else if constexpr (viewed_k)
            pull_entries<traits_t>(of(variant), *static_cast<std::size_t*>(cursor), arguments, memory, outcome);
        else propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
    }
    static void cursor_free(void* cursor) noexcept {
        if constexpr (is_ram_index<variant_t>) delete static_cast<ram_cursor_of<variant_t>*>(cursor);
        else if constexpr (viewed_k) delete static_cast<std::size_t*>(cursor);
    }

    /** The verb table of the variant, every entry one of the bodies above. */
    static constexpr verbs_t verbs() noexcept {
        return {
            .serves = &serves,
            .create = &create,
            .destroy = &destroy,
            .freeze = &freeze,
            .bound_shards = &bound_shards,
            .emplace = &emplace,
            .view = &view,
            .load = &load,
            .save = &save,
            .device = &device,
            .devices = &devices,
            .reserve = &reserve,
            .clear = &clear,
            .describe = &describe,
            .count = &count,
            .get = &get,
            .insert = &insert,
            .accepts_search = &accepts_search,
            .search = &search,
            .accepts_within = &accepts_within,
            .within = &within,
            .sample = &sample,
            .matches = &matches,
            .refinement_open = &refinement_open,
            .refinement_population = &refinement_population,
            .refinement_run = &refinement_run,
            .refinement_landed = &refinement_landed,
            .absorbed = &absorbed,
            .refinement_running = &refinement_running,
            .refinement_finish = &refinement_finish,
            .refinement_free = &refinement_free,
            .cursor_open = &cursor_open,
            .cursor_next = &cursor_next,
            .cursor_free = &cursor_free,
        };
    }
};

/** The verb tables of the variants in the list, in its order. */
template <typename traits_type_, dense_home_t home_, typename... variant_types_>
static consteval std::array<basic_verbs<traits_type_, home_>, sizeof...(variant_types_)> basic_tables_for(
    type_list<variant_types_...>) noexcept {
    return {specialized_variant<traits_type_, variant_types_>::verbs()...};
}

template <typename traits_type_, dense_home_t home_, auto class_>
basic_verbs<traits_type_, home_> const* basic_class_tables<traits_type_, home_, class_>::find(
    typename traits_type_::geometry_t const& geometry) noexcept {
    using variants_t = typename traits_type_::template variants<home_, class_>;
    static constexpr std::array<basic_verbs<traits_type_, home_>, variants_t::size_k> tables_k =
        basic_tables_for<traits_type_, home_>(variants_t {});
    for (basic_verbs<traits_type_, home_> const& verbs : tables_k)
        if (verbs.serves(geometry)) return &verbs;
    return nullptr;
}

#pragma endregion Variant Bodies
