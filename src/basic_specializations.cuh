/**
 *  @file src/basic_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What every basic modality's variants share, whatever they store: the crossings of keys
 *      and tapes, the fan-outs over the executor's threads, each residency's lookups and cursors,
 *      and the base of every variant on a GPU.
 *
 *  Each modality's @c <modality>_specializations.cuh spells its variants on top of this, and its
 *  class units in @c src/<modality>/ instantiate them; @c basic_index.cuh adds the handle and the
 *  verbs every modality's dispatch unit forwards its exported C entry points to. The basic index
 *  is the same type in every class, so a class differs only in the kits its metrics score with,
 *  and the variants keep the index's default waiter and atomic reference.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - a row an NVMe read borrowed, kept past it

#include <array>       // `std::array` - the one key a view scans for
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <charconv>    // `std::to_chars`, `std::from_chars` - the metadata a checkpoint records as text
#include <limits>      // `std::numeric_limits` - a GPU index's sentinel node ID
#include <memory>      // `std::make_obj_using_allocator` - values built from the index's allocator
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`
#include <variant>     // `std::get_if` - the snapshot or row a cursor stands on

#include "usearch/basic/ram_index.hpp"
#include "usearch/basic/nvme_index.hpp"    // `basic::nvme_index` and SmashTable's row kits, through its directories
#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a key's entries in insertion order
#include "usearch/utilities/safegraphs_format.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/basic/vram_nvidia_index.hpp"
#endif

#include "lib.hpp"

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

#pragma region Crossings and Fan Outs

/** Key @p row of the caller's keys, through their stride; zero means packed. */
inline usearch_key_t key_at(usearch_keys_view_t keys, std::size_t row) noexcept {
    std::size_t const stride = keys.stride ? keys.stride : sizeof(usearch_key_t);
    usearch_key_t key;
    std::memcpy(&key, reinterpret_cast<byte_t const*>(keys.data) + row * stride, sizeof(key));
    return key;
}

/** Entry @p row of a byte tape, as the view every string type converts from. */
inline std::string_view chars_at(usearch_tape_view_t tape, std::size_t row) noexcept {
    return {static_cast<char const*>(tape.data) + tape.offsets[row], tape.offsets[row + 1] - tape.offsets[row]};
}

/** Whether a tape's @p count + 1 offsets never fall; they may start anywhere, as an Arrow slice's
 *  do. */
inline bool monotone_offsets(usearch_tape_view_t tape, std::size_t count) noexcept {
    if (!tape.offsets) return false;
    for (std::size_t row = 0; row != count; ++row)
        if (tape.offsets[row + 1] < tape.offsets[row]) return false;
    return true;
}

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

/** Keeps the first failure any thread reports into @p first. */
inline void keep_first_failure(status_t& first, status_t status) noexcept {
    if (succeeded(status)) return;
    status_t unset = status_t::success_k;
    standard_atomic_ref<status_t>(first).compare_exchange_strong(unset, status, std::memory_order_relaxed,
                                                                 std::memory_order_relaxed);
}

/** Raises @p word to @p value, if it is larger, from any thread; a relaxed peek first, so a word
 *  already that large is never written. */
template <typename word_type_>
void raise_to(word_type_& word, std::type_identity_t<word_type_> value) noexcept {
    fu::standard_atomic_ref<word_type_> const raised {word};
    if (raised.load(std::memory_order_relaxed) < value) raised.fetch_max(value, std::memory_order_relaxed);
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
        for (traversal_context_t const& context : traversals) {
            total.computed_distances += context.telemetry.computed_distances;
            total.visited_members += context.telemetry.visited_members;
            total.restarts += context.telemetry.restarts;
            total.saturated_walks += context.telemetry.saturated_walks;
            total.computed_distances_in_rankings += context.telemetry.computed_distances_in_rankings;
            total.computed_distances_in_selections += context.telemetry.computed_distances_in_selections;
        }
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
 *  plan dispensed, every position lands and the progress handler can no longer cancel.
 */
template <typename index_type_, typename make_metric_type_, typename build_type_>
expected<traversal_telemetry_t> insert_entries(index_type_& index, usearch_keys_view_t keys, std::size_t count,
                                               std::size_t connectivity, std::size_t expansion,
                                               make_metric_type_&& make_metric, build_type_&& build,
                                               progress_ticks_t& ticks, forkunion_executor_t& executor) noexcept {
    using value_t = typename index_type_::value_t;
    using key_t = typename index_type_::key_t;
    using metric_t = made_metric<make_metric_type_>;
    thread_scratch<index_type_, metric_t> scratch {executor};
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
    drain_in_order<index_type_::template atomic_ref>(
        executor, count, [&](std::size_t row, std::size_t thread) noexcept {
            if (succeeded(standard_atomic_ref<status_t>(built).load(std::memory_order_relaxed)))
                keep_first_failure(built, build(row, values[row], thread));
        });
    if (failed(built)) return {{}, built};

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    basic::keep_diverse_t const selector {connectivity};
    typename index_type_::insert_config_t const config {expansion};
    drain_in_order<index_type_::template atomic_ref>(
        executor, count, [&](std::size_t row, std::size_t thread) noexcept {
            plan.insert(row, to_writable_span(values).subspan(row, 1), to_span(identities).subspan(row, 1),
                        scratch.metrics[thread], scratch.traversals[thread], selector, config);
            [[maybe_unused]] continuation_t const heard = ticks(row);
        });
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    return {scratch.telemetry(), status_t::success_k};
}

/** The scratch of one call driving @p plan, a metric from @p make_metric per thread, sized for
 *  @p row_width, the widest row a walk copies, or the plan's connectivity when that is wider. */
template <typename index_type_, typename make_metric_type_>
expected<thread_scratch<index_type_, made_metric<make_metric_type_>>> prepare_refinement(
    typename index_type_::refine_plan_t const& plan, std::size_t row_width, make_metric_type_&& make_metric,
    forkunion_executor_t& executor) noexcept {
    thread_scratch<index_type_, made_metric<make_metric_type_>> scratch {executor};
    status_t const prepared = scratch.prepare(executor, max_of(row_width, plan.connectivity()),
                                              plan.walk().expansion_factor, plan.audit.visible + plan.ticked,
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
template <typename index_type_, typename metric_type_, typename selector_type_>
status_t drive_refinement(typename index_type_::refine_plan_t& plan, thread_scratch<index_type_, metric_type_>& scratch,
                          selector_type_ const& selector, progress_ticks_t& ticks,
                          forkunion_executor_t& executor) noexcept {
    std::size_t const population = plan.audit.visible;
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
                std::size_t const done = standard_atomic_ref<std::size_t>(ticked).fetch_add(step.nodes.size(),
                                                                                            std::memory_order_relaxed) +
                                         step.nodes.size();
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

/** Lands the nearest @p wanted of one query's @p ranking in row @p row of @p landing. */
template <typename metric_type_, typename index_type_, typename ranking_type_>
void land_matches(index_type_ const& index, ranking_type_ const& ranking, std::size_t row, std::size_t wanted,
                  matches_landing_t& landing) noexcept {
    std::size_t const landed = min_of(wanted, ranking.size());
    for (std::size_t match = 0; match != landed; ++match) {
        landing.keys[row * wanted + match] = std::bit_cast<usearch_key_t>(index.key_of(ranking[match].node_id).value);
        landing.scores[row * wanted + match] = static_cast<usearch_distance_t>(ranking[match].distance) *
                                               distance_unit<metric_type_>;
    }
    landing.counts[row] = landed;
}

/** Hands @p body a callable building the node predicate an executor thread asks: the dummy without
 *  a handler, @p predicate asked a quad at a time when cheap, and when expensive, over that
 *  thread's own buffers, reserved for rounds of @p round_capacity candidates. */
template <typename index_type_, typename body_type_>
status_t with_node_predicates(index_type_ const& index, usearch_predicate_t predicate, std::size_t round_capacity,
                              forkunion_executor_t& executor, body_type_&& body) noexcept {
    using node_id_t = typename index_type_::node_id_t;
    auto const key_of = [&index](node_id_t node_id) noexcept { return index.key_of(node_id).value; };
    if (!predicate.callback) {
        body([](std::size_t) noexcept { return dummy_predicate_t {}; });
        return status_t::success_k;
    }
    if (predicate.cost != usearch_predicate_cost_expensive_k) {
        predicate_t cheap {predicate};
        body([&](std::size_t) noexcept { return node_predicate {cheap, key_of}; });
        return status_t::success_k;
    }
    // The basic RAM verbs take one thread's predicate, so this drives the family as an engine
    // would, after the rewind the call's scratch took.
    expensive_predicates<typename index_type_::scored_node_t> expensive {predicate};
    if (status_t const opened = open_predicate(expensive, executor); failed(opened)) return opened;
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        if (status_t const reserved = reserve_predicate(expensive.of_thread(thread), round_capacity); failed(reserved))
            return reserved;
    body([&](std::size_t thread) noexcept { return node_predicate {expensive.of_thread(thread), key_of}; });
    return status_t::success_k;
}

/** Walks @p index towards @p query, keeping in @p context the entries @p predicate admits. */
template <typename index_type_, typename query_type_, typename metric_type_, typename predicate_type_,
          typename context_type_>
expected<typename index_type_::search_result_t> walk_admitted(index_type_& index, query_type_ const& query,
                                                              metric_type_& metric, predicate_type_& predicate,
                                                              context_type_& context,
                                                              typename index_type_::search_config_t config) noexcept {
    dummy_predicate_t relation;
    return index.search_if(query, metric, relation, predicate, context, config);
}

/** An unfiltered walk keeps no admitted set beside its routing candidates. */
template <typename index_type_, typename query_type_, typename metric_type_, typename context_type_>
expected<typename index_type_::search_result_t> walk_admitted(index_type_& index, query_type_ const& query,
                                                              metric_type_& metric, dummy_predicate_t&,
                                                              context_type_& context,
                                                              typename index_type_::search_config_t config) noexcept {
    return index.search(query, metric, context, config);
}

/**
 *  @brief Searches @p count queries on the executor's threads, each built by @p query_of, which
 *      hands the query it prepared to a continuation, and lands every row in @p landing.
 *
 *  A predicate narrows each row to the entries it admits: a walk routes past the rest until the
 *  admitted ones fill its breadth, and a scan asks only about rows that would enter the row. A
 *  handler answering cancel stops the queries no thread has taken yet.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_telemetry_t> search_entries(index_type_& index, std::size_t count, std::size_t wanted,
                                               std::size_t connectivity, std::size_t expansion,
                                               usearch_traversal_t traversal, usearch_predicate_t predicate,
                                               make_metric_type_&& make_metric, query_of_type_&& query_of,
                                               matches_landing_t& landing, progress_ticks_t& ticks,
                                               forkunion_executor_t& executor) noexcept {
    using metric_t = made_metric<make_metric_type_>;
    // Pools narrower than the row they fill would short every row.
    std::size_t const breadth = max_of(expansion ? expansion : std::size_t(connectivity_32_k) * 2, wanted);
    thread_scratch<index_type_, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, connectivity, breadth, index.size(), make_metric);
        failed(prepared))
        return {{}, prepared};
    typename index_type_::search_config_t const config {breadth};
    using traversal_context_t = typename thread_scratch<index_type_, metric_t>::traversal_context_t;
    bool const scanning = traversal == usearch_traversal_scan_k;
    status_t searched = status_t::success_k;
    status_t const asked = with_node_predicates(
        index, predicate, scratch.round_capacity, executor, [&](auto&& admits_of) noexcept {
            drain_in_order<index_type_::template atomic_ref>(
                executor, count, [&](std::size_t row, std::size_t thread) noexcept {
                    if (failed(standard_atomic_ref<status_t>(searched).load(std::memory_order_relaxed))) return;
                    traversal_context_t& context = scratch.traversals[thread];
                    metric_t& metric = scratch.metrics[thread];
                    auto admits = admits_of(thread);
                    status_t const answered = query_of(row, thread, [&](auto const& query) noexcept {
                        dummy_predicate_t relation;
                        auto const [found, status] = scanning
                                                         ? index.scan(query, metric, relation, admits, context, wanted)
                                                         : walk_admitted(index, query, metric, admits, context, config);
                        if (succeeded(status)) land_matches<metric_t>(index, context.results, row, wanted, landing);
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

/** The node IDs stored under @p key, ascending, which is insertion order, in a buffer on
 *  @p allocator. */
template <typename index_type_>
growing_buffer<typename index_type_::node_id_t, thread_allocator_t> ids_under(
    index_type_ const& index, usearch_key_t key, thread_allocator_t const& allocator) noexcept {
    using node_id_t = typename index_type_::node_id_t;
    growing_buffer<node_id_t, thread_allocator_t> ids {allocator};
    std::size_t const holders = index.count(uuid_of(key));
    if (!holders || failed(ids.reserve_exactly(holders))) return ids;
    index.find_each(uuid_of(key), [&](node_id_t node_id) noexcept {
        if (ids.size() != ids.capacity()) ids.push_back_reserved(node_id);
    });
    partial_sort(to_writable_span(ids), ids.size());
    return ids;
}

/** The decimal spelling of @p value in @p text, which a checkpoint's metadata records. */
inline std::string_view decimal_of(std::size_t value, std::span<char> text) noexcept {
    std::to_chars_result const printed = std::to_chars(text.data(), text.data() + text.size(), value);
    return {text.data(), static_cast<std::size_t>(printed.ptr - text.data())};
}

/** The number a checkpoint's metadata spells in @p text, or zero when it spells none. */
inline std::size_t decimal_in(std::string_view text) noexcept {
    std::size_t value = 0;
    std::from_chars(text.data(), text.data() + text.size(), value);
    return value;
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

/** The GPU ordinal @p variant computes on, where a call's arena must live, or the host's. */
template <typename variant_type_>
int device_of(variant_type_ const& variant) noexcept {
    if constexpr (is_vram_index<variant_type_>) return variant.index.device_specs().ordinal;
    else return host_device_k;
}

#pragma endregion Crossings and Fan Outs

#pragma region Lookups and Cursors

/** The snapshot cursor over the index of the RAM variant @p variant_type_. */
template <typename variant_type_>
using ram_cursor_of =
    basic::ram_cursor<typename variant_type_::index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;

/** A RAM variant's snapshot cursor, and for any other variant an empty slot of its own. */
template <typename variant_type_>
struct basic_cursor_of {
    struct type {};
};
template <typename variant_type_>
    requires is_ram_index<variant_type_>
struct basic_cursor_of<variant_type_> {
    using type = ram_cursor_of<variant_type_>;
};

/** The cursors of every variant of @p variant_type_, and the next row of an NVMe view, whose rows
 *  never change, as one variant of their own. */
template <typename variant_type_>
struct basic_cursors_of;
template <typename... variant_types_>
struct basic_cursors_of<std::variant<variant_types_...>> {
    using type = std::variant<std::monostate, typename basic_cursor_of<variant_types_>::type..., std::size_t>;
};

/** Whether @p traits_type_ measures a key's entries from their values of @p value_type_, which a
 *  count then reads. */
template <typename traits_type_, typename stats_type_, typename value_type_>
concept tallies = requires(stats_type_& stats, value_type_ const& value) { traits_type_::tally(stats, value); };

/** Adds @p value to @p stats, from any thread, where @p traits_type_ measures such values. */
template <typename traits_type_, typename stats_type_, typename value_type_>
void tally_entry(stats_type_& stats, value_type_ const& value) noexcept {
    if constexpr (tallies<traits_type_, stats_type_, value_type_>) traits_type_::tally(stats, value);
}

/** Counts the entries a RAM variant stores under @p key into @p stats, tallying their sizes where
 *  @p traits_type_ measures any. */
template <typename traits_type_, typename variant_type_, typename stats_type_>
    requires is_ram_index<variant_type_>
void count_entries(variant_type_& variant, usearch_key_t key, stats_type_& stats, usearch_executor_t,
                   usearch_outcome_t* outcome) noexcept {
    variant.index.find_each(uuid_of(key), [&](typename variant_type_::index_t::node_id_t node) noexcept {
        ++stats.entries;
        tally_entry<traits_type_>(stats, variant.index.published_value_of(node));
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
    auto const ids = ids_under(variant.index, arguments.key, thread_allocator_t {executor->arena(0)});
    if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    auto value_at = [&](std::size_t entry) noexcept -> decltype(auto) {
        return variant.index.published_value_of(ids[entry]);
    };
    if (!traits_type_::land(variant, ids.size(), value_at, memory, arguments, outcome)) return;
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
    using kept_t = typename traits_type_::kept_t;
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
        keep_first_failure(keeping, traits_type_::keep(*executor, thread, value, kept[rank]));
    };
    if (status_t const read = values_under(variant.index, arguments.key, *executor, carve_entries, keep); failed(read))
        return propagate_error(read, outcome), void();
    if (failed(keeping)) return propagate_error(keeping, outcome), void();
    if (!held) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    auto kept_at = [&](std::size_t entry) noexcept -> decltype(auto) { return kept[entry]; };
    if (!traits_type_::land(variant, held, kept_at, memory, arguments, outcome)) return;
    if (arguments.written) *arguments.written = held;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Opens @p cursor on a RAM variant's snapshot, standing on its first entry so every pull copies
 *  from where it stands; an empty snapshot stays past the end. */
template <typename variant_type_, typename cursor_type_>
    requires is_ram_index<variant_type_>
void open_cursor(variant_type_& variant, cursor_type_& cursor, usearch_outcome_t*) noexcept {
    auto& opened = cursor.template emplace<ram_cursor_of<variant_type_>>(variant.index,
                                                                         forkunion_allocator_t {no_memory_domain_k});
    if (opened.visible_count()) [[maybe_unused]]
        status_t const entered = opened.seek(0);
}

/** Opens @p cursor on an NVMe view's first row, as its rows never change. */
template <typename variant_type_, typename cursor_type_>
    requires is_nvme_index<variant_type_>
void open_cursor(variant_type_&, cursor_type_& cursor, usearch_outcome_t*) noexcept {
    cursor.template emplace<std::size_t>(0);
}

/** Pulls the next keys of a RAM variant's snapshot into @p memory. */
template <typename traits_type_, typename variant_type_, typename cursor_type_, typename arguments_type_>
    requires is_ram_index<variant_type_>
void pull_entries(variant_type_& variant, cursor_type_& cursor, arguments_type_ const& arguments,
                  growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept {
    using node_id_t = typename variant_type_::index_t::node_id_t;
    auto* const typed = std::get_if<ram_cursor_of<variant_type_>>(&cursor);
    if (!typed) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its variant"), void();
    std::size_t const remaining = *typed ? typed->visible_count() - typed->id() : 0;
    std::size_t const taking = min_of(remaining, arguments.wanted);
    usearch_key_t* const keys = carve<usearch_key_t>(memory, taking);
    if (taking && !keys)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
    node_id_t const first = typed->id();
    for (std::size_t entry = 0; entry != taking; ++entry)
        keys[entry] = std::bit_cast<usearch_key_t>(variant.index.key_of(static_cast<node_id_t>(first + entry)).value);
    for (std::size_t step = 0; step != taking; ++step) ++*typed;
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.written = taking;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys of an NVMe view into @p memory through the executor. */
template <typename traits_type_, typename variant_type_, typename cursor_type_, typename arguments_type_>
    requires is_nvme_index<variant_type_>
void pull_entries(variant_type_& variant, cursor_type_& cursor, arguments_type_ const& arguments,
                  growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept {
    std::size_t* const row = std::get_if<std::size_t>(&cursor);
    if (!row) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its variant"), void();
    usearch_executor_handle_t call_executor;
    nvme_executor_t* const executor = executor_of(variant, arguments.executor, call_executor, outcome);
    if (!executor) return;
    std::size_t const taking = min_of(variant.index.size() - *row, arguments.wanted);
    if (!taking) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    usearch_key_t* const keys = carve<usearch_key_t>(memory, taking);
    if (!keys) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
    // The values the read passes along stay unread, as a pull lands keys alone.
    auto skip = [](std::size_t, std::size_t, auto const&) noexcept {};
    std::span<us::utilities::uuid_t> const identities {reinterpret_cast<us::utilities::uuid_t*>(keys), taking};
    if (status_t const read = variant.index.read_rows(*row, identities, *executor, skip); failed(read))
        return propagate_error(read, outcome), void();
    *row += taking;
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.written = taking;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Lookups and Cursors

#if USEARCH_ARCH_CUDA_
#pragma region GPU Variants

/** The executor a null handle runs a GPU variant's work on: a stream of its own on the index's
 *  device, never a legacy default stream. */
template <typename variant_type_>
    requires is_vram_index<variant_type_>
inline usearch_executor_t call_executor_for(variant_type_& variant, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    int const device = variant.index.device_specs().ordinal;
    auto [built, status] = cuda_executor_t::spawn(std::span<int const> {&device, 1});
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor = gpu_executor_t {std::move(built)};
    return &call_executor;
}

/**
 *  @brief What every basic alternative on one NVIDIA GPU holds: the VRAM index over one tape of
 *      values and the key of every row by node ID on the host, which a walk's matches resolve
 *      through.
 *
 *  A batch lands its keys before its rows turn visible, so batches run one at a time. Every walk
 *  rides the executor's stream on the index's device, and progress lands with an inserted batch or
 *  a searched step.
 */
template <typename index_type_>
struct vram_nvidia_variant {
    using executor_t = cuda_executor_t;
    using index_t = index_type_;
    using keys_column_t =
        chunked_column<us::utilities::uuid_t, chunk_schedule<10, 20>, index_t::max_node_count_k, forkunion_allocator_t>;

    index_t index;

    /** The key of each row, by node ID; a batch covers its rows before writing them. */
    keys_column_t keys;

    /** Places the index on CUDA device @p device, handing it @p index_arguments ahead of its
     *  allocator, such as the kernel kind its kits score under. */
    template <typename... index_arguments_types_>
    explicit vram_nvidia_variant(int device, index_arguments_types_... index_arguments) noexcept
        : index(index_arguments..., cuda_virtual_memory_allocator<byte_t> {device}),
          keys(forkunion_allocator_t {no_memory_domain_k}) {}

    /** The plan of a refinement, which no GPU variant opens yet: @ref plan_refine answers
     *  @c unsupported_k, so no call ever runs one. */
    struct refine_plan_t {
        struct {
            std::size_t visible = 0;
        } audit;
        bool landing() const noexcept { return false; }
        bool reselecting() const noexcept { return false; }
        void stop() noexcept {}
        std::span<std::size_t const> steps() const noexcept { return {}; }
        expected<basic::refine_result_t> finish() const noexcept { return {{}, status_t::unsupported_k}; }
    };

    /** The verbs no GPU variant serves yet, each answering @c unsupported_k. */
    status_t clear() noexcept { return status_t::unsupported_k; }
    expected<refine_plan_t> plan_refine(std::span<index_t*>, std::size_t) noexcept {
        return {{}, status_t::unsupported_k};
    }
    status_t refine(refine_plan_t&, progress_ticks_t&, executor_t&) noexcept { return status_t::unsupported_k; }
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    status_t load(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    template <typename arguments_type_>
    status_t sample(arguments_type_ const&) noexcept {
        return status_t::unsupported_k;
    }

  protected:

    /** Reserves @p entries rows of @p connectivity neighbors and @p bytes of values, and their
     *  keys. */
    status_t reserve_(std::size_t entries, std::size_t connectivity, std::size_t bytes) noexcept {
        status_t const reserved = index.reserve({entries, static_cast<connectivity_t>(connectivity), bytes}).status;
        return failed(reserved) ? reserved : keys.cover(entries);
    }

    /** Runs @p walk, as `walk(context, allocator)`, with a traversal context over the scratch of
     *  the executor's stream on the index's device, sized to @p bytes, then counts the batch of
     *  @p count as done. */
    template <typename walk_type_>
    expected<traversal_telemetry_t> walk_(executor_t& executor, std::size_t bytes, std::size_t count,
                                          progress_ticks_t& ticks, walk_type_&& walk) noexcept {
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        typename executor_t::stream_t* const stream = executor.stream_on(index.device_specs().ordinal);
        if (!stream) return {{}, status_t::config_mismatch_k};
        if (status_t const sized = executor.reserve(bytes); failed(sized)) return {{}, sized};
        cuda_arena_allocator_t allocator {stream->arena, *stream};
        typename index_t::traversal_context_t context {allocator};
        if (!context) return {{}, status_t::bad_alloc_k};
        auto const [walked, status] = walk(context, allocator);
        if (failed(status)) return {{}, status};
        [[maybe_unused]] continuation_t const heard = ticks(count - 1);
        return {{.computed_distances = walked.computed_distances,
                 .visited_members = walked.visited_members,
                 .restarts = walked.restarts},
                status_t::success_k};
    }

    /** Lands the keys of the rows from @p first, then inserts the tape @p values and @p offsets
     *  lay out, freezing @p connectivity first unless a reservation did. */
    expected<traversal_telemetry_t> insert_(usearch_keys_view_t keys_view, byte_t const* values,
                                            std::span<std::size_t const> offsets, std::size_t first,
                                            std::size_t connectivity, std::size_t expansion, progress_ticks_t& ticks,
                                            executor_t& executor) noexcept {
        std::size_t const count = offsets.size() - 1;
        if (!index.connectivity())
            if (status_t const reserved =
                    index.reserve({count, static_cast<connectivity_t>(connectivity), offsets[count] - offsets[0]})
                        .status;
                failed(reserved))
                return {{}, reserved};
        if (status_t const covered = keys.cover(first + count); failed(covered)) return {{}, covered};
        for (std::size_t row = 0; row != count; ++row) keys[first + row] = uuid_of(key_at(keys_view, row));
        typename index_t::insert_config_t const config {static_cast<u32_t>(expansion)};
        return walk_(executor, index.traversal_memory_usage(values, offsets, config), count, ticks,
                     [&](typename index_t::traversal_context_t& context, cuda_arena_allocator_t&) noexcept {
                         return index.insert(values, offsets, context, config);
                     });
    }

    /** Searches the @p count queries of the tape @p queries and @p offsets lay out a progress step
     *  at a time, one launch a step over one context, landing each row's matches in @p landing on
     *  host pages; a handler answering cancel stops the steps not begun. */
    expected<traversal_telemetry_t> search_(byte_t const* queries, std::size_t const* offsets, std::size_t count,
                                            std::size_t wanted, std::size_t expansion, matches_landing_t& landing,
                                            progress_ticks_t& ticks, executor_t& executor) noexcept {
        if (!wanted) {
            for (std::size_t row = 0; row != count; ++row) landing.counts[row] = 0;
            return {{}, status_t::success_k};
        }
        using node_id_t = typename index_t::node_id_t;
        using distance_t = typename index_t::distance_t;
        using search_result_t = typename index_t::search_result_t;
        std::span<std::size_t const> const tape {offsets, count + 1};
        typename index_t::search_config_t const config {static_cast<u32_t>(expansion), static_cast<u32_t>(wanted)};
        std::size_t const step = ticks.slice();
        auto search_steps = [&](typename index_t::traversal_context_t& context, cuda_arena_allocator_t&) noexcept {
            search_result_t total;
            for (std::size_t first = 0; first != count;) {
                std::size_t const taking = min_of(step, count - first);
                std::size_t const first_match = first * wanted, matches = taking * wanted;
                // IDs and distances land narrow at the fronts of the step's keys and scores, then
                // widen backward.
                node_id_t* const ids = reinterpret_cast<node_id_t*>(landing.keys + first_match);
                distance_t* const distances = reinterpret_cast<distance_t*>(landing.scores + first_match);
                auto const [found, status] = index.search(queries, tape.subspan(first, taking + 1),
                                                          {ids, {taking, wanted}}, {distances, {taking, wanted}},
                                                          context, config);
                if (failed(status)) return expected<search_result_t> {total, status};
                total.visited_members += found.visited_members;
                total.computed_distances += found.computed_distances;
                total.restarts += found.restarts;
                for (std::size_t row = first; row != first + taking; ++row) landing.counts[row] = wanted;
                for (std::size_t match = matches; match-- != 0;) {
                    node_id_t const id = ids[match];
                    usearch_distance_t const distance = distances[match] * distance_unit<index_t>;
                    if (id == std::numeric_limits<node_id_t>::max())
                        landing.counts[first + match / wanted] = match % wanted;
                    else {
                        landing.keys[first_match + match] = std::bit_cast<usearch_key_t>(keys[id]);
                        landing.scores[first_match + match] = distance;
                    }
                }
                first += taking;
                if (ticks(first, count) == continuation_t::cancel_k)
                    return expected<search_result_t> {total, status_t::cancelled_k};
            }
            return expected<search_result_t> {total, status_t::success_k};
        };
        return walk_(executor, index.traversal_memory_usage(queries, tape, config), count, ticks, search_steps);
    }
};

/** The device a basic GPU variant places on: USearch ordinal @p ordinal of the listing, on a vendor
 *  whose basic kernels this build carries; null once @p outcome records why not. */
inline gpu_device_t const* basic_device_at(std::size_t ordinal, usearch_outcome_t* outcome) noexcept {
    std::span<gpu_device_t const> const listed = gpu_devices();
    if (ordinal >= listed.size())
        return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), nullptr;
    if (listed[ordinal].vendor != gpu_vendor_t::nvidia_k)
        return propagate_error(usearch_status_device_k, outcome, "No basic GPU kernels serve this vendor yet"), nullptr;
    return &listed[ordinal];
}

/** A GPU variant counts no key's entries yet. */
template <typename traits_type_, typename index_type_, typename stats_type_>
void count_entries(vram_nvidia_variant<index_type_>&, usearch_key_t, stats_type_&, usearch_executor_t,
                   usearch_outcome_t* outcome) noexcept {
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
}

/** A GPU variant exports no key's entries yet. */
template <typename traits_type_, typename index_type_, typename arguments_type_>
void get_entries(vram_nvidia_variant<index_type_>&, arguments_type_ const&, growing_arena<forkunion_allocator_t>&,
                 usearch_outcome_t* outcome) noexcept {
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
}

/** A GPU variant opens no cursor yet, so @p cursor stays empty. */
template <typename index_type_, typename cursor_type_>
void open_cursor(vram_nvidia_variant<index_type_>&, cursor_type_&, usearch_outcome_t* outcome) noexcept {
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
}

/** A GPU variant pulls no rows, as it opens no cursor. */
template <typename traits_type_, typename index_type_, typename cursor_type_, typename arguments_type_>
void pull_entries(vram_nvidia_variant<index_type_>&, cursor_type_&, arguments_type_ const&,
                  growing_arena<forkunion_allocator_t>&, usearch_outcome_t* outcome) noexcept {
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
}

#pragma endregion GPU Variants
#endif
