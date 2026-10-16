/**
 *  @file src/basic_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What every basic modality's variants share, whatever they store: the crossings of keys
 *      and tapes, the fan-outs over the executor's threads, and the base of every variant on a
 *      GPU.
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

#include <array>       // `std::array` - a predicate's quad, the one key a view scans for
#include <bit>         // `std::bit_cast` - the C key struct & `uuid_t` share sixteen bytes
#include <charconv>    // `std::to_chars`, `std::from_chars` - the metadata a checkpoint records as text
#include <memory>      // `std::make_obj_using_allocator` - values built from the index's allocator
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`

#include "usearch/basic/ram_index.hpp"
#include "usearch/basic/nvme_index.hpp"
#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a key's entries in insertion order
#include "usearch/utilities/safegraphs_format.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/basic/vram_nvidia_index.hpp"
#endif

#include "lib.hpp"

using us::basic::traversal_telemetry_t;

#pragma region Crossings and Fan Outs

/** The allocator every per-call buffer of one executor thread comes from: its arena, rewound per
 *  verb. */
using thread_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;

/** Candidates one metric round may stage: the frontier rows one round expands, at most
 *  @p connectivity each, where a search derives its rows from 32 links and an insertion from
 *  @p connectivity, and never fewer than the landmarks a walk seeds with or the pool a back-link
 *  re-selects. */
inline std::size_t round_capacity_of(std::size_t expansion, std::size_t connectivity) noexcept {
    std::size_t const narrowest = min_of<std::size_t>(connectivity, connectivity_32_k);
    std::size_t const fan_out = max_of<std::size_t>(divide_round_up(expansion, narrowest), 1);
    return max_of(fan_out * connectivity, max_of<std::size_t>(connectivity, connectivity_32_k) + 1);
}

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
template <typename per_task_type_>
void drain_in_order(forkunion_executor_t& executor, std::size_t tasks, per_task_type_&& per_task) noexcept {
    if (!tasks) return;
    // Every task bumps it, so it takes its own line, away from the locals every thread reads.
    alignas(default_alignment_k) std::size_t cursor = 0;
    auto drain = [&](std::size_t thread, std::size_t) noexcept {
        standard_atomic_ref<std::size_t> const next {cursor};
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

/** Raises @p word to @p value, if it is larger, from any thread. */
template <typename word_type_>
void raise_to(word_type_& word, std::type_identity_t<word_type_> value) noexcept {
    standard_atomic_ref<word_type_> raised {word};
    word_type_ seen = raised.load(std::memory_order_relaxed);
    while (seen < value && !raised.compare_exchange_weak(seen, value, std::memory_order_relaxed));
}

/** One call's per-thread scratch: a traversal context and a metric per executor thread, both over
 *  that thread's arena, so the fan-out never allocates and never shares scratch. */
template <typename index_type_, typename metric_type_>
struct thread_scratch {
    using context_t = typename index_type_::template traversal_context_over<thread_allocator_t>;

    growing_buffer<context_t, forkunion_allocator_t> contexts {forkunion_allocator_t {}};
    growing_buffer<metric_type_, forkunion_allocator_t> metrics {forkunion_allocator_t {}};

    /** Rewinds every arena, keeping the high-water mark of earlier calls, and builds one context
     *  sized for @p connectivity and @p expansion plus one metric from @p make_metric per
     *  thread. */
    template <typename make_metric_type_>
    status_t prepare(forkunion_executor_t& executor, std::size_t connectivity, std::size_t expansion, u64_t seed,
                     make_metric_type_&& make_metric) noexcept {
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        std::size_t const threads = executor.size();
        if (failed(contexts.reserve_exactly(threads)) || failed(metrics.reserve_exactly(threads)))
            return status_t::bad_alloc_k;
        std::size_t const staging = round_capacity_of(expansion, connectivity);
        typename index_type_::insert_config_t const walk {expansion, 0};
        for (std::size_t thread = 0; thread != threads; ++thread) {
            thread_allocator_t const allocator {executor.arena(thread)};
            contexts.push_back_reserved(context_t {seed + thread, allocator});
            if (status_t const sized = index_type_::reserve_traversal(contexts[thread], connectivity, walk);
                failed(sized))
                return sized;
            auto [metric, make_status] = make_metric(allocator, staging);
            if (failed(make_status)) return make_status;
            metrics.push_back_reserved(std::move(metric));
        }
        return status_t::success_k;
    }

    /** The counters every thread's walks accumulated. */
    traversal_telemetry_t telemetry() const noexcept {
        traversal_telemetry_t total;
        for (context_t const& context : contexts) {
            total.computed_distances += context.telemetry.computed_distances;
            total.visited_members += context.telemetry.visited_members;
            total.restarts += context.telemetry.restarts;
            total.saturated_walks += context.telemetry.saturated_walks;
        }
        return total;
    }
};

/**
 *  @brief Inserts @p count entries under their keys as one batch: every thread builds its share of
 *      the values from the index's allocator, then one plan lands them on every thread and
 *      publishes once.
 *
 *  Everything fallible runs before the plan dispenses: the keys and values, the contexts and the
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
    using metric_t = decltype(make_metric(std::declval<thread_allocator_t const&>(), std::size_t {}).value);
    thread_scratch<index_type_, metric_t> scratch;
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
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        if (succeeded(standard_atomic_ref<status_t>(built).load(std::memory_order_relaxed)))
            keep_first_failure(built, build(row, values[row], thread));
    });
    if (failed(built)) return {{}, built};

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    basic::keep_diverse_t const selector {connectivity};
    typename index_type_::insert_config_t const config {expansion, 0};
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        plan.insert(row, to_writable_span(values).subspan(row, 1), to_span(identities).subspan(row, 1),
                    scratch.metrics[thread], scratch.contexts[thread], selector, config);
        [[maybe_unused]] continuation_t const heard = ticks(row);
    });
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    return {scratch.telemetry(), status_t::success_k};
}

/** Where a refinement's calls stand between them: the steps of the current phase already run, so a
 *  cancelled call continues at the next one, and the nodes the planned phases re-selected, which
 *  every tick reports. */
struct refine_cursor_t {
    std::size_t taken = 0;
    std::size_t ticked = 0;
};

/**
 *  @brief Runs @p plan 's pending phases on the executor's threads, each with a metric from
 *      @p make_metric and a walk context over its arena, until no phase follows or @p ticks
 *      cancels; @c cancelled_k means a later call has more to do.
 *
 *  Every step a thread takes runs, and a thread takes none once a tick cancelled, so @p cursor
 *  says exactly where the next call starts. The scratch is this call's, sized for @p row_width,
 *  the widest row a walk copies; the plan keeps everything the index needs between calls.
 */
template <typename index_type_, typename make_metric_type_, typename selector_type_>
status_t drive_refinement(typename index_type_::refine_plan_t& plan, refine_cursor_t& cursor,
                          make_metric_type_&& make_metric, selector_type_ const& selector, std::size_t row_width,
                          progress_ticks_t& ticks, forkunion_executor_t& executor) noexcept {
    using metric_t = decltype(make_metric(std::declval<thread_allocator_t const&>(), std::size_t {}).value);
    thread_scratch<index_type_, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, max_of(row_width, plan.connectivity()),
                                                  plan.walk().expansion_factor, plan.audit.visible + cursor.ticked,
                                                  make_metric);
        failed(prepared))
        return prepared;
    std::size_t const population = plan.audit.visible;
    // Every step bumps them, so they take their own lines, away from the locals every thread reads.
    alignas(default_alignment_k) std::size_t ticked = cursor.ticked;
    alignas(default_alignment_k) std::size_t taken = cursor.taken;
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
                [[maybe_unused]] status_t const ran = plan.run(step, scratch.metrics[thread], scratch.contexts[thread],
                                                               selector);
                if (!reselecting) continue;
                std::size_t const done = standard_atomic_ref<std::size_t>(ticked).fetch_add(step.nodes.size(),
                                                                                            std::memory_order_relaxed) +
                                         step.nodes.size();
                if (ticks(done, population) == continuation_t::cancel_k)
                    standard_atomic_ref<bool>(cancelled).store(true, std::memory_order_relaxed);
            }
        };
        if (failed(executor.for_threads(drain))) drain(0, 0);
        cursor.ticked = ticked;
        cursor.taken = min_of(taken, steps.size());
        if (cancelled) return status_t::cancelled_k;
        cursor.taken = taken = 0;
        if (!plan.advance()) return status_t::success_k;
    }
}

/** What one unit of @p metric_type_'s distance is worth on the C scale: a fixed-point metric's full
 *  scale lands as one. */
template <typename metric_type_>
constexpr usearch_distance_t distance_unit_of() noexcept {
    if constexpr (requires { metric_type_::resolution_k; }) return 1.0 / metric_type_::resolution_k;
    else return 1;
}

/** Lands one query's matches in row @p row of @p landing: the pool's nearest @p wanted, or with a
 *  @p predicate the nearest of those it admits, judged a quad at a time. */
template <typename metric_type_, typename index_type_, typename context_type_>
void land_matches(index_type_ const& index, context_type_ const& context, std::size_t row, std::size_t wanted,
                  usearch_predicate_t predicate, matches_landing_t& landing) noexcept {
    usearch_key_t* const keys = landing.keys + row * wanted;
    usearch_distance_t* const scores = landing.scores + row * wanted;
    std::size_t const pooled = context.results.size();
    std::size_t landed = 0;
    for (std::size_t first = 0; first < pooled && landed != wanted; first += 4) {
        std::size_t const quad = min_of<std::size_t>(4, pooled - first);
        std::array<us::utilities::uuid_t, 4> identities;
        std::array<admission_t, 4> verdicts;
        verdicts.fill(admission_t::admit_k);
        for (std::size_t lane = 0; lane != quad; ++lane)
            identities[lane] = index.key_of(context.results[first + lane].node_id).value;
        if (predicate.callback)
            key_predicate_t {predicate}(std::span {identities}.first(quad), std::span {verdicts}.first(quad));
        for (std::size_t lane = 0; lane != quad && landed != wanted; ++lane) {
            if (verdicts[lane] != admission_t::admit_k) continue;
            keys[landed] = std::bit_cast<usearch_key_t>(identities[lane]);
            scores[landed] = static_cast<usearch_distance_t>(context.results[first + lane].distance) *
                             distance_unit_of<metric_type_>();
            ++landed;
        }
    }
    landing.counts[row] = landed;
}

/**
 *  @brief Searches @p count queries on the executor's threads, each built by @p query_of, which
 *      hands the query it prepared to a continuation, and lands every row in @p landing.
 *
 *  A predicate narrows each row to the entries it admits among the walk's pool, so a selective one
 *  wants a wider @p expansion. A handler answering cancel stops the queries no thread has taken
 *  yet.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_telemetry_t> search_entries(index_type_& index, std::size_t count, std::size_t wanted,
                                               std::size_t connectivity, std::size_t expansion,
                                               usearch_predicate_t predicate, make_metric_type_&& make_metric,
                                               query_of_type_&& query_of, matches_landing_t& landing,
                                               progress_ticks_t& ticks, forkunion_executor_t& executor) noexcept {
    using metric_t = decltype(make_metric(std::declval<thread_allocator_t const&>(), std::size_t {}).value);
    // Pools narrower than the row they fill would short every row.
    std::size_t const breadth = max_of(expansion ? expansion : std::size_t(connectivity_32_k) * 2, wanted);
    thread_scratch<index_type_, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, connectivity, breadth, index.size(), make_metric);
        failed(prepared))
        return {{}, prepared};
    typename index_type_::search_config_t const config {breadth, 0};
    status_t searched = status_t::success_k;
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        if (failed(standard_atomic_ref<status_t>(searched).load(std::memory_order_relaxed))) return;
        auto& context = scratch.contexts[thread];
        status_t const answered = query_of(row, thread, [&](auto const& query) noexcept {
            auto const [found, status] = index.search(query, scratch.metrics[thread], context, config);
            if (failed(status)) return status;
            land_matches<metric_t>(index, context, row, wanted, predicate, landing);
            return status_t::success_k;
        });
        keep_first_failure(searched, answered);
        if (ticks(row) == continuation_t::cancel_k) keep_first_failure(searched, status_t::cancelled_k);
    });
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
 *  @brief Searches @p count queries on an NVMe view, the walks of many in flight on every thread of
 *      @p executor, each query built by @p query_of as the view takes it, and lands every row in
 *      @p landing; progress lands with the batch.
 *
 *  A predicate narrows each row to the entries it admits among the walk's pool, so a selective one
 *  wants a wider @p expansion.
 */
template <typename index_type_, typename make_metric_type_, typename query_of_type_>
expected<traversal_telemetry_t> search_entries(index_type_& index, std::size_t count, std::size_t wanted,
                                               std::size_t expansion, usearch_predicate_t predicate,
                                               make_metric_type_&& make_metric, query_of_type_&& query_of,
                                               matches_landing_t& landing, progress_ticks_t& ticks,
                                               nvme_executor_t& executor) noexcept {
    using key_t = typename index_type_::key_t;
    using metric_t = decltype(make_metric(std::declval<thread_allocator_t const&>(), std::size_t {}).value);
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walks resolve them");
    // Pools narrower than the row they fill would short every row.
    typename index_type_::search_config_t const config {max_of(index.non_zero_expansion_factor({expansion}), wanted)};
    std::span<key_t> const keys {reinterpret_cast<key_t*>(landing.keys), count * wanted};
    std::span<usearch_distance_t> const scores {landing.scores, count * wanted};
    std::span<std::size_t> const counts {landing.counts, count};
    auto const [found, status] = [&]() noexcept {
        if (!predicate.callback)
            return index.search(count, make_metric, query_of, executor, config, keys, scores, counts);
        key_predicate_t judge {predicate};
        return index.search(count, make_metric, query_of, executor, config, keys, scores, counts, judge);
    }();
    if (failed(status)) return {{}, status};
    for (std::size_t row = 0; row != count; ++row)
        for (std::size_t match = 0; match != counts[row]; ++match)
            scores[row * wanted + match] *= distance_unit_of<metric_t>();
    [[maybe_unused]] continuation_t const heard = ticks(count - 1);
    return {{found.computed_distances, found.visited_members, found.restarts, found.saturated_walks},
            status_t::success_k};
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

#pragma endregion Crossings and Fan Outs

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
 *  rides the executor's stream on the index's device, and progress lands with the batch.
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

    /** Places the index on CUDA device @p device. */
    explicit vram_nvidia_variant(int device) noexcept
        : index(cuda_virtual_memory_allocator<byte_t> {device}), keys(forkunion_allocator_t {no_memory_domain_k}) {}

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
        return {{walked.computed_distances, walked.visited_members, walked.restarts, 0}, status_t::success_k};
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

    /** Searches the @p count queries of the tape @p queries and @p offsets lay out in one launch,
     *  landing each row's matches in @p landing on host pages. */
    expected<traversal_telemetry_t> search_(byte_t const* queries, std::size_t const* offsets, std::size_t count,
                                            std::size_t wanted, std::size_t expansion, matches_landing_t& landing,
                                            progress_ticks_t& ticks, executor_t& executor) noexcept {
        if (!wanted) {
            for (std::size_t row = 0; row != count; ++row) landing.counts[row] = 0;
            return {{}, status_t::success_k};
        }
        using node_id_t = typename index_t::node_id_t;
        using distance_t = typename index_t::distance_t;
        std::span<std::size_t const> const tape {offsets, count + 1};
        typename index_t::search_config_t const config {static_cast<u32_t>(expansion), static_cast<u32_t>(wanted)};
        // IDs and distances land narrow at the fronts of keys and scores, then widen backward.
        node_id_t* const ids = reinterpret_cast<node_id_t*>(landing.keys);
        distance_t* const distances = reinterpret_cast<distance_t*>(landing.scores);
        expected<traversal_telemetry_t> const searched = walk_(
            executor, index.traversal_memory_usage(queries, tape, config), count, ticks,
            [&](typename index_t::traversal_context_t& context, cuda_arena_allocator_t&) noexcept {
                return index.search(queries, tape, {ids, {count, wanted}}, {distances, {count, wanted}}, context,
                                    config);
            });
        if (failed(searched.status)) return searched;
        for (std::size_t row = 0; row != count; ++row) landing.counts[row] = wanted;
        for (std::size_t match = count * wanted; match-- != 0;) {
            node_id_t const id = ids[match];
            usearch_distance_t const distance = distances[match];
            if (id == us::utilities::node_id_unknown_k) landing.counts[match / wanted] = match % wanted;
            else {
                landing.keys[match] = std::bit_cast<usearch_key_t>(keys[id]);
                landing.scores[match] = distance;
            }
        }
        return searched;
    }
};

#pragma endregion GPU Variants
#endif
