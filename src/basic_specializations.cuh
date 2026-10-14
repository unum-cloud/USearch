/**
 *  @file src/basic_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What one sequence or shapes arm is spelled with and compiled through: the CPU classes
 *      whose kits differ, the arms over @c basic::ram_index, and the fan-outs both modalities
 *      share.
 *
 *  Every basic class unit includes this to instantiate its arms; @c sequence_index.cpp and
 *  @c shapes_index.cpp add the variants, the handles and the exported C entry points on top of it.
 *  The basic index is the same type in every class, so a class differs only in the kits its metrics
 *  score with, and the arms keep the index's default waiter and atomic reference.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <array>       // `std::array` - a predicate's quad, the one cloud a query packs
#include <bit>         // `std::bit_cast` - the C key struct & `uuid_t` share sixteen bytes
#include <charconv>    // `std::to_chars`, `std::from_chars` - the metadata a checkpoint records as text
#include <memory>      // `std::make_obj_using_allocator` - values built from the index's allocator
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`

#include "usearch/sequence/ram_index.hpp"
#include "usearch/shapes/ram_index.hpp"
#include "usearch/geospatial/ram_metric.hpp"
#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a key's entries in insertion order
#include "usearch/utilities/safegraphs_format.hpp"

#include "lib.hpp"

using us::basic::traversal_telemetry_t;

#pragma region Classes

/** Lowers one CPU class onto StringZilla's capability bits, which name backends rather than
 *  accumulate, so each class spells every backend its kits may pick. */
constexpr sz_capability_t stringzilla_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_haswell_k: return static_cast<sz_capability_t>(sz_cap_serial_k | sz_cap_haswell_k);
    case cpu_class_t::intel_skylake_k:
        return static_cast<sz_capability_t>(sz_cap_serial_k | sz_cap_haswell_k | sz_cap_skylake_k);
    case cpu_class_t::aws_g2_k: return static_cast<sz_capability_t>(sz_cap_serial_k | sz_cap_neon_k);
    default: return sz_cap_serial_k;
    }
}

/** Whether one sequence rung picks a different Levenshtein or overlap kit than the class it
 *  degrades to - the one fact justifying its arm. */
template <cpu_class_t cpu_class_>
consteval bool sequence_rung_advances() noexcept {
    constexpr sz_capability_t upper_k = stringzilla_capabilities_of(cpu_class_);
    constexpr sz_capability_t lower_k = stringzilla_capabilities_of(parent_class_of(cpu_class_));
    return !std::is_same_v<us::sequence::levenshtein_kit<upper_k>, us::sequence::levenshtein_kit<lower_k>> ||
           !std::is_same_v<us::sequence::overlap_kit<upper_k>, us::sequence::overlap_kit<lower_k>>;
}

/** Whether one shapes rung picks a different kit for @p scalar_type_ than the class it degrades
 *  to. */
template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
consteval bool shapes_rung_advances() noexcept {
    using upper_t = us::shapes::shapes_kit<scalar_type_, numkong_capabilities_of(cpu_class_)>;
    using lower_t = us::shapes::shapes_kit<scalar_type_, numkong_capabilities_of(parent_class_of(cpu_class_))>;
    return !std::is_same_v<upper_t, lower_t>;
}

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

#pragma endregion Classes

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

#pragma endregion Crossings and Fan Outs

#pragma region Sequence Arms

/** The checkpoint token of one sequence kind, spelled as the SafeGraphs registry spells it; empty
 *  for the kinds the engine refuses. */
inline std::string_view token_of(usearch_sequence_metric_t metric) noexcept {
    switch (metric) {
    case usearch_sequence_metric_levenshtein_bytes_k: return "levenshtein";
    case usearch_sequence_metric_levenshtein_utf8_k: return "levenshtein-utf8";
    case usearch_sequence_metric_overlap_k: return "overlap";
    default: return {};
    }
}

/** The sequence kind a checkpoint token names, or @c unknown_k. */
inline usearch_sequence_metric_t sequence_metric_of(std::string_view token) noexcept {
    for (usearch_sequence_metric_t metric :
         {usearch_sequence_metric_levenshtein_bytes_k, usearch_sequence_metric_levenshtein_utf8_k,
          usearch_sequence_metric_overlap_k})
        if (token == token_of(metric)) return metric;
    return usearch_sequence_metric_unknown_k;
}

/** One sequence alternative: the basic index over owned strings, the kind it scores with, and the
 *  kits @p cpu_class_ picks. Its verbs are defined below the class, so an explicit instantiation in
 *  the class's unit compiles them with that class's flags and an @c extern one keeps them out of
 *  the dispatch unit. */
template <cpu_class_t cpu_class_>
struct sequence_arm {
    using executor_t = forkunion_executor_t;
    using index_t = us::sequence::ram_index<us::utilities::uuid_t, u32_t, dummy_waiter_t, standard_atomic_ref,
                                            forkunion_allocator_t>;
    using value_t = typename index_t::value_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;

    /** The kind every walk scores with. */
    usearch_sequence_metric_t metric = usearch_sequence_metric_levenshtein_bytes_k;

    /** The window widths the overlap kind hashes, in bytes; empty for every other kind. */
    growing_buffer<std::size_t, forkunion_allocator_t> window_widths;

    /** Neighbors per node the insertions select. */
    std::size_t connectivity = connectivity_32_k;

    explicit sequence_arm(forkunion_allocator_t const& allocator) noexcept
        : index(allocator), window_widths(allocator) {}

    expected<traversal_telemetry_t> insert(usearch_keys_view_t keys, usearch_tape_view_t sequences, std::size_t count,
                                           std::size_t expansion, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_tape_view_t queries, std::size_t count, std::size_t wanted,
                                           std::size_t expansion, usearch_predicate_t predicate,
                                           matches_landing_t& landing, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

  private:

    /** Calls @p callback with a factory of this arm's metric over a thread's arena, sized for a
     *  round of candidates, whichever of the three kinds it scores with. */
    template <typename callback_type_>
    decltype(auto) with_metric_(callback_type_&& callback) noexcept {
        using bytes_t = us::sequence::levenshtein_metric<capability_k, sz_levenshtein_bytes_k, thread_allocator_t>;
        using runes_t = us::sequence::levenshtein_metric<capability_k, sz_levenshtein_runes_k, thread_allocator_t>;
        using overlap_t = us::sequence::overlap_metric<capability_k, thread_allocator_t>;
        std::span<std::size_t const> const widths = to_span(window_widths);
        auto levenshtein = [&]<typename metric_type_>(std::type_identity<metric_type_>) noexcept {
            return callback([](thread_allocator_t const& allocator, std::size_t staging) noexcept {
                metric_type_ metric {allocator};
                status_t const ready = metric.init(staging);
                return expected<metric_type_> {std::move(metric), ready};
            });
        };
        switch (metric) {
        case usearch_sequence_metric_levenshtein_utf8_k: return levenshtein(std::type_identity<runes_t> {});
        case usearch_sequence_metric_overlap_k:
            return callback([widths](thread_allocator_t const& allocator, std::size_t staging) noexcept {
                overlap_t metric {allocator};
                status_t const ready = metric.init(staging, widths);
                return expected<overlap_t> {std::move(metric), ready};
            });
        default: return levenshtein(std::type_identity<bytes_t> {});
        }
    }
};

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> sequence_arm<cpu_class_>::insert(usearch_keys_view_t keys,
                                                                 usearch_tape_view_t sequences, std::size_t count,
                                                                 std::size_t expansion, progress_ticks_t& ticks,
                                                                 executor_t& executor) noexcept {
    return with_metric_([&](auto make_metric) noexcept {
        auto build = [&](std::size_t row, value_t& value, std::size_t) noexcept {
            return us::sequence::status_of(value.assign(chars_at(sequences, row)));
        };
        return insert_entries(index, keys, count, connectivity, expansion ? expansion : connectivity * 2, make_metric,
                              build, ticks, executor);
    });
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> sequence_arm<cpu_class_>::search(usearch_tape_view_t queries, std::size_t count,
                                                                 std::size_t wanted, std::size_t expansion,
                                                                 usearch_predicate_t predicate,
                                                                 matches_landing_t& landing, progress_ticks_t& ticks,
                                                                 executor_t& executor) noexcept {
    return with_metric_([&](auto make_metric) noexcept {
        auto query_of = [&](std::size_t row, std::size_t, auto&& answer) noexcept {
            return answer(chars_at(queries, row));
        };
        return search_entries(index, count, wanted, connectivity, expansion, predicate, make_metric, query_of, landing,
                              ticks, executor);
    });
}

template <cpu_class_t cpu_class_>
status_t sequence_arm<cpu_class_>::save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
    std::array<char, 24> connectivity_text;
    growing_buffer<char, forkunion_allocator_t> widths_text {forkunion_allocator_t {}};
    if (failed(widths_text.reserve_exactly(window_widths.size() * 21))) return status_t::bad_alloc_k;
    for (std::size_t const width : window_widths) {
        std::array<char, 21> width_text;
        if (!widths_text.empty()) widths_text.push_back_reserved(',');
        for (char const digit : decimal_of(width, width_text)) widths_text.push_back_reserved(digit);
    }
    std::array<safegraphs_custom_field_t, 2> const fields {{
        {"connectivity", decimal_of(connectivity, connectivity_text)},
        {"window_widths", to_string_view(widths_text)},
    }};
    save_config_t config;
    config.provenance = std::span {fields}.first(window_widths.empty() ? 1 : 2);
    return safegraphs_write_index(path, index, us::sequence::codec_t {}, executor, config, token_of(metric), ticks);
}

template <cpu_class_t cpu_class_>
status_t sequence_arm<cpu_class_>::load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
    return safegraphs_read_index(path, index, us::sequence::codec_t {}, executor, load_config_t {}, ticks);
}

#pragma endregion Sequence Arms

#pragma region Shapes Arms

/** The checkpoint token of one shapes kind, spelled as NumKong spells its kernels. */
inline std::string_view token_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_chamfer_k: return "chamfer";
    case usearch_shapes_metric_hausdorff_k: return "hausdorff";
    case usearch_shapes_metric_gromov_wasserstein_k: return "gromov_wasserstein";
    case usearch_shapes_metric_wasserstein_procrustes_k: return "wasserstein_procrustes";
    default: return {};
    }
}

/** The shapes kind a checkpoint token names, or @c unknown_k. */
inline usearch_shapes_metric_t shapes_metric_of(std::string_view token) noexcept {
    for (usearch_shapes_metric_t metric :
         {usearch_shapes_metric_chamfer_k, usearch_shapes_metric_hausdorff_k,
          usearch_shapes_metric_gromov_wasserstein_k, usearch_shapes_metric_wasserstein_procrustes_k})
        if (token == token_of(metric)) return metric;
    return usearch_shapes_metric_unknown_k;
}

/** The NumKong kernel one shapes kind scores with. */
constexpr nk_kernel_kind_t kernel_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_hausdorff_k: return nk_kernel_hausdorffs_packed_k;
    case usearch_shapes_metric_gromov_wasserstein_k: return nk_kernel_gromovwassersteins_packed_k;
    case usearch_shapes_metric_wasserstein_procrustes_k: return nk_kernel_wassersteinprocrustes_packed_k;
    default: return nk_kernel_chamfers_packed_k;
    }
}

/** One shapes alternative: the basic index over packs of @p scalar_type_, the kind it scores with,
 *  and the kits @p cpu_class_ picks; verbs defined below the class, as the sequence arm's are. */
template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_arm {
    using executor_t = forkunion_executor_t;
    using scalar_t = scalar_type_;
    using kit_t = us::shapes::shapes_kit<scalar_type_, numkong_capabilities_of(cpu_class_)>;
    using index_t =
        us::shapes::ram_index<kit_t, us::utilities::uuid_t, dummy_waiter_t, standard_atomic_ref, forkunion_allocator_t>;
    using value_t = typename index_t::value_t;
    using codec_t = us::shapes::codec<scalar_type_>;
    using metric_t = us::shapes::shapes_metric<kit_t, thread_allocator_t>;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr nk_dtype_t dtype_k = std::is_same_v<scalar_type_, nk::f16_t> ? nk_f16_k : nk_f32_k;

    index_t index;

    /** The kind every walk scores with. */
    usearch_shapes_metric_t metric = usearch_shapes_metric_chamfer_k;

    /** The frozen point count, or zero for any count. */
    std::size_t points = 0;

    /** Neighbors per node the insertions select. */
    std::size_t connectivity = connectivity_32_k;

    /** Points across every stored entry, added to by insertions and loads through @c atomic_ref. */
    std::size_t total_points = 0;

    explicit shapes_arm(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    expected<traversal_telemetry_t> insert(usearch_keys_view_t keys, usearch_tape_view_t shapes, std::size_t count,
                                           std::size_t expansion, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_tape_view_t queries, std::size_t count, std::size_t wanted,
                                           std::size_t expansion, usearch_predicate_t predicate,
                                           matches_landing_t& landing, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;

    /** Writes the @p entries points of one stored value into @p output in @p dtype, through
     *  @p scratch when the dtype is not the stored one. */
    status_t export_points(value_t const& value, nk_dtype_t dtype, void* output,
                           thread_allocator_t const& scratch) const noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

  private:

    /** Packs entry @p row of @p tape into @p packed under @p allocator, casting the caller's scalar
     *  into the stored one through @p scratch where they differ; refuses a count other than a
     *  frozen one. */
    template <typename packed_type_, typename allocator_type_>
    status_t pack_(usearch_tape_view_t tape, std::size_t row, packed_type_& packed, allocator_type_ const& allocator,
                   thread_allocator_t const& scratch) const noexcept;

    /** A factory of this arm's metric over a thread's arena, sized for a round of candidates. */
    auto metric_factory_() const noexcept {
        return [kernel = kernel_of(metric)](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            metric_t made {kernel, allocator};
            status_t const ready = made.init(staging);
            return expected<metric_t> {std::move(made), ready};
        };
    }
};

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
template <typename packed_type_, typename allocator_type_>
status_t shapes_arm<scalar_type_, cpu_class_>::pack_(usearch_tape_view_t tape, std::size_t row, packed_type_& packed,
                                                     allocator_type_ const& allocator,
                                                     thread_allocator_t const& scratch) const noexcept {
    std::size_t const count = tape.offsets[row + 1] - tape.offsets[row];
    if (points && count != points) return status_t::shape_mismatch_k;
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    std::size_t const elements = count * 3;
    bytes_view_t const coordinates = bytes_of(
        static_cast<byte_t const*>(tape.data) + tape.offsets[row] * 3 * from_bytes, elements * from_bytes);
    growing_buffer<scalar_type_, thread_allocator_t> converted {scratch};
    std::span<scalar_type_ const> cloud = elements_of<scalar_type_>(coordinates);
    if (from != dtype_k) {
        if (failed(converted.resize_uninitialized(elements))) return status_t::bad_alloc_k;
        if (status_t const cast = us::shapes::status_of(
                kit_t::cast(coordinates.data(), from, converted.data(), dtype_k, elements, nullptr));
            failed(cast))
            return cast;
        cloud = to_span(converted);
    }
    std::array const clouds {cloud};
    auto [made, status] = packed_type_::make(clouds, allocator, 0);
    if (nk::failed(status)) return us::shapes::status_of(status);
    packed = std::move(made);
    return status_t::success_k;
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
expected<traversal_telemetry_t> shapes_arm<scalar_type_, cpu_class_>::insert(usearch_keys_view_t keys,
                                                                             usearch_tape_view_t shapes,
                                                                             std::size_t count, std::size_t expansion,
                                                                             progress_ticks_t& ticks,
                                                                             executor_t& executor) noexcept {
    auto build = [&](std::size_t row, value_t& value, std::size_t thread) noexcept {
        return pack_(shapes, row, value, value.get_allocator(), thread_allocator_t {executor.arena(thread)});
    };
    expected<traversal_telemetry_t> const inserted = insert_entries(index, keys, count, connectivity,
                                                                    expansion ? expansion : connectivity * 2,
                                                                    metric_factory_(), build, ticks, executor);
    if (inserted)
        standard_atomic_ref<std::size_t>(total_points)
            .fetch_add(shapes.offsets[count] - shapes.offsets[0], std::memory_order_relaxed);
    return inserted;
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
expected<traversal_telemetry_t> shapes_arm<scalar_type_, cpu_class_>::search(
    usearch_tape_view_t queries, std::size_t count, std::size_t wanted, std::size_t expansion,
    usearch_predicate_t predicate, matches_landing_t& landing, progress_ticks_t& ticks, executor_t& executor) noexcept {
    using query_t = nk::shapes_packed<scalar_type_,
                                      typename std::allocator_traits<thread_allocator_t>::template rebind_alloc<char>>;
    auto query_of = [&](std::size_t row, std::size_t thread, auto&& answer) noexcept {
        thread_allocator_t const scratch {executor.arena(thread)};
        query_t query {scratch};
        if (status_t const packed = pack_(queries, row, query, scratch, scratch); failed(packed)) return packed;
        return answer(query);
    };
    return search_entries(index, count, wanted, connectivity, expansion, predicate, metric_factory_(), query_of,
                          landing, ticks, executor);
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_arm<scalar_type_, cpu_class_>::export_points(value_t const& value, nk_dtype_t dtype, void* output,
                                                             thread_allocator_t const& scratch) const noexcept {
    if (value.empty()) return status_t::success_k;
    std::size_t const elements = value.max_points() * 3;
    if (dtype == dtype_k)
        return us::shapes::status_of(
            nk::shapes_unpack<scalar_type_>(value[0], static_cast<typename scalar_type_::raw_t*>(output), 0));
    growing_buffer<typename scalar_type_::raw_t, thread_allocator_t> unpacked {scratch};
    if (failed(unpacked.resize_uninitialized(elements))) return status_t::bad_alloc_k;
    if (status_t const opened = us::shapes::status_of(nk::shapes_unpack<scalar_type_>(value[0], unpacked.data(), 0));
        failed(opened))
        return opened;
    return us::shapes::status_of(kit_t::cast(unpacked.data(), dtype_k, output, dtype, elements, nullptr));
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_arm<scalar_type_, cpu_class_>::save(std::string_view path, executor_t& executor,
                                                    progress_ticks_t& ticks) noexcept {
    std::array<char, 24> connectivity_text, points_text;
    std::array<safegraphs_custom_field_t, 2> const fields {{
        {"connectivity", decimal_of(connectivity, connectivity_text)},
        {"points", decimal_of(points, points_text)},
    }};
    save_config_t config;
    config.provenance = fields;
    return safegraphs_write_index(path, index, codec_t {}, executor, config, token_of(metric), ticks);
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_arm<scalar_type_, cpu_class_>::load(std::string_view path, executor_t& executor,
                                                    progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = safegraphs_read_index(path, index, codec_t {}, executor, load_config_t {}, ticks);
        failed(loaded))
        return loaded;
    std::size_t const nodes = index.size();
    for (std::size_t node = 0; node != nodes; ++node)
        total_points += index.published_value_of(static_cast<typename index_t::node_id_t>(node)).max_points();
    return status_t::success_k;
}

#pragma endregion Shapes Arms

#pragma region Geospatial Arms

/** Whether one geospatial rung picks a different kit than the class it degrades to - the one fact
 *  justifying its arm. */
template <cpu_class_t cpu_class_>
consteval bool geospatial_rung_advances() noexcept {
    using upper_t = us::geospatial::geospatial_kit<numkong_capabilities_of(cpu_class_)>;
    using lower_t = us::geospatial::geospatial_kit<numkong_capabilities_of(parent_class_of(cpu_class_))>;
    return !std::is_same_v<upper_t, lower_t>;
}

/** The engine's form of a C observation, field by field. */
inline us::geospatial::geodetic_observation_t geodetic_of(
    usearch_geospatial_observation_t const& observation) noexcept {
    return {observation.latitude_degrees,
            observation.longitude_degrees,
            observation.altitude_metres,
            observation.accuracy_metres,
            observation.heading_degrees,
            observation.pitch_degrees,
            observation.heading_accuracy_degrees,
            observation.field_of_view_degrees,
            observation.range_metres,
            observation.begin_seconds,
            observation.end_seconds};
}

/** The C form of an engine observation, field by field. */
inline usearch_geospatial_observation_t observation_of(
    us::geospatial::geodetic_observation_t const& observation) noexcept {
    return {observation.latitude_degrees,
            observation.longitude_degrees,
            observation.altitude_metres,
            observation.accuracy_metres,
            observation.heading_degrees,
            observation.pitch_degrees,
            observation.heading_accuracy_degrees,
            observation.field_of_view_degrees,
            observation.range_metres,
            observation.begin_seconds,
            observation.end_seconds};
}

/** One containment match before it lands: the query it answers, the entry's key and its chord in
 *  metres. */
struct geospatial_within_match_t {
    std::size_t query;
    usearch_key_t key;
    usearch_distance_t score;
};

/**
 *  @brief One geospatial arm: the basic index over observations, scored with one class's kit, with
 *      the terrain its lines of sight run over.
 *
 *  A borrowed terrain is packed once, at @ref adopt_terrain, and read in place by every call; a
 *  terrain filled by a callback is filled and packed per call, over the window the call's positions
 *  span.
 */
template <cpu_class_t cpu_class_>
struct geospatial_arm {
    using executor_t = forkunion_executor_t;
    using kit_t = us::geospatial::geospatial_kit<numkong_capabilities_of(cpu_class_)>;
    using index_t =
        us::geospatial::ram_index<us::utilities::uuid_t, dummy_waiter_t, standard_atomic_ref, forkunion_allocator_t>;
    using value_t = us::geospatial::earth_observation_t;
    using node_id_t = typename index_t::node_id_t;
    using metric_t = us::geospatial::chord_metric<kit_t, thread_allocator_t>;
    using relation_t = us::geospatial::relation_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;

    index_t index;

    /** Neighbors per node the insertions select: a winner per cone, and long links for the rest. */
    std::size_t connectivity = us::geospatial::surface_keep_octants_t {}.connectivity;

    /** The largest stored deviation in whole metres, raised through @c atomic_ref; it widens every
     *  radius walk. */
    u32_t largest_deviation_metres = 0;

    /** The grid lines of sight run over, empty for none. */
    us::geospatial::terrain_t terrain;

    /** The maxima of a borrowed grid, packed once. */
    growing_buffer<byte_t, forkunion_allocator_t> terrain_pack {forkunion_allocator_t {}};

    explicit geospatial_arm(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Adopts @p grid, packing it when it is borrowed whole; refuses a grid with no extent or
     *  steps. */
    status_t adopt_terrain(us::geospatial::terrain_t const& grid) noexcept;

    expected<traversal_telemetry_t> insert(usearch_keys_view_t keys,
                                           usearch_geospatial_observation_t const* observations, std::size_t count,
                                           std::size_t expansion, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_geospatial_observation_t const* queries, std::size_t count,
                                           relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                           std::size_t wanted, std::size_t expansion, usearch_predicate_t predicate,
                                           matches_landing_t& landing, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> within(usearch_geospatial_observation_t const* queries, std::size_t count,
                                           relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                           usearch_predicate_t predicate, growing_arena<forkunion_allocator_t>& staging,
                                           usearch_matches_tape_view_t& matches, std::size_t& written,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Raises the largest stored deviation to @p deviation_metres, if it is larger. */
    void widen_deviation(u32_t deviation_metres) noexcept {
        standard_atomic_ref<u32_t> largest {largest_deviation_metres};
        u32_t seen = largest.load(std::memory_order_relaxed);
        while (seen < deviation_metres &&
               !largest.compare_exchange_weak(seen, deviation_metres, std::memory_order_relaxed));
    }

  private:

    /** A factory of this arm's metric over a thread's arena, sized for a round of candidates. */
    auto metric_factory_() const noexcept {
        return [](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            metric_t made {allocator};
            status_t const ready = made.init(staging);
            return expected<metric_t> {std::move(made), ready};
        };
    }

    /** The terrain's height under @p observation when it records no altitude, else zero; the
     *  ellipsoid's own zero off the grid or beside a missing cell. */
    f64_t ground_metres_(usearch_geospatial_observation_t const& observation,
                         thread_allocator_t const& scratch) const noexcept;

    /** Keeps the verdicts of @p entries whose sight lines over the terrain clear it under
     *  @p relation, rejecting the rest; a pair off the grid counts as clear. */
    void clear_sight_(relation_t relation, value_t const& query, std::span<value_t const> entries,
                      std::span<admission_t> verdicts, thread_allocator_t const& scratch) const noexcept;

    /** Judges the pool's quad from @p first of @p context against @p query: the relation and time,
     *  then the predicate, then the terrain. */
    template <typename context_type_>
    void judge_(relation_t relation, usearch_geospatial_tolerance_t tolerance, value_t const& query,
                context_type_ const& context, std::size_t first, std::size_t quad, usearch_predicate_t predicate,
                std::array<us::utilities::uuid_t, 4>& identities, std::array<admission_t, 4>& verdicts,
                thread_allocator_t const& scratch) const noexcept;
};

template <cpu_class_t cpu_class_>
status_t geospatial_arm<cpu_class_>::adopt_terrain(us::geospatial::terrain_t const& grid) noexcept {
    if (grid.empty()) return status_t::success_k;
    if (!(grid.row_step_degrees > 0) || !(grid.column_step_degrees > 0)) return status_t::config_mismatch_k;
    terrain = grid;
    if (!terrain.heights_stride) terrain.heights_stride = terrain.columns * sizeof(f32_t);
    if (!terrain.heights) return status_t::success_k;
    nk_size_t bytes = 0;
    if (nk_status_t const sized = kit_t::terrain_pack_size(terrain.rows, terrain.columns, &bytes);
        sized != nk_success_k)
        return us::shapes::status_of(sized);
    if (status_t const reserved = terrain_pack.resize_uninitialized(bytes); failed(reserved)) return reserved;
    std::array<f64_t, 2> const steps = terrain.step_metres();
    return us::shapes::status_of(kit_t::terrain_pack(terrain.heights, terrain.rows, terrain.columns,
                                                     terrain.heights_stride, steps[0], steps[1], terrain_pack.data(),
                                                     nullptr));
}

template <cpu_class_t cpu_class_>
f64_t geospatial_arm<cpu_class_>::ground_metres_(usearch_geospatial_observation_t const& observation,
                                                 thread_allocator_t const& scratch) const noexcept {
    if (terrain.empty() || observation.altitude_metres == observation.altitude_metres) return 0;
    std::array<f64_t, 2> const cell = terrain.grid_point(observation.latitude_degrees, observation.longitude_degrees);
    f64_t const last_row = static_cast<f64_t>(terrain.rows - 1), last_column = static_cast<f64_t>(terrain.columns - 1);
    if (!(cell[0] >= 0 && cell[0] <= last_column && cell[1] >= 0 && cell[1] <= last_row)) return 0;
    f32_t height = 0;
    if (terrain.heights) {
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0]), static_cast<f32_t>(cell[1])};
        if (kit_t::terrain_heights(terrain.heights, terrain_pack.data(), point.data(), 1, &height, nullptr) !=
            nk_success_k)
            return 0;
    }
    else {
        // A filled grid answers for the two-by-two window around the point, packed on its own.
        std::size_t const first_column = min_of(static_cast<std::size_t>(cell[0]),
                                                terrain.columns > 1 ? terrain.columns - 2 : 0);
        std::size_t const first_row = min_of(static_cast<std::size_t>(cell[1]),
                                             terrain.rows > 1 ? terrain.rows - 2 : 0);
        std::size_t const window_rows = min_of<std::size_t>(2, terrain.rows),
                          window_columns = min_of<std::size_t>(2, terrain.columns);
        std::array<f32_t, 4> window {};
        terrain.callback(first_row, first_column, window_rows, window_columns, window.data(),
                         window_columns * sizeof(f32_t), terrain.state);
        nk_size_t bytes = 0;
        if (kit_t::terrain_pack_size(window_rows, window_columns, &bytes) != nk_success_k) return 0;
        growing_buffer<byte_t, thread_allocator_t> pack {scratch};
        if (failed(pack.resize_uninitialized(bytes))) return 0;
        std::array<f64_t, 2> const steps = terrain.step_metres();
        if (kit_t::terrain_pack(window.data(), window_rows, window_columns, window_columns * sizeof(f32_t), steps[0],
                                steps[1], pack.data(), nullptr) != nk_success_k)
            return 0;
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0] - static_cast<f64_t>(first_column)),
                                          static_cast<f32_t>(cell[1] - static_cast<f64_t>(first_row))};
        if (kit_t::terrain_heights(window.data(), pack.data(), point.data(), 1, &height, nullptr) != nk_success_k)
            return 0;
    }
    return height == height ? static_cast<f64_t>(height) : 0;
}

template <cpu_class_t cpu_class_>
void geospatial_arm<cpu_class_>::clear_sight_(relation_t relation, value_t const& query,
                                              std::span<value_t const> entries, std::span<admission_t> verdicts,
                                              thread_allocator_t const& scratch) const noexcept {
    // The query looks at the entries, except under `sees`, where every entry looks at the query.
    bool const entries_look = relation == relation_t::sees_k;
    std::array<f64_t, 3> const query_cell = terrain.grid_point(query.position);
    std::array<std::array<f64_t, 3>, 4> entry_cells;
    for (std::size_t lane = 0; lane != entries.size(); ++lane)
        entry_cells[lane] = terrain.grid_point(entries[lane].position);

    // A borrowed grid is read in place; a filled one is fetched over the window the on-grid
    // positions span.
    f32_t const* heights = terrain.heights;
    byte_t const* pack = terrain_pack.data();
    f64_t first_row = 0, first_column = 0;
    growing_buffer<f32_t, thread_allocator_t> window {scratch};
    growing_buffer<byte_t, thread_allocator_t> window_pack {scratch};
    if (!heights) {
        f64_t const last_row = static_cast<f64_t>(terrain.rows - 1),
                    last_column = static_cast<f64_t>(terrain.columns - 1);
        f64_t low_row = last_row, high_row = 0, low_column = last_column, high_column = 0;
        auto span_cell = [&](std::array<f64_t, 3> const& cell) noexcept {
            if (!(cell[0] >= 0 && cell[0] <= last_column && cell[1] >= 0 && cell[1] <= last_row)) return;
            low_column = std::fmin(low_column, cell[0]), high_column = std::fmax(high_column, cell[0]);
            low_row = std::fmin(low_row, cell[1]), high_row = std::fmax(high_row, cell[1]);
        };
        span_cell(query_cell);
        for (std::size_t lane = 0; lane != entries.size(); ++lane) span_cell(entry_cells[lane]);
        if (low_row > high_row || low_column > high_column) return;
        first_row = std::floor(low_row), first_column = std::floor(low_column);
        std::size_t const window_rows = min_of<std::size_t>(static_cast<std::size_t>(std::ceil(high_row)) + 1,
                                                            terrain.rows) -
                                        static_cast<std::size_t>(first_row);
        std::size_t const window_columns = min_of<std::size_t>(static_cast<std::size_t>(std::ceil(high_column)) + 1,
                                                               terrain.columns) -
                                           static_cast<std::size_t>(first_column);
        nk_size_t bytes = 0;
        if (failed(window.resize_uninitialized(window_rows * window_columns)) ||
            kit_t::terrain_pack_size(window_rows, window_columns, &bytes) != nk_success_k ||
            failed(window_pack.resize_uninitialized(bytes)))
            return;
        terrain.callback(static_cast<std::size_t>(first_row), static_cast<std::size_t>(first_column), window_rows,
                         window_columns, window.data(), window_columns * sizeof(f32_t), terrain.state);
        std::array<f64_t, 2> const steps = terrain.step_metres();
        if (kit_t::terrain_pack(window.data(), window_rows, window_columns, window_columns * sizeof(f32_t), steps[0],
                                steps[1], window_pack.data(), nullptr) != nk_success_k)
            return;
        heights = window.data(), pack = window_pack.data();
    }
    auto to_point = [&](std::array<f64_t, 3> const& cell) noexcept {
        return std::array<f32_t, 3> {static_cast<f32_t>(cell[0] - first_column),
                                     static_cast<f32_t>(cell[1] - first_row), static_cast<f32_t>(cell[2])};
    };
    std::array<f32_t, 3> const query_point = to_point(query_cell);
    std::array<f32_t, 12> entry_points;
    for (std::size_t lane = 0; lane != entries.size(); ++lane) {
        std::array<f32_t, 3> const point = to_point(entry_cells[lane]);
        entry_points[lane * 3 + 0] = point[0], entry_points[lane * 3 + 1] = point[1],
                                entry_points[lane * 3 + 2] = point[2];
    }
    // Light bends with a seventh of the Earth's curvature, which lowers the bulge under every sight
    // line.
    f64_t const refraction = 1.0 / 7;
    std::array<f32_t, 4> clearances;
    nk_status_t const tested = entries_look
                                   ? kit_t::clearances(heights, pack, entry_points.data(), nullptr, entries.size(),
                                                       query_point.data(), 1, refraction, clearances.data(),
                                                       sizeof(f32_t), nullptr)
                                   : kit_t::clearances(heights, pack, query_point.data(), nullptr, 1,
                                                       entry_points.data(), entries.size(), refraction,
                                                       clearances.data(), entries.size() * sizeof(f32_t), nullptr);
    if (tested != nk_success_k) return;
    // A negative clearance is an obstruction; NaN leaves the grid, which no terrain can block.
    for (std::size_t lane = 0; lane != entries.size(); ++lane)
        if (clearances[lane] < 0) verdicts[lane] = admission_t::reject_k;
}

template <cpu_class_t cpu_class_>
template <typename context_type_>
void geospatial_arm<cpu_class_>::judge_(relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                        value_t const& query, context_type_ const& context, std::size_t first,
                                        std::size_t quad, usearch_predicate_t predicate,
                                        std::array<us::utilities::uuid_t, 4>& identities,
                                        std::array<admission_t, 4>& verdicts,
                                        thread_allocator_t const& scratch) const noexcept {
    std::array<value_t, 4> entries;
    for (std::size_t lane = 0; lane != quad; ++lane) {
        node_id_t const node = context.results[first + lane].node_id;
        entries[lane] = index.published_value_of(node);
        identities[lane] = index.key_of(node).value;
        verdicts[lane] = us::geospatial::admits(relation, tolerance.metres, tolerance.seconds * 1000.0, query,
                                                entries[lane])
                             ? admission_t::admit_k
                             : admission_t::reject_k;
    }
    if (predicate.callback) {
        std::array<admission_t, 4> asked;
        asked.fill(admission_t::admit_k);
        key_predicate_t {predicate}(std::span {identities}.first(quad), std::span {asked}.first(quad));
        for (std::size_t lane = 0; lane != quad; ++lane)
            if (asked[lane] != admission_t::admit_k) verdicts[lane] = admission_t::reject_k;
    }
    if (relation != relation_t::near_k && !terrain.empty())
        clear_sight_(relation, query, std::span<value_t const>(entries).first(quad), std::span {verdicts}.first(quad),
                     scratch);
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_arm<cpu_class_>::insert(usearch_keys_view_t keys,
                                                                   usearch_geospatial_observation_t const* observations,
                                                                   std::size_t count, std::size_t expansion,
                                                                   progress_ticks_t& ticks,
                                                                   executor_t& executor) noexcept {
    using key_t = typename index_t::key_t;
    // Exact cone winners need a pool of at least 128, as the selector documents.
    std::size_t const breadth = expansion ? expansion : 128;
    thread_scratch<index_t, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, connectivity, breadth, index.size(), metric_factory_());
        failed(prepared))
        return {{}, prepared};

    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<key_t, thread_allocator_t> identities {caller};
    growing_buffer<value_t, thread_allocator_t> values {caller};
    if (failed(identities.reserve_exactly(count)) || failed(values.reserve_exactly(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) {
        identities.push_back_reserved(uuid_of(key_at(keys, row)));
        values.push_back_reserved(value_t {});
    }
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        thread_allocator_t const allocator {executor.arena(thread)};
        f64_t const ground = ground_metres_(observations[row], allocator);
        values[row] = us::geospatial::to_earth_observation(geodetic_of(observations[row]), ground);
    });
    u32_t deviation = 0;
    for (value_t const& value : values) deviation = max_of<u32_t>(deviation, value.deviation_metres);

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    us::geospatial::surface_keep_octants_t const selector {connectivity};
    typename index_t::insert_config_t const config {breadth, 0};
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        plan.insert(row, to_writable_span(values).subspan(row, 1), to_span(identities).subspan(row, 1),
                    scratch.metrics[thread], scratch.contexts[thread], selector, config);
        [[maybe_unused]] continuation_t const heard = ticks(row);
    });
    auto const range = plan.range;
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    widen_deviation(deviation);

    // Long links reach across the published graph, so they are drawn once the whole batch is
    // visible.
    constexpr std::size_t cones_k = us::geospatial::surface_keep_octants_t::cones_k;
    std::size_t const draws = connectivity > cones_k ? connectivity - cones_k : 0;
    status_t linked = status_t::success_k;
    drain_in_order(executor, range.size(), [&](std::size_t position, std::size_t thread) noexcept {
        splitmix64_t generator {mix_bits(static_cast<u64_t>(range.first) + position)};
        keep_first_failure(
            linked, us::geospatial::draw_long_links(index, static_cast<node_id_t>(range.first + position), draws,
                                                    scratch.metrics[thread], scratch.contexts[thread], generator));
    });
    if (failed(linked)) return {{}, linked};
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_arm<cpu_class_>::search(
    usearch_geospatial_observation_t const* queries, std::size_t count, relation_t relation,
    usearch_geospatial_tolerance_t tolerance, std::size_t wanted, std::size_t expansion, usearch_predicate_t predicate,
    matches_landing_t& landing, progress_ticks_t& ticks, executor_t& executor) noexcept {
    // Pools narrower than the row they fill would short every row.
    std::size_t const breadth = max_of(expansion ? expansion : std::size_t(connectivity_32_k) * 2, wanted);
    thread_scratch<index_t, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, connectivity, breadth, index.size(), metric_factory_());
        failed(prepared))
        return {{}, prepared};
    typename index_t::search_config_t const config {breadth, 0};
    status_t searched = status_t::success_k;
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        if (failed(standard_atomic_ref<status_t>(searched).load(std::memory_order_relaxed))) return;
        thread_allocator_t const allocator {executor.arena(thread)};
        value_t const query = us::geospatial::to_earth_observation(geodetic_of(queries[row]),
                                                                   ground_metres_(queries[row], allocator));
        auto& context = scratch.contexts[thread];
        auto const [found, status] = index.search(query, scratch.metrics[thread], context, config);
        if (failed(status)) return keep_first_failure(searched, status);
        usearch_key_t* const keys = landing.keys + row * wanted;
        usearch_distance_t* const scores = landing.scores + row * wanted;
        std::size_t const pooled = context.results.size();
        std::size_t landed = 0;
        for (std::size_t first = 0; first < pooled && landed != wanted; first += 4) {
            std::size_t const quad = min_of<std::size_t>(4, pooled - first);
            std::array<us::utilities::uuid_t, 4> identities;
            std::array<admission_t, 4> verdicts;
            judge_(relation, tolerance, query, context, first, quad, predicate, identities, verdicts, allocator);
            for (std::size_t lane = 0; lane != quad && landed != wanted; ++lane) {
                if (verdicts[lane] != admission_t::admit_k) continue;
                keys[landed] = std::bit_cast<usearch_key_t>(identities[lane]);
                scores[landed] = std::sqrt(static_cast<usearch_distance_t>(context.results[first + lane].distance)) /
                                 us::geospatial::centimetres_per_metre_k;
                ++landed;
            }
        }
        landing.counts[row] = landed;
        if (ticks(row) == continuation_t::cancel_k) keep_first_failure(searched, status_t::cancelled_k);
    });
    if (failed(searched)) return {{}, searched};
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_arm<cpu_class_>::within(
    usearch_geospatial_observation_t const* queries, std::size_t count, relation_t relation,
    usearch_geospatial_tolerance_t tolerance, usearch_predicate_t predicate,
    growing_arena<forkunion_allocator_t>& staging, usearch_matches_tape_view_t& matches, std::size_t& written,
    progress_ticks_t& ticks, executor_t& executor) noexcept {
    using found_t = growing_buffer<geospatial_within_match_t, thread_allocator_t>;
    thread_scratch<index_t, metric_t> scratch;
    std::size_t const breadth = std::size_t(connectivity_32_k) * 2;
    if (status_t const prepared = scratch.prepare(executor, connectivity, breadth, index.size(), metric_factory_());
        failed(prepared))
        return {{}, prepared};
    growing_buffer<found_t, forkunion_allocator_t> found {forkunion_allocator_t {}};
    if (failed(found.reserve_exactly(executor.size()))) return {{}, status_t::bad_alloc_k};
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        found.push_back_reserved(found_t {thread_allocator_t {executor.arena(thread)}});
    f64_t const largest_deviation =
        standard_atomic_ref<u32_t>(largest_deviation_metres).load(std::memory_order_relaxed);

    status_t walked = status_t::success_k;
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        if (failed(standard_atomic_ref<status_t>(walked).load(std::memory_order_relaxed))) return;
        thread_allocator_t const allocator {executor.arena(thread)};
        value_t const query = us::geospatial::to_earth_observation(geodetic_of(queries[row]),
                                                                   ground_metres_(queries[row], allocator));
        // The walk reaches as far as the widest pair could match, and `admits` tightens back to
        // each pair's own.
        f64_t const deviations = std::sqrt(static_cast<f64_t>(query.deviation_metres) * query.deviation_metres +
                                           largest_deviation * largest_deviation);
        i64_t const reach = static_cast<i64_t>(
            std::ceil((tolerance.metres + 3 * deviations) * us::geospatial::centimetres_per_metre_k));
        auto& context = scratch.contexts[thread];
        us::geospatial::radius_result_t const ball = us::geospatial::within(
            index, query.position, reach, us::geospatial::radius_margin_t::spanner_k, scratch.metrics[thread], context);
        if (!ball) return keep_first_failure(walked, ball.status);
        for (std::size_t first = 0; first < ball.count; first += 4) {
            std::size_t const quad = min_of<std::size_t>(4, ball.count - first);
            std::array<us::utilities::uuid_t, 4> identities;
            std::array<admission_t, 4> verdicts;
            judge_(relation, tolerance, query, context, first, quad, predicate, identities, verdicts, allocator);
            for (std::size_t lane = 0; lane != quad; ++lane) {
                if (verdicts[lane] != admission_t::admit_k) continue;
                f64_t const metres = std::sqrt(static_cast<f64_t>(context.results[first + lane].distance)) /
                                     us::geospatial::centimetres_per_metre_k;
                if (failed(found[thread].push_back({row, std::bit_cast<usearch_key_t>(identities[lane]), metres})))
                    return keep_first_failure(walked, status_t::bad_alloc_k);
            }
        }
        if (ticks(row) == continuation_t::cancel_k) keep_first_failure(walked, status_t::cancelled_k);
    });
    if (failed(walked)) return {{}, walked};

    // Each query landed whole on one thread in ascending order, so counting per query places every
    // match.
    std::size_t* const offsets = carve<std::size_t>(staging, count + 1);
    if (!offsets) return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count + 1; ++row) offsets[row] = 0;
    std::size_t total = 0;
    for (found_t const& thread_found : found)
        for (geospatial_within_match_t const& match : thread_found) ++offsets[match.query + 1], ++total;
    for (std::size_t row = 0; row != count; ++row) offsets[row + 1] += offsets[row];
    usearch_key_t* const keys = carve<usearch_key_t>(staging, total);
    usearch_distance_t* const scores = carve<usearch_distance_t>(staging, total);
    growing_buffer<std::size_t, thread_allocator_t> cursors {thread_allocator_t {executor.arena(0)}};
    if ((total && (!keys || !scores)) || failed(cursors.resize_uninitialized(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) cursors[row] = offsets[row];
    for (found_t const& thread_found : found)
        for (geospatial_within_match_t const& match : thread_found) {
            std::size_t const slot = cursors[match.query]++;
            keys[slot] = match.key, scores[slot] = match.score;
        }
    matches = {keys, scores, offsets};
    written = total;
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
status_t geospatial_arm<cpu_class_>::save(std::string_view path, executor_t& executor,
                                          progress_ticks_t& ticks) noexcept {
    std::array<char, 24> connectivity_text;
    std::array<safegraphs_custom_field_t, 1> const fields {
        {{"connectivity", decimal_of(connectivity, connectivity_text)}}};
    save_config_t config;
    config.provenance = fields;
    return safegraphs_write_index(path, index, us::geospatial::earth_observation_codec_t {}, executor, config, "chord",
                                  ticks);
}

template <cpu_class_t cpu_class_>
status_t geospatial_arm<cpu_class_>::load(std::string_view path, executor_t& executor,
                                          progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = safegraphs_read_index(path, index, us::geospatial::earth_observation_codec_t {},
                                                      executor, load_config_t {}, ticks);
        failed(loaded))
        return loaded;
    std::size_t const nodes = index.size();
    for (std::size_t node = 0; node != nodes; ++node)
        widen_deviation(index.published_value_of(static_cast<node_id_t>(node)).deviation_metres);
    return status_t::success_k;
}

#pragma endregion Geospatial Arms
