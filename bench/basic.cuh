/**
 *  @file bench/basic.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief What the sequence and shapes search benchmarks share: the flags, the ragged dataset, and
 *      the phases that build one index through the C interface, search it with held-out entries,
 *      and judge the hits against an exact scan.
 *
 *  The index runs as users run it, through the C interface and whichever CPU class it picks at
 *  runtime, or on a GPU; the oracle scores with the widest kit this binary compiled. Queries are
 *  the file's last entries, kept out of the index, so no query finds itself. A hit counts when its
 *  distance is within the exact k-th, so integer edit distances, where many entries tie, are not
 *  undercounted. An oracle rejects an entry with an infinite distance, and a query admitting fewer
 *  than k entries is judged over those it admits, one admitting none not at all.
 *
 *  @verbatim
 *  --metric            per binary                Distance kind, by its checkpoint token
 *  --connectivity      32                        Graph degree
 *  --expansion-add     256                       Candidates while inserting
 *  --expansion-search  96                        Candidates while searching, like 32,64,128
 *  --top-k             10                        Matches per query to judge
 *  --max-base-entries  the file but the queries  Cap on ingested entries
 *  --queries           100,000                   Entries held out of the file's tail as queries
 *  --oracle-queries    256                       Queries the exact oracle judges
 *  --threads           7/8 of the logical cores  Worker threads
 *  --inserts-per-core  2048                      Entries per insert call for each thread
 *  --queries-per-core  2048                      Queries per search call for each thread
 *  @endverbatim
 */
#pragma once

#include <cmath> // `std::fabs`

#include <algorithm> // `std::partial_sort`, `std::min`, `std::max`
#include <limits>    // `std::numeric_limits` - the distance of a rejected entry
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include <forkunion.h> // `fu_logical_cores_count`

#include "usearch/types.h"
#include "usearch/utilities/types.hpp" // `USEARCH_OS_WINDOWS_`

#if !USEARCH_OS_WINDOWS_
#include <unistd.h> // `isatty`
#endif

#include "arguments.hpp" // `flag_t`, `count_t`, `counts_t`, `parse_or_exit`
#include "bench.hpp"

namespace unum::usearch::bench {

/** Every flag both binaries read, its default as the initializer. */
struct basic_settings_t {

    /** The distance kind as its checkpoint token spells it. */
    std::string_view metric_name;

    /** Neighbors-list width per node. */
    count_t connectivity {.value = 32};

    /** How many candidates an insertion keeps. */
    count_t expansion_add {.value = 256};

    /** Search breadths, each judged over the one graph the run builds. */
    counts_t expansion_search {.values = {96}, .size = 1};

    /** Matches per query to return and judge. */
    count_t top_k {.value = 10};

    /** Cap on ingested entries; absent ingests every entry the queries leave. */
    count_t max_base_entries;

    /** Entries held out of the file's tail as queries. */
    count_t queries {.value = 100000};

    /** Queries the exact oracle judges. */
    count_t oracle_queries {.value = 256};

    /** Worker threads; absent takes 7/8 of the logical cores. */
    count_t threads;

    /** Entries each insert call takes per thread. */
    count_t inserts_per_core {.value = 2048};

    /** Queries each search call takes per thread. */
    count_t queries_per_core {.value = 2048};

    /** Whether stdout is a terminal, probed once, so progress redraws one line instead of printing
     *  deciles. */
    bool terminal = false;
};

/** The flags both binaries take, after the ones naming each binary's dataset. */
inline std::array<flag_t, 11> basic_flags(basic_settings_t& settings) noexcept {
    return {{
        {"--metric", "name", "distance kind, by its checkpoint token", &settings.metric_name},
        {"--connectivity", "degree", "graph degree; 32 by default", &settings.connectivity},
        {"--expansion-add", "breadth", "insertion candidates; 256 by default", &settings.expansion_add},
        {"--expansion-search", "breadths", "search breadths, like 32,64,128; 96 by default",
         &settings.expansion_search},
        {"--top-k", "count", "matches per query to judge; 10 by default", &settings.top_k},
        {"--max-base-entries", "count", "cap on ingested entries; every entry the queries leave by default",
         &settings.max_base_entries},
        {"--queries", "count", "entries held out of the file's tail as queries; 100,000 by default", &settings.queries},
        {"--oracle-queries", "count", "queries the exact oracle judges; 256 by default", &settings.oracle_queries},
        {"--threads", "count", "threads; 7/8 of the logical cores by default", &settings.threads},
        {"--inserts-per-core", "entries", "entries per insert call, per thread; 2048 by default",
         &settings.inserts_per_core},
        {"--queries-per-core", "queries", "queries per search call, per thread; 2048 by default",
         &settings.queries_per_core},
    }};
}

/** Parses @p binary_flags and the shared flags into @p settings, resolving the thread count and the
 *  terminal. On `--help` prints the usage and exits with status 0; on a bad flag prints one line
 *  and exits with status 1. */
template <std::size_t binary_flags_count_>
void read_basic_settings(int argc, char const* const* argv, std::string_view synopsis, basic_settings_t& settings,
                         std::array<flag_t, binary_flags_count_> const& binary_flags) {
    std::vector<flag_t> flags(binary_flags.begin(), binary_flags.end());
    auto const shared = basic_flags(settings);
    flags.insert(flags.end(), shared.begin(), shared.end());
    parse_or_exit(argc, argv, synopsis, flags);
    if (!settings.threads.value) {
        fu_topology_t topology = nullptr;
        std::size_t cores = 0;
        if (fu_topology_new(&topology) == fu_success_k) {
            if (fu_logical_cores_count(topology, &cores) != fu_success_k) cores = 0;
            fu_topology_delete(topology);
        }
        settings.threads.value = std::max<std::size_t>(1, cores * 7 / 8);
    }
#if !USEARCH_OS_WINDOWS_
    settings.terminal = isatty(fileno(stdout)) != 0;
#endif
}

/** A ragged dataset as the C interface takes one: elements end to end, and offsets in elements
 *  around them. */
struct tape_t {
    void const* elements = nullptr;
    std::vector<std::size_t> offsets {0};
    usearch_scalar_t scalar = usearch_scalar_unknown_k;

    std::size_t size() const noexcept { return offsets.size() - 1; }

    /** Entries from @p first on, as a slice whose offsets still count from the first element. */
    usearch_tape_view_t view(std::size_t first) const noexcept { return {elements, offsets.data() + first, scalar}; }
};

/** Wraps a row number into the 128-bit key the C interface takes, low bytes last. */
inline usearch_key_t key_of(std::size_t row) noexcept {
    usearch_key_t key {};
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        key.bytes[15 - byte_index] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(row) >> (byte_index * 8));
    return key;
}

/** Whether @p found ranks no further than @p exact, within the rounding two kits may differ by. */
inline bool within(double found, double exact) noexcept {
    return found <= exact + 1e-6 * std::max(1.0, std::fabs(exact));
}

/** Searches the index where it was built, on the executor that built it. */
struct serve_in_place_t {
    usearch_executor_t operator()(usearch_executor_t built_on, usearch_outcome_t*) const noexcept { return built_on; }
};

/**
 *  @brief Builds one index over the entries before @p query_first, searches it with the `--queries`
 *      from there on at every `--expansion-search` breadth, and judges the first `--oracle-queries`
 *      against one exact scan each, spread over the threads.
 *
 *  @param[in] vram The devices of a VRAM index, whose executor drives them; @c NULL for a RAM
 *      index, driven by `--threads` threads.
 *  @param[in] insert Lands entries through the C interface:
 *      `(executor, keys, entries, count, stats, outcome)`.
 *  @param[in] search Answers queries into an arena at a breadth:
 *      `(executor, arena, queries, count, expansion, matches, stats, outcome)`.
 *  @param[in] exact Fills one distance per base entry for the query at a tape row, infinite for an
 *      entry the query rejects, from any of `--threads` threads at once, each with its own
 *      scratch: `(thread, row, distances)`.
 *  @param[in] serve Moves the built index where the searches read it, answering the executor they
 *      run on, as `(executor, outcome)`; an NVMe residency saves it and views the checkpoint.
 *  @return The process exit code.
 */
template <typename stats_type_, typename insert_type_, typename search_type_, typename exact_type_,
          typename serve_type_ = serve_in_place_t>
int run_basic_bench(basic_settings_t const& settings, tape_t const& tape, usearch_vram_options_t const* vram,
                    insert_type_&& insert, search_type_&& search, exact_type_&& exact, serve_type_&& serve = {}) {
    if (tape.size() <= settings.queries.value)
        return fmt::println(stderr, "The file holds {} entries, too few to hold out {} queries", tape.size(),
                            settings.queries.value),
               1;
    std::size_t const query_first = tape.size() - settings.queries.value, query_count = settings.queries.value;
    std::size_t const base_count = settings.max_base_entries.value
                                       ? std::min(settings.max_base_entries.value, query_first)
                                       : query_first;
    std::size_t const threads = settings.threads.value, top_k = settings.top_k.value;
    fmt::println("- Threads: {}", threads);
    fmt::println("- Base entries: {} · queries: {} · top-k: {}", humanized(base_count), humanized(query_count), top_k);
    fmt::println("- Connectivity: {} · expansion: {} adding", settings.connectivity.value,
                 settings.expansion_add.value);

    usearch_outcome_t outcome {};
    usearch_executor_t executor = nullptr;
    if (vram) {
        usearch_executor_for_vram_t executor_arguments {.executor = &executor, .outcome = &outcome, .vram = *vram};
        usearch_executor_for_vram(&executor_arguments);
    }
    else {
        usearch_executor_for_ram_t executor_arguments {.executor = &executor, .outcome = &outcome, .threads = threads};
        usearch_executor_for_ram(&executor_arguments);
    }
    if (outcome.message) return fmt::println(stderr, "Executor: {}", outcome.message), 1;
    // A VRAM index lands its matches in a host arena, where the judgment reads them.
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_arguments {.arena = &arena, .executor = vram ? nullptr : executor, .outcome = &outcome};
    usearch_arena_init(&arena_arguments);
    if (outcome.message) return fmt::println(stderr, "Arena: {}", outcome.message), 1;

    std::vector<usearch_key_t> keys(base_count);
    for (std::size_t row = 0; row != base_count; ++row) keys[row] = key_of(row);
    std::size_t const insert_batch = std::min(settings.inserts_per_core.value * threads, base_count);
    stats_type_ built {};
    progress_bar_t building {"build", base_count, settings.terminal};
    time_point_t const build_start = steady_clock_t::now();
    for (std::size_t first = 0; first < base_count; first += insert_batch) {
        std::size_t const count = std::min(insert_batch, base_count - first);
        stats_type_ stats {};
        insert(executor, usearch_keys_view_t {keys.data() + first, sizeof(usearch_key_t)}, tape.view(first), count,
               &stats, &outcome);
        if (outcome.message) return fmt::println(stderr, "Insert: {}", outcome.message), 1;
        built.computed_distances += stats.computed_distances;
        building.tick(first + count);
    }
    building.finish();
    double const build_seconds = stdc::duration<double>(steady_clock_t::now() - build_start).count();
    fmt::println("- Build: {} entries in {:.2f} s · {}/s · {:.0f} distances per entry", humanized(base_count),
                 build_seconds, humanized(base_count / build_seconds),
                 static_cast<double>(built.computed_distances) / static_cast<double>(base_count));

    usearch_executor_t const serving = serve(executor, &outcome);
    if (outcome.message) return fmt::println(stderr, "Serve: {}", outcome.message), 1;

    // One exact scan per judged query, spread over the threads, each scanning into its own row;
    // every breadth is judged against the same two distances.
    std::size_t const judged = std::min(settings.oracle_queries.value, query_count);
    std::size_t const kept = std::min(top_k, base_count);
    std::vector<double> nearest_exact(judged), kept_exact(judged), scanned(threads * base_count);
    std::vector<std::size_t> tied_exact(judged), admitted_exact(judged);
    auto judge_exactly = [&](std::size_t query, std::size_t thread) noexcept {
        std::span<double> const distances {scanned.data() + thread * base_count, base_count};
        exact(thread, query_first + query, distances);
        // One past the kept, so a tie across the cut counts as one too.
        std::size_t const ranked = std::min(kept + 1, base_count);
        std::partial_sort(distances.begin(), distances.begin() + ranked, distances.end());
        std::size_t admitted = 0;
        while (admitted != kept && distances[admitted] != std::numeric_limits<double>::infinity()) ++admitted;
        admitted_exact[query] = admitted;
        nearest_exact[query] = distances[0], kept_exact[query] = distances[admitted ? admitted - 1 : 0];
        for (std::size_t rank = 0; rank != admitted; ++rank)
            tied_exact[query] += (rank && distances[rank] == distances[rank - 1]) ||
                                 (rank + 1 < ranked && distances[rank] == distances[rank + 1]);
    };
    using judge_t = decltype(judge_exactly);
    fu_topology_t topology = nullptr;
    fu_pool_t pool = nullptr;
    time_point_t const judge_start = steady_clock_t::now();
    bool const judging = fu_topology_new(&topology) == fu_success_k &&
                         fu_pool_new("usearch_bench_oracle", fu_capabilities_all_k, &pool) == fu_success_k &&
                         fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k) == fu_success_k &&
                         fu_pool_for_n_dynamic(
                             pool, judged,
                             [](fu_lambda_context_t context, std::size_t query, std::size_t thread, std::size_t) {
                                 (*static_cast<judge_t*>(context))(query, thread);
                             },
                             &judge_exactly) == fu_success_k;
    if (pool) fu_pool_delete(pool);
    if (topology) fu_topology_delete(topology);
    if (!judging) return fmt::println(stderr, "The exact scan could not spread over {} threads", threads), 1;
    std::size_t tied = 0, admitted = 0, answerable = 0;
    for (std::size_t const query_tied : tied_exact) tied += query_tied;
    for (std::size_t const query_admitted : admitted_exact)
        admitted += query_admitted, answerable += query_admitted != 0;
    fmt::println("- Exact scan: {} queries in {:.2f} s · {:.4f} of the top {} distances tied", judged,
                 stdc::duration<double>(steady_clock_t::now() - judge_start).count(),
                 static_cast<double>(tied) / static_cast<double>(admitted), kept);

    std::vector<double> found_scores(query_count * top_k);
    std::vector<std::size_t> found_counts(query_count);
    std::size_t const query_batch = std::min(settings.queries_per_core.value * threads, query_count);
    for (std::size_t const expansion : settings.expansion_search.view()) {
        stats_type_ searched {};
        progress_bar_t searching {"search", query_count, settings.terminal};
        time_point_t const search_start = steady_clock_t::now();
        for (std::size_t first = 0; first < query_count; first += query_batch) {
            std::size_t const count = std::min(query_batch, query_count - first);
            usearch_matches_view_t matches {};
            stats_type_ stats {};
            search(serving, arena, tape.view(query_first + first), count, expansion, &matches, &stats, &outcome);
            if (outcome.message) return fmt::println(stderr, "Search: {}", outcome.message), 1;
            searched.computed_distances += stats.computed_distances;
            for (std::size_t query = 0; query != count; ++query) {
                std::size_t const landed = std::min(matches.counts[query], top_k);
                found_counts[first + query] = landed;
                std::copy_n(matches.scores + query * matches.columns, landed,
                            found_scores.begin() + (first + query) * top_k);
            }
            searching.tick(first + count);
        }
        searching.finish();
        double const search_seconds = stdc::duration<double>(steady_clock_t::now() - search_start).count();
        double recall_at_one = 0, recall_at_k = 0;
        for (std::size_t query = 0; query != judged; ++query) {
            if (!admitted_exact[query]) continue;
            double const* const scores = found_scores.data() + query * top_k;
            std::size_t hits = 0;
            for (std::size_t rank = 0; rank != found_counts[query]; ++rank)
                hits += within(scores[rank], kept_exact[query]);
            recall_at_one += found_counts[query] && within(scores[0], nearest_exact[query]);
            recall_at_k += static_cast<double>(std::min(hits, admitted_exact[query])) /
                           static_cast<double>(admitted_exact[query]);
        }
        fmt::println("- Expansion {}: recall@1 {:.4f} · recall@{} {:.4f} over {} judged · " //
                     "{} queries in {:.2f} s · {}/s · {:.0f} distances per query",          //
                     expansion, recall_at_one / answerable, kept, recall_at_k / answerable, answerable,
                     humanized(query_count), search_seconds, humanized(query_count / search_seconds),
                     static_cast<double>(searched.computed_distances) / static_cast<double>(query_count));
    }

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    return 0;
}

} // namespace unum::usearch::bench
