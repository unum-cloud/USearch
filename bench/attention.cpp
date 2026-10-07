/**
 *  @file bench/attention.cpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief How much of a lightning indexer's selection a graph walk recovers: loads a dot-kernel graph that
 *      `usearch_bench_dense_ram_index --save` built over the indexer's keys, walks it for every query under the
 *      indexer's own `Σₕ wₕ·ReLU(qₕ·k)` score, and judges the top rows against the exact scan's.
 *
 *  Every input is a RetriEval-layout matrix: `--queries` holds each query's heads as consecutive `.bf16bin` rows,
 *  `--weights` one `.fbin` row of signed head weights per query with the model's scale folded in, `--positions` one
 *  `.ibin` row per query holding the first key row the query cannot see and the end of its window, and `--truth`
 *  the exact scan's top rows per query, best first. Each `--expansions` entry is one walk breadth per sweep line.
 */
#include <cassert> // `assert` - the CPU casts never refuse
#include <cstdlib> // `std::exit`
#include <cstring> // `std::memcpy` - the exact rows

#include <algorithm>   // `std::sort`, `std::partial_sort`, `std::binary_search`
#include <array>       // `std::array`
#include <atomic>      // `std::atomic`
#include <chrono>      // `std::chrono::steady_clock`
#include <memory>      // `std::unique_ptr`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join`

#include "arguments.hpp" // `flag_t`, `count_t`, `counts_t`, `switch_t`, `parse_or_exit`
#include "dense/ram_index.hpp"

namespace unum::usearch::bench {

using namespace unum::usearch::bench::dense;
using namespace unum::usearch::bench::utilities;

namespace attention {

/** Every flag this binary reads, its default as the initializer, filled once by @c read_settings. */
struct settings_t {

    /** The saved graph: its name, directory, degree and metric, plus the worker threads. */
    bench::settings_t graph {.kernel = nk_kernel_dot_k};

    /** The RetriEval matrices of the indexer, argv-backed, so null-terminated. */
    std::string_view queries_path;
    std::string_view weights_path;
    std::string_view positions_path;
    std::string_view truth_path;

    /** Optional `.ibin` rows scanned exactly per query; absent scans none. */
    std::string_view hubs_path;

    /** Walk breadths, one sweep line each. */
    counts_t breadths {.values = {512, 1024, 2048, 4096, 8192}, .size = 5};

    /** Queries to judge; absent judges every one. */
    count_t max_queries;

    /** First query to judge. */
    count_t first_query {.minimum = 0};

    /** Best hubs each walk starts from; absent starts from the graph's entry. */
    count_t hub_seeds;

    /** Whether walks navigate by `Σₕ wₕ·qₕ` alone and rescore the pool under the indexer. */
    switch_t proxy = switch_t::absent_k;

    /** Whether the graph was built under the angular kernel rather than the dot. */
    switch_t angular = switch_t::absent_k;
};

/** Parses the flags into one @ref settings_t. On `--help` prints the usage and exits with status 0; on a bad or
 *  missing flag prints one line and exits with status 1. */
inline settings_t read_settings(int argc, char const* const* argv) {
    settings_t settings;
    std::array<flag_t, 15> const flags {{
        {"--load", "name", "graph saved by the dense RAM bench", &settings.graph.load_name},
        {"--queries", "path", "`.bf16bin`, the heads of every query", &settings.queries_path},
        {"--weights", "path", "`.fbin`, one row of head weights", &settings.weights_path},
        {"--positions", "path", "`.ibin`, first unseen row & end", &settings.positions_path},
        {"--truth", "path", "`.ibin`, the exact top rows", &settings.truth_path},
        {"--checkpoints-dir", "dir", "where the graph was saved; /tmp by default", &settings.graph.checkpoints_dir},
        {"--connectivity", "degree", "its neighbor-list degree; 32 by default", &settings.graph.connectivity},
        {"--breadths", "counts", "walk breadths, one sweep line each; 512,1024,2048,4096,8192 by default",
         &settings.breadths},
        {"--max-queries", "count", "judge only the first queries; all by default", &settings.max_queries},
        {"--first-query", "index", "judge only from this query on; 0 by default", &settings.first_query},
        {"--hubs", "path", "`.ibin` rows scanned exactly per query", &settings.hubs_path},
        {"--hub-seeds", "count", "start walks from the best hubs", &settings.hub_seeds},
        {"--threads", "count", "threads; 7/8 of the logical cores by default", &settings.graph.threads},
        {"--proxy", "", "navigate by Σ w·q alone and rescore the pool", &settings.proxy},
        {"--angular", "", "the graph was built under cosine", &settings.angular},
    }};
    parse_or_exit(
        argc, argv,
        "usearch_bench_attention --load <name> --queries <path> --weights <path> --positions <path> --truth <path>",
        flags);
    for (flag_t const& flag : std::span {flags}.first(5))
        if (std::get<std::string_view*>(flag.destination)->empty()) {
            fmt::println(stderr, "{}=\"\" does not parse, expected a {}", flag.name, flag.placeholder);
            std::exit(1);
        }
    if (settings.angular == switch_t::present_k) settings.graph.kernel = nk_kernel_angular_k;
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar its flag parses from. */
inline void print(settings_t const& settings) {
    auto const or_else = [](std::string_view given, std::string_view fallback) noexcept {
        return given.empty() ? fallback : given;
    };
    auto const count_or_else = [](count_t const& count, std::string_view fallback) {
        return count.value ? fmt::format("{}", count.value) : std::string(fallback);
    };
    fmt::println("- Threads: {}", count_or_else(settings.graph.threads, "7/8 of the logical cores"));
    fmt::println("- Load: {}", settings.graph.load_name);
    fmt::println("- Checkpoints dir: {}", settings.graph.checkpoints_dir);
    fmt::println("- Connectivity: {}", settings.graph.connectivity.value);
    fmt::println("- Metric: {}", nk_kernel_name(settings.graph.kernel));
    fmt::println("- Queries: {}", settings.queries_path);
    fmt::println("- Weights: {}", settings.weights_path);
    fmt::println("- Positions: {}", settings.positions_path);
    fmt::println("- Truth: {}", settings.truth_path);
    fmt::println("- Hubs: {}", or_else(settings.hubs_path, "none"));
    fmt::println("- Hub seeds: {}", count_or_else(settings.hub_seeds, "none"));
    fmt::println("- Breadths: {}", fmt::join(settings.breadths.view(), ","));
    fmt::println("- Max queries: {}", count_or_else(settings.max_queries, "all"));
    fmt::println("- First query: {}", settings.first_query.value);
    fmt::println("- Proxy: {}", settings.proxy == switch_t::present_k);
}

} // namespace attention

/** One mapped RetriEval matrix and the typed view over its payload. */
template <typename scalar_type_>
struct mapped_matrix {
    mapped_file_t mapping;
    nk::matrix_view<scalar_type_> view;
};

template <typename scalar_type_>
bool map_matrix(std::string_view path, mapped_matrix<scalar_type_>& matrix) {
    auto [mapping, status] = mapped_file_t::read(path);
    if (failed(status)) {
        fmt::println("Failed to map {}", path);
        return false;
    }
    auto const [extents, parse_status] = parse_matrix_header<scalar_type_, u32_t>(mapping.view());
    if (failed(parse_status)) {
        fmt::println("Failed to parse the header of {}", path);
        return false;
    }
    matrix.mapping = std::move(mapping);
    matrix.view = view_matrix_payload<scalar_type_, u32_t>(matrix.mapping.view(), extents.rows, extents.columns);
    return true;
}

/** The per-query inputs of one sweep, shared by every walking thread. */
struct indexer_inputs_t {
    mapped_matrix<nk::bf16_t> queries;
    mapped_matrix<f32_t> weights;
    mapped_matrix<u32_t> positions;
    mapped_matrix<u32_t> truth;
    mapped_matrix<u32_t> hubs;
    std::size_t heads = 0;
    std::size_t dimensions = 0;
    std::size_t first_query = 0;
    std::size_t query_count = 0;
    std::size_t top = 0;
};

/** Candidate rows with their exact indexer scores, best first. */
using scored_rows_t = std::vector<std::pair<f64_t, u32_t>>;

/** The exact indexer score of one key row, in @c f64. */
inline f64_t indexer_score(indexer_inputs_t const& inputs, std::size_t query, nk::bf16_t const* key) noexcept {
    using kit_t = dense_elementwise_chain<nk_cap_any_k>::kit_t;
    std::array<f32_t, 256> widened_key, widened_head;
    [[maybe_unused]] nk_status_t status = kit_t::cast(key, nk_bf16_k, widened_key.data(), nk_f32_k, inputs.dimensions,
                                                      nullptr);
    assert(status == nk_success_k && "The serial tier casts every dtype pair");
    f64_t score = 0;
    for (std::size_t head = 0; head != inputs.heads; ++head) {
        status = kit_t::cast(inputs.queries.view.row(query * inputs.heads + head).data(), nk_bf16_k,
                             widened_head.data(), nk_f32_k, inputs.dimensions, nullptr);
        assert(status == nk_success_k && "The serial tier casts every dtype pair");
        f64_t dot = 0;
        for (std::size_t dimension = 0; dimension != inputs.dimensions; ++dimension)
            dot += static_cast<f64_t>(widened_head[dimension]) * widened_key[dimension];
        score += inputs.weights.view(query, head) * max_of(dot, f64_t {});
    }
    return score;
}

/** Everything one sweep line shares between its tasks and the report. */
template <typename index_type_>
struct indexer_sweep {
    using traversal_t = typename index_type_::dense_traversal_context_t;

    index_type_& index;
    indexer_inputs_t const& inputs;
    std::vector<u32_t> const& id_to_row;
    std::vector<std::unique_ptr<traversal_t>>& traversals;

    /** Every query's best hub rows, scanned exactly ahead of the sweep; empty without `--hubs`. */
    std::vector<scored_rows_t> const& hub_tops;

    /** Maps a key row back to its node, to seed walks from the best hubs. */
    std::vector<u32_t> const& row_to_id;

    std::vector<telemetry_lane_t> lanes;

    /** Per-thread staging: the navigation vector, then the selected rows and their scores. */
    std::vector<std::vector<nk::bf16_t>> navigation;
    std::vector<std::vector<u32_t>> selected;
    std::vector<scored_rows_t> rescored;
    std::vector<std::vector<typename index_type_::node_id_t>> seeds;

    /** Recall of every query against the exact top rows, and of the exact top 64 alone. */
    std::vector<f64_t> recalls;
    std::vector<f64_t> head_recalls;

    std::size_t expansion = 0;
    std::size_t hub_seeds = 0;
    bool proxy = false;
    std::atomic<std::size_t> failures = 0;
};

template <typename index_type_>
void walk_callback(fu_lambda_context_t punned, std::size_t task, std::size_t thread, std::size_t) noexcept {
    using scalar_t = typename index_type_::scalar_t;
    using small_size_t = typename index_type_::small_size_t;
    auto& sweep = *static_cast<indexer_sweep<index_type_>*>(punned);
    indexer_inputs_t const& inputs = sweep.inputs;
    std::size_t const query = inputs.first_query + task;

    // The walk navigates by `Σₕ wₕ·qₕ`, the indexer score where every head's ReLU is open.
    using kit_t = dense_elementwise_chain<nk_cap_any_k>::kit_t;
    std::array<nk::vector_view<scalar_t>, 64> heads;
    std::array<f32_t, 256> summed {}, widened;
    f32_t const* const weights = inputs.weights.view.row(query).data();
    [[maybe_unused]] nk_status_t cast = nk_success_k;
    for (std::size_t head = 0; head != inputs.heads; ++head) {
        heads[head] = inputs.queries.view.row(query * inputs.heads + head).as_vector();
        cast = kit_t::cast(heads[head].data(), nk_bf16_k, widened.data(), nk_f32_k, inputs.dimensions, nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
        for (std::size_t dimension = 0; dimension != inputs.dimensions; ++dimension)
            summed[dimension] += weights[head] * widened[dimension];
    }
    std::vector<nk::bf16_t>& navigation = sweep.navigation[thread];
    cast = kit_t::cast(summed.data(), nk_f32_k, navigation.data(), nk_bf16_k, inputs.dimensions, nullptr);
    assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    nk::vector_view<scalar_t> const navigation_vector {navigation.data(), inputs.dimensions};

    typename index_type_::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(sweep.expansion);
    config.synchronization = synchronization_t::immutable_k;
    if (sweep.hub_seeds && !sweep.hub_tops.empty()) {
        auto& seeds = sweep.seeds[thread];
        seeds.clear();
        for (auto const& [score, row] : sweep.hub_tops[task]) {
            if (seeds.size() == sweep.hub_seeds) break;
            seeds.push_back(sweep.row_to_id[row]);
        }
        config.seeds = seeds;
    }
    auto& traversal = *sweep.traversals[thread];
    auto const [result, status] = sweep.proxy
                                      ? sweep.index.search(navigation_vector, traversal, config)
                                      : sweep.index.search(dense_indexer_query<scalar_t> {navigation_vector,
                                                                                          {heads.data(), inputs.heads},
                                                                                          {weights, inputs.heads}},
                                                           traversal, config);
    if (failed(status)) {
        sweep.failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    telemetry_lane_t& lane = sweep.lanes[thread];
    lane.visited_members += result.visited_members;
    lane.computed_distances += result.computed_distances;

    // Rows past the query's own position are not keys it may attend to, whatever the graph returns.
    u32_t const first_invisible = inputs.positions.view(query, 0);
    u32_t const window_end = inputs.positions.view(query, 1);
    auto& rescored = sweep.rescored[thread];
    rescored.clear();
    for (dense_scored_node_t const& scored : to_span(traversal.results)) {
        u32_t const row = sweep.id_to_row[scored.node_id];
        if (row >= first_invisible && row < window_end) continue;
        f64_t const score = sweep.proxy ? indexer_score(inputs, query, sweep.index.vector(scored.node_id).data())
                                        : 1 - static_cast<f64_t>(scored.distance);
        rescored.emplace_back(score, row);
    }
    if (!sweep.hub_tops.empty()) {
        rescored.insert(rescored.end(), sweep.hub_tops[task].begin(), sweep.hub_tops[task].end());
        std::sort(rescored.begin(), rescored.end(), [](auto a, auto b) { return a.second < b.second; });
        rescored.erase(
            std::unique(rescored.begin(), rescored.end(), [](auto a, auto b) { return a.second == b.second; }),
            rescored.end());
    }
    if (sweep.proxy || !sweep.hub_tops.empty())
        std::sort(rescored.begin(), rescored.end(), [](auto a, auto b) { return a.first > b.first; });

    std::vector<u32_t>& selected = sweep.selected[thread];
    std::size_t const kept = std::min(inputs.top, rescored.size());
    selected.resize(kept);
    for (std::size_t index = 0; index != kept; ++index) selected[index] = rescored[index].second;
    std::sort(selected.begin(), selected.end());
    u32_t const* const truth = inputs.truth.view.row(query).data();
    std::array<u32_t, 4096> exact_rows;
    std::memcpy(exact_rows.data(), truth, inputs.top * sizeof(u32_t));
    std::sort(exact_rows.begin(), exact_rows.begin() + inputs.top);
    std::size_t found = 0;
    for (std::size_t left = 0, right = 0; left != kept && right != inputs.top;) {
        if (selected[left] < exact_rows[right]) ++left;
        else if (exact_rows[right] < selected[left]) ++right;
        else ++found, ++left, ++right;
    }
    sweep.recalls[task] = static_cast<f64_t>(found) / static_cast<f64_t>(inputs.top);
    std::size_t const head = std::min<std::size_t>(64, inputs.top);
    std::size_t head_found = 0;
    for (std::size_t rank = 0; rank != head; ++rank)
        head_found += std::binary_search(selected.begin(), selected.end(), truth[rank]);
    sweep.head_recalls[task] = static_cast<f64_t>(head_found) / static_cast<f64_t>(head);
}

/** The exact scan over the hub rows that every query pays before its walk. */
template <typename index_type_>
struct hub_scan {
    index_type_& index;
    indexer_inputs_t const& inputs;
    std::vector<u32_t> const& row_to_id;
    std::vector<scored_rows_t>& hub_tops;
};

template <typename index_type_>
void hub_callback(fu_lambda_context_t punned, std::size_t task, std::size_t, std::size_t) noexcept {
    auto& scan = *static_cast<hub_scan<index_type_>*>(punned);
    indexer_inputs_t const& inputs = scan.inputs;
    std::size_t const query = inputs.first_query + task;
    u32_t const first_invisible = inputs.positions.view(query, 0);
    u32_t const window_end = inputs.positions.view(query, 1);
    scored_rows_t& best = scan.hub_tops[task];
    for (std::size_t hub = 0; hub != inputs.hubs.view.extent(0); ++hub) {
        u32_t const row = inputs.hubs.view(hub, 0);
        if (row >= first_invisible && row < window_end) continue;
        best.emplace_back(indexer_score(inputs, query, scan.index.vector(scan.row_to_id[row]).data()), row);
    }
    std::size_t const kept = std::min(inputs.top, best.size());
    std::partial_sort(best.begin(), best.begin() + kept, best.end(), [](auto a, auto b) { return a.first > b.first; });
    best.resize(kept);
}

/** Loads the saved graph into one concretely-typed index, then walks every query at every breadth. The kernel only
 *  shaped the graph's edges: every walk scores its pairs under the indexer. */
template <nk_capability_t capability_, typename wait_type_, nk_kernel_kind_t kernel_>
int run_sweeps(attention::settings_t const& settings, indexer_inputs_t const& inputs, fu_pool_t pool) {
    using allocator_t = forkunion_allocator<std::byte>;
    using index_t = dense_ram_index<nk_bf16_k, kernel_, capability_, wait_type_, fu::preferred_atomic_ref,
                                    fu::preferred_cache_hints_t, allocator_t, allocator_t, allocator_t>;
    using traversal_t = typename index_t::dense_traversal_context_t;
    using clock_t = stdc::steady_clock;

    checkpoint_paths_t const paths = checkpoint_paths_of<index_t>(settings.graph, settings.graph.load_name, 0, 1);
    mapped_matrix<nk::bf16_t> stored;
    if (!map_matrix(paths.vectors, stored)) return 1;
    std::size_t const rows = stored.view.extent(0), dimensions = stored.view.extent(1);
    stored = {};
    if (dimensions != inputs.dimensions) {
        fmt::println("The graph holds {}d keys, the queries {}d heads", dimensions, inputs.dimensions);
        return 1;
    }

    auto index = std::make_unique<index_t>(allocator_t {-1}, allocator_t {-1});
    if (failed(index->reserve({rows, connectivity_of(settings.graph.connectivity.value), dimensions}).status)) {
        fmt::println("Failed to reserve {} × {}d", rows, dimensions);
        return 1;
    }
    std::vector<u32_t> id_to_row(rows);
    auto loading_started = clock_t::now();
    if (!load_checkpoint(settings.graph, *index, dimensions, std::span {id_to_row}, 0, 1)) return 1;
    fmt::println("Loaded {} keys of {}d in {:.1f} s", rows, dimensions,
                 stdc::duration<double>(clock_t::now() - loading_started).count());

    std::size_t threads = 0;
    if (fu_pool_threads_count(pool, &threads) != fu_success_k) return 1;
    std::vector<std::unique_ptr<traversal_t>> traversals(threads);
    fu_pool_for_threads(
        pool,
        [](fu_lambda_context_t punned, std::size_t thread, std::size_t) noexcept {
            auto& traversals = *static_cast<std::vector<std::unique_ptr<traversal_t>>*>(punned);
            nk_capability_t available = 0;
            if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
            [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(available);
            traversals[thread] = std::make_unique<traversal_t>(typename traversal_t::allocator_t {allocator_t {-1}});
        },
        &traversals);

    std::size_t const tasks = inputs.query_count - inputs.first_query;
    std::vector<scored_rows_t> hub_tops;
    std::size_t const hubs = inputs.hubs.view.extent(0);
    std::vector<u32_t> row_to_id(rows);
    for (std::size_t id = 0; id != rows; ++id) row_to_id[id_to_row[id]] = static_cast<u32_t>(id);
    if (hubs) {
        hub_tops.resize(tasks);
        hub_scan<index_t> scan {*index, inputs, row_to_id, hub_tops};
        fu_pool_for_n_dynamic(pool, tasks, &hub_callback<index_t>, &scan);
    }
    bool const proxy = settings.proxy == switch_t::present_k;
    fmt::println("\nWalks navigated by {}{}", proxy ? "Σ w·q, rescored" : "the indexer score",
                 hubs ? fmt::format(", joined with {} exactly scanned hubs", hubs) : std::string {});
    for (std::size_t const breadth : settings.breadths.view()) {
        indexer_sweep<index_t> sweep {*index, inputs, id_to_row, traversals, hub_tops, row_to_id};
        sweep.lanes.resize(threads);
        sweep.navigation.assign(threads, std::vector<nk::bf16_t>(dimensions));
        sweep.selected.resize(threads);
        sweep.rescored.resize(threads);
        sweep.seeds.resize(threads);
        sweep.hub_seeds = settings.hub_seeds.value;
        sweep.recalls.resize(tasks);
        sweep.head_recalls.resize(tasks);
        sweep.expansion = breadth;
        sweep.proxy = proxy;

        auto started = clock_t::now();
        fu_pool_for_n_dynamic(pool, tasks, &walk_callback<index_t>, &sweep);
        double const seconds = stdc::duration<double>(clock_t::now() - started).count();
        if (sweep.failures.load()) {
            fmt::println("  breadth {}: {} walks failed", breadth, sweep.failures.load());
            return 1;
        }

        telemetry_lane_t const sums = summed_lanes(sweep.lanes);
        std::vector<f64_t> sorted = sweep.recalls;
        std::sort(sorted.begin(), sorted.end());
        f64_t mean = 0;
        for (f64_t const recall : sorted) mean += recall;
        mean /= static_cast<f64_t>(sorted.size());
        f64_t const queries = static_cast<f64_t>(tasks);
        f64_t head_mean = 0;
        for (f64_t const recall : sweep.head_recalls) head_mean += recall;
        head_mean /= queries;
        double const scored = double(sums.computed_distances) / queries;
        fmt::println("  breadth {:>5}: recall@{} {:.4f}, p10 {:.4f}, p1 {:.4f}; top-64 found {:.4f}", breadth,
                     inputs.top, mean, sorted[sorted.size() / 10], sorted[sorted.size() / 100], head_mean);
        fmt::println("                 {:.0f} keys walked + {} hubs, {:.3f}% of the keys; {:.0f} expanded", scored,
                     hubs, 100.0 * (scored + double(hubs)) / double(rows), double(sums.visited_members) / queries);
        fmt::println("                 {:.0f} queries/s on {} threads", queries / seconds, threads);
    }
    return 0;
}

} // namespace unum::usearch::bench

using namespace unum::usearch;
using namespace unum::usearch::bench;
using namespace unum::usearch::bench::dense;

int main(int argc, char** argv) {
    attention::settings_t const settings = attention::read_settings(argc, argv);
    attention::print(settings);

    indexer_inputs_t inputs;
    if (!map_matrix(settings.queries_path, inputs.queries) || !map_matrix(settings.weights_path, inputs.weights) ||
        !map_matrix(settings.positions_path, inputs.positions) || !map_matrix(settings.truth_path, inputs.truth))
        return 1;
    inputs.heads = inputs.weights.view.extent(1);
    inputs.dimensions = inputs.queries.view.extent(1);
    inputs.query_count = inputs.weights.view.extent(0);
    inputs.top = inputs.truth.view.extent(1);
    if (settings.max_queries.value) inputs.query_count = std::min(inputs.query_count, settings.max_queries.value);
    inputs.first_query = settings.first_query.value;
    if (!settings.hubs_path.empty() && !map_matrix(settings.hubs_path, inputs.hubs)) return 1;
    if (inputs.queries.view.extent(0) != inputs.weights.view.extent(0) * inputs.heads || inputs.heads > 64 ||
        inputs.dimensions > 256 || inputs.top > 4096 || inputs.positions.view.extent(1) < 2 ||
        inputs.truth.view.extent(0) < inputs.query_count || inputs.first_query >= inputs.query_count) {
        fmt::println("Inconsistent inputs: {} head rows of {}d, {} weight rows of {} heads, top {}",
                     inputs.queries.view.extent(0), inputs.dimensions, inputs.weights.view.extent(0), inputs.heads,
                     inputs.top);
        return 1;
    }
    for (std::size_t const breadth : settings.breadths.view())
        if (breadth < inputs.top) {
            fmt::println("Breadth {} returns fewer rows than the judged top {}", breadth, inputs.top);
            return 1;
        }

    fu_topology_t topology = nullptr;
    fu_pool_t pool = nullptr;
    std::size_t logical_cores = 0;
    if (fu_topology_new(&topology) != fu_success_k ||
        fu_pool_new("usearch_bench_attention", fu_capabilities_all_k, &pool) != fu_success_k ||
        fu_logical_cores_count(topology, &logical_cores) != fu_success_k) {
        fmt::println("Failed to inspect the topology or create the pool");
        return 1;
    }
    std::size_t const threads = settings.graph.threads.value ? settings.graph.threads.value
                                                             : std::max<std::size_t>(1, logical_cores * 7 / 8);
    if (fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k) != fu_success_k) {
        fmt::println("Failed to spawn {} threads", threads);
        return 1;
    }
    fmt::println("{} queries of {} heads × {}d, judged at top {}", inputs.query_count, inputs.heads, inputs.dimensions,
                 inputs.top);

    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    nk_capability_t const effective = settings.graph.max_compute & available;
    int const exit_code = dispatch_value<nk_kernel_dot_k, nk_kernel_angular_k>(
        settings.graph.kernel,
        [&](auto kernel) {
            constexpr nk_kernel_kind_t kernel_k = decltype(kernel)::value;
            return dispatch_index<std::size(compute_choices_k)>(
                highest_index(compute_choices_k, effective),
                [&](auto row) {
                    return run_sweeps<prefix_mask(compute_choices_k, row.value), hint_waiter_t, kernel_k>(settings,
                                                                                                          inputs, pool);
                },
                [&] { return run_sweeps<nk_cap_serial_k, hint_waiter_t, kernel_k>(settings, inputs, pool); });
        },
        [&](auto) { return 1; });

    fu_pool_delete(pool);
    fu_topology_delete(topology);
    return exit_code;
}
