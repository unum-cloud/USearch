/**
 *  @file bench/dense/vram_indexes_search.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Profiling harness for searching the sharded @c dense_vram_indexes over real persisted datasets, shared by
 *      the per-dtype benchmark shards.
 *
 *  The multi-device category: one sharded index holding a shard per `--devices` ordinal, driven through one
 *  @c cuda_executor_t with batched insertions, and batched searches scattered over every shard and merged on the
 *  host - the handle the C surface exposes, measured end to end. The cascade ends in the workload its entry hop
 *  names.
 *
 *  Keys are dataset rows, so a merged ranking judges directly against the ground truth and no ID-to-row
 *  bookkeeping exists. Checkpoints ride the sharded index's own SafeGraphs verbs, under the names the CPU sharded
 *  binaries use. The dataset streaming and the process setup come from @c dense/vram_index_search.cuh.
 */
#pragma once

#include <algorithm> // `std::min`
#include <chrono>    // `std::chrono::steady_clock`
#include <numeric>   // `std::iota` - the keys
#include <span>      // `std::span` - the engaged devices
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "dense/vram_index_search.cuh"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** The flag every multi-device binary takes: the device group, one shard per ordinal. */
inline std::array<flag_t, 1> devices_flags(settings_t& settings) noexcept {
    return {{
        {"--devices", "list", "comma-separated ordinals, one shard per device; --device alone by default",
         &settings.devices},
    }};
}

/**
 *  @brief The state one fully-typed sharded-index run shares between its phases: the indexes, the
 *      executor driving them, and the embedded backend-neutral judgment results.
 *
 *  One host thread queues every launch; the shard count, not the thread count, fills the machine.
 */
template <typename indexes_type_>
struct cuda_indexes_context {
    using indexes_t = indexes_type_;
    using index_t = typename indexes_t::index_t;
    using key_t = typename indexes_t::key_t;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using vector_view_t = typename indexes_t::vector_view_t;
    using vectors_view_t = typename indexes_t::vectors_view_t;

    /** The parsed CLI, borrowed for the run's lifetime. */
    settings_t const& settings;

    /** The CUDA ordinals the shards land on, one shard each, in placement order. */
    std::span<int const> devices;

    /** The sharded index under measurement. */
    indexes_t indexes;

    /** One stream per device, paired positionally with the shards. */
    cuda_executor_t executor;

    /** The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    cuda_indexes_context(settings_t const& settings, std::span<int const> devices) noexcept
        : settings(settings), devices(devices), indexes(forkunion_allocator_t {no_memory_domain_k}) {}
};

/** The checkpoint manifest path: dtype, kernel and connectivity stamped into the name, so a
 *  geometry-mismatched run finds nothing rather than restoring the wrong graph. */
template <typename indexes_type_>
std::string checkpoint_path_of(settings_t const& settings, std::string_view name) {
    using index_t = typename indexes_type_::index_t;
    return fmt::format("{}/{}-{}-{}-c{}.safegraphs.json", settings.checkpoints_dir, name,
                       index_t::scalar_t::dtype_name(), nk_kernel_name(index_t::kernel_kind_k),
                       settings.connectivity.value);
}

/** Places one shard per engaged device, opens a stream on each, restores a checkpoint, and reserves for the
 *  viewed rows it lacks - everything fallible, before any clock. */
template <typename streamed_dataset_type_>
bool prepare_indexes(streamed_dataset_type_& streamed) {
    using indexes_t = typename streamed_dataset_type_::bench_t::indexes_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);

    connectivity_t const connectivity = connectivity_of(bench.settings.connectivity.value);
    if (connectivity == connectivity_unknown_k) {
        fmt::println("Unsupported --connectivity {}; pick 16, 32, or 128", bench.settings.connectivity.value);
        return false;
    }
    if (bench.settings.skip_vectors.value > rows) {
        fmt::println("--skip-vectors {} exceeds the {} viewed rows", bench.settings.skip_vectors.value, rows);
        return false;
    }

    for (int const device : bench.devices)
        if (failed(bench.indexes.emplace(device))) {
            fmt::println("Failed to place the shard on device {}", device);
            return false;
        }
    // The executor pairs its streams with the shards positionally, so it opens them in placement order.
    auto [executor, status] = cuda_executor_t::spawn(bench.devices);
    if (failed(status)) {
        fmt::println("Failed to open a stream on every device");
        return false;
    }
    bench.executor = std::move(executor);

    // The checkpoint's own summary sizes the reservation inside `load`, so shards reach it bare.
    if (!bench.settings.load_name.empty()) {
        std::string const path = checkpoint_path_of<indexes_t>(bench.settings, bench.settings.load_name);
        if (status_t const restored = bench.indexes.load(path, load_config_t {}, bench.executor); failed(restored)) {
            fmt::println("Failed to load {}: {}", path, status_to_string(restored));
            return false;
        }
    }

    // A base that misses the skipped prefix would judge recall against rows the graph never saw.
    if (bench.indexes.size() != bench.settings.skip_vectors.value) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", bench.indexes.size(),
                     bench.settings.skip_vectors.value);
        return false;
    }
    if (rows != bench.settings.skip_vectors.value &&
        failed(bench.indexes.reserve({rows, connectivity, dimensions}).status)) {
        fmt::println("Failed to reserve {} × {}d over {} shards", rows, dimensions, bench.indexes.shards_count());
        return false;
    }
    return true;
}

/** Streams the un-ingested topping into the shards, `--batch` rows per call keyed by their dataset rows, every
 *  call dealing its rows across the devices. */
template <typename streamed_dataset_type_>
bool run_indexes_insertion(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using scalar_t = typename bench_t::scalar_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const inserted_rows = rows - bench.settings.skip_vectors.value;
    if (!inserted_rows) {
        fmt::println("\nInsertions");
        fmt::println("  skipped: the loaded base already covers all {} rows", rows);
        return true;
    }

    typename index_t::insert_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_add.value);
    std::size_t const batch_rows = streamed_batch_rows(bench.settings, inserted_rows);
    std::vector<key_t> keys(batch_rows);
    telemetry_lane_t sums;

    progress_bar_t progress {"inserting", inserted_rows, bench.settings.terminal};
    auto started = steady_clock_t::now();
    for (std::size_t offset = 0; offset < inserted_rows;) {
        std::size_t const first_row = bench.settings.skip_vectors.value + offset;
        std::size_t const batch = std::min(batch_rows, inserted_rows - offset);
        std::iota(keys.begin(), keys.begin() + batch, static_cast<key_t>(first_row));
        auto const staged = stage_batch(streamed, streamed.vectors, first_row, batch);
        auto const [result, status] = bench.indexes.insert(std::span<key_t const> {keys.data(), batch}, staged,
                                                           bench.executor, config);
        if (failed(status)) {
            fmt::println("Insertion refused a batch of {} rows at row {}: {}", batch, first_row,
                         status_to_string(status));
            return false;
        }
        sums.visited_members += result.visited_members;
        sums.computed_distances += result.computed_distances;
        sums.restarts += result.restarts;
        sums.saturated_walks += result.visited_saturations;
        offset += batch;
        progress.tick(offset);
    }
    progress.finish();
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();

    auto shard_sizes = std::views::iota(std::size_t {0}, bench.indexes.shards_count()) |
                       std::views::transform([&](std::size_t shard) { return bench.indexes.shard(shard).size(); });
    fmt::println("\nInsertions");
    fmt::println("  shape: {} × {}d", inserted_rows, streamed.vectors.extent(1));
    if (bench.settings.skip_vectors.value) fmt::println("  topping: {} loaded rows", bench.settings.skip_vectors.value);
    fmt::println("  kit: {} · {} · {} kit · devices {}", scalar_t::dtype_name(), nk_kernel_name(index_t::kernel_kind_k),
                 bit_name(cuda_compute_choices_k, index_t::picked_capability_k, "ampere"),
                 fmt::join(bench.devices, ","));
    fmt::println("  shards: {}, one per device, filled {}", bench.indexes.shards_count(), fmt::join(shard_sizes, " "));
    fmt::println("  batch: {} rows per call", batch_rows);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} IPS", humanized(double(inserted_rows) / seconds));
    fmt::println("  distances: {}", humanized(sums.computed_distances));
    fmt::println("  visited: {}", humanized(sums.visited_members));
    fmt::println("  restarts: {}", sums.restarts);
    fmt::println("  saturated: {}", sums.saturated_walks);
    return true;
}

/** Re-selects every node's neighbors shard by shard, pass after pass, then re-links whatever no search can reach;
 *  a pass that re-wires nothing ends the phase. */
template <typename streamed_dataset_type_>
bool run_indexes_refinement(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using indexes_t = typename bench_t::indexes_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    std::size_t const max_refines = bench.settings.max_refines.value;
    std::size_t const nodes = bench.indexes.size();
    if (!max_refines || !nodes) return true;

    typename indexes_t::refine_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_refine.value);

    refinement_report_t report;
    report.nodes = nodes;
    report.expansion = bench.indexes.shard(0).non_zero_expansion_factor(config);
    report.threads = 1;
    std::vector<std::size_t> changed_per_pass;
    changed_per_pass.reserve(max_refines);
    progress_bar_t progress {"refining", nodes * max_refines, bench.settings.terminal};
    auto started = steady_clock_t::now();
    for (std::size_t pass = 0; pass != max_refines; ++pass) {
        std::size_t const swept_before = pass * nodes;
        auto const [refined, refined_status] = bench.indexes.refine(
            bench.executor, config, [&](std::size_t done, std::size_t) noexcept {
                return progress.tick(swept_before + done), continuation_t::continue_k;
            });
        if (failed(refined_status)) {
            fmt::println("Refinement failed on pass {}: {}", pass, status_to_string(refined_status));
            return false;
        }
        changed_per_pass.push_back(refined.changed_nodes);
        report.sums.visited_members += refined.visited_members;
        report.sums.computed_distances += refined.computed_distances;
        report.sums.restarts += refined.restarts;
        report.islands_found += refined.islands_found;
        report.islands_reconnected += refined.islands_reconnected;
        report.reachable_nodes = refined.reachable_nodes;
        report.sweeps = refined.sweeps;
        if (!refined.changed_nodes) break;
    }
    progress.finish();
    // Every pass heals what it strands, so the passes' wall time holds the healing too.
    report.refining_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    report.changed_per_pass = changed_per_pass;
    report_refinement_metrics(report);
    return true;
}

/** Serializes the sharded index as one SafeGraphs checkpoint - a monolith per shard and the manifest committed
 *  last - so later runs @c --load it instead of rebuilding. */
template <typename streamed_dataset_type_>
bool save_indexes(streamed_dataset_type_& streamed) {
    using indexes_t = typename streamed_dataset_type_::bench_t::indexes_t;
    auto& bench = streamed.bench;
    std::string const path = checkpoint_path_of<indexes_t>(bench.settings, bench.settings.save_name);
    // A view of the parts must stage its queries under the factor the rows were rescaled by.
    std::string const scale = fmt::format("{}", streamed.scale);
    safegraphs_custom_field_t const provenance[] = {{"scale", scale}};
    save_config_t config;
    config.layout = checkpoint_layout_t::distributed_k;
    config.provenance = provenance;
    auto started = steady_clock_t::now();
    if (status_t const saved = bench.indexes.save(path, config, bench.executor); failed(saved)) {
        fmt::println("Failed to save {}: {}", path, status_to_string(saved));
        return false;
    }
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("Saved {} shards: {:.2f} GB in {:.2f} s", bench.indexes.shards_count(),
                 gigabytes(bench.indexes.storage_requirements(config).value), seconds);
    return true;
}

/** Scatters every query over the shards through batched @c search calls, ranked by its heads under an indexer, the
 *  merged rankings landing straight in the judgment matrices; a batch-shaped verb has no per-query clock, so the
 *  report prints the mean. */
template <typename streamed_dataset_type_>
bool run_indexes_search(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using small_size_t = typename bench_t::small_size_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    auto& bench = streamed.bench;
    settings_t const& settings = bench.settings;
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t const wanted = settings.top_k.value;
    bench_indexer_heads<typename index_t::scalar_t> heads;
    if (!open_indexer_heads<index_t>(streamed, heads)) return false;
    std::size_t const query_count = searched_count(streamed, heads);
    if (!reserve_results(bench.results, query_count, wanted)) return false;

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(settings.expansion_search.value);
    config.count = static_cast<small_size_t>(wanted);
    config.synchronization = synchronization_t::immutable_k;
    std::size_t const batch_rows = streamed_batch_rows(settings, query_count);
    std::vector<std::size_t> counts(batch_rows);
    telemetry_lane_t sums;

    progress_bar_t progress {"searching", query_count, settings.terminal};
    double seconds = 0;
    for (std::size_t base = 0; base < query_count;) {
        std::size_t const count = std::min(batch_rows, query_count - base);
        vectors_view_t const batch = !heads.navigation.empty()
                                         ? vectors_view_t {heads.navigation.data() + base * dimensions,
                                                           {count, dimensions}}
                                         : stage_batch(streamed, searched_rows(streamed), base, count);
        std::span<key_t> const keys {bench.results.hit_rows.row(base).data(), count * wanted};
        std::span<dense_distance_t> const distances {bench.results.hit_distances.row(base).data(), count * wanted};
        auto search = [&](auto const&... ranking) noexcept {
            return bench.indexes.search(batch, ranking..., bench.executor, config, wanted, keys, distances,
                                        std::span<std::size_t> {counts.data(), count});
        };

        auto started = steady_clock_t::now();
        auto const [result, status] = [&]() noexcept {
            if constexpr (scores_indexer<index_t>())
                if (heads.engaged()) return search(heads.rows(base, count));
            return search();
        }();
        seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Search refused a batch of {} queries at query {}: {}", count, base, status_to_string(status));
            return false;
        }
        sums.visited_members += result.visited_members;
        sums.computed_distances += result.computed_distances;
        sums.restarts += result.restarts;
        sums.saturated_walks += result.visited_saturations;
        for (std::size_t query = 0; query != count; ++query)
            bench.results.hit_counts[base + query] = static_cast<u32_t>(counts[query]);
        base += count;
        progress.tick(base);
    }
    progress.finish();

    // The empty latency span asks the shared report for the batch-uniform mean.
    report_search_metrics(settings, query_count, seconds, 1, sums, {});
    judge_search(settings, heads, bench.results, streamed.ground_truth, streamed.vectors, query_count);
    return true;
}

/** The search workload the cascade ends in: batched searches scattered over the built shards, judged against the
 *  ground truth. */
struct indexes_search_workload_t {

    /** Whether the workload is defined over the dot kernel alone, so the cascade pins it instead of
     *  dispatching the CLI's metric. */
    static constexpr bool dot_only_k = false;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_indexes_search(streamed);
    }
};

#pragma endregion

#pragma region Dispatch Cascade

/** The innermost hop: binds the concrete sharded-index instantiation over every engaged device, builds it in
 *  checkpoint order, then runs @p workload. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, cuda_arch_t capability_,
          typename workload_type_>
int stage_indexes_and_run(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                          std::span<int const> devices, workload_type_ const& workload) {
    cuda_indexes_context<dense_vram_indexes<u32_t, stored_dtype_, kernel_, capability_>> bench {settings, devices};
    return stream_and_run(dataset, bench, [&](auto& streamed) {
        return prepare_indexes(streamed) && run_indexes_insertion(streamed) && run_indexes_refinement(streamed) &&
               (settings.save_name.empty() || save_indexes(streamed)) && workload.run(streamed);
    });
}

/** Kernel axis: lowers the runtime metric to a compile-time kernel kind, the terminal hop into
 *  @ref stage_indexes_and_run; a dot-only workload pins the dot kernel instead. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, cuda_arch_t capability_, typename workload_type_>
int run_indexes_with_metric(utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                            settings_t const& settings, std::span<int const> devices, workload_type_ const& workload) {
    if constexpr (workload_type_::dot_only_k)
        return stage_indexes_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_dot_k, capability_>(
            dataset, settings, devices, workload);
    else
        return dispatch_kernel<stored_dtype_>(
            settings.kernel,
            [&](auto kernel) {
                return stage_indexes_and_run<dataset_scalar_type_, stored_dtype_, kernel.value, capability_>(
                    dataset, settings, devices, workload);
            },
            [&](auto metric) {
                fmt::println("Unsupported metric {} for {} vectors", nk_kernel_name(metric), dtype_name(stored_dtype_));
                return 1;
            });
}

/** Compute axis: the named table row's ceiling instantiates every cell below; past the tail sit the Ampere kits
 *  every GPU runs. Every shard keeps its planes in VRAM. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, typename workload_type_>
int run_indexes_with_compute(utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                             settings_t const& settings, std::span<int const> devices, cuda_arch_t max_compute,
                             workload_type_ const& workload) {
    auto const named = std::ranges::find(cuda_compute_choices_k, max_compute, &capability_choice<cuda_arch_t>::bit);
    return dispatch_index<std::size(cuda_compute_choices_k)>(
        static_cast<std::size_t>(named - std::begin(cuda_compute_choices_k)),
        [&](auto row) {
            return run_indexes_with_metric<dataset_scalar_type_, stored_dtype_, cuda_compute_choices_k[row.value].bit>(
                dataset, settings, devices, workload);
        },
        [&] {
            return run_indexes_with_metric<dataset_scalar_type_, stored_dtype_, cuda_arch_ampere_k>(dataset, settings,
                                                                                                    devices, workload);
        });
}

/** The shard hop: the compute ceiling and the metric below one stored dtype, so the per-dtype shard translation
 *  units carry the whole device instantiation grid of their dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_indexes_with_stored_dtype(utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                                  settings_t const& settings, std::span<int const> devices, cuda_arch_t max_compute) {
    return run_indexes_with_compute<dataset_scalar_type_, stored_dtype_>(dataset, settings, devices, max_compute,
                                                                         indexes_search_workload_t {});
}

#pragma endregion

} // namespace unum::usearch::bench::dense
