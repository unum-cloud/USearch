/**
 *  @file bench/dense/nvme_indexes_search.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Profiling harness for searching the sharded @c dense_nvme_indexes over parts a RAM or
 *      VRAM bench saved, shared by the per-dtype benchmark shards.
 *
 *  No graph is built or loaded in RAM: `--view` names the manifest or the monolith, and the first
 *  part's header names the stored dtype and the metric the cascade instantiates, and the factor the
 *  producer rescaled its rows by. Every query is answered from storage in one batched call, each
 *  timed by the verb from its first armed walk to its final write, and every read counted by
 *  purpose. Keys are dataset rows, so the merged rankings judge directly against `--neighbors`.
 *
 *  @code{.sh}
 *  usearch_bench_dense_nvme_indexes_search --view /mnt/nvme0/sift10m.safegraphs.json --queries ... --neighbors ... \
 *      --transport linux-uring --queue-depth 512
 *  @endcode
 */
#pragma once

#include <cstdlib> // `std::strtof` - the recorded scale

#include <algorithm>    // `std::min`, `std::nth_element`
#include <array>        // `std::array`
#include <bit>          // `std::bit_cast` - the header prefix
#include <chrono>       // `std::chrono::steady_clock`
#include <filesystem>   // `std::filesystem::is_directory` - a part moved by `--paths`
#include <span>         // `std::span`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`
#include <vector>       // `std::vector`

#include <fmt/format.h>

#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/utilities/nvme_device.hpp"

#include "dense_search.cuh"
#include "utilities/persisted_dataset.hpp"

namespace unum::usearch::bench::dense {

#pragma region Settings

/** The `--transport` names, parallel to @ref transports_k. */
inline constexpr std::array<std::string_view, 4> transport_names_k {"auto", "blocking", "linux-aio", "linux-uring"};

/** The transports `--transport` picks among; `auto` takes io_uring, then AIO, then blocking
 *  reads. */
inline constexpr std::array<io_transport_t, 4> transports_k {io_transport_t::automatic_k, io_transport_t::blocking_k,
                                                             io_transport_t::linux_aio_k,
                                                             io_transport_t::linux_uring_k};

/** The name of the transport an engine opened, in the grammar `--transport` parses. */
inline std::string_view transport_name(io_transport_t transport) noexcept {
    switch (transport) {
    case io_transport_t::blocking_k: return "blocking";
    case io_transport_t::linux_aio_k: return "linux-aio";
    case io_transport_t::linux_uring_k: return "linux-uring";
    case io_transport_t::linux_uring_nvme_k: return "linux-uring-nvme";
    case io_transport_t::linux_spdk_k: return "linux-spdk";
    case io_transport_t::windows_ioring_k: return "windows-ioring";
    default: return "auto";
    }
}

/** The flags only the NVMe binaries take, beside the @ref settings_t fields they share with every
 *  binary, and the scale their parts recorded. */
struct nvme_settings_t {

    /** The manifest or the monolith the parts are viewed through. */
    std::string_view view_path;

    /** Where to serve each part from instead, comma-separated: a directory holding the part's file,
     *  or a file or device holding the part itself; empty serves every part beside the manifest. */
    std::string_view paths_list;
    std::vector<std::string> paths;

    /** The engines' transport as named, and the one it resolves to. */
    std::string_view transport_name;
    io_transport_t transport = io_transport_t::automatic_k;

    /** Reads every thread's engine keeps outstanding at most, which bounds the walks it
     *  overlaps. */
    count_t queue_depth {.value = 512};

    /** The factor the producer rescaled its rows by before the downcast, from the first part's
     *  header; one when it recorded none. */
    f32_t scale = 1;
};

/** The flags both NVMe binaries take: the parts and where they are served from, the queries, the
 *  engines, and the walks. */
inline std::array<flag_t, 9> nvme_flags(settings_t& settings, nvme_settings_t& nvme) noexcept {
    return {{
        {"--view", "path", "manifest or monolith of the saved parts", &nvme.view_path},
        {"--paths", "list", "comma-separated directory, file or device per part; beside the manifest by default",
         &nvme.paths_list},
        {"--queries", "path", "query vectors, `.fbin` family, cast to the parts' stored dtype", &settings.queries_path},
        {"--transport", "name", "auto, blocking, linux-aio or linux-uring; auto by default", &nvme.transport_name},
        {"--queue-depth", "reads", "reads every thread's engine keeps outstanding; 512 by default", &nvme.queue_depth},
        {"--expansion-search", "breadth", "search candidate pool; 96 by default", &settings.expansion_search},
        {"--top-k", "count", "matches per query to judge; 10 by default", &settings.top_k},
        {"--threads", "count", "threads; 7/8 of the logical cores by default", &settings.threads},
        {"--max-compute", "name", "compute ceiling by microarchitecture, like `icelake`, or `serial`",
         &settings.max_compute_name},
    }};
}

/** The flags only the NVMe search binary takes: the truth its hits are judged against, and a
 *  lightning indexer's heads to rank by. */
inline std::array<flag_t, 3> nvme_search_flags(settings_t& settings) noexcept {
    return {{
        {"--neighbors", "path", "ground truth over the dataset rows the parts' keys name, `.ibin`",
         &settings.neighbors_path},
        {"--heads", "path", "any `.fbin`-family file, every query's indexer heads", &settings.heads_path},
        {"--weights", "path", "any `.fbin`-family file, one row of head weights per query", &settings.weights_path},
    }};
}

/** Resolves the transport name and splits the part paths, printing the first flag that does not
 *  parse. */
inline bool resolve_nvme_settings(nvme_settings_t& nvme) {
    if (!nvme.transport_name.empty()) {
        auto const named = std::ranges::find(transport_names_k, nvme.transport_name);
        if (named == transport_names_k.end())
            return fmt::println(stderr, "--transport=\"{}\" does not parse, expected one of {}", nvme.transport_name,
                                fmt::join(transport_names_k, ", ")),
                   false;
        nvme.transport = transports_k[named - transport_names_k.begin()];
    }
    for (std::string_view list = nvme.paths_list; !list.empty();) {
        std::size_t const comma = list.find(',');
        nvme.paths.emplace_back(list.substr(0, comma));
        list = comma == std::string_view::npos ? std::string_view {} : list.substr(comma + 1);
    }
    return true;
}

/** The SafeGraphs header at the front of @p path, a file or a device: its prefix and the JSON the
 *  prefix sizes; empty when none starts there. */
inline std::vector<byte_t> read_header_bytes(std::string_view path) {
    auto [file, open_status] = seekable_file_t::open(path, file_mode_t::read_k);
    std::array<byte_t, sizeof(safegraphs_prefix_t)> prefix_bytes {};
    if (failed(open_status) || failed(file.read_at(0, prefix_bytes))) return {};
    safegraphs_prefix_t const prefix = std::bit_cast<safegraphs_prefix_t>(prefix_bytes);
    if (prefix.magic != safegraphs_prefix_t::magic_k) return {};
    std::vector<byte_t> header(sizeof(prefix) + static_cast<std::size_t>(little_endian(prefix.header_length)));
    if (failed(file.read_at(0, header))) return {};
    return header;
}

/** Reads the stored dtype and the metric into @p settings, and the scale into @p nvme, from the
 *  first part's header, wherever `--view` and `--paths` serve that part from. */
inline bool read_parts_header(settings_t& settings, nvme_settings_t& nvme) {
    std::string_view const view = nvme.view_path;
    std::vector<byte_t> header = read_header_bytes(view);
    std::string entry;
    if (header.empty()) {
        // A manifest lists its entries before opening any part, so moved parts keep their names.
        auto per_part = [&](safegraphs_part_t part) noexcept {
            if (entry.empty()) entry = part.path;
        };
        [[maybe_unused]] auto const listed = safegraphs_read_manifest(view, std::allocator<byte_t> {}, per_part);
        if (entry.empty())
            return fmt::println("--view {} is neither a SafeGraphs monolith nor a manifest", view), false;
    }
    std::string first;
    std::error_code directory_error;
    if (!nvme.paths.empty())
        first = std::filesystem::is_directory(nvme.paths.front(), directory_error)
                    ? fmt::format("{}/{}", nvme.paths.front(), safegraphs_basename(entry.empty() ? view : entry))
                    : nvme.paths.front();
    else if (!entry.empty())
        first = fmt::format("{}{}", view.substr(0, view.size() - safegraphs_basename(view).size()), entry);
    if (!first.empty()) header = read_header_bytes(first);
    if (header.empty()) return fmt::println("Failed to read the SafeGraphs header of {}", first), false;

    safegraphs_name_t metric, quantization, scale;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = safegraphs_name_t {field.value};
        if (field.key == "quantization") quantization = safegraphs_name_t {field.value};
        if (field.key == "scale") scale = safegraphs_name_t {field.value};
    };
    if (failed(safegraphs_read_header(bytes_view_t {header}, std::allocator<byte_t> {}, per_custom_field).status))
        return fmt::println("Failed to parse the SafeGraphs header of {}", first.empty() ? view : first), false;
    settings.kernel = nk_kernel_named(metric.characters, metric.view().size());
    settings.stored_dtype = nk_dtype_named(quantization.characters, quantization.view().size());
    if (settings.kernel == nk_kernel_unknown_k || settings.stored_dtype == nk_dtype_unknown_k)
        return fmt::println("The parts name metric \"{}\" and quantization \"{}\"; both are needed to dispatch",
                            metric.view(), quantization.view()),
               false;
    if (!scale.empty()) nvme.scale = std::strtof(scale.characters, nullptr);
    return true;
}

/** Prints the NVMe binaries' settings as "- Name: value", the stored dtype, the metric and the
 *  scale as the parts' header names them. */
inline void print(settings_t const& settings, nvme_settings_t const& nvme) {
    fmt::println("- View: {}", nvme.view_path);
    fmt::println("- Paths: {}", nvme.paths_list.empty() ? "beside the manifest" : nvme.paths_list);
    fmt::println("- Queries: {}", settings.queries_path);
    if (!settings.neighbors_path.empty()) fmt::println("- Neighbors: {}", settings.neighbors_path);
    if (!settings.vectors_path.empty()) fmt::println("- Vectors: {}", settings.vectors_path);
    fmt::println("- Stored dtype: {}", dtype_name(settings.stored_dtype));
    fmt::println("- Metric: {}", nk_kernel_name(settings.kernel));
    fmt::println("- Scale: {}", nvme.scale);
    fmt::println("- Max compute: {}",
                 settings.max_compute_name.empty() ? "the newest detected" : settings.max_compute_name);
    if (settings.threads.value) fmt::println("- Threads: {}", settings.threads.value);
    else fmt::println("- Threads: 7/8 of the logical cores");
    fmt::println("- Expansion search: {}", settings.expansion_search.value);
    fmt::println("- Top k: {}", settings.top_k.value);
    fmt::println("- Transport: {}", nvme.transport_name.empty() ? "auto" : nvme.transport_name);
    fmt::println("- Queue depth: {}", nvme.queue_depth.value);
    if (!settings.heads_path.empty()) {
        fmt::println("- Heads: {}", settings.heads_path);
        fmt::println("- Weights: {}", settings.weights_path);
    }
}

#pragma endregion

#pragma region Pipeline

/** The state one fully-typed run shares between its phases: the parts viewed from storage, the
 *  executor whose engines read them, and the backend-neutral judgment results. */
template <typename nvme_type_>
struct nvme_context {
    using indexes_t = nvme_type_;
    using scalar_t = typename nvme_type_::scalar_t;
    using vector_view_t = typename nvme_type_::vector_view_t;

    /** The parsed CLI, borrowed for the run's lifetime. */
    settings_t const& settings;

    nvme_type_ viewed;
    typename nvme_type_::executor_t executor;
    bench_results_t results;

    explicit nvme_context(settings_t const& settings) noexcept
        : settings(settings), viewed(forkunion_allocator_t {no_memory_domain_k}) {}
};

/** Spawns the executor on the transport `--transport` names and views the parts - everything
 *  fallible, before any clock. */
template <typename nvme_type_>
bool open_nvme(nvme_settings_t const& nvme, fu_topology_t topology, nvme_context<nvme_type_>& context) {
    using executor_t = typename nvme_type_::executor_t;
    io_engine_config_t config;
    config.queue_depth = nvme.queue_depth.value;
    config.transport = nvme.transport;
    auto [executor, spawn_status] = executor_t::spawn(topology, context.settings.threads.value, config);
    if (failed(spawn_status))
        return fmt::println("Failed to open the {} engines: {}", transport_name(nvme.transport),
                            status_to_string(spawn_status)),
               false;
    context.executor = std::move(executor);

    // AMX tile state needs a per-thread OS permission handshake before first use.
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    if (failed(context.executor.for_threads([available](std::size_t, std::size_t) noexcept {
            [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(available);
        })))
        return fmt::println("Failed to configure the storage threads"), false;

    std::vector<std::string_view> const paths(nvme.paths.begin(), nvme.paths.end());
    if (status_t const viewed = context.viewed.view(nvme.view_path, paths); failed(viewed))
        return fmt::println("Failed to view {}: {}", nvme.view_path, status_to_string(viewed)), false;

    fmt::println("\nStorage");
    fmt::println("  parts: {} · {} rows · {}d", context.viewed.shards_count(), context.viewed.size(),
                 context.viewed.dimensions());
    fmt::println("  transport: {} · queue depth {} · {} threads",
                 transport_name(context.executor.engine(0).transport()), nvme.queue_depth.value,
                 context.executor.size());
    return true;
}

/** Prints what a storage phase read per row by purpose and the rates it sustained from @p reads,
 *  then the p50 and p99 of @p latencies_ns, one clock per row, unless empty; @p latencies_ns
 *  reorders in place. */
template <typename result_type_>
void report_storage_metrics(std::string_view unit, std::size_t rows_count, double seconds, result_type_ const& reads,
                            std::span<u64_t> latencies_ns) {
    auto percentile_us = [&](double share) {
        std::size_t const rank = std::min(latencies_ns.size() - 1, std::size_t(share * double(latencies_ns.size())));
        std::nth_element(latencies_ns.begin(), latencies_ns.begin() + rank, latencies_ns.end());
        return double(latencies_ns[rank]) / 1000.0;
    };
    auto per_row = [&](dense_nvme_reads_t const& purpose) {
        return fmt::format("{} · {:.1f} KB", humanized(double(purpose.count) / double(rows_count)),
                           double(purpose.bytes) / double(rows_count) / 1024.0);
    };
    dense_nvme_reads_t total;
    for (dense_nvme_reads_t const& purpose :
         {reads.neighbor_reads, reads.value_reads, reads.key_reads, reads.scan_reads})
        total += purpose;
    fmt::println("  reads/{}: neighbors {} · values {} · keys {} · scans {}", unit, per_row(reads.neighbor_reads),
                 per_row(reads.value_reads), per_row(reads.key_reads), per_row(reads.scan_reads));
    fmt::println("  round trips/{}: {:.1f}", unit, double(reads.round_trips) / double(rows_count));
    fmt::println("  storage: {} IOPS · {:.2f} GB/s", humanized(double(total.count) / seconds),
                 gigabytes(total.bytes) / seconds);
    if (!latencies_ns.empty())
        fmt::println("  latency/{}: p50 {:.0f} µs · p99 {:.0f} µs", unit, percentile_us(0.50), percentile_us(0.99));
}

/** Answers every query from storage in one batched @c search call, which arms as many rows as the
 *  queue depth holds, the merged rankings landing straight in the judgment matrices. */
template <typename streamed_dataset_type_>
bool run_nvme_search(streamed_dataset_type_& streamed) {
    using nvme_t = typename streamed_dataset_type_::bench_t::indexes_t;
    using key_t = typename nvme_t::key_t;
    using scalar_t = typename nvme_t::scalar_t;
    using vectors_view_t = typename nvme_t::vectors_view_t;
    auto& nvme = streamed.bench;
    settings_t const& settings = nvme.settings;
    std::size_t const rows = nvme.viewed.size(), dimensions = streamed.queries.extent(1);
    std::size_t const wanted = settings.top_k.value;

    // Keys are dataset rows, so truth naming a row past the parts was computed over another base.
    nk::matrix_view<u32_t> const truth = streamed.ground_truth;
    u32_t farthest = 0;
    for (std::size_t query = 0; query != truth.extent(0); ++query)
        for (u32_t const row : to_span(truth.row(query).as_vector())) farthest = std::max(farthest, row);
    if (farthest >= rows)
        return fmt::println("--neighbors names row {}, past the {} rows the parts hold", farthest, rows), false;

    bench_indexer_heads<scalar_t> heads;
    if (!open_indexer_heads<nvme_t>(streamed, heads)) return false;
    std::size_t const search_count = searched_count(streamed, heads);
    if (!reserve_results(nvme.results, search_count, wanted)) return false;
    // Queries land in the stored scalar before the clock, so the call times storage, not casts.
    std::vector<scalar_t> const staged = stage_rows<scalar_t>(streamed.queries, streamed.scale);
    vectors_view_t const queries {staged.data(), search_count, dimensions};

    typename nvme_t::search_config_t config;
    config.expansion_factor = static_cast<u32_t>(settings.expansion_search.value);
    std::span<key_t> const keys {nvme.results.hit_rows.data(), search_count * wanted};
    std::span<dense_distance_t> const distances {nvme.results.hit_distances.data(), search_count * wanted};
    std::vector<std::size_t> counts(search_count);
    std::vector<u64_t> latencies_ns(search_count);
    auto search = [&](auto const&... ranking) noexcept {
        return nvme.viewed.search(queries, ranking..., nvme.executor, config, wanted, keys, distances,
                                  std::span {counts}, std::span {latencies_ns});
    };

    auto const started = steady_clock_t::now();
    auto const [result, status] = [&]() noexcept {
        if constexpr (scores_indexer<nvme_t>())
            if (heads.engaged()) return search(heads.rows(0, search_count));
        return search();
    }();
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    if (failed(status))
        return fmt::println("Search refused {} queries: {}", search_count, status_to_string(status)), false;

    telemetry_lane_t sums;
    sums.visited_members = result.visited_members;
    sums.computed_distances = result.computed_distances;
    sums.restarts = result.restarts;
    sums.saturated_walks = result.saturated_walks;
    std::vector<float> latencies_us(search_count);
    for (std::size_t query = 0; query != search_count; ++query) {
        nvme.results.hit_counts[query] = static_cast<u32_t>(counts[query]);
        latencies_us[query] = static_cast<float>(double(latencies_ns[query]) / 1000.0);
    }
    report_search_metrics(settings, search_count, seconds, nvme.executor.size(), sums, std::span {latencies_us});
    report_storage_metrics("query", search_count, seconds, result, {});
    judge_search(settings, heads, nvme.results, truth, streamed.vectors, search_count);
    return true;
}

/** The search workload the cascade ends in, under the kernel the parts were built for. */
struct nvme_search_workload_t {

    /** Whether the workload is defined over the dot kernel alone, so the cascade pins it instead of
     *  dispatching the parts' metric. */
    static constexpr bool dot_only_k = false;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_nvme_search(streamed);
    }
};

#pragma endregion

#pragma region Dispatch Cascade

/** Gate: a (stored dtype, kernel) cell is viewable once its kit chain is adapted in
 *  @c usearch/dense/ram_metric.hpp; unadapted combinations report as skipped. */
template <nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_>
concept ram_kit_adapted = requires {
    sizeof(dense_quads_chain<kernel_, typename nk::type_for<stored_dtype_>::type, capability_>);
};

/** No-op sibling picked when the cell's CPU kit chain is not adapted yet. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename workload_type_>
int view_and_run(bench_launch<dataset_scalar_type_> const&, nvme_settings_t const&, workload_type_ const&) {
    fmt::println("skipped: no CPU kit adapted for {} · {}", dtype_name(stored_dtype_), nk_kernel_name(kernel_));
    return 0;
}

/** The innermost hop: binds the concrete sharded NVMe instantiation, views the parts, and hands
 *  them, the queries and their truth to @p workload. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename workload_type_>
    requires ram_kit_adapted<stored_dtype_, kernel_, capability_>
int view_and_run(bench_launch<dataset_scalar_type_> const& launch, nvme_settings_t const& nvme,
                 workload_type_ const& workload) {
    using nvme_t = dense_nvme_indexes<u32_t, stored_dtype_, kernel_, capability_>;
    using context_t = nvme_context<nvme_t>;
    using vectors_t = nk::matrix_view<dataset_scalar_type_>;
    utilities::persisted_dataset<dataset_scalar_type_> const& dataset = launch.dataset;
    context_t context {launch.settings};
    if (!open_nvme(nvme, launch.machine.topology, context)) return 1;
    if (dataset.dimensions() != context.viewed.dimensions())
        return fmt::println("--queries holds {}d rows, but the parts store {}d", dataset.dimensions(),
                            context.viewed.dimensions()),
               1;

    vectors_t const vectors = launch.settings.vectors_path.empty() ? vectors_t {} : dataset.vectors();
    streamed_dataset<dataset_scalar_type_, context_t> streamed {context, vectors, dataset.queries()};
    streamed.ground_truth = dataset.neighbors();
    streamed.scale = nvme.scale;
    return workload.run(streamed) ? 0 : 1;
}

/** Kernel axis: lowers the metric the parts name to a compile-time kernel kind; a dot-only workload
 *  pins the dot kernel instead. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename workload_type_>
int view_with_metric(bench_launch<dataset_scalar_type_> const& launch, nvme_settings_t const& nvme,
                     workload_type_ const& workload) {
    if constexpr (workload_type_::dot_only_k)
        return view_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_dot_k, capability_>(launch, nvme, workload);
    else
        return dispatch_kernel<stored_dtype_>(
            launch.settings.kernel,
            [&](auto kernel) {
                return view_and_run<dataset_scalar_type_, stored_dtype_, kernel.value, capability_>(launch, nvme,
                                                                                                    workload);
            },
            [&](auto metric) {
                fmt::println("The parts name metric {}, which {} rows do not take", nk_kernel_name(metric),
                             dtype_name(stored_dtype_));
                return 1;
            });
}

/** Compute axis: the allow-mask intersected with the runtime detection picks the newest table
 *  row, whose @ref prefix_mask ceiling instantiates @ref view_with_metric; past the tail sit the
 *  serial kernels every machine owns. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, typename workload_type_>
int view_with_compute(bench_launch<dataset_scalar_type_> const& launch, nvme_settings_t const& nvme,
                      workload_type_ const& workload) {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    nk_capability_t const effective = launch.settings.max_compute & available;
    return dispatch_index<std::size(compute_choices_k)>(
        highest_index(compute_choices_k, effective),
        [&](auto row) {
            return view_with_metric<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_choices_k, row.value)>(
                launch, nvme, workload);
        },
        [&] { return view_with_metric<dataset_scalar_type_, stored_dtype_, nk_cap_serial_k>(launch, nvme, workload); });
}

/** The NVMe shard hop: everything below one stored dtype, so a per-dtype shard translation unit
 *  carries the whole instantiation grid of its dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_nvme_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch, nvme_settings_t const& nvme) {
    return view_with_compute<dataset_scalar_type_, stored_dtype_>(launch, nvme, nvme_search_workload_t {});
}

#pragma endregion

#pragma region Process Setup

/** The process both NVMe binaries run around their cascade: reads the flags of @p binary_name and
 *  the ones each of @p binary_flags adds, reads the stored dtype, the metric and the scale from the
 *  parts, probes the machine, prints both, then opens the queries and exits with what @p on_launch
 *  makes of their @ref bench_launch. */
template <typename on_launch_type_, typename... binary_flags_types_>
int run_nvme_bench(int argc, char const* const* argv, char const* binary_name, nvme_settings_t& nvme,
                   on_launch_type_&& on_launch, binary_flags_types_... binary_flags) {
    settings_t settings;
    std::vector<flag_t> flags;
    auto const append = [&](auto const& added) { flags.insert(flags.end(), added.begin(), added.end()); };
    append(nvme_flags(settings, nvme));
    (append(binary_flags(settings)), ...);
    parse_or_exit(argc, argv, fmt::format("{} --view <path> --queries <path> [flags]", binary_name), flags);
    if (nvme.view_path.empty() || settings.queries_path.empty())
        return fmt::println(stderr, "--view and --queries are required, see --help"), 1;
    resolve_max_compute(settings);
    if (!resolve_nvme_settings(nvme) || !read_parts_header(settings, nvme)) return 1;

    machine_t machine = probe_machine();
    if (!machine.topology || !machine.fabric) return fmt::println("Failed to inspect the topology"), 1;
    print(machine);
    print(settings, nvme);

    // Without `--vectors` the queries stand in as the opened base, which no phase then reads.
    std::string_view const base = settings.vectors_path.empty() ? settings.queries_path : settings.vectors_path;
    int const exit_code = open_dataset(
        base, settings.queries_path, settings.neighbors_path,
        [&]<typename dataset_scalar_type_>(utilities::persisted_dataset<dataset_scalar_type_> const& dataset) {
            return on_launch(bench_launch<dataset_scalar_type_> {dataset, settings, machine});
        });
    fu_fabric_delete(machine.fabric);
    fu_topology_delete(machine.topology);
    return exit_code;
}

#pragma endregion

} // namespace unum::usearch::bench::dense
