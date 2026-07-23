/**
 *  @brief The CLI entry of the CPU dense-index benchmark: option parsing, the pool &
 *      topology setup, and the dataset-scalar hop into the per-dtype shards.
 *  @author Ash Vardanian
 *  @file bench/dense_index.cpp
 *  @date July 23, 2026
 *
 *  The compile-time cascade lives in @c dense_index.hpp; this translation unit stays
 *  dtype-free by extern-declaring the @c run_with_compute instantiations the per-dtype
 *  shards - @c dense_index_bf16.cpp, @c dense_index_i8.cpp, @c dense_index_u8.cpp, and
 *  @c dense_index_i4.cpp - carry, so the heavy kernel grids build in parallel and this
 *  unit recompiles in seconds.
 */
#include "dense_index.hpp"

extern template int run_with_compute<nk::f32_t, nk_bf16_k>(bench_launch<nk::f32_t> const&);
extern template int run_with_compute<nk::bf16_t, nk_bf16_k>(bench_launch<nk::bf16_t> const&);
extern template int run_with_compute<nk::i8_t, nk_bf16_k>(bench_launch<nk::i8_t> const&);
extern template int run_with_compute<nk::u8_t, nk_bf16_k>(bench_launch<nk::u8_t> const&);

extern template int run_with_compute<nk::f32_t, nk_i8_k>(bench_launch<nk::f32_t> const&);
extern template int run_with_compute<nk::bf16_t, nk_i8_k>(bench_launch<nk::bf16_t> const&);
extern template int run_with_compute<nk::i8_t, nk_i8_k>(bench_launch<nk::i8_t> const&);
extern template int run_with_compute<nk::u8_t, nk_i8_k>(bench_launch<nk::u8_t> const&);

extern template int run_with_compute<nk::f32_t, nk_u8_k>(bench_launch<nk::f32_t> const&);
extern template int run_with_compute<nk::bf16_t, nk_u8_k>(bench_launch<nk::bf16_t> const&);
extern template int run_with_compute<nk::i8_t, nk_u8_k>(bench_launch<nk::i8_t> const&);
extern template int run_with_compute<nk::u8_t, nk_u8_k>(bench_launch<nk::u8_t> const&);

extern template int run_with_compute<nk::f32_t, nk_i4_k>(bench_launch<nk::f32_t> const&);
extern template int run_with_compute<nk::bf16_t, nk_i4_k>(bench_launch<nk::bf16_t> const&);
extern template int run_with_compute<nk::i8_t, nk_i4_k>(bench_launch<nk::i8_t> const&);
extern template int run_with_compute<nk::u8_t, nk_i4_k>(bench_launch<nk::u8_t> const&);

/** @brief Stored-dtype axis: the CLI's chosen store instantiates the compute chain, or reports
 *      the dtypes that carry adapted CPU kits. */
template <typename dataset_scalar_type_>
int run_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch) {
    return dispatch_value<nk_bf16_k, nk_i8_k, nk_u8_k, nk_i4_k>(
        launch.options.stored_dtype,
        [&](auto dtype) { return run_with_compute<dataset_scalar_type_, dtype.value>(launch); },
        [&](auto dtype) {
            fmt::println("Stored dtype {} has no adapted kernels; override with --bf16, --i8, --u8, or --i4",
                         dtype_name(dtype));
            return 1;
        });
}

/** @brief Opens the dataset with the file-native scalar and enters the typed cascade. */
template <typename dataset_scalar_type_>
int open_and_run(bench_options_t const& options, fu_pool_t pool, fu_topology_t topology) {
    auto opened = persisted_dataset<dataset_scalar_type_>::try_open(
        options.vectors_path.c_str(), options.queries_path.empty() ? nullptr : options.queries_path.c_str(),
        options.neighbors_path.empty() ? nullptr : options.neighbors_path.c_str());
    if (opened.status != status_t::success_k) {
        fmt::println("Failed to open the dataset: {}", options.vectors_path);
        return 1;
    }
    return run_with_stored_dtype(bench_launch<dataset_scalar_type_> {opened.value, options, pool, topology});
}

int main(int argc, char** argv) {
    bench_options_t options;
    std::string max_compute_name, max_schedule_name, memory_domain_name;

    auto cli = ( //
        (clipp::option("--vectors") & clipp::value("path", options.vectors_path)) % "base vectors, `.fbin` family",
        (clipp::option("--queries") & clipp::value("path", options.queries_path)) % "query vectors, same dtype",
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        (clipp::option("--bf16").set(options.stored_dtype, nk_bf16_k) |
         clipp::option("--i8").set(options.stored_dtype, nk_i8_k) |
         clipp::option("--u8").set(options.stored_dtype, nk_u8_k) |
         clipp::option("--i4").set(options.stored_dtype, nk_i4_k)) %
            "stored dtype override, staging `nk::cast` on ingest",
        (clipp::option("--max-compute") & clipp::value("name", max_compute_name)) %
            "compute ceiling by microarchitecture, like `icelake`, or `serial`; newest detected by default",
        (clipp::option("--max-schedule") & clipp::value("name", max_schedule_name)) %
            "scheduling ceiling by wait instruction, like `tpause`, or `standard`; newest offered by default",
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k)) %
            "metric; angular by default",
        (clipp::option("--threads") & clipp::value("count", options.threads)) %
            "worker threads; 7/8 of the logical cores by default",
        (clipp::option("--memory-domain") & clipp::value("id", memory_domain_name)) %
            "pin memory & threads to one NUMA domain, or `all` for one index per domain",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "staging trampoline rows shared by all threads; 4 × threads by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool; 128 by default, tuned for 99% recall@10",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool; 192 by default, tuned for 99% recall@10",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
        clipp::option("--self-recall").set(options.self_recall, true) % //
            "query every stored vector for itself instead of the query file",
        (clipp::option("--save") & clipp::value("name", options.save_name)) % //
            "checkpoint name to serialize every domain index into after ingestion",
        (clipp::option("--load") & clipp::value("name", options.load_name)) % //
            "checkpoint name to restore domain indexes from instead of building",
        (clipp::option("--checkpoints-dir") & clipp::value("dir", options.checkpoints_dir)) % //
            "directory for checkpoint blobs; /tmp by default",
        (clipp::option("--skip-vectors") & clipp::value("count", options.skip_vectors)) % //
            "rows the loaded indexes already cover",
        (clipp::option("--huge-pages") & clipp::value("count", options.huge_pages)) %
            "huge pages provisioned before launch; enforced, so base-page fallbacks fail fast",
        clipp::option("--help").set(options.help) % "print this message");

    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, "usearch_bench"))
            fmt::println("{}\n{}\n", section.title(), section.content());
        return options.help ? 0 : 1;
    }

    nk_dtype_t const file_dtype = dtype_of_extension(options.vectors_path.c_str());
    if (file_dtype == nk_dtype_unknown_k || file_dtype == nk_u32_k) {
        fmt::println("Unrecognized vectors extension: {}", options.vectors_path);
        return 1;
    }
    if (options.stored_dtype == nk_dtype_unknown_k) options.stored_dtype = file_dtype;

    // Ceiling names resolve into allow-masks - the native capability currency on both
    // sides. A named compute ceiling keeps its table prefix plus serial; "serial" alone
    // keeps just the fallback bit. Capping above the detection is harmless: the dispatch
    // intersects with the runtime bits anyway.
    if (max_compute_name == "serial") { options.max_compute = nk_cap_serial_k; }
    else if (!max_compute_name.empty()) {
        std::size_t const named = index_of_name(compute_span_k, max_compute_name);
        if (named == compute_choices_count_k) {
            fmt::print("Unknown --max-compute {}; this table offers: serial", max_compute_name);
            for (capability_choice<nk_capability_t> const& choice : compute_choices_k) fmt::print(" {}", choice.name);
            fmt::println("");
            return 1;
        }
        options.max_compute = nk_cap_serial_k | prefix_mask(compute_span_k, named);
    }

    // A schedule ceiling caps only the waiter bits; the pool-management bits - topology,
    // placement, huge pages - stay allowed, so `fu_pool_new` keeps its full toolbox. The
    // floor keyword is "standard", after @c fu::standard_yield_t - Arm's own YIELD hint
    // legitimately owns the "yield" row.
    if (max_schedule_name == "standard") { options.max_schedule = schedule_floor_k; }
    else if (!max_schedule_name.empty()) {
        std::size_t const named = index_of_name(schedule_span_k, max_schedule_name);
        if (named == schedule_choices_count_k) {
            fmt::print("Unknown --max-schedule {}; this table offers: standard", max_schedule_name);
            for (capability_choice<fu_capabilities_t> const& choice : schedule_choices_k)
                fmt::print(" {}", choice.name);
            fmt::println("");
            return 1;
        }
        options.max_schedule = static_cast<fu_capabilities_t>(schedule_floor_k | prefix_mask(schedule_span_k, named));
    }

    fu_topology_t topology = fu_topology_new();
    fu_fabric_t fabric = fu_fabric_new();
    fu_pool_t pool = fu_pool_new("usearch_bench", options.max_schedule);
    if (!topology || !pool) {
        fmt::println("Failed to inspect the topology or create the pool");
        return 1;
    }

    // Resolve `--memory-domain` into the engaged-domain list: empty means one machine-wide
    // heap index, a single id pins memory & threads to that domain, and `all` builds one
    // index per memory domain - the list IS the placement, no modes.
    if (memory_domain_name == "all")
        for (std::size_t domain = 0; domain != fu_memory_domains_count(topology); ++domain)
            options.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    else if (!memory_domain_name.empty()) {
        fu_memory_domain_id_t const domain = static_cast<fu_memory_domain_id_t>(
            std::strtol(memory_domain_name.c_str(), nullptr, 10));
        if (domain < 0 || static_cast<std::size_t>(domain) >= fu_memory_domains_count(topology)) {
            fmt::println("Unknown --memory-domain {}; this machine has {} memory domains", memory_domain_name,
                         fu_memory_domains_count(topology));
            return 1;
        }
        options.memory_domains.push_back(domain);
    }

    // Default to 7/8 of the available logical cores: at full occupancy the strictly-ordered
    // publication convoy serializes on scheduler latency, and throughput collapses by an order
    // of magnitude. A single pinned domain carves both the core budget and the spawn from the
    // compute domain whose local memory it is.
    bool spawned = false;
    std::size_t threads = options.threads;
    if (options.memory_domains.size() == 1) {
        std::size_t const compute_domains = fu_compute_domains_count(topology);
        for (std::size_t compute_domain = 0; compute_domain != compute_domains && !spawned; ++compute_domain) {
            if (fu_local_memory_of(topology, compute_domain) !=
                static_cast<std::size_t>(options.memory_domains.front()))
                continue;
            if (!threads)
                threads = std::max<std::size_t>(1, fu_logical_cores_count_in(topology, compute_domain) * 7 / 8);
            spawned = fu_pool_spawn_on(topology, pool, compute_domain, threads, fu_caller_inclusive_k);
        }
    }
    else {
        if (!threads) threads = std::max<std::size_t>(1, fu_logical_cores_count(topology) * 7 / 8);
        spawned = fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k);
    }
    if (!spawned) {
        fmt::println("Failed to spawn {} threads", threads);
        return 1;
    }
    print_machine(topology, fabric, options, threads);

    // Huge pages are provisioned outside the process - see the header's shell snippet - and
    // `--huge-pages` declares the expectation, enforced against what the domains report.
    if (options.huge_pages) {
        std::size_t volume = 0;
        for (std::size_t domain = 0; domain != fu_memory_domains_count(topology); ++domain)
            volume += fu_volume_huge_pages_in(topology, domain);
        fmt::println("  huge pages expected: {}, available: {:.2f} GB", options.huge_pages, double(volume) / 1e9);
        if (!volume) {
            fmt::println("--huge-pages expects {}, but the domains report none", options.huge_pages);
            return 1;
        }
    }

    // File-scalar axis: the newest hop of the compile-time cascade, lowering the file's dtype
    // to the C++ wrapper the whole run is instantiated over through NumKong's own `type_for`.
    int const exit_code = dispatch_value<nk_f32_k, nk_bf16_k, nk_i8_k, nk_u8_k>(
        file_dtype,
        [&](auto dtype) { return open_and_run<typename nk::type_for<dtype.value>::type>(options, pool, topology); },
        [&](auto dtype) {
            fmt::println("Unsupported dataset dtype {}", dtype_name(dtype));
            return 1;
        });

    fu_pool_delete(pool);
    fu_fabric_delete(fabric);
    fu_topology_delete(topology);
    return exit_code;
}
