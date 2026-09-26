/**
 *  @file bench/dense_ram_index.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The CLI entry of the single-index CPU benchmark: option parsing, the pool & topology
 *      setup, and the dataset-scalar hop into the per-dtype shards.
 *
 *  The compile-time cascade lives in @c dense/ram_index.hpp; this translation unit stays
 *  dtype-free by extern-declaring the @c run_with_stored_dtype instantiations the per-dtype
 *  shards - one per stored dtype: @c f32, @c bf16, @c f16, @c i8, @c i4, @c u8, and @c u1 -
 *  carry, so the heavy kernel grids build in parallel and this unit recompiles in seconds.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;
using namespace unum::usearch::bench;
using namespace unum::usearch::bench::dense;
using namespace unum::usearch::bench::utilities;
namespace us = unum::usearch;

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_f32_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_f32_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_f32_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_f32_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_f32_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_bf16_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_bf16_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_bf16_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_bf16_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_bf16_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_f16_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_f16_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_f16_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_f16_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_f16_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e5m2_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e5m2_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e5m2_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e5m2_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e5m2_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e4m3_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e4m3_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e4m3_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e4m3_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e4m3_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e3m2_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e3m2_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e3m2_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e3m2_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e3m2_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e2m3_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e2m3_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e2m3_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e2m3_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e2m3_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_i8_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_i8_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_i8_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_i8_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_i8_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_i4_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_i4_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_i4_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_i4_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_i4_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_u8_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_u8_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_u8_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_u8_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_u8_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_u4_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_u4_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_u4_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_u4_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_u4_k>(bench_launch<nk::u8_t> const&);

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_u1_k>(bench_launch<nk::f32_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_u1_k>(bench_launch<nk::bf16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_u1_k>(bench_launch<nk::f16_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_u1_k>(bench_launch<nk::i8_t> const&);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_u1_k>(bench_launch<nk::u8_t> const&);

/** Opens the dataset with the file-native scalar and enters the typed cascade. */
template <typename dataset_scalar_type_>
int open_and_run(bench_options_t const& options, fu_pool_t pool, fu_topology_t topology) {
    auto opened = persisted_dataset<dataset_scalar_type_>::open(options.vectors_path, options.queries_path,
                                                                options.neighbors_path);
    if (opened.status != status_t::success_k) {
        fmt::println("Failed to open the dataset: {}", options.vectors_path);
        return 1;
    }
    return dispatch_stored_dtype(
        options.stored_dtype,
        [&](auto dtype) {
            return run_with_stored_dtype<dataset_scalar_type_, dtype.value>(
                bench_launch<dataset_scalar_type_> {opened.value, options, pool, topology});
        },
        [&](auto dtype) {
            fmt::println("Stored dtype {} has no adapted kernels; override with {}", dtype_name(dtype),
                         "--f32, --bf16, --f16, --e5m2, --e4m3, --e3m2, --e2m3, --i8, --i4, --u8, --u4, or --u1");
            return 1;
        });
}

int main(int argc, char** argv) {
    bench_options_t options;
    std::string max_compute_name, max_schedule_name, memory_domain_name;

    auto cli = ( //
        (clipp::option("--vectors") & clipp::value("path", options.vectors_path)) % "base vectors, `.fbin` family",
        (clipp::option("--queries") & clipp::value("path", options.queries_path)) % "query vectors, same dtype",
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        (clipp::option("--f32").set(options.stored_dtype, nk_f32_k) |
         clipp::option("--bf16").set(options.stored_dtype, nk_bf16_k) |
         clipp::option("--f16").set(options.stored_dtype, nk_f16_k) |
         clipp::option("--e5m2").set(options.stored_dtype, nk_e5m2_k) |
         clipp::option("--e4m3").set(options.stored_dtype, nk_e4m3_k) |
         clipp::option("--e3m2").set(options.stored_dtype, nk_e3m2_k) |
         clipp::option("--e2m3").set(options.stored_dtype, nk_e2m3_k) |
         clipp::option("--i8").set(options.stored_dtype, nk_i8_k) |
         clipp::option("--i4").set(options.stored_dtype, nk_i4_k) |
         clipp::option("--u8").set(options.stored_dtype, nk_u8_k) |
         clipp::option("--u4").set(options.stored_dtype, nk_u4_k) |
         clipp::option("--u1").set(options.stored_dtype, nk_u1_k)) %
            "stored dtype override, staging `nk::cast` on ingest",
        (clipp::option("--max-compute") & clipp::value("name", max_compute_name)) %
            "compute ceiling by microarchitecture, like `icelake`, or `serial`; newest detected by default",
        (clipp::option("--max-schedule") & clipp::value("name", max_schedule_name)) %
            "scheduling ceiling by wait instruction, like `tpause`, or `standard`; newest offered by default",
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k) |
         clipp::option("--hamming").set(options.kernel, nk_kernel_hamming_k) |
         clipp::option("--jaccard").set(options.kernel, nk_kernel_jaccard_k)) %
            "metric; angular by default, hamming and jaccard for --u1",
        (clipp::option("--threads") & clipp::value("count", options.threads)) %
            "threads; 7/8 of the logical cores by default",
        (clipp::option("--memory-domain") & clipp::value("id", memory_domain_name)) %
            "pin memory & threads to one NUMA domain; the heap otherwise",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "staging trampoline rows shared by all threads; 4 × threads by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool; 256 by default",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool; 96 by default",
        (clipp::option("--max-refines") & clipp::value("passes", options.max_refines)) %
            "upper bound on whole-graph refinement passes after ingestion; 0 by default",
        (clipp::option("--expansion-refine") & clipp::value("breadth", options.expansion_refine)) %
            "refinement candidate pool; inherits --expansion-add by default",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
        (clipp::option("--first-vector") & clipp::value("row", options.first_vector)) %
            "first dataset row to view; slices one file across independent runs",
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
        for (auto const& section : clipp::make_man_page(cli, "usearch_bench_dense_ram_index"))
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
        std::size_t const named = index_of_name(compute_choices_k, max_compute_name);
        if (named == std::size(compute_choices_k)) {
            fmt::println("Unknown --max-compute {}; this table offers: {}", max_compute_name,
                         fmt::join(named_choices("serial", compute_choices_k, //
                                                 prefix_mask(compute_choices_k, std::size(compute_choices_k))),
                                   " "));
            return 1;
        }
        options.max_compute = nk_cap_serial_k | prefix_mask(compute_choices_k, named);
    }

    // A schedule ceiling caps only the waiter bits; the pool-management bits - topology,
    // placement, huge pages - stay allowed, so `fu_pool_new` keeps its full toolbox. The
    // floor keyword is "standard", after @c fu::standard_yield_t - Arm's own YIELD hint
    // legitimately owns the "yield" row.
    if (max_schedule_name == "standard") { options.max_schedule = schedule_floor_k; }
    else if (!max_schedule_name.empty()) {
        std::size_t const named = index_of_name(schedule_choices_k, max_schedule_name);
        if (named == std::size(schedule_choices_k)) {
            fmt::println("Unknown --max-schedule {}; this table offers: {}", max_schedule_name,
                         fmt::join(named_choices("standard", schedule_choices_k, //
                                                 prefix_mask(schedule_choices_k, std::size(schedule_choices_k))),
                                   " "));
            return 1;
        }
        options.max_schedule = static_cast<fu_capabilities_t>(schedule_floor_k |
                                                              prefix_mask(schedule_choices_k, named));
    }

    fu_topology_t topology = nullptr;
    fu_fabric_t fabric = nullptr;
    fu_pool_t pool = nullptr;
    if (fu_topology_new(&topology) != fu_success_k || fu_fabric_new(&fabric) != fu_success_k ||
        fu_pool_new("usearch_bench_dense_ram_index", options.max_schedule, &pool) != fu_success_k) {
        fmt::println("Failed to inspect the topology or create the pool");
        return 1;
    }

    std::size_t memory_domains = 0;
    if (fu_memory_domains_count(topology, &memory_domains) != fu_success_k) {
        fmt::println("Failed to count the memory domains");
        return 1;
    }

    // Resolve `--memory-domain`: empty means one machine-wide heap index, and a single ID
    // pins memory & threads to that domain. This benchmark measures exactly one index - the
    // multi-domain arrangement is `usearch_bench_dense_ram_indexes`, driving the sharded index.
    if (!memory_domain_name.empty()) {
        fu_memory_domain_id_t const domain = static_cast<fu_memory_domain_id_t>(
            std::strtol(memory_domain_name.c_str(), nullptr, 10));
        if (domain < 0 || static_cast<std::size_t>(domain) >= memory_domains) {
            fmt::println("Unknown --memory-domain {}; this machine has {} memory domains", memory_domain_name,
                         memory_domains);
            return 1;
        }
        options.memory_domains.push_back(domain);
    }

    // Default to 7/8 of the available logical cores: at full occupancy the strictly-ordered
    // publication convoy serializes on scheduler latency, and throughput collapses by an order
    // of magnitude. A single pinned domain carves the core budget from every compute domain
    // whose local memory it is - several clusters can share one NUMA node.
    bool spawned = false;
    std::size_t threads = options.threads;
    if (options.memory_domains.size() == 1) {
        std::size_t compute_domains = 0;
        if (fu_compute_domains_count(topology, &compute_domains) != fu_success_k) {
            fmt::println("Failed to count the compute domains");
            return 1;
        }
        if (!threads) {
            std::size_t local_cores = 0;
            for (std::size_t compute_domain = 0; compute_domain != compute_domains; ++compute_domain) {
                std::size_t local_memory = 0, domain_cores = 0;
                if (fu_local_memory_of(topology, compute_domain, &local_memory) != fu_success_k) continue;
                if (local_memory != static_cast<std::size_t>(options.memory_domains.front())) continue;
                if (fu_logical_cores_count_in(topology, compute_domain, &domain_cores) != fu_success_k) continue;
                local_cores += domain_cores;
            }
            threads = std::max<std::size_t>(1, local_cores * 7 / 8);
        }
        spawned = fu_pool_spawn_near_memory_domain(topology, pool, options.memory_domains.front(), threads,
                                                   fu_caller_inclusive_k) == fu_success_k;
    }
    else {
        if (!threads) {
            std::size_t logical_cores = 0;
            if (fu_logical_cores_count(topology, &logical_cores) != fu_success_k) {
                fmt::println("Failed to count the logical cores");
                return 1;
            }
            threads = std::max<std::size_t>(1, logical_cores * 7 / 8);
        }
        spawned = fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k) == fu_success_k;
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
        for (std::size_t domain = 0; domain != memory_domains; ++domain) {
            std::size_t domain_volume = 0;
            if (fu_volume_huge_pages_in(topology, domain, &domain_volume) != fu_success_k) continue;
            volume += domain_volume;
        }
        fmt::println("  huge pages expected: {}, available: {:.2f} GB", options.huge_pages, double(volume) / 1e9);
        if (!volume) {
            fmt::println("--huge-pages expects {}, but the domains report none", options.huge_pages);
            return 1;
        }
    }

    // File-scalar axis: the newest hop of the compile-time cascade, lowering the file's dtype
    // to the C++ wrapper the whole run is instantiated over through NumKong's own `type_for`.
    int const exit_code = dispatch_file_dtype(
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
