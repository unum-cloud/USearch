/**
 *  @file bench/dense_vram_index.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The CLI entry of the CUDA dense-index benchmark: option parsing, the device probe, and
 *      the dataset-scalar & stored-dtype hops into the per-dtype shards.
 *
 *  The CUDA pipeline and its compile-time cascade live in @c dense/vram_index.cuh; this
 *  translation unit stays dtype-free by extern-declaring the @c run_with_stored_dtype
 *  instantiations the per-dtype shards carry, so the heavy device instantiation grids
 *  build in parallel and this unit recompiles in seconds.
 */
#include <string> // `std::string` for the CLI collectors

#include <clipp.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include "dense/vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::bench;
using namespace unum::usearch::bench::dense;
using namespace unum::usearch::bench::utilities;
namespace us = unum::usearch;

extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f32_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::bf16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::f16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::i8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
extern template int us::bench::dense::run_with_stored_dtype<nk::u1x8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u1x8_t> const&, bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);

/** Opens the dataset with the file-native scalar and lowers the stored dtype onto
 *  the shard-carried cascade - or reports the dtypes that carry device kits. */
template <typename dataset_scalar_type_>
static int open_and_run(bench_options_t const& options, int device, cuda_arch_t max_compute,
                        cuda_address_space_t plane_space) {
    auto opened = persisted_dataset<dataset_scalar_type_>::open(options.vectors_path, options.queries_path,
                                                                options.neighbors_path);
    if (opened.status != status_t::success_k) {
        fmt::println("Failed to open the dataset: {}", options.vectors_path);
        return 1;
    }
    return dispatch_stored_value<dataset_scalar_type_, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k, nk_e2m3_k,
                                 nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(
        options.stored_dtype,
        [&](auto dtype) {
            return run_with_stored_dtype<dataset_scalar_type_, dtype.value>(opened.value, options, device, max_compute,
                                                                            plane_space);
        },
        [&](auto dtype) {
            fmt::println("Stored dtype {} has no device kits; " //
                         "override with --bf16, --f16, --e5m2, --e4m3, --e3m2, --e2m3, --i8, --i4, --u8, --u4, or --u1",
                         dtype_name(dtype));
            return 1;
        });
}

int main(int argc, char** argv) {
    bench_options_t options;
    int device = 0;
    std::string devices_list;
    cuda_address_space_t plane_space = cuda_address_space_t::device_k;
    std::string max_compute_name;

    auto cli = ( //
        (clipp::option("--vectors") & clipp::value("path", options.vectors_path)) % "base vectors, `.fbin` family",
        (clipp::option("--queries") & clipp::value("path", options.queries_path)) % "query vectors, same dtype",
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        (clipp::option("--device") & clipp::value("id", device)) % "CUDA device ordinal; 0 by default",
        (clipp::option("--devices") & clipp::value("list", devices_list)) %
            "comma-separated ordinals for the multi-index pipeline: one shard per device",
        clipp::option("--merge").set(options.merge, true) % //
            "stitch the shards into ONE graph after ingest; single-walk queries",
        (clipp::option("--expansion-merge") & clipp::value("breadth", options.expansion_merge)) %
            "merge candidate pool; inherits --expansion-add by default",
        (clipp::option("--max-refines") & clipp::value("passes", options.max_refines)) %
            "upper bound on whole-graph refinement passes after ingestion; 0 by default",
        (clipp::option("--expansion-refine") & clipp::value("breadth", options.expansion_refine)) %
            "refinement candidate pool; twice the connectivity by default",
        (clipp::option("--max-compute") & clipp::value("name", max_compute_name)) %
            "compute ceiling by GPU generation, like `hopper`, or `ampere`; newest by default",
        clipp::option("--managed-memory").set(plane_space, cuda_address_space_t::managed_k) % //
            "managed-memory planes: oversubscribe VRAM through the pager, host-RAM ceiling",
        (clipp::option("--bf16").set(options.stored_dtype, nk_bf16_k) |
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
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k) |
         clipp::option("--hamming").set(options.kernel, nk_kernel_hamming_k) |
         clipp::option("--jaccard").set(options.kernel, nk_kernel_jaccard_k)) %
            "metric; angular by default, hamming and jaccard for --u1",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "rows & queries per launch; 65536 by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool; 256 by default",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool; 96 by default",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
        clipp::option("--self-recall").set(options.self_recall, true) % //
            "query every stored vector for itself instead of the query file",
        (clipp::option("--save") & clipp::value("name", options.save_name)) % //
            "checkpoint name to serialize the index into after ingestion",
        (clipp::option("--load") & clipp::value("name", options.load_name)) % //
            "checkpoint name to restore the index from instead of building",
        (clipp::option("--checkpoints-dir") & clipp::value("dir", options.checkpoints_dir)) % //
            "directory for checkpoint blobs; /tmp by default",
        (clipp::option("--skip-vectors") & clipp::value("count", options.skip_vectors)) % //
            "rows the loaded index already covers",
        clipp::option("--help").set(options.help) % "print this message");

    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, "usearch_bench_dense_vram_index"))
            fmt::println("{}\n{}\n", section.title(), section.content());
        return options.help ? 0 : 1;
    }

    for (std::size_t begin = 0; begin < devices_list.size();) {
        std::size_t end = devices_list.find(',', begin);
        if (end == std::string::npos) end = devices_list.size();
        options.devices.push_back(std::stoi(devices_list.substr(begin, end - begin)));
        begin = end + 1;
    }
    int devices_count = 0;
    if (cudaGetDeviceCount(&devices_count) != cudaSuccess || device >= devices_count) {
        fmt::println("CUDA device {} unreachable; {} devices visible", device, devices_count);
        return 1;
    }
    cudaSetDevice(device);
    cudaDeviceProp properties = {};
    cudaGetDeviceProperties(&properties, device);
    fmt::print(header_style(), "Machine\n");
    fmt::println("  device {}: {} · {} SMs · {:.0f} GB", device, properties.name, properties.multiProcessorCount,
                 double(properties.totalGlobalMem) / 1e9);
    nk_dtype_t const file_dtype = dtype_of_extension(options.vectors_path.c_str());
    if (options.stored_dtype == nk_dtype_unknown_k) options.stored_dtype = file_dtype;

    // A named ceiling caps which kit-chain rows the cascade may select; "ampere" keeps just the floor.
    // Capping above what this binary compiled is harmless: a row also needs its tier compiled in.
    cuda_arch_t max_compute = cuda_compute_choices_k[std::size(cuda_compute_choices_k) - 1].bit;
    if (max_compute_name == "ampere") max_compute = cuda_arch_ampere_k;
    else if (!max_compute_name.empty()) {
        std::size_t const named = index_of_name(cuda_compute_choices_k, max_compute_name);
        if (named == std::size(cuda_compute_choices_k)) {
            fmt::println(
                "Unknown --max-compute {}; this table offers: ampere {}", max_compute_name,
                fmt::join(cuda_compute_choices_k | std::views::transform(&capability_choice<cuda_arch_t>::name), " "));
            return 1;
        }
        max_compute = cuda_compute_choices_k[named].bit;
    }
    fmt::println("  compute ceiling: {}", bit_name(cuda_compute_choices_k, max_compute, "ampere"));
    fmt::println("  kernels requested: {} · {}", dtype_name(options.stored_dtype), nk_kernel_name(options.kernel));

    // File-scalar axis: lower the file's dtype to the C++ wrapper the run is instantiated over,
    // through NumKong's own `type_for` - the same paradigm the typed cascade rides.
    return dispatch_file_dtype(
        file_dtype,
        [&](auto dtype) {
            return open_and_run<typename nk::type_for<dtype.value>::type>(options, device, max_compute, plane_space);
        },
        [&](auto dtype) {
            fmt::println("Unsupported dataset dtype {} for the CUDA kits", dtype_name(dtype));
            return 1;
        });
}
