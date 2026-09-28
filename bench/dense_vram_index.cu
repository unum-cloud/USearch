/**
 *  @file bench/dense_vram_index.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The CLI entry of the CUDA retrieval benchmark - build, refine, and search: the metric and
 *      ground-truth options, and the stored-dtype hop into the per-dtype shards.
 *
 *  The CUDA pipeline, its compile-time cascade, and the process setup both benchmark families
 *  share live in @c dense/vram_index.cuh; this translation unit stays dtype-free by
 *  extern-declaring the @c run_with_stored_dtype instantiations the per-dtype shards carry, so
 *  the heavy device instantiation grids build in parallel and this unit recompiles in seconds.
 */
#include <clipp.h>
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

int main(int argc, char** argv) {
    bench_options_t options;
    cuda_cli_t device_cli;
    auto const cli = ( //
        shared_cuda_cli(options, device_cli),
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k) |
         clipp::option("--hamming").set(options.kernel, nk_kernel_hamming_k) |
         clipp::option("--jaccard").set(options.kernel, nk_kernel_jaccard_k)) %
            "metric; angular by default, hamming and jaccard for --u1",
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        clipp::option("--help").set(options.help) % "print this message");

    auto const on_dataset = [&]<typename dataset_scalar_type_>(persisted_dataset<dataset_scalar_type_> const& dataset,
                                                               int device, cuda_arch_t max_compute,
                                                               cuda_address_space_t plane_space) {
        return dispatch_stored_value<dataset_scalar_type_, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                     nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(
            options.stored_dtype,
            [&](auto dtype) {
                return run_with_stored_dtype<dataset_scalar_type_, dtype.value>(dataset, options, device, max_compute,
                                                                                plane_space);
            },
            [&](auto dtype) {
                fmt::println("Stored dtype {} has no device kits; override with {}", dtype_name(dtype),
                             "--bf16, --f16, --e5m2, --e4m3, --e3m2, --e2m3, --i8, --i4, --u8, --u4, or --u1");
                return 1;
            });
    };
    return run_cuda_bench(argc, argv, cli, "usearch_bench_dense_vram_index", options, device_cli, on_dataset);
}
