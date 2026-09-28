/**
 *  @file bench/dense_ram_index.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The CLI entry of the single-index CPU retrieval benchmark: the metric and ground-truth
 *      flags over the shared process setup, and the stored-dtype hop into the per-dtype shards.
 *
 *  The compile-time cascade and the process setup live in @c dense/ram_index.hpp; this
 *  translation unit stays dtype-free by extern-declaring the @c run_with_stored_dtype
 *  instantiations the per-dtype shards - one per stored dtype: @c f32, @c bf16, @c f16, @c i8,
 *  @c i4, @c u8, and @c u1 - carry, so the heavy kernel grids build in parallel and this unit
 *  recompiles in seconds.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;
using namespace unum::usearch::bench;
using namespace unum::usearch::bench::dense;
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
extern template int us::bench::dense::run_with_stored_dtype<nk::u1x8_t, nk_u1_k>(bench_launch<nk::u1x8_t> const&);

int main(int argc, char** argv) {
    bench_options_t options;
    bench_names_t names;
    auto const cli = ( //
        common_cli(options, names),
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k) |
         clipp::option("--hamming").set(options.kernel, nk_kernel_hamming_k) |
         clipp::option("--jaccard").set(options.kernel, nk_kernel_jaccard_k)) %
            "metric; angular by default, hamming and jaccard for --u1",
        clipp::option("--help").set(options.help) % "print this message");

    return bench_main(
        argc, argv, "usearch_bench_dense_ram_index", cli, options, names,
        [&]<typename dataset_scalar_type_>(bench_launch<dataset_scalar_type_> const& launch) {
            return dispatch_stored_dtype<dataset_scalar_type_>(
                options.stored_dtype,
                [&](auto dtype) { return run_with_stored_dtype<dataset_scalar_type_, dtype.value>(launch); },
                [&](auto dtype) {
                    fmt::println(
                        "Stored dtype {} has no adapted kernels; override with {}", dtype_name(dtype),
                        "--f32, --bf16, --f16, --e5m2, --e4m3, --e3m2, --e2m3, --i8, --i4, --u8, --u4, or --u1");
                    return 1;
                });
        });
}
