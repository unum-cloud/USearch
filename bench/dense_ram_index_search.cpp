/**
 *  @file bench/dense_ram_index_search.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The entry of the single-index CPU search benchmark: the shared process setup, and the
 *      stored-dtype hop into the per-dtype shards.
 *
 *  The compile-time cascade lives in @c dense/ram_index_search.hpp and the process setup in @c dense.cuh;
 *  this translation unit stays dtype-free by extern-declaring the @c run_with_stored_dtype
 *  instantiations the per-dtype shards - one per stored dtype: @c f32, @c bf16, @c f16, @c e5m2,
 *  @c e4m3, @c e3m2, @c e2m3, @c i8, @c i4, @c u8, @c u4, and @c u1 - carry, so the heavy kernel
 *  grids build in parallel and this unit recompiles in seconds.
 */
#include "dense/ram_index_search.hpp"

namespace unum::usearch::bench {

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_f32_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_f32_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_f32_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_f32_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_f32_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_bf16_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_bf16_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_bf16_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_bf16_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_bf16_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_f16_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_f16_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_f16_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_f16_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_f16_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e5m2_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e5m2_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e5m2_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e5m2_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e5m2_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e4m3_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e4m3_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e4m3_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e4m3_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e4m3_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e3m2_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e3m2_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e3m2_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e3m2_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e3m2_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e2m3_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e2m3_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e2m3_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e2m3_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e2m3_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_i8_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_i8_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_i8_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_i8_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_i8_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_i4_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_i4_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_i4_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_i4_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_i4_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u8_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u8_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u8_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u8_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u8_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u4_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u4_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u4_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u4_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u4_k>(bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u1_k>(bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u1_k>(bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u1_k>(bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u1_k>(bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u1_k>(bench_launch<nk::u8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u1x8_t, nk_u1_k>(bench_launch<nk::u1x8_t> const&);

/** Runs the single-index search benchmark over the flags in @p argv. */
int bench_dense_ram_index_search(int argc, char const* const* argv) {
    return run_cpu_bench(
        argc, argv, "usearch_bench_dense_ram_index_search",
        [&]<typename dataset_scalar_type_>(bench_launch<dataset_scalar_type_> const& launch) {
            return dispatch_stored_dtype<dataset_scalar_type_>(
                launch.settings.stored_dtype,
                [&](auto dtype) { return dense::run_with_stored_dtype<dataset_scalar_type_, dtype.value>(launch); },
                [&](auto dtype) {
                    fmt::println(stderr, "--stored-dtype=\"{}\" has no adapted kernels for this file",
                                 dtype_name(dtype));
                    return 1;
                });
        });
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_ram_index_search(argc, argv); }
