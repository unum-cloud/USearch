/**
 *  @file bench/dense_vram_index.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The entry of the CUDA retrieval benchmark - build, refine, and search: the shared process
 *      setup, and the stored-dtype hop into the per-dtype shards.
 *
 *  The CUDA pipeline, its compile-time cascade, and the process setup both benchmark families
 *  share live in @c dense/vram_index.cuh; this translation unit stays dtype-free by
 *  extern-declaring the @c run_with_stored_dtype instantiations the per-dtype shards carry, so
 *  the heavy device instantiation grids build in parallel and this unit recompiles in seconds.
 */
#include <fmt/format.h>

#include "dense/vram_index.cuh"

namespace unum::usearch::bench {

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
extern template int dense::run_with_stored_dtype<nk::u1x8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u1x8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);

/** Runs the CUDA retrieval benchmark over the flags in @p argv. */
int bench_dense_vram_index(int argc, char const* const* argv) {
    auto const on_dataset = []<typename dataset_scalar_type_>(
                                utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                                settings_t const& settings, int device, cuda_arch_t max_compute,
                                cuda_address_space_t plane_space) {
        return dispatch_stored_value<dataset_scalar_type_, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                     nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(
            settings.stored_dtype,
            [&](auto dtype) {
                return dense::run_with_stored_dtype<dataset_scalar_type_, dtype.value>(dataset, settings, device,
                                                                                       max_compute, plane_space);
            },
            [&](auto dtype) {
                fmt::println(stderr, "--stored-dtype=\"{}\" has no device kits for this file", dtype_name(dtype));
                return 1;
            });
    };
    return dense::run_cuda_bench(argc, argv, "usearch_bench_dense_vram_index", on_dataset);
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_vram_index(argc, argv); }
