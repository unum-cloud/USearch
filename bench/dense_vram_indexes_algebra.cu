/**
 *  @file bench/dense_vram_indexes_algebra.cu
 *  @author Ash Vardanian
 *  @date October 9, 2026
 *  @brief The entry of the multi-device CUDA algebra benchmark: the algebra flags and the device group over the
 *      shared process setup, and the stored-dtype hop into the per-dtype shards.
 *
 *  The workload lives in @c dense/vram_indexes_algebra.cuh, over the cascade of @c dense/vram_indexes_search.cuh and
 *  the process setup of @c dense/vram_index_search.cuh; this translation unit stays dtype-free by extern-declaring
 *  the @c run_indexes_algebra_with_stored_dtype instantiations the per-dtype host shards carry, so
 *  they build in parallel. Its NVCC device-only pass compiles the binary's fatbin, with the attends
 *  over the keyed value lookup.
 */
#include <fmt/format.h>

#include "dense/vram_indexes_algebra.cuh"

template __device__ void us::bench::dense::instantiate_cuda_kernels<us::dense::cuda_keyed_lookup<
    us::utilities::u32_t, us::bench::dense::attention_value_lookup<uint4>>::device_lookup_k>();

namespace unum::usearch::bench {

extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
extern template int dense::run_indexes_algebra_with_stored_dtype<nk::u1x8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u1x8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);

/** Runs the multi-device CUDA algebra benchmark over the flags in @p argv. */
int bench_dense_vram_indexes_algebra(int argc, char const* const* argv) {
    algebra_settings_t algebra;
    auto const on_dataset = [&]<typename dataset_scalar_type_>(
                                utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                                settings_t const& settings, std::span<int const> devices, nk_capability_t max_compute,
                                cuda_address_space_t storage_space) {
        if (storage_space == cuda_address_space_t::managed_k)
            return fmt::println(stderr,
                                "--managed-memory pages one index's flat matrices; every shard here stays in VRAM"),
                   1;
        if (!resolve_algebra_settings(settings)) return 1;
        print(algebra);
        return dispatch_stored_value<dataset_scalar_type_, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                     nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(
            settings.stored_dtype,
            [&](auto dtype) {
                return dense::run_indexes_algebra_with_stored_dtype<dataset_scalar_type_, dtype.value>(
                    dataset, settings, algebra, devices, max_compute);
            },
            [&](auto dtype) {
                fmt::println(stderr, "--stored-dtype=\"{}\" has no algebra shard, expected one of {}",
                             dtype_name(dtype), "bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, i4, u8, u4, u1");
                return 1;
            });
    };
    return dense::run_cuda_bench(argc, argv, "usearch_bench_dense_vram_indexes_algebra", on_dataset,
                                 [&](settings_t& settings) { return algebra_flags(settings, algebra); });
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_vram_indexes_algebra(argc, argv); }
