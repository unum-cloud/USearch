/**
 *  @file bench/dense_nvme_indexes_search.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The entry of the sharded-index NVMe search benchmark: the search flags over the NVMe
 *      process setup, and the hop into the per-dtype shard the parts' stored dtype names.
 *
 *  The workload, the cascade and the process setup live in @c dense/nvme_indexes_search.hpp; this
 *  translation unit stays dtype-free by extern-declaring the @c run_nvme_with_stored_dtype
 *  instantiations the per-dtype shards - one per stored dtype: @c bf16, @c f16, @c e5m2, @c e4m3,
 *  @c e3m2, @c e2m3, @c i8, @c i4, @c u8, @c u4, and @c u1 - carry, so the heavy kernel grids build
 *  in parallel and this unit recompiles in seconds.
 */
#include "dense/nvme_indexes_search.hpp"

namespace unum::usearch::bench {

using dense::nvme_settings_t;

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_bf16_k>(bench_launch<nk::f32_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_bf16_k>(bench_launch<nk::bf16_t> const&,
                                                                             nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_bf16_k>(bench_launch<nk::f16_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_bf16_k>(bench_launch<nk::i8_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_bf16_k>(bench_launch<nk::u8_t> const&,
                                                                           nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_f16_k>(bench_launch<nk::f32_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_f16_k>(bench_launch<nk::bf16_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_f16_k>(bench_launch<nk::f16_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_f16_k>(bench_launch<nk::i8_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_f16_k>(bench_launch<nk::u8_t> const&,
                                                                          nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_e5m2_k>(bench_launch<nk::f32_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_e5m2_k>(bench_launch<nk::bf16_t> const&,
                                                                             nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_e5m2_k>(bench_launch<nk::f16_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_e5m2_k>(bench_launch<nk::i8_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_e5m2_k>(bench_launch<nk::u8_t> const&,
                                                                           nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_e4m3_k>(bench_launch<nk::f32_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_e4m3_k>(bench_launch<nk::bf16_t> const&,
                                                                             nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_e4m3_k>(bench_launch<nk::f16_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_e4m3_k>(bench_launch<nk::i8_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_e4m3_k>(bench_launch<nk::u8_t> const&,
                                                                           nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_e3m2_k>(bench_launch<nk::f32_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_e3m2_k>(bench_launch<nk::bf16_t> const&,
                                                                             nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_e3m2_k>(bench_launch<nk::f16_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_e3m2_k>(bench_launch<nk::i8_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_e3m2_k>(bench_launch<nk::u8_t> const&,
                                                                           nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_e2m3_k>(bench_launch<nk::f32_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_e2m3_k>(bench_launch<nk::bf16_t> const&,
                                                                             nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_e2m3_k>(bench_launch<nk::f16_t> const&,
                                                                            nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_e2m3_k>(bench_launch<nk::i8_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_e2m3_k>(bench_launch<nk::u8_t> const&,
                                                                           nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_i8_k>(bench_launch<nk::f32_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_i8_k>(bench_launch<nk::bf16_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_i8_k>(bench_launch<nk::f16_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_i8_k>(bench_launch<nk::i8_t> const&,
                                                                         nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_i8_k>(bench_launch<nk::u8_t> const&,
                                                                         nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_i4_k>(bench_launch<nk::f32_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_i4_k>(bench_launch<nk::bf16_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_i4_k>(bench_launch<nk::f16_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_i4_k>(bench_launch<nk::i8_t> const&,
                                                                         nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_i4_k>(bench_launch<nk::u8_t> const&,
                                                                         nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_u8_k>(bench_launch<nk::f32_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_u8_k>(bench_launch<nk::bf16_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_u8_k>(bench_launch<nk::f16_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_u8_k>(bench_launch<nk::i8_t> const&,
                                                                         nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_u8_k>(bench_launch<nk::u8_t> const&,
                                                                         nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_u4_k>(bench_launch<nk::f32_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_u4_k>(bench_launch<nk::bf16_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_u4_k>(bench_launch<nk::f16_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_u4_k>(bench_launch<nk::i8_t> const&,
                                                                         nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_u4_k>(bench_launch<nk::u8_t> const&,
                                                                         nvme_settings_t const&);

extern template int dense::run_nvme_with_stored_dtype<nk::f32_t, nk_u1_k>(bench_launch<nk::f32_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::bf16_t, nk_u1_k>(bench_launch<nk::bf16_t> const&,
                                                                           nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::f16_t, nk_u1_k>(bench_launch<nk::f16_t> const&,
                                                                          nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::i8_t, nk_u1_k>(bench_launch<nk::i8_t> const&,
                                                                         nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u8_t, nk_u1_k>(bench_launch<nk::u8_t> const&,
                                                                         nvme_settings_t const&);
extern template int dense::run_nvme_with_stored_dtype<nk::u1x8_t, nk_u1_k>(bench_launch<nk::u1x8_t> const&,
                                                                           nvme_settings_t const&);

/** Runs the sharded-index NVMe search benchmark over the flags in @p argv. */
int bench_dense_nvme_indexes_search(int argc, char const* const* argv) {
    nvme_settings_t nvme;
    return dense::run_nvme_bench(
        argc, argv, "usearch_bench_dense_nvme_indexes_search", nvme,
        [&]<typename dataset_scalar_type_>(bench_launch<dataset_scalar_type_> const& launch) {
            if (launch.settings.neighbors_path.empty())
                return fmt::println(stderr, "--neighbors is required, see --help"), 1;
            return dispatch_stored_value<dataset_scalar_type_, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                         nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(
                launch.settings.stored_dtype,
                [&](auto dtype) {
                    return dense::run_nvme_with_stored_dtype<dataset_scalar_type_, dtype.value>(launch, nvme);
                },
                [&](auto dtype) {
                    fmt::println(stderr, "No shard views {} parts from {} queries; parts store one of {}",
                                 dtype_name(dtype), dataset_scalar_type_::dtype_name(),
                                 "bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, i4, u8, u4 or u1");
                    return 1;
                });
        },
        dense::nvme_search_flags);
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_nvme_indexes_search(argc, argv); }
