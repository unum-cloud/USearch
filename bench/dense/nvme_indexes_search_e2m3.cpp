/**
 *  @file bench/dense/nvme_indexes_search_e2m3.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The e2m3-stored shard of the sharded-index NVMe search benchmark: carries the
 *      run_nvme_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/nvme_indexes_search.hpp"

namespace unum::usearch::bench::dense {

template int run_nvme_with_stored_dtype<nk::f32_t, nk_e2m3_k>(bench_launch<nk::f32_t> const&, nvme_settings_t const&);
template int run_nvme_with_stored_dtype<nk::bf16_t, nk_e2m3_k>(bench_launch<nk::bf16_t> const&, nvme_settings_t const&);
template int run_nvme_with_stored_dtype<nk::f16_t, nk_e2m3_k>(bench_launch<nk::f16_t> const&, nvme_settings_t const&);
template int run_nvme_with_stored_dtype<nk::i8_t, nk_e2m3_k>(bench_launch<nk::i8_t> const&, nvme_settings_t const&);
template int run_nvme_with_stored_dtype<nk::u8_t, nk_e2m3_k>(bench_launch<nk::u8_t> const&, nvme_settings_t const&);

} // namespace unum::usearch::bench::dense
