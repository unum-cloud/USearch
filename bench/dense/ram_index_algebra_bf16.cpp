/**
 *  @file bench/dense/ram_index_algebra_bf16.cpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief The bf16-stored shard of the single-index CPU algebra benchmark: carries the
 *      run_algebra_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/ram_index_algebra.hpp"

namespace unum::usearch::bench::dense {

template int run_algebra_with_stored_dtype<nk::f32_t, nk_bf16_k>(bench_launch<nk::f32_t> const&,
                                                                 algebra_settings_t const&);
template int run_algebra_with_stored_dtype<nk::bf16_t, nk_bf16_k>(bench_launch<nk::bf16_t> const&,
                                                                  algebra_settings_t const&);
template int run_algebra_with_stored_dtype<nk::f16_t, nk_bf16_k>(bench_launch<nk::f16_t> const&,
                                                                 algebra_settings_t const&);
template int run_algebra_with_stored_dtype<nk::i8_t, nk_bf16_k>(bench_launch<nk::i8_t> const&,
                                                                algebra_settings_t const&);
template int run_algebra_with_stored_dtype<nk::u8_t, nk_bf16_k>(bench_launch<nk::u8_t> const&,
                                                                algebra_settings_t const&);

} // namespace unum::usearch::bench::dense
