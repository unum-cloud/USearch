/**
 *  @brief The i8-stored shard of the CPU dense-index benchmark: carries the
 *      run_with_compute instantiations the dtype-free CLI unit extern-declares.
 *  @author Ash Vardanian
 *  @file bench/dense_index_i8.cpp
 *  @date July 23, 2026
 */
#include "dense_index.hpp"

namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;

template int us::run_with_compute<nk::f32_t, nk_i8_k>(us::bench_launch<nk::f32_t> const&);
template int us::run_with_compute<nk::bf16_t, nk_i8_k>(us::bench_launch<nk::bf16_t> const&);
template int us::run_with_compute<nk::i8_t, nk_i8_k>(us::bench_launch<nk::i8_t> const&);
template int us::run_with_compute<nk::u8_t, nk_i8_k>(us::bench_launch<nk::u8_t> const&);
