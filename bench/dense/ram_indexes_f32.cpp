/**
 *  @brief The f32-stored shard of the sharded-index CPU benchmark: carries the
 *      run_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 *  @author Ash Vardanian
 *  @file bench/dense/ram_indexes_f32.cpp
 *  @date September 2, 2026
 */
#include "dense/ram_indexes.hpp"

namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;

template int us::run_with_stored_dtype<nk::f32_t, nk_f32_k>(us::bench_launch<nk::f32_t> const&);
template int us::run_with_stored_dtype<nk::bf16_t, nk_f32_k>(us::bench_launch<nk::bf16_t> const&);
template int us::run_with_stored_dtype<nk::f16_t, nk_f32_k>(us::bench_launch<nk::f16_t> const&);
template int us::run_with_stored_dtype<nk::i8_t, nk_f32_k>(us::bench_launch<nk::i8_t> const&);
template int us::run_with_stored_dtype<nk::u8_t, nk_f32_k>(us::bench_launch<nk::u8_t> const&);
