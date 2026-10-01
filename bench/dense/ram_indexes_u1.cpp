/**
 *  @file bench/dense/ram_indexes_u1.cpp
 *  @author Ash Vardanian
 *  @date September 2, 2026
 *  @brief The u1-stored shard of the sharded-index CPU benchmark: carries the run_with_stored_dtype
 *      instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/ram_indexes.hpp"

namespace unum::usearch::bench::dense {

template int run_with_stored_dtype<nk::f32_t, nk_u1_k>(bench_launch<nk::f32_t> const&);
template int run_with_stored_dtype<nk::bf16_t, nk_u1_k>(bench_launch<nk::bf16_t> const&);
template int run_with_stored_dtype<nk::f16_t, nk_u1_k>(bench_launch<nk::f16_t> const&);
template int run_with_stored_dtype<nk::i8_t, nk_u1_k>(bench_launch<nk::i8_t> const&);
template int run_with_stored_dtype<nk::u8_t, nk_u1_k>(bench_launch<nk::u8_t> const&);
template int run_with_stored_dtype<nk::u1x8_t, nk_u1_k>(bench_launch<nk::u1x8_t> const&);

} // namespace unum::usearch::bench::dense
