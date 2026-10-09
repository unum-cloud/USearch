/**
 *  @file bench/dense/ram_index_search_e5m2.cpp
 *  @author Ash Vardanian
 *  @date September 6, 2026
 *  @brief The e5m2-stored shard of the single-index CPU search benchmark: carries the
 *      run_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/ram_index_search.hpp"

namespace unum::usearch::bench::dense {

template int run_with_stored_dtype<nk::f32_t, nk_e5m2_k>(bench_launch<nk::f32_t> const&);
template int run_with_stored_dtype<nk::bf16_t, nk_e5m2_k>(bench_launch<nk::bf16_t> const&);
template int run_with_stored_dtype<nk::f16_t, nk_e5m2_k>(bench_launch<nk::f16_t> const&);
template int run_with_stored_dtype<nk::i8_t, nk_e5m2_k>(bench_launch<nk::i8_t> const&);
template int run_with_stored_dtype<nk::u8_t, nk_e5m2_k>(bench_launch<nk::u8_t> const&);

} // namespace unum::usearch::bench::dense
