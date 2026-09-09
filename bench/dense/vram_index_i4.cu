/**
 *  @brief The i4-stored shard of the CUDA dense-index benchmark: carries the
 *      run_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 *  @author Ash Vardanian
 *  @file bench/dense/vram_index_i4.cu
 *  @date July 23, 2026
 */
#include "dense/vram_index.cuh"

namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;

template int us::run_with_stored_dtype<nk::f32_t, nk_i4_k>( //
    us::persisted_dataset<nk::f32_t> const&, us::bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
template int us::run_with_stored_dtype<nk::bf16_t, nk_i4_k>( //
    us::persisted_dataset<nk::bf16_t> const&, us::bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
template int us::run_with_stored_dtype<nk::f16_t, nk_i4_k>( //
    us::persisted_dataset<nk::f16_t> const&, us::bench_options_t const&, int, us::cuda_arch_t,
    us::cuda_address_space_t);
template int us::run_with_stored_dtype<nk::i8_t, nk_i4_k>( //
    us::persisted_dataset<nk::i8_t> const&, us::bench_options_t const&, int, us::cuda_arch_t, us::cuda_address_space_t);
template int us::run_with_stored_dtype<nk::u8_t, nk_i4_k>( //
    us::persisted_dataset<nk::u8_t> const&, us::bench_options_t const&, int, us::cuda_arch_t, us::cuda_address_space_t);
