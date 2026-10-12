/**
 *  @file bench/dense/vram_index_algebra_bf16.cpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief The bf16-stored shard of the single-device CUDA algebra benchmark: carries the
 *      run_algebra_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/vram_index_algebra.cuh"

namespace unum::usearch::bench::dense {

template int run_algebra_with_stored_dtype<nk::f32_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, int, nk_capability_t,
    cuda_address_space_t);
template int run_algebra_with_stored_dtype<nk::bf16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, int, nk_capability_t,
    cuda_address_space_t);
template int run_algebra_with_stored_dtype<nk::f16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, int, nk_capability_t,
    cuda_address_space_t);
template int run_algebra_with_stored_dtype<nk::i8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, int, nk_capability_t,
    cuda_address_space_t);
template int run_algebra_with_stored_dtype<nk::u8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, int, nk_capability_t,
    cuda_address_space_t);

} // namespace unum::usearch::bench::dense
