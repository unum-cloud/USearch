/**
 *  @file bench/dense/vram_index_search_u1.cu
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The u1-stored shard of the single-device CUDA search benchmark: carries the
 *      run_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/vram_index_search.cuh"

namespace unum::usearch::bench::dense {

template int run_with_stored_dtype<nk::f32_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, nk_capability_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::bf16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, nk_capability_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::f16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, nk_capability_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::i8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, nk_capability_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::u8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, nk_capability_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::u1x8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u1x8_t> const&, settings_t const&, int, nk_capability_t, cuda_address_space_t);

} // namespace unum::usearch::bench::dense
