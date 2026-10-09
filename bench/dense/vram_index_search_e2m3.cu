/**
 *  @file bench/dense/vram_index_search_e2m3.cu
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief The e2m3-stored shard of the single-device CUDA search benchmark: carries the
 *      run_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/vram_index_search.cuh"

namespace unum::usearch::bench::dense {

template int run_with_stored_dtype<nk::f32_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::bf16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::f16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::i8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);
template int run_with_stored_dtype<nk::u8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, int, cuda_arch_t, cuda_address_space_t);

} // namespace unum::usearch::bench::dense
