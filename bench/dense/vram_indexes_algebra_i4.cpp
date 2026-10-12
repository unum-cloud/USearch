/**
 *  @file bench/dense/vram_indexes_algebra_i4.cpp
 *  @author Ash Vardanian
 *  @date October 9, 2026
 *  @brief The i4-stored shard of the multi-device CUDA algebra benchmark: carries the
 *      run_indexes_algebra_with_stored_dtype instantiations the dtype-free CLI unit extern-declares.
 */
#include "dense/vram_indexes_algebra.cuh"

namespace unum::usearch::bench::dense {

template int run_indexes_algebra_with_stored_dtype<nk::f32_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
template int run_indexes_algebra_with_stored_dtype<nk::bf16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
template int run_indexes_algebra_with_stored_dtype<nk::f16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
template int run_indexes_algebra_with_stored_dtype<nk::i8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);
template int run_indexes_algebra_with_stored_dtype<nk::u8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, algebra_settings_t const&, std::span<int const>,
    nk_capability_t);

} // namespace unum::usearch::bench::dense
