/**
 *  @file bench/dense/vram_kernels.cuh
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The device half the CUDA benchmarks' entry units add to their host headers: the kernels
 *      every compiled ceiling instantiates, and the attention lookup's call.
 *
 *  Only the @c .cu entries include it, so the per-dtype host shards compile as plain C++ and never
 *  meet a device function.
 */
#pragma once

#include "usearch/dense/vram_nvidia_index.cuh"

#include "dense/vram_index_search.cuh"

namespace unum::usearch::bench::dense {

/** Whether the build's architectures all run @p generation 's kits, as CMake names them. */
constexpr bool cuda_generation_compiled(nk_capability_t generation) noexcept {
    switch (generation) {
    case nk_cap_ada_k: return NUMKONG_TARGET_ADA;
    case nk_cap_hopper_k: return NUMKONG_TARGET_HOPPER;
    case nk_cap_blackwell_k: return NUMKONG_TARGET_BLACKWELL;
    case nk_cap_blackwellrtx_k: return NUMKONG_TARGET_BLACKWELLRTX;
    default: return NUMKONG_TARGET_AMPERE;
    }
}

/** Instantiates every compiled ceiling's kernels for @p dtype_, with the attends over each of
 *  @p lookups_. */
template <nk_dtype_t dtype_, u32_t... lookups_>
__device__ void instantiate_cuda_ceilings() noexcept {
    cuda_instantiate<cuda_ceiling_of(nk_cap_ampere_k), dtype_, lookups_...>();
    [&]<std::size_t... rows_>(std::index_sequence<rows_...>) {
        (
            [&] {
                if constexpr (cuda_generation_compiled(cuda_compute_choices_k[rows_].bit))
                    cuda_instantiate<cuda_ceiling_of(cuda_compute_choices_k[rows_].bit), dtype_, lookups_...>();
            }(),
            ...);
    }(std::make_index_sequence<std::size(cuda_compute_choices_k)> {});
}

/** Instantiates the kernels of every compiled dtype under every ceiling, with the attends over each
 *  of @p lookups_ - what a binary's entry unit compiles into its fatbin. */
template <u32_t... lookups_>
__device__ void instantiate_cuda_kernels() noexcept {
    [&]<std::size_t... dtypes_>(std::index_sequence<dtypes_...>) {
        (instantiate_cuda_ceilings<cuda_dtypes_ladder_k[dtypes_], lookups_...>(), ...);
    }(std::make_index_sequence<USEARCH_DENSE_DTYPES> {});
}

/** The attention lookup as the kernels call it: the host's record, byte for byte, which an attend
 *  kernel copies its argument into. */
struct kernel_attention_lookup_t : attention_value_lookup_t {
    __device__ void const* operator()(std::size_t id) const noexcept { return rows + (id - first_id) * pitch; }
};
static_assert(sizeof(kernel_attention_lookup_t) == sizeof(attention_value_lookup_t));

} // namespace unum::usearch::bench::dense
