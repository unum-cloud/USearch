/**
 *  @file bench/dense/vram_kernels.cuh
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief What the CUDA benchmarks' host units share with their device units: the ceilings a build
 *      instantiates, the per-dtype fatbins, and the value lookup the algebra benchmarks attend
 *      through.
 *
 *  It includes nothing of the benchmarks' host pipeline, so the device units of
 *  @c vram_kernels_<dtype>.cu compile neither the CPU metrics nor NumKong's own kernels.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include "usearch/dense/vram_nvidia_index.cuh"

namespace unum::usearch::bench::dense {
using namespace unum::usearch::utilities;
using namespace unum::usearch::dense;

/** The NVIDIA capabilities one generation instantiates: the Ampere & Ada bits accumulate, as
 *  NumKong detects them, while @p generation alone stands for Hopper, Blackwell or RTX Blackwell,
 *  which run only their own kits. */
constexpr nk_capability_t cuda_ceiling_of(nk_capability_t generation) noexcept {
    nk_capability_t const ampere = nk_cap_cuda_k | nk_cap_ampere_k;
    return generation == nk_cap_ampere_k ? ampere : ampere | nk_cap_ada_k | generation;
}

/** The newest NVIDIA generation every listed architecture runs, Ampere when none past it: the one
 *  ceiling a build compiles beside Ampere's. */
inline constexpr nk_capability_t cuda_compiled_generation_k = NUMKONG_TARGET_BLACKWELLRTX ? nk_cap_blackwellrtx_k
                                                              : NUMKONG_TARGET_BLACKWELL  ? nk_cap_blackwell_k
                                                              : NUMKONG_TARGET_HOPPER     ? nk_cap_hopper_k
                                                              : NUMKONG_TARGET_ADA        ? nk_cap_ada_k
                                                                                          : nk_cap_ampere_k;

/** The benchmarks' fatbin of @p dtype_ 's kernels; @c vram_kernels.cpp defines one per compiled
 *  dtype beside its bundle. */
template <nk_dtype_t dtype_>
bytes_view_t cuda_bench_bundle() noexcept;

/** The fatbin of @p dtype_ 's kernels, named under the ceiling @p capability_ a benchmarked index
 *  runs. */
template <nk_capability_t capability_, nk_dtype_t dtype_>
struct cuda_bench_fatbin {
    inline static constexpr nk_capability_t capability_k = capability_;
    static bytes_view_t bytes() noexcept { return cuda_bench_bundle<dtype_>(); }
};

/** Resolves an ID, a node on one device or a key across the shards, to its value row in a device
 *  table of rows @c pitch bytes apart, the table starting at ID @c first_id. */
struct attention_value_lookup_t {
    byte_t const* rows;
    std::size_t pitch;
    std::size_t first_id = 0;
};

/** Points at the value row of @p id in the table of @p lookup. */
__device__ inline void const* vram_value_row(attention_value_lookup_t const& lookup, std::size_t id) noexcept {
    return lookup.rows + (id - lookup.first_id) * lookup.pitch;
}

/** Spells @c attention_value_lookup_t into an attend kernel's name, @c S0_ being
 *  @c unum::usearch. */
inline void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<attention_value_lookup_t>) noexcept {
    symbol.append("NS0_5bench5dense");
    symbol.append_name("attention_value_lookup_t");
    symbol.append("E");
}

/** The lookup each shard of the multi-device algebra benchmark attends through, by its keys. */
using keyed_attention_lookup_t = vram_keyed_lookup<u32_t, attention_value_lookup_t>;

/** Instantiates @p dtype_ 's kernels under Ampere's ceiling and the newest compiled generation's,
 *  with the attends over both algebra lookups. */
template <nk_dtype_t dtype_>
__device__ void instantiate_cuda_ceilings() noexcept {
    cuda_instantiate<cuda_ceiling_of(nk_cap_ampere_k), dtype_, attention_value_lookup_t, keyed_attention_lookup_t>();
    if constexpr (cuda_compiled_generation_k != nk_cap_ampere_k)
        cuda_instantiate<cuda_ceiling_of(cuda_compiled_generation_k), dtype_, attention_value_lookup_t,
                         keyed_attention_lookup_t>();
}

} // namespace unum::usearch::bench::dense
