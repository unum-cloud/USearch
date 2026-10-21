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

namespace us = unum::usearch;

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
 *  table of rows @c pitch bytes apart, the table starting at ID @c first_id;
 *  @c kernel_attention_lookup_t adds the call. */
struct attention_value_lookup_t {

    /** Names the lookup to the attend kernels the benchmarks' fatbins carry. */
    inline static constexpr u32_t device_lookup_k = 1;

    byte_t const* rows;
    std::size_t pitch;
    std::size_t first_id = 0;
};

/** The attention lookup as the kernels call it: the host's record, byte for byte, which an attend
 *  kernel copies its argument into. */
struct kernel_attention_lookup_t : attention_value_lookup_t {
    __device__ void const* operator()(std::size_t id) const noexcept { return rows + (id - first_id) * pitch; }
};
static_assert(sizeof(kernel_attention_lookup_t) == sizeof(attention_value_lookup_t));

/** The lookup each shard of the multi-device algebra benchmark attends through, by its keys. */
using keyed_attention_lookup_t = vram_keyed_lookup<u32_t, attention_value_lookup_t>;

/** Instantiates @p dtype_ 's kernels under Ampere's ceiling and the newest compiled generation's,
 *  with the attends over both algebra lookups. */
template <nk_dtype_t dtype_>
__device__ void instantiate_cuda_ceilings() noexcept {
    cuda_instantiate<cuda_ceiling_of(nk_cap_ampere_k), dtype_, attention_value_lookup_t::device_lookup_k,
                     keyed_attention_lookup_t::device_lookup_k>();
    if constexpr (cuda_compiled_generation_k != nk_cap_ampere_k)
        cuda_instantiate<cuda_ceiling_of(cuda_compiled_generation_k), dtype_, attention_value_lookup_t::device_lookup_k,
                         keyed_attention_lookup_t::device_lookup_k>();
}

} // namespace unum::usearch::bench::dense

/** The attend kernels resolve rows by node ID on the one device. */
template <>
struct us::dense::cuda_value_lookup_of<us::bench::dense::attention_value_lookup_t::device_lookup_k> {
    using type = us::bench::dense::kernel_attention_lookup_t;
};

/** The key-level lookup each shard's attend kernels resolve rows through. */
template <>
struct us::dense::cuda_value_lookup_of<us::bench::dense::keyed_attention_lookup_t::device_lookup_k> {
    using type = us::dense::vram_keyed_lookup<us::utilities::u32_t, us::bench::dense::kernel_attention_lookup_t>;
};
