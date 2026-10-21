/**
 *  @file test/dense/vram_nvidia_kernels.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief What the NVIDIA suite's host units share with its device units: the ceiling the cells
 *      run, the per-dtype fatbins, and the value lookup device attention resolves rows through.
 *
 *  It includes nothing of the shared test harness, so the device units of
 *  @c vram_nvidia_<dtype>.cu compile neither the CPU metrics nor NumKong's own kernels.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include "usearch/dense/vram_nvidia_index.cuh"

namespace us = unum::usearch;

namespace unum::usearch::test::dense {
using namespace unum::usearch::utilities;
using namespace unum::usearch::dense;

/** The newest NVIDIA ceiling every architecture of the build runs, as its @c NUMKONG_TARGET_*
 *  switches name them, so a device test cell runs each kit its build compiled: the Ampere & Ada
 *  bits accumulate, while Hopper, Blackwell & RTX Blackwell each add only their own. */
inline constexpr nk_capability_t dense_nvidia_capabilities_k = nk_cap_cuda_k | nk_cap_ampere_k |
                                                               (NUMKONG_TARGET_ADA ? nk_cap_ada_k : 0) |
                                                               (NUMKONG_TARGET_HOPPER         ? nk_cap_hopper_k
                                                                : NUMKONG_TARGET_BLACKWELL    ? nk_cap_blackwell_k
                                                                : NUMKONG_TARGET_BLACKWELLRTX ? nk_cap_blackwellrtx_k
                                                                                              : 0);

/** The suite's fatbin of @p dtype_ 's kernels and the attends over its value lookups;
 *  @c vram_nvidia.cpp defines @c bytes for each compiled dtype beside the bytes it embeds. */
template <nk_dtype_t dtype_>
struct dense_nvidia_fatbin {
    inline static constexpr nk_capability_t capability_k = dense_nvidia_capabilities_k;
    static bytes_view_t bytes() noexcept;
};

/** A device lookup over a value table whose rows sit a 16-byte pitch apart: an ID below @c count
 *  names its row, a node ID on one GPU and a key on shards, and any other has none. */
struct cuda_test_lookup_t {

    /** Names the lookup to the attend kernels the suite's fatbins carry. */
    inline static constexpr u32_t device_lookup_k = 1;

    byte_t const* rows;
    std::size_t pitch;
    std::size_t count;

    __device__ void const* operator()(u32_t id) const noexcept { return id < count ? rows + id * pitch : nullptr; }
};

/** The lookup the sharded suite's shards attend through, by their keys. */
using cuda_test_keyed_lookup_t = vram_keyed_lookup<u32_t, cuda_test_lookup_t>;

} // namespace unum::usearch::test::dense

template <>
struct us::dense::cuda_value_lookup_of<us::test::dense::cuda_test_lookup_t::device_lookup_k> {
    using type = us::test::dense::cuda_test_lookup_t;
};
template <>
struct us::dense::cuda_value_lookup_of<us::test::dense::cuda_test_keyed_lookup_t::device_lookup_k> {
    using type = us::test::dense::cuda_test_keyed_lookup_t;
};
