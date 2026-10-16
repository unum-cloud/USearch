/**
 *  @file src/dense_kernels.cuh
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The device half every dense GPU unit adds to @c dense_specializations.cuh: the C value
 *      lookup's call, and the traits naming it to each vendor's attend kernels.
 *
 *  Only the NVIDIA and AMD class units include it, so the dispatch units, which compile as plain
 *  C++, see the lookup as the record they fill and never a device function.
 */
#pragma once

#include "dense_specializations.cuh"

/** The C value lookup as the kernels call it: the host's record, byte for byte, which an attend
 *  kernel copies its argument into. */
struct kernel_value_lookup_t : device_value_lookup_t {
    __device__ void const* operator()(us::utilities::uuid_t key) const noexcept {
        usearch_key_t const scored = std::bit_cast<usearch_key_t>(key);
        void const* row = nullptr;
        callback(state, &scored, 1, scalar, dimensions, &row);
        return row;
    }
};
static_assert(sizeof(kernel_value_lookup_t) == sizeof(device_value_lookup_t));

/** The keyed lookup each shard's attend kernels resolve rows through, laid out as
 *  @c vram_attend_lookup_t. */
using kernel_attend_lookup_t = us::dense::vram_keyed_lookup<us::utilities::uuid_t, kernel_value_lookup_t>;

#if USEARCH_ARCH_CUDA_
template <>
struct us::dense::cuda_value_lookup_of<vram_attend_lookup_t::device_lookup_k> {
    using type = kernel_attend_lookup_t;
};
#endif

#if USEARCH_ARCH_ROCM_
template <>
struct us::dense::amd_value_lookup_of<vram_attend_lookup_t::device_lookup_k> {
    using type = kernel_attend_lookup_t;
};
#endif
