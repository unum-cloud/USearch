/**
 *  @file test/dense/vram_nvidia.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA index the residency-level VRAM scenarios run over: its ledgered flat-matrix
 *      backend and the cell the shared composers build it through.
 *
 *  Plain C++ over the NVIDIA faces, so the shared shards compile with the host compiler; the
 *  cells' kernels instantiate in @c vram_nvidia.cu, which also covers device attention, since its
 *  value lookups run on the device.
 */
#pragma once

#include "usearch/dense/vram_nvidia_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The ledgered virtual-memory flat-matrix backend every CUDA fixture places its index through. */
using counting_cuda_allocator_t = counting_device_allocator<cuda_virtual_memory_allocator<byte_t>>;

static_assert(cuda_reservable_allocator<counting_cuda_allocator_t>);

/** The newest NVIDIA ceiling every architecture of the build runs, as its @c NUMKONG_TARGET_*
 *  switches name them, so a device test cell runs each kit its build compiled: the Ampere & Ada
 *  bits accumulate, while Hopper, Blackwell & RTX Blackwell each add only their own. */
inline constexpr nk_capability_t dense_nvidia_capabilities_k = nk_cap_cuda_k | nk_cap_ampere_k |
                                                               (NUMKONG_TARGET_ADA ? nk_cap_ada_k : 0) |
                                                               (NUMKONG_TARGET_HOPPER         ? nk_cap_hopper_k
                                                                : NUMKONG_TARGET_BLACKWELL    ? nk_cap_blackwell_k
                                                                : NUMKONG_TARGET_BLACKWELLRTX ? nk_cap_blackwellrtx_k
                                                                                              : 0);

/** The suite's own fatbin, carrying every cell's kernels and the attends over its value lookups;
 *  @c vram_nvidia.cu defines @c bytes beside the bytes it embeds. */
struct dense_nvidia_fatbin_t {
    inline static constexpr nk_capability_t capability_k = dense_nvidia_capabilities_k;
    static bytes_view_t bytes() noexcept;
};

/** The NVIDIA index of one (dtype, kernel) cell over the ledgered flat matrices and its sharded
 *  collection, built over the stream of the executor the CUDA entry opened; device attention
 *  runs in @c vram_nvidia.cu instead of the shared composers. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::nvidia_k, dtype_, kernel_> {
    using index_t = vram_nvidia_index<dtype_, dense_nvidia_fatbin_t, dummy_cache_hints_t, counting_cuda_allocator_t,
                                      counting_cuda_allocator_t, cuda_device_allocator_t>;
    using indexes_t = vram_nvidia_indexes<u32_t, dtype_, dense_nvidia_fatbin_t>;
    inline static constexpr char const* vendor_prefix_k = "cuda";
    inline static constexpr bool attends_k = true;

    /** The stream the cell's contexts and checkpoints ride, on the device its indexes sit on. */
    cuda_executor_t::stream_t const& stream;

    /** Builds the cell's index over @p ledger, on the stream's device. */
    std::unique_ptr<index_t> operator()(allocation_ledger_t& ledger) const noexcept {
        counting_cuda_allocator_t const allocator {ledger, stream.device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel_, allocator, allocator));
    }

    /** A fresh traversal context on the stream. */
    typename index_t::traversal_context_t context() const noexcept { return {cuda_device_allocator_t {stream}}; }
};

} // namespace unum::usearch::test::dense
