/**
 *  @file test/dense/vram_amd.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index the residency-level VRAM scenarios run over: its ledgered flat-matrix
 *      backend and the cell the shared composers build it through.
 *
 *  The shared composers include it for the AMD suite, so the scenarios stay vendor-neutral while
 *  the harness's live-bytes-must-balance ledger audits the AMD flat matrices. Plain C++ over the
 *  AMD faces, so the shared shards compile with the host compiler; the cells' kernels instantiate
 *  in @c vram_amd.hip, which also covers device attention, since its value lookups run on the
 *  device.
 */
#pragma once

#include "usearch/dense/vram_amd_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The ledgered virtual-memory flat-matrix backend every AMD fixture places its index through. */
using counting_hip_allocator_t = counting_device_allocator<hip_virtual_memory_allocator<byte_t>>;

static_assert(hip_reservable_allocator<counting_hip_allocator_t>);

/** The ceiling of the suite's own code object: the CDNA3 class's, which CDNA3 and CDNA4 run. */
inline constexpr nk_capability_t dense_amd_capabilities_k = nk_cap_rocm_k | nk_cap_cdna3_k;

/** The suite's own code object, carrying the cells' kernels; @c vram_amd.hip defines @c bytes
 *  beside the bytes it embeds. */
struct dense_amd_code_object_t {
    inline static constexpr nk_capability_t capability_k = dense_amd_capabilities_k;
    static bytes_view_t bytes() noexcept;
};

/** The AMD index of one (dtype, kernel) cell over the ledgered flat matrices and its sharded
 *  collection, built over the stream of the executor the ROCm entry opened. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::amd_k, dtype_, kernel_> {
    using index_t = vram_amd_index<dtype_, dense_amd_code_object_t, dummy_cache_hints_t, counting_hip_allocator_t,
                                   counting_hip_allocator_t, hip_device_allocator_t>;
    using indexes_t = vram_amd_indexes<uuid_t, dtype_, dense_amd_code_object_t>;
    inline static constexpr char const* vendor_prefix_k = "rocm";

    /** The stream the cell's contexts and checkpoints ride, on the device its indexes sit on. */
    hip_executor_t::stream_t const& stream;

    /** Builds the cell's index of @p geometry over @p ledger, on the stream's device. */
    std::unique_ptr<index_t> operator()(allocation_ledger_t& ledger,
                                        typename index_t::geometry_t geometry) const noexcept {
        counting_hip_allocator_t const allocator {ledger, *stream.api, stream.device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel_, geometry, allocator, allocator));
    }

    /** A fresh traversal context on the stream. */
    typename index_t::traversal_context_t context() const noexcept { return {hip_device_allocator_t {stream}}; }
};

} // namespace unum::usearch::test::dense
