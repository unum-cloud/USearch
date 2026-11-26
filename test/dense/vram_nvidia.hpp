/**
 *  @file test/dense/vram_nvidia.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA index the VRAM rung cells run over: its ledgered flat-matrix backend, the
 *      library's fatbin of the class the build names, and the cell the shared composers build it
 *      through.
 *
 *  Plain C++ over the NVIDIA faces, so the cells compile with the host compiler; the kernels come
 *  from the class fatbins the library links in, so the suite compiles no device code of its own.
 */
#pragma once

#include "usearch/dense/vram_nvidia_indexes.hpp"

#include "dense.cuh"
#include "classes.hpp"    // `cuda_class_k` - the class the rung cells run
#include "dense_kits.hpp" // `dense_nvidia_fatbin` - the library's fatbin of that class

namespace unum::usearch::test::dense {

/** Why the rung cells skip on a device the library lists under another class. */
inline constexpr char needs_dense_nvidia_class_k[] = "needs cuda_class_k";

/** Whether @p device runs the cells of @c cuda_class_k , as the library lists its class. */
inline bool dense_nvidia_runs(gpu_device_t const& device) noexcept {
    return listed_class(device) == static_cast<usearch_gpu_class_t>(cuda_class_k);
}

/** The ledgered virtual-memory flat-matrix backend every CUDA fixture places its index through. */
using counting_cuda_allocator_t = counting_device_allocator<cuda_virtual_memory_allocator<byte_t>>;

static_assert(a_cuda_reservable_allocator<counting_cuda_allocator_t>);

/** The NVIDIA index of one (dtype, kernel) cell over the ledgered flat matrices and its sharded
 *  collection, built over the stream of the executor the CUDA entry opened. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::nvidia_k, dtype_, kernel_> {
    using index_t =
        vram_nvidia_index<dtype_, dense_nvidia_fatbin<cuda_class_k, dtype_>, fu::standard_yield_t, dummy_cache_hints_t,
                          counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
    using indexes_t = vram_nvidia_indexes<util::uuid128_t, dtype_, dense_nvidia_fatbin<cuda_class_k, dtype_>>;
    inline static constexpr char const* vendor_prefix_k = "cuda";

    /** The stream the cell's contexts and checkpoints ride, on the device its indexes sit on. */
    cuda_executor_t::stream_t const& stream;

    /** Builds the cell's index of @p geometry over @p ledger, on the stream's device. */
    std::unique_ptr<index_t> operator()(allocation_ledger_t& ledger,
                                        typename index_t::geometry_t geometry) const noexcept {
        counting_cuda_allocator_t const allocator {ledger, stream.handle.api(), test::gpu_drivers().extensions.cuda,
                                                   static_cast<int>(stream.ordinal)};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel_, geometry, allocator, allocator));
    }

    /** A fresh traversal context on the stream. */
    typename index_t::traversal_context_t context() const noexcept { return {cuda_device_allocator_t {stream}}; }
};

} // namespace unum::usearch::test::dense
