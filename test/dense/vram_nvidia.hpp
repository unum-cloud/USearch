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

#include <span> // `std::span` - the library's fatbin bounds

#include "usearch/dense/vram_nvidia_indexes.hpp"

#include "dense.cuh"

/** The bounds of every dtype's fatbin of one NVIDIA class, in pairs in the library's dtype ladder,
 *  defined in that class's @c src/dense/vram_variants_nvidia_<class>.cpp unit; an unnamed class's
 *  bundles are empty. */
template <us::utilities::gpu_class_t gpu_class_>
std::span<us::utilities::byte_t const* const> bundle_bounds() noexcept;

namespace unum::usearch::test::dense {

/** The newest NVIDIA class the build names, by the @c NUMKONG_TARGET_* switches its architectures
 *  set; the cells run only on a device of that class. */
inline constexpr gpu_class_t dense_nvidia_class_k = NUMKONG_TARGET_HOPPER         ? gpu_class_t::nvidia_hopper_k
                                                    : NUMKONG_TARGET_BLACKWELL    ? gpu_class_t::nvidia_blackwell_k
                                                    : NUMKONG_TARGET_BLACKWELLRTX ? gpu_class_t::nvidia_blackwellrtx_k
                                                    : NUMKONG_TARGET_ADA          ? gpu_class_t::nvidia_ada_k
                                                                                  : gpu_class_t::nvidia_ampere_k;

/** The library's fatbin of @p dtype_ 's kernels for @ref dense_nvidia_class_k, under the ceiling
 *  the library compiles that class with. */
template <nk_dtype_t dtype_>
struct dense_nvidia_fatbin {
    static_assert(dtype_ == nk_bf16_k, "the rung cells run bf16, which leads the library's dtype ladder");
    inline static constexpr nk_capability_t capability_k =
        dense_nvidia_class_k == gpu_class_t::nvidia_ampere_k
            ? nk_cap_cuda_k | nk_cap_ampere_k
            : nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k |
                  (dense_nvidia_class_k == gpu_class_t::nvidia_hopper_k         ? nk_cap_hopper_k
                   : dense_nvidia_class_k == gpu_class_t::nvidia_blackwell_k    ? nk_cap_blackwell_k
                   : dense_nvidia_class_k == gpu_class_t::nvidia_blackwellrtx_k ? nk_cap_blackwellrtx_k
                                                                                : 0);
    static bytes_view_t bytes() noexcept {
        std::span<byte_t const* const> const bounds = bundle_bounds<dense_nvidia_class_k>();
        return bytes_view_t {bounds[0], bounds[1]};
    }
};

/** The ledgered virtual-memory flat-matrix backend every CUDA fixture places its index through. */
using counting_cuda_allocator_t = counting_device_allocator<cuda_virtual_memory_allocator<byte_t>>;

static_assert(cuda_reservable_allocator<counting_cuda_allocator_t>);

/** The NVIDIA index of one (dtype, kernel) cell over the ledgered flat matrices and its sharded
 *  collection, built over the stream of the executor the CUDA entry opened. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell<gpu_vendor_t::nvidia_k, dtype_, kernel_> {
    using index_t = vram_nvidia_index<dtype_, dense_nvidia_fatbin<dtype_>, dummy_cache_hints_t,
                                      counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
    using indexes_t = vram_nvidia_indexes<uuid_t, dtype_, dense_nvidia_fatbin<dtype_>>;
    inline static constexpr char const* vendor_prefix_k = "cuda";

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
