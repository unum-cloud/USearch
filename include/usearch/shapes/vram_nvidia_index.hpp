/**
 *  @file include/usearch/shapes/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over 3D point clouds on NVIDIA GPUs: the basic VRAM index under the
 *      Chamfer and Hausdorff kits @c shapes_ladder picks in @c vram_nvidia_metric.cuh.
 *
 *  Plain C++, so the host classes build with any compiler: the members reading the kits, which size
 *  a launch's spans, are defined beside the kits in the metric, which only the modality's NVIDIA
 *  unit compiles. Rows are NumKong packs end to end, as @c nk::shapes_pack lays them out, in the
 *  caller's tape and in the index.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/capabilities.h> // `nk_kernel_chamfers_packed_k` & kin - the metric a kit scores under
#include <numkong/shapes.hpp>     // `nk::shapes_dtype` - the scalars a kit stores

#include "usearch/utilities/types.hpp"
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

/** Similarity Search over 3D point clouds of @p scalar_ on NVIDIA GPUs, under Chamfer or Hausdorff
 *  as the kernel kind it is built with names, scored by the kit @c shapes_ladder picks under
 *  @p capability_: a block walks each query, a warp scoring each candidate. Node IDs are
 *  @p node_id_type_, and @p kernels_type_ names the fatbin. */
template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
struct vram_nvidia_index
    : basic::vram_index<vram_nvidia_index<scalar_, node_id_type_, capability_, kernels_type_>, node_id_type_> {

    using base_t = basic::vram_index<vram_nvidia_index, node_id_type_>;

    /** Rungs of the kits' kernels: one, serving every cloud. */
    inline static constexpr std::size_t rungs_k = 1;

    /** Pins the index to the storage allocator's device, scoring under @p kernel,
     *  @c nk_kernel_chamfers_packed_k or @c nk_kernel_hausdorffs_packed_k. */
    explicit vram_nvidia_index(nk_kernel_kind_t kernel,
                               cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : base_t(kernel, allocator) {}

  private:
    friend base_t;

    /** The rung serving rows of @p longest bytes, or @c rungs_k when none does. Defined in
     *  @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    static std::size_t rung_serving_(std::size_t longest) noexcept;

    /** Lanes the kit scores a row with on @p rung. */
    static u32_t lanes_per_row_(std::size_t rung) noexcept;

    /** Bytes the @p bytes long pack at @p data takes once prepared, a multiple of 16. */
    static std::size_t span_bytes_(byte_t const* data, std::size_t bytes) noexcept;

    /** Bytes the @p stored pack saves into: its cloud unpacked, @b [points,3] scalars, as the RAM
     *  codec saves a cloud. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept;

    /** Unpacks the @p stored pack into @p saved. */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept;

    /** Bytes the @p saved cloud stores into once packed; @c malformed_k unless it holds whole
     *  points. */
    static expected<std::size_t> stored_bytes_(bytes_view_t saved) noexcept;

    /** Packs the @p saved cloud into @p stored, as an insertion packs one on the host. */
    static status_t load_value_(bytes_view_t saved, bytes_span_t stored) noexcept;

    /** Loads the fatbin onto @p device and finds every kernel by its symbol. */
    status_t load_kernels_(int device) noexcept;

    cuda_function_handle_t kernel_function_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[rung][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The fatbin on this device, and its kernels in @c vram_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[rungs_k][static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
};

} // namespace unum::usearch::shapes
