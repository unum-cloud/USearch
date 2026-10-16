/**
 *  @file include/usearch/shapes/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device kits for the shapes VRAM indexes, the GPU twin of @c ram_metric.hpp, their ladder,
 *      the kernels, and the members of @c vram_nvidia_index.hpp reading them.
 *
 *  A kit binds the shapes distances over one scalar's packs to one GPU generation, named like the
 *  dense kits; the generation slot names the oldest tier whose instructions it needs. Chamfer and
 *  Hausdorff need nothing past the @c ampere floor: NumKong's SIMT bodies score two packs in place,
 *  f16 packs in f32 math and f32 packs in f64 math as the CPU kits do, and stop once a candidate
 *  lies past the walk's radius. One kit covers both, picked at run time by the kernel kind the view
 *  carries, as dense quads kits pick their metric.
 *
 *  A stored row is its NumKong pack byte for byte, and a walk prepares one by copying it into its
 *  span, so the bodies read the same pack either way. The whole warp scores one candidate, its
 *  lanes splitting the outer cloud's points: a lane alone wins only on clouds of a few points,
 *  which a pack's bytes cannot tell apart from larger ones. A block walks each query while its four
 *  warps score a candidate each. Only the modality's NVIDIA unit compiles this file, so dispatch
 *  never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <iterator> // `std::size` - every rung's kernels
#include <utility>  // `std::index_sequence` - every rung's kernels

#include <numkong/shapes.h>        // `nk_shapes_pack_f16_serial` & kin, which `shapes/serial.h` expects declared ahead
#include <numkong/shapes/cuda.cuh> // `nk_chamfer_warp_packed_f16_cuda_` & kin
#include <numkong/types.h>         // `NUMKONG_ARCH_CUDA_AMPERE_` - the kits this unit carries
#include <numkong/types.hpp>       // `nk::type_for` - a kernel's scalar

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"        // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.cuh" // `basic::cuda_kernel_body`
#include "usearch/shapes/ram_metric.hpp"       // `shapes_ladder`
#include "usearch/shapes/vram_nvidia_index.hpp"

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

#pragma region Kits

/** The pack a walk scores against, in the walking warp's span. */
struct vram_pack_query_t {
    byte_t const* pack;
};

/** Lays the pack @p row holds out in @p span, every lane of the warp copying 16 bytes at a time;
 *  packs are whole alignment lines, so no tail remains. */
__device__ __forceinline__ vram_pack_query_t vram_prepare_pack(basic::vram_row_t row, byte_t* span,
                                                               u32_t lane) noexcept {
    using line_t = uint4;
    line_t const* const source = reinterpret_cast<line_t const*>(row.data);
    line_t* const target = reinterpret_cast<line_t*>(span);
    for (u32_t line = lane; line < row.bytes / sizeof(line_t); line += cuda_warp_t::width_k)
        target[line] = source[line];
    cuda_warp_t::sync();
    return {span};
}

/** Chamfer and Hausdorff between f16 packs in f32 math, a warp to a row. */
struct f16_ampere_shapes_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    inline static constexpr u32_t lanes_per_row_k = 32;

    using warp_t = cuda_warp_t;
    using query_t = vram_pack_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    static constexpr bool serves(std::size_t) noexcept { return true; }

    /** Bytes the pack at @p data takes once prepared: the pack itself. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return vram_prepare_pack(row, span, lane);
    }

    /** The gap between the two clouds' largest radii, from their headers alone, bounds Hausdorff;
     *  nothing bounds Chamfer. */
    static __device__ __forceinline__ distance_t bound(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_row_t row) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k ? nk_hausdorff_bound_packed_f16_simt_(query.pack, row.data) : 0;
    }

    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_row_t row, distance_t radius, u32_t lane) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k
                   ? nk_hausdorff_warp_packed_f16_cuda_(query.pack, row.data, radius, lane)
                   : nk_chamfer_warp_packed_f16_cuda_(query.pack, row.data, radius, lane);
    }
};

/** Chamfer and Hausdorff between f32 packs in f64 math, a warp to a row. */
struct f32_ampere_shapes_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    inline static constexpr u32_t lanes_per_row_k = 32;

    using warp_t = cuda_warp_t;
    using query_t = vram_pack_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    static constexpr bool serves(std::size_t) noexcept { return true; }
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return vram_prepare_pack(row, span, lane);
    }
    static __device__ __forceinline__ distance_t bound(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_row_t row) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k
                   ? static_cast<distance_t>(nk_hausdorff_bound_packed_f32_simt_(query.pack, row.data))
                   : 0;
    }
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_row_t row, distance_t radius, u32_t lane) noexcept {
        return static_cast<distance_t>(kernel == nk_kernel_hausdorffs_packed_k
                                           ? nk_hausdorff_warp_packed_f32_cuda_(query.pack, row.data, radius, lane)
                                           : nk_chamfer_warp_packed_f32_cuda_(query.pack, row.data, radius, lane));
    }
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct shapes_ladder<nk::f16_t, caps_> {
    using kit_t = typename highest_kit<caps_, f16_ampere_shapes_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct shapes_ladder<nk::f32_t, caps_> {
    using kit_t = typename highest_kit<caps_, f32_ampere_shapes_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

#pragma endregion Kits

#pragma region Face

template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
std::size_t vram_nvidia_index<scalar_, node_id_type_, capability_, kernels_type_>::rung_serving_(
    std::size_t longest) noexcept {
    using kit_t = typename shapes_ladder<scalar_, capability_>::kit_t;
    return kit_t::serves(longest) ? 0 : rungs_k;
}

template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
u32_t vram_nvidia_index<scalar_, node_id_type_, capability_, kernels_type_>::lanes_per_row_(std::size_t) noexcept {
    using kit_t = typename shapes_ladder<scalar_, capability_>::kit_t;
    return kit_t::lanes_per_row_k;
}

template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
std::size_t vram_nvidia_index<scalar_, node_id_type_, capability_, kernels_type_>::span_bytes_(
    byte_t const* data, std::size_t bytes) noexcept {
    using kit_t = typename shapes_ladder<scalar_, capability_>::kit_t;
    return kit_t::span_bytes(data, bytes);
}

template <nk::shapes_dtype scalar_, typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
status_t vram_nvidia_index<scalar_, node_id_type_, capability_, kernels_type_>::load_kernels_(int device) noexcept {
    auto [module, status] = cuda_module_t::load(device, kernels_type_::bytes());
    if (failed(status)) return status;
    module_ = std::move(module);
    vram_kernel_symbol_t symbol;
    for (basic::cuda_rung_t const instantiated : basic::cuda_rungs) {
        vram_kernel_symbol(symbol, "shapes", "cuda_kernel", {"vram_arguments", 3, itanium_scope_t::basic_k},
                           instantiated.kernel, capability_, scalar_::dtype(), itanium_type<node_id_type_> {},
                           instantiated.slots);
        cuda_function_handle_t& function =
            kernels_[0][static_cast<u32_t>(instantiated.kernel)][gpu_rung_index(instantiated.slots)];
        if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
    }
    return status_t::success_k;
}

/** Every NVIDIA shapes kernel: @p kernel_ picks the walk, @p capability_ and @p dtype_ the kit
 *  through @c shapes_ladder, @p node_id_type_ the IDs of the records, and @p slots_ the register
 *  rung. */
template <vram_kernel_t kernel_, nk_capability_t capability_, nk_dtype_t dtype_, typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, basic::cuda_resident_blocks(kernel_, slots_))
    cuda_kernel(basic::vram_arguments<node_id_type_> arguments) {
    using kit_t = typename shapes_ladder<typename nk::type_for<dtype_>::type, capability_>::kit_t;
    basic::cuda_kernel_body<kernel_, kit_t, 0, slots_>(arguments);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, typename node_id_type_, std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<basic::cuda_rungs[rungs_].kernel, capability_, dtype_, node_id_type_,
                                    basic::cuda_rungs[rungs_].slots>),
     ...);
}

/** Instantiates every kernel over packs of @p dtype_ under @p capability_ with @p node_id_type_
 *  IDs; the unit compiling the fatbin names it once per dtype and node-ID type. Taking the kernels'
 *  addresses in device code makes NVCC emit them, while a host-only pass emits no launch stub. */
template <nk_capability_t capability_, nk_dtype_t dtype_, typename node_id_type_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_rungs<capability_, dtype_, node_id_type_>(
        std::make_index_sequence<std::size(basic::cuda_rungs)> {});
}

#pragma endregion Face

} // namespace unum::usearch::shapes
