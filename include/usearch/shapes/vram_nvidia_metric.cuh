/**
 *  @file include/usearch/shapes/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device kits for the shapes VRAM indexes, the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one shapes distance over one scalar's packs to one GPU generation, named like the
 *  dense kits; the generation slot names the oldest tier whose instructions it needs. Chamfer and
 *  Hausdorff need nothing past the @c ampere floor: NumKong's SIMT bodies score two packs in place,
 *  f16 packs in f32 math and f32 packs in f64 math as the CPU kits do, and stop once a candidate
 *  lies past the walk's radius.
 *
 *  A stored row is its NumKong pack byte for byte, and a walk prepares one by copying it into its
 *  span, so the bodies read the same pack either way. Each kit comes in two forms: a lane scores a
 *  candidate alone, or the whole warp scores one, its lanes splitting the outer cloud's points. A
 *  lane alone wins only on clouds of a few points, which a pack's bytes cannot tell apart from
 *  larger ones, so each form serves every cloud and the index picks its form. The index takes the
 *  warp form, a block walking each query while its four warps score a candidate each.
 *
 *  Any compiler reads a kit's host facts, the rows it serves, the span a row prepares into and the
 *  host kernels that pack a row before upload, so the host index picks kits, sizes spans and packs
 *  rows from the kits themselves; the device members compile only where CUDA does.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/capabilities.h> // `nk_kernel_chamfers_packed_k` & kin - the kits' names in the kernels
#include <numkong/shapes.h>       // `nk_shapes_pack_f16_serial` & kin, which `shapes/serial.h` expects declared ahead

#include "usearch/utilities/types.hpp"

#if defined(__CUDACC__)
#include <numkong/shapes/cuda.cuh> // `nk_chamfer_warp_packed_f16_cuda_` & kin, and the lane bodies of `simt.cuh`

#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#endif

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

/** The integer naming a kit in the kernels: NumKong's dtype, its kernel kind and the lanes per row
 *  in fields of their own, so no two kits of any modality share one. */
constexpr u32_t vram_kit_name(nk_dtype_t dtype, nk_kernel_kind_t kernel, u32_t lanes_per_row) noexcept {
    return (static_cast<u32_t>(dtype) << 16) | (static_cast<u32_t>(kernel) << 8) | lanes_per_row;
}

#if defined(__CUDACC__)

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

#endif

/** Chamfer between f16 packs in f32 math, a lane or a warp to a row by @p lanes_per_row_. */
template <u32_t lanes_per_row_>
struct f16_ampere_chamfer_kit {
    inline static constexpr u32_t kit_k = vram_kit_name(nk_f16_k, nk_kernel_chamfers_packed_k, lanes_per_row_);
    inline static constexpr u32_t lanes_per_row_k = lanes_per_row_;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_serial;

    static constexpr bool serves(std::size_t) noexcept { return true; }

    /** Bytes the pack at @p data takes once prepared: the pack itself. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

#if defined(__CUDACC__)
    using warp_t = cuda_warp_t;
    using query_t = vram_pack_query_t;

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return vram_prepare_pack(row, span, lane);
    }

    /** No header bounds a Chamfer distance. */
    static __device__ __forceinline__ distance_t bound(query_t const&, basic::vram_row_t) noexcept { return 0; }

    static __device__ __forceinline__ distance_t score(query_t const& query, basic::vram_row_t row, distance_t radius,
                                                       u32_t lane) noexcept {
        if constexpr (lanes_per_row_ == 1) return nk_chamfer_packed_f16_simt_(query.pack, row.data, radius);
        else return nk_chamfer_warp_packed_f16_cuda_(query.pack, row.data, radius, lane);
    }
#endif
};

/** Hausdorff between f16 packs in f32 math, a lane or a warp to a row by @p lanes_per_row_. */
template <u32_t lanes_per_row_>
struct f16_ampere_hausdorff_kit {
    inline static constexpr u32_t kit_k = vram_kit_name(nk_f16_k, nk_kernel_hausdorffs_packed_k, lanes_per_row_);
    inline static constexpr u32_t lanes_per_row_k = lanes_per_row_;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_serial;

    static constexpr bool serves(std::size_t) noexcept { return true; }
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

#if defined(__CUDACC__)
    using warp_t = cuda_warp_t;
    using query_t = vram_pack_query_t;

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return vram_prepare_pack(row, span, lane);
    }

    /** The gap between the two clouds' largest radii, from their headers alone. */
    static __device__ __forceinline__ distance_t bound(query_t const& query, basic::vram_row_t row) noexcept {
        return nk_hausdorff_bound_packed_f16_simt_(query.pack, row.data);
    }

    static __device__ __forceinline__ distance_t score(query_t const& query, basic::vram_row_t row, distance_t radius,
                                                       u32_t lane) noexcept {
        if constexpr (lanes_per_row_ == 1) return nk_hausdorff_packed_f16_simt_(query.pack, row.data, radius);
        else return nk_hausdorff_warp_packed_f16_cuda_(query.pack, row.data, radius, lane);
    }
#endif
};

/** Chamfer between f32 packs in f64 math, a lane or a warp to a row by @p lanes_per_row_. */
template <u32_t lanes_per_row_>
struct f32_ampere_chamfer_kit {
    inline static constexpr u32_t kit_k = vram_kit_name(nk_f32_k, nk_kernel_chamfers_packed_k, lanes_per_row_);
    inline static constexpr u32_t lanes_per_row_k = lanes_per_row_;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_serial;

    static constexpr bool serves(std::size_t) noexcept { return true; }
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

#if defined(__CUDACC__)
    using warp_t = cuda_warp_t;
    using query_t = vram_pack_query_t;

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return vram_prepare_pack(row, span, lane);
    }
    static __device__ __forceinline__ distance_t bound(query_t const&, basic::vram_row_t) noexcept { return 0; }
    static __device__ __forceinline__ distance_t score(query_t const& query, basic::vram_row_t row, distance_t radius,
                                                       u32_t lane) noexcept {
        if constexpr (lanes_per_row_ == 1)
            return static_cast<distance_t>(nk_chamfer_packed_f32_simt_(query.pack, row.data, radius));
        else return static_cast<distance_t>(nk_chamfer_warp_packed_f32_cuda_(query.pack, row.data, radius, lane));
    }
#endif
};

/** Hausdorff between f32 packs in f64 math, a lane or a warp to a row by @p lanes_per_row_. */
template <u32_t lanes_per_row_>
struct f32_ampere_hausdorff_kit {
    inline static constexpr u32_t kit_k = vram_kit_name(nk_f32_k, nk_kernel_hausdorffs_packed_k, lanes_per_row_);
    inline static constexpr u32_t lanes_per_row_k = lanes_per_row_;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_serial;

    static constexpr bool serves(std::size_t) noexcept { return true; }
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

#if defined(__CUDACC__)
    using warp_t = cuda_warp_t;
    using query_t = vram_pack_query_t;

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return vram_prepare_pack(row, span, lane);
    }
    static __device__ __forceinline__ distance_t bound(query_t const& query, basic::vram_row_t row) noexcept {
        return static_cast<distance_t>(nk_hausdorff_bound_packed_f32_simt_(query.pack, row.data));
    }
    static __device__ __forceinline__ distance_t score(query_t const& query, basic::vram_row_t row, distance_t radius,
                                                       u32_t lane) noexcept {
        if constexpr (lanes_per_row_ == 1)
            return static_cast<distance_t>(nk_hausdorff_packed_f32_simt_(query.pack, row.data, radius));
        else return static_cast<distance_t>(nk_hausdorff_warp_packed_f32_cuda_(query.pack, row.data, radius, lane));
    }
#endif
};

} // namespace unum::usearch::shapes
