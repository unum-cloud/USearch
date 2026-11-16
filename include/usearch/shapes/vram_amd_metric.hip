/**
 *  @file include/usearch/shapes/vram_amd_metric.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Device kits for the shapes VRAM indexes on AMD GPUs, the GPU twin of @c ram_metric.hpp,
 *      their ladder, the kernels, and the members of @c vram_amd_index.hpp reading them.
 *
 *  A kit binds the shapes distances over one scalar's packs to one AMD generation, named like the
 *  dense kits: @c cdna3 kits run on the 64-lane wavefronts of gfx942 and gfx950, @c cdna5 kits on
 *  the 32-lane wavefronts of gfx1250 and gfx1251. The ROCm SIMT bodies score two packs in place,
 *  f16 packs in f32 math and f32 packs in f64 math as the CPU and NVIDIA kits do, and stop once a
 *  candidate lies past the walk's radius. One kit covers both metrics, picked at run time by the
 *  kernel kind the view carries, as dense quads kits pick their metric.
 *
 *  A stored shape is its pack byte for byte, and a walk prepares one by copying it into its span,
 *  where the bodies read the same pack as in the tape. A team of 32 lanes scores one candidate, its
 *  lanes splitting the outer cloud's points: a whole CDNA5 wavefront, or either half of a CDNA3
 *  one, whose shuffles under 32 lanes stay within the half, which lets one 64-lane wavefront score
 *  two candidates at once. A block walks each query while its four wavefronts score the candidates,
 *  and each AMD class unit compiles this file under its own ceiling, for dispatch to name a class
 *  and never a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <iterator>    // `std::size` - every rung's kernels
#include <type_traits> // `std::is_same_v` - a load's scalar
#include <utility>     // `std::index_sequence` - every rung's kernels

#include <numkong/shapes.h>        // `nk_shapes_pack_f16_serial` & kin, which `shapes/serial.h` expects declared ahead
#include <numkong/shapes/rocm.cuh> // `nk_chamfer_warp_packed_f16_rocm_` & kin
#include <numkong/types.hpp>       // `nk::type_for` - a kernel's scalar

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"     // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_amd_device.hip" // `hip_warp`, `cdna3_scan_t`, `cdna5_scan_t`
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_amd_index.hip" // `basic::hip_kernel_body`
#include "usearch/shapes/ram_metric.hpp"    // `shapes_ladder`
#include "usearch/shapes/ram_index.hpp"     // `codec::save_pack`, `codec::load_pack` - the pose conversion
#include "usearch/shapes/vram_amd_index.hpp"

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

#pragma region Kits

/** The pack a walk scores against, in the walking warp's span. */
struct vram_pack_query_t {
    byte_t const* pack;
};

/** Lays the pack @p shape holds out in @p span, every lane of the @p warp_type_ wavefront copying
 *  16 bytes at a time; packs are whole alignment lines, so no tail remains. */
template <typename warp_type_>
__device__ __forceinline__ vram_pack_query_t hip_prepare_pack(basic::vram_entry_t shape, byte_t* span,
                                                              u32_t lane) noexcept {
    using line_t = uint4;
    line_t const* const source = reinterpret_cast<line_t const*>(shape.data);
    line_t* const target = reinterpret_cast<line_t*>(span);
    for (u32_t line = lane; line < shape.bytes / sizeof(line_t); line += warp_type_::width_k)
        target[line] = source[line];
    warp_type_::sync();
    return {span};
}

/** Packs every cloud of @p load 's run into the tape, a team of 32 lanes per cloud, laid out as the
 *  serial pack lays one; the saved sums count scalars, three for every point. */
template <typename scalar_type_, typename node_id_type_>
__device__ __forceinline__ void hip_shapes_load_values(basic::vram_load<node_id_type_> const& load) noexcept {
    vram_grid_warp_t const team = vram_grid_warp<hip_warp<32>>();
    u64_t const first = load.saved_offsets[0];
    for (u32_t node = team.index; node < load.count; node += team.count) {
        scalar_type_ const* const points = reinterpret_cast<scalar_type_ const*>(load.saved) +
                                           (load.saved_offsets[node] - first);
        nk_size_t const count = static_cast<nk_size_t>(load.saved_offsets[node + 1] - load.saved_offsets[node]) / 3;
        void* const packed = load.tape + load.stored_offsets[node];
        if constexpr (std::is_same_v<scalar_type_, nk_f32_t>) nk_shapes_pack_f32_rocm_(points, count, packed);
        else nk_shapes_pack_f16_rocm_(points, count, packed);
    }
}

/** Chamfer and Hausdorff between f16 packs in f32 math on the 64-lane wavefronts of CDNA3 and
 *  CDNA4, two teams to a wavefront. */
struct f16_cdna3_shapes_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "f16_cdna3_shapes_kit_t";
    inline static constexpr u32_t lanes_per_entry_k = 32;

    using warp_t = hip_warp<64>;
    using scan_t = cdna3_scan_t;
    using query_t = vram_pack_query_t;
    using selector_t = basic::vram_keep_diverse_t;
    using scalar_t = nk::f16_t;

    static constexpr bool serves(std::size_t) noexcept { return true; }

    /** Packs a load's clouds into the tape on the device. */
    template <typename node_id_type_>
    static __device__ __forceinline__ void load_values(basic::vram_load<node_id_type_> const& load) noexcept {
        hip_shapes_load_values<nk_f16_t>(load);
    }

    /** Bytes the pack at @p data takes once prepared: the pack itself. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_entry_t shape, byte_t* span, u32_t lane) noexcept {
        return hip_prepare_pack<warp_t>(shape, span, lane);
    }

    /** The gap between the two clouds' largest radii, from their headers alone, bounds Hausdorff;
     *  nothing bounds Chamfer. */
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t kernel, query_t const& query,
                                                             basic::vram_entry_t shape) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k ? nk_hausdorff_bound_packed_f16_simt_(query.pack, shape.data)
                                                       : 0;
    }

    /** Scores on the team holding @p lane, whose lanes split the outer cloud's points. */
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_entry_t shape, distance_t radius,
                                                       u32_t lane) noexcept {
        u32_t const team_lane = lane % lanes_per_entry_k;
        return kernel == nk_kernel_hausdorffs_packed_k
                   ? nk_hausdorff_warp_packed_f16_rocm_(query.pack, shape.data, radius, team_lane)
                   : nk_chamfer_warp_packed_f16_rocm_(query.pack, shape.data, radius, team_lane);
    }
};

/** Chamfer and Hausdorff between f16 packs in f32 math on the 32-lane wavefronts of CDNA5, a team
 *  to a wavefront. */
struct f16_cdna5_shapes_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "f16_cdna5_shapes_kit_t";
    inline static constexpr u32_t lanes_per_entry_k = 32;

    using warp_t = hip_warp<32>;
    using scan_t = cdna5_scan_t;
    using query_t = vram_pack_query_t;
    using selector_t = basic::vram_keep_diverse_t;
    using scalar_t = nk::f16_t;

    static constexpr bool serves(std::size_t) noexcept { return true; }

    /** Packs a load's clouds into the tape on the device. */
    template <typename node_id_type_>
    static __device__ __forceinline__ void load_values(basic::vram_load<node_id_type_> const& load) noexcept {
        hip_shapes_load_values<nk_f16_t>(load);
    }

    /** Bytes the pack at @p data takes once prepared: the pack itself. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_entry_t shape, byte_t* span, u32_t lane) noexcept {
        return hip_prepare_pack<warp_t>(shape, span, lane);
    }

    /** The gap between the two clouds' largest radii, from their headers alone, bounds Hausdorff;
     *  nothing bounds Chamfer. */
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t kernel, query_t const& query,
                                                             basic::vram_entry_t shape) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k ? nk_hausdorff_bound_packed_f16_simt_(query.pack, shape.data)
                                                       : 0;
    }

    /** Scores on the team holding @p lane, whose lanes split the outer cloud's points. */
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_entry_t shape, distance_t radius,
                                                       u32_t lane) noexcept {
        u32_t const team_lane = lane % lanes_per_entry_k;
        return kernel == nk_kernel_hausdorffs_packed_k
                   ? nk_hausdorff_warp_packed_f16_rocm_(query.pack, shape.data, radius, team_lane)
                   : nk_chamfer_warp_packed_f16_rocm_(query.pack, shape.data, radius, team_lane);
    }
};

/** Chamfer and Hausdorff between f32 packs in f64 math on the 64-lane wavefronts of CDNA3 and
 *  CDNA4, two teams to a wavefront. */
struct f32_cdna3_shapes_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "f32_cdna3_shapes_kit_t";
    inline static constexpr u32_t lanes_per_entry_k = 32;

    using warp_t = hip_warp<64>;
    using scan_t = cdna3_scan_t;
    using query_t = vram_pack_query_t;
    using selector_t = basic::vram_keep_diverse_t;
    using scalar_t = nk::f32_t;

    static constexpr bool serves(std::size_t) noexcept { return true; }

    /** Packs a load's clouds into the tape on the device. */
    template <typename node_id_type_>
    static __device__ __forceinline__ void load_values(basic::vram_load<node_id_type_> const& load) noexcept {
        hip_shapes_load_values<nk_f32_t>(load);
    }

    /** Bytes the pack at @p data takes once prepared: the pack itself. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_entry_t shape, byte_t* span, u32_t lane) noexcept {
        return hip_prepare_pack<warp_t>(shape, span, lane);
    }

    /** The gap between the two clouds' largest radii, from their headers alone, bounds Hausdorff;
     *  nothing bounds Chamfer. */
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t kernel, query_t const& query,
                                                             basic::vram_entry_t shape) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k
                   ? static_cast<distance_t>(nk_hausdorff_bound_packed_f32_simt_(query.pack, shape.data))
                   : 0;
    }

    /** Scores on the team holding @p lane, whose lanes split the outer cloud's points. */
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_entry_t shape, distance_t radius,
                                                       u32_t lane) noexcept {
        u32_t const team_lane = lane % lanes_per_entry_k;
        return static_cast<distance_t>(
            kernel == nk_kernel_hausdorffs_packed_k
                ? nk_hausdorff_warp_packed_f32_rocm_(query.pack, shape.data, radius, team_lane)
                : nk_chamfer_warp_packed_f32_rocm_(query.pack, shape.data, radius, team_lane));
    }
};

/** Chamfer and Hausdorff between f32 packs in f64 math on the 32-lane wavefronts of CDNA5, a team
 *  to a wavefront. */
struct f32_cdna5_shapes_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "f32_cdna5_shapes_kit_t";
    inline static constexpr u32_t lanes_per_entry_k = 32;

    using warp_t = hip_warp<32>;
    using scan_t = cdna5_scan_t;
    using query_t = vram_pack_query_t;
    using selector_t = basic::vram_keep_diverse_t;
    using scalar_t = nk::f32_t;

    static constexpr bool serves(std::size_t) noexcept { return true; }

    /** Packs a load's clouds into the tape on the device. */
    template <typename node_id_type_>
    static __device__ __forceinline__ void load_values(basic::vram_load<node_id_type_> const& load) noexcept {
        hip_shapes_load_values<nk_f32_t>(load);
    }

    /** Bytes the pack at @p data takes once prepared: the pack itself. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(bytes, 16);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_entry_t shape, byte_t* span, u32_t lane) noexcept {
        return hip_prepare_pack<warp_t>(shape, span, lane);
    }

    /** The gap between the two clouds' largest radii, from their headers alone, bounds Hausdorff;
     *  nothing bounds Chamfer. */
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t kernel, query_t const& query,
                                                             basic::vram_entry_t shape) noexcept {
        return kernel == nk_kernel_hausdorffs_packed_k
                   ? static_cast<distance_t>(nk_hausdorff_bound_packed_f32_simt_(query.pack, shape.data))
                   : 0;
    }

    /** Scores on the team holding @p lane, whose lanes split the outer cloud's points. */
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t kernel, query_t const& query,
                                                       basic::vram_entry_t shape, distance_t radius,
                                                       u32_t lane) noexcept {
        u32_t const team_lane = lane % lanes_per_entry_k;
        return static_cast<distance_t>(
            kernel == nk_kernel_hausdorffs_packed_k
                ? nk_hausdorff_warp_packed_f32_rocm_(query.pack, shape.data, radius, team_lane)
                : nk_chamfer_warp_packed_f32_rocm_(query.pack, shape.data, radius, team_lane));
    }
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct shapes_ladder<nk::f16_t, caps_> {
    using kit_t = typename highest_kit<caps_, f16_cdna3_shapes_kit_t, f16_cdna5_shapes_kit_t>::type;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct shapes_ladder<nk::f32_t, caps_> {
    using kit_t = typename highest_kit<caps_, f32_cdna3_shapes_kit_t, f32_cdna5_shapes_kit_t>::type;
};

#pragma endregion Kits

#pragma region Face

/** Spells a shapes kit into a kernel's symbol by its name, inside the kernel's own scope, @c S1_,
 *  as @c vram_kernel_symbol documents. */
template <typename kit_type_>
    requires requires { kit_type_::name_k; }
void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<kit_type_>) noexcept {
    symbol.append("NS1_");
    symbol.append_name(kit_type_::name_k);
    symbol.append("E");
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::rung_serving_(std::size_t longest) noexcept {
    return kit_type_::serves(longest) ? 0 : rungs_k;
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
u32_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::lanes_per_entry_(std::size_t) noexcept {
    return kit_type_::lanes_per_entry_k;
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::span_bytes_(byte_t const* data,
                                                                                std::size_t bytes) noexcept {
    return kit_type_::span_bytes(data, bytes);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
expected<std::size_t> vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::saved_bytes_(
    bytes_view_t stored) noexcept {
    using pack_kit_t = typename shapes_ladder<typename kit_type_::scalar_t, nk_cap_serial_k>::kit_t;
    nk_size_t points = 0;
    if (status_t const read = status_of(
            pack_kit_t::packed_shape(stored.data(), &points, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
        failed(read))
        return {0, read};
    return {points * 3 * sizeof(typename kit_type_::scalar_t::raw_t), status_t::success_k};
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::save_value_(bytes_view_t stored,
                                                                             bytes_span_t saved) noexcept {
    using pack_kit_t = typename shapes_ladder<typename kit_type_::scalar_t, nk_cap_serial_k>::kit_t;
    return codec<pack_kit_t>::save_pack(stored, saved);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
expected<std::size_t> vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::stored_bytes_(
    std::size_t saved) noexcept {
    using pack_kit_t = typename shapes_ladder<typename kit_type_::scalar_t, nk_cap_serial_k>::kit_t;
    constexpr std::size_t point_bytes_k = 3 * sizeof(typename kit_type_::scalar_t::raw_t);
    if (saved % point_bytes_k) return {0, status_t::malformed_k};
    nk_size_t bytes = 0;
    if (status_t const sized = status_of(pack_kit_t::pack_size(saved / point_bytes_k, &bytes)); failed(sized))
        return {0, sized};
    return {bytes, status_t::success_k};
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
void vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::spell_kits_kernel_(
    vram_kernel_symbol_t& symbol, std::size_t, basic::amd_rung_t instantiated) noexcept {
    vram_kernel_symbol(symbol, "shapes", "amd_kernel", {"vram_arguments", 2, itanium_scope_t::basic_k},
                       instantiated.kernel, itanium_type<kit_type_> {}, itanium_type<node_id_type_> {},
                       instantiated.slots);
}

/** Every AMD shapes kernel: @p kernel_ picks the walk, @p kit_type_ scores, @p node_id_type_ names
 *  the IDs of the records, and @p slots_ the register rung. A block is @c vram_warps_per_block_k of
 *  the kit's wavefronts, and the waves each SIMD keeps resident follow from the blocks
 *  @c hip_resident_blocks keeps per compute unit. */
template <vram_kernel_t kernel_, typename kit_type_, typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k* kit_type_::warp_t::width_k,
                                  basic::hip_resident_blocks(kernel_, slots_) * vram_warps_per_block_k / 4)
    amd_kernel(basic::vram_arguments<node_id_type_> arguments) {
    basic::hip_kernel_body<kernel_, kit_type_, 0, slots_>(arguments);
}

template <typename kit_type_, typename node_id_type_, std::size_t... rungs_>
void amd_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(
         &amd_kernel<basic::amd_rungs[rungs_].kernel, kit_type_, node_id_type_, basic::amd_rungs[rungs_].slots>),
     ...);
}

/** Instantiates every kernel of @p kit_type_ with @p node_id_type_ IDs; the unit compiling the code
 *  object names it once per kit and node-ID type. It takes the addresses on the host, since
 *  HIP-Clang refuses a kernel's address in device code while a host reference still makes the
 *  device pass emit it; only a device-only unit names it, so the host pass emits no launch stub. */
template <typename kit_type_, typename node_id_type_>
void amd_instantiate() noexcept {
    amd_instantiate_rungs<kit_type_, node_id_type_>(std::make_index_sequence<std::size(basic::amd_rungs)> {});
    static_cast<void>(&amd_kernel<vram_kernel_t::load_values_k, kit_type_, node_id_type_, 0>);
    static_cast<void>(&amd_kernel<vram_kernel_t::size_spans_k, kit_type_, node_id_type_, 0>);
}

#pragma endregion Face

} // namespace unum::usearch::shapes
