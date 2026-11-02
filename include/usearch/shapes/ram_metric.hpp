/**
 *  @file include/usearch/shapes/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Batched shape distances between 3D point clouds of any size, over NumKong's many-to-many
 *      kernels.
 *
 *  The shapes index scores a quad at a time: @c reset opens it for a query pack, @c push stages up
 *  to a quad of candidate packs, and @c drain scores them in one kernel call. A metric here owns
 *  its arena, so one instance serves one thread and allocates on the scoring path only when a pair
 *  is wider than every pair it has seen.
 *
 *  @section shapes_ram_metric_kits Kits
 *
 *  A kit binds one ISA's four kernels, Chamfer, Hausdorff, Wasserstein-Procrustes and
 *  Gromov-Wasserstein, with their three workspace sizes, its pack kernel, the unpack and
 *  pack-reading kernels, which every ISA shares, and the cast that brings a caller's coordinates
 *  into the stored scalar and back out. @c shapes_ladder picks the widest kit this build compiled
 *  in under the requested capability mask, at compile time, so no capability is tested at run time.
 *
 *  @section shapes_ram_metric_bounds Bounds
 *
 *  A pack keeps the descending eigenvalues of its points' second moments and its largest point
 *  norm, and those bound two kinds the way norms bound dense distances:
 *
 *  - Wasserstein-Procrustes ≥ √Σₖ (√λq,ₖ − √λs,ₖ)², the Gelbrich bound minimized over rotations and
 *    reflections
 *  - Hausdorff ≥ |Rq − Rs|, since the farthest point of one cloud is at least that far from every
 *    point of the other
 *
 *  Both hold for the exact definitions and are symmetric. Chamfer and Hausdorff kernels take the
 *  round's radius and apply these themselves, as the GPU metric's do, while the transport kinds
 *  score every candidate exactly.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <limits> // `std::numeric_limits` - the distance a failed round reports
#include <span>   // `std::span`

#include <numkong/cast.h>     // `nk_cast_serial` & kin - the kits' scalar casts
#include <numkong/shapes.hpp> // `nk::shapes_packed`, and the per-backend kernels
#include <numkong/tensor.hpp> // `nk::matrix_view` - the staged clouds a round packs

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator_t`
#include "usearch/basic/ram_index.hpp"      // `basic::metric_staging_t`

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

/**
 *  @brief The USearch status of a NumKong one.
 *
 *  @par Statuses:
 *  - @c shape_mismatch_k NumKong found unexpected dimensions.
 *  - @c device_error_k NumKong found device code or memory of another device.
 *  - @c unsupported_k NumKong is missing the kernel or its library.
 *  - @c unsupported_layout_k NumKong found misaligned memory.
 *  - @c malformed_k NumKong found a pack of another layout.
 */
inline status_t status_of(nk_status_t status) noexcept {
    switch (status) {
    case nk_success_k: return status_t::success_k;
    case nk_bad_alloc_k: return status_t::bad_alloc_k;
    case nk_unexpected_dimensions_k: return status_t::shape_mismatch_k;
    case nk_missing_gpu_k:
    case nk_device_code_mismatch_k:
    case nk_device_memory_mismatch_k: return status_t::device_error_k;
    case nk_missing_kernel_k:
    case nk_missing_library_k: return status_t::unsupported_k;
    case nk_misaligned_k: return status_t::unsupported_layout_k;
    case nk_pack_mismatch_k: return status_t::malformed_k;
    default: return status_t::internal_k;
    }
}

/**
 *  @brief The USearch status of a NumKong C++ one, whose codes are the C ones.
 *
 *  @par Statuses:
 *  - @c shape_mismatch_k NumKong found unexpected dimensions.
 *  - @c device_error_k NumKong found device code or memory of another device.
 *  - @c unsupported_k NumKong is missing the kernel or its library.
 *  - @c unsupported_layout_k NumKong found misaligned memory.
 *  - @c malformed_k NumKong found a pack of another layout.
 */
inline status_t status_of(nk::status_t status) noexcept { return status_of(static_cast<nk_status_t>(status)); }

#pragma region Kits

/** The f32 shape kernels for any CPU. */
struct f32_serial_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f32_t;
    using distance_t = nk_f64_t;
    static constexpr auto cast = nk_cast_serial;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_serial;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_serial;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_serial;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f32_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f32_serial;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f32_serial;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f32_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f32_serial;
};

#if NUMKONG_TARGET_NEON

/** The f32 shape kernels for Arm NEON. */
struct f32_neon_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f32_t;
    using distance_t = nk_f64_t;
    static constexpr auto cast = nk_cast_neon;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_neon;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_neon;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_neon;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f32_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f32_neon;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f32_neon;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f32_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f32_neon_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL

/** The f32 shape kernels for Intel Haswell AVX2. */
struct f32_haswell_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f32_t;
    using distance_t = nk_f64_t;
    static constexpr auto cast = nk_cast_haswell;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_haswell;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_haswell;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_haswell;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f32_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f32_haswell;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f32_haswell;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f32_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f32_haswell_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** The f32 shape kernels for Intel Skylake-X AVX-512. */
struct f32_skylake_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f32_t;
    using distance_t = nk_f64_t;
    static constexpr auto cast = nk_cast_skylake;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_skylake;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_skylake;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_skylake;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f32_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f32_skylake;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f32_skylake;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f32_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f32_skylake_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV

/** The f32 shape kernels for the RISC-V Vector extension. */
struct f32_rvv_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f32_t;
    using distance_t = nk_f64_t;
    static constexpr auto cast = nk_cast_rvv;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_rvv;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_rvv;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_rvv;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f32_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f32_rvv;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f32_rvv;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f32_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f32_rvv_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED

/** The f32 shape kernels for WebAssembly Relaxed SIMD. */
struct f32_v128relaxed_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f32_t;
    using distance_t = nk_f64_t;
    static constexpr auto cast = nk_cast_v128relaxed;
    static constexpr auto pack_size = nk_shapes_pack_size_f32_serial;
    static constexpr auto pack = nk_shapes_pack_f32_v128relaxed;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_v128relaxed;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_v128relaxed;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f32_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f32_v128relaxed;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f32_v128relaxed;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f32_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f32_v128relaxed_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** The f16 shape kernels for any CPU. */
struct f16_serial_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_serial;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_serial;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_serial;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_serial;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_serial;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};

#if NUMKONG_TARGET_NEON

/** The f16 shape kernels for Arm NEON. */
struct f16_neon_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_neon;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_neon;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_neon;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_neon;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_neon;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_neon;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f16_neon_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONFHM

/** The f16 shape kernels for Arm NEON with FP16 multiply-accumulate: its own nearest-point kinds
 *  beside NEON's transport kinds. */
struct f16_neonfhm_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_neon;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_neon;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_neonfhm;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_neonfhm;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_neon;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_neon;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f16_neonfhm_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_HASWELL

/** The f16 shape kernels for Intel Haswell AVX2. */
struct f16_haswell_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_haswell;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_haswell;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_haswell;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_haswell;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_haswell;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_haswell;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f16_haswell_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** The f16 shape kernels for Intel Skylake-X AVX-512. */
struct f16_skylake_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_skylake;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_skylake;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_skylake;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_skylake;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_skylake;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_skylake;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f16_skylake_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV

/** The f16 shape kernels for the RISC-V Vector extension. */
struct f16_rvv_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_rvv;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_rvv;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_rvv;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_rvv;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_rvv;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_rvv;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f16_rvv_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED

/** The f16 shape kernels for WebAssembly Relaxed SIMD. */
struct f16_v128relaxed_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    using scalar_t = nk::f16_t;
    using distance_t = nk_f32_t;
    static constexpr auto cast = nk_cast_v128relaxed;
    static constexpr auto pack_size = nk_shapes_pack_size_f16_serial;
    static constexpr auto pack = nk_shapes_pack_f16_v128relaxed;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_v128relaxed;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_v128relaxed;
    static constexpr auto chamfers_workspace_size = nk_chamfers_workspace_size_f16_serial;
    static constexpr auto wassersteinprocrustes = nk_wassersteinprocrustes_packed_f16_v128relaxed;
    static constexpr auto gromovwassersteins = nk_gromovwassersteins_packed_f16_v128relaxed;
    static constexpr auto wassersteinprocrustes_workspace_size = nk_wassersteinprocrustes_workspace_size_f16_serial;
    static constexpr auto gromovwassersteins_workspace_size = nk_gromovwassersteins_workspace_size_f16_serial;
};
#else

/** The kit this build cannot carry, kept so @c shapes_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct f16_v128relaxed_shapes_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** The widest shapes kit of @p scalar_ this build compiled in among the backends @p caps_ admits,
 *  newest first; each kit states its @c capability_k and its @c compiled_k, so an unselected kit's
 *  body never has to exist. Only the scalars some kit stores specialize it. */
template <nk::shapes_dtype scalar_, nk_capability_t caps_>
struct shapes_ladder;

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cpus_k) != 0)
struct shapes_ladder<nk::f16_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        f16_serial_shapes_kit_t,        //
        f16_skylake_shapes_kit_t,       //
        f16_haswell_shapes_kit_t,       //
        f16_neonfhm_shapes_kit_t,       //
        f16_neon_shapes_kit_t,          //
        f16_rvv_shapes_kit_t,           //
        f16_v128relaxed_shapes_kit_t    //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cpus_k) != 0)
struct shapes_ladder<nk::f32_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        f32_serial_shapes_kit_t,        //
        f32_skylake_shapes_kit_t,       //
        f32_haswell_shapes_kit_t,       //
        f32_neon_shapes_kit_t,          //
        f32_rvv_shapes_kit_t,           //
        f32_v128relaxed_shapes_kit_t    //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Kits

#pragma region Metric

/**
 *  @brief One of four shape kinds over one quad of candidates: @c reset borrows the query
 *      pack, @c push stages up to a quad and then refuses, and @c drain scores what staged.
 *
 *  A borrowed pack stays valid until the drain that closes its quad, as the metric concept
 *  promises; a @b [points,3] cloud is packed into @c round_arena_, which every drain rewinds.
 */
template <typename kit_type_, typename allocator_type_ = forkunion_allocator_t>
class shapes_metric {
  public:
    using kit_t = kit_type_;
    using scalar_t = typename kit_t::scalar_t;
    using distance_t = typename kit_t::distance_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** Which of the four shape kernels scores the quads. */
    nk_kernel_kind_t kernel_;

    /** A batched @c reset's query packs, gathered into one table the kernels read. */
    growing_buffer<void const*, allocator_t> queries_;

    /** The table the kernels read: a single query's own, or @c queries_ for a batch. */
    std::span<void const* const> query_packs_;

    /** The quad's candidate packs in push order. */
    quad<void const*> staged_packs_;

    /** How many of @c staged_packs_ this quad filled. */
    std::size_t staged_count_ = 0;

    /** The most points in the query or any candidate since @c reset, which sizes the workspace. */
    std::size_t widest_points_ = 0;

    /** The packs of staged clouds and the kernels' workspace, rewound by every drain. */
    growing_arena<allocator_t> round_arena_;

    /** How the last @ref reset or @ref drain fared. */
    status_t last_status_ = status_t::success_k;

  public:
    explicit shapes_metric(nk_kernel_kind_t kernel, allocator_t const& allocator) noexcept
        : kernel_(kernel), queries_(allocator), round_arena_(allocator) {}
    shapes_metric(shapes_metric const&) = delete;
    shapes_metric& operator=(shapes_metric const&) = delete;
    shapes_metric(shapes_metric&&) noexcept = default;
    shapes_metric& operator=(shapes_metric&&) noexcept = default;

    /** How the last @ref reset or @ref drain fared, the only place a failure surfaces, since the
     *  scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a quad for @p query, which must hold exactly one cloud. */
    template <typename query_allocator_type_>
    void reset(nk::shapes_packed<scalar_t, query_allocator_type_> const& query) noexcept {
        staged_count_ = 0, round_arena_.clear();
        last_status_ = query.size() == 1 ? status_t::success_k : status_t::shape_mismatch_k;
        if (failed(last_status_)) return;
        query_packs_ = to_span(query), widest_points_ = query.max_points();
    }

    /** Opens rounds scoring every query of @p queries at once, each holding exactly one cloud, so
     *  each @ref drain writes @b [queries,candidates]. */
    template <typename query_allocator_type_>
    void reset(std::span<nk::shapes_packed<scalar_t, query_allocator_type_> const> queries) noexcept {
        if (queries.size() == 1) return reset(queries.front());
        staged_count_ = 0, round_arena_.clear(), queries_.clear(), widest_points_ = 0;
        last_status_ = queries_.reserve(queries.size());
        for (std::size_t query = 0; succeeded(last_status_) && query != queries.size(); ++query) {
            if (queries[query].size() != 1) last_status_ = status_t::shape_mismatch_k;
            else {
                queries_.push_back_reserved(queries[query][0]);
                widest_points_ = max_of(widest_points_, queries[query].max_points());
            }
        }
        query_packs_ = to_span(queries_);
    }

    /** Stages one candidate's pack, or refuses once the quad is full. */
    template <typename value_allocator_type_>
    basic::metric_staging_t push(nk::shapes_packed<scalar_t, value_allocator_type_> const& candidate) noexcept {
        if (staged_count_ == staged_packs_.size()) return basic::metric_staging_t::refused_k;
        staged_packs_[staged_count_++] = candidate[0];
        widest_points_ = max_of(widest_points_, candidate.max_points());
        return basic::metric_staging_t::staged_k;
    }

    /** Packs a @b [points,3] @p cloud into the arena and stages it, or refuses once the quad is
     *  full. A pack that fails leaves the quad failed, as @ref status reports. */
    basic::metric_staging_t push(nk::matrix_view<scalar_t> cloud) noexcept {
        if (staged_count_ == staged_packs_.size()) return basic::metric_staging_t::refused_k;
        std::size_t const points = cloud.extent(0);
        nk_size_t pack_bytes = 0;
        if (nk_status_t const sized = kit_t::pack_size(points, &pack_bytes); sized != nk_success_k)
            return last_status_ = status_of(sized), basic::metric_staging_t::refused_k;
        void* const pack = round_arena_.allocate(pack_bytes, nk_default_alignment_k);
        if (!pack) return last_status_ = status_t::bad_alloc_k, basic::metric_staging_t::refused_k;
        if (status_t const packed = status_of(kit_t::pack(cloud.raw_values_data(), points, pack, nullptr));
            failed(packed))
            last_status_ = packed;
        staged_packs_[staged_count_++] = pack;
        widest_points_ = max_of(widest_points_, points);
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores the quad within @p radius into @p distances and closes it.
     *
     *  A failed quad reports the largest finite distance, so full candidates reject it and unfilled
     *  ones admit it last; @ref status is the only signal.
     *
     *  @param[out] distances One slot per staged candidate, in push order, for every query,
     *      query-major.
     *  @param[in] radius Chamfer and Hausdorff distances up to it are exact; past it, any value
     *      past it. The transport kinds score every candidate exactly.
     */
    void drain(std::span<distance_t> distances, distance_t radius) noexcept {
        if (succeeded(last_status_) && staged_count_) last_status_ = score_(distances.data(), radius);
        if (failed(last_status_))
            for (distance_t& distance : distances) distance = (std::numeric_limits<distance_t>::max)();
        staged_count_ = 0, round_arena_.clear();
    }

  private:

    /** Scores every query against the quad in one call into @p results, @b [queries,candidates],
     *  with the kernel of the metric's one kind. */
    status_t score_(distance_t* results, distance_t radius) noexcept {
        void const* const* const queries = query_packs_.data();
        void const* const* const candidates = staged_packs_.data();
        std::size_t const queries_count = query_packs_.size(), count = staged_count_;
        std::size_t const stride = count * sizeof(distance_t);
        if (kernel_ == nk_kernel_hausdorffs_packed_k)
            return status_of(kit_t::hausdorffs(queries, queries_count, candidates, count, results, stride, nullptr,
                                               &radius, nullptr));

        nk_size_t bytes = 0;
        nk_status_t const sized =
            kernel_ == nk_kernel_chamfers_packed_k
                ? kit_t::chamfers_workspace_size(widest_points_, widest_points_, &bytes)
            : kernel_ == nk_kernel_wassersteinprocrustes_packed_k
                ? kit_t::wassersteinprocrustes_workspace_size(widest_points_, widest_points_, &bytes)
                : kit_t::gromovwassersteins_workspace_size(widest_points_, widest_points_, &bytes);
        if (sized != nk_success_k) return status_of(sized);
        void* const workspace = round_arena_.allocate(bytes, nk_default_alignment_k);
        if (!workspace) return status_t::bad_alloc_k;
        if (kernel_ == nk_kernel_chamfers_packed_k)
            return status_of(kit_t::chamfers(queries, queries_count, candidates, count, results, stride, nullptr,
                                             &radius, workspace, bytes, nullptr));
        if (kernel_ == nk_kernel_wassersteinprocrustes_packed_k)
            return status_of(kit_t::wassersteinprocrustes(queries, queries_count, candidates, count, results, stride,
                                                          nullptr, workspace, bytes, nullptr));
        return status_of(kit_t::gromovwassersteins(queries, queries_count, candidates, count, results, stride, nullptr,
                                                   workspace, bytes, nullptr));
    }
};

#pragma endregion Metric

} // namespace unum::usearch::shapes
