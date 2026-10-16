/**
 *  @file include/usearch/shapes/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Batched shape distances between 3D point clouds of any size, over NumKong's many-to-many
 *      kernels.
 *
 *  The shapes index scores one round at a time: @c reset opens it for a query pack, @c push stages
 *  candidate packs, and @c drain scores every staged candidate in one 1×M kernel call. A metric
 *  here owns its staging and the transport workspace, so one instance serves one thread and
 *  allocates on the scoring path only when a pair is wider than every pair it has seen.
 *
 *  @section shapes_ram_metric_kits Kits
 *
 *  A kit binds one ISA's four kernels, Chamfer, Hausdorff, Wasserstein-Procrustes and
 *  Gromov-Wasserstein, with the two transport workspace sizes and the pack, unpack and pack-reading
 *  kernels, which every ISA shares, and the cast that brings a caller's coordinates into the stored
 *  scalar and back out. @c shapes_ladder picks the widest kit this build compiled in under the
 *  requested capability mask, at compile time, so no capability is tested at run time.
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
 *  Both hold for the exact definitions and are symmetric, so a candidate whose bound alone clears
 *  the round's radius reports the bound and never reaches the kernel.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - where the next pack's boundary falls

#include <limits> // `std::numeric_limits` - the distance a failed round reports
#include <memory> // `std::align` - the workspace's start on the kernels' boundary
#include <span>   // `std::span`

#include <numkong/cast.h>     // `nk_cast_serial` & kin - the kits' scalar casts
#include <numkong/shapes.hpp> // `nk::shapes_packed`, and the per-backend kernels
#include <numkong/tensor.hpp> // `nk::matrix_view` - the landed clouds a round packs

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator_t`
#include "usearch/basic/ram_index.hpp"      // `basic::metric_staging_t`

namespace unum::usearch::shapes {
using namespace unum::usearch::utilities;

/** The USearch status of a NumKong one. */
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

/** The USearch status of a NumKong C++ one, whose codes are the C ones. */
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
    static constexpr auto pack = nk_shapes_pack_f32_serial;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_neon;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_neon;
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
    static constexpr auto pack = nk_shapes_pack_f32_serial;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_haswell;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_haswell;
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
    static constexpr auto pack = nk_shapes_pack_f32_serial;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_skylake;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_skylake;
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
    static constexpr auto pack = nk_shapes_pack_f32_serial;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_rvv;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_rvv;
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
    static constexpr auto pack = nk_shapes_pack_f32_serial;
    static constexpr auto unpack = nk_shapes_unpack_f32_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f32_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f32_v128relaxed;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f32_v128relaxed;
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
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_neon;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_neon;
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
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_neonfhm;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_neonfhm;
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
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_haswell;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_haswell;
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
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_skylake;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_skylake;
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
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_rvv;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_rvv;
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
    static constexpr auto pack = nk_shapes_pack_f16_serial;
    static constexpr auto unpack = nk_shapes_unpack_f16_serial;
    static constexpr auto packed_shape = nk_shapes_packed_shape_f16_serial;
    static constexpr auto chamfers = nk_chamfers_packed_f16_v128relaxed;
    static constexpr auto hausdorffs = nk_hausdorffs_packed_f16_v128relaxed;
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
 *  @brief One of NumKong's four shape kinds over one round of candidates: @c reset reads the query
 *      pack's constants, @c push stages candidate packs, and @c drain bounds what it can and scores
 *      the rest in one 1×M call, exact up to the radius.
 *
 *  One instance belongs to one thread and owns its staging and the transport workspace; the C
 *  variants place them in the executor thread's arena. Packs are read in place, never copied: a
 *  candidate stays valid until its round drains, and the query until the next @c reset.
 *
 *  @tparam kit_type_ The kernels, as one @c shapes_ladder names them.
 *  @tparam allocator_type_ Places the staging, the kernel's distances and the workspace.
 */
template <typename kit_type_, typename allocator_type_ = forkunion_allocator_t>
class shapes_metric {
  public:
    using kit_t = kit_type_;
    using scalar_t = typename kit_t::scalar_t;
    using distance_t = typename kit_t::distance_t;
    using allocator_t = allocator_type_;

    /** A query may be a pack under any allocator, like one a call builds in its arena. */
    using is_transparent = void;

  private:

    /** The type of a pack's constants: @c nk::f32_t for @c f16 clouds, @c nk::f64_t for @c f32
     *  ones. */
    using constant_t = typename scalar_t::shapes_distance_t;
    using raw_constant_t = typename constant_t::raw_t;

    /** Which of the four shape kernels scores the rounds. */
    nk_kernel_kind_t kernel_;

    /** The query's pack, borrowed from @c reset. */
    void const* query_ = nullptr;

    /** Points in the query's cloud. */
    std::size_t query_points_ = 0;

    /** Roots of the query's descending spreads, the query's side of the Gelbrich bound. */
    constant_t query_spread_roots_[3] {};

    /** The query's largest packed point norm, the query's side of the Hausdorff bound. */
    constant_t query_max_radius_ {};

    /** The round's candidate packs in push order; @c drain closes up the ones no bound settles at
     *  the front. */
    growing_buffer<void const*, allocator_t> staged_;

    /** The packs of the round's landed clouds, each on NumKong's boundary; it grows only while it
     *  holds none, so a staged pack never moves. */
    growing_buffer<byte_t, allocator_t> packed_;

    /** Points in the round's largest candidate, the transport workspace's column side. */
    std::size_t staged_max_points_ = 0;

    /** The kernel's distances to the candidates no bound settled, in push order. */
    growing_buffer<distance_t, allocator_t> scored_;

    /** The transport kernels' scratch, one boundary wider than they ask, so its start rounds up to
     *  their boundary whatever this allocator aligns to. */
    growing_buffer<byte_t, allocator_t> workspace_;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_t::uninitialized_k;

  public:
    explicit shapes_metric(nk_kernel_kind_t kernel, allocator_t const& allocator) noexcept
        : kernel_(kernel), staged_(allocator), packed_(allocator), scored_(allocator), workspace_(allocator) {}
    shapes_metric(shapes_metric const&) = delete;
    shapes_metric& operator=(shapes_metric const&) = delete;
    shapes_metric(shapes_metric&&) noexcept = default;
    shapes_metric& operator=(shapes_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the kernel's distances.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at
     *      least the index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or
     *      rounds saturate rather than fail.
     *  @return @c unknown_kind_k for a kernel kind other than the four shape kinds.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit) return status_t::config_mismatch_k;
        if (kernel_ != nk_kernel_chamfers_packed_k && kernel_ != nk_kernel_hausdorffs_packed_k &&
            kernel_ != nk_kernel_wassersteinprocrustes_packed_k && kernel_ != nk_kernel_gromovwassersteins_packed_k)
            return status_t::unknown_kind_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        last_status_ = scored_.resize_uninitialized(staged_.capacity());
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared, the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, which must hold exactly one cloud, discarding whatever the
     *  previous round staged. */
    template <typename query_allocator_type_>
    void reset(nk::shapes_packed<scalar_t, query_allocator_type_> const& query) noexcept {
        staged_.clear(), packed_.clear(), staged_max_points_ = 0;
        if (query.size() != 1) {
            last_status_ = status_t::shape_mismatch_k;
            return;
        }
        query_ = query[0], query_points_ = query.max_points();
        raw_constant_t spreads[3], max_radius;
        last_status_ = status_of(
            kit_t::packed_shape(query_, nullptr, nullptr, nullptr, nullptr, spreads, &max_radius, nullptr));
        query_max_radius_ = constant_t::from_raw(max_radius);
        for (std::size_t axis = 0; axis != 3; ++axis)
            query_spread_roots_[axis] = spread_root_(constant_t::from_raw(spreads[axis]));
    }

    /** Stages one candidate's pack, or refuses once the round is full. */
    template <typename value_allocator_type_>
    basic::metric_staging_t push(nk::shapes_packed<scalar_t, value_allocator_type_> const& candidate) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        staged_.push_back_reserved(candidate[0]);
        staged_max_points_ = max_of(staged_max_points_, candidate.max_points());
        return basic::metric_staging_t::staged_k;
    }

    /** Packs a landed @b [points,3] @p cloud into the metric's staging arena and stages it;
     *  @c refused_k once either is full, so the walk drains first. A pack that fails leaves the
     *  round failed, as @ref status reports. */
    basic::metric_staging_t push(nk::matrix_view<scalar_t> cloud) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        std::size_t const points = cloud.extent(0);
        nk_size_t pack_bytes = 0;
        if (nk_status_t const sized = kit_t::pack_size(points, &pack_bytes); sized != nk_success_k)
            return last_status_ = status_of(sized), basic::metric_staging_t::refused_k;
        auto pack_offset = [&]() noexcept {
            std::uintptr_t const start = reinterpret_cast<std::uintptr_t>(packed_.data());
            return round_up_to_multiple<std::uintptr_t>(start + packed_.size(), nk_default_alignment_k) - start;
        };
        if (pack_offset() + pack_bytes > packed_.capacity()) {
            if (!packed_.empty()) return basic::metric_staging_t::refused_k;
            if (status_t const grown = packed_.reserve(pack_bytes + nk_default_alignment_k); failed(grown))
                return last_status_ = grown, basic::metric_staging_t::refused_k;
        }
        std::size_t const offset = pack_offset();
        byte_t* const pack = packed_.data() + offset;
        packed_.extend_uninitialized(offset + pack_bytes);
        if (status_t const packed = status_of(kit_t::pack(cloud.raw_values_data(), points, pack, nullptr));
            failed(packed))
            last_status_ = packed;
        staged_.push_back_reserved(pack);
        staged_max_points_ = max_of(staged_max_points_, points);
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged and closes the round.
     *
     *  Distances up to and including @p radius are exact; above it a bounded kind may report a
     *  candidate's lower bound instead. A failed round reports the largest finite distance, so a
     *  full pool rejects it and an unfilled pool admits it last; @ref status is the only signal.
     *
     *  @param[out] distances One slot per staged candidate, in push order.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<distance_t> distances, distance_t radius) noexcept {
        std::size_t const count = staged_.size();
        if (succeeded(last_status_)) {
            // Bounds land in place, and the candidates still within reach close up at the front, in
            // push order.
            std::size_t reached = 0;
            for (std::size_t slot = 0; slot != count; ++slot) {
                distances[slot] = bound_(staged_[slot]);
                if (!(distances[slot] > radius)) staged_[reached++] = staged_[slot];
            }
            if (reached) last_status_ = score_(reached);
        }
        if (failed(last_status_)) {
            for (std::size_t slot = 0; slot != count; ++slot)
                distances[slot] = (std::numeric_limits<distance_t>::max)();
        }
        else {
            for (std::size_t slot = 0, scored = 0; slot != count; ++slot)
                if (!(distances[slot] > radius)) distances[slot] = scored_[scored++];
        }
        staged_.clear(), packed_.clear(), staged_max_points_ = 0;
    }

  private:

    /** The root of one spread, clamped at zero, where a flat cloud's smallest spread may round
     *  below it. */
    static constant_t spread_root_(constant_t spread) noexcept {
        return (spread < constant_t {} ? constant_t {} : spread).sqrt();
    }

    /** A lower bound of the distance to @p candidate from the two packs' constants alone, or zero
     *  for the kinds the constants do not bound. */
    distance_t bound_(void const* candidate) const noexcept {
        bool const procrustes = kernel_ == nk_kernel_wassersteinprocrustes_packed_k;
        if (!procrustes && kernel_ != nk_kernel_hausdorffs_packed_k) return 0;
        raw_constant_t spreads[3], max_radius;
        [[maybe_unused]] nk_status_t const read = kit_t::packed_shape(candidate, nullptr, nullptr, nullptr, nullptr,
                                                                      spreads, &max_radius, nullptr);
        if (!procrustes) return (query_max_radius_ - constant_t::from_raw(max_radius)).abs();
        constant_t gaps {};
        for (std::size_t axis = 0; axis != 3; ++axis) {
            constant_t const gap = query_spread_roots_[axis] - spread_root_(constant_t::from_raw(spreads[axis]));
            gaps = gaps + gap * gap;
        }
        return gaps.sqrt();
    }

    /** Scores the query against the first @p count staged packs in one call, into @ref scored_. */
    status_t score_(std::size_t count) noexcept {
        void const* const* const candidates = staged_.data();
        distance_t* const results = scored_.data();
        std::size_t const stride = count * sizeof(distance_t);
        if (kernel_ == nk_kernel_chamfers_packed_k)
            return status_of(kit_t::chamfers(&query_, 1, candidates, count, results, stride, nullptr, nullptr));
        if (kernel_ == nk_kernel_hausdorffs_packed_k)
            return status_of(kit_t::hausdorffs(&query_, 1, candidates, count, results, stride, nullptr, nullptr));

        bool const procrustes = kernel_ == nk_kernel_wassersteinprocrustes_packed_k;
        nk_size_t bytes = 0;
        nk_status_t const sized =
            procrustes ? kit_t::wassersteinprocrustes_workspace_size(query_points_, staged_max_points_, &bytes)
                       : kit_t::gromovwassersteins_workspace_size(query_points_, staged_max_points_, &bytes);
        if (sized != nk_success_k) return status_of(sized);
        // Grows only past the widest pair seen, and doubles when it does.
        if (status_t const grown = workspace_.reserve(bytes + nk_default_alignment_k); failed(grown)) return grown;
        void* workspace = workspace_.data();
        std::size_t workspace_bytes = workspace_.capacity();
        std::align(nk_default_alignment_k, bytes, workspace, workspace_bytes);
        return status_of(procrustes ? kit_t::wassersteinprocrustes(&query_, 1, candidates, count, results, stride,
                                                                   nullptr, workspace, workspace_bytes, nullptr)
                                    : kit_t::gromovwassersteins(&query_, 1, candidates, count, results, stride, nullptr,
                                                                workspace, workspace_bytes, nullptr));
    }
};

#pragma endregion Metric

} // namespace unum::usearch::shapes
