/**
 *  @file include/usearch/dense/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date April 21, 2026
 *  @brief Compile-time-dispatched 1-to-1, 1-to-4, and 16×16 tiled distance kernels backed by
 *      NumKong.
 *
 *  This header is the hot inner loop of @c basic_dense_hindex - everything called per
 *  graph-traversal step, per query, per candidate. Its job is to turn NumKong's stateful
 *  ISA-specific dot-product primitives into clean @c distance and @c distances entry
 *  points that callers can reach without naming an ISA, a state type, or a vector width.
 *
 *  @section File Organization
 *
 *  @b Norms - @c vector_norms carries per-vector @c sum (DPBUSD compensation) and @c sumsq
 *  (L2 norm, used by angular/euclidean). @c dense_norms<scalar, caps> dispatches to
 *  NumKong's @c nk_reduce_moments_* family at construction-time of the index.
 *
 *  @b Kits - one struct per @c (scalar,capability), e.g. @c bf16_genoa_inner_kit_t, bundling
 *  every piece of the inner-product engine - NumKong state type, operand vec type, finalized
 *  vec type, load/update/finalize function pointers, step-dimensions count - plus the
 *  @c *_from_dot throughs each derived metric finalizes with. The metric is a runtime value
 *  that reaches only @c convert_to_distances, so one kit serves every metric of its scalar.
 *  The shape is uniform - the @c dense_quads_kit concept enforces it.
 *
 *  @b Chains - one @c dense_quads_chain<scalar,caps> per supported scalar. A linear
 *  list of @c chain_layer rows fed to @c first_available_kit picks the first kit whose
 *  @c NUMKONG_TARGET_* is on AND whose runtime-caps bit is set, else falls back to serial.
 *  Unsupported scalars simply have no specialization. Adding an ISA
 *  tier is one row and zero preprocessor lines.
 *
 *  @b Trampoline - @c invoke_scores_<fan_out,kit> is the one place the load, update,
 *  finalize, through, prefetch, demote choreography lives. Reads everything it needs -
 *  @c scalar_t, @c step_vec_t, @c reduce_accumulators, ... - off the kit. Lane fan-out
 *  is unrolled via @c std::index_sequence so the same body emits clean code for both
 *  @b 1-to-1 and @b 1-to-4. The state array is always sized 4 (NumKong's finalize is
 *  hardcoded fan-out-4); for 1-to-1 the bottom three slots are left as register noise
 *  and the resulting lanes 1-3 are dropped at extract time.
 *
 *  @b Entry-points - free function templates @c distance<caps> and @c distances<caps>, with
 *  @c scalar deduced from the @c nk::vector_view inputs and the metric their first argument.
 *  Call sites read @c distance<caps>(nk_kernel_angular_k,...) - no per-spec method, no per-spec class.
 *
 *  @b Tiles - a second, independent engine axis over the same capability bitmask. Each
 *  @c dense_tiles_chain<scalar,caps> names a tile kit - @c no_tiles_kit_t unless a
 *  matrix engine is adapted for the scalar - and its @c has_tiles_k verdict gates every
 *  caller. A tile kit (e.g. @c u8_sapphireamx_tiles_kit) exposes the primitive walk:
 *  @c configure_tiles once per thread, @c gather_a_tile straight from vector views,
 *  @c pack_b_tile into the engine's interleaved operand layout, @c init_tile and
 *  @c update_tile over the depth, and @c finalize_tile - plus @c scratch_bytes so callers
 *  size scratch without naming tile shapes. All engines share the 16-row @c tile_span_k
 *  geometry, and finalization routes through the same quads kit the chain picked, so a
 *  tiled Gram of distances matches the 1-to-4 path bit for bit.
 *
 *  @section Design Goals & Decisions
 *
 *  1. @b Avoid-call-overhead. The arms build NumKong header-only, so every capability kernel and kit inlines
 *  into the arm, and no call crosses into the shared library.
 *
 *  2. @b Amortize-arithmetic-reductions. Both Euclidean (L2) and Angular (Cos) distances
 *  can be inferred from the inner product if the norms are known. This means @b 2-by-D
 *  arithmetic operations per distance computation instead of @b 6-by-D on the hot path,
 *  by precomputing and storing one extra norm value per vector.
 *
 *  3. @b Amortize-memory-loads-and-mini-float-upcasting. When computing an angular distance
 *  between a query vector and M neighbors of a graph node, we might be upcasting
 *  mini-floats to Float32 @b 2-by-M times. Storing Float32 in the index kills the value
 *  proposition of memory reductions. Pre-upcasting M vectors at once washes the
 *  register file. So we expose @b 1-to-4 distances - the query half of the binary
 *  operation is loaded and upcast four times less frequently, shrinking the total
 *  loads & upcasts to @b M+M/4. This also paves the way for cache-friendly streaming
 *  loads, keeping L1/L2 reserved for graph-traversal data structures while the vectors
 *  themselves only go as high as the L3 / cross-core LLC.
 *
 *  4. @b Amortize-horizontal-reductions. On low-dimensional vectors (64D Matryoshka or
 *  128D GQA), the most expensive part of inner-products on x86 is the final horizontal
 *  reduction of accumulators. The 1-to-4 path lets us amortize each reduction with
 *  cross-register shuffles.
 *
 *  @section Assumptions & Limitations
 *
 *  Designed for the most common fast-path use cases of USearch templates, where a distance
 *  fits in @c f32_t with acceptable precision loss. It @b does-not apply to:
 *
 *  - indexing arbitrary non-equidimensional entries like variable-length strings,
 *  - Haversine, Vincenty, or other geospatial metrics not built around inner products,
 *  - multi-vector entry descriptors produced by late-interaction ColBERT-like models.
 *
 *  Some NumKong kernels need both the L2 norm and the row-sum of the vector - both live
 *  side-by-side in @c vector_norms, which the index stores once per vector at insert time.
 */
#pragma once

#include <cassert>

#include <array>
#include <type_traits>
#include <utility>

#include <numkong/numkong.h>
#include <numkong/types.hpp>
#include <numkong/vector.hpp>

#include <forkunion/types.hpp> // `fu::standard_cache_hints_t`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_tables.hpp"
#include "usearch/dense/types.hpp"

namespace unum {
namespace usearch {

/** Per-lane @c 1 - dot in NumKong vec form. Exists to match the
 *  @c nk_*_through_*_from_dot_* signature so the trampoline routes every metric uniformly. */
template <typename norm_type_, typename finalized_inners_type_>
inline void inner_product_distance_(finalized_inners_type_ const* dots_vec, norm_type_ /*query_sumsq_unused*/,
                                    finalized_inners_type_ const* /*target_sumsqs_unused*/,
                                    finalized_inners_type_* result_vec) noexcept {
    if constexpr (nk::is_integral_dtype<norm_type_>()) {
        if constexpr (!nk::is_signed_dtype<norm_type_>()) {
            result_vec->i32s[0] = 1 - static_cast<nk_i32_t>(dots_vec->u32s[0]);
            result_vec->i32s[1] = 1 - static_cast<nk_i32_t>(dots_vec->u32s[1]);
            result_vec->i32s[2] = 1 - static_cast<nk_i32_t>(dots_vec->u32s[2]);
            result_vec->i32s[3] = 1 - static_cast<nk_i32_t>(dots_vec->u32s[3]);
        }
        else {
            result_vec->i32s[0] = 1 - dots_vec->i32s[0];
            result_vec->i32s[1] = 1 - dots_vec->i32s[1];
            result_vec->i32s[2] = 1 - dots_vec->i32s[2];
            result_vec->i32s[3] = 1 - dots_vec->i32s[3];
        }
    }
    else if constexpr (sizeof(norm_type_) == 4) {
        result_vec->f32s[0] = 1.0f - dots_vec->f32s[0];
        result_vec->f32s[1] = 1.0f - dots_vec->f32s[1];
        result_vec->f32s[2] = 1.0f - dots_vec->f32s[2];
        result_vec->f32s[3] = 1.0f - dots_vec->f32s[3];
    }
    else {
        result_vec->f64s[0] = 1.0 - dots_vec->f64s[0];
        result_vec->f64s[1] = 1.0 - dots_vec->f64s[1];
        result_vec->f64s[2] = 1.0 - dots_vec->f64s[2];
        result_vec->f64s[3] = 1.0 - dots_vec->f64s[3];
    }
}

/** Whether @p kit_ finalizes the bit metrics, hamming & jaccard, instead of angular & euclidean. */
template <typename kit_>
concept dense_bits_kit = requires {
    kit_::hamming_from_dot;
    kit_::jaccard_from_dot;
};

/** Compile-time concept for every @c inner_kit_t: the inner-product engine, and the throughs of either the
 *  numeric metrics or the bit ones. The trampoline takes only a kit, so substitution errors land at the call
 *  site as a one-line constraint-not-satisfied. */
template <typename kit_>
concept dense_quads_kit = requires {
    typename kit_::scalar_t;
    typename kit_::accumulator_t;
    typename kit_::step_vec_t;
    typename kit_::finalized_vec_t;
    kit_::init_accumulator;
    kit_::update_accumulator;
    kit_::load_step;
    kit_::load_tail;
    kit_::reduce_accumulators;
    kit_::compute_norms;
    { kit_::step_dimensions } -> std::convertible_to<std::size_t>;
} && (dense_bits_kit<kit_> || requires {
                              kit_::angular_from_dot;
                              kit_::euclidean_from_dot;
                          });

/**
 *  @brief Internal trampoline for using @c nk_reduce_moments NumKong APIs to compute L2 vector norms.
 *
 *  Returns @c vector_norms<dot_result_t> - the same width that downstream finalize and through helpers consume.
 *  For integer scalars this narrows the i64/u64 native output of @c nk_reduce_moments_* to @c i32/u32;
 *  for float scalars the @c static_cast is a no-op identity. Narrowing is safe for any reasonable embedding sizes.
 */
template <typename function_type_, nk::numeric_dtype scalar_type_>
inline vector_norms<typename scalar_type_::dot_result_t> invoke_dense_norms_( //
    function_type_ function, nk::vector_view<scalar_type_> vec) noexcept {

    using native_sum_t = typename scalar_type_::reduce_moments_sum_t;
    using native_sumsq_t = typename scalar_type_::reduce_moments_sumsq_t;
    using dot_result_t = typename scalar_type_::dot_result_t;
    using raw_scalar_t = typename scalar_type_::raw_t;
    using raw_sum_t = typename native_sum_t::raw_t;
    using raw_sumsq_t = typename native_sumsq_t::raw_t;

    // NumKong's `nk_reduce_moments` take count in dimensions, not number of values and need explict
    // downcasting to the underlying C-level raw types from its own C++ wrappers.
    native_sum_t native_sum;
    native_sumsq_t native_sumsq;
    [[maybe_unused]] nk_status_t const status = function(
        reinterpret_cast<raw_scalar_t const*>(vec.data()), vec.size(), sizeof(scalar_type_),
        reinterpret_cast<raw_sum_t*>(&native_sum), reinterpret_cast<raw_sumsq_t*>(&native_sumsq), nullptr);
    assert(status == nk_success_k && "A CPU tier reduces any vector, given no stream");
    return {static_cast<dot_result_t>(native_sum), static_cast<dot_result_t>(native_sumsq)};
}

/** Forward declarations for every "kit" regardless of @c NUMKONG_TARGET_* availability,
 *  so @c dense_quads_chain backend resolution chains name-resolve on every build. */
struct bf16_serial_inner_kit_t;
struct bf16_haswell_inner_kit_t;
struct bf16_skylake_inner_kit_t;
struct bf16_genoa_inner_kit_t;
struct bf16_neon_inner_kit_t;
struct bf16_neonbfdot_inner_kit_t;
struct bf16_v128_inner_kit_t;
struct bf16_v128relaxed_inner_kit_t;
struct bf16_powervsx_inner_kit_t;
struct f16_serial_inner_kit_t;
struct f16_haswell_inner_kit_t;
struct f16_skylake_inner_kit_t;
struct f16_neon_inner_kit_t;
struct f16_neonfhm_inner_kit_t;
struct f16_v128relaxed_inner_kit_t;
struct f16_powervsx_inner_kit_t;
struct e5m2_serial_inner_kit_t;
struct e5m2_diamond_inner_kit_t;
struct e5m2_haswell_inner_kit_t;
struct e5m2_skylake_inner_kit_t;
struct e5m2_genoa_inner_kit_t;
struct e5m2_neonfhm_inner_kit_t;
struct e5m2_neonfp8_inner_kit_t;
struct e5m2_v128relaxed_inner_kit_t;
struct e4m3_serial_inner_kit_t;
struct e4m3_diamond_inner_kit_t;
struct e4m3_haswell_inner_kit_t;
struct e4m3_skylake_inner_kit_t;
struct e4m3_genoa_inner_kit_t;
struct e4m3_neonfhm_inner_kit_t;
struct e4m3_neonfp8_inner_kit_t;
struct e4m3_v128relaxed_inner_kit_t;
struct e3m2_serial_inner_kit_t;
struct e3m2_haswell_inner_kit_t;
struct e3m2_skylake_inner_kit_t;
struct e3m2_neonsdot_inner_kit_t;
struct e3m2_neonfp8_inner_kit_t;
struct e3m2_v128relaxed_inner_kit_t;
struct e2m3_serial_inner_kit_t;
struct e2m3_haswell_inner_kit_t;
struct e2m3_skylake_inner_kit_t;
struct e2m3_alder_inner_kit_t;
struct e2m3_sierra_inner_kit_t;
struct e2m3_neonsdot_inner_kit_t;
struct e2m3_neonfp8_inner_kit_t;
struct e2m3_v128relaxed_inner_kit_t;
struct i8_serial_inner_kit_t;
struct i8_haswell_inner_kit_t;
struct i8_icelake_inner_kit_t;
struct i8_alder_inner_kit_t;
struct i8_sierra_inner_kit_t;
struct i8_neonsdot_inner_kit_t;
struct i8_v128_inner_kit_t;
struct i8_v128relaxed_inner_kit_t;
struct i8_powervsx_inner_kit_t;
struct u8_serial_inner_kit_t;
struct u8_icelake_inner_kit_t;
struct u8_haswell_inner_kit_t;
struct u8_alder_inner_kit_t;
struct u8_sierra_inner_kit_t;
struct u8_neonsdot_inner_kit_t;
struct u8_v128_inner_kit_t;
struct u8_v128relaxed_inner_kit_t;
struct u8_powervsx_inner_kit_t;
struct u4x2_serial_inner_kit_t;
struct u4x2_haswell_inner_kit_t;
struct u4x2_icelake_inner_kit_t;
struct u4x2_neonsdot_inner_kit_t;
struct u4x2_v128relaxed_inner_kit_t;
struct i4x2_serial_inner_kit_t;
struct i4x2_haswell_inner_kit_t;
struct i4x2_icelake_inner_kit_t;
struct i4x2_neonsdot_inner_kit_t;
struct i4x2_v128relaxed_inner_kit_t;
struct u1x8_serial_inner_kit_t;
struct u1x8_haswell_inner_kit_t;
struct u1x8_icelake_inner_kit_t;
struct u1x8_neon_inner_kit_t;
struct u1x8_v128_inner_kit_t;
struct u1x8_powervsx_inner_kit_t;

/** bf16 inner-product distance kit - serial reference. */
struct bf16_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_serial;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

#if NUMKONG_TARGET_HASWELL

/** bf16 inner-product distance kit - Intel Haswell AVX2. */
struct bf16_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x16_state_haswell_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x16_init_haswell;
    static constexpr auto update_accumulator = nk_dot_bf16x16_update_haswell;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b16x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x16_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_haswell;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** bf16 inner-product distance kit - Intel Skylake-X AVX-512 Foundation. */
struct bf16_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x32_state_skylake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x32_init_skylake;
    static constexpr auto update_accumulator = nk_dot_bf16x32_update_skylake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b16x32_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x32_finalize_skylake;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_skylake;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_GENOA

/** bf16 inner-product distance kit - AMD Genoa AVX-512 + VNNI BF16. */
struct bf16_genoa_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x32_state_genoa_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x32_init_genoa;
    static constexpr auto update_accumulator = nk_dot_bf16x32_update_genoa;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b16x32_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x32_finalize_genoa;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_genoa;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_NEON

/** bf16 inner-product distance kit - Arm NEON. */
struct bf16_neon_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_neon_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_neon;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_neon;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_neon;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_serial;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT

/** bf16 inner-product distance kit - Arm BFDOT. */
struct bf16_neonbfdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_neonbfdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_neonbfdot;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_neonbfdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_neonbfdot;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_neonbfdot;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_V128

/** bf16 inner-product distance kit - WebAssembly SIMD128. */
struct bf16_v128_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_v128_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_v128;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_v128;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_v128;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_v128;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED

/** bf16 inner-product distance kit - WebAssembly relaxed SIMD. */
struct bf16_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_v128;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX

/** bf16 inner-product distance kit - IBM POWER VSX. NumKong has no bf16 moments
 *  kernel for this tier, so the norms come from the serial one. */
struct bf16_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b16x8_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_powervsx;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_powervsx_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_powervsx_;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_serial;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_POWERVSX

/** f16 inner-product distance kit - serial reference. */
struct f16_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_serial;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

#if NUMKONG_TARGET_HASWELL

/**
 *  @brief f16 inner-product distance kit - Intel Haswell AVX2 + F16C.
 *
 *  There is no native f16 accumulator at this tier, so the loaders widen each step to
 *  f32 up front and the generic through-f32 accumulator does the arithmetic. That makes
 *  @c step_vec_t twice as wide as the eight f16 it holds - the trampoline addresses off
 *  @c scalar_t, so the narrower source stride is handled correctly.
 */
struct f16_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_through_f32_state_haswell_t_;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_haswell_;
    static constexpr auto update_accumulator = nk_dot_through_f32_update_haswell_;
    static constexpr auto load_step = nk_load_f16x8_to_f32x8_haswell_;
    static constexpr auto load_tail = nk_partial_load_f16x8_to_f32x8_haswell_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_haswell_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_haswell;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** f16 inner-product distance kit - Intel Skylake-X AVX-512, widening on load. */
struct f16_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_through_f32_state_skylake_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_skylake_;
    static constexpr auto update_accumulator = nk_dot_through_f32_update_skylake_;
    static constexpr auto load_step = nk_load_f16x16_to_f32x16_skylake_;
    static constexpr auto load_tail = nk_partial_load_f16x16_to_f32x16_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_skylake_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_skylake;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_NEON

/** f16 inner-product distance kit - Arm NEON. */
struct f16_neon_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_neon_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_neon;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_neon;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_neon;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_neon_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_neon_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_neon;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONFHM

/** f16 inner-product distance kit - Arm FP16 FML. NumKong has no f16 moments
 *  kernel for this tier, so the norms come from the serial one. */
struct f16_neonfhm_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_neonfhm_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_neonfhm;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_neonfhm;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_neonfhm;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_serial;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_V128RELAXED

/** f16 inner-product distance kit - WebAssembly relaxed SIMD, widening on load.
 *  Four dimensions per step, since the widened f32 quad fills the 128-bit vec. */
struct f16_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_through_f32x4_state_v128relaxed_t_;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32x4_init_v128relaxed_;
    static constexpr auto update_accumulator = nk_dot_through_f32x4_update_v128relaxed_;
    static constexpr auto load_step = nk_load_f16x4_to_f32x4_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_f16x4_to_f32x4_v128relaxed_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32x4_finalize_v128relaxed_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_v128relaxed;
    static constexpr std::size_t step_dimensions = 4;
};
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX

/** f16 inner-product distance kit - IBM POWER VSX. NumKong has no f16 moments
 *  kernel for this tier, so the norms come from the serial one. */
struct f16_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b16x8_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_powervsx;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_powervsx_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_powervsx_;
    static constexpr auto compute_norms = nk_reduce_moments_f16_serial;
    static constexpr std::size_t step_dimensions = 8;
};
#endif // NUMKONG_TARGET_POWERVSX

/** e5m2 inner-product distance kit - serial reference. */
struct e5m2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e5m2x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e5m2x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e5m2x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/**
 *  @brief e5m2 inner-product distance kit - Intel Haswell AVX2.
 *
 *  The step loads raw bytes and the specialised update widens them internally, so this
 *  pairs a byte-batched update with the generic through-f32 accumulator.
 */
struct e5m2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_f32_state_haswell_t_;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_haswell_;
    static constexpr auto update_accumulator = nk_dot_e5m2x32_update_haswell_;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_haswell_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_haswell;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** e5m2 inner-product distance kit - Intel Skylake-X AVX-512, widening in the update. */
struct e5m2_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_f32_state_skylake_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_skylake_;
    static constexpr auto update_accumulator = nk_dot_e5m2x64_update_skylake_;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_skylake_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_skylake;
    static constexpr std::size_t step_dimensions = 64;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_DIAMOND

/** e5m2 inner-product distance kit - Intel Diamond Rapids AVX10.2. The loaders
 *  widen each step to f16 with a native FP8 convert, so the dot rides the FP16
 *  accumulator. NumKong has no e5m2 moments kernel for this tier, so the norms
 *  come from the AVX-512 one. */
struct e5m2_diamond_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_diamond_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_f16_state_diamond_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f16_init_diamond_;
    static constexpr auto update_accumulator = nk_dot_through_f16_update_diamond_;
    static constexpr auto load_step = nk_load_e5m2x32_to_f16x32_diamond_;
    static constexpr auto load_tail = nk_partial_load_e5m2x32_to_f16x32_diamond_;
    static constexpr auto reduce_accumulators = nk_dot_through_f16_finalize_diamond_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_skylake;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_GENOA

/** e5m2 inner-product distance kit - AMD Genoa AVX-512 + VNNI BF16. The loaders
 *  widen each step to bf16 so the dot rides the BF16 accumulator. */
struct e5m2_genoa_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_bf16_state_genoa_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_bf16_init_genoa_;
    static constexpr auto update_accumulator = nk_dot_through_bf16_update_genoa_;
    static constexpr auto load_step = nk_load_e5m2x32_to_bf16x32_icelake_;
    static constexpr auto load_tail = nk_partial_load_e5m2x32_to_bf16x32_icelake_;
    static constexpr auto reduce_accumulators = nk_dot_through_bf16_finalize_genoa_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_genoa;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_NEONFHM

/** e5m2 inner-product distance kit - Arm FP16 FML. */
struct e5m2_neonfhm_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x16_state_neonfhm_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e5m2x16_init_neonfhm;
    static constexpr auto update_accumulator = nk_dot_e5m2x16_update_neonfhm;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e5m2x16_finalize_neonfhm;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_neonfhm;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONFP8

/** e5m2 inner-product distance kit - Arm FP8 DOT. NumKong has no e5m2 moments
 *  kernel for this tier, so the norms come from the serial one. */
struct e5m2_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e5m2x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e5m2x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e5m2x16_finalize_neonfp8;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_V128RELAXED

/** e5m2 inner-product distance kit - WebAssembly relaxed SIMD, widening on load.
 *  Four dimensions per step, since the widened f32 quad fills the 128-bit vec. */
struct e5m2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x4_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32x4_init_v128relaxed_;
    static constexpr auto update_accumulator = nk_dot_through_f32x4_update_v128relaxed_;
    static constexpr auto load_step = nk_load_e5m2x4_to_f32x4_v128_;
    static constexpr auto load_tail = nk_partial_load_e5m2x4_to_f32x4_v128_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32x4_finalize_v128relaxed_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_v128relaxed;
    static constexpr std::size_t step_dimensions = 4;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** e4m3 inner-product distance kit - serial reference. */
struct e4m3_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e4m3x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e4m3x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e4m3x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** e4m3 inner-product distance kit - Intel Haswell AVX2, widening in the update. */
struct e4m3_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_f32_state_haswell_t_;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_haswell_;
    static constexpr auto update_accumulator = nk_dot_e4m3x32_update_haswell_;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_haswell_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_haswell;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** e4m3 inner-product distance kit - Intel Skylake-X AVX-512, widening on load. */
struct e4m3_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_f32_state_skylake_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_skylake_;
    static constexpr auto update_accumulator = nk_dot_through_f32_update_skylake_;
    static constexpr auto load_step = nk_load_e4m3x16_to_f32x16_skylake_;
    static constexpr auto load_tail = nk_partial_load_e4m3x16_to_f32x16_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_skylake_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_skylake;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_DIAMOND

/** e4m3 inner-product distance kit - Intel Diamond Rapids AVX10.2. The loaders
 *  widen each step to f16 with a native FP8 convert, so the dot rides the FP16
 *  accumulator. NumKong has no e4m3 moments kernel for this tier, so the norms
 *  come from the AVX-512 one. */
struct e4m3_diamond_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_diamond_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_f16_state_diamond_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f16_init_diamond_;
    static constexpr auto update_accumulator = nk_dot_through_f16_update_diamond_;
    static constexpr auto load_step = nk_load_e4m3x32_to_f16x32_diamond_;
    static constexpr auto load_tail = nk_partial_load_e4m3x32_to_f16x32_diamond_;
    static constexpr auto reduce_accumulators = nk_dot_through_f16_finalize_diamond_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_skylake;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_GENOA

/** e4m3 inner-product distance kit - AMD Genoa AVX-512 + VNNI BF16. The loaders
 *  widen each step to bf16 so the dot rides the BF16 accumulator. */
struct e4m3_genoa_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_bf16_state_genoa_t_;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_bf16_init_genoa_;
    static constexpr auto update_accumulator = nk_dot_through_bf16_update_genoa_;
    static constexpr auto load_step = nk_load_e4m3x32_to_bf16x32_icelake_;
    static constexpr auto load_tail = nk_partial_load_e4m3x32_to_bf16x32_icelake_;
    static constexpr auto reduce_accumulators = nk_dot_through_bf16_finalize_genoa_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_genoa;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_NEONFHM

/** e4m3 inner-product distance kit - Arm FP16 FML. */
struct e4m3_neonfhm_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x16_state_neonfhm_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e4m3x16_init_neonfhm;
    static constexpr auto update_accumulator = nk_dot_e4m3x16_update_neonfhm;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e4m3x16_finalize_neonfhm;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_neonfhm;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONFP8

/** e4m3 inner-product distance kit - Arm FP8 DOT. NumKong has no e4m3 moments
 *  kernel for this tier, so the norms come from the serial one. */
struct e4m3_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e4m3x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e4m3x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e4m3x16_finalize_neonfp8;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_serial;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_V128RELAXED

/** e4m3 inner-product distance kit - WebAssembly relaxed SIMD, widening on load.
 *  Four dimensions per step, since the widened f32 quad fills the 128-bit vec. */
struct e4m3_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x4_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_through_f32x4_init_v128relaxed_;
    static constexpr auto update_accumulator = nk_dot_through_f32x4_update_v128relaxed_;
    static constexpr auto load_step = nk_load_e4m3x4_to_f32x4_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_e4m3x4_to_f32x4_v128relaxed_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32x4_finalize_v128relaxed_;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_v128relaxed;
    static constexpr std::size_t step_dimensions = 4;
};
#endif // NUMKONG_TARGET_V128RELAXED

/**
 *  @brief e3m2 inner-product distance kit - serial reference.
 *
 *  The six significant bits occupy a whole byte in memory, so the step vec is sized by
 *  @c step_dimensions bytes rather than by the format's bit budget - the tail loader is
 *  @c b8 wide, exactly as for @c i8 and @c u8.
 */
struct e3m2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** e3m2 inner-product distance kit - Intel Haswell AVX2. */
struct e3m2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x32_state_haswell_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_e3m2x32_update_haswell;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x32_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_haswell;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** e3m2 inner-product distance kit - Intel Skylake-X AVX-512 Foundation. */
struct e3m2_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x64_state_skylake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x64_init_skylake;
    static constexpr auto update_accumulator = nk_dot_e3m2x64_update_skylake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x64_finalize_skylake;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_skylake;
    static constexpr std::size_t step_dimensions = 64;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_NEONSDOT

/** e3m2 inner-product distance kit - Arm SDOT. NumKong has no e3m2 moments kernel
 *  for this tier, so the norms come from the serial one. */
struct e3m2_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_neonsdot;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFP8

/** e3m2 inner-product distance kit - Arm FP8 DOT. NumKong has no e3m2 moments
 *  kernel for this tier, so the norms come from the serial one. */
struct e3m2_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_neonfp8;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_V128RELAXED

/** e3m2 inner-product distance kit - WebAssembly relaxed SIMD. */
struct e3m2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_v128relaxed;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** e2m3 inner-product distance kit - serial reference. One element per byte. */
struct e2m3_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** e2m3 inner-product distance kit - Intel Haswell AVX2. */
struct e2m3_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x32_state_haswell_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_e2m3x32_update_haswell;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x32_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_haswell;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** e2m3 inner-product distance kit - Intel Skylake-X AVX-512 Foundation. */
struct e2m3_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x64_state_skylake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x64_init_skylake;
    static constexpr auto update_accumulator = nk_dot_e2m3x64_update_skylake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x64_finalize_skylake;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_skylake;
    static constexpr std::size_t step_dimensions = 64;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ALDER

/** e2m3 inner-product distance kit - Intel Alder Lake AVX-VNNI. */
struct e2m3_alder_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_alder_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x32_state_alder_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x32_init_alder;
    static constexpr auto update_accumulator = nk_dot_e2m3x32_update_alder;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x32_finalize_alder;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_alder;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA

/** e2m3 inner-product distance kit - Intel Sierra Forest AVX-VNNI. */
struct e2m3_sierra_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sierra_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x32_state_sierra_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x32_init_sierra;
    static constexpr auto update_accumulator = nk_dot_e2m3x32_update_sierra;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x32_finalize_sierra;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_sierra;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_NEONSDOT

/** e2m3 inner-product distance kit - Arm SDOT. */
struct e2m3_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_neonsdot;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFP8

/** e2m3 inner-product distance kit - Arm FP8 DOT. */
struct e2m3_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_neonfp8;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_V128RELAXED

/** e2m3 inner-product distance kit - WebAssembly relaxed SIMD. */
struct e2m3_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_f32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_f32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_v128relaxed;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** u8 inner-product distance kit - serial reference. */
struct u8_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** u8 inner-product distance kit - Haswell AVX2. */
struct u8_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_haswell;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_haswell;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE

/** u8 inner-product distance kit - Intel Ice Lake AVX-512 + VNNI VPDPBUSD. */
struct u8_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x64_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x64_init_icelake;
    static constexpr auto update_accumulator = nk_dot_u8x64_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_u8x64_finalize_icelake;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_icelake;
    static constexpr std::size_t step_dimensions = 64;
};
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_ALDER

/** u8 inner-product distance kit - Intel Alder Lake AVX-VNNI. */
struct u8_alder_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_alder_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x32_state_alder_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x32_init_alder;
    static constexpr auto update_accumulator = nk_dot_u8x32_update_alder;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x32_finalize_alder;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_alder;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA

/** u8 inner-product distance kit - Intel Sierra Forest AVX-VNNI. */
struct u8_sierra_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sierra_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x32_state_sierra_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x32_init_sierra;
    static constexpr auto update_accumulator = nk_dot_u8x32_update_sierra;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x32_finalize_sierra;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_sierra;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_NEONSDOT

/** u8 inner-product distance kit - Arm UDOT. */
struct u8_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_neonsdot;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_V128

/** u8 inner-product distance kit - WebAssembly SIMD128. */
struct u8_v128_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_v128_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_v128;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_v128;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_v128;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_v128;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED

/** u8 inner-product distance kit - WebAssembly relaxed SIMD. */
struct u8_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_v128;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX

/** u8 inner-product distance kit - IBM POWER VSX. */
struct u8_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b8x16_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_powervsx;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_powervsx_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_powervsx_;
    static constexpr auto compute_norms = nk_reduce_moments_u8_serial;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_POWERVSX

/** i8 inner-product distance kit - serial reference. */
struct i8_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** i8 inner-product distance kit - Intel Haswell AVX2. */
struct i8_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_haswell;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_haswell;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE

/** i8 inner-product distance kit - Intel Ice Lake AVX-512 + VNNI VPDPBUSD compensation. */
struct i8_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x64_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x64_init_icelake;
    static constexpr auto update_accumulator = nk_dot_i8x64_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_i8x64_finalize_icelake;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_icelake;
    static constexpr std::size_t step_dimensions = 64;
};
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_ALDER

/** i8 inner-product distance kit - Intel Alder Lake AVX-VNNI. NumKong has no
 *  signed-i8 moments kernel for this tier, so the norms come from the AVX2 one,
 *  which Alder Lake also implements. */
struct i8_alder_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_alder_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x32_state_alder_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x32_init_alder;
    static constexpr auto update_accumulator = nk_dot_i8x32_update_alder;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x32_finalize_alder;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_haswell;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA

/** i8 inner-product distance kit - Intel Sierra Forest AVX-VNNI. */
struct i8_sierra_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sierra_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x32_state_sierra_t;
    using step_vec_t = nk_b256_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x32_init_sierra;
    static constexpr auto update_accumulator = nk_dot_i8x32_update_sierra;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x32_finalize_sierra;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_sierra;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_NEONSDOT

/** i8 inner-product distance kit - Arm SDOT. */
struct i8_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_neonsdot;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_V128

/** i8 inner-product distance kit - WebAssembly SIMD128. */
struct i8_v128_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_v128_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_v128;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_v128;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_v128;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_v128;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED

/** i8 inner-product distance kit - WebAssembly relaxed SIMD. */
struct i8_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_v128;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX

/** i8 inner-product distance kit - IBM POWER VSX. */
struct i8_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b8x16_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_powervsx;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_powervsx_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_powervsx_;
    static constexpr auto compute_norms = nk_reduce_moments_i8_serial;
    static constexpr std::size_t step_dimensions = 16;
};
#endif // NUMKONG_TARGET_POWERVSX

/** u4x2 inner-product distance kit - serial reference. Two unsigned nibbles per
 *  byte, so a step of 16 dimensions rides in a 64-bit vec. The dot lanes land in
 *  @c i32 like every other integral kit, even though @c dot_result_t is unsigned. */
struct u4x2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x16_state_serial_t;
    using step_vec_t = nk_b64_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u4x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_u4x16_update_serial;
    static constexpr auto load_step = nk_load_b64_serial_;
    static constexpr auto load_tail = nk_partial_load_b4x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_u4_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** u4x2 inner-product distance kit - Intel Haswell AVX2. */
struct u4x2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x32_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u4x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_u4x32_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x32_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u4_haswell;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE

/** u4x2 inner-product distance kit - Ice Lake VNNI. */
struct u4x2_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x128_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u4x128_init_icelake;
    static constexpr auto update_accumulator = nk_dot_u4x128_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b4x128_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_u4x128_finalize_icelake;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u4_skylake;
    static constexpr std::size_t step_dimensions = 128;
};
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_NEONSDOT

/** u4x2 inner-product distance kit - Arm UDOT. */
struct u4x2_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x32_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u4x32_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_u4x32_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x32_finalize_neonsdot;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_u4_serial;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_V128RELAXED

/** u4x2 inner-product distance kit - WebAssembly relaxed SIMD. */
struct u4x2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x32_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u4x32_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_u4x32_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x32_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_u4_serial;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** i4x2 inner-product distance kit - serial reference. */
struct i4x2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x16_state_serial_t;
    using step_vec_t = nk_b64_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i4x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_i4x16_update_serial;
    static constexpr auto load_step = nk_load_b64_serial_;
    static constexpr auto load_tail = nk_partial_load_b4x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x16_finalize_serial;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_i4_serial;
    static constexpr std::size_t step_dimensions = 16;
};

#if NUMKONG_TARGET_HASWELL

/** i4x2 inner-product distance kit - Haswell AVX2. */
struct i4x2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x32_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i4x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_i4x32_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x32_finalize_haswell;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_i4_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE

/** i4 inner-product distance kit - Ice Lake VNNI. */
struct i4x2_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x128_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i4x128_init_icelake;
    static constexpr auto update_accumulator = nk_dot_i4x128_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b4x128_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_i4x128_finalize_icelake;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_haswell_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_i4_skylake;
    static constexpr std::size_t step_dimensions = 128;
};
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_NEONSDOT

/** i4 inner-product distance kit - Arm SDOT. */
struct i4x2_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x32_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i4x32_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_i4x32_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x32_finalize_neonsdot;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_serial_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_i4_serial;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_V128RELAXED

/** i4x2 inner-product distance kit - WebAssembly relaxed SIMD. NumKong has no
 *  4-bit moments kernel for this tier, so the norms come from the serial one. */
struct i4x2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x32_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_i4x32_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_i4x32_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x32_finalize_v128relaxed;
    static constexpr auto angular_from_dot = nk_angular_through_i32_from_dot_v128_;
    static constexpr auto euclidean_from_dot = nk_euclidean_through_i32_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_i4_serial;
    static constexpr std::size_t step_dimensions = 32;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** u1x8 inner-product distance kit - serial reference (binary AND-popcount). */
struct u1x8_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_serial;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_serial;
    static constexpr auto hamming_from_dot = nk_hamming_u32x4_from_dot_serial_;
    static constexpr auto jaccard_from_dot = nk_jaccard_f32x4_from_dot_serial_;
    static constexpr auto compute_norms = nk_reduce_moments_u1_serial;
    static constexpr std::size_t step_dimensions = 128;
};

#if NUMKONG_TARGET_HASWELL

/** u1x8 inner-product distance kit - Intel Haswell AVX2 + POPCNT. */
struct u1x8_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_haswell;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_haswell;
    static constexpr auto hamming_from_dot = nk_hamming_u32x4_from_dot_haswell_;
    static constexpr auto jaccard_from_dot = nk_jaccard_f32x4_from_dot_haswell_;
    static constexpr auto compute_norms = nk_reduce_moments_u1_haswell;
    static constexpr std::size_t step_dimensions = 128;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE

/** u1x8 inner-product distance kit - Intel Ice Lake AVX-512 + VPOPCNTQ. */
struct u1x8_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x512_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u1x512_init_icelake;
    static constexpr auto update_accumulator = nk_dot_u1x512_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b1x512_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_u1x512_finalize_icelake;
    static constexpr auto hamming_from_dot = nk_hamming_u32x4_from_dot_icelake_;
    static constexpr auto jaccard_from_dot = nk_jaccard_f32x4_from_dot_icelake_;
    static constexpr auto compute_norms = nk_reduce_moments_u1_skylake;
    static constexpr std::size_t step_dimensions = 512;
};
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_NEON

/** u1x8 inner-product distance kit - Arm NEON (binary AND-popcount). */
struct u1x8_neon_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_neon_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_neon;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_neon;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_neon;
    static constexpr auto hamming_from_dot = nk_hamming_u32x4_from_dot_neon_;
    static constexpr auto jaccard_from_dot = nk_jaccard_f32x4_from_dot_neon_;
    static constexpr auto compute_norms = nk_reduce_moments_u1_neon;
    static constexpr std::size_t step_dimensions = 128;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_V128

/** u1x8 inner-product distance kit - WebAssembly SIMD128. NumKong has no
 *  binary moments kernel for this tier, so the norms come from the serial one. */
struct u1x8_v128_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_v128_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_v128;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_v128;
    static constexpr auto load_step = nk_load_b128_v128_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_v128;
    static constexpr auto hamming_from_dot = nk_hamming_u32x4_from_dot_v128_;
    static constexpr auto jaccard_from_dot = nk_jaccard_f32x4_from_dot_v128_;
    static constexpr auto compute_norms = nk_reduce_moments_u1_serial;
    static constexpr std::size_t step_dimensions = 128;
};
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_POWERVSX

/** u1x8 inner-product distance kit - IBM POWER VSX. NumKong has no binary moments
 *  kernel for this tier, so the norms come from the serial one. */
struct u1x8_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using finalized_vec_t = nk_b128_vec_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b1x128_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_powervsx;
    static constexpr auto hamming_from_dot = nk_hamming_u32x4_from_dot_powervsx_;
    static constexpr auto jaccard_from_dot = nk_jaccard_f32x4_from_dot_powervsx_;
    static constexpr auto compute_norms = nk_reduce_moments_u1_serial;
    static constexpr std::size_t step_dimensions = 128;
};
#endif // NUMKONG_TARGET_POWERVSX

/** The visited set's kit for one capability ceiling: the Skylake group scan where the
 *  class carries AVX-512, the serial walk otherwise. */
template <nk_capability_t caps_>
struct dense_visits_hash_set_chain {
    using kit_t = typename first_available_kit<                                                                   //
        serial_visits_hash_set_kit_t,                                                                             //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, u32_skylake_visits_hash_set_kit_t> //
        >::type;
};

/** Capability cascade for every scalar, shared by all its metrics. Reading order is the
 *  priority order; the first row whose @c available flag is true wins, else the fallback is selected. */
template <nk::numeric_dtype scalar_, nk_capability_t caps_>
struct dense_quads_chain;

template <nk_capability_t caps_>
struct dense_quads_chain<nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                       //
        bf16_serial_inner_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_inner_kit_t>,                   //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_NEONBFDOT && (caps_ & nk_cap_neonbfdot_k) != 0, bf16_neonbfdot_inner_kit_t>,       //
        chain_layer<NUMKONG_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, bf16_neon_inner_kit_t>,                      //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, bf16_v128relaxed_inner_kit_t>, //
        chain_layer<NUMKONG_TARGET_V128 && (caps_ & nk_cap_v128_k) != 0, bf16_v128_inner_kit_t>,                      //
        chain_layer<NUMKONG_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, bf16_powervsx_inner_kit_t>           //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        f16_serial_inner_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, f16_skylake_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, f16_haswell_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, f16_neonfhm_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, f16_neon_inner_kit_t>,                      //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, f16_v128relaxed_inner_kit_t>, //
        chain_layer<NUMKONG_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, f16_powervsx_inner_kit_t>           //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit< //
        e5m2_serial_inner_kit_t,                //
        chain_layer<NUMKONG_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e5m2_diamond_inner_kit_t>,
        chain_layer<NUMKONG_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e5m2_genoa_inner_kit_t>,                  //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e5m2_skylake_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e5m2_haswell_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e5m2_neonfp8_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e5m2_neonfhm_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e5m2_v128relaxed_inner_kit_t> //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit< //
        e4m3_serial_inner_kit_t,                //
        chain_layer<NUMKONG_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e4m3_diamond_inner_kit_t>,
        chain_layer<NUMKONG_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e4m3_genoa_inner_kit_t>,                  //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e4m3_skylake_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e4m3_haswell_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e4m3_neonfp8_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e4m3_neonfhm_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e4m3_v128relaxed_inner_kit_t> //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::e3m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        e3m2_serial_inner_kit_t,                                                                                     //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e3m2_skylake_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e3m2_haswell_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e3m2_neonfp8_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e3m2_neonsdot_inner_kit_t>,         //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e3m2_v128relaxed_inner_kit_t> //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::e2m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        e2m3_serial_inner_kit_t,                                                                                     //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e2m3_skylake_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, e2m3_sierra_inner_kit_t>,               //
        chain_layer<NUMKONG_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, e2m3_alder_inner_kit_t>,                  //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e2m3_haswell_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e2m3_neonfp8_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e2m3_neonsdot_inner_kit_t>,         //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e2m3_v128relaxed_inner_kit_t> //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        i8_serial_inner_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, i8_sierra_inner_kit_t>,                //
        chain_layer<NUMKONG_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, i8_alder_inner_kit_t>,                   //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i8_neonsdot_inner_kit_t>,          //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i8_v128relaxed_inner_kit_t>, //
        chain_layer<NUMKONG_TARGET_V128 && (caps_ & nk_cap_v128_k) != 0, i8_v128_inner_kit_t>,                      //
        chain_layer<NUMKONG_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, i8_powervsx_inner_kit_t>           //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::u8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        u8_serial_inner_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u8_icelake_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, u8_sierra_inner_kit_t>,                //
        chain_layer<NUMKONG_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, u8_alder_inner_kit_t>,                   //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u8_haswell_inner_kit_t>,             //
        chain_layer<NUMKONG_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u8_neonsdot_inner_kit_t>,          //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u8_v128relaxed_inner_kit_t>, //
        chain_layer<NUMKONG_TARGET_V128 && (caps_ & nk_cap_v128_k) != 0, u8_v128_inner_kit_t>,                      //
        chain_layer<NUMKONG_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u8_powervsx_inner_kit_t>           //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::u4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        u4x2_serial_inner_kit_t,                                                                                     //
        chain_layer<NUMKONG_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u4x2_icelake_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u4x2_haswell_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u4x2_neonsdot_inner_kit_t>,         //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u4x2_v128relaxed_inner_kit_t> //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        i4x2_serial_inner_kit_t,                                                                                     //
        chain_layer<NUMKONG_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i4x2_icelake_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i4x2_haswell_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i4x2_neonsdot_inner_kit_t>,         //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i4x2_v128relaxed_inner_kit_t> //
        >::type;

    /** The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct dense_quads_chain<nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                             //
        u1x8_serial_inner_kit_t,                                                                            //
        chain_layer<NUMKONG_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_inner_kit_t>,   //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_inner_kit_t>,   //
        chain_layer<NUMKONG_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, u1x8_neon_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_V128 && (caps_ & nk_cap_v128_k) != 0, u1x8_v128_inner_kit_t>,            //
        chain_layer<NUMKONG_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u1x8_powervsx_inner_kit_t> //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

/** Widens value rows to @c f32, weighs them by base-2 exponentials and folds them into a running sum, with the
 *  kernels every CPU runs. */
struct serial_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr auto cast = nk_cast_serial;
    static constexpr auto scale = nk_each_scale_f32_serial;
    static constexpr auto blend = nk_each_blend_f32_serial;

    /** Raises two to every one of @p values in place, as NumKong's attention weighs its logits. */
    static void exp2(std::array<f32_t, 8>& values) noexcept {
        for (f32_t& value : values) value = nk_f32_exp2_serial_(value);
    }
};

struct haswell_elementwise_kit_t;
struct skylake_elementwise_kit_t;
struct icelake_elementwise_kit_t;
struct sapphire_elementwise_kit_t;
struct neon_elementwise_kit_t;
struct rvv_elementwise_kit_t;
struct v128relaxed_elementwise_kit_t;
struct powervsx_elementwise_kit_t;

#if NUMKONG_TARGET_HASWELL
struct haswell_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr auto cast = nk_cast_haswell;
    static constexpr auto scale = nk_each_scale_f32_haswell;
    static constexpr auto blend = nk_each_blend_f32_haswell;

    static void exp2(std::array<f32_t, 8>& values) noexcept {
        _mm256_storeu_ps(values.data(), nk_exp2_f32x8_haswell_(_mm256_loadu_ps(values.data())));
    }
};

#if NUMKONG_TARGET_SKYLAKE

/** Skylake's casts and arithmetic over Haswell's exponentials, as one tile fills a single AVX2 register. */
struct skylake_elementwise_kit_t : haswell_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr auto cast = nk_cast_skylake;
    static constexpr auto scale = nk_each_scale_f32_skylake;
    static constexpr auto blend = nk_each_blend_f32_skylake;
};

#if NUMKONG_TARGET_ICELAKE

/** Icelake's casts over Skylake's arithmetic, which Icelake adds no kernels to. */
struct icelake_elementwise_kit_t : skylake_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    static constexpr auto cast = nk_cast_icelake;
};

#if NUMKONG_TARGET_SAPPHIRE

/** Sapphire's half-precision casts over Skylake's arithmetic. */
struct sapphire_elementwise_kit_t : icelake_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sapphire_k;
    static constexpr auto cast = nk_cast_sapphire;
};
#endif // NUMKONG_TARGET_SAPPHIRE
#endif // NUMKONG_TARGET_ICELAKE
#endif // NUMKONG_TARGET_SKYLAKE
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_NEON
struct neon_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr auto cast = nk_cast_neon;
    static constexpr auto scale = nk_each_scale_f32_neon;
    static constexpr auto blend = nk_each_blend_f32_neon;

    static void exp2(std::array<f32_t, 8>& values) noexcept {
        vst1q_f32(values.data(), nk_exp2_f32x4_neon_(vld1q_f32(values.data())));
        vst1q_f32(values.data() + 4, nk_exp2_f32x4_neon_(vld1q_f32(values.data() + 4)));
    }
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_RVV
struct rvv_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr auto cast = nk_cast_rvv;
    static constexpr auto scale = nk_each_scale_f32_rvv;
    static constexpr auto blend = nk_each_blend_f32_rvv;

    static void exp2(std::array<f32_t, 8>& values) noexcept {
        vfloat32m4_t const lanes = __riscv_vle32_v_f32m4(values.data(), values.size());
        __riscv_vse32_v_f32m4(values.data(), nk_exp2_f32m4_rvv_(lanes, values.size()), values.size());
    }
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
struct v128relaxed_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr auto cast = nk_cast_v128relaxed;
    static constexpr auto scale = nk_each_scale_f32_v128relaxed;
    static constexpr auto blend = nk_each_blend_f32_v128relaxed;

    static void exp2(std::array<f32_t, 8>& values) noexcept {
        wasm_v128_store(values.data(), nk_exp2_f32x4_v128relaxed_(wasm_v128_load(values.data())));
        wasm_v128_store(values.data() + 4, nk_exp2_f32x4_v128relaxed_(wasm_v128_load(values.data() + 4)));
    }
};
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX

/** POWER's casts over the serial arithmetic, which POWER adds no kernels to. */
struct powervsx_elementwise_kit_t : serial_elementwise_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    static constexpr auto cast = nk_cast_powervsx;
};
#endif // NUMKONG_TARGET_POWERVSX

/** The values kit for one capability ceiling. Reading order is the priority order. */
template <nk_capability_t caps_>
struct dense_elementwise_chain {
    using kit_t = typename first_available_kit<                                                               //
        serial_elementwise_kit_t,                                                                             //
        chain_layer<NUMKONG_TARGET_SAPPHIRE && (caps_ & nk_cap_sapphire_k) != 0, sapphire_elementwise_kit_t>, //
        chain_layer<NUMKONG_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, icelake_elementwise_kit_t>,    //
        chain_layer<NUMKONG_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, skylake_elementwise_kit_t>,    //
        chain_layer<NUMKONG_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, haswell_elementwise_kit_t>,    //
        chain_layer<NUMKONG_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, neon_elementwise_kit_t>,             //
        chain_layer<NUMKONG_TARGET_RVV && (caps_ & nk_cap_rvv_k) != 0, rvv_elementwise_kit_t>,                //
        chain_layer<NUMKONG_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0,
                    v128relaxed_elementwise_kit_t>,                                                          //
        chain_layer<NUMKONG_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, powervsx_elementwise_kit_t> //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

/** Scales each of @p heads @c f32 sums, at most eight, by its correction, then adds @p count value rows into it, each
 *  row's weights one tile over the heads. Each chunk of the rows widens once and stays in L1 while every head
 *  blends it. */
template <nk::numeric_dtype scalar_type_, nk_capability_t caps_>
void dense_values_step(f32_t* outputs, std::size_t heads, std::size_t columns, char const* const* rows,
                       std::array<f32_t, 8> const* weights, std::array<f32_t, 8> const& corrections,
                       std::size_t count) noexcept {
    using kit_t = typename dense_elementwise_chain<caps_>::kit_t;
    // A multiple of eight columns, so every chunk starts on a byte even for the packed kinds.
    constexpr std::size_t chunk_columns_k = 64;
    std::size_t const bits = nk_dtype_bits(scalar_type_::dtype());
    std::array<std::array<f32_t, chunk_columns_k>, 4> widened;
    assert(count <= widened.size() && heads <= corrections.size() && "Four nodes and eight heads per step");
    f32_t const one = 1, zero = 0;
    for (std::size_t first = 0; first < columns; first += chunk_columns_k) {
        std::size_t const width = min_of(chunk_columns_k, columns - first);
        [[maybe_unused]] nk_status_t status = nk_success_k;
        for (std::size_t row = 0; row != count; ++row) {
            status = kit_t::cast(rows[row] + first * bits / 8, scalar_type_::dtype(), widened[row].data(), nk_f32_k,
                                 width, nullptr);
            assert(status == nk_success_k && "The serial tier casts every dtype pair");
        }
        for (std::size_t head = 0; head != heads; ++head) {
            f32_t* const target = outputs + head * columns + first;
            if (corrections[head] != 1)
                status = kit_t::scale(target, width, &corrections[head], &zero, target, nullptr);
            assert(status == nk_success_k && "The serial tier scales on every CPU");
            for (std::size_t row = 0; row != count; ++row) {
                status = kit_t::blend(target, widened[row].data(), width, &one, &weights[row][head], target, nullptr);
                assert(status == nk_success_k && "The serial tier blends on every CPU");
            }
        }
    }
}

/** One finalize read two ways: the native dots, and the distances derived from them. */
template <typename dot_result_type_, std::size_t fan_out_>
struct dense_scores {

    /** Native dot products in kernel lane order, before any distance conversion. */
    std::array<dot_result_type_, fan_out_> dots;

    /** The distances derived from @c dots, lane for lane. */
    std::array<f32_t, fan_out_> distances;
};

/** Whether @p kit_ 's finalize subtracts the candidates' row sums - the DPBUSD compensation - so even a dot
 *  reads the stored norms. Detected on the raw function pointer, so kits keep @c reduce_accumulators as a
 *  direct @c nk_* alias rather than a wrapper. */
template <dense_quads_kit kit_>
inline constexpr bool dense_quads_compensated = std::is_invocable_v<                           //
    decltype(kit_::reduce_accumulators),                                                       //
    typename kit_::accumulator_t const*, typename kit_::accumulator_t const*,                  //
    typename kit_::accumulator_t const*, typename kit_::accumulator_t const*,                  //
    std::size_t, typename kit_::scalar_t::dot_result_t, typename kit_::finalized_vec_t const*, //
    typename kit_::finalized_vec_t*>;

/** Whether @p kit_ finalizes @p kernel: dot, and the metrics its throughs derive from it. */
template <dense_quads_kit kit_>
constexpr bool dense_kernel_supported(nk_kernel_kind_t kernel) noexcept {
    if constexpr (dense_bits_kit<kit_>)
        return kernel == nk_kernel_dot_k || kernel == nk_kernel_hamming_k || kernel == nk_kernel_jaccard_k;
    else return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
}

/** Whether a quads kit covers @p scalar_ under @p caps_ - the chains specialize only the scalars they adapt. */
template <typename scalar_, nk_capability_t caps_>
concept dense_quads_adapted = requires { typename dense_quads_chain<scalar_, caps_>::kit_t; };

/** Whether that kit's epilogue also finalizes @p kernel_. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept dense_quads_supported = dense_quads_adapted<scalar_, caps_> &&
                                dense_kernel_supported<typename dense_quads_chain<scalar_, caps_>::kit_t>(kernel_);

/** Widens the four lanes an epilogue wrote as @p lane_type_ into distances. */
template <typename lane_type_, typename finalized_vec_type_>
[[gnu::always_inline]] inline std::array<f32_t, 4> distances_of_(finalized_vec_type_ const& lanes) noexcept {
    static_assert(sizeof(finalized_vec_type_) == 4 * sizeof(lane_type_), "An epilogue writes four lanes");
    std::array<lane_type_, 4> const& typed = *reinterpret_cast<std::array<lane_type_, 4> const*>(&lanes);
    return {static_cast<f32_t>(typed[0]), static_cast<f32_t>(typed[1]), static_cast<f32_t>(typed[2]),
            static_cast<f32_t>(typed[3])};
}

/**
 *  @brief The metric's epilogue over one finalize: four @p dots, the query's squared norm and the
 *      candidates' @p sumsqs become four distances under the runtime @p kernel.
 *
 *  The one place the metric reaches, so one kit serves them all; a walk takes the same branch every quad.
 *  Each through writes its own lane type - the dot one the dots' width, Hamming @c u32, the rest @c f32.
 */
template <dense_quads_kit kit_>
[[gnu::always_inline]] inline std::array<f32_t, 4> convert_to_distances(
    nk_kernel_kind_t kernel, typename kit_::finalized_vec_t const& dots,
    typename kit_::scalar_t::dot_result_t query_sumsq, typename kit_::finalized_vec_t const& sumsqs) noexcept {
    using dot_result_t = typename kit_::scalar_t::dot_result_t;
    alignas(16) typename kit_::finalized_vec_t distances;
    if constexpr (dense_bits_kit<kit_>) switch (kernel) {
        case nk_kernel_hamming_k:
            kit_::hamming_from_dot(&dots, query_sumsq, &sumsqs, &distances);
            return distances_of_<nk_u32_t>(distances);
        case nk_kernel_jaccard_k:
            kit_::jaccard_from_dot(&dots, query_sumsq, &sumsqs, &distances);
            return distances_of_<nk_f32_t>(distances);
        default: break;
        }
    else switch (kernel) {
        case nk_kernel_angular_k:
            kit_::angular_from_dot(&dots, query_sumsq, &sumsqs, &distances);
            return distances_of_<nk_f32_t>(distances);
        case nk_kernel_euclidean_k:
            kit_::euclidean_from_dot(&dots, query_sumsq, &sumsqs, &distances);
            return distances_of_<nk_f32_t>(distances);
        default: break;
        }
    inner_product_distance_<dot_result_t>(&dots, query_sumsq, &sumsqs, &distances);
    return distances_of_<std::conditional_t<nk::is_integral_dtype<dot_result_t>(), nk_i32_t, nk_f32_t>>(distances);
}

/**
 *  @brief The only place the load, update, finalize, through, prefetch, demote choreography lives.
 *
 *  Lane fan-out is unrolled via @c std::index_sequence so the same body emits clean code
 *  for both @b 1-to-1 and @b 1-to-4. The state array is always sized 4 (NumKong's
 *  finalize is hardcoded fan-out-4); for 1-to-1 the bottom three slots are left as
 *  register noise and the resulting lanes 1-3 are dropped at extract time.
 */
template <std::size_t fan_out_, dense_quads_kit kit_, typename cache_hints_type_>
[[gnu::always_inline]] inline dense_scores<typename kit_::scalar_t::dot_result_t, fan_out_> invoke_scores_( //
    nk_kernel_kind_t kernel,                                                                                //
    nk::vector_view<typename kit_::scalar_t> query,                                                         //
    vector_norms<typename kit_::scalar_t::dot_result_t> query_norms,                                        //
    std::array<nk::vector_view<typename kit_::scalar_t>, fan_out_> candidates,                              //
    std::array<vector_norms<typename kit_::scalar_t::dot_result_t>, fan_out_> candidates_norms,             //
    std::array<nk::vector_view<typename kit_::scalar_t>, fan_out_> next_candidates,                         //
    [[maybe_unused]] cache_hints_type_ hints) noexcept {

    static_assert(fan_out_ == 1 || fan_out_ == 4, "Only 1-to-1 and 1-to-4 fan-outs are supported");
    using scalar_t = typename kit_::scalar_t;
    using accumulator_t = typename kit_::accumulator_t;
    using step_vec_t = typename kit_::step_vec_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_t>();

    // Always allocate & initialize all 4 state slots. The 4-way shape of NumKong's
    // `reduce_accumulators` is the deliberate ABI - one batch primitive per kernel, no
    // scalar variants - and its caller contract is that every slot it reads is initialized,
    // even when the 1-to-1 instantiation only extracts lane 0. Three zero-stores buy a
    // well-defined finalize on every path.
    alignas(64) accumulator_t accumulators[4];
    kit_::init_accumulator(&accumulators[0]);
    kit_::init_accumulator(&accumulators[1]);
    kit_::init_accumulator(&accumulators[2]);
    kit_::init_accumulator(&accumulators[3]);

    auto const* query_base = query.data();
    auto candidate_bases = [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
        return std::array<decltype(query.data()), fan_out_> {candidates[lane_index].data()...};
    }(std::make_index_sequence<fan_out_> {});
    [[maybe_unused]] auto next_bases = [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
        return std::array<decltype(query.data()), fan_out_> {next_candidates[lane_index].data()...};
    }(std::make_index_sequence<fan_out_> {});

    std::size_t const total_dimensions = query.size();
    std::size_t const tail_dimensions = total_dimensions % kit_::step_dimensions;
    std::size_t passed_dimensions = 0;

    alignas(64) step_vec_t query_step_vec;
    alignas(64) step_vec_t candidate_step_vecs[fan_out_];
    for (; passed_dimensions + kit_::step_dimensions <= total_dimensions; passed_dimensions += kit_::step_dimensions) {
        std::size_t const value_offset = passed_dimensions / dims_per_value;
        kit_::load_step(query_base + value_offset, &query_step_vec);
        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (kit_::load_step(candidate_bases[lane_index] + value_offset, &candidate_step_vecs[lane_index]), ...);
        }(std::make_index_sequence<fan_out_> {});

        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (kit_::update_accumulator(&accumulators[lane_index], query_step_vec, candidate_step_vecs[lane_index],
                                      passed_dimensions, kit_::step_dimensions),
             ...);
        }(std::make_index_sequence<fan_out_> {});

        // One prefetch per line of the next quad, at the depth this quad has reached.
        if constexpr (!std::is_same_v<cache_hints_type_, fu::standard_cache_hints_t>)
            if (value_offset * sizeof(scalar_t) % fu::default_alignment_k == 0)
                [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
                    (hints(next_bases[lane_index] + value_offset, fu::prefetch_line_k), ...);
                }(std::make_index_sequence<fan_out_> {});
    }

    if (tail_dimensions) {
        std::size_t const value_offset = passed_dimensions / dims_per_value;
        kit_::load_tail(query_base + value_offset, &query_step_vec, tail_dimensions);
        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (kit_::load_tail(candidate_bases[lane_index] + value_offset, &candidate_step_vecs[lane_index],
                             tail_dimensions),
             ...);
        }(std::make_index_sequence<fan_out_> {});

        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (kit_::update_accumulator(&accumulators[lane_index], query_step_vec, candidate_step_vecs[lane_index],
                                      passed_dimensions, tail_dimensions),
             ...);
        }(std::make_index_sequence<fan_out_> {});
    }

    // u8/i8 finalize takes 8 args (DPBUSD bias correction); float-path takes 6.
    using finalized_vec_t = typename kit_::finalized_vec_t;
    alignas(64) finalized_vec_t dot_products_vec;
    if constexpr (dense_quads_compensated<kit_>) {
        alignas(16) std::array<dot_result_t, 4> candidates_sums_array {};
        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            ((candidates_sums_array[lane_index] = candidates_norms[lane_index].sum), ...);
        }(std::make_index_sequence<fan_out_> {});
        kit_::reduce_accumulators(
            &accumulators[0], &accumulators[1], &accumulators[2], &accumulators[3], total_dimensions, query_norms.sum,
            reinterpret_cast<finalized_vec_t const*>(candidates_sums_array.data()), &dot_products_vec);
    }
    else {
        kit_::reduce_accumulators(&accumulators[0], &accumulators[1], &accumulators[2], &accumulators[3],
                                  total_dimensions, &dot_products_vec);
    }

    // Now infer the metric's distance from the inner product, in vec form.
    alignas(16) std::array<dot_result_t, 4> candidates_sumsqs_array {};
    [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
        ((candidates_sumsqs_array[lane_index] = candidates_norms[lane_index].sumsq), ...);
    }(std::make_index_sequence<fan_out_> {});
    std::array<f32_t, 4> const distances_array = convert_to_distances<kit_>(
        kernel, dot_products_vec, query_norms.sumsq,
        *reinterpret_cast<finalized_vec_t const*>(candidates_sumsqs_array.data()));

    static_assert(sizeof(finalized_vec_t) == 4 * sizeof(dot_result_t), "A finalize holds four dot lanes");
    std::array<dot_result_t, 4> const& dots_array = *reinterpret_cast<std::array<dot_result_t, 4> const*>(
        &dot_products_vec);
    dense_scores<dot_result_t, fan_out_> scores;
    [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
        ((scores.dots[lane_index] = dots_array[lane_index]), ...);
        ((scores.distances[lane_index] = distances_array[lane_index]), ...);
    }(std::make_index_sequence<fan_out_> {});
    return scores;
}

/** Compute @c (sum,sumsq) of @p vec via the kit's @c compute_norms - one source of truth, the kit. */
template <nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline vector_norms<typename scalar_::dot_result_t> dense_norms(
    nk::vector_view<scalar_> vec) noexcept {
    using kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    return invoke_dense_norms_(kit_t::compute_norms, vec);
}

/**
 *  @brief 1-to-1 distance from @c query to @c candidate under @p kernel.
 *
 *  Free function template - scalar deduced from the @c nk::vector_view; capability passed
 *  explicitly, the metric at runtime. The chosen kit is internal to the trampoline.
 */
template <nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline f32_t distance(                 //
    nk_kernel_kind_t kernel,                                  //
    nk::vector_view<scalar_> query,                           //
    vector_norms<typename scalar_::dot_result_t> query_norms, //
    nk::vector_view<scalar_> candidate,                       //
    vector_norms<typename scalar_::dot_result_t> candidate_norms) noexcept {
    using kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    return invoke_scores_<1, kit_t>( //
               kernel, query, query_norms, std::array<nk::vector_view<scalar_>, 1> {candidate},
               std::array {candidate_norms}, std::array<nk::vector_view<scalar_>, 1> {candidate},
               fu::standard_cache_hints_t {})
        .distances[0];
}

/** 1-to-4 distances from @c query to four @c candidates under @p kernel. */
template <nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline std::array<f32_t, 4> distances( //
    nk_kernel_kind_t kernel,                                  //
    nk::vector_view<scalar_> query,                           //
    vector_norms<typename scalar_::dot_result_t> query_norms, //
    std::array<nk::vector_view<scalar_>, 4> candidates,       //
    std::array<vector_norms<typename scalar_::dot_result_t>, 4> candidates_norms) noexcept {
    using kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    return invoke_scores_<4, kit_t>(kernel, query, query_norms, candidates, candidates_norms, candidates,
                                    fu::standard_cache_hints_t {})
        .distances;
}

/** 1-to-4 native dots and the distances derived from them, from one finalize, under @p kernel. */
template <nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline dense_scores<typename scalar_::dot_result_t, 4> scores( //
    nk_kernel_kind_t kernel,                                                          //
    nk::vector_view<scalar_> query,                                                   //
    vector_norms<typename scalar_::dot_result_t> query_norms,                         //
    std::array<nk::vector_view<scalar_>, 4> candidates,                               //
    std::array<vector_norms<typename scalar_::dot_result_t>, 4> candidates_norms) noexcept {
    using kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    return invoke_scores_<4, kit_t>(kernel, query, query_norms, candidates, candidates_norms, candidates,
                                    fu::standard_cache_hints_t {});
}

/** As @c scores, while streaming @p next_candidates toward L2 one line per line scored, so the walk's next quad
 *  arrives as this one finishes. */
template <nk_capability_t caps_, nk::numeric_dtype scalar_, typename cache_hints_type_>
[[gnu::always_inline]] inline dense_scores<typename scalar_::dot_result_t, 4> scores( //
    nk_kernel_kind_t kernel,                                                          //
    nk::vector_view<scalar_> query,                                                   //
    vector_norms<typename scalar_::dot_result_t> query_norms,                         //
    std::array<nk::vector_view<scalar_>, 4> candidates,                               //
    std::array<vector_norms<typename scalar_::dot_result_t>, 4> candidates_norms,     //
    std::array<nk::vector_view<scalar_>, 4> next_candidates,                          //
    cache_hints_type_ hints) noexcept {
    using kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    return invoke_scores_<4, kit_t>(kernel, query, query_norms, candidates, candidates_norms, next_candidates, hints);
}

#pragma region Query Corrections

/** The walks' default rescoring: none, so every node keeps the score the kit gave it. */
struct dummy_rescoring_t {
    template <typename ids_type_, typename keys_type_, typename norms_type_, typename scores_type_>
    void operator()(ids_type_ const&, keys_type_ const&, norms_type_ const&, scores_type_&,
                    std::size_t) const noexcept {}
};

/** Indexer heads as a walk's rescoring: each scored dot becomes Σₕ wₕ·ReLU(⟨qₕ, k⟩) and its distance 1 − score. */
template <nk_capability_t capability_, typename scalar_type_>
struct dense_indexer_rescoring {

    /** The heads of the one query row being walked. */
    dense_indexer_heads<scalar_type_> heads;

    template <typename dot_type_, std::size_t fan_out_>
    void operator()(std::array<dense_node_id_t, fan_out_> const&,
                    std::array<nk::vector_view<scalar_type_>, fan_out_> const& keys,
                    std::array<vector_norms<dot_type_>, fan_out_> const& norms,
                    dense_scores<dot_type_, fan_out_>& scores, std::size_t count) const noexcept {
        static_assert(!nk::is_integral_dtype<scalar_type_>(), "Indexer scoring needs floating-point keys");
        std::array<dot_type_, fan_out_> indexed {};
        for (std::size_t head = 0; head != heads.heads.extent(0); ++head) {
            dense_scores<dot_type_, fan_out_> const head_scores = usearch::scores<capability_, scalar_type_>(
                nk_kernel_dot_k, heads.heads.row(head).as_vector(), {}, keys, norms);
            dot_type_ const weight = dot_type_(heads.weights(0, head));
            for (std::size_t lane = 0; lane != count; ++lane)
                indexed[lane] += weight * max_of(head_scores.dots[lane], dot_type_ {});
        }
        for (std::size_t lane = 0; lane != count; ++lane) {
            scores.dots[lane] = indexed[lane];
            scores.distances[lane] = 1 - indexed[lane];
        }
    }
};

#pragma endregion Query Corrections

/** Empty tile kit: the scalar has no matrix-engine backend on this
 *  target, so eager refinement stays unavailable and callers keep the lazy 4-way loops. */
struct no_tiles_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_serial_k;
};

template <typename finalize_kit_type_>
struct u8_sapphireamx_tiles_kit;
template <typename finalize_kit_type_>
struct i8_sapphireamx_tiles_kit;
template <typename finalize_kit_type_>
struct bf16_sapphireamx_tiles_kit;

#if NUMKONG_TARGET_SAPPHIREAMX

/** u8 tile kit for Sapphire Rapids AMX - the 4-way trio one power-of-two wider:
 *  @c init_tile / @c update_tile / @c finalize_tile over 16×16 accumulator tiles, plus
 *  the gather & pack that feed them. Finalizes through @p finalize_kit_type_ - the very
 *  kit the lazy path picked - so eager & lazy distances agree bit-for-bit. An SME twin
 *  fills the same slots with @c svmopa over ZA tiles, a CUDA twin with @c mma fragments. */
template <typename finalize_kit_type_>
struct u8_sapphireamx_tiles_kit {
    inline static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    using finalize_kit_t = finalize_kit_type_;
    using scalar_t = nk::u8_t;
    using norms_t = vector_norms<typename scalar_t::dot_result_t>;
    using a_tile_t = nk_dots_u8_a16x64_sapphireamx_t;
    using b_tile_t = nk_dots_u8_b64x16_sapphireamx_t;
    using accumulator_tile_t = nk_dots_u8_state_sapphireamx_t;

    inline static constexpr std::size_t tile_rows_k = 16;
    inline static constexpr std::size_t tile_columns_k = 16;
    inline static constexpr std::size_t tile_depth_k = 64;

    /** Row tiles covering @p count rows. */
    static constexpr std::size_t row_tiles(std::size_t count) noexcept { return divide_round_up(count, tile_rows_k); }

    /** Column tiles covering @p count columns. */
    static constexpr std::size_t column_tiles(std::size_t count) noexcept {
        return divide_round_up(count, tile_columns_k);
    }

    /** Depth tiles covering @p dimensions. */
    static constexpr std::size_t depth_tiles(std::size_t dimensions) noexcept {
        return divide_round_up(dimensions, tile_depth_k);
    }

    /** Bytes of tile scratch one Gram over @p count rows of @p dimensions needs -
     *  the packed B-tiles plus one row-tile of A-strips - derived from this kit's own
     *  tile types, so every dtype & engine sizes itself. Includes one alignment margin:
     *  generic byte allocators promise 16, the tile loads demand 64, and @c scratch_of
     *  aligns the base up. */
    static constexpr std::size_t scratch_bytes(std::size_t count, std::size_t dimensions) noexcept {
        return row_tiles(count) * depth_tiles(dimensions) * sizeof(b_tile_t) +
               depth_tiles(dimensions) * sizeof(a_tile_t) + alignof(a_tile_t);
    }

    /** The two regions of one Gram's scratch: the packed B-tiles first, the A-strips after. */
    struct scratch_t {
        b_tile_t* b_tiles;
        a_tile_t* a_tiles;
    };

    /** Lays @p packed_tiles B-tiles and the A-strips after them over @p scratch, from its first
     *  address on the tiles' own alignment. */
    static scratch_t scratch_of(byte_t* scratch, std::size_t packed_tiles) noexcept {
        std::uintptr_t const base = round_up_to_multiple(reinterpret_cast<std::uintptr_t>(scratch),
                                                         std::uintptr_t {alignof(a_tile_t)});
        b_tile_t* const b_tiles = reinterpret_cast<b_tile_t*>(base);
        return {b_tiles, reinterpret_cast<a_tile_t*>(b_tiles + packed_tiles)};
    }

    /** One palette programming per refinement - never per tile. */
    static void configure_tiles() noexcept { nk_amx_tile_configure_sapphireamx_(); }

    /** Gathers one @c tile_depth_k -deep strip of all 16 @p rows - scattered views,
     *  each touched once per depth sweep: the tile @b is the staging, no contiguous
     *  matrix ever exists. Row pads repeat valid views, like the 4-way tails; the
     *  depth tail zero-fills, adding nothing to any dot. */
    static void gather_a_tile(a_tile_t& a_tile, std::array<nk::vector_view<nk::u8_t>, tile_rows_k> const& rows,
                              std::size_t depth_offset) noexcept {
        std::size_t const remaining = rows[0].size() - depth_offset;
        __mmask64 const depth_mask = remaining >= tile_depth_k ? ~__mmask64 {0} : (__mmask64 {1} << remaining) - 1u;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            __m512i const strip = _mm512_maskz_loadu_epi8(
                depth_mask, reinterpret_cast<nk_u8_t const*>(rows[row].data()) + depth_offset);
            _mm512_store_si512(a_tile.data[row], strip);
        }
    }

    /** The transpose network NumKong packs with: a gathered A-tile becomes its own
     *  quad-interleaved B-tile - the Gram symmetry in one call. */
    static void pack_b_tile(a_tile_t const& a_tile, b_tile_t& b_tile) noexcept {
        nk_dots_pack_u8_transposed_sapphireamx_(&a_tile, &b_tile);
    }

    /** Zeroes the accumulator ahead of a depth walk. */
    static void init_tile(accumulator_tile_t& tile) noexcept { nk_dots_u8_init_sapphireamx_(&tile); }

    /** tile += A × B for one depth step - @c TDPBUUD here. Deliberately single-step:
     *  callers interleave the next tile's gather & pack between updates, hiding fetch
     *  latency behind the matrix op - progressive fetching outweighs the accumulator's
     *  per-step round-trip through memory. */
    static void update_tile(accumulator_tile_t& tile, a_tile_t const& a_tile, b_tile_t const& b_tile) noexcept {
        nk_compiler_barrier_sapphireamx_();
        _tile_loadd(0, tile.data, 64);
        _tile_loadd(1, a_tile.data, 64);
        _tile_loadd(2, b_tile.data, 64);
        _tile_dpbuud(0, 1, 2);
        _tile_stored(0, tile.data, 64);
        nk_compiler_barrier_sapphireamx_();
    }

    /** Distances under @p kernel through the stored norms for all 16×16 lanes: 4-lane groups
     *  through the finalize kit's @c convert_to_distances - the epilogue of
     *  @c invoke_scores_, tiled. Pad lanes finalize too and are simply never read. */
    static void finalize_tile(nk_kernel_kind_t kernel, accumulator_tile_t const& tile,
                              std::array<norms_t, tile_rows_k> const& a_norms,
                              std::array<norms_t, tile_columns_k> const& b_norms, f32_t* distances,
                              std::size_t stride) noexcept {
        using finalized_vec_t = typename finalize_kit_t::finalized_vec_t;
        using dot_result_t = typename scalar_t::dot_result_t;
        alignas(64) dot_result_t column_sumsqs[tile_columns_k];
        for (std::size_t column = 0; column != tile_columns_k; ++column) column_sumsqs[column] = b_norms[column].sumsq;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            f32_t* const distances_row = distances + row * stride;
            for (std::size_t group = 0; group != tile_columns_k / 4; ++group) {
                finalized_vec_t dots_vec, sumsqs_vec;
                dots_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&tile.data[row][group * 4]));
                sumsqs_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&column_sumsqs[group * 4]));
                std::array<f32_t, 4> const lanes = convert_to_distances<finalize_kit_t>(kernel, dots_vec,
                                                                                        a_norms[row].sumsq, sumsqs_vec);
                _mm_storeu_ps(distances_row + group * 4, _mm_loadu_ps(lanes.data()));
            }
        }
    }
};

/** i8 tile kit for Sapphire Rapids AMX - the signed twin of the u8 kit, @c TDPBSSD
 *  over the same 16×64 strips and 16×16 accumulator tiles. Finalizes through
 *  @p finalize_kit_type_ - the very kit the lazy path picked - so eager & lazy distances
 *  agree bit-for-bit. */
template <typename finalize_kit_type_>
struct i8_sapphireamx_tiles_kit {
    inline static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    using finalize_kit_t = finalize_kit_type_;
    using scalar_t = nk::i8_t;
    using norms_t = vector_norms<typename scalar_t::dot_result_t>;
    using a_tile_t = nk_dots_i8_a16x64_sapphireamx_t;
    using b_tile_t = nk_dots_i8_b64x16_sapphireamx_t;
    using accumulator_tile_t = nk_dots_i8_state_sapphireamx_t;

    inline static constexpr std::size_t tile_rows_k = 16;
    inline static constexpr std::size_t tile_columns_k = 16;
    inline static constexpr std::size_t tile_depth_k = 64;

    /** Row tiles covering @p count rows. */
    static constexpr std::size_t row_tiles(std::size_t count) noexcept { return divide_round_up(count, tile_rows_k); }

    /** Column tiles covering @p count columns. */
    static constexpr std::size_t column_tiles(std::size_t count) noexcept {
        return divide_round_up(count, tile_columns_k);
    }

    /** Depth tiles covering @p dimensions. */
    static constexpr std::size_t depth_tiles(std::size_t dimensions) noexcept {
        return divide_round_up(dimensions, tile_depth_k);
    }

    /** Bytes of tile scratch one Gram over @p count rows of @p dimensions needs -
     *  the packed B-tiles plus one row-tile of A-strips - derived from this kit's own
     *  tile types, so every dtype & engine sizes itself. Includes one alignment margin:
     *  generic byte allocators promise 16, the tile loads demand 64, and @c scratch_of
     *  aligns the base up. */
    static constexpr std::size_t scratch_bytes(std::size_t count, std::size_t dimensions) noexcept {
        return row_tiles(count) * depth_tiles(dimensions) * sizeof(b_tile_t) +
               depth_tiles(dimensions) * sizeof(a_tile_t) + alignof(a_tile_t);
    }

    /** The two regions of one Gram's scratch: the packed B-tiles first, the A-strips after. */
    struct scratch_t {
        b_tile_t* b_tiles;
        a_tile_t* a_tiles;
    };

    /** Lays @p packed_tiles B-tiles and the A-strips after them over @p scratch, from its first
     *  address on the tiles' own alignment. */
    static scratch_t scratch_of(byte_t* scratch, std::size_t packed_tiles) noexcept {
        std::uintptr_t const base = round_up_to_multiple(reinterpret_cast<std::uintptr_t>(scratch),
                                                         std::uintptr_t {alignof(a_tile_t)});
        b_tile_t* const b_tiles = reinterpret_cast<b_tile_t*>(base);
        return {b_tiles, reinterpret_cast<a_tile_t*>(b_tiles + packed_tiles)};
    }

    /** One palette programming per refinement - never per tile. */
    static void configure_tiles() noexcept { nk_amx_tile_configure_sapphireamx_(); }

    /** Gathers one @c tile_depth_k -deep strip of all 16 @p rows - scattered views,
     *  each touched once per depth sweep: the tile @b is the staging, no contiguous
     *  matrix ever exists. Row pads repeat valid views, like the 4-way tails; the
     *  depth tail zero-fills, adding nothing to any dot. */
    static void gather_a_tile(a_tile_t& a_tile, std::array<nk::vector_view<nk::i8_t>, tile_rows_k> const& rows,
                              std::size_t depth_offset) noexcept {
        std::size_t const remaining = rows[0].size() - depth_offset;
        __mmask64 const depth_mask = remaining >= tile_depth_k ? ~__mmask64 {0} : (__mmask64 {1} << remaining) - 1u;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            __m512i const strip = _mm512_maskz_loadu_epi8(
                depth_mask, reinterpret_cast<nk_i8_t const*>(rows[row].data()) + depth_offset);
            _mm512_store_si512(a_tile.data[row], strip);
        }
    }

    /** The transpose network NumKong packs with: a gathered A-tile becomes its own
     *  quad-interleaved B-tile - the Gram symmetry in one call. */
    static void pack_b_tile(a_tile_t const& a_tile, b_tile_t& b_tile) noexcept {
        nk_dots_pack_i8_transposed_sapphireamx_(&a_tile, &b_tile);
    }

    /** Zeroes the accumulator ahead of a depth walk. */
    static void init_tile(accumulator_tile_t& tile) noexcept { nk_dots_i8_init_sapphireamx_(&tile); }

    /** tile += A × B for one depth step - @c TDPBSSD here. Deliberately single-step:
     *  callers interleave the next tile's gather & pack between updates, hiding fetch
     *  latency behind the matrix op - progressive fetching outweighs the accumulator's
     *  per-step round-trip through memory. */
    static void update_tile(accumulator_tile_t& tile, a_tile_t const& a_tile, b_tile_t const& b_tile) noexcept {
        nk_compiler_barrier_sapphireamx_();
        _tile_loadd(0, tile.data, 64);
        _tile_loadd(1, a_tile.data, 64);
        _tile_loadd(2, b_tile.data, 64);
        _tile_dpbssd(0, 1, 2);
        _tile_stored(0, tile.data, 64);
        nk_compiler_barrier_sapphireamx_();
    }

    /** Distances under @p kernel through the stored norms for all 16×16 lanes: 4-lane groups
     *  through the finalize kit's @c convert_to_distances - the epilogue of
     *  @c invoke_scores_, tiled. Pad lanes finalize too and are simply never read. */
    static void finalize_tile(nk_kernel_kind_t kernel, accumulator_tile_t const& tile,
                              std::array<norms_t, tile_rows_k> const& a_norms,
                              std::array<norms_t, tile_columns_k> const& b_norms, f32_t* distances,
                              std::size_t stride) noexcept {
        using finalized_vec_t = typename finalize_kit_t::finalized_vec_t;
        using dot_result_t = typename scalar_t::dot_result_t;
        alignas(64) dot_result_t column_sumsqs[tile_columns_k];
        for (std::size_t column = 0; column != tile_columns_k; ++column) column_sumsqs[column] = b_norms[column].sumsq;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            f32_t* const distances_row = distances + row * stride;
            for (std::size_t group = 0; group != tile_columns_k / 4; ++group) {
                finalized_vec_t dots_vec, sumsqs_vec;
                dots_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&tile.data[row][group * 4]));
                sumsqs_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&column_sumsqs[group * 4]));
                std::array<f32_t, 4> const lanes = convert_to_distances<finalize_kit_t>(kernel, dots_vec,
                                                                                        a_norms[row].sumsq, sumsqs_vec);
                _mm_storeu_ps(distances_row + group * 4, _mm_loadu_ps(lanes.data()));
            }
        }
    }
};

/** bf16 tile kit for Sapphire Rapids AMX - @c TDPBF16PS over 16×32 strips, pair-interleaved
 *  B-tiles and f32 accumulator tiles, half the depth of the byte kits per step. Finalizes
 *  through @p finalize_kit_type_ - the very kit the lazy path picked - so eager & lazy
 *  distances agree bit-for-bit. */
template <typename finalize_kit_type_>
struct bf16_sapphireamx_tiles_kit {
    inline static constexpr nk_capability_t capability_k = nk_cap_sapphireamx_k;
    using finalize_kit_t = finalize_kit_type_;
    using scalar_t = nk::bf16_t;
    using norms_t = vector_norms<typename scalar_t::dot_result_t>;
    using a_tile_t = nk_dots_bf16_a16x32_sapphireamx_t;
    using b_tile_t = nk_dots_bf16_b32x16_sapphireamx_t;
    using accumulator_tile_t = nk_dots_bf16_state_sapphireamx_t;

    inline static constexpr std::size_t tile_rows_k = 16;
    inline static constexpr std::size_t tile_columns_k = 16;
    inline static constexpr std::size_t tile_depth_k = 32;

    /** Row tiles covering @p count rows. */
    static constexpr std::size_t row_tiles(std::size_t count) noexcept { return divide_round_up(count, tile_rows_k); }

    /** Column tiles covering @p count columns. */
    static constexpr std::size_t column_tiles(std::size_t count) noexcept {
        return divide_round_up(count, tile_columns_k);
    }

    /** Depth tiles covering @p dimensions. */
    static constexpr std::size_t depth_tiles(std::size_t dimensions) noexcept {
        return divide_round_up(dimensions, tile_depth_k);
    }

    /** Bytes of tile scratch one Gram over @p count rows of @p dimensions needs -
     *  the packed B-tiles plus one row-tile of A-strips - derived from this kit's own
     *  tile types, so every dtype & engine sizes itself. Includes one alignment margin:
     *  generic byte allocators promise 16, the tile loads demand 64, and @c scratch_of
     *  aligns the base up. */
    static constexpr std::size_t scratch_bytes(std::size_t count, std::size_t dimensions) noexcept {
        return row_tiles(count) * depth_tiles(dimensions) * sizeof(b_tile_t) +
               depth_tiles(dimensions) * sizeof(a_tile_t) + alignof(a_tile_t);
    }

    /** The two regions of one Gram's scratch: the packed B-tiles first, the A-strips after. */
    struct scratch_t {
        b_tile_t* b_tiles;
        a_tile_t* a_tiles;
    };

    /** Lays @p packed_tiles B-tiles and the A-strips after them over @p scratch, from its first
     *  address on the tiles' own alignment. */
    static scratch_t scratch_of(byte_t* scratch, std::size_t packed_tiles) noexcept {
        std::uintptr_t const base = round_up_to_multiple(reinterpret_cast<std::uintptr_t>(scratch),
                                                         std::uintptr_t {alignof(a_tile_t)});
        b_tile_t* const b_tiles = reinterpret_cast<b_tile_t*>(base);
        return {b_tiles, reinterpret_cast<a_tile_t*>(b_tiles + packed_tiles)};
    }

    /** One palette programming per refinement - never per tile. */
    static void configure_tiles() noexcept { nk_amx_tile_configure_sapphireamx_(); }

    /** Gathers one @c tile_depth_k -deep strip of all 16 @p rows - scattered views,
     *  each touched once per depth sweep: the tile @b is the staging, no contiguous
     *  matrix ever exists. Row pads repeat valid views, like the 4-way tails; the
     *  depth tail zero-fills, adding nothing to any dot. */
    static void gather_a_tile(a_tile_t& a_tile, std::array<nk::vector_view<nk::bf16_t>, tile_rows_k> const& rows,
                              std::size_t depth_offset) noexcept {
        std::size_t const remaining = rows[0].size() - depth_offset;
        __mmask32 const depth_mask = remaining >= tile_depth_k ? ~__mmask32 {0} : (__mmask32 {1} << remaining) - 1u;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            __m512i const strip = _mm512_maskz_loadu_epi16(
                depth_mask, reinterpret_cast<nk_bf16_t const*>(rows[row].data()) + depth_offset);
            _mm512_store_si512(a_tile.data[row], strip);
        }
    }

    /** The transpose network NumKong packs with: a gathered A-tile becomes its own
     *  pair-interleaved B-tile - the Gram symmetry in one call. */
    static void pack_b_tile(a_tile_t const& a_tile, b_tile_t& b_tile) noexcept {
        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tile, &b_tile);
    }

    /** Zeroes the accumulator ahead of a depth walk. */
    static void init_tile(accumulator_tile_t& tile) noexcept { nk_dots_bf16_init_sapphireamx_(&tile); }

    /** tile += A × B for one depth step - @c TDPBF16PS here. Deliberately single-step:
     *  callers interleave the next tile's gather & pack between updates, hiding fetch
     *  latency behind the matrix op - progressive fetching outweighs the accumulator's
     *  per-step round-trip through memory. */
    static void update_tile(accumulator_tile_t& tile, a_tile_t const& a_tile, b_tile_t const& b_tile) noexcept {
        nk_compiler_barrier_sapphireamx_();
        _tile_loadd(0, tile.data, 64);
        _tile_loadd(1, a_tile.data, 64);
        _tile_loadd(2, b_tile.data, 64);
        _tile_dpbf16ps(0, 1, 2);
        _tile_stored(0, tile.data, 64);
        nk_compiler_barrier_sapphireamx_();
    }

    /** Distances under @p kernel through the stored norms for all 16×16 lanes: 4-lane groups
     *  through the finalize kit's @c convert_to_distances - the epilogue of
     *  @c invoke_scores_, tiled. Pad lanes finalize too and are simply never read. */
    static void finalize_tile(nk_kernel_kind_t kernel, accumulator_tile_t const& tile,
                              std::array<norms_t, tile_rows_k> const& a_norms,
                              std::array<norms_t, tile_columns_k> const& b_norms, f32_t* distances,
                              std::size_t stride) noexcept {
        using finalized_vec_t = typename finalize_kit_t::finalized_vec_t;
        using dot_result_t = typename scalar_t::dot_result_t;
        alignas(64) dot_result_t column_sumsqs[tile_columns_k];
        for (std::size_t column = 0; column != tile_columns_k; ++column) column_sumsqs[column] = b_norms[column].sumsq;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            f32_t* const distances_row = distances + row * stride;
            for (std::size_t group = 0; group != tile_columns_k / 4; ++group) {
                finalized_vec_t dots_vec, sumsqs_vec;
                dots_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&tile.data[row][group * 4]));
                sumsqs_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&column_sumsqs[group * 4]));
                std::array<f32_t, 4> const lanes = convert_to_distances<finalize_kit_t>(kernel, dots_vec,
                                                                                        a_norms[row].sumsq, sumsqs_vec);
                _mm_storeu_ps(distances_row + group * 4, _mm_loadu_ps(lanes.data()));
            }
        }
    }
};

#endif // NUMKONG_TARGET_SAPPHIREAMX

/** Tile-kernel capability cascade - a second, independent axis over the same
 *  @p caps_ bitmask as @c dense_quads_chain, so one machine picks icelake for 1-to-1
 *  kernels and sapphireamx for tiles simultaneously. The primary template resolves
 *  to @c no_tiles_kit_t, and only the specialized scalars ever offer a matrix-engine
 *  backend - callers keep their lazy loops everywhere else. */
template <nk::numeric_dtype scalar_, nk_capability_t caps_>
struct dense_tiles_chain {
    using kit_t = no_tiles_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = false;
};

template <nk_capability_t caps_>
struct dense_tiles_chain<nk::u8_t, caps_> {
    using kit_t = typename first_available_kit< //
        no_tiles_kit_t,                         //
        chain_layer<NUMKONG_TARGET_SAPPHIREAMX && (caps_ & nk_cap_sapphireamx_k) != 0,
                    u8_sapphireamx_tiles_kit<typename dense_quads_chain<nk::u8_t, caps_>::kit_t>>> //
        ::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = !std::is_same_v<kit_t, no_tiles_kit_t>;
};

template <nk_capability_t caps_>
struct dense_tiles_chain<nk::i8_t, caps_> {
    using kit_t = typename first_available_kit< //
        no_tiles_kit_t,                         //
        chain_layer<NUMKONG_TARGET_SAPPHIREAMX && (caps_ & nk_cap_sapphireamx_k) != 0,
                    i8_sapphireamx_tiles_kit<typename dense_quads_chain<nk::i8_t, caps_>::kit_t>>> //
        ::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = !std::is_same_v<kit_t, no_tiles_kit_t>;
};

template <nk_capability_t caps_>
struct dense_tiles_chain<nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit< //
        no_tiles_kit_t,                         //
        chain_layer<NUMKONG_TARGET_SAPPHIREAMX && (caps_ & nk_cap_sapphireamx_k) != 0,
                    bf16_sapphireamx_tiles_kit<typename dense_quads_chain<nk::bf16_t, caps_>::kit_t>>> //
        ::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = !std::is_same_v<kit_t, no_tiles_kit_t>;
};
} // namespace usearch
} // namespace unum
