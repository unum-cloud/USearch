/**
 *  @brief Compile-time-dispatched 1-to-1, 1-to-4, and 16×16 tiled distance kernels
 *      backed by NumKong.
 *
 *  @author Ash Vardanian
 *  @file inner_metric.hpp
 *  @date April 21, 2026
 *
 *  This header is the hot inner loop of @c basic_dense_hindex - everything called per
 *  graph-traversal step, per query, per candidate. Its job is to turn NumKong's stateful
 *  ISA-specific dot-product primitives into clean @c distance and @c distances entry
 *  points that callers can reach without naming an ISA, a state type, or a vector width.
 *
 *  @section File Organization
 *
 *  @b Norms - @c vector_norms carries per-vector @c sum (DPBUSD compensation) and @c sumsq
 *  (L2 norm, used by angular/euclidean). @c inner_norms<scalar, caps> dispatches to
 *  NumKong's @c nk_reduce_moments_* family at construction-time of the index.
 *
 *  @b Kits - one struct per @c (scalar,kernel,capability), with inheritance: the base
 *  inner kit (e.g. @c bf16_genoa_inner_kit_t) is the dot-kernel kit and bundles every
 *  shared piece - NumKong state type, operand vec type, finalized vec type, output lane
 *  type, cache-management hint policies, load/update/finalize/through function pointers,
 *  step-dimensions count. Per-metric variants (e.g. @c bf16_genoa_angular_kit_t) inherit
 *  the base and override only @c convert_to_distances (and @c distance_lane_t when the
 *  through narrows or widens). The shape is uniform - the @c inner_quads_kit concept enforces it.
 *
 *  @b Chains - one @c inner_quads_chain<kernel,scalar,caps> per supported tuple. A linear
 *  list of @c chain_layer rows fed to @c first_available_kit picks the first kit whose
 *  @c NK_TARGET_* is on AND whose runtime-caps bit is set, else falls back to serial.
 *  Unsupported @c (kernel,scalar) pairs simply have no specialization. Adding an ISA
 *  tier is one row and zero preprocessor lines.
 *
 *  @b Trampoline - @c invoke_distances_<fan_out,kit> is the one place the load, update,
 *  finalize, through, prefetch, demote choreography lives. Reads everything it needs -
 *  @c scalar_t, @c distance_lane_t, @c convert_to_distances, ... - off the kit. Lane fan-out
 *  is unrolled via @c std::index_sequence so the same body emits clean code for both
 *  @b 1-to-1 and @b 1-to-4. The state array is always sized 4 (NumKong's finalize is
 *  hardcoded fan-out-4); for 1-to-1 the bottom three slots are left as register noise
 *  and the resulting lanes 1-3 are dropped at extract time.
 *
 *  @b Entry-points - free function templates @c distance<kernel,caps> and
 *  @c distances<kernel,caps>, with @c scalar deduced from the @c nk::vector_view inputs.
 *  Call sites read @c distance<nk_kernel_angular_k,caps> - no per-spec method, no per-spec class.
 *
 *  @b Tiles - a second, independent engine axis over the same capability bitmask. Each
 *  @c inner_tiles_chain<kernel,scalar,caps> names a tile kit - @c no_tiles_kit_t unless a
 *  matrix engine is adapted for the tuple - and its @c has_tiles_k verdict gates every
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
 *  1. @b Avoid-call-overhead. All NumKong logic is inlined from @c NK_PUBLIC ISA-specific
 *     APIs - the @c NK_DYNAMIC stack-frame construction in the shared lib is skipped.
 *
 *  2. @b Amortize-arithmetic-reductions. Both Euclidean (L2) and Angular (Cos) distances
 *     can be inferred from the inner product if the norms are known. This means @b 2-by-D
 *     arithmetic operations per distance computation instead of @b 6-by-D on the hot path,
 *     by precomputing and storing one extra norm value per vector.
 *
 *  3. @b Amortize-memory-loads-and-mini-float-upcasting. When computing an angular distance
 *     between a query vector and M neighbors of a graph node, we might be upcasting
 *     mini-floats to Float32 @b 2-by-M times. Storing Float32 in the index kills the value
 *     proposition of memory reductions. Pre-upcasting M vectors at once washes the
 *     register file. So we expose @b 1-to-4 distances - the query half of the binary
 *     operation is loaded and upcast four times less frequently, shrinking the total
 *     loads & upcasts to @b M+M/4. This also paves the way for cache-friendly streaming
 *     loads, keeping L1/L2 reserved for graph-traversal data structures while the vectors
 *     themselves only go as high as the L3 / cross-core LLC.
 *
 *  4. @b Amortize-horizontal-reductions. On low-dimensional vectors (64D Matryoshka or
 *     128D GQA), the most expensive part of inner-products on x86 is the final horizontal
 *     reduction of accumulators. The 1-to-4 path lets us amortize each reduction with
 *     cross-register shuffles.
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

#include <array>
#include <type_traits>
#include <utility>

#include <numkong/numkong.h>
#include <numkong/types.hpp>
#include <numkong/vector.hpp>

#include "memory_hints.hpp"
#include "types.hpp"

/**
 *  @brief Per-step cache-hint feature gates for the @c invoke_distances_ trampoline.
 *
 *  Both default to off. With the gates off the kit's hint type aliases still resolve, but
 *  the call sites are elided via @c if @c constexpr - no PREFETCHT* or CLDEMOTE micro-ops
 *  reach the hot loop.
 *
 *  Defaults were picked from a @b synthetic single-threaded microbench. On SPR / Zen 4
 *  silicon per-step CLDEMOTE costs 30-50% throughput, and per-step PREFETCHT0/T2 is
 *  neutral versus the HW stream detector. The bottleneck is uop pressure, not memory
 *  latency. Real HNSW workloads can differ; flip @c USEARCH_USE_PREFETCH=1 on cores with
 *  weak HW prefetchers (Atom, in-order), and @c USEARCH_USE_DEMOTE=1 for multi-threaded
 *  search where a sibling thread on another core re-reads consumed candidate lines
 *  (CLDEMOTE's designed-for use case).
 */
#if !defined(USEARCH_USE_PREFETCH)
#define USEARCH_USE_PREFETCH 0
#endif
#if !defined(USEARCH_USE_DEMOTE)
#define USEARCH_USE_DEMOTE 0
#endif

namespace unum {
namespace usearch {

namespace nk = ashvardanian::numkong;

using f32_t = float;

/**
 *  @brief Per-vector header carrying L2 norm and row-sum (used by DPBUSD compensation).
 *  Both fields share width @c dot_result_type_ - the lane width every downstream finalize
 *  and through-from-dot helper consumes.
 */
template <typename dot_result_type_>
struct vector_norms {
    dot_result_type_ sum;
    dot_result_type_ sumsq;
};

/**
 *  @brief Per-lane @c 1 - dot in NumKong vec form. Exists to match the
 *  @c nk_*_through_*_from_dot_* signature so the trampoline routes every metric uniformly.
 */
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

/**
 *  @brief Compile-time concept for every @c inner_kit_t. The trampoline takes only a kit,
 *  so substitution errors land at the call site as a one-line constraint-not-satisfied.
 */
template <typename kit_>
concept inner_quads_kit = requires {
    typename kit_::scalar_t;
    typename kit_::accumulator_t;
    typename kit_::step_vec_t;
    typename kit_::finalized_vec_t;
    typename kit_::distance_lane_t;
    typename kit_::prefetch_query_t;
    typename kit_::prefetch_candidate_t;
    typename kit_::demote_candidate_t;
    { kit_::init_accumulator };
    { kit_::update_accumulator };
    { kit_::load_step };
    { kit_::load_tail };
    { kit_::reduce_accumulators };
    { kit_::convert_to_distances };
    { kit_::compute_norms };
    { kit_::step_dimensions } -> std::convertible_to<std::size_t>;
};

/**
 *  @brief Internal trampoline for using @c nk_reduce_moments NumKong APIs to compute L2 vector norms.
 *
 *  Returns @c vector_norms<dot_result_t> - the same width that downstream finalize and through helpers consume.
 *  For integer scalars this narrows the i64/u64 native output of @c nk_reduce_moments_* to @c i32/u32;
 *  for float scalars the @c static_cast is a no-op identity. Narrowing is safe for any reasonable embedding sizes.
 */
template <typename function_type_, nk::numeric_dtype scalar_type_>
inline vector_norms<typename scalar_type_::dot_result_t> invoke_inner_norms_( //
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
    function(reinterpret_cast<raw_scalar_t const*>(vec.data()), vec.size(), sizeof(scalar_type_),
             reinterpret_cast<raw_sum_t*>(&native_sum), reinterpret_cast<raw_sumsq_t*>(&native_sumsq));
    return {static_cast<dot_result_t>(native_sum), static_cast<dot_result_t>(native_sumsq)};
}

/**
 *  @brief Forward declarations for every "kit" regardless of @c NK_TARGET_* availability,
 *  so @c inner_quads_chain backend resolution chains name-resolve on every build.
 */
struct bf16_serial_inner_kit_t;
struct bf16_serial_angular_kit_t;
struct bf16_serial_euclidean_kit_t;
struct bf16_haswell_inner_kit_t;
struct bf16_haswell_angular_kit_t;
struct bf16_haswell_euclidean_kit_t;
struct bf16_skylake_inner_kit_t;
struct bf16_skylake_angular_kit_t;
struct bf16_skylake_euclidean_kit_t;
struct bf16_genoa_inner_kit_t;
struct bf16_genoa_angular_kit_t;
struct bf16_genoa_euclidean_kit_t;
struct bf16_neon_inner_kit_t;
struct bf16_neon_angular_kit_t;
struct bf16_neon_euclidean_kit_t;
struct bf16_neonbfdot_inner_kit_t;
struct bf16_neonbfdot_angular_kit_t;
struct bf16_neonbfdot_euclidean_kit_t;
struct bf16_v128relaxed_inner_kit_t;
struct bf16_v128relaxed_angular_kit_t;
struct bf16_v128relaxed_euclidean_kit_t;
struct bf16_powervsx_inner_kit_t;
struct bf16_powervsx_angular_kit_t;
struct bf16_powervsx_euclidean_kit_t;
struct f16_serial_inner_kit_t;
struct f16_serial_angular_kit_t;
struct f16_serial_euclidean_kit_t;
struct f16_haswell_inner_kit_t;
struct f16_haswell_angular_kit_t;
struct f16_haswell_euclidean_kit_t;
struct f16_skylake_inner_kit_t;
struct f16_skylake_angular_kit_t;
struct f16_skylake_euclidean_kit_t;
struct f16_neon_inner_kit_t;
struct f16_neon_angular_kit_t;
struct f16_neon_euclidean_kit_t;
struct f16_neonfhm_inner_kit_t;
struct f16_neonfhm_angular_kit_t;
struct f16_neonfhm_euclidean_kit_t;
struct f16_v128relaxed_inner_kit_t;
struct f16_v128relaxed_angular_kit_t;
struct f16_v128relaxed_euclidean_kit_t;
struct f16_powervsx_inner_kit_t;
struct f16_powervsx_angular_kit_t;
struct f16_powervsx_euclidean_kit_t;
struct e5m2_serial_inner_kit_t;
struct e5m2_serial_angular_kit_t;
struct e5m2_serial_euclidean_kit_t;
struct e5m2_diamond_inner_kit_t;
struct e5m2_diamond_angular_kit_t;
struct e5m2_diamond_euclidean_kit_t;
struct e5m2_haswell_inner_kit_t;
struct e5m2_haswell_angular_kit_t;
struct e5m2_haswell_euclidean_kit_t;
struct e5m2_skylake_inner_kit_t;
struct e5m2_skylake_angular_kit_t;
struct e5m2_skylake_euclidean_kit_t;
struct e5m2_genoa_inner_kit_t;
struct e5m2_genoa_angular_kit_t;
struct e5m2_genoa_euclidean_kit_t;
struct e5m2_neonfhm_inner_kit_t;
struct e5m2_neonfhm_angular_kit_t;
struct e5m2_neonfhm_euclidean_kit_t;
struct e5m2_neonfp8_inner_kit_t;
struct e5m2_neonfp8_angular_kit_t;
struct e5m2_neonfp8_euclidean_kit_t;
struct e5m2_v128relaxed_inner_kit_t;
struct e5m2_v128relaxed_angular_kit_t;
struct e5m2_v128relaxed_euclidean_kit_t;
struct e4m3_serial_inner_kit_t;
struct e4m3_serial_angular_kit_t;
struct e4m3_serial_euclidean_kit_t;
struct e4m3_diamond_inner_kit_t;
struct e4m3_diamond_angular_kit_t;
struct e4m3_diamond_euclidean_kit_t;
struct e4m3_haswell_inner_kit_t;
struct e4m3_haswell_angular_kit_t;
struct e4m3_haswell_euclidean_kit_t;
struct e4m3_skylake_inner_kit_t;
struct e4m3_skylake_angular_kit_t;
struct e4m3_skylake_euclidean_kit_t;
struct e4m3_genoa_inner_kit_t;
struct e4m3_genoa_angular_kit_t;
struct e4m3_genoa_euclidean_kit_t;
struct e4m3_neonfhm_inner_kit_t;
struct e4m3_neonfhm_angular_kit_t;
struct e4m3_neonfhm_euclidean_kit_t;
struct e4m3_neonfp8_inner_kit_t;
struct e4m3_neonfp8_angular_kit_t;
struct e4m3_neonfp8_euclidean_kit_t;
struct e4m3_v128relaxed_inner_kit_t;
struct e4m3_v128relaxed_angular_kit_t;
struct e4m3_v128relaxed_euclidean_kit_t;
struct e3m2_serial_inner_kit_t;
struct e3m2_serial_angular_kit_t;
struct e3m2_serial_euclidean_kit_t;
struct e3m2_haswell_inner_kit_t;
struct e3m2_haswell_angular_kit_t;
struct e3m2_haswell_euclidean_kit_t;
struct e3m2_skylake_inner_kit_t;
struct e3m2_skylake_angular_kit_t;
struct e3m2_skylake_euclidean_kit_t;
struct e3m2_neonsdot_inner_kit_t;
struct e3m2_neonsdot_angular_kit_t;
struct e3m2_neonsdot_euclidean_kit_t;
struct e3m2_neonfp8_inner_kit_t;
struct e3m2_neonfp8_angular_kit_t;
struct e3m2_neonfp8_euclidean_kit_t;
struct e3m2_v128relaxed_inner_kit_t;
struct e3m2_v128relaxed_angular_kit_t;
struct e3m2_v128relaxed_euclidean_kit_t;
struct e2m3_serial_inner_kit_t;
struct e2m3_serial_angular_kit_t;
struct e2m3_serial_euclidean_kit_t;
struct e2m3_haswell_inner_kit_t;
struct e2m3_haswell_angular_kit_t;
struct e2m3_haswell_euclidean_kit_t;
struct e2m3_skylake_inner_kit_t;
struct e2m3_skylake_angular_kit_t;
struct e2m3_skylake_euclidean_kit_t;
struct e2m3_alder_inner_kit_t;
struct e2m3_alder_angular_kit_t;
struct e2m3_alder_euclidean_kit_t;
struct e2m3_sierra_inner_kit_t;
struct e2m3_sierra_angular_kit_t;
struct e2m3_sierra_euclidean_kit_t;
struct e2m3_neonsdot_inner_kit_t;
struct e2m3_neonsdot_angular_kit_t;
struct e2m3_neonsdot_euclidean_kit_t;
struct e2m3_neonfp8_inner_kit_t;
struct e2m3_neonfp8_angular_kit_t;
struct e2m3_neonfp8_euclidean_kit_t;
struct e2m3_v128relaxed_inner_kit_t;
struct e2m3_v128relaxed_angular_kit_t;
struct e2m3_v128relaxed_euclidean_kit_t;
struct i8_serial_inner_kit_t;
struct i8_serial_angular_kit_t;
struct i8_serial_euclidean_kit_t;
struct i8_haswell_inner_kit_t;
struct i8_haswell_angular_kit_t;
struct i8_haswell_euclidean_kit_t;
struct i8_icelake_inner_kit_t;
struct i8_icelake_angular_kit_t;
struct i8_icelake_euclidean_kit_t;
struct i8_alder_inner_kit_t;
struct i8_alder_angular_kit_t;
struct i8_alder_euclidean_kit_t;
struct i8_sierra_inner_kit_t;
struct i8_sierra_angular_kit_t;
struct i8_sierra_euclidean_kit_t;
struct i8_neonsdot_inner_kit_t;
struct i8_neonsdot_angular_kit_t;
struct i8_neonsdot_euclidean_kit_t;
struct i8_v128relaxed_inner_kit_t;
struct i8_v128relaxed_angular_kit_t;
struct i8_v128relaxed_euclidean_kit_t;
struct i8_powervsx_inner_kit_t;
struct i8_powervsx_angular_kit_t;
struct i8_powervsx_euclidean_kit_t;
struct u8_serial_inner_kit_t;
struct u8_serial_angular_kit_t;
struct u8_serial_euclidean_kit_t;
struct u8_icelake_inner_kit_t;
struct u8_icelake_angular_kit_t;
struct u8_icelake_euclidean_kit_t;
struct u8_haswell_inner_kit_t;
struct u8_haswell_angular_kit_t;
struct u8_haswell_euclidean_kit_t;
struct u8_alder_inner_kit_t;
struct u8_alder_angular_kit_t;
struct u8_alder_euclidean_kit_t;
struct u8_sierra_inner_kit_t;
struct u8_sierra_angular_kit_t;
struct u8_sierra_euclidean_kit_t;
struct u8_neonsdot_inner_kit_t;
struct u8_neonsdot_angular_kit_t;
struct u8_neonsdot_euclidean_kit_t;
struct u8_v128relaxed_inner_kit_t;
struct u8_v128relaxed_angular_kit_t;
struct u8_v128relaxed_euclidean_kit_t;
struct u8_powervsx_inner_kit_t;
struct u8_powervsx_angular_kit_t;
struct u8_powervsx_euclidean_kit_t;
struct u4x2_serial_inner_kit_t;
struct u4x2_serial_angular_kit_t;
struct u4x2_serial_euclidean_kit_t;
struct u4x2_haswell_inner_kit_t;
struct u4x2_haswell_angular_kit_t;
struct u4x2_haswell_euclidean_kit_t;
struct u4x2_icelake_inner_kit_t;
struct u4x2_icelake_angular_kit_t;
struct u4x2_icelake_euclidean_kit_t;
struct u4x2_neonsdot_inner_kit_t;
struct u4x2_neonsdot_angular_kit_t;
struct u4x2_neonsdot_euclidean_kit_t;
struct u4x2_v128relaxed_inner_kit_t;
struct u4x2_v128relaxed_angular_kit_t;
struct u4x2_v128relaxed_euclidean_kit_t;
struct i4x2_serial_inner_kit_t;
struct i4x2_serial_angular_kit_t;
struct i4x2_serial_euclidean_kit_t;
struct i4x2_haswell_inner_kit_t;
struct i4x2_haswell_euclidean_kit_t;
struct i4x2_haswell_angular_kit_t;
struct i4x2_icelake_inner_kit_t;
struct i4x2_icelake_angular_kit_t;
struct i4x2_icelake_euclidean_kit_t;
struct i4x2_neonsdot_inner_kit_t;
struct i4x2_neonsdot_angular_kit_t;
struct i4x2_neonsdot_euclidean_kit_t;
struct i4x2_v128relaxed_inner_kit_t;
struct i4x2_v128relaxed_angular_kit_t;
struct i4x2_v128relaxed_euclidean_kit_t;
struct u1x8_serial_inner_kit_t;
struct u1x8_serial_hamming_kit_t;
struct u1x8_serial_jaccard_kit_t;
struct u1x8_haswell_inner_kit_t;
struct u1x8_haswell_hamming_kit_t;
struct u1x8_icelake_inner_kit_t;
struct u1x8_icelake_hamming_kit_t;
struct u1x8_haswell_jaccard_kit_t;
struct u1x8_icelake_jaccard_kit_t;
struct u1x8_neon_inner_kit_t;
struct u1x8_neon_hamming_kit_t;
struct u1x8_neon_jaccard_kit_t;
struct u1x8_v128relaxed_inner_kit_t;
struct u1x8_v128relaxed_hamming_kit_t;
struct u1x8_v128relaxed_jaccard_kit_t;
struct u1x8_powervsx_inner_kit_t;
struct u1x8_powervsx_hamming_kit_t;
struct u1x8_powervsx_jaccard_kit_t;

/** @brief bf16 inner-product distance kit - serial reference. */
struct bf16_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_serial;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief bf16 angular distance kit - serial reference. */
struct bf16_serial_angular_kit_t : public bf16_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief bf16 Euclidean (L2) distance kit - serial reference. */
struct bf16_serial_euclidean_kit_t : public bf16_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief bf16 inner-product distance kit - Intel Haswell AVX2. */
struct bf16_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x16_state_haswell_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x16_init_haswell;
    static constexpr auto update_accumulator = nk_dot_bf16x16_update_haswell;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b16x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x16_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_haswell;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief bf16 angular distance kit - Haswell. */
struct bf16_haswell_angular_kit_t : public bf16_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief bf16 Euclidean (L2) distance kit - Haswell. */
struct bf16_haswell_euclidean_kit_t : public bf16_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @brief bf16 inner-product distance kit - Intel Skylake-X AVX-512 Foundation. */
struct bf16_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x32_state_skylake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x32_init_skylake;
    static constexpr auto update_accumulator = nk_dot_bf16x32_update_skylake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b16x32_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x32_finalize_skylake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_skylake;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief bf16 angular distance kit - Skylake inner + Haswell through. */
struct bf16_skylake_angular_kit_t : public bf16_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief bf16 Euclidean (L2) distance kit - Skylake inner + Haswell through. */
struct bf16_skylake_euclidean_kit_t : public bf16_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
/** @brief bf16 inner-product distance kit - AMD Genoa AVX-512 + VNNI BF16. */
struct bf16_genoa_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x32_state_genoa_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x32_init_genoa;
    static constexpr auto update_accumulator = nk_dot_bf16x32_update_genoa;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b16x32_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x32_finalize_genoa;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_genoa;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief bf16 angular distance kit - Genoa inner + Haswell through. */
struct bf16_genoa_angular_kit_t : public bf16_genoa_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief bf16 Euclidean (L2) distance kit - Genoa inner + Haswell through. */
struct bf16_genoa_euclidean_kit_t : public bf16_genoa_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_GENOA

#if NK_TARGET_NEON

/** @brief bf16 inner-product distance kit - Arm NEON. */
struct bf16_neon_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_neon_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_neon;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_neon;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_neon;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief bf16 angular distance kit - Arm NEON. */
struct bf16_neon_angular_kit_t : public bf16_neon_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief bf16 Euclidean (L2) distance kit - Arm NEON. */
struct bf16_neon_euclidean_kit_t : public bf16_neon_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT

/** @brief bf16 inner-product distance kit - Arm BFDOT. */
struct bf16_neonbfdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonbfdot_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_neonbfdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_neonbfdot;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_neonbfdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_neonbfdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_neonbfdot;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief bf16 angular distance kit - Arm BFDOT. */
struct bf16_neonbfdot_angular_kit_t : public bf16_neonbfdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief bf16 Euclidean (L2) distance kit - Arm BFDOT. */
struct bf16_neonbfdot_euclidean_kit_t : public bf16_neonbfdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_V128RELAXED
/** @brief bf16 inner-product distance kit - WebAssembly relaxed SIMD. */
struct bf16_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_v128relaxed;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief bf16 angular distance kit - WebAssembly relaxed SIMD. */
struct bf16_v128relaxed_angular_kit_t : public bf16_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_v128relaxed_;
};

/** @brief bf16 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct bf16_v128relaxed_euclidean_kit_t : public bf16_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_POWERVSX
/**
 *  @brief bf16 inner-product distance kit - IBM POWER VSX. NumKong has no bf16 moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct bf16_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::bf16_t;
    using accumulator_t = nk_dot_bf16x8_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_bf16x8_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_bf16x8_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b16x8_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_bf16x8_finalize_powervsx;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_bf16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief bf16 angular distance kit - IBM POWER VSX. */
struct bf16_powervsx_angular_kit_t : public bf16_powervsx_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_powervsx_;
};

/** @brief bf16 Euclidean (L2) distance kit - IBM POWER VSX. */
struct bf16_powervsx_euclidean_kit_t : public bf16_powervsx_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_powervsx_;
};
#endif // NK_TARGET_POWERVSX

/** @brief f16 inner-product distance kit - serial reference. */
struct f16_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_serial;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief f16 angular distance kit - serial reference. */
struct f16_serial_angular_kit_t : public f16_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief f16 Euclidean (L2) distance kit - serial reference. */
struct f16_serial_euclidean_kit_t : public f16_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
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
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_haswell_;
    static constexpr auto update_accumulator = nk_dot_through_f32_update_haswell_;
    static constexpr auto load_step = nk_load_f16x8_to_f32x8_haswell_;
    static constexpr auto load_tail = nk_partial_load_f16x8_to_f32x8_haswell_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_haswell_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_haswell;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief f16 angular distance kit - Intel Haswell AVX2 + F16C. */
struct f16_haswell_angular_kit_t : public f16_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief f16 Euclidean (L2) distance kit - Intel Haswell AVX2 + F16C. */
struct f16_haswell_euclidean_kit_t : public f16_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @brief f16 inner-product distance kit - Intel Skylake-X AVX-512, widening on load. */
struct f16_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_through_f32_state_skylake_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_skylake_;
    static constexpr auto update_accumulator = nk_dot_through_f32_update_skylake_;
    static constexpr auto load_step = nk_load_f16x16_to_f32x16_skylake_;
    static constexpr auto load_tail = nk_partial_load_f16x16_to_f32x16_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_skylake_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_skylake;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief f16 angular distance kit - Skylake-X inner + Haswell through. */
struct f16_skylake_angular_kit_t : public f16_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief f16 Euclidean (L2) distance kit - Skylake-X inner + Haswell through. */
struct f16_skylake_euclidean_kit_t : public f16_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_NEON
/** @brief f16 inner-product distance kit - Arm NEON. */
struct f16_neon_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_neon_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_neon;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_neon;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_neon;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_neon;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief f16 angular distance kit - Arm NEON. */
struct f16_neon_angular_kit_t : public f16_neon_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_neon_;
};

/** @brief f16 Euclidean (L2) distance kit - Arm NEON. */
struct f16_neon_euclidean_kit_t : public f16_neon_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_neon_;
};
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONFHM
/**
 *  @brief f16 inner-product distance kit - Arm FP16 FML. NumKong has no f16 moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct f16_neonfhm_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_neonfhm_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_neonfhm;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_neonfhm;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b16x8_serial_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_neonfhm;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief f16 angular distance kit - Arm FP16 FML. */
struct f16_neonfhm_angular_kit_t : public f16_neonfhm_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief f16 Euclidean (L2) distance kit - Arm FP16 FML. */
struct f16_neonfhm_euclidean_kit_t : public f16_neonfhm_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_V128RELAXED
/**
 *  @brief f16 inner-product distance kit - WebAssembly relaxed SIMD, widening on load.
 *         Four dimensions per step, since the widened f32 quad fills the 128-bit vec.
 */
struct f16_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_through_f32x4_state_v128relaxed_t_;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32x4_init_v128relaxed_;
    static constexpr auto update_accumulator = nk_dot_through_f32x4_update_v128relaxed_;
    static constexpr auto load_step = nk_load_f16x4_to_f32x4_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_f16x4_to_f32x4_v128relaxed_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32x4_finalize_v128relaxed_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_v128relaxed;
    static constexpr std::size_t step_dimensions = 4;
};

/** @brief f16 angular distance kit - WebAssembly relaxed SIMD. */
struct f16_v128relaxed_angular_kit_t : public f16_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_v128relaxed_;
};

/** @brief f16 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct f16_v128relaxed_euclidean_kit_t : public f16_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_POWERVSX
/**
 *  @brief f16 inner-product distance kit - IBM POWER VSX. NumKong has no f16 moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct f16_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::f16_t;
    using accumulator_t = nk_dot_f16x8_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_f16x8_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_f16x8_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b16x8_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_f16x8_finalize_powervsx;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_f16_serial;
    static constexpr std::size_t step_dimensions = 8;
};

/** @brief f16 angular distance kit - IBM POWER VSX. */
struct f16_powervsx_angular_kit_t : public f16_powervsx_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_powervsx_;
};

/** @brief f16 Euclidean (L2) distance kit - IBM POWER VSX. */
struct f16_powervsx_euclidean_kit_t : public f16_powervsx_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_powervsx_;
};
#endif // NK_TARGET_POWERVSX

/** @brief e5m2 inner-product distance kit - serial reference. */
struct e5m2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e5m2x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e5m2x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e5m2x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e5m2 angular distance kit - serial reference. */
struct e5m2_serial_angular_kit_t : public e5m2_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e5m2 Euclidean (L2) distance kit - serial reference. */
struct e5m2_serial_euclidean_kit_t : public e5m2_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
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
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_haswell_;
    static constexpr auto update_accumulator = nk_dot_e5m2x32_update_haswell_;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_haswell_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e5m2 angular distance kit - Intel Haswell AVX2. */
struct e5m2_haswell_angular_kit_t : public e5m2_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e5m2 Euclidean (L2) distance kit - Intel Haswell AVX2. */
struct e5m2_haswell_euclidean_kit_t : public e5m2_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @brief e5m2 inner-product distance kit - Intel Skylake-X AVX-512, widening in the update. */
struct e5m2_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_f32_state_skylake_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_skylake_;
    static constexpr auto update_accumulator = nk_dot_e5m2x64_update_skylake_;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_skylake_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_skylake;
    static constexpr std::size_t step_dimensions = 64;
};

/** @brief e5m2 angular distance kit - Skylake-X inner + Haswell through. */
struct e5m2_skylake_angular_kit_t : public e5m2_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e5m2 Euclidean (L2) distance kit - Skylake-X inner + Haswell through. */
struct e5m2_skylake_euclidean_kit_t : public e5m2_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_DIAMOND
/**
 *  @brief e5m2 inner-product distance kit - Intel Diamond Rapids AVX10.2. The loaders
 *         widen each step to f16 with a native FP8 convert, so the dot rides the FP16
 *         accumulator. NumKong has no e5m2 moments kernel for this tier, so the norms
 *         come from the AVX-512 one.
 */
struct e5m2_diamond_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_diamond_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_f16_state_diamond_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f16_init_diamond_;
    static constexpr auto update_accumulator = nk_dot_through_f16_update_diamond_;
    static constexpr auto load_step = nk_load_e5m2x32_to_f16x32_diamond_;
    static constexpr auto load_tail = nk_partial_load_e5m2x32_to_f16x32_diamond_;
    static constexpr auto reduce_accumulators = nk_dot_through_f16_finalize_diamond_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_skylake;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e5m2 angular distance kit - Diamond Rapids inner + Haswell through. */
struct e5m2_diamond_angular_kit_t : public e5m2_diamond_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e5m2 Euclidean (L2) distance kit - Diamond Rapids inner + Haswell through. */
struct e5m2_diamond_euclidean_kit_t : public e5m2_diamond_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_DIAMOND

#if NK_TARGET_GENOA
/**
 *  @brief e5m2 inner-product distance kit - AMD Genoa AVX-512 + VNNI BF16. The loaders
 *         widen each step to bf16 so the dot rides the BF16 accumulator.
 */
struct e5m2_genoa_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_through_bf16_state_genoa_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_bf16_init_genoa_;
    static constexpr auto update_accumulator = nk_dot_through_bf16_update_genoa_;
    static constexpr auto load_step = nk_load_e5m2x32_to_bf16x32_icelake_;
    static constexpr auto load_tail = nk_partial_load_e5m2x32_to_bf16x32_icelake_;
    static constexpr auto reduce_accumulators = nk_dot_through_bf16_finalize_genoa_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_genoa;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e5m2 angular distance kit - Genoa inner + Haswell through. */
struct e5m2_genoa_angular_kit_t : public e5m2_genoa_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e5m2 Euclidean (L2) distance kit - Genoa inner + Haswell through. */
struct e5m2_genoa_euclidean_kit_t : public e5m2_genoa_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_GENOA

#if NK_TARGET_NEONFHM
/** @brief e5m2 inner-product distance kit - Arm FP16 FML. */
struct e5m2_neonfhm_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x16_state_neonfhm_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e5m2x16_init_neonfhm;
    static constexpr auto update_accumulator = nk_dot_e5m2x16_update_neonfhm;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e5m2x16_finalize_neonfhm;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_neonfhm;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e5m2 angular distance kit - Arm FP16 FML. */
struct e5m2_neonfhm_angular_kit_t : public e5m2_neonfhm_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e5m2 Euclidean (L2) distance kit - Arm FP16 FML. */
struct e5m2_neonfhm_euclidean_kit_t : public e5m2_neonfhm_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONFP8
/**
 *  @brief e5m2 inner-product distance kit - Arm FP8 DOT. NumKong has no e5m2 moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct e5m2_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e5m2x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e5m2x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e5m2x16_finalize_neonfp8;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e5m2 angular distance kit - Arm FP8 DOT. */
struct e5m2_neonfp8_angular_kit_t : public e5m2_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e5m2 Euclidean (L2) distance kit - Arm FP8 DOT. */
struct e5m2_neonfp8_euclidean_kit_t : public e5m2_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_V128RELAXED
/**
 *  @brief e5m2 inner-product distance kit - WebAssembly relaxed SIMD, widening on load.
 *         Four dimensions per step, since the widened f32 quad fills the 128-bit vec.
 */
struct e5m2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e5m2_t;
    using accumulator_t = nk_dot_e5m2x4_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32x4_init_v128relaxed_;
    static constexpr auto update_accumulator = nk_dot_through_f32x4_update_v128relaxed_;
    static constexpr auto load_step = nk_load_e5m2x4_to_f32x4_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_e5m2x4_to_f32x4_v128relaxed_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32x4_finalize_v128relaxed_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e5m2_v128relaxed;
    static constexpr std::size_t step_dimensions = 4;
};

/** @brief e5m2 angular distance kit - WebAssembly relaxed SIMD. */
struct e5m2_v128relaxed_angular_kit_t : public e5m2_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_v128relaxed_;
};

/** @brief e5m2 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct e5m2_v128relaxed_euclidean_kit_t : public e5m2_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

/** @brief e4m3 inner-product distance kit - serial reference. */
struct e4m3_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e4m3x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e4m3x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e4m3x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e4m3 angular distance kit - serial reference. */
struct e4m3_serial_angular_kit_t : public e4m3_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e4m3 Euclidean (L2) distance kit - serial reference. */
struct e4m3_serial_euclidean_kit_t : public e4m3_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief e4m3 inner-product distance kit - Intel Haswell AVX2, widening in the update. */
struct e4m3_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_f32_state_haswell_t_;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_haswell_;
    static constexpr auto update_accumulator = nk_dot_e4m3x32_update_haswell_;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_haswell_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e4m3 angular distance kit - Intel Haswell AVX2. */
struct e4m3_haswell_angular_kit_t : public e4m3_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e4m3 Euclidean (L2) distance kit - Intel Haswell AVX2. */
struct e4m3_haswell_euclidean_kit_t : public e4m3_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @brief e4m3 inner-product distance kit - Intel Skylake-X AVX-512, widening on load. */
struct e4m3_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_f32_state_skylake_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32_init_skylake_;
    static constexpr auto update_accumulator = nk_dot_through_f32_update_skylake_;
    static constexpr auto load_step = nk_load_e4m3x16_to_f32x16_skylake_;
    static constexpr auto load_tail = nk_partial_load_e4m3x16_to_f32x16_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32_finalize_skylake_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_skylake;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e4m3 angular distance kit - Skylake-X inner + Haswell through. */
struct e4m3_skylake_angular_kit_t : public e4m3_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e4m3 Euclidean (L2) distance kit - Skylake-X inner + Haswell through. */
struct e4m3_skylake_euclidean_kit_t : public e4m3_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_DIAMOND
/**
 *  @brief e4m3 inner-product distance kit - Intel Diamond Rapids AVX10.2. The loaders
 *         widen each step to f16 with a native FP8 convert, so the dot rides the FP16
 *         accumulator. NumKong has no e4m3 moments kernel for this tier, so the norms
 *         come from the AVX-512 one.
 */
struct e4m3_diamond_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_diamond_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_f16_state_diamond_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f16_init_diamond_;
    static constexpr auto update_accumulator = nk_dot_through_f16_update_diamond_;
    static constexpr auto load_step = nk_load_e4m3x32_to_f16x32_diamond_;
    static constexpr auto load_tail = nk_partial_load_e4m3x32_to_f16x32_diamond_;
    static constexpr auto reduce_accumulators = nk_dot_through_f16_finalize_diamond_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_skylake;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e4m3 angular distance kit - Diamond Rapids inner + Haswell through. */
struct e4m3_diamond_angular_kit_t : public e4m3_diamond_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e4m3 Euclidean (L2) distance kit - Diamond Rapids inner + Haswell through. */
struct e4m3_diamond_euclidean_kit_t : public e4m3_diamond_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_DIAMOND

#if NK_TARGET_GENOA
/**
 *  @brief e4m3 inner-product distance kit - AMD Genoa AVX-512 + VNNI BF16. The loaders
 *         widen each step to bf16 so the dot rides the BF16 accumulator.
 */
struct e4m3_genoa_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_genoa_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_through_bf16_state_genoa_t_;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_bf16_init_genoa_;
    static constexpr auto update_accumulator = nk_dot_through_bf16_update_genoa_;
    static constexpr auto load_step = nk_load_e4m3x32_to_bf16x32_icelake_;
    static constexpr auto load_tail = nk_partial_load_e4m3x32_to_bf16x32_icelake_;
    static constexpr auto reduce_accumulators = nk_dot_through_bf16_finalize_genoa_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_genoa;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e4m3 angular distance kit - Genoa inner + Haswell through. */
struct e4m3_genoa_angular_kit_t : public e4m3_genoa_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e4m3 Euclidean (L2) distance kit - Genoa inner + Haswell through. */
struct e4m3_genoa_euclidean_kit_t : public e4m3_genoa_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_GENOA

#if NK_TARGET_NEONFHM
/** @brief e4m3 inner-product distance kit - Arm FP16 FML. */
struct e4m3_neonfhm_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfhm_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x16_state_neonfhm_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e4m3x16_init_neonfhm;
    static constexpr auto update_accumulator = nk_dot_e4m3x16_update_neonfhm;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e4m3x16_finalize_neonfhm;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_neonfhm;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e4m3 angular distance kit - Arm FP16 FML. */
struct e4m3_neonfhm_angular_kit_t : public e4m3_neonfhm_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e4m3 Euclidean (L2) distance kit - Arm FP16 FML. */
struct e4m3_neonfhm_euclidean_kit_t : public e4m3_neonfhm_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONFP8
/**
 *  @brief e4m3 inner-product distance kit - Arm FP8 DOT. NumKong has no e4m3 moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct e4m3_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e4m3x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e4m3x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e4m3x16_finalize_neonfp8;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e4m3 angular distance kit - Arm FP8 DOT. */
struct e4m3_neonfp8_angular_kit_t : public e4m3_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e4m3 Euclidean (L2) distance kit - Arm FP8 DOT. */
struct e4m3_neonfp8_euclidean_kit_t : public e4m3_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_V128RELAXED
/**
 *  @brief e4m3 inner-product distance kit - WebAssembly relaxed SIMD, widening on load.
 *         Four dimensions per step, since the widened f32 quad fills the 128-bit vec.
 */
struct e4m3_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e4m3_t;
    using accumulator_t = nk_dot_e4m3x4_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_through_f32x4_init_v128relaxed_;
    static constexpr auto update_accumulator = nk_dot_through_f32x4_update_v128relaxed_;
    static constexpr auto load_step = nk_load_e4m3x4_to_f32x4_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_e4m3x4_to_f32x4_v128relaxed_;
    static constexpr auto reduce_accumulators = nk_dot_through_f32x4_finalize_v128relaxed_;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e4m3_v128relaxed;
    static constexpr std::size_t step_dimensions = 4;
};

/** @brief e4m3 angular distance kit - WebAssembly relaxed SIMD. */
struct e4m3_v128relaxed_angular_kit_t : public e4m3_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_v128relaxed_;
};

/** @brief e4m3 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct e4m3_v128relaxed_euclidean_kit_t : public e4m3_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

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
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e3m2 angular distance kit - serial reference. */
struct e3m2_serial_angular_kit_t : public e3m2_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e3m2 Euclidean (L2) distance kit - serial reference. */
struct e3m2_serial_euclidean_kit_t : public e3m2_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief e3m2 inner-product distance kit - Intel Haswell AVX2. */
struct e3m2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x32_state_haswell_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_e3m2x32_update_haswell;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x32_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e3m2 angular distance kit - Intel Haswell AVX2. */
struct e3m2_haswell_angular_kit_t : public e3m2_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e3m2 Euclidean (L2) distance kit - Intel Haswell AVX2. */
struct e3m2_haswell_euclidean_kit_t : public e3m2_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @brief e3m2 inner-product distance kit - Intel Skylake-X AVX-512 Foundation. */
struct e3m2_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x64_state_skylake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x64_init_skylake;
    static constexpr auto update_accumulator = nk_dot_e3m2x64_update_skylake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x64_finalize_skylake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_skylake;
    static constexpr std::size_t step_dimensions = 64;
};

/** @brief e3m2 angular distance kit - Skylake-X inner + Haswell through. */
struct e3m2_skylake_angular_kit_t : public e3m2_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e3m2 Euclidean (L2) distance kit - Skylake-X inner + Haswell through. */
struct e3m2_skylake_euclidean_kit_t : public e3m2_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_NEONSDOT
/**
 *  @brief e3m2 inner-product distance kit - Arm SDOT. NumKong has no e3m2 moments kernel
 *         for this tier, so the norms come from the serial one.
 */
struct e3m2_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_neonsdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e3m2 angular distance kit - Arm SDOT. */
struct e3m2_neonsdot_angular_kit_t : public e3m2_neonsdot_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e3m2 Euclidean (L2) distance kit - Arm SDOT. */
struct e3m2_neonsdot_euclidean_kit_t : public e3m2_neonsdot_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFP8
/**
 *  @brief e3m2 inner-product distance kit - Arm FP8 DOT. NumKong has no e3m2 moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct e3m2_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_neonfp8;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e3m2 angular distance kit - Arm FP8 DOT. */
struct e3m2_neonfp8_angular_kit_t : public e3m2_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e3m2 Euclidean (L2) distance kit - Arm FP8 DOT. */
struct e3m2_neonfp8_euclidean_kit_t : public e3m2_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_V128RELAXED
/** @brief e3m2 inner-product distance kit - WebAssembly relaxed SIMD. */
struct e3m2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e3m2_t;
    using accumulator_t = nk_dot_e3m2x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e3m2x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_e3m2x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e3m2x16_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e3m2_v128relaxed;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e3m2 angular distance kit - WebAssembly relaxed SIMD. */
struct e3m2_v128relaxed_angular_kit_t : public e3m2_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_v128relaxed_;
};

/** @brief e3m2 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct e3m2_v128relaxed_euclidean_kit_t : public e3m2_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

/** @brief e2m3 inner-product distance kit - serial reference. One element per byte. */
struct e2m3_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e2m3 angular distance kit - serial reference. */
struct e2m3_serial_angular_kit_t : public e2m3_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e2m3 Euclidean (L2) distance kit - serial reference. */
struct e2m3_serial_euclidean_kit_t : public e2m3_serial_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief e2m3 inner-product distance kit - Intel Haswell AVX2. */
struct e2m3_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x32_state_haswell_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_e2m3x32_update_haswell;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x32_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e2m3 angular distance kit - Intel Haswell AVX2. */
struct e2m3_haswell_angular_kit_t : public e2m3_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e2m3 Euclidean (L2) distance kit - Intel Haswell AVX2. */
struct e2m3_haswell_euclidean_kit_t : public e2m3_haswell_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @brief e2m3 inner-product distance kit - Intel Skylake-X AVX-512 Foundation. */
struct e2m3_skylake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x64_state_skylake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x64_init_skylake;
    static constexpr auto update_accumulator = nk_dot_e2m3x64_update_skylake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x64_finalize_skylake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_skylake;
    static constexpr std::size_t step_dimensions = 64;
};

/** @brief e2m3 angular distance kit - Skylake-X inner + Haswell through. */
struct e2m3_skylake_angular_kit_t : public e2m3_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e2m3 Euclidean (L2) distance kit - Skylake-X inner + Haswell through. */
struct e2m3_skylake_euclidean_kit_t : public e2m3_skylake_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ALDER
/** @brief e2m3 inner-product distance kit - Intel Alder Lake AVX-VNNI. */
struct e2m3_alder_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_alder_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x32_state_alder_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x32_init_alder;
    static constexpr auto update_accumulator = nk_dot_e2m3x32_update_alder;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x32_finalize_alder;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_alder;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e2m3 angular distance kit - Alder Lake inner + Haswell through. */
struct e2m3_alder_angular_kit_t : public e2m3_alder_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e2m3 Euclidean (L2) distance kit - Alder Lake inner + Haswell through. */
struct e2m3_alder_euclidean_kit_t : public e2m3_alder_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
/** @brief e2m3 inner-product distance kit - Intel Sierra Forest AVX-VNNI. */
struct e2m3_sierra_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sierra_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x32_state_sierra_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x32_init_sierra;
    static constexpr auto update_accumulator = nk_dot_e2m3x32_update_sierra;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x32_finalize_sierra;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_sierra;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief e2m3 angular distance kit - Sierra Forest inner + Haswell through. */
struct e2m3_sierra_angular_kit_t : public e2m3_sierra_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_haswell_;
};

/** @brief e2m3 Euclidean (L2) distance kit - Sierra Forest inner + Haswell through. */
struct e2m3_sierra_euclidean_kit_t : public e2m3_sierra_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_haswell_;
};
#endif // NK_TARGET_SIERRA

#if NK_TARGET_NEONSDOT
/** @brief e2m3 inner-product distance kit - Arm SDOT. */
struct e2m3_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_neonsdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e2m3 angular distance kit - Arm SDOT. */
struct e2m3_neonsdot_angular_kit_t : public e2m3_neonsdot_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e2m3 Euclidean (L2) distance kit - Arm SDOT. */
struct e2m3_neonsdot_euclidean_kit_t : public e2m3_neonsdot_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFP8
/** @brief e2m3 inner-product distance kit - Arm FP8 DOT. */
struct e2m3_neonfp8_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonfp8_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_neonfp8_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_neonfp8;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_neonfp8;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_neonfp8;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e2m3 angular distance kit - Arm FP8 DOT. */
struct e2m3_neonfp8_angular_kit_t : public e2m3_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_serial_;
};

/** @brief e2m3 Euclidean (L2) distance kit - Arm FP8 DOT. */
struct e2m3_neonfp8_euclidean_kit_t : public e2m3_neonfp8_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_serial_;
};
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_V128RELAXED
/** @brief e2m3 inner-product distance kit - WebAssembly relaxed SIMD. */
struct e2m3_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::e2m3_t;
    using accumulator_t = nk_dot_e2m3x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_f32_t;
    static constexpr auto init_accumulator = nk_dot_e2m3x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_e2m3x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_e2m3x16_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_e2m3_v128relaxed;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief e2m3 angular distance kit - WebAssembly relaxed SIMD. */
struct e2m3_v128relaxed_angular_kit_t : public e2m3_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_angular_through_f32_from_dot_v128relaxed_;
};

/** @brief e2m3 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct e2m3_v128relaxed_euclidean_kit_t : public e2m3_v128relaxed_inner_kit_t {
    static constexpr auto convert_to_distances = nk_euclidean_through_f32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

/** @brief u8 inner-product distance kit - serial reference. */
struct u8_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief u8 angular distance kit - serial reference. Through writes f32. */
struct u8_serial_angular_kit_t : public u8_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief u8 Euclidean (L2) distance kit - serial reference. Through writes f32. */
struct u8_serial_euclidean_kit_t : public u8_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};

#if NK_TARGET_HASWELL

/** @brief u8 inner-product distance kit - Haswell AVX2. */
struct u8_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_haswell;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_haswell;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief u8 angular distance kit - Haswell AVX2. */
struct u8_haswell_angular_kit_t : public u8_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief u8 Euclidean (L2) distance kit - Haswell AVX2. */
struct u8_haswell_euclidean_kit_t : public u8_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
/** @brief u8 inner-product distance kit - Intel Ice Lake AVX-512 + VNNI VPDPBUSD. */
struct u8_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x64_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x64_init_icelake;
    static constexpr auto update_accumulator = nk_dot_u8x64_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_u8x64_finalize_icelake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_icelake;
    static constexpr std::size_t step_dimensions = 64;
};

/** @brief u8 angular distance kit - Ice Lake VNNI. */
struct u8_icelake_angular_kit_t : public u8_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief u8 Euclidean (L2) distance kit - Ice Lake VNNI. */
struct u8_icelake_euclidean_kit_t : public u8_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_ALDER
/** @brief u8 inner-product distance kit - Intel Alder Lake AVX-VNNI. */
struct u8_alder_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_alder_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x32_state_alder_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x32_init_alder;
    static constexpr auto update_accumulator = nk_dot_u8x32_update_alder;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x32_finalize_alder;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_alder;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief u8 angular distance kit - Alder Lake inner + Haswell through. */
struct u8_alder_angular_kit_t : public u8_alder_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief u8 Euclidean (L2) distance kit - Alder Lake inner + Haswell through. */
struct u8_alder_euclidean_kit_t : public u8_alder_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
/** @brief u8 inner-product distance kit - Intel Sierra Forest AVX-VNNI. */
struct u8_sierra_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sierra_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x32_state_sierra_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x32_init_sierra;
    static constexpr auto update_accumulator = nk_dot_u8x32_update_sierra;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x32_finalize_sierra;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_sierra;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief u8 angular distance kit - Sierra Forest inner + Haswell through. */
struct u8_sierra_angular_kit_t : public u8_sierra_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief u8 Euclidean (L2) distance kit - Sierra Forest inner + Haswell through. */
struct u8_sierra_euclidean_kit_t : public u8_sierra_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_SIERRA

#if NK_TARGET_NEONSDOT

/** @brief u8 inner-product distance kit - Arm UDOT. */
struct u8_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_neonsdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief u8 angular distance kit - Arm UDOT. */
struct u8_neonsdot_angular_kit_t : public u8_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief u8 Euclidean (L2) distance kit - Arm UDOT. */
struct u8_neonsdot_euclidean_kit_t : public u8_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_V128RELAXED
/** @brief u8 inner-product distance kit - WebAssembly relaxed SIMD. */
struct u8_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_v128relaxed;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief u8 angular distance kit - WebAssembly relaxed SIMD. */
struct u8_v128relaxed_angular_kit_t : public u8_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_v128relaxed_;
};

/** @brief u8 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct u8_v128relaxed_euclidean_kit_t : public u8_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_POWERVSX
/** @brief u8 inner-product distance kit - IBM POWER VSX. */
struct u8_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::u8_t;
    using accumulator_t = nk_dot_u8x16_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u8x16_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_u8x16_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b8x16_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_u8x16_finalize_powervsx;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u8_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief u8 angular distance kit - IBM POWER VSX. */
struct u8_powervsx_angular_kit_t : public u8_powervsx_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_powervsx_;
};

/** @brief u8 Euclidean (L2) distance kit - IBM POWER VSX. */
struct u8_powervsx_euclidean_kit_t : public u8_powervsx_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_powervsx_;
};
#endif // NK_TARGET_POWERVSX

/** @brief i8 inner-product distance kit - serial reference. */
struct i8_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief i8 angular distance kit - serial reference. Through writes f32. */
struct i8_serial_angular_kit_t : public i8_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief i8 Euclidean (L2) distance kit - serial reference. Through writes f32. */
struct i8_serial_euclidean_kit_t : public i8_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief i8 inner-product distance kit - Intel Haswell AVX2. */
struct i8_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_haswell;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_haswell;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief i8 angular distance kit - Haswell. */
struct i8_haswell_angular_kit_t : public i8_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief i8 Euclidean (L2) distance kit - Haswell. */
struct i8_haswell_euclidean_kit_t : public i8_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
/** @brief i8 inner-product distance kit - Intel Ice Lake AVX-512 + VNNI VPDPBUSD compensation. */
struct i8_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x64_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x64_init_icelake;
    static constexpr auto update_accumulator = nk_dot_i8x64_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b8x64_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_i8x64_finalize_icelake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_icelake;
    static constexpr std::size_t step_dimensions = 64;
};

/** @brief i8 angular distance kit - Icelake inner + Haswell through. */
struct i8_icelake_angular_kit_t : public i8_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief i8 Euclidean (L2) distance kit - Icelake inner + Haswell through. */
struct i8_icelake_euclidean_kit_t : public i8_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_ALDER
/**
 *  @brief i8 inner-product distance kit - Intel Alder Lake AVX-VNNI. NumKong has no
 *         signed-i8 moments kernel for this tier, so the norms come from the AVX2 one,
 *         which Alder Lake also implements.
 */
struct i8_alder_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_alder_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x32_state_alder_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x32_init_alder;
    static constexpr auto update_accumulator = nk_dot_i8x32_update_alder;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x32_finalize_alder;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief i8 angular distance kit - Alder Lake inner + Haswell through. */
struct i8_alder_angular_kit_t : public i8_alder_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief i8 Euclidean (L2) distance kit - Alder Lake inner + Haswell through. */
struct i8_alder_euclidean_kit_t : public i8_alder_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_ALDER

#if NK_TARGET_SIERRA
/** @brief i8 inner-product distance kit - Intel Sierra Forest AVX-VNNI. */
struct i8_sierra_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_sierra_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x32_state_sierra_t;
    using step_vec_t = nk_b256_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x32_init_sierra;
    static constexpr auto update_accumulator = nk_dot_i8x32_update_sierra;
    static constexpr auto load_step = nk_load_b256_haswell_;
    static constexpr auto load_tail = nk_partial_load_b8x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x32_finalize_sierra;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_sierra;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief i8 angular distance kit - Sierra Forest inner + Haswell through. */
struct i8_sierra_angular_kit_t : public i8_sierra_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief i8 Euclidean (L2) distance kit - Sierra Forest inner + Haswell through. */
struct i8_sierra_euclidean_kit_t : public i8_sierra_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_SIERRA

#if NK_TARGET_NEONSDOT

/** @brief i8 inner-product distance kit - Arm SDOT. */
struct i8_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_neonsdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_neonsdot;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief i8 angular distance kit - Arm SDOT. */
struct i8_neonsdot_angular_kit_t : public i8_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief i8 Euclidean (L2) distance kit - Arm SDOT. */
struct i8_neonsdot_euclidean_kit_t : public i8_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_V128RELAXED
/** @brief i8 inner-product distance kit - WebAssembly relaxed SIMD. */
struct i8_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b8x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_v128relaxed;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief i8 angular distance kit - WebAssembly relaxed SIMD. */
struct i8_v128relaxed_angular_kit_t : public i8_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_v128relaxed_;
};

/** @brief i8 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct i8_v128relaxed_euclidean_kit_t : public i8_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_POWERVSX
/** @brief i8 inner-product distance kit - IBM POWER VSX. */
struct i8_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::i8_t;
    using accumulator_t = nk_dot_i8x16_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i8x16_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_i8x16_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b8x16_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_i8x16_finalize_powervsx;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i8_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief i8 angular distance kit - IBM POWER VSX. */
struct i8_powervsx_angular_kit_t : public i8_powervsx_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_powervsx_;
};

/** @brief i8 Euclidean (L2) distance kit - IBM POWER VSX. */
struct i8_powervsx_euclidean_kit_t : public i8_powervsx_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_powervsx_;
};
#endif // NK_TARGET_POWERVSX

/**
 *  @brief u4x2 inner-product distance kit - serial reference. Two unsigned nibbles per
 *         byte, so a step of 16 dimensions rides in a 64-bit vec. The dot lanes land in
 *         @c i32 like every other integral kit, even though @c dot_result_t is unsigned.
 */
struct u4x2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x16_state_serial_t;
    using step_vec_t = nk_b64_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u4x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_u4x16_update_serial;
    static constexpr auto load_step = nk_load_b64_serial_;
    static constexpr auto load_tail = nk_partial_load_b4x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u4_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief u4x2 angular distance kit - serial reference. Through writes f32. */
struct u4x2_serial_angular_kit_t : public u4x2_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief u4x2 Euclidean (L2) distance kit - serial reference. Through writes f32. */
struct u4x2_serial_euclidean_kit_t : public u4x2_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief u4x2 inner-product distance kit - Intel Haswell AVX2. */
struct u4x2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x32_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u4x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_u4x32_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x32_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u4_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief u4x2 angular distance kit - Intel Haswell AVX2. */
struct u4x2_haswell_angular_kit_t : public u4x2_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief u4x2 Euclidean (L2) distance kit - Intel Haswell AVX2. */
struct u4x2_haswell_euclidean_kit_t : public u4x2_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
/** @brief u4x2 inner-product distance kit - Ice Lake VNNI. */
struct u4x2_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x128_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u4x128_init_icelake;
    static constexpr auto update_accumulator = nk_dot_u4x128_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b4x128_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_u4x128_finalize_icelake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u4_skylake;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief u4x2 angular distance kit - Icelake inner + Haswell through. */
struct u4x2_icelake_angular_kit_t : public u4x2_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief u4x2 Euclidean (L2) distance kit - Icelake inner + Haswell through. */
struct u4x2_icelake_euclidean_kit_t : public u4x2_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_NEONSDOT
/** @brief u4x2 inner-product distance kit - Arm UDOT. */
struct u4x2_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x32_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u4x32_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_u4x32_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x32_finalize_neonsdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u4_serial;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief u4x2 angular distance kit - Arm UDOT. */
struct u4x2_neonsdot_angular_kit_t : public u4x2_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief u4x2 Euclidean (L2) distance kit - Arm UDOT. */
struct u4x2_neonsdot_euclidean_kit_t : public u4x2_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_V128RELAXED
/** @brief u4x2 inner-product distance kit - WebAssembly relaxed SIMD. */
struct u4x2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::u4x2_t;
    using accumulator_t = nk_dot_u4x32_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u4x32_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_u4x32_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u4x32_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u4_serial;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief u4x2 angular distance kit - WebAssembly relaxed SIMD. */
struct u4x2_v128relaxed_angular_kit_t : public u4x2_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_v128relaxed_;
};

/** @brief u4x2 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct u4x2_v128relaxed_euclidean_kit_t : public u4x2_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

/** @brief i4x2 inner-product distance kit - serial reference. */
struct i4x2_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x16_state_serial_t;
    using step_vec_t = nk_b64_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i4x16_init_serial;
    static constexpr auto update_accumulator = nk_dot_i4x16_update_serial;
    static constexpr auto load_step = nk_load_b64_serial_;
    static constexpr auto load_tail = nk_partial_load_b4x16_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x16_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i4_serial;
    static constexpr std::size_t step_dimensions = 16;
};

/** @brief i4x2 angular distance kit - serial reference. Through writes f32. */
struct i4x2_serial_angular_kit_t : public i4x2_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/**
 *  @brief i4x2 Euclidean (L2) distance kit - serial reference. The through helper
 *         applies @c sqrt and writes f32, so we override @c distance_lane_t accordingly.
 */
struct i4x2_serial_euclidean_kit_t : public i4x2_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/**
 *  @brief i4x2 inner-product distance kit - Haswell AVX2.
 */
struct i4x2_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x32_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i4x32_init_haswell;
    static constexpr auto update_accumulator = nk_dot_i4x32_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x32_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i4_haswell;
    static constexpr std::size_t step_dimensions = 32;
};

/**
 *  @brief i4x2 Euclidean (L2) distance kit - Haswell inner + Haswell through. The
 *         through helper applies @c sqrt and writes f32, so override @c distance_lane_t.
 */
struct i4x2_haswell_euclidean_kit_t : public i4x2_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};

/** @brief i4 angular distance kit - Haswell AVX2. */
struct i4x2_haswell_angular_kit_t : public i4x2_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE

/** @brief i4 inner-product distance kit - Ice Lake VNNI. */
struct i4x2_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x128_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i4x128_init_icelake;
    static constexpr auto update_accumulator = nk_dot_i4x128_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b4x128_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_i4x128_finalize_icelake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i4_skylake;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief i4 angular distance kit - Ice Lake VNNI. */
struct i4x2_icelake_angular_kit_t : public i4x2_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_haswell_;
};

/** @brief i4 Euclidean (L2) distance kit - Ice Lake VNNI. */
struct i4x2_icelake_euclidean_kit_t : public i4x2_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_haswell_;
};
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_NEONSDOT

/** @brief i4 inner-product distance kit - Arm SDOT. */
struct i4x2_neonsdot_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neonsdot_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x32_state_neonsdot_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i4x32_init_neonsdot;
    static constexpr auto update_accumulator = nk_dot_i4x32_update_neonsdot;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x32_finalize_neonsdot;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i4_serial;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief i4 angular distance kit - Arm SDOT. */
struct i4x2_neonsdot_angular_kit_t : public i4x2_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_serial_;
};

/** @brief i4 Euclidean (L2) distance kit - Arm SDOT. */
struct i4x2_neonsdot_euclidean_kit_t : public i4x2_neonsdot_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_serial_;
};
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_V128RELAXED
/**
 *  @brief i4x2 inner-product distance kit - WebAssembly relaxed SIMD. NumKong has no
 *         4-bit moments kernel for this tier, so the norms come from the serial one.
 */
struct i4x2_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::i4x2_t;
    using accumulator_t = nk_dot_i4x32_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_i4x32_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_i4x32_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b4x32_serial_;
    static constexpr auto reduce_accumulators = nk_dot_i4x32_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_i4_serial;
    static constexpr std::size_t step_dimensions = 32;
};

/** @brief i4x2 angular distance kit - WebAssembly relaxed SIMD. */
struct i4x2_v128relaxed_angular_kit_t : public i4x2_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_angular_through_i32_from_dot_v128relaxed_;
};

/** @brief i4x2 Euclidean (L2) distance kit - WebAssembly relaxed SIMD. */
struct i4x2_v128relaxed_euclidean_kit_t : public i4x2_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_euclidean_through_i32_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

/** @brief u1x8 inner-product distance kit - serial reference (binary AND-popcount). */
struct u1x8_serial_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_serial_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_serial;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_serial;
    static constexpr auto load_step = nk_load_b128_serial_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_serial;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u1_serial;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief u1x8 Hamming distance kit - serial. Through computes @c popA+popB-2*intersection. */
struct u1x8_serial_hamming_kit_t : public u1x8_serial_inner_kit_t {
    using distance_lane_t = nk_u32_t;
    static constexpr auto convert_to_distances = nk_hamming_u32x4_from_dot_serial_;
};

/** @brief u1x8 Jaccard distance kit - serial. Through computes @c 1-intersection/union. */
struct u1x8_serial_jaccard_kit_t : public u1x8_serial_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_jaccard_f32x4_from_dot_serial_;
};

#if NK_TARGET_HASWELL
/** @brief u1x8 inner-product distance kit - Intel Haswell AVX2 + POPCNT. */
struct u1x8_haswell_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_haswell_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_haswell;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_haswell;
    static constexpr auto load_step = nk_load_b128_haswell_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_haswell;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u1_haswell;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief u1x8 Hamming distance kit - Haswell. */
struct u1x8_haswell_hamming_kit_t : public u1x8_haswell_inner_kit_t {
    using distance_lane_t = nk_u32_t;
    static constexpr auto convert_to_distances = nk_hamming_u32x4_from_dot_haswell_;
};

/** @brief u1x8 Jaccard distance kit - Haswell. */
struct u1x8_haswell_jaccard_kit_t : public u1x8_haswell_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_jaccard_f32x4_from_dot_haswell_;
};
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
/** @brief u1x8 inner-product distance kit - Intel Ice Lake AVX-512 + VPOPCNTQ. */
struct u1x8_icelake_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_icelake_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x512_state_icelake_t;
    using step_vec_t = nk_b512_vec_t;
    using prefetch_query_t = prefetch_query_sse_t;
    using prefetch_candidate_t = prefetch_candidate_sse_t;
    using demote_candidate_t = demote_candidate_cldemote_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u1x512_init_icelake;
    static constexpr auto update_accumulator = nk_dot_u1x512_update_icelake;
    static constexpr auto load_step = nk_load_b512_skylake_;
    static constexpr auto load_tail = nk_partial_load_b1x512_skylake_;
    static constexpr auto reduce_accumulators = nk_dot_u1x512_finalize_icelake;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u1_skylake;
    static constexpr std::size_t step_dimensions = 512;
};

/** @brief u1x8 Hamming distance kit - Icelake inner + Icelake through. */
struct u1x8_icelake_hamming_kit_t : public u1x8_icelake_inner_kit_t {
    using distance_lane_t = nk_u32_t;
    static constexpr auto convert_to_distances = nk_hamming_u32x4_from_dot_icelake_;
};

/** @brief u1x8 Jaccard distance kit - Icelake inner + Icelake through. */
struct u1x8_icelake_jaccard_kit_t : public u1x8_icelake_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_jaccard_f32x4_from_dot_icelake_;
};
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_NEON
/** @brief u1x8 inner-product distance kit - Arm NEON (binary AND-popcount). */
struct u1x8_neon_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_neon_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_neon;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_neon;
    static constexpr auto load_step = nk_load_b128_neon_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_neon;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u1_neon;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief u1x8 Hamming distance kit - Arm NEON. */
struct u1x8_neon_hamming_kit_t : public u1x8_neon_inner_kit_t {
    using distance_lane_t = nk_u32_t;
    static constexpr auto convert_to_distances = nk_hamming_u32x4_from_dot_neon_;
};

/** @brief u1x8 Jaccard distance kit - Arm NEON. */
struct u1x8_neon_jaccard_kit_t : public u1x8_neon_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_jaccard_f32x4_from_dot_neon_;
};
#endif // NK_TARGET_NEON

#if NK_TARGET_V128RELAXED
/**
 *  @brief u1x8 inner-product distance kit - WebAssembly relaxed SIMD. NumKong has no
 *         binary moments kernel for this tier, so the norms come from the serial one.
 */
struct u1x8_v128relaxed_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_v128relaxed_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_v128relaxed;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_v128relaxed;
    static constexpr auto load_step = nk_load_b128_v128relaxed_;
    static constexpr auto load_tail = nk_partial_load_b1x128_serial_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_v128relaxed;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u1_serial;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief u1x8 Hamming distance kit - WebAssembly relaxed SIMD. */
struct u1x8_v128relaxed_hamming_kit_t : public u1x8_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_u32_t;
    static constexpr auto convert_to_distances = nk_hamming_u32x4_from_dot_v128relaxed_;
};

/** @brief u1x8 Jaccard distance kit - WebAssembly relaxed SIMD. */
struct u1x8_v128relaxed_jaccard_kit_t : public u1x8_v128relaxed_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_jaccard_f32x4_from_dot_v128relaxed_;
};
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_POWERVSX
/**
 *  @brief u1x8 inner-product distance kit - IBM POWER VSX. NumKong has no binary moments
 *         kernel for this tier, so the norms come from the serial one.
 */
struct u1x8_powervsx_inner_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_powervsx_k;
    using scalar_t = nk::u1x8_t;
    using accumulator_t = nk_dot_u1x128_state_powervsx_t;
    using step_vec_t = nk_b128_vec_t;
    using prefetch_query_t = memory_hint_t;
    using prefetch_candidate_t = memory_hint_t;
    using demote_candidate_t = memory_hint_t;
    using finalized_vec_t = nk_b128_vec_t;
    using distance_lane_t = nk_i32_t;
    static constexpr auto init_accumulator = nk_dot_u1x128_init_powervsx;
    static constexpr auto update_accumulator = nk_dot_u1x128_update_powervsx;
    static constexpr auto load_step = nk_load_b128_powervsx_;
    static constexpr auto load_tail = nk_partial_load_b1x128_powervsx_;
    static constexpr auto reduce_accumulators = nk_dot_u1x128_finalize_powervsx;
    static constexpr auto convert_to_distances = inner_product_distance_<scalar_t::dot_result_t, nk_b128_vec_t>;
    static constexpr auto compute_norms = nk_reduce_moments_u1_serial;
    static constexpr std::size_t step_dimensions = 128;
};

/** @brief u1x8 Hamming distance kit - IBM POWER VSX. */
struct u1x8_powervsx_hamming_kit_t : public u1x8_powervsx_inner_kit_t {
    using distance_lane_t = nk_u32_t;
    static constexpr auto convert_to_distances = nk_hamming_u32x4_from_dot_powervsx_;
};

/** @brief u1x8 Jaccard distance kit - IBM POWER VSX. */
struct u1x8_powervsx_jaccard_kit_t : public u1x8_powervsx_inner_kit_t {
    using distance_lane_t = nk_f32_t;
    static constexpr auto convert_to_distances = nk_jaccard_f32x4_from_dot_powervsx_;
};
#endif // NK_TARGET_POWERVSX

/**
 *  @brief One row in a capability cascade - pairs an availability flag with a candidate kit.
 *
 *  The flag is the @c &&-of @c NK_TARGET_* (always 0/1) and the runtime-caps bit test, so
 *  rows for ISAs that aren't compiled-in fold to @c false at preprocessing time and the
 *  unselected kit's body never has to exist on the current target.
 */
template <bool available_, typename kit_>
struct chain_layer {
    static constexpr bool available = available_;
    using kit_t = kit_;
};

/**
 *  @brief Picks the first @c chain_layer whose @c available flag is true, else @c fallback_.
 *
 *  Lets capability cascades be written linearly with one row per tier and a single closing
 *  @c >::type - no nested @c std::conditional_t pile.
 */
template <typename fallback_, typename... layers_>
struct first_available_kit {
    using type = fallback_;
};

template <typename fallback_, typename head_, typename... tail_>
struct first_available_kit<fallback_, head_, tail_...> {
    using type = std::conditional_t<head_::available, typename head_::kit_t,
                                    typename first_available_kit<fallback_, tail_...>::type>;
};

/**
 *  @brief Capability cascase for every @c (kernel,scalar) combination. Reading order is the
 *  priority order; the first row whose @c available flag is true wins, else the fallback is selected.
 *  Most Eucldiean and Angular distances derive from the same "dot" parent kit.
 */
template <nk_kernel_kind_t kernel_, nk::numeric_dtype scalar_, nk_capability_t caps_>
struct inner_quads_chain;

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                  //
        bf16_serial_inner_kit_t,                                                                                 //
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_inner_kit_t>,                   //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_inner_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_inner_kit_t>,             //
        chain_layer<NK_TARGET_NEONBFDOT && (caps_ & nk_cap_neonbfdot_k) != 0, bf16_neonbfdot_inner_kit_t>,       //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, bf16_neon_inner_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, bf16_v128relaxed_inner_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, bf16_powervsx_inner_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                    //
        bf16_serial_angular_kit_t,                                                                                 //
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_angular_kit_t>,                   //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_angular_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_angular_kit_t>,             //
        chain_layer<NK_TARGET_NEONBFDOT && (caps_ & nk_cap_neonbfdot_k) != 0, bf16_neonbfdot_angular_kit_t>,       //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, bf16_neon_angular_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, bf16_v128relaxed_angular_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, bf16_powervsx_angular_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        bf16_serial_euclidean_kit_t,                                                                                 //
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_euclidean_kit_t>,                   //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_NEONBFDOT && (caps_ & nk_cap_neonbfdot_k) != 0, bf16_neonbfdot_euclidean_kit_t>,       //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, bf16_neon_euclidean_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, bf16_v128relaxed_euclidean_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, bf16_powervsx_euclidean_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                 //
        f16_serial_inner_kit_t,                                                                                 //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, f16_skylake_inner_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, f16_haswell_inner_kit_t>,             //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, f16_neonfhm_inner_kit_t>,             //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, f16_neon_inner_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, f16_v128relaxed_inner_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, f16_powervsx_inner_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                   //
        f16_serial_angular_kit_t,                                                                                 //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, f16_skylake_angular_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, f16_haswell_angular_kit_t>,             //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, f16_neonfhm_angular_kit_t>,             //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, f16_neon_angular_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, f16_v128relaxed_angular_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, f16_powervsx_angular_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        f16_serial_euclidean_kit_t,                                                                                 //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, f16_skylake_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, f16_haswell_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, f16_neonfhm_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, f16_neon_euclidean_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, f16_v128relaxed_euclidean_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, f16_powervsx_euclidean_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit< //
        e5m2_serial_inner_kit_t,                //
        chain_layer<NK_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e5m2_diamond_inner_kit_t>,
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e5m2_genoa_inner_kit_t>,                  //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e5m2_skylake_inner_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e5m2_haswell_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e5m2_neonfp8_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e5m2_neonfhm_inner_kit_t>,            //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e5m2_v128relaxed_inner_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit< //
        e5m2_serial_angular_kit_t,              //
        chain_layer<NK_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e5m2_diamond_angular_kit_t>,
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e5m2_genoa_angular_kit_t>,                  //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e5m2_skylake_angular_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e5m2_haswell_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e5m2_neonfp8_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e5m2_neonfhm_angular_kit_t>,            //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e5m2_v128relaxed_angular_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit< //
        e5m2_serial_euclidean_kit_t,            //
        chain_layer<NK_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e5m2_diamond_euclidean_kit_t>,
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e5m2_genoa_euclidean_kit_t>,                  //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e5m2_skylake_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e5m2_haswell_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e5m2_neonfp8_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e5m2_neonfhm_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e5m2_v128relaxed_euclidean_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit< //
        e4m3_serial_inner_kit_t,                //
        chain_layer<NK_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e4m3_diamond_inner_kit_t>,
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e4m3_genoa_inner_kit_t>,                  //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e4m3_skylake_inner_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e4m3_haswell_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e4m3_neonfp8_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e4m3_neonfhm_inner_kit_t>,            //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e4m3_v128relaxed_inner_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit< //
        e4m3_serial_angular_kit_t,              //
        chain_layer<NK_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e4m3_diamond_angular_kit_t>,
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e4m3_genoa_angular_kit_t>,                  //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e4m3_skylake_angular_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e4m3_haswell_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e4m3_neonfp8_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e4m3_neonfhm_angular_kit_t>,            //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e4m3_v128relaxed_angular_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit< //
        e4m3_serial_euclidean_kit_t,            //
        chain_layer<NK_TARGET_DIAMOND && (caps_ & nk_cap_diamond_k) != 0, e4m3_diamond_euclidean_kit_t>,
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, e4m3_genoa_euclidean_kit_t>,                  //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e4m3_skylake_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e4m3_haswell_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e4m3_neonfp8_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONFHM && (caps_ & nk_cap_neonfhm_k) != 0, e4m3_neonfhm_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e4m3_v128relaxed_euclidean_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::e3m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                 //
        e3m2_serial_inner_kit_t,                                                                                //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e3m2_skylake_inner_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e3m2_haswell_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e3m2_neonfp8_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e3m2_neonsdot_inner_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e3m2_v128relaxed_inner_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::e3m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                   //
        e3m2_serial_angular_kit_t,                                                                                //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e3m2_skylake_angular_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e3m2_haswell_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e3m2_neonfp8_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e3m2_neonsdot_angular_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e3m2_v128relaxed_angular_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::e3m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        e3m2_serial_euclidean_kit_t,                                                                                //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e3m2_skylake_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e3m2_haswell_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e3m2_neonfp8_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e3m2_neonsdot_euclidean_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e3m2_v128relaxed_euclidean_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::e2m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                                 //
        e2m3_serial_inner_kit_t,                                                                                //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e2m3_skylake_inner_kit_t>,            //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, e2m3_sierra_inner_kit_t>,               //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, e2m3_alder_inner_kit_t>,                  //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e2m3_haswell_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e2m3_neonfp8_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e2m3_neonsdot_inner_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e2m3_v128relaxed_inner_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::e2m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                                   //
        e2m3_serial_angular_kit_t,                                                                                //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e2m3_skylake_angular_kit_t>,            //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, e2m3_sierra_angular_kit_t>,               //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, e2m3_alder_angular_kit_t>,                  //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e2m3_haswell_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e2m3_neonfp8_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e2m3_neonsdot_angular_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e2m3_v128relaxed_angular_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::e2m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        e2m3_serial_euclidean_kit_t,                                                                                //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, e2m3_skylake_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, e2m3_sierra_euclidean_kit_t>,               //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, e2m3_alder_euclidean_kit_t>,                  //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, e2m3_haswell_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONFP8 && (caps_ & nk_cap_neonfp8_k) != 0, e2m3_neonfp8_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, e2m3_neonsdot_euclidean_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, e2m3_v128relaxed_euclidean_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                //
        i8_serial_inner_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_inner_kit_t>,             //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, i8_sierra_inner_kit_t>,                //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, i8_alder_inner_kit_t>,                   //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_inner_kit_t>,             //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i8_neonsdot_inner_kit_t>,          //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i8_v128relaxed_inner_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, i8_powervsx_inner_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                  //
        i8_serial_angular_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_angular_kit_t>,             //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, i8_sierra_angular_kit_t>,                //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, i8_alder_angular_kit_t>,                   //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_angular_kit_t>,             //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i8_neonsdot_angular_kit_t>,          //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i8_v128relaxed_angular_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, i8_powervsx_angular_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                    //
        i8_serial_euclidean_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, i8_sierra_euclidean_kit_t>,                //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, i8_alder_euclidean_kit_t>,                   //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i8_neonsdot_euclidean_kit_t>,          //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i8_v128relaxed_euclidean_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, i8_powervsx_euclidean_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::u8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                //
        u8_serial_inner_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u8_icelake_inner_kit_t>,             //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, u8_sierra_inner_kit_t>,                //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, u8_alder_inner_kit_t>,                   //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u8_haswell_inner_kit_t>,             //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u8_neonsdot_inner_kit_t>,          //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u8_v128relaxed_inner_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u8_powervsx_inner_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::u8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                  //
        u8_serial_angular_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u8_icelake_angular_kit_t>,             //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, u8_sierra_angular_kit_t>,                //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, u8_alder_angular_kit_t>,                   //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u8_haswell_angular_kit_t>,             //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u8_neonsdot_angular_kit_t>,          //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u8_v128relaxed_angular_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u8_powervsx_angular_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::u8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                    //
        u8_serial_euclidean_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u8_icelake_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_SIERRA && (caps_ & nk_cap_sierra_k) != 0, u8_sierra_euclidean_kit_t>,                //
        chain_layer<NK_TARGET_ALDER && (caps_ & nk_cap_alder_k) != 0, u8_alder_euclidean_kit_t>,                   //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u8_haswell_euclidean_kit_t>,             //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u8_neonsdot_euclidean_kit_t>,          //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u8_v128relaxed_euclidean_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u8_powervsx_euclidean_kit_t>           //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::u4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                 //
        u4x2_serial_inner_kit_t,                                                                                //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u4x2_icelake_inner_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u4x2_haswell_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u4x2_neonsdot_inner_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u4x2_v128relaxed_inner_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::u4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                   //
        u4x2_serial_angular_kit_t,                                                                                //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u4x2_icelake_angular_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u4x2_haswell_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u4x2_neonsdot_angular_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u4x2_v128relaxed_angular_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::u4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        u4x2_serial_euclidean_kit_t,                                                                                //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u4x2_icelake_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u4x2_haswell_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, u4x2_neonsdot_euclidean_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u4x2_v128relaxed_euclidean_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                 //
        i4x2_serial_inner_kit_t,                                                                                //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i4x2_icelake_inner_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i4x2_haswell_inner_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i4x2_neonsdot_inner_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i4x2_v128relaxed_inner_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_angular_k, nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                   //
        i4x2_serial_angular_kit_t,                                                                                //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i4x2_icelake_angular_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i4x2_haswell_angular_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i4x2_neonsdot_angular_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i4x2_v128relaxed_angular_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_euclidean_k, nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                     //
        i4x2_serial_euclidean_kit_t,                                                                                //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i4x2_icelake_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i4x2_haswell_euclidean_kit_t>,            //
        chain_layer<NK_TARGET_NEONSDOT && (caps_ & nk_cap_neonsdot_k) != 0, i4x2_neonsdot_euclidean_kit_t>,         //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, i4x2_v128relaxed_euclidean_kit_t> //
        >::type;
    /** @brief The capability bit of the layer that won - compare against the requested ceiling. */
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_dot_k, nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                  //
        u1x8_serial_inner_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_inner_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_inner_kit_t>,             //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, u1x8_neon_inner_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u1x8_v128relaxed_inner_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u1x8_powervsx_inner_kit_t>           //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_hamming_k, nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                    //
        u1x8_serial_hamming_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_hamming_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_hamming_kit_t>,             //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, u1x8_neon_hamming_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u1x8_v128relaxed_hamming_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u1x8_powervsx_hamming_kit_t>           //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct inner_quads_chain<nk_kernel_jaccard_k, nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                                    //
        u1x8_serial_jaccard_kit_t,                                                                                 //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_jaccard_kit_t>,             //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_jaccard_kit_t>,             //
        chain_layer<NK_TARGET_NEON && (caps_ & nk_cap_neon_k) != 0, u1x8_neon_jaccard_kit_t>,                      //
        chain_layer<NK_TARGET_V128RELAXED && (caps_ & nk_cap_v128relaxed_k) != 0, u1x8_v128relaxed_jaccard_kit_t>, //
        chain_layer<NK_TARGET_POWERVSX && (caps_ & nk_cap_powervsx_k) != 0, u1x8_powervsx_jaccard_kit_t>           //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

/**
 *  @brief The only place the load, update, finalize, through, prefetch, demote choreography lives.
 *
 *  Lane fan-out is unrolled via @c std::index_sequence so the same body emits clean code
 *  for both @b 1-to-1 and @b 1-to-4. The state array is always sized 4 (NumKong's
 *  finalize is hardcoded fan-out-4); for 1-to-1 the bottom three slots are left as
 *  register noise and the resulting lanes 1-3 are dropped at extract time.
 *
 *  The compensation-vs-no-compensation split in NumKong's finalize ABI is detected here
 *  via @c std::is_invocable_v on the kit's raw function pointer, so each kit can keep
 *  exposing @c reduce_accumulators as a direct @c nk_* alias rather than a wrapper.
 */
template <std::size_t fan_out_, inner_quads_kit kit_>
[[gnu::always_inline]] inline std::array<f32_t, fan_out_> invoke_distances_(   //
    nk::vector_view<typename kit_::scalar_t> query,                            //
    vector_norms<typename kit_::scalar_t::dot_result_t> query_norms,           //
    std::array<nk::vector_view<typename kit_::scalar_t>, fan_out_> candidates, //
    std::array<vector_norms<typename kit_::scalar_t::dot_result_t>, fan_out_> candidates_norms) noexcept {

    static_assert(fan_out_ == 1 || fan_out_ == 4, "Only 1-to-1 and 1-to-4 fan-outs are supported");
    using scalar_t = typename kit_::scalar_t;
    using accumulator_t = typename kit_::accumulator_t;
    using step_vec_t = typename kit_::step_vec_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using distance_lane_t = typename kit_::distance_lane_t;
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_t>();

    [[maybe_unused]] typename kit_::prefetch_query_t const prefetch_query {};
    [[maybe_unused]] typename kit_::prefetch_candidate_t const prefetch_candidate {};
    [[maybe_unused]] typename kit_::demote_candidate_t const demote_candidate {};

    // Always allocate & initialize all 4 state slots. The 4-way shape of NumKong's
    // `reduce_accumulators` is the deliberate ABI - one batch primitive per kernel, no
    // scalar variants - and its caller contract is that every slot it reads is initialized,
    // even when the 1-to-1 instantiation only extracts lane 0. Three zero-stores buy a
    // well-defined finalize on every path.
    NK_ALIGN64 accumulator_t accumulators[4];
    kit_::init_accumulator(&accumulators[0]);
    kit_::init_accumulator(&accumulators[1]);
    kit_::init_accumulator(&accumulators[2]);
    kit_::init_accumulator(&accumulators[3]);

    auto const* query_base = query.data();
    auto candidate_bases = [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
        return std::array<decltype(query.data()), fan_out_> {candidates[lane_index].data()...};
    }(std::make_index_sequence<fan_out_> {});

    if constexpr (USEARCH_USE_PREFETCH) {
        prefetch_query(query_base);
        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (prefetch_candidate(candidate_bases[lane_index]), ...);
        }(std::make_index_sequence<fan_out_> {});
    }

    std::size_t const total_dimensions = query.size();
    std::size_t const tail_dimensions = total_dimensions % kit_::step_dimensions;
    std::size_t passed_dimensions = 0;

    NK_ALIGN64 step_vec_t query_step_vec;
    NK_ALIGN64 step_vec_t candidate_step_vecs[fan_out_];
    for (; passed_dimensions + kit_::step_dimensions <= total_dimensions; passed_dimensions += kit_::step_dimensions) {
        std::size_t const value_offset = passed_dimensions / dims_per_value;
        std::size_t const next_value_offset = (passed_dimensions + kit_::step_dimensions) / dims_per_value;
        kit_::load_step(query_base + value_offset, &query_step_vec);
        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (kit_::load_step(candidate_bases[lane_index] + value_offset, &candidate_step_vecs[lane_index]), ...);
        }(std::make_index_sequence<fan_out_> {});

        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            (kit_::update_accumulator(&accumulators[lane_index], query_step_vec, candidate_step_vecs[lane_index],
                                      passed_dimensions, kit_::step_dimensions),
             ...);
        }(std::make_index_sequence<fan_out_> {});

        if constexpr (USEARCH_USE_PREFETCH) {
            prefetch_query(query_base + next_value_offset);
            [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
                (prefetch_candidate(candidate_bases[lane_index] + next_value_offset), ...);
            }(std::make_index_sequence<fan_out_> {});
        }
        if constexpr (USEARCH_USE_DEMOTE) {
            [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
                (demote_candidate(candidate_bases[lane_index] + value_offset), ...);
            }(std::make_index_sequence<fan_out_> {});
        }
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
        if constexpr (USEARCH_USE_DEMOTE) {
            [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
                (demote_candidate(candidate_bases[lane_index] + value_offset), ...);
            }(std::make_index_sequence<fan_out_> {});
        }
    }

    // u8/i8 finalize takes 8 args (DPBUSD bias correction); float-path takes 6. Detected
    // via `std::is_invocable_v` so kits keep `reduce_accumulators` as a raw alias.
    using finalized_vec_t = typename kit_::finalized_vec_t;
    NK_ALIGN64 finalized_vec_t dot_products_vec;
    constexpr bool needs_compensation = std::is_invocable_v<                                    //
        decltype(kit_::reduce_accumulators),                                                    //
        accumulator_t const*, accumulator_t const*, accumulator_t const*, accumulator_t const*, //
        std::size_t, dot_result_t, finalized_vec_t const*, finalized_vec_t*>;
    if constexpr (needs_compensation) {
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

    // Now infer the angular/euclidean distance from the inner product, in vec form.
    // The through helper (`nk_*_through_*_from_dot_*` or our `inner_product_distance_`)
    // takes the dots vec, query sumsq, candidate sumsqs vec, and writes a distances vec.
    NK_ALIGN64 finalized_vec_t distances_from_dot_products_vec;
    {
        alignas(16) std::array<dot_result_t, 4> candidates_sumsqs_array {};
        [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
            ((candidates_sumsqs_array[lane_index] = candidates_norms[lane_index].sumsq), ...);
        }(std::make_index_sequence<fan_out_> {});
        kit_::convert_to_distances(&dot_products_vec, query_norms.sumsq,
                                   reinterpret_cast<finalized_vec_t const*>(candidates_sumsqs_array.data()),
                                   &distances_from_dot_products_vec);
    }

    // Extract lanes typed as `distance_lane_t` - the width the through actually wrote -
    // which differs from `dot_result_t` when the metric narrows or widens (e.g. u8
    // angular: u32 dot in, f32 distance out).
    auto const& distances_array = *reinterpret_cast<std::array<distance_lane_t, 4> const*>(
        &distances_from_dot_products_vec);
    std::array<f32_t, fan_out_> result_array;
    [&]<std::size_t... lane_index>(std::index_sequence<lane_index...>) {
        ((result_array[lane_index] = static_cast<f32_t>(distances_array[lane_index])), ...);
    }(std::make_index_sequence<fan_out_> {});
    return result_array;
}

/**
 *  @brief Compute @c (sum,sumsq) of @p vec via the kit's @c compute_norms.
 *
 *  The capability cascade is the existing @c inner_quads_chain - the dot kernel's kit always
 *  exists when any kernel is wired up for the scalar, so we resolve through @c
 *  inner_quads_chain<nk_kernel_dot_k, scalar, caps>. Replaces the old @c struct @c inner_norms
 *  partial-specialization cascade with a single source of truth (the kit).
 */
template <nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline vector_norms<typename scalar_::dot_result_t> inner_norms(
    nk::vector_view<scalar_> vec) noexcept {
    using kit_t = typename inner_quads_chain<nk_kernel_dot_k, scalar_, caps_>::kit_t;
    return invoke_inner_norms_(kit_t::compute_norms, vec);
}

/**
 *  @brief 1-to-1 distance from @c query to @c candidate under @c kernel_.
 *
 *  Free function template - scalar deduced from the @c nk::vector_view; metric and
 *  capability passed explicitly. The chosen kit is internal to the trampoline.
 */
template <nk_kernel_kind_t kernel_, nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline f32_t distance(                 //
    nk::vector_view<scalar_> query,                           //
    vector_norms<typename scalar_::dot_result_t> query_norms, //
    nk::vector_view<scalar_> candidate,                       //
    vector_norms<typename scalar_::dot_result_t> candidate_norms) noexcept {
    using kit_t = typename inner_quads_chain<kernel_, scalar_, caps_>::kit_t;
    auto results = invoke_distances_<1, kit_t>(
        query, query_norms, std::array<nk::vector_view<scalar_>, 1> {candidate},
        std::array<vector_norms<typename scalar_::dot_result_t>, 1> {candidate_norms});
    return results[0];
}

/**
 *  @brief 1-to-4 distances from @c query to four @c candidates under @c kernel_.
 */
template <nk_kernel_kind_t kernel_, nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline std::array<f32_t, 4> distances( //
    nk::vector_view<scalar_> query,                           //
    vector_norms<typename scalar_::dot_result_t> query_norms, //
    std::array<nk::vector_view<scalar_>, 4> candidates,       //
    std::array<vector_norms<typename scalar_::dot_result_t>, 4> candidates_norms) noexcept {
    using kit_t = typename inner_quads_chain<kernel_, scalar_, caps_>::kit_t;
    return invoke_distances_<4, kit_t>(query, query_norms, candidates, candidates_norms);
}

/** @brief Empty tile kit: the (kernel, scalar) pair has no matrix-engine backend on this
 *      target, so eager refinement stays unavailable and callers keep the lazy 4-way loops. */
struct no_tiles_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_serial_k;
};

template <typename finalize_kit_type_>
struct u8_sapphireamx_tiles_kit;

#if NK_TARGET_SAPPHIREAMX

/**
 *  @brief u8 tile kit for Sapphire Rapids AMX - the 4-way trio one power-of-two wider:
 *      @c init_tile / @c update_tile / @c finalize_tile over 16×16 accumulator tiles, plus
 *      the gather & pack that feed them. Finalizes through @p finalize_kit_type_ - the very
 *      kit the lazy path picked - so eager & lazy distances agree bit-for-bit. An SME twin
 *      fills the same slots with @c svmopa over ZA tiles, a CUDA twin with @c mma fragments.
 */
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

    /** @brief Bytes of tile scratch one Gram over @p count rows of @p dimensions needs -
     *      the packed B-tiles plus one row-tile of A-strips - derived from this kit's own
     *      tile types, so every dtype & engine sizes itself. Includes one alignment margin:
     *      generic byte allocators promise 16, the tile loads demand 64, and the consumer
     *      aligns the base up before carving. */
    static constexpr std::size_t scratch_bytes(std::size_t count, std::size_t dimensions) noexcept {
        std::size_t const row_tiles = divide_round_up<std::size_t>(count, tile_rows_k);
        std::size_t const depth_tiles = divide_round_up<std::size_t>(dimensions, tile_depth_k);
        return row_tiles * depth_tiles * sizeof(b_tile_t) + depth_tiles * sizeof(a_tile_t) + alignof(a_tile_t);
    }

    /** @brief One palette programming per eager refinement - never per tile. */
    static void configure_tiles() noexcept { nk_amx_tile_configure_sapphireamx_(); }

    /** @brief Gathers one @c tile_depth_k -deep strip of all 16 @p rows - scattered views,
     *      each touched once per depth sweep: the tile @b is the staging, no contiguous
     *      matrix ever exists. Row pads repeat valid views, like the 4-way tails; the
     *      depth tail zero-fills, adding nothing to any dot. */
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

    /** @brief The transpose network NumKong packs with: a gathered A-tile becomes its own
     *      quad-interleaved B-tile - the Gram symmetry in one call. */
    static void pack_b_tile(a_tile_t const& a_tile, b_tile_t& b_tile) noexcept {
        nk_dots_pack_u8_transposed_sapphireamx_(&a_tile, &b_tile);
    }

    /** @brief Zeroes the accumulator ahead of a depth walk. */
    static void init_tile(accumulator_tile_t& tile) noexcept { nk_dots_u8_init_sapphireamx_(&tile); }

    /** @brief tile += A × B for one depth step - @c TDPBUUD here. The accumulator
     *      round-trips through memory per step, a couple of tile moves; keeping it
     *      register-resident across the walk is a known follow-up. */
    static void update_tile(accumulator_tile_t& tile, a_tile_t const& a_tile, b_tile_t const& b_tile) noexcept {
        nk_compiler_barrier_sapphireamx_();
        _tile_loadd(0, tile.data, 64);
        _tile_loadd(1, a_tile.data, 64);
        _tile_loadd(2, b_tile.data, 64);
        _tile_dpbuud(0, 1, 2);
        _tile_stored(0, tile.data, 64);
        nk_compiler_barrier_sapphireamx_();
    }

    /** @brief Distances through the stored norms for all 16×16 lanes: 4-lane groups through
     *      the finalize kit's own @c convert_to_distances - the epilogue of
     *      @c invoke_distances_, tiled. Pad lanes finalize too and are simply never read. */
    static void finalize_tile(accumulator_tile_t const& tile, std::array<norms_t, tile_rows_k> const& a_norms,
                              std::array<norms_t, tile_columns_k> const& b_norms, f32_t* distances,
                              std::size_t stride) noexcept {
        using finalized_vec_t = typename finalize_kit_t::finalized_vec_t;
        using dot_result_t = typename scalar_t::dot_result_t;
        alignas(64) dot_result_t column_sumsqs[tile_columns_k];
        for (std::size_t column = 0; column != tile_columns_k; ++column) column_sumsqs[column] = b_norms[column].sumsq;
        for (std::size_t row = 0; row != tile_rows_k; ++row) {
            f32_t* const distances_row = distances + row * stride;
            for (std::size_t group = 0; group != tile_columns_k / 4; ++group) {
                finalized_vec_t dots_vec, sumsqs_vec, distances_vec;
                dots_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&tile.data[row][group * 4]));
                sumsqs_vec.xmm = _mm_load_si128(reinterpret_cast<__m128i const*>(&column_sumsqs[group * 4]));
                finalize_kit_t::convert_to_distances(&dots_vec, a_norms[row].sumsq, &sumsqs_vec, &distances_vec);
                _mm_storeu_ps(reinterpret_cast<float*>(distances_row + group * 4), distances_vec.xmm_ps);
            }
        }
    }
};

#endif // NK_TARGET_SAPPHIREAMX

/**
 *  @brief Tile-kernel capability cascade - a second, independent axis over the same
 *      @p caps_ bitmask as @c inner_quads_chain, so one machine picks icelake for 1-to-1
 *      kernels and sapphireamx for tiles simultaneously. The primary template resolves
 *      to @c no_tiles_kit_t, and only the specialized (kernel, scalar) pairs ever offer
 *      a matrix-engine backend - callers keep their lazy loops everywhere else.
 */
template <nk_kernel_kind_t kernel_, nk::numeric_dtype scalar_, nk_capability_t caps_>
struct inner_tiles_chain {
    using kit_t = no_tiles_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = false;
};

template <nk_capability_t caps_>
struct inner_tiles_chain<nk_kernel_euclidean_k, nk::u8_t, caps_> {
    using kit_t = typename first_available_kit< //
        no_tiles_kit_t,                         //
        chain_layer<
            NK_TARGET_SAPPHIREAMX && (caps_ & nk_cap_sapphireamx_k) != 0,
            u8_sapphireamx_tiles_kit<typename inner_quads_chain<nk_kernel_euclidean_k, nk::u8_t, caps_>::kit_t>>> //
        ::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = !std::is_same_v<kit_t, no_tiles_kit_t>;
};

template <nk_capability_t caps_>
struct inner_tiles_chain<nk_kernel_angular_k, nk::u8_t, caps_> {
    using kit_t = typename first_available_kit< //
        no_tiles_kit_t,                         //
        chain_layer<
            NK_TARGET_SAPPHIREAMX && (caps_ & nk_cap_sapphireamx_k) != 0,
            u8_sapphireamx_tiles_kit<typename inner_quads_chain<nk_kernel_angular_k, nk::u8_t, caps_>::kit_t>>> //
        ::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
    static constexpr bool has_tiles_k = !std::is_same_v<kit_t, no_tiles_kit_t>;
};

} // namespace usearch
} // namespace unum
