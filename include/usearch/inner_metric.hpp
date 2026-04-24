/**
 *  @brief Compile-time-dispatched 1-to-1 and 1-to-4 distance kernels backed by NumKong.
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
 *  through narrows or widens). The shape is uniform - the @c inner_kit concept enforces it.
 *
 *  @b Chains - one @c kernel_chain<kernel,scalar,caps> per supported tuple. A linear
 *  list of @c chain_layer rows fed to @c first_available_kit picks the first kit whose
 *  @c NK_TARGET_* is on AND whose runtime-caps bit is set, else falls back to serial.
 *  Unsupported @c (kernel,scalar) pairs simply have no specialization. Adding an ISA
 *  tier is one row and zero preprocessor lines.
 *
 *  @b Trampoline - @c invoke_distances_<fan_out,kit> is the one place the load, update,
 *  finalize, through, prefetch, demote choreography lives. Reads everything it needs
 *  (@c scalar_t, @c distance_lane_t, @c convert_to_distances, ...) off the kit. Lane fan-out
 *  is unrolled via @c std::index_sequence so the same body emits clean code for both
 *  @b 1-to-1 and @b 1-to-4. The state array is always sized 4 (NumKong's finalize is
 *  hardcoded fan-out-4); for 1-to-1 the bottom three slots are left as register noise
 *  and the resulting lanes 1-3 are dropped at extract time.
 *
 *  @b Entry-points - free function templates @c distance<kernel,caps> and
 *  @c distances<kernel,caps>, with @c scalar deduced from the @c nk::vector_view inputs.
 *  Call sites read @c distance<nk_kernel_angular_k,caps> - no per-spec method, no per-spec class.
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
concept inner_kit = requires {
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
 *  so @c kernel_chain backend resolution chains name-resolve on every build.
 */
struct bf16_serial_inner_kit_t;
struct bf16_haswell_inner_kit_t;
struct bf16_skylake_inner_kit_t;
struct bf16_genoa_inner_kit_t;
struct bf16_serial_angular_kit_t;
struct bf16_haswell_angular_kit_t;
struct bf16_skylake_angular_kit_t;
struct bf16_genoa_angular_kit_t;
struct bf16_serial_euclidean_kit_t;
struct bf16_haswell_euclidean_kit_t;
struct bf16_skylake_euclidean_kit_t;
struct bf16_genoa_euclidean_kit_t;
struct i8_serial_inner_kit_t;
struct i8_haswell_inner_kit_t;
struct i8_icelake_inner_kit_t;
struct i8_serial_angular_kit_t;
struct i8_haswell_angular_kit_t;
struct i8_icelake_angular_kit_t;
struct i8_serial_euclidean_kit_t;
struct i8_haswell_euclidean_kit_t;
struct i8_icelake_euclidean_kit_t;
struct u8_serial_inner_kit_t;
struct u8_icelake_inner_kit_t;
struct i4x2_serial_inner_kit_t;
struct i4x2_haswell_inner_kit_t;
struct i4x2_serial_euclidean_kit_t;
struct i4x2_haswell_euclidean_kit_t;
struct u1x8_serial_inner_kit_t;
struct u1x8_haswell_inner_kit_t;
struct u1x8_icelake_inner_kit_t;
struct u1x8_serial_hamming_kit_t;
struct u1x8_haswell_hamming_kit_t;
struct u1x8_icelake_hamming_kit_t;
struct u1x8_serial_jaccard_kit_t;
struct u1x8_haswell_jaccard_kit_t;
struct u1x8_icelake_jaccard_kit_t;

/** @brief bf16 inner-product distance kit - serial reference. */
struct bf16_serial_inner_kit_t {
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

/** @brief u8 inner-product distance kit - serial reference. */
struct u8_serial_inner_kit_t {
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

#if NK_TARGET_ICELAKE
/** @brief u8 inner-product distance kit - Intel Ice Lake AVX-512 + VNNI VPDPBUSD. */
struct u8_icelake_inner_kit_t {
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
#endif // NK_TARGET_ICELAKE

/** @brief i8 inner-product distance kit - serial reference. */
struct i8_serial_inner_kit_t {
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

/** @brief i4x2 inner-product distance kit - serial reference. */
struct i4x2_serial_inner_kit_t {
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
#endif // NK_TARGET_HASWELL

/** @brief u1x8 inner-product distance kit - serial reference (binary AND-popcount). */
struct u1x8_serial_inner_kit_t {
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
struct kernel_chain;

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_dot_k, nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                      //
        bf16_serial_inner_kit_t,                                                                     //
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_inner_kit_t>,       //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_inner_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_inner_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_angular_k, nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                        //
        bf16_serial_angular_kit_t,                                                                     //
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_angular_kit_t>,       //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_angular_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_angular_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_euclidean_k, nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                          //
        bf16_serial_euclidean_kit_t,                                                                     //
        chain_layer<NK_TARGET_GENOA && (caps_ & nk_cap_genoa_k) != 0, bf16_genoa_euclidean_kit_t>,       //
        chain_layer<NK_TARGET_SKYLAKE && (caps_ & nk_cap_skylake_k) != 0, bf16_skylake_euclidean_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, bf16_haswell_euclidean_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_dot_k, nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                    //
        i8_serial_inner_kit_t,                                                                     //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_inner_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_inner_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_angular_k, nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                      //
        i8_serial_angular_kit_t,                                                                     //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_angular_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_angular_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_euclidean_k, nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                        //
        i8_serial_euclidean_kit_t,                                                                     //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, i8_icelake_euclidean_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i8_haswell_euclidean_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_dot_k, nk::u8_t, caps_> {
    using kit_t = typename first_available_kit<                                                   //
        u8_serial_inner_kit_t,                                                                    //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u8_icelake_inner_kit_t> //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_dot_k, nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                     //
        i4x2_serial_inner_kit_t,                                                                    //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i4x2_haswell_inner_kit_t> //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_euclidean_k, nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                         //
        i4x2_serial_euclidean_kit_t,                                                                    //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, i4x2_haswell_euclidean_kit_t> //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_dot_k, nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                      //
        u1x8_serial_inner_kit_t,                                                                     //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_inner_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_inner_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_hamming_k, nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                        //
        u1x8_serial_hamming_kit_t,                                                                     //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_hamming_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_hamming_kit_t>  //
        >::type;
};

template <nk_capability_t caps_>
struct kernel_chain<nk_kernel_jaccard_k, nk::u1x8_t, caps_> {
    using kit_t = typename first_available_kit<                                                        //
        u1x8_serial_jaccard_kit_t,                                                                     //
        chain_layer<NK_TARGET_ICELAKE && (caps_ & nk_cap_icelake_k) != 0, u1x8_icelake_jaccard_kit_t>, //
        chain_layer<NK_TARGET_HASWELL && (caps_ & nk_cap_haswell_k) != 0, u1x8_haswell_jaccard_kit_t>  //
        >::type;
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
template <std::size_t fan_out_, inner_kit kit_>
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

    [[maybe_unused]] typename kit_::prefetch_query_t prefetch_query;
    [[maybe_unused]] typename kit_::prefetch_candidate_t prefetch_candidate;
    [[maybe_unused]] typename kit_::demote_candidate_t demote_candidate;

    // Always allocate 4 state slots - NumKong's `reduce_accumulators` is hardcoded
    // fan-out-4. For 1-to-1 the bottom three slots are left as register noise; finalize
    // still reads them, but the resulting lanes 1-3 are dropped at extract time.
    NK_ALIGN64 accumulator_t accumulators[4];
    kit_::init_accumulator(&accumulators[0]);
    if constexpr (fan_out_ == 4) {
        kit_::init_accumulator(&accumulators[1]);
        kit_::init_accumulator(&accumulators[2]);
        kit_::init_accumulator(&accumulators[3]);
    }

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
 *  @brief Compute @c (sum, sumsq) of @p vec via the kit's @c compute_norms.
 *
 *  The capability cascade is the existing @c kernel_chain - the dot kernel's kit always
 *  exists when any kernel is wired up for the scalar, so we resolve through @c
 *  kernel_chain<nk_kernel_dot_k, scalar, caps>. Replaces the old @c struct @c inner_norms
 *  partial-specialization cascade with a single source of truth (the kit).
 */
template <nk_capability_t caps_, nk::numeric_dtype scalar_>
[[gnu::always_inline]] inline vector_norms<typename scalar_::dot_result_t> inner_norms(
    nk::vector_view<scalar_> vec) noexcept {
    using kit_t = typename kernel_chain<nk_kernel_dot_k, scalar_, caps_>::kit_t;
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
    using kit_t = typename kernel_chain<kernel_, scalar_, caps_>::kit_t;
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
    using kit_t = typename kernel_chain<kernel_, scalar_, caps_>::kit_t;
    return invoke_distances_<4, kit_t>(query, query_norms, candidates, candidates_norms);
}

} // namespace usearch
} // namespace unum
