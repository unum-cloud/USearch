/**
 *  @brief Compile-time dispatch mechanisms for Inner-Product-based metrics from NumKong.
 *  @author Ash Vardanian
 *  @file inner_metric.hpp
 *  @date April 21, 2026
 */
#pragma once

#include <numkong/numkong.h>
#include <numkong/types.hpp>
#include <numkong/vector.hpp>

#include "memory_hints.hpp"

namespace unum {
namespace usearch {

namespace nk = ashvardanian::numkong;

using f32_t = float;

/**
 *  @brief Specialized helper hardware-freidnly vector math & distance functions in @b `inner_index`.
 *
 *  @section Design Goals & Decisions
 *
 *  1. Avoid function call overhead. All of the NumKong logic is inlined from ISA-specific @b `NK_PUBLIC`
 *     APIs, skpping the stack-frame construction part of `NK_DYNAMIC` interfaces from the shared lib.
 *
 *  2. Amortize arithmetics reductions. Both Euclidean (L2) and Angular (Cos) distances can be trivially
 *     inferred from the inner product of two vectors if the norms are known. This means computing @b 2-by-D
 *     arithmetic operations per distance computation instead of @b 6-by-D on the hot path by precomputing
 *     and storing one extra norm value per vector.
 *
 *  3. Amortize memory loads & mini-float upcasting. When computing an angular distance between a query vector and
 *     M neighbors of a certain graph neighbor, we might be upcasting mini-floats to Float32 @b 2-by-M times.
 *     Storing Float32 in the index kills the value proposition of memory reductions. Pre-upcasting M
 *     vectors simultanously - washes the register contents. Instead of @b 1-to-1 distances like in older
 *     USearch releases, we now compute @b 1-to-4 distances, upcasting the query half of the binary operation
 *     four times less frequently, thus - shrinking the total number of loads & upcasts to @b M+M/4.
 *     This also paves the way for using cache-friendly streaming loads, making sure that L1/L2 is reserved for
 *     graph-traversal data-structures, while the vectors only go as high as the L3 or other cross-core LLC.
 *
 *  4. Amortize horizontal reductions. On low-dimenional vectors (64D Matryoshka or 128D Group Query Attention),
 *     the most expensive part of inner-products on platforms like x86 - is the final horizontal reduction of
 *     accumulators. Thanks to the @b 1-to-4 distance computation approach, we can further dampen the cost of
 *     each individual reduction by using cross-register shuffles.
 *
 *  @section Assumptions & Limitations
 *
 *  This is designed for the most common fast-path usecases of USearch templates, where a distance can be represented
 *  via @b `f32_t` with an acceptable level of precision loss. It @b doesn't apply to:
 *
 *  - indexing arbitrary non-equidimensional-vector entries, like variable-length strings,
 *  - Haversine, Vincenty, or other geospatial approaches that aren't built around Inner Products,
 *  - multi-vector entry descriptors produced by late-interaction ColBERT-like models,
 *
 *  Some of the NumKong kernels need both the L2 norm of the vector and the sum of values in it.
 */
template <                                               //
    nk::numeric_dtype scalar_type_ = nk::bf16_t,         //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k, //
    nk_capability_t capability_ = nk_cap_icelake_k       //
    >
struct inner_metric;

template <                                         //
    nk::numeric_dtype scalar_type_ = nk::bf16_t,   //
    nk_capability_t capability_ = nk_cap_icelake_k //
    >
struct inner_norms;

/**
 *  @brief Per-vector header carrying the L2 norm and the row-sum used by DPBUSD compensation.
 *
 *  Both fields share the same width @c dot_result_t for the scalar - that's the natural lane
 *  width every downstream finalize and through-from-dot helper consumes. For float scalars
 *  @c sum is the optional de-mean component (unused by plain angular/euclidean); for integer
 *  scalars @c sum is the column-sum the DPBUSD-trick finalize needs to subtract bias.
 */
template <typename dot_result_type_>
struct vector_norms {
    dot_result_type_ sum;
    dot_result_type_ sumsq;
};

/**
 *  @brief Internal trampoline for using `nk_reduce_moments` NumKong APIs to compute L2 vector norms.
 *
 *  Returns @c vector_norms<dot_result_t, dot_result_t> - the same width that downstream finalize
 *  and through helpers consume. For integer scalars this narrows the i64/u64 native output of
 *  @c nk_reduce_moments_* to i32/u32; for float scalars the @c static_cast is a no-op identity
 *  (f32→f32 for bf16/f16/fp8, f64→f64 for f32/f64). Narrowing is safe for embedding dimensions
 *  up to ~16 M for sums and ~260 K for sumsq on i8 - all realistic ranges.
 */
template <nk::numeric_dtype scalar_type_>
inline vector_norms<typename scalar_type_::dot_result_t> invoke_inner_norms_( //
    nk_kernel_reduce_moments_punned_t function, nk::vector_view<scalar_type_> vec) noexcept {

    using native_sum_t = typename scalar_type_::reduce_moments_sum_t;
    using native_sumsq_t = typename scalar_type_::reduce_moments_sumsq_t;
    using dot_result_t = typename scalar_type_::dot_result_t;

    // NumKong's `count` arg is in @b values (one packed element each); `vec.size_values()` is the
    // matching method. `vec.size()` would give dimensions instead, which mismatches for sub-byte.
    native_sum_t native_sum;
    native_sumsq_t native_sumsq;
    function(vec.data(), vec.size_values(), sizeof(scalar_type_), &native_sum, &native_sumsq);
    return {static_cast<dot_result_t>(native_sum), static_cast<dot_result_t>(native_sumsq)};
}

template <nk_capability_t capability_>
struct inner_norms<nk::bf16_t, capability_> {

    inline vector_norms<nk::bf16_t::dot_result_t> operator()(nk::vector_view<nk::bf16_t> vec) const noexcept {
#if NK_TARGET_GENOA
        if (capability_ & nk_cap_genoa_k) return invoke_inner_norms_(nk_reduce_moments_bf16_genoa, vec);
#endif
#if NK_TARGET_SKYLAKE
        if (capability_ & nk_cap_skylake_k) return invoke_inner_norms_(nk_reduce_moments_bf16_skylake, vec);
#endif
#if NK_TARGET_HASWELL
        if (capability_ & nk_cap_haswell_k) return invoke_inner_norms_(nk_reduce_moments_bf16_haswell, vec);
#endif
#if NK_TARGET_NEONBFDOT
        if (capability_ & nk_cap_neonbfdot_k) return invoke_inner_norms_(nk_reduce_moments_bf16_neonbfdot, vec);
#endif
#if NK_TARGET_RVV
        if (capability_ & nk_cap_rvv_k) return invoke_inner_norms_(nk_reduce_moments_bf16_rvv, vec);
#endif
#if NK_TARGET_V128RELAXED
        if (capability_ & nk_cap_v128relaxed_k) return invoke_inner_norms_(nk_reduce_moments_bf16_v128relaxed, vec);
#endif
        return invoke_inner_norms_(nk_reduce_moments_bf16_serial, vec);
    }
};

/**
 *  @brief Internal helper for assembling a 1-to-1 distance between a query and a single candidate.
 *
 *  Mirrors @c `invoke_distances_` in template shape and call sequence, but with a single state and
 *  a single load/update per step - the loop body and tail do @b 4x less compute. The finalize step
 *  is hardcoded fan-out-4 in NumKong, so we pad slots 1-3 with zero-initialized states (and zero
 *  norms / compensation sums) and read result lane 0. The padded lanes burn a small fixed amount
 *  of horizontal-reduction work in finalize but the @b O(D) loop savings dominate at any non-tiny
 *  dimension count.
 */
template <
    // SIMD state like `nk_dot_bf16x32_state_genoa_t` and the operand vector type like `nk_b512_vec_t`.
    typename state_type_, typename vector_type_, typename finalized_inners_type_,
    // Lane type the through-from-dot helper actually writes - picked at the call site, not deduced.
    typename output_lane_type_,
    // USearch's in-house L1/L2/L3 cache-management stateless hints.
    typename prefetch_query_, typename prefetch_candidate_, typename demote_candidate_,
    // NumKong's state init/update/finalize and the metric-specific through-from-dot post-processor.
    typename init_state_, typename update_state_, typename finalize_inner_states_, typename distances_from_inners_,
    // NumKong's full and partial vector loaders from `cast/skylake.h` and friends.
    typename load_step_, typename load_tail_,
    // Auto-inferred by compiler from the provided query vector.
    nk::numeric_dtype scalar_type_>

inline f32_t invoke_distance_(                                                                          //
    init_state_ init_state, update_state_ update_state,                                                 //
    finalize_inner_states_ finalize_inner_states, distances_from_inners_ distances_from_inners,         //
    load_step_ load_step, std::size_t step_dimensions, load_tail_ load_tail,                            //
    nk::vector_view<scalar_type_> query, vector_norms<typename scalar_type_::dot_result_t> query_norms, //
    nk::vector_view<scalar_type_> candidate,
    vector_norms<typename scalar_type_::dot_result_t> candidate_norms) noexcept {

    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_type_>();
    std::size_t const total_dimensions = query.size();
    std::size_t const tail_dimensions = total_dimensions % step_dimensions;
    using dot_result_t = typename scalar_type_::dot_result_t;

    prefetch_query_ prefetch_query;
    prefetch_candidate_ prefetch_candidate;
    demote_candidate_ demote_candidate;

    NK_ALIGN64 state_type_ state;
    init_state(&state);

    auto const* query_base = query.data();
    auto const* candidate_base = candidate.data();
    prefetch_query(query_base);
    prefetch_candidate(candidate_base);

    NK_ALIGN64 vector_type_ query_vec;
    NK_ALIGN64 vector_type_ candidate_vec;
    std::size_t passed_dimensions = 0;
    for (; passed_dimensions + step_dimensions <= total_dimensions; passed_dimensions += step_dimensions) {
        std::size_t const passed_values = passed_dimensions / dims_per_value;
        std::size_t const next_passed_values = (passed_dimensions + step_dimensions) / dims_per_value;
        load_step(query_base + passed_values, &query_vec);
        load_step(candidate_base + passed_values, &candidate_vec);

        update_state(&state, query_vec, candidate_vec, passed_dimensions, step_dimensions);

        prefetch_query(query_base + next_passed_values);
        prefetch_candidate(candidate_base + next_passed_values);
        demote_candidate(candidate_base + passed_values);
    }

    if (tail_dimensions) {
        std::size_t const passed_values = passed_dimensions / dims_per_value;
        load_tail(query_base + passed_values, &query_vec, tail_dimensions);
        load_tail(candidate_base + passed_values, &candidate_vec, tail_dimensions);

        update_state(&state, query_vec, candidate_vec, passed_dimensions, tail_dimensions);
        demote_candidate(candidate_base + passed_values);
    }

    // NumKong's `_finalize_*` is fan-out-4. Real state in slot 0; pad slots 1-3 with zero-init.
    NK_ALIGN64 state_type_ pad_states[3];
    init_state(&pad_states[0]);
    init_state(&pad_states[1]);
    init_state(&pad_states[2]);

    NK_ALIGN64 finalized_inners_type_ dot_products_vec;
    constexpr bool needs_compensation = std::is_invocable< //
        finalize_inner_states_,                            //
        state_type_*, state_type_*, state_type_*, state_type_*, nk_size_t, dot_result_t, finalized_inners_type_ const*,
        finalized_inners_type_*>::value;
    if constexpr (!needs_compensation) {
        finalize_inner_states(&state, &pad_states[0], &pad_states[1], &pad_states[2], total_dimensions, //
                              &dot_products_vec);
    }
    else {
        // Real candidate sum in lane 0; zeros in 1-3 (their dot lanes are discarded below).
        alignas(16) std::array<dot_result_t, 4> sums_pad = {candidate_norms.sum, 0, 0, 0};
        auto const* sums_vec = reinterpret_cast<finalized_inners_type_ const*>(sums_pad.data());
        finalize_inner_states(&state, &pad_states[0], &pad_states[1], &pad_states[2], total_dimensions, //
                              query_norms.sum, sums_vec, &dot_products_vec);
    }

    // Through helper consumes the 4-lane dot vec and writes a 4-lane distance vec; we read lane 0.
    NK_ALIGN64 finalized_inners_type_ distances_from_dot_products_vec;
    {
        alignas(16) std::array<dot_result_t, 4> sumsqs_pad = {candidate_norms.sumsq, 0, 0, 0};
        auto const* sumsqs_vec = reinterpret_cast<finalized_inners_type_ const*>(sumsqs_pad.data());
        auto* result_vec = reinterpret_cast<finalized_inners_type_*>(&distances_from_dot_products_vec);
        distances_from_inners(&dot_products_vec, query_norms.sumsq, sumsqs_vec, result_vec);
    }

    auto const& result_array = *reinterpret_cast<std::array<output_lane_type_, 4> const*>(
        &distances_from_dot_products_vec);
    return static_cast<f32_t>(result_array[0]);
}

/**
 *  @brief Inner-product-as-distance "through" for pure-dot kernels - returns @c 1 - dot per lane.
 *
 *  Signature mirrors NumKong's @c `nk_*_through_*_from_dot_*` family: dots vec, query sumsq
 *  scalar (unused for pure dot), target sumsqs vec (unused), result vec.
 */
template <typename norm_type_, typename finalized_inners_type_>
inline void inner_product_distance_(finalized_inners_type_ const* dots_vec, norm_type_ /*query_sumsq_unused*/,
                                    finalized_inners_type_ const* /*target_sumsqs_unused*/,
                                    finalized_inners_type_* result_vec) noexcept {

    if constexpr (std::is_integral_v<norm_type_>) {
        if constexpr (std::is_unsigned_v<norm_type_>) {
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
    else if constexpr (std::is_floating_point_v<norm_type_>) {
        if constexpr (std::is_same_v<norm_type_, nk_f32_t>) {
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
    else { static_assert(sizeof(norm_type_) == 0, "norm type must be integral or floating point"); }
}

/**
 *  @brief Internal helper for assembling 1-to-4 distances functions between a query vector and 4x candidates.
 *
 *  Unlike the @c `invoke_inner_norms_`, it doesn't use type-punned pointers to functions as arguments
 *  to simplify inlining for the compiler. Query vector typically remains in the L1/L2 cache, while the candidates
 *  are hinted to stay in L3/LLC to avoid washing out more relevant graph-traversal data-structures from L1/L2.
 */
template <
    // SIMD state like `nk_dot_bf16x32_state_genoa_t` and the operand vector type like `nk_b512_vec_t`.
    typename state_type_, typename vector_type_, typename finalized_inners_type_,
    // Lane type the through-from-dot helper actually writes - picked at the call site, not deduced.
    typename output_lane_type_,
    // USearch's in-house L1/L2/L3 cache-management stateless hints.
    typename prefetch_query_, typename prefetch_candidate_, typename demote_candidate_,
    // NumKong's state init/update/finalize and the metric-specific through-from-dot post-processor.
    typename init_state_, typename update_state_, typename finalize_inner_states_, typename distances_from_inners_,
    // NumKong's full and partial vector loaders from `cast/skylake.h` and friends.
    typename load_step_, typename load_tail_,
    // Auto-inferred by compiler from the provided query vector.
    nk::numeric_dtype scalar_type_>

inline std::array<f32_t, 4> invoke_distances_(                                                          //
    init_state_ init_state, update_state_ update_state,                                                 //
    finalize_inner_states_ finalize_inner_states, distances_from_inners_ distances_from_inners,         //
    load_step_ load_step, std::size_t step_dimensions, load_tail_ load_tail,                            //
    nk::vector_view<scalar_type_> query, vector_norms<typename scalar_type_::dot_result_t> query_norms, //
    std::array<nk::vector_view<scalar_type_>, 4> candidates,
    std::array<vector_norms<typename scalar_type_::dot_result_t>, 4> candidates_norms) noexcept {

    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_type_>();
    std::size_t const total_dimensions = query.size();
    std::size_t const tail_dimensions = total_dimensions % step_dimensions;
    using dot_result_t = typename scalar_type_::dot_result_t;

    // Stateless cache-hint policies.
    prefetch_query_ prefetch_query;
    prefetch_candidate_ prefetch_candidate;
    demote_candidate_ demote_candidate;

    // Each running dot-product needs its own state.
    NK_ALIGN64 state_type_ states[4];
    init_state(&states[0]);
    init_state(&states[1]);
    init_state(&states[2]);
    init_state(&states[3]);

    // Pre-set the streaming policy on the four candidates and the sticky policy on the query.
    auto const* query_base = query.data();
    auto const* candidate_bases[4] = {candidates[0].data(), candidates[1].data(), candidates[2].data(),
                                      candidates[3].data()};
    prefetch_query(query_base);
    prefetch_candidate(candidate_bases[0]);
    prefetch_candidate(candidate_bases[1]);
    prefetch_candidate(candidate_bases[2]);
    prefetch_candidate(candidate_bases[3]);

    // Step through one register worth of vector data per candidate, in parallel.
    NK_ALIGN64 vector_type_ query_vec;
    NK_ALIGN64 vector_type_ candidate_vecs[4];
    std::size_t passed_dimensions = 0;
    for (; passed_dimensions + step_dimensions <= total_dimensions; passed_dimensions += step_dimensions) {
        std::size_t const passed_values = passed_dimensions / dims_per_value;
        std::size_t const next_passed_values = (passed_dimensions + step_dimensions) / dims_per_value;
        load_step(query_base + passed_values, &query_vec);
        load_step(candidate_bases[0] + passed_values, &candidate_vecs[0]);
        load_step(candidate_bases[1] + passed_values, &candidate_vecs[1]);
        load_step(candidate_bases[2] + passed_values, &candidate_vecs[2]);
        load_step(candidate_bases[3] + passed_values, &candidate_vecs[3]);

        update_state(&states[0], query_vec, candidate_vecs[0], passed_dimensions, step_dimensions);
        update_state(&states[1], query_vec, candidate_vecs[1], passed_dimensions, step_dimensions);
        update_state(&states[2], query_vec, candidate_vecs[2], passed_dimensions, step_dimensions);
        update_state(&states[3], query_vec, candidate_vecs[3], passed_dimensions, step_dimensions);

        prefetch_query(query_base + next_passed_values);
        prefetch_candidate(candidate_bases[0] + next_passed_values);
        prefetch_candidate(candidate_bases[1] + next_passed_values);
        prefetch_candidate(candidate_bases[2] + next_passed_values);
        prefetch_candidate(candidate_bases[3] + next_passed_values);

        demote_candidate(candidate_bases[0] + passed_values);
        demote_candidate(candidate_bases[1] + passed_values);
        demote_candidate(candidate_bases[2] + passed_values);
        demote_candidate(candidate_bases[3] + passed_values);
    }

    if (tail_dimensions) {
        std::size_t const passed_values = passed_dimensions / dims_per_value;
        load_tail(query_base + passed_values, &query_vec, tail_dimensions);
        load_tail(candidate_bases[0] + passed_values, &candidate_vecs[0], tail_dimensions);
        load_tail(candidate_bases[1] + passed_values, &candidate_vecs[1], tail_dimensions);
        load_tail(candidate_bases[2] + passed_values, &candidate_vecs[2], tail_dimensions);
        load_tail(candidate_bases[3] + passed_values, &candidate_vecs[3], tail_dimensions);

        update_state(&states[0], query_vec, candidate_vecs[0], passed_dimensions, tail_dimensions);
        update_state(&states[1], query_vec, candidate_vecs[1], passed_dimensions, tail_dimensions);
        update_state(&states[2], query_vec, candidate_vecs[2], passed_dimensions, tail_dimensions);
        update_state(&states[3], query_vec, candidate_vecs[3], passed_dimensions, tail_dimensions);

        demote_candidate(candidate_bases[0] + passed_values);
        demote_candidate(candidate_bases[1] + passed_values);
        demote_candidate(candidate_bases[2] + passed_values);
        demote_candidate(candidate_bases[3] + passed_values);
    }

    // Finalize into the metric output in two back-to-back calls against the new pointer-convention
    // NumKong API. The dots vec is an ephemeral intermediate, never exposed to the caller.
    NK_ALIGN64 finalized_inners_type_ dot_products_vec;
    constexpr bool needs_compensation = std::is_invocable< //
        finalize_inner_states_,                            //
        state_type_*, state_type_*, state_type_*, state_type_*, nk_size_t, dot_result_t, finalized_inners_type_ const*,
        finalized_inners_type_*>::value;
    if constexpr (!needs_compensation) {
        finalize_inner_states(&states[0], &states[1], &states[2], &states[3], total_dimensions, //
                              &dot_products_vec);
    }
    else {
        alignas(16) std::array<dot_result_t, 4> candidates_sums_array = {
            candidates_norms[0].sum, candidates_norms[1].sum, candidates_norms[2].sum, candidates_norms[3].sum};
        auto const* candidates_sums_vec = reinterpret_cast<finalized_inners_type_ const*>(candidates_sums_array.data());
        finalize_inner_states(&states[0], &states[1], &states[2], &states[3], total_dimensions, //
                              query_norms.sum, candidates_sums_vec, &dot_products_vec);
    }

    // Now, infer the Angular or Euclidean distance from the inner product
    NK_ALIGN64 finalized_inners_type_ distances_from_dot_products_vec;
    {
        alignas(16) std::array<dot_result_t, 4> candidates_sumsqs_array = {
            candidates_norms[0].sumsq, candidates_norms[1].sumsq, candidates_norms[2].sumsq, candidates_norms[3].sumsq};
        auto const* candidates_sumsqs_vec = reinterpret_cast<finalized_inners_type_ const*>(
            candidates_sumsqs_array.data());
        auto* result_vec = reinterpret_cast<finalized_inners_type_*>(distances_from_dot_products_vec.data());
        distances_from_inners(&dot_products_vec, query_norms.sumsq, candidates_sumsqs_vec, result_vec);
    }

    // Explicitly downcast for when the dot-product or distance calculation result is an integer or double-precision.
    // The lane type here is `output_lane_type_` - whatever the caller's chosen through-function actually wrote into
    // `distances_from_dot_products_vec`. Using `dot_result_t` would silently misread bits whenever the through
    // changes lane width (e.g., u8 angular: dot is u32, angular_through writes f32).
    NK_ALIGN64 std::array<f32_t, 4> result_array;
    {
        auto const& distances_from_dot_products_array = *reinterpret_cast<std::array<output_lane_type_, 4> const*>(
            &distances_from_dot_products_vec);
        result_array[0] = static_cast<f32_t>(distances_from_dot_products_array[0]);
        result_array[1] = static_cast<f32_t>(distances_from_dot_products_array[1]);
        result_array[2] = static_cast<f32_t>(distances_from_dot_products_array[2]);
        result_array[3] = static_cast<f32_t>(distances_from_dot_products_array[3]);
    }
    return result_array;
}

template <nk_capability_t capability_> //
struct inner_metric<nk::bf16_t, nk_kernel_angular_k, capability_> {

    inline std::array<f32_t, 4> distances(      //
        nk::vector_view<nk::bf16_t> const from, //
        vector_norms<f32_t> from_norms,         //
        std::array<nk::vector_view<nk::bf16_t>, 4> const to_array,
        std::array<vector_norms<f32_t>, 4> to_norms) const noexcept {

#if NK_TARGET_GENOA
        if (capability_ & nk_cap_genoa_k)
            return invoke_distances_<                                                                 //
                nk_dot_bf16x32_state_genoa_t, nk_b512_vec_t, nk_b128_vec_t, /*output_lane=*/nk_f32_t, //
                prefetch_query_sse_t, prefetch_candidate_sse_t, demote_candidate_cldemote_t>(         //
                nk_dot_bf16x32_init_genoa, nk_dot_bf16x32_update_genoa, nk_dot_bf16x32_finalize_genoa,
                nk_angular_through_f32_from_dot_haswell_,                            //
                nk_load_b512_skylake_, /*step=*/32, nk_partial_load_b16x32_skylake_, //
                from, from_norm, to_array, to_norms);
#endif
        return invoke_distances_<                                                                 //
            nk_dot_bf16x8_state_serial_t, nk_b128_vec_t, nk_b128_vec_t, /*output_lane=*/nk_f32_t, //
            memory_hint_t, memory_hint_t, memory_hint_t>(                                         //
            nk_dot_bf16x8_init_serial, nk_dot_bf16x8_update_serial, nk_dot_bf16x8_finalize_serial,
            nk_angular_through_f32_from_dot_serial_,                         //
            nk_load_b128_serial_, /*step=*/8, nk_partial_load_b16x8_serial_, //
            from, from_norm, to_array, to_norms);
    }
};

template <nk_capability_t capability_> //
struct inner_metric<nk::u8_t, nk_kernel_dot_k, capability_> {

    inline std::array<f32_t, 4> distances(                                       //
        nk::vector_view<nk::u8_t> const from, vector_norms<nk_u32_t> from_norms, //
        std::array<nk::vector_view<nk::u8_t>, 4> const to_array,                 //
        std::array<vector_norms<nk_u32_t>, 4> to_norms) const noexcept {

#if NK_TARGET_ICELAKE
        if (capability_ & nk_cap_icelake_k)
            return invoke_distances_<                                                                 //
                nk_dot_u8x64_state_icelake_t, nk_b512_vec_t, nk_b128_vec_t, /*output_lane=*/nk_i32_t, //
                prefetch_query_sse_t, prefetch_candidate_sse_t, demote_candidate_cldemote_t>(
                nk_dot_u8x64_init_icelake, nk_dot_u8x64_update_icelake, nk_dot_u8x64_finalize_icelake,
                inner_product_distance_<nk_u32_t, nk_b128_vec_t>,                   //
                nk_load_b512_skylake_, /*step=*/64, nk_partial_load_b8x64_skylake_, //
                from, from_norms, to_array, to_norms);
#endif
        return invoke_distances_<                                                                //
            nk_dot_u8x16_state_serial_t, nk_b128_vec_t, nk_b128_vec_t, /*output_lane=*/nk_i32_t, //
            memory_hint_t, memory_hint_t, memory_hint_t>(                                        //
            nk_dot_u8x16_init_serial, nk_dot_u8x16_update_serial, nk_dot_u8x16_finalize_serial,
            inner_product_distance_<nk_u32_t, nk_b128_vec_t>,                 //
            nk_load_b128_serial_, /*step=*/16, nk_partial_load_b8x16_serial_, //
            from, from_norms, to_array, to_norms);
    }
};

template <nk_capability_t capability_> //
struct inner_metric<nk::i4x2_t, nk_kernel_euclidean_k, capability_> {

    inline std::array<f32_t, 4> distances(                                         //
        nk::vector_view<nk::i4x2_t> const from, vector_norms<nk_i32_t> from_norms, //
        std::array<nk::vector_view<nk::i4x2_t>, 4> const to_array,                 //
        std::array<vector_norms<nk_i32_t>, 4> to_norms) const noexcept {

#if NK_TARGET_HASWELL
        if (capability_ & nk_cap_haswell_k)
            return invoke_distances_<                                                                 //
                nk_dot_i4x32_state_haswell_t, nk_b128_vec_t, nk_b128_vec_t, /*output_lane=*/nk_f32_t, //
                prefetch_query_sse_t, prefetch_candidate_sse_t, demote_candidate_cldemote_t>(
                nk_dot_i4x32_init_haswell, nk_dot_i4x32_update_haswell, nk_dot_i4x32_finalize_haswell,
                nk_euclidean_through_i32_from_dot_haswell_,                        //
                nk_load_b128_haswell_, /*step=*/32, nk_partial_load_b4x32_serial_, //
                from, from_norms, to_array, to_norms);
#endif
        return invoke_distances_<                                                               //
            nk_dot_i4x16_state_serial_t, nk_b64_vec_t, nk_b128_vec_t, /*output_lane=*/nk_f32_t, //
            memory_hint_t, memory_hint_t, memory_hint_t>(                                       //
            nk_dot_i4x16_init_serial, nk_dot_i4x16_update_serial, nk_dot_i4x16_finalize_serial,
            nk_euclidean_through_i32_from_dot_serial_,                       //
            nk_load_b64_serial_, /*step=*/16, nk_partial_load_b4x16_serial_, //
            from, from_norms, to_array, to_norms);
    }
};

} // namespace usearch
} // namespace unum
