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
struct inner_norm;

/**
 *  @brief Per-vector header carrying the L2 norm and the row-sum used by DPBUSD compensation.
 *
 *  Both fields share the same width @c dot_result_t for the scalar - that's the natural lane
 *  width every downstream finalize and through-from-dot helper consumes. For float scalars
 *  @c sum is the optional de-mean component (unused by plain angular/euclidean); for integer
 *  scalars @c sum is the column-sum the DPBUSD-trick finalize needs to subtract bias.
 */
template <typename sum_type_, typename sumsq_type_>
struct vector_norms {
    sum_type_ sum;
    sumsq_type_ sumsq;
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
inline vector_norms<typename scalar_type_::dot_result_t, typename scalar_type_::dot_result_t> invoke_inner_norm_(
    nk_kernel_reduce_moments_punned_t function, std::span<scalar_type_> vec) noexcept {

    using native_sum_t = typename scalar_type_::reduce_moments_sum_t;
    using native_sumsq_t = typename scalar_type_::reduce_moments_sumsq_t;
    using stored_t = typename scalar_type_::dot_result_t;

    native_sum_t native_sum;
    native_sumsq_t native_sumsq;
    function(vec.data(), vec.size(), sizeof(scalar_type_), &native_sum, &native_sumsq);
    return {static_cast<stored_t>(native_sum), static_cast<stored_t>(native_sumsq)};
}

template <nk_capability_t capability_>
struct inner_norm<nk::bf16_t, capability_> {

    inline vector_norms<nk::bf16_t::dot_result_t, nk::bf16_t::dot_result_t> operator()(
        std::span<nk::bf16_t> vec) const noexcept {
#if NK_TARGET_GENOA
        if (capability_ & nk_cap_genoa_k) return invoke_inner_norm_(nk_reduce_moments_bf16_genoa, vec);
#endif
#if NK_TARGET_SKYLAKE
        if (capability_ & nk_cap_skylake_k) return invoke_inner_norm_(nk_reduce_moments_bf16_skylake, vec);
#endif
#if NK_TARGET_HASWELL
        if (capability_ & nk_cap_haswell_k) return invoke_inner_norm_(nk_reduce_moments_bf16_haswell, vec);
#endif
#if NK_TARGET_NEONBFDOT
        if (capability_ & nk_cap_neonbfdot_k) return invoke_inner_norm_(nk_reduce_moments_bf16_neonbfdot, vec);
#endif
#if NK_TARGET_RVV
        if (capability_ & nk_cap_rvv_k) return invoke_inner_norm_(nk_reduce_moments_bf16_rvv, vec);
#endif
#if NK_TARGET_V128RELAXED
        if (capability_ & nk_cap_v128relaxed_k) return invoke_inner_norm_(nk_reduce_moments_bf16_v128relaxed, vec);
#endif
        return invoke_inner_norm_(nk_reduce_moments_bf16_serial, vec);
    }
};

/**
 *  @brief Internal helper for assemmbling 1-to-1 distance functions between vectors
 *
 *  Unlike the @c `invoke_inner_norm_`, it doesn't use type-punned pointers to functions as arguments
 *  to simplify inlining for the compiler. Query vector typically remains in the L1 cache
 */
template <                                       //
    typename state_type_, typename vector_type_, // nk_dot_f32x16_state_skylake_t, __m512
    typename init_state_, typename update_state_, typename finalize_inner_states_,      // NumKong's APIs
    typename prefetch_query_, typename load_query_, typename demote_query_,             // assymmetric helpers for query
    typename prefetch_candidate_, typename load_candidate_, typename demote_candidate_, // and for the candidate vectors
    nk::numeric_dtype scalar_type_                                                      // auto-inferred by compiler
    >
inline f32_t invoke_distance_(                                                                        //
    init_state_ init_state, update_state_ update_state, finalize_inner_states_ finalize_inner_states, //
    std::span<scalar_type_> query, std::span<scalar_type_> candidate) noexcept {

    using dot_result_t = typename scalar_type_::dot_result_t;
    state_type_ state;
    init_state(&state);
    vector_type_ query_vec, candidate_vec;

    function(vec.data(), vec.size(), sizeof(scalar_type_), &sum, &sumsq);
    return static_cast<f32_t>(sumsq);
}

/**
 *  @brief Internal helper for assembling 1-to-4 distances functions between a query vector and 4x candidates.
 *
 *  Unlike the @c `invoke_inner_norm_`, it doesn't use type-punned pointers to functions as arguments
 *  to simplify inlining for the compiler. Query vector typically remains in the L1/L2 cache, while the candidates
 *  are hinted to stay in L3/LLC to avoid washing out more relevant graph-traversal data-structures from L1/L2.
 *
 *
 */
template <
    // SIMD state like `nk_dot_bf16x32_state_genoa_t` and the operand vector type like `nk_b512_vec_t`.
    typename state_type_, typename vector_type_, typename finalized_inners_type_,
    // USearch's in-house L1/L2/L3 cache-management stateless hints.
    typename prefetch_query_, typename prefetch_candidate_, typename demote_candidate_,
    // NumKong's state init/update/finalize and the metric-specific through-from-dot post-processor.
    typename init_state_, typename update_state_, typename finalize_inner_states_, typename distances_from_inners_,
    // NumKong's full and partial vector loaders from `cast/skylake.h` and friends.
    typename load_step_, typename load_tail_,
    // Auto-inferred by compiler from the provided query vector.
    nk::numeric_dtype scalar_type_, typename sum_type_ = f32_t, typename sumsq_type_ = f32_t>

inline std::array<f32_t, 4> invoke_distances_(                                                  //
    init_state_ init_state, update_state_ update_state,                                         //
    finalize_inner_states_ finalize_inner_states, distances_from_inners_ distances_from_inners, //
    load_step_ load_step, std::size_t step_dimensions, load_tail_ load_tail,                    //
    std::span<scalar_type_> query, vector_norms<sum_type_, sumsq_type_> query_norms,            //
    std::array<std::span<scalar_type_>, 4> candidates,
    std::array<vector_norms<sum_type_, sumsq_type_>, 4> candidates_norms) noexcept {

    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_type_>();
    std::size_t const total_dimensions = query.size() * dims_per_value;
    std::size_t const tail_dimensions = total_dimensions % step_dimensions;

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
    prefetch_query(query.data());
    prefetch_candidate(candidates[0].data());
    prefetch_candidate(candidates[1].data());
    prefetch_candidate(candidates[2].data());
    prefetch_candidate(candidates[3].data());

    // Step through one register worth of vector data per candidate, in parallel.
    NK_ALIGN64 vector_type_ query_vec;
    NK_ALIGN64 vector_type_ candidate_vecs[4];
    std::size_t passed_dimensions = 0;
    for (; passed_dimensions + step_dimensions <= total_dimensions; passed_dimensions += step_dimensions) {
        load_step(&query[passed_dimensions], &query_vec);
        load_step(&candidates[0][passed_dimensions], &candidate_vecs[0]);
        load_step(&candidates[1][passed_dimensions], &candidate_vecs[1]);
        load_step(&candidates[2][passed_dimensions], &candidate_vecs[2]);
        load_step(&candidates[3][passed_dimensions], &candidate_vecs[3]);

        update_state(&states[0], query_vec, candidate_vecs[0], passed_dimensions, step_dimensions);
        update_state(&states[1], query_vec, candidate_vecs[1], passed_dimensions, step_dimensions);
        update_state(&states[2], query_vec, candidate_vecs[2], passed_dimensions, step_dimensions);
        update_state(&states[3], query_vec, candidate_vecs[3], passed_dimensions, step_dimensions);

        prefetch_query(&query[passed_dimensions + step_dimensions]);
        prefetch_candidate(&candidates[0][passed_dimensions + step_dimensions]);
        prefetch_candidate(&candidates[1][passed_dimensions + step_dimensions]);
        prefetch_candidate(&candidates[2][passed_dimensions + step_dimensions]);
        prefetch_candidate(&candidates[3][passed_dimensions + step_dimensions]);

        demote_candidate(&candidates[0][passed_dimensions]);
        demote_candidate(&candidates[1][passed_dimensions]);
        demote_candidate(&candidates[2][passed_dimensions]);
        demote_candidate(&candidates[3][passed_dimensions]);
    }

    if (tail_dimensions) {
        load_tail(&query[passed_dimensions], &query_vec, tail_dimensions);
        load_tail(&candidates[0][passed_dimensions], &candidate_vecs[0], tail_dimensions);
        load_tail(&candidates[1][passed_dimensions], &candidate_vecs[1], tail_dimensions);
        load_tail(&candidates[2][passed_dimensions], &candidate_vecs[2], tail_dimensions);
        load_tail(&candidates[3][passed_dimensions], &candidate_vecs[3], tail_dimensions);

        update_state(&states[0], query_vec, candidate_vecs[0], passed_dimensions, tail_dimensions);
        update_state(&states[1], query_vec, candidate_vecs[1], passed_dimensions, tail_dimensions);
        update_state(&states[2], query_vec, candidate_vecs[2], passed_dimensions, tail_dimensions);
        update_state(&states[3], query_vec, candidate_vecs[3], passed_dimensions, tail_dimensions);

        demote_candidate(&candidates[0][passed_dimensions]);
        demote_candidate(&candidates[1][passed_dimensions]);
        demote_candidate(&candidates[2][passed_dimensions]);
        demote_candidate(&candidates[3][passed_dimensions]);
    }

    // Finalize into the metric output in two back-to-back calls against the new pointer-convention
    // NumKong API. The dot-vec is an ephemeral intermediate, never exposed to the caller.
    NK_ALIGN64 finalized_inners_type_ dots, distances;
    alignas(16) std::array<f32_t, 4> distances;
    auto const* norms_vec = reinterpret_cast<nk_b128_vec_t const*>(candidates_norms.data());
    auto* result_vec = reinterpret_cast<nk_b128_vec_t*>(distances.data());

    finalize_inner_states(&states[0], &states[1], &states[2], &states[3], total_dimensions, &dots);
    distances_from_inners(&dots, query_norms, norms_vec, result_vec);
    return distances;
}

template <nk_capability_t capability_> //
struct inner_metric<nk::bf16_t, nk_kernel_angular_k, capability_> {

    inline std::array<f32_t, 4> distances(                 //
        std::span<nk::bf16_t> const from, f32_t from_norm, //
        std::array<std::span<nk::bf16_t>, 4> const to_array, std::array<f32_t, 4> to_norms) const noexcept {

#if NK_TARGET_GENOA
        if (capability_ & nk_cap_genoa_k)
            return invoke_distances_<                                       //
                nk_dot_bf16x32_state_genoa_t, nk_b512_vec_t, nk_b128_vec_t, //
                prefetch_query_sse_t, prefetch_candidate_sse_t, demote_candidate_cldemote_t>(
                nk_dot_bf16x32_init_genoa, nk_dot_bf16x32_update_genoa, nk_dot_bf16x32_finalize_genoa,
                nk_angular_through_f32_from_dot_haswell_,                            //
                nk_load_b512_skylake_, /*step=*/32, nk_partial_load_b16x32_skylake_, //
                from, from_norm, to_array, to_norms);
#endif
        return {}; // runtime dispatcher should never route a non-Genoa capability here
    }
};

} // namespace usearch
} // namespace unum
