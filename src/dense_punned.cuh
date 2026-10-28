/**
 *  @file src/dense_punned.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What the dense dispatch unit and the class units agree on: the homes a variant runs in,
 *      the verb table one variant compiles, and the class tables each class unit instantiates.
 *
 *  It includes no tower: the dispatch unit sees a variant only through its home's verb table, and
 *  @c dense_specializations.cuh, which every class unit includes, fills those tables from the
 *  towers. The scalar tables, the CPU classes and the call scaffolding every modality shares come
 *  from @c lib.hpp.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <span>        // `std::span` - the search verbs' landing matrices
#include <string_view> // `std::string_view` - kernel tokens & checkpoint paths
#include <type_traits> // `std::is_same_v` - a home's executor

#include <numkong/tensor.hpp> // `nk::matrix_span` - the matching verbs' landing matrices

#include "usearch/dense_index.h"
#include "usearch/dense/types.hpp" // `dense::rows_view_t`, `dense::refine_result_t`, `dense::compact_result_t`

#include "lib.hpp"

#pragma region Scaffolding

/** A kernel kind lifted back onto the C metric vocabulary. */
inline usearch_dense_metric_t usearch_metric_of(nk_kernel_kind_t kernel) noexcept {
    switch (kernel) {
    case nk_kernel_dot_k: return usearch_dense_metric_inner_k;
    case nk_kernel_angular_k: return usearch_dense_metric_angular_k;
    case nk_kernel_euclidean_k: return usearch_dense_metric_euclidean_k;
    case nk_kernel_hamming_k: return usearch_dense_metric_hamming_k;
    case nk_kernel_jaccard_k: return usearch_dense_metric_jaccard_k;
    default: return usearch_dense_metric_unknown_k;
    }
}

/** One key a MaxSim query kept, waiting in its thread's run to scatter into the call's tape. */
struct maxsim_match_t {
    std::size_t query;
    usearch_key_t key;
    usearch_distance_t score;
};

static_assert(sizeof(maxsim_match_t) == 32, "A kept key is a 32-byte hot entry");

/** Where a MaxSim call's queries land: one run per executor thread, in rank order per query and
 *  growing in that thread's arena, and the call's own arena, where a host reduction stages the
 *  rectangle it ranks into first. Called as an engine's lander, it appends one kept key. */
struct maxsim_landing_t {
    std::span<growing_buffer<maxsim_match_t, thread_allocator_t>> runs;
    thread_allocator_t staging;

    template <typename key_type_, typename total_type_>
    status_t operator()(std::size_t thread, std::size_t query, key_type_ const& key, total_type_ total) const noexcept {
        return runs[thread].push_back(
            {query, std::bit_cast<usearch_key_t>(key), static_cast<usearch_distance_t>(total)});
    }
};

/** Opens one empty MaxSim run per thread that ranks queries into @p runs: one per thread of a host
 *  @p executor over that thread's arena, or one over the call's @p staging for a device executor,
 *  whose host reduction ranks on the calling thread. */
template <typename executor_type_>
status_t open_maxsim_runs(growing_buffer<growing_buffer<maxsim_match_t, thread_allocator_t>, thread_allocator_t>& runs,
                          executor_type_& executor, thread_allocator_t staging) noexcept {
    using run_t = growing_buffer<maxsim_match_t, thread_allocator_t>;
    if constexpr (is_gpu_executor<executor_type_>) {
        if (failed(runs.reserve_exactly(1))) return status_t::bad_alloc_k;
        runs.push_back_reserved(run_t {staging});
    }
    else {
        if (failed(runs.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            runs.push_back_reserved(run_t {thread_allocator_t {executor.arena(thread)}});
    }
    return status_t::success_k;
}

#pragma endregion Scaffolding

#pragma region Home Tables

/** The knobs every walking verb's C arguments name, which each variant lowers onto its own
 *  config. */
struct dense_walk_config_t {
    u32_t expansion_factor = 0;
    traversal_t traversal = traversal_t::walk_k;
};

/**
 *  @brief The verbs one dense variant compiles in its class unit, as function pointers over its
 *      index passed as @c void*, which only these entries cast back to its type.
 *
 *  Every home fills every entry, answering @c unsupported_k or @c read_only_k where it serves none,
 *  so the dispatch unit visits five homes and never a variant. Each walking verb answers the
 *  traversal telemetry the C stats report, and a refinement or a cursor lives behind a @c void* its
 *  own entries open, step and free.
 */
template <dense_home_t home_>
struct dense_verbs {
    using executor_t = typename dense_home_executor<home_>::type;
    using key_t = us::utilities::uuid_t;
    using keys_view_t = nk::vector_view<key_t>;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;
    using rows_view_t = dense::rows_view_t;
    using rows_span_t = dense::rows_span_t;
    using walked_t = expected<usearch_traversal_stats_t>;
    using refined_t = expected<dense::refine_result_t>;

    /** The scalar the variant stores rows in. */
    nk_dtype_t dtype;

    /** The CPU class a host or storage variant's kits compile for; unknown on a device. */
    cpu_class_t cpu_class;

    /** The GPU class a device variant's kernels compile for; unknown on the host and storage. */
    gpu_class_t gpu_class;

    void* (*create)(nk_kernel_kind_t kernel, forkunion_allocator_t tables) noexcept;
    void (*destroy)(void* index) noexcept;
    void (*bound_shards)(void* index, std::size_t rows) noexcept;

    /** Appends one shard on a host memory domain or a device's vendor ordinal. */
    status_t (*emplace)(void* index, int placement) noexcept;
    status_t (*view)(void* index, std::string_view path, std::span<std::string_view const> paths) noexcept;

    nk_kernel_kind_t (*kernel)(void const* index) noexcept;

    /** The device shard @p shard sits on, or @c host_device_k. */
    int (*device)(void const* index, std::size_t shard) noexcept;
    std::size_t (*shards_count)(void const* index) noexcept;
    std::size_t (*dimensions)(void const* index) noexcept;
    std::size_t (*connectivity)(void const* index) noexcept;
    std::size_t (*size)(void const* index) noexcept;
    std::size_t (*capacity)(void const* index) noexcept;
    std::size_t (*memory_usage)(void const* index) noexcept;
    expected<std::size_t> (*storage_requirements)(void const* index) noexcept;

    status_t (*reserve)(void* index, std::size_t capacity, connectivity_t connectivity,
                        std::size_t dimensions) noexcept;
    status_t (*clear)(void* index) noexcept;
    walked_t (*insert)(void* index, keys_view_t keys, rows_view_t rows, progress_ticks_t& ticks, executor_t& executor,
                       u32_t expansion_factor) noexcept;
    refined_t (*refine)(void* index, executor_t& executor, u32_t expansion_factor, progress_ticks_t& ticks) noexcept;

    /** Moves every row of each of @p donors in, nulling each slot whose index it consumed. */
    refined_t (*absorb)(void* index, std::span<void*> donors, executor_t& executor, u32_t expansion_factor,
                        progress_ticks_t& ticks) noexcept;
    expected<dense::compact_result_t> (*compact)(void* index, std::span<void* const> sources, executor_t& executor,
                                                 dense::compact_config_t config, progress_ticks_t& ticks) noexcept;

    expected<void*> (*refinement_open)(void* index, std::span<void* const> donors, executor_t& executor,
                                       u32_t expansion_factor) noexcept;
    status_t (*refinement_run)(void* index, void* refinement, executor_t& executor, progress_ticks_t& ticks) noexcept;
    dense::refine_result_t (*refinement_running)(void const* refinement) noexcept;

    /** Whether the refinement landed and consumed its donor at @p donor. */
    bool (*refinement_landed)(void const* refinement, std::size_t donor) noexcept;
    refined_t (*refinement_finish)(void* index, void* refinement, executor_t& executor) noexcept;
    void (*refinement_free)(void* refinement) noexcept;

    expected<dense::health_result_t> (*health)(void* index, executor_t& executor) noexcept;
    status_t (*count)(void* index, std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) noexcept;

    /** Reads the entries of @p keys in the stored scalar into @p stored, then exports them into
     *  @p outputs, which may be the same rows when the scalars agree. */
    status_t (*get)(void* index, std::span<key_t const> keys, rows_span_t stored, rows_span_t outputs,
                    std::size_t wanted, std::span<std::size_t> entries, executor_t& executor) noexcept;
    walked_t (*search)(void* index, rows_view_t queries, executor_t& executor, dense_walk_config_t config,
                       keys_span_t keys, scores_span_t distances, counts_span_t counts) noexcept;
    walked_t (*search_if)(void* index, rows_view_t queries, executor_t& executor, dense_walk_config_t config,
                          keys_span_t keys, scores_span_t distances, counts_span_t counts,
                          predicate_t& predicate) noexcept;

    /** Narrows a walk or a scan by an expensive host predicate, asked once per round each thread
     *  settles. */
    walked_t (*search_if_expensive)(void* index, rows_view_t queries, executor_t& executor, dense_walk_config_t config,
                                    keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                    usearch_predicate_t predicate) noexcept;
    walked_t (*multiply)(void* index, rows_view_t left, executor_t& executor, dense_walk_config_t config,
                         keys_span_t keys, scores_span_t products, counts_span_t counts) noexcept;
    walked_t (*multiply_if)(void* index, rows_view_t left, executor_t& executor, dense_walk_config_t config,
                            keys_span_t keys, scores_span_t products, counts_span_t counts,
                            predicate_t& predicate) noexcept;
    walked_t (*maxsim)(void* index, rows_view_t queries, std::span<std::size_t const> query_offsets,
                       executor_t& executor, dense_walk_config_t config, std::size_t wanted, counts_span_t scored,
                       maxsim_landing_t landing) noexcept;
    walked_t (*maxsim_if)(void* index, rows_view_t queries, std::span<std::size_t const> query_offsets,
                          executor_t& executor, dense_walk_config_t config, std::size_t wanted, counts_span_t scored,
                          maxsim_landing_t landing, predicate_t& predicate) noexcept;
    walked_t (*attend)(void* index, rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                       f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                       executor_t& executor, dense_walk_config_t config) noexcept;
    walked_t (*attend_if)(void* index, rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                          f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                          executor_t& executor, dense_walk_config_t config, predicate_t& predicate) noexcept;

    expected<void*> (*cursor_open)(void* index) noexcept;
    expected<std::size_t> (*cursor_next)(void* index, void* cursor, std::span<key_t> keys,
                                         executor_t& executor) noexcept;
    void (*cursor_free)(void* cursor) noexcept;

    status_t (*save)(void* index, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*save_buffer)(void* index, bytes_span_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*load)(void* index, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*load_buffer)(void* index, bytes_view_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;
};

/** The variants one class unit compiles for one home: @c find answers the verb table of a dtype, or
 *  null where the class picks nothing its parent does not, so a ladder falls to the next rung. */
template <dense_home_t home_, auto class_>
struct dense_class_tables {
    static dense_verbs<home_> const* find(nk_dtype_t dtype) noexcept;
};

/** One picked variant as the dispatch unit holds it for one call: its home's verbs, and the index
 *  they serve. */
template <dense_home_t home_>
struct punned_dense_index {
    using executor_t = typename dense_verbs<home_>::executor_t;

    dense_verbs<home_> const& verbs;
    void* index;
};

/** The executor a null handle runs a device variant's work on: streams or queues of its own on its
 *  devices, one per distinct device every shard there shares, never a legacy default stream, which
 *  would wait on every blocking stream of the process. */
template <dense_home_t home_>
    requires is_gpu_executor<typename punned_dense_index<home_>::executor_t>
usearch_executor_t call_executor_for(punned_dense_index<home_> const& punned, usearch_executor_handle_t& call_executor,
                                     usearch_outcome_t* outcome) noexcept {
    std::size_t const shards = punned.verbs.shards_count(punned.index);
    growing_buffer<int> devices {global_allocator};
    if (failed(devices.reserve_exactly(shards)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), nullptr;
    for (std::size_t shard = 0; shard != shards; ++shard) {
        int const device = punned.verbs.device(punned.index, shard);
        bool opened = false;
        for (int const earlier : devices) opened |= earlier == device;
        if (!opened) devices.push_back_reserved(device);
    }
    using executor_t = typename punned_dense_index<home_>::executor_t;
    auto [built, status] = executor_t::spawn(driver_of(gpu_drivers(), std::type_identity<executor_t> {}), devices);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor = gpu_executor_t {std::move(built)};
    return &call_executor;
}

#pragma endregion Home Tables
