/**
 *  @file src/dense_punned.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What the dense dispatch unit and the class units agree on: the residencies a variant runs
 *      in, the verb table one variant compiles, and the class tables each class unit instantiates.
 *
 *  It includes no tower: the dispatch unit sees a variant only through its residency's verb table,
 *  and @c dense_specializations.cuh, which every class unit includes, fills those tables from the
 *  towers. The scalar tables, the CPU classes and the call scaffolding every modality shares come
 *  from @c lib.hpp.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span>        // `std::span` - the search verbs' output matrices
#include <string_view> // `std::string_view` - kernel tokens & checkpoint paths
#include <type_traits> // `std::is_same_v` - a residency's executor

#include <numkong/tensor.hpp> // The verbs' output matrices

#include "usearch/dense_index.h"
#include "usearch/dense/types.hpp" // `dense::rows_view_t`, `dense::products_tape`, `dense::refine_report_t`

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

/** Where a ranking writes: @c wanted keys and scores per query, distances or products, and how
 *  many each kept. */
using ranked_matches_span_t = dense::ranked_matches_span<us::utilities::uuid128_t, f64_t>;

/** The keys each query's walk starts from, as a ranking's keys and counts lay them out. */
using ranked_hints_view_t = dense::ranked_hints_view<us::utilities::uuid128_t>;

/** Where a product or MaxSim writes, in the arena's memory for its residency: each query's keys
 *  and the ABI's scores from its offset, as the C tape reads them. */
using products_tape_t = dense::products_tape<us::utilities::uuid128_t, usearch_distance_t>;

/** Queries of many rows each: the rows, the offsets cutting them, one more than the queries, and
 *  the keys every row's walks of a query start from, empty for none. */
struct maxsim_queries_view_t {
    dense::rows_view_t rows;
    std::span<std::size_t const> offsets;
    ranked_hints_view_t hints;
};

/** What an attention reads besides its queries and where it writes: the heads each token's rows
 *  hold, the values a kept key weighs, the logits' scale, the outputs, and the carried softmax
 *  states, empty for none. */
struct attention_arguments_t {
    std::size_t heads = 1;
    value_lookup_t value_lookup;
    f32_t score_scale = 1;
    dense::rows_span_t outputs;
    std::span<dense::softmax_state_t> carry;
};

#pragma endregion Scaffolding

#pragma region Variant Tables

/**
 *  @brief The verbs one dense variant compiles in its class unit, as function pointers over its
 *      index passed as @c void*, which only these entries cast back to its type.
 *
 *  Every entry takes <tt>(index, inputs, outputs, config, executor, ticks)</tt>, as much of it as the verb
 *  reads, with a walk's @c predicate_t beside its config. Every residency fills every entry,
 *  answering @c unsupported_k or @c read_only_k where it serves none, so the dispatch unit visits
 *  five residencies and never a variant. Each walking verb answers the traversal telemetry the C
 *  stats report, and a refinement or a cursor lives behind a @c void* that its own entries open,
 *  step and free.
 */
template <vendored_residency_t residency_>
struct dense_verbs {
    using executor_t = typename vendored_residency_executor<residency_>::type;
    using key_t = us::utilities::uuid128_t;
    using keys_view_t = nk::vector_view<key_t>;
    using rows_view_t = dense::rows_view_t;
    using rows_span_t = dense::rows_span_t;

    /** The scalar the variant stores rows in. */
    nk_dtype_t dtype;

    /** The CPU class a host or storage variant's kits compile for; unknown on a device. */
    cpu_class_t cpu_class;

    /** The GPU class a device variant's kernels compile for; unknown on the host and storage. */
    gpu_class_t gpu_class;

    // Lifetime
    void* (*create)(nk_kernel_kind_t kernel, dense::geometry_t geometry, forkunion_allocator_t tables) noexcept;
    void (*destroy)(void* index) noexcept;
    void (*bound_shards)(void* index, std::size_t nodes) noexcept;

    /** Appends one shard on a device, or on RAM on @p memory_domain. */
    status_t (*emplace)(void* index, arena_residency_t residency, fu_memory_domain_id_t memory_domain) noexcept;

    /** Opens a checkpoint in place, with @p ram a key table in its limit filled on @p executor. */
    status_t (*view)(void* index, std::string_view path, std::span<std::string_view const> paths,
                     usearch_ram_options_t const* ram, usearch_executor_t executor) noexcept;

    // Shape
    nk_kernel_kind_t (*kernel)(void const* index) noexcept;
    dense::geometry_t (*geometry)(void const* index) noexcept;

    /** Where shard @p shard computes. */
    arena_residency_t (*residency)(void const* index, std::size_t shard) noexcept;
    std::size_t (*shards_count)(void const* index) noexcept;

    /** How the index's last load from a path read its parts. */
    usearch_load_path_t (*load_path)(void const* index) noexcept;
    std::size_t (*size)(void const* index) noexcept;
    std::size_t (*capacity)(void const* index) noexcept;
    std::size_t (*memory_usage)(void const* index) noexcept;
    expected<std::size_t> (*storage_requirements)(void const* index) noexcept;

    // Writes

    /** Whether the index takes writes, which a view refuses with @c read_only_k. */
    status_t (*serves_writes)(void const* index) noexcept;
    status_t (*reserve)(void* index, std::size_t capacity) noexcept;
    status_t (*clear)(void* index) noexcept;
    expected<traversal_stats_t> (*insert)(void* index, keys_view_t keys, rows_view_t rows, insert_config_t config,
                                          executor_t& executor, progress_ticks_t& ticks) noexcept;

    // Refining
    expected<refine_report_t> (*refine)(void* index, refine_config_t config, executor_t& executor,
                                        progress_ticks_t& ticks) noexcept;

    /** Moves every node of each of @p donors in, nulling each slot whose index it consumed. */
    expected<refine_report_t> (*absorb)(void* index, std::span<void*> donors, refine_config_t config,
                                        executor_t& executor, progress_ticks_t& ticks) noexcept;
    expected<dense::compact_result_t> (*compact)(void* index, std::span<void* const> sources,
                                                 dense::compact_config_t config, executor_t& executor,
                                                 progress_ticks_t& ticks) noexcept;

    /** Moves every node into one graph-local part per receiver of @p receivers, seeded by the
     *  config's seed. */
    expected<refine_report_t> (*split)(void* index, variants_view_t receivers, split_config_t config,
                                       executor_t& executor, progress_ticks_t& ticks) noexcept;
    expected<reachability_t> (*health)(void* index, executor_t& executor) noexcept;

    // Refinements

    /** Opens a refinement absorbing @p donors first, or splitting into @p receivers first. */
    expected<void*> (*refinement_init)(void* index, variants_view_t donors, variants_view_t receivers,
                                       split_config_t config, executor_t& executor) noexcept;
    status_t (*refinement_next)(void* index, void* refinement, executor_t& executor, progress_ticks_t& ticks) noexcept;
    refine_report_t (*refinement_running)(void const* refinement) noexcept;

    /** Whether the refinement consumed its donor at @p donor. */
    bool (*refinement_consumed)(void const* refinement, std::size_t donor) noexcept;
    expected<refine_report_t> (*refinement_finish)(void* index, void* refinement, executor_t& executor) noexcept;
    void (*refinement_free)(void* refinement) noexcept;

    // Lookups
    status_t (*count)(void* index, std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) noexcept;

    /** Reads up to @p wanted entries of @p keys into @p outputs, cast into their scalar where the
     *  index lives, on the device for a GPU one. */
    status_t (*get)(void* index, std::span<key_t const> keys, rows_span_t outputs, std::span<std::size_t> entries,
                    std::size_t wanted, executor_t& executor) noexcept;

    // Walks

    /** Ranks the entries nearest each query, narrowed by @p predicate and seeded by @p hints on the
     *  host. */
    expected<traversal_stats_t> (*search)(void* index, rows_view_t queries, ranked_hints_view_t hints,
                                          ranked_matches_span_t matches, search_config_t config, predicate_t predicate,
                                          executor_t& executor) noexcept;

    /** Keeps the largest dot products of each left row, up to the tape's stride, packed into @p
     *  products from its first offset, and on the host the pairs each row scored into @p scored. */
    expected<traversal_stats_t> (*multiply)(void* index, rows_view_t left, products_tape_t products,
                                            std::span<std::size_t> scored, search_config_t config,
                                            predicate_t predicate, executor_t& executor) noexcept;

    /** Late interaction of each query the offsets cut: up to @p wanted keys each, every scored one
     *  for zero, packed into @p matches from its first offset, and the keys each query scored into
     *  @p scored on the host; a GPU index walks uniform queries alone. */
    expected<traversal_stats_t> (*maxsim)(void* index, maxsim_queries_view_t queries, products_tape_t matches,
                                          std::span<std::size_t> scored, std::size_t wanted, search_config_t config,
                                          predicate_t predicate, executor_t& executor) noexcept;
    expected<traversal_stats_t> (*attend)(void* index, rows_view_t queries, attention_arguments_t attention,
                                          search_config_t config, predicate_t predicate, executor_t& executor) noexcept;

    // Exploration

    /** Opens a walk over every entry as stored, or breadth-first from @p starts. */
    expected<void*> (*cursor_init)(void* index, std::span<key_t const> starts) noexcept;

    /** Pulls the next entries' keys into @p keys, gathering their links into @p links where
     *  named. */
    expected<std::size_t> (*cursor_next)(void* index, void* cursor, std::span<key_t> keys, links_outputs_t* links,
                                         executor_t& executor) noexcept;
    void (*cursor_free)(void* cursor) noexcept;

    /** Draws up to the outputs' length around the focus keys, beside and placed among the shown
     *  ones, never scanning; only a RAM index samples yet. */
    expected<sample_result_t> (*sample)(void* index, sample_inputs<key_t> inputs, sample_outputs<key_t> outputs,
                                        sample_config_t config, executor_t& executor) noexcept;

    // Persistence
    status_t (*save)(void* index, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*save_buffer)(void* index, bytes_span_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*load)(void* index, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*load_buffer)(void* index, checkpoint_source_t source, executor_t& executor,
                            progress_ticks_t& ticks) noexcept;
};

/** The variants one class unit compiles for one residency: @c find answers the verb table of a
 *  dtype, or null where the class picks nothing its parent does not, so a ladder falls to the
 *  next rung of its parent. */
template <vendored_residency_t residency_, auto class_>
struct dense_class_tables {
    static dense_verbs<residency_> const* find(nk_dtype_t dtype) noexcept;
};

/** One picked variant as the dispatch unit holds it for one call: its residency's verbs, and the
 *  index they serve. */
template <vendored_residency_t residency_>
struct punned_dense_index {
    using executor_t = typename dense_verbs<residency_>::executor_t;

    dense_verbs<residency_> const& verbs;
    void* index;
};

/** The executor a null handle runs a device variant's work on: streams or queues of its own on its
 *  devices, one per distinct device every shard there shares, never a legacy default stream, which
 *  would wait on every blocking stream of the process. */
template <vendored_residency_t residency_>
    requires is_gpu_executor<typename punned_dense_index<residency_>::executor_t>
usearch_executor_t call_executor_for(punned_dense_index<residency_> const& punned,
                                     usearch_executor_handle_t& call_executor, usearch_outcome_t* outcome) noexcept {
    std::size_t const shards = punned.verbs.shards_count(punned.index);
    growing_buffer<int> devices {global_allocator};
    if (failed(devices.reserve_exactly(shards)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), nullptr;
    for (std::size_t shard = 0; shard != shards; ++shard) {
        int const device = static_cast<int>(punned.verbs.residency(punned.index, shard).device);
        bool opened = false;
        for (int const earlier : devices) opened |= earlier == device;
        if (!opened) devices.push_back_reserved(device);
    }
    using executor_t = typename punned_dense_index<residency_>::executor_t;
    auto [built, status] = executor_t::spawn(driver_of(gpu_drivers(), std::type_identity<executor_t> {}), devices);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor.emplace<gpu_executor_t>(std::in_place_type<executor_t>, std::move(built));
    return &call_executor;
}

#pragma endregion Variant Tables
