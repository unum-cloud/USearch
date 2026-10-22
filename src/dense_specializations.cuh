/**
 *  @file src/dense_specializations.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What one dense variant is spelled with and compiled through: the GPU classes, the
 *      residency aliases, and the @c specialized_index view every class unit instantiates.
 *
 *  Every class unit includes this to instantiate its variants; @c dense_index.cuh adds the variant
 *  grid, the handle and the exported C entry points on top of it. The scalar tables, the CPU
 *  classes and the call scaffolding every modality shares come from @c lib.hpp.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <array>       // `std::array` - the value lookup's batch
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <span>        // `std::span` - the search verbs' landing matrices
#include <string_view> // `std::string_view` - kernel tokens & checkpoint paths

#include "usearch/dense_index.h"
#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/dense/ram_indexes.hpp"
#include "usearch/dense/ram_compaction.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/dense/vram_nvidia_indexes.hpp"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/dense/vram_amd_indexes.hpp"
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/dense/vram_apple_indexes.hpp"
#endif

#include "lib.hpp"

/** How many storage dtypes the variants cover, from the front of the ladder bf16, u1, i8, e4m3,
 *  f16, u8, e5m2, i4, u4, e3m2, e2m3. CMake passes it, so only builds that bypass CMake take this
 *  default. */
#if !defined(USEARCH_DENSE_DTYPES)
#define USEARCH_DENSE_DTYPES 4
#endif
#if USEARCH_DENSE_DTYPES < 1 || USEARCH_DENSE_DTYPES > 11
#error "USEARCH_DENSE_DTYPES must be an integer from 1 to 11"
#endif

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

/** Codenames of the GPU classes a VRAM alternative dispatches onto. */
struct gpu_class_name_t {
    gpu_class_t gpu_class;
    char const* name;
};
inline constexpr gpu_class_name_t gpu_class_names_k[] = {
    {gpu_class_t::nvidia_ampere_k, "nvidia-ampere"},
    {gpu_class_t::nvidia_ada_k, "nvidia-ada"},
    {gpu_class_t::nvidia_hopper_k, "nvidia-hopper"},
    {gpu_class_t::nvidia_blackwell_k, "nvidia-blackwell"},
    {gpu_class_t::nvidia_blackwellrtx_k, "nvidia-blackwellrtx"},
    {gpu_class_t::nvidia_rubin_k, "nvidia-rubin"},
    {gpu_class_t::amd_cdna3_k, "amd-cdna3"},
    {gpu_class_t::amd_cdna4_k, "amd-cdna4"},
    {gpu_class_t::amd_cdna5_k, "amd-cdna5"},
    {gpu_class_t::apple_metal7_k, "apple-metal7"},
    {gpu_class_t::apple_metal9_k, "apple-metal9"},
    {gpu_class_t::apple_metal10_k, "apple-metal10"},
};

#pragma endregion Scaffolding

#pragma region Specialized Index

/** The kits one CPU class runs a dtype on, each the pick of its own ladder under the class's
 *  NumKong mask. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using quads_kit_of =
    typename dense::quads_ladder<typename nk::type_for<dtype_>::type, numkong_capabilities_of(cpu_class_)>::kit_t;
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using tiles_kit_of =
    typename dense::tiles_ladder<typename nk::type_for<dtype_>::type, numkong_capabilities_of(cpu_class_)>::kit_t;
template <cpu_class_t cpu_class_>
using elementwise_kit_of = typename dense::elementwise_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using rmscast_kit_of =
    typename dense::rmscast_ladder<typename nk::type_for<dtype_>::type, numkong_capabilities_of(cpu_class_)>::kit_t;
template <cpu_class_t cpu_class_>
using visits_kit_of = typename dense::visits_hash_set_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;

/** One RAM alternative: the CPU class alone names the rung, lowering onto the kits its NumKong mask
 *  picks, its waiter, its atomic reference and its cache hints. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using ram_indexes =
    dense::ram_indexes<us::utilities::uuid_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                       elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>,
                       waiter_of<cpu_class_>, atomic_ref_for_cpu_class<cpu_class_>::template type,
                       typename cache_hints_for_cpu_class<cpu_class_>::type>;

/** One NVMe alternative: scoring compiles per CPU class, while the metric comes from the parts'
 *  headers and the transport is the executor's runtime choice. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using nvme_indexes =
    dense::nvme_indexes<us::utilities::uuid_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                        elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>>;

/** One NVIDIA class's fatbin of one dtype's kernels; the class's host unit defines @c bytes beside
 *  the bytes it embeds. */
#if USEARCH_ARCH_CUDA_
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct cuda_fatbin {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static bytes_view_t bytes() noexcept;
};

/** One NVIDIA alternative: the class names the rung and its capability ceiling, and with the dtype
 *  the fatbin its kernels sit in. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_nvidia_indexes = dense::vram_nvidia_indexes<us::utilities::uuid_t, dtype_, cuda_fatbin<gpu_class_, dtype_>,
                                                       fu::preferred_cache_hints_t, fu::standard_yield_t>;
#endif

/** One AMD class's code object of one dtype's kernels, an offload bundle of its listed processors;
 *  the class's host unit defines @c bytes beside the bytes it embeds. */
#if USEARCH_ARCH_ROCM_
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct hip_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static bytes_view_t bytes() noexcept;
};

/** One AMD alternative: the class names the rung and its capability ceiling, and with the dtype the
 *  code object its kernels sit in. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_amd_indexes = dense::vram_amd_indexes<us::utilities::uuid_t, dtype_, hip_code_object<gpu_class_, dtype_>,
                                                 fu::preferred_cache_hints_t, fu::standard_yield_t>;
#endif

/** One Apple alternative: the class names the rung and the capability ceiling its kernels compile
 *  for at the first @c reserve. */
#if USEARCH_ARCH_METAL_
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_apple_indexes =
    dense::vram_apple_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(gpu_class_), fu::standard_yield_t>;
#endif

/** One traversal context per executor thread, carved after the call's one rewind from that
 *  thread's arena, which keeps its widest chunk between calls; the array sits in the first arena.
 *  Empty off RAM, where the towers keep their own. */
template <typename indexes_type_>
struct dense_scratch {
    explicit dense_scratch(typename indexes_type_::executor_t&) noexcept {}
    status_t prepare(typename indexes_type_::executor_t&) noexcept { return status_t::success_k; }
};

template <typename indexes_type_>
    requires is_ram_index<indexes_type_>
struct dense_scratch<indexes_type_> {
    using traversal_context_t = typename indexes_type_::traversal_context_t;

    growing_buffer<traversal_context_t, thread_allocator_t> traversals;

    explicit dense_scratch(forkunion_executor_t& executor) noexcept
        : traversals(thread_allocator_t {executor.arena(0)}) {}

    /** Rewinds every arena, keeping the high-water mark of earlier calls, and carves one context
     *  per thread; the verb sizes them. */
    status_t prepare(forkunion_executor_t& executor) noexcept {
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        if (failed(traversals.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            traversals.push_back_reserved(traversal_context_t {thread_allocator_t {executor.arena(thread)}});
        return status_t::success_k;
    }
};

/** One key a MaxSim query kept, waiting in its thread's run to scatter into the call's tape. */
struct maxsim_match_t {
    std::size_t query;
    usearch_key_t key;
    usearch_distance_t score;
};

static_assert(sizeof(maxsim_match_t) == 32, "A kept key is a 32-byte hot entry");

/** Opens one empty MaxSim run per thread that ranks @p indexes_type_ 's queries into @p runs: one
 *  per executor thread over that thread's arena, or one over the call's @p staging for a device
 *  index, whose host reduction ranks on the calling thread. */
template <typename indexes_type_>
status_t open_maxsim_runs(growing_buffer<growing_buffer<maxsim_match_t, thread_allocator_t>, thread_allocator_t>& runs,
                          typename indexes_type_::executor_t& executor, thread_allocator_t staging) noexcept {
    using run_t = growing_buffer<maxsim_match_t, thread_allocator_t>;
    if constexpr (is_vram_index<indexes_type_>) {
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

/** Searches @p index under an expensive @p predicate through a family of per-thread predicates
 *  gathering what the traversal @p config names hands them; the engine opens the family and
 *  reserves it for the widest round it settles. */
template <typename index_type_>
expected<typename index_type_::search_result_t> search_with_expensive_predicate(
    index_type_& index, dense::rows_view_t queries, typename index_type_::executor_t& executor,
    dense_scratch<index_type_>& scratch, typename index_type_::search_config_t config,
    nk::matrix_span<typename index_type_::key_t> keys, nk::matrix_span<f64_t> distances, std::span<std::size_t> counts,
    usearch_predicate_t predicate) noexcept {
    if (config.traversal == traversal_t::scan_k) {
        expensive_predicates<typename index_type_::scan_payload_t> scanned {predicate};
        if constexpr (is_ram_index<index_type_>)
            return index.search_if(queries, executor, to_writable_span(scratch.traversals), config, keys, distances,
                                   counts, scanned);
        else return index.search_if(queries, executor, config, keys, distances, counts, scanned);
    }
    expensive_predicates<typename index_type_::walk_payload_t> walked {predicate};
    if constexpr (is_ram_index<index_type_>)
        return index.search_if(queries, executor, to_writable_span(scratch.traversals), config, keys, distances, counts,
                               walked);
    else return index.search_if(queries, executor, config, keys, distances, counts, walked);
}

/** One variant's verbs, fixed to the C layer's callback types and wrapped as non-template members
 *  of a view over the index: an explicit instantiation compiles exactly these bodies in the
 *  variant's unit, and an @c extern one keeps every variant body out of the dispatch unit. The
 *  members are defined outside the class because in-class ones would be inline, which an explicit
 *  instantiation declaration does not restrain. */
template <typename index_type_>
struct specialized_index {
    using key_t = typename index_type_::key_t;
    using executor_t = typename index_type_::executor_t;
    using reserve_capacity_t = typename index_type_::reserve_capacity_t;
    using insert_config_t = typename index_type_::insert_config_t;
    using insert_result_t = typename index_type_::insert_result_t;
    using search_config_t = typename index_type_::search_config_t;
    using search_result_t = typename index_type_::search_result_t;
    using refine_config_t = typename index_type_::refine_config_t;
    using refine_result_t = typename index_type_::refine_result_t;
    using refine_plan_t = typename index_type_::refine_plan_t;
    using health_result_t = typename index_type_::health_result_t;
    using rows_view_t = dense::rows_view_t;
    using rows_span_t = dense::rows_span_t;
    using keys_view_t = nk::vector_view<key_t>;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;

    /** What the C layer carved for one call: the contexts on RAM, nothing elsewhere. */
    using scratch_t = dense_scratch<index_type_>;

    index_type_& index;

    void bound_shards(std::size_t rows) noexcept;
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept
        requires is_ram_index<index_type_>;
    status_t emplace(int device) noexcept
        requires is_vram_index<index_type_>;

    /** The device @p shard sits on; the first one's is what the capability string names. */
    int device(std::size_t shard) const noexcept
        requires is_vram_index<index_type_>;
    std::size_t shards_count() const noexcept;
    std::size_t dimensions() const noexcept;
    std::size_t connectivity() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t memory_usage() const noexcept;
    expected<std::size_t> storage_requirements() const noexcept;
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept;
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) noexcept;
    status_t reserve(reserve_capacity_t config) noexcept;
    status_t clear() noexcept;

    /** Lands @p rows under @p keys through one plan, a step per chunk, and publishes them all at once. */
    expected<insert_result_t> add(keys_view_t keys, rows_view_t rows, progress_ticks_t& ticks, executor_t& executor,
                                  insert_config_t config) noexcept;
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config, progress_ticks_t& ticks) noexcept;

    /** Moves every row of each of @p donors into the index, one donor after another, nulling each
     *  once consumed, then links them in across with one refine that @p ticks reports. */
    expected<refine_result_t> absorb(std::span<index_type_*> donors, executor_t& executor, refine_config_t config,
                                     progress_ticks_t& ticks) noexcept;

    /** Opens a refinement the caller holds, landing @p donors first when any are named. */
    expected<refine_plan_t> plan_absorb(std::span<index_type_*> donors, executor_t& executor,
                                        refine_config_t config) noexcept;

    /** Compacts @p sources into the index under @p config, in place when the index is one of them,
     *  the closing re-selection reported through @p ticks; a GPU index answers @c unsupported_k. */
    expected<dense::compact_result_t> compact(std::span<index_type_* const> sources, executor_t& executor,
                                              dense::compact_config_t config, progress_ticks_t& ticks) noexcept
        requires is_ram_index<index_type_>;
    expected<dense::compact_result_t> compact(std::span<index_type_* const> sources, executor_t& executor,
                                              dense::compact_config_t config, progress_ticks_t& ticks) noexcept
        requires is_vram_index<index_type_>;
    status_t run(refine_plan_t& plan, executor_t& executor, scratch_t& scratch, progress_ticks_t& ticks) noexcept;
    expected<refine_result_t> finish(refine_plan_t& plan, executor_t& executor, scratch_t& scratch) noexcept;
    expected<health_result_t> health(executor_t& executor) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, scratch_t& scratch,
                                     search_config_t config, keys_span_t keys, scores_span_t distances,
                                     counts_span_t counts, dummy_predicate_t) noexcept;

    /** Narrows a walk by a host predicate, which a device walk cannot take, so a GPU index answers
     *  @c unsupported_k. */
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, scratch_t& scratch,
                                     search_config_t config, keys_span_t keys, scores_span_t distances,
                                     counts_span_t counts, predicate_t& predicate) noexcept;

    /** Narrows a walk or a scan by an expensive host @p predicate, asked once per round each thread
     *  settles; a GPU index answers @c unsupported_k. */
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, scratch_t& scratch,
                                     search_config_t config, keys_span_t keys, scores_span_t distances,
                                     counts_span_t counts, usearch_predicate_t predicate) noexcept;
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, scratch_t& scratch,
                                       search_config_t config, keys_span_t keys, scores_span_t products,
                                       counts_span_t counts) noexcept;

    /** Narrows the written pairs by a host predicate, which a device walk cannot take, so a GPU
     *  index answers @c unsupported_k. */
    expected<search_result_t> multiply_if(rows_view_t left, executor_t& executor, scratch_t& scratch,
                                          search_config_t config, keys_span_t keys, scores_span_t products,
                                          counts_span_t counts, predicate_t& predicate) noexcept;

    /** Late interaction of the queries @p query_offsets cut @p queries into, each keeping @p wanted
     *  keys, or every key it scored when zero, in @p landing, and counting the keys it scored in
     *  @p scored; a GPU index walks uniform queries alone, answering @c unsupported_k for a scan,
     *  ragged offsets or a zero @p wanted. */
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     executor_t& executor, scratch_t& scratch, search_config_t config,
                                     std::size_t wanted, counts_span_t scored, maxsim_landing_t landing) noexcept;

    /** Narrows the nominated keys by a host predicate, which a device walk cannot take, so a GPU
     *  index answers @c unsupported_k. */
    expected<search_result_t> maxsim_if(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                        executor_t& executor, scratch_t& scratch, search_config_t config,
                                        std::size_t wanted, counts_span_t scored, maxsim_landing_t landing,
                                        predicate_t& predicate) noexcept;
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                                     executor_t& executor, scratch_t& scratch, search_config_t config) noexcept;

    /** Narrows the folded keys by a host predicate, which a device walk cannot take, so a GPU index
     *  answers @c unsupported_k. */
    expected<search_result_t> attend_if(rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                                        f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                                        executor_t& executor, scratch_t& scratch, search_config_t config,
                                        predicate_t& predicate) noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t save(bytes_span_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(bytes_view_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Opens a cursor over every row into @p walk, which holds any variant's cursor; a GPU index
     *  reads its rows back by key or search and answers @c unsupported_k. */
    template <typename walk_type_>
    status_t open(walk_type_& walk) noexcept
        requires is_ram_index<index_type_>;
    template <typename walk_type_>
    status_t open(walk_type_& walk) noexcept
        requires is_vram_index<index_type_>;

    /** Reads the next rows of @p walk, a cursor @ref open placed, into @p keys and @p rows,
     *  answering how many landed. */
    template <typename cursor_type_>
    expected<std::size_t> read(cursor_type_& walk, std::span<key_t> keys, rows_span_t rows,
                               executor_t& executor) noexcept;

    /** The C layer's save config: the path's suffix picks the layout, the residency the contract -
     *  device storage cannot be settlement-probed, so a GPU handle certifies. */
    static save_config_t save_config_of(std::string_view path) noexcept;
};

template <typename index_type_>
void specialized_index<index_type_>::bound_shards(std::size_t rows) noexcept {
    index.bound_shards(rows);
}
template <typename index_type_>
status_t specialized_index<index_type_>::emplace(fu_memory_domain_id_t memory_domain) noexcept
    requires is_ram_index<index_type_>
{
    return index.emplace(memory_domain);
}
template <typename index_type_>
status_t specialized_index<index_type_>::emplace(int device) noexcept
    requires is_vram_index<index_type_>
{
    // Init names devices and no executor yet, so a stream opened for the placement alone names
    // the device.
    auto [executor, status] = executor_t::spawn(std::span<int const> {&device, 1});
    return failed(status) ? status : index.emplace(executor.stream(0));
}
template <typename index_type_>
int specialized_index<index_type_>::device(std::size_t shard) const noexcept
    requires is_vram_index<index_type_>
{
    return index.shard(shard).device_specs().ordinal;
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::shards_count() const noexcept {
    return index.shards_count();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::dimensions() const noexcept {
    return index.dimensions();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::connectivity() const noexcept {
    return index.connectivity();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::size() const noexcept {
    return index.size();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::capacity() const noexcept {
    return index.capacity();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::memory_usage() const noexcept {
    return index.memory_usage();
}
template <typename index_type_>
expected<std::size_t> specialized_index<index_type_>::storage_requirements() const noexcept {
    return index.storage_requirements(save_config_of({}));
}
template <typename index_type_>
status_t specialized_index<index_type_>::count(std::span<key_t const> keys, std::span<std::size_t> entries,
                                               executor_t& executor) noexcept {
    return index.count(keys, entries, executor);
}
template <typename index_type_>
status_t specialized_index<index_type_>::get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted,
                                             std::span<std::size_t> entries, executor_t& executor) noexcept {
    return index.get(keys, vectors, wanted, entries, executor);
}
template <typename index_type_>
status_t specialized_index<index_type_>::reserve(reserve_capacity_t config) noexcept {
    return index.reserve(config).status;
}
template <typename index_type_>
status_t specialized_index<index_type_>::clear() noexcept {
    return index.clear();
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::insert_result_t> specialized_index<index_type_>::add(
    keys_view_t keys, rows_view_t rows, progress_ticks_t& ticks, executor_t& executor,
    insert_config_t config) noexcept {
    std::size_t const count = keys.size();
    // Only a handler splits the call, and one plan still publishes every slice at once.
    std::size_t const chunk = ticks.slice();
    scratch_t scratch {executor};
    if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
    auto [plan, status] = index.plan_insert(count, executor, config);
    if (failed(status)) return {{}, status};
    for (std::size_t landed = 0; landed != count;) {
        std::size_t const taking = min_of(chunk, count - landed);
        status_t stepped = status_t::success_k;
        if constexpr (is_ram_index<index_type_>)
            stepped = plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking),
                                  to_writable_span(scratch.traversals));
        else stepped = plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking));
        if (failed(stepped)) {
            // A refused RAM step dispensed nothing, so the dropped plan releases every claim.
            if constexpr (is_ram_index<index_type_>) return {{}, stepped};
            else break;
        }
        landed += taking;
        // Every row is claimed before the first report and none can be withdrawn, so a cancel
        // goes unheard.
        ticks(landed, count);
    }
    // After a failed device step, `finish` publishes nothing and reports the failure the plan kept.
    return plan.finish();
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::refine_result_t> specialized_index<index_type_>::refine(
    executor_t& executor, refine_config_t config, progress_ticks_t& ticks) noexcept {
    return index.refine(executor, config,
                        [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::refine_result_t> specialized_index<index_type_>::absorb(
    std::span<index_type_*> donors, executor_t& executor, refine_config_t config, progress_ticks_t& ticks) noexcept {
    return index.absorb(donors, executor, config,
                        [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
}
template <typename index_type_>
expected<dense::compact_result_t> specialized_index<index_type_>::compact(std::span<index_type_* const> sources,
                                                                          executor_t& executor,
                                                                          dense::compact_config_t config,
                                                                          progress_ticks_t& ticks) noexcept
    requires is_ram_index<index_type_>
{
    return dense::compact(sources, index, executor, config,
                          [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
}
template <typename index_type_>
expected<dense::compact_result_t> specialized_index<index_type_>::compact(std::span<index_type_* const>, executor_t&,
                                                                          dense::compact_config_t,
                                                                          progress_ticks_t&) noexcept
    requires is_vram_index<index_type_>
{
    return {{}, status_t::unsupported_k};
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::refine_plan_t> specialized_index<index_type_>::plan_absorb(
    std::span<index_type_*> donors, executor_t& executor, refine_config_t config) noexcept {
    return index.plan_absorb(donors, executor, config);
}
template <typename index_type_>
status_t specialized_index<index_type_>::run(refine_plan_t& plan, executor_t& executor, scratch_t& scratch,
                                             progress_ticks_t& ticks) noexcept {
    auto const report = [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); };
    if constexpr (is_ram_index<index_type_>) return plan.run(executor, to_writable_span(scratch.traversals), report);
    else return plan.run(executor, report);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::refine_result_t> specialized_index<index_type_>::finish(
    refine_plan_t& plan, executor_t& executor, scratch_t& scratch) noexcept {
    if constexpr (is_ram_index<index_type_>) return plan.finish(executor, to_writable_span(scratch.traversals));
    else return plan.finish(executor);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::health_result_t> specialized_index<index_type_>::health(
    executor_t& executor) noexcept {
    return index.health(executor);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t distances, counts_span_t counts, dummy_predicate_t) noexcept {
    if constexpr (is_ram_index<index_type_>)
        return index.search(queries, executor, to_writable_span(scratch.traversals), config, keys, distances, counts);
    else return index.search(queries, executor, config, keys, distances, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t distances, counts_span_t counts, predicate_t& predicate) noexcept {
    if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
    else
        return index.search_if(queries, executor, to_writable_span(scratch.traversals), config, keys, distances, counts,
                               predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t distances, counts_span_t counts, usearch_predicate_t predicate) noexcept {
    if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
    else
        return search_with_expensive_predicate(index, queries, executor, scratch, config, keys, distances, counts,
                                               predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    rows_view_t left, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t products, counts_span_t counts) noexcept {
    if constexpr (is_ram_index<index_type_>)
        return index.multiply(left, executor, to_writable_span(scratch.traversals), config, keys, products, counts);
    else return index.multiply(left, executor, config, keys, products, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply_if(
    rows_view_t left, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t products, counts_span_t counts, predicate_t& predicate) noexcept {
    if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
    else
        return index.multiply_if(left, executor, to_writable_span(scratch.traversals), config, keys, products, counts,
                                 predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    rows_view_t queries, std::span<std::size_t const> query_offsets, executor_t& executor, scratch_t& scratch,
    search_config_t config, std::size_t wanted, counts_span_t scored, maxsim_landing_t landing) noexcept {
    if constexpr (is_vram_index<index_type_>) {
        // The device reduction ranks into a rectangle on this thread, then lands in the one run.
        std::size_t const queries_count = query_offsets.size() - 1;
        std::size_t const rows_per_query = query_offsets[1] - query_offsets[0];
        for (std::size_t query = 0; query != queries_count; ++query)
            if (query_offsets[query + 1] - query_offsets[query] != rows_per_query) return {{}, status_t::unsupported_k};
        if (config.traversal != traversal_t::walk_k || !wanted) return {{}, status_t::unsupported_k};
        growing_buffer<key_t, thread_allocator_t> keys {landing.staging};
        growing_buffer<f64_t, thread_allocator_t> scores {landing.staging};
        growing_buffer<std::size_t, thread_allocator_t> counts {landing.staging};
        if (failed(keys.resize_uninitialized(queries_count * wanted)) ||
            failed(scores.resize_uninitialized(queries_count * wanted)) ||
            failed(counts.resize_uninitialized(queries_count)))
            return {{}, status_t::bad_alloc_k};
        expected<search_result_t> const reduced = index.maxsim(
            queries.slice(query_offsets.front(), query_offsets.back() - query_offsets.front()), rows_per_query,
            executor, config, keys_span_t {keys.data(), queries_count, wanted},
            scores_span_t {scores.data(), queries_count, wanted}, to_writable_span(counts), scored);
        for (std::size_t query = 0; reduced && query != queries_count; ++query)
            for (std::size_t rank = 0; rank != counts[query]; ++rank)
                if (failed(landing(0, query, keys[query * wanted + rank], scores[query * wanted + rank])))
                    return {reduced.value, status_t::bad_alloc_k};
        return reduced;
    }
    else
        return index.maxsim(queries, query_offsets, executor, to_writable_span(scratch.traversals), config, wanted,
                            scored, landing);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim_if(
    rows_view_t queries, std::span<std::size_t const> query_offsets, executor_t& executor, scratch_t& scratch,
    search_config_t config, std::size_t wanted, counts_span_t scored, maxsim_landing_t landing,
    predicate_t& predicate) noexcept {
    if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
    else
        return index.maxsim_if(queries, query_offsets, executor, to_writable_span(scratch.traversals), config, wanted,
                               scored, landing, predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<dense::softmax_state_t> carry, executor_t& executor, scratch_t& scratch,
    search_config_t config) noexcept {
    if constexpr (is_ram_index<index_type_>)
        return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor,
                            to_writable_span(scratch.traversals), config);
    else return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend_if(
    rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<dense::softmax_state_t> carry, executor_t& executor, scratch_t& scratch, search_config_t config,
    predicate_t& predicate) noexcept {
    if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
    else
        return index.attend_if(queries, head_count, value_lookup, score_scale, outputs, carry, executor,
                               to_writable_span(scratch.traversals), config, predicate);
}
template <typename index_type_>
status_t specialized_index<index_type_>::save(std::string_view path, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
    return index.save(path, save_config_of(path), executor, ticks);
}
template <typename index_type_>
status_t specialized_index<index_type_>::save(bytes_span_t buffer, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
    return index.save(buffer, save_config_of({}), executor, ticks);
}
template <typename index_type_>
status_t specialized_index<index_type_>::load(std::string_view path, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
    return index.load(path, load_config_t {}, executor, ticks);
}
template <typename index_type_>
status_t specialized_index<index_type_>::load(bytes_view_t buffer, executor_t& executor,
                                              progress_ticks_t& ticks) noexcept {
    return index.load(buffer, load_config_t {}, executor, ticks);
}
template <typename index_type_>
template <typename walk_type_>
status_t specialized_index<index_type_>::open(walk_type_& walk) noexcept
    requires is_ram_index<index_type_>
{
    walk.template emplace<typename index_type_::cursor_t>(index);
    return status_t::success_k;
}
template <typename index_type_>
template <typename walk_type_>
status_t specialized_index<index_type_>::open(walk_type_&) noexcept
    requires is_vram_index<index_type_>
{
    return status_t::unsupported_k;
}
template <typename index_type_>
template <typename cursor_type_>
expected<std::size_t> specialized_index<index_type_>::read(cursor_type_& walk, std::span<key_t> keys, rows_span_t rows,
                                                           executor_t&) noexcept {
    return walk.read(index, keys, rows);
}
template <typename index_type_>
save_config_t specialized_index<index_type_>::save_config_of(std::string_view path) noexcept {
    save_config_t config;
    config.layout = path.ends_with(".json") ? checkpoint_layout_t::distributed_k : checkpoint_layout_t::monolith_k;
    if constexpr (is_vram_index<index_type_>) config.synchronization = synchronization_t::immutable_k;
    return config;
}

/** One NVMe variant's verbs: the read verbs alone, since a view of parts on storage never mutates,
 *  defined outside the class for the same reason as the primary's. */
template <typename index_type_>
    requires is_nvme_index<index_type_>
struct specialized_index<index_type_> {
    using key_t = typename index_type_::key_t;
    using executor_t = typename index_type_::executor_t;
    using search_config_t = typename index_type_::search_config_t;
    using search_result_t = typename index_type_::search_result_t;
    using health_result_t = typename index_type_::health_result_t;
    using rows_view_t = dense::rows_view_t;
    using rows_span_t = dense::rows_span_t;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;

    /** What the C layer carved for one call: the contexts on RAM, nothing elsewhere. */
    using scratch_t = dense_scratch<index_type_>;

    index_type_& index;

    status_t view(std::string_view path, std::span<std::string_view const> paths) noexcept;
    std::size_t shards_count() const noexcept;
    std::size_t dimensions() const noexcept;
    std::size_t connectivity() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t memory_usage() const noexcept;
    expected<health_result_t> health(executor_t& executor) noexcept;
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept;
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, scratch_t& scratch,
                                     search_config_t config, keys_span_t keys, scores_span_t distances,
                                     counts_span_t counts, dummy_predicate_t) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, scratch_t& scratch,
                                     search_config_t config, keys_span_t keys, scores_span_t distances,
                                     counts_span_t counts, predicate_t& predicate) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, scratch_t& scratch,
                                     search_config_t config, keys_span_t keys, scores_span_t distances,
                                     counts_span_t counts, usearch_predicate_t predicate) noexcept;
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, scratch_t& scratch,
                                       search_config_t config, keys_span_t keys, scores_span_t products,
                                       counts_span_t counts) noexcept;
    expected<search_result_t> multiply_if(rows_view_t left, executor_t& executor, scratch_t& scratch,
                                          search_config_t config, keys_span_t keys, scores_span_t products,
                                          counts_span_t counts, predicate_t& predicate) noexcept;
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     executor_t& executor, scratch_t& scratch, search_config_t config,
                                     std::size_t wanted, counts_span_t scored, maxsim_landing_t landing) noexcept;
    expected<search_result_t> maxsim_if(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                        executor_t& executor, scratch_t& scratch, search_config_t config,
                                        std::size_t wanted, counts_span_t scored, maxsim_landing_t landing,
                                        predicate_t& predicate) noexcept;
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                                     executor_t& executor, scratch_t& scratch, search_config_t config) noexcept;
    expected<search_result_t> attend_if(rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                                        f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                                        executor_t& executor, scratch_t& scratch, search_config_t config,
                                        predicate_t& predicate) noexcept;

    /** Opens a cursor over every row into @p walk, which holds any variant's cursor. */
    template <typename walk_type_>
    status_t open(walk_type_& walk) noexcept;

    /** Reads the next rows of @p walk through the executor into @p keys and @p rows, answering how
     *  many landed. */
    template <typename cursor_type_>
    expected<std::size_t> read(cursor_type_& walk, std::span<key_t> keys, rows_span_t rows,
                               executor_t& executor) noexcept;

    /** A view already is its checkpoint, so saving it answers @c unsupported_k. */
    expected<std::size_t> storage_requirements() const noexcept { return {0, status_t::unsupported_k}; }
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    status_t save(bytes_span_t, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
};

template <typename index_type_>
    requires is_nvme_index<index_type_>
template <typename walk_type_>
status_t specialized_index<index_type_>::open(walk_type_& walk) noexcept {
    walk.template emplace<typename index_type_::cursor_t>(index);
    return status_t::success_k;
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
template <typename cursor_type_>
expected<std::size_t> specialized_index<index_type_>::read(cursor_type_& walk, std::span<key_t> keys, rows_span_t rows,
                                                           executor_t& executor) noexcept {
    auto const [scanned, status] = walk.read(index, keys, rows, executor);
    return {scanned.count, status};
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
status_t specialized_index<index_type_>::view(std::string_view path, std::span<std::string_view const> paths) noexcept {
    return index.view(path, paths);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::shards_count() const noexcept {
    return index.shards_count();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::dimensions() const noexcept {
    return index.dimensions();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::connectivity() const noexcept {
    return index.connectivity();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::size() const noexcept {
    return index.size();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::capacity() const noexcept {
    return index.capacity();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::memory_usage() const noexcept {
    return index.memory_usage();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::health_result_t> specialized_index<index_type_>::health(
    executor_t& executor) noexcept {
    return index.health(executor);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
status_t specialized_index<index_type_>::count(std::span<key_t const> keys, std::span<std::size_t> entries,
                                               executor_t& executor) noexcept {
    return index.count(keys, entries, executor);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
status_t specialized_index<index_type_>::get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted,
                                             std::span<std::size_t> entries, executor_t& executor) noexcept {
    return index.get(keys, vectors, wanted, entries, executor);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t distances, counts_span_t counts, dummy_predicate_t) noexcept {
    return index.search(queries, executor, config, keys, distances, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t distances, counts_span_t counts, predicate_t& predicate) noexcept {
    return index.search_if(queries, executor, config, keys, distances, counts, predicate);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t distances, counts_span_t counts, usearch_predicate_t predicate) noexcept {
    return search_with_expensive_predicate(index, queries, executor, scratch, config, keys, distances, counts,
                                           predicate);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    rows_view_t left, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t products, counts_span_t counts) noexcept {
    return index.multiply(left, executor, config, keys, products, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply_if(
    rows_view_t left, executor_t& executor, scratch_t& scratch, search_config_t config, keys_span_t keys,
    scores_span_t products, counts_span_t counts, predicate_t& predicate) noexcept {
    return index.multiply_if(left, executor, config, keys, products, counts, predicate);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    rows_view_t queries, std::span<std::size_t const> query_offsets, executor_t& executor, scratch_t&,
    search_config_t config, std::size_t wanted, counts_span_t scored, maxsim_landing_t landing) noexcept {
    return index.maxsim(queries, query_offsets, executor, config, wanted, scored, landing);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim_if(
    rows_view_t queries, std::span<std::size_t const> query_offsets, executor_t& executor, scratch_t&,
    search_config_t config, std::size_t wanted, counts_span_t scored, maxsim_landing_t landing,
    predicate_t& predicate) noexcept {
    return index.maxsim_if(queries, query_offsets, executor, config, wanted, scored, landing, predicate);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<dense::softmax_state_t> carry, executor_t& executor, scratch_t& scratch,
    search_config_t config) noexcept {
    return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend_if(
    rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<dense::softmax_state_t> carry, executor_t& executor, scratch_t& scratch, search_config_t config,
    predicate_t& predicate) noexcept {
    return index.attend_if(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config, predicate);
}

template <typename index_type_>
specialized_index(index_type_&) -> specialized_index<index_type_>;

/** The executor a null handle runs VRAM work on: streams or queues of its own on its devices, never
 *  a legacy default stream, which would wait on every blocking stream of the process. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
usearch_executor_t call_executor_for(multi_type_& variant, usearch_executor_handle_t& call_executor,
                                     usearch_outcome_t* outcome) noexcept {
    // One stream or queue per distinct device, which every shard there shares.
    specialized_index const shards {variant};
    growing_buffer<int> devices {global_allocator};
    if (failed(devices.reserve_exactly(shards.shards_count())))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), nullptr;
    for (std::size_t shard = 0; shard != shards.shards_count(); ++shard) {
        bool opened = false;
        for (int const earlier : devices) opened |= earlier == shards.device(shard);
        if (!opened) devices.push_back_reserved(shards.device(shard));
    }
    auto [built, status] = multi_type_::executor_t::spawn(devices);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor = gpu_executor_t {std::move(built)};
    return &call_executor;
}

/** The GPU ordinal @p variant computes on, or @c host_device_k. */
template <typename variant_type_>
int device_of(variant_type_& variant) noexcept {
    if constexpr (is_vram_index<variant_type_>) return specialized_index {variant}.device(0);
    else return host_device_k;
}

#pragma endregion Specialized Index
