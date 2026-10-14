/**
 *  @file src/dense_specializations.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What one dense arm is spelled with and compiled through: the GPU classes, the residency
 *      aliases, and the @c specialized_index view every class unit instantiates.
 *
 *  Every class unit includes this to instantiate its arms; @c dense_index.cuh adds the variant
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

/** How many storage dtypes the arms cover, from the front of the ladder bf16, u1, i8, e4m3, f16,
 *  u8, e5m2, i4, u4, e3m2, e2m3. CMake passes it, so only builds that bypass CMake take this
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

/** Lowers one GPU class onto NumKong's GPU capabilities. On NVIDIA, as
 *  @c nk_cuda_capabilities_detected reports them, the Ampere & Ada bits accumulate, while Hopper,
 *  Blackwell & RTX Blackwell each run only their own kits, and Rubin rides the Blackwell kits until
 *  its own land. AMD and Apple generations accumulate over their vendor's floor. */
constexpr nk_capability_t numkong_capabilities_of(gpu_class_t gpu_class) noexcept {
    nk_capability_t const ampere = nk_cap_cuda_k | nk_cap_ampere_k, ada = ampere | nk_cap_ada_k;
    nk_capability_t const cdna3 = nk_cap_rocm_k | nk_cap_cdna3_k, cdna4 = cdna3 | nk_cap_cdna4_k;
    nk_capability_t const metal9 = nk_cap_metal_k | nk_cap_apple9_k;
    switch (gpu_class) {
    case gpu_class_t::nvidia_ada_k: return ada;
    case gpu_class_t::nvidia_hopper_k: return ada | nk_cap_hopper_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_rubin_k: return ada | nk_cap_blackwell_k;
    case gpu_class_t::nvidia_blackwellrtx_k: return ada | nk_cap_blackwellrtx_k;
    case gpu_class_t::amd_cdna3_k: return cdna3;
    case gpu_class_t::amd_cdna4_k: return cdna4;
    case gpu_class_t::amd_cdna5_k: return cdna4 | nk_cap_cdna5_k;
    case gpu_class_t::apple_metal7_k: return nk_cap_metal_k;
    case gpu_class_t::apple_metal9_k: return metal9;
    case gpu_class_t::apple_metal10_k: return metal9 | nk_cap_apple10_k;
    case gpu_class_t::nvidia_ampere_k: return ampere;
    case gpu_class_t::unknown_k: return nk_cap_serial_k;
    }
    return nk_cap_serial_k;
}

#pragma endregion Scaffolding

#pragma region Specialized Index

/** One RAM alternative: the CPU class alone names the rung, lowering onto its NumKong mask, its waiter,
 *  its atomic reference and its cache hints. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using ram_indexes = us::dense::ram_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(cpu_class_),
                                           waiter_of<cpu_class_>, atomic_ref_for_cpu_class<cpu_class_>::template type,
                                           typename cache_hints_for_cpu_class<cpu_class_>::type>;

/** One NVMe alternative: scoring compiles per CPU class, while the metric comes from the parts'
 *  headers and the transport is the executor's runtime choice. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using nvme_indexes = us::dense::nvme_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(cpu_class_)>;

/** One NVIDIA alternative: the class names the rung and its capability ceiling; its unit holds the
 *  kernels. */
#if USEARCH_ARCH_CUDA_
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_nvidia_indexes =
    us::dense::vram_nvidia_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(gpu_class_)>;
#endif

/** One AMD alternative: the class names the rung and its capability ceiling; its unit holds the
 *  code object. */
#if USEARCH_ARCH_ROCM_
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_amd_indexes =
    us::dense::vram_amd_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(gpu_class_)>;
#endif

/** One Apple alternative, serving every Apple class until a family's kernels differ. */
#if USEARCH_ARCH_METAL_
template <nk_dtype_t dtype_>
using vram_apple_indexes = us::dense::vram_apple_indexes<us::utilities::uuid_t, dtype_>;
#endif

/** The C value lookup lifted onto the index's key vocabulary: one callback per scored batch, a NULL row
 *  answering an empty view that refuses the call. */
struct value_lookup_t {
    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

    template <typename scalar_type_>
    void operator()(std::span<us::utilities::uuid_t const> keys,
                    std::span<nk::vector_view<scalar_type_>> rows) const noexcept {
        std::array<usearch_key_t, 4> batch;
        std::array<void const*, 4> found;
        for (std::size_t lane = 0; lane != keys.size(); ++lane) batch[lane] = std::bit_cast<usearch_key_t>(keys[lane]);
        callback(state, batch.data(), keys.size(), scalar, dimensions, found.data());
        for (std::size_t lane = 0; lane != keys.size(); ++lane)
            rows[lane] = found[lane]
                             ? nk::vector_view<scalar_type_> {static_cast<scalar_type_ const*>(found[lane]), dimensions}
                             : nk::vector_view<scalar_type_> {};
    }
};

/** The C value lookup called on the device, one scored key at a time inside a GPU walk. */
struct device_value_lookup_t {

    /** Names the lookup to the NVIDIA attend kernels the class units instantiate over it. */
    inline static constexpr u32_t device_lookup_k = 1;

    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

#if defined(__CUDACC__)
    __device__ void const* operator()(us::utilities::uuid_t key) const noexcept {
        usearch_key_t const scored = std::bit_cast<usearch_key_t>(key);
        void const* row = nullptr;
        callback(state, &scored, 1, scalar, dimensions, &row);
        return row;
    }
#endif
};

#if USEARCH_ARCH_CUDA_

/** What each shard of a VRAM attend resolves rows through: the C lookup behind the shard's keys. */
using nvidia_attend_lookup_t = us::dense::cuda_keyed_lookup<us::utilities::uuid_t, device_value_lookup_t>;

template <>
struct us::dense::cuda_value_lookup_of<nvidia_attend_lookup_t::device_lookup_k> {
    using type = nvidia_attend_lookup_t;
};
#endif

/** One arm's verbs, fixed to the C layer's callback types and wrapped as non-template members of a
 *  view over the index: an explicit instantiation compiles exactly these bodies in the arm's unit,
 *  and an @c extern one keeps every arm body out of the dispatch unit. The members are defined
 *  outside the class because in-class ones would be inline, which an explicit instantiation
 *  declaration does not restrain. */
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
    using health_result_t = typename index_type_::health_result_t;
    using rows_view_t = us::dense::rows_view_t;
    using rows_span_t = us::dense::rows_span_t;
    using keys_view_t = nk::vector_view<key_t>;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;

    /** What an attend resolves value rows through: a host callback per batch on RAM, a device call per key on GPU. */
    using attend_lookup_t = std::conditional_t<is_vram_index<index_type_>, device_value_lookup_t, value_lookup_t>;

    index_type_& index;

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

    /** Moves every row of @p donor into the index, linked in across, and leaves @p donor empty. */
    expected<refine_result_t> absorb(index_type_& donor, executor_t& executor) noexcept;
    expected<health_result_t> health(executor_t& executor) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     dummy_predicate_t) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     key_predicate_t& predicate) noexcept
        requires is_ram_index<index_type_>;
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys,
                                       scores_span_t products, counts_span_t counts) noexcept;
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, keys_span_t keys, scores_span_t scores,
                                     counts_span_t counts) noexcept;
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup,
                                     f32_t score_scale, rows_span_t outputs,
                                     std::span<us::dense::softmax_state_t> carry, executor_t& executor,
                                     search_config_t config) noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t save(bytes_span_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(bytes_view_t buffer, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** The C layer's save config: the path's suffix picks the layout, the residency the contract -
     *  device storage cannot be settlement-probed, so a GPU handle certifies. */
    static save_config_t save_config_of(std::string_view path) noexcept;
};

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
    auto [plan, status] = index.plan_insert(count, executor, config);
    if (failed(status)) return {{}, status};
    for (std::size_t landed = 0; landed != count;) {
        std::size_t const taking = min_of(chunk, count - landed);
        if constexpr (is_ram_index<index_type_>)
            plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking));
        else if (failed(plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking))))
            break;
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
    index_type_& donor, executor_t& executor) noexcept {
    return index.absorb(donor, executor);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::health_result_t> specialized_index<index_type_>::health(
    executor_t& executor) noexcept {
    return index.health(executor);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, dummy_predicate_t) noexcept {
    return index.search(queries, executor, config, keys, distances, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, key_predicate_t& predicate) noexcept
    requires is_ram_index<index_type_>
{
    return index.search_if(queries, executor, config, keys, distances, counts, predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t products,
    counts_span_t counts) noexcept {
    return index.multiply(left, executor, config, keys, products, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    rows_view_t queries, std::size_t rows_per_query, executor_t& executor, search_config_t config, keys_span_t keys,
    scores_span_t scores, counts_span_t counts) noexcept {
    return index.maxsim(queries, rows_per_query, executor, config, keys, scores, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<us::dense::softmax_state_t> carry, executor_t& executor, search_config_t config) noexcept {
    return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
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
save_config_t specialized_index<index_type_>::save_config_of(std::string_view path) noexcept {
    save_config_t config;
    config.layout = path.ends_with(".json") ? checkpoint_layout_t::distributed_k : checkpoint_layout_t::monolith_k;
    if constexpr (is_vram_index<index_type_>) config.synchronization = synchronization_t::immutable_k;
    return config;
}

/** One NVMe arm's verbs: the read verbs alone, since a view of parts on storage never mutates,
 *  defined outside the class for the same reason as the primary's. */
template <typename index_type_>
    requires is_nvme_index<index_type_>
struct specialized_index<index_type_> {
    using key_t = typename index_type_::key_t;
    using executor_t = typename index_type_::executor_t;
    using search_config_t = typename index_type_::search_config_t;
    using search_result_t = typename index_type_::search_result_t;
    using health_result_t = typename index_type_::health_result_t;
    using rows_view_t = us::dense::rows_view_t;
    using rows_span_t = us::dense::rows_span_t;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;
    using attend_lookup_t = value_lookup_t;

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
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     dummy_predicate_t) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     key_predicate_t& predicate) noexcept;
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys,
                                       scores_span_t products, counts_span_t counts) noexcept;
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, keys_span_t keys, scores_span_t scores,
                                     counts_span_t counts) noexcept;
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup,
                                     f32_t score_scale, rows_span_t outputs,
                                     std::span<us::dense::softmax_state_t> carry, executor_t& executor,
                                     search_config_t config) noexcept;
};

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
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, dummy_predicate_t) noexcept {
    return index.search(queries, executor, config, keys, distances, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, key_predicate_t& predicate) noexcept {
    return index.search_if(queries, executor, config, keys, distances, counts, predicate);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t products,
    counts_span_t counts) noexcept {
    return index.multiply(left, executor, config, keys, products, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    rows_view_t queries, std::size_t rows_per_query, executor_t& executor, search_config_t config, keys_span_t keys,
    scores_span_t scores, counts_span_t counts) noexcept {
    return index.maxsim(queries, rows_per_query, executor, config, keys, scores, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<us::dense::softmax_state_t> carry, executor_t& executor, search_config_t config) noexcept {
    return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
}

template <typename index_type_>
specialized_index(index_type_&) -> specialized_index<index_type_>;

#pragma endregion Specialized Index
