/**
 *  @file src/dense_specializations.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The one place dense variants are enumerated: the residency aliases a class spells its
 *      variants with, the static bodies each variant's verb table points at, and the class tables a
 *      class unit instantiates once.
 *
 *  Only class units include it, each compiled with its class's instruction set, so every function
 *  it defines is @c static or a static member of a template over a variant's own types, and no body
 *  compiled under a class's flags escapes to the baseline dispatch unit through a merged copy.
 */
#pragma once

#include <array> // `std::array` - a class unit's variants
#include <new>   // `std::nothrow`

#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/dense/ram_compaction.hpp"
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

#include "dense_punned.cuh"

#pragma region Residencies

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

/** One RAM variant: the CPU class alone names the rung, lowering onto the kits its NumKong mask
 *  picks, its waiter, its atomic reference and its cache hints. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using ram_indexes =
    dense::ram_indexes<us::utilities::uuid_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                       elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>,
                       waiter_of<cpu_class_>, atomic_ref_for_cpu_class<cpu_class_>::template type,
                       typename cache_hints_for_cpu_class<cpu_class_>::type>;

/** One NVMe variant: scoring and the words its threads share compile per CPU class, while the
 *  metric comes from the parts' headers and the transport is the executor's runtime choice. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using nvme_indexes =
    dense::nvme_indexes<us::utilities::uuid_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                        elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>,
                        default_io_engine_t, atomic_ref_for_cpu_class<cpu_class_>::template type,
                        waiter_of<cpu_class_>>;

/** The start and end of every dtype's device code one GPU class unit embeds, in pairs in
 *  @c dense_dtypes_t order, as the table CMake writes for the class; each VRAM class unit
 *  specializes it. */
template <gpu_class_t gpu_class_>
std::span<byte_t const* const> bundle_bounds() noexcept;

/** The device code of one GPU class and dtype, between the bounds of the dtype's position. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
inline static bytes_view_t bundled_code() noexcept {
    std::span<byte_t const* const> const bounds = bundle_bounds<gpu_class_>();
    std::size_t const position = dense_dtypes_t::index_of(dtype_);
    return bytes_view_t {bounds[2 * position], bounds[2 * position + 1]};
}

#if USEARCH_ARCH_CUDA_

/** One NVIDIA class's fatbin of one dtype's kernels. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct cuda_fatbin {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static bytes_view_t bytes() noexcept { return bundled_code<gpu_class_, dtype_>(); }
};

/** One NVIDIA variant: the class names the rung and its capability ceiling, and with the dtype the
 *  fatbin its kernels sit in. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_nvidia_indexes = dense::vram_nvidia_indexes<us::utilities::uuid_t, dtype_, cuda_fatbin<gpu_class_, dtype_>,
                                                       fu::preferred_cache_hints_t, fu::standard_yield_t>;
#endif

#if USEARCH_ARCH_ROCM_

/** One AMD class's code object of one dtype's kernels, an offload bundle of its listed
 *  processors. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct hip_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static bytes_view_t bytes() noexcept { return bundled_code<gpu_class_, dtype_>(); }
};

/** One AMD variant: the class names the rung and its capability ceiling, and with the dtype the
 *  code object its kernels sit in. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_amd_indexes = dense::vram_amd_indexes<us::utilities::uuid_t, dtype_, hip_code_object<gpu_class_, dtype_>,
                                                 fu::preferred_cache_hints_t, fu::standard_yield_t>;
#endif

#if USEARCH_ARCH_METAL_

/** One Apple variant: the class names the rung and the capability ceiling its kernels compile for
 *  at the first @c reserve. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_apple_indexes =
    dense::vram_apple_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(gpu_class_), fu::standard_yield_t>;
#endif

/** The index type of the variant @p dtype_ and @p class_ compile in @p residency_. */
template <vendored_residency_t residency_, nk_dtype_t dtype_, auto class_>
struct dense_variants_of;
template <nk_dtype_t dtype_, auto class_>
struct dense_variants_of<vendored_residency_t::ram_k, dtype_, class_> {
    using type = ram_indexes<dtype_, class_>;
};
template <nk_dtype_t dtype_, auto class_>
struct dense_variants_of<vendored_residency_t::nvme_k, dtype_, class_> {
    using type = nvme_indexes<dtype_, class_>;
};
#if USEARCH_ARCH_CUDA_
template <nk_dtype_t dtype_, auto class_>
struct dense_variants_of<vendored_residency_t::vram_nvidia_k, dtype_, class_> {
    using type = vram_nvidia_indexes<dtype_, class_>;
};
#endif
#if USEARCH_ARCH_ROCM_
template <nk_dtype_t dtype_, auto class_>
struct dense_variants_of<vendored_residency_t::vram_amd_k, dtype_, class_> {
    using type = vram_amd_indexes<dtype_, class_>;
};
#endif
#if USEARCH_ARCH_METAL_
template <nk_dtype_t dtype_, auto class_>
struct dense_variants_of<vendored_residency_t::vram_apple_k, dtype_, class_> {
    using type = vram_apple_indexes<dtype_, class_>;
};
#endif

#pragma endregion Residencies

#pragma region Bodies

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

/** Bytes of cache each thread over @p topology keeps to itself: each core's share of the deepest
 *  cache a compute domain confines to its cores, at its smallest across domains, or zero where no
 *  platform source names one. */
inline static std::size_t thread_cache_bytes(fu_topology_t topology) noexcept {
    std::size_t domains = 0, smallest = 0;
    if (fu_compute_domains_count(topology, &domains) != fu_success_k) return 0;
    for (std::size_t domain = 0; domain != domains; ++domain) {
        std::size_t bytes = 0;
        if (fu_compute_cache_bytes_per_core_in(topology, domain, &bytes) != fu_success_k || !bytes) continue;
        smallest = smallest ? min_of(smallest, bytes) : bytes;
    }
    return smallest;
}

/** Searches @p index under an expensive @p predicate through a family of per-thread predicates
 *  gathering what the traversal @p config names hands them; the engine opens the family and
 *  reserves it for the widest round it settles. */
template <typename index_type_>
inline static expected<typename index_type_::search_result_t> search_with_expensive_predicate(
    index_type_& index, dense::rows_view_t queries, ranked_hints_view_t hints, ranked_matches_span_t matches,
    typename index_type_::search_config_t config, usearch_predicate_t predicate,
    typename index_type_::executor_t& executor, dense_scratch<index_type_>& scratch) noexcept {
    auto const search_if = [&]<typename predicates_type_>(predicates_type_& predicates) noexcept {
        if constexpr (is_ram_index<index_type_>)
            return index.search_if(queries, matches, config, predicates, executor, to_writable_span(scratch.traversals),
                                   hints);
        else return index.search_if(queries, matches, config, predicates, executor, {}, hints);
    };
    if (config.traversal == traversal_t::scan_k) {
        expensive_predicates<typename index_type_::scan_payload_t> scanned {predicate};
        return search_if(scanned);
    }
    expensive_predicates<typename index_type_::walk_payload_t> walked {predicate};
    return search_if(walked);
}

/** A refinement in flight: the typed donors it absorbs, each nulled once consumed, which the plan
 *  reads in place. */
template <typename index_type_>
struct dense_refinement {
    growing_buffer<index_type_*> donors;
    typename index_type_::refine_plan_t plan;
};

/**
 *  @brief One variant's bodies, every one a static member its verb table points at, taking the
 *      index as @c void* and casting it back to @p index_type_.
 *
 *  An NVMe view serves its checkpoint in place, so its mutating bodies answer
 *  @c usearch_status_read_only_k; a device index answers @c unsupported_k for what only a host walk
 *  takes, as host predicates and cursors.
 */
template <typename index_type_>
struct specialized_index {
    using index_t = index_type_;
    using executor_t = typename index_type_::executor_t;
    using verbs_t = dense_verbs<vendored_residency_of<executor_t>>;
    using key_t = typename verbs_t::key_t;
    using keys_view_t = typename verbs_t::keys_view_t;
    using rows_view_t = dense::rows_view_t;
    using rows_span_t = dense::rows_span_t;
    using scratch_t = dense_scratch<index_type_>;
    using search_config_t = typename index_type_::search_config_t;

    static constexpr bool viewed_k = is_nvme_index<index_type_>;

    /** Whether the engine sizes a pass's tiles from @c memory_limit; Apple's holds twice the
     *  neighbors' bytes instead, so it refuses a limit. */
    static constexpr bool limits_memory_k = vendored_residency_of<executor_t> != vendored_residency_t::vram_apple_k;

    static index_t& of(void* index) noexcept { return *static_cast<index_t*>(index); }
    static index_t const& of(void const* index) noexcept { return *static_cast<index_t const*>(index); }

    /** The traversal stats of one engine result, or its failure. */
    template <typename result_type_>
    static expected<traversal_stats_t> stats_of(expected<result_type_> const& result) noexcept {
        return {traversal_stats_of(result.value), result.status};
    }

    /** The engine's search config of the call's @p search, a RAM scan blocking its rows by each
     *  thread's own cache. */
    static search_config_t search_config_of(::search_config_t search, executor_t& executor) noexcept {
        search_config_t config {};
        config.expansion_factor = search.expansion_factor;
        config.traversal = search.traversal;
        if constexpr (requires { config.cache_bytes; }) config.cache_bytes = thread_cache_bytes(executor.topology());
        return config;
    }

    /** The engine's refine config of the call's @p refine, which only a mutable index spells. */
    template <typename engine_type_ = index_type_>
    static typename engine_type_::refine_config_t refine_config_of(::refine_config_t refine) noexcept {
        typename engine_type_::refine_config_t config {};
        config.expansion_factor = refine.expansion_factor;
        config.memory_limit = refine.memory_limit;
        return config;
    }

    /** The engine's split config of the call's @p split. */
    template <typename engine_type_ = index_type_>
    static typename engine_type_::split_config_t split_config_of(::split_config_t split) noexcept {
        return {refine_config_of<engine_type_>(split.refine), split.seed};
    }

    /** Hands @p walk the alternative @p predicate holds, as <tt>walk(narrowing)</tt>; a device walk
     *  answers @c unsupported_k for any predicate, so it never instantiates a narrowed walk. */
    template <typename walk_type_>
    static expected<traversal_stats_t> with_predicate(predicate_t& predicate, walk_type_&& walk) noexcept {
        return std::visit(
            [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept -> expected<traversal_stats_t> {
                if constexpr (std::is_same_v<narrowing_type_, no_predicate_t> || !is_vram_index<index_type_>)
                    return walk(narrowing);
                else return {{}, status_t::unsupported_k};
            },
            predicate);
    }

    /** The C layer's save config, whose contract the residency picks: device storage cannot be
     *  settlement-probed, so a GPU handle certifies. */
    static save_config_t save_config_of() noexcept {
        save_config_t config;
        if constexpr (is_vram_index<index_type_>) config.synchronization = synchronization_t::immutable_k;
        return config;
    }

    static void* create(nk_kernel_kind_t kernel, dense::geometry_t geometry, forkunion_allocator_t tables) noexcept {
        if constexpr (viewed_k) return new (std::nothrow) index_t(tables);
        else if constexpr (is_ram_index<index_type_>)
            return new (std::nothrow) index_t(kernel, geometry, tables, machine_cache_line_bytes());
        else return new (std::nothrow) index_t(kernel, geometry, tables);
    }
    static void destroy(void* index) noexcept {
        if constexpr (viewed_k) of(index).release();
        delete static_cast<index_t*>(index);
    }
    static void bound_shards(void* index, std::size_t nodes) noexcept {
        if constexpr (!viewed_k) of(index).bound_shards(nodes);
    }
    static status_t emplace(void* index, arena_residency_t residency, fu_memory_domain_id_t memory_domain) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else if constexpr (is_ram_index<index_type_>) return of(index).emplace(memory_domain);
        else {
            // Init names devices and no executor yet, so a stream opened for the shard alone
            // names the device.
            int const device = static_cast<int>(residency.device);
            auto [executor, status] = executor_t::spawn(driver_of(gpu_drivers(), std::type_identity<executor_t> {}),
                                                        std::span<int const> {&device, 1});
            return failed(status) ? status : of(index).emplace(executor.stream(0));
        }
    }
    static status_t view(void* index, std::string_view path, std::span<std::string_view const> paths,
                         usearch_ram_options_t const* ram, usearch_executor_t executor) noexcept {
        if constexpr (viewed_k) {
            executor_t* const engines = executor ? std::get_if<executor_t>(&executor->executor) : nullptr;
            if (ram && !engines) return status_t::invalid_argument_k;
            if (status_t const viewed = of(index).view(path, paths); failed(viewed) || !ram) return viewed;
            return of(index).hold_keys(forkunion_allocator_t {tables_domain(*ram)}, ram->memory_limit, *engines);
        }
        else return status_t::unsupported_k;
    }

    static nk_kernel_kind_t kernel(void const* index) noexcept { return of(index).kernel(); }
    static dense::geometry_t geometry(void const* index) noexcept {
        return {of(index).dimensions(), static_cast<connectivity_t>(of(index).connectivity())};
    }
    static arena_residency_t residency(void const* index, std::size_t shard) noexcept {
        if constexpr (is_vram_index<index_type_>)
            return vram_residency(static_cast<std::size_t>(of(index).shard(shard).device_specs().ordinal));
        else return ram_residency_k;
    }
    static std::size_t shards_count(void const* index) noexcept { return of(index).shards_count(); }
    static usearch_load_path_t load_path(void const* index) noexcept {
        if constexpr (requires { of(index).loaded_through(); }) return usearch_load_path_of(of(index).loaded_through());
        else return usearch_load_path_none_k;
    }
    static std::size_t size(void const* index) noexcept { return of(index).size(); }
    static std::size_t capacity(void const* index) noexcept { return of(index).capacity(); }
    static std::size_t memory_usage(void const* index) noexcept { return of(index).memory_usage(); }

    /** A view already is its checkpoint, so it answers @c unsupported_k. */
    static expected<std::size_t> storage_requirements(void const* index) noexcept {
        if constexpr (viewed_k) return {0, status_t::unsupported_k};
        else return of(index).storage_requirements(save_config_of());
    }

    static status_t serves_writes(void const*) noexcept {
        return viewed_k ? status_t::read_only_k : status_t::success_k;
    }
    static status_t reserve(void* index, std::size_t capacity) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(index).reserve({capacity}).status;
    }
    static status_t clear(void* index) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(index).clear();
    }

    /** Writes @p rows under @p keys through one plan, a step per chunk, and publishes them all at
     *  once. */
    static expected<traversal_stats_t> insert(void* index, keys_view_t keys, rows_view_t rows, insert_config_t insert,
                                              executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            typename index_t::insert_config_t config {};
            config.expansion_factor = insert.expansion_factor;
            std::size_t const count = keys.size();
            // Only a handler splits the call, and one plan still publishes every slice at once.
            std::size_t const chunk = ticks.slice();
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            auto [plan, status] = of(index).plan_insert(count, executor, config);
            if (failed(status)) return {{}, status};
            for (std::size_t inserted = 0; inserted != count;) {
                std::size_t const taking = min_of(chunk, count - inserted);
                status_t const stepped = [&]() noexcept {
                    if constexpr (is_ram_index<index_type_>)
                        return plan.insert(inserted, keys[nk::range(inserted, inserted + taking)],
                                           rows.slice(inserted, taking), to_writable_span(scratch.traversals));
                    else
                        return plan.insert(inserted, keys[nk::range(inserted, inserted + taking)],
                                           rows.slice(inserted, taking));
                }();
                if (failed(stepped)) {
                    // A refused RAM step dispensed nothing, so the dropped plan releases every
                    // claim.
                    if constexpr (is_ram_index<index_type_>) return {{}, stepped};
                    else break;
                }
                inserted += taking;
                // Every entry is claimed before the first report and none can be withdrawn, so a
                // cancel goes unheard.
                ticks(inserted, count);
            }
            // After a failed device step, `finish` publishes nothing and reports the failure the
            // plan kept.
            return stats_of(plan.finish());
        }
    }

    static expected<refine_report_t> refine(void* index, ::refine_config_t config, executor_t& executor,
                                            progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else if (!limits_memory_k && config.memory_limit) return {{}, status_t::unsupported_k};
        else
            return of(index).refine(
                refine_config_of(config), executor,
                [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
    }

    static expected<refine_report_t> absorb(void* index, std::span<void*> donors, ::refine_config_t config,
                                            executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else if (!limits_memory_k && config.memory_limit) return {{}, status_t::unsupported_k};
        else {
            growing_buffer<index_t*> typed {global_allocator};
            if (failed(typed.reserve_exactly(donors.size()))) return {{}, status_t::bad_alloc_k};
            for (void* const donor : donors) typed.push_back_reserved(static_cast<index_t*>(donor));
            expected<refine_report_t> const refined = of(index).absorb(
                to_writable_span(typed), refine_config_of(config), executor,
                [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
            for (std::size_t position = 0; position != donors.size(); ++position)
                if (!typed[position]) donors[position] = nullptr;
            return refined;
        }
    }

    static expected<dense::compact_result_t> compact(void* index, std::span<void* const> sources,
                                                     dense::compact_config_t config, executor_t& executor,
                                                     progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else {
            growing_buffer<index_t*> typed {global_allocator};
            if (failed(typed.reserve_exactly(sources.size()))) return {{}, status_t::bad_alloc_k};
            for (void* const source : sources) typed.push_back_reserved(static_cast<index_t*>(source));
            return dense::compact(
                to_span(typed), of(index), executor, config,
                [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
        }
    }

    /** The typed indexes behind @p indexes, or an empty buffer when it found no memory. */
    static growing_buffer<index_t*> indexes_of(variants_view_t indexes) noexcept {
        growing_buffer<index_t*> typed {global_allocator};
        if (failed(typed.reserve_exactly(indexes.size()))) return typed;
        for (std::size_t position = 0; position != indexes.size(); ++position)
            typed.push_back_reserved(static_cast<index_t*>(indexes[position]));
        return typed;
    }

    static expected<refine_report_t> split(void* index, variants_view_t receivers, ::split_config_t config,
                                           executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else if (!limits_memory_k && config.refine.memory_limit) return {{}, status_t::unsupported_k};
        else {
            growing_buffer<index_t*> typed = indexes_of(receivers);
            if (typed.size() != receivers.size()) return {{}, status_t::bad_alloc_k};
            return of(index).split(
                to_writable_span(typed), split_config_of(config), executor,
                [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
        }
    }

    static expected<void*> refinement_init(void* index, variants_view_t donors, variants_view_t receivers,
                                           ::split_config_t config, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {nullptr, status_t::read_only_k};
        else if (!limits_memory_k && config.refine.memory_limit) return {nullptr, status_t::unsupported_k};
        else {
            variants_view_t const others = receivers.empty() ? donors : receivers;
            growing_buffer<index_t*> typed = indexes_of(others);
            if (typed.size() != others.size()) return {nullptr, status_t::bad_alloc_k};
            // The plan borrows the receiver slots, whose storage moves along with the buffer.
            auto [plan, status] =
                receivers.empty()
                    ? of(index).plan_absorb(to_writable_span(typed), refine_config_of(config.refine), executor)
                    : of(index).plan_split(to_writable_span(typed), split_config_of(config), executor);
            if (failed(status)) return {nullptr, status};
            dense_refinement<index_t>* const opened = new (std::nothrow)
                dense_refinement<index_t> {std::move(typed), std::move(plan)};
            if (!opened) return {nullptr, status_t::bad_alloc_k};
            return {opened, status_t::success_k};
        }
    }
    static status_t refinement_next(void*, void* refinement, executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else {
            dense_refinement<index_t>& opened = *static_cast<dense_refinement<index_t>*>(refinement);
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return prepared;
            auto const report = [&](std::size_t swept, std::size_t population) noexcept {
                return ticks(swept, population);
            };
            if constexpr (is_ram_index<index_type_>)
                return opened.plan.run(executor, to_writable_span(scratch.traversals), report);
            else return opened.plan.run(executor, report);
        }
    }
    static refine_report_t refinement_running(void const* refinement) noexcept {
        if constexpr (viewed_k) return {};
        else return static_cast<dense_refinement<index_t> const*>(refinement)->plan.running();
    }
    static bool refinement_consumed(void const* refinement, std::size_t donor) noexcept {
        if constexpr (viewed_k) return false;
        else return !static_cast<dense_refinement<index_t> const*>(refinement)->donors[donor];
    }
    static expected<refine_report_t> refinement_finish(void*, void* refinement, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            dense_refinement<index_t>& opened = *static_cast<dense_refinement<index_t>*>(refinement);
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            if constexpr (is_ram_index<index_type_>)
                return opened.plan.finish(executor, to_writable_span(scratch.traversals));
            else return opened.plan.finish(executor);
        }
    }
    static void refinement_free(void* refinement) noexcept {
        if constexpr (!viewed_k) delete static_cast<dense_refinement<index_t>*>(refinement);
    }

    static expected<reachability_t> health(void* index, executor_t& executor) noexcept {
        return of(index).health(executor);
    }
    static status_t count(void* index, std::span<key_t const> keys, std::span<std::size_t> entries,
                          executor_t& executor) noexcept {
        return of(index).count(keys, entries, executor);
    }

    /** Every tower casts the stored scalar into the rows' on its own side, a GPU one on the device;
     *  Apple's reads the stored scalar alone. */
    static status_t get(void* index, std::span<key_t const> keys, rows_span_t outputs, std::span<std::size_t> entries,
                        std::size_t wanted, executor_t& executor) noexcept {
        return of(index).get(keys, outputs, entries, wanted, executor);
    }

    /** Ranks each query's nearest entries, every walk seeded by its query's @p hints; a cheap
     *  predicate is asked a quad at a time, an expensive one once per round each thread settles. A
     *  device walk answers @c unsupported_k for hints, as for a predicate. */
    static expected<traversal_stats_t> search(void* index, rows_view_t queries, ranked_hints_view_t hints,
                                              ranked_matches_span_t matches, ::search_config_t search,
                                              predicate_t predicate, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (!hints.counts.empty()) return {{}, status_t::unsupported_k};
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        search_config_t const config = search_config_of(search, executor);
        return with_predicate(predicate, [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept {
            if constexpr (is_vram_index<index_type_>)
                return stats_of(of(index).search(queries, matches, config, executor));
            else if constexpr (std::is_same_v<narrowing_type_, expensive_predicate_t>)
                return stats_of(search_with_expensive_predicate(of(index), queries, hints, matches, config,
                                                                narrowing.predicate, executor, scratch));
            else if constexpr (std::is_same_v<narrowing_type_, cheap_predicate_t>) {
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).search_if(queries, matches, config, narrowing, executor,
                                                        to_writable_span(scratch.traversals), hints));
                else return stats_of(of(index).search_if(queries, matches, config, narrowing, executor, {}, hints));
            }
            else if constexpr (is_ram_index<index_type_>)
                return stats_of(
                    of(index).search(queries, matches, config, executor, to_writable_span(scratch.traversals), hints));
            else return stats_of(of(index).search(queries, matches, config, executor, {}, hints));
        });
    }

    /** Keeps the largest products of each left row on the tape; any predicate is asked a quad at a
     *  time, and a GPU index answers @c unsupported_k for a scan. */
    static expected<traversal_stats_t> multiply(void* index, rows_view_t left, products_tape_t products,
                                                std::span<std::size_t> scored, ::search_config_t search,
                                                predicate_t predicate, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (search.traversal == traversal_t::scan_k) return {{}, status_t::unsupported_k};
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        search_config_t const config = search_config_of(search, executor);
        return with_predicate(predicate, [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept {
            if constexpr (is_vram_index<index_type_>)
                return stats_of(of(index).multiply(left, products, scored, config, executor));
            else if constexpr (std::is_same_v<narrowing_type_, no_predicate_t>) {
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).multiply(left, products, scored, config, executor,
                                                       to_writable_span(scratch.traversals)));
                else return stats_of(of(index).multiply(left, products, scored, config, executor));
            }
            else {
                cheap_predicate_t cheap {narrowing.predicate};
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).multiply_if(left, products, scored, config, cheap, executor,
                                                          to_writable_span(scratch.traversals)));
                else return stats_of(of(index).multiply_if(left, products, scored, config, cheap, executor));
            }
        });
    }

    /** Late interaction of the queries the offsets cut, packed on the tape, every row's walks
     *  seeded by its query's hints; a GPU index walks uniform queries alone, answering
     *  @c unsupported_k for a scan, ragged offsets or hints. */
    static expected<traversal_stats_t> maxsim(void* index, maxsim_queries_view_t queries, products_tape_t matches,
                                              std::span<std::size_t> scored, std::size_t wanted,
                                              ::search_config_t search, predicate_t predicate,
                                              executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (!queries.hints.counts.empty()) return {{}, status_t::unsupported_k};
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        typename index_t::maxsim_config_t const config {search_config_of(search, executor), wanted};
        std::span<std::size_t const> const query_offsets = queries.offsets;
        ranked_hints_view_t const hints = queries.hints;
        return with_predicate(predicate, [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept {
            if constexpr (is_vram_index<index_type_>)
                return stats_of(of(index).maxsim(queries.rows, query_offsets, scored, matches, config, executor));
            else if constexpr (std::is_same_v<narrowing_type_, no_predicate_t>) {
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).maxsim(queries.rows, query_offsets, scored, matches, config, executor,
                                                     to_writable_span(scratch.traversals), hints));
                else
                    return stats_of(
                        of(index).maxsim(queries.rows, query_offsets, scored, matches, config, executor, {}, hints));
            }
            else {
                cheap_predicate_t cheap {narrowing.predicate};
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).maxsim_if(queries.rows, query_offsets, scored, matches, config, cheap,
                                                        executor, to_writable_span(scratch.traversals), hints));
                else
                    return stats_of(of(index).maxsim_if(queries.rows, query_offsets, scored, matches, config, cheap,
                                                        executor, {}, hints));
            }
        });
    }

    /** Attends the queries over their walks' neighbors; a GPU index answers @c unsupported_k for a
     *  scan. */
    static expected<traversal_stats_t> attend(void* index, rows_view_t queries, attention_arguments_t attention,
                                              ::search_config_t search, predicate_t predicate,
                                              executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (search.traversal == traversal_t::scan_k) return {{}, status_t::unsupported_k};
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        search_config_t const config = search_config_of(search, executor);
        return with_predicate(predicate, [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept {
            if constexpr (is_vram_index<index_type_>)
                return stats_of(of(index).attend(queries, attention.heads, attention.value_lookup,
                                                 attention.score_scale, attention.outputs, attention.carry, config,
                                                 executor));
            else if constexpr (std::is_same_v<narrowing_type_, no_predicate_t>) {
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).attend(queries, attention.heads, attention.value_lookup,
                                                     attention.score_scale, attention.outputs, attention.carry, config,
                                                     executor, to_writable_span(scratch.traversals)));
                else
                    return stats_of(of(index).attend(queries, attention.heads, attention.value_lookup,
                                                     attention.score_scale, attention.outputs, attention.carry, config,
                                                     executor));
            }
            else {
                cheap_predicate_t cheap {narrowing.predicate};
                if constexpr (is_ram_index<index_type_>)
                    return stats_of(of(index).attend_if(queries, attention.heads, attention.value_lookup,
                                                        attention.score_scale, attention.outputs, attention.carry,
                                                        config, cheap, executor, to_writable_span(scratch.traversals)));
                else
                    return stats_of(of(index).attend_if(queries, attention.heads, attention.value_lookup,
                                                        attention.score_scale, attention.outputs, attention.carry,
                                                        config, cheap, executor));
            }
        });
    }

    /** Opens a cursor over every entry, breadth-first from @p starts when it names any, which a
     *  view resolves on its first pull; an index without a cursor answers @c unsupported_k. */
    static expected<void*> cursor_init(void* index, std::span<key_t const> starts) noexcept {
        if constexpr (requires { typename index_t::cursor_t; }) {
            typename index_t::cursor_t* const opened = new (std::nothrow)
                typename index_t::cursor_t(forkunion_allocator_t {no_memory_domain_k});
            if (!opened) return {nullptr, status_t::bad_alloc_k};
            if (status_t const walking = opened->open(of(index), starts); failed(walking))
                return delete opened, expected<void*> {nullptr, walking};
            return {opened, status_t::success_k};
        }
        else return {nullptr, status_t::unsupported_k};
    }
    static expected<std::size_t> cursor_next(void* index, void* cursor, std::span<key_t> keys, links_outputs_t* links,
                                             executor_t& executor) noexcept {
        if constexpr (!requires { typename index_t::cursor_t; }) return {0, status_t::unsupported_k};
        else {
            typename index_t::cursor_t& walk = *static_cast<typename index_t::cursor_t*>(cursor);
            if constexpr (viewed_k)
                return walk.pull(of(index), keys, links ? &links->keys : nullptr, links ? &links->offsets : nullptr,
                                 executor);
            else return walk.pull(of(index), keys, links ? &links->keys : nullptr, links ? &links->offsets : nullptr);
        }
    }
    static void cursor_free(void* cursor) noexcept {
        if constexpr (requires { typename index_t::cursor_t; }) delete static_cast<typename index_t::cursor_t*>(cursor);
    }

    /** Samples a RAM index or an NVMe view; a GPU index does not sample yet. */
    static expected<sample_result_t> sample(void* index, sample_inputs<key_t> inputs, sample_outputs<key_t> outputs,
                                            sample_config_t config, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else return of(index).sample(inputs, outputs, config, executor);
    }

    static status_t save(void* index, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::unsupported_k;
        else return of(index).save(path, save_config_of(), executor, ticks);
    }
    static status_t save_buffer(void* index, bytes_span_t buffer, executor_t& executor,
                                progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::unsupported_k;
        else return of(index).save(buffer, save_config_of(), executor, ticks);
    }
    static status_t load(void* index, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(index).load(path, load_config_t {}, executor, ticks);
    }

    /** Loads a checkpoint where it sits: host bytes into any index, a device's only into a tower
     *  whose shard plans copy it device to device. */
    static status_t load_buffer(void* index, checkpoint_source_t source, executor_t& executor,
                                progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else if constexpr (requires { of(index).load(source, load_config_t {}, executor, ticks); })
            return of(index).load(source, load_config_t {}, executor, ticks);
        else if (source.device != host_device_k) return status_t::unsupported_k;
        else return of(index).load(source.bytes, load_config_t {}, executor, ticks);
    }

    /** The verb table of the variant, every entry one of the bodies above. */
    static constexpr verbs_t verbs() noexcept {
        return {
            .dtype = index_t::vector_dtype_k,
            .cpu_class = cpu_class_t::unknown_k,
            .gpu_class = gpu_class_t::unknown_k,
            // Lifetime
            .create = &create,
            .destroy = &destroy,
            .bound_shards = &bound_shards,
            .emplace = &emplace,
            .view = &view,
            // Shape
            .kernel = &kernel,
            .geometry = &geometry,
            .residency = &residency,
            .shards_count = &shards_count,
            .load_path = &load_path,
            .size = &size,
            .capacity = &capacity,
            .memory_usage = &memory_usage,
            .storage_requirements = &storage_requirements,
            // Writes
            .serves_writes = &serves_writes,
            .reserve = &reserve,
            .clear = &clear,
            .insert = &insert,
            // Refining
            .refine = &refine,
            .absorb = &absorb,
            .compact = &compact,
            .split = &split,
            .health = &health,
            // Refinements
            .refinement_init = &refinement_init,
            .refinement_next = &refinement_next,
            .refinement_running = &refinement_running,
            .refinement_consumed = &refinement_consumed,
            .refinement_finish = &refinement_finish,
            .refinement_free = &refinement_free,
            // Lookups
            .count = &count,
            .get = &get,
            // Walks
            .search = &search,
            .multiply = &multiply,
            .maxsim = &maxsim,
            .attend = &attend,
            // Exploration
            .cursor_init = &cursor_init,
            .cursor_next = &cursor_next,
            .cursor_free = &cursor_free,
            .sample = &sample,
            // Persistence
            .save = &save,
            .save_buffer = &save_buffer,
            .load = &load,
            .load_buffer = &load_buffer,
        };
    }
};

#pragma endregion Bodies

#pragma region Class Tables

/**
 *  @brief Whether a class compiles its own variant of @p dtype_ in @p residency_.
 *
 *  On RAM it does where its 4-way or tiles kit differs from the class it degrades to, or its
 *  ForkUnion mask does; on NVMe where its kit alone differs, as a storage walk shares a few
 *  counters and never waits, so its parent's reference serves; on a device always, since each
 *  class's unit alone carries kernels for its generation. The serial floor always compiles its
 *  variants, and so does the monoversioned class, which has no parent; elsewhere a variant a class
 *  drops is its parent's, which the ladder reaches on the next rung.
 */
template <vendored_residency_t residency_, nk_dtype_t dtype_, auto class_>
static consteval bool variant_advances() noexcept {
    if constexpr (residency_ == vendored_residency_t::ram_k || residency_ == vendored_residency_t::nvme_k) {
        if constexpr (class_ == cpu_class_t::serial_k || class_ == cpu_class_t::monoversioned_k) return true;
        else {
            constexpr cpu_class_t parent_k = parent_class_of(class_);
            bool const kits_differ = !std::is_same_v<quads_kit_of<dtype_, class_>, quads_kit_of<dtype_, parent_k>> ||
                                     !std::is_same_v<tiles_kit_of<dtype_, class_>, tiles_kit_of<dtype_, parent_k>>;
            if constexpr (residency_ == vendored_residency_t::nvme_k) return kits_differ;
            else return kits_differ || forkunion_capabilities_of<class_>() != forkunion_capabilities_of<parent_k>();
        }
    }
    else return true;
}

/** Variants of the dtypes in the list that @p class_ compiles in @p residency_. */
template <vendored_residency_t residency_, auto class_, nk_dtype_t... dtypes_>
static consteval std::size_t tables_count(dtype_list<dtypes_...>) noexcept {
    return (std::size_t {variant_advances<residency_, dtypes_, class_>()} + ... + 0);
}

/** The verb tables of those variants, in ladder order; a dropped variant's index is never
 *  instantiated. */
template <vendored_residency_t residency_, auto class_, nk_dtype_t... dtypes_>
static consteval std::array<dense_verbs<residency_>, tables_count<residency_, class_>(dtype_list<dtypes_...> {})>
    tables_for(dtype_list<dtypes_...>) noexcept {
    std::array<dense_verbs<residency_>, tables_count<residency_, class_>(dtype_list<dtypes_...> {})> tables {};
    std::size_t position = 0;
    auto place = [&]<nk_dtype_t dtype_>() {
        if constexpr (variant_advances<residency_, dtype_, class_>()) {
            dense_verbs<residency_> table =
                specialized_index<typename dense_variants_of<residency_, dtype_, class_>::type>::verbs();
            if constexpr (residency_ == vendored_residency_t::ram_k || residency_ == vendored_residency_t::nvme_k)
                table.cpu_class = class_;
            else table.gpu_class = class_;
            tables[position++] = table;
        }
    };
    (place.template operator()<dtypes_>(), ...);
    return tables;
}

template <vendored_residency_t residency_, auto class_>
dense_verbs<residency_> const* dense_class_tables<residency_, class_>::find(nk_dtype_t dtype) noexcept {
    static constexpr std::array<dense_verbs<residency_>, tables_count<residency_, class_>(dense_dtypes_t {})> tables_k =
        tables_for<residency_, class_>(dense_dtypes_t {});
    for (dense_verbs<residency_> const& table : tables_k)
        if (table.dtype == dtype) return &table;
    return nullptr;
}

#pragma endregion Class Tables
