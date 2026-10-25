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

/** One NVMe variant: scoring compiles per CPU class, while the metric comes from the parts' headers
 *  and the transport is the executor's runtime choice. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using nvme_indexes =
    dense::nvme_indexes<us::utilities::uuid_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                        elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>>;

/** The start and end of every dtype's device code one GPU class unit embeds, in pairs in
 *  @c dense_dtypes_t order, as the table CMake writes for the class; each VRAM class unit
 *  specializes it. */
template <gpu_class_t gpu_class_>
std::span<byte_t const* const> bundle_bounds() noexcept;

/** The device code of one GPU class and dtype, between the bounds of the dtype's position. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
static bytes_view_t bundled_code() noexcept {
    std::span<byte_t const* const> const bounds = bundle_bounds<gpu_class_>();
    std::size_t const position = dense_dtypes_t::position_of(dtype_);
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

/** The index type of the variant @p dtype_ and @p class_ compile in @p home_. */
template <dense_home_t home_, nk_dtype_t dtype_, auto class_>
struct dense_indexes_of;
template <nk_dtype_t dtype_, auto class_>
struct dense_indexes_of<dense_home_t::ram_k, dtype_, class_> {
    using type = ram_indexes<dtype_, class_>;
};
template <nk_dtype_t dtype_, auto class_>
struct dense_indexes_of<dense_home_t::nvme_k, dtype_, class_> {
    using type = nvme_indexes<dtype_, class_>;
};
#if USEARCH_ARCH_CUDA_
template <nk_dtype_t dtype_, auto class_>
struct dense_indexes_of<dense_home_t::vram_nvidia_k, dtype_, class_> {
    using type = vram_nvidia_indexes<dtype_, class_>;
};
#endif
#if USEARCH_ARCH_ROCM_
template <nk_dtype_t dtype_, auto class_>
struct dense_indexes_of<dense_home_t::vram_amd_k, dtype_, class_> {
    using type = vram_amd_indexes<dtype_, class_>;
};
#endif
#if USEARCH_ARCH_METAL_
template <nk_dtype_t dtype_, auto class_>
struct dense_indexes_of<dense_home_t::vram_apple_k, dtype_, class_> {
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
static std::size_t thread_cache_bytes(fu_topology_t topology) noexcept {
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
static expected<typename index_type_::search_result_t> search_with_expensive_predicate(
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

/** A refinement in flight: the typed donors it lands, each nulled once landed, which the plan reads
 *  in place. */
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
    using verbs_t = dense_verbs<home_of<executor_t>>;
    using key_t = typename verbs_t::key_t;
    using keys_view_t = typename verbs_t::keys_view_t;
    using keys_span_t = typename verbs_t::keys_span_t;
    using scores_span_t = typename verbs_t::scores_span_t;
    using counts_span_t = typename verbs_t::counts_span_t;
    using rows_view_t = dense::rows_view_t;
    using rows_span_t = dense::rows_span_t;
    using walked_t = typename verbs_t::walked_t;
    using refined_t = typename verbs_t::refined_t;
    using scratch_t = dense_scratch<index_type_>;

    static constexpr bool viewed_k = is_nvme_index<index_type_>;

    static index_t& of(void* index) noexcept { return *static_cast<index_t*>(index); }
    static index_t const& of(void const* index) noexcept { return *static_cast<index_t const*>(index); }

    /** The C stats of one engine result. */
    template <typename result_type_>
    static walked_t walked(expected<result_type_> const& result) noexcept {
        usearch_traversal_stats_t stats {};
        accumulate_stats(&stats, result.value);
        return {stats, result.status};
    }

    /** The search config of the C knobs, a RAM scan blocking its rows by each thread's own
     *  cache. */
    static typename index_t::search_config_t search_config_of(dense_walk_config_t walk, executor_t& executor) noexcept {
        typename index_t::search_config_t config {};
        config.expansion_factor = walk.expansion_factor;
        config.traversal = walk.traversal;
        if constexpr (requires { config.cache_bytes; }) config.cache_bytes = thread_cache_bytes(executor.topology());
        return config;
    }

    /** The C layer's save config, whose contract the residency picks: device storage cannot be
     *  settlement-probed, so a GPU handle certifies. */
    static save_config_t save_config_of() noexcept {
        save_config_t config;
        if constexpr (is_vram_index<index_type_>) config.synchronization = synchronization_t::immutable_k;
        return config;
    }

    static void* create(nk_kernel_kind_t kernel, forkunion_allocator_t tables) noexcept {
        if constexpr (viewed_k) return new (std::nothrow) index_t(tables);
        else if constexpr (is_ram_index<index_type_>)
            return new (std::nothrow) index_t(kernel, tables, machine_cache_line_bytes());
        else return new (std::nothrow) index_t(kernel, tables);
    }
    static void destroy(void* index) noexcept { delete static_cast<index_t*>(index); }
    static void bound_shards(void* index, std::size_t rows) noexcept {
        if constexpr (!viewed_k) of(index).bound_shards(rows);
    }
    static status_t emplace(void* index, int placement) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else if constexpr (is_ram_index<index_type_>)
            return of(index).emplace(static_cast<fu_memory_domain_id_t>(placement));
        else {
            // Init names devices and no executor yet, so a stream opened for the placement alone
            // names the device.
            auto [executor, status] = executor_t::spawn(std::span<int const> {&placement, 1});
            return failed(status) ? status : of(index).emplace(executor.stream(0));
        }
    }
    static status_t view(void* index, std::string_view path, std::span<std::string_view const> paths) noexcept {
        if constexpr (viewed_k) return of(index).view(path, paths);
        else return status_t::unsupported_k;
    }

    static nk_kernel_kind_t kernel(void const* index) noexcept { return of(index).kernel(); }
    static int device(void const* index, std::size_t shard) noexcept {
        if constexpr (is_vram_index<index_type_>) return of(index).shard(shard).device_specs().ordinal;
        else return host_device_k;
    }
    static std::size_t shards_count(void const* index) noexcept { return of(index).shards_count(); }
    static std::size_t dimensions(void const* index) noexcept { return of(index).dimensions(); }
    static std::size_t connectivity(void const* index) noexcept { return of(index).connectivity(); }
    static std::size_t size(void const* index) noexcept { return of(index).size(); }
    static std::size_t capacity(void const* index) noexcept { return of(index).capacity(); }
    static std::size_t memory_usage(void const* index) noexcept { return of(index).memory_usage(); }

    /** A view already is its checkpoint, so it answers @c unsupported_k. */
    static expected<std::size_t> storage_requirements(void const* index) noexcept {
        if constexpr (viewed_k) return {0, status_t::unsupported_k};
        else return of(index).storage_requirements(save_config_of());
    }

    static status_t reserve(void* index, std::size_t capacity, connectivity_t connectivity,
                            std::size_t dimensions) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(index).reserve({capacity, connectivity, dimensions}).status;
    }
    static status_t clear(void* index) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(index).clear();
    }

    /** Lands @p rows under @p keys through one plan, a step per chunk, and publishes them all at
     *  once. */
    static walked_t insert(void* index, keys_view_t keys, rows_view_t rows, progress_ticks_t& ticks,
                           executor_t& executor, u32_t expansion_factor) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            typename index_t::insert_config_t config {};
            config.expansion_factor = expansion_factor;
            std::size_t const count = keys.size();
            // Only a handler splits the call, and one plan still publishes every slice at once.
            std::size_t const chunk = ticks.slice();
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            auto [plan, status] = of(index).plan_insert(count, executor, config);
            if (failed(status)) return {{}, status};
            for (std::size_t landed = 0; landed != count;) {
                std::size_t const taking = min_of(chunk, count - landed);
                status_t const stepped = [&]() noexcept {
                    if constexpr (is_ram_index<index_type_>)
                        return plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking),
                                           to_writable_span(scratch.traversals));
                    else
                        return plan.insert(landed, keys[nk::range(landed, landed + taking)],
                                           rows.slice(landed, taking));
                }();
                if (failed(stepped)) {
                    // A refused RAM step dispensed nothing, so the dropped plan releases every
                    // claim.
                    if constexpr (is_ram_index<index_type_>) return {{}, stepped};
                    else break;
                }
                landed += taking;
                // Every row is claimed before the first report and none can be withdrawn, so a
                // cancel goes unheard.
                ticks(landed, count);
            }
            // After a failed device step, `finish` publishes nothing and reports the failure the
            // plan kept.
            return walked(plan.finish());
        }
    }

    static refined_t refine(void* index, executor_t& executor, u32_t expansion_factor,
                            progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            typename index_t::refine_config_t config {};
            config.expansion_factor = expansion_factor;
            return of(index).refine(executor, config, [&](std::size_t swept, std::size_t population) noexcept {
                return ticks(swept, population);
            });
        }
    }

    static refined_t absorb(void* index, std::span<void*> donors, executor_t& executor, u32_t expansion_factor,
                            progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            growing_buffer<index_t*> typed {global_allocator};
            if (failed(typed.reserve_exactly(donors.size()))) return {{}, status_t::bad_alloc_k};
            for (void* const donor : donors) typed.push_back_reserved(static_cast<index_t*>(donor));
            typename index_t::refine_config_t config {};
            config.expansion_factor = expansion_factor;
            refined_t const refined = of(index).absorb(
                to_writable_span(typed), executor, config,
                [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
            for (std::size_t position = 0; position != donors.size(); ++position)
                if (!typed[position]) donors[position] = nullptr;
            return refined;
        }
    }

    static expected<dense::compact_result_t> compact(void* index, std::span<void* const> sources, executor_t& executor,
                                                     dense::compact_config_t config, progress_ticks_t& ticks) noexcept {
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

    static expected<void*> refinement_open(void* index, std::span<void* const> donors, executor_t& executor,
                                           u32_t expansion_factor) noexcept {
        if constexpr (viewed_k) return {nullptr, status_t::read_only_k};
        else {
            growing_buffer<index_t*> typed {global_allocator};
            if (failed(typed.reserve_exactly(donors.size()))) return {nullptr, status_t::bad_alloc_k};
            for (void* const donor : donors) typed.push_back_reserved(static_cast<index_t*>(donor));
            typename index_t::refine_config_t config {};
            config.expansion_factor = expansion_factor;
            // The plan borrows the landing slots, whose storage moves along with the buffer.
            auto [plan, status] = of(index).plan_absorb(to_writable_span(typed), executor, config);
            if (failed(status)) return {nullptr, status};
            dense_refinement<index_t>* const opened = new (std::nothrow)
                dense_refinement<index_t> {std::move(typed), std::move(plan)};
            if (!opened) return {nullptr, status_t::bad_alloc_k};
            return {opened, status_t::success_k};
        }
    }
    static status_t refinement_run(void*, void* refinement, executor_t& executor, progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else {
            dense_refinement<index_t>& refining = *static_cast<dense_refinement<index_t>*>(refinement);
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return prepared;
            auto const report = [&](std::size_t swept, std::size_t population) noexcept {
                return ticks(swept, population);
            };
            if constexpr (is_ram_index<index_type_>)
                return refining.plan.run(executor, to_writable_span(scratch.traversals), report);
            else return refining.plan.run(executor, report);
        }
    }
    static dense::refine_result_t refinement_running(void const* refinement) noexcept {
        if constexpr (viewed_k) return {};
        else return static_cast<dense_refinement<index_t> const*>(refinement)->plan.running();
    }
    static bool refinement_landed(void const* refinement, std::size_t donor) noexcept {
        if constexpr (viewed_k) return false;
        else return !static_cast<dense_refinement<index_t> const*>(refinement)->donors[donor];
    }
    static refined_t refinement_finish(void*, void* refinement, executor_t& executor) noexcept {
        if constexpr (viewed_k) return {{}, status_t::read_only_k};
        else {
            dense_refinement<index_t>& refining = *static_cast<dense_refinement<index_t>*>(refinement);
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            if constexpr (is_ram_index<index_type_>)
                return refining.plan.finish(executor, to_writable_span(scratch.traversals));
            else return refining.plan.finish(executor);
        }
    }
    static void refinement_free(void* refinement) noexcept {
        if constexpr (!viewed_k) delete static_cast<dense_refinement<index_t>*>(refinement);
    }

    static expected<dense::health_result_t> health(void* index, executor_t& executor) noexcept {
        return of(index).health(executor);
    }
    static status_t count(void* index, std::span<key_t const> keys, std::span<std::size_t> entries,
                          executor_t& executor) noexcept {
        return of(index).count(keys, entries, executor);
    }

    /** Rows in another scalar than the stored one are read in the stored one, which every face
     *  exports, and cast on the host by the serial kit, as a GPU face has no CPU class. */
    static status_t get(void* index, std::span<key_t const> keys, rows_span_t stored, rows_span_t outputs,
                        std::size_t wanted, std::span<std::size_t> entries, executor_t& executor) noexcept {
        if (status_t const got = of(index).get(keys, stored, wanted, entries, executor); failed(got)) return got;
        if (stored.data == outputs.data) return status_t::success_k;
        std::size_t landed = 0;
        for (std::size_t const found : entries) landed += found;
        for (std::size_t row = 0; row != min_of(landed, wanted); ++row)
            dense::export_row<dense::serial_elementwise_kit_t>(stored.row(row), stored.dtype, outputs, row);
        return status_t::success_k;
    }

    static walked_t search(void* index, rows_view_t queries, executor_t& executor, dense_walk_config_t walk,
                           keys_span_t keys, scores_span_t distances, counts_span_t counts) noexcept {
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        typename index_t::search_config_t const config = search_config_of(walk, executor);
        if constexpr (is_ram_index<index_type_>)
            return walked(of(index).search(queries, executor, to_writable_span(scratch.traversals), config, keys,
                                           distances, counts));
        else return walked(of(index).search(queries, executor, config, keys, distances, counts));
    }

    /** Narrows a walk by a host predicate, which a device walk cannot take, so a GPU index answers
     *  @c unsupported_k. */
    static walked_t search_if(void* index, rows_view_t queries, executor_t& executor, dense_walk_config_t walk,
                              keys_span_t keys, scores_span_t distances, counts_span_t counts,
                              predicate_t& predicate) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else {
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            typename index_t::search_config_t const config = search_config_of(walk, executor);
            if constexpr (is_ram_index<index_type_>)
                return walked(of(index).search_if(queries, executor, to_writable_span(scratch.traversals), config, keys,
                                                  distances, counts, predicate));
            else return walked(of(index).search_if(queries, executor, config, keys, distances, counts, predicate));
        }
    }
    static walked_t search_if_expensive(void* index, rows_view_t queries, executor_t& executor,
                                        dense_walk_config_t walk, keys_span_t keys, scores_span_t distances,
                                        counts_span_t counts, usearch_predicate_t predicate) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else {
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            return walked(search_with_expensive_predicate(of(index), queries, executor, scratch,
                                                          search_config_of(walk, executor), keys, distances, counts,
                                                          predicate));
        }
    }
    static walked_t multiply(void* index, rows_view_t left, executor_t& executor, dense_walk_config_t walk,
                             keys_span_t keys, scores_span_t products, counts_span_t counts) noexcept {
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        typename index_t::search_config_t const config = search_config_of(walk, executor);
        if constexpr (is_ram_index<index_type_>)
            return walked(of(index).multiply(left, executor, to_writable_span(scratch.traversals), config, keys,
                                             products, counts));
        else return walked(of(index).multiply(left, executor, config, keys, products, counts));
    }
    static walked_t multiply_if(void* index, rows_view_t left, executor_t& executor, dense_walk_config_t walk,
                                keys_span_t keys, scores_span_t products, counts_span_t counts,
                                predicate_t& predicate) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else {
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            typename index_t::search_config_t const config = search_config_of(walk, executor);
            if constexpr (is_ram_index<index_type_>)
                return walked(of(index).multiply_if(left, executor, to_writable_span(scratch.traversals), config, keys,
                                                    products, counts, predicate));
            else return walked(of(index).multiply_if(left, executor, config, keys, products, counts, predicate));
        }
    }

    /** Late interaction of the queries @p query_offsets cut @p queries into; a GPU index walks
     *  uniform queries alone, answering @c unsupported_k for a scan, ragged offsets or a zero
     *  @p wanted. */
    static walked_t maxsim(void* index, rows_view_t queries, std::span<std::size_t const> query_offsets,
                           executor_t& executor, dense_walk_config_t walk, std::size_t wanted, counts_span_t scored,
                           maxsim_landing_t landing) noexcept {
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        typename index_t::search_config_t const config = search_config_of(walk, executor);
        if constexpr (is_vram_index<index_type_>) {
            // The device reduction ranks into a rectangle on this thread, then lands in the one
            // run.
            std::size_t const queries_count = query_offsets.size() - 1;
            std::size_t const rows_per_query = query_offsets[1] - query_offsets[0];
            for (std::size_t query = 0; query != queries_count; ++query)
                if (query_offsets[query + 1] - query_offsets[query] != rows_per_query)
                    return {{}, status_t::unsupported_k};
            if (config.traversal != traversal_t::walk_k || !wanted) return {{}, status_t::unsupported_k};
            growing_buffer<key_t, thread_allocator_t> keys {landing.staging};
            growing_buffer<f64_t, thread_allocator_t> scores {landing.staging};
            growing_buffer<std::size_t, thread_allocator_t> counts {landing.staging};
            if (failed(keys.resize_uninitialized(queries_count * wanted)) ||
                failed(scores.resize_uninitialized(queries_count * wanted)) ||
                failed(counts.resize_uninitialized(queries_count)))
                return {{}, status_t::bad_alloc_k};
            expected<typename index_t::search_result_t> const reduced = of(index).maxsim(
                queries.slice(query_offsets.front(), query_offsets.back() - query_offsets.front()), rows_per_query,
                executor, config, keys_span_t {keys.data(), queries_count, wanted},
                scores_span_t {scores.data(), queries_count, wanted}, to_writable_span(counts), scored);
            for (std::size_t query = 0; reduced && query != queries_count; ++query)
                for (std::size_t rank = 0; rank != counts[query]; ++rank)
                    if (failed(landing(0, query, keys[query * wanted + rank], scores[query * wanted + rank])))
                        return {walked(reduced).value, status_t::bad_alloc_k};
            return walked(reduced);
        }
        else if constexpr (is_ram_index<index_type_>)
            return walked(of(index).maxsim(queries, query_offsets, executor, to_writable_span(scratch.traversals),
                                           config, wanted, scored, landing));
        else return walked(of(index).maxsim(queries, query_offsets, executor, config, wanted, scored, landing));
    }
    static walked_t maxsim_if(void* index, rows_view_t queries, std::span<std::size_t const> query_offsets,
                              executor_t& executor, dense_walk_config_t walk, std::size_t wanted, counts_span_t scored,
                              maxsim_landing_t landing, predicate_t& predicate) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else {
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            typename index_t::search_config_t const config = search_config_of(walk, executor);
            if constexpr (is_ram_index<index_type_>)
                return walked(of(index).maxsim_if(queries, query_offsets, executor,
                                                  to_writable_span(scratch.traversals), config, wanted, scored, landing,
                                                  predicate));
            else
                return walked(
                    of(index).maxsim_if(queries, query_offsets, executor, config, wanted, scored, landing, predicate));
        }
    }
    static walked_t attend(void* index, rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                           f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                           executor_t& executor, dense_walk_config_t walk) noexcept {
        scratch_t scratch {executor};
        if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        typename index_t::search_config_t const config = search_config_of(walk, executor);
        if constexpr (is_ram_index<index_type_>)
            return walked(of(index).attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor,
                                           to_writable_span(scratch.traversals), config));
        else
            return walked(
                of(index).attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config));
    }
    static walked_t attend_if(void* index, rows_view_t queries, std::size_t head_count, value_lookup_t value_lookup,
                              f32_t score_scale, rows_span_t outputs, std::span<dense::softmax_state_t> carry,
                              executor_t& executor, dense_walk_config_t walk, predicate_t& predicate) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, status_t::unsupported_k};
        else {
            scratch_t scratch {executor};
            if (status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            typename index_t::search_config_t const config = search_config_of(walk, executor);
            if constexpr (is_ram_index<index_type_>)
                return walked(of(index).attend_if(queries, head_count, value_lookup, score_scale, outputs, carry,
                                                  executor, to_writable_span(scratch.traversals), config, predicate));
            else
                return walked(of(index).attend_if(queries, head_count, value_lookup, score_scale, outputs, carry,
                                                  executor, config, predicate));
        }
    }

    /** Opens a cursor over every row; a GPU index reads its rows back by key or search and answers
     *  @c unsupported_k. */
    static expected<void*> cursor_open(void* index) noexcept {
        if constexpr (is_vram_index<index_type_>) return {nullptr, status_t::unsupported_k};
        else {
            typename index_t::cursor_t* const opened = new (std::nothrow) typename index_t::cursor_t(of(index));
            if (!opened) return {nullptr, status_t::bad_alloc_k};
            return {opened, status_t::success_k};
        }
    }
    static expected<std::size_t> cursor_next(void* index, void* cursor, std::span<key_t> keys,
                                             executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>) return {0, status_t::unsupported_k};
        else {
            typename index_t::cursor_t& walk = *static_cast<typename index_t::cursor_t*>(cursor);
            if constexpr (viewed_k) {
                auto const [scanned, status] = walk.read(of(index), keys, rows_span_t {}, executor);
                return {scanned.count, status};
            }
            else return walk.read(of(index), keys, rows_span_t {});
        }
    }
    static void cursor_free(void* cursor) noexcept {
        if constexpr (!is_vram_index<index_type_>) delete static_cast<typename index_t::cursor_t*>(cursor);
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
    static status_t load_buffer(void* index, bytes_view_t buffer, executor_t& executor,
                                progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return status_t::read_only_k;
        else return of(index).load(buffer, load_config_t {}, executor, ticks);
    }

    /** The verb table of the variant, every entry one of the bodies above. */
    static constexpr verbs_t verbs() noexcept {
        return {
            .dtype = index_t::vector_dtype_k,
            .cpu_class = cpu_class_t::unknown_k,
            .gpu_class = gpu_class_t::unknown_k,
            .create = &create,
            .destroy = &destroy,
            .bound_shards = &bound_shards,
            .emplace = &emplace,
            .view = &view,
            .kernel = &kernel,
            .device = &device,
            .shards_count = &shards_count,
            .dimensions = &dimensions,
            .connectivity = &connectivity,
            .size = &size,
            .capacity = &capacity,
            .memory_usage = &memory_usage,
            .storage_requirements = &storage_requirements,
            .reserve = &reserve,
            .clear = &clear,
            .insert = &insert,
            .refine = &refine,
            .absorb = &absorb,
            .compact = &compact,
            .refinement_open = &refinement_open,
            .refinement_run = &refinement_run,
            .refinement_running = &refinement_running,
            .refinement_landed = &refinement_landed,
            .refinement_finish = &refinement_finish,
            .refinement_free = &refinement_free,
            .health = &health,
            .count = &count,
            .get = &get,
            .search = &search,
            .search_if = &search_if,
            .search_if_expensive = &search_if_expensive,
            .multiply = &multiply,
            .multiply_if = &multiply_if,
            .maxsim = &maxsim,
            .maxsim_if = &maxsim_if,
            .attend = &attend,
            .attend_if = &attend_if,
            .cursor_open = &cursor_open,
            .cursor_next = &cursor_next,
            .cursor_free = &cursor_free,
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
 *  @brief Whether a class compiles its own variant of @p dtype_ in @p home_.
 *
 *  On RAM it does where its 4-way or tiles kit differs from the class it degrades to, or its
 *  ForkUnion mask does; on NVMe where its kit alone differs, as a storage walk carries no waiter or
 *  atomic reference; on a device always, since each class's unit alone carries kernels for its
 *  generation. The serial floor always compiles its variants, and a variant a class drops is its
 *  parent's, which the ladder reaches on the next rung.
 */
template <dense_home_t home_, nk_dtype_t dtype_, auto class_>
static consteval bool variant_advances() noexcept {
    if constexpr (home_ == dense_home_t::ram_k || home_ == dense_home_t::nvme_k) {
        if constexpr (class_ == cpu_class_t::serial_k) return true;
        else {
            constexpr cpu_class_t parent_k = parent_class_of(class_);
            bool const kits_differ = !std::is_same_v<quads_kit_of<dtype_, class_>, quads_kit_of<dtype_, parent_k>> ||
                                     !std::is_same_v<tiles_kit_of<dtype_, class_>, tiles_kit_of<dtype_, parent_k>>;
            if constexpr (home_ == dense_home_t::nvme_k) return kits_differ;
            else return kits_differ || forkunion_capabilities_of<class_>() != forkunion_capabilities_of<parent_k>();
        }
    }
    else return true;
}

/** Variants of the dtypes in the list that @p class_ compiles in @p home_. */
template <dense_home_t home_, auto class_, nk_dtype_t... dtypes_>
static consteval std::size_t tables_count(dtype_list<dtypes_...>) noexcept {
    return (std::size_t {variant_advances<home_, dtypes_, class_>()} + ... + 0);
}

/** The verb tables of those variants, in ladder order; a dropped variant's index is never
 *  instantiated. */
template <dense_home_t home_, auto class_, nk_dtype_t... dtypes_>
static consteval std::array<dense_verbs<home_>, tables_count<home_, class_>(dtype_list<dtypes_...> {})> tables_for(
    dtype_list<dtypes_...>) noexcept {
    std::array<dense_verbs<home_>, tables_count<home_, class_>(dtype_list<dtypes_...> {})> tables {};
    std::size_t position = 0;
    auto place = [&]<nk_dtype_t dtype_>() {
        if constexpr (variant_advances<home_, dtype_, class_>()) {
            dense_verbs<home_> table =
                specialized_index<typename dense_indexes_of<home_, dtype_, class_>::type>::verbs();
            if constexpr (home_ == dense_home_t::ram_k || home_ == dense_home_t::nvme_k) table.cpu_class = class_;
            else table.gpu_class = class_;
            tables[position++] = table;
        }
    };
    (place.template operator()<dtypes_>(), ...);
    return tables;
}

template <dense_home_t home_, auto class_>
dense_verbs<home_> const* dense_class_tables<home_, class_>::find(nk_dtype_t dtype) noexcept {
    static constexpr std::array<dense_verbs<home_>, tables_count<home_, class_>(dense_dtypes_t {})> tables_k =
        tables_for<home_, class_>(dense_dtypes_t {});
    for (dense_verbs<home_> const& table : tables_k)
        if (table.dtype == dtype) return &table;
    return nullptr;
}

#pragma endregion Class Tables
