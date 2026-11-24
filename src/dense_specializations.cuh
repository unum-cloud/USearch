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

#include <array>   // `std::array` - a class unit's variants
#include <new>     // `std::nothrow`
#include <utility> // `std::move` - an opened engine plan onto the heap

#include "usearch/dense/nvme_reshard.hpp" // `plan_refine`, `plan_reshard` - a loaded NVMe index's plans
#include "usearch/dense/ram_reshard.hpp"  // `plan_reshard` - RAM indexes' reshard
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

namespace util = unum::usearch::utilities;

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
    dense::ram_indexes<util::uuid128_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                       elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>,
                       waiter_of<cpu_class_>, atomic_ref_for_cpu_class<cpu_class_>::template type,
                       typename cache_hints_for_cpu_class<cpu_class_>::type>;

/** One NVMe variant: scoring and the words its threads share compile per CPU class, while the
 *  metric comes from the parts' headers and the transport is the executor's runtime choice. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using nvme_indexes =
    dense::nvme_indexes<util::uuid128_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                        elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>, visits_kit_of<cpu_class_>,
                        default_io_queue_t, atomic_ref_for_cpu_class<cpu_class_>::template type, waiter_of<cpu_class_>>;

/** The start and end of every dtype's device code one GPU class unit embeds, in pairs in
 *  @c dense_dtypes_t order, as the table CMake writes for the class; each VRAM class unit
 *  specializes it. */
template <gpu_class_t gpu_class_>
std::span<util::byte_t const* const> bundle_bounds() noexcept;

/** The device code of one GPU class and dtype, between the bounds of the dtype's position. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
inline static util::bytes_view_t bundled_code() noexcept {
    std::span<util::byte_t const* const> const bounds = bundle_bounds<gpu_class_>();
    std::size_t const position = dense_dtypes_t::index_of(dtype_);
    return util::bytes_view_t {bounds[2 * position], bounds[2 * position + 1]};
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
using vram_nvidia_indexes = dense::vram_nvidia_indexes<util::uuid128_t, dtype_, cuda_fatbin<gpu_class_, dtype_>,
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
using vram_amd_indexes = dense::vram_amd_indexes<util::uuid128_t, dtype_, hip_code_object<gpu_class_, dtype_>,
                                                 fu::preferred_cache_hints_t, fu::standard_yield_t>;
#endif

#if USEARCH_ARCH_METAL_

/** One Apple variant: the class names the rung and the capability ceiling its kernels compile for
 *  at the first @c reserve. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_apple_indexes =
    dense::vram_apple_indexes<util::uuid128_t, dtype_, numkong_capabilities_of(gpu_class_), fu::standard_yield_t>;
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
    util::status_t prepare(typename indexes_type_::executor_t&) noexcept { return util::status_code_t::success_k; }
};

template <typename indexes_type_>
    requires is_ram_index<indexes_type_>
struct dense_scratch<indexes_type_> {
    using traversal_context_t = typename indexes_type_::traversal_context_t;

    util::growing_buffer<traversal_context_t, thread_allocator_t> traversals;

    explicit dense_scratch(util::ram_executor_t& executor) noexcept
        : traversals(thread_allocator_t {executor.arena(0)}) {}

    /** Rewinds every arena, keeping the high-water mark of earlier calls, and carves one context
     *  per thread; the verb sizes them. */
    util::status_t prepare(util::ram_executor_t& executor) noexcept {
        if (util::status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        if (failed(traversals.reserve_exactly(executor.size()))) return util::status_code_t::out_of_memory_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            traversals.push_back_reserved(traversal_context_t {thread_allocator_t {executor.arena(thread)}});
        return util::status_code_t::success_k;
    }
};

/** Searches @p index under an expensive @p predicate through a family of per-thread predicates
 *  gathering what the traversal @p config names hands them; the engine opens the family and
 *  reserves it for the widest round it settles. */
template <typename index_type_>
inline static util::expected<typename index_type_::search_result_t> search_with_expensive_predicate(
    index_type_& index, dense::rows_view_t queries, ranked_hints_view_t hints, ranked_matches_span_t matches,
    typename index_type_::search_config_t config, usearch_predicate_t predicate,
    typename index_type_::executor_t& executor, dense_scratch<index_type_>& scratch) noexcept {
    auto const search_if = [&]<typename predicates_type_>(predicates_type_& predicates) noexcept {
        if constexpr (is_ram_index<index_type_>)
            return index.search_if(queries, matches, config, predicates, executor, to_writable_span(scratch.traversals),
                                   hints);
        else return index.search_if(queries, matches, config, predicates, executor, {}, hints);
    };
    if (config.traversal == util::traversal_t::scan_k) {
        expensive_predicates<typename index_type_::scan_payload_t> scanned {predicate};
        return search_if(scanned);
    }
    expensive_predicates<typename index_type_::walk_payload_t> walked {predicate};
    return search_if(walked);
}

/** The engine plan a refine of @p index_type_ opens: the tower's own, or a loaded NVMe index's
 *  part by part rewrite. */
template <typename index_type_>
struct refine_plan_of {
    using type = typename index_type_::refine_plan_t;
};
template <typename index_type_>
    requires is_nvme_index<index_type_>
struct refine_plan_of<index_type_> {
    using type = dense::nvme_refine_plan<index_type_>;
};

/** The engine plan a reshard of @p index_type_ opens: a device tower's own, or the freestanding
 *  RAM and NVMe ones. */
template <typename index_type_>
struct reshard_plan_of {
    using type = typename index_type_::reshard_plan_t;
};
template <typename index_type_>
    requires is_ram_index<index_type_>
struct reshard_plan_of<index_type_> {
    using type = dense::ram_reshard_plan<index_type_>;
};
template <typename index_type_>
    requires is_nvme_index<index_type_>
struct reshard_plan_of<index_type_> {
    using type = dense::nvme_reshard_plan<index_type_>;
};

/** Whether @p index_type_ walks its entries through a cursor. */
template <typename index_type_>
concept walks_cursors = requires { typename index_type_::cursor_t; };

/** Whether @p index_type_ loads a checkpoint where it sits, a device's included. */
template <typename index_type_>
concept loads_checkpoint_sources = requires(index_type_& index, util::checkpoint_source_t source,
                                            typename index_type_::executor_t& executor, progress_ticks_t& ticks) {
    index.load(source, util::load_config_t {}, executor, ticks);
};

/** Whether @p index_type_ answers if its media read through SPDK, as a loaded NVMe index does. */
template <typename index_type_>
concept reports_spdk_reads = requires(index_type_ const& index) { index.reads_spdk(); };

/**
 *  @brief One variant's bodies, every one a static member its verb table points at, taking the
 *      index as @c void* and casting it back to @p index_type_.
 *
 *  An NVMe view serves its checkpoint in place, so its mutating bodies answer
 *  @c usearch_status_read_only_k, and a loaded NVMe index, whose parts change only by its plans,
 *  @c unsupported_k; a device index answers @c unsupported_k for what only a host walk takes, as
 *  host predicates and cursors.
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
    using refine_plan_t = typename refine_plan_of<index_type_>::type;
    using reshard_plan_t = typename reshard_plan_of<index_type_>::type;

    static constexpr bool viewed_k = is_nvme_index<index_type_>;

    /** Whether the engine sizes a pass's tiles from @c memory_limit; Apple's holds twice the
     *  neighbors' bytes instead, so it refuses a limit. */
    static constexpr bool limits_memory_k = vendored_residency_of<executor_t> != vendored_residency_t::vram_apple_k;

    static index_t& of(void* index) noexcept { return *static_cast<index_t*>(index); }
    static index_t const& of(void const* index) noexcept { return *static_cast<index_t const*>(index); }

    /** What a write this index cannot take answers: an NVMe index's own refusal, which a view and a
     *  loaded index spell apart, and @c unsupported_k elsewhere. */
    static util::status_t refuses_writes(void const* index) noexcept {
        if constexpr (requires { of(index).refuses_writes(); }) return of(index).refuses_writes();
        else return util::status_code_t::unsupported_k;
    }

    /** Runs the reader @p read between the part swaps of an index that swaps parts, and once on
     *  every other index. */
    template <typename read_type_>
    static std::invoke_result_t<read_type_&> between_swaps(void const* index, read_type_&& read) noexcept {
        if constexpr (requires { of(index).read_between_swaps(read); }) return of(index).read_between_swaps(read);
        else return read();
    }

    /** The traversal stats of one engine result, or its failure. */
    template <typename result_type_>
    static util::expected<util::traversal_stats_t> stats_of(util::expected<result_type_> const& result) noexcept {
        return {traversal_stats_of(result.value), result.status};
    }

    /** The engine's search config of the call's @p search. */
    static search_config_t search_config_of(::search_config_t search) noexcept {
        search_config_t config {};
        config.expansion_factor = search.expansion_factor;
        config.traversal = search.traversal;
        return config;
    }

    /** The engine's refine config of the call's @p refine, which only a mutable index spells. */
    template <typename engine_type_ = index_type_>
    static typename engine_type_::refine_config_t refine_config_of(::refine_config_t refine) noexcept {
        typename engine_type_::refine_config_t config {};
        config.expansion_factor = refine.expansion_factor;
        config.memory_limit = refine.memory_limit;
        if constexpr (requires { config.cache_bytes; }) config.cache_bytes = refine.cache_bytes;
        return config;
    }

    /** Hands @p walk the alternative @p predicate holds, as <tt>walk(narrowing)</tt>; a device walk
     *  answers @c unsupported_k for any predicate, so it never instantiates a narrowed walk. */
    template <typename walk_type_>
    static util::expected<util::traversal_stats_t> with_predicate(predicate_t& predicate, walk_type_&& walk) noexcept {
        return std::visit(
            [&]<typename narrowing_type_>(
                narrowing_type_& narrowing) noexcept -> util::expected<util::traversal_stats_t> {
                if constexpr (std::is_same_v<narrowing_type_, no_predicate_t> || !is_vram_index<index_type_>)
                    return walk(narrowing);
                else return {{}, util::status_code_t::unsupported_k};
            },
            predicate);
    }

    /** The C layer's save config, whose contract the residency picks: device storage cannot be
     *  settlement-probed, so a GPU handle certifies. */
    static util::save_config_t save_config_of() noexcept {
        util::save_config_t config;
        if constexpr (is_vram_index<index_type_>) config.synchronization = util::synchronization_t::immutable_k;
        return config;
    }

    static void* create(nk_kernel_kind_t kernel, dense::geometry_t geometry,
                        util::forkunion_allocator_t tables) noexcept {
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
    static util::status_t emplace(void* index, arena_residency_t residency,
                                  fu_memory_domain_id_t memory_domain) noexcept {
        if constexpr (viewed_k) return util::status_code_t::read_only_k;
        else if constexpr (is_ram_index<index_type_>) return of(index).emplace(memory_domain);
        else {
            // Init names devices and no executor yet. A Metal shard keeps a queue of its own, since
            // Metal orders work only within one queue; CUDA and HIP name the device by its null
            // legacy stream and open nothing.
            int const device = static_cast<int>(residency.device);
            std::span<int const> const devices {&device, 1};
            auto [executor, status] = std::is_same_v<executor_t, util::metal_executor_t>
                                          ? executor_t::spawn(
                                                driver_of(gpu_drivers(), std::type_identity<executor_t> {}), devices)
                                          : executor_t::on_default_streams(
                                                driver_of(gpu_drivers(), std::type_identity<executor_t> {}), devices);
            return failed(status) ? status : of(index).emplace(executor.stream(0));
        }
    }
    static util::status_t view(void* index, std::string_view path, std::span<std::string_view const> paths,
                               usearch_ram_options_t const* ram, usearch_executor_t executor) noexcept {
        if constexpr (viewed_k) {
            executor_t* const engines = executor ? std::get_if<executor_t>(&executor->executor) : nullptr;
            if (ram && !engines) return util::status_code_t::invalid_argument_k;
            if (util::status_t const viewed = of(index).view(path, paths); failed(viewed) || !ram) return viewed;
            return of(index).hold_keys(util::forkunion_allocator_t {tables_domain(*ram)}, ram->memory_limit, *engines);
        }
        else return util::status_code_t::unsupported_k;
    }
    static util::status_t load_nvme(void* index, std::string_view path, usearch_ram_options_t const* ram,
                                    usearch_executor_t executor) noexcept
        requires is_nvme_index<index_type_>
    {
        executor_t* const engines = executor ? std::get_if<executor_t>(&executor->executor) : nullptr;
        if (ram && !engines) return util::status_code_t::invalid_argument_k;
        if (util::status_t const loaded = of(index).load(path); failed(loaded) || !ram) return loaded;
        return of(index).hold_keys(util::forkunion_allocator_t {tables_domain(*ram)}, ram->memory_limit, *engines);
    }
    static util::status_t load_nvme(void*, std::string_view, usearch_ram_options_t const*,
                                    usearch_executor_t) noexcept {
        return util::status_code_t::unsupported_k;
    }

    static nk_kernel_kind_t kernel(void const* index) noexcept { return of(index).kernel(); }
    static dense::geometry_t geometry(void const* index) noexcept {
        return {of(index).dimensions(), static_cast<util::connectivity_t>(of(index).connectivity())};
    }
    static arena_residency_t residency(void const* index, std::size_t shard) noexcept {
        if constexpr (is_vram_index<index_type_>)
            return vram_residency(static_cast<std::size_t>(of(index).shard(shard).device_specs().ordinal));
        else return ram_residency_k;
    }
    static std::size_t shards_count(void const* index) noexcept { return of(index).shards_count(); }
    static bool reads_spdk(void const* index) noexcept {
        if constexpr (reports_spdk_reads<index_type_>) return of(index).reads_spdk();
        else return false;
    }
    static usearch_load_path_t load_path(void const* index) noexcept {
        if constexpr (requires { of(index).loaded_through(); }) return usearch_load_path_of(of(index).loaded_through());
        else return usearch_load_path_none_k;
    }
    static std::size_t size(void const* index) noexcept { return of(index).size(); }
    static std::size_t capacity(void const* index) noexcept { return of(index).capacity(); }
    static std::size_t memory_usage(void const* index) noexcept { return of(index).memory_usage(); }

    /** A view already is its checkpoint, so it answers @c unsupported_k. */
    static util::expected<std::size_t> storage_requirements(void const* index) noexcept {
        if constexpr (viewed_k) return {0, util::status_code_t::unsupported_k};
        else return of(index).storage_requirements(save_config_of());
    }

    static util::status_t serves_writes(void const* index) noexcept {
        if constexpr (viewed_k) return refuses_writes(index);
        else return util::status_code_t::success_k;
    }
    static util::status_t reserve(void* index, std::size_t capacity) noexcept {
        if constexpr (viewed_k) return refuses_writes(index);
        else return of(index).reserve({capacity}).status;
    }
    static util::status_t clear(void* index) noexcept {
        if constexpr (viewed_k) return refuses_writes(index);
        else return of(index).clear();
    }

    /** Writes @p rows under @p keys through one plan, a task per chunk, and publishes them all at
     *  once. */
    static util::expected<util::traversal_stats_t> insert(void* index, keys_view_t keys, rows_view_t rows,
                                                          insert_config_t insert, executor_t& executor,
                                                          progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return {{}, refuses_writes(index)};
        else {
            typename index_t::insert_config_t config {};
            config.expansion_factor = insert.expansion_factor;
            std::size_t const count = keys.size();
            // Only a handler splits the call, and one plan still publishes every slice at once.
            std::size_t const chunk = ticks.slice();
            scratch_t scratch {executor};
            if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            auto [plan, status] = of(index).plan_insert(count, executor, config);
            if (failed(status)) return {{}, status};
            for (std::size_t inserted = 0; inserted != count;) {
                std::size_t const taking = util::min_of(chunk, count - inserted);
                util::status_t const stepped = [&]() noexcept {
                    if constexpr (is_ram_index<index_type_>)
                        return plan.insert(inserted, keys[nk::range(inserted, inserted + taking)],
                                           rows.slice(inserted, taking), to_writable_span(scratch.traversals));
                    else
                        return plan.insert(inserted, keys[nk::range(inserted, inserted + taking)],
                                           rows.slice(inserted, taking));
                }();
                if (failed(stepped)) {
                    // A refused RAM task dispensed nothing, so the dropped plan releases every
                    // claim.
                    if constexpr (is_ram_index<index_type_>) return {{}, stepped};
                    else break;
                }
                inserted += taking;
                // Every entry is claimed before the first report and none can be withdrawn, so a
                // cancel goes unheard.
                ticks(inserted, count);
            }
            // After a failed device task, `finish` publishes nothing and reports the failure the
            // plan kept.
            return stats_of(plan.finish());
        }
    }

    /** Opens one round over every shard, or a loaded NVMe index's pass over its parts; a view holds
     *  no writer, so it answers @c read_only_k. */
    static util::expected<void*> refine_plan_open(void* index, ::refine_config_t config,
                                                  executor_t& executor) noexcept {
        if (!limits_memory_k && config.memory_limit) return {nullptr, util::status_code_t::unsupported_k};
        auto [plan, status] = [&]() noexcept -> util::expected<refine_plan_t> {
            if constexpr (is_nvme_index<index_type_>)
                return dense::plan_refine(of(index), refine_config_of(config), executor);
            else return of(index).plan_refine(refine_config_of(config), executor);
        }();
        if (failed(status)) return {nullptr, status};
        refine_plan_t* const opened = new (std::nothrow) refine_plan_t(std::move(plan));
        if (!opened) return {nullptr, util::status_code_t::out_of_memory_k};
        return {opened, util::status_code_t::success_k};
    }

    /** What a reshard refuses of @p options: another residency's placement, which names nothing the
     *  handles hold, or named domains, devices or drives, which would place shards anew, or a
     *  memory limit, which a reshard cannot honour; a class ceiling the glue resolved passes. */
    static util::status_t refuses_placements(reshard_options_t options) noexcept {
        if constexpr (is_ram_index<index_type_>) {
            if (options.vram || options.nvme) return util::status_code_t::config_mismatch_k;
            if (options.ram && (options.ram->memory_domains_count || options.ram->memory_limit))
                return util::status_code_t::unsupported_k;
        }
        else if constexpr (is_nvme_index<index_type_>) {
            if (options.ram || options.vram) return util::status_code_t::config_mismatch_k;
            if (options.nvme && options.nvme->paths_count) return util::status_code_t::unsupported_k;
        }
        else {
            if (options.ram || options.nvme) return util::status_code_t::config_mismatch_k;
            if (options.vram && options.vram->devices_count) return util::status_code_t::unsupported_k;
        }
        return util::status_code_t::success_k;
    }

    /** The engine's reshard config of the call's @p options: its rounds, the capacity its own
     *  residency's placement names, and the seed of a split's part roots. */
    static dense::reshard_config_t reshard_config_of(reshard_options_t options) noexcept {
        dense::reshard_config_t config;
        config.refine = refine_config_of(options.refine);
        if constexpr (is_ram_index<index_type_>) config.shard_capacity = options.ram ? options.ram->shard_capacity : 0;
        else if constexpr (is_vram_index<index_type_>)
            config.shard_capacity = options.vram ? options.vram->shard_capacity : 0;
        config.seed = options.seed;
        return config;
    }

    /** The typed indexes behind @p indexes, or an empty buffer when it found no memory. */
    static util::growing_buffer<index_t*> indexes_of(variants_view_t indexes) noexcept {
        util::growing_buffer<index_t*> typed {util::global_allocator};
        if (failed(typed.reserve_exactly(indexes.size()))) return typed;
        for (std::size_t position = 0; position != indexes.size(); ++position)
            typed.push_back_reserved(static_cast<index_t*>(indexes[position]));
        return typed;
    }

    /** Opens a reshard of @p sources into @p targets, which copies both lists. */
    static util::expected<void*> reshard_plan_open(variants_view_t sources, variants_view_t targets,
                                                   reshard_options_t options, executor_t& executor) noexcept {
        if (util::status_t const refused = refuses_placements(options); failed(refused)) return {nullptr, refused};
        util::growing_buffer<index_t*> const typed_sources = indexes_of(sources);
        util::growing_buffer<index_t*> const typed_targets = indexes_of(targets);
        if (typed_sources.size() != sources.size() || typed_targets.size() != targets.size())
            return {nullptr, util::status_code_t::out_of_memory_k};
        auto [plan, status] = dense::plan_reshard(to_span(typed_sources), to_span(typed_targets), executor,
                                                  reshard_config_of(options));
        if (failed(status)) return {nullptr, status};
        reshard_plan_t* const opened = new (std::nothrow) reshard_plan_t(std::move(plan));
        if (!opened) return {nullptr, util::status_code_t::out_of_memory_k};
        return {opened, util::status_code_t::success_k};
    }

    /** Runs @p plan 's pending phases until its round ends or @p ticks cancels: a RAM plan over one
     *  traversal context per thread, an NVMe plan task by task with a tick at every barrier, and a
     *  device plan on its streams. */
    template <typename plan_type_>
    static util::status_t run_plan(void* plan, executor_t& executor, progress_ticks_t& ticks) noexcept {
        plan_type_& typed = *static_cast<plan_type_*>(plan);
        auto const report = [&](std::size_t swept, std::size_t population) noexcept {
            return ticks(swept, population);
        };
        if constexpr (is_ram_index<index_type_>) {
            scratch_t scratch {executor};
            if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return prepared;
            return typed.run(executor, to_writable_span(scratch.traversals), report);
        }
        else if constexpr (is_nvme_index<index_type_>) {
            while (true) {
                util::status_t ran = util::status_code_t::success_k;
                for (typename plan_type_::task_t const& task : typed.tasks())
                    ran = failed(ran) ? ran : typed.run(task, executor);
                if (failed(ran)) return ran;
                std::size_t const completed = typed.completed(), total = typed.total();
                bool const following = typed.advance();
                if (total && report(completed, total) == util::continuation_t::cancel_k)
                    return util::status_code_t::cancelled_k;
                if (!following) return util::status_code_t::success_k;
            }
        }
        else return typed.run(executor, report);
    }

    /** What a plan's closing @p finished result did to the graph: a reshard's rounds, or a refine's
     *  whole report. */
    template <typename result_type_>
    static util::expected<util::refine_report_t> graph_of(util::expected<result_type_> const& finished) noexcept {
        if constexpr (requires { finished.value.graph; }) return {finished.value.graph, finished.status};
        else return {finished.value, finished.status};
    }

    /** Closes @p plan as its operation's finish does, a RAM one over one traversal context per
     *  thread, and reports what it did to the graph. */
    template <typename plan_type_>
    static util::expected<util::refine_report_t> finish_plan(void* plan, executor_t& executor) noexcept {
        plan_type_& typed = *static_cast<plan_type_*>(plan);
        if constexpr (is_ram_index<index_type_>) {
            scratch_t scratch {executor};
            if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
            return graph_of(typed.finish(executor, to_writable_span(scratch.traversals)));
        }
        else return graph_of(typed.finish(executor));
    }

    template <typename plan_type_>
    static void next_round(void* plan) noexcept {
        static_cast<plan_type_*>(plan)->next_round();
    }
    template <typename plan_type_>
    static usearch_phase_progress_t plan_phase(void const* plan) noexcept {
        return phase_of(*static_cast<plan_type_ const*>(plan));
    }
    template <typename plan_type_>
    static util::refine_report_t plan_running(void const* plan) noexcept {
        return static_cast<plan_type_ const*>(plan)->running();
    }
    template <typename plan_type_>
    static void free_plan(void* plan) noexcept {
        delete static_cast<plan_type_*>(plan);
    }

    /** The entries every plan of @p plan_type_ runs through. */
    template <typename plan_type_>
    static constexpr plan_verbs<executor_t> plan_verbs_for() noexcept {
        return {&run_plan<plan_type_>,     &next_round<plan_type_>,  &plan_phase<plan_type_>,
                &plan_running<plan_type_>, &finish_plan<plan_type_>, &free_plan<plan_type_>};
    }

    static util::expected<util::reachability_t> health(void* index, executor_t& executor) noexcept {
        return between_swaps(index, [&]() noexcept { return of(index).health(executor); });
    }
    static util::status_t count(void* index, std::span<key_t const> keys, std::span<std::size_t> entries,
                                executor_t& executor) noexcept {
        return between_swaps(index, [&]() noexcept { return of(index).count(keys, entries, executor); });
    }

    /** Every tower casts the stored scalar into the rows' on its own side, a GPU one on the device;
     *  Apple's reads the stored scalar alone. */
    static util::status_t get(void* index, std::span<key_t const> keys, rows_span_t outputs,
                              std::span<std::size_t> entries, std::size_t wanted, executor_t& executor) noexcept {
        return between_swaps(index, [&]() noexcept { return of(index).get(keys, outputs, entries, wanted, executor); });
    }

    /** Ranks each query's nearest entries, every walk seeded by its query's @p hints; a cheap
     *  predicate is asked a quad at a time, an expensive one once per round each thread settles. A
     *  device walk answers @c unsupported_k for hints, as for a predicate. */
    static util::expected<util::traversal_stats_t> search(void* index, rows_view_t queries, ranked_hints_view_t hints,
                                                          ranked_matches_span_t matches, ::search_config_t search,
                                                          predicate_t predicate, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (!hints.counts.empty()) return {{}, util::status_code_t::unsupported_k};
        scratch_t scratch {executor};
        if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        search_config_t const config = search_config_of(search);
        return between_swaps(index, [&]() noexcept {
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
                    return stats_of(of(index).search(queries, matches, config, executor,
                                                     to_writable_span(scratch.traversals), hints));
                else return stats_of(of(index).search(queries, matches, config, executor, {}, hints));
            });
        });
    }

    /** Keeps the largest products of each left row on the tape; any predicate is asked a quad at a
     *  time, and a GPU index answers @c unsupported_k for a scan. */
    static util::expected<util::traversal_stats_t> multiply(void* index, rows_view_t left, products_tape_t products,
                                                            std::span<std::size_t> scored, ::search_config_t search,
                                                            predicate_t predicate, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (search.traversal == util::traversal_t::scan_k) return {{}, util::status_code_t::unsupported_k};
        scratch_t scratch {executor};
        if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        search_config_t const config = search_config_of(search);
        return between_swaps(index, [&]() noexcept {
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
        });
    }

    /** Late interaction of the queries the offsets cut, packed on the tape, every row's walks
     *  seeded by its query's hints; a GPU index walks uniform queries alone, answering
     *  @c unsupported_k for a scan, ragged offsets or hints. */
    static util::expected<util::traversal_stats_t> maxsim(void* index, maxsim_queries_view_t queries,
                                                          products_tape_t matches, std::span<std::size_t> scored,
                                                          std::size_t wanted, ::search_config_t search,
                                                          predicate_t predicate, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (!queries.hints.counts.empty()) return {{}, util::status_code_t::unsupported_k};
        scratch_t scratch {executor};
        if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        typename index_t::maxsim_config_t const config {search_config_of(search), wanted};
        std::span<std::size_t const> const query_offsets = queries.offsets;
        ranked_hints_view_t const hints = queries.hints;
        return between_swaps(index, [&]() noexcept {
            return with_predicate(predicate, [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept {
                if constexpr (is_vram_index<index_type_>)
                    return stats_of(of(index).maxsim(queries.rows, query_offsets, scored, matches, config, executor));
                else if constexpr (std::is_same_v<narrowing_type_, no_predicate_t>) {
                    if constexpr (is_ram_index<index_type_>)
                        return stats_of(of(index).maxsim(queries.rows, query_offsets, scored, matches, config, executor,
                                                         to_writable_span(scratch.traversals), hints));
                    else
                        return stats_of(of(index).maxsim(queries.rows, query_offsets, scored, matches, config, executor,
                                                         {}, hints));
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
        });
    }

    /** Attends the queries over their walks' neighbors; a GPU index answers @c unsupported_k for a
     *  scan. */
    static util::expected<util::traversal_stats_t> attend(void* index, rows_view_t queries,
                                                          attention_arguments_t attention, ::search_config_t search,
                                                          predicate_t predicate, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>)
            if (search.traversal == util::traversal_t::scan_k) return {{}, util::status_code_t::unsupported_k};
        scratch_t scratch {executor};
        if (util::status_t const prepared = scratch.prepare(executor); failed(prepared)) return {{}, prepared};
        search_config_t const config = search_config_of(search);
        return between_swaps(index, [&]() noexcept {
            return with_predicate(predicate, [&]<typename narrowing_type_>(narrowing_type_& narrowing) noexcept {
                if constexpr (is_vram_index<index_type_>)
                    return stats_of(of(index).attend(queries, attention.heads, attention.value_lookup,
                                                     attention.score_scale, attention.outputs, attention.carry, config,
                                                     executor));
                else if constexpr (std::is_same_v<narrowing_type_, no_predicate_t>) {
                    if constexpr (is_ram_index<index_type_>)
                        return stats_of(of(index).attend(queries, attention.heads, attention.value_lookup,
                                                         attention.score_scale, attention.outputs, attention.carry,
                                                         config, executor, to_writable_span(scratch.traversals)));
                    else
                        return stats_of(of(index).attend(queries, attention.heads, attention.value_lookup,
                                                         attention.score_scale, attention.outputs, attention.carry,
                                                         config, executor));
                }
                else {
                    cheap_predicate_t cheap {narrowing.predicate};
                    if constexpr (is_ram_index<index_type_>)
                        return stats_of(of(index).attend_if(
                            queries, attention.heads, attention.value_lookup, attention.score_scale, attention.outputs,
                            attention.carry, config, cheap, executor, to_writable_span(scratch.traversals)));
                    else
                        return stats_of(of(index).attend_if(queries, attention.heads, attention.value_lookup,
                                                            attention.score_scale, attention.outputs, attention.carry,
                                                            config, cheap, executor));
                }
            });
        });
    }

    /** Opens a cursor over every entry, breadth-first from @p starts when it names any, which a
     *  view resolves on its first pull, its state placed through @p allocator; an index without a
     *  cursor answers @c unsupported_k. */
    static util::expected<void*> cursor_init(void* index, std::span<key_t const> starts,
                                             util::forkunion_allocator_t allocator) noexcept {
        if constexpr (walks_cursors<index_t>) {
            typename index_t::cursor_t* const opened = new (std::nothrow) typename index_t::cursor_t(allocator);
            if (!opened) return {nullptr, util::status_code_t::out_of_memory_k};
            if (util::status_t const walking = opened->open(of(index), starts); failed(walking))
                return delete opened, util::expected<void*> {nullptr, walking};
            return {opened, util::status_code_t::success_k};
        }
        else return {nullptr, util::status_code_t::unsupported_k};
    }
    static util::expected<std::size_t> cursor_next(void* index, void* cursor, std::span<key_t> keys,
                                                   links_outputs_t* links, executor_t& executor) noexcept {
        if constexpr (!walks_cursors<index_t>) return {0, util::status_code_t::unsupported_k};
        else {
            typename index_t::cursor_t& walk = *static_cast<typename index_t::cursor_t*>(cursor);
            if constexpr (viewed_k)
                return between_swaps(index, [&]() noexcept {
                    return walk.pull(of(index), keys, links ? &links->keys : nullptr, links ? &links->offsets : nullptr,
                                     executor);
                });
            else return walk.pull(of(index), keys, links ? &links->keys : nullptr, links ? &links->offsets : nullptr);
        }
    }
    static void cursor_free(void* cursor) noexcept {
        if constexpr (walks_cursors<index_t>) delete static_cast<typename index_t::cursor_t*>(cursor);
    }

    /** Samples a RAM index or an NVMe view; a GPU index does not sample yet. */
    static util::expected<util::sample_result_t> sample(void* index, util::sample_inputs<key_t> inputs,
                                                        util::sample_outputs<key_t> outputs,
                                                        util::sample_config_t config, executor_t& executor) noexcept {
        if constexpr (is_vram_index<index_type_>) return {{}, util::status_code_t::unsupported_k};
        else
            return between_swaps(index, [&]() noexcept { return of(index).sample(inputs, outputs, config, executor); });
    }

    static util::status_t save(void* index, std::string_view path, executor_t& executor,
                               progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return util::status_code_t::unsupported_k;
        else return of(index).save(path, save_config_of(), executor, ticks);
    }
    static util::status_t save_buffer(void* index, util::bytes_span_t buffer, executor_t& executor,
                                      progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return util::status_code_t::unsupported_k;
        else return of(index).save(buffer, save_config_of(), executor, ticks);
    }
    static util::status_t load(void* index, std::string_view path, executor_t& executor,
                               progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return refuses_writes(index);
        else return of(index).load(path, util::load_config_t {}, executor, ticks);
    }

    /** Loads a checkpoint where it sits: host bytes into any index, a device's only into a tower
     *  whose shard plans copy it device to device. */
    static util::status_t load_buffer(void* index, util::checkpoint_source_t source, executor_t& executor,
                                      progress_ticks_t& ticks) noexcept {
        if constexpr (viewed_k) return refuses_writes(index);
        else if constexpr (loads_checkpoint_sources<index_t>)
            return of(index).load(source, util::load_config_t {}, executor, ticks);
        else if (source.device != util::host_device_k) return util::status_code_t::unsupported_k;
        else return of(index).load(source.bytes, util::load_config_t {}, executor, ticks);
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
            .load_nvme = &load_nvme,
            // Shape
            .kernel = &kernel,
            .geometry = &geometry,
            .residency = &residency,
            .shards_count = &shards_count,
            .reads_spdk = &reads_spdk,
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
            .health = &health,
            // Plans
            .refine_plan_open = &refine_plan_open,
            .reshard_plan_open = &reshard_plan_open,
            .refine_plan = plan_verbs_for<refine_plan_t>(),
            .reshard_plan = plan_verbs_for<reshard_plan_t>(),
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
 *  class's unit alone holds kernels for its generation. The serial floor always compiles its
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
    static_assert(tables_count<residency_, class_>(dense_ladder_dtypes_t {}) != 0,
                  "A compiled class advances past its parent for no dtype");
    for (dense_verbs<residency_> const& table : tables_k)
        if (table.dtype == dtype) return &table;
    return nullptr;
}

#pragma endregion Class Tables
