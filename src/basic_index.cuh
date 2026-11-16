/**
 *  @file src/basic_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The C surface every basic modality shares: one handle over a modality's picked variant,
 *      one cursor, the CPU and GPU ladders, and every verb once, which a modality's dispatch unit
 *      forwards its exported entry points to.
 *
 *  A modality's traits supply only what its verbs differ in: the geometry an init or a checkpoint
 *  states, the check of a call's values, and the composites each C argument structure lowers onto
 *  once, here, before a variant's verb table takes them. Every verb visits three residencies, and
 *  the variants' bodies compile in the modality's class units.
 */
#pragma once

#include <new>         // `std::nothrow`
#include <span>        // `std::span` - the listed devices, a view's paths and entries
#include <string_view> // `std::string_view` - checkpoint paths
#include <type_traits> // `std::remove_pointer_t`
#include <utility>     // `std::move` - a geometry into its variant
#include <variant>     // `std::variant`, `std::visit`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the cursors' busy bits

#include "basic_punned.cuh"

#pragma region Handles

/** The verb table of every residency one modality compiles, one of which a handle holds. */
template <typename traits_type_>
using basic_variant_tables = std::variant<basic_verbs<traits_type_, vendored_residency_t::ram_k> const*,
                                          basic_verbs<traits_type_, vendored_residency_t::nvme_k> const*,
                                          basic_verbs<traits_type_, vendored_residency_t::vram_nvidia_k> const*,
                                          basic_verbs<traits_type_, vendored_residency_t::vram_amd_k> const*,
                                          basic_verbs<traits_type_, vendored_residency_t::vram_apple_k> const*>;

/**
 *  @brief The one dynamically-typed index of a modality every binding consumes: the variant picked
 *      once at construction, and the class or device whose kits it scores with.
 *
 *  @tparam traits_type_ What the modality's verbs differ in.
 */
template <typename traits_type_>
struct basic_handle {
    using traits_t = traits_type_;

    /** The picked variant's residency table, and the variant only that table's entries cast back
     *  to its type. */
    basic_variant_tables<traits_type_> tables;
    void* variant = nullptr;

    basic_handle(basic_variant_tables<traits_type_> picked, void* created) noexcept
        : tables(picked), variant(created) {}
    basic_handle(basic_handle const&) = delete;
    basic_handle& operator=(basic_handle const&) = delete;
    ~basic_handle() noexcept {
        std::visit([&](auto const* verbs) noexcept { verbs->destroy(variant); }, tables);
    }
};

/** The cursor over a modality's handle: a RAM variant's snapshot, or the next entry of an NVMe
 *  view, with the table whose entries pull and free it, which outlives every handle. */
template <typename handle_type_>
struct basic_cursor {
    basic_variant_tables<typename handle_type_::traits_t> tables;
    void* cursor = nullptr;

    /** The handle the cursor was opened on, which every pull names again. */
    handle_type_* index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    u32_t busy = 0;

    ~basic_cursor() noexcept {
        if (cursor) std::visit([&](auto const* verbs) noexcept { verbs->cursor_free(cursor); }, tables);
    }
};

/** What every plan handle of @p operation_ over the modality @p index_handle_type_ holds. */
template <plan_operation_t operation_, typename index_handle_type_>
using basic_held_plan =
    held_plan<operation_, index_handle_type_, basic_variant_tables<typename index_handle_type_::traits_t>>;

/** The handle a call's arguments name, as the index they act on or the slot a new one fills. */
template <typename arguments_type_>
using handle_of = std::remove_pointer_t<std::remove_pointer_t<decltype(std::declval<arguments_type_&>().index)>>;

/** Applies @p visitor to the variant @p handle picked, as <tt>visitor(punned_basic_index<traits,
 *  residency>)</tt>. */
template <typename handle_type_, typename visitor_type_>
void visit_basic(handle_type_ const& handle, visitor_type_&& visitor) noexcept {
    using traits_t = typename handle_type_::traits_t;
    std::visit(
        [&]<vendored_residency_t residency_>(basic_verbs<traits_t, residency_> const* verbs) noexcept {
            visitor(punned_basic_index<traits_t, residency_> {*verbs, handle.basic.variant});
        },
        handle.basic.tables);
}

/** The variant of @p other when it holds the very variant @p punned is, so both share one class,
 *  residency and stored scalar; null otherwise. */
template <typename traits_type_, vendored_residency_t residency_, typename handle_type_>
void* same_basic_variant(punned_basic_index<traits_type_, residency_> const& punned,
                         handle_type_ const& other) noexcept {
    basic_verbs<traits_type_, residency_> const* const* const verbs =
        std::get_if<basic_verbs<traits_type_, residency_> const*>(&other.basic.tables);
    return verbs && *verbs == &punned.verbs ? other.basic.variant : nullptr;
}

/** Where a call's tape of queries starts, which the arena it rewinds must not hold. */
inline void const* address_of(usearch_tape_view_t const& tape) noexcept { return tape.data; }

/** Where a call's run of queries starts, which the arena it rewinds must not hold. */
inline void const* address_of(void const* values) noexcept { return values; }

/**
 *  @brief Bounds every shard of a RAM variant at @p ram 's shard capacity, then places one shard
 *      per memory domain it names.
 *
 *  Naming none places one shard per memory domain the process may allocate on, as a dense index
 *  does.
 */
template <typename traits_type_, vendored_residency_t residency_>
status_t place_shards(punned_basic_index<traits_type_, residency_> const& punned,
                      usearch_ram_options_t const& ram) noexcept {
    punned.verbs.bound_shards(punned.variant, ram.shard_capacity);
    if (!ram.memory_domains || !ram.memory_domains_count)
        return emplace_on_every_domain([&](fu_memory_domain_id_t domain) noexcept {
            return punned.verbs.emplace(punned.variant, ram_residency_k, domain);
        });
    for (std::size_t position = 0; position != ram.memory_domains_count; ++position)
        if (status_t const placed = punned.verbs.emplace(
                punned.variant, ram_residency_k, static_cast<fu_memory_domain_id_t>(ram.memory_domains[position]));
            failed(placed))
            return placed;
    return status_t::success_k;
}

/** The RAM or NVMe variant of @p traits_type_ serving @p geometry on the newest rung of @p ladder
 *  that fits under @p ceiling, the serial floor last; a class unit without a variant serving the
 *  geometry sends the walk on. */
template <typename traits_type_, vendored_residency_t residency_, cpu_class_t... classes_>
basic_verbs<traits_type_, residency_> const* pick_basic_cpu_variant(cpu_ladder<classes_...>,
                                                                    typename traits_type_::geometry_t const& geometry,
                                                                    cpu_ceiling_t ceiling) noexcept {
    basic_verbs<traits_type_, residency_> const* picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (basic_class_compiled(class_))
            if (!picked && cpu_rung_fits(class_, ceiling))
                picked = basic_class_tables<traits_type_, residency_, class_>::find(geometry);
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** The @p residency_ variant of @p traits_type_ serving @p geometry on the newest rung of @p ladder
 *  at or under the class of @p placement whose unit every placed device loads; a vendor's ladder
 *  holds only its own classes, so it answers null for another vendor's devices. */
template <typename traits_type_, vendored_residency_t residency_, gpu_class_t... classes_>
basic_verbs<traits_type_, residency_> const* pick_basic_gpu_variant(gpu_ladder<classes_...>,
                                                                    typename traits_type_::geometry_t const& geometry,
                                                                    gpu_placement_t const& placement) noexcept {
    basic_verbs<traits_type_, residency_> const* picked = nullptr;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        if constexpr (gpu_class_compiled(class_))
            if (!picked && class_ <= placement.gpu_class && (placement.loaded & gpu_class_bit(class_)))
                picked = basic_class_tables<traits_type_, residency_, class_>::find(geometry);
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/**
 *  @brief Lowers @p geometry, with what @p arguments name past it, onto the RAM or NVMe variant of
 *      @p handle_type_ the ladder picks, then hands it to @p fill, as
 *      <tt>fill(punned_basic_index) → bool</tt>; null once @p outcome records why not.
 *
 *  The variant's class is one the machine runs under @p ram 's class ceiling, and it serves
 *  the geometry. A RAM variant places its shards as @p ram names.
 */
template <typename handle_type_, vendored_residency_t residency_, typename arguments_type_, typename fill_type_>
handle_type_* make_basic_cpu_handle(typename handle_type_::traits_t::geometry_t& geometry,
                                    arguments_type_ const& arguments, usearch_ram_options_t const& ram,
                                    usearch_outcome_t* outcome, fill_type_&& fill) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    auto const [ceiling, pick_status] = pick_cpu_ceiling(ram.cpu_class_ceiling);
    if (failed(pick_status))
        return propagate_error(pick_status, outcome, "No compiled class fits under the CPU class ceiling"), nullptr;
    basic_verbs<traits_t, residency_> const* const verbs = pick_basic_cpu_variant<traits_t, residency_>(
        cpu_ladder_t {}, geometry, ceiling);
    if (!verbs)
        return propagate_error(usearch_status_unsupported_k, outcome, "No compiled variant serves the geometry"),
               nullptr;
    auto const [created, create_status] = verbs->create(std::move(geometry), traits_t::geometry_extras_of(arguments),
                                                        ram_residency_k, host_allocator_of(ram));
    if (failed(create_status)) return propagate_error(create_status, outcome), nullptr;
    handle_type_* const handle = new (std::nothrow) handle_type_ {{verbs, created}};
    if (!handle)
        return verbs->destroy(created),
               propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    punned_basic_index<traits_t, residency_> const punned {*verbs, created};
    // A view reads its shards from its files, so only a RAM variant places any.
    if constexpr (residency_ == vendored_residency_t::ram_k)
        if (status_t const placed = place_shards(punned, ram); failed(placed))
            return propagate_error(placed, outcome), delete handle, nullptr;
    if (!fill(punned)) return delete handle, nullptr;
    return handle;
}

#pragma endregion Handles

#pragma region Lifetime

template <typename arguments_type_>
void basic_init_ram(arguments_type_* arguments) noexcept {
    using handle_t = handle_of<arguments_type_>;
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
    typename handle_t::traits_t::geometry_t geometry;
    if (!handle_t::traits_t::state(*arguments, geometry, outcome)) return;
    handle_t* const index = make_basic_cpu_handle<handle_t, vendored_residency_t::ram_k>(
        geometry, *arguments, arguments->ram, outcome, [](auto const&) noexcept { return true; });
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Loads @p checkpoint, at a path or in memory, into the freshly placed @p punned on the executor
 *  @p arguments name; false once @p outcome records why not. */
template <typename traits_type_, vendored_residency_t residency_, typename checkpoint_type_, typename arguments_type_>
bool basic_load_variant(punned_basic_index<traits_type_, residency_> punned, checkpoint_type_ checkpoint,
                        arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
    usearch_executor_handle_t call_executor;
    typename punned_basic_index<traits_type_, residency_>::executor_t* const executor = executor_of(
        punned, arguments.executor, call_executor, outcome);
    if (!executor) return false;
    progress_ticks_t ticks {arguments.progress, executor->size()};
    status_t loaded = status_t::success_k;
    if constexpr (std::is_same_v<checkpoint_type_, bytes_view_t>)
        loaded = punned.verbs.load_buffer(punned.variant, checkpoint, *executor, ticks);
    else loaded = punned.verbs.load(punned.variant, checkpoint, *executor, ticks);
    if (loaded == status_t::config_mismatch_k &&
        safegraphs_read_metadata(checkpoint).value.neighbors_dtype == dtype_t::u64_k)
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "The checkpoint's node IDs are 64-bit; the library's shards hold 32-bit IDs");
    return propagate_error(loaded, outcome);
}

#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

/** Bounds every shard of a GPU variant at @p vram 's shard capacity, then places one shard per
 *  named device, on that device's vendor ordinal; the caller checked every named ordinal against
 *  the listing. */
template <typename traits_type_, vendored_residency_t residency_>
status_t place_shards(punned_basic_index<traits_type_, residency_> const& punned,
                      usearch_vram_options_t const& vram) noexcept {
    punned.verbs.bound_shards(punned.variant, vram.shard_capacity);
    std::span<gpu_device_t const> const devices = gpu_devices();
    for (std::size_t shard = 0; shard != named_count(vram); ++shard)
        if (status_t const placed = punned.verbs.emplace(
                punned.variant, vram_residency(devices[named_ordinal(vram, shard)].vendor_ordinal), no_memory_domain_k);
            failed(placed))
            return placed;
    return status_t::success_k;
}

/**
 *  @brief Places the GPU variant of @p handle_type_ for @p geometry, with what @p arguments name
 *      past it, one shard on every device they name, then hands it to @p fill; null once @p outcome
 *      records why not, as for a geometry no device kit serves.
 *
 *  The variant is the first on the modality's ladder of the devices' vendor at or under the best
 *  class every named device loads under the ceiling, whose unit they all load and which serves the
 *  geometry. Devices this build carries a class for but no variant of the geometry are refused with
 *  @c unsupported_k, as Apple GPUs are for geospatial and f32 shapes, short of their contract. The
 *  shards are placed once the variant holds its geometry, since each is built as it names; then the
 *  variant goes to <tt>fill(punned_basic_index) → bool</tt>.
 */
template <typename handle_type_, typename arguments_type_, typename fill_type_>
handle_type_* make_basic_vram_handle(typename handle_type_::traits_t::geometry_t& geometry,
                                     arguments_type_ const& arguments, usearch_outcome_t* outcome,
                                     fill_type_&& fill) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    if (arguments.values_per_key)
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU index bounds no key's entries"), nullptr;
    usearch_vram_options_t const& vram = arguments.vram;
    auto const [placement, placed_status] = resolve_placement(vram);
    if (placed_status == status_t::invalid_argument_k)
        return propagate_error(placed_status, outcome, "No GPU class has that value"), nullptr;
    if (placed_status == status_t::device_error_k)
        return propagate_error(placed_status, outcome, "A device ordinal is past the listing"), nullptr;
    if (placed_status == status_t::config_mismatch_k)
        return propagate_error(placed_status, outcome, "The devices must share one vendor"), nullptr;
    if (failed(placed_status))
        return propagate_error(placed_status, outcome,
                               "The devices share no class this build carries under the ceiling"),
               nullptr;
    // Each vendor's ladder serves only its own devices, so at most one of them picks a variant.
    handle_type_* index = nullptr;
    bool picked = false;
    auto on_vendor = [&]<vendored_residency_t residency_>(auto ladder) noexcept {
        basic_verbs<traits_t, residency_> const* const verbs =
            picked ? nullptr : pick_basic_gpu_variant<traits_t, residency_>(ladder, geometry, placement);
        if (!verbs) return;
        picked = true;
        auto const [created, create_status] = verbs->create(
            std::move(geometry), traits_t::geometry_extras_of(arguments),
            vram_residency(placement.first().vendor_ordinal), host_allocator_of(vram));
        if (failed(create_status)) return propagate_error(create_status, outcome), void();
        index = new (std::nothrow) handle_type_ {{verbs, created}};
        if (!index)
            return verbs->destroy(created),
                   propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), void();
        punned_basic_index<traits_t, residency_> const punned {*verbs, created};
        if (status_t const placed = place_shards(punned, vram); failed(placed))
            return propagate_error(placed, outcome), delete std::exchange(index, nullptr), void();
        if (!fill(punned)) delete std::exchange(index, nullptr);
    };
    on_vendor.template operator()<vendored_residency_t::vram_nvidia_k>(typename traits_t::nvidia_gpu_ladder_t {});
    on_vendor.template operator()<vendored_residency_t::vram_amd_k>(typename traits_t::amd_gpu_ladder_t {});
    on_vendor.template operator()<vendored_residency_t::vram_apple_k>(typename traits_t::apple_gpu_ladder_t {});
    if (picked) return index;
    return propagate_error(usearch_status_unsupported_k, outcome,
                           "No kernel this build carries serves the geometry on the devices"),
           nullptr;
}

#endif

/** Places a GPU index on the devices the arguments name, refusing what no device kit serves. */
template <typename arguments_type_>
void basic_init_vram(arguments_type_* arguments) noexcept {
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_
    using handle_t = handle_of<arguments_type_>;
    typename handle_t::traits_t::geometry_t geometry;
    if (!handle_t::traits_t::state(*arguments, geometry, outcome)) return;
    handle_t* const index = make_basic_vram_handle<handle_t>(geometry, *arguments, outcome,
                                                             [](auto const&) noexcept { return true; });
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
#else
    propagate_error(usearch_status_device_k, outcome, "This build compiled no GPU kernels for basic indexes");
#endif
}

/** Copies @p checkpoint, a path or the bytes of one in memory, into a new RAM index. */
template <auto verb_, typename arguments_type_, typename checkpoint_type_>
void basic_load_ram(arguments_type_* arguments, checkpoint_type_ checkpoint) noexcept {
    using handle_t = handle_of<arguments_type_>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
    if (checkpoint.empty())
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The checkpoint must not be empty"), void();
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(checkpoint, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto load = [&](auto const& punned) noexcept {
        return basic_load_variant(punned, checkpoint, *arguments, outcome);
    };
    handle_t* const index = make_basic_cpu_handle<handle_t, vendored_residency_t::ram_k>(geometry, *arguments,
                                                                                         arguments->ram, outcome, load);
    if (index) *arguments->index = index;
}

/** Whether the checkpoint @p bytes sit on the host, where a basic GPU variant reads one; false once
 *  @p outcome records why not. */
inline bool accepts_host_checkpoint(bytes_view_t bytes, usearch_outcome_t* outcome) noexcept {
    if (checkpoint_source_of(bytes).device == host_device_k) return true;
    return propagate_error(usearch_status_unsupported_k, outcome, "A basic checkpoint loads from host memory");
}

/** Whether a path, which only a host names, is a host checkpoint. */
inline bool accepts_host_checkpoint(std::string_view, usearch_outcome_t*) noexcept { return true; }

/** Loads @p checkpoint, a path or the bytes of one in host memory, onto the devices the arguments
 *  name, as the GPU variant's face answers. */
template <auto verb_, typename arguments_type_, typename checkpoint_type_>
void basic_load_vram(arguments_type_* arguments, [[maybe_unused]] checkpoint_type_ checkpoint) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_
    using handle_t = handle_of<arguments_type_>;
    if (checkpoint.empty())
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The checkpoint must not be empty"), void();
    if (!accepts_host_checkpoint(checkpoint, outcome)) return;
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(checkpoint, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto load = [&](auto const& punned) noexcept {
        return basic_load_variant(punned, checkpoint, *arguments, outcome);
    };
    handle_t* const index = make_basic_vram_handle<handle_t>(geometry, *arguments, outcome, load);
    if (index) *arguments->index = index;
#else
    propagate_error(usearch_status_device_k, outcome, "This build compiled no GPU kernels for basic indexes");
#endif
}

/** Serves a checkpoint in place on the host, as each RAM variant's face answers. */
template <typename arguments_type_>
void basic_view_ram(arguments_type_* arguments) noexcept {
    using handle_t = handle_of<arguments_type_>;
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(path, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto view = [&](auto const& punned) noexcept {
        return propagate_error(punned.verbs.view(punned.variant, path, {}, nullptr), outcome);
    };
    usearch_ram_options_t pinned {};
    pinned.cpu_class_ceiling = arguments->ram.cpu_class_ceiling;
    handle_t* const index = make_basic_cpu_handle<handle_t, vendored_residency_t::ram_k>(geometry, *arguments, pinned,
                                                                                         outcome, view);
    if (index) *arguments->index = index;
}

template <auto verb_, typename arguments_type_>
void basic_view_nvme(arguments_type_* arguments) noexcept {
    using handle_t = handle_of<arguments_type_>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_nvme_options_t const& nvme = arguments->nvme;
    *arguments->index = nullptr;
    if (arguments->executor && !std::holds_alternative<nvme_executor_t>(arguments->executor->executor))
        return propagate_error(usearch_status_device_k, outcome,
                               "The executor's kind must match the handle's residency"),
               void();
    if (!arguments->path.data || !arguments->path.length || (nvme.paths_count && !nvme.paths))
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The paths must name storage"), void();
    std::size_t const paths_count = nvme.paths ? nvme.paths_count : 0;
    growing_buffer<std::string_view> paths {global_allocator};
    if (failed(paths.resize(paths_count)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the paths"), void();
    for (std::size_t path = 0; path != paths_count; ++path) paths[path] = as_text(nvme.paths[path]);

    std::string_view const path = as_text(arguments->path);
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(path, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto view = [&](auto punned) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(punned, arguments->executor, call_executor, outcome);
        return executor && propagate_error(punned.verbs.view(punned.variant, path, to_span(paths), executor), outcome);
    };
    usearch_ram_options_t pinned {};
    pinned.cpu_class_ceiling = nvme.cpu_class_ceiling;
    handle_t* const index = make_basic_cpu_handle<handle_t, vendored_residency_t::nvme_k>(geometry, *arguments, pinned,
                                                                                          outcome, view);
    if (index) *arguments->index = index;
}

/** Frees an index or a cursor once no executor call names it. */
template <typename handle_type_>
void basic_free(handle_type_* handle) noexcept {
    if (!handle) return;
    wait_named_calls(handle->named_calls);
    delete handle;
}

template <auto verb_, typename arguments_type_>
void basic_reserve(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        propagate_error(
            punned.verbs.reserve(punned.variant, handle_of<arguments_type_>::traits_t::capacity_of(*arguments)),
            outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_clear(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    visit_basic(*arguments->index,
                [&](auto punned) noexcept { propagate_error(punned.verbs.clear(punned.variant), outcome); });
}

#pragma endregion Lifetime

#pragma region Plans

/** Whether every handle a reshard's @p arguments name holds the very variant @p punned is, shares
 *  its geometry and takes writes, and every target not among the sources is empty, waiting out the
 *  calls naming each first; false once @p outcome records why not. */
template <typename handle_type_, typename traits_type_, vendored_residency_t residency_, typename arguments_type_>
bool accepts_basic_reshard(punned_basic_index<traits_type_, residency_> const& punned, arguments_type_ const& arguments,
                           usearch_outcome_t* outcome) noexcept {
    auto const accepts = [&](handle_type_ const* handle) noexcept {
        void* const typed = same_basic_variant(punned, *handle);
        if (!typed || !punned.verbs.matches(punned.variant, typed))
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "Every handle must share the first source's residency, class and geometry");
        if (status_t const served = punned.verbs.serves_writes(typed); failed(served))
            return propagate_error(served, outcome);
        // Calls already naming the handle end first; the caller keeps it alive past the plan's
        // finish, as the headers require, so no count follows the plan.
        wait_named_calls(handle->named_calls);
        return true;
    };
    for (std::size_t source = 0; source != arguments.sources_count; ++source)
        if (!accepts(arguments.sources[source])) return false;
    for (std::size_t target = 0; target != arguments.targets_count; ++target) {
        handle_type_ const* const handle = arguments.targets[target];
        if (!accepts(handle)) return false;
        bool sourced = false;
        for (std::size_t source = 0; source != arguments.sources_count; ++source)
            sourced |= arguments.sources[source] == handle;
        typename traits_type_::description_t description {};
        if (!sourced) punned.verbs.describe(handle->basic.variant, description);
        if (description.size)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "A target must be a source or empty");
    }
    return true;
}

/** The variants behind the @p count handles @p others names, read in place. */
template <typename handle_type_>
variants_view_t view_basic_variants(handle_type_* const* others, std::size_t count) noexcept {
    return {others, count, [](void const* handles, std::size_t position) noexcept -> void* {
                handle_type_* const* const typed = static_cast<handle_type_* const*>(handles);
                return typed[position]->basic.variant;
            }};
}

template <auto verb_, typename arguments_type_>
void basic_refine(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    reset_refined(*arguments);
    if (!accepts_plan(*arguments, outcome)) return;
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        refine_config_t const config = refine_config_of(arguments->refine);
        progress_ticks_t ticks {arguments->progress, 0};
        auto const live = arguments->plan ? *arguments->plan : nullptr;
        expected<void*> const opened = live ? expected<void*> {live->held.engine_plan, status_t::success_k}
                                            : punned.verbs.refine_plan_open(punned.variant, config, *executor);
        publish_refined(
            *arguments,
            drive_planned(arguments->plan, punned.verbs.refine_plan, arguments->index->basic.tables, arguments->index,
                          arguments->executor, *executor, ticks, arguments->phase, opened),
            outcome);
    });
}

/** Moves the entries of a reshard's sources into its targets, then runs the rounds of every target
 *  that took entries, through its plan slot or a one-shot plan. */
template <auto verb_, typename handle_type_, typename arguments_type_>
void basic_reshard(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    reset_refined(*arguments);
    if (!accepts_plan(*arguments, outcome) || !accepts_reshard_lists(*arguments, outcome)) return;
    // A live plan reads its own lists and keeps the table it was opened with.
    auto const live = arguments->plan ? *arguments->plan : nullptr;
    handle_type_ const* const first = live ? nullptr : arguments->sources[0];
    std::visit(
        [&]<vendored_residency_t residency_>(basic_verbs<traits_t, residency_> const* verbs) noexcept {
            punned_basic_index<traits_t, residency_> const punned {*verbs, first ? first->basic.variant : nullptr};
            if (first && !accepts_basic_reshard<handle_type_>(punned, *arguments, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename punned_basic_index<traits_t, residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            reshard_options_t const options {nullptr, nullptr, nullptr, refine_config_of(arguments->refine),
                                             arguments->seed};
            progress_ticks_t ticks {arguments->progress, 0};
            expected<void*> const opened = live ? expected<void*> {live->held.engine_plan, status_t::success_k}
                                                : verbs->reshard_plan_open(
                                                      view_basic_variants(arguments->sources, arguments->sources_count),
                                                      view_basic_variants(arguments->targets, arguments->targets_count),
                                                      options, *executor);
            publish_refined(*arguments,
                            drive_planned(arguments->plan, verbs->reshard_plan, basic_variant_tables<traits_t> {verbs},
                                          static_cast<handle_type_ const*>(nullptr), arguments->executor, *executor,
                                          ticks, arguments->phase, opened),
                            outcome);
        },
        first ? first->basic.tables : live->held.tables);
}

#pragma endregion Plans

#pragma region Introspection and Search

template <auto verb_, typename arguments_type_>
void basic_describe(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    handle_of<arguments_type_> const& index = *arguments->index;
    visit_basic(index, [&](auto punned) noexcept {
        typename handle_of<arguments_type_>::traits_t::description_t& description = *arguments->description;
        punned.verbs.describe(punned.variant, description);
        description.memory_usage += sizeof(index);
    });
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Audits the reachability of every shard. */
template <auto verb_, typename arguments_type_>
void basic_health(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->reachability = {};
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        auto const [audit, status] = punned.verbs.health(punned.variant, *executor);
        if (failed(status)) return propagate_error(status, outcome), void();
        *arguments->reachability = {audit.audited_nodes, audit.reachable_nodes, audit.depth_from_root,
                                    audit.depth_to_root};
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Counts a key's entries, and tallies their sizes where the traits measure any. */
template <auto verb_, typename arguments_type_>
void basic_count(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->key_stats = {};
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        propagate_error(punned.verbs.count(punned.variant, arguments->key, *arguments->key_stats, *executor), outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_insert(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (arguments->stats) *arguments->stats = {};
    std::size_t const count = arguments->count;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!arguments->keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must not be null"), void();
    if (arguments->keys.stride && arguments->keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (!traits_t::accepts(*arguments, outcome)) return;
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        insert_config_t const config {static_cast<candidates_count_t>(arguments->expansion_factor)};
        if (crossed(punned.verbs.insert(punned.variant, arguments->keys, traits_t::values_of(*arguments), config,
                                        *executor, ticks),
                    arguments->stats, outcome))
            propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

template <auto verb_, typename arguments_type_>
void basic_search(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        arena_residency_t const residency = punned.verbs.residency(punned.variant);
        if (!accepts_traversal(residency, arguments->traversal, arguments->predicate, outcome)) return;
        if (arguments->traversal == usearch_traversal_scan_k && residency != ram_residency_k)
            return propagate_error(usearch_status_unsupported_k, outcome, "No device scan serves this modality"),
                   void();
        usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          staging_residency(arguments->arena, residency),
                                                          {address_of(arguments->queries)}, outcome);
        if (!arena) return;
        if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        if (!traits_t::accepts(*arguments, outcome)) return;
        auto const [queries, queries_status] = traits_t::queries_of(*arguments, arena->memory);
        if (failed(queries_status)) return propagate_error(queries_status, outcome), void();
        if (status_t const admitted = punned.verbs.serves_search(punned.variant, queries); failed(admitted))
            return propagate_error(admitted, outcome), void();
        auto [outputs, carve_status] = carve_matches(arena->memory, count, wanted);
        if (failed(carve_status)) return propagate_error(carve_status, outcome, "Failed to grow the arena"), void();
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        search_config_t const config = search_config_of(arguments->expansion_factor, arguments->traversal);
        usearch_matches_view_t const* hints = nullptr;
        if constexpr (requires { arguments->hints; }) hints = arguments->hints;
        if (rejects_hints(hints, count, outcome)) return;
        if (!crossed(punned.verbs.search(punned.variant, queries, hints, outputs, config,
                                         predicate_of(arguments->predicate), arena->memory, *executor, ticks),
                     arguments->stats, outcome))
            return;
        auto const [matches, publish_status] = publish_matches(*arena, outputs, count);
        if (failed(publish_status))
            return propagate_error(publish_status, outcome, "Failed to land the matches on the device"), void();
        *arguments->matches = matches;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Writes every entry within the radius a modality's containment walk names, a tape per query in
 *  the arena, as the variant's face answers. */
template <auto verb_, typename arguments_type_>
void basic_within(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::size_t const count = arguments->count;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      ram_residency_k, {address_of(arguments->queries)}, outcome);
    if (!arena) return;
    if (!traits_t::accepts(*arguments, outcome)) return;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        if (!accepts_traversal(punned.verbs.residency(punned.variant), arguments->traversal, arguments->predicate,
                               outcome))
            return;
        auto const [queries, queries_status] = traits_t::queries_of(*arguments, arena->memory);
        if (failed(queries_status)) return propagate_error(queries_status, outcome), void();
        predicate_t const predicate = predicate_of(arguments->predicate);
        if (status_t const admitted = punned.verbs.serves_within(punned.variant, queries, predicate); failed(admitted))
            return propagate_error(admitted, outcome), void();
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        within_outputs_t outputs;
        if (!crossed(punned.verbs.within(punned.variant, queries, outputs, search_config_of(0, arguments->traversal),
                                         predicate, arena->memory, *executor, ticks),
                     arguments->stats, outcome))
            return;
        *arguments->matches = outputs.matches;
        if (arguments->written) *arguments->written = outputs.written;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Exports a key's entries into the arena, in insertion order. */
template <auto verb_, typename arguments_type_>
void basic_get(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *traits_t::output_of(*arguments) = {};
    if (arguments->written) *arguments->written = 0;
    if constexpr (requires { traits_t::exports(*arguments, outcome); })
        if (!traits_t::exports(*arguments, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      ram_residency_k, {}, outcome);
    if (!arena) return;
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        auto const [written, status] = punned.verbs.get(punned.variant, arguments->key,
                                                        traits_t::get_outputs_of(*arguments), arena->memory, *executor);
        if (failed(status)) return propagate_error(status, outcome), void();
        if (arguments->written) *arguments->written = written;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Introspection and Search

#pragma region Cursors and Serialization

template <typename arguments_type_>
void basic_cursor_init(arguments_type_* arguments) noexcept {
    using cursor_handle_t = std::remove_pointer_t<std::remove_pointer_t<decltype(arguments->cursor)>>;
    call_outcome_t outcome(*arguments);
    *arguments->cursor = nullptr;
    if (arguments->starts_count && !arguments->starts)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The starts must name keys"), void();
    cursor_handle_t* const cursor = new (std::nothrow) cursor_handle_t {};
    if (!cursor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    cursor->basic.index = arguments->index;
    cursor->basic.tables = arguments->index->basic.tables;
    status_t opened = status_t::success_k;
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        auto const [walk, status] = punned.verbs.cursor_init(punned.variant,
                                                             {arguments->starts, arguments->starts_count});
        cursor->basic.cursor = walk;
        opened = status;
    });
    if (failed(opened)) return delete cursor, propagate_error(opened, outcome), void();
    *arguments->cursor = cursor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys into the arena. */
template <auto verb_, typename arguments_type_>
void basic_cursor_next(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (!arguments->cursor)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor must not be null"), void();
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    *arguments->written = 0;
    basic_cursor<handle_of<arguments_type_>>& cursor = arguments->cursor->basic;
    if (cursor.index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (fu::standard_atomic_ref<u32_t>(cursor.busy).fetch_test_and_set_bit(0, std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_arena_handle_t* const arena = claim_arena(
            arguments->arena, arguments->outcome, arguments->executor,
            staging_residency(arguments->arena, punned.verbs.residency(punned.variant)), {}, outcome);
        if (!arena) return;
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        links_outputs_t links {arena->memory};
        if (arguments->links && failed(links.open()))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        auto const [keys, status] = punned.verbs.cursor_next(punned.variant, cursor.cursor,
                                                             arguments->links ? &links : nullptr, arguments->wanted,
                                                             arena->memory, *executor);
        if (failed(status)) return propagate_error(status, outcome), void();
        auto const [published_keys, published_keys_status] = publish(*arena, keys.data(), keys.size());
        auto const [published_links, published_links_status] = publish_links(*arena, links);
        if (status_t const drained_status = drained(*arena,
                                                    first_failure({published_keys_status, published_links_status}));
            failed(drained_status))
            return propagate_error(drained_status, outcome, "Failed to upload the keys to the device"), void();
        if (arguments->links) *arguments->links = published_links;
        *arguments->keys = {published_keys, sizeof(usearch_key_t)};
        *arguments->written = keys.size();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
    fu::standard_atomic_ref<u32_t>(cursor.busy).clear_bits(1, std::memory_order_release);
}

template <auto verb_, typename arguments_type_>
void basic_save(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, executor->size()};
        propagate_error(punned.verbs.save(punned.variant, path, *executor, ticks), outcome);
    });
}

/** Writes the checkpoint into the arena, one monolith or a manifest leading its parts. */
template <auto verb_, typename arguments_type_>
void basic_save_buffer(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->monolith = {};
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          punned.verbs.residency(punned.variant), {}, outcome);
        if (!arena) return;
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, executor->size()};
        save_into_arena(
            *arena, punned.verbs.storage_requirements(punned.variant),
            [&](bytes_span_t checkpoint) noexcept {
                return punned.verbs.save_buffer(punned.variant, checkpoint, *executor, ticks);
            },
            *arguments->monolith, outcome);
    });
}

/** Draws around the focus keys beside the shown ones, writing every output on the arena's host
 *  pages; a GPU variant does not sample yet. */
template <auto verb_, typename arguments_type_>
void basic_sample(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    reset_sample(*arguments);
    if (!accepts_sample(*arguments, outcome)) return;
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        using key_t = us::utilities::uuid128_t;
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        usearch_arena_handle_t* const arena = claim_arena(
            arguments->arena, arguments->outcome, arguments->executor, punned.verbs.residency(punned.variant),
            {arguments->shown.data, arguments->shown_coordinates.data, arguments->focus.data}, outcome);
        if (!arena) return;
        auto const [inputs, view_status] = view_sample<key_t>(*arguments, arena->memory);
        auto const [staging, carve_status] = carve_sample(arena->memory, arguments->wanted);
        if (failed(view_status) || failed(carve_status))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        sample_config_t const config {arguments->seed, arguments->expansion_factor};
        auto const [sampled, status] = punned.verbs.sample(punned.variant, inputs, staging.template outputs<key_t>(),
                                                           config, *executor);
        if (failed(status)) return propagate_error(status, outcome), void();
        progress_ticks_t ticks {arguments->progress, arguments->wanted};
        if (sampled.written && ticks(sampled.written, arguments->wanted) == continuation_t::cancel_k)
            return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the work"),
                   void();
        publish_sample(*arguments, staging, sampled);
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Reads the geometry @p checkpoint, a path or the bytes of one in memory, recorded without loading
 *  it; @p traits_type_ names the modality, since the call names no handle. */
template <auto verb_, typename traits_type_, typename arguments_type_, typename checkpoint_type_>
void basic_metadata_read(arguments_type_* arguments, checkpoint_type_ checkpoint) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->metadata = {};
    if (checkpoint.empty())
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The checkpoint must not be empty"), void();
    typename traits_type_::geometry_t geometry;
    if (status_t const recalled = traits_type_::recall(checkpoint, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    *arguments->metadata = traits_type_::metadata_of(geometry);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Cursors and Serialization
