/**
 *  @file src/basic_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The C surface every basic modality shares: one handle over a modality's picked variant,
 *      one cursor, the CPU and GPU ladders, and every verb once, which a modality's dispatch unit
 *      forwards its exported entry points to.
 *
 *  A modality's traits supply only what its verbs differ in: the geometry an init or a checkpoint
 *  states, the check of a call's values, and the C argument structures its variants' verb tables
 *  take. Every verb visits three homes, and the variants' bodies compile in the modality's class
 *  units.
 */
#pragma once

#include <memory>      // `std::unique_ptr` - a finished refinement's handle and state
#include <new>         // `std::nothrow`
#include <span>        // `std::span` - the listed devices, a view's paths and rows
#include <string_view> // `std::string_view` - checkpoint paths
#include <type_traits> // `std::remove_pointer_t`
#include <utility>     // `std::exchange` - a landed donor's slot
#include <variant>     // `std::variant`, `std::visit`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the cursors' busy bits

#include "basic_punned.cuh"

#pragma region Handles

/** The verb table of every home one modality compiles, one of which a handle holds. */
template <typename traits_type_>
using basic_home_tables = std::variant<basic_verbs<traits_type_, dense_home_t::ram_k> const*,
                                       basic_verbs<traits_type_, dense_home_t::nvme_k> const*,
                                       basic_verbs<traits_type_, dense_home_t::vram_nvidia_k> const*>;

/**
 *  @brief The one dynamically-typed index of a modality every binding consumes: the variant picked
 *      once at construction, and the class or device whose kits it scores with.
 *
 *  @tparam traits_type_ What the modality's verbs differ in.
 */
template <typename traits_type_>
struct basic_handle {
    using traits_t = traits_type_;

    /** The picked variant's home table, and the variant only that table's entries cast back to its
     *  type. */
    basic_home_tables<traits_type_> tables;
    void* variant = nullptr;

    basic_handle(basic_home_tables<traits_type_> picked, void* created) noexcept : tables(picked), variant(created) {}
    basic_handle(basic_handle const&) = delete;
    basic_handle& operator=(basic_handle const&) = delete;
    ~basic_handle() noexcept {
        std::visit([&](auto const* verbs) noexcept { verbs->destroy(variant); }, tables);
    }
};

/** The cursor over a modality's handle: a RAM variant's snapshot, or the next row of an NVMe view,
 *  with the table whose entries pull and free it, which outlives every handle. */
template <typename handle_type_>
struct basic_cursor {
    basic_home_tables<typename handle_type_::traits_t> tables;
    void* cursor = nullptr;

    /** The handle the cursor was opened on, which every pull names again. */
    handle_type_* index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    u32_t busy = 0;

    ~basic_cursor() noexcept {
        if (cursor) std::visit([&](auto const* verbs) noexcept { verbs->cursor_free(cursor); }, tables);
    }
};

/** What a modality's refinement handle owns: one variant's plan with the donors it lands, which
 *  keeps its index closed until it dies, and the caller's donor slots it nulls. */
template <typename handle_type_>
struct basic_refinement {
    basic_home_tables<typename handle_type_::traits_t> tables;
    void* refining = nullptr;

    /** The handle and executor the refinement was opened with, only ever compared with the ones
     *  each call passes. */
    handle_type_ const* opened_on = nullptr;
    usearch_executor_t opened_with = nullptr;

    /** The caller's donor array, @c count slots. */
    handle_type_** donors = nullptr;
    std::size_t count = 0;

    ~basic_refinement() noexcept {
        if (refining) std::visit([&](auto const* verbs) noexcept { verbs->refinement_free(refining); }, tables);
    }
};

/** The handle a call's arguments name, as the index they act on or the slot a new one lands in. */
template <typename arguments_type_>
using handle_of = std::remove_pointer_t<std::remove_pointer_t<decltype(std::declval<arguments_type_&>().index)>>;

/** Applies @p visitor to the variant @p handle picked, as <tt>visitor(punned_basic_index<traits,
 *  home>)</tt>. */
template <typename handle_type_, typename visitor_type_>
void visit_basic(handle_type_ const& handle, visitor_type_&& visitor) noexcept {
    using traits_t = typename handle_type_::traits_t;
    std::visit(
        [&]<dense_home_t home_>(basic_verbs<traits_t, home_> const* verbs) noexcept {
            visitor(punned_basic_index<traits_t, home_> {*verbs, handle.basic.variant});
        },
        handle.basic.tables);
}

/** The variant of @p other when it holds the very variant @p punned is, so both share one class,
 *  home and stored scalar; null otherwise. */
template <typename traits_type_, dense_home_t home_, typename handle_type_>
void* same_basic_variant(punned_basic_index<traits_type_, home_> const& punned, handle_type_ const& other) noexcept {
    basic_verbs<traits_type_, home_> const* const* const verbs = std::get_if<basic_verbs<traits_type_, home_> const*>(
        &other.basic.tables);
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
template <typename traits_type_, dense_home_t home_>
status_t place_shards(punned_basic_index<traits_type_, home_> const& punned,
                      usearch_ram_options_t const& ram) noexcept {
    punned.verbs.bound_shards(punned.variant, ram.shard_capacity);
    if (!ram.memory_domains || !ram.memory_domains_count)
        return emplace_on_every_domain([&](fu_memory_domain_id_t domain) noexcept {
            return punned.verbs.emplace(punned.variant, static_cast<int>(domain));
        });
    for (std::size_t position = 0; position != ram.memory_domains_count; ++position)
        if (status_t const placed = punned.verbs.emplace(punned.variant,
                                                         static_cast<int>(ram.memory_domains[position]));
            failed(placed))
            return placed;
    return status_t::success_k;
}

/** The RAM or NVMe variant of @p traits_type_ serving @p geometry on the newest rung of @p ladder
 *  that fits under @p ceiling, the serial floor last; a class unit without a variant serving the
 *  geometry sends the walk on. */
template <typename traits_type_, dense_home_t home_, cpu_class_t... classes_>
basic_verbs<traits_type_, home_> const* pick_basic_cpu_variant(cpu_ladder<classes_...>,
                                                               typename traits_type_::geometry_t const& geometry,
                                                               cpu_ceiling_t ceiling) noexcept {
    basic_verbs<traits_type_, home_> const* picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (basic_class_compiled(class_))
            if (!picked && cpu_rung_fits(class_, ceiling))
                picked = basic_class_tables<traits_type_, home_, class_>::find(geometry);
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** The NVIDIA variant of @p traits_type_ serving @p geometry on the newest rung of @p ladder the
 *  build compiled and the devices' @p available capabilities admit. */
template <typename traits_type_, gpu_class_t... classes_>
basic_verbs<traits_type_, dense_home_t::vram_nvidia_k> const* pick_basic_gpu_variant(
    gpu_ladder<classes_...>, typename traits_type_::geometry_t const& geometry, nk_capability_t available) noexcept {
    basic_verbs<traits_type_, dense_home_t::vram_nvidia_k> const* picked = nullptr;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        if constexpr (basic_gpu_class_compiled(class_))
            if (!picked && numkong_admits(available, class_))
                picked = basic_class_tables<traits_type_, dense_home_t::vram_nvidia_k, class_>::find(geometry);
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/**
 *  @brief Lowers @p geometry, frozen with @p arguments, onto the RAM or NVMe variant of
 *      @p handle_type_ the ladder picks, then hands it to @p fill, as
 *      <tt>fill(punned_basic_index) → bool</tt>; null once @p outcome records why not.
 *
 *  The variant's class is one the machine admits and @p ram 's pinned class holds, and it serves
 *  the geometry. A RAM variant places its shards as @p ram names before it freezes.
 */
template <typename handle_type_, dense_home_t home_, typename arguments_type_, typename fill_type_>
handle_type_* make_basic_cpu_handle(typename handle_type_::traits_t::geometry_t& geometry,
                                    arguments_type_ const& arguments, usearch_ram_options_t const& ram,
                                    usearch_outcome_t* outcome, fill_type_&& fill) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    auto const [ceiling, pick_status] = pick_cpu_ceiling(ram.cpu_class);
    if (failed(pick_status))
        return propagate_error(usearch_status_unsupported_k, outcome, "The CPU does not admit the pinned class"),
               nullptr;
    basic_verbs<traits_t, home_> const* const verbs = pick_basic_cpu_variant<traits_t, home_>(cpu_ladder_t {}, geometry,
                                                                                              ceiling);
    if (!verbs)
        return propagate_error(usearch_status_unsupported_k, outcome, "No compiled variant serves the geometry"),
               nullptr;
    void* const created = verbs->create(0, geometry);
    if (!created)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    handle_type_* const handle = new (std::nothrow) handle_type_ {{verbs, created}};
    if (!handle)
        return verbs->destroy(created),
               propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    punned_basic_index<traits_t, home_> const punned {*verbs, created};
    // A view reads its shards from its files, so only a RAM variant places any.
    if constexpr (home_ == dense_home_t::ram_k)
        if (status_t const placed = place_shards(punned, ram); failed(placed))
            return propagate_error(placed, outcome), delete handle, nullptr;
    if (!verbs->freeze(created, geometry, traits_t::freeze_input(arguments), outcome) || !fill(punned))
        return delete handle, nullptr;
    return handle;
}

#pragma endregion Handles

#pragma region Lifetime

template <typename arguments_type_>
void basic_init_ram(arguments_type_* arguments) noexcept {
    using handle_t = handle_of<arguments_type_>;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    typename handle_t::traits_t::geometry_t geometry;
    if (!handle_t::traits_t::state(*arguments, geometry, outcome)) return;
    handle_t* const index = make_basic_cpu_handle<handle_t, dense_home_t::ram_k>(
        geometry, *arguments, arguments->ram, outcome, [](auto const&) noexcept { return true; });
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Loads the checkpoint at @p path into the freshly placed @p punned on the executor @p arguments
 *  name; false once @p outcome records why not. */
template <typename traits_type_, dense_home_t home_, typename arguments_type_>
bool basic_load_variant(punned_basic_index<traits_type_, home_> punned, std::string_view path,
                        arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
    usearch_executor_handle_t call_executor;
    typename punned_basic_index<traits_type_, home_>::executor_t* const executor = executor_of(
        punned, arguments.executor, call_executor, outcome);
    if (!executor) return false;
    progress_ticks_t ticks {arguments.progress, executor->size()};
    status_t const loaded = punned.verbs.load(punned.variant, path, *executor, ticks);
    if (loaded == status_t::config_mismatch_k && safegraphs_read_metadata(path).value.neighbors_dtype == dtype_t::u64_k)
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "The checkpoint's node IDs are 64-bit; the library's shards hold 32-bit IDs");
    return propagate_error(loaded, outcome);
}

#if USEARCH_ARCH_CUDA_

/** Bounds every shard of a GPU variant at @p vram 's shard capacity, then places one shard per
 *  named device, on that device's vendor ordinal; the caller checked every named ordinal against
 *  the listing. */
template <typename traits_type_, dense_home_t home_>
status_t place_shards(punned_basic_index<traits_type_, home_> const& punned,
                      usearch_vram_options_t const& vram) noexcept {
    punned.verbs.bound_shards(punned.variant, vram.shard_capacity);
    std::span<gpu_device_t const> const devices = gpu_devices();
    for (std::size_t shard = 0; shard != named_count(vram); ++shard)
        if (status_t const placed = punned.verbs.emplace(
                punned.variant, static_cast<int>(devices[named_ordinal(vram, shard)].vendor_ordinal));
            failed(placed))
            return placed;
    return status_t::success_k;
}

/** Places the GPU variant of @p handle_type_ for @p geometry, frozen with @p arguments, one shard
 *  on every device they name, then hands it to @p fill, as <tt>fill(punned_basic_index) →
 *  bool</tt>; null once @p outcome records why not, as for a geometry no device kit serves.
 *
 *  The variant is the first on the modality's GPU ladder whose class the first device's
 *  capabilities admit, narrowed to the pinned class, and which serves the geometry; every named
 *  device shares that class. The shards are placed once the variant froze, since each one is built
 *  as its geometry names. */
template <typename handle_type_, typename arguments_type_, typename fill_type_>
handle_type_* make_basic_vram_handle(typename handle_type_::traits_t::geometry_t& geometry,
                                     arguments_type_ const& arguments, usearch_outcome_t* outcome,
                                     fill_type_&& fill) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    if (arguments.values_per_key)
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU index bounds no key's entries"), nullptr;
    usearch_vram_options_t const& vram = arguments.vram;
    gpu_device_t const* const device = basic_device_at(named_ordinal(vram, 0), outcome);
    if (!device) return nullptr;
    for (std::size_t shard = 1; shard < named_count(vram); ++shard) {
        gpu_device_t const* const other = basic_device_at(named_ordinal(vram, shard), outcome);
        if (!other) return nullptr;
        if (gpu_class_of(*other) != gpu_class_of(*device))
            return propagate_error(usearch_status_config_mismatch_k, outcome, "The devices must share one class"),
                   nullptr;
    }
    nk_capability_t const available = gpu_pin_capabilities(gpu_class_of(*device), vram.gpu_class);
    if (!available)
        return propagate_error(usearch_status_unsupported_k, outcome, "The devices do not run the pinned class"),
               nullptr;
    basic_verbs<traits_t, dense_home_t::vram_nvidia_k> const* const verbs = pick_basic_gpu_variant<traits_t>(
        typename traits_t::gpu_ladder_t {}, geometry, available);
    if (!verbs)
        return propagate_error(usearch_status_device_k, outcome, "This build carries no kernels for the device"),
               nullptr;
    void* const created = verbs->create(static_cast<int>(device->vendor_ordinal), geometry);
    if (!created)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    handle_type_* const index = new (std::nothrow) handle_type_ {{verbs, created}};
    if (!index)
        return verbs->destroy(created),
               propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    punned_basic_index<traits_t, dense_home_t::vram_nvidia_k> const punned {*verbs, created};
    if (!verbs->freeze(created, geometry, traits_t::freeze_input(arguments), outcome)) return delete index, nullptr;
    if (status_t const placed = place_shards(punned, vram); failed(placed))
        return propagate_error(placed, outcome), delete index, nullptr;
    if (!fill(punned)) return delete index, nullptr;
    return index;
}

#endif

/** Places a GPU index on the devices the arguments name, refusing what no device kit serves. */
template <typename arguments_type_>
void basic_init_vram(arguments_type_* arguments) noexcept {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
#if USEARCH_ARCH_CUDA_
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

template <auto verb_, typename arguments_type_>
void basic_load_ram(arguments_type_* arguments) noexcept {
    using handle_t = handle_of<arguments_type_>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(path, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto load = [&](auto const& punned) noexcept { return basic_load_variant(punned, path, *arguments, outcome); };
    handle_t* const index = make_basic_cpu_handle<handle_t, dense_home_t::ram_k>(geometry, *arguments, arguments->ram,
                                                                                 outcome, load);
    if (index) *arguments->index = index;
}

/** Loads a checkpoint onto the devices the arguments name, as the GPU variant's face answers. */
template <auto verb_, typename arguments_type_>
void basic_load_vram(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
#if USEARCH_ARCH_CUDA_
    using handle_t = handle_of<arguments_type_>;
    std::string_view const path = as_text(arguments->path);
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(path, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto load = [&](auto const& punned) noexcept { return basic_load_variant(punned, path, *arguments, outcome); };
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
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    typename handle_t::traits_t::geometry_t geometry;
    if (status_t const recalled = handle_t::traits_t::recall(path, geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    auto view = [&](auto const& punned) noexcept {
        return propagate_error(punned.verbs.view(punned.variant, path, {}, nullptr), outcome);
    };
    usearch_ram_options_t pinned {};
    pinned.cpu_class = arguments->ram.cpu_class;
    handle_t* const index = make_basic_cpu_handle<handle_t, dense_home_t::ram_k>(geometry, *arguments, pinned, outcome,
                                                                                 view);
    if (index) *arguments->index = index;
}

template <auto verb_, typename arguments_type_>
void basic_view_nvme(arguments_type_* arguments) noexcept {
    using handle_t = handle_of<arguments_type_>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
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
    pinned.cpu_class = nvme.cpu_class;
    handle_t* const index = make_basic_cpu_handle<handle_t, dense_home_t::nvme_k>(geometry, *arguments, pinned, outcome,
                                                                                  view);
    if (index) *arguments->index = index;
}

/** Frees an index, a cursor or a refinement once no deferred call names it. */
template <typename handle_type_>
void basic_free(handle_type_* handle) noexcept {
    finish_calls_using(handle);
    delete handle;
}

template <auto verb_, typename arguments_type_>
void basic_reserve(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        propagate_error(punned.verbs.reserve(punned.variant, *arguments), outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_clear(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_basic(*arguments->index,
                [&](auto punned) noexcept { propagate_error(punned.verbs.clear(punned.variant), outcome); });
}

#pragma endregion Lifetime

#pragma region Refinement

/** Opens @p punned 's refinement of @p index after the @p count @p donors land, each checked
 *  against the variant and its geometry once no deferred call names it; null once @p outcome
 *  records why not. */
template <typename handle_type_, typename traits_type_, dense_home_t home_>
void* basic_open_refinement(handle_type_ const* index, punned_basic_index<traits_type_, home_> const& punned,
                            handle_type_** donors, std::size_t count, std::size_t expansion,
                            usearch_outcome_t* outcome) noexcept {
    growing_buffer<void*> landing {global_allocator};
    if (failed(landing.reserve_exactly(count)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the donors"), nullptr;
    for (std::size_t position = 0; position != count; ++position) {
        handle_type_* const donor = donors[position];
        if (!donor)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "A donor must not be null"), nullptr;
        if (donor == index)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "An index cannot absorb itself"), nullptr;
        void* const typed = same_basic_variant(punned, *donor);
        if (!typed || !punned.verbs.matches(punned.variant, typed))
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "A donor must share the index's residency, class and geometry"),
                   nullptr;
        finish_calls_using(donor);
        landing.push_back_reserved(typed);
    }
    auto const [refining, status] = punned.verbs.refinement_open(punned.variant, to_span(landing), expansion);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    return refining;
}

/** Frees the handles of the donors @p refining consumed, folding what their variants count into the
 *  variant's first, and nulls the caller's slots; the plan dropped its hold on each already. */
template <typename traits_type_, dense_home_t home_, typename handle_type_>
void basic_free_landed(punned_basic_index<traits_type_, home_> const& punned, void const* refining,
                       handle_type_** donors, std::size_t count) noexcept {
    for (std::size_t position = 0; position != count; ++position) {
        if (!donors[position] || !punned.verbs.refinement_landed(refining, position)) continue;
        punned.verbs.absorbed(punned.variant, donors[position]->basic.variant);
        delete std::exchange(donors[position], nullptr);
    }
}

/** Writes what a refinement's @p refined result reports into the outputs @p arguments name. */
template <typename arguments_type_>
void basic_report_refined(basic_refined_t const& refined, arguments_type_ const& arguments) noexcept {
    accumulate_stats(arguments.stats, refined.stats);
    if (arguments.changed) *arguments.changed = refined.changed;
    if constexpr (requires { arguments.reachability; })
        if (arguments.reachability) *arguments.reachability = refined.reachability;
    if constexpr (requires { arguments.visited; })
        if (arguments.visited) *arguments.visited = refined.visited;
}

/** Skips the re-selection @p refining left, heals what it strands, and reports. */
template <typename traits_type_, dense_home_t home_, typename arguments_type_>
status_t basic_finish_refinement(punned_basic_index<traits_type_, home_> const& punned, void* refining,
                                 arguments_type_ const& arguments,
                                 typename punned_basic_index<traits_type_, home_>::executor_t& executor) noexcept {
    auto const [refined, status] = punned.verbs.refinement_finish(punned.variant, refining, executor);
    if (succeeded(status)) basic_report_refined(refined, arguments);
    return status;
}

/** The one-shot over a refinement, for @c refine and @c absorb alike: opens it with the call's
 *  donors, runs it once under the call's handler, then heals and reports; a cancel still heals and
 *  answers @c cancelled_k. */
template <auto verb_, typename arguments_type_>
void basic_refine_once(arguments_type_* arguments, handle_of<arguments_type_>** donors, std::size_t count) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    if (count && !donors)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The donors must name handles"), void();
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        void* const refining = basic_open_refinement(arguments->index, punned, donors, count,
                                                     arguments->expansion_factor, outcome);
        if (!refining) return;
        std::unique_ptr<void, void (*)(void*) noexcept> const owned {refining, punned.verbs.refinement_free};
        progress_ticks_t ticks {arguments->progress, punned.verbs.refinement_population(refining)};
        status_t const ran = punned.verbs.refinement_run(punned.variant, refining, ticks, *executor);
        basic_free_landed(punned, refining, donors, count);
        if (failed(ran) && ran != status_t::cancelled_k) return propagate_error(ran, outcome), void();
        // A stopped pass still heals, so its report stands beside the cancel.
        if (status_t const finished = basic_finish_refinement(punned, refining, *arguments, *executor);
            failed(finished))
            return propagate_error(finished, outcome), void();
        propagate_error(ran, outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_refine(arguments_type_* arguments) noexcept {
    basic_refine_once<verb_>(arguments, nullptr, 0);
}

template <auto verb_, typename arguments_type_>
void basic_absorb(arguments_type_* arguments) noexcept {
    basic_refine_once<verb_>(arguments, arguments->donors, arguments->count);
}

template <auto verb_, typename arguments_type_>
void basic_refinement_init(arguments_type_* arguments) noexcept {
    using refinement_t = std::remove_pointer_t<std::remove_pointer_t<decltype(arguments->refinement)>>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->refinement = nullptr;
    if (!arguments->executor)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A refinement needs an executor that outlives it"),
               void();
    if (arguments->count && !arguments->donors)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The donors must name handles"), void();
    if (arguments->count && arguments->receivers_count)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A refinement absorbs donors or splits into receivers, never both"),
               void();
    if (arguments->receivers_count && (arguments->receivers_count < 2 || !arguments->receivers))
        return propagate_error(usearch_status_invalid_argument_k, outcome, "A split needs two receivers or more"),
               void();
    if (arguments->receivers_count)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency splits an index yet"), void();
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        usearch_executor_handle_t call_executor;
        if (!executor_of(punned, arguments->executor, call_executor, outcome)) return;
        void* const refining = basic_open_refinement(arguments->index, punned, arguments->donors, arguments->count,
                                                     arguments->expansion_factor, outcome);
        if (!refining) return;
        refinement_t* const refinement = new (std::nothrow) refinement_t {};
        if (!refinement)
            return punned.verbs.refinement_free(refining),
                   propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the refinement"),
                   void();
        refinement->basic.tables = arguments->index->basic.tables;
        refinement->basic.refining = refining;
        refinement->basic.opened_on = arguments->index;
        refinement->basic.opened_with = arguments->executor;
        refinement->basic.donors = arguments->donors;
        refinement->basic.count = arguments->count;
        *arguments->refinement = refinement;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Visits the variant the refinement @p arguments name was opened on with its state, refusing a
 *  call naming another index or executor, as <tt>visitor(variant, refining, executor)</tt>. */
template <typename arguments_type_, typename visitor_type_>
void visit_refinement(arguments_type_ const& arguments, usearch_outcome_t* outcome, visitor_type_&& visitor) noexcept {
    auto* const refinement = arguments.refinement;
    if (!refinement)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement must not be null"), void();
    if (arguments.index != refinement->basic.opened_on || arguments.executor != refinement->basic.opened_with)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The refinement was opened on another index or executor"),
               void();
    visit_basic(*arguments.index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments.executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        visitor(punned, refinement->basic.refining, *executor);
    });
}

template <auto verb_, typename arguments_type_>
void basic_refinement_next(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->visited) *arguments->visited = 0;
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->stats) *arguments->stats = {};
    visit_refinement(*arguments, outcome, [&](auto const& punned, void* refining, auto& executor) noexcept {
        progress_ticks_t ticks {arguments->progress, punned.verbs.refinement_population(refining)};
        status_t const ran = punned.verbs.refinement_run(punned.variant, refining, ticks, executor);
        basic_free_landed(punned, refining, arguments->refinement->basic.donors, arguments->refinement->basic.count);
        // Running totals, so a cancelled or failed call still says how far the pass got.
        basic_report_refined(punned.verbs.refinement_running(refining), *arguments);
        propagate_error(ran, outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_refinement_finish(arguments_type_* arguments) noexcept {
    using refinement_t = std::remove_pointer_t<decltype(arguments->refinement)>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->reachability) *arguments->reachability = {};
    // Freed whatever the call answers; a plan dying unfinished reopens its index.
    std::unique_ptr<refinement_t> const refinement {arguments->refinement};
    visit_refinement(*arguments, outcome, [&](auto const& punned, void* refining, auto& executor) noexcept {
        propagate_error(basic_finish_refinement(punned, refining, *arguments, executor), outcome);
    });
}

#pragma endregion Refinement

#pragma region Introspection and Search

template <auto verb_, typename arguments_type_>
void basic_describe(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    handle_of<arguments_type_> const& index = *arguments->index;
    visit_basic(index, [&](auto punned) noexcept {
        typename handle_of<arguments_type_>::traits_t::description_t& description = *arguments->description;
        punned.verbs.describe(punned.variant, description);
        description.memory_usage += sizeof(index);
    });
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Counts a key's entries, and tallies their sizes where the traits measure any. */
template <auto verb_, typename arguments_type_>
void basic_count(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        punned.verbs.count(punned.variant, arguments->key, *arguments->key_stats, arguments->executor, outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_insert(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
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
        if (crossed(punned.verbs.insert(punned.variant, *arguments, ticks, *executor), arguments->stats, outcome))
            propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

template <auto verb_, typename arguments_type_>
void basic_search(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        int const device = punned.verbs.device(punned.variant);
        if (!accepts_traversal(device, arguments->traversal, arguments->predicate, outcome)) return;
        if (arguments->traversal == usearch_traversal_scan_k && device != host_device_k)
            return propagate_error(usearch_status_unsupported_k, outcome, "No device scan serves this modality"),
                   void();
        // Matches land on host pages, which a device arena takes over and a host one keeps.
        bool const host_landing = arguments->arena && arguments->arena->device == host_device_k;
        usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          host_landing ? host_device_k : device,
                                                          {address_of(arguments->queries)}, outcome);
        if (!arena) return;
        if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        if (!punned.verbs.accepts_search(punned.variant, *arguments, outcome)) return;
        auto [landing, carve_status] = carve_matches(arena->memory, count, wanted);
        if (failed(carve_status)) return propagate_error(carve_status, outcome, "Failed to grow the arena"), void();
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        if (!crossed(punned.verbs.search(punned.variant, *arguments, arena->memory, landing, ticks, *executor),
                     arguments->stats, outcome))
            return;
        auto const [matches, publish_status] = publish_matches(*arena, landing, count);
        if (failed(publish_status))
            return propagate_error(publish_status, outcome, "Failed to land the matches on the device"), void();
        *arguments->matches = matches;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Lands every entry within the radius a modality's containment walk names, a tape per query in
 *  the arena, as the variant's face answers. */
template <auto verb_, typename arguments_type_>
void basic_within(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {address_of(arguments->queries)}, outcome);
    if (!arena) return;
    if (!traits_t::accepts(*arguments, outcome)) return;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    visit_basic(*arguments->index, [&]<typename punned_type_>(punned_type_ punned) noexcept {
        if (!punned.verbs.accepts_within(punned.variant, *arguments, outcome)) return;
        usearch_executor_handle_t call_executor;
        typename punned_type_::executor_t* const executor = executor_of(punned, arguments->executor, call_executor,
                                                                        outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        usearch_matches_tape_view_t matches {};
        std::size_t written = 0;
        if (!crossed(punned.verbs.within(punned.variant, *arguments, arena->memory, matches, written, ticks, *executor),
                     arguments->stats, outcome))
            return;
        *arguments->matches = matches;
        if (arguments->written) *arguments->written = written;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Exports a key's entries into the arena, in insertion order. */
template <auto verb_, typename arguments_type_>
void basic_get(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *traits_t::output_of(*arguments) = {};
    if (arguments->written) *arguments->written = 0;
    if constexpr (requires { traits_t::exports(*arguments, outcome); })
        if (!traits_t::exports(*arguments, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    visit_basic(*arguments->index,
                [&](auto punned) noexcept { punned.verbs.get(punned.variant, *arguments, arena->memory, outcome); });
}

#pragma endregion Introspection and Search

#pragma region Cursors and Serialization

template <typename arguments_type_>
void basic_cursor_init(arguments_type_* arguments) noexcept {
    using cursor_handle_t = std::remove_pointer_t<std::remove_pointer_t<decltype(arguments->cursor)>>;
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->starts_count && !arguments->starts)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The starts must name keys"), void();
    if (arguments->starts_count)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    cursor_handle_t* const cursor = new (std::nothrow) cursor_handle_t {};
    if (!cursor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    cursor->basic.index = arguments->index;
    cursor->basic.tables = arguments->index->basic.tables;
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        cursor->basic.cursor = punned.verbs.cursor_open(punned.variant, outcome);
    });
    // A variant that opened nothing said why in the outcome.
    if (!cursor->basic.cursor) return delete cursor, void();
    *arguments->cursor = cursor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys into the arena. */
template <auto verb_, typename arguments_type_>
void basic_cursor_next(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    *arguments->written = 0;
    basic_cursor<handle_of<arguments_type_>>& cursor = arguments->cursor->basic;
    if (cursor.index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (arguments->links)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency pulls graph links yet"), void();
    if (fu::standard_atomic_ref<u32_t>(cursor.busy).fetch_test_and_set_bit(0, std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (arena)
        visit_basic(*arguments->index, [&](auto punned) noexcept {
            punned.verbs.cursor_next(punned.variant, cursor.cursor, *arguments, arena->memory, outcome);
        });
    fu::standard_atomic_ref<u32_t>(cursor.busy).clear_bits(1, std::memory_order_release);
}

template <auto verb_, typename arguments_type_>
void basic_save(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
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

/** Samples the index as each variant's face answers. */
template <auto verb_, typename arguments_type_>
void basic_sample(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    *arguments->links = {};
    *arguments->written = 0;
    if (arguments->stats) *arguments->stats = {};
    visit_basic(*arguments->index, [&](auto punned) noexcept {
        propagate_error(punned.verbs.sample(punned.variant, *arguments), outcome);
    });
}

/** Splits the index into its receivers, which no residency serves yet. */
template <auto verb_, typename arguments_type_>
void basic_split(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->reachability) *arguments->reachability = {};
    if (arguments->stats) *arguments->stats = {};
    if (arguments->count < 2 || !arguments->receivers)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "A split needs two receivers or more"),
               void();
    propagate_error(usearch_status_unsupported_k, outcome, "No residency splits an index yet");
}

/** Reads the geometry a checkpoint recorded without loading it; @p traits_type_ names the modality,
 *  since the call names no handle. */
template <auto verb_, typename traits_type_, typename arguments_type_>
void basic_metadata_read(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    typename traits_type_::geometry_t geometry;
    if (status_t const recalled = traits_type_::recall(as_text(arguments->path), geometry); failed(recalled))
        return propagate_error(recalled, outcome), void();
    *arguments->metadata = traits_type_::metadata_of(geometry);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Cursors and Serialization
