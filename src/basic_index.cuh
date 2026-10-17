/**
 *  @file src/basic_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The C surface every basic modality shares: one handle over a modality's variant, one
 *      cursor, the CPU-class ladder, and every verb once, which a modality's dispatch unit forwards
 *      its exported entry points to.
 *
 *  A modality's traits supply only what its verbs differ in: the geometry an init or a checkpoint
 *  states and its freezing into a variant, the check of a call's values, the landing of stored
 *  values into an arena, and the statistics of one key's entries. The variants take a call's
 *  arguments whole.
 */
#pragma once

#include <cstdio> // `std::snprintf` - the capability string

#include <atomic>      // `std::atomic_flag` - a cursor's one pull at a time
#include <memory>      // `std::unique_ptr` - a finished refinement's handle
#include <new>         // `std::nothrow`
#include <optional>    // `std::optional` - a refinement opened or refused
#include <span>        // `std::span` - the listed devices, a view's paths and rows
#include <string_view> // `std::string_view` - checkpoint paths
#include <type_traits> // `std::type_identity`, `std::remove_pointer_t`
#include <utility>     // `std::exchange` - a landed donor's slot
#include <variant>     // `std::variant`, `std::visit`

#include "basic_specializations.cuh"

#pragma region Handles

/**
 *  @brief The one dynamically-typed index of a modality every binding consumes: a variant over the
 *      variants this build compiled, and the class or device whose kits the chosen one scores with.
 *
 *  @tparam variant_type_ The variants, each residency's CPU classes newest first and the serial
 *      floor last, which is the order the ladder tries them in.
 *  @tparam traits_type_ What the modality's verbs differ in.
 */
template <typename variant_type_, typename traits_type_>
struct basic_handle {
    using variant_t = variant_type_;
    using traits_t = traits_type_;

    variant_t variant;

    /** The class or device whose kits the variant scores with, named once at construction. */
    char capabilities_string[32] {};

    /** Builds the variant @p alternative_type_ over @p placement: a CPU variant's allocator, or a
     *  GPU variant's device. */
    template <typename alternative_type_, typename placement_type_>
    basic_handle(std::in_place_type_t<alternative_type_> tag, placement_type_ const& placement) noexcept
        : variant(tag, placement) {}
};

/** The cursor over a modality's handle: a RAM variant's snapshot, one of @p cursor_types_, or the
 *  next row of an NVMe view, whose rows never change. */
template <typename handle_type_, typename... cursor_types_>
struct basic_cursor {
    std::variant<std::monostate, cursor_types_..., std::size_t> cursor;

    /** The handle the cursor was opened on, which every pull names again. */
    handle_type_* index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The handle a call's arguments name, as the index they act on or the slot a new one lands in. */
template <typename arguments_type_>
using handle_of = std::remove_pointer_t<std::remove_pointer_t<decltype(std::declval<arguments_type_&>().index)>>;

/** Where a call's tape of queries starts, which the arena it rewinds must not hold. */
inline void const* address_of(usearch_tape_view_t const& tape) noexcept { return tape.data; }

/** Where a call's run of queries starts, which the arena it rewinds must not hold. */
inline void const* address_of(void const* values) noexcept { return values; }

/** Whether @p variant_type_ stores values in the scalar @p geometry names, as a variant storing
 *  them in one scalar must; a variant storing no scalar fits any geometry. */
template <typename variant_type_, typename geometry_type_>
bool stores_scalar_of(geometry_type_ const& geometry) noexcept {
    if constexpr (requires { variant_type_::dtype_k; })
        return nk_dtype_of(geometry.quantization) == variant_type_::dtype_k;
    else return true;
}

/**
 *  @brief Lowers @p geometry, frozen with @p arguments, onto the first CPU variant of
 *      @p handle_type_ driven by @p executor_type_, then hands it to @p fill, as
 *      `fill(variant) → bool`; null once @p outcome records why not.
 *
 *  The variant is RAM or NVMe, its class is one the machine admits, and its scalar, if it stores
 *  one, is the one the geometry names. The variant lists each residency's classes newest first
 *  and the serial floor last, and never a class the build left out, so its order is the ladder.
 */
template <typename handle_type_, typename executor_type_, typename arguments_type_, typename fill_type_>
handle_type_* make_cpu_handle(typename handle_type_::traits_t::geometry_t& geometry, arguments_type_ const& arguments,
                              usearch_ram_options_t const& ram, usearch_outcome_t* outcome,
                              fill_type_&& fill) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    // One basic index per handle, so one memory domain at most; values land where first touched.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A basic index places one shard"), nullptr;
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    handle_type_* handle = nullptr;
    bool picked = false;
    auto rung = [&]<typename variant_type_>(std::type_identity<variant_type_>) noexcept {
        if constexpr (std::is_same_v<typename variant_type_::executor_t, executor_type_>) {
            if (picked) return;
            if (variant_type_::cpu_class_k != cpu_class_t::serial_k &&
                !numkong_admits(available, variant_type_::cpu_class_k))
                return;
            if (!stores_scalar_of<variant_type_>(geometry)) return;
            picked = true;
            handle = new (std::nothrow)
                handle_type_(std::in_place_type<variant_type_>, forkunion_allocator_t {no_memory_domain_k});
            if (!handle)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), void();
            variant_type_& variant = std::get<variant_type_>(handle->variant);
            if (!traits_t::freeze(variant, geometry, arguments, outcome) || !fill(variant))
                return delete handle, handle = nullptr, void();
            std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                          usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(variant_type_::cpu_class_k)));
        }
    };
    [&]<typename... variant_types_>(std::type_identity<std::variant<variant_types_...>>) noexcept {
        (rung(std::type_identity<variant_types_> {}), ...);
    }(std::type_identity<typename handle_type_::variant_t> {});
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
    handle_t* const index = make_cpu_handle<handle_t, forkunion_executor_t>(
        geometry, *arguments, arguments->ram, outcome, [](auto&) noexcept { return true; });
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Loads the checkpoint at @p path into the freshly placed @p variant on the executor
 *  @p arguments name; false once @p outcome records why not. */
template <typename variant_type_, typename arguments_type_>
bool basic_load_variant(variant_type_& variant, std::string_view path, arguments_type_ const& arguments,
                        usearch_outcome_t* outcome) noexcept {
    usearch_executor_handle_t call_executor;
    typename variant_type_::executor_t* const executor = executor_of(variant, arguments.executor, call_executor,
                                                                     outcome);
    if (!executor) return false;
    progress_ticks_t ticks {arguments.progress, executor->size()};
    return propagate_error(variant.load(path, *executor, ticks), outcome);
}

#if USEARCH_ARCH_CUDA_

/** Places the GPU variant of @p handle_type_ for @p geometry, frozen with @p arguments, on the one
 *  device they name, then hands it to @p fill, as `fill(variant) → bool`; null once @p outcome
 *  records why not, as for a geometry no device kit serves. */
template <typename handle_type_, typename arguments_type_, typename fill_type_>
handle_type_* make_vram_handle(typename handle_type_::traits_t::geometry_t& geometry, arguments_type_ const& arguments,
                               usearch_outcome_t* outcome, fill_type_&& fill) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    if (arguments.values_per_key)
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU index bounds no key's entries"), nullptr;
    usearch_vram_options_t const& vram = arguments.vram;
    if (vram.devices_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A basic index places one shard"), nullptr;
    std::size_t const ordinal = vram.devices && vram.devices_count ? static_cast<std::size_t>(vram.devices[0]) : 0;
    gpu_device_t const* const device = basic_device_at(ordinal, outcome);
    if (!device) return nullptr;
    handle_type_* index = nullptr;
    traits_t::with_vram_variant(geometry, [&]<typename variant_type_>(std::type_identity<variant_type_>) noexcept {
        index = new (std::nothrow)
            handle_type_(std::in_place_type<variant_type_>, static_cast<int>(device->vendor_ordinal));
        if (!index)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), void();
        variant_type_& variant = std::get<variant_type_>(index->variant);
        if (!traits_t::freeze(variant, geometry, arguments, outcome) || !fill(variant))
            return delete index, index = nullptr, void();
        std::snprintf(index->capabilities_string, sizeof(index->capabilities_string), "%s",
                      usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device->gpu_class)));
    });
    return index;
}

#endif

/** Places a GPU index on the one device the arguments name, refusing what no device kit serves. */
template <typename arguments_type_>
void basic_init_vram(arguments_type_* arguments) noexcept {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
#if USEARCH_ARCH_CUDA_
    using handle_t = handle_of<arguments_type_>;
    typename handle_t::traits_t::geometry_t geometry;
    if (!handle_t::traits_t::state(*arguments, geometry, outcome)) return;
    handle_t* const index = make_vram_handle<handle_t>(geometry, *arguments, outcome,
                                                       [](auto&) noexcept { return true; });
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
    auto load = [&](auto& variant) noexcept { return basic_load_variant(variant, path, *arguments, outcome); };
    handle_t* const index = make_cpu_handle<handle_t, forkunion_executor_t>(geometry, *arguments, arguments->ram,
                                                                            outcome, load);
    if (index) *arguments->index = index;
}

/** Loads a checkpoint onto the one device the arguments name, as the GPU variant's face answers. */
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
    auto load = [&](auto& variant) noexcept { return basic_load_variant(variant, path, *arguments, outcome); };
    handle_t* const index = make_vram_handle<handle_t>(geometry, *arguments, outcome, load);
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
    auto view = [&](auto& variant) noexcept { return propagate_error(variant.view(path), outcome); };
    handle_t* const index = make_cpu_handle<handle_t, forkunion_executor_t>(geometry, *arguments,
                                                                            usearch_ram_options_t {}, outcome, view);
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
    auto view = [&](auto& variant) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(variant, arguments->executor, call_executor, outcome);
        return executor && propagate_error(variant.view(path, to_span(paths), *executor), outcome);
    };
    handle_t* const index = make_cpu_handle<handle_t, nvme_executor_t>(geometry, *arguments, usearch_ram_options_t {},
                                                                       outcome, view);
    if (index) *arguments->index = index;
}

/** Frees an index or a cursor once no deferred call names it. */
template <typename handle_type_>
void basic_free(handle_type_* handle) noexcept {
    finish_calls_using(handle);
    delete handle;
}

template <auto verb_, typename arguments_type_>
void basic_reserve(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_writable(*arguments->index, outcome,
                   [&](auto& variant) noexcept { propagate_error(variant.reserve(*arguments), outcome); });
}

template <auto verb_, typename arguments_type_>
void basic_clear(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_writable(*arguments->index, outcome,
                   [&](auto& variant) noexcept { propagate_error(variant.clear(), outcome); });
}

#pragma endregion Lifetime

#pragma region Refinement

/** The plan of a variant with the donors it lands: their variants, whose handles a call frees once
 *  landed, their indexes, which the plan nulls as it lands them, and where the calls stand. A
 *  variant declaring no plan, as an NVMe view, never mutates and refines nothing. */
template <typename variant_type_>
struct basic_refinement_of {
    using type = std::monostate;
};
template <typename variant_type_>
    requires requires { typename variant_type_::refine_plan_t; }
struct basic_refinement_of<variant_type_> {
    struct type {
        growing_buffer<variant_type_*, forkunion_allocator_t> donors;
        growing_buffer<typename variant_type_::index_t*, forkunion_allocator_t> landing;
        typename variant_type_::refine_plan_t plan;
        refine_cursor_t cursor {};
    };
};

/** The refinements of every variant of @p variant_type_, as one variant of their own. */
template <typename variant_type_>
struct basic_refinements_of;
template <typename... variant_types_>
struct basic_refinements_of<std::variant<variant_types_...>> {
    using type = std::variant<std::monostate, typename basic_refinement_of<variant_types_>::type...>;
};

/** What a modality's refinement handle owns: one variant's plan, which keeps its index closed
 *  until it dies, and the caller's donor slots it nulls. */
template <typename handle_type_>
struct basic_refinement {
    typename basic_refinements_of<typename handle_type_::variant_t>::type refining;

    /** The handle and executor the refinement was opened with, only ever compared with the ones
     *  each call passes. */
    handle_type_ const* opened_on = nullptr;
    usearch_executor_t opened_with = nullptr;

    /** The caller's donor array, @c count slots. */
    handle_type_** donors = nullptr;
    std::size_t count = 0;
};

/** Opens @p variant 's refinement of @p index after the @p count @p donors land, each checked
 *  against the variant's type and geometry once no deferred call names it; empty once @p outcome
 *  records why not. */
template <typename handle_type_, typename variant_type_>
std::optional<typename basic_refinement_of<variant_type_>::type> basic_open_refinement(
    handle_type_ const* index, variant_type_& variant, handle_type_** donors, std::size_t count, std::size_t expansion,
    usearch_outcome_t* outcome) noexcept {
    using traits_t = typename handle_type_::traits_t;
    using refining_t = typename basic_refinement_of<variant_type_>::type;
    forkunion_allocator_t const allocator {no_memory_domain_k};
    decltype(refining_t::donors) typed_donors {allocator};
    decltype(refining_t::landing) landing {allocator};
    if (failed(typed_donors.reserve_exactly(count)) || failed(landing.reserve_exactly(count)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the donors"), std::nullopt;
    for (std::size_t position = 0; position != count; ++position) {
        handle_type_* const donor = donors[position];
        if (!donor)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "A donor must not be null"),
                   std::nullopt;
        if (donor == index)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "An index cannot absorb itself"),
                   std::nullopt;
        variant_type_* const typed = std::get_if<variant_type_>(&donor->variant);
        if (!typed || !traits_t::matches(variant.geometry, typed->geometry))
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "A donor must share the index's residency, class and geometry"),
                   std::nullopt;
        finish_calls_using(donor);
        typed_donors.push_back_reserved(typed);
        landing.push_back_reserved(&typed->index);
    }
    // The plan borrows the landing slots, whose storage moves along with the buffer.
    auto [plan, status] = variant.plan_refine(to_writable_span(landing), expansion);
    if (failed(status)) return propagate_error(status, outcome), std::nullopt;
    return refining_t {std::move(typed_donors), std::move(landing), std::move(plan)};
}

/** Frees the handles of the donors @p refining landed, folding what their variants count into
 *  @p variant first, and nulls the caller's slots. */
template <typename handle_type_, typename variant_type_, typename refining_type_>
void basic_free_landed(variant_type_& variant, refining_type_& refining, handle_type_** donors) noexcept {
    for (std::size_t position = 0; position != refining.landing.size(); ++position) {
        if (refining.landing[position] || !donors[position]) continue;
        if constexpr (requires { variant.absorbed(*refining.donors[position]); })
            variant.absorbed(*refining.donors[position]);
        delete std::exchange(donors[position], nullptr);
    }
}

/** Writes what a refinement's @p refined result reports into the outputs @p arguments name. */
template <typename arguments_type_>
void basic_report_refined(basic::refine_result_t const& refined, arguments_type_ const& arguments) noexcept {
    accumulate_stats(arguments.stats, refined.telemetry);
    if (arguments.changed) *arguments.changed = refined.changed_nodes;
    if constexpr (requires { arguments.reachability; })
        if (arguments.reachability)
            *arguments.reachability = {refined.health.audited_nodes, refined.health.reachable_nodes,
                                       refined.health.sweeps};
    if constexpr (requires { arguments.visited; })
        if (arguments.visited) *arguments.visited = refined.visited_nodes;
}

/** Skips the re-selection @p refining left, heals what it strands, and reports; a refinement whose
 *  donors have not all landed heals nothing and reports zeros. */
template <typename variant_type_, typename refining_type_, typename arguments_type_>
status_t basic_finish_refinement(variant_type_& variant, refining_type_& refining, arguments_type_ const& arguments,
                                 typename variant_type_::executor_t& executor) noexcept {
    auto& plan = refining.plan;
    if (plan.landing()) return status_t::success_k;
    // The re-selection a cancel left is skipped rather than finished.
    plan.stop();
    if (plan.reselecting()) refining.cursor.taken = plan.steps().size();
    progress_ticks_t heal_alone {{}, 0};
    if (status_t const healed = variant.refine(plan, refining.cursor, heal_alone, executor); failed(healed))
        return healed;
    auto const [refined, status] = plan.finish();
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
    visit_writable(*arguments->index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        auto refining = basic_open_refinement(arguments->index, variant, donors, count, arguments->expansion_factor,
                                              outcome);
        if (!refining) return;
        progress_ticks_t ticks {arguments->progress, refining->plan.audit.visible};
        status_t const ran = variant.refine(refining->plan, refining->cursor, ticks, *executor);
        basic_free_landed(variant, *refining, donors);
        if (failed(ran) && ran != status_t::cancelled_k) return propagate_error(ran, outcome), void();
        // A stopped pass still heals, so its report stands beside the cancel.
        if (status_t const finished = basic_finish_refinement(variant, *refining, *arguments, *executor);
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
    visit_writable(*arguments->index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        using refining_t = typename basic_refinement_of<variant_type_>::type;
        usearch_executor_handle_t call_executor;
        if (!executor_of(variant, arguments->executor, call_executor, outcome)) return;
        auto refining = basic_open_refinement(arguments->index, variant, arguments->donors, arguments->count,
                                              arguments->expansion_factor, outcome);
        if (!refining) return;
        refinement_t* const refinement = new (std::nothrow) refinement_t {};
        if (!refinement)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the refinement"),
                   void();
        refinement->refining.template emplace<refining_t>(std::move(*refining));
        refinement->opened_on = arguments->index;
        refinement->opened_with = arguments->executor;
        refinement->donors = arguments->donors;
        refinement->count = arguments->count;
        *arguments->refinement = refinement;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

/** Visits the variant the refinement @p arguments name was opened on with its plan, refusing a call
 *  naming another index or executor; the refinement serves one variant, so any other is refused
 *  too. */
template <typename arguments_type_, typename visitor_type_>
void visit_refinement(arguments_type_ const& arguments, usearch_outcome_t* outcome, visitor_type_&& visitor) noexcept {
    auto* const refinement = arguments.refinement;
    if (!refinement)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement must not be null"), void();
    if (arguments.index != refinement->opened_on || arguments.executor != refinement->opened_with)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The refinement was opened on another index or executor"),
               void();
    visit_writable(*arguments.index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        auto* const refining = std::get_if<typename basic_refinement_of<variant_type_>::type>(&refinement->refining);
        if (!refining)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement serves another variant"),
                   void();
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments.executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        visitor(variant, *refining, *executor);
    });
}

template <auto verb_, typename arguments_type_>
void basic_refinement_next(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->visited) *arguments->visited = 0;
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->stats) *arguments->stats = {};
    visit_refinement(*arguments, outcome, [&](auto& variant, auto& refining, auto& executor) noexcept {
        progress_ticks_t ticks {arguments->progress, refining.plan.audit.visible};
        status_t const ran = variant.refine(refining.plan, refining.cursor, ticks, executor);
        basic_free_landed(variant, refining, arguments->refinement->donors);
        // Running totals, so a cancelled or failed call still says how far the pass got.
        basic_report_refined(refining.plan.finish().value, *arguments);
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
    visit_refinement(*arguments, outcome, [&](auto& variant, auto& refining, auto& executor) noexcept {
        propagate_error(basic_finish_refinement(variant, refining, *arguments, executor), outcome);
    });
}

#pragma endregion Refinement

#pragma region Introspection

template <auto verb_, typename arguments_type_>
void basic_describe(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    auto* const index = arguments->index;
    std::visit(
        [&](auto& variant) noexcept {
            auto& description = *arguments->description;
            description.size = variant.index.size();
            description.capacity = variant.index.capacity();
            description.memory_usage = sizeof(*index) + variant.index.memory_usage();
            description.capabilities = index->capabilities_string;
            traits_t::describe(variant, description);
        },
        index->variant);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Counts a key's entries, and tallies their sizes where the traits measure any. */
template <auto verb_, typename arguments_type_>
void basic_count(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    std::visit(
        [&](auto& variant) noexcept {
            count_entries<traits_t>(variant, arguments->key, *arguments->key_stats, arguments->executor, outcome);
        },
        arguments->index->variant);
}

#pragma endregion Introspection

#pragma region Insertion and Search

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
    visit_writable(*arguments->index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        if (crossed(variant.insert(*arguments, ticks, *executor), arguments->stats, outcome))
            propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

template <auto verb_, typename arguments_type_>
void basic_search(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (device_of(variant) != host_device_k && arguments->predicate.callback)
                return propagate_error(usearch_status_device_k, outcome,
                                       "A host predicate cannot narrow a device walk"),
                       void();
            // Matches land on host pages, which a device arena takes over and a host one keeps.
            bool const host_landing = arguments->arena && arguments->arena->device == host_device_k;
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              host_landing ? host_device_k : device_of(variant),
                                                              {address_of(arguments->queries)}, outcome);
            if (!arena) return;
            if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            if (!traits_t::accepts(variant, *arguments, outcome)) return;
            matches_landing_t landing;
            if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            // A device walk stages its queries on the arena's host pages, where its matches land.
            auto const searched = [&]() noexcept {
                if constexpr (requires { variant.search(*arguments, arena->memory, landing, ticks, *executor); })
                    return variant.search(*arguments, arena->memory, landing, ticks, *executor);
                else return variant.search(*arguments, landing, ticks, *executor);
            }();
            if (!crossed(searched, arguments->stats, outcome) || !publish_matches(*arena, count, landing, outcome))
                return;
            *arguments->matches = landing.view();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variant);
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
    std::visit([&](auto& variant) noexcept { get_entries<traits_t>(variant, *arguments, arena->memory, outcome); },
               arguments->index->variant);
}

#pragma endregion Insertion and Search

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
    cursor->index = arguments->index;
    std::visit([&](auto& variant) noexcept { open_cursor(variant, cursor->cursor, outcome); },
               arguments->index->variant);
    // A variant that opened nothing said why in the outcome.
    if (std::holds_alternative<std::monostate>(cursor->cursor)) return delete cursor, void();
    *arguments->cursor = cursor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys into the arena. */
template <auto verb_, typename arguments_type_>
void basic_cursor_next(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    *arguments->written = 0;
    auto* const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (arguments->links)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency pulls graph links yet"), void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (arena)
        std::visit(
            [&](auto& variant) noexcept {
                pull_entries<traits_t>(variant, cursor->cursor, *arguments, arena->memory, outcome);
            },
            arguments->index->variant);
    cursor->busy.clear(std::memory_order_release);
}

template <auto verb_, typename arguments_type_>
void basic_save(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    if (path.ends_with(".json"))
        return propagate_error(usearch_status_unsupported_k, outcome, "A basic index saves one monolith"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, executor->size()};
            propagate_error(variant.save(path, *executor, ticks), outcome);
        },
        arguments->index->variant);
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
    std::visit([&](auto& variant) noexcept { propagate_error(variant.sample(*arguments), outcome); },
               arguments->index->variant);
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
