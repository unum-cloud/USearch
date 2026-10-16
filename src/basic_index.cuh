/**
 *  @file src/basic_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The C surface every basic modality shares: one handle over a modality's variant, one
 *      cursor, the CPU-class ladder, and every verb once, which a modality's dispatch unit forwards
 *      its exported entry points to.
 *
 *  A modality's traits supply only what its verbs differ in: the geometry an init or a checkpoint
 *  states and its freezing into an arm, the check of a call's values, the landing of stored values
 *  into an arena, and the statistics of one key's entries. The arms take a call's arguments whole.
 */
#pragma once

#include <cstdio> // `std::snprintf` - the capability string

#include <array>       // `std::array` - the one key a count scans the key columns for
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
 *      arms this build compiled, and the class or device whose kits the chosen one scores with.
 *
 *  @tparam variant_type_ The arms, each residency's CPU classes newest first and the serial floor
 *      last, which is the order the ladder tries them in.
 *  @tparam traits_type_ What the modality's verbs differ in.
 */
template <typename variant_type_, typename traits_type_>
struct basic_handle {
    using variant_t = variant_type_;
    using traits_t = traits_type_;

    variant_t variants;

    /** The class or device whose kits the arm scores with, named once at construction. */
    char capabilities_string[32] {};

    /** Builds the arm @p arm_type_ over @p placement: a CPU arm's allocator, or a GPU arm's
     *  device. */
    template <typename arm_type_, typename placement_type_>
    basic_handle(std::in_place_type_t<arm_type_> tag, placement_type_ const& placement) noexcept
        : variants(tag, placement) {}
};

/** The snapshot cursor over the index of the RAM arm @p arm_type_, which no class changes. */
template <typename arm_type_>
using ram_cursor_of =
    basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;

/** The cursor over a modality's handle: a RAM arm's snapshot, one of @p cursor_types_, or the next
 *  row of an NVMe view, whose rows never change. */
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

/** Visits the arm of @p index with @p visitor when it lives in RAM; a GPU arm serves the
 *  reservation, insertion, search and description verbs alone, and refuses the rest, and an NVMe
 *  view refuses as read-only whatever its verbs don't serve first. */
template <typename handle_type_, typename visitor_type_>
void visit_ram_variant(handle_type_& index, usearch_outcome_t* outcome, visitor_type_&& visitor) noexcept {
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if constexpr (is_ram_index<arm_type_>) visitor(arm);
            else if constexpr (is_nvme_index<arm_type_>)
                propagate_error(usearch_status_read_only_k, outcome, "An NVMe view never mutates");
            else propagate_error(usearch_status_unsupported_k, outcome, "A GPU index serves no such verb yet");
        },
        index.variants);
}

/** Visits the arm of @p index with @p visitor when it is an NVMe view, answering whether it was. */
template <typename handle_type_, typename visitor_type_>
bool visit_nvme_variant(handle_type_& index, visitor_type_&& visitor) noexcept {
    return std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if constexpr (is_nvme_index<arm_type_>) return visitor(arm), true;
            else return false;
        },
        index.variants);
}

/** The GPU ordinal @p arm computes on, where a call's arena must live, or the host's. */
template <typename arm_type_>
int device_of(arm_type_ const& arm) noexcept {
    if constexpr (is_vram_index<arm_type_>) return arm.index.device_specs().ordinal;
    else return host_device_k;
}

/** Where a call's tape of queries starts, which the arena it rewinds must not hold. */
inline void const* address_of(usearch_tape_view_t const& tape) noexcept { return tape.data; }

/** Where a call's run of queries starts, which the arena it rewinds must not hold. */
inline void const* address_of(void const* values) noexcept { return values; }

/** Whether @p arm_type_ stores values in the scalar @p geometry names, as an arm storing them in
 *  one scalar must; an arm storing no scalar fits any geometry. */
template <typename arm_type_, typename geometry_type_>
bool stores_scalar_of(geometry_type_ const& geometry) noexcept {
    if constexpr (requires { arm_type_::dtype_k; }) return nk_dtype_of(geometry.quantization) == arm_type_::dtype_k;
    else return true;
}

/**
 *  @brief Lowers @p geometry, frozen with @p arguments, onto the first CPU arm of @p handle_type_
 *      driven by @p executor_type_, RAM or NVMe, whose class the machine admits and whose scalar,
 *      if it stores one, the geometry names; null once @p outcome records why not.
 *
 *  The variant lists each residency's classes newest first and the serial floor last, and never a
 *  class the build left out, so its order is the ladder.
 */
template <typename handle_type_, typename executor_type_, typename arguments_type_>
handle_type_* make_cpu_handle(typename handle_type_::traits_t::geometry_t& geometry, arguments_type_ const& arguments,
                              usearch_ram_options_t const& ram, usearch_outcome_t* outcome) noexcept {
    using traits_t = typename handle_type_::traits_t;
    if (!traits_t::admits(geometry, arguments, outcome)) return nullptr;
    // One basic index per handle, so one memory domain at most; values land where first touched.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A basic index places one shard"), nullptr;
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    handle_type_* handle = nullptr;
    bool picked = false;
    auto rung = [&]<typename arm_type_>(std::type_identity<arm_type_>) noexcept {
        if constexpr (std::is_same_v<typename arm_type_::executor_t, executor_type_>) {
            if (picked) return;
            if (arm_type_::cpu_class_k != cpu_class_t::serial_k && !numkong_admits(available, arm_type_::cpu_class_k))
                return;
            if (!stores_scalar_of<arm_type_>(geometry)) return;
            picked = true;
            handle = new (std::nothrow)
                handle_type_(std::in_place_type<arm_type_>, forkunion_allocator_t {no_memory_domain_k});
            if (!handle)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), void();
            if (!traits_t::freeze(std::get<arm_type_>(handle->variants), geometry, arguments, outcome))
                return delete handle, handle = nullptr, void();
            std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                          usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(arm_type_::cpu_class_k)));
        }
    };
    [&]<typename... arm_types_>(std::type_identity<std::variant<arm_types_...>>) noexcept {
        (rung(std::type_identity<arm_types_> {}), ...);
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
    handle_t* const index = make_cpu_handle<handle_t, forkunion_executor_t>(geometry, *arguments, arguments->ram,
                                                                            outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Places a GPU index on the one device the arguments name, refusing what no device kit serves. */
template <typename arguments_type_>
void basic_init_vram(arguments_type_* arguments) noexcept {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
#if USEARCH_ARCH_CUDA_
    using handle_t = handle_of<arguments_type_>;
    using traits_t = typename handle_t::traits_t;
    typename traits_t::geometry_t geometry;
    if (!traits_t::state(*arguments, geometry, outcome) || !traits_t::admits(geometry, *arguments, outcome)) return;
    if (arguments->values_per_key)
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU index bounds no key's entries"), void();
    usearch_vram_options_t const& vram = arguments->vram;
    if (vram.devices_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A basic index places one shard"), void();
    std::size_t const ordinal = vram.devices && vram.devices_count ? static_cast<std::size_t>(vram.devices[0]) : 0;
    std::span<gpu_device_t const> const listed = gpu_devices();
    if (ordinal >= listed.size())
        return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), void();
    gpu_device_t const& device = listed[ordinal];
    if (device.vendor != gpu_vendor_t::nvidia_k)
        return propagate_error(usearch_status_device_k, outcome, "No basic GPU kernels serve this vendor yet"), void();
    handle_t* index = nullptr;
    traits_t::with_vram_arm(geometry, [&]<typename arm_type_>(std::type_identity<arm_type_>) noexcept {
        index = new (std::nothrow) handle_t(std::in_place_type<arm_type_>, static_cast<int>(device.vendor_ordinal));
        if (!index)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), void();
        if (!traits_t::freeze(std::get<arm_type_>(index->variants), geometry, *arguments, outcome))
            return delete index, index = nullptr, void();
        std::snprintf(index->capabilities_string, sizeof(index->capabilities_string), "%s",
                      usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)));
    });
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
    handle_t* const index = make_cpu_handle<handle_t, forkunion_executor_t>(geometry, *arguments, arguments->ram,
                                                                            outcome);
    if (!index) return;
    bool loaded = false;
    visit_ram_variant(*index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, executor->size()};
        loaded = propagate_error(arm.load(path, *executor, ticks), outcome);
    });
    if (!loaded) return delete index, void();
    *arguments->index = index;
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
    handle_t* const index = make_cpu_handle<handle_t, nvme_executor_t>(geometry, *arguments, usearch_ram_options_t {},
                                                                       outcome);
    if (!index) return;
    bool viewed = false;
    visit_nvme_variant(*index, [&](auto& arm) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        viewed = executor && propagate_error(arm.view(path, to_span(paths), *executor), outcome);
    });
    if (!viewed) return delete index, void();
    *arguments->index = index;
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
    visit_writable(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        if constexpr (is_vram_index<arm_type_>) propagate_error(arm.reserve(*arguments), outcome);
        else propagate_error(arm.index.reserve(arguments->entries).status, outcome);
    });
}

template <auto verb_, typename arguments_type_>
void basic_clear(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_ram_variant(*arguments->index, outcome, [&](auto& arm) noexcept { propagate_error(arm.clear(), outcome); });
}

/** Refuses a verb the interface declares and no basic residency serves yet, zeroing what it
 *  reports; each is intended work, which replaces this refusal where it lands. */
template <auto verb_, typename arguments_type_>
void basic_unserved(arguments_type_* arguments) noexcept {
    if constexpr (requires { arguments->executor; })
        if (submitted<verb_>(arguments)) return;
    usearch_arena_t arena = nullptr;
    if constexpr (requires { arguments->arena; }) arena = arguments->arena;
    call_outcome_t outcome(arguments->outcome, arena);
    if constexpr (requires { *arguments->index = nullptr; }) *arguments->index = nullptr;
    if constexpr (requires { arguments->stats; })
        if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency serves this verb yet");
}

#pragma endregion Lifetime

#pragma region Refinement

/** The plan of a RAM arm with the donors it lands: their arms, whose handles a call frees once
 *  landed, their indexes, which the plan nulls as it lands them, and where the calls stand. Every
 *  other arm refines nothing. */
template <typename arm_type_>
struct basic_refinement_of {
    using type = std::monostate;
};
template <typename arm_type_>
    requires is_ram_index<arm_type_>
struct basic_refinement_of<arm_type_> {
    struct type {
        growing_buffer<arm_type_*, forkunion_allocator_t> donors;
        growing_buffer<typename arm_type_::index_t*, forkunion_allocator_t> landing;
        typename arm_type_::refine_plan_t plan;
        refine_cursor_t cursor {};
    };
};

/** The refinements of every arm of @p variant_type_, as one variant of their own. */
template <typename variant_type_>
struct basic_refinements_of;
template <typename... arm_types_>
struct basic_refinements_of<std::variant<arm_types_...>> {
    using type = std::variant<std::monostate, typename basic_refinement_of<arm_types_>::type...>;
};

/** What a modality's refinement handle owns: one RAM arm's plan, which keeps its index closed until
 *  it dies, and the caller's donor slots it nulls. */
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

/** Opens @p arm 's refinement of @p index after the @p count @p donors land, each checked against
 *  the arm's type and geometry once no deferred call names it; empty once @p outcome records why
 *  not. */
template <typename handle_type_, typename arm_type_>
std::optional<typename basic_refinement_of<arm_type_>::type> basic_open_refinement(
    handle_type_ const* index, arm_type_& arm, handle_type_** donors, std::size_t count, std::size_t expansion,
    usearch_outcome_t* outcome) noexcept {
    using traits_t = typename handle_type_::traits_t;
    using refining_t = typename basic_refinement_of<arm_type_>::type;
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
        arm_type_* const typed = std::get_if<arm_type_>(&donor->variants);
        if (!typed || !traits_t::matches(arm.geometry, typed->geometry))
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "A donor must share the index's residency, class and geometry"),
                   std::nullopt;
        finish_calls_using(donor);
        typed_donors.push_back_reserved(typed);
        landing.push_back_reserved(&typed->index);
    }
    // The plan borrows the landing slots, whose storage moves along with the buffer.
    auto [plan, status] = arm.plan_refine(to_writable_span(landing), expansion);
    if (failed(status)) return propagate_error(status, outcome), std::nullopt;
    return refining_t {std::move(typed_donors), std::move(landing), std::move(plan)};
}

/** Frees the handles of the donors @p refining landed, folding what their arms count into @p arm
 *  first, and nulls the caller's slots. */
template <typename handle_type_, typename arm_type_, typename refining_type_>
void basic_free_landed(arm_type_& arm, refining_type_& refining, handle_type_** donors) noexcept {
    for (std::size_t position = 0; position != refining.landing.size(); ++position) {
        if (refining.landing[position] || !donors[position]) continue;
        if constexpr (requires { arm.absorbed(*refining.donors[position]); }) arm.absorbed(*refining.donors[position]);
        delete std::exchange(donors[position], nullptr);
    }
}

/** Writes what a refinement's @p refined result reports into the outputs @p arguments name. */
template <typename arguments_type_>
void basic_report_refined(us::basic::refine_result_t const& refined, arguments_type_ const& arguments) noexcept {
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
template <typename arm_type_, typename refining_type_, typename arguments_type_>
status_t basic_finish_refinement(arm_type_& arm, refining_type_& refining, arguments_type_ const& arguments,
                                 forkunion_executor_t& executor) noexcept {
    auto& plan = refining.plan;
    if (plan.landing()) return status_t::success_k;
    // The re-selection a cancel left is skipped rather than finished.
    plan.stop();
    if (plan.reselecting()) refining.cursor.taken = plan.steps().size();
    progress_ticks_t heal_alone {{}, 0};
    if (status_t const healed = arm.refine(plan, refining.cursor, heal_alone, executor); failed(healed)) return healed;
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
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        auto refining = basic_open_refinement(arguments->index, arm, donors, count, arguments->expansion_factor,
                                              outcome);
        if (!refining) return;
        progress_ticks_t ticks {arguments->progress, refining->plan.audit.visible};
        status_t const ran = arm.refine(refining->plan, refining->cursor, ticks, *executor);
        basic_free_landed(arm, *refining, donors);
        if (failed(ran) && ran != status_t::cancelled_k) return propagate_error(ran, outcome), void();
        // A stopped pass still heals, so its report stands beside the cancel.
        if (status_t const finished = basic_finish_refinement(arm, *refining, *arguments, *executor); failed(finished))
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
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        using refining_t = typename basic_refinement_of<arm_type_>::type;
        usearch_executor_handle_t call_executor;
        if (!executor_of(arm, arguments->executor, call_executor, outcome)) return;
        auto refining = basic_open_refinement(arguments->index, arm, arguments->donors, arguments->count,
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

/** Visits the arm the refinement @p arguments name was opened on with its plan, refusing a call
 *  naming another index or executor; the refinement serves one arm, so any other is refused too. */
template <typename arguments_type_, typename visitor_type_>
void visit_refinement(arguments_type_ const& arguments, usearch_outcome_t* outcome, visitor_type_&& visitor) noexcept {
    auto* const refinement = arguments.refinement;
    if (!refinement)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement must not be null"), void();
    if (arguments.index != refinement->opened_on || arguments.executor != refinement->opened_with)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The refinement was opened on another index or executor"),
               void();
    visit_ram_variant(*arguments.index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        auto* const refining = std::get_if<typename basic_refinement_of<arm_type_>::type>(&refinement->refining);
        if (!refining)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement serves another arm"),
                   void();
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments.executor, call_executor, outcome);
        if (!executor) return;
        visitor(arm, *refining, *executor);
    });
}

template <auto verb_, typename arguments_type_>
void basic_refinement_next(arguments_type_* arguments) noexcept {
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->visited) *arguments->visited = 0;
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->stats) *arguments->stats = {};
    visit_refinement(*arguments, outcome, [&](auto& arm, auto& refining, auto& executor) noexcept {
        progress_ticks_t ticks {arguments->progress, refining.plan.audit.visible};
        status_t const ran = arm.refine(refining.plan, refining.cursor, ticks, executor);
        basic_free_landed(arm, refining, arguments->refinement->donors);
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
    visit_refinement(*arguments, outcome, [&](auto& arm, auto& refining, auto& executor) noexcept {
        propagate_error(basic_finish_refinement(arm, refining, *arguments, executor), outcome);
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
        [&](auto& arm) noexcept {
            auto& description = *arguments->description;
            description.size = arm.index.size();
            description.capacity = arm.index.capacity();
            description.memory_usage = sizeof(*index) + arm.index.memory_usage();
            description.capabilities = index->capabilities_string;
            traits_t::describe(arm, description);
        },
        index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Whether @p traits_type_ measures a key's entries from their values of @p value_type_, which a
 *  count then reads. */
template <typename traits_type_, typename stats_type_, typename value_type_>
concept tallies = requires(stats_type_& stats, value_type_ const& value) { traits_type_::tally(stats, value); };

/** Adds @p value to @p stats, from any thread, where @p traits_type_ measures such values. */
template <typename traits_type_, typename stats_type_, typename value_type_>
void tally_entry(stats_type_& stats, value_type_ const& value) noexcept {
    if constexpr (tallies<traits_type_, stats_type_, value_type_>) traits_type_::tally(stats, value);
}

/** Counts a key's entries, and tallies their sizes where the traits measure any: an NVMe view reads
 *  them only then. */
template <auto verb_, typename arguments_type_>
void basic_count(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    using stats_t = std::remove_pointer_t<decltype(arguments->key_stats)>;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    stats_t& stats = *arguments->key_stats;
    std::array<us::utilities::uuid_t, 1> const identities {uuid_of(arguments->key)};
    auto count_nvme = [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        using view_t = decltype(typename arm_type_::index_t::codec_t {}.view(bytes_view_t {}));
        if constexpr (tallies<traits_t, stats_t, view_t>) {
            auto take_count = [&](std::size_t entries) noexcept {
                return stats.entries = entries, status_t::success_k;
            };
            auto measure = [&](std::size_t, std::size_t, std::size_t, view_t const& value) noexcept {
                tally_entry<traits_t>(stats, value);
            };
            propagate_error(values_under(arm.index, arguments->key, *executor, take_count, measure), outcome);
        }
        else {
            std::array<std::size_t, 1> entries {};
            status_t const counted = arm.index.count(identities, entries, *executor);
            stats.entries = entries[0];
            propagate_error(counted, outcome);
        }
    };
    if (visit_nvme_variant(*arguments->index, count_nvme)) return;
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        arm.index.find_each(identities[0], [&](typename arm_type_::index_t::node_id_t node) noexcept {
            ++stats.entries;
            tally_entry<traits_t>(stats, arm.index.published_value_of(node));
        });
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
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
    visit_writable(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        if (crossed(arm.insert(*arguments, ticks, *executor), arguments->stats, outcome))
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
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if (is_vram_index<arm_type_> && arguments->predicate.callback)
                return propagate_error(usearch_status_device_k, outcome,
                                       "A host predicate cannot narrow a device walk"),
                       void();
            // Matches land on host pages, which a device arena takes over and a host one keeps.
            bool const host_landing = arguments->arena && arguments->arena->device == host_device_k;
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              host_landing ? host_device_k : device_of(arm),
                                                              {address_of(arguments->queries)}, outcome);
            if (!arena) return;
            if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            if (!traits_t::accepts(arm, *arguments, outcome)) return;
            matches_landing_t landing;
            if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            // A device walk stages its queries on the arena's host pages, where its matches land.
            auto const searched = [&]() noexcept {
                if constexpr (requires { arm.search(*arguments, arena->memory, landing, ticks, *executor); })
                    return arm.search(*arguments, arena->memory, landing, ticks, *executor);
                else return arm.search(*arguments, landing, ticks, *executor);
            }();
            if (!crossed(searched, arguments->stats, outcome) || !publish_matches(*arena, count, landing, outcome))
                return;
            *arguments->matches = landing.view();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

/** Exports a key's entries into the arena, in insertion order; a view's values land in its threads'
 *  arenas first, as their sizes are unknown until read. */
template <auto verb_, typename arguments_type_>
void basic_get(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    using kept_t = typename traits_t::kept_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *traits_t::output_of(*arguments) = {};
    if (arguments->written) *arguments->written = 0;
    if constexpr (requires { traits_t::exports(*arguments, outcome); })
        if (!traits_t::exports(*arguments, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    auto get_nvme = [&](auto& arm) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        std::size_t held = 0;
        kept_t* kept = nullptr;
        auto carve_entries = [&](std::size_t entries) noexcept {
            held = entries;
            if (!held) return status_t::success_k;
            kept = carve<kept_t>(arena->memory, held);
            return kept ? status_t::success_k : status_t::bad_alloc_k;
        };
        status_t keeping = status_t::success_k;
        auto keep = [&](std::size_t, std::size_t rank, std::size_t thread, auto const& value) noexcept {
            keep_first_failure(keeping, traits_t::keep(*executor, thread, value, kept[rank]));
        };
        if (status_t const read = values_under(arm.index, arguments->key, *executor, carve_entries, keep); failed(read))
            return propagate_error(read, outcome), void();
        if (failed(keeping)) return propagate_error(keeping, outcome), void();
        if (!held) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        auto kept_at = [&](std::size_t entry) noexcept -> decltype(auto) { return kept[entry]; };
        if (!traits_t::land(arm, held, kept_at, arena->memory, *arguments, outcome)) return;
        if (arguments->written) *arguments->written = held;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    };
    if (visit_nvme_variant(*arguments->index, get_nvme)) return;
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        if (status_t const rewound = executor->reserve(0); failed(rewound))
            return propagate_error(rewound, outcome), void();
        auto const ids = ids_under(arm.index, arguments->key, thread_allocator_t {executor->arena(0)});
        if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        auto value_at = [&](std::size_t entry) noexcept -> decltype(auto) {
            return arm.index.published_value_of(ids[entry]);
        };
        if (!traits_t::land(arm, ids.size(), value_at, arena->memory, *arguments, outcome)) return;
        if (arguments->written) *arguments->written = ids.size();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Insertion and Search

#pragma region Cursors and Serialization

template <typename arguments_type_>
void basic_cursor_init(arguments_type_* arguments) noexcept {
    using cursor_handle_t = std::remove_pointer_t<std::remove_pointer_t<decltype(arguments->cursor)>>;
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    cursor_handle_t* const cursor = new (std::nothrow) cursor_handle_t {};
    if (!cursor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    cursor->index = arguments->index;
    auto open_nvme = [&](auto&) noexcept { cursor->cursor.template emplace<std::size_t>(0); };
    if (!visit_nvme_variant(*arguments->index, open_nvme))
        visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
            auto& opened = cursor->cursor.template emplace<ram_cursor_of<arm_type_>>(
                arm.index, forkunion_allocator_t {no_memory_domain_k});
            // Stands on the first entry, so every pull copies from where it stands; an empty
            // snapshot stays past the end.
            if (opened.visible_count()) [[maybe_unused]]
                status_t const entered = opened.seek(0);
        });
    // A GPU arm opened nothing, and the outcome says why.
    if (std::holds_alternative<std::monostate>(cursor->cursor)) return delete cursor, void();
    *arguments->cursor = cursor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Pulls the next keys and, when asked, their values into the arena; a view reads its rows through
 *  the executor, landing values in its threads' arenas first. */
template <auto verb_, typename arguments_type_>
void basic_cursor_next(arguments_type_* arguments) noexcept {
    using traits_t = typename handle_of<arguments_type_>::traits_t;
    using kept_t = typename traits_t::kept_t;
    if (submitted<verb_>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    auto* const output = traits_t::output_of(*arguments);
    if (output) *output = {};
    *arguments->written = 0;
    auto* const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    auto next_nvme = [&](auto& arm) noexcept {
        std::size_t* const row = std::get_if<std::size_t>(&cursor->cursor);
        if (!row) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        std::size_t const taking = min_of(arm.index.size() - *row, arguments->wanted);
        if (!taking) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
        kept_t* const kept = output ? carve<kept_t>(arena->memory, taking) : nullptr;
        if (!keys || (output && !kept))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        status_t keeping = status_t::success_k;
        auto keep = [&](std::size_t entry, std::size_t thread, auto const& value) noexcept {
            if (kept) keep_first_failure(keeping, traits_t::keep(*executor, thread, value, kept[entry]));
        };
        std::span<us::utilities::uuid_t> const identities {reinterpret_cast<us::utilities::uuid_t*>(keys), taking};
        if (status_t const read = arm.index.read_rows(*row, identities, *executor, keep); failed(read))
            return propagate_error(read, outcome), void();
        if (failed(keeping)) return propagate_error(keeping, outcome), void();
        auto kept_at = [&](std::size_t entry) noexcept -> decltype(auto) { return kept[entry]; };
        if (output && !traits_t::land(arm, taking, kept_at, arena->memory, *arguments, outcome)) return;
        *row += taking;
        *arguments->keys = {keys, sizeof(usearch_key_t)};
        *arguments->written = taking;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    };
    auto next_ram = [&]<typename arm_type_>(arm_type_& arm) noexcept {
        using node_id_t = typename arm_type_::index_t::node_id_t;
        auto* const typed = std::get_if<ram_cursor_of<arm_type_>>(&cursor->cursor);
        if (!typed) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
        std::size_t const remaining = *typed ? typed->visible_count() - typed->id() : 0;
        std::size_t const taking = min_of(remaining, arguments->wanted);
        usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
        if (taking && !keys)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        node_id_t const first = typed->id();
        for (std::size_t entry = 0; entry != taking; ++entry)
            keys[entry] = std::bit_cast<usearch_key_t>(arm.index.key_of(static_cast<node_id_t>(first + entry)).value);
        auto value_at = [&](std::size_t entry) noexcept -> decltype(auto) {
            return arm.index.published_value_of(static_cast<node_id_t>(first + entry));
        };
        if (output && !traits_t::land(arm, taking, value_at, arena->memory, *arguments, outcome)) return;
        for (std::size_t step = 0; step != taking; ++step) ++*typed;
        *arguments->keys = {keys, sizeof(usearch_key_t)};
        *arguments->written = taking;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    };
    bool exported = true;
    if constexpr (requires { traits_t::exports(*arguments, outcome); })
        exported = !arena || traits_t::exports(*arguments, outcome);
    if (arena && exported && !visit_nvme_variant(*arguments->index, next_nvme))
        visit_ram_variant(*arguments->index, outcome, next_ram);
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
    auto refuse_nvme = [&](auto&) noexcept {
        propagate_error(usearch_status_unsupported_k, outcome, "An NVMe view already is its checkpoint");
    };
    if (visit_nvme_variant(*arguments->index, refuse_nvme)) return;
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, executor->size()};
        propagate_error(arm.save(path, *executor, ticks), outcome);
    });
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
