/**
 *  @file src/geospatial_index.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The geospatial modality's C surface: the variant over every compiled geospatial arm, the
 *      handle owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Arms are ordered newest class first, the serial floor last. Their bodies compile in the basic
 *  class units, never here: the @c extern declarations below keep them out, as the dense dispatch
 *  unit keeps its own.
 */
#include <cmath>   // `std::isnan`
#include <cstdio>  // `std::snprintf` - the capability string
#include <cstring> // `std::memcpy`

#include <atomic>      // `std::atomic_flag` - a cursor's one pull at a time
#include <new>         // `std::nothrow`
#include <string_view> // `std::string_view` - checkpoint paths
#include <variant>     // `std::variant`, `std::visit`

#include "basic_specializations.cuh"

#if NUMKONG_TARGET_SKYLAKE
extern template struct geospatial_arm<cpu_class_t::intel_skylake_k>;
#endif
#if NUMKONG_TARGET_HASWELL
extern template struct geospatial_arm<cpu_class_t::intel_haswell_k>;
#endif
#if NUMKONG_TARGET_NEON
extern template struct geospatial_arm<cpu_class_t::aws_g2_k>;
#endif
#if NUMKONG_TARGET_RVV
extern template struct geospatial_arm<cpu_class_t::alibaba_c930_k>;
#endif
#if NUMKONG_TARGET_V128RELAXED
extern template struct geospatial_arm<cpu_class_t::wasm64_extended_k>;
#endif
extern template struct geospatial_arm<cpu_class_t::serial_k>;

#pragma region Handles

/** The one dynamically-typed geospatial index every binding consumes: a variant over the arms this
 *  build compiled, each the basic index over observations scored with one class's kit. */
struct usearch_geospatial_index_handle_t {

    using variants_t = std::variant<
#if NUMKONG_TARGET_SKYLAKE
        geospatial_arm<cpu_class_t::intel_skylake_k>,
#endif
#if NUMKONG_TARGET_HASWELL
        geospatial_arm<cpu_class_t::intel_haswell_k>,
#endif
#if NUMKONG_TARGET_NEON
        geospatial_arm<cpu_class_t::aws_g2_k>,
#endif
#if NUMKONG_TARGET_RVV
        geospatial_arm<cpu_class_t::alibaba_c930_k>,
#endif
#if NUMKONG_TARGET_V128RELAXED
        geospatial_arm<cpu_class_t::wasm64_extended_k>,
#endif
        geospatial_arm<cpu_class_t::serial_k>>;

    variants_t variants;

    /** The class whose kit the arm scores with, named once at construction. */
    char capabilities_string[32] {};

    template <typename arm_type_>
    usearch_geospatial_index_handle_t(std::in_place_type_t<arm_type_> tag,
                                      forkunion_allocator_t const& allocator) noexcept
        : variants(tag, allocator) {}
};

/** The basic cursor over the index, whose type no class changes - only the kit a metric scores
 *  with. */
using geospatial_cursor_t = basic::ram_cursor<typename geospatial_arm<cpu_class_t::serial_k>::index_t,
                                              synchronization_t::concurrent_k, forkunion_allocator_t>;

struct usearch_geospatial_cursor_handle_t {
    geospatial_cursor_t cursor;

    /** The handle the cursor was opened on, which every pull names again. */
    usearch_geospatial_index_t index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The engine's form of a C terrain; a path stays the NVMe residency's to read. */
static us::geospatial::terrain_t terrain_of(usearch_geospatial_terrain_t const& terrain) noexcept {
    return {terrain.rows,
            terrain.columns,
            terrain.north_latitude_degrees,
            terrain.west_longitude_degrees,
            terrain.row_step_degrees,
            terrain.column_step_degrees,
            terrain.heights,
            terrain.heights_stride,
            terrain.callback,
            terrain.state};
}

/** Whether @p terrain names exactly one source and none an in-memory handle cannot serve; false
 *  once @p outcome records why not. */
static bool accepts_terrain(usearch_geospatial_terrain_t const& terrain, usearch_outcome_t* outcome) noexcept {
    if (terrain.path.data)
        return !propagate_error(usearch_status_unsupported_k, outcome, "A terrain path needs the NVMe residency");
    if (terrain.heights && terrain.callback)
        return !propagate_error(usearch_status_invalid_argument_k, outcome,
                                "A terrain is either borrowed or filled, never both");
    return true;
}

/** Allocates the handle holding @p arm_type_ with @p connectivity and @p terrain; null once
 *  @p outcome records why not. */
template <typename arm_type_>
static usearch_geospatial_index_t make_geospatial_handle(std::size_t connectivity,
                                                         usearch_geospatial_terrain_t const& terrain,
                                                         usearch_outcome_t* outcome) noexcept {
    usearch_geospatial_index_t const handle = new (std::nothrow)
        usearch_geospatial_index_handle_t(std::in_place_type<arm_type_>, forkunion_allocator_t {no_memory_domain_k});
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    arm_type_& arm = std::get<arm_type_>(handle->variants);
    if (connectivity) arm.connectivity = connectivity;
    if (status_t const adopted = arm.adopt_terrain(terrain_of(terrain)); failed(adopted)) {
        delete handle;
        return propagate_error(adopted, outcome, "The terrain could not be adopted"), nullptr;
    }
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(arm_type_::cpu_class_k)));
    return handle;
}

/** One rung: taken when the machine reports the class, else falls through; the serial floor ends
 *  every ladder, and a class that picks no kit of its own is skipped without instantiating its
 *  arm. */
template <cpu_class_t cpu_class_ = cpu_class_t::serial_k, cpu_class_t... lower_classes_>
static usearch_geospatial_index_t descend_geospatial_ladder(std::size_t connectivity,
                                                            usearch_geospatial_terrain_t const& terrain,
                                                            nk_capability_t available,
                                                            usearch_outcome_t* outcome) noexcept {
    using arm_t = geospatial_arm<cpu_class_>;
    if constexpr (cpu_class_ == cpu_class_t::serial_k)
        return make_geospatial_handle<arm_t>(connectivity, terrain, outcome);
    else {
        if constexpr (geospatial_rung_advances<cpu_class_>()) {
            static_assert(variant_carries<arm_t, usearch_geospatial_index_handle_t::variants_t>,
                          "This rung advances past the one below - list its arm in the variant");
            if (numkong_admits(available, cpu_class_))
                return make_geospatial_handle<arm_t>(connectivity, terrain, outcome);
        }
        else {
            static_assert(!variant_carries<arm_t, usearch_geospatial_index_handle_t::variants_t>,
                          "This rung no longer advances past the one below - drop its arm from the variant list");
        }
        return descend_geospatial_ladder<lower_classes_...>(connectivity, terrain, available, outcome);
    }
}

/** Picks the arm: CPU classes newest first, each compiled only where its kit differs from the class
 *  it degrades to; null once @p outcome records why not. */
static usearch_geospatial_index_t make_geospatial_index(std::size_t connectivity,
                                                        usearch_geospatial_terrain_t const& terrain,
                                                        usearch_ram_options_t const& ram,
                                                        usearch_outcome_t* outcome) noexcept {
    // One basic index per handle, so one memory domain at most.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A geospatial index places one shard"), nullptr;
    if (!accepts_terrain(terrain, outcome)) return nullptr;
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_geospatial_ladder<cpu_class_t::intel_skylake_k, cpu_class_t::intel_haswell_k>(connectivity, terrain,
                                                                                                 available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_geospatial_ladder<cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>(connectivity, terrain, available,
                                                                                     outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_geospatial_ladder<cpu_class_t::alibaba_c930_k>(connectivity, terrain, available, outcome);
#elif NUMKONG_ARCH_WASM_
    return descend_geospatial_ladder<cpu_class_t::wasm64_extended_k>(connectivity, terrain, available, outcome);
#else
    return descend_geospatial_ladder<>(connectivity, terrain, available, outcome);
#endif
}

/** Whether @p observations names storage for @p count entries, each with a position and an interval
 *  in order; false once @p outcome records why not. */
static bool accepts_observations(usearch_geospatial_observation_t const* observations, std::size_t count,
                                 usearch_outcome_t* outcome) noexcept {
    if (!observations)
        return !propagate_error(usearch_status_invalid_argument_k, outcome, "The observations must not be null");
    for (std::size_t row = 0; row != count; ++row) {
        usearch_geospatial_observation_t const& observation = observations[row];
        if (std::isnan(observation.latitude_degrees) || std::isnan(observation.longitude_degrees))
            return !propagate_error(usearch_status_invalid_argument_k, outcome,
                                    "An observation needs a latitude and a longitude");
        if (observation.end_seconds < observation.begin_seconds)
            return !propagate_error(usearch_status_invalid_argument_k, outcome, "An observation ends before it begins");
    }
    return true;
}

/** Whether @p relation names one of the four, and @c visible only on @p arm with terrain; false
 *  once @p outcome records why not. */
template <typename arm_type_>
static bool accepts_relation(arm_type_ const& arm, usearch_geospatial_relation_t relation,
                             usearch_outcome_t* outcome) noexcept {
    if (relation != usearch_geospatial_relation_near_k && relation != usearch_geospatial_relation_visible_k &&
        relation != usearch_geospatial_relation_sees_k && relation != usearch_geospatial_relation_seen_k)
        return !propagate_error(usearch_status_invalid_argument_k, outcome, "Unknown geospatial relation");
    if (relation == usearch_geospatial_relation_visible_k && arm.terrain.empty())
        return !propagate_error(usearch_status_unsupported_k, outcome, "Lines of sight need a handle with terrain");
    return true;
}

/** Reads a checkpoint's connectivity, the one field it records beside its observations. */
static std::size_t checkpoint_connectivity(std::string_view path, status_t& status) noexcept {
    std::size_t connectivity = 0;
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") connectivity = decimal_in(field.value);
    };
    auto const [metadata, read_status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
    status = read_status;
    return connectivity;
}

#pragma endregion Handles

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_geospatial_init_ram(usearch_geospatial_init_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_geospatial_index_t const index = make_geospatial_index(arguments->connectivity, arguments->terrain,
                                                                   arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_geospatial_init_vram(usearch_geospatial_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "No GPU kernels serve geospatial yet");
}

USEARCH_EXPORT void usearch_geospatial_load_ram(usearch_geospatial_load_ram_t* arguments) {
    if (submitted<usearch_geospatial_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    status_t described = status_t::success_k;
    std::size_t const connectivity = checkpoint_connectivity(path, described);
    if (failed(described)) return propagate_error(described, outcome), void();
    usearch_geospatial_index_t const index = make_geospatial_index(connectivity, arguments->terrain, arguments->ram,
                                                                   outcome);
    if (!index) return;
    bool loaded = false;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, executor->size()};
            loaded = propagate_error(arm.load(path, *executor, ticks), outcome);
        },
        index->variants);
    if (!loaded) return delete index, void();
    *arguments->index = index;
}

USEARCH_EXPORT void usearch_geospatial_load_vram(usearch_geospatial_load_vram_t* arguments) {
    if (submitted<usearch_geospatial_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "No GPU kernels serve geospatial yet");
}

USEARCH_EXPORT void usearch_geospatial_view_ram(usearch_geospatial_view_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency serves a view yet, so load instead");
}

USEARCH_EXPORT void usearch_geospatial_free(usearch_geospatial_index_t index) {
    finish_calls_using(index);
    delete index;
}

USEARCH_EXPORT void usearch_geospatial_reserve(usearch_geospatial_reserve_t* arguments) {
    if (submitted<usearch_geospatial_reserve>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::visit([&](auto& arm) noexcept { propagate_error(arm.index.reserve(arguments->entries).status, outcome); },
               arguments->index->variants);
}

USEARCH_EXPORT void usearch_geospatial_clear(usearch_geospatial_clear_t* arguments) {
    if (submitted<usearch_geospatial_clear>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::visit(
        [&](auto& arm) noexcept {
            status_t const cleared = arm.index.clear();
            if (succeeded(cleared))
                standard_atomic_ref<u32_t>(arm.largest_deviation_metres).store(0, std::memory_order_relaxed);
            propagate_error(cleared, outcome);
        },
        arguments->index->variants);
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT void usearch_geospatial_describe(usearch_geospatial_describe_t* arguments) {
    if (submitted<usearch_geospatial_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_t const index = arguments->index;
    std::visit(
        [&](auto& arm) noexcept {
            usearch_geospatial_index_description_t& description = *arguments->description;
            description.metadata = {arm.connectivity};
            description.size = arm.index.size();
            description.capacity = arm.index.capacity();
            description.memory_usage = sizeof(*index) + arm.index.memory_usage() + arm.terrain_pack.memory_usage();
            description.capabilities = index->capabilities_string;
        },
        index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_geospatial_count(usearch_geospatial_count_t* arguments) {
    if (submitted<usearch_geospatial_count>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            arm.index.find_each(uuid_of(arguments->key),
                                [&](typename arm_type_::node_id_t) noexcept { ++arguments->key_stats->entries; });
        },
        arguments->index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT void usearch_geospatial_insert(usearch_geospatial_insert_t* arguments) {
    if (submitted<usearch_geospatial_insert>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    std::size_t const count = arguments->count;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!arguments->keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must not be null"), void();
    if (arguments->keys.stride && arguments->keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (!accepts_observations(arguments->observations, count, outcome)) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            if (crossed(arm.insert(arguments->keys, arguments->observations, count, arguments->expansion_factor, ticks,
                                   *executor),
                        arguments->stats, outcome))
                propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_geospatial_refine(usearch_geospatial_refine_t* arguments) {
    if (submitted<usearch_geospatial_refine>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency refines yet");
}

USEARCH_EXPORT void usearch_geospatial_search(usearch_geospatial_search_t* arguments) {
    if (submitted<usearch_geospatial_search>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                      {arguments->queries}, outcome);
    if (!arena) return;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!accepts_observations(arguments->queries, count, outcome)) return;
    matches_landing_t landing;
    if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if (!accepts_relation(arm, arguments->relation, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            if (!crossed(
                    arm.search(arguments->queries, count, static_cast<us::geospatial::relation_t>(arguments->relation),
                               arguments->tolerance, wanted, arguments->expansion_factor, arguments->predicate, landing,
                               ticks, *executor),
                    arguments->stats, outcome))
                return;
            *arguments->matches = landing.view();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_geospatial_within(usearch_geospatial_within_t* arguments) {
    if (submitted<usearch_geospatial_within>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                      {arguments->queries}, outcome);
    if (!arena) return;
    if (!(arguments->tolerance.metres > 0))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A containment walk needs a positive radius in metres"),
               void();
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!accepts_observations(arguments->queries, count, outcome)) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if (!accepts_relation(arm, arguments->relation, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            usearch_matches_tape_view_t matches {};
            std::size_t written = 0;
            if (!crossed(arm.within(arguments->queries, count,
                                    static_cast<us::geospatial::relation_t>(arguments->relation), arguments->tolerance,
                                    arguments->predicate, arena->memory, matches, written, ticks, *executor),
                         arguments->stats, outcome))
                return;
            *arguments->matches = matches;
            if (arguments->written) *arguments->written = written;
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_geospatial_get(usearch_geospatial_get_t* arguments) {
    if (submitted<usearch_geospatial_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->observations = nullptr;
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k, {}, outcome);
    if (!arena) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            if (status_t const rewound = executor->reserve(0); failed(rewound))
                return propagate_error(rewound, outcome), void();
            thread_allocator_t const scratch {executor->arena(0)};
            auto const ids = ids_under(arm.index, arguments->key, scratch);
            if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            usearch_geospatial_observation_t* const observations = carve<usearch_geospatial_observation_t>(
                arena->memory, ids.size());
            if (!observations)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            for (std::size_t entry = 0; entry != ids.size(); ++entry)
                observations[entry] = observation_of(
                    us::geospatial::to_geodetic_observation(arm.index.published_value_of(ids[entry])));
            *arguments->observations = observations;
            if (arguments->written) *arguments->written = ids.size();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

#pragma endregion Insertion and Search

#pragma region Cursors and Merging

USEARCH_EXPORT void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    usearch_geospatial_index_t const index = arguments->index;
    usearch_geospatial_cursor_t cursor = nullptr;
    std::visit(
        [&](auto& arm) noexcept {
            cursor = new (std::nothrow) usearch_geospatial_cursor_handle_t {
                geospatial_cursor_t {arm.index, forkunion_allocator_t {no_memory_domain_k}},
                index,
                {}};
        },
        index->variants);
    if (!cursor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    // Stands on the first entry, so every pull copies from where it stands; an empty snapshot stays
    // past the end.
    if (cursor->cursor.visible_count()) [[maybe_unused]]
        status_t const entered = cursor->cursor.seek(0);
    *arguments->cursor = cursor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments) {
    if (submitted<usearch_geospatial_cursor_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->observations) *arguments->observations = nullptr;
    *arguments->written = 0;
    usearch_geospatial_cursor_t const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k, {}, outcome);
    if (arena)
        std::visit(
            [&]<typename arm_type_>(arm_type_& arm) noexcept {
                geospatial_cursor_t& typed = cursor->cursor;
                std::size_t const remaining = typed ? typed.visible_count() - typed.id() : 0;
                std::size_t const taking = min_of(remaining, arguments->wanted);
                usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
                usearch_geospatial_observation_t* const observations =
                    arguments->observations ? carve<usearch_geospatial_observation_t>(arena->memory, taking) : nullptr;
                if (taking && (!keys || (arguments->observations && !observations)))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                typename arm_type_::node_id_t const first = typed.id();
                for (std::size_t entry = 0; entry != taking; ++entry) {
                    auto const node = static_cast<typename arm_type_::node_id_t>(first + entry);
                    keys[entry] = std::bit_cast<usearch_key_t>(arm.index.key_of(node).value);
                    if (observations)
                        observations[entry] = observation_of(
                            us::geospatial::to_geodetic_observation(arm.index.published_value_of(node)));
                }
                for (std::size_t step = 0; step != taking; ++step) ++typed;
                *arguments->keys = {keys, sizeof(usearch_key_t)};
                if (arguments->observations) *arguments->observations = observations;
                *arguments->written = taking;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            },
            arguments->index->variants);
    cursor->busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor) {
    finish_calls_using(cursor);
    delete cursor;
}

USEARCH_EXPORT void usearch_geospatial_sample(usearch_geospatial_sample_t* arguments) {
    if (submitted<usearch_geospatial_sample>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency samples yet");
}

USEARCH_EXPORT void usearch_geospatial_merge(usearch_geospatial_merge_t* arguments) {
    if (submitted<usearch_geospatial_merge>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency absorbs donor handles yet");
}

#pragma endregion Cursors and Merging

#pragma region Serialization

USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_save_t* arguments) {
    if (submitted<usearch_geospatial_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    if (path.ends_with(".json"))
        return propagate_error(usearch_status_unsupported_k, outcome, "A geospatial index saves one monolith"), void();
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, executor->size()};
            propagate_error(arm.save(path, *executor, ticks), outcome);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments) {
    if (submitted<usearch_geospatial_metadata_read>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    status_t described = status_t::success_k;
    std::size_t const connectivity = checkpoint_connectivity(as_text(arguments->path), described);
    if (failed(described)) return propagate_error(described, outcome), void();
    *arguments->metadata = {connectivity};
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

} // extern "C"
