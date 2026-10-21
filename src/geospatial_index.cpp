/**
 *  @file src/geospatial_index.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The geospatial modality's C surface: the variant over every compiled geospatial
 *      alternative, the traits its verbs differ in, every exported entry point as one forward
 *      apiece, and the containment walk only this modality takes.
 *
 *  Variants are ordered newest class first, the serial floor last. Their bodies compile in the
 *  basic class units, never here: the @c extern declarations below keep them out, as the dense
 *  dispatch unit keeps its own.
 */
#include <cmath> // `std::isnan`

#include <bit>         // `std::bit_ceil` - a GPU row's connectivity
#include <string_view> // `std::string_view` - checkpoint paths
#include <variant>     // `std::variant`

#include "basic_index.cuh"
#include "geospatial_specializations.cuh"

#if USEARCH_TARGET_INTEL_SKYLAKE
extern template struct geospatial_terrain<cpu_class_t::intel_skylake_k>;
extern template struct geospatial_ram_variant<cpu_class_t::intel_skylake_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::intel_skylake_k>;
#endif // USEARCH_TARGET_INTEL_SKYLAKE
#if USEARCH_TARGET_INTEL_HASWELL
extern template struct geospatial_terrain<cpu_class_t::intel_haswell_k>;
extern template struct geospatial_ram_variant<cpu_class_t::intel_haswell_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::intel_haswell_k>;
#endif // USEARCH_TARGET_INTEL_HASWELL
#if USEARCH_TARGET_APPLE_M2
extern template struct geospatial_terrain<cpu_class_t::apple_m2_k>;
extern template struct geospatial_ram_variant<cpu_class_t::apple_m2_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::apple_m2_k>;
#endif // USEARCH_TARGET_APPLE_M2
#if USEARCH_TARGET_AWS_G2
extern template struct geospatial_terrain<cpu_class_t::aws_g2_k>;
extern template struct geospatial_ram_variant<cpu_class_t::aws_g2_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::aws_g2_k>;
#endif // USEARCH_TARGET_AWS_G2
#if USEARCH_TARGET_ALIBABA_C930
extern template struct geospatial_terrain<cpu_class_t::alibaba_c930_k>;
extern template struct geospatial_ram_variant<cpu_class_t::alibaba_c930_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::alibaba_c930_k>;
#endif // USEARCH_TARGET_ALIBABA_C930
#if USEARCH_TARGET_WASM64_EXTENDED
extern template struct geospatial_terrain<cpu_class_t::wasm64_extended_k>;
extern template struct geospatial_ram_variant<cpu_class_t::wasm64_extended_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::wasm64_extended_k>;
#endif // USEARCH_TARGET_WASM64_EXTENDED
extern template struct geospatial_terrain<cpu_class_t::serial_k>;
extern template struct geospatial_ram_variant<cpu_class_t::serial_k>;
extern template struct geospatial_nvme_variant<cpu_class_t::serial_k>;

#pragma region Handles

/** Every geospatial variant this build compiled: the basic index over observations scored with one
 *  class's kit, a read-only view of a checkpoint on storage scored and judged with the same, or the
 *  basic index on a GPU. */
using geospatial_variant_t = std::variant< //
#if USEARCH_TARGET_NVIDIA_AMPERE
    geospatial_vram_nvidia_variant, //
#endif
#if USEARCH_TARGET_INTEL_SKYLAKE
    geospatial_ram_variant<cpu_class_t::intel_skylake_k>,  //
    geospatial_nvme_variant<cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_TARGET_INTEL_HASWELL
    geospatial_ram_variant<cpu_class_t::intel_haswell_k>,  //
    geospatial_nvme_variant<cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_TARGET_APPLE_M2
    geospatial_ram_variant<cpu_class_t::apple_m2_k>,  //
    geospatial_nvme_variant<cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_TARGET_AWS_G2
    geospatial_ram_variant<cpu_class_t::aws_g2_k>,  //
    geospatial_nvme_variant<cpu_class_t::aws_g2_k>, //
#endif
#if USEARCH_TARGET_ALIBABA_C930
    geospatial_ram_variant<cpu_class_t::alibaba_c930_k>,  //
    geospatial_nvme_variant<cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_TARGET_WASM64_EXTENDED
    geospatial_ram_variant<cpu_class_t::wasm64_extended_k>,  //
    geospatial_nvme_variant<cpu_class_t::wasm64_extended_k>, //
#endif
    geospatial_ram_variant<cpu_class_t::serial_k>, //
    geospatial_nvme_variant<cpu_class_t::serial_k>>;

/** The engine's form of a C terrain; a path is refused before it gets here. */
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

/** Whether @p terrain names exactly one source and none a handle cannot serve yet; false once
 *  @p outcome records why not. */
static bool accepts_terrain(usearch_geospatial_terrain_t const& terrain, usearch_outcome_t* outcome) noexcept {
    if (terrain.path.data)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency reads a terrain dataset yet");
    if (terrain.heights && terrain.callback)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A terrain is either borrowed or filled, never both");
    return true;
}

/** Whether @p observations names storage for @p count entries, each with a position and an interval
 *  in order; false once @p outcome records why not. */
static bool accepts_observations(usearch_geospatial_observation_t const* observations, std::size_t count,
                                 usearch_outcome_t* outcome) noexcept {
    if (!observations)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The observations must not be null");
    for (std::size_t row = 0; row != count; ++row) {
        usearch_geospatial_observation_t const& observation = observations[row];
        if (std::isnan(observation.latitude_degrees) || std::isnan(observation.longitude_degrees))
            return propagate_error(usearch_status_invalid_argument_k, outcome,
                                   "An observation needs a latitude and a longitude");
        if (observation.end_seconds < observation.begin_seconds)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "An observation ends before it begins");
    }
    return true;
}

/** Whether @p relation names one of the four, and @c visible only on @p variant with terrain; false
 *  once @p outcome records why not. */
template <typename variant_type_>
static bool accepts_relation(variant_type_ const& variant, usearch_geospatial_relation_t relation,
                             usearch_outcome_t* outcome) noexcept {
    if (relation != usearch_geospatial_relation_near_k && relation != usearch_geospatial_relation_visible_k &&
        relation != usearch_geospatial_relation_sees_k && relation != usearch_geospatial_relation_seen_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown geospatial relation");
    if (relation != usearch_geospatial_relation_visible_k || variant.has_terrain()) return true;
    return propagate_error(usearch_status_unsupported_k, outcome, "Lines of sight need a handle with terrain");
}

/** What the geospatial verbs differ in: observations in time over a terrain, under one kind. */
struct geospatial_traits_t {
    using geometry_t = geospatial_geometry_t;

    /** What a view keeps of an observation it read: the stored form, decoded as it lands. */
    using kept_t = us::geospatial::earth_observation_t;

    template <typename arguments_type_>
    static bool state(arguments_type_ const& arguments, geometry_t& geometry, usearch_outcome_t*) noexcept {
        geometry.connectivity = arguments.connectivity;
        return true;
    }

    /** Reads the connectivity a checkpoint at @p path recorded, the one field beside its
     *  observations. */
    static status_t recall(std::string_view path, geometry_t& geometry) noexcept {
        auto per_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
        };
        auto const [metadata, status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
        return status;
    }

    template <typename arguments_type_>
    static bool admits(geometry_t const&, arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_terrain(arguments.terrain, outcome);
    }

    /** Refuses a terrain for a call placing on a device, whose lines of sight no device tests
     *  yet. */
    template <typename arguments_type_>
        requires requires(arguments_type_ const& arguments) { arguments.vram; }
    static bool admits(geometry_t const&, arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        if (!accepts_terrain(arguments.terrain, outcome)) return false;
        if (terrain_of(arguments.terrain).empty()) return true;
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU geospatial index tests no lines of sight");
    }

    /** Whether a donor of @p donor geometry lands in an index of @p index geometry: the same
     *  connectivity, since every observation is scored alike. */
    static bool matches(geometry_t const& index, geometry_t const& donor) noexcept {
        return index.connectivity == donor.connectivity;
    }

    /** The connectivity @p geometry freezes, a zero taking the cone winners and long links of the
     *  default. */
    static std::size_t connectivity_of(geometry_t const& geometry) noexcept {
        return geometry.connectivity ? geometry.connectivity : us::geospatial::surface_keep_octants_t {}.connectivity;
    }

    /** Takes @p geometry into the CPU @p variant, which adopts the terrain. */
    template <typename variant_type_, typename arguments_type_>
    static bool freeze(variant_type_& variant, geometry_t const& geometry, arguments_type_ const& arguments,
                       usearch_outcome_t* outcome) noexcept {
        variant.geometry.connectivity = connectivity_of(geometry);
        if (status_t const adopted = variant.adopt_terrain(terrain_of(arguments.terrain)); failed(adopted))
            return propagate_error(adopted, outcome, "The terrain could not be adopted");
        return true;
    }

#if USEARCH_ARCH_CUDA_

    /** Takes @p geometry into the GPU @p variant, its connectivity rounded up to the power of two a
     *  GPU row takes, and loads its judging kernel. */
    template <typename arguments_type_>
    static bool freeze(geospatial_vram_nvidia_variant& variant, geometry_t const& geometry, arguments_type_ const&,
                       usearch_outcome_t* outcome) noexcept {
        variant.geometry.connectivity = std::bit_ceil(connectivity_of(geometry));
        if (status_t const loaded = variant.load_admit(); failed(loaded))
            return propagate_error(loaded, outcome, "The GPU kernels could not be loaded");
        return true;
    }
#endif

    static usearch_geospatial_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.connectivity};
    }

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_geospatial_index_description_t& description) noexcept {
        description.metadata = metadata_of(variant.geometry);
        description.memory_usage += variant.terrain_memory_usage();
    }

    static bool accepts(usearch_geospatial_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_observations(arguments.observations, arguments.count, outcome);
    }

    template <typename variant_type_>
    static bool accepts(variant_type_ const& variant, usearch_geospatial_search_t const& arguments,
                        usearch_outcome_t* outcome) noexcept {
        return accepts_observations(arguments.queries, arguments.count, outcome) &&
               accepts_relation(variant, arguments.relation, outcome);
    }

    /** Where a get hands its observations out. */
    template <typename arguments_type_>
    static usearch_geospatial_observation_t const** output_of(arguments_type_ const& arguments) noexcept {
        return arguments.observations;
    }

    static status_t keep(nvme_executor_t&, std::size_t, kept_t const& stored, kept_t& kept) noexcept {
        kept = stored;
        return status_t::success_k;
    }

    /** Lands @p count observations, the @c i th as @p value_at hands it, in @p memory. */
    template <typename variant_type_, typename value_at_type_, typename arguments_type_>
    static bool land(variant_type_ const&, std::size_t count, value_at_type_&& value_at,
                     growing_arena<forkunion_allocator_t>& memory, arguments_type_ const& arguments,
                     usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_observation_t* const observations = carve<usearch_geospatial_observation_t>(memory, count);
        if (count && !observations)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        for (std::size_t entry = 0; entry != count; ++entry)
            observations[entry] = observation_of(us::geospatial::to_geodetic_observation(value_at(entry)));
        *arguments.observations = observations;
        return true;
    }
};

struct usearch_geospatial_index_handle_t : basic_handle<geospatial_variant_t, geospatial_traits_t> {
    using basic_handle::basic_handle;
};

struct usearch_geospatial_cursor_handle_t : basic_cursor<usearch_geospatial_index_handle_t> {};

struct usearch_geospatial_refinement_handle_t : basic_refinement<usearch_geospatial_index_handle_t> {};

#pragma endregion Handles

extern "C" {

USEARCH_EXPORT void usearch_geospatial_init_ram(usearch_geospatial_init_ram_t* arguments) { basic_init_ram(arguments); }
USEARCH_EXPORT void usearch_geospatial_init_vram(usearch_geospatial_init_vram_t* arguments) {
    basic_init_vram(arguments);
}
USEARCH_EXPORT void usearch_geospatial_load_ram(usearch_geospatial_load_ram_t* arguments) {
    basic_load_ram<usearch_geospatial_load_ram>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_load_vram(usearch_geospatial_load_vram_t* arguments) {
    basic_load_vram<usearch_geospatial_load_vram>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_view_ram(usearch_geospatial_view_ram_t* arguments) { basic_view_ram(arguments); }
USEARCH_EXPORT void usearch_geospatial_view_nvme(usearch_geospatial_view_nvme_t* arguments) {
    basic_view_nvme<usearch_geospatial_view_nvme>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_free(usearch_geospatial_index_t index) { basic_free(index); }
USEARCH_EXPORT void usearch_geospatial_reserve(usearch_geospatial_reserve_t* arguments) {
    basic_reserve<usearch_geospatial_reserve>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_clear(usearch_geospatial_clear_t* arguments) {
    basic_clear<usearch_geospatial_clear>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_describe(usearch_geospatial_describe_t* arguments) {
    basic_describe<usearch_geospatial_describe>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_count(usearch_geospatial_count_t* arguments) {
    basic_count<usearch_geospatial_count>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_insert(usearch_geospatial_insert_t* arguments) {
    basic_insert<usearch_geospatial_insert>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refine(usearch_geospatial_refine_t* arguments) {
    basic_refine<usearch_geospatial_refine>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_search(usearch_geospatial_search_t* arguments) {
    basic_search<usearch_geospatial_search>(arguments);
}

USEARCH_EXPORT void usearch_geospatial_within(usearch_geospatial_within_t* arguments) {
    if (submitted<usearch_geospatial_within>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {arguments->queries}, outcome);
    if (!arena) return;
    if (!(arguments->tolerance.metres > 0))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A containment walk needs a positive radius in metres"),
               void();
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!accepts_observations(arguments->queries, count, outcome)) return;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (!accepts_relation(variant, arguments->relation, outcome)) return;
            if (!accepts_traversal(device_of(variant), arguments->traversal, arguments->predicate, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            usearch_matches_tape_view_t matches {};
            std::size_t written = 0;
            if (!crossed(variant.within(arguments->queries, count,
                                        static_cast<us::geospatial::relation_t>(arguments->relation),
                                        arguments->tolerance, arguments->traversal, arguments->predicate, arena->memory,
                                        matches, written, ticks, *executor),
                         arguments->stats, outcome))
                return;
            *arguments->matches = matches;
            if (arguments->written) *arguments->written = written;
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variant);
}

USEARCH_EXPORT void usearch_geospatial_get(usearch_geospatial_get_t* arguments) {
    basic_get<usearch_geospatial_get>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments) {
    basic_cursor_init(arguments);
}
USEARCH_EXPORT void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments) {
    basic_cursor_next<usearch_geospatial_cursor_next>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor) { basic_free(cursor); }
USEARCH_EXPORT void usearch_geospatial_sample(usearch_geospatial_sample_t* arguments) {
    basic_sample<usearch_geospatial_sample>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_absorb(usearch_geospatial_absorb_t* arguments) {
    basic_absorb<usearch_geospatial_absorb>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_split(usearch_geospatial_split_t* arguments) {
    basic_split<usearch_geospatial_split>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refinement_init(usearch_geospatial_refinement_init_t* arguments) {
    basic_refinement_init<usearch_geospatial_refinement_init>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refinement_next(usearch_geospatial_refinement_next_t* arguments) {
    basic_refinement_next<usearch_geospatial_refinement_next>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refinement_finish(usearch_geospatial_refinement_finish_t* arguments) {
    basic_refinement_finish<usearch_geospatial_refinement_finish>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_save_t* arguments) {
    basic_save<usearch_geospatial_save>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments) {
    basic_metadata_read<usearch_geospatial_metadata_read, geospatial_traits_t>(arguments);
}

} // extern "C"
