/**
 *  @file src/basic_punned.cuh
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief What the basic dispatch units and the class units agree on: the verb table one variant
 *      compiles, the class tables each class unit instantiates, and the traits each modality's
 *      verbs differ in, spelled in C argument structures alone.
 *
 *  It includes no tower: a modality's dispatch unit reaches its variant only through its home's
 *  verb table, and @c basic_specializations.cuh with each @c <modality>_specializations.cuh fills
 *  those tables from the towers. What a verb does to one variant, as freezing a geometry into it or
 *  landing its values, is a @c variant_bodies specialization beside the variants.
 */
#pragma once

#include <cmath>   // `std::isnan` - an observation without a position
#include <cstring> // `std::memcpy` - a key read through its stride

#include <array>       // `std::array` - the devices a GPU executor spans
#include <charconv>    // `std::to_chars`, `std::from_chars` - the metadata a checkpoint records as text
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity_t`
#include <variant>     // `std::monostate` - a verb a modality takes no input for

#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_index_metadata` - a checkpoint's fields

#include "lib.hpp"

#pragma region Crossings

/** Key @p row of the caller's keys, through their stride; zero means packed. */
inline usearch_key_t key_at(usearch_keys_view_t keys, std::size_t row) noexcept {
    std::size_t const stride = keys.stride ? keys.stride : sizeof(usearch_key_t);
    usearch_key_t key;
    std::memcpy(&key, reinterpret_cast<byte_t const*>(keys.data) + row * stride, sizeof(key));
    return key;
}

/** Entry @p row of a byte tape, as the view every string type converts from. */
inline std::string_view chars_at(usearch_tape_view_t tape, std::size_t row) noexcept {
    return {static_cast<char const*>(tape.data) + tape.offsets[row], tape.offsets[row + 1] - tape.offsets[row]};
}

/** The @p count entries of the caller's byte tape @p tape, as the engines read tapes. */
inline tape_view<byte_t const> bytes_tape_of(usearch_tape_view_t tape, std::size_t count) noexcept {
    return {static_cast<byte_t const*>(tape.data), {tape.offsets, count + 1}};
}

/** Whether a tape's @p count + 1 offsets never fall; they may start anywhere, as an Arrow slice's
 *  do. */
inline bool monotone_offsets(usearch_tape_view_t tape, std::size_t count) noexcept {
    if (!tape.offsets) return false;
    for (std::size_t row = 0; row != count; ++row)
        if (tape.offsets[row + 1] < tape.offsets[row]) return false;
    return true;
}

/** Keeps the first failure any thread reports into @p first. */
inline void keep_first_failure(status_t& first, status_t status) noexcept {
    if (succeeded(status)) return;
    status_t unset = status_t::success_k;
    standard_atomic_ref<status_t>(first).compare_exchange_strong(unset, status, std::memory_order_relaxed,
                                                                 std::memory_order_relaxed);
}

/** Raises @p word to @p value, if it is larger, from any thread; a relaxed peek first, so a word
 *  already that large is never written. */
template <typename word_type_>
void raise_to(word_type_& word, std::type_identity_t<word_type_> value) noexcept {
    fu::standard_atomic_ref<word_type_> const raised {word};
    if (raised.load(std::memory_order_relaxed) < value) raised.fetch_max(value, std::memory_order_relaxed);
}

/** The decimal spelling of @p value in @p text, which a checkpoint's metadata records. */
inline std::string_view decimal_of(std::size_t value, std::span<char> text) noexcept {
    std::to_chars_result const printed = std::to_chars(text.data(), text.data() + text.size(), value);
    return {text.data(), static_cast<std::size_t>(printed.ptr - text.data())};
}

/** The number a checkpoint's metadata spells in @p text, or zero when it spells none. */
inline std::size_t decimal_in(std::string_view text) noexcept {
    std::size_t value = 0;
    std::from_chars(text.data(), text.data() + text.size(), value);
    return value;
}

#pragma endregion Crossings

#pragma region Home Tables

/** What a refinement reports, as the C outputs take it: the walks' telemetry, the nodes it visited
 *  and changed, and the closing audit's reach. */
struct basic_refined_t {
    usearch_traversal_stats_t stats {};
    std::size_t visited = 0;
    std::size_t changed = 0;
    usearch_reachability_t reachability {};
};

/** What a modality's verbs do to one variant beyond its index: freezing a geometry into it,
 *  describing it, and landing, keeping and tallying its values; each modality's specializations
 *  define it beside their variants. */
template <typename traits_type_>
struct variant_bodies;

/**
 *  @brief The verbs one basic variant compiles in its class unit, as function pointers over the
 *      variant passed as @c void*, which only these entries cast back to its type.
 *
 *  Every home fills every entry, answering @c read_only_k or @c unsupported_k where it serves none,
 *  so a modality's dispatch unit visits three homes and never a variant. A modality's verbs differ
 *  in the C argument structures @p traits_type_ names, and in the geometry a variant @c serves.
 */
template <typename traits_type_, dense_home_t home_>
struct basic_verbs {
    using traits_t = traits_type_;
    using executor_t = typename dense_home_executor<home_>::type;
    using geometry_t = typename traits_type_::geometry_t;
    using walked_t = expected<usearch_traversal_stats_t>;

    bool (*serves)(geometry_t const& geometry) noexcept;

    /** Builds an unplaced variant: a host one over the heap's domain, a device one for the device's
     *  vendor ordinal @p device and the @p geometry it scores under. */
    void* (*create)(int device, geometry_t const& geometry) noexcept;
    void (*destroy)(void* variant) noexcept;
    bool (*freeze)(void* variant, geometry_t& geometry, typename traits_type_::freeze_t const& input,
                   usearch_outcome_t* outcome) noexcept;
    void (*bound_shards)(void* variant, std::size_t rows) noexcept;

    /** Appends one shard on a host memory domain or a device's vendor ordinal. */
    status_t (*emplace)(void* variant, int placement) noexcept;

    /** Serves a checkpoint in place, an NVMe view through @p executor, which a host variant does
     *  without. */
    status_t (*view)(void* variant, std::string_view path, std::span<std::string_view const> paths,
                     executor_t* executor) noexcept;
    status_t (*load)(void* variant, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t (*save)(void* variant, std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** The device the variant computes on, where a call's arena must live, or @c host_device_k. */
    int (*device)(void const* variant) noexcept;

    /** The distinct devices the shards sit on into @p listed, as many as fit; answers how many. */
    std::size_t (*devices)(void const* variant, std::span<int> listed) noexcept;
    status_t (*reserve)(void* variant, typename traits_type_::reserve_t const& arguments) noexcept;
    status_t (*clear)(void* variant) noexcept;
    void (*describe)(void const* variant, typename traits_type_::description_t& description) noexcept;
    void (*count)(void* variant, usearch_key_t key, typename traits_type_::key_stats_t& stats,
                  usearch_executor_t executor, usearch_outcome_t* outcome) noexcept;
    void (*get)(void* variant, typename traits_type_::get_t const& arguments,
                growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept;
    walked_t (*insert)(void* variant, typename traits_type_::insert_t const& arguments, progress_ticks_t& ticks,
                       executor_t& executor) noexcept;
    bool (*accepts_search)(void const* variant, typename traits_type_::search_t const& arguments,
                           usearch_outcome_t* outcome) noexcept;
    walked_t (*search)(void* variant, typename traits_type_::search_t const& arguments,
                       growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing, progress_ticks_t& ticks,
                       executor_t& executor) noexcept;
    bool (*accepts_within)(void const* variant, typename traits_type_::within_t const& arguments,
                           usearch_outcome_t* outcome) noexcept;
    walked_t (*within)(void* variant, typename traits_type_::within_t const& arguments,
                       growing_arena<forkunion_allocator_t>& memory, usearch_matches_tape_view_t& matches,
                       std::size_t& written, progress_ticks_t& ticks, executor_t& executor) noexcept;
    status_t (*sample)(void* variant, typename traits_type_::sample_t const& arguments) noexcept;

    /** Whether the variant at @p donor shares the geometry of the one at @p variant. */
    bool (*matches)(void const* variant, void const* donor) noexcept;
    expected<void*> (*refinement_open)(void* variant, std::span<void* const> donors, std::size_t expansion) noexcept;
    std::size_t (*refinement_population)(void const* refinement) noexcept;
    status_t (*refinement_run)(void* variant, void* refinement, progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Whether the refinement consumed its donor at @p donor, which its owner may then free. */
    bool (*refinement_landed)(void const* refinement, std::size_t donor) noexcept;
    void (*absorbed)(void* variant, void const* donor) noexcept;
    basic_refined_t (*refinement_running)(void* refinement) noexcept;
    expected<basic_refined_t> (*refinement_finish)(void* variant, void* refinement, executor_t& executor) noexcept;
    void (*refinement_free)(void* refinement) noexcept;

    void* (*cursor_open)(void* variant, usearch_outcome_t* outcome) noexcept;
    void (*cursor_next)(void* variant, void* cursor, typename traits_type_::cursor_next_t const& arguments,
                        growing_arena<forkunion_allocator_t>& memory, usearch_outcome_t* outcome) noexcept;
    void (*cursor_free)(void* cursor) noexcept;
};

/** The variants one modality's class unit compiles for one home: @c find answers the verb table of
 *  the first variant serving the geometry, or null where none of the class's does. */
template <typename traits_type_, dense_home_t home_, auto class_>
struct basic_class_tables {
    static basic_verbs<traits_type_, home_> const* find(typename traits_type_::geometry_t const& geometry) noexcept;
};

/** One picked variant as a modality's dispatch unit holds it for one call: its home's verbs, and
 *  the variant they serve. */
template <typename traits_type_, dense_home_t home_>
struct punned_basic_index {
    using executor_t = typename basic_verbs<traits_type_, home_>::executor_t;

    basic_verbs<traits_type_, home_> const& verbs;
    void* variant;
};

/** The executor a null handle runs a device variant's work on: a stream of its own on every device
 *  its shards sit on, never a legacy default stream. */
template <typename traits_type_, dense_home_t home_>
    requires is_gpu_executor<typename punned_basic_index<traits_type_, home_>::executor_t>
usearch_executor_t call_executor_for(punned_basic_index<traits_type_, home_> const& punned,
                                     usearch_executor_handle_t& call_executor, usearch_outcome_t* outcome) noexcept {
    std::array<int, gpu_devices_capacity_k> devices;
    std::size_t devices_count = punned.verbs.devices(punned.variant, devices);
    if (!devices_count) devices[devices_count++] = punned.verbs.device(punned.variant);
    auto [built, status] = punned_basic_index<traits_type_, home_>::executor_t::spawn(
        std::span<int const> {devices.data(), devices_count});
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor = gpu_executor_t {std::move(built)};
    return &call_executor;
}

/** The device a basic GPU variant places on: USearch ordinal @p ordinal of the listing, on a vendor
 *  whose basic kernels this build carries; null once @p outcome records why not. */
inline gpu_device_t const* basic_device_at(std::size_t ordinal, usearch_outcome_t* outcome) noexcept {
    std::span<gpu_device_t const> const listed = gpu_devices();
    if (ordinal >= listed.size())
        return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), nullptr;
    if (listed[ordinal].vendor != gpu_vendor_t::nvidia_k)
        return propagate_error(usearch_status_device_k, outcome, "No basic GPU kernels serve this vendor yet"), nullptr;
    return &listed[ordinal];
}

#pragma endregion Home Tables

#pragma region Sequence Traits

/** The checkpoint token of one sequence kind, spelled as the SafeGraphs registry spells it; empty
 *  for the kinds the engine refuses. */
inline std::string_view token_of(usearch_sequence_metric_t metric) noexcept {
    switch (metric) {
    case usearch_sequence_metric_levenshtein_bytes_k: return "levenshtein";
    case usearch_sequence_metric_levenshtein_utf8_k: return "levenshtein-utf8";
    case usearch_sequence_metric_overlap_k: return "overlap";
    case usearch_sequence_metric_needleman_wunsch_k: return "needleman-wunsch";
    default: return {};
    }
}

/** The sequence kind a checkpoint token names, or @c unknown_k. */
inline usearch_sequence_metric_t sequence_metric_of(std::string_view token) noexcept {
    for (usearch_sequence_metric_t metric :
         {usearch_sequence_metric_levenshtein_bytes_k, usearch_sequence_metric_levenshtein_utf8_k,
          usearch_sequence_metric_overlap_k, usearch_sequence_metric_needleman_wunsch_k})
        if (token == token_of(metric)) return metric;
    return usearch_sequence_metric_unknown_k;
}

/** The kind, window widths, recorded costs and connectivity a sequence handle freezes, as an init
 *  or a checkpoint states them; every variant holds one. */
struct sequence_geometry_t {
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;

    /** The window widths the overlap kind hashes, in bytes; empty for every other kind. */
    growing_buffer<std::size_t, forkunion_allocator_t> window_widths {forkunion_allocator_t {no_memory_domain_k}};

    /** The folded substitution costs a checkpoint recorded, one hexadecimal digit per entry row by
     *  row, which a freeze unfolds; empty for a fresh index, whose weights a freeze folds. */
    growing_buffer<char, forkunion_allocator_t> substitution_costs {forkunion_allocator_t {no_memory_domain_k}};

    /** The gap cost a checkpoint recorded beside its substitution costs. */
    std::size_t gap_cost = 0;

    /** Neighbors per node the insertions select; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

/** Whether @p sequences holds @p count byte strings; false once @p outcome records why not. */
inline bool accepts_sequences(usearch_tape_view_t const& sequences, std::size_t count,
                              usearch_outcome_t* outcome) noexcept {
    if (!sequences.data || !monotone_offsets(sequences, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The sequences need data and non-decreasing offsets");
    if (sequences.scalar != usearch_scalar_u8_k)
        return propagate_error(usearch_status_unsupported_k, outcome, "Sequences arrive as u8 bytes");
    return true;
}

/** The sequence variants one class unit compiles in each home, which
 *  @c sequence_specializations.cuh lists. */
template <dense_home_t home_, auto class_>
struct sequence_variants_of {
    using type = type_list<>;
};

/** What the sequence verbs differ in: Levenshtein, overlap and Needleman-Wunsch kinds over u8
 *  strings. */
struct sequence_traits_t {
    using geometry_t = sequence_geometry_t;
    using description_t = usearch_sequence_index_description_t;
    using reserve_t = usearch_sequence_reserve_t;
    using insert_t = usearch_sequence_insert_t;
    using search_t = usearch_sequence_search_t;
    using get_t = usearch_sequence_get_t;
    using key_stats_t = usearch_sequence_key_stats_t;
    using cursor_next_t = usearch_sequence_cursor_next_t;
    using sample_t = usearch_sequence_sample_t;

    /** No radius walk serves strings. */
    using within_t = std::monostate;

    /** What a freeze reads past the geometry: the weights an alignment kind folds, or null. */
    using freeze_t = usearch_sequence_weights_t const*;

    /** The variants each class unit compiles per home. */
    template <dense_home_t home_, auto class_>
    using variants = typename sequence_variants_of<home_, class_>::type;

    /** The GPU classes with a unit of sequence kits, newest first. */
    using gpu_ladder_t = gpu_ladder<gpu_class_t::nvidia_ampere_k>;

    /** Takes the kind, widths and connectivity an init states, keeping the widths; false once
     *  @p outcome records why not. */
    template <typename arguments_type_>
    static bool state(arguments_type_ const& arguments, geometry_t& geometry, usearch_outcome_t* outcome) noexcept {
        if (arguments.window_widths_count && !arguments.window_widths)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "The window widths must not be null");
        geometry.metric = arguments.metric;
        geometry.connectivity = arguments.connectivity;
        if (failed(geometry.window_widths.reserve_exactly(arguments.window_widths_count)))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to keep the widths");
        for (std::size_t window = 0; window != arguments.window_widths_count; ++window)
            geometry.window_widths.push_back_reserved(arguments.window_widths[window]);
        return true;
    }

    /** Reads the kind, widths, costs and connectivity a checkpoint at @p path recorded, the widths
     *  spelled as decimals between commas, the costs kept as recorded for a freeze to unfold. */
    static status_t recall(std::string_view path, geometry_t& geometry) noexcept {
        status_t fields_status = status_t::success_k;
        auto per_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
            else if (field.key == "gap_cost") geometry.gap_cost = decimal_in(field.value);
            else if (field.key == "substitution_costs") {
                fields_status = geometry.substitution_costs.reserve_exactly(field.value.size());
                for (char const digit : field.value)
                    if (succeeded(fields_status)) geometry.substitution_costs.push_back_reserved(digit);
            }
            else if (field.key == "window_widths") {
                std::size_t windows = 1;
                for (char const character : field.value) windows += character == ',';
                fields_status = geometry.window_widths.reserve_exactly(windows);
                for (std::string_view rest = field.value; succeeded(fields_status) && !rest.empty();) {
                    std::size_t const comma = rest.find(',');
                    geometry.window_widths.push_back_reserved(decimal_in(rest.substr(0, comma)));
                    rest = comma == std::string_view::npos ? std::string_view {} : rest.substr(comma + 1);
                }
            }
        };
        auto const [metadata, status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
        geometry.metric = sequence_metric_of(metadata.metric.view());
        return failed(status) ? status : fields_status;
    }

    /** Refuses the kinds no CPU or GPU kit serves and the parameters a kind does not take. */
    template <typename arguments_type_>
    static bool admits(geometry_t const& geometry, arguments_type_ const& arguments,
                       usearch_outcome_t* outcome) noexcept {
        switch (geometry.metric) {
        case usearch_sequence_metric_levenshtein_bytes_k:
        case usearch_sequence_metric_levenshtein_utf8_k:
        case usearch_sequence_metric_overlap_k:
        case usearch_sequence_metric_needleman_wunsch_k: break;
        default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown sequence metric");
        }
        bool const aligned = geometry.metric == usearch_sequence_metric_needleman_wunsch_k;
        if (arguments.weights && !aligned)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "Only an alignment kind takes weights");
        if (arguments.weights && !arguments.weights->substitution_matrix)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "The substitution matrix is null");
        // A checkpoint recalls the costs it was built with, which weights given at load replace.
        if (aligned && !arguments.weights && geometry.substitution_costs.empty())
            return propagate_error(usearch_status_config_mismatch_k, outcome, "An alignment kind needs weights");
        bool const overlap = geometry.metric == usearch_sequence_metric_overlap_k;
        if (overlap == geometry.window_widths.empty())
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "The overlap kind needs window widths, and only it takes any");
        for (std::size_t const width : geometry.window_widths)
            if (!width) return propagate_error(usearch_status_invalid_argument_k, outcome, "A window is empty");
        return true;
    }

    /** The weights an init, a load or a view names, which a freeze folds. */
    template <typename arguments_type_>
    static freeze_t freeze_input(arguments_type_ const& arguments) noexcept {
        return arguments.weights;
    }

    static usearch_sequence_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.metric, geometry.connectivity};
    }

    static bool accepts(usearch_sequence_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_sequences(arguments.sequences, arguments.count, outcome);
    }

    /** Where a get hands its strings out. */
    template <typename arguments_type_>
    static usearch_tape_view_t* output_of(arguments_type_ const& arguments) noexcept {
        return arguments.sequences;
    }
};

#pragma endregion Sequence Traits

#pragma region Shapes Traits

/** The checkpoint token of one shapes kind, spelled as NumKong spells its kernels. */
inline std::string_view token_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_chamfer_k: return "chamfer";
    case usearch_shapes_metric_hausdorff_k: return "hausdorff";
    case usearch_shapes_metric_gromov_wasserstein_k: return "gromov_wasserstein";
    case usearch_shapes_metric_wasserstein_procrustes_k: return "wasserstein_procrustes";
    default: return {};
    }
}

/** The shapes kind a checkpoint token names, or @c unknown_k. */
inline usearch_shapes_metric_t shapes_metric_of(std::string_view token) noexcept {
    for (usearch_shapes_metric_t metric :
         {usearch_shapes_metric_chamfer_k, usearch_shapes_metric_hausdorff_k,
          usearch_shapes_metric_gromov_wasserstein_k, usearch_shapes_metric_wasserstein_procrustes_k})
        if (token == token_of(metric)) return metric;
    return usearch_shapes_metric_unknown_k;
}

/** The kind, scalar, point count and connectivity a shapes handle freezes, as an init or a
 *  checkpoint states them; every variant holds one. */
struct shapes_geometry_t {
    usearch_shapes_metric_t metric = usearch_shapes_metric_unknown_k;

    /** The scalar every point is stored in. */
    usearch_scalar_t quantization = usearch_scalar_unknown_k;

    /** The frozen point count, or zero for any count. */
    std::size_t points = 0;

    /** Neighbors per node the insertions select; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

/** Whether @p shapes holds @p count entries of points in a scalar the variants cast from, its
 *  offsets counting points; false once @p outcome records why not. */
inline bool accepts_shapes(usearch_tape_view_t const& shapes, std::size_t count, usearch_outcome_t* outcome) noexcept {
    if (!shapes.data || !monotone_offsets(shapes, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The shapes need data and non-decreasing offsets");
    nk_dtype_t const dtype = nk_dtype_of(shapes.scalar);
    if (dtype == nk_dtype_unknown_k || nk_dtype_bits(dtype) % 8)
        return propagate_error(usearch_status_unsupported_k, outcome, "Points arrive in a whole-byte scalar");
    return true;
}

/** The shapes variants one class unit compiles in each home, which @c shapes_specializations.cuh
 *  lists. */
template <dense_home_t home_, auto class_>
struct shapes_variants_of {
    using type = type_list<>;
};

/** What the shapes verbs differ in: clouds of 3D points stored in one scalar, crossing in any. */
struct shapes_traits_t {
    using geometry_t = shapes_geometry_t;
    using description_t = usearch_shapes_index_description_t;
    using reserve_t = usearch_shapes_reserve_t;
    using insert_t = usearch_shapes_insert_t;
    using search_t = usearch_shapes_search_t;
    using get_t = usearch_shapes_get_t;
    using key_stats_t = usearch_shapes_key_stats_t;
    using cursor_next_t = usearch_shapes_cursor_next_t;
    using sample_t = usearch_shapes_sample_t;

    /** No radius walk serves clouds. */
    using within_t = std::monostate;

    /** A freeze reads nothing past the geometry. */
    using freeze_t = std::monostate;

    /** The variants each class unit compiles per home. */
    template <dense_home_t home_, auto class_>
    using variants = typename shapes_variants_of<home_, class_>::type;

    /** The GPU classes with a unit of shapes kits, newest first: datacenter Blackwell's packed f16
     *  kit, then the Ampere floor. */
    using gpu_ladder_t = gpu_ladder<gpu_class_t::nvidia_blackwell_k, gpu_class_t::nvidia_ampere_k>;

    template <typename arguments_type_>
    static bool state(arguments_type_ const& arguments, geometry_t& geometry, usearch_outcome_t*) noexcept {
        geometry = {arguments.metric, arguments.quantization, arguments.points, arguments.connectivity};
        return true;
    }

    /** Reads the kind, scalar, point count and connectivity a checkpoint at @p path recorded. */
    static status_t recall(std::string_view path, geometry_t& geometry) noexcept {
        auto per_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
            else if (field.key == "points") geometry.points = decimal_in(field.value);
        };
        auto const [metadata, status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
        geometry.metric = shapes_metric_of(metadata.metric.view());
        geometry.quantization = metadata.value_dtype == dtype_t::f16_k   ? usearch_scalar_f16_k
                                : metadata.value_dtype == dtype_t::f32_k ? usearch_scalar_f32_k
                                                                         : usearch_scalar_unknown_k;
        return status;
    }

    /** Refuses the kinds and scalars no CPU variant serves. */
    template <typename arguments_type_>
    static bool admits(geometry_t const& geometry, arguments_type_ const&, usearch_outcome_t* outcome) noexcept {
        if (token_of(geometry.metric).empty())
            return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric");
        if (geometry.quantization != usearch_scalar_f16_k && geometry.quantization != usearch_scalar_f32_k)
            return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points");
        return true;
    }

    /** Refuses the kinds and scalars no device kit serves, for a call placing on a device. */
    template <typename arguments_type_>
        requires requires(arguments_type_ const& arguments) { arguments.vram; }
    static bool admits(geometry_t const& geometry, arguments_type_ const&, usearch_outcome_t* outcome) noexcept {
        switch (geometry.metric) {
        case usearch_shapes_metric_chamfer_k:
        case usearch_shapes_metric_hausdorff_k: break;
        case usearch_shapes_metric_gromov_wasserstein_k:
        case usearch_shapes_metric_wasserstein_procrustes_k:
            return propagate_error(usearch_status_unsupported_k, outcome, "Only Chamfer and Hausdorff serve a GPU yet");
        default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric");
        }
        if (geometry.quantization != usearch_scalar_f16_k && geometry.quantization != usearch_scalar_f32_k)
            return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points");
        return true;
    }

    /** Nothing an init, a load or a view names past the geometry reaches a freeze. */
    template <typename arguments_type_>
    static freeze_t freeze_input(arguments_type_ const&) noexcept {
        return {};
    }

    static usearch_shapes_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.metric, geometry.quantization, geometry.points, geometry.connectivity};
    }

    static bool accepts(usearch_shapes_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_shapes(arguments.shapes, arguments.count, outcome);
    }

    /** Whether the points a get asks for export to a whole-byte scalar; false once
     *  @p outcome records why not. */
    template <typename arguments_type_>
    static bool exports(arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        nk_dtype_t const exported = nk_dtype_of(arguments.outputs_scalar);
        if (!arguments.shapes || (exported != nk_dtype_unknown_k && nk_dtype_bits(exported) % 8 == 0)) return true;
        return propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar");
    }

    /** Where a get hands its clouds out. */
    template <typename arguments_type_>
    static usearch_tape_view_t* output_of(arguments_type_ const& arguments) noexcept {
        return arguments.shapes;
    }
};

#pragma endregion Shapes Traits

#pragma region Geospatial Traits

/** The connectivity a geospatial handle freezes, as an init or a checkpoint states it; every
 *  variant holds one. */
struct geospatial_geometry_t {

    /** Neighbors per node: a winner per cone and long links for the rest, rounded up to a power of
     *  two on a GPU; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

/** The checkpoint token of the one geospatial kind, the squared chord. */
inline std::string_view token_of(geospatial_geometry_t const&) noexcept { return "chord"; }

/** Whether @p terrain names exactly one source and none a handle cannot serve yet; false once
 *  @p outcome records why not. */
inline bool accepts_terrain(usearch_geospatial_terrain_t const& terrain, usearch_outcome_t* outcome) noexcept {
    if (terrain.path.data)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency reads a terrain dataset yet");
    if (terrain.heights && terrain.callback)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A terrain is either borrowed or filled, never both");
    return true;
}

/** Whether @p observations names storage for @p count entries, each with a position and an interval
 *  in order; false once @p outcome records why not. */
inline bool accepts_observations(usearch_geospatial_observation_t const* observations, std::size_t count,
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

/** The geospatial variants one class unit compiles in each home, which
 *  @c geospatial_specializations.cuh lists. */
template <dense_home_t home_, auto class_>
struct geospatial_variants_of {
    using type = type_list<>;
};

/** What the geospatial verbs differ in: observations in time over a terrain, under one kind. */
struct geospatial_traits_t {
    using geometry_t = geospatial_geometry_t;
    using description_t = usearch_geospatial_index_description_t;
    using reserve_t = usearch_geospatial_reserve_t;
    using insert_t = usearch_geospatial_insert_t;
    using search_t = usearch_geospatial_search_t;
    using get_t = usearch_geospatial_get_t;
    using key_stats_t = usearch_geospatial_key_stats_t;
    using cursor_next_t = usearch_geospatial_cursor_next_t;
    using sample_t = usearch_geospatial_sample_t;
    using within_t = usearch_geospatial_within_t;

    /** What a freeze reads past the geometry: the terrain the variant adopts. */
    using freeze_t = usearch_geospatial_terrain_t;

    /** The variants each class unit compiles per home. */
    template <dense_home_t home_, auto class_>
    using variants = typename geospatial_variants_of<home_, class_>::type;

    /** The GPU classes with a unit of the geospatial kit, newest first. */
    using gpu_ladder_t = gpu_ladder<gpu_class_t::nvidia_ampere_k>;

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

    /** Refuses a terrain a device cannot read: one filled by a host callback. */
    template <typename arguments_type_>
        requires requires(arguments_type_ const& arguments) { arguments.vram; }
    static bool admits(geometry_t const&, arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        if (!accepts_terrain(arguments.terrain, outcome)) return false;
        if (!arguments.terrain.callback) return true;
        return propagate_error(usearch_status_unsupported_k, outcome,
                               "A GPU terrain needs its heights, not a callback");
    }

    /** The terrain an init, a load or a view names, which a freeze adopts. */
    template <typename arguments_type_>
    static freeze_t freeze_input(arguments_type_ const& arguments) noexcept {
        return arguments.terrain;
    }

    static usearch_geospatial_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.connectivity};
    }

    static bool accepts(usearch_geospatial_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_observations(arguments.observations, arguments.count, outcome);
    }

    /** Whether a containment walk names a positive radius and, when it has queries, observations
     *  each with a position and an interval in order. */
    static bool accepts(usearch_geospatial_within_t const& arguments, usearch_outcome_t* outcome) noexcept {
        if (!(arguments.tolerance.metres > 0))
            return propagate_error(usearch_status_invalid_argument_k, outcome,
                                   "A containment walk needs a positive radius in metres");
        return !arguments.count || accepts_observations(arguments.queries, arguments.count, outcome);
    }

    /** Where a get hands its observations out. */
    template <typename arguments_type_>
    static usearch_geospatial_observation_t const** output_of(arguments_type_ const& arguments) noexcept {
        return arguments.observations;
    }
};

#pragma endregion Geospatial Traits
