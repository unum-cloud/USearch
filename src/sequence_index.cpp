/**
 *  @file src/sequence_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence modality's C surface: the variant over every compiled sequence
 *      alternative, the traits its verbs differ in, and every exported entry point as one forward
 *      apiece.
 *
 *  Variants are ordered newest class first, the serial floor last. Their bodies compile in the
 *  basic class units, never here: the @c extern declarations below keep them out.
 */
#include <cstring> // `std::memcpy` - strings landing in an arena

#include <string_view> // `std::string_view` - checkpoint paths and fields
#include <variant>     // `std::variant`

#include "basic_index.cuh"
#include "sequence_specializations.cuh"

#if USEARCH_TARGET_INTEL_SKYLAKE
extern template struct sequence_ram_variant<cpu_class_t::intel_skylake_k>;
extern template struct sequence_nvme_variant<cpu_class_t::intel_skylake_k>;
#endif // USEARCH_TARGET_INTEL_SKYLAKE
#if USEARCH_TARGET_INTEL_HASWELL
extern template struct sequence_ram_variant<cpu_class_t::intel_haswell_k>;
extern template struct sequence_nvme_variant<cpu_class_t::intel_haswell_k>;
#endif // USEARCH_TARGET_INTEL_HASWELL
#if USEARCH_TARGET_APPLE_M2
extern template struct sequence_ram_variant<cpu_class_t::apple_m2_k>;
extern template struct sequence_nvme_variant<cpu_class_t::apple_m2_k>;
#endif // USEARCH_TARGET_APPLE_M2
#if USEARCH_TARGET_AWS_G2
extern template struct sequence_ram_variant<cpu_class_t::aws_g2_k>;
extern template struct sequence_nvme_variant<cpu_class_t::aws_g2_k>;
#endif // USEARCH_TARGET_AWS_G2
#if USEARCH_TARGET_ALIBABA_C930
extern template struct sequence_ram_variant<cpu_class_t::alibaba_c930_k>;
extern template struct sequence_nvme_variant<cpu_class_t::alibaba_c930_k>;
#endif // USEARCH_TARGET_ALIBABA_C930
#if USEARCH_TARGET_WASM64_EXTENDED
extern template struct sequence_ram_variant<cpu_class_t::wasm64_extended_k>;
extern template struct sequence_nvme_variant<cpu_class_t::wasm64_extended_k>;
#endif // USEARCH_TARGET_WASM64_EXTENDED
extern template struct sequence_ram_variant<cpu_class_t::serial_k>;
extern template struct sequence_nvme_variant<cpu_class_t::serial_k>;

#pragma region Handles

/** Every sequence variant this build compiled: the basic index over owned strings scored with one
 *  class's kits, a read-only view of a checkpoint on storage scored with the same, or the basic
 *  index on a GPU. */
using sequence_variant_t = std::variant< //
#if USEARCH_ARCH_CUDA_
    sequence_vram_nvidia_variant<usearch_sequence_metric_levenshtein_bytes_k>, //
    sequence_vram_nvidia_variant<usearch_sequence_metric_levenshtein_utf8_k>,  //
    sequence_vram_nvidia_variant<usearch_sequence_metric_overlap_k>,           //
#endif
#if USEARCH_TARGET_INTEL_SKYLAKE
    sequence_ram_variant<cpu_class_t::intel_skylake_k>,  //
    sequence_nvme_variant<cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_TARGET_INTEL_HASWELL
    sequence_ram_variant<cpu_class_t::intel_haswell_k>,  //
    sequence_nvme_variant<cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_TARGET_APPLE_M2
    sequence_ram_variant<cpu_class_t::apple_m2_k>,  //
    sequence_nvme_variant<cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_TARGET_AWS_G2
    sequence_ram_variant<cpu_class_t::aws_g2_k>,  //
    sequence_nvme_variant<cpu_class_t::aws_g2_k>, //
#endif
#if USEARCH_TARGET_ALIBABA_C930
    sequence_ram_variant<cpu_class_t::alibaba_c930_k>,  //
    sequence_nvme_variant<cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_TARGET_WASM64_EXTENDED
    sequence_ram_variant<cpu_class_t::wasm64_extended_k>,  //
    sequence_nvme_variant<cpu_class_t::wasm64_extended_k>, //
#endif
    sequence_nvme_variant<cpu_class_t::serial_k>, //
    sequence_ram_variant<cpu_class_t::serial_k>>;

/** Whether @p sequences holds @p count byte strings; false once @p outcome records why not. */
static bool accepts_sequences(usearch_tape_view_t const& sequences, std::size_t count,
                              usearch_outcome_t* outcome) noexcept {
    if (!sequences.data || !monotone_offsets(sequences, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The sequences need data and non-decreasing offsets");
    if (sequences.scalar != usearch_scalar_u8_k)
        return propagate_error(usearch_status_unsupported_k, outcome, "Sequences arrive as u8 bytes");
    return true;
}

/** What the sequence verbs differ in: Levenshtein and overlap kinds over u8 strings. */
struct sequence_traits_t {
    using geometry_t = sequence_geometry_t;

    /** What a view keeps of a string it read, copied into its thread's arena. */
    using kept_t = bytes_view_t;

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

    /** Reads the kind, widths and connectivity a checkpoint at @p path recorded, the widths spelled
     *  as decimals between commas. */
    static status_t recall(std::string_view path, geometry_t& geometry) noexcept {
        status_t widths_status = status_t::success_k;
        auto per_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
            else if (field.key == "window_widths") {
                std::size_t windows = 1;
                for (char const character : field.value) windows += character == ',';
                widths_status = geometry.window_widths.reserve_exactly(windows);
                for (std::string_view rest = field.value; succeeded(widths_status) && !rest.empty();) {
                    std::size_t const comma = rest.find(',');
                    geometry.window_widths.push_back_reserved(decimal_in(rest.substr(0, comma)));
                    rest = comma == std::string_view::npos ? std::string_view {} : rest.substr(comma + 1);
                }
            }
        };
        auto const [metadata, status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
        geometry.metric = sequence_metric_of(metadata.metric.view());
        return failed(status) ? status : widths_status;
    }

    /** Refuses the kinds no CPU or GPU kit serves and the parameters a kind does not take. */
    template <typename arguments_type_>
    static bool admits(geometry_t const& geometry, arguments_type_ const& arguments,
                       usearch_outcome_t* outcome) noexcept {
        switch (geometry.metric) {
        case usearch_sequence_metric_levenshtein_bytes_k:
        case usearch_sequence_metric_levenshtein_utf8_k:
        case usearch_sequence_metric_overlap_k: break;
        case usearch_sequence_metric_needleman_wunsch_k:
        case usearch_sequence_metric_needleman_wunsch_gotoh_k:
        case usearch_sequence_metric_smith_waterman_k:
        case usearch_sequence_metric_smith_waterman_gotoh_k:
            return propagate_error(usearch_status_unsupported_k, outcome, "No alignment kernels serve this kind yet");
        default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown sequence metric");
        }
        if (arguments.weights)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "Only an alignment kind takes weights");
        bool const overlap = geometry.metric == usearch_sequence_metric_overlap_k;
        if (overlap == geometry.window_widths.empty())
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "The overlap kind needs window widths, and only it takes any");
        for (std::size_t const width : geometry.window_widths)
            if (!width) return propagate_error(usearch_status_invalid_argument_k, outcome, "A window is empty");
        return true;
    }

    /** Whether a donor of @p donor geometry lands in an index of @p index geometry: the same kind,
     *  window widths and connectivity. */
    static bool matches(geometry_t const& index, geometry_t const& donor) noexcept {
        if (index.metric != donor.metric || index.connectivity != donor.connectivity ||
            index.window_widths.size() != donor.window_widths.size())
            return false;
        for (std::size_t window = 0; window != index.window_widths.size(); ++window)
            if (index.window_widths[window] != donor.window_widths[window]) return false;
        return true;
    }

    /** Moves @p geometry into @p variant, a zero connectivity taking the default. */
    template <typename variant_type_, typename arguments_type_>
    static bool freeze(variant_type_& variant, geometry_t& geometry, arguments_type_ const&,
                       usearch_outcome_t*) noexcept {
        if (!geometry.connectivity) geometry.connectivity = connectivity_32_k;
        variant.geometry = std::move(geometry);
        return true;
    }

#if USEARCH_ARCH_CUDA_

    /** Moves @p geometry into the GPU @p variant and hands its index the window widths, which its
     *  first reservation uploads. */
    template <typename variant_type_>
    static bool freeze(variant_type_& variant, geometry_t& geometry, usearch_sequence_init_vram_t const&,
                       usearch_outcome_t* outcome) noexcept {
        if (!geometry.connectivity) geometry.connectivity = connectivity_32_k;
        variant.geometry = std::move(geometry);
        status_t const kept = variant.index.window_widths(to_span(variant.geometry.window_widths));
        return succeeded(kept) || propagate_error(kept, outcome);
    }

    /** Calls @p make with the GPU variant of the kind: Levenshtein over bytes or runes, or
     *  overlap. */
    template <typename make_type_>
    static void with_vram_variant(geometry_t const& geometry, make_type_&& make) noexcept {
        switch (geometry.metric) {
        case usearch_sequence_metric_levenshtein_utf8_k:
            return make(
                std::type_identity<sequence_vram_nvidia_variant<usearch_sequence_metric_levenshtein_utf8_k>> {});
        case usearch_sequence_metric_overlap_k:
            return make(std::type_identity<sequence_vram_nvidia_variant<usearch_sequence_metric_overlap_k>> {});
        default:
            return make(
                std::type_identity<sequence_vram_nvidia_variant<usearch_sequence_metric_levenshtein_bytes_k>> {});
        }
    }
#endif

    static usearch_sequence_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.metric, geometry.connectivity};
    }

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_sequence_index_description_t& description) noexcept {
        description.metadata = metadata_of(variant.geometry);
    }

    static bool accepts(usearch_sequence_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_sequences(arguments.sequences, arguments.count, outcome);
    }

    template <typename variant_type_>
    static bool accepts(variant_type_ const&, usearch_sequence_search_t const& arguments,
                        usearch_outcome_t* outcome) noexcept {
        return accepts_sequences(arguments.queries, arguments.count, outcome);
    }

    /** Where a get or a pull hands its strings out, null when a pull wants none. */
    template <typename arguments_type_>
    static usearch_tape_view_t* output_of(arguments_type_ const& arguments) noexcept {
        return arguments.sequences;
    }

    static status_t keep(nvme_executor_t& executor, std::size_t thread, std::string_view value, kept_t& kept) noexcept {
        return keep_row(executor, thread, bytes_of(value.data(), value.size()), kept);
    }

    /** Lands @p count strings, the @c i th as @p value_at hands it, as one u8 tape in @p memory. */
    template <typename variant_type_, typename value_at_type_, typename arguments_type_>
    static bool land(variant_type_ const&, std::size_t count, value_at_type_&& value_at,
                     growing_arena<forkunion_allocator_t>& memory, arguments_type_ const& arguments,
                     usearch_outcome_t* outcome) noexcept {
        std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
        if (!offsets) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != count; ++entry)
            offsets[entry + 1] = offsets[entry] + value_at(entry).size();
        byte_t* const bytes = carve<byte_t>(memory, offsets[count]);
        if (offsets[count] && !bytes)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        for (std::size_t entry = 0; entry != count; ++entry)
            if (auto const& value = value_at(entry); value.size())
                std::memcpy(bytes + offsets[entry], value.data(), value.size());
        *arguments.sequences = {bytes, offsets, usearch_scalar_u8_k};
        return true;
    }

    /** Adds one entry's bytes to a key's statistics, from any thread. */
    template <typename value_type_>
    static void tally(usearch_sequence_key_stats_t& stats, value_type_ const& value) noexcept {
        standard_atomic_ref<std::size_t>(stats.total_bytes).fetch_add(value.size(), std::memory_order_relaxed);
        raise_to(stats.largest_bytes, value.size());
    }
};

struct usearch_sequence_index_handle_t : basic_handle<sequence_variant_t, sequence_traits_t> {
    using basic_handle::basic_handle;
};

struct usearch_sequence_cursor_handle_t
    : basic_cursor<usearch_sequence_index_handle_t, ram_cursor_of<sequence_ram_variant<cpu_class_t::serial_k>>> {};

struct usearch_sequence_refinement_handle_t : basic_refinement<usearch_sequence_index_handle_t> {};

#pragma endregion Handles

extern "C" {

USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments) { basic_init_ram(arguments); }
USEARCH_EXPORT void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments) { basic_init_vram(arguments); }
USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    basic_load_ram<usearch_sequence_load_ram>(arguments);
}
USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments) {
    basic_unserved<usearch_sequence_view_ram>(arguments);
}
USEARCH_EXPORT void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments) {
    basic_view_nvme<usearch_sequence_view_nvme>(arguments);
}
USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index) { basic_free(index); }
USEARCH_EXPORT void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments) {
    basic_reserve<usearch_sequence_reserve>(arguments);
}
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments) {
    basic_clear<usearch_sequence_clear>(arguments);
}
USEARCH_EXPORT void usearch_sequence_describe(usearch_sequence_describe_t* arguments) {
    basic_describe<usearch_sequence_describe>(arguments);
}
USEARCH_EXPORT void usearch_sequence_count(usearch_sequence_count_t* arguments) {
    basic_count<usearch_sequence_count>(arguments);
}
USEARCH_EXPORT void usearch_sequence_insert(usearch_sequence_insert_t* arguments) {
    basic_insert<usearch_sequence_insert>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments) {
    basic_refine<usearch_sequence_refine>(arguments);
}
USEARCH_EXPORT void usearch_sequence_search(usearch_sequence_search_t* arguments) {
    basic_search<usearch_sequence_search>(arguments);
}
USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments) {
    basic_get<usearch_sequence_get>(arguments);
}
USEARCH_EXPORT void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments) {
    basic_cursor_init(arguments);
}
USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments) {
    basic_cursor_next<usearch_sequence_cursor_next>(arguments);
}
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor) { basic_free(cursor); }
USEARCH_EXPORT void usearch_sequence_sample(usearch_sequence_sample_t* arguments) {
    basic_unserved<usearch_sequence_sample>(arguments);
}
USEARCH_EXPORT void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments) {
    basic_absorb<usearch_sequence_absorb>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refinement_init(usearch_sequence_refinement_init_t* arguments) {
    basic_refinement_init<usearch_sequence_refinement_init>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refinement_next(usearch_sequence_refinement_next_t* arguments) {
    basic_refinement_next<usearch_sequence_refinement_next>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refinement_finish(usearch_sequence_refinement_finish_t* arguments) {
    basic_refinement_finish<usearch_sequence_refinement_finish>(arguments);
}
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    basic_save<usearch_sequence_save>(arguments);
}
USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments) {
    basic_metadata_read<usearch_sequence_metadata_read, sequence_traits_t>(arguments);
}

} // extern "C"
