/**
 *  @file src/sequence_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence modality's C surface: the variant over every compiled sequence arm, the
 *      handle owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Arms are ordered newest class first, the serial floor last. Their bodies compile in the basic
 *  class units, never here: the @c extern declarations below keep them out.
 */
#include <cstdio>  // `std::snprintf` - the capability string
#include <cstring> // `std::memcpy` - entries landing in an arena

#include <atomic>      // `std::atomic_flag` - a cursor's one pull at a time
#include <new>         // `std::nothrow`
#include <string_view> // `std::string_view` - checkpoint paths and tokens
#include <variant>     // `std::variant`, `std::visit`

#include "basic_specializations.cuh"

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

/** The one dynamically-typed sequence index every binding consumes: a variant over the arms this
 *  build compiled, each the basic index over owned strings scored with one class's kits, or a
 *  read-only view of a checkpoint on storage scored with the same. */
struct usearch_sequence_index_handle_t {

    using sequence_variant_t = std::variant< //
#if USEARCH_ARCH_CUDA_
        sequence_vram_nvidia_variant<sz_levenshtein_bytes_k>, //
        sequence_vram_nvidia_variant<sz_levenshtein_runes_k>, //
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

    sequence_variant_t variants;

    /** The class whose kits the arm scores with, named once at construction. */
    char capabilities_string[32] {};

    /** Builds the arm @p arm_type_ over @p placement: a RAM arm's allocator, or a GPU arm's
     *  device. */
    template <typename arm_type_, typename placement_type_>
    usearch_sequence_index_handle_t(std::in_place_type_t<arm_type_> tag, placement_type_ const& placement) noexcept
        : variants(tag, placement) {}
};

/** Visits the arm of @p index with @p visitor when it lives in RAM; a GPU arm serves the
 *  reservation, insertion, search and description verbs alone, and refuses the rest, and an NVMe
 *  view refuses as read-only whatever its verbs don't serve first. */
template <typename visitor_type_>
static void visit_ram_variant(usearch_sequence_index_handle_t& index, usearch_outcome_t* outcome,
                              visitor_type_&& visitor) noexcept {
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if constexpr (is_ram_index<arm_type_>) visitor(arm);
            else if constexpr (is_nvme_index<arm_type_>)
                propagate_error(usearch_status_read_only_k, outcome, "An NVMe view never mutates");
            else propagate_error(usearch_status_unsupported_k, outcome, "A GPU sequence index serves no such verb yet");
        },
        index.variants);
}

/** Visits the arm of @p index with @p visitor when it is an NVMe view, answering whether it was. */
template <typename visitor_type_>
static bool visit_nvme_variant(usearch_sequence_index_handle_t& index, visitor_type_&& visitor) noexcept {
    return std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if constexpr (is_nvme_index<arm_type_>) return visitor(arm), true;
            else return false;
        },
        index.variants);
}

/** The GPU ordinal @p arm computes on, where a call's arena must live, or the host's. */
template <typename arm_type_>
static int device_of(arm_type_ const& arm) noexcept {
    if constexpr (is_vram_index<arm_type_>) return arm.index.device_specs().ordinal;
    else return host_device_k;
}

/** The basic cursor over the sequence index, whose type no class changes - only the kits a metric
 *  scores with. */
using sequence_cursor_t = basic::ram_cursor<sequence_ram_variant<cpu_class_t::serial_k>::index_t,
                                            synchronization_t::concurrent_k, forkunion_allocator_t>;

struct usearch_sequence_cursor_handle_t {

    /** A RAM arm's snapshot cursor, or the next row of an NVMe view, whose rows never change. */
    std::variant<std::monostate, sequence_cursor_t, std::size_t> cursor;

    /** The handle the cursor was opened on, which every pull names again. */
    usearch_sequence_index_t index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The kind, window widths and connectivity one handle freezes, as an init or a checkpoint states
 *  them. */
struct sequence_geometry_t {
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
    std::span<std::size_t const> window_widths;
    std::size_t connectivity = 0;
};

/** Allocates the handle holding @p arm_type_, freezing @p geometry into it; null once @p outcome
 *  records why not. */
template <typename arm_type_>
static usearch_sequence_index_t make_sequence_handle(sequence_geometry_t const& geometry,
                                                     usearch_outcome_t* outcome) noexcept {
    usearch_sequence_index_t const handle = new (std::nothrow)
        usearch_sequence_index_handle_t(std::in_place_type<arm_type_>, forkunion_allocator_t {no_memory_domain_k});
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    arm_type_& arm = std::get<arm_type_>(handle->variants);
    arm.metric = geometry.metric;
    arm.connectivity = geometry.connectivity ? geometry.connectivity : std::size_t(connectivity_32_k);
    if (failed(arm.window_widths.reserve_exactly(geometry.window_widths.size())))
        return delete handle, propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to keep the widths"),
               nullptr;
    for (std::size_t const width : geometry.window_widths) arm.window_widths.push_back_reserved(width);
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(arm_type_::cpu_class_k)));
    return handle;
}

/** One rung of the arms @p arm_template_ names, RAM or NVMe: taken when the build compiled the
 *  class and the machine reports it, else falls through; the serial floor ends every ladder. */
template <template <cpu_class_t> class arm_template_, cpu_class_t cpu_class_ = cpu_class_t::serial_k,
          cpu_class_t... lower_classes_>
static usearch_sequence_index_t descend_sequence_ladder(sequence_geometry_t const& geometry, nk_capability_t available,
                                                        usearch_outcome_t* outcome) noexcept {
    using arm_t = arm_template_<cpu_class_>;
    if constexpr (cpu_class_ == cpu_class_t::serial_k) return make_sequence_handle<arm_t>(geometry, outcome);
    else {
        if constexpr (cpu_class_compiled(cpu_class_))
            if (numkong_admits(available, cpu_class_)) return make_sequence_handle<arm_t>(geometry, outcome);
        return descend_sequence_ladder<arm_template_, lower_classes_...>(geometry, available, outcome);
    }
}

/** Lowers an init's or a checkpoint's frozen kind onto one arm of @p arm_template_, refusing the
 *  kinds no kit serves and the parameters a kind does not take; null once @p outcome records why
 *  not. */
template <template <cpu_class_t> class arm_template_>
static usearch_sequence_index_t make_sequence_index(sequence_geometry_t const& geometry,
                                                    usearch_sequence_weights_t const* weights,
                                                    usearch_ram_options_t const& ram,
                                                    usearch_outcome_t* outcome) noexcept {
    switch (geometry.metric) {
    case usearch_sequence_metric_levenshtein_bytes_k:
    case usearch_sequence_metric_levenshtein_utf8_k:
    case usearch_sequence_metric_overlap_k: break;
    case usearch_sequence_metric_needleman_wunsch_k:
    case usearch_sequence_metric_needleman_wunsch_gotoh_k:
    case usearch_sequence_metric_smith_waterman_k:
    case usearch_sequence_metric_smith_waterman_gotoh_k:
        return propagate_error(usearch_status_unsupported_k, outcome, "No alignment kernels serve this kind yet"),
               nullptr;
    default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown sequence metric"), nullptr;
    }
    if (weights)
        return propagate_error(usearch_status_config_mismatch_k, outcome, "Only an alignment kind takes weights"),
               nullptr;
    bool const overlap = geometry.metric == usearch_sequence_metric_overlap_k;
    if (overlap == geometry.window_widths.empty())
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "The overlap kind needs window widths, and only it takes any"),
               nullptr;
    for (std::size_t const width : geometry.window_widths)
        if (!width) return propagate_error(usearch_status_invalid_argument_k, outcome, "A window is empty"), nullptr;
    // One basic index per handle, so one memory domain at most; values live on the heap and land
    // where first touched.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A sequence index places one shard"), nullptr;
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_sequence_ladder<arm_template_, cpu_class_t::intel_skylake_k, cpu_class_t::intel_haswell_k>(
        geometry, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_sequence_ladder<arm_template_, cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>(geometry, available,
                                                                                                  outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_sequence_ladder<arm_template_, cpu_class_t::alibaba_c930_k>(geometry, available, outcome);
#elif NUMKONG_ARCH_WASM_
    return descend_sequence_ladder<arm_template_, cpu_class_t::wasm64_extended_k>(geometry, available, outcome);
#else
    return descend_sequence_ladder<arm_template_>(geometry, available, outcome);
#endif
}

#if USEARCH_ARCH_CUDA_

/** Places a GPU sequence index on the one device @p arguments name, refusing the kinds no device
 *  kit serves and the parameters byte Levenshtein does not take; null once @p outcome records why
 *  not. */
static usearch_sequence_index_t make_sequence_vram_index(usearch_sequence_init_vram_t const& arguments,
                                                         usearch_outcome_t* outcome) noexcept {
    switch (arguments.metric) {
    case usearch_sequence_metric_levenshtein_bytes_k:
    case usearch_sequence_metric_levenshtein_utf8_k: break;
    case usearch_sequence_metric_overlap_k:
    case usearch_sequence_metric_needleman_wunsch_k:
    case usearch_sequence_metric_needleman_wunsch_gotoh_k:
    case usearch_sequence_metric_smith_waterman_k:
    case usearch_sequence_metric_smith_waterman_gotoh_k:
        return propagate_error(usearch_status_unsupported_k, outcome, "No GPU kit serves this sequence kind yet"),
               nullptr;
    default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown sequence metric"), nullptr;
    }
    if (arguments.weights || arguments.window_widths_count)
        return propagate_error(usearch_status_config_mismatch_k, outcome, "Levenshtein takes no parameters"), nullptr;
    if (arguments.values_per_key)
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU sequence index bounds no key's entries"),
               nullptr;
    usearch_vram_options_t const& vram = arguments.vram;
    if (vram.devices_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A sequence index places one shard"), nullptr;
    std::size_t const ordinal = vram.devices && vram.devices_count ? static_cast<std::size_t>(vram.devices[0]) : 0;
    std::span<gpu_device_t const> const listed = gpu_devices();
    if (ordinal >= listed.size())
        return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), nullptr;
    gpu_device_t const& device = listed[ordinal];
    if (device.vendor != gpu_vendor_t::nvidia_k)
        return propagate_error(usearch_status_device_k, outcome, "No GPU kernels serve sequences on this vendor yet"),
               nullptr;
    int const vendor_ordinal = static_cast<int>(device.vendor_ordinal);
    usearch_sequence_index_t const handle =
        arguments.metric == usearch_sequence_metric_levenshtein_utf8_k
            ? new (std::nothrow) usearch_sequence_index_handle_t(
                  std::in_place_type<sequence_vram_nvidia_variant<sz_levenshtein_runes_k>>, vendor_ordinal)
            : new (std::nothrow) usearch_sequence_index_handle_t(
                  std::in_place_type<sequence_vram_nvidia_variant<sz_levenshtein_bytes_k>>, vendor_ordinal);
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if constexpr (is_vram_index<arm_type_>)
                arm.connectivity = arguments.connectivity ? arguments.connectivity : std::size_t(connectivity_32_k);
        },
        handle->variants);
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)));
    return handle;
}

#endif

/** Reads the kind, window widths and connectivity a checkpoint at @p path froze, the widths into
 *  @p widths, since a field's text dies with the callback that hands it. */
static sequence_geometry_t checkpoint_geometry(std::string_view path,
                                               growing_buffer<std::size_t, forkunion_allocator_t>& widths,
                                               status_t& status) noexcept {
    std::size_t connectivity = 0;
    status_t widths_status = status_t::success_k;
    // A checkpoint spells its widths as decimals between commas.
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") connectivity = decimal_in(field.value);
        else if (field.key == "window_widths") {
            std::size_t windows = 1;
            for (char const character : field.value) windows += character == ',';
            widths_status = widths.reserve_exactly(windows);
            for (std::string_view rest = field.value; succeeded(widths_status) && !rest.empty();) {
                std::size_t const comma = rest.find(',');
                widths.push_back_reserved(decimal_in(rest.substr(0, comma)));
                rest = comma == std::string_view::npos ? std::string_view {} : rest.substr(comma + 1);
            }
        }
    };
    auto const [metadata, read_status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
    status = failed(read_status) ? read_status : widths_status;
    return {sequence_metric_of(metadata.metric.view()), to_span(widths), connectivity};
}

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

#pragma endregion Handles

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (arguments->window_widths_count && !arguments->window_widths)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The window widths must not be null"),
               void();
    sequence_geometry_t const geometry {
        arguments->metric,
        {arguments->window_widths, arguments->window_widths_count},
        arguments->connectivity,
    };
    usearch_sequence_index_t const index = make_sequence_index<sequence_ram_variant>(geometry, arguments->weights,
                                                                                     arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#if USEARCH_ARCH_CUDA_
USEARCH_EXPORT void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_sequence_index_t const index = make_sequence_vram_index(*arguments, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}
#else
USEARCH_EXPORT void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build compiled no GPU kernels for sequences");
}
#endif

USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    if (submitted<usearch_sequence_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    growing_buffer<std::size_t, forkunion_allocator_t> widths {forkunion_allocator_t {no_memory_domain_k}};
    status_t read_status = status_t::success_k;
    sequence_geometry_t const geometry = checkpoint_geometry(path, widths, read_status);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_sequence_index_t const index = make_sequence_index<sequence_ram_variant>(geometry, arguments->weights,
                                                                                     arguments->ram, outcome);
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

USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Sequences own their bytes, so a checkpoint loads instead");
}

USEARCH_EXPORT void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments) {
    if (submitted<usearch_sequence_view_nvme>(arguments)) return;
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
    growing_buffer<std::size_t, forkunion_allocator_t> widths {forkunion_allocator_t {no_memory_domain_k}};
    status_t read_status = status_t::success_k;
    sequence_geometry_t const geometry = checkpoint_geometry(path, widths, read_status);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_sequence_index_t const index = make_sequence_index<sequence_nvme_variant>(
        geometry, arguments->weights, usearch_ram_options_t {}, outcome);
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

USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index) {
    finish_calls_using(index);
    delete index;
}

USEARCH_EXPORT void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments) {
    if (submitted<usearch_sequence_reserve>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_writable(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        if constexpr (is_vram_index<arm_type_>) {
            status_t reserved = arm.index
                                    .reserve({arguments->entries, static_cast<connectivity_t>(arm.connectivity),
                                              arguments->payload_bytes})
                                    .status;
            if (succeeded(reserved)) reserved = arm.keys.cover(arguments->entries);
            propagate_error(reserved, outcome);
        }
        else propagate_error(arm.index.reserve(arguments->entries).status, outcome);
    });
}

USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments) {
    if (submitted<usearch_sequence_clear>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_ram_variant(*arguments->index, outcome,
                      [&](auto& arm) noexcept { propagate_error(arm.index.clear(), outcome); });
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT void usearch_sequence_describe(usearch_sequence_describe_t* arguments) {
    if (submitted<usearch_sequence_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_t const index = arguments->index;
    std::visit(
        [&](auto& arm) noexcept {
            usearch_sequence_index_description_t& description = *arguments->description;
            description.metadata = {arm.metric, arm.connectivity};
            description.size = arm.index.size();
            description.capacity = arm.index.capacity();
            description.memory_usage = sizeof(*index) + arm.index.memory_usage();
            description.capabilities = index->capabilities_string;
        },
        index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_sequence_count(usearch_sequence_count_t* arguments) {
    if (submitted<usearch_sequence_count>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    usearch_sequence_key_stats_t& stats = *arguments->key_stats;
    auto count_nvme = [&](auto& arm) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        auto measure = [&](std::size_t, std::size_t, std::size_t, std::string_view value) noexcept {
            std::atomic_ref<std::size_t>(stats.total_bytes).fetch_add(value.size(), std::memory_order_relaxed);
            std::atomic_ref<std::size_t> largest {stats.largest_bytes};
            std::size_t seen = largest.load(std::memory_order_relaxed);
            while (seen < value.size() &&
                   !largest.compare_exchange_weak(seen, value.size(), std::memory_order_relaxed));
        };
        auto take_count = [&](std::size_t entries) noexcept { return stats.entries = entries, status_t::success_k; };
        propagate_error(values_under(arm.index, arguments->key, *executor, take_count, measure), outcome);
    };
    if (visit_nvme_variant(*arguments->index, count_nvme)) return;
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        arm.index.find_each(uuid_of(arguments->key), [&](typename arm_type_::index_t::node_id_t node) noexcept {
            std::size_t const bytes = arm.index.published_value_of(node).size();
            ++stats.entries;
            stats.total_bytes += bytes;
            stats.largest_bytes = max_of(stats.largest_bytes, bytes);
        });
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT void usearch_sequence_insert(usearch_sequence_insert_t* arguments) {
    if (submitted<usearch_sequence_insert>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    std::size_t const count = arguments->count;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!arguments->keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must not be null"), void();
    if (arguments->keys.stride && arguments->keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (!accepts_sequences(arguments->sequences, count, outcome)) return;
    visit_writable(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        if (crossed(
                arm.insert(arguments->keys, arguments->sequences, count, arguments->expansion_factor, ticks, *executor),
                arguments->stats, outcome))
            propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments) {
    if (submitted<usearch_sequence_refine>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency refines yet");
}

USEARCH_EXPORT void usearch_sequence_search(usearch_sequence_search_t* arguments) {
    if (submitted<usearch_sequence_search>(arguments)) return;
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
            // The matches land on the arena's host pages, which a device arena takes over after, so
            // a device walk serves a host arena as well.
            bool const hosted = arguments->arena && arguments->arena->device == host_device_k;
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              hosted ? host_device_k : device_of(arm),
                                                              {arguments->queries.data}, outcome);
            if (!arena) return;
            if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            if (!accepts_sequences(arguments->queries, count, outcome)) return;
            matches_landing_t landing;
            if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            if (!crossed(arm.search(arguments->queries, count, wanted, arguments->expansion_factor,
                                    arguments->predicate, landing, ticks, *executor),
                         arguments->stats, outcome) ||
                !publish_matches(*arena, count, landing, outcome))
                return;
            *arguments->matches = landing.view();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments) {
    if (submitted<usearch_sequence_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->sequences = {};
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    // A view's strings land in its threads' arenas first, as their lengths are unknown until read.
    auto get_nvme = [&](auto& arm) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        bytes_view_t* kept = nullptr;
        status_t keeping = status_t::success_k;
        auto keep = [&](std::size_t, std::size_t rank, std::size_t thread, std::string_view value) noexcept {
            keep_first_failure(keeping, keep_row(*executor, thread, bytes_of(value.data(), value.size()), kept[rank]));
        };
        std::size_t held = 0;
        std::size_t* offsets = nullptr;
        auto carve_entries = [&](std::size_t entries) noexcept {
            held = entries;
            if (!held) return status_t::success_k;
            kept = carve<bytes_view_t>(arena->memory, held);
            offsets = carve<std::size_t>(arena->memory, held + 1);
            return kept && offsets ? status_t::success_k : status_t::bad_alloc_k;
        };
        if (status_t const read = values_under(arm.index, arguments->key, *executor, carve_entries, keep); failed(read))
            return propagate_error(read, outcome), void();
        if (failed(keeping)) return propagate_error(keeping, outcome), void();
        if (!held) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != held; ++entry) offsets[entry + 1] = offsets[entry] + kept[entry].size();
        byte_t* const bytes = carve<byte_t>(arena->memory, offsets[held]);
        if (offsets[held] && !bytes)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        for (std::size_t entry = 0; entry != held; ++entry)
            if (!kept[entry].empty()) std::memcpy(bytes + offsets[entry], kept[entry].data(), kept[entry].size());
        *arguments->sequences = {bytes, offsets, usearch_scalar_u8_k};
        if (arguments->written) *arguments->written = held;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    };
    if (visit_nvme_variant(*arguments->index, get_nvme)) return;
    visit_ram_variant(*arguments->index, outcome, [&](auto& arm) noexcept {
        usearch_executor_handle_t call_executor;
        auto* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        if (status_t const rewound = executor->reserve(0); failed(rewound))
            return propagate_error(rewound, outcome), void();
        auto const ids = ids_under(arm.index, arguments->key, thread_allocator_t {executor->arena(0)});
        if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        std::size_t* const offsets = carve<std::size_t>(arena->memory, ids.size() + 1);
        if (!offsets)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != ids.size(); ++entry)
            offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(ids[entry]).size();
        byte_t* const bytes = carve<byte_t>(arena->memory, offsets[ids.size()]);
        if (offsets[ids.size()] && !bytes)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        for (std::size_t entry = 0; entry != ids.size(); ++entry) {
            auto const& value = arm.index.published_value_of(ids[entry]);
            std::memcpy(bytes + offsets[entry], value.data(), value.size());
        }
        *arguments->sequences = {bytes, offsets, usearch_scalar_u8_k};
        if (arguments->written) *arguments->written = ids.size();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Insertion and Search

#pragma region Cursors Sampling and Absorbing

USEARCH_EXPORT void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    usearch_sequence_cursor_t const cursor = new (std::nothrow) usearch_sequence_cursor_handle_t {};
    if (!cursor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    cursor->index = arguments->index;
    auto open_nvme = [&](auto&) noexcept { cursor->cursor.template emplace<std::size_t>(0); };
    if (!visit_nvme_variant(*arguments->index, open_nvme))
        visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
            using cursor_t =
                basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;
            cursor_t& opened = cursor->cursor.template emplace<cursor_t>(arm.index,
                                                                         forkunion_allocator_t {no_memory_domain_k});
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

USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments) {
    if (submitted<usearch_sequence_cursor_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->sequences) *arguments->sequences = {};
    *arguments->written = 0;
    usearch_sequence_cursor_t const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    // A view reads its rows through the executor, landing strings in its threads' arenas first.
    auto next_nvme = [&](auto& arm) noexcept {
        std::size_t* const row = std::get_if<std::size_t>(&cursor->cursor);
        if (!row) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        std::size_t const taking = min_of(arm.index.size() - *row, arguments->wanted);
        if (!taking) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
        bytes_view_t* const kept = carve<bytes_view_t>(arena->memory, taking);
        std::size_t* const offsets = arguments->sequences ? carve<std::size_t>(arena->memory, taking + 1) : nullptr;
        if (!keys || !kept || (arguments->sequences && !offsets))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        status_t keeping = status_t::success_k;
        auto keep = [&](std::size_t entry, std::size_t thread, std::string_view value) noexcept {
            if (offsets)
                keep_first_failure(keeping,
                                   keep_row(*executor, thread, bytes_of(value.data(), value.size()), kept[entry]));
        };
        std::span<us::utilities::uuid_t> const identities {reinterpret_cast<us::utilities::uuid_t*>(keys), taking};
        if (status_t const read = arm.index.read_rows(*row, identities, *executor, keep); failed(read))
            return propagate_error(read, outcome), void();
        if (failed(keeping)) return propagate_error(keeping, outcome), void();
        if (offsets) {
            offsets[0] = 0;
            for (std::size_t entry = 0; entry != taking; ++entry)
                offsets[entry + 1] = offsets[entry] + kept[entry].size();
            byte_t* const bytes = carve<byte_t>(arena->memory, offsets[taking]);
            if (offsets[taking] && !bytes)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            for (std::size_t entry = 0; entry != taking; ++entry)
                if (!kept[entry].empty()) std::memcpy(bytes + offsets[entry], kept[entry].data(), kept[entry].size());
            *arguments->sequences = {bytes, offsets, usearch_scalar_u8_k};
        }
        *row += taking;
        *arguments->keys = {keys, sizeof(usearch_key_t)};
        *arguments->written = taking;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    };
    if (arena && !visit_nvme_variant(*arguments->index, next_nvme))
        visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
            using node_id_t = typename arm_type_::index_t::node_id_t;
            using cursor_t =
                basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;
            cursor_t* const typed = std::get_if<cursor_t>(&cursor->cursor);
            if (!typed) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
            std::size_t const remaining = *typed ? typed->visible_count() - typed->id() : 0;
            std::size_t const taking = min_of(remaining, arguments->wanted);
            usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
            std::size_t* const offsets = arguments->sequences ? carve<std::size_t>(arena->memory, taking + 1) : nullptr;
            if ((taking && !keys) || (arguments->sequences && !offsets))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            node_id_t const first = typed->id();
            if (offsets) offsets[0] = 0;
            for (std::size_t entry = 0; entry != taking; ++entry) {
                node_id_t const node = static_cast<node_id_t>(first + entry);
                keys[entry] = std::bit_cast<usearch_key_t>(arm.index.key_of(node).value);
                if (offsets) offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(node).size();
            }
            if (offsets) {
                byte_t* const bytes = carve<byte_t>(arena->memory, offsets[taking]);
                if (offsets[taking] && !bytes)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                for (std::size_t entry = 0; entry != taking; ++entry) {
                    auto const& value = arm.index.published_value_of(static_cast<node_id_t>(first + entry));
                    std::memcpy(bytes + offsets[entry], value.data(), value.size());
                }
                *arguments->sequences = {bytes, offsets, usearch_scalar_u8_k};
            }
            for (std::size_t step = 0; step != taking; ++step) ++*typed;
            *arguments->keys = {keys, sizeof(usearch_key_t)};
            *arguments->written = taking;
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
    cursor->busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor) {
    finish_calls_using(cursor);
    delete cursor;
}

USEARCH_EXPORT void usearch_sequence_sample(usearch_sequence_sample_t* arguments) {
    if (submitted<usearch_sequence_sample>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency samples yet");
}

USEARCH_EXPORT void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments) {
    if (submitted<usearch_sequence_absorb>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency absorbs donor handles yet");
}

#pragma endregion Cursors Sampling and Absorbing

#pragma region Serialization

USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    if (submitted<usearch_sequence_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    if (path.ends_with(".json"))
        return propagate_error(usearch_status_unsupported_k, outcome, "A sequence index saves one monolith"), void();
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

USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments) {
    if (submitted<usearch_sequence_metadata_read>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_sequence_metadata_t& metadata = *arguments->metadata;
    metadata = {};
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") metadata.connectivity = decimal_in(field.value);
    };
    auto const [read, status] = safegraphs_read_index_metadata(as_text(arguments->path), std::allocator<byte_t> {},
                                                               per_field);
    if (failed(status)) return propagate_error(status, outcome), void();
    metadata.metric = sequence_metric_of(read.metric.view());
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

} // extern "C"
