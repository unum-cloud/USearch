/**
 *  @file src/shapes_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The shapes modality's C surface: the variant over every compiled shapes arm, the handle
 *      owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Arms are ordered newest class first inside each scalar, the serial floor last. Their bodies
 *  compile in the basic class units, never here: the @c extern declarations below keep them out, as
 *  the dense dispatch unit keeps its own.
 */
#include <cstdio>  // `std::snprintf` - the capability string
#include <cstring> // `std::memcpy` - keys landing in a cursor's run

#include <atomic>      // `std::atomic_flag` - a cursor's one pull at a time
#include <new>         // `std::nothrow`
#include <string_view> // `std::string_view` - checkpoint paths and tokens
#include <variant>     // `std::variant`, `std::visit`

#include "basic_specializations.cuh"

#if USEARCH_TARGET_INTEL_SKYLAKE
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::intel_skylake_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::intel_skylake_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::intel_skylake_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::intel_skylake_k>;
#endif // USEARCH_TARGET_INTEL_SKYLAKE
#if USEARCH_TARGET_INTEL_HASWELL
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::intel_haswell_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::intel_haswell_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::intel_haswell_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::intel_haswell_k>;
#endif // USEARCH_TARGET_INTEL_HASWELL
#if USEARCH_TARGET_APPLE_M2
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::apple_m2_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::apple_m2_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::apple_m2_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::apple_m2_k>;
#endif // USEARCH_TARGET_APPLE_M2
#if USEARCH_TARGET_AWS_G2
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::aws_g2_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::aws_g2_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::aws_g2_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::aws_g2_k>;
#endif // USEARCH_TARGET_AWS_G2
#if USEARCH_TARGET_ALIBABA_C930
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>;
#endif // USEARCH_TARGET_ALIBABA_C930
#if USEARCH_TARGET_WASM64_EXTENDED
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::wasm64_extended_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::wasm64_extended_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::wasm64_extended_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::wasm64_extended_k>;
#endif // USEARCH_TARGET_WASM64_EXTENDED
extern template struct shapes_ram_variant<nk::f32_t, cpu_class_t::serial_k>;
extern template struct shapes_ram_variant<nk::f16_t, cpu_class_t::serial_k>;
extern template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::serial_k>;
extern template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::serial_k>;

#pragma region Handles

/** The one dynamically-typed shapes index every binding consumes: a variant over the arms this
 *  build compiled, each the basic index over one scalar's packs scored with one class's kits, or a
 *  read-only view of a checkpoint on storage scored with the same. */
struct usearch_shapes_index_handle_t {

    using shapes_variant_t = std::variant< //
#if USEARCH_ARCH_CUDA_
        shapes_vram_nvidia_variant<nk::f16_t, us::shapes::f16_ampere_chamfer_kit>,   //
        shapes_vram_nvidia_variant<nk::f16_t, us::shapes::f16_ampere_hausdorff_kit>, //
        shapes_vram_nvidia_variant<nk::f32_t, us::shapes::f32_ampere_chamfer_kit>,   //
        shapes_vram_nvidia_variant<nk::f32_t, us::shapes::f32_ampere_hausdorff_kit>, //
#endif
#if USEARCH_TARGET_INTEL_SKYLAKE
        shapes_ram_variant<nk::f32_t, cpu_class_t::intel_skylake_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::intel_skylake_k>,  //
        shapes_nvme_variant<nk::f32_t, cpu_class_t::intel_skylake_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_TARGET_INTEL_HASWELL
        shapes_ram_variant<nk::f32_t, cpu_class_t::intel_haswell_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::intel_haswell_k>,  //
        shapes_nvme_variant<nk::f32_t, cpu_class_t::intel_haswell_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_TARGET_APPLE_M2
        shapes_ram_variant<nk::f32_t, cpu_class_t::apple_m2_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::apple_m2_k>,  //
        shapes_nvme_variant<nk::f32_t, cpu_class_t::apple_m2_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_TARGET_AWS_G2
        shapes_ram_variant<nk::f32_t, cpu_class_t::aws_g2_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::aws_g2_k>,  //
        shapes_nvme_variant<nk::f32_t, cpu_class_t::aws_g2_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::aws_g2_k>, //
#endif
#if USEARCH_TARGET_ALIBABA_C930
        shapes_ram_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>,  //
        shapes_nvme_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_TARGET_WASM64_EXTENDED
        shapes_ram_variant<nk::f32_t, cpu_class_t::wasm64_extended_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::wasm64_extended_k>,  //
        shapes_nvme_variant<nk::f32_t, cpu_class_t::wasm64_extended_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::wasm64_extended_k>, //
#endif
        shapes_nvme_variant<nk::f32_t, cpu_class_t::serial_k>, //
        shapes_nvme_variant<nk::f16_t, cpu_class_t::serial_k>, //
        shapes_ram_variant<nk::f32_t, cpu_class_t::serial_k>,  //
        shapes_ram_variant<nk::f16_t, cpu_class_t::serial_k>>;

    shapes_variant_t variants;

    /** The class whose kits the arm scores with, named once at construction. */
    char capabilities_string[32] {};

    /** Builds the arm @p arm_type_ over @p placement: a RAM arm's allocator, or a GPU arm's
     *  device. */
    template <typename arm_type_, typename placement_type_>
    usearch_shapes_index_handle_t(std::in_place_type_t<arm_type_> tag, placement_type_ const& placement) noexcept
        : variants(tag, placement) {}
};

/** Visits the arm of @p index with @p visitor when it lives in RAM; a GPU arm serves the
 *  reservation, insertion, search and description verbs alone, and refuses the rest, and an NVMe
 *  view refuses as read-only whatever its verbs don't serve first. */
template <typename visitor_type_>
static void visit_ram_variant(usearch_shapes_index_handle_t& index, usearch_outcome_t* outcome,
                              visitor_type_&& visitor) noexcept {
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            if constexpr (is_ram_index<arm_type_>) visitor(arm);
            else if constexpr (is_nvme_index<arm_type_>)
                propagate_error(usearch_status_read_only_k, outcome, "An NVMe view never mutates");
            else propagate_error(usearch_status_unsupported_k, outcome, "A GPU shapes index serves no such verb yet");
        },
        index.variants);
}

/** Visits the arm of @p index with @p visitor when it is an NVMe view, answering whether it was. */
template <typename visitor_type_>
static bool visit_nvme_variant(usearch_shapes_index_handle_t& index, visitor_type_&& visitor) noexcept {
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

/** The basic cursor over one scalar's index, whose type no class changes - only the kits a metric
 *  scores with. */
template <typename scalar_type_>
using shapes_cursor_t = basic::ram_cursor<typename shapes_ram_variant<scalar_type_, cpu_class_t::serial_k>::index_t,
                                          synchronization_t::concurrent_k, forkunion_allocator_t>;

struct usearch_shapes_cursor_handle_t {

    /** A RAM arm's snapshot cursor, or the next row of an NVMe view, whose rows never change. */
    std::variant<std::monostate, shapes_cursor_t<nk::f32_t>, shapes_cursor_t<nk::f16_t>, std::size_t> cursor;

    /** The handle the cursor was opened on, which every pull names again. */
    usearch_shapes_index_t index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The geometry and metric one handle freezes, as an init or a checkpoint states them. */
struct shapes_geometry_t {
    usearch_shapes_metric_t metric = usearch_shapes_metric_unknown_k;
    std::size_t points = 0;
    std::size_t connectivity = 0;
};

/** Allocates the handle holding @p arm_type_, freezing @p geometry into it; null once @p outcome
 *  records why not. */
template <typename arm_type_>
static usearch_shapes_index_t make_shapes_handle(shapes_geometry_t const& geometry,
                                                 usearch_outcome_t* outcome) noexcept {
    usearch_shapes_index_t const handle = new (std::nothrow)
        usearch_shapes_index_handle_t(std::in_place_type<arm_type_>, forkunion_allocator_t {no_memory_domain_k});
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    arm_type_& arm = std::get<arm_type_>(handle->variants);
    arm.metric = geometry.metric;
    arm.points = geometry.points;
    arm.connectivity = geometry.connectivity ? geometry.connectivity : std::size_t(connectivity_32_k);
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(arm_type_::cpu_class_k)));
    return handle;
}

/** One rung of the arms @p arm_template_ names, RAM or NVMe: taken when the build compiled the
 *  class and the machine reports it, else falls through; the serial floor ends every ladder. */
template <template <nk::shapes_dtype, cpu_class_t> class arm_template_, typename scalar_type_,
          cpu_class_t cpu_class_ = cpu_class_t::serial_k, cpu_class_t... lower_classes_>
static usearch_shapes_index_t descend_shapes_ladder(shapes_geometry_t const& geometry, nk_capability_t available,
                                                    usearch_outcome_t* outcome) noexcept {
    using arm_t = arm_template_<scalar_type_, cpu_class_>;
    if constexpr (cpu_class_ == cpu_class_t::serial_k) return make_shapes_handle<arm_t>(geometry, outcome);
    else {
        if constexpr (cpu_class_compiled(cpu_class_))
            if (numkong_admits(available, cpu_class_)) return make_shapes_handle<arm_t>(geometry, outcome);
        return descend_shapes_ladder<arm_template_, scalar_type_, lower_classes_...>(geometry, available, outcome);
    }
}

/** Picks the arm of @p arm_template_ for one scalar: CPU classes newest first, the serial floor
 *  last. */
template <template <nk::shapes_dtype, cpu_class_t> class arm_template_, typename scalar_type_>
static usearch_shapes_index_t make_shapes_index(shapes_geometry_t const& geometry,
                                                usearch_outcome_t* outcome) noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_shapes_ladder<arm_template_, scalar_type_, cpu_class_t::intel_skylake_k,
                                 cpu_class_t::intel_haswell_k>(geometry, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_shapes_ladder<arm_template_, scalar_type_, cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>(
        geometry, available, outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_shapes_ladder<arm_template_, scalar_type_, cpu_class_t::alibaba_c930_k>(geometry, available,
                                                                                           outcome);
#elif NUMKONG_ARCH_WASM_
    return descend_shapes_ladder<arm_template_, scalar_type_, cpu_class_t::wasm64_extended_k>(geometry, available,
                                                                                              outcome);
#else
    return descend_shapes_ladder<arm_template_, scalar_type_>(geometry, available, outcome);
#endif
}

/** Lowers an init's or a checkpoint's frozen geometry onto one arm of @p arm_template_, refusing
 *  the scalars and kinds no arm serves; null once @p outcome records why not. */
template <template <nk::shapes_dtype, cpu_class_t> class arm_template_>
static usearch_shapes_index_t make_shapes_index(shapes_geometry_t const& geometry, usearch_scalar_t quantization,
                                                usearch_ram_options_t const& ram, usearch_outcome_t* outcome) noexcept {
    if (token_of(geometry.metric).empty())
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric"), nullptr;
    // One basic index per handle, so one memory domain at most; values live on the heap and land
    // where first touched.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A shapes index places one shard"), nullptr;
    switch (quantization) {
    case usearch_scalar_f16_k: return make_shapes_index<arm_template_, nk::f16_t>(geometry, outcome);
    case usearch_scalar_f32_k: return make_shapes_index<arm_template_, nk::f32_t>(geometry, outcome);
    default: return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points"), nullptr;
    }
}

#if USEARCH_ARCH_CUDA_

/** Allocates the handle holding the GPU arm @p arm_type_ on @p device, freezing @p arguments into
 *  it; null once @p outcome records why not. */
template <typename arm_type_>
static usearch_shapes_index_t make_shapes_vram_handle(usearch_shapes_init_vram_t const& arguments,
                                                      gpu_device_t const& device, usearch_outcome_t* outcome) noexcept {
    usearch_shapes_index_t const handle = new (std::nothrow)
        usearch_shapes_index_handle_t(std::in_place_type<arm_type_>, static_cast<int>(device.vendor_ordinal));
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    arm_type_& arm = std::get<arm_type_>(handle->variants);
    arm.metric = arguments.metric;
    arm.points = arguments.points;
    arm.connectivity = arguments.connectivity ? arguments.connectivity : std::size_t(connectivity_32_k);
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)));
    return handle;
}

/** Places a GPU shapes index on the one device @p arguments name, refusing the kinds and scalars no
 *  device kit serves; null once @p outcome records why not. */
static usearch_shapes_index_t make_shapes_vram_index(usearch_shapes_init_vram_t const& arguments,
                                                     usearch_outcome_t* outcome) noexcept {
    switch (arguments.metric) {
    case usearch_shapes_metric_chamfer_k:
    case usearch_shapes_metric_hausdorff_k: break;
    case usearch_shapes_metric_gromov_wasserstein_k:
    case usearch_shapes_metric_wasserstein_procrustes_k:
        return propagate_error(usearch_status_unsupported_k, outcome, "Only Chamfer and Hausdorff serve a GPU yet"),
               nullptr;
    default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric"), nullptr;
    }
    if (arguments.quantization != usearch_scalar_f16_k && arguments.quantization != usearch_scalar_f32_k)
        return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points"), nullptr;
    if (arguments.values_per_key)
        return propagate_error(usearch_status_unsupported_k, outcome, "A GPU shapes index bounds no key's entries"),
               nullptr;
    usearch_vram_options_t const& vram = arguments.vram;
    if (vram.devices_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A shapes index places one shard"), nullptr;
    std::size_t const ordinal = vram.devices && vram.devices_count ? static_cast<std::size_t>(vram.devices[0]) : 0;
    std::span<gpu_device_t const> const listed = gpu_devices();
    if (ordinal >= listed.size())
        return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), nullptr;
    gpu_device_t const& device = listed[ordinal];
    if (device.vendor != gpu_vendor_t::nvidia_k)
        return propagate_error(usearch_status_device_k, outcome, "No GPU kernels serve shapes on this vendor yet"),
               nullptr;
    bool const hausdorff = arguments.metric == usearch_shapes_metric_hausdorff_k;
    if (arguments.quantization == usearch_scalar_f16_k)
        return hausdorff
                   ? make_shapes_vram_handle<
                         shapes_vram_nvidia_variant<nk::f16_t, us::shapes::f16_ampere_hausdorff_kit>>(arguments, device,
                                                                                                      outcome)
                   : make_shapes_vram_handle<shapes_vram_nvidia_variant<nk::f16_t, us::shapes::f16_ampere_chamfer_kit>>(
                         arguments, device, outcome);
    return hausdorff
               ? make_shapes_vram_handle<shapes_vram_nvidia_variant<nk::f32_t, us::shapes::f32_ampere_hausdorff_kit>>(
                     arguments, device, outcome)
               : make_shapes_vram_handle<shapes_vram_nvidia_variant<nk::f32_t, us::shapes::f32_ampere_chamfer_kit>>(
                     arguments, device, outcome);
}

#endif

/** Reads the kind, point count and connectivity a checkpoint at @p path froze, with the scalar its
 *  points are stored in as @p quantization. */
static shapes_geometry_t checkpoint_geometry(std::string_view path, usearch_scalar_t& quantization,
                                             status_t& status) noexcept {
    shapes_geometry_t geometry;
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
        else if (field.key == "points") geometry.points = decimal_in(field.value);
    };
    auto const [metadata, read_status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
    status = read_status;
    geometry.metric = shapes_metric_of(metadata.metric.view());
    quantization = metadata.value_dtype == dtype_t::f16_k   ? usearch_scalar_f16_k
                   : metadata.value_dtype == dtype_t::f32_k ? usearch_scalar_f32_k
                                                            : usearch_scalar_unknown_k;
    return geometry;
}

/** Whether @p shapes holds @p count entries of points in a scalar the arms cast from, its offsets
 *  counting points; false once @p outcome records why not. */
static bool accepts_shapes(usearch_tape_view_t const& shapes, std::size_t count, usearch_outcome_t* outcome) noexcept {
    if (!shapes.data || !monotone_offsets(shapes, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The shapes need data and non-decreasing offsets");
    nk_dtype_t const dtype = nk_dtype_of(shapes.scalar);
    if (dtype == nk_dtype_unknown_k || nk_dtype_bits(dtype) % 8)
        return propagate_error(usearch_status_unsupported_k, outcome, "Points arrive in a whole-byte scalar");
    return true;
}

#pragma endregion Handles

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_shapes_init_ram(usearch_shapes_init_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    shapes_geometry_t const geometry {arguments->metric, arguments->points, arguments->connectivity};
    usearch_shapes_index_t const index = make_shapes_index<shapes_ram_variant>(geometry, arguments->quantization,
                                                                               arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#if USEARCH_ARCH_CUDA_
USEARCH_EXPORT void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_shapes_index_t const index = make_shapes_vram_index(*arguments, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}
#else
USEARCH_EXPORT void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build compiled no GPU kernels for shapes");
}
#endif

USEARCH_EXPORT void usearch_shapes_load_ram(usearch_shapes_load_ram_t* arguments) {
    if (submitted<usearch_shapes_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    usearch_scalar_t quantization = usearch_scalar_unknown_k;
    status_t read_status = status_t::success_k;
    shapes_geometry_t const geometry = checkpoint_geometry(path, quantization, read_status);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_shapes_index_t const index = make_shapes_index<shapes_ram_variant>(geometry, quantization, arguments->ram,
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

USEARCH_EXPORT void usearch_shapes_load_vram(usearch_shapes_load_vram_t* arguments) {
    if (submitted<usearch_shapes_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "A GPU shapes index loads no checkpoint yet");
}

USEARCH_EXPORT void usearch_shapes_view_ram(usearch_shapes_view_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Shapes own their packs, so a checkpoint loads instead");
}

USEARCH_EXPORT void usearch_shapes_view_nvme(usearch_shapes_view_nvme_t* arguments) {
    if (submitted<usearch_shapes_view_nvme>(arguments)) return;
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
    usearch_scalar_t quantization = usearch_scalar_unknown_k;
    status_t read_status = status_t::success_k;
    shapes_geometry_t const geometry = checkpoint_geometry(path, quantization, read_status);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_shapes_index_t const index = make_shapes_index<shapes_nvme_variant>(geometry, quantization,
                                                                                usearch_ram_options_t {}, outcome);
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

USEARCH_EXPORT void usearch_shapes_free(usearch_shapes_index_t index) {
    finish_calls_using(index);
    delete index;
}

USEARCH_EXPORT void usearch_shapes_reserve(usearch_shapes_reserve_t* arguments) {
    if (submitted<usearch_shapes_reserve>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_writable(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        if constexpr (is_vram_index<arm_type_>) {
            // Past its points, a pack takes a header and a partial alignment line per axis.
            using scalar_t = typename arm_type_::scalar_t;
            std::size_t const pack_overhead = nk::shapes_pack_size<scalar_t>(0, 0).value + 3 * nk_default_alignment_k;
            std::size_t const bytes = arguments->entries * pack_overhead +
                                      arguments->points * 3 * sizeof(typename scalar_t::raw_t);
            status_t reserved =
                arm.index.reserve({arguments->entries, static_cast<connectivity_t>(arm.connectivity), bytes}).status;
            if (succeeded(reserved)) reserved = arm.keys.cover(arguments->entries);
            propagate_error(reserved, outcome);
        }
        else propagate_error(arm.index.reserve(arguments->entries).status, outcome);
    });
}

USEARCH_EXPORT void usearch_shapes_clear(usearch_shapes_clear_t* arguments) {
    if (submitted<usearch_shapes_clear>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_ram_variant(*arguments->index, outcome, [&](auto& arm) noexcept {
        status_t const cleared = arm.index.clear();
        if (succeeded(cleared)) standard_atomic_ref<std::size_t>(arm.total_points).store(0, std::memory_order_relaxed);
        propagate_error(cleared, outcome);
    });
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT void usearch_shapes_describe(usearch_shapes_describe_t* arguments) {
    if (submitted<usearch_shapes_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_shapes_index_t const index = arguments->index;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_shapes_index_description_t& description = *arguments->description;
            description.metadata = {
                arm.metric,
                std::is_same_v<typename arm_type_::scalar_t, nk::f16_t> ? usearch_scalar_f16_k : usearch_scalar_f32_k,
                arm.points, arm.connectivity};
            description.size = arm.index.size();
            description.capacity = arm.index.capacity();
            description.total_points =
                standard_atomic_ref<std::size_t const>(arm.total_points).load(std::memory_order_relaxed);
            description.memory_usage = sizeof(*index) + arm.index.memory_usage();
            description.capabilities = index->capabilities_string;
        },
        index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_shapes_count(usearch_shapes_count_t* arguments) {
    if (submitted<usearch_shapes_count>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    usearch_shapes_key_stats_t& stats = *arguments->key_stats;
    auto count_nvme = [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        auto measure = [&](std::size_t, std::size_t, std::size_t, auto const& cloud) noexcept {
            std::size_t const points = cloud.extent(0);
            std::atomic_ref<std::size_t>(stats.total_points).fetch_add(points, std::memory_order_relaxed);
            std::atomic_ref<std::size_t> largest {stats.largest_points};
            std::size_t seen = largest.load(std::memory_order_relaxed);
            while (seen < points && !largest.compare_exchange_weak(seen, points, std::memory_order_relaxed));
        };
        auto take_count = [&](std::size_t entries) noexcept { return stats.entries = entries, status_t::success_k; };
        propagate_error(values_under(arm.index, arguments->key, *executor, take_count, measure), outcome);
    };
    if (visit_nvme_variant(*arguments->index, count_nvme)) return;
    visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        arm.index.find_each(uuid_of(arguments->key), [&](typename arm_type_::index_t::node_id_t node) noexcept {
            std::size_t const points = arm.index.published_value_of(node).max_points();
            ++stats.entries;
            stats.total_points += points;
            stats.largest_points = max_of(stats.largest_points, points);
        });
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT void usearch_shapes_insert(usearch_shapes_insert_t* arguments) {
    if (submitted<usearch_shapes_insert>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    std::size_t const count = arguments->count;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!arguments->keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must not be null"), void();
    if (arguments->keys.stride && arguments->keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (!accepts_shapes(arguments->shapes, count, outcome)) return;
    visit_writable(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
        usearch_executor_handle_t call_executor;
        typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, count};
        if (crossed(
                arm.insert(arguments->keys, arguments->shapes, count, arguments->expansion_factor, ticks, *executor),
                arguments->stats, outcome))
            propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_shapes_refine(usearch_shapes_refine_t* arguments) {
    if (submitted<usearch_shapes_refine>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency refines yet");
}

USEARCH_EXPORT void usearch_shapes_search(usearch_shapes_search_t* arguments) {
    if (submitted<usearch_shapes_search>(arguments)) return;
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
                                                              {arguments->queries.data}, outcome);
            if (!arena) return;
            if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            if (!accepts_shapes(arguments->queries, count, outcome)) return;
            matches_landing_t landing;
            if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            expected<traversal_telemetry_t> searched {};
            if constexpr (is_vram_index<arm_type_>)
                searched = arm.search(arguments->queries, count, wanted, arguments->expansion_factor, arena->memory,
                                      landing, ticks, *executor);
            else
                searched = arm.search(arguments->queries, count, wanted, arguments->expansion_factor,
                                      arguments->predicate, landing, ticks, *executor);
            if (!crossed(searched, arguments->stats, outcome) || !publish_matches(*arena, count, landing, outcome))
                return;
            *arguments->matches = landing.view();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_shapes_get(usearch_shapes_get_t* arguments) {
    if (submitted<usearch_shapes_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->shapes = {};
    if (arguments->written) *arguments->written = 0;
    nk_dtype_t const exported = nk_dtype_of(arguments->outputs_scalar);
    if (exported == nk_dtype_unknown_k || nk_dtype_bits(exported) % 8)
        return propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    // A view's clouds land in its threads' arenas first, as their sizes are unknown until read.
    auto get_nvme = [&]<typename arm_type_>(arm_type_& arm) noexcept {
        using raw_t = typename arm_type_::scalar_t::raw_t;
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        std::size_t held = 0;
        bytes_view_t* kept = nullptr;
        std::size_t* offsets = nullptr;
        auto carve_entries = [&](std::size_t entries) noexcept {
            held = entries;
            if (!held) return status_t::success_k;
            kept = carve<bytes_view_t>(arena->memory, held);
            offsets = carve<std::size_t>(arena->memory, held + 1);
            return kept && offsets ? status_t::success_k : status_t::bad_alloc_k;
        };
        status_t keeping = status_t::success_k;
        auto keep = [&](std::size_t, std::size_t rank, std::size_t thread, auto const& cloud) noexcept {
            bytes_view_t const bytes = bytes_of(reinterpret_cast<raw_t const*>(cloud.data()), cloud.extent(0) * 3);
            keep_first_failure(keeping, keep_row(*executor, thread, bytes, kept[rank]));
        };
        if (status_t const read = values_under(arm.index, arguments->key, *executor, carve_entries, keep); failed(read))
            return propagate_error(read, outcome), void();
        if (failed(keeping)) return propagate_error(keeping, outcome), void();
        if (!held) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != held; ++entry)
            offsets[entry + 1] = offsets[entry] + kept[entry].size() / (3 * sizeof(raw_t));
        byte_t* const points = carve<byte_t>(arena->memory, offsets[held] * 3 * element_bytes);
        if (offsets[held] && !points)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        for (std::size_t entry = 0; entry != held; ++entry)
            if (status_t const exported_status = arm.export_points(kept[entry], exported,
                                                                   points + offsets[entry] * 3 * element_bytes);
                failed(exported_status))
                return propagate_error(exported_status, outcome), void();
        *arguments->shapes = {points, offsets, arguments->outputs_scalar};
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
        thread_allocator_t const scratch {executor->arena(0)};
        auto const ids = ids_under(arm.index, arguments->key, scratch);
        if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
        std::size_t* const offsets = carve<std::size_t>(arena->memory, ids.size() + 1);
        if (!offsets)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != ids.size(); ++entry)
            offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(ids[entry]).max_points();
        byte_t* const points = carve<byte_t>(arena->memory, offsets[ids.size()] * 3 * element_bytes);
        if (offsets[ids.size()] && !points)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        for (std::size_t entry = 0; entry != ids.size(); ++entry)
            if (status_t const exported_status = arm.export_points(arm.index.published_value_of(ids[entry]), exported,
                                                                   points + offsets[entry] * 3 * element_bytes,
                                                                   scratch);
                failed(exported_status))
                return propagate_error(exported_status, outcome), void();
        *arguments->shapes = {points, offsets, arguments->outputs_scalar};
        if (arguments->written) *arguments->written = ids.size();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Insertion and Search

#pragma region Cursors and Absorbing

USEARCH_EXPORT void usearch_shapes_cursor_init(usearch_shapes_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    usearch_shapes_cursor_t const cursor = new (std::nothrow) usearch_shapes_cursor_handle_t {};
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

USEARCH_EXPORT void usearch_shapes_cursor_next(usearch_shapes_cursor_next_t* arguments) {
    if (submitted<usearch_shapes_cursor_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->shapes) *arguments->shapes = {};
    *arguments->written = 0;
    usearch_shapes_cursor_t const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    nk_dtype_t const exported = nk_dtype_of(arguments->outputs_scalar);
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    // A view reads its rows through the executor, landing clouds in its threads' arenas first.
    auto next_nvme = [&]<typename arm_type_>(arm_type_& arm) noexcept {
        using raw_t = typename arm_type_::scalar_t::raw_t;
        std::size_t* const row = std::get_if<std::size_t>(&cursor->cursor);
        if (!row) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
        usearch_executor_handle_t call_executor;
        nvme_executor_t* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
        if (!executor) return;
        std::size_t const taking = min_of(arm.index.size() - *row, arguments->wanted);
        if (!taking) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
        bytes_view_t* const kept = carve<bytes_view_t>(arena->memory, taking);
        std::size_t* const offsets = arguments->shapes ? carve<std::size_t>(arena->memory, taking + 1) : nullptr;
        if (!keys || !kept || (arguments->shapes && !offsets))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        status_t keeping = status_t::success_k;
        auto keep = [&](std::size_t entry, std::size_t thread, auto const& cloud) noexcept {
            bytes_view_t const bytes = bytes_of(reinterpret_cast<raw_t const*>(cloud.data()), cloud.extent(0) * 3);
            if (offsets) keep_first_failure(keeping, keep_row(*executor, thread, bytes, kept[entry]));
        };
        std::span<us::utilities::uuid_t> const identities {reinterpret_cast<us::utilities::uuid_t*>(keys), taking};
        if (status_t const read = arm.index.read_rows(*row, identities, *executor, keep); failed(read))
            return propagate_error(read, outcome), void();
        if (failed(keeping)) return propagate_error(keeping, outcome), void();
        if (offsets) {
            std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
            offsets[0] = 0;
            for (std::size_t entry = 0; entry != taking; ++entry)
                offsets[entry + 1] = offsets[entry] + kept[entry].size() / (3 * sizeof(raw_t));
            byte_t* const points = carve<byte_t>(arena->memory, offsets[taking] * 3 * element_bytes);
            if (offsets[taking] && !points)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            for (std::size_t entry = 0; entry != taking; ++entry)
                if (status_t const exported_status = arm.export_points(kept[entry], exported,
                                                                       points + offsets[entry] * 3 * element_bytes);
                    failed(exported_status))
                    return propagate_error(exported_status, outcome), void();
            *arguments->shapes = {points, offsets, arguments->outputs_scalar};
        }
        *row += taking;
        *arguments->keys = {keys, sizeof(usearch_key_t)};
        *arguments->written = taking;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    };
    if (arena && arguments->shapes && (exported == nk_dtype_unknown_k || nk_dtype_bits(exported) % 8))
        propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar");
    else if (arena && !visit_nvme_variant(*arguments->index, next_nvme))
        visit_ram_variant(*arguments->index, outcome, [&]<typename arm_type_>(arm_type_& arm) noexcept {
            using cursor_t =
                basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;
            cursor_t* const typed = std::get_if<cursor_t>(&cursor->cursor);
            if (!typed) return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            if (status_t const rewound = executor->reserve(0); failed(rewound))
                return propagate_error(rewound, outcome), void();
            thread_allocator_t const scratch {executor->arena(0)};
            std::size_t const remaining = *typed ? typed->visible_count() - typed->id() : 0;
            std::size_t const taking = min_of(remaining, arguments->wanted);
            usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
            std::size_t* const offsets = arguments->shapes ? carve<std::size_t>(arena->memory, taking + 1) : nullptr;
            if ((taking && !keys) || (arguments->shapes && !offsets))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            typename arm_type_::index_t::node_id_t const first = typed->id();
            if (offsets) offsets[0] = 0;
            for (std::size_t entry = 0; entry != taking; ++entry) {
                auto const node = static_cast<typename arm_type_::index_t::node_id_t>(first + entry);
                keys[entry] = std::bit_cast<usearch_key_t>(arm.index.key_of(node).value);
                if (offsets) offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(node).max_points();
            }
            if (offsets) {
                std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
                byte_t* const points = carve<byte_t>(arena->memory, offsets[taking] * 3 * element_bytes);
                if (offsets[taking] && !points)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                for (std::size_t entry = 0; entry != taking; ++entry) {
                    auto const node = static_cast<typename arm_type_::index_t::node_id_t>(first + entry);
                    if (status_t const exported_status = arm.export_points(arm.index.published_value_of(node), exported,
                                                                           points + offsets[entry] * 3 * element_bytes,
                                                                           scratch);
                        failed(exported_status))
                        return propagate_error(exported_status, outcome), void();
                }
                *arguments->shapes = {points, offsets, arguments->outputs_scalar};
            }
            for (std::size_t step = 0; step != taking; ++step) ++*typed;
            *arguments->keys = {keys, sizeof(usearch_key_t)};
            *arguments->written = taking;
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
    cursor->busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_shapes_cursor_free(usearch_shapes_cursor_t cursor) {
    finish_calls_using(cursor);
    delete cursor;
}

USEARCH_EXPORT void usearch_shapes_sample(usearch_shapes_sample_t* arguments) {
    if (submitted<usearch_shapes_sample>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency samples yet");
}

USEARCH_EXPORT void usearch_shapes_absorb(usearch_shapes_absorb_t* arguments) {
    if (submitted<usearch_shapes_absorb>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency absorbs donor handles yet");
}

#pragma endregion Cursors and Absorbing

#pragma region Serialization

USEARCH_EXPORT void usearch_shapes_save(usearch_shapes_save_t* arguments) {
    if (submitted<usearch_shapes_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    if (path.ends_with(".json"))
        return propagate_error(usearch_status_unsupported_k, outcome, "A shapes index saves one monolith"), void();
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

USEARCH_EXPORT void usearch_shapes_metadata_read(usearch_shapes_metadata_read_t* arguments) {
    if (submitted<usearch_shapes_metadata_read>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_shapes_metadata_t& metadata = *arguments->metadata;
    metadata = {};
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") metadata.connectivity = decimal_in(field.value);
        else if (field.key == "points") metadata.points = decimal_in(field.value);
    };
    auto const [read, status] = safegraphs_read_index_metadata(as_text(arguments->path), std::allocator<byte_t> {},
                                                               per_field);
    if (failed(status)) return propagate_error(status, outcome), void();
    metadata.metric = shapes_metric_of(read.metric.view());
    metadata.quantization = read.value_dtype == dtype_t::f16_k   ? usearch_scalar_f16_k
                            : read.value_dtype == dtype_t::f32_k ? usearch_scalar_f32_k
                                                                 : usearch_scalar_unknown_k;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

} // extern "C"
