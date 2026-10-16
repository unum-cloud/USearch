/**
 *  @file src/shapes_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The shapes modality's C surface: the variant over every compiled shapes alternative,
 *      the traits its verbs differ in, and every exported entry point as one forward apiece.
 *
 *  Variants are ordered newest class first inside each scalar, the serial floor last. Their
 *  bodies compile in the basic class units, never here: the @c extern declarations below keep
 *  them out, as the dense dispatch unit keeps its own.
 */
#include <string_view> // `std::string_view` - checkpoint paths and fields
#include <variant>     // `std::variant`

#include "basic_index.cuh"
#include "shapes_specializations.cuh"

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

/** Every shapes variant this build compiled: the basic index over one scalar's packs scored with
 *  one class's kits, a read-only view of a checkpoint on storage scored with the same, or the
 *  basic index on a GPU. */
using shapes_variant_t = std::variant< //
#if USEARCH_ARCH_CUDA_
    shapes_vram_nvidia_variant<nk::f16_t, nk_kernel_chamfers_packed_k>,   //
    shapes_vram_nvidia_variant<nk::f16_t, nk_kernel_hausdorffs_packed_k>, //
    shapes_vram_nvidia_variant<nk::f32_t, nk_kernel_chamfers_packed_k>,   //
    shapes_vram_nvidia_variant<nk::f32_t, nk_kernel_hausdorffs_packed_k>, //
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

/** Whether @p shapes holds @p count entries of points in a scalar the variants cast from, its
 *  offsets counting points; false once @p outcome records why not. */
static bool accepts_shapes(usearch_tape_view_t const& shapes, std::size_t count, usearch_outcome_t* outcome) noexcept {
    if (!shapes.data || !monotone_offsets(shapes, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The shapes need data and non-decreasing offsets");
    nk_dtype_t const dtype = nk_dtype_of(shapes.scalar);
    if (dtype == nk_dtype_unknown_k || nk_dtype_bits(dtype) % 8)
        return propagate_error(usearch_status_unsupported_k, outcome, "Points arrive in a whole-byte scalar");
    return true;
}

/** What the shapes verbs differ in: clouds of 3D points stored in one scalar, crossing in any. */
struct shapes_traits_t {
    using geometry_t = shapes_geometry_t;

    /** What a view keeps of a cloud it read: its points as saved, in its thread's arena. */
    using kept_t = bytes_view_t;

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

    /** Refuses the kinds and scalars no device kit serves. */
    static bool admits(geometry_t const& geometry, usearch_shapes_init_vram_t const&,
                       usearch_outcome_t* outcome) noexcept {
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

    /** Whether a donor of @p donor geometry lands in an index of @p index geometry: the same kind,
     *  scalar, point count and connectivity. */
    static bool matches(geometry_t const& index, geometry_t const& donor) noexcept {
        return index.metric == donor.metric && index.quantization == donor.quantization &&
               index.points == donor.points && index.connectivity == donor.connectivity;
    }

    /** Copies @p geometry into @p variant, a zero connectivity taking the default. */
    template <typename variant_type_, typename arguments_type_>
    static bool freeze(variant_type_& variant, geometry_t& geometry, arguments_type_ const&,
                       usearch_outcome_t*) noexcept {
        if (!geometry.connectivity) geometry.connectivity = connectivity_32_k;
        variant.geometry = geometry;
        return true;
    }

#if USEARCH_ARCH_CUDA_

    /** Calls @p make with the GPU variant over the stored scalar under the kind's kernel. */
    template <typename make_type_>
    static void with_vram_variant(geometry_t const& geometry, make_type_&& make) noexcept {
        bool const hausdorff = geometry.metric == usearch_shapes_metric_hausdorff_k;
        if (geometry.quantization == usearch_scalar_f16_k) {
            if (hausdorff)
                make(std::type_identity<shapes_vram_nvidia_variant<nk::f16_t, nk_kernel_hausdorffs_packed_k>> {});
            else make(std::type_identity<shapes_vram_nvidia_variant<nk::f16_t, nk_kernel_chamfers_packed_k>> {});
        }
        else if (hausdorff)
            make(std::type_identity<shapes_vram_nvidia_variant<nk::f32_t, nk_kernel_hausdorffs_packed_k>> {});
        else make(std::type_identity<shapes_vram_nvidia_variant<nk::f32_t, nk_kernel_chamfers_packed_k>> {});
    }
#endif

    static usearch_shapes_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.metric, geometry.quantization, geometry.points, geometry.connectivity};
    }

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_shapes_index_description_t& description) noexcept {
        description.metadata = metadata_of(variant.geometry);
        description.total_points =
            standard_atomic_ref<std::size_t const>(variant.total_points).load(std::memory_order_relaxed);
    }

    static bool accepts(usearch_shapes_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_shapes(arguments.shapes, arguments.count, outcome);
    }

    template <typename variant_type_>
    static bool accepts(variant_type_ const&, usearch_shapes_search_t const& arguments,
                        usearch_outcome_t* outcome) noexcept {
        return accepts_shapes(arguments.queries, arguments.count, outcome);
    }

    /** Whether the points a get or a pull asks for export to a whole-byte scalar; false once
     *  @p outcome records why not. */
    template <typename arguments_type_>
    static bool exports(arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        nk_dtype_t const exported = nk_dtype_of(arguments.outputs_scalar);
        if (!arguments.shapes || (exported != nk_dtype_unknown_k && nk_dtype_bits(exported) % 8 == 0)) return true;
        return propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar");
    }

    /** Where a get or a pull hands its clouds out, null when a pull wants none. */
    template <typename arguments_type_>
    static usearch_tape_view_t* output_of(arguments_type_ const& arguments) noexcept {
        return arguments.shapes;
    }

    template <typename cloud_type_>
    static status_t keep(nvme_executor_t& executor, std::size_t thread, cloud_type_ const& cloud,
                         kept_t& kept) noexcept {
        return keep_row(executor, thread, bytes_of(cloud.data(), cloud.extent(0) * 3), kept);
    }

    /** Lands @p count clouds, the @c i th as @p value_at hands it, as points in the scalar the call
     *  asks for in @p memory, their offsets counting points; a variant's own values are packs,
     *  which unpack through @p memory too where the scalar is not the stored one. */
    template <typename variant_type_, typename value_at_type_, typename arguments_type_>
    static bool land(variant_type_ const& variant, std::size_t count, value_at_type_&& value_at,
                     growing_arena<forkunion_allocator_t>& memory, arguments_type_ const& arguments,
                     usearch_outcome_t* outcome) noexcept {
        using raw_t = typename variant_type_::scalar_t::raw_t;
        nk_dtype_t const exported = nk_dtype_of(arguments.outputs_scalar);
        std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
        std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
        if (!offsets) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != count; ++entry)
            if constexpr (is_nvme_index<variant_type_>)
                offsets[entry + 1] = offsets[entry] + value_at(entry).size() / (3 * sizeof(raw_t));
            else offsets[entry + 1] = offsets[entry] + value_at(entry).max_points();
        byte_t* const points = carve<byte_t>(memory, offsets[count] * 3 * element_bytes);
        if (offsets[count] && !points)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        thread_allocator_t const scratch {memory};
        for (std::size_t entry = 0; entry != count; ++entry) {
            byte_t* const output = points + offsets[entry] * 3 * element_bytes;
            status_t exported_status;
            if constexpr (is_nvme_index<variant_type_>)
                exported_status = variant.export_points(value_at(entry), exported, output);
            else exported_status = variant.export_points(value_at(entry), exported, output, scratch);
            if (failed(exported_status)) return propagate_error(exported_status, outcome);
        }
        *arguments.shapes = {points, offsets, arguments.outputs_scalar};
        return true;
    }

    /** Adds one entry's points to a key's statistics, from any thread: a variant's packs count
     *  their points, and a view's clouds hold a row per point. */
    template <typename cloud_type_>
    static void tally(usearch_shapes_key_stats_t& stats, cloud_type_ const& cloud) noexcept {
        std::size_t points = 0;
        if constexpr (requires { cloud.max_points(); }) points = cloud.max_points();
        else points = cloud.extent(0);
        standard_atomic_ref<std::size_t>(stats.total_points).fetch_add(points, std::memory_order_relaxed);
        raise_to(stats.largest_points, points);
    }
};

struct usearch_shapes_index_handle_t : basic_handle<shapes_variant_t, shapes_traits_t> {
    using basic_handle::basic_handle;
};

struct usearch_shapes_cursor_handle_t
    : basic_cursor<usearch_shapes_index_handle_t, ram_cursor_of<shapes_ram_variant<nk::f32_t, cpu_class_t::serial_k>>,
                   ram_cursor_of<shapes_ram_variant<nk::f16_t, cpu_class_t::serial_k>>> {};

struct usearch_shapes_refinement_handle_t : basic_refinement<usearch_shapes_index_handle_t> {};

#pragma endregion Handles

extern "C" {

USEARCH_EXPORT void usearch_shapes_init_ram(usearch_shapes_init_ram_t* arguments) { basic_init_ram(arguments); }
USEARCH_EXPORT void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments) { basic_init_vram(arguments); }
USEARCH_EXPORT void usearch_shapes_load_ram(usearch_shapes_load_ram_t* arguments) {
    basic_load_ram<usearch_shapes_load_ram>(arguments);
}
USEARCH_EXPORT void usearch_shapes_load_vram(usearch_shapes_load_vram_t* arguments) {
    basic_unserved<usearch_shapes_load_vram>(arguments);
}
USEARCH_EXPORT void usearch_shapes_view_ram(usearch_shapes_view_ram_t* arguments) {
    basic_unserved<usearch_shapes_view_ram>(arguments);
}
USEARCH_EXPORT void usearch_shapes_view_nvme(usearch_shapes_view_nvme_t* arguments) {
    basic_view_nvme<usearch_shapes_view_nvme>(arguments);
}
USEARCH_EXPORT void usearch_shapes_free(usearch_shapes_index_t index) { basic_free(index); }
USEARCH_EXPORT void usearch_shapes_reserve(usearch_shapes_reserve_t* arguments) {
    basic_reserve<usearch_shapes_reserve>(arguments);
}
USEARCH_EXPORT void usearch_shapes_clear(usearch_shapes_clear_t* arguments) {
    basic_clear<usearch_shapes_clear>(arguments);
}
USEARCH_EXPORT void usearch_shapes_describe(usearch_shapes_describe_t* arguments) {
    basic_describe<usearch_shapes_describe>(arguments);
}
USEARCH_EXPORT void usearch_shapes_count(usearch_shapes_count_t* arguments) {
    basic_count<usearch_shapes_count>(arguments);
}
USEARCH_EXPORT void usearch_shapes_insert(usearch_shapes_insert_t* arguments) {
    basic_insert<usearch_shapes_insert>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refine(usearch_shapes_refine_t* arguments) {
    basic_refine<usearch_shapes_refine>(arguments);
}
USEARCH_EXPORT void usearch_shapes_search(usearch_shapes_search_t* arguments) {
    basic_search<usearch_shapes_search>(arguments);
}
USEARCH_EXPORT void usearch_shapes_get(usearch_shapes_get_t* arguments) { basic_get<usearch_shapes_get>(arguments); }
USEARCH_EXPORT void usearch_shapes_cursor_init(usearch_shapes_cursor_init_t* arguments) {
    basic_cursor_init(arguments);
}
USEARCH_EXPORT void usearch_shapes_cursor_next(usearch_shapes_cursor_next_t* arguments) {
    basic_cursor_next<usearch_shapes_cursor_next>(arguments);
}
USEARCH_EXPORT void usearch_shapes_cursor_free(usearch_shapes_cursor_t cursor) { basic_free(cursor); }
USEARCH_EXPORT void usearch_shapes_sample(usearch_shapes_sample_t* arguments) {
    basic_unserved<usearch_shapes_sample>(arguments);
}
USEARCH_EXPORT void usearch_shapes_absorb(usearch_shapes_absorb_t* arguments) {
    basic_absorb<usearch_shapes_absorb>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refinement_init(usearch_shapes_refinement_init_t* arguments) {
    basic_refinement_init<usearch_shapes_refinement_init>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refinement_next(usearch_shapes_refinement_next_t* arguments) {
    basic_refinement_next<usearch_shapes_refinement_next>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refinement_finish(usearch_shapes_refinement_finish_t* arguments) {
    basic_refinement_finish<usearch_shapes_refinement_finish>(arguments);
}
USEARCH_EXPORT void usearch_shapes_save(usearch_shapes_save_t* arguments) {
    basic_save<usearch_shapes_save>(arguments);
}
USEARCH_EXPORT void usearch_shapes_metadata_read(usearch_shapes_metadata_read_t* arguments) {
    basic_metadata_read<usearch_shapes_metadata_read, shapes_traits_t>(arguments);
}

} // extern "C"
