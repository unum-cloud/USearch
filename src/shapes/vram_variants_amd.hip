/**
 *  @file src/shapes/vram_variants_amd.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The AMD unit of the shapes kits: Chamfer and Hausdorff need nothing past each class's
 *      wavefront and scans, so one unit serves every AMD class, compiled once per class it is kept
 *      for, under that class's ceiling.
 *
 *  HIP-Clang compiles it host-only, where it defines the faces' members reading the kits and names
 *  each dtype's code object, linked in beside it; HIP-Clang compiles every
 *  @c vram_kernels_amd_<dtype>.hip device-only into those code objects, once per class.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_amd_metric.hip"

/** The bounds of the f16 then the f32 code object, linked in as raw bytes, from the table CMake
 *  writes for the class this unit compiles as; the other classes' tables stay unreferenced. */
extern "C" byte_t const* const shapes_amd_cdna3_bundles[];
extern "C" byte_t const* const shapes_amd_cdna4_bundles[];
extern "C" byte_t const* const shapes_amd_cdna5_bundles[];
template <gpu_class_t gpu_class_>
byte_t const* const* shapes_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::amd_cdna5_k) return shapes_amd_cdna5_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::amd_cdna4_k) return shapes_amd_cdna4_bundles;
    else return shapes_amd_cdna3_bundles;
}

template <>
bytes_view_t shapes_amd_code_object<hip_class_k, nk_f16_k>::bytes() noexcept {
    byte_t const* const* const bounds = shapes_bundles<hip_class_k>();
    return {bounds[0], bounds[1]};
}
template struct us::shapes::vram_amd_index<
    typename us::shapes::shapes_ladder<nk::f16_t, numkong_capabilities_of(hip_class_k)>::kit_t, u32_t,
    shapes_amd_code_object<hip_class_k, nk_f16_k>>;
template struct us::shapes::vram_amd_kits<
    typename us::shapes::shapes_ladder<nk::f16_t, numkong_capabilities_of(hip_class_k)>::kit_t, u32_t,
    shapes_amd_code_object<hip_class_k, nk_f16_k>>;

template <>
bytes_view_t shapes_amd_code_object<hip_class_k, nk_f32_k>::bytes() noexcept {
    byte_t const* const* const bounds = shapes_bundles<hip_class_k>();
    return {bounds[2], bounds[3]};
}
template struct us::shapes::vram_amd_index<
    typename us::shapes::shapes_ladder<nk::f32_t, numkong_capabilities_of(hip_class_k)>::kit_t, u32_t,
    shapes_amd_code_object<hip_class_k, nk_f32_k>>;
template struct us::shapes::vram_amd_kits<
    typename us::shapes::shapes_ladder<nk::f32_t, numkong_capabilities_of(hip_class_k)>::kit_t, u32_t,
    shapes_amd_code_object<hip_class_k, nk_f32_k>>;

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::vram_amd_k, hip_class_k>;
