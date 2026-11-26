/**
 *  @file src/geospatial/vram_variants_amd.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The AMD unit of the geospatial kit: squared chords need nothing past each class's
 *      wavefront and scans, so one unit serves every AMD class, compiled once per class it is kept
 *      for, under that class's ceiling.
 *
 *  Compiled twice per class: by HIP-Clang device-only, over 32-bit node IDs and the kernels judging
 *  the walks' matches, into the class's code object, and host-only, where it defines the face's
 *  members reading the kit and names that code object, linked in beside it as raw bytes, to the
 *  host index. A kernel's address is taken in a host function, which only the device pass may see,
 *  or the host pass would emit launch stubs.
 */
#include "geospatial_specializations.cuh"
#include "usearch/geospatial/vram_amd_metric.hip"

#if __HIP_DEVICE_COMPILE__

template void us::geospatial::amd_instantiate<geospatial_vram_kit<hip_class_k>, util::u32_t>();

#else

/** The bounds of the unit's code object, linked in as raw bytes, from the table CMake writes for
 *  the class this unit compiles as; the other classes' tables stay unreferenced. */
extern "C" util::byte_t const* const geospatial_amd_cdna3_bundles[];
extern "C" util::byte_t const* const geospatial_amd_cdna4_bundles[];
extern "C" util::byte_t const* const geospatial_amd_cdna5_bundles[];
template <gpu_class_t gpu_class_>
util::byte_t const* const* geospatial_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::amd_cdna5_k) return geospatial_amd_cdna5_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::amd_cdna4_k) return geospatial_amd_cdna4_bundles;
    else return geospatial_amd_cdna3_bundles;
}

template <>
util::bytes_view_t geospatial_amd_code_object<hip_class_k>::bytes() noexcept {
    util::byte_t const* const* const bounds = geospatial_bundles<hip_class_k>();
    return {bounds[0], bounds[1]};
}

template struct us::geospatial::vram_amd_index<geospatial_vram_kit<hip_class_k>, util::u32_t,
                                               geospatial_amd_code_object<hip_class_k>>;
template struct us::geospatial::vram_amd_kits<geospatial_vram_kit<hip_class_k>, util::u32_t,
                                              geospatial_amd_code_object<hip_class_k>>;

template struct basic_class_tables<geospatial_traits_t, vendored_residency_t::vram_amd_k, hip_class_k>;

#endif
