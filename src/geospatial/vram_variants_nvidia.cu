/**
 *  @file src/geospatial/vram_variants_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the geospatial kit: squared chords need nothing past the Ampere floor,
 *      so one unit serves every NVIDIA class, compiled as the Ampere floor or as a monoversioned
 *      build's one class.
 *
 *  Compiled twice: by NVCC device-only, over both node-ID types and the kernel judging the walks'
 *  matches, into the unit's fatbin, and by Clang host-only, where it defines the face's members
 *  reading the kit and names that fatbin, linked in beside it, to the host index.
 */
#include "geospatial_specializations.cuh"
#include "usearch/geospatial/vram_nvidia_metric.cuh"

/** The bounds of the unit's fatbin, linked in as raw bytes, from the table CMake writes
 *  for the class this unit compiles as; the other classes' tables stay unreferenced. */
extern "C" byte_t const* const geospatial_nvidia_ampere_bundles[];
extern "C" byte_t const* const geospatial_nvidia_ada_bundles[];
extern "C" byte_t const* const geospatial_nvidia_hopper_bundles[];
extern "C" byte_t const* const geospatial_nvidia_blackwell_bundles[];
extern "C" byte_t const* const geospatial_nvidia_blackwellrtx_bundles[];
template <gpu_class_t gpu_class_>
byte_t const* const* geospatial_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::nvidia_ada_k) return geospatial_nvidia_ada_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_hopper_k) return geospatial_nvidia_hopper_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_blackwell_k) return geospatial_nvidia_blackwell_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_blackwellrtx_k) return geospatial_nvidia_blackwellrtx_bundles;
    else return geospatial_nvidia_ampere_bundles;
}

bytes_view_t geospatial_nvidia_fatbin_t::bytes() noexcept {
    byte_t const* const* const bounds = geospatial_bundles<cuda_class_k>();
    return {bounds[0], bounds[1]};
}

template struct us::geospatial::vram_nvidia_index<
    typename us::geospatial::geospatial_ladder<geospatial_nvidia_capabilities_k>::kit_t, u32_t,
    geospatial_nvidia_fatbin_t>;
template struct us::geospatial::vram_nvidia_kits<
    typename us::geospatial::geospatial_ladder<geospatial_nvidia_capabilities_k>::kit_t, u32_t,
    geospatial_nvidia_fatbin_t>;

template __device__ void
us::geospatial::cuda_instantiate<us::geospatial::geospatial_ladder<geospatial_nvidia_capabilities_k>::kit_t, u32_t>();

template struct basic_class_tables<geospatial_traits_t, vendored_residency_t::vram_nvidia_k, cuda_class_k>;
