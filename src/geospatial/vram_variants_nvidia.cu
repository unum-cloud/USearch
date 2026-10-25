/**
 *  @file src/geospatial/vram_variants_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the geospatial kit: squared chords need nothing past the Ampere floor,
 *      so one unit carries a cubin for every NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only, over both node-ID types and the kernel judging the walks'
 *  matches, into the unit's fatbin, and by Clang host-only, where it defines the face's members
 *  reading the kit and names that fatbin, linked in beside it, to the host index.
 */
#include "geospatial_specializations.cuh"
#include "usearch/geospatial/vram_nvidia_metric.cuh"

/** The unit's fatbin, linked in as raw bytes. */
extern "C" byte_t const geospatial_nvidia_bundle_start[], geospatial_nvidia_bundle_end[];

bytes_view_t geospatial_nvidia_fatbin_t::bytes() noexcept {
    return {geospatial_nvidia_bundle_start,
            static_cast<std::size_t>(geospatial_nvidia_bundle_end - geospatial_nvidia_bundle_start)};
}

template struct us::geospatial::vram_nvidia_index<u32_t, geospatial_nvidia_capabilities_k, geospatial_nvidia_fatbin_t>;
template struct us::geospatial::vram_nvidia_kits<u32_t, geospatial_nvidia_capabilities_k, geospatial_nvidia_fatbin_t>;

template __device__ void us::geospatial::cuda_instantiate<geospatial_nvidia_capabilities_k, u32_t>();

template struct basic_class_tables<geospatial_traits_t, dense_home_t::vram_nvidia_k, gpu_class_t::nvidia_ampere_k>;
