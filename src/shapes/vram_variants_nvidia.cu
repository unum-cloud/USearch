/**
 *  @file src/shapes/vram_variants_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the shapes kits: Chamfer and Hausdorff need nothing past the Ampere
 *      floor, so one unit carries a cubin for every NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only, over every scalar and both node-ID types, into the unit's
 *  fatbin, and by Clang host-only, where it defines the faces' members reading the kits and names
 *  that fatbin, linked in beside it, to the host indexes.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_nvidia_metric.cuh"

/** The unit's fatbin, linked in as raw bytes, which every shapes index loads. */
extern "C" byte_t const shapes_nvidia_bundle_start[], shapes_nvidia_bundle_end[];

bytes_view_t shapes_nvidia_fatbin_t::bytes() noexcept {
    return {shapes_nvidia_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_bundle_end - shapes_nvidia_bundle_start)};
}

template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t, shapes_nvidia_capabilities_k, shapes_nvidia_fatbin_t>;
template struct us::shapes::vram_nvidia_index<nk::f32_t, u32_t, shapes_nvidia_capabilities_k, shapes_nvidia_fatbin_t>;
template struct us::shapes::vram_nvidia_kits<nk::f16_t, u32_t, shapes_nvidia_capabilities_k, shapes_nvidia_fatbin_t>;
template struct us::shapes::vram_nvidia_kits<nk::f32_t, u32_t, shapes_nvidia_capabilities_k, shapes_nvidia_fatbin_t>;

template __device__ void us::shapes::cuda_instantiate<shapes_nvidia_capabilities_k, nk_f16_k, u32_t>();
template __device__ void us::shapes::cuda_instantiate<shapes_nvidia_capabilities_k, nk_f32_k, u32_t>();
