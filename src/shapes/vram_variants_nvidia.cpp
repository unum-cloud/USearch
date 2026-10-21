/**
 *  @file src/shapes/vram_variants_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the shapes kits: Chamfer and Hausdorff need nothing past the Ampere
 *      floor, so one unit carries a cubin for every NVIDIA class.
 *
 *  Clang compiles it host-only, where it defines the faces' members reading the kits and names each
 *  dtype's fatbin, linked in beside it; NVCC compiles every @c vram_kernels_nvidia_<dtype>.cu
 *  device-only into those fatbins.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_nvidia_metric.cuh"

/** Each dtype's fatbin, linked in as raw bytes beside the variants. */
extern "C" byte_t const shapes_nvidia_f16_bundle_start[], shapes_nvidia_f16_bundle_end[];
template <>
bytes_view_t shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f16_k>::bytes() noexcept {
    return {shapes_nvidia_f16_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_f16_bundle_end - shapes_nvidia_f16_bundle_start)};
}
template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t, numkong_capabilities_of(gpu_class_t::nvidia_ampere_k),
                                              shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f16_k>>;
template struct us::shapes::vram_nvidia_kits<nk::f16_t, u32_t, numkong_capabilities_of(gpu_class_t::nvidia_ampere_k),
                                             shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f16_k>>;

extern "C" byte_t const shapes_nvidia_f32_bundle_start[], shapes_nvidia_f32_bundle_end[];
template <>
bytes_view_t shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f32_k>::bytes() noexcept {
    return {shapes_nvidia_f32_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_f32_bundle_end - shapes_nvidia_f32_bundle_start)};
}
template struct us::shapes::vram_nvidia_index<nk::f32_t, u32_t, numkong_capabilities_of(gpu_class_t::nvidia_ampere_k),
                                              shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f32_k>>;
template struct us::shapes::vram_nvidia_kits<nk::f32_t, u32_t, numkong_capabilities_of(gpu_class_t::nvidia_ampere_k),
                                             shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f32_k>>;
