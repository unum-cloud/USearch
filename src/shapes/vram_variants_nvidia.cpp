/**
 *  @file src/shapes/vram_variants_nvidia.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the shapes kits: Chamfer and Hausdorff need nothing past the Ampere
 *      floor, so one unit serves every NVIDIA class, compiled as the Ampere floor or as a
 *      monoversioned build's one class.
 *
 *  Clang compiles it host-only, where it defines the faces' members reading the kits and names each
 *  dtype's fatbin, linked in beside it; NVCC compiles every @c vram_kernels_nvidia_<dtype>.cu
 *  device-only into those fatbins.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_nvidia_metric.cuh"

/** The bounds of the f16 then the f32 fatbin, linked in as raw bytes, from the table CMake writes
 *  for the class this unit compiles as; the other classes' tables stay unreferenced. */
extern "C" byte_t const* const shapes_nvidia_ampere_bundles[];
extern "C" byte_t const* const shapes_nvidia_ada_bundles[];
extern "C" byte_t const* const shapes_nvidia_hopper_bundles[];
extern "C" byte_t const* const shapes_nvidia_blackwell_bundles[];
extern "C" byte_t const* const shapes_nvidia_blackwellrtx_bundles[];
template <gpu_class_t gpu_class_>
byte_t const* const* shapes_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::nvidia_ada_k) return shapes_nvidia_ada_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_hopper_k) return shapes_nvidia_hopper_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_blackwell_k) return shapes_nvidia_blackwell_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::nvidia_blackwellrtx_k) return shapes_nvidia_blackwellrtx_bundles;
    else return shapes_nvidia_ampere_bundles;
}

template <>
bytes_view_t shapes_nvidia_fatbin<cuda_class_k, nk_f16_k>::bytes() noexcept {
    byte_t const* const* const bounds = shapes_bundles<cuda_class_k>();
    return {bounds[0], bounds[1]};
}
template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t, cuda_class_capabilities_k,
                                              shapes_nvidia_fatbin<cuda_class_k, nk_f16_k>>;
template struct us::shapes::vram_nvidia_kits<nk::f16_t, u32_t, cuda_class_capabilities_k,
                                             shapes_nvidia_fatbin<cuda_class_k, nk_f16_k>>;

template <>
bytes_view_t shapes_nvidia_fatbin<cuda_class_k, nk_f32_k>::bytes() noexcept {
    byte_t const* const* const bounds = shapes_bundles<cuda_class_k>();
    return {bounds[2], bounds[3]};
}
template struct us::shapes::vram_nvidia_index<nk::f32_t, u32_t, cuda_class_capabilities_k,
                                              shapes_nvidia_fatbin<cuda_class_k, nk_f32_k>>;
template struct us::shapes::vram_nvidia_kits<nk::f32_t, u32_t, cuda_class_capabilities_k,
                                             shapes_nvidia_fatbin<cuda_class_k, nk_f32_k>>;

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::vram_nvidia_k, cuda_class_k>;
