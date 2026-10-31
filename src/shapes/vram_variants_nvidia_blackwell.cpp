/**
 *  @file src/shapes/vram_variants_nvidia_blackwell.cpp
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief The nvidia_blackwell shapes variants: f16 packs scored through packed f32x2 math, so
 *      their device unit compiles for "100f" alone.
 *
 *  Clang compiles it host-only, where it defines the faces' members reading the kits and names the
 *  f16 fatbin, linked in beside it; NVCC compiles @c vram_kernels_nvidia_f16.cu device-only into
 *  that fatbin. f32 packs keep the vendor-wide unit's Ampere kit.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_nvidia_metric.cuh"

extern "C" byte_t const shapes_nvidia_blackwell_f16_bundle_start[], shapes_nvidia_blackwell_f16_bundle_end[];
template <>
bytes_view_t shapes_nvidia_fatbin<gpu_class_t::nvidia_blackwell_k, nk_f16_k>::bytes() noexcept {
    return {
        shapes_nvidia_blackwell_f16_bundle_start,
        static_cast<std::size_t>(shapes_nvidia_blackwell_f16_bundle_end - shapes_nvidia_blackwell_f16_bundle_start)};
}
template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t,
                                              numkong_capabilities_of(gpu_class_t::nvidia_blackwell_k),
                                              shapes_nvidia_fatbin<gpu_class_t::nvidia_blackwell_k, nk_f16_k>>;
template struct us::shapes::vram_nvidia_kits<nk::f16_t, u32_t, numkong_capabilities_of(gpu_class_t::nvidia_blackwell_k),
                                             shapes_nvidia_fatbin<gpu_class_t::nvidia_blackwell_k, nk_f16_k>>;

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::vram_nvidia_k,
                                   gpu_class_t::nvidia_blackwell_k>;
