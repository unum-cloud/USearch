/**
 *  @file src/basic/vram_variants_nvidia_shapes.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the shapes kits: Chamfer and Hausdorff need nothing past the Ampere
 *      floor, so one unit carries a cubin for every NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only, over every kit, into the unit's fatbin, and by Clang
 *  host-only, where it names that fatbin, linked in beside it, to the host indexes.
 */
#include "usearch/shapes/vram_nvidia_index.hpp"
#include "usearch/basic/vram_nvidia_index.cuh"

namespace us = unum::usearch;

/** The Chamfer and Hausdorff kits over f16 and f32 packs, by the integers their kits build. */
template <>
struct us::basic::cuda_kit_of<us::shapes::f16_ampere_chamfer_kit<32>::kit_k> {
    using type = us::shapes::f16_ampere_chamfer_kit<32>;
};
template <>
struct us::basic::cuda_kit_of<us::shapes::f16_ampere_hausdorff_kit<32>::kit_k> {
    using type = us::shapes::f16_ampere_hausdorff_kit<32>;
};
template <>
struct us::basic::cuda_kit_of<us::shapes::f32_ampere_chamfer_kit<32>::kit_k> {
    using type = us::shapes::f32_ampere_chamfer_kit<32>;
};
template <>
struct us::basic::cuda_kit_of<us::shapes::f32_ampere_hausdorff_kit<32>::kit_k> {
    using type = us::shapes::f32_ampere_hausdorff_kit<32>;
};

/** The unit's fatbin, linked in as raw bytes, which every shapes index loads. */
extern "C" us::utilities::byte_t const shapes_nvidia_bundle_start[], shapes_nvidia_bundle_end[];

template <>
us::utilities::bytes_view_t
us::basic::cuda_fatbin<us::shapes::vram_nvidia_index<us::shapes::f16_ampere_chamfer_kit>>() noexcept {
    return {shapes_nvidia_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_bundle_end - shapes_nvidia_bundle_start)};
}
template <>
us::utilities::bytes_view_t
us::basic::cuda_fatbin<us::shapes::vram_nvidia_index<us::shapes::f16_ampere_hausdorff_kit>>() noexcept {
    return {shapes_nvidia_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_bundle_end - shapes_nvidia_bundle_start)};
}
template <>
us::utilities::bytes_view_t
us::basic::cuda_fatbin<us::shapes::vram_nvidia_index<us::shapes::f32_ampere_chamfer_kit>>() noexcept {
    return {shapes_nvidia_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_bundle_end - shapes_nvidia_bundle_start)};
}
template <>
us::utilities::bytes_view_t
us::basic::cuda_fatbin<us::shapes::vram_nvidia_index<us::shapes::f32_ampere_hausdorff_kit>>() noexcept {
    return {shapes_nvidia_bundle_start,
            static_cast<std::size_t>(shapes_nvidia_bundle_end - shapes_nvidia_bundle_start)};
}

template __device__ void us::basic::cuda_instantiate<us::shapes::f16_ampere_chamfer_kit<32>::kit_k>();
template __device__ void us::basic::cuda_instantiate<us::shapes::f16_ampere_hausdorff_kit<32>::kit_k>();
template __device__ void us::basic::cuda_instantiate<us::shapes::f32_ampere_chamfer_kit<32>::kit_k>();
template __device__ void us::basic::cuda_instantiate<us::shapes::f32_ampere_hausdorff_kit<32>::kit_k>();
