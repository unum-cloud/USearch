/**
 *  @file src/geospatial/vram_variants_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The NVIDIA unit of the geospatial kit: squared chords need nothing past the Ampere floor,
 *      so one unit carries a cubin for every NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only, over the kit and the kernel judging its walks' matches,
 *  into the unit's fatbin, and by Clang host-only, where it names that fatbin, linked in beside
 *  it, to the host index.
 */
#include "usearch/geospatial/vram_nvidia_index.hpp"
#include "usearch/basic/vram_nvidia_index.cuh"

namespace us = unum::usearch;

template <>
struct us::basic::cuda_kit_of<us::geospatial::i32_ampere_chord_kit_t::kit_k> {
    using type = us::geospatial::i32_ampere_chord_kit_t;
};

/** Rows keep their neighbors by tangent-plane cone, never by the diversity rule. */
template <>
struct us::basic::cuda_selector_of<us::geospatial::i32_ampere_chord_kit_t::kit_k> {
    using type = us::geospatial::vram_keep_octants_t;
};

/** The unit's fatbin, linked in as raw bytes. */
extern "C" us::utilities::byte_t const geospatial_nvidia_bundle_start[], geospatial_nvidia_bundle_end[];

template <>
us::utilities::bytes_view_t us::basic::cuda_fatbin<us::geospatial::vram_nvidia_index>() noexcept {
    return {geospatial_nvidia_bundle_start,
            static_cast<std::size_t>(geospatial_nvidia_bundle_end - geospatial_nvidia_bundle_start)};
}

template __device__ void us::basic::cuda_instantiate<us::geospatial::i32_ampere_chord_kit_t::kit_k>();

/** Taking the judging kernel's address in device code makes NVCC emit it, as @c cuda_instantiate
 *  does the index's. */
static __device__ void instantiate_admit_() noexcept {
    static_cast<void>(&us::geospatial::cuda_admit<us::geospatial::i32_ampere_chord_kit_t>);
}
