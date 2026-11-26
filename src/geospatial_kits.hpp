/**
 *  @file src/geospatial_kits.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The kit the geospatial kernels score with on a GPU class, and the device code each
 *      vendor's unit embeds for it.
 *
 *  The class units and the device passes name the same kit and device code through it, so none of
 *  them restates a ladder pick.
 */
#pragma once

#include "usearch/geospatial/ram_metric.hpp" // `geospatial_ladder`

#include "classes.hpp" // `gpu_class_t`, `numkong_capabilities_of`

/** The kit observations score with on a GPU of @p gpu_class_, its ladder's pick at the class's
 *  NumKong mask; the vendor's metric header supplies the rungs. */
template <gpu_class_t gpu_class_>
using geospatial_vram_kit = typename us::geospatial::geospatial_ladder<numkong_capabilities_of(gpu_class_)>::kit_t;

/** The fatbin of the geospatial kit and its judging kernel, defined in
 *  @c src/geospatial/vram_variants_nvidia.cu. */
struct geospatial_nvidia_fatbin_t {
    static util::bytes_view_t bytes() noexcept;
};

/** One AMD class's code object of the geospatial kit and its judging kernels, defined in its class
 *  unit, @c src/geospatial/vram_variants_amd.hip, under that class's ceiling. */
template <gpu_class_t gpu_class_>
struct geospatial_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static util::bytes_view_t bytes() noexcept;
};
