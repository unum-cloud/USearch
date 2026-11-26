/**
 *  @file src/shapes_kits.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The kit the shapes kernels score with on a GPU class, and the device code each vendor's
 *      class unit embeds for it.
 *
 *  The class units, the device kernel units and the tests name the same kits and device code
 *  through it, so none of them restates a ladder pick.
 */
#pragma once

#include <numkong/types.h> // `nk_dtype_t`

#include "usearch/shapes/ram_metric.hpp" // `shapes_ladder`

#include "classes.hpp" // `gpu_class_t`, `numkong_capabilities_of`

/** The kit packs of @p scalar_type_ score with on a GPU of @p gpu_class_, its ladder's pick at the
 *  class's NumKong mask; the vendor's metric header supplies the rungs. */
template <typename scalar_type_, gpu_class_t gpu_class_>
using shapes_vram_kit = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(gpu_class_)>::kit_t;

/** One class's shapes kits over one dtype as a fatbin, defined in its class unit,
 *  @c src/shapes/vram_variants_nvidia.cpp for the floor or a monoversioned build's one class,
 *  beside the bytes it embeds. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_nvidia_fatbin {
    static util::bytes_view_t bytes() noexcept;
};

/** One class's shapes kits over one dtype as a code object, defined in its class unit,
 *  @c src/shapes/vram_variants_amd.hip, beside the bytes it embeds. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static util::bytes_view_t bytes() noexcept;
};
