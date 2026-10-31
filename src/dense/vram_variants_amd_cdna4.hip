/**
 *  @file src/dense/vram_variants_amd_cdna4.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_cdna4 unit: MI350X & MI355X, gfx950; its variants inline the bf16 dots and the OCP
 *      fp8 conversions.
 *
 *  HIP-Clang compiles it host-only into the variants, which find each dtype's code object linked in
 *  beside them; HIP-Clang compiles every @c vram_kernels_amd_<dtype>.hip device-only into this
 *  class's.
 */

#include "dense_specializations.cuh"
#include "usearch/dense/vram_amd_index.hip"

/** The bounds of each dtype's code object, linked in as raw bytes beside the variants, in the table
 *  CMake writes. */
extern "C" byte_t const* const dense_amd_cdna4_bundles[];
template <>
std::span<byte_t const* const> bundle_bounds<gpu_class_t::amd_cdna4_k>() noexcept {
    return {dense_amd_cdna4_bundles, 2 * dense_dtypes_t::size_k};
}
template struct dense_class_tables<vendored_residency_t::vram_amd_k, gpu_class_t::amd_cdna4_k>;
