/**
 *  @file src/dense/vram_variants_nvidia_blackwell.cpp
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The nvidia_blackwell variants: they inline the mixed-precision multiply-add &
 *      fp6-conversion kits, so their device units compile for "100f" alone.
 *
 *  Clang compiles it host-only into the variants, which find each dtype's fatbin linked in beside
 *  them; NVCC compiles every @c vram_kernels_nvidia_<dtype>.cu device-only into this class's.
 */

#include "dense_specializations.cuh"
#include "usearch/dense/vram_nvidia_index.cuh"

/** The bounds of each dtype's fatbin, linked in as raw bytes beside the variants, in the table
 *  CMake writes. */
extern "C" byte_t const* const dense_nvidia_blackwell_bundles[];
template <>
std::span<byte_t const* const> bundle_bounds<gpu_class_t::nvidia_blackwell_k>() noexcept {
    return {dense_nvidia_blackwell_bundles, 2 * dense_dtypes_t::size_k};
}
template struct dense_class_tables<dense_home_t::vram_nvidia_k, gpu_class_t::nvidia_blackwell_k>;
