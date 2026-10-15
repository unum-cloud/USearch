/**
 *  @file include/usearch/shapes/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over 3D point clouds on NVIDIA GPUs: the basic VRAM index over the
 *      Chamfer and Hausdorff kits of @c vram_nvidia_metric.cuh.
 *
 *  Plain C++, so the host classes build with any compiler: the kits' device members compile only
 *  into the fatbin, while their host facts pick a launch's kit and size its spans. Rows are NumKong
 *  packs end to end, as @c nk::shapes_pack lays them out, in the caller's tape and in the index.
 */
#pragma once

#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/shapes/vram_nvidia_metric.cuh"

namespace unum::usearch::shapes {

/** Similarity Search over 3D point clouds on NVIDIA GPUs, under the kit family @p kit_template_,
 *  such as @c f16_ampere_chamfer_kit: a block walks each query, a warp scoring each candidate. */
template <template <u32_t> class kit_template_>
using vram_nvidia_index = basic::vram_nvidia_index<kit_template_<32>>;

} // namespace unum::usearch::shapes
