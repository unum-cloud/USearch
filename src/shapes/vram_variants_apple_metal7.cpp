/**
 *  @file src/shapes/vram_variants_apple_metal7.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Apple unit of the shapes kit: Chamfer and Hausdorff over f16 packs need nothing past
 *      the Apple7 floor, so one unit serves every Apple class, compiling the kit's library from its
 *      embedded MSL at the first @c reserve.
 *
 *  It defines the face and the tables of the shapes variant over Apple GPUs. NumKong ships no f32
 *  Metal body meeting CUDA's f32 guarantee, so no f32 variant exists here, and the dispatch refuses
 *  f32 clouds on Apple GPUs with @c unsupported_k.
 */
#include "shapes_specializations.cuh"

template struct us::shapes::vram_apple_index<nk::f16_t, u32_t>;
template struct us::shapes::vram_apple_kits<nk::f16_t, u32_t>;

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::vram_apple_k, gpu_class_t::apple_metal7_k>;
