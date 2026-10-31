/**
 *  @file src/dense/vram_variants_apple_metal10.cpp
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The Apple10 unit: one variant per dtype, each compiling its kernels from the
 *      embedded source for the class's capability set at the first @c reserve.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::vram_apple_k, gpu_class_t::apple_metal10_k>;
