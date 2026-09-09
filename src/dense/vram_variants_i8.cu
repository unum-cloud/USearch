/**
 *  @brief The i8 unit of the dense index library: its VRAM arms' explicit instantiations.
 *  @author Ash Vardanian
 *  @file src/dense/vram_variants_i8.cu
 *  @date September 1, 2026
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k>>;
template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k>>;
#endif
