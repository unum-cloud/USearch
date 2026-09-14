/**
 *  @file src/dense/vram_variants_bf16.cu
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief The bf16 unit of the dense index library: its VRAM arms' explicit instantiations.
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k>>;
template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k>>;
#endif
