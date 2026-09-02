/**
 *  @brief The bf16 shard of the dense index library: its VRAM arms' explicit instantiations.
 *  @author Ash Vardanian
 *  @file src/dense/vram_indexes_bf16.cu
 *  @date September 1, 2026
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k>>;
template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k>>;
#endif
