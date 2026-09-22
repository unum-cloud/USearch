/**
 *  @file src/dense/vram_variants_e5m2.cu
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief The e5m2 unit of the dense index library: its VRAM arms' explicit instantiations.
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k>>;
template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_angular_k>>;
template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_euclidean_k>>;
#endif
