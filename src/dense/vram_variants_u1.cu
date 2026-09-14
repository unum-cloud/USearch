/**
 *  @file src/dense/vram_variants_u1.cu
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief The u1 unit of the dense index library: its VRAM arms' explicit instantiations.
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k>>;
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k>>;
#endif
