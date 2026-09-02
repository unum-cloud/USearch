/**
 *  @brief The u1 shard of the dense index library: its VRAM arms' explicit instantiations.
 *  @author Ash Vardanian
 *  @file src/dense/vram_indexes_u1.cu
 *  @date September 1, 2026
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k>>;
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k>>;
#endif
