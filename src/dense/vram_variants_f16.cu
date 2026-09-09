/**
 *  @brief The f16 unit of the dense index library: its VRAM arms' explicit instantiations.
 *  @author Ash Vardanian
 *  @file src/dense/vram_variants_f16.cu
 *  @date September 1, 2026
 */
#include "dense_specializations.cuh"

#if USEARCH_BUILD_CUDA
template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k>>;
#endif
