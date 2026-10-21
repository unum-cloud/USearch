/**
 *  @file src/sequence/vram_kernels_nvidia_levenshtein_utf8.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The UTF-8 Levenshtein sequence kernels at the Ampere floor, for every NVIDIA class.
 *
 *  NVCC compiles it device-only into the levenshtein_utf8 fatbin, which @c vram_variants_nvidia.cpp
 *  finds linked in beside its variants.
 */
#include "sequence_specializations.cuh"
#include "usearch/sequence/vram_nvidia_metric.cuh"

template __device__ void
us::sequence::cuda_instantiate<sequence_nvidia_capabilities_k, usearch_sequence_metric_levenshtein_utf8_k, u32_t>();
