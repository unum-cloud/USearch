/**
 *  @file src/sequence/vram_kernels_nvidia_levenshtein_bytes.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The byte Levenshtein sequence kernels at the Ampere floor, for every NVIDIA class.
 *
 *  NVCC compiles it device-only into the levenshtein_bytes fatbin, which
 *  @c vram_variants_nvidia.cpp finds linked in beside its variants.
 */
#include "sequence_specializations.cuh"

template __device__ void
us::sequence::cuda_instantiate<sequence_nvidia_kit<usearch_sequence_metric_levenshtein_bytes_k>, u32_t>();
