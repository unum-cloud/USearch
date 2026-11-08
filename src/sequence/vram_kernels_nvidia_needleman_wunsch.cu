/**
 *  @file src/sequence/vram_kernels_nvidia_needleman_wunsch.cu
 *  @author Ash Vardanian
 *  @date October 21, 2026
 *  @brief The Needleman-Wunsch sequence kernels at the Ampere floor, for every NVIDIA class.
 *
 *  NVCC compiles it device-only into the needleman_wunsch fatbin, which
 *  @c vram_variants_nvidia.cpp finds linked in beside its variants.
 */
#include "sequence_specializations.cuh"
#include "usearch/sequence/vram_nvidia_metric.cuh"

template __device__ void us::sequence::cuda_instantiate<
    sequence_nvidia_capabilities_k, us::sequence::vram_sequence_metric_t::needleman_wunsch_k, u32_t>();
