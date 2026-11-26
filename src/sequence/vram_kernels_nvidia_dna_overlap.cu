/**
 *  @file src/sequence/vram_kernels_nvidia_dna_overlap.cu
 *  @author Ash Vardanian
 *  @date November 24, 2026
 *  @brief The banded overlap sequence kernels at the Ampere floor, for every NVIDIA class.
 *
 *  NVCC compiles it device-only into the banded overlap fatbin, which @c vram_variants_nvidia.cpp
 *  finds linked in beside its variants.
 */
#include "usearch/sequence/vram_nvidia_metric.cuh" // `sequence::cuda_instantiate`, the NVIDIA rungs
#include "sequence_kits.hpp"                       // `sequence_vram_kit`
#include "classes.hpp"                             // `cuda_class_k`

template __device__ void
us::sequence::cuda_instantiate<sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, cuda_class_k>, util::u32_t>();
