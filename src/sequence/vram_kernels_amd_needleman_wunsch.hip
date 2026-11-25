/**
 *  @file src/sequence/vram_kernels_amd_needleman_wunsch.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Needleman-Wunsch sequence kernels of one AMD class, under its class's kit.
 *
 *  HIP-Clang compiles it device-only, once per class with that class's processors, into its
 *  needleman_wunsch code object, which @c vram_variants_amd.hip finds linked in beside it.
 */
#include "sequence_specializations.cuh"

template void
us::sequence::amd_instantiate<sequence_amd_kit<usearch_sequence_metric_needleman_wunsch_k>, util::u32_t>();
