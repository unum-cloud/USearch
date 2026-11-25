/**
 *  @file src/sequence/vram_kernels_amd_levenshtein_utf8.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The UTF-8 Levenshtein sequence kernels of one AMD class, under its class's kit.
 *
 *  HIP-Clang compiles it device-only, once per class with that class's processors, into its
 *  levenshtein_utf8 code object, which @c vram_variants_amd.hip finds linked in beside it.
 */
#include "sequence_specializations.cuh"

template void
us::sequence::amd_instantiate<sequence_amd_kit<usearch_sequence_metric_levenshtein_utf8_k>, util::u32_t>();
