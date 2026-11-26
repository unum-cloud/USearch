/**
 *  @file src/sequence/vram_kernels_amd_utf8_containment.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The containment sequence kernels of one AMD class, under its class's kit.
 *
 *  HIP-Clang compiles it device-only, once per class with that class's processors, into its
 *  containment code object, which @c vram_variants_amd.hip finds linked in beside it.
 */
#include "usearch/sequence/vram_amd_metric.hip" // `sequence::amd_instantiate`, the AMD rungs
#include "sequence_kits.hpp"                    // `sequence_vram_kit`
#include "classes.hpp"                          // `hip_class_k`

template void us::sequence::amd_instantiate<sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, hip_class_k>,
                                            util::u32_t>();
