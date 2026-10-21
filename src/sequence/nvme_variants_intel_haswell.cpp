/**
 *  @file src/sequence/nvme_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_haswell NVMe unit of the sequence variants: they inline the AVX2 kits, so the
 *      unit compiles with them.
 */
#include "sequence_specializations.cuh"

template struct sequence_nvme_variant<cpu_class_t::intel_haswell_k>;
