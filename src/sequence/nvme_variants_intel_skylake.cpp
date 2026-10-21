/**
 *  @file src/sequence/nvme_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_skylake NVMe unit of the sequence variants: they inline the AVX-512 kits, so
 *      the unit compiles with them.
 */
#include "sequence_specializations.cuh"

template struct sequence_nvme_variant<cpu_class_t::intel_skylake_k>;
