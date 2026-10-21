/**
 *  @file src/sequence/nvme_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The wasm64_extended NVMe unit of the sequence variants: they fall back to serial kits,
 *      compiled with the class's flags.
 */
#include "sequence_specializations.cuh"

template struct sequence_nvme_variant<cpu_class_t::wasm64_extended_k>;
