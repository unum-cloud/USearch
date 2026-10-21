/**
 *  @file src/sequence/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The alibaba_c930 unit of the sequence variants: they fall back to serial kits, compiled
 *      with the class's flags.
 */
#include "sequence_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::alibaba_c930_k>;
