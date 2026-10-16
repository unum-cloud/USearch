/**
 *  @file src/sequence/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The aws_g2 unit of the sequence variants: they inline the NEON kits, so the unit compiles
 *      with them.
 */
#include "sequence_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::aws_g2_k>;
template struct sequence_nvme_variant<cpu_class_t::aws_g2_k>;
