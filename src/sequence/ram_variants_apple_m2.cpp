/**
 *  @file src/sequence/ram_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The apple_m2 unit of the sequence variants: they fall back to the NEON kits they inline
 *      like aws_g2.
 */
#include "sequence_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::apple_m2_k>;
template struct sequence_nvme_variant<cpu_class_t::apple_m2_k>;
