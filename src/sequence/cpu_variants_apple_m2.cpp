/**
 *  @file src/sequence/cpu_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The apple_m2 unit of the sequence variants, on the host and on storage: they fall back to
 *      the NEON kits they inline like aws_g2.
 */
#include "sequence_specializations.cuh"

template struct basic_class_tables<sequence_traits_t, vendored_residency_t::ram_k, cpu_class_t::apple_m2_k>;
template struct basic_class_tables<sequence_traits_t, vendored_residency_t::nvme_k, cpu_class_t::apple_m2_k>;
