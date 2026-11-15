/**
 *  @file src/shapes/cpu_variants_ibm_power10.cpp
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief The ibm_power10 unit of the shapes variants, on the host and on storage: they inline the
 *      VSX kits, so the unit compiles with them.
 */
#include "shapes_specializations.cuh"

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::ram_k, cpu_class_t::ibm_power10_k>;
template struct basic_class_tables<shapes_traits_t, vendored_residency_t::nvme_k, cpu_class_t::ibm_power10_k>;
