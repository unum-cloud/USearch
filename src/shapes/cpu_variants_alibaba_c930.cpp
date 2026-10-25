/**
 *  @file src/shapes/cpu_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The alibaba_c930 unit of the shapes variants, on the host and on storage: they inline the
 *      RVV kits, so the unit compiles with them.
 */
#include "shapes_specializations.cuh"

template struct basic_class_tables<shapes_traits_t, dense_home_t::ram_k, cpu_class_t::alibaba_c930_k>;
template struct basic_class_tables<shapes_traits_t, dense_home_t::nvme_k, cpu_class_t::alibaba_c930_k>;
