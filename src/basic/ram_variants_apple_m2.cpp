/**
 *  @file src/basic/ram_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The apple_m2 unit of the basic modalities: only the f16 shapes arm advances here,
 *      widening through the NEONFHM Chamfer and Hausdorff kernels.
 */
#include "basic_specializations.cuh"

#if NUMKONG_TARGET_NEONFHM
template struct shapes_arm<nk::f16_t, cpu_class_t::apple_m2_k>;
#endif
