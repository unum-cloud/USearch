/**
 *  @brief Smoke tests for the u4 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 *  @author Ash Vardanian
 *  @file test/dense_metric_u4.cpp
 *  @date July 23, 2026
 */
#include "dense_metric.hpp"
#include "test.hpp"

static_assert(scalar_wired<nk::u4x2_t>);

std::size_t test_dense_metric_u4() { return static_cast<std::size_t>(check_scalar<nk::u4x2_t>()); }
