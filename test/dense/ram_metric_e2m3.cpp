/**
 *  @brief Smoke tests for the e2m3 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 *  @author Ash Vardanian
 *  @file test/dense/ram_metric_e2m3.cpp
 *  @date July 23, 2026
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;

static_assert(scalar_wired<nk::e2m3_t>);

std::size_t test_dense_metric_e2m3() { return static_cast<std::size_t>(check_scalar<nk::e2m3_t>()); }
