/**
 *  @brief Smoke tests for the e3m2 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 *  @author Ash Vardanian
 *  @file test/dense/ram_metric_e3m2.cpp
 *  @date July 23, 2026
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;

static_assert(scalar_wired<nk::e3m2_t>);

std::size_t test_dense_metric_e3m2() { return static_cast<std::size_t>(check_scalar<nk::e3m2_t>()); }
