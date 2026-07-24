/**
 *  @brief Smoke tests for the u1 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 *  @author Ash Vardanian
 *  @file test/dense_metric_u1.cpp
 *  @date July 23, 2026
 */
#include "dense_metric.hpp"

using namespace unum::usearch;

static_assert(scalar_wired<nk::u1x8_t>);

std::size_t test_dense_metric_u1() { return static_cast<std::size_t>(check_scalar<nk::u1x8_t>()); }
