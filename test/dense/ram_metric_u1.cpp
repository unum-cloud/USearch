/**
 *  @file test/dense/ram_metric_u1.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke tests for the u1 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;

static_assert(scalar_wired<nk::u1x8_t>);

std::size_t test_dense_metric_u1() { return static_cast<std::size_t>(check_scalar<nk::u1x8_t>()); }
