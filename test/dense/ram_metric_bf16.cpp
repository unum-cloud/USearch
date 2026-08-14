/**
 *  @brief Smoke tests for the bf16 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 *  @author Ash Vardanian
 *  @file test/dense/ram_metric_bf16.cpp
 *  @date July 23, 2026
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;

static_assert(scalar_wired<nk::bf16_t>);

std::size_t test_dense_metric_bf16() { return static_cast<std::size_t>(check_scalar<nk::bf16_t>()); }
