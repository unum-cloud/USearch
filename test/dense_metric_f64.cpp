/**
 *  @brief Smoke tests for the f64 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 *  @author Ash Vardanian
 *  @file test/dense_metric_f64.cpp
 *  @date July 23, 2026
 */
#include "dense_metric.hpp"

using namespace unum::usearch;

std::size_t test_dense_metric_f64() { return static_cast<std::size_t>(check_scalar<nk::f64_t>()); }
