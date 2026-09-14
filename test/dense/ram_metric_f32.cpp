/**
 *  @file test/dense/ram_metric_f32.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke tests for the f32 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;

std::size_t test_dense_metric_f32() { return static_cast<std::size_t>(check_scalar<nk::f32_t>()); }
