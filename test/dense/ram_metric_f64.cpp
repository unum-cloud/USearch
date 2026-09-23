/**
 *  @file test/dense/ram_metric_f64.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke tests for the f64 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_metric_f64() { return static_cast<std::size_t>(check_scalar<nk::f64_t>()); }
