/**
 *  @file test/dense/ram_metric_e5m2.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke tests for the e5m2 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test::dense;

static_assert(scalar_wired<nk::e5m2_t>);

std::size_t test::dense::check_ram_metric_e5m2() { return static_cast<std::size_t>(check_scalar<nk::e5m2_t>()); }
