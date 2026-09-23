/**
 *  @file test/dense/ram_metric_e4m3.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke tests for the e4m3 kernels of @c dense/ram_metric.hpp against NumKong's reference.
 */
#include "dense/ram_metric.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test::dense;

static_assert(scalar_wired<nk::e4m3_t>);

std::size_t test::dense::check_ram_metric_e4m3() { return static_cast<std::size_t>(check_scalar<nk::e4m3_t>()); }
