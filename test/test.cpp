/**
 *  @brief The dense suite's entry point: composes its areas' shards explicitly and rolls
 *      their failure counts into the exit code; the header-only areas run under @c test_header_only.cpp.
 *  @author Ash Vardanian
 *  @file test/test.cpp
 *  @date July 23, 2026
 */
#include <fmt/format.h>

#include "test.hpp"

using namespace unum::usearch;

int main() {
    std::size_t failures_count = 0;

    failures_count += test_safegraphs_format();
    failures_count += test_dense_all();

    return report("total", failures_count) ? 1 : 0;
}
