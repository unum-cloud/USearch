/**
 *  @brief The suite's one entry point: composes every area's shards explicitly and rolls
 *      their failure counts into the exit code.
 *  @author Ash Vardanian
 *  @file test/test.cpp
 *  @date July 23, 2026
 */
#include <fmt/format.h>

#include "test.hpp"

using namespace unum::usearch;

int main() {
    std::size_t failures_count = 0;

    failures_count += test_ram_buffers();
    failures_count += test_dense_all();

    return report("total", failures_count) ? 1 : 0;
}
