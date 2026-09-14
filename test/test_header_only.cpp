/**
 *  @file test/test_header_only.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The header-only suite's entry point: the basic graph index and the utilities under it, on
 *      the standard library alone.
 */
#include <fmt/format.h>

#include "test.hpp"

using namespace unum::usearch;

int main() {
    std::size_t failures_count = 0;

    failures_count += test_ram_buffers();
    failures_count += test_ram_tables();
    failures_count += test_sentences();
    failures_count += test_bioinformatics();
    failures_count += test_meshes();

    return report("total", failures_count) ? 1 : 0;
}
