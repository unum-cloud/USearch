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
using namespace unum::usearch::test;

int main() {
    std::size_t failures_count = 0;

    failures_count += utilities::check_ram_buffers();
    failures_count += utilities::check_ram_tables();
    failures_count += basic::check_sentences();
    failures_count += basic::check_bioinformatics();
    failures_count += basic::check_meshes();

    return report("total", failures_count) ? 1 : 0;
}
