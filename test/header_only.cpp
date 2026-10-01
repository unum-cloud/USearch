/**
 *  @file test/header_only.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The header-only suite's entry point: the basic graph index and the utilities under it, on
 *      the standard library alone.
 */
#include <fmt/format.h>

#include "harness.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

int main() {
    settings_t const settings = read_settings();
    print(settings);
    std::size_t failures_count = 0;

    if (settings.selects("ram_buffers")) failures_count += utilities::check_ram_buffers(settings);
    if (settings.selects("ram_tables")) failures_count += utilities::check_ram_tables(settings);
    if (settings.selects("sentences")) failures_count += basic::check_sentences(settings);
    if (settings.selects("bioinformatics")) failures_count += basic::check_bioinformatics(settings);
    if (settings.selects("meshes")) failures_count += basic::check_meshes(settings);

    return report("total", failures_count) ? 1 : 0;
}
