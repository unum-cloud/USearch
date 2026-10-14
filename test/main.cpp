/**
 *  @file test/main.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The dense suite's entry point: composes its areas' shards explicitly and rolls their
 *      failure counts into the exit code; the header-only areas run under @c header_only.cpp.
 */
#include "harness.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

int main() {
    settings_t const settings = read_settings();
    print(probe_machine());
    print(settings);
    std::size_t failures_count = 0;

    if (settings.selects("safegraphs_format")) failures_count += test::utilities::check_safegraphs_format(settings);
    failures_count += check_dense(settings);
    if (settings.selects("sequence")) failures_count += test::sequence::check_sequence(settings);
    if (settings.selects("shapes")) failures_count += test::shapes::check_shapes(settings);
    if (settings.selects("geospatial")) failures_count += test::geospatial::check_geospatial(settings);

    return report("total", failures_count) ? 1 : 0;
}
