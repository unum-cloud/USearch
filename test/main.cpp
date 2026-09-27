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
    log_environment();
    std::size_t failures_count = 0;

    failures_count += utilities::check_safegraphs_format();
    failures_count += check_dense();

    return report("total", failures_count) ? 1 : 0;
}
