/**
 *  @file test/header_only.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The header-only suite's entry point: the basic graph index and the utilities under it, on
 *      the standard library alone.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <string> // `std::string` - the version line

#include <fmt/format.h>

#include "harness.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

/** The headers' version alone, as this suite links no library to name the capabilities. */
machine_t test::probe_machine() noexcept {
    static std::string const version = fmt::format("{}.{}.{}", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR,
                                                   USEARCH_VERSION_PATCH);
    return {version.c_str(), "headers only", "headers only"};
}

/** Reads no class names, as this suite links no library to parse them. */
test_settings_t test::read_test_settings(int argc, char** argv) noexcept { return read_host_settings(argc, argv); }

int main(int argc, char** argv) {
    start_test_run(argc, argv);
    test::utilities::run_ram_buffers_cases();
    test::utilities::run_ram_tables_cases();
    test::basic::run_sentences_cases();
    test::basic::run_bioinformatics_cases();
    test::basic::run_meshes_cases();
    return finish_test_run();
}
