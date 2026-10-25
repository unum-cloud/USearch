/**
 *  @file test/main.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The engine suite's entry point, rolling every area's failures into the exit code.
 *
 *  It composes its areas' cells explicitly, then the suites over the C interface, then every
 *  selected GPU through the suites its vendor's object registered; the header-only areas run under
 *  @c header_only.cpp. A device whose vendor registered nothing is reported and skipped, and an
 *  unset @c USEARCH_DEVICES runs the first device of every vendor present.
 */
#include <cstdlib> // `std::exit` - a device past the listing

#include <vector> // `std::vector` - the selected devices

#include "usearch/utilities/vram_device.hpp" // `gpu_devices` - the listing `USEARCH_DEVICES` indexes

#include "harness.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

/** The positions in the listing @p settings names, or of the first listed device of every vendor
 *  when it names none. An ordinal past the listing exits with status 1 once the listing printed. */
static std::vector<std::size_t> selected_devices(settings_t const& settings) {
    std::span<gpu_device_t const> const listed = gpu_devices();
    std::vector<std::size_t> selected;
    for (std::int32_t const ordinal : settings.devices) {
        if (static_cast<std::size_t>(ordinal) < listed.size()) {
            selected.push_back(static_cast<std::size_t>(ordinal));
            continue;
        }
        fmt::println(stderr, "USEARCH_DEVICES names device {}, past the {} listed", ordinal, listed.size());
        std::exit(1);
    }
    if (settings.devices.empty())
        for (std::size_t ordinal = 0; ordinal != listed.size(); ++ordinal)
            if (selected.empty() || listed[selected.back()].vendor != listed[ordinal].vendor)
                selected.push_back(ordinal);
    return selected;
}

/** Every selected device through the suites its vendor's unit registered. */
static std::size_t check_devices(settings_t const& settings) {
    std::span<gpu_device_t const> const listed = gpu_devices();
    for (std::size_t position = 0; position != listed.size(); ++position)
        fmt::println("- GPU {}: {}, {} #{}, {} compute units, {} MiB", position, listed[position].name,
                     usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(listed[position].gpu_class)),
                     listed[position].vendor_ordinal, listed[position].compute_units,
                     listed[position].memory_bytes >> 20);
    std::vector<std::size_t> const ordinals = selected_devices(settings);
    if (ordinals.empty()) return fmt::println("- GPU: none listed"), 0;
    std::size_t failures_count = 0;
    for (std::size_t const ordinal : ordinals) {
        gpu_device_t const& device = listed[ordinal];
        registered_suite_t const* suite = nullptr;
        for (registered_suite_t const& registered : registered_suites())
            if (registered.vendor == device.vendor) suite = &registered;
        if (!suite) fmt::println("- GPU: {} #{} has no suite linked", device.name, device.vendor_ordinal);
        else failures_count += suite->check(settings, device);
    }
    return failures_count;
}

int main(int argc, char** argv) {
    settings_t const settings = read_settings(argc, argv, "usearch_test");
    print(probe_machine());
    print(settings);
    std::size_t failures_count = 0;

    if (settings.filter.selects("safegraphs_format"))
        failures_count += test::utilities::check_safegraphs_format(settings);
    failures_count += check_dense(settings);
    for (registered_suite_t const& registered : registered_suites())
        if (registered.vendor == gpu_vendor_t::unknown_k) failures_count += registered.check(settings, {});
    if (settings.filter.selects("sequence")) failures_count += test::sequence::check_sequence(settings);
    if (settings.filter.selects("shapes")) failures_count += test::shapes::check_shapes(settings);
    if (settings.filter.selects("geospatial")) failures_count += test::geospatial::check_geospatial(settings);

    failures_count += test::abi::check_dense_abi(settings);
    failures_count += test::abi::check_sequence_abi(settings);
    failures_count += test::abi::check_shapes_abi(settings);
    failures_count += test::abi::check_geospatial_abi(settings);

    failures_count += check_devices(settings);
    return report("total", failures_count) ? 1 : 0;
}
