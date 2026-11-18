/**
 *  @file test/main.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The engine suite's entry point, running every area's cases into the run.
 *
 *  It runs its areas' cases explicitly, then the cases over the C interface, then every selected
 *  GPU through the cases its vendor's object registered; the header-only areas run under
 *  @c header_only.cpp. A device whose vendor registered nothing is a skipped case, and an unset
 *  @c USEARCH_DEVICES runs the first device of every vendor present.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdlib> // `std::exit`, `std::getenv` - a device past the listing, the job summary

#include <algorithm> // `std::none_of` - a vendor with no device
#include <string>    // `std::string` - a device case's label
#include <vector>    // `std::vector` - the selected devices

#include "usearch/utilities/vram_device.hpp" // `gpu_drivers_t`, `list_gpu_devices` - the listing `USEARCH_DEVICES` indexes

#include "harness.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

std::span<gpu_device_t const> test::gpu_devices() noexcept {
    struct listing_t {
        gpu_device_t devices[3 * gpu_devices_capacity_k];
        std::size_t count = 0;
    };
    // The C interface numbers only the devices some class runs on, so the runner keeps the same.
    static listing_t const listing = [] {
        listing_t reported, numbered;
        reported.count = list_gpu_devices(gpu_drivers(), reported.devices);
        usearch_gpu_device_t devices[3 * gpu_devices_capacity_k];
        std::size_t devices_count = 0;
        usearch_outcome_t outcome {};
        usearch_gpu_devices_t arguments {&outcome, devices, std::size(devices), &devices_count};
        usearch_gpu_devices(&arguments);
        for (std::size_t position = 0; position != devices_count; ++position)
            for (std::size_t raw = 0; raw != reported.count; ++raw)
                if (static_cast<usearch_gpu_vendor_t>(reported.devices[raw].vendor) == devices[position].vendor &&
                    reported.devices[raw].vendor_ordinal == devices[position].vendor_ordinal)
                    numbered.devices[numbered.count++] = reported.devices[raw];
        return numbered;
    }();
    return {listing.devices, listing.count};
}

machine_t test::probe_machine() noexcept { return probe_library_machine(); }

test_settings_t test::read_test_settings(int argc, char** argv) noexcept {
    test_settings_t settings = read_host_settings(argc, argv);
    read_session_settings(argc, argv, settings);
    return settings;
}

/** The positions in the listing @c USEARCH_DEVICES names, or of the first listed device of every
 *  vendor when it names none. An ordinal past the listing prints the listing, then the process
 *  exits with status 1. */
static std::vector<std::size_t> selected_devices() {
    std::span<gpu_device_t const> const listed = gpu_devices();
    std::vector<std::size_t> selected;
    for (std::int32_t const ordinal : test_run.settings.devices) {
        if (static_cast<std::size_t>(ordinal) < listed.size()) {
            selected.push_back(static_cast<std::size_t>(ordinal));
            continue;
        }
        fmt::println(stderr, "USEARCH_DEVICES names device {}, past the {} listed", ordinal, listed.size());
        std::exit(1);
    }
    if (test_run.settings.devices.empty())
        for (std::size_t ordinal = 0; ordinal != listed.size(); ++ordinal)
            if (selected.empty() || listed[selected.back()].vendor != listed[ordinal].vendor)
                selected.push_back(ordinal);
    return selected;
}

/** Why a device of @p vendor skips when the build registered no cases for it. */
static case_result_t skip_unbuilt_vendor(gpu_vendor_t vendor) noexcept {
    switch (vendor) {
    case gpu_vendor_t::nvidia_k: return case_result_t::skipped("built without USEARCH_CUDA_ARCH");
    case gpu_vendor_t::amd_k: return case_result_t::skipped("built without USEARCH_ROCM_ARCH");
    default: return case_result_t::skipped("built without USEARCH_METAL_ARCH");
    }
}

/** Every selected device through the cases its vendor's unit registered. */
static void run_device_cases() {
    std::span<gpu_device_t const> const listed = gpu_devices();
    for (std::size_t position = 0; position != listed.size(); ++position)
        fmt::println("- GPU {}: {}, architecture {}.{} #{}, {} compute units, {} MiB", position, listed[position].name,
                     listed[position].architecture_major, listed[position].architecture_minor,
                     listed[position].vendor_ordinal, listed[position].compute_units,
                     listed[position].memory_bytes >> 20);
    for (std::size_t const ordinal : selected_devices()) {
        gpu_device_t const& device = listed[ordinal];
        registered_suite_t const* suite = nullptr;
        for (registered_suite_t const& registered : registered_suites())
            if (registered.vendor == device.vendor) suite = &registered;
        if (suite) suite->run_cases(device);
        else
            run_test_case(fmt::format("registered_suite_t::run_cases({})", constant_of(device.vendor)).c_str(),
                          skip_unbuilt_vendor, device.vendor);
    }
    // A vendor's cases that found no device of its own report as skipped rather than vanish.
    for (registered_suite_t const& registered : registered_suites())
        if (registered.vendor != gpu_vendor_t::unknown_k &&
            std::none_of(listed.begin(), listed.end(),
                         [&](gpu_device_t const& device) { return device.vendor == registered.vendor; })) {
            run_test_case(fmt::format("registered_suite_t::run_cases({})", constant_of(registered.vendor)).c_str(),
                          [] { return case_result_t::skipped(no_gpu_device_k); });
        }
}

int main(int argc, char** argv) {
    start_test_run(argc, argv);
    test::utilities::run_safegraphs_format_cases();
    test::dense::run_ram_metric_cases();
    test::dense::run_nvme_indexes_cases();
    for (registered_suite_t const& registered : registered_suites())
        if (registered.vendor == gpu_vendor_t::unknown_k) registered.run_cases({});
    test::sequence::run_sequence_cases();
    test::shapes::run_shapes_cases();
    test::geospatial::run_geospatial_cases();
    test::abi::run_dense_abi_cases();
    test::abi::run_sequence_abi_cases();
    test::abi::run_shapes_abi_cases();
    test::abi::run_geospatial_abi_cases();
    run_device_cases();
    int const exit_code = finish_test_run();
    // ctest hides passing output, so the job summary is where a green run shows what it covered
    if (char const* summary_path = std::getenv("GITHUB_STEP_SUMMARY"))
        if (std::FILE* summary = std::fopen(summary_path, "a")) {
            print(probe_machine(), summary);
            fmt::println(summary, "{} passed, {} failed, {} skipped, seed {}", test_run.passed_cases,
                         test_run.failed_cases, test_run.skipped_cases, test_run.settings.seed);
            std::fclose(summary);
        }
    return exit_code;
}
