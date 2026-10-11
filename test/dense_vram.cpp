/**
 *  @file test/dense_vram.cpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The one VRAM runner: resolves @c USEARCH_DEVICES against the process listing and runs
 *      each named device through the suites its vendor's test object registered.
 *
 *  Every vendor links an object library of its own into @c usearch_test_vram, compiled from the
 *  same vendor-neutral shards over its cell, and registers one entry with @c vram_suites before
 *  @c main runs. The runner names no vendor: a device whose vendor registered nothing is reported
 *  and skipped, and an unset @c USEARCH_DEVICES runs the first device of every vendor present.
 */
#include <vector> // `std::vector` - the selected devices

#include <fmt/format.h>

#include "harness.hpp"
#include "usearch/utilities/vram_device.hpp" // `gpu_devices` - the listing `USEARCH_DEVICES` indexes

using namespace unum::usearch;
using namespace unum::usearch::test;

/** The devices @p settings names, or the first listed one of every vendor when it names none. An
 *  ordinal past the listing prints the listing and exits with status 1. */
static std::vector<gpu_device_t> selected_devices(settings_t const& settings) {
    std::span<gpu_device_t const> const listed = gpu_devices();
    std::vector<gpu_device_t> selected;
    for (std::uint32_t const ordinal : settings.devices) {
        if (ordinal < listed.size()) {
            selected.push_back(listed[ordinal]);
            continue;
        }
        fmt::println(stderr, "USEARCH_DEVICES names device {}, past the {} listed:", ordinal, listed.size());
        for (std::size_t position = 0; position != listed.size(); ++position)
            fmt::println(stderr, "- {}: {} #{}", position,
                         usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(listed[position].gpu_class)),
                         listed[position].vendor_ordinal);
        std::exit(1);
    }
    if (settings.devices.empty())
        for (gpu_device_t const& device : listed)
            if (selected.empty() || selected.back().vendor != device.vendor) selected.push_back(device);
    return selected;
}

int main() {
    settings_t const settings = read_settings();
    print(probe_machine());
    print(settings);
    std::vector<gpu_device_t> const devices = selected_devices(settings);
    if (devices.empty()) return fmt::println("- GPU: none listed"), 0;
    std::size_t failures_count = 0;
    for (gpu_device_t const& device : devices) {
        vram_suite_t const* suite = nullptr;
        for (vram_suite_t const& registered : vram_suites())
            if (registered.vendor == device.vendor) suite = &registered;
        if (!suite) {
            fmt::println("- GPU: {} #{} has no suite linked",
                         usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)),
                         device.vendor_ordinal);
            continue;
        }
        failures_count += suite->check(settings, device);
    }
    return report("total", failures_count) ? 1 : 0;
}
