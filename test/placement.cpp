/**
 *  @file test/placement.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Resolves the devices a VRAM index names on emulated machines, with no GPU behind them.
 *
 *  Each machine's listing is built from its profile, and a class loads on a device whose
 *  architecture runs one of the codes the build compiled that class for, as a loader answers.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdlib> // `std::strtol`

#include <initializer_list> // `std::initializer_list` - the ordinals a call names
#include <string>           // `std::string` - a code spelled for `std::strtol`, an outcome printed
#include <string_view>      // `std::string_view` - architectures and codes
#include <vector>           // `std::vector` - a machine, its listing and its loaded classes

#include "harness.hpp" // `expect_true`, `expect_equal`, `run_test_case`
#include "lib.hpp"     // `resolve_placement`, `gpu_placement_t`, `gpu_class_bit`

namespace unum::usearch::test {

struct profile_device_t {
    gpu_vendor_t vendor;
    std::string_view architecture;
};

using emulated_machine_t = std::vector<profile_device_t>;

static profile_device_t nvidia(std::string_view architecture) { return {gpu_vendor_t::nvidia_k, architecture}; }
static profile_device_t amd(std::string_view architecture) { return {gpu_vendor_t::amd_k, architecture}; }

struct class_codes_t {
    gpu_class_t gpu_class;
    std::string_view codes;
};

static constexpr class_codes_t build_k[] {
    {gpu_class_t::nvidia_ampere_k, "80"},       {gpu_class_t::nvidia_ada_k, "89"},
    {gpu_class_t::nvidia_hopper_k, "90a-real"}, {gpu_class_t::nvidia_blackwellrtx_k, "120-real"},
    {gpu_class_t::amd_cdna3_k, "gfx942"},       {gpu_class_t::amd_cdna4_k, "gfx950"},
    {gpu_class_t::amd_cdna5_k, "gfx1250"},
};

static bool runs(std::string_view architecture, std::string_view code) {
    if (!architecture.starts_with("sm_")) return architecture == code;
    std::string const spelled {code};
    char* suffix = nullptr;
    long const device = std::strtol(architecture.data() + 3, nullptr, 10);
    long const compiled = std::strtol(spelled.c_str(), &suffix, 10);
    if (*suffix == 'a') return device == compiled; // arch-specific codes skip forward compatibility
    // PTX runs on any later architecture, `-real` SASS only on later minors of its own major
    return device >= compiled && (std::string_view {suffix} != "-real" || device / 10 == compiled / 10);
}

static gpu_class_mask_t loaded_on(profile_device_t const& device) {
    gpu_class_mask_t loaded = 0;
    for (class_codes_t const& built : build_k) {
        if (gpu_vendor_of(built.gpu_class) != device.vendor) continue;
        for (std::string_view codes = built.codes; !codes.empty();) {
            std::size_t const end = codes.find(';');
            if (runs(device.architecture, codes.substr(0, end))) loaded |= gpu_class_bit(built.gpu_class);
            codes = end == std::string_view::npos ? std::string_view {} : codes.substr(end + 1);
        }
    }
    return loaded;
}

struct outcome_t {
    status_t status = status_t::success_k;
    gpu_class_t gpu_class = gpu_class_t::unknown_k;

    bool operator==(outcome_t const&) const noexcept = default;
};

static outcome_t outcome(gpu_class_t gpu_class) { return {status_t::success_k, gpu_class}; }
static outcome_t outcome(status_t status) { return {status, gpu_class_t::unknown_k}; }

static std::string format_as(outcome_t const& placed) {
    return succeeded(placed.status) ? fmt::format("class {}", static_cast<u32_t>(placed.gpu_class))
                                    : std::string {status_to_string(placed.status)};
}

/** Resolves @p devices on @p machine, failing a placement that breaks what it promises as an
 *  internal error. */
static outcome_t place(emulated_machine_t const& machine, std::initializer_list<std::int32_t> devices,
                       usearch_gpu_class_t ceiling = usearch_gpu_class_unknown_k) {
    std::vector<gpu_device_t> table(machine.size());
    std::vector<gpu_class_mask_t> loaded(machine.size());
    for (std::size_t position = 0; position != machine.size(); ++position)
        table[position].vendor = machine[position].vendor, loaded[position] = loaded_on(machine[position]);
    usearch_vram_options_t vram {};
    vram.devices = devices.size() ? devices.begin() : nullptr;
    vram.devices_count = devices.size();
    vram.gpu_class_ceiling = ceiling;
    auto const [placement, status] = resolve_placement(table, loaded, vram);
    if (failed(status)) return outcome(status);
    bool placed = expect_true(devices.size() ? placement.positions.data() == devices.begin() &&
                                                   placement.positions.size() == devices.size()
                                             : placement.positions.size() == 1 && placement.positions[0] == 0);
    for (std::int32_t const position : placement.positions) {
        placed = expect_true(table[position].vendor == placement.vendor) && placed;
        placed = expect_true((loaded[position] & gpu_class_bit(placement.gpu_class)) != 0) && placed;
    }
    return placed ? outcome(placement.gpu_class) : outcome(status_t::internal_k);
}

/** Three vendors' classes side by side: each pair of devices places on the newest class all of
 *  them load, a ceiling caps it, and mixed vendors, a device past the listing and foreign
 *  classes all refuse. */
static void test_placement_rtx_h100_cdna3() {
    emulated_machine_t const rtx_h100_cdna3 {nvidia("sm_120"), nvidia("sm_120"), nvidia("sm_90"),
                                             nvidia("sm_90"),  amd("gfx942"),    amd("gfx942")};
    expect_equal(place(rtx_h100_cdna3, {0, 1}), outcome(gpu_class_t::nvidia_blackwellrtx_k));
    expect_equal(place(rtx_h100_cdna3, {2, 3}), outcome(gpu_class_t::nvidia_hopper_k));
    expect_equal(place(rtx_h100_cdna3, {4, 5}), outcome(gpu_class_t::amd_cdna3_k));
    expect_equal(place(rtx_h100_cdna3, {}), outcome(gpu_class_t::nvidia_blackwellrtx_k));
    expect_equal(place(rtx_h100_cdna3, {1, 2}), outcome(gpu_class_t::nvidia_ada_k));
    expect_equal(place(rtx_h100_cdna3, {1, 2}, usearch_gpu_class_nvidia_ampere_k),
                 outcome(gpu_class_t::nvidia_ampere_k));
    expect_equal(place(rtx_h100_cdna3, {0, 4}), outcome(status_t::config_mismatch_k));
    expect_equal(place(rtx_h100_cdna3, {6}), outcome(status_t::device_error_k));
    expect_equal(place(rtx_h100_cdna3, {2}, usearch_gpu_class_amd_cdna3_k), outcome(status_t::unsupported_k));
    expect_equal(place(rtx_h100_cdna3, {2}, static_cast<usearch_gpu_class_t>(13)),
                 outcome(status_t::invalid_argument_k));
}

/** An Ampere and an Ada device place alone on their own classes, and together on the older. */
static void test_placement_a100_l40s() {
    emulated_machine_t const a100_l40s {nvidia("sm_80"), nvidia("sm_89")};
    expect_equal(place(a100_l40s, {1}), outcome(gpu_class_t::nvidia_ada_k));
    expect_equal(place(a100_l40s, {0, 1}), outcome(gpu_class_t::nvidia_ampere_k));
}

/** A CDNA2 device loads no class of the build, so only its CDNA3 neighbour places. */
static void test_placement_mi250x_mi300x() {
    emulated_machine_t const mi250x_mi300x {amd("gfx90a"), amd("gfx942")};
    expect_equal(place(mi250x_mi300x, {0}), outcome(status_t::unsupported_k));
    expect_equal(place(mi250x_mi300x, {1}), outcome(gpu_class_t::amd_cdna3_k));
}

/** A CDNA5 device places on its own class, and no class spans it and a CDNA3 neighbour. */
static void test_placement_mi400_mi300x() {
    emulated_machine_t const mi400_mi300x {amd("gfx1250"), amd("gfx942")};
    expect_equal(place(mi400_mi300x, {0}), outcome(gpu_class_t::amd_cdna5_k));
    expect_equal(place(mi400_mi300x, {0, 1}), outcome(status_t::unsupported_k));
}

} // namespace unum::usearch::test

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
    run_test_case("test_placement_rtx_h100_cdna3", test_placement_rtx_h100_cdna3);
    run_test_case("test_placement_a100_l40s", test_placement_a100_l40s);
    run_test_case("test_placement_mi250x_mi300x", test_placement_mi250x_mi300x);
    run_test_case("test_placement_mi400_mi300x", test_placement_mi400_mi300x);
    return finish_test_run();
}
