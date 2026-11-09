/**
 *  @file src/types.cpp
 *  @author Ash Vardanian
 *  @date August 27, 2026
 *  @brief The library-wide probes of the C interface.
 *
 *  The functions that take no handle and cannot fail: what version this is, which kernel
 *  families it was compiled with, which of those the machine in front of it can run, and the
 *  canonical spelling of every closed set - a lookup back walks the same table, so the two
 *  directions never disagree. They live apart from any modality because every modality reports
 *  the same answer.
 *
 *  A handle's own @c capabilities is a narrower question - what dispatch picked for the geometry
 *  that handle froze - and each modality answers that one for itself.
 */
#include <cstdio> // `std::snprintf`

#include <array>       // `std::array` - a capability list rendered at compile time
#include <string_view> // `std::string_view` - a name compared without a terminator

#include <numkong/numkong.h>

#include "usearch/types.h"

#include "lib.hpp" // `cpu_ladder_t`, `numkong_capabilities_of`, `pick_cpu_ceiling`, `cpu_rung_fits`

/** Rendered once on first use, so the pointer the C interface hands out stays valid. */
static char const* rendered_version() {
    static char rendered[32] = {0};
    if (!rendered[0])
        std::snprintf(rendered, sizeof(rendered), "%d.%d.%d", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR,
                      USEARCH_VERSION_PATCH);
    return rendered;
}

/** The kernel families a capability mask can name, newest class last. */
static constexpr struct {
    nk_capability_t bit;
    char const* name;
} capability_names_k[] = {
    {nk_cap_serial_k, "serial"},     {nk_cap_neon_k, "neon"},
    {nk_cap_neonsdot_k, "neonsdot"}, {nk_cap_neonbfdot_k, "neonbfdot"},
    {nk_cap_neonfhm_k, "neonfhm"},   {nk_cap_neonfp8_k, "neonfp8"},
    {nk_cap_haswell_k, "haswell"},   {nk_cap_skylake_k, "skylake"},
    {nk_cap_icelake_k, "icelake"},   {nk_cap_genoa_k, "genoa"},
    {nk_cap_sapphire_k, "sapphire"}, {nk_cap_sapphireamx_k, "sapphireamx"},
    {nk_cap_v128_k, "v128"},         {nk_cap_v128relaxed_k, "v128relaxed"},
};

/** A comma-separated list of kernel families, terminated. */
using capabilities_name_t = std::array<char, 256>;

/** Composes @p mask as a comma-separated list of the families above, at compile time. */
static constexpr capabilities_name_t render_capabilities(nk_capability_t mask) noexcept {
    capabilities_name_t rendered {};
    std::size_t length = 0;
    for (auto const& named : capability_names_k) {
        if (!(mask & named.bit)) continue;
        if (length && length + 1 < rendered.size()) rendered[length++] = ',';
        for (char const* name = named.name; *name && length + 1 < rendered.size();) rendered[length++] = *name++;
    }
    return rendered;
}

/** The families one CPU class's kits reach. */
template <cpu_class_t cpu_class_>
inline constexpr capabilities_name_t cpu_class_capabilities_name =
    render_capabilities(numkong_capabilities_of(cpu_class_));

/** The families every compiled CPU class's kits reach together. */
static constexpr capabilities_name_t compiled_capabilities_name_k =
    []<cpu_class_t... classes_>(cpu_ladder<classes_...>) noexcept {
        return render_capabilities(((cpu_class_compiled(classes_) ? numkong_capabilities_of(classes_) : 0) | ...));
    }(cpu_ladder_t {});

/** @brief Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
static bool named(usearch_bytes_view_t name, char const* spelling) noexcept {
    return std::string_view {reinterpret_cast<char const*>(name.data), name.length} == spelling;
}

extern "C" {

USEARCH_EXPORT void usearch_version(int* major, int* minor, int* patch) {
    *major = USEARCH_VERSION_MAJOR, *minor = USEARCH_VERSION_MINOR, *patch = USEARCH_VERSION_PATCH;
}

USEARCH_EXPORT char const* usearch_version_name(void) { return rendered_version(); }

USEARCH_EXPORT char const* usearch_capabilities_compiled(void) { return compiled_capabilities_name_k.data(); }

USEARCH_EXPORT char const* usearch_capabilities_available(void) {
    cpu_ceiling_t const ceiling = pick_cpu_ceiling(usearch_cpu_class_unknown_k).value;
    return []<cpu_class_t... classes_>(cpu_ladder<classes_...>, cpu_ceiling_t machine) noexcept {
        char const* name = nullptr;
        auto rung = [&]<cpu_class_t class_>() noexcept {
            if (!name && cpu_rung_fits(class_, machine)) name = cpu_class_capabilities_name<class_>.data();
        };
        (rung.template operator()<classes_>(), ...);
        return name ? name : "";
    }(cpu_ladder_t {}, ceiling);
}

USEARCH_EXPORT char const* usearch_status_name(usearch_status_t status) {
    switch (status) {
    case usearch_status_success_k: return "success";
    case usearch_status_out_of_memory_k: return "out-of-memory";
    case usearch_status_shape_mismatch_k: return "shape-mismatch";
    case usearch_status_unsupported_layout_k: return "unsupported-layout";
    case usearch_status_unknown_kind_k: return "unknown-kind";
    case usearch_status_read_only_k: return "read-only";
    case usearch_status_io_k: return "io";
    case usearch_status_device_k: return "device";
    case usearch_status_malformed_k: return "malformed";
    case usearch_status_internal_k: return "internal";
    case usearch_status_overflow_k: return "overflow";
    case usearch_status_uninitialized_k: return "uninitialized";
    case usearch_status_config_mismatch_k: return "config-mismatch";
    case usearch_status_contended_k: return "contended";
    case usearch_status_cancelled_k: return "cancelled";
    case usearch_status_unsupported_k: return "unsupported";
    case usearch_status_deferred_k: return "deferred";
    case usearch_status_invalid_argument_k: return "invalid-argument";
    default: return "unrecognized";
    }
}

USEARCH_EXPORT usearch_status_t usearch_status_from_name(usearch_bytes_view_t name, usearch_status_t* status) {
    for (int code = usearch_status_success_k; code <= usearch_status_invalid_argument_k; ++code)
        if (named(name, usearch_status_name(static_cast<usearch_status_t>(code))))
            return *status = static_cast<usearch_status_t>(code), usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

USEARCH_EXPORT char const* usearch_scalar_name(usearch_scalar_t scalar_kind) {
    switch (scalar_kind) {
    case usearch_scalar_f64_k: return "f64";
    case usearch_scalar_f32_k: return "f32";
    case usearch_scalar_bf16_k: return "bf16";
    case usearch_scalar_f16_k: return "f16";
    case usearch_scalar_e5m2_k: return "e5m2";
    case usearch_scalar_e4m3_k: return "e4m3";
    case usearch_scalar_e3m2_k: return "e3m2";
    case usearch_scalar_e2m3_k: return "e2m3";
    case usearch_scalar_i8_k: return "i8";
    case usearch_scalar_i4x2_k: return "i4x2";
    case usearch_scalar_i32_k: return "i32";
    case usearch_scalar_u8_k: return "u8";
    case usearch_scalar_u4x2_k: return "u4x2";
    case usearch_scalar_u1x8_k: return "u1x8";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_scalar_t usearch_scalar_from_name(usearch_bytes_view_t name) {
    // The enumerators jump over the gaps their families leave, and every gap spells "unknown", so
    // a counted walk would read that spelling back as a code naming no scalar.
    usearch_scalar_t const kinds[] = {
        usearch_scalar_f64_k,  usearch_scalar_f32_k,  usearch_scalar_bf16_k, usearch_scalar_f16_k,
        usearch_scalar_e5m2_k, usearch_scalar_e4m3_k, usearch_scalar_e3m2_k, usearch_scalar_e2m3_k,
        usearch_scalar_i8_k,   usearch_scalar_i4x2_k, usearch_scalar_i32_k,  usearch_scalar_u8_k,
        usearch_scalar_u4x2_k, usearch_scalar_u1x8_k,
    };
    for (usearch_scalar_t const scalar_kind : kinds)
        if (named(name, usearch_scalar_name(scalar_kind))) return scalar_kind;
    return usearch_scalar_unknown_k;
}

USEARCH_EXPORT char const* usearch_residency_name(usearch_residency_t residency) {
    switch (residency) {
    case usearch_residency_ram_k: return "ram";
    case usearch_residency_vram_k: return "vram";
    case usearch_residency_nvme_k: return "nvme";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_status_t usearch_residency_from_name(usearch_bytes_view_t name, usearch_residency_t* residency) {
    for (int code = usearch_residency_ram_k; code <= usearch_residency_nvme_k; ++code)
        if (named(name, usearch_residency_name(static_cast<usearch_residency_t>(code))))
            return *residency = static_cast<usearch_residency_t>(code), usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

USEARCH_EXPORT char const* usearch_traversal_name(usearch_traversal_t traversal) {
    switch (traversal) {
    case usearch_traversal_walk_k: return "walk";
    case usearch_traversal_scan_k: return "scan";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_status_t usearch_traversal_from_name(usearch_bytes_view_t name, usearch_traversal_t* traversal) {
    if (named(name, "walk")) return *traversal = usearch_traversal_walk_k, usearch_status_success_k;
    if (named(name, "scan")) return *traversal = usearch_traversal_scan_k, usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

USEARCH_EXPORT char const* usearch_load_path_name(usearch_load_path_t load_path) {
    switch (load_path) {
    case usearch_load_path_none_k: return "none";
    case usearch_load_path_host_copy_k: return "host-copy";
    case usearch_load_path_cufile_k: return "cufile";
    default: return "unknown";
    }
}

USEARCH_EXPORT char const* usearch_mood_name(usearch_mood_t mood) {
    switch (mood) {
    case usearch_mood_parked_k: return "parked";
    case usearch_mood_spinning_k: return "spinning";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_status_t usearch_mood_from_name(usearch_bytes_view_t name, usearch_mood_t* mood) {
    if (named(name, "parked")) return *mood = usearch_mood_parked_k, usearch_status_success_k;
    if (named(name, "spinning")) return *mood = usearch_mood_spinning_k, usearch_status_success_k;
    return usearch_status_invalid_argument_k;
}

USEARCH_EXPORT char const* usearch_cpu_class_name(usearch_cpu_class_t cpu_class) {
    switch (cpu_class) {
    case usearch_cpu_class_serial_k: return "serial";
    case usearch_cpu_class_intel_haswell_k: return "intel-haswell";
    case usearch_cpu_class_intel_sierra_k: return "intel-sierra";
    case usearch_cpu_class_intel_skylake_k: return "intel-skylake";
    case usearch_cpu_class_intel_icelake_k: return "intel-icelake";
    case usearch_cpu_class_amd_genoa_k: return "amd-genoa";
    case usearch_cpu_class_intel_sapphire_k: return "intel-sapphire";
    case usearch_cpu_class_intel_granite_k: return "intel-granite";
    case usearch_cpu_class_intel_diamond_k: return "intel-diamond";
    case usearch_cpu_class_aws_g2_k: return "aws-g2";
    case usearch_cpu_class_apple_m2_k: return "apple-m2";
    case usearch_cpu_class_apple_m4_k: return "apple-m4";
    case usearch_cpu_class_aws_g5_k: return "aws-g5";
    case usearch_cpu_class_nvidia_vera_k: return "nvidia-vera";
    case usearch_cpu_class_alibaba_c930_k: return "alibaba-c930";
    case usearch_cpu_class_wasm32_v128_k: return "wasm32-v128";
    case usearch_cpu_class_wasm64_extended_k: return "wasm64-extended";
    case usearch_cpu_class_monoversioned_k: return "monoversioned";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_cpu_class_t usearch_cpu_class_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_cpu_class_serial_k; code <= usearch_cpu_class_monoversioned_k; ++code) {
        usearch_cpu_class_t const cpu_class = static_cast<usearch_cpu_class_t>(code);
        if (named(name, usearch_cpu_class_name(cpu_class))) return cpu_class;
    }
    return usearch_cpu_class_unknown_k;
}

USEARCH_EXPORT char const* usearch_gpu_class_name(usearch_gpu_class_t gpu_class) {
    switch (gpu_class) {
    case usearch_gpu_class_nvidia_ampere_k: return "nvidia-ampere";
    case usearch_gpu_class_nvidia_ada_k: return "nvidia-ada";
    case usearch_gpu_class_nvidia_hopper_k: return "nvidia-hopper";
    case usearch_gpu_class_nvidia_blackwell_k: return "nvidia-blackwell";
    case usearch_gpu_class_nvidia_blackwellrtx_k: return "nvidia-blackwellrtx";
    case usearch_gpu_class_nvidia_rubin_k: return "nvidia-rubin";
    case usearch_gpu_class_amd_cdna3_k: return "amd-cdna3";
    case usearch_gpu_class_amd_cdna4_k: return "amd-cdna4";
    case usearch_gpu_class_amd_cdna5_k: return "amd-cdna5";
    case usearch_gpu_class_apple_metal7_k: return "apple-metal7";
    case usearch_gpu_class_apple_metal9_k: return "apple-metal9";
    case usearch_gpu_class_apple_metal10_k: return "apple-metal10";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_gpu_class_t usearch_gpu_class_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_gpu_class_nvidia_ampere_k; code <= usearch_gpu_class_apple_metal10_k; ++code) {
        usearch_gpu_class_t const gpu_class = static_cast<usearch_gpu_class_t>(code);
        if (named(name, usearch_gpu_class_name(gpu_class))) return gpu_class;
    }
    return usearch_gpu_class_unknown_k;
}

USEARCH_EXPORT char const* usearch_nvme_class_name(usearch_nvme_class_t nvme_class) {
    switch (nvme_class) {
    case usearch_nvme_class_posix_k: return "posix";
    case usearch_nvme_class_linux_aio_k: return "linux-aio";
    case usearch_nvme_class_linux_uring_k: return "linux-uring";
    case usearch_nvme_class_linux_uring_passthrough_k: return "linux-uring-passthrough";
    case usearch_nvme_class_linux_spdk_k: return "linux-spdk";
    case usearch_nvme_class_windows_ioring_k: return "windows-ioring";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_nvme_class_t usearch_nvme_class_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_nvme_class_posix_k; code <= usearch_nvme_class_windows_ioring_k; ++code) {
        usearch_nvme_class_t const nvme_class = static_cast<usearch_nvme_class_t>(code);
        if (named(name, usearch_nvme_class_name(nvme_class))) return nvme_class;
    }
    return usearch_nvme_class_unknown_k;
}

} // extern "C"
