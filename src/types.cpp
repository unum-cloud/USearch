/**
 *  @file src/types.cpp
 *  @author Ash Vardanian
 *  @date August 27, 2026
 *  @brief The library-wide probes of the C interface - the twin of `include/usearch/types.h`.
 *
 *  The functions that take no handle and cannot fail: what version this is, which kernel
 *  families it was compiled with, which of those the machine in front of it can run, and the
 *  canonical spelling of every closed set - a lookup back walks the same table, so the two
 *  directions never disagree. They live apart from any modality because every modality reports
 *  the same answer.
 *
 *  A handle's own `capabilities` is a narrower question - what dispatch picked for the geometry
 *  that handle froze - and each modality answers that one for itself.
 */
#include <cstdio>      // `std::snprintf`
#include <string_view> // `std::string_view` - a name compared without a terminator

#include <usearch/types.h>

#include <numkong/numkong.h>

namespace {

/** Rendered once on first use, so the pointer the C interface hands out stays valid. */
char const* rendered_version() {
    static char rendered[32] = {0};
    if (!rendered[0])
        std::snprintf(rendered, sizeof(rendered), "%d.%d.%d", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR,
                      USEARCH_VERSION_PATCH);
    return rendered;
}

/** The kernel families a capability mask can name, newest class last. */
constexpr struct {
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

/** Composes @p mask into @p rendered as a comma-separated list, once. */
char const* render_capabilities(char* rendered, size_t capacity, nk_capability_t mask) {
    if (rendered[0]) return rendered;
    char* cursor = rendered;
    char* const sentinel = rendered + capacity - 1;
    for (auto const& named : capability_names_k) {
        if (!(mask & named.bit)) continue;
        if (cursor != rendered && cursor != sentinel) *cursor++ = ',';
        for (char const* name = named.name; *name && cursor != sentinel;) *cursor++ = *name++;
    }
    *cursor = '\0';
    return rendered;
}

/** @brief Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
bool named(usearch_bytes_view_t name, char const* spelling) noexcept {
    return std::string_view {reinterpret_cast<char const*>(name.data), name.length} == spelling;
}

} // namespace

extern "C" {

USEARCH_EXPORT char const* usearch_version(void) { return rendered_version(); }

USEARCH_EXPORT char const* usearch_capabilities_compiled(void) {
    static char rendered[256] = {0};
    return render_capabilities(rendered, sizeof(rendered), nk_capabilities_compiled());
}

USEARCH_EXPORT char const* usearch_capabilities_available(void) {
    static char rendered[256] = {0};
    return render_capabilities(rendered, sizeof(rendered), nk_capabilities_available());
}

USEARCH_EXPORT char const* usearch_status_name(usearch_status_t status) {
    switch (status) {
    case usearch_success_k: return "success";
    case usearch_error_out_of_memory_k: return "out-of-memory";
    case usearch_error_shape_mismatch_k: return "shape-mismatch";
    case usearch_error_unsupported_layout_k: return "unsupported-layout";
    case usearch_error_unknown_kind_k: return "unknown-kind";
    case usearch_error_read_only_k: return "read-only";
    case usearch_error_io_k: return "io";
    case usearch_error_device_k: return "device";
    case usearch_error_malformed_input_k: return "malformed-input";
    case usearch_error_internal_k: return "internal";
    case usearch_error_overflow_k: return "overflow";
    case usearch_error_uninitialized_k: return "uninitialized";
    case usearch_error_config_mismatch_k: return "config-mismatch";
    case usearch_error_contended_k: return "contended";
    case usearch_error_cancelled_k: return "cancelled";
    case usearch_error_unsupported_k: return "unsupported";
    default: return "unrecognized";
    }
}

USEARCH_EXPORT usearch_status_t usearch_status_from_name(usearch_bytes_view_t name) {
    for (int code = usearch_success_k; code <= usearch_error_unsupported_k; ++code) {
        usearch_status_t const status = static_cast<usearch_status_t>(code);
        if (named(name, usearch_status_name(status))) return status;
    }
    return usearch_error_internal_k;
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
        usearch_scalar_i8_k,   usearch_scalar_i4x2_k, usearch_scalar_u8_k,   usearch_scalar_u4x2_k,
        usearch_scalar_u1x8_k,
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

USEARCH_EXPORT usearch_residency_t usearch_residency_from_name(usearch_bytes_view_t name) {
    if (named(name, "vram")) return usearch_residency_vram_k;
    if (named(name, "nvme")) return usearch_residency_nvme_k;
    return usearch_residency_ram_k;
}

USEARCH_EXPORT char const* usearch_cpu_class_name(usearch_cpu_class_t cpu_class) {
    switch (cpu_class) {
    case usearch_cpu_class_serial_k: return "serial";
    case usearch_cpu_class_intel_haswell_k: return "intel-haswell";
    case usearch_cpu_class_intel_sierra_k: return "intel-sierra";
    case usearch_cpu_class_intel_skylake_k: return "intel-skylake";
    case usearch_cpu_class_intel_icelake_k: return "intel-icelake";
    case usearch_cpu_class_amd_genoa_k: return "amd-genoa";
    case usearch_cpu_class_intel_diamond_k: return "intel-diamond";
    case usearch_cpu_class_aws_g2_k: return "aws-g2";
    case usearch_cpu_class_apple_m2_k: return "apple-m2";
    case usearch_cpu_class_aws_g5_k: return "aws-g5";
    case usearch_cpu_class_nvidia_vera_k: return "nvidia-vera";
    case usearch_cpu_class_alibaba_c930_k: return "alibaba-c930";
    default: return "unknown";
    }
}

USEARCH_EXPORT char const* usearch_gpu_class_name(usearch_gpu_class_t gpu_class) {
    switch (gpu_class) {
    case usearch_gpu_class_nvidia_ampere_k: return "nvidia-ampere";
    case usearch_gpu_class_nvidia_hopper_k: return "nvidia-hopper";
    case usearch_gpu_class_nvidia_blackwell_k: return "nvidia-blackwell";
    case usearch_gpu_class_nvidia_blackwellrtx_k: return "nvidia-blackwellrtx";
    case usearch_gpu_class_nvidia_rubin_k: return "nvidia-rubin";
    default: return "unknown";
    }
}

} // extern "C"
