/**
 *  @file src/classes.hpp
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The CPU and GPU classes USearch ships variants for: what each lowers to in NumKong's and
 *      StringZilla's masks, which ones this build compiled, the ladders a dispatch walks, and the
 *      class a placement picks among those a caller says loaded.
 *
 *  It never asks the machine, so tests, benchmarks and device kernel units include it alone, while
 *  detection, driver probing and the waiters stay in @c lib.hpp.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t` - the ordinals a placement names

#include <span> // `std::span` - the listing a placement resolves against

#include <numkong/capabilities.h>     // `nk_capability_t` - the compute masks a class lowers to
#include <stringzilla/capabilities.h> // `sz_capability_t` - the string masks a class lowers to

#include "usearch/types.h"             // `usearch_cpu_class_t`, `usearch_gpu_class_t`, `usearch_vram_options_t`
#include "usearch/utilities/types.hpp" // `gpu_vendor_t`, `gpu_device_t`, `expected`

namespace util = unum::usearch::utilities;
namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;

#pragma region CPU Classes

/** The CPU families USearch ships tuned variants for - our own vocabulary, never an
 *  alias of a sister library's feature bits. Each class bundles the compute features,
 *  wait instruction, and atomics its silicon generation guarantees; explicit helpers
 *  lower a class onto NumKong, ForkUnion & StringZilla capability masks where those
 *  libraries are linked. */
enum class cpu_class_t : util::u32_t {

    /** No class detected yet, or none of the classes below. */
    unknown_k = 0,

    // Portable classes

    /** The portable floor: no SIMD assumptions, standard yields. */
    serial_k = usearch_cpu_class_serial_k,

    /** The one class of a monoversioned build: every capability @c USEARCH_MARCH implies or a
     *  preset vouches for, as the libraries compiled them. */
    monoversioned_k = usearch_cpu_class_monoversioned_k,

    // x86 classes

    /** Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    intel_haswell_k = usearch_cpu_class_intel_haswell_k,

    /** Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI & its INT8 / IFMA / NE-CONVERT
     *  successors without AVX-512, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_sierra_k = usearch_cpu_class_intel_sierra_k,

    /** Skylake-class x86: the AVX-512 foundation tier. */
    intel_skylake_k = usearch_cpu_class_intel_skylake_k,

    /** Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    intel_icelake_k = usearch_cpu_class_intel_icelake_k,

    /** Genoa-class x86 - Zen 4 and newer: native @c bf16 dot products over AVX-512. */
    amd_genoa_k = usearch_cpu_class_amd_genoa_k,

    /** Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 &
     *  AMX-INT8 tiles. */
    intel_sapphire_k = usearch_cpu_class_intel_sapphire_k,

    /** Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles. */
    intel_granite_k = usearch_cpu_class_intel_granite_k,

    /** Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions and
     *  the Sierra vector bundle, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_diamond_k = usearch_cpu_class_intel_diamond_k,

    // Arm classes

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; pre-Armv8.6 phones fall
     *  here too. */
    aws_g2_k = usearch_cpu_class_aws_g2_k,

    /** Armv8.6+ NEON: dot products, @c bf16 & @c f16 fused multiplies. Named for Apple M2;
     *  Cortex-A710 & X2 - Snapdragon 8 Gen 1, AmpereOne - hold the same. */
    apple_m2_k = usearch_cpu_class_apple_m2_k,

    /** Apple M4-class Arm: the @c apple_m2_k bundle plus the SME outer-product tiles,
     *  @c BMOPA among them, without plain SVE. */
    apple_m4_k = usearch_cpu_class_apple_m4_k,

    /** Graviton5-class Arm - Neoverse V3, Armv9.2: the @c apple_m2_k bundle with waits that
     *  monitor through @c WFE instead of spinning; no build compiles it yet. */
    aws_g5_k = usearch_cpu_class_aws_g5_k,

    /** Vera-class Arm - NVIDIA's Armv9.2+ cores with @c FEAT_FP8 dot products. */
    nvidia_vera_k = usearch_cpu_class_nvidia_vera_k,

    // RISC-V classes

    /** Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23: the vector extension with
     *  @c Zvbb, @c Zawrs monitored waits, @c Zacas single-instruction compare-and-swap. */
    alibaba_c930_k = usearch_cpu_class_alibaba_c930_k,

    // Power classes

    /** IBM POWER10-class Power, ISA 3.1: VSX vectors, the MMA outer-product accumulators and
     *  @c pause_short waits. */
    ibm_power10_k = usearch_cpu_class_ibm_power10_k,

    // WebAssembly classes

    /** WebAssembly every engine loads: 32-bit, one caller-inclusive thread, the fixed-width
     *  @c simd128 tier, so it runs on WebKit and in pages served without cross-origin isolation. */
    wasm32_v128_k = usearch_cpu_class_wasm32_v128_k,

    /** WebAssembly with relaxed SIMD, a shared memory for Web Worker pools and 64-bit addressing,
     *  bundled because every engine that ships one ships the other two. */
    wasm64_extended_k = usearch_cpu_class_wasm64_extended_k,
};

/** The GPU families USearch ships tuned variants for, the device twin of @c cpu_class_t: one
 *  class per generation of kernel kits, each compiled for its own architecture alone. A class
 *  implies its vendor, and an unknown newer device takes the newest class it satisfies within
 *  that vendor's classes. */
enum class gpu_class_t : util::u32_t {

    /** No class detected yet, or none of the classes below. */
    unknown_k = 0,

    // NVIDIA classes

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30. */
    nvidia_ampere_k = usearch_gpu_class_nvidia_ampere_k,

    /** Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions. */
    nvidia_ada_k = usearch_gpu_class_nvidia_ada_k,

    /** Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    nvidia_hopper_k = usearch_gpu_class_nvidia_hopper_k,

    /** Blackwell-class NVIDIA - B200, B300 & GB300: TMA @c gather4 scattered fetches
     *  and @c tcgen05 tensor-core GEMMs. */
    nvidia_blackwell_k = usearch_gpu_class_nvidia_blackwell_k,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark: Blackwell's conversions
     *  without @c tcgen05. */
    nvidia_blackwellrtx_k = usearch_gpu_class_nvidia_blackwellrtx_k,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    nvidia_rubin_k = usearch_gpu_class_nvidia_rubin_k,

    // AMD classes

    /** CDNA3 AMD - MI300X & MI325X: gfx942, 64-wide waves, FNUZ fp8. */
    amd_cdna3_k = usearch_gpu_class_amd_cdna3_k,

    /** CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8. */
    amd_cdna4_k = usearch_gpu_class_amd_cdna4_k,

    /** CDNA5 AMD - MI400: gfx1250, 32-wide waves. */
    amd_cdna5_k = usearch_gpu_class_amd_cdna5_k,

    // Apple classes

    /** Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2. */
    apple_metal7_k = usearch_gpu_class_apple_metal7_k,

    /** Apple9-family GPUs - M3 & M4. */
    apple_metal9_k = usearch_gpu_class_apple_metal9_k,

    /** Apple10-family GPUs - M5: the Neural Accelerator matrix units. */
    apple_metal10_k = usearch_gpu_class_apple_metal10_k,
};

static_assert(usearch_cpu_class_serial_k == 1 && usearch_cpu_class_monoversioned_k == 2 &&
              usearch_cpu_class_ibm_power10_k == 18);

/** Every CPU class the C enum names, in value order: the one table ceilings are checked against and
 *  names are read back from, so a value naming no class this build has is never one. */
constexpr usearch_cpu_class_t cpu_classes_k[] {
    usearch_cpu_class_serial_k,          usearch_cpu_class_monoversioned_k,  usearch_cpu_class_intel_haswell_k,
    usearch_cpu_class_intel_sierra_k,    usearch_cpu_class_intel_skylake_k,  usearch_cpu_class_intel_icelake_k,
    usearch_cpu_class_amd_genoa_k,       usearch_cpu_class_intel_sapphire_k, usearch_cpu_class_intel_granite_k,
    usearch_cpu_class_intel_diamond_k,   usearch_cpu_class_aws_g2_k,         usearch_cpu_class_apple_m2_k,
    usearch_cpu_class_apple_m4_k,        usearch_cpu_class_aws_g5_k,         usearch_cpu_class_nvidia_vera_k,
    usearch_cpu_class_alibaba_c930_k,    usearch_cpu_class_ibm_power10_k,    usearch_cpu_class_wasm32_v128_k,
    usearch_cpu_class_wasm64_extended_k,
};

/** Every GPU class the C enum names, in value order. */
constexpr usearch_gpu_class_t gpu_classes_k[] {
    usearch_gpu_class_nvidia_ampere_k,    usearch_gpu_class_nvidia_ada_k,          usearch_gpu_class_nvidia_hopper_k,
    usearch_gpu_class_nvidia_blackwell_k, usearch_gpu_class_nvidia_blackwellrtx_k, usearch_gpu_class_nvidia_rubin_k,
    usearch_gpu_class_amd_cdna3_k,        usearch_gpu_class_amd_cdna4_k,           usearch_gpu_class_amd_cdna5_k,
    usearch_gpu_class_apple_metal7_k,     usearch_gpu_class_apple_metal9_k,        usearch_gpu_class_apple_metal10_k,
};

/** Every NVMe class the C enum names, in value order. */
constexpr usearch_nvme_class_t nvme_classes_k[] {
    usearch_nvme_class_posix_k,       usearch_nvme_class_linux_aio_k,
    usearch_nvme_class_linux_uring_k, usearch_nvme_class_linux_uring_passthrough_k,
    usearch_nvme_class_linux_spdk_k,  usearch_nvme_class_windows_ioring_k,
};

/** Whether @p value is one of the entries of @p table. */
template <typename value_type_, std::size_t count_>
constexpr bool listed(value_type_ const (&table)[count_], value_type_ value) noexcept {
    for (value_type_ const entry : table)
        if (entry == value) return true;
    return false;
}

/** Lowers one CPU class onto NumKong's compute mask - cumulative within a platform, so the chains
 *  pick their best row per dtype under it. */
constexpr nk_capability_t numkong_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_haswell_k: return nk_cap_serial_k | nk_cap_haswell_k;
    case cpu_class_t::intel_sierra_k:
        return numkong_capabilities_of(cpu_class_t::intel_haswell_k) | nk_cap_alder_k | nk_cap_sierra_k;
    case cpu_class_t::intel_skylake_k: return numkong_capabilities_of(cpu_class_t::intel_haswell_k) | nk_cap_skylake_k;
    case cpu_class_t::intel_icelake_k: return numkong_capabilities_of(cpu_class_t::intel_skylake_k) | nk_cap_icelake_k;
    case cpu_class_t::amd_genoa_k: return numkong_capabilities_of(cpu_class_t::intel_icelake_k) | nk_cap_genoa_k;
    case cpu_class_t::intel_sapphire_k:
        return numkong_capabilities_of(cpu_class_t::amd_genoa_k) | nk_cap_sapphire_k | nk_cap_sapphireamx_k;
    case cpu_class_t::intel_granite_k:
        return numkong_capabilities_of(cpu_class_t::intel_sapphire_k) | nk_cap_graniteamx_k;
    case cpu_class_t::intel_diamond_k:
        return numkong_capabilities_of(cpu_class_t::intel_granite_k) | nk_cap_alder_k | nk_cap_sierra_k |
               nk_cap_diamond_k;
    case cpu_class_t::aws_g2_k: return nk_cap_serial_k | nk_cap_neon_k;
    case cpu_class_t::apple_m2_k:
    case cpu_class_t::aws_g5_k:
        return numkong_capabilities_of(cpu_class_t::aws_g2_k) | nk_cap_neonsdot_k | nk_cap_neonbfdot_k |
               nk_cap_neonfhm_k;
    case cpu_class_t::apple_m4_k:
        return numkong_capabilities_of(cpu_class_t::apple_m2_k) | nk_cap_sme_k | nk_cap_smebi32_k;
    case cpu_class_t::nvidia_vera_k: return numkong_capabilities_of(cpu_class_t::apple_m2_k) | nk_cap_neonfp8_k;
    case cpu_class_t::monoversioned_k: return nk_capabilities_compiled_cpu_mask();
    case cpu_class_t::alibaba_c930_k: return nk_cap_serial_k | nk_cap_rvv_k | nk_cap_rvvbb_k;
    case cpu_class_t::ibm_power10_k: return nk_cap_serial_k | nk_cap_powervsx_k | nk_cap_powermma_k;
    case cpu_class_t::wasm32_v128_k: return nk_cap_serial_k | nk_cap_v128_k;
    case cpu_class_t::wasm64_extended_k:
        return numkong_capabilities_of(cpu_class_t::wasm32_v128_k) | nk_cap_v128relaxed_k;
    default: return nk_cap_serial_k;
    }
}

/** The class a refused rung degrades to - the next one down its own lineage, which is the class an
 *  variant must be distinct from to earn instantiation. On x86 the AVX-512 P-cores and the AVX-VNNI
 *  E-cores share the Haswell floor. */
constexpr cpu_class_t parent_class_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_diamond_k: return cpu_class_t::intel_granite_k;
    case cpu_class_t::intel_granite_k: return cpu_class_t::intel_sapphire_k;
    case cpu_class_t::intel_sapphire_k: return cpu_class_t::amd_genoa_k;
    case cpu_class_t::amd_genoa_k: return cpu_class_t::intel_icelake_k;
    case cpu_class_t::intel_icelake_k: return cpu_class_t::intel_skylake_k;
    case cpu_class_t::intel_skylake_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::intel_sierra_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::nvidia_vera_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::aws_g5_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m4_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m2_k: return cpu_class_t::aws_g2_k;
    case cpu_class_t::wasm64_extended_k: return cpu_class_t::wasm32_v128_k;
    default: return cpu_class_t::serial_k;
    }
}

/** Lowers one CPU class onto StringZilla's capability bits: each class spells every backend its
 *  kits may pick, and a class adding none spells its parent's. */
constexpr sz_capability_t stringzilla_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::serial_k: return sz_cap_serial_k;
    case cpu_class_t::intel_haswell_k: return sz_cap_serial_k | sz_cap_haswell_k | sz_cap_westmere_k;
    case cpu_class_t::intel_skylake_k:
        return stringzilla_capabilities_of(cpu_class_t::intel_haswell_k) | sz_cap_skylake_k;
    case cpu_class_t::intel_icelake_k:
        return stringzilla_capabilities_of(cpu_class_t::intel_skylake_k) | sz_cap_icelake_k | sz_cap_goldmont_k;
    case cpu_class_t::intel_sierra_k:
        return stringzilla_capabilities_of(cpu_class_t::intel_haswell_k) | sz_cap_goldmont_k;
    case cpu_class_t::aws_g2_k:
    case cpu_class_t::apple_m2_k: return sz_cap_serial_k | sz_cap_neon_k | sz_cap_neonaes_k | sz_cap_neonsha_k;
    case cpu_class_t::alibaba_c930_k: return sz_cap_serial_k | sz_cap_rvv_k;
    case cpu_class_t::ibm_power10_k: return sz_cap_serial_k | sz_cap_powervsx_k;
    case cpu_class_t::monoversioned_k: return sz_capabilities_compiled_cpu_mask();
    default: return stringzilla_capabilities_of(parent_class_of(cpu_class));
    }
}

/** Whether the build compiled the unit of @p cpu_class, as CMake decided and told the dispatch
 *  units through @c USEARCH_TARGET_<CLASS>; a ladder never picks a class without one. */
constexpr bool cpu_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if USEARCH_TARGET_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if USEARCH_TARGET_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if USEARCH_TARGET_INTEL_SIERRA
    case cpu_class_t::intel_sierra_k: return true;
#endif
#if USEARCH_TARGET_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if USEARCH_TARGET_INTEL_ICELAKE
    case cpu_class_t::intel_icelake_k: return true;
#endif
#if USEARCH_TARGET_AMD_GENOA
    case cpu_class_t::amd_genoa_k: return true;
#endif
#if USEARCH_TARGET_INTEL_SAPPHIRE
    case cpu_class_t::intel_sapphire_k: return true;
#endif
#if USEARCH_TARGET_INTEL_GRANITE
    case cpu_class_t::intel_granite_k: return true;
#endif
#if USEARCH_TARGET_INTEL_DIAMOND
    case cpu_class_t::intel_diamond_k: return true;
#endif
#if USEARCH_TARGET_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if USEARCH_TARGET_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if USEARCH_TARGET_APPLE_M4
    case cpu_class_t::apple_m4_k: return true;
#endif
#if USEARCH_TARGET_AWS_G5
    case cpu_class_t::aws_g5_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_VERA
    case cpu_class_t::nvidia_vera_k: return true;
#endif
#if USEARCH_TARGET_ALIBABA_C930
    case cpu_class_t::alibaba_c930_k: return true;
#endif
#if USEARCH_TARGET_IBM_POWER10
    case cpu_class_t::ibm_power10_k: return true;
#endif
#if USEARCH_TARGET_WASM32_V128
    case cpu_class_t::wasm32_v128_k: return true;
#endif
#if USEARCH_TARGET_WASM64_EXTENDED
    case cpu_class_t::wasm64_extended_k: return true;
#endif
#if USEARCH_TARGET_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

/** The C enum's class as the engines name it; @c usearch_cpu_class_unknown_k derives nothing, so
 *  it names the serial floor. */
constexpr cpu_class_t cpu_class_of(usearch_cpu_class_t cpu_class) noexcept {
    if (!listed(cpu_classes_k, cpu_class)) return cpu_class_t::serial_k;
    return static_cast<cpu_class_t>(cpu_class);
}

/** CPU classes newest first, the order a ladder tries them in. */
template <cpu_class_t... classes_>
struct cpu_ladder {};

/** GPU classes of one vendor newest first, the order its ladder tries them in. */
template <gpu_class_t... classes_>
struct gpu_ladder {};

/** Rungs of a ladder this build compiled; a monoversioned build compiles at most one. */
template <cpu_class_t... classes_>
consteval std::size_t compiled_rungs(cpu_ladder<classes_...>) noexcept {
    return (std::size_t {cpu_class_compiled(classes_)} + ... + 0);
}

/** Every class holding variants, each lineage newest first and the serial floor last, or the one
 *  class of a monoversioned build; the walk, never the list, skips a rung the build left out. */
#if USEARCH_TARGET_MONOVERSIONED
using cpu_ladder_t = cpu_ladder<cpu_class_t::monoversioned_k>;
inline constexpr cpu_class_t cpu_ladder_floor_k = cpu_class_t::monoversioned_k;
static_assert(compiled_rungs(cpu_ladder_t {}) <= 1);
#else
using cpu_ladder_t =
    cpu_ladder<cpu_class_t::intel_diamond_k, cpu_class_t::intel_granite_k, cpu_class_t::intel_sapphire_k,
               cpu_class_t::amd_genoa_k, cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k,
               cpu_class_t::intel_sierra_k, cpu_class_t::intel_haswell_k, cpu_class_t::nvidia_vera_k,
               cpu_class_t::apple_m4_k, cpu_class_t::aws_g5_k, cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k,
               cpu_class_t::alibaba_c930_k, cpu_class_t::ibm_power10_k, cpu_class_t::wasm64_extended_k,
               cpu_class_t::wasm32_v128_k, cpu_class_t::serial_k>;
inline constexpr cpu_class_t cpu_ladder_floor_k = cpu_class_t::serial_k;
#endif

/** Whether the dense NVMe variants compiled a unit for @p cpu_class, as their row in
 *  @c src/CMakeLists.txt told the dispatch unit through @c USEARCH_TARGET_NVME_<CLASS>: each kept
 *  class compiles as itself or the nearest ancestor the row lists. A storage walk shares a few
 *  counters and never waits, so the classes advancing by their waiter or reference alone take none,
 *  and a WebAssembly module reads no storage. */
constexpr bool nvme_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if USEARCH_TARGET_NVME_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_SIERRA
    case cpu_class_t::intel_sierra_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_ICELAKE
    case cpu_class_t::intel_icelake_k: return true;
#endif
#if USEARCH_TARGET_NVME_AMD_GENOA
    case cpu_class_t::amd_genoa_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_SAPPHIRE
    case cpu_class_t::intel_sapphire_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_GRANITE
    case cpu_class_t::intel_granite_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_DIAMOND
    case cpu_class_t::intel_diamond_k: return true;
#endif
#if USEARCH_TARGET_NVME_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if USEARCH_TARGET_NVME_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if USEARCH_TARGET_NVME_APPLE_M4
    case cpu_class_t::apple_m4_k: return true;
#endif
#if USEARCH_TARGET_NVME_NVIDIA_VERA
    case cpu_class_t::nvidia_vera_k: return true;
#endif
#if USEARCH_TARGET_NVME_IBM_POWER10
    case cpu_class_t::ibm_power10_k: return true;
#endif
#if USEARCH_TARGET_NVME_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

/** Whether the basic modalities compiled a unit for @p cpu_class, as their rows in
 *  @c src/CMakeLists.txt told each dispatch unit through @c USEARCH_TARGET_BASIC_<CLASS>: each kept
 *  class compiles as itself or the nearest ancestor where an inlined StringZilla or NumKong kit
 *  advances over its parent's. */
constexpr bool basic_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if USEARCH_TARGET_BASIC_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if USEARCH_TARGET_BASIC_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if USEARCH_TARGET_BASIC_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if USEARCH_TARGET_BASIC_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if USEARCH_TARGET_BASIC_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if USEARCH_TARGET_BASIC_ALIBABA_C930
    case cpu_class_t::alibaba_c930_k: return true;
#endif
#if USEARCH_TARGET_BASIC_IBM_POWER10
    case cpu_class_t::ibm_power10_k: return true;
#endif
#if USEARCH_TARGET_BASIC_WASM64_EXTENDED
    case cpu_class_t::wasm64_extended_k: return true;
#endif
#if USEARCH_TARGET_BASIC_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

#pragma endregion CPU Classes

#pragma region GPU Classes

/** A set of GPU classes, one bit per C enum value. */
using gpu_class_mask_t = util::u32_t;

constexpr gpu_class_mask_t gpu_class_bit(gpu_class_t gpu_class) noexcept {
    return gpu_class_mask_t {1} << static_cast<util::u32_t>(gpu_class);
}

/** Whether the build compiled device code for @p gpu_class, as CMake decided and told the dispatch
 *  units through @c USEARCH_TARGET_<VENDOR>_<CLASS>; a ladder never picks a class without one. */
constexpr bool gpu_class_compiled(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
#if USEARCH_TARGET_NVIDIA_AMPERE
    case gpu_class_t::nvidia_ampere_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_ADA
    case gpu_class_t::nvidia_ada_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_HOPPER
    case gpu_class_t::nvidia_hopper_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_BLACKWELL
    case gpu_class_t::nvidia_blackwell_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_BLACKWELLRTX
    case gpu_class_t::nvidia_blackwellrtx_k: return true;
#endif
#if USEARCH_TARGET_AMD_CDNA3
    case gpu_class_t::amd_cdna3_k: return true;
#endif
#if USEARCH_TARGET_AMD_CDNA4
    case gpu_class_t::amd_cdna4_k: return true;
#endif
#if USEARCH_TARGET_AMD_CDNA5
    case gpu_class_t::amd_cdna5_k: return true;
#endif
#if USEARCH_TARGET_APPLE_METAL7
    case gpu_class_t::apple_metal7_k: return true;
#endif
#if USEARCH_TARGET_APPLE_METAL9
    case gpu_class_t::apple_metal9_k: return true;
#endif
#if USEARCH_TARGET_APPLE_METAL10
    case gpu_class_t::apple_metal10_k: return true;
#endif
    default: return false;
    }
}

/** The vendor whose devices @p gpu_class serves, read off the C enum's per-vendor value ranges. */
constexpr util::gpu_vendor_t gpu_vendor_of(gpu_class_t gpu_class) noexcept {
    util::u32_t const value = static_cast<util::u32_t>(gpu_class);
    return value == usearch_gpu_class_unknown_k         ? util::gpu_vendor_t::unknown_k
           : value <= usearch_gpu_class_nvidia_rubin_k  ? util::gpu_vendor_t::nvidia_k
           : value <= usearch_gpu_class_amd_cdna5_k     ? util::gpu_vendor_t::amd_k
           : value <= usearch_gpu_class_apple_metal10_k ? util::gpu_vendor_t::apple_k
                                                        : util::gpu_vendor_t::unknown_k;
}

/** The newest Apple GPU family an Apple class needs, Apple7 and Apple8 sharing the floor. */
constexpr util::u32_t apple_family_of(gpu_class_t gpu_class) noexcept {
    return gpu_class == gpu_class_t::apple_metal10_k ? 10 : gpu_class == gpu_class_t::apple_metal9_k ? 9 : 7;
}

/** The class a GPU class extends, the next one down its own lineage, or @c unknown_k past the
 *  lineage's root: the oldest class of each vendor, and CDNA5, which roots its own lineage on the
 *  ROCm floor, since it drops instructions CDNA3 and CDNA4 keep. */
constexpr gpu_class_t parent_class_of(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case gpu_class_t::nvidia_rubin_k: return gpu_class_t::nvidia_blackwell_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_blackwellrtx_k:
    case gpu_class_t::nvidia_hopper_k: return gpu_class_t::nvidia_ada_k;
    case gpu_class_t::nvidia_ada_k: return gpu_class_t::nvidia_ampere_k;
    case gpu_class_t::amd_cdna4_k: return gpu_class_t::amd_cdna3_k;
    case gpu_class_t::apple_metal10_k: return gpu_class_t::apple_metal9_k;
    case gpu_class_t::apple_metal9_k: return gpu_class_t::apple_metal7_k;
    default: return gpu_class_t::unknown_k;
    }
}

/** Whether @p gpu_class sits on the lineage of @p ceiling, the ceiling or one of its ancestors,
 *  every class when no ceiling is set. */
constexpr bool gpu_class_under_ceiling(gpu_class_t gpu_class, usearch_gpu_class_t ceiling) noexcept {
    if (ceiling == usearch_gpu_class_unknown_k) return true;
    for (gpu_class_t rung = static_cast<gpu_class_t>(ceiling); rung != gpu_class_t::unknown_k;
         rung = parent_class_of(rung))
        if (rung == gpu_class) return true;
    return false;
}

/** NVIDIA classes newest first, Rubin's unit to come. */
using nvidia_ladder_t =
    gpu_ladder<gpu_class_t::nvidia_rubin_k, gpu_class_t::nvidia_blackwellrtx_k, gpu_class_t::nvidia_blackwell_k,
               gpu_class_t::nvidia_hopper_k, gpu_class_t::nvidia_ada_k, gpu_class_t::nvidia_ampere_k>;

/** AMD classes newest first. */
using amd_ladder_t = gpu_ladder<gpu_class_t::amd_cdna5_k, gpu_class_t::amd_cdna4_k, gpu_class_t::amd_cdna3_k>;

/** Apple classes newest first. */
using apple_ladder_t =
    gpu_ladder<gpu_class_t::apple_metal10_k, gpu_class_t::apple_metal9_k, gpu_class_t::apple_metal7_k>;

/** A basic modality's multiversioned ladder, or the vendor ladder in a monoversioned build, whose
 *  one kept class holds every modality. */
#if USEARCH_TARGET_MONOVERSIONED
template <typename multiversioned_ladder_type_>
using basic_gpu_ladder = nvidia_ladder_t;
#else
template <typename multiversioned_ladder_type_>
using basic_gpu_ladder = multiversioned_ladder_type_;
#endif

/** The newest class of @p ladder at or under @p ceiling among @p loaded, or @c unknown_k. */
template <gpu_class_t... classes_>
gpu_class_t pick_gpu_class(gpu_ladder<classes_...>, gpu_class_mask_t loaded, usearch_gpu_class_t ceiling) noexcept {
    gpu_class_t picked = gpu_class_t::unknown_k;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        if (picked == gpu_class_t::unknown_k && gpu_class_under_ceiling(class_, ceiling) &&
            (loaded & gpu_class_bit(class_)))
            picked = class_;
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** The class a device of @p vendor whose loader took @p loaded takes under @p ceiling, walking
 *  its own vendor's ladder; @c unknown_k when no unit this build holds runs on it, or none at
 *  or under the ceiling does. */
inline gpu_class_t pick_gpu_class(util::gpu_vendor_t vendor, gpu_class_mask_t loaded,
                                  usearch_gpu_class_t ceiling) noexcept {
    switch (vendor) {
    case util::gpu_vendor_t::nvidia_k: return pick_gpu_class(nvidia_ladder_t {}, loaded, ceiling);
    case util::gpu_vendor_t::amd_k: return pick_gpu_class(amd_ladder_t {}, loaded, ceiling);
    case util::gpu_vendor_t::apple_k: return pick_gpu_class(apple_ladder_t {}, loaded, ceiling);
    default: return gpu_class_t::unknown_k;
    }
}

/** The devices one index places its shards on: their one vendor, the best class all of them load
 *  under the ceiling, every class they all load, and their positions in the listing @c table. */
struct gpu_placement_t {
    util::gpu_vendor_t vendor = util::gpu_vendor_t::unknown_k;
    gpu_class_t gpu_class = gpu_class_t::unknown_k;
    gpu_class_mask_t loaded = 0;
    std::span<util::gpu_device_t const> table;
    std::span<std::int32_t const> positions;

    util::gpu_device_t const& first() const noexcept { return table[static_cast<std::size_t>(positions[0])]; }
};

/**
 *  @brief Resolves the devices @p vram names, device zero when it names none, against the listing
 *      @p table, whose classes @p loaded holds as the loader answered at listing.
 *
 *  An index spans one vendor, since its shards merge through peer copies, and runs the best class
 *  all its devices load, never above the ceiling, as a CPU executor runs what all its cores share.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The ceiling is a value no class takes.
 *  - @c device_k An ordinal is past the listing.
 *  - @c config_mismatch_k The devices span vendors.
 *  - @c unsupported_k The devices share no class this build holds at or under the ceiling.
 */
inline util::expected<gpu_placement_t> resolve_placement(std::span<util::gpu_device_t const> table,
                                                         std::span<gpu_class_mask_t const> loaded,
                                                         usearch_vram_options_t const& vram) noexcept {
    static constexpr std::int32_t device_zero_k[] {0};
    if (vram.gpu_class_ceiling != usearch_gpu_class_unknown_k && !listed(gpu_classes_k, vram.gpu_class_ceiling))
        return {{}, {util::status_code_t::invalid_argument_k, "No GPU class has that value"}};
    gpu_placement_t placement;
    placement.table = table, placement.loaded = ~gpu_class_mask_t {0};
    placement.positions = vram.devices && vram.devices_count
                              ? std::span<std::int32_t const> {vram.devices, vram.devices_count}
                              : std::span<std::int32_t const> {device_zero_k};
    for (std::int32_t const ordinal : placement.positions) {
        std::size_t const position = static_cast<std::size_t>(ordinal);
        if (ordinal < 0 || position >= table.size())
            return {{}, {util::status_code_t::device_k, "A device ordinal is past the listing"}};
        util::gpu_vendor_t const vendor = table[position].vendor;
        if (placement.vendor != util::gpu_vendor_t::unknown_k && vendor != placement.vendor)
            return {{}, {util::status_code_t::config_mismatch_k, "The devices must share one vendor"}};
        placement.vendor = vendor, placement.loaded &= loaded[position];
    }
    placement.gpu_class = pick_gpu_class(placement.vendor, placement.loaded, vram.gpu_class_ceiling);
    if (placement.gpu_class == gpu_class_t::unknown_k)
        return {{}, {util::status_code_t::unsupported_k, "No carried GPU class fits the devices"}};
    return {placement, util::status_code_t::success_k};
}

/** Lowers one GPU class onto NumKong's GPU capabilities. On NVIDIA, the Ampere & Ada bits
 *  accumulate, while Hopper, Blackwell & RTX Blackwell each run only their own kits, and Rubin
 *  rides the Blackwell kits until its own land. On AMD, CDNA4 accumulates over CDNA3, while CDNA5
 *  drops the half and byte dots and roots its own lineage on the ROCm floor, so each kit lists the
 *  classes it runs on. Apple generations accumulate over their vendor's floor. */
constexpr nk_capability_t numkong_capabilities_of(gpu_class_t gpu_class) noexcept {
    nk_capability_t const ampere = nk_cap_cuda_k | nk_cap_ampere_k, ada = ampere | nk_cap_ada_k;
    nk_capability_t const cdna3 = nk_cap_rocm_k | nk_cap_cdna3_k, cdna4 = cdna3 | nk_cap_cdna4_k;
    nk_capability_t const cdna5 = nk_cap_rocm_k | nk_cap_cdna5_k;
    nk_capability_t const metal9 = nk_cap_metal_k | nk_cap_apple9_k;
    switch (gpu_class) {
    case gpu_class_t::nvidia_ada_k: return ada;
    case gpu_class_t::nvidia_hopper_k: return ada | nk_cap_hopper_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_rubin_k: return ada | nk_cap_blackwell_k;
    case gpu_class_t::nvidia_blackwellrtx_k: return ada | nk_cap_blackwellrtx_k;
    case gpu_class_t::amd_cdna3_k: return cdna3;
    case gpu_class_t::amd_cdna4_k: return cdna4;
    case gpu_class_t::amd_cdna5_k: return cdna5;
    case gpu_class_t::apple_metal7_k: return nk_cap_metal_k;
    case gpu_class_t::apple_metal9_k: return metal9;
    case gpu_class_t::apple_metal10_k: return metal9 | nk_cap_apple10_k;
    case gpu_class_t::nvidia_ampere_k: return ampere;
    case gpu_class_t::unknown_k: return nk_cap_serial_k;
    }
    return nk_cap_serial_k;
}

/** Lowers one GPU class onto StringZilla's GPU capabilities: every NVIDIA class reaches @c ampere ,
 *  and the tiers past it start at compute capability 9.0 for @c hopper and 10.0 for @c blackwell ,
 *  consumer Blackwell and Rubin included; each AMD class is its own CDNA tier, as StringZilla
 *  detects gfx942, gfx950 and gfx1250; every Apple class runs Metal's one tier. */
constexpr sz_capability_t stringzilla_capabilities_of(gpu_class_t gpu_class) noexcept {
    sz_capability_t const ampere = sz_cap_cuda_k | sz_cap_ampere_k, hopper = ampere | sz_cap_hopper_k;
    switch (gpu_class) {
    case gpu_class_t::amd_cdna3_k: return sz_cap_rocm_k | sz_cap_cdna3_k;
    case gpu_class_t::amd_cdna4_k: return sz_cap_rocm_k | sz_cap_cdna3_k | sz_cap_cdna4_k;
    case gpu_class_t::amd_cdna5_k: return sz_cap_rocm_k | sz_cap_cdna5_k;
    case gpu_class_t::nvidia_ampere_k:
    case gpu_class_t::nvidia_ada_k: return ampere;
    case gpu_class_t::nvidia_hopper_k: return hopper;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_blackwellrtx_k:
    case gpu_class_t::nvidia_rubin_k: return hopper | sz_cap_blackwell_k;
    case gpu_class_t::apple_metal7_k:
    case gpu_class_t::apple_metal9_k:
    case gpu_class_t::apple_metal10_k: return sz_cap_metal_k;
    default: return sz_cap_serial_k;
    }
}

/** The NVIDIA class a class unit and its device passes compile as, the one its CMake row sets
 *  @c USEARCH_TARGET_NVIDIA_<CLASS> for; where gates name several, the newest. */
constexpr gpu_class_t cuda_class_k =
#if USEARCH_TARGET_NVIDIA_BLACKWELLRTX
    gpu_class_t::nvidia_blackwellrtx_k;
#elif USEARCH_TARGET_NVIDIA_BLACKWELL
    gpu_class_t::nvidia_blackwell_k;
#elif USEARCH_TARGET_NVIDIA_HOPPER
    gpu_class_t::nvidia_hopper_k;
#elif USEARCH_TARGET_NVIDIA_ADA
    gpu_class_t::nvidia_ada_k;
#else
    gpu_class_t::nvidia_ampere_k;
#endif

/** The ceiling an NVIDIA device pass instantiates: its class's. */
constexpr nk_capability_t cuda_class_capabilities_k = numkong_capabilities_of(cuda_class_k);

/** The AMD class a class unit and its device passes compile as, the one its CMake row sets
 *  @c USEARCH_TARGET_AMD_<CLASS> for; where gates name several, the newest. */
constexpr gpu_class_t hip_class_k =
#if USEARCH_TARGET_AMD_CDNA5
    gpu_class_t::amd_cdna5_k;
#elif USEARCH_TARGET_AMD_CDNA4
    gpu_class_t::amd_cdna4_k;
#else
    gpu_class_t::amd_cdna3_k;
#endif

/** The ceiling an AMD device pass instantiates: its class's. */
constexpr nk_capability_t hip_class_capabilities_k = numkong_capabilities_of(hip_class_k);

template <gpu_class_t... classes_>
consteval std::size_t compiled_rungs(gpu_ladder<classes_...>) noexcept {
    return (std::size_t {gpu_class_compiled(classes_)} + ... + 0);
}

#if USEARCH_TARGET_MONOVERSIONED
static_assert(compiled_rungs(nvidia_ladder_t {}) <= 1 && compiled_rungs(amd_ladder_t {}) <= 1 &&
              compiled_rungs(apple_ladder_t {}) <= 1);
#endif

#pragma endregion GPU Classes
