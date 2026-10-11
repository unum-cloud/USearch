/**
 *  @file src/dense_index.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense modality's C surface: the variant grid over every compiled backend, the handle
 *      owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Compiled twice: as @c dense_index.cpp for the portable library and as @c dense_index.cu for
 *  the CUDA one, with @c USEARCH_BUILD_CUDA deciding whether the VRAM arms join the variant and
 *  @c USEARCH_DENSE_DTYPES how many storage dtypes every class contributes to it.
 *  Arms are ordered newest-capability-first inside each dtype cell, since a machine
 *  reporting Ice Lake also reports Haswell and Skylake; the serial floor is last and
 *  unconditional. The arms' bodies compile in the per-class units, never here.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cmath>   // `std::sqrt` - attention's default logit scale
#include <cstdio>  // `std::snprintf` - composing the capability string
#include <cstring> // `std::memcpy`, `std::strlen` - scattering staged rows & the capability string

#include <atomic>      // `std::atomic` - serializing the progress ticks
#include <bit>         // `std::bit_cast` - the softmax carry crossing the ABI
#include <mutex>       // `std::lock_guard` over the ticks' lock
#include <memory>      // `std::unique_ptr` - the executor's topology
#include <new>         // `std::nothrow`
#include <span>        // `std::span` - key batches crossing the ABI
#include <string_view> // `std::string_view` - checkpoint paths
#include <type_traits> // `std::is_same_v` - the variant-membership audit
#include <variant>     // `std::variant`, `std::visit`

#include <forkunion.h>            // `fu_topology_new` - the executor factories' discovery
#include <forkunion/types.hpp>    // `fu::spin_mutex` - the ticks' lock
#include <numkong/capabilities.h> // `nk_cpu_capabilities_enabled`
#include <numkong/cast.h>         // `nk_cast_best` - converting foreign scalar rows at the boundary

#include "dense_specializations.cuh"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator`, `no_memory_domain_k` - the multi-indexes' host tables

#pragma region Multi Index

/**
 *  @brief The one dynamically-typed dense index every binding consumes: a variant over the
 *      compiled multi-index alternatives, plus the geometry & capability metadata no shard
 *      answers.
 *
 *  Arms exist per wired @c (dtype,capability) cell - those whose NumKong kernel kits are
 *  adapted - never the full cross product: an unwired cell would not compile, and advertising
 *  it would only move the failure from the build to the caller. Each arm keeps its metric as a
 *  runtime field, since only the distance epilogue reads it. The VRAM
 *  siblings join under the CUDA build; the NVMe siblings join with the block-storage
 *  executor, and until then their residency refuses at construction.
 */
struct usearch_dense_index_handle_t {

    using variants_t = std::variant<
#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_HASWELL
        // intel_haswell over serial by haswell: every dtype, 11 arms
        ram_index<nk_bf16_k, cpu_class_t::intel_haswell_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::intel_haswell_k>, //
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
        // intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd: bf16 u1 e4m3 f16 e5m2 i4 u4 e3m2, 8 arms
        ram_index<nk_bf16_k, cpu_class_t::intel_sierra_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::intel_sierra_k>, //
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
    // intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd, alder, sierra: i8 u8 e2m3, 3 arms
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::intel_sierra_k>, //
#endif
#endif
#if NUMKONG_TARGET_SKYLAKE
        // intel_skylake over intel_haswell by skylake: bf16 e4m3 f16 e5m2 e3m2 e2m3, 6 arms
        ram_index<nk_bf16_k, cpu_class_t::intel_skylake_k>, //
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::intel_skylake_k>, //
#endif
#endif
#if NUMKONG_TARGET_ICELAKE
    // intel_icelake over intel_skylake by icelake: u1 i8 u8 i4 u4, 5 arms
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::intel_icelake_k>, //
#endif
#endif
#if NUMKONG_TARGET_GENOA
        // amd_genoa over intel_icelake by genoa: bf16 e4m3 e5m2, 3 arms
        ram_index<nk_bf16_k, cpu_class_t::amd_genoa_k>, //
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::amd_genoa_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::amd_genoa_k>, //
#endif
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        // intel_sapphire over amd_genoa by sapphireamx: bf16 i8 e4m3 u8 e5m2 e3m2 e2m3, 7 arms
        ram_index<nk_bf16_k, cpu_class_t::intel_sapphire_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::intel_sapphire_k>, //
#endif
#endif
#if NUMKONG_TARGET_GRANITEAMX
    // intel_granite over intel_sapphire by graniteamx: f16 e5m2, 2 arms
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::intel_granite_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::intel_granite_k>, //
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
        // intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd: bf16 u1 f16 i4 u4 e3m2, 6 arms
        ram_index<nk_bf16_k, cpu_class_t::intel_diamond_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
    // intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, diamond: e4m3 e5m2, 2 arms
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
    // intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, alder, sierra: i8 u8 e2m3, 3 arms
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif
#elif NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEON
        // aws_g2 over serial by neon: bf16 u1 f16, 3 arms
        ram_index<nk_bf16_k, cpu_class_t::aws_g2_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::aws_g2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::aws_g2_k>, //
#endif
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONBFDOT
        // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonbfdot: bf16, 1 arm
        ram_index<nk_bf16_k, cpu_class_t::apple_m2_k>, //
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
    // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonfhm: e4m3 f16 e5m2, 3 arms
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::apple_m2_k>, //
#endif
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
    // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonsdot: i8 u8 i4 u4 e3m2 e2m3, 6 arms
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::apple_m2_k>, //
#endif
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
    // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc: u1, 1 arm
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::apple_m2_k>, //
#endif
#endif
#if NUMKONG_TARGET_SME
        // apple_m4 over apple_m2 by sme: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms
        ram_index<nk_bf16_k, cpu_class_t::apple_m4_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::apple_m4_k>, //
#endif
#endif
#if NUMKONG_TARGET_SMEBI32
    // apple_m4 over apple_m2 by smebi32: u1, 1 arm
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::apple_m4_k>, //
#endif
#endif
#if FORKUNION_TARGET_ARM64_WFET
        // aws_g5 over apple_m2 by arm64_wfet: every dtype, 11 arms
        ram_index<nk_bf16_k, cpu_class_t::aws_g5_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::aws_g5_k>, //
#endif
#endif
#if NUMKONG_TARGET_NEONFP8
    // nvidia_vera over aws_g5 by neonfp8: e4m3 e5m2 e3m2 e2m3, 4 arms
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::nvidia_vera_k>, //
#endif
#endif
#elif NUMKONG_ARCH_RISCV64_
#if FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
        // alibaba_c930 over serial by risc5_wrs, risc5_zacas: every dtype, 11 arms
        ram_index<nk_bf16_k, cpu_class_t::alibaba_c930_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::alibaba_c930_k>, //
#endif
#endif
#elif NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128
        // wasm32_v128 over serial by v128: bf16 u1 i8 u8, 4 arms
        ram_index<nk_bf16_k, cpu_class_t::wasm32_v128_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::wasm32_v128_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::wasm32_v128_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::wasm32_v128_k>, //
#endif
#endif
#if NUMKONG_TARGET_V128RELAXED
        // wasm64_extended over wasm32_v128 by v128relaxed: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms
        ram_index<nk_bf16_k, cpu_class_t::wasm64_extended_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::wasm64_extended_k>, //
#endif
#endif
#endif
#if USEARCH_BUILD_CUDA
        // nvidia_blackwellrtx over nvidia_ampere by "120f": every cell, 11 arms
        vram_index<nk_bf16_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_index<nk_u1_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_index<nk_i8_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_index<nk_e4m3_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_index<nk_f16_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_index<nk_u8_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_index<nk_e5m2_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_index<nk_i4_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_index<nk_u4_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_index<nk_e3m2_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_index<nk_e2m3_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
        // nvidia_blackwell over nvidia_ampere by "100f": every cell, 11 arms
        vram_index<nk_bf16_k, gpu_class_t::nvidia_blackwell_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_index<nk_u1_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_index<nk_i8_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_index<nk_e4m3_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_index<nk_f16_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_index<nk_u8_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_index<nk_e5m2_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_index<nk_i4_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_index<nk_u4_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_index<nk_e3m2_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_index<nk_e2m3_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
        // nvidia_hopper over nvidia_ampere by "90a": every cell, 11 arms
        vram_index<nk_bf16_k, gpu_class_t::nvidia_hopper_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_index<nk_u1_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_index<nk_i8_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_index<nk_e4m3_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_index<nk_f16_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_index<nk_u8_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_index<nk_e5m2_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_index<nk_i4_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_index<nk_u4_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_index<nk_e3m2_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_index<nk_e2m3_k, gpu_class_t::nvidia_hopper_k>, //
#endif
        // nvidia_ada over nvidia_ampere by "89": every cell, 11 arms
        vram_index<nk_bf16_k, gpu_class_t::nvidia_ada_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_index<nk_u1_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_index<nk_i8_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_index<nk_e4m3_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_index<nk_f16_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_index<nk_u8_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_index<nk_e5m2_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_index<nk_i4_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_index<nk_u4_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_index<nk_e3m2_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_index<nk_e2m3_k, gpu_class_t::nvidia_ada_k>, //
#endif
        // nvidia_ampere, the floor by "80", its PTX serving every GPU without a class: every cell, 11 arms
        vram_index<nk_bf16_k, gpu_class_t::nvidia_ampere_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_index<nk_u1_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_index<nk_i8_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_index<nk_e4m3_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_index<nk_f16_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_index<nk_u8_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_index<nk_e5m2_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_index<nk_i4_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_index<nk_u4_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_index<nk_e3m2_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_index<nk_e2m3_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#endif
    // The serial floor: every wired cell, every platform, the standard yield. The bf16 rung every build keeps
    // closes the list, so no trailing comma ever depends on the ladder.
#if USEARCH_DENSE_DTYPES >= 2
        ram_index<nk_u1_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_index<nk_i8_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_index<nk_e4m3_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_index<nk_f16_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_index<nk_u8_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_index<nk_e5m2_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_index<nk_i4_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_index<nk_u4_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_index<nk_e3m2_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_index<nk_e2m3_k, cpu_class_t::serial_k>, //
#endif
        ram_index<nk_bf16_k, cpu_class_t::serial_k>>; //

    /** The chosen backend - one alternative of the compiled grid, selected once at construction.
     *  Residency, quantization and compute tier all live in the alternative's own type. */
    variants_t variants;

    /** The dimensionality the init arguments requested, consumed by the first @c reserve; once any shard
     *  froze real geometry, the variant answers and this stops mattering. */
    std::size_t requested_dimensions = 0;

    /** The neighbors-per-node bound the init arguments requested, consumed like the dimensionality. */
    connectivity_t requested_connectivity = connectivity_32_k;

    /** The dispatched tiers, composed once at construction - @c usearch_dense_describe hands out
     *  this buffer, so it must outlive every call. */
    char capabilities_string[128] = {};

    template <typename variant_type_>
    explicit usearch_dense_index_handle_t(std::in_place_type_t<variant_type_> tag, nk_kernel_kind_t kernel,
                                          forkunion_allocator_t allocator) noexcept
        : variants(tag, kernel, allocator) {}
};

/** The arms' explicit instantiation declarations: every body compiles in its class's unit, so this
 *  unit only visits. */
#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_HASWELL
/* intel_haswell over serial by haswell: every dtype, 11 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::intel_haswell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
/* intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd: bf16 u1 e4m3 f16 e5m2 i4 u4 e3m2, 8 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::intel_sierra_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::intel_sierra_k>>;
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd, alder, sierra: i8 u8 e2m3, 3 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::intel_sierra_k>>;
#endif
#endif
#if NUMKONG_TARGET_SKYLAKE
/* intel_skylake over intel_haswell by skylake: bf16 e4m3 f16 e5m2 e3m2 e2m3, 6 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::intel_skylake_k>>;
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#endif
#if NUMKONG_TARGET_ICELAKE
/* intel_icelake over intel_skylake by icelake: u1 i8 u8 i4 u4, 5 arms */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::intel_icelake_k>>;
#endif
#endif
#if NUMKONG_TARGET_GENOA
/* amd_genoa over intel_icelake by genoa: bf16 e4m3 e5m2, 3 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::amd_genoa_k>>;
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::amd_genoa_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::amd_genoa_k>>;
#endif
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
/* intel_sapphire over amd_genoa by sapphireamx: bf16 i8 e4m3 u8 e5m2 e3m2 e2m3, 7 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::intel_sapphire_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#endif
#if NUMKONG_TARGET_GRANITEAMX
/* intel_granite over intel_sapphire by graniteamx: f16 e5m2, 2 arms */
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::intel_granite_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::intel_granite_k>>;
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
/* intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd: bf16 u1 f16 i4 u4 e3m2, 6 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::intel_diamond_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
/* intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, diamond: e4m3 e5m2, 2 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, alder, sierra: i8 u8 e2m3, 3 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif
#elif NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEON
/* aws_g2 over serial by neon: bf16 u1 f16, 3 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::aws_g2_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::aws_g2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::aws_g2_k>>;
#endif
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONBFDOT
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonbfdot: bf16, 1 arm */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::apple_m2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonfhm: e4m3 f16 e5m2, 3 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::apple_m2_k>>;
#endif
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonsdot: i8 u8 i4 u4 e3m2 e2m3, 6 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::apple_m2_k>>;
#endif
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc: u1, 1 arm */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::apple_m2_k>>;
#endif
#endif
#if NUMKONG_TARGET_SME
/* apple_m4 over apple_m2 by sme: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::apple_m4_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::apple_m4_k>>;
#endif
#endif
#if NUMKONG_TARGET_SMEBI32
/* apple_m4 over apple_m2 by smebi32: u1, 1 arm */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::apple_m4_k>>;
#endif
#endif
#if FORKUNION_TARGET_ARM64_WFET
/* aws_g5 over apple_m2 by arm64_wfet: every dtype, 11 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::aws_g5_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::aws_g5_k>>;
#endif
#endif
#if NUMKONG_TARGET_NEONFP8
/* nvidia_vera over aws_g5 by neonfp8: e4m3 e5m2 e3m2 e2m3, 4 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#endif
#elif NUMKONG_ARCH_RISCV64_
#if FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
/* alibaba_c930 over serial by risc5_wrs, risc5_zacas: every dtype, 11 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::alibaba_c930_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::alibaba_c930_k>>;
#endif
#endif
#elif NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128
/* wasm32_v128 over serial by v128: bf16 u1 i8 u8, 4 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::wasm32_v128_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::wasm32_v128_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::wasm32_v128_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::wasm32_v128_k>>;
#endif
#endif
#if NUMKONG_TARGET_V128RELAXED
/* wasm64_extended over wasm32_v128 by v128relaxed: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::wasm64_extended_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::wasm64_extended_k>>;
#endif
#endif
#endif
#if USEARCH_BUILD_CUDA
/* nvidia_blackwellrtx over nvidia_ampere by "120f": every cell, 11 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
/* nvidia_blackwell over nvidia_ampere by "100f": every cell, 11 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_blackwell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
/* nvidia_hopper over nvidia_ampere by "90a": every cell, 11 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_hopper_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_hopper_k>>;
#endif
/* nvidia_ada over nvidia_ampere by "89": every cell, 11 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_ada_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_ada_k>>;
#endif
/* nvidia_ampere, the floor by "80", its PTX serving every GPU without a class: every cell, 11 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_ampere_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#endif
/* The serial floor: every wired cell, every platform, the standard yield. */
extern template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::serial_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::serial_k>>;
#endif

/** Whether @p alternative_type_ is listed in @p variant_type_ - the membership fact the ladder
 *  asserts against the explicit list. */
template <typename alternative_type_, typename variant_type_>
inline constexpr bool variant_carries = false;
template <typename alternative_type_, typename... alternative_types_>
inline constexpr bool variant_carries<alternative_type_, std::variant<alternative_types_...>> =
    (std::is_same_v<alternative_type_, alternative_types_> || ...);

#pragma endregion Multi Index

#pragma region Construction

/** Appends one RAM shard per named memory domain - or one per allocatable domain of a transient
 *  topology when the list is empty, so a bare init still spans the machine. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
static status_t place_shards(multi_type_& variant, usearch_ram_options_t const& ram) noexcept {
    if (ram.memory_domains && ram.memory_domains_count) {
        for (std::size_t position = 0; position != ram.memory_domains_count; ++position)
            if (status_t const placed = specialized_index {variant}.emplace(
                    static_cast<fu_memory_domain_id_t>(ram.memory_domains[position]));
                failed(placed))
                return placed;
        return status_t::success_k;
    }
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k) return status_t::uninitialized_k;
    std::size_t domains_count = 0;
    status_t placed = status_t::uninitialized_k;
    if (fu_memory_domains_count(topology, &domains_count) == fu_success_k && domains_count)
        placed = status_t::success_k;
    for (std::size_t position = 0; position != domains_count && succeeded(placed); ++position) {
        fu_memory_domain_id_t domain = -1;
        if (fu_memory_domain_id_at_index(topology, position, &domain) != fu_success_k)
            placed = status_t::uninitialized_k;
        else placed = specialized_index {variant}.emplace(domain);
    }
    fu_topology_delete(topology);
    return placed;
}

/** Appends one comma-separated codename to the handle's capability string. */
static void append_capability(usearch_dense_index_handle_t& handle, char const* name) noexcept {
    std::size_t const used = std::strlen(handle.capabilities_string);
    std::snprintf(handle.capabilities_string + used, sizeof(handle.capabilities_string) - used, "%s%s", used ? "," : "",
                  name);
}

/** Fills the handle's capability string once, from the picked NumKong kit. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
static void compose_capabilities(usearch_dense_index_handle_t& handle, multi_type_ const&) noexcept {
    constexpr nk_capability_t picked =
        dense_quads_chain<typename multi_type_::scalar_t, multi_type_::capability_k>::picked_k;
    for (numkong_capability_name_t const& named : numkong_capability_names_k)
        if (picked & named.bit) append_capability(handle, named.name);
}

#if USEARCH_BUILD_CUDA

/** Appends one VRAM shard per named device ordinal; an empty list takes ordinal zero. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static status_t place_shards(multi_type_& variant, usearch_vram_options_t const& vram) noexcept {
    if (!vram.devices || !vram.devices_count) return specialized_index {variant}.emplace(0);
    for (std::size_t position = 0; position != vram.devices_count; ++position)
        if (status_t const placed = specialized_index {variant}.emplace(static_cast<int>(vram.devices[position]));
            failed(placed))
            return placed;
    return status_t::success_k;
}

/** Fills the handle's capability string once, from the resident device's generation. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static void compose_capabilities(usearch_dense_index_handle_t& handle, multi_type_& variant) noexcept {
    if (!specialized_index {variant}.shards_count()) return;
    auto const [specs, status] = cuda_device_specs_t::query(specialized_index {variant}.device(0));
    if (failed(status)) return;
    gpu_class_t const running = gpu_class_of(specs.arch());
    for (gpu_class_name_t const& named : gpu_class_names_k)
        if (named.gpu_class == running) return append_capability(handle, named.name);
}
#endif

/** The host allocator a multi-index keeps its tables on: the first named memory domain on RAM,
 *  the heap's domain otherwise. */
template <typename placement_type_>
static forkunion_allocator_t host_allocator_of(placement_type_ const& placement) noexcept {
    if constexpr (std::is_same_v<placement_type_, usearch_ram_options_t>)
        if (placement.memory_domains && placement.memory_domains_count)
            return forkunion_allocator_t {static_cast<fu_memory_domain_id_t>(placement.memory_domains[0])};
    return forkunion_allocator_t {no_memory_domain_k};
}

/** Allocates the handle holding @p variant_type_ and places its shards. The constructors below
 *  finish @p outcome only on failure, since a load still has work to do on success. */
template <typename variant_type_, typename placement_type_>
static usearch_dense_index_t make_punned_dense_index(usearch_dense_metadata_t const& geometry,
                                                     placement_type_ const& placement, nk_kernel_kind_t kernel,
                                                     usearch_outcome_t* outcome) noexcept {
    usearch_dense_index_t const handle = new (std::nothrow)
        usearch_dense_index_handle_t(std::in_place_type_t<variant_type_>(), kernel, host_allocator_of(placement));
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    handle->requested_dimensions = geometry.dimensions;
    handle->requested_connectivity = geometry.connectivity ? static_cast<connectivity_t>(geometry.connectivity)
                                                           : connectivity_32_k;
    // Placement is init's job, so a bad domain fails here rather than at the first reservation.
    variant_type_& variant = std::get<variant_type_>(handle->variants);
    if (status_t const placed = place_shards(variant, placement); failed(placed))
        return delete handle, propagate_error(placed, outcome), nullptr;
    compose_capabilities(*handle, variant);
    return handle;
}

/** Whether one rung is genuinely distinct from the class it degrades to: a different picked 4-way
 *  or tiles kit, or a different ForkUnion mask - the one fact justifying its instantiation. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
consteval bool rung_advances() noexcept {
    constexpr cpu_class_t parent_k = parent_class_of(cpu_class_);
    using scalar_t = typename nk::type_for<dtype_>::type;
    using upper_chain_t = dense_quads_chain<scalar_t, numkong_capabilities_of(cpu_class_)>;
    using lower_chain_t = dense_quads_chain<scalar_t, numkong_capabilities_of(parent_k)>;
    using upper_tiles_t = dense_tiles_chain<scalar_t, numkong_capabilities_of(cpu_class_)>;
    using lower_tiles_t = dense_tiles_chain<scalar_t, numkong_capabilities_of(parent_k)>;
    return upper_chain_t::picked_k != lower_chain_t::picked_k || upper_tiles_t::picked_k != lower_tiles_t::picked_k ||
           forkunion_capabilities_of<cpu_class_>() != forkunion_capabilities_of<parent_k>();
}

/** The unconditional serial floor - every ladder's last rung. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_kernel_kind_t kernel,
                                                nk_capability_t, usearch_outcome_t* outcome) noexcept {
    return make_punned_dense_index<ram_index<dtype_, cpu_class_t::serial_k>>(geometry, ram, kernel, outcome);
}

/** One distinct rung: taken when the machine reports the class, else falls through. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
    requires(rung_advances<dtype_, cpu_class_>())
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_kernel_kind_t kernel,
                                                nk_capability_t available, usearch_outcome_t* outcome) noexcept {
    if (numkong_admits(available, cpu_class_) && forkunion_admits<cpu_class_>())
        return make_punned_dense_index<ram_index<dtype_, cpu_class_>>(geometry, ram, kernel, outcome);
    return descend_ram_ladder<dtype_, lower_classes_...>(geometry, ram, kernel, available, outcome);
}

/** A rung whose class picks nothing new: skipped without instantiating its index. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
    requires(!rung_advances<dtype_, cpu_class_>())
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_kernel_kind_t kernel,
                                                nk_capability_t available, usearch_outcome_t* outcome) noexcept {
    static_assert(!variant_carries<ram_index<dtype_, cpu_class_>, usearch_dense_index_handle_t::variants_t>,
                  "This rung no longer advances past the one below - drop its arm from the variant list");
    return descend_ram_ladder<dtype_, lower_classes_...>(geometry, ram, kernel, available, outcome);
}

/** Picks the RAM variant for one dtype, walking by @p kernel: CPU classes newest-first, each rung
 *  compiled only where it advances past the class it degrades to. A refused rung falls to the next
 *  in the list, so lineages that share a floor interleave. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_ram_options_t const& ram,
                                        nk_kernel_kind_t kernel, usearch_outcome_t* outcome) noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_ram_ladder<dtype_, cpu_class_t::intel_diamond_k, cpu_class_t::intel_granite_k,
                              cpu_class_t::intel_sapphire_k, cpu_class_t::amd_genoa_k, cpu_class_t::intel_icelake_k,
                              cpu_class_t::intel_skylake_k, cpu_class_t::intel_sierra_k,
                              cpu_class_t::intel_haswell_k>( //
        geometry, ram, kernel, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_ram_ladder<dtype_, cpu_class_t::nvidia_vera_k, cpu_class_t::apple_m4_k, cpu_class_t::aws_g5_k,
                              cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>( //
        geometry, ram, kernel, available, outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_ram_ladder<dtype_, cpu_class_t::alibaba_c930_k>(geometry, ram, kernel, available, outcome);
#elif NUMKONG_ARCH_WASM_
    return descend_ram_ladder<dtype_, cpu_class_t::wasm64_extended_k, cpu_class_t::wasm32_v128_k>( //
        geometry, ram, kernel, available, outcome);
#else
    return descend_ram_ladder<dtype_>(geometry, ram, kernel, available, outcome);
#endif
}

#if USEARCH_BUILD_CUDA

/** The generation every named device shares - the one each shard's cubins must be built for - or
 *  @c unknown_k when a device cannot be queried or two devices differ. */
static gpu_class_t shared_gpu_class(usearch_vram_options_t const& vram) noexcept {
    bool const named = vram.devices && vram.devices_count;
    gpu_class_t shared = gpu_class_t::unknown_k;
    for (std::size_t position = 0; position != (named ? vram.devices_count : 1); ++position) {
        auto const [specs, status] = cuda_device_specs_t::query(named ? static_cast<int>(vram.devices[position]) : 0);
        if (failed(status)) return gpu_class_t::unknown_k;
        gpu_class_t const device_class = gpu_class_of(specs.arch());
        if (position && device_class != shared) return gpu_class_t::unknown_k;
        shared = device_class;
    }
    return shared;
}

/** The VRAM sibling of the RAM @ref make_index: the devices' generation picks the class, whose unit
 *  alone carries cubins for them, the way the CPU ladder picks the class the machine reports; every
 *  dtype has a kit there, and @ref with_metric admits only the metrics its family scores by. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_vram_options_t const& vram,
                                        nk_kernel_kind_t kernel, usearch_outcome_t* outcome) noexcept {
    switch (shared_gpu_class(vram)) {
    case gpu_class_t::nvidia_blackwellrtx_k:
        return make_punned_dense_index<vram_index<dtype_, gpu_class_t::nvidia_blackwellrtx_k>>(geometry, vram, kernel,
                                                                                               outcome);
    case gpu_class_t::nvidia_blackwell_k:
        return make_punned_dense_index<vram_index<dtype_, gpu_class_t::nvidia_blackwell_k>>(geometry, vram, kernel,
                                                                                            outcome);
    case gpu_class_t::nvidia_hopper_k:
        return make_punned_dense_index<vram_index<dtype_, gpu_class_t::nvidia_hopper_k>>(geometry, vram, kernel,
                                                                                         outcome);
    case gpu_class_t::nvidia_ada_k:
        return make_punned_dense_index<vram_index<dtype_, gpu_class_t::nvidia_ada_k>>(geometry, vram, kernel, outcome);
    case gpu_class_t::nvidia_ampere_k:
        return make_punned_dense_index<vram_index<dtype_, gpu_class_t::nvidia_ampere_k>>(geometry, vram, kernel,
                                                                                         outcome);
    default:
        return propagate_error(usearch_status_device_k, outcome,
                               "The devices could not be queried, or do not share one GPU generation"),
               nullptr;
    }
}
#else

/** The portable build's stand-in: every VRAM request lands on one static message. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const&, usearch_vram_options_t const&,
                                        nk_kernel_kind_t, usearch_outcome_t* outcome) noexcept {
    return propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends"), nullptr;
}
#endif

/** Routes a numeric dtype's metric onto its three wired kernels, built where @p placement names. */
template <nk_dtype_t dtype_, typename placement_type_>
    requires(dtype_ != nk_u1_k)
static usearch_dense_index_t with_metric(usearch_dense_metadata_t const& geometry, placement_type_ const& placement,
                                         usearch_outcome_t* outcome) noexcept {
    switch (geometry.metric) {
    case usearch_dense_metric_inner_k: return make_index<dtype_>(geometry, placement, nk_kernel_dot_k, outcome);
    case usearch_dense_metric_angular_k: return make_index<dtype_>(geometry, placement, nk_kernel_angular_k, outcome);
    case usearch_dense_metric_euclidean_k:
        return make_index<dtype_>(geometry, placement, nk_kernel_euclidean_k, outcome);
    default:
        return propagate_error(usearch_status_unknown_kind_k, outcome, "This metric needs a bit-packed quantization"),
               nullptr;
    }
}

/** Routes the bit-packed dtype's metric onto the bit kernels. */
template <nk_dtype_t dtype_, typename placement_type_>
    requires(dtype_ == nk_u1_k)
static usearch_dense_index_t with_metric(usearch_dense_metadata_t const& geometry, placement_type_ const& placement,
                                         usearch_outcome_t* outcome) noexcept {
    switch (geometry.metric) {
    case usearch_dense_metric_inner_k: return make_index<dtype_>(geometry, placement, nk_kernel_dot_k, outcome);
    case usearch_dense_metric_hamming_k: return make_index<dtype_>(geometry, placement, nk_kernel_hamming_k, outcome);
    case usearch_dense_metric_jaccard_k: return make_index<dtype_>(geometry, placement, nk_kernel_jaccard_k, outcome);
    default:
        return propagate_error(usearch_status_unknown_kind_k, outcome,
                               "No kernel kit is wired for this (quantization,metric) pair"),
               nullptr;
    }
}

/** Lowers the runtime @c (quantization,metric) pair onto the wired cells - every numeric dtype
 *  speaks inner, angular & euclidean; the bit dtype speaks inner, hamming & jaccard. A storage
 *  dtype the build's @c USEARCH_DENSE_DTYPES left out is unsupported rather than unknown. */
template <typename placement_type_>
static usearch_dense_index_t with_cell(usearch_dense_metadata_t const& geometry, placement_type_ const& placement,
                                       usearch_outcome_t* outcome) noexcept {
    switch (geometry.quantization) {
    case usearch_scalar_bf16_k: return with_metric<nk_bf16_k>(geometry, placement, outcome);
#if USEARCH_DENSE_DTYPES >= 2
    case usearch_scalar_u1x8_k: return with_metric<nk_u1_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    case usearch_scalar_i8_k: return with_metric<nk_i8_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 4
    case usearch_scalar_e4m3_k: return with_metric<nk_e4m3_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    case usearch_scalar_f16_k: return with_metric<nk_f16_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    case usearch_scalar_u8_k: return with_metric<nk_u8_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    case usearch_scalar_e5m2_k: return with_metric<nk_e5m2_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    case usearch_scalar_i4x2_k: return with_metric<nk_i4_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 9
    case usearch_scalar_u4x2_k: return with_metric<nk_u4_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 10
    case usearch_scalar_e3m2_k: return with_metric<nk_e3m2_k>(geometry, placement, outcome);
#endif
#if USEARCH_DENSE_DTYPES >= 11
    case usearch_scalar_e2m3_k: return with_metric<nk_e2m3_k>(geometry, placement, outcome);
#endif
    default: break;
    }
    // Only f32 & f64 cross the boundary without ever being stored, so any other scalar the table knows was left out.
    nk_dtype_t const dtype = nk_dtype_of(geometry.quantization);
    if (dtype == nk_dtype_unknown_k || dtype == nk_f32_k || dtype == nk_f64_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unsupported quantization"), nullptr;
    return propagate_error(usearch_status_unsupported_k, outcome,
                           "This build's USEARCH_DENSE_DTYPES leaves the quantization out"),
           nullptr;
}

#pragma endregion Construction

#pragma region Multi Verbs

/** The state behind @c usearch_executor_t: the discovered topology a RAM executor was spawned over,
 *  and one residency's executor - the variant is the kind tag, so no second word can disagree. The
 *  executor is declared last, so its pool dies before the topology. */
struct usearch_executor_handle_t {
    std::unique_ptr<void, decltype(&fu_topology_delete)> topology {nullptr, &fu_topology_delete};
    std::variant<forkunion_executor_t
#if USEARCH_BUILD_CUDA
                 ,
                 cuda_executor_t
#endif
                 >
        executor;
};

/** Discovers the topology and spawns @p threads over it into @p handle, answering @p handle, or null
 *  once the outcome records why not; one thread is the caller. */
static usearch_executor_t spawn_ram_executor(usearch_executor_handle_t& handle, std::size_t threads,
                                             usearch_outcome_t* outcome) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_status_device_k, outcome, "Topology discovery failed"), nullptr;
    handle.topology.reset(topology);
    auto [spawned, status] = forkunion_executor_t::spawn(topology, threads);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    handle.executor = std::move(spawned);
    return &handle;
}

/** The executor a null handle runs RAM work on: the calling thread, built into @p call_executor. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
static usearch_executor_t call_executor_for(multi_type_&, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    return spawn_ram_executor(call_executor, 1, outcome);
}

#if USEARCH_BUILD_CUDA

/** The executor a null handle runs VRAM work on: the default streams of the index's devices. */
template <typename multi_type_>
    requires(!is_ram_index<multi_type_>)
static usearch_executor_t call_executor_for(multi_type_& variant, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    specialized_index const shards {variant};
    growing_buffer<int> devices {global_allocator};
    if (failed(devices.resize(shards.shards_count())))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), nullptr;
    for (std::size_t shard = 0; shard != devices.size(); ++shard) devices[shard] = shards.device(shard);
    auto [built, status] = cuda_executor_t::on_default_streams(devices);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor = std::move(built);
    return &call_executor;
}
#endif

/** The typed executor behind the C handle, or null with @c usearch_status_device_k on a kind
 *  mismatch. A null handle builds @p call_executor for this call alone. */
template <typename multi_type_>
static typename multi_type_::executor_t* executor_of(multi_type_& variant, usearch_executor_t executor,
                                                     usearch_executor_handle_t& call_executor,
                                                     usearch_outcome_t* outcome) noexcept {
    if (!executor) executor = call_executor_for(variant, call_executor, outcome);
    if (!executor) return nullptr;
    typename multi_type_::executor_t* const typed = std::get_if<typename multi_type_::executor_t>(&executor->executor);
    if (!typed)
        propagate_error(usearch_status_device_k, outcome, "The executor's kind must match the handle's residency");
    return typed;
}

/** What @c usearch_arena_t owns: host memory that one call at a time rewinds and refills. */
struct usearch_arena_handle_t {
    growing_arena<forkunion_allocator_t> memory {forkunion_allocator_t {no_memory_domain_k}};
    fu::spin_mutex_t busy;
};

/** Claims the caller's arena for one call: creates it behind a null pointee, else rewinds it, which
 *  frees the previous call's outputs, so an @p input still inside it is refused. The caller unlocks
 *  @c busy before its final store. */
static usearch_arena_handle_t* claim_arena(usearch_arena_t* arena, void const* input,
                                           usearch_outcome_t* outcome) noexcept {
    if (!arena)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The arena pointer must not be null"),
               nullptr;
    if (!*arena) *arena = new (std::nothrow) usearch_arena_handle_t {};
    if (!*arena)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the arena"), nullptr;
    if (!(*arena)->busy.lock_if_free())
        return propagate_error(usearch_status_contended_k, outcome, "The arena serves another call"), nullptr;
    if (input && (*arena)->memory.contains(input))
        return (*arena)->busy.unlock(),
               propagate_error(usearch_status_invalid_argument_k, outcome, "An input lies inside the arena it rewinds"),
               nullptr;
    (*arena)->memory.clear();
    return *arena;
}

/** Bytes one packed row of @p dimensions spans in @p dtype. */
static std::size_t packed_row_bytes(nk_dtype_t dtype, std::size_t dimensions) noexcept {
    return divide_round_up<std::size_t>(nk_dtype_bits(dtype) * dimensions, 8);
}

/** The export tail every row reader shares: @p fill produces up to @c vectors.capacity rows in the
 *  stored type and reports how many, landing in the caller's own plane when the two types agree and
 *  in a staged one converted row by row otherwise. */
template <typename scalar_type_, typename fill_type_>
static expected<std::size_t> export_rows(usearch_rows_span_t const& vectors, std::size_t dimensions,
                                         nk_dtype_t stored_dtype, fill_type_&& fill) noexcept {
    std::size_t const wanted = vectors.capacity;
    nk_dtype_t const target_dtype = nk_dtype_of(vectors.scalar);
    if (target_dtype == nk_dtype_unknown_k) return {0, status_t::unknown_kind_k};
    if (vectors.columns != dimensions) return {0, status_t::shape_mismatch_k};
    if (vectors.stride < packed_row_bytes(target_dtype, dimensions)) return {0, status_t::unsupported_layout_k};
    if (target_dtype == stored_dtype) {
        nk::matrix_span<scalar_type_> const rows {static_cast<char*>(vectors.data),
                                                  rows_shape_of<scalar_type_>(wanted, vectors.stride, dimensions)};
        return {fill(rows), status_t::success_k};
    }
    auto zeros = nk::matrix<scalar_type_>::zeros({wanted, dimensions});
    if (!zeros)
        return {0, zeros.status == nk::status_t::bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k};
    nk::matrix<scalar_type_> staged = std::move(zeros.value);
    std::size_t const exported = fill(staged.span());
    for (std::size_t row = 0; row != min_of(exported, wanted); ++row) {
        [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
            staged.span().row(row).as_vector().byte_data(), stored_dtype,
            static_cast<char*>(vectors.data) + row * vectors.stride, target_dtype, dimensions, nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    }
    return {exported, status_t::success_k};
}

/**
 *  @brief The one loop every row verb drives on an arm that serves it: converts @p rows chunk by
 *      chunk and hands each chunk to the sink @p make_sink builds.
 *
 *  Chunks hold whole queries of @p rows_per_query rows, and progress counts queries. The sink is built
 *  once per call, so its staging is too, and it writes its own outputs. Finishes @p outcome only on
 *  failure, leaving the success to the verb once its last output is written.
 */
template <typename variant_type_, typename make_sink_type_>
static bool for_each_dense_chunk(variant_type_& variant, usearch_executor_t executor_handle,         //
                                 std::size_t rows_count, usearch_rows_view_t const& rows,            //
                                 std::size_t rows_per_query, std::size_t expansion_factor,           //
                                 usearch_progress_callback_t progress, void* progress_state,         //
                                 usearch_dense_traversal_stats_t* stats, usearch_outcome_t* outcome, //
                                 make_sink_type_&& make_sink) noexcept {
    if (stats) *stats = {};
    if (!rows_count) return true;
    if (!rows.data) return propagate_error(usearch_status_invalid_argument_k, outcome, "The rows must name storage");
    nk_dtype_t const source_dtype = nk_dtype_of(rows.scalar);
    if (source_dtype == nk_dtype_unknown_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind");
    usearch_executor_handle_t call_executor;
    typename variant_type_::executor_t* const executor = executor_of(variant, executor_handle, call_executor, outcome);
    if (!executor) return false;
    using scalar_t = typename variant_type_::scalar_t;
    typename variant_type_::search_config_t search_config {};
    search_config.expansion_factor = static_cast<u32_t>(expansion_factor);

    std::size_t const dimensions = specialized_index {variant}.dimensions();
    if (rows.stride && rows.stride < packed_row_bytes(source_dtype, dimensions))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row");

    // Rows that stage convert chunk by chunk, never splitting a query.
    std::size_t const chunk = rows_per_query *
                              max_of<std::size_t>(1, min_of(rows_count, increment_rows_k) / rows_per_query);
    status_t staged = status_t::success_k;
    auto sink = make_sink(chunk, staged);
    if (failed(staged)) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to stage the outputs");
    nk::matrix<scalar_t> staged_rows;
    if (stages_rows<variant_type_>(source_dtype, rows.stride)) {
        auto zeros = nk::matrix<scalar_t>::zeros({chunk, dimensions});
        if (!zeros)
            return propagate_error(zeros.status == nk::status_t::bad_alloc_k ? usearch_status_out_of_memory_k
                                                                             : usearch_status_shape_mismatch_k,
                                   outcome, "Failed to stage the converted rows");
        staged_rows = std::move(zeros.value);
    }

    char const* const rows_base = static_cast<char const*>(rows.data);
    std::size_t resolved = 0;
    while (resolved != rows_count) {
        std::size_t const taking = min_of(chunk, rows_count - resolved);
        nk::matrix_view<scalar_t> const chunk_rows = stored_rows_of<variant_type_>(
            rows_base + resolved * rows.stride, taking, rows.stride, source_dtype, dimensions, staged_rows);
        expected<typename variant_type_::search_result_t> const found = sink(
            specialized_index {variant}, chunk_rows, *executor, search_config, resolved, taking);
        if (!crossed(found, stats, outcome)) return false;
        resolved += taking;
        if (progress &&
            progress(resolved / rows_per_query, rows_count / rows_per_query, progress_state) == usearch_cancel_k)
            return propagate_error(usearch_status_cancelled_k, outcome, "The handler stopped the call");
    }
    return true;
}

/** Whether @p matches cannot take @p count rows of its @c wanted matches; the refusal lands in @p outcome. */
static bool rejects_matches(usearch_matches_span_t const& matches, std::size_t count,
                            usearch_outcome_t* outcome) noexcept {
    std::size_t const wanted = matches.wanted;
    if (matches.capacity < count)
        return !propagate_error(usearch_status_shape_mismatch_k, outcome, "The matches hold fewer rows than the call");
    if (!matches.counts || (wanted && (!matches.keys || !matches.scores)))
        return !propagate_error(usearch_status_invalid_argument_k, outcome, "The output planes must name storage");
    if (wanted && (matches.keys_stride < wanted * sizeof(usearch_key_t) ||
                   matches.scores_stride < wanted * sizeof(usearch_distance_t)))
        return !propagate_error(usearch_status_unsupported_layout_k, outcome, "An output stride is shorter than a row");
    return false;
}

/** Widens @p staged, the engine's scores tagged @p staged_dtype, row by row into the ABI's doubles. */
template <typename score_type_>
static void land_scores(nk::matrix_view<score_type_> staged, nk_dtype_t staged_dtype,
                        nk::matrix_span<f64_t> scores) noexcept {
    for (std::size_t row = 0; staged.extent(1) && row != staged.extent(0); ++row) {
        [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
            staged.row(row).data(), staged_dtype, scores.row(row).data(), nk_f64_k, staged.extent(1), nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    }
}

/** The matching verbs' sink on one serving arm: keys land in the caller's plane as they are, while
 *  @p score_type_ scores, tagged @p score_dtype, stage at the engine's width and widen to the ABI's
 *  double; @p rows_per_query input rows share one output row, and @p count output rows land. */
template <typename score_type_, typename variant_type_, typename verb_type_>
static void match_dense_rows(variant_type_& variant, usearch_executor_t executor,                //
                             std::size_t count, usearch_rows_view_t const& rows,                 //
                             std::size_t rows_per_query, usearch_matches_span_t const& matches,  //
                             nk_dtype_t score_dtype, std::size_t expansion_factor,               //
                             usearch_progress_callback_t progress, void* progress_state,         //
                             usearch_dense_traversal_stats_t* stats, usearch_outcome_t* outcome, //
                             verb_type_&& verb) noexcept {
    using key_t = typename variant_type_::key_t;
    std::size_t const wanted = matches.wanted;
    if (rejects_matches(matches, count, outcome)) return;
    nk::matrix_span<key_t> const keys {reinterpret_cast<char*>(matches.keys),
                                       rows_shape_of<key_t>(count, matches.keys_stride, wanted)};
    nk::matrix_span<f64_t> const scores {reinterpret_cast<char*>(matches.scores),
                                         rows_shape_of<f64_t>(count, matches.scores_stride, wanted)};
    auto make_sink = [&](std::size_t chunk, status_t& staged) noexcept {
        growing_buffer<score_type_> staged_scores {global_allocator};
        if (failed(staged_scores.resize(chunk / rows_per_query * wanted))) staged = status_t::bad_alloc_k;
        return [&, staged_scores = std::move(staged_scores)](
                   specialized_index<variant_type_> view, typename variant_type_::vectors_view_t chunk_rows,
                   typename variant_type_::executor_t& executor, typename variant_type_::search_config_t config,
                   std::size_t resolved, std::size_t taking) mutable noexcept {
            std::size_t const first = resolved / rows_per_query, matched = taking / rows_per_query;
            nk::range_t const landing = nk::range(first, first + matched);
            nk::matrix_span<score_type_> const chunk_scores {staged_scores.data(), matched, wanted};
            expected<typename variant_type_::search_result_t> const found = verb(
                view, chunk_rows, executor, config, keys(landing, nk::slice), chunk_scores,
                std::span<std::size_t> {matches.counts + first, matched});
            if (!found) return found;
            land_scores<score_type_>(chunk_scores, score_dtype, scores(landing, nk::slice));
            return found;
        };
    };
    if (for_each_dense_chunk(variant, executor, count * rows_per_query, rows, rows_per_query, expansion_factor,
                             progress, progress_state, stats, outcome, make_sink))
        propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Multi Verbs

/** @brief The cursor of a RAM arm; a device arm reads its rows through a search and has none. */
template <typename variant_type_>
struct cursor_of {
    using type = std::monostate;
};
template <typename variant_type_>
    requires is_ram_index<variant_type_>
struct cursor_of<variant_type_> {
    using type = typename variant_type_::cursor_t;
};

/** @brief The cursors of every arm of @p variants_type_, as one variant of their own. */
template <typename variants_type_>
struct cursors_of;
template <typename... alternatives_>
struct cursors_of<std::variant<alternatives_...>> {
    using type = std::variant<std::monostate, typename cursor_of<alternatives_>::type...>;
};

/** What @c usearch_dense_cursor_t owns: the walk state of one arm, with no hold on its index. */
struct usearch_dense_cursor_handle_t {
    typename cursors_of<usearch_dense_index_handle_t::variants_t>::type walk;

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    usearch_dense_index_handle_t const* opened_on;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The run read as text - a path, which is bytes that happen to decode. The ABI spells a byte
 *  @c uint8_t and the language spells it @c char, so the two never meet implicitly. */
static std::string_view as_text(usearch_bytes_view_t bytes) noexcept {
    return {reinterpret_cast<char const*>(bytes.data), bytes.length};
}

/** The run read as the engine's bytes - a buffer the ABI spells in @c uint8_t. */
static bytes_view_t as_bytes(usearch_bytes_view_t bytes) noexcept {
    return {reinterpret_cast<byte_t const*>(bytes.data), bytes.length};
}

/** Fills every field a checkpoint froze; the type carries nothing a checkpoint cannot say. */
static void adopt_metadata(safegraphs_metadata_t const& source, usearch_dense_metadata_t* metadata) noexcept {
    metadata->dimensions = regular_width(source.value_shape).value_or(0);
    metadata->connectivity = source.connectivity;
    // Our own checkpoints name the stored scalar outright; a foreign file speaks only its column dtype.
    std::string_view const quantization = source.quantization.empty() ? dtype_name(source.value_dtype)
                                                                      : source.quantization.view();
    metadata->quantization = usearch_scalar_of(nk_dtype_named(quantization.data(), quantization.size()));
    std::string_view const metric = source.metric.view();
    metadata->metric = usearch_metric_of(nk_kernel_named(metric.data(), metric.size()));
}

/** Builds a handle from the geometry @p source froze, placed where @p placement names, and copies
 *  @p source into it; a failed or cancelled load frees the handle and leaves @c *index null. */
template <typename source_type_, typename placement_type_>
static void load_dense_index(source_type_ source, placement_type_ const& placement, usearch_executor_t executor,
                             usearch_progress_callback_t progress, void* progress_state, usearch_dense_index_t* index,
                             usearch_outcome_t* outcome) noexcept {
    propagate_pending(outcome);
    *index = nullptr;
    if (source.empty())
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The checkpoint must not be empty"), void();
    auto const [checkpoint, read_status] = safegraphs_read_metadata(source);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_dense_metadata_t geometry {};
    adopt_metadata(checkpoint, &geometry);
    usearch_dense_index_t const loaded = with_cell(geometry, placement, outcome);
    if (!loaded) return;
    bool const filled = std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const typed = executor_of(variant, executor, call_executor, outcome);
            if (!typed) return false;
            shard_ticks_t ticks {progress, progress_state, specialized_index {variant}.shards_count()};
            status_t const status = specialized_index {variant}.load(source, *typed, ticks);
            return succeeded(status) || propagate_error(status, outcome);
        },
        loaded->variants);
    if (!filled) return delete loaded, void();
    *index = loaded;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_dense_init_ram(usearch_dense_init_ram_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->index = nullptr;
    usearch_dense_metadata_t const geometry {arguments->metric, arguments->quantization, arguments->dimensions,
                                             arguments->connectivity};
    usearch_dense_index_t const index = with_cell(geometry, arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_init_vram(usearch_dense_init_vram_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->index = nullptr;
    usearch_dense_metadata_t const geometry {arguments->metric, arguments->quantization, arguments->dimensions,
                                             arguments->connectivity};
    usearch_dense_index_t const index = with_cell(geometry, arguments->vram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_view_ram(usearch_dense_view_ram_t* arguments) {
    propagate_pending(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, arguments->outcome,
                    "RAM shards copy a checkpoint in and cannot serve a mapping in place yet");
}

USEARCH_EXPORT void usearch_dense_view_nvme(usearch_dense_view_nvme_t* arguments) {
    propagate_pending(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, arguments->outcome,
                    "NVMe checkpoints carry no keys and no shard wrapper, so no C verb answers on them yet");
}

USEARCH_EXPORT void usearch_dense_free(usearch_dense_index_t index) { delete index; }

USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_reserve_t* arguments) {
    usearch_dense_index_t const index = arguments->index;
    propagate_pending(arguments->outcome);
    std::visit(
        [&](auto& variant) {
            propagate_error(specialized_index {variant}.reserve(
                                {arguments->capacity, index->requested_connectivity, index->requested_dimensions}),
                            arguments->outcome);
        },
        index->variants);
}

USEARCH_EXPORT void usearch_dense_clear(usearch_dense_clear_t* arguments) {
    propagate_pending(arguments->outcome);
    propagate_error(
        std::visit([](auto& variant) { return specialized_index {variant}.clear(); }, arguments->index->variants),
        arguments->outcome);
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT char const* usearch_dense_metric_name(usearch_dense_metric_t metric) {
    switch (metric) {
    case usearch_dense_metric_inner_k: return "inner";
    case usearch_dense_metric_angular_k: return "angular";
    case usearch_dense_metric_euclidean_k: return "euclidean";
    case usearch_dense_metric_hamming_k: return "hamming";
    case usearch_dense_metric_jaccard_k: return "jaccard";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_dense_metric_t usearch_dense_metric_from_name(usearch_bytes_view_t name) {
    std::string_view const spelled {reinterpret_cast<char const*>(name.data), name.length};
    for (int code = usearch_dense_metric_inner_k; code <= usearch_dense_metric_jaccard_k; ++code) {
        usearch_dense_metric_t const metric = static_cast<usearch_dense_metric_t>(code);
        if (spelled == usearch_dense_metric_name(metric)) return metric;
    }
    return usearch_dense_metric_unknown_k;
}

USEARCH_EXPORT void usearch_dense_describe(usearch_dense_describe_t* arguments) {
    usearch_dense_index_t const index = arguments->index;
    propagate_pending(arguments->outcome);
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            specialized_index const view {variant};
            std::size_t const dimensions = view.dimensions(), connectivity = view.connectivity();
            *arguments->description = {
                {usearch_metric_of(variant.kernel()), usearch_scalar_of(variant_type_::vector_dtype_k),
                 dimensions ? dimensions : index->requested_dimensions,
                 connectivity ? connectivity : static_cast<std::size_t>(index->requested_connectivity)},
                view.size(),
                view.capacity(),
                view.memory_usage(),
                index->capabilities_string};
        },
        index->variants);
    propagate_error(usearch_status_success_k, arguments->outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_count(usearch_dense_count_t* arguments) {
    propagate_pending(arguments->outcome);
    *arguments->entries = std::visit(
        [&](auto& variant) { return specialized_index {variant}.count(uuid_of(arguments->key)); },
        arguments->index->variants);
    propagate_error(usearch_status_success_k, arguments->outcome, nullptr);
}

#pragma endregion Introspection

#pragma region Insertion and Search

/** Refuses @p count rows of @p columns values unless that is the geometry @p index froze - its
 *  shards' once reserved, and the init's until then. */
static bool rejects_columns(usearch_dense_index_handle_t& index, std::size_t count, std::size_t columns,
                            usearch_outcome_t* outcome) noexcept {
    std::size_t const reserved = std::visit(
        [](auto& variant) noexcept { return specialized_index {variant}.dimensions(); }, index.variants);
    std::size_t const frozen = reserved ? reserved : index.requested_dimensions;
    return count && columns != frozen &&
           !propagate_error(usearch_status_shape_mismatch_k, outcome,
                            "The rows' width differs from the frozen geometry");
}

USEARCH_EXPORT void usearch_dense_insert(usearch_dense_insert_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    usearch_dense_traversal_stats_t* const stats = arguments->stats;
    usearch_keys_view_t const keys = arguments->keys;
    usearch_rows_view_t const vectors = arguments->vectors;
    usearch_progress_callback_t const progress = arguments->progress;
    std::size_t const count = arguments->count;
    propagate_pending(outcome);
    if (stats) *stats = {};
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!keys.data || !vectors.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "Keys and vectors must not be null"), void();
    if (keys.stride && keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (rejects_columns(*arguments->index, count, vectors.columns, outcome)) return;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            nk_dtype_t const source_dtype = nk_dtype_of(vectors.scalar);
            if (source_dtype == nk_dtype_unknown_k)
                return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            using key_t = typename variant_type_::key_t;
            typename variant_type_::insert_config_t insert_config {};
            insert_config.expansion_factor = static_cast<u32_t>(arguments->expansion_factor);

            // An insertion takes one packed run of keys, so any other stride gathers into one first.
            std::span<key_t const> uuids {reinterpret_cast<key_t const*>(keys.data), count};
            growing_buffer<key_t> gathered_keys {global_allocator};
            if (keys.stride != sizeof(usearch_key_t)) {
                if (failed(gathered_keys.resize(count)))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to gather the keys"),
                           void();
                for (std::size_t row = 0; row != count; ++row)
                    std::memcpy(&gathered_keys[row], reinterpret_cast<char const*>(keys.data) + row * keys.stride,
                                sizeof(key_t));
                uuids = {gathered_keys.data(), count};
            }

            std::size_t const dimensions = specialized_index {variant}.dimensions();
            if (vectors.stride && vectors.stride < packed_row_bytes(source_dtype, dimensions))
                return propagate_error(usearch_status_unsupported_layout_k, outcome,
                                       "The stride is shorter than one row"),
                       void();
            insertion_rows_t const rows {static_cast<char const*>(vectors.data), vectors.stride, source_dtype, progress,
                                         arguments->progress_state};
            expected<typename variant_type_::insert_result_t> const inserted = specialized_index {variant}.add(
                uuids, rows, *executor, insert_config);
            if (crossed(inserted, stats, outcome)) propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_refine(usearch_dense_refine_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    usearch_dense_traversal_stats_t* const stats = arguments->stats;
    propagate_pending(outcome);
    if (stats) *stats = {};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            typename variant_type_::refine_config_t config {};
            config.expansion_factor = static_cast<u32_t>(arguments->expansion_factor);
            shard_ticks_t ticks {arguments->progress, arguments->progress_state, specialized_index {variant}.size()};
            auto const [refined, refined_status] = specialized_index {variant}.refine(*executor, config, ticks);
            accumulate_stats(stats, refined);
            // A stopped pass is not a failure: it still healed, the graph stands, and the next call
            // sweeps again toward the same fixed point.
            if (failed(refined_status) && refined_status != status_t::cancelled_k)
                return propagate_error(refined_status, outcome), void();
            if (arguments->changed) *arguments->changed = refined.changed_nodes;
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_health(usearch_dense_health_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->reachability = {};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            auto const [audit, status] = specialized_index {variant}.health(*executor);
            if (failed(status)) return propagate_error(status, outcome), void();
            *arguments->reachability = {audit.audited_nodes, audit.reachable_nodes, audit.sweeps};
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_search(usearch_dense_search_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    usearch_matches_span_t const& matches = arguments->matches;
    propagate_pending(outcome);
    if (rejects_columns(*arguments->index, arguments->count, arguments->queries.columns, outcome)) return;
    key_predicate_t keyed {arguments->predicate, arguments->predicate_state};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            auto search_with = [&](auto&& predicate) noexcept {
                match_dense_rows<typename variant_type_::distance_t>(
                    variant, arguments->executor, arguments->count, arguments->queries, 1, matches, nk_f32_k,
                    arguments->expansion_factor, arguments->progress, arguments->progress_state, arguments->stats,
                    outcome,
                    [&](auto view, auto rows, auto& executor, auto config, auto keys, auto distances, auto counts) {
                        return view.search(rows, executor, config, keys, distances, counts, predicate);
                    });
            };
            if (!keyed.callback) search_with(dummy_predicate_t {});
            else if constexpr (!is_ram_index<variant_type_>)
                propagate_error(usearch_status_device_k, outcome, "A host predicate cannot narrow a device walk");
            else search_with(keyed);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_get(usearch_dense_get_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    usearch_key_t const key = arguments->key;
    propagate_pending(outcome);
    *arguments->vectors = nullptr;
    if (arguments->written) *arguments->written = 0;
    if (arguments->stride) *arguments->stride = 0;
    if (arguments->columns) *arguments->columns = 0;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            if constexpr (!is_ram_index<variant_type_>)
                propagate_error(
                    arguments->arena && *arguments->arena ? usearch_status_device_k : usearch_status_unsupported_k,
                    outcome, "Arenas live in host memory until device arenas land");
            else {
                using scalar_t = typename variant_type_::scalar_t;
                // The rows are copied on the caller's thread, so the executor is only checked, never built.
                if (arguments->executor && !std::holds_alternative<forkunion_executor_t>(arguments->executor->executor))
                    return propagate_error(usearch_status_device_k, outcome,
                                           "The executor's kind must match the handle's residency"),
                           void();
                nk_dtype_t const target_dtype = nk_dtype_of(arguments->scalar);
                if (target_dtype == nk_dtype_unknown_k)
                    return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
                std::size_t const entries = specialized_index {variant}.count(uuid_of(key));
                if (!entries) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, nullptr, outcome);
                if (!arena) return;
                std::size_t const dimensions = specialized_index {variant}.dimensions();
                std::size_t const row_bytes = packed_row_bytes(target_dtype, dimensions);
                std::byte* const rows = arena->memory.allocate(entries * row_bytes, default_alignment_k);
                expected<std::size_t> const exported =
                    rows ? export_rows<scalar_t>(
                               usearch_rows_span_t {rows, row_bytes, dimensions, entries, arguments->scalar},
                               dimensions, variant_type_::vector_dtype_k,
                               [&](nk::matrix_span<scalar_t> span) {
                                   return specialized_index {variant}.get(uuid_of(key), span, entries);
                               })
                         : expected<std::size_t> {0, status_t::bad_alloc_k};
                arena->busy.unlock();
                if (!exported) return propagate_error(exported.status, outcome, "Failed to export the rows"), void();
                *arguments->vectors = rows;
                if (arguments->written) *arguments->written = min_of(exported.value, entries);
                if (arguments->stride) *arguments->stride = row_bytes;
                if (arguments->columns) *arguments->columns = dimensions;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_search_stream_init(usearch_dense_search_stream_init_t* arguments) {
    propagate_pending(arguments->outcome);
    *arguments->stream = nullptr;
    propagate_error(usearch_status_unsupported_k, arguments->outcome, "No residency streams matches yet");
}

USEARCH_EXPORT void usearch_dense_search_stream_next(usearch_dense_search_stream_next_t* arguments) {
    propagate_pending(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, arguments->outcome, "No residency streams matches yet");
}

USEARCH_EXPORT void usearch_dense_search_stream_free(usearch_dense_search_stream_t) {}

#pragma endregion Insertion and Search

#pragma region Cursors and Merging

USEARCH_EXPORT void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->cursor = nullptr;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            if constexpr (!is_ram_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome,
                                "A cursor walks host rows; device rows are read through a search");
            else {
                usearch_dense_cursor_t const opened = new (std::nothrow)
                    usearch_dense_cursor_handle_t {decltype(usearch_dense_cursor_handle_t::walk) {
                                                       std::in_place_type<typename variant_type_::cursor_t>, variant},
                                                   arguments->index,
                                                   {}};
                if (!opened)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"),
                           void();
                *arguments->cursor = opened;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_cursor_next(usearch_dense_cursor_next_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    usearch_keys_span_t const keys = arguments->keys;
    usearch_rows_span_t const& vectors = arguments->vectors;
    propagate_pending(outcome);
    *arguments->written = 0;
    if (!arguments->cursor)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor must not be null"), void();
    if (arguments->index != arguments->cursor->opened_on)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (!keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must name storage"), void();
    if (keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (vectors.data && vectors.capacity < keys.capacity)
        return propagate_error(usearch_status_shape_mismatch_k, outcome,
                               "The vector output holds fewer rows than the key run"),
               void();
    std::atomic_flag& busy = arguments->cursor->busy;
    if (busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull is still running on the cursor"),
               void();
    std::visit(
        [&]<typename cursor_type_>(cursor_type_& walk) {
            if constexpr (std::is_same_v<cursor_type_, std::monostate>)
                propagate_error(usearch_status_unsupported_k, outcome,
                                "A cursor walks host rows; device rows are read through a search");
            else {
                using scalar_t = typename cursor_type_::scalar_t;
                using key_t = typename cursor_type_::key_t;
                // A cursor reads into one packed run of keys, so any other stride scatters from one.
                bool const packed = keys.stride == sizeof(usearch_key_t);
                growing_buffer<key_t> staged_keys {global_allocator};
                if (!packed && failed(staged_keys.resize(keys.capacity)))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to stage the keys"), void();
                std::span<key_t> const identities {packed ? reinterpret_cast<key_t*>(keys.data) : staged_keys.data(),
                                                   keys.capacity};
                // A handle never changes its arm, so it is still the one the cursor was opened on.
                auto const& indexes = std::get<typename cursor_type_::indexes_t>(arguments->index->variants);
                expected<std::size_t> const read =
                    vectors.data
                        ? export_rows<scalar_t>(
                              vectors, indexes.dimensions(), cursor_type_::indexes_t::vector_dtype_k,
                              [&](nk::matrix_span<scalar_t> rows) { return walk.read(indexes, identities, rows); })
                        : expected<std::size_t> {walk.read(indexes, identities, {}), status_t::success_k};
                if (!read) return propagate_error(read.status, outcome, "Failed to export the rows"), void();
                for (std::size_t row = 0; !packed && row != read.value; ++row)
                    std::memcpy(reinterpret_cast<char*>(keys.data) + row * keys.stride, &identities[row],
                                sizeof(key_t));
                *arguments->written = read.value;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->cursor->walk);
    busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_dense_cursor_free(usearch_dense_cursor_t cursor) { delete cursor; }

USEARCH_EXPORT void usearch_dense_merge(usearch_dense_merge_t* arguments) {
    propagate_pending(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, arguments->outcome, "No residency absorbs donor handles yet");
}

#pragma endregion Cursors and Merging

#pragma region Linear Algebra

USEARCH_EXPORT void usearch_dense_multiply(usearch_dense_multiply_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    std::size_t const count = arguments->count;
    propagate_pending(outcome);
    *arguments->keys = nullptr, *arguments->scores = nullptr;
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    if (!arguments->offsets)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The offsets must name storage"), void();
    if (rejects_columns(*arguments->index, count, arguments->left.columns, outcome)) return;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (variant.kernel() != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A product is a dot, so the kind must be inner product");
            else if constexpr (!is_ram_index<variant_type_>)
                propagate_error(
                    arguments->arena && *arguments->arena ? usearch_status_device_k : usearch_status_unsupported_k,
                    outcome, "Arenas live in host memory until device arenas land");
            else {
                using key_t = typename variant_type_::key_t;
                using dot_result_t = typename variant_type_::dot_result_t;
                growing_buffer<std::size_t> scored {global_allocator};
                growing_buffer<key_t> staged_keys {global_allocator};
                growing_buffer<dot_result_t> staged_products {global_allocator};
                if (failed(scored.resize(count)))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to stage the counts"),
                           void();
                // Every row keeps the pairs its walks scored until `width` of them fill its staged row.
                auto walk = [&](std::size_t width, usearch_progress_callback_t progress) noexcept {
                    if (failed(staged_keys.resize(count * width)) || failed(staged_products.resize(count * width)))
                        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to stage the products");
                    nk::matrix_span<key_t> const kept_keys {staged_keys.data(), count, width};
                    nk::matrix_span<dot_result_t> const kept_products {staged_products.data(), count, width};
                    auto make_sink = [&](std::size_t, status_t&) noexcept {
                        return [&](specialized_index<variant_type_> view,
                                   typename variant_type_::vectors_view_t chunk_rows,
                                   typename variant_type_::executor_t& executor,
                                   typename variant_type_::search_config_t config, std::size_t resolved,
                                   std::size_t taking) noexcept {
                            nk::range_t const chunk = nk::range(resolved, resolved + taking);
                            return view.multiply(chunk_rows, executor, config, kept_keys(chunk, nk::slice),
                                                 kept_products(chunk, nk::slice),
                                                 to_writable_span(scored).subspan(resolved, taking));
                        };
                    };
                    return for_each_dense_chunk(variant, arguments->executor, count, arguments->left, 1,
                                                arguments->expansion_factor, progress, arguments->progress_state,
                                                arguments->stats, outcome, make_sink);
                };
                // Keeping every pair needs the widest row first, which only a counting walk can tell.
                std::size_t width = arguments->wanted;
                if (!width) {
                    if (!walk(0, nullptr)) return;
                    for (std::size_t row = 0; row != count; ++row) width = max_of(width, scored[row]);
                }
                if (!walk(width, arguments->progress)) return;

                std::size_t* const offsets = arguments->offsets;
                std::size_t required = 0;
                offsets[0] = 0;
                for (std::size_t row = 0; row != count; ++row)
                    offsets[row + 1] = offsets[row] + min_of(scored[row], width), required += scored[row];
                std::size_t const written = offsets[count];
                if (arguments->written) *arguments->written = written;
                if (arguments->required) *arguments->required = required;
                if (!written) return propagate_error(usearch_status_success_k, outcome, nullptr), void();

                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->left.data, outcome);
                if (!arena) return;
                auto* const keys = reinterpret_cast<usearch_key_t*>(
                    arena->memory.allocate(written * sizeof(usearch_key_t), alignof(usearch_key_t)));
                auto* const scores = reinterpret_cast<usearch_distance_t*>(
                    arena->memory.allocate(written * sizeof(usearch_distance_t), alignof(usearch_distance_t)));
                nk::matrix_span<key_t> const kept_keys {staged_keys.data(), count, width};
                nk::matrix_span<dot_result_t> const kept_products {staged_products.data(), count, width};
                for (std::size_t row = 0; keys && scores && row != count; ++row) {
                    std::size_t const kept = offsets[row + 1] - offsets[row];
                    if (!kept) continue;
                    std::memcpy(keys + offsets[row], kept_keys.row(row).data(), kept * sizeof(key_t));
                    [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
                        kept_products.row(row).data(), dot_result_t::dtype(), scores + offsets[row], nk_f64_k, kept,
                        nullptr);
                    assert(cast == nk_success_k && "The serial tier casts every dtype pair");
                }
                arena->busy.unlock();
                if (!keys || !scores)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                *arguments->keys = keys, *arguments->scores = scores;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

static_assert(sizeof(us::dense_softmax_state_t) == sizeof(usearch_dense_softmax_state_t));

USEARCH_EXPORT void usearch_dense_attend(usearch_dense_attend_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    usearch_rows_span_t const& outputs = arguments->outputs;
    usearch_dense_softmax_state_t* const carry = arguments->carry;
    std::size_t const count = arguments->count, head_count = max_of<std::size_t>(arguments->head_count, 1);
    propagate_pending(outcome);
    if (!arguments->queries.data || !outputs.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The queries and outputs must name storage"),
               void();
    nk_dtype_t const outputs_dtype = nk_dtype_of(outputs.scalar);
    if (outputs_dtype == nk_dtype_unknown_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
    std::size_t const columns = arguments->queries.columns;
    if (count % head_count || outputs.capacity < count || outputs.columns != columns)
        return propagate_error(usearch_status_shape_mismatch_k, outcome,
                               "The heads must divide the rows, and the outputs hold every row as wide as the keys"),
               void();
    if (rejects_columns(*arguments->index, count, columns, outcome)) return;
    if (outputs.stride < packed_row_bytes(outputs_dtype, columns))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "A stride is shorter than one row"),
               void();
    char* const outputs_base = static_cast<char*>(outputs.data);
    f32_t const requested_scale = arguments->score_scale;

    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (variant.kernel() != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A softmax weighs logits, so the kind must be inner product");
            else if (!arguments->value_lookup)
                propagate_error(usearch_status_invalid_argument_k, outcome, "The value lookup must be set");
            else {
                typename specialized_index<variant_type_>::attend_lookup_t const value_lookup {
                    arguments->value_lookup, arguments->value_lookup_state,
                    usearch_scalar_of(variant_type_::vector_dtype_k), columns};
                // Outputs stage as `f32` and narrow one row at a time; the carry stages beside them.
                auto make_sink = [&](std::size_t chunk, status_t& staged) noexcept {
                    growing_buffer<f32_t> staged_outputs {global_allocator};
                    growing_buffer<us::dense_softmax_state_t> staged_carry {global_allocator};
                    if (failed(staged_outputs.resize(chunk * columns)) || failed(staged_carry.resize(chunk)))
                        staged = status_t::bad_alloc_k;
                    return [&, staged_outputs = std::move(staged_outputs), staged_carry = std::move(staged_carry)](
                               specialized_index<variant_type_> view, typename variant_type_::vectors_view_t chunk_rows,
                               typename variant_type_::executor_t& executor,
                               typename variant_type_::search_config_t config, std::size_t resolved,
                               std::size_t taking) mutable noexcept {
                        f32_t const score_scale = requested_scale > 0
                                                      ? requested_scale
                                                      : 1 / std::sqrt(static_cast<f32_t>(view.dimensions()));
                        nk::matrix_span<f32_t> const attended {staged_outputs.data(), taking, columns};
                        expected<typename variant_type_::search_result_t> const found = view.attend(
                            chunk_rows, head_count, value_lookup, score_scale, attended, {staged_carry.data(), taking},
                            executor, config);
                        if (!found) return found;
                        for (std::size_t row = 0; row != taking; ++row) {
                            [[maybe_unused]] nk_status_t const cast =
                                dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
                                    attended.row(row).data(), nk_f32_k,
                                    outputs_base + (resolved + row) * outputs.stride, outputs_dtype, columns, nullptr);
                            assert(cast == nk_success_k && "The serial tier casts every dtype pair");
                            if (carry)
                                carry[resolved + row] = std::bit_cast<usearch_dense_softmax_state_t>(staged_carry[row]);
                        }
                        return found;
                    };
                };
                if (for_each_dense_chunk(variant, arguments->executor, count, arguments->queries, head_count,
                                         arguments->expansion_factor, arguments->progress, arguments->progress_state,
                                         arguments->stats, outcome, make_sink))
                    propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_maxsim(usearch_dense_maxsim_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    std::size_t const rows_per_query = arguments->rows_per_query;
    usearch_matches_span_t const& matches = arguments->matches;
    propagate_pending(outcome);
    if (!rows_per_query)
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "Every query needs at least one row"), void();
    if (rejects_columns(*arguments->index, arguments->count, arguments->queries.columns, outcome)) return;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (variant.kernel() != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A maximum over logits needs the inner-product kind");
            else
                match_dense_rows<f64_t>(
                    variant, arguments->executor, arguments->count, arguments->queries, rows_per_query, matches,
                    nk_f64_k, arguments->expansion_factor, arguments->progress, arguments->progress_state,
                    arguments->stats, outcome,
                    [&](auto view, auto rows, auto& executor, auto config, auto keys, auto scores, auto counts) {
                        return view.maxsim(rows, rows_per_query, executor, config, keys, scores, counts);
                    });
        },
        arguments->index->variants);
}

#pragma endregion Linear Algebra

#pragma region Serialization

USEARCH_EXPORT void usearch_dense_load_ram(usearch_dense_load_ram_t* arguments) {
    load_dense_index(as_text(arguments->path), arguments->ram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, arguments->outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments) {
    load_dense_index(as_text(arguments->path), arguments->vram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, arguments->outcome);
}

USEARCH_EXPORT void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments) {
    load_dense_index(as_bytes(arguments->buffer), arguments->ram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, arguments->outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments) {
    load_dense_index(as_bytes(arguments->buffer), arguments->vram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, arguments->outcome);
}

USEARCH_EXPORT void usearch_dense_save(usearch_dense_save_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    if (!arguments->path.data || !arguments->path.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The path must not be empty"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            shard_ticks_t ticks {arguments->progress, arguments->progress_state,
                                 specialized_index {variant}.shards_count()};
            propagate_error(specialized_index {variant}.save(as_text(arguments->path), *executor, ticks), outcome);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_save_buffer(usearch_dense_save_buffer_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->bytes = nullptr;
    *arguments->length = 0;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            if constexpr (!is_ram_index<variant_type_>)
                propagate_error(
                    arguments->arena && *arguments->arena ? usearch_status_device_k : usearch_status_unsupported_k,
                    outcome, "Arenas live in host memory until device arenas land");
            else {
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                                 call_executor, outcome);
                if (!executor) return;
                auto const [length, status] = specialized_index {variant}.storage_requirements();
                if (failed(status)) return propagate_error(status, outcome), void();
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, nullptr, outcome);
                if (!arena) return;
                std::byte* const monolith = arena->memory.allocate(length, default_alignment_k);
                shard_ticks_t ticks {arguments->progress, arguments->progress_state,
                                     specialized_index {variant}.shards_count()};
                status_t const saved = monolith ? specialized_index {variant}.save(
                                                      bytes_span_t {reinterpret_cast<byte_t*>(monolith), length},
                                                      *executor, ticks)
                                                : status_t::bad_alloc_k;
                arena->busy.unlock();
                if (failed(saved)) return propagate_error(saved, outcome), void();
                *arguments->bytes = reinterpret_cast<uint8_t*>(monolith);
                *arguments->length = length;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_metadata_read(usearch_dense_metadata_read_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    if (!arguments->path.data || !arguments->path.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The path must not be empty"), void();
    auto const [source, status] = safegraphs_read_metadata(as_text(arguments->path));
    if (failed(status)) return propagate_error(status, outcome), void();
    adopt_metadata(source, arguments->metadata);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_metadata_read_buffer(usearch_dense_metadata_read_buffer_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    if (!arguments->buffer.data || !arguments->buffer.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The buffer must not be empty"), void();
    auto const [source, status] = safegraphs_read_metadata(as_bytes(arguments->buffer));
    if (failed(status)) return propagate_error(status, outcome), void();
    adopt_metadata(source, arguments->metadata);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

#pragma region Executors

USEARCH_EXPORT usearch_status_t usearch_outcome_wait(usearch_outcome_t const* outcome) {
    // The status word is written by the library and only read here, so the cast never races a store.
    std::atomic_ref<usearch_status_t> status {const_cast<usearch_status_t&>(outcome->status)};
    usearch_status_t seen = status.load(std::memory_order_acquire);
    for (; seen == usearch_status_deferred_k; seen = status.load(std::memory_order_acquire))
        status.wait(usearch_status_deferred_k, std::memory_order_acquire);
    return seen;
}

USEARCH_EXPORT void usearch_arena_free(usearch_arena_t arena) { delete arena; }

USEARCH_EXPORT void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->executor = nullptr;
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    if (!spawn_ram_executor(*executor, arguments->threads, outcome)) return delete executor, void();
    *arguments->executor = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Borrowed pools are not wired yet");
}

USEARCH_EXPORT void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    *arguments->executor = nullptr;
#if USEARCH_BUILD_CUDA
    static_assert(sizeof(int) == sizeof(int32_t));
    usearch_vram_options_t const& vram = arguments->vram;
    int const fallback_device = 0;
    std::span<int const> const devices = vram.devices && vram.devices_count
                                             ? std::span {reinterpret_cast<int const*>(vram.devices),
                                                          vram.devices_count}
                                             : std::span {&fallback_device, 1};
    if (arguments->streams)
        return propagate_error(usearch_status_unsupported_k, outcome, "Borrowed streams are not wired yet"), void();
    auto [spawned, status] = cuda_executor_t::spawn(devices);
    if (failed(status)) return propagate_error(status, outcome), void();
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    executor->executor = std::move(spawned);
    *arguments->executor = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
#else
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA executor");
#endif
}

USEARCH_EXPORT void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments) {
    propagate_pending(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_unsupported_k, arguments->outcome, "This build carries no NVMe residency");
}

USEARCH_EXPORT void usearch_executor_drain(usearch_executor_drain_t* arguments) {
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    // A null executor runs every call to completion before returning, so nothing waits behind it.
    if (!arguments->executor) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    // A drain from a callback meets the claim of the call running it, and answers contended.
    auto const drain = [](auto& executor) noexcept {
        auto const [claim, status] = executor.claim();
        return failed(status) ? status : executor.drain();
    };
    propagate_error(std::visit(drain, arguments->executor->executor), outcome);
}

USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor) { delete executor; }

#pragma endregion Executors

} // extern "C"
