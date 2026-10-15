/**
 *  @file src/dense_index.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense modality's C surface: the variant grid over every compiled backend, the handle
 *      owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Compiled once, as @c dense_index.cpp by the host compiler, with @c USEARCH_ARCH_CUDA_,
 *  @c USEARCH_ARCH_ROCM_ and @c USEARCH_ARCH_METAL_ deciding which vendors' VRAM arms join the
 *  variant and @c USEARCH_DENSE_DTYPES how many storage dtypes every class contributes to it.
 *  Arms are ordered newest-capability-first inside each dtype cell, since a machine reporting
 *  Ice Lake also reports Haswell and Skylake; the serial floor is last and unconditional. The
 *  arms' bodies compile in the per-class units, never here.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cmath>   // `std::sqrt` - attention's default logit scale
#include <cstdio>  // `std::snprintf` - composing the capability string
#include <cstring> // `std::memcpy`, `std::memmove`, `std::strlen` - landing rows & the capability string

#include <atomic>             // `std::atomic` - serializing the progress ticks
#include <bit>                // `std::bit_cast` - the softmax carry crossing the ABI
#include <condition_variable> // `std::condition_variable` - waiting out deferred calls other threads run
#include <initializer_list>   // `std::initializer_list` - the inputs a call checks against its arena
#include <mutex>              // `std::lock_guard` over the ticks' lock and the deferred tables
#include <memory>             // `std::unique_ptr` - the executor's topology
#include <new>                // `std::nothrow`
#include <span>               // `std::span` - key batches crossing the ABI
#include <string_view>        // `std::string_view` - checkpoint paths
#include <type_traits>        // `std::is_same_v` - the variant-membership audit
#include <utility>            // `std::exchange` - nulling each donor slot an absorb consumes
#include <variant>            // `std::variant`, `std::visit`

#include <forkunion.h>            // `fu_topology_new` - the executor factories' discovery
#include <forkunion/types.hpp>    // `fu::spin_mutex` - the ticks' lock
#include <numkong/capabilities.h> // `nk_cpu_capabilities_enabled`, `nk_cpu_configure_thread`
#include <numkong/cast.h>         // `nk_cast_best` - converting foreign scalar rows at the boundary

#include "dense_specializations.cuh"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator`, `no_memory_domain_k` - the multi-indexes' host tables
#include "usearch/utilities/vram_device.hpp" // `gpu_devices`, `gpu_executor_t`, `gpu_arena_t`

#pragma region Multi Index

/**
 *  @brief The one dynamically-typed dense index every binding consumes: a variant over the
 *      compiled multi-index alternatives, plus the geometry & capability metadata no shard
 *      answers.
 *
 *  Arms exist per wired @c (dtype,capability) cell - those whose NumKong kernel kits are
 *  adapted - never the full cross product: an unwired cell would not compile, and advertising
 *  it would only move the failure from the build to the caller. Each arm keeps its metric as a
 *  runtime field, since only the distance epilogue reads it. The VRAM siblings join per GPU vendor
 *  the build carries, and the read-only NVMe siblings wherever a CPU class picks a kit of its own.
 */
struct usearch_dense_index_handle_t {

    using variants_t = std::variant<
#if NUMKONG_TARGET_HASWELL
        // intel_haswell over serial by haswell: every dtype, 11 arms
        ram_indexes<nk_bf16_k, cpu_class_t::intel_haswell_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::intel_haswell_k>, //
#endif
#endif // NUMKONG_TARGET_HASWELL
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
        // intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd:
        // bf16 u1 e4m3 f16 e5m2 i4 u4 e3m2, 8 arms
        ram_indexes<nk_bf16_k, cpu_class_t::intel_sierra_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::intel_sierra_k>, //
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
    // intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd, alder, sierra:
    // i8 u8 e2m3, 3 arms
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::intel_sierra_k>, //
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
#if NUMKONG_TARGET_SKYLAKE
        // intel_skylake over intel_haswell by skylake: bf16 e4m3 f16 e5m2 e3m2 e2m3, 6 arms
        ram_indexes<nk_bf16_k, cpu_class_t::intel_skylake_k>, //
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::intel_skylake_k>, //
#endif
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_ICELAKE
    // intel_icelake over intel_skylake by icelake: u1 i8 u8 i4 u4, 5 arms
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::intel_icelake_k>, //
#endif
#endif // NUMKONG_TARGET_ICELAKE
#if NUMKONG_TARGET_GENOA
        // amd_genoa over intel_icelake by genoa: bf16 e4m3 e5m2, 3 arms
        ram_indexes<nk_bf16_k, cpu_class_t::amd_genoa_k>, //
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::amd_genoa_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::amd_genoa_k>, //
#endif
#endif // NUMKONG_TARGET_GENOA
#if NUMKONG_TARGET_SAPPHIREAMX
        // intel_sapphire over amd_genoa by sapphireamx: bf16 i8 e4m3 u8 e5m2 e3m2 e2m3, 7 arms
        ram_indexes<nk_bf16_k, cpu_class_t::intel_sapphire_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::intel_sapphire_k>, //
#endif
#endif // NUMKONG_TARGET_SAPPHIREAMX
#if NUMKONG_TARGET_GRANITEAMX
    // intel_granite over intel_sapphire by graniteamx: f16 e5m2, 2 arms
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::intel_granite_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::intel_granite_k>, //
#endif
#endif // NUMKONG_TARGET_GRANITEAMX
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
        // intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd:
        // bf16 u1 f16 i4 u4 e3m2, 6 arms
        ram_indexes<nk_bf16_k, cpu_class_t::intel_diamond_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
    // intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, diamond: e4m3 e5m2, 2 arms
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
    // intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, alder, sierra:
    // i8 u8 e2m3, 3 arms
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
#if NUMKONG_TARGET_NEON
        // aws_g2 over serial by neon: bf16 u1 f16, 3 arms
        ram_indexes<nk_bf16_k, cpu_class_t::aws_g2_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::aws_g2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::aws_g2_k>, //
#endif
#endif // NUMKONG_TARGET_NEON
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONBFDOT
        // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonbfdot: bf16, 1 arm
        ram_indexes<nk_bf16_k, cpu_class_t::apple_m2_k>, //
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
    // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonfhm: e4m3 f16 e5m2, 3 arms
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::apple_m2_k>, //
#endif
#endif // FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
    // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonsdot: i8 u8 i4 u4 e3m2 e2m3, 6 arms
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::apple_m2_k>, //
#endif
#endif // FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
    // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc: u1, 1 arm
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::apple_m2_k>, //
#endif
#endif // FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
#if NUMKONG_TARGET_SME
        // apple_m4 over apple_m2 by sme: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms
        ram_indexes<nk_bf16_k, cpu_class_t::apple_m4_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::apple_m4_k>, //
#endif
#endif // NUMKONG_TARGET_SME
#if NUMKONG_TARGET_SMEBI32
    // apple_m4 over apple_m2 by smebi32: u1, 1 arm
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::apple_m4_k>, //
#endif
#endif // NUMKONG_TARGET_SMEBI32
#if FORKUNION_TARGET_ARM64_WFET
        // aws_g5 over apple_m2 by arm64_wfet: every dtype, 11 arms
        ram_indexes<nk_bf16_k, cpu_class_t::aws_g5_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::aws_g5_k>, //
#endif
#endif // FORKUNION_TARGET_ARM64_WFET
#if NUMKONG_TARGET_NEONFP8
    // nvidia_vera over aws_g5 by neonfp8: e4m3 e5m2 e3m2 e2m3, 4 arms
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::nvidia_vera_k>, //
#endif
#endif // NUMKONG_TARGET_NEONFP8
#if FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
        // alibaba_c930 over serial by risc5_wrs, risc5_zacas: every dtype, 11 arms
        ram_indexes<nk_bf16_k, cpu_class_t::alibaba_c930_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::alibaba_c930_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::alibaba_c930_k>, //
#endif
#endif // FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
#if NUMKONG_TARGET_V128
        // wasm32_v128 over serial by v128: bf16 u1 i8 u8, 4 arms
        ram_indexes<nk_bf16_k, cpu_class_t::wasm32_v128_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::wasm32_v128_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::wasm32_v128_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::wasm32_v128_k>, //
#endif
#endif // NUMKONG_TARGET_V128
#if NUMKONG_TARGET_V128RELAXED
        // wasm64_extended over wasm32_v128 by v128relaxed:
        // bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms
        ram_indexes<nk_bf16_k, cpu_class_t::wasm64_extended_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::wasm64_extended_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::wasm64_extended_k>, //
#endif
#endif // NUMKONG_TARGET_V128RELAXED
#if NUMKONG_TARGET_HASWELL
        // NVMe intel_haswell over serial by haswell: every dtype, 11 arms
        nvme_indexes<nk_bf16_k, cpu_class_t::intel_haswell_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        nvme_indexes<nk_u1_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        nvme_indexes<nk_i4_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        nvme_indexes<nk_u4_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::intel_haswell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::intel_haswell_k>, //
#endif
#endif // NUMKONG_TARGET_HASWELL
#if NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
    // NVMe intel_sierra over intel_haswell by alder, sierra: i8 u8 e2m3, 3 arms
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::intel_sierra_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::intel_sierra_k>, //
#endif
#endif // NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
#if NUMKONG_TARGET_SKYLAKE
        // NVMe intel_skylake over intel_haswell by skylake: bf16 e4m3 f16 e5m2 e3m2 e2m3, 6 arms
        nvme_indexes<nk_bf16_k, cpu_class_t::intel_skylake_k>, //
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::intel_skylake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::intel_skylake_k>, //
#endif
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_ICELAKE
    // NVMe intel_icelake over intel_skylake by icelake: u1 i8 u8 i4 u4, 5 arms
#if USEARCH_DENSE_DTYPES >= 2
        nvme_indexes<nk_u1_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        nvme_indexes<nk_i4_k, cpu_class_t::intel_icelake_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        nvme_indexes<nk_u4_k, cpu_class_t::intel_icelake_k>, //
#endif
#endif // NUMKONG_TARGET_ICELAKE
#if NUMKONG_TARGET_GENOA
        // NVMe amd_genoa over intel_icelake by genoa: bf16 e4m3 e5m2, 3 arms
        nvme_indexes<nk_bf16_k, cpu_class_t::amd_genoa_k>, //
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::amd_genoa_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::amd_genoa_k>, //
#endif
#endif // NUMKONG_TARGET_GENOA
#if NUMKONG_TARGET_SAPPHIREAMX
        // NVMe intel_sapphire over amd_genoa by sapphireamx: bf16 i8 e4m3 u8 e5m2 e3m2 e2m3, 7 arms
        nvme_indexes<nk_bf16_k, cpu_class_t::intel_sapphire_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::intel_sapphire_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::intel_sapphire_k>, //
#endif
#endif // NUMKONG_TARGET_SAPPHIREAMX
#if NUMKONG_TARGET_GRANITEAMX
    // NVMe intel_granite over intel_sapphire by graniteamx: f16 e5m2, 2 arms
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::intel_granite_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::intel_granite_k>, //
#endif
#endif // NUMKONG_TARGET_GRANITEAMX
#if NUMKONG_TARGET_DIAMOND
    // NVMe intel_diamond over intel_granite by diamond: e4m3 e5m2, 2 arms
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::intel_diamond_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::intel_diamond_k>, //
#endif
#endif // NUMKONG_TARGET_DIAMOND
#if NUMKONG_TARGET_NEON
        // NVMe aws_g2 over serial by neon: bf16 u1 f16, 3 arms
        nvme_indexes<nk_bf16_k, cpu_class_t::aws_g2_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        nvme_indexes<nk_u1_k, cpu_class_t::aws_g2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::aws_g2_k>, //
#endif
#endif // NUMKONG_TARGET_NEON
#if NUMKONG_TARGET_NEONBFDOT
        // NVMe apple_m2 over aws_g2 by neonbfdot: bf16, 1 arm
        nvme_indexes<nk_bf16_k, cpu_class_t::apple_m2_k>, //
#endif
#if NUMKONG_TARGET_NEONFHM
    // NVMe apple_m2 over aws_g2 by neonfhm: e4m3 f16 e5m2, 3 arms
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::apple_m2_k>, //
#endif
#endif // NUMKONG_TARGET_NEONFHM
#if NUMKONG_TARGET_NEONSDOT
    // NVMe apple_m2 over aws_g2 by neonsdot: i8 u8 i4 u4 e3m2 e2m3, 6 arms
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        nvme_indexes<nk_i4_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        nvme_indexes<nk_u4_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::apple_m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::apple_m2_k>, //
#endif
#endif // NUMKONG_TARGET_NEONSDOT
#if NUMKONG_TARGET_SME
        // NVMe apple_m4 over apple_m2 by sme: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms
        nvme_indexes<nk_bf16_k, cpu_class_t::apple_m4_k>, //
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        nvme_indexes<nk_i4_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        nvme_indexes<nk_u4_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::apple_m4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::apple_m4_k>, //
#endif
#endif // NUMKONG_TARGET_SME
#if NUMKONG_TARGET_SMEBI32
    // NVMe apple_m4 over apple_m2 by smebi32: u1, 1 arm
#if USEARCH_DENSE_DTYPES >= 2
        nvme_indexes<nk_u1_k, cpu_class_t::apple_m4_k>, //
#endif
#endif // NUMKONG_TARGET_SMEBI32
#if NUMKONG_TARGET_NEONFP8
    // NVMe nvidia_vera over aws_g5 by neonfp8: e4m3 e5m2 e3m2 e2m3, 4 arms
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::nvidia_vera_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::nvidia_vera_k>, //
#endif
#endif // NUMKONG_TARGET_NEONFP8
#if USEARCH_ARCH_CUDA_
        // nvidia_blackwellrtx over nvidia_ampere by "120f": every cell, 11 arms
        vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_blackwellrtx_k>, //
#endif
        // nvidia_blackwell over nvidia_ampere by "100f": every cell, 11 arms
        vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_blackwell_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_blackwell_k>, //
#endif
        // nvidia_hopper over nvidia_ampere by "90a": every cell, 11 arms
        vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_hopper_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_hopper_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_hopper_k>, //
#endif
        // nvidia_ada over nvidia_ampere by "89": every cell, 11 arms
        vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_ada_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_ada_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_ada_k>, //
#endif
        // nvidia_ampere, the floor by "80", its PTX serving every GPU without a class:
        // every cell, 11 arms
        vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_ampere_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_ampere_k>, //
#endif
#endif // USEARCH_ARCH_CUDA_
#if USEARCH_ARCH_ROCM_
        // amd_cdna5 over amd_cdna3 by "gfx1250", 32-wide waves: every cell, 11 arms
        vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna5_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna5_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna5_k>, //
#endif
        // amd_cdna4 over amd_cdna3 by "gfx950": every cell, 11 arms
        vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna4_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna4_k>, //
#endif

        // amd_cdna3, the AMD floor by "gfx942": every cell, 11 arms
        vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna3_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna3_k>, //
#endif
#endif // USEARCH_ARCH_ROCM_
#if USEARCH_ARCH_METAL_
        // apple, serving every Apple class until a family's kernels differ: every cell, 11 arms
        vram_apple_indexes<nk_bf16_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        vram_apple_indexes<nk_u1_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        vram_apple_indexes<nk_i8_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        vram_apple_indexes<nk_e4m3_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        vram_apple_indexes<nk_f16_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        vram_apple_indexes<nk_u8_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        vram_apple_indexes<nk_e5m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        vram_apple_indexes<nk_i4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        vram_apple_indexes<nk_u4_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        vram_apple_indexes<nk_e3m2_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        vram_apple_indexes<nk_e2m3_k>, //
#endif
#endif // USEARCH_ARCH_METAL_

        // The NVMe serial floor: every wired cell, every platform.
        nvme_indexes<nk_bf16_k, cpu_class_t::serial_k>, //
#if USEARCH_DENSE_DTYPES >= 2
        nvme_indexes<nk_u1_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        nvme_indexes<nk_i8_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        nvme_indexes<nk_e4m3_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        nvme_indexes<nk_f16_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        nvme_indexes<nk_u8_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        nvme_indexes<nk_e5m2_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        nvme_indexes<nk_i4_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        nvme_indexes<nk_u4_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        nvme_indexes<nk_e3m2_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        nvme_indexes<nk_e2m3_k, cpu_class_t::serial_k>, //
#endif
    // The serial floor: every wired cell, every platform, the standard yield. The bf16 rung every
    // build keeps closes the list, so no trailing comma ever depends on the ladder.
#if USEARCH_DENSE_DTYPES >= 2
        ram_indexes<nk_u1_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 3
        ram_indexes<nk_i8_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 4
        ram_indexes<nk_e4m3_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 5
        ram_indexes<nk_f16_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 6
        ram_indexes<nk_u8_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 7
        ram_indexes<nk_e5m2_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 8
        ram_indexes<nk_i4_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 9
        ram_indexes<nk_u4_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 10
        ram_indexes<nk_e3m2_k, cpu_class_t::serial_k>, //
#endif
#if USEARCH_DENSE_DTYPES >= 11
        ram_indexes<nk_e2m3_k, cpu_class_t::serial_k>, //
#endif
        ram_indexes<nk_bf16_k, cpu_class_t::serial_k>>; //

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

    template <typename variant_type_, typename... arguments_types_>
    explicit usearch_dense_index_handle_t(std::in_place_type_t<variant_type_> tag,
                                          arguments_types_... arguments) noexcept
        : variants(tag, arguments...) {}
};

/** The arms' explicit instantiation declarations: every body compiles in its class's unit, so this
 *  unit only visits. */
#if NUMKONG_TARGET_HASWELL
/* intel_haswell over serial by haswell: every dtype, 11 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_haswell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#endif // NUMKONG_TARGET_HASWELL
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
/* intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd:
 * bf16 u1 e4m3 f16 e5m2 i4 u4 e3m2, 8 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_sierra_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_sierra_k>>;
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd, alder, sierra:
 * i8 u8 e2m3, 3 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_sierra_k>>;
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
#if NUMKONG_TARGET_SKYLAKE
/* intel_skylake over intel_haswell by skylake: bf16 e4m3 f16 e5m2 e3m2 e2m3, 6 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_skylake_k>>;
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_ICELAKE
/* intel_icelake over intel_skylake by icelake: u1 i8 u8 i4 u4, 5 arms */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::intel_icelake_k>>;
#endif
#endif // NUMKONG_TARGET_ICELAKE
#if NUMKONG_TARGET_GENOA
/* amd_genoa over intel_icelake by genoa: bf16 e4m3 e5m2, 3 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::amd_genoa_k>>;
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::amd_genoa_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::amd_genoa_k>>;
#endif
#endif // NUMKONG_TARGET_GENOA
#if NUMKONG_TARGET_SAPPHIREAMX
/* intel_sapphire over amd_genoa by sapphireamx: bf16 i8 e4m3 u8 e5m2 e3m2 e2m3, 7 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_sapphire_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#endif // NUMKONG_TARGET_SAPPHIREAMX
#if NUMKONG_TARGET_GRANITEAMX
/* intel_granite over intel_sapphire by graniteamx: f16 e5m2, 2 arms */
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_granite_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_granite_k>>;
#endif
#endif // NUMKONG_TARGET_GRANITEAMX
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
/* intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd: bf16 u1 f16 i4 u4 e3m2, 6 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_diamond_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
/* intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, diamond: e4m3 e5m2, 2 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* intel_diamond over intel_granite by x86_tpause, x86_cmpccxadd, alder, sierra:
 * i8 u8 e2m3, 3 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif // FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
#if NUMKONG_TARGET_NEON
/* aws_g2 over serial by neon: bf16 u1 f16, 3 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::aws_g2_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::aws_g2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::aws_g2_k>>;
#endif
#endif // NUMKONG_TARGET_NEON
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONBFDOT
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonbfdot: bf16, 1 arm */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::apple_m2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonfhm: e4m3 f16 e5m2, 3 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::apple_m2_k>>;
#endif
#endif // FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonsdot: i8 u8 i4 u4 e3m2 e2m3, 6 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::apple_m2_k>>;
#endif
#endif // FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc: u1, 1 arm */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::apple_m2_k>>;
#endif
#endif // FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
#if NUMKONG_TARGET_SME
/* apple_m4 over apple_m2 by sme: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::apple_m4_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::apple_m4_k>>;
#endif
#endif // NUMKONG_TARGET_SME
#if NUMKONG_TARGET_SMEBI32
/* apple_m4 over apple_m2 by smebi32: u1, 1 arm */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::apple_m4_k>>;
#endif
#endif // NUMKONG_TARGET_SMEBI32
#if FORKUNION_TARGET_ARM64_WFET
/* aws_g5 over apple_m2 by arm64_wfet: every dtype, 11 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::aws_g5_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::aws_g5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::aws_g5_k>>;
#endif
#endif // FORKUNION_TARGET_ARM64_WFET
#if NUMKONG_TARGET_NEONFP8
/* nvidia_vera over aws_g5 by neonfp8: e4m3 e5m2 e3m2 e2m3, 4 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#endif // NUMKONG_TARGET_NEONFP8
#if FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
/* alibaba_c930 over serial by risc5_wrs, risc5_zacas: every dtype, 11 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::alibaba_c930_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::alibaba_c930_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::alibaba_c930_k>>;
#endif
#endif // FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
#if NUMKONG_TARGET_V128
/* wasm32_v128 over serial by v128: bf16 u1 i8 u8, 4 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::wasm32_v128_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::wasm32_v128_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::wasm32_v128_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::wasm32_v128_k>>;
#endif
#endif // NUMKONG_TARGET_V128
#if NUMKONG_TARGET_V128RELAXED
/* wasm64_extended over wasm32_v128 by v128relaxed:
 * bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::wasm64_extended_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::wasm64_extended_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::wasm64_extended_k>>;
#endif
#endif // NUMKONG_TARGET_V128RELAXED
#if USEARCH_ARCH_CUDA_
/* nvidia_blackwellrtx over nvidia_ampere by "120f": every cell, 11 arms */
extern template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_blackwellrtx_k>>;
#endif
/* nvidia_blackwell over nvidia_ampere by "100f": every cell, 11 arms */
extern template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_blackwell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
/* nvidia_hopper over nvidia_ampere by "90a": every cell, 11 arms */
extern template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_hopper_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_hopper_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_hopper_k>>;
#endif
/* nvidia_ada over nvidia_ampere by "89": every cell, 11 arms */
extern template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_ada_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_ada_k>>;
#endif
/* nvidia_ampere, the floor by "80", its PTX serving every GPU without a class:
 * every cell, 11 arms */
extern template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_ampere_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#endif // USEARCH_ARCH_CUDA_
#if USEARCH_ARCH_ROCM_
/* amd_cdna5 over amd_cdna3 by "gfx1250", 32-wide waves: every cell, 11 arms */
extern template struct specialized_index<vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna5_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna5_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna5_k>>;
#endif
/* amd_cdna4 over amd_cdna3 by "gfx950": every cell, 11 arms */
extern template struct specialized_index<vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna4_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna4_k>>;
#endif
/* amd_cdna3, the AMD floor by "gfx942": every cell, 11 arms */
extern template struct specialized_index<vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna3_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna3_k>>;
#endif
#endif // USEARCH_ARCH_ROCM_
#if USEARCH_ARCH_METAL_
/* apple, serving every Apple class until a family's kernels differ: every cell, 11 arms */
extern template struct specialized_index<vram_apple_indexes<nk_bf16_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<vram_apple_indexes<nk_u1_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<vram_apple_indexes<nk_i8_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<vram_apple_indexes<nk_e4m3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<vram_apple_indexes<nk_f16_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<vram_apple_indexes<nk_u8_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<vram_apple_indexes<nk_e5m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<vram_apple_indexes<nk_i4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<vram_apple_indexes<nk_u4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<vram_apple_indexes<nk_e3m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<vram_apple_indexes<nk_e2m3_k>>;
#endif
#endif // USEARCH_ARCH_METAL_
#if NUMKONG_TARGET_HASWELL
/* NVMe intel_haswell over serial by haswell: every dtype, 11 arms */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::intel_haswell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#endif // NUMKONG_TARGET_HASWELL
#if NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* NVMe intel_sierra over intel_haswell by alder, sierra: i8 u8 e2m3, 3 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::intel_sierra_k>>;
#endif
#endif // NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
#if NUMKONG_TARGET_SKYLAKE
/* NVMe intel_skylake over intel_haswell by skylake: bf16 e4m3 f16 e5m2 e3m2 e2m3, 6 arms */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::intel_skylake_k>>;
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_ICELAKE
/* NVMe intel_icelake over intel_skylake by icelake: u1 i8 u8 i4 u4, 5 arms */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::intel_icelake_k>>;
#endif
#endif // NUMKONG_TARGET_ICELAKE
#if NUMKONG_TARGET_GENOA
/* NVMe amd_genoa over intel_icelake by genoa: bf16 e4m3 e5m2, 3 arms */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::amd_genoa_k>>;
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::amd_genoa_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::amd_genoa_k>>;
#endif
#endif // NUMKONG_TARGET_GENOA
#if NUMKONG_TARGET_SAPPHIREAMX
/* NVMe intel_sapphire over amd_genoa by sapphireamx: bf16 i8 e4m3 u8 e5m2 e3m2 e2m3, 7 arms */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::intel_sapphire_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#endif // NUMKONG_TARGET_SAPPHIREAMX
#if NUMKONG_TARGET_GRANITEAMX
/* NVMe intel_granite over intel_sapphire by graniteamx: f16 e5m2, 2 arms */
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::intel_granite_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::intel_granite_k>>;
#endif
#endif // NUMKONG_TARGET_GRANITEAMX
#if NUMKONG_TARGET_DIAMOND
/* NVMe intel_diamond over intel_granite by diamond: e4m3 e5m2, 2 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::intel_diamond_k>>;
#endif
#endif // NUMKONG_TARGET_DIAMOND
#if NUMKONG_TARGET_NEON
/* NVMe aws_g2 over serial by neon: bf16 u1 f16, 3 arms */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::aws_g2_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::aws_g2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::aws_g2_k>>;
#endif
#endif // NUMKONG_TARGET_NEON
#if NUMKONG_TARGET_NEONBFDOT
/* NVMe apple_m2 over aws_g2 by neonbfdot: bf16, 1 arm */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::apple_m2_k>>;
#endif
#if NUMKONG_TARGET_NEONFHM
/* NVMe apple_m2 over aws_g2 by neonfhm: e4m3 f16 e5m2, 3 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::apple_m2_k>>;
#endif
#endif // NUMKONG_TARGET_NEONFHM
#if NUMKONG_TARGET_NEONSDOT
/* NVMe apple_m2 over aws_g2 by neonsdot: i8 u8 i4 u4 e3m2 e2m3, 6 arms */
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::apple_m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::apple_m2_k>>;
#endif
#endif // NUMKONG_TARGET_NEONSDOT
#if NUMKONG_TARGET_SME
/* NVMe apple_m4 over apple_m2 by sme: bf16 i8 e4m3 f16 u8 e5m2 i4 u4 e3m2 e2m3, 10 arms */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::apple_m4_k>>;
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::apple_m4_k>>;
#endif
#endif // NUMKONG_TARGET_SME
#if NUMKONG_TARGET_SMEBI32
/* NVMe apple_m4 over apple_m2 by smebi32: u1, 1 arm */
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::apple_m4_k>>;
#endif
#endif // NUMKONG_TARGET_SMEBI32
#if NUMKONG_TARGET_NEONFP8
/* NVMe nvidia_vera over aws_g5 by neonfp8: e4m3 e5m2 e3m2 e2m3, 4 arms */
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#endif // NUMKONG_TARGET_NEONFP8
/* The NVMe serial floor: every wired cell, every platform. */
extern template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::serial_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::serial_k>>;
#endif
/* The serial floor: every wired cell, every platform, the standard yield. */
extern template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::serial_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::serial_k>>;
#endif

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

/** Where an NVMe view's parts come from: the checkpoint, and one device or directory per part when
 *  any is named. */
struct nvme_placement_t {
    std::string_view path;
    std::span<std::string_view const> paths;
};

/** Opens every part of the checkpoint @p nvme names, served in place. */
template <typename multi_type_>
    requires is_nvme_index<multi_type_>
static status_t place_shards(multi_type_& variant, nvme_placement_t const& nvme) noexcept {
    return specialized_index {variant}.view(nvme.path, nvme.paths);
}

/** Appends one comma-separated codename to the handle's capability string. */
static void append_capability(usearch_dense_index_handle_t& handle, char const* name) noexcept {
    std::size_t const used = std::strlen(handle.capabilities_string);
    std::snprintf(handle.capabilities_string + used, sizeof(handle.capabilities_string) - used, "%s%s", used ? "," : "",
                  name);
}

/** Fills the handle's capability string once, from the picked NumKong kit. */
template <typename multi_type_>
    requires is_ram_index<multi_type_> || is_nvme_index<multi_type_>
static void compose_capabilities(usearch_dense_index_handle_t& handle, multi_type_ const&) noexcept {
    constexpr nk_capability_t picked =
        us::dense::quads_chain<typename multi_type_::scalar_t, multi_type_::capability_k>::picked_k;
    for (numkong_capability_name_t const& named : numkong_capability_names_k)
        if (picked & named.bit) append_capability(handle, named.name);
}

/** Devices @p vram names, one when it names none. */
static std::size_t named_count(usearch_vram_options_t const& vram) noexcept {
    return vram.devices && vram.devices_count ? vram.devices_count : 1;
}

/** The USearch device ordinal shard @p shard of @p vram sits on - the named devices dealt
 *  round-robin, device zero when it names none. */
static std::size_t named_ordinal(usearch_vram_options_t const& vram, std::size_t shard) noexcept {
    return vram.devices && vram.devices_count ? static_cast<std::size_t>(vram.devices[shard % vram.devices_count]) : 0;
}

/** Appends one VRAM shard per named USearch device ordinal, on that device's vendor ordinal. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static status_t place_shards(multi_type_& variant, usearch_vram_options_t const& vram) noexcept {
    std::span<gpu_device_t const> const devices = gpu_devices();
    for (std::size_t shard = 0; shard != named_count(vram); ++shard) {
        std::size_t const ordinal = named_ordinal(vram, shard);
        if (ordinal >= devices.size()) return status_t::device_error_k;
        if (status_t const placed = specialized_index {variant}.emplace(
                static_cast<int>(devices[ordinal].vendor_ordinal));
            failed(placed))
            return placed;
    }
    return status_t::success_k;
}

/** Fills the handle's capability string once, from the listed class of the first shard's device. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static void compose_capabilities(usearch_dense_index_handle_t& handle, multi_type_& variant) noexcept {
    if (!specialized_index {variant}.shards_count()) return;
    u32_t const resident = static_cast<u32_t>(specialized_index {variant}.device(0));
    for (gpu_device_t const& device : gpu_devices()) {
        if (device.vendor != multi_type_::vendor_k || device.vendor_ordinal != resident) continue;
        for (gpu_class_name_t const& named : gpu_class_names_k)
            if (named.gpu_class == device.gpu_class) return append_capability(handle, named.name);
    }
}

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
    std::in_place_type_t<variant_type_> const tag;
    usearch_dense_index_t handle = nullptr;
    // An NVMe view takes its metric from the parts' headers.
    if constexpr (is_nvme_index<variant_type_>)
        handle = new (std::nothrow) usearch_dense_index_handle_t(tag, host_allocator_of(placement));
    else handle = new (std::nothrow) usearch_dense_index_handle_t(tag, kernel, host_allocator_of(placement));
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
    using upper_chain_t = us::dense::quads_chain<scalar_t, numkong_capabilities_of(cpu_class_)>;
    using lower_chain_t = us::dense::quads_chain<scalar_t, numkong_capabilities_of(parent_k)>;
    using upper_tiles_t = us::dense::tiles_chain<scalar_t, numkong_capabilities_of(cpu_class_)>;
    using lower_tiles_t = us::dense::tiles_chain<scalar_t, numkong_capabilities_of(parent_k)>;
    return upper_chain_t::picked_k != lower_chain_t::picked_k || upper_tiles_t::picked_k != lower_tiles_t::picked_k ||
           forkunion_capabilities_of<cpu_class_>() != forkunion_capabilities_of<parent_k>();
}

/** The unconditional serial floor - every ladder's last rung. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_kernel_kind_t kernel,
                                                nk_capability_t, usearch_outcome_t* outcome) noexcept {
    return make_punned_dense_index<ram_indexes<dtype_, cpu_class_t::serial_k>>(geometry, ram, kernel, outcome);
}

/** One rung: taken when the machine reports the class, else falls through; a class that picks
 *  nothing new is skipped without instantiating its index. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_kernel_kind_t kernel,
                                                nk_capability_t available, usearch_outcome_t* outcome) noexcept {
    if constexpr (rung_advances<dtype_, cpu_class_>()) {
        if (numkong_admits(available, cpu_class_) && forkunion_admits<cpu_class_>())
            return make_punned_dense_index<ram_indexes<dtype_, cpu_class_>>(geometry, ram, kernel, outcome);
    }
    else {
        static_assert(!variant_carries<ram_indexes<dtype_, cpu_class_>, usearch_dense_index_handle_t::variants_t>,
                      "This rung no longer advances past the one below - drop its arm from the variant list");
    }
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

/** Whether one NVMe rung picks a different 4-way or tiles kit than the class it degrades to: a
 *  storage walk carries no waiter or atomic reference, and a scan streams through the tiles kit, so
 *  the kits alone tell two classes apart. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
consteval bool nvme_rung_advances() noexcept {
    using scalar_t = typename nk::type_for<dtype_>::type;
    constexpr nk_capability_t upper = numkong_capabilities_of(cpu_class_);
    constexpr nk_capability_t lower = numkong_capabilities_of(parent_class_of(cpu_class_));
    return us::dense::quads_chain<scalar_t, upper>::picked_k != us::dense::quads_chain<scalar_t, lower>::picked_k ||
           us::dense::tiles_chain<scalar_t, upper>::picked_k != us::dense::tiles_chain<scalar_t, lower>::picked_k;
}

/** The unconditional serial floor of every NVMe ladder. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t descend_nvme_ladder(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                                 nk_kernel_kind_t kernel, nk_capability_t,
                                                 usearch_outcome_t* outcome) noexcept {
    return make_punned_dense_index<nvme_indexes<dtype_, cpu_class_t::serial_k>>(geometry, nvme, kernel, outcome);
}

/** One NVMe rung: taken when the machine reports the class's kits, else falls through; a class that
 *  picks no kit of its own is skipped without instantiating its index. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
static usearch_dense_index_t descend_nvme_ladder(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                                 nk_kernel_kind_t kernel, nk_capability_t available,
                                                 usearch_outcome_t* outcome) noexcept {
    if constexpr (nvme_rung_advances<dtype_, cpu_class_>()) {
        if (numkong_admits(available, cpu_class_))
            return make_punned_dense_index<nvme_indexes<dtype_, cpu_class_>>(geometry, nvme, kernel, outcome);
    }
    else {
        // A class sharing its parent's mask names the parent's very arm, which the variant rightly
        // carries.
        static_assert(numkong_capabilities_of(cpu_class_) == numkong_capabilities_of(parent_class_of(cpu_class_)) ||
                          !variant_carries<nvme_indexes<dtype_, cpu_class_>, usearch_dense_index_handle_t::variants_t>,
                      "This NVMe rung no longer advances past the one below - drop its arm from the variant list");
    }
    return descend_nvme_ladder<dtype_, lower_classes_...>(geometry, nvme, kernel, available, outcome);
}

/** Picks the NVMe variant for one dtype on the RAM ladder's classes, the WebAssembly ones aside,
 *  since a module reads no storage directly. The parts' headers name the metric @p kernel was read
 *  from, so the view checks them against each other rather than against it. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                        nk_kernel_kind_t kernel, usearch_outcome_t* outcome) noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_nvme_ladder<dtype_, cpu_class_t::intel_diamond_k, cpu_class_t::intel_granite_k,
                               cpu_class_t::intel_sapphire_k, cpu_class_t::amd_genoa_k, cpu_class_t::intel_icelake_k,
                               cpu_class_t::intel_skylake_k, cpu_class_t::intel_sierra_k,
                               cpu_class_t::intel_haswell_k>( //
        geometry, nvme, kernel, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_nvme_ladder<dtype_, cpu_class_t::nvidia_vera_k, cpu_class_t::apple_m4_k, cpu_class_t::aws_g5_k,
                               cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>( //
        geometry, nvme, kernel, available, outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_nvme_ladder<dtype_, cpu_class_t::alibaba_c930_k>(geometry, nvme, kernel, available, outcome);
#else
    return descend_nvme_ladder<dtype_>(geometry, nvme, kernel, available, outcome);
#endif
}

/** The class every named device shares, read off the process listing: an index spans one vendor,
 *  since its shards merge through peer copies, and one generation, since its arms run one class's
 *  kernels. Mixed vendors or generations refuse with @c config_mismatch_k, and an unlisted
 *  ordinal with @c device_error_k. */
static expected<gpu_class_t> shared_gpu_class(usearch_vram_options_t const& vram) noexcept {
    std::span<gpu_device_t const> const devices = gpu_devices();
    gpu_device_t shared;
    for (std::size_t position = 0; position != named_count(vram); ++position) {
        std::size_t const ordinal = named_ordinal(vram, position);
        if (ordinal >= devices.size()) return {gpu_class_t::unknown_k, status_t::device_error_k};
        gpu_device_t const device = devices[ordinal];
        if (position && (device.vendor != shared.vendor || device.gpu_class != shared.gpu_class))
            return {gpu_class_t::unknown_k, status_t::config_mismatch_k};
        shared = device;
    }
    return {shared.gpu_class, status_t::success_k};
}

/** The VRAM sibling of the RAM @ref make_index: the devices' class picks the arm, whose unit alone
 *  carries kernels for them, the way the CPU ladder picks the class the machine reports; every
 *  dtype has a kit there, and @ref with_metric admits only the metrics its family scores by. */
template <nk_dtype_t dtype_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_vram_options_t const& vram,
                                        nk_kernel_kind_t kernel, usearch_outcome_t* outcome) noexcept {
    auto const [gpu_class, status] = shared_gpu_class(vram);
    if (status == status_t::config_mismatch_k)
        return propagate_error(status, outcome, "One index spans one GPU vendor and generation"), nullptr;
    switch (gpu_class) {
#if USEARCH_ARCH_CUDA_
    case gpu_class_t::nvidia_blackwellrtx_k:
        return make_punned_dense_index<vram_nvidia_indexes<dtype_, gpu_class_t::nvidia_blackwellrtx_k>>(
            geometry, vram, kernel, outcome);
    case gpu_class_t::nvidia_blackwell_k:
        return make_punned_dense_index<vram_nvidia_indexes<dtype_, gpu_class_t::nvidia_blackwell_k>>(geometry, vram,
                                                                                                     kernel, outcome);
    case gpu_class_t::nvidia_hopper_k:
        return make_punned_dense_index<vram_nvidia_indexes<dtype_, gpu_class_t::nvidia_hopper_k>>(geometry, vram,
                                                                                                  kernel, outcome);
    case gpu_class_t::nvidia_ada_k:
        return make_punned_dense_index<vram_nvidia_indexes<dtype_, gpu_class_t::nvidia_ada_k>>(geometry, vram, kernel,
                                                                                               outcome);
    case gpu_class_t::nvidia_ampere_k:
        return make_punned_dense_index<vram_nvidia_indexes<dtype_, gpu_class_t::nvidia_ampere_k>>(geometry, vram,
                                                                                                  kernel, outcome);
#endif
#if USEARCH_ARCH_ROCM_
    case gpu_class_t::amd_cdna5_k:
        return make_punned_dense_index<vram_amd_indexes<dtype_, gpu_class_t::amd_cdna5_k>>(geometry, vram, kernel,
                                                                                           outcome);
    case gpu_class_t::amd_cdna4_k:
        return make_punned_dense_index<vram_amd_indexes<dtype_, gpu_class_t::amd_cdna4_k>>(geometry, vram, kernel,
                                                                                           outcome);
    case gpu_class_t::amd_cdna3_k:
        return make_punned_dense_index<vram_amd_indexes<dtype_, gpu_class_t::amd_cdna3_k>>(geometry, vram, kernel,
                                                                                           outcome);
#endif // USEARCH_ARCH_ROCM_
#if USEARCH_ARCH_METAL_
    case gpu_class_t::apple_metal10_k:
    case gpu_class_t::apple_metal9_k:
    case gpu_class_t::apple_metal7_k:
        return make_punned_dense_index<vram_apple_indexes<dtype_>>(geometry, vram, kernel, outcome);
#endif
    default:
        return propagate_error(usearch_status_device_k, outcome,
                               "The devices are not listed, or this build carries no kernels for them"),
               nullptr;
    }
}

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
    // Only f32, f64 & the geospatial i32 cross the boundary without ever being stored as rows, so
    // any other scalar the table knows was left out.
    nk_dtype_t const dtype = nk_dtype_of(geometry.quantization);
    if (dtype == nk_dtype_unknown_k || dtype == nk_f32_k || dtype == nk_f64_k || dtype == nk_i32_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unsupported quantization"), nullptr;
    return propagate_error(usearch_status_unsupported_k, outcome,
                           "This build's USEARCH_DENSE_DTYPES leaves the quantization out"),
           nullptr;
}

#pragma endregion Construction

#pragma region Multi Verbs

/** The executor a null handle runs VRAM work on: streams or queues of its own on its devices, never
 *  a legacy default stream, which would wait on every blocking stream of the process. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static usearch_executor_t call_executor_for(multi_type_& variant, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    // One stream or queue per distinct device, which every shard there shares.
    specialized_index const shards {variant};
    growing_buffer<int> devices {global_allocator};
    if (failed(devices.reserve_exactly(shards.shards_count())))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), nullptr;
    for (std::size_t shard = 0; shard != shards.shards_count(); ++shard) {
        bool opened = false;
        for (int const earlier : devices) opened |= earlier == shards.device(shard);
        if (!opened) devices.push_back_reserved(shards.device(shard));
    }
    auto [built, status] = multi_type_::executor_t::spawn(devices);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    call_executor.executor = gpu_executor_t {std::move(built)};
    return &call_executor;
}

/** The GPU ordinal @p variant computes on, or @c host_device_k. */
template <typename variant_type_>
static int device_of(variant_type_& variant) noexcept {
    if constexpr (is_vram_index<variant_type_>) return specialized_index {variant}.device(0);
    else return host_device_k;
}

/** @p count rows of the C view @p rows as the engines take them, their scalar riding along. */
static us::dense::rows_view_t rows_of(usearch_rows_view_t const& rows, std::size_t count) noexcept {
    return {static_cast<byte_t const*>(rows.data), count, rows.columns, rows.stride, nk_dtype_of(rows.scalar)};
}

/** The dtype of rows in @p scalar, @p columns wide, or unknown once @p outcome refuses a
 *  scalar NumKong lacks or sub-byte rows ending inside a byte, which neither its casts nor
 *  indexes hold. */
static nk_dtype_t rows_dtype_of(usearch_scalar_t scalar, std::size_t columns, usearch_outcome_t* outcome) noexcept {
    nk_dtype_t const dtype = nk_dtype_of(scalar);
    if (dtype == nk_dtype_unknown_k) propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind");
    else if (columns % nk_dimensions_per_value(dtype))
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "Sub-byte rows come in whole bytes"),
               nk_dtype_unknown_k;
    return dtype;
}

/**
 *  @brief The one loop every row verb drives on an arm that serves it: hands @p rows to @p sink
 *      chunk by chunk, each arm importing its rows into its own scalar.
 *
 *  Chunks hold whole queries of @p rows_per_query rows, one per step of @p progress, or the whole
 *  call without a handler, as each engine bounds its own scratch; progress counts queries. Finishes
 *  @p outcome only on failure, leaving the success to the verb once its last output is written.
 */
template <typename variant_type_, typename sink_type_>
static bool for_each_dense_chunk(variant_type_& variant, usearch_executor_t executor_handle,         //
                                 std::size_t rows_count, usearch_rows_view_t const& rows,            //
                                 std::size_t rows_per_query, std::size_t expansion_factor,           //
                                 usearch_progress_t progress,                                        //
                                 usearch_dense_traversal_stats_t* stats, usearch_outcome_t* outcome, //
                                 sink_type_&& sink) noexcept {
    if (stats) *stats = {};
    if (!rows_count) return true;
    if (!rows.data) return propagate_error(usearch_status_invalid_argument_k, outcome, "The rows must name storage");
    nk_dtype_t const source_dtype = rows_dtype_of(rows.scalar, rows.columns, outcome);
    if (source_dtype == nk_dtype_unknown_k) return false;
    usearch_executor_handle_t call_executor;
    typename variant_type_::executor_t* const executor = executor_of(variant, executor_handle, call_executor, outcome);
    if (!executor) return false;
    typename variant_type_::search_config_t search_config {};
    search_config.expansion_factor = static_cast<u32_t>(expansion_factor);
    if (rows.stride && rows.stride < us::dense::packed_bytes(source_dtype, rows.columns))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row");

    std::size_t const queries = rows_count / rows_per_query;
    progress_ticks_t ticks {progress, queries};
    std::size_t const chunk = rows_per_query * ticks.slice();
    us::dense::rows_view_t const all = rows_of(rows, rows_count);
    for (std::size_t resolved = 0; resolved != rows_count;) {
        std::size_t const taking = min_of(chunk, rows_count - resolved);
        expected<typename variant_type_::search_result_t> const found = sink(
            specialized_index {variant}, all.slice(resolved, taking), *executor, search_config, resolved, taking);
        if (!crossed(found, stats, outcome)) return false;
        resolved += taking;
        if (ticks(resolved / rows_per_query, queries) == continuation_t::cancel_k)
            return propagate_error(usearch_status_cancelled_k, outcome, "The handler stopped the call");
    }
    return true;
}

/** The matching verbs' sink on one serving arm: keys and scores land in @p landing as the walk
 *  writes them, viewed as one row of @c wanted per output row; @p rows_per_query input rows share
 *  one output row, and @p count output rows land. */
template <typename variant_type_, typename verb_type_>
static bool match_dense_rows(variant_type_& variant, usearch_executor_t executor,                //
                             std::size_t count, usearch_rows_view_t const& rows,                 //
                             std::size_t rows_per_query, matches_landing_t const& landing,       //
                             std::size_t expansion_factor, usearch_progress_t progress,          //
                             usearch_dense_traversal_stats_t* stats, usearch_outcome_t* outcome, //
                             verb_type_&& verb) noexcept {
    using key_t = typename variant_type_::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk writes them");
    std::size_t const wanted = landing.wanted;
    nk::matrix_span<key_t> const keys {reinterpret_cast<key_t*>(landing.keys), count, wanted};
    nk::matrix_span<f64_t> const scores {landing.scores, count, wanted};
    auto sink = [&](specialized_index<variant_type_> view, us::dense::rows_view_t chunk_rows,
                    typename variant_type_::executor_t& executor, typename variant_type_::search_config_t config,
                    std::size_t resolved, std::size_t taking) noexcept {
        std::size_t const first = resolved / rows_per_query, matched = taking / rows_per_query;
        nk::range_t const chunk_range = nk::range(first, first + matched);
        return verb(view, chunk_rows, executor, config, keys(chunk_range, nk::slice), scores(chunk_range, nk::slice),
                    std::span<std::size_t> {landing.counts + first, matched});
    };
    return for_each_dense_chunk(variant, executor, count * rows_per_query, rows, rows_per_query, expansion_factor,
                                progress, stats, outcome, sink);
}

#pragma endregion Multi Verbs

/** The cursor of a RAM or NVMe arm; a VRAM arm reads its rows through a search and has none. */
template <typename variant_type_>
struct cursor_of {
    using type = std::monostate;
};
template <typename variant_type_>
    requires is_ram_index<variant_type_> || is_nvme_index<variant_type_>
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

/** Fills every field a checkpoint froze; the type carries nothing a checkpoint cannot say. */
static void adopt_metadata(safegraphs_metadata_t const& source, usearch_dense_metadata_t* metadata) noexcept {
    metadata->dimensions = regular_width(source.value_shape).value_or(0);
    metadata->connectivity = source.connectivity;
    metadata->quantization = usearch_scalar_of(dense::nk_dtype_of(source.value_dtype));
    std::string_view const metric = source.metric.view();
    metadata->metric = usearch_metric_of(nk_kernel_named(metric.data(), metric.size()));
}

/** Builds a handle from the geometry @p source froze, placed where @p placement names, and copies
 *  @p source into it; a failed or cancelled load frees the handle and leaves @c *index null. */
template <typename source_type_, typename placement_type_>
static void load_dense_index(source_type_ source, placement_type_ const& placement, usearch_executor_t executor,
                             usearch_progress_t progress, usearch_dense_index_t* index,
                             usearch_outcome_t* outcome) noexcept {
    *index = nullptr;
    if (source.empty())
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The checkpoint must not be empty"), void();
    auto const [checkpoint, read_status] = safegraphs_read_metadata(source);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_dense_metadata_t geometry {};
    adopt_metadata(checkpoint, &geometry);
    // More parts than named devices keep a shard each, part `p` on device `p` modulo their count.
    placement_type_ placed = placement;
    growing_buffer<int32_t> dealt {global_allocator};
    if constexpr (std::is_same_v<placement_type_, usearch_vram_options_t>)
        if (checkpoint.parts > named_count(placement)) {
            if (failed(dealt.resize(checkpoint.parts)))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to deal the parts"), void();
            for (std::size_t part = 0; part != dealt.size(); ++part)
                dealt[part] = static_cast<int32_t>(named_ordinal(placement, part));
            placed.devices = dealt.data();
            placed.devices_count = checkpoint.parts;
        }
    usearch_dense_index_t const loaded = with_cell(geometry, placed, outcome);
    if (!loaded) return;
    bool const filled = std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            // A load places RAM or VRAM arms alone, so a view's arm never reaches here.
            if constexpr (is_nvme_index<variant_type_>)
                return propagate_error(usearch_status_internal_k, outcome, "A load never builds a view");
            else {
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const typed = executor_of(variant, executor, call_executor,
                                                                              outcome);
                if (!typed) return false;
                progress_ticks_t ticks {progress, specialized_index {variant}.shards_count()};
                status_t const status = specialized_index {variant}.load(source, *typed, ticks);
                return succeeded(status) || propagate_error(status, outcome);
            }
        },
        loaded->variants);
    if (!filled) return delete loaded, void();
    *index = loaded;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** The one registry and resuming slot @c lib.hpp declares for every modality's calls, defined
 *  beside the executor, arena and outcome verbs that drain them. */
deferred_registry_t deferred_registry;
thread_local void const* resuming_call = nullptr;

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_dense_init_ram(usearch_dense_init_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_dense_metadata_t const geometry {arguments->metric, arguments->quantization, arguments->dimensions,
                                             arguments->connectivity};
    usearch_dense_index_t const index = with_cell(geometry, arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_init_vram(usearch_dense_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_dense_metadata_t const geometry {arguments->metric, arguments->quantization, arguments->dimensions,
                                             arguments->connectivity};
    usearch_dense_index_t const index = with_cell(geometry, arguments->vram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_view_ram(usearch_dense_view_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome,
                    "RAM shards copy a checkpoint in and cannot serve a mapping in place yet");
}

USEARCH_EXPORT void usearch_dense_view_nvme(usearch_dense_view_nvme_t* arguments) {
    if (submitted<usearch_dense_view_nvme>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_nvme_options_t const& nvme = arguments->nvme;
    *arguments->index = nullptr;
    if (arguments->executor && !std::holds_alternative<nvme_executor_t>(arguments->executor->executor))
        return propagate_error(usearch_status_device_k, outcome,
                               "The executor's kind must match the handle's residency"),
               void();
    if (!arguments->path.data || !arguments->path.length || (nvme.paths_count && !nvme.paths))
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The paths must name storage"), void();

    std::size_t const paths_count = nvme.paths ? nvme.paths_count : 0;
    growing_buffer<std::string_view> paths {global_allocator};
    if (failed(paths.resize(paths_count)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the paths"), void();
    for (std::size_t path = 0; path != paths_count; ++path) paths[path] = as_text(nvme.paths[path]);

    // The view's own header read, so a transport ID names a checkpoint as a file path does.
    auto const [checkpoint, read_status] = us::dense::nvme_read_metadata(as_text(arguments->path));
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_dense_metadata_t geometry {};
    adopt_metadata(checkpoint, &geometry);
    nvme_placement_t const placement {as_text(arguments->path), to_span(paths)};
    usearch_dense_index_t const index = with_cell(geometry, placement, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_free(usearch_dense_index_t index) {
    finish_calls_using(index);
    delete index;
}

USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_reserve_t* arguments) {
    if (submitted<usearch_dense_reserve>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_dense_index_t const index = arguments->index;
    visit_writable(*index, outcome, [&](auto& variant) noexcept {
        propagate_error(specialized_index {variant}.reserve(
                            {arguments->capacity, index->requested_connectivity, index->requested_dimensions}),
                        outcome);
    });
}

USEARCH_EXPORT void usearch_dense_clear(usearch_dense_clear_t* arguments) {
    if (submitted<usearch_dense_clear>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    visit_writable(*arguments->index, outcome,
                   [&](auto& variant) noexcept { propagate_error(specialized_index {variant}.clear(), outcome); });
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
    if (submitted<usearch_dense_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_dense_index_t const index = arguments->index;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
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
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_count(usearch_dense_count_t* arguments) {
    if (submitted<usearch_dense_count>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    us::utilities::uuid_t const key = uuid_of(arguments->key);
    *arguments->entries = 0;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            // The single key rides the batched lookup, which an NVMe view answers with one scan of
            // its key columns.
            propagate_error(specialized_index {variant}.count({&key, 1}, {arguments->entries, 1}, *executor), outcome);
        },
        arguments->index->variants);
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
    if (count && !frozen)
        return !propagate_error(usearch_status_uninitialized_k, outcome, "The geometry was never frozen");
    return count && columns != frozen &&
           !propagate_error(usearch_status_shape_mismatch_k, outcome,
                            "The rows' width differs from the frozen geometry");
}

USEARCH_EXPORT void usearch_dense_insert(usearch_dense_insert_t* arguments) {
    if (submitted<usearch_dense_insert>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_dense_traversal_stats_t* const stats = arguments->stats;
    usearch_keys_view_t const keys = arguments->keys;
    usearch_rows_view_t const vectors = arguments->vectors;
    std::size_t const count = arguments->count;
    if (stats) *stats = {};
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!keys.data || !vectors.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "Keys and vectors must not be null"), void();
    if (keys.stride && keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (rejects_columns(*arguments->index, count, vectors.columns, outcome)) return;
    visit_writable(*arguments->index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        nk_dtype_t const source_dtype = rows_dtype_of(vectors.scalar, vectors.columns, outcome);
        if (source_dtype == nk_dtype_unknown_k) return;
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        using key_t = typename variant_type_::key_t;
        static_assert(sizeof(key_t) == sizeof(usearch_key_t), "The engine reads the caller's keys in place");
        typename variant_type_::insert_config_t insert_config {};
        insert_config.expansion_factor = static_cast<u32_t>(arguments->expansion_factor);
        if (vectors.stride && vectors.stride < us::dense::packed_bytes(source_dtype, vectors.columns))
            return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row"),
                   void();
        nk::vector_view<key_t> const uuids {reinterpret_cast<char const*>(keys.data), count, keys.stride};
        progress_ticks_t ticks {arguments->progress, count};
        expected<typename variant_type_::insert_result_t> const inserted = specialized_index {variant}.add(
            uuids, rows_of(vectors, count), ticks, *executor, insert_config);
        if (crossed(inserted, stats, outcome)) propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_refine(usearch_dense_refine_t* arguments) {
    if (submitted<usearch_dense_refine>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_dense_traversal_stats_t* const stats = arguments->stats;
    if (stats) *stats = {};
    visit_writable(*arguments->index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        typename variant_type_::refine_config_t config {};
        config.expansion_factor = static_cast<u32_t>(arguments->expansion_factor);
        progress_ticks_t ticks {arguments->progress, specialized_index {variant}.size()};
        auto const [refined, refined_status] = specialized_index {variant}.refine(*executor, config, ticks);
        accumulate_stats(stats, refined);
        // A stopped pass is not a failure: it still healed, the graph stands, and the next call
        // sweeps again toward the same fixed point.
        if (failed(refined_status) && refined_status != status_t::cancelled_k)
            return propagate_error(refined_status, outcome), void();
        if (arguments->changed) *arguments->changed = refined.changed_nodes;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_health(usearch_dense_health_t* arguments) {
    if (submitted<usearch_dense_health>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->reachability = {};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
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
    if (submitted<usearch_dense_search>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (rejects_columns(*arguments->index, count, arguments->queries.columns, outcome)) return;
    if (arguments->traversal != usearch_traversal_walk_k && arguments->traversal != usearch_traversal_scan_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown traversal"), void();
    key_predicate_t keyed {arguments->predicate};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (is_vram_index<variant_type_> && arguments->traversal != usearch_traversal_walk_k)
                propagate_error(usearch_status_unsupported_k, outcome, "Scans run on RAM and NVMe handles");
            else if (is_vram_index<variant_type_> && keyed.predicate.callback)
                propagate_error(usearch_status_device_k, outcome, "A host predicate cannot narrow a device walk");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                                  arguments->executor, device_of(variant),
                                                                  {arguments->queries.data}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                if (!wanted) {
                    // No walk serves zero matches; the counts are zeros staged on the host.
                    std::size_t* counts = carve<std::size_t>(arena->memory, count);
                    if (!counts)
                        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"),
                               void();
                    for (std::size_t row = 0; row != count; ++row) counts[row] = 0;
                    if (status_t const landed = drained(*arena, publish(*arena, counts, count)); failed(landed))
                        return propagate_error(landed, outcome), void();
                    *arguments->matches = {nullptr, nullptr, counts, 0};
                    return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                }
                // A VRAM arena's matches are carved on its device, where the shards' rankings merge
                // in place.
                matches_landing_t landing;
                if (!(arena->device_memory ? carve_matches(arena->device_memory, count, wanted, landing, outcome)
                                           : carve_matches(arena->memory, count, wanted, landing, outcome)))
                    return;
                auto search_with = [&](auto&& predicate) noexcept {
                    return match_dense_rows(
                        variant, arguments->executor, count, arguments->queries, 1, landing,
                        arguments->expansion_factor, arguments->progress, arguments->stats, outcome,
                        [&](auto view, auto rows, auto& executor, auto config, auto keys, auto distances, auto counts) {
                            if constexpr (!is_vram_index<variant_type_>)
                                config.traversal = static_cast<traversal_t>(arguments->traversal);
                            return view.search(rows, executor, config, keys, distances, counts, predicate);
                        });
                };
                if constexpr (is_vram_index<variant_type_>) {
                    if (!search_with(dummy_predicate_t {})) return;
                }
                else if (!(keyed.predicate.callback ? search_with(keyed) : search_with(dummy_predicate_t {}))) return;
                *arguments->matches = landing.view();
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_get(usearch_dense_get_t* arguments) {
    if (submitted<usearch_dense_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    us::utilities::uuid_t const key = uuid_of(arguments->key);
    usearch_scalar_t const scalar = arguments->outputs_scalar;
    *arguments->vectors = {};
    if (arguments->written) *arguments->written = 0;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            specialized_index view {variant};
            std::size_t const dimensions = view.dimensions();
            nk_dtype_t const target_dtype = rows_dtype_of(scalar, dimensions, outcome);
            if (target_dtype == nk_dtype_unknown_k) return;
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              device_of(variant), {}, outcome);
            if (!arena) return;
            usearch_executor_handle_t call_executor;
            typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                             call_executor, outcome);
            if (!executor) return;
            // The single key rides the batched lookups, which size the landing first and fill it
            // second.
            std::size_t entries = 0;
            if (status_t const counted = view.count({&key, 1}, {&entries, 1}, *executor); failed(counted))
                return propagate_error(counted, outcome), void();
            if (!entries) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            std::size_t const row_bytes = us::dense::packed_bytes(target_dtype, dimensions);
            std::byte* rows = carve<std::byte>(arena->memory, entries * row_bytes);
            if (!rows)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            std::size_t total = 0;
            us::dense::rows_span_t const target {rows, entries, dimensions, row_bytes, target_dtype};
            if constexpr (is_vram_index<variant_type_>) {
                // A device exports its stored scalar alone, so rows cast on the host before they
                // land.
                nk_dtype_t const stored_dtype = variant_type_::vector_dtype_k;
                std::size_t const stored_bytes = us::dense::packed_bytes(stored_dtype, dimensions);
                std::byte* const stored = carve<std::byte>(arena->memory, entries * stored_bytes);
                if (!stored)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                if (status_t const got = view.get({&key, 1}, {stored, entries, dimensions, stored_bytes, stored_dtype},
                                                  entries, {&total, 1}, *executor);
                    failed(got))
                    return propagate_error(got, outcome, "Failed to export the rows"), void();
                for (std::size_t row = 0; row != min_of(total, entries); ++row)
                    us::dense::export_row<nk_cap_any_k>(stored + row * stored_bytes, stored_dtype, target, row);
                if (status_t const landed = drained(*arena, publish(*arena, rows, entries * row_bytes)); failed(landed))
                    return propagate_error(landed, outcome, "Failed to land the rows on the device"), void();
            }
            else if (status_t const got = view.get({&key, 1}, target, entries, {&total, 1}, *executor); failed(got))
                return propagate_error(got, outcome, "Failed to export the rows"), void();
            *arguments->vectors = {rows, row_bytes, dimensions, scalar};
            if (arguments->written) *arguments->written = min_of(total, entries);
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_search_stream_init(usearch_dense_search_stream_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->stream = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "No residency streams matches yet");
}

USEARCH_EXPORT void usearch_dense_search_stream_next(usearch_dense_search_stream_next_t* arguments) {
    if (submitted<usearch_dense_search_stream_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->matches = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency streams matches yet");
}

USEARCH_EXPORT void usearch_dense_search_stream_free(usearch_dense_search_stream_t stream) {
    finish_calls_using(stream);
}

#pragma endregion Insertion and Search

#pragma region Cursors and Absorbing

USEARCH_EXPORT void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_vram_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome,
                                "A cursor walks host rows; VRAM rows are read back by key or search");
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
    if (submitted<usearch_dense_cursor_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_rows_view_t* const vectors = arguments->vectors;
    std::size_t const wanted = arguments->wanted;
    *arguments->written = 0;
    *arguments->keys = {};
    if (vectors) *vectors = {};
    if (!arguments->cursor)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor must not be null"), void();
    if (arguments->index != arguments->cursor->opened_on)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    std::atomic_flag& busy = arguments->cursor->busy;
    if (busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull is still running on the cursor"),
               void();
    std::visit(
        [&]<typename cursor_type_>(cursor_type_& walk) noexcept {
            if constexpr (std::is_same_v<cursor_type_, std::monostate>)
                propagate_error(usearch_status_unsupported_k, outcome,
                                "A cursor walks host rows; VRAM rows are read back by key or search");
            else {
                using key_t = typename cursor_type_::key_t;
                using indexes_t = typename cursor_type_::indexes_t;
                static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk reads them");
                // A handle never changes its arm, so it is still the one the cursor was opened on.
                auto& indexes = std::get<indexes_t>(arguments->index->variants);
                std::size_t const dimensions = indexes.dimensions();
                nk_dtype_t const target_dtype = vectors ? rows_dtype_of(arguments->outputs_scalar, dimensions, outcome)
                                                        : nk_dtype_unknown_k;
                if (vectors && target_dtype == nk_dtype_unknown_k) return;
                if (!wanted) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                // An NVMe cursor reads its rows through an executor, a RAM one copies them.
                usearch_executor_handle_t call_executor;
                [[maybe_unused]] typename indexes_t::executor_t* executor = nullptr;
                if constexpr (is_nvme_index<indexes_t>) {
                    executor = executor_of(indexes, arguments->executor, call_executor, outcome);
                    if (!executor) return;
                }
                std::size_t const row_bytes = vectors ? us::dense::packed_bytes(target_dtype, dimensions) : 0;
                usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, wanted);
                std::byte* const rows = vectors && keys ? carve<std::byte>(arena->memory, wanted * row_bytes) : nullptr;
                if (!keys || (vectors && !rows))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                std::span<key_t> const identities {reinterpret_cast<key_t*>(keys), wanted};
                us::dense::rows_span_t const landing = vectors ? us::dense::rows_span_t {rows, wanted, dimensions,
                                                                                         row_bytes, target_dtype}
                                                               : us::dense::rows_span_t {};
                expected<std::size_t> read {0, status_t::success_k};
                if constexpr (is_nvme_index<indexes_t>) {
                    auto const [scanned, status] = walk.read(indexes, identities, landing, *executor);
                    read = {scanned.count, status};
                }
                else read = {walk.read(indexes, identities, landing), status_t::success_k};
                if (!read) return propagate_error(read.status, outcome, "Failed to export the rows"), void();
                if (read.value) {
                    *arguments->keys = {keys, sizeof(usearch_key_t)};
                    if (vectors) *vectors = {rows, row_bytes, dimensions, arguments->outputs_scalar};
                    *arguments->written = read.value;
                }
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->cursor->walk);
    busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_dense_cursor_free(usearch_dense_cursor_t cursor) {
    finish_calls_using(cursor);
    delete cursor;
}

USEARCH_EXPORT void usearch_dense_absorb(usearch_dense_absorb_t* arguments) {
    if (submitted<usearch_dense_absorb>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->count && !arguments->donors)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The donors must name handles"), void();
    visit_writable(*arguments->index, outcome, [&]<typename variant_type_>(variant_type_& variant) noexcept {
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        progress_ticks_t ticks {arguments->progress, arguments->count};
        for (std::size_t position = 0; position != arguments->count; ++position) {
            usearch_dense_index_t& donor = arguments->donors[position];
            if (!donor)
                return propagate_error(usearch_status_invalid_argument_k, outcome, "A donor must not be null"), void();
            // Ahead of the wait below, which a deferred absorb would spend on its own call.
            if (donor == arguments->index)
                return propagate_error(usearch_status_config_mismatch_k, outcome, "An index cannot absorb itself"),
                       void();
            // Every scalar, class and residency is an arm of its own, so a donor of another shape
            // is another arm.
            variant_type_* const typed = std::get_if<variant_type_>(&donor->variants);
            if (!typed)
                return propagate_error(usearch_status_config_mismatch_k, outcome,
                                       "A donor must share the index's residency, scalar and class"),
                       void();
            finish_calls_using(donor);
            status_t const absorbed = specialized_index {variant}.absorb(*typed, *executor).status;
            if (failed(absorbed)) return propagate_error(absorbed, outcome), void();
            delete std::exchange(donor, nullptr);
            if (ticks(position + 1, arguments->count) == continuation_t::cancel_k)
                return propagate_error(usearch_status_cancelled_k, outcome, "The handler stopped the call"), void();
        }
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_sample(usearch_dense_sample_t* arguments) {
    if (submitted<usearch_dense_sample>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    *arguments->links = {};
    *arguments->written = 0;
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency samples yet");
}

#pragma endregion Cursors and Absorbing

#pragma region Linear Algebra

USEARCH_EXPORT void usearch_dense_multiply(usearch_dense_multiply_t* arguments) {
    if (submitted<usearch_dense_multiply>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count;
    *arguments->products = {};
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    if (arguments->stats) *arguments->stats = {};
    if (rejects_columns(*arguments->index, count, arguments->left.columns, outcome)) return;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (variant.kernel() != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A product is a dot, so the kind must be inner product");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                                  arguments->executor, device_of(variant),
                                                                  {arguments->left.data}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                using key_t = typename variant_type_::key_t;
                static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk writes them");
                // A row's scored count lands one past its offset, which it becomes once compacted.
                std::size_t* offsets = carve<std::size_t>(arena->memory, count + 1);
                if (!offsets)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                std::span<std::size_t> const scored {offsets + 1, count};
                usearch_key_t* keys = nullptr;
                usearch_distance_t* scores = nullptr;
                // Every row keeps the pairs its walks scored until `width` of them fill its row,
                // compacted once the counts are known.
                auto walk = [&](std::size_t width, usearch_progress_t progress) noexcept {
                    keys = width ? carve<usearch_key_t>(arena->memory, count * width) : nullptr;
                    scores = width ? carve<usearch_distance_t>(arena->memory, count * width) : nullptr;
                    if (width && (!keys || !scores))
                        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
                    nk::matrix_span<key_t> const kept_keys {reinterpret_cast<key_t*>(keys), count, width};
                    nk::matrix_span<f64_t> const kept_scores {scores, count, width};
                    auto sink = [&](specialized_index<variant_type_> view, us::dense::rows_view_t chunk_rows,
                                    typename variant_type_::executor_t& executor,
                                    typename variant_type_::search_config_t config, std::size_t resolved,
                                    std::size_t taking) noexcept {
                        nk::range_t const chunk = nk::range(resolved, resolved + taking);
                        return view.multiply(chunk_rows, executor, config, kept_keys(chunk, nk::slice),
                                             kept_scores(chunk, nk::slice), scored.subspan(resolved, taking));
                    };
                    return for_each_dense_chunk(variant, arguments->executor, count, arguments->left, 1,
                                                arguments->expansion_factor, progress, arguments->stats, outcome, sink);
                };
                // Keeping every pair needs the widest row first, which only a counting walk can tell.
                std::size_t width = arguments->wanted;
                if (!width) {
                    if (!walk(0, usearch_progress_t {})) return;
                    for (std::size_t const row_scored : scored) width = max_of(width, row_scored);
                }
                if (!walk(width, arguments->progress)) return;

                std::size_t required = 0;
                offsets[0] = 0;
                for (std::size_t row = 0; row != count; ++row) {
                    std::size_t const row_scored = offsets[row + 1], kept = min_of(row_scored, width);
                    required += row_scored;
                    offsets[row + 1] = offsets[row] + kept;
                    if (!kept) continue;
                    // A run starts no later than its own row, so moving rows in order
                    // overwrites only rows already moved.
                    std::memmove(keys + offsets[row], keys + row * width, kept * sizeof(usearch_key_t));
                    std::memmove(scores + offsets[row], scores + row * width, kept * sizeof(usearch_distance_t));
                }
                std::size_t const written = offsets[count];
                if (!written) keys = nullptr, scores = nullptr;
                status_t landed = publish(*arena, keys, written);
                if (succeeded(landed)) landed = publish(*arena, scores, written);
                if (succeeded(landed)) landed = publish(*arena, offsets, count + 1);
                landed = drained(*arena, landed);
                if (failed(landed))
                    return propagate_error(landed, outcome, "Failed to land the products on the device"), void();
                *arguments->products = {keys, scores, offsets};
                if (arguments->written) *arguments->written = written;
                if (arguments->required) *arguments->required = required;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

static_assert(sizeof(us::dense::softmax_state_t) == sizeof(usearch_dense_softmax_state_t));

USEARCH_EXPORT void usearch_dense_attend(usearch_dense_attend_t* arguments) {
    if (submitted<usearch_dense_attend>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_dense_softmax_state_t* const carry = arguments->carry;
    std::size_t const count = arguments->count, head_count = max_of<std::size_t>(arguments->head_count, 1);
    usearch_scalar_t const outputs_scalar = arguments->outputs_scalar ? arguments->outputs_scalar
                                                                      : arguments->queries.scalar;
    if (arguments->stats) *arguments->stats = {};
    *arguments->outputs = {};
    if (!arguments->queries.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The queries must name storage"), void();
    std::size_t const columns = arguments->queries.columns;
    nk_dtype_t const outputs_dtype = rows_dtype_of(outputs_scalar, columns, outcome);
    if (outputs_dtype == nk_dtype_unknown_k) return;
    if (count % head_count)
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "The heads must divide the rows"), void();
    if (rejects_columns(*arguments->index, count, columns, outcome)) return;
    std::size_t const row_bytes = us::dense::packed_bytes(outputs_dtype, columns);
    f32_t const requested_scale = arguments->score_scale;

    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (variant.kernel() != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A softmax weighs logits, so the kind must be inner product");
            else if (!arguments->value_lookup)
                propagate_error(usearch_status_invalid_argument_k, outcome, "The value lookup must be set");
            else if constexpr (gpu_vendor_of_index<variant_type_>() == gpu_vendor_t::apple_k)
                propagate_error(usearch_status_unsupported_k, outcome, "A Metal kernel cannot call the value lookup");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                                  arguments->executor, device_of(variant),
                                                                  {arguments->queries.data, carry}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                std::byte* landing = carve<std::byte>(arena->memory, count * row_bytes);
                if (!landing)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                typename specialized_index<variant_type_>::attend_lookup_t const value_lookup {
                    arguments->value_lookup, arguments->value_lookup_state,
                    usearch_scalar_of(variant_type_::vector_dtype_k), columns};
                us::dense::rows_span_t const outputs {landing, count, columns, row_bytes, outputs_dtype};
                // The engine's carry mirrors the C one, so it writes the caller's states in place.
                std::span<us::dense::softmax_state_t> const carried {
                    reinterpret_cast<us::dense::softmax_state_t*>(carry), carry ? count : 0};
                auto sink = [&](specialized_index<variant_type_> view, us::dense::rows_view_t chunk_rows,
                                typename variant_type_::executor_t& executor,
                                typename variant_type_::search_config_t config, std::size_t resolved,
                                std::size_t taking) noexcept {
                    f32_t const score_scale = requested_scale > 0
                                                  ? requested_scale
                                                  : 1 / std::sqrt(static_cast<f32_t>(view.dimensions()));
                    return view.attend(chunk_rows, head_count, value_lookup, score_scale,
                                       outputs.slice(resolved, taking),
                                       carried.empty() ? carried : carried.subspan(resolved, taking), executor, config);
                };
                if (!for_each_dense_chunk(variant, arguments->executor, count, arguments->queries, head_count,
                                          arguments->expansion_factor, arguments->progress, arguments->stats, outcome,
                                          sink))
                    return;
                if (status_t const landed = drained(*arena, publish(*arena, landing, count * row_bytes));
                    failed(landed))
                    return propagate_error(landed, outcome, "Failed to land the outputs on the device"), void();
                *arguments->outputs = {landing, row_bytes, columns, outputs_scalar};
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_maxsim(usearch_dense_maxsim_t* arguments) {
    if (submitted<usearch_dense_maxsim>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, rows_per_query = arguments->rows_per_query;
    std::size_t const wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (!rows_per_query)
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "Every query needs at least one row"), void();
    if (rejects_columns(*arguments->index, count, arguments->queries.columns, outcome)) return;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if (variant.kernel() != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "Late interaction sums maxima over logits, so the kind must be inner product");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                                  arguments->executor, device_of(variant),
                                                                  {arguments->queries.data}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                matches_landing_t landing;
                if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
                if (!match_dense_rows(
                        variant, arguments->executor, count, arguments->queries, rows_per_query, landing,
                        arguments->expansion_factor, arguments->progress, arguments->stats, outcome,
                        [&](auto view, auto rows, auto& executor, auto config, auto keys, auto scores, auto counts) {
                            return view.maxsim(rows, rows_per_query, executor, config, keys, scores, counts);
                        }))
                    return;
                if (!publish_matches(*arena, count, landing, outcome)) return;
                *arguments->matches = landing.view();
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

#pragma endregion Linear Algebra

#pragma region Serialization

USEARCH_EXPORT void usearch_dense_load_ram(usearch_dense_load_ram_t* arguments) {
    if (submitted<usearch_dense_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    load_dense_index(as_text(arguments->path), arguments->ram, arguments->executor, arguments->progress,
                     arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments) {
    if (submitted<usearch_dense_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    load_dense_index(as_text(arguments->path), arguments->vram, arguments->executor, arguments->progress,
                     arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments) {
    if (submitted<usearch_dense_load_ram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    load_dense_index(as_bytes(arguments->buffer), arguments->ram, arguments->executor, arguments->progress,
                     arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments) {
    if (submitted<usearch_dense_load_vram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    // The header parses on the host, so a monolith on an NVIDIA or AMD device, where a VRAM
    // `usearch_dense_save_buffer` lands it, is read back first; Apple's unified memory is read in
    // place.
    bytes_view_t source = as_bytes(arguments->buffer);
    growing_buffer<byte_t> staged {global_allocator};
    int const nvidia = cuda_device_of(source.data()), amd = nvidia < 0 ? hip_device_of(source.data()) : -1;
    if (nvidia >= 0 || amd >= 0) {
        if (failed(staged.resize_uninitialized(source.size())))
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to stage the buffer"), void();
        bool const copied = nvidia >= 0 ? cuda_set_device(nvidia) == cuda_success_k &&
                                              cuda_copy(staged.data(), source.data(), source.size()) == cuda_success_k
                                        : hip_set_device(amd) == hip_success_k &&
                                              hip_api().copy(staged.data(), source.data(), source.size(),
                                                             hip_copy_device_to_host_k) == hip_success_k;
        if (!copied)
            return propagate_error(status_t::device_error_k, outcome, "Failed to read the buffer off its device"),
                   void();
        source = to_span(staged);
    }
    load_dense_index(source, arguments->vram, arguments->executor, arguments->progress, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_save(usearch_dense_save_t* arguments) {
    if (submitted<usearch_dense_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->path.data || !arguments->path.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The path must not be empty"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_nvme_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome, "An NVMe view already is its checkpoint");
            else {
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                                 call_executor, outcome);
                if (!executor) return;
                progress_ticks_t ticks {arguments->progress, specialized_index {variant}.shards_count()};
                propagate_error(specialized_index {variant}.save(as_text(arguments->path), *executor, ticks), outcome);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_save_buffer(usearch_dense_save_buffer_t* arguments) {
    if (submitted<usearch_dense_save_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->monolith = {};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_nvme_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome, "An NVMe view already is its checkpoint");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome,
                                                                  arguments->executor, device_of(variant), {}, outcome);
                if (!arena) return;
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                                 call_executor, outcome);
                if (!executor) return;
                auto const [length, status] = specialized_index {variant}.storage_requirements();
                if (failed(status)) return propagate_error(status, outcome), void();
                std::byte* monolith = carve<std::byte>(arena->memory, length);
                if (!monolith)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                progress_ticks_t ticks {arguments->progress, specialized_index {variant}.shards_count()};
                status_t const saved = specialized_index {variant}.save(
                    bytes_span_t {reinterpret_cast<byte_t*>(monolith), length}, *executor, ticks);
                if (failed(saved)) return propagate_error(saved, outcome), void();
                // A VRAM monolith is written on the host, then lands on the arena's device
                // in one upload.
                if (status_t const landed = drained(*arena, publish(*arena, monolith, length)); failed(landed))
                    return propagate_error(landed, outcome, "Failed to land the monolith on the device"), void();
                *arguments->monolith = {reinterpret_cast<uint8_t const*>(monolith), length};
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_metadata_read(usearch_dense_metadata_read_t* arguments) {
    if (submitted<usearch_dense_metadata_read>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->path.data || !arguments->path.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The path must not be empty"), void();
    auto const [source, status] = safegraphs_read_metadata(as_text(arguments->path));
    if (failed(status)) return propagate_error(status, outcome), void();
    adopt_metadata(source, arguments->metadata);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_metadata_read_buffer(usearch_dense_metadata_read_buffer_t* arguments) {
    if (submitted<usearch_dense_metadata_read_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (!arguments->buffer.data || !arguments->buffer.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The buffer must not be empty"), void();
    auto const [source, status] = safegraphs_read_metadata(as_bytes(arguments->buffer));
    if (failed(status)) return propagate_error(status, outcome), void();
    adopt_metadata(source, arguments->metadata);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

#pragma region Executors

USEARCH_EXPORT usearch_status_t usearch_outcome_wait(usearch_outcome_t* outcome) {
    finish_calls_using(outcome);
    std::atomic_ref<usearch_status_t> status {outcome->status};
    usearch_status_t seen = status.load(std::memory_order_acquire);
    for (; seen == usearch_status_deferred_k; seen = status.load(std::memory_order_acquire))
        status.wait(usearch_status_deferred_k, std::memory_order_acquire);
    return seen;
}

USEARCH_EXPORT void usearch_arena_init(usearch_arena_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->arena = nullptr;
    usearch_arena_t const arena = new (std::nothrow) usearch_arena_handle_t {};
    if (!arena) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the arena"), void();
    // A VRAM executor lands its outputs on its vendor's first stream; every other executor computes
    // on the host.
    if (arguments->executor)
        if (gpu_executor_t* const devices = std::get_if<gpu_executor_t>(&arguments->executor->executor)) {
            arena->device_memory = gpu_arena_t::on(*devices);
            arena->device = arena->device_memory.device();
            arena->executor = arguments->executor;
        }
    *arguments->arena = arena;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_arena_free(usearch_arena_t arena) {
    finish_calls_using(arena);
    delete arena;
}

USEARCH_EXPORT void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    if (!spawn_ram_executor(*executor, arguments->threads, outcome)) return delete executor, void();
    hand_out_executor(executor, arguments->deferred_limit, arguments->executor, outcome);
}

USEARCH_EXPORT void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Borrowed pools are not wired yet");
}

USEARCH_EXPORT void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->executor = nullptr;
    usearch_vram_options_t const& vram = arguments->vram;
    if (arguments->streams)
        return propagate_error(usearch_status_unsupported_k, outcome, "Borrowed streams are not wired yet"), void();
    // The ABI names USearch ordinals, which the listing resolves to each vendor's own; the spawn
    // keeps one stream per distinct device.
    std::span<gpu_device_t const> const listed = gpu_devices();
    growing_buffer<gpu_device_t> devices {global_allocator};
    if (failed(devices.resize(named_count(vram))))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), void();
    for (std::size_t shard = 0; shard != devices.size(); ++shard) {
        std::size_t const ordinal = named_ordinal(vram, shard);
        if (ordinal >= listed.size())
            return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), void();
        devices[shard] = listed[ordinal];
    }
    auto [spawned, status] = spawn_gpu_executor(devices, arguments->scratch_limit);
    if (status == status_t::config_mismatch_k)
        return propagate_error(status, outcome, "One executor drives one GPU vendor"), void();
    if (failed(status)) return propagate_error(status, outcome), void();
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    executor->executor = std::move(spawned);
    hand_out_executor(executor, arguments->deferred_limit, arguments->executor, outcome);
}

static_assert(static_cast<u32_t>(gpu_vendor_t::apple_k) == usearch_gpu_vendor_apple_k);
static_assert(static_cast<u32_t>(gpu_class_t::apple_metal10_k) == usearch_gpu_class_apple_metal10_k);

USEARCH_EXPORT void usearch_gpu_devices(usearch_gpu_devices_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    std::span<gpu_device_t const> const listed = gpu_devices();
    std::size_t const copied = arguments->devices ? min_of(listed.size(), arguments->devices_capacity) : 0;
    for (std::size_t position = 0; position != copied; ++position) {
        gpu_device_t const& device = listed[position];
        usearch_gpu_device_t& landed = arguments->devices[position];
        landed = {static_cast<usearch_gpu_vendor_t>(device.vendor),
                  device.vendor_ordinal,
                  static_cast<usearch_gpu_class_t>(device.gpu_class),
                  device.compute_units,
                  device.memory_bytes,
                  {}};
        std::memcpy(landed.name, device.name, sizeof(landed.name));
    }
    if (arguments->devices_count) *arguments->devices_count = listed.size();
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    usearch_nvme_options_t const& nvme = arguments->nvme;
    *arguments->executor = nullptr;
    if (nvme.paths_count && !nvme.paths)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The paths must name storage"), void();

    // A transport identifier names a namespace the kernel no longer owns, which only a userspace
    // engine reaches, so one executor serves either kernel paths or transport identifiers, never
    // both.
    std::size_t transport_ids = 0, pcie_ids = 0;
    for (std::size_t path = 0; path != nvme.paths_count; ++path) {
        std::string_view const named = as_text(nvme.paths[path]);
        transport_ids += named.find("trtype:") != std::string_view::npos;
        pcie_ids += named.find("trtype:PCIe") != std::string_view::npos;
    }
    if (transport_ids && transport_ids != nvme.paths_count)
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "Kernel paths and transport identifiers need different engines"),
               void();
    io_engine_config_t config;
    config.queue_depth = nvme_queue_depth_k;
    if (transport_ids) {
#if USEARCH_WITH_LINUX_SPDK
        // Fabrics copy every byte through a socket, so only PCIe controllers need hugepages to DMA
        // into.
        linux_spdk_driver_t::config_t environment;
        environment.memory = pcie_ids ? linux_spdk_memory_t::hugepages_k : linux_spdk_memory_t::plain_pages_k;
        auto [driver, open_status] = linux_spdk_driver_t::open(environment);
        if (failed(open_status))
            return propagate_error(open_status, outcome, "The SPDK environment did not start"), void();
        config.transport = io_transport_t::linux_spdk_k;
        config.spdk_driver = std::move(driver);
#else
        (void)pcie_ids;
        return propagate_error(usearch_status_unsupported_k, outcome, "Transport identifiers need an SPDK build"),
               void();
#endif
    }
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    if (!spawn_nvme_executor(*executor, arguments->threads, config, outcome)) return delete executor, void();
    hand_out_executor(executor, arguments->deferred_limit, arguments->executor, outcome);
}

USEARCH_EXPORT void usearch_executor_drain(usearch_executor_drain_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    // A null executor runs every call to completion before returning, so nothing waits behind it.
    if (!arguments->executor) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    usearch_executor_handle_t& executor = *arguments->executor;
    // A drain from a callback meets the claim of the call running it, and answers contended rather
    // than waiting on itself.
    auto const claimable = [](auto& typed) noexcept { return typed.claim().status; };
    if (status_t const claimed = visit_executor(executor, claimable); failed(claimed))
        return propagate_error(claimed, outcome), void();
    std::size_t last = 0;
    {
        std::lock_guard<std::mutex> lock(deferred_registry.guard);
        last = deferred_registry.tickets;
    }
    run_held(executor, last);
    wait_running(executor, last);
    auto const drain = [](auto& typed) noexcept {
        auto const [claim, status] = typed.claim();
        return failed(status) ? status : typed.drain();
    };
    propagate_error(visit_executor(executor, drain), outcome);
}

USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor) {
    if (!executor) return;
    finish_calls_using(executor);
    delist_executor(*executor);
    delete executor;
}

#pragma endregion Executors

} // extern "C"
