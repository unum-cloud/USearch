/**
 *  @file src/dense_index.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense modality's C surface: the variant grid over every compiled backend, the handle
 *      owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Compiled twice: as @c dense_index.cpp for the portable library and as @c dense_index.cu for
 *  the CUDA one, with @c USEARCH_BUILD_CUDA deciding whether the VRAM arms join the variant.
 *  Arms are ordered newest-capability-first inside each @c (dtype,kernel) cell, since a machine
 *  reporting Ice Lake also reports Haswell and Skylake; the serial floor is last and
 *  unconditional. The arms' bodies compile in the per-class units, never here.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cmath>   // `std::sqrt` - attention's default logit scale
#include <cstdio>  // `std::snprintf` - composing the capability string
#include <cstring> // `std::memcpy`, `std::strlen` - scattering staged rows & the capability string

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
#include <variant>            // `std::variant`, `std::visit`

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
 *  Arms exist per wired @c (dtype,kernel,capability) cell - those whose NumKong kernel
 *  kits are adapted - never the full cross product: an unwired cell would not compile, and
 *  advertising it would only move the failure from the build to the caller. The VRAM
 *  siblings join under the CUDA build, and the read-only NVMe siblings wherever a CPU
 *  class picks a kit of its own.
 */
struct usearch_dense_index_handle_t {

    using variants_t = std::variant<
#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_HASWELL
        // intel_haswell over serial by haswell: every dtype, 33 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>,     //
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
        // intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd: bf16 f16 e5m2 e4m3 e3m2 i4 u4 u1, 24 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_sierra_k>,     //
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
        // intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd, alder, sierra: e2m3 i8 u8, 9 arms
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
#endif
#if NUMKONG_TARGET_SKYLAKE
        // intel_skylake over intel_haswell by skylake: bf16 f16 e5m2 e4m3 e3m2 e2m3, 18 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
#endif
#if NUMKONG_TARGET_ICELAKE
        // intel_icelake over intel_skylake by icelake: i8 i4 u8 u4 u1, 15 arms
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>,   //
#endif
#if NUMKONG_TARGET_GENOA
        // amd_genoa over intel_icelake by genoa: bf16 e5m2 e4m3, 9 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>, //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>, //
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
        // intel_diamond over amd_genoa by x86_tpause, x86_cmpccxadd: bf16 f16 e3m2 i4 u4 u1, 18 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,  //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_diamond_k>,     //
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
        // intel_diamond over amd_genoa by x86_tpause, x86_cmpccxadd, diamond: e5m2 e4m3, 6 arms
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
        // intel_diamond over amd_genoa by x86_tpause, x86_cmpccxadd, alder, sierra: e2m3 i8 u8, 9 arms
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
#endif
#elif NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEON
        // aws_g2 over serial by neon: bf16 f16 u1, 9 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>,  //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>,     //
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONBFDOT
        // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonbfdot: bf16, 3 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
        // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonfhm: f16 e5m2 e4m3, 9 arms
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
        // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonsdot: e3m2 e2m3 i8 i4 u8 u4, 18 arms
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
        // apple_m2 over aws_g2 by arm64_lse, arm64_rcpc: u1, 3 arms
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::apple_m2_k>, //
#endif
#if FORKUNION_TARGET_ARM64_WFET
        // aws_g5 over apple_m2 by arm64_wfet: every dtype, 33 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g5_k>,     //
#endif
#if NUMKONG_TARGET_NEONFP8
        // nvidia_vera over aws_g5 by neonfp8: e5m2 e4m3 e3m2 e2m3, 12 arms
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
#endif
#elif NUMKONG_ARCH_RISCV64_
#if FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
        // alibaba_c930 over serial by risc5_wrs, risc5_zacas: every dtype, 33 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::alibaba_c930_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::alibaba_c930_k>,     //
#endif
#elif NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128
        // wasm32_v128 over serial by v128: bf16 i8 u8 u1, 12 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::wasm32_v128_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::wasm32_v128_k>,     //
#endif
#if NUMKONG_TARGET_V128RELAXED
        // wasm64_extended over wasm32_v128 by v128relaxed: bf16 f16 e5m2 e4m3 e3m2 e2m3 i8 i4 u8 u4, 30 arms
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>,   //
#endif
#endif // NUMKONG_ARCH_*_
#if USEARCH_BUILD_CUDA
        // nvidia_blackwellrtx over nvidia_ampere by "120f": every cell, 33 arms
        vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,       //
        vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>, //
        vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,        //
        vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,    //
        vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>,  //
        vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,       //
        vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>, //
        vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,       //
        vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>, //
        vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,       //
        vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>, //
        vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,       //
        vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>, //
        vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,         //
        vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,         //
        vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,     //
        vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,         //
        vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,         //
        vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>,     //
        vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>,   //
        vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>,         //
        vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_blackwellrtx_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_blackwellrtx_k>,     //
        // nvidia_blackwell over nvidia_ampere by "100f": every cell, 33 arms
        vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,       //
        vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>, //
        vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,        //
        vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,    //
        vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>,  //
        vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,       //
        vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>, //
        vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,       //
        vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>, //
        vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,       //
        vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>, //
        vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,       //
        vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>, //
        vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,         //
        vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,         //
        vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,     //
        vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,         //
        vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,         //
        vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>,     //
        vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>,   //
        vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>,         //
        vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_blackwell_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_blackwell_k>,     //
        // nvidia_hopper over nvidia_ampere by "90a": every cell, 33 arms
        vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,       //
        vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>, //
        vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,        //
        vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,    //
        vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>,  //
        vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,       //
        vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>, //
        vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,       //
        vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>, //
        vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,       //
        vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>, //
        vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,       //
        vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>, //
        vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,         //
        vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,         //
        vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,     //
        vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,         //
        vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,         //
        vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>,     //
        vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>,   //
        vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>,         //
        vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_hopper_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_hopper_k>,     //
        // nvidia_ada over nvidia_ampere by "89": every cell, 33 arms
        vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,       //
        vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>, //
        vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,        //
        vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,    //
        vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>,  //
        vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,       //
        vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>, //
        vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,       //
        vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>, //
        vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,       //
        vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>, //
        vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,       //
        vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>, //
        vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,         //
        vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,         //
        vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,     //
        vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,         //
        vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,         //
        vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>,     //
        vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>,   //
        vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>,         //
        vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_ada_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_ada_k>,     //
        // nvidia_ampere, the floor by "80", its PTX serving every GPU without a class: every cell, 33 arms
        vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,       //
        vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>, //
        vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,        //
        vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,    //
        vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>,  //
        vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,       //
        vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>, //
        vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,       //
        vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>, //
        vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,       //
        vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>, //
        vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,       //
        vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>, //
        vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,         //
        vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,         //
        vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,     //
        vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,         //
        vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,         //
        vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>,     //
        vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>,   //
        vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>,         //
        vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_ampere_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_ampere_k>,     //
#endif
#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_HASWELL
        // NVMe intel_haswell over serial by haswell: every dtype, 33 arms
        nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,        //
        nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,    //
        nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,  //
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,     //
        nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>,   //
        nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,         //
        nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>,     //
        nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>,     //
#endif
#if NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
        // NVMe intel_sierra over intel_haswell by alder, sierra: e2m3 i8 u8, 9 arms
        nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
#endif
#if NUMKONG_TARGET_SKYLAKE
        // NVMe intel_skylake over intel_haswell by skylake: bf16 f16 e5m2 e4m3 e3m2 e2m3, 18 arms
        nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,        //
        nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,    //
        nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>,  //
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
#endif
#if NUMKONG_TARGET_ICELAKE
        // NVMe intel_icelake over intel_skylake by icelake: i8 i4 u8 u4 u1, 15 arms
        nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>,   //
        nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>,   //
#endif
#if NUMKONG_TARGET_GENOA
        // NVMe amd_genoa over intel_icelake by genoa: bf16 e5m2 e4m3, 9 arms
        nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,       //
        nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,   //
        nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>, //
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>, //
#endif
#if NUMKONG_TARGET_DIAMOND
        // NVMe intel_diamond over amd_genoa by diamond: e5m2 e4m3, 6 arms
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
#endif
#elif NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEON
        // NVMe aws_g2 over serial by neon: bf16 f16 u1, 9 arms
        nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,       //
        nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>,   //
        nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>, //
        nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,        //
        nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>,    //
        nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>,  //
        nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,         //
        nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>,     //
        nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>,     //
#endif
#if NUMKONG_TARGET_NEONBFDOT
        // NVMe apple_m2 over aws_g2 by neonbfdot: bf16, 3 arms
        nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
#endif
#if NUMKONG_TARGET_NEONFHM
        // NVMe apple_m2 over aws_g2 by neonfhm: f16 e5m2 e4m3, 9 arms
        nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,        //
        nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,    //
        nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,  //
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
#endif
#if NUMKONG_TARGET_NEONSDOT
        // NVMe apple_m2 over aws_g2 by neonsdot: e3m2 e2m3 i8 i4 u8 u4, 18 arms
        nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
        nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,         //
        nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,     //
        nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,   //
#endif
#if NUMKONG_TARGET_NEONFP8
        // NVMe nvidia_vera over aws_g5 by neonfp8: e5m2 e4m3 e3m2 e2m3, 12 arms
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
#endif
#endif // NUMKONG_ARCH_*_
       // The NVMe serial floor: every wired cell, every platform.
        nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::serial_k>,        //
        nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::serial_k>,    //
        nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,  //
        nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::serial_k>,     //
        nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::serial_k>,     //
        // The serial floor: every wired cell, every platform, the standard yield.
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::serial_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::serial_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::serial_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::serial_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::serial_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::serial_k>>;    //

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
    explicit usearch_dense_index_handle_t(std::in_place_type_t<variant_type_> tag,
                                          forkunion_allocator_t allocator) noexcept
        : variants(tag, allocator) {}
};

/** The arms' explicit instantiation declarations: every body compiles in its class's unit, so this
 *  unit only visits. */
#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_HASWELL
/* intel_haswell over serial by haswell: every dtype, 33 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>>;
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
/* intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd: bf16 f16 e5m2 e4m3 e3m2 i4 u4 u1, 24 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_sierra_k>>;
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* intel_sierra over intel_haswell by x86_tpause, x86_cmpccxadd, alder, sierra: e2m3 i8 u8, 9 arms */
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
#endif
#if NUMKONG_TARGET_SKYLAKE
/* intel_skylake over intel_haswell by skylake: bf16 f16 e5m2 e4m3 e3m2 e2m3, 18 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
#endif
#if NUMKONG_TARGET_ICELAKE
/* intel_icelake over intel_skylake by icelake: i8 i4 u8 u4 u1, 15 arms */
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>>;
#endif
#if NUMKONG_TARGET_GENOA
/* amd_genoa over intel_icelake by genoa: bf16 e5m2 e4m3, 9 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD
/* intel_diamond over amd_genoa by x86_tpause, x86_cmpccxadd: bf16 f16 e3m2 i4 u4 u1, 18 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_diamond_k>>;
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_DIAMOND
/* intel_diamond over amd_genoa by x86_tpause, x86_cmpccxadd, diamond: e5m2 e4m3, 6 arms */
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
#endif
#if FORKUNION_TARGET_X86_TPAUSE || FORKUNION_TARGET_X86_CMPCCXADD || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* intel_diamond over amd_genoa by x86_tpause, x86_cmpccxadd, alder, sierra: e2m3 i8 u8, 9 arms */
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
#endif
#elif NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEON
/* aws_g2 over serial by neon: bf16 f16 u1, 9 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONBFDOT
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonbfdot: bf16, 3 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONFHM
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonfhm: f16 e5m2 e4m3, 9 arms */
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC || NUMKONG_TARGET_NEONSDOT
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc, neonsdot: e3m2 e2m3 i8 i4 u8 u4, 18 arms */
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_LSE || FORKUNION_TARGET_ARM64_RCPC
/* apple_m2 over aws_g2 by arm64_lse, arm64_rcpc: u1, 3 arms */
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::apple_m2_k>>;
#endif
#if FORKUNION_TARGET_ARM64_WFET
/* aws_g5 over apple_m2 by arm64_wfet: every dtype, 33 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g5_k>>;
#endif
#if NUMKONG_TARGET_NEONFP8
/* nvidia_vera over aws_g5 by neonfp8: e5m2 e4m3 e3m2 e2m3, 12 arms */
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
#endif
#elif NUMKONG_ARCH_RISCV64_
#if FORKUNION_TARGET_RISC5_WRS || FORKUNION_TARGET_RISC5_ZACAS
/* alibaba_c930 over serial by risc5_wrs, risc5_zacas: every dtype, 33 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::alibaba_c930_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::alibaba_c930_k>>;
#endif
#elif NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128
/* wasm32_v128 over serial by v128: bf16 i8 u8 u1, 12 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::wasm32_v128_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::wasm32_v128_k>>;
#endif
#if NUMKONG_TARGET_V128RELAXED
/* wasm64_extended over wasm32_v128 by v128relaxed: bf16 f16 e5m2 e4m3 e3m2 e2m3 i8 i4 u8 u4, 30 arms */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
#endif
#endif // NUMKONG_ARCH_*_
#if USEARCH_BUILD_CUDA
/* nvidia_blackwellrtx over nvidia_ampere by "120f": every cell, 33 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<
    vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_blackwellrtx_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_blackwellrtx_k>>;
/* nvidia_blackwell over nvidia_ampere by "100f": every cell, 33 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_blackwell_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_blackwell_k>>;
/* nvidia_hopper over nvidia_ampere by "90a": every cell, 33 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_hopper_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_hopper_k>>;
/* nvidia_ada over nvidia_ampere by "89": every cell, 33 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_ada_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_ada_k>>;
/* nvidia_ampere, the floor by "80", its PTX serving every GPU without a class: every cell, 33 arms */
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_ampere_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_ampere_k>>;
#endif
/* The serial floor: every wired cell, every platform, the standard yield. */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::serial_k>>;
#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_HASWELL
/* NVMe intel_haswell over serial by haswell: every dtype, 33 arms */
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>>;
#endif
#if NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA
/* NVMe intel_sierra over intel_haswell by alder, sierra: e2m3 i8 u8, 9 arms */
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
#endif
#if NUMKONG_TARGET_SKYLAKE
/* NVMe intel_skylake over intel_haswell by skylake: bf16 f16 e5m2 e4m3 e3m2 e2m3, 18 arms */
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
#endif
#if NUMKONG_TARGET_ICELAKE
/* NVMe intel_icelake over intel_skylake by icelake: i8 i4 u8 u4 u1, 15 arms */
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>>;
#endif
#if NUMKONG_TARGET_GENOA
/* NVMe amd_genoa over intel_icelake by genoa: bf16 e5m2 e4m3, 9 arms */
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
#endif
#if NUMKONG_TARGET_DIAMOND
/* NVMe intel_diamond over amd_genoa by diamond: e5m2 e4m3, 6 arms */
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
#endif
#elif NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEON
/* NVMe aws_g2 over serial by neon: bf16 f16 u1, 9 arms */
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>>;
#endif
#if NUMKONG_TARGET_NEONBFDOT
/* NVMe apple_m2 over aws_g2 by neonbfdot: bf16, 3 arms */
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
#endif
#if NUMKONG_TARGET_NEONFHM
/* NVMe apple_m2 over aws_g2 by neonfhm: f16 e5m2 e4m3, 9 arms */
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
#endif
#if NUMKONG_TARGET_NEONSDOT
/* NVMe apple_m2 over aws_g2 by neonsdot: e3m2 e2m3 i8 i4 u8 u4, 18 arms */
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
#endif
#if NUMKONG_TARGET_NEONFP8
/* NVMe nvidia_vera over aws_g5 by neonfp8: e5m2 e4m3 e3m2 e2m3, 12 arms */
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
#endif
#endif // NUMKONG_ARCH_*_
/* The NVMe serial floor: every wired cell, every platform. */
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::serial_k>>;
extern template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::serial_k>>;

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
    constexpr nk_capability_t picked = dense_quads_chain<multi_type_::kernel_kind_k, typename multi_type_::scalar_t,
                                                         multi_type_::capability_k>::picked_k;
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
                                                     placement_type_ const& placement,
                                                     usearch_outcome_t* outcome) noexcept {
    usearch_dense_index_t const handle = new (std::nothrow)
        usearch_dense_index_handle_t(std::in_place_type_t<variant_type_>(), host_allocator_of(placement));
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

/** Whether one rung is genuinely distinct from the class it degrades to: a different picked kit or
 *  a different ForkUnion mask - the one fact justifying its instantiation. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_>
consteval bool rung_advances() noexcept {
    constexpr cpu_class_t parent_k = parent_class_of(cpu_class_);
    using scalar_t = typename nk::type_for<dtype_>::type;
    using upper_chain_t = dense_quads_chain<kernel_, scalar_t, numkong_capabilities_of(cpu_class_)>;
    using lower_chain_t = dense_quads_chain<kernel_, scalar_t, numkong_capabilities_of(parent_k)>;
    return upper_chain_t::picked_k != lower_chain_t::picked_k ||
           forkunion_capabilities_of<cpu_class_>() != forkunion_capabilities_of<parent_k>();
}

/** The unconditional serial floor - every ladder's last rung. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_capability_t,
                                                usearch_outcome_t* outcome) noexcept {
    return make_punned_dense_index<ram_index<dtype_, kernel_, cpu_class_t::serial_k>>(geometry, ram, outcome);
}

/** One rung: taken when the machine reports the class, else falls through; a class that picks
 *  nothing new is skipped without instantiating its index. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
static usearch_dense_index_t descend_ram_ladder(usearch_dense_metadata_t const& geometry,
                                                usearch_ram_options_t const& ram, nk_capability_t available,
                                                usearch_outcome_t* outcome) noexcept {
    if constexpr (rung_advances<dtype_, kernel_, cpu_class_>()) {
        if (numkong_admits(available, cpu_class_) && forkunion_admits<cpu_class_>())
            return make_punned_dense_index<ram_index<dtype_, kernel_, cpu_class_>>(geometry, ram, outcome);
    }
    else {
        static_assert(
            !variant_carries<ram_index<dtype_, kernel_, cpu_class_>, usearch_dense_index_handle_t::variants_t>,
            "This rung no longer advances past the one below - drop its arm from the variant list");
    }
    return descend_ram_ladder<dtype_, kernel_, lower_classes_...>(geometry, ram, available, outcome);
}

/** Picks the RAM variant for one @c (dtype,kernel) cell: CPU classes newest-first, each rung
 *  compiled only where it advances past the class it degrades to. A refused rung falls to the next
 *  in the list, so lineages that share a floor interleave. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_ram_options_t const& ram,
                                        usearch_outcome_t* outcome) noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::intel_diamond_k, cpu_class_t::amd_genoa_k,
                              cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k, cpu_class_t::intel_sierra_k,
                              cpu_class_t::intel_haswell_k>( //
        geometry, ram, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::nvidia_vera_k, cpu_class_t::aws_g5_k,
                              cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>( //
        geometry, ram, available, outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::alibaba_c930_k>(geometry, ram, available, outcome);
#elif NUMKONG_ARCH_WASM_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::wasm64_extended_k, cpu_class_t::wasm32_v128_k>( //
        geometry, ram, available, outcome);
#else
    return descend_ram_ladder<dtype_, kernel_>(geometry, ram, available, outcome);
#endif
}

/** Whether one NVMe rung picks a different kit than the class it degrades to: a storage walk
 *  carries no waiter or atomic reference, so only the kit tells two classes apart. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_>
consteval bool nvme_rung_advances() noexcept {
    using scalar_t = typename nk::type_for<dtype_>::type;
    return dense_quads_chain<kernel_, scalar_t, numkong_capabilities_of(cpu_class_)>::picked_k !=
           dense_quads_chain<kernel_, scalar_t, numkong_capabilities_of(parent_class_of(cpu_class_))>::picked_k;
}

/** The unconditional serial floor of every NVMe ladder. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t descend_nvme_ladder(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                                 nk_capability_t, usearch_outcome_t* outcome) noexcept {
    return make_punned_dense_index<nvme_index<dtype_, kernel_, cpu_class_t::serial_k>>(geometry, nvme, outcome);
}

/** One NVMe rung: taken when the machine reports the class's kits, else falls through; a class that
 *  picks no kit of its own is skipped without instantiating its index. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
static usearch_dense_index_t descend_nvme_ladder(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                                 nk_capability_t available, usearch_outcome_t* outcome) noexcept {
    if constexpr (nvme_rung_advances<dtype_, kernel_, cpu_class_>()) {
        if (numkong_admits(available, cpu_class_))
            return make_punned_dense_index<nvme_index<dtype_, kernel_, cpu_class_>>(geometry, nvme, outcome);
    }
    else {
        // A class sharing its parent's mask names the parent's very arm, which the variant rightly
        // carries.
        static_assert(
            numkong_capabilities_of(cpu_class_) == numkong_capabilities_of(parent_class_of(cpu_class_)) ||
                !variant_carries<nvme_index<dtype_, kernel_, cpu_class_>, usearch_dense_index_handle_t::variants_t>,
            "This NVMe rung no longer advances past the one below - drop its arm from the variant list");
    }
    return descend_nvme_ladder<dtype_, kernel_, lower_classes_...>(geometry, nvme, available, outcome);
}

/** Picks the NVMe variant for one @c (dtype,kernel) cell on the RAM ladder's classes, the
 *  WebAssembly ones aside, since a module reads no storage directly. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                        usearch_outcome_t* outcome) noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_nvme_ladder<dtype_, kernel_, cpu_class_t::intel_diamond_k, cpu_class_t::amd_genoa_k,
                               cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k, cpu_class_t::intel_sierra_k,
                               cpu_class_t::intel_haswell_k>( //
        geometry, nvme, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_nvme_ladder<dtype_, kernel_, cpu_class_t::nvidia_vera_k, cpu_class_t::aws_g5_k,
                               cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>( //
        geometry, nvme, available, outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_nvme_ladder<dtype_, kernel_, cpu_class_t::alibaba_c930_k>(geometry, nvme, available, outcome);
#else
    return descend_nvme_ladder<dtype_, kernel_>(geometry, nvme, available, outcome);
#endif
}

#if USEARCH_BUILD_CUDA

/** The VRAM cells with adapted CUDA kits - the device half of the variant list. */
constexpr bool vram_cell_wired(nk_dtype_t dtype, nk_kernel_kind_t kernel) noexcept {
    switch (dtype) {
    case nk_bf16_k:
        return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_f16_k: return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_e5m2_k:
        return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_e4m3_k:
        return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_e3m2_k:
        return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_e2m3_k:
        return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_i8_k: return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_i4_k: return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_u8_k: return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_u4_k: return kernel == nk_kernel_dot_k || kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_u1_k: return kernel == nk_kernel_dot_k || kernel == nk_kernel_hamming_k || kernel == nk_kernel_jaccard_k;
    default: return false;
    }
}

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
 *  alone carries cubins for them, the way the CPU ladder picks the class the machine reports. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_vram_options_t const& vram,
                                        usearch_outcome_t* outcome) noexcept {
    if constexpr (!vram_cell_wired(dtype_, kernel_))
        return propagate_error(usearch_status_unknown_kind_k, outcome,
                               "No CUDA kit is wired for this (quantization,metric) pair"),
               nullptr;
    else switch (shared_gpu_class(vram)) {
        case gpu_class_t::nvidia_blackwellrtx_k:
            return make_punned_dense_index<vram_index<dtype_, kernel_, gpu_class_t::nvidia_blackwellrtx_k>>(
                geometry, vram, outcome);
        case gpu_class_t::nvidia_blackwell_k:
            return make_punned_dense_index<vram_index<dtype_, kernel_, gpu_class_t::nvidia_blackwell_k>>(geometry, vram,
                                                                                                         outcome);
        case gpu_class_t::nvidia_hopper_k:
            return make_punned_dense_index<vram_index<dtype_, kernel_, gpu_class_t::nvidia_hopper_k>>(geometry, vram,
                                                                                                      outcome);
        case gpu_class_t::nvidia_ada_k:
            return make_punned_dense_index<vram_index<dtype_, kernel_, gpu_class_t::nvidia_ada_k>>(geometry, vram,
                                                                                                   outcome);
        case gpu_class_t::nvidia_ampere_k:
            return make_punned_dense_index<vram_index<dtype_, kernel_, gpu_class_t::nvidia_ampere_k>>(geometry, vram,
                                                                                                      outcome);
        default:
            return propagate_error(usearch_status_device_k, outcome,
                                   "The devices could not be queried, or do not share one GPU generation"),
                   nullptr;
        }
}
#else

/** The portable build's stand-in: every VRAM request lands on one static message. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_index(usearch_dense_metadata_t const&, usearch_vram_options_t const&,
                                        usearch_outcome_t* outcome) noexcept {
    return propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA backends"), nullptr;
}
#endif

/** Routes a numeric dtype's metric onto its three wired kernels, built where @p placement names. */
template <nk_dtype_t dtype_, typename placement_type_>
    requires(dtype_ != nk_u1_k)
static usearch_dense_index_t with_metric(usearch_dense_metadata_t const& geometry, placement_type_ const& placement,
                                         usearch_outcome_t* outcome) noexcept {
    switch (geometry.metric) {
    case usearch_dense_metric_inner_k: return make_index<dtype_, nk_kernel_dot_k>(geometry, placement, outcome);
    case usearch_dense_metric_angular_k: return make_index<dtype_, nk_kernel_angular_k>(geometry, placement, outcome);
    case usearch_dense_metric_euclidean_k:
        return make_index<dtype_, nk_kernel_euclidean_k>(geometry, placement, outcome);
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
    case usearch_dense_metric_inner_k: return make_index<dtype_, nk_kernel_dot_k>(geometry, placement, outcome);
    case usearch_dense_metric_hamming_k: return make_index<dtype_, nk_kernel_hamming_k>(geometry, placement, outcome);
    case usearch_dense_metric_jaccard_k: return make_index<dtype_, nk_kernel_jaccard_k>(geometry, placement, outcome);
    default:
        return propagate_error(usearch_status_unknown_kind_k, outcome,
                               "No kernel kit is wired for this (quantization,metric) pair"),
               nullptr;
    }
}

/** Lowers the runtime @c (quantization,metric) pair onto the wired cells - every numeric dtype
 *  speaks inner, angular & euclidean; the bit dtype speaks inner, hamming & jaccard. */
template <typename placement_type_>
static usearch_dense_index_t with_cell(usearch_dense_metadata_t const& geometry, placement_type_ const& placement,
                                       usearch_outcome_t* outcome) noexcept {
    switch (geometry.quantization) {
    case usearch_scalar_bf16_k: return with_metric<nk_bf16_k>(geometry, placement, outcome);
    case usearch_scalar_f16_k: return with_metric<nk_f16_k>(geometry, placement, outcome);
    case usearch_scalar_e5m2_k: return with_metric<nk_e5m2_k>(geometry, placement, outcome);
    case usearch_scalar_e4m3_k: return with_metric<nk_e4m3_k>(geometry, placement, outcome);
    case usearch_scalar_e3m2_k: return with_metric<nk_e3m2_k>(geometry, placement, outcome);
    case usearch_scalar_e2m3_k: return with_metric<nk_e2m3_k>(geometry, placement, outcome);
    case usearch_scalar_i8_k: return with_metric<nk_i8_k>(geometry, placement, outcome);
    case usearch_scalar_i4x2_k: return with_metric<nk_i4_k>(geometry, placement, outcome);
    case usearch_scalar_u8_k: return with_metric<nk_u8_k>(geometry, placement, outcome);
    case usearch_scalar_u4x2_k: return with_metric<nk_u4_k>(geometry, placement, outcome);
    case usearch_scalar_u1x8_k: return with_metric<nk_u1_k>(geometry, placement, outcome);
    default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unsupported quantization"), nullptr;
    }
}

#pragma endregion Construction

#pragma region Multi Verbs

/** Every argument structure that names an executor, so a deferred submission keeps its call by
 *  value; whatever the copy points to stays the caller's until final. */
using deferred_arguments_t =
    std::variant<usearch_dense_load_ram_t, usearch_dense_load_vram_t, usearch_dense_load_ram_buffer_t,
                 usearch_dense_load_vram_buffer_t, usearch_dense_view_nvme_t, usearch_dense_reserve_t,
                 usearch_dense_clear_t, usearch_dense_insert_t, usearch_dense_refine_t, usearch_dense_health_t,
                 usearch_dense_search_t, usearch_dense_count_t, usearch_dense_get_t, usearch_dense_merge_t,
                 usearch_dense_cursor_next_t, usearch_dense_search_stream_next_t, usearch_dense_save_t,
                 usearch_dense_save_buffer_t, usearch_dense_metadata_read_t, usearch_dense_metadata_read_buffer_t,
                 usearch_dense_describe_t, usearch_dense_multiply_t, usearch_dense_attend_t, usearch_dense_maxsim_t>;

/** Where one slot of a deferred table stands. */
enum class deferred_stage_t : unsigned char {

    /** Nothing in it; a submission may take it. */
    free_k,

    /** A submitted call waits in it for its wait, a drain, a blocking call or a free. */
    held_k,

    /** Some thread took the call and runs it. */
    running_k,
};

/** One slot of an executor's deferred table: the copied call, the verb that runs it, and its place
 *  in line. */
struct deferred_call_t {
    deferred_arguments_t arguments;
    void (*verb)(void*) = nullptr;

    /** Submission order, counted from one across every executor. */
    std::size_t ticket = 0;
    deferred_stage_t stage = deferred_stage_t::free_k;
};

/** Slots a deferred table holds when its constructor names no limit. */
inline constexpr std::size_t deferred_limit_default_k = 64;

/** Reads every NVMe engine keeps in flight: deep enough to cover a device's latency, which the C
 *  interface leaves to the library. */
inline constexpr std::size_t nvme_queue_depth_k = 64;

/** The state behind @c usearch_executor_t: the discovered topology a RAM or NVMe executor was
 *  spawned over, its deferred table, and one residency's executor; the variant is the kind tag, so
 *  no second word can disagree. The executor is declared last, so its pool dies before the
 *  topology. */
struct usearch_executor_handle_t {
    std::unique_ptr<void, decltype(&fu_topology_delete)> topology {nullptr, &fu_topology_delete};

    /** One slot per call it tracks at once, sized at creation, so a submission never allocates. */
    growing_buffer<deferred_call_t> deferred {global_allocator};

    /** The next listed executor, so a wait or a free finds the calls naming its handle. */
    usearch_executor_handle_t* next_listed = nullptr;

    std::variant<forkunion_executor_t, nvme_executor_t
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

/** Discovers the topology and spawns @p threads into @p handle, each opening the engine @p config
 *  names, answering @p handle, or null once the outcome records why not. */
static usearch_executor_t spawn_nvme_executor(usearch_executor_handle_t& handle, std::size_t threads,
                                              io_engine_config_t const& config, usearch_outcome_t* outcome) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_status_device_k, outcome, "Topology discovery failed"), nullptr;
    handle.topology.reset(topology);
    auto [spawned, status] = nvme_executor_t::spawn(topology, threads, config);
    if (failed(status))
        return propagate_error(status, outcome, "No I/O engine opened on the executor's threads"), nullptr;
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

/** The executor a null handle runs NVMe work on: one thread of synchronous reads. */
template <typename multi_type_>
    requires is_nvme_index<multi_type_>
static usearch_executor_t call_executor_for(multi_type_&, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    io_engine_config_t config;
    config.queue_depth = nvme_queue_depth_k;
    config.transport = io_transport_t::blocking_k;
    return spawn_nvme_executor(call_executor, 1, config, outcome);
}

#if USEARCH_BUILD_CUDA

/** The executor a null handle runs VRAM work on: the default streams of the index's devices. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
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

/** The device an arena of host pages names, which no GPU ordinal does. */
inline constexpr int host_device_k = -1;

/** What @c usearch_arena_t owns: memory that one call at a time rewinds and refills, as host pages
 *  or on one device, and the call holding it. */
struct usearch_arena_handle_t {
    growing_arena<forkunion_allocator_t> memory {forkunion_allocator_t {no_memory_domain_k}};
#if USEARCH_BUILD_CUDA
    growing_arena<cuda_device_allocator_t> device_memory {cuda_device_allocator_t {}};
#endif

    /** The GPU ordinal the memory lives on, or @c host_device_k. */
    int device = host_device_k;

    /** The outcome of the call holding the arena, from its claim or deferred submission until just
     *  before its final store; atomic, since a held call finishes on whichever thread runs it. */
    std::atomic<usearch_outcome_t const*> holder {nullptr};
};

/** Holds @p arena for the call whose outcome is @p call, answering whether it was free or already
 *  that call's; a null arena holds nothing, left for the verb to refuse. */
static bool hold_arena(usearch_arena_t arena, usearch_outcome_t const* call) noexcept {
    if (!arena) return true;
    usearch_outcome_t const* holder = nullptr;
    return arena->holder.compare_exchange_strong(holder, call, std::memory_order_acquire) || holder == call;
}

/** Lets go of @p arena when the call whose outcome is @p call holds it. */
static void release_arena(usearch_arena_t arena, usearch_outcome_t const* call) noexcept {
    if (arena) arena->holder.compare_exchange_strong(call, nullptr, std::memory_order_release);
}

/** Whether @p address lies in memory @p arena handed out, which its next rewind reuses or frees. */
static bool arena_contains(usearch_arena_handle_t const& arena, void const* address) noexcept {
#if USEARCH_BUILD_CUDA
    if (arena.device != host_device_k) return arena.device_memory.contains(address);
#endif
    return arena.memory.contains(address);
}

/** Holds @p arena for the call whose outcome is @p call and rewinds it, freeing the previous call's
 *  outputs, so an input of @p inputs still inside it is refused, as is an arena on another device
 *  than @p device. The call's @c call_outcome_t lets go of it before the final store. */
static usearch_arena_handle_t* claim_arena(usearch_arena_t arena, usearch_outcome_t const* call, int device,
                                           std::initializer_list<void const*> inputs,
                                           usearch_outcome_t* outcome) noexcept {
    if (!arena)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The arena must not be null"), nullptr;
    if (arena->device != device)
        return propagate_error(usearch_status_device_k, outcome, "The arena lives where the call does not compute"),
               nullptr;
    if (!hold_arena(arena, call))
        return propagate_error(usearch_status_contended_k, outcome, "The arena serves another call"), nullptr;
    for (void const* input : inputs)
        if (input && arena_contains(*arena, input))
            return propagate_error(usearch_status_invalid_argument_k, outcome,
                                   "An input lies inside the arena it rewinds"),
                   nullptr;
#if USEARCH_BUILD_CUDA
    arena->device_memory.clear();
#endif
    arena->memory.clear();
    return arena;
}

/** Carves @p count values of @p value_type_ out of a claimed @p arena, as host pages or on its
 *  device; null when the arena cannot grow. */
template <typename value_type_>
static value_type_* carve(usearch_arena_handle_t& arena, std::size_t count) noexcept {
#if USEARCH_BUILD_CUDA
    if (arena.device != host_device_k)
        return cudaSetDevice(arena.device) == cudaSuccess ? reinterpret_cast<value_type_*>(arena.device_memory.allocate(
                                                                count * sizeof(value_type_), default_alignment_k))
                                                          : nullptr;
#endif
    return reinterpret_cast<value_type_*>(arena.memory.allocate(count * sizeof(value_type_), default_alignment_k));
}

/**
 *  @brief A verb's outcome while it runs: the caller's reads deferred from the start, every step
 *      records into a private one, and the end lets go of the arena before the one final store.
 *
 *  Declared first in a verb, it unwinds last, so a waiter that sees the final status may reuse the
 *  arena at once. A path that forgot to report finishes @c usearch_status_internal_k rather than
 *  hanging its waiters.
 */
class call_outcome_t {
  public:
    explicit call_outcome_t(usearch_outcome_t* published, usearch_arena_t arena = nullptr) noexcept
        : published_(published), arena_(arena) {
        propagate_pending(published_);
    }
    ~call_outcome_t() noexcept {
        release_arena(arena_, published_);
        propagate_error(recorded_.status, published_, recorded_.message);
    }
    call_outcome_t(call_outcome_t const&) = delete;
    call_outcome_t& operator=(call_outcome_t const&) = delete;

    /** The private record every step reports into. */
    operator usearch_outcome_t*() noexcept { return &recorded_; }

  private:
    usearch_outcome_t* published_;
    usearch_arena_t arena_;
    usearch_outcome_t recorded_ {usearch_status_internal_k, "The call finished without reporting a status"};
};

#pragma region Deferred Calls

/** Every listed executor and the lock every deferred table is read under, so a wait or a free finds
 *  the calls naming its handle without asking each executor. */
struct deferred_registry_t {
    std::mutex guard;

    /** Signalled whenever a taken call becomes final. */
    std::condition_variable finished;

    /** The last ticket handed out. */
    std::size_t tickets = 0;

    /** The newest listed executor. */
    usearch_executor_handle_t* executors = nullptr;
};

static deferred_registry_t deferred_registry;

/** The held call whose arguments this thread passes back into their verb, which then runs them in
 *  place rather than holding them again. */
static thread_local void const* resuming_call = nullptr;

/** Sizes @p executor 's deferred table at @p limit slots, zero taking the default, and lists it. */
static status_t list_executor(usearch_executor_handle_t& executor, std::size_t limit) noexcept {
    if (failed(executor.deferred.resize(limit ? limit : deferred_limit_default_k))) return status_t::bad_alloc_k;
    std::lock_guard<std::mutex> lock(deferred_registry.guard);
    executor.next_listed = deferred_registry.executors;
    deferred_registry.executors = &executor;
    return status_t::success_k;
}

/** Takes @p executor off the list; the caller already finished every call it held. */
static void delist_executor(usearch_executor_handle_t& executor) noexcept {
    std::lock_guard<std::mutex> lock(deferred_registry.guard);
    for (usearch_executor_handle_t** link = &deferred_registry.executors; *link; link = &(*link)->next_listed)
        if (*link == &executor) {
            *link = executor.next_listed;
            return;
        }
}

/** Whether the call in @p slot reads or writes @p handle: its outcome, its executor, or an index,
 *  cursor, stream or arena it names. An output slot for a handle the call creates holds nothing
 *  yet. */
static bool deferred_uses(deferred_call_t const& slot, void const* handle) noexcept {
    return std::visit(
        [&](auto const& arguments) noexcept {
            auto const names = [&](auto named) noexcept {
                if constexpr (std::is_pointer_v<std::remove_pointer_t<decltype(named)>>) return false;
                else return static_cast<void const*>(named) == handle;
            };
            bool used = static_cast<void const*>(arguments.outcome) == handle ||
                        static_cast<void const*>(arguments.executor) == handle;
            if constexpr (requires { arguments.index; }) used = used || names(arguments.index);
            if constexpr (requires { arguments.cursor; }) used = used || names(arguments.cursor);
            if constexpr (requires { arguments.stream; }) used = used || names(arguments.stream);
            if constexpr (requires { arguments.arena; }) used = used || names(arguments.arena);
            return used;
        },
        slot.arguments);
}

/** Runs every call @p executor holds with a ticket up to @p last on this thread, oldest first. */
static void run_held(usearch_executor_handle_t& executor, std::size_t last) noexcept {
    for (;;) {
        deferred_call_t* oldest = nullptr;
        {
            std::lock_guard<std::mutex> lock(deferred_registry.guard);
            for (deferred_call_t& slot : executor.deferred)
                if (slot.stage == deferred_stage_t::held_k && slot.ticket <= last &&
                    (!oldest || slot.ticket < oldest->ticket))
                    oldest = &slot;
            if (!oldest) return;
            oldest->stage = deferred_stage_t::running_k;
        }
        std::visit(
            [&](auto& arguments) noexcept {
                resuming_call = &arguments;
                oldest->verb(&arguments);
            },
            oldest->arguments);
        std::lock_guard<std::mutex> lock(deferred_registry.guard);
        oldest->stage = deferred_stage_t::free_k;
        deferred_registry.finished.notify_all();
    }
}

/** Waits until no call @p executor took with a ticket up to @p last still runs on another
 *  thread. */
static void wait_running(usearch_executor_handle_t& executor, std::size_t last) noexcept {
    std::unique_lock<std::mutex> lock(deferred_registry.guard);
    deferred_registry.finished.wait(lock, [&]() noexcept {
        for (deferred_call_t const& slot : executor.deferred)
            if (slot.stage == deferred_stage_t::running_k && slot.ticket <= last) return false;
        return true;
    });
}

/** Returns once no deferred call reads or writes @p handle, running the held ones on this thread
 *  together with every call held before them on the same executor - what every wait and every free
 *  does first. */
static void finish_calls_using(void const* handle) noexcept {
    if (!handle) return;
    std::unique_lock<std::mutex> lock(deferred_registry.guard);
    for (;;) {
        usearch_executor_handle_t* holding = nullptr;
        std::size_t last = 0, running = 0;
        for (usearch_executor_handle_t* executor = deferred_registry.executors; executor && !holding;
             executor = executor->next_listed)
            for (deferred_call_t const& slot : executor->deferred)
                if (slot.stage == deferred_stage_t::held_k && deferred_uses(slot, handle))
                    holding = executor, last = max_of(last, slot.ticket);
                else if (slot.stage == deferred_stage_t::running_k && deferred_uses(slot, handle)) ++running;
        if (holding) {
            // Running a held call takes this lock itself, so it runs outside it.
            lock.unlock();
            run_held(*holding, last);
            lock.lock();
        }
        else if (running) deferred_registry.finished.wait(lock);
        else return;
    }
}

/**
 *  @brief Every verb's entry. A deferred call on an executor is copied into the executor's table,
 *      holding its arena, and answers true; a blocking one first runs the calls the executor holds
 *      and answers false.
 *
 *  A call the runner of a held call passes back answers false at once, so the verb runs it in
 *  place, and so does a call with no executor, which is final when it returns whatever its
 *  completion says.
 */
template <auto verb_, typename arguments_type_>
static bool submitted(arguments_type_* arguments) noexcept {
    if (resuming_call == arguments) return resuming_call = nullptr, false;
    usearch_executor_t const executor = arguments->executor;
    if (!executor) return false;
    if (arguments->completion != usearch_completion_deferred_k) return run_held(*executor, ~std::size_t {0}), false;
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    usearch_arena_t arena = nullptr;
    if constexpr (requires { arguments->arena; }) arena = arguments->arena;
    if (!hold_arena(arena, outcome))
        return propagate_error(usearch_status_contended_k, outcome, "The arena serves another call"), true;
    std::lock_guard<std::mutex> lock(deferred_registry.guard);
    for (deferred_call_t& slot : executor->deferred)
        if (slot.stage == deferred_stage_t::free_k) {
            slot.arguments.template emplace<arguments_type_>(*arguments);
            slot.verb = [](void* erased) { verb_(static_cast<arguments_type_*>(erased)); };
            slot.ticket = ++deferred_registry.tickets;
            slot.stage = deferred_stage_t::held_k;
            return true;
        }
    // A refused submission never runs, so nothing else would let go of the arena it took.
    release_arena(arena, outcome);
    propagate_error(usearch_status_contended_k, outcome, "The deferred table is full");
    return true;
}

#pragma endregion Deferred Calls

/** Bytes one packed row of @p dimensions spans in @p dtype. */
static std::size_t packed_row_bytes(nk_dtype_t dtype, std::size_t dimensions) noexcept {
    return divide_round_up<std::size_t>(nk_dtype_bits(dtype) * dimensions, 8);
}

/** The export tail every row reader shares: @p fill produces up to @p rows rows in the stored type
 *  and reports how many, landing packed in @p landing when the two types agree and in a staged
 *  plane converted row by row into @p target_dtype otherwise. */
template <typename scalar_type_, typename fill_type_>
static expected<std::size_t> export_rows(std::byte* landing, std::size_t rows, nk_dtype_t target_dtype,
                                         std::size_t dimensions, nk_dtype_t stored_dtype, fill_type_&& fill) noexcept {
    std::size_t const row_bytes = packed_row_bytes(target_dtype, dimensions);
    if (target_dtype == stored_dtype)
        return fill(nk::matrix_span<scalar_type_> {reinterpret_cast<char*>(landing),
                                                   rows_shape_of<scalar_type_>(rows, row_bytes, dimensions)});
    auto zeros = nk::matrix<scalar_type_>::zeros({rows, dimensions});
    if (!zeros)
        return {0, zeros.status == nk::status_t::bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k};
    nk::matrix<scalar_type_> staged = std::move(zeros.value);
    expected<std::size_t> const exported = fill(staged.span());
    for (std::size_t row = 0; exported && row != min_of(exported.value, rows); ++row) {
        [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
            staged.span().row(row).as_vector().byte_data(), stored_dtype, landing + row * row_bytes, target_dtype,
            dimensions, nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    }
    return exported;
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

/** Where a ranked run lands inside a claimed arena: @c wanted keys and scores per row, and a count
 *  per row. */
struct matches_landing_t {
    usearch_key_t* keys = nullptr;
    usearch_distance_t* scores = nullptr;
    std::size_t* counts = nullptr;
    std::size_t wanted = 0;

    /** The view the call publishes once every row landed. */
    usearch_matches_view_t view() const noexcept { return {keys, scores, counts, wanted}; }
};

/** Carves @p count rows of @p wanted matches out of a claimed @p arena; a zero @p wanted carves
 *  only the counts. False once @p outcome records why not. */
static bool carve_matches(usearch_arena_handle_t& arena, std::size_t count, std::size_t wanted,
                          matches_landing_t& landing, usearch_outcome_t* outcome) noexcept {
    landing.wanted = wanted;
    landing.counts = carve<std::size_t>(arena, count);
    if (wanted) {
        landing.keys = carve<usearch_key_t>(arena, count * wanted);
        landing.scores = carve<usearch_distance_t>(arena, count * wanted);
    }
    if (landing.counts && (!wanted || (landing.keys && landing.scores))) return true;
    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
}

/** The matching verbs' sink on one serving arm: keys land in @p landing as the walk writes them,
 *  and @p score_type_ scores, tagged @p score_dtype, stage at the engine's width and widen to the
 *  ABI's double; @p rows_per_query input rows share one output row, and @p count output rows
 *  land. */
template <typename score_type_, typename variant_type_, typename verb_type_>
static bool match_dense_rows(variant_type_& variant, usearch_executor_t executor,                //
                             std::size_t count, usearch_rows_view_t const& rows,                 //
                             std::size_t rows_per_query, matches_landing_t const& landing,       //
                             nk_dtype_t score_dtype, std::size_t expansion_factor,               //
                             usearch_progress_callback_t progress, void* progress_state,         //
                             usearch_dense_traversal_stats_t* stats, usearch_outcome_t* outcome, //
                             verb_type_&& verb) noexcept {
    using key_t = typename variant_type_::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk writes them");
    std::size_t const wanted = landing.wanted;
    auto make_sink = [&](std::size_t chunk, status_t& staged) noexcept {
        growing_buffer<score_type_> staged_scores {global_allocator};
        if (failed(staged_scores.resize(chunk / rows_per_query * wanted))) staged = status_t::bad_alloc_k;
        return [&, staged_scores = std::move(staged_scores)](
                   specialized_index<variant_type_> view, typename variant_type_::vectors_view_t chunk_rows,
                   typename variant_type_::executor_t& executor, typename variant_type_::search_config_t config,
                   std::size_t resolved, std::size_t taking) mutable noexcept {
            std::size_t const first = resolved / rows_per_query, matched = taking / rows_per_query;
            expected<typename variant_type_::search_result_t> const found = verb(
                view, chunk_rows, executor, config,
                std::span<key_t> {reinterpret_cast<key_t*>(landing.keys) + first * wanted, matched * wanted},
                std::span<score_type_> {staged_scores.data(), matched * wanted},
                std::span<std::size_t> {landing.counts + first, matched});
            if (!found || !wanted) return found;
            [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
                staged_scores.data(), score_dtype, landing.scores + first * wanted, nk_f64_k, matched * wanted,
                nullptr);
            assert(cast == nk_success_k && "The serial tier casts every dtype pair");
            return found;
        };
    };
    return for_each_dense_chunk(variant, executor, count * rows_per_query, rows, rows_per_query, expansion_factor,
                                progress, progress_state, stats, outcome, make_sink);
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
            // A load places RAM or VRAM arms alone, so a view's arm never reaches here.
            if constexpr (is_nvme_index<variant_type_>)
                return propagate_error(usearch_status_internal_k, outcome, "A load never builds a view");
            else {
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const typed = executor_of(variant, executor, call_executor,
                                                                              outcome);
                if (!typed) return false;
                shard_ticks_t ticks {progress, progress_state, specialized_index {variant}.shards_count()};
                status_t const status = specialized_index {variant}.load(source, *typed, ticks);
                return succeeded(status) || propagate_error(status, outcome);
            }
        },
        loaded->variants);
    if (!filled) return delete loaded, void();
    *index = loaded;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Visits the arm of @p index with @p mutate when it is writable; an NVMe view serves its
 *  checkpoint in place, so it refuses every mutation with @c usearch_status_read_only_k. */
template <typename mutate_type_>
static void visit_writable(usearch_dense_index_handle_t& index, usearch_outcome_t* outcome,
                           mutate_type_&& mutate) noexcept {
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_nvme_index<variant_type_>)
                propagate_error(usearch_status_read_only_k, outcome, "An NVMe view never mutates");
            else mutate(variant);
        },
        index.variants);
}

/** Lands one finished output row in an arena on @p device, or on the host when it names none; a
 *  device row rides one blocking upload, since the walk that produced it staged on the host. */
static bool land_row(std::byte* landing, void const* row, std::size_t bytes, int device) noexcept {
#if USEARCH_BUILD_CUDA
    if (device != host_device_k) return cudaMemcpy(landing, row, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
#endif
    std::memcpy(landing, row, bytes);
    return true;
}

/** Lists a spawned @p executor with a deferred table of @p deferred_limit slots and hands it out
 *  through @p output, or frees it once @p outcome records why not. */
static void hand_out_executor(usearch_executor_t executor, std::size_t deferred_limit, usearch_executor_t* output,
                              usearch_outcome_t* outcome) noexcept {
    if (failed(list_executor(*executor, deferred_limit)))
        return delete executor,
               propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the deferred table"),
               void();
    *output = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

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
    auto const [checkpoint, read_status] = dense_nvme_read_metadata(as_text(arguments->path));
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
                {usearch_metric_of(variant_type_::kernel_kind_k), usearch_scalar_of(variant_type_::vector_dtype_k),
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
    us::uuid_t const key = uuid_of(arguments->key);
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
    usearch_progress_callback_t const progress = arguments->progress;
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
        nk_dtype_t const source_dtype = nk_dtype_of(vectors.scalar);
        if (source_dtype == nk_dtype_unknown_k)
            return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
        usearch_executor_handle_t call_executor;
        typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor, call_executor,
                                                                         outcome);
        if (!executor) return;
        using key_t = typename variant_type_::key_t;
        typename variant_type_::insert_config_t insert_config {};
        insert_config.expansion_factor = static_cast<u32_t>(arguments->expansion_factor);

        // An insertion takes one packed run of keys, so any other stride gathers into one first.
        std::span<key_t const> uuids {reinterpret_cast<key_t const*>(keys.data), count};
        growing_buffer<key_t> gathered_keys {global_allocator};
        if (keys.stride != sizeof(usearch_key_t)) {
            if (failed(gathered_keys.resize(count)))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to gather the keys"), void();
            for (std::size_t row = 0; row != count; ++row)
                std::memcpy(&gathered_keys[row], reinterpret_cast<char const*>(keys.data) + row * keys.stride,
                            sizeof(key_t));
            uuids = {gathered_keys.data(), count};
        }

        std::size_t const dimensions = specialized_index {variant}.dimensions();
        if (vectors.stride && vectors.stride < packed_row_bytes(source_dtype, dimensions))
            return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row"),
                   void();
        insertion_rows_t const rows {static_cast<char const*>(vectors.data), vectors.stride, source_dtype, progress,
                                     arguments->progress_state};
        expected<typename variant_type_::insert_result_t> const inserted = specialized_index {variant}.add(
            uuids, rows, *executor, insert_config);
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
        shard_ticks_t ticks {arguments->progress, arguments->progress_state, specialized_index {variant}.size()};
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
    key_predicate_t keyed {arguments->predicate, arguments->predicate_state};
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_vram_index<variant_type_>)
                propagate_error(usearch_status_device_k, outcome,
                                keyed.callback ? "A host predicate cannot narrow a device walk"
                                               : "A VRAM search waits on device merges into a device arena");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                                  {arguments->queries.data}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                matches_landing_t landing;
                if (!carve_matches(*arena, count, wanted, landing, outcome)) return;
                auto search_with = [&](auto&& predicate) noexcept {
                    return match_dense_rows<typename variant_type_::distance_t>(
                        variant, arguments->executor, count, arguments->queries, 1, landing, nk_f32_k,
                        arguments->expansion_factor, arguments->progress, arguments->progress_state, arguments->stats,
                        outcome,
                        [&](auto view, auto rows, auto& executor, auto config, auto keys, auto distances, auto counts) {
                            return view.search(rows, executor, config, wanted, keys, distances, counts, predicate);
                        });
                };
                if (!(keyed.callback ? search_with(keyed) : search_with(dummy_predicate_t {}))) return;
                *arguments->matches = landing.view();
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_dense_get(usearch_dense_get_t* arguments) {
    if (submitted<usearch_dense_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    us::uuid_t const key = uuid_of(arguments->key);
    usearch_scalar_t const scalar = arguments->outputs_scalar;
    *arguments->vectors = {};
    if (arguments->written) *arguments->written = 0;
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_vram_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome,
                                "VRAM rows wait on device arenas, so a search reads them back");
            else {
                using scalar_t = typename variant_type_::scalar_t;
                nk_dtype_t const target_dtype = nk_dtype_of(scalar);
                if (target_dtype == nk_dtype_unknown_k)
                    return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                                  {}, outcome);
                if (!arena) return;
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                                 call_executor, outcome);
                if (!executor) return;
                // The single key rides the batched lookups, which size the landing first and fill
                // it second.
                specialized_index view {variant};
                std::size_t entries = 0;
                if (status_t const counted = view.count({&key, 1}, {&entries, 1}, *executor); failed(counted))
                    return propagate_error(counted, outcome), void();
                if (!entries) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                std::size_t const dimensions = view.dimensions();
                std::size_t const row_bytes = packed_row_bytes(target_dtype, dimensions);
                std::byte* const rows = carve<std::byte>(*arena, entries * row_bytes);
                if (!rows)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                expected<std::size_t> const exported = export_rows<scalar_t>(
                    rows, entries, target_dtype, dimensions, variant_type_::vector_dtype_k,
                    [&](nk::matrix_span<scalar_t> span) noexcept {
                        std::size_t total = 0;
                        status_t const got = view.get({&key, 1}, span, entries, {&total, 1}, *executor);
                        return expected<std::size_t> {min_of(total, entries), got};
                    });
                if (!exported) return propagate_error(exported.status, outcome, "Failed to export the rows"), void();
                *arguments->vectors = {rows, row_bytes, dimensions, scalar};
                if (arguments->written) *arguments->written = exported.value;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
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

#pragma region Cursors and Merging

USEARCH_EXPORT void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
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
    nk_dtype_t const target_dtype = vectors ? nk_dtype_of(arguments->outputs_scalar) : nk_dtype_unknown_k;
    if (vectors && target_dtype == nk_dtype_unknown_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k, {}, outcome);
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
                using scalar_t = typename cursor_type_::scalar_t;
                using key_t = typename cursor_type_::key_t;
                using indexes_t = typename cursor_type_::indexes_t;
                static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk reads them");
                if (!wanted) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                // A handle never changes its arm, so it is still the one the cursor was opened on.
                auto& indexes = std::get<indexes_t>(arguments->index->variants);
                // An NVMe cursor reads its rows through an executor, a RAM one copies them.
                usearch_executor_handle_t call_executor;
                [[maybe_unused]] typename indexes_t::executor_t* executor = nullptr;
                if constexpr (is_nvme_index<indexes_t>) {
                    executor = executor_of(indexes, arguments->executor, call_executor, outcome);
                    if (!executor) return;
                }
                std::size_t const dimensions = indexes.dimensions();
                std::size_t const row_bytes = vectors ? packed_row_bytes(target_dtype, dimensions) : 0;
                usearch_key_t* const keys = carve<usearch_key_t>(*arena, wanted);
                std::byte* const rows = vectors && keys ? carve<std::byte>(*arena, wanted * row_bytes) : nullptr;
                if (!keys || (vectors && !rows))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                std::span<key_t> const identities {reinterpret_cast<key_t*>(keys), wanted};
                auto read_into = [&](nk::matrix_span<scalar_t> span) noexcept -> expected<std::size_t> {
                    if constexpr (is_nvme_index<indexes_t>) {
                        auto const [scanned, status] = walk.read(indexes, identities, span, *executor);
                        return {scanned.count, status};
                    }
                    else return {walk.read(indexes, identities, span), status_t::success_k};
                };
                expected<std::size_t> const read = vectors
                                                       ? export_rows<scalar_t>(rows, wanted, target_dtype, dimensions,
                                                                               indexes_t::vector_dtype_k, read_into)
                                                       : read_into({});
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

USEARCH_EXPORT void usearch_dense_merge(usearch_dense_merge_t* arguments) {
    if (submitted<usearch_dense_merge>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, outcome, "No residency absorbs donor handles yet");
}

#pragma endregion Cursors and Merging

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
            if constexpr (variant_type_::kernel_kind_k != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A product is a dot, so the kind must be inner product");
            else if constexpr (is_vram_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome, "VRAM products wait on device arenas");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                                  {arguments->left.data}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
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
                    auto make_sink = [&](std::size_t, status_t&) noexcept {
                        return [&](specialized_index<variant_type_> view,
                                   typename variant_type_::vectors_view_t chunk_rows,
                                   typename variant_type_::executor_t& executor,
                                   typename variant_type_::search_config_t config, std::size_t resolved,
                                   std::size_t taking) noexcept {
                            return view.multiply(chunk_rows, executor, config, width,
                                                 {staged_keys.data() + resolved * width, taking * width},
                                                 {staged_products.data() + resolved * width, taking * width},
                                                 {scored.data() + resolved, taking});
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

                std::size_t* const offsets = carve<std::size_t>(*arena, count + 1);
                if (!offsets)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                std::size_t required = 0;
                offsets[0] = 0;
                for (std::size_t row = 0; row != count; ++row)
                    offsets[row + 1] = offsets[row] + min_of(scored[row], width), required += scored[row];
                std::size_t const written = offsets[count];
                usearch_key_t* const keys = written ? carve<usearch_key_t>(*arena, written) : nullptr;
                usearch_distance_t* const scores = written ? carve<usearch_distance_t>(*arena, written) : nullptr;
                if (written && (!keys || !scores))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                for (std::size_t row = 0; row != count; ++row) {
                    std::size_t const kept = offsets[row + 1] - offsets[row];
                    if (!kept) continue;
                    std::memcpy(keys + offsets[row], staged_keys.data() + row * width, kept * sizeof(key_t));
                    [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
                        staged_products.data() + row * width, dot_result_t::dtype(), scores + offsets[row], nk_f64_k,
                        kept, nullptr);
                    assert(cast == nk_success_k && "The serial tier casts every dtype pair");
                }
                *arguments->products = {keys, scores, offsets};
                if (arguments->written) *arguments->written = written;
                if (arguments->required) *arguments->required = required;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            }
        },
        arguments->index->variants);
}

static_assert(sizeof(us::dense_softmax_state_t) == sizeof(usearch_dense_softmax_state_t));

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
    nk_dtype_t const outputs_dtype = nk_dtype_of(outputs_scalar);
    if (outputs_dtype == nk_dtype_unknown_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind"), void();
    std::size_t const columns = arguments->queries.columns;
    if (count % head_count)
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "The heads must divide the rows"), void();
    if (rejects_columns(*arguments->index, count, columns, outcome)) return;
    std::size_t const row_bytes = packed_row_bytes(outputs_dtype, columns);
    f32_t const requested_scale = arguments->score_scale;

    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (variant_type_::kernel_kind_k != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A softmax weighs logits, so the kind must be inner product");
            else if (!arguments->value_lookup)
                propagate_error(usearch_status_invalid_argument_k, outcome, "The value lookup must be set");
            else {
                int device = host_device_k;
                if constexpr (is_vram_index<variant_type_>) device = specialized_index {variant}.device(0);
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, device,
                                                                  {arguments->queries.data, carry}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                std::byte* const landing = carve<std::byte>(*arena, count * row_bytes);
                if (!landing)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                typename specialized_index<variant_type_>::attend_lookup_t const value_lookup {
                    arguments->value_lookup, arguments->value_lookup_state,
                    usearch_scalar_of(variant_type_::vector_dtype_k), columns};
                // Outputs stage as `f32` and narrow one row at a time; the carry stages beside them.
                auto make_sink = [&](std::size_t chunk, status_t& staged) noexcept {
                    growing_buffer<f32_t> staged_outputs {global_allocator};
                    growing_buffer<us::dense_softmax_state_t> staged_carry {global_allocator};
                    growing_buffer<std::byte> narrowed {global_allocator};
                    if (failed(staged_outputs.resize(chunk * columns)) || failed(staged_carry.resize(chunk)) ||
                        failed(narrowed.resize(row_bytes)))
                        staged = status_t::bad_alloc_k;
                    return [&, staged_outputs = std::move(staged_outputs), staged_carry = std::move(staged_carry),
                            narrowed = std::move(narrowed)](specialized_index<variant_type_> view,
                                                            typename variant_type_::vectors_view_t chunk_rows,
                                                            typename variant_type_::executor_t& executor,
                                                            typename variant_type_::search_config_t config,
                                                            std::size_t resolved, std::size_t taking) mutable noexcept {
                        f32_t const score_scale = requested_scale > 0
                                                      ? requested_scale
                                                      : 1 / std::sqrt(static_cast<f32_t>(view.dimensions()));
                        expected<typename variant_type_::search_result_t> found = view.attend(
                            chunk_rows, head_count, value_lookup, score_scale,
                            {staged_outputs.data(), taking * columns}, {staged_carry.data(), taking}, executor, config);
                        if (!found) return found;
                        for (std::size_t row = 0; row != taking; ++row) {
                            std::byte* const target = landing + (resolved + row) * row_bytes;
                            [[maybe_unused]] nk_status_t const cast =
                                dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
                                    staged_outputs.data() + row * columns, nk_f32_k,
                                    device == host_device_k ? target : narrowed.data(), outputs_dtype, columns,
                                    nullptr);
                            assert(cast == nk_success_k && "The serial tier casts every dtype pair");
                            if (device != host_device_k && !land_row(target, narrowed.data(), row_bytes, device))
                                return found.status = status_t::device_error_k, found;
                            if (carry)
                                carry[resolved + row] = std::bit_cast<usearch_dense_softmax_state_t>(staged_carry[row]);
                        }
                        return found;
                    };
                };
                if (!for_each_dense_chunk(variant, arguments->executor, count, arguments->queries, head_count,
                                          arguments->expansion_factor, arguments->progress, arguments->progress_state,
                                          arguments->stats, outcome, make_sink))
                    return;
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
            if constexpr (variant_type_::kernel_kind_k != nk_kernel_dot_k)
                propagate_error(usearch_status_config_mismatch_k, outcome,
                                "A maximum over logits needs the inner-product kind");
            else if constexpr (is_vram_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome, "VRAM late interaction waits on device arenas");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                                  {arguments->queries.data}, outcome);
                if (!arena) return;
                if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
                matches_landing_t landing;
                if (!carve_matches(*arena, count, wanted, landing, outcome)) return;
                if (!match_dense_rows<f64_t>(
                        variant, arguments->executor, count, arguments->queries, rows_per_query, landing, nk_f64_k,
                        arguments->expansion_factor, arguments->progress, arguments->progress_state, arguments->stats,
                        outcome,
                        [&](auto view, auto rows, auto& executor, auto config, auto keys, auto scores, auto counts) {
                            return view.maxsim(rows, rows_per_query, executor, config, wanted, keys, scores, counts);
                        }))
                    return;
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
                     arguments->progress_state, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments) {
    if (submitted<usearch_dense_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    load_dense_index(as_text(arguments->path), arguments->vram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments) {
    if (submitted<usearch_dense_load_ram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    load_dense_index(as_bytes(arguments->buffer), arguments->ram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments) {
    if (submitted<usearch_dense_load_vram_buffer>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    load_dense_index(as_bytes(arguments->buffer), arguments->vram, arguments->executor, arguments->progress,
                     arguments->progress_state, arguments->index, outcome);
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
                shard_ticks_t ticks {arguments->progress, arguments->progress_state,
                                     specialized_index {variant}.shards_count()};
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
            else if constexpr (is_vram_index<variant_type_>)
                propagate_error(usearch_status_unsupported_k, outcome, "VRAM monoliths wait on device arenas");
            else {
                usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, host_device_k,
                                                                  {}, outcome);
                if (!arena) return;
                usearch_executor_handle_t call_executor;
                typename variant_type_::executor_t* const executor = executor_of(variant, arguments->executor,
                                                                                 call_executor, outcome);
                if (!executor) return;
                auto const [length, status] = specialized_index {variant}.storage_requirements();
                if (failed(status)) return propagate_error(status, outcome), void();
                std::byte* const monolith = carve<std::byte>(*arena, length);
                if (!monolith)
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                shard_ticks_t ticks {arguments->progress, arguments->progress_state,
                                     specialized_index {variant}.shards_count()};
                status_t const saved = specialized_index {variant}.save(
                    bytes_span_t {reinterpret_cast<byte_t*>(monolith), length}, *executor, ticks);
                if (failed(saved)) return propagate_error(saved, outcome), void();
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

USEARCH_EXPORT usearch_status_t usearch_outcome_wait(usearch_outcome_t const* outcome) {
    finish_calls_using(outcome);
    // The status word is written by the library and only read here, so the cast never races a store.
    std::atomic_ref<usearch_status_t> status {const_cast<usearch_status_t&>(outcome->status)};
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
#if USEARCH_BUILD_CUDA
    // A VRAM executor lands its outputs on its first device; every other executor computes on the
    // host.
    if (arguments->executor)
        if (cuda_executor_t const* const devices = std::get_if<cuda_executor_t>(&arguments->executor->executor)) {
            arena->device = devices->device(0);
            arena->device_memory = growing_arena<cuda_device_allocator_t> {
                cuda_device_allocator_t {nullptr, arena->device}};
        }
#endif
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
    hand_out_executor(executor, arguments->deferred_limit, arguments->executor, outcome);
#else
    propagate_error(usearch_status_device_k, outcome, "This build carries no CUDA executor");
#endif
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
    if (status_t const claimed = std::visit(claimable, executor.executor); failed(claimed))
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
    propagate_error(std::visit(drain, executor.executor), outcome);
}

USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor) {
    if (!executor) return;
    finish_calls_using(executor);
    delist_executor(*executor);
    delete executor;
}

#pragma endregion Executors

} // extern "C"
