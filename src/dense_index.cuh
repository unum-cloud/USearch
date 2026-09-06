/**
 *  @brief The dense modality's C surface: the variant grid over every compiled backend, the
 *      handle owning the chosen one, and every exported entry point as one visit apiece.
 *  @author Ash Vardanian
 *  @file src/dense_index.cuh
 *  @date August 14, 2026
 *
 *  Compiled twice: as @c dense_index.cpp for the portable library and as @c dense_index.cu for
 *  the CUDA one, with @c USEARCH_BUILD_CUDA deciding whether the VRAM arms join the variant.
 *  Arms are ordered newest-capability-first inside each @c (dtype,kernel) cell, since a machine
 *  reporting Ice Lake also reports Haswell and Skylake; the serial floor is last and
 *  unconditional. The arms' bodies compile in the dtype shard TUs, never here.
 */
#pragma once

#include <cstdio>  // `std::snprintf` - composing the capability string
#include <cstring> // `std::memcpy`, `std::strlen` - scattering staged rows & the capability string

#include <atomic>      // `std::atomic` - serializing the progress ticks
#include <mutex>       // `std::lock_guard` over the ticks' lock
#include <new>         // `std::nothrow`
#include <span>        // `std::span` - key batches crossing the ABI
#include <string_view> // `std::string_view` - checkpoint paths
#include <type_traits> // `std::is_same_v` - the variant-membership audit
#include <variant>     // `std::variant`, `std::visit`

#include <forkunion.h>            // `fu_topology_new` - the executor factories' discovery
#include <forkunion/types.hpp>    // `fu::spin_mutex` - the ticks' lock
#include <numkong/capabilities.h> // `nk_capabilities_available`
#include <numkong/cast.h>         // `nk_cast` - converting foreign scalar rows at the boundary

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
 *  siblings join under the CUDA build; the NVMe siblings join with the block-storage
 *  executor, and until then their residency refuses at construction.
 */
struct punned_dense_index_t {

    using variants_t = std::variant<
#if NK_TARGET_X8664_
        // bf16: intel_haswell → intel_skylake → amd_genoa
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,           //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>,     //
        // f16: intel_haswell → intel_skylake
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        // e5m2: intel_haswell → intel_skylake → amd_genoa
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,           //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>,     //
        // e4m3: intel_haswell → intel_skylake → amd_genoa
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>,           //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>,     //
        // e3m2: intel_haswell → intel_skylake
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        // e2m3: intel_haswell → intel_skylake
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>, //
        // i8: intel_haswell → intel_icelake
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        // i4: intel_haswell → intel_icelake
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        // u8: intel_haswell → intel_icelake
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        // u4: intel_haswell → intel_icelake
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>, //
        // u1: intel_haswell → intel_icelake, bit kernels
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>, //
        // intel_sierra: every cell - the E-core vector bundle, TPAUSE waits & CMPCCXADD atomics
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
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_sierra_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_sierra_k>,     //
        // intel_diamond: every cell - the AVX10.2 lineage, TPAUSE waits & CMPCCXADD atomics
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_diamond_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_diamond_k>,     //
#elif NK_TARGET_ARM64_
        // bf16 / f16: aws_g2 floor → apple_m2 → aws_g5
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,         //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>,     //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>,   //
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,          //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>,      //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>,    //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,  //
        ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,          //
        ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,      //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,    //
        // fp8 family: apple_m2 → aws_g5 → nvidia_vera
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>, //
        // i8 / i4 / u8 / u4: apple_m2 → aws_g5
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>,   //
        // u1: aws_g2 floor → aws_g5, bit kernels; the wider bundles pick the same NEON kit
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>, //
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g5_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g5_k>, //
        // apple_m2: the remaining cells - the aws_g2 kits under LSE & RCpc atomics
        ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::apple_m2_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::apple_m2_k>, //
#elif NK_TARGET_RISCV64_
        // alibaba_c930: every cell - the serial kits under Zawrs waits & Zacas atomics
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
#if USEARCH_BUILD_CUDA
        vram_index<nk_bf16_k, nk_kernel_angular_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k>, //
        vram_index<nk_f16_k, nk_kernel_angular_k>,    //
        vram_index<nk_i8_k, nk_kernel_angular_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k>,   //
        vram_index<nk_i4_k, nk_kernel_angular_k>,     //
        vram_index<nk_u8_k, nk_kernel_angular_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k>,   //
        vram_index<nk_u1_k, nk_kernel_hamming_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k>,     //
#endif
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

    /** @brief The chosen backend - one alternative of the compiled grid, selected once at
     *      construction and visited once per call. Residency, quantization, and compute tier
     *      all live in the alternative's own type - no second tag exists to disagree. */
    variants_t variants;

    /** @brief The dimensionality the options requested, consumed by the first @c reserve;
     *      once any shard froze real geometry, the variant answers and this stops mattering. */
    std::size_t requested_dimensions = 0;
    /** @brief The neighbors-per-node bound the options requested, consumed like the
     *      dimensionality above. */
    connectivity_t requested_connectivity = connectivity_32_k;
    /** @brief The dispatched tiers, composed once at construction and read-only after - the
     *      C @c capabilities verb returns this buffer, so it must outlive every call. */
    char capabilities_string[128] = {};

    template <typename variant_type_>
    explicit punned_dense_index_t(std::in_place_type_t<variant_type_> tag, forkunion_allocator_t allocator) noexcept
        : variants(tag, allocator) {}
};

/** @brief The arms' explicit instantiation declarations: every body compiles in its dtype's shard TU,
 *      so this TU only visits. */
#if NK_TARGET_X8664_
/* bf16: intel_haswell → intel_skylake → amd_genoa */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
/* f16: intel_haswell → intel_skylake */
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
/* e5m2: intel_haswell → intel_skylake → amd_genoa */
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
/* e4m3: intel_haswell → intel_skylake → amd_genoa */
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
/* e3m2: intel_haswell → intel_skylake */
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
/* e2m3: intel_haswell → intel_skylake */
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
/* i8: intel_haswell → intel_icelake */
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
/* i4: intel_haswell → intel_icelake */
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
/* u8: intel_haswell → intel_icelake */
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
/* u4: intel_haswell → intel_icelake */
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
/* u1: intel_haswell → intel_icelake, bit kernels */
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>>;
/* intel_sierra: every cell - the E-core vector bundle, TPAUSE waits & CMPCCXADD atomics */
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
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_sierra_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_sierra_k>>;
/* intel_diamond: every cell - the AVX10.2 lineage, TPAUSE waits & CMPCCXADD atomics */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_diamond_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_diamond_k>>;
#elif NK_TARGET_ARM64_
/* bf16 / f16: aws_g2 floor → apple_m2 → aws_g5 */
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
/* fp8 family: apple_m2 → aws_g5 → nvidia_vera */
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
extern template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
/* i8 / i4 / u8 / u4: apple_m2 → aws_g5 */
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
/* u1: aws_g2 floor → aws_g5, bit kernels; the wider bundles pick the same NEON kit */
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g5_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g5_k>>;
/* apple_m2: the remaining cells - the aws_g2 kits under LSE & RCpc atomics */
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::apple_m2_k>>;
extern template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::apple_m2_k>>;
#elif NK_TARGET_RISCV64_
/* alibaba_c930: every cell - the serial kits under Zawrs waits & Zacas atomics */
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
#if USEARCH_BUILD_CUDA
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k>>;
extern template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k>>;
extern template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k>>;
extern template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k>>;
extern template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k>>;
extern template struct specialized_index<vram_index<nk_u8_k, nk_kernel_euclidean_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k>>;
extern template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k>>;
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

/** @brief Whether @p alternative_type_ is listed in @p variant_type_ - the membership fact the
 *      ladder asserts against the explicit list. */
template <typename alternative_type_, typename variant_type_>
inline constexpr bool variant_carries = false;
template <typename alternative_type_, typename... alternative_types_>
inline constexpr bool variant_carries<alternative_type_, std::variant<alternative_types_...>> =
    (std::is_same_v<alternative_type_, alternative_types_> || ...);

#pragma endregion Multi Index

#pragma region Construction

/** @brief Appends one RAM shard per named memory domain - or one per allocatable domain of a
 *      transient topology when the list is empty, so a bare init still spans the machine. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
static status_t place_shards(multi_type_& variant, usearch_dense_options_t const& options) noexcept {
    usearch_ram_options_t const& ram = options.ram;
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

/** @brief Appends one comma-separated codename to the handle's capability string. */
static void append_capability(punned_dense_index_t& handle, char const* name) noexcept {
    std::size_t const used = std::strlen(handle.capabilities_string);
    std::snprintf(handle.capabilities_string + used, sizeof(handle.capabilities_string) - used, "%s%s", used ? "," : "",
                  name);
}

/** @brief Fills the handle's capability string once, from the picked NumKong kit. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
static void compose_capabilities(punned_dense_index_t& handle, multi_type_ const&) noexcept {
    constexpr nk_capability_t picked = dense_quads_chain<multi_type_::kernel_kind_k, typename multi_type_::scalar_t,
                                                         multi_type_::capability_k>::picked_k;
    for (numkong_capability_name_t const& named : numkong_capability_names_k)
        if (picked & named.bit) append_capability(handle, named.name);
}

#if USEARCH_BUILD_CUDA
/** @brief Appends one VRAM shard per named device ordinal; an empty list takes ordinal zero. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static status_t place_shards(multi_type_& variant, usearch_dense_options_t const& options) noexcept {
    usearch_vram_options_t const& vram = options.vram;
    if (!vram.devices || !vram.devices_count) return specialized_index {variant}.emplace(0);
    for (std::size_t position = 0; position != vram.devices_count; ++position)
        if (status_t const placed = specialized_index {variant}.emplace(static_cast<int>(vram.devices[position]));
            failed(placed))
            return placed;
    return status_t::success_k;
}

/** @brief Fills the handle's capability string once, from the resident device's generation. */
template <typename multi_type_>
    requires is_vram_index<multi_type_>
static void compose_capabilities(punned_dense_index_t& handle, multi_type_& variant) noexcept {
    if (!specialized_index {variant}.shards_count()) return;
    auto const [specs, queried] = cuda_device_specs_t::query(specialized_index {variant}.device());
    if (failed(queried)) return;
    for (gpu_class_name_t const& named : gpu_class_names_k)
        if (cuda_arch_of(named.gpu_class) == specs.arch()) return append_capability(handle, named.name);
}
#endif

/** @brief The host allocator a multi-index keeps its tables on: the first named memory domain
 *      on RAM, the heap's domain otherwise. */
template <typename multi_type_>
static forkunion_allocator_t host_allocator_of(usearch_dense_options_t const& options) noexcept {
    if constexpr (is_ram_index<multi_type_>)
        if (options.ram.memory_domains && options.ram.memory_domains_count)
            return forkunion_allocator_t {static_cast<fu_memory_domain_id_t>(options.ram.memory_domains[0])};
    return forkunion_allocator_t {no_memory_domain_k};
}

/** @brief Allocates the handle holding @p variant_type_, places its shards, and publishes the
 *      opaque pointer. */
template <typename variant_type_>
static usearch_dense_index_t make_punned_dense_index(usearch_dense_options_t const* options,
                                                     usearch_error_t* error) noexcept {
    punned_dense_index_t* const handle = new (std::nothrow)
        punned_dense_index_t(std::in_place_type_t<variant_type_>(), host_allocator_of<variant_type_>(*options));
    if (!handle) return propagate_error(usearch_error_out_of_memory_k, error, "Failed to allocate the index"), nullptr;
    handle->requested_dimensions = options->dimensions;
    handle->requested_connectivity = options->connectivity ? static_cast<connectivity_t>(options->connectivity)
                                                           : connectivity_32_k;
    // Placement is init's job, so a bad domain fails here rather than at the first reservation.
    variant_type_& variant = std::get<variant_type_>(handle->variants);
    if (status_t const placed = place_shards(variant, *options); failed(placed))
        return delete handle, propagate_error(placed, error), nullptr;
    compose_capabilities(*handle, variant);
    return propagate_success(error), reinterpret_cast<usearch_dense_index_t>(handle);
}

/** @brief Whether one rung is genuinely distinct from the class it degrades to: a different picked
 *      kit or a different ForkUnion mask - the one fact justifying its instantiation. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_>
consteval bool rung_advances() noexcept {
    constexpr cpu_class_t parent_k = parent_class_of(cpu_class_);
    using scalar_t = typename nk::type_for<dtype_>::type;
    using upper_chain_t = dense_quads_chain<kernel_, scalar_t, numkong_capabilities_of(cpu_class_)>;
    using lower_chain_t = dense_quads_chain<kernel_, scalar_t, numkong_capabilities_of(parent_k)>;
    return upper_chain_t::picked_k != lower_chain_t::picked_k ||
           forkunion_capabilities_of<cpu_class_>() != forkunion_capabilities_of<parent_k>();
}

/** @brief The unconditional serial floor - every ladder's last rung. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t descend_ram_ladder(usearch_dense_options_t const* options, nk_capability_t,
                                                usearch_error_t* error) noexcept {
    return make_punned_dense_index<ram_index<dtype_, kernel_, cpu_class_t::serial_k>>(options, error);
}

/** @brief One distinct rung: taken when the machine reports the class, else falls through. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
    requires(rung_advances<dtype_, kernel_, cpu_class_>())
static usearch_dense_index_t descend_ram_ladder(usearch_dense_options_t const* options, nk_capability_t available,
                                                usearch_error_t* error) noexcept {
    if (numkong_admits(available, cpu_class_) && forkunion_admits<cpu_class_>())
        return make_punned_dense_index<ram_index<dtype_, kernel_, cpu_class_>>(options, error);
    return descend_ram_ladder<dtype_, kernel_, lower_classes_...>(options, available, error);
}

/** @brief A rung whose class picks nothing new: skipped without instantiating its index. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
    requires(!rung_advances<dtype_, kernel_, cpu_class_>())
static usearch_dense_index_t descend_ram_ladder(usearch_dense_options_t const* options, nk_capability_t available,
                                                usearch_error_t* error) noexcept {
    static_assert(!variant_carries<ram_index<dtype_, kernel_, cpu_class_>, punned_dense_index_t::variants_t>,
                  "This rung no longer advances past the one below - drop its arm from the variant list");
    return descend_ram_ladder<dtype_, kernel_, lower_classes_...>(options, available, error);
}

/** @brief Picks the RAM variant for one @c (dtype,kernel) cell: CPU classes newest-first, each
 *      rung compiled only where it advances past the class it degrades to, so the ladders stay
 *      sparse per type and every alternative is genuinely distinct. A refused rung falls to the
 *      next in the list, so lineages that share a floor simply interleave. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_ram_index(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    nk_capability_t const available = nk_capabilities_available();
#if NK_TARGET_X8664_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::intel_diamond_k, cpu_class_t::amd_genoa_k,
                              cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k, cpu_class_t::intel_sierra_k,
                              cpu_class_t::intel_haswell_k>( //
        options, available, error);
#elif NK_TARGET_ARM64_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::nvidia_vera_k, cpu_class_t::aws_g5_k,
                              cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>( //
        options, available, error);
#elif NK_TARGET_RISCV64_
    return descend_ram_ladder<dtype_, kernel_, cpu_class_t::alibaba_c930_k>(options, available, error);
#else
    return descend_ram_ladder<dtype_, kernel_>(options, available, error);
#endif
}

#if USEARCH_BUILD_CUDA
/** @brief The VRAM cells with adapted CUDA kits - the device half of the variant list. */
constexpr bool vram_cell_wired(nk_dtype_t dtype, nk_kernel_kind_t kernel) noexcept {
    switch (dtype) {
    case nk_i8_k: return kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_u8_k: return kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_f16_k: return kernel == nk_kernel_angular_k;
    case nk_bf16_k: return kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_i4_k: return kernel == nk_kernel_angular_k;
    case nk_u1_k: return kernel == nk_kernel_hamming_k || kernel == nk_kernel_jaccard_k;
    default: return false;
    }
}

/** @brief The VRAM sibling of @ref make_ram_index; the device tier is a template argument
 *      of the index itself, so the cell alone selects the variant. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_vram_index(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    if constexpr (!vram_cell_wired(dtype_, kernel_))
        return propagate_error(usearch_error_unknown_kind_k, error,
                               "No CUDA kit is wired for this (quantization,metric) pair"),
               nullptr;
    else return make_punned_dense_index<vram_index<dtype_, kernel_>>(options, error);
}
#else
/** @brief The portable build's stand-in: every VRAM request lands on one static message. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t make_vram_index(usearch_dense_options_t const*, usearch_error_t* error) noexcept {
    return propagate_error(usearch_error_device_k, error, "This build carries no CUDA backends"), nullptr;
}
#endif

/** @brief Routes one lowered cell onto its residency's constructor. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_index_t with_residency(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->residency) {
    case usearch_residency_ram_k: return make_ram_index<dtype_, kernel_>(options, error);
    case usearch_residency_vram_k: return make_vram_index<dtype_, kernel_>(options, error);
    case usearch_residency_nvme_k:
        return propagate_error(usearch_error_config_mismatch_k, error,
                               "The NVMe variants join this dispatch with the normalized batched search"),
               nullptr;
    default: return propagate_error(usearch_error_unknown_kind_k, error, "Unknown residency"), nullptr;
    }
}

/** @brief Routes a numeric dtype's metric onto its three wired kernels. */
template <nk_dtype_t dtype_>
    requires(dtype_ != nk_u1_k)
static usearch_dense_index_t with_metric(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->metric) {
    case usearch_dense_metric_inner_k: return with_residency<dtype_, nk_kernel_dot_k>(options, error);
    case usearch_dense_metric_angular_k: return with_residency<dtype_, nk_kernel_angular_k>(options, error);
    case usearch_dense_metric_euclidean_k: return with_residency<dtype_, nk_kernel_euclidean_k>(options, error);
    default:
        return propagate_error(usearch_error_unknown_kind_k, error, "This metric needs a bit-packed quantization"),
               nullptr;
    }
}

/** @brief Routes the bit-packed dtype's metric onto the bit kernels. */
template <nk_dtype_t dtype_>
    requires(dtype_ == nk_u1_k)
static usearch_dense_index_t with_metric(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->metric) {
    case usearch_dense_metric_inner_k: return with_residency<dtype_, nk_kernel_dot_k>(options, error);
    case usearch_dense_metric_hamming_k: return with_residency<dtype_, nk_kernel_hamming_k>(options, error);
    case usearch_dense_metric_jaccard_k: return with_residency<dtype_, nk_kernel_jaccard_k>(options, error);
    default:
        return propagate_error(usearch_error_unknown_kind_k, error,
                               "No kernel kit is wired for this (quantization,metric) pair"),
               nullptr;
    }
}

/** @brief Lowers the runtime @c (quantization,metric) pair onto the wired cells - every
 *      numeric dtype speaks inner, angular & euclidean; the bit dtype speaks inner,
 *      hamming & jaccard. */
static usearch_dense_index_t with_cell(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->quantization) {
    case usearch_scalar_bf16_k: return with_metric<nk_bf16_k>(options, error);
    case usearch_scalar_f16_k: return with_metric<nk_f16_k>(options, error);
    case usearch_scalar_e5m2_k: return with_metric<nk_e5m2_k>(options, error);
    case usearch_scalar_e4m3_k: return with_metric<nk_e4m3_k>(options, error);
    case usearch_scalar_e3m2_k: return with_metric<nk_e3m2_k>(options, error);
    case usearch_scalar_e2m3_k: return with_metric<nk_e2m3_k>(options, error);
    case usearch_scalar_i8_k: return with_metric<nk_i8_k>(options, error);
    case usearch_scalar_i4x2_k: return with_metric<nk_i4_k>(options, error);
    case usearch_scalar_u8_k: return with_metric<nk_u8_k>(options, error);
    case usearch_scalar_u4x2_k: return with_metric<nk_u4_k>(options, error);
    case usearch_scalar_u1x8_k: return with_metric<nk_u1_k>(options, error);
    default: return propagate_error(usearch_error_unknown_kind_k, error, "Unsupported quantization"), nullptr;
    }
}

#pragma endregion Construction

#pragma region Multi Verbs

/** @brief The state behind @c usearch_executor_t: the discovered topology it owns, and one
 *      residency's executor - the variant is the kind tag, so no second word can disagree. */
struct punned_executor_t {
    fu_topology_t topology = nullptr;
    std::variant<forkunion_executor_t
#if USEARCH_BUILD_CUDA
                 ,
                 cuda_executor_t
#endif
                 >
        executor;
};

/** @brief The typed executor behind the C handle, or null with @c usearch_error_device_k on a
 *      kind mismatch. */
template <typename multi_type_>
static typename multi_type_::executor_t* executor_of(usearch_executor_t punned_executor,
                                                     usearch_error_t* error) noexcept {
    punned_executor_t* const state = static_cast<punned_executor_t*>(punned_executor);
    typename multi_type_::executor_t* const typed =
        state ? std::get_if<typename multi_type_::executor_t>(&state->executor) : nullptr;
    if (!typed) propagate_error(usearch_error_device_k, error, "The executor's kind must match the handle's residency");
    return typed;
}

/** @brief Adds one fan-out's telemetry into the caller's stats - accumulating, so chunked
 *      verbs report the whole call rather than the last chunk. */
template <typename result_type_>
static void accumulate_stats(usearch_dense_traversal_stats_t* stats, result_type_ const& result) noexcept {
    if (!stats) return;
    stats->visited_members += result.visited_members;
    stats->computed_distances += result.computed_distances;
    stats->restarts += result.restarts;
}

/** @brief One strided C matrix's rank-2 shape: extents in storage values, so packed dtypes count
 *      bytes; a zero stride means contiguous rows. */
template <typename scalar_type_>
static nk::shape_storage_<2> rows_shape_of(std::size_t rows_count, std::size_t stride,
                                           std::size_t dimensions) noexcept {
    std::size_t const row_values = divide_round_up<std::size_t>(dimensions, nk::dimensions_per_value<scalar_type_>());
    std::ptrdiff_t const row_bytes = static_cast<std::ptrdiff_t>(stride ? stride : row_values * sizeof(scalar_type_));
    return {{rows_count, row_values}, {row_bytes, static_cast<std::ptrdiff_t>(sizeof(scalar_type_))}, 2};
}

/** @brief A chunk's rows in @p multi_type_ 's stored scalar: the caller's own when they match,
 *      else cast into @p staged. */
template <typename multi_type_>
static nk::matrix_view<typename multi_type_::scalar_t> stored_rows_of( //
    char const* rows, std::size_t rows_count, std::size_t row_bytes, nk_dtype_t source_dtype, std::size_t dimensions,
    nk::matrix<typename multi_type_::scalar_t>& staged) noexcept {
    using scalar_t = typename multi_type_::scalar_t;
    if (source_dtype == multi_type_::vector_dtype_k)
        return {rows, rows_shape_of<scalar_t>(rows_count, row_bytes, dimensions)};
    for (std::size_t row = 0; row != rows_count; ++row)
        nk_cast(rows + row * row_bytes, source_dtype, dimensions, staged.span().row(row).as_vector().byte_data(),
                multi_type_::vector_dtype_k);
    return {reinterpret_cast<char const*>(staged.data()), rows_shape_of<scalar_t>(rows_count, 0, dimensions)};
}

/** @brief The one search engine both C verbs drive: chunked, staged, scattered; a real predicate
 *      narrows it to CPU handles. */
template <typename predicate_type_>
static void search_dense_rows(usearch_dense_index_t index, usearch_executor_t punned_executor,        //
                              void const* queries, size_t queries_count, size_t queries_stride,       //
                              usearch_scalar_t scalar_kind, size_t wanted,                            //
                              usearch_dense_expansion_config_t const* config,                         //
                              predicate_type_ predicate,                                              //
                              usearch_progress_callback_t progress, void* progress_state,             //
                              usearch_key_t* keys, size_t keys_stride,                                //
                              usearch_distance_t* distances, size_t distances_stride, size_t* counts, //
                              usearch_dense_traversal_stats_t* stats, usearch_error_t* error) noexcept {
    punned_dense_index_t* const punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    if (!queries || !keys || !distances || !counts)
        return propagate_error(usearch_error_malformed_input_k, error,
                               "Queries and all three output planes must not be null"),
               void();
    if (stats) *stats = {};
    if (!queries_count) return propagate_success(error);
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            constexpr bool filtered_k = !std::is_same_v<predicate_type_, dummy_predicate_t>;
            if constexpr (filtered_k && !is_ram_index<variant_type_>)
                propagate_error(usearch_error_device_k, error, "Filtered search needs a CPU-resident handle");
            else {
                nk_dtype_t const source_dtype = nk_dtype_of(scalar_kind);
                if (source_dtype == nk_dtype_unknown_k)
                    return propagate_error(usearch_error_unknown_kind_k, error, "Unknown scalar kind"), void();
                typename variant_type_::executor_t* const executor = executor_of<variant_type_>(punned_executor, error);
                if (!executor) return;
                using scalar_t = typename variant_type_::scalar_t;
                using distance_t = typename variant_type_::distance_t;
                using key_t = typename variant_type_::key_t;
                typename variant_type_::search_config_t search_config {};
                if (config) search_config.expansion_factor = static_cast<u32_t>(config->expansion_factor);

                std::size_t const dimensions = specialized_index {variant}.dimensions();
                std::size_t const query_bytes = queries_stride ? queries_stride
                                                               : divide_round_up<std::size_t>(
                                                                     nk_dtype_bits(source_dtype) * dimensions, 8);
                std::size_t const keys_bytes = keys_stride ? keys_stride : wanted * sizeof(usearch_key_t);
                std::size_t const distances_bytes = distances_stride ? distances_stride
                                                                     : wanted * sizeof(usearch_distance_t);

                // The engines score in f32 and the ABI in double, so every ranking is staged and
                // scattered; queries in a foreign scalar type convert chunk by chunk.
                std::size_t const chunk = min_of(queries_count, increment_rows_k);
                growing_buffer<key_t> staged_keys {global_allocator};
                growing_buffer<distance_t> staged_distances {global_allocator};
                if (failed(staged_keys.resize(chunk * wanted)) || failed(staged_distances.resize(chunk * wanted)))
                    return propagate_error(usearch_error_out_of_memory_k, error, "Failed to stage the output planes"),
                           void();
                nk::matrix<scalar_t> staged_queries;
                if (source_dtype != variant_type_::vector_dtype_k) {
                    staged_queries = nk::matrix<scalar_t>::try_zeros({chunk, dimensions});
                    if (staged_queries.empty())
                        return propagate_error(usearch_error_out_of_memory_k, error,
                                               "Failed to stage the converted queries"),
                               void();
                }

                char const* const queries_base = static_cast<char const*>(queries);
                char* const keys_base = reinterpret_cast<char*>(keys);
                char* const distances_base = reinterpret_cast<char*>(distances);
                std::size_t resolved = 0;
                while (resolved != queries_count) {
                    std::size_t const taking = min_of(chunk, queries_count - resolved);
                    nk::matrix_view<scalar_t> const chunk_queries = stored_rows_of<variant_type_>(
                        queries_base + resolved * query_bytes, taking, query_bytes, source_dtype, dimensions,
                        staged_queries);
                    std::span<key_t> const keys_span {staged_keys.data(), taking * wanted};
                    std::span<distance_t> const distances_span {staged_distances.data(), taking * wanted};
                    std::span<std::size_t> const counts_span {counts + resolved, taking};
                    typename variant_type_::search_result_t found;
                    if constexpr (filtered_k)
                        found = specialized_index {variant}.search_if(chunk_queries, *executor, search_config, wanted,
                                                                      keys_span, distances_span, counts_span,
                                                                      predicate);
                    else
                        found = specialized_index {variant}.search(chunk_queries, *executor, search_config, wanted,
                                                                   keys_span, distances_span, counts_span);
                    accumulate_stats(stats, found);
                    if (failed(found.status)) return propagate_error(found.status, error), void();
                    for (std::size_t query = 0; query != taking; ++query) {
                        std::memcpy(keys_base + (resolved + query) * keys_bytes, staged_keys.data() + query * wanted,
                                    wanted * sizeof(key_t));
                        nk_cast(staged_distances.data() + query * wanted, nk_f32_k, wanted,
                                distances_base + (resolved + query) * distances_bytes, nk_f64_k);
                    }
                    resolved += taking;
                    if (progress && progress(resolved, queries_count, progress_state) == usearch_cancel_k)
                        return propagate_error(usearch_error_cancelled_k, error, "The handler stopped the search"),
                               void();
                }
                propagate_success(error);
            }
        },
        punned_index->variants);
}

#pragma endregion Multi Verbs

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT usearch_dense_index_t usearch_dense_init(usearch_dense_options_t const* options,
                                                        usearch_error_t* error) {
    if (!options) return propagate_error(usearch_error_malformed_input_k, error, "Options must not be null"), nullptr;
    return with_cell(options, error);
}

USEARCH_EXPORT void usearch_dense_free(usearch_dense_index_t index) {
    delete reinterpret_cast<punned_dense_index_t*>(index);
}

USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_index_t index, size_t capacity, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    std::visit(
        [&](auto& variant) {
            propagate_error(specialized_index {variant}.reserve(
                                {capacity, punned_index->requested_connectivity, punned_index->requested_dimensions}),
                            error);
        },
        punned_index->variants);
}

USEARCH_EXPORT void usearch_dense_clear(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    propagate_error(
        std::visit([](auto& variant) { return specialized_index {variant}.clear(); }, punned_index->variants), error);
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT size_t usearch_dense_size(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    propagate_success(error);
    return std::visit([](auto& variant) { return specialized_index {variant}.size(); }, punned_index->variants);
}

USEARCH_EXPORT size_t usearch_dense_capacity(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    propagate_success(error);
    return std::visit([](auto& variant) { return specialized_index {variant}.capacity(); }, punned_index->variants);
}

USEARCH_EXPORT size_t usearch_dense_memory_usage(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    propagate_success(error);
    return std::visit([](auto& variant) { return specialized_index {variant}.memory_usage(); }, punned_index->variants);
}

USEARCH_EXPORT size_t usearch_dense_connectivity(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    propagate_success(error);
    std::size_t const frozen = std::visit([](auto& variant) { return specialized_index {variant}.connectivity(); }, //
                                          punned_index->variants);
    return frozen ? frozen : static_cast<size_t>(punned_index->requested_connectivity);
}

USEARCH_EXPORT size_t usearch_dense_dimensions(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    propagate_success(error);
    std::size_t const frozen = std::visit([](auto& variant) { return specialized_index {variant}.dimensions(); }, //
                                          punned_index->variants);
    return frozen ? frozen : punned_index->requested_dimensions;
}

USEARCH_EXPORT char const* usearch_dense_capabilities(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    return propagate_success(error), punned_index->capabilities_string;
}

USEARCH_EXPORT size_t usearch_dense_count(usearch_dense_index_t index, usearch_key_t key, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    return propagate_success(error),
           std::visit([&](auto& variant) { return specialized_index {variant}.count(uuid_of(key)); },
                      punned_index->variants);
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT size_t usearch_dense_add(                             //
    usearch_dense_index_t index, usearch_executor_t punned_executor, //
    usearch_key_t const* keys,                                       //
    void const* vectors, size_t count, size_t stride,                //
    usearch_scalar_t scalar_kind,                                    //
    usearch_dense_expansion_config_t const* config,                  //
    usearch_progress_callback_t progress, void* progress_state,      //
    usearch_dense_traversal_stats_t* stats,                          //
    usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    if (!keys || !vectors)
        return propagate_error(usearch_error_malformed_input_k, error, "Keys and vectors must not be null"), 0ul;
    if (stats) *stats = {};
    if (!count) return propagate_success(error), 0ul;
    return std::visit(
        [&]<typename variant_type_>(variant_type_& variant) -> size_t {
            nk_dtype_t const source_dtype = nk_dtype_of(scalar_kind);
            if (source_dtype == nk_dtype_unknown_k)
                return propagate_error(usearch_error_unknown_kind_k, error, "Unknown scalar kind"), 0ul;
            typename variant_type_::executor_t* const executor = executor_of<variant_type_>(punned_executor, error);
            if (!executor) return 0ul;
            using scalar_t = typename variant_type_::scalar_t;
            std::span const uuids {reinterpret_cast<uuid_t const*>(keys), count};
            typename variant_type_::insert_config_t insert_config {};
            if (config) insert_config.expansion_factor = static_cast<u32_t>(config->expansion_factor);

            // Only a handler or a conversion splits the batch; otherwise it is one all-or-nothing call.
            std::size_t const dimensions = specialized_index {variant}.dimensions();
            bool const converting = source_dtype != variant_type_::vector_dtype_k;
            std::size_t const row_bytes =
                stride ? stride : divide_round_up<std::size_t>(nk_dtype_bits(source_dtype) * dimensions, 8);
            char const* const rows_base = static_cast<char const*>(vectors);
            std::size_t const chunk = progress || converting ? min_of(count, increment_rows_k) : count;
            nk::matrix<scalar_t> staged;
            if (converting) {
                staged = nk::matrix<scalar_t>::try_zeros({chunk, dimensions});
                if (staged.empty())
                    return propagate_error(usearch_error_out_of_memory_k, error, "Failed to stage the converted rows"),
                           0ul;
            }
            std::size_t published = 0;
            while (published != count) {
                std::size_t const taking = min_of(chunk, count - published);
                nk::matrix_view<scalar_t> const rows = stored_rows_of<variant_type_>(
                    rows_base + published * row_bytes, taking, row_bytes, source_dtype, dimensions, staged);
                typename variant_type_::insert_result_t const inserted = specialized_index {variant}.add(
                    uuids.subspan(published, taking), rows, *executor, insert_config);
                accumulate_stats(stats, inserted);
                if (failed(inserted.status)) return propagate_error(inserted.status, error), published;
                published += taking;
                if (progress && progress(published, count, progress_state) == usearch_cancel_k)
                    return propagate_error(usearch_error_cancelled_k, error, "The handler stopped the insertion"),
                           published;
            }
            return propagate_success(error), count;
        },
        punned_index->variants);
}

USEARCH_EXPORT void usearch_dense_search(                             //
    usearch_dense_index_t index, usearch_executor_t punned_executor,  //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t scalar_kind, size_t wanted,                      //
    usearch_dense_expansion_config_t const* config,                   //
    usearch_progress_callback_t progress, void* progress_state,       //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    size_t* counts,                                                   //
    usearch_dense_traversal_stats_t* stats,                           //
    usearch_error_t* error) {
    search_dense_rows(index, punned_executor, queries, queries_count, queries_stride, scalar_kind, wanted, config,
                      dummy_predicate_t {}, progress, progress_state, keys, keys_stride, distances, distances_stride,
                      counts, stats, error);
}

USEARCH_EXPORT void usearch_dense_search_if(                          //
    usearch_dense_index_t index, usearch_executor_t punned_executor,  //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t scalar_kind, size_t wanted,                      //
    usearch_dense_expansion_config_t const* config,                   //
    usearch_predicate_t predicate, void* predicate_state,             //
    usearch_progress_callback_t progress, void* progress_state,       //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    size_t* counts,                                                   //
    usearch_dense_traversal_stats_t* stats,                           //
    usearch_error_t* error) {
    if (!predicate)
        return propagate_error(usearch_error_malformed_input_k, error, "The predicate must not be null"), void();
    key_predicate_t const admitted {predicate, predicate_state};
    search_dense_rows(index, punned_executor, queries, queries_count, queries_stride, scalar_kind, wanted, config,
                      admitted, progress, progress_state, keys, keys_stride, distances, distances_stride, counts, stats,
                      error);
}

USEARCH_EXPORT size_t usearch_dense_get(            //
    usearch_dense_index_t index, usearch_key_t key, //
    void* vectors, size_t wanted, size_t stride,    //
    usearch_scalar_t scalar_kind, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    return std::visit(
        [&]<typename variant_type_>(variant_type_& variant) -> size_t {
            using scalar_t = typename variant_type_::scalar_t;
            if (!vectors) return propagate_success(error), specialized_index {variant}.count(uuid_of(key));
            nk_dtype_t const target_dtype = nk_dtype_of(scalar_kind);
            if (target_dtype == nk_dtype_unknown_k)
                return propagate_error(usearch_error_unknown_kind_k, error, "Unknown scalar kind"), 0ul;
            std::size_t const dimensions = specialized_index {variant}.dimensions();
            if (target_dtype == variant_type_::vector_dtype_k) {
                nk::matrix_span<scalar_t> const rows {static_cast<char*>(vectors),
                                                      rows_shape_of<scalar_t>(wanted, stride, dimensions)};
                return propagate_success(error), specialized_index {variant}.get(uuid_of(key), rows, wanted);
            }
            // A foreign export stages the stored rows and converts each into the caller's type.
            nk::matrix<scalar_t> staged = nk::matrix<scalar_t>::try_zeros({wanted, dimensions});
            if (wanted && staged.empty())
                return propagate_error(usearch_error_out_of_memory_k, error, "Failed to stage the exported rows"), 0ul;
            std::size_t const total = specialized_index {variant}.get(uuid_of(key), staged.span(), wanted);
            std::size_t const row_bytes =
                stride ? stride : divide_round_up<std::size_t>(nk_dtype_bits(target_dtype) * dimensions, 8);
            for (std::size_t row = 0; row != min_of(total, wanted); ++row)
                nk_cast(staged.span().row(row).as_vector().byte_data(), variant_type_::vector_dtype_k, dimensions,
                        static_cast<char*>(vectors) + row * row_bytes, target_dtype);
            return propagate_success(error), total;
        },
        punned_index->variants);
}

#pragma endregion Insertion and Search

#pragma region Serialization

USEARCH_EXPORT size_t usearch_dense_storage_requirements(usearch_dense_index_t index, usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    return propagate_success(error),
           std::visit([](auto& variant) { return specialized_index {variant}.storage_requirements(); },
                      punned_index->variants);
}

/** @brief Fills the geometry and metric a checkpoint froze into @p options; placement stays. */
static void adopt_metadata(safegraphs_metadata_t const& metadata, usearch_dense_options_t* options) noexcept {
    options->dimensions = regular_width(metadata.value_shape).value_or(0);
    options->connectivity = metadata.connectivity;
    // Our own checkpoints name the stored scalar outright; a foreign file speaks only its column dtype.
    std::string_view const quantization = metadata.quantization.empty() ? dtype_name(metadata.value_dtype)
                                                                        : metadata.quantization.view();
    options->quantization = usearch_scalar_of(nk_dtype_named(quantization.data(), quantization.size()));
    options->metric = usearch_metric_of(metadata.metric.view());
}

USEARCH_EXPORT void usearch_dense_save(                              //
    usearch_dense_index_t index, usearch_executor_t punned_executor, //
    usearch_bytes_view_t path,                                       //
    usearch_progress_callback_t progress, void* progress_state,      //
    usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    if (!path.data || !path.length)
        return propagate_error(usearch_error_malformed_input_k, error, "The path must not be empty"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            typename variant_type_::executor_t* const executor = executor_of<variant_type_>(punned_executor, error);
            if (!executor) return;
            shard_ticks_t ticks {progress, progress_state, specialized_index {variant}.shards_count()};
            propagate_error(
                specialized_index {variant}.save(std::string_view {path.data, path.length}, *executor, ticks), error);
        },
        punned_index->variants);
}

USEARCH_EXPORT void usearch_dense_load(                              //
    usearch_dense_index_t index, usearch_executor_t punned_executor, //
    usearch_bytes_view_t path,                                       //
    usearch_progress_callback_t progress, void* progress_state,      //
    usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    if (!path.data || !path.length)
        return propagate_error(usearch_error_malformed_input_k, error, "The path must not be empty"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            typename variant_type_::executor_t* const executor = executor_of<variant_type_>(punned_executor, error);
            if (!executor) return;
            shard_ticks_t ticks {progress, progress_state, specialized_index {variant}.shards_count()};
            propagate_error(
                specialized_index {variant}.load(std::string_view {path.data, path.length}, *executor, ticks), error);
        },
        punned_index->variants);
}

USEARCH_EXPORT void usearch_dense_metadata( //
    usearch_bytes_view_t path, usearch_dense_options_t* options, usearch_error_t* error) {
    if (!path.data || !path.length || !options)
        return propagate_error(usearch_error_malformed_input_k, error, "The path and options must not be null"), void();
    auto const [metadata, read] = safegraphs_read_metadata(std::string_view {path.data, path.length});
    if (failed(read)) return propagate_error(read, error), void();
    adopt_metadata(metadata, options);
    propagate_success(error);
}

USEARCH_EXPORT void usearch_dense_save_buffer(                       //
    usearch_dense_index_t index, usearch_executor_t punned_executor, //
    usearch_bytes_span_t buffer,                                     //
    usearch_progress_callback_t progress, void* progress_state,      //
    usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    if (!buffer.data && buffer.length)
        return propagate_error(usearch_error_malformed_input_k, error, "The buffer must not be null"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            typename variant_type_::executor_t* const executor = executor_of<variant_type_>(punned_executor, error);
            if (!executor) return;
            shard_ticks_t ticks {progress, progress_state, specialized_index {variant}.shards_count()};
            propagate_error(specialized_index {variant}.save(
                                bytes_span_t {reinterpret_cast<byte_t*>(buffer.data), buffer.length}, *executor, ticks),
                            error);
        },
        punned_index->variants);
}

USEARCH_EXPORT void usearch_dense_load_buffer(                       //
    usearch_dense_index_t index, usearch_executor_t punned_executor, //
    usearch_bytes_view_t buffer,                                     //
    usearch_progress_callback_t progress, void* progress_state,      //
    usearch_error_t* error) {
    auto* punned_index = reinterpret_cast<punned_dense_index_t*>(index);
    if (!buffer.data || !buffer.length)
        return propagate_error(usearch_error_malformed_input_k, error, "The buffer must not be empty"), void();
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) {
            typename variant_type_::executor_t* const executor = executor_of<variant_type_>(punned_executor, error);
            if (!executor) return;
            shard_ticks_t ticks {progress, progress_state, specialized_index {variant}.shards_count()};
            propagate_error(
                specialized_index {variant}.load(
                    bytes_view_t {reinterpret_cast<byte_t const*>(buffer.data), buffer.length}, *executor, ticks),
                error);
        },
        punned_index->variants);
}

USEARCH_EXPORT void usearch_dense_metadata_buffer( //
    usearch_bytes_view_t buffer, usearch_dense_options_t* options, usearch_error_t* error) {
    if (!buffer.data || !buffer.length || !options)
        return propagate_error(usearch_error_malformed_input_k, error, "The buffer and options must not be null"),
               void();
    auto const [metadata, read] = safegraphs_read_metadata(
        bytes_view_t {reinterpret_cast<byte_t const*>(buffer.data), buffer.length});
    if (failed(read)) return propagate_error(read, error), void();
    adopt_metadata(metadata, options);
    propagate_success(error);
}

#pragma endregion Serialization

#pragma region Executors

USEARCH_EXPORT usearch_executor_t usearch_executor_for_ram([[maybe_unused]] usearch_ram_options_t const* ram,
                                                           size_t threads, usearch_error_t* error) {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_error_device_k, error, "Topology discovery failed"), nullptr;
    auto [executor, spawned] = forkunion_executor_t::spawn(topology, threads);
    if (failed(spawned)) return fu_topology_delete(topology), propagate_error(spawned, error), nullptr;
    punned_executor_t* const state = new (std::nothrow) punned_executor_t {};
    if (!state)
        return fu_topology_delete(topology),
               propagate_error(usearch_error_out_of_memory_k, error, "Failed to allocate the executor"), nullptr;
    state->topology = topology;
    state->executor = std::move(executor);
    return propagate_success(error), reinterpret_cast<usearch_executor_t>(state);
}

USEARCH_EXPORT usearch_executor_t usearch_executor_for_vram([[maybe_unused]] usearch_vram_options_t const* vram,
                                                            usearch_error_t* error) {
#if USEARCH_BUILD_CUDA
    static_assert(sizeof(int) == sizeof(int32_t));
    int const fallback_device = 0;
    std::span<int const> const devices = vram && vram->devices && vram->devices_count
                                             ? std::span {reinterpret_cast<int const*>(vram->devices),
                                                          vram->devices_count}
                                             : std::span {&fallback_device, 1};
    auto [executor, spawned] = cuda_executor_t::spawn(devices);
    if (failed(spawned)) return propagate_error(spawned, error), nullptr;
    punned_executor_t* const state = new (std::nothrow) punned_executor_t {};
    if (!state)
        return propagate_error(usearch_error_out_of_memory_k, error, "Failed to allocate the executor"), nullptr;
    state->executor = std::move(executor);
    return propagate_success(error), reinterpret_cast<usearch_executor_t>(state);
#else
    return propagate_error(usearch_error_device_k, error, "This build carries no CUDA executor"), nullptr;
#endif
}

USEARCH_EXPORT void usearch_executor_drain(usearch_executor_t punned_executor, usearch_error_t* error) {
    if (!punned_executor)
        return propagate_error(usearch_error_malformed_input_k, error, "The executor must not be null"), void();
    punned_executor_t* const state = static_cast<punned_executor_t*>(punned_executor);
    propagate_error(std::visit([](auto& executor) { return executor.drain(); }, state->executor), error);
}

USEARCH_EXPORT void usearch_executor_free(usearch_executor_t punned_executor) {
    punned_executor_t* const state = static_cast<punned_executor_t*>(punned_executor);
    if (!state) return;
    fu_topology_t const topology = state->topology;
    delete state; // The executor drops first: the pool dies before the topology it was spawned over.
    if (topology) fu_topology_delete(topology);
}

#pragma endregion Executors

} // extern "C"
