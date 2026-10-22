/**
 *  @file src/dense_kernels.cuh
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief What every dense GPU device unit compiles beside the kernels: the C value lookup's device
 *      call, the lookup a sharded attend walks through, and the ceiling of the class a pass
 *      compiles for.
 *
 *  Only the NVIDIA and AMD device units include it, so the dispatch units, which compile as plain
 *  C++, see the lookup as the record they fill and never a device function. It stays clear of
 *  @c dense_specializations.cuh, whose CPU metrics would bring NumKong's own kernels into every
 *  device pass.
 */
#pragma once

#include <bit> // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes

#include "usearch/dense/vram_index.hpp" // `dense::vram_keyed_lookup`

#include "lib.hpp"

/** Resolves @p key to its value row through the C lookup, one key per call. */
__device__ inline void const* vram_value_row(value_lookup_t const& lookup, us::utilities::uuid_t key) noexcept {
    usearch_key_t const scored = std::bit_cast<usearch_key_t>(key);
    void const* row = nullptr;
    lookup.callback(lookup.state, &scored, 1, lookup.scalar, lookup.dimensions, &row);
    return row;
}

/** What each shard of a VRAM attend resolves rows through: the C lookup behind the shard's keys. */
using vram_attend_lookup_t = dense::vram_keyed_lookup<us::utilities::uuid_t, value_lookup_t>;

/** The ceiling an AMD device pass instantiates: its class's, read off the one NumKong target the
 *  class compiles under. */
inline constexpr nk_capability_t amd_class_capabilities_k = numkong_capabilities_of(
    NUMKONG_TARGET_CDNA5   ? gpu_class_t::amd_cdna5_k
    : NUMKONG_TARGET_CDNA4 ? gpu_class_t::amd_cdna4_k
                           : gpu_class_t::amd_cdna3_k);
