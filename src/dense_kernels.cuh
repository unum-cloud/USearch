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

#include <bit> // `std::bit_cast` - the C key struct & `us::util::uuid128_t` share sixteen bytes

#include "usearch/dense/vram_index.hpp" // `dense::vram_keyed_lookup`

#include "classes.hpp"    // `cuda_class_capabilities_k`, `hip_class_capabilities_k` - the ceiling a pass compiles for
#include "dense_kits.hpp" // `value_lookup_t` - the C lookup a device call resolves keys through

namespace util = unum::usearch::utilities;

/** Resolves @p key to its value row through the C lookup, one key per call. */
__device__ inline void const* vram_value_row(value_lookup_t const& lookup, util::uuid128_t key) noexcept {
    usearch_key_t const scored = std::bit_cast<usearch_key_t>(key);
    void const* row = nullptr;
    lookup.callback(lookup.callback_state, &scored, 1, lookup.scalar, lookup.dimensions, &row);
    return row;
}

/** What each shard of a VRAM attend resolves rows through: the C lookup behind the shard's keys. */
using vram_attend_lookup_t = us::dense::vram_keyed_lookup<util::uuid128_t, value_lookup_t>;
