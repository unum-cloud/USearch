/**
 *  @brief The CPU-side suite composer for @c dense_ram_indexes - the sharded index over
 *      one shard per memory domain - shared by its translation unit.
 *
 *  @author Ash Vardanian
 *  @file test/dense/ram_indexes.hpp
 *  @date August 31, 2026
 *
 *  The scenarios live in @c dense.cuh, templated over the sharded-index type; this composer
 *  names a curated handful of cells rather than the per-dtype grid, since the routing, key
 *  columns, and executor logic under test are dtype-independent. The counting-allocator
 *  ledger stays with the single-index suite: the sharded index bakes its own placement
 *  allocators, so its leak discipline is the sanitizers' job rather than the ledger's.
 */
#pragma once

#include <fmt/format.h>
#include <numkong/numkong.hpp>

#include "usearch/dense/ram_indexes.hpp"

#include "dense.cuh"

namespace unum {
namespace usearch {

/** @brief Gate: a (dtype, kernel) cell is testable once its serial kernel kit is adapted. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
concept indexes_combo_supported = requires {
    typename dense_quads_chain<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t;
};

/** @brief No-op sibling picked when the cell cannot be compiled on this target: its serial
 *      kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t,      //
    nk_capability_t caps_ = nk_cap_serial_k,     //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires(!indexes_combo_supported<dtype_, kernel_> || !dense_cell_compiled<dtype_, kernel_, caps_>)
static std::size_t test_configuration(char const* name) {
    if constexpr (!indexes_combo_supported<dtype_, kernel_>) return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t,      //
    nk_capability_t caps_ = nk_cap_serial_k,     //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires(indexes_combo_supported<dtype_, kernel_> && dense_cell_compiled<dtype_, kernel_, caps_>)
static std::size_t test_configuration(char const* name) {

    using indexes_t = dense_ram_indexes<u32_t, dtype_, kernel_, caps_, waiter_type_, atomic_ref_>;
    if (!dense_cell_admitted<caps_, atomic_ref_>()) return report(name, "capabilities missing at runtime");

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t>();
    failures_count += test_dense_indexes_duplicate_keys<indexes_t>();
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>();
    failures_count += test_dense_indexes_split_load<indexes_t, kernel_>();
    failures_count += test_dense_indexes_pooled_save<indexes_t, kernel_>();
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>();
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>();
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>();
    failures_count += test_dense_indexes_search_race<indexes_t, kernel_>();
    failures_count += test_dense_indexes_arena_growth<indexes_t, kernel_>();
    return report(name, failures_count);
}

} // namespace usearch
} // namespace unum

#pragma region Suite Entry Points

/** @brief The sharded-index suite entry its translation unit defines and @c dense.cpp
 *      composes explicitly; returns the suite's failure count. */
std::size_t test_dense_indexes();

#pragma endregion Suite Entry Points
