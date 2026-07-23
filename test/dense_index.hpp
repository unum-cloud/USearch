/**
 *  @brief The CPU-side suite composer for @c dense_ram_index smoke & concurrency tests,
 *      shared by the per-dtype shards - each names a curated set of (dtype, kernel)
 *      cells, never the whole combinatorial space.
 *
 *  @author Ash Vardanian
 *  @file test/dense_index.hpp
 *  @date July 17, 2026
 *
 *  The scenarios themselves live in @c dense_index.cuh, templated over the index type;
 *  each shard names its instantiations, their compile-time & runtime gates, and composes
 *  the suite explicitly. Cells are gated by a compile-time concept, so combinations whose
 *  serial kernels are not yet adapted in @c dense/ram_metric.hpp report as skipped and
 *  light up automatically once the kits land. Shards may repeat cells with a deliberately
 *  flaky waiter that oversleeps at random, widening the interleaving windows around locks
 *  and watermark turns far beyond hardware timing, and may add tiles cells exercising the
 *  matrix engines end to end where the machine offers them. Removal, update, and
 *  filtering tests are deliberately absent: those features are banned at this layer by
 *  design.
 */
#pragma once

#include <fmt/format.h>
#include <numkong/numkong.hpp>

#include "usearch/dense/ram_index.hpp"

#include "dense_index.cuh"

using namespace unum::usearch;

/** @brief Gate: a (dtype, kernel) cell is testable once its serial kernel kit is adapted. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
concept index_combo_supported = requires {
    typename dense_quads_chain<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t;
};

/** @brief No-op sibling picked when the tuple's serial kernels are not adapted yet. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename waiter_type_ = dummy_waiter_t,
          nk_capability_t caps_ = nk_cap_serial_k>
    requires(!index_combo_supported<dtype_, kernel_>)
static std::size_t test_configuration(char const* name) {
    fmt::println("skipped  {:<20} : serial kernels not adapted yet", name);
    return 0;
}

template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t, nk_capability_t caps_ = nk_cap_serial_k>
    requires(index_combo_supported<dtype_, kernel_>)
static std::size_t test_configuration(char const* name) {

    // SIMD-tier cells execute their intrinsics unconditionally, so they run only where
    // the machine offers every requested bit; the tile permission handshake is per-process.
    if constexpr (caps_ != nk_cap_serial_k) {
        if ((nk_capabilities_available() & caps_) != caps_) {
            fmt::println("skipped  {:<20} : capabilities missing at runtime", name);
            return 0;
        }
        nk_configure_thread(nk_capabilities_available());
    }

    using index_t = dense_ram_index<dtype_, kernel_, caps_, waiter_type_, counting_allocator<std::byte>,
                                    counting_allocator<std::byte>, std::allocator<std::byte>>;
    auto make_index = [](allocation_ledger_t& ledger) {
        return index_t {counting_allocator<std::byte>(ledger), counting_allocator<std::byte>(ledger)};
    };

    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_guards<index_t>(make_index);
    failures_count += test_three_vectors<index_t, kernel_>(make_index);
    failures_count += test_growth_and_recall<index_t, kernel_>(make_index);
    failures_count += test_predicated_search<index_t, kernel_>(make_index);
    failures_count += test_save_load_roundtrip<index_t, kernel_>(make_index);
    failures_count += test_concurrent_build<index_t, kernel_>(make_index);
    failures_count += test_concurrent_search<index_t, kernel_>(make_index);
    return report_scenarios(name, failures_count);
}
