/**
 *  @file test/dense/ram_index.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief The CPU-side suite composer for @c dense::ram_index smoke & concurrency tests, shared
 *      by the per-dtype shards - each names a curated set of (dtype, kernel) cells, never the
 *      whole combinatorial space.
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

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t,      //
    nk_capability_t caps_ = nk_cap_serial_k,     //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static std::size_t test_ram_index_configuration(char const* name, u64_t) {
    if constexpr (!quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t,      //
    nk_capability_t caps_ = nk_cap_serial_k,     //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires dense_cell_testable<dtype_, kernel_, caps_>
static std::size_t test_ram_index_configuration(char const* name, u64_t seed) {

    using index_t = ram_index<dtype_, caps_, waiter_type_, atomic_ref_, fu::standard_cache_hints_t,
                              counting_allocator_t, counting_allocator_t, std::allocator<byte_t>>;

    if (!dense_cell_admitted<caps_, atomic_ref_>()) return report(name, "capabilities missing at runtime");

    auto make_index = [](allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(
            new (std::nothrow) index_t(kernel_, counting_allocator_t(ledger), counting_allocator_t(ledger)));
    };

    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_guards<index_t>(make_index, seed);
    failures_count += test_dense_three_vectors<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_refine_settles<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_health_audits<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_heal_islands<index_t>(make_index, seed);
    failures_count += test_dense_predicated_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_scan<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_multiply<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_attend<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_indexer_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_save_load_roundtrip<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_load_hostile<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_load_append<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_build<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_save_race<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_immutable_parity<index_t, kernel_>(make_index, seed);
    using concurrent_cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, //
                                           counting_allocator_t>;
    using immutable_cursor_t = ram_cursor<index_t, synchronization_t::immutable_k, //
                                          counting_allocator_t>;
    failures_count += test_dense_cursor<index_t, concurrent_cursor_t, immutable_cursor_t>(make_index, seed);
    failures_count += test_dense_allocation_budget<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_frontier_shedding<index_t>(make_index, seed);
    failures_count += test_dense_overflow_race<index_t>(make_index, seed);
    return report(name, failures_count);
}

/** The starved cell: the same index over ledgered traversal scratch, so a refused re-table
 *  has somewhere to come from, walked by the saturation scenario alone. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
    requires quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>
static std::size_t test_starved_configuration(char const* name, u64_t seed) {
    using index_t =
        ram_index<dtype_, nk_cap_serial_k, dummy_waiter_t, fu::standard_atomic_ref, fu::standard_cache_hints_t,
                  counting_allocator_t, counting_allocator_t, counting_allocator_t>;
    auto make_index = [](allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(
            new (std::nothrow) index_t(kernel_, counting_allocator_t(ledger), counting_allocator_t(ledger)));
    };
    return report(
        name, test_dense_saturated<index_t>(make_index, seed) + test_dense_refine_atomic<index_t>(make_index, seed));
}

/** Per-dtype CPU index suite entries the shard translation units define and
 *  @c dense.cpp composes explicitly; each returns its shard's failure count. */
std::size_t check_ram_index_f32(settings_t const& settings);
std::size_t check_ram_index_bf16(settings_t const& settings);
std::size_t check_ram_index_f16(settings_t const& settings);
std::size_t check_ram_index_e5m2(settings_t const& settings);
std::size_t check_ram_index_i8(settings_t const& settings);
std::size_t check_ram_index_i4(settings_t const& settings);
std::size_t check_ram_index_u8(settings_t const& settings);
std::size_t check_ram_index_u1(settings_t const& settings);

} // namespace unum::usearch::test::dense
