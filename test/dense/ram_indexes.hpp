/**
 *  @file test/dense/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date August 31, 2026
 *  @brief The CPU-side suite composer for @c dense::ram_indexes - the sharded index over one
 *      shard per memory domain - shared by its translation unit.
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

namespace unum::usearch::test::dense {

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t,      //
    nk_capability_t caps_ = nk_cap_serial_k,     //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static std::size_t test_ram_indexes_configuration(char const* name, u64_t) {
    if constexpr (!quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

/**
 *  @brief An insertion arriving while an absorb into the index runs answers @c contended_k and
 *      lands nothing, and an absorb meeting a batch in flight answers the same and leaves the
 *      donor whole; every row that did land resolves exactly once.
 *
 *  One thread absorbs, retrying while a racing insertion holds a batch in flight, while another
 *  inserts single rows until the absorb returns. A round that sees no refusal refills the donor and
 *  races again.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_absorb_refusals(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const part = 2 * dense_thread_nodes_k, rounds = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto vectors = make_vectors<scalar_t>(part * (rounds + 1), dimensions, stream_key(seed, "indexes_absorb_refusals"));
    auto rows_of = [&](std::size_t first) { return vectors(nk::range(first, first + part), nk::slice); };
    std::vector<key_t> keys(part * (rounds + 1));
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = static_cast<key_t>(1 + row);
    auto keys_of = [&](std::size_t first) { return std::span<key_t const> {keys.data() + first, part}; };

    std::array<dense_indexes_executor<indexes_type_>, 2> drivers;
    for (dense_indexes_executor<indexes_type_> const& driver : drivers)
        if (failed(driver.status)) return failures + expect(false);
    indexes_type_ recipient {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ donor {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(recipient)) && succeeded(place_shards(donor)));
    for (indexes_type_* built : {&recipient, &donor})
        failures += expect(succeeded(built->reserve({part * (rounds + 1), connectivity_16_k, dimensions}).status));
    failures += expect(succeeded(recipient.insert(keys_of(0), rows_of(0), drivers[0].executor).status));
    failures += expect(succeeded(donor.insert(keys_of(part), rows_of(part), drivers[1].executor).status));

    // Racing rows take keys past every part's, so each resolves to the one row it landed.
    std::vector<key_t> landed_keys;
    key_t next_key = static_cast<key_t>(keys.size() + 1);
    {
        auto [plan, plan_status] = recipient.plan_insert(1, drivers[0].executor);
        failures += expect(succeeded(plan_status));
        failures += expect_eq(
            plan.insert(0, std::span<key_t const> {&next_key, 1}, vectors(nk::range(0, 1), nk::slice)),
            status_t::success_k);
        failures += expect_eq(recipient.absorb(donor, drivers[1].executor).status, status_t::contended_k);
        failures += expect_eq(donor.size(), part);
        failures += expect(succeeded(plan.finish().status));
        landed_keys.push_back(next_key++);
    }

    dense_indexes_row_driver<indexes_type_> rows_driver {recipient, drivers[0]};
    failures += expect(succeeded(rows_driver.status));
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::size_t refused = 0, unexpected = 0, absorbed_rows = 0;
    for (std::size_t round = 0; round != rounds && !refused; ++round) {
        if (round)
            failures += expect(succeeded(
                donor.insert(keys_of(part * (round + 1)), rows_of(part * (round + 1)), drivers[1].executor).status));
        std::atomic<bool> absorbing {false}, absorbed {false};
        status_t absorb_status = status_t::success_k;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                absorbing.store(true, std::memory_order_release);
                do absorb_status = recipient.absorb(donor, drivers[1].executor).status;
                while (absorb_status == status_t::contended_k);
                absorbed.store(true, std::memory_order_release);
            }
            else if (thread == 1) {
                while (!absorbing.load(std::memory_order_acquire)) std::this_thread::yield();
                for (std::size_t row = 0; !absorbed.load(std::memory_order_acquire); row = (row + 1) % part) {
                    status_t const status =
                        recipient.insert(next_key, vectors.view().row(row).as_vector(), rows_driver.handle()).status;
                    if (status == status_t::success_k) landed_keys.push_back(next_key++);
                    else if (status == status_t::contended_k) ++refused;
                    else ++unexpected;
                    std::this_thread::yield();
                }
            }
        });
        failures += expect_eq(absorb_status, status_t::success_k);
        failures += expect_eq(donor.size(), 0u);
        absorbed_rows += part;
    }
    failures += expect(refused > 0);
    failures += expect_eq(unexpected, 0u);
    failures += expect_eq(recipient.size(), part + absorbed_rows + landed_keys.size());
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != part + absorbed_rows; ++row) miscounted += recipient.count(keys[row]) != 1;
    for (key_t const key : landed_keys) miscounted += recipient.count(key) != 1;
    failures += expect_eq(miscounted, 0u);
    return failures;
}

template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t,      //
    nk_capability_t caps_ = nk_cap_serial_k,     //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires dense_cell_testable<dtype_, kernel_, caps_>
static std::size_t test_ram_indexes_configuration(char const* name, u64_t seed) {

    using indexes_t = ram_indexes<u32_t, dtype_, caps_, waiter_type_, atomic_ref_>;
    if (!dense_cell_admitted<caps_, atomic_ref_>()) return report(name, "capabilities missing at runtime");

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_split_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_pooled_save<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_search_race<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_scan<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_arena_growth<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_refine<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_absorb<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_absorb_many<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_absorb_refusals<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_growth<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_products<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_indexer_search<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_maxsim<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_attend<indexes_t, kernel_>(seed);
    return report(name, failures_count);
}

/** The sharded-index suite entry its translation unit defines and @c dense.cpp
 *  composes explicitly; returns the suite's failure count. */
std::size_t check_ram_indexes(settings_t const& settings);

} // namespace unum::usearch::test::dense
