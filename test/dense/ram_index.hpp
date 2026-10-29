/**
 *  @file test/dense/ram_index.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief The cells of the CPU-resident @c dense::ram_index the per-dtype units compose: the full
 *      suite over the allocation ledger, and the starved cell over ledgered scratch.
 *
 *  The scenarios live in @c dense.cuh, templated over the index type. Cells are gated by a
 *  compile-time concept, so a kit absent from the build reports as skipped. The flaky waiter
 *  oversleeps at random, widening the interleaving windows around locks and watermark turns far
 *  beyond hardware timing. Removal, update, and filtering tests are deliberately absent: those
 *  features are banned at this layer by design.
 */
#pragma once

#include <cstring> // `std::memcmp` - a stored row against its source

#include <fmt/format.h>

#include <numkong/numkong.hpp>

#include "usearch/dense/ram_index.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** Whether the stored row of @p node in @p index holds exactly the bytes of row @p row of
 *  @p vectors. */
template <typename index_type_, typename vectors_type_>
static bool ram_index_row_matches(index_type_ const& index, node_id_t node, vectors_type_ const& vectors,
                                  std::size_t row) {
    std::size_t const bytes = packed_bytes(index_type_::vector_dtype_k, index.dimensions());
    return std::memcmp(index.vector(node).byte_data(), vectors.view().row(row).as_vector().byte_data(), bytes) == 0;
}

/** A lone index keeps its keys without a reservation: the first insert grows it, every key resolves
 *  to both of its rows through @c key_of, @c count, @c find_each and @c get, a split carries every
 *  key beside its row, and a save and load round trip keeps every key. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_ram_index_keys(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    using keys_view_t = typename index_type_::keys_view_t;
    std::size_t failures = 0;
    std::size_t const rows = dense_thread_nodes_k, dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "index_keys"));
    // Rows 2k and 2k + 1 share key k + 1, so every lookup must find a pair.
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row / 2);
    auto rows_of_key = [&](key_t const& key) { return 2 * (dense_key_number(key) - 1); };
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        auto assigned = nk::vector<node_id_t>::zeros(rows).value;
        failures += expect_eq(index.capacity(), 0u);
        failures += expect(
            succeeded(index.insert(keys_view_t {keys.data(), rows}, vectors.view(), assigned.span(), context).status));
        failures += expect_eq(index.size(), rows);

        std::size_t wrong = 0;
        for (std::size_t row = 0; row != rows; ++row) {
            wrong += index.key_of(assigned[row]).value != keys[row];
            std::size_t own = 0;
            index.find_each(keys[row], [&](node_id_t node) noexcept { own += node == assigned[row]; });
            wrong += own != 1 || index.count(keys[row]) != 2 || !index.contains(keys[row]);
        }
        failures += expect_eq(wrong, 0u);
        failures += expect_eq(index.key_of(static_cast<node_id_t>(rows)).status, status_t::config_mismatch_k);
        failures += expect(!index.contains(dense_key<key_t>(rows)));

        // A pair exports whole, and a narrower export still reports the true total.
        auto exported = nk::matrix<scalar_t>::zeros({2, dimensions}).value;
        failures += expect_eq(index.get(keys[0], rows_span_t {exported.span()}, 2), 2u);
        failures += expect_eq(index.get(keys[0], rows_span_t {exported.span()}, 1), 2u);
        failures += expect_eq(index.get(dense_key<key_t>(rows), rows_span_t {exported.span()}, 2), 0u);

        // A split carries every key beside its row into one of the parts, and the source forgets
        // it.
        std::unique_ptr<index_type_> const first_ptr = make_index(ledger), second_ptr = make_index(ledger);
        std::array<index_type_*, 2> receivers {first_ptr.get(), second_ptr.get()};
        {
            auto [splitting, split_status] = index.plan_split(std::span<index_type_*> {receivers});
            failures += expect(succeeded(split_status));
            if (succeeded(split_status)) {
                do
                    for (auto const step : splitting.steps())
                        failures += expect(succeeded(splitting.run(step, context)));
                while (splitting.advance());
                failures += expect(succeeded(splitting.finish().status));
            }
        }
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index.count(keys[0]), 0u);
        failures += expect_eq(receivers[0]->size() + receivers[1]->size(), rows);
        wrong = 0;
        for (index_type_ const* part : receivers)
            for (node_id_t node = 0; node != part->size(); ++node) {
                key_t const key = part->key_of(node).value;
                std::size_t const row = rows_of_key(key);
                wrong += !ram_index_row_matches(*part, node, vectors, row) &&
                         !ram_index_row_matches(*part, node, vectors, row + 1);
            }
        for (std::size_t row = 0; row != rows; row += 2)
            wrong += receivers[0]->count(keys[row]) + receivers[1]->count(keys[row]) != 2;
        failures += expect_eq(wrong, 0u);

        // A save and a load through the plans keep every key beside its row.
        index_type_& saved = *receivers[0];
        auto const shape = saved.saved_shape();
        auto saved_vectors = nk::matrix<scalar_t>::zeros({shape.nodes, shape.dimensions}).value;
        auto saved_neighbors = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
        auto scratch = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
        failures += expect_eq(saved.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                                         synchronization_t::exclusive_k, scratch.span()),
                              status_t::success_k);
        std::vector<key_t> saved_keys(shape.nodes);
        for (node_id_t node = 0; node != shape.nodes; ++node) saved_keys[node] = saved.key_of(node).value;
        std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
        index_type_& restored = *restored_ptr;
        failures += expect_eq(
            restored.load({dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                          to_span(saved_keys), shape.nodes, {0, static_cast<node_id_t>(shape.nodes)}),
            status_t::success_k);
        failures += expect_eq(restored.size(), shape.nodes);
        wrong = 0;
        for (node_id_t node = 0; node != shape.nodes; ++node) {
            wrong += restored.key_of(node).value != saved_keys[node];
            wrong += restored.count(saved_keys[node]) != saved.count(saved_keys[node]);
            wrong += !ram_index_row_matches(restored, node, vectors, rows_of_key(saved_keys[node])) &&
                     !ram_index_row_matches(restored, node, vectors, rows_of_key(saved_keys[node]) + 1);
        }
        failures += expect_eq(wrong, 0u);
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                        //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,  //
    typename waiter_type_ = fu::standard_yield_t, //
    nk_capability_t caps_ = nk_cap_serial_k,      //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static std::size_t test_ram_index_configuration(char const* name, u64_t) {
    if constexpr (!quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

template <                                        //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,  //
    typename waiter_type_ = fu::standard_yield_t, //
    nk_capability_t caps_ = nk_cap_serial_k,      //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires dense_cell_testable<dtype_, kernel_, caps_>
static std::size_t test_ram_index_configuration(char const* name, u64_t seed) {

    using index_t = ram_index<typename quads_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                              typename tiles_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                              typename elementwise_ladder<caps_>::kit_t,
                              typename rmscast_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                              typename visits_hash_set_ladder<caps_>::kit_t, waiter_type_, atomic_ref_,
                              dummy_cache_hints_t, counting_allocator_t, counting_allocator_t, std::allocator<byte_t>>;

    if (!dense_cell_admitted<caps_, atomic_ref_>()) return report(name, "capabilities missing at runtime");

    using geometry_t = typename index_t::geometry_t;
    auto make_index = [](allocation_ledger_t& ledger,
                         geometry_t geometry = {dense_dimensions_for<typename index_t::scalar_t>(dense_values_k),
                                                connectivity_16_k}) {
        return std::unique_ptr<index_t>(
            new (std::nothrow) index_t(kernel_, geometry, counting_allocator_t(ledger), counting_allocator_t(ledger)));
    };

    std::size_t failures_count = 0;
    failures_count += test_ram_index_keys<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_guards<index_t>(make_index, seed);
    failures_count += test_dense_three_vectors<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_refine_settles<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_duplicate_basin<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_health_audits<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_heal_islands<index_t>(make_index, seed);
    failures_count += test_dense_split<index_t, kernel_>(make_index, seed);
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
    using index_t = ram_index<typename quads_ladder<typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t,
                              typename tiles_ladder<typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t,
                              typename elementwise_ladder<nk_cap_serial_k>::kit_t,
                              typename rmscast_ladder<typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t,
                              typename visits_hash_set_ladder<nk_cap_serial_k>::kit_t, fu::standard_yield_t,
                              fu::standard_atomic_ref, dummy_cache_hints_t, counting_allocator_t, counting_allocator_t,
                              counting_allocator_t>;
    using geometry_t = typename index_t::geometry_t;
    auto make_index = [](allocation_ledger_t& ledger,
                         geometry_t geometry = {dense_dimensions_for<typename index_t::scalar_t>(dense_values_k),
                                                connectivity_16_k}) {
        return std::unique_ptr<index_t>(
            new (std::nothrow) index_t(kernel_, geometry, counting_allocator_t(ledger), counting_allocator_t(ledger)));
    };
    return report(
        name, test_dense_saturated<index_t>(make_index, seed) + test_dense_refine_atomic<index_t>(make_index, seed));
}

} // namespace unum::usearch::test::dense
