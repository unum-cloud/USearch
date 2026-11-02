/**
 *  @file test/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The arena cell of the VRAM-resident index: builds whose scratch and frontier outgrow the
 *      register ladder and shared memory, composed over each vendor's cell.
 *
 *  The cell compiles once per vendor, instantiating the executor of the one vendor whose
 *  @c USEARCH_ARCH_<VENDOR>_ it carries, host-only through that vendor's compiler mode. A vendor's
 *  test header, free of its SDK like the library's own, specializes @ref vram_cell with its
 *  ledgered index, its sharded collection and their factory over the stream its entry opened, so
 *  the scenarios stay templated on the index alone while the harness's live-bytes-must-balance
 *  ledger audits the GPU flat matrices.
 */
#pragma once

#include <string> // `std::string` - a cell's vendor-prefixed name

#include "dense.cuh"

#if USEARCH_ARCH_CUDA_
#include "usearch/dense/vram_nvidia_index.cuh" // the index's out-of-class members, which the NVIDIA shards compile
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/dense/vram_amd_index.hip" // the index's out-of-class members, which the AMD shards compile
#endif

#include "vram_amd.hpp"
#include "vram_apple.hpp"
#include "vram_nvidia.hpp"

namespace unum::usearch::test::dense {

/** Whether this shard carries @p vendor_ 's GPUs; the others compose nothing. */
template <gpu_vendor_t vendor_>
inline constexpr bool vram_compiled = (vendor_ == gpu_vendor_t::nvidia_k && USEARCH_ARCH_CUDA_) ||
                                      (vendor_ == gpu_vendor_t::amd_k && USEARCH_ARCH_ROCM_) ||
                                      (vendor_ == gpu_vendor_t::apple_k && USEARCH_ARCH_METAL_);

/** Builds and refines vectors of 2 KB whatever the dtype, too wide for shared memory to fit a
 *  builder's candidate cache, so every insert, refine and merge keeps its scratch in the warp's
 *  arena slice; the refine must reach every node and the walks must find them, then the ledger must
 *  balance. The vectors are sign sketches, since independent values this wide leave every pair of
 *  vectors near-equidistant. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_wide_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        std::size_t const dimensions = 8 * 2048 / scalar_t::bits_per_dimension();
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger, {dimensions, connectivity_16_k});
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        auto const vectors = make_sketched_vectors<scalar_t>(dense_growth_nodes_k, dimensions,
                                                             stream_key(seed, "wide"));
        std::vector<key_t> keys(dense_growth_nodes_k);
        for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
        failures += expect(index.reserve({dense_growth_nodes_k}).status == status_t::success_k);
        // Apple keeps every builder's scratch in the arena, so only shared-memory devices can miss,
        // even caching no more than a full neighbor list and the node's own.
        if constexpr (requires { index.device_specs().shared_memory_per_block_bytes; })
            failures += expect(!index.build_scratch_fits(index.build_scratch_bytes_per_warp(connectivity_16_k + 1)));
        failures += expect(index.insert(to_span(keys), vectors.view(), {}, context).status == status_t::success_k);
        failures += expect_eq(index.size(), dense_growth_nodes_k);
        auto const [refined, refined_status] = index.refine({0, static_cast<node_id_t>(index.size())}, {}, {}, context);
        failures += expect(refined_status == status_t::success_k && refined.changed_nodes > 0 &&
                           refined.changed_nodes <= refined.reselected_nodes);
        // A dot graph's hubs can stall the heal short of every node.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            failures += expect_eq(refined.reachability.reachable_nodes, dense_growth_nodes_k);
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        }
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** Builds and refines the growth set at an expansion past the widest register candidates of 32
 *  lanes, so both walk the arena frontier's rung: no tail entry drops, the walks still find the
 *  entries, and the ledger must balance. Nothing reserves, so two batches grow the index on demand,
 *  the first binding the device, and every key counts exactly once, its first node its own. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_broad_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger, {dimensions, connectivity_16_k});
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        auto const vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dimensions, stream_key(seed, "broad"));
        std::vector<key_t> keys(dense_growth_nodes_k);
        for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
        typename index_type_::insert_config_t broad_insert;
        broad_insert.expansion_factor = 1024;
        std::size_t const half = dense_growth_nodes_k / 2;
        failures += expect_eq(
            index.insert({keys.data(), half}, vectors(nk::range(0, half), nk::slice), {}, context, broad_insert).status,
            status_t::success_k);
        auto const [inserted, inserted_status] = index.insert({keys.data() + half, keys.size() - half},
                                                              vectors(nk::range(half, keys.size()), nk::slice), {},
                                                              context, broad_insert);
        failures += expect(inserted_status == status_t::success_k && inserted.tail_compactions == 0 &&
                           inserted.tail_dropped == 0);
        failures += expect_eq(index.size(), dense_growth_nodes_k);
        std::size_t miscounted = 0, mislabeled = 0;
        for (std::size_t row = 0; row != keys.size(); ++row) {
            miscounted += index.count(keys[row]) != 1;
            mislabeled += index.key_of(static_cast<node_id_t>(row)).value != keys[row];
        }
        failures += expect_eq(miscounted, 0u);
        failures += expect_eq(mislabeled, 0u);
        failures += expect(!index.contains(dense_key<key_t>(0)));
        failures += expect_eq(index.key_of(static_cast<node_id_t>(dense_growth_nodes_k)).status,
                              status_t::config_mismatch_k);
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        typename index_type_::refine_config_t broad_refine;
        broad_refine.expansion_factor = 1024;
        failures += expect(index.refine({0, static_cast<node_id_t>(index.size())}, {}, broad_refine, context).status ==
                           status_t::success_k);
        // The greedy over many tile blocks of a wide pool keeps every entry findable.
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** Plants a one-way node in a built graph: the last node keeps the in-links the build gave it but
 *  points only at itself, so the landmarks reach it while it reaches none back. The audit must not
 *  count it, the refine must find it among its islands and heal it, and a second refine must find
 *  nothing left to heal. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_one_way_islands(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        constexpr std::size_t nodes_count = 512;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "one-way"));
        std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
        std::unique_ptr<index_type_> const built_ptr = make_index(ledger, {dimensions, connectivity_16_k});
        index_type_& built = *built_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        failures += expect(built.reserve({nodes_count}).status == status_t::success_k);
        failures += expect(built.insert(to_span(keys), vectors.view(), {}, context).status == status_t::success_k);

        std::size_t const connectivity = static_cast<std::size_t>(built.connectivity());
        auto saved_vectors = nk::matrix<scalar_t>::zeros({nodes_count, dimensions}).value;
        auto saved_neighbors = nk::matrix<node_id_t>::zeros({nodes_count, connectivity}).value;
        failures += expect(!saved_vectors.empty() && !saved_neighbors.empty());
        if (saved_vectors.empty() || saved_neighbors.empty()) return failures;
        failures += expect_eq(
            dense_save(built, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                       synchronization_t::exclusive_k),
            status_t::success_k);
        // No landmark: they sit at multiples of the rows over the connectivity.
        node_id_t const sink = static_cast<node_id_t>(nodes_count - 1);
        auto island_sink_neighbors = saved_neighbors.span().row(sink).as_vector();
        island_sink_neighbors[0] = sink;
        for (std::size_t slot = 1; slot != connectivity; ++slot) island_sink_neighbors[slot] = node_id_unknown_k;

        std::unique_ptr<index_type_> const planted_ptr = make_index(ledger, {dimensions, connectivity_16_k});
        index_type_& planted = *planted_ptr;
        failures += expect(planted.reserve({nodes_count}).status == status_t::success_k);
        failures += expect_eq(
            dense_load(planted, make_index,
                       {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                       to_span(keys), nodes_count, {0, static_cast<node_id_t>(nodes_count)}),
            status_t::success_k);

        auto const [before, before_status] = dense_audit_health(planted, context);
        failures += expect(succeeded(before_status) && before.reachable_nodes < nodes_count && before.back_sweeps >= 1);
        auto const [repair, repair_status] = planted.refine({}, {}, {}, context);
        failures += expect(succeeded(repair_status) && repair.islands_found >= 1);
        // A dot graph's hubs can stall the heal short of every node.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            failures += expect_eq(repair.reachability.reachable_nodes, nodes_count);
            auto const [settled, settled_status] = planted.refine({}, {}, {}, context);
            failures += expect(succeeded(settled_status));
            failures += expect_eq(settled.islands_found, 0u);
        }
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** A vendor this build leaves out composes nothing. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
std::size_t test_vram_arena_configuration(char const*, u64_t, executor_type_&) {
    return 0;
}

/** Composes the arena rung over the ledgered index of @p executor 's vendor, on its first stream:
 *  rows too wide for a builder's shared-memory cache, an expansion past the register ladder, then
 *  a planted one-way node, reported under @p name prefixed by the vendor's toolkit. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
    requires vram_compiled<executor_type_::vendor_k>
std::size_t test_vram_arena_configuration(char const* name, u64_t seed, executor_type_& executor) {
    using cell_t = vram_cell<executor_type_::vendor_k, dtype_, kernel_>;
    using index_t = typename cell_t::index_t;
    cell_t const cell {executor.stream(0)};
    std::string const vendor_name = fmt::format("{}-{}", cell_t::vendor_prefix_k, name);
    std::size_t failures_count = 0;
    failures_count += test_dense_vram_wide_build<index_t, kernel_>(cell, seed);
    failures_count += test_dense_vram_broad_build<index_t, kernel_>(cell, seed);
    failures_count += test_dense_vram_one_way_islands<index_t, kernel_>(cell, seed);
    return report(vendor_name.c_str(), failures_count);
}

} // namespace unum::usearch::test::dense
