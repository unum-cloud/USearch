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

/** Builds and refines rows of 2 KB whatever the dtype, too wide for shared memory to fit a
 *  builder's candidate cache, so every insert, refine and merge keeps its scratch in the warp's
 *  arena slice; the refine must reach every row and the walks must find them, then the ledger must
 *  balance. The rows are sign sketches, since independent values this wide leave every pair of
 *  rows near-equidistant. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_wide_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = 8 * 2048 / scalar_t::bits_per_dimension();
        auto const vectors = make_sketched_vectors<scalar_t>(dense_growth_nodes_k, dimensions,
                                                             stream_key(seed, "wide"));
        failures += expect(index.reserve({dense_growth_nodes_k, connectivity_16_k, dimensions}).status ==
                           status_t::success_k);
        // Apple keeps every builder's scratch in the arena, so only shared-memory devices can miss,
        // even caching no more than a full row and the node's own.
        if constexpr (requires { index.device_specs().shared_memory_per_block_bytes; })
            failures += expect(!index.build_scratch_fits(index.build_scratch_bytes_per_warp(connectivity_16_k + 1)));
        failures += expect(index.insert(vectors.view(), {}, context).status == status_t::success_k);
        failures += expect_eq(index.size(), dense_growth_nodes_k);
        auto const [refined, refined_status] = index.refine({0, static_cast<node_id_t>(index.size())}, context);
        failures += expect(refined_status == status_t::success_k && refined.changed_nodes > 0 &&
                           refined.changed_nodes <= refined.visited_nodes);
        // A dot graph's hubs can stall the heal short of every row.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            failures += expect_eq(refined.health.reachable_nodes, dense_growth_nodes_k);
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        }
    }
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Builds and refines the growth set at an expansion past the widest register candidates of 32
 *  lanes, so both walk the arena frontier's rung: no tail entry drops, the walks still find the
 *  rows, and the ledger must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_vram_broad_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    allocation_ledger_t ledger;
    std::size_t failures = 0;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        auto const vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dimensions, stream_key(seed, "broad"));
        failures += expect(index.reserve({dense_growth_nodes_k, connectivity_16_k, dimensions}).status ==
                           status_t::success_k);
        typename index_type_::insert_config_t broad_insert;
        broad_insert.expansion_factor = 1024;
        auto const [inserted, inserted_status] = index.insert(vectors.view(), {}, context, broad_insert);
        failures += expect(inserted_status == status_t::success_k && inserted.tail_compactions == 0 &&
                           inserted.tail_dropped == 0);
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        typename index_type_::refine_config_t broad_refine;
        broad_refine.expansion_factor = 1024;
        failures += expect(index.refine({0, static_cast<node_id_t>(index.size())}, context, broad_refine).status ==
                           status_t::success_k);
        // The greedy over many tile blocks of a wide pool keeps every row findable.
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
    }
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** A vendor this build leaves out composes nothing. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
std::size_t test_vram_arena_configuration(char const*, u64_t, executor_type_&) {
    return 0;
}

/** Composes the arena rung over the ledgered index of @p executor 's vendor, on its first stream:
 *  rows too wide for a builder's shared-memory cache, then an expansion past the register ladder,
 *  reported under @p name prefixed by the vendor's toolkit. */
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
    return report(vendor_name.c_str(), failures_count);
}

} // namespace unum::usearch::test::dense
