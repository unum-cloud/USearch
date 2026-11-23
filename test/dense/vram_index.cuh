/**
 *  @file test/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The arena cell of the VRAM-resident index: builds whose scratch and frontier outgrow the
 *      register ladder and shared memory, composed over each vendor's cell.
 *
 *  The cell compiles once per vendor, instantiating the executor of the one vendor whose
 *  @c USEARCH_ARCH_<VENDOR>_ it defines, host-only through that vendor's compiler mode. A vendor's
 *  test header, free of its SDK like the library's own, specializes @ref vram_cell with its
 *  ledgered index, its sharded collection and their factory over the stream its entry opened, so
 *  the scenarios stay templated on the index alone while the harness's live-bytes-must-balance
 *  ledger audits the GPU flat matrices.
 */
#pragma once

#include <string>      // `std::string` - a cell's vendor-prefixed name
#include <string_view> // `std::string_view` - the phase a round's barrier names

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

/** Whether this shard covers @p vendor_ 's GPUs; the others compose nothing. Each vendor's shard
 *  defines it differently, so it keeps internal linkage. */
template <gpu_vendor_t vendor_>
static constexpr bool vram_compiled = (vendor_ == gpu_vendor_t::nvidia_k && USEARCH_ARCH_CUDA_) ||
                                      (vendor_ == gpu_vendor_t::amd_k && USEARCH_ARCH_ROCM_) ||
                                      (vendor_ == gpu_vendor_t::apple_k && USEARCH_ARCH_METAL_);

/** Builds and refines vectors of 2 KB whatever the dtype, too wide for shared memory to fit a
 *  builder's candidate cache, so every insert, refine and merge keeps its scratch in the warp's
 *  arena slice; the refine must reach every node and the walks must find them, then the ledger must
 *  balance. The vectors are sign sketches, since independent values this wide leave every pair of
 *  vectors near-equidistant. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_vram_wide_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    allocation_ledger_t ledger;
    {
        std::size_t const dimensions = 8 * 2048 / scalar_t::bits_per_dimension();
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger, {dimensions, connectivity_16_k});
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        auto const vectors = make_sketched_vectors<scalar_t>(dense_growth_nodes_k, dimensions,
                                                             stream_key(seed, "wide"));
        std::vector<key_t> keys(dense_growth_nodes_k);
        for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
        expect_true(index.reserve({dense_growth_nodes_k}).status == status_t::success_k);
        // Apple keeps every builder's scratch in the arena, so only shared-memory devices can miss,
        // even caching no more than a full neighbor list and the node's own.
        if constexpr (requires { index.device_specs().shared_memory_per_block_bytes; })
            expect_true(!index.build_scratch_fits(index.build_scratch_bytes_per_warp(connectivity_16_k + 1)));
        expect_true(index.insert(to_span(keys), vectors.view(), {}, context).status == status_t::success_k);
        expect_equal(index.size(), dense_growth_nodes_k);
        auto const [refined, refined_status] = index.refine({}, context);
        expect_true(refined_status == status_t::success_k && refined.changed_nodes > 0 &&
                    refined.changed_nodes <= refined.reselected_nodes);
        // A dot graph's hubs can stall the heal short of every node.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            expect_equal(refined.reachability.reachable_nodes, dense_growth_nodes_k);
            dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        }
    }
    expect_equal(ledger.live_bytes, 0);
}

/** Builds and refines the growth set at an expansion past the widest register candidates of 32
 *  lanes, so both walk the arena frontier's rung: no tail entry drops, the walks still find the
 *  entries, and the ledger must balance. Nothing reserves, so two batches grow the index on demand,
 *  the first binding the device, and every key counts exactly once, its first node its own. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_vram_broad_build(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    allocation_ledger_t ledger;
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
        expect_equal(
            index.insert({keys.data(), half}, vectors(nk::range(0, half), nk::slice), {}, context, broad_insert).status,
            status_t::success_k);
        auto const [inserted, inserted_status] = index.insert({keys.data() + half, keys.size() - half},
                                                              vectors(nk::range(half, keys.size()), nk::slice), {},
                                                              context, broad_insert);
        expect_true(inserted_status == status_t::success_k && inserted.tail_compactions == 0 &&
                    inserted.tail_dropped == 0);
        expect_equal(index.size(), dense_growth_nodes_k);
        std::size_t miscounted = 0, mislabeled = 0;
        for (std::size_t row = 0; row != keys.size(); ++row) {
            miscounted += index.count(keys[row]) != 1;
            mislabeled += index.key_of(static_cast<node_id_t>(row)).value != keys[row];
        }
        expect_equal(miscounted, 0u);
        expect_equal(mislabeled, 0u);
        expect_true(!index.contains(dense_key<key_t>(0)));
        expect_equal(index.key_of(static_cast<node_id_t>(dense_growth_nodes_k)).status, status_t::config_mismatch_k);
        if constexpr (kernel_ != nk_kernel_dot_k)
            dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
        expect_true(index.refine({}, context).status == status_t::success_k);
        // A round over walks of the arena frontier's rung keeps every entry findable.
        if constexpr (kernel_ != nk_kernel_dot_k)
            dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
    }
    expect_equal(ledger.live_bytes, 0);
}

/** Plants a one-way node in a built graph: the last node keeps the in-links the build gave it but
 *  points only at itself, so the root reaches it while it reaches nothing back. The health check
 *  must not count it, the round must find it among its islands and heal it, and a second round
 *  must find nothing left to heal. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_vram_one_way_islands(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    allocation_ledger_t ledger;
    {
        constexpr std::size_t nodes_count = 512;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "one-way"));
        std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
        std::unique_ptr<index_type_> const built_ptr = make_index(ledger, {dimensions, connectivity_16_k});
        index_type_& built = *built_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        expect_true(built.reserve({nodes_count}).status == status_t::success_k);
        expect_true(built.insert(to_span(keys), vectors.view(), {}, context).status == status_t::success_k);

        std::size_t const connectivity = static_cast<std::size_t>(built.connectivity());
        auto saved_vectors = nk::matrix<scalar_t>::zeros({nodes_count, dimensions}).value;
        auto saved_neighbors = nk::matrix<node_id_t>::zeros({nodes_count, connectivity}).value;
        expect_true(!saved_vectors.empty() && !saved_neighbors.empty());
        if (saved_vectors.empty() || saved_neighbors.empty()) return;
        expect_equal(
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
        expect_true(planted.reserve({nodes_count}).status == status_t::success_k);
        expect_equal(dense_load(planted, make_index,
                                {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                                to_span(keys), nodes_count, {0, static_cast<node_id_t>(nodes_count)}),
                     status_t::success_k);

        auto const [before, before_status] = dense_audit_health(planted, context);
        expect_true(succeeded(before_status) && before.reachable_nodes < nodes_count && before.depth_to_root >= 1);
        auto const [repair, repair_status] = planted.refine({}, context);
        expect_true(succeeded(repair_status) && repair.islands_found >= 1);
        // A dot graph's hubs can stall the heal short of every node.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            expect_equal(repair.reachability.reachable_nodes, nodes_count);
            auto const [settled, settled_status] = planted.refine({}, context);
            expect_true(succeeded(settled_status));
            expect_equal(settled.islands_found, 0u);
        }
    }
    expect_equal(ledger.live_bytes, 0);
}

/** One round over @p index under @p config on @p context 's stream, @p at_barrier hearing the plan
 *  past every barrier that leaves another phase to run. */
template <typename index_type_, typename at_barrier_type_>
expected<refine_report_t> dense_vram_round(index_type_& index, typename index_type_::refine_config_t config,
                                           typename index_type_::traversal_context_t& context,
                                           at_barrier_type_&& at_barrier) {
    auto [plan, status] = index.plan_refine(config, context);
    if (failed(status)) return {{}, status};
    while (succeeded(plan.queue(context, config)) && succeeded(plan.wait(context)) && plan.advance()) at_barrier(plan);
    return plan.finish(context);
}

/** A VRAM index whose rows hold @p links, one list of targets per entry, its vectors drawn from
 *  @p seed: a graph written by hand rather than built. */
template <typename index_type_, typename make_index_type_>
std::unique_ptr<index_type_> dense_vram_index_of_links(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                                       std::vector<std::vector<node_id_t>> const& links, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const entries = links.size(), dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> index = make_index(ledger, {dimensions, connectivity_16_k});
    std::size_t const connectivity = static_cast<std::size_t>(index->connectivity());
    auto vectors = make_vectors<scalar_t>(entries, dimensions, stream_key(seed, "round_links"));
    auto neighbors = nk::matrix<node_id_t>::zeros({entries, connectivity}).value;
    for (std::size_t entry = 0; entry != entries; ++entry) {
        auto row = neighbors.span().row(entry).as_vector();
        for (std::size_t slot = 0; slot != connectivity; ++slot)
            row[slot] = slot < links[entry].size() ? links[entry][slot] : node_id_unknown_k;
        if (links[entry].empty()) row[0] = static_cast<node_id_t>(entry);
    }
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(entries);
    expect_true(index->reserve({entries}).status == status_t::success_k);
    expect_equal(
        dense_load(*index, make_index, {dense_matrix_bytes(vectors.view()), dense_matrix_bytes(neighbors.view())},
                   to_span(keys), entries, {0, static_cast<node_id_t>(entries)}),
        status_t::success_k);
    return index;
}

/** The round's contract on a device, as the RAM round keeps it, one behaviour a scenario: planted
 *  islands end at zero with no distance spent healing them, a round stopped in any phase leaves
 *  searches no worse, a cache of a few rows gathers the wide neighborhoods again until each fits
 *  its capacity, identical entries over capacity are cut by ID, and a graph refined until converged
 *  changes no link set on the next round. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_vram_rounds(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    using refine_config_t = typename index_type_::refine_config_t;
    using links_t = std::vector<std::vector<node_id_t>>;
    allocation_ledger_t ledger;
    {
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        auto ignore_barriers = [](auto&) noexcept {};

        // An entry linking into a cycle that nothing links to, and a pair the cycle links into that
        // links only to itself: the heal links both through the root's component with no distance.
        {
            constexpr std::size_t entries = 200, cycled = entries - 3;
            links_t links(entries);
            for (std::size_t entry = 0; entry != cycled; ++entry) {
                std::size_t const next = (entry + 1) % cycled, jump = (entry * 7 + 3) % cycled;
                links[entry].push_back(static_cast<node_id_t>(next));
                if (jump != entry && jump != next) links[entry].push_back(static_cast<node_id_t>(jump));
            }
            links[cycled] = {5, 50};
            links[cycled + 1] = {cycled + 2};
            links[cycled + 2] = {cycled + 1};
            links[20].push_back(cycled + 1);
            std::unique_ptr<index_type_> const index = dense_vram_index_of_links<index_type_>(make_index, ledger, links,
                                                                                              seed);
            std::size_t healed_from = 0, healing = 0;
            bool in_heal = false;
            auto const [report, status] = dense_vram_round(*index, {}, context, [&](auto const& plan) noexcept {
                std::size_t const scored = context.spent().scored;
                if (in_heal) healing += scored - healed_from;
                in_heal = plan.phase() == phase_t::heal_k;
                healed_from = scored;
            });
            expect_true(succeeded(status));
            expect_equal(report.opening.unreached_islands, 1u);
            expect_equal(report.opening.stranded_islands, 1u);
            expect_equal(report.reachability.unreached_islands, 0u);
            expect_equal(report.reachability.stranded_islands, 0u);
            expect_equal(report.reachability.reachable_nodes, entries);
            expect_equal(healing, 0u);
        }

        std::size_t const built = 600;
        auto vectors = make_vectors<scalar_t>(built, dimensions, stream_key(seed, "rounds"));
        std::vector<key_t> const keys = dense_keys<index_type_>(built);
        auto saved_vectors = nk::matrix<scalar_t>::zeros({built, dimensions}).value;
        auto saved_neighbors = nk::matrix<node_id_t>::zeros({built, std::size_t {connectivity_16_k}}).value;
        // Draws of a capacity's worth of hubs and a capacity of one row of mates and the entry, the
        // floor, so neighborhoods gather again.
        std::size_t const narrow_capacity = connectivity_16_k + 1;
        refine_config_t narrow;
        {
            std::unique_ptr<index_type_> const index = make_index(ledger, {dimensions, connectivity_16_k});
            std::size_t const staged_row = index->row_bytes() +
                                           (kernel_ == nk_kernel_dot_k ? 0 : sizeof(typename index_type_::norms_t));
            narrow.cache_bytes = 2 * narrow_capacity * staged_row;
            expect_true(succeeded(index->reserve({built}).status));
            expect_true(succeeded(index->insert(to_span(keys), vectors.view(), {}, context).status));
            expect_equal(
                dense_save(*index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                           synchronization_t::exclusive_k),
                status_t::success_k);
        }
        auto load = [&]() {
            std::unique_ptr<index_type_> index = make_index(ledger, {dimensions, connectivity_16_k});
            expect_true(succeeded(index->reserve({built}).status));
            expect_equal(
                dense_load(*index, make_index,
                           {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                           to_span(keys), built, {0, static_cast<node_id_t>(built)}),
                status_t::success_k);
            return index;
        };
        auto queries = vectors(nk::range(0, 64), nk::slice);

        // Stopped at the barrier opening each phase, from the opening health check to the restitch,
        // the round keeps what ran and no search does worse for it.
        if constexpr (kernel_ != nk_kernel_dot_k)
            for (std::size_t phase = 1; phase != 6; ++phase) {
                std::unique_ptr<index_type_> const index = load();
                std::size_t const before = dense_recall_at_ten(*index, make_index, queries);
                bool stopped = false;
                auto const [report, status] = dense_vram_round(*index, narrow, context, [&](auto& plan) noexcept {
                    if (stopped || plan.ordinal() != phase) return;
                    plan.stop();
                    stopped = true;
                });
                expect_true(stopped);
                expect_equal(status, status_t::cancelled_k);
                expect_true(report.verdict == refine_verdict_t::stopped_k);
                expect_true(dense_recall_at_ten(*index, make_index, queries) >= before);
            }

        // An ended round answers false at every later barrier and reports alike when finished
        // twice; the next round then runs from its opening check to a verdict of its own.
        {
            std::unique_ptr<index_type_> const index = load();
            auto [plan, plan_status] = index->plan_refine(narrow, context);
            expect_true(succeeded(plan_status));
            while (succeeded(plan.queue(context, narrow)) && succeeded(plan.wait(context)) && plan.advance()) {}
            expect_true(!plan.advance());
            auto const first = plan.finish(context), again = plan.finish(context);
            expect_true(succeeded(first.status) && succeeded(again.status));
            expect_equal(first.value.changed_nodes, again.value.changed_nodes);
            plan.next_round();
            expect_true(plan.phase() == phase_t::health_k);
            while (succeeded(plan.queue(context, narrow)) && succeeded(plan.wait(context)) && plan.advance()) {}
            auto const [next, next_status] = plan.finish(context);
            expect_true(succeeded(next_status));
            expect_true(next.verdict != refine_verdict_t::none_k);
        }

        // A cache of a few rows puts the first neighborhoods over capacity, which gather again
        // among their own entries until none is.
        {
            std::unique_ptr<index_type_> const index = load();
            std::size_t gathers = 0;
            auto const [report, status] = dense_vram_round(*index, narrow, context, [&](auto const& plan) noexcept {
                gathers += plan.phase() == phase_t::gather_k;
            });
            expect_true(succeeded(status));
            expect_true(gathers >= 2);
            expect_equal(report.reselected_nodes, built);
            expect_true(report.neighborhoods > 1 && report.widest_neighborhood <= narrow_capacity);
            expect_true(report.reachability.reachable_nodes >= report.opening.reachable_nodes);
        }

        // Identical entries all lie nearest to one hub, so no gather splits them; they are cut into
        // pieces by ID instead, and the round ends.
        {
            constexpr std::size_t entries = 10 * (connectivity_16_k + 1);
            auto copies = nk::matrix<scalar_t>::zeros({entries, dimensions}).value;
            for (std::size_t entry = 0; entry != entries; ++entry) {
                auto const source = vectors.view().row(0).as_vector();
                auto target = copies.row(entry).as_vector();
                for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
                    target[dimension] = source[dimension];
            }
            std::unique_ptr<index_type_> const index = make_index(ledger, {dimensions, connectivity_16_k});
            expect_true(succeeded(index->reserve({entries}).status));
            expect_true(succeeded(index->insert(std::span(keys).first(entries), copies.view(), {}, context).status));
            auto const [report, status] = dense_vram_round(*index, narrow, context, ignore_barriers);
            expect_true(succeeded(status));
            expect_equal(report.reselected_nodes, entries);
            expect_true(report.widest_neighborhood <= narrow_capacity);
            expect_true(report.reachability.reachable_nodes >= report.opening.reachable_nodes);
        }

        // Rounds until one converges, then the next changes no link set and keeps every entry.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            std::unique_ptr<index_type_> const index = load();
            refine_report_t report;
            for (std::size_t round = 0; round != 16 && report.verdict != refine_verdict_t::converged_k; ++round)
                report = dense_vram_round(*index, {}, context, ignore_barriers).value;
            expect_true(report.verdict == refine_verdict_t::converged_k);
            auto const [again, again_status] = dense_vram_round(*index, {}, context, ignore_barriers);
            expect_true(succeeded(again_status));
            expect_true(again.verdict == refine_verdict_t::converged_k && again.changed_nodes == 0);
            expect_equal(again.reachability.reachable_nodes, report.reachability.reachable_nodes);
        }
    }
    expect_equal(ledger.live_bytes, 0);
}

/** A vendor this build leaves out composes nothing. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
void test_vram_arena_configuration(u64_t, executor_type_&) {}

/** Composes the arena rung over the ledgered index of @p executor 's vendor, on its first stream:
 *  rows too wide for a builder's shared-memory cache, an expansion past the register ladder, a
 *  planted one-way node, then the round's contract. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
    requires vram_compiled<executor_type_::vendor_k>
void test_vram_arena_configuration(u64_t seed, executor_type_& executor) {
    using cell_t = vram_cell<executor_type_::vendor_k, dtype_, kernel_>;
    using index_t = typename cell_t::index_t;
    cell_t const cell {executor.stream(0)};
    test_dense_vram_wide_build<index_t, kernel_>(cell, seed);
    test_dense_vram_broad_build<index_t, kernel_>(cell, seed);
    test_dense_vram_one_way_islands<index_t, kernel_>(cell, seed);
    // Apple's round sizes its neighborhoods by threadgroup memory, on its own tower.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        test_dense_vram_rounds<index_t, kernel_>(cell, seed);
}

} // namespace unum::usearch::test::dense
