/**
 *  @brief The index-agnostic test scenario library shared by every dense-index residency -
 *      volatile in-RAM, GPU-resident, and persisted alike.
 *
 *  @author Ash Vardanian
 *  @file test/dense.cuh
 *  @date July 22, 2026
 *
 *  Every scenario is a template over the index type, and each translation unit composes its
 *  own suite explicitly: the calls a TU makes ARE its capability claims, so a backend that
 *  cannot race simply never calls the racing scenarios, and the omission is auditable at
 *  the call site instead of hiding in a trait. Translation units name their concrete index
 *  instantiations and pass a ledgered factory constructing them - this header names no
 *  index and assumes no constructor shape, so the same scenarios compile under host
 *  compilers and nvcc.
 *
 *  Allocators are tested by construction: a counting allocator flows through the vectors &
 *  neighbors parameters to prove propagation, tallying live bytes into a ledger that must
 *  balance to zero when each scenario's scope closes. Parallel scenarios use ForkUnion's
 *  default flat pool - no NUMA machinery in tests, except the sharded-index scenarios, which
 *  need a real topology because an executor sites its arenas by asking where its workers run.
 */
#pragma once

#include <algorithm> // `std::sort`, `std::unique`
#include <array>     // `std::array`
#include <atomic>    // `std::atomic`
#include <bit>       // `std::bit_cast` - the ULP distance of the oracle audits
#include <chrono>    // `std::chrono::microseconds`
#include <cstdlib>   // `std::abs`
#include <thread>    // `std::this_thread::sleep_for`
#include <vector>    // `std::vector`

#include <fmt/format.h>
#include <forkunion.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>

#include "test.hpp"
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/sanitizers.hpp" // `assert_no_locks_held` - the racing lanes' tail check
#include "usearch/dense/ram_metric.hpp"     // `usearch::distance`, `usearch::dense_norms` - the oracle's kernels

namespace unum {
namespace usearch {

#pragma region Constants and Seeds

/** @brief Storage values per vector in the larger scenarios - NumKong's unit of one
 *      scalar object; small enough to keep serial kernels fast. */
constexpr std::size_t dense_values_k = 16;
/** @brief Storage values per vector in the tiny smoke scenarios. */
constexpr std::size_t dense_tiny_values_k = 8;
/** @brief Logical dimensions spanned by @p values_count storage values of one dtype -
 *      NumKong's @c dimensions_per_value conversion, so every dtype gets the same storage
 *      budget: 16 i8 values carry 16 dimensions, 16 u1x8 values carry 128. */
template <typename scalar_>
constexpr std::size_t dense_dimensions_for(std::size_t values_count) noexcept {
    return values_count * nk::dimensions_per_value<scalar_>();
}
/** @brief Node count marching the growth scenario across several doubling chunks. */
constexpr std::size_t dense_growth_nodes_k = 1'500;
/** @brief Nodes inserted by each lane of the concurrent scenarios. */
constexpr std::size_t dense_lane_nodes_k = 256;
/** @brief Self-recall sampling threshold: at least this share of samples must find themselves. */
constexpr std::size_t dense_minimal_recall_percent_k = 90;
/** @brief Distance bound for a stored vector matching itself under symmetric kernels. */
constexpr float dense_self_match_tolerance_k = 1e-3f;
/** @brief The reader lane performs at least this many searches, even if writers finish first. */
constexpr std::size_t dense_reader_minimal_searches_k = 16;

/** @brief Data seeds, one named hex constant per scenario family; derived draws follow
 *      the salt rule documented beside @c salt_queries_k in @c test.hpp. */
constexpr u64_t dense_seed_guards_k = 0x6A24D5D0ull;
constexpr u64_t dense_seed_three_vectors_k = 0x3FEC0003ull;
constexpr u64_t dense_seed_growth_k = 0x62047F00ull;
constexpr u64_t dense_seed_predicated_k = 0xF117E2EDull;
constexpr u64_t dense_seed_roundtrip_k = 0x5AFE10ADull;
constexpr u64_t dense_seed_load_hostile_k = 0x10ADBADDull;
constexpr u64_t dense_seed_concurrent_build_k = 0xB111DACEull;
constexpr u64_t dense_seed_concurrent_search_k = 0x5EA2CACEull;
constexpr u64_t dense_seed_indexes_add_k = 0x62009ADDull;
constexpr u64_t dense_seed_indexes_race_k = 0x62009ACEull;
constexpr u64_t dense_seed_save_race_k = 0x5AFECACEull;
constexpr u64_t dense_seed_immutable_parity_k = 0x5E771ED0ull;
constexpr u64_t dense_seed_cursor_k = 0x17E2A702ull;
constexpr u64_t dense_seed_allocation_budget_k = 0xB0D6E7EDull;
constexpr u64_t dense_seed_overflow_race_k = 0xF100DED0ull;

#pragma endregion Constants and Seeds

#pragma region Shared Helpers

/** @brief Per-lane state for the concurrent scenarios: one vector slice & one traversal context each. */
template <typename index_type_>
struct dense_lanes_t {
    std::vector<nk::matrix<typename index_type_::scalar_t>> slices;
    std::vector<typename index_type_::dense_traversal_context_t> contexts;
};

/** @brief Prepares @p lanes_count lanes with distinctly-seeded vector slices & fresh contexts. */
template <typename index_type_>
dense_lanes_t<index_type_> dense_make_lanes(std::size_t lanes_count, std::size_t nodes_per_lane, std::size_t dimensions,
                                            u64_t seed) {
    using context_t = typename index_type_::dense_traversal_context_t;
    dense_lanes_t<index_type_> lanes;
    for (std::size_t lane = 0; lane != lanes_count; ++lane) {
        lanes.slices.push_back(make_vectors<typename index_type_::scalar_t>(nodes_per_lane, dimensions, seed + lane));
        lanes.contexts.emplace_back(typename context_t::allocator_t {});
    }
    return lanes;
}

/** @brief Bytes of the first @p rows_count rows of a row-major matrix view - the stride
 *      already accounts for sub-byte packing. */
template <typename matrix_view_type_>
constexpr std::size_t dense_rows_bytes(matrix_view_type_ view, std::size_t rows_count) noexcept {
    return rows_count * static_cast<std::size_t>(view.stride_bytes(0));
}

/** @brief Sampled self-recall over the whole result pool: each sampled stored row must
 *      surface some pool entry inside the self-match band. Pool-scan by distance is the
 *      one criterion: duplicate ties - binary vectors above all - make the top-1 slot
 *      and id-matching unfair judges of a healthy graph, and a distance criterion needs
 *      no assigned-ids plumbing, so it survives reloads & lane-sliced builds unchanged.
 *      Sampling is evenly spaced; callers keep the dot-kernel gate, where "distance" is
 *      a similarity and self-matching is undefined. Requires @p rows_count ≥ @p samples_count. */
template <typename index_type_, typename vectors_view_type_>
std::size_t dense_expect_self_recall(index_type_& index, vectors_view_type_ vectors, std::size_t rows_count,
                                     std::size_t samples_count) {
    std::size_t failures = 0, hits = 0;
    typename index_type_::dense_traversal_context_t context {{}};
    for (std::size_t sample = 0; sample != samples_count; ++sample) {
        std::size_t const row = sample * (rows_count / samples_count);
        auto const found = index.search(vectors.row(row).as_vector(), context);
        failures += expect(found.status == status_t::success_k && found.count > 0);
        for (std::size_t match = 0; match != found.count; ++match) {
            dense_scored_node_t const scored = context.results.data()[match];
            if (scored.distance < dense_self_match_tolerance_k && scored.distance > -dense_self_match_tolerance_k) {
                ++hits;
                break;
            }
        }
    }
    failures += expect(hits * 100 >= samples_count * dense_minimal_recall_percent_k);
    return failures;
}

#pragma endregion Shared Helpers

#pragma region Foundations

/** @brief Guards: pre-reserve misuse and geometry freezing, on a fresh index per scenario. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_guards(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(3, dense_dimensions_for<scalar_t>(dense_tiny_values_k),
                                              dense_seed_guards_k);

        auto inserted = index.insert(vectors.view(), {}, context);
        failures += expect(inserted.status == status_t::uninitialized_k);
        auto found = index.search(vectors.view().row(0).as_vector(), context);
        failures += expect(found.status == status_t::success_k && found.count == 0);

        auto reserved = index.reserve({64, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
        failures += expect(reserved.status == status_t::success_k && reserved.new_capacity >= 64);
        auto mismatched = index.reserve({64, connectivity_32_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
        failures += expect(mismatched.status == status_t::config_mismatch_k);
        auto matching = index.reserve({32, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
        failures += expect(matching.status == status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The classic smoke scenario: three vectors in, each finds itself first. The dot kernel
 *      maximizes similarity instead of minimizing distance, so self-nearest is only asserted
 *      for the symmetric kernels. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_three_vectors(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(3, dense_dimensions_for<scalar_t>(dense_tiny_values_k),
                                              dense_seed_three_vectors_k);

        failures += expect(
            index.reserve({8, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)}).status ==
            status_t::success_k);
        std::array<dense_node_id_t, 3> assigned_ids;
        auto inserted = index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context);
        failures += expect(inserted.status == status_t::success_k);
        failures += expect_eq(inserted.new_size, 3u);
        failures += expect_eq(index.size(), 3u);

        for (std::size_t row = 0; row != 3; ++row) {
            auto found = index.search(vectors.view().row(row).as_vector(), context);
            failures += expect(found.status == status_t::success_k && found.count >= 1);
            if constexpr (kernel_ != nk_kernel_dot_k) {
                dense_scored_node_t const nearest = context.results.data()[0];
                failures += expect_eq(nearest.node_id, assigned_ids[row]);
                failures += expect(nearest.distance < dense_self_match_tolerance_k && //
                                   nearest.distance > -dense_self_match_tolerance_k);
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Growth across several doubling chunks, then sampled self-recall on the big graph,
 *      with telemetry accounting sanity checks along the way. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_growth_and_recall(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                              dense_seed_growth_k);

        failures += expect(
            index.reserve({16, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
            status_t::success_k);
        std::vector<dense_node_id_t> assigned_ids(dense_growth_nodes_k);
        auto inserted = index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context);
        failures += expect(inserted.status == status_t::success_k);
        failures += expect_eq(inserted.new_size, dense_growth_nodes_k);
        failures += expect_eq(index.size(), dense_growth_nodes_k);
        failures += expect(inserted.computed_distances > 0 && inserted.visited_members > 0);
        failures += expect(inserted.computed_distances_in_refines <= inserted.computed_distances);
        failures += expect(inserted.computed_distances_in_reverse_refines <= inserted.computed_distances);
        // The context was fresh, so the batch delta must equal the absolute counter - the
        // belt against a dropped snapshot or accumulation in the restarts telemetry.
        failures += expect_eq(inserted.restarts, context.restarts);

        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(index, vectors.view(), dense_growth_nodes_k, 32);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/**
 *  @brief The predicated search contract, audited against an exact oracle.
 *
 *  The oracle scans every stored vector through the same free kernels the index picked -
 *  @c picked_quads_capability_k - so its distances are bit-identical to the traversal's and
 *  the checks can be exact where approximation permits: a trivially-true predicate must
 *  reproduce @c search verbatim; every reported id must satisfy its predicate and carry the
 *  oracle's distance for that id, bit-for-bit; the per-node and batched callable shapes must
 *  agree; excluded self-matches must never surface; recall of the true filtered nearest is
 *  held to the same bar as the unfiltered suite across three selectivity decades - every
 *  other node, one in 16, and one in 64; and an impossible predicate drains legally.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_predicated_search(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr nk_capability_t caps_k = index_type_::picked_quads_capability_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                              dense_seed_predicated_k);

        failures += expect(
            index.reserve({dense_growth_nodes_k, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)})
                .status == status_t::success_k);
        std::vector<dense_node_id_t> assigned_ids(dense_growth_nodes_k);
        failures += expect(index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context).status ==
                           status_t::success_k);

        // The oracle: distances of one query against the whole store, through the very 4-way
        // kernels the traversal dispatches. A wrong id or vector diverges by orders of
        // magnitude, which is what this audit exists to catch.
        auto oracle_distance = [&](std::size_t query_row, dense_node_id_t node_id) noexcept {
            auto query_view = vectors.view().row(query_row).as_vector();
            auto node_view = vectors.view().row(node_id).as_vector();
            auto node_norms = usearch::dense_norms<kernel_, caps_k, scalar_t>(node_view);
            auto lane_distances = usearch::distances<kernel_, caps_k, scalar_t>(
                query_view, usearch::dense_norms<kernel_, caps_k, scalar_t>(query_view),
                {node_view, node_view, node_view, node_view}, {node_norms, node_norms, node_norms, node_norms});
            return lane_distances[0];
        };
        // The stochastic landmark seeding rides the context's persistent generator, so exact
        // walk parity is only defined between FRESH contexts with identical generator state.
        std::size_t const query_row = dense_growth_nodes_k / 2;
        auto query = vectors.view().row(query_row).as_vector();
        typename index_type_::dense_traversal_context_t unfiltered_context {{}};
        auto unfiltered = index.search(query, unfiltered_context);
        failures += expect(unfiltered.status == status_t::success_k && unfiltered.count > 0);
        dense_scored_node_t const* reference = unfiltered_context.results.data();

        // The trivially-true predicate walks and reports exactly like the unfiltered sibling.
        auto all_pass = [](dense_node_id_t) noexcept { return admission_t::admit_k; };
        typename index_type_::dense_traversal_context_t all_pass_context {{}};
        auto with_true = index.search_if(query, all_pass, all_pass_context);
        failures += expect_eq(with_true.count, unfiltered.count);
        for (std::size_t match = 0; match != with_true.count && match != unfiltered.count; ++match) {
            failures += expect_eq(all_pass_context.results.data()[match].node_id, reference[match].node_id);
            failures += expect(all_pass_context.results.data()[match].distance == reference[match].distance);
        }

        // The batched shape must agree with the per-node shape on the same walk.
        auto even_only = [](dense_node_id_t node_id) noexcept {
            return node_id % 2 == 0 ? admission_t::admit_k : admission_t::reject_k;
        };
        auto even_batched = [](std::span<dense_scored_node_t const> scored, auto& admitted) noexcept {
            for (dense_scored_node_t const candidate : scored)
                if (candidate.node_id % 2 == 0) {
                    [[maybe_unused]] fu::status_t const kept = admitted.push_back(candidate);
                }
        };
        typename index_type_::dense_traversal_context_t scalar_context {{}}, batched_context {{}};
        auto with_scalar = index.search_if(query, even_only, scalar_context);
        auto with_batched = index.search_if(query, even_batched, batched_context);
        failures += expect(with_scalar.status == status_t::success_k && with_scalar.count > 0);
        failures += expect_eq(with_batched.count, with_scalar.count);
        for (std::size_t match = 0; match != with_batched.count && match != with_scalar.count; ++match)
            failures += expect_eq(batched_context.results.data()[match].node_id,
                                  scalar_context.results.data()[match].node_id);

        // Three selectivity decades over sampled queries: every reported id satisfies its
        // predicate and carries the oracle's distance bit-for-bit; distances ascend; a query
        // whose own id is excluded never surfaces itself. Even samples query a matching row -
        // the filtered analog of the suite's self-recall bar, scored per decade on the two
        // denser ones; odd samples query an excluded row, auditing admission purity. The
        // sparsest decade is exactness-audited only: recall there belongs to the future
        // cardinality-driven brute-force crossover, not the graph walk.
        std::size_t const moduli[] = {2, 16, 64};
        std::size_t hits = 0, recall_samples = 0;
        for (std::size_t modulus : moduli) {
            auto matches = [modulus](dense_node_id_t node_id) noexcept {
                return node_id % modulus == 0 ? admission_t::admit_k : admission_t::reject_k;
            };
            for (std::size_t sample = 0; sample != 8; ++sample) {
                std::size_t sampled_row = ((sample * 641 + modulus) % dense_growth_nodes_k) / modulus * modulus;
                if (sample % 2) sampled_row += 1; // An excluded neighbor of a matching row.
                typename index_type_::dense_traversal_context_t sampled_context {{}};
                auto found = index.search_if(vectors.view().row(sampled_row).as_vector(), matches, sampled_context);
                bool const self_matches = sampled_row % modulus == 0;
                // The sparsest decade may legally drain empty on excluded-row queries: until
                // the admitted pool fills, the navigation radius bounds the walk, and no match
                // needs to fall inside it. A matching self is always near enough to admit.
                failures += expect(found.status == status_t::success_k);
                failures += expect(found.count > 0 || (modulus == 64 && !self_matches));
                dense_scored_node_t const* results = sampled_context.results.data();
                bool self_found = false;
                for (std::size_t match = 0; match != found.count; ++match) {
                    failures += expect(results[match].node_id % modulus == 0);
                    dense_distance_t const oracle = oracle_distance(sampled_row, results[match].node_id);
                    failures += expect_near(results[match].distance, oracle);
                    if (match) failures += expect(results[match].distance >= results[match - 1].distance);
                    if (!self_matches) failures += expect(results[match].node_id != assigned_ids[sampled_row]);
                    self_found |= results[match].node_id == assigned_ids[sampled_row];
                }
                if constexpr (kernel_ != nk_kernel_dot_k)
                    if (self_matches && modulus != 64) hits += self_found, ++recall_samples;
            }
        }
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += expect(hits * 100 >= recall_samples * dense_minimal_recall_percent_k);

        // An impossible predicate drains legally: empty result set, successful status.
        auto none_pass = [](dense_node_id_t) noexcept { return admission_t::reject_k; };
        typename index_type_::dense_traversal_context_t none_context {{}};
        auto with_none = index.search_if(query, none_pass, none_context);
        failures += expect(with_none.status == status_t::success_k);
        failures += expect_eq(with_none.count, std::size_t {0});
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Foundations

#pragma region Snapshots

/** @brief Round-trips the graph through the two-matrix serialization: build, save, load into
 *      a fresh reserved index, and expect the restored graph to answer self-queries just as
 *      the original did - norms re-derive on load and are never serialized. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_save_load_roundtrip(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                              dense_seed_roundtrip_k);

        failures += expect(
            index.reserve({dense_growth_nodes_k, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)})
                .status == status_t::success_k);
        std::vector<dense_node_id_t> assigned_ids(dense_growth_nodes_k);
        failures += expect(index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context).status ==
                           status_t::success_k);

        auto const shape = index.saved_shape();
        auto saved_vectors = nk::matrix<scalar_t>::try_zeros({shape.nodes, shape.dimensions});
        auto saved_neighbors = nk::matrix<node_id_t>::try_zeros({shape.nodes, shape.connectivity});
        failures += expect(!saved_vectors.empty() && !saved_neighbors.empty());
        failures += expect(index.save(saved_vectors.span(), saved_neighbors.span(), synchronization_t::exclusive_k) ==
                           status_t::success_k);

        index_type_ restored = make_index(ledger);
        failures += expect(
            restored.reserve({shape.nodes, static_cast<connectivity_t>(shape.connectivity), shape.dimensions}).status ==
            status_t::success_k);
        failures += expect(restored.load(saved_vectors.view(), saved_neighbors.view(),
                                         synchronization_t::exclusive_k) == status_t::success_k);
        failures += expect_eq(restored.size(), shape.nodes);

        // A restored index must re-save on every contract: the settlement counters arrive
        // fully credited, so the optimistic arm's probe sees a quiescent chain, and the
        // re-saved bytes mirror the originals exactly.
        auto resaved_vectors = nk::matrix<scalar_t>::try_zeros({shape.nodes, shape.dimensions});
        auto resaved_neighbors = nk::matrix<node_id_t>::try_zeros({shape.nodes, shape.connectivity});
        failures += expect(!resaved_vectors.empty() && !resaved_neighbors.empty());
        failures += expect(restored.save(resaved_vectors.span(), resaved_neighbors.span(),
                                         synchronization_t::exclusive_k) == status_t::success_k);
        failures += expect(restored.save(resaved_vectors.span(), resaved_neighbors.span(),
                                         synchronization_t::immutable_k) == status_t::success_k);
        failures += expect(restored.save(resaved_vectors.span(), resaved_neighbors.span(),
                                         synchronization_t::concurrent_k) == status_t::success_k);
        failures += expect(std::memcmp(resaved_vectors.data(), saved_vectors.data(),
                                       dense_rows_bytes(saved_vectors.view(), shape.nodes)) == 0);
        failures += expect(std::memcmp(resaved_neighbors.data(), saved_neighbors.data(),
                                       dense_rows_bytes(saved_neighbors.view(), shape.nodes)) == 0);

        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(restored, vectors.view(), dense_growth_nodes_k, 32);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Hostile & mismatched load submissions: every pre-copy refusal answers
 *      @c config_mismatch_k and leaves the target loadable with the intact payload; a
 *      count past the reservation grows by design; and an out-of-range neighbor id rides
 *      the sentinel convention - stored, but never surfaced by any search. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_load_hostile(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const nodes_count = 300;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, dense_seed_load_hostile_k);
        failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
        failures += expect(index.insert(vectors.view(), {}, context).status == status_t::success_k);

        auto saved_vectors = nk::matrix<scalar_t>::try_zeros({nodes_count, dimensions});
        auto saved_neighbors = nk::matrix<node_id_t>::try_zeros({nodes_count, connectivity_16_k});
        failures += expect(!saved_vectors.empty() && !saved_neighbors.empty());
        failures += expect(index.save(saved_vectors.span(), saved_neighbors.span(), synchronization_t::exclusive_k) ==
                           status_t::success_k);

        // Pre-copy refusals, all against one reserved empty target that stays loadable.
        index_type_ target = make_index(ledger);
        failures += expect(target.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
        failures += expect(target.load(saved_vectors.view(), saved_neighbors.view(), synchronization_t::concurrent_k) ==
                           status_t::config_mismatch_k);
        failures += expect(target.load(saved_vectors.view(), saved_neighbors.view(), synchronization_t::immutable_k) ==
                           status_t::config_mismatch_k);
        auto narrow_vectors = nk::matrix<scalar_t>::try_zeros({nodes_count, dimensions / 2});
        auto wide_neighbors = nk::matrix<node_id_t>::try_zeros({nodes_count, connectivity_32_k});
        auto short_vectors = nk::matrix<scalar_t>::try_zeros({nodes_count - 1, dimensions});
        failures += expect(!narrow_vectors.empty() && !wide_neighbors.empty() && !short_vectors.empty());
        failures += expect(target.load(narrow_vectors.view(), saved_neighbors.view(), synchronization_t::exclusive_k) ==
                           status_t::config_mismatch_k);
        failures += expect(target.load(saved_vectors.view(), wide_neighbors.view(), synchronization_t::exclusive_k) ==
                           status_t::config_mismatch_k);
        failures += expect(target.load(short_vectors.view(), saved_neighbors.view(), synchronization_t::exclusive_k) ==
                           status_t::config_mismatch_k);
        index_type_ unfrozen = make_index(ledger);
        failures += expect(unfrozen.load(saved_vectors.view(), saved_neighbors.view(),
                                         synchronization_t::exclusive_k) == status_t::config_mismatch_k);

        // Every refusal above was pre-copy: the same target now loads the intact payload,
        // and a second load into the non-empty target is the documented misuse.
        failures += expect(target.load(saved_vectors.view(), saved_neighbors.view(), synchronization_t::exclusive_k) ==
                           status_t::success_k);
        failures += expect_eq(target.size(), nodes_count);
        failures += expect(target.load(saved_vectors.view(), saved_neighbors.view(), synchronization_t::exclusive_k) ==
                           status_t::config_mismatch_k);

        // A count past the reservation grows by design - the chunk directory follows.
        index_type_ growing = make_index(ledger);
        failures += expect(growing.reserve({1, connectivity_16_k, dimensions}).status == status_t::success_k);
        failures += expect(growing.load(saved_vectors.view(), saved_neighbors.view(), synchronization_t::exclusive_k) ==
                           status_t::success_k);
        failures += expect_eq(growing.size(), nodes_count);

        // An out-of-range neighbor id loads raw - the sentinel convention owns validation -
        // but the translation frontier keeps it from ever surfacing in a search.
        auto patched_neighbors = nk::matrix<node_id_t>::try_zeros({nodes_count, connectivity_16_k});
        failures += expect(!patched_neighbors.empty());
        std::size_t const row_ids = dense_rows_bytes(saved_neighbors.view(), 1) / sizeof(node_id_t);
        std::memcpy(patched_neighbors.data(), saved_neighbors.data(),
                    dense_rows_bytes(saved_neighbors.view(), nodes_count));
        patched_neighbors.data()[5 * row_ids] = static_cast<node_id_t>(nodes_count + 5);
        index_type_ patched = make_index(ledger);
        failures += expect(patched.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
        failures += expect(patched.load(saved_vectors.view(), patched_neighbors.view(),
                                        synchronization_t::exclusive_k) == status_t::success_k);
        typename index_type_::dense_traversal_context_t patched_context {{}};
        for (std::size_t sample = 0; sample != 16; ++sample) {
            auto found = patched.search(vectors.view().row((sample * 19) % nodes_count).as_vector(), patched_context);
            failures += expect(found.status == status_t::success_k);
            for (std::size_t match = 0; match != found.count; ++match)
                failures += expect(static_cast<std::size_t>(patched_context.results.data()[match].node_id) <
                                   patched.size());
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Snapshots

#pragma region Races

/** @brief Concurrent construction on ForkUnion's default flat pool: distinct slices per lane,
 *      one traversal context each, then id-uniqueness and sampled self-recall checks. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_concurrent_build(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const writer_lanes_count = 4;
        std::size_t const nodes_count = writer_lanes_count * dense_lane_nodes_k;

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writer_lanes_count))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        failures += expect(
            index.reserve({nodes_count, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
            status_t::success_k);

        auto lanes = dense_make_lanes<index_type_>(writer_lanes_count, dense_lane_nodes_k,
                                                   dense_dimensions_for<scalar_t>(dense_values_k),
                                                   dense_seed_concurrent_build_k);
        std::vector<dense_node_id_t> assigned_ids(nodes_count);
        std::atomic<std::size_t> lane_failures {0};

        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= writer_lanes_count) return;
            auto result = index.insert( //
                lanes.slices[lane].view(), {assigned_ids.data() + lane * dense_lane_nodes_k, dense_lane_nodes_k},
                lanes.contexts[lane]);
            if (result.status != status_t::success_k) lane_failures.fetch_add(1);
            assert_no_locks_held();
        });

        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
        failures += expect_contiguous_ids(assigned_ids, 0);

        if constexpr (kernel_ != nk_kernel_dot_k)
            for (std::size_t lane = 0; lane != writer_lanes_count; ++lane)
                failures += dense_expect_self_recall(index, lanes.slices[lane].view(), dense_lane_nodes_k, 4);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/**
 *  @brief A reader lane races three writer lanes, asserting only interleaving-independent
 *      invariants: every search succeeds, respects the candidate-pool bound, returns distances
 *      in ascending order, and never surfaces an id at or above the watermark it could have
 *      observed - the translation-frontier invariant, audited live.
 *
 *  The query vectors are never inserted: a search does not require its query to be stored, so
 *  the reader & writers share only the index itself, and any sanitizer report is an index bug
 *  rather than test-harness noise.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_concurrent_search(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const writer_lanes_count = 3, queries_count = 64;
        std::size_t const nodes_count = writer_lanes_count * dense_lane_nodes_k;
        std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writer_lanes_count + 1))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        failures += expect(
            index.reserve({nodes_count, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
            status_t::success_k);

        auto lanes = dense_make_lanes<index_type_>(writer_lanes_count + 1, dense_lane_nodes_k,
                                                   dense_dimensions_for<scalar_t>(dense_values_k),
                                                   dense_seed_concurrent_search_k);
        auto query_vectors = make_vectors<scalar_t>(queries_count, dense_dimensions_for<scalar_t>(dense_values_k),
                                                    dense_seed_concurrent_search_k ^ salt_queries_k);

        std::atomic<std::size_t> writer_lanes_remaining {writer_lanes_count};
        std::atomic<std::size_t> lane_failures {0};

        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane != 0 && lane <= writer_lanes_count) {
                auto result = index.insert(lanes.slices[lane].view(), {}, lanes.contexts[lane]);
                if (result.status != status_t::success_k) lane_failures.fetch_add(1);
                writer_lanes_remaining.fetch_sub(1, std::memory_order_release);
                assert_no_locks_held();
                return;
            }
            if (lane != 0) return;

            // Every other step runs the predicated sibling under the same race: all the
            // unfiltered invariants must hold, plus admission purity - no id the predicate
            // rejects may ever surface, no matter how the writers interleave.
            auto odd_only = [](dense_node_id_t node_id) noexcept {
                return node_id % 2 == 1 ? admission_t::admit_k : admission_t::reject_k;
            };
            for (std::size_t step = 0; step < dense_reader_minimal_searches_k || //
                                       writer_lanes_remaining.load(std::memory_order_acquire) != 0;
                 ++step) {
                auto query = query_vectors.view().row(step % queries_count).as_vector();
                // The predicated sibling runs only where the backend exposes the verb -
                // the GPU joins this arm the moment a device-side search_if lands.
                constexpr bool predicable_k = requires { index.search_if(query, odd_only, lanes.contexts[0]); };
                bool const predicated = predicable_k && step % 2 == 1;
                auto found = [&] {
                    if constexpr (predicable_k)
                        if (predicated) return index.search_if(query, odd_only, lanes.contexts[0]);
                    return index.search(query, lanes.contexts[0]);
                }();
                if (found.status != status_t::success_k) lane_failures.fetch_add(1);
                lane_failures.fetch_add(
                    scored_failures(lanes.contexts[0].results.data(), found.count, index.size(), expansion_bound));
                if (predicated)
                    for (std::size_t match = 0; match != found.count; ++match)
                        if (lanes.contexts[0].results.data()[match].node_id % 2 != 1) lane_failures.fetch_add(1);
            }
            assert_no_locks_held();
        });

        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/**
 *  @brief The sharded index's refusal surface, all deterministic: verbs before placement,
 *      placement after reservation, geometry drift, mismatched spans, and a doubly-claimed
 *      executor - every misuse answers with a status, never with a write.
 */
template <typename indexes_type_>
std::size_t test_dense_indexes_guards() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using executor_t = typename indexes_type_::executor_t;
    std::size_t failures = 0;
    std::size_t const rows = 64;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k) return failures + expect(false);
    {
        auto spawned = executor_t::spawn(topology, 2);
        if (failed(spawned.status)) {
            fu_topology_delete(topology);
            return failures + expect(false);
        }
        executor_t executor = std::move(spawned.value);
        auto vectors = make_vectors<scalar_t>(rows, dimensions, dense_seed_guards_k);
        std::vector<key_t> keys(rows);
        for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

        // Verbs before any shard exists refuse rather than write.
        indexes_type_ indexes;
        failures += expect_eq(indexes.add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                          executor)
                                  .status,
                              status_t::uninitialized_k);
        failures += expect_eq(indexes.reserve({rows, connectivity_16_k, dimensions}), status_t::uninitialized_k);

        // Reserving closes the shard set, and the frozen geometry refuses drift.
        failures += expect(succeeded(indexes.emplace(fu_memory_domain_id_t {-1})));
        failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions})));
        failures += expect_eq(indexes.emplace(fu_memory_domain_id_t {-1}), status_t::config_mismatch_k);
        failures += expect_eq(indexes.reserve({rows, connectivity_32_k, dimensions}), status_t::config_mismatch_k);
        failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions})));

        // A keys span disagreeing with the batch's rows refuses before anything is taken.
        failures += expect_eq(indexes.add(std::span<key_t const> {keys.data(), rows - 1}, vectors.view(), //
                                          executor)
                                  .status,
                              status_t::config_mismatch_k);

        // The executor admits one caller: a live claim turns the second away, and releasing
        // it restores service - which the closing insertion proves end to end.
        {
            auto held = executor.claim();
            failures += expect(succeeded(held.status));
            failures += expect_eq(executor.claim().status, status_t::contended_k);
            failures += expect_eq(indexes.add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                              executor)
                                      .status,
                              status_t::contended_k);
        }
        failures += expect(succeeded(indexes.add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                                 executor)
                                         .status));
        failures += expect_eq(indexes.size(), rows);
    }
    fu_topology_delete(topology);
    return failures;
}

/**
 *  @brief Two executors drive one sharded index concurrently: disjoint key ranges, one
 *      batched @c add per executor, both interleaving into the same two shards - asserted
 *      through the public surface only.
 *
 *  This scenario needs a real topology to spawn executors - the one departure from the
 *  flat-pool rule, since an executor sites its arenas by asking where its workers run.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_concurrent_add() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using executor_t = typename indexes_type_::executor_t;
    using traversal_t = typename indexes_type_::traversal_t;
    std::size_t failures = 0;

    std::size_t const executors_count = 2;
    std::size_t const rows_per_executor = 2 * dense_lane_nodes_k;
    std::size_t const nodes_count = executors_count * rows_per_executor;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k) return failures + expect(false);
    {
        indexes_type_ indexes;
        failures += expect(succeeded(indexes.emplace(fu_memory_domain_id_t {-1})));
        failures += expect(succeeded(indexes.emplace(fu_memory_domain_id_t {-1})));
        failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions})));

        std::vector<executor_t> executors;
        std::vector<nk::matrix<scalar_t>> slices;
        std::vector<std::vector<key_t>> keys;
        for (std::size_t which = 0; which != executors_count; ++which) {
            auto spawned = executor_t::spawn(topology, 2);
            if (failed(spawned.status)) return failures + expect(false);
            executors.push_back(std::move(spawned.value));
            slices.push_back(make_vectors<scalar_t>(rows_per_executor, dimensions, dense_seed_indexes_add_k + which));
            // Keys never take the value zero, so a stale read of a zero-filled key plane can
            // never masquerade as a legitimate identity.
            std::vector<key_t> range(rows_per_executor);
            for (std::size_t row = 0; row != rows_per_executor; ++row)
                range[row] = static_cast<key_t>(1 + which * rows_per_executor + row);
            keys.push_back(std::move(range));
        }

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(executors_count))) return failures + expect(false);
        std::atomic<std::size_t> lane_failures {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= executors_count) return;
            auto added = indexes.add(std::span<key_t const> {keys[lane].data(), rows_per_executor},
                                   slices[lane].view(), executors[lane]);
            if (added.status != status_t::success_k) lane_failures.fetch_add(1);
            assert_no_locks_held();
        });
        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect_eq(indexes.size(), nodes_count);

        // Rotation splits every batch evenly, so two shards end exactly half-and-half full.
        failures += expect_eq(indexes.shard(0).size() + indexes.shard(1).size(), nodes_count);
        failures += expect_eq(indexes.shard(0).size(), nodes_count / 2);

        // Every key is stored exactly once, and resolvable without naming a shard or an id.
        std::size_t miscounted = 0;
        for (std::size_t row = 0; row != nodes_count; ++row)
            miscounted += indexes.count(static_cast<key_t>(1 + row)) != 1;
        failures += expect_eq(miscounted, 0u);

        // Sampled self-recall through the merged cross-shard search, judged by distance band.
        if constexpr (kernel_ != nk_kernel_dot_k) {
            typename executor_t::thread_arena_t arena;
            failures += expect(succeeded(arena.reserve_exactly(indexes.shard(0).traversal_memory_usage())));
            traversal_t context {typename traversal_t::allocator_t {arena}};
            std::size_t hits = 0;
            std::size_t const samples_count = 8;
            std::array<key_t, 8> found_keys {};
            std::array<typename indexes_type_::distance_t, 8> found_distances {};
            for (std::size_t sample = 0; sample != samples_count; ++sample) {
                std::size_t const row = sample * (rows_per_executor / samples_count);
                auto const found = indexes.search(slices[0].view().row(row).as_vector(), context, {},
                                                std::span<key_t> {found_keys}, std::span {found_distances});
                failures += expect(found.status == status_t::success_k && found.count > 0);
                for (std::size_t match = 0; match != found.count; ++match)
                    if (found_distances[match] < dense_self_match_tolerance_k &&
                        found_distances[match] > -dense_self_match_tolerance_k) {
                        ++hits;
                        break;
                    }
            }
            failures += expect(hits * 100 >= samples_count * dense_minimal_recall_percent_k);
        }
    }
    fu_topology_delete(topology);
    return failures;
}

/**
 *  @brief A reader lane races one executor-driven batched @c add into two shards, asserting
 *      the key-visibility invariant: a search may reach a node the moment it publishes, so
 *      the key it resolves must already be written - never the plane's zero-fill, never
 *      outside the assigned range.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_search_race() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using executor_t = typename indexes_type_::executor_t;
    using traversal_t = typename indexes_type_::traversal_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_lane_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k) return failures + expect(false);
    {
        indexes_type_ indexes;
        failures += expect(succeeded(indexes.emplace(fu_memory_domain_id_t {-1})));
        failures += expect(succeeded(indexes.emplace(fu_memory_domain_id_t {-1})));
        failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions})));

        auto spawned = executor_t::spawn(topology, 2);
        if (failed(spawned.status)) return failures + expect(false);
        executor_t executor = std::move(spawned.value);

        auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, dense_seed_indexes_race_k);
        std::vector<key_t> keys(nodes_count);
        for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(1 + row);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<bool> writer_done {false};
        std::atomic<std::size_t> lane_failures {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) {
                auto added = indexes.add(std::span<key_t const> {keys.data(), nodes_count}, vectors.view(), executor);
                if (added.status != status_t::success_k) lane_failures.fetch_add(1);
                writer_done.store(true, std::memory_order_release);
                assert_no_locks_held();
                return;
            }
            typename executor_t::thread_arena_t arena;
            if (failed(arena.reserve_exactly(indexes.shard(0).traversal_memory_usage()))) {
                lane_failures.fetch_add(1);
                writer_done.store(true, std::memory_order_release);
                return;
            }
            traversal_t context {typename traversal_t::allocator_t {arena}};
            std::array<key_t, 8> found_keys {};
            std::array<typename indexes_type_::distance_t, 8> found_distances {};
            std::size_t searches = 0;
            for (; !writer_done.load(std::memory_order_acquire) || searches < dense_reader_minimal_searches_k;
                 ++searches) {
                auto const found = indexes.search(vectors.view().row(searches % nodes_count).as_vector(), context, {},
                                                std::span<key_t> {found_keys}, std::span {found_distances});
                if (found.status != status_t::success_k) {
                    lane_failures.fetch_add(1);
                    break;
                }
                for (std::size_t match = 0; match != found.count; ++match)
                    if (found_keys[match] < key_t {1} || found_keys[match] > static_cast<key_t>(nodes_count))
                        lane_failures.fetch_add(1);
            }
            assert_no_locks_held();
        });
        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect_eq(indexes.size(), nodes_count);
    }
    fu_topology_delete(topology);
    return failures;
}

/**
 *  @brief The optimistic save arm under fire: a saver lane loops concurrent-arm saves
 *      against two inserter lanes & a searcher, classifying every answer as @c contended_k
 *      or @c success_k - and every settled payload must be structurally sane and reload
 *      into a fresh index whose recall holds. Inserters keep trickling spare rows until
 *      the saver has provably contended at least once, and a settle phase after the race
 *      demands a deterministic success.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_save_race(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const inserter_lanes_count = 2, spare_rows_count = 64;
        std::size_t const prefill_rows_count = 256;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t const nodes_ceiling = prefill_rows_count + inserter_lanes_count * (dense_lane_nodes_k + //
                                                                                       spare_rows_count);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(inserter_lanes_count + 2))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        failures += expect(index.reserve({nodes_ceiling, connectivity_16_k, dimensions}).status == status_t::success_k);

        auto lanes = dense_make_lanes<index_type_>(inserter_lanes_count + 2, dense_lane_nodes_k + spare_rows_count,
                                                   dimensions, dense_seed_save_race_k);
        auto prefill_vectors = make_vectors<scalar_t>(prefill_rows_count, dimensions,
                                                      dense_seed_save_race_k ^ salt_extras_k);
        typename index_type_::dense_traversal_context_t prefill_context {{}};
        failures += expect(index.insert(prefill_vectors.view(), {}, prefill_context).status == status_t::success_k);

        // The saver's destinations are sized once at the ceiling; each attempt carves the
        // current shape's rows-prefix, so nothing allocates inside the race loop.
        auto saver_vectors = nk::matrix<scalar_t>::try_zeros({nodes_ceiling, dimensions});
        auto saver_neighbors = nk::matrix<node_id_t>::try_zeros({nodes_ceiling, connectivity_16_k});
        failures += expect(!saver_vectors.empty() && !saver_neighbors.empty());

        struct recorded_payload_t {
            nk::matrix<scalar_t> vectors;
            nk::matrix<node_id_t> neighbors;
            std::size_t nodes = 0;
        };
        std::vector<recorded_payload_t> recorded_payloads;
        recorded_payloads.reserve(4);

        std::atomic<std::size_t> inserter_lanes_remaining {inserter_lanes_count};
        std::atomic<std::size_t> contended_saves {0}, settled_saves {0}, lane_failures {0};

        auto attempt_save = [&]() noexcept {
            auto const shape = index.saved_shape();
            auto vectors_prefix = saver_vectors.span()(nk::range(0, shape.nodes), nk::slice);
            auto neighbors_prefix = saver_neighbors.span()(nk::range(0, shape.nodes), nk::slice);
            status_t const saved = index.save(vectors_prefix, neighbors_prefix, synchronization_t::concurrent_k);
            // The watermark may outrun the sizing before the probe: a shape mismatch
            // names the same contention only when the size provably moved.
            if (saved == status_t::config_mismatch_k && index.size() != shape.nodes)
                return std::pair {status_t::contended_k, shape.nodes};
            return std::pair {saved, shape.nodes};
        };
        auto audit_settled = [&](std::size_t nodes) noexcept {
            std::size_t const row_ids = dense_rows_bytes(saver_neighbors.view(), 1) / sizeof(node_id_t);
            for (std::size_t row = 0; row != nodes; ++row) {
                node_id_t const* ids = saver_neighbors.data() + row * row_ids;
                if (ids[0] == dense_node_id_unknown_k) return false; // Never a parked lock word.
                if (ids[0] == static_cast<node_id_t>(row)) continue; // Zero-neighbors self-loop.
                for (std::size_t slot = 0; slot != connectivity_16_k; ++slot) {
                    if (ids[slot] == dense_node_id_unknown_k) break; // The sentinel ends the row.
                    if (static_cast<std::size_t>(ids[slot]) >= nodes) return false;
                }
            }
            return true;
        };
        auto record_settled = [&](std::size_t nodes) noexcept {
            settled_saves.fetch_add(1);
            if (!audit_settled(nodes)) lane_failures.fetch_add(1);
            if (recorded_payloads.size() == recorded_payloads.capacity()) return;
            recorded_payload_t recorded {nk::matrix<scalar_t>::try_zeros({nodes, dimensions}),
                                         nk::matrix<node_id_t>::try_zeros({nodes, connectivity_16_k}), nodes};
            if (recorded.vectors.empty() || recorded.neighbors.empty()) return;
            std::memcpy(recorded.vectors.data(), saver_vectors.data(), dense_rows_bytes(saver_vectors.view(), nodes));
            std::memcpy(recorded.neighbors.data(), saver_neighbors.data(),
                        dense_rows_bytes(saver_neighbors.view(), nodes));
            recorded_payloads.push_back(std::move(recorded));
        };
        auto saver_lane = [&]() noexcept { // Classify, audit, and record every settled payload.
            for (std::size_t attempts = 0; inserter_lanes_remaining.load(std::memory_order_acquire) != 0; ++attempts) {
                if (attempts == spin_limit_k) return void(lane_failures.fetch_add(1)); // A stuck racer.
                auto const [saved, nodes] = attempt_save();
                if (saved == status_t::contended_k) contended_saves.fetch_add(1);
                else if (saved == status_t::success_k) record_settled(nodes);
                else lane_failures.fetch_add(1);
            }
            // The settle phase: with every racer done, a save must soon succeed.
            for (std::size_t attempts = 0; attempts != attempt_limit_k; ++attempts) {
                auto const [saved, nodes] = attempt_save();
                if (saved == status_t::success_k) return record_settled(nodes);
                if (saved != status_t::contended_k) return void(lane_failures.fetch_add(1));
            }
            lane_failures.fetch_add(1); // The settle phase starved.
        };
        auto inserter_lane = [&](std::size_t lane) noexcept { // A slice, then spare singles
                                                              // until the saver has contended.
            auto slice = lanes.slices[lane].view();
            auto main_rows = slice(nk::range(0, dense_lane_nodes_k), nk::slice);
            if (index.insert(main_rows, {}, lanes.contexts[lane]).status != status_t::success_k)
                lane_failures.fetch_add(1);
            for (std::size_t spare = 0; spare != spare_rows_count && //
                                        contended_saves.load(std::memory_order_acquire) == 0;
                 ++spare) {
                auto spare_row = slice(nk::range(dense_lane_nodes_k + spare, dense_lane_nodes_k + spare + 1),
                                       nk::slice);
                if (index.insert(spare_row, {}, lanes.contexts[lane]).status != status_t::success_k)
                    lane_failures.fetch_add(1);
                std::this_thread::yield();
            }
            inserter_lanes_remaining.fetch_sub(1, std::memory_order_release);
            assert_no_locks_held();
        };
        auto searcher_lane = [&](std::size_t lane) noexcept { // The reader-lane invariants.
            std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;
            auto query_slice = lanes.slices[lane].view();
            for (std::size_t step = 0; step < dense_reader_minimal_searches_k ||
                                       inserter_lanes_remaining.load(std::memory_order_acquire) != 0;
                 ++step) {
                auto query = query_slice.row(step % dense_lane_nodes_k).as_vector();
                auto found = index.search(query, lanes.contexts[lane]);
                if (found.status != status_t::success_k) lane_failures.fetch_add(1);
                lane_failures.fetch_add(
                    scored_failures(lanes.contexts[lane].results.data(), found.count, index.size(), expansion_bound));
            }
            assert_no_locks_held();
        };
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) return saver_lane();
            if (lane <= inserter_lanes_count) return inserter_lane(lane);
            if (lane == inserter_lanes_count + 1) return searcher_lane(lane);
        });

        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect(contended_saves.load() > 0);
        failures += expect(settled_saves.load() > 0);
        failures += expect(recorded_payloads.size() >= 1);

        // Every settled payload reloads into a fresh index whose self-recall holds, the
        // payload's own vector rows serving as the queries.
        for (recorded_payload_t const& recorded : recorded_payloads) {
            index_type_ restored = make_index(ledger);
            failures += expect(restored.reserve({recorded.nodes, connectivity_16_k, dimensions}).status ==
                               status_t::success_k);
            failures += expect(restored.load(recorded.vectors.view(), recorded.neighbors.view(),
                                             synchronization_t::exclusive_k) == status_t::success_k);
            failures += expect_eq(restored.size(), recorded.nodes);
            if constexpr (kernel_ != nk_kernel_dot_k)
                failures += dense_expect_self_recall(restored, recorded.vectors.view(), recorded.nodes, 16);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Races

#pragma region Readers

/** @brief On a quiesced graph the @c immutable_k search arm reproduces the concurrent arm
 *      bit for bit - fresh identically-seeded contexts, so the landmark draws align - at
 *      the default expansion and at an explicit one. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_immutable_parity(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const nodes_count = 512, stored_queries_count = 16, novel_queries_count = 8;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t build_context {{}};
        auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, dense_seed_immutable_parity_k);
        failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
        failures += expect(index.insert(vectors.view(), {}, build_context).status == status_t::success_k);
        auto novel_queries = make_vectors<scalar_t>(novel_queries_count, dimensions,
                                                    dense_seed_immutable_parity_k ^ salt_queries_k);

        auto compare_arms = [&](auto query, std::size_t expansion_factor) noexcept {
            typename index_type_::dense_traversal_context_t concurrent_context {{}}, immutable_context {{}};
            typename index_type_::search_config_t concurrent_config {};
            concurrent_config.expansion_factor = static_cast<decltype(concurrent_config.expansion_factor)>( //
                expansion_factor);
            typename index_type_::search_config_t immutable_config = concurrent_config;
            immutable_config.synchronization = synchronization_t::immutable_k;
            auto concurrent_found = index.search(query, concurrent_context, concurrent_config);
            auto immutable_found = index.search(query, immutable_context, immutable_config);
            failures += expect_eq(immutable_found.status, concurrent_found.status);
            failures += expect_eq(immutable_found.count, concurrent_found.count);
            for (std::size_t match = 0; match != concurrent_found.count; ++match) {
                failures += expect_eq(immutable_context.results.data()[match].node_id,
                                      concurrent_context.results.data()[match].node_id);
                failures += expect(immutable_context.results.data()[match].distance ==
                                   concurrent_context.results.data()[match].distance);
            }
        };
        for (std::size_t expansion_factor : {std::size_t {0}, std::size_t {64}}) {
            for (std::size_t sample = 0; sample != stored_queries_count; ++sample)
                compare_arms(vectors.view().row((sample * 31) % nodes_count).as_vector(), expansion_factor);
            for (std::size_t sample = 0; sample != novel_queries_count; ++sample)
                compare_arms(novel_queries.view().row(sample).as_vector(), expansion_factor);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The external reader's cursor, exercised across chunk seams: full-prefix parity
 *      of ids & vector bytes, follow semantics on the staged row, refresh stickiness -
 *      past the end stays past the end - a lane racing live inserters, and the immutable
 *      arm agreeing with the concurrent one on the settled graph. */
template <typename index_type_, typename concurrent_cursor_type_, typename immutable_cursor_type_,
          typename make_index_type_>
std::size_t test_dense_cursor(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const nodes_count = 1'200, racing_inserts_count = 64;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(nodes_count + 1 + racing_inserts_count, dimensions, dense_seed_cursor_k);
        failures += expect(
            index.reserve({nodes_count + 1 + racing_inserts_count, connectivity_16_k, dimensions}).status ==
            status_t::success_k);
        auto main_rows = vectors.view()(nk::range(0, nodes_count), nk::slice);
        failures += expect(index.insert(main_rows, {}, context).status == status_t::success_k);
        std::size_t const row_bytes = dense_rows_bytes(vectors.view(), 1);

        concurrent_cursor_type_ cursor {index};
        failures += expect(!cursor); // A fresh cursor starts past the end.
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_count);
        failures += expect_eq(cursor.seek(0), status_t::uninitialized_k); // Before its reserve.
        failures += expect_eq(cursor.reserve(counting_allocator<std::byte> {ledger}), status_t::success_k);

        // Full-prefix parity: id order, exact stored bytes, every staged id in bounds.
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        std::size_t visited = 0;
        for (; cursor; ++cursor, ++visited) {
            if (cursor.id() != static_cast<node_id_t>(visited)) failures += expect(false);
            if (std::memcmp(cursor.vector().data(), vectors.view().row(visited).as_vector().data(), row_bytes) != 0)
                failures += expect(false);
            auto const staged = cursor.neighbors();
            if (staged.size() > connectivity_16_k) failures += expect(false);
            for (node_id_t const neighbor : staged)
                if (neighbor >= cursor.visible_count()) failures += expect(false);
        }
        failures += expect_eq(visited, nodes_count);

        // Follow semantics: every staged edge lands where the row said; misuse refuses.
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        std::vector<node_id_t> first_row {cursor.neighbors().begin(), cursor.neighbors().end()};
        for (std::size_t slot = 0; slot != first_row.size(); ++slot) {
            failures += expect_eq(cursor.seek(0), status_t::success_k);
            failures += expect_eq(cursor.follow(static_cast<typename index_type_::small_size_t>(slot)),
                                  status_t::success_k);
            failures += expect_eq(cursor.id(), first_row[slot]);
        }
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        failures += expect_eq(cursor.follow(static_cast<typename index_type_::small_size_t>(cursor.neighbors().size())),
                              status_t::malformed_k);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_count)), status_t::malformed_k);

        // The snapshot is sticky until refreshed; a past-the-end cursor never resurrects.
        auto late_row = vectors.view()(nk::range(nodes_count, nodes_count + 1), nk::slice);
        failures += expect(index.insert(late_row, {}, context).status == status_t::success_k);
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_count);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_count)), status_t::malformed_k);
        cursor.refresh();
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_count + 1);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_count)), status_t::success_k);
        concurrent_cursor_type_ parked_cursor {index};
        failures += expect_eq(parked_cursor.reserve(counting_allocator<std::byte> {ledger}), status_t::success_k);

        // A cursor lane races an inserting lane; every visited id stays below that pass's
        // snapshot, and the parked cursor re-anchors past the end after the race.
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> lane_failures {0};
        std::atomic<bool> writer_done {false};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) {
                typename index_type_::dense_traversal_context_t lane_context {{}};
                for (std::size_t ordinal = 0; ordinal != racing_inserts_count; ++ordinal) {
                    auto racing_row = vectors.view()(nk::range(nodes_count + 1 + ordinal, nodes_count + 2 + ordinal),
                                                     nk::slice);
                    if (index.insert(racing_row, {}, lane_context).status != status_t::success_k)
                        lane_failures.fetch_add(1);
                    std::this_thread::yield();
                }
                writer_done.store(true, std::memory_order_release);
                assert_no_locks_held();
                return;
            }
            if (lane != 1) return;
            concurrent_cursor_type_ racing_cursor {index};
            if (racing_cursor.reserve(counting_allocator<std::byte> {ledger}) != status_t::success_k)
                return void(lane_failures.fetch_add(1));
            std::size_t sweeps = 0;
            do {
                if (sweeps++ == attempt_limit_k) return void(lane_failures.fetch_add(1)); // A stuck writer.
                racing_cursor.refresh();
                node_id_t const snapshot = racing_cursor.visible_count();
                if (failed(racing_cursor.seek(0))) return void(lane_failures.fetch_add(1));
                for (; racing_cursor; ++racing_cursor)
                    if (racing_cursor.id() >= snapshot) lane_failures.fetch_add(1);
            } while (!writer_done.load(std::memory_order_acquire));
        });
        failures += expect_eq(lane_failures.load(), 0u);
        parked_cursor.refresh();
        failures += expect(!parked_cursor);
        failures += expect_eq(static_cast<std::size_t>(parked_cursor.id()), index.size());
        failures += expect_eq(parked_cursor.seek(0), status_t::success_k);

        // The immutable arm sweeps the settled graph to the same count & the same rows.
        immutable_cursor_type_ settled_cursor {index};
        failures += expect_eq(settled_cursor.reserve(counting_allocator<std::byte> {ledger}), status_t::success_k);
        failures += expect_eq(settled_cursor.seek(0), status_t::success_k);
        std::size_t settled_visited = 0;
        for (; settled_cursor; ++settled_cursor) ++settled_visited;
        failures += expect_eq(settled_visited, index.size());
        for (std::size_t sample = 0; sample != 8; ++sample) {
            node_id_t const node = static_cast<node_id_t>((sample * 149) % index.size());
            failures += expect_eq(settled_cursor.seek(node), status_t::success_k);
            failures += expect_eq(cursor.seek(node), status_t::success_k);
            auto const settled_staged = settled_cursor.neighbors();
            auto const concurrent_staged = cursor.neighbors();
            failures += expect_eq(settled_staged.size(), concurrent_staged.size());
            for (std::size_t slot = 0; slot != settled_staged.size(); ++slot)
                failures += expect_eq(settled_staged[slot], concurrent_staged[slot]);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Readers

#pragma region Allocation Budgets

/** @brief The ledger's byte budget provably starves exactly the chunk-growth path: a
 *      refused mid-insert growth is all-or-nothing & retryable, and a starved load
 *      publishes nothing - the fresh-index retry then succeeds with recall intact. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_allocation_budget(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const extra_rows_count = 100;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto reserved = index.reserve({1'000, connectivity_16_k, dimensions});
        failures += expect(reserved.status == status_t::success_k);
        std::size_t const fill_rows_count = reserved.new_capacity; // Fill to the brim, so
                                                                   // one more row must grow.
        auto vectors = make_vectors<scalar_t>(fill_rows_count + extra_rows_count, dimensions,
                                              dense_seed_allocation_budget_k);
        auto fill_rows = vectors.view()(nk::range(0, fill_rows_count), nk::slice);
        auto extra_rows = vectors.view()(nk::range(fill_rows_count, fill_rows_count + extra_rows_count), nk::slice);
        failures += expect(index.insert(fill_rows, {}, context).status == status_t::success_k);

        // Freeze the budget: the growth election must refuse, all-or-nothing.
        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect(index.insert(extra_rows, {}, context).status == status_t::bad_alloc_k);
        failures += expect_eq(index.size(), fill_rows_count);

        // Lift it: the identical batch is retryable.
        ledger.byte_budget.store(PTRDIFF_MAX);
        failures += expect(index.insert(extra_rows, {}, context).status == status_t::success_k);
        std::size_t const nodes_count = fill_rows_count + extra_rows_count;
        failures += expect_eq(index.size(), nodes_count);

        auto saved_vectors = nk::matrix<scalar_t>::try_zeros({nodes_count, dimensions});
        auto saved_neighbors = nk::matrix<node_id_t>::try_zeros({nodes_count, connectivity_16_k});
        failures += expect(!saved_vectors.empty() && !saved_neighbors.empty());
        failures += expect(index.save(saved_vectors.span(), saved_neighbors.span(), synchronization_t::exclusive_k) ==
                           status_t::success_k);

        // A starved load publishes nothing; the documented contract destroys the victim.
        {
            index_type_ starved = make_index(ledger);
            auto starved_reserved = starved.reserve({1, connectivity_16_k, dimensions});
            failures += expect(starved_reserved.status == status_t::success_k);
            failures += expect(starved_reserved.new_capacity < nodes_count); // The growth is genuine.
            ledger.byte_budget.store(ledger.live_bytes.load());
            failures += expect(starved.load(saved_vectors.view(), saved_neighbors.view(),
                                            synchronization_t::exclusive_k) == status_t::bad_alloc_k);
            failures += expect_eq(starved.size(), 0u);
            ledger.byte_budget.store(PTRDIFF_MAX);
        }
        index_type_ restored = make_index(ledger);
        failures += expect(restored.reserve({nodes_count, connectivity_16_k, dimensions}).status ==
                           status_t::success_k);
        failures += expect(restored.load(saved_vectors.view(), saved_neighbors.view(),
                                         synchronization_t::exclusive_k) == status_t::success_k);
        failures += expect_eq(restored.size(), nodes_count);
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(restored, vectors.view(), nodes_count, 16);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Racing a bounded arena: with the budget frozen at exactly two batches of spare
 *      capacity, four uniform lanes resolve to exactly two winners & two @c bad_alloc_k
 *      refusals, the winners' ids land dense & duplicate-free, the id-space ceiling still
 *      answers @c overflow_k, and a lifted budget makes a refused batch retryable. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_overflow_race(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const racing_lanes_count = 4, batch_rows_count = 512;
        std::size_t const contested_rows_count = 2 * batch_rows_count;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(racing_lanes_count))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        auto reserved = index.reserve({contested_rows_count, connectivity_16_k, dimensions});
        failures += expect(reserved.status == status_t::success_k);
        failures += expect(reserved.new_capacity >= contested_rows_count);

        // Pre-fill any chunk-rounding surplus, so exactly two batches of capacity remain.
        std::size_t const prefill_rows_count = reserved.new_capacity - contested_rows_count;
        typename index_type_::dense_traversal_context_t context {{}};
        if (prefill_rows_count) {
            auto prefill_vectors = make_vectors<scalar_t>(prefill_rows_count, dimensions,
                                                          dense_seed_overflow_race_k ^ salt_extras_k);
            failures += expect(index.insert(prefill_vectors.view(), {}, context).status == status_t::success_k);
        }

        auto lanes = dense_make_lanes<index_type_>(racing_lanes_count, batch_rows_count, dimensions,
                                                   dense_seed_overflow_race_k);
        std::vector<dense_node_id_t> assigned_ids(racing_lanes_count * batch_rows_count, dense_node_id_unknown_k);
        std::array<status_t, 4> lane_statuses {};
        ledger.byte_budget.store(ledger.live_bytes.load()); // No growth: the arena is the arena.

        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= racing_lanes_count) return;
            lane_statuses[lane] = index
                                      .insert(lanes.slices[lane].view(),
                                              {assigned_ids.data() + lane * batch_rows_count, batch_rows_count},
                                              lanes.contexts[lane])
                                      .status;
            assert_no_locks_held();
        });

        std::size_t winners_count = 0;
        std::size_t refused_lane = racing_lanes_count;
        std::vector<dense_node_id_t> winner_ids;
        for (std::size_t lane = 0; lane != racing_lanes_count; ++lane) {
            if (lane_statuses[lane] == status_t::success_k) {
                ++winners_count;
                winner_ids.insert(winner_ids.end(), assigned_ids.begin() + lane * batch_rows_count,
                                  assigned_ids.begin() + (lane + 1) * batch_rows_count);
            }
            else {
                failures += expect_eq(lane_statuses[lane], status_t::bad_alloc_k);
                refused_lane = lane;
            }
        }
        failures += expect_eq(winners_count, 2u);
        failures += expect_eq(index.size(), reserved.new_capacity);
        failures += expect_contiguous_ids(winner_ids, prefill_rows_count);

        // The genuine overflow_k belongs to the id-space ceiling alone, refused before any
        // counter dance or allocation - a probe only a 32-bit id space can express.
        using node_id_t = typename index_type_::node_id_t;
        if constexpr (sizeof(node_id_t) == 4)
            failures += expect_eq(index.reserve({std::size_t {1} << 33, connectivity_16_k, dimensions}).status,
                                  status_t::overflow_k);

        // A lifted budget makes a refused batch retryable.
        ledger.byte_budget.store(PTRDIFF_MAX);
        failures += expect(refused_lane != racing_lanes_count);
        failures += expect(index
                               .insert(lanes.slices[refused_lane].view(), {}, //
                                       lanes.contexts[refused_lane])
                               .status == status_t::success_k);
        failures += expect_eq(index.size(), reserved.new_capacity + batch_rows_count);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Allocation Budgets

} // namespace usearch
} // namespace unum
