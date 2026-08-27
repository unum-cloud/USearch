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
 *  default flat pool - no NUMA machinery in tests.
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
#include "usearch/dense/ram_metric.hpp" // `usearch::distance`, `usearch::dense_norms` - the oracle's kernels

namespace unum {
namespace usearch {

/** @brief Storage values per vector in the larger scenarios - NumKong's unit of one
 *      scalar object; small enough to keep serial kernels fast. */
constexpr std::size_t test_values_k = 16;
/** @brief Storage values per vector in the tiny smoke scenarios. */
constexpr std::size_t tiny_values_k = 8;
/** @brief Logical dimensions spanned by @p values_count storage values of one dtype -
 *      NumKong's @c dimensions_per_value conversion, so every dtype gets the same storage
 *      budget: 16 i8 values carry 16 dimensions, 16 u1x8 values carry 128. */
template <typename scalar_>
constexpr std::size_t dimensions_for(std::size_t values_count) noexcept {
    return values_count * nk::dimensions_per_value<scalar_>();
}
/** @brief Node count marching the growth scenario across several doubling chunks. */
constexpr std::size_t growth_nodes_count_k = 1'500;
/** @brief Nodes inserted by each lane of the concurrent scenarios. */
constexpr std::size_t lane_nodes_count_k = 256;
/** @brief Self-recall sampling threshold: at least this share of samples must find themselves. */
constexpr std::size_t minimal_recall_percent_k = 90;
/** @brief Distance bound for a stored vector matching itself under symmetric kernels. */
constexpr float self_match_tolerance_k = 1e-3f;
/** @brief The reader lane performs at least this many searches, even if writers finish first. */
constexpr std::size_t reader_minimal_searches_k = 16;

/** @brief Per-lane state for the concurrent scenarios: one vector slice & one traversal context each. */
template <typename index_type_>
struct test_lanes {
    std::vector<nk::matrix<typename index_type_::scalar_t>> slices;
    std::vector<typename index_type_::dense_traversal_context_t> contexts;
};

/** @brief Prepares @p lanes_count lanes with distinctly-seeded vector slices & fresh contexts. */
template <typename index_type_>
test_lanes<index_type_> make_lanes(std::size_t lanes_count, std::size_t nodes_per_lane, std::size_t dimensions,
                                   u64_t seed) {
    using context_t = typename index_type_::dense_traversal_context_t;
    test_lanes<index_type_> lanes;
    for (std::size_t lane = 0; lane != lanes_count; ++lane) {
        lanes.slices.push_back(make_vectors<typename index_type_::scalar_t>(nodes_per_lane, dimensions, seed + lane));
        lanes.contexts.emplace_back(typename context_t::allocator_t {});
    }
    return lanes;
}

/** @brief Guards: pre-reserve misuse and geometry freezing, on a fresh index per scenario. */
template <typename index_type_, typename make_index_type_>
static std::size_t test_guards(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(3, dimensions_for<scalar_t>(tiny_values_k), 42);

        auto inserted = index.insert(vectors.view(), {}, context);
        failures_count += expect(inserted.status == status_t::uninitialized_k);
        auto found = index.search(vectors.view().row(0).as_vector(), context);
        failures_count += expect(found.status == status_t::success_k && found.count == 0);

        auto reserved = index.reserve({64, connectivity_16_k, dimensions_for<scalar_t>(tiny_values_k)});
        failures_count += expect(reserved.status == status_t::success_k && reserved.new_capacity >= 64);
        auto mismatched = index.reserve({64, connectivity_32_k, dimensions_for<scalar_t>(tiny_values_k)});
        failures_count += expect(mismatched.status == status_t::config_mismatch_k);
        auto matching = index.reserve({32, connectivity_16_k, dimensions_for<scalar_t>(tiny_values_k)});
        failures_count += expect(matching.status == status_t::success_k);
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
}

/** @brief The classic smoke scenario: three vectors in, each finds itself first. The dot kernel
 *      maximizes similarity instead of minimizing distance, so self-nearest is only asserted
 *      for the symmetric kernels. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_three_vectors(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(3, dimensions_for<scalar_t>(tiny_values_k), 7);

        failures_count += expect(
            index.reserve({8, connectivity_16_k, dimensions_for<scalar_t>(tiny_values_k)}).status ==
            status_t::success_k);
        std::array<dense_node_id_t, 3> assigned_ids;
        auto inserted = index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context);
        failures_count += expect(inserted.status == status_t::success_k);
        failures_count += expect_eq(inserted.new_size, 3u);
        failures_count += expect_eq(index.size(), 3u);

        for (std::size_t row = 0; row != 3; ++row) {
            auto found = index.search(vectors.view().row(row).as_vector(), context);
            failures_count += expect(found.status == status_t::success_k && found.count >= 1);
            if constexpr (kernel_ != nk_kernel_dot_k) {
                dense_scored_node_t const nearest = context.results.data()[0];
                failures_count += expect_eq(nearest.node_id, assigned_ids[row]);
                failures_count += expect(nearest.distance < self_match_tolerance_k && //
                                         nearest.distance > -self_match_tolerance_k);
            }
        }
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
}

/** @brief Growth across several doubling chunks, then sampled self-recall on the big graph,
 *      with telemetry accounting sanity checks along the way. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_growth_and_recall(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(growth_nodes_count_k, dimensions_for<scalar_t>(test_values_k), 2026);

        failures_count += expect(
            index.reserve({16, connectivity_16_k, dimensions_for<scalar_t>(test_values_k)}).status ==
            status_t::success_k);
        std::vector<dense_node_id_t> assigned_ids(growth_nodes_count_k);
        auto inserted = index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context);
        failures_count += expect(inserted.status == status_t::success_k);
        failures_count += expect_eq(inserted.new_size, growth_nodes_count_k);
        failures_count += expect_eq(index.size(), growth_nodes_count_k);
        failures_count += expect(inserted.computed_distances > 0 && inserted.visited_members > 0);
        failures_count += expect(inserted.computed_distances_in_refines <= inserted.computed_distances);
        failures_count += expect(inserted.computed_distances_in_reverse_refines <= inserted.computed_distances);

        if constexpr (kernel_ != nk_kernel_dot_k) {
            std::size_t hits = 0, samples = 32;
            for (std::size_t sample = 0; sample != samples; ++sample) {
                std::size_t const row = sample * (growth_nodes_count_k / samples);
                auto found = index.search(vectors.view().row(row).as_vector(), context);
                failures_count += expect(found.status == status_t::success_k && found.count > 0);
                for (std::size_t match = 0; match != found.count; ++match)
                    if (context.results.data()[match].node_id == assigned_ids[row]) {
                        ++hits;
                        break;
                    }
            }
            failures_count += expect(hits * 100 >= samples * minimal_recall_percent_k);
        }
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
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
static std::size_t test_predicated_search(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr nk_capability_t caps_k = index_type_::picked_quads_capability_k;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(growth_nodes_count_k, dimensions_for<scalar_t>(test_values_k), 7071);

        failures_count += expect(
            index.reserve({growth_nodes_count_k, connectivity_16_k, dimensions_for<scalar_t>(test_values_k)}).status ==
            status_t::success_k);
        std::vector<dense_node_id_t> assigned_ids(growth_nodes_count_k);
        failures_count += expect(
            index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context).status ==
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
        std::size_t const query_row = growth_nodes_count_k / 2;
        auto query = vectors.view().row(query_row).as_vector();
        typename index_type_::dense_traversal_context_t unfiltered_context {{}};
        auto unfiltered = index.search(query, unfiltered_context);
        failures_count += expect(unfiltered.status == status_t::success_k && unfiltered.count > 0);
        dense_scored_node_t const* reference = unfiltered_context.results.data();

        // The trivially-true predicate walks and reports exactly like the unfiltered sibling.
        auto all_pass = [](dense_node_id_t) noexcept { return true; };
        typename index_type_::dense_traversal_context_t all_pass_context {{}};
        auto with_true = index.search_if(query, all_pass, all_pass_context);
        failures_count += expect_eq(with_true.count, unfiltered.count);
        for (std::size_t match = 0; match != with_true.count && match != unfiltered.count; ++match) {
            failures_count += expect_eq(all_pass_context.results.data()[match].node_id, reference[match].node_id);
            failures_count += expect(all_pass_context.results.data()[match].distance == reference[match].distance);
        }

        // The batched shape must agree with the per-node shape on the same walk.
        auto even_only = [](dense_node_id_t node_id) noexcept { return node_id % 2 == 0; };
        auto even_batched = [](std::span<dense_scored_node_t const> scored, auto& admitted) noexcept {
            for (dense_scored_node_t const candidate : scored)
                if (candidate.node_id % 2 == 0) {
                    [[maybe_unused]] fu::status_t const kept = admitted.push_back(candidate);
                }
        };
        typename index_type_::dense_traversal_context_t scalar_context {{}}, batched_context {{}};
        auto with_scalar = index.search_if(query, even_only, scalar_context);
        auto with_batched = index.search_if(query, even_batched, batched_context);
        failures_count += expect(with_scalar.status == status_t::success_k && with_scalar.count > 0);
        failures_count += expect_eq(with_batched.count, with_scalar.count);
        for (std::size_t match = 0; match != with_batched.count && match != with_scalar.count; ++match)
            failures_count += expect_eq(batched_context.results.data()[match].node_id,
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
            auto matches = [modulus](dense_node_id_t node_id) noexcept { return node_id % modulus == 0; };
            for (std::size_t sample = 0; sample != 8; ++sample) {
                std::size_t sampled_row = ((sample * 641 + modulus) % growth_nodes_count_k) / modulus * modulus;
                if (sample % 2) sampled_row += 1; // An excluded neighbor of a matching row.
                typename index_type_::dense_traversal_context_t sampled_context {{}};
                auto found = index.search_if(vectors.view().row(sampled_row).as_vector(), matches, sampled_context);
                bool const self_matches = sampled_row % modulus == 0;
                // The sparsest decade may legally drain empty on excluded-row queries: until
                // the admitted pool fills, the navigation radius bounds the walk, and no match
                // needs to fall inside it. A matching self is always near enough to admit.
                failures_count += expect(found.status == status_t::success_k);
                failures_count += expect(found.count > 0 || (modulus == 64 && !self_matches));
                dense_scored_node_t const* results = sampled_context.results.data();
                bool self_found = false;
                for (std::size_t match = 0; match != found.count; ++match) {
                    failures_count += expect(results[match].node_id % modulus == 0);
                    dense_distance_t const oracle = oracle_distance(sampled_row, results[match].node_id);
                    failures_count += expect_near(results[match].distance, oracle);
                    if (match) failures_count += expect(results[match].distance >= results[match - 1].distance);
                    if (!self_matches) failures_count += expect(results[match].node_id != assigned_ids[sampled_row]);
                    self_found |= results[match].node_id == assigned_ids[sampled_row];
                }
                if constexpr (kernel_ != nk_kernel_dot_k)
                    if (self_matches && modulus != 64) hits += self_found, ++recall_samples;
            }
        }
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures_count += expect(hits * 100 >= recall_samples * minimal_recall_percent_k);

        // An impossible predicate drains legally: empty result set, successful status.
        auto none_pass = [](dense_node_id_t) noexcept { return false; };
        typename index_type_::dense_traversal_context_t none_context {{}};
        auto with_none = index.search_if(query, none_pass, none_context);
        failures_count += expect(with_none.status == status_t::success_k);
        failures_count += expect_eq(with_none.count, std::size_t {0});
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
}

/** @brief Round-trips the graph through the two-matrix serialization: build, save, load into
 *      a fresh reserved index, and expect the restored graph to answer self-queries just as
 *      the original did - norms re-derive on load and are never serialized. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_save_load_roundtrip(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        index_type_ index = make_index(ledger);
        typename index_type_::dense_traversal_context_t context {{}};
        auto vectors = make_vectors<scalar_t>(growth_nodes_count_k, dimensions_for<scalar_t>(test_values_k), 4242);

        failures_count += expect(
            index.reserve({growth_nodes_count_k, connectivity_16_k, dimensions_for<scalar_t>(test_values_k)}).status ==
            status_t::success_k);
        std::vector<dense_node_id_t> assigned_ids(growth_nodes_count_k);
        failures_count += expect(
            index.insert(vectors.view(), {assigned_ids.data(), assigned_ids.size()}, context).status ==
            status_t::success_k);

        auto const shape = index.saved_shape();
        auto saved_vectors = nk::matrix<scalar_t>::try_zeros({shape.nodes, shape.dimensions});
        auto saved_neighbors = nk::matrix<node_id_t>::try_zeros({shape.nodes, shape.connectivity});
        failures_count += expect(!saved_vectors.empty() && !saved_neighbors.empty());
        failures_count += expect(index.save(saved_vectors.span(), saved_neighbors.span(),
                                            synchronization_t::exclusive_k) == status_t::success_k);

        index_type_ restored = make_index(ledger);
        failures_count += expect(
            restored.reserve({shape.nodes, static_cast<connectivity_t>(shape.connectivity), shape.dimensions}).status ==
            status_t::success_k);
        failures_count += expect(restored.load(saved_vectors.view(), saved_neighbors.view(),
                                               synchronization_t::exclusive_k) == status_t::success_k);
        failures_count += expect_eq(restored.size(), shape.nodes);

        if constexpr (kernel_ != nk_kernel_dot_k) {
            std::size_t hits = 0, samples = 32;
            for (std::size_t sample = 0; sample != samples; ++sample) {
                std::size_t const row = sample * (growth_nodes_count_k / samples);
                auto found = restored.search(vectors.view().row(row).as_vector(), context);
                failures_count += expect(found.status == status_t::success_k && found.count > 0);
                for (std::size_t match = 0; match != found.count; ++match)
                    if (context.results.data()[match].node_id == assigned_ids[row]) {
                        ++hits;
                        break;
                    }
            }
            failures_count += expect(hits * 100 >= samples * minimal_recall_percent_k);
        }
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
}

/** @brief Concurrent construction on ForkUnion's default flat pool: distinct slices per lane,
 *      one traversal context each, then id-uniqueness and sampled self-recall checks. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_concurrent_build(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const lanes_count = 4;
        std::size_t const nodes_count = lanes_count * lane_nodes_count_k;

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(lanes_count))) return expect(false);

        index_type_ index = make_index(ledger);
        failures_count += expect(
            index.reserve({nodes_count, connectivity_16_k, dimensions_for<scalar_t>(test_values_k)}).status ==
            status_t::success_k);

        auto lanes = make_lanes<index_type_>(lanes_count, lane_nodes_count_k, dimensions_for<scalar_t>(test_values_k),
                                             100);
        std::vector<dense_node_id_t> assigned_ids(nodes_count);
        std::atomic<std::size_t> failed_lanes_count {0};

        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= lanes_count) return;
            auto result = index.insert( //
                lanes.slices[lane].view(), {assigned_ids.data() + lane * lane_nodes_count_k, lane_nodes_count_k},
                lanes.contexts[lane]);
            if (result.status != status_t::success_k) failed_lanes_count.fetch_add(1);
        });

        failures_count += expect_eq(failed_lanes_count.load(), 0u);
        failures_count += expect_eq(index.size(), nodes_count);
        std::sort(assigned_ids.begin(), assigned_ids.end());
        failures_count += expect(std::unique(assigned_ids.begin(), assigned_ids.end()) == assigned_ids.end());
        failures_count += expect(assigned_ids.front() == 0 && assigned_ids.back() == nodes_count - 1);

        if constexpr (kernel_ != nk_kernel_dot_k) {
            std::size_t hits = 0, samples = 16;
            for (std::size_t sample = 0; sample != samples; ++sample) {
                std::size_t const lane = sample % lanes_count, row = (sample * 37) % lane_nodes_count_k;
                auto found = index.search(lanes.slices[lane].view().row(row).as_vector(), lanes.contexts[0]);
                failures_count += expect(found.status == status_t::success_k && found.count > 0);
                dense_scored_node_t const nearest = lanes.contexts[0].results.data()[0];
                hits += nearest.distance < self_match_tolerance_k && nearest.distance > -self_match_tolerance_k;
            }
            failures_count += expect(hits * 100 >= samples * minimal_recall_percent_k);
        }
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
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
static std::size_t test_concurrent_search(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures_count = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const writers_count = 3, queries_count = 64;
        std::size_t const nodes_count = writers_count * lane_nodes_count_k;
        std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writers_count + 1))) return expect(false);

        index_type_ index = make_index(ledger);
        failures_count += expect(
            index.reserve({nodes_count, connectivity_16_k, dimensions_for<scalar_t>(test_values_k)}).status ==
            status_t::success_k);

        auto lanes = make_lanes<index_type_>(writers_count + 1, lane_nodes_count_k,
                                             dimensions_for<scalar_t>(test_values_k), 200);
        auto query_vectors = make_vectors<scalar_t>(queries_count, dimensions_for<scalar_t>(test_values_k), 777);

        std::atomic<std::size_t> writers_remaining {writers_count};
        std::atomic<std::size_t> racing_violations {0};

        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane != 0 && lane <= writers_count) {
                auto result = index.insert(lanes.slices[lane].view(), {}, lanes.contexts[lane]);
                if (result.status != status_t::success_k) racing_violations.fetch_add(1);
                writers_remaining.fetch_sub(1, std::memory_order_release);
                return;
            }
            if (lane != 0) return;

            // Every other step runs the predicated sibling under the same race: all the
            // unfiltered invariants must hold, plus admission purity - no id the predicate
            // rejects may ever surface, no matter how the writers interleave.
            auto odd_only = [](dense_node_id_t node_id) noexcept { return node_id % 2 == 1; };
            std::size_t violations = 0;
            for (std::size_t step = 0; step < reader_minimal_searches_k || //
                                       writers_remaining.load(std::memory_order_acquire) != 0;
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
                std::size_t const published_after = index.size();
                violations += found.status != status_t::success_k;
                violations += found.count > expansion_bound;
                for (std::size_t match = 0; match != found.count; ++match) {
                    dense_scored_node_t const scored = lanes.contexts[0].results.data()[match];
                    violations += scored.node_id >= published_after;
                    violations += predicated && scored.node_id % 2 != 1;
                    if (match) violations += scored.distance < lanes.contexts[0].results.data()[match - 1].distance;
                }
            }
            racing_violations.fetch_add(violations);
        });

        failures_count += expect_eq(racing_violations.load(), 0u);
        failures_count += expect_eq(index.size(), nodes_count);
    }
    failures_count += expect_eq(ledger.live_bytes.load(), 0);
    return failures_count;
}

} // namespace usearch
} // namespace unum
