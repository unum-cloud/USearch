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
 *  need a real topology because an executor colocates each arena with the thread that walks it.
 */
#pragma once

#include <cstdint> // `std::uintptr_t` - the checkpoint paths' per-run salt
#include <cstdio>  // `std::snprintf`, `std::remove` - the checkpoint round-trip's files
#include <cstdlib> // `std::abs`
#include <cstring> // `std::strlen`, `std::memcmp` - the fetched-row byte comparison

#include <algorithm> // `std::sort`, `std::unique`
#include <array>     // `std::array`
#include <atomic>    // `std::atomic`
#include <bit>       // `std::bit_cast` - the ULP distance of the oracle audits
#include <chrono>    // `std::chrono::microseconds`
#include <thread>    // `std::this_thread::sleep_for`
#include <vector>    // `std::vector`

#include <fmt/format.h>
#include <forkunion.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>

#include "test.hpp"
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp" // `forkunion_executor_t` - the RAM residency's executor
#include "usearch/utilities/sanitizers.hpp" // `assert_no_locks_held` - the racing threads' tail check
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
/** @brief Nodes inserted by each thread of the concurrent scenarios. */
constexpr std::size_t dense_thread_nodes_k = 256;
/** @brief Self-recall sampling threshold: at least this share of samples must find themselves. */
constexpr std::size_t dense_minimal_recall_percent_k = 90;
/** @brief Distance bound for a stored vector matching itself under symmetric kernels. */
constexpr float dense_self_match_tolerance_k = 1e-3f;
/** @brief The reader thread performs at least this many searches, even if writers finish first. */
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

/** @brief Per-thread state for the concurrent scenarios: one vector slice & one traversal context each. */
template <typename index_type_>
struct dense_threads {
    std::vector<nk::matrix<typename index_type_::scalar_t>> slices;
    std::vector<typename index_type_::dense_traversal_context_t> contexts;
};

/** @brief Prepares @p threads_count threads with distinctly-seeded vector slices & fresh contexts. */
template <typename index_type_>
dense_threads<index_type_> dense_make_threads(std::size_t threads_count, std::size_t nodes_per_thread,
                                              std::size_t dimensions, u64_t seed) {
    using context_t = typename index_type_::dense_traversal_context_t;
    dense_threads<index_type_> threads;
    for (std::size_t thread = 0; thread != threads_count; ++thread) {
        threads.slices.push_back(
            make_vectors<typename index_type_::scalar_t>(nodes_per_thread, dimensions, seed + thread));
        threads.contexts.emplace_back(typename context_t::allocator_t {});
    }
    return threads;
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
 *      no assigned-ids plumbing, so it survives reloads & thread-sliced builds unchanged.
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

/** @brief One caller-built traversal context over its own arena, sized for the shard geometry -
 *      non-movable, since the context's allocator points into the arena beside it. */
template <typename indexes_type_>
struct dense_arena_context {
    /** @brief The backing run every carve of the context lands in. */
    typename indexes_type_::executor_t::arena_t arena;
    /** @brief The traversal context, allocating from the arena beside it. */
    typename indexes_type_::traversal_t context;
    /** @brief The arena reservation's answer; every use checks it before walking. */
    status_t status;

    explicit dense_arena_context(indexes_type_& indexes) noexcept
        : arena(), context(typename indexes_type_::traversal_t::allocator_t {arena}),
          status(arena.reserve_exactly(indexes.shard(0).traversal_memory_usage())) {}
    dense_arena_context(dense_arena_context&&) = delete;
    dense_arena_context(dense_arena_context const&) = delete;
};

/** @brief Whether a sharded index runs on ForkUnion threads - the RAM residency. */
template <typename indexes_type_>
concept ram_indexes = std::is_same_v<typename indexes_type_::executor_t, forkunion_executor_t>;

/** @brief The executor one sharded scenario drives, owned together with whatever it was
 *      spawned over; @c status reports the spawn. Defined per residency. */
template <typename indexes_type_>
struct dense_indexes_executor;

/** @brief The RAM form: two threads over a topology of its own, both released with it. */
template <ram_indexes indexes_type_>
struct dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;

    fu_topology_t topology = nullptr;
    executor_t executor;
    status_t status = status_t::uninitialized_k;

    dense_indexes_executor() noexcept {
        if (fu_topology_new(&topology) != fu_success_k) return;
        expected<executor_t> spawned = executor_t::spawn(topology, 2);
        status = spawned.status;
        executor = std::move(spawned.value);
    }
    ~dense_indexes_executor() noexcept {
        executor = executor_t {};
        if (topology) fu_topology_delete(topology);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;
};

/** @brief Places the shards one sharded scenario drives: two RAM shards on the heap domain. */
template <ram_indexes indexes_type_>
status_t place_shards(indexes_type_& indexes) noexcept {
    if (status_t const placed = indexes.emplace(no_memory_domain_k); failed(placed)) return placed;
    return indexes.emplace(no_memory_domain_k);
}

/** @brief What the one-row verbs walk with; @c status reports its construction. Defined per
 *      residency. */
template <typename indexes_type_>
struct dense_indexes_row_driver;

/** @brief The RAM form: a caller-built context over its own arena, sized for the shard geometry. */
template <ram_indexes indexes_type_>
struct dense_indexes_row_driver<indexes_type_> {
    dense_arena_context<indexes_type_> arena;
    status_t status;

    dense_indexes_row_driver(indexes_type_& indexes, dense_indexes_executor<indexes_type_>&) noexcept
        : arena(indexes), status(arena.status) {}
    typename indexes_type_::traversal_t& handle() noexcept { return arena.context; }
};

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
        if constexpr (requires { context.restarts; }) failures += expect_eq(inserted.restarts, context.restarts);

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
        // A device index refuses the concurrent arm by contract; the RAM index settles it.
        constexpr bool device_resident_k = requires(index_type_ const& index) { index.device_specs(); };
        failures += expect(
            restored.save(resaved_vectors.span(), resaved_neighbors.span(), synchronization_t::concurrent_k) ==
            (device_resident_k ? status_t::config_mismatch_k : status_t::success_k));
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

/** @brief Concurrent construction on ForkUnion's default flat pool: distinct slices per thread,
 *      one traversal context each, then id-uniqueness and sampled self-recall checks. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_concurrent_build(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const writer_threads_count = 4;
        std::size_t const nodes_count = writer_threads_count * dense_thread_nodes_k;

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writer_threads_count))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        failures += expect(
            index.reserve({nodes_count, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
            status_t::success_k);

        auto threads = dense_make_threads<index_type_>(writer_threads_count, dense_thread_nodes_k,
                                                       dense_dimensions_for<scalar_t>(dense_values_k),
                                                       dense_seed_concurrent_build_k);
        std::vector<dense_node_id_t> assigned_ids(nodes_count);
        std::atomic<std::size_t> thread_failures {0};

        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread >= writer_threads_count) return;
            auto result = index.insert( //
                threads.slices[thread].view(),
                {assigned_ids.data() + thread * dense_thread_nodes_k, dense_thread_nodes_k}, threads.contexts[thread]);
            if (result.status != status_t::success_k) thread_failures.fetch_add(1);
            assert_no_locks_held();
        });

        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
        failures += expect_contiguous_ids(assigned_ids, 0);

        if constexpr (kernel_ != nk_kernel_dot_k)
            for (std::size_t thread = 0; thread != writer_threads_count; ++thread)
                failures += dense_expect_self_recall(index, threads.slices[thread].view(), dense_thread_nodes_k, 4);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/**
 *  @brief A reader thread races three writer threads, asserting only interleaving-independent
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
        std::size_t const writer_threads_count = 3, queries_count = 64;
        std::size_t const nodes_count = writer_threads_count * dense_thread_nodes_k;
        std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writer_threads_count + 1))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        failures += expect(
            index.reserve({nodes_count, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
            status_t::success_k);

        auto threads = dense_make_threads<index_type_>(writer_threads_count + 1, dense_thread_nodes_k,
                                                       dense_dimensions_for<scalar_t>(dense_values_k),
                                                       dense_seed_concurrent_search_k);
        auto query_vectors = make_vectors<scalar_t>(queries_count, dense_dimensions_for<scalar_t>(dense_values_k),
                                                    dense_seed_concurrent_search_k ^ salt_queries_k);

        std::atomic<std::size_t> writer_lanes_remaining {writer_threads_count};
        std::atomic<std::size_t> thread_failures {0};

        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread != 0 && thread <= writer_threads_count) {
                auto result = index.insert(threads.slices[thread].view(), {}, threads.contexts[thread]);
                if (result.status != status_t::success_k) thread_failures.fetch_add(1);
                writer_lanes_remaining.fetch_sub(1, std::memory_order_release);
                assert_no_locks_held();
                return;
            }
            if (thread != 0) return;

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
                constexpr bool predicable_k = requires { index.search_if(query, odd_only, threads.contexts[0]); };
                bool const predicated = predicable_k && step % 2 == 1;
                auto found = [&] {
                    if constexpr (predicable_k)
                        if (predicated) return index.search_if(query, odd_only, threads.contexts[0]);
                    return index.search(query, threads.contexts[0]);
                }();
                if (found.status != status_t::success_k) thread_failures.fetch_add(1);
                thread_failures.fetch_add(
                    scored_failures(threads.contexts[0].results.data(), found.count, index.size(), expansion_bound));
                if (predicated)
                    for (std::size_t match = 0; match != found.count; ++match)
                        if (threads.contexts[0].results.data()[match].node_id % 2 != 1) thread_failures.fetch_add(1);
            }
            assert_no_locks_held();
        });

        failures += expect_eq(thread_failures.load(), 0u);
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
    std::size_t failures = 0;
    std::size_t const rows = 64;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, dense_seed_guards_k);
    auto oversized = make_vectors<scalar_t>(rows + 1, dimensions, dense_seed_guards_k + 1);
    std::vector<key_t> keys(rows + 1);
    for (std::size_t row = 0; row != rows + 1; ++row) keys[row] = static_cast<key_t>(1 + row);

    // Verbs before any shard exists refuse rather than write.
    indexes_type_ indexes;
    failures += expect_eq(indexes
                              .add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                   executor)
                              .status,
                          status_t::uninitialized_k);
    failures += expect_eq(indexes.reserve({rows, connectivity_16_k, dimensions}), status_t::uninitialized_k);

    // Reserving closes the shard set, and the frozen geometry refuses drift.
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions})));
    failures += expect_eq(place_shards(indexes), status_t::config_mismatch_k);
    failures += expect_eq(indexes.reserve({rows, connectivity_32_k, dimensions}), status_t::config_mismatch_k);
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions})));

    // A keys span disagreeing with the batch's rows refuses before anything is taken.
    failures += expect_eq(indexes
                              .add(std::span<key_t const> {keys.data(), rows - 1}, vectors.view(), //
                                   executor)
                              .status,
                          status_t::config_mismatch_k);

    // The executor admits one caller: a live claim turns the second away, and releasing
    // it restores service - which the closing insertion proves end to end.
    {
        auto [held, claimed] = executor.claim();
        failures += expect(succeeded(claimed));
        failures += expect_eq(executor.claim().status, status_t::contended_k);
        failures += expect_eq(indexes
                                  .add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                       executor)
                                  .status,
                              status_t::contended_k);
    }
    // A batch past the reservation refuses whole - no partial fill, no id holes.
    failures += expect_eq(indexes
                              .add(std::span<key_t const> {keys.data(), rows + 1}, oversized.view(), //
                                   executor)
                              .status,
                          status_t::overflow_k);

    // A dropped capacity claim credits itself back two-sidedly: the exact-fit insertion
    // below cannot succeed unless these rows returned in full.
    if constexpr (requires { indexes.shard(0).claim(rows / 4); }) {
        [[maybe_unused]] auto claimed = indexes.shard(0).claim(rows / 4);
    }

    failures += expect(succeeded(indexes
                                     .add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                          executor)
                                     .status));
    failures += expect_eq(indexes.size(), rows);
    std::size_t rows_in_shards = 0;
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) rows_in_shards += indexes.shard(shard).size();
    failures += expect_eq(rows_in_shards, rows);
    return failures;
}

/**
 *  @brief One key, several rows: the reverse plane's whole reason to exist, driven through
 *      the single-row @c add with a caller-built arena context.
 */
template <typename indexes_type_>
std::size_t test_dense_indexes_duplicate_keys() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes;
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions})));

    auto vectors = make_vectors<scalar_t>(rows, dimensions, dense_seed_guards_k + 2);
    dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
    if (failed(rows_driver.status)) return failures + expect(false);

    // Every second row lands under one shared key; the rest keep distinct ones.
    key_t const shared = static_cast<key_t>(1000);
    for (std::size_t row = 0; row != rows; ++row) {
        key_t const key = row % 2 ? shared : static_cast<key_t>(1 + row);
        failures += expect(
            succeeded(indexes.add(key, vectors.view().row(row).as_vector(), rows_driver.handle()).status));
    }
    failures += expect_eq(indexes.size(), rows);
    failures += expect_eq(indexes.count(shared), rows / 2);
    failures += expect_eq(indexes.count(static_cast<key_t>(1)), 1u);
    failures += expect(indexes.contains(shared));
    failures += expect(!indexes.contains(static_cast<key_t>(2)));

    // The probe-then-fetch contract of `get`: the true total returns even when the copy
    // truncates, and an absent key answers zero without touching the output.
    auto fetched = make_vectors<scalar_t>(2, dimensions, dense_seed_guards_k + 3);
    failures += expect_eq(indexes.get(shared, fetched.span(), 2), rows / 2);
    failures += expect_eq(indexes.get(static_cast<key_t>(2), fetched.span(), 2), 0u);
    return failures;
}

/**
 *  @brief The sharded checkpoint's whole life: batched rows in, a manifest plus one monolith
 *      per shard out, a summary read without opening a shard, a fresh index restored into the
 *      same placement, and byte-identical vectors fetched back by key - with a cancelled save
 *      leaving no manifest, so the failed checkpoint reads as absent.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_save_load() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, dense_seed_roundtrip_k + 1);
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ indexes;
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions})));
    failures += expect(succeeded(indexes
                                     .add(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                          executor)
                                     .status));

    // Distinct per-process paths, so parallel suite runs never clobber each other.
    char manifest_path[64], shard_path[64];
    std::snprintf(manifest_path, sizeof(manifest_path), "/tmp/usearch-%08x.safegraphs.json",
                  static_cast<unsigned>(dense_seed_roundtrip_k ^ reinterpret_cast<std::uintptr_t>(&indexes)));
    std::size_t const stem_length = std::strlen(manifest_path) - std::strlen(".safegraphs.json");

    // A cancelled save writes shards but never the manifest - the checkpoint reads as absent.
    failures += expect_eq(
        indexes.save(manifest_path, executor, [](std::size_t) noexcept { return continuation_t::cancel_k; }),
        status_t::cancelled_k);
    failures += expect(failed(indexes_type_::metadata(manifest_path).status));

    failures += expect(succeeded(indexes.save(manifest_path, executor)));
    failures += expect(indexes.serialized_length() > 0);

    // The summary answers from the manifest and the first shard's header, metric included.
    auto summary = indexes_type_::metadata(manifest_path);
    failures += expect(succeeded(summary.status));
    failures += expect_eq(summary.value.nodes, rows);
    failures += expect_eq(summary.value.files, indexes.shards_count());
    failures += expect_eq(summary.value.dimensions, dimensions);
    failures += expect(nk_kernel_named(summary.value.metric.characters, summary.value.metric.view().size()) == kernel_);

    // A fresh index in the same placement restores it whole, keys and all.
    indexes_type_ restored;
    failures += expect(succeeded(place_shards(restored)));
    failures += expect(succeeded(restored.load(manifest_path, executor)));
    failures += expect_eq(restored.size(), rows);
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != rows; ++row) miscounted += restored.count(keys[row]) != 1;
    failures += expect_eq(miscounted, 0u);

    // Fetched rows come back byte-identical to what went in.
    auto fetched = make_vectors<scalar_t>(1, dimensions, dense_seed_roundtrip_k + 2);
    std::size_t const row_bytes = static_cast<std::size_t>(vectors.view().stride_bytes(0));
    std::size_t mismatched = 0;
    for (std::size_t sample = 0; sample != 8; ++sample) {
        std::size_t const row = sample * (rows / 8);
        if (restored.get(keys[row], fetched.span(), 1) != 1) mismatched += 1;
        else
            mismatched += std::memcmp(fetched.data(), //
                                      vectors.view().row(row).as_vector().byte_data(), row_bytes) != 0;
    }
    failures += expect_eq(mismatched, 0u);

    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {restored, driver};
        failures += expect(succeeded(rows_driver.status));
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        auto const found = restored.search(vectors.view().row(0).as_vector(), rows_driver.handle(), {},
                                           std::span<key_t> {found_keys}, std::span {found_distances});
        failures += expect(found.status == status_t::success_k && found.count > 0);
    }

    std::remove(manifest_path);
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) {
        std::snprintf(shard_path, sizeof(shard_path), "%.*s.%zu.safegraphs", static_cast<int>(stem_length),
                      manifest_path, shard);
        std::remove(shard_path);
    }
    return failures;
}

/**
 *  @brief Two executors drive one sharded index concurrently: disjoint key ranges, one
 *      batched @c add per executor, both interleaving into the same shards - asserted
 *      through the public surface only.
 *
 *  A RAM index admits both batches at once. A VRAM index admits one mutation at a time and
 *  answers the overlapping one with @c contended_k, so a refused batch is retried once the
 *  threads join; either way every row lands exactly once.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_concurrent_add() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const executors_count = 2;
    std::size_t const rows_per_executor = 2 * dense_thread_nodes_k;
    std::size_t const nodes_count = executors_count * rows_per_executor;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes;
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions})));

    std::array<dense_indexes_executor<indexes_type_>, executors_count> drivers;
    std::vector<nk::matrix<scalar_t>> slices;
    std::vector<std::vector<key_t>> keys;
    for (std::size_t which = 0; which != executors_count; ++which) {
        if (failed(drivers[which].status)) return failures + expect(false);
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
    std::array<status_t, executors_count> outcomes {};
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= executors_count) return;
        outcomes[thread] = indexes
                               .add(std::span<key_t const> {keys[thread].data(), rows_per_executor},
                                    slices[thread].view(), drivers[thread].executor)
                               .status;
        assert_no_locks_held();
    });
    for (std::size_t which = 0; which != executors_count; ++which) {
        if (outcomes[which] == status_t::contended_k && !ram_indexes<indexes_type_>)
            outcomes[which] = indexes
                                  .add(std::span<key_t const> {keys[which].data(), rows_per_executor},
                                       slices[which].view(), drivers[which].executor)
                                  .status;
        failures += expect_eq(outcomes[which], status_t::success_k);
    }
    failures += expect_eq(indexes.size(), nodes_count);

    // Rotation splits every batch evenly, so the shards end exactly equally full.
    std::size_t rows_in_shards = 0;
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) rows_in_shards += indexes.shard(shard).size();
    failures += expect_eq(rows_in_shards, nodes_count);
    failures += expect_eq(indexes.shard(0).size(), nodes_count / indexes.shards_count());

    // Every key is stored exactly once, and resolvable without naming a shard or an id.
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != nodes_count; ++row) miscounted += indexes.count(static_cast<key_t>(1 + row)) != 1;
    failures += expect_eq(miscounted, 0u);

    // Sampled self-recall through the merged cross-shard search, judged by distance band.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {indexes, drivers[0]};
        failures += expect(succeeded(rows_driver.status));
        std::size_t hits = 0;
        std::size_t const samples_count = 8;
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        for (std::size_t sample = 0; sample != samples_count; ++sample) {
            std::size_t const row = sample * (rows_per_executor / samples_count);
            auto const found = indexes.search(slices[0].view().row(row).as_vector(), rows_driver.handle(), {},
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
    return failures;
}

/**
 *  @brief A reader thread races one executor-driven batched @c add into the shards, asserting
 *      the key-visibility invariant: a search may reach a node the moment it publishes, so
 *      the key it resolves must already be written - never the plane's zero-fill, never
 *      outside the assigned range. RAM only: a VRAM index admits no search during a mutation.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
    requires ram_indexes<indexes_type_>
std::size_t test_dense_indexes_search_race() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes;
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions})));

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;

    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, dense_seed_indexes_race_k);
    std::vector<key_t> keys(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(1 + row);

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::atomic<bool> writer_done {false};
    std::atomic<std::size_t> thread_failures {0};
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            auto added = indexes.add(std::span<key_t const> {keys.data(), nodes_count}, vectors.view(), executor);
            if (added.status != status_t::success_k) thread_failures.fetch_add(1);
            writer_done.store(true, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
        if (failed(rows_driver.status)) {
            thread_failures.fetch_add(1);
            writer_done.store(true, std::memory_order_release);
            return;
        }
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        std::size_t searches = 0;
        for (; !writer_done.load(std::memory_order_acquire) || searches < dense_reader_minimal_searches_k; ++searches) {
            auto const found = indexes.search(vectors.view().row(searches % nodes_count).as_vector(),
                                              rows_driver.handle(), {}, std::span<key_t> {found_keys},
                                              std::span {found_distances});
            if (found.status != status_t::success_k) {
                thread_failures.fetch_add(1);
                break;
            }
            for (std::size_t match = 0; match != found.count; ++match)
                if (found_keys[match] < key_t {1} || found_keys[match] > static_cast<key_t>(nodes_count))
                    thread_failures.fetch_add(1);
        }
        assert_no_locks_held();
    });
    failures += expect_eq(thread_failures.load(), 0u);
    failures += expect_eq(indexes.size(), nodes_count);
    return failures;
}

/**
 *  @brief The optimistic save arm under fire: a saver thread loops concurrent-arm saves
 *      against two inserter threads & a searcher, classifying every answer as @c contended_k
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
        std::size_t const inserter_threads_count = 2, spare_rows_count = 64;
        std::size_t const prefill_rows_count = 256;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t const nodes_ceiling = prefill_rows_count + inserter_threads_count * (dense_thread_nodes_k + //
                                                                                         spare_rows_count);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(inserter_threads_count + 2))) return failures + expect(false);

        index_type_ index = make_index(ledger);
        failures += expect(index.reserve({nodes_ceiling, connectivity_16_k, dimensions}).status == status_t::success_k);

        auto threads = dense_make_threads<index_type_>(
            inserter_threads_count + 2, dense_thread_nodes_k + spare_rows_count, dimensions, dense_seed_save_race_k);
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

        std::atomic<std::size_t> inserter_threads_remaining {inserter_threads_count};
        std::atomic<std::size_t> contended_saves {0}, settled_saves {0}, thread_failures {0};

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
            if (!audit_settled(nodes)) thread_failures.fetch_add(1);
            if (recorded_payloads.size() == recorded_payloads.capacity()) return;
            recorded_payload_t recorded {nk::matrix<scalar_t>::try_zeros({nodes, dimensions}),
                                         nk::matrix<node_id_t>::try_zeros({nodes, connectivity_16_k}), nodes};
            if (recorded.vectors.empty() || recorded.neighbors.empty()) return;
            std::memcpy(recorded.vectors.data(), saver_vectors.data(), dense_rows_bytes(saver_vectors.view(), nodes));
            std::memcpy(recorded.neighbors.data(), saver_neighbors.data(),
                        dense_rows_bytes(saver_neighbors.view(), nodes));
            recorded_payloads.push_back(std::move(recorded));
        };
        auto saver_thread = [&]() noexcept { // Classify, audit, and record every settled payload.
            for (std::size_t attempts = 0; inserter_threads_remaining.load(std::memory_order_acquire) != 0;
                 ++attempts) {
                if (attempts == spin_limit_k) return void(thread_failures.fetch_add(1)); // A stuck racer.
                auto const [saved, nodes] = attempt_save();
                if (saved == status_t::contended_k) contended_saves.fetch_add(1);
                else if (saved == status_t::success_k) record_settled(nodes);
                else thread_failures.fetch_add(1);
            }
            // The settle phase: with every racer done, a save must soon succeed.
            for (std::size_t attempts = 0; attempts != attempt_limit_k; ++attempts) {
                auto const [saved, nodes] = attempt_save();
                if (saved == status_t::success_k) return record_settled(nodes);
                if (saved != status_t::contended_k) return void(thread_failures.fetch_add(1));
            }
            thread_failures.fetch_add(1); // The settle phase starved.
        };
        auto inserter_thread = [&](std::size_t thread) noexcept { // A slice, then spare singles
                                                                  // until the saver has contended.
            auto slice = threads.slices[thread].view();
            auto main_rows = slice(nk::range(0, dense_thread_nodes_k), nk::slice);
            if (index.insert(main_rows, {}, threads.contexts[thread]).status != status_t::success_k)
                thread_failures.fetch_add(1);
            for (std::size_t spare = 0; spare != spare_rows_count && //
                                        contended_saves.load(std::memory_order_acquire) == 0;
                 ++spare) {
                auto spare_row = slice(nk::range(dense_thread_nodes_k + spare, dense_thread_nodes_k + spare + 1),
                                       nk::slice);
                if (index.insert(spare_row, {}, threads.contexts[thread]).status != status_t::success_k)
                    thread_failures.fetch_add(1);
                std::this_thread::yield();
            }
            inserter_threads_remaining.fetch_sub(1, std::memory_order_release);
            assert_no_locks_held();
        };
        auto searcher_thread = [&](std::size_t thread) noexcept { // The reader-thread invariants.
            std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;
            auto query_slice = threads.slices[thread].view();
            for (std::size_t step = 0; step < dense_reader_minimal_searches_k ||
                                       inserter_threads_remaining.load(std::memory_order_acquire) != 0;
                 ++step) {
                auto query = query_slice.row(step % dense_thread_nodes_k).as_vector();
                auto found = index.search(query, threads.contexts[thread]);
                if (found.status != status_t::success_k) thread_failures.fetch_add(1);
                thread_failures.fetch_add(scored_failures(threads.contexts[thread].results.data(), found.count,
                                                          index.size(), expansion_bound));
            }
            assert_no_locks_held();
        };
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) return saver_thread();
            if (thread <= inserter_threads_count) return inserter_thread(thread);
            if (thread == inserter_threads_count + 1) return searcher_thread(thread);
        });

        failures += expect_eq(thread_failures.load(), 0u);
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
 *      past the end stays past the end - a thread racing live inserters, and the immutable
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
        failures += expect_eq(cursor.reserve(counting_allocator<byte_t> {ledger}), status_t::success_k);

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
        failures += expect_eq(parked_cursor.reserve(counting_allocator<byte_t> {ledger}), status_t::success_k);

        // A cursor thread races an inserting thread; every visited id stays below that pass's
        // snapshot, and the parked cursor re-anchors past the end after the race.
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> thread_failures {0};
        std::atomic<bool> writer_done {false};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                typename index_type_::dense_traversal_context_t lane_context {{}};
                for (std::size_t ordinal = 0; ordinal != racing_inserts_count; ++ordinal) {
                    auto racing_row = vectors.view()(nk::range(nodes_count + 1 + ordinal, nodes_count + 2 + ordinal),
                                                     nk::slice);
                    if (index.insert(racing_row, {}, lane_context).status != status_t::success_k)
                        thread_failures.fetch_add(1);
                    std::this_thread::yield();
                }
                writer_done.store(true, std::memory_order_release);
                assert_no_locks_held();
                return;
            }
            if (thread != 1) return;
            concurrent_cursor_type_ racing_cursor {index};
            if (racing_cursor.reserve(counting_allocator<byte_t> {ledger}) != status_t::success_k)
                return void(thread_failures.fetch_add(1));
            std::size_t sweeps = 0;
            do {
                if (sweeps++ == attempt_limit_k) return void(thread_failures.fetch_add(1)); // A stuck writer.
                racing_cursor.refresh();
                node_id_t const snapshot = racing_cursor.visible_count();
                if (failed(racing_cursor.seek(0))) return void(thread_failures.fetch_add(1));
                for (; racing_cursor; ++racing_cursor)
                    if (racing_cursor.id() >= snapshot) thread_failures.fetch_add(1);
            } while (!writer_done.load(std::memory_order_acquire));
        });
        failures += expect_eq(thread_failures.load(), 0u);
        parked_cursor.refresh();
        failures += expect(!parked_cursor);
        failures += expect_eq(static_cast<std::size_t>(parked_cursor.id()), index.size());
        failures += expect_eq(parked_cursor.seek(0), status_t::success_k);

        // The immutable arm sweeps the settled graph to the same count & the same rows.
        immutable_cursor_type_ settled_cursor {index};
        failures += expect_eq(settled_cursor.reserve(counting_allocator<byte_t> {ledger}), status_t::success_k);
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
 *      capacity, four uniform threads resolve to exactly two winners & two @c bad_alloc_k
 *      refusals, the winners' ids land dense & duplicate-free, the id-space ceiling still
 *      answers @c overflow_k, and a lifted budget makes a refused batch retryable. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_overflow_race(make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::size_t const racing_threads_count = 4, batch_rows_count = 512;
        std::size_t const contested_rows_count = 2 * batch_rows_count;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(racing_threads_count))) return failures + expect(false);

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

        auto threads = dense_make_threads<index_type_>(racing_threads_count, batch_rows_count, dimensions,
                                                       dense_seed_overflow_race_k);
        std::vector<dense_node_id_t> assigned_ids(racing_threads_count * batch_rows_count, dense_node_id_unknown_k);
        std::array<status_t, 4> thread_statuses {};
        ledger.byte_budget.store(ledger.live_bytes.load()); // No growth: the arena is the arena.

        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread >= racing_threads_count) return;
            thread_statuses[thread] = index
                                          .insert(threads.slices[thread].view(),
                                                  {assigned_ids.data() + thread * batch_rows_count, batch_rows_count},
                                                  threads.contexts[thread])
                                          .status;
            assert_no_locks_held();
        });

        std::size_t winners_count = 0;
        std::size_t refused_thread = racing_threads_count;
        std::vector<dense_node_id_t> winner_ids;
        for (std::size_t thread = 0; thread != racing_threads_count; ++thread) {
            if (thread_statuses[thread] == status_t::success_k) {
                ++winners_count;
                winner_ids.insert(winner_ids.end(), assigned_ids.begin() + thread * batch_rows_count,
                                  assigned_ids.begin() + (thread + 1) * batch_rows_count);
            }
            else {
                failures += expect_eq(thread_statuses[thread], status_t::bad_alloc_k);
                refused_thread = thread;
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
        failures += expect(refused_thread != racing_threads_count);
        failures += expect(index
                               .insert(threads.slices[refused_thread].view(), {}, //
                                       threads.contexts[refused_thread])
                               .status == status_t::success_k);
        failures += expect_eq(index.size(), reserved.new_capacity + batch_rows_count);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Allocation Budgets

} // namespace usearch
} // namespace unum
