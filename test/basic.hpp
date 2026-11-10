/**
 *  @file test/basic.hpp
 *  @author Ash Vardanian
 *  @date August 26, 2026
 *  @brief The use-case-agnostic test scenario library shared by every basic-index unit - sentences,
 *      bioinformatics sequences, and 3D meshes alike.
 *
 *  Every scenario is a template over a unit fixture supplying the index instantiation, a
 *  metric factory, a value generator, a value codec, and - unlike the dense tier, where
 *  the metric is library code - an @b exact brute-force oracle, so returned IDs and
 *  distances are asserted against ground truth rather than self-recall. Fixtures name
 *  their concrete index instantiations; this header names no index and assumes no
 *  constructor shape beyond a ledgered allocator.
 *
 *  Allocators are tested by construction: one counting allocator flows through the whole index -
 *  records, neighbor lists, and the key multimap - and through the values the fixtures build, as
 *  @c make_values and @c make_handmade take it last, tallying live bytes into a ledger that must
 *  balance to zero when each scenario's scope closes. Parallel scenarios use ForkUnion's default
 *  flat pool.
 *
 *  Every scenario keys its streams through @ref stream_key under the run's seed and its own name, like
 *  @c "starved"; a scenario's further streams take distinct indices, and per-thread streams index by thread.
 */
#pragma once

#include <cmath>   // `std::isfinite` - the sampled coordinates
#include <cstdint> // `std::uintptr_t` - per-process checkpoint paths
#include <cstdio>  // `std::snprintf`, `std::remove`, `std::rename` - the checkpoint files

#include <algorithm>   // `std::sort`, `std::unique`, `std::any_of`, `std::equal`
#include <array>       // `std::array` - handmade triples
#include <bit>         // `std::bit_cast` - planting an ID into a monolith
#include <limits>      // `std::numeric_limits` - the widest node ID and the unarmed budgets
#include <string>      // `std::string` - the metric token read back
#include <string_view> // `std::string_view` - dtype names
#include <thread>      // `std::this_thread::yield` - the gated metric's spin
#include <utility>     // `std::declval` - the thread bundle's metric type
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include <forkunion.hpp>

#include "harness.hpp"
#include "usearch/basic/ram_index.hpp"
#include "usearch/basic/ram_indexes.hpp"
#include "usearch/basic/nvme_indexes.hpp"
#include "usearch/utilities/safegraphs_format.hpp"

namespace unum::usearch::test {
using namespace unum::usearch::basic;

#pragma region Instrumented Metrics

/** Nodes in the single-threaded growth & oracle-recall scenarios. */
constexpr std::size_t basic_growth_nodes_k = 256;

/** Queries sampled against the exact oracle. */
constexpr std::size_t basic_recall_samples_k = 24;

/** Share of sampled queries whose best distance must match the oracle's. */
constexpr std::size_t basic_minimal_recall_percent_k = 90;

/** The floor for @c basic::keep_closest_t builds: closest-only wiring promises
 *  navigability, never the diverse selector's cross-cluster recall. */
constexpr std::size_t basic_relaxed_recall_percent_k = 50;

/** Threads in the concurrent scenarios. */
constexpr std::size_t basic_threads_k = 4;

/** Nodes inserted by each thread of the concurrent scenarios. */
constexpr std::size_t basic_thread_nodes_k = 96;

/** The selectors' neighbors-per-node budget across the scenarios. */
constexpr std::size_t basic_connectivity_k = 8;

/** The reader thread performs at least this many searches, even if writers finish first. */
constexpr std::size_t basic_reader_minimal_searches_k = 16;

/** Contract-auditing wrapper over any @c basic::metric: counts the verbs, verifies every
 *  drain is sized to that round's accepted pushes, flags abandoned pushed rounds, and can
 *  refuse pushes after an armed budget to drive every saturation path, or past a round cap to
 *  drive every fold. */
template <typename inner_metric_type_, typename distance_type_>
struct basic_auditing_metric {
    inner_metric_type_ inner;

    std::size_t resets = 0;
    std::size_t pushes = 0;
    std::size_t drains = 0;
    std::size_t pushes_this_round = 0;
    std::size_t refusals = 0;

    /** Contract breaches observed; every scenario asserts this stays zero. */
    std::size_t violations = 0;

    /** Accepted pushes before refusal begins; the type's maximum means unarmed. */
    std::size_t budget_left = (std::numeric_limits<std::size_t>::max)();

    /** Accepted pushes per round, refusing the rest until a drain; the type's maximum means
     *  uncapped. */
    std::size_t round_limit = (std::numeric_limits<std::size_t>::max)();

    template <typename query_type_>
    void reset(query_type_ const& query) noexcept {
        if (pushes_this_round) ++violations; // A pushed round was abandoned without a drain.
        inner.reset(query);
        ++resets;
        pushes_this_round = 0;
    }
    template <typename staged_type_>
    metric_staging_t push(staged_type_ const& staged) noexcept {
        if (!resets) ++violations; // A round was opened implicitly.
        if (!budget_left || pushes_this_round == round_limit) return ++refusals, metric_staging_t::refused_k;
        if (inner.push(staged) == metric_staging_t::refused_k) return ++refusals, metric_staging_t::refused_k;
        --budget_left, ++pushes, ++pushes_this_round;
        return metric_staging_t::staged_k;
    }
    void drain(std::span<distance_type_> distances, distance_type_ radius) noexcept {
        if (!resets) ++violations;
        if (distances.size() != pushes_this_round) ++violations; // The drain-width contract.
        inner.drain(distances, radius);
        ++drains;
        pushes_this_round = 0;
    }

    void arm(std::size_t budget) noexcept { budget_left = budget; }
    void disarm() noexcept { budget_left = (std::numeric_limits<std::size_t>::max)(); }
};

/** Wrapper whose first push parks until released - the deterministic way to hold
 *  one insert mid-flight, so the settled-past guards can be observed refusing. */
template <typename inner_metric_type_, typename distance_type_>
struct basic_gated_metric {
    inner_metric_type_ inner;
    bool* entered = nullptr;
    bool* released = nullptr;

    template <typename query_type_>
    void reset(query_type_ const& query) noexcept {
        inner.reset(query);
    }
    template <typename staged_type_>
    metric_staging_t push(staged_type_ const& staged) noexcept {
        fu::standard_atomic_ref<bool>(*entered).store(true, std::memory_order_release);
        while (!fu::standard_atomic_ref<bool>(*released).load(std::memory_order_acquire)) std::this_thread::yield();
        return inner.push(staged);
    }
    void drain(std::span<distance_type_> distances, distance_type_ radius) noexcept { inner.drain(distances, radius); }
};

#pragma endregion Instrumented Metrics

#pragma region Shared Helpers

/** Distances must ascend in every result list; answers the failures found. */
template <typename context_type_>
std::size_t basic_expect_ascending(context_type_ const& context) noexcept {
    std::size_t failures = 0;
    for (std::size_t position = 1; position < context.results.size(); ++position)
        failures += expect(!(context.results[position].distance < context.results[position - 1].distance));
    return failures;
}

/** The family a metric's distances compare in: an integral distance is exact, a floating one
 *  approximate at the fast-math tolerance. */
template <typename distance_type_>
constexpr comparison_family_t basic_comparison_family() noexcept {
    return std::is_integral_v<distance_type_> ? comparison_family_t::exact_k : comparison_family_t::approximate_k;
}

/** Inserts @p count values under fixture keys from @p first_key_ordinal in batches, collecting
 *  assigned IDs; the workhorse the single-threaded scenarios build graphs with. */
template <typename fixture_type_, typename index_type_, typename metric_type_, typename context_type_,
          typename selector_type_>
std::size_t basic_build(index_type_& index, metric_type_& metric, std::vector<typename index_type_::value_t>& values,
                        std::vector<typename index_type_::node_id_t>& assigned_ids, context_type_& context,
                        selector_type_ selector, typename index_type_::insert_config_t config = {},
                        std::size_t first_key_ordinal = 0) {
    using key_t = typename index_type_::key_t;
    constexpr std::size_t batch_size_k = 32;
    std::size_t const count = values.size();
    std::size_t failures = 0;
    assigned_ids.resize(count);
    std::vector<key_t> keys(count);
    for (std::size_t ordinal = 0; ordinal != count; ++ordinal)
        keys[ordinal] = fixture_type_::make_key(first_key_ordinal + ordinal);
    for (std::size_t begin = 0; begin < count; begin += batch_size_k) {
        std::size_t const batch = min_of(batch_size_k, count - begin);
        auto const [result, status] = index.insert({values.data() + begin, batch}, {keys.data() + begin, batch}, metric,
                                                   {assigned_ids.data() + begin, batch}, context, selector, config);
        failures += expect_eq(status, status_t::success_k);
    }
    return failures;
}

/** Per-thread state of the concurrent scenarios: distinctly-seeded values, the
 *  matching sequential keys, one context, and one metric per thread. */
template <typename fixture_type_>
struct basic_threads {
    using index_t = typename fixture_type_::index_t;
    std::vector<std::vector<typename index_t::value_t>> values;
    std::vector<std::vector<typename index_t::key_t>> keys;
    std::vector<typename index_t::traversal_context_t> contexts;
    std::vector<decltype(std::declval<fixture_type_ const&>().make_metric())> metrics;
};

/** Prepares @p threads_count threads: per-thread draws & contexts take the @p name stream at the
 *  thread's index, and keys run sequentially from @p first_key_ordinal across the threads. */
template <typename fixture_type_>
basic_threads<fixture_type_> basic_make_threads(allocation_ledger_t& ledger, std::size_t threads_count,
                                                std::size_t nodes_per_thread, u64_t seed, std::string_view name,
                                                std::size_t first_key_ordinal) {
    basic_threads<fixture_type_> threads;
    for (std::size_t thread = 0; thread != threads_count; ++thread) {
        threads.values.push_back(fixture_type_::make_values(nodes_per_thread, stream_key(seed, name, thread),
                                                            counting_allocator_t {ledger}));
        std::vector<typename fixture_type_::index_t::key_t> keys(nodes_per_thread);
        for (std::size_t ordinal = 0; ordinal != nodes_per_thread; ++ordinal)
            keys[ordinal] = fixture_type_::make_key(first_key_ordinal + thread * nodes_per_thread + ordinal);
        threads.keys.push_back(std::move(keys));
        threads.contexts.emplace_back(stream_key(seed, name, thread), counting_allocator_t {ledger});
        threads.metrics.push_back(fixture_type_::make_metric());
    }
    return threads;
}

/** One index mirrors another across the first @p nodes_count IDs: size, key,
 *  degree, and the stored value under the fixture's own equality. */
template <typename fixture_type_, typename index_type_>
std::size_t basic_expect_mirrored(index_type_& mirrored, index_type_& original, std::size_t nodes_count) {
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = expect_eq(mirrored.size(), original.size());
    for (std::size_t node = 0; node != nodes_count; ++node) {
        node_id_t const node_id = static_cast<node_id_t>(node);
        failures += expect(mirrored.key_of(node_id).value == original.key_of(node_id).value);
        failures += expect_eq(mirrored.degree_of(node_id).value, original.degree_of(node_id).value);
        failures += expect(fixture_type_::value_equals(mirrored.published_value_of(node_id), //
                                                       original.published_value_of(node_id)));
    }
    return failures;
}

/** The exact oracle's verdict on one search: every returned ID must carry its true
 *  distance, and the best must match the graph-independent minimum. */
template <typename fixture_type_, typename index_type_, typename query_type_>
std::size_t basic_audit_search(index_type_ const& index, query_type_ const& query,
                               typename index_type_::traversal_context_t const& context, bool& found_best) {
    using distance_t = typename index_type_::distance_t;
    std::size_t failures = 0;
    std::size_t const stored_count = index.size();
    error_stats_t distance_errors;
    for (std::size_t position = 0; position != context.results.size(); ++position) {
        auto const scored = context.results[position];
        failures += expect(static_cast<std::size_t>(scored.node_id) < stored_count);
        distance_errors.add(static_cast<f64_t>(scored.distance),
                            static_cast<f64_t>(fixture_type_::oracle(query, index.published_value_of(scored.node_id))));
    }
    failures += expect_errors("audited distances", basic_comparison_family<distance_t>(),
                              comparison_verdict_t::assert_k, distance_errors, 0);
    distance_t truly_nearest = fixture_type_::oracle(query, index.published_value_of(0));
    for (std::size_t node = 1; node != stored_count; ++node) {
        distance_t const candidate = fixture_type_::oracle(
            query, index.published_value_of(static_cast<typename index_type_::node_id_t>(node)));
        if (candidate < truly_nearest) truly_nearest = candidate;
    }
    if constexpr (std::is_integral_v<typename index_type_::distance_t>)
        found_best = context.results.size() && context.results[0].distance == truly_nearest;
    else
        found_best = context.results.size() &&
                     static_cast<f64_t>(context.results[0].distance) <= static_cast<f64_t>(truly_nearest) + 1e-6;
    return failures;
}

/** The five columns of one snapshot, each owned by a vector - the container every
 *  scenario saves into and loads from, sized from the index's own extents. Equality is the
 *  byte-for-byte proof that two saves of a settled graph agree. */
template <typename index_type_>
struct basic_columns_storage {
    using key_t = typename index_type_::key_t;
    using node_id_t = typename index_type_::node_id_t;
    using columns_t = typename index_type_::columns_t;
    using columns_view_t = typename index_type_::columns_view_t;

    std::vector<key_t> keys;
    std::vector<u64_t> neighbor_offsets;
    std::vector<node_id_t> neighbor_ids;
    std::vector<u64_t> value_offsets;
    std::vector<std::byte> values;

    basic_columns_storage() = default;
    explicit basic_columns_storage(columns_extents_t extents, std::byte fill = std::byte {})
        : keys(extents.nodes), neighbor_offsets(extents.nodes + 1), neighbor_ids(extents.neighbors),
          value_offsets(extents.nodes + 1), values(extents.value_bytes, fill) {}

    columns_t columns() noexcept { return {keys, neighbor_offsets, neighbor_ids, value_offsets, values}; }
    columns_view_t view() const noexcept { return {keys, neighbor_offsets, neighbor_ids, value_offsets, values}; }
    bool operator==(basic_columns_storage const& other) const {
        return keys == other.keys && neighbor_offsets == other.neighbor_offsets && neighbor_ids == other.neighbor_ids &&
               value_offsets == other.value_offsets && values == other.values;
    }
};

/** Sizes fresh columns from @p index and saves into them in one call. */
template <typename index_type_, typename codec_type_>
expected<basic_columns_storage<index_type_>> basic_saved_columns(index_type_& index, codec_type_ const& codec) {
    auto const [extents, status] = index.columns_extents(codec);
    if (failed(status)) return {{}, status};
    basic_columns_storage<index_type_> storage {extents};
    status_t const saved = index.save(storage.columns(), codec);
    return {std::move(storage), saved};
}

/** A ForkUnion flat pool as the glue's executor: one region per thread when the
 *  counts agree, the calling thread otherwise. */
struct pooled_executor_t {
    fu::flat_pool_t pool;
    std::size_t threads = 0;

    status_t spawn(std::size_t count) noexcept {
        if (fu::failed(pool.spawn(count))) return status_t::bad_alloc_k;
        threads = count;
        return status_t::success_k;
    }
    std::size_t size() const noexcept { return threads; }
    template <typename callback_type_>
    status_t for_tasks(std::size_t tasks, callback_type_&& callback) noexcept {
        if (tasks != threads) {
            for (std::size_t task = 0; task != tasks; ++task) callback(task, 0, 0);
            return status_t::success_k;
        }
        pool.for_threads([&](std::size_t thread) noexcept { callback(thread, thread, 0); });
        return status_t::success_k;
    }
};

#pragma endregion Shared Helpers

#pragma region Foundations

/** An empty index answers every verb with no reserve, argument guards refuse with zero side
 *  effects, inserts cover their own nodes, and @c reserve covers ahead at any time with no
 *  ceiling below the ID space. */
template <typename fixture_type_>
std::size_t test_basic_guards(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t batch_k = 8;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "guards"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(batch_k + 1, stream_key(seed, "guards"),
                                                 counting_allocator_t {ledger});
        std::vector<key_t> keys(batch_k + 1);
        for (std::size_t ordinal = 0; ordinal != keys.size(); ++ordinal)
            keys[ordinal] = fixture_type_::make_key(ordinal);

        // An empty index answers everything with nothing covered, and searches find nothing.
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index.capacity(), 0u);
        failures += expect_eq(index.memory_usage(), 0u);
        failures += expect(index.get_allocator() == counting_allocator_t {ledger});
        auto const [empty_result, empty_status] = index.search(values[0], {}, metric, context);
        failures += expect_eq(empty_status, status_t::success_k);
        failures += expect_eq(empty_result.count, 0u);
        failures += expect_eq(index.find(keys[0]).status, status_t::malformed_k);
        failures += expect(!index.contains(keys[0]));
        failures += expect_eq(index.count(keys[0]), 0u);
        failures += expect_eq(index.key_of(0).status, status_t::malformed_k);
        failures += expect_eq(index.value_of(0).status, status_t::malformed_k);
        failures += expect_eq(index.degree_of(0).status, status_t::malformed_k);
        failures += expect_eq(index.link(0, 1).status, status_t::malformed_k);
        failures += expect_eq(index.unlink(0, 1), status_t::malformed_k);

        // Argument guards, each with zero side effects.
        std::vector<node_id_t> assigned_ids(1);
        failures += expect_eq(
            index.insert({values.data(), 2}, {keys.data(), 2}, metric, {assigned_ids.data(), 1}, context).status,
            status_t::config_mismatch_k);
        failures += expect_eq(index.insert({values.data(), 2}, {keys.data(), 1}, metric, {}, context).status,
                              status_t::config_mismatch_k);
        failures += expect_eq(
            index.insert({values.data(), 1}, {keys.data(), 1}, metric, {}, context, keep_closest_t {0}).status,
            status_t::config_mismatch_k);
        failures += expect_eq(index.insert({values.data(), 0}, {keys.data(), 0}, metric, {}, context).status,
                              status_t::success_k);
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index.capacity(), 0u);

        // With no reserve, an insert covers the nodes it claims.
        failures += expect_eq(index
                                  .insert({values.data(), batch_k}, {keys.data(), batch_k}, metric, {}, context,
                                          keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        failures += expect_eq(index.size(), batch_k);
        failures += expect(index.capacity() >= batch_k);
        failures += expect(index.memory_usage() > 0);

        // A reserve covers ahead at any time and answers the capacity covered; past the ID ceiling
        // it overflows and covers nothing more.
        std::size_t const covered_before = index.capacity();
        auto const [capacity, status] = index.reserve(covered_before + 1);
        failures += expect_eq(status, status_t::success_k);
        failures += expect(capacity > covered_before);
        failures += expect_eq(index.capacity(), capacity);
        failures += expect_eq(index.reserve(0).value, capacity);
        failures += expect_eq(index.reserve(index_t::max_node_count_k + 1).status, status_t::overflow_k);
        failures += expect_eq(index.capacity(), capacity);

        // A ceiling at the current size refuses the next entry whole, and a plain reserve lifts it.
        failures += expect_eq(index.reserve({0, batch_k}).status, status_t::success_k);
        failures += expect_eq(
            index.insert({values.data() + batch_k, 1}, {keys.data() + batch_k, 1}, metric, {}, context).status,
            status_t::overflow_k);
        failures += expect_eq(index.size(), batch_k);
        failures += expect_eq(index.reserve(0).status, status_t::success_k);
        failures += expect_eq(
            index.insert({values.data() + batch_k, 1}, {keys.data() + batch_k, 1}, metric, {}, context).status,
            status_t::success_k);
        failures += expect_eq(index.size(), batch_k + 1);

        // The index's storage is provably on the ledger, not the plain heap.
        failures += expect(ledger.live_bytes > 0);
        failures += expect(ledger.total_allocations > 0);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Three values whose pairwise distances the unit states by hand: the oracle must
 *  reproduce the stated matrix, and every value must find itself at distance zero. */
template <typename fixture_type_>
std::size_t test_basic_handmade(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "handmade"), counting_allocator_t {ledger}};
        auto handmade = fixture_type_::make_handmade(counting_allocator_t {ledger});
        // Deterministic, so a fresh unmoved copy.
        auto queries = fixture_type_::make_handmade(counting_allocator_t {ledger});

        failures += expect(bool(index.reserve(3)));
        std::array<typename index_t::key_t, 3> keys {fixture_type_::make_key(0), fixture_type_::make_key(1),
                                                     fixture_type_::make_key(2)};
        std::array<typename index_t::node_id_t, 3> assigned_ids {};
        failures += expect_eq(index.insert({handmade.values}, {keys}, metric, {assigned_ids}, context).status,
                              status_t::success_k);

        using distance_t = typename index_t::distance_t;
        error_stats_t oracle_errors, search_errors;
        for (std::size_t query = 0; query != 3; ++query)
            for (std::size_t entry = 0; entry != 3; ++entry)
                oracle_errors.add(static_cast<f64_t>(fixture_type_::oracle(
                                      queries.values[query], index.published_value_of(assigned_ids[entry]))),
                                  static_cast<f64_t>(handmade.distances[query][entry]));

        for (std::size_t query = 0; query != 3; ++query) {
            auto const [result, status] = index.search(queries.values[query], {}, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += expect(result.count >= 3);
            search_errors.add(static_cast<f64_t>(context.results[0].distance), 0.0);
            distance_t runner_up = handmade.distances[query][(query + 1) % 3];
            if (handmade.distances[query][(query + 2) % 3] < runner_up)
                runner_up = handmade.distances[query][(query + 2) % 3];
            search_errors.add(static_cast<f64_t>(context.results[1].distance), static_cast<f64_t>(runner_up));
        }
        failures += expect_errors("handmade oracle", basic_comparison_family<distance_t>(),
                                  comparison_verdict_t::assert_k, oracle_errors, 0);
        failures += expect_errors("handmade nearest", basic_comparison_family<distance_t>(),
                                  comparison_verdict_t::assert_k, search_errors, 0);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Fills the index, then samples queries against the exact brute-force oracle:
 *  returned distances must be true, the best must be the true best for at least the
 *  recall bar, and the telemetry must have counted the work. */
template <typename fixture_type_>
std::size_t test_basic_growth_and_recall(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "growth"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(basic_growth_nodes_k + basic_recall_samples_k,
                                                 stream_key(seed, "growth"), counting_allocator_t {ledger});
        std::vector<typename index_t::value_t> queries;
        for (std::size_t sample = 0; sample != basic_recall_samples_k; ++sample)
            queries.push_back(std::move(values[basic_growth_nodes_k + sample]));
        values.resize(basic_growth_nodes_k);

        failures += expect(bool(index.reserve(basic_growth_nodes_k)));
        std::vector<typename index_t::node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});
        failures += expect_eq(index.size(), basic_growth_nodes_k);

        // A single-threaded build dispenses a contiguous ID window in submission order.
        for (std::size_t ordinal = 0; ordinal != assigned_ids.size(); ++ordinal)
            failures += expect_eq(static_cast<std::size_t>(assigned_ids[ordinal]), ordinal);
        failures += expect(context.telemetry.computed_distances > 0);
        failures += expect(context.telemetry.visited_members > 0);

        std::size_t found_best_count = 0;
        for (auto const& query : queries) {
            auto const [result, status] = index.search(query, {}, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += expect(result.count > 0 && result.count <= connectivity_32_k * 2);
            failures += basic_expect_ascending(context);
            bool found_best = false;
            failures += basic_audit_search<fixture_type_>(index, query, context, found_best);
            found_best_count += found_best;
        }
        if (found_best_count * 100 < basic_minimal_recall_percent_k * basic_recall_samples_k)
            fmt::print(stderr, "  recall diagnostics: {} of {} queries found the true best\n", found_best_count,
                       basic_recall_samples_k);
        failures += expect(found_best_count * 100 >= basic_minimal_recall_percent_k * basic_recall_samples_k);

        // An explicit breadth, and candidates covering the whole graph, whose best is the true best
        // by construction.
        for (auto const config : {typename index_t::search_config_t {32}, //
                                  typename index_t::search_config_t {basic_growth_nodes_k}}) {
            auto const [result, status] = index.search(queries[0], config, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(context);
        }
        typename index_t::search_config_t const exhaustive_config {basic_growth_nodes_k};
        for (std::size_t sample = 0; sample != 4; ++sample) {
            failures += expect_eq(index.search(queries[sample], exhaustive_config, metric, context).status,
                                  status_t::success_k);
            bool found_best = false;
            failures += basic_audit_search<fixture_type_>(index, queries[sample], context, found_best);
            failures += expect(found_best); // Whole-graph candidates cannot miss.
        }

        // A scan is exact by construction, and a filter keeps only what it admits, still as deep.
        auto even = quad_predicate([](typename index_t::node_id_t node_id) noexcept {
            return node_id % 2 ? admission_t::reject_k : admission_t::admit_k;
        });
        dummy_predicate_t unfiltered;
        for (std::size_t sample = 0; sample != 4; ++sample) {
            auto const [scanned, status] = index.scan(queries[sample], 10, unfiltered, unfiltered, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += expect_eq(scanned.count, std::size_t {10});
            failures += basic_expect_ascending(context);
            bool found_best = false;
            failures += basic_audit_search<fixture_type_>(index, queries[sample], context, found_best);
            failures += expect(found_best);
            failures += expect_eq(index.scan(queries[sample], 10, unfiltered, even, metric, context).value.count,
                                  std::size_t {10});
            for (std::size_t position = 0; position != context.results.size(); ++position)
                failures += expect_eq(context.results[position].node_id % 2, 0u);

            // A filtered walk routes through every node it scores and keeps only what it admits,
            // filling its whole breadth.
            failures += expect_eq(index.search_if(queries[sample], {32}, unfiltered, even, metric, context).status,
                                  status_t::success_k);
            failures += expect_eq(context.results.size(), std::size_t {32});
            for (std::size_t position = 0; position != context.results.size(); ++position)
                failures += expect_eq(context.results[position].node_id % 2, 0u);
        }
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** The same data built through both shipped selectors - the per-call budget makes
 *  this one index instantiation, two graphs. The diverse build must clear the full
 *  recall bar, the closest build its navigability floor, and the wiring must differ
 *  structurally, or the selector argument is being ignored. */
template <typename fixture_type_>
std::size_t test_basic_selectors(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    auto build_and_audit_recall = [&](auto selector, std::size_t minimal_recall_percent,
                                      typename index_t::graph_health_t& health) {
        allocation_ledger_t ledger;
        {
            std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
            index_t& index = *index_ptr;
            auto metric = fixture_type_::make_metric();
            context_t context {stream_key(seed, "selectors"), counting_allocator_t {ledger}};
            // Queries split off the same draw as the stored values: a separate generator
            // call would plant different clusters, leaving every query off-distribution.
            auto values = fixture_type_::make_values(basic_growth_nodes_k / 2 + basic_recall_samples_k,
                                                     stream_key(seed, "selectors"), counting_allocator_t {ledger});
            std::vector<typename index_t::value_t> queries;
            for (std::size_t sample = 0; sample != basic_recall_samples_k; ++sample)
                queries.push_back(std::move(values[basic_growth_nodes_k / 2 + sample]));
            values.resize(basic_growth_nodes_k / 2);

            failures += expect(bool(index.reserve(values.size())));
            std::vector<typename index_t::node_id_t> assigned_ids;
            failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, selector);
            health = index.health(context);
            failures += expect_eq(health.status, status_t::success_k);
            failures += expect(health.degree_min >= 1);

            std::size_t found_best_count = 0;
            for (auto const& query : queries) {
                auto const [result, status] = index.search(query, {}, metric, context);
                failures += expect_eq(status, status_t::success_k);
                bool found_best = false;
                failures += basic_audit_search<fixture_type_>(index, query, context, found_best);
                found_best_count += found_best;
            }
            if (found_best_count * 100 < minimal_recall_percent * queries.size())
                fmt::print(stderr, "  recall diagnostics: {} of {} queries found the true best, mean degree {}\n",
                           found_best_count, queries.size(), health.mean_degree);
            failures += expect(found_best_count * 100 >= minimal_recall_percent * queries.size());
        }
        failures += expect_eq(ledger.live_bytes, 0);
    };

    typename index_t::graph_health_t closest_health, diverse_health;
    build_and_audit_recall(keep_closest_t {basic_connectivity_k}, basic_relaxed_recall_percent_k, closest_health);
    build_and_audit_recall(keep_diverse_t {basic_connectivity_k}, basic_minimal_recall_percent_k, diverse_health);
    failures += expect(closest_health.mean_degree != diverse_health.mean_degree ||
                       closest_health.reciprocal_edges != diverse_health.reciprocal_edges);
    return failures;
}

/** Keyed lookups with planted duplicates, ID → key staying authoritative, and the
 *  checked ID-readers answering @c malformed_k at & past the watermark. */
template <typename fixture_type_>
std::size_t test_basic_keys(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using key_t = typename index_t::key_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t unique_k = nodes_k / 2;
    constexpr std::size_t shared_keys_k = 4;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "keys"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "keys"), counting_allocator_t {ledger});
        std::vector<key_t> keys(nodes_k);
        for (std::size_t ordinal = 0; ordinal != unique_k; ++ordinal) keys[ordinal] = fixture_type_::make_key(ordinal);
        for (std::size_t ordinal = unique_k; ordinal != nodes_k; ++ordinal)
            keys[ordinal] = fixture_type_::make_key(1000 + ordinal % shared_keys_k);

        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids(nodes_k);
        failures += expect_eq(
            index.insert({values}, {keys}, metric, {assigned_ids}, context, keep_diverse_t {basic_connectivity_k})
                .status,
            status_t::success_k);

        std::vector<std::size_t> all_found_ids;
        for (std::size_t ordinal = 0; ordinal != unique_k; ++ordinal) {
            failures += expect(index.contains(keys[ordinal]));
            failures += expect_eq(index.count(keys[ordinal]), 1u);
            auto const [match, status] = index.find(keys[ordinal]);
            failures += expect(succeeded(status));
            failures += expect(index.key_of(match).value == keys[ordinal]);
            all_found_ids.push_back(static_cast<std::size_t>(match));
        }
        for (std::size_t shared = 0; shared != shared_keys_k; ++shared) {
            key_t const key = fixture_type_::make_key(1000 + shared);
            failures += expect_eq(index.count(key), (nodes_k - unique_k) / shared_keys_k);
            std::vector<std::size_t> holders;
            index.find_each(key, [&](node_id_t node_id) noexcept {
                holders.push_back(static_cast<std::size_t>(node_id));
                failures += expect(index.key_of(node_id).value == key);
            });
            failures += expect_eq(holders.size(), (nodes_k - unique_k) / shared_keys_k);
            auto const one_of = index.find(key);
            failures += expect(bool(one_of));
            all_found_ids.insert(all_found_ids.end(), holders.begin(), holders.end());
        }
        failures += expect_eq(all_found_ids.size(), nodes_k);
        failures += expect_contiguous_ids(std::span {all_found_ids}, 0);

        key_t const absent = fixture_type_::make_key(5000);
        failures += expect_eq(index.find(absent).status, status_t::malformed_k);
        failures += expect(!index.contains(absent));
        failures += expect_eq(index.count(absent), 0u);

        // The checked ID-readers gate every input on the watermark.
        failures += expect(bool(index.key_of(0)));
        failures += expect(bool(index.value_of(0)));
        failures += expect(bool(index.degree_of(0)));
        node_id_t const first_unpublished = static_cast<node_id_t>(index.size());
        failures += expect_eq(index.key_of(first_unpublished).status, status_t::malformed_k);
        failures += expect_eq(index.value_of(first_unpublished).status, status_t::malformed_k);
        failures += expect_eq(index.degree_of(first_unpublished).status, status_t::malformed_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Manual wiring: idempotent links, honest rejections, degree bookkeeping, growth
 *  past the build budget, and a snapshot roundtrip preserving variable degrees. */
template <typename fixture_type_>
std::size_t test_basic_wiring(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 24;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "wiring"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "wiring"), counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, keep_diverse_t {4});

        node_id_t const from = 0, to = static_cast<node_id_t>(nodes_k - 1);
        [[maybe_unused]] status_t const cleared = index.unlink(from, to); // Absent or not, now absent.
        std::size_t const degree_before = index.degree_of(from).value;

        auto const [membership, link_status] = index.link(from, to);
        failures += expect_eq(link_status, status_t::success_k);
        failures += expect(membership == membership_t::absent_k);
        failures += expect_eq(index.degree_of(from).value, degree_before + 1);
        auto const [relink_membership, relink_status] = index.link(from, to);
        failures += expect_eq(relink_status, status_t::success_k);
        failures += expect(relink_membership == membership_t::present_k);
        failures += expect_eq(index.degree_of(from).value, degree_before + 1);
        failures += expect_eq(index.unlink(from, to), status_t::success_k);
        failures += expect_eq(index.degree_of(from).value, degree_before);
        failures += expect_eq(index.unlink(from, to), status_t::malformed_k);

        failures += expect_eq(index.link(from, from).status, status_t::malformed_k);
        failures += expect_eq(index.link(from, static_cast<node_id_t>(nodes_k)).status, status_t::malformed_k);
        failures += expect_eq(index.link(static_cast<node_id_t>(nodes_k), from).status, status_t::malformed_k);
        failures += expect_eq(index.unlink(from, static_cast<node_id_t>(nodes_k)), status_t::malformed_k);

        // Over-wire one node far past the build budget - neighbor lists are unbounded.
        for (std::size_t target = 1; target != 13; ++target)
            failures += expect(succeeded(index.link(from, static_cast<node_id_t>(target)).status));
        failures += expect(index.degree_of(from).value >= 12);
        failures += expect_eq(index.search(values[0], {}, metric, context).status, status_t::success_k);
        failures += expect_eq(index.health(context).status, status_t::success_k);

        // Variable degrees survive the snapshot as they are - each neighbor list's own prefix sums
        // say its degree.
        auto const [columns, save_status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(save_status));
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect(bool(restored.reserve(nodes_k)));
        failures += expect_eq(restored.load(columns.view(), codec), status_t::success_k);
        failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Growth with no reserve: inserts cross three node chunks of 1'024 nodes, then twice as many per
 *  chunk, the nodes on both sides of each boundary keep their keys and values, and every key, filed
 *  in the multimap the batches' own claims covered, counts once; then @c clear forgets every node
 *  and key but keeps the chunks, and a rebuild restarts from ID zero. */
template <typename fixture_type_>
std::size_t test_basic_unreserved_growth(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t first_chunk_k = 1024;
    constexpr std::size_t nodes_k = first_chunk_k * 3 + 64; // Past the second chunk, into the third.
    constexpr std::size_t batch_k = 64;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "unreserved"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "unreserved"),
                                                 counting_allocator_t {ledger});
        std::vector<key_t> keys(nodes_k);
        for (std::size_t ordinal = 0; ordinal != nodes_k; ++ordinal) keys[ordinal] = fixture_type_::make_key(ordinal);
        std::vector<node_id_t> assigned_ids(nodes_k);

        // Narrow candidates keep the many walks cheap; the chunks are what this scenario is about.
        typename index_t::insert_config_t const narrow_config {8};
        for (std::size_t first = 0; first != nodes_k; first += batch_k)
            failures += expect_eq(
                index
                    .insert(std::span(values).subspan(first, batch_k),
                            std::span<key_t const>(keys).subspan(first, batch_k), metric,
                            std::span(assigned_ids).subspan(first, batch_k), context, keep_closest_t {4}, narrow_config)
                    .status,
                status_t::success_k);
        failures += expect_eq(index.size(), nodes_k);
        std::size_t const grown_capacity = index.capacity();
        failures += expect(grown_capacity >= nodes_k);
        for (std::size_t ordinal = 0; ordinal != nodes_k; ++ordinal)
            failures += expect_eq(static_cast<std::size_t>(assigned_ids[ordinal]), ordinal);

        // The insert moved the originals, so the entries compare against a fresh identical draw.
        auto expected_values = fixture_type_::make_values(nodes_k, stream_key(seed, "unreserved"),
                                                          counting_allocator_t {ledger});
        for (std::size_t const offset : {std::size_t {0}, first_chunk_k - 1, first_chunk_k, first_chunk_k * 3 - 1,
                                         first_chunk_k * 3, nodes_k - 1}) {
            node_id_t const node_id = static_cast<node_id_t>(offset);
            failures += expect(index.key_of(node_id).value == keys[offset]);
            failures += expect(index.find(keys[offset]).value == node_id);
            failures += expect(fixture_type_::value_equals(index.published_value_of(node_id), expected_values[offset]));
        }
        std::size_t miscounted = 0;
        for (key_t const& key : keys) miscounted += index.count(key) != 1;
        failures += expect_eq(miscounted, 0u);
        // Whole-graph candidates cannot miss, wherever the best entry lies.
        auto const [found, found_status] = index.search(expected_values[nodes_k - 1],
                                                        typename index_t::search_config_t {nodes_k}, metric, context);
        failures += expect_eq(found_status, status_t::success_k);
        bool found_best = false;
        failures += basic_audit_search<fixture_type_>(index, expected_values[nodes_k - 1], context, found_best);
        failures += expect(found_best);

        // Clear keeps every chunk and forgets everything else.
        std::size_t const grown_memory = index.memory_usage();
        failures += expect_eq(index.clear(), status_t::success_k);
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index.capacity(), grown_capacity);
        failures += expect_eq(index.memory_usage(), grown_memory);
        failures += expect(!index.contains(keys[0]));
        failures += expect_eq(index.key_of(0).status, status_t::malformed_k);

        // A rebuild restarts from ID zero over the kept chunks, and only its own keys are findable.
        auto rebuilt_values = fixture_type_::make_values(batch_k, stream_key(seed, "unreserved", 1),
                                                         counting_allocator_t {ledger});
        std::vector<node_id_t> rebuilt_ids(batch_k);
        failures += expect_eq(index
                                  .insert(std::span(rebuilt_values), std::span<key_t const>(keys).first(batch_k),
                                          metric, std::span(rebuilt_ids), context, keep_closest_t {4})
                                  .status,
                              status_t::success_k);
        failures += expect_eq(static_cast<std::size_t>(rebuilt_ids[0]), 0u);
        failures += expect_eq(index.size(), batch_k);
        failures += expect_eq(index.count(keys[0]), 1u);
        failures += expect(!index.contains(keys[batch_k]));
        failures += expect_eq(index.capacity(), grown_capacity);
        failures += expect_eq(index.health(context).status, status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Foundations

#pragma region Allocation Budgets

/** All-or-nothing under a starved ledger: a refused reserve covers nothing & retries, a
 *  refused insert leaves the index empty & retries, and a refused link-growth leaves the
 *  neighbor list untouched. */
template <typename fixture_type_>
std::size_t test_basic_allocation_budget(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "budget"), counting_allocator_t {ledger}};

        ledger.byte_budget = ledger.live_bytes + 64;
        failures += expect_eq(index.reserve(basic_growth_nodes_k).status, status_t::bad_alloc_k);
        failures += expect_eq(index.capacity(), 0u);
        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        failures += expect_eq(index.reserve(basic_growth_nodes_k).status, status_t::success_k);

        // A budget swept up from the live bytes refuses an insert's allocations one by one: the key
        // multimap's, then the node chunk's, which a step far below its size refuses at least once.
        // Every refusal, the cover's included, leaves the index empty and its claim given back,
        // until the first success dispenses ID zero.
        std::unique_ptr<index_t> const swept_ptr = fixture_type_::make_index(ledger);
        index_t& swept = *swept_ptr;
        auto swept_values = fixture_type_::make_values(1, stream_key(seed, "budget", 1), counting_allocator_t {ledger});
        std::array<key_t, 1> const swept_keys {fixture_type_::make_key(0)};
        std::array<node_id_t, 1> swept_ids {node_id_t {1}};
        failures += expect_eq(swept.search(swept_values[0], {}, metric, context).status, status_t::success_k); // Warm.
        constexpr std::ptrdiff_t sweep_step_k = 4096;
        std::ptrdiff_t const swept_base = ledger.live_bytes;
        std::size_t refusals = 0;
        status_t swept_status = status_t::bad_alloc_k;
        for (std::ptrdiff_t slack = 0; swept_status == status_t::bad_alloc_k && slack < (std::ptrdiff_t {1} << 30);
             slack += sweep_step_k) {
            ledger.byte_budget = swept_base + slack;
            swept_status =
                swept.insert({swept_values}, {swept_keys}, metric, {swept_ids}, context, keep_closest_t {1}).status;
            if (swept_status != status_t::bad_alloc_k) break;
            ++refusals;
            failures += expect_eq(swept.size(), 0u);
            failures += expect_eq(swept.capacity(), 0u);
        }
        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        failures += expect_eq(swept_status, status_t::success_k);
        failures += expect(refusals > 0);
        failures += expect_eq(static_cast<std::size_t>(swept_ids[0]), 0u);
        failures += expect_eq(swept.size(), 1u);
        failures += expect(swept.contains(swept_keys[0]));

        constexpr std::size_t linked_nodes_k = 40;
        auto values = fixture_type_::make_values(linked_nodes_k, stream_key(seed, "budget"),
                                                 counting_allocator_t {ledger});
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, keep_closest_t {1});

        // Amortized neighbor list growth leaves power-of-two slack, so links absorb it for free -
        // but the slack is finite, and a starved ledger must refuse the first real growth without
        // touching the list.
        std::size_t const degree_before = index.degree_of(0).value;
        ledger.byte_budget = ledger.live_bytes;
        std::size_t fresh_links = 0;
        node_id_t refused_target = 0;
        for (std::size_t target = 1; target != linked_nodes_k && !refused_target; ++target) {
            auto const [membership, status] = index.link(0, static_cast<node_id_t>(target));
            if (status == status_t::bad_alloc_k) refused_target = static_cast<node_id_t>(target);
            else if (succeeded(status) && membership == membership_t::absent_k) ++fresh_links;
        }
        failures += expect(refused_target != 0);
        failures += expect_eq(index.degree_of(0).value, degree_before + fresh_links);

        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        auto const [retry_membership, retry_status] = index.link(0, refused_target);
        failures += expect_eq(retry_status, status_t::success_k);
        failures += expect(retry_membership == membership_t::absent_k);
        failures += expect_eq(index.degree_of(0).value, degree_before + fresh_links + 1);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** The staging a warm @c {8} walk leaves in a fresh context on a graph of @p nodes_count nodes
 *  built like the starved scenario's - what its over-wired neighbor list must outgrow, decided by
 *  the growth policy and the fixture's values rather than a constant. */
template <typename fixture_type_>
std::size_t basic_warm_staging_capacity(u64_t seed, std::size_t nodes_count, allocation_ledger_t& ledger,
                                        std::size_t& failures) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
    index_t& index = *index_ptr;
    auto metric = fixture_type_::make_metric();
    context_t context {stream_key(seed, "starved"), counting_allocator_t {ledger}};
    auto values = fixture_type_::make_values(nodes_count, stream_key(seed, "starved"), counting_allocator_t {ledger});
    std::vector<typename index_t::node_id_t> assigned_ids;
    failures += expect(bool(index.reserve(nodes_count)));
    failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, keep_closest_t {4});
    auto query = fixture_type_::make_values(1, stream_key(seed, "starved", 3), counting_allocator_t {ledger});
    context_t small_context {stream_key(seed, "starved", 1), counting_allocator_t {ledger}};
    failures += expect_eq(index.search(query[0], {8}, metric, small_context).status, status_t::success_k);
    return small_context.staged_ids.capacity();
}

/** The mid-flight allocation-failure battery: every fallible allocation inside a
 *  running operation keeps its stated consistency story - thinner nodes publish,
 *  refused staging saturates, cold scratch refuses all-or-nothing, and a starved
 *  load stays retryable. The warm-scratch trick isolates each site: a reused context
 *  already holds its buffers, so a frozen ledger budget starves exactly one thing. */
template <typename fixture_type_>
std::size_t test_basic_starved(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t base_nodes_k = 96;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    std::size_t const nodes_count = max_of(
        base_nodes_k, basic_warm_staging_capacity<fixture_type_>(seed, base_nodes_k, ledger, failures) + 2);
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t warm_context {stream_key(seed, "starved"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_count, stream_key(seed, "starved"),
                                                 counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_count + 2)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, warm_context, keep_closest_t {4});

        auto extras = fixture_type_::make_values(2, stream_key(seed, "starved", 4), counting_allocator_t {ledger});
        std::array<key_t, 2> const extra_keys {fixture_type_::make_key(nodes_count),
                                               fixture_type_::make_key(nodes_count + 1)};
        auto query = fixture_type_::make_values(1, stream_key(seed, "starved", 3), counting_allocator_t {ledger});

        // Warm a small round shape, then over-wire node 0 past its staging capacity, so a later
        // frozen-budget search must refuse growing its staging to that list. The backward edges are
        // what make the fat node reachable: a walk only expands what some other node points at.
        context_t small_context {stream_key(seed, "starved", 1), counting_allocator_t {ledger}};
        failures += expect_eq(index.search(query[0], {8}, metric, small_context).status, status_t::success_k);
        for (std::size_t target = 1; target != nodes_count; ++target) {
            [[maybe_unused]] auto const outward = index.link(0, static_cast<node_id_t>(target));
            [[maybe_unused]] auto const inward = index.link(static_cast<node_id_t>(target), 0);
        }
        // The premise: the fat neighbor list outgrows the staging the warm walk left behind,
        // whatever its growth rounds to.
        failures += expect(index.degree_of(0).value > small_context.staged_ids.capacity());

        // Cold scratch under a frozen budget: all-or-nothing refusals, nothing moves.
        context_t cold_context {stream_key(seed, "starved", 2), counting_allocator_t {ledger}};
        ledger.byte_budget = ledger.live_bytes;
        failures += expect_eq(index.search(query[0], {}, metric, cold_context).status, status_t::bad_alloc_k);
        failures += expect_eq(index.insert({extras.data(), 1}, {extra_keys.data(), 1}, metric, {}, cold_context).status,
                              status_t::bad_alloc_k);
        failures += expect_eq(index.size(), nodes_count);
        failures += expect_eq(index.health(cold_context).status, status_t::bad_alloc_k);

        // Warm scratch under the same budget: the claimed ID must still publish, only thinner - the
        // fresh neighbor list's growth is the one refused allocation, and it is counted.
        auto const [thinner, thinner_status] = index.insert({extras.data(), 1}, {extra_keys.data(), 1}, metric, {},
                                                            warm_context, keep_closest_t {4});
        failures += expect_eq(thinner_status, status_t::success_k);
        failures += expect(thinner.telemetry.saturated_walks > 0);
        failures += expect_eq(index.size(), nodes_count + 1);
        failures += expect_eq(index.degree_of(static_cast<node_id_t>(nodes_count)).value, 0u);

        // A search stages one round of the widest neighbor list, sized up front: the warm context,
        // too small for the over-wired list, refuses all-or-nothing rather than saturating partway.
        failures += expect_eq(index.search(query[0], {8}, metric, small_context).status, status_t::bad_alloc_k);

        // Lifting the budget makes every refusal retryable.
        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        failures += expect_eq(index.health(warm_context).status, status_t::success_k);
        failures += expect_eq(
            index.insert({extras.data() + 1, 1}, {extra_keys.data() + 1, 1}, metric, {}, cold_context).status,
            status_t::success_k);
        failures += expect_eq(index.size(), nodes_count + 2);

        // A starved load fails mid-decode with the watermark unpublished, then retries.
        auto const [columns, status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(status));
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect(bool(restored.reserve(nodes_count + 2)));
        ledger.byte_budget = ledger.live_bytes;
        failures += expect_eq(restored.load(columns.view(), codec), status_t::bad_alloc_k);
        failures += expect_eq(restored.size(), 0u);
        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        failures += expect_eq(restored.load(columns.view(), codec), status_t::success_k);
        failures += expect_eq(restored.size(), index.size());
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Allocation Budgets

#pragma region Snapshots

/** Snapshot roundtrip: sized exactly, refused when short, loaded into a fresh
 *  index that answers every query identically and mirrors every ID's key & degree. */
template <typename fixture_type_>
std::size_t test_basic_snapshot_roundtrip(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using key_t = typename index_t::key_t; // Shadows POSIX ::key_t, which binds otherwise.
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 48;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "snapshot"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "snapshot"), counting_allocator_t {ledger});
        auto queries = fixture_type_::make_values(8, stream_key(seed, "snapshot", 2), counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto const [extents, measure_status] = index.columns_extents(codec);
        failures += expect(succeeded(measure_status));
        auto const [resized_extents, resized_status] = index.columns_extents(codec); // Sizing is repeatable.
        failures += expect(resized_extents.nodes == extents.nodes && resized_extents.neighbors == extents.neighbors &&
                           resized_extents.value_bytes == extents.value_bytes);
        basic_columns_storage<index_t> storage {extents};
        auto const columns = storage.columns();
        // A column short by one element overflows; one long by one disagrees about the nodes.
        auto short_values = columns;
        short_values.values = columns.values.first(columns.values.size() - 1);
        failures += expect_eq(index.save(short_values, codec), status_t::overflow_k);
        auto short_keys = columns;
        short_keys.keys = columns.keys.first(columns.keys.size() - 1);
        failures += expect_eq(index.save(short_keys, codec), status_t::overflow_k);
        auto short_ids = columns;
        short_ids.neighbor_ids = columns.neighbor_ids.first(columns.neighbor_ids.size() - 1);
        failures += expect_eq(index.save(short_ids, codec), status_t::overflow_k);
        std::vector<key_t> longer_keys(extents.nodes + 1);
        auto long_keys = columns;
        long_keys.keys = longer_keys;
        failures += expect_eq(index.save(long_keys, codec), status_t::config_mismatch_k);
        failures += expect_eq(index.save(columns, codec), status_t::success_k);

        // A load needs no reserve: its plan covers the nodes under its hold.
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect_eq(restored.load(storage.view(), codec), status_t::success_k);
        failures += expect(restored.capacity() >= nodes_k);
        failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);

        // Same seed, same graph, same code path: the answers must be identical.
        error_stats_t restored_errors;
        for (auto const& query : queries) {
            context_t original_context {stream_key(seed, "snapshot", 1), counting_allocator_t {ledger}};
            context_t restored_context {stream_key(seed, "snapshot", 1), counting_allocator_t {ledger}};
            auto original_metric = fixture_type_::make_metric();
            auto restored_metric = fixture_type_::make_metric();
            auto const [original_result, original_status] = index.search(query, {}, original_metric, original_context);
            auto const [restored_result, restored_status] = restored.search(query, {}, restored_metric,
                                                                            restored_context);
            failures += expect_eq(original_status, restored_status);
            failures += expect_eq(original_result.count, restored_result.count);
            for (std::size_t position = 0; position != original_result.count; ++position) {
                failures += expect_eq(original_context.results[position].node_id,
                                      restored_context.results[position].node_id);
                restored_errors.add(static_cast<f64_t>(restored_context.results[position].distance),
                                    static_cast<f64_t>(original_context.results[position].distance));
            }
        }
        failures += expect_errors("restored distances", basic_comparison_family<typename index_t::distance_t>(),
                                  comparison_verdict_t::assert_k, restored_errors, 0);

        // Loading into a non-empty index is refused; the restored index stays intact.
        failures += expect_eq(restored.load(storage.view(), codec), status_t::config_mismatch_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Hostile columns: every structural mutation of a good snapshot's prefix sums and
 *  IDs must be refused with its documented status, and the same index must load the
 *  intact columns after every refusal - the retryability contract. */
template <typename fixture_type_>
std::size_t test_basic_snapshot_hostile(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using storage_t = basic_columns_storage<index_t>;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 16;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "hostile"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "hostile"), counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, keep_diverse_t {4});
        // One extra manual edge de-uniformizes the degrees and guarantees at least one
        // serialized neighbor ID to corrupt.
        [[maybe_unused]] auto const bumped = index.link(0, static_cast<node_id_t>(nodes_k / 2));

        auto const [columns, status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(status));
        storage_t const& storage = columns;
        failures += expect(storage.neighbor_ids.size() > 0u);
        failures += expect(storage.values.size() > 0u);

        auto expect_refused = [&](auto mutate, status_t wanted) {
            std::unique_ptr<index_t> const fresh_ptr = fixture_type_::make_index(ledger);
            index_t& fresh = *fresh_ptr;
            failures += expect(bool(fresh.reserve(nodes_k)));
            storage_t corrupted = storage;
            mutate(corrupted);
            failures += expect_eq(fresh.load(corrupted.view(), codec), wanted);
            // Every refusal must leave the same index loadable - the retryability contract.
            failures += expect_eq(fresh.load(storage.view(), codec), status_t::success_k);
        };

        expect_refused([](storage_t& c) { c.neighbor_offsets[0] = 1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_offsets[1] = -1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_offsets.back() += 1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_offsets[1] = c.neighbor_ids.size() + 1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_offsets.pop_back(); }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_offsets.push_back(c.neighbor_offsets.back()); },
                       status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_ids[0] = static_cast<node_id_t>(nodes_k); },
                       status_t::malformed_k);
        expect_refused([](storage_t& c) { c.neighbor_ids.pop_back(); }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.value_offsets[0] = 1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.value_offsets[1] = c.values.size() + 1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.value_offsets.back() -= 1; }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.value_offsets.push_back(c.value_offsets.back()); }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.values.pop_back(); }, status_t::malformed_k);
        expect_refused([](storage_t& c) { c.keys.push_back(c.keys.back()); }, status_t::malformed_k);

        // An unreserved target covers its own nodes and loads the intact columns.
        std::unique_ptr<index_t> const unreserved_ptr = fixture_type_::make_index(ledger);
        index_t& unreserved = *unreserved_ptr;
        failures += expect_eq(unreserved.load(storage.view(), codec), status_t::success_k);
        failures += basic_expect_mirrored<fixture_type_>(unreserved, index, nodes_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Region-scoped serialization: a save plan whose uneven regions run from a pool
 *  must emit bytes identical to the single-call form, a region-parallel load must
 *  rebuild a graph whose own re-save is byte-identical, and the misuse cases -
 *  premature finish, out-of-bounds region, dropped plan - answer their documented
 *  statuses while the hold blocks & then frees the unique verbs. */
template <typename fixture_type_>
std::size_t test_basic_snapshot_regions(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 96;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "regions"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "regions"), counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto const [single, single_status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(single_status));
        auto const [extents, measure_status] = index.columns_extents(codec);
        failures += expect(succeeded(measure_status));

        // Uneven splits from a pool must come out identically to the single call.
        basic_columns_storage<index_t> pooled {extents, std::byte {0xEE}};
        std::array<iota_view<node_id_t>, 3> const regions {iota_view<node_id_t> {0, 7}, iota_view<node_id_t> {7, 61},
                                                           iota_view<node_id_t> {61, nodes_k}};
        {
            auto [plan, status] = index.plan_save(pooled.columns(), codec);
            failures += expect(succeeded(status));
            failures += expect_eq(plan.finish(), status_t::config_mismatch_k); // Nothing covered yet.
            failures += expect_eq(plan.save({0, nodes_k + 1}, codec), status_t::config_mismatch_k);
            failures += expect_eq(index.columns_extents(codec).status, status_t::contended_k); // The plan holds unique.

            fu::flat_pool_t pool;
            if (fu::failed(pool.spawn(regions.size()))) return failures + expect(false);
            std::size_t thread_failures = 0;
            // No lock-held check in these threads: the calling thread runs one thread itself
            // while legitimately holding the plan's unique side for the whole pool.
            pool.for_threads([&](std::size_t thread) noexcept {
                if (plan.save(regions[thread], codec) != status_t::success_k)
                    fu::standard_atomic_ref<std::size_t>(thread_failures).store_add(1);
            });
            failures += expect_eq(thread_failures, 0u);
            failures += expect_eq(plan.finish(), status_t::success_k);
            failures += expect_eq(plan.finish(), status_t::uninitialized_k); // Spent: no second unlock.
        }
        failures += expect(pooled == single);
        failures += expect(bool(index.columns_extents(codec))); // The finished plan released its hold.

        // A dropped save plan releases the hold without pronouncing anything.
        {
            auto [plan, status] = index.plan_save(pooled.columns(), codec);
            failures += expect(succeeded(status));
        }
        failures += expect(bool(index.columns_extents(codec)));

        // The load mirror: uneven regions decode concurrently, publication waits for finish.
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect(bool(restored.reserve(nodes_k)));
        {
            auto [plan, status] = restored.plan_load(single.view());
            failures += expect(succeeded(status));
            failures += expect_eq(plan.finish(), status_t::config_mismatch_k); // Nothing covered yet.
            failures += expect_eq(restored.size(), 0u);                        // Nothing published either.

            fu::flat_pool_t pool;
            if (fu::failed(pool.spawn(regions.size()))) return failures + expect(false);
            std::size_t thread_failures = 0;
            pool.for_threads([&](std::size_t thread) noexcept {
                if (plan.load(regions[thread], codec) != status_t::success_k)
                    fu::standard_atomic_ref<std::size_t>(thread_failures).store_add(1);
            });
            failures += expect_eq(thread_failures, 0u);
            failures += expect_eq(restored.size(), 0u); // Still unpublished before the finish.
            failures += expect_eq(plan.finish(), status_t::success_k);
            failures += expect_eq(plan.finish(), status_t::uninitialized_k); // Spent: no second publish.
            failures += expect_eq(restored.size(), static_cast<std::size_t>(nodes_k));
        }
        failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);

        // Closure: the region-loaded graph re-saves to the very same columns.
        basic_columns_storage<index_t> resaved {extents, std::byte {0x77}};
        failures += expect_eq(restored.save(resaved.columns(), codec), status_t::success_k);
        failures += expect(resaved == single);

        // A dropped load plan publishes nothing and leaves a clean retry behind.
        std::unique_ptr<index_t> const abandoned_ptr = fixture_type_::make_index(ledger);
        index_t& abandoned = *abandoned_ptr;
        failures += expect(bool(abandoned.reserve(nodes_k)));
        {
            auto [plan, status] = abandoned.plan_load(single.view());
            failures += expect(succeeded(status));
            failures += expect_eq(plan.load(regions[0], codec), status_t::success_k);
        }
        failures += expect_eq(abandoned.size(), 0u);
        failures += expect_eq(abandoned.load(single.view(), codec), status_t::success_k);
        failures += expect_eq(abandoned.size(), index.size());
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Snapshots

#pragma region Plans and Holds

/** Three rendezvous threads around one held save plan: every try-unique verb answers
 *  @c contended_k for the hold's whole span, the concurrent mode & wait-free readers
 *  stay fully served, and an inserter parked at the shared gate inserts only after the
 *  plan's finish - with the planner never waiting on the parked inserter. */
template <typename fixture_type_>
std::size_t test_basic_plan_race(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    using cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, counting_allocator_t>;
    constexpr std::size_t nodes_k = 64;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "plan_race"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "plan_race"), counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k + 1)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto const [reference, save_status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(save_status));
        auto const [extents, measure_status] = index.columns_extents(codec);
        failures += expect(succeeded(measure_status));
        basic_columns_storage<index_t> held {extents, std::byte {0xEE}};
        basic_columns_storage<index_t> prober {extents};
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "plan_race", 5), counting_allocator_t {ledger});
        key_t const one_more_key = fixture_type_::make_key(nodes_k);
        auto query = fixture_type_::make_values(1, stream_key(seed, "plan_race", 4), counting_allocator_t {ledger});

        bool plan_held_word = false, plan_abandoned_word = false;
        bool inserter_started_word = false, prober_done_word = false;
        fu::standard_atomic_ref<bool> const plan_held {plan_held_word}, plan_abandoned {plan_abandoned_word};
        fu::standard_atomic_ref<bool> const inserter_started {inserter_started_word}, prober_done {prober_done_word};
        std::size_t thread_failures_count = 0;
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};

        auto planner_thread = [&]() noexcept { // Waits on the prober only - never on the inserter.
            auto [plan, status] = index.plan_save(held.columns(), codec);
            if (failed(status)) {
                thread_failures.store_add(1);
                plan_abandoned.store(true, std::memory_order_release);
            }
            plan_held.store(true, std::memory_order_release);
            if (failed(status)) return;
            if (!spin_until([&]() noexcept { return prober_done.load(std::memory_order_acquire); }))
                thread_failures.store_add(1); // Finish regardless - the hold must die.
            if (plan.save({0, static_cast<node_id_t>(nodes_k)}, codec) != status_t::success_k)
                thread_failures.store_add(1);
            if (plan.finish() != status_t::success_k) thread_failures.store_add(1);
        };
        auto inserter_thread = [&]() noexcept { // Provably parked at the shared gate.
            if (!spin_until([&]() noexcept { return plan_held.load(std::memory_order_acquire); }))
                return thread_failures.store_add(1);
            inserter_started.store(true, std::memory_order_release);
            auto thread_metric = fixture_type_::make_metric();
            context_t thread_context {stream_key(seed, "plan_race", 1), counting_allocator_t {ledger}};
            auto const [result, status] = index.insert({one_more.data(), 1}, {&one_more_key, 1}, thread_metric, {},
                                                       thread_context, keep_diverse_t {basic_connectivity_k});
            if (status != status_t::success_k) thread_failures.store_add(1);
            assert_no_locks_held();
        };
        auto prober_thread = [&]() noexcept { // Every check runs strictly inside the hold.
            bool const rendezvoused = spin_until([&]() noexcept {
                return plan_abandoned.load(std::memory_order_acquire) ||
                       (plan_held.load(std::memory_order_acquire) && inserter_started.load(std::memory_order_acquire));
            });
            if (plan_abandoned.load(std::memory_order_acquire))
                return prober_done.store(true, std::memory_order_release); // Counted by thread 0 already.
            if (!rendezvoused) {
                thread_failures.store_add(1);
                return prober_done.store(true, std::memory_order_release);
            }
            context_t prober_context {stream_key(seed, "plan_race", 2), counting_allocator_t {ledger}};
            auto prober_metric = fixture_type_::make_metric();
            // The parked inserter cannot publish under the hold - the overlap proof.
            if (index.size() != nodes_k) thread_failures.store_add(1);
            // Every try-unique verb refuses with contended_k, nothing blocks.
            if (index.columns_extents(codec).status != status_t::contended_k) thread_failures.store_add(1);
            if (index.health(prober_context).status != status_t::contended_k) thread_failures.store_add(1);
            if (index.plan_save(prober.columns(), codec).status != status_t::contended_k) thread_failures.store_add(1);
            if (index.plan_load(reference.view()).status != status_t::contended_k) thread_failures.store_add(1);
            if (index.clear() != status_t::contended_k) thread_failures.store_add(1);
            if (index.template search<synchronization_t::immutable_k>(query[0], {}, prober_metric, prober_context)
                    .status != status_t::contended_k)
                thread_failures.store_add(1);
            if (index.template search<synchronization_t::exclusive_k>(query[0], {}, prober_metric, prober_context)
                    .status != status_t::contended_k)
                thread_failures.store_add(1);
            // The concurrent mode and the wait-free readers stay fully served meanwhile.
            auto const [result, status] = index.search(query[0], {}, prober_metric, prober_context);
            if (status != status_t::success_k) thread_failures.store_add(1);
            thread_failures.store_add(scored_failures(prober_context.results.data(), prober_context.results.size(),
                                                      nodes_k, connectivity_32_k * 2));
            for (std::size_t node = 0; node != nodes_k; ++node) {
                node_id_t const node_id = static_cast<node_id_t>(node);
                if (!(index.key_of(node_id).value == fixture_type_::make_key(node))) thread_failures.store_add(1);
                if (!index.degree_of(node_id)) thread_failures.store_add(1);
                [[maybe_unused]] auto const& published_value = index.published_value_of(node_id);
            }
            cursor_t prober_cursor {index, counting_allocator_t {ledger}};
            if (failed(prober_cursor.seek(0))) thread_failures.store_add(1);
            std::size_t visited = 0;
            for (; prober_cursor; ++prober_cursor) ++visited;
            if (visited != nodes_k) thread_failures.store_add(1);
            assert_no_locks_held();
            prober_done.store(true, std::memory_order_release);
        };

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(3))) return failures + expect(false);
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) return planner_thread();
            if (thread == 1) return inserter_thread();
            if (thread == 2) return prober_thread();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_k + 1); // The parked insert placed after the finish.
        // The graph provably could not move under the hold: the held columns mirror the
        // pre-race reference element for element.
        failures += expect(held == reference);
        // The refusal was the hold, not the graph: a settled certified search succeeds.
        context_t settled_context {stream_key(seed, "plan_race", 3), counting_allocator_t {ledger}};
        auto settled_metric = fixture_type_::make_metric();
        failures += expect_eq(
            index.template search<synchronization_t::immutable_k>(query[0], {}, settled_metric, settled_context).status,
            status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Racing planners: one deterministic refusal round - the hold provably standing
 *  while every other thread tries - then statistical rounds where each round crowns at
 *  least one winner, every refusal is exactly @c contended_k, and every winner's
 *  payload mirrors the settled reference; last, a gated inserter proves the shared
 *  side alone makes @c plan_load & @c plan_save answer @c contended_k. */
template <typename fixture_type_>
std::size_t test_basic_plan_contention(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture_type_::make_metric());
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t planner_threads_k = 4;
    constexpr std::size_t plan_rounds_k = 16;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "plan_contention"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "plan_contention"),
                                                 counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto const [reference, save_status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(save_status));
        auto const [extents, measure_status] = index.columns_extents(codec);
        failures += expect(succeeded(measure_status));

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(planner_threads_k))) return failures + expect(false);
        std::array<basic_columns_storage<index_t>, planner_threads_k> thread_storages;
        for (auto& thread_storage : thread_storages) thread_storage = basic_columns_storage<index_t> {extents};

        // Phase A - a deterministic refusal round: thread 0 provably holds while the rest try.
        bool plan_held_word = false;
        fu::standard_atomic_ref<bool> const plan_held {plan_held_word};
        std::size_t refusals_confirmed_count = 0, thread_failures_count = 0;
        fu::standard_atomic_ref<std::size_t> const refusals_confirmed {refusals_confirmed_count};
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                auto [plan, status] = index.plan_save(thread_storages[0].columns(), codec);
                if (failed(status)) thread_failures.store_add(1);
                plan_held.store(true, std::memory_order_release);
                if (failed(status)) return;
                if (!spin_until([&]() noexcept {
                        return refusals_confirmed.load(std::memory_order_acquire) == planner_threads_k - 1;
                    }))
                    thread_failures.store_add(1); // Finish regardless - the hold must die.
                if (plan.save({0, static_cast<node_id_t>(nodes_k)}, codec) != status_t::success_k)
                    thread_failures.store_add(1);
                if (plan.finish() != status_t::success_k) thread_failures.store_add(1);
                return;
            }
            if (!spin_until([&]() noexcept { return plan_held.load(std::memory_order_acquire); }))
                thread_failures.store_add(1); // A broken rendezvous is a failure, never a shrug.
            if (index.plan_save(thread_storages[thread].columns(), codec).status != status_t::contended_k)
                thread_failures.store_add(1);
            refusals_confirmed.store_add(1); // Unconditional, so thread 0's wait terminates.
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);

        // Phase B - statistical rounds: winners serialize, refusals stay clean.
        std::array<status_t, planner_threads_k> round_statuses {};
        for (std::size_t round = 0; round != plan_rounds_k; ++round) {
            pool.for_threads([&](std::size_t thread) noexcept {
                auto [plan, status] = index.plan_save(thread_storages[thread].columns(), codec);
                round_statuses[thread] = status;
                if (failed(status)) return;
                if (plan.save({0, static_cast<node_id_t>(nodes_k)}, codec) != status_t::success_k ||
                    plan.finish() != status_t::success_k)
                    round_statuses[thread] = status_t::internal_k;
                assert_no_locks_held();
            });
            std::size_t winners_this_round = 0;
            for (std::size_t thread = 0; thread != planner_threads_k; ++thread) {
                if (round_statuses[thread] == status_t::success_k) {
                    ++winners_this_round;
                    // A settled graph serializes deterministically: every winner's columns
                    // equal the reference's - the strongest reload proof.
                    failures += expect(thread_storages[thread] == reference);
                }
                else failures += expect_eq(round_statuses[thread], status_t::contended_k);
            }
            failures += expect(winners_this_round >= 1); // Some thread's try-lock always wins.
        }

        // Phase C - the shared side alone: a gated two-value batch parks an inserter with
        // the shared gate provably held, and every try-unique verb refuses meanwhile.
        std::unique_ptr<index_t> const target_ptr = fixture_type_::make_index(ledger);
        index_t& target = *target_ptr;
        failures += expect(bool(target.reserve(nodes_k)));
        auto pair_values = fixture_type_::make_values(2, stream_key(seed, "plan_contention", 2),
                                                      counting_allocator_t {ledger});
        std::array<key_t, 2> const pair_keys {fixture_type_::make_key(nodes_k), fixture_type_::make_key(nodes_k + 1)};
        basic_columns_storage<index_t> prober {extents};
        bool entered_word = false, released_word = false;
        fu::standard_atomic_ref<bool> const entered {entered_word}, released {released_word};
        // Phase A already proved the counter zero, so Phase C keeps accumulating into it.
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                  &entered_word, &released_word};
                context_t parked_context {stream_key(seed, "plan_contention", 1), counting_allocator_t {ledger}};
                auto const [result, status] = target.insert({pair_values.data(), 2}, {pair_keys.data(), 2}, gated, {},
                                                            parked_context, keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.store_add(1);
                assert_no_locks_held();
                return;
            }
            if (thread != 1) return;
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                thread_failures.store_add(1);
                released.store(true, std::memory_order_release); // The gate opens even on failure.
                return;
            }
            if (target.plan_load(reference.view()).status != status_t::contended_k) thread_failures.store_add(1);
            if (target.plan_save(prober.columns(), codec).status != status_t::contended_k) thread_failures.store_add(1);
            if (target.columns_extents(codec).status != status_t::contended_k) thread_failures.store_add(1);
            if (target.clear() != status_t::contended_k) thread_failures.store_add(1);
            released.store(true, std::memory_order_release);
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(target.size(), 2u);
        // Settled & non-empty: the refusal decays into the documented misuse status.
        failures += expect_eq(target.plan_load(reference.view()).status, status_t::config_mismatch_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Plans and Holds

#pragma region Refinement

/** Drives every phase of @p plan on the calling thread, answering the failures its steps report. */
template <typename plan_type_, typename metric_type_, typename context_type_, typename selector_type_>
std::size_t basic_drive(plan_type_& plan, metric_type_& metric, context_type_& context, selector_type_ selector) {
    std::size_t failures = 0;
    do
        for (typename plan_type_::step_t const step : plan.steps())
            failures += expect_eq(plan.run(step, metric, context, selector), status_t::success_k);
    while (plan.advance());
    return failures;
}

/** Of @p queries, how many narrow walks over @p index find the true best of, every walk drawing its
 *  landmarks from the @p stream, so two counts over two graphs differ by the graph alone. */
template <typename fixture_type_, typename index_type_, typename metric_type_>
std::size_t basic_recall_count(index_type_& index, metric_type_& metric,
                               std::vector<typename index_type_::value_t> const& queries, u64_t stream,
                               allocation_ledger_t& ledger, std::size_t& failures) {
    typename index_type_::traversal_context_t probe {stream, counting_allocator_t {ledger}};
    std::size_t found = 0;
    for (auto const& query : queries) {
        failures += expect_eq(index.search(query, {basic_connectivity_k}, metric, probe).status, status_t::success_k);
        bool found_best = false;
        failures += basic_audit_search<fixture_type_>(index, query, probe, found_best);
        found += found_best;
    }
    return found;
}

/** A refine of a deliberately weak graph, every batch wired from two candidates, keeps or improves
 *  the oracle recall of the same walks, re-selects every node and leaves every one reachable. While
 *  its plan lives batches and saves refuse and searches run, a plan dying unfinished reopens the
 *  claims, and the one-shot refines the grown graph. */
template <typename fixture_type_>
std::size_t test_basic_refine(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "refine"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(basic_growth_nodes_k + basic_recall_samples_k,
                                                 stream_key(seed, "refine"), counting_allocator_t {ledger});
        std::vector<typename index_t::value_t> queries;
        for (std::size_t sample = 0; sample != basic_recall_samples_k; ++sample)
            queries.push_back(std::move(values[basic_growth_nodes_k + sample]));
        values.resize(basic_growth_nodes_k);
        std::vector<node_id_t> assigned_ids;
        keep_diverse_t const selector {basic_connectivity_k};
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, selector, {2});
        u64_t const probes = stream_key(seed, "refine", 1);
        std::size_t const weak_recall = basic_recall_count<fixture_type_>(index, metric, queries, probes, ledger,
                                                                          failures);

        auto extra = fixture_type_::make_values(1, stream_key(seed, "refine", 2), counting_allocator_t {ledger});
        key_t const extra_key = fixture_type_::make_key(basic_growth_nodes_k);
        iota_view<node_id_t> const every {0, static_cast<node_id_t>(basic_growth_nodes_k)};
        {
            auto [plan, plan_status] = index.plan_refine(every, {}, selector, context);
            failures += expect_eq(plan_status, status_t::success_k);
            failures += expect_eq(
                index.insert({extra.data(), 1}, {&extra_key, 1}, metric, {}, context, selector).status,
                status_t::contended_k);
            failures += expect_eq(index.columns_extents(codec).status, status_t::contended_k);
            failures += expect_eq(index.search(queries[0], {}, metric, context).status, status_t::success_k);
            failures += basic_drive(plan, metric, context, selector);
            auto const [refined, finish_status] = plan.finish();
            failures += expect_eq(finish_status, status_t::success_k);
            failures += expect(refined.reselected_nodes >= basic_growth_nodes_k);
            failures += expect(refined.changed_nodes <= refined.reselected_nodes && refined.changed_nodes > 0);
            failures += expect_eq(refined.reachability.audited_nodes, basic_growth_nodes_k);
            failures += expect_eq(refined.reachability.reachable_nodes, basic_growth_nodes_k);
            failures += expect(refined.reachability.back_sweeps != 0);
            failures += expect(refined.traversal_stats.computed_distances > 0);
        }
        std::size_t const refined_recall = basic_recall_count<fixture_type_>(index, metric, queries, probes, ledger,
                                                                             failures);
        if (refined_recall < weak_recall)
            fmt::print(stderr, "  refine diagnostics: {} of {} queries found the true best, {} before\n",
                       refined_recall, basic_recall_samples_k, weak_recall);
        failures += expect(refined_recall >= weak_recall);

        // An unfinished plan heals nothing, and its death reopens the claims.
        {
            auto [plan, plan_status] = index.plan_refine(every, {}, selector, context);
            failures += expect_eq(plan_status, status_t::success_k);
        }
        failures += expect_eq(index.insert({extra.data(), 1}, {&extra_key, 1}, metric, {}, context, selector).status,
                              status_t::success_k);
        auto const [grown, grown_status] = index.refine({0, static_cast<node_id_t>(index.size())}, {}, selector, metric,
                                                        context);
        failures += expect_eq(grown_status, status_t::success_k);
        failures += expect_eq(grown.reachability.audited_nodes, basic_growth_nodes_k + 1);
        failures += expect(grown.changed_nodes <= grown.reselected_nodes);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Three donors built apart move into a receiver holding a quarter of the entries: nothing moves at
 *  planning, then every entry arrives under its key with its value, every donor empties and its
 *  slot nulls, every node stays reachable, and narrow walks over the union keep the relaxed bar. */
template <typename fixture_type_>
std::size_t test_basic_absorb(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    using node_id_t = typename index_t::node_id_t;
    constexpr std::size_t parts_k = 4, entries_k = basic_growth_nodes_k / parts_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "absorb"), counting_allocator_t {ledger}};
        keep_diverse_t const selector {basic_connectivity_k};
        std::array<std::unique_ptr<index_t>, parts_k> indexes;
        std::vector<typename index_t::value_t> originals;
        for (std::size_t part = 0; part != parts_k; ++part) {
            indexes[part] = fixture_type_::make_index(ledger);
            auto values = fixture_type_::make_values(entries_k, stream_key(seed, "absorb", part),
                                                     counting_allocator_t {ledger});
            for (auto& original :
                 fixture_type_::make_values(entries_k, stream_key(seed, "absorb", part), counting_allocator_t {ledger}))
                originals.push_back(std::move(original));
            std::vector<node_id_t> assigned_ids;
            failures += basic_build<fixture_type_>(*indexes[part], metric, values, assigned_ids, context, selector, {},
                                                   part * entries_k);
        }
        index_t& receiver = *indexes[0];
        std::array<index_t*, parts_k - 1> donors;
        for (std::size_t donor = 0; donor != donors.size(); ++donor) donors[donor] = indexes[donor + 1].get();

        {
            auto [plan, plan_status] = receiver.plan_absorb(donors, {}, selector, context);
            failures += expect_eq(plan_status, status_t::success_k);
            failures += expect(plan.moving() && receiver.size() == entries_k && donors[0] == indexes[1].get());
            failures += basic_drive(plan, metric, context, selector);
            auto const [absorbed, finish_status] = plan.finish();
            failures += expect_eq(finish_status, status_t::success_k);
            for (std::size_t donor = 0; donor != donors.size(); ++donor) {
                failures += expect(donors[donor] == nullptr);
                failures += expect_eq(indexes[donor + 1]->size(), 0u);
            }
            failures += expect_eq(receiver.size(), parts_k * entries_k);
            failures += expect_eq(absorbed.reachability.audited_nodes, parts_k * entries_k);
            failures += expect_eq(absorbed.reachability.reachable_nodes, parts_k * entries_k);
            failures += expect(absorbed.reachability.back_sweeps != 0);
            failures += expect(absorbed.changed_nodes <= absorbed.reselected_nodes);
        }
        for (std::size_t entry = 0; entry != parts_k * entries_k; ++entry) {
            auto const [node_id, find_status] = receiver.find(fixture_type_::make_key(entry));
            failures += expect_eq(find_status, status_t::success_k);
            if (succeeded(find_status))
                failures += expect(fixture_type_::value_equals(receiver.published_value_of(node_id), originals[entry]));
        }

        auto const queries = fixture_type_::make_values(basic_recall_samples_k, stream_key(seed, "absorb", parts_k),
                                                        counting_allocator_t {ledger});
        // An absorb walks and sweeps only the donors' rows; a refine then re-selects every row.
        failures += expect(succeeded(
            receiver.refine({0, static_cast<node_id_t>(receiver.size())}, {}, selector, metric, context).status));
        std::size_t const found = basic_recall_count<fixture_type_>(
            receiver, metric, queries, stream_key(seed, "absorb", parts_k + 1), ledger, failures);
        failures += expect(found * 100 >= basic_relaxed_recall_percent_k * basic_recall_samples_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Of the ten nearest entries under the exact oracle, how many the walks over @p index return,
 *  summed over @p queries; a match within the tenth's distance counts, so ties never punish a
 *  walk. */
template <typename fixture_type_, typename index_type_, typename metric_type_>
std::size_t basic_recall_at_ten(index_type_& index, metric_type_& metric,
                                std::vector<typename index_type_::value_t> const& queries, u64_t stream,
                                allocation_ledger_t& ledger, std::size_t& failures) {
    using distance_t = typename index_type_::distance_t;
    using node_id_t = typename index_type_::node_id_t;
    constexpr std::size_t wanted_k = 10;
    typename index_type_::traversal_context_t probe {stream, counting_allocator_t {ledger}};
    std::vector<distance_t> exact(index.size());
    std::size_t found = 0;
    for (auto const& query : queries) {
        failures += expect_eq(index.search(query, {basic_connectivity_k * 4}, metric, probe).status,
                              status_t::success_k);
        for (std::size_t node = 0; node != exact.size(); ++node)
            exact[node] = fixture_type_::oracle(query, index.published_value_of(static_cast<node_id_t>(node)));
        std::sort(exact.begin(), exact.end());
        std::size_t const ranked = min_of(wanted_k, exact.size());
        if (!ranked) continue;
        f64_t const tenth = static_cast<f64_t>(exact[ranked - 1]);
        for (std::size_t rank = 0; rank != min_of(ranked, probe.results.size()); ++rank)
            found += static_cast<f64_t>(probe.results[rank].distance) <= tenth + 1e-3 * max_of(tenth, 1.0);
    }
    return found;
}

/** Drives one split of a freshly built index into @p receivers under @p seed, asserting what it
 *  leaves: every entry in exactly one receiver under its key with its value, the donor empty, and
 *  every receiver reaching all its nodes. The walks draw from a context keyed by @p seed alone, so
 *  equal graphs split alike. Answers the plan's result. */
template <typename fixture_type_, typename index_type_, typename metric_type_>
refine_report_t basic_split(index_type_& donor, std::vector<index_type_*>& receivers, metric_type_& metric,
                            std::vector<typename index_type_::value_t> const& originals, u64_t seed,
                            allocation_ledger_t& ledger, std::size_t& failures) {
    keep_diverse_t const selector {basic_connectivity_k};
    std::size_t const entries = donor.size();
    typename index_type_::traversal_context_t context {seed, counting_allocator_t {ledger}};
    refine_report_t result;
    {
        auto [plan, plan_status] = donor.plan_split(receivers, {{}, seed}, selector, context);
        failures += expect_eq(plan_status, status_t::success_k);
        failures += expect(plan.moving() && donor.size() == entries && receivers[0]->size() == 0);
        // The receivers' claims stay closed while the plan lives.
        auto extra = fixture_type_::make_values(1, stream_key(seed, "split_extra"), counting_allocator_t {ledger});
        auto const extra_key = fixture_type_::make_key(entries);
        failures += expect_eq(
            receivers[0]->insert({extra.data(), 1}, {&extra_key, 1}, metric, {}, context, selector).status,
            status_t::contended_k);
        failures += basic_drive(plan, metric, context, selector);
        auto const [split, finish_status] = plan.finish();
        failures += expect_eq(finish_status, status_t::success_k);
        result = split;
    }
    failures += expect_eq(donor.size(), 0u);
    std::size_t placed = 0;
    for (index_type_* receiver : receivers) {
        placed += receiver->size();
        auto [audit, audit_status] = receiver->plan_health(context);
        failures += expect_eq(audit_status, status_t::success_k);
        do
            for (auto const step : audit.steps()) [[maybe_unused]]
                std::size_t const expanded = audit.sweep(step);
        while (audit.advance_level());
        auto const [health, health_status] = audit.finish();
        failures += expect_eq(health_status, status_t::success_k);
        failures += expect_eq(health.reachable_nodes, receiver->size());
    }
    failures += expect_eq(placed, entries);
    failures += expect_eq(result.reachability.audited_nodes, entries);
    failures += expect_eq(result.reachability.reachable_nodes, entries);
    failures += expect(result.reachability.back_sweeps != 0);
    for (std::size_t entry = 0; entry != entries; ++entry) {
        std::size_t holders = 0;
        for (index_type_* receiver : receivers) {
            auto const key = fixture_type_::make_key(entry);
            holders += receiver->count(key);
            auto const [node_id, find_status] = receiver->find(key);
            if (succeeded(find_status))
                failures += expect(
                    fixture_type_::value_equals(receiver->published_value_of(node_id), originals[entry]));
        }
        failures += expect_eq(holders, 1u);
    }
    return result;
}

/**
 *  @brief A donor splits into two and into three graph-local parts, and absorbing the parts back
 *      restores it: every key arrives once with its value, every receiver reaches all its nodes,
 *      and recall at ten against the exact oracle stays within noise of the donor's at every step.
 *
 *  Two splits of equal graphs under one seed save byte-identical receivers. Refusals leave the
 *  donor whole and every receiver open: fewer than two receivers, a repeated or non-empty one, or
 *  the donor among them; the donor then takes batches again once its split published.
 */
template <typename fixture_type_>
std::size_t test_basic_split(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    using node_id_t = typename index_t::node_id_t;
    constexpr std::size_t entries_k = basic_growth_nodes_k, noise_percent_k = 10;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "split"), counting_allocator_t {ledger}};
        keep_diverse_t const selector {basic_connectivity_k};
        auto const originals = fixture_type_::make_values(entries_k, stream_key(seed, "split"),
                                                          counting_allocator_t {ledger});
        auto const queries = fixture_type_::make_values(basic_recall_samples_k, stream_key(seed, "split", 1),
                                                        counting_allocator_t {ledger});
        u64_t const probes = stream_key(seed, "split", 2);
        std::size_t const recall_total = basic_recall_samples_k * 10;
        // Equal streams under fresh contexts build equal graphs, one batch at a time.
        auto build = [&](index_t& index) {
            context_t building {stream_key(seed, "split", 3), counting_allocator_t {ledger}};
            auto values = fixture_type_::make_values(entries_k, stream_key(seed, "split"),
                                                     counting_allocator_t {ledger});
            std::vector<node_id_t> assigned_ids;
            failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, building, selector);
        };
        auto make_receivers = [&](std::vector<std::unique_ptr<index_t>>& owned, std::size_t count) {
            std::vector<index_t*> receivers;
            for (std::size_t receiver = 0; receiver != count; ++receiver) {
                owned.push_back(fixture_type_::make_index(ledger));
                receivers.push_back(owned.back().get());
            }
            return receivers;
        };
        auto report_recall = [&](char const* stage, std::size_t found, std::size_t baseline) {
            if (found * 100 + noise_percent_k * recall_total < baseline * 100)
                fmt::print(stderr, "  split diagnostics: {} found {} of {} at ten, the donor {}\n", stage, found,
                           recall_total, baseline);
            return expect(found * 100 + noise_percent_k * recall_total >= baseline * 100);
        };

        std::unique_ptr<index_t> const donor = fixture_type_::make_index(ledger);
        std::unique_ptr<index_t> const twin = fixture_type_::make_index(ledger);
        build(*donor);
        build(*twin);
        std::size_t const donor_recall = basic_recall_at_ten<fixture_type_>(*donor, metric, queries, probes, ledger,
                                                                            failures);

        // Refusals precede every hold, or release what they took: the donor stays whole.
        {
            std::vector<std::unique_ptr<index_t>> owned;
            std::vector<index_t*> receivers = make_receivers(owned, 2);
            std::vector<index_t*> single {receivers[0]};
            std::vector<index_t*> repeated {receivers[0], receivers[0]};
            std::vector<index_t*> including {receivers[0], donor.get()};
            failures += expect_eq(donor->plan_split(single, {}, selector, context).status, status_t::config_mismatch_k);
            failures += expect_eq(donor->plan_split(repeated, {}, selector, context).status,
                                  status_t::config_mismatch_k);
            failures += expect_eq(donor->plan_split(including, {}, selector, context).status,
                                  status_t::config_mismatch_k);
            auto filled = fixture_type_::make_values(1, stream_key(seed, "split", 4), counting_allocator_t {ledger});
            auto const filled_key = fixture_type_::make_key(entries_k);
            failures += expect_eq(
                receivers[1]->insert({filled.data(), 1}, {&filled_key, 1}, metric, {}, context, selector).status,
                status_t::success_k);
            failures += expect_eq(donor->plan_split(receivers, {}, selector, context).status,
                                  status_t::config_mismatch_k);
            failures += expect_eq(donor->size(), entries_k);
            failures += expect_eq(receivers[1]->size(), 1u);
            failures += expect_eq(basic_saved_columns(*receivers[0], codec).status, status_t::success_k);
            failures += expect_eq(basic_saved_columns(*donor, codec).status, status_t::success_k);
        }

        // Two parts, twice over equal graphs: the receivers save alike, byte for byte.
        std::vector<std::unique_ptr<index_t>> halves_owned, twin_halves_owned;
        std::vector<index_t*> halves = make_receivers(halves_owned, 2);
        std::vector<index_t*> twin_halves = make_receivers(twin_halves_owned, 2);
        u64_t const split_seed = stream_key(seed, "split", 5);
        refine_report_t const halved = basic_split<fixture_type_>(*donor, halves, metric, originals, split_seed, ledger,
                                                                  failures);
        failures += expect(halved.reselected_nodes > 0);
        [[maybe_unused]] refine_report_t const twin_halved = basic_split<fixture_type_>(
            *twin, twin_halves, metric, originals, split_seed, ledger, failures);
        for (std::size_t half = 0; half != halves.size(); ++half) {
            auto const saved = basic_saved_columns(*halves[half], codec);
            auto const twin_saved = basic_saved_columns(*twin_halves[half], codec);
            failures += expect(succeeded(saved.status) && succeeded(twin_saved.status));
            failures += expect(saved.value == twin_saved.value);
            failures += expect(halves[half]->size() * halves.size() <= entries_k + halves.size());
            failures += report_recall(
                "a half", basic_recall_at_ten<fixture_type_>(*halves[half], metric, queries, probes, ledger, failures),
                donor_recall);
        }

        // The halves absorbed back split again into thirds, which absorb back once more.
        {
            auto [regrown, regrown_status] = halves[0]->plan_absorb({halves.data() + 1, 1}, {}, selector, context);
            failures += expect_eq(regrown_status, status_t::success_k);
            failures += basic_drive(regrown, metric, context, selector);
            failures += expect_eq(regrown.finish().status, status_t::success_k);
        }
        failures += expect_eq(halves[0]->size(), entries_k);
        std::vector<std::unique_ptr<index_t>> thirds_owned;
        std::vector<index_t*> thirds = make_receivers(thirds_owned, 3);
        [[maybe_unused]] refine_report_t const thirded = basic_split<fixture_type_>(
            *halves[0], thirds, metric, originals, stream_key(seed, "split", 6), ledger, failures);
        for (index_t* third : thirds)
            failures += expect(third->size() * thirds.size() <= entries_k + thirds.size() && third->size());

        {
            std::vector<index_t*> donors {thirds[1], thirds[2]};
            auto [absorbed, absorbed_status] = thirds[0]->plan_absorb(donors, {}, selector, context);
            failures += expect_eq(absorbed_status, status_t::success_k);
            failures += basic_drive(absorbed, metric, context, selector);
            auto const [merged, merged_status] = absorbed.finish();
            failures += expect_eq(merged_status, status_t::success_k);
            failures += expect_eq(merged.reachability.reachable_nodes, entries_k);
        }
        failures += expect_eq(thirds[0]->size(), entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) {
            auto const [node_id, find_status] = thirds[0]->find(fixture_type_::make_key(entry));
            failures += expect_eq(find_status, status_t::success_k);
            if (succeeded(find_status))
                failures += expect(
                    fixture_type_::value_equals(thirds[0]->published_value_of(node_id), originals[entry]));
        }
        // An absorb walks and sweeps only the donors' rows; a refine then re-selects every row.
        failures += expect(succeeded(
            thirds[0]->refine({0, static_cast<node_id_t>(thirds[0]->size())}, {}, selector, metric, context).status));
        failures += report_recall(
            "the reunion", basic_recall_at_ten<fixture_type_>(*thirds[0], metric, queries, probes, ledger, failures),
            donor_recall);

        // The emptied donor takes batches from zero again.
        auto again = fixture_type_::make_values(1, stream_key(seed, "split", 7), counting_allocator_t {ledger});
        auto const again_key = fixture_type_::make_key(0);
        failures += expect_eq(donor->insert({again.data(), 1}, {&again_key, 1}, metric, {}, context, selector).status,
                              status_t::success_k);
        failures += expect_eq(donor->size(), 1u);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Places @p shards empty shards in @p indexes and inserts every one of @p keys at the values
 *  @p make_values builds, in one plan; the failures it counted. */
template <typename indexes_type_, typename metric_type_, typename make_values_type_, typename context_type_>
std::size_t basic_fill_shards(indexes_type_& indexes, std::size_t shards,
                              std::span<typename indexes_type_::key_t const> keys, make_values_type_&& make_values,
                              metric_type_& metric, context_type_& context) {
    using value_t = typename indexes_type_::value_t;
    std::size_t failures = 0;
    for (std::size_t shard = 0; shard != shards; ++shard)
        failures += expect_eq(indexes.emplace(no_memory_domain_k), status_t::success_k);
    std::vector<value_t> values = make_values();
    auto [plan, plan_status] = indexes.plan_insert(keys.size());
    failures += expect_eq(plan_status, status_t::success_k);
    if (failed(plan_status)) return failures;
    plan.insert(0, std::span<value_t>(values), keys, metric, context, keep_diverse_t {basic_connectivity_k});
    return failures + expect_eq(plan.finish(), status_t::success_k);
}

/**
 *  @brief A sharded index of two shards splits into two sharded indexes of two shards.
 *
 *  Their sizes add up to the donor's, every key counts once across them, the donor empties, every
 *  receiver reaches all its entries, and a self-search finds its own key across both as often as in
 *  the donor, within noise.
 *
 *  @p indexes_type_ is a @c basic::ram_indexes whose shards place on memory domains; @p make_values
 *  answers the same @p keys .size() values on every call, built for its shards.
 */
template <typename indexes_type_, typename metric_type_, typename make_values_type_>
std::size_t test_basic_sharded_split(u64_t seed, metric_type_& metric,
                                     std::span<typename indexes_type_::key_t const> keys,
                                     make_values_type_&& make_values) {
    using index_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using distance_t = typename indexes_type_::distance_t;
    using value_t = typename indexes_type_::value_t;
    constexpr std::size_t shards_k = 2, wanted_k = 10, noise_percent_k = 10;
    std::size_t const entries = keys.size();
    std::size_t failures = 0;
    forkunion_allocator_t const allocator {no_memory_domain_k};
    keep_diverse_t const selector {basic_connectivity_k};
    typename index_t::traversal_context_t context {stream_key(seed, "plural_split"), allocator};
    failures += expect_eq(index_t::reserve_traversal(context, basic_connectivity_k, {}), status_t::success_k);
    auto place = [&](indexes_type_& indexes) {
        for (std::size_t shard = 0; shard != shards_k; ++shard)
            failures += expect_eq(indexes.emplace(no_memory_domain_k), status_t::success_k);
    };

    indexes_type_ donor {allocator};
    failures += basic_fill_shards(donor, shards_k, keys, make_values, metric, context);
    if (failures) return failures;
    failures += expect_eq(donor.size(), entries);

    // A sample of entries searched by their own values: found once its key ranks among the best.
    std::vector<value_t> const queries = make_values();
    std::size_t const samples = min_of(entries, basic_recall_samples_k);
    auto self_found = [&](std::span<indexes_type_* const> searched) {
        std::size_t found = 0;
        std::array<key_t, wanted_k> found_keys;
        std::array<distance_t, wanted_k> found_distances;
        dummy_predicate_t unfiltered;
        for (std::size_t sample = 0; sample != samples; ++sample) {
            std::size_t const entry = sample * entries / samples;
            bool hit = false;
            for (indexes_type_* indexes : searched) {
                std::vector<node_id_t> watermarks(indexes->shards_count());
                failures += expect_eq(indexes->snapshot(watermarks).status, status_t::success_k);
                auto const [ranked, status] = indexes->search_if(queries[entry], watermarks, found_keys,
                                                                 found_distances, {}, unfiltered, metric, context);
                failures += expect_eq(status, status_t::success_k);
                for (std::size_t rank = 0; rank != ranked; ++rank) hit = hit || found_keys[rank] == keys[entry];
            }
            found += hit;
        }
        return found;
    };
    indexes_type_* const source_alone[1] = {&donor};
    std::size_t const source_found = self_found(source_alone);

    indexes_type_ first {allocator}, second {allocator};
    place(first);
    place(second);
    indexes_type_* receivers[2] = {&first, &second};
    auto const [split, split_status] = donor.split(receivers, {{}, seed}, selector, metric, context);
    failures += expect_eq(split_status, status_t::success_k);
    failures += expect_eq(split.reachability.audited_nodes, entries);
    failures += expect_eq(donor.size(), 0u);
    failures += expect_eq(first.size() + second.size(), entries);
    failures += expect(first.size() && second.size());
    std::size_t miscounted = 0;
    for (key_t const& key : keys) miscounted += first.count(key) + second.count(key) != 1;
    failures += expect_eq(miscounted, 0u);
    for (indexes_type_* receiver : receivers) {
        auto const [health, health_status] = receiver->health(context);
        failures += expect_eq(health_status, status_t::success_k);
        failures += expect_eq(health.reachable_nodes, receiver->size());
    }
    std::size_t const split_found = self_found(receivers);
    failures += expect(split_found * 100 + noise_percent_k * samples >= source_found * 100);
    return failures;
}

/**
 *  @brief Two shards bounded at a fifth of the entries take every one of @p keys in one plan,
 *      which opens two more generations of two.
 *
 *  Every key counts once across the shards, and the merged searches find the true best as often as
 *  one unbounded index does, the merged scans always. A donor's entries absorbed then count once
 *  too, and a clear empties every shard.
 *
 *  @param[in] make_values Builds the entries' values in key order, as <tt>make_values()</tt>.
 *  @param[in] query_of Builds sample query @c s of the recall samples, as <tt>query_of(s)</tt>.
 *  @param[in] best_of The exact best distance of sample query @c s, as <tt>best_of(s)</tt>.
 */
template <typename indexes_type_, typename metric_type_, typename make_values_type_, typename query_of_type_,
          typename best_of_type_>
std::size_t test_basic_sharded(u64_t seed, metric_type_& metric, std::span<typename indexes_type_::key_t const> keys,
                               make_values_type_&& make_values, query_of_type_&& query_of, best_of_type_&& best_of) {
    using index_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename indexes_type_::node_id_t;
    using value_t = typename indexes_type_::value_t;
    using distance_t = typename indexes_type_::distance_t;
    using query_t = std::invoke_result_t<query_of_type_&, std::size_t>;
    constexpr std::size_t placed_shards_k = 2, generations_k = 3, wanted_k = 10;
    std::size_t const entries = keys.size(), shard_capacity = entries / (placed_shards_k * (generations_k - 1) + 1);
    std::size_t failures = 0;
    forkunion_allocator_t const allocator {no_memory_domain_k};
    keep_diverse_t const selector {basic_connectivity_k};
    typename index_t::traversal_context_t context {stream_key(seed, "plural"), allocator};
    failures += expect_eq(index_t::reserve_traversal(context, basic_connectivity_k, {}), status_t::success_k);

    indexes_type_ indexes {allocator};
    indexes.bound_shards(shard_capacity);
    for (std::size_t shard = 0; shard != placed_shards_k; ++shard)
        failures += expect_eq(indexes.emplace(no_memory_domain_k), status_t::success_k);
    failures += expect(bool(indexes.reserve(entries)));
    std::vector<value_t> values = make_values();
    auto [plan, plan_status] = indexes.plan_insert(entries);
    failures += expect_eq(plan_status, status_t::success_k);
    if (failed(plan_status)) return failures;
    for (std::size_t entry = 0; entry != entries; ++entry)
        plan.insert(entry, std::span<value_t>(values).subspan(entry, 1), keys.subspan(entry, 1), metric, context,
                    selector);
    failures += expect_eq(plan.finish(), status_t::success_k);
    failures += expect_eq(indexes.shards_count(), placed_shards_k * generations_k);
    failures += expect_eq(indexes.size(), entries);
    std::size_t miscounted = 0;
    for (key_t const& key : keys) miscounted += indexes.count(key) != 1;
    failures += expect_eq(miscounted, std::size_t {0});

    index_t single {allocator};
    std::vector<value_t> single_values = make_values();
    failures += expect(bool(single.insert(std::span<value_t>(single_values), keys, metric, {}, context, selector)));

    std::vector<node_id_t> watermarks(indexes.shards_count());
    failures += expect_eq(indexes.snapshot(watermarks).status, status_t::success_k);
    std::array<key_t, wanted_k> found_keys;
    std::array<distance_t, wanted_k> found_distances;
    dummy_predicate_t unfiltered;
    std::size_t sharded_best = 0, single_best = 0;
    for (std::size_t sample = 0; sample != basic_recall_samples_k; ++sample) {
        query_t const query = query_of(sample);
        distance_t const best = best_of(sample);
        auto const [found, status] = indexes.search_if(query, watermarks, found_keys, found_distances, {}, unfiltered,
                                                       metric, context);
        failures += expect_eq(status, status_t::success_k);
        sharded_best += found && found_distances[0] == best;
        failures += expect_eq(single.search(query, {}, metric, context).status, status_t::success_k);
        single_best += !context.results.empty() && context.results[0].distance == best;
    }
    failures += expect(sharded_best * 100 >= basic_minimal_recall_percent_k * basic_recall_samples_k);
    failures += expect(sharded_best + basic_recall_samples_k / 10 >= single_best);
    for (std::size_t sample = 0; sample != placed_shards_k * 2; ++sample) {
        auto const [found, status] = indexes.scan(query_of(sample), watermarks, found_keys, found_distances, unfiltered,
                                                  metric, context);
        failures += expect_eq(status, status_t::success_k);
        failures += expect(found == wanted_k && found_distances[0] == best_of(sample));
    }

    // The donor's entries are the single index's values under keys of their own.
    std::size_t const donor_entries = shard_capacity / 3;
    indexes_type_ donor {allocator};
    failures += expect_eq(donor.emplace(no_memory_domain_k), status_t::success_k);
    std::vector<key_t> donor_keys(donor_entries);
    for (std::size_t entry = 0; entry != donor_entries; ++entry)
        donor_keys[entry] = uuid_of(stream_key(seed, "plural_donor", entry));
    std::vector<value_t> donor_values = make_values();
    auto [donor_plan, donor_status] = donor.plan_insert(donor_entries);
    failures += expect_eq(donor_status, status_t::success_k);
    if (failed(donor_status)) return failures;
    donor_plan.insert(0, std::span<value_t>(donor_values).first(donor_entries), std::span<key_t const>(donor_keys),
                      metric, context, selector);
    failures += expect_eq(donor_plan.finish(), status_t::success_k);
    std::array<indexes_type_*, 1> donors {&donor};
    {
        auto [absorbing, absorb_status] = indexes.plan_absorb(donors, {}, selector, context);
        failures += expect_eq(absorb_status, status_t::success_k);
        if (failed(absorb_status)) return failures;
        do
            for (typename indexes_type_::refine_plan_t::step_t const step : absorbing.steps())
                failures += expect_eq(absorbing.run(step, metric, context, selector), status_t::success_k);
        while (absorbing.advance());
        failures += expect_eq(absorbing.finish().status, status_t::success_k);
    }
    failures += expect_eq(indexes.size(), entries + donor_entries);
    failures += expect_eq(donor.size(), std::size_t {0});
    // A consumed donor stays sealed: its placed shard keeps its claims, so a clear refuses.
    failures += expect_eq(donor.plan_insert(0).status, status_t::contended_k);
    failures += expect_eq(donor.clear(), status_t::contended_k);
    miscounted = 0;
    for (key_t const& key : donor_keys) miscounted += indexes.count(key) != 1;
    failures += expect_eq(miscounted, std::size_t {0});

    failures += expect_eq(indexes.clear(), status_t::success_k);
    failures += expect_eq(indexes.size(), std::size_t {0});
    failures += expect_eq(indexes.count(keys[0]), std::size_t {0});
    return failures;
}

/** Whether @p key is among the first @p count of @p keys. */
template <typename key_type_>
bool basic_drawn_before(std::span<key_type_ const> keys, std::size_t count, key_type_ const& key) noexcept {
    return std::any_of(keys.begin(), keys.begin() + count, [&](key_type_ const& earlier) { return earlier == key; });
}

/**
 *  @brief A sample of two shards holding every one of @p keys draws as many distinct keys as it
 *      wants, each one held.
 *
 *  Each draw is its key's only entry, hanging from an earlier draw at finite coordinates; an equal
 *  seed draws the same again, a smaller sample draws a prefix of it, and a sample around one key
 *  draws around it.
 *
 *  @param[in] make_values Builds the entries' values in key order, as <tt>make_values()</tt>.
 *  @param[in] coordinates_of Fixed coordinates of a value, or @c no_fixed_coordinates_t.
 */
template <typename indexes_type_, typename metric_type_, typename make_values_type_, typename coordinates_type_>
std::size_t test_basic_sharded_sample(u64_t seed, metric_type_& metric,
                                      std::span<typename indexes_type_::key_t const> keys,
                                      make_values_type_&& make_values, coordinates_type_ const& coordinates_of) {
    using index_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename indexes_type_::node_id_t;
    constexpr std::size_t shards_k = 2;
    std::size_t const entries = keys.size(), wanted = min_of(entries, shards_k * basic_connectivity_k);
    std::size_t failures = 0;
    forkunion_allocator_t const allocator {no_memory_domain_k};
    typename index_t::traversal_context_t context {stream_key(seed, "plural_sample"), allocator};
    failures += expect_eq(index_t::reserve_traversal(context, basic_connectivity_k, {}), status_t::success_k);
    indexes_type_ indexes {allocator};
    failures += basic_fill_shards(indexes, shards_k, keys, make_values, metric, context);
    if (failures) return failures;
    std::vector<node_id_t> watermarks(indexes.shards_count());
    failures += expect_eq(indexes.snapshot(watermarks).status, status_t::success_k);

    struct drawn_t {
        std::vector<key_t> keys, parents;
        std::vector<std::size_t> ordinals;
        std::vector<f32_t> parent_distances, coordinates;
        sample_result_t result;

        explicit drawn_t(std::size_t wanted)
            : keys(wanted), parents(wanted), ordinals(wanted), parent_distances(wanted), coordinates(wanted * 3) {}
    };
    sample_config_t const config {stream_key(seed, "plural_sample"), 0};
    auto draw = [&](std::size_t count, std::span<key_t const> focus) {
        drawn_t drawn {count};
        auto const [result, status] = indexes.sample(
            {{}, {}, focus}, watermarks,
            {drawn.keys, drawn.ordinals, drawn.parents, drawn.parent_distances, drawn.coordinates}, config, metric,
            context, coordinates_of);
        failures += expect_eq(status, status_t::success_k);
        drawn.result = result;
        return drawn;
    };
    drawn_t const whole = draw(wanted, {}), repeated = draw(wanted, {}), nested = draw(wanted / 2, {});
    failures += expect_eq(whole.result.written, wanted);
    std::size_t misdrawn = 0;
    for (std::size_t index = 0; index != wanted; ++index) {
        misdrawn += indexes.count(whole.keys[index]) != 1 || whole.ordinals[index] != 0;
        misdrawn += basic_drawn_before<key_t>(whole.keys, index, whole.keys[index]);
        misdrawn += index && !basic_drawn_before<key_t>(whole.keys, index, whole.parents[index]);
        for (std::size_t axis = 0; axis != 3; ++axis) misdrawn += !std::isfinite(whole.coordinates[index * 3 + axis]);
    }
    failures += expect_eq(misdrawn, 0u);
    failures += expect(repeated.keys == whole.keys && repeated.parents == whole.parents &&
                       repeated.coordinates == whole.coordinates);
    failures += expect(std::equal(nested.keys.begin(), nested.keys.end(), whole.keys.begin()));

    std::array<key_t, 1> const focus {keys[0]};
    drawn_t const around = draw(wanted, focus);
    failures += expect(around.result.written > 0);
    return failures;
}

/**
 *  @brief A cursor over two shards holding every one of @p keys yields each key once as stored.
 *
 *  From the first key breadth-first it yields that key first and each reached entry once, every
 *  entry's links held keys at non-decreasing offsets; once a clear opened a window, the cursor
 *  answers @c contended_k.
 *
 *  @param[in] make_values Builds the entries' values in key order, as <tt>make_values()</tt>.
 */
template <typename indexes_type_, typename metric_type_, typename make_values_type_>
std::size_t test_basic_sharded_cursor(u64_t seed, metric_type_& metric,
                                      std::span<typename indexes_type_::key_t const> keys,
                                      make_values_type_&& make_values) {
    using index_t = typename indexes_type_::index_t;
    using key_t = typename indexes_type_::key_t;
    using cursor_t = typename indexes_type_::cursor_t;
    constexpr std::size_t shards_k = 2;
    std::size_t failures = 0;
    forkunion_allocator_t const allocator {no_memory_domain_k};
    typename index_t::traversal_context_t context {stream_key(seed, "plural_cursor"), allocator};
    failures += expect_eq(index_t::reserve_traversal(context, basic_connectivity_k, {}), status_t::success_k);
    indexes_type_ indexes {allocator};
    failures += basic_fill_shards(indexes, shards_k, keys, make_values, metric, context);
    if (failures) return failures;

    failures += expect_sharded_cursor(indexes, keys[0], keys.size(), basic_connectivity_k + 1);

    cursor_t cursor {allocator};
    failures += expect_eq(cursor.open(indexes, {}), status_t::success_k);
    failures += expect_eq(indexes.clear(), status_t::success_k);
    std::array<key_t, 1> read_keys;
    failures += expect_eq(cursor.pull(indexes, std::span<key_t> {read_keys}, nullptr, nullptr).status,
                          status_t::contended_k);
    return failures;
}

/** Allocation refusals at planning leave both sides byte-identical: a budget swept up from the live
 *  bytes refuses an absorb at its allocations in turn, each refusal leaving the receiver's and the
 *  donor's saved columns as they were, which saving proves free of holds, until a budget wide
 *  enough plans; that plan, dying before its first move, changes nothing either. */
template <typename fixture_type_>
std::size_t test_basic_refine_starved(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    using node_id_t = typename index_t::node_id_t;
    constexpr std::size_t rows_k = 64;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "refine_starved"), counting_allocator_t {ledger}};
        keep_diverse_t const selector {basic_connectivity_k};
        std::unique_ptr<index_t> const receiver = fixture_type_::make_index(ledger);
        std::unique_ptr<index_t> const donor = fixture_type_::make_index(ledger);
        for (std::size_t part = 0; part != 2; ++part) {
            auto values = fixture_type_::make_values(rows_k, stream_key(seed, "refine_starved", part),
                                                     counting_allocator_t {ledger});
            std::vector<node_id_t> assigned_ids;
            failures += basic_build<fixture_type_>(part ? *donor : *receiver, metric, values, assigned_ids, context,
                                                   selector, {2}, part * rows_k);
        }
        auto const receiver_before = basic_saved_columns(*receiver, codec);
        auto const donor_before = basic_saved_columns(*donor, codec);
        failures += expect_eq(receiver_before.status, status_t::success_k);
        failures += expect_eq(donor_before.status, status_t::success_k);

        std::size_t refusals = 0;
        bool planned = false;
        for (std::size_t slack = 0; !planned && slack < (std::size_t {1} << 30);
             slack += max_of<std::size_t>(16, slack / 8)) {
            std::array<index_t*, 1> donors {donor.get()};
            status_t status = status_t::success_k;
            {
                context_t planning {stream_key(seed, "refine_starved", 2), counting_allocator_t {ledger}};
                ledger.byte_budget = ledger.live_bytes + static_cast<std::ptrdiff_t>(slack);
                auto [plan, plan_status] = receiver->plan_absorb(donors, {}, selector, planning);
                ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
                status = plan_status;
            }
            planned = succeeded(status);
            if (!planned) {
                failures += expect_eq(status, status_t::bad_alloc_k);
                ++refusals;
            }
            failures += expect(donors[0] == donor.get());
            auto const receiver_after = basic_saved_columns(*receiver, codec);
            auto const donor_after = basic_saved_columns(*donor, codec);
            failures += expect_eq(receiver_after.status, status_t::success_k);
            failures += expect_eq(donor_after.status, status_t::success_k);
            failures += expect(receiver_after.value == receiver_before.value);
            failures += expect(donor_after.value == donor_before.value);
        }
        failures += expect(planned && refusals > 0);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Crafted neighbor lists pin what a heal may take. The cycle 0 → 1 → 2 → 1 links node 1 from two
 *  reached nodes, yet only node 0 reached it first, so that in-link stays; node 6, linked from
 *  nodes 3 and 4 settled a level before it, gives one up once. Node 2 goes back only through node
 *  0, so it keeps that link, while node 7, back through nodes 8 and 9 that went back a level before
 *  it, gives one up once. */
template <typename fixture_type_>
std::size_t test_basic_refine_releases(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 10;
    std::array<std::initializer_list<node_id_t>, nodes_k> const links {
        {{1, 5}, {2}, {1, 0}, {6}, {6}, {3, 4}, {0, 7}, {8, 9}, {0}, {0}}};
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "refine_releases"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "refine_releases"),
                                                 counting_allocator_t {ledger});
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});
        for (node_id_t from = 0; from != nodes_k; ++from) {
            for (node_id_t to = 0; to != nodes_k; ++to) [[maybe_unused]]
                status_t const unlinked = index.unlink(from, to);
            for (node_id_t const to : links[from])
                failures += expect_eq(index.link(from, to).status, status_t::success_k);
        }

        auto [audit, audit_status] = index.plan_health(context);
        failures += expect_eq(audit_status, status_t::success_k);
        do
            for (auto const step : audit.steps()) [[maybe_unused]]
                std::size_t const expanded = audit.sweep(step);
        while (audit.advance_level());
        auto const [health, health_status] = audit.finish();
        failures += expect_eq(health_status, status_t::success_k);
        failures += expect_eq(health.reachable_nodes, nodes_k);
        failures += expect(!audit.release_link(1));
        failures += expect(audit.release_link(6) && !audit.release_link(6));
        failures += expect(!audit.release_out_link(2));
        failures += expect(audit.release_out_link(7) && !audit.release_out_link(7));
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Refinement

#pragma region Parked Mutations

/** The health report on a settled graph, its determinism across repeated calls,
 *  and - through a gated metric parking one insert mid-flight - the settled-past
 *  guard refusing @c health & @c columns_extents with @c contended_k. */
template <typename fixture_type_>
std::size_t test_basic_health(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture_type_::make_metric());
    constexpr std::size_t nodes_k = 64;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "health"), counting_allocator_t {ledger}};

        failures += expect(bool(index.reserve(nodes_k + 1)));
        auto const empty_report = index.health(context);
        failures += expect_eq(empty_report.status, status_t::success_k);
        failures += expect_eq(empty_report.connected_components, 0u);
        failures += expect_eq(empty_report.degree_max, 0u);

        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "health"), counting_allocator_t {ledger});
        std::vector<typename index_t::node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto const first_report = index.health(context);
        failures += expect_eq(first_report.status, status_t::success_k);
        failures += expect(first_report.connected_components >= 1);
        failures += expect(first_report.largest_component <= nodes_k);
        failures += expect(first_report.degree_min >= 1);
        failures += expect(first_report.degree_max <= nodes_k - 1);
        failures += expect(static_cast<f64_t>(first_report.degree_min) <= first_report.mean_degree &&
                           first_report.mean_degree <= static_cast<f64_t>(first_report.degree_max));

        // The deterministic half must reproduce; only seed coverage may vary.
        auto const second_report = index.health(context);
        failures += expect_eq(second_report.connected_components, first_report.connected_components);
        failures += expect_eq(second_report.largest_component, first_report.largest_component);
        failures += expect_eq(second_report.degree_min, first_report.degree_min);
        failures += expect_eq(second_report.degree_max, first_report.degree_max);
        failures += expect_eq(second_report.reciprocal_edges, first_report.reciprocal_edges);

        // Park one insert mid-walk: the guards must answer contended_k, not tear.
        bool entered_word = false, released_word = false;
        fu::standard_atomic_ref<bool> const entered {entered_word}, released {released_word};
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "health", 3), counting_allocator_t {ledger});
        typename index_t::key_t const one_more_key = fixture_type_::make_key(nodes_k);
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::size_t thread_failures_count = 0;
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                  &entered_word, &released_word};
                context_t parked_context {stream_key(seed, "health", 1), counting_allocator_t {ledger}};
                auto const [result, status] = index.insert({one_more.data(), 1}, {&one_more_key, 1}, gated, {},
                                                           parked_context, keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.store_add(1);
                return;
            }
            if (thread != 1) return;
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                // The gate must open even on a failed rendezvous, or the pool deadlocks.
                thread_failures.store_add(1);
                released.store(true, std::memory_order_release);
                return;
            }
            context_t checker_context {stream_key(seed, "health", 2), counting_allocator_t {ledger}};
            if (index.health(checker_context).status != status_t::contended_k) thread_failures.store_add(1);
            if (index.columns_extents(codec).status != status_t::contended_k) thread_failures.store_add(1);
            released.store(true, std::memory_order_release);
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_k + 1);
        failures += expect_eq(index.health(context).status, status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** The certified modes at runtime: an exclusive build equals a concurrent
 *  single-threaded one, certified searches answer identically to the concurrent mode
 *  on a settled graph, and a parked insert makes both certified modes answer
 *  @c contended_k instead of tearing. */
template <typename fixture_type_>
std::size_t test_basic_certified(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture_type_::make_metric());
    constexpr std::size_t nodes_k = 48;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const exclusive_index_ptr = fixture_type_::make_index(ledger);
        std::unique_ptr<index_t> const concurrent_index_ptr = fixture_type_::make_index(ledger);
        index_t& exclusive_index = *exclusive_index_ptr;
        index_t& concurrent_index = *concurrent_index_ptr;
        failures += expect(bool(exclusive_index.reserve(nodes_k + 1)));
        failures += expect(bool(concurrent_index.reserve(nodes_k + 1)));
        auto exclusive_values = fixture_type_::make_values(nodes_k, stream_key(seed, "certified"),
                                                           counting_allocator_t {ledger});
        auto concurrent_values = fixture_type_::make_values(nodes_k, stream_key(seed, "certified"),
                                                            counting_allocator_t {ledger});
        std::vector<key_t> keys(nodes_k);
        for (std::size_t ordinal = 0; ordinal != nodes_k; ++ordinal) keys[ordinal] = fixture_type_::make_key(ordinal);
        auto exclusive_metric = fixture_type_::make_metric();
        auto concurrent_metric = fixture_type_::make_metric();
        context_t exclusive_context {stream_key(seed, "certified"), counting_allocator_t {ledger}};
        context_t concurrent_context {stream_key(seed, "certified"), counting_allocator_t {ledger}};

        failures += expect_eq(exclusive_index
                                  .template insert<synchronization_t::exclusive_k>(
                                      {exclusive_values}, {keys}, exclusive_metric, {}, exclusive_context,
                                      keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        failures += expect_eq(concurrent_index
                                  .insert({concurrent_values}, {keys}, concurrent_metric, {}, concurrent_context,
                                          keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        // Same data, same seeds, single-threaded: the lock elision must be invisible.
        failures += basic_expect_mirrored<fixture_type_>(exclusive_index, concurrent_index, nodes_k);

        // Three search modes over the settled graph must agree bit-for-bit.
        auto queries = fixture_type_::make_values(4, stream_key(seed, "certified", 4), counting_allocator_t {ledger});
        for (auto const& query : queries) {
            context_t concurrent_mode {stream_key(seed, "certified", 1), counting_allocator_t {ledger}};
            context_t immutable_mode {stream_key(seed, "certified", 1), counting_allocator_t {ledger}};
            context_t exclusive_mode {stream_key(seed, "certified", 1), counting_allocator_t {ledger}};
            auto mode_metric = fixture_type_::make_metric();
            auto const [concurrent_result, concurrent_status] = exclusive_index.search(query, {}, mode_metric,
                                                                                       concurrent_mode);
            auto const [immutable_result, immutable_status] =
                exclusive_index.template search<synchronization_t::immutable_k>(query, {}, mode_metric, immutable_mode);
            auto const [exclusive_result, exclusive_status] =
                exclusive_index.template search<synchronization_t::exclusive_k>(query, {}, mode_metric, exclusive_mode);
            failures += expect_eq(immutable_status, concurrent_status);
            failures += expect_eq(exclusive_status, concurrent_status);
            failures += expect_eq(immutable_result.count, concurrent_result.count);
            failures += expect_eq(exclusive_result.count, concurrent_result.count);
            for (std::size_t position = 0; position != concurrent_result.count; ++position) {
                failures += expect_eq(immutable_mode.results[position].node_id,
                                      concurrent_mode.results[position].node_id);
                failures += expect_eq(exclusive_mode.results[position].node_id,
                                      concurrent_mode.results[position].node_id);
            }
        }

        // A parked insert violates any certification; the guard must say so.
        bool entered_word = false, released_word = false;
        fu::standard_atomic_ref<bool> const entered {entered_word}, released {released_word};
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "certified", 5), counting_allocator_t {ledger});
        key_t const one_more_key = fixture_type_::make_key(nodes_k);
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::size_t thread_failures_count = 0;
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                  &entered_word, &released_word};
                context_t parked_context {stream_key(seed, "certified", 2), counting_allocator_t {ledger}};
                auto const [result, status] = exclusive_index.insert({one_more.data(), 1}, {&one_more_key, 1}, gated,
                                                                     {}, parked_context,
                                                                     keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.store_add(1);
                return;
            }
            if (thread != 1) return;
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                thread_failures.store_add(1);
                released.store(true, std::memory_order_release); // The gate opens even on failure.
                return;
            }
            context_t checker_context {stream_key(seed, "certified", 3), counting_allocator_t {ledger}};
            auto checker_metric = fixture_type_::make_metric();
            if (exclusive_index
                    .template search<synchronization_t::immutable_k>(one_more[0], {}, checker_metric, checker_context)
                    .status != status_t::contended_k)
                thread_failures.store_add(1);
            if (exclusive_index
                    .template search<synchronization_t::exclusive_k>(one_more[0], {}, checker_metric, checker_context)
                    .status != status_t::contended_k)
                thread_failures.store_add(1);
            released.store(true, std::memory_order_release);
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(exclusive_index.size(), nodes_k + 1);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** The certified exit probe: a gated metric parks a certified search mid-walk -
 *  entry probe already passed - an insert runs to completion inside the parked walk,
 *  and the exit probe must pronounce @c contended_k. The speculative reads formally
 *  race the writer, so the racing block skips under ThreadSanitizer. */
template <typename fixture_type_>
std::size_t test_basic_torn_bracket(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using key_t = typename index_t::key_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture_type_::make_metric());
    constexpr std::size_t nodes_k = 48;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "torn_bracket"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "torn_bracket"),
                                                 counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k + 1)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});
        auto query = fixture_type_::make_values(1, stream_key(seed, "torn_bracket", 5), counting_allocator_t {ledger});
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "torn_bracket", 6),
                                                   counting_allocator_t {ledger});
        key_t const one_more_key = fixture_type_::make_key(nodes_k);

        // The control: on a settled graph the same bracket certifies cleanly.
        context_t control_context {stream_key(seed, "torn_bracket", 1), counting_allocator_t {ledger}};
        auto control_metric = fixture_type_::make_metric();
        failures += expect_eq(
            index.template search<synchronization_t::immutable_k>(query[0], {}, control_metric, control_context).status,
            status_t::success_k);

        if constexpr (!compiled_with_thread_sanitizer_k) {
            bool entered_word = false, released_word = false;
            fu::standard_atomic_ref<bool> const entered {entered_word}, released {released_word};
            std::size_t thread_failures_count = 0;
            fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
            u32_t parked_status = ~u32_t {0};
            fu::flat_pool_t pool;
            if (fu::failed(pool.spawn(2))) return failures + expect(false);
            pool.for_threads([&](std::size_t thread) noexcept {
                if (thread == 0) { // The certified search is the parked party - no locks held.
                    basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                      &entered_word, &released_word};
                    context_t parked_context {stream_key(seed, "torn_bracket", 2), counting_allocator_t {ledger}};
                    auto const [result, status] = index.template search<synchronization_t::immutable_k>(
                        query[0], {}, gated, parked_context);
                    parked_status = static_cast<u32_t>(status); // Read only after the pool is awaited.
                    return;
                }
                if (thread != 1) return; // The mutator: runs strictly inside the parked walk.
                if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                    thread_failures.store_add(1);
                    released.store(true, std::memory_order_release); // The gate opens even on failure.
                    return;
                }
                auto thread_metric = fixture_type_::make_metric();
                context_t thread_context {stream_key(seed, "torn_bracket", 3), counting_allocator_t {ledger}};
                auto const [result, status] = index.insert({one_more.data(), 1}, {&one_more_key, 1}, thread_metric, {},
                                                           thread_context, keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.store_add(1);
                released.store(true, std::memory_order_release); // Every path - the search must resume.
                assert_no_locks_held();
            });
            failures += expect_eq(thread_failures.load(), 0u);
            // The entry probe passed - `entered` fired after it - so contended_k here is the
            // exit probe's verdict: the generation moved & the watermark advanced.
            failures += expect_eq(static_cast<status_t>(parked_status), status_t::contended_k);
            failures += expect_eq(index.size(), nodes_k + 1);
        }

        // The refusal was transient: a settled re-run certifies again.
        context_t settled_context {stream_key(seed, "torn_bracket", 4), counting_allocator_t {ledger}};
        auto settled_metric = fixture_type_::make_metric();
        failures += expect_eq(
            index.template search<synchronization_t::immutable_k>(query[0], {}, settled_metric, settled_context).status,
            status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Parked Mutations

#pragma region Contracts

/** The metric-contract audit: every drain sized to its round's pushes, no round
 *  abandoned mid-flight, telemetry equal to the audited push count, a fold absorbing
 *  every drained round whole, a sated fold ending its walk early and successfully, and
 *  armed refusals driving the saturation paths without ever breaking a walk. */
template <typename fixture_type_>
std::size_t test_basic_metric_contract(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using distance_t = typename index_t::distance_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture_type_::make_metric());
    constexpr std::size_t nodes_k = 48;

    /** Tallies the rounds & pairs a walk scores, sated once it has seen @c rounds_limit rounds. */
    struct counting_fold_t {
        std::size_t rounds_limit = (std::numeric_limits<std::size_t>::max)();
        std::size_t rounds = 0;
        std::size_t pairs = 0;

        walk_demand_t absorb(std::span<node_id_t const> node_ids, std::span<distance_t const>) noexcept {
            ++rounds, pairs += node_ids.size();
            return rounds == rounds_limit ? walk_demand_t::sated_k : walk_demand_t::continue_k;
        }
    };

    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        basic_auditing_metric<metric_t, distance_t> metric {fixture_type_::make_metric()};
        context_t context {stream_key(seed, "contract"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "contract"), counting_allocator_t {ledger});
        auto queries = fixture_type_::make_values(4, stream_key(seed, "contract", 1), counting_allocator_t {ledger});

        failures += expect(bool(index.reserve(nodes_k + 2)));
        std::vector<typename index_t::node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});
        failures += expect_eq(metric.violations, 0u);
        failures += expect(metric.resets > 0 && metric.drains > 0);
        // Every drained thread is one counted distance, so the two tallies must agree.
        failures += expect_eq(context.telemetry.computed_distances, metric.pushes);

        for (auto const& query : queries) {
            counting_fold_t fold;
            std::size_t const drains_before = metric.drains;
            auto const [result, status] = index.search(query, fold, {}, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(context);
            failures += expect_eq(fold.rounds, metric.drains - drains_before);
            failures += expect_eq(fold.pairs, result.telemetry.computed_distances);
        }
        failures += expect_eq(metric.violations, 0u);

        // Sated by its landmarks, a walk ends before they reach the candidates, scoring less than a
        // whole-graph walk.
        counting_fold_t sated {1};
        auto const [sated_result, sated_status] = index.search(queries[0], sated, {}, metric, context);
        failures += expect_eq(sated_status, status_t::success_k);
        failures += expect_eq(sated.rounds, 1u);
        failures += expect_eq(sated_result.count, 0u);
        auto const [unfolded_result, unfolded_status] = index.search(queries[0], {}, metric, context);
        failures += expect_eq(unfolded_status, status_t::success_k);
        failures += expect(sated_result.telemetry.computed_distances < unfolded_result.telemetry.computed_distances);

        // Armed refusal: the walk must survive every saturation path, stay ranked, and say so.
        std::size_t saturations = 0;
        for (std::size_t budget : {basic_connectivity_k, std::size_t {1}, std::size_t {0}}) {
            metric.arm(budget);
            auto const [result, status] = index.search(queries[0], {}, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(context);
            saturations += result.telemetry.saturated_walks;
            metric.disarm();
        }
        failures += expect(saturations > 0);
        failures += expect_eq(metric.violations, 0u);

        // A round narrower than a neighbor list folds instead of refusing: the walk ranks
        // unsaturated, the diversity rule compares every candidate as one wide round would, and a
        // twin of a stored value ends up in that value's full list through the re-selection.
        constexpr std::size_t narrow_round_k = 3;
        auto const [wide, wide_status] = index.search(queries[0], {}, metric, context);
        failures += expect_eq(wide_status, status_t::success_k);
        keep_diverse_t const diverse {basic_connectivity_k};
        diverse(index, metric, context);
        std::vector<node_id_t> wide_selection;
        for (auto const& selected : context.selected_neighbors) wide_selection.push_back(selected.node_id);
        metric.round_limit = narrow_round_k;
        diverse(index, metric, context);
        failures += expect_eq(context.selected_neighbors.size(), wide_selection.size());
        for (std::size_t position = 0; position != context.selected_neighbors.size(); ++position)
            failures += expect_eq(context.selected_neighbors[position].node_id, wide_selection[position]);
        auto const [narrow, narrow_status] = index.search(queries[0], {}, metric, context);
        failures += expect_eq(narrow_status, status_t::success_k);
        failures += expect_eq(narrow.telemetry.saturated_walks, 0u);

        node_id_t full_row = 0;
        while (full_row + 1u != index.size() && index.degree_of(full_row).value != basic_connectivity_k) ++full_row;
        failures += expect_eq(index.degree_of(full_row).value, basic_connectivity_k);
        auto twins = fixture_type_::make_values(nodes_k, stream_key(seed, "contract"), counting_allocator_t {ledger});
        typename index_t::key_t const twin_key = fixture_type_::make_key(nodes_k);
        node_id_t twin_id = 0;
        failures += expect_eq(
            index.insert({twins.data() + full_row, 1}, {&twin_key, 1}, metric, {&twin_id, 1}, context, diverse).status,
            status_t::success_k);
        failures += expect(index.link(twin_id, full_row).value == membership_t::present_k);
        failures += expect(index.link(full_row, twin_id).value == membership_t::present_k);
        metric.round_limit = (std::numeric_limits<std::size_t>::max)();
        failures += expect_eq(metric.violations, 0u);

        // An armed insert wires a thinner node but still publishes it.
        metric.arm(2);
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "contract", 2), counting_allocator_t {ledger});
        typename index_t::key_t const one_more_key = fixture_type_::make_key(nodes_k + 1);
        failures += expect_eq(index
                                  .insert({one_more.data(), 1}, {&one_more_key, 1}, metric, {}, context,
                                          keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        metric.disarm();
        failures += expect_eq(index.size(), nodes_k + 2);
        failures += expect_eq(metric.violations, 0u);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Frontier starvation drives re-seeding: islands - nodes manually stripped of
 *  every edge in both directions - are reachable only through a restart's landmark
 *  draws, and whole-graph candidates must still find every node. */
template <typename fixture_type_>
std::size_t test_basic_restarts(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t islands_k = 4;
    constexpr std::size_t attempts_k = 4;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "restarts"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k + 1, stream_key(seed, "restarts"),
                                                 counting_allocator_t {ledger});
        auto query = std::move(values[nodes_k]);
        values.resize(nodes_k);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        for (std::size_t island = nodes_k - islands_k; island != nodes_k; ++island) {
            for (std::size_t node = 0; node != nodes_k; ++node) {
                if (node == island) continue;
                [[maybe_unused]] status_t const forward = index.unlink(static_cast<node_id_t>(island),
                                                                       static_cast<node_id_t>(node));
                [[maybe_unused]] status_t const backward = index.unlink(static_cast<node_id_t>(node),
                                                                        static_cast<node_id_t>(island));
            }
            failures += expect_eq(index.degree_of(static_cast<node_id_t>(island)).value, 0u);
        }

        // Any single attempt may draw every island as a first-round landmark; across
        // distinct seeds at least one walk must starve its frontier and re-seed.
        bool any_restarts = false;
        for (std::size_t attempt = 0; attempt != attempts_k; ++attempt) {
            context_t attempt_context {stream_key(seed, "restarts", 1 + attempt), counting_allocator_t {ledger}};
            auto const [result, status] = index.search(query, {nodes_k}, metric, attempt_context);
            failures += expect_eq(status, status_t::success_k);
            failures += expect_eq(result.count, nodes_k); // The islands included - full coverage.
            any_restarts = any_restarts || attempt_context.telemetry.restarts > 0;
        }
        failures += expect(any_restarts);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Contracts

#pragma region Races

/** Concurrent build on distinct-seeded contexts: dense unique IDs, every thread's
 *  keys placed beside its own values, and the payloads surviving the move intact. */
template <typename fixture_type_>
std::size_t test_basic_concurrent_build(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    std::size_t const nodes_count = basic_threads_k * basic_thread_nodes_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        failures += expect(bool(index.reserve(nodes_count)));

        auto threads = basic_make_threads<fixture_type_>(ledger, basic_threads_k, basic_thread_nodes_k, seed,
                                                         "build_thread", 0);
        std::vector<node_id_t> assigned_ids(nodes_count);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(basic_threads_k))) return failures + expect(false);
        std::size_t thread_failures = 0;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread >= basic_threads_k) return;
            auto const [result, status] = index.insert(
                {threads.values[thread]}, {threads.keys[thread]}, threads.metrics[thread],
                {assigned_ids.data() + thread * basic_thread_nodes_k, basic_thread_nodes_k}, threads.contexts[thread],
                keep_diverse_t {basic_connectivity_k});
            if (status != status_t::success_k) fu::standard_atomic_ref<std::size_t>(thread_failures).store_add(1);
            assert_no_locks_held();
        });

        failures += expect_eq(thread_failures, 0u);
        failures += expect_eq(index.size(), nodes_count);
        // IDs are dense & unique across threads - the watermark's dispensed-order turn.
        std::vector<node_id_t> sorted_ids = assigned_ids;
        failures += expect_contiguous_ids(std::span {sorted_ids}, 0);

        // Every thread's keys & values placed on the IDs that thread was assigned; the values
        // compare against a fresh identically-seeded draw - the insert moved the originals.
        for (std::size_t thread = 0; thread != basic_threads_k; ++thread) {
            auto expected_values = fixture_type_::make_values(
                basic_thread_nodes_k, stream_key(seed, "build_thread", thread), counting_allocator_t {ledger});
            for (std::size_t ordinal = 0; ordinal != basic_thread_nodes_k; ++ordinal) {
                node_id_t const node_id = assigned_ids[thread * basic_thread_nodes_k + ordinal];
                failures += expect(index.key_of(node_id).value == threads.keys[thread][ordinal]);
                failures += expect(fixture_type_::value_equals(index.published_value_of(node_id), //
                                                               expected_values[ordinal]));
            }
        }
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** A reader thread races writers: every result ID sits below a watermark sampled
 *  after its search, distances ascend, and the checked ID-readers stay honest under
 *  any input while the graph grows underneath them. */
template <typename fixture_type_>
std::size_t test_basic_concurrent_search(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t writer_threads_k = basic_threads_k - 1;
    constexpr std::size_t prefill_k = 32;
    std::size_t const nodes_count = prefill_k + writer_threads_k * basic_thread_nodes_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        failures += expect(bool(index.reserve(nodes_count)));
        auto prefill_metric = fixture_type_::make_metric();
        context_t prefill_context {stream_key(seed, "search_thread"), counting_allocator_t {ledger}};
        auto prefill_values = fixture_type_::make_values(prefill_k, stream_key(seed, "search_thread"),
                                                         counting_allocator_t {ledger});
        std::vector<node_id_t> prefill_ids;
        failures += basic_build<fixture_type_>(index, prefill_metric, prefill_values, prefill_ids, prefill_context,
                                               keep_diverse_t {basic_connectivity_k});

        auto threads = basic_make_threads<fixture_type_>(ledger, writer_threads_k, basic_thread_nodes_k, seed,
                                                         "search_writer", prefill_k);
        // Queries are never inserted, so any sanitizer report here is an index bug.
        auto queries = fixture_type_::make_values(basic_reader_minimal_searches_k, stream_key(seed, "search_thread", 3),
                                                  counting_allocator_t {ledger});

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writer_threads_k + 1))) return failures + expect(false);
        std::size_t thread_failures_count = 0, writer_threads_remaining_count = writer_threads_k;
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        fu::standard_atomic_ref<std::size_t> const writer_threads_remaining {writer_threads_remaining_count};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread < writer_threads_k) {
                auto const [result, status] = index.insert({threads.values[thread]}, {threads.keys[thread]},
                                                           threads.metrics[thread], {}, threads.contexts[thread],
                                                           keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.store_add(1);
                writer_threads_remaining.store_sub(1, std::memory_order_release);
                assert_no_locks_held();
                return;
            }
            if (thread != writer_threads_k) return;
            auto reader_metric = fixture_type_::make_metric();
            context_t reader_context {stream_key(seed, "search_thread", 1), counting_allocator_t {ledger}};
            splitmix64_t reader_random {stream_key(seed, "search_thread", 2)};
            uniform_below<u32_t> probe_distribution {static_cast<u32_t>(nodes_count)};
            std::size_t const results_bound = connectivity_32_k * 2; // The default expansion.
            std::size_t searches = 0;
            while (writer_threads_remaining.load(std::memory_order_acquire) != 0 ||
                   searches < basic_reader_minimal_searches_k) {
                auto const& query = queries[searches % queries.size()];
                auto const [search_result, search_status] = index.search(query, {}, reader_metric, reader_context);
                if (search_status != status_t::success_k) thread_failures.store_add(1);
                thread_failures.store_add(scored_failures(reader_context.results.data(), reader_context.results.size(),
                                                          index.size(), results_bound));
                // Any ID is a legal input to the checked readers, published or not.
                node_id_t const probe = static_cast<node_id_t>(probe_distribution(reader_random));
                std::size_t const published_before = index.size();
                auto const [probed_key, key_status] = index.key_of(probe);
                auto const probed_degree = index.degree_of(probe);
                if (static_cast<std::size_t>(probe) < published_before)
                    if (failed(key_status) || !probed_degree) thread_failures.store_add(1);
                if (failed(key_status) && key_status != status_t::malformed_k) thread_failures.store_add(1);
                ++searches;
            }
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** One planned batch over @c basic_threads_k threads, each placing a disjoint slice through
 *  @c insert_plan_t::insert, while a reader sees none of the batch or all of it: the size jumps
 *  once, and a batch key is findable only once the whole batch is. Every context lives on the
 *  plain heap rather than the index's allocator, as the executors' arenas place them. */
template <typename fixture_type_>
std::size_t test_basic_planned_batch(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using heap_context_t = typename index_t::template traversal_context_over<std::allocator<byte_t>>;
    using metric_t = decltype(fixture_type_::make_metric());
    constexpr std::size_t prefill_k = 32;
    constexpr std::size_t batch_k = basic_threads_k * basic_thread_nodes_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        heap_context_t context {stream_key(seed, "planned")};
        auto prefill_values = fixture_type_::make_values(prefill_k, stream_key(seed, "planned"),
                                                         counting_allocator_t {ledger});
        std::vector<node_id_t> prefill_ids;
        failures += basic_build<fixture_type_>(index, metric, prefill_values, prefill_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto batch_values = fixture_type_::make_values(batch_k, stream_key(seed, "planned", 1),
                                                       counting_allocator_t {ledger});
        std::vector<key_t> batch_keys(batch_k);
        for (std::size_t ordinal = 0; ordinal != batch_k; ++ordinal)
            batch_keys[ordinal] = fixture_type_::make_key(prefill_k + ordinal);
        auto queries = fixture_type_::make_values(basic_reader_minimal_searches_k, stream_key(seed, "planned", 2),
                                                  counting_allocator_t {ledger});
        std::vector<heap_context_t> contexts;
        std::vector<metric_t> metrics;
        for (std::size_t thread = 0; thread != basic_threads_k; ++thread) {
            contexts.emplace_back(stream_key(seed, "planned", 3 + thread));
            metrics.push_back(fixture_type_::make_metric());
            failures += expect_eq(index_t::reserve_traversal(contexts.back(), basic_connectivity_k, {}),
                                  status_t::success_k);
        }
        // Spawned before the plan: past its dispense, nothing may refuse.
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(basic_threads_k + 1))) return failures + expect(false);

        // An empty plan dispenses nothing and finishes at once.
        {
            auto [plan, status] = index.plan_insert(0);
            failures += expect_eq(status, status_t::success_k);
            failures += expect(plan.range.empty());
            failures += expect_eq(plan.finish(), status_t::success_k);
        }

        // The dispense happened in the plan, so the steps cannot fail; an early finish refuses and
        // keeps the plan whole.
        auto [plan, plan_status] = index.plan_insert(batch_k);
        failures += expect_eq(plan_status, status_t::success_k);
        failures += expect_eq(static_cast<std::size_t>(plan.range.first), prefill_k);
        failures += expect_eq(plan.range.size(), batch_k);
        failures += expect_eq(plan.finish(), status_t::config_mismatch_k);
        failures += expect_eq(index.size(), prefill_k);

        std::size_t thread_failures_count = 0;
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        bool finished = false;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread < basic_threads_k) {
                std::size_t const first = thread * basic_thread_nodes_k;
                plan.insert(first, std::span(batch_values).subspan(first, basic_thread_nodes_k),
                            std::span<key_t const>(batch_keys).subspan(first, basic_thread_nodes_k), metrics[thread],
                            contexts[thread], keep_diverse_t {basic_connectivity_k});
                if (thread != 0) return assert_no_locks_held();
                // The calling thread runs thread zero and holds the plan, so it finishes, retrying
                // the refusal until every slice placed.
                status_t finished_status = plan.finish();
                for (std::size_t attempt = 0; finished_status == status_t::config_mismatch_k && attempt != spin_limit_k;
                     ++attempt) {
                    std::this_thread::yield();
                    finished_status = plan.finish();
                }
                if (finished_status != status_t::success_k) thread_failures.store_add(1);
                fu::standard_atomic_ref<bool>(finished).store(true, std::memory_order_release);
                return assert_no_locks_held();
            }
            if (thread != basic_threads_k) return;
            auto reader_metric = fixture_type_::make_metric();
            heap_context_t reader_context {stream_key(seed, "planned", 3 + basic_threads_k)};
            std::size_t searches = 0;
            while (!fu::standard_atomic_ref<bool>(finished).load(std::memory_order_acquire) ||
                   searches < basic_reader_minimal_searches_k) {
                std::size_t const published = index.size();
                if (published != prefill_k && published != prefill_k + batch_k) thread_failures.store_add(1);
                auto const [result, status] = index.search(queries[searches % queries.size()], {}, reader_metric,
                                                           reader_context);
                if (status != status_t::success_k) thread_failures.store_add(1);
                thread_failures.store_add(scored_failures(reader_context.results.data(), reader_context.results.size(),
                                                          index.size(), connectivity_32_k * 2));
                // Keys are filed after the turn, so a findable batch key implies the whole batch is
                // published.
                if (index.contains(batch_keys.back()) && index.size() != prefill_k + batch_k)
                    thread_failures.store_add(1);
                ++searches;
            }
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), prefill_k + batch_k);
        failures += expect_eq(plan.finish(), status_t::uninitialized_k); // Spent: no second turn.

        // Every slice's keys and values placed on the plan's IDs, in position order.
        auto expected_values = fixture_type_::make_values(batch_k, stream_key(seed, "planned", 1),
                                                          counting_allocator_t {ledger});
        for (std::size_t ordinal = 0; ordinal != batch_k; ++ordinal) {
            node_id_t const node_id = static_cast<node_id_t>(plan.range.first + ordinal);
            failures += expect(index.key_of(node_id).value == batch_keys[ordinal]);
            failures += expect(index.find(batch_keys[ordinal]).value == node_id);
            failures += expect(
                fixture_type_::value_equals(index.published_value_of(node_id), expected_values[ordinal]));
        }
        failures += expect_eq(index.health(context).status, status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Manual wiring raced against inserts, searches & saves - the regression net for
 *  the census word and the duplicate-scan: overlapping saves answer @c contended_k,
 *  settled saves parse duplicate-free and reload cleanly. */
template <typename fixture_type_>
std::size_t test_basic_wiring_race(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t inserts_k = 8;
    constexpr std::size_t wiring_rounds_k = 4000;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "wiring_race"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "wiring_race"),
                                                 counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k + inserts_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        auto insert_values = fixture_type_::make_values(inserts_k, stream_key(seed, "wiring_race", 4),
                                                        counting_allocator_t {ledger});
        std::vector<key_t> insert_keys(inserts_k);
        for (std::size_t ordinal = 0; ordinal != inserts_k; ++ordinal)
            insert_keys[ordinal] = fixture_type_::make_key(nodes_k + ordinal);
        auto queries = fixture_type_::make_values(4, stream_key(seed, "wiring_race", 3), counting_allocator_t {ledger});

        std::size_t thread_failures_count = 0, contended_saves_count = 0, settled_saves_count = 0;
        u32_t unexpected_status_word = ~u32_t {0}, unexpected_thread_word = ~u32_t {0};
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        fu::standard_atomic_ref<std::size_t> const contended_saves {contended_saves_count};
        fu::standard_atomic_ref<std::size_t> const settled_saves {settled_saves_count};
        fu::standard_atomic_ref<u32_t> const unexpected_status {unexpected_status_word};
        fu::standard_atomic_ref<u32_t> const unexpected_thread {unexpected_thread_word};
        auto record_unexpected = [&](std::size_t thread, status_t status) noexcept {
            thread_failures.store_add(1);
            unexpected_status.store(static_cast<u32_t>(status), std::memory_order_relaxed);
            unexpected_thread.store(static_cast<u32_t>(thread), std::memory_order_relaxed);
        };
        bool wiring_done_word = false, inserting_done_word = false;
        fu::standard_atomic_ref<bool> const wiring_done {wiring_done_word}, inserting_done {inserting_done_word};

        auto saver_thread = [&]() noexcept { // Contends while the others run, settles after.
            basic_columns_storage<index_t> storage;
            std::size_t save_attempts = 0;
            // Even attempts take the single-call form, odd ones the plan form with two
            // regions - the same hold, so both alternately meet the same racers.
            auto attempt_save = [&]() noexcept {
                std::size_t const attempt = save_attempts++;
                auto const [extents, measure_status] = index.columns_extents(codec);
                if (failed(measure_status)) return measure_status;
                storage = basic_columns_storage<index_t> {extents};
                if (attempt % 2 == 0) return index.save(storage.columns(), codec);
                auto [plan, plan_status] = index.plan_save(storage.columns(), codec);
                if (failed(plan_status)) return plan_status;
                node_id_t const covered_bound = static_cast<node_id_t>(plan.columns.keys.size());
                node_id_t const covered_middle = covered_bound / 2;
                if (status_t const front = plan.save({0, covered_middle}, codec); failed(front)) return front;
                if (status_t const back = plan.save({covered_middle, covered_bound}, codec); failed(back)) return back;
                return plan.finish();
            };
            // The race phase runs as long as the racers do - contention & post-sizing
            // growth are the expected answers, anything else is a defect.
            for (std::size_t attempts = 0;
                 !(wiring_done.load(std::memory_order_acquire) && inserting_done.load(std::memory_order_acquire));
                 ++attempts) {
                if (attempts == spin_limit_k) return record_unexpected(0, status_t::internal_k); // A stuck racer.
                status_t const saved = attempt_save();
                if (saved == status_t::contended_k || saved == status_t::overflow_k) contended_saves.store_add(1);
                else if (saved == status_t::success_k) settled_saves.store_add(1);
                else return record_unexpected(0, saved);
                std::this_thread::yield();
            }
            // The settle phase: with every racer done, a save must soon succeed.
            for (std::size_t attempt = 0; attempt != attempt_limit_k; ++attempt) {
                status_t const saved = attempt_save();
                if (saved == status_t::success_k) return settled_saves.store_add(1);
                if (saved != status_t::contended_k && saved != status_t::overflow_k) return record_unexpected(0, saved);
            }
            record_unexpected(0, status_t::contended_k); // The settle phase never settled.
        };
        auto inserter_thread = [&]() noexcept {
            auto thread_metric = fixture_type_::make_metric();
            context_t thread_context {stream_key(seed, "wiring_race", 1), counting_allocator_t {ledger}};
            auto const [result, status] = index.insert({insert_values}, {insert_keys}, thread_metric, {},
                                                       thread_context, keep_diverse_t {basic_connectivity_k});
            if (status != status_t::success_k) record_unexpected(1, status);
            inserting_done.store(true, std::memory_order_release);
            assert_no_locks_held();
        };
        auto wirer_thread = [&]() noexcept { // Every pair ends removed; it keeps wiring until
                                             // the saver has provably contended at least once.
            for (std::size_t round = 0;
                 round != attempt_limit_k &&
                 (round < wiring_rounds_k || contended_saves.load(std::memory_order_acquire) == 0);
                 ++round) {
                node_id_t const from = static_cast<node_id_t>(round % 8);
                node_id_t const to = static_cast<node_id_t>(8 + round % 8);
                auto const [membership, status] = index.link(from, to);
                if (failed(status)) record_unexpected(2, status);
                // A racing reverse re-selection may prune the manual edge first -
                // the documented advisory-wiring semantics - so absence is legal.
                status_t const unlinked = index.unlink(from, to);
                if (unlinked != status_t::success_k && unlinked != status_t::malformed_k)
                    record_unexpected(2, unlinked);
            }
            wiring_done.store(true, std::memory_order_release);
            assert_no_locks_held();
        };
        auto searcher_thread = [&]() noexcept {
            auto thread_metric = fixture_type_::make_metric();
            context_t thread_context {stream_key(seed, "wiring_race", 2), counting_allocator_t {ledger}};
            std::size_t searches = 0;
            while (!wiring_done.load(std::memory_order_acquire) || !inserting_done.load(std::memory_order_acquire) ||
                   searches < basic_reader_minimal_searches_k) {
                auto const [result, status] = index.search(queries[searches % queries.size()], {}, thread_metric,
                                                           thread_context);
                if (status != status_t::success_k) record_unexpected(3, status);
                if (scored_failures(thread_context.results.data(), thread_context.results.size(), index.size(),
                                    connectivity_32_k * 2))
                    record_unexpected(3, status_t::internal_k);
                ++searches;
            }
            assert_no_locks_held();
        };

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(4))) return failures + expect(false);
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) return saver_thread();
            if (thread == 1) return inserter_thread();
            if (thread == 2) return wirer_thread();
            if (thread == 3) return searcher_thread();
        });
        if (thread_failures.load())
            fmt::print(stderr, "  wiring-race diag: thread {} saw status {}\n", unexpected_thread.load(),
                       status_to_string(static_cast<status_t>(unexpected_status.load())));
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect(contended_saves.load() + settled_saves.load() > 0);
        failures += expect(settled_saves.load() > 0);
        for (std::size_t pair = 0; pair != 8; ++pair)
            failures += expect_eq(index.unlink(static_cast<node_id_t>(pair), static_cast<node_id_t>(8 + pair)),
                                  status_t::malformed_k);

        // The settled columns must hold duplicate-free neighbor lists and reload - the B2 audit.
        auto const [storage, status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(status));
        basic_columns_storage<index_t> const& columns = storage;
        for (std::size_t node = 0; node != columns.keys.size(); ++node) {
            std::vector<node_id_t> neighbors(columns.neighbor_ids.begin() + columns.neighbor_offsets[node],
                                             columns.neighbor_ids.begin() + columns.neighbor_offsets[node + 1]);
            std::sort(neighbors.begin(), neighbors.end());
            failures += expect(std::unique(neighbors.begin(), neighbors.end()) == neighbors.end());
        }
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect(bool(restored.reserve(nodes_k + inserts_k)));
        failures += expect_eq(restored.load(columns.view(), codec), status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Concurrent growth: threads submit batches past a one-chunk reservation, so claims ending past
 *  the first chunk race to cover the next, and every batch arrives with dense IDs, no lost entries,
 *  and every key on its own node across the chunk boundary. */
template <typename fixture_type_>
std::size_t test_basic_growth_race(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    constexpr std::size_t threads_k = 4;
    constexpr std::size_t batch_k = 32;
    constexpr std::size_t batches_per_thread_k = 9;
    constexpr std::size_t per_thread_k = batch_k * batches_per_thread_k; // 1'152 nodes outgrow a 1'024-node chunk
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        failures += expect(bool(index.reserve(1)));
        std::size_t const first_chunk_rows = index.capacity();
        failures += expect(first_chunk_rows < threads_k * per_thread_k);
        auto threads = basic_make_threads<fixture_type_>(ledger, threads_k, per_thread_k, seed, "growth_race", 0);
        std::vector<node_id_t> assigned_ids(threads_k * per_thread_k);
        std::array<status_t, threads_k> thread_statuses {};

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(threads_k))) return failures + expect(false);
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread >= threads_k) return;
            std::span<node_id_t> const thread_ids =
                std::span(assigned_ids).subspan(thread * per_thread_k, per_thread_k);
            status_t status = status_t::success_k;
            for (std::size_t batch = 0; batch != batches_per_thread_k && succeeded(status); ++batch)
                status = index
                             .insert(std::span(threads.values[thread]).subspan(batch * batch_k, batch_k),
                                     std::span<key_t const>(threads.keys[thread]).subspan(batch * batch_k, batch_k),
                                     threads.metrics[thread], thread_ids.subspan(batch * batch_k, batch_k),
                                     threads.contexts[thread], keep_diverse_t {basic_connectivity_k})
                             .status;
            thread_statuses[thread] = status;
            assert_no_locks_held();
        });

        for (std::size_t thread = 0; thread != threads_k; ++thread)
            failures += expect_eq(thread_statuses[thread], status_t::success_k);
        failures += expect_eq(index.size(), threads_k * per_thread_k);
        failures += expect(index.capacity() >= index.size());
        for (std::size_t thread = 0; thread != threads_k; ++thread)
            for (std::size_t ordinal = 0; ordinal != per_thread_k; ++ordinal)
                failures += expect(index.key_of(assigned_ids[thread * per_thread_k + ordinal]).value ==
                                   threads.keys[thread][ordinal]);
        failures += expect_contiguous_ids(std::span {assigned_ids}, 0);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** The reader's cursor: ID-order iteration over the watermark, snapshot
 *  discipline, edge-following, refresh widening, and a thread racing live mutators. */
template <typename fixture_type_>
std::size_t test_basic_cursor(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    using cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, counting_allocator_t>;
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t racing_inserts_k = 8;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        context_t context {stream_key(seed, "cursor"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k + 1 + racing_inserts_k, stream_key(seed, "cursor"),
                                                 counting_allocator_t {ledger});
        std::vector<typename index_t::value_t> extras;
        for (std::size_t extra = 0; extra != 1 + racing_inserts_k; ++extra)
            extras.push_back(std::move(values[nodes_k + extra]));
        values.resize(nodes_k);
        failures += expect(bool(index.reserve(nodes_k + 1 + racing_inserts_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context,
                                               keep_diverse_t {basic_connectivity_k});

        // A fresh cursor stands past the end; seek(0) enters the graph.
        cursor_t cursor {index, counting_allocator_t {ledger}};
        failures += expect(!cursor);
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_k);
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        std::size_t visited = 0;
        for (; cursor; ++cursor) {
            node_id_t const node_id = cursor.id();
            failures += expect(node_id < cursor.visible_count());
            failures += expect(cursor.key() == index.key_of(node_id).value);
            failures += expect(fixture_type_::value_equals(cursor.value(), index.published_value_of(node_id)));
            ++visited;
        }
        failures += expect_eq(visited, nodes_k);

        // Edge-following arrives at the staged neighbor, or refuses past the snapshot.
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        auto const staged = cursor.neighbors();
        std::vector<node_id_t> first_row {staged.begin(), staged.end()};
        failures += expect_eq(first_row.size(), index.degree_of(0).value);
        for (std::size_t position = 0; position != first_row.size(); ++position) {
            failures += expect_eq(cursor.seek(0), status_t::success_k);
            status_t const followed = cursor.follow(position);
            if (succeeded(followed)) failures += expect_eq(cursor.id(), first_row[position]);
            else failures += expect_eq(followed, status_t::malformed_k);
        }
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        failures += expect_eq(cursor.follow(cursor.neighbors().size()), status_t::malformed_k);

        // The snapshot is sticky until refreshed.
        std::array<key_t, 1> const late_key {fixture_type_::make_key(nodes_k)};
        failures += expect_eq(index.insert({extras.data(), 1}, {late_key}, metric, {}, context).status,
                              status_t::success_k);
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_k);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_k)), status_t::malformed_k);
        cursor.refresh();
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_k + 1);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_k)), status_t::success_k);

        // Parked before the race, checked after it: a past-the-end cursor must never
        // resurrect through a refresh, however far the watermark moved meanwhile.
        cursor_t parked_cursor {index, counting_allocator_t {ledger}};

        // A cursor thread races an inserting & wiring thread; every visited ID stays below
        // that pass's snapshot, and the final refresh converges on the full prefix.
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::size_t thread_failures_count = 0;
        fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
        bool writer_done = false;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                auto thread_metric = fixture_type_::make_metric();
                context_t thread_context {stream_key(seed, "cursor", 1), counting_allocator_t {ledger}};
                std::vector<key_t> thread_keys(racing_inserts_k);
                for (std::size_t ordinal = 0; ordinal != racing_inserts_k; ++ordinal)
                    thread_keys[ordinal] = fixture_type_::make_key(nodes_k + 1 + ordinal);
                auto const [result, status] = index.insert({extras.data() + 1, racing_inserts_k}, {thread_keys},
                                                           thread_metric, {}, thread_context,
                                                           keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.store_add(1);
                for (std::size_t round = 0; round != 512; ++round) {
                    [[maybe_unused]] auto const linked = index.link(1, 2 + round % 8);
                    [[maybe_unused]] status_t const unlinked = index.unlink(1, 2 + round % 8);
                }
                fu::standard_atomic_ref<bool>(writer_done).store(true, std::memory_order_release);
                return;
            }
            if (thread != 1) return;
            cursor_t racing_cursor {index, counting_allocator_t {ledger}};
            std::size_t sweeps = 0;
            do {
                if (sweeps++ == attempt_limit_k) return thread_failures.store_add(1); // A stuck writer.
                racing_cursor.refresh();
                node_id_t const snapshot = racing_cursor.visible_count();
                if (failed(racing_cursor.seek(0))) {
                    thread_failures.store_add(1);
                    return;
                }
                for (; racing_cursor; ++racing_cursor)
                    if (racing_cursor.id() >= snapshot) thread_failures.store_add(1);
            } while (!fu::standard_atomic_ref<bool>(writer_done).load(std::memory_order_acquire));
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        cursor.refresh();
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), index.size());

        // The parked cursor re-anchors past the end: an explicit seek enters the tail.
        parked_cursor.refresh();
        failures += expect(!parked_cursor);
        failures += expect_eq(static_cast<std::size_t>(parked_cursor.id()), index.size());
        failures += expect_eq(parked_cursor.seek(0), status_t::success_k);

        // The certified mode agrees with the concurrent one on a settled graph.
        ram_cursor<index_t, synchronization_t::immutable_k, counting_allocator_t> settled_cursor {
            index, counting_allocator_t {ledger}};
        failures += expect_eq(settled_cursor.seek(0), status_t::success_k);
        std::size_t settled_visited = 0;
        for (; settled_cursor; ++settled_cursor) ++settled_visited;
        failures += expect_eq(settled_visited, index.size());
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

#pragma endregion Races

#pragma region Adapters

/** The SafeGraphs round trip through the shared glue: the monolith's header names the
 *  key dtype, an irregular value column of the codec's dtype, an irregular neighbors column
 *  and the metric token; memory and file forms hold the same bytes at the default alignment
 *  and at the floor, serial and pooled; and a fresh index restored from either mirrors the
 *  original. Composed per unit, never by @c test_basic_scenarios: the key dtype's name and
 *  the metric token are unit knowledge - sentences passes "u64", while bioinformatics and
 *  meshes both carry the @c uuid128_t key and pass "uuid". */
template <typename fixture_type_>
std::size_t test_basic_safegraphs_roundtrip(u64_t seed, std::string_view key_dtype_name,
                                            std::string_view metric_token) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    using codec_t = decltype(fixture_type_::make_codec());
    constexpr std::size_t nodes_k = 24;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "safegraphs"), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "safegraphs"),
                                                 counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, keep_diverse_t {4});
        // One manual edge de-uniformizes the degrees: the irregular-neighbors mapping.
        [[maybe_unused]] auto const bumped = index.link(0, static_cast<node_id_t>(nodes_k / 2));
        auto const [extents, measure_status] = index.columns_extents(codec);
        failures += expect(succeeded(measure_status));
        serial_executor_t serial;

        // One monolith in memory under @p alignment, parsed back field by field and restored.
        auto roundtrip_in_memory = [&](std::size_t alignment, auto& executor) {
            save_config_t config;
            config.alignment = alignment;
            auto const [bytes, bytes_status] = safegraphs_storage_requirements(index, codec, config, metric_token);
            failures += expect(succeeded(bytes_status));
            std::vector<std::byte> monolith(bytes);
            bytes_span_t const target {monolith};
            failures += expect_eq(safegraphs_write_index(target, index, codec, executor, config, metric_token),
                                  status_t::success_k);

            bytes_view_t const source {monolith};
            std::string metric_seen;
            auto const [header, header_status] = safegraphs_read_header(
                source, std::allocator<byte_t> {}, [&](safegraphs_custom_field_t field) noexcept {
                    if (field.key == "metric") metric_seen.assign(field.value.data(), field.value.size());
                });
            failures += expect(succeeded(header_status));
            if (succeeded(header_status)) {
                failures += expect_eq(header.nodes(), extents.nodes);
                failures += expect_eq(header.alignment, alignment);
                failures += expect_eq(header.data_offset % alignment, 0u);
                failures += expect(!header.key.offsets);
                failures += expect(std::string_view {dtype_name(header.key.dtype)} == key_dtype_name);
                failures += expect_eq(header.key.data.size(), extents.nodes * sizeof(typename index_t::key_t));
                failures += expect(header.value.dtype == codec_t::value_dtype_k);
                // A codec of fixed-width values writes a column addressed by multiplication;
                // anything else carries the prefix sums that address its entries.
                failures += expect_eq(header.value.offsets.has_value(), !safegraphs_value_is_fixed<codec_t>());
                if (header.value.offsets) {
                    failures += expect(header.value.offsets->dtype == dtype_t::u64_k);
                    failures += expect_eq(header.value.offsets->bytes.size(), (extents.nodes + 1) * sizeof(u64_t));
                }
                failures += expect_eq(header.value.data.size(), extents.value_bytes);
                failures += expect(header.neighbors.offsets.has_value());
                failures += expect_eq(header.neighbors.data.size(), extents.neighbors * sizeof(node_id_t));
                failures += expect_eq(header.neighbors.offsets->bytes.size(), (extents.nodes + 1) * sizeof(u64_t));
                failures += expect(metric_seen == metric_token);
            }
            auto const [summary, summary_status] = safegraphs_read_index_metadata(source);
            failures += expect(succeeded(summary_status));
            failures += expect_eq(summary.nodes, extents.nodes);
            failures += expect_eq(summary.neighbors, extents.neighbors);
            failures += expect_eq(summary.value_bytes, extents.value_bytes);
            failures += expect_eq(summary.alignment, alignment);
            failures += expect(summary.value_dtype == codec_t::value_dtype_k);
            failures += expect(summary.metric.view() == metric_token);

            std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
            index_t& restored = *restored_ptr;
            failures += expect_eq(safegraphs_read_index(source, restored, codec, executor, load_config_t {}),
                                  status_t::success_k);
            failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);
            return monolith;
        };
        std::vector<std::byte> const monolith = roundtrip_in_memory(safegraphs_alignment_default_k, serial);
        std::vector<std::byte> const floor_monolith = roundtrip_in_memory(safegraphs_alignment_floor_k, serial);
        failures += expect(floor_monolith.size() < monolith.size());

        // Regions from a pool come out byte-identically to the serial pass.
        pooled_executor_t pooled;
        failures += expect_eq(pooled.spawn(3), status_t::success_k);
        std::vector<std::byte> const pooled_monolith = roundtrip_in_memory(safegraphs_alignment_default_k, pooled);
        failures += expect(pooled_monolith == monolith);

        // The file form: the same bytes on disk, written and restored through one mapping each.
        char path[64];
        std::snprintf(path, sizeof(path), "/tmp/usearch-basic-%08x.safegraphs",
                      static_cast<unsigned>(stream_key(seed, "safegraphs") ^ reinterpret_cast<std::uintptr_t>(&index)));
        {
            auto [mapping, status] = mapped_file_t::write(path, monolith.size());
            failures += expect(succeeded(status));
            failures += expect_eq(
                safegraphs_write_index(mapping.span(), index, codec, serial, save_config_t {}, metric_token),
                status_t::success_k);
            failures += expect_eq(mapping.sync(), status_t::success_k);
        }
        {
            auto [mapping, status] = mapped_file_t::read(path);
            failures += expect(succeeded(status));
            failures += expect_eq(mapping.size(), monolith.size());
            bytes_view_t const read_back = mapping.view();
            failures += expect(std::equal(read_back.begin(), read_back.end(), monolith.begin(), monolith.end()));
            std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
            index_t& restored = *restored_ptr;
            failures += expect_eq(safegraphs_read_index(read_back, restored, codec, serial, load_config_t {}),
                                  status_t::success_k);
            failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);
        }
        std::remove(path);

        // The reservation comes from the config when the index has none.
        std::unique_ptr<index_t> const roomy_ptr = fixture_type_::make_index(ledger);
        index_t& roomy = *roomy_ptr;
        load_config_t roomy_config;
        roomy_config.capacity = nodes_k * 2;
        failures += expect_eq(safegraphs_read_index(bytes_view_t {monolith}, roomy, codec, serial, roomy_config),
                              status_t::success_k);
        failures += expect(roomy.capacity() >= nodes_k * 2);
        failures += expect_eq(roomy.size(), nodes_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** The distributed form: two indexes over halves of one value set become two part files
 *  beside a manifest named as the interchange illustrates, the manifest answers the summary
 *  by content without loading a payload, every entry resolves beside it, and each part
 *  restores on its own into a fresh index that mirrors the half that wrote it. */
template <typename fixture_type_>
std::size_t test_basic_safegraphs_distributed(u64_t seed, std::string_view metric_token) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    using value_t = typename index_t::value_t;
    constexpr std::size_t parts_k = 2;
    constexpr std::size_t nodes_k = 24;
    constexpr std::size_t half_k = nodes_k / parts_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        auto const codec = fixture_type_::make_codec();
        serial_executor_t serial;
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "safegraphs", 2),
                                                 counting_allocator_t {ledger});
        std::array<std::vector<value_t>, parts_k> halves {std::vector<value_t>(values.begin(), values.begin() + half_k),
                                                          std::vector<value_t>(values.begin() + half_k, values.end())};
        // An index holds atomics and never moves; each element is initialized straight from a
        // prvalue, which elides rather than moves, so the parts sit in the array itself.
        std::array<std::unique_ptr<index_t>, parts_k> const indexes {fixture_type_::make_index(ledger),
                                                                     fixture_type_::make_index(ledger)};
        for (std::size_t part = 0; part != parts_k; ++part) {
            auto metric = fixture_type_::make_metric();
            context_t context {stream_key(seed, "safegraphs", part), counting_allocator_t {ledger}};
            std::vector<node_id_t> assigned_ids;
            failures += expect(bool(indexes[part]->reserve(half_k)));
            failures += basic_build<fixture_type_>(*indexes[part], metric, halves[part], assigned_ids, context,
                                                   keep_diverse_t {4});
        }

        char manifest_path[64];
        std::snprintf(
            manifest_path, sizeof(manifest_path), "/tmp/usearch-basic-%08x.safegraphs",
            static_cast<unsigned>(stream_key(seed, "safegraphs") ^ reinterpret_cast<std::uintptr_t>(&indexes)));
        std::string_view const stem = safegraphs_stem(manifest_path);
        std::array<std::string, parts_k> part_paths;
        std::array<std::size_t, parts_k> part_bytes {};

        // Every part is its own monolith beside the manifest; the layout lives in the file names.
        save_config_t config;
        for (std::size_t part = 0; part != parts_k; ++part) {
            growing_buffer<char> formatted {global_allocator};
            auto const [named_path, name_status] = safegraphs_part_path(formatted, stem, part, parts_k);
            failures += expect(succeeded(name_status));
            part_paths[part] = std::string {named_path};
            auto const [part_size, measure_status] = safegraphs_storage_requirements(*indexes[part], codec, config,
                                                                                     metric_token);
            failures += expect(succeeded(measure_status));
            auto [mapping, map_status] = mapped_file_t::write(part_paths[part], part_size);
            failures += expect(succeeded(map_status));
            failures += expect_eq(
                safegraphs_write_index(mapping.span(), *indexes[part], codec, serial, config, metric_token),
                status_t::success_k);
            part_bytes[part] = mapping.size();
            failures += expect_eq(mapping.sync(), status_t::success_k);
        }
        auto part_at = [&](std::size_t part) noexcept {
            return safegraphs_part_t {safegraphs_basename(part_paths[part]), part_bytes[part], half_k};
        };
        failures += expect_eq(safegraphs_write_manifest(manifest_path, parts_k, part_at), status_t::success_k);

        // The summary is answered by content, never by extension, and by the first part's header.
        auto const [summary, summary_status] = safegraphs_read_metadata(manifest_path, counting_allocator_t {ledger});
        failures += expect(succeeded(summary_status));
        failures += expect_eq(summary.parts, parts_k);
        failures += expect_eq(summary.nodes, nodes_k);
        failures += expect(summary.metric.view() == metric_token);

        // Every entry resolves beside the manifest and restores into a mirror of its own half.
        std::vector<std::string> relative_paths;
        auto per_part = [&](safegraphs_part_t part) noexcept { relative_paths.emplace_back(part.path); };
        auto const [catalog, catalog_status] = safegraphs_read_manifest(manifest_path, counting_allocator_t {ledger},
                                                                        per_part);
        failures += expect(succeeded(catalog_status));
        failures += expect_eq(relative_paths.size(), parts_k);
        for (std::size_t part = 0; part != parts_k && part != relative_paths.size(); ++part) {
            growing_buffer<char> resolved {global_allocator};
            auto const [part_path, join_status] = safegraphs_sibling_path(resolved, manifest_path,
                                                                          relative_paths[part]);
            failures += expect(succeeded(join_status));
            failures += expect(std::string_view {part_path} == part_paths[part]);
            auto [mapping, map_status] = mapped_file_t::read(part_path);
            failures += expect(succeeded(map_status));
            std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
            index_t& restored = *restored_ptr;
            failures += expect_eq(safegraphs_read_index(mapping.view(), restored, codec, serial, load_config_t {}),
                                  status_t::success_k);
            failures += basic_expect_mirrored<fixture_type_>(restored, *indexes[part], half_k);
        }

        std::remove(manifest_path);
        for (std::string const& path : part_paths) std::remove(path.c_str());
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Hostile monoliths: every header mutation of a good monolith is refused with its documented
 *  status before a payload byte is copied, a planted out-of-range ID is the plan's refusal, a short
 *  target overflows, an illegal config is refused, and the same index restores from the intact
 *  monolith after every refusal. */
template <typename fixture_type_>
std::size_t test_basic_safegraphs_hostile(u64_t seed, std::string_view metric_token) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    using codec_t = decltype(fixture_type_::make_codec());
    constexpr std::size_t nodes_k = 16;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = fixture_type_::make_index(ledger);
        index_t& index = *index_ptr;
        auto metric = fixture_type_::make_metric();
        auto const codec = fixture_type_::make_codec();
        context_t context {stream_key(seed, "safegraphs", 1), counting_allocator_t {ledger}};
        auto values = fixture_type_::make_values(nodes_k, stream_key(seed, "safegraphs", 1),
                                                 counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build<fixture_type_>(index, metric, values, assigned_ids, context, keep_diverse_t {4});
        [[maybe_unused]] auto const bumped = index.link(0, static_cast<node_id_t>(nodes_k / 2));
        serial_executor_t serial;

        auto const [bytes, bytes_status] = safegraphs_storage_requirements(index, codec, save_config_t {},
                                                                           metric_token);
        failures += expect(succeeded(bytes_status));
        std::vector<std::byte> monolith(bytes);
        bytes_span_t const target {monolith};
        failures += expect_eq(safegraphs_write_index(target, index, codec, serial, save_config_t {}, metric_token),
                              status_t::success_k);
        auto const [parsed_header, header_status] = safegraphs_read_header(bytes_view_t {monolith});
        failures += expect(succeeded(header_status));
        safegraphs_header_t const header = parsed_header;

        auto expect_refused = [&](auto mutate, status_t wanted) {
            std::vector<std::byte> corrupted = monolith;
            mutate(corrupted);
            std::unique_ptr<index_t> const fresh_ptr = fixture_type_::make_index(ledger);
            index_t& fresh = *fresh_ptr;
            failures += expect_eq(
                safegraphs_read_index(bytes_view_t {corrupted}, fresh, codec, serial, load_config_t {}), wanted);
            // Every refusal must leave the same index loadable - the retryability contract.
            failures += expect_eq(
                safegraphs_read_index(bytes_view_t {monolith}, fresh, codec, serial, load_config_t {}),
                status_t::success_k);
        };
        // Header mutations re-frame the untouched data segment through the writer itself.
        auto rewrite_header = [&](auto mutate_header) {
            return [&, mutate_header](std::vector<std::byte>& corrupted) {
                safegraphs_header_t patched = header;
                mutate_header(patched);
                bytes_span_t const patched_target {corrupted};
                std::array<safegraphs_custom_field_t, 1> const fields {{{"metric", metric_token}}};
                [[maybe_unused]] status_t const rewritten = safegraphs_write_header(patched_target, patched, fields);
            };
        };

        expect_refused([](std::vector<std::byte>& c) { c.resize(8); }, status_t::malformed_k);
        expect_refused([](std::vector<std::byte>& c) { c[0] ^= std::byte {1}; }, status_t::malformed_k);
        expect_refused(rewrite_header([](safegraphs_header_t& h) { h.alignment = 96; }), status_t::malformed_k);
        // A non-power of two under the floor is refused too, before the floor could round it up.
        expect_refused(rewrite_header([](safegraphs_header_t& h) { h.alignment = 3; }), status_t::malformed_k);
        expect_refused(rewrite_header([](safegraphs_header_t& h) {
                           h.key.dtype = h.key.dtype == dtype_t::u64_k ? dtype_t::u32_k : dtype_t::u64_k;
                       }),
                       status_t::config_mismatch_k);
        expect_refused(rewrite_header([](safegraphs_header_t& h) {
                           h.value.dtype = h.value.dtype == dtype_t::u8_k ? dtype_t::utf8_k : dtype_t::u8_k;
                       }),
                       status_t::config_mismatch_k);
        // A shape no codec declared is refused whichever layout the value column takes.
        expect_refused(rewrite_header([](safegraphs_header_t& h) { h.value.shape = matrix_shape_t {5, 7}; }),
                       status_t::config_mismatch_k);
        if constexpr (!safegraphs_value_is_fixed<codec_t>())
            expect_refused(rewrite_header([](safegraphs_header_t& h) { h.value.offsets->dtype = dtype_t::i32_k; }),
                           status_t::config_mismatch_k);
        expect_refused(rewrite_header([](safegraphs_header_t& h) {
                           h.neighbors.dtype = h.neighbors.dtype == dtype_t::u32_k ? dtype_t::u64_k : dtype_t::u32_k;
                       }),
                       status_t::config_mismatch_k);
        expect_refused(rewrite_header([](safegraphs_header_t& h) { h.neighbors.data.bound += h.alignment; }),
                       status_t::malformed_k);
        expect_refused(rewrite_header([](safegraphs_header_t& h) { h.key.data.bound -= 1; }), status_t::malformed_k);
        // An ID at the node count is the plan's refusal, not the header's.
        expect_refused(
            [&](std::vector<std::byte>& c) {
                std::size_t const at = header.data_offset + header.neighbors.data.first;
                auto const planted = std::bit_cast<std::array<std::byte, sizeof(node_id_t)>>(
                    static_cast<node_id_t>(nodes_k));
                std::copy(planted.begin(), planted.end(), c.begin() + static_cast<std::ptrdiff_t>(at));
            },
            status_t::malformed_k);

        // A target one byte short overflows; an illegal alignment and a reserved provenance
        // key are refused before a byte moves.
        std::vector<std::byte> short_monolith(monolith.size() - 1);
        bytes_span_t const short_target {short_monolith};
        failures += expect_eq(
            safegraphs_write_index(short_target, index, codec, serial, save_config_t {}, metric_token),
            status_t::overflow_k);
        save_config_t misaligned;
        misaligned.alignment = 32;
        failures += expect_eq(safegraphs_write_index(target, index, codec, serial, misaligned, metric_token),
                              status_t::config_mismatch_k);
        std::array<safegraphs_custom_field_t, 1> const clashing {{{"metric", "other"}}};
        save_config_t clash;
        clash.provenance = clashing;
        failures += expect_eq(safegraphs_write_index(target, index, codec, serial, clash, metric_token),
                              status_t::config_mismatch_k);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Parts a manifest-led cell saves: more than two, so a part sits between two others. */
constexpr std::size_t basic_nvme_parts_k = 3;

/** The arena allocator a view hands every call's metrics and queries. */
using basic_nvme_scratch_t = arena_allocator<byte_t, forkunion_allocator_t>;

/**
 *  @brief The RAM group an NVMe view is judged against: the fixture's entries keyed by @p key_of
 *      over @p shards shards, saved under @c /tmp as a monolith for one shard or as a manifest
 *      beside its parts, and an executor of @p engine_type_ to view it with.
 *
 *  Each modality's suite defines its NVMe fixture beside its RAM ones and composes
 *  @ref test_basic_nvme_engines over it. A fixture names:
 *  - @c ram_fixture_t, the RAM fixture whose metric, codec and keys the RAM side builds with;
 *  - @c shard_index_t, the RAM shard over @c forkunion_allocator_t the sharded index holds, and
 *    @c nvme_t over an engine and node-ID width, the view of its checkpoint;
 *  - @c scratch_metric_t and @c query_t, what a view's call scores with and queries by;
 *  - @c metric_token_k, the token the checkpoint records, and @c coordinates_t, a sample's fixed
 *    coordinates or @c no_fixed_coordinates_t;
 *  - @c size(), the entries, @c make_value of an entry over an index's allocator,
 *    @c make_scratch_metric over a call's arena, @c query_of an entry and @c stored_query of a
 *    stored view over the same arena, and @c matches between an entry and a stored view;
 *  - optionally @c wide_fixture_t, a RAM fixture of 64-bit node IDs, for the widened view.
 */
template <typename fixture_type_, typename engine_type_>
struct basic_nvme_reference {
    using fixture_t = fixture_type_;
    using shard_index_t = typename fixture_t::shard_index_t;
    using ram_t = ram_indexes<shard_index_t>;
    using nvme_t = typename fixture_t::template nvme_t<engine_type_>;
    using key_t = typename shard_index_t::key_t;
    using value_t = typename shard_index_t::value_t;
    using node_id_t = typename shard_index_t::node_id_t;
    using distance_t = typename shard_index_t::distance_t;
    using executor_t = typename nvme_t::executor_t;
    using metric_t = typename fixture_t::ram_fixture_t::metric_t;
    using context_t = typename shard_index_t::traversal_context_t;

    fixture_t const& fixture;
    ram_t built;
    metric_t metric;
    context_t context;
    std::vector<key_t> keys;

    /** Every entry's value once more, the RAM side's queries, as the insert moved the first. */
    std::vector<value_t> queries;
    std::vector<node_id_t> watermarks;
    std::string stem, path;
    fu_topology_t topology = nullptr;
    executor_t executor;
    std::size_t failures = 0;

    template <typename key_of_type_ = key_t (*)(std::size_t)>
    basic_nvme_reference(fixture_t const& fixture, std::string_view scenario, std::size_t shards, u64_t seed,
                         key_of_type_&& key_of = &fixture_t::ram_fixture_t::make_key)
        : fixture(fixture), built(forkunion_allocator_t {no_memory_domain_k}),
          metric(fixture_t::ram_fixture_t::make_metric()),
          context(stream_key(seed, scenario), forkunion_allocator_t {no_memory_domain_k}) {
        forkunion_allocator_t const allocator {no_memory_domain_k};
        std::size_t const count = fixture.size();
        std::vector<value_t> values;
        for (std::size_t entry = 0; entry != count; ++entry) {
            keys.push_back(key_of(entry));
            values.push_back(fixture.template make_value<shard_index_t>(entry, allocator));
            queries.push_back(fixture.template make_value<shard_index_t>(entry, allocator));
        }
        failures += expect_eq(shard_index_t::reserve_traversal(context, basic_connectivity_k, {}), status_t::success_k);
        for (std::size_t shard = 0; shard != shards; ++shard)
            failures += expect_eq(built.emplace(no_memory_domain_k), status_t::success_k);
        auto [plan, plan_status] = built.plan_insert(count);
        failures += expect_eq(plan_status, status_t::success_k);
        if (failed(plan_status)) return;
        plan.insert(0, std::span<value_t> {values}, std::span<key_t const> {keys}, metric, context,
                    keep_diverse_t {basic_connectivity_k});
        failures += expect_eq(plan.finish(), status_t::success_k);
        watermarks.resize(built.shards_count());
        failures += expect_eq(built.snapshot(watermarks).status, status_t::success_k);

        u64_t const salt = mix_bits(reinterpret_cast<std::uintptr_t>(this) ^ stream_key(seed, scenario));
        stem = fmt::format("/tmp/usearch-basic-nvme-{:016x}", salt);
        path = stem + ".safegraphs";
        serial_executor_t serial;
        failures += expect_eq(built.save(path, fixture_t::ram_fixture_t::make_codec(), serial, save_config_t {},
                                         fixture_t::metric_token_k),
                              status_t::success_k);

        failures += expect(fu_topology_new(&topology) == fu_success_k);
        auto [spawned, spawn_status] = executor_t::spawn(topology, 2, io_engine_config_t {});
        failures += expect_eq(spawn_status, status_t::success_k);
        executor = std::move(spawned);
    }
    basic_nvme_reference(basic_nvme_reference const&) = delete;
    basic_nvme_reference& operator=(basic_nvme_reference const&) = delete;
    ~basic_nvme_reference() noexcept {
        executor = executor_t {};
        if (topology) fu_topology_delete(topology);
        for (std::string const& file : tools::files_starting_with(stem)) std::remove(file.c_str());
    }

    /** Views the checkpoint through @p viewed, its parts served from @p paths where named. */
    status_t view(nvme_t& viewed, std::span<std::string_view const> paths = {}) noexcept {
        return viewed.view(path, paths, fixture_t::metric_token_k, executor);
    }

    /** The path of part @p part beside the manifest, formatted into @p buffer. */
    std::string_view part_path(growing_buffer<char>& buffer, std::size_t part) const noexcept {
        auto const [named, status] = safegraphs_part_path(buffer, safegraphs_stem(path), part, built.shards_count());
        return succeeded(status) ? named : std::string_view {};
    }

    /** A view's search of every entry's query, as many per query as @p counts leaves room for in
     *  @p found_keys, under walks @p expansion wide. */
    expected<typename nvme_t::search_result_t> search(nvme_t& viewed, std::size_t expansion,
                                                      std::span<key_t> found_keys,
                                                      std::span<distance_t> found_distances,
                                                      std::span<std::size_t> counts) noexcept {
        typename nvme_t::search_config_t config;
        config.expansion_factor = expansion;
        return viewed.search(fixture.size(), make_metric(), query_of(), found_keys, found_distances, counts, config,
                             executor);
    }

    /** The view's metric factory over a call's arena. */
    auto make_metric() const noexcept {
        return [](basic_nvme_scratch_t const& allocator, std::size_t staging) noexcept {
            return fixture_t::make_scratch_metric(allocator, staging);
        };
    }

    /** The view's queries, entry by entry. */
    auto query_of() const noexcept {
        return [this](std::size_t entry, basic_nvme_scratch_t const& allocator) noexcept {
            return fixture.query_of(entry, allocator);
        };
    }

    /** The RAM group's rankings of every entry's query, @p wanted per query under walks as wide as
     *  the index, which rank every entry of every shard. */
    void rank(std::size_t wanted, std::vector<key_t>& ranked_keys, std::vector<distance_t>& ranked_distances) {
        std::size_t const count = fixture.size();
        ranked_keys.resize(count * wanted), ranked_distances.resize(count * wanted);
        dummy_predicate_t unfiltered;
        for (std::size_t query = 0; query != count; ++query) {
            auto const [found, status] = built.search_if(
                queries[query], watermarks, std::span {ranked_keys}.subspan(query * wanted, wanted),
                std::span {ranked_distances}.subspan(query * wanted, wanted), {count}, unfiltered, metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += expect_eq(found, wanted);
        }
    }
};

/** Every query of a view's ranking agrees with the RAM group's, both @p wanted wide and ranking
 *  every entry: rank for rank the same distances, and every key closer than the RAM ranking's last
 *  listed by the view, so ties alone may swap keys. */
template <typename key_type_, typename distance_type_>
std::size_t basic_nvme_expect_rankings(char const* cell, std::size_t wanted, std::span<key_type_ const> ram_keys,
                                       std::span<distance_type_ const> ram_distances,
                                       std::span<key_type_ const> nvme_keys,
                                       std::span<distance_type_ const> nvme_distances,
                                       std::span<std::size_t const> counts) {
    std::size_t failures = 0, unlisted = 0;
    error_stats_t errors;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        failures += expect_eq(counts[query], wanted);
        distance_type_ const last = ram_distances[query * wanted + wanted - 1];
        for (std::size_t rank = 0; rank != wanted; ++rank) {
            std::size_t const match = query * wanted + rank;
            errors.add(static_cast<f64_t>(nvme_distances[match]), static_cast<f64_t>(ram_distances[match]));
            if (!(ram_distances[match] < last)) continue;
            unlisted += std::none_of(nvme_keys.begin() + query * wanted, nvme_keys.begin() + (query + 1) * wanted,
                                     [&](key_type_ const& key) { return key == ram_keys[match]; });
        }
    }
    failures += expect_eq(unlisted, 0u);
    return failures +
           expect_errors(cell, basic_comparison_family<distance_type_>(), comparison_verdict_t::assert_k, errors, 0);
}

/** @p metric_type_ seen through its single-query @c reset over @p query_type_ alone, so a scan
 *  reopens every query per round as it does for a metric without a batched form. */
template <typename metric_type_, typename query_type_>
struct per_query_metric {
    metric_type_ metric;

    void reset(query_type_ const& query) noexcept { metric.reset(query); }

    template <typename value_type_>
    metric_staging_t push(value_type_ const& value) noexcept {
        return metric.push(value);
    }

    template <typename distance_type_>
    void drain(std::span<distance_type_> distances, distance_type_ radius) noexcept {
        metric.drain(distances, radius);
    }
};

/**
 *  @brief A monolith viewed from storage: the fixture's entries saved by a RAM index of one shard.
 *
 *  Every entry finds itself at distance zero and ranks the same nearest entries as the RAM index it
 *  was saved from, both with candidates as wide as the graph; a scan ranks alike, one query at a
 *  time or batched, under a predicate admitting everything or one in eight entries, answered a quad
 *  at a time or gathered per thread; @c count finds every key once and an absent one never, and
 *  @c get and @c read_entries hand back every entry's own value.
 */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_search(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using nvme_t = typename reference_t::nvme_t;
    using key_t = typename reference_t::key_t;
    using distance_t = typename reference_t::distance_t;
    using search_config_t = typename nvme_t::search_config_t;
    constexpr std::size_t wanted_k = 4;
    reference_t reference {fixture, "nvme_search", 1, seed};
    std::size_t const count = fixture.size();
    std::vector<key_t> ram_keys;
    std::vector<distance_t> ram_distances;
    reference.rank(wanted_k, ram_keys, ram_distances);
    nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    failures += expect_eq(viewed.size(), count);
    if (failures) return failures;
    auto& executor = reference.executor;
    auto const make_metric = reference.make_metric();
    auto const query_of = reference.query_of();

    // Candidates as wide as the graph walk all of it, so every entry meets itself.
    std::vector<key_t> keys(count * wanted_k);
    std::vector<distance_t> distances(count * wanted_k);
    std::vector<std::size_t> counts(count);
    auto const [found, search_status] = reference.search(viewed, count, keys, distances, counts);
    failures += expect_eq(search_status, status_t::success_k);
    failures += expect_eq(found.count, count * wanted_k);
    constexpr comparison_family_t comparison_family_k = basic_comparison_family<distance_t>();
    error_stats_t self_errors;
    for (std::size_t query = 0; query != count; ++query)
        self_errors.add(static_cast<f64_t>(distances[query * wanted_k]), 0.0);
    failures += expect_errors("nvme self distances", comparison_family_k, comparison_verdict_t::assert_k, self_errors,
                              0);
    failures += basic_nvme_expect_rankings<key_t, distance_t>("nvme search distances", wanted_k, ram_keys,
                                                              ram_distances, keys, distances, counts);

    // A scan reads every entry, so its distances are the whole-graph walk's, rank for rank.
    auto const [scanned, scan_status] = viewed.scan(count, make_metric, query_of, std::span<key_t> {keys},
                                                    std::span<distance_t> {distances}, std::span<std::size_t> {counts},
                                                    executor);
    failures += expect_eq(scan_status, status_t::success_k);
    failures += expect_eq(scanned.count, count * wanted_k);
    error_stats_t scan_errors;
    for (std::size_t match = 0; match != count * wanted_k; ++match)
        scan_errors.add(static_cast<f64_t>(distances[match]), static_cast<f64_t>(ram_distances[match]));
    failures += expect_errors("nvme scan distances", comparison_family_k, comparison_verdict_t::assert_k, scan_errors,
                              0);

    // Scoring one query at a time ranks the very entries a batched metric does, and a predicate
    // admitting everything hears each entry at most once whatever the queries, changing nothing.
    using metric_t = typename fixture_type_::scratch_metric_t;
    using query_t = typename fixture_type_::query_t;
    static_assert(batched_metric<metric_t, query_t> && !batched_metric<per_query_metric<metric_t, query_t>, query_t>);
    auto make_per_query_metric = [&](basic_nvme_scratch_t const& allocator, std::size_t staging) noexcept {
        auto [metric, metric_status] = make_metric(allocator, staging);
        return expected<per_query_metric<metric_t, query_t>> {{std::move(metric)}, metric_status};
    };
    std::size_t heard = 0;
    auto admit_all = quad_predicate([&heard](key_t const&) noexcept {
        ++heard;
        return admission_t::admit_k;
    });
    error_stats_t rescan_errors;
    for (bool const filtered : {false, true}) {
        std::vector<key_t> rescanned_keys(count * wanted_k);
        std::vector<distance_t> rescanned_distances(count * wanted_k);
        auto const [rescanned, rescan_status] =
            filtered
                ? viewed.scan(count, make_metric, query_of, std::span<key_t> {rescanned_keys},
                              std::span<distance_t> {rescanned_distances}, std::span<std::size_t> {counts}, admit_all,
                              dummy_predicate_t {}, executor)
                : viewed.scan(count, make_per_query_metric, query_of, std::span<key_t> {rescanned_keys},
                              std::span<distance_t> {rescanned_distances}, std::span<std::size_t> {counts}, executor);
        failures += expect_eq(rescan_status, status_t::success_k);
        failures += expect_eq(rescanned.count, count * wanted_k);
        for (std::size_t match = 0; match != count * wanted_k; ++match) {
            failures += expect(rescanned_keys[match] == keys[match]);
            rescan_errors.add(static_cast<f64_t>(rescanned_distances[match]), static_cast<f64_t>(distances[match]));
        }
    }
    failures += expect_errors("nvme rescan distances", comparison_family_k, comparison_verdict_t::assert_k,
                              rescan_errors, 0);
    failures += expect(heard > 0 && heard <= count);

    // A predicate admitting one entry in eight still fills every query at the default breadth, as
    // each walk stops on what it admitted, and ranks as a scan under the same predicate does,
    // whether it answers each quad or gathers each completion batch.
    std::vector<key_t> admitted_keys;
    for (std::size_t entry = 0; entry < count; entry += 8) admitted_keys.push_back(reference.keys[entry]);
    auto const is_admitted = [&](key_t const& key) noexcept {
        bool listed = false;
        for (key_t const& admitted : admitted_keys) listed |= admitted == key;
        return listed ? admission_t::admit_k : admission_t::reject_k;
    };
    std::size_t const filled = min_of(wanted_k, admitted_keys.size());
    std::vector<distance_t> scanned_distances(count * wanted_k);
    auto const [filtered_scan, filtered_scan_status] = viewed.scan(
        count, make_metric, query_of, std::span<key_t> {keys}, std::span<distance_t> {scanned_distances},
        std::span<std::size_t> {counts}, quad_predicate(is_admitted), dummy_predicate_t {}, executor);
    failures += expect_eq(filtered_scan_status, status_t::success_k);
    failures += expect_eq(filtered_scan.count, count * filled);
    auto gathering = gathering_per_thread<key_t, typename nvme_t::walk_payload_t>(is_admitted, executor.size());
    error_stats_t filtered_errors;
    for (bool const gathered : {false, true}) {
        auto const [filtered, filtered_status] =
            gathered ? viewed.search(count, make_metric, query_of, std::span<key_t> {keys},
                                     std::span<distance_t> {distances}, std::span<std::size_t> {counts},
                                     search_config_t {}, gathering, dummy_predicate_t {}, executor)
                     : viewed.search(count, make_metric, query_of, std::span<key_t> {keys},
                                     std::span<distance_t> {distances}, std::span<std::size_t> {counts},
                                     search_config_t {}, quad_predicate(is_admitted), dummy_predicate_t {}, executor);
        failures += expect_eq(filtered_status, status_t::success_k);
        failures += expect_eq(filtered.count, count * filled);
        for (std::size_t query = 0; query != count; ++query) {
            failures += expect_eq(counts[query], filled);
            for (std::size_t rank = 0; rank != filled; ++rank) {
                std::size_t const match = query * wanted_k + rank;
                failures += expect(is_admitted(keys[match]) == admission_t::admit_k);
                filtered_errors.add(static_cast<f64_t>(distances[match]), static_cast<f64_t>(scanned_distances[match]));
            }
        }
    }
    failures += expect_errors("nvme filtered distances", comparison_family_k, comparison_verdict_t::assert_k,
                              filtered_errors, 0);

    std::vector<key_t> asked(reference.keys);
    asked.push_back(fixture_type_::ram_fixture_t::make_key(count * 2));
    std::vector<std::size_t> entries(count + 1);
    failures += expect_eq(viewed.count(std::span<key_t const> {asked}, entries, executor), status_t::success_k);
    for (std::size_t entry = 0; entry != count; ++entry) failures += expect_eq(entries[entry], 1u);
    failures += expect_eq(entries[count], 0u);

    std::size_t handed = 0, mismatched = 0;
    auto on_value = [&](std::size_t position, std::size_t rank, std::size_t, auto const& view) noexcept {
        fu::standard_atomic_ref<std::size_t>(handed).store_add(1, std::memory_order_relaxed);
        if (rank || !fixture.matches(position, view))
            fu::standard_atomic_ref<std::size_t>(mismatched).store_add(1, std::memory_order_relaxed);
    };
    failures += expect_eq(viewed.get(std::span<key_t const> {asked}, entries, 1, executor, on_value),
                          status_t::success_k);
    failures += expect_eq(handed, count);
    failures += expect_eq(mismatched, 0u);

    // Entries read in placement order come in insertion order, the last one included.
    std::vector<key_t> read_keys(count - 1);
    handed = 0, mismatched = 0;
    auto on_entry = [&](std::size_t offset, std::size_t, auto const& view) noexcept {
        fu::standard_atomic_ref<std::size_t>(handed).store_add(1, std::memory_order_relaxed);
        if (!fixture.matches(offset + 1, view))
            fu::standard_atomic_ref<std::size_t>(mismatched).store_add(1, std::memory_order_relaxed);
    };
    failures += expect_eq(viewed.read_entries(1, std::span<key_t> {read_keys}, executor, on_entry),
                          status_t::success_k);
    failures += expect_eq(handed, count - 1);
    failures += expect_eq(mismatched, 0u);
    for (std::size_t offset = 0; offset + 1 != count; ++offset)
        failures += expect(read_keys[offset] == asked[offset + 1]);
    failures += expect_eq(viewed.read_entries(count, std::span<key_t> {read_keys}.first(1), executor, on_entry),
                          status_t::config_mismatch_k);
    return failures;
}

/** A save of several shards viewed through its manifest: one shard per part, every entry counted,
 *  and the parts' merged rankings those of the RAM group's merged walks as wide as the index. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_parts(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using key_t = typename reference_t::key_t;
    using distance_t = typename reference_t::distance_t;
    std::size_t const count = fixture.size(), wanted = basic_connectivity_k;
    reference_t reference {fixture, "nvme_parts", basic_nvme_parts_k, seed};
    std::vector<key_t> ram_keys, keys(count * wanted);
    std::vector<distance_t> ram_distances, distances(count * wanted);
    std::vector<std::size_t> counts(count);
    reference.rank(wanted, ram_keys, ram_distances);
    typename reference_t::nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    failures += expect_eq(viewed.shards_count(), basic_nvme_parts_k);
    failures += expect_eq(viewed.size(), count);
    if (failures) return failures;
    failures += expect_eq(reference.search(viewed, count, keys, distances, counts).status, status_t::success_k);
    return failures + basic_nvme_expect_rankings<key_t, distance_t>("nvme parts distances", wanted, ram_keys,
                                                                    ram_distances, keys, distances, counts);
}

/** The key scans across parts against the RAM group's lookups, a key shared by every third entry
 *  and an absent key included: every count and verdict agrees, and @c get hands each key's earliest
 *  entries up to the wanted, each one's value an entry stored under that key. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_key_scans(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using key_t = typename reference_t::key_t;
    using ram_fixture_t = typename fixture_type_::ram_fixture_t;
    constexpr std::size_t wanted_k = 2;
    std::size_t const count = fixture.size();
    key_t const shared = ram_fixture_t::make_key(count);
    auto const key_of = [&](std::size_t entry) { return entry % 3 ? ram_fixture_t::make_key(entry) : shared; };
    reference_t reference {fixture, "nvme_key_scans", basic_nvme_parts_k, seed, key_of};
    typename reference_t::nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    if (failures) return failures;

    std::array<key_t, 4> const probes {shared, ram_fixture_t::make_key(1), ram_fixture_t::make_key(count - 1),
                                       ram_fixture_t::make_key(count + 1)};
    std::array<std::size_t, probes.size()> counted {}, contained {}, gotten {};
    std::array<std::size_t, probes.size()> handed {};
    std::size_t foreign = 0;
    auto on_value = [&](std::size_t position, std::size_t, std::size_t, auto const& view) noexcept {
        fu::standard_atomic_ref<std::size_t>(handed[position]).store_add(1, std::memory_order_relaxed);
        bool stored = false;
        for (std::size_t entry = 0; entry != count; ++entry)
            stored |= reference.keys[entry] == probes[position] && fixture.matches(entry, view);
        if (!stored) fu::standard_atomic_ref<std::size_t>(foreign).store_add(1, std::memory_order_relaxed);
    };
    auto& executor = reference.executor;
    failures += expect_eq(viewed.count(std::span<key_t const> {probes}, counted, executor), status_t::success_k);
    failures += expect_eq(viewed.contains(std::span<key_t const> {probes}, std::span {contained}, executor),
                          status_t::success_k);
    failures += expect_eq(viewed.get(std::span<key_t const> {probes}, gotten, wanted_k, executor, on_value),
                          status_t::success_k);
    for (std::size_t probe = 0; probe != probes.size(); ++probe) {
        std::size_t const stored = reference.built.count(probes[probe]);
        failures += expect_eq(counted[probe], stored);
        failures += expect_eq(contained[probe], std::size_t {stored != 0});
        failures += expect_eq(gotten[probe], stored);
        failures += expect_eq(handed[probe], min_of(stored, wanted_k));
    }
    failures += expect_eq(counted[0], divide_round_up(count, std::size_t {3}));
    failures += expect_eq(counted[3], 0u);
    return failures + expect_eq(foreign, 0u);
}

/** The cursor walks the parts as the RAM group's cursor walks its shards, pull for pull: every
 *  entry once as stored, then breadth-first from the first entry's key with every entry's links,
 *  the same keys, the same links and the same offsets. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_cursor(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using nvme_t = typename reference_t::nvme_t;
    using ram_cursor_t = typename reference_t::ram_t::cursor_t;
    using nvme_cursor_t = typename nvme_t::cursor_t;
    using key_t = typename reference_t::key_t;
    std::size_t const count = fixture.size(), chunk = basic_connectivity_k + 1;
    reference_t reference {fixture, "nvme_cursor", basic_nvme_parts_k, seed};
    nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    if (failures) return failures;

    std::array<key_t, 1> const start {reference.keys[0]};
    for (std::span<key_t const> const starts : {std::span<key_t const> {}, std::span<key_t const> {start}}) {
        growing_arena<forkunion_allocator_t> arena {forkunion_allocator_t {no_memory_domain_k}};
        typename ram_cursor_t::links_allocator_t const links_allocator {arena};
        typename ram_cursor_t::links_keys_t ram_links {links_allocator}, nvme_links {links_allocator};
        typename ram_cursor_t::links_offsets_t ram_offsets {links_allocator}, nvme_offsets {links_allocator};
        bool const linked = !starts.empty();
        ram_cursor_t ram_cursor {forkunion_allocator_t {no_memory_domain_k}};
        nvme_cursor_t nvme_cursor {forkunion_allocator_t {no_memory_domain_k}};
        failures += expect_eq(ram_cursor.open(reference.built, starts), status_t::success_k);
        failures += expect_eq(nvme_cursor.open(viewed, starts), status_t::success_k);
        std::vector<key_t> ram_keys(chunk), nvme_keys(chunk);
        std::size_t pulled = 0, differing = 0;
        while (true) {
            auto const [ram_read, ram_status] = ram_cursor.pull(
                reference.built, std::span {ram_keys}, linked ? &ram_links : nullptr, linked ? &ram_offsets : nullptr);
            auto const [nvme_read, nvme_status] = nvme_cursor.pull(
                viewed, std::span {nvme_keys}, linked ? &nvme_links : nullptr, linked ? &nvme_offsets : nullptr,
                reference.executor);
            failures += expect_eq(ram_status, status_t::success_k);
            failures += expect_eq(nvme_status, status_t::success_k);
            failures += expect_eq(nvme_read, ram_read);
            if (!ram_read || nvme_read != ram_read) break;
            for (std::size_t entry = 0; entry != ram_read; ++entry) differing += !(nvme_keys[entry] == ram_keys[entry]);
            pulled += ram_read;
        }
        failures += expect_eq(differing, 0u);
        failures += linked ? expect(pulled > 1 && pulled <= count) : expect_eq(pulled, count);
        failures += expect_eq(nvme_links.size(), ram_links.size());
        failures += expect_eq(nvme_offsets.size(), ram_offsets.size());
        failures += expect(std::equal(nvme_links.begin(), nvme_links.end(), ram_links.begin(), ram_links.end()));
        failures += expect(
            std::equal(nvme_offsets.begin(), nvme_offsets.end(), ram_offsets.begin(), ram_offsets.end()));
    }
    return failures;
}

/** The reachability audit of a whole graph agrees with the RAM index's: every entry audited, as
 *  many reached, through at least one sweep. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_health(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    reference_t reference {fixture, "nvme_health", 1, seed};
    typename reference_t::nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    if (failures) return failures;
    auto const [ram, ram_status] = reference.built.health(reference.context);
    auto const [nvme, nvme_status] = viewed.health(reference.executor);
    failures += expect_eq(ram_status, status_t::success_k);
    failures += expect_eq(nvme_status, status_t::success_k);
    failures += expect_eq(nvme.audited_nodes, fixture.size());
    failures += expect_eq(nvme.reachable_nodes, ram.reachable_nodes);
    return failures + expect(nvme.sweeps > 0);
}

/** A sample of the parts draws what the RAM group draws under the same seed: the same keys, entry
 *  ordinals and parents at the same parent distances, every draw at finite coordinates. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_sample(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using key_t = typename reference_t::key_t;
    using coordinates_t = typename fixture_type_::coordinates_t;
    std::size_t const wanted = min_of(fixture.size(), basic_nvme_parts_k * basic_connectivity_k);
    reference_t reference {fixture, "nvme_sample", basic_nvme_parts_k, seed};
    typename reference_t::nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    if (failures) return failures;

    struct drawn_t {
        std::vector<key_t> keys, parents;
        std::vector<std::size_t> ordinals;
        std::vector<f32_t> parent_distances, coordinates;

        explicit drawn_t(std::size_t wanted)
            : keys(wanted), parents(wanted), ordinals(wanted), parent_distances(wanted), coordinates(wanted * 3) {}
        sample_outputs<key_t> outputs() noexcept { return {keys, ordinals, parents, parent_distances, coordinates}; }
    };
    drawn_t ram {wanted}, nvme {wanted};
    sample_config_t const config {stream_key(seed, "nvme_sample"), 0};
    auto const [ram_drawn, ram_status] = reference.built.sample({}, reference.watermarks, ram.outputs(), config,
                                                                reference.metric, reference.context, coordinates_t {});
    auto const stored_query = [](auto const& stored, basic_nvme_scratch_t const& allocator) noexcept {
        return fixture_type_::stored_query(stored, allocator);
    };
    auto const [nvme_drawn, nvme_status] = viewed.sample(sample_inputs<key_t> {}, nvme.outputs(), config,
                                                         reference.make_metric(), stored_query, reference.executor,
                                                         coordinates_t {});
    failures += expect_eq(ram_status, status_t::success_k);
    failures += expect_eq(nvme_status, status_t::success_k);
    failures += expect_eq(ram_drawn.written, wanted);
    failures += expect_eq(nvme_drawn.written, ram_drawn.written);
    failures += expect(nvme.keys == ram.keys && nvme.ordinals == ram.ordinals && nvme.parents == ram.parents);
    error_stats_t parent_errors;
    for (std::size_t draw = 0; draw != wanted; ++draw)
        parent_errors.add(nvme.parent_distances[draw], ram.parent_distances[draw]);
    failures += expect_errors("nvme parent distances", basic_comparison_family<typename reference_t::distance_t>(),
                              comparison_verdict_t::assert_k, parent_errors, 0);
    return failures + expect(std::all_of(nvme.coordinates.begin(), nvme.coordinates.end(),
                                         [](f32_t coordinate) { return std::isfinite(coordinate); }));
}

/** Reads by purpose over a monolith: a search reads one neighbors row per expansion, one value row
 *  per distance and one key per written result, and a filtered one reads a key beside every value
 *  row and one more per written result. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_telemetry(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using key_t = typename reference_t::key_t;
    using distance_t = typename reference_t::distance_t;
    using nvme_t = typename reference_t::nvme_t;
    std::size_t const count = fixture.size(), wanted = basic_connectivity_k;
    reference_t reference {fixture, "nvme_telemetry", 1, seed};
    nvme_t viewed {forkunion_allocator_t {}};
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.view(viewed), status_t::success_k);
    if (failures) return failures;

    std::vector<key_t> keys(count * wanted);
    std::vector<distance_t> distances(count * wanted);
    std::vector<std::size_t> counts(count);
    auto const [searched, search_status] = reference.search(viewed, 0, keys, distances, counts);
    failures += expect_eq(search_status, status_t::success_k);
    failures += expect_eq(searched.neighbor_reads.count, searched.visited_members);
    failures += expect_eq(searched.value_reads.count, searched.computed_distances);
    failures += expect_eq(searched.key_reads.count, searched.count);

    auto const odd = quad_predicate([&](key_t const& key) noexcept {
        bool listed = false;
        for (std::size_t entry = 1; entry < count; entry += 2) listed |= reference.keys[entry] == key;
        return listed ? admission_t::admit_k : admission_t::reject_k;
    });
    auto const [filtered, filtered_status] = viewed.search(
        count, reference.make_metric(), reference.query_of(), std::span<key_t> {keys},
        std::span<distance_t> {distances}, std::span<std::size_t> {counts}, typename nvme_t::search_config_t {}, odd,
        dummy_predicate_t {}, reference.executor);
    failures += expect_eq(filtered_status, status_t::success_k);
    failures += expect(filtered.count > 0);
    return failures + expect_eq(filtered.key_reads.count, filtered.value_reads.count + filtered.count);
}

/** A read that comes back short of its row stops the call: a part truncated under a live view fails
 *  search, filtered search and the key scans with @c io_k, again on every retry, and the very
 *  executor then serves an intact view exactly. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_failed_read(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using key_t = typename reference_t::key_t;
    using distance_t = typename reference_t::distance_t;
    using nvme_t = typename reference_t::nvme_t;
    std::size_t const count = fixture.size(), wanted = basic_connectivity_k;
    reference_t broken_reference {fixture, "nvme_failed_read", basic_nvme_parts_k, seed};
    reference_t intact_reference {fixture, "nvme_intact_read", 1, seed};
    nvme_t broken {forkunion_allocator_t {}}, intact {forkunion_allocator_t {}};
    std::size_t failures = broken_reference.failures + intact_reference.failures;
    failures += expect_eq(broken_reference.view(broken), status_t::success_k);
    if (failures) return failures;
    {
        growing_buffer<char> part_path {global_allocator};
        auto [file, status] = seekable_file_t::open(broken_reference.part_path(part_path, 1),
                                                    file_access_t::truncate_read_write_k);
        failures += expect_eq(status, status_t::success_k);
    }

    std::vector<key_t> keys(count * wanted);
    std::vector<distance_t> distances(count * wanted);
    std::vector<std::size_t> counts(count), entries(1);
    auto const any = quad_predicate([](key_t const&) noexcept { return admission_t::admit_k; });
    for (std::size_t attempt = 0; attempt != 2; ++attempt) {
        failures += expect_eq(broken_reference.search(broken, 0, keys, distances, counts).status, status_t::io_k);
        failures += expect_eq(
            broken
                .search(count, broken_reference.make_metric(), broken_reference.query_of(), std::span<key_t> {keys},
                        std::span<distance_t> {distances}, std::span<std::size_t> {counts},
                        typename nvme_t::search_config_t {}, any, dummy_predicate_t {}, broken_reference.executor)
                .status,
            status_t::io_k);
        failures += expect_eq(
            broken.count(std::span<key_t const> {broken_reference.keys}.first(1), entries, broken_reference.executor),
            status_t::io_k);
    }

    std::vector<key_t> ram_keys;
    std::vector<distance_t> ram_distances;
    intact_reference.rank(wanted, ram_keys, ram_distances);
    failures += expect_eq(
        intact.view(intact_reference.path, {}, fixture_type_::metric_token_k, broken_reference.executor),
        status_t::success_k);
    typename nvme_t::search_config_t config;
    config.expansion_factor = count;
    failures += expect_eq(intact
                              .search(count, broken_reference.make_metric(), broken_reference.query_of(),
                                      std::span<key_t> {keys}, std::span<distance_t> {distances},
                                      std::span<std::size_t> {counts}, config, broken_reference.executor)
                              .status,
                          status_t::success_k);
    return failures + basic_nvme_expect_rankings<key_t, distance_t>("nvme intact distances", wanted, ram_keys,
                                                                    ram_distances, keys, distances, counts);
}

/** Parts served from elsewhere than the manifest points: one moved under another name and named
 *  itself, the rest through the directory holding them. The view ranks exactly like the RAM group,
 *  while one naming too few parts is refused and one following the manifest fails to open. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_relocated(fixture_type_ const& fixture, u64_t seed) {
    using reference_t = basic_nvme_reference<fixture_type_, engine_type_>;
    using key_t = typename reference_t::key_t;
    using distance_t = typename reference_t::distance_t;
    std::size_t const count = fixture.size(), wanted = basic_connectivity_k;
    reference_t reference {fixture, "nvme_relocated", basic_nvme_parts_k, seed};
    std::size_t failures = reference.failures;
    if (failures) return failures;

    std::string const moved = reference.stem + "-moved.safegraphs";
    growing_buffer<char> part_path {global_allocator};
    failures += expect_eq(std::rename(reference.part_path(part_path, 1).data(), moved.c_str()), 0);
    std::string const directory = reference.path.substr(0, reference.path.rfind('/'));
    std::array<std::string_view, basic_nvme_parts_k> paths;
    paths.fill(directory);
    paths[1] = moved;

    typename reference_t::nvme_t relocated {forkunion_allocator_t {}};
    failures += expect_eq(reference.view(relocated, std::span {paths}.first(1)), status_t::config_mismatch_k);
    failures += expect(failed(reference.view(relocated)));
    failures += expect_eq(reference.view(relocated, paths), status_t::success_k);
    failures += expect_eq(relocated.size(), count);

    std::vector<key_t> ram_keys, keys(count * wanted);
    std::vector<distance_t> ram_distances, distances(count * wanted);
    std::vector<std::size_t> counts(count);
    reference.rank(wanted, ram_keys, ram_distances);
    failures += expect_eq(reference.search(relocated, count, keys, distances, counts).status, status_t::success_k);
    return failures + basic_nvme_expect_rankings<key_t, distance_t>("nvme relocated distances", wanted, ram_keys,
                                                                    ram_distances, keys, distances, counts);
}

/**
 *  @brief A monolith viewed at its own node-ID width: the fixture's entries saved by a RAM index of
 *      64-bit node IDs, searched through a view of 64-bit IDs and refused by one of 32-bit IDs.
 *
 *  Every entry finds itself at distance zero through candidates as wide as the graph, and the
 *  mismatched view answers @c config_mismatch_k and stays empty, as the RAM loader refuses it.
 */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_widened_keys(fixture_type_ const& fixture, u64_t seed) {
    using wide_fixture_t = typename fixture_type_::wide_fixture_t;
    using index_t = typename wide_fixture_t::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    using key_t = typename index_t::key_t;
    using nvme_t = typename fixture_type_::template nvme_t<engine_type_, node_id_t>;
    using mismatched_t = typename fixture_type_::template nvme_t<engine_type_, u32_t>;
    using distance_t = typename nvme_t::distance_t;
    using executor_t = typename nvme_t::executor_t;
    static_assert(!std::is_same_v<node_id_t, u32_t>);
    std::size_t const count = fixture.size();
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    std::string const path = fmt::format(
        "/tmp/usearch-basic-nvme-ids-{:016x}.safegraphs",
        mix_bits(stream_key(seed, "nvme_node_ids") ^ reinterpret_cast<std::uintptr_t>(&ledger)));
    {
        std::unique_ptr<index_t> const index_ptr = wide_fixture_t::make_index(ledger);
        index_t& index = *index_ptr;
        std::vector<typename index_t::value_t> values;
        for (std::size_t entry = 0; entry != count; ++entry)
            values.push_back(fixture.template make_value<index_t>(entry, counting_allocator_t {ledger}));
        auto metric = wide_fixture_t::make_metric();
        context_t context {stream_key(seed, "nvme_node_ids"), counting_allocator_t {ledger}};
        std::vector<node_id_t> assigned_ids;
        failures += expect(bool(index.reserve(count)));
        failures += basic_build<wide_fixture_t>(index, metric, values, assigned_ids, context,
                                                keep_diverse_t {basic_connectivity_k});
        serial_executor_t serial;
        failures += expect_eq(safegraphs_write_index(std::string_view {path}, index, wide_fixture_t::make_codec(),
                                                     serial, save_config_t {}, fixture_type_::metric_token_k),
                              status_t::success_k);
    }

    fu_topology_t topology = nullptr;
    failures += expect(fu_topology_new(&topology) == fu_success_k);
    {
        auto [executor, spawn_status] = executor_t::spawn(topology, 1, io_engine_config_t {});
        failures += expect_eq(spawn_status, status_t::success_k);
        nvme_t viewed {forkunion_allocator_t {}};
        failures += expect_eq(viewed.view(path, {}, fixture_type_::metric_token_k, executor), status_t::success_k);
        failures += expect_eq(viewed.size(), count);

        typename nvme_t::search_config_t search_config;
        search_config.expansion_factor = count;
        std::vector<key_t> keys(count);
        std::vector<distance_t> distances(count);
        std::vector<std::size_t> counts(count);
        auto const make_metric = [](basic_nvme_scratch_t const& allocator, std::size_t staging) noexcept {
            return fixture_type_::make_scratch_metric(allocator, staging);
        };
        auto const query_of = [&](std::size_t entry, basic_nvme_scratch_t const& allocator) noexcept {
            return fixture.query_of(entry, allocator);
        };
        auto const [found, search_status] = viewed.search(count, make_metric, query_of, std::span<key_t> {keys},
                                                          std::span<distance_t> {distances},
                                                          std::span<std::size_t> {counts}, search_config, executor);
        failures += expect_eq(search_status, status_t::success_k);
        failures += expect_eq(found.count, count);
        error_stats_t self_errors;
        for (std::size_t query = 0; query != count; ++query) {
            failures += expect_eq(counts[query], 1u);
            self_errors.add(static_cast<f64_t>(distances[query]), 0.0);
        }
        failures += expect_errors("nvme self distances", basic_comparison_family<distance_t>(),
                                  comparison_verdict_t::assert_k, self_errors, 0);

        mismatched_t mismatched {forkunion_allocator_t {}};
        failures += expect_eq(mismatched.view(path, {}, fixture_type_::metric_token_k, executor),
                              status_t::config_mismatch_k);
        failures += expect_eq(mismatched.size(), 0u);
    }
    fu_topology_delete(topology);
    std::remove(path.c_str());
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** Composes one engine's NVMe cell over @p fixture, skipped where the host refuses the engine; the
 *  widened view runs where the fixture names a RAM index of 64-bit node IDs. */
template <typename engine_type_, typename fixture_type_>
std::size_t test_basic_nvme_configuration(fixture_type_ const& fixture, std::string const& name, u64_t seed) {
    using executor_t = typename fixture_type_::template nvme_t<engine_type_>::executor_t;
    {
        fu_topology_t topology = nullptr;
        if (fu_topology_new(&topology) != fu_success_k) return report(name.c_str(), expect(false));
        status_t const spawned = executor_t::spawn(topology, 1, io_engine_config_t {}).status;
        fu_topology_delete(topology);
        if (failed(spawned)) return report(name.c_str(), "engine refused by this host");
    }
    std::size_t failures = 0;
    failures += test_basic_nvme_search<engine_type_>(fixture, seed);
    failures += test_basic_nvme_parts<engine_type_>(fixture, seed);
    failures += test_basic_nvme_key_scans<engine_type_>(fixture, seed);
    failures += test_basic_nvme_cursor<engine_type_>(fixture, seed);
    failures += test_basic_nvme_health<engine_type_>(fixture, seed);
    failures += test_basic_nvme_sample<engine_type_>(fixture, seed);
    failures += test_basic_nvme_telemetry<engine_type_>(fixture, seed);
    failures += test_basic_nvme_failed_read<engine_type_>(fixture, seed);
    failures += test_basic_nvme_relocated<engine_type_>(fixture, seed);
    if constexpr (requires { typename fixture_type_::wide_fixture_t; })
        failures += test_basic_nvme_widened_keys<engine_type_>(fixture, seed);
    return report(name.c_str(), failures);
}

/** Every engine's NVMe cell over @p fixture, each under the name @p modality and the engine. */
template <typename fixture_type_>
std::size_t test_basic_nvme_engines(fixture_type_ const& fixture, std::string_view modality, u64_t seed) {
    std::size_t failures = 0;
    failures += test_basic_nvme_configuration<blocking_files_engine<fu::standard_yield_t>>(
        fixture, fmt::format("{} nvme blocking", modality), seed);
#if USEARCH_WITH_LINUX_AIO
    failures += test_basic_nvme_configuration<linux_aio_files_engine<fu::standard_yield_t>>(
        fixture, fmt::format("{} nvme linux-aio", modality), seed);
#endif
#if USEARCH_WITH_LINUX_URING
    failures += test_basic_nvme_configuration<linux_uring_files_engine<fu::standard_yield_t>>(
        fixture, fmt::format("{} nvme linux-uring", modality), seed);
#endif
    return failures;
}

#pragma endregion Adapters

/** The whole shared battery over one fixture, in file order - units add their own
 *  extras after; only the SafeGraphs round trip and hostile battery compose per unit, as
 *  their docblocks say. */
template <typename fixture_type_>
std::size_t test_basic_scenarios(u64_t seed) {
    std::size_t failures = 0;
    failures += test_basic_guards<fixture_type_>(seed);
    failures += test_basic_handmade<fixture_type_>(seed);
    failures += test_basic_growth_and_recall<fixture_type_>(seed);
    failures += test_basic_selectors<fixture_type_>(seed);
    failures += test_basic_keys<fixture_type_>(seed);
    failures += test_basic_wiring<fixture_type_>(seed);
    failures += test_basic_unreserved_growth<fixture_type_>(seed);
    failures += test_basic_allocation_budget<fixture_type_>(seed);
    failures += test_basic_starved<fixture_type_>(seed);
    failures += test_basic_snapshot_roundtrip<fixture_type_>(seed);
    failures += test_basic_snapshot_hostile<fixture_type_>(seed);
    failures += test_basic_snapshot_regions<fixture_type_>(seed);
    failures += test_basic_plan_race<fixture_type_>(seed);
    failures += test_basic_plan_contention<fixture_type_>(seed);
    failures += test_basic_refine<fixture_type_>(seed);
    failures += test_basic_absorb<fixture_type_>(seed);
    failures += test_basic_split<fixture_type_>(seed);
    failures += test_basic_refine_releases<fixture_type_>(seed);
    failures += test_basic_refine_starved<fixture_type_>(seed);
    failures += test_basic_health<fixture_type_>(seed);
    failures += test_basic_certified<fixture_type_>(seed);
    failures += test_basic_torn_bracket<fixture_type_>(seed);
    failures += test_basic_metric_contract<fixture_type_>(seed);
    failures += test_basic_restarts<fixture_type_>(seed);
    failures += test_basic_concurrent_build<fixture_type_>(seed);
    failures += test_basic_concurrent_search<fixture_type_>(seed);
    failures += test_basic_planned_batch<fixture_type_>(seed);
    failures += test_basic_wiring_race<fixture_type_>(seed);
    failures += test_basic_growth_race<fixture_type_>(seed);
    failures += test_basic_cursor<fixture_type_>(seed);
    return failures;
}

} // namespace unum::usearch::test
