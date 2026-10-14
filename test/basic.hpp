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
 *  Allocators are tested by construction: one counting allocator flows through the whole
 *  index - records, neighbor rows, and the key multimap - and through the values the fixtures
 *  build, as @c make_values and @c make_handmade take it last, tallying live bytes into a
 *  ledger that must balance to zero when each scenario's scope closes. Parallel scenarios
 *  use ForkUnion's default flat pool.
 *
 *  Every scenario keys its streams through @ref stream_key under the run's seed and its own name, like
 *  @c "starved"; a scenario's further streams take distinct indices, and per-thread streams index by thread.
 */
#pragma once

#include <cstdint> // `std::uintptr_t` - per-process checkpoint paths
#include <cstdio>  // `std::snprintf`, `std::remove` - the checkpoint files

#include <algorithm>   // `std::sort`, `std::unique`
#include <array>       // `std::array` - handmade triples
#include <atomic>      // `std::atomic` - per-thread failure counters & metric gates
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
 *  refuse pushes after an armed budget to drive every saturation path. */
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
    std::size_t budget_left = std::numeric_limits<std::size_t>::max();

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
        if (!budget_left) return ++refusals, metric_staging_t::refused_k;
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
    void disarm() noexcept { budget_left = std::numeric_limits<std::size_t>::max(); }
};

/** Wrapper whose first push parks until released - the deterministic way to hold
 *  one insert mid-flight, so the settled-past guards can be observed refusing. */
template <typename inner_metric_type_, typename distance_type_>
struct basic_gated_metric {
    inner_metric_type_ inner;
    std::atomic<bool>* entered = nullptr;
    std::atomic<bool>* released = nullptr;

    template <typename query_type_>
    void reset(query_type_ const& query) noexcept {
        inner.reset(query);
    }
    template <typename staged_type_>
    metric_staging_t push(staged_type_ const& staged) noexcept {
        entered->store(true, std::memory_order_release);
        while (!released->load(std::memory_order_acquire)) std::this_thread::yield();
        return inner.push(staged);
    }
    void drain(std::span<distance_type_> distances, distance_type_ radius) noexcept { inner.drain(distances, radius); }
};

#pragma endregion Instrumented Metrics

#pragma region Shared Helpers

/** Distances must ascend in every result pool; answers the failures found. */
template <typename context_type_>
std::size_t basic_expect_ascending(context_type_ const& context) noexcept {
    std::size_t failures = 0;
    for (std::size_t position = 1; position < context.results.size(); ++position)
        failures += expect(!(context.results[position].distance < context.results[position - 1].distance));
    return failures;
}

/** Compares two distances by the metric's own type: an integral distance is exact and
 *  compares bit-for-bit, a floating one at the fast-math tolerance. */
template <typename fixture_type_, typename distance_type_>
std::size_t basic_expect_distance(distance_type_ actual, distance_type_ wanted) noexcept {
    if constexpr (std::is_integral_v<distance_type_>) return expect_eq(actual, wanted);
    else return expect_near(static_cast<f64_t>(actual), static_cast<f64_t>(wanted));
}

/** Inserts @p count values under sequential fixture keys in batches, collecting
 *  assigned IDs; the workhorse the single-threaded scenarios build graphs with. */
template <typename fixture_type_, typename index_type_, typename metric_type_, typename context_type_,
          typename selector_type_>
std::size_t basic_build(index_type_& index, metric_type_& metric, std::vector<typename index_type_::value_t>& values,
                        std::vector<typename index_type_::node_id_t>& assigned_ids, context_type_& context,
                        selector_type_ selector) {
    using key_t = typename index_type_::key_t;
    constexpr std::size_t batch_size_k = 32;
    std::size_t const count = values.size();
    std::size_t failures = 0;
    assigned_ids.resize(count);
    std::vector<key_t> keys(count);
    for (std::size_t ordinal = 0; ordinal != count; ++ordinal) keys[ordinal] = fixture_type_::make_key(ordinal);
    for (std::size_t begin = 0; begin < count; begin += batch_size_k) {
        std::size_t const batch = std::min(batch_size_k, count - begin);
        auto const [result, status] = index.insert({values.data() + begin, batch}, {keys.data() + begin, batch}, metric,
                                                   {assigned_ids.data() + begin, batch}, context, selector);
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
    for (std::size_t position = 0; position != context.results.size(); ++position) {
        auto const scored = context.results[position];
        failures += expect(static_cast<std::size_t>(scored.node_id) < stored_count);
        failures += basic_expect_distance<fixture_type_>(
            scored.distance, fixture_type_::oracle(query, index.published_value_of(scored.node_id)));
    }
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

/** The one-thread executor the SafeGraphs glue fans out over: every region runs on
 *  the calling thread. */
struct serial_executor_t {
    std::size_t size() const noexcept { return 1; }
    template <typename callback_type_>
    status_t for_tasks(std::size_t tasks, callback_type_&& callback) noexcept {
        for (std::size_t task = 0; task != tasks; ++task) callback(task, 0, 0);
        return status_t::success_k;
    }
};

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
 *  effects, inserts cover their own rows, and @c reserve covers ahead at any time with no
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
        auto const [empty_result, empty_status] = index.search(values[0], metric, context);
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

        // With no reserve, an insert covers the rows it claims.
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
        failures += expect_eq(
            index.insert({values.data() + batch_k, 1}, {keys.data() + batch_k, 1}, metric, {}, context).status,
            status_t::success_k);
        failures += expect_eq(index.size(), batch_k + 1);

        // The index's storage is provably on the ledger, not the plain heap.
        failures += expect(ledger.live_bytes.load() > 0);
        failures += expect(ledger.total_allocations.load() > 0);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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

        for (std::size_t row = 0; row != 3; ++row)
            for (std::size_t column = 0; column != 3; ++column)
                failures += basic_expect_distance<fixture_type_>(
                    fixture_type_::oracle(queries.values[row], index.published_value_of(assigned_ids[column])),
                    handmade.distances[row][column]);

        for (std::size_t row = 0; row != 3; ++row) {
            auto const [result, status] = index.search(queries.values[row], metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += expect(result.count >= 3);
            failures += basic_expect_distance<fixture_type_>(context.results[0].distance,
                                                             typename index_t::distance_t {});
            auto runner_up = handmade.distances[row][(row + 1) % 3];
            if (handmade.distances[row][(row + 2) % 3] < runner_up) runner_up = handmade.distances[row][(row + 2) % 3];
            failures += basic_expect_distance<fixture_type_>(context.results[1].distance, runner_up);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
            auto const [result, status] = index.search(query, metric, context);
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

        // Explicit tuning knobs: a starved fan-out, an explicit round width, and a pool
        // covering the whole graph - whose best is the true best by construction.
        for (auto const config : {typename index_t::search_config_t {0, 1},  //
                                  typename index_t::search_config_t {32, 8}, //
                                  typename index_t::search_config_t {basic_growth_nodes_k, 0}}) {
            auto const [result, status] = index.search(queries[0], metric, context, config);
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(context);
        }
        typename index_t::search_config_t const exhaustive_config {basic_growth_nodes_k, 0};
        for (std::size_t sample = 0; sample != 4; ++sample) {
            failures += expect_eq(index.search(queries[sample], metric, context, exhaustive_config).status,
                                  status_t::success_k);
            bool found_best = false;
            failures += basic_audit_search<fixture_type_>(index, queries[sample], context, found_best);
            failures += expect(found_best); // A whole-graph pool cannot miss.
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
                auto const [result, status] = index.search(query, metric, context);
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
        failures += expect_eq(ledger.live_bytes.load(), 0);
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
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

        // Over-wire one node far past the build budget - rows are unbounded.
        for (std::size_t target = 1; target != 13; ++target)
            failures += expect(succeeded(index.link(from, static_cast<node_id_t>(target)).status));
        failures += expect(index.degree_of(from).value >= 12);
        failures += expect_eq(index.search(values[0], metric, context).status, status_t::success_k);
        failures += expect_eq(index.health(context).status, status_t::success_k);

        // Variable degrees survive the snapshot as they are - each row's own prefix sums say its degree.
        auto const [columns, save_status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(save_status));
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect(bool(restored.reserve(nodes_k)));
        failures += expect_eq(restored.load(columns.view(), codec), status_t::success_k);
        failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** Growth with no reserve: inserts cross three node chunks of 1'024 rows, then twice as many per
 *  chunk, and the rows on both sides of each boundary keep their keys and values; then @c clear
 *  forgets every row and key but keeps the chunks, and a rebuild restarts from ID zero. */
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

        // A narrow pool keeps the many walks cheap; the chunks are what this scenario is about.
        typename index_t::insert_config_t const narrow_config {8, 0};
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

        // The insert moved the originals, so the rows compare against a fresh identical draw.
        auto expected_values = fixture_type_::make_values(nodes_k, stream_key(seed, "unreserved"),
                                                          counting_allocator_t {ledger});
        for (std::size_t const row : {std::size_t {0}, first_chunk_k - 1, first_chunk_k, first_chunk_k * 3 - 1,
                                      first_chunk_k * 3, nodes_k - 1}) {
            node_id_t const node_id = static_cast<node_id_t>(row);
            failures += expect(index.key_of(node_id).value == keys[row]);
            failures += expect(index.find(keys[row]).value == node_id);
            failures += expect(fixture_type_::value_equals(index.published_value_of(node_id), expected_values[row]));
        }
        // A whole-graph pool cannot miss, wherever the best row lies.
        auto const [found, found_status] = index.search(expected_values[nodes_k - 1], metric, context,
                                                        typename index_t::search_config_t {nodes_k, 0});
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Foundations

#pragma region Allocation Budgets

/** All-or-nothing under a starved ledger: a refused reserve covers nothing & retries, a
 *  refused insert leaves the index empty & retries, and a refused link-growth leaves the row
 *  untouched. */
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

        ledger.byte_budget.store(ledger.live_bytes.load() + 64);
        failures += expect_eq(index.reserve(basic_growth_nodes_k).status, status_t::bad_alloc_k);
        failures += expect_eq(index.capacity(), 0u);
        ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
        failures += expect_eq(index.reserve(basic_growth_nodes_k).status, status_t::success_k);

        // A budget swept up from the live bytes refuses an insert's allocations one by one: the key
        // seats', then the node chunk's, which a step far below its size refuses at least once.
        // Every refusal, the cover's included, leaves the index empty and its claim given back,
        // until the first success dispenses ID zero.
        std::unique_ptr<index_t> const swept_ptr = fixture_type_::make_index(ledger);
        index_t& swept = *swept_ptr;
        auto swept_values = fixture_type_::make_values(1, stream_key(seed, "budget", 1), counting_allocator_t {ledger});
        std::array<key_t, 1> const swept_keys {fixture_type_::make_key(0)};
        std::array<node_id_t, 1> swept_ids {node_id_t {1}};
        failures += expect_eq(swept.search(swept_values[0], metric, context).status, status_t::success_k); // Warm.
        constexpr std::ptrdiff_t sweep_step_k = 4096;
        std::ptrdiff_t const swept_base = ledger.live_bytes.load();
        std::size_t refusals = 0;
        status_t swept_status = status_t::bad_alloc_k;
        for (std::ptrdiff_t slack = 0; swept_status == status_t::bad_alloc_k && slack < (std::ptrdiff_t {1} << 30);
             slack += sweep_step_k) {
            ledger.byte_budget.store(swept_base + slack);
            swept_status =
                swept.insert({swept_values}, {swept_keys}, metric, {swept_ids}, context, keep_closest_t {1}).status;
            if (swept_status != status_t::bad_alloc_k) break;
            ++refusals;
            failures += expect_eq(swept.size(), 0u);
            failures += expect_eq(swept.capacity(), 0u);
        }
        ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
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

        // Amortized row growth leaves power-of-two slack, so links absorb it for free -
        // but the slack is finite, and a starved ledger must refuse the first real
        // growth without touching the row.
        std::size_t const degree_before = index.degree_of(0).value;
        ledger.byte_budget.store(ledger.live_bytes.load());
        std::size_t fresh_links = 0;
        node_id_t refused_target = 0;
        for (std::size_t target = 1; target != linked_nodes_k && !refused_target; ++target) {
            auto const [membership, status] = index.link(0, static_cast<node_id_t>(target));
            if (status == status_t::bad_alloc_k) refused_target = static_cast<node_id_t>(target);
            else if (succeeded(status) && membership == membership_t::absent_k) ++fresh_links;
        }
        failures += expect(refused_target != 0);
        failures += expect_eq(index.degree_of(0).value, degree_before + fresh_links);

        ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
        auto const [retry_membership, retry_status] = index.link(0, refused_target);
        failures += expect_eq(retry_status, status_t::success_k);
        failures += expect(retry_membership == membership_t::absent_k);
        failures += expect_eq(index.degree_of(0).value, degree_before + fresh_links + 1);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** The staging a warm @c {8, 1} walk leaves in a fresh context on a graph of @p nodes_count rows built like the starved
 *  scenario's - what its over-wired row must outgrow, decided by the growth policy and the fixture's rows rather than a
 *  constant. */
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
    failures += expect_eq(index.search(query[0], metric, small_context, {8, 1}).status, status_t::success_k);
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

        // Warm a small round shape, then over-wire node 0 past its staging capacity, so
        // a later frozen-budget search can only proceed by refusing the top-up. The backward
        // edges are what make the fat row reachable: a walk saturates on rows it expands, and
        // it can only expand what some other row points at.
        context_t small_context {stream_key(seed, "starved", 1), counting_allocator_t {ledger}};
        failures += expect_eq(index.search(query[0], metric, small_context, {8, 1}).status, status_t::success_k);
        for (std::size_t target = 1; target != nodes_count; ++target) {
            [[maybe_unused]] auto const outward = index.link(0, static_cast<node_id_t>(target));
            [[maybe_unused]] auto const inward = index.link(static_cast<node_id_t>(target), 0);
        }
        // The premise: the fat row outgrows the staging the warm walk left behind, whatever its growth rounds to.
        failures += expect(index.degree_of(0).value > small_context.staged_ids.capacity());

        // Cold scratch under a frozen budget: all-or-nothing refusals, nothing moves.
        context_t cold_context {stream_key(seed, "starved", 2), counting_allocator_t {ledger}};
        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect_eq(index.search(query[0], metric, cold_context).status, status_t::bad_alloc_k);
        failures += expect_eq(index.insert({extras.data(), 1}, {extra_keys.data(), 1}, metric, {}, cold_context).status,
                              status_t::bad_alloc_k);
        failures += expect_eq(index.size(), nodes_count);
        failures += expect_eq(index.health(cold_context).status, status_t::bad_alloc_k);

        // Warm scratch under the same budget: the claimed ID must still publish, only
        // thinner - the fresh row's growth is the one refused allocation.
        failures += expect_eq(
            index.insert({extras.data(), 1}, {extra_keys.data(), 1}, metric, {}, warm_context, keep_closest_t {4})
                .status,
            status_t::success_k);
        failures += expect_eq(index.size(), nodes_count + 1);
        failures += expect_eq(index.degree_of(static_cast<node_id_t>(nodes_count)).value, 0u);

        // The saturated walk: each query is a node's own stored value, and every row points at the
        // over-wired one, so at least the walk for node 0 must expand it, which cannot stage under
        // the frozen budget. The receipt is asserted across the batch and every answer stays ranked.
        std::size_t saturations = 0;
        for (std::size_t sample = 0; sample != nodes_count; ++sample) {
            auto const [result, status] = index.search(index.published_value_of(static_cast<node_id_t>(sample)), metric,
                                                       small_context, {8, 1});
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(small_context);
            saturations += result.telemetry.saturated_walks;
        }
        failures += expect(saturations > 0);

        // Lifting the budget makes every refusal retryable.
        ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
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
        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect_eq(restored.load(columns.view(), codec), status_t::bad_alloc_k);
        failures += expect_eq(restored.size(), 0u);
        ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
        failures += expect_eq(restored.load(columns.view(), codec), status_t::success_k);
        failures += expect_eq(restored.size(), index.size());
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
        // A column short by one element overflows; one long by an element disagrees about the rows.
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

        // A load needs no reserve: its plan covers the rows under its hold.
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect_eq(restored.load(storage.view(), codec), status_t::success_k);
        failures += expect(restored.capacity() >= nodes_k);
        failures += basic_expect_mirrored<fixture_type_>(restored, index, nodes_k);

        // Same seed, same graph, same code path: the answers must be identical.
        for (auto const& query : queries) {
            context_t original_context {stream_key(seed, "snapshot", 1), counting_allocator_t {ledger}};
            context_t restored_context {stream_key(seed, "snapshot", 1), counting_allocator_t {ledger}};
            auto original_metric = fixture_type_::make_metric();
            auto restored_metric = fixture_type_::make_metric();
            auto const [original_result, original_status] = index.search(query, original_metric, original_context);
            auto const [restored_result, restored_status] = restored.search(query, restored_metric, restored_context);
            failures += expect_eq(original_status, restored_status);
            failures += expect_eq(original_result.count, restored_result.count);
            for (std::size_t position = 0; position != original_result.count; ++position) {
                failures += expect_eq(original_context.results[position].node_id,
                                      restored_context.results[position].node_id);
                failures += basic_expect_distance<fixture_type_>(restored_context.results[position].distance,
                                                                 original_context.results[position].distance);
            }
        }

        // Loading into a non-empty index is refused; the restored index stays intact.
        failures += expect_eq(restored.load(storage.view(), codec), status_t::config_mismatch_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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

        // An unreserved target covers its own rows and loads the intact columns.
        std::unique_ptr<index_t> const unreserved_ptr = fixture_type_::make_index(ledger);
        index_t& unreserved = *unreserved_ptr;
        failures += expect_eq(unreserved.load(storage.view(), codec), status_t::success_k);
        failures += basic_expect_mirrored<fixture_type_>(unreserved, index, nodes_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** Region-scoped serialization: a save plan whose uneven regions run from a pool
 *  must emit bytes identical to the single-call form, a region-parallel load must
 *  rebuild a graph whose own re-save is byte-identical, and the misuse rows -
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

        // Uneven splits from a pool must land identically to the single call.
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
            std::atomic<std::size_t> thread_failures {0};
            // No lock-held check in these threads: the calling thread runs one thread itself
            // while legitimately holding the plan's unique side for the whole pool.
            pool.for_threads([&](std::size_t thread) noexcept {
                if (plan.save(regions[thread], codec) != status_t::success_k) thread_failures.fetch_add(1);
            });
            failures += expect_eq(thread_failures.load(), 0u);
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
            std::atomic<std::size_t> thread_failures {0};
            pool.for_threads([&](std::size_t thread) noexcept {
                if (plan.load(regions[thread], codec) != status_t::success_k) thread_failures.fetch_add(1);
            });
            failures += expect_eq(thread_failures.load(), 0u);
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Snapshots

#pragma region Plans and Holds

/** Three rendezvous threads around one held save plan: every try-unique verb answers
 *  @c contended_k for the hold's whole span, the concurrent arm & wait-free readers
 *  stay fully served, and an inserter parked at the shared gate lands only after the
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

        std::atomic<bool> plan_held {false}, plan_abandoned {false};
        std::atomic<bool> inserter_started {false}, prober_done {false};
        std::atomic<std::size_t> thread_failures {0};

        auto planner_thread = [&]() noexcept { // Waits on the prober only - never on the inserter.
            auto [plan, status] = index.plan_save(held.columns(), codec);
            if (failed(status)) {
                thread_failures.fetch_add(1);
                plan_abandoned.store(true, std::memory_order_release);
            }
            plan_held.store(true, std::memory_order_release);
            if (failed(status)) return;
            if (!spin_until([&]() noexcept { return prober_done.load(std::memory_order_acquire); }))
                thread_failures.fetch_add(1); // Finish regardless - the hold must die.
            if (plan.save({0, static_cast<node_id_t>(nodes_k)}, codec) != status_t::success_k)
                thread_failures.fetch_add(1);
            if (plan.finish() != status_t::success_k) thread_failures.fetch_add(1);
        };
        auto inserter_thread = [&]() noexcept { // Provably parked at the shared gate.
            if (!spin_until([&]() noexcept { return plan_held.load(std::memory_order_acquire); }))
                return void(thread_failures.fetch_add(1));
            inserter_started.store(true, std::memory_order_release);
            auto thread_metric = fixture_type_::make_metric();
            context_t thread_context {stream_key(seed, "plan_race", 1), counting_allocator_t {ledger}};
            auto const [result, status] = index.insert({one_more.data(), 1}, {&one_more_key, 1}, thread_metric, {},
                                                       thread_context, keep_diverse_t {basic_connectivity_k});
            if (status != status_t::success_k) thread_failures.fetch_add(1);
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
                thread_failures.fetch_add(1);
                return prober_done.store(true, std::memory_order_release);
            }
            context_t prober_context {stream_key(seed, "plan_race", 2), counting_allocator_t {ledger}};
            auto prober_metric = fixture_type_::make_metric();
            // The parked inserter cannot publish under the hold - the overlap proof.
            if (index.size() != nodes_k) thread_failures.fetch_add(1);
            // Every try-unique verb refuses with contended_k, nothing blocks.
            if (index.columns_extents(codec).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (index.health(prober_context).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (index.plan_save(prober.columns(), codec).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (index.plan_load(reference.view()).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (index.clear() != status_t::contended_k) thread_failures.fetch_add(1);
            if (index.template search<synchronization_t::immutable_k>(query[0], prober_metric, prober_context).status !=
                status_t::contended_k)
                thread_failures.fetch_add(1);
            if (index.template search<synchronization_t::exclusive_k>(query[0], prober_metric, prober_context).status !=
                status_t::contended_k)
                thread_failures.fetch_add(1);
            // The concurrent arm and the wait-free readers stay fully served meanwhile.
            auto const [result, status] = index.search(query[0], prober_metric, prober_context);
            if (status != status_t::success_k) thread_failures.fetch_add(1);
            thread_failures.fetch_add(scored_failures(prober_context.results.data(), prober_context.results.size(),
                                                      nodes_k, connectivity_32_k * 2));
            for (std::size_t node = 0; node != nodes_k; ++node) {
                node_id_t const node_id = static_cast<node_id_t>(node);
                if (!(index.key_of(node_id).value == fixture_type_::make_key(node))) thread_failures.fetch_add(1);
                if (!index.degree_of(node_id)) thread_failures.fetch_add(1);
                [[maybe_unused]] auto const& published_value = index.published_value_of(node_id);
            }
            cursor_t prober_cursor {index, counting_allocator_t {ledger}};
            if (failed(prober_cursor.seek(0))) thread_failures.fetch_add(1);
            std::size_t visited = 0;
            for (; prober_cursor; ++prober_cursor) ++visited;
            if (visited != nodes_k) thread_failures.fetch_add(1);
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
        failures += expect_eq(index.size(), nodes_k + 1); // The parked insert landed after the finish.
        // The graph provably could not move under the hold: the held columns mirror the
        // pre-race reference element for element.
        failures += expect(held == reference);
        // The refusal was the hold, not the graph: a settled certified search succeeds.
        context_t settled_context {stream_key(seed, "plan_race", 3), counting_allocator_t {ledger}};
        auto settled_metric = fixture_type_::make_metric();
        failures += expect_eq(
            index.template search<synchronization_t::immutable_k>(query[0], settled_metric, settled_context).status,
            status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
        std::atomic<bool> plan_held {false};
        std::atomic<std::size_t> refusals_confirmed {0}, thread_failures {0};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                auto [plan, status] = index.plan_save(thread_storages[0].columns(), codec);
                if (failed(status)) thread_failures.fetch_add(1);
                plan_held.store(true, std::memory_order_release);
                if (failed(status)) return;
                if (!spin_until([&]() noexcept {
                        return refusals_confirmed.load(std::memory_order_acquire) == planner_threads_k - 1;
                    }))
                    thread_failures.fetch_add(1); // Finish regardless - the hold must die.
                if (plan.save({0, static_cast<node_id_t>(nodes_k)}, codec) != status_t::success_k)
                    thread_failures.fetch_add(1);
                if (plan.finish() != status_t::success_k) thread_failures.fetch_add(1);
                return;
            }
            if (!spin_until([&]() noexcept { return plan_held.load(std::memory_order_acquire); }))
                thread_failures.fetch_add(1); // A broken rendezvous is a failure, never a shrug.
            if (index.plan_save(thread_storages[thread].columns(), codec).status != status_t::contended_k)
                thread_failures.fetch_add(1);
            refusals_confirmed.fetch_add(1); // Unconditional, so thread 0's wait terminates.
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
        std::atomic<bool> entered {false}, released {false};
        // Phase A already proved the counter zero, so Phase C keeps accumulating into it.
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                  &entered, &released};
                context_t parked_context {stream_key(seed, "plan_contention", 1), counting_allocator_t {ledger}};
                auto const [result, status] = target.insert({pair_values.data(), 2}, {pair_keys.data(), 2}, gated, {},
                                                            parked_context, keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                assert_no_locks_held();
                return;
            }
            if (thread != 1) return;
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                thread_failures.fetch_add(1);
                released.store(true, std::memory_order_release); // The gate opens even on failure.
                return;
            }
            if (target.plan_load(reference.view()).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (target.plan_save(prober.columns(), codec).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (target.columns_extents(codec).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (target.clear() != status_t::contended_k) thread_failures.fetch_add(1);
            released.store(true, std::memory_order_release);
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(target.size(), 2u);
        // Settled & non-empty: the refusal decays into the documented misuse status.
        failures += expect_eq(target.plan_load(reference.view()).status, status_t::config_mismatch_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Plans and Holds

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
        std::atomic<bool> entered {false}, released {false};
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "health", 3), counting_allocator_t {ledger});
        typename index_t::key_t const one_more_key = fixture_type_::make_key(nodes_k);
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> thread_failures {0};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                  &entered, &released};
                context_t parked_context {stream_key(seed, "health", 1), counting_allocator_t {ledger}};
                auto const [result, status] = index.insert({one_more.data(), 1}, {&one_more_key, 1}, gated, {},
                                                           parked_context, keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                return;
            }
            if (thread != 1) return;
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                // The gate must open even on a failed rendezvous, or the pool deadlocks.
                thread_failures.fetch_add(1);
                released.store(true, std::memory_order_release);
                return;
            }
            context_t checker_context {stream_key(seed, "health", 2), counting_allocator_t {ledger}};
            if (index.health(checker_context).status != status_t::contended_k) thread_failures.fetch_add(1);
            if (index.columns_extents(codec).status != status_t::contended_k) thread_failures.fetch_add(1);
            released.store(true, std::memory_order_release);
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_k + 1);
        failures += expect_eq(index.health(context).status, status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** The certified arms at runtime: an exclusive build equals a concurrent
 *  single-threaded one, certified searches answer identically to the concurrent arm
 *  on a settled graph, and a parked insert makes both certified arms answer
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

        // Three search arms over the settled graph must agree bit-for-bit.
        auto queries = fixture_type_::make_values(4, stream_key(seed, "certified", 4), counting_allocator_t {ledger});
        for (auto const& query : queries) {
            context_t concurrent_arm {stream_key(seed, "certified", 1), counting_allocator_t {ledger}};
            context_t immutable_arm {stream_key(seed, "certified", 1), counting_allocator_t {ledger}};
            context_t exclusive_arm {stream_key(seed, "certified", 1), counting_allocator_t {ledger}};
            auto arm_metric = fixture_type_::make_metric();
            auto const [concurrent_result, concurrent_status] = exclusive_index.search(query, arm_metric,
                                                                                       concurrent_arm);
            auto const [immutable_result, immutable_status] =
                exclusive_index.template search<synchronization_t::immutable_k>(query, arm_metric, immutable_arm);
            auto const [exclusive_result, exclusive_status] =
                exclusive_index.template search<synchronization_t::exclusive_k>(query, arm_metric, exclusive_arm);
            failures += expect_eq(immutable_status, concurrent_status);
            failures += expect_eq(exclusive_status, concurrent_status);
            failures += expect_eq(immutable_result.count, concurrent_result.count);
            failures += expect_eq(exclusive_result.count, concurrent_result.count);
            for (std::size_t position = 0; position != concurrent_result.count; ++position) {
                failures += expect_eq(immutable_arm.results[position].node_id,
                                      concurrent_arm.results[position].node_id);
                failures += expect_eq(exclusive_arm.results[position].node_id,
                                      concurrent_arm.results[position].node_id);
            }
        }

        // A parked insert violates any certification; the guard must say so.
        std::atomic<bool> entered {false}, released {false};
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "certified", 5), counting_allocator_t {ledger});
        key_t const one_more_key = fixture_type_::make_key(nodes_k);
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> thread_failures {0};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                  &entered, &released};
                context_t parked_context {stream_key(seed, "certified", 2), counting_allocator_t {ledger}};
                auto const [result, status] = exclusive_index.insert({one_more.data(), 1}, {&one_more_key, 1}, gated,
                                                                     {}, parked_context,
                                                                     keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                return;
            }
            if (thread != 1) return;
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                thread_failures.fetch_add(1);
                released.store(true, std::memory_order_release); // The gate opens even on failure.
                return;
            }
            context_t checker_context {stream_key(seed, "certified", 3), counting_allocator_t {ledger}};
            auto checker_metric = fixture_type_::make_metric();
            if (exclusive_index
                    .template search<synchronization_t::immutable_k>(one_more[0], checker_metric, checker_context)
                    .status != status_t::contended_k)
                thread_failures.fetch_add(1);
            if (exclusive_index
                    .template search<synchronization_t::exclusive_k>(one_more[0], checker_metric, checker_context)
                    .status != status_t::contended_k)
                thread_failures.fetch_add(1);
            released.store(true, std::memory_order_release);
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(exclusive_index.size(), nodes_k + 1);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
            index.template search<synchronization_t::immutable_k>(query[0], control_metric, control_context).status,
            status_t::success_k);

        if constexpr (!thread_sanitizer_k) {
            std::atomic<bool> entered {false}, released {false};
            std::atomic<std::size_t> thread_failures {0};
            std::atomic<u32_t> parked_status {~u32_t {0}};
            fu::flat_pool_t pool;
            if (fu::failed(pool.spawn(2))) return failures + expect(false);
            pool.for_threads([&](std::size_t thread) noexcept {
                if (thread == 0) { // The certified search is the parked party - no locks held.
                    basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture_type_::make_metric(),
                                                                                      &entered, &released};
                    context_t parked_context {stream_key(seed, "torn_bracket", 2), counting_allocator_t {ledger}};
                    auto const [result, status] = index.template search<synchronization_t::immutable_k>(query[0], gated,
                                                                                                        parked_context);
                    parked_status.store(static_cast<u32_t>(status), std::memory_order_relaxed);
                    return;
                }
                if (thread != 1) return; // The mutator: runs strictly inside the parked walk.
                if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                    thread_failures.fetch_add(1);
                    released.store(true, std::memory_order_release); // The gate opens even on failure.
                    return;
                }
                auto thread_metric = fixture_type_::make_metric();
                context_t thread_context {stream_key(seed, "torn_bracket", 3), counting_allocator_t {ledger}};
                auto const [result, status] = index.insert({one_more.data(), 1}, {&one_more_key, 1}, thread_metric, {},
                                                           thread_context, keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                released.store(true, std::memory_order_release); // Every path - the search must resume.
                assert_no_locks_held();
            });
            failures += expect_eq(thread_failures.load(), 0u);
            // The entry probe passed - `entered` fired after it - so contended_k here is the
            // exit probe's verdict: the generation moved & the watermark advanced.
            failures += expect_eq(static_cast<status_t>(parked_status.load()), status_t::contended_k);
            failures += expect_eq(index.size(), nodes_k + 1);
        }

        // The refusal was transient: a settled re-run certifies again.
        context_t settled_context {stream_key(seed, "torn_bracket", 4), counting_allocator_t {ledger}};
        auto settled_metric = fixture_type_::make_metric();
        failures += expect_eq(
            index.template search<synchronization_t::immutable_k>(query[0], settled_metric, settled_context).status,
            status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
        std::size_t rounds_limit = std::numeric_limits<std::size_t>::max();
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

        failures += expect(bool(index.reserve(nodes_k + 1)));
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
            auto const [result, status] = index.search(query, metric, fold, context);
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(context);
            failures += expect_eq(fold.rounds, metric.drains - drains_before);
            failures += expect_eq(fold.pairs, result.telemetry.computed_distances);
        }
        failures += expect_eq(metric.violations, 0u);

        // Sated by its landmarks, a walk ends before they reach the pool, scoring less than a whole-graph walk.
        counting_fold_t sated {1};
        auto const [sated_result, sated_status] = index.search(queries[0], metric, sated, context);
        failures += expect_eq(sated_status, status_t::success_k);
        failures += expect_eq(sated.rounds, 1u);
        failures += expect_eq(sated_result.count, 0u);
        auto const [unfolded_result, unfolded_status] = index.search(queries[0], metric, context);
        failures += expect_eq(unfolded_status, status_t::success_k);
        failures += expect(sated_result.telemetry.computed_distances < unfolded_result.telemetry.computed_distances);

        // Armed refusal: the walk must survive every saturation path & stay ranked.
        for (std::size_t budget : {basic_connectivity_k, std::size_t {1}, std::size_t {0}}) {
            metric.arm(budget);
            auto const [result, status] = index.search(queries[0], metric, context);
            failures += expect_eq(status, status_t::success_k);
            failures += basic_expect_ascending(context);
            metric.disarm();
        }
        failures += expect_eq(metric.violations, 0u);

        // An armed insert wires a thinner node but still publishes it.
        metric.arm(2);
        auto one_more = fixture_type_::make_values(1, stream_key(seed, "contract", 2), counting_allocator_t {ledger});
        typename index_t::key_t const one_more_key = fixture_type_::make_key(nodes_k);
        failures += expect_eq(index
                                  .insert({one_more.data(), 1}, {&one_more_key, 1}, metric, {}, context,
                                          keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        metric.disarm();
        failures += expect_eq(index.size(), nodes_k + 1);
        failures += expect_eq(metric.violations, 0u);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** Frontier starvation drives re-seeding: islands - nodes manually stripped of
 *  every edge in both directions - are reachable only through a restart's landmark
 *  draws, and a whole-graph pool must still find every node. */
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
            auto const [result, status] = index.search(query, metric, attempt_context, {nodes_k, 0});
            failures += expect_eq(status, status_t::success_k);
            failures += expect_eq(result.count, nodes_k); // The islands included - full coverage.
            any_restarts = any_restarts || attempt_context.telemetry.restarts > 0;
        }
        failures += expect(any_restarts);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Contracts

#pragma region Races

/** Concurrent build on distinct-seeded contexts: dense unique IDs, every thread's
 *  keys landing on its own values, and the payloads surviving the move intact. */
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
        std::atomic<std::size_t> thread_failures {0};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread >= basic_threads_k) return;
            auto const [result, status] = index.insert(
                {threads.values[thread]}, {threads.keys[thread]}, threads.metrics[thread],
                {assigned_ids.data() + thread * basic_thread_nodes_k, basic_thread_nodes_k}, threads.contexts[thread],
                keep_diverse_t {basic_connectivity_k});
            if (status != status_t::success_k) thread_failures.fetch_add(1);
            assert_no_locks_held();
        });

        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
        // IDs are dense & unique across threads - the watermark's dispensed-order turn.
        std::vector<node_id_t> sorted_ids = assigned_ids;
        failures += expect_contiguous_ids(std::span {sorted_ids}, 0);

        // Every thread's keys & values landed on the IDs that thread was assigned; the values
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
        std::atomic<std::size_t> thread_failures {0};
        std::atomic<std::size_t> writer_threads_remaining {writer_threads_k};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread < writer_threads_k) {
                auto const [result, status] = index.insert({threads.values[thread]}, {threads.keys[thread]},
                                                           threads.metrics[thread], {}, threads.contexts[thread],
                                                           keep_diverse_t {basic_connectivity_k});
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                writer_threads_remaining.fetch_sub(1, std::memory_order_release);
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
                auto const [search_result, search_status] = index.search(query, reader_metric, reader_context);
                if (search_status != status_t::success_k) thread_failures.fetch_add(1);
                thread_failures.fetch_add(scored_failures(reader_context.results.data(), reader_context.results.size(),
                                                          index.size(), results_bound));
                // Any ID is a legal input to the checked readers, published or not.
                node_id_t const probe = static_cast<node_id_t>(probe_distribution(reader_random));
                std::size_t const published_before = index.size();
                auto const [probed_key, key_status] = index.key_of(probe);
                auto const probed_degree = index.degree_of(probe);
                if (static_cast<std::size_t>(probe) < published_before)
                    if (failed(key_status) || !probed_degree) thread_failures.fetch_add(1);
                if (failed(key_status) && key_status != status_t::malformed_k) thread_failures.fetch_add(1);
                ++searches;
            }
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** One planned batch over @c basic_threads_k threads, each landing a disjoint slice through
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

        std::atomic<std::size_t> thread_failures {0};
        std::atomic<bool> finished {false};
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread < basic_threads_k) {
                std::size_t const first = thread * basic_thread_nodes_k;
                plan.insert(first, std::span(batch_values).subspan(first, basic_thread_nodes_k),
                            std::span<key_t const>(batch_keys).subspan(first, basic_thread_nodes_k), metrics[thread],
                            contexts[thread], keep_diverse_t {basic_connectivity_k});
                if (thread != 0) return assert_no_locks_held();
                // The calling thread runs thread zero and holds the plan, so it finishes, retrying
                // the refusal until every slice landed.
                status_t finished_status = plan.finish();
                for (std::size_t attempt = 0; finished_status == status_t::config_mismatch_k && attempt != spin_limit_k;
                     ++attempt) {
                    std::this_thread::yield();
                    finished_status = plan.finish();
                }
                if (finished_status != status_t::success_k) thread_failures.fetch_add(1);
                finished.store(true, std::memory_order_release);
                return assert_no_locks_held();
            }
            if (thread != basic_threads_k) return;
            auto reader_metric = fixture_type_::make_metric();
            heap_context_t reader_context {stream_key(seed, "planned", 3 + basic_threads_k)};
            std::size_t searches = 0;
            while (!finished.load(std::memory_order_acquire) || searches < basic_reader_minimal_searches_k) {
                std::size_t const published = index.size();
                if (published != prefill_k && published != prefill_k + batch_k) thread_failures.fetch_add(1);
                auto const [result, status] = index.search(queries[searches % queries.size()], reader_metric,
                                                           reader_context);
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                thread_failures.fetch_add(scored_failures(reader_context.results.data(), reader_context.results.size(),
                                                          index.size(), connectivity_32_k * 2));
                // Keys land after the turn, so a findable batch key implies the whole batch is
                // published.
                if (index.contains(batch_keys.back()) && index.size() != prefill_k + batch_k)
                    thread_failures.fetch_add(1);
                ++searches;
            }
            assert_no_locks_held();
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), prefill_k + batch_k);
        failures += expect_eq(plan.finish(), status_t::uninitialized_k); // Spent: no second turn.

        // Every slice's keys and values landed on the plan's IDs, in position order.
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
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

        std::atomic<std::size_t> thread_failures {0};
        std::atomic<std::size_t> contended_saves {0};
        std::atomic<std::size_t> settled_saves {0};
        std::atomic<u32_t> unexpected_status {~u32_t {0}};
        std::atomic<u32_t> unexpected_thread {~u32_t {0}};
        auto record_unexpected = [&](std::size_t thread, status_t status) noexcept {
            thread_failures.fetch_add(1);
            unexpected_status.store(static_cast<u32_t>(status), std::memory_order_relaxed);
            unexpected_thread.store(static_cast<u32_t>(thread), std::memory_order_relaxed);
        };
        std::atomic<bool> wiring_done {false}, inserting_done {false};

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
                if (saved == status_t::contended_k || saved == status_t::overflow_k) contended_saves.fetch_add(1);
                else if (saved == status_t::success_k) settled_saves.fetch_add(1);
                else return record_unexpected(0, saved);
                std::this_thread::yield();
            }
            // The settle phase: with every racer done, a save must soon succeed.
            for (std::size_t attempt = 0; attempt != attempt_limit_k; ++attempt) {
                status_t const saved = attempt_save();
                if (saved == status_t::success_k) return void(settled_saves.fetch_add(1));
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
                auto const [result, status] = index.search(queries[searches % queries.size()], thread_metric,
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

        // The settled columns must hold duplicate-free rows and reload - the B2 audit.
        auto const [storage, status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(status));
        basic_columns_storage<index_t> const& columns = storage;
        for (std::size_t node = 0; node != columns.keys.size(); ++node) {
            std::vector<node_id_t> row(columns.neighbor_ids.begin() + columns.neighbor_offsets[node],
                                       columns.neighbor_ids.begin() + columns.neighbor_offsets[node + 1]);
            std::sort(row.begin(), row.end());
            failures += expect(std::unique(row.begin(), row.end()) == row.end());
        }
        std::unique_ptr<index_t> const restored_ptr = fixture_type_::make_index(ledger);
        index_t& restored = *restored_ptr;
        failures += expect(bool(restored.reserve(nodes_k + inserts_k)));
        failures += expect_eq(restored.load(columns.view(), codec), status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** Concurrent growth: threads submit batches past a one-chunk reservation, so claims ending past
 *  the first chunk race to cover the next, and every batch lands with dense IDs, no lost rows, and
 *  every key on its own row across the chunk boundary. */
template <typename fixture_type_>
std::size_t test_basic_growth_race(u64_t seed) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    constexpr std::size_t threads_k = 4;
    constexpr std::size_t batch_k = 32;
    constexpr std::size_t batches_per_thread_k = 9;
    constexpr std::size_t per_thread_k = batch_k * batches_per_thread_k; // 1'152 rows outgrow a 1'024-row chunk
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
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

        // Edge-following lands on the staged neighbor, or refuses past the snapshot.
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
        std::atomic<std::size_t> thread_failures {0};
        std::atomic<bool> writer_done {false};
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
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                for (std::size_t round = 0; round != 512; ++round) {
                    [[maybe_unused]] auto const linked = index.link(1, 2 + round % 8);
                    [[maybe_unused]] status_t const unlinked = index.unlink(1, 2 + round % 8);
                }
                writer_done.store(true, std::memory_order_release);
                return;
            }
            if (thread != 1) return;
            cursor_t racing_cursor {index, counting_allocator_t {ledger}};
            std::size_t sweeps = 0;
            do {
                if (sweeps++ == attempt_limit_k) return void(thread_failures.fetch_add(1)); // A stuck writer.
                racing_cursor.refresh();
                node_id_t const snapshot = racing_cursor.visible_count();
                if (failed(racing_cursor.seek(0))) {
                    thread_failures.fetch_add(1);
                    return;
                }
                for (; racing_cursor; ++racing_cursor)
                    if (racing_cursor.id() >= snapshot) thread_failures.fetch_add(1);
            } while (!writer_done.load(std::memory_order_acquire));
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

        // The certified arm agrees with the concurrent one on a settled graph.
        ram_cursor<index_t, synchronization_t::immutable_k, counting_allocator_t> settled_cursor {
            index, counting_allocator_t {ledger}};
        failures += expect_eq(settled_cursor.seek(0), status_t::success_k);
        std::size_t settled_visited = 0;
        for (; settled_cursor; ++settled_cursor) ++settled_visited;
        failures += expect_eq(settled_visited, index.size());
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
 *  meshes both carry the @c uuid_t key and pass "uuid". */
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
                // anything else carries the prefix sums that address its rows.
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

        // Regions from a pool land byte-identically to the serial pass.
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
            manifest_path, sizeof(manifest_path), "/tmp/usearch-basic-%08x.safegraphs.json",
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** Hostile monoliths: every header mutation of a good monolith is refused with its
 *  documented status before a payload byte lands, a planted out-of-range ID is the plan's
 *  refusal, a short target overflows, an illegal config is refused, and the same index
 *  restores from the intact monolith after every refusal. */
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
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
