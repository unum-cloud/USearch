/**
 *  @brief The use-case-agnostic test scenario library shared by every basic-index unit -
 *      sentences, bioinformatics sequences, and 3D meshes alike.
 *
 *  @author Ash Vardanian
 *  @file test/basic.hpp
 *  @date August 26, 2026
 *
 *  Every scenario is a template over a unit fixture supplying the index instantiation, a
 *  metric factory, a value generator, a value codec, and - unlike the dense tier, where
 *  the metric is library code - an @b exact brute-force oracle, so returned ids and
 *  distances are asserted against ground truth rather than self-recall. Fixtures name
 *  their concrete index instantiations; this header names no index and assumes no
 *  constructor shape beyond a ledgered allocator.
 *
 *  Allocators are tested by construction: one counting allocator flows through the whole
 *  index - records, neighbor rows, and the key multimap - tallying live bytes into a
 *  ledger that must balance to zero when each scenario's scope closes. Parallel scenarios
 *  use ForkUnion's default flat pool.
 */
#pragma once

#include <algorithm> // `std::sort`, `std::unique`
#include <array>     // `std::array` - handmade triples
#include <atomic>    // `std::atomic` - lane failure counters & metric gates
#include <bit>       // `std::bit_cast` - header & word surgery on snapshots
#include <cstdio>    // `std::tmpfile` - the SafeGraphs adapter's monolith
#include <thread>    // `std::this_thread::yield` - the gated metric's spin
#include <vector>    // `std::vector`

#include <fmt/format.h>
#include <forkunion.hpp>

#include "test.hpp"
#include "usearch/basic/ram_index.hpp"
#include "usearch/utilities/serialization.hpp"

namespace unum {
namespace usearch {

#pragma region Constants and Seeds

/** @brief Nodes in the single-threaded growth & oracle-recall scenarios. */
constexpr std::size_t basic_growth_nodes_k = 256;
/** @brief Queries sampled against the exact oracle. */
constexpr std::size_t basic_recall_samples_k = 24;
/** @brief Share of sampled queries whose best distance must match the oracle's. */
constexpr std::size_t basic_minimal_recall_percent_k = 90;
/** @brief The floor for @c basic_keep_closest_t builds: closest-only wiring promises
 *      navigability, never the diverse selector's cross-cluster recall. */
constexpr std::size_t basic_relaxed_recall_percent_k = 50;
/** @brief Lanes in the concurrent scenarios. */
constexpr std::size_t basic_lanes_k = 4;
/** @brief Nodes inserted by each lane of the concurrent scenarios. */
constexpr std::size_t basic_lane_nodes_k = 96;
/** @brief The selectors' neighbors-per-node budget across the scenarios. */
constexpr std::size_t basic_connectivity_k = 8;
/** @brief The reader lane performs at least this many searches, even if writers finish first. */
constexpr std::size_t basic_reader_minimal_searches_k = 16;

/** @brief Data seeds, one per scenario family, so no two scenarios share a draw. */
constexpr u64_t basic_seed_guards_k = 0xB001100Aull;
constexpr u64_t basic_seed_growth_k = 0x20260826ull;
constexpr u64_t basic_seed_selectors_k = 0x5E1EC709ull;
constexpr u64_t basic_seed_keys_k = 0x0E750000ull;
constexpr u64_t basic_seed_wiring_k = 0x11EA711Full;
constexpr u64_t basic_seed_budget_k = 0xBADA110Cull;
constexpr u64_t basic_seed_snapshot_k = 0x5AFE5A7Eull;
constexpr u64_t basic_seed_regions_k = 0x4E610135ull;
constexpr u64_t basic_seed_hostile_k = 0xDEADBEEFull;
constexpr u64_t basic_seed_health_k = 0x6EA17700ull;
constexpr u64_t basic_seed_contract_k = 0xC0478AC7ull;
constexpr u64_t basic_seed_restarts_k = 0x0135A27Aull;
constexpr u64_t basic_seed_certified_k = 0xCE271F1Eull;
constexpr u64_t basic_seed_starved_k = 0x57A24EDull;
constexpr u64_t basic_seed_wiring_race_k = 0x11A6CACEull;
constexpr u64_t basic_seed_overflow_race_k = 0x0F10CACEull;
/** @brief Context seeds per concurrent lane: identical seeds would draw identical
 *      landmarks and silently stop testing decorrelation, so lanes offset from these. */
constexpr u64_t basic_seed_build_lane_k = 0x0C000000ull;
constexpr u64_t basic_seed_search_lane_k = 0x05EA8C00ull;

#pragma endregion Constants and Seeds

#pragma region Instrumented Metrics

/**
 *  @brief Contract-auditing wrapper over any @c basic_metric: counts the verbs, verifies
 *      every drain is sized to that round's accepted pushes, flags abandoned pushed
 *      rounds, and can refuse pushes after an armed budget to drive every saturation path.
 */
template <typename inner_metric_type_, typename distance_type_>
struct basic_auditing_metric {
    inner_metric_type_ inner;

    std::size_t resets = 0;
    std::size_t pushes = 0;
    std::size_t drains = 0;
    std::size_t pushes_this_round = 0;
    std::size_t refusals = 0;
    /** @brief Contract breaches observed; every scenario asserts this stays zero. */
    std::size_t violations = 0;
    /** @brief Accepted pushes before refusal begins; @c SIZE_MAX means unarmed. */
    std::size_t budget_left = SIZE_MAX;

    template <typename query_type_>
    void reset(query_type_ const& query) noexcept {
        if (pushes_this_round) ++violations; // A pushed round was abandoned without a drain.
        inner.reset(query);
        ++resets;
        pushes_this_round = 0;
    }
    template <typename staged_type_>
    basic_metric_staging_t push(staged_type_ const& staged) noexcept {
        if (!resets) ++violations; // A round was opened implicitly.
        if (!budget_left) return ++refusals, basic_metric_staging_t::refused_k;
        if (inner.push(staged) == basic_metric_staging_t::refused_k)
            return ++refusals, basic_metric_staging_t::refused_k;
        --budget_left, ++pushes, ++pushes_this_round;
        return basic_metric_staging_t::staged_k;
    }
    void drain(std::span<distance_type_> distances) noexcept {
        if (!resets) ++violations;
        if (distances.size() != pushes_this_round) ++violations; // The drain-width contract.
        inner.drain(distances);
        ++drains;
        pushes_this_round = 0;
    }

    void arm(std::size_t budget) noexcept { budget_left = budget; }
    void disarm() noexcept { budget_left = SIZE_MAX; }
};

/** @brief Wrapper whose first push parks until released - the deterministic way to hold
 *      one insert mid-flight, so the settled-past guards can be observed refusing. */
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
    basic_metric_staging_t push(staged_type_ const& staged) noexcept {
        entered->store(true, std::memory_order_release);
        while (!released->load(std::memory_order_acquire)) std::this_thread::yield();
        return inner.push(staged);
    }
    void drain(std::span<distance_type_> distances) noexcept { inner.drain(distances); }
};

#pragma endregion Instrumented Metrics

#pragma region Shared Helpers

/** @brief Distances must ascend in every result pool; answers the failures found. */
template <typename context_type_>
std::size_t basic_expect_ascending(context_type_ const& context) noexcept {
    std::size_t failures = 0;
    for (std::size_t position = 1; position < context.results.size(); ++position)
        failures += expect(!(context.results[position].distance < context.results[position - 1].distance));
    return failures;
}

/** @brief Compares two distances by the fixture's exactness: integral metrics compare
 *      bit-for-bit, floating ones at the fast-math tolerance. */
template <typename fixture_type_, typename distance_type_>
std::size_t basic_expect_distance(distance_type_ actual, distance_type_ wanted) noexcept {
    if constexpr (fixture_type_::exact_distances_k) return expect_eq(actual, wanted);
    else return expect_near(static_cast<f64_t>(actual), static_cast<f64_t>(wanted));
}

/** @brief Inserts @p count values under sequential fixture keys in batches, collecting
 *      assigned ids; the workhorse the single-threaded scenarios build graphs with. */
template <typename fixture_type_, typename index_type_, typename metric_type_, typename selector_type_>
std::size_t basic_build(fixture_type_ const& fixture, index_type_& index, metric_type_& metric,
                        std::vector<typename index_type_::value_t>& values,
                        std::vector<typename index_type_::node_id_t>& assigned_ids,
                        typename index_type_::traversal_context_t& context, selector_type_ selector) {
    using key_t = typename index_type_::key_t;
    constexpr std::size_t batch_size_k = 32;
    std::size_t const count = values.size();
    std::size_t failures = 0;
    assigned_ids.resize(count);
    std::vector<key_t> keys(count);
    for (std::size_t ordinal = 0; ordinal != count; ++ordinal) keys[ordinal] = fixture.make_key(ordinal);
    for (std::size_t begin = 0; begin < count; begin += batch_size_k) {
        std::size_t const batch = std::min(batch_size_k, count - begin);
        auto const inserted = index.insert({values.data() + begin, batch}, {keys.data() + begin, batch}, metric,
                                           {assigned_ids.data() + begin, batch}, context, selector);
        failures += expect_eq(inserted.status, status_t::success_k);
    }
    return failures;
}

/** @brief The exact oracle's verdict on one search: every returned id must carry its true
 *      distance, and the best must match the graph-independent minimum. */
template <typename fixture_type_, typename index_type_, typename query_type_>
std::size_t basic_audit_search(fixture_type_ const& fixture, index_type_ const& index, query_type_ const& query,
                               typename index_type_::traversal_context_t const& context, bool& found_best) {
    using distance_t = typename index_type_::distance_t;
    std::size_t failures = 0;
    std::size_t const stored_count = index.size();
    for (std::size_t position = 0; position != context.results.size(); ++position) {
        auto const scored = context.results[position];
        failures += expect(static_cast<std::size_t>(scored.node_id) < stored_count);
        failures += basic_expect_distance<fixture_type_>(
            scored.distance, fixture.oracle(query, index.published_value_of(scored.node_id)));
    }
    distance_t truly_nearest = fixture.oracle(query, index.published_value_of(0));
    for (std::size_t node = 1; node != stored_count; ++node) {
        distance_t const candidate = fixture.oracle(
            query, index.published_value_of(static_cast<typename index_type_::node_id_t>(node)));
        if (candidate < truly_nearest) truly_nearest = candidate;
    }
    if constexpr (fixture_type_::exact_distances_k)
        found_best = context.results.size() && context.results[0].distance == truly_nearest;
    else
        found_best = context.results.size() &&
                     static_cast<f64_t>(context.results[0].distance) <= static_cast<f64_t>(truly_nearest) + 1e-6;
    return failures;
}

/** @brief The five section offsets of a basic snapshot, re-derived from its header
 *      independently of the implementation's own layout code - the shared oracle of the
 *      hostile battery, the duplicate-edge audit, and the re-alignment encoder. */
struct basic_payload_sections_t {
    std::size_t keys_offset = 0;
    std::size_t neighbor_sums_offset = 0;
    std::size_t neighbor_ids_offset = 0;
    std::size_t value_sums_offset = 0;
    std::size_t values_offset = 0;
    std::size_t total_bytes = 0;
};

template <typename key_type_, typename node_id_type_>
basic_payload_sections_t basic_payload_sections(basic_snapshot_header_t const& header) noexcept {
    std::size_t const alignment = header.alignment;
    std::size_t const node_count = static_cast<std::size_t>(header.node_count);
    auto aligned = [&](std::size_t offset) noexcept { return divide_round_up(offset, alignment) * alignment; };
    basic_payload_sections_t sections;
    sections.keys_offset = aligned(sizeof(basic_snapshot_header_t));
    sections.neighbor_sums_offset = aligned(sections.keys_offset + node_count * sizeof(key_type_));
    sections.neighbor_ids_offset = aligned(sections.neighbor_sums_offset + (node_count + 1) * sizeof(i64_t));
    sections.value_sums_offset = aligned(sections.neighbor_ids_offset +
                                         static_cast<std::size_t>(header.neighbors_count) * sizeof(node_id_type_));
    sections.values_offset = aligned(sections.value_sums_offset + (node_count + 1) * sizeof(i64_t));
    sections.total_bytes = sections.values_offset + static_cast<std::size_t>(header.value_bytes);
    return sections;
}

/** @brief Reads one trivially-copyable object out of a payload at @p offset. */
template <typename object_type_>
object_type_ basic_read_object(std::vector<std::byte> const& payload, std::size_t offset) noexcept {
    std::array<std::byte, sizeof(object_type_)> bytes;
    for (std::size_t position = 0; position != bytes.size(); ++position) bytes[position] = payload[offset + position];
    return std::bit_cast<object_type_>(bytes);
}

/** @brief Writes one trivially-copyable object into a payload at @p offset. */
template <typename object_type_>
void basic_write_object(std::vector<std::byte>& payload, std::size_t offset, object_type_ const& object) noexcept {
    auto const bytes = std::bit_cast<std::array<std::byte, sizeof(object_type_)>>(object);
    for (std::size_t position = 0; position != bytes.size(); ++position) payload[offset + position] = bytes[position];
}

/** @brief Re-encodes a saved payload at a different section alignment - same sections,
 *      tighter packing - so a load can prove it honors the header's alignment field. */
template <typename key_type_, typename node_id_type_>
std::vector<std::byte> basic_reencode_alignment(std::vector<std::byte> const& payload, u32_t new_alignment) {
    basic_snapshot_header_t header = basic_read_object<basic_snapshot_header_t>(payload, 0);
    auto const old_sections = basic_payload_sections<key_type_, node_id_type_>(header);
    header.alignment = new_alignment;
    auto const new_sections = basic_payload_sections<key_type_, node_id_type_>(header);
    std::vector<std::byte> reencoded(new_sections.total_bytes);
    basic_write_object(reencoded, 0, header);
    auto copy_section = [&](std::size_t old_offset, std::size_t new_offset, std::size_t bytes) noexcept {
        for (std::size_t position = 0; position != bytes; ++position)
            reencoded[new_offset + position] = payload[old_offset + position];
    };
    std::size_t const node_count = static_cast<std::size_t>(header.node_count);
    copy_section(old_sections.keys_offset, new_sections.keys_offset, node_count * sizeof(key_type_));
    copy_section(old_sections.neighbor_sums_offset, new_sections.neighbor_sums_offset,
                 (node_count + 1) * sizeof(i64_t));
    copy_section(old_sections.neighbor_ids_offset, new_sections.neighbor_ids_offset,
                 static_cast<std::size_t>(header.neighbors_count) * sizeof(node_id_type_));
    copy_section(old_sections.value_sums_offset, new_sections.value_sums_offset, (node_count + 1) * sizeof(i64_t));
    copy_section(old_sections.values_offset, new_sections.values_offset, static_cast<std::size_t>(header.value_bytes));
    return reencoded;
}

#pragma endregion Shared Helpers

#pragma region Scenarios

/** @brief Pre-reserve misuse, the frozen capacity, argument guards, and the all-or-nothing
 *      overflow rollback that must leave the claim counter exactly as it found it. */
template <typename fixture_type_>
std::size_t test_basic_guards(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t quota_k = 8;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_guards_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(quota_k + 1, basic_seed_guards_k);
        std::vector<key_t> keys(quota_k + 1);
        for (std::size_t ordinal = 0; ordinal != keys.size(); ++ordinal) keys[ordinal] = fixture.make_key(ordinal);

        // Everything is defined before the first reserve, and nothing is usable.
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index.capacity(), 0u);
        failures += expect_eq(index.search(values[0], metric, context).status, status_t::uninitialized_k);
        failures += expect_eq(index.insert({values.data(), 1}, {keys.data(), 1}, metric, {}, context).status,
                              status_t::uninitialized_k);
        failures += expect_eq(index.find(keys[0]).status, status_t::uninitialized_k);
        failures += expect(!index.contains(keys[0]));
        failures += expect_eq(index.count(keys[0]), 0u);
        failures += expect_eq(index.key_of(0).status, status_t::malformed_k);
        failures += expect_eq(index.value_of(0).status, status_t::malformed_k);
        failures += expect_eq(index.degree_of(0).status, status_t::malformed_k);
        failures += expect_eq(index.link(0, 1).status, status_t::malformed_k);
        failures += expect_eq(index.unlink(0, 1), status_t::malformed_k);

        // The capacity freezes on the first success; later calls have zero side effects.
        failures += expect_eq(index.reserve(0).status, status_t::config_mismatch_k);
        if constexpr (sizeof(node_id_t) == 4)
            failures += expect_eq(index.reserve(std::size_t {1} << 32).status, status_t::overflow_k);
        auto const reserved = index.reserve(quota_k);
        failures += expect_eq(reserved.status, status_t::success_k);
        failures += expect_eq(reserved.value, quota_k);
        failures += expect_eq(index.reserve(quota_k).status, status_t::success_k);
        failures += expect_eq(index.reserve(quota_k - 1).status, status_t::success_k);
        failures += expect_eq(index.reserve(quota_k + 1).status, status_t::overflow_k);
        failures += expect_eq(index.reserve(quota_k + 1).value, quota_k);
        failures += expect_eq(index.capacity(), quota_k);

        // Argument guards, each with zero side effects.
        std::vector<node_id_t> assigned_ids(1);
        failures += expect_eq(
            index.insert({values.data(), 2}, {keys.data(), 2}, metric, {assigned_ids.data(), 1}, context).status,
            status_t::config_mismatch_k);
        failures += expect_eq(index.insert({values.data(), 2}, {keys.data(), 1}, metric, {}, context).status,
                              status_t::config_mismatch_k);
        failures += expect_eq(
            index.insert({values.data(), 1}, {keys.data(), 1}, metric, {}, context, basic_keep_closest_t {0}).status,
            status_t::config_mismatch_k);
        failures += expect_eq(index.insert({values.data(), 0}, {keys.data(), 0}, metric, {}, context).status,
                              status_t::success_k);
        failures += expect_eq(index.size(), 0u);

        // A rejected over-capacity batch must roll its claim back completely.
        failures += expect_eq(
            index.insert({values.data(), quota_k + 1}, {keys.data(), quota_k + 1}, metric, {}, context).status,
            status_t::overflow_k);
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index
                                  .insert({values.data(), quota_k}, {keys.data(), quota_k}, metric, {}, context,
                                          basic_keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        failures += expect_eq(index.size(), quota_k);
        failures += expect_eq(
            index.insert({values.data() + quota_k, 1}, {keys.data() + quota_k, 1}, metric, {}, context).status,
            status_t::overflow_k);
        failures += expect_eq(index.size(), quota_k);

        // The index's planes are provably on the ledger, not the plain heap.
        failures += expect(ledger.live_bytes.load() > 0);
        failures += expect(ledger.total_allocations.load() > 0);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Three values whose pairwise distances the unit states by hand: the oracle must
 *      reproduce the stated matrix, and every value must find itself at distance zero. */
template <typename fixture_type_>
std::size_t test_basic_handmade(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_guards_k, counting_allocator<std::byte> {ledger}};
        auto handmade = fixture.make_handmade();
        auto queries = fixture.make_handmade(); // Deterministic, so a fresh unmoved copy.

        failures += expect(bool(index.reserve(3)));
        std::array<typename index_t::key_t, 3> keys {fixture.make_key(0), fixture.make_key(1), fixture.make_key(2)};
        std::array<typename index_t::node_id_t, 3> assigned_ids {};
        failures += expect_eq(index.insert({handmade.values}, {keys}, metric, {assigned_ids}, context).status,
                              status_t::success_k);

        for (std::size_t row = 0; row != 3; ++row)
            for (std::size_t column = 0; column != 3; ++column)
                failures += basic_expect_distance<fixture_type_>(
                    fixture.oracle(queries.values[row], index.published_value_of(assigned_ids[column])),
                    handmade.distances[row][column]);

        for (std::size_t row = 0; row != 3; ++row) {
            auto const found = index.search(queries.values[row], metric, context);
            failures += expect_eq(found.status, status_t::success_k);
            failures += expect(found.count >= 3);
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

/** @brief Fills the index, then samples queries against the exact brute-force oracle:
 *      returned distances must be true, the best must be the true best for at least the
 *      recall bar, and the telemetry must have counted the work. */
template <typename fixture_type_>
std::size_t test_basic_growth_and_recall(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_growth_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(basic_growth_nodes_k + basic_recall_samples_k, basic_seed_growth_k);
        std::vector<typename index_t::value_t> queries;
        for (std::size_t sample = 0; sample != basic_recall_samples_k; ++sample)
            queries.push_back(std::move(values[basic_growth_nodes_k + sample]));
        values.resize(basic_growth_nodes_k);

        failures += expect(bool(index.reserve(basic_growth_nodes_k)));
        std::vector<typename index_t::node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});
        failures += expect_eq(index.size(), basic_growth_nodes_k);

        // A single-threaded build dispenses a contiguous id window in submission order.
        for (std::size_t ordinal = 0; ordinal != assigned_ids.size(); ++ordinal)
            failures += expect_eq(static_cast<std::size_t>(assigned_ids[ordinal]), ordinal);
        failures += expect(context.telemetry.computed_distances > 0);
        failures += expect(context.telemetry.visited_members > 0);

        std::size_t found_best_count = 0;
        for (auto const& query : queries) {
            auto const found = index.search(query, metric, context);
            failures += expect_eq(found.status, status_t::success_k);
            failures += expect(found.count > 0 && found.count <= connectivity_32_k * 2);
            failures += basic_expect_ascending(context);
            bool found_best = false;
            failures += basic_audit_search(fixture, index, query, context, found_best);
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
            auto const found = index.search(queries[0], metric, context, config);
            failures += expect_eq(found.status, status_t::success_k);
            failures += basic_expect_ascending(context);
        }
        typename index_t::search_config_t const exhaustive_config {basic_growth_nodes_k, 0};
        for (std::size_t sample = 0; sample != 4; ++sample) {
            failures += expect_eq(index.search(queries[sample], metric, context, exhaustive_config).status,
                                  status_t::success_k);
            bool found_best = false;
            failures += basic_audit_search(fixture, index, queries[sample], context, found_best);
            failures += expect(found_best); // A whole-graph pool cannot miss.
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The same data built through both shipped selectors - the per-call budget makes
 *      this one index instantiation, two graphs. The diverse build must clear the full
 *      recall bar, the closest build its navigability floor, and the wiring must differ
 *      structurally, or the selector argument is being ignored. */
template <typename fixture_type_>
std::size_t test_basic_selectors(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    auto build_and_report = [&](auto selector, std::size_t minimal_recall_percent,
                                typename index_t::graph_health_t& health) {
        allocation_ledger_t ledger;
        {
            index_t index = fixture.make_index(ledger);
            auto metric = fixture.make_metric();
            context_t context {basic_seed_selectors_k, counting_allocator<std::byte> {ledger}};
            // Queries split off the same draw as the stored values: a separate generator
            // call would plant different clusters, leaving every query off-distribution.
            auto values = fixture.make_values(basic_growth_nodes_k / 2 + basic_recall_samples_k,
                                              basic_seed_selectors_k);
            std::vector<typename index_t::value_t> queries;
            for (std::size_t sample = 0; sample != basic_recall_samples_k; ++sample)
                queries.push_back(std::move(values[basic_growth_nodes_k / 2 + sample]));
            values.resize(basic_growth_nodes_k / 2);

            failures += expect(bool(index.reserve(values.size())));
            std::vector<typename index_t::node_id_t> assigned_ids;
            failures += basic_build(fixture, index, metric, values, assigned_ids, context, selector);
            health = index.health(context);
            failures += expect_eq(health.status, status_t::success_k);
            failures += expect(health.degree_min >= 1);

            std::size_t found_best_count = 0;
            for (auto const& query : queries) {
                auto const found = index.search(query, metric, context);
                failures += expect_eq(found.status, status_t::success_k);
                bool found_best = false;
                failures += basic_audit_search(fixture, index, query, context, found_best);
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
    build_and_report(basic_keep_closest_t {basic_connectivity_k}, basic_relaxed_recall_percent_k, closest_health);
    build_and_report(basic_keep_diverse_t {basic_connectivity_k}, basic_minimal_recall_percent_k, diverse_health);
    failures += expect(closest_health.mean_degree != diverse_health.mean_degree ||
                       closest_health.reciprocal_edges != diverse_health.reciprocal_edges);
    return failures;
}

/** @brief Keyed lookups with planted duplicates, id → key staying authoritative, and the
 *      checked id-readers answering @c malformed_k at & past the watermark. */
template <typename fixture_type_>
std::size_t test_basic_keys(fixture_type_ const& fixture) {
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
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_keys_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_keys_k);
        std::vector<key_t> keys(nodes_k);
        for (std::size_t ordinal = 0; ordinal != unique_k; ++ordinal) keys[ordinal] = fixture.make_key(ordinal);
        for (std::size_t ordinal = unique_k; ordinal != nodes_k; ++ordinal)
            keys[ordinal] = fixture.make_key(1000 + ordinal % shared_keys_k);

        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids(nodes_k);
        failures += expect_eq(
            index.insert({values}, {keys}, metric, {assigned_ids}, context, basic_keep_diverse_t {basic_connectivity_k})
                .status,
            status_t::success_k);

        std::vector<std::size_t> all_found_ids;
        for (std::size_t ordinal = 0; ordinal != unique_k; ++ordinal) {
            failures += expect(index.contains(keys[ordinal]));
            failures += expect_eq(index.count(keys[ordinal]), 1u);
            auto const found = index.find(keys[ordinal]);
            failures += expect(bool(found));
            failures += expect(index.key_of(found.value).value == keys[ordinal]);
            all_found_ids.push_back(static_cast<std::size_t>(found.value));
        }
        for (std::size_t shared = 0; shared != shared_keys_k; ++shared) {
            key_t const key = fixture.make_key(1000 + shared);
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
        std::sort(all_found_ids.begin(), all_found_ids.end());
        failures += expect(std::unique(all_found_ids.begin(), all_found_ids.end()) == all_found_ids.end());
        failures += expect_eq(all_found_ids.size(), nodes_k);

        key_t const absent = fixture.make_key(5000);
        failures += expect_eq(index.find(absent).status, status_t::malformed_k);
        failures += expect(!index.contains(absent));
        failures += expect_eq(index.count(absent), 0u);

        // The checked id-readers gate every input on the watermark.
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

/** @brief Manual wiring: idempotent links, honest rejections, degree bookkeeping, growth
 *      past the build budget, and a snapshot roundtrip preserving variable degrees. */
template <typename fixture_type_>
std::size_t test_basic_wiring(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 24;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_wiring_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_wiring_k);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context, basic_keep_diverse_t {4});

        node_id_t const from = 0, to = static_cast<node_id_t>(nodes_k - 1);
        [[maybe_unused]] status_t const cleared = index.unlink(from, to); // Absent or not, now absent.
        std::size_t const degree_before = index.degree_of(from).value;

        auto const linked = index.link(from, to);
        failures += expect_eq(linked.status, status_t::success_k);
        failures += expect(linked.value == membership_t::fresh_k);
        failures += expect_eq(index.degree_of(from).value, degree_before + 1);
        auto const relinked = index.link(from, to);
        failures += expect_eq(relinked.status, status_t::success_k);
        failures += expect(relinked.value == membership_t::present_k);
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

        // Variable degrees survive the snapshot - the header's zero-connectivity path.
        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);
        index_t restored = fixture.make_index(ledger);
        failures += expect(bool(restored.reserve(nodes_k)));
        failures += expect_eq(restored.load({payload}, codec), status_t::success_k);
        failures += expect_eq(restored.size(), index.size());
        for (std::size_t node = 0; node != nodes_k; ++node)
            failures += expect_eq(restored.degree_of(static_cast<node_id_t>(node)).value,
                                  index.degree_of(static_cast<node_id_t>(node)).value);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief All-or-nothing under a starved ledger: a refused reserve leaves the shape
 *      unfrozen & retryable, and a refused link-growth leaves the row untouched. */
template <typename fixture_type_>
std::size_t test_basic_allocation_budget(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_budget_k, counting_allocator<std::byte> {ledger}};

        ledger.byte_budget.store(ledger.live_bytes.load() + 64);
        failures += expect_eq(index.reserve(basic_growth_nodes_k).status, status_t::bad_alloc_k);
        failures += expect_eq(index.capacity(), 0u);
        ledger.byte_budget.store(PTRDIFF_MAX);
        failures += expect_eq(index.reserve(basic_growth_nodes_k).status, status_t::success_k);

        constexpr std::size_t linked_nodes_k = 40;
        auto values = fixture.make_values(linked_nodes_k, basic_seed_budget_k);
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context, basic_keep_closest_t {1});

        // Amortized row growth leaves power-of-two slack, so links absorb it for free -
        // but the slack is finite, and a starved ledger must refuse the first real
        // growth without touching the row.
        std::size_t const degree_before = index.degree_of(0).value;
        ledger.byte_budget.store(ledger.live_bytes.load());
        std::size_t fresh_links = 0;
        node_id_t refused_target = 0;
        for (std::size_t target = 1; target != linked_nodes_k && !refused_target; ++target) {
            auto const linked = index.link(0, static_cast<node_id_t>(target));
            if (linked.status == status_t::bad_alloc_k) refused_target = static_cast<node_id_t>(target);
            else if (succeeded(linked.status) && linked.value == membership_t::fresh_k) ++fresh_links;
        }
        failures += expect(refused_target != 0);
        failures += expect_eq(index.degree_of(0).value, degree_before + fresh_links);

        ledger.byte_budget.store(PTRDIFF_MAX);
        auto const retried = index.link(0, refused_target);
        failures += expect_eq(retried.status, status_t::success_k);
        failures += expect(retried.value == membership_t::fresh_k);
        failures += expect_eq(index.degree_of(0).value, degree_before + fresh_links + 1);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Snapshot roundtrip: sized exactly, refused when short, loaded into a fresh
 *      index that answers every query identically and mirrors every id's key & degree. */
template <typename fixture_type_>
std::size_t test_basic_snapshot_roundtrip(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using key_t = typename index_t::key_t; // Shadows POSIX ::key_t, which binds otherwise.
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 48;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_snapshot_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_snapshot_k);
        auto queries = fixture.make_values(8, basic_seed_snapshot_k ^ 0xFFull);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});

        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        failures += expect_eq(index.snapshot_bytes(codec).value, bytes.value); // Sizing is repeatable.
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload.data(), payload.size() - 1}, codec), status_t::overflow_k);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);

        index_t restored = fixture.make_index(ledger);
        failures += expect_eq(restored.load({payload}, codec), status_t::uninitialized_k);
        failures += expect(bool(restored.reserve(nodes_k)));
        failures += expect_eq(restored.load({payload}, codec), status_t::success_k);
        failures += expect_eq(restored.size(), index.size());
        for (std::size_t node = 0; node != nodes_k; ++node) {
            node_id_t const node_id = static_cast<node_id_t>(node);
            failures += expect(restored.key_of(node_id).value == index.key_of(node_id).value);
            failures += expect_eq(restored.degree_of(node_id).value, index.degree_of(node_id).value);
            failures += expect(fixture.value_equals(restored.published_value_of(node_id), //
                                                    index.published_value_of(node_id)));
        }

        // Same seed, same graph, same code path: the answers must be identical.
        for (auto const& query : queries) {
            context_t original_context {basic_seed_snapshot_k + 7, counting_allocator<std::byte> {ledger}};
            context_t restored_context {basic_seed_snapshot_k + 7, counting_allocator<std::byte> {ledger}};
            auto original_metric = fixture.make_metric();
            auto restored_metric = fixture.make_metric();
            auto const original_found = index.search(query, original_metric, original_context);
            auto const restored_found = restored.search(query, restored_metric, restored_context);
            failures += expect_eq(original_found.status, restored_found.status);
            failures += expect_eq(original_found.count, restored_found.count);
            for (std::size_t position = 0; position != original_found.count; ++position) {
                failures += expect_eq(original_context.results[position].node_id,
                                      restored_context.results[position].node_id);
                failures += basic_expect_distance<fixture_type_>(restored_context.results[position].distance,
                                                                 original_context.results[position].distance);
            }
        }

        // Loading into a non-empty index is refused; the restored index stays intact.
        failures += expect_eq(restored.load({payload}, codec), status_t::config_mismatch_k);

        // The 64-byte alignment floor: the same sections re-packed tighter, and the
        // reader must honor the header's alignment rather than assume the writer's.
        auto const repacked = basic_reencode_alignment<key_t, node_id_t>(payload, 64);
        index_t repacked_restored = fixture.make_index(ledger);
        failures += expect(bool(repacked_restored.reserve(nodes_k)));
        failures += expect_eq(repacked_restored.load({repacked}, codec), status_t::success_k);
        failures += expect_eq(repacked_restored.size(), index.size());
        for (std::size_t node = 0; node != nodes_k; ++node) {
            node_id_t const node_id = static_cast<node_id_t>(node);
            failures += expect(repacked_restored.key_of(node_id).value == index.key_of(node_id).value);
            failures += expect_eq(repacked_restored.degree_of(node_id).value, index.degree_of(node_id).value);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Hostile payloads: every structural mutation of a good snapshot must be refused
 *      with its documented status, and the same index must load the intact payload after
 *      every refusal - the retryability contract. */
template <typename fixture_type_>
std::size_t test_basic_snapshot_hostile(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 16;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_hostile_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_hostile_k);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context, basic_keep_diverse_t {4});
        // One extra manual edge de-uniformizes the degrees, forcing the variable-degree
        // header convention - and guaranteeing at least one serialized neighbor id.
        [[maybe_unused]] auto const bumped = index.link(0, static_cast<node_id_t>(nodes_k / 2));

        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);
        basic_snapshot_header_t const header = basic_read_object<basic_snapshot_header_t>(payload, 0);
        failures += expect_eq(header.connectivity, 0u); // Varying degrees serialize as zero.
        failures += expect(header.neighbors_count > 0u);

        auto const sections = basic_payload_sections<key_t, node_id_t>(header);
        std::size_t const neighbor_sums_offset = sections.neighbor_sums_offset;
        std::size_t const neighbor_ids_offset = sections.neighbor_ids_offset;
        std::size_t const value_sums_offset = sections.value_sums_offset;

        auto expect_refused = [&](auto mutate, status_t wanted) {
            index_t fresh = fixture.make_index(ledger);
            failures += expect(bool(fresh.reserve(nodes_k)));
            std::vector<std::byte> corrupted = payload;
            mutate(corrupted);
            failures += expect_eq(fresh.load({corrupted}, codec), wanted);
            // Every refusal must leave the same index loadable - the retryability contract.
            failures += expect_eq(fresh.load({payload}, codec), status_t::success_k);
        };

        auto patch_header = [&](auto mutate_header) {
            return [&, mutate_header](std::vector<std::byte>& corrupted) {
                basic_snapshot_header_t patched = basic_read_object<basic_snapshot_header_t>(corrupted, 0);
                mutate_header(patched);
                basic_write_object(corrupted, 0, patched);
            };
        };

        expect_refused([](std::vector<std::byte>& c) { c.resize(sizeof(basic_snapshot_header_t) - 1); },
                       status_t::malformed_k);
        expect_refused([](std::vector<std::byte>& c) { c.resize(c.size() - 1); }, status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.magic ^= 1; }), status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.layout ^= 1; }), status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.version += 1; }), status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.alignment = 32; }), status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.alignment = 96; }), status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.node_id_bytes = 3; }),
                       status_t::config_mismatch_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.key_bytes = 3; }), status_t::config_mismatch_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.node_count = nodes_k + 1; }),
                       status_t::overflow_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.connectivity = nodes_k + 1; }),
                       status_t::malformed_k);
        expect_refused(patch_header([](basic_snapshot_header_t& h) { h.neighbors_count = nodes_k * nodes_k + 1; }),
                       status_t::malformed_k);
        expect_refused(patch_header([&](basic_snapshot_header_t& h) { h.value_bytes = payload.size() + 1; }),
                       status_t::malformed_k);
        expect_refused([&](std::vector<std::byte>& c) { basic_write_object(c, neighbor_sums_offset, i64_t {1}); },
                       status_t::malformed_k);
        expect_refused(
            [&](std::vector<std::byte>& c) { basic_write_object(c, neighbor_sums_offset + sizeof(i64_t), i64_t {-1}); },
            status_t::malformed_k);
        expect_refused(
            [&](std::vector<std::byte>& c) {
                std::size_t const last_sum_offset = neighbor_sums_offset + nodes_k * sizeof(i64_t);
                basic_write_object(c, last_sum_offset, basic_read_object<i64_t>(c, last_sum_offset) + 1);
            },
            status_t::malformed_k);
        expect_refused(
            [&](std::vector<std::byte>& c) {
                basic_write_object(c, neighbor_ids_offset, static_cast<node_id_t>(nodes_k));
            },
            status_t::malformed_k);
        expect_refused([&](std::vector<std::byte>& c) { basic_write_object(c, value_sums_offset, i64_t {1}); },
                       status_t::malformed_k);
        expect_refused(
            [&](std::vector<std::byte>& c) {
                basic_write_object(c, value_sums_offset + sizeof(i64_t), static_cast<i64_t>(header.value_bytes) + 1);
            },
            status_t::malformed_k);

        // The one row that needs an unreserved target instead of a fresh reserved one.
        index_t unreserved = fixture.make_index(ledger);
        failures += expect_eq(unreserved.load({payload}, codec), status_t::uninitialized_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Region-scoped serialization: a save plan whose uneven regions run from a pool
 *      must emit bytes identical to the single-call form, a region-parallel load must
 *      rebuild a graph whose own re-save is byte-identical, and the misuse rows -
 *      premature finish, out-of-bounds region, dropped plan - answer their documented
 *      statuses while the hold blocks & then frees the unique verbs. */
template <typename fixture_type_>
std::size_t test_basic_snapshot_regions(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 96;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_regions_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_regions_k);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});

        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> single_payload(bytes.value);
        failures += expect_eq(index.save({single_payload}, codec), status_t::success_k);

        // Uneven splits from a pool must land byte-identically to the single call.
        std::vector<std::byte> pooled_payload(bytes.value, std::byte {0xEE});
        std::array<iota_view<node_id_t>, 3> const regions {iota_view<node_id_t> {0, 7}, iota_view<node_id_t> {7, 61},
                                                           iota_view<node_id_t> {61, nodes_k}};
        {
            auto planned = index.plan_save({pooled_payload}, codec);
            failures += expect(bool(planned));
            failures += expect_eq(planned.value.finish(), status_t::config_mismatch_k); // Nothing covered yet.
            failures += expect_eq(planned.value.save({0, nodes_k + 1}, codec), status_t::config_mismatch_k);
            failures += expect_eq(index.snapshot_bytes(codec).status, status_t::contended_k); // The plan holds unique.

            fu::flat_pool_t pool;
            if (fu::failed(pool.spawn(regions.size()))) return failures + expect(false);
            std::atomic<std::size_t> region_failures {0};
            pool.for_threads([&](std::size_t lane) noexcept {
                if (planned.value.save(regions[lane], codec) != status_t::success_k) region_failures.fetch_add(1);
            });
            failures += expect_eq(region_failures.load(), 0u);
            failures += expect_eq(planned.value.finish(), status_t::success_k);
        }
        failures += expect(std::equal(pooled_payload.begin(), pooled_payload.end(), single_payload.begin()));
        failures += expect(bool(index.snapshot_bytes(codec))); // The finished plan released its hold.

        // A dropped save plan releases the hold without pronouncing anything.
        {
            auto planned = index.plan_save({pooled_payload}, codec);
            failures += expect(bool(planned));
        }
        failures += expect(bool(index.snapshot_bytes(codec)));

        // The load mirror: uneven regions decode concurrently, publication waits for finish.
        index_t restored = fixture.make_index(ledger);
        failures += expect(bool(restored.reserve(nodes_k)));
        {
            auto planned = restored.plan_load({single_payload});
            failures += expect(bool(planned));
            failures += expect_eq(planned.value.finish(), status_t::config_mismatch_k); // Nothing covered yet.
            failures += expect_eq(restored.size(), 0u);                                 // Nothing published either.

            fu::flat_pool_t pool;
            if (fu::failed(pool.spawn(regions.size()))) return failures + expect(false);
            std::atomic<std::size_t> region_failures {0};
            pool.for_threads([&](std::size_t lane) noexcept {
                if (planned.value.load(regions[lane], codec) != status_t::success_k) region_failures.fetch_add(1);
            });
            failures += expect_eq(region_failures.load(), 0u);
            failures += expect_eq(restored.size(), 0u); // Still unpublished before the finish.
            failures += expect_eq(planned.value.finish(), status_t::success_k);
        }
        failures += expect_eq(restored.size(), index.size());
        for (std::size_t node = 0; node != nodes_k; ++node) {
            node_id_t const node_id = static_cast<node_id_t>(node);
            failures += expect(restored.key_of(node_id).value == index.key_of(node_id).value);
            failures += expect_eq(restored.degree_of(node_id).value, index.degree_of(node_id).value);
            failures += expect(fixture.value_equals(restored.published_value_of(node_id), //
                                                    index.published_value_of(node_id)));
        }

        // Closure: the region-loaded graph re-saves to the very same bytes.
        std::vector<std::byte> resaved_payload(bytes.value, std::byte {0x77});
        failures += expect_eq(restored.save({resaved_payload}, codec), status_t::success_k);
        failures += expect(std::equal(resaved_payload.begin(), resaved_payload.end(), single_payload.begin()));

        // A dropped load plan publishes nothing and leaves a clean retry behind.
        index_t abandoned = fixture.make_index(ledger);
        failures += expect(bool(abandoned.reserve(nodes_k)));
        {
            auto planned = abandoned.plan_load({single_payload});
            failures += expect(bool(planned));
            failures += expect_eq(planned.value.load(regions[0], codec), status_t::success_k);
        }
        failures += expect_eq(abandoned.size(), 0u);
        failures += expect_eq(abandoned.load({single_payload}, codec), status_t::success_k);
        failures += expect_eq(abandoned.size(), index.size());
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The health report on a settled graph, its determinism across repeated calls,
 *      and - through a gated metric parking one insert mid-flight - the settled-past
 *      guard refusing @c health & @c snapshot_bytes with @c contended_k. */
template <typename fixture_type_>
std::size_t test_basic_health(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture.make_metric());
    constexpr std::size_t nodes_k = 64;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_health_k, counting_allocator<std::byte> {ledger}};

        failures += expect(bool(index.reserve(nodes_k + 1)));
        auto const empty_report = index.health(context);
        failures += expect_eq(empty_report.status, status_t::success_k);
        failures += expect_eq(empty_report.connected_components, 0u);
        failures += expect_eq(empty_report.degree_max, 0u);

        auto values = fixture.make_values(nodes_k, basic_seed_health_k);
        std::vector<typename index_t::node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});

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
        auto one_more = fixture.make_values(1, basic_seed_health_k ^ 0xF00Dull);
        typename index_t::key_t const one_more_key = fixture.make_key(nodes_k);
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> parked_failures {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture.make_metric(), &entered,
                                                                                  &released};
                context_t parked_context {basic_seed_health_k + 1, counting_allocator<std::byte> {ledger}};
                auto const inserted = index.insert({one_more.data(), 1}, {&one_more_key, 1}, gated, {}, parked_context,
                                                   basic_keep_diverse_t {basic_connectivity_k});
                if (inserted.status != status_t::success_k) parked_failures.fetch_add(1);
                return;
            }
            if (lane != 1) return;
            for (std::size_t spins = 0; !entered.load(std::memory_order_acquire); ++spins) {
                if (spins > (std::size_t {1} << 28)) {
                    // The gate must open even on a failed rendezvous, or the pool deadlocks.
                    parked_failures.fetch_add(1);
                    released.store(true, std::memory_order_release);
                    return;
                }
                std::this_thread::yield();
            }
            context_t checker_context {basic_seed_health_k + 2, counting_allocator<std::byte> {ledger}};
            if (index.health(checker_context).status != status_t::contended_k) parked_failures.fetch_add(1);
            if (index.snapshot_bytes(codec).status != status_t::contended_k) parked_failures.fetch_add(1);
            released.store(true, std::memory_order_release);
        });
        failures += expect_eq(parked_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_k + 1);
        failures += expect_eq(index.health(context).status, status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Concurrent build on distinct-seeded contexts: dense unique ids, every lane's
 *      keys landing on its own values, and the payloads surviving the move intact. */
template <typename fixture_type_>
std::size_t test_basic_concurrent_build(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    std::size_t const nodes_count = basic_lanes_k * basic_lane_nodes_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        failures += expect(bool(index.reserve(nodes_count)));

        std::vector<std::vector<typename index_t::value_t>> lane_values;
        std::vector<std::vector<key_t>> lane_keys;
        std::vector<context_t> lane_contexts;
        std::vector<decltype(fixture.make_metric())> lane_metrics;
        for (std::size_t lane = 0; lane != basic_lanes_k; ++lane) {
            lane_values.push_back(fixture.make_values(basic_lane_nodes_k, basic_seed_build_lane_k + lane));
            std::vector<key_t> keys(basic_lane_nodes_k);
            for (std::size_t ordinal = 0; ordinal != basic_lane_nodes_k; ++ordinal)
                keys[ordinal] = fixture.make_key(lane * basic_lane_nodes_k + ordinal);
            lane_keys.push_back(std::move(keys));
            lane_contexts.emplace_back(basic_seed_build_lane_k + lane, counting_allocator<std::byte> {ledger});
            lane_metrics.push_back(fixture.make_metric());
        }
        std::vector<node_id_t> assigned_ids(nodes_count);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(basic_lanes_k))) return failures + expect(false);
        std::atomic<std::size_t> failed_lanes {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= basic_lanes_k) return;
            auto const inserted = index.insert({lane_values[lane]}, {lane_keys[lane]}, lane_metrics[lane],
                                               {assigned_ids.data() + lane * basic_lane_nodes_k, basic_lane_nodes_k},
                                               lane_contexts[lane], basic_keep_diverse_t {basic_connectivity_k});
            if (inserted.status != status_t::success_k) failed_lanes.fetch_add(1);
        });

        failures += expect_eq(failed_lanes.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
        // Ids are dense & unique across lanes - the watermark's dispensed-order turn.
        std::vector<node_id_t> sorted_ids = assigned_ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        failures += expect(std::unique(sorted_ids.begin(), sorted_ids.end()) == sorted_ids.end());
        failures += expect(sorted_ids.front() == 0 && static_cast<std::size_t>(sorted_ids.back()) == nodes_count - 1);

        // Every lane's keys & values landed on the ids that lane was assigned.
        for (std::size_t lane = 0; lane != basic_lanes_k; ++lane) {
            auto expected_values = fixture.make_values(basic_lane_nodes_k, basic_seed_build_lane_k + lane);
            for (std::size_t ordinal = 0; ordinal != basic_lane_nodes_k; ++ordinal) {
                node_id_t const node_id = assigned_ids[lane * basic_lane_nodes_k + ordinal];
                failures += expect(index.key_of(node_id).value ==
                                   fixture.make_key(lane * basic_lane_nodes_k + ordinal));
                failures += expect(fixture.value_equals(index.published_value_of(node_id), expected_values[ordinal]));
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief A reader lane races writers: every result id sits below a watermark sampled
 *      after its search, distances ascend, and the checked id-readers stay honest under
 *      any input while the graph grows underneath them. */
template <typename fixture_type_>
std::size_t test_basic_concurrent_search(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t writer_lanes_k = basic_lanes_k - 1;
    constexpr std::size_t prefill_k = 32;
    std::size_t const nodes_count = prefill_k + writer_lanes_k * basic_lane_nodes_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        failures += expect(bool(index.reserve(nodes_count)));
        auto prefill_metric = fixture.make_metric();
        context_t prefill_context {basic_seed_search_lane_k, counting_allocator<std::byte> {ledger}};
        auto prefill_values = fixture.make_values(prefill_k, basic_seed_search_lane_k);
        std::vector<node_id_t> prefill_ids;
        failures += basic_build(fixture, index, prefill_metric, prefill_values, prefill_ids, prefill_context,
                                basic_keep_diverse_t {basic_connectivity_k});

        std::vector<std::vector<typename index_t::value_t>> lane_values;
        std::vector<std::vector<key_t>> lane_keys;
        std::vector<context_t> lane_contexts;
        std::vector<decltype(fixture.make_metric())> lane_metrics;
        for (std::size_t lane = 0; lane != writer_lanes_k; ++lane) {
            lane_values.push_back(fixture.make_values(basic_lane_nodes_k, basic_seed_search_lane_k + 1 + lane));
            std::vector<key_t> keys(basic_lane_nodes_k);
            for (std::size_t ordinal = 0; ordinal != basic_lane_nodes_k; ++ordinal)
                keys[ordinal] = fixture.make_key(prefill_k + lane * basic_lane_nodes_k + ordinal);
            lane_keys.push_back(std::move(keys));
            lane_contexts.emplace_back(basic_seed_search_lane_k + 1 + lane, counting_allocator<std::byte> {ledger});
            lane_metrics.push_back(fixture.make_metric());
        }
        // Queries are never inserted, so any sanitizer report here is an index bug.
        auto queries = fixture.make_values(basic_reader_minimal_searches_k, basic_seed_search_lane_k ^ 0xFFFFull);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(writer_lanes_k + 1))) return failures + expect(false);
        std::atomic<std::size_t> lane_failures {0};
        std::atomic<std::size_t> writers_done {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane < writer_lanes_k) {
                auto const inserted = index.insert({lane_values[lane]}, {lane_keys[lane]}, lane_metrics[lane], {},
                                                   lane_contexts[lane], basic_keep_diverse_t {basic_connectivity_k});
                if (inserted.status != status_t::success_k) lane_failures.fetch_add(1);
                writers_done.fetch_add(1);
                return;
            }
            if (lane != writer_lanes_k) return;
            auto reader_metric = fixture.make_metric();
            context_t reader_context {basic_seed_search_lane_k + 100, counting_allocator<std::byte> {ledger}};
            splitmix64_t reader_random {basic_seed_search_lane_k + 200};
            uniform_below<u32_t> probe_distribution {static_cast<u32_t>(nodes_count)};
            std::size_t searches = 0;
            while (writers_done.load(std::memory_order_acquire) != writer_lanes_k ||
                   searches < basic_reader_minimal_searches_k) {
                auto const& query = queries[searches % queries.size()];
                auto const found = index.search(query, reader_metric, reader_context);
                if (found.status != status_t::success_k) lane_failures.fetch_add(1);
                std::size_t const visible_after = index.size();
                for (std::size_t position = 0; position != reader_context.results.size(); ++position) {
                    if (static_cast<std::size_t>(reader_context.results[position].node_id) >= visible_after)
                        lane_failures.fetch_add(1);
                    if (position &&
                        reader_context.results[position].distance < reader_context.results[position - 1].distance)
                        lane_failures.fetch_add(1);
                }
                // Any id is a legal input to the checked readers, published or not.
                node_id_t const probe = static_cast<node_id_t>(probe_distribution(reader_random));
                std::size_t const visible_before = index.size();
                auto const probed_key = index.key_of(probe);
                auto const probed_degree = index.degree_of(probe);
                if (static_cast<std::size_t>(probe) < visible_before)
                    if (!probed_key || !probed_degree) lane_failures.fetch_add(1);
                if (!probed_key && probed_key.status != status_t::malformed_k) lane_failures.fetch_add(1);
                ++searches;
            }
        });
        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect_eq(index.size(), nodes_count);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The metric-contract audit: every drain sized to its round's pushes, no round
 *      abandoned mid-flight, telemetry equal to the audited push count, and armed
 *      refusals driving the saturation paths without ever breaking a walk. */
template <typename fixture_type_>
std::size_t test_basic_metric_contract(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using distance_t = typename index_t::distance_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture.make_metric());
    constexpr std::size_t nodes_k = 48;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        basic_auditing_metric<metric_t, distance_t> metric {fixture.make_metric()};
        context_t context {basic_seed_contract_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_contract_k);
        auto queries = fixture.make_values(4, basic_seed_contract_k ^ 0xFFull);

        failures += expect(bool(index.reserve(nodes_k + 1)));
        std::vector<typename index_t::node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});
        failures += expect_eq(metric.violations, 0u);
        failures += expect(metric.resets > 0 && metric.drains > 0);
        // Every drained lane is one counted distance, so the two tallies must agree.
        failures += expect_eq(context.telemetry.computed_distances, metric.pushes);

        for (auto const& query : queries) {
            failures += expect_eq(index.search(query, metric, context).status, status_t::success_k);
            failures += basic_expect_ascending(context);
        }
        failures += expect_eq(metric.violations, 0u);

        // Armed refusal: the walk must survive every saturation path & stay ranked.
        for (std::size_t budget : {basic_connectivity_k, std::size_t {1}, std::size_t {0}}) {
            metric.arm(budget);
            auto const found = index.search(queries[0], metric, context);
            failures += expect_eq(found.status, status_t::success_k);
            failures += basic_expect_ascending(context);
            metric.disarm();
        }
        failures += expect_eq(metric.violations, 0u);

        // An armed insert wires a thinner node but still publishes it.
        metric.arm(2);
        auto one_more = fixture.make_values(1, basic_seed_contract_k ^ 0xAAull);
        typename index_t::key_t const one_more_key = fixture.make_key(nodes_k);
        failures += expect_eq(index
                                  .insert({one_more.data(), 1}, {&one_more_key, 1}, metric, {}, context,
                                          basic_keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        metric.disarm();
        failures += expect_eq(index.size(), nodes_k + 1);
        failures += expect_eq(metric.violations, 0u);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Frontier starvation drives re-seeding: islands - nodes manually stripped of
 *      every edge in both directions - are reachable only through a restart's landmark
 *      draws, and a whole-graph pool must still find every node. */
template <typename fixture_type_>
std::size_t test_basic_restarts(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t islands_k = 4;
    constexpr std::size_t attempts_k = 4;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_restarts_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k + 1, basic_seed_restarts_k);
        auto query = std::move(values[nodes_k]);
        values.resize(nodes_k);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});

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
            context_t attempt_context {basic_seed_restarts_k + 1 + attempt, counting_allocator<std::byte> {ledger}};
            auto const found = index.search(query, metric, attempt_context, {nodes_k, 0});
            failures += expect_eq(found.status, status_t::success_k);
            failures += expect_eq(found.count, nodes_k); // The islands included - full coverage.
            any_restarts = any_restarts || attempt_context.telemetry.restarts > 0;
        }
        failures += expect(any_restarts);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The certified arms at runtime: an exclusive build equals a concurrent
 *      single-threaded one, certified searches answer identically to the concurrent arm
 *      on a settled graph, and a parked insert makes both certified arms answer
 *      @c contended_k instead of tearing. */
template <typename fixture_type_>
std::size_t test_basic_certified(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    using metric_t = decltype(fixture.make_metric());
    constexpr std::size_t nodes_k = 48;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t exclusive_index = fixture.make_index(ledger);
        index_t concurrent_index = fixture.make_index(ledger);
        failures += expect(bool(exclusive_index.reserve(nodes_k + 1)));
        failures += expect(bool(concurrent_index.reserve(nodes_k + 1)));
        auto exclusive_values = fixture.make_values(nodes_k, basic_seed_certified_k);
        auto concurrent_values = fixture.make_values(nodes_k, basic_seed_certified_k);
        std::vector<key_t> keys(nodes_k);
        for (std::size_t ordinal = 0; ordinal != nodes_k; ++ordinal) keys[ordinal] = fixture.make_key(ordinal);
        auto exclusive_metric = fixture.make_metric();
        auto concurrent_metric = fixture.make_metric();
        context_t exclusive_context {basic_seed_certified_k, counting_allocator<std::byte> {ledger}};
        context_t concurrent_context {basic_seed_certified_k, counting_allocator<std::byte> {ledger}};

        failures += expect_eq(exclusive_index
                                  .template insert<synchronization_t::exclusive_k>(
                                      {exclusive_values}, {keys}, exclusive_metric, {}, exclusive_context,
                                      basic_keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        failures += expect_eq(concurrent_index
                                  .insert({concurrent_values}, {keys}, concurrent_metric, {}, concurrent_context,
                                          basic_keep_diverse_t {basic_connectivity_k})
                                  .status,
                              status_t::success_k);
        // Same data, same seeds, single-threaded: the lock elision must be invisible.
        failures += expect_eq(exclusive_index.size(), concurrent_index.size());
        for (std::size_t node = 0; node != nodes_k; ++node) {
            node_id_t const node_id = static_cast<node_id_t>(node);
            failures += expect(exclusive_index.key_of(node_id).value == concurrent_index.key_of(node_id).value);
            failures += expect_eq(exclusive_index.degree_of(node_id).value, concurrent_index.degree_of(node_id).value);
        }

        // Three search arms over the settled graph must agree bit-for-bit.
        auto queries = fixture.make_values(4, basic_seed_certified_k ^ 0xFFull);
        for (auto const& query : queries) {
            context_t concurrent_arm {basic_seed_certified_k + 7, counting_allocator<std::byte> {ledger}};
            context_t immutable_arm {basic_seed_certified_k + 7, counting_allocator<std::byte> {ledger}};
            context_t exclusive_arm {basic_seed_certified_k + 7, counting_allocator<std::byte> {ledger}};
            auto arm_metric = fixture.make_metric();
            auto const concurrent_found = exclusive_index.search(query, arm_metric, concurrent_arm);
            auto const immutable_found = exclusive_index.template search<synchronization_t::immutable_k>(
                query, arm_metric, immutable_arm);
            auto const exclusive_found = exclusive_index.template search<synchronization_t::exclusive_k>(
                query, arm_metric, exclusive_arm);
            failures += expect_eq(immutable_found.status, concurrent_found.status);
            failures += expect_eq(exclusive_found.status, concurrent_found.status);
            failures += expect_eq(immutable_found.count, concurrent_found.count);
            failures += expect_eq(exclusive_found.count, concurrent_found.count);
            for (std::size_t position = 0; position != concurrent_found.count; ++position) {
                failures += expect_eq(immutable_arm.results[position].node_id,
                                      concurrent_arm.results[position].node_id);
                failures += expect_eq(exclusive_arm.results[position].node_id,
                                      concurrent_arm.results[position].node_id);
            }
        }

        // A parked insert violates any certification; the guard must say so.
        std::atomic<bool> entered {false}, released {false};
        auto one_more = fixture.make_values(1, basic_seed_certified_k ^ 0xAAull);
        key_t const one_more_key = fixture.make_key(nodes_k);
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> parked_failures {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) {
                basic_gated_metric<metric_t, typename index_t::distance_t> gated {fixture.make_metric(), &entered,
                                                                                  &released};
                context_t parked_context {basic_seed_certified_k + 9, counting_allocator<std::byte> {ledger}};
                auto const inserted = exclusive_index.insert({one_more.data(), 1}, {&one_more_key, 1}, gated, {},
                                                             parked_context,
                                                             basic_keep_diverse_t {basic_connectivity_k});
                if (inserted.status != status_t::success_k) parked_failures.fetch_add(1);
                return;
            }
            if (lane != 1) return;
            for (std::size_t spins = 0; !entered.load(std::memory_order_acquire); ++spins) {
                if (spins > (std::size_t {1} << 28)) {
                    parked_failures.fetch_add(1);
                    released.store(true, std::memory_order_release);
                    return;
                }
                std::this_thread::yield();
            }
            context_t checker_context {basic_seed_certified_k + 10, counting_allocator<std::byte> {ledger}};
            auto checker_metric = fixture.make_metric();
            if (exclusive_index
                    .template search<synchronization_t::immutable_k>(one_more[0], checker_metric, checker_context)
                    .status != status_t::contended_k)
                parked_failures.fetch_add(1);
            if (exclusive_index
                    .template search<synchronization_t::exclusive_k>(one_more[0], checker_metric, checker_context)
                    .status != status_t::contended_k)
                parked_failures.fetch_add(1);
            released.store(true, std::memory_order_release);
        });
        failures += expect_eq(parked_failures.load(), 0u);
        failures += expect_eq(exclusive_index.size(), nodes_k + 1);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The mid-flight allocation-failure battery: every fallible allocation inside a
 *      running operation keeps its stated consistency story - thinner nodes publish,
 *      refused staging saturates, cold scratch refuses all-or-nothing, and a starved
 *      load stays retryable. The warm-scratch trick isolates each site: a reused context
 *      already holds its buffers, so a frozen ledger budget starves exactly one thing. */
template <typename fixture_type_>
std::size_t test_basic_starved(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 96;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t warm_context {basic_seed_starved_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_starved_k);
        failures += expect(bool(index.reserve(nodes_k + 2)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, warm_context, basic_keep_closest_t {4});

        auto extras = fixture.make_values(2, basic_seed_starved_k ^ 0xAAull);
        std::array<key_t, 2> const extra_keys {fixture.make_key(nodes_k), fixture.make_key(nodes_k + 1)};
        auto query = fixture.make_values(1, basic_seed_starved_k ^ 0xBBull);

        // Warm a small round shape, then over-wire node 0 past its staging capacity, so
        // a later frozen-budget search can only proceed by refusing the top-up.
        context_t small_context {basic_seed_starved_k + 1, counting_allocator<std::byte> {ledger}};
        failures += expect_eq(index.search(query[0], metric, small_context, {8, 1}).status, status_t::success_k);
        for (std::size_t target = 1; target != nodes_k; ++target) [[maybe_unused]]
            auto const linked = index.link(0, static_cast<node_id_t>(target));

        // Cold scratch under a frozen budget: all-or-nothing refusals, nothing moves.
        context_t cold_context {basic_seed_starved_k + 2, counting_allocator<std::byte> {ledger}};
        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect_eq(index.search(query[0], metric, cold_context).status, status_t::bad_alloc_k);
        failures += expect_eq(index.insert({extras.data(), 1}, {extra_keys.data(), 1}, metric, {}, cold_context).status,
                              status_t::bad_alloc_k);
        failures += expect_eq(index.size(), nodes_k);
        failures += expect_eq(index.health(cold_context).status, status_t::bad_alloc_k);

        // Warm scratch under the same budget: the claimed id must still publish, only
        // thinner - the fresh row's growth is the one refused allocation.
        failures += expect_eq(
            index.insert({extras.data(), 1}, {extra_keys.data(), 1}, metric, {}, warm_context, basic_keep_closest_t {4})
                .status,
            status_t::success_k);
        failures += expect_eq(index.size(), nodes_k + 1);
        failures += expect_eq(index.degree_of(static_cast<node_id_t>(nodes_k)).value, 0u);

        // The saturated walk: the over-wired row cannot stage, the answer stays ranked.
        auto const saturated = index.search(query[0], metric, small_context, {8, 1});
        failures += expect_eq(saturated.status, status_t::success_k);
        failures += basic_expect_ascending(small_context);

        // Lifting the budget makes every refusal retryable.
        ledger.byte_budget.store(PTRDIFF_MAX);
        failures += expect_eq(index.health(warm_context).status, status_t::success_k);
        failures += expect_eq(
            index.insert({extras.data() + 1, 1}, {extra_keys.data() + 1, 1}, metric, {}, cold_context).status,
            status_t::success_k);
        failures += expect_eq(index.size(), nodes_k + 2);

        // A starved load fails mid-decode with the watermark unpublished, then retries.
        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);
        index_t restored = fixture.make_index(ledger);
        failures += expect(bool(restored.reserve(nodes_k + 2)));
        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect_eq(restored.load({payload}, codec), status_t::bad_alloc_k);
        failures += expect_eq(restored.size(), 0u);
        ledger.byte_budget.store(PTRDIFF_MAX);
        failures += expect_eq(restored.load({payload}, codec), status_t::success_k);
        failures += expect_eq(restored.size(), index.size());
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Manual wiring raced against inserts, searches & saves - the regression net for
 *      the census word and the duplicate-scan: overlapping saves answer @c contended_k,
 *      settled saves parse duplicate-free and reload cleanly. */
template <typename fixture_type_>
std::size_t test_basic_wiring_race(fixture_type_ const& fixture) {
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
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_wiring_race_k, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_wiring_race_k);
        failures += expect(bool(index.reserve(nodes_k + inserts_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});

        auto insert_values = fixture.make_values(inserts_k, basic_seed_wiring_race_k ^ 0x11ull);
        std::vector<key_t> insert_keys(inserts_k);
        for (std::size_t ordinal = 0; ordinal != inserts_k; ++ordinal)
            insert_keys[ordinal] = fixture.make_key(nodes_k + ordinal);
        auto queries = fixture.make_values(4, basic_seed_wiring_race_k ^ 0x22ull);

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(4))) return failures + expect(false);
        std::atomic<std::size_t> lane_failures {0};
        std::atomic<std::size_t> contended_saves {0};
        std::atomic<std::size_t> settled_saves {0};
        std::atomic<u32_t> unexpected_status {~u32_t {0}};
        std::atomic<u32_t> unexpected_lane {~u32_t {0}};
        auto record_unexpected = [&](std::size_t lane, status_t status) noexcept {
            lane_failures.fetch_add(1);
            unexpected_status.store(static_cast<u32_t>(status), std::memory_order_relaxed);
            unexpected_lane.store(static_cast<u32_t>(lane), std::memory_order_relaxed);
        };
        std::atomic<bool> wiring_done {false}, inserting_done {false};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) { // The saver: contends while the others run, settles after.
                std::vector<std::byte> buffer;
                auto attempt_save = [&]() noexcept {
                    auto const bytes = index.snapshot_bytes(codec);
                    if (failed(bytes.status)) return bytes.status;
                    buffer.resize(bytes.value);
                    return index.save({buffer}, codec);
                };
                // The race phase runs as long as the racers do - contention & post-sizing
                // growth are the expected answers, anything else is a defect.
                while (
                    !(wiring_done.load(std::memory_order_acquire) && inserting_done.load(std::memory_order_acquire))) {
                    status_t const saved = attempt_save();
                    if (saved == status_t::contended_k || saved == status_t::overflow_k) contended_saves.fetch_add(1);
                    else if (saved == status_t::success_k) settled_saves.fetch_add(1);
                    else return record_unexpected(0, saved);
                    std::this_thread::yield();
                }
                // The settle phase: with every racer done, a save must soon succeed.
                for (std::size_t attempt = 0; attempt != 1u << 20; ++attempt) {
                    status_t const saved = attempt_save();
                    if (saved == status_t::success_k) return void(settled_saves.fetch_add(1));
                    if (saved != status_t::contended_k && saved != status_t::overflow_k)
                        return record_unexpected(0, saved);
                }
                record_unexpected(0, status_t::contended_k); // The settle phase never settled.
                return;
            }
            if (lane == 1) { // The inserter.
                auto lane_metric = fixture.make_metric();
                context_t lane_context {basic_seed_wiring_race_k + 1, counting_allocator<std::byte> {ledger}};
                auto const inserted = index.insert({insert_values}, {insert_keys}, lane_metric, {}, lane_context,
                                                   basic_keep_diverse_t {basic_connectivity_k});
                if (inserted.status != status_t::success_k) record_unexpected(1, inserted.status);
                inserting_done.store(true, std::memory_order_release);
                return;
            }
            if (lane == 2) { // The manual wirer: every pair ends removed. It keeps wiring
                             // until the saver has provably contended with it at least once.
                for (std::size_t round = 0;
                     round != (1u << 20) &&
                     (round < wiring_rounds_k || contended_saves.load(std::memory_order_acquire) == 0);
                     ++round) {
                    node_id_t const from = static_cast<node_id_t>(round % 8);
                    node_id_t const to = static_cast<node_id_t>(8 + round % 8);
                    auto const linked = index.link(from, to);
                    if (failed(linked.status)) record_unexpected(2, linked.status);
                    // A racing reverse re-selection may prune the manual edge first -
                    // the documented advisory-wiring semantics - so absence is legal.
                    status_t const unlinked = index.unlink(from, to);
                    if (unlinked != status_t::success_k && unlinked != status_t::malformed_k)
                        record_unexpected(2, unlinked);
                }
                wiring_done.store(true, std::memory_order_release);
                return;
            }
            if (lane != 3) return; // The searcher.
            auto lane_metric = fixture.make_metric();
            context_t lane_context {basic_seed_wiring_race_k + 2, counting_allocator<std::byte> {ledger}};
            std::size_t searches = 0;
            while (!wiring_done.load(std::memory_order_acquire) || !inserting_done.load(std::memory_order_acquire) ||
                   searches < basic_reader_minimal_searches_k) {
                auto const found = index.search(queries[searches % queries.size()], lane_metric, lane_context);
                if (found.status != status_t::success_k) record_unexpected(3, found.status);
                std::size_t const visible_after = index.size();
                for (std::size_t position = 0; position != lane_context.results.size(); ++position)
                    if (static_cast<std::size_t>(lane_context.results[position].node_id) >= visible_after)
                        record_unexpected(3, status_t::internal_k);
                ++searches;
            }
        });
        if (lane_failures.load())
            fmt::print(stderr, "  wiring-race diag: lane {} saw status {}\n", unexpected_lane.load(),
                       status_to_string(static_cast<status_t>(unexpected_status.load())));
        failures += expect_eq(lane_failures.load(), 0u);
        failures += expect(contended_saves.load() > 0);
        failures += expect(settled_saves.load() > 0);
        for (std::size_t pair = 0; pair != 8; ++pair)
            failures += expect_eq(index.unlink(static_cast<node_id_t>(pair), static_cast<node_id_t>(8 + pair)),
                                  status_t::malformed_k);

        // The settled payload must parse duplicate-free and reload - the B2 audit.
        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);
        auto const header = basic_read_object<basic_snapshot_header_t>(payload, 0);
        auto const sections = basic_payload_sections<key_t, node_id_t>(header);
        i64_t row_begin = basic_read_object<i64_t>(payload, sections.neighbor_sums_offset);
        for (std::size_t node = 0; node != static_cast<std::size_t>(header.node_count); ++node) {
            i64_t const row_end = basic_read_object<i64_t>(payload,
                                                           sections.neighbor_sums_offset + (node + 1) * sizeof(i64_t));
            std::vector<node_id_t> row;
            for (i64_t link = row_begin; link != row_end; ++link)
                row.push_back(basic_read_object<node_id_t>(
                    payload, sections.neighbor_ids_offset + static_cast<std::size_t>(link) * sizeof(node_id_t)));
            std::sort(row.begin(), row.end());
            failures += expect(std::unique(row.begin(), row.end()) == row.end());
            row_begin = row_end;
        }
        index_t restored = fixture.make_index(ledger);
        failures += expect(bool(restored.reserve(nodes_k + inserts_k)));
        failures += expect_eq(restored.load({payload}, codec), status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Concurrent overflow: lanes submit batches totalling past capacity, and the
 *      claim protocol admits exactly the capacity - no lost room, no over-admission. */
template <typename fixture_type_>
std::size_t test_basic_overflow_race(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t lanes_k = 4;
    constexpr std::size_t per_lane_k = 32;
    constexpr std::size_t capacity_k = 2 * per_lane_k;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        failures += expect(bool(index.reserve(capacity_k)));
        std::vector<std::vector<typename index_t::value_t>> lane_values;
        std::vector<std::vector<key_t>> lane_keys;
        std::vector<context_t> lane_contexts;
        std::vector<decltype(fixture.make_metric())> lane_metrics;
        for (std::size_t lane = 0; lane != lanes_k; ++lane) {
            lane_values.push_back(fixture.make_values(per_lane_k, basic_seed_overflow_race_k + lane));
            std::vector<key_t> keys(per_lane_k);
            for (std::size_t ordinal = 0; ordinal != per_lane_k; ++ordinal)
                keys[ordinal] = fixture.make_key(lane * per_lane_k + ordinal);
            lane_keys.push_back(std::move(keys));
            lane_contexts.emplace_back(basic_seed_overflow_race_k + lane, counting_allocator<std::byte> {ledger});
            lane_metrics.push_back(fixture.make_metric());
        }
        std::vector<node_id_t> assigned_ids(lanes_k * per_lane_k);
        std::array<status_t, lanes_k> lane_statuses {};

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(lanes_k))) return failures + expect(false);
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= lanes_k) return;
            lane_statuses[lane] = index
                                      .insert({lane_values[lane]}, {lane_keys[lane]}, lane_metrics[lane],
                                              {assigned_ids.data() + lane * per_lane_k, per_lane_k},
                                              lane_contexts[lane], basic_keep_diverse_t {basic_connectivity_k})
                                      .status;
        });

        std::size_t admitted_lanes = 0;
        std::vector<node_id_t> admitted_ids;
        for (std::size_t lane = 0; lane != lanes_k; ++lane) {
            if (lane_statuses[lane] == status_t::success_k) {
                ++admitted_lanes;
                admitted_ids.insert(admitted_ids.end(), assigned_ids.data() + lane * per_lane_k,
                                    assigned_ids.data() + (lane + 1) * per_lane_k);
            }
            else failures += expect_eq(lane_statuses[lane], status_t::overflow_k);
        }
        failures += expect_eq(admitted_lanes, capacity_k / per_lane_k);
        failures += expect_eq(index.size(), capacity_k);
        std::sort(admitted_ids.begin(), admitted_ids.end());
        failures += expect(std::unique(admitted_ids.begin(), admitted_ids.end()) == admitted_ids.end());
        failures += expect(admitted_ids.front() == 0 &&
                           static_cast<std::size_t>(admitted_ids.back()) == capacity_k - 1);

        auto one_more = fixture.make_values(1, basic_seed_overflow_race_k ^ 0xFFull);
        key_t const one_more_key = fixture.make_key(1000);
        failures += expect_eq(
            index.insert({one_more.data(), 1}, {&one_more_key, 1}, lane_metrics[0], {}, lane_contexts[0]).status,
            status_t::overflow_k);
        failures += expect_eq(index.size(), capacity_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The reader's cursor: id-order iteration over the published prefix, snapshot
 *      discipline, edge-following, refresh widening, and a lane racing live mutators. */
template <typename fixture_type_>
std::size_t test_basic_cursor(fixture_type_ const& fixture) {
    using index_t = typename fixture_type_::index_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using context_t = typename index_t::traversal_context_t;
    using cursor_t = basic_ram_cursor<index_t, synchronization_t::concurrent_k, counting_allocator<std::byte>>;
    constexpr std::size_t nodes_k = 64;
    constexpr std::size_t racing_inserts_k = 8;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        context_t context {basic_seed_wiring_k + 1, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k + 1 + racing_inserts_k, basic_seed_wiring_k + 1);
        std::vector<typename index_t::value_t> extras;
        for (std::size_t extra = 0; extra != 1 + racing_inserts_k; ++extra)
            extras.push_back(std::move(values[nodes_k + extra]));
        values.resize(nodes_k);
        failures += expect(bool(index.reserve(nodes_k + 1 + racing_inserts_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context,
                                basic_keep_diverse_t {basic_connectivity_k});

        // A fresh cursor stands past the end; seek(0) enters the graph.
        cursor_t cursor {index, counting_allocator<std::byte> {ledger}};
        failures += expect(!cursor);
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_k);
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        std::size_t visited = 0;
        for (; cursor; ++cursor) {
            node_id_t const node_id = cursor.id();
            failures += expect(node_id < cursor.visible_count());
            failures += expect(cursor.key() == index.key_of(node_id).value);
            failures += expect(fixture.value_equals(cursor.value(), index.published_value_of(node_id)));
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
        std::array<key_t, 1> const late_key {fixture.make_key(nodes_k)};
        failures += expect_eq(index.insert({extras.data(), 1}, {late_key}, metric, {}, context).status,
                              status_t::success_k);
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_k);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_k)), status_t::malformed_k);
        cursor.refresh();
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_k + 1);
        failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_k)), status_t::success_k);

        // A cursor lane races an inserting & wiring lane; every visited id stays below
        // that pass's snapshot, and the final refresh converges on the full prefix.
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        std::atomic<std::size_t> lane_failures {0};
        std::atomic<bool> writer_done {false};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane == 0) {
                auto lane_metric = fixture.make_metric();
                context_t lane_context {basic_seed_wiring_k + 2, counting_allocator<std::byte> {ledger}};
                std::vector<key_t> lane_keys(racing_inserts_k);
                for (std::size_t ordinal = 0; ordinal != racing_inserts_k; ++ordinal)
                    lane_keys[ordinal] = fixture.make_key(nodes_k + 1 + ordinal);
                auto const inserted = index.insert({extras.data() + 1, racing_inserts_k}, {lane_keys}, lane_metric, {},
                                                   lane_context, basic_keep_diverse_t {basic_connectivity_k});
                if (inserted.status != status_t::success_k) lane_failures.fetch_add(1);
                for (std::size_t round = 0; round != 512; ++round) {
                    [[maybe_unused]] auto const linked = index.link(1, 2 + round % 8);
                    [[maybe_unused]] status_t const unlinked = index.unlink(1, 2 + round % 8);
                }
                writer_done.store(true, std::memory_order_release);
                return;
            }
            if (lane != 1) return;
            cursor_t racing_cursor {index, counting_allocator<std::byte> {ledger}};
            do {
                racing_cursor.refresh();
                node_id_t const snapshot = racing_cursor.visible_count();
                if (failed(racing_cursor.seek(0))) {
                    lane_failures.fetch_add(1);
                    return;
                }
                for (; racing_cursor; ++racing_cursor)
                    if (racing_cursor.id() >= snapshot) lane_failures.fetch_add(1);
            } while (!writer_done.load(std::memory_order_acquire));
        });
        failures += expect_eq(lane_failures.load(), 0u);
        cursor.refresh();
        failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), index.size());

        // The certified arm agrees with the concurrent one on a settled graph.
        basic_ram_cursor<index_t, synchronization_t::immutable_k, counting_allocator<std::byte>> settled_cursor {
            index, counting_allocator<std::byte> {ledger}};
        failures += expect_eq(settled_cursor.seek(0), status_t::success_k);
        std::size_t settled_visited = 0;
        for (; settled_cursor; ++settled_cursor) ++settled_visited;
        failures += expect_eq(settled_visited, index.size());
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The SafeGraphs adapter loop: re-frames a saved basic snapshot as a SafeGraphs
 *      monolith - the "SAFEGRPH" magic, a JSON header, and the untouched aligned buffers
 *      re-based to the data segment - then parses it back through the repo's own header
 *      reader and asserts every field survives the trip. */
template <typename fixture_type_>
std::size_t test_basic_safegraphs_adapter(fixture_type_ const& fixture, char const* key_dtype_name) {
    using index_t = typename fixture_type_::index_t;
    using key_t = typename index_t::key_t;
    using node_id_t = typename index_t::node_id_t;
    using context_t = typename index_t::traversal_context_t;
    constexpr std::size_t nodes_k = 24;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        index_t index = fixture.make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        context_t context {basic_seed_snapshot_k ^ 0x5AFEull, counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(nodes_k, basic_seed_snapshot_k ^ 0x5AFEull);
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<node_id_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context, basic_keep_diverse_t {4});
        // One manual edge de-uniformizes the degrees: the irregular-neighbors mapping.
        [[maybe_unused]] auto const bumped = index.link(0, static_cast<node_id_t>(nodes_k / 2));

        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);
        auto const header = basic_read_object<basic_snapshot_header_t>(payload, 0);
        failures += expect_eq(header.connectivity, 0u);
        auto const sections = basic_payload_sections<key_t, node_id_t>(header);

        // Column starts are data-segment-relative; the segment begins at the keys, and
        // every basic section offset is an alignment multiple, so re-basing is legal.
        std::size_t const data_origin = sections.keys_offset;
        std::size_t const node_count = static_cast<std::size_t>(header.node_count);
        char const* const node_id_dtype_name = sizeof(node_id_t) == 4 ? "u32" : "u64";
        std::string const json = fmt::format(
            R"({{"__metadata__":{{"nodes":{},"alignment":{}}},)" R"("key":{{"dtype":"{}","values_start":{},"values_length":{}}},)" R"("value":{{"dtype":"u8","values_start":{},"values_length":{},)" R"("offsets_dtype":"u64","offsets_start":{},"offsets_length":{}}},)" R"("neighbors":{{"dtype":"{}","values_start":{},"values_length":{},)" R"("offsets_dtype":"u64","offsets_start":{},"offsets_length":{}}}}})",
            node_count, header.alignment,                  //
            key_dtype_name, 0, node_count * sizeof(key_t), //
            sections.values_offset - data_origin, static_cast<std::size_t>(header.value_bytes),
            sections.value_sums_offset - data_origin, (node_count + 1) * sizeof(i64_t), //
            node_id_dtype_name, sections.neighbor_ids_offset - data_origin,
            static_cast<std::size_t>(header.neighbors_count) * sizeof(node_id_t),
            sections.neighbor_sums_offset - data_origin, (node_count + 1) * sizeof(i64_t));

        // The 16-byte prefix plus the space-padded header land the data segment on an
        // alignment boundary - the format's own padding rule.
        std::size_t const alignment = header.alignment;
        std::size_t const header_bytes = divide_round_up(16 + json.size(), alignment) * alignment - 16;
        std::vector<std::byte> monolith(16 + header_bytes + (payload.size() - data_origin));
        char const magic[8] = {'S', 'A', 'F', 'E', 'G', 'R', 'P', 'H'};
        for (std::size_t position = 0; position != 8; ++position)
            monolith[position] = static_cast<std::byte>(magic[position]);
        basic_write_object(monolith, 8, static_cast<u64_t>(header_bytes));
        for (std::size_t position = 0; position != header_bytes; ++position)
            monolith[16 + position] = static_cast<std::byte>(position < json.size() ? json[position] : ' ');
        for (std::size_t position = data_origin; position != payload.size(); ++position)
            monolith[16 + header_bytes + position - data_origin] = payload[position];

        std::FILE* const file = std::tmpfile();
        failures += expect(file != nullptr);
        if (file) {
            failures += expect_eq(std::fwrite(monolith.data(), 1, monolith.size(), file), monolith.size());
            std::rewind(file);
            auto const parsed = safegraphs_read_header(file);
            failures += expect(bool(parsed));
            if (parsed) {
                failures += expect_eq(parsed.value.nodes, node_count);
                failures += expect_eq(parsed.value.alignment, alignment);
                failures += expect_eq(parsed.value.data_offset, 16 + header_bytes);
                failures += expect(parsed.value.key.is_regular());
                failures += expect_eq(parsed.value.key.values_length, node_count * sizeof(key_t));
                failures += expect(!parsed.value.neighbors.is_regular());
                failures += expect_eq(parsed.value.neighbors.values_length,
                                      static_cast<std::size_t>(header.neighbors_count) * sizeof(node_id_t));
                failures += expect_eq(parsed.value.neighbors.offsets_length, (node_count + 1) * sizeof(i64_t));
                failures += expect(!parsed.value.value.is_regular());
                failures += expect_eq(parsed.value.value.values_length, static_cast<std::size_t>(header.value_bytes));
            }
            std::fclose(file);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Scenarios

/** @brief The whole shared battery over one fixture - units add their own extras after. */
template <typename fixture_type_>
std::size_t test_basic_scenarios(fixture_type_ const& fixture) {
    std::size_t failures = 0;
    failures += test_basic_guards(fixture);
    failures += test_basic_handmade(fixture);
    failures += test_basic_growth_and_recall(fixture);
    failures += test_basic_selectors(fixture);
    failures += test_basic_keys(fixture);
    failures += test_basic_wiring(fixture);
    failures += test_basic_allocation_budget(fixture);
    failures += test_basic_snapshot_roundtrip(fixture);
    failures += test_basic_snapshot_regions(fixture);
    failures += test_basic_snapshot_hostile(fixture);
    failures += test_basic_health(fixture);
    failures += test_basic_concurrent_build(fixture);
    failures += test_basic_concurrent_search(fixture);
    failures += test_basic_metric_contract(fixture);
    failures += test_basic_restarts(fixture);
    failures += test_basic_certified(fixture);
    failures += test_basic_starved(fixture);
    failures += test_basic_wiring_race(fixture);
    failures += test_basic_overflow_race(fixture);
    failures += test_basic_cursor(fixture);
    return failures;
}

} // namespace usearch
} // namespace unum
