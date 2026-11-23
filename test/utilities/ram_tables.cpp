/**
 *  @file test/utilities/ram_tables.cpp
 *  @author Ash Vardanian
 *  @date August 26, 2026
 *  @brief Tests for the @c ram_tables.hpp concurrent multimap and the keyed nodes over it - the
 *      chunk growth, clear, watermark filtering and reader-vs-writer paths the index suites never
 *      isolate.
 *
 *  Serial checks define the oracle through @c std::unordered_multimap; the concurrent checks give
 *  every writer the nodes of its own chunk, whose encoding lets a reader re-derive the key any
 *  observed node must be filed under, so a torn or misfiled slot is always detectable.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <algorithm>     // `std::sort`
#include <limits>        // `std::numeric_limits` - a watermark past every node
#include <memory>        // `std::allocator` - every map names its allocator
#include <unordered_map> // The reference oracle of the growth-parity audit
#include <vector>        // `std::vector`

#include <fmt/format.h>

#include <forkunion.hpp>

#include "usearch/utilities/ram_tables.hpp"

#include "harness.hpp"

namespace unum::usearch::test::utilities {

/** Sends every key to one bucket, so probe chains reach their worst case. */
struct colliding_hash_t {
    std::size_t operator()(u64_t) const noexcept { return 0; }
};

/** A watermark past every node, so a lookup sees whatever was filed. */
constexpr u32_t all_visible_k = (std::numeric_limits<u32_t>::max)();

/** Covers @p node, then files it under @p key - the whole life of a single insert. */
template <typename multimap_type_>
status_t insert_one(multimap_type_& multimap, u64_t key, u32_t node) noexcept {
    if (status_t const reserved = multimap.reserve(std::size_t {node} + 1); failed(reserved)) return reserved;
    multimap.insert(key, node);
    return status_t::success_k;
}

/** An unreserved map answers every lookup empty, then grows its first chunk on its first insert. */
void test_multimap_uninitialized() noexcept {
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    expect_true(!multimap.contains(42, all_visible_k));
    expect_equal(multimap.count(42, all_visible_k), 0u);
    expect_equal(multimap.capacity(), 0u);
    multimap.find_each(42, all_visible_k, [&](u32_t) noexcept { expect_true(false); });
    expect_true(succeeded(insert_one(multimap, 42, 7)));
    expect_true(multimap.capacity() > 0);
    expect_equal(multimap.count(42, all_visible_k), 1u);
}

void test_multimap_single_key() noexcept {
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    expect_true(succeeded(multimap.reserve(8)));
    for (u32_t entry = 0; entry != 8; ++entry) expect_true(succeeded(insert_one(multimap, 100 + entry, entry)));
    for (u32_t entry = 0; entry != 8; ++entry) {
        std::vector<u32_t> matches;
        multimap.find_each(100 + entry, all_visible_k, [&](u32_t node) noexcept { matches.push_back(node); });
        expect_true(matches.size() == 1 && matches[0] == entry);
        expect_true(multimap.contains(100 + entry, all_visible_k));
        expect_equal(multimap.count(100 + entry, all_visible_k), 1u);
    }
    expect_true(!multimap.contains(999, all_visible_k));
}

void test_multimap_duplicate_keys() noexcept {
    constexpr std::size_t multiplicity_k = 12;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    expect_true(succeeded(multimap.reserve(multiplicity_k + 2)));
    for (u32_t duplicate = 0; duplicate != multiplicity_k; ++duplicate)
        expect_true(succeeded(insert_one(multimap, 7, duplicate)));
    expect_true(succeeded(insert_one(multimap, 8, 100)));

    expect_equal(multimap.count(7, all_visible_k), multiplicity_k);
    std::vector<u32_t> observed;
    multimap.find_each(7, all_visible_k, [&](u32_t node) noexcept { observed.push_back(node); });
    std::sort(observed.begin(), observed.end());
    expect_equal(observed.size(), multiplicity_k);
    for (std::size_t position = 0; position != observed.size(); ++position)
        expect_equal(observed[position], static_cast<u32_t>(position));

    // Filed but at or above the watermark: invisible, as a node its writer has not published yet.
    expect_equal(multimap.count(7, 5), 5u);
    expect_true(!multimap.contains(8, 100) && multimap.contains(8, 101));
    multimap.find_each(7, 0, [&](u32_t) noexcept { expect_true(false); });
}

/** Every node a chunk holds fills it, and the next node grows another chunk rather than refusing,
 *  every entry staying findable across the growth. */
void test_multimap_growth_past_capacity() noexcept {
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    expect_true(succeeded(multimap.reserve(16)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) expect_true(succeeded(insert_one(multimap, entry, entry)));

    expect_true(succeeded(insert_one(multimap, 999, static_cast<u32_t>(capacity))));
    expect_true(multimap.capacity() > capacity);
    expect_true(multimap.contains(999, all_visible_k));
    for (u64_t key = 0; key != capacity; ++key) expect_true(multimap.contains(key, all_visible_k));
    expect_equal(multimap.count(static_cast<u64_t>(capacity) + 5, all_visible_k), 0u);
}

/** The retry primitive @c load leans on: a cleared table forgets every entry while keeping its
 *  chunks, so a rebuild can never inherit stale duplicates. */
void test_multimap_clear_retry() noexcept {
    constexpr std::size_t ceiling_k = 24;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    expect_true(succeeded(multimap.reserve(ceiling_k)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) expect_true(succeeded(insert_one(multimap, entry, entry)));

    multimap.clear();
    expect_equal(multimap.capacity(), capacity);
    expect_true(!multimap.contains(0, all_visible_k));

    // Every node is available again, and only the fresh keys surface.
    for (u32_t entry = 0; entry != capacity; ++entry) expect_true(succeeded(insert_one(multimap, entry + 1000, entry)));
    expect_true(!multimap.contains(1, all_visible_k));
    std::vector<u32_t> refreshed;
    multimap.find_each(1000, all_visible_k, [&](u32_t node) noexcept { refreshed.push_back(node); });
    expect_true(refreshed.size() == 1 && refreshed[0] == 0);
}

/** Randomized parity against @c std::unordered_multimap across repeated growth
 *  cycles: every pair ever inserted must stay findable after every chunk it grows, with
 *  matching multiplicities - the grown-into-unreachable audit. */
void test_multimap_growth_parity(u64_t seed) noexcept {
    constexpr std::size_t growth_cycles_k = 4;
    constexpr std::size_t inserts_per_cycle_k = 200;
    constexpr unsigned keys_alphabet_k = 64;
    allocation_ledger_t ledger;
    {
        chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t> multimap {counting_allocator_t {ledger}};
        std::unordered_multimap<u64_t, u32_t> oracle;
        split_mix_generator_t random_generator {seed};
        uniform_below<u32_t> key_distribution {keys_alphabet_k};

        for (std::size_t cycle = 0; cycle != growth_cycles_k; ++cycle) {
            expect_true(succeeded(multimap.reserve((cycle + 1) * inserts_per_cycle_k)));
            for (std::size_t insertion = 0; insertion != inserts_per_cycle_k; ++insertion) {
                u64_t const key = key_distribution(random_generator);
                u32_t const value = static_cast<u32_t>(cycle * inserts_per_cycle_k + insertion);
                expect_true(succeeded(insert_one(multimap, key, value)));
                oracle.emplace(key, value);
            }
            for (u64_t key = 0; key != keys_alphabet_k; ++key) {
                expect_equal(multimap.count(key, all_visible_k), oracle.count(key));
                std::vector<u32_t> observed, wanted;
                multimap.find_each(key, all_visible_k, [&](u32_t node) noexcept { observed.push_back(node); });
                auto const [wanted_begin, wanted_end] = oracle.equal_range(key);
                for (auto pair = wanted_begin; pair != wanted_end; ++pair) wanted.push_back(pair->second);
                std::sort(observed.begin(), observed.end());
                std::sort(wanted.begin(), wanted.end());
                expect_true(observed == wanted);
            }
        }
    }
    expect_equal(ledger.live_bytes, 0);
}

/** All keys hash into one bucket & the table fills to its ceiling, so this audits the
 *  "load factor at most one half keeps probe chains terminating" claim directly. */
void test_multimap_probe_termination() noexcept {
    constexpr std::size_t ceiling_k = 32;
    chunked_multimap<u64_t, u32_t, colliding_hash_t> multimap {global_allocator};
    expect_true(succeeded(multimap.reserve(ceiling_k)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) expect_true(succeeded(insert_one(multimap, entry, entry)));

    for (u64_t key = 0; key != capacity; ++key) {
        expect_true(multimap.contains(key, all_visible_k));
        expect_equal(multimap.count(key, all_visible_k), 1u);
    }
    expect_equal(multimap.count(static_cast<u64_t>(capacity) + 1, all_visible_k), 0u); // Absent probes end.
}

/** N threads file interleaved nodes of the same chunks under a shared key alphabet, several writers
 *  per chunk as a load's runs are: no lost inserts, no duplicates, and exact counts per key. */
void test_multimap_concurrent_insert() noexcept {
    using multimap_t = chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t>;
    using nodes_schedule_t = typename multimap_t::nodes_schedule_t;
    constexpr std::size_t threads_k = 4;
    constexpr unsigned keys_alphabet_k = 16;
    std::size_t const nodes = nodes_schedule_t::nodes_before(threads_k);
    allocation_ledger_t ledger;
    {
        multimap_t multimap {counting_allocator_t {ledger}};
        expect_true(succeeded(multimap.reserve(nodes)));

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(threads_k))) return void(expect_true(false));
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread >= threads_k) return;
            for (std::size_t node = thread; node < nodes; node += threads_k)
                multimap.insert(node % keys_alphabet_k, static_cast<u32_t>(node));
        });

        std::vector<u32_t> observed;
        for (u64_t key = 0; key != keys_alphabet_k; ++key) {
            std::size_t const before = observed.size();
            multimap.find_each(key, all_visible_k, [&](u32_t node) noexcept { observed.push_back(node); });
            for (std::size_t position = before; position != observed.size(); ++position)
                expect_equal(observed[position] % keys_alphabet_k, key);
        }
        std::sort(observed.begin(), observed.end());
        expect_equal(observed.size(), nodes);
        for (std::size_t position = 0; position != observed.size(); ++position)
            expect_equal(observed[position], static_cast<u32_t>(position));
    }
    expect_equal(ledger.live_bytes, 0);
}

/** A reader thread probes while each writer files the nodes of its own chunk: every observed node
 *  must decode to the key it was filed under - so a slot being written never surfaces torn - and
 *  per-key counts must be monotone non-decreasing across successive reads. */
void test_multimap_reader_during_writers() noexcept {
    using multimap_t = chunked_multimap<u64_t, u32_t>;
    using nodes_schedule_t = typename multimap_t::nodes_schedule_t;
    constexpr std::size_t writers_k = 3;
    constexpr unsigned keys_alphabet_k = 8;
    constexpr std::size_t reader_minimal_sweeps_k = 16;
    std::size_t const nodes = nodes_schedule_t::nodes_before(writers_k);
    multimap_t multimap {global_allocator};
    expect_true(succeeded(multimap.reserve(nodes)));

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(writers_k + 1))) return void(expect_true(false));
    std::size_t writers_done = 0, torn_observations = 0, shrinking_counts = 0;

    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= writers_k + 1) return;
        if (thread != writers_k) {
            for (std::size_t node = nodes_schedule_t::nodes_before(thread);
                 node != nodes_schedule_t::nodes_before(thread + 1); ++node)
                multimap.insert(node % keys_alphabet_k, static_cast<u32_t>(node));
            fu::standard_atomic_ref<std::size_t>(writers_done).store_add(1);
            return;
        }
        std::size_t previous_counts[keys_alphabet_k] = {};
        std::size_t sweeps = 0;
        while (fu::standard_atomic_ref<std::size_t>(writers_done).load(std::memory_order_acquire) != writers_k ||
               sweeps < reader_minimal_sweeps_k) {
            for (u64_t key = 0; key != keys_alphabet_k; ++key) {
                std::size_t observed_count = 0;
                multimap.find_each(key, all_visible_k, [&](u32_t node) noexcept {
                    ++observed_count;
                    bool const planted = node < nodes && node % keys_alphabet_k == key;
                    if (!planted) fu::standard_atomic_ref<std::size_t>(torn_observations).store_add(1);
                });
                if (observed_count < previous_counts[key]) ++shrinking_counts;
                previous_counts[key] = observed_count;
            }
            ++sweeps;
        }
    });

    expect_equal(torn_observations, 0u);
    expect_equal(shrinking_counts, 0u);
    std::size_t placed = 0;
    for (u64_t key = 0; key != keys_alphabet_k; ++key) placed += multimap.count(key, all_visible_k);
    expect_equal(placed, nodes);
}

/** A starved growth must leave the published chunks fully intact & retryable. */
void test_multimap_starved_growth() noexcept {
    allocation_ledger_t ledger;
    {
        chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t> multimap {counting_allocator_t {ledger}};
        expect_true(succeeded(multimap.reserve(16)));
        std::size_t const capacity = multimap.capacity();
        for (u32_t entry = 0; entry != 16; ++entry) expect_true(succeeded(insert_one(multimap, entry, entry)));

        ledger.byte_budget = ledger.live_bytes;
        expect_equal(multimap.reserve(capacity + 1), status_t::bad_alloc_k);
        expect_equal(multimap.capacity(), capacity);
        for (u64_t key = 0; key != 16; ++key) expect_true(multimap.contains(key, all_visible_k));

        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        expect_true(succeeded(multimap.reserve(capacity + 1)));
        for (u64_t key = 0; key != 16; ++key) expect_true(multimap.contains(key, all_visible_k));
    }
    expect_equal(ledger.live_bytes, 0);
}

void test_multimap_move() noexcept {
    constexpr std::size_t ceiling_k = 16;
    chunked_multimap<u64_t, u32_t> source {global_allocator};
    expect_true(succeeded(source.reserve(ceiling_k)));
    for (u32_t entry = 0; entry != ceiling_k; ++entry) expect_true(succeeded(insert_one(source, entry, entry)));

    chunked_multimap<u64_t, u32_t> destination {std::move(source)};
    expect_equal(source.capacity(), 0u);
    expect_true(!source.contains(0, all_visible_k));
    for (u64_t key = 0; key != ceiling_k; ++key) expect_true(destination.contains(key, all_visible_k));
}

/** A reader meeting control bytes whose slots no cover published yet stops there, answering empty -
 *  the chunk a concurrent cover is halfway through publishing. */
void test_multimap_uncovered_slots() noexcept {
    using multimap_t = chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t>;
    allocation_ledger_t ledger;
    {
        multimap_t multimap {counting_allocator_t {ledger}};
        std::ptrdiff_t const states_bytes = static_cast<std::ptrdiff_t>(multimap_t::slots_schedule_t::nodes_in(0));
        ledger.byte_budget = states_bytes;
        expect_equal(multimap.reserve(1), status_t::bad_alloc_k);
        expect_equal(ledger.live_bytes, states_bytes);
        expect_equal(multimap.capacity(), 0u);
        expect_true(!multimap.contains(42, all_visible_k));
        expect_equal(multimap.count(42, all_visible_k), 0u);

        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        expect_true(succeeded(multimap.reserve(1)));
        multimap.insert(42, 0);
        expect_equal(multimap.count(42, all_visible_k), 1u);
    }
    expect_equal(ledger.live_bytes, 0);
}

using keyed_nodes_t = keyed_nodes<u64_t, u32_t, chunk_schedule<10, 20>, std::size_t {1} << 32, std::allocator<byte_t>,
                                  standard_atomic_ref>;

/** Keys file from an empty table with no reserve first, duplicates spread across nodes, and a
 *  reader sees only the nodes below the watermark it passes. */
void test_keyed_nodes_visible() noexcept {
    constexpr u32_t nodes_k = 3000; // Past the first chunk, so the nodes span two map chunks.
    constexpr u64_t keys_alphabet_k = 7;
    keyed_nodes_t keyed {global_allocator};
    expect_equal(keyed.memory_usage(), 0u);
    expect_true(!keyed.contains(0, nodes_k));
    expect_true(succeeded(keyed.cover(nodes_k)));
    for (u32_t node = 0; node != nodes_k; ++node) keyed.assign(node, node % keys_alphabet_k);
    keyed.publish({0, nodes_k});
    expect_equal(keyed[nodes_k - 1], (nodes_k - 1) % keys_alphabet_k);
    keyed_nodes_t::keys_column_t const& column = keyed.column();
    expect_equal(column[5], 5u % keys_alphabet_k);
    expect_true(keyed.memory_usage() != 0);

    for (u64_t key = 0; key != keys_alphabet_k; ++key) {
        std::vector<u32_t> observed;
        keyed.find_each(key, nodes_k, [&](u32_t node) noexcept { observed.push_back(node); });
        std::sort(observed.begin(), observed.end());
        expect_equal(observed.size(), keyed.count(key, nodes_k));
        for (std::size_t position = 0; position != observed.size(); ++position)
            expect_equal(observed[position], static_cast<u32_t>(key + position * keys_alphabet_k));
    }

    // Published but above the watermark: invisible, as an entry the turn holder has not released
    // yet.
    constexpr u32_t visible_k = 3;
    expect_equal(keyed.count(1, visible_k), 1u);
    expect_true(!keyed.contains(4, visible_k));
    keyed.find_each(4, visible_k, [&](u32_t) noexcept { expect_true(false); });
    expect_true(!keyed.contains(keys_alphabet_k, nodes_k));
}

/** A forgotten table answers empty, then files a fresh publish from node zero with none of the old
 *  keys. */
void test_keyed_nodes_forget() noexcept {
    constexpr u32_t nodes_k = 64;
    keyed_nodes_t keyed {global_allocator};
    expect_true(succeeded(keyed.cover(nodes_k)));
    for (u32_t node = 0; node != nodes_k; ++node) keyed.assign(node, 100 + node);
    keyed.publish({0, nodes_k});
    std::size_t const usage = keyed.memory_usage();

    keyed.forget();
    expect_equal(keyed.memory_usage(), usage);
    for (u32_t node = 0; node != nodes_k; ++node) expect_true(!keyed.contains(100 + node, nodes_k));

    for (u32_t node = 0; node != nodes_k / 2; ++node) keyed.assign(node, 500 + node);
    keyed.publish({0, nodes_k / 2});
    for (u32_t node = 0; node != nodes_k / 2; ++node) {
        expect_equal(keyed.count(500 + node, nodes_k), 1u);
        expect_true(!keyed.contains(100 + node, nodes_k));
    }
    expect_true(!keyed.contains(100 + nodes_k - 1, nodes_k));

    // Forgetting from a node inside a chunk keeps the earlier nodes that chunk held.
    keyed.forget_from(nodes_k / 4);
    for (u32_t node = 0; node != nodes_k / 2; ++node)
        expect_equal(keyed.count(500 + node, nodes_k), node < nodes_k / 4 ? 1u : 0u);
}

void run_ram_tables_cases() {
    run_test_case("test_multimap_uninitialized", test_multimap_uninitialized);
    run_test_case("test_multimap_single_key", test_multimap_single_key);
    run_test_case("test_multimap_duplicate_keys", test_multimap_duplicate_keys);
    run_test_case("test_multimap_growth_past_capacity", test_multimap_growth_past_capacity);
    run_test_case("test_multimap_clear_retry", test_multimap_clear_retry);
    run_test_case("test_multimap_growth_parity", test_multimap_growth_parity,
                  stream_key(test_run.settings.seed, "multimap_growth_parity"));
    run_test_case("test_multimap_probe_termination", test_multimap_probe_termination);
    run_test_case("test_multimap_concurrent_insert", test_multimap_concurrent_insert);
    run_test_case("test_multimap_reader_during_writers", test_multimap_reader_during_writers);
    run_test_case("test_multimap_starved_growth", test_multimap_starved_growth);
    run_test_case("test_multimap_move", test_multimap_move);
    run_test_case("test_multimap_uncovered_slots", test_multimap_uncovered_slots);
    run_test_case("test_keyed_nodes_visible", test_keyed_nodes_visible);
    run_test_case("test_keyed_nodes_forget", test_keyed_nodes_forget);
}

} // namespace unum::usearch::test::utilities
