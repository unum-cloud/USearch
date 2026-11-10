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
std::size_t check_multimap_uninitialized() noexcept {
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(!multimap.contains(42, all_visible_k));
    failures += expect_eq(multimap.count(42, all_visible_k), 0u);
    failures += expect_eq(multimap.capacity(), 0u);
    multimap.find_each(42, all_visible_k, [&](u32_t) noexcept { ++failures; });
    failures += expect(succeeded(insert_one(multimap, 42, 7)));
    failures += expect(multimap.capacity() > 0);
    failures += expect_eq(multimap.count(42, all_visible_k), 1u);
    return failures;
}

std::size_t check_multimap_single_key() noexcept {
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(8)));
    for (u32_t entry = 0; entry != 8; ++entry) failures += expect(succeeded(insert_one(multimap, 100 + entry, entry)));
    for (u32_t entry = 0; entry != 8; ++entry) {
        std::vector<u32_t> matches;
        multimap.find_each(100 + entry, all_visible_k, [&](u32_t node) noexcept { matches.push_back(node); });
        failures += expect(matches.size() == 1 && matches[0] == entry);
        failures += expect(multimap.contains(100 + entry, all_visible_k));
        failures += expect_eq(multimap.count(100 + entry, all_visible_k), 1u);
    }
    failures += expect(!multimap.contains(999, all_visible_k));
    return failures;
}

std::size_t check_multimap_duplicate_keys() noexcept {
    constexpr std::size_t multiplicity_k = 12;
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(multiplicity_k + 2)));
    for (u32_t duplicate = 0; duplicate != multiplicity_k; ++duplicate)
        failures += expect(succeeded(insert_one(multimap, 7, duplicate)));
    failures += expect(succeeded(insert_one(multimap, 8, 100)));

    failures += expect_eq(multimap.count(7, all_visible_k), multiplicity_k);
    std::vector<u32_t> observed;
    multimap.find_each(7, all_visible_k, [&](u32_t node) noexcept { observed.push_back(node); });
    std::sort(observed.begin(), observed.end());
    failures += expect_eq(observed.size(), multiplicity_k);
    for (std::size_t position = 0; position != observed.size(); ++position)
        failures += expect_eq(observed[position], static_cast<u32_t>(position));

    // Filed but at or above the watermark: invisible, as a node its writer has not published yet.
    failures += expect_eq(multimap.count(7, 5), 5u);
    failures += expect(!multimap.contains(8, 100) && multimap.contains(8, 101));
    multimap.find_each(7, 0, [&](u32_t) noexcept { ++failures; });
    return failures;
}

/** Every node a chunk holds fills it, and the next node grows another chunk rather than refusing,
 *  every entry staying findable across the growth. */
std::size_t check_multimap_growth_past_capacity() noexcept {
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(16)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) failures += expect(succeeded(insert_one(multimap, entry, entry)));

    failures += expect(succeeded(insert_one(multimap, 999, static_cast<u32_t>(capacity))));
    failures += expect(multimap.capacity() > capacity);
    failures += expect(multimap.contains(999, all_visible_k));
    for (u64_t key = 0; key != capacity; ++key) failures += expect(multimap.contains(key, all_visible_k));
    failures += expect_eq(multimap.count(static_cast<u64_t>(capacity) + 5, all_visible_k), 0u);
    return failures;
}

/** The retry primitive @c load leans on: a cleared table forgets every entry while keeping its
 *  chunks, so a rebuild can never inherit stale duplicates. */
std::size_t check_multimap_clear_retry() noexcept {
    constexpr std::size_t ceiling_k = 24;
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(ceiling_k)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) failures += expect(succeeded(insert_one(multimap, entry, entry)));

    multimap.clear();
    failures += expect_eq(multimap.capacity(), capacity);
    failures += expect(!multimap.contains(0, all_visible_k));

    // Every node is available again, and only the fresh keys surface.
    for (u32_t entry = 0; entry != capacity; ++entry)
        failures += expect(succeeded(insert_one(multimap, entry + 1000, entry)));
    failures += expect(!multimap.contains(1, all_visible_k));
    std::vector<u32_t> refreshed;
    multimap.find_each(1000, all_visible_k, [&](u32_t node) noexcept { refreshed.push_back(node); });
    failures += expect(refreshed.size() == 1 && refreshed[0] == 0);
    return failures;
}

/** Randomized parity against @c std::unordered_multimap across repeated growth
 *  cycles: every pair ever inserted must stay findable after every chunk it grows, with
 *  matching multiplicities - the grown-into-unreachable audit. */
std::size_t check_multimap_growth_parity(u64_t seed) noexcept {
    constexpr std::size_t growth_cycles_k = 4;
    constexpr std::size_t inserts_per_cycle_k = 200;
    constexpr unsigned keys_alphabet_k = 64;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t> multimap {counting_allocator_t {ledger}};
        std::unordered_multimap<u64_t, u32_t> oracle;
        splitmix64_t random_generator {seed};
        uniform_below<u32_t> key_distribution {keys_alphabet_k};

        for (std::size_t cycle = 0; cycle != growth_cycles_k; ++cycle) {
            failures += expect(succeeded(multimap.reserve((cycle + 1) * inserts_per_cycle_k)));
            for (std::size_t insertion = 0; insertion != inserts_per_cycle_k; ++insertion) {
                u64_t const key = key_distribution(random_generator);
                u32_t const value = static_cast<u32_t>(cycle * inserts_per_cycle_k + insertion);
                failures += expect(succeeded(insert_one(multimap, key, value)));
                oracle.emplace(key, value);
            }
            for (u64_t key = 0; key != keys_alphabet_k; ++key) {
                failures += expect_eq(multimap.count(key, all_visible_k), oracle.count(key));
                std::vector<u32_t> observed, wanted;
                multimap.find_each(key, all_visible_k, [&](u32_t node) noexcept { observed.push_back(node); });
                auto const [wanted_begin, wanted_end] = oracle.equal_range(key);
                for (auto pair = wanted_begin; pair != wanted_end; ++pair) wanted.push_back(pair->second);
                std::sort(observed.begin(), observed.end());
                std::sort(wanted.begin(), wanted.end());
                failures += expect(observed == wanted);
            }
        }
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** All keys hash into one bucket & the table fills to its ceiling, so this audits the
 *  "load factor at most one half keeps probe chains terminating" claim directly. */
std::size_t check_multimap_probe_termination() noexcept {
    constexpr std::size_t ceiling_k = 32;
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t, colliding_hash_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(ceiling_k)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) failures += expect(succeeded(insert_one(multimap, entry, entry)));

    for (u64_t key = 0; key != capacity; ++key) {
        failures += expect(multimap.contains(key, all_visible_k));
        failures += expect_eq(multimap.count(key, all_visible_k), 1u);
    }
    failures += expect_eq(multimap.count(static_cast<u64_t>(capacity) + 1, all_visible_k), 0u); // Absent probes end.
    return failures;
}

/** N lanes each file every node of the chunk they hold under a shared key alphabet - one writer
 *  per chunk, as an index files its own nodes: no lost inserts, no duplicates, exact counts. */
std::size_t check_multimap_concurrent_insert() noexcept {
    using multimap_t = chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t>;
    using nodes_schedule_t = typename multimap_t::nodes_schedule_t;
    constexpr std::size_t lanes_k = 4;
    constexpr unsigned keys_alphabet_k = 16;
    std::size_t const nodes = nodes_schedule_t::nodes_before(lanes_k);
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        multimap_t multimap {counting_allocator_t {ledger}};
        failures += expect(succeeded(multimap.reserve(nodes)));

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(lanes_k))) return failures + expect(false);
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= lanes_k) return;
            for (std::size_t node = nodes_schedule_t::nodes_before(lane);
                 node != nodes_schedule_t::nodes_before(lane + 1); ++node)
                multimap.insert(node % keys_alphabet_k, static_cast<u32_t>(node));
        });

        std::vector<u32_t> observed;
        for (u64_t key = 0; key != keys_alphabet_k; ++key) {
            std::size_t const before = observed.size();
            multimap.find_each(key, all_visible_k, [&](u32_t node) noexcept { observed.push_back(node); });
            for (std::size_t position = before; position != observed.size(); ++position)
                failures += expect_eq(observed[position] % keys_alphabet_k, key);
        }
        std::sort(observed.begin(), observed.end());
        failures += expect_eq(observed.size(), nodes);
        for (std::size_t position = 0; position != observed.size(); ++position)
            failures += expect_eq(observed[position], static_cast<u32_t>(position));
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** A reader lane probes while each writer files the nodes of its own chunk: every observed node
 *  must decode to the key it was filed under - so a slot being written never surfaces torn - and
 *  per-key counts must be monotone non-decreasing across successive reads. */
std::size_t check_multimap_reader_during_writers() noexcept {
    using multimap_t = chunked_multimap<u64_t, u32_t>;
    using nodes_schedule_t = typename multimap_t::nodes_schedule_t;
    constexpr std::size_t writers_k = 3;
    constexpr unsigned keys_alphabet_k = 8;
    constexpr std::size_t reader_minimal_sweeps_k = 16;
    std::size_t const nodes = nodes_schedule_t::nodes_before(writers_k);
    std::size_t failures = 0;
    multimap_t multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(nodes)));

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(writers_k + 1))) return failures + expect(false);
    std::size_t writers_done = 0, torn_observations = 0, shrinking_counts = 0;

    pool.for_threads([&](std::size_t lane) noexcept {
        if (lane >= writers_k + 1) return;
        if (lane != writers_k) {
            for (std::size_t node = nodes_schedule_t::nodes_before(lane);
                 node != nodes_schedule_t::nodes_before(lane + 1); ++node)
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

    failures += expect_eq(torn_observations, 0u);
    failures += expect_eq(shrinking_counts, 0u);
    std::size_t placed = 0;
    for (u64_t key = 0; key != keys_alphabet_k; ++key) placed += multimap.count(key, all_visible_k);
    failures += expect_eq(placed, nodes);
    return failures;
}

/** A starved growth must leave the published chunks fully intact & retryable. */
std::size_t check_multimap_starved_growth() noexcept {
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t> multimap {counting_allocator_t {ledger}};
        failures += expect(succeeded(multimap.reserve(16)));
        std::size_t const capacity = multimap.capacity();
        for (u32_t entry = 0; entry != 16; ++entry) failures += expect(succeeded(insert_one(multimap, entry, entry)));

        ledger.byte_budget = ledger.live_bytes;
        failures += expect_eq(multimap.reserve(capacity + 1), status_t::bad_alloc_k);
        failures += expect_eq(multimap.capacity(), capacity);
        for (u64_t key = 0; key != 16; ++key) failures += expect(multimap.contains(key, all_visible_k));

        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        failures += expect(succeeded(multimap.reserve(capacity + 1)));
        for (u64_t key = 0; key != 16; ++key) failures += expect(multimap.contains(key, all_visible_k));
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

std::size_t check_multimap_move() noexcept {
    constexpr std::size_t ceiling_k = 16;
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> source {global_allocator};
    failures += expect(succeeded(source.reserve(ceiling_k)));
    for (u32_t entry = 0; entry != ceiling_k; ++entry) failures += expect(succeeded(insert_one(source, entry, entry)));

    chunked_multimap<u64_t, u32_t> destination {std::move(source)};
    failures += expect_eq(source.capacity(), 0u);
    failures += expect(!source.contains(0, all_visible_k));
    for (u64_t key = 0; key != ceiling_k; ++key) failures += expect(destination.contains(key, all_visible_k));
    return failures;
}

/** A reader meeting control bytes whose slots no cover published yet stops there, answering empty -
 *  the chunk a concurrent cover is halfway through publishing. */
std::size_t check_multimap_uncovered_slots() noexcept {
    using multimap_t = chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t>;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        multimap_t multimap {counting_allocator_t {ledger}};
        std::ptrdiff_t const states_bytes = static_cast<std::ptrdiff_t>(multimap_t::slots_schedule_t::nodes_in(0));
        ledger.byte_budget = states_bytes;
        failures += expect_eq(multimap.reserve(1), status_t::bad_alloc_k);
        failures += expect_eq(ledger.live_bytes, states_bytes);
        failures += expect_eq(multimap.capacity(), 0u);
        failures += expect(!multimap.contains(42, all_visible_k));
        failures += expect_eq(multimap.count(42, all_visible_k), 0u);

        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        failures += expect(succeeded(multimap.reserve(1)));
        multimap.insert(42, 0);
        failures += expect_eq(multimap.count(42, all_visible_k), 1u);
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

using keyed_nodes_t = keyed_nodes<u64_t, u32_t, chunk_schedule<10, 20>, std::size_t {1} << 32, std::allocator<byte_t>,
                                  standard_atomic_ref>;

/** Keys file from an empty table with no reserve first, duplicates spread across nodes, and a
 *  reader sees only the nodes below the watermark it passes. */
std::size_t check_keyed_nodes_visible() noexcept {
    constexpr u32_t nodes_k = 3000; // Past the first chunk, so the nodes span two map chunks.
    constexpr u64_t keys_alphabet_k = 7;
    std::size_t failures = 0;
    keyed_nodes_t keyed {global_allocator};
    failures += expect_eq(keyed.memory_usage(), 0u);
    failures += expect(!keyed.contains(0, nodes_k));
    failures += expect(succeeded(keyed.cover(nodes_k)));
    for (u32_t node = 0; node != nodes_k; ++node) keyed.assign(node, node % keys_alphabet_k);
    keyed.publish({0, nodes_k});
    failures += expect_eq(keyed[nodes_k - 1], (nodes_k - 1) % keys_alphabet_k);
    keyed_nodes_t::keys_column_t const& column = keyed.column();
    failures += expect_eq(column[5], 5u % keys_alphabet_k);
    failures += expect(keyed.memory_usage() != 0);

    for (u64_t key = 0; key != keys_alphabet_k; ++key) {
        std::vector<u32_t> observed;
        keyed.find_each(key, nodes_k, [&](u32_t node) noexcept { observed.push_back(node); });
        std::sort(observed.begin(), observed.end());
        failures += expect_eq(observed.size(), keyed.count(key, nodes_k));
        for (std::size_t position = 0; position != observed.size(); ++position)
            failures += expect_eq(observed[position], static_cast<u32_t>(key + position * keys_alphabet_k));
    }

    // Published but above the watermark: invisible, as an entry the turn holder has not released
    // yet.
    constexpr u32_t visible_k = 3;
    failures += expect_eq(keyed.count(1, visible_k), 1u);
    failures += expect(!keyed.contains(4, visible_k));
    keyed.find_each(4, visible_k, [&](u32_t) noexcept { ++failures; });
    failures += expect(!keyed.contains(keys_alphabet_k, nodes_k));
    return failures;
}

/** A forgotten table answers empty, then files a fresh publish from node zero with none of the old
 *  keys. */
std::size_t check_keyed_nodes_forget() noexcept {
    constexpr u32_t nodes_k = 64;
    std::size_t failures = 0;
    keyed_nodes_t keyed {global_allocator};
    failures += expect(succeeded(keyed.cover(nodes_k)));
    for (u32_t node = 0; node != nodes_k; ++node) keyed.assign(node, 100 + node);
    keyed.publish({0, nodes_k});
    std::size_t const usage = keyed.memory_usage();

    keyed.forget();
    failures += expect_eq(keyed.memory_usage(), usage);
    for (u32_t node = 0; node != nodes_k; ++node) failures += expect(!keyed.contains(100 + node, nodes_k));

    for (u32_t node = 0; node != nodes_k / 2; ++node) keyed.assign(node, 500 + node);
    keyed.publish({0, nodes_k / 2});
    for (u32_t node = 0; node != nodes_k / 2; ++node) {
        failures += expect_eq(keyed.count(500 + node, nodes_k), 1u);
        failures += expect(!keyed.contains(100 + node, nodes_k));
    }
    failures += expect(!keyed.contains(100 + nodes_k - 1, nodes_k));
    return failures;
}

std::size_t check_ram_tables(settings_t const& settings) {
    std::size_t failures = 0;
    failures += check_multimap_uninitialized();
    failures += check_multimap_single_key();
    failures += check_multimap_duplicate_keys();
    failures += check_multimap_growth_past_capacity();
    failures += check_multimap_clear_retry();
    failures += check_multimap_growth_parity(stream_key(settings.seed, "multimap_growth_parity"));
    failures += check_multimap_probe_termination();
    failures += check_multimap_concurrent_insert();
    failures += check_multimap_reader_during_writers();
    failures += check_multimap_starved_growth();
    failures += check_multimap_move();
    failures += check_multimap_uncovered_slots();
    failures += check_keyed_nodes_visible();
    failures += check_keyed_nodes_forget();
    return failures;
}

} // namespace unum::usearch::test::utilities
