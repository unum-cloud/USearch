/**
 *  @file test/utilities/ram_tables.cpp
 *  @author Ash Vardanian
 *  @date August 26, 2026
 *  @brief Tests for the @c ram_tables.hpp concurrent multimap - the chunk growth, clear, and
 *      reader-vs-writer paths the index suites never isolate.
 *
 *  Serial checks define the oracle through @c std::unordered_multimap; the concurrent
 *  checks plant disjoint value ranges whose encoding lets a reader re-derive the key any
 *  observed value must be filed under, so a torn or misfiled slot is always detectable.
 */
#include <algorithm>     // `std::sort`
#include <atomic>        // `std::atomic`
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

/** One seat, then one entry - the whole life of a single insert. */
template <typename multimap_type_>
status_t insert_one(multimap_type_& multimap, u64_t key, u32_t value) noexcept {
    auto [seats, seat_status] = multimap.claim(1);
    if (failed(seat_status)) return seat_status;
    multimap.insert(seats, key, value);
    return status_t::success_k;
}

/** An unreserved map answers every lookup empty, then grows its first chunk on its first insert. */
std::size_t check_multimap_uninitialized() noexcept {
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect_eq(multimap.find(42).status, status_t::malformed_k);
    failures += expect(!multimap.contains(42));
    failures += expect_eq(multimap.count(42), 0u);
    failures += expect_eq(multimap.capacity(), 0u);
    multimap.find_each(42, [&](u32_t) noexcept { ++failures; });
    failures += expect(succeeded(insert_one(multimap, 42, 7)));
    failures += expect(multimap.capacity() > 0);
    failures += expect_eq(multimap.count(42), 1u);
    return failures;
}

std::size_t check_multimap_single_key() noexcept {
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(8)));
    for (u32_t entry = 0; entry != 8; ++entry) failures += expect(succeeded(insert_one(multimap, 100 + entry, entry)));
    for (u32_t entry = 0; entry != 8; ++entry) {
        auto const [match, status] = multimap.find(100 + entry);
        failures += expect(succeeded(status));
        failures += expect_eq(match, entry);
        failures += expect(multimap.contains(100 + entry));
        failures += expect_eq(multimap.count(100 + entry), 1u);
    }
    failures += expect_eq(multimap.find(999).status, status_t::malformed_k);
    failures += expect(!multimap.contains(999));
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

    failures += expect_eq(multimap.count(7), multiplicity_k);
    std::vector<u32_t> observed;
    multimap.find_each(7, [&](u32_t value) noexcept { observed.push_back(value); });
    std::sort(observed.begin(), observed.end());
    failures += expect_eq(observed.size(), multiplicity_k);
    for (std::size_t position = 0; position != observed.size(); ++position)
        failures += expect_eq(observed[position], static_cast<u32_t>(position));

    auto const [one_of, one_of_status] = multimap.find(7);
    failures += expect(succeeded(one_of_status));
    failures += expect(one_of < multiplicity_k);
    return failures;
}

/** A full map grows another chunk rather than refusing, and every entry stays findable across the growth. */
std::size_t check_multimap_growth_past_capacity() noexcept {
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(16)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) failures += expect(succeeded(insert_one(multimap, entry, entry)));

    failures += expect(succeeded(insert_one(multimap, 999, 999)));
    failures += expect(multimap.capacity() > capacity);
    failures += expect(multimap.contains(999));
    for (u64_t key = 0; key != capacity; ++key) failures += expect(multimap.contains(key));
    failures += expect_eq(multimap.count(static_cast<u64_t>(capacity) + 5), 0u);
    return failures;
}

/** The retry primitive @c load leans on: a cleared table forgets every entry & seat
 *  while keeping its chunks, so a rebuild can never inherit stale duplicates. */
std::size_t check_multimap_clear_retry() noexcept {
    constexpr std::size_t ceiling_k = 24;
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(ceiling_k)));
    std::size_t const capacity = multimap.capacity();
    for (u32_t entry = 0; entry != capacity; ++entry) failures += expect(succeeded(insert_one(multimap, entry, entry)));

    multimap.clear();
    failures += expect_eq(multimap.capacity(), capacity);
    failures += expect(!multimap.contains(0));

    // The full ceiling is available again, and only the fresh values surface.
    for (u32_t entry = 0; entry != capacity; ++entry)
        failures += expect(succeeded(insert_one(multimap, entry, entry + 1000)));
    auto const [refreshed, refreshed_status] = multimap.find(0);
    failures += expect(succeeded(refreshed_status));
    failures += expect_eq(refreshed, 1000u);
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
                failures += expect_eq(multimap.count(key), oracle.count(key));
                std::vector<u32_t> observed, wanted;
                multimap.find_each(key, [&](u32_t value) noexcept { observed.push_back(value); });
                auto const [wanted_begin, wanted_end] = oracle.equal_range(key);
                for (auto pair = wanted_begin; pair != wanted_end; ++pair) wanted.push_back(pair->second);
                std::sort(observed.begin(), observed.end());
                std::sort(wanted.begin(), wanted.end());
                failures += expect(observed == wanted);
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
        failures += expect(multimap.contains(key));
        failures += expect_eq(multimap.count(key), 1u);
    }
    failures += expect_eq(multimap.count(static_cast<u64_t>(capacity) + 1), 0u); // Absent probes terminate too.
    return failures;
}

/** N lanes insert disjoint value ranges under a shared key alphabet - the CAS-once
 *  claim protocol's only real audit: no lost inserts, no duplicates, exact counts. */
std::size_t check_multimap_concurrent_insert() noexcept {
    constexpr std::size_t lanes_k = 4;
    constexpr std::size_t inserts_per_lane_k = 2048;
    constexpr unsigned keys_alphabet_k = 16;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        chunked_multimap<u64_t, u32_t, std::hash<u64_t>, counting_allocator_t> multimap {counting_allocator_t {ledger}};
        failures += expect(succeeded(multimap.reserve(16)));

        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(lanes_k))) return failures + expect(false);
        std::atomic<std::size_t> failed_inserts {0};
        pool.for_threads([&](std::size_t lane) noexcept {
            if (lane >= lanes_k) return;
            for (std::size_t insertion = 0; insertion != inserts_per_lane_k; ++insertion) {
                u32_t const value = static_cast<u32_t>(lane * inserts_per_lane_k + insertion);
                if (failed(insert_one(multimap, value % keys_alphabet_k, value))) failed_inserts.fetch_add(1);
            }
        });

        failures += expect_eq(failed_inserts.load(), 0u);
        std::vector<u32_t> observed;
        for (u64_t key = 0; key != keys_alphabet_k; ++key) {
            std::size_t const before = observed.size();
            multimap.find_each(key, [&](u32_t value) noexcept { observed.push_back(value); });
            for (std::size_t position = before; position != observed.size(); ++position)
                failures += expect_eq(observed[position] % keys_alphabet_k, key);
        }
        std::sort(observed.begin(), observed.end());
        failures += expect_eq(observed.size(), lanes_k * inserts_per_lane_k);
        for (std::size_t position = 0; position != observed.size(); ++position)
            failures += expect_eq(observed[position], static_cast<u32_t>(position));
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** A reader lane probes while writers insert: every observed value must decode to
 *  the key it was filed under - so a building slot can never surface torn - and per-key
 *  counts must be monotone non-decreasing across successive reads. */
std::size_t check_multimap_reader_during_writers() noexcept {
    constexpr std::size_t writers_k = 3;
    constexpr std::size_t inserts_per_writer_k = 2048;
    constexpr unsigned keys_alphabet_k = 8;
    constexpr std::size_t reader_minimal_sweeps_k = 16;
    std::size_t failures = 0;
    chunked_multimap<u64_t, u32_t> multimap {global_allocator};
    failures += expect(succeeded(multimap.reserve(16)));

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(writers_k + 1))) return failures + expect(false);
    std::atomic<std::size_t> writers_done {0};
    std::atomic<std::size_t> torn_observations {0};
    std::atomic<std::size_t> shrinking_counts {0};

    pool.for_threads([&](std::size_t lane) noexcept {
        if (lane >= writers_k + 1) return;
        if (lane != writers_k) {
            for (std::size_t insertion = 0; insertion != inserts_per_writer_k; ++insertion) {
                u32_t const value = static_cast<u32_t>(lane * inserts_per_writer_k + insertion);
                if (failed(insert_one(multimap, value % keys_alphabet_k, value))) torn_observations.fetch_add(1);
            }
            writers_done.fetch_add(1);
            return;
        }
        std::size_t previous_counts[keys_alphabet_k] = {};
        std::size_t sweeps = 0;
        while (writers_done.load(std::memory_order_acquire) != writers_k || sweeps < reader_minimal_sweeps_k) {
            for (u64_t key = 0; key != keys_alphabet_k; ++key) {
                std::size_t observed_count = 0;
                multimap.find_each(key, [&](u32_t value) noexcept {
                    ++observed_count;
                    bool const planted = value < writers_k * inserts_per_writer_k && value % keys_alphabet_k == key;
                    if (!planted) torn_observations.fetch_add(1);
                });
                if (observed_count < previous_counts[key]) shrinking_counts.fetch_add(1);
                previous_counts[key] = observed_count;
            }
            ++sweeps;
        }
    });

    failures += expect_eq(torn_observations.load(), 0u);
    failures += expect_eq(shrinking_counts.load(), 0u);
    std::size_t landed = 0;
    for (u64_t key = 0; key != keys_alphabet_k; ++key) landed += multimap.count(key);
    failures += expect_eq(landed, writers_k * inserts_per_writer_k);
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

        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect_eq(multimap.reserve(capacity + 1), status_t::bad_alloc_k);
        failures += expect_eq(multimap.capacity(), capacity);
        for (u64_t key = 0; key != 16; ++key) failures += expect(multimap.contains(key));

        ledger.byte_budget.store(PTRDIFF_MAX);
        failures += expect(succeeded(multimap.reserve(capacity + 1)));
        for (u64_t key = 0; key != 16; ++key) failures += expect(multimap.contains(key));
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
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
    failures += expect(!source.contains(0));
    for (u64_t key = 0; key != ceiling_k; ++key) failures += expect(destination.contains(key));
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
    return failures;
}

} // namespace unum::usearch::test::utilities
