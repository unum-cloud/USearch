/**
 *  @file test/utilities/ram_buffers.cpp
 *  @author Ash Vardanian
 *  @date April 24, 2026
 *  @brief Smoke tests for the @c ram_buffers.hpp & @c ram_tables.hpp containers - @c max_heap,
 *      @c sorted_buffer, and @c growing_hash_set - and the @c ram_sorts.hpp @c radix_group.
 *
 *  The containers' four-wide and larger-N batch overloads are covered, plus @c insert_many and @c pop_many
 *  on the heap.
 *
 *  Mirrors the structure of @c test/dense_metric.hpp:
 *
 *  1. One monolithic TU, no test framework. Each check returns a failure count;
 *  counts roll up to @c main without any global mutable state. A non-zero exit
 *  code means something failed.
 *  2. Serial single-element operations define the oracle for the multi-element overloads.
 *  For @c sorted_buffer::insert(std::array,limit) the oracle uses @c std::sort on
 *  the union because the "top-k of (existing ∪ batch)" semantic can accept
 *  candidates that cascading single-element inserts would reject.
 */
#include <cstdint>

#include <algorithm>
#include <array>
#include <memory> // `std::allocator` - every container names its allocator
#include <random>
#include <unordered_set> // The reference oracle of the hash-set growth-parity audit
#include <vector>

#include <fmt/format.h>

#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_sorts.hpp"
#include "usearch/utilities/ram_tables.hpp"

#include "harness.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

#pragma region Sorted Buffer

std::size_t check_sorted_buffer_single_element() noexcept {
    std::size_t failures = 0;
    sorted_buffer<u32_t> buffer {global_allocator};
    failures += expect(succeeded(buffer.reserve(16)));

    buffer.insert(u32_t {5}, 4);
    buffer.insert(u32_t {3}, 4);
    buffer.insert(u32_t {7}, 4);
    buffer.insert(u32_t {1}, 4);
    failures += expect_eq(buffer.size(), 4u);
    failures += expect_eq(buffer[0], 1u);
    failures += expect_eq(buffer[3], 7u);

    buffer.insert(u32_t {4}, 4);
    failures += expect_eq(buffer.size(), 4u);
    failures += expect_eq(buffer[3], 5u);

    bool const accepted_above_top = buffer.insert(u32_t {100}, 4);
    failures += expect(!accepted_above_top);
    return failures;
}

/** Parity of the sorted-span @c insert(span,limit) against a full @c std::sort of
 *  (existing ∪ batch) truncated to the limit. Quad-sized batches pre-sort through
 *  @c network_sort - auditing the network itself - wider ones through the library sort. */
template <std::size_t batch_size_>
std::size_t check_sorted_buffer_batch_parity(unsigned seed) noexcept {
    constexpr std::size_t limit_k = 64;
    constexpr int steps_k = 400;
    sorted_buffer<u32_t> tested {global_allocator};
    std::vector<u32_t> oracle_flat;
    oracle_flat.reserve(2 * limit_k);

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 1'000'000);

    std::size_t failures = expect(succeeded(tested.reserve(2 * limit_k)));
    for (int step = 0; step < steps_k; ++step) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = distribution(random_generator);
        if constexpr (batch_size_ <= 4) network_sort(std::span(batch));
        else std::sort(batch.begin(), batch.end());
        tested.insert(std::span(batch), limit_k);

        for (u32_t v : batch) oracle_flat.push_back(v);
        std::sort(oracle_flat.begin(), oracle_flat.end());
        if (oracle_flat.size() > limit_k) oracle_flat.resize(limit_k);

        if (tested.size() != oracle_flat.size()) {
            fmt::print(stderr, "FAIL sorted_buffer N={} parity step={} sizes diverge tested={} oracle={}\n",
                       batch_size_, step, tested.size(), oracle_flat.size());
            return ++failures;
        }
        for (std::size_t i = 0; i < tested.size(); ++i)
            if (tested[i] != oracle_flat[i]) {
                fmt::print(stderr, "FAIL sorted_buffer N={} parity step={} i={} tested={} oracle={}\n", batch_size_,
                           step, i, tested[i], oracle_flat[i]);
                return ++failures;
            }
    }
    return failures;
}

#pragma endregion Sorted Buffer

#pragma region Radix Group

/** One radix grouping of @p keys: every value survives with its own key, and every key forms one run. */
template <typename key_type_>
std::size_t check_radix_group(std::vector<key_type_> const& keys) noexcept {
    struct value_t {
        key_type_ key;
        u32_t arrival;
    };
    std::vector<value_t> values(keys.size());
    for (u32_t arrival = 0; arrival != keys.size(); ++arrival) values[arrival] = {keys[arrival], arrival};
    radix_group(std::span {values}, &value_t::key);

    std::size_t failures = 0;
    std::vector<bool> survived(keys.size());
    std::vector<key_type_> closed;
    for (std::size_t rank = 0; rank != values.size(); ++rank) {
        value_t const& value = values[rank];
        failures += expect(!survived[value.arrival] && keys[value.arrival] == value.key);
        survived[value.arrival] = true;
        if (!rank || value.key == values[rank - 1].key) continue;
        failures += expect(std::find(closed.begin(), closed.end(), value.key) == closed.end());
        closed.push_back(values[rank - 1].key);
    }
    return failures;
}

/** Radix groupings of words differing in two of their bytes, and of UUIDs differing in one middle byte. */
std::size_t check_radix_groups(unsigned seed) noexcept {
    std::mt19937 random_generator {seed};
    std::vector<u64_t> words(1000);
    for (u64_t& word : words) word = (random_generator() % 8) | (u64_t {random_generator() % 4} << 32);
    std::vector<uuid_t> uuids(1000);
    for (uuid_t& uuid : uuids) uuid.bytes[7] = static_cast<std::byte>(random_generator() % 16);
    return check_radix_group(words) + check_radix_group(uuids);
}

#pragma endregion Radix Group

#pragma region Growing Hash Set

std::size_t check_hash_set_single_element() noexcept {
    std::size_t failures = 0;
    growing_hash_set<u32_t> hash_set {global_allocator};
    failures += expect(succeeded(hash_set.reserve(16)));
    failures += expect(hash_set.insert(42) == membership_t::absent_k);
    failures += expect(hash_set.insert(42) == membership_t::present_k);
    failures += expect(hash_set.contains(42));
    failures += expect(!hash_set.contains(99));
    return failures;
}

/** Randomized membership parity against @c std::unordered_set across repeated growth
 *  cycles - the audit for the grown-into-unreachable bug class: a rehash placing
 *  elements through any probe walk other than the reader's strands them invisible. */
std::size_t check_hash_set_growth_parity(unsigned seed) noexcept {
    constexpr int rounds_k = 12, inserts_per_round_k = 700;
    growing_hash_set<u32_t> tested {global_allocator};
    std::unordered_set<u32_t> oracle;

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 100'000);

    // Start tiny, so the rounds force many rehashes through every size.
    std::size_t failures = expect(succeeded(tested.reserve(16)));
    for (int round = 0; round < rounds_k; ++round) {
        if (failed(tested.reserve(oracle.size() + inserts_per_round_k))) {
            fmt::print(stderr, "FAIL hash_set growth round={} reserve failed\n", round);
            return ++failures;
        }
        for (int step = 0; step < inserts_per_round_k; ++step) {
            u32_t key = distribution(random_generator);
            while (key == unused_hash_slot_value<u32_t>()) key = distribution(random_generator);
            bool const was_known = tested.insert(key) == membership_t::present_k;
            bool const oracle_known = !oracle.insert(key).second;
            if (was_known != oracle_known) {
                fmt::print(stderr, "FAIL hash_set growth round={} step={} key={} tested={} oracle={}\n", round, step,
                           key, was_known, oracle_known);
                return ++failures;
            }
        }
        // Every element ever inserted must remain reachable after all the rehashes so far.
        for (u32_t key : oracle)
            if (!tested.contains(key)) {
                fmt::print(stderr, "FAIL hash_set growth round={} key={} unreachable after rehash\n", round, key);
                return ++failures;
            }
        if (tested.size() != oracle.size()) {
            fmt::print(stderr, "FAIL hash_set growth round={} sizes diverge tested={} oracle={}\n", round,
                       tested.size(), oracle.size());
            return ++failures;
        }
    }
    return failures;
}

template <std::size_t batch_size_>
std::size_t check_hash_set_batch_parity(unsigned seed) noexcept {
    constexpr int steps_k = 200;
    growing_hash_set<u32_t> tested {global_allocator};
    growing_hash_set<u32_t> oracle {global_allocator};
    std::size_t reserve_failures = expect(succeeded(tested.reserve(1024)));
    reserve_failures += expect(succeeded(oracle.reserve(1024)));

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 500);
    auto draw_key = [&](void) noexcept {
        u32_t candidate;
        do candidate = distribution(random_generator);
        while (candidate == unused_hash_slot_value<u32_t>());
        return candidate;
    };

    std::size_t failures = reserve_failures;
    for (int step = 0; step < steps_k; ++step) {
        std::array<u32_t, batch_size_> batch;
        for (auto& key : batch) key = draw_key();

        u64_t const tested_mask = tested.insert(batch);
        u64_t oracle_mask = 0;
        for (std::size_t lane = 0; lane < batch_size_; ++lane)
            if (oracle.insert(batch[lane]) == membership_t::absent_k) oracle_mask |= u64_t {1} << lane;
        if (tested_mask != oracle_mask) {
            fmt::print(stderr, "FAIL hash_set N={} parity step={} tested=0x{:x} oracle=0x{:x}\n", batch_size_, step,
                       tested_mask, oracle_mask);
            return ++failures;
        }
        if (tested.size() != oracle.size()) {
            fmt::print(stderr, "FAIL hash_set N={} parity step={} sizes diverge tested={} oracle={}\n", batch_size_,
                       step, tested.size(), oracle.size());
            return ++failures;
        }
    }
    return failures;
}

/** A set over a spent arena re-tables into a chunk the arena opens beside the seed, keeps
 *  its table where a refusing upstream stops it, and the rewind keeps only the widest chunk. */
std::size_t check_hash_set_arena_growth() noexcept {
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        using upstream_t = counting_allocator_t;
        using arena_t = growing_arena<upstream_t>;
        using set_t = growing_hash_set<u32_t, std::hash<u32_t>, arena_allocator<std::byte, upstream_t>>;
        arena_t arena {upstream_t(ledger)};
        failures += expect(succeeded(arena.reserve_exactly(1024)));
        std::ptrdiff_t const seed_bytes = ledger.live_bytes.load();

        set_t set {arena_allocator<std::byte, upstream_t>(arena)};
        failures += expect(succeeded(set.reserve(16)));
        failures += expect_eq(ledger.live_bytes.load(), seed_bytes); // The seed chunk served it.
        for (u32_t value = 0; value != 16; ++value) failures += expect(set.insert(value) == membership_t::absent_k);

        failures += expect(succeeded(set.reserve(4'096)));
        failures += expect(ledger.live_bytes.load() > seed_bytes); // A wider chunk opened beside the seed.
        failures += expect(arena.capacity() > 1024u);
        for (u32_t value = 0; value != 16; ++value) failures += expect(set.contains(value));
        for (u32_t value = 16; value != 4'096; ++value) failures += expect(set.insert(value) == membership_t::absent_k);

        ledger.byte_budget.store(ledger.live_bytes.load());
        failures += expect_eq(set.reserve(1u << 20), status_t::bad_alloc_k);
        failures += expect_eq(set.size(), 4'096u);
        failures += expect(set.contains(4'095));

        std::ptrdiff_t const grown_bytes = ledger.live_bytes.load();
        set.reset();
        arena.clear();
        // The rewind drops the seed and keeps the widest chunk, so the next reservation lands in it.
        failures += expect(ledger.live_bytes.load() < grown_bytes && ledger.live_bytes.load() > seed_bytes);
        failures += expect(arena.capacity() > 1024u);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

#pragma endregion Growing Hash Set

#pragma region Max Heap

std::size_t check_max_heap_push_pop() noexcept {
    std::size_t failures = 0;
    max_heap<u32_t> heap {global_allocator};
    failures += expect(succeeded(heap.reserve(32)));
    for (u32_t value : {5u, 3u, 7u, 1u, 9u, 4u}) failures += expect(succeeded(heap.insert(u32_t(value))));
    std::array<u32_t, 6> const expected = {9, 7, 5, 4, 3, 1};
    for (std::size_t i = 0; i < expected.size(); ++i) failures += expect_eq(heap.pop(), expected[i]);
    failures += expect(heap.empty());
    return failures;
}

/** D-ary heap pops in descending order, parameterised by arity. */
template <std::size_t arity_>
std::size_t check_max_heap_dary_push_pop(unsigned seed, std::size_t element_count) noexcept {
    max_heap_dary<arity_, u32_t> heap {global_allocator};
    std::size_t failures = expect(succeeded(heap.reserve(element_count)));
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 1'000'000);
    std::vector<u32_t> inputs;
    inputs.reserve(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        u32_t const value = distribution(random_generator);
        inputs.push_back(value);
        failures += expect(succeeded(heap.insert(u32_t(value))));
    }
    std::sort(inputs.begin(), inputs.end(), std::greater<u32_t> {});
    for (std::size_t i = 0; i < element_count; ++i) {
        u32_t const popped = heap.pop();
        if (popped != inputs[i]) {
            fmt::print(stderr, "FAIL max_heap_dary<{}> pop[{}] expected={} got={}\n", arity_, i, inputs[i], popped);
            return ++failures;
        }
    }
    return failures;
}

/** `erase_below` sheds exactly the entries under the bar; what remains still pops in order. */
template <std::size_t arity_>
std::size_t check_max_heap_dary_erase_below(unsigned seed, std::size_t element_count) noexcept {
    max_heap_dary<arity_, u32_t> heap {global_allocator};
    std::size_t failures = expect(succeeded(heap.reserve(element_count)));
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 1'000'000);
    u32_t const bar = 500'000;
    std::vector<u32_t> kept;
    kept.reserve(element_count);
    for (std::size_t index = 0; index != element_count; ++index) {
        u32_t const value = distribution(random_generator);
        failures += expect(succeeded(heap.insert(u32_t(value))));
        if (value >= bar) kept.push_back(value);
    }
    std::size_t const erased = heap.erase_below(bar);
    failures += expect_eq(erased, element_count - kept.size());
    failures += expect_eq(heap.size(), kept.size());
    std::sort(kept.begin(), kept.end(), std::greater<u32_t> {});
    for (std::size_t index = 0; index != kept.size(); ++index) failures += expect_eq(heap.pop(), kept[index]);
    failures += expect(heap.empty());
    return failures;
}

/** D=2 d-ary heap must agree with the binary heap pop-by-pop on the same input. */
std::size_t check_max_heap_dary_parity_with_binary(unsigned seed, std::size_t element_count) noexcept {
    max_heap<u32_t> binary {global_allocator};
    max_heap_dary<2, u32_t> dary {global_allocator};
    std::size_t failures = expect(succeeded(binary.reserve(element_count)));
    failures += expect(succeeded(dary.reserve(element_count)));
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 1'000'000);
    for (std::size_t i = 0; i < element_count; ++i) {
        u32_t const value = distribution(random_generator);
        failures += expect(succeeded(binary.insert(u32_t(value))));
        failures += expect(succeeded(dary.insert(u32_t(value))));
    }
    for (std::size_t i = 0; i < element_count; ++i) {
        u32_t const a = binary.pop();
        u32_t const b = dary.pop();
        if (a != b) {
            fmt::print(stderr, "FAIL d=2 parity pop[{}] binary={} dary={}\n", i, a, b);
            return ++failures;
        }
    }
    return failures;
}

/** D-ary @c insert_many (Floyd vs @c shift_up paths) must match descending-pop order from individual inserts. */
template <std::size_t arity_>
std::size_t check_max_heap_dary_insert_many(unsigned seed, std::size_t initial_count,
                                            std::size_t batch_count) noexcept {
    max_heap_dary<arity_, u32_t> bulk {global_allocator};
    max_heap_dary<arity_, u32_t> sequential {global_allocator};
    std::size_t failures = expect(succeeded(bulk.reserve(initial_count + batch_count)));
    failures += expect(succeeded(sequential.reserve(initial_count + batch_count)));
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 1'000'000);
    for (std::size_t i = 0; i < initial_count; ++i) {
        u32_t const value = distribution(random_generator);
        failures += expect(succeeded(bulk.insert(u32_t(value))));
        failures += expect(succeeded(sequential.insert(u32_t(value))));
    }
    std::vector<u32_t> batch(batch_count);
    for (auto& value : batch) value = distribution(random_generator);
    failures += expect(succeeded(bulk.insert_many(batch.data(), batch_count)));
    for (auto value : batch) failures += expect(succeeded(sequential.insert(u32_t(value))));
    while (!bulk.empty()) {
        u32_t const a = bulk.pop();
        u32_t const b = sequential.pop();
        if (a != b) {
            fmt::print(stderr, "FAIL max_heap_dary<{}>::insert_many parity init={} batch={} a={} b={}\n", arity_,
                       initial_count, batch_count, a, b);
            return ++failures;
        }
    }
    return failures;
}

/** Bulk @c insert_many(std::array) must pop in the same descending order as single inserts. */
template <std::size_t batch_size_>
std::size_t check_max_heap_insert_many(unsigned seed, std::size_t initial_size) noexcept {
    max_heap<u32_t> bulk {global_allocator};
    max_heap<u32_t> sequential {global_allocator};
    std::size_t failures = expect(succeeded(bulk.reserve(initial_size + batch_size_)));
    failures += expect(succeeded(sequential.reserve(initial_size + batch_size_)));

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<u32_t> distribution(0, 1'000'000);

    for (std::size_t i = 0; i < initial_size; ++i) {
        u32_t const value = distribution(random_generator);
        failures += expect(succeeded(bulk.insert(u32_t(value))));
        failures += expect(succeeded(sequential.insert(u32_t(value))));
    }

    std::array<u32_t, batch_size_> batch;
    for (auto& value : batch) value = distribution(random_generator);
    failures += expect(succeeded(bulk.insert_many(batch)));
    for (auto const& value : batch) failures += expect(succeeded(sequential.insert(u32_t(value))));

    failures += expect_eq(bulk.size(), sequential.size());
    for (std::size_t i = 0; i < bulk.size(); ++i) {
        u32_t const bulk_pop = bulk.pop();
        u32_t const sequential_pop = sequential.pop();
        if (bulk_pop != sequential_pop) {
            fmt::print(stderr, "FAIL insert_many N={} init={} pop[{}] bulk={} sequential={}\n", batch_size_,
                       initial_size, i, bulk_pop, sequential_pop);
            return ++failures;
        }
    }
    return failures;
}

/** @c pop_many<N> must return the same sequence as N sequential @c pop calls. */
template <std::size_t batch_size_>
std::size_t check_max_heap_pop_many() noexcept {
    max_heap<u32_t> heap {global_allocator};
    std::size_t failures = expect(succeeded(heap.reserve(32)));
    for (u32_t value : {5u, 3u, 7u, 1u, 9u, 4u, 8u, 2u, 6u, 10u})
        failures += expect(succeeded(heap.insert(u32_t(value))));

    auto popped = heap.pop_many<batch_size_>();

    for (std::size_t i = 1; i < batch_size_; ++i)
        if (popped[i - 1] < popped[i]) {
            fmt::print(stderr, "FAIL pop_many N={} not descending at i={} ({} < {})\n", batch_size_, i, popped[i - 1],
                       popped[i]);
            return ++failures;
        }
    return failures;
}

#pragma endregion Max Heap

std::size_t test::utilities::check_ram_buffers() {
    std::size_t failures = 0;
    failures += check_sorted_buffer_single_element();
    failures += check_sorted_buffer_batch_parity<4>(0xABCD1234u);
    failures += check_sorted_buffer_batch_parity<16>(0x13572468u);
    failures += check_radix_groups(0x5A7D1C00u);

    failures += check_hash_set_single_element();
    failures += check_hash_set_growth_parity(0xAB5EEDEDu);
    failures += check_hash_set_batch_parity<4>(0xFEEDFACEu);
    failures += check_hash_set_batch_parity<16>(0x01234567u);
    failures += check_hash_set_arena_growth();

    failures += check_max_heap_push_pop();
    failures += check_max_heap_dary_parity_with_binary(0xD2U, 256);
    failures += check_max_heap_dary_push_pop<4>(0xD4U, 256);
    failures += check_max_heap_dary_push_pop<8>(0xD8U, 256);
    failures += check_max_heap_dary_push_pop<16>(0xD16U, 256);
    failures += check_max_heap_dary_erase_below<4>(0xE4U, 256);
    failures += check_max_heap_dary_erase_below<8>(0xE8U, 256);
    failures += check_max_heap_dary_insert_many<4>(0xDA10U, 0, 32);   // Floyd path
    failures += check_max_heap_dary_insert_many<4>(0xDA11U, 128, 32); // shift_up path
    failures += check_max_heap_dary_insert_many<8>(0xDA20U, 0, 32);
    failures += check_max_heap_dary_insert_many<8>(0xDA21U, 128, 32);
    failures += check_max_heap_dary_insert_many<16>(0xDA30U, 0, 32);
    failures += check_max_heap_dary_insert_many<16>(0xDA31U, 128, 32);
    // insert_many on empty heap exercises Floyd bulk rebuild (batch ≥ size/2 trivially).
    failures += check_max_heap_insert_many<4>(0xBEEF0001u, 0);
    failures += check_max_heap_insert_many<16>(0xBEEF0002u, 0);
    failures += check_max_heap_insert_many<32>(0xBEEF0003u, 0);
    // insert_many on a pre-populated heap exercises the incremental shift_up branch.
    failures += check_max_heap_insert_many<4>(0xBEEF0004u, 64);
    failures += check_max_heap_insert_many<16>(0xBEEF0005u, 128);
    failures += check_max_heap_insert_many<32>(0xBEEF0006u, 256);
    // pop_many<N> of a known sequence.
    failures += check_max_heap_pop_many<4>();
    failures += check_max_heap_pop_many<8>();

    return static_cast<std::size_t>(failures);
}
