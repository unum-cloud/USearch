/**
 *  @file test/utilities/ram_buffers.cpp
 *  @author Ash Vardanian
 *  @date April 24, 2026
 *  @brief Smoke tests for the @c ram_buffers.hpp & @c ram_tables.hpp containers - @c max_heap,
 *      @c sorted_buffer, and @c growing_hash_set - and the @c ram_sorts.hpp @c radix_group.
 *
 *  The containers' four-wide and larger-N batch overloads are covered, plus @c insert_many and @c pop_many
 *  on the heap. Eight threads race the first-failure latch of @c types.hpp.
 *
 *  Mirrors the structure of @c test/dense_metric.hpp:
 *
 *  1. One monolithic TU over the harness's run: each case is a @c run_test_case of its own, and its
 *     checks count into the run, which exits non-zero when any failed.
 *  2. Serial single-element operations define the oracle for the multi-element overloads.
 *  For @c sorted_buffer::insert(std::array,limit) the oracle uses @c std::sort on
 *  the union because the "top-k of (existing ∪ batch)" semantic can accept
 *  candidates that cascading single-element inserts would reject.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include <cstdint>

#include <algorithm>
#include <array>
#include <memory>        // `std::allocator` - every container names its allocator
#include <unordered_set> // The reference oracle of the hash-set growth-parity audit
#include <vector>

#include <fmt/format.h>

#include <forkunion.hpp> // `fu::flat_pool_t`, `fu::standard_atomic_ref` - the threads racing the latch

#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_sorts.hpp"
#include "usearch/utilities/ram_tables.hpp"

#include "harness.hpp"

namespace unum::usearch::test::utilities {

#pragma region Sorted Buffer

void test_sorted_buffer_single_element() noexcept {
    util::sorted_buffer<util::u32_t> buffer {util::global_allocator};
    expect_true(succeeded(buffer.reserve(16)));

    buffer.insert(util::u32_t {5}, 4);
    buffer.insert(util::u32_t {3}, 4);
    buffer.insert(util::u32_t {7}, 4);
    buffer.insert(util::u32_t {1}, 4);
    expect_equal(buffer.size(), 4u);
    expect_equal(buffer[0], 1u);
    expect_equal(buffer[3], 7u);

    buffer.insert(util::u32_t {4}, 4);
    expect_equal(buffer.size(), 4u);
    expect_equal(buffer[3], 5u);

    bool const accepted_above_top = buffer.insert(util::u32_t {100}, 4);
    expect_true(!accepted_above_top);
}

/** Parity of the sorted-span @c insert(span,limit) against a full @c std::sort of
 *  (existing ∪ batch) truncated to the limit. Quad-sized batches pre-sort through
 *  @c network_sort - auditing the network itself - wider ones through the library sort. */
template <std::size_t batch_size_>
void test_sorted_buffer_batch_parity(util::u64_t seed) noexcept {
    constexpr std::size_t limit_k = 64;
    constexpr int steps_k = 400;
    util::sorted_buffer<util::u32_t> tested {util::global_allocator};
    std::vector<util::u32_t> oracle_flat;
    oracle_flat.reserve(2 * limit_k);

    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {1'000'000 + 1};

    expect_true(succeeded(tested.reserve(2 * limit_k)));
    for (int step = 0; step < steps_k; ++step) {
        std::array<util::u32_t, batch_size_> batch;
        for (auto& value : batch) value = distribution(random_generator);
        if constexpr (batch_size_ <= 4) util::network_sort(std::span(batch));
        else std::sort(batch.begin(), batch.end());
        tested.insert(std::span(batch), limit_k);

        for (util::u32_t v : batch) oracle_flat.push_back(v);
        std::sort(oracle_flat.begin(), oracle_flat.end());
        if (oracle_flat.size() > limit_k) oracle_flat.resize(limit_k);

        if (tested.size() != oracle_flat.size()) {
            fmt::print(stderr, "FAIL sorted_buffer N={} parity step={} sizes diverge tested={} oracle={}\n",
                       batch_size_, step, tested.size(), oracle_flat.size());
            return void(expect_true(false));
        }
        for (std::size_t i = 0; i < tested.size(); ++i)
            if (tested[i] != oracle_flat[i]) {
                fmt::print(stderr, "FAIL sorted_buffer N={} parity step={} i={} tested={} oracle={}\n", batch_size_,
                           step, i, tested[i], oracle_flat[i]);
                return void(expect_true(false));
            }
    }
}

#pragma endregion Sorted Buffer

#pragma region Radix Group

/** One radix grouping of @p keys: every value survives with its own key, and every key forms one run. */
template <typename key_type_>
void check_radix_group(std::vector<key_type_> const& keys) noexcept {
    struct value_t {
        key_type_ key;
        util::u32_t arrival;
    };
    std::vector<value_t> values(keys.size());
    for (util::u32_t arrival = 0; arrival != keys.size(); ++arrival) values[arrival] = {keys[arrival], arrival};
    util::radix_group(std::span {values}, &value_t::key);

    std::vector<bool> survived(keys.size());
    std::vector<key_type_> closed;
    for (std::size_t rank = 0; rank != values.size(); ++rank) {
        value_t const& value = values[rank];
        expect_true(!survived[value.arrival] && keys[value.arrival] == value.key);
        survived[value.arrival] = true;
        if (!rank || value.key == values[rank - 1].key) continue;
        expect_true(std::find(closed.begin(), closed.end(), value.key) == closed.end());
        closed.push_back(values[rank - 1].key);
    }
}

/** Radix groupings of words differing in two of their bytes, and of UUIDs differing in one middle byte. */
void test_radix_groups(util::u64_t seed) noexcept {
    split_mix_generator_t random_generator {seed};
    std::vector<util::u64_t> words(1000);
    for (util::u64_t& word : words) word = (random_generator() % 8) | (util::u64_t {random_generator() % 4} << 32);
    std::vector<util::uuid128_t> uuids(1000);
    for (util::uuid128_t& uuid : uuids) uuid.bytes[7] = static_cast<std::byte>(random_generator() % 16);
    check_radix_group(words);
    check_radix_group(uuids);
}

#pragma endregion Radix Group

#pragma region Growing Hash Set

void test_hash_set_single_element() noexcept {
    util::growing_hash_set<util::u32_t> hash_set {util::global_allocator};
    expect_true(succeeded(hash_set.reserve(16)));
    expect_true(hash_set.insert(42) == util::membership_t::absent_k);
    expect_true(hash_set.insert(42) == util::membership_t::present_k);
    expect_true(hash_set.contains(42));
    expect_true(!hash_set.contains(99));
}

/** Randomized membership parity against @c std::unordered_set across repeated growth
 *  cycles - the audit for the grown-into-unreachable bug class: a rehash placing
 *  elements through any probe walk other than the reader's strands them invisible. */
void test_hash_set_growth_parity(util::u64_t seed) noexcept {
    constexpr int rounds_k = 12, inserts_per_round_k = 700;
    util::growing_hash_set<util::u32_t> tested {util::global_allocator};
    std::unordered_set<util::u32_t> oracle;

    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {100'000 + 1};

    // Start tiny, so the rounds force many rehashes through every size.
    expect_true(succeeded(tested.reserve(16)));
    for (int round = 0; round < rounds_k; ++round) {
        if (failed(tested.reserve(oracle.size() + inserts_per_round_k))) {
            fmt::print(stderr, "FAIL hash_set growth round={} reserve failed\n", round);
            return void(expect_true(false));
        }
        for (int step = 0; step < inserts_per_round_k; ++step) {
            util::u32_t key = distribution(random_generator);
            while (key == util::unused_hash_slot_value<util::u32_t>()) key = distribution(random_generator);
            bool const was_known = tested.insert(key) == util::membership_t::present_k;
            bool const oracle_known = !oracle.insert(key).second;
            if (was_known != oracle_known) {
                fmt::print(stderr, "FAIL hash_set growth round={} step={} key={} tested={} oracle={}\n", round, step,
                           key, was_known, oracle_known);
                return void(expect_true(false));
            }
        }
        // Every element ever inserted must remain reachable after all the rehashes so far.
        for (util::u32_t key : oracle)
            if (!tested.contains(key)) {
                fmt::print(stderr, "FAIL hash_set growth round={} key={} unreachable after rehash\n", round, key);
                return void(expect_true(false));
            }
        if (tested.size() != oracle.size()) {
            fmt::print(stderr, "FAIL hash_set growth round={} sizes diverge tested={} oracle={}\n", round,
                       tested.size(), oracle.size());
            return void(expect_true(false));
        }
    }
}

template <std::size_t batch_size_>
void test_hash_set_batch_parity(util::u64_t seed) noexcept {
    constexpr int steps_k = 200;
    util::growing_hash_set<util::u32_t> tested {util::global_allocator};
    util::growing_hash_set<util::u32_t> oracle {util::global_allocator};
    expect_true(succeeded(tested.reserve(1024)));
    expect_true(succeeded(oracle.reserve(1024)));

    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {500 + 1};
    auto draw_key = [&](void) noexcept {
        util::u32_t candidate;
        do candidate = distribution(random_generator);
        while (candidate == util::unused_hash_slot_value<util::u32_t>());
        return candidate;
    };

    for (int step = 0; step < steps_k; ++step) {
        std::array<util::u32_t, batch_size_> batch;
        for (auto& key : batch) key = draw_key();

        util::u64_t const tested_mask = tested.insert(batch);
        util::u64_t oracle_mask = 0;
        for (std::size_t lane = 0; lane < batch_size_; ++lane)
            if (oracle.insert(batch[lane]) == util::membership_t::absent_k) oracle_mask |= util::u64_t {1} << lane;
        if (tested_mask != oracle_mask) {
            fmt::print(stderr, "FAIL hash_set N={} parity step={} tested=0x{:x} oracle=0x{:x}\n", batch_size_, step,
                       tested_mask, oracle_mask);
            return void(expect_true(false));
        }
        if (tested.size() != oracle.size()) {
            fmt::print(stderr, "FAIL hash_set N={} parity step={} sizes diverge tested={} oracle={}\n", batch_size_,
                       step, tested.size(), oracle.size());
            return void(expect_true(false));
        }
    }
}

/** A set over a spent arena re-tables into a chunk the arena opens beside the seed, keeps
 *  its table where a refusing upstream stops it, and the rewind keeps only the widest chunk. */
void test_hash_set_arena_growth() noexcept {
    allocation_ledger_t ledger;
    {
        using upstream_t = counting_allocator_t;
        using arena_t = util::growing_arena<upstream_t>;
        using set_t =
            util::growing_hash_set<util::u32_t, std::hash<util::u32_t>, util::arena_allocator<std::byte, upstream_t>>;
        arena_t arena {upstream_t(ledger)};
        expect_true(succeeded(arena.reserve_exactly(1024)));
        std::ptrdiff_t const seed_bytes = ledger.live_bytes;

        set_t set {util::arena_allocator<std::byte, upstream_t>(arena)};
        expect_true(succeeded(set.reserve(16)));
        expect_equal(ledger.live_bytes, seed_bytes); // The seed chunk served it.
        for (util::u32_t value = 0; value != 16; ++value)
            expect_true(set.insert(value) == util::membership_t::absent_k);

        expect_true(succeeded(set.reserve(4'096)));
        expect_true(ledger.live_bytes > seed_bytes); // A wider chunk opened beside the seed.
        expect_true(arena.capacity() > 1024u);
        for (util::u32_t value = 0; value != 16; ++value) expect_true(set.contains(value));
        for (util::u32_t value = 16; value != 4'096; ++value)
            expect_true(set.insert(value) == util::membership_t::absent_k);

        ledger.byte_budget = ledger.live_bytes;
        expect_equal(set.reserve(1u << 20), util::status_code_t::out_of_memory_k);
        expect_equal(set.size(), 4'096u);
        expect_true(set.contains(4'095));

        std::ptrdiff_t const grown_bytes = ledger.live_bytes;
        set.reset();
        arena.clear();
        // The rewind drops the seed and keeps the widest chunk, so the next reservation goes in it.
        expect_true(ledger.live_bytes < grown_bytes && ledger.live_bytes > seed_bytes);
        expect_true(arena.capacity() > 1024u);
    }
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Growing Hash Set

#pragma region Max Heap

void test_max_heap_push_pop() noexcept {
    util::max_heap<util::u32_t> heap {util::global_allocator};
    expect_true(succeeded(heap.reserve(32)));
    for (util::u32_t value : {5u, 3u, 7u, 1u, 9u, 4u}) expect_true(succeeded(heap.insert(util::u32_t(value))));
    std::array<util::u32_t, 6> const expected = {9, 7, 5, 4, 3, 1};
    for (std::size_t i = 0; i < expected.size(); ++i) expect_equal(heap.pop(), expected[i]);
    expect_true(heap.empty());
}

/** D-ary heap pops in descending order, parameterised by arity. */
template <std::size_t arity_>
void test_max_heap_dary_push_pop(util::u64_t seed, std::size_t element_count) noexcept {
    util::max_heap_dary<arity_, util::u32_t> heap {util::global_allocator};
    expect_true(succeeded(heap.reserve(element_count)));
    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {1'000'000 + 1};
    std::vector<util::u32_t> inputs;
    inputs.reserve(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        util::u32_t const value = distribution(random_generator);
        inputs.push_back(value);
        expect_true(succeeded(heap.insert(util::u32_t(value))));
    }
    std::sort(inputs.begin(), inputs.end(), std::greater<util::u32_t> {});
    for (std::size_t i = 0; i < element_count; ++i) {
        util::u32_t const popped = heap.pop();
        if (popped != inputs[i]) {
            fmt::print(stderr, "FAIL max_heap_dary<{}> pop[{}] expected={} got={}\n", arity_, i, inputs[i], popped);
            return void(expect_true(false));
        }
    }
}

/** `erase_below` sheds exactly the entries under the bar; what remains still pops in order. */
template <std::size_t arity_>
void test_max_heap_dary_erase_below(util::u64_t seed, std::size_t element_count) noexcept {
    util::max_heap_dary<arity_, util::u32_t> heap {util::global_allocator};
    expect_true(succeeded(heap.reserve(element_count)));
    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {1'000'000 + 1};
    util::u32_t const bar = 500'000;
    std::vector<util::u32_t> kept;
    kept.reserve(element_count);
    for (std::size_t index = 0; index != element_count; ++index) {
        util::u32_t const value = distribution(random_generator);
        expect_true(succeeded(heap.insert(util::u32_t(value))));
        if (value >= bar) kept.push_back(value);
    }
    std::size_t const erased = heap.erase_below(bar);
    expect_equal(erased, element_count - kept.size());
    expect_equal(heap.size(), kept.size());
    std::sort(kept.begin(), kept.end(), std::greater<util::u32_t> {});
    for (std::size_t index = 0; index != kept.size(); ++index) expect_equal(heap.pop(), kept[index]);
    expect_true(heap.empty());
}

/** D=2 d-ary heap must agree with the binary heap pop-by-pop on the same input. */
void test_max_heap_dary_parity_with_binary(util::u64_t seed, std::size_t element_count) noexcept {
    util::max_heap<util::u32_t> binary {util::global_allocator};
    util::max_heap_dary<2, util::u32_t> dary {util::global_allocator};
    expect_true(succeeded(binary.reserve(element_count)));
    expect_true(succeeded(dary.reserve(element_count)));
    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {1'000'000 + 1};
    for (std::size_t i = 0; i < element_count; ++i) {
        util::u32_t const value = distribution(random_generator);
        expect_true(succeeded(binary.insert(util::u32_t(value))));
        expect_true(succeeded(dary.insert(util::u32_t(value))));
    }
    for (std::size_t i = 0; i < element_count; ++i) {
        util::u32_t const a = binary.pop();
        util::u32_t const b = dary.pop();
        if (a != b) {
            fmt::print(stderr, "FAIL d=2 parity pop[{}] binary={} dary={}\n", i, a, b);
            return void(expect_true(false));
        }
    }
}

/** D-ary @c insert_many (Floyd vs @c shift_up paths) must match descending-pop order from individual inserts. */
template <std::size_t arity_>
void test_max_heap_dary_insert_many(util::u64_t seed, std::size_t initial_count, std::size_t batch_count) noexcept {
    util::max_heap_dary<arity_, util::u32_t> bulk {util::global_allocator};
    util::max_heap_dary<arity_, util::u32_t> sequential {util::global_allocator};
    expect_true(succeeded(bulk.reserve(initial_count + batch_count)));
    expect_true(succeeded(sequential.reserve(initial_count + batch_count)));
    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {1'000'000 + 1};
    for (std::size_t i = 0; i < initial_count; ++i) {
        util::u32_t const value = distribution(random_generator);
        expect_true(succeeded(bulk.insert(util::u32_t(value))));
        expect_true(succeeded(sequential.insert(util::u32_t(value))));
    }
    std::vector<util::u32_t> batch(batch_count);
    for (auto& value : batch) value = distribution(random_generator);
    expect_true(succeeded(bulk.insert_many(batch.data(), batch_count)));
    for (auto value : batch) expect_true(succeeded(sequential.insert(util::u32_t(value))));
    while (!bulk.empty()) {
        util::u32_t const a = bulk.pop();
        util::u32_t const b = sequential.pop();
        if (a != b) {
            fmt::print(stderr, "FAIL max_heap_dary<{}>::insert_many parity init={} batch={} a={} b={}\n", arity_,
                       initial_count, batch_count, a, b);
            return void(expect_true(false));
        }
    }
}

/** Bulk @c insert_many(std::array) must pop in the same descending order as single inserts. */
template <std::size_t batch_size_>
void test_max_heap_insert_many(util::u64_t seed, std::size_t initial_size) noexcept {
    util::max_heap<util::u32_t> bulk {util::global_allocator};
    util::max_heap<util::u32_t> sequential {util::global_allocator};
    expect_true(succeeded(bulk.reserve(initial_size + batch_size_)));
    expect_true(succeeded(sequential.reserve(initial_size + batch_size_)));

    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> distribution {1'000'000 + 1};

    for (std::size_t i = 0; i < initial_size; ++i) {
        util::u32_t const value = distribution(random_generator);
        expect_true(succeeded(bulk.insert(util::u32_t(value))));
        expect_true(succeeded(sequential.insert(util::u32_t(value))));
    }

    std::array<util::u32_t, batch_size_> batch;
    for (auto& value : batch) value = distribution(random_generator);
    expect_true(succeeded(bulk.insert_many(batch)));
    for (auto const& value : batch) expect_true(succeeded(sequential.insert(util::u32_t(value))));

    expect_equal(bulk.size(), sequential.size());
    for (std::size_t i = 0; i < bulk.size(); ++i) {
        util::u32_t const bulk_pop = bulk.pop();
        util::u32_t const sequential_pop = sequential.pop();
        if (bulk_pop != sequential_pop) {
            fmt::print(stderr, "FAIL insert_many N={} init={} pop[{}] bulk={} sequential={}\n", batch_size_,
                       initial_size, i, bulk_pop, sequential_pop);
            return void(expect_true(false));
        }
    }
}

/** @c pop_many<N> must return the same sequence as N sequential @c pop calls. */
template <std::size_t batch_size_>
void test_max_heap_pop_many() noexcept {
    util::max_heap<util::u32_t> heap {util::global_allocator};
    expect_true(succeeded(heap.reserve(32)));
    for (util::u32_t value : {5u, 3u, 7u, 1u, 9u, 4u, 8u, 2u, 6u, 10u})
        expect_true(succeeded(heap.insert(util::u32_t(value))));

    auto popped = heap.pop_many<batch_size_>();

    for (std::size_t i = 1; i < batch_size_; ++i)
        if (popped[i - 1] < popped[i]) {
            fmt::print(stderr, "FAIL pop_many N={} not descending at i={} ({} < {})\n", batch_size_, i, popped[i - 1],
                       popped[i]);
            return void(expect_true(false));
        }
}

#pragma endregion Max Heap

/** An entry owning one ledgered word drawn from the column's allocator, so the ledger counts every
 *  entry built and balances only once every one is destroyed. */
struct ledgered_entry_t {
    counting_allocator<util::u32_t> allocator;
    util::u32_t* word = nullptr;

    explicit ledgered_entry_t(counting_allocator_t const& column_allocator) noexcept
        : allocator(column_allocator), word(allocator.allocate(1)) {}
    ledgered_entry_t(ledgered_entry_t const&) = delete;
    ledgered_entry_t& operator=(ledgered_entry_t const&) = delete;
    ~ledgered_entry_t() noexcept {
        if (word) allocator.deallocate(word, 1);
    }
};

/** Covered entries start zeroed and keep their addresses as the column grows, and a bulk walk
 *  visits every entry once, run by run; entries that are not trivially copyable are built from the
 *  column's allocator and destroyed with their chunks. */
void test_chunked_column() noexcept {
    using schedule_t = util::chunk_schedule<4, 6>;
    allocation_ledger_t ledger;
    {
        util::chunked_column<util::u32_t, schedule_t, 1024, counting_allocator_t> column {
            counting_allocator_t {ledger}};
        expect_true(succeeded(column.cover(20)));
        expect_equal(column[19], 0u); // A covered chunk starts zeroed.
        for (util::u32_t entry = 0; entry != 20; ++entry) column[entry] = entry;
        util::u32_t const* const early = &column[3];
        expect_true(succeeded(column.cover(500)));
        expect_true(early == &column[3]);
        for (util::u32_t entry = 20; entry != 500; ++entry) column[entry] = entry;
        std::size_t visited = 0, runs = 0;
        column.for_each_span({7, 487}, [&](std::span<util::u32_t> run) noexcept {
            for (util::u32_t const value : run) expect_equal(value, static_cast<util::u32_t>(7 + visited++));
            ++runs;
        });
        expect_equal(visited, 480u);
        expect_true(runs > 1);
        expect_equal(column.cover(2048), util::status_code_t::overflow_k);
    }
    expect_equal(ledger.live_bytes, 0);

    allocation_ledger_t built_ledger;
    {
        util::chunked_column<ledgered_entry_t, schedule_t, 1024, counting_allocator_t> column {
            counting_allocator_t {built_ledger}};
        expect_true(succeeded(column.cover(20)));
        // Two chunks, then one word per entry: every entry was built, from the column's allocator.
        expect_equal(built_ledger.allocations, 2 + column.capacity());
        expect_true(column[19].allocator.ledger_ == &built_ledger);
        column.reset();
        expect_equal(built_ledger.live_bytes, 0);
        expect_true(succeeded(column.cover(20))); // The column's destructor destroys these.
    }
    expect_equal(built_ledger.live_bytes, 0);
}

/** Rows read packed into the tail of padded storage spread forward to their own places, each
 *  padded with the fill, and rows already as wide as their storage stay put. */
void test_expand_entries_forward() noexcept {
    constexpr std::size_t rows = 5, stored_bytes = 3, padded_bytes = 8;
    std::array<util::byte_t, rows * padded_bytes> storage {};
    for (std::size_t byte = 0; byte != rows * stored_bytes; ++byte)
        storage[storage.size() - rows * stored_bytes + byte] = static_cast<util::byte_t>(byte + 1);
    util::expand_entries_forward(storage, rows, stored_bytes, padded_bytes, util::byte_t {0xFF});
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t byte = 0; byte != padded_bytes; ++byte)
            expect_equal(std::to_integer<unsigned>(storage[row * padded_bytes + byte]),
                         byte < stored_bytes ? static_cast<unsigned>(row * stored_bytes + byte + 1) : 0xFFu);
    std::array<util::byte_t, rows * padded_bytes> const unchanged = storage;
    util::expand_entries_forward(storage, rows, padded_bytes, padded_bytes, util::byte_t {0});
    expect_true(storage == unchanged);
}

/** Threads fail at once with distinct codes and notes, and the latch keeps one thread's pair whole,
 *  never one thread's code beside another's note. */
void test_keep_first_failure_pairs() noexcept {
    constexpr std::size_t threads_k = 8;
    std::array<util::status_t, threads_k> const failures {{
        {util::status_code_t::out_of_memory_k, "First thread failed"},
        {util::status_code_t::shape_mismatch_k, "Second thread failed"},
        {util::status_code_t::io_k, "Third thread failed"},
        {util::status_code_t::device_k, "Fourth thread failed"},
        {util::status_code_t::malformed_k, "Fifth thread failed"},
        {util::status_code_t::overflow_k, "Sixth thread failed"},
        {util::status_code_t::contended_k, "Seventh thread failed"},
        {util::status_code_t::invalid_argument_k, "Eighth thread failed"},
    }};
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(threads_k))) return void(expect_true(false));
    for (std::size_t repetition = 0; repetition != 256; ++repetition) {
        util::status_t first;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread < threads_k) util::keep_first_failure<fu::standard_atomic_ref>(first, failures[thread]);
        });
        auto const winner = std::ranges::find(failures, first.code, &util::status_t::code);
        expect_true(winner != failures.end() && winner->note == first.note);
    }
}

void run_ram_buffers_cases() {
    util::u64_t const seed = test_run.settings.seed;
    run_test_case("test_sorted_buffer_single_element", test_sorted_buffer_single_element);
    run_test_case("test_sorted_buffer_batch_parity<4>", test_sorted_buffer_batch_parity<4>,
                  stream_key(seed, "sorted_buffer_batch_parity"));
    run_test_case("test_sorted_buffer_batch_parity<16>", test_sorted_buffer_batch_parity<16>,
                  stream_key(seed, "sorted_buffer_batch_parity", 1));
    run_test_case("test_radix_groups", test_radix_groups, stream_key(seed, "radix_groups"));

    run_test_case("test_hash_set_single_element", test_hash_set_single_element);
    run_test_case("test_hash_set_growth_parity", test_hash_set_growth_parity,
                  stream_key(seed, "hash_set_growth_parity"));
    run_test_case("test_hash_set_batch_parity<4>", test_hash_set_batch_parity<4>,
                  stream_key(seed, "hash_set_batch_parity"));
    run_test_case("test_hash_set_batch_parity<16>", test_hash_set_batch_parity<16>,
                  stream_key(seed, "hash_set_batch_parity", 1));
    run_test_case("test_hash_set_arena_growth", test_hash_set_arena_growth);

    run_test_case("test_max_heap_push_pop", test_max_heap_push_pop);
    run_test_case("test_max_heap_dary_parity_with_binary", test_max_heap_dary_parity_with_binary,
                  stream_key(seed, "max_heap_dary_parity_with_binary"), 256);
    run_test_case("test_max_heap_dary_push_pop<4>", test_max_heap_dary_push_pop<4>,
                  stream_key(seed, "max_heap_dary_push_pop"), 256);
    run_test_case("test_max_heap_dary_push_pop<8>", test_max_heap_dary_push_pop<8>,
                  stream_key(seed, "max_heap_dary_push_pop", 1), 256);
    run_test_case("test_max_heap_dary_push_pop<16>", test_max_heap_dary_push_pop<16>,
                  stream_key(seed, "max_heap_dary_push_pop", 2), 256);
    run_test_case("test_max_heap_dary_erase_below<4>", test_max_heap_dary_erase_below<4>,
                  stream_key(seed, "max_heap_dary_erase_below"), 256);
    run_test_case("test_max_heap_dary_erase_below<8>", test_max_heap_dary_erase_below<8>,
                  stream_key(seed, "max_heap_dary_erase_below", 1), 256);
    // An empty heap takes Floyd's path, a populated one shifts each entry up.
    run_test_case("test_max_heap_dary_insert_many<4>(0, 32)", test_max_heap_dary_insert_many<4>,
                  stream_key(seed, "max_heap_dary_insert_many"), 0, 32);
    run_test_case("test_max_heap_dary_insert_many<4>(128, 32)", test_max_heap_dary_insert_many<4>,
                  stream_key(seed, "max_heap_dary_insert_many", 1), 128, 32);
    run_test_case("test_max_heap_dary_insert_many<8>(0, 32)", test_max_heap_dary_insert_many<8>,
                  stream_key(seed, "max_heap_dary_insert_many", 2), 0, 32);
    run_test_case("test_max_heap_dary_insert_many<8>(128, 32)", test_max_heap_dary_insert_many<8>,
                  stream_key(seed, "max_heap_dary_insert_many", 3), 128, 32);
    run_test_case("test_max_heap_dary_insert_many<16>(0, 32)", test_max_heap_dary_insert_many<16>,
                  stream_key(seed, "max_heap_dary_insert_many", 4), 0, 32);
    run_test_case("test_max_heap_dary_insert_many<16>(128, 32)", test_max_heap_dary_insert_many<16>,
                  stream_key(seed, "max_heap_dary_insert_many", 5), 128, 32);
    run_test_case("test_max_heap_insert_many<4>(0)", test_max_heap_insert_many<4>,
                  stream_key(seed, "max_heap_insert_many"), 0);
    run_test_case("test_max_heap_insert_many<16>(0)", test_max_heap_insert_many<16>,
                  stream_key(seed, "max_heap_insert_many", 1), 0);
    run_test_case("test_max_heap_insert_many<32>(0)", test_max_heap_insert_many<32>,
                  stream_key(seed, "max_heap_insert_many", 2), 0);
    run_test_case("test_max_heap_insert_many<4>(64)", test_max_heap_insert_many<4>,
                  stream_key(seed, "max_heap_insert_many", 3), 64);
    run_test_case("test_max_heap_insert_many<16>(128)", test_max_heap_insert_many<16>,
                  stream_key(seed, "max_heap_insert_many", 4), 128);
    run_test_case("test_max_heap_insert_many<32>(256)", test_max_heap_insert_many<32>,
                  stream_key(seed, "max_heap_insert_many", 5), 256);
    run_test_case("test_max_heap_pop_many<4>", test_max_heap_pop_many<4>);
    run_test_case("test_max_heap_pop_many<8>", test_max_heap_pop_many<8>);
    run_test_case("test_chunked_column", test_chunked_column);
    run_test_case("test_expand_entries_forward", test_expand_entries_forward);
    run_test_case("test_keep_first_failure_pairs", test_keep_first_failure_pairs);
}

} // namespace unum::usearch::test::utilities
