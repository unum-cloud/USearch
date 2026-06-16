/**
 *  @brief Smoke tests for `traversal_queue.hpp` containers - @c max_heap,
 *      @c sorted_buffer, and @c growing_hash_set - including their four-wide and
 *      larger-N batch overloads plus @c insert_many / @c pop_many on the heap.
 *
 *  @author Ash Vardanian
 *  @file test/traversal_queue.cpp
 *  @date April 24, 2026
 *
 *  Mirrors the structure of `test/inner_metric.cpp`:
 *
 *  1. One monolithic TU, no test framework. Each check returns a failure count;
 *     counts roll up to @c main without any global mutable state. A non-zero exit
 *     code means something failed.
 *  2. Serial single-element operations define the oracle for the multi-element overloads.
 *     For @c sorted_buffer::insert(std::array,limit) the oracle uses @c std::sort on
 *     the union because the "top-k of (existing ∪ batch)" semantic can accept
 *     candidates that cascading single-element inserts would reject.
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "usearch/traversal_queue.hpp"

using namespace unum::usearch;

/** @brief Prints a one-line failure report; returns 0 on success, 1 on mismatch. */
inline int check_true(bool condition, std::string_view label) noexcept {
    if (condition) return 0;
    fmt::print(stderr, "FAIL {}\n", label);
    return 1;
}

template <typename value_>
inline int check_eq(value_ const& actual, value_ const& expected, std::string_view label) noexcept {
    if (actual == expected) return 0;
    fmt::print(stderr, "FAIL {:<40}  actual={}  expected={}\n", label, actual, expected);
    return 1;
}

// sorted_buffer

int check_sorted_buffer_single_element() noexcept {
    int failures = 0;
    sorted_buffer<std::uint32_t> buffer;
    buffer.reserve(16);

    buffer.insert(std::uint32_t {5}, 4);
    buffer.insert(std::uint32_t {3}, 4);
    buffer.insert(std::uint32_t {7}, 4);
    buffer.insert(std::uint32_t {1}, 4);
    failures += check_eq<std::size_t>(buffer.size(), 4, "sorted_buffer size after four inserts");
    failures += check_eq<std::uint32_t>(buffer.data()[0], 1, "sorted_buffer data[0] after fill");
    failures += check_eq<std::uint32_t>(buffer.data()[3], 7, "sorted_buffer data[3] after fill");

    buffer.insert(std::uint32_t {4}, 4);
    failures += check_eq<std::size_t>(buffer.size(), 4, "sorted_buffer size after overfill");
    failures += check_eq<std::uint32_t>(buffer.data()[3], 5, "sorted_buffer data[3] after overfill");

    bool const accepted_above_top = buffer.insert(std::uint32_t {100}, 4);
    failures += check_true(!accepted_above_top, "sorted_buffer rejects values above the top");
    return failures;
}

/**
 *  @brief Parity of `insert(std::array, limit)` against `std::sort(existing ∪ batch)[0..limit)`.
 *      Tested at batch sizes 4 and 16 to exercise both the hand-friendly case and the
 *      generic-template case.
 */
template <std::size_t batch_size_>
int check_sorted_buffer_batch_parity(unsigned seed) noexcept {
    constexpr std::size_t limit_k = 64;
    constexpr int steps_k = 400;
    sorted_buffer<std::uint32_t> tested;
    tested.reserve(2 * limit_k);
    std::vector<std::uint32_t> oracle_flat;
    oracle_flat.reserve(2 * limit_k);

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1'000'000);

    int failures = 0;
    for (int step = 0; step < steps_k; ++step) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = distribution(random_generator);
        tested.insert(batch, limit_k);

        for (std::uint32_t v : batch) oracle_flat.push_back(v);
        std::sort(oracle_flat.begin(), oracle_flat.end());
        if (oracle_flat.size() > limit_k) oracle_flat.resize(limit_k);

        if (tested.size() != oracle_flat.size()) {
            fmt::print(stderr, "FAIL sorted_buffer N={} parity step={} sizes diverge tested={} oracle={}\n",
                       batch_size_, step, tested.size(), oracle_flat.size());
            return ++failures;
        }
        for (std::size_t i = 0; i < tested.size(); ++i)
            if (tested.data()[i] != oracle_flat[i]) {
                fmt::print(stderr, "FAIL sorted_buffer N={} parity step={} i={} tested={} oracle={}\n", batch_size_,
                           step, i, tested.data()[i], oracle_flat[i]);
                return ++failures;
            }
    }
    return failures;
}

/** @brief `insert_sorted` on a pre-sorted batch must match `insert` on the same unsorted batch. */
template <std::size_t batch_size_>
int check_sorted_buffer_insert_sorted_parity(unsigned seed) noexcept {
    constexpr std::size_t limit_k = 64;
    constexpr int steps_k = 400;
    sorted_buffer<std::uint32_t> sorted_path;
    sorted_buffer<std::uint32_t> unsorted_path;
    sorted_path.reserve(2 * limit_k);
    unsorted_path.reserve(2 * limit_k);

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1'000'000);

    int failures = 0;
    for (int step = 0; step < steps_k; ++step) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = distribution(random_generator);

        std::array<std::uint32_t, batch_size_> presorted = batch;
        std::sort(presorted.begin(), presorted.end());

        unsorted_path.insert(batch, limit_k);
        sorted_path.insert_sorted(presorted, limit_k);

        if (unsorted_path.size() != sorted_path.size()) {
            fmt::print(stderr, "FAIL insert_sorted N={} step={} sizes diverge sorted={} unsorted={}\n", batch_size_,
                       step, sorted_path.size(), unsorted_path.size());
            return ++failures;
        }
        for (std::size_t i = 0; i < sorted_path.size(); ++i)
            if (sorted_path.data()[i] != unsorted_path.data()[i]) {
                fmt::print(stderr, "FAIL insert_sorted N={} step={} i={} sorted={} unsorted={}\n", batch_size_, step, i,
                           sorted_path.data()[i], unsorted_path.data()[i]);
                return ++failures;
            }
    }
    return failures;
}

// growing_hash_set

int check_hash_set_single_element() noexcept {
    int failures = 0;
    growing_hash_set<std::uint32_t> hash_set;
    hash_set.reserve(16);
    failures += check_true(hash_set.set(42) == false, "hash_set first insert returns false");
    failures += check_true(hash_set.set(42) == true, "hash_set repeat insert returns true");
    failures += check_true(hash_set.test(42), "hash_set test sees inserted key");
    failures += check_true(!hash_set.test(99), "hash_set test does not see absent key");
    return failures;
}

template <std::size_t batch_size_>
int check_hash_set_batch_parity(unsigned seed) noexcept {
    constexpr int steps_k = 200;
    growing_hash_set<std::uint32_t> tested;
    growing_hash_set<std::uint32_t> oracle;
    tested.reserve(1024);
    oracle.reserve(1024);

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 500);
    auto draw_key = [&](void) noexcept {
        std::uint32_t candidate;
        do candidate = distribution(random_generator);
        while (candidate == default_free_value<std::uint32_t>());
        return candidate;
    };

    int failures = 0;
    for (int step = 0; step < steps_k; ++step) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& key : batch) key = draw_key();

        std::uint64_t const tested_mask = tested.set(batch);
        std::uint64_t oracle_mask = 0;
        for (std::size_t lane = 0; lane < batch_size_; ++lane)
            if (!oracle.set(batch[lane])) oracle_mask |= std::uint64_t {1} << lane;
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

// max_heap

int check_max_heap_push_pop() noexcept {
    int failures = 0;
    max_heap<std::uint32_t> heap;
    heap.reserve(32);
    for (std::uint32_t value : {5u, 3u, 7u, 1u, 9u, 4u}) heap.insert(std::uint32_t(value));
    std::array<std::uint32_t, 6> const expected = {9, 7, 5, 4, 3, 1};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        std::uint32_t const popped = heap.pop();
        failures += check_eq<std::uint32_t>(popped, expected[i], fmt::format("max_heap pop[{}]", i));
    }
    failures += check_true(heap.empty(), "max_heap empty after draining");
    return failures;
}

/** @brief D-ary heap pops in descending order, parameterised by arity. */
template <std::size_t arity_>
int check_max_heap_dary_push_pop(unsigned seed, std::size_t element_count) noexcept {
    max_heap_dary<arity_, std::uint32_t> heap;
    heap.reserve(element_count);
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1'000'000);
    std::vector<std::uint32_t> inputs;
    inputs.reserve(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        std::uint32_t const value = distribution(random_generator);
        inputs.push_back(value);
        heap.insert(std::uint32_t(value));
    }
    std::sort(inputs.begin(), inputs.end(), std::greater<std::uint32_t> {});
    int failures = 0;
    for (std::size_t i = 0; i < element_count; ++i) {
        std::uint32_t const popped = heap.pop();
        if (popped != inputs[i]) {
            fmt::print(stderr, "FAIL max_heap_dary<{}> pop[{}] expected={} got={}\n", arity_, i, inputs[i], popped);
            return ++failures;
        }
    }
    return failures;
}

/** @brief D=2 d-ary heap must agree with the binary heap pop-by-pop on the same input. */
int check_max_heap_dary_parity_with_binary(unsigned seed, std::size_t element_count) noexcept {
    max_heap<std::uint32_t> binary;
    max_heap_dary<2, std::uint32_t> dary;
    binary.reserve(element_count);
    dary.reserve(element_count);
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1'000'000);
    for (std::size_t i = 0; i < element_count; ++i) {
        std::uint32_t const value = distribution(random_generator);
        binary.insert(std::uint32_t(value));
        dary.insert(std::uint32_t(value));
    }
    int failures = 0;
    for (std::size_t i = 0; i < element_count; ++i) {
        std::uint32_t const a = binary.pop();
        std::uint32_t const b = dary.pop();
        if (a != b) {
            fmt::print(stderr, "FAIL d=2 parity pop[{}] binary={} dary={}\n", i, a, b);
            return ++failures;
        }
    }
    return failures;
}

/** @brief D-ary `insert_many` (Floyd vs shift_up paths) must match descending-pop order from individual inserts. */
template <std::size_t arity_>
int check_max_heap_dary_insert_many(unsigned seed, std::size_t initial_count, std::size_t batch_count) noexcept {
    max_heap_dary<arity_, std::uint32_t> bulk;
    max_heap_dary<arity_, std::uint32_t> sequential;
    bulk.reserve(initial_count + batch_count);
    sequential.reserve(initial_count + batch_count);
    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1'000'000);
    for (std::size_t i = 0; i < initial_count; ++i) {
        std::uint32_t const value = distribution(random_generator);
        bulk.insert(std::uint32_t(value));
        sequential.insert(std::uint32_t(value));
    }
    std::vector<std::uint32_t> batch(batch_count);
    for (auto& value : batch) value = distribution(random_generator);
    bulk.insert_many(batch.data(), batch_count);
    for (auto value : batch) sequential.insert(std::uint32_t(value));
    int failures = 0;
    while (!bulk.empty()) {
        std::uint32_t const a = bulk.pop();
        std::uint32_t const b = sequential.pop();
        if (a != b) {
            fmt::print(stderr, "FAIL max_heap_dary<{}>::insert_many parity init={} batch={} a={} b={}\n", arity_,
                       initial_count, batch_count, a, b);
            return ++failures;
        }
    }
    return failures;
}

/** @brief Bulk `insert_many(std::array)` must pop in the same descending order as single inserts. */
template <std::size_t batch_size_>
int check_max_heap_insert_many(unsigned seed, std::size_t initial_size) noexcept {
    max_heap<std::uint32_t> bulk;
    max_heap<std::uint32_t> sequential;
    bulk.reserve(initial_size + batch_size_);
    sequential.reserve(initial_size + batch_size_);

    std::mt19937 random_generator {seed};
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1'000'000);

    for (std::size_t i = 0; i < initial_size; ++i) {
        std::uint32_t const value = distribution(random_generator);
        bulk.insert(std::uint32_t(value));
        sequential.insert(std::uint32_t(value));
    }

    std::array<std::uint32_t, batch_size_> batch;
    for (auto& value : batch) value = distribution(random_generator);
    bulk.insert_many(batch);
    for (auto const& value : batch) sequential.insert(std::uint32_t(value));

    int failures = 0;
    failures += check_eq<std::size_t>(bulk.size(), sequential.size(),
                                      fmt::format("insert_many size N={} init={}", batch_size_, initial_size));
    for (std::size_t i = 0; i < bulk.size(); ++i) {
        std::uint32_t const bulk_pop = bulk.pop();
        std::uint32_t const sequential_pop = sequential.pop();
        if (bulk_pop != sequential_pop) {
            fmt::print(stderr, "FAIL insert_many N={} init={} pop[{}] bulk={} sequential={}\n", batch_size_,
                       initial_size, i, bulk_pop, sequential_pop);
            return ++failures;
        }
    }
    return failures;
}

/** @brief `pop_many<N>` must return the same sequence as N sequential @c pop calls. */
template <std::size_t batch_size_>
int check_max_heap_pop_many() noexcept {
    max_heap<std::uint32_t> heap;
    heap.reserve(32);
    for (std::uint32_t value : {5u, 3u, 7u, 1u, 9u, 4u, 8u, 2u, 6u, 10u}) heap.insert(std::uint32_t(value));

    auto popped = heap.pop_many<batch_size_>();

    int failures = 0;
    for (std::size_t i = 1; i < batch_size_; ++i)
        if (popped[i - 1] < popped[i]) {
            fmt::print(stderr, "FAIL pop_many N={} not descending at i={} ({} < {})\n", batch_size_, i, popped[i - 1],
                       popped[i]);
            return ++failures;
        }
    return failures;
}

int main(void) {
    int failures = 0;
    failures += check_sorted_buffer_single_element();
    failures += check_sorted_buffer_batch_parity<4>(0xABCD1234u);
    failures += check_sorted_buffer_batch_parity<16>(0x13572468u);
    failures += check_sorted_buffer_insert_sorted_parity<4>(0xF00DBABEu);
    failures += check_sorted_buffer_insert_sorted_parity<16>(0xDEADBEEFu);

    failures += check_hash_set_single_element();
    failures += check_hash_set_batch_parity<4>(0xFEEDFACEu);
    failures += check_hash_set_batch_parity<16>(0x01234567u);

    failures += check_max_heap_push_pop();
    failures += check_max_heap_dary_parity_with_binary(0xD2U, 256);
    failures += check_max_heap_dary_push_pop<4>(0xD4U, 256);
    failures += check_max_heap_dary_push_pop<8>(0xD8U, 256);
    failures += check_max_heap_dary_push_pop<16>(0xD16U, 256);
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

    if (failures != 0) fmt::print(stderr, "{} failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
