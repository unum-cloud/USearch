/**
 *  @brief Throughput benchmarks for @c ram_queue.hpp containers - @c max_heap,
 *      @c sorted_buffer, and @c growing_hash_set - including their multi-element overloads.
 *
 *  @author Ash Vardanian
 *  @file bench/ram_queue.cpp
 *  @date April 25, 2026
 *
 *  Pure data-structure microbenches. No L1-noise simulator, no HNSW context. Each cell
 *  measures only the container ops.
 *
 *  For the multi-element APIs we run two cells per (container, batch_size): one that
 *  issues @c batch_size separate single-element calls per iteration, and one that issues
 *  a single batch call per iteration with the same items. Same per-iteration overhead,
 *  same total work, only the call-site shape differs - so any time gap is real.
 *
 *  PRNG choice: @c std::mt19937 at ~1.6 ns/call - 2.8x faster than @c std::minstd_rand
 *  (which spends ~4.5 ns on a 64-bit modulo in libstdc++) and within 2x of inline
 *  splitmix64. STL-only, no custom code to maintain. The 0.7 ns/call gap vs splitmix64
 *  is invisible against the data-structure work being measured.
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "usearch/ram_queue.hpp"

using namespace unum::usearch;

/** @brief Single-element @c insert into a steady-state buffer kept near full.
 *  Hits the O(1) fast-reject path nearly every iteration on uniform-u32 input. */
void bench_sorted_buffer_insert_one(benchmark::State& state) {
    auto const limit = static_cast<std::size_t>(state.range(0));
    sorted_buffer<std::uint32_t> buffer;
    buffer.reserve(limit);
    std::mt19937 random_generator {0x5EEDu};
    for (std::size_t i = 0; i < limit; ++i) buffer.insert(random_generator(), limit);

    for (auto _ : state) benchmark::DoNotOptimize(buffer.insert(random_generator(), limit));
    state.SetItemsProcessed(state.iterations());
}

/** @brief Apples-to-apples: @c batch_size separate @c insert calls per iteration. */
template <std::size_t batch_size_>
void bench_sorted_buffer_insert_n_separate(benchmark::State& state) {
    auto const limit = static_cast<std::size_t>(state.range(0));
    sorted_buffer<std::uint32_t> buffer;
    buffer.reserve(limit);
    std::mt19937 random_generator {0x5EEDu};
    for (std::size_t i = 0; i < limit; ++i) buffer.insert(random_generator(), limit);

    for (auto _ : state) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        bool any_accepted = false;
        for (std::size_t i = 0; i < batch_size_; ++i) any_accepted |= buffer.insert(std::uint32_t(batch[i]), limit);
        benchmark::DoNotOptimize(any_accepted);
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief Apples-to-apples: sort the batch - the network for quads, the library beyond -
 *      then one sorted-span @c insert(span, limit) call per iteration. */
template <std::size_t batch_size_>
void bench_sorted_buffer_insert_span(benchmark::State& state) {
    auto const limit = static_cast<std::size_t>(state.range(0));
    sorted_buffer<std::uint32_t> buffer;
    buffer.reserve(limit);
    std::mt19937 random_generator {0x5EEDu};
    for (std::size_t i = 0; i < limit; ++i) buffer.insert(random_generator(), limit);

    for (auto _ : state) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        if constexpr (batch_size_ <= 4) network_sort(std::span(batch));
        else std::sort(batch.begin(), batch.end());
        benchmark::DoNotOptimize(buffer.insert(std::span(batch), limit));
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

BENCHMARK(bench_sorted_buffer_insert_one)->Arg(16)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_sorted_buffer_insert_n_separate, 4)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_sorted_buffer_insert_span, 4)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_sorted_buffer_insert_n_separate, 16)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_sorted_buffer_insert_span, 16)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_sorted_buffer_insert_n_separate, 32)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_sorted_buffer_insert_span, 32)->Arg(64)->Arg(256);

void bench_hash_set_set_one(benchmark::State& state) {
    auto const expected_size = static_cast<std::size_t>(state.range(0));
    growing_hash_set<std::uint32_t> hash_set;
    hash_set.reserve(expected_size * 4);
    std::size_t const clear_every = expected_size;
    std::mt19937 random_generator {0xBEEFu};
    std::size_t step_count = 0;
    for (auto _ : state) {
        std::uint32_t const key = random_generator();
        benchmark::DoNotOptimize(hash_set.set(key));
        if ((++step_count % clear_every) == 0) hash_set.clear();
    }
    state.SetItemsProcessed(state.iterations());
}

/** @brief Apples-to-apples: @c batch_size separate @c set calls per iteration. */
template <std::size_t batch_size_>
void bench_hash_set_set_n_separate(benchmark::State& state) {
    auto const expected_size = static_cast<std::size_t>(state.range(0));
    growing_hash_set<std::uint32_t> hash_set;
    hash_set.reserve(expected_size * 4);
    std::size_t const clear_every = expected_size / batch_size_ + 1;
    std::mt19937 random_generator {0xBEEFu};
    std::size_t step_count = 0;
    for (auto _ : state) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        std::uint64_t accepted_mask = 0;
        for (std::size_t i = 0; i < batch_size_; ++i)
            if (!hash_set.set(batch[i])) accepted_mask |= std::uint64_t {1} << i;
        benchmark::DoNotOptimize(accepted_mask);
        if ((++step_count % clear_every) == 0) hash_set.clear();
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief Apples-to-apples: one @c set(std::array) call per iteration. */
template <std::size_t batch_size_>
void bench_hash_set_set_one_batch(benchmark::State& state) {
    auto const expected_size = static_cast<std::size_t>(state.range(0));
    growing_hash_set<std::uint32_t> hash_set;
    hash_set.reserve(expected_size * 4);
    std::size_t const clear_every = expected_size / batch_size_ + 1;
    std::mt19937 random_generator {0xBEEFu};
    std::size_t step_count = 0;
    for (auto _ : state) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        benchmark::DoNotOptimize(hash_set.set(batch));
        if ((++step_count % clear_every) == 0) hash_set.clear();
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

BENCHMARK(bench_hash_set_set_one)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_set_n_separate, 4)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_set_one_batch, 4)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_set_n_separate, 16)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_set_one_batch, 16)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_set_n_separate, 32)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_set_one_batch, 32)->Arg(256)->Arg(2048)->Arg(16384);

void bench_max_heap_push_pop(benchmark::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<std::uint32_t> heap;
    heap.reserve(target_size * 4);
    std::mt19937 random_generator {0xDEADu};
    for (std::size_t i = 0; i < target_size; ++i) heap.insert(random_generator());

    for (auto _ : state) {
        heap.insert(std::uint32_t(random_generator()));
        heap.insert(std::uint32_t(random_generator()));
        benchmark::DoNotOptimize(heap.pop());
    }
    state.SetItemsProcessed(state.iterations() * 3);
}

/** @brief Apples-to-apples: @c batch_size separate @c insert calls per iteration. */
template <std::size_t batch_size_>
void bench_max_heap_insert_n_separate(benchmark::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<std::uint32_t> heap;
    heap.reserve(target_size * 64);
    std::mt19937 random_generator {0xDEADu};
    for (std::size_t i = 0; i < target_size; ++i) heap.insert(random_generator());

    for (auto _ : state) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        for (auto value : batch) heap.insert(std::uint32_t(value));
        benchmark::ClobberMemory();
        if (heap.size() >= target_size + 64 * batch_size_) {
            heap.clear();
            for (std::size_t i = 0; i < target_size; ++i) heap.insert(random_generator());
        }
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief Apples-to-apples: one @c insert_many(std::array) call per iteration. */
template <std::size_t batch_size_>
void bench_max_heap_insert_one_batch(benchmark::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<std::uint32_t> heap;
    heap.reserve(target_size * 64);
    std::mt19937 random_generator {0xDEADu};
    for (std::size_t i = 0; i < target_size; ++i) heap.insert(random_generator());

    for (auto _ : state) {
        std::array<std::uint32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        benchmark::DoNotOptimize(heap.insert_many(batch));
        if (heap.size() >= target_size + 64 * batch_size_) {
            heap.clear();
            for (std::size_t i = 0; i < target_size; ++i) heap.insert(random_generator());
        }
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief D-ary heap mixed push/pop - same workload as the binary cell, parameterised by arity. */
template <std::size_t arity_>
void bench_max_heap_dary_push_pop(benchmark::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap_dary<arity_, std::uint32_t> heap;
    heap.reserve(target_size * 4);
    std::mt19937 random_generator {0xDEADu};
    auto draw = [&](void) noexcept { return static_cast<std::uint32_t>(random_generator()); };
    for (std::size_t i = 0; i < target_size; ++i) heap.insert(draw());
    for (auto _ : state) {
        heap.insert(std::uint32_t(draw()));
        heap.insert(std::uint32_t(draw()));
        benchmark::DoNotOptimize(heap.pop());
    }
    state.SetItemsProcessed(state.iterations() * 3);
    state.counters["arity"] = static_cast<double>(arity_);
}

/** @brief D-ary heap insert-only steady state - pop periodically to keep size bounded. */
template <std::size_t arity_>
void bench_max_heap_dary_insert_only(benchmark::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap_dary<arity_, std::uint32_t> heap;
    heap.reserve(target_size * 4);
    std::mt19937 random_generator {0xDEADu};
    auto draw = [&](void) noexcept { return static_cast<std::uint32_t>(random_generator()); };
    for (std::size_t i = 0; i < target_size; ++i) heap.insert(draw());
    for (auto _ : state) {
        heap.insert(std::uint32_t(draw()));
        if (heap.size() >= target_size * 2) {
            while (heap.size() > target_size) benchmark::DoNotOptimize(heap.pop());
        }
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["arity"] = static_cast<double>(arity_);
}

void bench_max_heap_insert_only_binary(benchmark::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<std::uint32_t> heap;
    heap.reserve(target_size * 4);
    std::mt19937 random_generator {0xDEADu};
    auto draw = [&](void) noexcept { return static_cast<std::uint32_t>(random_generator()); };
    for (std::size_t i = 0; i < target_size; ++i) heap.insert(draw());
    for (auto _ : state) {
        heap.insert(std::uint32_t(draw()));
        if (heap.size() >= target_size * 2) {
            while (heap.size() > target_size) benchmark::DoNotOptimize(heap.pop());
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(bench_max_heap_push_pop)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_push_pop, 2)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_push_pop, 4)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_push_pop, 8)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_push_pop, 16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(bench_max_heap_insert_only_binary)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_insert_only, 4)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_insert_only, 8)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_dary_insert_only, 16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_TEMPLATE(bench_max_heap_insert_n_separate, 4)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_max_heap_insert_one_batch, 4)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_max_heap_insert_n_separate, 16)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_max_heap_insert_one_batch, 16)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_max_heap_insert_n_separate, 32)->Arg(64)->Arg(256);
BENCHMARK_TEMPLATE(bench_max_heap_insert_one_batch, 32)->Arg(64)->Arg(256);

BENCHMARK_MAIN();
