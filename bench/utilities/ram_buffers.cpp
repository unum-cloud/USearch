/**
 *  @brief Throughput benchmarks for the @c ram_buffers.hpp & @c ram_tables.hpp containers - @c max_heap,
 *      @c sorted_buffer, and @c growing_hash_set - including their multi-element overloads.
 *
 *  @author Ash Vardanian
 *  @file bench/utilities/ram_buffers.cpp
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
 *  PRNG choice: the library's own @c splitmix64_t - sub-nanosecond raw draws with no
 *  standard-library engine machinery in the timed loop, so the generator stays invisible
 *  against the data-structure work being measured.
 */
#include <cstdint>

#include <algorithm>
#include <array>
#include <memory> // `std::allocator` - every container names its allocator
#include <vector>

#include <benchmark/benchmark.h>

#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_tables.hpp"

namespace bm = benchmark;
using namespace unum::usearch;

/** @brief Single-element @c insert into a steady-state buffer kept near full.
 *  Hits the O(1) fast-reject path nearly every iteration on uniform-u32 input. */
void bench_sorted_buffer_insert_one(bm::State& state) {
    auto const limit = static_cast<std::size_t>(state.range(0));
    sorted_buffer<u32_t> buffer {global_allocator};
    if (failed(buffer.reserve(limit))) return state.SkipWithError("the buffer refused its reservation");
    splitmix64_t random_generator {0x5EEDull};
    for (std::size_t i = 0; i < limit; ++i) buffer.insert(random_generator(), limit);

    for (auto _ : state) bm::DoNotOptimize(buffer.insert(random_generator(), limit));
    state.SetItemsProcessed(state.iterations());
}

/** @brief Apples-to-apples: @c batch_size separate @c insert calls per iteration. */
template <std::size_t batch_size_>
void bench_sorted_buffer_insert_n_separate(bm::State& state) {
    auto const limit = static_cast<std::size_t>(state.range(0));
    sorted_buffer<u32_t> buffer {global_allocator};
    if (failed(buffer.reserve(limit))) return state.SkipWithError("the buffer refused its reservation");
    splitmix64_t random_generator {0x5EEDull};
    for (std::size_t i = 0; i < limit; ++i) buffer.insert(random_generator(), limit);

    for (auto _ : state) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        bool any_accepted = false;
        for (std::size_t i = 0; i < batch_size_; ++i) any_accepted |= buffer.insert(u32_t(batch[i]), limit);
        bm::DoNotOptimize(any_accepted);
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief Apples-to-apples: sort the batch - the network for quads, the library beyond -
 *      then one sorted-span @c insert(span, limit) call per iteration. */
template <std::size_t batch_size_>
void bench_sorted_buffer_insert_span(bm::State& state) {
    auto const limit = static_cast<std::size_t>(state.range(0));
    sorted_buffer<u32_t> buffer {global_allocator};
    if (failed(buffer.reserve(limit))) return state.SkipWithError("the buffer refused its reservation");
    splitmix64_t random_generator {0x5EEDull};
    for (std::size_t i = 0; i < limit; ++i) buffer.insert(random_generator(), limit);

    for (auto _ : state) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        if constexpr (batch_size_ <= 4) network_sort(std::span(batch));
        else std::sort(batch.begin(), batch.end());
        bm::DoNotOptimize(buffer.insert(std::span(batch), limit));
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

void bench_hash_set_insert_one(bm::State& state) {
    auto const expected_size = static_cast<std::size_t>(state.range(0));
    growing_hash_set<u32_t> hash_set {global_allocator};
    if (failed(hash_set.reserve(expected_size * 4))) return state.SkipWithError("the set refused its reservation");
    std::size_t const clear_every = expected_size;
    splitmix64_t random_generator {0xBEEFull};
    std::size_t step_count = 0;
    for (auto _ : state) {
        u32_t const key = random_generator();
        bm::DoNotOptimize(hash_set.insert(key));
        if ((++step_count % clear_every) == 0) hash_set.clear();
    }
    state.SetItemsProcessed(state.iterations());
}

/** @brief Apples-to-apples: @c batch_size separate @c insert calls per iteration. */
template <std::size_t batch_size_>
void bench_hash_set_insert_n_separate(bm::State& state) {
    auto const expected_size = static_cast<std::size_t>(state.range(0));
    growing_hash_set<u32_t> hash_set {global_allocator};
    if (failed(hash_set.reserve(expected_size * 4))) return state.SkipWithError("the set refused its reservation");
    std::size_t const clear_every = expected_size / batch_size_ + 1;
    splitmix64_t random_generator {0xBEEFull};
    std::size_t step_count = 0;
    for (auto _ : state) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        u64_t accepted_mask = 0;
        for (std::size_t i = 0; i < batch_size_; ++i)
            if (hash_set.insert(batch[i]) == membership_t::absent_k) accepted_mask |= u64_t {1} << i;
        bm::DoNotOptimize(accepted_mask);
        if ((++step_count % clear_every) == 0) hash_set.clear();
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief Apples-to-apples: one @c insert(std::array) call per iteration. */
template <std::size_t batch_size_>
void bench_hash_set_insert_one_batch(bm::State& state) {
    auto const expected_size = static_cast<std::size_t>(state.range(0));
    growing_hash_set<u32_t> hash_set {global_allocator};
    if (failed(hash_set.reserve(expected_size * 4))) return state.SkipWithError("the set refused its reservation");
    std::size_t const clear_every = expected_size / batch_size_ + 1;
    splitmix64_t random_generator {0xBEEFull};
    std::size_t step_count = 0;
    for (auto _ : state) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        bm::DoNotOptimize(hash_set.insert(batch));
        if ((++step_count % clear_every) == 0) hash_set.clear();
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

BENCHMARK(bench_hash_set_insert_one)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_insert_n_separate, 4)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_insert_one_batch, 4)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_insert_n_separate, 16)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_insert_one_batch, 16)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_insert_n_separate, 32)->Arg(256)->Arg(2048)->Arg(16384);
BENCHMARK_TEMPLATE(bench_hash_set_insert_one_batch, 32)->Arg(256)->Arg(2048)->Arg(16384);

void bench_max_heap_push_pop(bm::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<u32_t> heap {global_allocator};
    if (failed(heap.reserve(target_size * 4))) return state.SkipWithError("the heap refused its reservation");
    splitmix64_t random_generator {0xDEADull};
    for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(random_generator()));

    for (auto _ : state) {
        bm::DoNotOptimize(heap.insert(u32_t(random_generator())));
        bm::DoNotOptimize(heap.insert(u32_t(random_generator())));
        bm::DoNotOptimize(heap.pop());
    }
    state.SetItemsProcessed(state.iterations() * 3);
}

/** @brief Apples-to-apples: @c batch_size separate @c insert calls per iteration. */
template <std::size_t batch_size_>
void bench_max_heap_insert_n_separate(bm::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<u32_t> heap {global_allocator};
    if (failed(heap.reserve(target_size * 64))) return state.SkipWithError("the heap refused its reservation");
    splitmix64_t random_generator {0xDEADull};
    for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(random_generator()));

    for (auto _ : state) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        for (auto value : batch) bm::DoNotOptimize(heap.insert(u32_t(value)));
        bm::ClobberMemory();
        if (heap.size() >= target_size + 64 * batch_size_) {
            heap.clear();
            for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(random_generator()));
        }
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief Apples-to-apples: one @c insert_many(std::array) call per iteration. */
template <std::size_t batch_size_>
void bench_max_heap_insert_one_batch(bm::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<u32_t> heap {global_allocator};
    if (failed(heap.reserve(target_size * 64))) return state.SkipWithError("the heap refused its reservation");
    splitmix64_t random_generator {0xDEADull};
    for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(random_generator()));

    for (auto _ : state) {
        std::array<u32_t, batch_size_> batch;
        for (auto& value : batch) value = random_generator();
        bm::DoNotOptimize(heap.insert_many(batch));
        if (heap.size() >= target_size + 64 * batch_size_) {
            heap.clear();
            for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(random_generator()));
        }
    }
    state.SetItemsProcessed(state.iterations() * batch_size_);
    state.counters["batch_N"] = static_cast<double>(batch_size_);
}

/** @brief D-ary heap mixed push/pop - same workload as the binary cell, parameterised by arity. */
template <std::size_t arity_>
void bench_max_heap_dary_push_pop(bm::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap_dary<arity_, u32_t> heap {global_allocator};
    if (failed(heap.reserve(target_size * 4))) return state.SkipWithError("the heap refused its reservation");
    splitmix64_t random_generator {0xDEADull};
    auto draw = [&](void) noexcept { return static_cast<u32_t>(random_generator()); };
    for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(draw()));
    for (auto _ : state) {
        bm::DoNotOptimize(heap.insert(u32_t(draw())));
        bm::DoNotOptimize(heap.insert(u32_t(draw())));
        bm::DoNotOptimize(heap.pop());
    }
    state.SetItemsProcessed(state.iterations() * 3);
    state.counters["arity"] = static_cast<double>(arity_);
}

/** @brief D-ary heap insert-only steady state - pop periodically to keep size bounded. */
template <std::size_t arity_>
void bench_max_heap_dary_insert_only(bm::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap_dary<arity_, u32_t> heap {global_allocator};
    if (failed(heap.reserve(target_size * 4))) return state.SkipWithError("the heap refused its reservation");
    splitmix64_t random_generator {0xDEADull};
    auto draw = [&](void) noexcept { return static_cast<u32_t>(random_generator()); };
    for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(draw()));
    for (auto _ : state) {
        bm::DoNotOptimize(heap.insert(u32_t(draw())));
        if (heap.size() >= target_size * 2) {
            while (heap.size() > target_size) bm::DoNotOptimize(heap.pop());
        }
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["arity"] = static_cast<double>(arity_);
}

void bench_max_heap_insert_only_binary(bm::State& state) {
    auto const target_size = static_cast<std::size_t>(state.range(0));
    max_heap<u32_t> heap {global_allocator};
    if (failed(heap.reserve(target_size * 4))) return state.SkipWithError("the heap refused its reservation");
    splitmix64_t random_generator {0xDEADull};
    auto draw = [&](void) noexcept { return static_cast<u32_t>(random_generator()); };
    for (std::size_t i = 0; i < target_size; ++i) bm::DoNotOptimize(heap.insert(draw()));
    for (auto _ : state) {
        bm::DoNotOptimize(heap.insert(u32_t(draw())));
        if (heap.size() >= target_size * 2) {
            while (heap.size() > target_size) bm::DoNotOptimize(heap.pop());
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
