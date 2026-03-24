/**
 *  @file       test_sorted_buffer_overflow.cpp
 *  @brief      Regression tests for sorted_buffer_gt heap-buffer-overflow fix.
 *
 *  Root cause: search_to_insert_() / search_to_update_() never called
 *  top.reserve() / next.reserve() before using sorted_buffer_gt, and
 *  reserve() had an off-by-one (< vs <=) causing spurious reallocation.
 *
 *  Compile (with ASAN):
 *    g++ -std=c++17 -fsanitize=address -fno-omit-frame-pointer -g \
 *        -DUSEARCH_USE_SIMSIMD=0 -DUSEARCH_USE_FP16LIB=0 \
 *        -I../include test_sorted_buffer_overflow.cpp \
 *        -o test_sorted_buffer_overflow -lpthread
 */
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>

#define USEARCH_USE_SIMSIMD 0
#define USEARCH_USE_FP16LIB 0
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace unum::usearch;

static int tests_passed = 0;
static int tests_failed = 0;

#define EXPECT(cond)                                                                     \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "  FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__);   \
            tests_failed++;                                                              \
            return;                                                                      \
        }                                                                                \
    } while (0)

// Types matching what index_gt uses internally
using distance_t = float;
enum slot32_t : std::uint32_t {};
template <> struct unum::usearch::hash_gt<slot32_t> : public unum::usearch::hash_gt<std::uint32_t> {};
template <> struct unum::usearch::default_free_value_gt<slot32_t> {
    static slot32_t value() noexcept { return static_cast<slot32_t>(std::numeric_limits<std::uint32_t>::max()); }
};

struct candidate_t {
    distance_t distance;
    slot32_t slot;
    inline bool operator<(candidate_t other) const noexcept { return distance < other.distance; }
};

using candidates_allocator_t = std::allocator<candidate_t>;
using sorted_buffer_t = sorted_buffer_gt<candidate_t, std::less<candidate_t>, candidates_allocator_t>;
using max_heap_t = max_heap_gt<candidate_t, std::less<candidate_t>, candidates_allocator_t>;

// ---- sorted_buffer_gt tests ----

void test_reserve_equal_capacity() {
    std::printf("  test_reserve_equal_capacity ... ");
    sorted_buffer_t buf;
    buf.reserve(16);
    std::size_t cap1 = buf.capacity();
    buf.reserve(16); // Should be no-op with <= fix
    std::size_t cap2 = buf.capacity();
    EXPECT(cap1 == cap2);
    std::printf("PASS (cap=%zu)\n", cap1);
    tests_passed++;
}

void test_normal_fill_and_eviction() {
    std::printf("  test_normal_fill_and_eviction ... ");
    sorted_buffer_t buf;
    std::size_t limit = 4;
    buf.reserve(limit);
    for (std::size_t i = 1; i <= limit; i++)
        buf.insert({static_cast<float>(i), static_cast<slot32_t>(i)}, limit);
    EXPECT(buf.size() == limit);
    // Insert closer candidate — should evict farthest (4.0)
    bool inserted = buf.insert({0.5f, static_cast<slot32_t>(99)}, limit);
    EXPECT(inserted);
    EXPECT(buf.size() == limit);
    EXPECT(buf.top().distance < 4.0f);
    std::printf("PASS\n");
    tests_passed++;
}

void test_insert_reserved_with_capacity() {
    std::printf("  test_insert_reserved_with_capacity ... ");
    sorted_buffer_t buf;
    buf.reserve(4);
    buf.insert_reserved({0.5f, static_cast<slot32_t>(0)});
    buf.insert_reserved({0.3f, static_cast<slot32_t>(1)});
    buf.insert_reserved({0.7f, static_cast<slot32_t>(2)});
    EXPECT(buf.size() == 3);
    EXPECT(buf.top().distance == 0.7f);
    std::printf("PASS\n");
    tests_passed++;
}

// ---- max_heap_gt reserve test ----

void test_max_heap_reserve_equal_capacity() {
    std::printf("  test_max_heap_reserve_equal_capacity ... ");
    max_heap_t heap;
    heap.reserve(16);
    std::size_t cap1 = heap.capacity();
    heap.reserve(16);
    std::size_t cap2 = heap.capacity();
    EXPECT(cap1 == cap2);
    std::printf("PASS (cap=%zu)\n", cap1);
    tests_passed++;
}

// ---- Full index search path test ----

void test_search_path_no_overflow() {
    std::printf("  test_search_path_no_overflow ... ");
    using index_t = index_dense_gt<std::int64_t, slot32_t>;
    index_config_t config;
    config.connectivity = 16;

    metric_punned_t metric(3, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
    index_t index = index_t::make(metric, config);
    index.reserve(100);

    std::default_random_engine engine(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < 100; i++) {
        float vec[3] = {dist(engine), dist(engine), dist(engine)};
        index.add(static_cast<std::int64_t>(i), vec);
    }

    // Search with various expansion values
    for (std::size_t expansion : {10, 50, 100, 256, 512}) {
        float query[3] = {dist(engine), dist(engine), dist(engine)};
        auto results = index.search(query, 10, index_t::any_thread(), false, expansion);
        EXPECT(results.size() <= 10);
    }
    std::printf("PASS\n");
    tests_passed++;
}

void test_concurrent_search() {
    std::printf("  test_concurrent_search ... ");
    using index_t = index_dense_gt<std::int64_t, slot32_t>;
    index_config_t config;
    config.connectivity = 32;

    metric_punned_t metric(16, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
    index_t index = index_t::make(metric, config);
    index.reserve(200);

    std::default_random_engine engine(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < 200; i++) {
        std::vector<float> vec(16);
        for (auto& v : vec) v = dist(engine);
        index.add(static_cast<std::int64_t>(i), vec.data());
    }

    // Concurrent searches with high expansion — exercises thread context reuse
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    for (int t = 0; t < 8; t++) {
        threads.emplace_back([&, t]() {
            std::default_random_engine eng(t * 1000);
            std::uniform_real_distribution<float> d(-1.0f, 1.0f);
            for (int q = 0; q < 50; q++) {
                std::vector<float> query(16);
                for (auto& v : query) v = d(eng);
                auto results = index.search(query.data(), 10, index_t::any_thread(), false, 256);
                if (results.size() > 10) errors++;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT(errors == 0);
    std::printf("PASS (8 threads x 50 queries)\n");
    tests_passed++;
}

int main() {
    std::printf("=== sorted_buffer_gt heap-buffer-overflow regression tests ===\n\n");

    std::printf("sorted_buffer_gt unit tests:\n");
    test_reserve_equal_capacity();
    test_normal_fill_and_eviction();
    test_insert_reserved_with_capacity();

    std::printf("\nmax_heap_gt unit tests:\n");
    test_max_heap_reserve_equal_capacity();

    std::printf("\nFull index path tests:\n");
    test_search_path_no_overflow();
    test_concurrent_search();

    std::printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
