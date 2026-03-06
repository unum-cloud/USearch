/**
 * @file index_gt_test.cpp
 * @brief GTest-based unit tests for index.hpp (index_gt).
 *
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;

// ---------------------------------------------------------------------------
// Mock vector store
// ---------------------------------------------------------------------------
template <typename vector_element_at = float> class mock_vector_store_gt {
  public:
    using element_t = vector_element_at;

    mock_vector_store_gt() = default;

    mock_vector_store_gt(std::size_t count, std::size_t dims, unsigned seed = 2025) : dimensions_(dims) {
        std::default_random_engine gen(seed);
        std::uniform_real_distribution<float> dist{0.f, 1.f};
        vectors_.resize(count);
        for (auto& v : vectors_) {
            v.resize(dims);
            std::generate(v.begin(), v.end(), [&] { return static_cast<element_t>(dist(gen)); });
        }
    }

    std::size_t size() const noexcept { return vectors_.size(); }
    std::size_t dimensions() const noexcept { return dimensions_; }
    element_t const* row(std::size_t slot) const noexcept { return vectors_[slot].data(); }
    std::vector<element_t> const& vector(std::size_t slot) const noexcept { return vectors_[slot]; }

  private:
    std::size_t dimensions_ = 0;
    std::vector<std::vector<element_t>> vectors_;
};

// ---------------------------------------------------------------------------
// Custom metric (same pattern as usearch-example.cpp)
// ---------------------------------------------------------------------------
template <typename vector_element_at = float, typename key_at = std::size_t, typename slot_at = std::size_t>
struct cosine_metric_gt {
    using index_typed_t = index_gt<vector_element_at, key_at, slot_at>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    mock_vector_store_gt<vector_element_at> const* store = nullptr;
    std::size_t dimensions = 0;

    vector_element_at const* row(std::size_t slot) const noexcept { return store->row(slot); }

    float operator()(member_cref_t const& a, member_cref_t const& b) const {
        return metric_cos_gt<vector_element_at, float>{}(row(get_slot(b)), row(get_slot(a)), dimensions);
    }
    float operator()(vector_element_at const* query, member_cref_t const& member) const {
        return metric_cos_gt<vector_element_at, float>{}(query, row(get_slot(member)), dimensions);
    }
    float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
        return metric_cos_gt<vector_element_at, float>{}(row(get_slot(*b)), row(get_slot(*a)), dimensions);
    }
    float operator()(vector_element_at const* query, member_citerator_t const& member) const {
        return metric_cos_gt<vector_element_at, float>{}(query, row(get_slot(*member)), dimensions);
    }
};

// ---------------------------------------------------------------------------
// Helper: aligned allocation for index_gt (same pattern as usearch example)
// ---------------------------------------------------------------------------
template <typename index_at> struct aligned_index_gt {
    using index_t = index_at;
    using alloc_t = aligned_allocator_gt<index_t, 64>;

    index_t* index = nullptr;

    template <typename... args_at> explicit aligned_index_gt(args_at&&... args) {
        index = alloc_t{}.allocate(1);
        assert(index != nullptr);
        new (index) index_t(std::forward<args_at>(args)...);
    }

    ~aligned_index_gt() {
        if (index) {
            index->~index_t();
            alloc_t{}.deallocate(index, 1);
        }
    }

    aligned_index_gt(aligned_index_gt const&) = delete;
    aligned_index_gt& operator=(aligned_index_gt const&) = delete;
};

// ---------------------------------------------------------------------------
// Mock prefetch: tracks invocations and caches vectors
// ---------------------------------------------------------------------------
template <typename vector_element_at = float> struct mock_prefetch_gt {
    mock_vector_store_gt<vector_element_at> const* store = nullptr;
    std::unordered_map<std::size_t, vector_element_at const*>* cache = nullptr;
    std::size_t* call_count = nullptr;

    template <typename member_citerator_like_at>
    inline void operator()(member_citerator_like_at begin, member_citerator_like_at end) const noexcept {
        if (call_count)
            ++(*call_count);
        if (cache && store) {
            for (auto it = begin; it != end; ++it) {
                auto slot = get_slot(it);
                (*cache)[slot] = store->row(slot);
            }
        }
    }
};

// Prefetch that verifies candidates_iterator_t postfix ++ semantics (PR #718).
// When the range has >= 2 elements, checks that it++ returns old position and advances *this.
struct candidates_iterator_postfix_prefetch_gt {
    bool* postfix_ok = nullptr; // set to false if postfix semantics check fails

    template <typename iterator_at> void operator()(iterator_at begin, iterator_at end) const noexcept {
        if (!postfix_ok)
            return;
        iterator_at it = begin;
        if (it == end)
            return;
        auto slot_before = get_slot(it);
        iterator_at prev = it++; // postfix: prev = old position, it = advanced
        if (it != end) {
            if (get_slot(prev) != slot_before)
                *postfix_ok = false;
            else if (get_slot(prev) == get_slot(it))
                *postfix_ok = false; // it must have advanced to a different slot
        }
    }
};

// ===========================================================================
// Test fixtures / parameterized tests
// ===========================================================================

struct index_gt_test_param_t {
    std::size_t collection_size;
    std::size_t dimensions;
    std::size_t connectivity;
    std::size_t wanted_count;
    std::size_t ef_construction;
    std::size_t ef_search;
};

class index_gt_test : public ::testing::TestWithParam<index_gt_test_param_t> {};

// ---------------------------------------------------------------------------
// TEST: Basic add + search
// ---------------------------------------------------------------------------
TEST_P(index_gt_test, add_and_search) {
    auto [collection_size, dimensions, connectivity, wanted_count, ef_construction, ef_search] = GetParam();

    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(collection_size, dimensions);
    metric_t metric{&store, dimensions};
    index_config_t config(connectivity);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(collection_size);

    index_update_config_t update_config;
    update_config.expansion = ef_construction;
    for (std::size_t i = 0; i < collection_size; ++i) {
        auto res = index.add(i, store.row(i), metric, update_config);
        ASSERT_TRUE(static_cast<bool>(res)) << "add failed at key=" << i;
    }

    EXPECT_EQ(index.size(), collection_size);

    // Search: query with the first vector, expect itself as nearest
    index_search_config_t search_config;
    search_config.expansion = ef_search;
    auto result = index.search(store.row(0), wanted_count, metric, search_config);
    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    EXPECT_EQ(result[0].member.key, vector_key_t{0}) << "query vector should be its own nearest neighbor";
}

// ---------------------------------------------------------------------------
// TEST: Distances are non-decreasing in search results
// ---------------------------------------------------------------------------
TEST_P(index_gt_test, search_distances_non_decreasing) {
    auto [collection_size, dimensions, connectivity, wanted_count, ef_construction, ef_search] = GetParam();

    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(collection_size, dimensions, 42);
    metric_t metric{&store, dimensions};
    index_config_t config(connectivity);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(collection_size);

    index_update_config_t update_config;
    update_config.expansion = ef_construction;
    for (std::size_t i = 0; i < collection_size; ++i)
        index.add(i, store.row(i), metric, update_config);

    index_search_config_t search_config;
    search_config.expansion = ef_search;
    auto result = index.search(store.row(0), wanted_count, metric, search_config);
    ASSERT_TRUE(static_cast<bool>(result));

    for (std::size_t i = 1; i < result.size(); ++i) {
        EXPECT_GE(result[i].distance, result[i - 1].distance) << "distances must be non-decreasing at rank " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(                                 //
    various_configs, index_gt_test,                       //
    ::testing::Values(                                    //
        index_gt_test_param_t{64, 16, 8, 10, 32, 32},     //
        index_gt_test_param_t{128, 32, 16, 16, 64, 64},   //
        index_gt_test_param_t{256, 64, 32, 20, 128, 128}, //
        index_gt_test_param_t{512, 128, 32, 16, 128, 128}));

// ===========================================================================
// Non-parameterized tests
// ===========================================================================

// ---------------------------------------------------------------------------
// TEST: Iterator covers all inserted keys
// ---------------------------------------------------------------------------
TEST(index_gt_basic, iterator_covers_all_keys) {
    constexpr std::size_t n = 50, d = 8, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    std::size_t count = 0;
    for (auto member : index) {
        (void)member;
        ++count;
    }
    EXPECT_EQ(count, n);
}

// ---------------------------------------------------------------------------
// TEST: member_iterator_gt postfix operator++/operator-- (STL-compliant, PR #718)
// ---------------------------------------------------------------------------
TEST(index_gt_basic, member_iterator_postfix_semantics) {
    constexpr std::size_t n = 5, d = 4, m = 4;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using member_iterator_t = typename index_t::member_iterator_t;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    index.add(10, store.row(0), metric);
    index.add(11, store.row(1), metric);
    index.add(12, store.row(2), metric);
    ASSERT_EQ(index.size(), 3u);

    member_iterator_t it = index.begin();
    ASSERT_NE(it, index.end());

    // Postfix ++: return value is current position, *this advances
    member_iterator_t it0 = it++;
    EXPECT_EQ(get_slot(it0), 0u);
    EXPECT_EQ(get_key(it0), vector_key_t(10));
    EXPECT_EQ(get_slot(it), 1u);
    EXPECT_EQ(get_key(it), vector_key_t(11));

    member_iterator_t it1 = it++;
    EXPECT_EQ(get_slot(it1), 1u);
    EXPECT_EQ(get_key(it1), vector_key_t(11));
    EXPECT_EQ(get_slot(it), 2u);
    EXPECT_EQ(get_key(it), vector_key_t(12));

    // Postfix --: return value is current position, *this retreats
    member_iterator_t it2 = it--;
    EXPECT_EQ(get_slot(it2), 2u);
    EXPECT_EQ(get_key(it2), vector_key_t(12));
    EXPECT_EQ(get_slot(it), 1u);
    EXPECT_EQ(get_key(it), vector_key_t(11));

    // *it++ semantics: dereference current, then advance
    vector_key_t key_at_1 = get_key(it);
    vector_key_t key_from_expr = get_key(it++);
    EXPECT_EQ(key_from_expr, key_at_1);
    EXPECT_EQ(get_slot(it), 2u);
}

// ---------------------------------------------------------------------------
// TEST: Empty index returns zero results
// ---------------------------------------------------------------------------
TEST(index_gt_basic, empty_search_returns_nothing) {
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(1, 4);
    metric_t metric{&store, 4};
    index_config_t config(4);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(1);

    auto result = index.search(store.row(0), 5, metric);
    EXPECT_EQ(result.size(), 0u);
}

// ---------------------------------------------------------------------------
// TEST: Single element returns itself
// ---------------------------------------------------------------------------
TEST(index_gt_basic, single_element_self_search) {
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(1, 8);
    metric_t metric{&store, 8};
    index_config_t config(4);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(1);
    index.add(42, store.row(0), metric);

    auto result = index.search(store.row(0), 5, metric);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].member.key, vector_key_t{42});
    EXPECT_NEAR(result[0].distance, 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// TEST: Search with wanted_count larger than collection
// ---------------------------------------------------------------------------
TEST(index_gt_basic, wanted_count_exceeds_size) {
    constexpr std::size_t n = 5, d = 4, m = 4;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    auto result = index.search(store.row(0), 100, metric);
    ASSERT_TRUE(static_cast<bool>(result));
    EXPECT_EQ(result.size(), n);
}

// ---------------------------------------------------------------------------
// TEST: Multiple connectivity values produce valid results
// ---------------------------------------------------------------------------
TEST(index_gt_basic, different_connectivity) {
    constexpr std::size_t n = 100, d = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 7);

    for (std::size_t connectivity : {2, 4, 8, 16, 32}) {
        metric_t metric{&store, d};
        index_config_t config(connectivity);

        aligned_index_gt<index_t> aligned(config);
        auto& index = *aligned.index;
        index.reserve(n);

        for (std::size_t i = 0; i < n; ++i)
            index.add(i, store.row(i), metric);

        auto result = index.search(store.row(0), 5, metric);
        ASSERT_TRUE(static_cast<bool>(result)) << "connectivity=" << connectivity;
        ASSERT_GT(result.size(), 0u) << "connectivity=" << connectivity;
        EXPECT_EQ(result[0].member.key, vector_key_t{0}) << "connectivity=" << connectivity;
    }
}

// ---------------------------------------------------------------------------
// TEST: index_update_config_t expansion affects quality
// ---------------------------------------------------------------------------
TEST(index_gt_basic, higher_ef_search_finds_closer) {
    constexpr std::size_t n = 200, d = 32, m = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 99);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);

    index_update_config_t add_config;
    add_config.expansion = 64;
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric, add_config);

    // Low ef_search
    index_search_config_t low_ef;
    low_ef.expansion = 4;
    auto res_low = index.search(store.row(0), 1, metric, low_ef);

    // High ef_search
    index_search_config_t high_ef;
    high_ef.expansion = 128;
    auto res_high = index.search(store.row(0), 1, metric, high_ef);

    // Both should find key 0 (itself), but high ef is more reliable
    ASSERT_TRUE(static_cast<bool>(res_high));
    ASSERT_GT(res_high.size(), 0u);
    EXPECT_EQ(res_high[0].member.key, vector_key_t{0});
}

// ---------------------------------------------------------------------------
// TEST: copy and move semantics
// ---------------------------------------------------------------------------
TEST(index_gt_basic, copy_index) {
    constexpr std::size_t n = 30, d = 8, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    // Copy
    auto copy_result = index.copy();
    ASSERT_TRUE(static_cast<bool>(copy_result));
    auto& copied = copy_result.index;
    EXPECT_EQ(copied.size(), n);

    auto res = copied.search(store.row(0), 3, metric);
    ASSERT_GT(res.size(), 0u);
    EXPECT_EQ(res[0].member.key, vector_key_t{0});

    // Move
    index_t moved(std::move(copied));
    EXPECT_EQ(moved.size(), n);
    auto res2 = moved.search(store.row(0), 3, metric);
    ASSERT_GT(res2.size(), 0u);
    EXPECT_EQ(res2[0].member.key, vector_key_t{0});
}

// ---------------------------------------------------------------------------
// TEST: Edge case - 1-dimensional vectors
// ---------------------------------------------------------------------------
TEST(index_gt_edge, one_dimensional) {
    constexpr std::size_t n = 10, d = 1, m = 2;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 123);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    EXPECT_EQ(index.size(), n);
    auto result = index.search(store.row(0), 5, metric);
    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
}

// ---------------------------------------------------------------------------
// TEST: Edge case - minimum connectivity (2)
// ---------------------------------------------------------------------------
TEST(index_gt_edge, minimum_connectivity) {
    constexpr std::size_t n = 20, d = 4;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 77);
    metric_t metric{&store, d};
    index_config_t config(2); // minimum connectivity

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    EXPECT_EQ(index.size(), n);
    auto result = index.search(store.row(0), 5, metric);
    ASSERT_TRUE(static_cast<bool>(result));
    EXPECT_GT(result.size(), 0u);
}

// ---------------------------------------------------------------------------
// TEST: Large batch (stress)
// ---------------------------------------------------------------------------
TEST(index_gt_stress, large_batch) {
    constexpr std::size_t n = 2048, d = 64, m = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 1234);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);

    index_update_config_t add_config;
    add_config.expansion = 64;
    for (std::size_t i = 0; i < n; ++i) {
        auto res = index.add(i, store.row(i), metric, add_config);
        ASSERT_TRUE(static_cast<bool>(res)) << "add failed at " << i;
    }
    EXPECT_EQ(index.size(), n);

    index_search_config_t search_config;
    search_config.expansion = 128;
    auto result = index.search(store.row(0), 10, metric, search_config);
    ASSERT_TRUE(static_cast<bool>(result));
    EXPECT_EQ(result[0].member.key, vector_key_t{0});
}

// ---------------------------------------------------------------------------
// TEST: Different key types (uint64_t key, uint32_t slot)
// ---------------------------------------------------------------------------
TEST(index_gt_types, uint64_key_uint32_slot) {
    constexpr std::size_t n = 50, d = 16, m = 8;
    using vector_key_t = std::uint64_t;
    using compressed_slot_t = std::uint32_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;

    using member_cref_t = typename index_t::member_cref_t;
    using member_citerator_t = typename index_t::member_citerator_t;

    mock_vector_store_gt<float> store(n, d, 55);

    struct metric_t {
        mock_vector_store_gt<float> const* store;
        std::size_t dimensions;

        float operator()(member_cref_t const& a, member_cref_t const& b) const {
            return metric_cos_gt<float>{}(store->row(get_slot(b)), store->row(get_slot(a)), dimensions);
        }
        float operator()(float const* query, member_cref_t const& member) const {
            return metric_cos_gt<float>{}(query, store->row(get_slot(member)), dimensions);
        }
        float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return metric_cos_gt<float>{}(store->row(get_slot(*b)), store->row(get_slot(*a)), dimensions);
        }
        float operator()(float const* query, member_citerator_t const& member) const {
            return metric_cos_gt<float>{}(query, store->row(get_slot(*member)), dimensions);
        }
    };

    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        vector_key_t key = 1000 + i;
        index.add(key, store.row(i), metric);
    }

    EXPECT_EQ(index.size(), n);
    auto result = index.search(store.row(0), 5, metric);
    ASSERT_GT(result.size(), 0u);
    EXPECT_EQ(result[0].member.key, vector_key_t{1000});
}

// ---------------------------------------------------------------------------
// TEST: stats() returns correct node count
// ---------------------------------------------------------------------------
TEST(index_gt_basic, stats_node_count) {
    constexpr std::size_t n = 25, d = 8, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 11);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    EXPECT_EQ(index.stats(0).nodes, n);
}

// ===========================================================================
// Prefetch tests
// ===========================================================================

// ---------------------------------------------------------------------------
// TEST: Prefetch is invoked during search and results remain correct
// ---------------------------------------------------------------------------
TEST(index_gt_prefetch, search_invokes_prefetch) {
    constexpr std::size_t n = 128, d = 32, m = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 42);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    std::size_t prefetch_calls = 0;
    mock_prefetch_gt<float> prefetch{&store, nullptr, &prefetch_calls};

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, metric, search_config, dummy_predicate_t{}, prefetch);

    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    EXPECT_EQ(result[0].member.key, vector_key_t{0});
    EXPECT_GT(prefetch_calls, 0u) << "prefetch should be invoked during search";
}

// ---------------------------------------------------------------------------
// TEST: candidates_iterator_t postfix operator++ (STL-compliant, PR #718)
// Search uses candidates_range_t; prefetch receives candidates_iterator_t.
// ---------------------------------------------------------------------------
TEST(index_gt_prefetch, candidates_iterator_postfix_semantics) {
    constexpr std::size_t n = 128, d = 32, m = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 42);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    bool postfix_ok = true;
    candidates_iterator_postfix_prefetch_gt prefetch{&postfix_ok};

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, metric, search_config, dummy_predicate_t{}, prefetch);

    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    EXPECT_TRUE(postfix_ok) << "candidates_iterator_t postfix ++ must return old position and advance *this (PR #718)";
}

// ---------------------------------------------------------------------------
// TEST: Prefetch is invoked during add
// ---------------------------------------------------------------------------
TEST(index_gt_prefetch, add_invokes_prefetch) {
    constexpr std::size_t n = 64, d = 16, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 77);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);

    std::size_t prefetch_calls = 0;
    mock_prefetch_gt<float> prefetch{&store, nullptr, &prefetch_calls};

    index_update_config_t update_config;
    update_config.expansion = 64;
    for (std::size_t i = 0; i < n; ++i) {
        auto res = index.add(i, store.row(i), metric, update_config, dummy_callback_t{}, prefetch);
        ASSERT_TRUE(static_cast<bool>(res)) << "add failed at key=" << i;
    }

    EXPECT_GT(prefetch_calls, 0u) << "prefetch should be invoked during add";

    // Verify index correctness after add-with-prefetch
    auto result = index.search(store.row(0), 5, metric);
    ASSERT_TRUE(static_cast<bool>(result));
    EXPECT_EQ(result[0].member.key, vector_key_t{0});
}

// ---------------------------------------------------------------------------
// TEST: Search with prefetch-populated cache metric
// ---------------------------------------------------------------------------
TEST(index_gt_prefetch, search_with_cache_metric) {
    constexpr std::size_t n = 128, d = 32, m = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using add_metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    using member_cref_t = typename index_t::member_cref_t;
    using member_citerator_t = typename index_t::member_citerator_t;

    mock_vector_store_gt<float> store(n, d, 99);
    add_metric_t add_metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), add_metric);

    // Cache populated by prefetch during search traversal
    std::unordered_map<std::size_t, float const*> cache;
    std::size_t prefetch_calls = 0;
    mock_prefetch_gt<float> prefetch{&store, &cache, &prefetch_calls};

    // Search metric: reads from cache, falls back to store
    struct search_metric_t {
        std::unordered_map<std::size_t, float const*>* cache;
        mock_vector_store_gt<float> const* fallback;
        std::size_t dimensions;

        float const* row(std::size_t slot) const noexcept {
            auto it = cache->find(slot);
            return (it != cache->end()) ? it->second : fallback->row(slot);
        }

        float operator()(member_cref_t const& a, member_cref_t const& b) const {
            return metric_cos_gt<float>{}(row(get_slot(b)), row(get_slot(a)), dimensions);
        }
        float operator()(float const* query, member_cref_t const& member) const {
            return metric_cos_gt<float>{}(query, row(get_slot(member)), dimensions);
        }
        float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return metric_cos_gt<float>{}(row(get_slot(*b)), row(get_slot(*a)), dimensions);
        }
        float operator()(float const* query, member_citerator_t const& member) const {
            return metric_cos_gt<float>{}(query, row(get_slot(*member)), dimensions);
        }
    };

    search_metric_t search_metric{&cache, &store, d};

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, search_metric, search_config, dummy_predicate_t{}, prefetch);

    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    EXPECT_EQ(result[0].member.key, vector_key_t{0});
    EXPECT_GT(prefetch_calls, 0u) << "prefetch should have been invoked";
    EXPECT_GT(cache.size(), 0u) << "cache should have been populated";
}

// ===========================================================================
// Predicate tests
// ===========================================================================

// ---------------------------------------------------------------------------
// TEST: Search with predicate filtering even keys only
// ---------------------------------------------------------------------------
TEST(index_gt_predicate, even_key_filter) {
    constexpr std::size_t n = 100, d = 16, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 33);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    // Predicate: accept only members with even keys
    auto even_predicate = [](auto&& member) noexcept -> bool { return (get_key(member) % 2) == 0; };

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, metric, search_config, even_predicate);

    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i].member.key % 2, 0u) << "result[" << i << "].key=" << result[i].member.key << " is odd";
    }
}

// ---------------------------------------------------------------------------
// TEST: Predicate that rejects everything returns empty results
// ---------------------------------------------------------------------------
TEST(index_gt_predicate, reject_all_returns_empty) {
    constexpr std::size_t n = 50, d = 8, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 11);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    auto reject_all = [](auto&&) noexcept -> bool { return false; };

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, metric, search_config, reject_all);

    EXPECT_EQ(result.size(), 0u) << "reject-all predicate should yield nothing";
}

// ---------------------------------------------------------------------------
// TEST: Predicate excludes the query vector itself from results
// ---------------------------------------------------------------------------
TEST(index_gt_predicate, exclude_query_self) {
    constexpr std::size_t n = 100, d = 16, m = 8;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 44);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    // Exclude key 0 (the query vector itself)
    auto exclude_zero = [](auto&& member) noexcept -> bool { return get_key(member) != 0; };

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, metric, search_config, exclude_zero);

    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_NE(result[i].member.key, vector_key_t{0}) << "key 0 should be excluded by predicate";
    }
}

// ---------------------------------------------------------------------------
// TEST: Predicate with prefetch combined
// ---------------------------------------------------------------------------
TEST(index_gt_predicate, with_prefetch) {
    constexpr std::size_t n = 128, d = 32, m = 16;
    using vector_key_t = std::size_t;
    using compressed_slot_t = std::size_t;
    using index_t = index_gt<float, vector_key_t, compressed_slot_t>;
    using metric_t = cosine_metric_gt<float, vector_key_t, compressed_slot_t>;

    mock_vector_store_gt<float> store(n, d, 55);
    metric_t metric{&store, d};
    index_config_t config(m);

    aligned_index_gt<index_t> aligned(config);
    auto& index = *aligned.index;
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        index.add(i, store.row(i), metric);

    std::size_t prefetch_calls = 0;
    mock_prefetch_gt<float> prefetch{&store, nullptr, &prefetch_calls};

    // Only allow keys divisible by 3
    auto mod3_predicate = [](auto&& member) noexcept -> bool { return (get_key(member) % 3) == 0; };

    index_search_config_t search_config;
    search_config.expansion = 64;
    auto result = index.search(store.row(0), 10, metric, search_config, mod3_predicate, prefetch);

    ASSERT_TRUE(static_cast<bool>(result));
    ASSERT_GT(result.size(), 0u);
    EXPECT_GT(prefetch_calls, 0u) << "prefetch should be invoked";
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i].member.key % 3, 0u)
            << "result[" << i << "].key=" << result[i].member.key << " is not divisible by 3";
    }
}
