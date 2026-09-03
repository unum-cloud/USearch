/**
 *  @file       test.cpp
 *  @author     Ash Vardanian
 *  @brief      Unit-testing vector-search functionality.
 *  @date       June 10, 2023
 *
 *
 *  Key and slot types:
 *      - 64-bit `std::int64_t` keys and `std::uint32_t` slots are most popular.
 *      - 64-bit `std::uint64_t` keys and `std::uint40_t` are most space-efficient for
 *        point clouds 4B+ in size.
 *      - 128-bit `uuid_t` keys and `enum slot64_t : std::uint64_t` make most sense for
 *        for database users, implementing portable, concurrent systems.
 */
#include <cassert> // `assert`
#include <cmath>   // `std::abs`
#include <csignal> // `std::signal`, `SIGSEGV`, ...
#include <cstdio>  // `std::fprintf`
#include <cstdlib> // `std::_Exit`
#include <limits>  // `std::numeric_limits`

#include <algorithm>     // `std::shuffle`
#include <atomic>        // `std::atomic`
#include <random>        // `std::default_random_engine`
#include <stdexcept>     // `std::terminate`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

// Back-trace support. Prefer the C++23 `<stacktrace>` library when the
// toolchain + stdlib expose it (`__cpp_lib_stacktrace`); otherwise fall back
// to the OS-native facility so that unit-test crashes in CI log something
// useful beyond a bare exit code.
#if defined(__has_include)
#if __has_include(<stacktrace>)
#include <stacktrace>
#endif
#endif
#if defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
#define USEARCH_HAS_STD_STACKTRACE 1
#else
#define USEARCH_HAS_STD_STACKTRACE 0
#if defined(_WIN32)
// `windows.h` must precede `dbghelp.h` — the latter uses `PSTR` and friends
// that are only defined after `windows.h`. The blank line keeps clang-format
// from re-sorting the two headers into a single alphabetized block.
#include <windows.h>

#include <dbghelp.h>
#pragma comment(lib, "Dbghelp.lib")
#else
#include <execinfo.h>
#include <unistd.h>
#endif
#endif

#define SZ_USE_X86_AVX512 0            // Sanitizers hate AVX512
#include <stringzilla/stringzilla.hpp> // Levenshtein distance implementation

/**
 *  Counters for the cosine norm cache diagnostic hook in `index_dense.hpp`. They let the tests assert that the
 *  sidecar route is actually taken, not just that results are correct, which a fallback would also deliver.
 */
struct cosine_cache_diagnostics_t {
    // Atomic so that the concurrent add/search test is race-free under ThreadSanitizer.
    std::atomic<std::size_t> cache_hit_{0}, invalid_fallback_{0}, population_{0}, route_bypass_{0},
        query_preparation_{0};
    std::size_t cache_hit() const { return cache_hit_.load(); }
    std::size_t invalid_fallback() const { return invalid_fallback_.load(); }
    std::size_t population() const { return population_.load(); }
    std::size_t route_bypass() const { return route_bypass_.load(); }
    std::size_t query_preparation() const { return query_preparation_.load(); }
    void reset() { cache_hit_ = invalid_fallback_ = population_ = route_bypass_ = query_preparation_ = 0; }
};
static cosine_cache_diagnostics_t cosine_cache_diagnostics;
#define USEARCH_COSINE_NORM_CACHE_DIAGNOSTIC(event) (++cosine_cache_diagnostics.event##_)

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;
using namespace unum;

void __expect(bool must_be_true, char const* file, int line, char const* message = nullptr) {
    if (must_be_true)
        return;
    message = message ? message : "C++ unit test failed";
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), "%s at %s:%d", message, file, line);
    usearch_raise_runtime_error(buffer);
}

template <typename value_at>
void __expect_eq(value_at a, value_at b, char const* file, int line, char const* message = nullptr) {
    __expect(a == b, file, line, message);
}

#define expect(cond) __expect((bool)(cond), __FILE__, __LINE__)
#define expect_eq(a, b) __expect_eq<decltype(a)>((a), (b), __FILE__, __LINE__)

/**
 *  Less error-prone type definition to differentiate `std::uint32_t` and other native
 *  types in logs and avoid implicit conversions.
 */
enum slot32_t : std::uint32_t {};
template <> struct unum::usearch::hash_gt<slot32_t> : public unum::usearch::hash_gt<std::uint32_t> {};
template <> struct unum::usearch::default_free_value_gt<slot32_t> {
    static slot32_t value() noexcept { return static_cast<slot32_t>((std::numeric_limits<std::uint32_t>::max)()); }
};

/*
 *  Let's instantiate several templates to make all of their symbols available for testing.
 *  https://dhashe.com/how-to-build-highly-debuggable-c-binaries.html
 */
template class unum::usearch::index_gt<float, std::int64_t, slot32_t>;
template class unum::usearch::index_gt<float, std::int64_t, uint40_t>;
template class unum::usearch::index_dense_gt<std::int64_t, slot32_t>;
template class unum::usearch::index_dense_gt<std::int64_t, uint40_t>;

/**
 *  @brief  Convenience wrapper combining combined allocation and construction of an index.
 */
template <typename index_at> struct aligned_wrapper_gt {
    using index_t = index_at;
    using alloc_t = aligned_allocator_gt<index_t, 64>;

    alloc_t alloc;
    index_t* index = nullptr;

    template <typename... args_at> aligned_wrapper_gt(args_at&&... args) {

        alloc_t index_alloc;
        index_t* index_typed = index_alloc.allocate(1);
        expect(index_typed != nullptr);
        expect(((unsigned long long)(index_typed) % 64ull) == 0ull);

        new (index_typed) index_t(std::forward<args_at>(args)...);
        index = index_typed;
    }

    ~aligned_wrapper_gt() {
        if (index != nullptr) {
            index->~index_t();
            alloc.deallocate(index, 1);
        }
    }
};

/**
 * Tests the functionality of the custom uint40_t type ensuring consistent
 * behavior across various constructors from uint32_t, uint64_t, and size_t types,
 * and validates the relational ordering between various values.
 */
void test_uint40() {

    union uint64_octets_t {
        std::uint64_t value;
        std::uint8_t octets[8];
    };

    // Constants for tests
    std::uint64_t max_uint40_k = (1ULL << 40) - 1;

    // Set of test numbers
    std::vector<std::uint64_t> test_numbers = {
        42ull,            // Typical small number
        4242ull,          // Larger number still within uint40 range
        (1ull << 39),     // A high number within range
        (1ull << 40) - 1, // Maximum value representable in uint40
        1ull << 40,       // Exactly at the boundary of uint40
        (1ull << 40) + 1, // Just beyond the boundary of uint40
        1ull << 63        // Well beyond the uint40 boundary, tests masking
    };

    for (std::uint64_t input_u64 : test_numbers) {
        std::uint32_t input_u32 = static_cast<std::uint32_t>(input_u64);
        std::size_t input_size = static_cast<std::size_t>(input_u64);

        // Create uint40_t instances from different types
        uint40_t u40_from_u32(input_u32);
        uint40_t u40_from_u64(input_u64);
        uint40_t u40_from_size(input_size);

        // Expected value after masking
        uint64_octets_t input_clamped;
        input_clamped.value = input_u64 & max_uint40_k;

        // Check if all conversions are equal to the masked value
        expect_eq(u40_from_u32, input_clamped.value & 0xFFFFFFFF);
        expect_eq(u40_from_u64, input_clamped.value);
        expect_eq(u40_from_size, input_clamped.value);

        // Check relative ordering against all other test numbers
        for (std::uint64_t other_u64 : test_numbers) {
            uint64_octets_t other_clamped;
            other_clamped.value = other_u64 & max_uint40_k;
            uint40_t other_u40(other_clamped.value);

            // Check < and >
            expect_eq(input_clamped.value < other_clamped.value, u40_from_u64 < other_u40);
            expect_eq(input_clamped.value > other_clamped.value, u40_from_u64 > other_u40);

            // Check <= and >=
            expect_eq(input_clamped.value <= other_clamped.value, u40_from_u64 <= other_u40);
            expect_eq(input_clamped.value >= other_clamped.value, u40_from_u64 >= other_u40);
        }
    }

    // Test equality and inequality operators
    for (std::uint64_t input_u64 : test_numbers) {
        uint64_octets_t input_clamped;
        input_clamped.value = input_u64 & max_uint40_k;
        uint40_t u40(input_clamped.value);

        expect_eq(u40 == uint40_t(input_clamped.value), true);
        expect_eq(u40 != uint40_t(input_clamped.value + 1), true);
    }

    // Test min and max functions
    expect_eq((uint40_t::min)(), uint40_t(0u));
    expect_eq((uint40_t::max)(), uint40_t(max_uint40_k));

    // Test copy and move semantics
    for (std::uint64_t input_u64 : test_numbers) {
        uint40_t u40_orig(input_u64);

        uint40_t u40_copy(u40_orig);
        expect_eq(u40_orig, u40_copy);

        uint40_t u40_move(std::move(u40_orig));
        expect_eq(u40_copy, u40_move);
    }

    // Test default constructor (zero initialization)
    uint40_t u40_default;
    expect_eq(u40_default, uint40_t(0u));
}

void test_checked_size_arithmetic() {

    std::printf("Testing checked size arithmetic\n");
    std::size_t max = (std::numeric_limits<std::size_t>::max)();

    checked_size_result_t cast = checked_size_from_u64(42);
    expect(cast);
    expect_eq(cast.value, static_cast<std::size_t>(42));
    if (sizeof(std::size_t) < sizeof(std::uint64_t))
        expect(!checked_size_from_u64((std::numeric_limits<std::uint64_t>::max)()));

    checked_size_result_t sum = checked_add(max - 1, 1);
    expect(sum);
    expect_eq(sum.value, max);
    expect(!checked_add(max, 1));

    checked_size_result_t product = checked_mul(max / 2, 2);
    expect(product);
    expect_eq(product.value, max - 1);
    expect(!checked_mul(max / 2 + 1, 2));

    checked_size_result_t fused = checked_mul_add(10, 20, 30);
    expect(fused);
    expect_eq(fused.value, static_cast<std::size_t>(230));
    expect(!checked_mul_add(max / 2 + 1, 2, 0));
    expect(!checked_mul_add(max / 2, 2, 2));

    checked_size_result_t rounded = checked_round_up(17, 8);
    expect(rounded);
    expect_eq(rounded.value, static_cast<std::size_t>(24));
    expect(!checked_round_up(max, 8));

    checked_size_result_t power = checked_ceil2(17);
    expect(power);
    expect_eq(power.value, static_cast<std::size_t>(32));
    checked_size_result_t zero_power = checked_ceil2(0);
    expect(zero_power);
    expect_eq(zero_power.value, static_cast<std::size_t>(0));
    expect(!checked_ceil2(max));
}

/**
 *  @brief  Tests the functionality of the custom float16_t type ensuring consistent.
 */
void test_float16() {}

/**
 * The goal of this test is to invoke as many different interfaces as possible, making sure that all code-paths compile.
 * For that it only uses a tiny set of 3 predefined vectors.
 *
 * @param index Reference to the index where vectors will be stored and searched.
 * @param vectors A collection of vectors to be tested.
 * @param args Additional arguments for configuring search or index operations.
 * @tparam punned_ak Template parameter that determines specific behaviors or checks in the test based on its value.
 * @tparam index_at Type of the index being tested.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam extra_args_at Variadic template parameter types for additional configuration.
 */
template <bool punned_ak, typename index_at, typename scalar_at, typename... extra_args_at>
void test_minimal_three_vectors(index_at& index, //
                                typename index_at::vector_key_t key_first, std::vector<scalar_at> const& vector_first,
                                typename index_at::vector_key_t key_second, std::vector<scalar_at> const& vector_second,
                                typename index_at::vector_key_t key_third, std::vector<scalar_at> const& vector_third,
                                extra_args_at&&... args) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;
    using distance_t = typename index_t::distance_t;

    // Try checking the empty state
    if constexpr (punned_ak) {
        if (index.config().enable_key_lookups) {
            expect(!index.contains(key_first));
            expect(!index.get(key_first, (f32_t*)nullptr, 1));
        }
    }

    // Add data
    expect(index.try_reserve(10));
    expect(index.add(key_first, vector_first.data(), args...));

    // Default approximate search
    vector_key_t matched_keys[10] = {0};
    distance_t matched_distances[10] = {0};
    std::size_t matched_count = index.search(vector_first.data(), 5, args...).dump_to(matched_keys, matched_distances);

    expect(matched_count == 1);
    expect(matched_keys[0] == key_first);
    expect(std::abs(matched_distances[0]) < 0.01);

    // Add more entries
    index.add(key_second, vector_second.data(), args...);
    index.add(key_third, vector_third.data(), args...);
    expect(index.size() == 3);

    // Perform single entry search
    {
        auto search_result = index.search(vector_first.data(), 5, args...);
        expect(search_result);
        matched_count = search_result.dump_to(matched_keys, matched_distances);
        expect(matched_count != 0);
    }

    // Perform filtered exact search, keeping only odd values
    if constexpr (punned_ak) {
        auto is_odd = [](vector_key_t key) -> bool { return (key & 1) != 0; };
        auto search_result = index.filtered_search(vector_first.data(), 5, is_odd, args...);
        expect(search_result);
        matched_count = search_result.dump_to(matched_keys, matched_distances);
        expect(matched_count != 0);
        for (std::size_t i = 0; i < matched_count; i++)
            expect(is_odd(matched_keys[i]));
    }

    // Validate scans
    std::size_t count = 0;
    for (auto member : index) {
        vector_key_t id = member.key;
        expect(id >= key_first && id <= key_third);
        count++;
    }
    expect((count == 3));
    expect((index.stats(0).nodes == 3));

    // Check if clustering endpoint compiles
    index.cluster(vector_first.data(), 0, args...);

    // Try removals and replacements
    if constexpr (punned_ak) {
        if (index.config().enable_key_lookups) {
            using labeling_result_t = typename index_t::labeling_result_t;
            labeling_result_t result = index.remove(key_third);
            expect(result);
            expect(index.size() == 2);
            index.add(key_third, vector_third.data(), args...);
            expect(index.size() == 3);
        }
    }

    expect(index.save("tmp.usearch"));

    // Perform content and scan validations for a copy
    {
        auto copy_result = index.copy();
        expect(copy_result);
        index_at copied_index = std::move(copy_result.index);

        // Perform single entry search
        auto search_result = copied_index.search(vector_first.data(), 5, args...);
        expect(search_result);
        matched_count = search_result.dump_to(matched_keys, matched_distances);
        expect(matched_count != 0);

        // Validate scans
        std::size_t count = 0;
        for (auto member : copied_index) {
            vector_key_t id = member.key;
            expect(id >= key_first && id <= key_third);
            count++;
        }
        expect_eq(count, 3);
        expect_eq(copied_index.stats(0).nodes, 3);
    }

    // Perform content and scan validations for a moved
    {
        index_at moved_index(std::move(index));

        // Perform single entry search
        auto search_result = moved_index.search(vector_first.data(), 5, args...);
        expect(search_result);
        matched_count = search_result.dump_to(matched_keys, matched_distances);
        expect(matched_count != 0);

        // Validate scans
        std::size_t count = 0;
        for (auto member : moved_index) {
            vector_key_t id = member.key;
            expect(id >= key_first && id <= key_third);
            count++;
        }
        expect_eq(count, 3);
        expect_eq(moved_index.stats(0).nodes, 3);
    }

    // Check if metadata is retrieved correctly
    if constexpr (punned_ak) {
        auto head_result = index_dense_metadata_from_path("tmp.usearch");
        expect(head_result);
        expect_eq(3ull, head_result.head.count_present);
    }

    // Try loading and move assignment
    {
        index_at loaded_index;
        auto load_result = loaded_index.load("tmp.usearch");
        expect(load_result);
        index = std::move(loaded_index);
    }

    // Check the copy of the restored index
    {
        auto copy_result = index.copy();
        expect(copy_result);
        index_at copied_index = std::move(copy_result.index);
        expect_eq(copied_index.size(), 3);
    }

    // Search again over reconstructed index
    {
        matched_count = index.search(vector_first.data(), 5, args...).dump_to(matched_keys, matched_distances);
        expect_eq(matched_count, 3);
        expect_eq(matched_keys[0], key_first);
        expect(std::abs(matched_distances[0]) < 0.01);
    }

    // Try retrieving a vector from a deserialized index
    if constexpr (punned_ak) {
        if (index.config().enable_key_lookups) {
            std::size_t dimensions = vector_first.size();
            std::vector<scalar_t> vector_reloaded(dimensions);
            expect(index.get(key_second, vector_reloaded.data()));
            expect(std::equal(vector_second.data(), vector_second.data() + dimensions, vector_reloaded.data()));
        }
    }
}

/**
 *  @brief  Tests value removals, by repeatedly adding and removing a couple of vectors.
 *
 *  @param index Reference to the index where vectors will be stored and searched.
 *  @param vector_first First vector to be tested.
 *  @param vector_second Second vector to be tested.
 *  @param args Additional arguments for configuring search or index operations.
 */
template <typename index_at, typename scalar_at>
void test_punned_add_remove_vector(             //
    index_at& index,                            //
    std::vector<scalar_at> const& vector_first, //
    std::vector<scalar_at> const& vector_second) {

    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;

    // Creating the index
    expect(index.try_reserve(10));
    expect(index.capacity() >= 10);

    // Max 64-bit unsigned key value is: 18446744073709551615
    vector_key_t key_first = 483367403120493160;
    vector_key_t key_second = 483367403120558696;
    vector_key_t key_third = 483367403120624232;
    vector_key_t key_fourth = 483367403120624233;

    // Adding, getting, and removing vectors
    expect(index.add(key_first, vector_first.data()));
    std::vector<float> found_slice(vector_first.size(), 0.0f);
    expect_eq(index.get(key_first, found_slice.data()), 1);
    expect(index.remove(key_first));

    expect(index.add(key_second, vector_second.data()));
    expect_eq(index.get(key_second, found_slice.data()), 1);
    expect(index.remove(key_second));

    expect(index.add(key_third, vector_second.data()));
    expect_eq(index.get(key_third, found_slice.data()), 1);
    expect(index.remove(key_third));

    expect(index.add(key_fourth, vector_second.data()));
    expect_eq(index.get(key_fourth, found_slice.data()), 1);
    expect(index.remove(key_fourth));

    expect_eq(index.size(), 0);
}

/**
 * Tests the normal operational mode of the library, dealing with a variable length collection
 * of `vectors` with monotonically increasing keys starting from `start_key`.
 *
 * @param index Reference to the index where vectors will be stored and searched.
 * @param start_key The key for the first `vector`, others are generated with increments.
 * @param vectors A collection of vectors to be tested.
 * @param args Additional arguments for configuring search or index operations.
 * @tparam punned_ak Template parameter that determines specific behaviors or checks in the test based on its value.
 * @tparam index_at Type of the index being tested.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam extra_args_at Variadic template parameter types for additional configuration.
 */
template <bool punned_ak, typename index_at, typename scalar_at, typename... extra_args_at>
void test_collection(index_at& index, typename index_at::vector_key_t const start_key,
                     std::vector<std::vector<scalar_at>> const& vectors, extra_args_at&&... args) {

    using scalar_t = scalar_at;
    using index_t = index_at;
    using vector_key_t = typename index_t::vector_key_t;
    using distance_t = typename index_t::distance_t;
    using index_add_result_t = typename index_t::add_result_t;
    using index_search_result_t = typename index_t::search_result_t;

    // Generate some keys starting from end,
    // for three vectors from the dataset
    vector_key_t const key_first = start_key;
    std::vector<scalar_at> const& vector_first = vectors[0];
    std::size_t dimensions = vector_first.size();

    // Try batch requests, heavily over-subscribing the CPU cores
    std::size_t executor_threads = std::thread::hardware_concurrency();
    executor_default_t executor(executor_threads);
    expect(index.try_reserve({vectors.size(), executor.size()}));
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        if constexpr (punned_ak) {
            index_add_result_t result = index.add(start_key + task, vectors[task].data(), args...);
            expect(result);
        } else {
            index_update_config_t config;
            config.thread = thread;
            index_add_result_t result = index.add(start_key + task, vectors[task].data(), args..., config);
            expect(result);
        }
    });

    // Make sure we didn't lose parallelism settings after reload
    expect(index.limits().threads_search >= executor.size());
    if constexpr (punned_ak)
        expect(index.currently_available_threads() >= executor.size());

    // Parallel search over the same vectors
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        std::size_t max_possible_matches = vectors.size();
        std::size_t count_requested = max_possible_matches;
        std::vector<vector_key_t> matched_keys(count_requested);
        std::vector<distance_t> matched_distances(count_requested);
        std::size_t matched_count = 0;

        // Invoke the search kernel
        if constexpr (punned_ak) {
            index_search_result_t result = index.search(vectors[task].data(), count_requested, args...);
            expect(result);
            matched_count = result.dump_to(matched_keys.data(), matched_distances.data());
        } else {
            index_search_config_t config;
            config.thread = thread;
            index_search_result_t result = index.search(vectors[task].data(), count_requested, args..., config);
            expect(result);
            matched_count = result.dump_to(matched_keys.data(), matched_distances.data());
        }

        // In approximate search we can't always expect the right answer to be found
        //      expect_eq(matched_count, max_possible_matches);
        //      expect_eq(matched_keys[0], start_key + task);
        //      expect(std::abs(matched_distances[0]) < 0.01);
        expect(matched_count <= max_possible_matches);

        // Check that all the distance are monotonically rising
        for (std::size_t i = 1; i < matched_count; i++)
            expect(matched_distances[i - 1] <= matched_distances[i]);
    });

    // Search again over mapped index
    expect(index.save("tmp.usearch"));

    {
        auto copy_result = index.copy();
        expect(copy_result);
        index_at copied_index = std::move(copy_result.index);
        expect_eq(copied_index.size(), vectors.size());
    }

    // Check for duplicates
    if constexpr (punned_ak) {
        if (index.config().enable_key_lookups) {
            expect(index.try_reserve({vectors.size() + 1u, executor.size()}));
            index_add_result_t result = index.add(key_first, vector_first.data(), args...);
            expect_eq(!!result, index.multi());
            result.error.release();

            std::size_t first_key_count = index.count(key_first);
            expect_eq(first_key_count, 1ul + index.multi());
        }
    }

    // Recover the state before the duplicate insertion
    expect(index.view("tmp.usearch"));

    // Parallel search over the same vectors
    executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
        // Check over-sampling beyond the size of the collection
        std::size_t max_possible_matches = vectors.size();
        std::size_t count_requested = max_possible_matches * 10;
        std::vector<vector_key_t> matched_keys(count_requested);
        std::vector<distance_t> matched_distances(count_requested);
        std::size_t matched_count = 0;

        // Invoke the search kernel
        if constexpr (punned_ak) {
            index_search_result_t result = index.search(vectors[task].data(), count_requested, args...);
            expect(result);
            matched_count = result.dump_to(matched_keys.data(), matched_distances.data());
        } else {
            index_search_config_t config;
            config.thread = thread;
            index_search_result_t result = index.search(vectors[task].data(), count_requested, args..., config);
            expect(result);
            matched_count = result.dump_to(matched_keys.data(), matched_distances.data());
        }

        // In approximate search we can't always expect the right answer to be found
        //      expect_eq(matched_count, max_possible_matches);
        //      expect_eq(matched_keys[0], start_key + task);
        //      expect(std::abs(matched_distances[0]) < 0.01);
        expect(matched_count <= max_possible_matches);

        // Check that all the distance are monotonically rising
        for (std::size_t i = 1; i < matched_count; i++)
            expect(matched_distances[i - 1] <= matched_distances[i]);
    });

    // Try retrieving a vector from a deserialized index
    if constexpr (punned_ak) {
        if (index.config().enable_key_lookups) {
            expect(index.contains(key_first));
            expect_eq(index.count(key_first), 1);
            std::vector<scalar_t> vector_reloaded(dimensions);
            index.get(key_first, vector_reloaded.data());
            expect(std::equal(vector_first.data(), vector_first.data() + dimensions, vector_reloaded.data()));
        }

        auto compaction_result = index.compact();
        expect(compaction_result);
    }

    expect(index.memory_usage() > 0);
    expect(index.stats().max_edges > 0);

    // Check metadata
    if constexpr (punned_ak) {
        index_dense_metadata_result_t meta = index_dense_metadata_from_path("tmp.usearch");
        expect(meta);
    }
}

/**
 * Stress-tests the behavior of the type-punned higher-level index under heavy concurrent insertions,
 * removals and updates.
 *
 * @param index Reference to the index where vectors will be stored and searched.
 * @param start_key The key for the first `vector`, others are generated with increments.
 * @param vectors A collection of vectors to be tested.
 * @param executor_threads Number of threads to be used for concurrent operations.
 * @tparam punned_ak Template parameter that determines specific behaviors or checks in the test based on its value.
 * @tparam index_at Type of the index being tested.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam extra_args_at Variadic template parameter types for additional configuration.
 */
template <typename index_at, typename scalar_at, typename... extra_args_at>
void test_punned_concurrent_updates(index_at& index, typename index_at::vector_key_t const start_key,
                                    std::vector<std::vector<scalar_at>> const& vectors, std::size_t executor_threads) {

    using index_t = index_at;

    // Try batch requests, heavily oversubscribing the CPU cores
    executor_default_t executor(executor_threads);
    expect(index.try_reserve({vectors.size(), executor.size()}));
    executor.fixed(vectors.size(), [&](std::size_t, std::size_t task) {
        using add_result_t = typename index_t::add_result_t;
        add_result_t result = index.add(start_key + task, vectors[task].data());
        expect(result);
    });
    expect_eq(index.size(), vectors.size());

    // Without key lookups we can't do much more
    if (!index.config().enable_key_lookups)
        return;

    // Remove all the keys
    executor.fixed(vectors.size(), [&](std::size_t, std::size_t task) {
        using labeling_result_t = typename index_t::labeling_result_t;
        labeling_result_t result = index.remove(start_key + task);
        expect(result);
    });
    expect_eq(index.size(), 0);

    // Add them back, which under the hood will trigger the `update`
    executor.fixed(vectors.size(), [&](std::size_t, std::size_t task) {
        using add_result_t = typename index_t::add_result_t;
        add_result_t result = index.add(start_key + task, vectors[task].data());
        expect(result);
    });
    expect_eq(index.size(), vectors.size());
}

/**
 * Overloaded function to test cosine similarity index functionality using specific scalar, key, and slot types.
 *
 * This function initializes vectors and an index instance to test cosine similarity calculations and index operations.
 * It involves creating vectors with random values, constructing an index, and verifying that the index operations
 * like search work correctly with respect to the cosine similarity metric.
 *
 * @param collection_size Number of vectors to be included in the test.
 * @param dimensions Number of dimensions each vector should have.
 * @tparam scalar_at Data type of the elements in the vectors.
 * @tparam key_at Data type used for the keys in the index.
 * @tparam slot_at Data type used for slots in the index.
 */
template <typename scalar_at, typename key_at, typename slot_at> //
void test_cosine(std::size_t collection_size, std::size_t dimensions) {

    using scalar_t = scalar_at;
    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_typed_t = index_gt<float, vector_key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    using vector_of_vectors_t = std::vector<std::vector<scalar_at>>;
    vector_of_vectors_t vector_of_vectors(collection_size);
    for (auto& vector : vector_of_vectors) {
        vector.resize(dimensions);
        std::generate(vector.begin(), vector.end(), [=] { return float(std::rand()) / float(RAND_MAX); });
    }

    struct metric_t {
        vector_of_vectors_t const* vector_of_vectors_ptr = nullptr;
        std::size_t dimensions = 0;

        scalar_t const* row(std::size_t i) const noexcept { return (*vector_of_vectors_ptr)[i].data(); }

        float operator()(member_cref_t const& a, member_cref_t const& b) const {
            return metric_cos_gt<scalar_t, float>{}(row(get_slot(b)), row(get_slot(a)), dimensions);
        }
        float operator()(scalar_t const* some_vector, member_cref_t const& member) const {
            return metric_cos_gt<scalar_t, float>{}(some_vector, row(get_slot(member)), dimensions);
        }
        float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return metric_cos_gt<scalar_t, float>{}(row(get_slot(b)), row(get_slot(a)), dimensions);
        }
        float operator()(scalar_t const* some_vector, member_citerator_t const& member) const {
            return metric_cos_gt<scalar_t, float>{}(some_vector, row(get_slot(member)), dimensions);
        }
    };

    // Template:
    auto run_templated = [&](std::size_t connectivity) {
        std::printf("-- templates with connectivity %zu \n", connectivity);
        metric_t metric{&vector_of_vectors, dimensions};
        index_config_t config(connectivity);

        // Toy example
        if (vector_of_vectors.size() >= 3) {
            aligned_wrapper_gt<index_typed_t> aligned_index(config);
            test_minimal_three_vectors<false, index_typed_t>( //
                *aligned_index.index,                         //
                42, vector_of_vectors[0],                     //
                43, vector_of_vectors[1],                     //
                44, vector_of_vectors[2], metric);
        }
        // Larger collection
        {
            aligned_wrapper_gt<index_typed_t> aligned_index(config);
            test_collection<false, index_typed_t>(*aligned_index.index, 42, vector_of_vectors, metric);
        }
    };
    for (std::size_t connectivity : {3, 13, 50})
        run_templated(connectivity);

    // Type-punned:
    auto run_punned = [&](bool multi, bool enable_key_lookups, std::size_t connectivity) {
        std::printf("-- punned with connectivity %zu, multi: %s, lookups: %s \n", connectivity, multi ? "yes" : "no",
                    enable_key_lookups ? "yes" : "no");
        using index_t = index_dense_gt<vector_key_t, slot_t>;
        using index_result_t = typename index_t::state_result_t;
        metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind<scalar_at>());
        index_dense_config_t config(connectivity);
        config.multi = multi;
        config.enable_key_lookups = enable_key_lookups;

        // Toy example
        if (vector_of_vectors.size() >= 3) {
            index_result_t index_result = index_t::make(metric, config);
            test_minimal_three_vectors<true>( //
                index_result.index,           //
                42, vector_of_vectors[0],     //
                43, vector_of_vectors[1],     //
                44, vector_of_vectors[2]);
        }
        if (vector_of_vectors.size() >= 3 && enable_key_lookups) {
            index_result_t index_result = index_t::make(metric, config);
            test_punned_add_remove_vector( //
                index_result.index,        //
                vector_of_vectors[0],      //
                vector_of_vectors[1]);
        }
        // Larger collection
        {
            index_result_t index_result = index_t::make(metric, config);
            test_collection<true>(index_result.index, 42, vector_of_vectors);
        }

        // Try running benchmarks with a different number of threads
        for (std::size_t threads : {
                 static_cast<std::size_t>(1),
                 // TODO: Multithreaded updates should word differently and may involve a search first
                 //  static_cast<std::size_t>(2),
                 //  static_cast<std::size_t>(std::thread::hardware_concurrency()),
                 //  static_cast<std::size_t>(std::thread::hardware_concurrency() * 4),
                 //  static_cast<std::size_t>(vector_of_vectors.size()),
             }) {
            index_result_t index_result = index_t::make(metric, config);
            index_t& index = index_result.index;
            test_punned_concurrent_updates(index, 42, vector_of_vectors, threads);
        }
    };

    for (bool multi : {false, true})
        for (bool enable_key_lookups : {true, false})
            for (std::size_t connectivity : {3, 13, 50})
                run_punned(multi, enable_key_lookups, connectivity);
}

/**
 * Tests the functionality of the Tanimoto coefficient calculation and indexing.
 *
 * Initializes a dense index configured for Tanimoto similarity and fills it with randomly generated binary vectors.
 * It performs concurrent additions of these vectors to the index to ensure thread safety and correctness of concurrent
 * operations.
 *
 * @param dimensions Number of dimensions for the binary vectors.
 * @param connectivity The degree of connectivity for the index configuration.
 * @tparam key_at Data type used for the keys in the index.
 * @tparam slot_at Data type used for slots in the index.
 */
template <typename key_at, typename slot_at> void test_tanimoto(std::size_t dimensions, std::size_t connectivity) {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    std::size_t words = divide_round_up<CHAR_BIT>(dimensions);
    metric_punned_t metric(words, metric_kind_t::tanimoto_k, scalar_kind_t::b1x8_k);
    index_config_t config(connectivity);
    auto index_result = index_punned_t::make(metric, config);
    expect(index_result);
    index_punned_t& index = index_result.index;

    executor_default_t executor;
    std::size_t batch_size = 1000;
    std::vector<b1x8_t> scalars(batch_size * index.scalar_words());
    std::generate(scalars.begin(), scalars.end(), [] { return static_cast<b1x8_t>(std::rand()); });

    index.try_reserve({batch_size + index.size(), executor.size()});
    executor.fixed(batch_size, [&](std::size_t thread, std::size_t task) {
        index.add(task + 25000, scalars.data() + index.scalar_words() * task, thread);
    });
}

/**
 * Performs a unit test on the index with a ridiculous variety of configurations and parameters.
 *
 * This test aims to evaluate the index under extreme conditions, including small and potentially invalid parameters for
 * connectivity, dimensions, and other configurations. It tests both the addition of vectors and their retrieval in
 * these edge cases to ensure stability and error handling.
 *
 * @param dimensions Number of dimensions for the vectors.
 * @param connectivity Index connectivity configuration.
 * @param expansion_add Expansion factor during addition operations.
 * @param expansion_search Expansion factor during search operations.
 * @param count_vectors Number of vectors to add to the index.
 * @param count_wanted Number of results wanted from search operations.
 * @tparam key_at Data type used for the keys in the index.
 * @tparam slot_at Data type used for slots in the index.
 */
template <typename key_at, typename slot_at>
void test_absurd(std::size_t dimensions, std::size_t connectivity, std::size_t expansion_add,
                 std::size_t expansion_search, std::size_t count_vectors, std::size_t count_wanted) {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind_t::f32_k);
    index_dense_config_t config(connectivity, expansion_add, expansion_search);
    auto index_result = index_punned_t::make(metric, config);
    expect(index_result);
    index_punned_t& index = index_result.index;

    std::size_t count_max = (std::max)(count_vectors, count_wanted);
    std::size_t needed_scalars = count_max * dimensions;
    std::vector<f32_t> scalars(needed_scalars);
    std::generate(scalars.begin(), scalars.end(), [] { return static_cast<f32_t>(std::rand()); });

    expect(index.try_reserve({count_vectors, count_max}));
    index.change_expansion_add(expansion_add);
    index.change_expansion_search(expansion_search);

    // Parallel construction
    {
        executor_default_t executor(count_vectors);
        executor.fixed(count_vectors, [&](std::size_t thread, std::size_t task) {
            expect(index.add(task + 25000, scalars.data() + index.scalar_words() * task, thread));
        });
    }

    // Parallel search
    {
        executor_default_t executor(count_max);
        executor.fixed(count_max, [&](std::size_t thread, std::size_t task) {
            std::vector<vector_key_t> keys(count_wanted);
            std::vector<f32_t> distances(count_wanted);
            auto results = index.search(scalars.data() + index.scalar_words() * task, count_wanted, thread);
            expect(results);
            auto count_found = results.dump_to(keys.data(), distances.data());
            expect(count_found <= count_wanted);
            if (count_vectors && count_wanted)
                expect(count_found > 0);
        });
    }
}

/**
 * Tests the exact search functionality over a dataset of vectors, @b without constructing the index.
 *
 * Generates a dataset of vectors and performs exact search queries to verify that the search results are correct.
 * This function mainly validates the basic functionality of exact searches using a given similarity metric.
 *
 * @param dataset_count Number of vectors in the dataset.
 * @param queries_count Number of query vectors.
 * @param wanted_count Number of top matches required from each query.
 * @tparam scalar_at Data type of the elements in the vectors.
 */
template <typename scalar_at>
void test_exact_search(std::size_t dataset_count, std::size_t queries_count, std::size_t wanted_count) {
    std::size_t dimensions = 32;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind<scalar_at>());

    std::random_device seed_source;
    std::mt19937 generator(seed_source());
    std::uniform_real_distribution<> distribution(0.0, 1.0); // ! We can't pass `scalar_at` to the distribution
    std::vector<scalar_at> dataset(dataset_count * dimensions);
    std::generate(dataset.begin(), dataset.end(), [&] { return static_cast<scalar_at>(distribution(generator)); });

    exact_search_t search;
    auto results = search(                                                            //
        (byte_t const*)dataset.data(), dataset_count, dimensions * sizeof(scalar_at), //
        (byte_t const*)dataset.data(), queries_count, dimensions * sizeof(scalar_at), //
        wanted_count, metric);

    for (std::size_t i = 0; i < results.size(); ++i)
        assert(results.at(i)[0].offset == i); // Validate the top match
}

/**
 * Tests handling of variable length sets (group of sorted unique integers), as opposed to @b equi-dimensional vectors.
 *
 * Adds a predefined number of vectors to an index and checks if the size of the index is updated correctly.
 * It serves as a simple verification and showcase of how the same index can be used to handle strings and other types.
 *
 * @param index A reference to the index instance to be tested.
 * @tparam index_at Type of the index being tested.
 */
template <typename key_at, typename slot_at>
void test_sets(std::size_t collection_size, std::size_t min_set_length, std::size_t max_set_length) {

    /// Type of set elements, should support strong ordering
    using set_member_t = std::uint32_t;
    /// Jaccard is a fraction, so let's use a some float
    using set_distance_t = double;

    // Aliases for the index overload
    using vector_key_t = key_at;
    using slot_t = slot_at;
    using index_t = index_gt<set_distance_t, vector_key_t, slot_t>;

    // Let's allocate some data for indexing
    using set_view_t = span_gt<set_member_t const>;
    using sets_t = std::vector<std::vector<set_member_t>>;
    sets_t sets(collection_size);
    for (auto& set : sets) {
        std::size_t set_size = min_set_length + std::rand() % (max_set_length - min_set_length);
        set.resize(set_size);
        std::size_t upper_bound = (max_set_length - min_set_length) * 3;
        std::generate(set.begin(), set.end(), [=] { return static_cast<set_member_t>(std::rand() % upper_bound); });
        std::sort(set.begin(), set.end());
    }

    // Wrap the data into a proxy object
    struct metric_t {
        using member_cref_t = typename index_t::member_cref_t;
        using member_citerator_t = typename index_t::member_citerator_t;

        sets_t const* sets_ptr = nullptr;

        set_view_t set_at(std::size_t i) const noexcept { return {(*sets_ptr)[i].data(), (*sets_ptr)[i].size()}; }
        set_distance_t between(set_view_t a, set_view_t b) const {
            return metric_jaccard_gt<set_member_t, set_distance_t>{}(a.data(), b.data(), a.size(), b.size());
        }

        set_distance_t operator()(member_cref_t const& a, member_cref_t const& b) const {
            return between(set_at(get_slot(b)), set_at(get_slot(a)));
        }
        set_distance_t operator()(set_view_t some_vector, member_cref_t const& member) const {
            return between(some_vector, set_at(get_slot(member)));
        }
        set_distance_t operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return between(set_at(get_slot(b)), set_at(get_slot(a)));
        }
        set_distance_t operator()(set_view_t some_vector, member_citerator_t const& member) const {
            return between(some_vector, set_at(get_slot(member)));
        }
    };

    // Perform indexing
    aligned_wrapper_gt<index_t> aligned_index;
    aligned_index.index->reserve(sets.size());
    for (std::size_t i = 0; i < sets.size(); i++)
        aligned_index.index->add(i, set_view_t{sets[i].data(), sets[i].size()}, metric_t{&sets});
    expect(aligned_index.index->size() == sets.size());

    // Perform the search queries
    for (std::size_t i = 0; i < sets.size(); i++) {
        auto results = aligned_index.index->search(set_view_t{sets[i].data(), sets[i].size()}, 5, metric_t{&sets});
        expect(results.size() > 0);
    }
}

/**
 * Tests similarity search over strings using Levenshtein distances
 * implementation from StringZilla.
 *
 * Adds a predefined number of long strings, comparing them.
 *
 * @param index A reference to the index instance to be tested.
 * @tparam index_at Type of the index being tested.
 */
template <typename key_at, typename slot_at> void test_strings() {

    namespace sz = ashvardanian::stringzilla;

    /// Levenshtein distance is an integer
    using levenshtein_distance_t = std::int64_t;

    // Aliases for the index overload
    using vector_key_t = key_at;
    using slot_t = slot_at;
    using index_t = index_gt<levenshtein_distance_t, vector_key_t, slot_t>;

    std::string_view str0 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    std::string_view str1 = "ACG_ACTC_TAC-TACGTA_GTACACG_ACGT";
    std::string_view str2 = "A_GTACTACGTA-GTAC_TACGTACGTA-GTAGT";
    std::string_view str3 = "GTACGTAGT-ACGTACGACGTACGTACG-TACGTAC";
    std::vector<std::string_view> strings({str0, str1, str2, str3});

    // Wrap the data into a proxy object
    struct metric_t {
        using member_cref_t = typename index_t::member_cref_t;
        using member_citerator_t = typename index_t::member_citerator_t;

        std::vector<std::string_view> const* strings_ptr = nullptr;

        std::string_view str_at(std::size_t i) const noexcept { return (*strings_ptr)[i]; }
        levenshtein_distance_t between(std::string_view a, std::string_view b) const {
            sz::string_view asz{a.data(), a.size()};
            sz::string_view bsz{b.data(), b.size()};
            return sz::edit_distance<char const>(asz, bsz);
        }

        levenshtein_distance_t operator()(member_cref_t const& a, member_cref_t const& b) const {
            return between(str_at(get_slot(b)), str_at(get_slot(a)));
        }
        levenshtein_distance_t operator()(std::string_view some_vector, member_cref_t const& member) const {
            return between(some_vector, str_at(get_slot(member)));
        }
        levenshtein_distance_t operator()(member_citerator_t const& a, member_citerator_t const& b) const {
            return between(str_at(get_slot(*b)), str_at(get_slot(*a)));
        }
        levenshtein_distance_t operator()(std::string_view some_vector, member_citerator_t const& member) const {
            return between(some_vector, str_at(get_slot(*member)));
        }
    };

    // Perform indexing
    aligned_wrapper_gt<index_t> aligned_index;
    aligned_index.index->reserve(strings.size());
    for (std::size_t i = 0; i < strings.size(); i++)
        aligned_index.index->add(i, strings[i], metric_t{&strings});
    expect(aligned_index.index->size() == strings.size());

    // Perform the search queries
    for (std::size_t i = 0; i < strings.size(); i++) {
        auto results = aligned_index.index->search(strings[i], 5, metric_t{&strings});
        expect(results.size() > 0);
    }
}

/**
 *  @brief  Churns a tightly-reserved index, so `slot_lookup_` accumulates tombstones.
 *
 *  Removals leave tombstones that probes must walk past. If they are never reclaimed the
 *  table runs out of empty slots, probes can no longer terminate, and live keys go missing:
 *  `remove` reports nothing removed while `contains` still finds the key. Copying such a
 *  table must rebuild the probe chains, not reproduce the slot layout.
 */
template <typename key_at, typename slot_at> void test_slot_lookup_churn() {
    constexpr std::size_t live_count = 128;
    constexpr std::size_t churn_count = live_count * 8;
    constexpr std::size_t dimensions = 4;

    using index_punned_t = index_dense_gt<key_at, slot_at>;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k);

    std::random_device seed_source;
    std::mt19937 generator(seed_source());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    using vector_of_vectors_t = std::vector<std::vector<float>>;

    vector_of_vectors_t vector_of_vectors(live_count + churn_count);
    for (auto& vector : vector_of_vectors) {
        vector.resize(dimensions);
        std::generate(vector.begin(), vector.end(), [&] { return distribution(generator); });
    }

    index_punned_t index = index_punned_t::make(metric);

    // Reserve tightly, so the churn below exhausts the empty slots
    index.reserve(live_count * 3);

    std::vector<key_at> live_keys;
    std::size_t added = 0;
    for (; added < live_count; ++added) {
        index.add(static_cast<key_at>(added), vector_of_vectors[added].data());
        live_keys.push_back(static_cast<key_at>(added));
    }

    // Every iteration frees one slot and immediately reuses it
    for (std::size_t idx = 0; idx < churn_count; ++idx, ++added) {
        key_at victim = live_keys[idx % live_count];
        expect(index.contains(victim));
        expect_eq(index.remove(victim).completed, 1);
        expect(!index.contains(victim));
        index.add(static_cast<key_at>(added), vector_of_vectors[added].data());
        live_keys[idx % live_count] = static_cast<key_at>(added);
    }

    // Every live key stays reachable and the size never drifts
    expect_eq(index.size(), live_count);
    for (key_at key : live_keys)
        expect(index.contains(key));

    // Copying a table that still holds tombstones must not strand live keys
    expect_eq(index.remove(live_keys.back()).completed, 1);
    live_keys.pop_back();

    auto copy_result = index.copy();
    expect(copy_result);
    index_punned_t& copy = copy_result.index;
    expect_eq(copy.size(), live_keys.size());
    for (key_at key : live_keys)
        expect(copy.contains(key));
}

/**
 *  @brief  Tests replacing and updating entries in `index_dense_gt`, to ensure consistency
 *          after modifications.
 */
template <typename key_at, typename slot_at> void test_replacing_update() {

    using vector_key_t = key_at;
    using slot_t = slot_at;

    using index_punned_t = index_dense_gt<vector_key_t, slot_t>;
    metric_punned_t metric(1, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
    auto index_result = index_punned_t::make(metric);
    expect(index_result);
    index_punned_t& index = index_result.index;

    // Reserve space for 3 entries
    index.try_reserve(3);
    auto as_ptr = [](float v) {
        static float value;
        value = v;
        return &value;
    };

    // Add 3 entries
    index.add(42, as_ptr(1.1f));
    index.add(43, as_ptr(2.1f));
    index.add(44, as_ptr(3.1f));
    expect_eq(index.size(), 3);

    // Assert initial state
    auto initial_search = index.search(as_ptr(1.0f), 3);
    expect_eq(initial_search.size(), 3);
    expect_eq(initial_search[0].member.key, 42);
    expect_eq(initial_search[1].member.key, 43);
    expect_eq(initial_search[2].member.key, 44);

    // Replace the second entry
    index.remove(43);
    index.add(43, as_ptr(2.2f));
    expect_eq(index.size(), 3);

    // Assert state after replacing second entry
    auto post_second_replacement = index.search(as_ptr(1.0f), 3);
    expect_eq(post_second_replacement.size(), 3);
    expect_eq(post_second_replacement[0].member.key, 42);
    expect_eq(post_second_replacement[1].member.key, 43);
    expect_eq(post_second_replacement[2].member.key, 44);

    // Replace the first entry
    index.remove(42);
    index.add(42, as_ptr(1.2f));
    expect_eq(index.size(), 3);

    // Assert state after replacing first entry
    auto final_search = index.search(as_ptr(1.0f), 3, 0);
    expect_eq(final_search.size(), 3);
    expect_eq(final_search[0].member.key, 42);
    expect_eq(final_search[1].member.key, 43);
    expect_eq(final_search[2].member.key, 44);
}

/**
 * Tests the filtered search functionality of the index.
 */
void test_filtered_search() {
    constexpr std::size_t dataset_count = 2048;
    constexpr std::size_t dimensions = 32;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k);

    std::random_device seed_source;
    std::mt19937 generator(seed_source());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    using vector_of_vectors_t = std::vector<std::vector<float>>;

    vector_of_vectors_t vector_of_vectors(dataset_count);
    for (auto& vector : vector_of_vectors) {
        vector.resize(dimensions);
        std::generate(vector.begin(), vector.end(), [&] { return distribution(generator); });
    }

    index_dense_t index = index_dense_t::make(metric);
    index.reserve(dataset_count);
    for (std::size_t idx = 0; idx < dataset_count; ++idx)
        index.add(idx, vector_of_vectors[idx].data());
    expect_eq(index.size(), dataset_count);

    {
        auto predicate = [](index_dense_t::key_t key) { return key != 0; };
        auto results = index.filtered_search(vector_of_vectors[0].data(), 10, predicate);
        expect_eq(10u, results.size()); // ! Should not contain 0
        for (std::size_t i = 0; i != results.size(); ++i)
            expect(0 != results[i].member.key);
    }
    {
        auto predicate = [](index_dense_t::key_t) { return false; };
        auto results = index.filtered_search(vector_of_vectors[0].data(), 10, predicate);
        expect_eq(0u, results.size()); // ! Should not contain 0
    }
    {
        auto predicate = [](index_dense_t::key_t key) { return key == 10; };
        auto results = index.filtered_search(vector_of_vectors[0].data(), 10, predicate);
        expect_eq(1u, results.size()); // ! Should not contain 0
        expect_eq(index_dense_t::key_t(10), results[0].member.key);
    }
}

void test_isolate() {
    constexpr std::size_t dataset_count = 16;
    constexpr std::size_t dimensions = 32;
    metric_punned_t metric(dimensions, metric_kind_t::cos_k);

    std::random_device seed_source;
    std::mt19937 generator(seed_source());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    using vector_of_vectors_t = std::vector<std::vector<float>>;

    vector_of_vectors_t vector_of_vectors(dataset_count);
    for (auto& vector : vector_of_vectors) {
        vector.resize(dimensions);
        std::generate(vector.begin(), vector.end(), [&] { return distribution(generator); });
    }

    index_dense_t index = index_dense_t::make(metric);
    index.reserve(dataset_count);
    for (std::size_t idx = 0; idx < dataset_count; ++idx) {
        index.add(idx, vector_of_vectors[idx].data());
    }
    expect_eq(index.size(), dataset_count);

    for (std::size_t idx = 0; idx < dataset_count; ++idx) {
        if (idx % 2 == 0)
            index.remove(idx);
    }

    auto result = index.isolate();
    for (std::size_t idx = 0; idx < dataset_count; ++idx) {
        auto result = index.search(vector_of_vectors[idx].data(), 16);
        expect_eq(result.size(), dataset_count / 2);
    }
}

static void usearch_write_backtrace(int signal_number) {
    std::fprintf(stderr, "\n[usearch] Fatal signal %d. Back-trace:\n", signal_number);
#if USEARCH_HAS_STD_STACKTRACE
    // C++23 `std::stacktrace` covers every platform the library can reach.
    auto const current_trace = std::stacktrace::current();
    std::size_t frame_index = 0;
    for (auto const& frame : current_trace) {
        std::fprintf(stderr, "  #%2zu %s\n", frame_index, std::to_string(frame).c_str());
        ++frame_index;
    }
#elif defined(_WIN32)
    // Fallback for MSVC stdlibs without `<stacktrace>`: DbgHelp API.
    constexpr USHORT backtrace_depth_limit = 64;
    void* backtrace_frames[backtrace_depth_limit];
    USHORT backtrace_depth = CaptureStackBackTrace(0, backtrace_depth_limit, backtrace_frames, nullptr);
    HANDLE current_process = GetCurrentProcess();
    SymInitialize(current_process, nullptr, TRUE);

    unsigned char symbol_info_buffer[sizeof(SYMBOL_INFO) + 256 * sizeof(char)];
    SYMBOL_INFO* symbol_info = reinterpret_cast<SYMBOL_INFO*>(symbol_info_buffer);
    symbol_info->MaxNameLen = 255;
    symbol_info->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (USHORT frame_index = 0; frame_index < backtrace_depth; ++frame_index) {
        if (SymFromAddr(current_process, reinterpret_cast<DWORD64>(backtrace_frames[frame_index]), 0, symbol_info))
            std::fprintf(stderr, "  #%2u %s + 0x%llx\n", static_cast<unsigned>(frame_index), symbol_info->Name,
                         static_cast<unsigned long long>(reinterpret_cast<DWORD64>(backtrace_frames[frame_index]) -
                                                         symbol_info->Address));
        else
            std::fprintf(stderr, "  #%2u %p\n", static_cast<unsigned>(frame_index), backtrace_frames[frame_index]);
    }
#else
    // Fallback for POSIX stdlibs without `<stacktrace>`: `<execinfo.h>`.
    constexpr int backtrace_depth_limit = 64;
    void* backtrace_frames[backtrace_depth_limit];
    int const backtrace_depth = backtrace(backtrace_frames, backtrace_depth_limit);
    backtrace_symbols_fd(backtrace_frames, backtrace_depth, STDERR_FILENO);
#endif
    std::fflush(stderr);
}

static void usearch_crash_handler(int signal_number) {
    usearch_write_backtrace(signal_number);
    // Restore the default disposition and re-raise so the shell / CI sees the true exit status.
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

static void install_crash_handlers() {
    int const fatal_signals[] = {SIGSEGV, SIGABRT, SIGILL, SIGFPE};
    for (int signal_number : fatal_signals)
        std::signal(signal_number, &usearch_crash_handler);
}

/**
 *  @brief  Cosine norm sidecar: verifies through the diagnostic hook that the fast route is taken exactly when it
 *          should be, that every lifecycle path keeps the sidecar populated, and that distances match the reference
 *          `metric_cos_gt` kernel within floating-point noise.
 */
template <typename scalar_at> void test_cosine_norm_cache(std::size_t count, std::size_t dimensions) {

    using vector_key_t = std::int64_t;
    using slot_t = std::uint32_t;
    using index_t = index_dense_gt<vector_key_t, slot_t>;
    cosine_cache_diagnostics_t& diag = cosine_cache_diagnostics;

    scalar_kind_t const kind = scalar_kind<scalar_at>();
    bool const with_bf16 = kind == scalar_kind_t::bf16_k;
    float const tolerance = with_bf16 ? 2e-3f : 2e-5f;

    // Deterministic data with mixed signs and magnitudes, plus one zero vector.
    std::vector<std::vector<scalar_at>> vectors(count, std::vector<scalar_at>(dimensions));
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (std::size_t i = 0; i != count; ++i)
        for (std::size_t d = 0; d != dimensions; ++d)
            vectors[i][d] = static_cast<scalar_at>(i + 1 == count ? 0.f : distribution(generator) * float(1 + i % 7));
    std::vector<scalar_at> const& query = vectors[0];

    metric_punned_t const builtin(dimensions, metric_kind_t::cos_k, kind);
    metric_punned_t const l2(dimensions, metric_kind_t::l2sq_k, kind);
    // The sidecar applies to the built-in auto-vectorized cosine kernels only. Without NumKong they are the
    // built-in kernels, so the route must be active; with NumKong the kernels come from NumKong and it must not.
    bool const sidecar_expected = builtin.supports_cosine_norm_cache();
    if (!USEARCH_USE_NUMKONG)
        expect(sidecar_expected);

    // Two references. `true_cosine` is the reference kernel `metric_cos_gt`, independent of how the index computes
    // distances; it is the right yardstick whenever the sidecar route is active (the route must reproduce true
    // cosine) and for the user-kernel index below, whose kernel is exactly `metric_cos_gt`. When the route is
    // inactive (NumKong builds), the only guarantee is pass-through of the configured kernel, whose bf16 results
    // legitimately differ from `metric_cos_gt` by more than the tolerance, so the configured metric is the
    // reference there.
    auto true_cosine = [&](scalar_at const* a, scalar_at const* b) -> float {
        return metric_cos_gt<scalar_at, float>{}(a, b, dimensions);
    };
    auto configured = [&](scalar_at const* a, scalar_at const* b) -> float {
        return builtin(reinterpret_cast<byte_t const*>(a), reinterpret_cast<byte_t const*>(b));
    };
    float const self_tolerance = sidecar_expected ? tolerance : 10 * tolerance;
    // Exact search scores every live member through the same proxy that approximate search uses. Members are
    // resolved by key against the original data, since the sidecar is addressed by slot and slots move on `compact`.
    std::vector<scalar_at> replacement(dimensions);
    for (std::size_t d = 0; d != dimensions; ++d)
        replacement[d] = static_cast<scalar_at>(d % 2 ? 1.f : -1.f);
    // `replaced` says whether key 1 holds `replacement` in the index being checked (the primary index and its
    // copies do, after the re-insertion below; every other index keeps the original vector).
    auto expect_matches_reference = [&](index_t const& index, std::vector<scalar_at> const& q, bool replaced,
                                        bool independent_reference) {
        auto result = index.search(q.data(), index.size(), 0, true);
        expect(result);
        expect_eq(result.size(), index.size());
        for (std::size_t i = 0; i != result.size(); ++i) {
            auto match = result[i];
            vector_key_t key = vector_key_t(match.member.key);
            scalar_at const* member = replaced && key == 1 ? replacement.data() : vectors[std::size_t(key)].data();
            float expected = independent_reference ? true_cosine(q.data(), member) : configured(q.data(), member);
            expect(std::abs(match.distance - expected) <= tolerance);
            if (i)
                expect(result[i - 1].distance <= match.distance + tolerance);
        }
    };
    auto expect_route = [&](bool used) {
        expect_eq(diag.query_preparation() > 0, used);
        expect_eq(diag.cache_hit() > 0, used);
        expect_eq(diag.invalid_fallback(), std::size_t(0));
    };

    // --- Reference L2 twin, measured right after `reserve`, before any vectors are copied in. ---
    auto l2_made = index_t::make(l2);
    expect(l2_made);
    index_t& l2_index = l2_made.index;
    expect(l2_index.try_reserve(count));
    std::size_t const l2_reserved_memory = l2_index.memory_usage();
    std::size_t const sidecar_bytes = sidecar_expected ? l2_index.capacity() * sizeof(std::uint32_t) : 0;

    // --- Fresh cosine index: exactly 4 bytes per capacity slot, populated on insertion, served from the sidecar. ---
    auto made = index_t::make(builtin);
    expect(made);
    index_t& index = made.index;
    expect(index.try_reserve(count));
    expect_eq(index.capacity(), l2_index.capacity());
    expect_eq(index.memory_usage(), l2_reserved_memory + sidecar_bytes);
    diag.reset();
    for (std::size_t i = 0; i != count; ++i)
        expect(index.add(vector_key_t(i), vectors[i].data()));
    expect_eq(diag.population(), sidecar_expected ? count : std::size_t(0));
    diag.reset();
    expect_matches_reference(index, query, false, sidecar_expected);
    expect_route(sidecar_expected);
    diag.reset();
    {
        auto approximate = index.search(query.data(), 5);
        expect(approximate);
        expect_eq(approximate.size(), std::size_t(5));
        expect_eq(vector_key_t(approximate[0].member.key), vector_key_t(0));
        expect(std::abs(approximate[0].distance) <= self_tolerance);
    }
    expect_route(sidecar_expected);

    // --- Metric change L2 -> cosine must populate; cosine -> L2 must release; equivalent cosine preserves. ---
    for (std::size_t i = 0; i != count; ++i)
        expect(l2_index.add(vector_key_t(i), vectors[i].data()));
    std::size_t const l2_memory = l2_index.memory_usage();
    diag.reset();
    expect(l2_index.try_change_metric(metric_punned_t(dimensions, metric_kind_t::cos_k, kind)));
    expect_eq(diag.population(), sidecar_expected ? count : std::size_t(0));
    expect_eq(l2_index.memory_usage(), l2_memory + sidecar_bytes);
    diag.reset();
    expect_matches_reference(l2_index, query, false, sidecar_expected);
    expect_route(sidecar_expected);
    expect(l2_index.try_change_metric(metric_punned_t(dimensions, metric_kind_t::l2sq_k, kind)));
    expect_eq(l2_index.memory_usage(), l2_memory);
    expect(l2_index.try_change_metric(metric_punned_t(dimensions, metric_kind_t::cos_k, kind)));
    diag.reset();
    expect(l2_index.try_change_metric(metric_punned_t(dimensions, metric_kind_t::cos_k, kind)));
    expect_eq(diag.population(), std::size_t(0));
    diag.reset();
    expect_matches_reference(l2_index, query, false, sidecar_expected);
    expect_route(sidecar_expected);

    // --- Remove and re-insert a key: the slot is invalidated, then repopulated with the new norm. ---
    expect(index.remove(vector_key_t(1)));
    diag.reset();
    expect(index.add(vector_key_t(1), replacement.data()));
    expect_eq(diag.population(), sidecar_expected ? std::size_t(1) : std::size_t(0));
    diag.reset();
    expect_matches_reference(index, replacement, true, sidecar_expected);
    expect_route(sidecar_expected);
    {
        auto top = index.search(replacement.data(), 1);
        expect_eq(top.size(), std::size_t(1));
        expect_eq(vector_key_t(top[0].member.key), vector_key_t(1));
        expect(std::abs(top[0].distance) <= self_tolerance);
    }

    // --- Compaction rebuilds the sidecar for the remapped slots. ---
    expect(index.remove(vector_key_t(2)));
    diag.reset();
    expect(index.compact());
    expect_eq(diag.population(), sidecar_expected ? index.size() : std::size_t(0));
    diag.reset();
    expect_matches_reference(index, query, true, sidecar_expected);
    expect_route(sidecar_expected);

    // --- Copy duplicates a valid sidecar instead of recomputing it. ---
    diag.reset();
    auto copied = index.copy();
    expect(copied);
    expect_eq(diag.population(), std::size_t(0));
    diag.reset();
    expect_matches_reference(copied.index, query, true, sidecar_expected);
    expect_route(sidecar_expected);

    // --- Serialization: the sidecar is never written, rebuilt on load, absent for memory-mapped views. ---
    char const* path = "tmp_cosine_norm_cache.usearch";
    expect(index.save(path));
    auto loaded = index_t::make(builtin);
    expect(loaded);
    diag.reset();
    expect(loaded.index.load(path));
    expect_eq(diag.population(), sidecar_expected ? index.size() : std::size_t(0));
    diag.reset();
    expect_matches_reference(loaded.index, query, true, sidecar_expected);
    expect_route(sidecar_expected);
    auto viewed = index_t::make(builtin);
    expect(viewed);
    expect(viewed.index.view(path));
    diag.reset();
    expect_matches_reference(viewed.index, query, true, sidecar_expected);
    expect_route(false);
    expect(diag.route_bypass() > 0);
    std::remove(path);

    // --- Indexes that don't own their vectors never allocate a sidecar. ---
    index_dense_config_t external_config;
    external_config.exclude_vectors = true;
    auto external = index_t::make(builtin, external_config);
    expect(external);
    expect(external.index.try_reserve(count));
    expect_eq(external.index.memory_usage(), l2_reserved_memory);
    diag.reset();
    for (std::size_t i = 0; i != count; ++i)
        expect(external.index.add(vector_key_t(i), vectors[i].data()));
    expect_eq(diag.population(), std::size_t(0));
    diag.reset();
    expect_matches_reference(external.index, query, false, sidecar_expected);
    expect_route(false);

    // --- A cosine-labelled user callback with the three-argument signature must not be captured, even when an
    //     identical-code-folding linker merges the trampolines: provenance is the kernel address, not the trampoline.
    {
        static std::size_t user_calls = 0;
        user_calls = 0;
        auto user_kernel = +[](std::uintptr_t a_ptr, std::uintptr_t b_ptr, std::uintptr_t n) -> distance_punned_t {
            ++user_calls;
            return metric_cos_gt<scalar_at, float>{}((scalar_at const*)a_ptr, (scalar_at const*)b_ptr, n);
        };
        metric_punned_t user_metric = metric_punned_t::stateless(dimensions, (std::uintptr_t)user_kernel,
                                                                 metric_punned_signature_t::array_array_size_k,
                                                                 metric_kind_t::cos_k, kind);
        expect(!user_metric.supports_cosine_norm_cache());
        auto user_made = index_t::make(user_metric);
        expect(user_made);
        expect(user_made.index.try_reserve(count));
        expect_eq(user_made.index.memory_usage(), l2_reserved_memory);
        for (std::size_t i = 0; i != count; ++i)
            expect(user_made.index.add(vector_key_t(i), vectors[i].data()));
        diag.reset();
        user_calls = 0;
        expect_matches_reference(user_made.index, query, false, true);
        expect_route(false);
        expect(user_calls >= count);
    }

    // --- Non-finite members fall back per comparison; a non-finite query bypasses the route entirely. ---
    if (!with_bf16) {
        std::vector<scalar_at> nan_vector(dimensions, static_cast<scalar_at>(1.f));
        nan_vector[0] = static_cast<scalar_at>(std::numeric_limits<float>::quiet_NaN());
        auto nonfinite = index_t::make(builtin);
        expect(nonfinite);
        expect(nonfinite.index.try_reserve(count + 1));
        for (std::size_t i = 0; i != count; ++i)
            expect(nonfinite.index.add(vector_key_t(i), vectors[i].data()));
        diag.reset();
        expect(nonfinite.index.add(vector_key_t(count), nan_vector.data()));
        expect_eq(diag.population(), std::size_t(0));
        diag.reset();
        auto result = nonfinite.index.search(query.data(), count + 1, 0, true);
        expect(result);
        expect_eq(result.size(), count + 1);
        expect_eq(diag.invalid_fallback(), sidecar_expected ? std::size_t(1) : std::size_t(0));
        expect_eq(diag.cache_hit(), sidecar_expected ? count : std::size_t(0));
        diag.reset();
        auto nan_result = nonfinite.index.search(nan_vector.data(), count + 1, 0, true);
        expect(nan_result);
        expect_eq(nan_result.size(), count + 1);
        expect_eq(diag.query_preparation(), std::size_t(0));
        expect_eq(diag.cache_hit(), std::size_t(0));
        // The rejected query norm is reported once as a fallback, then the whole search bypasses the route.
        expect_eq(diag.invalid_fallback(), sidecar_expected ? std::size_t(1) : std::size_t(0));
        expect(diag.route_bypass() > 0);
    }

    // --- Concurrent insertions and searches share the sidecar without locks; every result must stay finite and
    //     ordered, and once the writers finish every slot must be populated. ---
    {
        auto concurrent = index_t::make(builtin);
        expect(concurrent);
        std::size_t const threads = 8;
        expect(concurrent.index.try_reserve({count, threads}));
        for (std::size_t i = 0; i != count / 2; ++i)
            expect(concurrent.index.add(vector_key_t(i), vectors[i].data(), i % threads));
        diag.reset();
        executor_default_t executor(threads);
        std::atomic<std::size_t> searches_done{0};
        executor.fixed(count, [&](std::size_t thread, std::size_t task) {
            if (task < count / 2) {
                auto result = concurrent.index.search(vectors[task].data(), 5, thread);
                expect(result);
                for (std::size_t i = 0; i != result.size(); ++i) {
                    auto match = result[i];
                    expect(match.distance >= -tolerance && match.distance <= 2 + tolerance);
                    if (i)
                        expect(result[i - 1].distance <= match.distance + tolerance);
                }
                ++searches_done;
            } else
                expect(concurrent.index.add(vector_key_t(task), vectors[task].data(), thread));
        });
        expect_eq(searches_done.load(), count / 2);
        expect_eq(concurrent.index.size(), count);
        expect_eq(diag.population(), sidecar_expected ? count - count / 2 : std::size_t(0));
        diag.reset();
        expect_matches_reference(concurrent.index, query, false, sidecar_expected);
        expect_route(sidecar_expected);
    }
}

/**
 *  @brief  Regression test: `make(metric, config)` must return an index that is
 *          immediately usable - no explicit `reserve` required before `load` /
 *          `view` / `search`. Previously the typed graph's `{0, 0}` thread
 *          limits leaked into `load_from_stream`, leaving `available_threads_`
 *          empty and making the first `search` throw "No available threads to
 *          lock".
 */
void test_load_after_metric_make() {
    std::printf("Testing load and view into a metric-made index\n");

    using index_t = index_dense_gt<std::int64_t, std::uint32_t>;
    std::size_t const dimensions = 32;
    std::size_t const collection = 64;

    std::default_random_engine rng(7);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::vector<std::vector<float>> data(collection);
    for (auto& vector : data) {
        vector.resize(dimensions);
        for (auto& value : vector)
            value = distribution(rng);
    }

    metric_punned_t metric(dimensions, metric_kind_t::cos_k, scalar_kind<f32_t>());
    index_dense_config_t config(16);

    // Build an index and persist it to disk.
    index_t::state_result_t built = index_t::make(metric, config);
    expect(built);
    expect(built.index.try_reserve(collection));
    for (std::size_t i = 0; i != collection; ++i)
        expect(built.index.add(static_cast<std::int64_t>(i), data[i].data()));
    char const* path = "tmp_metric_make.usearch";
    expect(built.index.save(path));

    // Load into a fresh, metric-made index that was never explicitly reserved.
    // The first `search` must not throw "No available threads to lock".
    index_t::state_result_t loaded = index_t::make(metric, config);
    expect(loaded);
    expect(loaded.index.load(path));
    expect_eq(loaded.index.size(), collection);
    std::int64_t found[8];
    expect(loaded.index.search(data[0].data(), 5).dump_to(found) != 0);

    // Same check for the memory-mapped `view` path.
    index_t::state_result_t viewed = index_t::make(metric, config);
    expect(viewed);
    expect(viewed.index.view(path));
    expect_eq(viewed.index.size(), collection);
    expect(viewed.index.search(data[0].data(), 5).dump_to(found) != 0);

    std::remove(path);
}

/**
 *  @brief  Feeds `view` a truncated index file.
 *
 *  The matrix dimensions and the node count come from the file itself, so a short file
 *  describes a matrix and a level array that reach past the mapping. Both must be rejected
 *  rather than walked, and a caller-supplied offset past the end must fail the same way.
 */
void test_view_of_truncated_file() {
    std::printf("Testing view of a truncated index file\n");

    using index_t = index_dense_gt<std::int64_t, std::uint32_t>;
    std::size_t const dimensions = 64;
    std::size_t const collection = 2048;
    char const* path = "tmp_truncated.usearch";

    metric_punned_t metric(dimensions, metric_kind_t::l2sq_k, scalar_kind<f32_t>());
    index_t::state_result_t built = index_t::make(metric);
    expect(built);
    expect(built.index.try_reserve(collection));
    std::vector<float> vector(dimensions);
    for (std::size_t i = 0; i != collection; ++i) {
        for (std::size_t d = 0; d != dimensions; ++d)
            vector[d] = static_cast<float>((i * 31 + d) % 997) / 997.f;
        expect(built.index.add(static_cast<std::int64_t>(i), vector.data()));
    }
    expect(built.index.save(path));

    std::vector<char> prefix;
    {
        std::FILE* file = std::fopen(path, "rb");
        expect(file != nullptr);
        std::fseek(file, 0, SEEK_END);
        long const bytes = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        prefix.resize(static_cast<std::size_t>(bytes) / 8);
        expect(std::fread(prefix.data(), 1, prefix.size(), file) == prefix.size());
        std::fclose(file);
    }
    {
        std::FILE* file = std::fopen(path, "wb");
        expect(file != nullptr);
        std::fwrite(prefix.data(), 1, prefix.size(), file);
        std::fclose(file);
    }

    index_t::state_result_t viewed = index_t::make(metric);
    expect(viewed);
    serialization_result_t truncated = viewed.index.view(path);
    expect(!truncated);
    truncated.error.release();

    // A caller-supplied offset past the end must fail the same way.
    index_t::state_result_t offset_viewed = index_t::make(metric);
    expect(offset_viewed);
    serialization_result_t beyond = offset_viewed.index.view(memory_mapped_file_t(path), prefix.size() * 4);
    expect(!beyond);
    beyond.error.release();

    std::remove(path);
}

int main(int, char**) {
    install_crash_handlers();

    std::printf("Hardware acceleration compiled: %s\n", hardware_acceleration_compiled());
    std::printf("Hardware acceleration available: %s\n", hardware_acceleration_available());

    // Non-default floating-point types may result in many compilation & rounding issues.
    test_uint40();
    test_checked_size_arithmetic();
    test_cosine<f32_t, std::int64_t, uint40_t>(10, 10);
    test_cosine<bf16_t, std::int64_t, uint40_t>(10, 10);
    test_cosine<f16_t, std::int64_t, uint40_t>(10, 10);
    test_cosine<e5m2_t, std::int64_t, uint40_t>(10, 10);
    test_cosine<e4m3_t, std::int64_t, uint40_t>(10, 10);

    // Test plugins, like K-Means clustering.
    {
        std::size_t vectors_count = 1000, centroids_count = 10, dimensions = 256;
        kmeans_clustering_t clustering;
        clustering.max_iterations = 2;
        std::vector<float> vectors(vectors_count * dimensions), centroids(centroids_count * dimensions);
        matrix_slice_gt<float const> vectors_slice(vectors.data(), dimensions, vectors_count);
        matrix_slice_gt<float> centroids_slice(centroids.data(), dimensions, centroids_count);
        std::generate(vectors.begin(), vectors.end(), [] { return float(std::rand()) / float(RAND_MAX); });
        std::vector<std::size_t> assignments(vectors_count);
        std::vector<distance_punned_t> distances(vectors_count);
        auto clustering_result = clustering(vectors_slice, centroids_slice, {assignments.data(), assignments.size()},
                                            {distances.data(), distances.size()});
        expect(clustering_result);
    }

    // Exact search without constructing indexes.
    // Great for validating the distance functions.
    std::printf("Testing exact search\n");
    for (std::size_t dataset_count : {10, 100})
        for (std::size_t queries_count : {1, 10})
            for (std::size_t wanted_count : {1, 5}) {
                test_exact_search<f32_t>(dataset_count, queries_count, wanted_count);
                test_exact_search<bf16_t>(dataset_count, queries_count, wanted_count);
                test_exact_search<f16_t>(dataset_count, queries_count, wanted_count);
                test_exact_search<e5m2_t>(dataset_count, queries_count, wanted_count);
                test_exact_search<e4m3_t>(dataset_count, queries_count, wanted_count);
            }

    // Make sure the initializers and the algorithms can work with inadequately small values.
    // Be warned - this combinatorial explosion of tests produces close to __500'000__ tests!
    std::printf("Testing allowed, but absurd index configs\n");
    for (std::size_t connectivity : {2, 3})   // ! Zero maps to default, one degenerates
        for (std::size_t dimensions : {1, 3}) // ! Zero will raise
            for (std::size_t expansion : {0, 1, 3})
                for (std::size_t count_vectors : {0, 1, 2, 17})
                    for (std::size_t count_wanted : {0, 1, 3, 19}) {
                        test_absurd<std::int64_t, slot32_t>(dimensions, connectivity, expansion, expansion,
                                                            count_vectors, count_wanted);
                        test_absurd<uint40_t, uint40_t>(dimensions, connectivity, expansion, expansion, count_vectors,
                                                        count_wanted);
                    }

    // TODO: Test absurd configs that are banned
    // for (metric_kind_t metric_kind : {metric_kind_t::cos_k, metric_kind_t::unknown_k, metric_kind_t::haversine_k}) {}

    // Test with cosine metric - the most common use case
    std::printf("Testing common cases\n");
    for (std::size_t collection_size : {10, 500})
        for (std::size_t dimensions : {97, 256}) {
            std::printf("- Indexing %zu vectors with cos: <f32_t, std::int64_t, slot32_t> \n", collection_size);
            test_cosine<f32_t, std::int64_t, slot32_t>(collection_size, dimensions);
            std::printf("- Indexing %zu vectors with cos: <f32_t, std::int64_t, uint40_t> \n", collection_size);
            test_cosine<f32_t, std::int64_t, uint40_t>(collection_size, dimensions);
            std::printf("- Indexing %zu vectors with cos: <bf16, std::int64_t, uint40_t> \n", collection_size);
            test_cosine<bf16_t, std::int64_t, uint40_t>(collection_size, dimensions);
            std::printf("- Indexing %zu vectors with cos: <e5m2, std::int64_t, uint40_t> \n", collection_size);
            test_cosine<e5m2_t, std::int64_t, uint40_t>(collection_size, dimensions);
            std::printf("- Indexing %zu vectors with cos: <e4m3, std::int64_t, uint40_t> \n", collection_size);
            test_cosine<e4m3_t, std::int64_t, uint40_t>(collection_size, dimensions);
        }

    // Test with binary vectors
    std::printf("Testing binary vectors\n");
    for (std::size_t connectivity : {3, 13, 50})
        for (std::size_t dimensions : {97, 256})
            test_tanimoto<std::int64_t, slot32_t>(dimensions, connectivity);

    // Beyond dense equi-dimensional vectors - integer sets
    std::printf("Testing sparse vectors, strings, and sets\n");
    for (std::size_t set_size : {1, 100, 1000})
        test_sets<std::int64_t, slot32_t>(set_size, 20, 30);
    test_strings<std::int64_t, slot32_t>();

    std::printf("Testing key lookups under churn\n");
    test_slot_lookup_churn<std::int64_t, std::uint32_t>();
    test_slot_lookup_churn<std::int64_t, uint40_t>();

    test_filtered_search();
    test_isolate();
    std::printf("Testing cosine norm cache\n");
    test_cosine_norm_cache<f32_t>(64, 33);
    test_cosine_norm_cache<bf16_t>(64, 33);
    test_load_after_metric_make();
    test_view_of_truncated_file();
    return 0;
}
