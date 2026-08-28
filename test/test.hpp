/**
 *  @brief Shared helpers for the C++ test suites: failure counting, ledgered allocators,
 *      and universally-representable synthetic data generation.
 *
 *  @author Ash Vardanian
 *  @file test/test.hpp
 *  @date July 17, 2026
 *
 *  Checks return their own failure counts and roll up to @c main without global mutable
 *  state. Four verbs cover every case: @c expect for a condition, @c expect_eq and
 *  @c expect_near when diagnosis needs the values, and @c report for the verdict column.
 *  A status becomes a condition through @c succeeded(), so no fifth verb is needed.
 */
#pragma once

#include <cmath>   // `std::fabs`, `std::fmax` - the relative tolerance of `expect_near`
#include <cstdint> // `PTRDIFF_MAX` - the unbounded allocation budget
#include <cstdlib> // `std::malloc`, `std::free`, `std::abort`

#include <algorithm> // `std::sort`, `std::unique` - the dispensed-id density check
#include <atomic>    // `std::atomic`
#include <chrono>    // `std::chrono::microseconds` - the flaky waiter's oversleep
#include <thread>    // `std::this_thread::yield`, `std::this_thread::sleep_for`
#include <vector>    // `std::vector` - the dispensed-id density check

#include <fmt/format.h>

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"

namespace unum::usearch {

#define expect(cond) expect_((bool)(cond), #cond, __FILE__, __LINE__)
inline std::size_t expect_(bool condition, char const* expression, char const* file, int line) {
    if (!condition) fmt::print(stderr, "  check failed at {}:{}: {}\n", file, line, expression);
    return condition ? 0u : 1u;
}

#define expect_eq(actual, wanted) expect_eq_((actual), (wanted), #actual " == " #wanted, __FILE__, __LINE__)
template <typename actual_type_, typename wanted_type_>
inline std::size_t expect_eq_(actual_type_ const& actual, wanted_type_ const& wanted, char const* expression,
                              char const* file, int line) {
    if (actual == wanted) return 0;
    fmt::print(stderr, "  check failed at {}:{}: {} - expected {}, got {}\n", file, line, expression, wanted, actual);
    return 1;
}

/** @brief The @c expect_eq twin for floats, at the 1e-3 relative tolerance a @c -ffast-math
 *      build needs: it re-derives finalizes per inlining context, so bit-equality is wrong. */
#define expect_near(actual, wanted) expect_near_((actual), (wanted), #actual " ~ " #wanted, __FILE__, __LINE__)
inline std::size_t expect_near_(f64_t actual, f64_t wanted, char const* expression, char const* file, int line) {
    f64_t const scale = std::fmax(std::fmax(std::fabs(actual), std::fabs(wanted)), 1.0);
    f64_t const difference = std::fabs(actual - wanted);
    if (difference <= 1e-3 * scale) return 0;
    fmt::print(stderr, "  check failed at {}:{}: {} - expected {:.6g}, got {:.6g}, off by {:.4g}\n", file, line,
               expression, wanted, actual, difference);
    return 1;
}

/** @brief One verdict column, whichever way the case ended; answers the count it was given,
 *      so a caller accumulates through it. The @c char-pointer overload names a skip. */
inline std::size_t report(char const* name, std::size_t failures) {
    fmt::println("{}  {:<20} : {} failures", failures ? "FAILED" : "passed", name, failures);
    return failures;
}

inline std::size_t report(char const* name, char const* skip_reason) {
    fmt::println("skipped  {:<20} : {}", name, skip_reason);
    return 0;
}

/** @brief Ids dispensed to racing lanes must land dense & duplicate-free: sorted in
 *      place, unique, and running from @p first_id through the half-open bound
 *      @p first_id plus the id count. */
template <typename node_id_type_>
inline std::size_t expect_contiguous_ids(std::vector<node_id_type_>& ids, std::size_t first_id) {
    std::sort(ids.begin(), ids.end());
    std::size_t failures = expect(std::unique(ids.begin(), ids.end()) == ids.end());
    failures += expect(!ids.empty() && static_cast<std::size_t>(ids.front()) == first_id &&
                       static_cast<std::size_t>(ids.back()) == first_id + ids.size() - 1);
    return failures;
}

/** @brief Derived-draw salts shared by both scenario libraries. Every scenario family
 *      owns exactly one named hex-leet seed and never reuses it verbatim: query draws
 *      XOR @c salt_queries_k, auxiliary value draws - extras, prefills, pairs - XOR
 *      @c salt_extras_k, per-lane draws add the lane index, and per-context re-seeds add
 *      small literal offsets. Two salts, disjoint from every lane index and offset, keep
 *      a family's draws distinct without per-site magic. */
constexpr u64_t salt_queries_k = 0xFFull;
/** @brief The auxiliary-draw twin of @ref salt_queries_k. */
constexpr u64_t salt_extras_k = 0xAAull;

/** @brief Spin ceiling for waits on prompt flags - flags a peer sets within bounded
 *      work: a gate entry, a hold acquisition, a refusal confirmation. Expiry proves a
 *      broken rendezvous. Waits on a peer's completion of unbounded-duration work are
 *      deliberately not hatched at this bound - see @ref attempt_limit_k. */
constexpr std::size_t spin_limit_k = std::size_t {1} << 28;
/** @brief Retry ceiling for settle phases & sweep loops: once every racer is done, an
 *      operation retried this many times without success has provably starved. */
constexpr std::size_t attempt_limit_k = std::size_t {1} << 20;

/** @brief Spins - yielding - until @p condition holds or @c spin_limit_k expires;
 *      answers whether the condition held. It never counts failures itself: the caller
 *      owns the on-expiry story - count & proceed so a hold can die, count & return, or
 *      open a gate before leaving so the pool cannot deadlock. */
template <typename condition_type_>
[[nodiscard]] inline bool spin_until(condition_type_&& condition) noexcept {
    for (std::size_t spins = 0; spins != spin_limit_k; ++spins) {
        if (condition()) return true;
        std::this_thread::yield();
    }
    return false;
}

/** @brief The reader-lane invariants over one search's scored results: every id below
 *      the watermark sampled after the search, distances ascending, the pool bound
 *      honored. Answers the violation count silently - racing lanes add it to their
 *      shared counter, since printing from inside a race would interleave & flood - and
 *      the status verdict stays at the call site: lanes differ in how they record it. */
template <typename scored_type_>
inline std::size_t scored_failures(scored_type_ const* results, std::size_t results_count, std::size_t published_after,
                                   std::size_t results_bound) noexcept {
    std::size_t failures = results_count > results_bound;
    for (std::size_t position = 0; position != results_count; ++position) {
        failures += static_cast<std::size_t>(results[position].node_id) >= published_after;
        if (position) failures += results[position].distance < results[position - 1].distance;
    }
    return failures;
}

/** @brief Shared tally of live bytes & allocation counts, mutated by every @c counting_allocator. */
struct allocation_ledger_t {
    /** @brief Bytes allocated minus bytes freed; must balance to zero at scope exit. */
    std::atomic<std::ptrdiff_t> live_bytes {0};
    /** @brief Lifetime number of @c allocate calls, for coarse allocation-pressure checks. */
    std::atomic<std::size_t> total_allocations {0};
    /** @brief Allocation ceiling in live bytes; scenarios lower it to drive the all-or-
     *      nothing refusal paths, so a request past it is refused, never counted. */
    std::atomic<std::ptrdiff_t> byte_budget {PTRDIFF_MAX};
};

/**
 *  @brief Byte-counting allocator proving propagation through a container: every allocation and
 *      deallocation lands in the shared ledger, which must balance to zero at scope exit.
 *
 *  The default constructor exists only because @c nk::tensor requires a default-constructible
 *  allocator for its empty instances; a defaulted copy aborts on any real allocation, so
 *  every allocation on an audited plane provably flows through ledgered instances. The
 *  audit covers the planes instantiated with this allocator - the CPU cells ledger their
 *  vectors & neighbors planes, while per-context scratch stays on the plain heap.
 */
template <typename value_type_>
struct counting_allocator {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    allocation_ledger_t* ledger_ = nullptr;

    counting_allocator() noexcept = default;
    explicit counting_allocator(allocation_ledger_t& ledger) noexcept : ledger_(&ledger) {}

    template <typename other_type_>
    counting_allocator(counting_allocator<other_type_> const& other) noexcept : ledger_(other.ledger_) {}

    value_type_* allocate(std::size_t count) noexcept {
        if (!ledger_) std::abort(); // A default-constructed instance must never allocate.
        std::ptrdiff_t const wanted = static_cast<std::ptrdiff_t>(count * sizeof(value_type_));
        if (ledger_->live_bytes.load() + wanted > ledger_->byte_budget.load()) return nullptr;
        ledger_->live_bytes.fetch_add(wanted);
        ledger_->total_allocations.fetch_add(1);
        return static_cast<value_type_*>(std::malloc(count * sizeof(value_type_)));
    }
    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (ledger_) ledger_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(count * sizeof(value_type_)));
        std::free(pointer);
    }
    bool operator==(counting_allocator const& other) const noexcept { return ledger_ == other.ledger_; }
};

/** @brief The largest magnitude one stored dimension can carry - per component for the
 *      packed types, the finite maximum for the plain ones. The conditional papers over a
 *      NumKong asymmetry: plain scalars name their @c component_t but only the packed ones
 *      expose @c component_max. */
template <typename scalar_>
constexpr float dimension_amplitude() noexcept {
    if constexpr (requires { scalar_::component_max(); }) return float(scalar_::component_max());
    else return float(scalar_::finite_max());
}

/** @brief Builds a scalar from a small unsigned - sub-byte types go through the raw nibble path,
 *      float wrappers via @c from_f32, byte-or-larger integers via the integral constructor. */
template <typename scalar_>
auto make_scalar(unsigned value) noexcept {
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<i8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** @brief Waiter that misbehaves on purpose: mostly yields, but occasionally oversleeps for
 *      hundreds of microseconds, stretching the lock & watermark contention windows far beyond
 *      anything hardware timing produces - a poor man's schedule fuzzer. */
struct flaky_waiter_t {
    template <typename value_type_, typename thread_index_type_, typename bound_type_>
    void operator()(std::atomic<value_type_> const&, value_type_, thread_index_type_, bound_type_) const noexcept {
        misbehave_();
    }
    template <typename value_type_, typename thread_index_type_, typename bound_type_>
    void operator()(value_type_ const*, value_type_, thread_index_type_, bound_type_) const noexcept {
        misbehave_();
    }

  private:
    static void misbehave_() noexcept {
        // A deliberate exception to the no-thread-locals rule: the index constructs its
        // waiter as a stateless temporary at every wait site, so a persistent per-thread
        // fuzz stream has nowhere else to live until the waiter surface takes instances.
        static thread_local splitmix64_t random_generator {0xF1A6ull};
        if (uniform_below<u32_t> {64}(random_generator) == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        else std::this_thread::yield();
    }
};

/** @brief Builds vectors with planted cluster structure: a few random centroid rows, each
 *      emitted row a centroid copy with a small share of dimensions redrawn. Uniform
 *      random data concentrates pairwise distances into one narrow band - a flat landscape
 *      no graph can navigate, at its worst for binary vectors - while planting bimodalizes
 *      them: within a cluster only the redrawn dimensions differ, across clusters most do,
 *      so self-recall measures navigation instead of fighting near-equidistance. Both
 *      constants derive from the shape: eight centroids keep every cluster deep at the
 *      scenarios' row counts, and redrawing dimensions/16 keeps cluster members distinct
 *      while preserving a several-fold within-to-across distance margin. */
template <typename scalar_>
nk::matrix<scalar_> make_clustered_vectors(std::size_t count, std::size_t dimensions, u64_t seed) {
    constexpr std::size_t centroids_count_k = 8;
    std::size_t const redrawn_dimensions = divide_round_up<std::size_t>(dimensions, 16);
    auto centroids = nk::matrix<scalar_>::try_zeros({centroids_count_k, dimensions});
    auto vectors = nk::matrix<scalar_>::try_zeros({count, dimensions});
    splitmix64_t random_generator {seed};
    uniform_below<u32_t> component_distribution {8};
    uniform_below<u32_t> centroid_distribution {centroids_count_k};
    uniform_below<u32_t> dimension_distribution {static_cast<u32_t>(dimensions)};
    for (std::size_t centroid = 0; centroid != centroids_count_k; ++centroid) {
        auto centroid_span = centroids.row(centroid).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            centroid_span[dimension] = make_scalar<scalar_>(component_distribution(random_generator));
    }
    for (std::size_t row = 0; row != count; ++row) {
        auto row_span = vectors.row(row).as_vector();
        auto centroid_span = centroids.row(centroid_distribution(random_generator)).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            row_span[dimension] = centroid_span[dimension];
        for (std::size_t redraw = 0; redraw != redrawn_dimensions; ++redraw)
            row_span[dimension_distribution(random_generator)] = make_scalar<scalar_>(
                component_distribution(random_generator));
    }
    return vectors;
}

/** @brief Fills a row-major matrix with pseudo-random small values from {0..7}, exactly
 *      representable in every supported scalar type down to i4 nibbles. Binary vectors
 *      route through the planted-cluster generator: their flat draw has no navigable
 *      structure, and each bit keeps a fair coin either way. */
template <typename scalar_>
nk::matrix<scalar_> make_vectors(std::size_t count, std::size_t dimensions, u64_t seed) {
    if constexpr (std::is_same_v<scalar_, nk::u1x8_t>) return make_clustered_vectors<scalar_>(count, dimensions, seed);
    auto vectors = nk::matrix<scalar_>::try_zeros({count, dimensions});
    splitmix64_t random_generator {seed};
    uniform_below<u32_t> component_distribution {8};
    for (std::size_t row = 0; row != count; ++row) {
        auto row_span = vectors.row(row).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            row_span[dimension] = make_scalar<scalar_>(component_distribution(random_generator));
    }
    return vectors;
}
} // namespace unum::usearch

/** @brief Teaches the expect verbs to print statuses by name - @c status_t is the most
 *      compared enum in the suite, and {fmt} refuses unknown enums at compile time. */
template <>
struct fmt::formatter<unum::usearch::status_t> : fmt::formatter<char const*> {
    auto format(unum::usearch::status_t status, format_context& context) const {
        return fmt::formatter<char const*>::format(unum::usearch::status_to_string(status), context);
    }
};

#pragma region Suite Entry Points

std::size_t test_ram_buffers();

std::size_t test_ram_tables();

std::size_t test_dense_all();

std::size_t test_sentences();

std::size_t test_bioinformatics();

std::size_t test_meshes();

#pragma endregion Suite Entry Points
