/**
 *  @brief Shared helpers for the C++ test suites: failure counting, ledgered allocators,
 *      and universally-representable synthetic data generation.
 *
 *  @author Ash Vardanian
 *  @file cpp/test.hpp
 *  @date July 17, 2026
 *
 *  Checks return their own failure counts and roll up to @c main without global mutable
 *  state. @c expect reports the location alone; @c expect_eq also prints both values, for
 *  the places where diagnosis needs data - ledger balances, sizes, recall ratios.
 */
#pragma once

#include <atomic>  // `std::atomic`
#include <cstdlib> // `std::malloc`, `std::free`, `std::abort`

#include <fmt/format.h>

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/types.hpp"

namespace unum::usearch {

#define expect(cond) expect_((bool)(cond), __FILE__, __LINE__)
inline std::size_t expect_(bool condition, char const* file, int line) {
    if (!condition) fmt::print(stderr, "  check failed at {}:{}\n", file, line);
    return condition ? 0u : 1u;
}

#define expect_eq(actual, wanted) expect_eq_((actual), (wanted), __FILE__, __LINE__)
template <typename actual_type_, typename wanted_type_>
inline std::size_t expect_eq_(actual_type_ const& actual, wanted_type_ const& wanted, char const* file, int line) {
    if (actual == wanted) return 0;
    fmt::print(stderr, "  check failed at {}:{}: expected {}, got {}\n", file, line, wanted, actual);
    return 1;
}

/** @brief Shared tally of live bytes & allocation counts, mutated by every @c counting_allocator. */
struct allocation_ledger_t {
    /** @brief Bytes allocated minus bytes freed; must balance to zero at scope exit. */
    std::atomic<std::ptrdiff_t> live_bytes {0};
    /** @brief Lifetime number of @c allocate calls, for coarse allocation-pressure checks. */
    std::atomic<std::size_t> total_allocations {0};
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
        ledger_->live_bytes.fetch_add(static_cast<std::ptrdiff_t>(count * sizeof(value_type_)));
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
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<std::int8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** @brief Minimal SplitMix64-based generator for synthetic data - deliberately independent of the
 *      library's internal PRNG, so a generator bug there can never shape the test data. */
struct test_random_t {
    u64_t state_;

    explicit test_random_t(u64_t seed) noexcept : state_(seed) {}

    /** @brief The next pseudo-random value in [0, @p bound). */
    unsigned below(unsigned bound) noexcept {
        u64_t mixed = (state_ += 0x9E3779B97F4A7C15ull);
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        return static_cast<unsigned>((mixed ^ (mixed >> 31)) % bound);
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
    test_random_t random_generator(seed);
    for (std::size_t centroid = 0; centroid != centroids_count_k; ++centroid) {
        auto centroid_span = centroids.row(centroid).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            centroid_span[dimension] = make_scalar<scalar_>(random_generator.below(8));
    }
    for (std::size_t row = 0; row != count; ++row) {
        auto row_span = vectors.row(row).as_vector();
        auto centroid_span = centroids.row(random_generator.below(centroids_count_k)).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            row_span[dimension] = centroid_span[dimension];
        for (std::size_t redraw = 0; redraw != redrawn_dimensions; ++redraw)
            row_span[random_generator.below(static_cast<unsigned>(dimensions))] = make_scalar<scalar_>(
                random_generator.below(8));
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
    test_random_t random_generator(seed);
    for (std::size_t row = 0; row != count; ++row) {
        auto row_span = vectors.row(row).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            row_span[dimension] = make_scalar<scalar_>(random_generator.below(8));
    }
    return vectors;
}
} // namespace unum::usearch
