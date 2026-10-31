/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Shared helpers for the C++ test suites: failure counting, ledgered allocators, and
 *      universally-representable synthetic data generation.
 *
 *  Checks return their own failure counts and roll up to @c main without global mutable
 *  state. Four verbs cover every case: @c expect for a condition, @c expect_eq when diagnosis
 *  needs the values, @c expect_errors for a numeric pipeline's errors against its exact
 *  reference, and @c report for the verdict column. A status becomes a condition through
 *  @c succeeded(), so no fifth verb is needed.
 *
 *  The suite reads its settings once, at the top of @c main, into the one @c settings_t the
 *  benchmarks and tools read too, with @c USEARCH_SEED, @c USEARCH_FILTER, @c USEARCH_DEVICES and
 *  @c USEARCH_NVME_PATHS from the environment.
 */
#pragma once

#include <cmath>   // `std::fabs`, `std::fmax`, `std::sqrt` - the statistics of `expect_errors`
#include <cstdio>  // `std::remove` - a viewed checkpoint's files
#include <cstdlib> // `std::malloc`, `std::free`, `std::abort`
#include <cstring> // `std::memset` - the scribble over a freed block

#include <algorithm>   // `std::sort`, `std::unique` - the dispensed-ID density check
#include <array>       // `std::array` - one latent point of the sign sketches
#include <bit>         // `std::bit_cast` - the units in the last place of `error_stats_t`
#include <chrono>      // `std::chrono::microseconds` - the flaky waiter's oversleep
#include <limits>      // `std::numeric_limits` - the unbounded allocation budget
#include <span>        // `std::span` - the dispensed-ID density check
#include <string>      // `std::string` - a cell's checkpoint stem
#include <string_view> // `std::string_view` - the stream names
#include <thread>      // `std::this_thread::yield`, `std::this_thread::sleep_for`
#include <vector>      // `std::vector` - the gathered predicate items

#include <fmt/format.h>

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/types.h" // `usearch_version_name`, `usearch_capabilities_compiled`, `usearch_capabilities_available`
#include "usearch/utilities/types.hpp"

#include "abi.hpp" // `tools::settings_t`, `tools::copied_matches_t` - what every suite shares

/** The short spelling every test reaches the library through where its `using` directive is
 *  ambiguous - Darwin declares a global `uuid_t` too - or unwanted, as inside foreign specializations. */
namespace us = unum::usearch;

namespace unum::usearch::test {
using namespace unum::usearch::utilities;

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

/** The relative error a @c -ffast-math build leaves, which @c expect_errors allows. */
inline constexpr f64_t approximate_tolerance_k = 1e-3;

/** How a cell's results relate to their exact reference, the families NumKong's harness judges its
 *  kernels by: equal, within rounding, within rounding of the largest reference for weighted
 *  averages that pass through zero, or only statistically close for paths approximate by design. */
enum class comparison_family_t { exact_k, approximate_k, normalized_reduction_k, bounded_k };

/** Whether a cell beyond its family's bound fails the suite or only warns: sparse paths warn, exact
 *  paths assert. */
enum class comparison_verdict_t { assert_k, warn_k };

/** Errors of one cell against its exact reference: max absolute and relative errors, the absolute
 *  errors' mean and Welford sum of squares that sigma bounds read, and the largest distance in
 *  @c f32 units in the last place. */
struct error_stats_t {
    std::size_t count = 0;
    f64_t max_absolute_error = 0;
    f64_t max_relative_error = 0;
    f64_t max_reference = 0;
    f64_t mean = 0;
    f64_t sum_of_squares = 0;
    u64_t max_ulp = 0;
    std::size_t exact_matches = 0;

    void add(f64_t actual, f64_t reference) noexcept {
        f64_t const difference = actual == reference ? 0.0 : std::fabs(actual - reference);
        // NaN counts as the largest error, which every family's bound refuses.
        f64_t const error = std::isnan(difference) ? std::numeric_limits<f64_t>::infinity() : difference;
        ++count;
        exact_matches += error == 0;
        max_absolute_error = std::fmax(max_absolute_error, error);
        max_relative_error = std::fmax(max_relative_error,
                                       error / std::fmax(std::fmax(std::fabs(actual), std::fabs(reference)), 1.0));
        max_reference = std::fmax(max_reference, std::fabs(reference));
        f64_t const delta = error - mean;
        mean += delta / static_cast<f64_t>(count);
        sum_of_squares += delta * (error - mean);
        max_ulp = std::max(max_ulp, ulp_distance(static_cast<f32_t>(actual), static_cast<f32_t>(reference)));
    }

    f64_t deviation() const noexcept {
        return count > 1 ? std::sqrt(sum_of_squares / static_cast<f64_t>(count - 1)) : 0.0;
    }

    static u64_t ulp_distance(f32_t first, f32_t second) noexcept {
        auto const ordered = [](f32_t value) noexcept {
            u32_t const bits = std::bit_cast<u32_t>(value);
            return bits >> 31 ? -static_cast<i64_t>(bits & 0x7FFFFFFFu) : static_cast<i64_t>(bits);
        };
        i64_t const gap = ordered(first) - ordered(second);
        return static_cast<u64_t>(gap < 0 ? -gap : gap);
    }
};

/** Judges the statistics of @p cell by its @p family's bound and, when @p sigmas is positive, by no
 *  error past rounding lying beyond @p sigmas standard deviations past the mean; on @c warn_k
 *  prints a warning instead of failing. A cell of no samples judged nothing and always fails.
 *  Answers the failures to add. */
inline std::size_t expect_errors(char const* cell, comparison_family_t family, comparison_verdict_t verdict,
                                 error_stats_t const& errors, double sigmas) {
    if (!errors.count) {
        fmt::println(stderr, "  check failed in {}: no samples were compared", cell);
        return 1;
    }
    bool within = std::isfinite(errors.max_absolute_error);
    switch (family) {
    case comparison_family_t::exact_k: within = within && errors.exact_matches == errors.count; break;
    case comparison_family_t::approximate_k:
        within = within && errors.max_relative_error <= approximate_tolerance_k;
        break;
    case comparison_family_t::normalized_reduction_k:
        within = within && errors.max_absolute_error <= approximate_tolerance_k * std::fmax(errors.max_reference, 1.0);
        break;
    case comparison_family_t::bounded_k: break;
    }
    // Rounding-level outliers say nothing about a path, so the sigma bound reads only the errors
    // beyond rounding level.
    if (sigmas > 0 && errors.max_relative_error > approximate_tolerance_k)
        within = within && errors.max_absolute_error <= errors.mean + sigmas * errors.deviation();
    if (within) return 0;
    fmt::println(stderr,
                 "  {} {}: {} of {} exact, max abs {:.4g}, max rel {:.4g}, max ulp {}, mean abs {:.4g} ± {:.4g}",
                 verdict == comparison_verdict_t::warn_k ? "warning in" : "check failed in", cell, errors.exact_matches,
                 errors.count, errors.max_absolute_error, errors.max_relative_error, errors.max_ulp, errors.mean,
                 errors.deviation());
    return verdict == comparison_verdict_t::assert_k;
}

/** One verdict column, whichever way the case ended; answers the count it was given,
 *  so a caller accumulates through it. The @c char-pointer overload names a skip. */
inline std::size_t report(char const* name, std::size_t failures) {
    fmt::println("{}  {:<20} : {} failures", failures ? "FAILED" : "passed", name, failures);
    return failures;
}

inline std::size_t report(char const* name, char const* skip_reason) {
    fmt::println("skipped  {:<20} : {}", name, skip_reason);
    return 0;
}

/** A per-item test filter in the quad shape the engines ask, each item answered on its own. */
template <typename per_item_type_>
auto quad_predicate(per_item_type_ per_item) noexcept {
    return [per_item]<typename item_type_, std::size_t extent_>(std::span<item_type_ const, extent_> items,
                                                                std::span<admission_t, extent_> admissions) noexcept {
        for (std::size_t lane = 0; lane != items.size(); ++lane) admissions[lane] = per_item(items[lane]);
    };
}

/** A per-item test filter in the aggregating shape: it gathers every item an engine pushes beside
 *  its payload and answers them all at the drain. */
template <typename per_item_type_, typename item_type_, typename payload_type_>
struct gathering_predicate {
    per_item_type_ per_item;
    std::vector<item_type_> items;
    std::vector<payload_type_> payloads;

    status_t reserve(std::size_t capacity) noexcept {
        items.reserve(capacity);
        payloads.reserve(capacity);
        return status_t::success_k;
    }

    template <std::size_t extent_>
    void push(std::span<item_type_ const, extent_> asked, std::span<payload_type_ const, extent_> carried) noexcept {
        items.insert(items.end(), asked.begin(), asked.end());
        payloads.insert(payloads.end(), carried.begin(), carried.end());
    }

    template <typename on_admitted_type_>
    void drain(on_admitted_type_&& on_admitted) noexcept {
        for (std::size_t position = 0; position != items.size(); ++position)
            if (per_item(items[position]) == admission_t::admit_k) on_admitted(payloads[position]);
        items.clear();
        payloads.clear();
    }
};

/** One @ref gathering_predicate per executor thread, the family an engine driving its own threads
 *  hands each thread its own of. */
template <typename per_item_type_, typename item_type_, typename payload_type_>
struct gathering_predicates {
    using thread_predicate_t = gathering_predicate<per_item_type_, item_type_, payload_type_>;
    std::vector<thread_predicate_t> threads;

    thread_predicate_t& of_thread(std::size_t thread) noexcept { return threads[thread]; }
};

/** A @ref gathering_predicates of @p threads, each gathering @p item_type_ beside @p payload_type_
 *  and admitting what @p per_item admits. */
template <typename item_type_, typename payload_type_, typename per_item_type_>
gathering_predicates<per_item_type_, item_type_, payload_type_> gathering_per_thread(per_item_type_ per_item,
                                                                                     std::size_t threads) {
    return {std::vector(threads, gathering_predicate<per_item_type_, item_type_, payload_type_> {per_item, {}, {}})};
}

#pragma region Settings

/** Every suite reads the settings the benchmarks and tools read, once, at the top of @c main. */
using us::tools::print;
using us::tools::read_environment;
using us::tools::read_settings;
using us::tools::settings_t;

/** One unit's suites, as it registers them with the runner: a GPU vendor's, run once per selected
 *  device of that vendor, or a host unit's under @c gpu_vendor_t::unknown_k, run once with no
 *  device, so the runner names neither the vendors nor the dtypes a build compiled. */
struct registered_suite_t {
    gpu_vendor_t vendor = gpu_vendor_t::unknown_k;
    std::size_t (*check)(settings_t const& settings, gpu_device_t const& device) = nullptr;
};

/** Every vendor's driver table, shared with the tools' read-backs. */
using us::tools::gpu_drivers;

/** Every device the C interface numbers, in its order, which @c USEARCH_DEVICES indexes. */
std::span<gpu_device_t const> gpu_devices() noexcept;

/** Registers @p added when it names a check, and answers every suite registered so far. */
inline std::span<registered_suite_t const> registered_suites(registered_suite_t added = {}) noexcept {
    static registered_suite_t suites[16];
    static std::size_t count = 0;
    if (added.check && count != std::size(suites)) suites[count++] = added;
    return {suites, count};
}

/** The names the C interface reports for this build and this machine, so the suite checks the strings users see. */
struct machine_t {
    char const* version = nullptr;
    char const* compiled = nullptr;
    char const* available = nullptr;
};

/** Probes the names @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    return {usearch_version_name(), usearch_capabilities_compiled(), usearch_capabilities_available()};
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const& machine, std::FILE* stream = stdout) {
    fmt::println(stream, "USearch {}", machine.version);
    fmt::println(stream, "- Compiled for: {}", machine.compiled);
    fmt::println(stream, "- This machine: {}", machine.available);
}

/** The 64-bit FNV-1a hash of @p text, folding stream names into @ref stream_key. */
constexpr u64_t fnv1a64(std::string_view text) noexcept {
    u64_t hash = 0xCBF29CE484222325ull;
    for (char const character : text) hash = (hash ^ static_cast<u8_t>(character)) * 0x100000001B3ull;
    return hash;
}

/** The @p index-th stream of the family @p name under the run's @p seed, identical in C++, Rust and Python. */
constexpr u64_t stream_key(u64_t seed, std::string_view name, u64_t index = 0) noexcept {
    return mix_bits(mix_bits(seed ^ fnv1a64(name)) + index);
}

#pragma endregion Settings

/** IDs dispensed to racing lanes must come out dense & duplicate-free: sorted in
 *  place, unique, and running from @p first_id through the half-open bound
 *  @p first_id plus the ID count. */
template <typename node_id_type_>
inline std::size_t expect_contiguous_ids(std::span<node_id_type_> ids, std::size_t first_id) {
    std::sort(ids.begin(), ids.end());
    std::size_t failures = expect(std::unique(ids.begin(), ids.end()) == ids.end());
    failures += expect(!ids.empty() && static_cast<std::size_t>(ids.front()) == first_id &&
                       static_cast<std::size_t>(ids.back()) == first_id + ids.size() - 1);
    return failures;
}

/** Spin ceiling for waits on prompt flags - flags a peer sets within bounded
 *  work: a gate entry, a hold acquisition, a refusal confirmation. Expiry proves a
 *  broken rendezvous. Waits on a peer's completion of unbounded-duration work are
 *  deliberately not hatched at this bound - see @ref attempt_limit_k. */
constexpr std::size_t spin_limit_k = std::size_t {1} << 28;

/** Retry ceiling for settle phases & sweep loops: once every racer is done, an
 *  operation retried this many times without success has provably starved. */
constexpr std::size_t attempt_limit_k = std::size_t {1} << 20;

/** Spins - yielding - until @p condition holds or @c spin_limit_k expires;
 *  answers whether the condition held. It never counts failures itself: the caller
 *  owns the on-expiry story - count & proceed so a hold can die, count & return, or
 *  open a gate before leaving so the pool cannot deadlock. */
template <typename condition_type_>
[[nodiscard]] inline bool spin_until(condition_type_&& condition) noexcept {
    for (std::size_t spins = 0; spins != spin_limit_k; ++spins) {
        if (condition()) return true;
        std::this_thread::yield();
    }
    return false;
}

/** The reader-lane invariants over one search's scored results: every ID below the watermark
 *  sampled after the search, distances ascending, the candidate breadth bound honored. Answers the
 *  violation count silently - racing lanes add it to their shared counter, since printing from
 *  inside a race would interleave & flood - and the status verdict stays at the call site: lanes
 *  differ in how they record it. */
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

/** Shared tally of live bytes & allocation counts, mutated by every @c counting_allocator. */
struct allocation_ledger_t {

    /** Bytes allocated minus bytes freed; must balance to zero at scope exit. */
    std::ptrdiff_t live_bytes = 0;

    /** Lifetime number of @c allocate calls, for coarse allocation-pressure checks. */
    std::size_t total_allocations = 0;

    /** Allocation ceiling in live bytes; scenarios lower it to drive the all-or-
     *  nothing refusal paths, so a request past it is refused, never counted. */
    std::ptrdiff_t byte_budget = std::numeric_limits<std::ptrdiff_t>::max();
};

/**
 *  @brief Byte-counting allocator proving propagation through a container: every allocation and
 *      deallocation is tallied in the shared ledger, which must balance to zero at scope exit.
 *
 *  The default constructor exists only because @c nk::tensor requires a default-constructible
 *  allocator for its empty instances; a defaulted copy aborts on any real allocation, so
 *  every allocation on an audited matrix provably flows through ledgered instances. The audit
 *  covers the matrices instantiated with this allocator - the CPU cells ledger their vectors
 *  & neighbors matrices, while per-context scratch stays on the plain heap.
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
        fu::standard_atomic_ref<std::ptrdiff_t> const live_bytes {ledger_->live_bytes};
        if (live_bytes.load() + wanted > fu::standard_atomic_ref<std::ptrdiff_t>(ledger_->byte_budget).load())
            return nullptr;
        live_bytes.add(wanted);
        fu::standard_atomic_ref<std::size_t>(ledger_->total_allocations).add(1);
        return static_cast<value_type_*>(std::malloc(count * sizeof(value_type_)));
    }
    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (ledger_)
            fu::standard_atomic_ref<std::ptrdiff_t>(ledger_->live_bytes)
                .sub(static_cast<std::ptrdiff_t>(count * sizeof(value_type_)));
        // Scribbled first, so a read after free sees garbage in every build rather than stale bytes.
        if (pointer) std::memset(static_cast<void*>(pointer), 0xA5, count * sizeof(value_type_));
        std::free(pointer);
    }
    bool operator==(counting_allocator const& other) const noexcept { return ledger_ == other.ledger_; }
};

/** The byte specialization every fixture places its index through. */
using counting_allocator_t = counting_allocator<byte_t>;

/**
 *  @brief Distinct levels one component of @p scalar_ draws from: a packed bit has two, since
 *      its proxy sets the bit for any nonzero and a wider draw would bias it toward ones; every
 *      other type keeps the eight levels the fixtures are built around.
 *  @sa @c nk::fill_uniform, which ranges every dtype by its own @c finite_min & @c finite_max.
 */
template <typename scalar_>
constexpr u32_t random_component_levels() noexcept {
    if constexpr (std::is_same_v<scalar_, nk::u1x8_t>) return 2;
    else return 8;
}

/** Builds a scalar from a small unsigned - sub-byte types go through the raw nibble path,
 *  float wrappers via @c from_f32, byte-or-larger integers via the integral constructor. */
template <typename scalar_>
auto make_scalar(unsigned value) noexcept {
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<i8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** Waiter that misbehaves on purpose: mostly yields, but occasionally oversleeps for
 *  hundreds of microseconds, stretching the lock & watermark contention windows far beyond
 *  anything hardware timing produces - a poor man's schedule fuzzer. */
struct flaky_waiter_t {
    template <typename value_type_, typename thread_index_type_, typename bound_type_ = wait_capped_t>
    void operator()(value_type_ const*, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {
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

/** A uniform real in @b [lower,upper) - the real-valued sibling of @ref uniform_below, identical on every standard
 *  library, where @c std::uniform_real_distribution is implementation-defined and would draw different test data on
 *  each. */
template <typename real_type_>
struct uniform_between;

template <>
struct uniform_between<f64_t> {
    f64_t lower = 0, upper = 1;

    /** One draw from the top 53 random bits, the double's whole mantissa. */
    template <typename generator_type_>
    constexpr f64_t operator()(generator_type_& generator) const noexcept {
        return lower + static_cast<f64_t>(generator() >> 11) * 0x1p-53 * (upper - lower);
    }
};

/** Builds binary vectors as sign sketches of points in a small latent space: bit @c b of a row is the side of the
 *  @c b-th random hyperplane its point falls on, the way binary embeddings arise. Hamming distance then tracks the
 *  angle between points, so the landscape stays graded, where independent bits - or clusters of independently
 *  flipped ones - leave every pair near-equidistant and no walk can navigate. Eight latent dimensions keep
 *  neighboring rows distinct at the scenarios' row counts. */
template <typename scalar_, typename allocator_type_ = nk::aligned_allocator<scalar_>>
nk::matrix<scalar_, allocator_type_> make_sketched_vectors(std::size_t count, std::size_t dimensions, u64_t seed,
                                                           allocator_type_ allocator = {}) {
    constexpr std::size_t latent_dimensions_k = 8;
    // One set of hyperplanes for every call, as one embedding model serves every batch: slices drawn under
    // different seeds, like the concurrent builds', still fall in one sketch space.
    splitmix64_t hyperplanes_generator {0x5E7C4ull}, points_generator {seed};
    uniform_between<f64_t> const centered {-1, 1};
    auto hyperplanes = nk::matrix<f64_t>::zeros({dimensions, latent_dimensions_k}).value;
    for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
        for (std::size_t latent = 0; latent != latent_dimensions_k; ++latent)
            hyperplanes(dimension, latent) = centered(hyperplanes_generator);
    auto vectors = nk::matrix<scalar_, allocator_type_>::zeros({count, dimensions}, allocator).value;
    std::array<f64_t, latent_dimensions_k> point;
    for (auto row : vectors.rows()) {
        for (f64_t& coordinate : point) coordinate = centered(points_generator);
        auto bits = row.as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension) {
            f64_t side = 0;
            for (std::size_t latent = 0; latent != latent_dimensions_k; ++latent)
                side += hyperplanes(dimension, latent) * point[latent];
            bits[dimension] = make_scalar<scalar_>(side > 0);
        }
    }
    return vectors;
}

/** Fills a row-major matrix with pseudo-random small values from {0..7}, exactly
 *  representable in every supported scalar type down to i4 nibbles. Binary vectors
 *  route through the sign-sketch generator: their flat draw has no navigable structure. */
template <typename scalar_, typename allocator_type_ = nk::aligned_allocator<scalar_>>
nk::matrix<scalar_, allocator_type_> make_vectors(std::size_t count, std::size_t dimensions, u64_t seed,
                                                  allocator_type_ allocator = {}) {
    if constexpr (std::is_same_v<scalar_, nk::u1x8_t>)
        return make_sketched_vectors<scalar_>(count, dimensions, seed, allocator);
    auto vectors = nk::matrix<scalar_, allocator_type_>::zeros({count, dimensions}, allocator).value;
    splitmix64_t random_generator {seed};
    uniform_below<u32_t> component_distribution {random_component_levels<scalar_>()};
    for (auto row : vectors.rows()) {
        auto values = row.as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            values[dimension] = make_scalar<scalar_>(component_distribution(random_generator));
    }
    return vectors;
}

using us::tools::copied_matches_t;
using us::tools::copy_matches;

std::size_t check_dense(settings_t const& settings);

namespace dense {
std::size_t check_ram_metric(settings_t const& settings);
} // namespace dense

namespace utilities {
std::size_t check_ram_buffers(settings_t const& settings);
std::size_t check_ram_tables(settings_t const& settings);
std::size_t check_safegraphs_format(settings_t const& settings);
} // namespace utilities

namespace basic {
std::size_t check_sentences(settings_t const& settings);
std::size_t check_bioinformatics(settings_t const& settings);
std::size_t check_meshes(settings_t const& settings);
} // namespace basic

namespace sequence {
std::size_t check_sequence(settings_t const& settings);
std::size_t check_sequence_vram(settings_t const& settings, gpu_device_t const& device);
} // namespace sequence

namespace shapes {
std::size_t check_shapes(settings_t const& settings);
std::size_t check_shapes_vram(settings_t const& settings, gpu_device_t const& device);
} // namespace shapes

namespace geospatial {
std::size_t check_geospatial(settings_t const& settings);
std::size_t check_geospatial_vram(settings_t const& settings, gpu_device_t const& device);
} // namespace geospatial

/** The suites over the C interface, each sweeping its modality's diagonal, and the checks every
 *  modality's cells share. */
namespace abi {
using namespace unum::usearch::tools;

std::size_t check_dense_abi(settings_t const& settings);
std::size_t check_sequence_abi(settings_t const& settings);
std::size_t check_shapes_abi(settings_t const& settings);
std::size_t check_geospatial_abi(settings_t const& settings);

/** Entries or queries each call of a cell takes, so a build spans several calls. */
inline constexpr std::size_t cell_batch_k = 384;

/** Matches a cell judges per query. */
inline constexpr std::size_t cell_wanted_k = 10;

/** Inserts @p count entries of @p inputs into the empty @p index under their keys, which must
 *  spread across the cell's shards and each count once where the residency serves a count. */
template <typename verbs_type_>
std::size_t expect_built(sweep_cell_t const& cell, handle<verbs_type_>& index, typename verbs_type_::values_t inputs,
                         std::size_t count) {
    usearch_traversal_stats_t stats {};
    std::size_t failures = check(insert_entries(index, inputs, count, 0, cell_batch_k, 0, stats));
    if (failures) return failures;
    typename verbs_type_::description_t const built = describe(index);
    failures += check(built.size == count && built.shards_count == cell.shards);
    failures += check(stats.computed_distances > 0);
    std::size_t miscounted = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        std::size_t entries = 0;
        verbs_type_::count(index.get(), index.executor(), key_of_offset(index.session->seed, offset), &entries,
                           &outcome);
        // A residency that serves no count yet leaves the keys to the walks' recall.
        if (outcome.status == usearch_status_unsupported_k) break;
        if (outcome.status == usearch_status_success_k && entries == 1) continue;
        if (!miscounted++)
            fmt::println(stderr, "  a key counts {} entries: {}", entries, outcome.message ? outcome.message : "");
    }
    return failures + check(miscounted == 0);
}

/** Writes into @p recall the share of the exact scan's top ten in @p oracle a walk at @p expansion
 *  keeps over the first @p count queries of @p queries, ties at the cut counting however the scan
 *  happened to break them. */
template <typename verbs_type_>
usearch_outcome_t walk_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                              std::size_t expansion, copied_matches_t const& oracle, double& recall) {
    copied_matches_t found;
    usearch_traversal_stats_t stats {};
    usearch_outcome_t const walked = search_queries(index, queries, count, cell_batch_k, cell_wanted_k, expansion,
                                                    usearch_traversal_walk_k, stats, found);
    if (walked.status == usearch_status_success_k) recall = recall_at(cell_wanted_k, found, oracle);
    return walked;
}

/** A walk at @p expansion keeps @p floor of the exact scan's top ten over the first @p count
 *  queries of @p queries, as @ref walk_recall judges it; a residency that serves no scan yet has no
 *  oracle to judge by. */
template <typename verbs_type_>
std::size_t expect_walk_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                               std::size_t expansion, double floor) {
    copied_matches_t oracle;
    usearch_outcome_t const scanned = scan_oracle(index, queries, count, cell_wanted_k, cell_batch_k, oracle);
    if (scanned.status == usearch_status_unsupported_k) return 0;
    double recall = 0;
    std::size_t const failures = check(scanned) || check(walk_recall(index, queries, count, expansion, oracle, recall));
    if (failures) return failures;
    if (recall < floor) fmt::println(stderr, "  recall at ten {:.3f} under the floor {:.3f}", recall, floor);
    return check(recall >= floor);
}

/** The index round-trips through all residencies under @p stem, as @ref check_checkpoint_round_trip
 *  judges it on the first @p count queries of @p queries, within @p score_band where stored entries
 *  round; in an NVMe cell its saved view then answers the same walk and refuses every write, one
 *  entry of @p inputs included. */
template <typename verbs_type_>
std::size_t expect_persistence(sweep_cell_t const& cell, handle<verbs_type_>& index,
                               typename verbs_type_::values_t inputs, typename verbs_type_::queries_t queries,
                               std::size_t count, std::size_t expansion, double floor, std::string const& stem,
                               saved_bytes_t bytes, double score_band) {
    // Entries that round on the way may steer a walk down another path, so they compare by exact
    // scans wherever the residency serves one.
    auto const search = [&](handle<verbs_type_>& searched) {
        copied_matches_t found;
        usearch_traversal_stats_t stats {};
        usearch_outcome_t outcome {};
        if (score_band)
            outcome = search_queries(searched, queries, count, cell_batch_k, cell_wanted_k, 0, usearch_traversal_scan_k,
                                     stats, found);
        if (!score_band || outcome.status == usearch_status_unsupported_k)
            found = {}, outcome = search_queries(searched, queries, count, cell_batch_k, cell_wanted_k, expansion,
                                                 usearch_traversal_walk_k, stats, found);
        return outcome.status == usearch_status_success_k ? found : copied_matches_t {};
    };
    std::size_t failures = check_checkpoint_round_trip(index, stem, cell.shards, bytes, score_band, search);
    if (cell.residency != usearch_residency_nvme_k) return failures;
    handle<verbs_type_> view = save_and_reopen(std::move(index), stem + "-view.safegraphs", usearch_residency_nvme_k);
    failures += check(bool(view));
    if (view) {
        failures += expect_walk_recall(view, queries, count, expansion, floor);
        usearch_outcome_t outcome {};
        usearch_key_t const key = key_of_offset(view.session->seed, 0);
        verbs_type_::insert(view.get(), view.executor(), 1, keys_view({&key, 1}), inputs, 0, nullptr, &outcome);
        failures += check(outcome.status == usearch_status_read_only_k);
        verbs_type_::clear(view.get(), view.executor(), &outcome);
        failures += check(outcome.status == usearch_status_read_only_k);
    }
    view = {};
    for (std::string const& file : files_starting_with(stem + "-view")) std::remove(file.c_str());
    return failures;
}

/** Two refinements settle: each changes no more entries than it re-selected, the second no more
 *  than the first, and a walk at @p expansion keeps as much of the exact scan's top ten over the
 *  first @p count queries of @p queries, so the scan and the walk that judge it must both succeed;
 *  a VRAM index whose tower still declares its refinement answers @p on_device instead and is left
 *  out of the judging. */
template <typename verbs_type_>
std::size_t expect_refined(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                           std::size_t expansion, usearch_status_t on_device) {
    copied_matches_t oracle;
    double recall_before = 0, recall_after = 0;
    char const* refused = nullptr;
    usearch_outcome_t oracle_outcome = scan_oracle(index, queries, count, cell_wanted_k, cell_batch_k, oracle);
    if (oracle_outcome.status != usearch_status_success_k) refused = "scan_oracle";
    else if (oracle_outcome = walk_recall(index, queries, count, expansion, oracle, recall_before);
             oracle_outcome.status != usearch_status_success_k)
        refused = "walk_recall";
    usearch_refine_report_t reports[2] {};
    std::size_t failures = 0;
    for (usearch_refine_report_t& report : reports) {
        usearch_outcome_t outcome {};
        verbs_type_::refine(index.get(), index.executor(), &report, &outcome);
        if (index.residency == usearch_residency_vram_k && on_device != usearch_status_success_k)
            return check(outcome.status == on_device);
        failures += check(outcome) + check(report.changed <= report.reselected);
    }
    failures += check(reports[1].changed <= reports[0].changed);
    if (refused) {
        fmt::println(stderr, "  the recall oracle's {} refused with {}: {}", refused,
                     usearch_status_name(oracle_outcome.status),
                     oracle_outcome.message ? oracle_outcome.message : "no message");
        return failures + 1;
    }
    failures += check(walk_recall(index, queries, count, expansion, oracle, recall_after));
    if (recall_after < recall_before)
        fmt::println(stderr, "  recall at ten fell from {:.3f} to {:.3f} over two refinements", recall_before,
                     recall_after);
    return failures + check(recall_after >= recall_before);
}

/** A donor of @p count entries of @p inputs, keyed past the index's own @p first_offset entries,
 *  absorbs whole: its slot is consumed and the sizes add up; a device index answers @p on_device
 *  instead, where absorption is not implemented yet. */
template <typename verbs_type_>
std::size_t expect_absorbed(handle<verbs_type_>& index, typename verbs_type_::values_t inputs, std::size_t count,
                            std::size_t first_offset, usearch_status_t on_device) {
    handle<verbs_type_> donor = open_index<verbs_type_>(*index.session, index.geometry, 1);
    if (!donor) return 1;
    usearch_traversal_stats_t stats {};
    std::size_t failures = check(insert_entries(donor, inputs, count, first_offset, cell_batch_k, 0, stats));
    if (failures) return failures;
    usearch_outcome_t outcome {};
    verbs_type_::absorb(index.get(), index.executor(), &donor.index, 1, &outcome);
    if (index.residency == usearch_residency_vram_k && on_device != usearch_status_success_k)
        return check(outcome.status == on_device);
    failures += check(outcome) + check(!donor);
    return failures + check(describe(index).size == first_offset + count);
}

/** A cleared index holds nothing. */
template <typename verbs_type_>
std::size_t expect_cleared(handle<verbs_type_>& index) {
    usearch_outcome_t outcome {};
    verbs_type_::clear(index.get(), index.executor(), &outcome);
    return check(outcome) + check(describe(index).size == 0);
}

} // namespace abi

} // namespace unum::usearch::test

/** Teaches the expect verbs to print statuses by name - @c status_t is the most
 *  compared enum in the suite, and {fmt} refuses unknown enums at compile time. */
template <>
struct fmt::formatter<unum::usearch::utilities::status_t> : fmt::formatter<char const*> {
    auto format(us::utilities::status_t status, format_context& context) const {
        return fmt::formatter<char const*>::format(us::utilities::status_to_string(status), context);
    }
};
