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
#include <cstring> // `std::memset`, `std::memcmp` - the scribble over a freed block, the drawn keys

#include <algorithm>   // `std::sort`, `std::unique`, `std::any_of`, `std::find` - the dispensed IDs, draws, cursors
#include <array>       // `std::array` - one latent point of the sign sketches
#include <bit>         // `std::bit_cast` - the units in the last place of `error_stats_t`
#include <chrono>      // `std::chrono::microseconds` - the flaky waiter's oversleep
#include <limits>      // `std::numeric_limits` - the unbounded allocation budget
#include <optional>    // `std::optional` - a yielded key's minted offset
#include <span>        // `std::span` - the dispensed-ID density check
#include <string>      // `std::string` - a cell's checkpoint stem
#include <string_view> // `std::string_view` - the stream names
#include <thread>      // `std::this_thread::yield`, `std::this_thread::sleep_for`
#include <vector>      // `std::vector` - the gathered predicate items

#include <fmt/format.h>

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/types.h" // `usearch_version_name`, `usearch_capabilities_compiled`, `usearch_capabilities_available`
#include "usearch/utilities/ram_buffers.hpp" // `growing_arena` - a sharded cursor's links
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

/** A cursor over the sharded @p indexes, pulled @p chunk keys at a time, yields each of its
 *  @p entries once as stored, and from @p start breadth-first that key first and each reached entry
 *  once, every link a key @p indexes holds once and each entry's run closing at ordered offsets. */
template <typename indexes_type_>
std::size_t expect_sharded_cursor(indexes_type_& indexes, typename indexes_type_::key_t const& start,
                                  std::size_t entries, std::size_t chunk) {
    using key_t = typename indexes_type_::key_t;
    using cursor_t = typename indexes_type_::cursor_t;
    std::size_t failures = 0;
    for (std::span<key_t const> const starts : {std::span<key_t const> {}, std::span<key_t const> {&start, 1}}) {
        growing_arena<forkunion_allocator_t> arena {forkunion_allocator_t {no_memory_domain_k}};
        typename cursor_t::links_allocator_t const links_allocator {arena};
        typename cursor_t::links_keys_t links {links_allocator};
        typename cursor_t::links_offsets_t offsets {links_allocator};
        cursor_t cursor {forkunion_allocator_t {no_memory_domain_k}};
        failures += expect_eq(cursor.open(indexes, starts), status_t::success_k);
        std::vector<key_t> pulled_keys(chunk), pulled;
        while (true) {
            auto const [read, status] = cursor.pull(indexes, std::span {pulled_keys}, &links, &offsets);
            failures += expect_eq(status, status_t::success_k);
            if (!read || failed(status)) break;
            pulled.insert(pulled.end(), pulled_keys.begin(), pulled_keys.begin() + read);
        }
        std::size_t repeated = 0, foreign = 0;
        for (auto yielded = pulled.begin(); yielded != pulled.end(); ++yielded)
            repeated += std::find(pulled.begin(), yielded, *yielded) != yielded;
        for (key_t const& linked : links) foreign += indexes.count(linked) != 1;
        failures += expect_eq(repeated, 0u);
        failures += expect_eq(foreign, 0u);
        failures += expect_eq(offsets.size(), pulled.size());
        failures += expect(std::is_sorted(offsets.begin(), offsets.end()) &&
                           (offsets.empty() || offsets[offsets.size() - 1] == links.size()));
        failures += starts.empty() ? expect_eq(pulled.size(), entries)
                                   : expect(pulled.size() > 1 && pulled.size() <= entries && pulled[0] == start);
    }
    return failures;
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

/** Whether @p first and @p second are the same key, byte for byte. */
inline bool same_key(usearch_key_t const& first, usearch_key_t const& second) noexcept {
    return !std::memcmp(first.bytes, second.bytes, sizeof(first.bytes));
}

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

/** The first @p count queries' first matches hint a second search of them, which answers
 *  @p accepted: on success a key no entry holds is counted once per query, and each query ranks at
 *  least as near as its hint. */
template <typename verbs_type_>
std::size_t expect_hints(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                         std::size_t expansion, usearch_status_t accepted) {
    copied_matches_t found;
    usearch_traversal_stats_t stats {};
    std::size_t failures = check(search_queries(index, queries, count, cell_batch_k, cell_wanted_k, expansion,
                                                usearch_traversal_walk_k, stats, found));
    if (failures) return failures;
    std::vector<std::size_t> const offsets = match_offsets(found);
    std::vector<usearch_key_t> keys(2 * count);
    std::vector<std::size_t> const counts(count, 2);
    for (std::size_t query = 0; query != count; ++query) {
        keys[2 * query] = std::bit_cast<usearch_key_t>(found.keys[offsets[query]]);
        keys[2 * query + 1] = key_of_offset(index.session->seed, std::numeric_limits<u64_t>::max() - query);
    }
    usearch_matches_view_t const hints {keys.data(), nullptr, counts.data(), 2};
    copied_matches_t hinted;
    usearch_traversal_stats_t hinted_stats {};
    // Several batches, so each takes its own rows of the hints.
    usearch_outcome_t const outcome = search_queries(index, queries, count, divide_round_up<std::size_t>(count, 4),
                                                     cell_wanted_k, expansion, usearch_traversal_walk_k, hinted_stats,
                                                     hinted, &hints);
    if (accepted != usearch_status_success_k) return check(outcome.status == accepted);
    failures += check(outcome);
    if (failures) return failures;
    std::vector<std::size_t> const hinted_offsets = match_offsets(hinted);
    std::size_t farther = 0;
    for (std::size_t query = 0; query != count; ++query)
        farther += !hinted.counts[query] || !within(hinted.scores[hinted_offsets[query]], found.scores[offsets[query]]);
    // A warm second call, its scratch already sized, ranks alike.
    copied_matches_t warm;
    usearch_traversal_stats_t warm_stats {};
    failures += check(search_queries(index, queries, count, divide_round_up<std::size_t>(count, 4), cell_wanted_k,
                                     expansion, usearch_traversal_walk_k, warm_stats, warm, &hints));
    failures += check(warm.counts == hinted.counts && warm_stats.missing_hints == count);
    return failures + check(hinted_stats.missing_hints == count) + check(farther == 0);
}

/** An audit reaches every one of @p count entries. */
template <typename verbs_type_>
std::size_t expect_health(handle<verbs_type_>& index, std::size_t count) {
    usearch_outcome_t outcome {};
    usearch_reachability_t reachability {};
    verbs_type_::health(index.get(), index.executor(), &reachability, &outcome);
    return check(outcome) + check(reachability.audited == count && reachability.reachable == count);
}

/** A cursor over @p index yields each of its @p count entries once as stored, and from the first
 *  entry's key breadth-first that key first and then only what links lead to, once each; every
 *  yielded and linked key was minted, and a walk past one entry hands links. */
template <typename verbs_type_>
std::size_t expect_cursor(handle<verbs_type_>& index, std::size_t count) {
    usearch_key_t const start = key_of_offset(index.session->seed, 0);
    std::size_t failures = 0;
    for (std::span<usearch_key_t const> const starts : {std::span<usearch_key_t const> {}, std::span {&start, 1}}) {
        usearch_outcome_t outcome {};
        typename verbs_type_::cursor_t cursor = nullptr;
        verbs_type_::cursor_init(index.get(), starts, &cursor, &outcome);
        failures += check(outcome);
        if (!cursor) continue;
        std::vector<bool> yielded(count);
        std::size_t pulled = 0, repeated = 0, unminted = 0, unordered = 0, written = 0, linked_count = 0;
        bool started = starts.empty();
        while (true) {
            usearch_keys_view_t keys {};
            usearch_keys_tape_view_t links {};
            verbs_type_::cursor_next(index.get(), index.executor(), index.session->arena, cursor, cell_batch_k, &keys,
                                     &links, &written, &outcome);
            failures += check(outcome);
            if (outcome.status != usearch_status_success_k || !written) break;
            std::vector<usearch_key_t> batch(written);
            std::vector<std::size_t> offsets(written + 1);
            failures += check(download(batch.data(), keys.data, written * sizeof(usearch_key_t)));
            failures += check(download(offsets.data(), links.offsets, offsets.size() * sizeof(std::size_t)));
            std::vector<usearch_key_t> linked(offsets.back());
            failures += check(download(linked.data(), links.keys, linked.size() * sizeof(usearch_key_t)));
            if (!pulled && !starts.empty()) started = same_key(batch[0], start);
            for (std::size_t entry = 0; entry != written; ++entry) {
                std::optional<u64_t> const offset = offset_of_key(index.session->seed, batch[entry]);
                unminted += !offset || *offset >= count;
                if (offset && *offset < count) repeated += yielded[*offset], yielded[*offset] = true;
                unordered += offsets[entry] > offsets[entry + 1];
            }
            for (usearch_key_t const& key : linked) unminted += !offset_of_key(index.session->seed, key);
            pulled += written, linked_count += linked.size();
        }
        verbs_type_::cursor_free(cursor);
        failures += check(started) + check(repeated == 0) + check(unminted == 0) + check(unordered == 0) +
                    check(pulled < 2 || linked_count > 0);
        failures += starts.empty() ? check(pulled == count) : check(pulled > 1 && pulled <= count);
    }
    return failures;
}

/** A sample of the whole of @p index draws as many of its @p count entries as it wants, each a
 *  distinct minted key, its key's only entry, hanging from an earlier draw at finite coordinates,
 *  and an equal seed draws the same keys again; a GPU index refuses it and says why. */
template <typename verbs_type_>
std::size_t expect_sample(handle<verbs_type_>& index, std::size_t count) {
    std::size_t const wanted = std::min(cell_wanted_k, count);
    std::array<std::vector<usearch_key_t>, 2> drawn;
    std::size_t failures = 0;
    for (std::vector<usearch_key_t>& keys : drawn) {
        usearch_outcome_t outcome {};
        sample_outputs_view_t outputs {};
        verbs_type_::sample(index.get(), index.executor(), index.session->arena, {}, {}, {}, wanted,
                            index.session->seed, outputs, &outcome);
        if (index.residency == usearch_residency_vram_k)
            return check(outcome.status == usearch_status_unsupported_k && outcome.message);
        if (check(outcome)) return failures + 1;
        failures += check(outputs.written == wanted);
        keys.resize(outputs.written);
        std::vector<std::size_t> ordinals(outputs.written), parented(outputs.written);
        std::vector<usearch_key_t> parents(outputs.written * outputs.parents.columns);
        std::vector<f32_t> coordinates(outputs.written * outputs.coordinates.stride / sizeof(f32_t));
        failures += check(download(keys.data(), outputs.keys.data, keys.size() * sizeof(usearch_key_t)));
        failures += check(download(ordinals.data(), outputs.entry_ordinals, ordinals.size() * sizeof(std::size_t)));
        failures += check(download(parented.data(), outputs.parents.counts, parented.size() * sizeof(std::size_t)));
        failures += check(download(parents.data(), outputs.parents.keys, parents.size() * sizeof(usearch_key_t)));
        failures += check(download(coordinates.data(), outputs.coordinates.data, coordinates.size() * sizeof(f32_t)));
        auto const drawn_before = [&](usearch_key_t const& key, std::size_t draw) {
            return std::any_of(keys.begin(), keys.begin() + draw,
                               [&](usearch_key_t const& earlier) { return same_key(earlier, key); });
        };
        std::size_t misdrawn = 0;
        for (std::size_t draw = 0; draw != keys.size(); ++draw) {
            std::optional<u64_t> const offset = offset_of_key(index.session->seed, keys[draw]);
            misdrawn += !offset || *offset >= count || ordinals[draw] != 0 || drawn_before(keys[draw], draw);
            misdrawn += parented[draw] && !drawn_before(parents[draw * outputs.parents.columns], draw);
            f32_t const* const point = coordinates.data() + draw * outputs.coordinates.stride / sizeof(f32_t);
            misdrawn += !std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]);
        }
        failures += check(misdrawn == 0);
    }
    return failures + check(drawn[0].size() == drawn[1].size() &&
                            std::equal(drawn[0].begin(), drawn[0].end(), drawn[1].begin(), same_key));
}

/** Splitting @p index , whose entries are keyed by its first offsets, into two empty receivers of
 *  its own session and shard count leaves @p index empty and every key in one receiver once, where
 *  the residency counts entries; an NVMe view refuses it as read-only and a GPU index answers
 *  @p on_device, a refusal saying why. */
template <typename verbs_type_>
std::size_t expect_split(handle<verbs_type_>& index, usearch_status_t on_device) {
    usearch_status_t const wanted = index.residency == usearch_residency_nvme_k   ? usearch_status_read_only_k
                                    : index.residency == usearch_residency_vram_k ? on_device
                                                                                  : usearch_status_success_k;
    typename verbs_type_::description_t const before = describe(index);
    std::array<handle<verbs_type_>, 2> receivers {
        open_index<verbs_type_>(*index.session, index.geometry, before.shards_count),
        open_index<verbs_type_>(*index.session, index.geometry, before.shards_count)};
    if (!receivers[0] || !receivers[1]) return 1;
    std::array<typename verbs_type_::index_t, 2> const handles {receivers[0].get(), receivers[1].get()};
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    verbs_type_::split(index.get(), index.executor(), handles.data(), handles.size(), index.session->seed, &report,
                       &outcome);
    if (wanted != usearch_status_success_k) return check(outcome.status == wanted && outcome.message);
    std::size_t failures = check(outcome);
    failures += check(describe(index).size == 0 &&
                      describe(receivers[0]).size + describe(receivers[1]).size == before.size);
    std::size_t miscounted = 0;
    for (std::size_t offset = 0; offset != before.size; ++offset) {
        usearch_key_t const key = key_of_offset(index.session->seed, offset);
        std::size_t landed = 0;
        for (handle<verbs_type_>& receiver : receivers) {
            std::size_t entries = 0;
            verbs_type_::count(receiver.get(), receiver.executor(), key, &entries, &outcome);
            if (outcome.status == usearch_status_unsupported_k) return failures;
            landed += entries;
        }
        miscounted += landed != 1;
    }
    return failures + check(miscounted == 0);
}

/** A RAM index saves into its session's arena, whose bytes name their metadata, and loads from them
 *  into RAM again with as many entries over as many shards, a walk at @p expansion keeping @p floor
 *  of the exact scan's top ten over the first @p count queries of @p queries; an NVMe view refuses
 *  the save as unsupported and a GPU index answers @p on_device, a refusal saying why. */
template <typename verbs_type_>
std::size_t expect_buffer_round_trip(handle<verbs_type_>& index, typename verbs_type_::queries_t queries,
                                     std::size_t count, std::size_t expansion, double floor,
                                     usearch_status_t on_device) {
    usearch_status_t const wanted = index.residency == usearch_residency_nvme_k   ? usearch_status_unsupported_k
                                    : index.residency == usearch_residency_vram_k ? on_device
                                                                                  : usearch_status_success_k;
    usearch_outcome_t outcome {};
    usearch_bytes_view_t monolith {};
    verbs_type_::save_buffer(index.get(), index.executor(), index.session->arena, &monolith, &outcome);
    if (wanted != usearch_status_success_k) return check(outcome.status == wanted && outcome.message);
    std::size_t failures = check(outcome);
    if (failures) return failures;
    std::string bytes(monolith.length, '\0');
    failures += check(download(bytes.data(), monolith.data, bytes.size()));
    typename verbs_type_::metadata_t metadata {};
    verbs_type_::metadata_read_buffer(index.session->executor, bytes_view(bytes), &metadata, &outcome);
    failures += check(outcome) + check(metadata.connectivity != 0);
    std::vector<std::int32_t> placement;
    typename verbs_type_::index_t copy = nullptr;
    verbs_type_::load_ram_buffer(&copy, index.session->executor, bytes_view(bytes),
                                 index.session->ram_options(0, placement), index.geometry, &outcome);
    failures += check(outcome);
    handle<verbs_type_> reloaded {copy, *index.session, usearch_residency_ram_k, index.geometry};
    if (!reloaded) return failures + 1;
    typename verbs_type_::description_t const original = describe(index), restored = describe(reloaded);
    failures += check(restored.size == original.size && restored.shards_count == original.shards_count);
    return failures + expect_walk_recall(reloaded, queries, count, expansion, floor);
}

/** Every metric kind the modality spells reads back by its name, and "unknown" reads back as the
 *  unknown kind. */
template <typename verbs_type_>
std::size_t expect_metric_names() {
    using metric_t = typename verbs_type_::metric_t;
    std::size_t spelled = 0, misread = 0;
    for (u32_t kind = 1; std::string_view {verbs_type_::metric_name(static_cast<metric_t>(kind))} != "unknown";
         ++kind) {
        metric_t const metric = static_cast<metric_t>(kind);
        misread += verbs_type_::metric_from_name(bytes_view(verbs_type_::metric_name(metric))) != metric;
        ++spelled;
    }
    return check(spelled > 0) + check(misread == 0) +
           check(verbs_type_::metric_from_name(bytes_view("unknown")) == metric_t {});
}

/** The index round-trips through all residencies under @p stem, as @ref check_checkpoint_round_trip
 *  judges it on the first @p count queries of @p queries, within @p score_band where stored entries
 *  round; in an NVMe cell its saved view then answers the same walk, audits, walks, samples and
 *  refuses every write, one entry of @p inputs included. */
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
    std::size_t const entries = describe(index).size;
    handle<verbs_type_> view = save_and_reopen(std::move(index), stem + "-view.safegraphs", usearch_residency_nvme_k);
    failures += check(bool(view));
    if (view) {
        failures += expect_walk_recall(view, queries, count, expansion, floor);
        failures += expect_health(view, entries);
        failures += expect_cursor(view, entries);
        failures += expect_sample(view, entries);
        failures += expect_split(view, usearch_status_unsupported_k);
        failures += expect_buffer_round_trip(view, queries, count, expansion, floor, usearch_status_unsupported_k);
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

/** A refinement opened on @p index holds it closed to an insertion of the first entry of @p inputs
 *  under a fresh key, pulls its handler cancels at their first report carry the running totals on
 *  until the pass ends, no more entries changing than were re-selected, and its finish reports at
 *  least the changes the pulls made and audits all @p count entries, the refused one never landing;
 *  a VRAM index whose tower still declares its refinement answers @p on_device when it opens. */
template <typename verbs_type_>
std::size_t expect_refinement(handle<verbs_type_>& index, typename verbs_type_::values_t inputs, std::size_t count,
                              usearch_status_t on_device) {
    usearch_outcome_t outcome {};
    typename verbs_type_::refinement_t refinement = nullptr;
    verbs_type_::refinement_init(index.get(), index.executor(), &refinement, &outcome);
    if (index.residency == usearch_residency_vram_k && on_device != usearch_status_success_k)
        return check(outcome.status == on_device && !refinement);
    std::size_t failures = check(outcome);
    if (!refinement) return failures + 1;
    usearch_key_t const fresh = key_of_offset(index.session->seed, count);
    verbs_type_::insert(index.get(), index.executor(), 1, keys_view({&fresh, 1}), inputs, 0, nullptr, &outcome);
    failures += check(outcome.status == usearch_status_contended_k);
    usearch_progress_t const cancelling {[](std::size_t, std::size_t, void*) noexcept { return usearch_cancel_k; },
                                         nullptr, 0};
    std::size_t reselected = 0, changed = 0, shrunk = 0, pulls = 0;
    // Every pull finishes at least one report's slice of the entries before its handler cancels it.
    do {
        std::size_t pulled_reselected = 0, pulled_changed = 0;
        verbs_type_::refinement_next(index.get(), index.executor(), refinement, cancelling, &pulled_reselected,
                                     &pulled_changed, &outcome);
        shrunk += pulled_reselected < reselected || pulled_changed < changed || pulled_changed > pulled_reselected;
        reselected = pulled_reselected, changed = pulled_changed;
    } while (outcome.status == usearch_status_cancelled_k && ++pulls <= count);
    failures += check(outcome) + check(shrunk == 0);
    usearch_refine_report_t report {};
    verbs_type_::refinement_finish(index.get(), index.executor(), refinement, &report, &outcome);
    failures += check(outcome) + check(report.changed >= changed && report.reachability.audited == count);
    return failures + check(describe(index).size == count);
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

/** A cleared index holds nothing, and a cursor opened over its entries before the clear answers
 *  @c contended_k after it; an index a split emptied first takes the first entry of @p inputs
 *  back, since clearing nothing need not end a cursor. */
template <typename verbs_type_>
std::size_t expect_cleared(handle<verbs_type_>& index, typename verbs_type_::values_t inputs) {
    usearch_traversal_stats_t stats {};
    std::size_t failures = 0;
    if (!describe(index).size) failures += check(insert_entries(index, inputs, 1, 0, cell_batch_k, 0, stats));
    usearch_outcome_t outcome {};
    typename verbs_type_::cursor_t cursor = nullptr;
    verbs_type_::cursor_init(index.get(), {}, &cursor, &outcome);
    failures += check(outcome);
    verbs_type_::clear(index.get(), index.executor(), &outcome);
    failures += check(outcome) + check(describe(index).size == 0);
    if (!cursor) return failures + 1;
    usearch_keys_view_t keys {};
    std::size_t written = 0;
    verbs_type_::cursor_next(index.get(), index.executor(), index.session->arena, cursor, cell_batch_k, &keys, nullptr,
                             &written, &outcome);
    verbs_type_::cursor_free(cursor);
    return failures + check(outcome.status == usearch_status_contended_k && written == 0);
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
