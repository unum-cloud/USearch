/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Shared helpers for the C++ test suites: failure counting, ledgered allocators, and
 *      universally-representable synthetic data generation.
 *
 *  Checks return their own failure counts and roll up to @c main without global mutable
 *  state. Four verbs cover every case: @c expect for a condition, @c expect_eq and
 *  @c expect_near when diagnosis needs the values, and @c report for the verdict column.
 *  A status becomes a condition through @c succeeded(), so no fifth verb is needed.
 *
 *  Both suites read their settings once, at the top of @c main, into one @c settings_t:
 *
 *  @verbatim
 *  USEARCH_SEED    42    Seed every scenario's data derives from, an unsigned integer or random
 *  USEARCH_FILTER  none  Checks to run, by name, as an ECMAScript regex or else as a substring
 *  @endverbatim
 */
#pragma once

#include <cmath>   // `std::fabs`, `std::fmax` - the relative tolerance of `expect_near`
#include <cstdint> // `std::uint32_t` - the seed
#include <cstdlib> // `std::malloc`, `std::free`, `std::abort`, `std::getenv`, `std::exit`
#include <cstring> // `std::memset` - the scribble over a freed block

#include <algorithm>    // `std::sort`, `std::unique` - the dispensed-ID density check
#include <array>        // `std::array` - one latent point of the sign sketches
#include <atomic>       // `std::atomic`
#include <charconv>     // `std::from_chars` - the seed
#include <chrono>       // `std::chrono::microseconds` - the flaky waiter's oversleep
#include <limits>       // `std::numeric_limits` - the unbounded allocation budget
#include <optional>     // `std::optional` - the settings readers
#include <random>       // `std::random_device` - a drawn seed
#include <regex>        // `std::regex`, `std::regex_search` - the filter
#include <span>         // `std::span` - the dispensed-ID density check
#include <string_view>  // `std::string_view` - the settings readers
#include <system_error> // `std::errc`
#include <thread>       // `std::this_thread::yield`, `std::this_thread::sleep_for`
#include <vector>       // `std::vector` - the selected devices

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join` - the selected devices

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/types.h" // `usearch_version_name`, `usearch_capabilities_compiled`, `usearch_capabilities_available`
#include "usearch/utilities/types.hpp"

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

/** The @c expect_eq twin for floats, at the 1e-3 relative tolerance a @c -ffast-math
 *  build needs: it re-derives finalizes per inlining context, so bit-equality is wrong. */
#define expect_near(actual, wanted) expect_near_((actual), (wanted), #actual " ~ " #wanted, __FILE__, __LINE__)
inline std::size_t expect_near_(f64_t actual, f64_t wanted, char const* expression, char const* file, int line) {
    f64_t const scale = std::fmax(std::fmax(std::fabs(actual), std::fabs(wanted)), 1.0);
    f64_t const difference = std::fabs(actual - wanted);
    if (difference <= 1e-3 * scale) return 0;
    fmt::print(stderr, "  check failed at {}:{}: {} - expected {:.6g}, got {:.6g}, off by {:.4g}\n", file, line,
               expression, wanted, actual, difference);
    return 1;
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

#pragma region Settings

/** The text of the environment variable @p name, or nothing when it is unset or empty. */
inline std::optional<std::string_view> env_text(char const* name) noexcept {
    char const* const text = std::getenv(name);
    if (!text || !*text) return std::nullopt;
    return std::string_view(text);
}

/** Parses the environment variable @p name with @p parse, or returns @p fallback when it is unset
 *  or empty. Text that does not parse prints `NAME="text" does not parse, expected <expected>` and
 *  exits with status 1, which leaves crash handlers quiet. */
template <typename value_type_, typename parse_type_>
[[nodiscard]] value_type_ env_parsed(char const* name, value_type_ fallback, parse_type_&& parse,
                                     char const* expected) noexcept {
    std::optional<std::string_view> const text = env_text(name);
    if (!text) return fallback;
    if (std::optional<value_type_> value = parse(*text)) return *std::move(value);
    fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", name, *text, expected);
    std::exit(1);
}

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<std::uint32_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return static_cast<std::uint32_t>(std::random_device {}());
    std::uint32_t seed = 0;
    auto const [end, parse_status] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (parse_status != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed;
}

inline std::uint32_t env_seed(char const* name, std::uint32_t fallback) noexcept {
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random");
}

/** Comma-separated unsigned device ordinals, like "0" or "0,2". */
inline std::optional<std::vector<std::uint32_t>> parse_ordinals(std::string_view text) {
    std::vector<std::uint32_t> ordinals;
    for (std::size_t begin = 0; begin <= text.size();) {
        std::size_t const end = min_of(text.find(',', begin), text.size());
        std::uint32_t ordinal = 0;
        auto const [parsed, parse_status] = std::from_chars(text.data() + begin, text.data() + end, ordinal);
        if (parse_status != std::errc {} || parsed != text.data() + end) return std::nullopt;
        ordinals.push_back(ordinal);
        begin = end + 1;
    }
    return ordinals;
}

/** Comma-separated paths in the grammar of @c usearch_nvme_options_t::paths: files, block devices
 *  or SPDK transport IDs like "trtype:PCIe traddr:0000:01:00.0". */
inline std::optional<std::vector<std::string_view>> parse_paths(std::string_view text) {
    std::vector<std::string_view> paths;
    for (std::size_t begin = 0; begin <= text.size();) {
        std::size_t const end = min_of(text.find(',', begin), text.size());
        if (end == begin) return std::nullopt;
        paths.push_back(text.substr(begin, end - begin));
        begin = end + 1;
    }
    return paths;
}

/** Every setting the suites read, its default as the initializer, filled once by @c read_settings. */
struct settings_t {

    /** Checks to run, by ECMAScript regex or else substring. Override: @c USEARCH_FILTER. */
    std::string_view filter;
    std::optional<std::regex> filter_regex;

    /** Seed every scenario family derives its streams from. Override: @c USEARCH_SEED, where random draws one. */
    std::uint32_t seed = 42;

    /** USearch device ordinals the GPU suites run on, positions in @c gpu_devices; none runs the
     *  first device of every vendor present. Override: @c USEARCH_DEVICES, comma-separated. */
    std::vector<std::uint32_t> devices;

    /** NVMe namespaces the SPDK suites view, as @c usearch_nvme_options_t::paths names them; none
     *  saves the part they would view instead. Override: @c USEARCH_NVME_PATHS, comma-separated. */
    std::vector<std::string_view> nvme_paths;

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const {
        if (filter.empty()) return true;
        if (filter_regex) return std::regex_search(name.begin(), name.end(), *filter_regex);
        return name.find(filter) != std::string_view::npos;
    }
};

/** Reads every @c settings_t variable, exiting with status 1 on the first that does not parse. */
inline settings_t read_settings() noexcept {
    settings_t settings;
    settings.filter = env_text("USEARCH_FILTER").value_or("");
    if (!settings.filter.empty()) {
#if defined(__cpp_exceptions) && __cpp_exceptions
        try {
            settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
        }
        catch (std::regex_error const&) {
        }
#else
        settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
#endif
    }
    settings.seed = env_seed("USEARCH_SEED", settings.seed);
    settings.devices = env_parsed("USEARCH_DEVICES", settings.devices, parse_ordinals,
                                  "comma-separated unsigned device ordinals");
    settings.nvme_paths = env_parsed("USEARCH_NVME_PATHS", settings.nvme_paths, parse_paths,
                                     "comma-separated files, block devices or transport IDs");
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const& settings) {
    fmt::println("- Seed: {}", settings.seed);
    fmt::println("- Filter: {}", settings.filter.empty() ? std::string_view("none") : settings.filter);
    if (settings.devices.empty()) fmt::println("- Devices: first of each vendor");
    else fmt::println("- Devices: {}", fmt::join(settings.devices, ","));
    if (settings.nvme_paths.empty()) fmt::println("- NVMe paths: none");
    else fmt::println("- NVMe paths: {}", fmt::join(settings.nvme_paths, ","));
}

/** One vendor's VRAM suites, as its test object registers them with the runner. */
struct vram_suite_t {
    gpu_vendor_t vendor = gpu_vendor_t::unknown_k;
    std::size_t (*check)(settings_t const& settings, gpu_device_t const& device) = nullptr;

    /** Copies @p bytes of the vendor's device memory at @p device to @p host and waits, true once
     *  they landed. */
    bool (*download)(void* host, void const* device, std::size_t bytes) = nullptr;
};

/** Registers @p added when it names a check, and answers every suite registered so far. */
inline std::span<vram_suite_t const> vram_suites(vram_suite_t added = {}) noexcept {
    static vram_suite_t suites[4];
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
inline void print(machine_t const& machine) {
    fmt::println("USearch {}", machine.version);
    fmt::println("- Compiled for: {}", machine.compiled);
    fmt::println("- This machine: {}", machine.available);
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

/** IDs dispensed to racing lanes must land dense & duplicate-free: sorted in
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

/** The reader-lane invariants over one search's scored results: every ID below
 *  the watermark sampled after the search, distances ascending, the pool bound
 *  honored. Answers the violation count silently - racing lanes add it to their
 *  shared counter, since printing from inside a race would interleave & flood - and
 *  the status verdict stays at the call site: lanes differ in how they record it. */
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
    std::atomic<std::ptrdiff_t> live_bytes {0};

    /** Lifetime number of @c allocate calls, for coarse allocation-pressure checks. */
    std::atomic<std::size_t> total_allocations {0};

    /** Allocation ceiling in live bytes; scenarios lower it to drive the all-or-
     *  nothing refusal paths, so a request past it is refused, never counted. */
    std::atomic<std::ptrdiff_t> byte_budget {std::numeric_limits<std::ptrdiff_t>::max()};
};

/**
 *  @brief Byte-counting allocator proving propagation through a container: every allocation and
 *      deallocation lands in the shared ledger, which must balance to zero at scope exit.
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
        if (ledger_->live_bytes.load() + wanted > ledger_->byte_budget.load()) return nullptr;
        ledger_->live_bytes.fetch_add(wanted);
        ledger_->total_allocations.fetch_add(1);
        return static_cast<value_type_*>(std::malloc(count * sizeof(value_type_)));
    }
    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (ledger_) ledger_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(count * sizeof(value_type_)));
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
 *  @see @c nk::fill_uniform, which ranges every dtype by its own @c finite_min & @c finite_max.
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
    void operator()(std::atomic<value_type_> const&, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {
        misbehave_();
    }
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
    // different seeds, like the concurrent builds', still land in one sketch space.
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

std::size_t check_dense(settings_t const& settings);

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

} // namespace unum::usearch::test

/** Teaches the expect verbs to print statuses by name - @c status_t is the most
 *  compared enum in the suite, and {fmt} refuses unknown enums at compile time. */
template <>
struct fmt::formatter<unum::usearch::utilities::status_t> : fmt::formatter<char const*> {
    auto format(us::utilities::status_t status, format_context& context) const {
        return fmt::formatter<char const*>::format(us::utilities::status_to_string(status), context);
    }
};
