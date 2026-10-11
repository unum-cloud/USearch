/**
 *  @file bench/dense/ram_metric.cpp
 *  @author Ash Vardanian
 *  @date April 23, 2026
 *  @brief Throughput benchmarks for @c dense/ram_metric.hpp distance functions for 1-to-1, 1-to-4,
 *      and tiled kernels against NumKong's monolithic 1-to-1 reference.
 *
 *  Mirrors the structure of @c test/dense/ram_metric.hpp - the @c dense_quads_supported concept
 *  gates each cell so unadapted (kernel, scalar, caps) tuples never run, and new kits or
 *  chains light up matching rows automatically. Three flavors per supported tuple: USearch's
 *  @c distance (1-to-1), USearch's @c distances (1-to-4 fan-out), and NumKong's @c nk::dot /
 *  @c nk::angular / @c nk::euclidean / @c nk::hamming / @c nk::jaccard monolithic baseline
 *  using the @c nk::default_capabilities() mask (run once per kernel-scalar, since the
 *  baseline isn't caps-parameterized). Where a tile engine is adapted and the machine offers
 *  it, a fourth row measures one 16×16 tile of distances per call - gather, pack, and the
 *  depth walk - so the tile engine's primitive throughput reads next to 256 quads-path distances.
 *
 *  Rows are named `kernel/scalar/caps/shape/dimensions`, like `dot/f32/neon/1to4/768d`, and
 *  every flag has a default:
 *
 *  @verbatim
 *  --seed        42    Seed the row picks and the adjacency noise derive from, an unsigned integer or random
 *  --filter      none  Rows to run, as an ECMAScript regex or else a substring
 *  --warmup      1s    Untimed run ahead of each row
 *  --time-limit  10s   Timed run of each row
 *  @endverbatim
 */
#include <cstdint> // `std::uint32_t`
#include <cstring> // `std::strcmp`, `std::memcpy` - the dot lanes a finalizer takes

#include <algorithm>   // `std::max`
#include <array>       // `std::array`
#include <chrono>      // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <optional>    // `std::optional`
#include <regex>       // `std::regex`, `std::regex_search`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include <numkong/dot.hpp>
#include <numkong/set.hpp>
#include <numkong/spatial.hpp>
#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/types.h" // `USEARCH_VERSION_MAJOR`
#include "usearch/dense/ram_metric.hpp"

#include "arguments.hpp" // `flag_t`, `parse_or_exit`
#include "bench.hpp"

namespace unum::usearch::bench {

#pragma region Harness

/** Keeps @p value, and every write before it, from being optimized away. */
template <typename value_type_>
inline void do_not_optimize(value_type_&& value) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    [[maybe_unused]] auto volatile* pointer = &value;
    _ReadWriteBarrier();
#elif defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/** A named amount per call: printed per second when @c is_rate, or as it is. */
struct counter_t {
    char const* name = nullptr;
    double value = 0;
    bool is_rate = false;
};

/** A finished benchmark: its name, calls per second and counters, or why it was skipped. */
struct row_t {
    std::string_view name;
    double calls_per_second = 0;
    std::array<counter_t, 4> counters {};
    char const* skipped = nullptr;
};

/** Prints @p row on one line: byte rates in binary units like "GB/s", others like "M/s". */
inline void print(row_t const& row) {
    if (row.skipped) return fmt::println("{:<56} skipped: {}", row.name, row.skipped);
    auto const spell_rate = [](char const* label, double per_second) {
        bool const bytes = std::strcmp(label, "bytes") == 0;
        double const step = bytes ? 1024 : 1000;
        char const* prefix = "";
        for (char const* next : bytes ? std::array {"K", "M", "G", "T", "P"} : std::array {"k", "M", "G", "T", "P"})
            if (per_second >= step) per_second /= step, prefix = next;
        return fmt::format("  {} {:.3f} {}{}/s", label, per_second, prefix, bytes ? "B" : "");
    };
    std::string line = fmt::format("{:<56}{}", row.name, spell_rate("calls", row.calls_per_second));
    for (counter_t const& counter : row.counters)
        if (counter.is_rate) line += spell_rate(counter.name, counter.value * row.calls_per_second);
        else if (counter.name) line += fmt::format("  {} {:.3f}", counter.name, counter.value);
    fmt::println("{}", line);
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed. The loop runs untimed for the warm-up, then counts calls
 *  until the time limit. It reads the clock once per 64th of the calls so far, so a short kernel
 *  doesn't time the clock itself.
 */
class loop_t {
    std::chrono::milliseconds warmup_, time_limit_;
    time_point_t start_ {};
    steady_clock_t::duration elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0;
    bool warming_up_ = true;
    char const* skipped_ = nullptr;
    std::array<counter_t, 4> counters_ {};

    bool keep_running() noexcept {
        if (skipped_) return false;
        if (calls_ < next_check_) return true;
        elapsed_ = steady_clock_t::now() - start_;
        if (warming_up_ && elapsed_ >= warmup_)
            warming_up_ = false, start_ = steady_clock_t::now(), elapsed_ = {}, calls_ = 0;
        else if (!warming_up_ && elapsed_ >= time_limit_) return false;
        next_check_ = calls_ + calls_ / 64 + 1;
        return true;
    }

    void add(counter_t counter) noexcept {
        for (counter_t& slot : counters_)
            if (!slot.name) return void(slot = counter);
    }

  public:
    struct end_t {};
    struct iterator_t {
        loop_t* loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Stops the loop and reports @p reason instead of results. */
    void skip(char const* reason) noexcept { skipped_ = reason; }

    /** Reports @p per_call of @p name per second. */
    void rate(char const* name, double per_call) noexcept { add({name, per_call, true}); }

    /** The finished benchmark under @p name. */
    row_t row(std::string_view name) const noexcept {
        if (skipped_) return {name, 0, {}, skipped_};
        double const seconds = std::chrono::duration<double>(elapsed_).count();
        return {name, calls_ / seconds, counters_, nullptr};
    }
};

/** Every flag this binary reads, its default as the initializer, filled once by @c read_settings. */
struct settings_t {

    /** Rows to run, by ECMAScript regex or else substring. Override: @c --filter. */
    std::string_view filter;
    std::optional<std::regex> filter_regex;

    /** Seed the row picks and the adjacency noise derive from. Override: @c --seed, where random draws one. */
    std::uint32_t seed = 42;

    /** Untimed run ahead of each row. Override: @c --warmup. */
    std::chrono::milliseconds warmup = std::chrono::seconds(1);

    /** Timed run of each row. Override: @c --time-limit. */
    std::chrono::milliseconds time_limit = std::chrono::seconds(10);

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const {
        if (filter.empty()) return true;
        if (filter_regex) return std::regex_search(name.begin(), name.end(), *filter_regex);
        return name.find(filter) != std::string_view::npos;
    }
};

/** Parses the flags into one @ref settings_t. On `--help` prints the usage and exits with status 0; on a bad flag
 *  prints one line and exits with status 1. */
inline settings_t read_settings(int argc, char const* const* argv) {
    settings_t settings;
    std::array<flag_t, 4> const flags {{
        {"--seed", "seed", "seed of the row picks and the adjacency noise, or `random`; 42 by default", &settings.seed},
        {"--filter", "pattern", "rows to run, as a regex or else a substring, like `dot/f32/serial`", &settings.filter},
        {"--warmup", "duration", "untimed run ahead of each row, like `200ms`; 1s by default", &settings.warmup},
        {"--time-limit", "duration", "timed run of each row, like `200ms`; 10s by default", &settings.time_limit},
    }};
    parse_or_exit(argc, argv, "usearch_bench_dense_ram_metric [flags]", flags);
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
    return settings;
}

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
}

/** Prints each setting as "- Name: value", in the grammar its flag parses from. */
inline void print(settings_t const& settings) {
    fmt::println("- Seed: {}", settings.seed);
    fmt::println("- Filter: {}", settings.filter.empty() ? std::string_view("none") : settings.filter);
    fmt::println("- Warm-up: {}", spell_duration(settings.warmup));
    fmt::println("- Time limit: {}", spell_duration(settings.time_limit));
}

/** The capabilities this binary holds and this machine runs. */
struct machine_t {
    nk_capability_t compiled = nk_cap_serial_k;
    nk_capability_t detected = nk_cap_serial_k;
};

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    if (nk_cpu_capabilities_compiled(&machine.compiled) != nk_success_k) machine.compiled = nk_cap_serial_k;
    if (nk_cpu_capabilities_detected(&machine.detected) != nk_success_k) machine.detected = nk_cap_serial_k;
    return machine;
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const& machine) {
    char compiled[NUMKONG_CAPABILITIES_NAME_CAPACITY], detected[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(machine.compiled, compiled, sizeof(compiled));
    nk_capabilities_name(machine.detected, detected, sizeof(detected));
    fmt::println("USearch {}.{}.{}", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR, USEARCH_VERSION_PATCH);
    fmt::println("- Compiled for: {}", compiled);
    fmt::println("- This machine: {}", detected);
}

/** Everything a benchmark reads, built once in @c main and passed down by reference. */
struct environment_t {
    settings_t settings;
    machine_t machine;
};

/** Runs @p measure over @p arguments when @p name passes the filter, then prints its row. */
template <typename measure_type_, typename... arguments_types_>
void run_benchmark(environment_t const& env, std::string const& name, measure_type_ measure,
                   arguments_types_... arguments) {
    if (!env.settings.selects(name)) return;
    loop_t loop(env.settings.warmup, env.settings.time_limit);
    measure(loop, env, arguments...);
    print(loop.row(name));
}

/** The key of one stream family under the run's @p seed: two mixes away from both, so neither adjacent seeds nor
 *  adjacent families yield related keys. */
constexpr u64_t stream_key(u64_t seed, u64_t family) noexcept { return mix_bits(mix_bits(seed ^ family)); }

/** Stream families: the candidate rows each call picks, and the adjacency noise pool. */
constexpr u64_t rows_family_k = 0xDEADBEEFull;
constexpr u64_t adjacency_family_k = 0xABCDEF01ull;

#pragma endregion Harness

constexpr std::array<std::size_t, 3> dimensions_to_bench = {256, 768, 1536};

/** True iff USearch has any kit for this (kernel, scalar) pair across capability
 *  tiers - serial is the always-available floor, so checking it covers the matrix. */
template <nk_kernel_kind_t kernel_, typename scalar_>
concept quads_in_scope = dense_quads_supported<kernel_, scalar_, nk_cap_serial_k>;

/** True iff a tile engine is compiled for this tuple - the tiles chain always names
 *  a kit, so the gate is its @c has_tiles_k verdict, never the alias's existence. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept tiles_supported = dense_quads_supported<kernel_, scalar_, caps_> &&
                          dense_tiles_chain<scalar_, caps_>::has_tiles_k;

/** Whether this machine runs every bit of @p caps_; serial always runs. */
template <nk_capability_t caps_>
bool runs_on(machine_t const& machine) noexcept {
    return caps_ == nk_cap_serial_k || (machine.detected & caps_) == caps_;
}

/** Build a scalar from a small unsigned - sub-byte goes through @c sub_byte_ref as
 *  raw @c int8_t, float wrappers via @c from_f32, byte-or-larger ints via the integral ctor. */
template <typename scalar_>
static auto make_scalar(unsigned value) noexcept {
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<i8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** Populates @p vec with cyclic small representable values at @p rotation. */
template <typename scalar_>
static void fill_cyclic(nk::vector_span<scalar_> vec, std::size_t rotation) noexcept {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    for (std::size_t dimension_index = 0; dimension_index < vec.size(); ++dimension_index)
        vec[dimension_index] = make_scalar<scalar_>(
            universally_representable_values[(dimension_index + rotation) % universally_representable_values.size()]);
}

/** Round up to the next power of two so we can use a bitmask for cyclic indexing
 *  instead of a slow modulo on every loop iteration. */
static std::size_t round_up_pow2(std::size_t value) noexcept {
    std::size_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

/** Pool footprint target - well above any current x86 L3 (16-128 MB), so candidate
 *  rows stream from RAM rather than cache and the query-load amortization actually matters. */
constexpr std::size_t candidate_pool_target_bytes_k = 1ull << 28; // 256 MB

/** Synthetic L1 working-set noise modelling an HNSW worker's per-iteration touches
 *  (neighbor list of @c u32_t IDs, visited bitset, top-K queue). Without this, the
 *  kernel-only microbench has no L1 contestants - the query stays resident and the cache
 *  hints inside the kits (PREFETCHT0 / PREFETCHT2 / CLDEMOTE) have nothing to optimize for.
 *  A pool larger than L1 (32 KB on most x86) plus a random per-iter pick defeats the HW
 *  prefetcher and forces capacity pressure on the query lines. */
struct adjacency_simulator_t {
    using node_id_t = u32_t;
    static constexpr std::size_t bytes_per_node_k = 256; // 4 cache lines / iter
    static constexpr std::size_t ids_per_node_k = bytes_per_node_k / sizeof(node_id_t);
    static constexpr std::size_t node_count_k = 256; // 64 KB pool, > 32 KB L1
    static constexpr std::size_t node_index_mask_k = node_count_k - 1;
    std::vector<node_id_t> pool;

    explicit adjacency_simulator_t(u64_t seed) : pool(node_count_k * ids_per_node_k, 0) {
        splitmix64_t seed_generator {seed};
        for (auto& id : pool) id = static_cast<node_id_t>(seed_generator());
    }

    /** Streaming xor over one "node"'s worth of IDs; one barrier per call. */
    inline void touch(std::size_t random_index) noexcept {
        node_id_t const* node = pool.data() + (random_index & node_index_mask_k) * ids_per_node_k;
        node_id_t accumulator = 0;
        for (std::size_t id_index = 0; id_index < ids_per_node_k; ++id_index) accumulator ^= node[id_index];
        do_not_optimize(accumulator);
    }
};

/** Owning row-major candidate matrix plus per-row precomputed norms. Two fields only. */
template <typename scalar_>
struct candidate_pool {
    nk::tensor<scalar_, nk::aligned_allocator<scalar_, 64>, 2> matrix;
    std::vector<vector_norms<typename scalar_::dot_result_t>> norms;
};

/** Allocates a candidate matrix sized to overflow L3 and precomputes per-row norms.
 *  Row count is rounded up to a power of two so the bench loop can mask instead of modulo.
 *  Each row is populated with a different cyclic rotation so the branch predictor can't
 *  shortcut on identical content; correctness isn't tested here. */
template <typename scalar_>
static candidate_pool<scalar_> make_candidate_pool(std::size_t dim) {
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_>();
    std::size_t const row_bytes = (dim / dims_per_value) * sizeof(scalar_);
    std::size_t const target_rows = std::max<std::size_t>(8, candidate_pool_target_bytes_k / row_bytes);
    std::size_t const row_count = round_up_pow2(target_rows);

    candidate_pool<scalar_> pool {
        nk::tensor<scalar_, nk::aligned_allocator<scalar_, 64>, 2>::zeros({row_count, dim}).value,
        {}};
    for (std::size_t row = 0; row < row_count; ++row) fill_cyclic(pool.matrix.span().row(row).as_vector(), row);
    pool.norms.reserve(row_count);
    for (std::size_t row = 0; row < row_count; ++row)
        pool.norms.push_back(dense_norms<nk_cap_serial_k>(pool.matrix.view().row(row).as_vector()));
    return pool;
}

/** Reports distances, candidate bytes - the query is reused - and scalar operations per call, one distance
 *  approximated as @c 2*dim multiply-add or bit-op pairs across all five kernels. */
static void rate_distances(loop_t& loop, std::size_t dim, std::size_t distances_per_call,
                           std::size_t bytes_per_call) noexcept {
    loop.rate("distances", static_cast<double>(distances_per_call));
    loop.rate("bytes", static_cast<double>(bytes_per_call));
    loop.rate("scalar-ops", 2.0 * static_cast<double>(dim) * static_cast<double>(distances_per_call));
}

/** Computes a row's contiguous byte width once per row. */
template <typename scalar_>
static constexpr std::size_t row_bytes_of(std::size_t dim) noexcept {
    return (dim / nk::dimensions_per_value<scalar_>()) * sizeof(scalar_);
}

/** USearch 1-to-1 row - random row pick per call so candidates stream from RAM. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void measure_distance(loop_t& loop, environment_t const& env, std::size_t dim) {
    if (!runs_on<caps_>(env.machine)) return loop.skip("capabilities missing at runtime");
    auto query = nk::vector<scalar_>::zeros(dim).value;
    fill_cyclic(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();
    auto query_norms = dense_norms<caps_>(query_view);

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    splitmix64_t random_generator {stream_key(env.settings.seed, rows_family_k)};
    adjacency_simulator_t adjacency {stream_key(env.settings.seed, adjacency_family_k)};
    for ([[maybe_unused]] std::size_t call : loop) {
        std::size_t const row = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        auto candidate_view = pool.matrix.view().row(row).as_vector();
        auto distance_value = distance<caps_>(kernel_, query_view, query_norms, candidate_view, pool.norms[row]);
        do_not_optimize(distance_value);
    }
    rate_distances(loop, dim, 1, row_bytes_of<scalar_>(dim));
}

/** USearch 1-to-4 row - four random rows per call; query is amortized 4x. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void measure_distances_quad(loop_t& loop, environment_t const& env, std::size_t dim) {
    if (!runs_on<caps_>(env.machine)) return loop.skip("capabilities missing at runtime");
    auto query = nk::vector<scalar_>::zeros(dim).value;
    fill_cyclic(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();
    auto query_norms = dense_norms<caps_>(query_view);

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    splitmix64_t random_generator {stream_key(env.settings.seed, rows_family_k)};
    adjacency_simulator_t adjacency {stream_key(env.settings.seed, adjacency_family_k)};
    for ([[maybe_unused]] std::size_t call : loop) {
        std::size_t const row0 = random_generator() & row_index_mask;
        std::size_t const row1 = random_generator() & row_index_mask;
        std::size_t const row2 = random_generator() & row_index_mask;
        std::size_t const row3 = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        std::array<nk::vector_view<scalar_>, 4> const candidate_views {
            pool.matrix.view().row(row0).as_vector(), pool.matrix.view().row(row1).as_vector(),
            pool.matrix.view().row(row2).as_vector(), pool.matrix.view().row(row3).as_vector()};
        std::array<vector_norms<typename scalar_::dot_result_t>, 4> const candidate_norms {
            pool.norms[row0], pool.norms[row1], pool.norms[row2], pool.norms[row3]};
        auto distance_values = distances<caps_>(kernel_, query_view, query_norms, candidate_views, candidate_norms);
        do_not_optimize(distance_values);
    }
    rate_distances(loop, dim, 4, 4 * row_bytes_of<scalar_>(dim));
}

/** NumKong monolithic 1-to-1 baseline using the @c nk::default_capabilities() - one
 *  candidate per call, randomly picked from the same overflowed pool as the USearch rows. */
template <nk_kernel_kind_t kernel_, typename scalar_>
static void measure_baseline(loop_t& loop, environment_t const& env, std::size_t dim) {
    auto query = nk::vector<scalar_>::zeros(dim).value;
    fill_cyclic(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    splitmix64_t random_generator {stream_key(env.settings.seed, rows_family_k)};
    adjacency_simulator_t adjacency {stream_key(env.settings.seed, adjacency_family_k)};
    for ([[maybe_unused]] std::size_t call : loop) {
        std::size_t const row = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        auto candidate_view = pool.matrix.view().row(row).as_vector();
        if constexpr (kernel_ == nk_kernel_dot_k) {
            auto [result, status] = nk::dot<scalar_>(query_view, candidate_view);
            if (nk::failed(status)) return loop.skip("dot failed");
            do_not_optimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_angular_k) {
            auto [result, status] = nk::angular<scalar_>(query_view, candidate_view);
            if (nk::failed(status)) return loop.skip("angular failed");
            do_not_optimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_euclidean_k) {
            auto [result, status] = nk::euclidean<scalar_>(query_view, candidate_view);
            if (nk::failed(status)) return loop.skip("euclidean failed");
            do_not_optimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_hamming_k) {
            auto [result, status] = nk::hamming<scalar_>(query_view, candidate_view);
            if (nk::failed(status)) return loop.skip("hamming failed");
            do_not_optimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_jaccard_k) {
            auto [result, status] = nk::jaccard<scalar_>(query_view, candidate_view);
            if (nk::failed(status)) return loop.skip("jaccard failed");
            do_not_optimize(result);
        }
    }
    rate_distances(loop, dim, 1, row_bytes_of<scalar_>(dim));
}

/** One tile of queries, packed once, against 256 stored rows per call - the stream through the matrix engine, then
 *  the quads kit's finalizer - comparable against as many quads-path distances. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires tiles_supported<kernel_, scalar_, caps_>
static void measure_distances_tile(loop_t& loop, environment_t const& env, std::size_t dim) {
    if (!runs_on<caps_>(env.machine)) return loop.skip("capabilities missing at runtime");
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k || nk_cpu_configure_thread(available) != nk_success_k)
        return loop.skip("the thread cannot run these capabilities");
    using tiles_kit_t = typename dense_tiles_chain<scalar_, caps_>::kit_t;
    using quads_kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    using raw_dot_t = typename scalar_::dot_result_t::raw_t;
    constexpr std::size_t queries_k = tiles_kit_t::tile_columns_k, rows_k = 256;

    using lanes_t = std::array<typename quads_kit_t::finalized_vec_t, queries_k / 4>;
    auto storage = nk::matrix<scalar_>::zeros({queries_k + rows_k, dim}).value;
    for (std::size_t row = 0; row != queries_k + rows_k; ++row) fill_cyclic(storage.row(row).as_vector(), row);
    nk::matrix_view<scalar_> const queries = storage(nk::range(std::size_t {0}, queries_k), nk::slice);
    nk::matrix_view<scalar_> const rows = storage(nk::range(queries_k, queries_k + rows_k), nk::slice);
    std::size_t const stride = static_cast<std::size_t>(queries.stride_bytes(0));
    nk_size_t packed_bytes = 0;
    if (tiles_kit_t::pack_size(queries_k, dim, &packed_bytes) != nk_success_k) return loop.skip("pack size failed");
    auto packed = nk::vector<nk::u8_t>::zeros(packed_bytes).value;
    if (tiles_kit_t::pack(queries.raw_values_data(), queries_k, dim, stride, packed.data(), 0, queries_k, nullptr) !=
        nk_success_k)
        return loop.skip("pack failed");

    // The finalizer takes queries four lanes at a time, so the per-query sums regroup into lanes whole.
    std::array<typename scalar_::dot_result_t, queries_k> query_sumsqs;
    for (std::size_t query = 0; query != queries_k; ++query)
        query_sumsqs[query] = dense_norms<caps_>(queries.row(query).as_vector()).sumsq;
    lanes_t query_lanes;
    static_assert(sizeof(query_lanes) == sizeof(query_sumsqs), "four sums fill one finalizer lane group");
    std::memcpy(query_lanes.data(), query_sumsqs.data(), sizeof(query_lanes));
    std::array<typename scalar_::dot_result_t, rows_k> row_sumsqs;
    for (std::size_t row = 0; row != rows_k; ++row)
        row_sumsqs[row] = dense_norms<caps_>(rows.row(row).as_vector()).sumsq;
    auto dots = nk::matrix<raw_dot_t>::zeros({rows_k, queries_k}).value;
    std::array<std::array<std::array<f32_t, 4>, queries_k / 4>, rows_k> distances;

    for ([[maybe_unused]] std::size_t call : loop) {
        if (tiles_kit_t::dots_packed(rows.raw_values_data(), packed.data(), dots.data(), rows_k, queries_k, dim, stride,
                                     queries_k * sizeof(raw_dot_t), nullptr) != nk_success_k)
            return loop.skip("streaming failed");
        for (std::size_t row = 0; row != rows_k; ++row) {
            lanes_t row_lanes;
            std::memcpy(row_lanes.data(), dots.row(row).data(), sizeof(row_lanes));
            for (std::size_t group = 0; group != queries_k / 4; ++group)
                distances[row][group] = convert_to_distances<quads_kit_t>(kernel_, row_lanes[group], row_sumsqs[row],
                                                                          query_lanes[group]);
        }
        do_not_optimize(distances);
    }
    rate_distances(loop, dim, rows_k * queries_k, rows_k * row_bytes_of<scalar_>(dim));
}

/** Runs the 1-to-1 and 1-to-4 USearch rows for one tuple; unadapted tuples have none. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void bench_quads(environment_t const& env, std::string_view kernel_name, std::string_view scalar_name,
                        std::string_view caps_name) {
    if constexpr (dense_quads_supported<kernel_, scalar_, caps_>)
        for (std::size_t const dim : dimensions_to_bench) {
            run_benchmark(env, fmt::format("{}/{}/{}/1to1/{}d", kernel_name, scalar_name, caps_name, dim),
                          measure_distance<kernel_, scalar_, caps_>, dim);
            run_benchmark(env, fmt::format("{}/{}/{}/1to4/{}d", kernel_name, scalar_name, caps_name, dim),
                          measure_distances_quad<kernel_, scalar_, caps_>, dim);
        }
}

/** Runs the matrix-engine row for one tuple; tuples without a tiles kit have none. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void bench_tiles(environment_t const& env, std::string_view kernel_name, std::string_view scalar_name,
                        std::string_view caps_name) {
    if constexpr (tiles_supported<kernel_, scalar_, caps_>)
        for (std::size_t const dim : dimensions_to_bench)
            run_benchmark(env, fmt::format("{}/{}/{}/tiles/{}d", kernel_name, scalar_name, caps_name, dim),
                          measure_distances_tile<kernel_, scalar_, caps_>, dim);
}

template <nk_kernel_kind_t kernel_, typename scalar_>
static void bench_kernel(environment_t const& env, std::string_view kernel_name, std::string_view scalar_name) {
    // The NumKong baseline is caps-agnostic: its `nk::default_capabilities()` mask picks the best backend it holds.
    if constexpr (quads_in_scope<kernel_, scalar_>)
        for (std::size_t const dim : dimensions_to_bench)
            run_benchmark(env, fmt::format("{}/{}/numkong/1to1/{}d", kernel_name, scalar_name, dim),
                          measure_baseline<kernel_, scalar_>, dim);
    // Newest silicon first, split by the architecture-family macros like the test's tier
    // sweep - the concept gates silently absorb any tier without a kit.
#if NUMKONG_ARCH_X8664_
    bench_quads<kernel_, scalar_, nk_cap_diamond_k>(env, kernel_name, scalar_name, "diamond");
    bench_tiles<kernel_, scalar_, nk_cap_graniteamx_k>(env, kernel_name, scalar_name, "graniteamx");
    bench_quads<kernel_, scalar_, nk_cap_sierra_k>(env, kernel_name, scalar_name, "sierra");
    bench_tiles<kernel_, scalar_, nk_cap_sapphireamx_k>(env, kernel_name, scalar_name, "sapphireamx");
    bench_quads<kernel_, scalar_, nk_cap_genoa_k>(env, kernel_name, scalar_name, "genoa");
    bench_quads<kernel_, scalar_, nk_cap_alder_k>(env, kernel_name, scalar_name, "alder");
    bench_quads<kernel_, scalar_, nk_cap_icelake_k>(env, kernel_name, scalar_name, "icelake");
    bench_quads<kernel_, scalar_, nk_cap_skylake_k>(env, kernel_name, scalar_name, "skylake");
    bench_quads<kernel_, scalar_, nk_cap_haswell_k>(env, kernel_name, scalar_name, "haswell");
#elif NUMKONG_ARCH_ARM64_
    bench_quads<kernel_, scalar_, nk_cap_neonfp8_k>(env, kernel_name, scalar_name, "neonfp8");
    bench_tiles<kernel_, scalar_, nk_cap_smebi32_k>(env, kernel_name, scalar_name, "smebi32");
    bench_tiles<kernel_, scalar_, nk_cap_sme_k>(env, kernel_name, scalar_name, "sme");
    bench_quads<kernel_, scalar_, nk_cap_neonbfdot_k>(env, kernel_name, scalar_name, "neonbfdot");
    bench_quads<kernel_, scalar_, nk_cap_neonfhm_k>(env, kernel_name, scalar_name, "neonfhm");
    bench_quads<kernel_, scalar_, nk_cap_neonsdot_k>(env, kernel_name, scalar_name, "neonsdot");
    bench_quads<kernel_, scalar_, nk_cap_neon_k>(env, kernel_name, scalar_name, "neon");
#elif NUMKONG_ARCH_WASM_
    bench_quads<kernel_, scalar_, nk_cap_v128relaxed_k>(env, kernel_name, scalar_name, "v128relaxed");
    bench_quads<kernel_, scalar_, nk_cap_v128_k>(env, kernel_name, scalar_name, "v128");
#elif NUMKONG_ARCH_PPC64_
    bench_quads<kernel_, scalar_, nk_cap_powervsx_k>(env, kernel_name, scalar_name, "powervsx");
#endif
    bench_quads<kernel_, scalar_, nk_cap_serial_k>(env, kernel_name, scalar_name, "serial");
}

template <typename scalar_>
static void bench_scalar(environment_t const& env, std::string_view scalar_name) {
    // Binary inputs run dot/hamming/jaccard instead of dot/angular/euclidean.
    bench_kernel<nk_kernel_dot_k, scalar_>(env, "dot", scalar_name);
    if constexpr (std::is_same_v<scalar_, nk::u1x8_t>) {
        bench_kernel<nk_kernel_hamming_k, scalar_>(env, "hamming", scalar_name);
        bench_kernel<nk_kernel_jaccard_k, scalar_>(env, "jaccard", scalar_name);
    }
    else {
        bench_kernel<nk_kernel_angular_k, scalar_>(env, "angular", scalar_name);
        bench_kernel<nk_kernel_euclidean_k, scalar_>(env, "euclidean", scalar_name);
    }
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) {
    using namespace unum::usearch::bench;
    environment_t const env {read_settings(argc, argv), probe_machine()};
    print(env.machine);
    print(env.settings);

    bench_scalar<nk::f64_t>(env, "f64");
    bench_scalar<nk::f32_t>(env, "f32");
    bench_scalar<nk::bf16_t>(env, "bf16");
    bench_scalar<nk::f16_t>(env, "f16");
    bench_scalar<nk::e5m2_t>(env, "e5m2");
    bench_scalar<nk::e4m3_t>(env, "e4m3");
    bench_scalar<nk::e3m2_t>(env, "e3m2");
    bench_scalar<nk::e2m3_t>(env, "e2m3");

    bench_scalar<nk::i8_t>(env, "i8");
    bench_scalar<nk::i4x2_t>(env, "i4x2");

    bench_scalar<nk::u8_t>(env, "u8");
    bench_scalar<nk::u4x2_t>(env, "u4x2");
    bench_scalar<nk::u1x8_t>(env, "u1x8");
    return 0;
}
