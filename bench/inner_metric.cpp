/**
 *  @brief Throughput benchmarks for `inner_metric` distance functions for 1-to-1 and 1-to-4
 *      kernels against NumKong's monolithic 1-to-1 reference.
 *
 *  @author Ash Vardanian
 *  @file bench/inner_metric.cpp
 *  @date April 23, 2026
 *
 *  Mirrors the structure of test/inner_metric.cpp - the @c kernel_supported concept
 *  gates each cell so unadapted (kernel, scalar, caps) tuples don't get registered, and
 *  new kits or chains light up matching benchmark cells automatically. Three flavors per
 *  supported tuple: USearch's @c distance (1-to-1), USearch's @c distances (1-to-4 fan-out),
 *  and NumKong's @c nk::dot / @c nk::angular / @c nk::euclidean / @c nk::hamming /
 *  @c nk::jaccard monolithic baseline using the default @c prefer_simd_k compile-time
 *  dispatch (registered once per kernel-scalar, since the baseline isn't caps-parameterized).
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <numkong/dot.hpp>
#include <numkong/set.hpp>
#include <numkong/spatial.hpp>
#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/inner_metric.hpp"

namespace nk = ashvardanian::numkong;
using namespace unum::usearch;

constexpr std::array<std::size_t, 3> dimensions_to_bench = {256, 768, 1536};

/**
 *  @brief True iff @c kernel_chain has a specialization for this (kernel, scalar, caps)
 *  tuple - unadapted tuples are skipped at registration time.
 */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept kernel_supported = requires { typename kernel_chain<kernel_, scalar_, caps_>::kit_t; };

/**
 *  @brief True iff USearch has any kit for this (kernel, scalar) pair across capability
 *  tiers - serial is the always-available floor, so checking it covers the matrix.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
concept kernel_in_scope = kernel_supported<kernel_, scalar_, nk_cap_serial_k>;

/**
 *  @brief Build a scalar from a small unsigned - sub-byte goes through @c sub_byte_ref as
 *  raw @c int8_t, float wrappers via @c from_f32, byte-or-larger ints via the integral ctor.
 */
template <typename scalar_>
auto make_scalar(unsigned value) noexcept {
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<std::int8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** @brief Populates @p vec with cyclic small representable values at @p rotation. */
template <typename scalar_>
void populate(nk::vector_span<scalar_> vec, std::size_t rotation) noexcept {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    for (std::size_t dimension_index = 0; dimension_index < vec.size(); ++dimension_index)
        vec[dimension_index] = make_scalar<scalar_>(
            universally_representable_values[(dimension_index + rotation) % universally_representable_values.size()]);
}

/**
 *  @brief Round up to the next power of two so we can use a bitmask for cyclic indexing
 *  instead of a slow modulo on every loop iteration.
 */
inline std::size_t round_up_pow2(std::size_t value) noexcept {
    std::size_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

/** @brief Pool footprint target - well above any current x86 L3 (16-128 MiB), so candidate
 *  rows stream from RAM rather than cache and the query-load amortization actually matters. */
constexpr std::size_t candidate_pool_target_bytes_k = 1ull << 28; // 256 MiB

/**
 *  @brief Synthetic L1 working-set noise modelling an HNSW worker's per-iteration touches
 *  (neighbor list of @c std::uint32_t IDs, visited bitset, top-K queue). Without this, the
 *  kernel-only microbench has no L1 contestants - the query stays resident and the cache
 *  hints inside the kits (PREFETCHT0 / PREFETCHT2 / CLDEMOTE) have nothing to optimize for.
 *  A pool larger than L1 (32 KiB on most x86) plus a random per-iter pick defeats the HW
 *  prefetcher and forces capacity pressure on the query lines.
 */
struct adjacency_simulator_t {
    using node_id_t = std::uint32_t;
    static constexpr std::size_t bytes_per_node_k = 256; // 4 cache lines / iter
    static constexpr std::size_t ids_per_node_k = bytes_per_node_k / sizeof(node_id_t);
    static constexpr std::size_t node_count_k = 256; // 64 KiB pool, > 32 KiB L1
    static constexpr std::size_t node_index_mask_k = node_count_k - 1;
    std::vector<node_id_t> pool;

    adjacency_simulator_t() : pool(node_count_k * ids_per_node_k, 0) {
        std::mt19937 seed_rng {0xABCDEF01u};
        for (auto& id : pool) id = static_cast<node_id_t>(seed_rng());
    }

    /** @brief Streaming xor over one "node"'s worth of IDs; one DoNotOptimize per call. */
    inline void touch(std::size_t random_index) noexcept {
        node_id_t const* node = pool.data() + (random_index & node_index_mask_k) * ids_per_node_k;
        node_id_t accumulator = 0;
        for (std::size_t id_index = 0; id_index < ids_per_node_k; ++id_index) accumulator ^= node[id_index];
        benchmark::DoNotOptimize(accumulator);
    }
};

/** @brief Owning row-major candidate matrix plus per-row precomputed norms. Two fields only. */
template <typename scalar_>
struct candidate_pool {
    nk::tensor<scalar_, nk::aligned_allocator<scalar_, 64>, 2> matrix;
    std::vector<vector_norms<typename scalar_::dot_result_t>> norms;
};

/**
 *  @brief Allocates a candidate matrix sized to overflow L3 and precomputes per-row norms.
 *  Row count is rounded up to a power of two so the bench loop can mask instead of modulo.
 *  Each row is populated with a different cyclic rotation so the branch predictor can't
 *  shortcut on identical content; correctness isn't tested here.
 */
template <typename scalar_>
candidate_pool<scalar_> make_candidate_pool(std::size_t dim) {
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_>();
    std::size_t const row_bytes = (dim / dims_per_value) * sizeof(scalar_);
    std::size_t const target_rows = std::max<std::size_t>(8, candidate_pool_target_bytes_k / row_bytes);
    std::size_t const row_count = round_up_pow2(target_rows);

    candidate_pool<scalar_> pool {
        nk::tensor<scalar_, nk::aligned_allocator<scalar_, 64>, 2>::try_zeros({row_count, dim}),
        {}};
    for (std::size_t row = 0; row < row_count; ++row) populate(pool.matrix.span().row(row).as_vector(), row);
    pool.norms.reserve(row_count);
    for (std::size_t row = 0; row < row_count; ++row)
        pool.norms.push_back(inner_norms<nk_cap_serial_k>(pool.matrix.view().row(row).as_vector()));
    return pool;
}

/**
 *  @brief Reports per-cell counters - items/sec (distances), bytes/sec (candidate memory
 *  traffic; query is reused), and gso/s (giga scalar-operations per second; one distance
 *  is approximated as 2*dim multiply-add or bit-op pairs across all five kernels).
 */
inline void set_counters(benchmark::State& state, std::size_t dim, std::size_t distances_per_iter,
                         std::size_t bytes_loaded_per_iter) noexcept {
    state.SetItemsProcessed(state.iterations() * distances_per_iter);
    state.SetBytesProcessed(state.iterations() * bytes_loaded_per_iter);
    double const scalar_ops_per_iter = 2.0 * static_cast<double>(dim) * static_cast<double>(distances_per_iter);
    state.counters["gso/s"] = benchmark::Counter(scalar_ops_per_iter, benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::kIs1000);
}

/** @brief Computes a row's contiguous byte width once per cell (used for bytes_processed). */
template <typename scalar_>
constexpr std::size_t row_bytes_of(std::size_t dim) noexcept {
    return (dim / nk::dimensions_per_value<scalar_>()) * sizeof(scalar_);
}

/** @brief USearch 1-to-1 cell - random row pick per iter so candidates stream from RAM. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
void bench_distance_1to1(benchmark::State& state) {
    auto const dim = static_cast<std::size_t>(state.range(0));
    auto query = nk::vector<scalar_>::try_zeros(dim);
    populate(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();
    auto query_norms = inner_norms<caps_>(query_view);

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    std::mt19937 random_generator {0xDEADBEEFu};
    adjacency_simulator_t adjacency;
    for (auto _ : state) {
        std::size_t const row = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        auto candidate_view = pool.matrix.view().row(row).as_vector();
        auto distance_value = distance<kernel_, caps_>(query_view, query_norms, candidate_view, pool.norms[row]);
        benchmark::DoNotOptimize(distance_value);
    }
    set_counters(state, dim, /*distances=*/1, /*bytes=*/row_bytes_of<scalar_>(dim));
}

/** @brief USearch 1-to-4 cell - four random rows per iter; query is amortized 4x. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
void bench_distances_1to4(benchmark::State& state) {
    auto const dim = static_cast<std::size_t>(state.range(0));
    auto query = nk::vector<scalar_>::try_zeros(dim);
    populate(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();
    auto query_norms = inner_norms<caps_>(query_view);

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    std::mt19937 random_generator {0xDEADBEEFu};
    adjacency_simulator_t adjacency;
    for (auto _ : state) {
        std::size_t const row0 = random_generator() & row_index_mask;
        std::size_t const row1 = random_generator() & row_index_mask;
        std::size_t const row2 = random_generator() & row_index_mask;
        std::size_t const row3 = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        std::array<nk::vector_view<scalar_>, 4> candidate_views {
            pool.matrix.view().row(row0).as_vector(), pool.matrix.view().row(row1).as_vector(),
            pool.matrix.view().row(row2).as_vector(), pool.matrix.view().row(row3).as_vector()};
        std::array<vector_norms<typename scalar_::dot_result_t>, 4> candidate_norms {
            pool.norms[row0], pool.norms[row1], pool.norms[row2], pool.norms[row3]};
        auto distance_values = distances<kernel_, caps_>(query_view, query_norms, candidate_views, candidate_norms);
        benchmark::DoNotOptimize(distance_values);
    }
    set_counters(state, dim, /*distances=*/4, /*bytes=*/4 * row_bytes_of<scalar_>(dim));
}

/**
 *  @brief NumKong monolithic 1-to-1 baseline using the default @c prefer_simd_k - one
 *  candidate per loop iter, randomly picked from the same overflowed pool as the USearch cells.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
void bench_baseline(benchmark::State& state) {
    auto const dim = static_cast<std::size_t>(state.range(0));
    auto query = nk::vector<scalar_>::try_zeros(dim);
    populate(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    std::mt19937 random_generator {0xDEADBEEFu};
    adjacency_simulator_t adjacency;
    for (auto _ : state) {
        std::size_t const row = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        auto candidate_view = pool.matrix.view().row(row).as_vector();
        if constexpr (kernel_ == nk_kernel_dot_k) {
            typename scalar_::dot_result_t result {};
            nk::dot(query_view, candidate_view, dim, &result);
            benchmark::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_angular_k) {
            typename scalar_::angular_result_t result {};
            nk::angular(query_view, candidate_view, dim, &result);
            benchmark::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_euclidean_k) {
            typename scalar_::euclidean_result_t result {};
            nk::euclidean(query_view, candidate_view, dim, &result);
            benchmark::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_hamming_k) {
            typename scalar_::hamming_result_t result {};
            nk::hamming(query_view, candidate_view, dim, &result);
            benchmark::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_jaccard_k) {
            typename scalar_::jaccard_result_t result {};
            nk::jaccard(query_view, candidate_view, dim, &result);
            benchmark::DoNotOptimize(result);
        }
    }
    set_counters(state, dim, /*distances=*/1, /*bytes=*/row_bytes_of<scalar_>(dim));
}

/** @brief No-op overload picked when the (kernel, scalar, caps) tuple isn't wired up. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires(!kernel_supported<kernel_, scalar_, caps_>)
void register_one(std::string_view, std::string_view, std::string_view) noexcept {}

/** @brief Registers the 1-to-1 and 1-to-4 USearch cells for one supported tuple. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires kernel_supported<kernel_, scalar_, caps_>
void register_one(std::string_view kernel_name, std::string_view scalar_name, std::string_view caps_name) noexcept {
    auto register_with = [&](auto* function, std::string_view shape) {
        std::string const label = fmt::format("{}/{}/{}/{}", kernel_name, scalar_name, caps_name, shape);
        auto* registered = benchmark::RegisterBenchmark(label.c_str(), function);
        for (std::size_t dim : dimensions_to_bench) registered->Arg(static_cast<std::int64_t>(dim));
    };
    register_with(&bench_distance_1to1<kernel_, scalar_, caps_>, "1to1");
    register_with(&bench_distances_1to4<kernel_, scalar_, caps_>, "1to4");
}

/** @brief No-op overload picked when this (kernel, scalar) pair isn't wired up by USearch. */
template <nk_kernel_kind_t kernel_, typename scalar_>
    requires(!kernel_in_scope<kernel_, scalar_>)
void register_baseline(std::string_view, std::string_view) noexcept {}

/**
 *  @brief Registers the NumKong baseline cell for one (kernel, scalar) pair. Caps-agnostic
 *  - NumKong's default @c prefer_simd_k template arg picks the best backend at compile time.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
    requires kernel_in_scope<kernel_, scalar_>
void register_baseline(std::string_view kernel_name, std::string_view scalar_name) noexcept {
    std::string const label = fmt::format("{}/{}/baseline", kernel_name, scalar_name);
    auto* registered = benchmark::RegisterBenchmark(label.c_str(), &bench_baseline<kernel_, scalar_>);
    for (std::size_t dim : dimensions_to_bench) registered->Arg(static_cast<std::int64_t>(dim));
}

template <nk_kernel_kind_t kernel_, typename scalar_>
void register_metric_for_type(std::string_view kernel_name, std::string_view scalar_name) noexcept {
    register_baseline<kernel_, scalar_>(kernel_name, scalar_name);
    register_one<kernel_, scalar_, nk_cap_genoa_k>(kernel_name, scalar_name, "genoa");
    register_one<kernel_, scalar_, nk_cap_icelake_k>(kernel_name, scalar_name, "icelake");
    register_one<kernel_, scalar_, nk_cap_skylake_k>(kernel_name, scalar_name, "skylake");
    register_one<kernel_, scalar_, nk_cap_haswell_k>(kernel_name, scalar_name, "haswell");
    register_one<kernel_, scalar_, nk_cap_serial_k>(kernel_name, scalar_name, "serial");
}

template <typename scalar_>
void register_metrics_for_type(std::string_view scalar_name) noexcept {
    register_metric_for_type<nk_kernel_dot_k, scalar_>("dot", scalar_name);
    register_metric_for_type<nk_kernel_angular_k, scalar_>("angular", scalar_name);
    register_metric_for_type<nk_kernel_euclidean_k, scalar_>("euclidean", scalar_name);
}

/** @brief Binary inputs run dot/hamming/jaccard instead of dot/angular/euclidean. */
template <>
void register_metrics_for_type<nk::u1x8_t>(std::string_view scalar_name) noexcept {
    register_metric_for_type<nk_kernel_dot_k, nk::u1x8_t>("dot", scalar_name);
    register_metric_for_type<nk_kernel_hamming_k, nk::u1x8_t>("hamming", scalar_name);
    register_metric_for_type<nk_kernel_jaccard_k, nk::u1x8_t>("jaccard", scalar_name);
}

int main(int argc, char** argv) {
    register_metrics_for_type<nk::f64_t>("f64");
    register_metrics_for_type<nk::f32_t>("f32");
    register_metrics_for_type<nk::bf16_t>("bf16");
    register_metrics_for_type<nk::f16_t>("f16");
    register_metrics_for_type<nk::e5m2_t>("e5m2");
    register_metrics_for_type<nk::e4m3_t>("e4m3");
    register_metrics_for_type<nk::e3m2_t>("e3m2");
    register_metrics_for_type<nk::e2m3_t>("e2m3");
    register_metrics_for_type<nk::i8_t>("i8");
    register_metrics_for_type<nk::i4x2_t>("i4x2");
    register_metrics_for_type<nk::u8_t>("u8");
    register_metrics_for_type<nk::u4x2_t>("u4x2");
    register_metrics_for_type<nk::u1x8_t>("u1x8");

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
