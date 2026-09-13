/**
 *  @brief Throughput benchmarks for @c dense/ram_metric.hpp distance functions for 1-to-1, 1-to-4,
 *      and tiled kernels against NumKong's monolithic 1-to-1 reference.
 *
 *  @author Ash Vardanian
 *  @file bench/dense/ram_metric.cpp
 *  @date April 23, 2026
 *
 *  Mirrors the structure of @c test/dense_metric.hpp - the @c quads_supported concept
 *  gates each cell so unadapted (kernel, scalar, caps) tuples don't get registered, and
 *  new kits or chains light up matching benchmark cells automatically. Three flavors per
 *  supported tuple: USearch's @c distance (1-to-1), USearch's @c distances (1-to-4 fan-out),
 *  and NumKong's @c nk::dot / @c nk::angular / @c nk::euclidean / @c nk::hamming /
 *  @c nk::jaccard monolithic baseline using the default @c prefer_simd_k compile-time
 *  dispatch (registered once per kernel-scalar, since the baseline isn't caps-parameterized).
 *  Where a tile engine is adapted and the machine offers it, a fourth cell measures one
 *  16×16 tile of distances per iteration - gather, pack, and the depth walk - so the tile
 *  engine's primitive throughput reads next to 256 quads-path distances.
 */
#include <algorithm>
#include <array>
#include <cstdint>
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

#include "usearch/dense/ram_metric.hpp"

namespace bm = benchmark;
using namespace unum::usearch;

constexpr std::array<std::size_t, 3> dimensions_to_bench = {256, 768, 1536};

/**
 *  @brief True iff @c dense_quads_chain has a specialization for this (kernel, scalar, caps)
 *  tuple - unadapted tuples are skipped at registration time.
 */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept quads_supported = requires { typename dense_quads_chain<kernel_, scalar_, caps_>::kit_t; };

/**
 *  @brief True iff USearch has any kit for this (kernel, scalar) pair across capability
 *  tiers - serial is the always-available floor, so checking it covers the matrix.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
concept quads_in_scope = quads_supported<kernel_, scalar_, nk_cap_serial_k>;

/**
 *  @brief Build a scalar from a small unsigned - sub-byte goes through @c sub_byte_ref as
 *  raw @c int8_t, float wrappers via @c from_f32, byte-or-larger ints via the integral ctor.
 */
template <typename scalar_>
static auto make_scalar(unsigned value) noexcept {
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<i8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** @brief Populates @p vec with cyclic small representable values at @p rotation. */
template <typename scalar_>
static void fill_cyclic(nk::vector_span<scalar_> vec, std::size_t rotation) noexcept {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    for (std::size_t dimension_index = 0; dimension_index < vec.size(); ++dimension_index)
        vec[dimension_index] = make_scalar<scalar_>(
            universally_representable_values[(dimension_index + rotation) % universally_representable_values.size()]);
}

/**
 *  @brief Round up to the next power of two so we can use a bitmask for cyclic indexing
 *  instead of a slow modulo on every loop iteration.
 */
static std::size_t round_up_pow2(std::size_t value) noexcept {
    std::size_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

/** @brief Pool footprint target - well above any current x86 L3 (16-128 MiB), so candidate
 *  rows stream from RAM rather than cache and the query-load amortization actually matters. */
constexpr std::size_t candidate_pool_target_bytes_k = 1ull << 28; // 256 MiB

/**
 *  @brief Synthetic L1 working-set noise modelling an HNSW worker's per-iteration touches
 *  (neighbor list of @c u32_t IDs, visited bitset, top-K queue). Without this, the
 *  kernel-only microbench has no L1 contestants - the query stays resident and the cache
 *  hints inside the kits (PREFETCHT0 / PREFETCHT2 / CLDEMOTE) have nothing to optimize for.
 *  A pool larger than L1 (32 KiB on most x86) plus a random per-iter pick defeats the HW
 *  prefetcher and forces capacity pressure on the query lines.
 */
struct adjacency_simulator_t {
    using node_id_t = u32_t;
    static constexpr std::size_t bytes_per_node_k = 256; // 4 cache lines / iter
    static constexpr std::size_t ids_per_node_k = bytes_per_node_k / sizeof(node_id_t);
    static constexpr std::size_t node_count_k = 256; // 64 KiB pool, > 32 KiB L1
    static constexpr std::size_t node_index_mask_k = node_count_k - 1;
    std::vector<node_id_t> pool;

    adjacency_simulator_t() : pool(node_count_k * ids_per_node_k, 0) {
        splitmix64_t seed_generator {0xABCDEF01ull};
        for (auto& id : pool) id = static_cast<node_id_t>(seed_generator());
    }

    /** @brief Streaming xor over one "node"'s worth of IDs; one DoNotOptimize per call. */
    inline void touch(std::size_t random_index) noexcept {
        node_id_t const* node = pool.data() + (random_index & node_index_mask_k) * ids_per_node_k;
        node_id_t accumulator = 0;
        for (std::size_t id_index = 0; id_index < ids_per_node_k; ++id_index) accumulator ^= node[id_index];
        bm::DoNotOptimize(accumulator);
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
static candidate_pool<scalar_> make_candidate_pool(std::size_t dim) {
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<scalar_>();
    std::size_t const row_bytes = (dim / dims_per_value) * sizeof(scalar_);
    std::size_t const target_rows = std::max<std::size_t>(8, candidate_pool_target_bytes_k / row_bytes);
    std::size_t const row_count = round_up_pow2(target_rows);

    candidate_pool<scalar_> pool {
        nk::tensor<scalar_, nk::aligned_allocator<scalar_, 64>, 2>::try_zeros({row_count, dim}),
        {}};
    for (std::size_t row = 0; row < row_count; ++row) fill_cyclic(pool.matrix.span().row(row).as_vector(), row);
    pool.norms.reserve(row_count);
    for (std::size_t row = 0; row < row_count; ++row)
        pool.norms.push_back(dense_norms<nk_cap_serial_k>(pool.matrix.view().row(row).as_vector()));
    return pool;
}

/**
 *  @brief Reports per-cell counters - items/sec (distances), bytes/sec (candidate memory
 *  traffic; query is reused), and gso/s (giga scalar-operations per second; one distance
 *  is approximated as 2*dim multiply-add or bit-op pairs across all five kernels).
 */
static void set_counters(bm::State& state, std::size_t dim, std::size_t distances_per_iter,
                         std::size_t bytes_loaded_per_iter) noexcept {
    state.SetItemsProcessed(state.iterations() * distances_per_iter);
    state.SetBytesProcessed(state.iterations() * bytes_loaded_per_iter);
    double const scalar_ops_per_iter = 2.0 * static_cast<double>(dim) * static_cast<double>(distances_per_iter);
    state.counters["gso/s"] = bm::Counter(scalar_ops_per_iter, bm::Counter::kIsIterationInvariantRate,
                                          bm::Counter::kIs1000);
}

/** @brief Computes a row's contiguous byte width once per cell (used for bytes_processed). */
template <typename scalar_>
static constexpr std::size_t row_bytes_of(std::size_t dim) noexcept {
    return (dim / nk::dimensions_per_value<scalar_>()) * sizeof(scalar_);
}

/** @brief USearch 1-to-1 cell - random row pick per iter so candidates stream from RAM. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void bench_distance(bm::State& state) {
    auto const dim = static_cast<std::size_t>(state.range(0));
    auto query = nk::vector<scalar_>::try_zeros(dim);
    fill_cyclic(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();
    auto query_norms = dense_norms<caps_>(query_view);

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    splitmix64_t random_generator {0xDEADBEEFull};
    adjacency_simulator_t adjacency;
    for (auto _ : state) {
        std::size_t const row = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        auto candidate_view = pool.matrix.view().row(row).as_vector();
        auto distance_value = distance<kernel_, caps_>(query_view, query_norms, candidate_view, pool.norms[row]);
        bm::DoNotOptimize(distance_value);
    }
    set_counters(state, dim, /*distances_per_iter=*/1, /*bytes_loaded_per_iter=*/row_bytes_of<scalar_>(dim));
}

/** @brief USearch 1-to-4 cell - four random rows per iter; query is amortized 4x. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void bench_distances_quad(bm::State& state) {
    auto const dim = static_cast<std::size_t>(state.range(0));
    auto query = nk::vector<scalar_>::try_zeros(dim);
    fill_cyclic(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();
    auto query_norms = dense_norms<caps_>(query_view);

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    splitmix64_t random_generator {0xDEADBEEFull};
    adjacency_simulator_t adjacency;
    for (auto _ : state) {
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
        auto distance_values = distances<kernel_, caps_>(query_view, query_norms, candidate_views, candidate_norms);
        bm::DoNotOptimize(distance_values);
    }
    set_counters(state, dim, /*distances_per_iter=*/4, /*bytes_loaded_per_iter=*/4 * row_bytes_of<scalar_>(dim));
}

/**
 *  @brief NumKong monolithic 1-to-1 baseline using the default @c prefer_simd_k - one
 *  candidate per loop iter, randomly picked from the same overflowed pool as the USearch cells.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
static void bench_baseline(bm::State& state) {
    auto const dim = static_cast<std::size_t>(state.range(0));
    auto query = nk::vector<scalar_>::try_zeros(dim);
    fill_cyclic(query.span(), 0);
    auto pool = make_candidate_pool<scalar_>(dim);
    auto query_view = query.view();

    std::size_t const row_index_mask = pool.matrix.extent(0) - 1;
    splitmix64_t random_generator {0xDEADBEEFull};
    adjacency_simulator_t adjacency;
    for (auto _ : state) {
        std::size_t const row = random_generator() & row_index_mask;
        adjacency.touch(random_generator());
        auto candidate_view = pool.matrix.view().row(row).as_vector();
        if constexpr (kernel_ == nk_kernel_dot_k) {
            typename scalar_::dot_result_t result {};
            nk::dot(query_view, candidate_view, dim, &result);
            bm::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_angular_k) {
            typename scalar_::angular_result_t result {};
            nk::angular(query_view, candidate_view, dim, &result);
            bm::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_euclidean_k) {
            typename scalar_::euclidean_result_t result {};
            nk::euclidean(query_view, candidate_view, dim, &result);
            bm::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_hamming_k) {
            typename scalar_::hamming_result_t result {};
            nk::hamming(query_view, candidate_view, dim, &result);
            bm::DoNotOptimize(result);
        }
        else if constexpr (kernel_ == nk_kernel_jaccard_k) {
            typename scalar_::jaccard_result_t result {};
            nk::jaccard(query_view, candidate_view, dim, &result);
            bm::DoNotOptimize(result);
        }
    }
    set_counters(state, dim, /*distances_per_iter=*/1, /*bytes_loaded_per_iter=*/row_bytes_of<scalar_>(dim));
}

/** @brief No-op overload picked when the (kernel, scalar, caps) tuple isn't wired up. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires(!quads_supported<kernel_, scalar_, caps_>)
static void register_quads(std::string_view, std::string_view, std::string_view) noexcept {}

/** @brief Registers the 1-to-1 and 1-to-4 USearch cells for one supported tuple. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires quads_supported<kernel_, scalar_, caps_>
static void register_quads(std::string_view kernel_name, std::string_view scalar_name,
                           std::string_view caps_name) noexcept {
    auto register_with = [&](auto* function, std::string_view shape) {
        std::string const label = fmt::format("{}/{}/{}/{}", kernel_name, scalar_name, caps_name, shape);
        auto* registered = bm::RegisterBenchmark(label.c_str(), function);
        for (std::size_t const dim : dimensions_to_bench) registered->Arg(static_cast<i64_t>(dim));
    };
    register_with(&bench_distance<kernel_, scalar_, caps_>, "1to1");
    register_with(&bench_distances_quad<kernel_, scalar_, caps_>, "1to4");
}

/** @brief No-op overload picked when this (kernel, scalar) pair isn't wired up by USearch. */
template <nk_kernel_kind_t kernel_, typename scalar_>
    requires(!quads_in_scope<kernel_, scalar_>)
static void register_baseline(std::string_view, std::string_view) noexcept {}

/**
 *  @brief Registers the NumKong baseline cell for one (kernel, scalar) pair. Caps-agnostic
 *  - NumKong's default @c prefer_simd_k template arg picks the best backend at compile time.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
    requires quads_in_scope<kernel_, scalar_>
static void register_baseline(std::string_view kernel_name, std::string_view scalar_name) noexcept {
    std::string const label = fmt::format("{}/{}/baseline", kernel_name, scalar_name);
    auto* registered = bm::RegisterBenchmark(label.c_str(), &bench_baseline<kernel_, scalar_>);
    for (std::size_t const dim : dimensions_to_bench) registered->Arg(static_cast<i64_t>(dim));
}

/** @brief True iff a tile engine is compiled for this tuple - the tiles chain always names
 *      a kit, so the gate is its @c has_tiles_k verdict, never the alias's existence. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept tiles_supported = dense_tiles_chain<kernel_, scalar_, caps_>::has_tiles_k;

/** @brief One 16×16 tile of distances per iteration - gather, pack, and the depth walk - the tile
 *      engine's primitive throughput, comparable against 256 quads-path distances. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires tiles_supported<kernel_, scalar_, caps_>
static void bench_distances_tile(bm::State& state) {
    nk_configure_thread(nk_capabilities_available());
    using tiles_kit_t = typename dense_tiles_chain<kernel_, scalar_, caps_>::kit_t;
    constexpr std::size_t rows_k = tiles_kit_t::tile_rows_k;
    std::size_t const dim = static_cast<std::size_t>(state.range(0));
    std::size_t const depth_tiles = dim / tiles_kit_t::tile_depth_k;

    auto storage = nk::matrix<scalar_>::try_zeros({rows_k, dim});
    auto const vectors = storage.view();
    std::array<nk::vector_view<scalar_>, rows_k> views;
    std::array<vector_norms<typename scalar_::dot_result_t>, rows_k> norms;
    for (std::size_t row = 0; row != rows_k; ++row) {
        fill_cyclic(storage.row(row).as_vector(), row);
        views[row] = vectors.row(row).as_vector();
        norms[row] = dense_norms<caps_>(views[row]);
    }
    typename tiles_kit_t::a_tile_t a_tile;
    typename tiles_kit_t::b_tile_t b_tile;
    auto tiled = nk::matrix<f32_t>::try_zeros({rows_k, rows_k});
    tiles_kit_t::configure_tiles();

    for (auto _ : state) {
        typename tiles_kit_t::accumulator_tile_t accumulator;
        tiles_kit_t::init_tile(accumulator);
        for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile) {
            tiles_kit_t::gather_a_tile(a_tile, views, depth_tile * tiles_kit_t::tile_depth_k);
            tiles_kit_t::pack_b_tile(a_tile, b_tile);
            tiles_kit_t::update_tile(accumulator, a_tile, b_tile);
        }
        tiles_kit_t::finalize_tile(accumulator, norms, norms, tiled.data(), rows_k);
        bm::DoNotOptimize(tiled);
    }
    set_counters(state, dim, /*distances_per_iter=*/rows_k * rows_k,
                 /*bytes_loaded_per_iter=*/rows_k * row_bytes_of<scalar_>(dim));
}

/** @brief No-op sibling when no tile engine exists for the tuple. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires(!tiles_supported<kernel_, scalar_, caps_>)
static void register_tiles(std::string_view, std::string_view, std::string_view) noexcept {}

/** @brief Registers the tile-engine Gram cell for one supported tuple, only where the
 *      machine actually offers the engine - a registered-but-crashing cell helps nobody. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires tiles_supported<kernel_, scalar_, caps_>
static void register_tiles(std::string_view kernel_name, std::string_view scalar_name,
                           std::string_view caps_name) noexcept {
    if ((nk_capabilities_available() & caps_) != caps_) return;
    std::string const label = fmt::format("{}/{}/{}/16x16", kernel_name, scalar_name, caps_name);
    auto* registered = bm::RegisterBenchmark(label.c_str(), &bench_distances_tile<kernel_, scalar_, caps_>);
    for (std::size_t const dim : dimensions_to_bench) registered->Arg(static_cast<i64_t>(dim));
}

template <nk_kernel_kind_t kernel_, typename scalar_>
static void register_kernel(std::string_view kernel_name, std::string_view scalar_name) noexcept {
    register_baseline<kernel_, scalar_>(kernel_name, scalar_name);
    // Newest silicon first, split by the architecture-family macros like the test's tier
    // sweep - the concept-gated no-ops silently absorb any tier without a kit.
#if NK_TARGET_X8664_
    register_quads<kernel_, scalar_, nk_cap_diamond_k>(kernel_name, scalar_name, "diamond");
    register_quads<kernel_, scalar_, nk_cap_sierra_k>(kernel_name, scalar_name, "sierra");
    register_quads<kernel_, scalar_, nk_cap_genoa_k>(kernel_name, scalar_name, "genoa");
    register_quads<kernel_, scalar_, nk_cap_alder_k>(kernel_name, scalar_name, "alder");
    register_quads<kernel_, scalar_, nk_cap_icelake_k>(kernel_name, scalar_name, "icelake");
    register_quads<kernel_, scalar_, nk_cap_skylake_k>(kernel_name, scalar_name, "skylake");
    register_quads<kernel_, scalar_, nk_cap_haswell_k>(kernel_name, scalar_name, "haswell");
    register_tiles<kernel_, scalar_, nk_cap_sapphireamx_k>(kernel_name, scalar_name, "sapphireamx");
#elif NK_TARGET_ARM64_
    register_quads<kernel_, scalar_, nk_cap_neonfp8_k>(kernel_name, scalar_name, "neonfp8");
    register_quads<kernel_, scalar_, nk_cap_neonfhm_k>(kernel_name, scalar_name, "neonfhm");
    register_quads<kernel_, scalar_, nk_cap_neonbfdot_k>(kernel_name, scalar_name, "neonbfdot");
    register_quads<kernel_, scalar_, nk_cap_neonsdot_k>(kernel_name, scalar_name, "neonsdot");
    register_quads<kernel_, scalar_, nk_cap_neon_k>(kernel_name, scalar_name, "neon");
#elif NK_TARGET_WASM_
    register_quads<kernel_, scalar_, nk_cap_v128relaxed_k>(kernel_name, scalar_name, "v128relaxed");
    register_quads<kernel_, scalar_, nk_cap_v128_k>(kernel_name, scalar_name, "v128");
#elif NK_TARGET_POWER64_
    register_quads<kernel_, scalar_, nk_cap_powervsx_k>(kernel_name, scalar_name, "powervsx");
#endif
    register_quads<kernel_, scalar_, nk_cap_serial_k>(kernel_name, scalar_name, "serial");
}

template <typename scalar_>
static void register_scalar(std::string_view scalar_name) noexcept {
    register_kernel<nk_kernel_dot_k, scalar_>("dot", scalar_name);
    register_kernel<nk_kernel_angular_k, scalar_>("angular", scalar_name);
    register_kernel<nk_kernel_euclidean_k, scalar_>("euclidean", scalar_name);
}

/** @brief Binary inputs run dot/hamming/jaccard instead of dot/angular/euclidean. */
template <>
void register_scalar<nk::u1x8_t>(std::string_view scalar_name) noexcept {
    register_kernel<nk_kernel_dot_k, nk::u1x8_t>("dot", scalar_name);
    register_kernel<nk_kernel_hamming_k, nk::u1x8_t>("hamming", scalar_name);
    register_kernel<nk_kernel_jaccard_k, nk::u1x8_t>("jaccard", scalar_name);
}

int main(int argc, char** argv) {

    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv)) return 1;

    register_scalar<nk::f64_t>("f64");
    register_scalar<nk::f32_t>("f32");
    register_scalar<nk::bf16_t>("bf16");
    register_scalar<nk::f16_t>("f16");
    register_scalar<nk::e5m2_t>("e5m2");
    register_scalar<nk::e4m3_t>("e4m3");
    register_scalar<nk::e3m2_t>("e3m2");
    register_scalar<nk::e2m3_t>("e2m3");

    register_scalar<nk::i8_t>("i8");
    register_scalar<nk::i4x2_t>("i4x2");

    register_scalar<nk::u8_t>("u8");
    register_scalar<nk::u4x2_t>("u4x2");
    register_scalar<nk::u1x8_t>("u1x8");

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}
