/**
 *  @file test/dense.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The index-agnostic test scenario library shared by every dense-index residency - volatile
 *      in-RAM, GPU-resident, and persisted alike.
 *
 *  Every scenario is a template over the index type, and each translation unit composes its
 *  own suite explicitly: the calls a TU makes ARE its capability claims, so a backend that
 *  cannot race simply never calls the racing scenarios, and the omission is auditable at
 *  the call site instead of hiding in a trait. Translation units name their concrete index
 *  instantiations and pass a ledgered factory constructing them - this header names no
 *  index and assumes no constructor shape, so the same scenarios compile under host
 *  compilers and nvcc.
 *
 *  Allocators are tested by construction: a counting allocator flows through the vectors &
 *  neighbors parameters to prove propagation, tallying live bytes into a ledger that must
 *  balance to zero when each scenario's scope closes. Parallel scenarios use ForkUnion's
 *  default flat pool - no NUMA machinery in tests, except the sharded-index scenarios, which
 *  need a real topology because an executor colocates each arena with the thread that walks it.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cmath>   // `std::exp`, `std::fabs` - the search and algebra oracles
#include <cstdint> // `std::uintptr_t` - the checkpoint paths' per-run salt
#include <cstdio>  // `std::snprintf`, `std::remove` - the checkpoint round-trip's files
#include <cstdlib> // `std::abs`
#include <cstring> // `std::strlen`, `std::memcmp` - the fetched-row byte comparison

#include <algorithm> // `std::sort`, `std::unique`, `std::all_of`, `std::rotate`
#include <array>     // `std::array`
#include <atomic>    // `std::atomic`
#include <bit>       // `std::bit_cast` - the ULP distance of the oracle audits; `std::bit_floor` - a cell's ceiling
#include <chrono>    // `std::chrono::microseconds`
#include <memory>    // `std::allocator` - the part path buffer, `std::unique_ptr` - every scenario's index
#include <thread>    // `std::this_thread::sleep_for`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include <forkunion.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>

#include "harness.hpp"
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp"        // `forkunion_executor_t` - the RAM residency's executor
#include "usearch/utilities/sanitizers.hpp"        // `assert_no_locks_held` - the racing threads' tail check
#include "usearch/utilities/safegraphs_format.hpp" // `save_config_t`, `safegraphs_read_header` - the checkpoints' surface
#include "usearch/dense/ram_metric.hpp"            // `dense::distance`, `dense::norms` - the oracle's kernels

namespace unum::usearch::test {
using namespace unum::usearch::dense;

#pragma region Constants

/** Storage values per vector in the larger scenarios - NumKong's unit of one
 *  scalar object; small enough to keep serial kernels fast. */
constexpr std::size_t dense_values_k = 16;

/** Storage values per vector in the tiny smoke scenarios. */
constexpr std::size_t dense_tiny_values_k = 8;

/** Logical dimensions spanned by @p values_count storage values of one dtype -
 *  NumKong's @c dimensions_per_value conversion, so every dtype gets the same storage
 *  budget: 16 i8 values carry 16 dimensions, 16 u1x8 values carry 128. */
template <typename scalar_>
constexpr std::size_t dense_dimensions_for(std::size_t values_count) noexcept {
    return values_count * nk::dimensions_per_value<scalar_>();
}

/** Node count marching the growth scenario across several doubling chunks. */
constexpr std::size_t dense_growth_nodes_k = 1'500;

/** Nodes inserted by each thread of the concurrent scenarios. */
constexpr std::size_t dense_thread_nodes_k = 256;

/** Self-recall sampling threshold: at least this share of samples must find themselves. */
constexpr std::size_t dense_minimal_recall_percent_k = 90;

/** Distance bound for a stored vector matching itself under symmetric kernels. */
constexpr float dense_self_match_tolerance_k = 1e-3f;

/** The reader thread performs at least this many searches, even if writers finish first. */
constexpr std::size_t dense_reader_minimal_searches_k = 16;

#pragma endregion Constants

#pragma region Shared Helpers

/** Every NVIDIA capability, so a device test cell runs each kit its build compiled. */
inline constexpr nk_capability_t dense_nvidia_capabilities_k = nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k |
                                                               nk_cap_hopper_k | nk_cap_blackwell_k |
                                                               nk_cap_blackwellrtx_k | nk_cap_blackwellultra_k;

/** Gate: a cell's kit is compiled once its requested ceiling is a bit some chain actually
 *  picked - a build without the kit must not pass its serial floor under a SIMD name. */
template <nk_dtype_t dtype_, nk_capability_t caps_>
concept dense_cell_compiled =
    caps_ == nk_cap_serial_k ||
    std::bit_floor(caps_) == quads_chain<typename nk::type_for<dtype_>::type, caps_>::picked_k ||
    std::bit_floor(caps_) == tiles_chain<typename nk::type_for<dtype_>::type, caps_>::picked_k;

/** Gate: a cell runs once its serial kit finalizes the kernel and its requested ceiling picked a
 *  kit. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, nk_capability_t caps_>
concept dense_cell_testable = quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k> &&
                              dense_cell_compiled<dtype_, caps_>;

/** Whether this machine offers every bit a cell's kit and atomic reference declare - both
 *  execute their instructions unconditionally - arming the per-process tile handshake on the way. */
template <nk_capability_t caps_, template <typename> class atomic_ref_>
static bool dense_cell_admitted() noexcept {
    constexpr fu::capabilities_t atomics_k = atomic_ref_<u32_t>::capabilities_k;
    if ((fu::runtime_capabilities() & atomics_k) != atomics_k) return false;
    if constexpr (caps_ != nk_cap_serial_k) {
        nk_capability_t available = 0;
        if (nk_cpu_capabilities_enabled(&available) != nk_success_k || (available & caps_) != caps_) return false;
        if (nk_cpu_configure_thread(available) != nk_success_k) return false;
    }
    return true;
}

/** A fresh traversal context for the index @p make_index builds: a device cell's own, on its
 *  stream, and a default-built one for every other factory. */
template <typename index_type_, typename make_index_type_>
typename index_type_::traversal_context_t dense_make_context(make_index_type_ const& make_index) {
    if constexpr (requires { make_index.context(); }) return make_index.context();
    else return typename index_type_::traversal_context_t {{}};
}

/** Per-thread state for the concurrent scenarios: one vector slice & one traversal context each. */
template <typename index_type_>
struct dense_threads {
    std::vector<nk::matrix<typename index_type_::scalar_t>> slices;
    std::vector<typename index_type_::traversal_context_t> contexts;
};

/** Prepares @p threads_count threads with distinctly-seeded vector slices & fresh contexts for the
 *  index @p make_index builds. */
template <typename index_type_, typename make_index_type_>
dense_threads<index_type_> dense_make_threads(make_index_type_ const& make_index, std::size_t threads_count,
                                              std::size_t nodes_per_thread, std::size_t dimensions, u64_t seed) {
    dense_threads<index_type_> threads;
    for (std::size_t thread = 0; thread != threads_count; ++thread) {
        threads.slices.push_back(
            make_vectors<typename index_type_::scalar_t>(nodes_per_thread, dimensions, seed + thread));
        threads.contexts.push_back(dense_make_context<index_type_>(make_index));
    }
    return threads;
}

/** Bytes of the first @p rows_count rows of a row-major matrix view - the stride
 *  already accounts for sub-byte packing. */
template <typename matrix_view_type_>
constexpr std::size_t dense_rows_bytes(matrix_view_type_ view, std::size_t rows_count) noexcept {
    return rows_count * static_cast<std::size_t>(view.stride_bytes(0));
}

/** Sampled self-recall over the whole result pool: each sampled stored row must
 *  surface some pool entry inside the self-match band. Pool-scan by distance is the
 *  one criterion: duplicate ties - binary vectors above all - make the top-1 slot
 *  and ID-matching unfair judges of a healthy graph, and a distance criterion needs
 *  no assigned-IDs plumbing, so it survives reloads & thread-sliced builds unchanged.
 *  Sampling is evenly spaced; callers keep the dot-kernel gate, where "distance" is a
 *  similarity and self-matching is undefined. Requires @p rows_count ≥ @p samples_count.
 *  The searches walk a fresh context for the index @p make_index builds. */
template <typename index_type_, typename make_index_type_, typename vectors_view_type_>
std::size_t dense_count_self_hits(index_type_& index, make_index_type_ const& make_index, vectors_view_type_ vectors,
                                  std::size_t rows_count, std::size_t samples_count) {
    std::size_t hits = 0;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    for (std::size_t sample = 0; sample != samples_count; ++sample) {
        std::size_t const row = sample * (rows_count / samples_count);
        auto const [result, status] = index.search(vectors.row(row).as_vector(), context);
        if (failed(status) || !result.count) continue;
        for (std::size_t match = 0; match != result.count; ++match) {
            scored_node_t const scored = context.results[match];
            if (scored.distance < dense_self_match_tolerance_k && scored.distance > -dense_self_match_tolerance_k) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

/** The judged form of @ref dense_count_self_hits: the share found must clear the floor. */
template <typename index_type_, typename make_index_type_, typename vectors_view_type_>
std::size_t dense_expect_self_recall(index_type_& index, make_index_type_ const& make_index, vectors_view_type_ vectors,
                                     std::size_t rows_count, std::size_t samples_count) {
    std::size_t const hits = dense_count_self_hits(index, make_index, vectors, rows_count, samples_count);
    return expect(hits * 100 >= samples_count * dense_minimal_recall_percent_k);
}

/** The ranking in @p context ascends by distance - the pool's invariant after any walk. */
template <typename context_type_>
std::size_t dense_expect_ascending(context_type_ const& context) noexcept {
    std::size_t failures = 0;
    for (std::size_t rank = 1; rank < context.results.size(); ++rank)
        failures += expect(!(context.results[rank].distance < context.results[rank - 1].distance));
    return failures;
}

/** One caller-built traversal context over its own arena, sized for the shard geometry -
 *  non-movable, since the context's allocator points into the arena beside it. */
template <typename indexes_type_>
struct dense_arena_context {

    /** The backing run every carve of the context lands in. */
    typename indexes_type_::executor_t::arena_t arena;

    /** The traversal context, allocating from the arena beside it. */
    typename indexes_type_::traversal_context_t context;

    /** The arena reservation's answer; every use checks it before walking. */
    status_t status;

    explicit dense_arena_context(indexes_type_& indexes) noexcept
        : arena(global_allocator), context(typename indexes_type_::traversal_context_t::allocator_t {arena}),
          status(arena.reserve_exactly(indexes.shard(0).traversal_memory_usage())) {}
    dense_arena_context(dense_arena_context&&) = delete;
    dense_arena_context(dense_arena_context const&) = delete;
};

/** Whether an index lives on a device, so its checkpoint plans take a stream. */
template <typename index_type_>
constexpr bool dense_device_resident = requires(index_type_ const& index) { index.device_specs(); };

/** Whether a dot index still writes norms: a compensated host kit finalizes even a dot through the
 *  row sums, while the device kits never do - asked without naming a device index's kit, which a
 *  device index need not expose. */
template <typename index_type_>
constexpr bool dense_dot_keeps_norms() noexcept {
    if constexpr (dense_device_resident<index_type_>) return false;
    else return quads_compensated<typename index_type_::quads_kit_t>;
}

namespace dense {

/**
 *  @brief One vendor's (dtype, kernel) cell: @c index_t over ledgered flat matrices, the sharded
 *      @c indexes_t, and the @c vendor_prefix_k its reports carry, built over the @c stream of the
 *      executor the vendor's entry opened.
 *
 *  A cell is the factory the shared scenarios take: called on a ledger it builds the index on the
 *  stream's device, and its @c context builds a traversal context on the stream. A cell whose
 *  vendor runs device attention names @c attends_k and covers it in its own unit; every other
 *  cell's index must refuse attention with @c unsupported_k.
 */
template <gpu_vendor_t vendor_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
struct vram_cell;

/** Ledgered device flat-matrix backend: a vendor's virtual-memory one with every committed slab
 *  tallied into the harness's @c allocation_ledger_t, so the ledger provably balances device
 *  bytes. */
template <typename virtual_memory_type_>
struct counting_device_allocator {
    using value_type = typename virtual_memory_type_::value_type;
    inline static constexpr auto address_space_k = virtual_memory_type_::address_space_k;
    template <typename other_>
    struct rebind {
        using other = counting_device_allocator<typename virtual_memory_type_::template rebind<other_>::other>;
    };

    virtual_memory_type_ virtual_memory_;
    allocation_ledger_t* ledger_ = nullptr;
    std::size_t tallied_bytes_ = 0;

    counting_device_allocator() noexcept = default;
    counting_device_allocator(allocation_ledger_t& ledger, int device) noexcept
        : virtual_memory_(device), ledger_(&ledger) {}
    counting_device_allocator(counting_device_allocator const& other) noexcept
        : virtual_memory_(other.virtual_memory_), ledger_(other.ledger_) {}
    template <typename other_>
    counting_device_allocator(counting_device_allocator<other_> const& other) noexcept
        : virtual_memory_(other.virtual_memory_), ledger_(other.ledger_) {}
    ~counting_device_allocator() noexcept { release(); }

    status_t reserve_range(std::size_t bytes) noexcept { return virtual_memory_.reserve_range(bytes); }
    status_t commit(std::size_t bytes) noexcept {
        status_t const status = virtual_memory_.commit(bytes);
        if (ledger_ && virtual_memory_.committed_bytes() > tallied_bytes_) {
            ledger_->live_bytes.fetch_add(
                static_cast<std::ptrdiff_t>(virtual_memory_.committed_bytes() - tallied_bytes_));
            ledger_->total_allocations.fetch_add(1);
            tallied_bytes_ = virtual_memory_.committed_bytes();
        }
        return status;
    }
    value_type* base() const noexcept { return virtual_memory_.base(); }
    std::size_t reserved_bytes() const noexcept { return virtual_memory_.reserved_bytes(); }
    std::size_t committed_bytes() const noexcept { return virtual_memory_.committed_bytes(); }
    int device() const noexcept { return virtual_memory_.device(); }
    void release() noexcept {
        if (ledger_ && tallied_bytes_) ledger_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(tallied_bytes_));
        tallied_bytes_ = 0;
        virtual_memory_.release();
    }
};

} // namespace dense

/** Whether a sharded index runs on ForkUnion threads - the RAM residency. */
template <typename indexes_type_>
concept ram_collection = std::is_same_v<typename indexes_type_::executor_t, forkunion_executor_t>;

/** The executor one sharded scenario drives, owned together with whatever it was spawned
 *  over; @c status reports the spawn. Defined per residency, each built from the @c placement
 *  the shared scenarios forward: none on RAM and NVMe, and on VRAM the executor the vendor's
 *  entry opened, whose device the shards and every further executor land on. */
template <typename indexes_type_>
struct dense_indexes_executor;

/** The RAM form: two threads over a topology of its own, both released with it. */
template <ram_collection indexes_type_>
struct dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;

    fu_topology_t topology = nullptr;
    executor_t executor;
    status_t status = status_t::uninitialized_k;

    dense_indexes_executor() noexcept {
        if (fu_topology_new(&topology) != fu_success_k) return;
        auto [spawn_executor, spawn_status] = executor_t::spawn(topology, 2);
        status = spawn_status;
        this->executor = std::move(spawn_executor);
    }
    ~dense_indexes_executor() noexcept {
        executor = executor_t {};
        if (topology) fu_topology_delete(topology);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;

    /** Places the shards one sharded scenario drives: two RAM shards on the heap domain. */
    static status_t place_shards(indexes_type_& indexes) noexcept {
        if (status_t const placed = indexes.emplace(no_memory_domain_k); failed(placed)) return placed;
        return indexes.emplace(no_memory_domain_k);
    }
};

/** Places the shards one sharded scenario drives, the way its residency's executor does over
 *  @p placement. */
template <typename indexes_type_, typename... placement_types_>
status_t place_shards(indexes_type_& indexes, placement_types_&... placement) noexcept {
    return dense_indexes_executor<indexes_type_>::place_shards(indexes, placement...);
}

/** What the one-row verbs walk with; @c status reports its construction. Defined per
 *  residency. */
template <typename indexes_type_>
struct dense_indexes_row_driver;

/** The RAM form: a caller-built context over its own arena, sized for the shard geometry. */
template <ram_collection indexes_type_>
struct dense_indexes_row_driver<indexes_type_> {
    dense_arena_context<indexes_type_> arena;
    status_t status;

    dense_indexes_row_driver(indexes_type_& indexes, dense_indexes_executor<indexes_type_>&) noexcept
        : arena(indexes), status(arena.status) {}
    typename indexes_type_::traversal_context_t& handle() noexcept { return arena.context; }
};

/** Every row of @p view widened to @c f32, row-major - exact for every stored scalar kind. */
template <typename scalar_type_>
std::vector<f32_t> dense_widened_rows(nk::matrix_view<scalar_type_> view, nk_dtype_t dtype) {
    std::size_t const rows = view.extent(0), columns = view.extent(1);
    std::vector<f32_t> widened(rows * columns);
    for (std::size_t row = 0; row != rows; ++row) {
        [[maybe_unused]] nk_status_t const cast = elementwise_chain<nk_cap_any_k>::kit_t::cast(
            view.byte_data() + row * view.stride_bytes(0), dtype, widened.data() + row * columns, nk_f32_k, columns,
            nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    }
    return widened;
}

/** An exact dot beside the magnitude a float accumulation's rounding scales with. */
struct dense_oracle_dot_t {

    /** The dot, exact in @c f64 for every widened pair of rows. */
    f64_t dot;

    /** The sum of absolute products - what bounds a float accumulation's error. */
    f64_t magnitude;
};

/** The exact dot of two widened rows. */
template <typename widened_type_>
dense_oracle_dot_t dense_widened_dot(widened_type_ const* first, widened_type_ const* second,
                                     std::size_t columns) noexcept {
    dense_oracle_dot_t oracle {0, 0};
    for (std::size_t column = 0; column != columns; ++column) {
        f64_t const product = static_cast<f64_t>(first[column]) * second[column];
        oracle.dot += product;
        oracle.magnitude += std::fabs(product);
    }
    return oracle;
}

/** Oracle dots of every query row against every stored row, row-major. */
inline std::vector<dense_oracle_dot_t> dense_oracle_dots(std::vector<f32_t> const& widened_queries,
                                                         std::vector<f32_t> const& widened_rows, std::size_t columns) {
    std::size_t const queries_count = widened_queries.size() / columns, rows_count = widened_rows.size() / columns;
    std::vector<dense_oracle_dot_t> dots(queries_count * rows_count);
    for (std::size_t query = 0; query != queries_count; ++query)
        for (std::size_t row = 0; row != rows_count; ++row)
            dots[query * rows_count + row] = dense_widened_dot(widened_queries.data() + query * columns,
                                                               widened_rows.data() + row * columns, columns);
    return dots;
}

/** The @c f32 widening of @p query, as one row. */
template <typename scalar_type_>
std::vector<f32_t> dense_widened_query(nk::vector_view<scalar_type_> query) {
    return dense_widened_rows(
        nk::matrix_view<scalar_type_> {reinterpret_cast<scalar_type_ const*>(query.byte_data()), 1, query.size()},
        scalar_type_::dtype());
}

#pragma endregion Shared Helpers

#pragma region Foundations

/** Guards: pre-reserve misuse and geometry freezing, on a fresh index per scenario. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_guards(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    auto vectors = make_vectors<scalar_t>(3, dense_dimensions_for<scalar_t>(dense_tiny_values_k),
                                          stream_key(seed, "guards"));

    auto const [insert_result, insert_status] = index.insert(vectors.view(), {}, context);
    failures += expect(insert_status == status_t::uninitialized_k);
    auto const [search_result, search_status] = index.search(vectors.view().row(0).as_vector(), context);
    failures += expect(search_status == status_t::success_k && search_result.count == 0);

    // Geometries no row can take are refused before the freeze, so the valid one below still wins it.
    auto [dimensionless_capacity, dimensionless_status] = index.reserve({64, connectivity_16_k, 0});
    failures += expect(dimensionless_status == status_t::config_mismatch_k);
    auto [uneven_capacity, uneven_status] = index.reserve( //
        {64, static_cast<connectivity_t>(24), dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
    failures += expect(uneven_status == status_t::config_mismatch_k);

    auto [capacity, reserve_status] = index.reserve( //
        {64, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
    failures += expect(reserve_status == status_t::success_k && capacity >= 64);
    failures += expect_eq(capacity, index.capacity());
    auto [mismatched_capacity, mismatched_status] = index.reserve( //
        {64, connectivity_32_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
    failures += expect(mismatched_status == status_t::config_mismatch_k);
    auto [matching_capacity, matching_status] = index.reserve( //
        {32, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)});
    failures += expect(matching_status == status_t::success_k);
    return failures;
}

/** @ref dense_expect_guards, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_guards(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_guards<index_type_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** The classic smoke scenario: three vectors in, each finds itself first, and a narrower
 *  query is refused. The dot kernel maximizes similarity instead of minimizing distance, so
 *  self-nearest is only asserted for the symmetric kernels. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_three_vectors(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    auto vectors = make_vectors<scalar_t>(3, dense_dimensions_for<scalar_t>(dense_tiny_values_k),
                                          stream_key(seed, "three_vectors"));

    failures += expect(
        index.reserve({8, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_tiny_values_k)}).status ==
        status_t::success_k);
    auto assigned_ids = nk::vector<node_id_t>::zeros(3).value;
    auto const [insert_result, insert_status] = index.insert(vectors.view(), assigned_ids.span(), context);
    failures += expect(insert_status == status_t::success_k);
    failures += expect_eq(insert_result.new_size, 3u);
    failures += expect_eq(index.size(), 3u);

    for (std::size_t row = 0; row != 3; ++row) {
        auto const [search_result, search_status] = index.search(vectors.view().row(row).as_vector(), context);
        failures += expect(search_status == status_t::success_k && search_result.count >= 1);
        if constexpr (kernel_ != nk_kernel_dot_k) {
            scored_node_t const nearest = context.results[0];
            failures += expect_eq(nearest.node_id, assigned_ids[row]);
            failures += expect(nearest.distance < dense_self_match_tolerance_k && //
                               nearest.distance > -dense_self_match_tolerance_k);
        }

        // A query narrower than the frozen width is refused, never read past its end.
        auto narrower = make_vectors<scalar_t>(1, dense_dimensions_for<scalar_t>(dense_tiny_values_k / 2),
                                               stream_key(seed, "three_vectors"));
        failures += expect_eq(index.search(narrower.view().row(0).as_vector(), context).status,
                              status_t::config_mismatch_k);
    }
    return failures;
}

/** @ref dense_expect_three_vectors, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_three_vectors(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_three_vectors<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Growth across several doubling chunks, then sampled self-recall on the big graph,
 *  with telemetry accounting sanity checks along the way. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_growth_and_recall(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                           u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                          stream_key(seed, "growth"));

    failures += expect(index.reserve({16, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
                       status_t::success_k);
    auto assigned_ids = nk::vector<node_id_t>::zeros(dense_growth_nodes_k).value;
    auto const [result, status] = index.insert(vectors.view(), assigned_ids.span(), context);
    failures += expect(status == status_t::success_k);
    failures += expect_eq(result.new_size, dense_growth_nodes_k);
    failures += expect_eq(index.size(), dense_growth_nodes_k);
    failures += expect(result.computed_distances > 0 && result.visited_members > 0);
    failures += expect(result.computed_distances_in_refines <= result.computed_distances);
    failures += expect(result.computed_distances_in_reverse_refines <= result.computed_distances);
    // The context was fresh, so the batch delta must equal the absolute counter - the
    // belt against a dropped snapshot or accumulation in the restarts telemetry.
    if constexpr (requires { context.restarts; }) failures += expect_eq(result.restarts, context.restarts);

    if constexpr (kernel_ != nk_kernel_dot_k)
        failures += dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k, 32);
    // A single query lands its whole expansion, past any inline landing.
    typename index_type_::search_config_t wide {};
    wide.expansion_factor = 200;
    auto const [wide_found, wide_status] = index.search(vectors.view().row(0).as_vector(), context, wide);
    failures += expect(succeeded(wide_status) && wide_found.count == 200 && context.results.size() == 200);
    failures += dense_expect_ascending(context);
    return failures;
}

/** @ref dense_expect_growth_and_recall, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_growth_and_recall(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_growth_and_recall<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Foundations

#pragma region Search

/**
 *  @brief The predicated search contract, audited against an exact oracle.
 *
 *  The oracle scans every stored vector through the same free kernels the index picked -
 *  @c picked_quads_capability_k - so its distances are bit-identical to the traversal's and
 *  the checks can be exact where approximation permits: a trivially-true predicate must
 *  reproduce @c search verbatim; every reported ID must satisfy its predicate and carry the
 *  oracle's distance for that ID, bit-for-bit; the per-node and batched callable shapes must
 *  agree; excluded self-matches must never surface; recall of the true filtered nearest is
 *  held to the same bar as the unfiltered suite across three selectivity decades - every
 *  other node, one in 16, and one in 64; and an impossible predicate drains legally.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_predicated_search(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                           u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr nk_capability_t caps_k = index_type_::picked_quads_capability_k;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                          stream_key(seed, "predicated"));

    failures += expect(
        index.reserve({dense_growth_nodes_k, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)})
            .status == status_t::success_k);
    auto assigned_ids = nk::vector<node_id_t>::zeros(dense_growth_nodes_k).value;
    failures += expect(index.insert(vectors.view(), assigned_ids.span(), context).status == status_t::success_k);

    // The oracle: distances of one query against the whole store, through the very 4-way
    // kernels the traversal dispatches. A wrong ID or vector diverges by orders of
    // magnitude, which is what this audit exists to catch.
    auto oracle_distance = [&](std::size_t query_row, node_id_t node_id) noexcept {
        auto query_view = vectors.view().row(query_row).as_vector();
        auto node_view = vectors.view().row(node_id).as_vector();
        auto node_norms = norms<caps_k, scalar_t>(node_view);
        auto lane_distances = distances<caps_k, scalar_t>(kernel_, query_view, norms<caps_k, scalar_t>(query_view),
                                                          {node_view, node_view, node_view, node_view},
                                                          {node_norms, node_norms, node_norms, node_norms});
        return lane_distances[0];
    };
    // The stochastic landmark seeding rides the context's persistent generator, so exact
    // walk parity is only defined between FRESH contexts with identical generator state.
    std::size_t const query_row = dense_growth_nodes_k / 2;
    auto query = vectors.view().row(query_row).as_vector();
    typename index_type_::traversal_context_t unfiltered_context {{}};
    auto const [unfiltered_result, unfiltered_status] = index.search(query, unfiltered_context);
    failures += expect(unfiltered_status == status_t::success_k && unfiltered_result.count > 0);
    scored_node_t const* reference = unfiltered_context.results.data();

    // The trivially-true predicate walks and reports exactly like the unfiltered sibling.
    auto all_pass = [](node_id_t) noexcept { return admission_t::admit_k; };
    typename index_type_::traversal_context_t all_pass_context {{}};
    auto const [with_true, with_true_status] = index.search_if(query, all_pass, all_pass_context);
    failures += expect(with_true_status == status_t::success_k);
    failures += expect_eq(with_true.count, unfiltered_result.count);
    for (std::size_t match = 0; match != with_true.count && match != unfiltered_result.count; ++match) {
        failures += expect_eq(all_pass_context.results[match].node_id, reference[match].node_id);
        failures += expect(all_pass_context.results[match].distance == reference[match].distance);
    }

    // The batched shape must agree with the per-node shape on the same walk.
    auto even_only = [](node_id_t node_id) noexcept {
        return node_id % 2 == 0 ? admission_t::admit_k : admission_t::reject_k;
    };
    auto even_batched = [](std::span<scored_node_t const> scored, auto& admitted) noexcept {
        for (scored_node_t const candidate : scored)
            if (candidate.node_id % 2 == 0) {
                [[maybe_unused]] fu::status_t const kept = admitted.push_back(candidate);
            }
    };
    typename index_type_::traversal_context_t scalar_context {{}}, batched_context {{}};
    auto const [with_scalar, with_scalar_status] = index.search_if(query, even_only, scalar_context);
    auto const [with_batched, with_batched_status] = index.search_if(query, even_batched, batched_context);
    failures += expect(with_scalar_status == status_t::success_k && with_scalar.count > 0);
    failures += expect(with_batched_status == status_t::success_k);
    failures += expect_eq(with_batched.count, with_scalar.count);
    for (std::size_t match = 0; match != with_batched.count && match != with_scalar.count; ++match)
        failures += expect_eq(batched_context.results[match].node_id, scalar_context.results[match].node_id);

    // Three selectivity decades over sampled queries: every reported ID satisfies its
    // predicate and carries the oracle's distance bit-for-bit; distances ascend; a query
    // whose own ID is excluded never surfaces itself. Even samples query a matching row -
    // the filtered analog of the suite's self-recall bar, scored per decade on the two
    // denser ones; odd samples query an excluded row, auditing admission purity. The
    // sparsest decade is exactness-audited only: recall there belongs to the future
    // cardinality-driven brute-force crossover, not the graph walk.
    std::size_t const moduli[] = {2, 16, 64};
    std::size_t hits = 0, recall_samples = 0;
    for (std::size_t modulus : moduli) {
        auto matches = [modulus](node_id_t node_id) noexcept {
            return node_id % modulus == 0 ? admission_t::admit_k : admission_t::reject_k;
        };
        for (std::size_t sample = 0; sample != 8; ++sample) {
            std::size_t sampled_row = ((sample * 641 + modulus) % dense_growth_nodes_k) / modulus * modulus;
            if (sample % 2) sampled_row += 1; // An excluded neighbor of a matching row.
            typename index_type_::traversal_context_t sampled_context {{}};
            auto const [sampled_result, sampled_status] = index.search_if(vectors.view().row(sampled_row).as_vector(),
                                                                          matches, sampled_context);
            bool const self_matches = sampled_row % modulus == 0;
            // The sparsest decade may legally drain empty on excluded-row queries: until
            // the admitted pool fills, the navigation radius bounds the walk, and no match
            // needs to fall inside it. A matching self is always near enough to admit.
            failures += expect(sampled_status == status_t::success_k);
            failures += expect(sampled_result.count > 0 || (modulus == 64 && !self_matches));
            scored_node_t const* results = sampled_context.results.data();
            bool self_found = false;
            for (std::size_t match = 0; match != sampled_result.count; ++match) {
                failures += expect(results[match].node_id % modulus == 0);
                distance_t const oracle = oracle_distance(sampled_row, results[match].node_id);
                failures += expect_near(results[match].distance, oracle);
                if (match) failures += expect(results[match].distance >= results[match - 1].distance);
                if (!self_matches) failures += expect(results[match].node_id != assigned_ids[sampled_row]);
                self_found |= results[match].node_id == assigned_ids[sampled_row];
            }
            if constexpr (kernel_ != nk_kernel_dot_k)
                if (self_matches && modulus != 64) hits += self_found, ++recall_samples;
        }
    }
    if constexpr (kernel_ != nk_kernel_dot_k)
        failures += expect(hits * 100 >= recall_samples * dense_minimal_recall_percent_k);

    // An impossible predicate drains legally: empty result set, successful status.
    auto none_pass = [](node_id_t) noexcept { return admission_t::reject_k; };
    typename index_type_::traversal_context_t none_context {{}};
    auto const [with_none, with_none_status] = index.search_if(query, none_pass, none_context);
    failures += expect(with_none_status == status_t::success_k);
    failures += expect_eq(with_none.count, std::size_t {0});
    return failures;
}

/** @ref dense_expect_predicated_search, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_predicated_search(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_predicated_search<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Single-index scans against the one-to-one oracle of @ref test_dense_indexes_scan: over a store
 *  spanning two chunks, each query ranks its true nearest at their own distances, a predicate of
 *  either shape narrows the ranking to the nodes it admits, and a breadth wider than the store
 *  returns every row. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_scan(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using vector_view_t = typename index_type_::vector_view_t;
    using search_config_t = typename index_type_::search_config_t;
    using small_size_t = typename index_type_::small_size_t;
    constexpr nk_capability_t caps_k = index_type_::picked_quads_capability_k;
    constexpr std::size_t rows_count = dense_growth_nodes_k, wanted = 10, wide = rows_count + 5;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "scan"));
    failures += expect(succeeded(index.reserve({rows_count, connectivity_16_k, dimensions}).status));
    auto assigned_ids = nk::vector<node_id_t>::zeros(rows_count).value;
    failures += expect(succeeded(index.insert(vectors.view(), assigned_ids.span(), context).status));

    auto all = [](node_id_t) noexcept { return true; };
    auto even = [](node_id_t node) noexcept { return node % 2 == 0; };
    auto admit_even = [&](node_id_t node) noexcept {
        return even(node) ? admission_t::admit_k : admission_t::reject_k;
    };
    auto admit_even_batched = [&](std::span<scored_node_t const> scored, auto& admitted) noexcept {
        for (scored_node_t const candidate : scored)
            if (even(candidate.node_id)) { [[maybe_unused]] fu::status_t const kept = admitted.push_back(candidate); }
    };
    // The ranking left in `context.results` against the oracle's, over the nodes `admits` admits.
    auto expect_ranking = [&](vector_view_t query, std::size_t breadth, auto admits) {
        auto const query_norms = norms<caps_k, scalar_t>(query);
        auto oracle = [&](node_id_t node) noexcept {
            vector_view_t const row = index.vector(node);
            return distance<caps_k, scalar_t>(kernel_, query, query_norms, row, norms<caps_k, scalar_t>(row));
        };
        std::vector<distance_t> admitted;
        for (node_id_t node = 0; node != rows_count; ++node)
            if (admits(node)) admitted.push_back(oracle(node));
        std::sort(admitted.begin(), admitted.end());
        std::size_t const kept = min_of(breadth, admitted.size());
        std::size_t ranking_failures = expect_eq(context.results.size(), kept);
        for (std::size_t rank = 0; rank != min_of(context.results.size(), kept); ++rank) {
            scored_node_t const match = context.results[rank];
            ranking_failures += expect(admits(match.node_id));
            ranking_failures += expect_near(match.distance, admitted[rank]);
            ranking_failures += expect_near(match.distance, oracle(match.node_id));
        }
        return ranking_failures;
    };

    search_config_t const scan {.expansion_factor = static_cast<small_size_t>(wanted),
                                .traversal = traversal_t::scan_k};
    for (std::size_t sample = 0; sample != 4; ++sample) {
        vector_view_t const query = vectors.view().row(sample * (rows_count / 4) + 1).as_vector();
        auto const [found, status] = index.search(query, context, scan);
        failures += expect(succeeded(status));
        failures += expect_eq(found.count, context.results.size());
        failures += expect_eq(found.computed_distances, rows_count);
        failures += expect_ranking(query, wanted, all);
        failures += expect(succeeded(index.search_if(query, admit_even, context, scan).status));
        failures += expect_ranking(query, wanted, even);
        failures += expect(succeeded(index.search_if(query, admit_even_batched, context, scan).status));
        failures += expect_ranking(query, wanted, even);
    }
    search_config_t const wide_scan {.expansion_factor = static_cast<small_size_t>(wide),
                                     .traversal = traversal_t::scan_k};
    vector_view_t const query = vectors.view().row(0).as_vector();
    failures += expect(succeeded(index.search(query, context, wide_scan).status));
    failures += expect_ranking(query, wide, all);
    return failures;
}

/** @ref dense_expect_scan, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_scan(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_scan<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** A float dot against its oracle, within accumulation rounding. */
template <typename dot_result_type_>
std::size_t dense_expect_dot(dot_result_type_ found, dense_oracle_dot_t oracle) {
    return expect(std::fabs(static_cast<f64_t>(found) - oracle.dot) <= 1e-4 * oracle.magnitude + 1e-6);
}

/** The @p picks are the exact top `picks.size()` rows of @p keys by ⟨query, key⟩, as an exhaustive walk lands
 *  them. */
template <typename scalar_type_, typename row_type_>
std::size_t dense_expect_top_logits(nk::matrix_view<scalar_type_> keys, nk::vector_view<scalar_type_> query,
                                    std::span<row_type_ const> picks) {
    std::size_t const rows_count = keys.extent(0);
    std::size_t failures = expect(!picks.empty() && picks.size() <= rows_count);
    if (failures) return failures;
    std::vector<dense_oracle_dot_t> const dots = dense_oracle_dots(
        dense_widened_query(query), dense_widened_rows(keys, scalar_type_::dtype()), keys.extent(1));
    std::vector<f64_t> ranked(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) ranked[row] = dots[row].dot;
    std::sort(ranked.rbegin(), ranked.rend());
    f64_t const threshold = ranked[picks.size() - 1];

    std::vector<bool> seen(rows_count);
    for (row_type_ const pick : picks) {
        std::size_t const row = static_cast<std::size_t>(pick);
        failures += expect(row < rows_count && !seen[row]);
        if (row >= rows_count || seen[row]) continue;
        seen[row] = true;
        failures += expect(dots[row].dot + 1e-4 * dots[row].magnitude + 1e-6 >= threshold);
    }
    return failures;
}

/** What every indexer test feeds its index: zero-centered heads under signed weights, so the ReLU clamps some keys. */
template <typename scalar_type_>
struct dense_indexer_inputs {
    static constexpr std::size_t heads_count_k = 3;

    std::size_t dimensions = 0;

    /** The stored rows widened to @c f32, row-major. */
    std::vector<f32_t> widened;

    /** Every query's heads widened, then as stored, @b [queries·heads,dimensions]. */
    std::vector<f32_t> heads_widened;
    std::vector<scalar_type_> heads_stored;

    /** Every query's head weights, @b [queries,heads]. */
    std::vector<f32_t> weights;

    /** The indexer heads of every query and their weights. */
    indexer_heads<scalar_type_> heads() const noexcept {
        return {{heads_stored.data(), heads_stored.size() / dimensions, dimensions},
                {weights.data(), weights.size() / heads_count_k, heads_count_k}};
    }
};

/** Builds @ref dense_indexer_inputs over @p vectors for @p queries_count queries. */
template <typename scalar_type_>
dense_indexer_inputs<scalar_type_> dense_make_indexer_inputs(nk::matrix_view<scalar_type_> vectors,
                                                             std::size_t queries_count) {
    using inputs_t = dense_indexer_inputs<scalar_type_>;
    inputs_t inputs;
    inputs.dimensions = vectors.extent(1);
    inputs.widened = dense_widened_rows(vectors, scalar_type_::dtype());

    // Every head is a stored row shifted from {0..7} to {-4..3}, exact in every floating kind.
    std::size_t const heads_rows = queries_count * inputs_t::heads_count_k;
    inputs.heads_widened.assign(inputs.widened.begin() + inputs.dimensions,
                                inputs.widened.begin() + (heads_rows + 1) * inputs.dimensions);
    for (f32_t& value : inputs.heads_widened) value -= 4;
    inputs.heads_stored.resize(inputs.heads_widened.size());
    [[maybe_unused]] nk_status_t const narrowed = elementwise_chain<nk_cap_any_k>::kit_t::cast(
        inputs.heads_widened.data(), nk_f32_k, inputs.heads_stored.data(), scalar_type_::dtype(),
        inputs.heads_stored.size(), nullptr);
    assert(narrowed == nk_success_k && "The serial tier casts every dtype pair");
    for (std::size_t query = 0; query != queries_count; ++query)
        inputs.weights.insert(inputs.weights.end(), {0.75f, -0.5f, 0.25f});
    return inputs;
}

/** Σₕ wₕ·ReLU(⟨qₕ, k⟩) of stored row @p row against the heads of query @p query, in @c f64. */
template <typename scalar_type_>
dense_oracle_dot_t dense_indexer_score(dense_indexer_inputs<scalar_type_> const& inputs, std::size_t query,
                                       std::size_t row) noexcept {
    using inputs_t = dense_indexer_inputs<scalar_type_>;
    dense_oracle_dot_t score {0, 0};
    for (std::size_t head = 0; head != inputs_t::heads_count_k; ++head) {
        std::size_t const head_row = query * inputs_t::heads_count_k + head;
        f64_t const weight = inputs.weights[head_row];
        dense_oracle_dot_t const oracle = dense_widened_dot(inputs.heads_widened.data() + head_row * inputs.dimensions,
                                                            inputs.widened.data() + row * inputs.dimensions,
                                                            inputs.dimensions);
        score.dot += weight * max_of(oracle.dot, f64_t {});
        score.magnitude += std::fabs(weight) * oracle.magnitude;
    }
    return score;
}

/** Every returned distance is 1 − the f64 indexer score of its row, and the rows are the exact top `rows.size()`. */
template <typename scalar_type_, typename row_type_>
std::size_t dense_expect_indexer_matches(dense_indexer_inputs<scalar_type_> const& inputs, std::size_t query,
                                         nk::vector_view<row_type_> rows, nk::vector_view<distance_t> distances) {
    std::size_t const rows_count = inputs.widened.size() / inputs.dimensions;
    std::size_t failures = expect(!rows.empty() && rows.size() <= rows_count && distances.size() == rows.size());
    if (failures) return failures;
    std::vector<f64_t> ranked(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) ranked[row] = dense_indexer_score(inputs, query, row).dot;
    std::sort(ranked.rbegin(), ranked.rend());
    f64_t const threshold = ranked[rows.size() - 1];

    std::vector<bool> seen(rows_count);
    for (std::size_t rank = 0; rank != rows.size(); ++rank) {
        std::size_t const row = static_cast<std::size_t>(rows[rank]);
        failures += expect(row < rows_count && !seen[row]);
        if (row >= rows_count || seen[row]) continue;
        seen[row] = true;
        dense_oracle_dot_t const score = dense_indexer_score(inputs, query, row);
        failures += dense_expect_dot(1 - static_cast<f64_t>(distances[rank]), score);
        failures += expect(score.dot + 1e-4 * score.magnitude + 1e-6 >= threshold);
    }
    return failures;
}

/** Indexer search on one index under an exhaustive walk, every node addressing its own row. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_indexer_search(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const nodes_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexer"));
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    std::vector<node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);

    dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
    typename index_type_::search_config_t const exhaustive {
        static_cast<typename index_type_::small_size_t>(nodes_count)};
    auto node_ids = nk::vector<node_id_t>::zeros(wanted).value;
    auto distances = nk::vector<distance_t>::zeros(wanted).value;
    nk::vector_view<node_id_t> const ranked_ids = node_ids.view();
    nk::vector_view<distance_t> const ranked_distances = distances.view();
    for (std::size_t query = 0; query != queries_count; ++query) {
        auto const [searched, status] = index.search(vectors.view().row(query).as_vector(), inputs.heads().row(query),
                                                     context, exhaustive);
        failures += expect(status == status_t::success_k);
        std::size_t const kept = min_of(searched.count, wanted);
        for (std::size_t rank = 0; rank != kept; ++rank)
            node_ids[rank] = context.results[rank].node_id, distances[rank] = context.results[rank].distance;
        failures += dense_expect_indexer_matches(inputs, query, ranked_ids[nk::range(std::size_t {0}, kept)],
                                                 ranked_distances[nk::range(std::size_t {0}, kept)]);
    }
    return failures;
}

/** @ref dense_expect_indexer_search for the dot kernel over floating-point keys, then the ledger must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_indexer_search(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<typename index_type_::scalar_t>()) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_indexer_search<index_type_, kernel_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

/** Sharded indexer search under an exhaustive walk: keys equal their rows, so the merged top of every query is
 *  judged against the same @c f64 oracle whichever shard scored it. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_indexer_search(u64_t seed) {
    using scalar_t = typename indexes_type_::scalar_t;
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<scalar_t>()) return 0;
    else {
        using key_t = typename indexes_type_::key_t;
        constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;

        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes)));
        failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_indexer"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(row);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

        nk::matrix_view<scalar_t> const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
        typename indexes_type_::search_config_t const exhaustive {
            static_cast<typename indexes_type_::index_t::small_size_t>(rows_count)};
        auto found_keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
        auto found_distances = nk::matrix<distance_t>::zeros({queries_count, wanted}).value;
        std::array<std::size_t, queries_count> counts;
        auto const [searched, status] = indexes.search(queries, inputs.heads(), driver.executor, exhaustive,
                                                       found_keys.span(), found_distances.span(), std::span {counts});
        failures += expect(succeeded(status));
        nk::matrix_view<key_t> const ranked_keys = found_keys.view();
        nk::matrix_view<distance_t> const ranked_distances = found_distances.view();
        for (std::size_t query = 0; query != queries_count; ++query) {
            nk::range_t const kept = nk::range(std::size_t {0}, counts[query]);
            failures += dense_expect_indexer_matches(inputs, query, ranked_keys(query, kept, nk::slice).as_vector(),
                                                     ranked_distances(query, kept, nk::slice).as_vector());
        }
        return failures;
    }
}

/**
 *  @brief Sharded scans against a one-to-one oracle over every stored row.
 *
 *  Two shards hold more rows than one first chunk, and the queries outnumber one packed block, so
 *  the scan crosses both seams. Each query - a stored row - ranks its true nearest at their own
 *  distances, a NaN for a self-match included; a predicate narrows the same ranking to the keys it
 *  admits; and a request wider than the store returns every row. RAM and NVMe only: device scans
 *  are not wired yet.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_scan(u64_t seed) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    constexpr nk_capability_t caps_k = indexes_type_::index_t::picked_quads_capability_k;
    constexpr std::size_t rows_count = 2100, queries_count = 260, wanted = 10, wide_queries_count = 3;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_scan"));
    std::vector<key_t> keys(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(1 + row);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

    // Every query against every row through the one-to-one kernel the walk scores with.
    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    auto oracle = nk::matrix<distance_t>::zeros({queries_count, rows_count}).value;
    for (std::size_t query = 0; query != queries_count; ++query) {
        auto const query_vector = queries.row(query).as_vector();
        auto const query_norms = norms<caps_k, scalar_t>(query_vector);
        for (std::size_t row = 0; row != rows_count; ++row) {
            auto const row_vector = vectors.view().row(row).as_vector();
            oracle(query, row) = distance<caps_k, scalar_t>(kernel_, query_vector, query_norms, row_vector,
                                                            norms<caps_k, scalar_t>(row_vector));
        }
    }

    auto found_keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
    auto found_distances = nk::matrix<distance_t>::zeros({queries_count, wanted}).value;
    auto wide_keys = nk::matrix<key_t>::zeros({wide_queries_count, rows_count}).value;
    auto wide_distances = nk::matrix<distance_t>::zeros({wide_queries_count, rows_count}).value;
    std::vector<std::size_t> counts(queries_count);
    auto all = [](key_t) noexcept { return true; };
    auto even = [](key_t key) noexcept { return static_cast<std::size_t>(key) % 2 == 0; };
    auto admit_even = [&](key_t key) noexcept { return even(key) ? admission_t::admit_k : admission_t::reject_k; };
    // One query's found ranking against the oracle's, sorted over the keys `admits` lets through.
    auto expect_ranking = [&](nk::matrix_view<key_t> ranked_keys, nk::matrix_view<distance_t> ranked_distances,
                              std::size_t query, auto admits) {
        std::vector<distance_t> admitted;
        for (std::size_t row = 0; row != rows_count; ++row)
            if (admits(keys[row])) admitted.push_back(oracle(query, row));
        std::sort(admitted.begin(), admitted.end());
        std::size_t ranking_failures = expect_eq(counts[query], min_of(ranked_keys.extent(1), admitted.size()));
        for (std::size_t rank = 0; rank != min_of(counts[query], admitted.size()); ++rank) {
            key_t const key = ranked_keys(query, rank);
            distance_t const distance = ranked_distances(query, rank);
            ranking_failures += expect(admits(key));
            ranking_failures += expect_near(distance, admitted[rank]);
            ranking_failures += expect_near(distance, oracle(query, static_cast<std::size_t>(key) - 1));
        }
        return ranking_failures;
    };

    typename indexes_type_::search_config_t const scan {.traversal = traversal_t::scan_k};
    auto const [found, status] = indexes.search(queries, driver.executor, scan, found_keys.span(),
                                                found_distances.span(), std::span {counts});
    failures += expect(succeeded(status));
    failures += expect_eq(found.computed_distances, queries_count * rows_count);
    for (std::size_t query = 0; query != queries_count; ++query)
        failures += expect_ranking(found_keys.view(), found_distances.view(), query, all);

    auto const filtered_status = indexes
                                     .search_if(queries, driver.executor, scan, found_keys.span(),
                                                found_distances.span(), std::span {counts}, admit_even)
                                     .status;
    failures += expect(succeeded(filtered_status));
    for (std::size_t query = 0; query != queries_count; ++query)
        failures += expect_ranking(found_keys.view(), found_distances.view(), query, even);

    auto const wide_queries = vectors(nk::range(std::size_t {0}, wide_queries_count), nk::slice);
    auto const wide_status = indexes
                                 .search(wide_queries, driver.executor, scan, wide_keys.span(), wide_distances.span(),
                                         std::span {counts}.first(wide_queries_count))
                                 .status;
    failures += expect(succeeded(wide_status));
    for (std::size_t query = 0; query != wide_queries_count; ++query)
        failures += expect_ranking(wide_keys.view(), wide_distances.view(), query, all);
    return failures;
}

#pragma endregion Search

#pragma region Snapshots

/** The bytes of a host matrix's rows - a positional target over its own storage. */
template <typename value_type_>
bytes_span_t dense_matrix_bytes(nk::matrix<value_type_>& matrix) noexcept {
    return {reinterpret_cast<byte_t*>(matrix.data()), dense_rows_bytes(matrix.view(), matrix.extent(0))};
}

/** The bytes of a matrix view's rows. */
template <typename value_type_>
bytes_view_t dense_matrix_bytes(nk::matrix_view<value_type_> view) noexcept {
    return {reinterpret_cast<byte_t const*>(view.byte_data()), dense_rows_bytes(view, view.extent(0))};
}

/** The bytes of a writable matrix span's rows - what a row-range slice hands back. */
template <typename value_type_>
bytes_span_t dense_matrix_bytes(nk::matrix_span<value_type_> span) noexcept {
    return {reinterpret_cast<byte_t*>(span.data()), dense_rows_bytes(span, span.extent(0))};
}

/** Saves every row of @p index into @p columns in one call, on the stream of the cell
 *  @p make_index where the index takes one. */
template <typename index_type_, typename make_index_type_>
status_t dense_save(index_type_& index, make_index_type_ const& make_index, typename index_type_::columns_t columns,
                    synchronization_t synchronization, nk::matrix_span<typename index_type_::node_id_t> scratch) {
    if constexpr (requires { index.save(columns, synchronization, scratch, make_index.stream); })
        return index.save(columns, synchronization, scratch, make_index.stream);
    else return index.save(columns, synchronization, scratch);
}

/** Appends @p rows rows of @p columns to @p index in one call, on the stream of the cell
 *  @p make_index where the index takes one. */
template <typename index_type_, typename make_index_type_>
status_t dense_load(index_type_& index, make_index_type_ const& make_index,
                    typename index_type_::columns_view_t columns, std::size_t rows,
                    iota_view<typename index_type_::node_id_t> source_range) {
    if constexpr (requires { index.load(columns, rows, source_range, make_index.stream); })
        return index.load(columns, rows, source_range, make_index.stream);
    else return index.load(columns, rows, source_range);
}

/** Rows a plan spans - the bound of its last step. */
template <typename plan_type_>
std::size_t dense_plan_rows(plan_type_ const& plan) noexcept {
    auto const steps = plan.steps();
    return steps.size() ? static_cast<std::size_t>(steps[steps.size() - 1].index_rows.bound) : 0;
}

/** Widest chunk of a plan - what one neighbors scratch must hold. */
template <typename plan_type_>
std::size_t dense_plan_widest_step(plan_type_ const& plan) noexcept {
    std::size_t widest = 0;
    for (auto const step : plan.steps()) widest = std::max(widest, step.index_rows.size());
    return widest;
}

/** Any number of lanes may take chunks from one plan: a shared cursor hands them out,
 *  each lane rebases through its own scratch, and the bytes match the serial save. */
template <typename index_type_, typename scalar_type_, typename node_id_type_>
std::size_t dense_expect_pooled_save(index_type_& index, typename index_type_::saved_shape_t shape,
                                     nk::matrix<scalar_type_>& saved_vectors,
                                     nk::matrix<node_id_type_>& saved_neighbors) {
    std::size_t failures = 0;
    auto pooled_vectors = nk::matrix<scalar_type_>::zeros({shape.nodes, shape.dimensions}).value;
    auto pooled_neighbors = nk::matrix<node_id_type_>::zeros({shape.nodes, shape.connectivity}).value;
    failures += expect(!pooled_vectors.empty() && !pooled_neighbors.empty());
    auto [plan, status] = index.plan_save({dense_matrix_bytes(pooled_vectors), dense_matrix_bytes(pooled_neighbors)},
                                          synchronization_t::immutable_k);
    failures += expect(succeeded(status));
    std::size_t const widest = dense_plan_widest_step(plan);
    std::atomic<std::size_t> cursor {0};
    std::atomic<std::size_t> lane_failures {0};
    constexpr std::size_t lanes_count = 3;
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(lanes_count))) return failures + expect(false);
    pool.for_threads([&](std::size_t) noexcept {
        auto scratch = nk::matrix<node_id_type_>::zeros({widest, shape.connectivity}).value;
        if (scratch.empty()) return void(lane_failures.fetch_add(1));
        auto const steps = plan.steps();
        for (std::size_t position = cursor.fetch_add(1); position < steps.size(); position = cursor.fetch_add(1))
            if (failed(plan.save(steps[position], scratch.span()))) lane_failures.fetch_add(1);
    });
    failures += expect_eq(lane_failures.load(), 0u);
    failures += expect_eq(plan.finish(), status_t::success_k);
    failures += expect(std::memcmp(pooled_vectors.data(), saved_vectors.data(),
                                   dense_rows_bytes(saved_vectors.view(), shape.nodes)) == 0);
    failures += expect(std::memcmp(pooled_neighbors.data(), saved_neighbors.data(),
                                   dense_rows_bytes(saved_neighbors.view(), shape.nodes)) == 0);
    return failures;
}

/** Round-trips the graph through one plan per direction: build, save into two host
 *  matrices chunk by chunk, load into a fresh reserved index the same way, and expect the
 *  restored graph to answer self-queries just as the original did - norms re-derive on
 *  load and are never serialized. On RAM a pool of lanes then takes chunks from one plan
 *  by a shared cursor and must land the same bytes. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_save_load_roundtrip(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                             u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                          stream_key(seed, "roundtrip"));

    failures += expect(
        index.reserve({dense_growth_nodes_k, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)})
            .status == status_t::success_k);
    auto assigned_ids = nk::vector<node_id_t>::zeros(dense_growth_nodes_k).value;
    failures += expect(index.insert(vectors.view(), assigned_ids.span(), context).status == status_t::success_k);

    auto const shape = index.saved_shape();
    auto saved_vectors = nk::matrix<scalar_t>::zeros({shape.nodes, shape.dimensions}).value;
    auto saved_neighbors = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
    auto scratch = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
    failures += expect(!saved_vectors.empty() && !saved_neighbors.empty() && !scratch.empty());
    failures += expect_eq(
        dense_save(index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                   synchronization_t::exclusive_k, scratch.span()),
        status_t::success_k);

    std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
    index_type_& restored = *restored_ptr;
    failures += expect(
        restored.reserve({shape.nodes, static_cast<connectivity_t>(shape.connectivity), shape.dimensions}).status ==
        status_t::success_k);
    failures += expect_eq(
        dense_load(restored, make_index,
                   {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())}, shape.nodes,
                   {0, static_cast<node_id_t>(shape.nodes)}),
        status_t::success_k);
    failures += expect_eq(restored.size(), shape.nodes);

    // A restored index must re-save on every contract: the settlement counters arrive
    // fully released, so the optimistic arm's probe sees a quiescent chain, and the
    // re-saved bytes mirror the originals exactly.
    auto resaved_vectors = nk::matrix<scalar_t>::zeros({shape.nodes, shape.dimensions}).value;
    auto resaved_neighbors = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
    failures += expect(!resaved_vectors.empty() && !resaved_neighbors.empty());
    typename index_type_::columns_t const resaved {dense_matrix_bytes(resaved_vectors),
                                                   dense_matrix_bytes(resaved_neighbors)};
    failures += expect_eq(dense_save(restored, make_index, resaved, synchronization_t::exclusive_k, scratch.span()),
                          status_t::success_k);
    failures += expect_eq(dense_save(restored, make_index, resaved, synchronization_t::immutable_k, scratch.span()),
                          status_t::success_k);
    // A device index refuses the concurrent arm by contract; the RAM index settles it.
    failures += expect_eq(dense_save(restored, make_index, resaved, synchronization_t::concurrent_k, scratch.span()),
                          dense_device_resident<index_type_> ? status_t::config_mismatch_k : status_t::success_k);
    failures += expect(std::memcmp(resaved_vectors.data(), saved_vectors.data(),
                                   dense_rows_bytes(saved_vectors.view(), shape.nodes)) == 0);
    failures += expect(std::memcmp(resaved_neighbors.data(), saved_neighbors.data(),
                                   dense_rows_bytes(saved_neighbors.view(), shape.nodes)) == 0);

    // Lanes take chunks from one plan only where the residency admits the concurrent arm.
    if constexpr (!dense_device_resident<index_type_>)
        failures += dense_expect_pooled_save(restored, shape, saved_vectors, saved_neighbors);

    if constexpr (kernel_ != nk_kernel_dot_k)
        failures += dense_expect_self_recall(restored, make_index, vectors.view(), dense_growth_nodes_k, 32);
    return failures;
}

/** @ref dense_expect_save_load_roundtrip, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_save_load_roundtrip(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_save_load_roundtrip<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Columns one row short of the rows asked for are refused at planning, so no step ever
 *  reads past them and nothing is claimed. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
std::size_t dense_expect_clipped_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                      nk::matrix<node_id_type_>& saved_neighbors,
                                      typename index_type_::columns_view_t source, std::size_t nodes_count,
                                      std::size_t dimensions, iota_view<node_id_type_> local) {
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const clipped_ptr = make_index(ledger);
    index_type_& clipped = *clipped_ptr;
    failures += expect(clipped.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    std::size_t const neighbors_row_bytes = dense_rows_bytes(saved_neighbors.view(), 1);
    typename index_type_::columns_view_t const short_source {
        source.vectors, source.neighbors.first(source.neighbors.size() - neighbors_row_bytes)};
    failures += expect_eq(clipped.plan_load(short_source, nodes_count, local).status, status_t::config_mismatch_k);
    failures += expect_eq(clipped.load(short_source, nodes_count, local), status_t::config_mismatch_k);
    failures += expect_eq(clipped.size(), 0u);
    return failures;
}

/** A plan finished with nothing landed refuses and publishes nothing, and so does a step
 *  past its last. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
std::size_t dense_expect_partial_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                      typename index_type_::columns_view_t source, std::size_t nodes_count,
                                      std::size_t dimensions, iota_view<node_id_type_> local) {
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const failing_ptr = make_index(ledger);
    index_type_& failing = *failing_ptr;
    failures += expect(failing.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto [plan, status] = failing.plan_load(source, nodes_count, local);
    failures += expect(succeeded(status));
    auto const steps = plan.steps();
    failures += expect(steps.size() >= 2);
    failures += expect_eq(plan.finish(), status_t::config_mismatch_k);
    failures += expect_eq(failing.size(), 0u);
    failures += expect_eq(plan.load(steps[0]), status_t::success_k);
    failures += expect_eq(plan.load(steps[steps.size()]), status_t::config_mismatch_k);
    failures += expect_eq(plan.finish(), status_t::config_mismatch_k);
    failures += expect_eq(failing.size(), 0u);

    // A refused finish spends nothing: the missing steps land and the same plan publishes once.
    for (std::size_t position = 1; position < steps.size(); ++position)
        failures += expect_eq(plan.load(steps[position]), status_t::success_k);
    failures += expect_eq(plan.finish(), status_t::success_k);
    failures += expect_eq(failing.size(), nodes_count);
    failures += expect_eq(plan.finish(), status_t::uninitialized_k); // Spent: no second publish.
    failures += expect_eq(failing.size(), nodes_count);
    return failures;
}

/** A moved-from load plan refuses every verb, while the plan moved into keeps the steps. */
template <typename index_type_, typename node_id_type_>
std::size_t dense_expect_moved_plan_refuses(index_type_& index, typename index_type_::columns_view_t source,
                                            std::size_t nodes_count, iota_view<node_id_type_> local) {
    std::size_t failures = 0;
    auto [plan, status] = index.plan_load(source, nodes_count, local);
    failures += expect(succeeded(status));
    auto moved = std::move(plan);
    failures += expect_eq(plan.steps().size(), 0u);
    failures += expect_eq(plan.finish(), status_t::uninitialized_k); // Moved from: refuses.
    failures += expect(moved.steps().size() >= 1);
    return failures;
}

/** A load plan that dies before its finish releases the rows it claimed, once even when
 *  moved: the next plan appends at the same base, in the rows the reservation hosts, and lands
 *  the whole checkpoint. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
std::size_t dense_expect_abandoned_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                        typename index_type_::columns_view_t source, std::size_t nodes_count,
                                        std::size_t dimensions, iota_view<node_id_type_> local) {
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const abandoned_ptr = make_index(ledger);
    index_type_& abandoned = *abandoned_ptr;
    auto const [reserved_rows, reserve_status] = abandoned.reserve({nodes_count, connectivity_16_k, dimensions});
    failures += expect(reserve_status == status_t::success_k);
    failures += dense_expect_moved_plan_refuses(abandoned, source, nodes_count, local);
    failures += expect_eq(abandoned.size(), 0u);
    failures += expect_eq(abandoned.load(source, nodes_count, local), status_t::success_k);
    failures += expect_eq(abandoned.size(), nodes_count);
    failures += expect_eq(abandoned.capacity(), reserved_rows); // The released rows hosted the load.
    return failures;
}

/** A save plan refuses a step past its watermark, and a finish with nothing landed. */
template <typename index_type_, typename scalar_type_, typename node_id_type_>
std::size_t dense_expect_save_plan_misuse(index_type_& index, nk::matrix<scalar_type_>& saved_vectors,
                                          nk::matrix<node_id_type_>& saved_neighbors,
                                          nk::matrix<node_id_type_>& scratch) {
    std::size_t failures = 0;
    auto [plan, status] = index.plan_save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                                          synchronization_t::exclusive_k);
    failures += expect(succeeded(status));
    auto const steps = plan.steps();
    failures += expect_eq(plan.save(steps[steps.size()], scratch.span()), status_t::config_mismatch_k);
    failures += expect_eq(plan.finish(), status_t::config_mismatch_k);
    return failures;
}

/** Hostile load submissions: an unfrozen target refuses before any row lands, a source
 *  failing on a later chunk publishes nothing, a misused plan refuses, a count past the
 *  reservation grows by design, and an out-of-range neighbor ID rides the sentinel
 *  convention - stored, but never surfaced by any search. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_load_hostile(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    // Enough rows for the walk to span two storage chunks, so a second run exists to fail on.
    std::size_t const nodes_count = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "load_hostile"));
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect(index.insert(vectors.view(), {}, context).status == status_t::success_k);

    auto saved_vectors = nk::matrix<scalar_t>::zeros({nodes_count, dimensions}).value;
    auto saved_neighbors = nk::matrix<node_id_t>::zeros({nodes_count, connectivity_16_k}).value;
    auto scratch = nk::matrix<node_id_t>::zeros({nodes_count, connectivity_16_k}).value;
    failures += expect(!saved_vectors.empty() && !saved_neighbors.empty() && !scratch.empty());
    failures += expect_eq(index.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                                     synchronization_t::exclusive_k, scratch.span()),
                          status_t::success_k);

    typename index_type_::columns_view_t const source {dense_matrix_bytes(saved_vectors.view()),
                                                       dense_matrix_bytes(saved_neighbors.view())};
    iota_view<node_id_t> const local {0, static_cast<node_id_t>(nodes_count)};

    // An unfrozen target has no geometry to validate against.
    std::unique_ptr<index_type_> const unfrozen_ptr = make_index(ledger);
    index_type_& unfrozen = *unfrozen_ptr;
    failures += expect_eq(unfrozen.load(source, nodes_count, local), status_t::config_mismatch_k);
    failures += expect_eq(unfrozen.size(), 0u);

    failures += dense_expect_clipped_load<index_type_>(make_index, ledger, saved_neighbors, source, nodes_count,
                                                       dimensions, local);
    failures += dense_expect_partial_load<index_type_>(make_index, ledger, source, nodes_count, dimensions, local);
    failures += dense_expect_abandoned_load<index_type_>(make_index, ledger, source, nodes_count, dimensions, local);
    failures += dense_expect_save_plan_misuse(index, saved_vectors, saved_neighbors, scratch);

    // A count past the reservation grows by design - the chunk directory follows.
    std::unique_ptr<index_type_> const growing_ptr = make_index(ledger);
    index_type_& growing = *growing_ptr;
    failures += expect(growing.reserve({1, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect_eq(growing.load(source, nodes_count, local), status_t::success_k);
    failures += expect_eq(growing.size(), nodes_count);
    // A reservation one row above the hosted ones grows them, though that one row fits the free rows.
    std::size_t const hosted_rows = growing.capacity();
    failures += expect(growing.reserve({hosted_rows + 1, connectivity_16_k, dimensions}).value > hosted_rows);

    // Out-of-range neighbor IDs load as sentinels - the interchange admits any - and the
    // translation frontier keeps them from ever surfacing in a search.
    auto patched_neighbors = nk::matrix<node_id_t>::zeros({nodes_count, connectivity_16_k}).value;
    failures += expect(!patched_neighbors.empty());
    std::size_t const row_ids = dense_rows_bytes(saved_neighbors.view(), 1) / sizeof(node_id_t);
    std::memcpy(patched_neighbors.data(), saved_neighbors.data(),
                dense_rows_bytes(saved_neighbors.view(), nodes_count));
    node_id_t* const patched_ids = patched_neighbors.data();
    patched_ids[5 * row_ids] = static_cast<node_id_t>(nodes_count + 5);
    patched_ids[7 * row_ids + 1] = node_id_unknown_k;
    std::unique_ptr<index_type_> const patched_ptr = make_index(ledger);
    index_type_& patched = *patched_ptr;
    failures += expect(patched.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect_eq(
        patched.load({source.vectors, dense_matrix_bytes(patched_neighbors.view())}, nodes_count, local),
        status_t::success_k);
    typename index_type_::traversal_context_t patched_context {{}};
    for (std::size_t sample = 0; sample != 16; ++sample) {
        auto const [result, status] = patched.search(vectors.view().row((sample * 19) % nodes_count).as_vector(),
                                                     patched_context);
        failures += expect(status == status_t::success_k);
        for (std::size_t match = 0; match != result.count; ++match)
            failures += expect(static_cast<std::size_t>(patched_context.results[match].node_id) < patched.size());
    }
    return failures;
}

/** @ref dense_expect_load_hostile, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_load_hostile(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_load_hostile<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Loads append: an ID space past the source's rows lands nothing, the same payload landed twice
 *  doubles the index, the second copy's IDs shift by the first copy's rows with sentinels intact,
 *  and a self-query lands on either copy. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_load_append(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    std::size_t const nodes_count = 300;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "load_hostile", 1));
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect(index.insert(vectors.view(), {}, context).status == status_t::success_k);

    auto saved_vectors = nk::matrix<scalar_t>::zeros({nodes_count, dimensions}).value;
    auto saved_neighbors = nk::matrix<node_id_t>::zeros({nodes_count, connectivity_16_k}).value;
    auto scratch = nk::matrix<node_id_t>::zeros({2 * nodes_count, connectivity_16_k}).value;
    failures += expect(!saved_vectors.empty() && !saved_neighbors.empty() && !scratch.empty());
    failures += expect_eq(
        dense_save(index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                   synchronization_t::exclusive_k, scratch.span()),
        status_t::success_k);

    typename index_type_::columns_view_t const source {dense_matrix_bytes(saved_vectors.view()),
                                                       dense_matrix_bytes(saved_neighbors.view())};
    iota_view<node_id_t> const local {0, static_cast<node_id_t>(nodes_count)};
    std::unique_ptr<index_type_> const doubled_ptr = make_index(ledger);
    index_type_& doubled = *doubled_ptr;
    failures += expect(doubled.reserve({2 * nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    // An ID space past the source's rows would rebase IDs beyond the landed ones, so nothing lands.
    iota_view<node_id_t> const overreaching {0, static_cast<node_id_t>(nodes_count + 1)};
    failures += expect_eq(dense_load(doubled, make_index, source, nodes_count, overreaching),
                          status_t::config_mismatch_k);
    failures += expect_eq(doubled.size(), 0u);
    failures += expect_eq(dense_load(doubled, make_index, source, nodes_count, local), status_t::success_k);
    failures += expect_eq(doubled.size(), nodes_count);
    failures += expect_eq(dense_load(doubled, make_index, source, nodes_count, local), status_t::success_k);
    failures += expect_eq(doubled.size(), 2 * nodes_count);

    auto doubled_vectors = nk::matrix<scalar_t>::zeros({2 * nodes_count, dimensions}).value;
    auto doubled_neighbors = nk::matrix<node_id_t>::zeros({2 * nodes_count, connectivity_16_k}).value;
    failures += expect(!doubled_vectors.empty() && !doubled_neighbors.empty());
    failures += expect_eq(
        dense_save(doubled, make_index, {dense_matrix_bytes(doubled_vectors), dense_matrix_bytes(doubled_neighbors)},
                   synchronization_t::exclusive_k, scratch.span()),
        status_t::success_k);

    // The first copy lands verbatim; the second copy's vectors match and its IDs shift.
    failures += expect(std::memcmp(doubled_vectors.data(), saved_vectors.data(),
                                   dense_rows_bytes(saved_vectors.view(), nodes_count)) == 0);
    failures += expect(std::memcmp(doubled_neighbors.data(), saved_neighbors.data(),
                                   dense_rows_bytes(saved_neighbors.view(), nodes_count)) == 0);
    failures += expect(std::memcmp(doubled_vectors.view().row(nodes_count).byte_data(), saved_vectors.data(),
                                   dense_rows_bytes(saved_vectors.view(), nodes_count)) == 0);
    std::size_t const row_ids = dense_rows_bytes(saved_neighbors.view(), 1) / sizeof(node_id_t);
    node_id_t const* const saved_ids = saved_neighbors.data();
    node_id_t const* const doubled_ids = doubled_neighbors.data();
    std::size_t unshifted = 0;
    for (std::size_t row = 0; row != nodes_count; ++row)
        for (std::size_t slot = 0; slot != row_ids; ++slot) {
            node_id_t const first = saved_ids[row * row_ids + slot];
            node_id_t const second = doubled_ids[(nodes_count + row) * row_ids + slot];
            node_id_t const shifted = first == node_id_unknown_k ? node_id_unknown_k
                                                                 : static_cast<node_id_t>(first + nodes_count);
            unshifted += second != shifted;
        }
    failures += expect_eq(unshifted, 0u);

    // A self-query lands on one of the two copies with a self-match distance.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        typename index_type_::traversal_context_t doubled_context = dense_make_context<index_type_>(make_index);
        for (std::size_t sample = 0; sample != 16; ++sample) {
            std::size_t const row = (sample * 19) % nodes_count;
            auto const [result, status] = doubled.search(vectors.view().row(row).as_vector(), doubled_context);
            failures += expect(status == status_t::success_k && result.count > 0);
            if (!result.count) continue;
            scored_node_t const best = doubled_context.results[0];
            failures += expect(best.node_id == row || best.node_id == row + nodes_count);
            failures += expect(best.distance < dense_self_match_tolerance_k &&
                               best.distance > -dense_self_match_tolerance_k);
        }
    }
    return failures;
}

/** @ref dense_expect_load_append, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_load_append(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_load_append<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Snapshots

#pragma region Refinement

/** Sweeps one audit to convergence: the bitsets restart, then every step of every level
 *  until no level expands a node - the serial spelling of what an executor fans out. The
 *  restart is what lets a repaired graph be re-audited; without it the expanded bits still
 *  stand and no fresh edge is ever followed. */
template <typename plan_type_>
void dense_sweep_reachability(plan_type_& plan) {
    plan.restart_sweep();
    do
        for (auto const step : plan.steps()) plan.sweep(step);
    while (plan.advance_level());
}

/** One shard's reachability, whichever way its residency spells the audit: the device
 *  answers in one call, the host drives the plan the caller owns. */
template <typename index_type_>
auto dense_audit_health(index_type_& index, typename index_type_::traversal_context_t& context) {
    if constexpr (requires { index.health(context); }) return index.health(context);
    else {
        auto [plan, status] = index.plan_health(context);
        if (failed(status)) return expected<typename index_type_::health_result_t> {{}, status};
        dense_sweep_reachability(plan);
        return plan.finish();
    }
}

/**
 *  @brief The settling contract: a graph built under a narrow pool has work for the first sweep,
 *      the second finds far less, and recall never falls for the trouble.
 *
 *  The bound on the second sweep is a ratio rather than zero, since the walks re-seed from random
 *  landmarks and a later back-link can still disturb a row a earlier step already settled.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_refine_settles(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                          stream_key(seed, "refine"));
    failures += expect(index.reserve({16, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
                       status_t::success_k);

    // Narrow enough that the first sweep has real work to do - the recipe the verb exists for.
    typename index_type_::insert_config_t cheap;
    cheap.expansion_factor = 4;
    failures += expect(index.insert(vectors.view(), {}, context, cheap).status == status_t::success_k);
    failures += expect_eq(index.size(), dense_growth_nodes_k);

    node_id_t const published = static_cast<node_id_t>(index.size());
    std::size_t const hits_before = kernel_ != nk_kernel_dot_k
                                        ? dense_count_self_hits(index, make_index, vectors.view(), dense_growth_nodes_k,
                                                                32)
                                        : 0;
    auto const [first_refinement, first_status] = index.refine({0, published}, context);
    failures += expect(first_status == status_t::success_k);
    // Every pass heals what it strands, so islands add their re-selections to the range's, and only then.
    failures += expect(first_refinement.visited_nodes >= dense_growth_nodes_k &&
                       (first_refinement.visited_nodes == dense_growth_nodes_k) == !first_refinement.islands_found);
    failures += expect(first_refinement.changed_nodes > 0 &&
                       first_refinement.changed_nodes <= first_refinement.visited_nodes);
    failures += expect(first_refinement.computed_distances > 0 && first_refinement.visited_members > 0);
    if constexpr (requires { first_refinement.computed_distances_in_refines; })
        failures += expect(first_refinement.computed_distances_in_refines +
                               first_refinement.computed_distances_in_reverse_refines <=
                           first_refinement.computed_distances);

    auto const [second_refinement, second_status] = index.refine({0, published}, context);
    failures += expect(second_status == status_t::success_k);
    failures += expect(second_refinement.visited_nodes >= dense_growth_nodes_k &&
                       (second_refinement.visited_nodes == dense_growth_nodes_k) == !second_refinement.islands_found);
    failures += expect(second_refinement.changed_nodes < first_refinement.changed_nodes);

    // A range past the end re-selects only islands, and one straddling it stops at the watermark.
    auto const [beyond, beyond_status] = index.refine({published, published + 16}, context);
    failures += expect(beyond_status == status_t::success_k);
    failures += expect((beyond.visited_nodes == 0) == !beyond.islands_found);
    auto const [straddling, straddling_status] = index.refine({published / 2, published * 2}, context);
    failures += expect(straddling_status == status_t::success_k);
    std::size_t const straddled = dense_growth_nodes_k - dense_growth_nodes_k / 2;
    failures += expect(straddling.visited_nodes >= straddled &&
                       (straddling.visited_nodes == straddled) == !straddling.islands_found);

    // A sweep is a build step, so what it must never do is cost the graph its answers.
    if constexpr (kernel_ != nk_kernel_dot_k)
        failures += expect(dense_count_self_hits(index, make_index, vectors.view(), dense_growth_nodes_k, 32) >=
                           hits_before);
    return failures;
}

/** @ref dense_expect_refine_settles, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_refine_settles(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_refine_settles<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** The audit's contract on a graph nothing stranded: an empty index reaches nothing, a
 *  healthy one reaches everything, and two audits of one graph agree - the landmarks are
 *  stride-spread, so the answer is a property of the graph rather than of the draw. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_health_audits(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    failures += expect(index.reserve({16, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
                       status_t::success_k);

    auto const [empty_health, empty_status] = dense_audit_health(index, context);
    failures += expect(succeeded(empty_status));
    failures += expect_eq(empty_health.reachable_nodes, 0u);

    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k),
                                          stream_key(seed, "health"));
    failures += expect(index.insert(vectors.view(), {}, context).status == status_t::success_k);

    // An insertion wires each node against the graph as it stood, so a few can end up with no
    // in-edge any search follows; reaching every one of them is what the repair is for, and
    // what the healing scenario asserts. Here the claim is that the audit sees the bulk, which
    // a dot graph's hubs never promise.
    auto const [health, audit_status] = dense_audit_health(index, context);
    failures += expect(succeeded(audit_status));
    failures += expect(health.reachable_nodes <= index.size());
    if constexpr (kernel_ != nk_kernel_dot_k)
        failures += expect(health.reachable_nodes * 100 >= index.size() * dense_minimal_recall_percent_k);
    failures += expect(health.sweeps >= 1);

    auto const [again_health, again_status] = dense_audit_health(index, context);
    failures += expect(succeeded(again_status));
    failures += expect_eq(again_health.reachable_nodes, health.reachable_nodes);
    failures += expect_eq(again_health.sweeps, health.sweeps);
    return failures;
}

/** @ref dense_expect_health_audits, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_health_audits(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_health_audits<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Strands the rows from @p first_island on: every reference to the tail leaves the graph,
 *  and each stranded row keeps only the published-empty self-loop, pointed at by nothing. */
template <typename node_id_type_>
void dense_strand_islands(nk::matrix_span<node_id_type_> neighbors, node_id_type_ first_island) {
    for (std::size_t node = 0; node != first_island; ++node) {
        auto mainland_row = neighbors.row(node).as_vector();
        for (std::size_t slot = 0; slot != mainland_row.size(); ++slot)
            if (mainland_row[slot] >= first_island && mainland_row[slot] != node_id_unknown_k)
                mainland_row[slot] = node_id_unknown_k;
    }
    for (std::size_t node = first_island; node != neighbors.extent(0); ++node) {
        auto island_row = neighbors.row(node).as_vector();
        island_row[0] = static_cast<node_id_type_>(node);
        for (std::size_t slot = 1; slot != island_row.size(); ++slot) island_row[slot] = node_id_unknown_k;
    }
}

/**
 *  @brief The repair's contract, against islands planted by construction rather than coaxed out
 *      of a build: a checkpoint's neighbor rows are rewritten so a tail of nodes points only at
 *      itself, which no walk can enter, and the load's own compaction keeps that encoding.
 *
 *  Healing must then reach every node, and a second pass must find nothing left to do.
 */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_heal_islands(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    constexpr std::size_t nodes_count = 512, islands_count = 32;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "heal"));

    // Build every node, then strand the tail: its rows become self-loops, the "published with
    // no neighbors" encoding, so nothing points into them and they point nowhere.
    std::unique_ptr<index_type_> const built_ptr = make_index(ledger);
    index_type_& built = *built_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    failures += expect(built.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect(built.insert(vectors.view(), {}, context).status == status_t::success_k);

    auto saved_vectors = nk::matrix<scalar_t>::zeros({nodes_count, dimensions}).value;
    auto saved_neighbors =
        nk::matrix<node_id_t>::zeros({nodes_count, static_cast<std::size_t>(built.connectivity())}).value;
    auto scratch = nk::matrix<node_id_t>::zeros({nodes_count, static_cast<std::size_t>(built.connectivity())}).value;
    failures += expect(!saved_vectors.empty() && !saved_neighbors.empty() && !scratch.empty());
    if (saved_vectors.empty() || saved_neighbors.empty() || scratch.empty()) return failures;

    failures += expect_eq(
        dense_save(built, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                   synchronization_t::exclusive_k, scratch.span()),
        status_t::success_k);
    failures += expect_eq(built.size(), nodes_count);
    dense_strand_islands(saved_neighbors.span(), static_cast<node_id_t>(nodes_count - islands_count));

    std::unique_ptr<index_type_> const islanded_ptr = make_index(ledger);
    index_type_& islanded = *islanded_ptr;
    failures += expect(islanded.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect_eq(
        dense_load(islanded, make_index,
                   {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())}, nodes_count,
                   {0, static_cast<node_id_t>(nodes_count)}),
        status_t::success_k);

    // The landmarks are spread across the whole ID space, so a planted island that happens
    // to be one is reachable by a walk starting there - what must hold is that the rest
    // are not, and that the repair closes every one of them.
    auto const [before_health, before_status] = dense_audit_health(islanded, context);
    failures += expect(succeeded(before_status));
    failures += expect(before_health.reachable_nodes < nodes_count);

    auto const [repair, repair_status] = islanded.refine({}, context);
    failures += expect(succeeded(repair_status));
    failures += expect_eq(repair.reachable_nodes, nodes_count);
    failures += expect(repair.islands_found > 0);

    auto const [after_health, after_status] = dense_audit_health(islanded, context);
    failures += expect(succeeded(after_status));
    failures += expect_eq(after_health.reachable_nodes, nodes_count);

    auto const [settled_repair, settled_status] = islanded.refine({}, context);
    failures += expect(succeeded(settled_status));
    failures += expect_eq(settled_repair.islands_found, 0u);
    failures += expect_eq(settled_repair.islands_reconnected, 0u);
    return failures;
}

/** @ref dense_expect_heal_islands, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_heal_islands(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_heal_islands<index_type_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Refinement

#pragma region Algebra

/** A logit scale spreading the oracle's dots over @c [-4,4], so no scored row's weight underflows. */
inline f32_t dense_attend_scale(std::vector<dense_oracle_dot_t> const& dots) noexcept {
    f64_t widest = 0;
    for (dense_oracle_dot_t const oracle : dots) widest = std::fmax(widest, std::fabs(oracle.dot));
    return widest > 0 ? static_cast<f32_t>(4 / widest) : 1;
}

/** Every head's attended row and carry equal exact softmax attention over every key, in f64. Rows of
 *  @p queries, @p outputs and @p carry are tasks t · @p head_count + h; @p value_of maps a key row to its value row. */
template <typename scalar_type_, typename value_of_type_>
std::size_t dense_expect_full_attention(nk::matrix_view<scalar_type_> keys, nk::matrix_view<scalar_type_> queries,
                                        std::size_t head_count, value_of_type_&& value_of, f32_t score_scale,
                                        nk::matrix_view<f32_t> outputs, std::span<softmax_state_t const> carry) {
    std::size_t const rows_count = keys.extent(0), columns = keys.extent(1), tasks = queries.extent(0);
    std::size_t failures = expect(rows_count != 0 && head_count != 0 && tasks % head_count == 0);
    failures += expect(outputs.extent(0) == tasks && outputs.extent(1) == columns && carry.size() == tasks);
    if (failures) return failures;

    std::vector<f32_t> widened_values(rows_count * columns);
    for (std::size_t row = 0; row != rows_count; ++row) {
        nk::vector_view<scalar_type_> const value = value_of(row);
        if ((failures += expect(value.size() == columns))) return failures;
        std::vector<f32_t> const widened = dense_widened_query(value);
        std::copy(widened.begin(), widened.end(), widened_values.begin() + row * columns);
    }
    std::vector<dense_oracle_dot_t> const dots = dense_oracle_dots(
        dense_widened_rows(queries, scalar_type_::dtype()), dense_widened_rows(keys, scalar_type_::dtype()), columns);
    std::vector<f64_t> expected(columns);
    for (std::size_t task = 0; task != tasks; ++task) {
        f64_t maximum = std::numeric_limits<f64_t>::lowest(), partition = 0;
        for (std::size_t row = 0; row != rows_count; ++row)
            maximum = std::fmax(maximum, score_scale * dots[task * rows_count + row].dot);
        std::fill(expected.begin(), expected.end(), 0);
        for (std::size_t row = 0; row != rows_count; ++row) {
            f64_t const weight = std::exp(score_scale * dots[task * rows_count + row].dot - maximum);
            partition += weight;
            for (std::size_t column = 0; column != columns; ++column)
                expected[column] += weight * widened_values[row * columns + column];
        }
        for (std::size_t column = 0; column != columns; ++column)
            failures += expect_near(outputs(task, column), expected[column] / partition);
        failures += expect_near(carry[task].maximum_score, maximum);
        failures += expect_near(carry[task].partition, partition);
    }
    return failures;
}

/** What any walk's attention must hold, however narrow: every attended row is finite and inside the hull of
 *  the @p values rows, and every carry's partition counts its own maximum's key and at most one unit per key,
 *  that maximum no larger than the exact one over every key. */
template <typename scalar_type_>
std::size_t dense_expect_attention_bounds(nk::matrix_view<scalar_type_> keys, nk::matrix_view<scalar_type_> values,
                                          nk::matrix_view<scalar_type_> queries, f32_t score_scale,
                                          nk::matrix_view<f32_t> outputs, std::span<softmax_state_t const> carry) {
    std::size_t const rows_count = keys.extent(0), columns = keys.extent(1), tasks = queries.extent(0);
    std::size_t failures = expect(outputs.extent(0) == tasks && outputs.extent(1) == columns && carry.size() == tasks);
    if (failures) return failures;
    std::vector<f32_t> const widened = dense_widened_rows(values, scalar_type_::dtype());
    std::vector<f32_t> lowest(columns, std::numeric_limits<f32_t>::max()),
        highest(columns, std::numeric_limits<f32_t>::lowest());
    for (std::size_t row = 0; row != rows_count; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            lowest[column] = std::fmin(lowest[column], widened[row * columns + column]);
            highest[column] = std::fmax(highest[column], widened[row * columns + column]);
        }
    std::vector<dense_oracle_dot_t> const dots = dense_oracle_dots(
        dense_widened_rows(queries, scalar_type_::dtype()), dense_widened_rows(keys, scalar_type_::dtype()), columns);
    for (std::size_t task = 0; task != tasks; ++task) {
        f64_t maximum = std::numeric_limits<f64_t>::lowest();
        for (std::size_t row = 0; row != rows_count; ++row)
            maximum = std::fmax(maximum, score_scale * dots[task * rows_count + row].dot);
        softmax_state_t const state = carry[task];
        failures += expect(state.maximum_score <= maximum + 1e-3 * std::fmax(1, std::fabs(maximum)));
        failures += expect(state.partition >= 1 - 1e-3 && state.partition <= rows_count * (1 + 1e-3));
        for (std::size_t column = 0; column != columns; ++column) {
            f32_t const value = outputs(task, column);
            f32_t const slack = 1e-3f * std::fmax(1, std::fmax(std::fabs(lowest[column]), std::fabs(highest[column])));
            failures += expect(value >= lowest[column] - slack && value <= highest[column] + slack);
        }
    }
    return failures;
}

/** The dot a walk reports for @p left against @p right. Test rows hold small integers, so every kernel and
 *  accumulation order lands it exactly, and the serial kernel stands in for every residency's. */
template <typename scalar_type_>
typename scalar_type_::dot_result_t dense_native_dot(nk::vector_view<scalar_type_> left,
                                                     nk::vector_view<scalar_type_> right) noexcept {
    auto const right_norms = norms<nk_cap_serial_k, scalar_type_>(right);
    return scores<nk_cap_serial_k, scalar_type_>(nk_kernel_dot_k, left, norms<nk_cap_serial_k, scalar_type_>(left),
                                                 {right, right, right, right},
                                                 {right_norms, right_norms, right_norms, right_norms})
        .dots[0];
}

/** One row of graph products: every kept pair names a distinct row of @p vectors, through @p row_of, and
 *  carries its exact dot against @p left. */
template <std::size_t rows_count_, typename scalar_type_, typename id_type_, typename dot_type_, typename row_of_type_>
std::size_t dense_expect_product_row(nk::vector_view<scalar_type_> left, nk::matrix_view<scalar_type_> vectors,
                                     nk::vector_view<id_type_> ids, nk::vector_view<dot_type_> products,
                                     row_of_type_&& row_of) {
    std::size_t failures = expect_eq(ids.size(), products.size());
    std::array<bool, rows_count_> seen {};
    for (std::size_t pair = 0; pair != ids.size(); ++pair) {
        std::size_t const row = row_of(ids[pair]);
        failures += expect(row < rows_count_ && !seen[row]);
        if (row >= rows_count_ || seen[row]) continue;
        seen[row] = true;
        failures += expect(products[pair] == dense_native_dot(left, vectors.row(row).as_vector()));
    }
    return failures;
}

/** Graph products against the native dots: every scored pair is counted, a narrow row keeps its width
 *  while counting past it, every kept pair names a distinct stored row with its exact dot, and mismatched
 *  spans refuse. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_multiply(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    failures += expect(index.reserve({rows_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "multiply"));

    // An empty index scores nothing and writes nothing.
    auto node_ids = nk::vector<node_id_t>::full(rows_count, node_id_unknown_k).value;
    auto products = nk::vector<typename index_type_::dot_result_t>::zeros(rows_count).value;
    auto const empty = index.multiply(vectors.view().row(0).as_vector(), to_span(node_ids.span()),
                                      to_span(products.span()), context);
    failures += expect(empty.status == status_t::success_k && empty.value.count == 0);
    failures += expect(node_ids[0] == node_id_unknown_k);

    auto assigned_ids = nk::vector<node_id_t>::zeros(rows_count).value;
    failures += expect(index.insert(vectors.view(), assigned_ids.span(), context).status == status_t::success_k);

    typename index_type_::search_config_t const exhaustive {
        static_cast<typename index_type_::small_size_t>(rows_count)};
    nk::vector_view<node_id_t> const kept_ids = node_ids.view();
    nk::vector_view<typename index_type_::dot_result_t> const kept_products = products.view();
    for (std::size_t const wanted : {rows_count, std::size_t {4}})
        for (std::size_t row = 0; row != left_count; ++row) {
            auto const left = vectors.view().row(row).as_vector();
            nk::range_t const width = nk::range(std::size_t {0}, wanted);
            auto const [multiplied, status] = index.multiply(left, to_span(node_ids[width]), to_span(products[width]),
                                                             context, exhaustive);
            failures += expect(status == status_t::success_k);
            failures += expect_eq(multiplied.count, multiplied.computed_distances);
            failures += expect(multiplied.count > 0 && multiplied.count <= rows_count);
            if (wanted < rows_count) failures += expect(multiplied.count > wanted);
            nk::range_t const kept = nk::range(std::size_t {0}, min_of<std::size_t>(multiplied.count, wanted));
            failures += dense_expect_product_row<rows_count>(left, vectors.view(), kept_ids[kept], kept_products[kept],
                                                             [](node_id_t id) noexcept { return std::size_t {id}; });
        }

    auto const refused = index.multiply(vectors.view().row(0).as_vector(), to_span(node_ids.span()),
                                        to_span(products[nk::range(std::size_t {0}, std::size_t {3})]), context);
    failures += expect(refused.status == status_t::config_mismatch_k);
    return failures;
}

/** @ref dense_expect_multiply for the dot kernel, then the ledger it left must balance; no other
 *  kernel offers it. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_multiply(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_multiply<index_type_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

/** Attention against exact full attention in @c f64 under an exhaustive walk, one head per walk and four sharing
 *  one, every node ID its stored row; a narrow walk stays bounded; a node without a value row, a mismatched
 *  output and heads that do not divide the rows refuse. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_attend(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using vector_view_t = typename index_type_::vector_view_t;
    std::size_t failures = 0;
    std::size_t const nodes_count = dense_thread_nodes_k, rows_count = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend"));
    auto values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend", 1));
    auto queries = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "attend", 2));
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    auto value_of = [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); };
    auto lookup = [&](std::span<node_id_t const> nodes, std::span<vector_view_t> rows) noexcept {
        for (std::size_t lane = 0; lane != nodes.size(); ++lane) rows[lane] = value_of(nodes[lane]);
    };
    auto outputs = nk::matrix<f32_t>::full({rows_count, dimensions}, 1).value;
    std::vector<softmax_state_t> carry(rows_count);

    // An empty index folds nothing: zero rows and empty partitions.
    {
        auto const [attended, status] = index.attend(queries.view(), 1, lookup, 1, outputs.span(), std::span {carry},
                                                     context);
        failures += expect(status == status_t::success_k && attended.count == rows_count);
        nk::matrix_view<f32_t> const attended_rows = outputs.view();
        failures += expect(std::all_of(attended_rows.begin(), attended_rows.end(),
                                       [](auto const& entry) { return entry.second == 0; }));
        failures += expect(
            std::all_of(carry.begin(), carry.end(), [](softmax_state_t state) { return state.partition == 0; }));
    }

    std::vector<node_id_t> assigned_ids(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return failures + expect(false);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);

    f32_t const score_scale = dense_attend_scale(
        dense_oracle_dots(dense_widened_rows(queries.view(), index_type_::vector_dtype_k),
                          dense_widened_rows(vectors.view(), index_type_::vector_dtype_k), dimensions));
    typename index_type_::search_config_t const exhaustive {
        static_cast<typename index_type_::small_size_t>(nodes_count)};
    for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
        auto const [attended, status] = index.attend(queries.view(), head_count, lookup, score_scale, outputs.span(),
                                                     std::span {carry}, context, exhaustive);
        failures += expect(status == status_t::success_k);
        failures += expect_eq(attended.count, rows_count);
        failures += dense_expect_full_attention(vectors.view(), queries.view(), head_count, value_of, score_scale,
                                                outputs.view(), std::span<softmax_state_t const> {carry});
        failures += expect(
            index.attend(queries.view(), head_count, lookup, score_scale, outputs.span(), std::span {carry}, context)
                .status == status_t::success_k);
        failures += dense_expect_attention_bounds(vectors.view(), values.view(), queries.view(), score_scale,
                                                  outputs.view(), std::span<softmax_state_t const> {carry});
    }

    auto const row_less = [&](std::span<node_id_t const> nodes, std::span<vector_view_t> rows) noexcept {
        lookup(nodes, rows);
        for (std::size_t lane = 0; lane != nodes.size(); ++lane)
            if (nodes[lane] + 1 == nodes_count) rows[lane] = vector_view_t {};
    };
    failures += expect(
        index.attend(queries.view(), 1, row_less, score_scale, outputs.span(), std::span {carry}, context, exhaustive)
            .status == status_t::shape_mismatch_k);
    failures += expect(index
                           .attend(queries.view(), 1, lookup, score_scale,
                                   outputs(nk::range(std::size_t {0}, rows_count - 1), nk::slice), std::span {carry},
                                   context)
                           .status == status_t::config_mismatch_k);
    failures += expect(
        index.attend(queries.view(), 3, lookup, score_scale, outputs.span(), std::span {carry}, context).status ==
        status_t::shape_mismatch_k);
    return failures;
}

/** @ref dense_expect_attend for the dot kernel, then the ledger it left must balance; no other kernel offers it. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_attend(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        allocation_ledger_t ledger;
        std::size_t const failures = dense_expect_attend<index_type_, kernel_>(make_index, ledger, seed);
        return failures + expect_eq(ledger.live_bytes.load(), 0);
    }
}

/** Sharded graph products against the native dots: every scored pair is counted, a narrow row keeps its
 *  width while counting past it, every kept pair names a distinct stored row with its exact dot, and
 *  mismatched spans refuse. */
template <typename indexes_type_, typename... placement_types_>
std::size_t dense_expect_indexes_products(u64_t seed, placement_types_&... placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {nk_kernel_dot_k, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_products"));
    std::array<key_t, rows_count> keys;
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(1 + row);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

    auto const left = vectors(nk::range(std::size_t {0}, left_count), nk::slice);
    auto found_keys = nk::matrix<key_t>::zeros({left_count, rows_count}).value;
    auto found_products = nk::matrix<typename indexes_type_::dot_result_t>::zeros({left_count, rows_count}).value;
    std::array<std::size_t, left_count> counts;
    // A narrow call writes the leading columns, so its output rows stride past their width.
    auto multiply = [&](std::size_t wanted, std::span<std::size_t> counts_span) noexcept {
        nk::range_t const columns = nk::range(std::size_t {0}, wanted);
        return indexes.multiply(left, driver.executor, {}, found_keys(nk::all, columns, nk::slice),
                                found_products(nk::all, columns, nk::slice), counts_span);
    };
    nk::matrix_view<key_t> const kept_keys = found_keys.view();
    nk::matrix_view<typename indexes_type_::dot_result_t> const kept_products = found_products.view();
    for (std::size_t const wanted : {rows_count, std::size_t {4}}) {
        auto const [multiplied, status] = multiply(wanted, counts);
        failures += expect(succeeded(status));
        failures += expect_eq(multiplied.count, multiplied.computed_distances);
        for (std::size_t row = 0; row != left_count; ++row) {
            failures += expect(counts[row] > 0 && counts[row] <= rows_count);
            if (wanted < rows_count) failures += expect(counts[row] > wanted);
            nk::range_t const kept = nk::range(std::size_t {0}, min_of(counts[row], wanted));
            failures += dense_expect_product_row<rows_count>(
                left.row(row).as_vector(), vectors.view(), kept_keys(row, kept, nk::slice).as_vector(),
                kept_products(row, kept, nk::slice).as_vector(),
                [](key_t key) noexcept { return static_cast<std::size_t>(key) - 1; });
        }
    }

    failures += expect(multiply(rows_count, std::span {counts}.first(left_count - 1)).status ==
                       status_t::config_mismatch_k);
    return failures;
}

/** @ref dense_expect_indexes_products for the dot kernel; no other kernel offers graph products. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_products(u64_t seed, placement_types_&... placement) {
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else return dense_expect_indexes_products<indexes_type_>(seed, placement...);
}

/** Sharded MaxSim against the native dots: 64 documents of 4 token rows share their keys, a walk as wide as
 *  the index ranks exactly the oracle's best documents at their exact scores, a default one never scores a
 *  key above its own, and a ragged batch refuses. */
template <typename indexes_type_, typename... placement_types_>
std::size_t dense_expect_indexes_maxsim(u64_t seed, placement_types_&... placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using search_config_t = typename indexes_type_::search_config_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4;
    constexpr std::size_t documents_count = rows_count / tokens_per_document;
    constexpr std::size_t rows_per_query = 4, queries_count = 3, wanted = 5;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {nk_kernel_dot_k, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_maxsim"));
    std::array<key_t, rows_count> keys;
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(1 + row / tokens_per_document);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

    // Each document's exact score per query: every query row's best dot among its tokens, summed. Test rows
    // are non-negative, so every dot is too, and a row that misses a document only lowers its score.
    auto const queries = vectors(nk::range(std::size_t {0}, queries_count * rows_per_query), nk::slice);
    std::array<std::array<f64_t, documents_count>, queries_count> exact {};
    for (std::size_t row = 0; row != queries_count * rows_per_query; ++row)
        for (std::size_t document = 0; document != documents_count; ++document) {
            f64_t best = 0;
            for (std::size_t token = 0; token != tokens_per_document; ++token)
                best = max_of(best, static_cast<f64_t>(dense_native_dot(
                                        queries.row(row).as_vector(),
                                        vectors.view().row(document * tokens_per_document + token).as_vector())));
            exact[row / rows_per_query][document] += best;
        }

    auto found_keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
    auto found_scores = nk::matrix<f64_t>::zeros({queries_count, wanted}).value;
    std::array<std::size_t, queries_count> counts;
    auto maxsim = [&](nk::matrix_view<scalar_t> rows, search_config_t config) noexcept {
        return indexes.maxsim(rows, rows_per_query, driver.executor, config, found_keys.span(), found_scores.span(),
                              std::span {counts});
    };
    auto document_of = [&](std::size_t query, std::size_t rank) noexcept {
        return static_cast<std::size_t>(found_keys(query, rank)) - 1;
    };

    // A walk as wide as the index ranks every token, so the best documents land at their exact scores, up to the
    // rounding of a device that sums its dots in another order.
    search_config_t const exhaustive {static_cast<typename indexes_type_::index_t::small_size_t>(rows_count)};
    failures += expect(succeeded(maxsim(queries, exhaustive).status));
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::array<f64_t, documents_count> ranked = exact[query];
        std::sort(ranked.rbegin(), ranked.rend());
        failures += expect_eq(counts[query], wanted);
        for (std::size_t rank = 0; rank != counts[query]; ++rank) {
            std::size_t const document = document_of(query, rank);
            failures += expect(document < documents_count);
            if (document >= documents_count) continue;
            failures += expect_near(found_scores(query, rank), ranked[rank]) +
                        expect_near(found_scores(query, rank), exact[query][document]);
        }
    }

    // A default walk may miss a row's best token, so no document scores above its exact value.
    failures += expect(succeeded(maxsim(queries, {}).status));
    for (std::size_t query = 0; query != queries_count; ++query)
        for (std::size_t rank = 0; rank != counts[query]; ++rank) {
            std::size_t const document = document_of(query, rank);
            failures += expect(document < documents_count);
            if (document >= documents_count) continue;
            // Above its exact value only by rounding.
            if (found_scores(query, rank) > exact[query][document])
                failures += expect_near(found_scores(query, rank), exact[query][document]);
        }

    failures += expect(maxsim(queries(nk::range(std::size_t {0}, rows_per_query + 1), nk::slice), {}).status ==
                       status_t::shape_mismatch_k);
    return failures;
}

/** @ref dense_expect_indexes_maxsim for the dot kernel; no other kernel offers MaxSim. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_maxsim(u64_t seed, placement_types_&... placement) {
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else return dense_expect_indexes_maxsim<indexes_type_>(seed, placement...);
}

/** Attention across the shards against exact full attention in @c f64 under exhaustive walks, one head per walk
 *  and four sharing one, rows keyed by their own position: every shard's walk folds into one running state per
 *  head. Narrow walks stay bounded; a key without a value row, a mismatched output and heads that do not divide
 *  the rows refuse. Only the dot kernel attends. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_attend(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using vector_view_t = typename index_t::vector_view_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        std::size_t failures = 0;
        std::size_t const nodes_count = dense_thread_nodes_k, rows_count = 8;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes)));
        failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));

        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_attend"));
        auto values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_attend", 1));
        auto queries = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_attend", 2));
        std::vector<key_t> keys(nodes_count);
        for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(row);
        failures += expect(succeeded(
            indexes.insert(std::span<key_t const> {keys.data(), nodes_count}, vectors.view(), driver.executor).status));

        auto value_of = [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); };
        auto lookup = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
            for (std::size_t lane = 0; lane != batch.size(); ++lane)
                rows[lane] = value_of(static_cast<std::size_t>(batch[lane]));
        };
        f32_t const score_scale = dense_attend_scale(
            dense_oracle_dots(dense_widened_rows(queries.view(), index_t::vector_dtype_k),
                              dense_widened_rows(vectors.view(), index_t::vector_dtype_k), dimensions));
        auto outputs = nk::matrix<f32_t>::zeros({rows_count, dimensions}).value;
        std::vector<softmax_state_t> carry(rows_count);

        typename index_t::search_config_t const exhaustive {static_cast<typename index_t::small_size_t>(nodes_count)};
        for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
            auto const [attended, status] = indexes.attend(queries.view(), head_count, lookup, score_scale,
                                                           outputs.span(), std::span {carry}, driver.executor,
                                                           exhaustive);
            failures += expect(succeeded(status));
            failures += expect_eq(attended.count, rows_count);
            failures += dense_expect_full_attention(vectors.view(), queries.view(), head_count, value_of, score_scale,
                                                    outputs.view(), std::span<softmax_state_t const> {carry});
            failures += expect(succeeded(indexes
                                             .attend(queries.view(), head_count, lookup, score_scale, outputs.span(),
                                                     std::span {carry}, driver.executor)
                                             .status));
            failures += dense_expect_attention_bounds(vectors.view(), values.view(), queries.view(), score_scale,
                                                      outputs.view(), std::span<softmax_state_t const> {carry});
        }

        auto const row_less = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
            lookup(batch, rows);
            for (std::size_t lane = 0; lane != batch.size(); ++lane)
                if (static_cast<std::size_t>(batch[lane]) + 1 == nodes_count) rows[lane] = vector_view_t {};
        };
        failures += expect(indexes
                               .attend(queries.view(), 1, row_less, score_scale, outputs.span(), std::span {carry},
                                       driver.executor, exhaustive)
                               .status == status_t::shape_mismatch_k);
        failures += expect(indexes
                               .attend(queries.view(), 1, lookup, score_scale,
                                       outputs(nk::range(std::size_t {0}, rows_count - 1), nk::slice),
                                       std::span {carry}, driver.executor)
                               .status == status_t::config_mismatch_k);
        failures += expect(
            indexes.attend(queries.view(), 3, lookup, score_scale, outputs.span(), std::span {carry}, driver.executor)
                .status == status_t::shape_mismatch_k);
        return failures;
    }
}

#pragma endregion Algebra

#pragma region Races

/** Concurrent construction on ForkUnion's default flat pool: distinct slices per thread, one
 *  traversal context each, then ID-uniqueness and sampled self-recall checks. A device index admits
 *  one batch at a time and answers the overlapping ones with @c contended_k, which retry. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_concurrent_build(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const writer_threads_count = 4;
    std::size_t const nodes_count = writer_threads_count * dense_thread_nodes_k;

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(writer_threads_count))) return failures + expect(false);

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    failures += expect(
        index.reserve({nodes_count, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
        status_t::success_k);

    auto threads = dense_make_threads<index_type_>(make_index, writer_threads_count, dense_thread_nodes_k,
                                                   dense_dimensions_for<scalar_t>(dense_values_k),
                                                   stream_key(seed, "concurrent_build"));
    std::vector<node_id_t> assigned_ids(nodes_count);
    std::atomic<std::size_t> thread_failures {0};

    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= writer_threads_count) return;
        status_t status = status_t::contended_k;
        while (status == status_t::contended_k)
            status = index
                         .insert(threads.slices[thread].view(),
                                 {assigned_ids.data() + thread * dense_thread_nodes_k, dense_thread_nodes_k},
                                 threads.contexts[thread])
                         .status;
        if (status != status_t::success_k) thread_failures.fetch_add(1);
        assert_no_locks_held();
    });

    failures += expect_eq(thread_failures.load(), 0u);
    failures += expect_eq(index.size(), nodes_count);
    // One batch's IDs come from one dispense: contiguous, in the order of its rows.
    std::size_t scattered_ids = 0;
    for (std::size_t row = 0; row != nodes_count; ++row) {
        std::size_t const offset = row % dense_thread_nodes_k;
        scattered_ids += assigned_ids[row] != assigned_ids[row - offset] + offset;
    }
    failures += expect_eq(scattered_ids, 0u);
    failures += expect_contiguous_ids(std::span {assigned_ids}, 0);

    if constexpr (kernel_ != nk_kernel_dot_k)
        for (std::size_t thread = 0; thread != writer_threads_count; ++thread)
            failures += dense_expect_self_recall(index, make_index, threads.slices[thread].view(), dense_thread_nodes_k,
                                                 4);
    return failures;
}

/** @ref dense_expect_concurrent_build, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_concurrent_build(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_concurrent_build<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/**
 *  @brief A reader thread races three writer threads, asserting only interleaving-independent
 *      invariants.
 *
 *  Every search succeeds, respects the candidate-pool bound, returns distances in ascending order,
 *  and never surfaces an ID at or above the watermark it could have observed - the
 *  translation-frontier invariant, audited live.
 *
 *  The query vectors are never inserted: a search does not require its query to be stored, so
 *  the reader & writers share only the index itself, and any sanitizer report is an index bug
 *  rather than test-harness noise.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_concurrent_search(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                           u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const writer_threads_count = 3, queries_count = 64;
    std::size_t const nodes_count = writer_threads_count * dense_thread_nodes_k;
    std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(writer_threads_count + 1))) return failures + expect(false);

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    failures += expect(
        index.reserve({nodes_count, connectivity_16_k, dense_dimensions_for<scalar_t>(dense_values_k)}).status ==
        status_t::success_k);

    auto threads = dense_make_threads<index_type_>(make_index, writer_threads_count + 1, dense_thread_nodes_k,
                                                   dense_dimensions_for<scalar_t>(dense_values_k),
                                                   stream_key(seed, "concurrent_search"));
    auto query_vectors = make_vectors<scalar_t>(queries_count, dense_dimensions_for<scalar_t>(dense_values_k),
                                                stream_key(seed, "concurrent_search", 1));

    std::atomic<std::size_t> writer_lanes_remaining {writer_threads_count};
    std::atomic<std::size_t> thread_failures {0};

    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread != 0 && thread <= writer_threads_count) {
            status_t status = status_t::contended_k;
            while (status == status_t::contended_k)
                status = index.insert(threads.slices[thread].view(), {}, threads.contexts[thread]).status;
            if (status != status_t::success_k) thread_failures.fetch_add(1);
            writer_lanes_remaining.fetch_sub(1, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        if (thread != 0) return;

        // Every other step runs the predicated sibling under the same race: all the
        // unfiltered invariants must hold, plus admission purity - no ID the predicate
        // rejects may ever surface, no matter how the writers interleave.
        auto odd_only = [](node_id_t node_id) noexcept {
            return node_id % 2 == 1 ? admission_t::admit_k : admission_t::reject_k;
        };
        // The predicated sibling runs only where the backend exposes the verb - the GPU joins
        // this arm the moment a device-side search_if lands.
        constexpr bool predicable_k = requires(typename index_type_::vector_view_t query) {
            index.search_if(query, odd_only, threads.contexts[0]);
        };
        auto search_step = [&](typename index_type_::vector_view_t query, bool predicated) {
            if constexpr (predicable_k)
                if (predicated) return index.search_if(query, odd_only, threads.contexts[0]);
            return index.search(query, threads.contexts[0]);
        };
        for (std::size_t step = 0; step < dense_reader_minimal_searches_k || //
                                   writer_lanes_remaining.load(std::memory_order_acquire) != 0;
             ++step) {
            auto query = query_vectors.view().row(step % queries_count).as_vector();
            bool const predicated = predicable_k && step % 2 == 1;
            auto const [result, status] = search_step(query, predicated);
            if (status != status_t::success_k) thread_failures.fetch_add(1);
            thread_failures.fetch_add(
                scored_failures(threads.contexts[0].results.data(), result.count, index.size(), expansion_bound));
            // Every writer inserts its slice as one batch, which publishes whole.
            if (index.size() % dense_thread_nodes_k) thread_failures.fetch_add(1);
            if (predicated)
                for (std::size_t match = 0; match != result.count; ++match)
                    if (threads.contexts[0].results[match].node_id % 2 != 1) thread_failures.fetch_add(1);
        }
        assert_no_locks_held();
    });

    failures += expect_eq(thread_failures.load(), 0u);
    failures += expect_eq(index.size(), nodes_count);
    return failures;
}

/** @ref dense_expect_concurrent_search, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_concurrent_search(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_concurrent_search<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** The sharded index's refusal surface, all deterministic: verbs before placement,
 *  placement after reservation, geometry drift, mismatched spans, and a doubly-claimed
 *  executor - every misuse answers with a status, never with a write. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_guards(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 64;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "guards"));
    auto oversized = make_vectors<scalar_t>(rows + 1, dimensions, stream_key(seed, "guards", 1));
    std::vector<key_t> keys(rows + 1);
    for (std::size_t row = 0; row != rows + 1; ++row) keys[row] = static_cast<key_t>(1 + row);

    // Verbs before any shard exists refuse rather than write.
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect_eq(indexes
                              .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                      executor)
                              .status,
                          status_t::uninitialized_k);
    failures += expect_eq(indexes.reserve({rows, connectivity_16_k, dimensions}).status, status_t::uninitialized_k);

    // Reserving closes the shard set, and the frozen geometry refuses drift.
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));
    failures += expect_eq(place_shards(indexes, placement...), status_t::config_mismatch_k);
    failures += expect_eq(indexes.reserve({rows, connectivity_16_k, dimensions}).value, indexes.capacity());
    failures += expect_eq(indexes.reserve({rows, connectivity_32_k, dimensions}).status, status_t::config_mismatch_k);
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));

    // A keys span disagreeing with the batch's rows refuses before anything is taken.
    failures += expect_eq(indexes
                              .insert(std::span<key_t const> {keys.data(), rows - 1}, vectors.view(), //
                                      executor)
                              .status,
                          status_t::config_mismatch_k);

    // The executor admits one caller: a live claim turns the second away, and releasing
    // it restores service - which the closing insertion proves end to end.
    {
        auto [claim, status] = executor.claim();
        failures += expect(succeeded(status));
        failures += expect_eq(executor.claim().status, status_t::contended_k);
        failures += expect_eq(indexes
                                  .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                          executor)
                                  .status,
                              status_t::contended_k);
    }
    failures += expect(succeeded(indexes
                                     .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                             executor)
                                     .status));
    failures += expect_eq(indexes.size(), rows);
    std::size_t rows_in_shards = 0;
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) rows_in_shards += indexes.shard(shard).size();
    failures += expect_eq(rows_in_shards, rows);

    // A batch past the reservation grows into it - whole, with no ID holes - and the capacity
    // reported afterwards covers every row that landed.
    failures += expect(succeeded(indexes
                                     .insert(std::span<key_t const> {keys.data(), rows + 1}, oversized.view(), //
                                             executor)
                                     .status));
    failures += expect_eq(indexes.size(), 2 * rows + 1);
    failures += expect(indexes.capacity() >= indexes.size());
    return failures;
}

/** A composition grows past any reservation: rows well beyond it land one at a time and batched, every key stays
 *  findable, and a second, larger reservation is a hint rather than a refusal. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_growth(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const reserved = 8, singles = 1100, batched = 1100, rows = singles + batched;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({reserved, connectivity_16_k, dimensions}).status));

    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_growth"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);
    dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
    if (failed(rows_driver.status)) return failures + expect(false);
    for (std::size_t row = 0; row != singles; ++row)
        failures += expect(
            succeeded(indexes.insert(keys[row], vectors.view().row(row).as_vector(), rows_driver.handle()).status));
    failures += expect(succeeded(indexes
                                     .insert(std::span<key_t const> {keys.data() + singles, batched},
                                             vectors(nk::range(singles, rows), nk::slice), driver.executor)
                                     .status));

    failures += expect_eq(indexes.size(), rows);
    failures += expect(indexes.capacity() >= rows);
    std::size_t found = 0;
    for (key_t const key : keys) found += indexes.contains(key);
    failures += expect_eq(found, rows);
    auto fetched = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "indexes_growth", 1));
    failures += expect_eq(indexes.get(keys[rows - 1], fetched.span(), 1, driver.executor), 1u);
    failures += expect(succeeded(indexes.reserve({rows * 2, connectivity_16_k, dimensions}).status));
    failures += expect(indexes.capacity() >= rows * 2);
    return failures;
}

/** A sharded pass under a narrow build pool: the first re-wires nodes, the second fewer, each heals what it strands
 *  without losing reach, and the audit after them counts what the last pass ended on. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_refine(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));

    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_refine"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);
    typename index_t::insert_config_t const narrow {4};
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), executor, narrow).status));

    auto const [first, first_status] = indexes.refine(executor);
    failures += expect(succeeded(first_status));
    failures += expect(first.visited_nodes >= rows && first.changed_nodes <= first.visited_nodes);
    failures += expect(first.changed_nodes > 0);
    failures += expect(first.reachable_nodes <= rows && first.reachable_nodes + first.islands_found >= rows);

    auto const [second, second_status] = indexes.refine(executor);
    failures += expect(succeeded(second_status));
    failures += expect(second.changed_nodes < first.changed_nodes);

    // The audit reads the graph the last pass left, so it counts what that pass ended on.
    auto const [health, health_status] = indexes.health(executor);
    failures += expect(succeeded(health_status));
    failures += expect_eq(health.reachable_nodes, second.reachable_nodes);
    failures += expect(health.sweeps >= 1);
    return failures;
}

/** Executor arenas seeded at exactly the first-size estimate serve every walk: a re-table
 *  past the seed chunk lands in a chunk the arena opens beside it, so no walk saturates and
 *  every query ranks. A pool as wide as the whole reservation makes every walk exhaustive by
 *  construction - approximate search promises nothing narrower - so every thread's visited set
 *  outgrows its first size, and every key a sparse predicate admits surfaces. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_arena_growth(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 8 * dense_thread_nodes_k;
    std::size_t const queries = 16;
    std::size_t const wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));

    auto vectors = make_vectors<scalar_t>(rows + queries, dimensions, stream_key(seed, "upstream"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);
    auto const [insert_result, insert_status] = indexes.insert(std::span<key_t const> {keys},
                                                               vectors(nk::range(0, rows), nk::slice), //
                                                               executor);
    failures += expect(succeeded(insert_status));
    failures += expect_eq(insert_result.saturated_walks, 0u);

    // Sixteen keys match; with the pool as wide as the reservation every walk visits its whole shard,
    // so all eight surface - a narrower pool could promise only that what surfaces is admitted and ranked.
    typename index_t::search_config_t const exhaustive {static_cast<typename index_t::small_size_t>(rows)};
    auto sparse = [](key_t key) noexcept { return key % 128 == 0 ? admission_t::admit_k : admission_t::reject_k; };
    auto found_keys = nk::matrix<key_t>::zeros({queries, wanted}).value;
    auto found_distances = nk::matrix<typename indexes_type_::distance_t>::zeros({queries, wanted}).value;
    std::vector<std::size_t> counts(queries);
    auto const [search_result, search_status] = indexes.search_if(vectors(nk::range(rows, rows + queries), nk::slice),
                                                                  executor, exhaustive, found_keys.span(),
                                                                  found_distances.span(), std::span {counts}, sparse);
    failures += expect(succeeded(search_status));
    failures += expect_eq(search_result.saturated_walks, 0u);
    for (std::size_t const count : counts) failures += expect_eq(count, wanted);
    // The seed chunk is exactly the estimate, so a re-table can only have opened another.
    failures += expect(executor.arena(0).capacity() > indexes.shard(0).traversal_memory_usage());
    return failures;
}

/** One key, several rows: the locator map's whole reason to exist, driven through
 *  the single-row @c insert with a caller-built arena context. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_duplicate_keys(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));

    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "guards", 2));
    dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
    if (failed(rows_driver.status)) return failures + expect(false);

    // Every second row lands under one shared key; the rest keep distinct ones.
    key_t const shared = static_cast<key_t>(1000);
    for (std::size_t row = 0; row != rows; ++row) {
        key_t const key = row % 2 ? shared : static_cast<key_t>(1 + row);
        failures += expect(
            succeeded(indexes.insert(key, vectors.view().row(row).as_vector(), rows_driver.handle()).status));
    }
    failures += expect_eq(indexes.size(), rows);
    failures += expect_eq(indexes.count(shared), rows / 2);
    failures += expect_eq(indexes.count(static_cast<key_t>(1)), 1u);
    failures += expect(indexes.contains(shared));
    failures += expect(!indexes.contains(static_cast<key_t>(2)));

    // The probe-then-fetch contract of `get`: the true total returns even when the copy
    // truncates, and an absent key answers zero without touching the output.
    auto fetched = make_vectors<scalar_t>(2, dimensions, stream_key(seed, "guards", 3));
    failures += expect_eq(indexes.get(shared, fetched.span(), 2, driver.executor), rows / 2);
    failures += expect_eq(indexes.get(static_cast<key_t>(2), fetched.span(), 2, driver.executor), 0u);

    // The batched verbs answer what the single-key ones do, repeats and absent keys included, and
    // every key's rows land in its own band of the batch, in the single-key order.
    std::array<key_t, 4> const probes {shared, static_cast<key_t>(1), static_cast<key_t>(2), shared};
    std::span<key_t const> const probed {probes};
    std::array<std::size_t, 4> counted {}, contained {}, gotten {};
    auto batch = make_vectors<scalar_t>(probes.size() * 2, dimensions, stream_key(seed, "guards", 4));
    failures += expect(succeeded(indexes.count(probed, std::span {counted}, driver.executor)));
    failures += expect(succeeded(indexes.contains(probed, std::span {contained}, driver.executor)));
    failures += expect(succeeded(indexes.get(probed, batch.span(), 2, std::span {gotten}, driver.executor)));
    std::size_t const row_bytes = dense_rows_bytes(fetched.view(), 1);
    for (std::size_t probe = 0; probe != probes.size(); ++probe) {
        failures += expect_eq(counted[probe], indexes.count(probes[probe]));
        failures += expect_eq(contained[probe], indexes.contains(probes[probe]) ? 1u : 0u);
        failures += expect_eq(gotten[probe], indexes.get(probes[probe], fetched.span(), 2, driver.executor));
        for (std::size_t row = 0; row != min_of<std::size_t>(gotten[probe], 2); ++row)
            failures += expect(std::memcmp(batch.view().row(probe * 2 + row).as_vector().byte_data(),
                                           fetched.view().row(row).as_vector().byte_data(), row_bytes) == 0);
    }
    failures += expect_eq(indexes.get(probed, batch.span(), 3, std::span {gotten}, driver.executor),
                          status_t::config_mismatch_k);
    return failures;
}

/** A save config the sharded scenarios pass: the device residency certifies quiescence, since
 *  its flat matrices cannot be settlement-probed; the RAM residency lets the probe run. */
template <typename indexes_type_>
save_config_t dense_indexes_save_config(checkpoint_layout_t layout) noexcept {
    save_config_t config;
    config.layout = layout;
    if constexpr (dense_device_resident<typename indexes_type_::index_t>)
        config.synchronization = synchronization_t::immutable_k;
    return config;
}

/** The paths one sharded checkpoint scenario writes: distinct per process, so parallel
 *  suite runs never clobber each other; the part files follow the interchange's naming. */
struct dense_checkpoint_paths_t {
    char manifest[64] = {};
    char monolith[64] = {};

    explicit dense_checkpoint_paths_t(std::uintptr_t salt) noexcept {
        std::snprintf(manifest, sizeof(manifest), "/tmp/usearch-%08x.safegraphs.json", static_cast<unsigned>(salt));
        std::snprintf(monolith, sizeof(monolith), "/tmp/usearch-%08x.safegraphs", static_cast<unsigned>(salt));
    }

    /** Removes the manifest, its @p parts part files, and the monolith. */
    void remove(std::size_t parts) const noexcept {
        std::remove(manifest);
        std::remove(monolith);
        growing_buffer<char> part_path {global_allocator};
        for (std::size_t part = 0; part != parts; ++part) {
            auto const [path, status] = safegraphs_part_path(part_path, safegraphs_stem(manifest), part, parts);
            if (succeeded(status)) std::remove(path.data());
        }
    }
};

/** Builds @p rows keyed @c 1+row into @p indexes, freshly placed over @p placement and reserved. */
template <typename indexes_type_, typename executor_type_, typename vectors_type_, typename keys_type_,
          typename... placement_types_>
std::size_t dense_indexes_build(indexes_type_& indexes, executor_type_& executor, vectors_type_ const& vectors,
                                keys_type_ const& keys, std::size_t dimensions, placement_types_&... placement) {
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = keys.size();
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));
    failures += expect(succeeded(indexes
                                     .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                             executor)
                                     .status));
    return failures;
}

/** Every key of @p keys resolves to exactly one row of @p indexes, and sampled rows read back
 *  through @p executor come back byte-identical to @p vectors. */
template <typename indexes_type_, typename vectors_type_, typename keys_type_>
std::size_t dense_indexes_expect_restored(indexes_type_& indexes, vectors_type_ const& vectors, keys_type_ const& keys,
                                          std::size_t dimensions, typename indexes_type_::executor_t& executor,
                                          u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    std::size_t failures = 0;
    std::size_t const rows = keys.size();
    failures += expect_eq(indexes.size(), rows);
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != rows; ++row) miscounted += indexes.count(keys[row]) != 1;
    failures += expect_eq(miscounted, 0u);

    auto fetched = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "roundtrip", 2));
    std::size_t const row_bytes = static_cast<std::size_t>(vectors.view().stride_bytes(0));
    std::size_t mismatched = 0;
    for (std::size_t sample = 0; sample != 8; ++sample) {
        std::size_t const row = sample * (rows / 8);
        if (indexes.get(keys[row], fetched.span(), 1, executor) != 1) mismatched += 1;
        else
            mismatched += std::memcmp(fetched.data(), //
                                      vectors.view().row(row).as_vector().byte_data(), row_bytes) != 0;
    }
    failures += expect_eq(mismatched, 0u);
    return failures;
}

/** Two indexes built over the halves of one dataset merge into one, the second half passing through
 *  an index never reserved, which takes its geometry: the donors end empty, every key resolves to
 *  one row with its vector intact, the refine re-selects every row of the recipient's shards that
 *  took an appended part while the rest heal, and the merged search finds rows of either half -
 *  while absorbing itself refuses. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_absorb(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = dense_growth_nodes_k, half = rows / 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_absorb"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ recipient {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ donor {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ relay {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    // Twice the donor's shards, so half the recipient's take no appended part and must still heal.
    failures += expect(succeeded(place_shards(recipient, placement...)) &&
                       succeeded(place_shards(recipient, placement...)));
    failures += expect(succeeded(place_shards(donor, placement...)) && succeeded(place_shards(relay, placement...)));
    for (indexes_type_* built : {&recipient, &donor})
        failures += expect(succeeded(built->reserve({half, connectivity_16_k, dimensions}).status));
    failures += expect(succeeded(
        recipient.insert(std::span<key_t const> {keys.data(), half}, vectors(nk::range(0, half), nk::slice), executor)
            .status));
    failures += expect(succeeded(donor
                                     .insert(std::span<key_t const> {keys.data() + half, rows - half},
                                             vectors(nk::range(half, rows), nk::slice), executor)
                                     .status));
    failures += expect_eq(recipient.absorb(recipient, executor).status, status_t::config_mismatch_k);

    failures += expect(succeeded(relay.absorb(donor, executor).status));
    failures += expect(donor.size() == 0 && relay.size() == rows - half);
    failures += expect(relay.dimensions() == dimensions && relay.connectivity() == connectivity_16_k);

    auto const [absorbed, absorbed_status] = recipient.absorb(relay, executor);
    failures += expect(succeeded(absorbed_status));
    failures += expect_eq(relay.size(), 0u);
    failures += dense_indexes_expect_restored(recipient, vectors, keys, dimensions, executor, seed);
    // Relay shard s lands in recipient shard s, every row of which the refine must re-select.
    std::size_t refined_rows = 0;
    for (std::size_t shard = 0; shard != relay.shards_count(); ++shard) refined_rows += recipient.shard(shard).size();
    failures += expect(absorbed.visited_nodes >= refined_rows && absorbed.changed_nodes > 0);
    // The refine heals while its island count shrinks, which a dot graph's hubs can stall short of
    // every row.
    auto const [health, health_status] = recipient.health(executor);
    failures += expect(succeeded(health_status));
    failures += expect_eq(health.reachable_nodes, absorbed.reachable_nodes);
    failures += expect(health.reachable_nodes * 100 >= rows * dense_minimal_recall_percent_k);

    // Sampled self-recall through the merged search, judged by distance band on each half alone, so
    // donor rows the refine strands cannot hide behind the recipient's.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {recipient, driver};
        failures += expect(succeeded(rows_driver.status));
        std::size_t const samples_count = 64;
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        for (std::size_t const first_row : {std::size_t {0}, half}) {
            std::size_t hits = 0;
            for (std::size_t sample = 0; sample != samples_count; ++sample) {
                std::size_t const row = first_row + sample * (half / samples_count);
                auto const [result, status] = recipient.search(vectors.view().row(row).as_vector(),
                                                               rows_driver.handle(), {}, std::span<key_t> {found_keys},
                                                               std::span {found_distances});
                failures += expect(status == status_t::success_k && result.count > 0);
                for (std::size_t match = 0; match != result.count; ++match)
                    if (found_distances[match] < dense_self_match_tolerance_k &&
                        found_distances[match] > -dense_self_match_tolerance_k) {
                        ++hits;
                        break;
                    }
            }
            failures += expect(hits * 100 >= samples_count * dense_minimal_recall_percent_k);
        }
    }
    return failures;
}

/** The sharded checkpoint's whole life, in both layouts: batched rows in, a manifest
 *  plus one monolith per shard out and one consolidated monolith out, summaries read from
 *  either, a fresh index restored into the same placement, and byte-identical vectors
 *  fetched back by key - with a cancelled save leaving no manifest and no header, so the
 *  failed checkpoint reads as absent. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_save_load(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 1));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(indexes, executor, vectors, keys, dimensions, placement...);
    dense_checkpoint_paths_t const paths {reinterpret_cast<std::uintptr_t>(&indexes)};
    save_config_t const distributed = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::distributed_k);
    save_config_t monolith = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k);

    // A cancelled save writes parts but never the manifest - the checkpoint reads as absent.
    failures += expect_eq(indexes.save(paths.manifest, distributed, executor,
                                       [](std::size_t) noexcept { return continuation_t::cancel_k; }),
                          status_t::cancelled_k);
    failures += expect(failed(indexes_type_::metadata(paths.manifest).status));

    failures += expect(succeeded(indexes.save(paths.manifest, distributed, executor)));
    failures += expect(indexes.storage_requirements(distributed).value > 0);

    // The summary answers from the manifest and the first part's header, metric included.
    auto [manifest_summary, manifest_status] = indexes_type_::metadata(paths.manifest);
    failures += expect(succeeded(manifest_status));
    failures += expect_eq(manifest_summary.nodes, rows);
    failures += expect_eq(manifest_summary.parts, indexes.shards_count());
    failures += expect_eq(regular_width(manifest_summary.value_shape).value_or(0), dimensions);
    failures += expect(nk_kernel_named(manifest_summary.metric.characters, manifest_summary.metric.view().size()) ==
                       kernel_);

    // A fresh index in the same placement restores it whole, keys and all.
    indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(restored, placement...)));
    failures += expect(succeeded(restored.load(paths.manifest, load_config_t {}, executor)));
    failures += dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);

    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {restored, driver};
        failures += expect(succeeded(rows_driver.status));
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        auto const [search_result, search_status] = restored.search(
            vectors.view().row(0).as_vector(), rows_driver.handle(), {}, std::span<key_t> {found_keys},
            std::span {found_distances});
        failures += expect(search_status == status_t::success_k && search_result.count > 0);
    }

    // One consolidated monolith: a cancelled save leaves no header, an illegal alignment is
    // refused, and the provenance rides in the header the file's own reader parses back.
    std::array<safegraphs_custom_field_t, 1> const provenance {{{"embedding_model", "test"}}};
    monolith.provenance = provenance;
    failures += expect_eq(
        indexes.save(paths.monolith, monolith, executor, [](std::size_t) noexcept { return continuation_t::cancel_k; }),
        status_t::cancelled_k);
    failures += expect(failed(indexes_type_::metadata(paths.monolith).status));
    save_config_t misaligned = monolith;
    misaligned.alignment = 32;
    failures += expect_eq(indexes.save(paths.monolith, misaligned, executor), status_t::config_mismatch_k);
    failures += expect(succeeded(indexes.save(paths.monolith, monolith, executor)));
    if constexpr (dense_device_resident<index_t>) {
        save_config_t probing = monolith;
        probing.synchronization = synchronization_t::concurrent_k;
        failures += expect_eq(indexes.save(paths.monolith, probing, executor), status_t::config_mismatch_k);
        failures += expect(succeeded(indexes.save(paths.monolith, monolith, executor)));
    }
    {
        auto [mapping, status] = mapped_file_t::read(paths.monolith);
        failures += expect(succeeded(status));
        failures += expect_eq(indexes.storage_requirements(monolith).value, mapping.size());
    }
    auto [monolith_summary, monolith_status] = indexes_type_::metadata(paths.monolith);
    failures += expect(succeeded(monolith_status));
    failures += expect_eq(monolith_summary.nodes, rows);
    failures += expect_eq(monolith_summary.parts, 1u);
    std::size_t provenance_seen = 0;
    auto const [header, header_status] = safegraphs_read_header(
        paths.monolith, {}, [&](safegraphs_custom_field_t field) {
            provenance_seen += field.key == "embedding_model" && field.value == "test";
        });
    failures += expect(succeeded(header_status));
    failures += expect_eq(provenance_seen, 1u);
    failures += expect_eq(header.nodes(), rows);

    // The monolith restores whole into shard zero under the default placement.
    indexes_type_ whole {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(whole, placement...)));
    failures += expect(succeeded(whole.load(paths.monolith, load_config_t {}, executor)));
    failures += dense_indexes_expect_restored(whole, vectors, keys, dimensions, executor, seed);
    failures += expect_eq(whole.shard(0).size(), rows);

    paths.remove(indexes.shards_count());
    return failures;
}

/** Splitting and partitioning: a monolith written by @c S rotation-filled shards splits
 *  back into the very same per-shard fills, lands whole on shard zero when partitioned, and
 *  a manifest splits the same way; a reservation too small for the widest shard refuses, as
 *  does a second load into a populated index. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_split_load(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k + 3;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 3));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(indexes, executor, vectors, keys, dimensions, placement...);
    std::size_t const shards = indexes.shards_count();
    dense_checkpoint_paths_t const paths {reinterpret_cast<std::uintptr_t>(&indexes)};
    failures += expect(succeeded(indexes.save(
        paths.monolith, dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k), executor)));
    failures += expect(succeeded(indexes.save(
        paths.manifest, dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::distributed_k), executor)));
    load_config_t split;
    split.placement = checkpoint_placement_t::split_k;

    // The stripes are the rotation's own fills, so every shard restores to its original size.
    indexes_type_ striped {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(striped, placement...)));
    failures += expect(succeeded(striped.load(paths.monolith, split, executor)));
    failures += dense_indexes_expect_restored(striped, vectors, keys, dimensions, executor, seed);
    for (std::size_t shard = 0; shard != shards; ++shard)
        failures += expect_eq(striped.shard(shard).size(), indexes.shard(shard).size());

    // Partitioned, the one part lands whole on shard zero.
    indexes_type_ whole {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(whole, placement...)));
    failures += expect(succeeded(whole.load(paths.monolith, load_config_t {}, executor)));
    failures += dense_indexes_expect_restored(whole, vectors, keys, dimensions, executor, seed);
    failures += expect_eq(whole.shard(0).size(), rows);
    for (std::size_t shard = 1; shard < shards; ++shard) failures += expect_eq(whole.shard(shard).size(), 0u);

    // A manifest's parts concatenate in order, so its stripes are the same fills.
    indexes_type_ restriped {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(restriped, placement...)));
    failures += expect(succeeded(restriped.load(paths.manifest, split, executor)));
    failures += dense_indexes_expect_restored(restriped, vectors, keys, dimensions, executor, seed);
    for (std::size_t shard = 0; shard != shards; ++shard)
        failures += expect_eq(restriped.shard(shard).size(), indexes.shard(shard).size());

    // A reservation smaller than the parts still fits them, since a shard rounds up to whole
    // chunks; a populated index refuses a second load.
    indexes_type_ cramped {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(cramped, placement...)));
    failures += expect(succeeded(cramped.reserve({shards, connectivity_16_k, dimensions}).status));
    failures += expect(succeeded(cramped.load(paths.monolith, split, executor)));
    failures += dense_indexes_expect_restored(cramped, vectors, keys, dimensions, executor, seed);
    failures += expect_eq(striped.load(paths.monolith, split, executor), status_t::config_mismatch_k);
    failures += expect_eq(striped.size(), rows);

    paths.remove(shards);
    return failures;
}

/** Any number of lanes may take steps from one sharded plan: a shared cursor hands them
 *  out regardless of which shard they read, the plan finishes once, and the monolith they
 *  wrote restores into the same graph the one-shot save writes. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_pooled_save(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 5));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(indexes, executor, vectors, keys, dimensions);
    save_config_t const monolith = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k);
    auto const [bytes, measure_status] = indexes.storage_requirements(monolith);
    failures += expect(succeeded(measure_status));

    std::vector<std::byte> buffer(bytes);
    auto [plan, plan_status] = indexes.plan_save(bytes_span_t {buffer}, monolith);
    failures += expect(succeeded(plan_status));
    auto const steps = plan.steps();
    failures += expect(steps.size() >= indexes.shards_count());

    std::atomic<std::size_t> cursor {0};
    std::atomic<std::size_t> lane_failures {0};
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(3))) return failures + expect(false);
    pool.for_threads([&](std::size_t) noexcept {
        auto scratch = nk::matrix<node_id_t>::zeros({rows, connectivity_16_k}).value;
        if (scratch.empty()) return void(lane_failures.fetch_add(1));
        for (std::size_t position = cursor.fetch_add(1); position < steps.size(); position = cursor.fetch_add(1))
            if (failed(plan.save(steps[position], scratch.span()))) lane_failures.fetch_add(1);
    });
    failures += expect_eq(lane_failures.load(), 0u);
    failures += expect_eq(plan.finish(), status_t::success_k);

    indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(restored)));
    failures += expect(succeeded(restored.load(bytes_view_t {buffer}, load_config_t {}, executor)));
    failures += dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);
    return failures;
}

/** The norms column of @p monolith, which a sharded index saved over @p rows rows: present where
 *  the shards keep norms, taken as written by a restore, and derived again for a part without the
 *  column or with a dtype the shards never keep. Every restore saves again, and the norms that save
 *  writes are the ones it holds; every restore is placed over @p placement. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename executor_type_, typename... placement_types_>
std::size_t dense_indexes_expect_norms(std::vector<std::byte> const& monolith, executor_type_& executor,
                                       std::size_t rows, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    using norms_t = vector_norms<dot_result_t>;
    constexpr bool keeps_norms_k = kernel_ != nk_kernel_dot_k || dense_dot_keeps_norms<index_t>();
    std::size_t failures = 0;
    auto const [header, header_status] = safegraphs_read_header(bytes_view_t {monolith});
    failures += expect(succeeded(header_status));
    if (!keeps_norms_k) return failures + expect(header.norms.dtype == dtype_t::unknown_k);
    failures += expect(header.norms.dtype == dtype_of(dot_result_t::dtype()) &&
                       header.norms.shape == safegraphs_norms_shape_k && header.norms.nodes() == rows);
    if (failures) return failures;

    save_config_t const config = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k);
    auto resaved_norms = [&](std::vector<std::byte> const& source) {
        std::vector<norms_t> norms(rows);
        indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(restored, placement...)));
        failures += expect(succeeded(restored.load(bytes_view_t {source}, load_config_t {}, executor)));
        std::vector<std::byte> resaved(restored.storage_requirements(config).value);
        failures += expect(succeeded(restored.save(bytes_span_t {resaved}, config, executor)));
        auto const [resaved_header, resaved_status] = safegraphs_read_header(bytes_view_t {resaved});
        bytes_view_t const column = resaved_header.data_of(bytes_view_t {resaved}, resaved_header.norms);
        failures += expect(succeeded(resaved_status) && column.size() == rows * sizeof(norms_t));
        if (column.size() == rows * sizeof(norms_t)) std::memcpy(norms.data(), column.data(), column.size());
        return norms;
    };
    // Every copy rotates the stored rows by one, so a restore that took them would show it.
    auto doctored = [&](safegraphs_column_t norms) {
        std::vector<std::byte> copy = monolith;
        bytes_span_t const column = header.data_of(bytes_span_t {copy}, header.norms);
        std::rotate(column.begin(), column.begin() + sizeof(norms_t), column.end());
        safegraphs_header_t patched = header;
        patched.norms = norms;
        std::array<safegraphs_custom_field_t, 1> const fields {{{"metric", nk_kernel_name(kernel_)}}};
        failures += expect(succeeded(safegraphs_write_header(bytes_span_t {copy}, patched, fields)));
        return copy;
    };

    std::vector<norms_t> original(rows);
    std::memcpy(original.data(), header.data_of(bytes_view_t {monolith}, header.norms).data(), rows * sizeof(norms_t));
    std::vector<std::byte> const rotated = doctored(header.norms);
    std::vector<norms_t> const taken = resaved_norms(rotated);
    failures += expect(std::memcmp(taken.data(), header.data_of(bytes_view_t {rotated}, header.norms).data(),
                                   rows * sizeof(norms_t)) == 0);

    safegraphs_column_t mistyped = header.norms;
    mistyped.dtype = sizeof(dot_result_t) == 8 ? dtype_t::b64_k : dtype_t::b32_k;
    std::vector<norms_t> const without_column = resaved_norms(doctored({}));
    std::vector<norms_t> const under_mistype = resaved_norms(doctored(mistyped));
    for (std::size_t sample = 0; sample != 8; ++sample) {
        std::size_t const row = sample * (rows / 8);
        f64_t const sum = static_cast<f64_t>(original[row].sum), sumsq = static_cast<f64_t>(original[row].sumsq);
        failures += expect_near(static_cast<f64_t>(without_column[row].sum), sum);
        failures += expect_near(static_cast<f64_t>(without_column[row].sumsq), sumsq);
        failures += expect_near(static_cast<f64_t>(under_mistype[row].sum), sum);
        failures += expect_near(static_cast<f64_t>(under_mistype[row].sumsq), sumsq);
    }
    return failures;
}

/** Buffers hold one consolidated monolith: @c storage_requirements sizes it, the summary reads from
 *  it, a fresh index restores from it, its norms as written or derived again, and a short buffer, a
 *  distributed layout, or a cancelled save leave nothing a reader accepts. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_buffer_roundtrip(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 4));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(indexes, executor, vectors, keys, dimensions, placement...);
    save_config_t const monolith = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k);
    auto const [bytes, measure_status] = indexes.storage_requirements(monolith);
    failures += expect(succeeded(measure_status));
    failures += expect(bytes > 0);

    std::vector<std::byte> buffer(bytes);
    failures += expect(succeeded(indexes.save(bytes_span_t {buffer}, monolith, executor)));
    auto [summary, metadata_status] = indexes_type_::metadata(bytes_view_t {buffer});
    failures += expect(succeeded(metadata_status));
    failures += expect_eq(summary.nodes, rows);
    failures += expect_eq(summary.parts, 1u);
    failures += expect_eq(regular_width(summary.value_shape).value_or(0), dimensions);

    indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(restored, placement...)));
    failures += expect(succeeded(restored.load(bytes_view_t {buffer}, load_config_t {}, executor)));
    failures += dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);
    failures += dense_indexes_expect_norms<indexes_type_, kernel_>(buffer, executor, rows, placement...);

    // A buffer one byte short, a distributed layout, and a cancelled save all leave no monolith.
    std::vector<std::byte> shorter(bytes - 1);
    failures += expect_eq(indexes.save(bytes_span_t {shorter}, monolith, executor), status_t::overflow_k);
    save_config_t const distributed = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::distributed_k);
    failures += expect_eq(indexes.save(bytes_span_t {buffer}, distributed, executor), status_t::config_mismatch_k);
    std::vector<std::byte> cancelled(bytes);
    failures += expect_eq(indexes.save(bytes_span_t {cancelled}, monolith, executor,
                                       [](std::size_t) noexcept { return continuation_t::cancel_k; }),
                          status_t::cancelled_k);
    failures += expect(failed(indexes_type_::metadata(bytes_view_t {cancelled}).status));
    return failures;
}

/** A cancelled load publishes nothing: cancelling at every shard boundary answers
 *  @c cancelled_k, leaves the index empty, and the next plain load restores it whole. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_cancelled_load(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 5));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(indexes, executor, vectors, keys, dimensions, placement...);
    std::size_t const shards = indexes.shards_count();
    dense_checkpoint_paths_t const paths {reinterpret_cast<std::uintptr_t>(&indexes)};
    failures += expect(succeeded(indexes.save(
        paths.manifest, dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::distributed_k), executor)));

    for (std::size_t boundary = 0; boundary != shards; ++boundary) {
        indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(restored, placement...)));
        auto cancel_at = [boundary](std::size_t shard) noexcept {
            return shard == boundary ? continuation_t::cancel_k : continuation_t::continue_k;
        };
        failures += expect_eq(restored.load(paths.manifest, load_config_t {}, executor, cancel_at),
                              status_t::cancelled_k);
        failures += expect_eq(restored.size(), 0u);
        failures += expect_eq(restored.count(keys[0]), 0u);
        failures += expect(succeeded(restored.load(paths.manifest, load_config_t {}, executor)));
        failures += dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);
    }

    paths.remove(shards);
    return failures;
}

/**
 *  @brief Two executors drive one sharded index concurrently: disjoint key ranges, one
 *      batched @c insert per executor, both interleaving into the same shards - asserted
 *      through the public surface only.
 *
 *  A RAM index admits both batches at once. A VRAM index admits one mutation at a time and
 *  answers the overlapping one with @c contended_k, so a refused batch is retried once the
 *  threads join; either way every row lands exactly once.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
std::size_t test_dense_indexes_concurrent_add(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const executors_count = 2;
    std::size_t const rows_per_executor = 2 * dense_thread_nodes_k;
    std::size_t const nodes_count = executors_count * rows_per_executor;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement...)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));

    std::array<dense_indexes_executor<indexes_type_>, executors_count> drivers {
        dense_indexes_executor<indexes_type_> {placement...}, dense_indexes_executor<indexes_type_> {placement...}};
    std::vector<nk::matrix<scalar_t>> slices;
    std::vector<std::vector<key_t>> keys;
    for (std::size_t which = 0; which != executors_count; ++which) {
        if (failed(drivers[which].status)) return failures + expect(false);
        slices.push_back(make_vectors<scalar_t>(rows_per_executor, dimensions, stream_key(seed, "indexes_add", which)));
        // Keys never take the value zero, so a stale read of a zero-filled key column can
        // never masquerade as a legitimate identity.
        std::vector<key_t> range(rows_per_executor);
        for (std::size_t row = 0; row != rows_per_executor; ++row)
            range[row] = static_cast<key_t>(1 + which * rows_per_executor + row);
        keys.push_back(std::move(range));
    }

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(executors_count))) return failures + expect(false);
    std::array<status_t, executors_count> outcomes {};
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= executors_count) return;
        outcomes[thread] = indexes
                               .insert(std::span<key_t const> {keys[thread].data(), rows_per_executor},
                                       slices[thread].view(), drivers[thread].executor)
                               .status;
        assert_no_locks_held();
    });
    for (std::size_t which = 0; which != executors_count; ++which) {
        if (outcomes[which] == status_t::contended_k && !ram_collection<indexes_type_>)
            outcomes[which] = indexes
                                  .insert(std::span<key_t const> {keys[which].data(), rows_per_executor},
                                          slices[which].view(), drivers[which].executor)
                                  .status;
        failures += expect_eq(outcomes[which], status_t::success_k);
    }
    failures += expect_eq(indexes.size(), nodes_count);

    // Rotation splits every batch evenly, so the shards end exactly equally full.
    std::size_t rows_in_shards = 0;
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) rows_in_shards += indexes.shard(shard).size();
    failures += expect_eq(rows_in_shards, nodes_count);
    failures += expect_eq(indexes.shard(0).size(), nodes_count / indexes.shards_count());

    // Every key is stored exactly once, and resolvable without naming a shard or an ID.
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != nodes_count; ++row) miscounted += indexes.count(static_cast<key_t>(1 + row)) != 1;
    failures += expect_eq(miscounted, 0u);

    // Sampled self-recall through the merged cross-shard search, judged by distance band.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {indexes, drivers[0]};
        failures += expect(succeeded(rows_driver.status));
        std::size_t hits = 0;
        std::size_t const samples_count = 8;
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        for (std::size_t sample = 0; sample != samples_count; ++sample) {
            std::size_t const row = sample * (rows_per_executor / samples_count);
            auto const [result, status] = indexes.search(slices[0].view().row(row).as_vector(), rows_driver.handle(),
                                                         {}, std::span<key_t> {found_keys},
                                                         std::span {found_distances});
            failures += expect(status == status_t::success_k && result.count > 0);
            for (std::size_t match = 0; match != result.count; ++match)
                if (found_distances[match] < dense_self_match_tolerance_k &&
                    found_distances[match] > -dense_self_match_tolerance_k) {
                    ++hits;
                    break;
                }
        }
        failures += expect(hits * 100 >= samples_count * dense_minimal_recall_percent_k);
    }
    return failures;
}

/** A reader thread races executor-driven batched @c insert calls into the shards, asserting
 *  the key-visibility invariant: a search may reach a node the moment it publishes, so
 *  the key it resolves must already be written - never the key column's zero-fill, never
 *  outside the assigned range. Every batch spans both shards yet shows whole or not at all.
 *  RAM only: a VRAM index admits no search during a mutation. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_search_race(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;

    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_race"));
    std::vector<key_t> keys(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(1 + row);

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::atomic<bool> writer_done {false};
    std::atomic<std::size_t> thread_failures {0};
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            for (std::size_t first = 0; first != nodes_count; first += dense_thread_nodes_k) {
                auto [result, status] = indexes.insert(
                    std::span<key_t const> {keys.data() + first, dense_thread_nodes_k},
                    vectors(nk::range(first, first + dense_thread_nodes_k), nk::slice), executor);
                if (status != status_t::success_k) thread_failures.fetch_add(1);
            }
            writer_done.store(true, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
        if (failed(rows_driver.status)) {
            thread_failures.fetch_add(1);
            writer_done.store(true, std::memory_order_release);
            return;
        }
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        std::size_t searches = 0;
        for (; !writer_done.load(std::memory_order_acquire) || searches < dense_reader_minimal_searches_k; ++searches) {
            auto const [result, status] = indexes.search(vectors.view().row(searches % nodes_count).as_vector(),
                                                         rows_driver.handle(), {}, std::span<key_t> {found_keys},
                                                         std::span {found_distances});
            if (status != status_t::success_k) {
                thread_failures.fetch_add(1);
                break;
            }
            for (std::size_t match = 0; match != result.count; ++match)
                if (found_keys[match] < key_t {1} || found_keys[match] > static_cast<key_t>(nodes_count))
                    thread_failures.fetch_add(1);
            if (indexes.size() % dense_thread_nodes_k) thread_failures.fetch_add(1);
        }
        assert_no_locks_held();
    });
    failures += expect_eq(thread_failures.load(), 0u);
    failures += expect_eq(indexes.size(), nodes_count);
    return failures;
}

/** The optimistic save arm under fire: a saver thread loops concurrent-arm saves
 *  against two inserter threads & a searcher, classifying every answer as @c contended_k
 *  or @c success_k - and every settled payload must be structurally sane and reload
 *  into a fresh index whose recall holds. Inserters keep trickling spare rows until
 *  the saver has provably contended at least once, and a settle phase after the race
 *  demands a deterministic success. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_save_race(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    std::size_t const inserter_threads_count = 2, spare_rows_count = 64;
    std::size_t const prefill_rows_count = 256;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t const nodes_ceiling = prefill_rows_count + inserter_threads_count * (dense_thread_nodes_k + //
                                                                                     spare_rows_count);

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(inserter_threads_count + 2))) return failures + expect(false);

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    failures += expect(index.reserve({nodes_ceiling, connectivity_16_k, dimensions}).status == status_t::success_k);

    auto threads = dense_make_threads<index_type_>(make_index, inserter_threads_count + 2,
                                                   dense_thread_nodes_k + spare_rows_count, dimensions,
                                                   stream_key(seed, "save_race"));
    auto prefill_vectors = make_vectors<scalar_t>(prefill_rows_count, dimensions, stream_key(seed, "save_race", 1));
    typename index_type_::traversal_context_t prefill_context {{}};
    failures += expect(index.insert(prefill_vectors.view(), {}, prefill_context).status == status_t::success_k);

    // The saver's destinations are sized once at the ceiling; each attempt fills the
    // walked prefix, so nothing allocates inside the race loop.
    auto saver_vectors = nk::matrix<scalar_t>::zeros({nodes_ceiling, dimensions}).value;
    auto saver_neighbors = nk::matrix<node_id_t>::zeros({nodes_ceiling, connectivity_16_k}).value;
    auto saver_scratch = nk::matrix<node_id_t>::zeros({nodes_ceiling, connectivity_16_k}).value;
    failures += expect(!saver_vectors.empty() && !saver_neighbors.empty() && !saver_scratch.empty());

    struct recorded_payload_t {
        nk::matrix<scalar_t> vectors;
        nk::matrix<node_id_t> neighbors;
        std::size_t nodes = 0;
    };
    std::vector<recorded_payload_t> recorded_payloads;
    recorded_payloads.reserve(4);

    std::atomic<std::size_t> inserter_threads_remaining {inserter_threads_count};
    std::atomic<std::size_t> contended_saves {0}, settled_saves {0}, thread_failures {0};

    auto attempt_save = [&]() noexcept {
        auto [plan, status] = index.plan_save({dense_matrix_bytes(saver_vectors), dense_matrix_bytes(saver_neighbors)},
                                              synchronization_t::concurrent_k);
        if (failed(status)) return std::pair {std::size_t {0}, status};
        std::size_t const nodes = static_cast<std::size_t>(plan.visible);
        for (auto const step : plan.steps())
            if (status_t const saved = plan.save(step, saver_scratch.span()); failed(saved))
                return std::pair {nodes, saved};
        return std::pair {nodes, plan.finish()};
    };
    auto audit_settled = [&](std::size_t nodes) noexcept {
        std::size_t const row_ids = dense_rows_bytes(saver_neighbors.view(), 1) / sizeof(node_id_t);
        for (std::size_t row = 0; row != nodes; ++row) {
            node_id_t const* ids = saver_neighbors.data() + row * row_ids;
            if (ids[0] == node_id_unknown_k) return false;       // Never a parked lock word.
            if (ids[0] == static_cast<node_id_t>(row)) continue; // Zero-neighbors self-loop.
            for (std::size_t slot = 0; slot != connectivity_16_k; ++slot) {
                if (ids[slot] == node_id_unknown_k) break; // The sentinel ends the row.
                if (static_cast<std::size_t>(ids[slot]) >= nodes) return false;
            }
        }
        return true;
    };
    auto record_settled = [&](std::size_t nodes) noexcept {
        settled_saves.fetch_add(1);
        if (!audit_settled(nodes)) thread_failures.fetch_add(1);
        if (recorded_payloads.size() == recorded_payloads.capacity()) return;
        recorded_payload_t recorded {nk::matrix<scalar_t>::zeros({nodes, dimensions}).value,
                                     nk::matrix<node_id_t>::zeros({nodes, connectivity_16_k}).value, nodes};
        if (recorded.vectors.empty() || recorded.neighbors.empty()) return;
        std::memcpy(recorded.vectors.data(), saver_vectors.data(), dense_rows_bytes(saver_vectors.view(), nodes));
        std::memcpy(recorded.neighbors.data(), saver_neighbors.data(), dense_rows_bytes(saver_neighbors.view(), nodes));
        recorded_payloads.push_back(std::move(recorded));
    };
    auto saver_thread = [&]() noexcept { // Classify, audit, and record every settled payload.
        for (std::size_t attempts = 0; inserter_threads_remaining.load(std::memory_order_acquire) != 0; ++attempts) {
            if (attempts == spin_limit_k) return void(thread_failures.fetch_add(1)); // A stuck racer.
            auto const [nodes, save_status] = attempt_save();
            if (save_status == status_t::contended_k) contended_saves.fetch_add(1);
            else if (save_status == status_t::success_k) record_settled(nodes);
            else thread_failures.fetch_add(1);
        }
        // The settle phase: with every racer done, a save must soon succeed.
        for (std::size_t attempts = 0; attempts != attempt_limit_k; ++attempts) {
            auto const [nodes, save_status] = attempt_save();
            if (save_status == status_t::success_k) return record_settled(nodes);
            if (save_status != status_t::contended_k) return void(thread_failures.fetch_add(1));
        }
        thread_failures.fetch_add(1); // The settle phase starved.
    };
    auto inserter_thread = [&](std::size_t thread) noexcept { // A slice, then spare singles
                                                              // until the saver has contended.
        auto slice = threads.slices[thread].view();
        auto main_rows = slice(nk::range(0, dense_thread_nodes_k), nk::slice);
        if (index.insert(main_rows, {}, threads.contexts[thread]).status != status_t::success_k)
            thread_failures.fetch_add(1);
        for (std::size_t spare = 0; spare != spare_rows_count && //
                                    contended_saves.load(std::memory_order_acquire) == 0;
             ++spare) {
            auto spare_row = slice(nk::range(dense_thread_nodes_k + spare, dense_thread_nodes_k + spare + 1),
                                   nk::slice);
            if (index.insert(spare_row, {}, threads.contexts[thread]).status != status_t::success_k)
                thread_failures.fetch_add(1);
            std::this_thread::yield();
        }
        inserter_threads_remaining.fetch_sub(1, std::memory_order_release);
        assert_no_locks_held();
    };
    auto searcher_thread = [&](std::size_t thread) noexcept { // The reader-thread invariants.
        std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;
        auto query_slice = threads.slices[thread].view();
        for (std::size_t step = 0;
             step < dense_reader_minimal_searches_k || inserter_threads_remaining.load(std::memory_order_acquire) != 0;
             ++step) {
            auto query = query_slice.row(step % dense_thread_nodes_k).as_vector();
            auto const [result, status] = index.search(query, threads.contexts[thread]);
            if (status != status_t::success_k) thread_failures.fetch_add(1);
            thread_failures.fetch_add(
                scored_failures(threads.contexts[thread].results.data(), result.count, index.size(), expansion_bound));
        }
        assert_no_locks_held();
    };
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) return saver_thread();
        if (thread <= inserter_threads_count) return inserter_thread(thread);
        if (thread == inserter_threads_count + 1) return searcher_thread(thread);
    });

    failures += expect_eq(thread_failures.load(), 0u);
    failures += expect(contended_saves.load() + settled_saves.load() > 0);
    failures += expect(settled_saves.load() > 0);
    failures += expect(recorded_payloads.size() >= 1);

    // Every settled payload reloads into a fresh index whose self-recall holds, the
    // payload's own vector rows serving as the queries.
    for (recorded_payload_t const& recorded : recorded_payloads) {
        std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
        index_type_& restored = *restored_ptr;
        failures += expect(restored.reserve({recorded.nodes, connectivity_16_k, dimensions}).status ==
                           status_t::success_k);
        failures += expect_eq(
            restored.load({dense_matrix_bytes(recorded.vectors.view()), dense_matrix_bytes(recorded.neighbors.view())},
                          recorded.nodes, {0, static_cast<node_id_t>(recorded.nodes)}),
            status_t::success_k);
        failures += expect_eq(restored.size(), recorded.nodes);
        if constexpr (kernel_ != nk_kernel_dot_k)
            failures += dense_expect_self_recall(restored, make_index, recorded.vectors.view(), recorded.nodes, 16);
    }
    return failures;
}

/** @ref dense_expect_save_race, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_save_race(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_save_race<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Races

#pragma region Readers

/** On a quiesced graph the @c immutable_k search arm reproduces the concurrent arm
 *  bit for bit - fresh identically-seeded contexts, so the landmark draws align - at
 *  the default expansion and at an explicit one. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_immutable_parity(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const nodes_count = 512, stored_queries_count = 16, novel_queries_count = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t build_context {{}};
    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "immutable_parity"));
    failures += expect(index.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect(index.insert(vectors.view(), {}, build_context).status == status_t::success_k);
    auto novel_queries = make_vectors<scalar_t>(novel_queries_count, dimensions,
                                                stream_key(seed, "immutable_parity", 1));

    auto compare_arms = [&](auto query, std::size_t expansion_factor) noexcept {
        typename index_type_::traversal_context_t concurrent_context {{}}, immutable_context {{}};
        typename index_type_::search_config_t concurrent_config {};
        concurrent_config.expansion_factor = static_cast<decltype(concurrent_config.expansion_factor)>( //
            expansion_factor);
        typename index_type_::search_config_t immutable_config = concurrent_config;
        immutable_config.synchronization = synchronization_t::immutable_k;
        auto const [concurrent_result, concurrent_status] = index.search(query, concurrent_context, concurrent_config);
        auto const [immutable_result, immutable_status] = index.search(query, immutable_context, immutable_config);
        failures += expect_eq(immutable_status, concurrent_status);
        failures += expect_eq(immutable_result.count, concurrent_result.count);
        for (std::size_t match = 0; match != concurrent_result.count; ++match) {
            failures += expect_eq(immutable_context.results[match].node_id, concurrent_context.results[match].node_id);
            failures += expect(immutable_context.results[match].distance == concurrent_context.results[match].distance);
        }
    };
    for (std::size_t expansion_factor : {std::size_t {0}, std::size_t {64}}) {
        for (std::size_t sample = 0; sample != stored_queries_count; ++sample)
            compare_arms(vectors.view().row((sample * 31) % nodes_count).as_vector(), expansion_factor);
        for (std::size_t sample = 0; sample != novel_queries_count; ++sample)
            compare_arms(novel_queries.view().row(sample).as_vector(), expansion_factor);
    }
    return failures;
}

/** @ref dense_expect_immutable_parity, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_immutable_parity(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_immutable_parity<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** The external reader's cursor, exercised across chunk seams: full-prefix parity
 *  of IDs & vector bytes, follow semantics on the staged row, refresh stickiness -
 *  past the end stays past the end - a thread racing live inserters, and the immutable
 *  arm agreeing with the concurrent one on the settled graph. */
template <typename index_type_, typename concurrent_cursor_type_, typename immutable_cursor_type_,
          typename make_index_type_>
std::size_t dense_expect_cursor(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    std::size_t const nodes_count = 1'200, racing_inserts_count = 64;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto vectors = make_vectors<scalar_t>(nodes_count + 1 + racing_inserts_count, dimensions,
                                          stream_key(seed, "cursor"));
    failures += expect(index.reserve({nodes_count + 1 + racing_inserts_count, connectivity_16_k, dimensions}).status ==
                       status_t::success_k);
    auto main_rows = vectors(nk::range(0, nodes_count), nk::slice);
    failures += expect(index.insert(main_rows, {}, context).status == status_t::success_k);
    std::size_t const row_bytes = dense_rows_bytes(vectors.view(), 1);

    concurrent_cursor_type_ cursor {index};
    failures += expect(!cursor); // A fresh cursor starts past the end.
    failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_count);
    failures += expect_eq(cursor.seek(0), status_t::uninitialized_k); // Before its reserve.
    failures += expect_eq(cursor.reserve(counting_allocator<byte_t> {ledger}), status_t::success_k);

    // Full-prefix parity: ID order, exact stored bytes, every staged ID in bounds.
    failures += expect_eq(cursor.seek(0), status_t::success_k);
    std::size_t visited = 0;
    for (; cursor; ++cursor, ++visited) {
        if (cursor.id() != static_cast<node_id_t>(visited)) failures += expect(false);
        if (std::memcmp(cursor.vector().data(), vectors.view().row(visited).as_vector().data(), row_bytes) != 0)
            failures += expect(false);
        auto const staged = cursor.neighbors();
        if (staged.size() > connectivity_16_k) failures += expect(false);
        for (node_id_t const neighbor : staged)
            if (neighbor >= cursor.visible_count()) failures += expect(false);
    }
    failures += expect_eq(visited, nodes_count);

    // Follow semantics: every staged edge lands where the row said; misuse refuses.
    failures += expect_eq(cursor.seek(0), status_t::success_k);
    std::vector<node_id_t> first_row {cursor.neighbors().begin(), cursor.neighbors().end()};
    for (std::size_t slot = 0; slot != first_row.size(); ++slot) {
        failures += expect_eq(cursor.seek(0), status_t::success_k);
        failures += expect_eq(cursor.follow(static_cast<typename index_type_::small_size_t>(slot)),
                              status_t::success_k);
        failures += expect_eq(cursor.id(), first_row[slot]);
    }
    failures += expect_eq(cursor.seek(0), status_t::success_k);
    failures += expect_eq(cursor.follow(static_cast<typename index_type_::small_size_t>(cursor.neighbors().size())),
                          status_t::malformed_k);
    failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_count)), status_t::malformed_k);

    // The snapshot is sticky until refreshed; a past-the-end cursor never resurrects.
    auto late_row = vectors(nk::range(nodes_count, nodes_count + 1), nk::slice);
    failures += expect(index.insert(late_row, {}, context).status == status_t::success_k);
    failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_count);
    failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_count)), status_t::malformed_k);
    cursor.refresh();
    failures += expect_eq(static_cast<std::size_t>(cursor.visible_count()), nodes_count + 1);
    failures += expect_eq(cursor.seek(static_cast<node_id_t>(nodes_count)), status_t::success_k);
    concurrent_cursor_type_ parked_cursor {index};
    failures += expect_eq(parked_cursor.reserve(counting_allocator<byte_t> {ledger}), status_t::success_k);

    // A cursor thread races an inserting thread; every visited ID stays below that pass's
    // snapshot, and the parked cursor re-anchors past the end after the race.
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::atomic<std::size_t> thread_failures {0};
    std::atomic<bool> writer_done {false};
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            typename index_type_::traversal_context_t lane_context {{}};
            for (std::size_t ordinal = 0; ordinal != racing_inserts_count; ++ordinal) {
                auto racing_row = vectors(nk::range(nodes_count + 1 + ordinal, nodes_count + 2 + ordinal), nk::slice);
                if (index.insert(racing_row, {}, lane_context).status != status_t::success_k)
                    thread_failures.fetch_add(1);
                std::this_thread::yield();
            }
            writer_done.store(true, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        if (thread != 1) return;
        concurrent_cursor_type_ racing_cursor {index};
        if (racing_cursor.reserve(counting_allocator<byte_t> {ledger}) != status_t::success_k)
            return void(thread_failures.fetch_add(1));
        std::size_t sweeps = 0;
        do {
            if (sweeps++ == attempt_limit_k) return void(thread_failures.fetch_add(1)); // A stuck writer.
            racing_cursor.refresh();
            node_id_t const snapshot = racing_cursor.visible_count();
            if (failed(racing_cursor.seek(0))) return void(thread_failures.fetch_add(1));
            for (; racing_cursor; ++racing_cursor)
                if (racing_cursor.id() >= snapshot) thread_failures.fetch_add(1);
        } while (!writer_done.load(std::memory_order_acquire));
    });
    failures += expect_eq(thread_failures.load(), 0u);
    parked_cursor.refresh();
    failures += expect(!parked_cursor);
    failures += expect_eq(static_cast<std::size_t>(parked_cursor.id()), index.size());
    failures += expect_eq(parked_cursor.seek(0), status_t::success_k);

    // The immutable arm sweeps the settled graph to the same count & the same rows.
    immutable_cursor_type_ settled_cursor {index};
    failures += expect_eq(settled_cursor.reserve(counting_allocator<byte_t> {ledger}), status_t::success_k);
    failures += expect_eq(settled_cursor.seek(0), status_t::success_k);
    std::size_t settled_visited = 0;
    for (; settled_cursor; ++settled_cursor) ++settled_visited;
    failures += expect_eq(settled_visited, index.size());
    for (std::size_t sample = 0; sample != 8; ++sample) {
        node_id_t const node = static_cast<node_id_t>((sample * 149) % index.size());
        failures += expect_eq(settled_cursor.seek(node), status_t::success_k);
        failures += expect_eq(cursor.seek(node), status_t::success_k);
        auto const settled_staged = settled_cursor.neighbors();
        auto const concurrent_staged = cursor.neighbors();
        failures += expect_eq(settled_staged.size(), concurrent_staged.size());
        for (std::size_t slot = 0; slot != settled_staged.size(); ++slot)
            failures += expect_eq(settled_staged[slot], concurrent_staged[slot]);
    }
    return failures;
}

/** @ref dense_expect_cursor, then the ledger it left must balance. */
template <typename index_type_, typename concurrent_cursor_type_, typename immutable_cursor_type_,
          typename make_index_type_>
std::size_t test_dense_cursor(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_cursor<index_type_, concurrent_cursor_type_, immutable_cursor_type_>(
        make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Readers

#pragma region Allocation Budgets

/** A starved load publishes nothing; the documented contract destroys the victim. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
std::size_t dense_expect_starved_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                      typename index_type_::columns_view_t source, std::size_t nodes_count,
                                      std::size_t dimensions, iota_view<node_id_type_> local) {
    std::size_t failures = 0;
    std::unique_ptr<index_type_> const starved_ptr = make_index(ledger);
    index_type_& starved = *starved_ptr;
    auto [starved_capacity, starved_status] = starved.reserve({1, connectivity_16_k, dimensions});
    failures += expect(starved_status == status_t::success_k);
    failures += expect(starved_capacity < nodes_count); // The growth is genuine.
    ledger.byte_budget.store(ledger.live_bytes.load());
    failures += expect_eq(starved.load(source, nodes_count, local), status_t::bad_alloc_k);
    failures += expect_eq(starved.size(), 0u);
    ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
    return failures;
}

/** The ledger's byte budget provably starves exactly the chunk-growth path: a
 *  refused mid-insert growth is all-or-nothing & retryable, and a starved load
 *  publishes nothing - the fresh-index retry then succeeds with recall intact. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t dense_expect_allocation_budget(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                           u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t failures = 0;
    std::size_t const extra_rows_count = 100;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto [capacity, reserve_status] = index.reserve({1'000, connectivity_16_k, dimensions});
    failures += expect(reserve_status == status_t::success_k);
    std::size_t const fill_rows_count = capacity; // Fill to the brim, so one more row must grow.
    auto vectors = make_vectors<scalar_t>(fill_rows_count + extra_rows_count, dimensions,
                                          stream_key(seed, "allocation_budget"));
    auto fill_rows = vectors(nk::range(0, fill_rows_count), nk::slice);
    auto extra_rows = vectors(nk::range(fill_rows_count, fill_rows_count + extra_rows_count), nk::slice);
    failures += expect(index.insert(fill_rows, {}, context).status == status_t::success_k);

    // Freeze the budget: the growth election must refuse, all-or-nothing.
    ledger.byte_budget.store(ledger.live_bytes.load());
    failures += expect(index.insert(extra_rows, {}, context).status == status_t::bad_alloc_k);
    failures += expect_eq(index.size(), fill_rows_count);

    // Lift it: the identical batch is retryable.
    ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
    failures += expect(index.insert(extra_rows, {}, context).status == status_t::success_k);
    std::size_t const nodes_count = fill_rows_count + extra_rows_count;
    failures += expect_eq(index.size(), nodes_count);

    auto saved_vectors = nk::matrix<scalar_t>::zeros({nodes_count, dimensions}).value;
    auto saved_neighbors = nk::matrix<node_id_t>::zeros({nodes_count, connectivity_16_k}).value;
    auto scratch = nk::matrix<node_id_t>::zeros({nodes_count, connectivity_16_k}).value;
    failures += expect(!saved_vectors.empty() && !saved_neighbors.empty() && !scratch.empty());
    failures += expect_eq(index.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                                     synchronization_t::exclusive_k, scratch.span()),
                          status_t::success_k);

    typename index_type_::columns_view_t const source {dense_matrix_bytes(saved_vectors.view()),
                                                       dense_matrix_bytes(saved_neighbors.view())};
    iota_view<node_id_t> const local {0, static_cast<node_id_t>(nodes_count)};

    failures += dense_expect_starved_load<index_type_>(make_index, ledger, source, nodes_count, dimensions, local);

    std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
    index_type_& restored = *restored_ptr;
    failures += expect(restored.reserve({nodes_count, connectivity_16_k, dimensions}).status == status_t::success_k);
    failures += expect_eq(restored.load(source, nodes_count, local), status_t::success_k);
    failures += expect_eq(restored.size(), nodes_count);
    if constexpr (kernel_ != nk_kernel_dot_k)
        failures += dense_expect_self_recall(restored, make_index, vectors.view(), nodes_count, 16);
    return failures;
}

/** @ref dense_expect_allocation_budget, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
std::size_t test_dense_allocation_budget(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_allocation_budget<index_type_, kernel_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Shedding the frontier past a full pool's radius is lossless: a walk whose frontier
 *  never fills ranks the same nodes, and the shedding frontier never outgrows its reservation. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_frontier_shedding(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                           u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    failures += expect(succeeded(index.reserve({dense_growth_nodes_k, connectivity_16_k, dimensions}).status));
    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k + 1, dimensions, stream_key(seed, "shedding"));
    auto rows = vectors(nk::range(0, dense_growth_nodes_k), nk::slice);
    failures += expect(succeeded(index.insert(rows, {}, context).status));

    // Same query, same seeds, two contexts: one sheds within 4 × ef, the other holds every
    // node and never fills.
    typename index_type_::insert_config_t wide;
    wide.expansion_factor = 64;
    typename index_type_::search_config_t config;
    config.expansion_factor = wide.expansion_factor;
    typename index_type_::traversal_context_t shedding {{}};
    typename index_type_::traversal_context_t unbounded {{}};
    failures += expect(succeeded(index.reserve_traversal(shedding, wide)));
    failures += expect(succeeded(unbounded.frontier.reserve(dense_growth_nodes_k)));
    std::size_t const frontier_capacity = shedding.frontier.capacity();
    auto const query = vectors.view().row(dense_growth_nodes_k).as_vector();
    auto const [shed, shed_status] = index.search(query, shedding, config);
    auto const [kept, kept_status] = index.search(query, unbounded, config);
    failures += expect(succeeded(shed_status) && succeeded(kept_status));
    failures += expect_eq(shed.count, kept.count);
    for (std::size_t rank = 0; rank < shed.count && rank < kept.count; ++rank)
        failures += expect_eq(shedding.results[rank].node_id, unbounded.results[rank].node_id);
    failures += dense_expect_ascending(shedding);
    failures += expect_eq(shedding.frontier.capacity(), frontier_capacity);
    failures += expect_eq(shed.saturated_walks, 0u);
    return failures;
}

/** @ref dense_expect_frontier_shedding, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_frontier_shedding(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_frontier_shedding<index_type_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** A refused re-table saturates the walk, it never fails it: under a frozen budget a
 *  search answers @c success_k with a short ascending ranking and one saturated walk, and
 *  an insertion still publishes its row, wired thinner. The one-wide pool makes the first
 *  expansion's quad the re-table that is refused. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_saturated(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    failures += expect(succeeded(index.reserve({dense_growth_nodes_k + 1, connectivity_16_k, dimensions}).status));
    auto vectors = make_vectors<scalar_t>(dense_growth_nodes_k + 2, dimensions, stream_key(seed, "saturated"));
    typename index_type_::traversal_context_t warm {counting_allocator_t(ledger)};
    auto rows = vectors(nk::range(0, dense_growth_nodes_k), nk::slice);
    failures += expect(succeeded(index.insert(rows, {}, warm).status));

    typename index_type_::insert_config_t narrow;
    narrow.expansion_factor = 1;
    typename index_type_::search_config_t config;
    config.expansion_factor = narrow.expansion_factor;
    typename index_type_::traversal_context_t starved {counting_allocator_t(ledger)};
    failures += expect(succeeded(index.reserve_traversal(starved, narrow)));
    ledger.byte_budget.store(ledger.live_bytes.load());

    auto const [search_result, search_status] = index.search(vectors.view().row(dense_growth_nodes_k).as_vector(),
                                                             starved, config);
    failures += expect_eq(search_status, status_t::success_k);
    failures += expect(search_result.count > 0);
    failures += expect_eq(search_result.saturated_walks, 1u);
    failures += dense_expect_ascending(starved);

    auto extra = vectors(nk::range(dense_growth_nodes_k, dense_growth_nodes_k + 1), nk::slice);
    auto const [insert_result, insert_status] = index.insert(extra, {}, starved, narrow);
    failures += expect_eq(insert_status, status_t::success_k);
    failures += expect_eq(insert_result.saturated_walks, 1u);
    failures += expect_eq(index.size(), dense_growth_nodes_k + 1);
    ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
    return failures;
}

/** @ref dense_expect_saturated, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_saturated(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_saturated<index_type_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

/** Racing a bounded arena: with the budget frozen at exactly two batches of spare
 *  capacity, four uniform threads resolve to exactly two winners & two @c bad_alloc_k
 *  refusals, the winners' IDs land dense & duplicate-free, the ID-space ceiling still
 *  answers @c overflow_k, and a lifted budget makes a refused batch retryable. */
template <typename index_type_, typename make_index_type_>
std::size_t dense_expect_overflow_race(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t failures = 0;
    std::size_t const racing_threads_count = 4, batch_rows_count = 512;
    std::size_t const contested_rows_count = 2 * batch_rows_count;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(racing_threads_count))) return failures + expect(false);

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    auto [capacity, status] = index.reserve({contested_rows_count, connectivity_16_k, dimensions});
    failures += expect(status == status_t::success_k);
    failures += expect(capacity >= contested_rows_count);

    // Pre-fill any chunk-rounding surplus, so exactly two batches of capacity remain.
    std::size_t const prefill_rows_count = capacity - contested_rows_count;
    typename index_type_::traversal_context_t context {{}};
    if (prefill_rows_count) {
        auto prefill_vectors = make_vectors<scalar_t>(prefill_rows_count, dimensions,
                                                      stream_key(seed, "overflow_race", 1));
        failures += expect(index.insert(prefill_vectors.view(), {}, context).status == status_t::success_k);
    }

    auto threads = dense_make_threads<index_type_>(make_index, racing_threads_count, batch_rows_count, dimensions,
                                                   stream_key(seed, "overflow_race"));
    std::vector<node_id_t> assigned_ids(racing_threads_count * batch_rows_count, node_id_unknown_k);
    std::array<status_t, 4> thread_statuses {};
    ledger.byte_budget.store(ledger.live_bytes.load()); // No growth: the arena is the arena.

    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= racing_threads_count) return;
        thread_statuses[thread] = index
                                      .insert(threads.slices[thread].view(),
                                              {assigned_ids.data() + thread * batch_rows_count, batch_rows_count},
                                              threads.contexts[thread])
                                      .status;
        assert_no_locks_held();
    });

    std::size_t winners_count = 0;
    std::size_t refused_thread = racing_threads_count;
    std::vector<node_id_t> winner_ids;
    for (std::size_t thread = 0; thread != racing_threads_count; ++thread) {
        if (thread_statuses[thread] == status_t::success_k) {
            ++winners_count;
            winner_ids.insert(winner_ids.end(), assigned_ids.begin() + thread * batch_rows_count,
                              assigned_ids.begin() + (thread + 1) * batch_rows_count);
        }
        else {
            failures += expect_eq(thread_statuses[thread], status_t::bad_alloc_k);
            refused_thread = thread;
        }
    }
    failures += expect_eq(winners_count, 2u);
    failures += expect_eq(index.size(), capacity);
    failures += expect_contiguous_ids(std::span {winner_ids}, prefill_rows_count);

    // The genuine overflow_k belongs to the ID-space ceiling alone, refused before any
    // counter dance or allocation - a probe only a 32-bit ID space can express.
    using node_id_t = typename index_type_::node_id_t;
    if constexpr (sizeof(node_id_t) == 4)
        failures += expect_eq(index.reserve({std::size_t {1} << 33, connectivity_16_k, dimensions}).status,
                              status_t::overflow_k);

    // A lifted budget makes a refused batch retryable.
    ledger.byte_budget.store(std::numeric_limits<std::ptrdiff_t>::max());
    failures += expect(refused_thread != racing_threads_count);
    failures += expect(index
                           .insert(threads.slices[refused_thread].view(), {}, //
                                   threads.contexts[refused_thread])
                           .status == status_t::success_k);
    failures += expect_eq(index.size(), capacity + batch_rows_count);
    return failures;
}

/** @ref dense_expect_overflow_race, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
std::size_t test_dense_overflow_race(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    std::size_t const failures = dense_expect_overflow_race<index_type_>(make_index, ledger, seed);
    return failures + expect_eq(ledger.live_bytes.load(), 0);
}

#pragma endregion Allocation Budgets

} // namespace unum::usearch::test
