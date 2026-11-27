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
 *  Allocators are tested by construction: a counting allocator flows through the vectors and
 *  neighbors parameters to prove propagation, tallying live bytes into a ledger that must balance
 *  to zero when each scenario's scope closes. Parallel scenarios use ForkUnion's pool over the
 *  harvested topology, and the sharded-index scenarios lean on it, because an executor colocates
 *  each arena with the thread that walks it.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cmath>   // `std::exp`, `std::fabs` - the search and algebra oracles
#include <cstdio>  // `stderr` - the repeat warning
#include <cstdlib> // `std::abs`
#include <cstring> // `std::strlen`, `std::memcmp` - the fetched-row byte comparison

#include <algorithm> // `std::sort`, `std::unique`, `std::all_of`, `std::rotate`
#include <array>     // `std::array`
#include <bit>       // `std::bit_cast` - the ULP distance of the oracle audits; `std::bit_floor` - a cell's ceiling
#include <chrono>    // `std::chrono::microseconds`
#include <memory>    // `std::allocator` - the part path buffer, `std::unique_ptr` - every scenario's index
#include <thread>    // `std::this_thread::sleep_for`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include <forkunion/atomics.hpp>   // `fu::standard_atomic_ref` - the flags and counters threads share
#include <forkunion/forkunion.hpp> // `fu::topology_t`, `fu::pool_t` - the threads a case races
#include <numkong/numkong.h>
#include <numkong/tensor.hpp>

#include "harness.hpp"
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp"        // `ram_executor` - the RAM residency's executor
#include "usearch/utilities/sanitizers.hpp"        // `assert_no_locks_held` - the racing threads' tail check
#include "usearch/utilities/safegraphs_format.hpp" // `save_config_t`, `safegraphs_read_header` - the checkpoints' surface
#include "usearch/dense/ram_metric.hpp"            // `dense::distance`, `dense::norms` - the oracle's kernels
#include "usearch/dense/shard_checkpoints.hpp" // `dense::norms_column_of`, `dense::norms_column_name_k` - the norms column

namespace unum::usearch::test {
using namespace unum::usearch::dense;

/** The key numbered @p number: the number itself, or the @c uuid128_t the bindings render it as. */
template <typename key_type_>
constexpr key_type_ dense_key(u64_t number) noexcept {
    if constexpr (std::is_same_v<key_type_, uuid128_t>) return uuid_of(number);
    else return static_cast<key_type_>(number);
}

/** The number @ref dense_key made @p key from, or zero for a key it never makes. */
template <typename key_type_>
constexpr u64_t dense_key_number(key_type_ const& key) noexcept {
    if constexpr (std::is_same_v<key_type_, uuid128_t>) return identity_of(key).value_or(0);
    else return static_cast<u64_t>(key);
}

/** One distinct key per row, numbered from @p first: what a scenario that never looks keys up
 *  inserts under. */
template <typename index_type_>
std::vector<typename index_type_::key_t> dense_keys(std::size_t count, u64_t first = 1) {
    std::vector<typename index_type_::key_t> keys(count);
    for (std::size_t row = 0; row != count; ++row) keys[row] = dense_key<typename index_type_::key_t>(first + row);
    return keys;
}

#pragma region Constants

/** Storage values per vector in the larger scenarios - NumKong's unit of one
 *  scalar object; small enough to keep serial kernels fast. */
constexpr std::size_t dense_values_k = 16;

/** Storage values per vector in the tiny smoke scenarios. */
constexpr std::size_t dense_tiny_values_k = 8;

/** Logical dimensions spanned by @p values_count storage values of one dtype -
 *  NumKong's @c dimensions_per_value conversion, so every dtype gets the same storage
 *  budget: 16 i8 values hold 16 dimensions, 16 u1x8 values hold 128. */
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

/** The capability ceiling an oracle's kernels match @p index_type_ bit for bit under: the CPU
 *  shard's own quads kit's, or the serial floor for a device shard, whose kernels agree with it. */
template <typename index_type_>
constexpr nk_capability_t dense_reference_capability() noexcept {
    if constexpr (requires { typename index_type_::quads_kit_t; }) return index_type_::quads_kit_t::capability_k;
    else return nk_cap_serial_k;
}

/** Gate: a cell's kit is compiled once its requested ceiling is a bit some ladder actually
 *  picked - a build without the kit must not pass its serial floor under a SIMD name. */
template <nk_dtype_t dtype_, nk_capability_t caps_>
concept a_compiled_dense_cell =
    caps_ == nk_cap_serial_k ||
    std::bit_floor(caps_) == quads_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t::capability_k ||
    std::bit_floor(caps_) == tiles_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t::capability_k;

/** Gate: a cell runs once its serial kit finalizes the kernel and its requested ceiling picked a
 *  kit. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, nk_capability_t caps_>
concept a_testable_dense_cell =
    a_quads_supported_kernel<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k> &&
    a_compiled_dense_cell<dtype_, caps_>;

/** Whether this machine offers every bit a cell's kit and atomic reference declare - both
 *  execute their instructions unconditionally - arming the per-process tile handshake on the way. */
template <nk_capability_t caps_, template <typename> class atomic_ref_>
static bool dense_cell_admitted() noexcept {
    constexpr fu::capabilities_t atomics_k = atomic_ref_<u32_t>::capabilities_k;
    if ((fu::runtime_capabilities() & atomics_k) != atomics_k) return false;
    if constexpr (caps_ != nk_cap_serial_k) {
        nk_capability_t available = 0;
        if (nk_capabilities_enabled_cpu(&available) != nk_success_k || (available & caps_) != caps_) return false;
        if (nk_configure_thread_cpu(available) != nk_success_k) return false;
    }
    return true;
}

/** Why a cell skips whose dtype's serial kernels do not finalize its kernel yet. */
inline constexpr char needs_serial_kernels_k[] = "needs a_quads_supported_kernel";

/** Why a cell skips whose requested kit this build's target does not compile. */
inline constexpr char needs_kit_target_k[] = "needs a_testable_dense_cell";

/** Why a cell skips on a machine lacking the instructions its kit or atomic reference execute. */
inline constexpr char needs_capabilities_k[] = "needs dense_cell_admitted";

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

/** Sampled self-recall over all result candidates: each sampled stored row must surface some
 *  candidate inside the self-match band. Candidate scan by distance is the one criterion: duplicate
 *  ties - binary vectors above all - make the top-1 slot and ID-matching unfair judges of a healthy
 *  graph, and a distance criterion needs no assigned-IDs plumbing, so it survives reloads &
 *  thread-sliced builds unchanged.
 *  Sampling is evenly spaced over @c recall_samples_k rows, or over all of fewer @p rows_count;
 *  callers keep the dot-kernel gate, where "distance" is a similarity and self-matching has no
 *  meaning at all.
 *  The searches walk a fresh context for the index @p make_index builds. */
template <typename index_type_, typename make_index_type_, typename vectors_view_type_>
std::size_t dense_count_self_hits(index_type_& index, make_index_type_ const& make_index, vectors_view_type_ vectors,
                                  std::size_t rows_count) {
    std::size_t const samples_count = min_of(recall_samples_k, rows_count);
    std::size_t hits = 0;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    for (std::size_t sample = 0; sample != samples_count; ++sample) {
        std::size_t const row = sample * (rows_count / samples_count);
        auto const [result, status] = index.search(vectors.row(row).as_vector(), {}, context);
        if (failed(status) || !result.count) continue;
        for (std::size_t match = 0; match != result.count; ++match) {
            scored_node_t const scored = context.results[match];
            if (std::fabs(scored.distance) < dense_self_match_tolerance_k) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

/** The judged form of @ref dense_count_self_hits: the share found must clear the floor. */
template <typename index_type_, typename make_index_type_, typename vectors_view_type_>
void dense_expect_self_recall(index_type_& index, make_index_type_ const& make_index, vectors_view_type_ vectors,
                              std::size_t rows_count) {
    std::size_t const samples_count = min_of(recall_samples_k, rows_count);
    std::size_t const hits = dense_count_self_hits(index, make_index, vectors, rows_count);
    expect_true(hits + recall_margin_k * samples_count >= samples_count * dense_minimal_recall_percent_k / 100.0);
}

/** The ranking in @p context ascends by distance - the candidates' invariant after any walk. */
template <typename context_type_>
void dense_expect_ascending(context_type_ const& context) noexcept {
    for (std::size_t rank = 1; rank < context.results.size(); ++rank)
        expect_true(!(context.results[rank].distance < context.results[rank - 1].distance));
}

/** One caller-built traversal context over its own arena, sized for the shard geometry -
 *  non-movable, since the context's allocator points into the arena beside it. */
template <typename indexes_type_>
struct dense_arena_context {

    /** The backing run every carve of the context comes from. */
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
 *      @c indexes_t, and the @c vendor_prefix_k its reports print, built over the @c stream of the
 *      executor the vendor's entry opened.
 *
 *  A cell is the factory the shared scenarios take: called on a ledger it builds the index on the
 *  stream's device, and its @c context builds a traversal context on the stream.
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
    template <typename api_type_, typename extensions_type_>
    counting_device_allocator(allocation_ledger_t& ledger, api_type_ const& api, extensions_type_ const& extensions,
                              int device) noexcept
        : virtual_memory_(api, extensions, device), ledger_(&ledger) {}
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
            fu::standard_atomic_ref<std::ptrdiff_t>(ledger_->live_bytes)
                .store_add(static_cast<std::ptrdiff_t>(virtual_memory_.committed_bytes() - tallied_bytes_));
            fu::standard_atomic_ref<std::size_t>(ledger_->allocations).store_add(1);
            tallied_bytes_ = virtual_memory_.committed_bytes();
        }
        return status;
    }
    value_type* base() const noexcept { return virtual_memory_.base(); }
    std::size_t reserved_bytes() const noexcept { return virtual_memory_.reserved_bytes(); }
    std::size_t committed_bytes() const noexcept { return virtual_memory_.committed_bytes(); }
    int device() const noexcept { return virtual_memory_.device(); }
    auto const& api() const noexcept { return virtual_memory_.api(); }
    auto const& extensions() const noexcept { return virtual_memory_.extensions(); }
    void share_with(std::bitset<gpu_devices_capacity_k> devices) noexcept { virtual_memory_.share_with(devices); }
    bool shared_with(int device) const noexcept { return virtual_memory_.shared_with(device); }
    status_t grant(std::bitset<gpu_devices_capacity_k> devices) noexcept { return virtual_memory_.grant(devices); }
    void release() noexcept {
        if (ledger_ && tallied_bytes_)
            fu::standard_atomic_ref<std::ptrdiff_t>(ledger_->live_bytes)
                .store_sub(static_cast<std::ptrdiff_t>(tallied_bytes_));
        tallied_bytes_ = 0;
        virtual_memory_.release();
    }
};

} // namespace dense

/** Whether a sharded index runs on ForkUnion threads - the RAM residency. */
template <typename indexes_type_>
concept a_ram_collection =
    std::is_same_v<typename indexes_type_::executor_t, ram_executor<typename indexes_type_::executor_t::threads_t>>;

/** The executor one sharded scenario drives, owned together with whatever it was spawned
 *  over; @c status reports the spawn. Defined per residency, each built from the @c placement
 *  the shared scenarios forward: none on RAM and NVMe, and on VRAM the executor the vendor's
 *  entry opened, whose device the shards and every further executor run on. */
template <typename indexes_type_>
struct dense_indexes_executor;

/** The RAM form: two threads, or @p threads_count, the caller outside them, over a topology of its
 *  own, both released with it, and one traversal context per thread over an arena of its own,
 *  built once and handed to every batched verb, so the executor's own rewinds never reach them. */
template <a_ram_collection indexes_type_>
struct dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;
    using traversal_context_t = typename indexes_type_::traversal_context_t;

    fu_topology_t topology = nullptr;
    executor_t executor;
    status_t status = status_code_t::uninitialized_k;
    growing_buffer<typename executor_t::arena_t, forkunion_allocator_t> arenas {forkunion_allocator_t {}};
    growing_buffer<traversal_context_t, forkunion_allocator_t> traversals {forkunion_allocator_t {}};

    explicit dense_indexes_executor(std::size_t threads_count = 2) noexcept {
        if (fu_topology_new(&topology) != fu_success_k) return;
        auto [spawn_executor, spawn_status] = executor_t::spawn(topology, threads_count);
        status = spawn_status;
        this->executor = std::move(spawn_executor);
        if (failed(status)) return;
        std::size_t const threads = executor.size();
        if (failed(arenas.reserve_exactly(threads)) || failed(traversals.reserve_exactly(threads))) {
            status = status_code_t::out_of_memory_k;
            return;
        }
        for (std::size_t thread = 0; thread != threads; ++thread) {
            arenas.push_back_reserved(
                typename executor_t::arena_t {forkunion_allocator_t {executor.memory_domain(thread)}});
            traversals.push_back_reserved(
                traversal_context_t {arena_allocator<byte_t, forkunion_allocator_t>(arenas[thread])});
        }
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
template <a_ram_collection indexes_type_>
struct dense_indexes_row_driver<indexes_type_> {
    dense_arena_context<indexes_type_> arena;
    status_t status;

    dense_indexes_row_driver(indexes_type_& indexes, dense_indexes_executor<indexes_type_>&) noexcept
        : arena(indexes), status(arena.status) {}
    typename indexes_type_::traversal_context_t& handle() noexcept { return arena.context; }
};

/** The batched verbs through @p driver: its executor, then on RAM the contexts it keeps for that
 *  executor across calls, which only the RAM collection's verbs take. Each ranking is written into
 *  @p keys, @p scores and @p counts, passed as one @c ranked_matches_span. */
template <typename indexes_type_, typename queries_type_, typename score_type_>
auto drive_search(indexes_type_& indexes, queries_type_ const& queries,
                  nk::matrix_span<typename indexes_type_::key_t> keys, nk::matrix_span<score_type_> scores,
                  std::span<std::size_t> counts, typename indexes_type_::search_config_t config,
                  dense_indexes_executor<indexes_type_>& driver) noexcept {
    ranked_matches_span<typename indexes_type_::key_t, score_type_> const matches {keys, scores, counts};
    if constexpr (a_ram_collection<indexes_type_>)
        return indexes.search(queries, matches, config, driver.executor, to_writable_span(driver.traversals));
    else return indexes.search(queries, matches, config, driver.executor);
}
template <typename indexes_type_, typename queries_type_, typename score_type_, typename predicate_type_>
auto drive_search_if(indexes_type_& indexes, queries_type_ const& queries,
                     nk::matrix_span<typename indexes_type_::key_t> keys, nk::matrix_span<score_type_> scores,
                     std::span<std::size_t> counts, typename indexes_type_::search_config_t config,
                     predicate_type_&& predicate, dense_indexes_executor<indexes_type_>& driver) noexcept {
    ranked_matches_span<typename indexes_type_::key_t, score_type_> const matches {keys, scores, counts};
    if constexpr (a_ram_collection<indexes_type_>)
        return indexes.search_if(queries, matches, config, predicate, driver.executor,
                                 to_writable_span(driver.traversals));
    else return indexes.search_if(queries, matches, config, predicate, driver.executor);
}
template <typename indexes_type_, typename queries_type_, typename heads_type_>
auto drive_indexer_search(indexes_type_& indexes, queries_type_ const& queries, heads_type_ const& heads,
                          nk::matrix_span<typename indexes_type_::key_t> keys, nk::matrix_span<distance_t> scores,
                          std::span<std::size_t> counts, typename indexes_type_::search_config_t config,
                          dense_indexes_executor<indexes_type_>& driver) noexcept {
    ranked_matches_span<typename indexes_type_::key_t, distance_t> const matches {keys, scores, counts};
    if constexpr (a_ram_collection<indexes_type_>)
        return indexes.search(queries, heads, matches, config, driver.executor, to_writable_span(driver.traversals));
    else return indexes.search(queries, heads, matches, config, driver.executor);
}

/** The tape verbs through @p driver, over a host tape: RAM and NVMe towers write it where it is,
 *  and a VRAM driver stages it on its device, where those towers write, and copies it home. */
template <typename indexes_type_, typename left_type_, typename product_type_>
auto drive_multiply(indexes_type_& indexes, left_type_ const& left,
                    products_tape<typename indexes_type_::key_t, product_type_> products, std::span<std::size_t> scored,
                    typename indexes_type_::search_config_t config,
                    dense_indexes_executor<indexes_type_>& driver) noexcept {
    if constexpr (a_ram_collection<indexes_type_>)
        return indexes.multiply(left, products, scored, config, driver.executor, to_writable_span(driver.traversals));
    else if constexpr (requires { indexes_type_::vendor_k; })
        return driver.on_device(products,
                                [&](products_tape<typename indexes_type_::key_t, product_type_> staged) noexcept {
                                    return indexes.multiply(left, staged, scored, config, driver.executor);
                                });
    else return indexes.multiply(left, products, scored, config, driver.executor);
}

template <typename indexes_type_, typename rows_type_, typename score_type_>
auto drive_maxsim(indexes_type_& indexes, rows_type_ const& rows, std::span<std::size_t const> query_offsets,
                  std::span<std::size_t> scored, products_tape<typename indexes_type_::key_t, score_type_> matches,
                  typename indexes_type_::maxsim_config_t maxsim_config,
                  dense_indexes_executor<indexes_type_>& driver) noexcept {
    if constexpr (a_ram_collection<indexes_type_>)
        return indexes.maxsim(rows, query_offsets, scored, matches, maxsim_config, driver.executor,
                              to_writable_span(driver.traversals));
    else if constexpr (requires { indexes_type_::vendor_k; })
        return driver.on_device(
            matches, [&](products_tape<typename indexes_type_::key_t, score_type_> staged) noexcept {
                return indexes.maxsim(rows, query_offsets, scored, staged, maxsim_config, driver.executor);
            });
    else return indexes.maxsim(rows, query_offsets, scored, matches, maxsim_config, driver.executor);
}
template <typename plan_type_, typename indexes_type_, typename... arguments_types_>
status_t drive_run(plan_type_& plan, dense_indexes_executor<indexes_type_>& driver,
                   arguments_types_&&... arguments) noexcept {
    if constexpr (a_ram_collection<indexes_type_>)
        return plan.run(driver.executor, to_writable_span(driver.traversals), arguments...);
    else return plan.run(driver.executor, arguments...);
}
template <typename plan_type_, typename indexes_type_>
auto drive_finish(plan_type_& plan, dense_indexes_executor<indexes_type_>& driver) noexcept {
    if constexpr (a_ram_collection<indexes_type_>)
        return plan.finish(driver.executor, to_writable_span(driver.traversals));
    else return plan.finish(driver.executor);
}
template <typename indexes_type_, typename queries_type_, typename lookup_type_, typename outputs_type_>
auto drive_attend(indexes_type_& indexes, queries_type_ const& queries, std::size_t head_count, lookup_type_&& lookup,
                  f32_t score_scale, outputs_type_ const& outputs, std::span<softmax_state_t> carry,
                  typename indexes_type_::search_config_t config,
                  dense_indexes_executor<indexes_type_>& driver) noexcept {
    if constexpr (a_ram_collection<indexes_type_>)
        return indexes.attend(queries, head_count, lookup, score_scale, outputs, carry, config, driver.executor,
                              to_writable_span(driver.traversals));
    else return indexes.attend(queries, head_count, lookup, score_scale, outputs, carry, config, driver.executor);
}

/** Every row of @p view widened to @c f32 into @p widened, shaped as @p view - exact for every
 *  stored scalar kind. */
template <typename scalar_type_>
void dense_widened_rows(nk::matrix_view<scalar_type_> view, nk_dtype_t dtype, nk::matrix_span<f32_t> widened) {
    for (std::size_t row = 0; row != widened.extent(0); ++row) {
        [[maybe_unused]] nk_status_t const cast = elementwise_ladder<nk_cap_any_k>::kit_t::cast(
            view.byte_data() + row * view.stride_bytes(0), dtype, widened.row(row).data(), nk_f32_k, widened.extent(1),
            nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    }
}

/** An exact dot beside the magnitude a float accumulation's rounding scales with. */
struct dense_dot_reference_t {

    /** The dot, exact in @c f64 for every widened pair of rows. */
    f64_t dot;

    /** The sum of absolute products - what bounds a float accumulation's error. */
    f64_t magnitude;
};

/** The exact dot of two widened rows. */
template <typename widened_type_>
dense_dot_reference_t dense_widened_dot(widened_type_ const* first, widened_type_ const* second,
                                        std::size_t columns) noexcept {
    dense_dot_reference_t oracle {0, 0};
    for (std::size_t column = 0; column != columns; ++column) {
        f64_t const product = static_cast<f64_t>(first[column]) * second[column];
        oracle.dot += product;
        oracle.magnitude += std::fabs(product);
    }
    return oracle;
}

/** Oracle dots of every query row against every stored row into @p dots, @b [queries,rows]. */
inline void dense_dots_reference(nk::matrix_view<f32_t> widened_queries, nk::matrix_view<f32_t> widened_rows,
                                 nk::matrix_span<dense_dot_reference_t> dots) {
    for (std::size_t query = 0; query != dots.extent(0); ++query)
        for (std::size_t row = 0; row != dots.extent(1); ++row)
            dots(query, row) = dense_widened_dot(widened_queries.row(query).data(), widened_rows.row(row).data(),
                                                 widened_rows.extent(1));
}

/** The @c f32 widening of @p query into the one row of @p widened. */
template <typename scalar_type_>
void dense_widened_query(nk::vector_view<scalar_type_> query, nk::matrix_span<f32_t> widened) {
    dense_widened_rows(
        nk::matrix_view<scalar_type_> {reinterpret_cast<scalar_type_ const*>(query.byte_data()), 1, query.size()},
        scalar_type_::dtype(), widened);
}

#pragma endregion Shared Helpers

#pragma region Foundations

/** Guards: an insert succeeds without a reservation, and every other geometry is refused, on a
 *  fresh index per scenario. */
template <typename index_type_, typename make_index_type_>
void dense_expect_guards(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger, {dimensions, connectivity_16_k});
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(3, dimensions, stream_key(seed, "guards"));
    nk::matrix<scalar_t> wider = make_vectors<scalar_t>(3, 2 * dimensions, stream_key(seed, "guards"));
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(3);

    auto const [search_result, search_status] = index.search(vectors.view().row(0).as_vector(), {}, context);
    expect_true(search_status == status_code_t::success_k && search_result.count == 0);
    expect_equal(index.insert(to_span(keys), wider.view(), {}, context).status, status_code_t::config_mismatch_k);
    auto const [insert_result, insert_status] = index.insert(to_span(keys), vectors.view(), {}, context);
    expect_true(insert_status == status_code_t::success_k && insert_result.new_size == 3);

    // Geometries no row can take are refused at the first growth, and a ceiling is never
    // positional.
    static_assert(!std::is_constructible_v<typename index_type_::reserve_capacity_t, std::size_t, connectivity_t>);
    std::unique_ptr<index_type_> const dimensionless_ptr = make_index(ledger, {0, connectivity_16_k});
    expect_equal(dimensionless_ptr->reserve({64}).status, status_code_t::config_mismatch_k);
    std::unique_ptr<index_type_> const uneven_ptr = make_index(ledger, {dimensions, static_cast<connectivity_t>(24)});
    expect_equal(uneven_ptr->reserve({64}).status, status_code_t::config_mismatch_k);

    auto [capacity, reserve_status] = index.reserve({64});
    expect_true(reserve_status == status_code_t::success_k && capacity >= 64);
    expect_equal(capacity, index.capacity());
    auto [smaller_capacity, smaller_status] = index.reserve({32});
    expect_true(smaller_status == status_code_t::success_k && smaller_capacity == capacity);
}

/** @ref dense_expect_guards, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
void test_dense_guards(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_guards<index_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** The classic smoke scenario: three vectors in, each finds itself first, and a narrower
 *  query is refused. The dot kernel maximizes similarity instead of minimizing distance, so
 *  self-nearest is only asserted for the symmetric kernels. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_three_vectors(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::unique_ptr<index_type_> const index_ptr = make_index(
        ledger, {dense_dimensions_for<scalar_t>(dense_tiny_values_k), connectivity_16_k});
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(3, dense_dimensions_for<scalar_t>(dense_tiny_values_k),
                                                          stream_key(seed, "three_vectors"));

    expect_true(index.reserve({8}).status == status_code_t::success_k);
    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(3);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(3);
    auto const [insert_result, insert_status] = index.insert(to_span(keys), vectors.view(), assigned_ids.span(),
                                                             context);
    expect_true(insert_status == status_code_t::success_k);
    expect_equal(insert_result.new_size, 3u);
    expect_equal(index.size(), 3u);

    for (std::size_t row = 0; row != 3; ++row) {
        auto const [search_result, search_status] = index.search(vectors.view().row(row).as_vector(), {}, context);
        expect_true(search_status == status_code_t::success_k && search_result.count >= 1);
        if constexpr (kernel_ != nk_kernel_dot_k) {
            scored_node_t const nearest = context.results[0];
            expect_equal(nearest.node_id, assigned_ids[row]);
            expect_true(std::fabs(nearest.distance) < dense_self_match_tolerance_k);
        }

        // A query narrower than the index's width is refused, never read past its end.
        nk::matrix<scalar_t> narrower = make_vectors<scalar_t>(
            1, dense_dimensions_for<scalar_t>(dense_tiny_values_k / 2), stream_key(seed, "three_vectors"));
        expect_equal(index.search(narrower.view().row(0).as_vector(), {}, context).status,
                     status_code_t::config_mismatch_k);
    }
}

/** @ref dense_expect_three_vectors, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_three_vectors(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_three_vectors<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Growth across several doubling chunks, then sampled self-recall on the big graph,
 *  with telemetry accounting sanity checks along the way. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_growth_and_recall(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(
        dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k), stream_key(seed, "growth"));

    expect_true(index.reserve({16}).status == status_code_t::success_k);
    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(dense_growth_nodes_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
    auto const [result, status] = index.insert(to_span(keys), vectors.view(), assigned_ids.span(), context);
    expect_true(status == status_code_t::success_k);
    expect_equal(result.new_size, dense_growth_nodes_k);
    expect_equal(index.size(), dense_growth_nodes_k);
    expect_true(result.computed_distances > 0 && result.visited_members > 0);
    expect_true(result.computed_distances_in_refines <= result.computed_distances);
    expect_true(result.computed_distances_in_reverse_refines <= result.computed_distances);
    // The context was fresh, so the batch delta must equal the absolute counter - the
    // belt against a dropped snapshot or accumulation in the restarts telemetry.
    if constexpr (requires { context.restarts; }) expect_equal(result.restarts, context.restarts);

    if constexpr (kernel_ != nk_kernel_dot_k)
        dense_expect_self_recall(index, make_index, vectors.view(), dense_growth_nodes_k);
    // A single query keeps its whole expansion, past any inline staging.
    typename index_type_::search_config_t wide {};
    wide.expansion_factor = 200;
    auto const [wide_found, wide_status] = index.search(vectors.view().row(0).as_vector(), wide, context);
    expect_true(succeeded(wide_status) && wide_found.count == 200 && context.results.size() == 200);
    dense_expect_ascending(context);
}

/** @ref dense_expect_growth_and_recall, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_growth_and_recall(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_growth_and_recall<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Foundations

#pragma region Search

/**
 *  @brief The predicated search contract, audited against an exact oracle.
 *
 *  The oracle scans every stored vector through the same free kernels the index picked - its quads
 *  kit - so its distances are bit-identical to the traversal's and the checks can be exact where
 *  approximation permits: a trivially-true predicate must reproduce @c search verbatim; every
 *  reported ID must satisfy its predicate and hold the oracle's distance for that ID, bit-for-bit;
 *  the per-node and batched callable shapes must agree; excluded self-matches must never surface;
 *  recall of the true filtered nearest is held to the same bar as the unfiltered suite across three
 *  selectivity decades - every other node, one in 16, and one in 64; and an impossible predicate
 *  drains legally.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_predicated_search(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using oracle_kit_t = typename index_type_::quads_kit_t;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(
        dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k), stream_key(seed, "predicated"));

    expect_true(index.reserve({dense_growth_nodes_k}).status == status_code_t::success_k);
    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(dense_growth_nodes_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
    expect_true(index.insert(to_span(keys), vectors.view(), assigned_ids.span(), context).status ==
                status_code_t::success_k);

    // The oracle: distances of one query against the whole store, through the very 4-way
    // kernels the traversal dispatches. A wrong ID or vector diverges by orders of
    // magnitude, which is what this audit exists to catch.
    auto oracle_distance = [&](std::size_t query_row, node_id_t node_id) noexcept {
        auto query_view = vectors.view().row(query_row).as_vector();
        auto node_view = vectors.view().row(node_id).as_vector();
        auto node_norms = norms<oracle_kit_t>(node_view);
        auto lane_distances = distances<oracle_kit_t>(kernel_, query_view, norms<oracle_kit_t>(query_view),
                                                      {node_view, node_view, node_view, node_view},
                                                      {node_norms, node_norms, node_norms, node_norms});
        return lane_distances[0];
    };
    // The stochastic landmark seeding rides the context's persistent generator, so exact
    // walk parity is only defined between FRESH contexts with identical generator state.
    std::size_t const query_row = dense_growth_nodes_k / 2;
    auto query = vectors.view().row(query_row).as_vector();
    typename index_type_::traversal_context_t unfiltered_context {{}};
    auto const [unfiltered_result, unfiltered_status] = index.search(query, {}, unfiltered_context);
    expect_true(unfiltered_status == status_code_t::success_k && unfiltered_result.count > 0);
    scored_node_t const* reference = unfiltered_context.results.data();

    // The trivially-true predicate walks and reports exactly like the unfiltered sibling.
    auto all_pass = quad_predicate([](node_id_t) noexcept { return admission_t::admit_k; });
    typename index_type_::traversal_context_t all_pass_context {{}};
    auto const [with_true, with_true_status] = index.search_if(query, {}, all_pass, all_pass_context);
    expect_true(with_true_status == status_code_t::success_k);
    expect_equal(with_true.count, unfiltered_result.count);
    for (std::size_t match = 0; match != with_true.count && match != unfiltered_result.count; ++match) {
        expect_equal(all_pass_context.results[match].node_id, reference[match].node_id);
        expect_true(all_pass_context.results[match].distance == reference[match].distance);
    }

    // The batched shape must agree with the per-node shape on the same walk.
    auto even_only = quad_predicate(
        [](node_id_t node_id) noexcept { return node_id % 2 == 0 ? admission_t::admit_k : admission_t::reject_k; });
    auto even_batched = [](std::span<node_id_t const> nodes, std::span<admission_t> admissions) noexcept {
        for (std::size_t lane = 0; lane != nodes.size(); ++lane)
            admissions[lane] = nodes[lane] % 2 == 0 ? admission_t::admit_k : admission_t::reject_k;
    };
    typename index_type_::traversal_context_t scalar_context {{}}, batched_context {{}};
    auto const [with_scalar, with_scalar_status] = index.search_if(query, {}, even_only, scalar_context);
    auto const [with_batched, with_batched_status] = index.search_if(query, {}, even_batched, batched_context);
    expect_true(with_scalar_status == status_code_t::success_k && with_scalar.count > 0);
    expect_true(with_batched_status == status_code_t::success_k);
    expect_equal(with_batched.count, with_scalar.count);
    for (std::size_t match = 0; match != with_batched.count && match != with_scalar.count; ++match)
        expect_equal(batched_context.results[match].node_id, scalar_context.results[match].node_id);

    // Three selectivity decades over sampled queries: every reported ID satisfies its
    // predicate and holds the oracle's distance bit-for-bit; distances ascend; a query
    // whose own ID is excluded never surfaces itself. Even samples query a matching row -
    // the filtered analog of the suite's self-recall bar, scored per decade on the two
    // denser ones; odd samples query an excluded row, auditing admission purity. The
    // sparsest decade is exactness-audited only: recall there belongs to the future
    // cardinality-driven brute-force crossover, not the graph walk.
    std::size_t const moduli[] = {2, 16, 64};
    std::size_t hits = 0, recall_samples = 0;
    error_stats_t distance_errors;
    for (std::size_t modulus : moduli) {
        auto matches = quad_predicate([modulus](node_id_t node_id) noexcept {
            return node_id % modulus == 0 ? admission_t::admit_k : admission_t::reject_k;
        });
        for (std::size_t sample = 0; sample != recall_samples_k; ++sample) {
            std::size_t sampled_row = ((sample * 641 + modulus) % dense_growth_nodes_k) / modulus * modulus;
            if (sample % 2) sampled_row += 1; // An excluded neighbor of a matching row.
            typename index_type_::traversal_context_t sampled_context {{}};
            auto const [sampled_result, sampled_status] = index.search_if(vectors.view().row(sampled_row).as_vector(),
                                                                          {}, matches, sampled_context);
            bool const self_matches = sampled_row % modulus == 0;
            // The walk routes on until its matches fill the breadth or none is left unvisited.
            expect_true(sampled_status == status_code_t::success_k);
            expect_equal(sampled_result.count, min_of(default_expansion_factor(connectivity_16_k),
                                                      divide_round_up(dense_growth_nodes_k, modulus)));
            scored_node_t const* results = sampled_context.results.data();
            bool self_found = false;
            for (std::size_t match = 0; match != sampled_result.count; ++match) {
                expect_true(results[match].node_id % modulus == 0);
                distance_errors.add(results[match].distance, oracle_distance(sampled_row, results[match].node_id));
                if (match) expect_true(results[match].distance >= results[match - 1].distance);
                if (!self_matches) expect_true(results[match].node_id != assigned_ids[sampled_row]);
                self_found |= results[match].node_id == assigned_ids[sampled_row];
            }
            if constexpr (kernel_ != nk_kernel_dot_k)
                if (self_matches && modulus != 64) hits += self_found, ++recall_samples;
        }
    }
    if constexpr (kernel_ != nk_kernel_dot_k)
        expect_true(hits + recall_margin_k * recall_samples >= recall_samples * dense_minimal_recall_percent_k / 100.0);
    expect_errors("predicated distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  distance_errors, 0);

    // An impossible predicate drains legally: empty result set, successful status.
    auto none_pass = quad_predicate([](node_id_t) noexcept { return admission_t::reject_k; });
    typename index_type_::traversal_context_t none_context {{}};
    auto const [with_none, with_none_status] = index.search_if(query, {}, none_pass, none_context);
    expect_true(with_none_status == status_code_t::success_k);
    expect_equal(with_none.count, std::size_t {0});
}

/** @ref dense_expect_predicated_search, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_predicated_search(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_predicated_search<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Single-index scans against the one-to-one oracle of @ref test_dense_indexes_scan: over a store
 *  spanning two chunks, each query ranks its true nearest at their own distances, a predicate of
 *  either shape narrows the ranking to the nodes it admits, and a breadth wider than the store
 *  returns every row. A device index takes no predicate, and the serial kit judges its ranking. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_scan(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using vector_view_t = typename index_type_::vector_view_t;
    using search_config_t = typename index_type_::search_config_t;
    using candidates_count_t = typename index_type_::candidates_count_t;
    constexpr bool device_k = requires { make_index.context(); };
    using oracle_kit_t = typename quads_ladder<scalar_t, dense_reference_capability<index_type_>()>::kit_t;
    constexpr std::size_t rows_count = dense_growth_nodes_k, wanted = 10, wide = rows_count + 5;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "scan"));
    expect_true(succeeded(index.reserve({rows_count}).status));
    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(rows_count);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(rows_count);
    expect_true(succeeded(index.insert(to_span(keys), vectors.view(), assigned_ids.span(), context).status));
    nk::vector<std::size_t> row_of_node = make_vector<std::size_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) row_of_node[assigned_ids[row]] = row;

    auto all = [](node_id_t) noexcept { return true; };
    auto even = [](node_id_t node) noexcept { return node % 2 == 0; };
    auto admit_even = quad_predicate(
        [&](node_id_t node) noexcept { return even(node) ? admission_t::admit_k : admission_t::reject_k; });
    auto admit_even_batched = [&](std::span<node_id_t const> nodes, std::span<admission_t> admissions) noexcept {
        for (std::size_t lane = 0; lane != nodes.size(); ++lane)
            admissions[lane] = even(nodes[lane]) ? admission_t::admit_k : admission_t::reject_k;
    };
    // The ranking left in `context.results` against the oracle's, over the nodes `admits` admits.
    error_stats_t scan_errors;
    nk::vector<distance_t> admitted_distances = make_vector<distance_t>(rows_count);
    auto expect_ranking = [&](vector_view_t query, std::size_t breadth, auto admits) {
        auto const query_norms = norms<oracle_kit_t>(query);
        auto oracle = [&](node_id_t node) noexcept {
            vector_view_t const row = vectors.view().row(row_of_node[node]).as_vector();
            // The serial kits have no squared Euclidean, so the oracle squares their Euclidean.
            if constexpr (kernel_ == nk_kernel_sqeuclidean_k) {
                distance_t const root = distance<oracle_kit_t>(nk_kernel_euclidean_k, query, query_norms, row,
                                                               norms<oracle_kit_t>(row));
                return root * root;
            }
            else return distance<oracle_kit_t>(kernel_, query, query_norms, row, norms<oracle_kit_t>(row));
        };
        std::size_t admitted_count = 0;
        for (node_id_t node = 0; node != rows_count; ++node)
            if (admits(node)) admitted_distances[admitted_count++] = oracle(node);
        std::span<distance_t> const admitted = to_writable_span(admitted_distances).first(admitted_count);
        std::sort(admitted.begin(), admitted.end());
        std::size_t const kept = min_of(breadth, admitted.size());
        expect_equal(context.results.size(), kept);
        for (std::size_t rank = 0; rank != min_of(context.results.size(), kept); ++rank) {
            scored_node_t const match = context.results[rank];
            expect_true(admits(match.node_id));
            scan_errors.add(match.distance, admitted[rank]);
            scan_errors.add(match.distance, oracle(match.node_id));
        }
        return;
    };

    search_config_t const scan {.expansion_factor = static_cast<candidates_count_t>(wanted),
                                .traversal = traversal_t::scan_k};
    for (std::size_t sample = 0; sample != 4; ++sample) {
        vector_view_t const query = vectors.view().row(sample * (rows_count / 4) + 1).as_vector();
        auto const [found, status] = index.search(query, scan, context);
        expect_true(succeeded(status));
        expect_equal(found.count, context.results.size());
        expect_equal(found.computed_distances, rows_count);
        expect_ranking(query, wanted, all);
        if constexpr (!device_k) {
            expect_true(succeeded(index.search_if(query, scan, admit_even, context).status));
            expect_ranking(query, wanted, even);
            expect_true(succeeded(index.search_if(query, scan, admit_even_batched, context).status));
            expect_ranking(query, wanted, even);
        }
    }
    search_config_t const wide_scan {.expansion_factor = static_cast<candidates_count_t>(wide),
                                     .traversal = traversal_t::scan_k};
    vector_view_t const query = vectors.view().row(0).as_vector();
    expect_true(succeeded(index.search(query, wide_scan, context).status));
    expect_ranking(query, wide, all);
    expect_errors("scan distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, scan_errors, 0);
}

/** @ref dense_expect_scan, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_scan(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_scan<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** A float dot against its oracle, within accumulation rounding. */
template <typename dot_result_type_>
void dense_expect_dot(dot_result_type_ found, dense_dot_reference_t oracle) {
    expect_true(std::fabs(static_cast<f64_t>(found) - oracle.dot) <= 1e-4 * oracle.magnitude + 1e-6);
}

/** The @p picks are the exact top <tt>picks.size()</tt> rows of @p keys by ⟨query, key⟩, as an exhaustive
 *  walk finds them. */
template <typename scalar_type_, typename row_type_>
void dense_expect_top_logits(nk::matrix_view<scalar_type_> keys, nk::vector_view<scalar_type_> query,
                             std::span<row_type_ const> picks) {
    std::size_t const rows_count = keys.extent(0);
    if (!expect_true(!picks.empty() && picks.size() <= rows_count)) return;
    nk::matrix<f32_t> widened_query = make_matrix<f32_t>(1, keys.extent(1));
    nk::matrix<f32_t> widened_keys = make_matrix<f32_t>(rows_count, keys.extent(1));
    nk::matrix<dense_dot_reference_t> dots = make_matrix<dense_dot_reference_t>(1, rows_count);
    dense_widened_query(query, widened_query.span());
    dense_widened_rows(keys, scalar_type_::dtype(), widened_keys.span());
    dense_dots_reference(widened_query.view(), widened_keys.view(), dots.span());
    nk::vector<f64_t> ranked = make_vector<f64_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) ranked[row] = dots(0, row).dot;
    std::span<f64_t> const descending = to_writable_span(ranked);
    std::sort(descending.rbegin(), descending.rend());
    f64_t const threshold = ranked[picks.size() - 1];

    std::vector<bool> found(rows_count);
    for (row_type_ const pick : picks) {
        std::size_t const row = static_cast<std::size_t>(pick);
        expect_true(row < rows_count && !found[row]);
        if (row >= rows_count || found[row]) continue;
        found[row] = true;
        expect_true(dots(0, row).dot + 1e-4 * dots(0, row).magnitude + 1e-6 >= threshold);
    }
}

/** What every indexer test feeds its index: zero-centered heads under signed weights, so the ReLU clamps some keys. */
template <typename scalar_type_>
struct dense_indexer_inputs {
    static constexpr std::size_t heads_count_k = 3;

    /** The stored rows widened to @c f32, @b [rows,dimensions]. */
    nk::matrix<f32_t> widened;

    /** Every query's heads widened, then as stored, @b [queries·heads,dimensions]. */
    nk::matrix<f32_t> heads_widened;
    nk::matrix<scalar_type_> heads_stored;

    /** Every query's head weights, @b [queries,heads]. */
    nk::matrix<f32_t> weights;

    /** The indexer heads of every query and their weights. */
    indexer_heads<scalar_type_> heads() const noexcept { return {heads_stored.view(), weights.view()}; }
};

/** Builds @ref dense_indexer_inputs over @p vectors for @p queries_count queries. */
template <typename scalar_type_>
dense_indexer_inputs<scalar_type_> dense_make_indexer_inputs(nk::matrix_view<scalar_type_> vectors,
                                                             std::size_t queries_count) {
    using inputs_t = dense_indexer_inputs<scalar_type_>;
    std::size_t const dimensions = vectors.extent(1), heads_rows = queries_count * inputs_t::heads_count_k;
    inputs_t inputs {make_matrix<f32_t>(vectors.extent(0), dimensions), make_matrix<f32_t>(heads_rows, dimensions),
                     make_matrix<scalar_type_>(heads_rows, dimensions),
                     make_matrix<f32_t>(queries_count, inputs_t::heads_count_k)};
    dense_widened_rows(vectors, scalar_type_::dtype(), inputs.widened.span());

    // Every head is a stored row shifted from {0..7} to {-4..3}, exact in every floating kind.
    for (std::size_t head_row = 0; head_row != heads_rows; ++head_row)
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            inputs.heads_widened(head_row, dimension) = inputs.widened(head_row + 1, dimension) - 4;
    [[maybe_unused]] nk_status_t const narrowed = elementwise_ladder<nk_cap_any_k>::kit_t::cast(
        inputs.heads_widened.data(), nk_f32_k, inputs.heads_stored.data(), scalar_type_::dtype(),
        heads_rows * dimensions, nullptr);
    assert(narrowed == nk_success_k && "The serial tier casts every dtype pair");
    std::array<f32_t, inputs_t::heads_count_k> const head_weights {0.75f, -0.5f, 0.25f};
    for (std::size_t query = 0; query != queries_count; ++query)
        for (std::size_t head = 0; head != inputs_t::heads_count_k; ++head)
            inputs.weights(query, head) = head_weights[head];
    return inputs;
}

/** Σₕ wₕ·ReLU(⟨qₕ, k⟩) of stored row @p row against the heads of query @p query, in @c f64. */
template <typename scalar_type_>
dense_dot_reference_t dense_indexer_score(dense_indexer_inputs<scalar_type_> const& inputs, std::size_t query,
                                          std::size_t row) noexcept {
    using inputs_t = dense_indexer_inputs<scalar_type_>;
    dense_dot_reference_t score {0, 0};
    for (std::size_t head = 0; head != inputs_t::heads_count_k; ++head) {
        std::size_t const head_row = query * inputs_t::heads_count_k + head;
        f64_t const weight = inputs.weights(query, head);
        dense_dot_reference_t const oracle = dense_widened_dot(inputs.heads_widened.view().row(head_row).data(),
                                                               inputs.widened.view().row(row).data(),
                                                               inputs.widened.extent(1));
        score.dot += weight * max_of(oracle.dot, f64_t {});
        score.magnitude += std::fabs(weight) * oracle.magnitude;
    }
    return score;
}

/** Every returned distance is 1 − the f64 indexer score of its row, and the rows are the exact top
 *  @c rows.size(); @p ranked is the caller's scratch of one score per stored row. */
template <typename scalar_type_, typename row_type_>
void dense_expect_indexer_matches(dense_indexer_inputs<scalar_type_> const& inputs, std::size_t query,
                                  nk::vector_view<row_type_> rows, nk::vector_view<distance_t> distances,
                                  std::span<f64_t> ranked) {
    std::size_t const rows_count = inputs.widened.extent(0);
    if (!expect_true(!rows.empty() && rows.size() <= rows_count && distances.size() == rows.size())) return;
    for (std::size_t row = 0; row != rows_count; ++row) ranked[row] = dense_indexer_score(inputs, query, row).dot;
    std::sort(ranked.rbegin(), ranked.rend());
    f64_t const threshold = ranked[rows.size() - 1];

    std::vector<bool> found(rows_count);
    for (std::size_t rank = 0; rank != rows.size(); ++rank) {
        std::size_t const row = dense_key_number(rows[rank]);
        expect_true(row < rows_count && !found[row]);
        if (row >= rows_count || found[row]) continue;
        found[row] = true;
        dense_dot_reference_t const score = dense_indexer_score(inputs, query, row);
        dense_expect_dot(1 - static_cast<f64_t>(distances[rank]), score);
        expect_true(score.dot + 1e-4 * score.magnitude + 1e-6 >= threshold);
    }
}

/** Indexer search on one index under an exhaustive walk, every node addressing its own row. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_indexer_search(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const nodes_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexer"));
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);
    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return void(expect_true(false));
    expect_contiguous_ids(to_writable_span(assigned_ids), 0);

    dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
    typename index_type_::search_config_t const exhaustive {
        static_cast<typename index_type_::candidates_count_t>(nodes_count)};
    nk::vector<node_id_t> node_ids = make_vector<node_id_t>(wanted);
    nk::vector<distance_t> distances = make_vector<distance_t>(wanted);
    nk::vector_view<node_id_t> const ranked_ids = node_ids.view();
    nk::vector_view<distance_t> const ranked_distances = distances.view();
    nk::vector<f64_t> ranked = make_vector<f64_t>(nodes_count);
    for (std::size_t query = 0; query != queries_count; ++query) {
        auto const [searched, status] = index.search(vectors.view().row(query).as_vector(), inputs.heads().row(query),
                                                     exhaustive, context);
        expect_true(status == status_code_t::success_k);
        std::size_t const kept = min_of(searched.count, wanted);
        for (std::size_t rank = 0; rank != kept; ++rank)
            node_ids[rank] = context.results[rank].node_id, distances[rank] = context.results[rank].distance;
        dense_expect_indexer_matches(inputs, query, ranked_ids[nk::range(std::size_t {0}, kept)],
                                     ranked_distances[nk::range(std::size_t {0}, kept)], to_writable_span(ranked));
    }
}

/** @ref dense_expect_indexer_search for the dot kernel over floating-point keys, then the ledger must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_indexer_search(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<typename index_type_::scalar_t>()) return;
    else {
        allocation_ledger_t ledger;
        dense_expect_indexer_search<index_type_, kernel_>(make_index, ledger, seed);
        return void(expect_equal(ledger.live_bytes, 0));
    }
}

/** Sharded indexer search under an exhaustive walk: keys equal their rows, so the merged top of every query is
 *  judged against the same @c f64 oracle whichever shard scored it. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_indexer_search(u64_t seed) {
    using scalar_t = typename indexes_type_::scalar_t;
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<scalar_t>()) return;
    else {
        using key_t = typename indexes_type_::key_t;
        constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return void(expect_true(false));
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes)));
        expect_true(succeeded(indexes.reserve({rows_count}).status));
        nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                    stream_key(seed, "indexes_indexer"));
        nk::vector<key_t> keys = make_vector<key_t>(rows_count);
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row);
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

        nk::matrix_view<scalar_t> const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
        typename indexes_type_::search_config_t const exhaustive {
            static_cast<typename indexes_type_::index_t::candidates_count_t>(rows_count)};
        nk::matrix<key_t> found_keys = make_matrix<key_t>(queries_count, wanted);
        nk::matrix<distance_t> found_distances = make_matrix<distance_t>(queries_count, wanted);
        std::array<std::size_t, queries_count> counts;
        auto const [searched, status] = drive_indexer_search(indexes, queries, inputs.heads(), found_keys.span(),
                                                             found_distances.span(), std::span {counts}, exhaustive,
                                                             driver);
        expect_true(succeeded(status));
        nk::matrix_view<key_t> const ranked_keys = found_keys.view();
        nk::matrix_view<distance_t> const ranked_distances = found_distances.view();
        nk::vector<f64_t> ranked = make_vector<f64_t>(rows_count);
        for (std::size_t query = 0; query != queries_count; ++query) {
            nk::range_t const kept = nk::range(std::size_t {0}, counts[query]);
            dense_expect_indexer_matches(inputs, query, ranked_keys(query, kept, nk::slice).as_vector(),
                                         ranked_distances(query, kept, nk::slice).as_vector(),
                                         to_writable_span(ranked));
        }
        return;
    }
}

/**
 *  @brief Sharded scans against a one-to-one oracle over every stored row.
 *
 *  Two shards hold more rows than one first chunk, and one of two threads' share of the queries
 *  outnumbers one packed block, so the scan crosses both block boundaries. Each query - a stored
 *  row - ranks its true nearest at their own distances, a NaN for a self-match included; a
 *  predicate narrows the same ranking to the keys it admits where the residency takes one; rows
 *  streamed in blocks of one kit task, where a cache size blocks them, rank alike; and a request
 *  wider than the store returns every row. The serial kit judges a device's ranking.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_scan(u64_t seed, placement_types_&... placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    constexpr bool device_k = !a_ram_collection<indexes_type_>;
    using oracle_kit_t =
        typename quads_ladder<scalar_t, dense_reference_capability<typename indexes_type_::index_t>()>::kit_t;
    constexpr std::size_t rows_count = 2100, queries_count = 520, wanted = 10, wide_queries_count = 3;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement...)));
    // A device placement takes one shard a call.
    if constexpr (device_k) expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows_count}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                stream_key(seed, "indexes_scan"));
    nk::vector<key_t> keys = make_vector<key_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

    // Every query against every row through the one-to-one kernel the walk scores with.
    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    nk::matrix<distance_t> oracle = make_matrix<distance_t>(queries_count, rows_count);
    for (std::size_t query = 0; query != queries_count; ++query) {
        auto const query_vector = queries.row(query).as_vector();
        auto const query_norms = norms<oracle_kit_t>(query_vector);
        for (std::size_t row = 0; row != rows_count; ++row) {
            auto const row_vector = vectors.view().row(row).as_vector();
            oracle(query, row) = distance<oracle_kit_t>(kernel_, query_vector, query_norms, row_vector,
                                                        norms<oracle_kit_t>(row_vector));
        }
    }

    nk::matrix<key_t> found_keys = make_matrix<key_t>(queries_count, wanted);
    nk::matrix<distance_t> found_distances = make_matrix<distance_t>(queries_count, wanted);
    nk::matrix<key_t> wide_keys = make_matrix<key_t>(wide_queries_count, rows_count);
    nk::matrix<distance_t> wide_distances = make_matrix<distance_t>(wide_queries_count, rows_count);
    nk::vector<std::size_t> counts = make_vector<std::size_t>(queries_count);
    auto all = [](key_t) noexcept { return true; };
    auto even = [](key_t key) noexcept { return dense_key_number(key) % 2 == 0; };
    auto admit_even = quad_predicate(
        [&](key_t key) noexcept { return even(key) ? admission_t::admit_k : admission_t::reject_k; });
    // One query's found ranking against the oracle's, sorted over the keys `admits` lets through.
    error_stats_t scan_errors;
    nk::vector<distance_t> admitted_distances = make_vector<distance_t>(rows_count);
    auto expect_ranking = [&](nk::matrix_view<key_t> ranked_keys, nk::matrix_view<distance_t> ranked_distances,
                              std::size_t query, auto admits) {
        std::size_t admitted_count = 0;
        for (std::size_t row = 0; row != rows_count; ++row)
            if (admits(keys[row])) admitted_distances[admitted_count++] = oracle(query, row);
        std::span<distance_t> const admitted = to_writable_span(admitted_distances).first(admitted_count);
        std::sort(admitted.begin(), admitted.end());
        expect_equal(counts[query], min_of(ranked_keys.extent(1), admitted.size()));
        for (std::size_t rank = 0; rank != min_of(counts[query], admitted.size()); ++rank) {
            key_t const key = ranked_keys(query, rank);
            distance_t const distance = ranked_distances(query, rank);
            expect_true(admits(key));
            scan_errors.add(distance, admitted[rank]);
            scan_errors.add(distance, oracle(query, dense_key_number(key) - 1));
        }
        return;
    };

    typename indexes_type_::search_config_t const scan {.traversal = traversal_t::scan_k};
    auto const [found, status] = drive_search(indexes, queries, found_keys.span(), found_distances.span(),
                                              to_writable_span(counts), scan, driver);
    expect_true(succeeded(status));
    expect_equal(found.computed_distances, queries_count * rows_count);
    for (std::size_t query = 0; query != queries_count; ++query)
        expect_ranking(found_keys.view(), found_distances.view(), query, all);

    if constexpr (!device_k) {
        auto const filtered_status = drive_search_if(indexes, queries, found_keys.span(), found_distances.span(),
                                                     to_writable_span(counts), scan, admit_even, driver)
                                         .status;
        expect_true(succeeded(filtered_status));
        for (std::size_t query = 0; query != queries_count; ++query)
            expect_ranking(found_keys.view(), found_distances.view(), query, even);
    }
    // A cache smaller than one pack leaves blocks of a single kit task.
    if constexpr (requires { scan.cache_bytes; }) {
        typename indexes_type_::search_config_t const blocked {.traversal = traversal_t::scan_k, .cache_bytes = 1};
        auto const blocked_status = drive_search(indexes, queries, found_keys.span(), found_distances.span(),
                                                 to_writable_span(counts), blocked, driver)
                                        .status;
        expect_true(succeeded(blocked_status));
        for (std::size_t query = 0; query != queries_count; ++query)
            expect_ranking(found_keys.view(), found_distances.view(), query, all);
    }

    auto const wide_queries = vectors(nk::range(std::size_t {0}, wide_queries_count), nk::slice);
    auto const wide_status = drive_search(indexes, wide_queries, wide_keys.span(), wide_distances.span(),
                                          to_writable_span(counts).first(wide_queries_count), scan, driver)

                                 .status;
    expect_true(succeeded(wide_status));
    for (std::size_t query = 0; query != wide_queries_count; ++query)
        expect_ranking(wide_keys.view(), wide_distances.view(), query, all);
    expect_errors("sharded scan distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  scan_errors, 0);
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

/** The bytes of a writable matrix span's rows - what a row-range slice returns. */
template <typename value_type_>
bytes_span_t dense_matrix_bytes(nk::matrix_span<value_type_> span) noexcept {
    return {reinterpret_cast<byte_t*>(span.data()), dense_rows_bytes(span, span.extent(0))};
}

/** Saves every row of @p index into @p columns in one call, on the stream of the cell
 *  @p make_index where the index takes one. */
template <typename index_type_, typename make_index_type_>
status_t dense_save(index_type_& index, make_index_type_ const& make_index, typename index_type_::columns_t columns,
                    synchronization_t synchronization) {
    if constexpr (requires { index.save(columns, synchronization, make_index.stream); })
        return index.save(columns, synchronization, make_index.stream);
    else return index.save(columns, synchronization);
}

/** Appends @p rows rows of @p columns under @p keys to @p index in one call, on the stream of the
 *  cell @p make_index where the index takes one. */
template <typename index_type_, typename make_index_type_>
status_t dense_load(index_type_& index, make_index_type_ const& make_index,
                    typename index_type_::columns_view_t columns, typename index_type_::keys_view_t keys,
                    std::size_t rows, iota_view<typename index_type_::node_id_t> source_range) {
    if constexpr (requires { index.load(columns, keys, rows, source_range, make_index.stream); })
        return index.load(columns, keys, rows, source_range, make_index.stream);
    else return index.load(columns, keys, rows, source_range);
}

/** Nodes a plan spans - the bound of its last task. */
template <typename plan_type_>
std::size_t dense_plan_rows(plan_type_ const& plan) noexcept {
    auto const tasks = plan.tasks();
    return tasks.size() ? static_cast<std::size_t>(tasks[tasks.size() - 1].index_nodes.bound) : 0;
}

/** Any number of threads may take chunks from one plan: a shared cursor hands them out,
 *  and the bytes match the serial save. */
template <typename index_type_, typename scalar_type_, typename node_id_type_>
void dense_expect_pooled_save(index_type_& index, typename index_type_::saved_shape_t shape,
                              nk::matrix<scalar_type_>& saved_vectors, nk::matrix<node_id_type_>& saved_neighbors) {
    nk::matrix<scalar_type_> pooled_vectors = make_matrix<scalar_type_>(shape.nodes, shape.dimensions);
    nk::matrix<node_id_type_> pooled_neighbors = make_matrix<node_id_type_>(shape.nodes, shape.connectivity);
    expect_true(!pooled_vectors.empty() && !pooled_neighbors.empty());
    auto [plan, status] = index.plan_save({dense_matrix_bytes(pooled_vectors), dense_matrix_bytes(pooled_neighbors)},
                                          synchronization_t::immutable_k);
    expect_true(succeeded(status));
    std::size_t cursor_position = 0, lane_failures_count = 0;
    fu::standard_atomic_ref<std::size_t> const cursor {cursor_position};
    fu::standard_atomic_ref<std::size_t> const lane_failures {lane_failures_count};
    constexpr std::size_t lanes_count = 3;
    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, lanes_count, fu::caller_inclusive_k)))
        return void(expect_true(false));
    expect_true(!fu::failed(pool.for_threads([&](std::size_t) noexcept {
        auto const tasks = plan.tasks();
        for (std::size_t position = cursor.fetch_add(1); position < tasks.size(); position = cursor.fetch_add(1))
            if (failed(plan.save(tasks[position]))) lane_failures.store_add(1);
    })));
    expect_equal(lane_failures_count, 0u);
    expect_equal(plan.finish(), status_code_t::success_k);
    expect_true(std::memcmp(pooled_vectors.data(), saved_vectors.data(),
                            dense_rows_bytes(saved_vectors.view(), shape.nodes)) == 0);
    expect_true(std::memcmp(pooled_neighbors.data(), saved_neighbors.data(),
                            dense_rows_bytes(saved_neighbors.view(), shape.nodes)) == 0);
}

/** Round-trips the graph through one plan per direction: build, save into two host
 *  matrices chunk by chunk, load into a fresh reserved index the same way, and expect the
 *  restored graph to answer self-queries just as the original did - norms re-derive on
 *  load and are never serialized. On RAM a pool of threads then takes chunks from one plan
 *  by a shared cursor and must write the same bytes. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_save_load_round_trip(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(
        dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k), stream_key(seed, "roundtrip"));

    expect_true(index.reserve({dense_growth_nodes_k}).status == status_code_t::success_k);
    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(dense_growth_nodes_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
    expect_true(index.insert(to_span(keys), vectors.view(), assigned_ids.span(), context).status ==
                status_code_t::success_k);

    auto const shape = index.saved_shape();
    nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(shape.nodes, shape.dimensions);
    nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(shape.nodes, shape.connectivity);
    expect_true(!saved_vectors.empty() && !saved_neighbors.empty());
    expect_equal(dense_save(index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                            synchronization_t::exclusive_k),
                 status_code_t::success_k);

    std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
    index_type_& restored = *restored_ptr;
    expect_true(restored.reserve({shape.nodes}).status == status_code_t::success_k);
    expect_equal(dense_load(restored, make_index,
                            {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                            {keys.data(), shape.nodes}, shape.nodes, {0, static_cast<node_id_t>(shape.nodes)}),
                 status_code_t::success_k);
    expect_equal(restored.size(), shape.nodes);

    // A restored index must re-save on every contract: the settlement counters arrive
    // fully released, so the optimistic mode's probe sees a quiescent chain, and the
    // re-saved bytes mirror the originals exactly.
    nk::matrix<scalar_t> resaved_vectors = make_matrix<scalar_t>(shape.nodes, shape.dimensions);
    nk::matrix<node_id_t> resaved_neighbors = make_matrix<node_id_t>(shape.nodes, shape.connectivity);
    expect_true(!resaved_vectors.empty() && !resaved_neighbors.empty());
    typename index_type_::columns_t const resaved {dense_matrix_bytes(resaved_vectors),
                                                   dense_matrix_bytes(resaved_neighbors)};
    expect_equal(dense_save(restored, make_index, resaved, synchronization_t::exclusive_k), status_code_t::success_k);
    expect_equal(dense_save(restored, make_index, resaved, synchronization_t::immutable_k), status_code_t::success_k);
    // A device index refuses the concurrent mode by contract; the RAM index settles it.
    expect_equal(dense_save(restored, make_index, resaved, synchronization_t::concurrent_k),
                 dense_device_resident<index_type_> ? status_code_t::config_mismatch_k : status_code_t::success_k);
    expect_true(std::memcmp(resaved_vectors.data(), saved_vectors.data(),
                            dense_rows_bytes(saved_vectors.view(), shape.nodes)) == 0);
    expect_true(std::memcmp(resaved_neighbors.data(), saved_neighbors.data(),
                            dense_rows_bytes(saved_neighbors.view(), shape.nodes)) == 0);

    // Threads take chunks from one plan only where the residency admits the concurrent mode.
    if constexpr (!dense_device_resident<index_type_>)
        dense_expect_pooled_save(restored, shape, saved_vectors, saved_neighbors);

    if constexpr (kernel_ != nk_kernel_dot_k)
        dense_expect_self_recall(restored, make_index, vectors.view(), dense_growth_nodes_k);
}

/** @ref dense_expect_save_load_round_trip, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_save_load_round_trip(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_save_load_round_trip<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Columns one row short of the rows asked for are refused before planning, so no task ever
 *  reads past them and nothing is claimed. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
void dense_expect_clipped_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                               nk::matrix<node_id_type_>& saved_neighbors, typename index_type_::columns_view_t source,
                               std::size_t nodes_count, iota_view<node_id_type_> local) {
    std::unique_ptr<index_type_> const clipped_ptr = make_index(ledger);
    index_type_& clipped = *clipped_ptr;
    expect_true(clipped.reserve({nodes_count}).status == status_code_t::success_k);
    std::size_t const neighbors_row_bytes = dense_rows_bytes(saved_neighbors.view(), 1);
    typename index_type_::columns_view_t const short_source {
        source.vectors, source.neighbors.first(source.neighbors.size() - neighbors_row_bytes)};
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_equal(clipped.load(short_source, to_span(keys), nodes_count, local), status_code_t::config_mismatch_k);
    expect_equal(clipped.size(), 0u);
}

/** A plan finished with nothing placed refuses and publishes nothing, and so does a task
 *  past its last. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
void dense_expect_partial_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                               typename index_type_::columns_view_t source, std::size_t nodes_count,
                               iota_view<node_id_type_> local) {
    std::unique_ptr<index_type_> const failing_ptr = make_index(ledger);
    index_type_& failing = *failing_ptr;
    expect_true(failing.reserve({nodes_count}).status == status_code_t::success_k);
    auto [plan, status] = failing.plan_load(nodes_count);
    expect_true(succeeded(status));
    auto const tasks = plan.tasks();
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    auto const key_of = [&keys](node_id_type_ row) noexcept { return keys[row]; };
    expect_true(tasks.size() >= 2);
    expect_equal(plan.finish(), status_code_t::config_mismatch_k);
    expect_equal(failing.size(), 0u);
    expect_equal(plan.load(tasks[0], source, local, key_of), status_code_t::success_k);
    expect_equal(plan.load(tasks[tasks.size()], source, local, key_of), status_code_t::config_mismatch_k);
    expect_equal(plan.finish(), status_code_t::config_mismatch_k);
    expect_equal(failing.size(), 0u);

    // A refused finish spends nothing: the missing tasks run and the same plan publishes once.
    for (std::size_t position = 1; position < tasks.size(); ++position)
        expect_equal(plan.load(tasks[position], source, local, key_of), status_code_t::success_k);
    expect_equal(plan.finish(), status_code_t::success_k);
    expect_equal(failing.size(), nodes_count);
    expect_equal(plan.finish(), status_code_t::uninitialized_k); // Spent: no second publish.
    expect_equal(failing.size(), nodes_count);
}

/** A moved-from load plan refuses every verb, while the plan moved into keeps the tasks. */
template <typename index_type_>
void dense_expect_moved_plan_refuses(index_type_& index, std::size_t nodes_count) {
    auto [plan, status] = index.plan_load(nodes_count);
    expect_true(succeeded(status));
    auto moved = std::move(plan);
    expect_equal(plan.tasks().size(), 0u);
    expect_equal(plan.finish(), status_code_t::uninitialized_k); // Moved from: refuses.
    expect_true(moved.tasks().size() >= 1);
}

/** A load plan that dies before its finish releases the nodes it claimed, once even when moved: the
 *  next plan appends at the same base, in the nodes the reservation hosts, and places the whole
 *  checkpoint. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
void dense_expect_abandoned_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                 typename index_type_::columns_view_t source, std::size_t nodes_count,
                                 iota_view<node_id_type_> local) {
    std::unique_ptr<index_type_> const abandoned_ptr = make_index(ledger);
    index_type_& abandoned = *abandoned_ptr;
    auto const [reserved_rows, reserve_status] = abandoned.reserve({nodes_count});
    expect_true(reserve_status == status_code_t::success_k);
    dense_expect_moved_plan_refuses(abandoned, nodes_count);
    expect_equal(abandoned.size(), 0u);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_equal(abandoned.load(source, to_span(keys), nodes_count, local), status_code_t::success_k);
    expect_equal(abandoned.size(), nodes_count);
    expect_equal(abandoned.capacity(), reserved_rows); // The released rows hosted the load.
}

/** A save plan refuses a task past its watermark, and a finish with nothing placed. */
template <typename index_type_, typename scalar_type_, typename node_id_type_>
void dense_expect_save_plan_misuse(index_type_& index, nk::matrix<scalar_type_>& saved_vectors,
                                   nk::matrix<node_id_type_>& saved_neighbors) {
    auto [plan, status] = index.plan_save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                                          synchronization_t::exclusive_k);
    expect_true(succeeded(status));
    auto const tasks = plan.tasks();
    expect_equal(plan.save(tasks[tasks.size()]), status_code_t::config_mismatch_k);
    expect_equal(plan.finish(), status_code_t::config_mismatch_k);
}

/** Hostile load submissions: a target of another geometry refuses before any entry is copied, a
 *  source failing on a later chunk publishes nothing, a misused plan refuses, a count past the
 *  reservation grows by design, and an out-of-range neighbor ID rides the sentinel convention -
 *  stored, but never surfaced by any search. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_load_hostile(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    // Enough nodes for the walk to span two storage chunks, so a second run exists to fail on.
    std::size_t const nodes_count = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "load_hostile"));
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_true(index.insert(to_span(keys), vectors.view(), {}, context).status == status_code_t::success_k);

    nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(nodes_count, dimensions);
    nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(nodes_count, connectivity_16_k);
    expect_true(!saved_vectors.empty() && !saved_neighbors.empty());
    expect_equal(index.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                            synchronization_t::exclusive_k),
                 status_code_t::success_k);

    typename index_type_::columns_view_t const source {dense_matrix_bytes(saved_vectors.view()),
                                                       dense_matrix_bytes(saved_neighbors.view())};
    iota_view<node_id_t> const local {0, static_cast<node_id_t>(nodes_count)};

    // A target of another geometry finds the columns too short for its rows.
    std::unique_ptr<index_type_> const wider_ptr = make_index(ledger, {2 * dimensions, connectivity_16_k});
    index_type_& wider = *wider_ptr;
    expect_equal(wider.load(source, to_span(keys), nodes_count, local), status_code_t::config_mismatch_k);
    expect_equal(wider.size(), 0u);

    dense_expect_clipped_load<index_type_>(make_index, ledger, saved_neighbors, source, nodes_count, local);
    dense_expect_partial_load<index_type_>(make_index, ledger, source, nodes_count, local);
    dense_expect_abandoned_load<index_type_>(make_index, ledger, source, nodes_count, local);
    dense_expect_save_plan_misuse(index, saved_vectors, saved_neighbors);

    // A count past the reservation grows by design - the chunk directory follows.
    std::unique_ptr<index_type_> const growing_ptr = make_index(ledger);
    index_type_& growing = *growing_ptr;
    expect_true(growing.reserve({1}).status == status_code_t::success_k);
    expect_equal(growing.load(source, to_span(keys), nodes_count, local), status_code_t::success_k);
    expect_equal(growing.size(), nodes_count);
    // A reservation one node above the hosted ones grows them, though that one node fits the free
    // nodes.
    std::size_t const hosted_rows = growing.capacity();
    auto const [reserved, reserve_status] = growing.reserve({hosted_rows + 1});
    expect_true(succeeded(reserve_status) && reserved > hosted_rows);

    // Out-of-range neighbor IDs load as sentinels - the interchange admits any - and the
    // translation frontier keeps them from ever surfacing in a search.
    nk::matrix<node_id_t> patched_neighbors = make_matrix<node_id_t>(nodes_count, connectivity_16_k);
    expect_true(!patched_neighbors.empty());
    std::memcpy(patched_neighbors.data(), saved_neighbors.data(),
                dense_rows_bytes(saved_neighbors.view(), nodes_count));
    patched_neighbors(5, 0) = static_cast<node_id_t>(nodes_count + 5);
    patched_neighbors(7, 1) = node_id_unknown_k;

    std::unique_ptr<index_type_> const patched_ptr = make_index(ledger);
    index_type_& patched = *patched_ptr;
    expect_true(patched.reserve({nodes_count}).status == status_code_t::success_k);
    expect_equal(
        patched.load({source.vectors, dense_matrix_bytes(patched_neighbors.view())}, to_span(keys), nodes_count, local),
        status_code_t::success_k);
    typename index_type_::traversal_context_t patched_context {{}};
    for (std::size_t sample = 0; sample != 16; ++sample) {
        auto const [result, status] = patched.search(vectors.view().row((sample * 19) % nodes_count).as_vector(), {},
                                                     patched_context);
        expect_true(status == status_code_t::success_k);
        for (std::size_t match = 0; match != result.count; ++match)
            expect_true(static_cast<std::size_t>(patched_context.results[match].node_id) < patched.size());
    }
}

/** @ref dense_expect_load_hostile, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_load_hostile(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_load_hostile<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Loads append: an ID space past the source's nodes places nothing, the same payload placed twice
 *  doubles the index, the second copy's IDs shift by the first copy's nodes with sentinels intact,
 *  and a self-query finds either copy. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_load_append(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t const nodes_count = 300;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "load_hostile", 1));
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_true(index.insert(to_span(keys), vectors.view(), {}, context).status == status_code_t::success_k);

    nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(nodes_count, dimensions);
    nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(nodes_count, connectivity_16_k);
    expect_true(!saved_vectors.empty() && !saved_neighbors.empty());
    expect_equal(dense_save(index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                            synchronization_t::exclusive_k),
                 status_code_t::success_k);

    typename index_type_::columns_view_t const source {dense_matrix_bytes(saved_vectors.view()),
                                                       dense_matrix_bytes(saved_neighbors.view())};
    iota_view<node_id_t> const local {0, static_cast<node_id_t>(nodes_count)};
    std::unique_ptr<index_type_> const doubled_ptr = make_index(ledger);
    index_type_& doubled = *doubled_ptr;
    expect_true(doubled.reserve({2 * nodes_count}).status == status_code_t::success_k);
    // An ID space past the source's nodes would rebase IDs beyond the placed ones, so the plan
    // places nothing at all.
    iota_view<node_id_t> const overreaching {0, static_cast<node_id_t>(nodes_count + 1)};
    expect_equal(dense_load(doubled, make_index, source, to_span(keys), nodes_count, overreaching),
                 status_code_t::config_mismatch_k);
    expect_equal(doubled.size(), 0u);
    expect_equal(dense_load(doubled, make_index, source, to_span(keys), nodes_count, local), status_code_t::success_k);
    expect_equal(doubled.size(), nodes_count);
    expect_equal(dense_load(doubled, make_index, source, to_span(keys), nodes_count, local), status_code_t::success_k);
    expect_equal(doubled.size(), 2 * nodes_count);

    nk::matrix<scalar_t> doubled_vectors = make_matrix<scalar_t>(2 * nodes_count, dimensions);
    nk::matrix<node_id_t> doubled_neighbors = make_matrix<node_id_t>(2 * nodes_count, connectivity_16_k);
    expect_true(!doubled_vectors.empty() && !doubled_neighbors.empty());
    expect_equal(
        dense_save(doubled, make_index, {dense_matrix_bytes(doubled_vectors), dense_matrix_bytes(doubled_neighbors)},
                   synchronization_t::exclusive_k),
        status_code_t::success_k);

    // The first copy is placed verbatim; the second copy's vectors match and its IDs shift.
    expect_true(std::memcmp(doubled_vectors.data(), saved_vectors.data(),
                            dense_rows_bytes(saved_vectors.view(), nodes_count)) == 0);
    expect_true(std::memcmp(doubled_neighbors.data(), saved_neighbors.data(),
                            dense_rows_bytes(saved_neighbors.view(), nodes_count)) == 0);
    expect_true(std::memcmp(doubled_vectors.view().row(nodes_count).byte_data(), saved_vectors.data(),
                            dense_rows_bytes(saved_vectors.view(), nodes_count)) == 0);
    std::size_t unshifted = 0;
    for (std::size_t row = 0; row != nodes_count; ++row)
        for (std::size_t slot = 0; slot != saved_neighbors.extent(1); ++slot) {
            node_id_t const first = saved_neighbors(row, slot);
            node_id_t const second = doubled_neighbors(nodes_count + row, slot);
            node_id_t const shifted = first == node_id_unknown_k ? node_id_unknown_k
                                                                 : static_cast<node_id_t>(first + nodes_count);
            unshifted += second != shifted;
        }
    expect_equal(unshifted, 0u);

    // A self-query finds one of the two copies with a self-match distance.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        typename index_type_::traversal_context_t doubled_context = dense_make_context<index_type_>(make_index);
        for (std::size_t sample = 0; sample != 16; ++sample) {
            std::size_t const row = (sample * 19) % nodes_count;
            auto const [result, status] = doubled.search(vectors.view().row(row).as_vector(), {}, doubled_context);
            expect_true(status == status_code_t::success_k && result.count > 0);
            if (!result.count) continue;
            scored_node_t const best = doubled_context.results[0];
            expect_true(best.node_id == row || best.node_id == row + nodes_count);
            expect_true(std::fabs(best.distance) < dense_self_match_tolerance_k);
        }
    }
}

/** @ref dense_expect_load_append, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_load_append(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_load_append<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Snapshots

#pragma region Refinement

/** Sweeps one audit to convergence: the bitsets restart, then every task of every depth
 *  until no depth expands a node - the serial spelling of what an executor fans out. The
 *  restart is what lets a repaired graph be re-audited; without it the expanded bits still
 *  stand and no fresh edge is ever followed. */
template <typename plan_type_>
void dense_sweep_reachability(plan_type_& plan) {
    plan.restart_sweep();
    do
        for (auto const task : plan.tasks()) plan.sweep(task);
    while (plan.advance_depth());
}

/** One shard's reachability, whichever way its residency spells the audit: the device
 *  answers in one call, the host drives the plan the caller owns. */
template <typename index_type_>
auto dense_audit_health(index_type_& index, typename index_type_::traversal_context_t& context) {
    if constexpr (requires { index.health(context); }) return index.health(context);
    else {
        auto [plan, status] = index.plan_health();
        if (failed(status)) return expected<typename index_type_::reachability_t> {{}, status};
        dense_sweep_reachability(plan);
        return plan.finish();
    }
}

/**
 *  @brief The settling contract: a graph built under narrow candidates has work for the first
 *      round, the second changes fewer entries, neither walks, and recall never falls.
 *
 *  The bound on the second round is a ratio rather than zero, since each round draws its hubs
 *  afresh and so files some entries beside new neighborhood-mates.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_refine_settles(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(
        dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k), stream_key(seed, "refine"));
    expect_true(index.reserve({16}).status == status_code_t::success_k);

    // Narrow enough that the first sweep has real work to do - the recipe the verb exists for.
    typename index_type_::insert_config_t cheap;
    cheap.expansion_factor = 4;
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
    expect_true(index.insert(to_span(keys), vectors.view(), {}, context, cheap).status == status_code_t::success_k);
    expect_equal(index.size(), dense_growth_nodes_k);

    std::size_t const hits_before = kernel_ != nk_kernel_dot_k
                                        ? dense_count_self_hits(index, make_index, vectors.view(), dense_growth_nodes_k)
                                        : 0;
    // A round re-stitches every entry from its links and neighborhood-mates alone.
    auto const [first_refinement, first_status] = index.refine({}, context);
    expect_true(first_status == status_code_t::success_k);
    expect_true(first_refinement.reselected_nodes >= dense_growth_nodes_k);
    expect_true(first_refinement.changed_nodes > 0 &&
                first_refinement.changed_nodes <= first_refinement.reselected_nodes);
    expect_true(first_refinement.neighborhoods != 0 && first_refinement.verdict == refine_verdict_t::improved_k);
    traversal_stats_t const& spent = first_refinement.traversal_stats;
    expect_true(spent.computed_distances > 0 && spent.visited_members == 0);
    expect_true(spent.computed_distances_in_refines <= spent.computed_distances);

    auto const [second_refinement, second_status] = index.refine({}, context);
    expect_true(second_status == status_code_t::success_k);
    expect_true(second_refinement.reselected_nodes >= dense_growth_nodes_k);
    expect_true(second_refinement.changed_nodes < first_refinement.changed_nodes);

    // A round is a build task, so what it must never do is cost the graph its answers.
    if constexpr (kernel_ != nk_kernel_dot_k)
        expect_true(dense_count_self_hits(index, make_index, vectors.view(), dense_growth_nodes_k) +
                        recall_margin_k * recall_samples_k >=
                    hits_before);
}

/** @ref dense_expect_refine_settles, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_refine_settles(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_refine_settles<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Builds and refines distinct vectors of which every stride-th repeats the first, a duplicate
 *  class of a hundred entries: every member must reach an entry outside the class within two hops,
 *  no member may draw a quarter of the class's links, and a search for the vector, unlike an
 *  insertion walk, must keep more than one member. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_duplicate_basin(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    // A dot scores identical rows by their norm and an angular through approximate roots, so
    // neither meets zero.
    if constexpr (kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_angular_k) return;
    constexpr std::size_t duplicate_rows_k = 100;
    constexpr std::size_t stride_k = dense_growth_nodes_k / duplicate_rows_k;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dimensions,
                                                              stream_key(seed, "basin"));
        // Read through a view: a span's packed dimension is a proxy, and proxy-to-proxy assignment
        // copies no bit.
        auto const source = vectors.view().row(0).as_vector();
        for (std::size_t row = stride_k; row < dense_growth_nodes_k; row += stride_k) {
            auto target = vectors.row(row).as_vector();
            for (std::size_t dimension = 0; dimension != dimensions; ++dimension) target[dimension] = source[dimension];
        }
        expect_true(index.reserve({dense_growth_nodes_k}).status == status_code_t::success_k);
        nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(dense_growth_nodes_k);
        std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
        expect_true(index.insert(to_span(keys), vectors.view(), assigned_ids.span(), context).status ==
                    status_code_t::success_k);
        expect_true(index.refine({}, context).status == status_code_t::success_k);

        auto const shape = index.saved_shape();
        nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(shape.nodes, shape.dimensions);
        nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(shape.nodes, shape.connectivity);
        expect_equal(
            dense_save(index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                       synchronization_t::exclusive_k),
            status_code_t::success_k);

        std::vector<bool> duplicate(shape.nodes, false);
        for (std::size_t row = 0; row < dense_growth_nodes_k; row += stride_k) duplicate[assigned_ids[row]] = true;
        auto const escapes = [&](node_id_t node) noexcept {
            auto const links = saved_neighbors.row(node).as_vector();
            bool found = false;
            for (std::size_t slot = 0; slot != links.size(); ++slot)
                found |= links[slot] != node_id_unknown_k && links[slot] != node && !duplicate[links[slot]];
            return found;
        };
        std::vector<std::size_t> class_links(shape.nodes, 0);
        std::size_t trapped = 0;
        for (std::size_t offset = 0; offset < dense_growth_nodes_k; offset += stride_k) {
            node_id_t const member = assigned_ids[offset];
            auto const links = saved_neighbors.row(member).as_vector();
            bool reached = escapes(member);
            for (std::size_t slot = 0; slot != links.size(); ++slot) {
                node_id_t const neighbor = links[slot];
                if (neighbor == node_id_unknown_k || neighbor == member || !duplicate[neighbor]) continue;
                ++class_links[neighbor];
                reached |= escapes(neighbor);
            }
            trapped += !reached;
        }
        expect_equal(trapped, 0u);
        expect_true(*std::max_element(class_links.begin(), class_links.end()) * 4 < duplicate_rows_k);

        auto const [found, found_status] = index.search(vectors.row(0).as_vector(), {}, context);
        std::size_t members_found = 0;
        for (std::size_t match = 0; succeeded(found_status) && match != found.count; ++match)
            members_found += context.results[match].distance < dense_self_match_tolerance_k;
        expect_true(members_found > 1);
    }
    expect_equal(ledger.live_bytes, 0);
}

/** The audit's contract on a graph without islands: an empty index reaches nothing, a
 *  healthy one reaches everything, and two audits of one graph agree - the landmarks are
 *  stride-spread, so the answer is a property of the graph rather than of the draw. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_health_audits(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    expect_true(index.reserve({16}).status == status_code_t::success_k);

    auto const [empty_health, empty_status] = dense_audit_health(index, context);
    expect_true(succeeded(empty_status));
    expect_equal(empty_health.reachable_nodes, 0u);

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(
        dense_growth_nodes_k, dense_dimensions_for<scalar_t>(dense_values_k), stream_key(seed, "health"));
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
    expect_true(index.insert(to_span(keys), vectors.view(), {}, context).status == status_code_t::success_k);

    // An insertion wires each node against the graph as it stood, so a few can end up with no
    // in-edge any search follows; reaching every one of them is what the repair is for, and
    // what the healing scenario asserts. Here the claim is that the audit sees the bulk, which
    // a dot graph's hubs never promise.
    auto const [health, audit_status] = dense_audit_health(index, context);
    expect_true(succeeded(audit_status));
    expect_true(health.reachable_nodes <= index.size());
    if constexpr (kernel_ != nk_kernel_dot_k)
        expect_true(health.reachable_nodes * 100 >= index.size() * dense_minimal_recall_percent_k);
    expect_true(health.depth_from_root >= 1);

    auto const [again_health, again_status] = dense_audit_health(index, context);
    expect_true(succeeded(again_status));
    expect_equal(again_health.reachable_nodes, health.reachable_nodes);
    expect_equal(again_health.depth_from_root, health.depth_from_root);
}

/** @ref dense_expect_health_audits, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_health_audits(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_health_audits<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Makes islands of the nodes from @p first_island on: every reference to the tail leaves the
 *  graph, and each island keeps only the published-empty self-loop, pointed at by nothing. */
template <typename node_id_type_>
void dense_make_islands(nk::matrix_span<node_id_type_> neighbors, node_id_type_ first_island) {
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
 *      of a build: a checkpoint's neighbor lists are rewritten so a tail of nodes points only at
 *      itself, which no walk can enter, and the load keeps that encoding.
 *
 *  One round must then reach every node, and a second must find no island left.
 */
template <typename index_type_, typename make_index_type_>
void dense_expect_heal_islands(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    constexpr std::size_t nodes_count = 512, islands_count = 32;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "heal"));

    // Build every node, then make islands of the tail: its neighbor lists become self-loops, the
    // "published with no neighbors" encoding, so nothing points into them and they point nowhere.
    std::unique_ptr<index_type_> const built_ptr = make_index(ledger);
    index_type_& built = *built_ptr;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    expect_true(built.reserve({nodes_count}).status == status_code_t::success_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_true(built.insert(to_span(keys), vectors.view(), {}, context).status == status_code_t::success_k);

    nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(nodes_count, dimensions);
    nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(nodes_count,
                                                                   static_cast<std::size_t>(built.connectivity()));
    expect_true(!saved_vectors.empty() && !saved_neighbors.empty());
    if (saved_vectors.empty() || saved_neighbors.empty()) return;

    expect_equal(dense_save(built, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                            synchronization_t::exclusive_k),
                 status_code_t::success_k);
    expect_equal(built.size(), nodes_count);
    dense_make_islands(saved_neighbors.span(), static_cast<node_id_t>(nodes_count - islands_count));

    std::unique_ptr<index_type_> const islanded_ptr = make_index(ledger);
    index_type_& islanded = *islanded_ptr;
    expect_true(islanded.reserve({nodes_count}).status == status_code_t::success_k);
    expect_equal(dense_load(islanded, make_index,
                            {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                            to_span(keys), nodes_count, {0, static_cast<node_id_t>(nodes_count)}),
                 status_code_t::success_k);

    // The landmarks are spread across the whole ID space, so a planted island that happens
    // to be one is reachable by a walk starting there - what must hold is that the rest
    // are not, and that the repair closes every one of them.
    auto const [before_health, before_status] = dense_audit_health(islanded, context);
    expect_true(succeeded(before_status));
    expect_true(before_health.reachable_nodes < nodes_count);

    auto const [repair, repair_status] = islanded.refine({}, context);
    expect_true(succeeded(repair_status));
    expect_equal(repair.reachability.reachable_nodes, nodes_count);
    expect_true(repair.islands_found > 0);

    auto const [after_health, after_status] = dense_audit_health(islanded, context);
    expect_true(succeeded(after_status));
    expect_equal(after_health.reachable_nodes, nodes_count);

    auto const [settled_repair, settled_status] = islanded.refine({}, context);
    expect_true(succeeded(settled_status));
    expect_equal(settled_repair.islands_found, 0u);
    expect_equal(settled_repair.islands_reconnected, 0u);
}

/** @ref dense_expect_heal_islands, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
void test_dense_heal_islands(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_heal_islands<index_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/**
 *  @brief The round's in-place contract on a graph every row of which is reached both ways.
 *
 *  Driven task by task, a round never grows past what planning and @c reserve placed, under the
 *  default cache and under one of a few rows, and every row it re-stitches stays reached, so its
 *  closing health check finds no island. Row zero sits at the center of the value range, a hub
 *  most rows pick. Islands planted in the tail, which nothing links to and which link nowhere, and
 *  an orphan linked from one far row alone then rejoin in one round, which heals again from its
 *  closing health check while the islands shrink.
 *
 *  One round first heals whatever the build left, which a dot graph's hubs can stall short of
 *  every row; the contract is then judged on the graph that round left.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_round_in_place(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    using refine_config_t = typename index_type_::refine_config_t;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(dense_growth_nodes_k, dimensions,
                                                              stream_key(seed, "shadow"));
        auto hub = vectors.row(0).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension) hub[dimension] = make_scalar<scalar_t>(4);
        expect_true(succeeded(index.reserve({dense_growth_nodes_k}).status));
        typename index_type_::insert_config_t cheap;
        cheap.expansion_factor = 4;
        std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
        expect_true(succeeded(index.insert(to_span(keys), vectors.view(), {}, context, cheap).status));
        node_id_t const published = static_cast<node_id_t>(index.size());
        expect_true(succeeded(index.refine({}, context).status));
        auto const [health, health_status] = dense_audit_health(index, context);
        expect_true(succeeded(health_status));
        bool const whole = health.reachable_nodes == published;
        if constexpr (kernel_ != nk_kernel_dot_k) expect_true(whole);

        // Sixteen rows of cache a core put the first neighborhoods over capacity.
        std::size_t const row_bytes = index_type_::padded_row_values({dimensions, connectivity_16_k}) *
                                      sizeof(scalar_t);
        for (std::size_t const cache_bytes : {std::size_t {0}, 32 * row_bytes}) {
            refine_config_t config;
            config.cache_bytes = cache_bytes;
            expect_equal(index.reserve_traversal(context, {}), status_code_t::success_k);
            auto [plan, plan_status] = index.plan_refine(config);
            expect_equal(plan_status, status_code_t::success_k);
            expect_equal(plan.reserve(context), status_code_t::success_k);
            std::ptrdiff_t const planned_bytes = ledger.live_bytes;
            std::size_t ran = 0;
            do
                for (auto const task : plan.tasks()) {
                    expect_equal(plan.run(task, context), status_code_t::success_k);
                    expect_true(ledger.live_bytes <= planned_bytes);
                    ++ran;
                }
            while (plan.advance());
            expect_true(ledger.live_bytes <= planned_bytes);
            auto const [refined, finish_status] = plan.finish();
            expect_equal(finish_status, status_code_t::success_k);
            expect_true(ran != 0);
            expect_true(refined.reselected_nodes >= published);
            expect_true(refined.neighborhoods != 0 && refined.widest_neighborhood != 0);
            if (!whole) continue;
            expect_equal(refined.islands_found, 0u);
            expect_equal(refined.reachability.reachable_nodes, published);
        }

        constexpr std::size_t islands_count = 32;
        auto const shape = index.saved_shape();
        nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(shape.nodes, shape.dimensions);
        nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(shape.nodes, shape.connectivity);
        expect_equal(
            dense_save(index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                       synchronization_t::exclusive_k),
            status_code_t::success_k);
        dense_make_islands(saved_neighbors.span(), static_cast<node_id_t>(published - islands_count));
        node_id_t const orphan = published / 2 + 1, parent = published / 6;
        for (std::size_t node = 0; node != published; ++node) {
            auto row = saved_neighbors.row(node).as_vector();
            for (std::size_t slot = 0; slot != row.size(); ++slot)
                if (row[slot] == orphan) row[slot] = node_id_unknown_k;
        }
        auto parent_row = saved_neighbors.row(parent).as_vector();
        std::size_t parent_slot = 0;
        while (parent_slot + 1 != parent_row.size() && parent_row[parent_slot] != node_id_unknown_k) ++parent_slot;
        parent_row[parent_slot] = orphan;
        std::unique_ptr<index_type_> const islanded_ptr = make_index(ledger);
        index_type_& islanded = *islanded_ptr;
        expect_true(succeeded(islanded.reserve({dense_growth_nodes_k}).status));
        expect_equal(dense_load(islanded, make_index,
                                {dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                                to_span(keys), published, {0, published}),
                     status_code_t::success_k);
        auto const [planted_health, planted_status] = dense_audit_health(islanded, context);
        expect_true(succeeded(planted_status) && planted_health.reachable_nodes < published);
        auto const [joined, joined_status] = islanded.refine({}, context);
        expect_true(succeeded(joined_status));
        expect_true(joined.islands_found >= islands_count);
        if (whole) expect_equal(joined.reachability.reachable_nodes, published);
    }
    expect_equal(ledger.live_bytes, 0);
}

/** Of the ten nearest entries by @p index 's own exact scan, how many its walks return, summed over
 *  the rows of @p queries; a match within the tenth's distance counts, so ties never punish one. */
template <typename index_type_, typename make_index_type_, typename vectors_view_type_>
std::size_t dense_recall_at_ten(index_type_& index, make_index_type_ const& make_index, vectors_view_type_ queries) {
    using search_config_t = typename index_type_::search_config_t;
    using candidates_count_t = typename index_type_::candidates_count_t;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    search_config_t const scan {.expansion_factor = static_cast<candidates_count_t>(recall_wanted_k),
                                .traversal = traversal_t::scan_k};
    std::size_t found = 0;
    for (std::size_t query = 0; query != queries.extent(0); ++query) {
        auto const query_vector = queries.row(query).as_vector();
        if (failed(index.search(query_vector, scan, context).status) || !context.results.size()) continue;
        f64_t const tenth = context.results[context.results.size() - 1].distance;
        auto const [walked, status] = index.search(query_vector, {}, context);
        if (failed(status)) continue;
        for (std::size_t rank = 0; rank != min_of(recall_wanted_k, walked.count); ++rank)
            found += context.results[rank].distance <= tenth + 1e-3 * max_of(std::fabs(tenth), 1.0);
    }
    return found;
}

/**
 *  @brief A built index is cut into two and, refilled, into three graph-local parts: every entry
 *      ends in exactly one target, the source empties and takes entries from zero again, and
 *      every target's round reaches all its rows, its walks within noise of the source's.
 *
 *  Equal graphs cut under one seed place every entry alike. Refusals leave every index whole:
 *  fewer than two targets, a repeated or non-empty one, the source among them, another metric
 *  or another geometry.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_cut(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    constexpr std::size_t rows_k = dense_growth_nodes_k;
    constexpr nk_kernel_kind_t other_kernel_k = kernel_ == nk_kernel_euclidean_k ? nk_kernel_angular_k
                                                                                 : nk_kernel_euclidean_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_k, dimensions, stream_key(seed, "cut"));
    nk::matrix<scalar_t> const queries = make_vectors<scalar_t>(recall_samples_k, dimensions,
                                                                stream_key(seed, "cut", 1));
    std::size_t const row_bytes = dense_rows_bytes(vectors.view(), 1);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(rows_k);
    // Equal vectors under fresh contexts build equal graphs.
    auto build = [&](index_type_& index) {
        typename index_type_::traversal_context_t building = dense_make_context<index_type_>(make_index);
        expect_true(succeeded(index.reserve({rows_k}).status));
        expect_true(succeeded(index.insert(to_span(keys), vectors.view(), {}, building).status));
    };
    auto make_targets = [&](std::vector<std::unique_ptr<index_type_>>& owned, std::size_t count) {
        std::vector<index_type_*> targets;
        for (std::size_t target = 0; target != count; ++target) {
            owned.push_back(make_index(ledger));
            targets.push_back(owned.back().get());
        }
        return targets;
    };
    auto recall_close = [&](std::size_t found, std::size_t baseline) {
        return expect_true(found + recall_margin_k * recall_samples_k * recall_wanted_k >= baseline);
    };
    // The one source a mono-index reshard moves out of.
    auto source_of = [](index_type_& source) { return std::array<index_type_*, 1> {&source}; };
    // Drives one cut, then checks that every source entry ends in exactly one target, byte for
    // byte, and that every target reaches its nodes.
    auto cut = [&](index_type_& source, std::vector<index_type_*>& targets, u64_t cut_seed) {
        {
            expect_equal(source.reserve_traversal(context, {}), status_code_t::success_k);
            reshard_config_t config;
            config.seed = cut_seed;
            auto const sources = source_of(source);
            auto [plan, plan_status] = source.plan_reshard(sources, targets, config);
            expect_equal(plan_status, status_code_t::success_k);
            expect_equal(plan.reserve(context), status_code_t::success_k);
            expect_true(plan.moving() && source.size() == rows_k && targets[0]->size() == 0);
            do
                for (auto const task : plan.tasks()) expect_equal(plan.run(task, context), status_code_t::success_k);
            while (plan.advance());
            auto const [cut_result, finish_status] = plan.finish();
            expect_equal(finish_status, status_code_t::success_k);
            expect_equal(cut_result.reachability.audited_nodes, rows_k);
            // Every target runs a round after the moves, and no phase walks.
            expect_true(cut_result.neighborhoods >= targets.size());
            expect_true(cut_result.reselected_nodes >= rows_k);
            expect_equal(cut_result.traversal_stats.visited_members, 0u);
        }
        expect_equal(source.size(), 0u);
        std::vector<std::size_t> holders(rows_k);
        for (index_type_* target : targets) {
            expect_true(target->size() * targets.size() <= rows_k + targets.size() && target->size());
            for (node_id_t node = 0; node != target->size(); ++node)
                for (std::size_t row = 0; row != rows_k; ++row)
                    holders[row] += std::memcmp(target->vector(node).byte_data(),
                                                vectors.view().row(row).as_vector().byte_data(), row_bytes) == 0;
            auto const [health, health_status] = dense_audit_health(*target, context);
            expect_true(succeeded(health_status));
            if constexpr (kernel_ != nk_kernel_dot_k) expect_equal(health.reachable_nodes, target->size());
            else expect_true(health.reachable_nodes * 100 >= target->size() * dense_minimal_recall_percent_k);
        }
        std::size_t misplaced = 0;
        for (std::size_t const held : holders) misplaced += held != 1;
        expect_equal(misplaced, 0u);
    };

    std::unique_ptr<index_type_> const source = make_index(ledger);
    std::unique_ptr<index_type_> const twin = make_index(ledger);
    build(*source);
    build(*twin);
    std::size_t const source_recall = dense_recall_at_ten(*source, make_index, queries.view());

    // Refusals precede every claim: the source and every target stay as they were.
    {
        std::vector<std::unique_ptr<index_type_>> owned;
        std::vector<index_type_*> targets = make_targets(owned, 2);
        std::unique_ptr<index_type_> const foreign {
            new (std::nothrow) index_type_(other_kernel_k, {dimensions, connectivity_16_k},
                                           counting_allocator_t(ledger), counting_allocator_t(ledger))};
        std::unique_ptr<index_type_> const narrower = make_index(ledger, {dimensions * 2, connectivity_16_k});
        expect_true(succeeded(narrower->reserve({16}).status));
        for (std::vector<index_type_*> refused :
             {std::vector<index_type_*> {targets[0]}, std::vector<index_type_*> {targets[0], source.get()},
              std::vector<index_type_*> {targets[0], foreign.get()},
              std::vector<index_type_*> {targets[0], narrower.get()}})
            expect_equal(source->plan_reshard(source_of(*source), refused).status, status_code_t::config_mismatch_k);
        // Malformed lists, empty, holding a null or naming an index twice, refuse as arguments.
        for (std::vector<index_type_*> malformed :
             {std::vector<index_type_*> {}, std::vector<index_type_*> {targets[0], nullptr},
              std::vector<index_type_*> {targets[0], targets[0]}})
            expect_equal(source->plan_reshard(source_of(*source), malformed).status, status_code_t::invalid_argument_k);
        expect_equal(source->plan_reshard(std::span<index_type_* const> {}, targets).status,
                     status_code_t::invalid_argument_k);
        std::array<index_type_*, 2> const several {source.get(), twin.get()};
        expect_equal(source->plan_reshard(several, targets).status, status_code_t::unsupported_k);
        expect_true(succeeded(targets[1]->reserve({16}).status));
        expect_true(
            succeeded(targets[1]->insert({keys.data(), 1u}, vectors(nk::range(0, 1), nk::slice), {}, context).status));
        expect_equal(source->plan_reshard(source_of(*source), targets).status, status_code_t::config_mismatch_k);
        expect_true(source->size() == rows_k && targets[0]->size() == 0 && targets[1]->size() == 1);
    }

    // Two parts, twice over equal graphs: every entry ends at the same node of the same target.
    std::vector<std::unique_ptr<index_type_>> halves_owned, twin_halves_owned;
    std::vector<index_type_*> halves = make_targets(halves_owned, 2);
    std::vector<index_type_*> twin_halves = make_targets(twin_halves_owned, 2);
    u64_t const cut_seed = stream_key(seed, "cut", 2);
    cut(*source, halves, cut_seed);
    cut(*twin, twin_halves, cut_seed);
    for (std::size_t half = 0; half != halves.size(); ++half) {
        expect_equal(halves[half]->size(), twin_halves[half]->size());
        std::size_t moved = 0;
        for (node_id_t node = 0; node != min_of(halves[half]->size(), twin_halves[half]->size()); ++node)
            moved += std::memcmp(halves[half]->vector(node).byte_data(), twin_halves[half]->vector(node).byte_data(),
                                 row_bytes) != 0;
        expect_equal(moved, 0u);
        recall_close(dense_recall_at_ten(*halves[half], make_index, queries.view()), source_recall);
    }

    // The emptied source takes entries from zero again, so it refills and cuts into three.
    std::vector<std::unique_ptr<index_type_>> thirds_owned;
    std::vector<index_type_*> thirds = make_targets(thirds_owned, 3);
    expect_true(succeeded(source->insert(to_span(keys), vectors.view(), {}, context).status));
    expect_equal(source->size(), rows_k);
    cut(*source, thirds, stream_key(seed, "cut", 3));
    for (index_type_* third : thirds)
        recall_close(dense_recall_at_ten(*third, make_index, queries.view()), source_recall);
}

/** @ref dense_expect_cut, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_cut(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_cut<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Refinement

#pragma region Algebra

/** A logit scale spreading the oracle's dots over @c [-4,4], so no scored row's weight underflows. */
inline f32_t dense_attend_scale(nk::matrix_view<dense_dot_reference_t> dots) noexcept {
    f64_t widest = 0;
    for (std::size_t query = 0; query != dots.extent(0); ++query)
        for (std::size_t row = 0; row != dots.extent(1); ++row)
            widest = std::fmax(widest, std::fabs(dots(query, row).dot));
    return widest > 0 ? static_cast<f32_t>(4 / widest) : 1;
}

/** Every head's attended row and carry equal exact softmax attention over every key, in f64, from
 *  the oracle @p dots of every task against every key, @b [tasks,keys], and the @p widened_values
 *  row of every key. Rows of @p dots, @p outputs and @p carry are tasks t · @p head_count + h. */
inline void dense_expect_full_attention(nk::matrix_view<dense_dot_reference_t> dots,
                                        nk::matrix_view<f32_t> widened_values, std::size_t head_count,
                                        f32_t score_scale, nk::matrix_view<f32_t> outputs,
                                        std::span<softmax_state_t const> carry) {
    std::size_t const rows_count = dots.extent(1), columns = widened_values.extent(1), tasks = dots.extent(0);
    if (!expect_true(rows_count != 0 && head_count != 0 && tasks % head_count == 0) ||
        !expect_true(outputs.extent(0) == tasks && outputs.extent(1) == columns && carry.size() == tasks))
        return;

    nk::vector<f64_t> expected = make_vector<f64_t>(columns);
    error_stats_t output_errors, carry_errors;
    for (std::size_t task = 0; task != tasks; ++task) {
        f64_t maximum = std::numeric_limits<f64_t>::lowest(), partition = 0;
        for (std::size_t row = 0; row != rows_count; ++row)
            maximum = std::fmax(maximum, score_scale * dots(task, row).dot);
        std::fill_n(expected.data(), columns, 0);
        for (std::size_t row = 0; row != rows_count; ++row) {
            f64_t const weight = std::exp(score_scale * dots(task, row).dot - maximum);
            partition += weight;
            for (std::size_t column = 0; column != columns; ++column)
                expected[column] += weight * widened_values(row, column);
        }
        for (std::size_t column = 0; column != columns; ++column)
            output_errors.add(outputs(task, column), expected[column] / partition);
        carry_errors.add(carry[task].maximum_score, maximum);
        carry_errors.add(carry[task].partition, partition);
    }
    expect_errors("attention outputs", comparison_family_t::normalized_reduction_k, comparison_verdict_t::assert_k,
                  output_errors, 0);
    expect_errors("attention holds", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, carry_errors,
                  0);
}

/** What any walk's attention must hold, however narrow: every attended row is finite and inside the
 *  hull of the @p widened_values rows, and every carry's partition counts its own maximum's key and
 *  at most one unit per key, that maximum no larger than the exact one over every key by the oracle
 *  @p dots, @b [tasks,keys]. */
inline void dense_expect_attention_bounds(nk::matrix_view<dense_dot_reference_t> dots,
                                          nk::matrix_view<f32_t> widened_values, f32_t score_scale,
                                          nk::matrix_view<f32_t> outputs, std::span<softmax_state_t const> carry) {
    std::size_t const rows_count = dots.extent(1), columns = widened_values.extent(1), tasks = dots.extent(0);
    if (!expect_true(outputs.extent(0) == tasks && outputs.extent(1) == columns && carry.size() == tasks)) return;
    nk::vector<f32_t> lowest = make_vector<f32_t>(columns), highest = make_vector<f32_t>(columns);
    std::fill_n(lowest.data(), columns, (std::numeric_limits<f32_t>::max)());
    std::fill_n(highest.data(), columns, std::numeric_limits<f32_t>::lowest());
    for (std::size_t row = 0; row != rows_count; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            lowest[column] = std::fmin(lowest[column], widened_values(row, column));
            highest[column] = std::fmax(highest[column], widened_values(row, column));
        }
    for (std::size_t task = 0; task != tasks; ++task) {
        f64_t maximum = std::numeric_limits<f64_t>::lowest();
        for (std::size_t row = 0; row != rows_count; ++row)
            maximum = std::fmax(maximum, score_scale * dots(task, row).dot);
        softmax_state_t const state = carry[task];
        expect_true(state.maximum_score <= maximum + 1e-3 * std::fmax(1, std::fabs(maximum)));
        expect_true(state.partition >= 1 - 1e-3 && state.partition <= rows_count * (1 + 1e-3));
        for (std::size_t column = 0; column != columns; ++column) {
            f32_t const value = outputs(task, column);
            f32_t const slack = 1e-3f * std::fmax(1, std::fmax(std::fabs(lowest[column]), std::fabs(highest[column])));
            expect_true(value >= lowest[column] - slack && value <= highest[column] + slack);
        }
    }
}

/** The dot a walk reports for @p left against @p right. Test rows hold small integers, so every
 *  kernel and accumulation order computes it exactly, and the serial kernel stands in for the
 *  kernels of every residency. */
template <typename scalar_type_>
typename scalar_type_::dot_result_t dense_native_dot(nk::vector_view<scalar_type_> left,
                                                     nk::vector_view<scalar_type_> right) noexcept {
    using serial_kit_t = typename quads_ladder<scalar_type_, nk_cap_serial_k>::kit_t;
    auto const right_norms = norms<serial_kit_t>(right);
    return scores<serial_kit_t>(nk_kernel_dot_k, left, norms<serial_kit_t>(left), {right, right, right, right},
                                {right_norms, right_norms, right_norms, right_norms})
        .dots[0];
}

/** One row of graph products: every kept pair names a distinct row of @p vectors, through @p row_of, and
 *  holds its exact dot against @p left. */
template <std::size_t rows_count_, typename scalar_type_, typename id_type_, typename dot_type_, typename row_of_type_>
void dense_expect_product_row(nk::vector_view<scalar_type_> left, nk::matrix_view<scalar_type_> vectors,
                              nk::vector_view<id_type_> ids, nk::vector_view<dot_type_> products,
                              row_of_type_&& row_of) {
    expect_equal(ids.size(), products.size());
    std::array<bool, rows_count_> found {};
    error_stats_t product_errors;
    for (std::size_t pair = 0; pair != ids.size(); ++pair) {
        std::size_t const row = row_of(ids[pair]);
        expect_true(row < rows_count_ && !found[row]);
        if (row >= rows_count_ || found[row]) continue;
        found[row] = true;
        product_errors.add(static_cast<f64_t>(products[pair]),
                           static_cast<f64_t>(dense_native_dot(left, vectors.row(row).as_vector())));
    }
    expect_errors("graph products", comparison_family_t::exact_k, comparison_verdict_t::assert_k, product_errors, 0);
}

/** Graph products against the native dots: every scored pair is counted, a narrow row keeps its width
 *  while counting past it, every kept pair names a distinct stored row with its exact dot, and mismatched
 *  spans refuse. */
template <typename index_type_, typename make_index_type_>
void dense_expect_multiply(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    expect_true(index.reserve({rows_count}).status == status_code_t::success_k);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "multiply"));

    // An empty index scores nothing and writes nothing.
    nk::vector<node_id_t> node_ids = make_vector<node_id_t>(rows_count);
    std::fill_n(node_ids.data(), rows_count, node_id_unknown_k);
    nk::vector<typename index_type_::dot_result_t> products = make_vector<typename index_type_::dot_result_t>(
        rows_count);
    auto const empty = index.multiply(vectors.view().row(0).as_vector(), to_span(node_ids.span()),
                                      to_span(products.span()), {}, context);
    expect_true(empty.status == status_code_t::success_k && empty.value.count == 0);
    expect_true(node_ids[0] == node_id_unknown_k);

    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(rows_count);
    expect_true(index.insert(vectors.view(), assigned_ids.span(), context).status == status_code_t::success_k);

    typename index_type_::search_config_t const exhaustive {
        static_cast<typename index_type_::candidates_count_t>(rows_count)};
    nk::vector_view<node_id_t> const kept_ids = node_ids.view();
    nk::vector_view<typename index_type_::dot_result_t> const kept_products = products.view();
    for (std::size_t const wanted : {rows_count, std::size_t {4}})
        for (std::size_t row = 0; row != left_count; ++row) {
            auto const left = vectors.view().row(row).as_vector();
            nk::range_t const width = nk::range(std::size_t {0}, wanted);
            auto const [multiplied, status] = index.multiply(left, to_span(node_ids[width]), to_span(products[width]),
                                                             exhaustive, context);
            expect_true(status == status_code_t::success_k);
            expect_equal(multiplied.count, multiplied.computed_distances);
            expect_true(multiplied.count > 0 && multiplied.count <= rows_count);
            if (wanted < rows_count) expect_true(multiplied.count > wanted);
            nk::range_t const kept = nk::range(std::size_t {0}, min_of<std::size_t>(multiplied.count, wanted));
            dense_expect_product_row<rows_count>(left, vectors.view(), kept_ids[kept], kept_products[kept],
                                                 [](node_id_t id) noexcept { return std::size_t {id}; });
        }

    auto const refused = index.multiply(vectors.view().row(0).as_vector(), to_span(node_ids.span()),
                                        to_span(products[nk::range(std::size_t {0}, std::size_t {3})]), {}, context);
    expect_true(refused.status == status_code_t::config_mismatch_k);
}

/** @ref dense_expect_multiply for the dot kernel, then the ledger it left must balance; no other
 *  kernel offers it. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_multiply(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k) return;
    else {
        allocation_ledger_t ledger;
        dense_expect_multiply<index_type_>(make_index, ledger, seed);
        return void(expect_equal(ledger.live_bytes, 0));
    }
}

/** Attention against exact full attention in @c f64 under an exhaustive walk, one head per walk and four sharing
 *  one, every node ID its stored row; a narrow walk stays bounded; a node without a value row, a mismatched
 *  output and heads that do not divide the rows refuse. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_attend(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using vector_view_t = typename index_type_::vector_view_t;
    std::size_t const nodes_count = dense_thread_nodes_k, rows_count = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend"));
    nk::matrix<scalar_t> values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "attend", 1));
    nk::matrix<scalar_t> queries = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "attend", 2));
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);
    auto value_of = [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); };
    auto lookup = [&](std::span<node_id_t const> nodes, std::span<vector_view_t> rows) noexcept {
        for (std::size_t lane = 0; lane != nodes.size(); ++lane) rows[lane] = value_of(nodes[lane]);
    };
    nk::matrix<f32_t> outputs = make_matrix<f32_t>(rows_count, dimensions);
    std::fill_n(outputs.data(), outputs.size(), 1.0f);
    nk::vector<softmax_state_t> carry = make_vector<softmax_state_t>(rows_count);

    // An empty index folds nothing: zero rows and empty partitions.
    {
        auto const [attended, status] = index.attend(queries.view(), 1, lookup, 1, outputs.span(),
                                                     to_writable_span(carry), {}, context);
        expect_true(status == status_code_t::success_k && attended.count == rows_count);
        nk::matrix_view<f32_t> const attended_rows = outputs.view();
        expect_true(std::all_of(attended_rows.begin(), attended_rows.end(),
                                [](auto const& entry) { return entry.second == 0; }));
        std::span<softmax_state_t const> const carried = to_span(carry);
        expect_true(
            std::all_of(carried.begin(), carried.end(), [](softmax_state_t state) { return state.partition == 0; }));
    }

    nk::vector<node_id_t> assigned_ids = make_vector<node_id_t>(nodes_count);
    if (failed(index.insert(vectors.view(), {assigned_ids.data(), nodes_count}, context).status))
        return void(expect_true(false));
    expect_contiguous_ids(to_writable_span(assigned_ids), 0);

    nk::matrix<f32_t> widened_queries = make_matrix<f32_t>(rows_count, dimensions);
    nk::matrix<f32_t> widened_vectors = make_matrix<f32_t>(nodes_count, dimensions);
    nk::matrix<dense_dot_reference_t> dots = make_matrix<dense_dot_reference_t>(rows_count, nodes_count);
    dense_widened_rows(queries.view(), index_type_::vector_dtype_k, widened_queries.span());
    dense_widened_rows(vectors.view(), index_type_::vector_dtype_k, widened_vectors.span());
    dense_dots_reference(widened_queries.view(), widened_vectors.view(), dots.span());
    f32_t const score_scale = dense_attend_scale(dots.view());
    nk::matrix<f32_t> widened_values = make_matrix<f32_t>(nodes_count, dimensions);
    dense_widened_rows(values.view(), index_type_::vector_dtype_k, widened_values.span());
    typename index_type_::search_config_t const exhaustive {
        static_cast<typename index_type_::candidates_count_t>(nodes_count)};
    for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
        auto const [attended, status] = index.attend(queries.view(), head_count, lookup, score_scale, outputs.span(),
                                                     to_writable_span(carry), exhaustive, context);
        expect_true(status == status_code_t::success_k);
        expect_equal(attended.count, rows_count);
        dense_expect_full_attention(dots.view(), widened_values.view(), head_count, score_scale, outputs.view(),
                                    to_span(carry));
        expect_true(index
                        .attend(queries.view(), head_count, lookup, score_scale, outputs.span(),
                                to_writable_span(carry), {}, context)
                        .status == status_code_t::success_k);
        dense_expect_attention_bounds(dots.view(), widened_values.view(), score_scale, outputs.view(), to_span(carry));
    }

    auto const row_less = [&](std::span<node_id_t const> nodes, std::span<vector_view_t> rows) noexcept {
        lookup(nodes, rows);
        for (std::size_t lane = 0; lane != nodes.size(); ++lane)
            if (nodes[lane] + 1 == nodes_count) rows[lane] = vector_view_t {};
    };
    expect_true(index
                    .attend(queries.view(), 1, row_less, score_scale, outputs.span(), to_writable_span(carry),
                            exhaustive, context)
                    .status == status_code_t::shape_mismatch_k);
    expect_true(index
                    .attend(queries.view(), 1, lookup, score_scale,
                            outputs(nk::range(std::size_t {0}, rows_count - 1), nk::slice), to_writable_span(carry), {},
                            context)
                    .status == status_code_t::config_mismatch_k);
    expect_true(
        index.attend(queries.view(), 3, lookup, score_scale, outputs.span(), to_writable_span(carry), {}, context)
            .status == status_code_t::shape_mismatch_k);
}

/** @ref dense_expect_attend for the dot kernel, then the ledger it left must balance; no other kernel offers it. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_attend(make_index_type_ const& make_index, u64_t seed) {
    if constexpr (kernel_ != nk_kernel_dot_k) return;
    else {
        allocation_ledger_t ledger;
        dense_expect_attend<index_type_, kernel_>(make_index, ledger, seed);
        return void(expect_equal(ledger.live_bytes, 0));
    }
}

/** Sharded graph products against the native dots: every scored pair is counted, a narrow row keeps
 *  a stride of pairs while counting past it, every kept pair names a distinct stored row with its
 *  exact dot, and a short scored span refuses. */
template <typename indexes_type_, typename... placement_types_>
void dense_expect_indexes_products(u64_t seed, placement_types_&... placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {nk_kernel_dot_k,
                           {dimensions, connectivity_16_k},
                           forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows_count}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                stream_key(seed, "indexes_products"));
    nk::vector<key_t> keys = make_vector<key_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

    auto const left = vectors(nk::range(std::size_t {0}, left_count), nk::slice);
    nk::vector<key_t> found_keys = make_vector<key_t>(left_count * rows_count);
    nk::vector<dot_result_t> found_products = make_vector<dot_result_t>(left_count * rows_count);
    std::array<std::size_t, left_count + 1> offsets {};
    std::array<std::size_t, left_count> scored {};
    // A narrow call keeps a stride of every row's pairs, so each run publishes at most that many.
    auto multiply = [&](std::size_t wanted, std::span<std::size_t> scored_span) noexcept {
        offsets[0] = 0;
        products_tape<key_t, dot_result_t> const products {
            to_writable_span(found_keys), to_writable_span(found_products), std::span {offsets}, wanted};
        return drive_multiply(indexes, left, products, scored_span, typename indexes_type_::search_config_t {}, driver);
    };
    nk::vector_view<key_t> const found_keys_view = found_keys.view();
    nk::vector_view<dot_result_t> const found_products_view = found_products.view();
    for (std::size_t const wanted : {rows_count, std::size_t {4}}) {
        auto const [multiplied, status] = multiply(wanted, std::span {scored});
        expect_true(succeeded(status));
        expect_equal(multiplied.count, multiplied.computed_distances);
        for (std::size_t row = 0; row != left_count; ++row) {
            expect_true(scored[row] > 0 && scored[row] <= rows_count);
            if (wanted < rows_count) expect_true(scored[row] > wanted);
            std::size_t const first = offsets[row], kept = min_of(scored[row], wanted);
            expect_equal(offsets[row + 1] - first, kept);
            if (offsets[row + 1] - first != kept) continue;
            nk::range_t const run = nk::range(first, first + kept);
            dense_expect_product_row<rows_count>(left.row(row).as_vector(), vectors.view(), found_keys_view[run],
                                                 found_products_view[run],
                                                 [](key_t key) noexcept { return dense_key_number(key) - 1; });
        }
    }

    expect_true(multiply(rows_count, std::span {scored}.first(left_count - 1)).status ==
                status_code_t::config_mismatch_k);
}

/** @ref dense_expect_indexes_products for the dot kernel; no other kernel offers graph products. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_products(u64_t seed, placement_types_&... placement) {
    if constexpr (kernel_ == nk_kernel_dot_k) dense_expect_indexes_products<indexes_type_>(seed, placement...);
}

/** Sharded MaxSim against the native dots: 64 documents of 4 token rows share their keys, a walk
 *  as wide as the index ranks the oracle's best documents, a default one ranks only keys of the
 *  index, every query keeps the wanted count on a tape with room for more, and offsets past the
 *  rows refuse. Host towers also rank a scan, cut queries of 3 and 5 rows from one batch, and rank
 *  a 16-row query, as wide as a matrix engine's tile. Every score is recorded against its exact
 *  one and warns past three sigmas: the reducer truncates each row's ranking and imputes the rows
 *  it misses at the floor, so no path promises exact scores. */
template <typename indexes_type_, typename... placement_types_>
void dense_expect_indexes_maxsim(u64_t seed, placement_types_&... placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using search_config_t = typename indexes_type_::search_config_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4;
    constexpr std::size_t documents_count = rows_count / tokens_per_document;
    // The tape holds more than a query wants, so a publish cutting runs at the stride shows.
    constexpr std::size_t tokens_per_query = 4, queries_count = 3, wanted = 5, stride = wanted + 3;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {nk_kernel_dot_k,
                           {dimensions, connectivity_16_k},
                           forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows_count}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                stream_key(seed, "indexes_maxsim"));
    nk::vector<key_t> keys = make_vector<key_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row / tokens_per_document);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

    // A document's exact score over query rows [first, end), the leading stored rows: every row's
    // best dot among its tokens, summed. Test rows are non-negative, so every dot is too, and a row
    // that misses a document only lowers its score.
    auto const queries = vectors(nk::range(std::size_t {0}, queries_count * tokens_per_query), nk::slice);
    auto exact_of = [&](std::size_t first, std::size_t end, std::size_t document) noexcept {
        f64_t total = 0;
        for (std::size_t row = first; row != end; ++row) {
            f64_t best = 0;
            for (std::size_t token = 0; token != tokens_per_document; ++token)
                best = max_of(best, static_cast<f64_t>(dense_native_dot(
                                        vectors.view().row(row).as_vector(),
                                        vectors.view().row(document * tokens_per_document + token).as_vector())));
            total += best;
        }
        return total;
    };
    nk::matrix<f64_t> exact = make_matrix<f64_t>(queries_count, documents_count);
    for (std::size_t query = 0; query != queries_count; ++query)
        for (std::size_t document = 0; document != documents_count; ++document)
            exact(query, document) = exact_of(query * tokens_per_query, (query + 1) * tokens_per_query, document);

    nk::vector<key_t> found_keys = make_vector<key_t>(queries_count * stride);
    nk::vector<f64_t> found_scores = make_vector<f64_t>(queries_count * stride);
    std::array<std::size_t, queries_count + 1> offsets {};
    std::array<std::size_t, queries_count> scored {};
    std::array<std::size_t, queries_count + 1> uniform_offsets;
    for (std::size_t query = 0; query != uniform_offsets.size(); ++query)
        uniform_offsets[query] = query * tokens_per_query;
    auto maxsim_cut = [&](nk::matrix_view<scalar_t> rows, std::span<std::size_t const> query_offsets,
                          search_config_t config) noexcept {
        std::size_t const placed = query_offsets.size() - 1;
        offsets[0] = 0;
        products_tape<key_t, f64_t> const matches {to_writable_span(found_keys), to_writable_span(found_scores),
                                                   std::span {offsets}.first(placed + 1), stride};
        return drive_maxsim(indexes, rows, query_offsets, std::span {scored}.first(placed), matches, {config, wanted},
                            driver);
    };
    auto maxsim = [&](nk::matrix_view<scalar_t> rows, search_config_t config) noexcept {
        return maxsim_cut(rows, std::span<std::size_t const> {uniform_offsets}, config);
    };
    auto kept_of = [&](std::size_t query) noexcept { return offsets[query + 1] - offsets[query]; };
    auto document_of = [&](std::size_t query, std::size_t rank) noexcept {
        return dense_key_number(found_keys[offsets[query] + rank]) - 1;
    };

    // Every placed query scores every document and ranks the best of `expected_scores`.
    error_stats_t score_errors;
    nk::vector<f64_t> ranked = make_vector<f64_t>(documents_count);
    auto expect_ranked = [&](nk::matrix_view<f64_t> expected_scores, std::size_t placed) noexcept {
        for (std::size_t query = 0; query != placed; ++query) {
            for (std::size_t document = 0; document != documents_count; ++document)
                ranked[document] = expected_scores(query, document);
            std::span<f64_t> const descending = to_writable_span(ranked);
            std::sort(descending.rbegin(), descending.rend());
            expect_equal(kept_of(query), wanted);
            expect_equal(scored[query], documents_count);
            for (std::size_t rank = 0; rank != kept_of(query); ++rank) {
                std::size_t const document = document_of(query, rank);
                if (!expect_true(document < documents_count)) continue;
                score_errors.add(found_scores[offsets[query] + rank], ranked[rank]);
                score_errors.add(found_scores[offsets[query] + rank], expected_scores(query, document));
            }
        }
    };

    // A walk as wide as the index ranks every token, so the best documents are kept.
    search_config_t const exhaustive {static_cast<typename indexes_type_::index_t::candidates_count_t>(rows_count)};
    expect_true(succeeded(maxsim(queries, exhaustive).status));
    expect_ranked(exact.view(), queries_count);

    // A default walk may miss a document and score the ones it nominates from truncated rankings.
    expect_true(succeeded(maxsim(queries, {}).status));
    for (std::size_t query = 0; query != queries_count; ++query)
        for (std::size_t rank = 0; rank != kept_of(query); ++rank) {
            std::size_t const document = document_of(query, rank);
            expect_true(document < documents_count);
            if (document >= documents_count) continue;
            score_errors.add(found_scores[offsets[query] + rank], exact(query, document));
        }

    // The device towers take no scan and whole batches of uniform queries only.
    if constexpr (!requires { indexes_type_::vendor_k; }) {
        // A scan nominates every document, so it ranks as exactly as the widest walk.
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;
        expect_true(succeeded(maxsim(queries, scan).status));
        expect_ranked(exact.view(), queries_count);

        // Queries of 3 and 5 rows cut from one batch rank as their own rows dictate.
        std::array<std::size_t, 3> const ragged_offsets {0, 3, 8};
        nk::matrix<f64_t> ragged_exact = make_matrix<f64_t>(queries_count, documents_count);
        for (std::size_t query = 0; query + 1 != ragged_offsets.size(); ++query)
            for (std::size_t document = 0; document != documents_count; ++document)
                ragged_exact(query, document) = exact_of(ragged_offsets[query], ragged_offsets[query + 1], document);
        expect_true(succeeded(maxsim_cut(queries, std::span<std::size_t const> {ragged_offsets}, exhaustive).status));
        expect_ranked(ragged_exact.view(), ragged_offsets.size() - 1);

        // A query as wide as a matrix engine's tile folds through it, walked or scanned, exactly.
        std::array<std::size_t, 2> const wide_offsets {0, 16};
        nk::matrix<f64_t> wide_exact = make_matrix<f64_t>(queries_count, documents_count);
        for (std::size_t document = 0; document != documents_count; ++document)
            wide_exact(0, document) = exact_of(0, 16, document);
        auto const wide_queries = vectors(nk::range(std::size_t {0}, std::size_t {16}), nk::slice);
        for (search_config_t const config : {exhaustive, scan}) {
            expect_true(
                succeeded(maxsim_cut(wide_queries, std::span<std::size_t const> {wide_offsets}, config).status));
            expect_ranked(wide_exact.view(), 1);
        }
    }

    // Offsets past the rows refuse their layout.
    expect_equal(maxsim(queries(nk::range(std::size_t {0}, tokens_per_query + 1), nk::slice), {}).status,
                 status_code_t::config_mismatch_k);
    // Every residency sums each row's best native dot over exact rankings: all assert alike.
    expect_errors("maxsim scores", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, score_errors, 0);
}

/** @ref dense_expect_indexes_maxsim for the dot kernel; no other kernel offers MaxSim. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_maxsim(u64_t seed, placement_types_&... placement) {
    if constexpr (kernel_ == nk_kernel_dot_k) dense_expect_indexes_maxsim<indexes_type_>(seed, placement...);
}

/** Attention across the shards against exact full attention in @c f64 under exhaustive walks, one head per walk
 *  and four sharing one, rows keyed by their own position: every shard's walk folds into one running state per
 *  head. Narrow walks stay bounded; a key without a value row, a mismatched output and heads that do not divide
 *  the rows refuse. Only the dot kernel attends. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_attend(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using vector_view_t = typename index_t::vector_view_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return;
    else {
        std::size_t const nodes_count = dense_thread_nodes_k, rows_count = 8;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes)));
        expect_true(succeeded(indexes.reserve({nodes_count}).status));

        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return void(expect_true(false));
        nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions,
                                                              stream_key(seed, "indexes_attend"));
        nk::matrix<scalar_t> values = make_vectors<scalar_t>(nodes_count, dimensions,
                                                             stream_key(seed, "indexes_attend", 1));
        nk::matrix<scalar_t> queries = make_vectors<scalar_t>(rows_count, dimensions,
                                                              stream_key(seed, "indexes_attend", 2));
        nk::vector<key_t> keys = make_vector<key_t>(nodes_count);
        for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = dense_key<key_t>(row);
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

        auto value_of = [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); };
        auto lookup = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
            for (std::size_t lane = 0; lane != batch.size(); ++lane)
                rows[lane] = value_of(dense_key_number(batch[lane]));
        };
        nk::matrix<f32_t> widened_queries = make_matrix<f32_t>(rows_count, dimensions);
        nk::matrix<f32_t> widened_vectors = make_matrix<f32_t>(nodes_count, dimensions);
        nk::matrix<dense_dot_reference_t> dots = make_matrix<dense_dot_reference_t>(rows_count, nodes_count);
        dense_widened_rows(queries.view(), index_t::vector_dtype_k, widened_queries.span());
        dense_widened_rows(vectors.view(), index_t::vector_dtype_k, widened_vectors.span());
        dense_dots_reference(widened_queries.view(), widened_vectors.view(), dots.span());
        f32_t const score_scale = dense_attend_scale(dots.view());
        nk::matrix<f32_t> widened_values = make_matrix<f32_t>(nodes_count, dimensions);
        dense_widened_rows(values.view(), index_t::vector_dtype_k, widened_values.span());
        nk::matrix<f32_t> outputs = make_matrix<f32_t>(rows_count, dimensions);
        nk::vector<softmax_state_t> carry = make_vector<softmax_state_t>(rows_count);

        typename index_t::search_config_t const exhaustive {
            static_cast<typename index_t::candidates_count_t>(nodes_count)};
        for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
            auto const [attended, status] = drive_attend(indexes, queries.view(), head_count, lookup, score_scale,
                                                         outputs.span(), to_writable_span(carry), exhaustive, driver);
            expect_true(succeeded(status));
            expect_equal(attended.count, rows_count);
            dense_expect_full_attention(dots.view(), widened_values.view(), head_count, score_scale, outputs.view(),
                                        to_span(carry));
            expect_true(succeeded(drive_attend(indexes, queries.view(), head_count, lookup, score_scale, outputs.span(),
                                               to_writable_span(carry), {}, driver)
                                      .status));
            dense_expect_attention_bounds(dots.view(), widened_values.view(), score_scale, outputs.view(),
                                          to_span(carry));
        }

        auto const row_less = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
            lookup(batch, rows);
            for (std::size_t lane = 0; lane != batch.size(); ++lane)
                if (dense_key_number(batch[lane]) + 1 == nodes_count) rows[lane] = vector_view_t {};
        };
        expect_true(drive_attend(indexes, queries.view(), 1, row_less, score_scale, outputs.span(),
                                 to_writable_span(carry), exhaustive, driver)
                        .status == status_code_t::shape_mismatch_k);
        expect_true(drive_attend(indexes, queries.view(), 1, lookup, score_scale,
                                 outputs(nk::range(std::size_t {0}, rows_count - 1), nk::slice),
                                 to_writable_span(carry), {}, driver)
                        .status == status_code_t::config_mismatch_k);
        expect_true(drive_attend(indexes, queries.view(), 3, lookup, score_scale, outputs.span(),
                                 to_writable_span(carry), {}, driver)
                        .status == status_code_t::shape_mismatch_k);
        return;
    }
}

#pragma endregion Algebra

#pragma region Races

/** Concurrent construction on ForkUnion's pool: distinct slices per thread, one traversal context
 *  each, then ID-uniqueness and sampled self-recall checks. A device index admits one batch at a
 *  time and answers the overlapping ones with @c contended_k, which retry. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_concurrent_build(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const writer_threads_count = 4;
    std::size_t const nodes_count = writer_threads_count * dense_thread_nodes_k;

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) ||
        fu::failed(pool.spawn(topology, writer_threads_count, fu::caller_inclusive_k)))
        return void(expect_true(false));

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);

    auto threads = dense_make_threads<index_type_>(make_index, writer_threads_count, dense_thread_nodes_k,
                                                   dense_dimensions_for<scalar_t>(dense_values_k),
                                                   stream_key(seed, "concurrent_build"));
    nk::matrix<node_id_t> assigned_ids = make_matrix<node_id_t>(writer_threads_count, dense_thread_nodes_k);
    std::size_t thread_failures = 0;

    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= writer_threads_count) return;
        status_t status = status_code_t::contended_k;
        while (status == status_code_t::contended_k)
            status = index
                         .insert({keys.data() + thread * dense_thread_nodes_k, dense_thread_nodes_k},
                                 threads.slices[thread].view(), assigned_ids.row(thread).as_vector(),
                                 threads.contexts[thread])
                         .status;
        if (status != status_code_t::success_k) fu::standard_atomic_ref<std::size_t>(thread_failures).store_add(1);
        assert_no_locks_held();
    })));

    expect_equal(thread_failures, 0u);
    expect_equal(index.size(), nodes_count);
    // One batch's IDs come from one dispense: contiguous, in the order of its rows.
    // Walked batch by batch: MSVC 14.51 `/O2` vectorizes a `row % batch` loop into wild reads.
    std::size_t scattered_ids = 0;
    for (std::size_t thread = 0; thread != writer_threads_count; ++thread)
        for (std::size_t offset = 0; offset != dense_thread_nodes_k; ++offset)
            scattered_ids += assigned_ids(thread, offset) != assigned_ids(thread, 0) + offset;
    expect_equal(scattered_ids, 0u);
    expect_contiguous_ids(to_writable_span(assigned_ids), 0);

    if constexpr (kernel_ != nk_kernel_dot_k)
        for (std::size_t thread = 0; thread != writer_threads_count; ++thread)
            dense_expect_self_recall(index, make_index, threads.slices[thread].view(), dense_thread_nodes_k);
}

/** @ref dense_expect_concurrent_build, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_concurrent_build(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_concurrent_build<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/**
 *  @brief A reader thread races three writer threads, asserting only interleaving-independent
 *      invariants.
 *
 *  Every search succeeds, respects the candidate breadth bound, returns distances in ascending
 *  order, and never surfaces an ID at or above the watermark it could have observed - the
 *  translation-frontier invariant, audited live.
 *
 *  The query vectors are never inserted: a search does not require its query to be stored, so
 *  the reader & writers share only the index itself, and any sanitizer report is an index bug
 *  rather than test-harness noise.
 */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_concurrent_search(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const writer_threads_count = 3, queries_count = 64;
    std::size_t const nodes_count = writer_threads_count * dense_thread_nodes_k;
    std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) ||
        fu::failed(pool.spawn(topology, writer_threads_count + 1, fu::caller_inclusive_k)))
        return void(expect_true(false));

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);

    auto threads = dense_make_threads<index_type_>(make_index, writer_threads_count + 1, dense_thread_nodes_k,
                                                   dense_dimensions_for<scalar_t>(dense_values_k),
                                                   stream_key(seed, "concurrent_search"));
    nk::matrix<scalar_t> query_vectors = make_vectors<scalar_t>(
        queries_count, dense_dimensions_for<scalar_t>(dense_values_k), stream_key(seed, "concurrent_search", 1));

    std::size_t writer_lanes_remaining_count = writer_threads_count, thread_failures_count = 0;
    fu::standard_atomic_ref<std::size_t> const writer_lanes_remaining {writer_lanes_remaining_count};
    fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};

    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread != 0 && thread <= writer_threads_count) {
            status_t status = status_code_t::contended_k;
            while (status == status_code_t::contended_k)
                status = index
                             .insert({keys.data() + (thread - 1) * dense_thread_nodes_k, dense_thread_nodes_k},
                                     threads.slices[thread].view(), {}, threads.contexts[thread])
                             .status;
            if (status != status_code_t::success_k) thread_failures.store_add(1);
            writer_lanes_remaining.store_sub(1, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        if (thread != 0) return;

        // Every other task runs the predicated sibling under the same race: all the
        // unfiltered invariants must hold, plus admission purity - no ID the predicate
        // rejects may ever surface, no matter how the writers interleave.
        auto odd_only = quad_predicate(
            [](node_id_t node_id) noexcept { return node_id % 2 == 1 ? admission_t::admit_k : admission_t::reject_k; });
        // A device index answers the predicated sibling with `device_k`, its walk running no
        // host predicate, and takes the unfiltered tasks alone.
        bool const predicable =
            index.search_if(query_vectors.view().row(0).as_vector(), {}, odd_only, threads.contexts[0]).status !=
            status_code_t::device_k;
        auto search_step = [&](typename index_type_::vector_view_t query, bool predicated) {
            if (predicated) return index.search_if(query, {}, odd_only, threads.contexts[0]);
            return index.search(query, {}, threads.contexts[0]);
        };
        for (std::size_t task = 0; task < dense_reader_minimal_searches_k || //
                                   writer_lanes_remaining.load(std::memory_order_acquire) != 0;
             ++task) {
            auto query = query_vectors.view().row(task % queries_count).as_vector();
            bool const predicated = predicable && task % 2 == 1;
            auto const [result, status] = search_step(query, predicated);
            if (status != status_code_t::success_k) thread_failures.store_add(1);
            thread_failures.store_add(
                scored_violations(threads.contexts[0].results.data(), result.count, index.size(), expansion_bound));
            // Every writer inserts its slice as one batch, which publishes whole.
            if (index.size() % dense_thread_nodes_k) thread_failures.store_add(1);
            if (predicated)
                for (std::size_t match = 0; match != result.count; ++match)
                    if (threads.contexts[0].results[match].node_id % 2 != 1) thread_failures.store_add(1);
        }
        assert_no_locks_held();
    })));

    expect_equal(thread_failures.load(), 0u);
    expect_equal(index.size(), nodes_count);
}

/** @ref dense_expect_concurrent_search, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_concurrent_search(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_concurrent_search<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** The sharded index's refusal surface, all deterministic: verbs before placement,
 *  placement after reservation, geometry drift, mismatched spans, and a doubly-claimed
 *  executor - every misuse answers with a status, never with a write. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_guards(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 64;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "guards"));
    nk::matrix<scalar_t> oversized = make_vectors<scalar_t>(rows + 1, dimensions, stream_key(seed, "guards", 1));
    nk::vector<key_t> keys = make_vector<key_t>(rows + 1);
    for (std::size_t row = 0; row != rows + 1; ++row) keys[row] = dense_key<key_t>(1 + row);

    // Verbs before any shard exists refuse rather than write.
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(indexes
                     .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                             executor)
                     .status,
                 status_code_t::uninitialized_k);
    expect_equal(indexes.reserve({rows}).status, status_code_t::uninitialized_k);

    // Reserving closes the shard set.
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows}).status));
    expect_equal(place_shards(indexes, placement...), status_code_t::config_mismatch_k);
    auto const [reserved, reserve_status] = indexes.reserve({rows});
    expect_true(succeeded(reserve_status) && reserved == indexes.capacity());

    // A keys span disagreeing with the batch's rows refuses before anything is taken.
    expect_equal(indexes
                     .insert(std::span<key_t const> {keys.data(), rows - 1}, vectors.view(), //
                             executor)
                     .status,
                 status_code_t::config_mismatch_k);

    // The executor admits one caller: a live claim turns the second away, and releasing
    // it restores service - which the closing insertion proves end to end.
    {
        auto [claim, status] = executor.claim();
        expect_true(succeeded(status));
        expect_equal(executor.claim().status, status_code_t::contended_k);
        expect_equal(indexes
                         .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                 executor)
                         .status,
                     status_code_t::contended_k);
    }
    expect_true(succeeded(indexes
                              .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                      executor)
                              .status));
    expect_equal(indexes.size(), rows);
    std::size_t rows_in_shards = 0;
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) rows_in_shards += indexes.shard(shard).size();
    expect_equal(rows_in_shards, rows);

    // A batch past the reservation grows into it - whole, with no ID holes - and the capacity
    // reported afterwards covers every entry that placed.
    expect_true(succeeded(indexes
                              .insert(std::span<key_t const> {keys.data(), rows + 1}, oversized.view(), //
                                      executor)
                              .status));
    expect_equal(indexes.size(), 2 * rows + 1);
    expect_true(indexes.capacity() >= indexes.size());
}

/** A composition grows past any reservation: entries well beyond it are inserted one at a time and
 *  batched, every key stays findable, and a second, larger reservation is a hint, not a refusal. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_growth(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const reserved = 8, singles = 1100, batched = 1100, rows = singles + batched;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({reserved}).status));

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_growth"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
    if (failed(rows_driver.status)) return void(expect_true(false));
    for (std::size_t row = 0; row != singles; ++row)
        expect_true(
            succeeded(indexes.insert(keys[row], vectors.view().row(row).as_vector(), rows_driver.handle()).status));
    expect_true(succeeded(indexes
                              .insert(std::span<key_t const> {keys.data() + singles, batched},
                                      vectors(nk::range(singles, rows), nk::slice), driver.executor)
                              .status));

    expect_equal(indexes.size(), rows);
    expect_true(indexes.capacity() >= rows);
    std::size_t found = 0;
    for (key_t const key : keys) found += indexes.contains(key);
    expect_equal(found, rows);
    nk::matrix<scalar_t> fetched = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "indexes_growth", 1));
    expect_equal(indexes.get(keys[rows - 1], fetched.span(), 1, driver.executor), 1u);
    expect_true(succeeded(indexes.reserve({rows * 2}).status));
    expect_true(indexes.capacity() >= rows * 2);
}

/** A sharded pass under narrow build candidates: the first, cancelled at every barrier and resumed,
 *  re-wires nodes and reports one phase's units at each barrier, summed over the shards in that
 *  phase alone; the second re-wires fewer; each heals the islands it leaves without losing reach,
 *  and the audit after them counts what the last pass ended on. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_refine(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows}).status));

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_refine"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    typename index_t::insert_config_t const narrow {4};
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), executor, narrow).status));

    // The first pass is cancelled at every barrier and resumed. Shards drift apart - a health check
    // takes one barrier per depth of its own - so a barrier's units are the lead shard's phase's:
    // twice the entries of the shards in a health check, once those of the shards re-stitching.
    auto one_phase_units = [&](std::size_t total, std::size_t unit) noexcept {
        std::size_t const shards = indexes.shards_count();
        for (std::size_t subset = 1; subset != (std::size_t {1} << shards); ++subset) {
            std::size_t entries = 0;
            for (std::size_t shard = 0; shard != shards; ++shard)
                if (subset >> shard & 1) entries += indexes.shard(shard).size();
            if (total == unit * entries) return true;
        }
        return false;
    };
    auto refine_resumed = [&] {
        auto [plan, plan_status] = indexes.plan_refine({}, executor);
        expect_equal(plan_status, status_code_t::success_k);
        auto cancel = [](std::size_t, std::size_t) noexcept { return continuation_t::cancel_k; };
        while (drive_run(plan, driver, cancel) == status_code_t::cancelled_k) {
            if (plan.phase() == phase_t::health_k) expect_true(one_phase_units(plan.total(), 2));
            if (plan.phase() == phase_t::restitch_k) expect_true(one_phase_units(plan.total(), 1));
        }
        return drive_finish(plan, driver);
    };
    auto const [first, first_status] = refine_resumed();
    expect_true(succeeded(first_status));
    expect_true(first.reselected_nodes >= rows && first.changed_nodes <= first.reselected_nodes);
    expect_true(first.changed_nodes > 0);
    expect_true(first.reachability.reachable_nodes <= rows &&
                first.reachability.reachable_nodes + first.islands_found >= rows);

    auto const [second, second_status] = indexes.refine({}, executor);
    expect_true(succeeded(second_status));
    expect_true(second.changed_nodes < first.changed_nodes);

    // The audit reads the graph the last pass left, so it counts what that pass ended on.
    auto const [health, health_status] = indexes.health(executor);
    expect_true(succeeded(health_status));
    expect_equal(health.reachable_nodes, second.reachability.reachable_nodes);
    expect_equal(health.audited_nodes, second.reachability.audited_nodes);
    expect_true(health.depth_from_root >= 1);

    // A plan dropped after a cancelled run heals nothing, but reopens the index it held closed.
    key_t const late_key = dense_key<key_t>(rows + 1);
    auto insert_late = [&] {
        return indexes.insert(std::span<key_t const> {&late_key, 1}, vectors(nk::range(0, 1), nk::slice), executor)
            .status;
    };
    {
        auto [plan, plan_status] = indexes.plan_refine({}, executor);
        expect_equal(plan_status, status_code_t::success_k);
        auto cancel = [](std::size_t, std::size_t) noexcept { return continuation_t::cancel_k; };
        expect_equal(drive_run(plan, driver, cancel), status_code_t::cancelled_k);
        expect_equal(insert_late(), status_code_t::contended_k);
        expect_equal(indexes.plan_refine({}, executor).status, status_code_t::contended_k);
    }
    expect_equal(insert_late(), status_code_t::success_k);
    expect_equal(indexes.size(), rows + 1);
}

/** Executor arenas seeded at exactly the first-size estimate serve every walk: a re-table past the
 *  seed chunk goes in a chunk the arena opens beside it, so no walk saturates and every query
 *  ranks. Candidates as wide as the whole reservation make every walk exhaustive by construction -
 *  approximate search promises nothing narrower - so every thread's visited set outgrows its first
 *  size, and every key a sparse predicate admits surfaces. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_arena_growth(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 8 * dense_thread_nodes_k;
    std::size_t const queries = 16;
    std::size_t const wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({rows}).status));

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows + queries, dimensions, stream_key(seed, "upstream"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    auto const [insert_result, insert_status] = indexes.insert(to_span(keys), vectors(nk::range(0, rows), nk::slice), //
                                                               executor);
    expect_true(succeeded(insert_status));
    expect_equal(insert_result.saturated_walks, 0u);

    // Sixteen keys match; with candidates as wide as the reservation every walk visits its whole
    // shard, so all eight surface - narrower candidates could promise only that what surfaces is
    // admitted and ranked.
    typename index_t::search_config_t const exhaustive {static_cast<typename index_t::candidates_count_t>(rows)};
    auto sparse = quad_predicate([](key_t key) noexcept {
        return dense_key_number(key) % 128 == 0 ? admission_t::admit_k : admission_t::reject_k;
    });
    nk::matrix<key_t> found_keys = make_matrix<key_t>(queries, wanted);
    nk::matrix<typename indexes_type_::distance_t> found_distances = make_matrix<typename indexes_type_::distance_t>(
        queries, wanted);
    nk::vector<std::size_t> counts = make_vector<std::size_t>(queries);
    auto const [search_result, search_status] = indexes.search_if(
        vectors(nk::range(rows, rows + queries), nk::slice),
        ranked_matches_span<key_t, typename indexes_type_::distance_t> {found_keys.span(), found_distances.span(),
                                                                        to_writable_span(counts)},

        exhaustive, sparse, executor, to_writable_span(driver.traversals));
    expect_true(succeeded(search_status));
    expect_equal(search_result.saturated_walks, 0u);
    for (std::size_t const count : counts) expect_equal(count, wanted);
    // The seed chunk is exactly the estimate, so a re-table can only have opened another.
    expect_true(executor.arena(0).capacity() > indexes.shard(0).traversal_memory_usage());
}

/** One key, several entries, which every key lookup finds across the shards, driven through
 *  the single-row @c insert with a caller-built arena context. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_duplicate_keys(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_tiny_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows}).status));

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "guards", 2));
    dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
    if (failed(rows_driver.status)) return void(expect_true(false));

    // Every second entry is inserted under one shared key; the rest keep distinct ones.
    key_t const shared = dense_key<key_t>(1000);
    for (std::size_t row = 0; row != rows; ++row) {
        key_t const key = row % 2 ? shared : dense_key<key_t>(1 + row);
        expect_true(succeeded(indexes.insert(key, vectors.view().row(row).as_vector(), rows_driver.handle()).status));
    }
    expect_equal(indexes.size(), rows);
    expect_equal(indexes.count(shared), rows / 2);
    expect_equal(indexes.count(dense_key<key_t>(1)), 1u);
    expect_true(indexes.contains(shared));
    expect_true(!indexes.contains(dense_key<key_t>(2)));

    // The probe-then-fetch contract of `get`: the true total returns even when the copy
    // truncates, and an absent key answers zero without touching the output.
    nk::matrix<scalar_t> fetched = make_vectors<scalar_t>(2, dimensions, stream_key(seed, "guards", 3));
    expect_equal(indexes.get(shared, fetched.span(), 2, driver.executor), rows / 2);
    expect_equal(indexes.get(dense_key<key_t>(2), fetched.span(), 2, driver.executor), 0u);

    // The batched verbs answer what the single-key ones do, repeats and absent keys included, and
    // every key's entries go in its own band of the batch, in the single-key order.
    std::array<key_t, 4> const probes {shared, dense_key<key_t>(1), dense_key<key_t>(2), shared};
    std::span<key_t const> const probed {probes};
    std::array<std::size_t, 4> counted {}, contained {}, gotten {};
    nk::matrix<scalar_t> batch = make_vectors<scalar_t>(probes.size() * 2, dimensions, stream_key(seed, "guards", 4));
    expect_true(succeeded(indexes.count(probed, std::span {counted}, driver.executor)));
    expect_true(succeeded(indexes.contains(probed, std::span {contained}, driver.executor)));
    expect_true(succeeded(indexes.get(probed, batch.span(), std::span {gotten}, 2, driver.executor)));
    std::size_t const row_bytes = dense_rows_bytes(fetched.view(), 1);
    for (std::size_t probe = 0; probe != probes.size(); ++probe) {
        expect_equal(counted[probe], indexes.count(probes[probe]));
        expect_equal(contained[probe], indexes.contains(probes[probe]) ? 1u : 0u);
        expect_equal(gotten[probe], indexes.get(probes[probe], fetched.span(), 2, driver.executor));
        for (std::size_t row = 0; row != min_of<std::size_t>(gotten[probe], 2); ++row)
            expect_true(std::memcmp(batch.view().row(probe * 2 + row).as_vector().byte_data(),
                                    fetched.view().row(row).as_vector().byte_data(), row_bytes) == 0);
    }
    expect_equal(indexes.get(probed, batch.span(), std::span {gotten}, 3, driver.executor),
                 status_code_t::config_mismatch_k);
}

/** A save config the sharded scenarios pass: the device residency certifies quiescence, since
 *  its flat matrices cannot be settlement-probed; the RAM residency lets the probe run. */
template <typename indexes_type_>
save_config_t dense_indexes_save_config() noexcept {
    save_config_t config;
    if constexpr (dense_device_resident<typename indexes_type_::index_t>)
        config.synchronization = synchronization_t::immutable_k;
    return config;
}

/** The path one sharded checkpoint scenario writes, in a scratch directory of its own: a lone
 *  shard's monolith, or a manifest with its parts beside it under the interchange's naming. */
struct dense_checkpoint_path_t {
    scratch_directory_t directory;
    std::string path = directory.path("checkpoint.safegraphs");

    /** The file of part @p part among @p parts: the path itself for a lone part. */
    std::string part(std::size_t part, std::size_t parts) const {
        if (parts == 1) return path;
        growing_buffer<char> part_path {global_allocator};
        auto const [named, status] = safegraphs_part_path(part_path, safegraphs_stem(path), part, parts);
        return succeeded(status) ? std::string {named} : std::string {};
    }
};

/** Builds @p rows keyed @c 1+row into @p indexes, freshly placed over @p placement and reserved. */
template <typename indexes_type_, typename executor_type_, typename vectors_type_, typename keys_type_,
          typename... placement_types_>
void dense_indexes_build(indexes_type_& indexes, executor_type_& executor, vectors_type_ const& vectors,
                         keys_type_ const& keys, placement_types_&... placement) {
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = keys.size();
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({rows}).status));
    expect_true(succeeded(indexes
                              .insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), //
                                      executor)
                              .status));
}

/** Every key of @p keys resolves to exactly one entry of @p indexes, and sampled entries read back
 *  through @p executor come back byte-identical to @p vectors. */
template <typename indexes_type_, typename vectors_type_, typename keys_type_>
void dense_indexes_expect_restored(indexes_type_& indexes, vectors_type_ const& vectors, keys_type_ const& keys,
                                   std::size_t dimensions, typename indexes_type_::executor_t& executor, u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    std::size_t const rows = keys.size();
    expect_equal(indexes.size(), rows);
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != rows; ++row) miscounted += indexes.count(keys[row]) != 1;
    expect_equal(miscounted, 0u);

    nk::matrix<scalar_t> fetched = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "roundtrip", 2));
    std::size_t const row_bytes = static_cast<std::size_t>(vectors.view().stride_bytes(0));
    std::size_t mismatched = 0;
    for (std::size_t sample = 0; sample != 8; ++sample) {
        std::size_t const row = sample * (rows / 8);
        if (indexes.get(keys[row], fetched.span(), 1, executor) != 1) mismatched += 1;
        else
            mismatched += std::memcmp(fetched.data(), //
                                      vectors.view().row(row).as_vector().byte_data(), row_bytes) != 0;
    }
    expect_equal(mismatched, 0u);
}

/** Two indexes built over the halves of one dataset gather into one through reshards into one
 *  target, the second half passing through an index never reserved, which takes its geometry: the
 *  sources end empty, every key resolves to one entry with its vector intact, every shard's round
 *  re-stitches every entry with no walk, and the gathered search finds entries of either half; a
 *  repeated source refuses. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_gather(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = dense_growth_nodes_k, half = rows / 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_gather"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ target {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ relay {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    // Twice the source's shards, so half the target's take no appended part and must still heal.
    expect_true(succeeded(place_shards(target, placement...)) && succeeded(place_shards(target, placement...)));
    expect_true(succeeded(place_shards(source, placement...)) && succeeded(place_shards(relay, placement...)));
    for (indexes_type_* built : {&target, &source}) expect_true(succeeded(built->reserve({half}).status));
    expect_true(succeeded(
        target.insert(std::span<key_t const> {keys.data(), half}, vectors(nk::range(0, half), nk::slice), executor)
            .status));
    expect_true(succeeded(source
                              .insert(std::span<key_t const> {keys.data() + half, rows - half},
                                      vectors(nk::range(half, rows), nk::slice), executor)
                              .status));
    // The target leads its own sources, and keeps its shards as a zero capacity asks.
    auto gather = [&](indexes_type_& into, indexes_type_& from) {
        std::array<indexes_type_*, 2> const sources {&into, &from};
        std::array<indexes_type_*, 1> const targets {&into};
        auto const [report, status] = reshard(std::span<indexes_type_* const> {sources},
                                              std::span<indexes_type_* const> {targets}, executor);
        return expected<refine_report_t> {report.graph, status};
    };
    expect_equal(gather(source, source).status, status_code_t::invalid_argument_k);
    // An empty sharded index keeps its own geometry rather than adopting the source's.
    indexes_type_ wider {kernel_, {2 * dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(wider, placement...)));
    expect_equal(gather(wider, source).status, status_code_t::config_mismatch_k);
    expect_equal(source.size(), rows - half);

    expect_true(succeeded(gather(relay, source).status));
    expect_true(source.size() == 0 && relay.size() == rows - half);
    expect_true(relay.dimensions() == dimensions && relay.connectivity() == connectivity_16_k);

    auto const [gathered, gathered_status] = gather(target, relay);
    expect_true(succeeded(gathered_status));
    expect_equal(relay.size(), 0u);
    dense_indexes_expect_restored(target, vectors, keys, dimensions, executor, seed);
    // The moves place the source's rows with no distance, and the rounds stitch them with no walk.
    expect_true(gathered.reselected_nodes >= rows);
    expect_true(gathered.changed_nodes > 0 && gathered.verdict != refine_verdict_t::stopped_k);
    expect_equal(gathered.traversal_stats.visited_members, 0u);
    // A dot graph's hubs can stall a round short of every node.
    auto const [health, health_status] = target.health(executor);
    expect_true(succeeded(health_status));
    expect_equal(health.reachable_nodes, gathered.reachability.reachable_nodes);
    expect_true(health.reachable_nodes * 100 >= rows * dense_minimal_recall_percent_k);

    // Sampled self-recall through the gathered search, judged by distance band on each half alone,
    // so source entries the refine leaves as islands cannot hide behind the target's.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {target, driver};
        expect_true(succeeded(rows_driver.status));
        std::size_t const samples_count = min_of(recall_samples_k, half);
        nk::vector<key_t> found_keys = make_vector<key_t>(8);
        nk::vector<typename indexes_type_::distance_t> found_distances =
            make_vector<typename indexes_type_::distance_t>(8);
        for (std::size_t const first_row : {std::size_t {0}, half}) {
            std::size_t hits = 0;
            for (std::size_t sample = 0; sample != samples_count; ++sample) {
                std::size_t const row = first_row + sample * (half / samples_count);
                auto const [result, status] = target.search(
                    vectors.view().row(row).as_vector(), to_writable_span(found_keys),
                    to_writable_span(found_distances), {}, rows_driver.handle());
                expect_true(status == status_code_t::success_k && result.count > 0);
                for (std::size_t match = 0; match != result.count; ++match)
                    if (found_distances[match] < dense_self_match_tolerance_k &&
                        found_distances[match] > -dense_self_match_tolerance_k) {
                        ++hits;
                        break;
                    }
            }
            expect_true(hits + recall_margin_k * samples_count >=
                        samples_count * dense_minimal_recall_percent_k / 100.0);
        }
    }
}

/** Four indexes over the quarters of one dataset gather into an empty target through one reshard
 *  plan: a group naming a null source refuses before any moves, leaving every source whole; a plan
 *  whose first run is cancelled at its first barrier has moved every source in, and refuses
 *  insertions, and on a GPU saves, while it lives; the next run continues it, and it finishes its
 *  rounds as an uncancelled reshard of the same quarters does, re-stitching every entry with no
 *  walk, its entries spread evenly over the target's shards and each key resolving to its own
 *  entry, found by its key. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_gather_many(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = dense_growth_nodes_k, quarter = rows / 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_gather_many"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    auto build_quarters = [&](std::array<std::unique_ptr<indexes_type_>, 4>& quarters,
                              std::array<indexes_type_*, 4>& sources) {
        for (std::size_t which = 0; which != quarters.size(); ++which) {
            quarters[which] = std::make_unique<indexes_type_>(
                kernel_, typename indexes_type_::geometry_t {dimensions, connectivity_16_k},
                forkunion_allocator_t {no_memory_domain_k});
            indexes_type_& built = *quarters[which];
            expect_true(succeeded(place_shards(built, placement...)));
            expect_true(succeeded(built.reserve({quarter}).status));
            std::size_t const first = which * quarter;
            expect_true(succeeded(built
                                      .insert(std::span<key_t const> {keys.data() + first, quarter},
                                              vectors(nk::range(first, first + quarter), nk::slice), executor)
                                      .status));
            sources[which] = &built;
        }
    };
    // The furthest count a barrier reports wins.
    std::size_t reported_count = 0;
    fu::standard_atomic_ref<std::size_t> const reported {reported_count};
    auto note = [&](std::size_t visited) noexcept { reported.store_max(visited); };
    auto cancel = [&](std::size_t visited, std::size_t) noexcept { return note(visited), continuation_t::cancel_k; };
    auto proceed = [&](std::size_t visited, std::size_t) noexcept { return note(visited), continuation_t::continue_k; };

    // The baseline: the same quarters gathered in one uncancelled call.
    indexes_type_ baseline {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(baseline, placement...)));
    std::array<std::unique_ptr<indexes_type_>, 4> baseline_quarters;
    std::array<indexes_type_*, 4> baseline_sources {};
    build_quarters(baseline_quarters, baseline_sources);
    std::array<indexes_type_*, 1> const baseline_targets {&baseline};
    expect_equal(reshard(std::span<indexes_type_* const> {baseline_sources},
                         std::span<indexes_type_* const> {baseline_targets}, executor, {}, proceed)
                     .status,
                 status_code_t::success_k);
    std::size_t const baseline_visited = reported.exchange(0);

    indexes_type_ target {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(target, placement...)));
    std::array<std::unique_ptr<indexes_type_>, 4> quarters;
    std::array<indexes_type_*, 4> sources {};
    build_quarters(quarters, sources);

    // Validation comes before the first move, so a null slot or an empty list leaves every source
    // as it was.
    std::array<indexes_type_*, 1> const targets {&target};
    std::array<indexes_type_*, 2> const refused {sources[0], nullptr};
    expect_equal(
        plan_reshard(std::span<indexes_type_* const> {refused}, std::span<indexes_type_* const> {targets}, executor)
            .status,
        status_code_t::invalid_argument_k);
    expect_equal(
        plan_reshard(std::span<indexes_type_* const> {sources}, std::span<indexes_type_* const> {}, executor).status,
        status_code_t::invalid_argument_k);
    expect_true(sources[0]->size() == quarter && target.size() == 0);

    {
        auto [plan, plan_status] = plan_reshard(std::span<indexes_type_* const> {sources},
                                                std::span<indexes_type_* const> {targets}, executor);
        expect_equal(plan_status, status_code_t::success_k);
        expect_equal(drive_run(plan, driver, cancel), status_code_t::cancelled_k);
        // Every source moves in before the first round opens, so a cancel at its first barrier
        // finds them all consumed.
        for (std::size_t which = 0; which != sources.size(); ++which)
            expect_true(plan.moved(sources[which]) && sources[which]->size() == 0);
        expect_true(reported.load() > 0 && reported.load() < baseline_visited);

        // The plan holds the index closed between its calls, never the executor.
        key_t const late_key = dense_key<key_t>(rows + 1);
        expect_equal(
            target.insert(std::span<key_t const> {&late_key, 1}, vectors(nk::range(0, 1), nk::slice), executor).status,
            status_code_t::contended_k);
        if constexpr (!a_ram_collection<indexes_type_>) {
            save_config_t const config = dense_indexes_save_config<indexes_type_>();
            auto const [bytes, measure_status] = target.storage_requirements(config);
            expect_true(succeeded(measure_status));
            std::vector<std::byte> buffer(bytes);
            expect_equal(target.save(bytes_span_t {buffer}, config, executor), status_code_t::contended_k);
        }

        expect_equal(drive_run(plan, driver, proceed), status_code_t::success_k);
        auto const [gathered, finish_status] = drive_finish(plan, driver);
        expect_equal(finish_status, status_code_t::success_k);
        expect_true(gathered.graph.reselected_nodes >= rows);
        expect_true(gathered.graph.verdict != refine_verdict_t::stopped_k);
        expect_equal(gathered.graph.traversal_stats.visited_members, 0u);
        // A finished plan is bound to nothing, so it neither runs nor finishes again.
        expect_equal(drive_run(plan, driver, proceed), status_code_t::uninitialized_k);
        expect_equal(drive_finish(plan, driver).status, status_code_t::uninitialized_k);
    }
    expect_equal(target.size(), rows);
    // Source shards differ by an entry at most, and each moves to the target shard with the
    // fewest.
    std::size_t fewest = rows, most = 0;
    for (std::size_t shard = 0; shard != target.shards_count(); ++shard)
        fewest = min_of(fewest, target.shard(shard).size()), most = max_of(most, target.shard(shard).size());
    expect_true(most - fewest <= sources.size());
    dense_indexes_expect_restored(target, vectors, keys, dimensions, executor, seed);
    auto const [health, health_status] = target.health(executor);
    expect_true(succeeded(health_status));
    expect_true(health.reachable_nodes * 100 >= rows * dense_minimal_recall_percent_k);
}

/**
 *  @brief A sharded index is cut into two and into three graph-local parts, each gathered back,
 *      and recall at ten against the exact oracle stays within noise of the source's throughout.
 *
 *  Shard @c s moves into every target's shard @c s, every key resolves in exactly one target
 *  with its vector intact, the source empties and takes entries again, and every target reaches
 *  its nodes. While a cut's plan lives, an insertion into a target refuses. Refusals leave
 *  every index whole: a repeated or non-empty target, the source among them, another metric or
 *  another shard count; a single target is a reshard into one target, never a cut.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_cut(u64_t seed, placement_types_&... placement) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    using oracle_kit_t =
        typename quads_ladder<scalar_t, dense_reference_capability<typename indexes_type_::index_t>()>::kit_t;
    constexpr std::size_t wanted_k = 10;
    constexpr nk_kernel_kind_t other_kernel_k = kernel_ == nk_kernel_euclidean_k ? nk_kernel_angular_k
                                                                                 : nk_kernel_euclidean_k;
    std::size_t const rows = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_cut"));
    nk::matrix<scalar_t> const queries = make_vectors<scalar_t>(recall_samples_k, dimensions,
                                                                stream_key(seed, "indexes_cut", 1));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    nk::matrix<distance_t> oracle = make_matrix<distance_t>(recall_samples_k, rows);
    for (std::size_t query = 0; query != recall_samples_k; ++query) {
        auto const query_vector = queries.view().row(query).as_vector();
        auto const query_norms = norms<oracle_kit_t>(query_vector);
        for (std::size_t row = 0; row != rows; ++row) {
            auto const row_vector = vectors.view().row(row).as_vector();
            oracle(query, row) = distance<oracle_kit_t>(kernel_, query_vector, query_norms, row_vector,
                                                        norms<oracle_kit_t>(row_vector));
        }
    }
    // Of the ten nearest entries `indexes` holds by the oracle, how many its walks return; a match
    // within the tenth's distance counts, so ties never punish a walk.
    auto recall_at_ten = [&](indexes_type_& indexes) {
        dense_indexes_row_driver<indexes_type_> walker {indexes, driver};
        expect_true(succeeded(walker.status));
        std::vector<std::size_t> held;
        for (std::size_t row = 0; row != rows; ++row)
            if (indexes.count(keys[row])) held.push_back(row);
        nk::vector<key_t> found_keys = make_vector<key_t>(wanted_k);
        nk::vector<distance_t> found_distances = make_vector<distance_t>(wanted_k);
        nk::vector<distance_t> held_distances = make_vector<distance_t>(held.size());
        std::span<distance_t> const exact = to_writable_span(held_distances);
        std::size_t found = 0;
        for (std::size_t query = 0; query != recall_samples_k && held.size() >= wanted_k; ++query) {
            for (std::size_t held_row = 0; held_row != held.size(); ++held_row)
                exact[held_row] = oracle(query, held[held_row]);
            std::sort(exact.begin(), exact.end());
            f64_t const tenth = exact[wanted_k - 1];
            auto const [result, status] = indexes.search(queries.view().row(query).as_vector(),
                                                         to_writable_span(found_keys),
                                                         to_writable_span(found_distances), {}, walker.handle());
            expect_true(succeeded(status));
            for (std::size_t rank = 0; rank != result.count; ++rank)
                found += found_distances[rank] <= tenth + 1e-3 * max_of(std::fabs(tenth), 1.0);
        }
        return found;
    };
    auto recall_close = [&](std::size_t found, std::size_t baseline) {
        return expect_true(found + recall_margin_k * recall_samples_k * wanted_k >= baseline);
    };
    auto make_targets = [&](std::vector<std::unique_ptr<indexes_type_>>& owned, std::size_t count) {
        std::vector<indexes_type_*> targets;
        for (std::size_t target = 0; target != count; ++target) {
            owned.push_back(std::make_unique<indexes_type_>(
                kernel_, typename indexes_type_::geometry_t {dimensions, connectivity_16_k},
                forkunion_allocator_t {no_memory_domain_k}));
            expect_true(succeeded(place_shards(*owned.back(), placement...)));
            targets.push_back(owned.back().get());
        }
        return targets;
    };
    // Every key in exactly one target, sampled vectors intact, and every target reaching its
    // nodes and walking within noise of the source.
    nk::matrix<scalar_t> fetched = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "indexes_cut", 2));
    std::size_t const row_bytes = dense_rows_bytes(vectors.view(), 1);
    auto expect_parts = [&](std::vector<indexes_type_*> const& targets, std::size_t baseline) {
        std::size_t placed = 0, misplaced = 0, mismatched = 0;
        for (indexes_type_* target : targets) placed += target->size();
        expect_equal(placed, rows);
        for (std::size_t row = 0; row != rows; ++row) {
            std::size_t holders = 0;
            for (indexes_type_* target : targets) {
                std::size_t const held = target->count(keys[row]);
                holders += held;
                if (held && row % 16 == 0)
                    mismatched += target->get(keys[row], fetched.span(), 1, executor) != 1 ||
                                  std::memcmp(fetched.data(), vectors.view().row(row).as_vector().byte_data(),
                                              row_bytes) != 0;
            }
            misplaced += holders != 1;
        }
        expect_equal(misplaced, 0u);
        expect_equal(mismatched, 0u);
        for (indexes_type_* target : targets) {
            auto const [health, health_status] = target->health(executor);
            expect_true(succeeded(health_status));
            if constexpr (kernel_ != nk_kernel_dot_k) expect_equal(health.reachable_nodes, target->size());
            else expect_true(health.reachable_nodes * 100 >= target->size() * dense_minimal_recall_percent_k);
            recall_close(recall_at_ten(*target), baseline);
        }
    };

    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(source, placement...)));
    expect_true(succeeded(source.reserve({rows}).status));
    expect_true(succeeded(source.insert(to_span(keys), vectors.view(), executor).status));
    std::size_t const source_recall = recall_at_ten(source);
    // A reshard out of one source into several targets, and one into a single target, the target
    // leading its own sources.
    auto cut_plan = [&](indexes_type_& from, std::vector<indexes_type_*> const& into, u64_t cut_seed = 0) {
        std::array<indexes_type_*, 1> const sources {&from};
        reshard_config_t config;
        config.seed = cut_seed;
        return plan_reshard(std::span<indexes_type_* const> {sources}, std::span<indexes_type_* const> {into}, executor,
                            config);
    };
    auto gather = [&](indexes_type_& into, std::vector<indexes_type_*> sources) {
        sources.insert(sources.begin(), &into);
        std::array<indexes_type_*, 1> const targets {&into};
        return reshard(std::span<indexes_type_* const> {sources}, std::span<indexes_type_* const> {targets}, executor);
    };

    // Refusals precede every hold: the source and every target stay as they were, and open.
    {
        std::vector<std::unique_ptr<indexes_type_>> owned;
        std::vector<indexes_type_*> targets = make_targets(owned, 2);
        indexes_type_ foreign {other_kernel_k,
                               {dimensions, connectivity_16_k},
                               forkunion_allocator_t {no_memory_domain_k}};
        indexes_type_ wider {kernel_, {2 * dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(foreign, placement...)));
        expect_true(succeeded(place_shards(wider, placement...)) && succeeded(place_shards(wider, placement...)));
        for (std::vector<indexes_type_*> refused :
             {std::vector<indexes_type_*> {targets[0], &source}, std::vector<indexes_type_*> {targets[0], &foreign},
              std::vector<indexes_type_*> {targets[0], &wider}})
            expect_equal(cut_plan(source, refused).status, status_code_t::config_mismatch_k);
        for (std::vector<indexes_type_*> malformed :
             {std::vector<indexes_type_*> {}, std::vector<indexes_type_*> {targets[0], nullptr},
              std::vector<indexes_type_*> {targets[0], targets[0]}})
            expect_equal(cut_plan(source, malformed).status, status_code_t::invalid_argument_k);
        // A second source of another kernel refuses as any foreign handle does, before the gather.
        std::array<indexes_type_*, 2> const several {&source, &foreign};
        expect_equal(
            plan_reshard(std::span<indexes_type_* const> {several}, std::span<indexes_type_* const> {targets}, executor)
                .status,
            status_code_t::config_mismatch_k);
        expect_true(succeeded(targets[1]->reserve({16}).status));
        expect_true(succeeded(
            targets[1]
                ->insert(std::span<key_t const> {keys.data(), 1}, vectors(nk::range(0, 1), nk::slice), executor)
                .status));
        expect_equal(cut_plan(source, targets).status, status_code_t::config_mismatch_k);
        expect_true(source.size() == rows && targets[0]->size() == 0 && targets[1]->size() == 1);
    }

    // Two parts, the targets closed while the plan lives, then gathered back.
    std::vector<std::unique_ptr<indexes_type_>> halves_owned;
    std::vector<indexes_type_*> halves = make_targets(halves_owned, 2);
    {
        auto [plan, plan_status] = cut_plan(source, halves, stream_key(seed, "indexes_cut", 3));
        expect_equal(plan_status, status_code_t::success_k);
        expect_equal(
            halves[0]
                ->insert(std::span<key_t const> {keys.data(), 1}, vectors(nk::range(0, 1), nk::slice), executor)
                .status,
            status_code_t::contended_k);
        expect_equal(drive_run(plan, driver), status_code_t::success_k);
        auto const [cut, finish_status] = drive_finish(plan, driver);
        expect_equal(finish_status, status_code_t::success_k);
        expect_equal(cut.graph.reachability.audited_nodes, rows);
        // Every target shard runs a round after the moves, and no phase walks.
        expect_true(cut.graph.neighborhoods != 0 && cut.graph.verdict != refine_verdict_t::stopped_k);
        expect_equal(cut.graph.traversal_stats.visited_members, 0u);
        expect_equal(drive_run(plan, driver), status_code_t::uninitialized_k);
        expect_equal(drive_finish(plan, driver).status, status_code_t::uninitialized_k);
    }
    expect_equal(source.size(), 0u);
    expect_parts(halves, source_recall);
    expect_true(succeeded(gather(*halves[0], {halves[1]}).status));
    expect_equal(halves[1]->size(), 0u);
    dense_indexes_expect_restored(*halves[0], vectors, keys, dimensions, executor, seed);
    recall_close(recall_at_ten(*halves[0]), source_recall);

    // The reunion is cut into three through the one-shot, and gathered back once more.
    std::vector<std::unique_ptr<indexes_type_>> thirds_owned;
    std::vector<indexes_type_*> thirds = make_targets(thirds_owned, 3);
    {
        auto [plan, plan_status] = cut_plan(*halves[0], thirds, stream_key(seed, "indexes_cut", 4));
        expect_equal(plan_status, status_code_t::success_k);
        expect_equal(drive_run(plan, driver), status_code_t::success_k);
        expect_true(succeeded(drive_finish(plan, driver).status));
    }
    expect_equal(halves[0]->size(), 0u);
    expect_parts(thirds, source_recall);
    expect_true(succeeded(gather(*thirds[0], {thirds[1], thirds[2]}).status));
    dense_indexes_expect_restored(*thirds[0], vectors, keys, dimensions, executor, seed);
    recall_close(recall_at_ten(*thirds[0]), source_recall);

    // The emptied source takes entries again.
    expect_true(succeeded(source.reserve({rows}).status));
    expect_true(succeeded(source.insert(to_span(keys), vectors.view(), executor).status));
    expect_equal(source.size(), rows);
}

/** The sharded checkpoint's whole life: batched rows in, a lone shard's monolith or a manifest plus
 *  one monolith per shard out, summaries read from it, a fresh index restored into the same
 *  placement one shard per part, and byte-identical vectors fetched back by key - with a cancelled
 *  save leaving no manifest and no header, so the failed checkpoint reads as absent, an illegal
 *  alignment refused, and the provenance riding in every part's header. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_save_load(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 1));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(indexes, executor, vectors, keys, placement...);
    std::size_t const shards = indexes.shards_count();
    dense_checkpoint_path_t const checkpoint;
    save_config_t config = dense_indexes_save_config<indexes_type_>();

    // A cancelled save writes parts but no manifest or header, so the checkpoint reads as absent.
    expect_equal(
        indexes.save(checkpoint.path, config, executor, [](std::size_t) noexcept { return continuation_t::cancel_k; }),
        status_code_t::cancelled_k);
    expect_true(failed(indexes_type_::metadata(checkpoint.path).status));
    save_config_t misaligned = config;
    misaligned.alignment = 32;
    expect_equal(indexes.save(checkpoint.path, misaligned, executor), status_code_t::config_mismatch_k);
    if constexpr (dense_device_resident<index_t>) {
        save_config_t probing = config;
        probing.synchronization = synchronization_t::concurrent_k;
        expect_equal(indexes.save(checkpoint.path, probing, executor), status_code_t::config_mismatch_k);
    }

    std::array<safegraphs_custom_field_t, 1> const provenance {{{"embedding_model", "test"}}};
    config.provenance = provenance;
    expect_true(succeeded(indexes.save(checkpoint.path, config, executor)));
    auto const [required_bytes, requirements_status] = indexes.storage_requirements(config);
    expect_true(succeeded(requirements_status) && required_bytes > 0);
    if (shards == 1) {
        auto [mapping, status] = mapped_file_t::read(checkpoint.path);
        expect_true(succeeded(status));
        expect_true(succeeded(requirements_status) && required_bytes == mapping.size());
    }

    // The summary answers from the manifest or the lone monolith, metric included.
    auto [summary, summary_status] = indexes_type_::metadata(checkpoint.path);
    expect_true(succeeded(summary_status));
    expect_equal(summary.nodes, rows);
    expect_equal(summary.parts, shards);
    expect_equal(regular_width(summary.value_shape).value_or(0), dimensions);
    expect_true(nk_kernel_named(summary.metric.characters, summary.metric.view().size()) == kernel_);
    for (std::size_t part = 0; part != shards; ++part) {
        std::size_t provenance_seen = 0;
        auto const [header, header_status] = safegraphs_read_header(
            checkpoint.part(part, shards), {}, [&](safegraphs_custom_field_t field) {
                provenance_seen += field.key == "embedding_model" && field.value == "test";
            });
        expect_true(succeeded(header_status));
        expect_equal(provenance_seen, 1u);
        expect_equal(header.nodes(), indexes.shard(part).size());
    }

    // A fresh index in the same placement restores it whole, one shard per part, keys and all.
    indexes_type_ restored {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(restored, placement...)));
    expect_true(succeeded(restored.load(checkpoint.path, load_config_t {}, executor)));
    dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);
    expect_equal(restored.shards_count(), shards);
    for (std::size_t shard = 0; shard != min_of(shards, restored.shards_count()); ++shard)
        expect_equal(restored.shard(shard).size(), indexes.shard(shard).size());

    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {restored, driver};
        expect_true(succeeded(rows_driver.status));
        nk::vector<key_t> found_keys = make_vector<key_t>(8);
        nk::vector<typename indexes_type_::distance_t> found_distances =
            make_vector<typename indexes_type_::distance_t>(8);
        auto const [search_result, search_status] = restored.search(
            vectors.view().row(0).as_vector(), to_writable_span(found_keys), to_writable_span(found_distances), {},
            rows_driver.handle());
        expect_true(search_status == status_code_t::success_k && search_result.count > 0);
    }
}

/** A load places each part on a shard of its own: a reservation smaller than the parts still fits
 *  them, since a shard rounds up to whole chunks, and a populated index refuses a second load. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_part_load(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 2 * dense_thread_nodes_k + 3;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 3));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(indexes, executor, vectors, keys, placement...);
    std::size_t const shards = indexes.shards_count();
    dense_checkpoint_path_t const checkpoint;
    expect_true(succeeded(indexes.save(checkpoint.path, dense_indexes_save_config<indexes_type_>(), executor)));

    indexes_type_ cramped {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(cramped, placement...)));
    expect_true(succeeded(cramped.reserve({shards}).status));
    expect_true(succeeded(cramped.load(checkpoint.path, load_config_t {}, executor)));
    dense_indexes_expect_restored(cramped, vectors, keys, dimensions, executor, seed);
    for (std::size_t shard = 0; shard != min_of(shards, cramped.shards_count()); ++shard)
        expect_equal(cramped.shard(shard).size(), indexes.shard(shard).size());
    expect_equal(cramped.load(checkpoint.path, load_config_t {}, executor), status_code_t::config_mismatch_k);
    expect_equal(cramped.size(), rows);
}

/** Any number of threads may take tasks from one sharded plan: a shared cursor hands them
 *  out regardless of which shard they read, the plan finishes once, and the buffer they
 *  wrote restores into the same graph the one-shot save writes. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_pooled_save(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 5));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(indexes, executor, vectors, keys);
    save_config_t const config = dense_indexes_save_config<indexes_type_>();
    auto const [bytes, measure_status] = indexes.storage_requirements(config);
    expect_true(succeeded(measure_status));

    std::vector<std::byte> buffer(bytes);
    auto [plan, plan_status] = indexes.plan_save(bytes_span_t {buffer}, config);
    expect_true(succeeded(plan_status));
    auto const tasks = plan.tasks();
    expect_true(tasks.size() >= indexes.shards_count());

    std::size_t cursor_position = 0, lane_failures_count = 0;
    fu::standard_atomic_ref<std::size_t> const cursor {cursor_position};
    fu::standard_atomic_ref<std::size_t> const lane_failures {lane_failures_count};
    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, 3, fu::caller_inclusive_k)))
        return void(expect_true(false));
    expect_true(!fu::failed(pool.for_threads([&](std::size_t) noexcept {
        for (std::size_t position = cursor.fetch_add(1); position < tasks.size(); position = cursor.fetch_add(1))
            if (failed(plan.save(tasks[position]))) lane_failures.store_add(1);
    })));
    expect_equal(lane_failures_count, 0u);
    expect_equal(plan.finish(), status_code_t::success_k);

    indexes_type_ restored {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(restored)));
    expect_true(succeeded(restored.load(bytes_view_t {buffer}, load_config_t {}, executor)));
    dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);
}

/** The first part of a saved @p buffer: the buffer itself for a lone monolith, else the first part
 *  the manifest leading it lists. */
inline bytes_view_t dense_first_part(bytes_view_t buffer) noexcept {
    if (safegraphs_is_monolith(buffer)) return buffer;
    bytes_view_t first;
    auto const per_part = [&](safegraphs_part_t part) {
        if (first.empty()) first = buffer.subspan(part.offset, part.bytes);
    };
    if (failed(safegraphs_read_manifest(buffer, std::allocator<byte_t> {}, per_part).status)) return {};
    return first;
}

/** Distinct rows of which every 32nd repeats the first, built over @p placement: no repeat takes
 *  links from more than two others, so the class forms a chain a walk passes along rather than a
 *  star around one member. Only warns until duplicate classes are handled. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void dense_expect_duplicates_build(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using node_id_t = typename index_t::node_id_t;
    // A dot scores identical rows by their norm and an angular through approximate roots, so
    // neither ties them.
    if constexpr (kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_angular_k) return;
    constexpr std::size_t rows = 1024, stride = 32;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "tied_build"));
    // Read through a view: a span's packed dimension is a proxy, and proxy-to-proxy assignment
    // copies no bit.
    auto const source = vectors.view().row(0).as_vector();
    for (std::size_t row = stride; row < rows; row += stride) {
        auto target = vectors.row(row).as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension) target[dimension] = source[dimension];
    }
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    dense_indexes_build(indexes, driver.executor, vectors, keys, placement...);

    std::vector<std::byte> buffer;
    auto save = [&]<typename saved_type_>(saved_type_& saved, typename saved_type_::executor_t& executor) {
        save_config_t const config = dense_indexes_save_config<saved_type_>();
        auto const [required_bytes, requirements_status] = saved.storage_requirements(config);
        if (failed(requirements_status)) return requirements_status;
        buffer.resize(required_bytes);
        return saved.save(bytes_span_t {buffer}, config, executor);
    };
    // An NVMe view serves the graph its RAM group built.
    bool saved = false;
    if constexpr (requires { indexes.built; })
        saved = expect_true(succeeded(save(indexes.built, indexes.builder.executor)));
    else saved = expect_true(succeeded(save(indexes, driver.executor)));
    if (!saved) return;

    bytes_view_t const whole {buffer};
    std::vector<bytes_view_t> parts;
    if (safegraphs_is_monolith(whole)) parts.push_back(whole);
    else
        expect_true(succeeded(safegraphs_read_manifest(whole, std::allocator<byte_t> {}, [&](safegraphs_part_t part) {
                                  parts.push_back(whole.subspan(part.offset, part.bytes));
                              }).status));
    std::size_t widest = 0;
    for (bytes_view_t const part : parts) {
        auto const [header, header_status] = safegraphs_read_header(part, std::allocator<byte_t> {});
        std::size_t const nodes = header.nodes();
        std::size_t const width = regular_width(header.neighbors.shape).value_or(0);
        bytes_view_t const stored_keys = header.data_of(part, header.key);
        bytes_view_t const links = header.data_of(part, header.neighbors);
        if (!expect_true(succeeded(header_status) && stored_keys.size() == nodes * sizeof(key_t) &&
                         links.size() == nodes * width * sizeof(node_id_t)))
            return;
        std::vector<bool> tied(nodes);
        for (std::size_t node = 0; node != nodes; ++node) {
            key_t key;
            std::memcpy(&key, stored_keys.data() + node * sizeof(key_t), sizeof(key_t));
            tied[node] = (dense_key_number(key) - 1) % stride == 0;
        }
        std::vector<std::size_t> in_degree(nodes);
        for (std::size_t node = 0; node != nodes; ++node)
            for (std::size_t slot = 0; tied[node] && slot != width; ++slot) {
                node_id_t link;
                std::memcpy(&link, links.data() + (node * width + slot) * sizeof(node_id_t), sizeof(node_id_t));
                // Out-of-range IDs are the column's empty slots.
                if (static_cast<std::size_t>(link) < nodes && tied[link]) ++in_degree[link];
            }
        for (std::size_t const degree : in_degree) widest = max_of(widest, degree);
    }
    if (widest > 2) fmt::println(stderr, "  warning: {} of {} repeats link to one of them", widest, rows / stride);
}

/** The norms column of @p monolith, one part of @p rows rows a sharded index of @p geometry saved:
 *  present where the shards keep norms, taken as written by a restore, and derived again for a part
 *  without the column or with a dtype the shards never keep. Every restore is placed over
 *  @p placement, places the part on its first shard and saves again, and the norms that save writes
 *  are the ones it holds. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename executor_type_, typename... placement_types_>
void dense_indexes_expect_norms(std::vector<std::byte> const& monolith, executor_type_& executor, std::size_t rows,
                                typename indexes_type_::geometry_t geometry, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    using norms_t = vector_norms<dot_result_t>;
    constexpr bool keeps_norms_k = kernel_ != nk_kernel_dot_k || dense_dot_keeps_norms<index_t>();
    dtype_t const norms_dtype = dtype_of(dot_result_t::dtype());
    growing_buffer<safegraphs_named_column_t, std::allocator<byte_t>> extra_columns {std::allocator<byte_t> {}};
    auto const [header, header_status] = safegraphs_read_header(bytes_view_t {monolith}, std::allocator<byte_t> {},
                                                                dummy_callback_t {}, &extra_columns);
    expect_true(succeeded(header_status));
    safegraphs_column_t const* const written = header.extra_column(norms_column_name_k);
    if (!keeps_norms_k) return void(expect_true(!written));
    if (!expect_true(written && written->dtype == norms_dtype && written->shape == norms_column_shape_k &&
                     written->nodes() == rows))
        return;
    safegraphs_column_t const norms_column = *written;

    save_config_t const config = dense_indexes_save_config<indexes_type_>();
    auto resaved_norms = [&](std::vector<std::byte> const& source) {
        nk::vector<norms_t> norms = make_vector<norms_t>(rows);
        indexes_type_ restored {kernel_, geometry, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(restored, placement...)));
        expect_true(succeeded(restored.load(bytes_view_t {source}, load_config_t {}, executor)));
        auto const [resaved_bytes, requirements_status] = restored.storage_requirements(config);
        if (!expect_true(succeeded(requirements_status))) return norms;
        std::vector<std::byte> resaved(resaved_bytes);
        expect_true(succeeded(restored.save(bytes_span_t {resaved}, config, executor)));
        bytes_view_t const part = dense_first_part(bytes_view_t {resaved});
        growing_buffer<safegraphs_named_column_t, std::allocator<byte_t>> resaved_columns {std::allocator<byte_t> {}};
        auto const [resaved_header, resaved_status] = safegraphs_read_header(part, std::allocator<byte_t> {},
                                                                             dummy_callback_t {}, &resaved_columns);
        bytes_view_t const column = norms_column_of(resaved_header, part, norms_dtype);
        expect_true(succeeded(resaved_status) && column.size() == rows * sizeof(norms_t));
        if (column.size() == rows * sizeof(norms_t)) std::memcpy(norms.data(), column.data(), column.size());
        return norms;
    };
    // Every copy rotates the stored rows by one, so a restore that took them would show it; an
    // unknown dtype leaves the column out of the copy's header.
    auto doctored = [&](safegraphs_column_t norms) {
        std::vector<std::byte> copy = monolith;
        bytes_span_t const column = header.data_of(bytes_span_t {copy}, norms_column);
        std::rotate(column.begin(), column.begin() + sizeof(norms_t), column.end());
        safegraphs_named_column_t const named {safegraphs_name_t {norms_column_name_k}, norms};
        safegraphs_header_t patched = header;
        patched.extra_columns = norms.dtype == dtype_t::unknown_k
                                    ? std::span<safegraphs_named_column_t const> {}
                                    : std::span<safegraphs_named_column_t const> {&named, 1};
        std::array<safegraphs_custom_field_t, 1> const fields {{{"metric", nk_kernel_name(kernel_)}}};
        expect_true(succeeded(safegraphs_write_header(bytes_span_t {copy}, patched, fields)));
        return copy;
    };

    nk::vector<norms_t> original = make_vector<norms_t>(rows);
    std::memcpy(original.data(), header.data_of(bytes_view_t {monolith}, norms_column).data(), rows * sizeof(norms_t));
    std::vector<std::byte> const rotated = doctored(norms_column);
    nk::vector<norms_t> const taken = resaved_norms(rotated);
    expect_true(std::memcmp(taken.data(), header.data_of(bytes_view_t {rotated}, norms_column).data(),
                            rows * sizeof(norms_t)) == 0);

    safegraphs_column_t mistyped = norms_column;
    mistyped.dtype = sizeof(dot_result_t) == 8 ? dtype_t::b64_k : dtype_t::b32_k;
    nk::vector<norms_t> const without_column = resaved_norms(doctored({}));
    nk::vector<norms_t> const under_mistype = resaved_norms(doctored(mistyped));

    error_stats_t norm_errors;
    for (std::size_t sample = 0; sample != 8; ++sample) {
        std::size_t const row = sample * (rows / 8);
        f64_t const sum = static_cast<f64_t>(original[row].sum), sumsq = static_cast<f64_t>(original[row].sumsq);
        norm_errors.add(static_cast<f64_t>(without_column[row].sum), sum);
        norm_errors.add(static_cast<f64_t>(without_column[row].sumsq), sumsq);
        norm_errors.add(static_cast<f64_t>(under_mistype[row].sum), sum);
        norm_errors.add(static_cast<f64_t>(under_mistype[row].sumsq), sumsq);
    }
    expect_errors("derived norms", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, norm_errors, 0);
}

/** Buffers hold a lone shard's monolith or a manifest leading one part per shard:
 *  @c storage_requirements sizes them, the summary reads from them, a fresh index restores from
 *  them, each part's norms as written or derived again, and a short buffer or a cancelled save
 *  leave nothing a reader accepts. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_buffer_round_trip(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 4));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(indexes, executor, vectors, keys, placement...);
    save_config_t const config = dense_indexes_save_config<indexes_type_>();
    auto const [bytes, measure_status] = indexes.storage_requirements(config);
    expect_true(succeeded(measure_status));
    expect_true(bytes > 0);

    std::vector<std::byte> buffer(bytes);
    expect_true(succeeded(indexes.save(bytes_span_t {buffer}, config, executor)));
    auto [summary, metadata_status] = indexes_type_::metadata(bytes_view_t {buffer});
    expect_true(succeeded(metadata_status));
    expect_equal(summary.nodes, rows);
    expect_equal(summary.parts, indexes.shards_count());
    expect_equal(regular_width(summary.value_shape).value_or(0), dimensions);

    indexes_type_ restored {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(restored, placement...)));
    expect_true(succeeded(restored.load(bytes_view_t {buffer}, load_config_t {}, executor)));
    dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);

    bytes_view_t const first_part = dense_first_part(bytes_view_t {buffer});
    dense_indexes_expect_norms<indexes_type_, kernel_>(std::vector<std::byte>(first_part.begin(), first_part.end()),
                                                       executor, indexes.shard(0).size(), indexes.geometry(),
                                                       placement...);

    // A buffer one byte short and a cancelled save both leave nothing a reader accepts.
    std::vector<std::byte> shorter(bytes - 1);
    expect_equal(indexes.save(bytes_span_t {shorter}, config, executor), status_code_t::overflow_k);
    std::vector<std::byte> cancelled(bytes);
    expect_equal(indexes.save(bytes_span_t {cancelled}, config, executor,
                              [](std::size_t) noexcept { return continuation_t::cancel_k; }),
                 status_code_t::cancelled_k);
    expect_true(failed(indexes_type_::metadata(bytes_view_t {cancelled}).status));
}

/** A cancelled load publishes nothing: cancelling at every shard boundary answers
 *  @c cancelled_k, leaves the index empty, and the next plain load restores it whole. */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_cancelled_load(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement...};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "roundtrip", 5));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(indexes, executor, vectors, keys, placement...);
    std::size_t const shards = indexes.shards_count();
    dense_checkpoint_path_t const checkpoint;
    expect_true(succeeded(indexes.save(checkpoint.path, dense_indexes_save_config<indexes_type_>(), executor)));

    for (std::size_t boundary = 0; boundary != shards; ++boundary) {
        indexes_type_ restored {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(restored, placement...)));
        auto cancel_at = [boundary](std::size_t shard) noexcept {
            return shard == boundary ? continuation_t::cancel_k : continuation_t::continue_k;
        };
        expect_equal(restored.load(checkpoint.path, load_config_t {}, executor, cancel_at), status_code_t::cancelled_k);
        expect_equal(restored.size(), 0u);
        expect_equal(restored.count(keys[0]), 0u);
        expect_true(succeeded(restored.load(checkpoint.path, load_config_t {}, executor)));
        dense_indexes_expect_restored(restored, vectors, keys, dimensions, executor, seed);
    }
}

/**
 *  @brief Two executors drive one sharded index concurrently: disjoint key ranges, one
 *      batched @c insert per executor, both interleaving into the same shards - asserted
 *      through the public surface only.
 *
 *  A RAM index admits both batches at once. A VRAM index admits one mutation at a time and
 *  answers the overlapping one with @c contended_k, so a refused batch is retried once the
 *  threads are awaited; either way every entry is inserted exactly once.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_, typename... placement_types_>
void test_dense_indexes_concurrent_add(u64_t seed, placement_types_&... placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;

    std::size_t const executors_count = 2;
    std::size_t const rows_per_executor = 2 * dense_thread_nodes_k;
    std::size_t const nodes_count = executors_count * rows_per_executor;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement...)));
    expect_true(succeeded(indexes.reserve({nodes_count}).status));

    std::array<dense_indexes_executor<indexes_type_>, executors_count> drivers {
        dense_indexes_executor<indexes_type_> {placement...}, dense_indexes_executor<indexes_type_> {placement...}};
    std::vector<nk::matrix<scalar_t>> slices;
    nk::matrix<key_t> keys = make_matrix<key_t>(executors_count, rows_per_executor);
    for (std::size_t which = 0; which != executors_count; ++which) {
        if (failed(drivers[which].status)) return void(expect_true(false));
        slices.push_back(make_vectors<scalar_t>(rows_per_executor, dimensions, stream_key(seed, "indexes_add", which)));
        // Keys never take the value zero, so a stale read of a zero-filled key column can
        // never masquerade as a legitimate identity.
        for (std::size_t row = 0; row != rows_per_executor; ++row)
            keys(which, row) = dense_key<key_t>(1 + which * rows_per_executor + row);
    }

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, executors_count, fu::caller_inclusive_k)))
        return void(expect_true(false));
    std::array<status_t, executors_count> outcomes {};
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= executors_count) return;
        outcomes[thread] =
            indexes.insert(to_span(keys.row(thread).as_vector()), slices[thread].view(), drivers[thread].executor)
                .status;
        assert_no_locks_held();
    })));
    for (std::size_t which = 0; which != executors_count; ++which) {
        if (outcomes[which] == status_code_t::contended_k && !a_ram_collection<indexes_type_>)
            outcomes[which] = indexes
                                  .insert(to_span(keys.row(which).as_vector()), slices[which].view(),
                                          drivers[which].executor)

                                  .status;
        expect_equal(outcomes[which], status_code_t::success_k);
    }
    expect_equal(indexes.size(), nodes_count);

    // Rotation splits every batch evenly, so the shards end exactly equally full.
    std::size_t rows_in_shards = 0;
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) rows_in_shards += indexes.shard(shard).size();
    expect_equal(rows_in_shards, nodes_count);
    expect_equal(indexes.shard(0).size(), nodes_count / indexes.shards_count());

    // Every key is stored exactly once, and resolvable without naming a shard or an ID.
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != nodes_count; ++row) miscounted += indexes.count(dense_key<key_t>(1 + row)) != 1;
    expect_equal(miscounted, 0u);

    // Sampled self-recall through the merged cross-shard search, judged by distance band.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        dense_indexes_row_driver<indexes_type_> rows_driver {indexes, drivers[0]};
        expect_true(succeeded(rows_driver.status));
        std::size_t hits = 0;
        std::size_t const samples_count = min_of(recall_samples_k, rows_per_executor);
        nk::vector<key_t> found_keys = make_vector<key_t>(8);
        nk::vector<typename indexes_type_::distance_t> found_distances =
            make_vector<typename indexes_type_::distance_t>(8);
        for (std::size_t sample = 0; sample != samples_count; ++sample) {
            std::size_t const row = sample * (rows_per_executor / samples_count);
            auto const [result, status] = indexes.search(slices[0].view().row(row).as_vector(),
                                                         to_writable_span(found_keys),
                                                         to_writable_span(found_distances), {}, rows_driver.handle());
            expect_true(status == status_code_t::success_k && result.count > 0);
            for (std::size_t match = 0; match != result.count; ++match)
                if (found_distances[match] < dense_self_match_tolerance_k &&
                    found_distances[match] > -dense_self_match_tolerance_k) {
                    ++hits;
                    break;
                }
        }
        expect_true(hits + recall_margin_k * samples_count >= samples_count * dense_minimal_recall_percent_k / 100.0);
    }
}

/** A reader thread races executor-driven batched @c insert calls into the shards, asserting
 *  the key-visibility invariant: a search may reach a node the moment it publishes, so
 *  the key it resolves must already be written - never the key column's zero-fill, never
 *  outside the assigned range. Every batch spans both shards yet shows whole or not at all.
 *  RAM only: a VRAM index admits no search during a mutation. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_search_race(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({nodes_count}).status));

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_race"));
    nk::vector<key_t> keys = make_vector<key_t>(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = dense_key<key_t>(1 + row);

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, 2, fu::caller_inclusive_k)))
        return void(expect_true(false));
    bool writer_done_word = false;
    fu::standard_atomic_ref<bool> const writer_done {writer_done_word};
    std::size_t thread_failures_count = 0;
    fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            for (std::size_t first = 0; first != nodes_count; first += dense_thread_nodes_k) {
                auto [result, status] = indexes.insert(
                    std::span<key_t const> {keys.data() + first, dense_thread_nodes_k},
                    vectors(nk::range(first, first + dense_thread_nodes_k), nk::slice), executor);
                if (status != status_code_t::success_k) thread_failures.store_add(1);
            }
            writer_done.store(true, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        dense_indexes_row_driver<indexes_type_> rows_driver {indexes, driver};
        if (failed(rows_driver.status)) {
            thread_failures.store_add(1);
            writer_done.store(true, std::memory_order_release);
            return;
        }
        nk::vector<key_t> found_keys = make_vector<key_t>(8);
        nk::vector<typename indexes_type_::distance_t> found_distances =
            make_vector<typename indexes_type_::distance_t>(8);
        std::size_t searches = 0;
        for (; !writer_done.load(std::memory_order_acquire) || searches < dense_reader_minimal_searches_k; ++searches) {
            auto const [result, status] = indexes.search(vectors.view().row(searches % nodes_count).as_vector(),
                                                         to_writable_span(found_keys),
                                                         to_writable_span(found_distances), {}, rows_driver.handle());
            if (status != status_code_t::success_k) {
                thread_failures.store_add(1);
                break;
            }
            for (std::size_t match = 0; match != result.count; ++match)
                if (dense_key_number(found_keys[match]) < 1 || dense_key_number(found_keys[match]) > nodes_count)
                    thread_failures.store_add(1);
            if (indexes.size() % dense_thread_nodes_k) thread_failures.store_add(1);
        }
        assert_no_locks_held();
    })));
    expect_equal(thread_failures.load(), 0u);
    expect_equal(indexes.size(), nodes_count);
}

/** The optimistic save mode under fire: a saver thread loops concurrent-mode saves
 *  against two inserter threads & a searcher, classifying every answer as @c contended_k
 *  or @c success_k - and every settled payload must be structurally sane and reload
 *  into a fresh index whose recall holds. Inserters keep trickling spare rows until
 *  the saver has provably contended at least once, and a settle phase after the race
 *  demands a deterministic success. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_save_race(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t const inserter_threads_count = 2, spare_rows_count = 64;
    std::size_t const prefill_rows_count = 256;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t const nodes_ceiling = prefill_rows_count + inserter_threads_count * (dense_thread_nodes_k + //
                                                                                     spare_rows_count);

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) ||
        fu::failed(pool.spawn(topology, inserter_threads_count + 2, fu::caller_inclusive_k)))
        return void(expect_true(false));

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    expect_true(index.reserve({nodes_ceiling}).status == status_code_t::success_k);

    auto threads = dense_make_threads<index_type_>(make_index, inserter_threads_count + 2,
                                                   dense_thread_nodes_k + spare_rows_count, dimensions,
                                                   stream_key(seed, "save_race"));
    nk::matrix<scalar_t> prefill_vectors = make_vectors<scalar_t>(prefill_rows_count, dimensions,
                                                                  stream_key(seed, "save_race", 1));
    typename index_type_::traversal_context_t prefill_context {{}};
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_ceiling);
    expect_true(index.insert({keys.data(), prefill_rows_count}, prefill_vectors.view(), {}, prefill_context).status ==
                status_code_t::success_k);

    // The saver's destinations are sized once at the ceiling; each attempt fills the
    // walked prefix, so nothing allocates inside the race loop.
    nk::matrix<scalar_t> saver_vectors = make_matrix<scalar_t>(nodes_ceiling, dimensions);
    nk::matrix<node_id_t> saver_neighbors = make_matrix<node_id_t>(nodes_ceiling, connectivity_16_k);
    expect_true(!saver_vectors.empty() && !saver_neighbors.empty());

    struct recorded_payload_t {
        nk::matrix<scalar_t> vectors;
        nk::matrix<node_id_t> neighbors;
        std::size_t nodes = 0;
    };
    std::vector<recorded_payload_t> recorded_payloads;
    recorded_payloads.reserve(4);

    std::size_t inserter_threads_remaining_count = inserter_threads_count;
    std::size_t contended_saves_count = 0, settled_saves_count = 0, thread_failures_count = 0;
    fu::standard_atomic_ref<std::size_t> const inserter_threads_remaining {inserter_threads_remaining_count};
    fu::standard_atomic_ref<std::size_t> const contended_saves {contended_saves_count};
    fu::standard_atomic_ref<std::size_t> const settled_saves {settled_saves_count};
    fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};

    auto attempt_save = [&]() noexcept {
        auto [plan, status] = index.plan_save({dense_matrix_bytes(saver_vectors), dense_matrix_bytes(saver_neighbors)},
                                              synchronization_t::concurrent_k);
        if (failed(status)) return std::pair {std::size_t {0}, status};
        std::size_t const nodes = static_cast<std::size_t>(plan.visible);
        for (auto const task : plan.tasks())
            if (status_t const saved = plan.save(task); failed(saved)) return std::pair {nodes, saved};
        return std::pair {nodes, plan.finish()};
    };
    auto audit_settled = [&](std::size_t nodes) noexcept {
        for (std::size_t row = 0; row != nodes; ++row) {
            nk::vector_view<node_id_t> const ids = saver_neighbors.view().row(row).as_vector();

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
        settled_saves.store_add(1);
        if (!audit_settled(nodes)) thread_failures.store_add(1);
        if (recorded_payloads.size() == recorded_payloads.capacity()) return;
        recorded_payload_t recorded {make_matrix<scalar_t>(nodes, dimensions),
                                     make_matrix<node_id_t>(nodes, connectivity_16_k), nodes};
        if (recorded.vectors.empty() || recorded.neighbors.empty()) return;
        std::memcpy(recorded.vectors.data(), saver_vectors.data(), dense_rows_bytes(saver_vectors.view(), nodes));
        std::memcpy(recorded.neighbors.data(), saver_neighbors.data(), dense_rows_bytes(saver_neighbors.view(), nodes));
        recorded_payloads.push_back(std::move(recorded));
    };
    auto saver_thread = [&]() noexcept { // Classify, audit, and record every settled payload.
        auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
        while (inserter_threads_remaining.load(std::memory_order_acquire) != 0) {
            if (std::chrono::steady_clock::now() > deadline) return thread_failures.store_add(1); // A stuck racer.
            auto const [nodes, save_status] = attempt_save();
            if (save_status == status_code_t::contended_k) contended_saves.store_add(1);
            else if (save_status == status_code_t::success_k) record_settled(nodes);
            else thread_failures.store_add(1);
        }
        // The settle phase: with every racer done, a save must soon succeed.
        while (std::chrono::steady_clock::now() <= deadline + wait_limit_k) {
            auto const [nodes, save_status] = attempt_save();
            if (save_status == status_code_t::success_k) return record_settled(nodes);
            if (save_status != status_code_t::contended_k) return thread_failures.store_add(1);
        }
        thread_failures.store_add(1); // The settle phase starved.
    };
    auto inserter_thread = [&](std::size_t thread) noexcept { // A slice, then spare singles
                                                              // until the saver has contended.
        auto slice = threads.slices[thread].view();
        auto main_rows = slice(nk::range(0, dense_thread_nodes_k), nk::slice);
        if (index.insert({keys.data(), dense_thread_nodes_k}, main_rows, {}, threads.contexts[thread]).status !=
            status_code_t::success_k)
            thread_failures.store_add(1);
        for (std::size_t spare = 0; spare != spare_rows_count && //
                                    contended_saves.load(std::memory_order_acquire) == 0;
             ++spare) {
            auto spare_row = slice(nk::range(dense_thread_nodes_k + spare, dense_thread_nodes_k + spare + 1),
                                   nk::slice);
            if (index.insert({keys.data(), 1u}, spare_row, {}, threads.contexts[thread]).status !=
                status_code_t::success_k)
                thread_failures.store_add(1);
            std::this_thread::yield();
        }
        inserter_threads_remaining.store_sub(1, std::memory_order_release);
        assert_no_locks_held();
    };
    auto searcher_thread = [&](std::size_t thread) noexcept { // The reader-thread invariants.
        std::size_t const expansion_bound = static_cast<std::size_t>(connectivity_16_k) * 2;
        auto query_slice = threads.slices[thread].view();
        for (std::size_t task = 0;
             task < dense_reader_minimal_searches_k || inserter_threads_remaining.load(std::memory_order_acquire) != 0;
             ++task) {
            auto query = query_slice.row(task % dense_thread_nodes_k).as_vector();
            auto const [result, status] = index.search(query, {}, threads.contexts[thread]);
            if (status != status_code_t::success_k) thread_failures.store_add(1);
            thread_failures.store_add(scored_violations(threads.contexts[thread].results.data(), result.count,
                                                        index.size(), expansion_bound));
        }
        assert_no_locks_held();
    };
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) return saver_thread();
        if (thread <= inserter_threads_count) return inserter_thread(thread);
        if (thread == inserter_threads_count + 1) return searcher_thread(thread);
    })));

    expect_equal(thread_failures.load(), 0u);
    expect_true(contended_saves.load() + settled_saves.load() > 0);
    expect_true(settled_saves.load() > 0);
    expect_true(recorded_payloads.size() >= 1);

    // Every settled payload reloads into a fresh index whose self-recall holds, the
    // payload's own vector rows serving as the queries.
    for (recorded_payload_t const& recorded : recorded_payloads) {
        std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
        index_type_& restored = *restored_ptr;
        expect_true(restored.reserve({recorded.nodes}).status == status_code_t::success_k);
        expect_equal(
            restored.load({dense_matrix_bytes(recorded.vectors.view()), dense_matrix_bytes(recorded.neighbors.view())},
                          {keys.data(), recorded.nodes}, recorded.nodes, {0, static_cast<node_id_t>(recorded.nodes)}),
            status_code_t::success_k);
        expect_equal(restored.size(), recorded.nodes);
        if constexpr (kernel_ != nk_kernel_dot_k)
            dense_expect_self_recall(restored, make_index, recorded.vectors.view(), recorded.nodes);
    }
}

/** @ref dense_expect_save_race, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_save_race(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_save_race<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Races

#pragma region Readers

/** On a quiesced graph the @c immutable_k search mode reproduces the concurrent mode
 *  bit for bit - fresh identically-seeded contexts, so the landmark draws align - at
 *  the default expansion and at an explicit one. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_immutable_parity(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const nodes_count = 512, stored_queries_count = 16, novel_queries_count = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t build_context {{}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions,
                                                          stream_key(seed, "immutable_parity"));
    expect_true(index.reserve({nodes_count}).status == status_code_t::success_k);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_true(index.insert(to_span(keys), vectors.view(), {}, build_context).status == status_code_t::success_k);
    nk::matrix<scalar_t> novel_queries = make_vectors<scalar_t>(novel_queries_count, dimensions,
                                                                stream_key(seed, "immutable_parity", 1));

    auto compare_modes = [&](auto query, std::size_t expansion_factor) noexcept {
        typename index_type_::traversal_context_t concurrent_context {{}}, immutable_context {{}};
        typename index_type_::search_config_t concurrent_config {};
        concurrent_config.expansion_factor = static_cast<decltype(concurrent_config.expansion_factor)>( //
            expansion_factor);
        typename index_type_::search_config_t immutable_config = concurrent_config;
        immutable_config.synchronization = synchronization_t::immutable_k;
        auto const [concurrent_result, concurrent_status] = index.search(query, concurrent_config, concurrent_context);
        auto const [immutable_result, immutable_status] = index.search(query, immutable_config, immutable_context);
        expect_equal(immutable_status.code, concurrent_status.code);
        expect_equal(immutable_result.count, concurrent_result.count);
        for (std::size_t match = 0; match != concurrent_result.count; ++match) {
            expect_equal(immutable_context.results[match].node_id, concurrent_context.results[match].node_id);
            expect_true(immutable_context.results[match].distance == concurrent_context.results[match].distance);
        }
    };
    for (std::size_t expansion_factor : {std::size_t {0}, std::size_t {64}}) {
        for (std::size_t sample = 0; sample != stored_queries_count; ++sample)
            compare_modes(vectors.view().row((sample * 31) % nodes_count).as_vector(), expansion_factor);
        for (std::size_t sample = 0; sample != novel_queries_count; ++sample)
            compare_modes(novel_queries.view().row(sample).as_vector(), expansion_factor);
    }
}

/** @ref dense_expect_immutable_parity, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_immutable_parity(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_immutable_parity<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** The external reader's cursor, exercised across chunk boundaries: full-prefix parity of IDs &
 *  vector bytes, follow semantics on the staged neighbor list, refresh stickiness - past the end
 *  stays past the end - a thread racing live inserters, and the immutable mode agreeing with the
 *  concurrent one on the settled graph. */
template <typename index_type_, typename concurrent_cursor_type_, typename immutable_cursor_type_,
          typename make_index_type_>
void dense_expect_cursor(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t const nodes_count = 1'200, racing_inserts_count = 64;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count + 1 + racing_inserts_count, dimensions,
                                                          stream_key(seed, "cursor"));
    expect_true(index.reserve({nodes_count + 1 + racing_inserts_count}).status == status_code_t::success_k);
    auto main_rows = vectors(nk::range(0, nodes_count), nk::slice);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count + 1 +
                                                                                  racing_inserts_count);
    expect_true(index.insert({keys.data(), nodes_count}, main_rows, {}, context).status == status_code_t::success_k);
    std::size_t const row_bytes = dense_rows_bytes(vectors.view(), 1);

    concurrent_cursor_type_ cursor {index};
    expect_true(!cursor); // A fresh cursor starts past the end.
    expect_equal(static_cast<std::size_t>(cursor.visible_count()), nodes_count);
    expect_equal(cursor.seek(0), status_code_t::uninitialized_k); // Before its reserve.
    expect_equal(cursor.reserve(counting_allocator<byte_t> {ledger}), status_code_t::success_k);

    // Full-prefix parity: ID order, exact stored bytes, every staged ID in bounds.
    expect_equal(cursor.seek(0), status_code_t::success_k);
    std::size_t visited = 0;
    for (; cursor; ++cursor, ++visited) {
        if (cursor.id() != static_cast<node_id_t>(visited)) expect_true(false);
        if (std::memcmp(cursor.vector().data(), vectors.view().row(visited).as_vector().data(), row_bytes) != 0)
            expect_true(false);
        auto const staged = cursor.neighbors();
        if (staged.size() > connectivity_16_k) expect_true(false);
        for (node_id_t const neighbor : staged)
            if (neighbor >= cursor.visible_count()) expect_true(false);
    }
    expect_equal(visited, nodes_count);

    // Follow semantics: every staged edge points where the neighbor list said; misuse refuses.
    expect_equal(cursor.seek(0), status_code_t::success_k);
    std::vector<node_id_t> first_row {cursor.neighbors().begin(), cursor.neighbors().end()};
    for (std::size_t slot = 0; slot != first_row.size(); ++slot) {
        expect_equal(cursor.seek(0), status_code_t::success_k);
        expect_equal(cursor.follow(static_cast<typename index_type_::candidates_count_t>(slot)),
                     status_code_t::success_k);
        expect_equal(cursor.id(), first_row[slot]);
    }
    expect_equal(cursor.seek(0), status_code_t::success_k);
    expect_equal(cursor.follow(static_cast<typename index_type_::candidates_count_t>(cursor.neighbors().size())),
                 status_code_t::malformed_k);
    expect_equal(cursor.seek(static_cast<node_id_t>(nodes_count)), status_code_t::malformed_k);

    // The snapshot is sticky until refreshed; a past-the-end cursor never resurrects.
    auto late_row = vectors(nk::range(nodes_count, nodes_count + 1), nk::slice);
    expect_true(index.insert({keys.data() + nodes_count, 1u}, late_row, {}, context).status ==
                status_code_t::success_k);
    expect_equal(static_cast<std::size_t>(cursor.visible_count()), nodes_count);
    expect_equal(cursor.seek(static_cast<node_id_t>(nodes_count)), status_code_t::malformed_k);
    cursor.refresh();
    expect_equal(static_cast<std::size_t>(cursor.visible_count()), nodes_count + 1);
    expect_equal(cursor.seek(static_cast<node_id_t>(nodes_count)), status_code_t::success_k);
    concurrent_cursor_type_ parked_cursor {index};
    expect_equal(parked_cursor.reserve(counting_allocator<byte_t> {ledger}), status_code_t::success_k);

    // A cursor thread races an inserting thread; every visited ID stays below that pass's
    // snapshot, and the parked cursor re-anchors past the end after the race.
    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, 2, fu::caller_inclusive_k)))
        return void(expect_true(false));
    std::size_t thread_failures_count = 0;
    fu::standard_atomic_ref<std::size_t> const thread_failures {thread_failures_count};
    bool writer_done_word = false;
    fu::standard_atomic_ref<bool> const writer_done {writer_done_word};
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            typename index_type_::traversal_context_t lane_context {{}};
            for (std::size_t ordinal = 0; ordinal != racing_inserts_count; ++ordinal) {
                auto racing_row = vectors(nk::range(nodes_count + 1 + ordinal, nodes_count + 2 + ordinal), nk::slice);
                if (index.insert({keys.data() + nodes_count + 1 + ordinal, 1u}, racing_row, {}, lane_context).status !=
                    status_code_t::success_k)
                    thread_failures.store_add(1);
                std::this_thread::yield();
            }
            writer_done.store(true, std::memory_order_release);
            assert_no_locks_held();
            return;
        }
        if (thread != 1) return;
        concurrent_cursor_type_ racing_cursor {index};
        if (racing_cursor.reserve(counting_allocator<byte_t> {ledger}) != status_code_t::success_k)
            return thread_failures.store_add(1);
        auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
        do {
            if (std::chrono::steady_clock::now() > deadline) return thread_failures.store_add(1); // A stuck writer.
            racing_cursor.refresh();
            node_id_t const snapshot = racing_cursor.visible_count();
            if (failed(racing_cursor.seek(0))) return thread_failures.store_add(1);
            for (; racing_cursor; ++racing_cursor)
                if (racing_cursor.id() >= snapshot) thread_failures.store_add(1);
        } while (!writer_done.load(std::memory_order_acquire));
    })));
    expect_equal(thread_failures.load(), 0u);
    parked_cursor.refresh();
    expect_true(!parked_cursor);
    expect_equal(static_cast<std::size_t>(parked_cursor.id()), index.size());
    expect_equal(parked_cursor.seek(0), status_code_t::success_k);

    // The immutable mode sweeps the settled graph to the same count & the same neighbor lists.
    immutable_cursor_type_ settled_cursor {index};
    expect_equal(settled_cursor.reserve(counting_allocator<byte_t> {ledger}), status_code_t::success_k);
    expect_equal(settled_cursor.seek(0), status_code_t::success_k);
    std::size_t settled_visited = 0;
    for (; settled_cursor; ++settled_cursor) ++settled_visited;
    expect_equal(settled_visited, index.size());
    for (std::size_t sample = 0; sample != 8; ++sample) {
        node_id_t const node = static_cast<node_id_t>((sample * 149) % index.size());
        expect_equal(settled_cursor.seek(node), status_code_t::success_k);
        expect_equal(cursor.seek(node), status_code_t::success_k);
        auto const settled_staged = settled_cursor.neighbors();
        auto const concurrent_staged = cursor.neighbors();
        expect_equal(settled_staged.size(), concurrent_staged.size());
        for (std::size_t slot = 0; slot != settled_staged.size(); ++slot)
            expect_equal(settled_staged[slot], concurrent_staged[slot]);
    }
}

/** @ref dense_expect_cursor, then the ledger it left must balance. */
template <typename index_type_, typename concurrent_cursor_type_, typename immutable_cursor_type_,
          typename make_index_type_>
void test_dense_cursor(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_cursor<index_type_, concurrent_cursor_type_, immutable_cursor_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Readers

#pragma region Allocation Budgets

/** A starved load publishes nothing; the documented contract destroys the victim. */
template <typename index_type_, typename node_id_type_, typename make_index_type_>
void dense_expect_starved_load(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                               typename index_type_::columns_view_t source, std::size_t nodes_count,
                               iota_view<node_id_type_> local) {
    std::unique_ptr<index_type_> const starved_ptr = make_index(ledger);
    index_type_& starved = *starved_ptr;
    auto [starved_capacity, starved_status] = starved.reserve({1});
    expect_true(starved_status == status_code_t::success_k);
    expect_true(starved_capacity < nodes_count); // The growth is genuine.
    ledger.byte_budget = ledger.live_bytes;
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(nodes_count);
    expect_equal(starved.load(source, to_span(keys), nodes_count, local), status_code_t::out_of_memory_k);
    expect_equal(starved.size(), 0u);
    ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
}

/** The ledger's byte budget provably starves exactly the chunk-growth path: a
 *  refused mid-insert growth is all-or-nothing & retryable, and a starved load
 *  publishes nothing - the fresh-index retry then succeeds with recall intact. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void dense_expect_allocation_budget(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t const extra_rows_count = 100;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    auto [capacity, reserve_status] = index.reserve({1'000});
    expect_true(reserve_status == status_code_t::success_k);
    std::size_t const fill_rows_count = capacity; // Fill to the brim, so one more row must grow.
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(fill_rows_count + extra_rows_count, dimensions,
                                                          stream_key(seed, "allocation_budget"));
    auto fill_rows = vectors(nk::range(0, fill_rows_count), nk::slice);
    auto extra_rows = vectors(nk::range(fill_rows_count, fill_rows_count + extra_rows_count), nk::slice);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(fill_rows_count + extra_rows_count);
    expect_true(index.insert({keys.data(), fill_rows_count}, fill_rows, {}, context).status ==
                status_code_t::success_k);

    // Freeze the budget: the growth election must refuse, all-or-nothing.
    ledger.byte_budget = ledger.live_bytes;
    expect_true(index.insert({keys.data() + fill_rows_count, extra_rows_count}, extra_rows, {}, context).status ==
                status_code_t::out_of_memory_k);
    expect_equal(index.size(), fill_rows_count);

    // Lift it: the identical batch is retryable.
    ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
    expect_true(index.insert({keys.data() + fill_rows_count, extra_rows_count}, extra_rows, {}, context).status ==
                status_code_t::success_k);
    std::size_t const nodes_count = fill_rows_count + extra_rows_count;
    expect_equal(index.size(), nodes_count);

    nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(nodes_count, dimensions);
    nk::matrix<node_id_t> saved_neighbors = make_matrix<node_id_t>(nodes_count, connectivity_16_k);
    expect_true(!saved_vectors.empty() && !saved_neighbors.empty());
    expect_equal(index.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                            synchronization_t::exclusive_k),
                 status_code_t::success_k);

    typename index_type_::columns_view_t const source {dense_matrix_bytes(saved_vectors.view()),
                                                       dense_matrix_bytes(saved_neighbors.view())};
    iota_view<node_id_t> const local {0, static_cast<node_id_t>(nodes_count)};

    dense_expect_starved_load<index_type_>(make_index, ledger, source, nodes_count, local);

    std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
    index_type_& restored = *restored_ptr;
    expect_true(restored.reserve({nodes_count}).status == status_code_t::success_k);
    expect_equal(restored.load(source, to_span(keys), nodes_count, local), status_code_t::success_k);
    expect_equal(restored.size(), nodes_count);
    if constexpr (kernel_ != nk_kernel_dot_k)
        dense_expect_self_recall(restored, make_index, vectors.view(), nodes_count);
}

/** @ref dense_expect_allocation_budget, then the ledger it left must balance. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
void test_dense_allocation_budget(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_allocation_budget<index_type_, kernel_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Shedding the frontier past the radius of full candidates is lossless: a walk whose frontier
 *  never fills ranks the same nodes, and the shedding frontier never outgrows its reservation. */
template <typename index_type_, typename make_index_type_>
void dense_expect_frontier_shedding(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    typename index_type_::traversal_context_t context {{}};
    expect_true(succeeded(index.reserve({dense_growth_nodes_k}).status));
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(dense_growth_nodes_k + 1, dimensions,
                                                          stream_key(seed, "shedding"));
    auto rows = vectors(nk::range(0, dense_growth_nodes_k), nk::slice);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k);
    expect_true(succeeded(index.insert({keys.data(), dense_growth_nodes_k}, rows, {}, context).status));

    // Same query, same seeds, two contexts: one sheds within 4 × ef, the other holds every
    // node and never fills.
    typename index_type_::insert_config_t wide;
    wide.expansion_factor = 64;
    typename index_type_::search_config_t config;
    config.expansion_factor = wide.expansion_factor;
    typename index_type_::traversal_context_t shedding {{}};
    typename index_type_::traversal_context_t unbounded {{}};
    expect_true(succeeded(index.reserve_traversal(shedding, wide)));
    expect_true(succeeded(unbounded.frontier.reserve(dense_growth_nodes_k)));
    std::size_t const frontier_capacity = shedding.frontier.capacity();
    auto const query = vectors.view().row(dense_growth_nodes_k).as_vector();
    auto const [shed, shed_status] = index.search(query, config, shedding);
    auto const [kept, kept_status] = index.search(query, config, unbounded);
    expect_true(succeeded(shed_status) && succeeded(kept_status));
    expect_equal(shed.count, kept.count);
    for (std::size_t rank = 0; rank < shed.count && rank < kept.count; ++rank)
        expect_equal(shedding.results[rank].node_id, unbounded.results[rank].node_id);
    dense_expect_ascending(shedding);
    expect_equal(shedding.frontier.capacity(), frontier_capacity);
    expect_equal(shed.saturated_walks, 0u);
}

/** @ref dense_expect_frontier_shedding, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
void test_dense_frontier_shedding(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_frontier_shedding<index_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** A refused re-table saturates the walk, it never fails it: under a frozen budget a search answers
 *  @c success_k with a short ascending ranking and one saturated walk, and an insertion still
 *  publishes its entry, wired thinner. One-wide candidates make the first expansion's quad the
 *  re-table that is refused. */
template <typename index_type_, typename make_index_type_>
void dense_expect_saturated(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    expect_true(succeeded(index.reserve({dense_growth_nodes_k + 1}).status));
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(dense_growth_nodes_k + 2, dimensions,
                                                          stream_key(seed, "saturated"));
    typename index_type_::traversal_context_t warm {counting_allocator_t(ledger)};
    auto rows = vectors(nk::range(0, dense_growth_nodes_k), nk::slice);
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(dense_growth_nodes_k + 1);
    expect_true(succeeded(index.insert({keys.data(), dense_growth_nodes_k}, rows, {}, warm).status));

    typename index_type_::insert_config_t narrow;
    narrow.expansion_factor = 1;
    typename index_type_::search_config_t config;
    config.expansion_factor = narrow.expansion_factor;
    typename index_type_::traversal_context_t starved {counting_allocator_t(ledger)};
    expect_true(succeeded(index.reserve_traversal(starved, narrow)));
    ledger.byte_budget = ledger.live_bytes;

    auto const [search_result, search_status] = index.search(vectors.view().row(dense_growth_nodes_k).as_vector(),
                                                             config, starved);
    expect_equal(search_status, status_code_t::success_k);
    expect_true(search_result.count > 0);
    expect_equal(search_result.saturated_walks, 1u);
    dense_expect_ascending(starved);

    auto extra = vectors(nk::range(dense_growth_nodes_k, dense_growth_nodes_k + 1), nk::slice);
    auto const [insert_result, insert_status] = index.insert({keys.data() + dense_growth_nodes_k, 1u}, extra, {},
                                                             starved, narrow);
    expect_equal(insert_status, status_code_t::success_k);
    expect_equal(insert_result.saturated_walks, 1u);
    expect_equal(index.size(), dense_growth_nodes_k + 1);
    ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
}

/** @ref dense_expect_saturated, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
void test_dense_saturated(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_saturated<index_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** A round is atomic by construction: under every budget short of its plan's needs the plan refuses
 *  with @c out_of_memory_k and leaves the neighbors byte-identical, and once planned and its
 *  context reserved it runs every task and barrier to its end, and finishes, under a budget frozen
 *  where the reservation left it, since neither @c run, @c advance nor @c finish allocates. */
template <typename index_type_, typename make_index_type_>
void dense_expect_refine_atomic(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    std::size_t const rows = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    expect_true(succeeded(index.reserve({rows}).status));
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "refine_atomic"));
    typename index_type_::traversal_context_t context {counting_allocator_t(ledger)};
    typename index_type_::insert_config_t cheap;
    cheap.expansion_factor = 4;
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(rows);
    expect_true(succeeded(index.insert(to_span(keys), vectors.view(), {}, context, cheap).status));

    nk::matrix<scalar_t> saved_vectors = make_matrix<scalar_t>(rows, dimensions);
    nk::matrix<node_id_t> before = make_matrix<node_id_t>(rows, connectivity_16_k);
    nk::matrix<node_id_t> after = make_matrix<node_id_t>(rows, connectivity_16_k);
    expect_true(!saved_vectors.empty() && !before.empty() && !after.empty());
    auto const snapshot = [&](nk::matrix<node_id_t>& neighbors) {
        return index.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(neighbors)},
                          synchronization_t::exclusive_k);
    };
    expect_equal(snapshot(before), status_code_t::success_k);

    expect_equal(index.reserve_traversal(context, {}), status_code_t::success_k);
    std::size_t refusals = 0;
    for (std::ptrdiff_t slack = 0;; slack = slack * 2 + 64) {
        ledger.byte_budget = ledger.live_bytes + slack;
        auto [plan, status] = index.plan_refine();
        if (failed(status)) {
            expect_equal(status, status_code_t::out_of_memory_k);
            expect_equal(snapshot(after), status_code_t::success_k);
            expect_true(std::memcmp(before.data(), after.data(), dense_matrix_bytes(before).size()) == 0);
            ++refusals;
            continue;
        }
        // Tasks cannot fail on a context the plan reserved, so none grows past the budget.
        ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
        expect_equal(plan.reserve(context), status_code_t::success_k);
        ledger.byte_budget = ledger.live_bytes;
        do
            for (typename index_type_::refine_plan_t::task_t const task : plan.tasks())
                expect_equal(plan.run(task, context), status_code_t::success_k);
        while (plan.advance());
        auto const [refined, refined_status] = plan.finish();
        expect_equal(refined_status, status_code_t::success_k);
        expect_true(refined.reselected_nodes >= rows && refined.changed_nodes > 0);
        break;
    }
    ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
    expect_true(refusals > 0);
}

/** @ref dense_expect_refine_atomic, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
void test_dense_refine_atomic(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_refine_atomic<index_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

/** Racing a bounded arena: with the budget frozen at exactly two batches of spare
 *  capacity, four uniform threads resolve to exactly two winners & two @c out_of_memory_k
 *  refusals, the winners' IDs come out dense & duplicate-free, the ID-space ceiling still
 *  answers @c overflow_k, and a lifted budget makes a refused batch retryable. */
template <typename index_type_, typename make_index_type_>
void dense_expect_overflow_race(make_index_type_ const& make_index, allocation_ledger_t& ledger, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const racing_threads_count = 4, batch_rows_count = 512;
    std::size_t const contested_rows_count = 2 * batch_rows_count;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) ||
        fu::failed(pool.spawn(topology, racing_threads_count, fu::caller_inclusive_k)))
        return void(expect_true(false));

    std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
    index_type_& index = *index_ptr;
    auto [capacity, status] = index.reserve({contested_rows_count});
    expect_true(status == status_code_t::success_k);
    expect_true(capacity >= contested_rows_count);

    // Pre-fill any chunk-rounding surplus, so exactly two batches of capacity remain.
    std::size_t const prefill_rows_count = capacity - contested_rows_count;
    typename index_type_::traversal_context_t context {{}};
    if (prefill_rows_count) {
        nk::matrix<scalar_t> prefill_vectors = make_vectors<scalar_t>(prefill_rows_count, dimensions,
                                                                      stream_key(seed, "overflow_race", 1));
        std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(prefill_rows_count);
        expect_true(index.insert(to_span(keys), prefill_vectors.view(), {}, context).status ==
                    status_code_t::success_k);
    }

    auto threads = dense_make_threads<index_type_>(make_index, racing_threads_count, batch_rows_count, dimensions,
                                                   stream_key(seed, "overflow_race"));
    nk::matrix<node_id_t> assigned_ids = make_matrix<node_id_t>(racing_threads_count, batch_rows_count);
    std::fill_n(assigned_ids.data(), assigned_ids.size(), node_id_unknown_k);
    std::array<status_t, 4> thread_statuses {};
    ledger.byte_budget = ledger.live_bytes; // No growth: the arena is the arena.

    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(racing_threads_count *
                                                                                  batch_rows_count);
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread >= racing_threads_count) return;
        thread_statuses[thread] = index
                                      .insert({keys.data() + thread * batch_rows_count, batch_rows_count},
                                              threads.slices[thread].view(), assigned_ids.row(thread).as_vector(),
                                              threads.contexts[thread])
                                      .status;
        assert_no_locks_held();
    })));

    std::size_t winners_count = 0;
    std::size_t refused_thread = racing_threads_count;
    std::vector<node_id_t> winner_ids;
    for (std::size_t thread = 0; thread != racing_threads_count; ++thread) {
        if (thread_statuses[thread] == status_code_t::success_k) {
            ++winners_count;
            std::span<node_id_t const> const won = to_span(assigned_ids.view().row(thread).as_vector());
            winner_ids.insert(winner_ids.end(), won.begin(), won.end());
        }
        else {
            expect_equal(thread_statuses[thread], status_code_t::out_of_memory_k);
            refused_thread = thread;
        }
    }
    expect_equal(winners_count, 2u);
    expect_equal(index.size(), capacity);
    expect_contiguous_ids(std::span {winner_ids}, prefill_rows_count);

    // The genuine overflow_k belongs to the ID-space ceiling alone, refused before any
    // counter dance or allocation - a probe only a 32-bit ID space can express.
    using node_id_t = typename index_type_::node_id_t;
    if constexpr (sizeof(node_id_t) == 4)
        expect_equal(index.reserve({std::size_t {1} << 33}).status, status_code_t::overflow_k);

    // A lifted budget makes a refused batch retryable.
    ledger.byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
    expect_true(refused_thread != racing_threads_count);
    expect_true(index
                    .insert({keys.data() + refused_thread * batch_rows_count, batch_rows_count},
                            threads.slices[refused_thread].view(), {}, //
                            threads.contexts[refused_thread])
                    .status == status_code_t::success_k);
    expect_equal(index.size(), capacity + batch_rows_count);
}

/** @ref dense_expect_overflow_race, then the ledger it left must balance. */
template <typename index_type_, typename make_index_type_>
void test_dense_overflow_race(make_index_type_ const& make_index, u64_t seed) {
    allocation_ledger_t ledger;
    dense_expect_overflow_race<index_type_>(make_index, ledger, seed);
    expect_equal(ledger.live_bytes, 0);
}

#pragma endregion Allocation Budgets

} // namespace unum::usearch::test
