/**
 *  @file cpp/usearch/dense_index.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief RAII handles over the dynamically dispatched dense modality, adding no ABI of its own and
 *      throwing nothing.
 *
 *  Every verb takes the library's argument struct, so the caller names the inputs with designated
 *  initializers. The wrapper fills the handle, executor, arena and outcome slots, runs the call to
 *  completion, and returns the verb's first output beside the outcome, while any further output
 *  stays a pointer the caller sets in the struct.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [executor, executor_status, executor_message] = us::executor_t::for_ram();
 *  auto [index, index_status, index_message] = us::dense_index_t::init_ram({
 *      .metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k,
 *      .dimensions = 256});
 *  auto [arena, arena_status, arena_message] = us::arena_t::make({.executor = executor.handle()});
 *  auto [matches, status, message] = index.search(
 *      {.count = 1, .queries = {query, 0, 256, usearch_scalar_f32_k}, .wanted = 10}, arena,
 *      executor);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <utility> // `std::exchange`, `std::swap`

#include "usearch/dense_index.h"

namespace unum::usearch {

/** A call's first output beside its status and static message, decomposed as <tt>auto
 *  [value, status, message]</tt>; the value is meaningful only on success. */
template <typename value_type_ = void>
struct [[nodiscard]] outcome {
    value_type_ value {};
    usearch_status_t status = usearch_status_success_k;
    char const* message = nullptr;

    explicit operator bool() const noexcept { return status == usearch_status_success_k; }
};

/** The outcome of a call that outputs nothing but its status. */
template <>
struct [[nodiscard]] outcome<void> {
    usearch_status_t status = usearch_status_success_k;
    char const* message = nullptr;

    explicit operator bool() const noexcept { return status == usearch_status_success_k; }
};

/** Runs @p verb to completion with @p arguments, which name everything but the outcome. */
template <typename arguments_type_>
outcome<> run_to_completion(void (*verb)(arguments_type_*), arguments_type_& arguments) noexcept {
    usearch_outcome_t reported {};
    arguments.outcome = &reported;
    if constexpr (requires { arguments.completion; }) arguments.completion = usearch_completion_blocking_k;
    verb(&arguments);
    return {reported.status, reported.message};
}

/** Owns one executor; an empty one runs every call on the caller's own thread. */
class executor_t {
  public:
    executor_t() noexcept = default;
    executor_t(executor_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    executor_t& operator=(executor_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    executor_t(executor_t const&) = delete;
    executor_t& operator=(executor_t const&) = delete;
    ~executor_t() noexcept { usearch_executor_free(handle_); }

    /** Spawns host threads for RAM-resident indexes, interleaved across the compute domains local
     *  to the memory domains @p options names. */
    static outcome<executor_t> for_ram(usearch_executor_for_ram_t options = {}) noexcept {
        return spawned_(usearch_executor_for_ram, options);
    }

    /**
     *  Wraps a borrowed ForkUnion pool, so several libraries in one process share its threads.
     *
     *  @par Statuses:
     *  - @c usearch_status_unsupported_k Always, as borrowed pools are not wired yet.
     */
    static outcome<executor_t> on_pool(usearch_executor_on_pool_t options) noexcept {
        return spawned_(usearch_executor_on_pool, options);
    }

    /**
     *  Opens one submission stream on every distinct device @p options names.
     *
     *  @par Statuses:
     *  - @c usearch_status_unsupported_k Borrowed streams, which are not wired yet.
     *  - @c usearch_status_device_k An ordinal past the listing, or no device named.
     *  - @c usearch_status_config_mismatch_k Devices of more than one vendor.
     */
    static outcome<executor_t> for_vram(usearch_executor_for_vram_t options = {}) noexcept {
        return spawned_(usearch_executor_for_vram, options);
    }

    /**
     *  Spawns host threads for NVMe-resident indexes, one submission ring apiece.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k @c NULL paths for a nonzero count, or an unknown
     *    engine class.
     *  - @c usearch_status_config_mismatch_k Kernel paths mixed with transport identifiers, or SPDK
     *    pinned over kernel paths or a kernel engine over transport identifiers.
     *  - @c usearch_status_unsupported_k Transport identifiers in a build without SPDK, or a pinned
     *    engine this build lacks.
     */
    static outcome<executor_t> for_nvme(usearch_executor_for_nvme_t options = {}) noexcept {
        return spawned_(usearch_executor_for_nvme, options);
    }

    /**
     *  @brief Waits until no call holds the executor, then sets its threads' idle mood.
     *
     *  @par Statuses:
     *  - @c usearch_status_contended_k A call running on the executor holds it, as for a drain from
     *    that call's own progress handler, predicate or host function.
     *  - @c usearch_status_device_k A GPU executor with no vendor behind it.
     */
    outcome<> drain(usearch_executor_drain_t arguments = {}) noexcept {
        arguments.executor = handle_;
        return run_to_completion(usearch_executor_drain, arguments);
    }

    usearch_executor_t handle() const noexcept { return handle_; }

  private:
    template <typename options_type_>
    static outcome<executor_t> spawned_(void (*verb)(options_type_*), options_type_& options) noexcept {
        executor_t spawned;
        options.executor = &spawned.handle_;
        outcome<> const reported = run_to_completion(verb, options);
        return {std::move(spawned), reported.status, reported.message};
    }

    usearch_executor_t handle_ = nullptr;
};

/** Owns one arena; every call writing outputs into it rewinds it first. */
class arena_t {
  public:
    arena_t() noexcept = default;
    arena_t(arena_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    arena_t& operator=(arena_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    arena_t(arena_t const&) = delete;
    arena_t& operator=(arena_t const&) = delete;
    ~arena_t() noexcept { usearch_arena_free(handle_); }

    /** Creates an empty arena where the executor @p options names computes. */
    static outcome<arena_t> make(usearch_arena_init_t options = {}) noexcept {
        arena_t made;
        options.arena = &made.handle_;
        outcome<> const reported = run_to_completion(usearch_arena_init, options);
        return {std::move(made), reported.status, reported.message};
    }

    usearch_arena_t handle() const noexcept { return handle_; }

  private:
    usearch_arena_t handle_ = nullptr;
};

/** Owns one cursor, opened by @c dense_index_t::cursor_init. */
class dense_cursor_t {
  public:
    dense_cursor_t() noexcept = default;
    dense_cursor_t(dense_cursor_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    dense_cursor_t& operator=(dense_cursor_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    dense_cursor_t(dense_cursor_t const&) = delete;
    dense_cursor_t& operator=(dense_cursor_t const&) = delete;
    ~dense_cursor_t() noexcept { usearch_dense_cursor_free(handle_); }

    usearch_dense_cursor_t handle() const noexcept { return handle_; }

  private:
    friend class dense_index_t;
    usearch_dense_cursor_t handle_ = nullptr;
};

/** Owns one search stream, opened by @c dense_index_t::search_stream_init. */
class dense_search_stream_t {
  public:
    dense_search_stream_t() noexcept = default;
    dense_search_stream_t(dense_search_stream_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    dense_search_stream_t& operator=(dense_search_stream_t&& other) noexcept {
        return std::swap(handle_, other.handle_), *this;
    }
    dense_search_stream_t(dense_search_stream_t const&) = delete;
    dense_search_stream_t& operator=(dense_search_stream_t const&) = delete;
    ~dense_search_stream_t() noexcept { usearch_dense_search_stream_free(handle_); }

    usearch_dense_search_stream_t handle() const noexcept { return handle_; }

  private:
    friend class dense_index_t;
    usearch_dense_search_stream_t handle_ = nullptr;
};

/** Owns one dense index; destruction waits for its deferred calls. */
class dense_index_t {
  public:
    dense_index_t() noexcept = default;
    dense_index_t(dense_index_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    dense_index_t& operator=(dense_index_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    dense_index_t(dense_index_t const&) = delete;
    dense_index_t& operator=(dense_index_t const&) = delete;
    ~dense_index_t() noexcept { usearch_dense_free(handle_); }

    usearch_dense_index_t handle() const noexcept { return handle_; }

#pragma region Lifecycle and Mutation

    /**
     *  Creates an empty RAM index of the geometry @p options freezes.
     *
     *  @par Statuses:
     *  - @c usearch_status_config_mismatch_k Zero @c options.dimensions, dimensions a packed
     *    quantization leaves a storage value partly filled by, or a connectivity that is not a
     *    power of two.
     */
    static outcome<dense_index_t> init_ram(usearch_dense_init_ram_t options) noexcept {
        return created_(usearch_dense_init_ram, options);
    }

    /**
     *  Creates an empty VRAM index of the geometry @p options freezes, on the devices it names.
     *
     *  @par Statuses:
     *  - @c usearch_status_config_mismatch_k Zero @c options.dimensions, a connectivity that is not
     *    a power of two, or on NVIDIA and AMD a geometry whose walks cannot fit the device's shared
     *    memory.
     */
    static outcome<dense_index_t> init_vram(usearch_dense_init_vram_t options) noexcept {
        return created_(usearch_dense_init_vram, options);
    }

    /**
     *  Copies a checkpoint into a new, mutable index in host memory.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty path, or a checkpoint or part path that does
     *    not parse.
     *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header or its
     *    key column does not parse, or its connectivity or dimensions are unusable.
     *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does
     *    not score, or a kernel a shard cannot finalize.
     *  - @c usearch_status_unsupported_k A quantization this build leaves out, or a pinned class
     *    the CPU does not admit.
     *  - @c usearch_status_uninitialized_k No memory domain to place a shard on, or @p executor
     *    holds no pool workers.
     *  - @c usearch_status_config_mismatch_k A memory domain refusing a shard, or the checkpoint's
     *    geometry, dtypes or metric differ from the shards'.
     *  - @c usearch_status_device_k @p executor is not a RAM executor.
     *  - @c usearch_status_contended_k Another call holds @p executor, or its pool runs another
     *    dispatch.
     *  - @c usearch_status_overflow_k A part passes a shard's bound, a key column or the locator
     *    map.
     */
    static outcome<dense_index_t> load_ram(usearch_dense_load_ram_t options, executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_ram, options);
    }

    /**
     *  Copies a checkpoint into a new, mutable index in device memory.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty path, or a checkpoint or part path that does
     *    not parse.
     *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header or its
     *    key column does not parse, or its connectivity or dimensions are unusable.
     *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does
     *    not score, or a kernel a shard cannot finalize.
     *  - @c usearch_status_unsupported_k A quantization this build leaves out, or devices that do
     *    not run the pinned class.
     *  - @c usearch_status_config_mismatch_k Devices of more than one vendor or generation,
     *    @p executor driving no stream on some shard's device, or the checkpoint's geometry,
     *    dtypes or metric differing from the shards'.
     *  - @c usearch_status_device_k An ordinal past the listing, devices this build carries no
     *    kernels for, or an executor that is not a VRAM executor.
     *  - @c usearch_status_contended_k Another call holds @p executor.
     *  - @c usearch_status_overflow_k A part passes a shard's bound, a key column or the locator
     *    map.
     */
    static outcome<dense_index_t> load_vram(usearch_dense_load_vram_t options,
                                            executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_vram, options);
    }

    /**
     *  Copies a checkpoint from a memory buffer into a new, mutable index in host memory.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty buffer.
     *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header or its key
     *    column does not parse, or its connectivity or dimensions are unusable.
     *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does
     *    not score, or a kernel a shard cannot finalize.
     *  - @c usearch_status_unsupported_k A quantization this build leaves out, or a pinned class
     *    the CPU does not admit.
     *  - @c usearch_status_uninitialized_k No memory domain to place a shard on, or @p executor
     *    holds no pool workers.
     *  - @c usearch_status_config_mismatch_k A memory domain refusing a shard, or the checkpoint's
     *    geometry, dtypes or metric differ from the shards'.
     *  - @c usearch_status_device_k @p executor is not a RAM executor.
     *  - @c usearch_status_contended_k Another call holds @p executor, or its pool runs another
     *    dispatch.
     *  - @c usearch_status_overflow_k A part passes a shard's bound, a key column or the locator
     *    map.
     */
    static outcome<dense_index_t> load_ram_buffer(usearch_dense_load_ram_buffer_t options,
                                                  executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_ram_buffer, options);
    }

    /**
     *  Copies a checkpoint from a memory buffer into a new, mutable index in device memory.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty buffer.
     *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header or its key
     *    column does not parse, or its connectivity or dimensions are unusable.
     *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does
     *    not score, or a kernel a shard cannot finalize.
     *  - @c usearch_status_unsupported_k A quantization this build leaves out, or devices that do
     *    not run the pinned class.
     *  - @c usearch_status_config_mismatch_k Devices of more than one vendor or generation,
     *    @p executor driving no stream on some shard's device, or the checkpoint's geometry,
     *    dtypes or metric differing from the shards'.
     *  - @c usearch_status_device_k An ordinal past the listing, devices this build carries no
     *    kernels for, or an executor that is not a VRAM executor.
     *  - @c usearch_status_contended_k Another call holds @p executor.
     *  - @c usearch_status_overflow_k A part passes a shard's bound, a key column or the locator
     *    map.
     */
    static outcome<dense_index_t> load_vram_buffer(usearch_dense_load_vram_buffer_t options,
                                                   executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_vram_buffer, options);
    }

    /**
     *  Serves a checkpoint in place through a memory mapping, as a read-only index.
     *
     *  @par Statuses:
     *  - @c usearch_status_unsupported_k Always, as RAM shards cannot serve a mapping in place yet.
     */
    static outcome<dense_index_t> view_ram(usearch_dense_view_ram_t options) noexcept {
        return created_(usearch_dense_view_ram, options);
    }

    /**
     *  Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines.
     *
     *  @par Statuses:
     *  - @c usearch_status_device_k @p executor is not an NVMe executor.
     *  - @c usearch_status_invalid_argument_k An empty path, @c NULL paths for a nonzero count, or
     *    a path that does not parse or names a base off its medium's block.
     *  - @c usearch_status_config_mismatch_k Parts disagreeing with each other, with the scalar or
     *    with the devices named.
     *  - @c usearch_status_unsupported_k A transport identifier in a build without SPDK, a
     *    quantization this build leaves out, or a pinned class the CPU does not admit.
     *  - @c usearch_status_malformed_k The manifest names no part, or a manifest or header does not
     *    parse.
     *  - @c usearch_status_unknown_kind_k An unknown quantization, a metric its quantization does
     *    not score, or a kernel the stored scalar's kits cannot finalize.
     */
    static outcome<dense_index_t> view_nvme(usearch_dense_view_nvme_t options,
                                            executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_view_nvme, options);
    }

    /**
     *  Grows the shards ahead of a number of incoming vectors.
     *
     *  @par Statuses:
     *  - @c usearch_status_read_only_k The index is an NVMe view.
     *  - @c usearch_status_uninitialized_k No shard is placed.
     *  - @c usearch_status_unknown_kind_k The stored scalar's kits cannot finalize the metric.
     *  - @c usearch_status_config_mismatch_k The frozen geometry differs from a shard's, or on VRAM
     *    a shard's neighbors live on another device or no walk fits the device.
     *  - @c usearch_status_overflow_k The capacity passes a shard's ceiling, its ID space, its
     *    chunk directory or its key column.
     */
    outcome<> reserve(usearch_dense_reserve_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_reserve, arguments, executor);
    }

    /**
     *  Erases all entries, keeping the reserved capacity and frozen geometry.
     *
     *  @par Statuses:
     *  - @c usearch_status_read_only_k The index is an NVMe view.
     *  - @c usearch_status_contended_k A call in flight, a batch or load holding claims, or a plan
     *    holding the index closed.
     */
    outcome<> clear(usearch_dense_clear_t arguments = {}, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_clear, arguments, executor);
    }

    /**
     *  @brief Adds a run of vectors under their keys, all or nothing per call.
     *
     *  @par Statuses:
     *  - @c usearch_status_read_only_k The index is a view.
     *  - @c usearch_status_contended_k A refinement, absorption or clear holds the index, or on
     *    VRAM a save or another insertion does; nothing written.
     */
    outcome<> insert(usearch_dense_insert_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_insert, arguments, executor);
    }

    /**
     *  @brief Refines the graph, answering its report: the nodes re-selected and changed, and the
     *      islands found and reconnected.
     *
     *  @par Statuses:
     *  - @c usearch_status_device_k @p executor 's kind does not match the residency.
     *  - @c usearch_status_read_only_k The index is an NVMe view.
     *  - @c usearch_status_uninitialized_k No shard is placed or froze its geometry, @p executor
     *    holds no threads, or a VRAM index keeps such an earlier failure.
     *  - @c usearch_status_contended_k An insertion, load, save, refinement or absorption holds the
     *    index, or another call holds @p executor or its pool.
     *  - @c usearch_status_config_mismatch_k @p executor drives no stream on some VRAM shard's
     *    device, or a VRAM index keeps such an earlier failure.
     *  - @c usearch_status_overflow_k A queue capacity wraps when rounded, or a VRAM index keeps an
     *    earlier overflow.
     *  - @c usearch_status_unsupported_k or @c usearch_status_unsupported_layout_k A VRAM index
     *    keeps such an earlier failure.
     */
    outcome<usearch_refine_report_t> refine(usearch_dense_refine_t arguments = {},
                                            executor_t const& executor = {}) noexcept {
        return answered_(usearch_dense_refine, arguments, &usearch_dense_refine_t::report, executor);
    }

    /**
     *  Audits how much of the graph a search can reach.
     *
     *  @par Statuses:
     *  - @c usearch_status_device_k @p executor 's kind does not match the residency.
     *  - @c usearch_status_uninitialized_k No shard is placed, @p executor holds no pool workers or
     *    streams, or a VRAM index keeps such an earlier failure.
     *  - @c usearch_status_contended_k Another call holds @p executor or its pool, or a destructive
     *    window overlapped the audit twice.
     *  - @c usearch_status_config_mismatch_k @p executor drives no stream on some VRAM shard's
     *    device, an NVMe medium the engines cannot read, or a VRAM index keeps such an earlier
     *    failure.
     *  - @c usearch_status_overflow_k A VRAM index keeps an earlier overflow.
     *  - @c usearch_status_unsupported_k or @c usearch_status_unsupported_layout_k A VRAM index
     *    keeps such an earlier failure.
     */
    outcome<usearch_reachability_t> health(usearch_dense_health_t arguments = {},
                                           executor_t const& executor = {}) noexcept {
        return answered_(usearch_dense_health, arguments, &usearch_dense_health_t::reachability, executor);
    }

    /**
     *  Absorbs the donor indexes @p arguments names into this one, consuming them.
     *
     *  @par Statuses:
     *  - @c usearch_status_read_only_k The index is an NVMe view.
     *  - @c usearch_status_invalid_argument_k A @c NULL donor slot, or @c NULL donors for a nonzero
     *    count.
     *  - @c usearch_status_config_mismatch_k A donor of another residency, scalar, class, metric or
     *    geometry, a donor on a device the executor does not drive, or this index itself.
     *  - @c usearch_status_contended_k An insertion in flight on this index or on the next donor,
     *    or on VRAM a save of either; that donor and every later one stay live indexes.
     */
    outcome<> absorb(usearch_dense_absorb_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_absorb, arguments, executor);
    }

#pragma endregion Lifecycle and Mutation

#pragma region Reads

    /**
     *  @brief Finds the nearest entries of every query @p arguments names, written into @p arena.
     *
     *  @par Statuses:
     *  - @c usearch_status_unknown_kind_k A traversal other than a walk or a scan.
     *  - @c usearch_status_unsupported_k A scan of an AMD or Apple VRAM index, or a predicate on
     *    any VRAM index.
     */
    outcome<usearch_matches_view_t> search(usearch_dense_search_t arguments, arena_t const& arena,
                                           executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_search, arguments, &usearch_dense_search_t::matches, executor);
    }

    /**
     *  Answers how many entries @c arguments.key holds.
     *
     *  @par Statuses:
     *  - @c usearch_status_device_k @p executor 's kind does not match the residency.
     *  - @c usearch_status_uninitialized_k @p executor holds no pool workers.
     *  - @c usearch_status_contended_k Another call holds @p executor, or its pool runs another
     *    dispatch.
     *  - @c usearch_status_config_mismatch_k An NVMe medium the engines cannot read.
     */
    outcome<std::size_t> count(usearch_dense_count_t arguments, executor_t const& executor = {}) noexcept {
        return answered_(usearch_dense_count, arguments, &usearch_dense_count_t::entries, executor);
    }

    /**
     *  Answers the vectors @c arguments.key holds; @c arguments.written counts them when the caller
     *  sets it.
     *
     *  @par Statuses:
     *  - @c usearch_status_unknown_kind_k An unknown output scalar.
     *  - @c usearch_status_shape_mismatch_k Sub-byte rows ending inside a byte.
     *  - @c usearch_status_invalid_argument_k An empty @p arena.
     *  - @c usearch_status_device_k An arena or an executor on another residency or device.
     *  - @c usearch_status_contended_k @p arena serves another call, another call holds
     *    @p executor or its pool, or a destructive window overlapped the read twice.
     *  - @c usearch_status_uninitialized_k @p executor holds no pool workers, or a VRAM index has
     *    no shards.
     *  - @c usearch_status_config_mismatch_k @p executor drives no stream on some shard's device,
     *    or an NVMe medium the engines cannot read.
     */
    outcome<usearch_rows_view_t> get(usearch_dense_get_t arguments, arena_t const& arena,
                                     executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_get, arguments, &usearch_dense_get_t::vectors, executor);
    }

    /**
     *  Opens a cursor over every entry of the index.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k @c NULL starts for a nonzero count of them.
     */
    outcome<dense_cursor_t> cursor_init(usearch_dense_cursor_init_t arguments = {}) noexcept {
        dense_cursor_t opened;
        arguments.cursor = &opened.handle_, arguments.index = handle_;
        outcome<> const reported = run_to_completion(usearch_dense_cursor_init, arguments);
        return {std::move(opened), reported.status, reported.message};
    }

    /**
     *  Answers the next keys @p cursor walks to; the entries and their count land where the caller
     *  points.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k @p cursor was opened on another index.
     */
    outcome<usearch_keys_view_t> cursor_next(dense_cursor_t const& cursor, usearch_dense_cursor_next_t arguments,
                                             arena_t const& arena, executor_t const& executor = {}) noexcept {
        arguments.cursor = cursor.handle(), arguments.arena = arena.handle();
        return answered_(usearch_dense_cursor_next, arguments, &usearch_dense_cursor_next_t::keys, executor);
    }

    /**
     *  Opens a stream of matches that later calls extend.
     *
     *  @par Statuses:
     *  - @c usearch_status_unsupported_k No residency implements the stream yet.
     */
    outcome<dense_search_stream_t> search_stream_init(usearch_dense_search_stream_init_t arguments) noexcept {
        dense_search_stream_t opened;
        arguments.stream = &opened.handle_, arguments.index = handle_;
        outcome<> const reported = run_to_completion(usearch_dense_search_stream_init, arguments);
        return {std::move(opened), reported.status, reported.message};
    }

    /**
     *  @brief Answers the next matches of @p stream, writing them into @p arena.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k @p stream was opened on another index.
     */
    outcome<usearch_matches_view_t> search_stream_next(dense_search_stream_t const& stream,
                                                       usearch_dense_search_stream_next_t arguments,
                                                       arena_t const& arena, executor_t const& executor = {}) noexcept {
        arguments.stream = stream.handle(), arguments.arena = arena.handle();
        return answered_(usearch_dense_search_stream_next, arguments, &usearch_dense_search_stream_next_t::matches,
                         executor);
    }

    outcome<usearch_dense_index_description_t> describe(usearch_dense_describe_t arguments = {},
                                                        executor_t const& executor = {}) const noexcept {
        return answered_(usearch_dense_describe, arguments, &usearch_dense_describe_t::description, executor);
    }

    /**
     *  Writes the index to the checkpoint path @p arguments names.
     *
     *  @par Statuses:
     *  - @c usearch_status_contended_k A RAM save meeting an insertion in flight or one publishing
     *    under it, or a VRAM save meeting an insertion, refinement, absorption or another save in
     *    flight.
     */
    outcome<> save(usearch_dense_save_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_save, arguments, executor);
    }

    /**
     *  Writes the index into one monolith in @p arena.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty @p arena.
     *  - @c usearch_status_device_k An arena or an executor on another residency or device.
     *  - @c usearch_status_contended_k @p arena serves another call, another call holds
     *    @p executor or its pool, a dispensed batch has yet to publish, or a call, batch, load or
     *    plan holds the index.
     *  - @c usearch_status_unsupported_k The index is an NVMe view, which already is its
     *    checkpoint.
     *  - @c usearch_status_uninitialized_k No shard is placed or froze its geometry, or @p executor
     *    holds no pool workers.
     *  - @c usearch_status_unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c usearch_status_config_mismatch_k @p executor drives no stream on some VRAM shard's
     *    device.
     */
    outcome<usearch_bytes_view_t> save_buffer(usearch_dense_save_buffer_t arguments, arena_t const& arena,
                                              executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_save_buffer, arguments, &usearch_dense_save_buffer_t::monolith, executor);
    }

    /**
     *  Reads a checkpoint's geometry & metric without loading it.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty path, or a path that does not parse.
     *  - @c usearch_status_malformed_k The file is empty, or its manifest or header does not parse.
     */
    static outcome<usearch_dense_metadata_t> metadata_read(usearch_dense_metadata_read_t arguments,
                                                           executor_t const& executor = {}) noexcept {
        usearch_dense_metadata_t metadata {};
        arguments.executor = executor.handle(), arguments.metadata = &metadata;
        outcome<> const reported = run_to_completion(usearch_dense_metadata_read, arguments);
        return {metadata, reported.status, reported.message};
    }

    /**
     *  Reads geometry & metric from a serialized checkpoint buffer.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k An empty buffer.
     *  - @c usearch_status_malformed_k The manifest or the header does not parse.
     */
    static outcome<usearch_dense_metadata_t> metadata_read_buffer(usearch_dense_metadata_read_buffer_t arguments,
                                                                  executor_t const& executor = {}) noexcept {
        usearch_dense_metadata_t metadata {};
        arguments.executor = executor.handle(), arguments.metadata = &metadata;
        outcome<> const reported = run_to_completion(usearch_dense_metadata_read_buffer, arguments);
        return {metadata, reported.status, reported.message};
    }

    /**
     *  Answers the products; their count and the width they needed land where the caller
     *  points.
     *
     *  @par Statuses:
     *  - @c usearch_status_unknown_kind_k A traversal other than a walk or a scan.
     *  - @c usearch_status_unsupported_k A scan, or a predicate on a VRAM index.
     */
    outcome<usearch_matches_tape_view_t> multiply(usearch_dense_multiply_t arguments, arena_t const& arena,
                                                  executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_multiply, arguments, &usearch_dense_multiply_t::products, executor);
    }

    /**
     *  @brief Attends the query heads @p arguments names over the entries their walks score,
     *      writing the rows into @p arena.
     *
     *  @par Statuses:
     *  - @c usearch_status_invalid_argument_k A @c NULL value lookup.
     *  - @c usearch_status_shape_mismatch_k The head count does not divide the rows, or a lookup
     *    yields a @c NULL row.
     *  - @c usearch_status_unknown_kind_k A traversal other than a walk or a scan.
     *  - @c usearch_status_unsupported_k A scan, or a predicate on a VRAM index.
     */
    outcome<usearch_rows_view_t> attend(usearch_dense_attend_t arguments, arena_t const& arena,
                                        executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_attend, arguments, &usearch_dense_attend_t::outputs, executor);
    }

    /**
     *  @brief Scores query groups by late interaction against the index, writing the matches into
     *      @p arena.
     *
     *  @par Statuses:
     *  - @c usearch_status_unknown_kind_k A traversal other than a walk or a scan.
     *  - @c usearch_status_shape_mismatch_k A query without rows, or decreasing query offsets.
     *  - @c usearch_status_unsupported_k A scan, a predicate, ragged query offsets or a zero
     *    @c wanted on a VRAM index.
     */
    outcome<usearch_matches_tape_view_t> maxsim(usearch_dense_maxsim_t arguments, arena_t const& arena,
                                                executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_maxsim, arguments, &usearch_dense_maxsim_t::matches, executor);
    }

#pragma endregion Reads

  private:
    template <typename options_type_>
    static outcome<dense_index_t> created_(void (*verb)(options_type_*), options_type_& options) noexcept {
        dense_index_t created;
        options.index = &created.handle_;
        outcome<> const reported = run_to_completion(verb, options);
        return {std::move(created), reported.status, reported.message};
    }

    template <typename arguments_type_>
    outcome<> ran_(void (*verb)(arguments_type_*), arguments_type_& arguments,
                   executor_t const& executor) const noexcept {
        arguments.index = handle_, arguments.executor = executor.handle();
        return run_to_completion(verb, arguments);
    }

    /** Runs @p verb with its @p output slot pointing at the value the outcome answers. */
    template <typename arguments_type_, typename value_type_>
    outcome<value_type_> answered_(void (*verb)(arguments_type_*), arguments_type_& arguments,
                                   value_type_* arguments_type_::* output, executor_t const& executor) const noexcept {
        value_type_ value {};
        arguments.*output = &value;
        outcome<> const reported = ran_(verb, arguments, executor);
        return {value, reported.status, reported.message};
    }

    usearch_dense_index_t handle_ = nullptr;
};

} // namespace unum::usearch
