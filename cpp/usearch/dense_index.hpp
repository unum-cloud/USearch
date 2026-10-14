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

/** A call's first output beside its status and static message, decomposed as
 *  `auto [value, status, message]`; the value is meaningful only on success. */
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

    static outcome<executor_t> for_ram(usearch_executor_for_ram_t options = {}) noexcept {
        return spawned_(usearch_executor_for_ram, options);
    }
    static outcome<executor_t> on_pool(usearch_executor_on_pool_t options) noexcept {
        return spawned_(usearch_executor_on_pool, options);
    }
    static outcome<executor_t> for_vram(usearch_executor_for_vram_t options = {}) noexcept {
        return spawned_(usearch_executor_for_vram, options);
    }
    static outcome<executor_t> for_nvme(usearch_executor_for_nvme_t options = {}) noexcept {
        return spawned_(usearch_executor_for_nvme, options);
    }

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

/** Owns one arena; every call landing outputs in it rewinds it first. */
class arena_t {
  public:
    arena_t() noexcept = default;
    arena_t(arena_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    arena_t& operator=(arena_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    arena_t(arena_t const&) = delete;
    arena_t& operator=(arena_t const&) = delete;
    ~arena_t() noexcept { usearch_arena_free(handle_); }

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

/** Owns one cursor, opened by @ref dense_index_t::cursor_init. */
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

/** Owns one search stream, opened by @ref dense_index_t::search_stream_init. */
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

    static outcome<dense_index_t> init_ram(usearch_dense_init_ram_t options) noexcept {
        return created_(usearch_dense_init_ram, options);
    }
    static outcome<dense_index_t> init_vram(usearch_dense_init_vram_t options) noexcept {
        return created_(usearch_dense_init_vram, options);
    }
    static outcome<dense_index_t> load_ram(usearch_dense_load_ram_t options, executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_ram, options);
    }
    static outcome<dense_index_t> load_vram(usearch_dense_load_vram_t options,
                                            executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_vram, options);
    }
    static outcome<dense_index_t> load_ram_buffer(usearch_dense_load_ram_buffer_t options,
                                                  executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_ram_buffer, options);
    }
    static outcome<dense_index_t> load_vram_buffer(usearch_dense_load_vram_buffer_t options,
                                                   executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_load_vram_buffer, options);
    }
    static outcome<dense_index_t> view_ram(usearch_dense_view_ram_t options) noexcept {
        return created_(usearch_dense_view_ram, options);
    }
    static outcome<dense_index_t> view_nvme(usearch_dense_view_nvme_t options,
                                            executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return created_(usearch_dense_view_nvme, options);
    }

    outcome<> reserve(usearch_dense_reserve_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_reserve, arguments, executor);
    }
    outcome<> clear(usearch_dense_clear_t arguments = {}, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_clear, arguments, executor);
    }

    outcome<> insert(usearch_dense_insert_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_insert, arguments, executor);
    }

    /** Refines the graph, answering the entries whose edges changed. */
    outcome<std::size_t> refine(usearch_dense_refine_t arguments = {}, executor_t const& executor = {}) noexcept {
        return answered_(usearch_dense_refine, arguments, &usearch_dense_refine_t::changed, executor);
    }

    outcome<usearch_dense_reachability_t> health(usearch_dense_health_t arguments = {},
                                                 executor_t const& executor = {}) noexcept {
        return answered_(usearch_dense_health, arguments, &usearch_dense_health_t::reachability, executor);
    }

    /** Absorbs the donor indexes @p arguments names into this one, consuming them. */
    outcome<> absorb(usearch_dense_absorb_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_absorb, arguments, executor);
    }

#pragma endregion Lifecycle and Mutation

#pragma region Reads

    outcome<usearch_matches_view_t> search(usearch_dense_search_t arguments, arena_t const& arena,
                                           executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_search, arguments, &usearch_dense_search_t::matches, executor);
    }

    /** Answers how many entries @c arguments.key holds. */
    outcome<std::size_t> count(usearch_dense_count_t arguments, executor_t const& executor = {}) noexcept {
        return answered_(usearch_dense_count, arguments, &usearch_dense_count_t::entries, executor);
    }

    /** Answers the rows @c arguments.key holds; @c arguments.written counts them when the caller
     *  sets it. */
    outcome<usearch_rows_view_t> get(usearch_dense_get_t arguments, arena_t const& arena,
                                     executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_get, arguments, &usearch_dense_get_t::vectors, executor);
    }

    outcome<dense_cursor_t> cursor_init(usearch_dense_cursor_init_t arguments = {}) noexcept {
        dense_cursor_t opened;
        arguments.cursor = &opened.handle_, arguments.index = handle_;
        outcome<> const reported = run_to_completion(usearch_dense_cursor_init, arguments);
        return {std::move(opened), reported.status, reported.message};
    }

    /** Answers the next keys @p cursor walks to; the rows and their count land where the caller
     *  points. */
    outcome<usearch_keys_view_t> cursor_next(dense_cursor_t const& cursor, usearch_dense_cursor_next_t arguments,
                                             arena_t const& arena, executor_t const& executor = {}) noexcept {
        arguments.cursor = cursor.handle(), arguments.arena = arena.handle();
        return answered_(usearch_dense_cursor_next, arguments, &usearch_dense_cursor_next_t::keys, executor);
    }

    outcome<dense_search_stream_t> search_stream_init(usearch_dense_search_stream_init_t arguments) noexcept {
        dense_search_stream_t opened;
        arguments.stream = &opened.handle_, arguments.index = handle_;
        outcome<> const reported = run_to_completion(usearch_dense_search_stream_init, arguments);
        return {std::move(opened), reported.status, reported.message};
    }

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

    outcome<> save(usearch_dense_save_t arguments, executor_t const& executor = {}) noexcept {
        return ran_(usearch_dense_save, arguments, executor);
    }

    outcome<usearch_bytes_view_t> save_buffer(usearch_dense_save_buffer_t arguments, arena_t const& arena,
                                              executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_save_buffer, arguments, &usearch_dense_save_buffer_t::monolith, executor);
    }

    static outcome<usearch_dense_metadata_t> metadata_read(usearch_dense_metadata_read_t arguments,
                                                           executor_t const& executor = {}) noexcept {
        usearch_dense_metadata_t metadata {};
        arguments.executor = executor.handle(), arguments.metadata = &metadata;
        outcome<> const reported = run_to_completion(usearch_dense_metadata_read, arguments);
        return {metadata, reported.status, reported.message};
    }

    static outcome<usearch_dense_metadata_t> metadata_read_buffer(usearch_dense_metadata_read_buffer_t arguments,
                                                                  executor_t const& executor = {}) noexcept {
        usearch_dense_metadata_t metadata {};
        arguments.executor = executor.handle(), arguments.metadata = &metadata;
        outcome<> const reported = run_to_completion(usearch_dense_metadata_read_buffer, arguments);
        return {metadata, reported.status, reported.message};
    }

    /** Answers the products; their count and the width they needed land where the caller
     *  points. */
    outcome<usearch_matches_tape_view_t> multiply(usearch_dense_multiply_t arguments, arena_t const& arena,
                                                  executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_multiply, arguments, &usearch_dense_multiply_t::products, executor);
    }

    outcome<usearch_rows_view_t> attend(usearch_dense_attend_t arguments, arena_t const& arena,
                                        executor_t const& executor = {}) noexcept {
        arguments.arena = arena.handle();
        return answered_(usearch_dense_attend, arguments, &usearch_dense_attend_t::outputs, executor);
    }

    outcome<usearch_matches_view_t> maxsim(usearch_dense_maxsim_t arguments, arena_t const& arena,
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
