/**
 *  @file cpp/usearch/dense_index.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief RAII handles and typed verbs over the dynamically dispatched dense modality, adding no
 *      ABI of its own and throwing nothing.
 *
 *  An index answers its verbs through a receiver, whose type picks how they complete:
 *  @c dense_index_t::on blocks on an executor, or on the caller's own thread without one, and
 *  answers an @c outcome; @c dense_index_t::deferred submits to an executor and answers a
 *  @c deferred call its one owner polls or waits on. Each verb names the scalar of the matrix it
 *  passes, as in @c search<float>, and takes its handlers as @c noexcept callables borrowed for the
 *  call. Each verb answers the statuses its C function lists in @c usearch/dense_index.h.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [executor, executor_status, executor_message] = us::executor_t::for_ram();
 *  auto [index, index_status, index_message] = us::dense_index_t::init_ram({
 *      .metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k,
 *      .dimensions = 256});
 *  auto [arena, arena_status, arena_message] = us::arena_t::make({.executor = executor.handle()});
 *  auto [matches, status, message] = index.on(executor).search<float>({query, 1, 256}, 10, arena);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::int8_t`, `std::int32_t`

#include <span>        // `std::span`
#include <type_traits> // `std::conditional_t`, `std::is_nothrow_invocable_r_v`, `std::is_void_v`
#include <utility>     // `std::exchange`, `std::swap`

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

/** The output slot of a call that outputs nothing but its status. */
struct nothing_t {};

/**
 *  @brief A submitted call: its outcome and its output slot, owned until the call is final, while
 *      the arena, the handles, the inputs and the handlers it was handed stay borrowed until then.
 *
 *  It never moves, as the library writes into it, and its one owner polls @c status or blocks in
 *  @c wait; destruction waits the same way.
 */
template <typename value_type_ = void>
class [[nodiscard]] deferred {
    using slot_type_ = std::conditional_t<std::is_void_v<value_type_>, nothing_t, value_type_>;

  public:
    /** Runs @p submit with this call's outcome and output slot, where the library reports. */
    template <typename submit_type_>
    explicit deferred(submit_type_ const& submit) noexcept {
        submit(&outcome_, &slot_);
    }

    /** A call refused before it crossed, final from the start. */
    deferred(usearch_status_t status, char const* message) noexcept : outcome_ {status, message} {}

    deferred(deferred const&) = delete;
    deferred& operator=(deferred const&) = delete;
    ~deferred() noexcept { usearch_outcome_wait(&outcome_); }

    /** @c usearch_status_deferred_k until the call is final, then its final status. */
    usearch_status_t status() noexcept { return usearch_outcome_status(&outcome_); }

    /** Blocks until the call is final and answers what a blocking call would have. */
    outcome<value_type_> wait() noexcept {
        usearch_outcome_wait(&outcome_);
        if constexpr (std::is_void_v<value_type_>) return {outcome_.status, outcome_.message};
        else return {slot_, outcome_.status, outcome_.message};
    }

  private:
    usearch_outcome_t outcome_ {};
    slot_type_ slot_ {};
};

#pragma region Scalars and Rows

/** Bit containers for the formats C++ has no arithmetic type for, so a call can still name them. */
struct bf16_t {
    std::uint16_t encoding;
};
struct f16_t {
    std::uint16_t encoding;
};
struct e5m2_t {
    std::uint8_t encoding;
};
struct e4m3_t {
    std::uint8_t encoding;
};
struct e3m2_t {
    std::uint8_t encoding;
};
struct e2m3_t {
    std::uint8_t encoding;
};
struct i4x2_t {
    std::uint8_t encoding;
};
struct u4x2_t {
    std::uint8_t encoding;
};
struct u1x8_t {
    std::uint8_t encoding;
};

/** The scalar kind a C++ element type names, or @c usearch_scalar_unknown_k for none. */
template <typename element_type_>
inline constexpr usearch_scalar_t scalar_of = usearch_scalar_unknown_k;
template <>
inline constexpr usearch_scalar_t scalar_of<double> = usearch_scalar_f64_k;
template <>
inline constexpr usearch_scalar_t scalar_of<float> = usearch_scalar_f32_k;
template <>
inline constexpr usearch_scalar_t scalar_of<bf16_t> = usearch_scalar_bf16_k;
template <>
inline constexpr usearch_scalar_t scalar_of<f16_t> = usearch_scalar_f16_k;
template <>
inline constexpr usearch_scalar_t scalar_of<e5m2_t> = usearch_scalar_e5m2_k;
template <>
inline constexpr usearch_scalar_t scalar_of<e4m3_t> = usearch_scalar_e4m3_k;
template <>
inline constexpr usearch_scalar_t scalar_of<e3m2_t> = usearch_scalar_e3m2_k;
template <>
inline constexpr usearch_scalar_t scalar_of<e2m3_t> = usearch_scalar_e2m3_k;
template <>
inline constexpr usearch_scalar_t scalar_of<std::int8_t> = usearch_scalar_i8_k;
template <>
inline constexpr usearch_scalar_t scalar_of<i4x2_t> = usearch_scalar_i4x2_k;
template <>
inline constexpr usearch_scalar_t scalar_of<std::int32_t> = usearch_scalar_i32_k;
template <>
inline constexpr usearch_scalar_t scalar_of<std::uint8_t> = usearch_scalar_u8_k;
template <>
inline constexpr usearch_scalar_t scalar_of<u4x2_t> = usearch_scalar_u4x2_k;
template <>
inline constexpr usearch_scalar_t scalar_of<u1x8_t> = usearch_scalar_u1x8_k;

/** An element type a call can state as the scalar of the matrix it passes or asks for. */
template <typename element_type_>
concept element = scalar_of<element_type_> != usearch_scalar_unknown_k;

/** Values one element packs: eight for bit sets, two for nibbles, one otherwise. */
template <element element_type_>
inline constexpr std::size_t values_per_element = 1;
template <>
inline constexpr std::size_t values_per_element<i4x2_t> = 2;
template <>
inline constexpr std::size_t values_per_element<u4x2_t> = 2;
template <>
inline constexpr std::size_t values_per_element<u1x8_t> = 8;

/** Rows of @p element_type_ the caller lends a call, along its batch axis. */
template <element element_type_>
struct rows_view {

    /** The first row. */
    element_type_ const* first_row = nullptr;

    /** Rows along the batch axis. */
    std::size_t count = 0;

    /** Values per row, which sub-byte elements pack several of. */
    std::size_t columns = 0;

    /** Bytes between the starts of consecutive rows; zero repeats the first row. */
    std::size_t stride = 0;

    rows_view() noexcept = default;

    /** Rows packed back to back. */
    rows_view(element_type_ const* first, std::size_t rows, std::size_t values) noexcept
        : first_row(first), count(rows), columns(values),
          stride(values / values_per_element<element_type_> * sizeof(element_type_)) {}

    /** Rows @p bytes apart. */
    rows_view(element_type_ const* first, std::size_t rows, std::size_t values, std::size_t bytes) noexcept
        : first_row(first), count(rows), columns(values), stride(bytes) {}

    usearch_rows_view_t view() const noexcept { return {first_row, stride, columns, scalar_of<element_type_>}; }
};

/** Rows a call wrote packed into its arena as @p element_type_, readable until the arena's next
 *  call: the entries @c get found under a key, or the outputs of @c attend. */
template <element element_type_>
struct dense_vectors {

    /** The rows as the call wrote them. */
    usearch_rows_view_t view {};

    /** Rows written. */
    std::size_t written = 0;

    /** The row at @p position, below @c written. */
    element_type_ const* row(std::size_t position) const noexcept {
        return reinterpret_cast<element_type_ const*>(static_cast<std::uint8_t const*>(view.data) +
                                                      position * view.stride);
    }
};

/** The ragged products of @c multiply or the matches of @c maxsim, laid out in the arena. */
struct dense_products_t {

    /** Keys, scores and offsets, end to end. */
    usearch_matches_tape_view_t view {};

    /** Pairs written across every row or query. */
    std::size_t written = 0;

    /** Pairs scored before @c wanted cut any. */
    std::size_t required = 0;
};

/** What one @c cursor_next pulled into the arena. */
struct dense_cursor_batch_t {

    /** The keys of the entries pulled. */
    usearch_keys_view_t keys {};

    /** Each entry's linked keys, filled when the pull asked for links. */
    usearch_keys_tape_view_t links {};

    /** Entries pulled. */
    std::size_t written = 0;
};

#pragma endregion Scalars and Rows

#pragma region Handlers

/** A progress handler: hears units completed of a total and answers whether the call goes on. */
template <typename handler_type_>
concept progress_handler =
    std::is_nothrow_invocable_r_v<usearch_continuation_t, handler_type_&, std::size_t, std::size_t>;

/** A predicate handler: writes one admission per key it is asked about. */
template <typename handler_type_>
concept predicate_handler =
    std::is_nothrow_invocable_r_v<void, handler_type_&, usearch_key_t const*, usearch_admission_t*, std::size_t>;

/** A value lookup handler: points each key at one row of the columns and scalar asked for. */
template <typename handler_type_>
concept value_lookup_handler = std::is_nothrow_invocable_r_v<void, handler_type_&, usearch_key_t const*, std::size_t,
                                                             usearch_scalar_t, std::size_t, void const**>;

/** A progress handler borrowed for one call as its own state; empty hears nothing. */
class progress_t {
  public:
    progress_t() noexcept = default;

    /** Hears @p handler after each of @p steps slices of a batch verb; zero derives. */
    template <progress_handler handler_type_>
    progress_t(handler_type_& handler, std::size_t steps = 0) noexcept
        : raw_ {&trampoline_<handler_type_>, &handler, steps} {}

    usearch_progress_t raw() const noexcept { return raw_; }

  private:
    template <typename handler_type_>
    static usearch_continuation_t trampoline_(std::size_t completed, std::size_t total, void* state) noexcept {
        return (*static_cast<handler_type_*>(state))(completed, total);
    }

    usearch_progress_t raw_ {};
};

/** A predicate handler borrowed for one call as its own state; empty admits every key. */
class predicate_t {
  public:
    predicate_t() noexcept = default;

    /** Asks @p handler about candidates as @p cost says. */
    template <predicate_handler handler_type_>
    predicate_t(handler_type_& handler, usearch_predicate_cost_t cost = usearch_predicate_cost_cheap_k) noexcept
        : raw_ {&trampoline_<handler_type_>, &handler, cost} {}

    usearch_predicate_t raw() const noexcept { return raw_; }

  private:
    template <typename handler_type_>
    static void trampoline_(usearch_key_t const* keys, usearch_admission_t* admissions, std::size_t count,
                            void* state) noexcept {
        (*static_cast<handler_type_*>(state))(keys, admissions, count);
    }

    usearch_predicate_t raw_ {};
};

/** The value rows @c attend gathers: a host handler borrowed for the call, or a device function. */
class value_lookup_t {
  public:
    value_lookup_t() noexcept = default;

    /** Resolves keys on the host through @p handler, for a RAM handle. */
    template <value_lookup_handler handler_type_>
    value_lookup_t(handler_type_& handler) noexcept
        : function_(&trampoline_<handler_type_>), handler_state_(&handler) {}

    /** A device function and its device state for a VRAM handle, untyped as the host sees them. */
    static value_lookup_t device(usearch_value_lookup_t function, void* state) noexcept {
        value_lookup_t lookup;
        lookup.function_ = function, lookup.handler_state_ = state;
        return lookup;
    }

    usearch_value_lookup_t function() const noexcept { return function_; }
    void* state() const noexcept { return handler_state_; }

  private:
    template <typename handler_type_>
    static void trampoline_(void* state, usearch_key_t const* keys, std::size_t count, usearch_scalar_t scalar,
                            std::size_t columns, void const** rows) noexcept {
        (*static_cast<handler_type_*>(state))(keys, count, scalar, columns, rows);
    }

    usearch_value_lookup_t function_ = nullptr;
    void* handler_state_ = nullptr;
};

#pragma endregion Handlers

/** Runs a verb with no deferred form with @p arguments, which name everything but the outcome. */
template <typename arguments_type_>
outcome<> called_(void (*verb)(arguments_type_*), arguments_type_& arguments) noexcept {
    usearch_outcome_t reported {};
    arguments.outcome = &reported;
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

    /** Creates an executor of host threads for RAM-resident indexes, interleaved across the compute
     *  domains local to the memory domains @p options names. */
    static outcome<executor_t> for_ram(usearch_executor_for_ram_t options = {}) noexcept {
        return spawned_(usearch_executor_for_ram, options);
    }

    /**
     *  Wraps a borrowed ForkUnion pool, so several libraries in one process share its threads.
     */
    static outcome<executor_t> on_pool(usearch_executor_on_pool_t options) noexcept {
        return spawned_(usearch_executor_on_pool, options);
    }

    /** Opens one submission stream on every distinct device @p options names. */
    static outcome<executor_t> for_vram(usearch_executor_for_vram_t options = {}) noexcept {
        return spawned_(usearch_executor_for_vram, options);
    }

    /** Creates an executor of host threads for NVMe-resident indexes, one submission ring
     *  apiece, interleaved across the compute domains local to the devices the paths name. */
    static outcome<executor_t> for_nvme(usearch_executor_for_nvme_t options = {}) noexcept {
        return spawned_(usearch_executor_for_nvme, options);
    }

    /** Waits until no call holds the executor, then sets its threads' idle mood. */
    outcome<> drain(usearch_executor_drain_t arguments = {}) noexcept {
        arguments.executor = handle_;
        return called_(usearch_executor_drain, arguments);
    }

    usearch_executor_t handle() const noexcept { return handle_; }

  private:
    template <typename options_type_>
    static outcome<executor_t> spawned_(void (*verb)(options_type_*), options_type_& options) noexcept {
        executor_t spawned;
        options.executor = &spawned.handle_;
        outcome<> const reported = called_(verb, options);
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
        outcome<> const reported = called_(usearch_arena_init, options);
        return {std::move(made), reported.status, reported.message};
    }

    usearch_arena_t handle() const noexcept { return handle_; }

  private:
    usearch_arena_t handle_ = nullptr;
};

template <usearch_completion_t completion_>
class dense_index_on;

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

/**
 *  @brief A refine in flight, which @c refine opens on its first call and continues on every later
 *      one, until @c refine_plan_finish consumes it.
 *
 *  Nothing finishes it on destruction: a dropped live plan leaks and keeps its index closed to
 *  insertion and saving, as an unfinished plan does in C.
 */
class dense_refine_plan_t {
  public:
    dense_refine_plan_t() noexcept = default;
    dense_refine_plan_t(dense_refine_plan_t&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), phase_(other.phase_) {}
    dense_refine_plan_t& operator=(dense_refine_plan_t&& other) noexcept {
        return std::swap(handle_, other.handle_), std::swap(phase_, other.phase_), *this;
    }
    dense_refine_plan_t(dense_refine_plan_t const&) = delete;
    dense_refine_plan_t& operator=(dense_refine_plan_t const&) = delete;

    /** Where the last call naming the plan stopped; @c usearch_phase_name spells its phase. */
    usearch_phase_progress_t phase() const noexcept { return phase_; }

    usearch_dense_refine_plan_t handle() const noexcept { return handle_; }

  private:
    template <usearch_completion_t>
    friend class dense_index_on;
    usearch_dense_refine_plan_t handle_ = nullptr;
    usearch_phase_progress_t phase_ {};
};

/**
 *  @brief A reshard in flight, which @c reshard opens on its first call and continues on every
 *      later one, until @c reshard_plan_finish consumes it.
 *
 *  Nothing finishes it on destruction: a dropped live plan leaks and keeps every handle it names
 *  closed to insertion and saving, as an unfinished plan does in C.
 */
class dense_reshard_plan_t {
  public:
    dense_reshard_plan_t() noexcept = default;
    dense_reshard_plan_t(dense_reshard_plan_t&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), phase_(other.phase_) {}
    dense_reshard_plan_t& operator=(dense_reshard_plan_t&& other) noexcept {
        return std::swap(handle_, other.handle_), std::swap(phase_, other.phase_), *this;
    }
    dense_reshard_plan_t(dense_reshard_plan_t const&) = delete;
    dense_reshard_plan_t& operator=(dense_reshard_plan_t const&) = delete;

    /** Where the last call naming the plan stopped; @c usearch_phase_name spells its phase. */
    usearch_phase_progress_t phase() const noexcept { return phase_; }

    usearch_dense_reshard_plan_t handle() const noexcept { return handle_; }

  private:
    template <usearch_completion_t>
    friend class dense_index_on;
    usearch_dense_reshard_plan_t handle_ = nullptr;
    usearch_phase_progress_t phase_ {};
};

#pragma region Configs

/** The knobs of @c insert. */
struct dense_insert_config_t {

    /** The breadth of the candidates the build walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** Hears the entries admitted, and cannot cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c refine and the plan it opens or continues. */
struct dense_refine_config_t {

    /** The walk breadth and the bytes a loaded NVMe handle holds per part; zero derives each. */
    usearch_refine_options_t refine {};

    /** Hears every barrier of every phase, and may cancel the round there. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;

    /** The plan a cancel leaves live; @c nullptr runs one round to its end. */
    dense_refine_plan_t* plan = nullptr;
};

/** The placement @c reshard recuts toward, the knobs of its closing rounds and its plan. */
struct dense_reshard_config_t {

    /** The placement of a RAM handle recut in place; @c nullptr keeps it. */
    usearch_ram_options_t const* ram = nullptr;

    /** The placement of a VRAM handle recut in place; @c nullptr keeps it. */
    usearch_vram_options_t const* vram = nullptr;

    /** The placement of an NVMe handle recut in place; @c nullptr keeps it. */
    usearch_nvme_options_t const* nvme = nullptr;

    /** The walk breadth and the bytes a loaded NVMe handle holds per part; zero derives each. */
    usearch_refine_options_t refine {};

    /** Picks the part roots of a split: equal seeds over an unchanged source split equally. */
    std::uint64_t seed = 0;

    /** Hears every barrier of the closing rounds, and may cancel them there, never a move. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;

    /** The plan a cancel leaves live; @c nullptr moves and runs to the end. */
    dense_reshard_plan_t* plan = nullptr;
};

/** The knobs of @c search. */
struct dense_search_config_t {

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be reported; empty admits every key. */
    predicate_t predicate {};

    /** Matches of an earlier search to start from; @c nullptr starts where searches start. */
    usearch_matches_view_t const* hints = nullptr;

    /** Hears the queries answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c multiply. */
struct dense_multiply_config_t {

    /** Pairs kept per left row, the first ones its walk scored; zero keeps every pair. */
    std::size_t wanted = 0;

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be reported; empty admits every key. */
    predicate_t predicate {};

    /** Hears the rows answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c attend. */
struct dense_attend_config_t {

    /** Query heads sharing the index's key-value head, dividing the row count; zero means one. */
    std::size_t head_count = 0;

    /** One online-softmax state per query row, lent by the caller; @c nullptr skips it. */
    usearch_dense_softmax_state_t* carry = nullptr;

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** The logit scale; zero derives it from the frozen geometry. */
    float score_scale = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be attended; empty admits every key. */
    predicate_t predicate {};

    /** Hears the rows answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c maxsim. */
struct dense_maxsim_config_t {

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be reported; empty admits every key. */
    predicate_t predicate {};

    /** Matches of an earlier search to start from; @c nullptr starts where searches start. */
    usearch_matches_view_t const* hints = nullptr;

    /** Hears the queries answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

#pragma endregion Configs

/** A blocking call's outcome, or a deferred call the caller holds, as @p completion_ says. */
template <usearch_completion_t completion_, typename value_type_>
using answer =
    std::conditional_t<completion_ == usearch_completion_deferred_k, deferred<value_type_>, outcome<value_type_>>;

/**
 *  @brief The verbs of a dense index, completed as @p completion_ says: blocking answers an
 *      @c outcome, deferred answers a @c deferred call. Made by @c dense_index_t::on and
 *      @c dense_index_t::deferred.
 *
 *  Each verb is written once, filling its C struct with the result as its output slot. Every
 *  input, handler, plan and arena a deferred call names stays borrowed until it is final.
 */
template <usearch_completion_t completion_>
class dense_index_on {
  public:
#pragma region Mutation

    /** Grows the shards ahead of @p capacity entries in all. */
    answer<completion_, void> reserve(std::size_t capacity) const noexcept {
        return called_<void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_dense_reserve_t arguments {index_, executor_, outcome, completion_, capacity};
            usearch_dense_reserve(&arguments);
        });
    }

    /** Erases all entries, keeping the reserved capacity and frozen geometry. */
    answer<completion_, void> clear() const noexcept {
        return called_<void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_dense_clear_t arguments {index_, executor_, outcome, completion_};
            usearch_dense_clear(&arguments);
        });
    }

    /** Adds @p vectors under @p keys, one key per row, all or nothing per call. */
    template <element element_type_>
    answer<completion_, void> insert(std::span<usearch_key_t const> keys, rows_view<element_type_> vectors,
                                     dense_insert_config_t const& config = {}) const noexcept {
        // The call has one count, so only this side sees the keys and the rows disagree.
        if (keys.size() != vectors.count)
            return refused_<void>(usearch_status_shape_mismatch_k, "The key count differs from the row count");
        return called_<void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_dense_insert_t arguments {
                .index = index_,
                .executor = executor_,
                .outcome = outcome,
                .completion = completion_,
                .count = keys.size(),
                .keys = {keys.data(), sizeof(usearch_key_t)},
                .vectors = vectors.view(),
                .expansion_factor = config.expansion_factor,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_dense_insert(&arguments);
        });
    }

    /** Runs one refine round, or continues @c config.plan, answering the report: entries
     *  re-selected and changed, islands found and reconnected, neighborhoods, links added and
     *  dropped, the verdict @c usearch_refine_verdict_name spells, and both health checks. */
    answer<completion_, usearch_refine_report_t> refine(dense_refine_config_t const& config = {}) const noexcept {
        return called_<usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_dense_refine_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .report = report,
                    .refine = config.refine,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = config.plan ? &config.plan->handle_ : nullptr,
                    .phase = config.plan ? &config.plan->phase_ : nullptr,
                };
                usearch_dense_refine(&arguments);
            });
    }

    /** Closes and frees @p plan, opened on this index and executor, answering all it did. */
    answer<completion_, usearch_refine_report_t> refine_plan_finish(dense_refine_plan_t plan) const noexcept {
        return called_<usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_dense_refine_plan_finish_t arguments {index_,      executor_,    outcome,
                                                              completion_, plan.handle_, report};
                usearch_dense_refine_plan_finish(&arguments);
            });
    }

    /** Moves the entries of @p sources into @p targets with this receiver's executor, answering the
     *  report of the closing rounds; every handle named outlives the call and the plan's finish. */
    answer<completion_, usearch_refine_report_t> reshard(std::span<usearch_dense_index_t const> sources,
                                                         std::span<usearch_dense_index_t const> targets,
                                                         dense_reshard_config_t const& config = {}) const noexcept {
        return called_<usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_dense_reshard_t arguments {
                    .sources_count = sources.size(),
                    .sources = sources.data(),
                    .targets_count = targets.size(),
                    .targets = targets.data(),
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .ram = config.ram,
                    .vram = config.vram,
                    .nvme = config.nvme,
                    .report = report,
                    .refine = config.refine,
                    .seed = config.seed,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = config.plan ? &config.plan->handle_ : nullptr,
                    .phase = config.plan ? &config.plan->phase_ : nullptr,
                };
                usearch_dense_reshard(&arguments);
            });
    }

    /** Closes and frees @p plan, opened with this executor, answering what it did in all. */
    answer<completion_, usearch_refine_report_t> reshard_plan_finish(dense_reshard_plan_t plan) const noexcept {
        return called_<usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_dense_reshard_plan_finish_t arguments {executor_, outcome, completion_, plan.handle_, report};
                usearch_dense_reshard_plan_finish(&arguments);
            });
    }

    /** Writes the index to the checkpoint at @p path. */
    answer<completion_, void> save(usearch_bytes_view_t path, progress_t progress = {}) const noexcept {
        return called_<void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_dense_save_t arguments {index_, executor_, outcome, completion_, path, progress.raw()};
            usearch_dense_save(&arguments);
        });
    }

    /** Writes the index into one monolith in @p arena. */
    answer<completion_, usearch_bytes_view_t> save_buffer(arena_t& arena, progress_t progress = {}) const noexcept {
        return called_<usearch_bytes_view_t>([&](usearch_outcome_t* outcome, usearch_bytes_view_t* monolith) noexcept {
            usearch_dense_save_buffer_t arguments {index_,   executor_,      outcome,       completion_,
                                                   monolith, arena.handle(), progress.raw()};
            usearch_dense_save_buffer(&arguments);
        });
    }

#pragma endregion Mutation

#pragma region Reads

    /** Reports the frozen metadata and the current counters. */
    answer<completion_, usearch_dense_index_description_t> describe() const noexcept {
        return called_<usearch_dense_index_description_t>(
            [&](usearch_outcome_t* outcome, usearch_dense_index_description_t* description) noexcept {
                usearch_dense_describe_t arguments {index_, executor_, outcome, completion_, description};
                usearch_dense_describe(&arguments);
            });
    }

    /** Counts the entries stored under @p key. */
    answer<completion_, std::size_t> count(usearch_key_t key) const noexcept {
        return called_<std::size_t>([&](usearch_outcome_t* outcome, std::size_t* entries) noexcept {
            usearch_dense_count_t arguments {index_, executor_, outcome, completion_, key, entries};
            usearch_dense_count(&arguments);
        });
    }

    /** Reads every vector stored under @p key into @p arena as @p element_type_ rows. */
    template <element element_type_>
    answer<completion_, dense_vectors<element_type_>> get(usearch_key_t key, arena_t& arena) const noexcept {
        return called_<dense_vectors<element_type_>>(
            [&](usearch_outcome_t* outcome, dense_vectors<element_type_>* vectors) noexcept {
                usearch_dense_get_t arguments {index_,
                                               executor_,
                                               outcome,
                                               completion_,
                                               key,
                                               &vectors->view,
                                               &vectors->written,
                                               arena.handle(),
                                               scalar_of<element_type_>};
                usearch_dense_get(&arguments);
            });
    }

    /** Audits how much of the graph a search can reach. */
    answer<completion_, usearch_reachability_t> health() const noexcept {
        return called_<usearch_reachability_t>(
            [&](usearch_outcome_t* outcome, usearch_reachability_t* reachability) noexcept {
                usearch_dense_health_t arguments {index_, executor_, outcome, completion_, reachability};
                usearch_dense_health(&arguments);
            });
    }

    /** Finds the @p wanted nearest entries of every row of @p queries, written into @p arena. */
    template <element element_type_>
    answer<completion_, usearch_matches_view_t> search(rows_view<element_type_> queries, std::size_t wanted,
                                                       arena_t& arena,
                                                       dense_search_config_t const& config = {}) const noexcept {
        return called_<usearch_matches_view_t>(
            [&](usearch_outcome_t* outcome, usearch_matches_view_t* matches) noexcept {
                usearch_dense_search_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .count = queries.count,
                    .queries = queries.view(),
                    .matches = matches,
                    .arena = arena.handle(),
                    .wanted = wanted,
                    .expansion_factor = config.expansion_factor,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .hints = config.hints,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_dense_search(&arguments);
            });
    }

    /** Scores every row of @p left against the index, written into @p arena. */
    template <element element_type_>
    answer<completion_, dense_products_t> multiply(rows_view<element_type_> left, arena_t& arena,
                                                   dense_multiply_config_t const& config = {}) const noexcept {
        return called_<dense_products_t>([&](usearch_outcome_t* outcome, dense_products_t* products) noexcept {
            usearch_dense_multiply_t arguments {
                .index = index_,
                .executor = executor_,
                .outcome = outcome,
                .completion = completion_,
                .count = left.count,
                .left = left.view(),
                .products = &products->view,
                .written = &products->written,
                .required = &products->required,
                .arena = arena.handle(),
                .wanted = config.wanted,
                .expansion_factor = config.expansion_factor,
                .traversal = config.traversal,
                .predicate = config.predicate.raw(),
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_dense_multiply(&arguments);
        });
    }

    /** Attends every row of @p queries over the values @p value_lookup resolves, writing one
     *  @p output_type_ row per query into @p arena. */
    template <element output_type_, element element_type_>
    answer<completion_, dense_vectors<output_type_>> attend(rows_view<element_type_> queries,
                                                            value_lookup_t value_lookup, arena_t& arena,
                                                            dense_attend_config_t const& config = {}) const noexcept {
        return called_<dense_vectors<output_type_>>(
            [&](usearch_outcome_t* outcome, dense_vectors<output_type_>* outputs) noexcept {
                usearch_dense_attend_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .count = queries.count,
                    .queries = queries.view(),
                    .head_count = config.head_count,
                    .value_lookup = value_lookup.function(),
                    .value_lookup_state = value_lookup.state(),
                    .carry = config.carry,
                    .outputs = &outputs->view,
                    .arena = arena.handle(),
                    .outputs_scalar = scalar_of<output_type_>,
                    .expansion_factor = config.expansion_factor,
                    .score_scale = config.score_scale,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                outputs->written = queries.count;
                usearch_dense_attend(&arguments);
            });
    }

    /** Scores groups of @p rows_per_query rows of @p queries by late interaction, keeping the
     *  @p wanted best keys per query in @p arena. */
    template <element element_type_>
    answer<completion_, dense_products_t> maxsim(rows_view<element_type_> queries, std::size_t rows_per_query,
                                                 std::size_t wanted, arena_t& arena,
                                                 dense_maxsim_config_t const& config = {}) const noexcept {
        // The call counts queries, so only this side sees the rows fail to divide into them.
        if (rows_per_query == 0 || queries.count % rows_per_query != 0)
            return refused_<dense_products_t>(usearch_status_shape_mismatch_k,
                                              "The rows do not divide into whole queries");
        return called_<dense_products_t>([&](usearch_outcome_t* outcome, dense_products_t* matches) noexcept {
            usearch_dense_maxsim_t arguments {
                .index = index_,
                .executor = executor_,
                .outcome = outcome,
                .completion = completion_,
                .count = queries.count / rows_per_query,
                .queries = queries.view(),
                .rows_per_query = rows_per_query,
                .matches = &matches->view,
                .written = &matches->written,
                .required = &matches->required,
                .arena = arena.handle(),
                .wanted = wanted,
                .expansion_factor = config.expansion_factor,
                .traversal = config.traversal,
                .predicate = config.predicate.raw(),
                .hints = config.hints,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_dense_maxsim(&arguments);
        });
    }

    /** Pulls the next @p wanted entries @p cursor walks to into @p arena, with their links when
     *  @p links asks. */
    answer<completion_, dense_cursor_batch_t> cursor_next(dense_cursor_t const& cursor, std::size_t wanted,
                                                          arena_t& arena, bool links = false) const noexcept {
        return called_<dense_cursor_batch_t>([&](usearch_outcome_t* outcome, dense_cursor_batch_t* batch) noexcept {
            usearch_dense_cursor_next_t arguments {index_,
                                                   executor_,
                                                   outcome,
                                                   completion_,
                                                   cursor.handle(),
                                                   &batch->keys,
                                                   links ? &batch->links : nullptr,
                                                   &batch->written,
                                                   arena.handle(),
                                                   wanted};
            usearch_dense_cursor_next(&arguments);
        });
    }

    /** Answers the next @p wanted matches of @p stream, written into @p arena. */
    answer<completion_, usearch_matches_view_t> search_stream_next(dense_search_stream_t const& stream,
                                                                   std::size_t wanted, arena_t& arena) const noexcept {
        return called_<usearch_matches_view_t>(
            [&](usearch_outcome_t* outcome, usearch_matches_view_t* matches) noexcept {
                usearch_dense_search_stream_next_t arguments {index_,          executor_, outcome,        completion_,
                                                              stream.handle(), matches,   arena.handle(), wanted};
                usearch_dense_search_stream_next(&arguments);
            });
    }

#pragma endregion Reads

  private:
    friend class dense_index_t;

    dense_index_on(usearch_dense_index_t index, usearch_executor_t executor) noexcept
        : index_(index), executor_(executor) {}

    /** Runs @p submit with an outcome and an output slot: on the stack when blocking, inside the
     *  answered @c deferred otherwise. */
    template <typename value_type_, typename submit_type_>
    static answer<completion_, value_type_> called_(submit_type_ const& submit) noexcept {
        if constexpr (completion_ == usearch_completion_deferred_k) return deferred<value_type_>(submit);
        else {
            usearch_outcome_t reported {};
            if constexpr (std::is_void_v<value_type_>) {
                nothing_t nothing;
                submit(&reported, &nothing);
                return {reported.status, reported.message};
            }
            else {
                value_type_ value {};
                submit(&reported, &value);
                return {value, reported.status, reported.message};
            }
        }
    }

    /** Answers a call refused before it crossed. */
    template <typename value_type_>
    static answer<completion_, value_type_> refused_(usearch_status_t status, char const* message) noexcept {
        if constexpr (completion_ == usearch_completion_deferred_k) return deferred<value_type_>(status, message);
        else if constexpr (std::is_void_v<value_type_>) return {status, message};
        else return {value_type_ {}, status, message};
    }

    usearch_dense_index_t index_ = nullptr;
    usearch_executor_t executor_ = nullptr;
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

    /** The receiver whose verbs block on @p executor, or on the caller's own thread without one. */
    dense_index_on<usearch_completion_blocking_k> on(executor_t const& executor = {}) const noexcept {
        return {handle_, executor.handle()};
    }

    /** The receiver whose verbs submit to @p executor, each answering a @c deferred call. */
    dense_index_on<usearch_completion_deferred_k> deferred(executor_t const& executor) const noexcept {
        return {handle_, executor.handle()};
    }

#pragma region Lifecycle

    /** Creates an empty RAM index of the geometry @p options freezes. */
    static outcome<dense_index_t> init_ram(usearch_dense_init_ram_t options) noexcept {
        return created_(usearch_dense_init_ram, options);
    }

    /** Creates an empty VRAM index of the geometry @p options freezes, on the devices it names. */
    static outcome<dense_index_t> init_vram(usearch_dense_init_vram_t options) noexcept {
        return created_(usearch_dense_init_vram, options);
    }

    /** Copies a checkpoint into a new, mutable index in host memory. */
    static outcome<dense_index_t> load_ram(usearch_dense_load_ram_t options, executor_t const& executor = {},
                                           progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_(usearch_dense_load_ram, options);
    }

    /** Copies a checkpoint into a new, mutable index in device memory. */
    static outcome<dense_index_t> load_vram(usearch_dense_load_vram_t options, executor_t const& executor = {},
                                            progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_(usearch_dense_load_vram, options);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in host memory. */
    static outcome<dense_index_t> load_ram_buffer(usearch_dense_load_ram_buffer_t options,
                                                  executor_t const& executor = {}, progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_(usearch_dense_load_ram_buffer, options);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in device memory. */
    static outcome<dense_index_t> load_vram_buffer(usearch_dense_load_vram_buffer_t options,
                                                   executor_t const& executor = {}, progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_(usearch_dense_load_vram_buffer, options);
    }

    /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
    static outcome<dense_index_t> view_ram(usearch_dense_view_ram_t options) noexcept {
        return created_(usearch_dense_view_ram, options);
    }

    /** Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines. */
    static outcome<dense_index_t> view_nvme(usearch_dense_view_nvme_t options,
                                            executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return loaded_(usearch_dense_view_nvme, options);
    }

    /** Opens a checkpoint on NVMe drives as a handle that refines and reshards by replacing whole
     *  parts, reading it through @p executor 's engines. */
    static outcome<dense_index_t> load_nvme(usearch_dense_load_nvme_t options,
                                            executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return loaded_(usearch_dense_load_nvme, options);
    }

    /** Reads a checkpoint's geometry & metric without loading it. */
    static outcome<usearch_dense_metadata_t> metadata_read(usearch_dense_metadata_read_t arguments,
                                                           executor_t const& executor = {}) noexcept {
        usearch_dense_metadata_t metadata {};
        arguments.executor = executor.handle(), arguments.metadata = &metadata;
        arguments.completion = usearch_completion_blocking_k;
        outcome<> const reported = called_(usearch_dense_metadata_read, arguments);
        return {metadata, reported.status, reported.message};
    }

    /** Reads geometry & metric from a serialized checkpoint buffer. */
    static outcome<usearch_dense_metadata_t> metadata_read_buffer(usearch_dense_metadata_read_buffer_t arguments,
                                                                  executor_t const& executor = {}) noexcept {
        usearch_dense_metadata_t metadata {};
        arguments.executor = executor.handle(), arguments.metadata = &metadata;
        arguments.completion = usearch_completion_blocking_k;
        outcome<> const reported = called_(usearch_dense_metadata_read_buffer, arguments);
        return {metadata, reported.status, reported.message};
    }

    /** Opens a cursor over every entry, or over the entries @p starts names first. */
    outcome<dense_cursor_t> cursor_init(std::span<usearch_key_t const> starts = {}) const noexcept {
        dense_cursor_t opened;
        usearch_dense_cursor_init_t arguments {handle_, nullptr, starts.size(), starts.data(), &opened.handle_};
        outcome<> const reported = called_(usearch_dense_cursor_init, arguments);
        return {std::move(opened), reported.status, reported.message};
    }

    /** Opens a stream of matches around the one row of @p query that later calls extend. */
    template <element element_type_>
    outcome<dense_search_stream_t> search_stream_init(rows_view<element_type_> query,
                                                      std::size_t expansion_factor = 0) const noexcept {
        dense_search_stream_t opened;
        usearch_dense_search_stream_init_t arguments {&opened.handle_, handle_, nullptr, query.view(),
                                                      expansion_factor};
        outcome<> const reported = called_(usearch_dense_search_stream_init, arguments);
        return {std::move(opened), reported.status, reported.message};
    }

#pragma endregion Lifecycle

  private:
    template <typename options_type_>
    static outcome<dense_index_t> created_(void (*verb)(options_type_*), options_type_& options) noexcept {
        dense_index_t created;
        options.index = &created.handle_;
        outcome<> const reported = called_(verb, options);
        return {std::move(created), reported.status, reported.message};
    }

    /** Creates through a verb that could defer, blocking, as the handle is its answer. */
    template <typename options_type_>
    static outcome<dense_index_t> loaded_(void (*verb)(options_type_*), options_type_& options) noexcept {
        options.completion = usearch_completion_blocking_k;
        return created_(verb, options);
    }

    usearch_dense_index_t handle_ = nullptr;
};

} // namespace unum::usearch
