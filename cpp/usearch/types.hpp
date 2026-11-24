/**
 *  @file cpp/usearch/types.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The pieces every C++ index wrapper shares: outcomes and deferred calls, element types,
 *      rows and tapes, handlers, executors, arenas, plans and cursors, adding no ABI of their own
 *      and throwing nothing.
 *
 *  Every index wrapper answers its verbs through a receiver whose type picks how they complete:
 *  a blocking receiver answers an @c outcome, a deferred one answers a @c deferred call its one
 *  owner polls or waits on. Handlers are @c noexcept callables read only during one call, so no
 *  fault ever unwinds through the library.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::uint32_t`, `std::uint64_t`, `std::int8_t`

#include <span>        // `std::span`
#include <type_traits> // `std::conditional_t`, `std::is_nothrow_invocable_r_v`, `std::is_void_v`
#include <utility>     // `std::exchange`, `std::move`, `std::swap`

#include "usearch/types.h"

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
 *      the arena, the handles, the inputs and the handlers it was handed stay allocated and
 *      unchanged until then.
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

/** A blocking call's outcome, or a deferred call the caller holds, as @p completion_ says. */
template <usearch_completion_t completion_, typename value_type_>
using answer =
    std::conditional_t<completion_ == usearch_completion_deferred_k, deferred<value_type_>, outcome<value_type_>>;

/** Runs @p submit with an outcome and an output slot: on the stack when blocking, inside the
 *  answered @c deferred otherwise. Every receiver's verbs answer through it. */
template <usearch_completion_t completion_, typename value_type_, typename submit_type_>
answer<completion_, value_type_> answered_(submit_type_ const& submit) noexcept {
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
template <usearch_completion_t completion_, typename value_type_>
answer<completion_, value_type_> refused_(usearch_status_t status, char const* message) noexcept {
    if constexpr (completion_ == usearch_completion_deferred_k) return deferred<value_type_>(status, message);
    else if constexpr (std::is_void_v<value_type_>) return {status, message};
    else return {value_type_ {}, status, message};
}

/** Runs a verb with no deferred form with @p arguments, which name everything but the outcome. */
template <typename arguments_type_>
outcome<> called_(void (*verb)(arguments_type_*), arguments_type_& arguments) noexcept {
    usearch_outcome_t reported {};
    arguments.outcome = &reported;
    verb(&arguments);
    return {reported.status, reported.message};
}

#pragma region Scalars Rows and Tapes

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

/** Rows of @p element_type_ the caller passes a call, along its batch axis. */
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

/** Entries of @p element_type_ the caller passes a call end to end, as an Apache Arrow tape: one
 *  more offset than entries, counted in the modality's own elements, bytes or points. */
template <element element_type_>
struct tape_view {

    /** The first element, which every offset counts from. */
    element_type_ const* first_element = nullptr;

    /** Entries along the batch axis, one fewer than the offsets. */
    std::size_t count = 0;

    /** Which member of @c offsets the tape spells them in. */
    usearch_tape_layout_t layout = usearch_tape_u64_k;

    /** The 32-bit offsets of an Arrow @c Binary or @c List tape. */
    std::uint32_t const* offsets_u32 = nullptr;

    /** The 64-bit offsets of an Arrow @c LargeBinary or @c LargeList tape. */
    std::uint64_t const* offsets_u64 = nullptr;

    tape_view() noexcept = default;

    /** Entries around 32-bit @p offsets, one more than entries. */
    tape_view(element_type_ const* first, std::span<std::uint32_t const> offsets) noexcept
        : first_element(first), count(offsets.empty() ? 0 : offsets.size() - 1), layout(usearch_tape_u32_k),
          offsets_u32(offsets.data()) {}

    /** Entries around 64-bit @p offsets, one more than entries. */
    tape_view(element_type_ const* first, std::span<std::uint64_t const> offsets) noexcept
        : first_element(first), count(offsets.empty() ? 0 : offsets.size() - 1), layout(usearch_tape_u64_k),
          offsets_u64(offsets.data()) {}

    usearch_tape_view_t view() const noexcept {
        usearch_tape_view_t tape {};
        tape.data = first_element, tape.layout = layout, tape.scalar = scalar_of<element_type_>;
        if (layout == usearch_tape_u32_k) tape.offsets.u32 = offsets_u32;
        else tape.offsets.u64 = offsets_u64;
        return tape;
    }
};

/** Entries a call wrote end to end into its arena as @p element_type_, @p lanes_ values per
 *  element its offsets count, readable until the arena's next call. */
template <element element_type_, std::size_t lanes_ = 1>
struct tape_entries {

    /** The tape as the call wrote it, with offsets as wide as @c std::size_t. */
    usearch_tape_view_t view {};

    /** Entries written. */
    std::size_t written = 0;

    /** The values of the entry at @p position, below @c written. */
    std::span<element_type_ const> entry(std::size_t position) const noexcept {
        std::size_t begin = 0, end = 0;
        if (view.layout == usearch_tape_u32_k) begin = view.offsets.u32[position], end = view.offsets.u32[position + 1];
        else begin = view.offsets.u64[position], end = view.offsets.u64[position + 1];
        element_type_ const* first = static_cast<element_type_ const*>(view.data);
        return {first + begin * lanes_, (end - begin) * lanes_};
    }
};

/** What one @c cursor_next pulled into the arena. */
struct cursor_batch_t {

    /** The keys of the entries pulled. */
    usearch_keys_view_t keys {};

    /** Each entry's linked keys, filled when the pull asked for links. */
    usearch_keys_tape_view_t links {};

    /** Entries pulled. */
    std::size_t written = 0;
};

#pragma endregion Scalars Rows and Tapes

#pragma region Handlers

/** A progress handler: hears units completed of a total and answers whether the call goes on. */
template <typename handler_type_>
concept progress_handler =
    std::is_nothrow_invocable_r_v<usearch_continuation_t, handler_type_&, std::size_t, std::size_t>;

/** A predicate handler: writes one admission per key it is asked about. */
template <typename handler_type_>
concept predicate_handler =
    std::is_nothrow_invocable_r_v<void, handler_type_&, usearch_key_t const*, usearch_admission_t*, std::size_t>;

/** A progress handler read only during one call as its own state; empty hears nothing. */
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

/** A predicate handler read only during one call as its own state; empty admits every key. */
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

#pragma endregion Handlers

#pragma region Executors and Arenas

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

/** Creates an @p owner_type_ through @p verb, whose @p options name everything but the outcome and
 *  the new handle. */
template <typename owner_type_, typename options_type_>
outcome<owner_type_> opened_(void (*verb)(options_type_*), options_type_& options) noexcept {
    owner_type_ opened;
    options.index = opened.handle_field();
    outcome<> const reported = called_(verb, options);
    return {std::move(opened), reported.status, reported.message};
}

/** Creates an @p owner_type_ through a verb that could defer, blocking, as the handle answers. */
template <typename owner_type_, typename options_type_>
outcome<owner_type_> loaded_(void (*verb)(options_type_*), options_type_& options) noexcept {
    options.completion = usearch_completion_blocking_k;
    return opened_<owner_type_>(verb, options);
}

#pragma endregion Executors and Arenas

#pragma region Plans and Cursors

/**
 *  @brief A long operation in flight behind @p handle_type_, which its verb opens on the first call
 *      naming it and continues on every later one, until the operation's finish consumes it.
 *
 *  Nothing finishes it on destruction: a dropped live plan leaks and keeps every handle it names
 *  closed to insertion and saving, as an unfinished plan does in C.
 */
template <typename handle_type_>
class plan {
  public:
    plan() noexcept = default;
    plan(plan&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)), phase_(other.phase_) {}
    plan& operator=(plan&& other) noexcept {
        return std::swap(handle_, other.handle_), std::swap(phase_, other.phase_), *this;
    }
    plan(plan const&) = delete;
    plan& operator=(plan const&) = delete;

    /** Where the last call naming the plan stopped; @c usearch_phase_name spells its phase. */
    usearch_phase_progress_t phase() const noexcept { return phase_; }

    handle_type_ handle() const noexcept { return handle_; }

    /** The lazy handle field a verb naming the plan opens, continues and leaves. */
    handle_type_* handle_field() noexcept { return &handle_; }

    /** The progress field a verb naming the plan writes where it stops. */
    usearch_phase_progress_t* phase_field() noexcept { return &phase_; }

  private:
    handle_type_ handle_ = nullptr;
    usearch_phase_progress_t phase_ {};
};

/** Owns one cursor behind @p handle_type_, which @p free_ releases, opened by an index's
 *  @c cursor_init. */
template <typename handle_type_, void (*free_)(handle_type_)>
class cursor {
  public:
    cursor() noexcept = default;
    cursor(cursor&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    cursor& operator=(cursor&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    cursor(cursor const&) = delete;
    cursor& operator=(cursor const&) = delete;
    ~cursor() noexcept { free_(handle_); }

    handle_type_ handle() const noexcept { return handle_; }

    /** The output field @c cursor_init writes the opened cursor into. */
    handle_type_* handle_field() noexcept { return &handle_; }

  private:
    handle_type_ handle_ = nullptr;
};

/** Opens a cursor through @p verb, whose @p arguments name everything but the outcome and the
 *  output field. */
template <typename cursor_type_, typename arguments_type_>
outcome<cursor_type_> cursor_opened_(void (*verb)(arguments_type_*), arguments_type_& arguments) noexcept {
    cursor_type_ opened;
    arguments.cursor = opened.handle_field();
    outcome<> const reported = called_(verb, arguments);
    return {std::move(opened), reported.status, reported.message};
}

/** Reads a checkpoint's frozen metadata through @p verb, blocking, as the metadata answers it. */
template <typename metadata_type_, typename arguments_type_>
outcome<metadata_type_> metadata_read_(void (*verb)(arguments_type_*), arguments_type_& arguments,
                                       usearch_executor_t executor) noexcept {
    metadata_type_ metadata {};
    arguments.executor = executor, arguments.metadata = &metadata;
    arguments.completion = usearch_completion_blocking_k;
    outcome<> const reported = called_(verb, arguments);
    return {metadata, reported.status, reported.message};
}

#pragma endregion Plans and Cursors

#pragma region Configs

/** The knobs of @c insert. */
struct insert_config_t {

    /** The breadth of the candidates the build walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** Hears the entries admitted, and cannot cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c refine and the @p plan_type_ it opens or continues. */
template <typename plan_type_>
struct refine_config {

    /** The walk breadth and the bytes a loaded NVMe handle holds per part; zero derives each. */
    usearch_refine_options_t refine {};

    /** Hears every barrier of every phase, and may cancel the round there. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;

    /** The plan a cancel leaves live; @c nullptr runs one round to its end. */
    plan_type_* plan = nullptr;
};

/** The knobs of a modality's @c reshard, whose handles recut nothing in place, and the
 *  @p plan_type_ it opens or continues. */
template <typename plan_type_>
struct reshard_config {

    /** The walk breadth and the bytes a loaded NVMe handle holds per part; zero derives each. */
    usearch_refine_options_t refine {};

    /** Picks the part roots of a split: equal seeds over an unchanged source split equally. */
    std::uint64_t seed = 0;

    /** Hears every barrier of the closing rounds, and may cancel them there, never a move. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;

    /** The plan a cancel leaves live; @c nullptr moves and runs to the end. */
    plan_type_* plan = nullptr;
};

/** The keys already shown and the keys to look around, which every @c sample call reads. */
struct sample_config_t {

    /** The keys already shown: never drawn again, and the paths reach toward them. */
    std::span<usearch_key_t const> shown {};

    /** Where the @c shown keys sit, one row of three @c float values per key. */
    rows_view<float> shown_coordinates {};

    /** The keys to look around; empty looks around the whole index. */
    std::span<usearch_key_t const> focus {};

    /** The draw: equal seeds, shown keys and focus over an unchanged index draw equally. */
    std::uint64_t seed = 0;

    /** The breadth of the candidates of the walks behind the draws; zero derives. */
    std::size_t expansion_factor = 0;

    /** Hears the draws written of those wanted. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

#pragma endregion Configs

#pragma region Samples

/** The draws a @c sample call wrote into its arena, readable until the arena's next call. */
struct sample_t {

    /** The keys drawn. */
    usearch_keys_view_t keys {};

    /** Which of its key's entries each draw is, in the order @c get writes them. */
    std::size_t const* entry_ordinals = nullptr;

    /** The one shown or earlier draw each draw hangs from and its distance, one column. */
    usearch_matches_view_t parents {};

    /** Where each draw is placed, @c written rows of three @c float values. */
    usearch_rows_view_t coordinates {};

    /** Draws written; fewer than wanted only once the region is exhausted. */
    std::size_t written = 0;

    /** The three coordinates of the draw at @p position, below @c written. */
    float const* coordinate(std::size_t position) const noexcept {
        return reinterpret_cast<float const*>(static_cast<char const*>(coordinates.data) +
                                              position * coordinates.stride);
    }
};

/** Fills a modality's sample arguments @p arguments_type_ from @p config, @p wanted and @p sample
 *  around @p index, so every modality's @c sample is one body. */
template <typename arguments_type_, typename index_type_>
arguments_type_ sample_arguments_(index_type_ index, usearch_executor_t executor, usearch_outcome_t* outcome,
                                  usearch_completion_t completion, std::size_t wanted, arena_t& arena,
                                  sample_config_t const& config, sample_t* sample) noexcept {
    arguments_type_ arguments {};
    arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
    arguments.completion = completion;
    arguments.shown_count = config.shown.size();
    arguments.shown = {config.shown.data(), sizeof(usearch_key_t)};
    arguments.shown_coordinates = config.shown_coordinates.view();
    arguments.focus_count = config.focus.size();
    arguments.focus = {config.focus.data(), sizeof(usearch_key_t)};
    arguments.keys = &sample->keys, arguments.entry_ordinals = &sample->entry_ordinals;
    arguments.parents = &sample->parents, arguments.coordinates = &sample->coordinates;
    arguments.written = &sample->written, arguments.arena = arena.handle();
    arguments.wanted = wanted, arguments.seed = config.seed;
    arguments.expansion_factor = config.expansion_factor;
    arguments.progress = config.progress.raw(), arguments.stats = config.stats;
    return arguments;
}

/** Reads a metric kind spelled @p name back through @p verb, as a modality's
 *  @c metric_from_name does. */
template <typename metric_type_>
outcome<metric_type_> metric_read_(usearch_status_t (*verb)(usearch_bytes_view_t, metric_type_*),
                                   usearch_bytes_view_t name) noexcept {
    metric_type_ metric {};
    usearch_status_t const status = verb(name, &metric);
    return {metric, status, status == usearch_status_success_k ? nullptr : "The name spells no metric kind"};
}

#pragma endregion Samples

} // namespace unum::usearch
