/**
 *  @file cpp/usearch/types.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The pieces every C++ index wrapper shares: statuses, expected values and deferred calls,
 *      element types, rows and tapes, handlers, executors, arenas, plans and owners, adding no ABI
 *      of their own and throwing nothing.
 *
 *  Every index wrapper answers its verbs through a receiver whose type picks how they complete: a
 *  blocking receiver answers a @c status_t or an @c expected, a deferred one answers a @c deferred
 *  call its one owner polls, waits on or awaits. Handlers are @c noexcept callables read only
 *  during one call, so no fault ever unwinds through the library.
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstdint> // `std::uint8_t`, `std::uint16_t`, `std::uint32_t`, `std::uint64_t`, `std::int8_t`, `std::int32_t`

#include <concepts>    // `std::invocable`
#include <coroutine>   // `std::coroutine_handle`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::conditional_t`, `std::is_void_v`, `std::is_nothrow_invocable_r_v`
#include <utility>     // `std::exchange`, `std::move`, `std::swap`

#include "usearch/types.h" // `usearch_outcome_t`, `usearch_outcome_status`, `usearch_outcome_wait`

namespace unum::usearch {

/** What went wrong, with one enumerator per status the library reports. */
enum class status_code_t : int {
    success_k = usearch_status_success_k,
    deferred_k = usearch_status_deferred_k,
    out_of_memory_k = usearch_status_out_of_memory_k,
    shape_mismatch_k = usearch_status_shape_mismatch_k,
    unsupported_layout_k = usearch_status_unsupported_layout_k,
    unknown_kind_k = usearch_status_unknown_kind_k,
    read_only_k = usearch_status_read_only_k,
    io_k = usearch_status_io_k,
    device_k = usearch_status_device_k,
    malformed_k = usearch_status_malformed_k,
    internal_k = usearch_status_internal_k,
    overflow_k = usearch_status_overflow_k,
    uninitialized_k = usearch_status_uninitialized_k,
    config_mismatch_k = usearch_status_config_mismatch_k,
    contended_k = usearch_status_contended_k,
    cancelled_k = usearch_status_cancelled_k,
    unsupported_k = usearch_status_unsupported_k,
    invalid_argument_k = usearch_status_invalid_argument_k,
};

#pragma region Shared Core

/** Why a call failed: a code recovery branches on, and an optional static note naming the broken
 *  invariant, step or resource, for logs and regression analysis. */
struct [[nodiscard]] status_t {
    status_code_t code = status_code_t::success_k;
    char const* note = nullptr;

    constexpr status_t() noexcept = default;
    constexpr status_t(status_code_t reason, char const* detail = nullptr) noexcept : code(reason), note(detail) {}
    constexpr bool succeeded() const noexcept { return code == status_code_t::success_k; }
    constexpr bool failed() const noexcept { return code != status_code_t::success_k; }
    friend constexpr bool operator==(status_t const& status, status_code_t code) noexcept {
        return status.code == code;
    }
};

static_assert(sizeof(status_t) == 2 * sizeof(void*) && std::is_trivially_copyable_v<status_t>,
              "a status is a code and a note pointer, returned in two registers");

/** A call's value beside its status, bound natively as <tt>auto [value, status]</tt>. */
template <typename value_type_>
struct [[nodiscard]] expected {
    value_type_ value {};
    status_t status {};
};

/** Where a call writes what @p value_type_ is made of. */
template <typename value_type_>
struct outputs {

    /** The answer, which the call writes in place. */
    value_type_ value {};

    value_type_ answer() const noexcept { return value; }
};

/** A call that outputs nothing but its status. */
template <>
struct outputs<void> {};

/** Runs @p poll with @p state later on its own loop, storing the two words and allocating
 *  nothing. */
template <typename scheduler_type_>
concept a_scheduler = requires(scheduler_type_& loop, void (*poll)(void*) noexcept, void* state) {
    loop.post(poll, state);
};

template <typename value_type_, a_scheduler scheduler_type_>
class awaiter;

/**
 *  @brief A submitted call: its outcome and its outputs, owned until the call is final, while the
 *      receiver, inputs, arena and handlers it was handed stay allocated and unchanged until then.
 *
 *  It never moves, as the library writes into it, and its one owner polls @c status, blocks in
 *  @c wait or awaits @c wait on its own loop; destruction waits, so nothing the call writes into
 *  goes away under it.
 */
template <typename value_type_ = void>
class [[nodiscard]] deferred {
    usearch_outcome_t outcome_ {};
    outputs<value_type_> outputs_ {};

  public:
    /** What @c wait answers: what the blocking call would have. */
    using answer_type = std::conditional_t<std::is_void_v<value_type_>, status_t, expected<value_type_>>;

    /** An empty slot, final from the start, which @c resubmit fills. */
    deferred() noexcept = default;

    /** Runs @p submit with this call's outcome and outputs, where the library reports. */
    template <typename submit_type_>
        requires std::invocable<submit_type_ const&, usearch_outcome_t*, outputs<value_type_>*>
    explicit deferred(submit_type_ const& submit) noexcept {
        submit(&outcome_, &outputs_);
    }

    /** Waits only while the slot's previous call still runs, then runs @p submit in its place. */
    template <typename submit_type_>
        requires std::invocable<submit_type_ const&, usearch_outcome_t*, outputs<value_type_>*>
    void resubmit(submit_type_ const& submit) noexcept {
        usearch_outcome_wait(&outcome_);
        outcome_ = {};
        outputs_ = {};
        submit(&outcome_, &outputs_);
    }

    /** A call refused before it crossed, final from the start. */
    explicit deferred(status_t status) noexcept : outcome_ {static_cast<usearch_status_t>(status.code), status.note} {}

    deferred(deferred const&) = delete;
    deferred& operator=(deferred const&) = delete;
    ~deferred() noexcept { usearch_outcome_wait(&outcome_); }

    /** @c status_code_t::deferred_k until the call is final, then its final status, polled without
     *  blocking. */
    status_code_t status() noexcept { return static_cast<status_code_t>(usearch_outcome_status(&outcome_)); }

    /** Blocks until the call is final and answers what a blocking call would have. */
    answer_type wait() noexcept {
        usearch_outcome_wait(&outcome_);
        status_t const status {static_cast<status_code_t>(outcome_.status), outcome_.message};
        if constexpr (std::is_void_v<value_type_>) return status;
        else if (status.failed()) return {{}, status};
        else return {outputs_.answer()};
    }

    /** Awaits the call on @p loop, which polls its status until it is final. */
    template <a_scheduler scheduler_type_>
    awaiter<value_type_, scheduler_type_> wait(scheduler_type_& loop) noexcept {
        return {*this, loop};
    }
};

/** A @c deferred being awaited, living in the awaiting coroutine's frame: each poll reads the
 *  status word, and one not yet final posts itself again. */
template <typename value_type_, a_scheduler scheduler_type_>
class awaiter {
    deferred<value_type_>& call_;
    scheduler_type_& loop_;
    std::coroutine_handle<> caller_ {};

    static void poll_(void* state) noexcept {
        awaiter& awaited = *static_cast<awaiter*>(state);
        if (awaited.await_ready()) return awaited.caller_.resume();
        awaited.loop_.post(&poll_, state);
    }

  public:
    awaiter(deferred<value_type_>& call, scheduler_type_& loop) noexcept : call_(call), loop_(loop) {}
    bool await_ready() noexcept { return call_.status() != status_code_t::deferred_k; }
    void await_suspend(std::coroutine_handle<> caller) noexcept {
        caller_ = caller;
        loop_.post(&poll_, this);
    }
    typename deferred<value_type_>::answer_type await_resume() noexcept { return call_.wait(); }
};

/** Whether @p answer_type_ is a call the caller holds rather than a final answer. */
template <typename answer_type_>
inline constexpr bool is_deferred = false;
template <typename value_type_>
inline constexpr bool is_deferred<deferred<value_type_>> = true;

/** What a final @p answer_type_ carries beside its status, @c void for a bare @c status_t. */
template <typename answer_type_>
struct answer_value {
    using type = void;
};
template <typename value_type_>
struct answer_value<expected<value_type_>> {
    using type = value_type_;
};
template <typename answer_type_>
using answer_value_t = typename answer_value<answer_type_>::type;

/** Answers @p answer_type_ from a verb's @p body: a deferred call passes C its own outcome and
 *  outputs; a blocking call writes its outputs in place, then takes the C outcome's two words once
 *  it is final. */
template <typename answer_type_, typename body_type_>
answer_type_ answered_(body_type_ const& body) noexcept {
    if constexpr (is_deferred<answer_type_>) return answer_type_(body);
    else {
        using value_type = answer_value_t<answer_type_>;
        usearch_outcome_t reported {};
        outputs<value_type> written {};
        body(&reported, &written);
        status_t const status {static_cast<status_code_t>(reported.status), reported.message};
        if constexpr (std::is_void_v<value_type>) return status;
        else if (status.failed()) return {{}, status};
        else return {written.answer()};
    }
}

/** Answers a call refused before it crossed, final from the start. */
template <typename answer_type_>
answer_type_ answered_(status_t refused) noexcept {
    if constexpr (is_deferred<answer_type_> || std::is_same_v<answer_type_, status_t>) return answer_type_(refused);
    else return {{}, refused};
}

/** Owns one handle of @p handle_type_, which @p free_ releases, moving it and never copying it. */
template <typename handle_type_, void (*free_)(handle_type_)>
class owner {
    handle_type_ handle_ = nullptr;

  public:
    owner() noexcept = default;
    owner(owner&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    owner& operator=(owner&& other) noexcept {
        std::swap(handle_, other.handle_);
        return *this;
    }
    owner(owner const&) = delete;
    owner& operator=(owner const&) = delete;
    ~owner() noexcept { free_(handle_); }

    handle_type_ handle() const noexcept { return handle_; }

    /** The output field a lifecycle verb writes the new handle into. */
    handle_type_* handle_field() noexcept { return &handle_; }
};

/** Creates an @p owner_type_ through @p verb, which writes the new handle into @p field of
 *  @p arguments. */
template <typename owner_type_, typename arguments_type_, typename handle_type_>
expected<owner_type_> opened_(void (*verb)(arguments_type_*), arguments_type_& arguments,
                              handle_type_* arguments_type_::* field) noexcept {
    expected<owner_type_> opened;
    arguments.*field = opened.value.handle_field();
    opened.status = answered_<status_t>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
        arguments.outcome = outcome;
        verb(&arguments);
    });
    return opened;
}

#pragma endregion Shared Core

/** A blocking call's answer, a bare status for void verbs, or a deferred call the caller holds. */
template <typename value_type_ = void, usearch_completion_t completion_ = usearch_completion_blocking_k>
using deferrable = std::conditional_t<completion_ == usearch_completion_deferred_k, deferred<value_type_>,
                                      std::conditional_t<std::is_void_v<value_type_>, status_t, expected<value_type_>>>;

/** A handle owned, moved and never copied, whose verbs block on the calling thread. */
template <typename handle_type_, void (*free_)(handle_type_)>
struct owning {
    static constexpr usearch_completion_t completion = usearch_completion_blocking_k;
    owner<handle_type_, free_> owner_;
    handle_type_ handle() const noexcept { return owner_.handle(); }
    usearch_executor_t executor() const noexcept { return nullptr; }
    handle_type_* handle_field() noexcept { return owner_.handle_field(); }
};

/** A handle its caller owns and the executor its verbs run on, both valid while this lives,
 *  completing as @p completion_ says. */
template <typename handle_type_, usearch_completion_t completion_>
struct caller_owned {
    static constexpr usearch_completion_t completion = completion_;
    handle_type_ handle_ = nullptr;
    usearch_executor_t executor_ = nullptr;
    handle_type_ handle() const noexcept { return handle_; }
    usearch_executor_t executor() const noexcept { return executor_; }
};

/** Only an owning receiver opens, loads and hands out cursors. */
template <typename receiver_type_>
concept a_handle_owner = requires(receiver_type_& receiver) { receiver.handle_field(); };

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
concept an_element = scalar_of<element_type_> != usearch_scalar_unknown_k;

/** Values one element packs: eight for bit sets, two for nibbles, one otherwise. */
template <an_element element_type_>
inline constexpr std::size_t values_per_element = 1;
template <>
inline constexpr std::size_t values_per_element<i4x2_t> = 2;
template <>
inline constexpr std::size_t values_per_element<u4x2_t> = 2;
template <>
inline constexpr std::size_t values_per_element<u1x8_t> = 8;

/** The bytes of @p text as a call takes a path, a name or a buffer, read only during the call. */
inline usearch_bytes_view_t bytes_view(std::string_view text) noexcept {
    return {reinterpret_cast<std::uint8_t const*>(text.data()), text.size()};
}

/** Rows of @p element_type_ the caller passes a call, along its batch axis. */
template <an_element element_type_>
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
template <an_element element_type_>
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
        usearch_tape_view_t tape {.data = first_element,
                                  .layout = layout,
                                  .offsets = {},
                                  .scalar = scalar_of<element_type_>};
        if (layout == usearch_tape_u32_k) tape.offsets.u32 = offsets_u32;
        else tape.offsets.u64 = offsets_u64;
        return tape;
    }
};

/** Entries a call wrote end to end into its arena as @p element_type_, @p lanes_ values per
 *  element its offsets count, readable until the arena's next call. */
template <an_element element_type_, std::size_t lanes_ = 1>
struct tape_entries {

    /** The tape as the call wrote it, with offsets as wide as @c std::size_t. */
    usearch_tape_view_t view {};

    /** Entries written. */
    std::size_t written = 0;

    /** The values of the entry at @p position, below @c written. */
    std::span<element_type_ const> entry(std::size_t position) const noexcept {
        std::size_t const begin = offset_(position), end = offset_(position + 1);
        element_type_ const* first = static_cast<element_type_ const*>(view.data);
        return {first + begin * lanes_, (end - begin) * lanes_};
    }

    /** Walks the entries in order, each read as @c entry reads it. */
    class iterator {
        tape_entries const* entries_ = nullptr;
        std::size_t position_ = 0;

      public:
        using value_type = std::span<element_type_ const>;
        using difference_type = std::ptrdiff_t;

        iterator() noexcept = default;
        iterator(tape_entries const& entries, std::size_t position) noexcept
            : entries_(&entries), position_(position) {}

        value_type operator*() const noexcept { return entries_->entry(position_); }
        iterator& operator++() noexcept {
            ++position_;
            return *this;
        }
        iterator operator++(int) noexcept {
            iterator const previous = *this;
            ++position_;
            return previous;
        }
        bool operator==(iterator const&) const noexcept = default;
    };

    iterator begin() const noexcept { return {*this, 0}; }
    iterator end() const noexcept { return {*this, written}; }

  private:
    std::size_t offset_(std::size_t index) const noexcept {
        if (view.layout == usearch_tape_u32_k) return view.offsets.u32[index];
        return view.offsets.u64[index];
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
concept a_progress_handler =
    std::is_nothrow_invocable_r_v<usearch_continuation_t, handler_type_&, std::size_t, std::size_t>;

/** A per-key predicate handler: answers whether one candidate key may be reported, through a const
 *  call operator, since every executor thread may ask it at once. */
template <typename handler_type_>
concept a_key_predicate_handler = std::is_nothrow_invocable_r_v<bool, handler_type_ const&, usearch_key_t>;

/** A batch predicate handler: writes one admission per key of a batch it is asked about. */
template <typename handler_type_>
concept a_keys_predicate_handler =
    std::is_nothrow_invocable_r_v<void, handler_type_ const&, std::span<usearch_key_t const>,
                                  std::span<usearch_admission_t>>;

/** A progress handler read only during one call as its own state; empty hears nothing. */
class progress_t {
  public:
    progress_t() noexcept = default;

    /** Hears @p handler after each of @p steps slices of a batch verb; zero derives. */
    template <a_progress_handler handler_type_>
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

    /** Asks @p handler one key at a time, which is cheap, as candidates are scored. */
    template <a_key_predicate_handler handler_type_>
    predicate_t(handler_type_& handler) noexcept
        : raw_ {&each_<handler_type_>, &handler, usearch_predicate_cost_cheap_k} {}

    /** Asks @p handler a batch at a time, once per round unless @p cost says it is cheap. */
    template <a_keys_predicate_handler handler_type_>
    predicate_t(handler_type_& handler, usearch_predicate_cost_t cost = usearch_predicate_cost_expensive_k) noexcept
        : raw_ {&batch_<handler_type_>, &handler, cost} {}

    usearch_predicate_t raw() const noexcept { return raw_; }

  private:
    template <typename handler_type_>
    static void each_(usearch_key_t const* keys, usearch_admission_t* admissions, std::size_t count,
                      void* state) noexcept {
        handler_type_ const& handler = *static_cast<handler_type_ const*>(state);
        for (std::size_t position = 0; position != count; ++position)
            admissions[position] = handler(keys[position]) ? usearch_admit_k : usearch_reject_k;
    }

    template <typename handler_type_>
    static void batch_(usearch_key_t const* keys, usearch_admission_t* admissions, std::size_t count,
                       void* state) noexcept {
        (*static_cast<handler_type_ const*>(state))(std::span<usearch_key_t const> {keys, count},
                                                    std::span<usearch_admission_t> {admissions, count});
    }

    usearch_predicate_t raw_ {};
};

#pragma endregion Handlers

#pragma region Executors and Arenas

/** Owns one executor; an empty one runs every call on the caller's own thread. */
class executor_t {
    owner<usearch_executor_t, usearch_executor_free> owner_;

  public:
    usearch_executor_t handle() const noexcept { return owner_.handle(); }

    /** The output field a constructor writes the new executor into. */
    usearch_executor_t* handle_field() noexcept { return owner_.handle_field(); }

    /** Creates an executor of host threads for RAM-resident indexes, interleaved across the compute
     *  domains local to the memory domains @p arguments names. */
    static expected<executor_t> for_ram(usearch_executor_for_ram_t arguments = {}) noexcept {
        return opened_<executor_t>(usearch_executor_for_ram, arguments, &usearch_executor_for_ram_t::executor);
    }

    /** Opens or adopts one non-blocking stream on every distinct device @p arguments names: CUDA
     *  and HIP streams die with the executor, Metal queues are retained and the executor releases
     *  its own reference, and caller @c streams stay the caller's when creation fails. */
    static expected<executor_t> for_vram(usearch_executor_for_vram_t arguments = {}) noexcept {
        return opened_<executor_t>(usearch_executor_for_vram, arguments, &usearch_executor_for_vram_t::executor);
    }

    /** Creates an executor of host threads for NVMe-resident indexes, one submission ring
     *  apiece, near the first memory domain @c ram names, else the one the devices sit on. */
    static expected<executor_t> for_nvme(usearch_executor_for_nvme_t arguments = {}) noexcept {
        return opened_<executor_t>(usearch_executor_for_nvme, arguments, &usearch_executor_for_nvme_t::executor);
    }

    /** Waits until no call holds the executor, then sets its threads' idle mood. */
    status_t drain(usearch_executor_drain_t arguments = {}) noexcept {
        arguments.executor = handle();
        return answered_<status_t>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            arguments.outcome = outcome;
            usearch_executor_drain(&arguments);
        });
    }
};

/** Creates an @p owner_type_ through @p verb, blocking on @p executor, or on the calling thread
 *  without one, as every open and load does. */
template <typename owner_type_, typename arguments_type_, typename handle_type_>
expected<owner_type_> opened_(void (*verb)(arguments_type_*), arguments_type_ arguments,
                              handle_type_* arguments_type_::* field, executor_t const& executor) noexcept {
    arguments.executor = executor.handle();
    arguments.completion = usearch_completion_blocking_k;
    return opened_<owner_type_>(verb, arguments, field);
}

/** Creates an @p owner_type_ through @p verb as the overload above does, reporting to @p progress
 *  as a load does. */
template <typename owner_type_, typename arguments_type_, typename handle_type_>
expected<owner_type_> opened_(void (*verb)(arguments_type_*), arguments_type_ arguments,
                              handle_type_* arguments_type_::* field, executor_t const& executor,
                              progress_t progress) noexcept {
    arguments.progress = progress.raw();
    return opened_<owner_type_>(verb, arguments, field, executor);
}

/** Owns one arena; every call writing outputs into it rewinds it first. */
class arena_t {
    owner<usearch_arena_t, usearch_arena_free> owner_;

  public:
    usearch_arena_t handle() const noexcept { return owner_.handle(); }

    /** The output field @c make writes the new arena into. */
    usearch_arena_t* handle_field() noexcept { return owner_.handle_field(); }

    /** Creates an empty arena where the executor @p arguments names computes. */
    static expected<arena_t> make(usearch_arena_init_t arguments = {}) noexcept {
        return opened_<arena_t>(usearch_arena_init, arguments, &usearch_arena_init_t::arena);
    }

    /** Reports where the arena's outputs live, which decides whether host code may read them. */
    expected<usearch_arena_description_t> describe() const noexcept {
        return answered_<expected<usearch_arena_description_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_arena_description_t>* written) noexcept {
                usearch_arena_describe_t arguments {.arena = owner_.handle(),
                                                    .outcome = outcome,
                                                    .description = &written->value};
                usearch_arena_describe(&arguments);
            });
    }
};

/** Fills @p devices with the GPUs this process can drive, NVIDIA first, then AMD, then Apple, so a
 *  device's position is the ordinal @c executor_t::for_vram names it by, answering how many there
 *  are, which may exceed what @p devices holds. */
inline expected<std::size_t> gpu_devices(std::span<usearch_gpu_device_t> devices) noexcept {
    return answered_<expected<std::size_t>>([&](usearch_outcome_t* outcome, outputs<std::size_t>* written) noexcept {
        usearch_gpu_devices_t arguments {.outcome = outcome,
                                         .devices = devices.data(),
                                         .devices_capacity = devices.size(),
                                         .devices_count = &written->value};
        usearch_gpu_devices(&arguments);
    });
}

/** The linked library's version, a release-candidate suffix included. */
inline std::string_view version_name() noexcept { return usearch_version_name(); }

#pragma endregion Executors and Arenas

#pragma region Plans and Configs

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
        std::swap(handle_, other.handle_);
        std::swap(phase_, other.phase_);
        return *this;
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

/** The handle field of @p live, or @c nullptr when a call names no plan. */
template <typename handle_type_>
handle_type_* handle_field_of(plan<handle_type_>* live) noexcept {
    return live ? live->handle_field() : nullptr;
}

/** The progress field of @p live, or @c nullptr when a call names no plan. */
template <typename handle_type_>
usearch_phase_progress_t* phase_field_of(plan<handle_type_>* live) noexcept {
    return live ? live->phase_field() : nullptr;
}

/** Reads a checkpoint's frozen metadata through @p verb, blocking, as the metadata answers it. */
template <typename metadata_type_, typename arguments_type_>
expected<metadata_type_> metadata_read_(void (*verb)(arguments_type_*), arguments_type_& arguments,
                                        usearch_executor_t executor) noexcept {
    return answered_<expected<metadata_type_>>(
        [&](usearch_outcome_t* outcome, outputs<metadata_type_>* written) noexcept {
            arguments.executor = executor;
            arguments.outcome = outcome;
            arguments.completion = usearch_completion_blocking_k;
            arguments.metadata = &written->value;
            verb(&arguments);
        });
}

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

#pragma endregion Plans and Configs

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
    arguments.index = index;
    arguments.executor = executor;
    arguments.outcome = outcome;
    arguments.completion = completion;
    arguments.shown_count = config.shown.size();
    arguments.shown = {config.shown.data(), sizeof(usearch_key_t)};
    arguments.shown_coordinates = config.shown_coordinates.view();
    arguments.focus_count = config.focus.size();
    arguments.focus = {config.focus.data(), sizeof(usearch_key_t)};
    arguments.keys = &sample->keys;
    arguments.entry_ordinals = &sample->entry_ordinals;
    arguments.parents = &sample->parents;
    arguments.coordinates = &sample->coordinates;
    arguments.written = &sample->written;
    arguments.arena = arena.handle();
    arguments.wanted = wanted;
    arguments.seed = config.seed;
    arguments.expansion_factor = config.expansion_factor;
    arguments.progress = config.progress.raw();
    arguments.stats = config.stats;
    return arguments;
}

/** The status a converter answered, with @p message beside a failure. */
inline status_t converted_(usearch_status_t status, char const* message) noexcept {
    if (status == usearch_status_success_k) return {};
    return {static_cast<status_code_t>(status), message};
}

/** Reads a metric kind spelled @p name back through @p verb, as a modality's
 *  @c metric_from_name does. */
template <typename metric_type_>
expected<metric_type_> metric_read_(usearch_status_t (*verb)(usearch_bytes_view_t, metric_type_*),
                                    usearch_bytes_view_t name) noexcept {
    metric_type_ metric {};
    status_t const status = converted_(verb(name, &metric), "The name spells no metric kind");
    return {metric, status};
}

#pragma endregion Samples

} // namespace unum::usearch
