/**
 *  @file src_stub/lib.hpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief Machinery every modality's stand-in shares - outcomes, arenas, scalars, files, and the
 *      append-only store that keeps the headers' thread-safety promises.
 *
 *  Nothing here is part of the C ABI: each @c <modality>.cpp owns its verbs and includes this
 *  for the parts that have nothing to do with which modality it answers. The one shared lock is
 *  the reason a stand-in can be brute-force and still honest about concurrency - a binding that
 *  shares a handle across threads is exercising real synchronization, not a comment promising
 *  some.
 *
 *  Everything here reaches for the plainest standard-library facility that answers: this is a
 *  throwaway validation layer, so a @c std::vector under a @c std::shared_mutex is the right
 *  shape and a hand-drawn primitive never is.
 *
 *  The project headers this pulls in are the sized-integer vocabulary, the blocking files, and the
 *  engine's own arena, so both libraries rewind and refuse aliasing inputs alike. None costs a
 *  submodule or a runtime.
 */
#ifndef UNUM_USEARCH_STUB_LIB_HPP
#define UNUM_USEARCH_STUB_LIB_HPP

#include <cmath>   // `std::ldexp`, `std::frexp`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint8_t`
#include <cstring> // `std::memcmp`

#include <algorithm>        // `std::copy`, `std::sort`, `std::stable_sort`, `std::min`
#include <array>            // `std::array`
#include <atomic>           // `std::atomic`, `std::atomic_ref`
#include <bit>              // `std::bit_cast`
#include <functional>       // `std::function`
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits`
#include <mutex>            // `std::unique_lock`
#include <new>              // `std::bad_alloc`
#include <optional>         // `std::optional`
#include <shared_mutex>     // `std::shared_mutex`, `std::shared_lock`
#include <span>             // `std::span`
#include <string>           // `std::string`
#include <string_view>      // `std::string_view` - a byte run read as text
#include <type_traits>      // `std::is_nothrow_move_constructible_v`, `std::is_pointer_v`, `std::remove_pointer_t`
#include <unordered_map>    // `std::unordered_map`, `std::unordered_multimap`
#include <variant>          // `std::variant`
#include <vector>           // `std::vector`

#include "usearch/usearch.h" // every argument structure a call copies

#include "usearch/utilities/blocking_files.hpp" // `seekable_file_t`, `mapped_file_t` - the checkpoints' handles
#include "usearch/utilities/ram_buffers.hpp"    // `growing_arena` - the engine's arena, `contains` included
#include "usearch/utilities/types.hpp"

namespace us = unum::usearch;

/** What @c usearch_arena_t owns: host memory that one call at a time rewinds and refills. */
struct usearch_arena_handle_t {
    us::utilities::growing_arena<> memory {us::utilities::global_allocator};

    /** The outcome of the call holding the arena, from its claim or deferred submission until just
     *  before its final status; null while free. Atomic, since a held call finishes on any thread. */
    std::atomic<usearch_outcome_t const*> holder {nullptr};
};

namespace unum {
namespace usearch {
namespace stub {
using namespace unum::usearch::utilities;

#pragma region Vocabulary and Outcomes

/**
 *  @brief The run read as text - a path or a spelling, which are bytes that happen to decode.
 *
 *  The one place the byte run is reinterpreted. Every verb that names a file or a closed-set
 *  member arrives here rather than casting at its own call site.
 */
inline std::string_view as_text(usearch_bytes_view_t bytes) {
    return {reinterpret_cast<char const*>(bytes.data), bytes.length};
}

/** Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
inline bool named(usearch_bytes_view_t name, char const* spelling) { return as_text(name) == spelling; }

/**
 *  @brief A C++ byte run lent to the C interface, which spells a byte @c uint8_t.
 *
 *  The mirror of @c as_text, and the only place the two byte vocabularies meet: @c std::byte is
 *  the language's, @c uint8_t is the ABI's, and they are distinct types however alike they are.
 */
inline usearch_bytes_view_t as_bytes(bytes_view_t bytes) {
    return {reinterpret_cast<std::uint8_t const*>(bytes.data()), bytes.size()};
}

/** The read-only twin, for the buffers a caller lends the engine to read. */
inline bytes_view_t as_readable_bytes(usearch_bytes_view_t bytes) {
    return {reinterpret_cast<byte_t const*>(bytes.data), bytes.length};
}

/** Marks @p outcome deferred - every verb's first act, so a waiter never reads a stale status. */
inline void propagate_deferred(usearch_outcome_t* outcome) {
    if (outcome) std::atomic_ref(outcome->status).store(usearch_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status with release ordering, waking
 *  every waiter. The message is null exactly on success. */
inline void propagate_error(usearch_status_t status, usearch_outcome_t* outcome, char const* message) {
    if (!outcome) return;
    outcome->message = status == usearch_status_success_k ? nullptr : message;
    std::atomic_ref final_status(outcome->status);
    final_status.store(status, std::memory_order_release);
    final_status.notify_all();
}

/**
 *  @brief A verb's outcome while it runs: deferred from construction, and published once at
 *      destruction with whatever its guards recorded - success when none did.
 *
 *  Every verb declares it first, so it unwinds last: each lock the verb took is released, each
 *  output written, and the arena it was given let go before the final status lets a waiter free
 *  or reuse what the call named.
 */
class deferred_outcome_t {
    usearch_outcome_t* published_;
    usearch_arena_t arena_;
    usearch_outcome_t recorded_ {usearch_status_success_k, nullptr};

  public:
    explicit deferred_outcome_t(usearch_outcome_t* published, usearch_arena_t arena = nullptr) noexcept
        : published_(published), arena_(arena) {
        propagate_deferred(published_);
    }
    ~deferred_outcome_t() noexcept {
        // Only the holder lets go, so a call refused as contended leaves the other call's hold alone.
        usearch_outcome_t const* holder = published_;
        if (arena_) arena_->holder.compare_exchange_strong(holder, nullptr, std::memory_order_release);
        propagate_error(recorded_.status, published_, recorded_.message);
    }
    deferred_outcome_t(deferred_outcome_t const&) = delete;
    deferred_outcome_t& operator=(deferred_outcome_t const&) = delete;

    /** The private slot every guard records into, so a verb passes this wherever an outcome goes. */
    operator usearch_outcome_t*() noexcept { return &recorded_; }
};

/** Reports @p status through @p outcome when @p wrong, and answers whether it did - the
 *  one shape every guard below takes, so a caller reads `if (reject_if(...)) return;`. */
inline bool reject_if(bool wrong, usearch_status_t status, char const* message, usearch_outcome_t* outcome) {
    if (!wrong) return false;
    propagate_error(status, outcome, message);
    return true;
}

/** Zeroes the telemetry, which every header calls a pure output fully overwritten. */
template <typename stats_>
void reset(stats_* stats) {
    if (stats) *stats = {};
}

#pragma endregion Vocabulary and Outcomes

#pragma region Runs

/** Refuses a key run that names no storage; a batch's identities are never optional. */
inline bool reject_keys(usearch_keys_view_t keys, usearch_outcome_t* outcome) {
    return reject_if(!keys.data, usearch_status_invalid_argument_k, "The key run must name storage", outcome);
}

/** The @p row -th key of a run, where a zero stride repeats the first. */
inline usearch_key_t key_at(usearch_keys_view_t keys, std::size_t row) {
    return *reinterpret_cast<usearch_key_t const*>(reinterpret_cast<std::byte const*>(keys.data) + row * keys.stride);
}

/** Refuses a tape without its storage or offsets, or whose offsets ever decrease. */
inline bool reject_tape(usearch_tape_view_t tape, std::size_t count, usearch_outcome_t* outcome) {
    if (reject_if(!tape.offsets || (!tape.data && count), usearch_status_invalid_argument_k,
                  "The tape must name storage", outcome))
        return true;
    for (std::size_t entry = 0; entry != count; ++entry)
        if (reject_if(tape.offsets[entry + 1] < tape.offsets[entry], usearch_status_malformed_k,
                      "The tape's offsets must not decrease", outcome))
            return true;
    return false;
}

/** Ticks the caller's handler when @p completed closes one of its steps, and answers whether to
 *  carry on; a null handler always does. Every loop here advances a unit at a time, so the unit
 *  that crosses into a new step is the one that reports. */
inline bool keep_going(usearch_progress_t progress, std::size_t completed, std::size_t total) {
    if (!progress.callback || !completed) return true;
    // The library's default when the caller leaves the steps zero, a tenth of the call apart.
    std::size_t const steps = progress.steps ? progress.steps : 10;
    bool const closes = completed >= total || completed * steps / total != (completed - 1) * steps / total;
    return !closes || progress.callback(completed, total, progress.state) == usearch_continue_k;
}

/** Reports a handler's cancellation once a loop stopped short, and answers whether it did. */
inline bool reject_cancelled(std::size_t completed, std::size_t total, usearch_outcome_t* outcome) {
    return reject_if(completed < total, usearch_status_cancelled_k, "The caller's handler stopped the work", outcome);
}

#pragma endregion Runs

#pragma region Executors and Arenas

/** The copied arguments of any verb that takes an executor; whatever they point to stays the caller's until final. */
using call_arguments_t = std::variant<
    usearch_dense_load_ram_t, usearch_dense_load_vram_t, usearch_dense_load_ram_buffer_t,
    usearch_dense_load_vram_buffer_t, usearch_dense_view_nvme_t, usearch_dense_reserve_t, usearch_dense_clear_t,
    usearch_dense_insert_t, usearch_dense_refine_t, usearch_dense_health_t, usearch_dense_search_t,
    usearch_dense_count_t, usearch_dense_get_t, usearch_dense_merge_t, usearch_dense_cursor_next_t,
    usearch_dense_search_stream_next_t, usearch_dense_save_t, usearch_dense_save_buffer_t,
    usearch_dense_metadata_read_t, usearch_dense_metadata_read_buffer_t, usearch_dense_describe_t,
    usearch_dense_multiply_t, usearch_dense_attend_t, usearch_dense_maxsim_t, usearch_sequence_load_ram_t,
    usearch_sequence_view_nvme_t, usearch_sequence_reserve_t, usearch_sequence_clear_t, usearch_sequence_insert_t,
    usearch_sequence_refine_t, usearch_sequence_search_t, usearch_sequence_count_t, usearch_sequence_get_t,
    usearch_sequence_merge_t, usearch_sequence_cursor_next_t, usearch_sequence_save_t, usearch_sequence_metadata_read_t,
    usearch_sequence_describe_t, usearch_shapes_load_ram_t, usearch_shapes_load_vram_t, usearch_shapes_view_nvme_t,
    usearch_shapes_reserve_t, usearch_shapes_clear_t, usearch_shapes_insert_t, usearch_shapes_refine_t,
    usearch_shapes_search_t, usearch_shapes_count_t, usearch_shapes_get_t, usearch_shapes_merge_t,
    usearch_shapes_cursor_next_t, usearch_shapes_save_t, usearch_shapes_metadata_read_t, usearch_shapes_describe_t,
    usearch_geospatial_load_ram_t, usearch_geospatial_load_vram_t, usearch_geospatial_reserve_t,
    usearch_geospatial_clear_t, usearch_geospatial_insert_t, usearch_geospatial_refine_t, usearch_geospatial_search_t,
    usearch_geospatial_within_t, usearch_geospatial_count_t, usearch_geospatial_get_t, usearch_geospatial_merge_t,
    usearch_geospatial_cursor_next_t, usearch_geospatial_save_t, usearch_geospatial_metadata_read_t,
    usearch_geospatial_describe_t, usearch_dense_sample_t, usearch_sequence_sample_t, usearch_shapes_sample_t,
    usearch_geospatial_sample_t>;

/** A stand-in's executor: no threads, so every call it is given runs on the thread of a caller -
 *  the one that submits it blocking, drains the executor, waits on the call, or frees a handle it uses. */
struct executor_t {

    /** Worker threads the caller asked for; honoured by remembering, never by spawning. */
    std::size_t threads = 1;

    /** Submissions beyond this many finish @c usearch_status_contended_k. */
    std::size_t deferred_limit = 64;
};

/** A call given to an executor, waiting to run or running on some caller's thread. */
struct call_t {

    /** The executor whose deferred slot the call holds; null for a blocking call, which takes none. */
    executor_t const* deferred_on;

    /** A copy of the arguments with a null executor, so running them goes straight to the verb's body. */
    call_arguments_t arguments;
    void (*verb)(void*);

    /** The call's outcome, its executor, and every index, cursor, stream and arena it reads or writes:
     *  the addresses a wait or a free looks for before it returns. */
    std::vector<void const*> handles;
};

/** Holds a deferred @p call until its executor drains, or finishes it contended once that executor
 *  holds @c deferred_limit calls that are not yet final, answering whether it was held. */
bool submit_deferred(call_t call, usearch_outcome_t* outcome);

/** Runs a blocking @p call on the calling thread, listed while it runs so a free of a handle it uses waits. */
void run_blocking(call_t call);

/** Runs every deferred call held for @p executor, oldest first, on the calling thread. */
void drain(executor_t const* executor);

/** Returns once no call uses @p handle, draining the executors that still hold one - what every wait
 *  and every free does first. */
void wait_for_calls_using(void const* handle);

/** Collects @c call_t::handles from a verb's @p arguments. An output slot - a pointer to a handle
 *  the call creates - holds nothing yet, so it is skipped. */
template <typename arguments_>
std::vector<void const*> handles_used_by(arguments_ const& arguments) {
    std::vector<void const*> handles {arguments.outcome, arguments.executor};
    auto const add = [&](auto handle) {
        if constexpr (!std::is_pointer_v<std::remove_pointer_t<decltype(handle)>>) handles.push_back(handle);
    };
    if constexpr (requires { arguments.index; }) add(arguments.index);
    if constexpr (requires { arguments.cursor; }) add(arguments.cursor);
    if constexpr (requires { arguments.stream; }) add(arguments.stream);
    if constexpr (requires { arguments.arena; }) add(arguments.arena);
    return handles;
}

/** Holds @p arena for the call whose outcome is @p call, answering whether that call holds it now:
 *  it was free, or already that call's. A null arena holds nothing, left for the verb to refuse. */
inline bool hold_arena(usearch_arena_t arena, usearch_outcome_t const* call) {
    if (!arena) return true;
    usearch_outcome_t const* holder = nullptr;
    return arena->holder.compare_exchange_strong(holder, call, std::memory_order_acquire) || holder == call;
}

/** Every verb's first act. A call without an executor answers false and runs the verb's body in
 *  place. A call with one is copied without it and answers true: a deferred copy holds its arena
 *  and is held for the executor, a blocking one runs right after the deferred calls submitted before it. */
template <auto verb_, typename arguments_>
bool through_executor(arguments_* arguments) {
    executor_t* const executor = arguments->executor;
    if (!executor) return false;
    arguments_ copy = *arguments;
    copy.executor = nullptr;
    call_t call {nullptr, copy, [](void* erased) { verb_(static_cast<arguments_*>(erased)); },
                 handles_used_by(*arguments)};
    if (arguments->completion != usearch_completion_deferred_k)
        return drain(executor), run_blocking(std::move(call)), true;
    call.deferred_on = executor;
    usearch_arena_t arena = nullptr;
    if constexpr (requires { arguments->arena; }) arena = arguments->arena;
    if (!hold_arena(arena, arguments->outcome))
        return propagate_error(usearch_status_contended_k, arguments->outcome, "The arena serves another call"), true;
    // A refused submission never runs, so nothing else would let go of the arena it took.
    usearch_outcome_t const* holder = arguments->outcome;
    if (!submit_deferred(std::move(call), arguments->outcome) && arena)
        arena->holder.compare_exchange_strong(holder, nullptr, std::memory_order_release);
    return true;
}

/** Claims @p arena for the call whose outcome is @p call and rewinds it, which frees the previous
 *  call's outputs, so any of @p inputs still inside it is refused. A null arena is refused, never
 *  created. The call's @c deferred_outcome_t lets go of it just before the final store. */
inline usearch_arena_handle_t* claim_arena(usearch_arena_t arena, usearch_outcome_t const* call,
                                           std::initializer_list<void const*> inputs, usearch_outcome_t* outcome) {
    if (reject_if(!arena, usearch_status_invalid_argument_k, "The arena must not be null", outcome)) return nullptr;
    if (reject_if(!hold_arena(arena, call), usearch_status_contended_k, "The arena serves another call", outcome))
        return nullptr;
    for (void const* input : inputs)
        if (reject_if(input && arena->memory.contains(input), usearch_status_invalid_argument_k,
                      "An input lies inside the arena it rewinds", outcome))
            return nullptr;
    arena->memory.clear();
    return arena;
}

/** Carves @p count values of @c value_type_ out of a claimed @p arena; null once @p outcome records why not. */
template <typename value_type_>
value_type_* carve(usearch_arena_handle_t& arena, std::size_t count, usearch_outcome_t* outcome) {
    std::byte* const landing = arena.memory.allocate(count * sizeof(value_type_), default_alignment_k);
    if (!landing) propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    return reinterpret_cast<value_type_*>(landing);
}

#pragma endregion Executors and Arenas

#pragma region Scalars

/** The stored width of one value, in bytes; zero names a kind the stand-ins reject. */
inline std::size_t scalar_bytes(usearch_scalar_t kind) {
    switch (kind) {
    case usearch_scalar_f64_k: return 8;
    case usearch_scalar_f32_k: return 4;
    case usearch_scalar_bf16_k: return 2;
    case usearch_scalar_f16_k: return 2;
    case usearch_scalar_e5m2_k: return 1;
    case usearch_scalar_e4m3_k: return 1;
    case usearch_scalar_e3m2_k: return 1;
    case usearch_scalar_e2m3_k: return 1;
    case usearch_scalar_i8_k: return 1;
    case usearch_scalar_i4x2_k: return 1;
    case usearch_scalar_i32_k: return 4;
    case usearch_scalar_u8_k: return 1;
    case usearch_scalar_u4x2_k: return 1;
    case usearch_scalar_u1x8_k: return 1;
    default: return 0;
    }
}

/** How many logical dimensions share one stored value, as the @c xN suffixes spell. */
inline std::size_t dimensions_per_value(usearch_scalar_t kind) {
    switch (kind) {
    case usearch_scalar_i4x2_k: return 2;
    case usearch_scalar_u4x2_k: return 2;
    case usearch_scalar_u1x8_k: return 8;
    default: return 1;
    }
}

/**
 *  @brief The bytes one row of @p dimensions occupies under @p kind, rounding a partial value up.
 *
 *  The one place packing enters the arithmetic. A bit-set row is a byte per eight dimensions, so
 *  multiplying a width by a scalar width - the shape every other kind takes - overstates it
 *  eightfold and walks a caller's rows apart.
 */
inline std::size_t row_extent(usearch_scalar_t kind, std::size_t dimensions) {
    return divide_round_up(dimensions, dimensions_per_value(kind)) * scalar_bytes(kind);
}

/** Rejects a kind that names no scalar, rather than reading it as @c f32 - a wrong
 *  row-byte computation would otherwise pass. */
inline bool reject_scalar(usearch_scalar_t kind, usearch_outcome_t* outcome) {
    return reject_if(!scalar_bytes(kind), usearch_status_unknown_kind_k, "The kind names no scalar", outcome);
}

inline double read_half(u16_t bits) {
    int const sign = (bits >> 15) & 1;
    int const exponent = (bits >> 10) & 0x1F;
    int const mantissa = bits & 0x3FF;
    double value;
    if (exponent == 0) { value = std::ldexp(static_cast<double>(mantissa), -24); }
    else if (exponent == 31) {
        value = mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    }
    else { value = std::ldexp(1024.0 + mantissa, exponent - 25); }
    return sign ? -value : value;
}

inline u16_t write_half(double value) {
    u32_t const bits = std::bit_cast<u32_t>(static_cast<float>(value));
    u32_t const sign = (bits >> 16) & 0x8000;
    i32_t const exponent = static_cast<i32_t>((bits >> 23) & 0xFF) - 127 + 15;
    u32_t const mantissa = (bits >> 13) & 0x3FF;
    if (exponent <= 0) return static_cast<u16_t>(sign);
    if (exponent >= 31) return static_cast<u16_t>(sign | 0x7C00);
    return static_cast<u16_t>(sign | (static_cast<u32_t>(exponent) << 10) | mantissa);
}

/** A brain float is the high half of a single, so widening is a shift. */
inline double read_brain(u16_t bits) { return std::bit_cast<float>(static_cast<u32_t>(bits) << 16); }

/** Narrows to the high half, rounding half to even and keeping a NaN a NaN. */
inline u16_t write_brain(double value) {
    u32_t const bits = std::bit_cast<u32_t>(static_cast<float>(value));
    bool const is_nan = (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu);
    if (is_nan) return static_cast<u16_t>((bits >> 16) | 0x0040u);
    // The tie-breaking increment carries into the exponent on its own, so an overflow
    // saturates to infinity without a branch.
    u32_t const rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<u16_t>(rounded >> 16);
}

/** Which of a mini-float's codes name something other than a number. */
enum class minifloat_specials_t {

    /** Every code is a number, as both six-bit kinds are. */
    none_k,

    /** The all-ones code of either sign is a NaN, and there are no infinities. */
    nan_only_k,

    /** A saturated exponent is an infinity, or a NaN once its mantissa is set. */
    inf_and_nan_k,
};

/** One mini-float's shape, which is all a codec needs beyond the byte itself. */
struct minifloat_t {

    /** Exponent bits, which fix both the bias and the dynamic range. */
    unsigned exponent_bits;

    /** Mantissa bits, which fix the step between neighbouring codes. */
    unsigned mantissa_bits;

    /** Which codes are not numbers, since only the eight-bit kinds reserve any. */
    minifloat_specials_t specials;
};

/** The shape of @p kind, or a zeroed one for a kind that is not a mini-float. */
inline minifloat_t minifloat_of(usearch_scalar_t kind) {
    switch (kind) {
    case usearch_scalar_e5m2_k: return {5, 2, minifloat_specials_t::inf_and_nan_k};
    case usearch_scalar_e4m3_k: return {4, 3, minifloat_specials_t::nan_only_k};
    case usearch_scalar_e3m2_k: return {3, 2, minifloat_specials_t::none_k};
    case usearch_scalar_e2m3_k: return {2, 3, minifloat_specials_t::none_k};
    default: return {0, 0, minifloat_specials_t::none_k};
    }
}

/**
 *  @brief Widens one mini-float code, ignoring the padding bits a six-bit kind leaves above it.
 *
 *  One decoder serves all four kinds: they differ only in what @p format names.
 */
inline double read_minifloat(u8_t bits, minifloat_t format) {
    unsigned const width = 1 + format.exponent_bits + format.mantissa_bits;
    unsigned const mantissa_span = 1u << format.mantissa_bits;
    unsigned const saturated = (1u << format.exponent_bits) - 1;
    unsigned const code = bits & ((1u << width) - 1);
    unsigned const mantissa = code & (mantissa_span - 1);
    unsigned const exponent = (code >> format.mantissa_bits) & saturated;
    int const bias = (1 << (format.exponent_bits - 1)) - 1;
    double const sign = code >> (width - 1) ? -1.0 : 1.0;
    if (exponent == saturated && format.specials == minifloat_specials_t::inf_and_nan_k)
        return mantissa ? std::numeric_limits<double>::quiet_NaN() : sign * std::numeric_limits<double>::infinity();
    if (exponent == saturated && mantissa == mantissa_span - 1 && format.specials == minifloat_specials_t::nan_only_k)
        return std::numeric_limits<double>::quiet_NaN();
    // A zero exponent field is the subnormal step, so the arms differ only in the implied leading bit.
    unsigned const significand = exponent ? mantissa + mantissa_span : mantissa;
    int const scale = static_cast<int>(exponent ? exponent : 1) - bias - static_cast<int>(format.mantissa_bits);
    return sign * std::ldexp(static_cast<double>(significand), scale);
}

/** Narrows to the nearest code, ties to even, saturating rather than wrapping. */
inline u8_t write_minifloat(double value, minifloat_t format) {
    unsigned const width = 1 + format.exponent_bits + format.mantissa_bits;
    unsigned const mantissa_span = 1u << format.mantissa_bits;
    unsigned const saturated = (1u << format.exponent_bits) - 1;
    int const bias = (1 << (format.exponent_bits - 1)) - 1;
    unsigned const sign = std::signbit(value) ? 1u << (width - 1) : 0u;
    unsigned const reserved = (saturated << format.mantissa_bits) | (mantissa_span - 1);
    unsigned const largest = format.specials == minifloat_specials_t::inf_and_nan_k
                                 ? ((saturated - 1) << format.mantissa_bits) | (mantissa_span - 1)
                             : format.specials == minifloat_specials_t::nan_only_k ? reserved - 1
                                                                                   : reserved;
    // A kind reserving no code cannot carry a NaN, so it lands on a zero.
    if (std::isnan(value))
        return static_cast<u8_t>(format.specials == minifloat_specials_t::none_k ? 0u : sign | reserved);
    if (std::isinf(value) && format.specials == minifloat_specials_t::inf_and_nan_k)
        return static_cast<u8_t>(sign | (saturated << format.mantissa_bits));
    if (std::isinf(value)) return static_cast<u8_t>(sign | largest);

    double const magnitude = std::fabs(value);
    // A zero has no leading bit for `frexp` to find, so its code is the sign and nothing else.
    if (magnitude == 0) return static_cast<u8_t>(sign);
    int leading = 0;
    // `frexp` answers one above the leading bit's exponent; below the smallest normal the step is fixed.
    std::frexp(magnitude, &leading);
    int const exponent = std::max(leading - 1, 1 - bias);
    double const scaled = std::ldexp(magnitude, static_cast<int>(format.mantissa_bits) - exponent);
    double const truncated = std::floor(scaled);
    double const remainder = scaled - truncated;
    unsigned const carry = remainder > 0.5 || (remainder == 0.5 && std::fmod(truncated, 2.0) != 0.0) ? 1u : 0u;
    // The significand carries into the exponent on its own, so neither overflow case needs an arm.
    unsigned const significand = static_cast<unsigned>(truncated) + carry;
    unsigned const code = ((static_cast<unsigned>(exponent + bias) - 1) << format.mantissa_bits) + significand;
    return static_cast<u8_t>(sign | std::min(code, largest));
}

/** Widens a bit-set row, eight dimensions a byte from the high bit down, as NumKong packs them. */
inline void widen_bits(bytes_view_t row, std::span<double> values) {
    u8_t const* packed = static_cast<u8_t const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position)
        values[position] = (packed[position / 8] >> (7 - position % 8)) & 1 ? 1.0 : 0.0;
}

/** Narrows to a bit-set row, where every value that is not a zero sets its bit. */
inline void narrow_bits(bytes_span_t row, std::span<double const> values) {
    u8_t* packed = static_cast<u8_t*>(static_cast<void*>(row.data()));
    // The row's own extent is the packed byte count every caller sized through `row_extent`, so
    // the partial byte a ragged width leaves is cleared without rounding the count up again here.
    std::fill_n(packed, row.size(), u8_t {0});
    for (std::size_t position = 0; position != values.size(); ++position)
        if (values[position] != 0) packed[position / 8] |= static_cast<u8_t>(1u << (7 - position % 8));
}

/** Widens a nibble row, even dimensions in the high half of each byte, as NumKong packs them. */
template <typename nibble_>
void widen_nibbles(bytes_view_t row, std::span<double> values) {
    u8_t const* packed = static_cast<u8_t const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position) {
        int const nibble = (packed[position / 2] >> ((1 - position % 2) * 4)) & 0xF;
        // A signed nibble carries its sign in the high bit, so the upper half of the range wraps.
        values[position] = std::is_signed_v<nibble_> && nibble >= 8 ? nibble - 16 : nibble;
    }
}

/** Narrows into a nibble row, clamping to the four bits each dimension is given. */
template <typename nibble_>
void narrow_nibbles(bytes_span_t row, std::span<double const> values) {
    double const floor_value = std::is_signed_v<nibble_> ? -8.0 : 0.0;
    double const ceiling_value = std::is_signed_v<nibble_> ? 7.0 : 15.0;
    u8_t* packed = static_cast<u8_t*>(static_cast<void*>(row.data()));
    std::fill_n(packed, row.size(), u8_t {0});
    for (std::size_t position = 0; position != values.size(); ++position) {
        int const clamped = static_cast<int>(std::clamp(values[position], floor_value, ceiling_value));
        packed[position / 2] |= static_cast<u8_t>((clamped & 0xF) << ((1 - position % 2) * 4));
    }
}

/** Widens a whole row stored as @c scalar_, converting each element to a double. */
template <typename scalar_>
void widen_into(bytes_view_t row, std::span<double> values) {
    scalar_ const* typed = static_cast<scalar_ const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position) values[position] = typed[position];
}

/** Narrows a whole row of doubles back into storage as @c scalar_. */
template <typename scalar_>
void narrow_from(bytes_span_t row, std::span<double const> values) {
    scalar_* typed = static_cast<scalar_*>(static_cast<void*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position)
        typed[position] = static_cast<scalar_>(values[position]);
}

/** Widens one stored row into @p values, whose extent decides how much is read. The kind is
 *  fixed for the whole row, so it is switched on once here rather than once per element.
 *  Callers gate on @c scalar_bytes first, so an unreachable arm leaves @p values untouched
 *  rather than silently reinterpreting the row as another width. */
inline void read_row(bytes_view_t row, usearch_scalar_t kind, std::span<double> values) {
    switch (kind) {
    case usearch_scalar_f64_k: return widen_into<double>(row, values);
    case usearch_scalar_f32_k: return widen_into<float>(row, values);
    case usearch_scalar_i8_k: return widen_into<i8_t>(row, values);
    case usearch_scalar_i32_k: return widen_into<i32_t>(row, values);
    case usearch_scalar_u8_k: return widen_into<u8_t>(row, values);
    case usearch_scalar_i4x2_k: return widen_nibbles<i8_t>(row, values);
    case usearch_scalar_u4x2_k: return widen_nibbles<u8_t>(row, values);
    case usearch_scalar_u1x8_k: return widen_bits(row, values);
    case usearch_scalar_e5m2_k:
    case usearch_scalar_e4m3_k:
    case usearch_scalar_e3m2_k:
    case usearch_scalar_e2m3_k: {
        // The mini-floats decode from one byte apiece, their shape read once for the whole row.
        minifloat_t const format = minifloat_of(kind);
        u8_t const* typed = static_cast<u8_t const*>(static_cast<void const*>(row.data()));
        for (std::size_t position = 0; position != values.size(); ++position)
            values[position] = read_minifloat(typed[position], format);
        return;
    }
    case usearch_scalar_bf16_k:
    case usearch_scalar_f16_k: break;
    default: return;
    }
    // The half kinds decode rather than load, so they convert element by element.
    u16_t const* typed = static_cast<u16_t const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position)
        values[position] = kind == usearch_scalar_bf16_k ? read_brain(typed[position]) : read_half(typed[position]);
}

/** Quantizes @p values back into one stored row, switching on the kind once. */
inline void write_row(bytes_span_t row, usearch_scalar_t kind, std::span<double const> values) {
    switch (kind) {
    case usearch_scalar_f64_k: return narrow_from<double>(row, values);
    case usearch_scalar_f32_k: return narrow_from<float>(row, values);
    case usearch_scalar_i8_k: return narrow_from<i8_t>(row, values);
    case usearch_scalar_i32_k: return narrow_from<i32_t>(row, values);
    case usearch_scalar_u8_k: return narrow_from<u8_t>(row, values);
    case usearch_scalar_i4x2_k: return narrow_nibbles<i8_t>(row, values);
    case usearch_scalar_u4x2_k: return narrow_nibbles<u8_t>(row, values);
    case usearch_scalar_u1x8_k: return narrow_bits(row, values);
    case usearch_scalar_e5m2_k:
    case usearch_scalar_e4m3_k:
    case usearch_scalar_e3m2_k:
    case usearch_scalar_e2m3_k: {
        minifloat_t const format = minifloat_of(kind);
        u8_t* typed = static_cast<u8_t*>(static_cast<void*>(row.data()));
        for (std::size_t position = 0; position != values.size(); ++position)
            typed[position] = write_minifloat(values[position], format);
        return;
    }
    case usearch_scalar_bf16_k:
    case usearch_scalar_f16_k: break;
    default: return;
    }
    u16_t* typed = static_cast<u16_t*>(static_cast<void*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position)
        typed[position] = kind == usearch_scalar_bf16_k ? write_brain(values[position]) : write_half(values[position]);
}

/** The @p row -th row of a strided matrix, as @p count elements of @c value_type_. */
template <typename value_type_>
std::span<value_type_> row_at(void* base, std::size_t row, std::size_t stride, std::size_t count) {
    std::byte* const at = static_cast<std::byte*>(base) + row * stride;
    return std::span<value_type_>(static_cast<value_type_*>(static_cast<void*>(at)), count);
}

/** The read-only sibling, for the matrices the ABI hands over as pure inputs. */
template <typename value_type_>
std::span<value_type_ const> row_at(void const* base, std::size_t row, std::size_t stride, std::size_t count) {
    std::byte const* const at = static_cast<std::byte const*>(base) + row * stride;
    return std::span<value_type_ const>(static_cast<value_type_ const*>(static_cast<void const*>(at)), count);
}

#pragma endregion Scalars

#pragma region Storage

/** One stored row: its key, and whatever the modality calls a payload. */
template <typename payload_>
struct entry {
    usearch_key_t key;
    payload_ payload;
};

/** The project's own key type, which already knows how to hash and compare itself.
 *  Layout-identical to @c usearch_key_t, so the conversion is a reinterpretation. */
inline uuid_t keyed(usearch_key_t const& key) noexcept { return std::bit_cast<uuid_t>(key); }

/**
 *  @brief One handle's whole state. Readers hold @c mutex shared and writers exclusively, which
 *      is what keeps a search overlapping an insertion.
 *
 *  Reallocating @c entries moves each row without copying the buffer it owns - true only while the
 *  payload's move constructor stays @c noexcept, which the modality files assert.
 */
template <typename payload_, typename options_>
struct handle {

    /** The options the handle was initialized with, or a checkpoint later adopted. */
    options_ options {};

    /** The rows themselves, in insertion order. */
    std::vector<entry<payload_>> entries;

    /** Row positions under each key, so a lookup is not a scan; keys need not be unique. */
    std::unordered_multimap<uuid_t, std::size_t> by_key;

    /** Held shared by every reader, exclusively by every mutation. */
    std::shared_mutex mutex;

    /** The checkpoint a @c view_ram verb serves from, whose presence refuses every mutating verb. */
    mapped_file_t served;

    /** The residency the handle answers as, so an NVMe view refuses a save as the engine's does. */
    usearch_residency_t residency = usearch_residency_ram_k;

    /** Drops every row, which @c clear promises without touching the frozen options. */
    void reset() noexcept { entries.clear(), by_key.clear(); }

    /** Sizes both containers for @p total entries, answering whether the memory came. */
    bool accommodate(std::size_t total) {
        try {
            entries.reserve(total);
            by_key.reserve(total);
        }
        catch (std::bad_alloc const&) {
            return false;
        }
        return true;
    }

    /** Appends one row and indexes its key, the only way a row is ever added. */
    void append(usearch_key_t key, payload_ payload) {
        by_key.emplace(keyed(key), entries.size());
        entries.push_back({key, std::move(payload)});
    }

    /** Drops every row published past @p watermark, which an unwound batch reverts to. */
    void unwind(std::size_t watermark) {
        for (std::size_t position = entries.size(); position != watermark; --position) {
            auto const range = by_key.equal_range(keyed(entries[position - 1].key));
            for (auto slot = range.first; slot != range.second; ++slot)
                if (slot->second == position - 1) {
                    by_key.erase(slot);
                    break;
                }
        }
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(watermark), entries.end());
    }
};

/**
 *  @brief Refuses a batch that would carry any key past @p values_per_key entries, as the tape
 *      modalities' headers promise. A zeroed bound derives to one, so keys stay unique.
 *
 *  The whole batch is refused rather than the offending row, since a smaller retry cannot help,
 *  and the count spans both the entries already stored and the duplicates the batch itself brings.
 */
template <typename handle_>
bool reject_oversubscribed(handle_ const& index, usearch_keys_view_t keys, std::size_t count,
                           std::size_t values_per_key, usearch_outcome_t* outcome) {
    std::size_t const bound = values_per_key ? values_per_key : 1;
    std::unordered_map<uuid_t, std::size_t> admitted;
    for (std::size_t row = 0; row != count; ++row) {
        uuid_t const key = keyed(key_at(keys, row));
        std::size_t const held = index.by_key.count(key) + ++admitted[key];
        if (reject_if(held > bound, usearch_status_overflow_k, "A key would hold more entries than the index admits",
                      outcome))
            return true;
    }
    return false;
}

/** Refuses a mutating verb on a handle that serves its checkpoint in place. */
template <typename handle_>
bool reject_read_only(handle_ const& index, usearch_outcome_t* outcome) {
    return reject_if(static_cast<bool>(index.served), usearch_status_read_only_k, "The handle is a read-only view",
                     outcome);
}

/** Row positions under @p key, ascending - the insertion order every @c get promises.
 *  An unordered multimap keeps equivalent keys adjacent but in no stated order, so the
 *  ascending sort is what restores the contract rather than a tidiness. */
template <typename handle_>
std::vector<std::size_t> rows_under(handle_ const& index, usearch_key_t key) {
    std::vector<std::size_t> rows;
    auto const range = index.by_key.equal_range(keyed(key));
    for (auto position = range.first; position != range.second; ++position) rows.push_back(position->second);
    std::sort(rows.begin(), rows.end());
    return rows;
}

#pragma endregion Storage

/** Writes @p value's own bytes at the head of @p target - the serializer's one primitive. */
template <typename value_>
void write_pod(bytes_span_t target, value_ const& value) {
    std::array<std::byte, sizeof(value_)> const raw = std::bit_cast<std::array<std::byte, sizeof(value_)>>(value);
    std::copy(raw.begin(), raw.end(), target.begin());
}

/** Reads one @c value_ back from the head of @p source. */
template <typename value_>
value_ read_pod(bytes_view_t source) {
    std::array<std::byte, sizeof(value_)> raw {};
    std::copy_n(source.begin(), sizeof(value_), raw.begin());
    return std::bit_cast<value_>(raw);
}

#pragma region Ranking

/** One scored pair, in the order a search reports them. */
struct match_t {

    /** What the modality's kind scored the pair at; smaller is nearer. */
    double distance;

    /** The key of the entry scored. */
    usearch_key_t key;
};

/** A ranked run's storage inside a claimed arena: @c wanted keys and scores per row, and a count per row. */
struct ranked_landing_t {
    usearch_key_t* keys;
    usearch_distance_t* scores;
    std::size_t* counts;
    std::size_t wanted;

    /** Lands the first @c wanted of @p ranked as row @p row, and their number as its count. */
    void write(std::size_t row, std::span<match_t const> ranked) const noexcept {
        std::size_t const found = std::min(wanted, ranked.size());
        for (std::size_t match = 0; match != found; ++match)
            keys[row * wanted + match] = ranked[match].key, scores[row * wanted + match] = ranked[match].distance;
        counts[row] = found;
    }

    /** The view the call publishes once every row landed. */
    usearch_matches_view_t view() const noexcept { return {keys, scores, counts, wanted}; }
};

/** Carves @p count rows of @p wanted matches out of a claimed @p arena; false once @p outcome records why not. */
inline bool carve_ranked(usearch_arena_handle_t& arena, std::size_t count, std::size_t wanted,
                         ranked_landing_t& landing, usearch_outcome_t* outcome) {
    landing.wanted = wanted;
    return (landing.keys = carve<usearch_key_t>(arena, count * wanted, outcome)) &&
           (landing.scores = carve<usearch_distance_t>(arena, count * wanted, outcome)) &&
           (landing.counts = carve<std::size_t>(arena, count, outcome));
}

/** Scores the @p entries that @p predicate admits, judged in one call, into @p ranked, nearest
 *  first. The sort is stable because every modality's docblock promises ties fall in insertion
 *  order. */
template <typename payload_>
void rank(std::span<entry<payload_> const> entries, usearch_predicate_t predicate,
          std::function<double(payload_ const&)> const& score, std::vector<match_t>& ranked) {
    std::vector<usearch_admission_t> admissions(entries.size(), usearch_admit_k);
    if (predicate.callback && !entries.empty()) {
        std::vector<usearch_key_t> keys(entries.size());
        for (std::size_t row = 0; row != entries.size(); ++row) keys[row] = entries[row].key;
        predicate.callback(keys.data(), admissions.data(), keys.size(), predicate.state);
    }
    ranked.clear();
    ranked.reserve(entries.size());
    for (std::size_t row = 0; row != entries.size(); ++row)
        if (admissions[row] == usearch_admit_k) ranked.push_back({score(entries[row].payload), entries[row].key});
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](match_t const& left, match_t const& right) { return left.distance < right.distance; });
}

#pragma endregion Ranking

#pragma region Sampling

/** Where @p slot falls in the sequence @p seed names: SplitMix64's output at that slot of the
 *  seed's stream, so equal seeds draw a region in one order and distinct seeds in others. */
inline u64_t draw_rank(u64_t seed, std::size_t slot) noexcept {
    u64_t mixed = seed + (static_cast<u64_t>(slot) + 1) * 0x9E3779B97F4A7C15ull;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    return mixed ^ (mixed >> 31);
}

/** The rows below @p rows that @p within admits - a sample's region - in the order @p seed draws
 *  them. */
template <typename within_>
std::vector<std::size_t> drawing_order(std::size_t rows, u64_t seed, within_ const& within) {
    std::vector<std::size_t> order;
    for (std::size_t row = 0; row != rows; ++row)
        if (within(row)) order.push_back(row);
    // The slot breaks a tie between equal ranks, so the order is total and repeats exactly.
    std::sort(order.begin(), order.end(), [seed](std::size_t first, std::size_t second) {
        u64_t const first_rank = draw_rank(seed, first), second_rank = draw_rank(seed, second);
        return first_rank != second_rank ? first_rank < second_rank : first < second;
    });
    return order;
}

/**
 *  @brief The body of every modality's @c sample: lands the draws ranked @c offset onward in
 *      @p order, at most @c wanted, each linked to its @c links_per_draw nearest earlier draws.
 *
 *  Earlier means every draw ranked before it, those an earlier call wrote included, so successive
 *  calls extend one sample. Equidistant draws link in the order they were drawn. The outputs of
 *  @p arguments are written only once every draw landed, so a cancelled call leaves them reset.
 */
template <typename arguments_, typename payload_, typename score_>
void land_sample(arguments_ const& arguments, std::span<entry<payload_> const> entries,
                 std::span<std::size_t const> order, score_ const& score, usearch_arena_handle_t& arena,
                 usearch_outcome_t* outcome) {
    std::size_t const first = std::min(arguments.offset, order.size());
    std::size_t const drawn = std::min(arguments.wanted, order.size() - first);
    std::size_t links_count = 0;
    for (std::size_t rank = first; rank != first + drawn; ++rank)
        links_count += std::min(arguments.links_per_draw, rank);
    usearch_key_t* const keys = carve<usearch_key_t>(arena, drawn, outcome);
    std::size_t* const offsets = keys ? carve<std::size_t>(arena, drawn + 1, outcome) : nullptr;
    usearch_key_t* const linked = offsets ? carve<usearch_key_t>(arena, links_count, outcome) : nullptr;
    usearch_distance_t* const distances = linked ? carve<usearch_distance_t>(arena, links_count, outcome) : nullptr;
    if (!distances) return;

    // Each earlier draw as its distance and rank, so sorting pairs breaks ties by draw order.
    std::vector<std::pair<double, std::size_t>> earlier;
    offsets[0] = 0;
    std::size_t landed = 0;
    while (landed != drawn) {
        std::size_t const rank = first + landed;
        payload_ const& payload = entries[order[rank]].payload;
        earlier.clear();
        for (std::size_t previous = 0; previous != rank; ++previous)
            earlier.push_back({score(payload, entries[order[previous]].payload), previous});
        std::size_t const kept = std::min(arguments.links_per_draw, rank);
        std::partial_sort(earlier.begin(), earlier.begin() + kept, earlier.end());
        for (std::size_t link = 0; link != kept; ++link) {
            linked[offsets[landed] + link] = entries[order[earlier[link].second]].key;
            distances[offsets[landed] + link] = earlier[link].first;
        }
        offsets[landed + 1] = offsets[landed] + kept;
        keys[landed] = entries[order[rank]].key;
        if (arguments.stats) arguments.stats->visited_members += rank, arguments.stats->computed_distances += rank;
        ++landed;
        if (!keep_going(arguments.progress, landed, arguments.wanted)) break;
    }
    if (reject_cancelled(landed, drawn, outcome)) return;
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.links = {linked, distances, offsets};
    *arguments.written = drawn;
}

/**
 *  @brief The rows a cursor opened at @p key yields: those the key's entries link to, once each
 *      and in link order, never an entry under @p key itself. An absent key links to none.
 *
 *  A stand-in keeps no graph, so an entry's links are its @c connectivity nearest other entries by
 *  @p score, nearest first - the lists a perfectly refined graph would hold.
 */
template <typename handle_, typename score_>
std::vector<std::size_t> linked_rows(handle_ const& index, usearch_key_t key, score_ const& score) {
    std::size_t const rows = index.entries.size();
    std::vector<double> distances(rows);
    std::vector<std::size_t> nearest, linked;
    for (std::size_t from : rows_under(index, key)) {
        nearest.clear();
        for (std::size_t row = 0; row != rows; ++row) {
            if (row == from) continue;
            distances[row] = score(index.entries[from].payload, index.entries[row].payload);
            nearest.push_back(row);
        }
        std::stable_sort(nearest.begin(), nearest.end(),
                         [&](std::size_t first, std::size_t second) { return distances[first] < distances[second]; });
        nearest.resize(std::min(nearest.size(), index.options.connectivity));
        for (std::size_t row : nearest)
            if (keyed(index.entries[row].key) != keyed(key) &&
                std::find(linked.begin(), linked.end(), row) == linked.end())
                linked.push_back(row);
    }
    return linked;
}

#pragma endregion Sampling

#pragma region Files

/** The extent of a checkpoint, empty when that many rows cannot be addressed. Every
 *  modality's blob is a prefix followed by @p count entries of @p each bytes. */
inline std::optional<std::size_t> blob_bytes(std::size_t prefix, std::size_t count, std::size_t each) {
    if (each && count > (std::numeric_limits<std::size_t>::max() - prefix) / each) return std::nullopt;
    return prefix + count * each;
}

/** Whether @p blob is long enough for @p count records of @p each bytes past @p header,
 *  reporting through @p outcome which way it fell short. Every modality's @c parse_prefix
 *  ends in this check; only the extent it computes differs. */
inline bool fits_blob(bytes_view_t blob, std::size_t header, std::size_t count, std::size_t each,
                      usearch_outcome_t* outcome) {
    std::optional<std::size_t> const needed = blob_bytes(header, count, each);
    if (reject_if(!needed, usearch_status_overflow_k, "The checkpoint claims more than can be addressed", outcome))
        return false;
    return !reject_if(blob.size() < *needed, usearch_status_malformed_k,
                      "The checkpoint is shorter than the entries it claims", outcome);
}

inline bool read_file(usearch_bytes_view_t path, std::vector<std::byte>& blob, usearch_outcome_t* outcome) {
    auto [file, status] = seekable_file_t::open(as_text(path), file_mode_t::read_k);
    if (failed(status)) {
        propagate_error(usearch_status_io_k, outcome, "The checkpoint could not be opened for reading");
        return false;
    }
    auto const [bytes, length_status] = file.length();
    if (failed(length_status)) {
        propagate_error(usearch_status_io_k, outcome, "The checkpoint could not be read");
        return false;
    }
    blob.resize(bytes);
    if (failed(file.read_at(0, bytes_span_t {blob}))) {
        propagate_error(usearch_status_io_k, outcome, "The checkpoint could not be read");
        return false;
    }
    return true;
}

/** Maps the checkpoint at @p path for a view to serve from; empty once the outcome records why not. */
inline mapped_file_t map_file(usearch_bytes_view_t path, usearch_outcome_t* outcome) {
    auto [mapping, status] = mapped_file_t::read(as_text(path));
    // An empty file maps nothing, which is the short checkpoint every parser would refuse.
    if (status == status_t::malformed_k)
        propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends early");
    else if (failed(status))
        propagate_error(usearch_status_io_k, outcome, "The checkpoint could not be opened for reading");
    return std::move(mapping);
}

inline bool write_file(usearch_bytes_view_t path, bytes_view_t blob, usearch_outcome_t* outcome) {
    auto [file, status] = seekable_file_t::open(as_text(path), file_mode_t::write_k);
    if (failed(status)) {
        propagate_error(usearch_status_io_k, outcome, "The checkpoint could not be opened for writing");
        return false;
    }
    if (failed(file.write_at(0, blob)) || failed(file.close())) {
        propagate_error(usearch_status_io_k, outcome, "The checkpoint could not be written");
        return false;
    }
    return true;
}

#pragma endregion Files

} // namespace stub
} // namespace usearch
} // namespace unum

struct usearch_executor_handle_t : us::stub::executor_t {};

#endif // UNUM_USEARCH_STUB_LIB_HPP
