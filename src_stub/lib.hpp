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

#include <cstdio> // `std::snprintf` - a part's name and a manifest's entry

#include <algorithm>        // `std::copy`, `std::sort`, `std::stable_sort`, `std::min`
#include <array>            // `std::array`
#include <bit>              // `std::bit_cast`
#include <functional>       // `std::function`
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <limits>           // `std::numeric_limits`
#include <mutex>            // `std::unique_lock`
#include <new>              // `std::bad_alloc`
#include <numbers>          // `std::numbers::pi` - the sample projection's angles
#include <optional>         // `std::optional`
#include <shared_mutex>     // `std::shared_mutex`, `std::shared_lock`
#include <span>             // `std::span`
#include <string>           // `std::string`
#include <string_view>      // `std::string_view` - a byte run read as text
#include <type_traits>      // `std::is_nothrow_move_constructible_v`, `std::is_pointer_v`, `std::remove_pointer_t`
#include <unordered_map>    // `std::unordered_map`, `std::unordered_multimap`
#include <unordered_set>    // `std::unordered_set` - a sample region's visits
#include <variant>          // `std::variant`
#include <vector>           // `std::vector`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the serial class's reference over every shared word
#include <forkunion/types.hpp>   // `fu::standard_yield_t` - the serial class's waiter

#include "usearch/usearch.h" // every argument structure a call copies

#include "usearch/utilities/blocking_files.hpp" // `seekable_file_t`, `mapped_file_t` - the checkpoints' handles
#include "usearch/utilities/ram_buffers.hpp"    // `growing_arena` - the engine's arena, `contains` included
#include "usearch/utilities/types.hpp"

namespace us = unum::usearch;

/** What @c usearch_arena_t owns: host memory that one call at a time rewinds and refills. */
struct usearch_arena_handle_t {
    us::utilities::growing_arena<> memory {us::utilities::global_allocator};

    /** The outcome of the call holding the arena, from its claim or deferred submission until just
     *  before its final status; null while free. Touched through @c fu::standard_atomic_ref, since
     *  a held call finishes on any thread. */
    usearch_outcome_t const* holder = nullptr;
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
    if (outcome)
        fu::standard_atomic_ref<usearch_status_t>(outcome->status)
            .store(usearch_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status with release ordering, which
 *  waiters watch by address, so nothing notifies. The message is null exactly on success. */
inline void propagate_error(usearch_status_t status, usearch_outcome_t* outcome, char const* message) {
    if (!outcome) return;
    outcome->message = status == usearch_status_success_k ? nullptr : message;
    fu::standard_atomic_ref<usearch_status_t>(outcome->status).store(status, std::memory_order_release);
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
        if (arena_)
            fu::standard_atomic_ref<usearch_outcome_t const*>(arena_->holder)
                .compare_exchange_strong(holder, nullptr, std::memory_order_release, std::memory_order_relaxed);
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

/** Refuses a traversal other than a walk, a scan, or nothing, which derives the walk; this stand-in
 *  answers both exactly. */
inline bool reject_traversal(usearch_traversal_t traversal, usearch_outcome_t* outcome) {
    return reject_if(traversal != usearch_traversal_unknown_k && traversal != usearch_traversal_walk_k &&
                         traversal != usearch_traversal_scan_k,
                     usearch_status_unknown_kind_k, "Unknown traversal", outcome);
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

/** The key at @p offset of a run, where a zero stride repeats the first. */
inline usearch_key_t key_at(usearch_keys_view_t keys, std::size_t offset) {
    return *reinterpret_cast<usearch_key_t const*>(reinterpret_cast<std::byte const*>(keys.data) +
                                                   offset * keys.stride);
}

/** Refuses a tape without its storage or offsets, or whose offsets ever decrease. */
inline bool reject_tape(usearch_tape_view_t tape, std::size_t count, usearch_outcome_t* outcome) {
    if (reject_if(!tape.offsets || (!tape.data && count), usearch_status_invalid_argument_k,
                  "The tape must name storage", outcome))
        return true;
    for (std::size_t entry = 0; entry != count; ++entry)
        if (reject_if(tape.offsets[entry + 1] < tape.offsets[entry], usearch_status_invalid_argument_k,
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
using call_arguments_t = std::variant< //

    /* Loading */
    usearch_dense_load_ram_t, usearch_dense_load_ram_buffer_t, usearch_dense_view_nvme_t,          //
    usearch_dense_load_vram_t, usearch_dense_load_vram_buffer_t,                                   //
    usearch_sequence_load_ram_t, usearch_sequence_view_nvme_t, usearch_sequence_load_vram_t,       //
    usearch_shapes_load_ram_t, usearch_shapes_view_nvme_t, usearch_shapes_load_vram_t,             //
    usearch_geospatial_load_ram_t, usearch_geospatial_view_nvme_t, usearch_geospatial_load_vram_t, //

    /* Building */
    usearch_dense_reserve_t, usearch_dense_clear_t, usearch_dense_insert_t,                //
    usearch_sequence_reserve_t, usearch_sequence_clear_t, usearch_sequence_insert_t,       //
    usearch_shapes_reserve_t, usearch_shapes_clear_t, usearch_shapes_insert_t,             //
    usearch_geospatial_reserve_t, usearch_geospatial_clear_t, usearch_geospatial_insert_t, //

    /* Refining */
    usearch_dense_refine_t, usearch_dense_health_t, usearch_dense_absorb_t, usearch_dense_split_t,             //
    usearch_sequence_refine_t, usearch_sequence_health_t, usearch_sequence_absorb_t, usearch_sequence_split_t, //
    usearch_shapes_refine_t, usearch_shapes_health_t, usearch_shapes_absorb_t, usearch_shapes_split_t,         //
    usearch_geospatial_refine_t, usearch_geospatial_health_t, usearch_geospatial_absorb_t,                     //
    usearch_geospatial_split_t,                                                                                //

    /* Refinements */
    usearch_dense_refinement_init_t, usearch_dense_refinement_next_t, usearch_dense_refinement_finish_t,          //
    usearch_sequence_refinement_init_t, usearch_sequence_refinement_next_t, usearch_sequence_refinement_finish_t, //
    usearch_shapes_refinement_init_t, usearch_shapes_refinement_next_t, usearch_shapes_refinement_finish_t,       //
    usearch_geospatial_refinement_init_t, usearch_geospatial_refinement_next_t,                                   //
    usearch_geospatial_refinement_finish_t,                                                                       //

    /* Reading */
    usearch_dense_search_t, usearch_dense_search_stream_next_t, usearch_dense_count_t, usearch_dense_get_t,      //
    usearch_dense_cursor_next_t, usearch_dense_sample_t, usearch_dense_describe_t,                               //
    usearch_sequence_search_t, usearch_sequence_count_t, usearch_sequence_get_t, usearch_sequence_cursor_next_t, //
    usearch_sequence_sample_t, usearch_sequence_describe_t,                                                      //
    usearch_shapes_search_t, usearch_shapes_count_t, usearch_shapes_get_t, usearch_shapes_cursor_next_t,         //
    usearch_shapes_sample_t, usearch_shapes_describe_t,                                                          //
    usearch_geospatial_search_t, usearch_geospatial_within_t, usearch_geospatial_count_t,                        //
    usearch_geospatial_get_t, usearch_geospatial_cursor_next_t, usearch_geospatial_sample_t,                     //
    usearch_geospatial_describe_t,                                                                               //

    /* Persisting */
    usearch_dense_save_t, usearch_dense_save_buffer_t, usearch_dense_metadata_read_t, //
    usearch_dense_metadata_read_buffer_t,                                             //
    usearch_sequence_save_t, usearch_sequence_metadata_read_t,                        //
    usearch_shapes_save_t, usearch_shapes_metadata_read_t,                            //
    usearch_geospatial_save_t, usearch_geospatial_metadata_read_t,                    //

    /* Products */
    usearch_dense_multiply_t, usearch_dense_attend_t, usearch_dense_maxsim_t>;

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

    /** The call's outcome, its executor, and every index, cursor, refinement, stream and arena it
     *  reads or writes: the addresses a wait or a free looks for before it returns. */
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
    if constexpr (requires { arguments.refinement; }) add(arguments.refinement);
    if constexpr (requires { arguments.stream; }) add(arguments.stream);
    if constexpr (requires { arguments.arena; }) add(arguments.arena);
    return handles;
}

/** Holds @p arena for the call whose outcome is @p call, answering whether that call holds it now:
 *  it was free, or already that call's. A null arena holds nothing, left for the verb to refuse. */
inline bool hold_arena(usearch_arena_t arena, usearch_outcome_t const* call) {
    if (!arena) return true;
    usearch_outcome_t const* holder = nullptr;
    return fu::standard_atomic_ref<usearch_outcome_t const*>(arena->holder)
               .compare_exchange_strong(holder, call, std::memory_order_acquire, std::memory_order_acquire) ||
           holder == call;
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
        fu::standard_atomic_ref<usearch_outcome_t const*>(arena->holder)
            .compare_exchange_strong(holder, nullptr, std::memory_order_release, std::memory_order_relaxed);
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
    std::byte* const carved = arena.memory.allocate(count * sizeof(value_type_), default_alignment_k);
    if (!carved) propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
    return reinterpret_cast<value_type_*>(carved);
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
    // A zero exponent field is the subnormal step, so the two branches differ only in the implied
    // leading bit.
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
    // A kind reserving no code cannot carry a NaN, so it falls to zero.
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

/** One stored entry: its key, and whatever the modality calls a payload. */
template <typename payload_>
struct entry {
    usearch_key_t key;
    payload_ payload;
};

/** The project's own key type, which already knows how to hash and compare itself.
 *  Layout-identical to @c usearch_key_t, so the conversion is a reinterpretation. */
inline uuid_t uuid_of(usearch_key_t const& key) noexcept { return std::bit_cast<uuid_t>(key); }

/**
 *  @brief One handle's whole state. Readers hold @c mutex shared and writers exclusively, which
 *      is what keeps a search overlapping an insertion.
 *
 *  Reallocating @c entries moves each entry without copying the buffer it owns - true only while
 *  the payload's move constructor stays @c noexcept, which the modality files assert.
 */
template <typename payload_, typename options_>
struct handle {

    /** The options the handle was initialized with, or a checkpoint later adopted. */
    options_ options {};

    /** The entries themselves, in insertion order. */
    std::vector<entry<payload_>> entries;

    /** Entry positions under each key, so a lookup is not a scan; keys need not be unique. */
    std::unordered_multimap<uuid_t, std::size_t> by_key;

    /** Held shared by every reader, exclusively by every mutation. */
    std::shared_mutex mutex;

    /** The checkpoint a @c view_ram verb serves from, whose presence refuses every mutating verb. */
    mapped_file_t served;

    /** The residency the handle answers as, so an NVMe view refuses a save as the engine's does. */
    usearch_residency_t residency = usearch_residency_ram_k;

    /** The refinement holding the handle closed to insertions, saves and other refinements until it
     *  finishes, or null; read and written under @c mutex. */
    void const* refinement = nullptr;

    /** Entries one shard holds before the next opens; zero means the engine's 2^32 - 2. */
    std::size_t shard_capacity = 0;

    /** Parts the checkpoint the handle opened from had; zero for a handle built by insertion. */
    std::size_t parts_opened = 0;

    /** Live shards: the parts a checkpoint opened, or one per @c shard_capacity entries, one at
     *  the least. */
    std::size_t shards() const noexcept {
        std::size_t const capacity = shard_capacity ? shard_capacity : std::numeric_limits<std::size_t>::max();
        return std::max({std::size_t {1}, parts_opened, divide_round_up(entries.size(), capacity)});
    }

    /** Drops every entry, which @c clear promises without touching the frozen options. */
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

    /** Appends one entry and indexes its key, the only way an entry is ever added. */
    void append(usearch_key_t key, payload_ payload) {
        by_key.emplace(uuid_of(key), entries.size());
        entries.push_back({key, std::move(payload)});
    }

    /** Drops every entry published past @p watermark, which an unwound batch reverts to. */
    void unwind(std::size_t watermark) {
        for (std::size_t position = entries.size(); position != watermark; --position) {
            auto const range = by_key.equal_range(uuid_of(entries[position - 1].key));
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
 *  The whole batch is refused rather than the offending entry, since a smaller retry cannot help,
 *  and the count spans both the entries already stored and the duplicates the batch itself brings.
 */
template <typename handle_>
bool reject_oversubscribed(handle_ const& index, usearch_keys_view_t keys, std::size_t count,
                           std::size_t values_per_key, usearch_outcome_t* outcome) {
    std::size_t const bound = values_per_key ? values_per_key : 1;
    std::unordered_map<uuid_t, std::size_t> admitted;
    for (std::size_t offset = 0; offset != count; ++offset) {
        uuid_t const key = uuid_of(key_at(keys, offset));
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

/** Refuses an insertion, a save or another refinement while a refinement holds the handle closed;
 *  the caller holds the handle's mutex. */
template <typename handle_>
bool reject_refining(handle_ const& index, usearch_outcome_t* outcome) {
    return reject_if(index.refinement != nullptr, usearch_status_contended_k, "A refinement holds the handle closed",
                     outcome);
}

/** Entry positions under @p key, ascending - the insertion order every @c get promises.
 *  An unordered multimap keeps equivalent keys adjacent but in no stated order, so the
 *  ascending sort is what restores the contract rather than a tidiness. */
template <typename handle_>
std::vector<std::size_t> nodes_of(handle_ const& index, usearch_key_t key) {
    std::vector<std::size_t> entries;
    auto const range = index.by_key.equal_range(uuid_of(key));
    for (auto position = range.first; position != range.second; ++position) entries.push_back(position->second);
    std::sort(entries.begin(), entries.end());
    return entries;
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

/** A ranked run's storage inside a claimed arena: @c wanted keys and scores per query, and a count
 *  per query. */
struct matches_outputs_t {
    usearch_key_t* keys = nullptr;
    usearch_distance_t* scores = nullptr;
    std::size_t* counts = nullptr;
    std::size_t wanted = 0;

    /** Writes the first @c wanted of @p ranked as the matches of @p query, and their number as its
     *  count. */
    void write(std::size_t query, std::span<match_t const> ranked) const noexcept {
        std::size_t const found = std::min(wanted, ranked.size());
        for (std::size_t match = 0; match != found; ++match)
            keys[query * wanted + match] = ranked[match].key, scores[query * wanted + match] = ranked[match].distance;
        counts[query] = found;
    }

    /** The view the call publishes once every query is written. */
    usearch_matches_view_t view() const noexcept { return {keys, scores, counts, wanted}; }
};

/** Carves @p wanted matches for each of @p count queries out of a claimed @p arena; false once
 *  @p outcome records why not. */
inline bool carve_ranked(usearch_arena_handle_t& arena, std::size_t count, std::size_t wanted,
                         matches_outputs_t& outputs, usearch_outcome_t* outcome) {
    outputs.wanted = wanted;
    return (outputs.keys = carve<usearch_key_t>(arena, count * wanted, outcome)) &&
           (outputs.scores = carve<usearch_distance_t>(arena, count * wanted, outcome)) &&
           (outputs.counts = carve<std::size_t>(arena, count, outcome));
}

/** Scores the @p entries that @p predicate admits, judged a quad at a time as the engine asks, into
 *  @p ranked, nearest first. The sort is stable because every modality's docblock promises ties
 *  fall in insertion order. */
template <typename payload_>
void rank(std::span<entry<payload_> const> entries, usearch_predicate_t predicate,
          std::function<double(payload_ const&)> const& score, std::vector<match_t>& ranked) {
    std::vector<usearch_admission_t> admissions(entries.size(), usearch_admit_k);
    if (predicate.callback && !entries.empty()) {
        std::vector<usearch_key_t> keys(entries.size());
        for (std::size_t entry = 0; entry != entries.size(); ++entry) keys[entry] = entries[entry].key;
        for (std::size_t first = 0; first < keys.size(); first += 4)
            predicate.callback(keys.data() + first, admissions.data() + first,
                               std::min<std::size_t>(4, keys.size() - first), predicate.state);
    }
    ranked.clear();
    ranked.reserve(entries.size());
    for (std::size_t entry = 0; entry != entries.size(); ++entry)
        if (admissions[entry] == usearch_admit_k) ranked.push_back({score(entries[entry].payload), entries[entry].key});
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](match_t const& left, match_t const& right) { return left.distance < right.distance; });
}

#pragma endregion Ranking

#pragma region Graph Walks

/**
 *  @brief The nodes @p from links to: its @c connectivity nearest other nodes below @p limit by
 *      @p score, nearest first, ties in storage order.
 *
 *  A stand-in keeps no graph, so these are the lists a perfectly refined graph would hold, where
 *  the engine's are the ones it stored.
 */
template <typename handle_, typename score_>
std::vector<std::size_t> links_of(handle_ const& index, std::size_t from, std::size_t limit, score_ const& score) {
    std::vector<std::pair<double, std::size_t>> nearest;
    nearest.reserve(limit);
    for (std::size_t node = 0; node != limit; ++node)
        if (node != from) nearest.push_back({score(index.entries[from].payload, index.entries[node].payload), node});
    std::size_t const kept = std::min(nearest.size(), index.options.connectivity);
    std::partial_sort(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(kept), nearest.end());
    std::vector<std::size_t> linked(kept);
    for (std::size_t link = 0; link != kept; ++link) linked[link] = nearest[link].second;
    return linked;
}

/** A stand-in's cursor: every node below the creation watermark as stored, or the start keys' nodes
 *  and then, breadth-first, the nodes they link to, each once. */
template <typename handle_>
struct graph_cursor {

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    handle_ const* opened_on = nullptr;

    /** The creation watermark, which later insertions cannot raise. */
    std::size_t limit = 0;

    /** Nodes already handed out. */
    std::size_t position = 0;

    /** The breadth-first order discovered so far; empty for a scan, which yields nodes as
     *  stored. */
    std::vector<std::size_t> order;

    /** Per node below @c limit, whether @c order holds it; empty for a scan. */
    std::vector<std::uint8_t> visited;

    /** Nodes of @c order whose links are already queued behind them. */
    std::size_t expanded = 0;

    /** Each node's links, kept once a pull or an expansion computed them. */
    std::unordered_map<std::size_t, std::vector<std::size_t>> links;

    /** Bit zero is set while a pull runs, so an overlapping pull is refused instead of racing. */
    u32_t pulling = 0;
};

/** Opens @p cursor on @p index: a scan without @p starts, else a walk beginning with the start
 *  keys' own nodes, in the order given, each once. */
template <typename handle_>
void open_walk(graph_cursor<handle_>& cursor, std::type_identity_t<handle_> const& index, usearch_key_t const* starts,
               std::size_t starts_count) {
    cursor.opened_on = &index;
    cursor.limit = index.entries.size();
    if (!starts_count) return;
    cursor.visited.assign(cursor.limit, 0);
    for (std::size_t start = 0; start != starts_count; ++start)
        for (std::size_t node : nodes_of(index, starts[start]))
            if (node < cursor.limit && !cursor.visited[node]) cursor.visited[node] = 1, cursor.order.push_back(node);
}

/** The links of @p node, computed on first use and kept. */
template <typename handle_, typename score_>
std::vector<std::size_t> const& walk_links(graph_cursor<handle_>& cursor, std::size_t node, score_ const& score) {
    auto found = cursor.links.find(node);
    if (found == cursor.links.end())
        found = cursor.links.emplace(node, links_of(*cursor.opened_on, node, cursor.limit, score)).first;
    return found->second;
}

/**
 *  @brief The body of every modality's @c cursor_next: writes the next @c wanted nodes' keys and,
 *      when asked, their links as a keys tape. A walk expands only as far as the pull needs.
 *
 *  A batch the arena could not hold leaves the cursor where it was, so the next pull retries it.
 */
template <typename arguments_, typename handle_, typename score_>
void pull_walk(arguments_ const& arguments, graph_cursor<handle_>& cursor, score_ const& score,
               usearch_arena_handle_t& arena, usearch_outcome_t* outcome) {
    handle_ const& index = *cursor.opened_on;
    std::size_t const wanted = arguments.wanted;
    std::vector<std::size_t> nodes;
    if (cursor.visited.empty()) {
        std::size_t const taking = std::min(wanted, cursor.limit - std::min(cursor.position, cursor.limit));
        for (std::size_t offset = 0; offset != taking; ++offset) nodes.push_back(cursor.position + offset);
    }
    else {
        while (cursor.order.size() < cursor.position + wanted && cursor.expanded < cursor.order.size()) {
            for (std::size_t linked : walk_links(cursor, cursor.order[cursor.expanded], score))
                if (!cursor.visited[linked]) cursor.visited[linked] = 1, cursor.order.push_back(linked);
            ++cursor.expanded;
        }
        std::size_t const end = std::min(cursor.order.size(), cursor.position + wanted);
        nodes.assign(cursor.order.begin() + static_cast<std::ptrdiff_t>(cursor.position),
                     cursor.order.begin() + static_cast<std::ptrdiff_t>(end));
    }
    std::size_t total_links = 0;
    if (arguments.links)
        for (std::size_t node : nodes) total_links += walk_links(cursor, node, score).size();
    usearch_key_t* const keys = carve<usearch_key_t>(arena, nodes.size(), outcome);
    std::size_t* const offsets = keys && arguments.links ? carve<std::size_t>(arena, nodes.size() + 1, outcome)
                                                         : nullptr;
    usearch_key_t* const linked_keys = offsets ? carve<usearch_key_t>(arena, total_links, outcome) : nullptr;
    if (!keys || (arguments.links && !linked_keys)) return;
    for (std::size_t offset = 0; offset != nodes.size(); ++offset) keys[offset] = index.entries[nodes[offset]].key;
    if (arguments.links) {
        offsets[0] = 0;
        for (std::size_t offset = 0; offset != nodes.size(); ++offset) {
            std::vector<std::size_t> const& linked = walk_links(cursor, nodes[offset], score);
            for (std::size_t link = 0; link != linked.size(); ++link)
                linked_keys[offsets[offset] + link] = index.entries[linked[link]].key;
            offsets[offset + 1] = offsets[offset] + linked.size();
        }
        *arguments.links = {linked_keys, offsets};
    }
    cursor.position += nodes.size();
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.written = nodes.size();
}

#pragma endregion Graph Walks

#pragma region Sampling

/** Shown points each draw is placed against, at most, as the library's plan keeps them. */
inline constexpr std::size_t sample_nearest_shown_k = 8;

/** Gauss-Newton steps one draw's coordinates take. */
inline constexpr std::size_t sample_gauss_newton_steps_k = 8;

/** Draws a classical scaling spans; later draws are placed among them. */
inline constexpr std::size_t sample_scaling_draws_k = 32;

/** Entries a region visits per draw it may make, when the call names no breadth. */
inline constexpr std::size_t sample_expansion_factor_k = 16;

/** SplitMix64's output at @p counter of the stream @p seed starts: its state after @p counter + 1
 *  steps, mixed. */
constexpr u64_t splitmix64_at(u64_t seed, u64_t counter) noexcept {
    return mix_bits(seed + (counter + 1) * golden_ratio_u64_k);
}

/** The top 53 bits of @p bits as a double in (0, 1]: ((bits >> 11) + 1) · 2⁻⁵³. */
constexpr f64_t uniform_unit_f64(u64_t bits) noexcept { return static_cast<f64_t>((bits >> 11) + 1) * 0x1p-53; }

/**
 *  @brief Projects one row to three coordinates through a dense Gaussian matrix no one stores:
 *      coordinates[c] = Σ_d vector[d] · G[d][c] / sqrt(dimensions).
 *
 *  G[d][c], with n = 3d + c, is the Box-Muller cosine sqrt(−2 ln u₁) · cos(2π u₂) in f64. Its u₁ is
 *  @ref uniform_unit_f64 of @ref splitmix64_at (seed, 2n) and u₂ the same at 2n + 1. The sums run
 *  in f64, d ascending then c, each adding (value · sqrt(−2 ln u₁)) · cos(2π u₂); each divides by
 *  sqrt(dimensions) in f64 and rounds to f32 once. A row of no dimensions sits at the origin.
 */
inline void sample_projection(f32_t const* vector, std::size_t dimensions, u64_t seed, f32_t* coordinates) noexcept {
#pragma float_control(precise, on)
#pragma clang fp contract(off)
    std::array<f64_t, 3> sums {0, 0, 0};
    for (std::size_t dimension = 0; dimension != dimensions; ++dimension) {
        f64_t const value = vector[dimension];
        for (std::size_t column = 0; column != 3; ++column) {
            u64_t const counter = 2 * (3 * static_cast<u64_t>(dimension) + column);
            f64_t const radius = std::sqrt(-2.0 * std::log(uniform_unit_f64(splitmix64_at(seed, counter))));
            f64_t const angle = 2.0 * std::numbers::pi * uniform_unit_f64(splitmix64_at(seed, counter + 1));
            sums[column] += value * radius * std::cos(angle);
        }
    }
    f64_t const root = std::sqrt(static_cast<f64_t>(dimensions));
    for (std::size_t column = 0; column != 3; ++column)
        coordinates[column] = dimensions ? static_cast<f32_t>(sums[column] / root) : 0.0f;
}

/**
 *  @brief Places one entry where its distances to @p references_count anchors best match
 *      @p distances in relative error, Σ_l (‖p − y_l‖ − d_l)² / d_l², by
 *      @ref sample_gauss_newton_steps_k damped Gauss-Newton steps.
 *
 *  @param[in] anchors @b [references_count,3] the anchors' coordinates.
 *  @param[in] distances @b [references_count] the entry's distances to them, in coordinate units.
 *  @param[in] references_count Anchors, at least one.
 *  @param[in] key_seed The entry's own seed, which nudges the start off the anchors.
 *  @param[out] coordinates The entry's three coordinates.
 *
 *  A distance that is not positive puts the entry on the first such anchor. Otherwise it runs in
 *  f32, anchors ascending:
 *  1. p starts at the mean of the anchors weighted by 1 / d_l.
 *  2. p moves by 0.01 · min_l d_l · (2u_c − 1), u_c = @ref uniform_unit_f64 of @ref splitmix64_at
 *     over @p key_seed and c, rounded to f32.
 *  3. Each step sums H = Σ J_lᵀJ_l and g = Σ J_lᵀ e_l over the anchors p is off, with J_l = (p −
 *     y_l) / (‖p − y_l‖ · d_l) and e_l = (‖p − y_l‖ − d_l) / d_l.
 *  4. It solves (H + λI) s = −g, λ = 0.001 · trace(H), by Cramer's rule, and moves p by s; a
 *     determinant that is zero or not finite stops the steps.
 */
inline void sample_gauss_newton(f32_t const* anchors, f32_t const* distances, std::size_t references_count,
                                u64_t key_seed, f32_t* coordinates) noexcept {
#pragma float_control(precise, on)
#pragma clang fp contract(off)
    for (std::size_t anchor = 0; anchor != references_count; ++anchor) {
        if (distances[anchor] > 0.0f) continue;
        for (std::size_t column = 0; column != 3; ++column) coordinates[column] = anchors[anchor * 3 + column];
        return;
    }
    f32_t weights = 0.0f, nearest = distances[0];
    std::array<f32_t, 3> point {0.0f, 0.0f, 0.0f};
    for (std::size_t anchor = 0; anchor != references_count; ++anchor) {
        f32_t const weight = 1.0f / distances[anchor];
        weights += weight;
        for (std::size_t column = 0; column != 3; ++column) point[column] += weight * anchors[anchor * 3 + column];
        nearest = min_of(nearest, distances[anchor]);
    }
    for (std::size_t column = 0; column != 3; ++column) {
        f32_t const unit = static_cast<f32_t>(uniform_unit_f64(splitmix64_at(key_seed, column)));
        point[column] = point[column] / weights + 0.01f * nearest * (2.0f * unit - 1.0f);
    }

    for (std::size_t step = 0; step != sample_gauss_newton_steps_k; ++step) {
        f32_t hessian_00 = 0.0f, hessian_01 = 0.0f, hessian_02 = 0.0f, hessian_11 = 0.0f, hessian_12 = 0.0f,
              hessian_22 = 0.0f;
        f32_t gradient_0 = 0.0f, gradient_1 = 0.0f, gradient_2 = 0.0f;
        for (std::size_t anchor = 0; anchor != references_count; ++anchor) {
            f32_t const offset_0 = point[0] - anchors[anchor * 3 + 0];
            f32_t const offset_1 = point[1] - anchors[anchor * 3 + 1];
            f32_t const offset_2 = point[2] - anchors[anchor * 3 + 2];
            f32_t const reach = std::sqrt(offset_0 * offset_0 + offset_1 * offset_1 + offset_2 * offset_2);
            if (!(reach > 0.0f)) continue;
            f32_t const wanted = distances[anchor];
            f32_t const scale = 1.0f / (reach * wanted);
            f32_t const jacobian_0 = offset_0 * scale, jacobian_1 = offset_1 * scale, jacobian_2 = offset_2 * scale;
            f32_t const error = (reach - wanted) / wanted;
            hessian_00 += jacobian_0 * jacobian_0, hessian_01 += jacobian_0 * jacobian_1,
                hessian_02 += jacobian_0 * jacobian_2, hessian_11 += jacobian_1 * jacobian_1,
                hessian_12 += jacobian_1 * jacobian_2, hessian_22 += jacobian_2 * jacobian_2;
            gradient_0 += jacobian_0 * error, gradient_1 += jacobian_1 * error, gradient_2 += jacobian_2 * error;
        }
        f32_t const damping = 0.001f * (hessian_00 + hessian_11 + hessian_22);
        f32_t const damped_00 = hessian_00 + damping, damped_11 = hessian_11 + damping,
                    damped_22 = hessian_22 + damping;
        f32_t const cofactor_0 = damped_11 * damped_22 - hessian_12 * hessian_12,
                    cofactor_1 = hessian_01 * damped_22 - hessian_12 * hessian_02,
                    cofactor_2 = hessian_01 * hessian_12 - damped_11 * hessian_02;
        f32_t const determinant = damped_00 * cofactor_0 - hessian_01 * cofactor_1 + hessian_02 * cofactor_2;
        if (determinant == 0.0f || !std::isfinite(determinant)) break;
        f32_t const negative_gradient_0 = -gradient_0, negative_gradient_1 = -gradient_1,
                    negative_gradient_2 = -gradient_2;
        f32_t const step_0 = (negative_gradient_0 * cofactor_0 -
                              hessian_01 * (negative_gradient_1 * damped_22 - hessian_12 * negative_gradient_2) +
                              hessian_02 * (negative_gradient_1 * hessian_12 - damped_11 * negative_gradient_2)) /
                             determinant;
        f32_t const step_1 = (damped_00 * (negative_gradient_1 * damped_22 - hessian_12 * negative_gradient_2) -
                              negative_gradient_0 * cofactor_1 +
                              hessian_02 * (hessian_01 * negative_gradient_2 - negative_gradient_1 * hessian_02)) /
                             determinant;
        f32_t const step_2 = (damped_00 * (damped_11 * negative_gradient_2 - negative_gradient_1 * hessian_12) -
                              hessian_01 * (hessian_01 * negative_gradient_2 - negative_gradient_1 * hessian_02) +
                              negative_gradient_0 * cofactor_2) /
                             determinant;
        point[0] += step_0, point[1] += step_1, point[2] += step_2;
    }
    for (std::size_t column = 0; column != 3; ++column) coordinates[column] = point[column];
}

/** Places draws without shown points by classical scaling among themselves, as sequences and
 *  shapes do. */
struct no_fixed_coordinates_t {};

/**
 *  @brief One SAMPLE call over the stand-in's exhaustive face.
 *
 *  Every link is a perfect graph's, every distance computed afresh, and the steps those of the
 *  library's plan - roots, paths, regions, coordinates - in its order, so equal inputs over an
 *  equal graph draw and place equally.
 *
 *  A node is an entry's position, the stand-in's one shard. A classical scaling takes neither of
 *  the @p coordinates_kind_ answers: <tt>coordinates_of(payload, coordinates)</tt> for fixed
 *  coordinates, or <tt>vector_of(payload)</tt> for a projected row.
 */
template <typename arguments_, typename handle_, typename score_, typename coordinates_kind_>
struct sample_plan {
    inline static constexpr std::size_t none_k = std::numeric_limits<std::size_t>::max();

    arguments_ const& arguments;
    handle_ const& index;
    score_ const& score;
    std::size_t limit;
    std::size_t wanted;
    std::size_t visits;

    usearch_key_t* keys;
    std::size_t* entry_ordinals;
    usearch_key_t* parents;
    f32_t* parent_distances;
    f32_t* coordinates;

    std::vector<std::size_t> drawn {};
    std::unordered_map<std::size_t, std::size_t> drawn_at {};
    std::unordered_map<uuid_t, std::size_t> shown_rows {};
    std::vector<std::size_t> shown_nodes {}, shown_offsets {};
    std::vector<std::size_t> nearest_shown {};
    std::vector<f32_t> nearest_shown_distances {};
    std::vector<std::size_t> roots {}, root_draws {};
    std::vector<f32_t> root_radii {};
    std::unordered_set<std::size_t> shown_roots {};
    std::unordered_map<std::size_t, std::vector<std::size_t>> links {};
    std::size_t nearest_shown_count = 0;
    std::size_t visited_members = 0, computed_distances = 0;

    static u64_t key_hash(usearch_key_t const& key) noexcept {
        return static_cast<u64_t>(std::hash<uuid_t> {}(uuid_of(key)));
    }
    usearch_key_t shown_key(std::size_t row) const noexcept {
        std::span<usearch_key_t const> const keys = row_at<usearch_key_t>(arguments.shown.data, row,
                                                                          arguments.shown.stride, 1);
        return keys[0];
    }
    f32_t const* shown_coordinates_of(std::size_t row) const noexcept {
        return row_at<f32_t>(arguments.shown_coordinates.data, row, arguments.shown_coordinates.stride, 3).data();
    }
    usearch_key_t focus_key(std::size_t row) const noexcept {
        std::span<usearch_key_t const> const keys = row_at<usearch_key_t>(arguments.focus.data, row,
                                                                          arguments.focus.stride, 1);
        return keys[0];
    }
    bool shown(usearch_key_t const& key) const noexcept { return shown_rows.contains(uuid_of(key)); }

    f32_t distance(std::size_t from, std::size_t to) noexcept {
        ++computed_distances;
        return static_cast<f32_t>(score(index.entries[from].payload, index.entries[to].payload));
    }
    std::vector<std::size_t> const& neighbors(std::size_t node) {
        auto found = links.find(node);
        if (found == links.end()) found = links.emplace(node, links_of(index, node, limit, score)).first;
        return found->second;
    }
    std::vector<std::size_t> nodes_of(usearch_key_t const& key) const {
        std::vector<std::size_t> nodes = stub::nodes_of(index, key);
        std::erase_if(nodes, [&](std::size_t node) { return node >= limit; });
        return nodes;
    }

    /** Steps from @p from to the link nearest @p toward while it comes closer, as the library's
     *  greedy walk does. */
    std::vector<std::size_t> walk(std::size_t from, std::size_t toward) {
        std::vector<std::size_t> chain;
        std::size_t current = from;
        f32_t current_distance = static_cast<f32_t>(score(index.entries[toward].payload, index.entries[from].payload));
        while (current != toward) {
            std::size_t best = current;
            f32_t best_distance = current_distance;
            for (std::size_t next : neighbors(current)) {
                f32_t const reach = static_cast<f32_t>(
                    score(index.entries[toward].payload, index.entries[next].payload));
                if (reach < best_distance) best = next, best_distance = reach;
            }
            if (best == current) break;
            chain.push_back(best);
            current = best, current_distance = best_distance;
        }
        return chain;
    }

    std::size_t draw(std::size_t node) {
        if (drawn.size() == wanted) return none_k;
        usearch_key_t const key = index.entries[node].key;
        if (shown(key) || drawn_at.contains(node)) return none_k;
        std::size_t const at = drawn.size();
        drawn_at.emplace(node, at);
        drawn.push_back(node);
        keys[at] = key;
        std::vector<std::size_t> const entries = nodes_of(key);
        entry_ordinals[at] = static_cast<std::size_t>(std::find(entries.begin(), entries.end(), node) -
                                                      entries.begin());
        if (!nearest_shown_count) return at;
        f32_t* const kept_distances = nearest_shown_distances.data() + at * sample_nearest_shown_k;
        std::size_t* const kept_rows = nearest_shown.data() + at * sample_nearest_shown_k;
        std::size_t kept = 0;
        for (std::size_t anchor = 0; anchor != shown_nodes.size(); ++anchor) {
            f32_t const reach = distance(node, shown_nodes[anchor]);
            if (kept == nearest_shown_count && !(reach < kept_distances[kept - 1])) continue;
            std::size_t slot = kept == nearest_shown_count ? kept - 1 : kept++;
            for (; slot && reach < kept_distances[slot - 1]; --slot)
                kept_distances[slot] = kept_distances[slot - 1], kept_rows[slot] = kept_rows[slot - 1];
            kept_distances[slot] = reach, kept_rows[slot] = anchor;
        }
        return at;
    }

    std::size_t draw_nearest(std::size_t node) {
        std::size_t const at = draw(node);
        if (at == none_k) return none_k;
        usearch_key_t parent = keys[at];
        f32_t parent_distance = 0, best = std::numeric_limits<f32_t>::infinity();
        if (nearest_shown_count) {
            best = parent_distance = nearest_shown_distances[at * sample_nearest_shown_k];
            parent = shown_key(shown_offsets[nearest_shown[at * sample_nearest_shown_k]]);
        }
        for (std::size_t earlier = 0; earlier != at; ++earlier)
            if (f32_t const reach = distance(node, drawn[earlier]); reach < best)
                best = parent_distance = reach, parent = keys[earlier];
        parents[at] = parent;
        parent_distances[at] = parent_distance;
        return at;
    }

    void walk_between(std::size_t from, std::size_t toward) {
        std::size_t previous = from;
        for (std::size_t step : walk(from, toward)) {
            if (drawn.size() == wanted) break;
            if (std::size_t const at = draw(step); at != none_k)
                parents[at] = index.entries[previous].key, parent_distances[at] = distance(previous, step);
            previous = step;
        }
    }

    void file_root(std::size_t node, std::size_t at) {
        roots.push_back(node);
        root_draws.push_back(at);
        root_radii.push_back(std::numeric_limits<f32_t>::infinity());
    }

    /** Attaches root @p root to its nearest earlier one, lowest first on ties, exactly as the
     *  library itself does. */
    void attach_root(std::size_t root) {
        if (!root) return;
        std::vector<f32_t> reach(root);
        for (std::size_t earlier = 0; earlier != root; ++earlier)
            reach[earlier] = distance(roots[root], roots[earlier]);
        std::size_t nearest = 0;
        for (std::size_t earlier = 1; earlier != root; ++earlier)
            if (reach[earlier] < reach[nearest]) nearest = earlier;
        root_radii[root] = std::min(root_radii[root], reach[nearest]);
        root_radii[nearest] = std::min(root_radii[nearest], reach[nearest]);
        walk_between(roots[nearest], roots[root]);
    }

    /** The library's passes: roots, the walks attaching and leaving them, then the regions around
     *  them, under a doubling quota that never reads @c wanted but to stop. */
    void draw_passes() {
        for (std::size_t row = 0; row != arguments.shown_count; ++row) {
            if (!shown_rows.emplace(uuid_of(shown_key(row)), row).second) continue;
            std::vector<std::size_t> const entries = nodes_of(shown_key(row));
            if (entries.empty()) continue;
            shown_nodes.push_back(entries.front());
            shown_offsets.push_back(row);
        }
        nearest_shown_count = std::min(sample_nearest_shown_k, shown_nodes.size());
        nearest_shown.resize(wanted * sample_nearest_shown_k);
        nearest_shown_distances.resize(wanted * sample_nearest_shown_k);
        if (!wanted) return;
        for (std::size_t row = 0; row != arguments.focus_count; ++row) {
            usearch_key_t const key = focus_key(row);
            std::vector<std::size_t> const entries = nodes_of(key);
            if (auto const found = shown_rows.find(uuid_of(key)); found != shown_rows.end()) {
                if (!entries.empty() && shown_roots.insert(found->second).second) file_root(entries.front(), none_k);
                continue;
            }
            for (std::size_t node : entries)
                if (std::size_t const at = draw_nearest(node); at != none_k) file_root(node, at);
        }
        std::size_t position = 0;
        for (std::size_t quota = 1, attached = 0;;) {
            std::size_t const before = drawn.size();
            if (!arguments.focus_count)
                while (roots.size() != quota && position != limit && drawn.size() != wanted) {
                    std::size_t const node = static_cast<std::size_t>(
                        seeded_bijection(position++, limit, arguments.seed));
                    if (std::size_t const at = draw_nearest(node); at != none_k) file_root(node, at);
                }
            std::size_t const first_new = attached;
            for (; attached != roots.size(); ++attached) attach_root(attached);
            if (nearest_shown_count)
                for (std::size_t root = first_new; root != roots.size(); ++root)
                    if (root_draws[root] != none_k)
                        walk_between(roots[root],
                                     shown_nodes[nearest_shown[root_draws[root] * sample_nearest_shown_k]]);
            for (std::size_t root = 0; root != roots.size() && drawn.size() != wanted; ++root)
                draw_region(roots[root], root_radii[root], quota);
            if (drawn.size() == wanted) break;
            std::size_t const next = std::min(2 * quota, drawn.size() + 1);
            if (drawn.size() == before && next == quota) break;
            quota = next;
        }
    }

    void draw_region(std::size_t root, f32_t radius, std::size_t quota) {
        std::size_t const room = std::min(visits, quota * (visits / wanted));
        std::vector<std::size_t> visited {root};
        std::vector<f32_t> visited_distances {0};
        std::unordered_set<std::size_t> reached {root};
        for (std::size_t head = 0; head != visited.size(); ++head) {
            if (visited_distances[head] > radius) continue;
            ++visited_members;
            for (std::size_t next : neighbors(visited[head]))
                if (visited.size() != room && reached.insert(next).second) {
                    visited.push_back(next);
                    visited_distances.push_back(distance(root, next));
                }
        }
        // Rank, then node: the library's order, its one shard leaving the node to break ties.
        std::vector<std::pair<u64_t, std::size_t>> candidates;
        for (std::size_t visit = 1; visit != visited.size(); ++visit) {
            std::size_t const node = visited[visit];
            if (!(visited_distances[visit] <= radius) || drawn_at.contains(node)) continue;
            usearch_key_t const& key = index.entries[node].key;
            if (!shown(key)) candidates.push_back({splitmix64_at(arguments.seed, key_hash(key)), node});
        }
        std::size_t const kept = std::min(candidates.size(), quota);
        std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(kept), candidates.end());
        for (std::size_t candidate = 0; candidate != kept; ++candidate) draw_nearest(candidates[candidate].second);
    }

    u64_t coordinates_seed(std::size_t at) const noexcept {
        return splitmix64_at(splitmix64_at(arguments.seed, key_hash(keys[at])), entry_ordinals[at]);
    }

    f32_t shown_scale() {
        std::size_t const pairs = std::min(shown_nodes.size(), sample_scaling_draws_k + 1) - 1;
        f64_t shown_sum = 0, measured = 0;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            f32_t const reach = distance(shown_nodes[pair], shown_nodes[pair + 1]);
            f32_t const* first = shown_coordinates_of(shown_offsets[pair]);
            f32_t const* second = shown_coordinates_of(shown_offsets[pair + 1]);
            f64_t squares = 0;
            for (std::size_t column = 0; column != 3; ++column) {
                f64_t const offset = f64_t {first[column]} - f64_t {second[column]};
                squares += offset * offset;
            }
            shown_sum += std::sqrt(squares), measured += reach;
        }
        return shown_sum > 0 && measured > 0 ? static_cast<f32_t>(shown_sum / measured) : 1.0f;
    }

    void place_among_shown(std::size_t at, f32_t scale) {
        std::array<f32_t, sample_nearest_shown_k * 3> anchor_coordinates;
        std::array<f32_t, sample_nearest_shown_k> distances;
        for (std::size_t anchor = 0; anchor != nearest_shown_count; ++anchor) {
            std::size_t const kept = at * sample_nearest_shown_k + anchor;
            f32_t const* shown_at = shown_coordinates_of(shown_offsets[nearest_shown[kept]]);
            for (std::size_t column = 0; column != 3; ++column)
                anchor_coordinates[anchor * 3 + column] = shown_at[column];
            distances[anchor] = nearest_shown_distances[kept] * scale;
        }
        sample_gauss_newton(anchor_coordinates.data(), distances.data(), nearest_shown_count, coordinates_seed(at),
                            coordinates + at * 3);
    }

    void scale_classically() {
        std::size_t const anchors = std::min(drawn.size(), sample_scaling_draws_k);
        std::vector<f32_t> centered(anchors * anchors), vector(anchors), product(anchors);
        for (std::size_t row = 0; row != anchors; ++row)
            for (std::size_t column = 0; column != anchors; ++column) {
                f32_t const reach = distance(drawn[row], drawn[column]);
                centered[row * anchors + column] = reach * reach;
            }
        f32_t total = 0;
        for (std::size_t row = 0; row != anchors; ++row) {
            f32_t sum = 0;
            for (std::size_t column = 0; column != anchors; ++column) {
                f32_t const square = 0.5f * (centered[row * anchors + column] + centered[column * anchors + row]);
                sum += square;
            }
            product[row] = sum / static_cast<f32_t>(anchors);
            total += product[row];
        }
        total /= static_cast<f32_t>(anchors);
        for (std::size_t row = 0; row != anchors; ++row)
            for (std::size_t column = row; column != anchors; ++column) {
                f32_t const square = 0.5f * (centered[row * anchors + column] + centered[column * anchors + row]);
                f32_t const entry = -0.5f * (square - product[row] - product[column] + total);
                centered[row * anchors + column] = centered[column * anchors + row] = entry;
            }
        for (std::size_t axis = 0; axis != 3; ++axis) {
            for (std::size_t row = 0; row != anchors; ++row)
                vector[row] = 2.0f * static_cast<f32_t>(
                                         uniform_unit_f64(splitmix64_at(arguments.seed, axis * anchors + row))) -
                              1.0f;
            auto const multiply = [&]() noexcept {
                f32_t squares = 0;
                for (std::size_t row = 0; row != anchors; ++row) {
                    f32_t sum = 0;
                    for (std::size_t column = 0; column != anchors; ++column)
                        sum += centered[row * anchors + column] * vector[column];
                    product[row] = sum, squares += sum * sum;
                }
                return std::sqrt(squares);
            };
            for (std::size_t iteration = 0; iteration != 64; ++iteration) {
                f32_t const norm = multiply();
                if (!(norm > 0)) break;
                for (std::size_t row = 0; row != anchors; ++row) vector[row] = product[row] / norm;
            }
            // The Rayleigh quotient keeps the sign a norm would lose, so deflation subtracts the
            // pair with the right sign.
            multiply();
            f32_t eigenvalue = 0;
            for (std::size_t row = 0; row != anchors; ++row) eigenvalue += vector[row] * product[row];
            f32_t const spread = std::sqrt(std::max(eigenvalue, 0.0f));
            for (std::size_t row = 0; row != anchors; ++row) {
                coordinates[row * 3 + axis] = vector[row] * spread;
                for (std::size_t column = 0; column != anchors; ++column)
                    centered[row * anchors + column] -= eigenvalue * vector[row] * vector[column];
            }
        }

        std::array<f32_t, sample_nearest_shown_k * 3> anchor_coordinates;
        std::array<f32_t, sample_nearest_shown_k> kept_distances;
        std::size_t const nearest = std::min(sample_nearest_shown_k, anchors);
        for (std::size_t at = anchors; at != drawn.size(); ++at) {
            std::array<std::size_t, sample_nearest_shown_k> kept_anchors;
            std::size_t kept = 0;
            for (std::size_t anchor = 0; anchor != anchors; ++anchor) {
                f32_t const reach = distance(drawn[at], drawn[anchor]);
                if (kept == nearest && !(reach < kept_distances[kept - 1])) continue;
                std::size_t slot = kept == nearest ? kept - 1 : kept++;
                for (; slot && reach < kept_distances[slot - 1]; --slot)
                    kept_distances[slot] = kept_distances[slot - 1], kept_anchors[slot] = kept_anchors[slot - 1];
                kept_distances[slot] = reach, kept_anchors[slot] = anchor;
            }
            for (std::size_t anchor = 0; anchor != kept; ++anchor)
                for (std::size_t column = 0; column != 3; ++column)
                    anchor_coordinates[anchor * 3 + column] = coordinates[kept_anchors[anchor] * 3 + column];
            sample_gauss_newton(anchor_coordinates.data(), kept_distances.data(), kept, coordinates_seed(at),
                                coordinates + at * 3);
        }
    }

    void assign_coordinates(coordinates_kind_ const& coordinates_kind) {
        if constexpr (requires { coordinates_kind.coordinates_of(index.entries[0].payload, coordinates); })
            for (std::size_t at = 0; at != drawn.size(); ++at)
                coordinates_kind.coordinates_of(index.entries[drawn[at]].payload, coordinates + at * 3);
        else if (nearest_shown_count) {
            f32_t const scale = shown_scale();
            for (std::size_t at = 0; at != drawn.size(); ++at) place_among_shown(at, scale);
        }
        else if constexpr (requires { coordinates_kind.vector_of(index.entries[0].payload); })
            for (std::size_t at = 0; at != drawn.size(); ++at) {
                std::vector<f32_t> const row = coordinates_kind.vector_of(index.entries[drawn[at]].payload);
                sample_projection(row.data(), row.size(), arguments.seed, coordinates + at * 3);
            }
        else if (!drawn.empty()) scale_classically();
    }
};

/** Whether a sample's @p arguments name their shown keys and focus as the header asks; true once
 *  @p outcome records why not. */
template <typename arguments_>
bool reject_sample(arguments_ const& arguments, usearch_outcome_t* outcome) {
    usearch_rows_view_t const& coordinates = arguments.shown_coordinates;
    return reject_if(
               (arguments.shown_count && !arguments.shown.data) || (arguments.focus_count && !arguments.focus.data),
               usearch_status_invalid_argument_k, "The shown keys and focus must name keys", outcome) ||
           reject_if((arguments.shown.stride && arguments.shown.stride < sizeof(usearch_key_t)) ||
                         (arguments.focus.stride && arguments.focus.stride < sizeof(usearch_key_t)),
                     usearch_status_unsupported_layout_k, "The key stride is shorter than a key", outcome) ||
           reject_if(arguments.shown_count &&
                         (!coordinates.data || coordinates.columns != 3 || coordinates.scalar != usearch_scalar_f32_k ||
                          (coordinates.stride && coordinates.stride < 3 * sizeof(f32_t))),
                     usearch_status_invalid_argument_k, "The shown coordinates must be three f32 values per key",
                     outcome);
}

/** The body of every modality's @c sample: resets the outputs, checks the shown keys and focus, and
 *  draws up to @c wanted entries as @ref sample_plan does, writing them to @p arena only once every
 *  draw is placed, so a cancelled call leaves them reset. */
template <typename arguments_, typename handle_, typename score_, typename coordinates_kind_>
void draw_sample(arguments_ const& arguments, handle_& index, score_ const& score,
                 coordinates_kind_ const& coordinates_kind, usearch_outcome_t* outcome) {
    reset(arguments.stats);
    *arguments.keys = {};
    *arguments.entry_ordinals = nullptr;
    *arguments.parents = {};
    *arguments.coordinates = {};
    *arguments.written = 0;
    if (reject_sample(arguments, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(
        arguments.arena, arguments.outcome,
        {arguments.shown.data, arguments.shown_coordinates.data, arguments.focus.data}, outcome);
    if (!arena) return;
    std::shared_lock readers(index.mutex);
    std::size_t const wanted = arguments.wanted;
    std::size_t const expansion = arguments.expansion_factor ? arguments.expansion_factor : sample_expansion_factor_k;
    usearch_key_t* const keys = carve<usearch_key_t>(*arena, wanted, outcome);
    std::size_t* const entry_ordinals = keys ? carve<std::size_t>(*arena, wanted, outcome) : nullptr;
    usearch_key_t* const parents = entry_ordinals ? carve<usearch_key_t>(*arena, wanted, outcome) : nullptr;
    usearch_distance_t* const scores = parents ? carve<usearch_distance_t>(*arena, wanted, outcome) : nullptr;
    std::size_t* const counts = scores ? carve<std::size_t>(*arena, wanted, outcome) : nullptr;
    f32_t* const coordinates = counts ? carve<f32_t>(*arena, wanted * 3, outcome) : nullptr;
    if (!coordinates) return;
    std::vector<f32_t> parent_distances(wanted);
    sample_plan<arguments_, handle_, score_, coordinates_kind_> plan {
        arguments, index,          score,   index.entries.size(),    wanted,     wanted * expansion,
        keys,      entry_ordinals, parents, parent_distances.data(), coordinates};
    plan.draw_passes();
    plan.assign_coordinates(coordinates_kind);
    std::size_t const written = plan.drawn.size();
    if (arguments.stats)
        arguments.stats->visited_members += plan.visited_members,
            arguments.stats->computed_distances += plan.computed_distances;
    if (reject_if(written && !keep_going(arguments.progress, written, written), usearch_status_cancelled_k,
                  "The caller's handler stopped the work", outcome))
        return;
    for (std::size_t at = 0; at != written; ++at) {
        scores[at] = parent_distances[at];
        // The first draw of a call without shown points hangs from itself, so it has no parent.
        bool const own = !std::memcmp(&parents[at], &keys[at], sizeof(usearch_key_t)) && parent_distances[at] == 0;
        counts[at] = own ? 0 : 1;
    }
    if (!written) return;
    *arguments.keys = {keys, sizeof(usearch_key_t)};
    *arguments.entry_ordinals = entry_ordinals;
    *arguments.parents = {parents, scores, counts, 1};
    *arguments.coordinates = {coordinates, 3 * sizeof(f32_t), 3, usearch_scalar_f32_k};
    *arguments.written = written;
}

#pragma endregion Sampling

#pragma region Splitting

/** Where @p slot falls in the sequence @p seed names: SplitMix64's output at that slot of the
 *  seed's stream, so equal seeds draw split roots in one order and distinct seeds in others. */
inline u64_t draw_rank(u64_t seed, std::size_t slot) noexcept {
    u64_t mixed = seed + (static_cast<u64_t>(slot) + 1) * 0x9E3779B97F4A7C15ull;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    return mixed ^ (mixed >> 31);
}

/** The entries below @p count that @p within admits, in the order @p seed draws them. */
template <typename within_>
std::vector<std::size_t> drawing_order(std::size_t count, u64_t seed, within_ const& within) {
    std::vector<std::size_t> order;
    for (std::size_t entry = 0; entry != count; ++entry)
        if (within(entry)) order.push_back(entry);
    // The slot breaks a tie between equal ranks, so the order is total and repeats exactly.
    std::sort(order.begin(), order.end(), [seed](std::size_t first, std::size_t second) {
        u64_t const first_rank = draw_rank(seed, first), second_rank = draw_rank(seed, second);
        return first_rank != second_rank ? first_rank < second_rank : first < second;
    });
    return order;
}

/** The part each node of @p index falls in, of @p parts: roots drawn from @p seed, each part grown
 *  breadth-first over the links from its root, one node per part per round and capped at an equal
 *  share, and every node no root reached falling to the smallest part. */
template <typename handle_, typename score_>
std::vector<std::size_t> partition_nodes(handle_ const& index, std::size_t parts, u64_t seed, score_ const& score) {
    std::size_t const nodes = index.entries.size();
    std::size_t const unassigned = std::numeric_limits<std::size_t>::max();
    std::size_t const cap = divide_round_up(nodes, parts);
    std::vector<std::size_t> part_of(nodes, unassigned), sizes(parts, 0), heads(parts, 0);
    std::vector<std::vector<std::size_t>> frontiers(parts);
    std::vector<std::size_t> const order = drawing_order(nodes, seed, [](std::size_t) { return true; });
    for (std::size_t part = 0; part != std::min(parts, nodes); ++part)
        part_of[order[part]] = part, sizes[part] = 1, frontiers[part].push_back(order[part]);
    for (bool grew = true; grew;) {
        grew = false;
        for (std::size_t part = 0; part != parts; ++part) {
            if (heads[part] == frontiers[part].size() || sizes[part] >= cap) continue;
            std::size_t const from = frontiers[part][heads[part]++];
            for (std::size_t linked : links_of(index, from, nodes, score))
                if (part_of[linked] == unassigned && sizes[part] < cap)
                    part_of[linked] = part, ++sizes[part], frontiers[part].push_back(linked);
            grew = true;
        }
    }
    for (std::size_t node = 0; node != nodes; ++node) {
        if (part_of[node] != unassigned) continue;
        std::size_t const smallest = static_cast<std::size_t>(std::min_element(sizes.begin(), sizes.end()) -
                                                              sizes.begin());
        part_of[node] = smallest, ++sizes[smallest];
    }
    return part_of;
}

/**
 *  @brief Moves every entry of @p index into @p receivers, part @c p of @ref partition_nodes
 *      moving to receiver @c p in storage order, and empties @p index; false once @p outcome
 *      records why not, having moved nothing.
 *
 *  Every receiver must be a distinct, empty handle other than @p index whose options
 *  @p same_kind accepts beside the source's. Keeping no edges, a part loses none to rebase.
 */
template <typename handle_, typename receiver_, typename score_, typename same_kind_>
bool split_into(handle_& index, std::span<receiver_* const> receivers, u64_t seed, score_ const& score,
                same_kind_ const& same_kind, usearch_outcome_t* outcome) {
    if (reject_read_only(index, outcome)) return false;
    if (reject_if(receivers.size() < 2, usearch_status_invalid_argument_k, "A split needs two receivers or more",
                  outcome))
        return false;
    for (std::size_t position = 0; position != receivers.size(); ++position) {
        receiver_* const receiver = receivers[position];
        if (reject_if(!receiver, usearch_status_invalid_argument_k, "A receiver must not be null", outcome) ||
            reject_if(static_cast<handle_ const*>(receiver) == &index, usearch_status_config_mismatch_k,
                      "An index cannot split into itself", outcome) ||
            reject_if(std::find(receivers.begin(), receivers.begin() + static_cast<std::ptrdiff_t>(position),
                                receiver) != receivers.begin() + static_cast<std::ptrdiff_t>(position),
                      usearch_status_config_mismatch_k, "A receiver is named twice", outcome))
            return false;
    }
    std::unique_lock source(index.mutex);
    std::vector<std::unique_lock<std::shared_mutex>> held;
    for (receiver_* receiver : receivers) {
        held.emplace_back(receiver->mutex);
        if (reject_if(!receiver->entries.empty() || static_cast<bool>(receiver->served),
                      usearch_status_config_mismatch_k, "A receiver must be an empty, mutable handle", outcome) ||
            reject_if(!same_kind(index.options, receiver->options), usearch_status_config_mismatch_k,
                      "A receiver must share the index's kind and geometry", outcome))
            return false;
    }
    std::vector<std::size_t> const part_of = partition_nodes(index, receivers.size(), seed, score);
    std::vector<std::size_t> sizes(receivers.size(), 0);
    for (std::size_t part : part_of) ++sizes[part];
    for (std::size_t part = 0; part != receivers.size(); ++part)
        if (reject_if(!receivers[part]->accommodate(sizes[part]), usearch_status_out_of_memory_k,
                      "A receiver could not accommodate its part", outcome))
            return false;
    for (std::size_t entry = 0; entry != part_of.size(); ++entry)
        receivers[part_of[entry]]->append(index.entries[entry].key, std::move(index.entries[entry].payload));
    index.reset();
    return true;
}

/** Refuses a refinement naming donors and receivers both, or receivers no split can fill: fewer
 *  than two, a null slot, or the index itself. */
template <typename arguments_>
bool reject_receivers(arguments_ const& arguments, usearch_outcome_t* outcome) {
    if (reject_if(arguments.count && arguments.receivers_count, usearch_status_invalid_argument_k,
                  "A refinement absorbs donors or splits into receivers, never both", outcome))
        return true;
    if (!arguments.receivers_count) return false;
    if (reject_if(arguments.receivers_count < 2 || !arguments.receivers, usearch_status_invalid_argument_k,
                  "A split needs two receivers or more", outcome))
        return true;
    for (std::size_t position = 0; position != arguments.receivers_count; ++position)
        if (reject_if(!arguments.receivers[position], usearch_status_invalid_argument_k, "A receiver must not be null",
                      outcome) ||
            reject_if(arguments.receivers[position] == arguments.index, usearch_status_config_mismatch_k,
                      "An index cannot split into itself", outcome))
            return true;
    return false;
}

/** The body of every modality's @c split: moves the entries as @ref split_into does, then reports a
 *  closing sweep over every receiver, which re-selects nothing in a stand-in keeping no edges. */
template <typename arguments_, typename handle_, typename score_, typename same_kind_>
void split_and_report(arguments_ const& arguments, handle_& index, score_ const& score, same_kind_ const& same_kind,
                      usearch_outcome_t* outcome) {
    using receiver_t = std::remove_pointer_t<std::remove_cvref_t<decltype(*arguments.receivers)>>;
    if (reject_if(arguments.count && !arguments.receivers, usearch_status_invalid_argument_k,
                  "The receivers must name handles", outcome))
        return;
    std::span<receiver_t* const> const receivers {arguments.receivers, arguments.receivers ? arguments.count : 0};
    if (!split_into(index, receivers, arguments.seed, score, same_kind, outcome)) return;
    std::size_t entries = 0, reached = 0;
    for (receiver_t* receiver : receivers) entries += receiver->entries.size(), reached += !receiver->entries.empty();
    bool cancelled = false;
    for (std::size_t entry = 0; entry != entries && !cancelled; ++entry)
        cancelled = !keep_going(arguments.progress, entry + 1, entries);
    if (arguments.stats) arguments.stats->visited_members = entries;
    if (arguments.report) arguments.report->reachability = {entries, entries, reached ? 1u : 0u, 0};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

#pragma endregion Splitting

#pragma region Files

/** The extent of a checkpoint, empty when that many entries cannot be addressed. Every
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

/** A path in the shape every verb takes it. */
inline usearch_bytes_view_t as_path(std::string const& text) noexcept {
    return {reinterpret_cast<std::uint8_t const*>(text.data()), text.size()};
}

/** Whether @p blob opens as a JSON object, which only a manifest does; a stand-in monolith opens
 *  with a connectivity word. */
inline bool is_manifest(bytes_view_t blob) noexcept {
    return blob.size() > 1 && static_cast<char>(blob[0]) == '{' && static_cast<char>(blob[1]) == '"';
}

/** The directory of @p path, its trailing separator included; empty for a bare file name. */
inline std::string directory_of(std::string_view path) {
    std::size_t const slash = path.find_last_of("/\\");
    return std::string(slash == std::string_view::npos ? std::string_view {} : path.substr(0, slash + 1));
}

/** The part paths a manifest at @p manifest_path names, each resolved beside it, in file order;
 *  empty once @p outcome records that it names none. A buffer manifest's entries carry offsets
 *  rather than paths and are passed over, as the engine passes over them. */
inline std::vector<std::string> manifest_parts(bytes_view_t manifest, usearch_bytes_view_t manifest_path,
                                               usearch_outcome_t* outcome) {
    std::string_view const json {reinterpret_cast<char const*>(manifest.data()), manifest.size()};
    std::string const directory = directory_of(as_text(manifest_path));
    std::vector<std::string> parts;
    for (std::size_t at = json.find("\"path\""); at != std::string_view::npos; at = json.find("\"path\"", at)) {
        at = json.find_first_not_of(" \t\r\n:", at + 6);
        if (at == std::string_view::npos || json[at] != '"') break;
        std::size_t const end = json.find('"', at + 1);
        if (end == std::string_view::npos) break;
        parts.push_back(directory + std::string(json.substr(at + 1, end - at - 1)));
        at = end + 1;
    }
    reject_if(parts.empty(), usearch_status_malformed_k, "The checkpoint lists no part", outcome);
    return parts;
}

/** Replaces a manifest in @p blob with its first part's bytes, where every schema question is
 *  answered; a monolith stays as read. */
inline bool read_first_part(std::vector<std::byte>& blob, usearch_bytes_view_t path, usearch_outcome_t* outcome) {
    if (!is_manifest(blob)) return true;
    std::vector<std::string> const parts = manifest_parts(blob, path, outcome);
    return !parts.empty() && read_file(as_path(parts.front()), blob, outcome);
}

/** Opens the checkpoint @p blob read from @p path through @p open: one monolith as it is, or every
 *  part a manifest names, each opened on its own and appended in file order into the first; null
 *  once @p outcome records why not. */
template <typename handle_, typename open_>
handle_* open_parts(bytes_view_t blob, usearch_bytes_view_t path, open_ const& open, usearch_outcome_t* outcome) {
    std::string_view const magic {"SAFEGRPH"};
    if (reject_if(blob.size() >= magic.size() && std::memcmp(blob.data(), magic.data(), magic.size()) == 0,
                  usearch_status_unsupported_k, "The stand-in reads its own image, not a SafeGraphs monolith", outcome))
        return nullptr;
    if (!is_manifest(blob)) return open(blob);
    std::vector<std::string> const parts = manifest_parts(blob, path, outcome);
    if (parts.empty()) return nullptr;
    handle_* index = nullptr;
    std::vector<std::byte> part_blob;
    for (std::string const& part : parts) {
        if (!read_file(as_path(part), part_blob, outcome)) break;
        handle_* const opened = open(part_blob);
        if (!opened) break;
        if (!index) {
            index = opened;
            continue;
        }
        bool const accommodated = index->accommodate(index->entries.size() + opened->entries.size());
        if (accommodated)
            for (auto& entry : opened->entries) index->append(entry.key, std::move(entry.payload));
        delete opened;
        if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The parts' entries could not be accommodated",
                      outcome))
            break;
    }
    if (!index) return nullptr;
    if (outcome->status != usearch_status_success_k) {
        delete index;
        return nullptr;
    }
    index->parts_opened = parts.size();
    return index;
}

/** @p path without its @c .safegraphs suffix, the stem the engine names parts under. */
inline std::string stem_of(std::string_view path) {
    if (path.ends_with(".safegraphs")) path.remove_suffix(std::string_view {".safegraphs"}.size());
    return std::string(path);
}

/** Part @p part of @p parts under @p stem, as <tt>stem-00001-of-00003.safegraphs</tt>. */
inline std::string part_path(std::string const& stem, std::size_t part, std::size_t parts) {
    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), "-%05zu-of-%05zu.safegraphs", part, parts);
    return stem + suffix;
}

/**
 *  @brief The body of every modality's @c save: one monolith at @p path for one shard, else one
 *      part per shard beside the manifest at @p path, each part the image @p serialize writes of
 *      a handle holding that shard's entries alone.
 *
 *  A cancelled save leaves the parts already written and no manifest naming them, so the
 *  checkpoint reads as absent rather than half-valid, as the headers promise.
 */
template <typename handle_, typename serialize_>
void save_parts(handle_ const& index, usearch_bytes_view_t path, usearch_progress_t progress,
                serialize_ const& serialize, usearch_outcome_t* outcome) {
    std::size_t const shards = index.shards();
    std::vector<std::byte> blob;
    if (shards == 1) {
        if (!serialize(index, blob, outcome)) return;
        // The blob is one chunk, so the handler's only say is whether it is written at all.
        if (!keep_going(progress, 0, 1))
            return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
        if (!write_file(path, blob, outcome)) return;
        keep_going(progress, 1, 1);
        return;
    }
    std::string const stem = stem_of(as_text(path));
    std::size_t const total = index.entries.size();
    std::size_t const per_shard = index.shard_capacity ? index.shard_capacity : divide_round_up(total, shards);
    std::string manifest = "{\"files\":[";
    for (std::size_t shard = 0; shard != shards; ++shard) {
        if (!keep_going(progress, shard, shards))
            return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
        handle_ part;
        part.options = index.options;
        std::size_t const begin = std::min(shard * per_shard, total), end = std::min(begin + per_shard, total);
        part.entries.assign(index.entries.begin() + static_cast<std::ptrdiff_t>(begin),
                            index.entries.begin() + static_cast<std::ptrdiff_t>(end));
        if (!serialize(part, blob, outcome)) return;
        std::string const name = part_path(stem, shard + 1, shards);
        if (!write_file(as_path(name), blob, outcome)) return;
        char entry[160];
        std::snprintf(entry, sizeof(entry), "%s{\"path\":\"%s\",\"bytes\":%zu,\"nodes\":%zu}", shard ? "," : "",
                      name.c_str() + directory_of(name).size(), blob.size(), end - begin);
        manifest += entry;
    }
    manifest += "]}";
    if (!write_file(path, bytes_view_t {reinterpret_cast<std::byte const*>(manifest.data()), manifest.size()}, outcome))
        return;
    keep_going(progress, shards, shards);
}

#pragma endregion Files

} // namespace stub
} // namespace usearch
} // namespace unum

struct usearch_executor_handle_t : us::stub::executor_t {};

#endif // UNUM_USEARCH_STUB_LIB_HPP
