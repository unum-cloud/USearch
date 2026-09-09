/**
 *  @brief Machinery every modality's stand-in shares - errors, scalars, files, and the
 *      append-only store that keeps the headers' thread-safety promises.
 *
 *  @file src_stub/lib.hpp
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
 *  The one project header this pulls in is @c usearch/utilities/types.hpp, for the sized-integer
 *  vocabulary the rest of the codebase writes in. It costs no submodule and no runtime.
 */
#ifndef UNUM_USEARCH_STUB_LIB_HPP
#define UNUM_USEARCH_STUB_LIB_HPP

#include <cmath>   // `std::ldexp`, `std::frexp`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `SIZE_MAX`
#include <cstring> // `std::memcmp`

#include <algorithm>     // `std::copy`, `std::sort`, `std::stable_sort`
#include <array>         // `std::array`
#include <bit>           // `std::bit_cast`
#include <functional>    // `std::function`
#include <mutex>         // `std::lock_guard`, `std::unique_lock`
#include <new>           // `std::bad_alloc`
#include <optional>      // `std::optional`
#include <shared_mutex>  // `std::shared_mutex`, `std::shared_lock`
#include <span>          // `std::span`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view` - a byte run read as text
#include <type_traits>   // `std::is_nothrow_move_constructible_v`
#include <unordered_map> // `std::unordered_map`, `std::unordered_multimap`
#include <vector>        // `std::vector`

#include "usearch/types.h"
#include "usearch/utilities/blocking_files.hpp" // `seekable_file_t` - the checkpoints' resource handle
#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {
namespace stub {

#pragma region Vocabulary

/**
 *  @brief The run read as text - a path or a spelling, which are bytes that happen to decode.
 *
 *  The one place the byte run is reinterpreted. Every verb that names a file or a closed-set
 *  member arrives here rather than casting at its own call site.
 */
inline std::string_view as_text(usearch_bytes_view_t bytes) {
    return {reinterpret_cast<char const*>(bytes.data), bytes.length};
}

/** @brief Whether @p name spells @p spelling exactly, with no terminator assumed on either side. */
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

/** @brief The writable twin, for the buffers a caller lends the engine to fill. */
inline bytes_span_t as_writable_bytes(usearch_bytes_span_t bytes) {
    return {reinterpret_cast<byte_t*>(bytes.data), bytes.length};
}

/** @brief The read-only twin, for the buffers a caller lends the engine to read. */
inline bytes_view_t as_readable_bytes(usearch_bytes_view_t bytes) {
    return {reinterpret_cast<byte_t const*>(bytes.data), bytes.length};
}

#pragma endregion Vocabulary

#pragma region Errors

inline void propagate_success(usearch_error_t* error) {
    if (error) { error->status = usearch_success_k, error->message = nullptr; }
}

/** @brief The one write path to @p error; the message is null exactly on success. */
inline void propagate_error(usearch_status_t status, usearch_error_t* error, char const* message) {
    if (error) *error = usearch_error_t {status, status == usearch_success_k ? nullptr : message};
}

/** @brief Reports @p status through @p error when @p wrong, and answers whether it did - the
 *      one shape every guard below takes, so a caller reads `if (reject_if(...)) return;`. */
inline bool reject_if(bool wrong, usearch_status_t status, char const* message, usearch_error_t* error) {
    if (!wrong) return false;
    propagate_error(status, error, message);
    return true;
}

/** @brief Zeroes the telemetry, which every header calls a pure output fully overwritten. */
template <typename stats_>
void reset(stats_* stats) {
    if (stats) *stats = {};
}

#pragma endregion Errors

#pragma region Runs

/** @brief Refuses a key run that names no storage; a batch's identities are never optional. */
inline bool reject_keys(usearch_keys_view_t keys, usearch_error_t* error) {
    return reject_if(!keys.data, usearch_error_malformed_input_k, "The key run must name storage", error);
}

/** @brief Refuses a key run that does not pair one key with every row it accompanies. */
inline bool reject_keys_pairing(usearch_keys_view_t keys, std::size_t rows, usearch_error_t* error) {
    if (reject_keys(keys, error)) return true;
    return reject_if(keys.count != rows, usearch_error_shape_mismatch_k,
                     "The key count does not match the row count it accompanies", error);
}

/** @brief The output twin of @c reject_keys, whose capacity bounds what a streaming verb writes. */
inline bool reject_keys_span(usearch_keys_span_t keys, usearch_error_t* error) {
    if (reject_if(!keys.data, usearch_error_malformed_input_k, "The key run must name storage", error)) return true;
    return reject_if(!keys.capacity, usearch_error_shape_mismatch_k, "A batch must hold at least one key", error);
}

/** @brief Refuses a ranked-output triple that names no storage, resolving its two pitches. */
inline bool reject_matches(usearch_matches_span_t const* matches, std::size_t& keys_stride, std::size_t& scores_stride,
                           usearch_error_t* error) {
    if (reject_if(!matches || !matches->keys || !matches->scores || !matches->counts, usearch_error_malformed_input_k,
                  "The matches span must name all three outputs", error))
        return true;
    if (reject_if(!matches->wanted, usearch_error_shape_mismatch_k, "A run must want at least one match", error))
        return true;
    keys_stride = matches->keys_stride_bytes ? matches->keys_stride_bytes : matches->wanted * sizeof(usearch_key_t);
    scores_stride = matches->scores_stride_bytes ? matches->scores_stride_bytes
                                                 : matches->wanted * sizeof(usearch_distance_t);
    return false;
}

#pragma endregion Runs

#pragma region Executors

/**
 *  @brief A stand-in's executor: a handle that exists, remembers what was asked for, and drives
 *      nothing - every verb runs on the thread that called it.
 *
 *  Keeping the handle real rather than ignoring the parameter is what lets a binding suite
 *  exercise the lifetime it will meet against the engine - spawn, pass, drain, free - and see a
 *  null or wrong-kind executor refused at the same place.
 */
struct executor_t {
    /** @brief Worker threads the caller asked for; honoured by remembering, never by spawning. */
    std::size_t threads = 1;
};

/** @brief Refuses an executor a verb cannot run on, reporting where the engine's dispatch does. */
inline bool reject_executor(usearch_executor_t executor, usearch_error_t* error) {
    return reject_if(!executor, usearch_error_device_k, "The executor's kind must match the handle's residency", error);
}

/** @brief Ticks the caller's handler and answers whether to carry on; a null handler always does. */
inline bool keep_going(usearch_progress_callback_t progress, void* progress_state, std::size_t completed,
                       std::size_t total) {
    return !progress || progress(completed, total, progress_state) == usearch_continue_k;
}

/** @brief Reports a handler's cancellation once a loop stopped short, and answers whether it did. */
inline bool reject_cancelled(std::size_t completed, std::size_t total, usearch_error_t* error) {
    return reject_if(completed < total, usearch_error_cancelled_k, "The caller's handler stopped the work", error);
}

#pragma endregion Executors

#pragma region Scalars

/** @brief The stored width of one value, in bytes; zero names a kind the stand-ins reject. */
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
    case usearch_scalar_u8_k: return 1;
    case usearch_scalar_u4x2_k: return 1;
    case usearch_scalar_u1x8_k: return 1;
    default: return 0;
    }
}

/** @brief How many logical dimensions share one stored value, as the @c xN suffixes spell. */
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

/** @brief Rejects a kind that names no scalar, rather than reading it as @c f32 - a wrong
 *      row-byte computation would otherwise pass. */
inline bool reject_scalar(usearch_scalar_t kind, usearch_error_t* error) {
    return reject_if(!scalar_bytes(kind), usearch_error_unknown_kind_k, "The kind names no scalar", error);
}

inline double read_half(u16_t bits) {
    int const sign = (bits >> 15) & 1;
    int const exponent = (bits >> 10) & 0x1F;
    int const mantissa = bits & 0x3FF;
    double value;
    if (exponent == 0) { value = std::ldexp(static_cast<double>(mantissa), -24); }
    else if (exponent == 31) { value = mantissa ? NAN : INFINITY; }
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

/** @brief A brain float is the high half of a single, so widening is a shift. */
inline double read_brain(u16_t bits) { return std::bit_cast<float>(static_cast<u32_t>(bits) << 16); }

/** @brief Narrows to the high half, rounding half to even and keeping a NaN a NaN. */
inline u16_t write_brain(double value) {
    u32_t const bits = std::bit_cast<u32_t>(static_cast<float>(value));
    bool const is_nan = (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu);
    if (is_nan) return static_cast<u16_t>((bits >> 16) | 0x0040u);
    // The tie-breaking increment carries into the exponent on its own, so an overflow
    // saturates to infinity without a branch.
    u32_t const rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<u16_t>(rounded >> 16);
}

/** @brief Which of a mini-float's codes name something other than a number. */
enum class minifloat_specials_t {
    /** @brief Every code is a number, as both six-bit kinds are. */
    none_k,
    /** @brief The all-ones code of either sign is a NaN, and there are no infinities. */
    nan_only_k,
    /** @brief A saturated exponent is an infinity, or a NaN once its mantissa is set. */
    inf_and_nan_k,
};

/** @brief One mini-float's shape, which is all a codec needs beyond the byte itself. */
struct minifloat_t {
    /** @brief Exponent bits, which fix both the bias and the dynamic range. */
    unsigned exponent_bits;
    /** @brief Mantissa bits, which fix the step between neighbouring codes. */
    unsigned mantissa_bits;
    /** @brief Which codes are not numbers, since only the eight-bit kinds reserve any. */
    minifloat_specials_t specials;
};

/** @brief The shape of @p kind, or a zeroed one for a kind that is not a mini-float. */
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
        return mantissa ? NAN : sign * INFINITY;
    if (exponent == saturated && mantissa == mantissa_span - 1 && format.specials == minifloat_specials_t::nan_only_k)
        return NAN;
    // A zero exponent field is the subnormal step, so the arms differ only in the implied leading bit.
    unsigned const significand = exponent ? mantissa + mantissa_span : mantissa;
    int const scale = static_cast<int>(exponent ? exponent : 1) - bias - static_cast<int>(format.mantissa_bits);
    return sign * std::ldexp(static_cast<double>(significand), scale);
}

/** @brief Narrows to the nearest code, ties to even, saturating rather than wrapping. */
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

/** @brief Widens a bit-set row, whose eight dimensions per byte ascend from the low bit. */
inline void widen_bits(bytes_view_t row, std::span<double> values) {
    u8_t const* packed = static_cast<u8_t const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position)
        values[position] = (packed[position / 8] >> (position % 8)) & 1 ? 1.0 : 0.0;
}

/** @brief Narrows to a bit-set row, where every value that is not a zero sets its bit. */
inline void narrow_bits(bytes_span_t row, std::span<double const> values) {
    u8_t* packed = static_cast<u8_t*>(static_cast<void*>(row.data()));
    // The row's own extent is the packed byte count every caller sized through `row_extent`, so
    // the partial byte a ragged width leaves is cleared without rounding the count up again here.
    std::fill_n(packed, row.size(), u8_t {0});
    for (std::size_t position = 0; position != values.size(); ++position)
        if (values[position] != 0) packed[position / 8] |= static_cast<u8_t>(1u << (position % 8));
}

/** @brief Widens a nibble row, whose even dimensions sit in the low half of each byte. */
template <typename nibble_>
void widen_nibbles(bytes_view_t row, std::span<double> values) {
    u8_t const* packed = static_cast<u8_t const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position) {
        int const nibble = (packed[position / 2] >> (position % 2 * 4)) & 0xF;
        // A signed nibble carries its sign in the high bit, so the upper half of the range wraps.
        values[position] = std::is_signed_v<nibble_> && nibble >= 8 ? nibble - 16 : nibble;
    }
}

/** @brief Narrows into a nibble row, clamping to the four bits each dimension is given. */
template <typename nibble_>
void narrow_nibbles(bytes_span_t row, std::span<double const> values) {
    double const floor_value = std::is_signed_v<nibble_> ? -8.0 : 0.0;
    double const ceiling_value = std::is_signed_v<nibble_> ? 7.0 : 15.0;
    u8_t* packed = static_cast<u8_t*>(static_cast<void*>(row.data()));
    std::fill_n(packed, row.size(), u8_t {0});
    for (std::size_t position = 0; position != values.size(); ++position) {
        int const clamped = static_cast<int>(std::clamp(values[position], floor_value, ceiling_value));
        packed[position / 2] |= static_cast<u8_t>((clamped & 0xF) << (position % 2 * 4));
    }
}

/** @brief Widens a whole row stored as @c scalar_, converting each element to a double. */
template <typename scalar_>
void widen_into(bytes_view_t row, std::span<double> values) {
    scalar_ const* typed = static_cast<scalar_ const*>(static_cast<void const*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position) values[position] = typed[position];
}

/** @brief Narrows a whole row of doubles back into storage as @c scalar_. */
template <typename scalar_>
void narrow_from(bytes_span_t row, std::span<double const> values) {
    scalar_* typed = static_cast<scalar_*>(static_cast<void*>(row.data()));
    for (std::size_t position = 0; position != values.size(); ++position)
        typed[position] = static_cast<scalar_>(values[position]);
}

/**
 *  @brief Widens one stored row into @p values, whose extent decides how much is read. The kind is
 *      fixed for the whole row, so it is switched on once here rather than once per element.
 *      Callers gate on @c scalar_bytes first, so an unreachable arm leaves @p values untouched
 *      rather than silently reinterpreting the row as another width.
 */
inline void read_row(bytes_view_t row, usearch_scalar_t kind, std::span<double> values) {
    switch (kind) {
    case usearch_scalar_f64_k: return widen_into<double>(row, values);
    case usearch_scalar_f32_k: return widen_into<float>(row, values);
    case usearch_scalar_i8_k: return widen_into<i8_t>(row, values);
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

/** @brief Quantizes @p values back into one stored row, switching on the kind once. */
inline void write_row(bytes_span_t row, usearch_scalar_t kind, std::span<double const> values) {
    switch (kind) {
    case usearch_scalar_f64_k: return narrow_from<double>(row, values);
    case usearch_scalar_f32_k: return narrow_from<float>(row, values);
    case usearch_scalar_i8_k: return narrow_from<i8_t>(row, values);
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

/** @brief The @p row -th row of a strided matrix, as @p count elements of @c value_type_. */
template <typename value_type_>
std::span<value_type_> row_at(void* base, std::size_t row, std::size_t stride, std::size_t count) {
    std::byte* const at = static_cast<std::byte*>(base) + row * stride;
    return std::span<value_type_>(static_cast<value_type_*>(static_cast<void*>(at)), count);
}

/** @brief The read-only sibling, for the matrices the ABI hands over as pure inputs. */
template <typename value_type_>
std::span<value_type_ const> row_at(void const* base, std::size_t row, std::size_t stride, std::size_t count) {
    std::byte const* const at = static_cast<std::byte const*>(base) + row * stride;
    return std::span<value_type_ const>(static_cast<value_type_ const*>(static_cast<void const*>(at)), count);
}

#pragma endregion Scalars

#pragma region Storage

/**
 *  @brief Unwraps one of the ABI's opaque handles. Every verb's first act, and the only place a
 *      handle's cast is written - the C headers declare all of them as @c void*.
 */
template <typename handle_type_>
handle_type_& borrowed(void* handle) noexcept {
    return *static_cast<handle_type_*>(handle);
}

/** @brief One stored row: its key, and whatever the modality calls a payload. */
template <typename payload_>
struct entry {
    usearch_key_t key;
    payload_ payload;
};

/**
 *  @brief The project's own key type, which already knows how to hash and compare itself.
 *      Layout-identical to @c usearch_key_t, so the conversion is a reinterpretation.
 */
inline uuid_t keyed(usearch_key_t const& key) noexcept { return std::bit_cast<uuid_t>(key); }

/**
 *  @brief One handle's whole state. Readers hold @c mutex shared and writers exclusively, which
 *      is what keeps a search overlapping an insertion.
 *
 *  A cursor hands out views borrowing a payload's own bytes, and those survive a later insertion
 *  because reallocating @c entries moves each row without moving the buffer it owns - true only
 *  while the payload's move constructor stays @c noexcept, which the modality files assert.
 */
template <typename payload_, typename options_>
struct handle {
    /** @brief The options the handle was initialized with, or a checkpoint later adopted. */
    options_ options {};
    /** @brief The rows themselves, in insertion order. */
    std::vector<entry<payload_>> entries;
    /** @brief Row positions under each key, so a lookup is not a scan; keys need not be unique. */
    std::unordered_multimap<uuid_t, std::size_t> by_key;
    /** @brief Held shared by every reader, exclusively by every mutation. */
    mutable std::shared_mutex mutex;

    /** @brief Drops every row, which @c clear promises without touching the frozen options. */
    void reset() noexcept { entries.clear(), by_key.clear(); }

    /** @brief Sizes both containers for @p total entries, answering whether the memory came. */
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

    /** @brief Appends one row and indexes its key, the only way a row is ever added. */
    void append(usearch_key_t key, payload_ payload) {
        by_key.emplace(keyed(key), entries.size());
        entries.push_back({key, std::move(payload)});
    }

    /** @brief Drops every row published past @p watermark, which an unwound batch reverts to. */
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
bool reject_oversubscribed(handle_ const& index, usearch_keys_view_t keys, std::size_t values_per_key,
                           usearch_error_t* error) {
    std::size_t const bound = values_per_key ? values_per_key : 1;
    std::unordered_map<uuid_t, std::size_t> admitted;
    for (std::size_t row = 0; row != keys.count; ++row) {
        uuid_t const key = keyed(keys.data[row]);
        std::size_t const held = index.by_key.count(key) + ++admitted[key];
        if (reject_if(held > bound, usearch_error_overflow_k, "A key would hold more entries than the options admit",
                      error))
            return true;
    }
    return false;
}

/**
 *  @brief Row positions under @p key, ascending - the insertion order every @c get promises.
 *      An unordered multimap keeps equivalent keys adjacent but in no stated order, so the
 *      ascending sort is what restores the contract rather than a tidiness.
 */
template <typename handle_>
std::vector<std::size_t> rows_under(handle_ const& index, usearch_key_t key) {
    std::vector<std::size_t> rows;
    auto const range = index.by_key.equal_range(keyed(key));
    for (auto position = range.first; position != range.second; ++position) rows.push_back(position->second);
    std::sort(rows.begin(), rows.end());
    return rows;
}

#pragma endregion Storage

/** @brief Writes @p value's own bytes at the head of @p target - the serializer's one primitive. */
template <typename value_>
void write_pod(bytes_span_t target, value_ const& value) {
    std::array<std::byte, sizeof(value_)> const raw = std::bit_cast<std::array<std::byte, sizeof(value_)>>(value);
    std::copy(raw.begin(), raw.end(), target.begin());
}

/** @brief Reads one @c value_ back from the head of @p source. */
template <typename value_>
value_ read_pod(bytes_view_t source) {
    std::array<std::byte, sizeof(value_)> raw {};
    std::copy_n(source.begin(), sizeof(value_), raw.begin());
    return std::bit_cast<value_>(raw);
}

#pragma region Ranking

/** @brief Whether a filtered search admits one key. Named states, never a bare @c bool. */
enum class admission_t : bool { rejected_k, admitted_k };

/** @brief Wraps the ABI's predicate pair once, at the boundary; a null predicate admits every key. */
inline std::function<admission_t(usearch_key_t)> admitting(usearch_predicate_t predicate, void* state) {
    if (!predicate) return [](usearch_key_t) { return admission_t::admitted_k; };
    return [predicate, state](usearch_key_t key) {
        return predicate(key, state) ? admission_t::admitted_k : admission_t::rejected_k;
    };
}

/** @brief One scored pair, in the order a search reports them. */
struct match_t {
    /** @brief What the modality's kind scored the pair at; smaller is nearer. */
    double distance;
    /** @brief The key of the entry scored. */
    usearch_key_t key;
};

/**
 *  @brief Scores @p entries into @p ranked, nearest first. The sort is stable because every
 *      modality's docblock promises ties fall in insertion order.
 */
template <typename payload_>
void rank(std::span<entry<payload_> const> entries, std::function<admission_t(usearch_key_t)> const& admits,
          std::function<double(payload_ const&)> const& score, std::vector<match_t>& ranked) {
    ranked.clear();
    ranked.reserve(entries.size());
    for (entry<payload_> const& row : entries) {
        if (admits(row.key) == admission_t::rejected_k) continue;
        ranked.push_back({score(row.payload), row.key});
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](match_t const& left, match_t const& right) { return left.distance < right.distance; });
}

#pragma endregion Ranking

#pragma region Files

/**
 *  @brief The extent of a checkpoint, empty when that many rows cannot be addressed. Every
 *      modality's blob is a prefix followed by @p count entries of @p each bytes.
 */
inline std::optional<std::size_t> blob_bytes(std::size_t prefix, std::size_t count, std::size_t each) {
    if (each && count > (SIZE_MAX - prefix) / each) return std::nullopt;
    return prefix + count * each;
}

/**
 *  @brief Whether @p blob is long enough for @p count records of @p each bytes past @p header,
 *      reporting through @p error which way it fell short. Every modality's @c parse_prefix
 *      ends in this check; only the extent it computes differs.
 */
inline bool fits_blob(bytes_view_t blob, std::size_t header, std::size_t count, std::size_t each,
                      usearch_error_t* error) {
    std::optional<std::size_t> const needed = blob_bytes(header, count, each);
    if (reject_if(!needed, usearch_error_overflow_k, "The checkpoint claims more than can be addressed", error))
        return false;
    return !reject_if(blob.size() < *needed, usearch_error_malformed_input_k,
                      "The checkpoint is shorter than the entries it claims", error);
}

inline bool read_file(usearch_bytes_view_t path, std::vector<std::byte>& blob, usearch_error_t* error) {
    auto [file, opened] = seekable_file_t::open(as_text(path), file_mode_t::read_k);
    if (failed(opened)) {
        propagate_error(usearch_error_io_k, error, "The checkpoint could not be opened for reading");
        return false;
    }
    auto const [bytes, measured] = file.length();
    if (failed(measured)) {
        propagate_error(usearch_error_io_k, error, "The checkpoint could not be read");
        return false;
    }
    blob.resize(bytes);
    if (failed(file.read_at(0, bytes_span_t {blob}))) {
        propagate_error(usearch_error_io_k, error, "The checkpoint could not be read");
        return false;
    }
    return true;
}

inline bool write_file(usearch_bytes_view_t path, bytes_view_t blob, usearch_error_t* error) {
    auto [file, opened] = seekable_file_t::open(as_text(path), file_mode_t::write_k);
    if (failed(opened)) {
        propagate_error(usearch_error_io_k, error, "The checkpoint could not be opened for writing");
        return false;
    }
    if (failed(file.write_at(0, blob)) || failed(file.close())) {
        propagate_error(usearch_error_io_k, error, "The checkpoint could not be written");
        return false;
    }
    return true;
}

#pragma endregion Files

} // namespace stub
} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_STUB_LIB_HPP
