/**
 *  @brief Machinery every modality's stand-in shares - errors, scalars, files, and the
 *      append-only store that keeps the headers' thread-safety promises.
 *
 *  @file test/stub/lib.hpp
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

#include <cmath>   // `std::ldexp`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `SIZE_MAX`
#include <cstdio>  // `std::fopen`, `std::fread`, `std::fwrite`
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
#include <type_traits>   // `std::is_nothrow_move_constructible_v`
#include <unordered_map> // `std::unordered_multimap`
#include <vector>        // `std::vector`

#include "usearch/types.h"
#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {
namespace stub {

#pragma region Errors

inline void propagate_success(usearch_error_t* error) {
    if (error) { error->status = usearch_success_k, error->message = nullptr; }
}

inline void propagate_error(usearch_error_t* error, usearch_status_t status, char const* message) {
    if (error) { error->status = status, error->message = message; }
}

/** @brief Reports @p status through @p error when @p wrong, and answers whether it did - the
 *      one shape every guard below takes, so a caller reads `if (reject_if(...)) return;`. */
inline bool reject_if(bool wrong, usearch_status_t status, char const* message, usearch_error_t* error) {
    if (!wrong) return false;
    propagate_error(error, status, message);
    return true;
}

/** @brief Zeroes the telemetry, which every header calls a pure output fully overwritten. */
template <typename stats_>
void reset(stats_* stats) {
    if (stats) *stats = {};
}

#pragma endregion Errors

#pragma region Scalars

/** @brief The stored width of one scalar, in bytes; zero names a kind the stand-ins reject. */
inline std::size_t scalar_bytes(usearch_scalar_t kind) {
    switch (kind) {
    case usearch_scalar_f64_k: return 8;
    case usearch_scalar_f32_k: return 4;
    case usearch_scalar_bf16_k: return 2;
    case usearch_scalar_f16_k: return 2;
    case usearch_scalar_i8_k: return 1;
    case usearch_scalar_u8_k: return 1;
    default: return 0;
    }
}

/** @brief Rejects a kind the stand-ins cannot read, rather than reading it as @c f32 - a wrong
 *      row-byte computation would otherwise pass. */
inline bool reject_scalar(usearch_scalar_t kind, usearch_error_t* error) {
    return reject_if(!scalar_bytes(kind), usearch_error_unknown_kind_k,
                     "the stand-in implements f64, f32, bf16, f16, i8 and u8 only", error);
}

inline double read_half(u16_t bits) {
    int const sign = (bits >> 15) & 1;
    int const exponent = (bits >> 10) & 0x1F;
    int const mantissa = bits & 0x3FF;
    double value;
    if (exponent == 0) { value = std::ldexp((double)mantissa, -24); }
    else if (exponent == 31) { value = mantissa ? NAN : INFINITY; }
    else { value = std::ldexp(1024.0 + mantissa, exponent - 25); }
    return sign ? -value : value;
}

inline u16_t write_half(double value) {
    u32_t const bits = std::bit_cast<u32_t>((float)value);
    u32_t const sign = (bits >> 16) & 0x8000;
    i32_t const exponent = (i32_t)((bits >> 23) & 0xFF) - 127 + 15;
    u32_t const mantissa = (bits >> 13) & 0x3FF;
    if (exponent <= 0) return (u16_t)sign;
    if (exponent >= 31) return (u16_t)(sign | 0x7C00);
    return (u16_t)(sign | ((u32_t)exponent << 10) | mantissa);
}

/** @brief A brain float is the high half of a single, so widening is a shift. */
inline double read_brain(u16_t bits) { return std::bit_cast<float>((u32_t)bits << 16); }

/** @brief Narrows to the high half, rounding half to even and keeping a NaN a NaN. */
inline u16_t write_brain(double value) {
    u32_t const bits = std::bit_cast<u32_t>((float)value);
    bool const is_nan = (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu);
    if (is_nan) return (u16_t)((bits >> 16) | 0x0040u);
    // The tie-breaking increment carries into the exponent on its own, so an overflow
    // saturates to infinity without a branch.
    u32_t const rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (u16_t)(rounded >> 16);
}

/** @brief The @p position -th element of @p row read as @c scalar_. */
template <typename scalar_>
scalar_ element_at(std::span<std::byte const> row, std::size_t position) {
    scalar_ const* typed = (scalar_ const*)row.data();
    return typed[position];
}

/** @brief Assigns the @p position -th element of @p row written as @c scalar_. */
template <typename scalar_>
void element_to(std::span<std::byte> row, std::size_t position, scalar_ value) {
    scalar_* typed = (scalar_*)row.data();
    typed[position] = value;
}

/** @brief Widens one stored scalar. Callers gate on @c scalar_bytes first, so the unreachable
 *      arms return zero rather than silently reinterpreting the row as another width. */
inline double read_scalar(std::span<std::byte const> row, std::size_t position, usearch_scalar_t kind) {
    switch (kind) {
    case usearch_scalar_f64_k: return element_at<double>(row, position);
    case usearch_scalar_f32_k: return element_at<float>(row, position);
    case usearch_scalar_bf16_k: return read_brain(element_at<u16_t>(row, position));
    case usearch_scalar_f16_k: return read_half(element_at<u16_t>(row, position));
    case usearch_scalar_i8_k: return element_at<i8_t>(row, position);
    case usearch_scalar_u8_k: return element_at<u8_t>(row, position);
    default: return 0;
    }
}

inline void write_scalar(std::span<std::byte> row, std::size_t position, usearch_scalar_t kind, double value) {
    switch (kind) {
    case usearch_scalar_f64_k: element_to<double>(row, position, value); break;
    case usearch_scalar_f32_k: element_to<float>(row, position, (float)value); break;
    case usearch_scalar_bf16_k: element_to<u16_t>(row, position, write_brain(value)); break;
    case usearch_scalar_f16_k: element_to<u16_t>(row, position, write_half(value)); break;
    case usearch_scalar_i8_k: element_to<i8_t>(row, position, (i8_t)value); break;
    case usearch_scalar_u8_k: element_to<u8_t>(row, position, (u8_t)value); break;
    default: break;
    }
}

/** @brief Widens one stored row into @p values, whose extent decides how much is read. */
inline void read_row(std::span<std::byte const> row, usearch_scalar_t kind, std::span<double> values) {
    for (std::size_t position = 0; position != values.size(); ++position)
        values[position] = read_scalar(row, position, kind);
}

/** @brief Quantizes @p values back into one stored row. */
inline void write_row(std::span<std::byte> row, usearch_scalar_t kind, std::span<double const> values) {
    for (std::size_t position = 0; position != values.size(); ++position)
        write_scalar(row, position, kind, values[position]);
}

/** @brief The @p row -th row of a strided matrix, as @p count elements of @c value_type_. */
template <typename value_type_>
std::span<value_type_> row_at(void* base, std::size_t row, std::size_t stride, std::size_t count) {
    return std::span<value_type_>((value_type_*)((std::byte*)base + row * stride), count);
}

/** @brief The read-only sibling, for the matrices the ABI hands over as pure inputs. */
template <typename value_type_>
std::span<value_type_ const> row_at(void const* base, std::size_t row, std::size_t stride, std::size_t count) {
    return std::span<value_type_ const>((value_type_ const*)((std::byte const*)base + row * stride), count);
}

inline bool same_key(usearch_key_t const& first, usearch_key_t const& second) {
    return std::memcmp(first.bytes, second.bytes, sizeof(first.bytes)) == 0;
}

#pragma endregion Scalars

#pragma region Storage

/** @brief One stored row: its key, and whatever the modality calls a payload. */
template <typename payload_>
struct entry {
    usearch_key_t key;
    payload_ payload;
};

/** @brief Hashes the sixteen opaque bytes, which the C struct carries no hasher for. */
struct key_hash_t {
    std::size_t operator()(usearch_key_t const& key) const noexcept {
        std::size_t mixed = 1469598103934665603ull;
        for (std::size_t position = 0; position != sizeof(key.bytes); ++position)
            mixed = (mixed ^ key.bytes[position]) * 1099511628211ull;
        return mixed;
    }
};

/** @brief Compares the sixteen opaque bytes, which the C struct carries no @c operator== for. */
struct key_equal_t {
    bool operator()(usearch_key_t const& first, usearch_key_t const& second) const noexcept {
        return same_key(first, second);
    }
};

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
    std::unordered_multimap<usearch_key_t, std::size_t, key_hash_t, key_equal_t> by_key;
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
        by_key.emplace(key, entries.size());
        entries.push_back({key, std::move(payload)});
    }
};

/**
 *  @brief Row positions under @p key, ascending - the insertion order every @c get promises.
 *      An unordered multimap keeps equivalent keys adjacent but in no stated order, so the
 *      ascending sort is what restores the contract rather than a tidiness.
 */
template <typename handle_>
std::vector<std::size_t> rows_under(handle_ const& index, usearch_key_t key) {
    std::vector<std::size_t> rows;
    auto const range = index.by_key.equal_range(key);
    for (auto position = range.first; position != range.second; ++position) rows.push_back(position->second);
    std::sort(rows.begin(), rows.end());
    return rows;
}

#pragma endregion Storage

/** @brief Writes @p value's own bytes at the head of @p target - the serializer's one primitive. */
template <typename value_>
void write_pod(std::span<std::byte> target, value_ const& value) {
    std::array<std::byte, sizeof(value_)> const raw = std::bit_cast<std::array<std::byte, sizeof(value_)>>(value);
    std::copy(raw.begin(), raw.end(), target.begin());
}

/** @brief Reads one @c value_ back from the head of @p source. */
template <typename value_>
value_ read_pod(std::span<std::byte const> source) {
    std::array<std::byte, sizeof(value_)> raw {};
    std::copy_n(source.begin(), sizeof(value_), raw.begin());
    return std::bit_cast<value_>(raw);
}

/** @brief Copies every byte of @p source to the head of @p target. */
inline void copy_bytes(std::span<std::byte> target, std::span<std::byte const> source) {
    std::copy(source.begin(), source.end(), target.begin());
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

/** @brief Owns a @c FILE* so an early return closes it; the stand-ins' only resource handle. */
class file_t {
    std::FILE* handle_ = nullptr;

  public:
    file_t(std::string const& path, char const* mode) : handle_(std::fopen(path.c_str(), mode)) {}
    file_t(file_t const&) = delete;
    file_t& operator=(file_t const&) = delete;
    ~file_t() {
        if (handle_) std::fclose(handle_);
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    std::FILE* stream() const noexcept { return handle_; }
};

inline std::string path_string(usearch_string_t path) { return std::string(path.data, path.length); }

inline bool read_file(usearch_string_t path, std::vector<std::byte>& blob, usearch_error_t* error) {
    file_t file(path_string(path), "rb");
    if (!file) {
        propagate_error(error, usearch_error_io_k, "the checkpoint could not be opened for reading");
        return false;
    }
    std::byte buffer[4096];
    for (std::size_t read = std::fread(buffer, 1, sizeof(buffer), file.stream()); read;
         read = std::fread(buffer, 1, sizeof(buffer), file.stream()))
        blob.insert(blob.end(), buffer, buffer + read);
    if (std::ferror(file.stream())) {
        propagate_error(error, usearch_error_io_k, "the checkpoint could not be read");
        return false;
    }
    return true;
}

inline bool write_file(usearch_string_t path, std::span<std::byte const> blob, usearch_error_t* error) {
    file_t file(path_string(path), "wb");
    if (!file) {
        propagate_error(error, usearch_error_io_k, "the checkpoint could not be opened for writing");
        return false;
    }
    if (std::fwrite(blob.data(), 1, blob.size(), file.stream()) != blob.size() || std::fflush(file.stream()) != 0) {
        propagate_error(error, usearch_error_io_k, "the checkpoint could not be written");
        return false;
    }
    return true;
}

#pragma endregion Files

} // namespace stub
} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_STUB_LIB_HPP
