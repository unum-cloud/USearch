/**
 *  @file include/usearch/utilities/types.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Vector Search helper structures and types
 */
#pragma once
#include <cassert> // `assert` on the `to_span` and `compact_neighbors` contracts
#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memmove`, `std::memcmp`, `std::memcpy`

#include <array>       // `std::array` - a quad's keys and admissions
#include <atomic>      // `std::atomic`
#include <bit>         // `std::bit_cast` - hashing `uuid_t` as two words, `std::bit_width` - the bijection's span
#include <concepts>    // `std::convertible_to` - the string-view conversion's constraint
#include <functional>  // `std::hash` - specialized for `uuid_t` below, `std::less`
#include <limits>      // `std::numeric_limits` - the claim word's closed sentinel
#include <optional>    // `std::optional` - the identity a `uuid_t` may render
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::type_identity_t`, `std::conditional_t` - the kit cascades
#include <utility>     // `std::make_index_sequence` - a round's tail at its static extent

#include "usearch/types.h" // `usearch_status_t`, so the C++ mirror cannot drift from the ABI

/* Inferring target OS: Windows, macOS, Linux, FreeBSD, or WASI */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define USEARCH_OS_WINDOWS_ 1
#else
#define USEARCH_OS_WINDOWS_ 0
#endif
#if defined(__APPLE__) && defined(__MACH__)
#define USEARCH_OS_APPLE_ 1
#else
#define USEARCH_OS_APPLE_ 0
#endif
#if defined(__linux__)
#define USEARCH_OS_LINUX_ 1
#else
#define USEARCH_OS_LINUX_ 0
#endif
#if defined(__FreeBSD__)
#define USEARCH_OS_FREEBSD_ 1
#else
#define USEARCH_OS_FREEBSD_ 0
#endif
#if defined(__wasi__)
#define USEARCH_OS_WASI_ 1
#else
#define USEARCH_OS_WASI_ 0
#endif
#define USEARCH_OS_POSIX_ (USEARCH_OS_LINUX_ || USEARCH_OS_APPLE_ || USEARCH_OS_FREEBSD_)

/*  Compiling for x86: USEARCH_ARCH_X8664_
 *  @see Intel predefined macros: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-2/additional-predefined-macros.html */
#if !defined(USEARCH_ARCH_X8664_)
#if defined(__x86_64__) || defined(_M_X64)
#define USEARCH_ARCH_X8664_ 1
#else
#define USEARCH_ARCH_X8664_ 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(USEARCH_ARCH_X8664_)

/* Compiling for IBM Z: USEARCH_ARCH_S390X_ */
#if !defined(USEARCH_ARCH_S390X_)
#if defined(__s390x__)
#define USEARCH_ARCH_S390X_ 1
#else
#define USEARCH_ARCH_S390X_ 0
#endif // defined(__s390x__)
#endif // !defined(USEARCH_ARCH_S390X_)

/* Compiling for WASM: USEARCH_ARCH_WASM_ */
#if !defined(USEARCH_ARCH_WASM_)
#if defined(__wasm__) || defined(__EMSCRIPTEN__)
#define USEARCH_ARCH_WASM_ 1
#else
#define USEARCH_ARCH_WASM_ 0
#endif // defined(__wasm__) || defined(__EMSCRIPTEN__)
#endif // !defined(USEARCH_ARCH_WASM_)

/*  GPU backends compiled into this unit: USEARCH_ARCH_CUDA_, USEARCH_ARCH_ROCM_,
 *  USEARCH_ARCH_METAL_. CMake defines each from its `USEARCH_BUILD_*` option, since the toolchain
 *  alone cannot tell a dispatch unit that carries a vendor's variants from one that merely sees its
 *  headers. */
#if !defined(USEARCH_ARCH_CUDA_)
#define USEARCH_ARCH_CUDA_ 0
#endif
#if !defined(USEARCH_ARCH_ROCM_)
#define USEARCH_ARCH_ROCM_ 0
#endif
#if !defined(USEARCH_ARCH_METAL_)
#define USEARCH_ARCH_METAL_ 0
#endif

/*  The `alignas` width that keeps two threads' words off one cache line, picked from the target as
 *  ForkUnion picks `FORKUNION_DEFAULT_ALIGNMENT`; `ram_device.hpp` asserts the two agree. */
#if !defined(USEARCH_DEFAULT_ALIGNMENT)
#if USEARCH_ARCH_S390X_
#define USEARCH_DEFAULT_ALIGNMENT 256
#elif USEARCH_ARCH_WASM_
#define USEARCH_DEFAULT_ALIGNMENT 64
#else
#define USEARCH_DEFAULT_ALIGNMENT 128
#endif
#endif // !defined(USEARCH_DEFAULT_ALIGNMENT)

/** Pre-announced sister-library namespaces, so the project-wide shortcuts below are
 *  legal before any of their headers is included. */
namespace ashvardanian {
namespace numkong {}
namespace stringzilla {}
namespace forkunion {}
namespace smashtable {}
} // namespace ashvardanian

namespace unum::usearch {
namespace nk = ashvardanian::numkong;
namespace fu = ashvardanian::forkunion;
namespace sz = ashvardanian::stringzilla;
namespace st = ashvardanian::smashtable;
} // namespace unum::usearch

namespace unum::usearch::utilities {

using f64_t = double;
using f32_t = float;

using i64_t = std::int64_t;
using i32_t = std::int32_t;
using i16_t = std::int16_t;
using i8_t = std::int8_t;

using u64_t = std::uint64_t;
using u32_t = std::uint32_t;
using u16_t = std::uint16_t;
using u8_t = std::uint8_t;

/** One word of 64 packed single-bit lanes - the storage a membership bitset walks. An
 *  alias rather than an enumeration: only one of the seven atomic references admits an
 *  enumeration's width, and a bitset binds whichever one its CPU class names. */
using u1x64_t = u64_t;

/** The half-width sibling, for matrices a device addresses in 32-bit words. */
using u1x32_t = u32_t;

/** "usearch3" read as a little-endian word - the one magic fronting every USearch
 *  payload; the word right after it discriminates the layout. */
inline constexpr u64_t usearch_magic_k = 0x3368637261657375ull;

/** The 128-bit key default - the SafeGraphs @c uuid column, one UUID per node. */
struct uuid_t {
    std::byte bytes[16];

    bool operator==(uuid_t const& other) const noexcept { return std::memcmp(bytes, other.bytes, sizeof(bytes)) == 0; }
};
static_assert(std::is_trivially_copyable_v<uuid_t> && sizeof(uuid_t) == 16);

/** The @c uuid_t every binding renders the integer @p identity as: big-endian in the trailing
 *  eight bytes. */
constexpr uuid_t uuid_of(u64_t identity) noexcept {
    uuid_t key {};
    for (std::size_t byte = 0; byte != sizeof(identity); ++byte)
        key.bytes[sizeof(key.bytes) - 1 - byte] = static_cast<std::byte>(identity >> (byte * 8));
    return key;
}

/** The integer identity @ref uuid_of renders as @p key, or none when its leading eight bytes are
 *  not zero. */
constexpr std::optional<u64_t> identity_of(uuid_t const& key) noexcept {
    u64_t identity = 0;
    for (std::size_t byte = 0; byte != sizeof(key.bytes); ++byte) {
        if (byte < sizeof(identity) && key.bytes[byte] != std::byte {0}) return std::nullopt;
        identity = identity << 8 | static_cast<u64_t>(key.bytes[byte]);
    }
    return identity;
}

/** The single-byte scalar every allocator deals in - the unit of raw storage, and the C++
 *  counterpart of @c usearch_byte_t. */
using byte_t = std::byte;
static_assert(sizeof(byte_t) == sizeof(usearch_byte_t), "One raw byte on both sides of the C interface");

/** A mutable run of raw bytes - the staging & carve currency of writers. */
using bytes_span_t = std::span<byte_t>;

/** A read-only run of raw bytes - the currency of readers & mapped payloads. */
using bytes_view_t = std::span<byte_t const>;

/** The bytes behind @p count typed elements at @p data, read-only. */
template <typename value_type_>
bytes_view_t bytes_of(value_type_ const* data, std::size_t count) noexcept {
    return std::as_bytes(std::span {data, count});
}

/** The bytes behind @p count typed elements at @p data, writable. */
template <typename value_type_>
bytes_span_t bytes_of(value_type_* data, std::size_t count) noexcept {
    return std::as_writable_bytes(std::span {data, count});
}

/** The typed elements behind @p bytes, the inverse of @ref bytes_of and the one place a payload's
 *  bytes are read back as its elements; a trailing partial element is left out. */
template <typename element_type_>
std::span<element_type_ const> elements_of(bytes_view_t bytes) noexcept {
    return {reinterpret_cast<element_type_ const*>(bytes.data()), bytes.size() / sizeof(element_type_)};
}

/** The writable typed elements behind @p bytes, for a codec decoding straight into its slice. */
template <typename element_type_>
std::span<element_type_> elements_of(bytes_span_t bytes) noexcept {
    return {reinterpret_cast<element_type_*>(bytes.data()), bytes.size() / sizeof(element_type_)};
}

/** The characters behind @p bytes, as the view every string type converts from. */
inline std::string_view chars_of(bytes_view_t bytes) noexcept {
    return {reinterpret_cast<char const*>(bytes.data()), bytes.size()};
}

/** A borrowed tape: entries end to end in @c elements, entry @c i spanning @c offsets[i] up to
 *  @c offsets[i+1], so one offset more than entries closes the last. */
template <typename element_type_>
struct tape_view {
    element_type_* elements = nullptr;
    std::span<std::size_t const> offsets;

    /** Entries the tape holds. */
    std::size_t size() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }

    /** Entry @p row. */
    std::span<element_type_> operator[](std::size_t row) const noexcept {
        return {elements + offsets[row], offsets[row + 1] - offsets[row]};
    }

    /** The @p count entries from @p first, their offsets still counting from @c elements. */
    tape_view subview(std::size_t first, std::size_t count) const noexcept {
        return {elements, offsets.subspan(first, count + 1)};
    }
};

/** The process-wide allocator, named rather than defaulted: a container takes this when
 *  no arena, memory domain or device owns its bytes, so every such placement is one grep
 *  away instead of hiding in an omitted argument. */
struct global_allocator_t {};
inline constexpr global_allocator_t global_allocator {};

/** The padding that keeps two threads' words off one line; see @c USEARCH_DEFAULT_ALIGNMENT. */
inline constexpr std::size_t default_alignment_k = USEARCH_DEFAULT_ALIGNMENT;
static_assert(default_alignment_k >= 8 && (default_alignment_k & (default_alignment_k - 1)) == 0,
              "USEARCH_DEFAULT_ALIGNMENT must be a power of two, no narrower than a pointer pair");

/** The user-space virtual address width of x86-64 and Arm64 under four-level paging, which bounds
 *  how many records any one collection can ever place. */
inline constexpr std::size_t address_bits_k = 48;

/** How one step of a walk - a batch of landmarks or one expansion round - left it, in every tier. */
enum class walk_step_t : u32_t {

    /** The walk goes on: landmarks entered the frontier, or a round staged candidates to drain. */
    advanced_k,

    /** Nothing left can improve the candidates: every landmark is visited, or the frontier's
     *  nearest lies past the full candidates' radius. */
    exhausted_k,

    /** Scratch or metric staging refused mid-step: drain what staged, then stop. */
    saturated_k,

    /** The walk's fold asked to stop, so the walk ends with what it holds. */
    sated_k,
};

/** A fold's answer to each batch a walk scores: keep walking, or end the walk successfully. */
enum class walk_demand_t : bool {

    /** Keep walking. */
    continue_k,

    /** End the walk now, successfully, with what the fold holds. */
    sated_k,
};

/** The default fold of every tier: reads nothing and never ends the walk, so a walk that folds
 *  nothing inlines the axis away. */
struct dummy_fold_t {
    template <typename... batch_types_>
    constexpr walk_demand_t absorb(batch_types_ const&...) const noexcept {
        return walk_demand_t::continue_k;
    }
};

/**
 *  @brief Why a call failed, in the vocabulary the engine and every binding share.
 *
 *  Every enumerator takes its value from @c usearch_status_t in @c types.h, so the C++ mirror
 *  and the ABI cannot drift - there is nothing to keep in step and nothing to assert, and the
 *  crossing in @c src/ is a cast rather than a table.
 *
 *  The set describes the project's failures, not this tier's exits: several enumerators name
 *  conditions only a binding or a higher layer can see, and are never returned from here.
 */
enum class [[nodiscard]] status_t : int {

    /** The call completed and any paired value is meaningful. */
    success_k = usearch_status_success_k,

    /** The underlying allocator failed to provide memory; retrying a smaller batch may succeed. */
    bad_alloc_k = usearch_status_out_of_memory_k,

    /** Dimensionality, row counts, or output capacity disagree. */
    shape_mismatch_k = usearch_status_shape_mismatch_k,

    /** A stride or packing no traversal can walk. */
    unsupported_layout_k = usearch_status_unsupported_layout_k,

    /** A scalar, metric, or residency tag this build does not know. */
    unknown_kind_k = usearch_status_unknown_kind_k,

    /** A mutating verb on a structure that opens its checkpoint in place. */
    read_only_k = usearch_status_read_only_k,

    /** A file could not be opened, read, written, or parsed. */
    io_k = usearch_status_io_k,

    /** An accelerator runtime call failed - the device state should be treated as
     *  poisoned and the structure rebuilt. */
    device_error_k = usearch_status_device_k,

    /** The serialized payload is truncated or does not describe this structure. */
    malformed_k = usearch_status_malformed_k,

    /** Neither success nor a named failure was reported - a defect, not a state. */
    internal_k = usearch_status_internal_k,

    /** The structure reached a hard design ceiling, like the 2^32 ID space; retrying can't help. */
    overflow_k = usearch_status_overflow_k,

    /** The operation requires a prior successful @c reserve() -like initialization call. */
    uninitialized_k = usearch_status_uninitialized_k,

    /** The requested configuration contradicts an already-frozen one, like re-reserving
     *  with a different geometry; pass matching parameters or start a fresh structure. */
    config_mismatch_k = usearch_status_config_mismatch_k,

    /** A certified synchronization contract was violated by a concurrent operation;
     *  discard any produced output and retry in a quieter phase. */
    contended_k = usearch_status_contended_k,

    /** A progress callback asked the operation to stop; it unwound without publishing,
     *  so the structure carries whatever it did before the call. */
    cancelled_k = usearch_status_cancelled_k,
    /** @brief The verb is declared for every residency but this one does not implement it. */
    unsupported_k = usearch_status_unsupported_k,

    /** The call was submitted deferred and is not final yet; never a final status. */
    deferred_k = usearch_status_deferred_k,

    /** An argument breaks the verb's contract, like a required pointer left null. */
    invalid_argument_k = usearch_status_invalid_argument_k,
};

/** Whether @p status reports success. */
constexpr bool succeeded(status_t status) noexcept { return status == status_t::success_k; }

/** Whether @p status reports failure. */
constexpr bool failed(status_t status) noexcept { return status != status_t::success_k; }

/** Copies @p source over @p destination; the sizes must agree. */
inline status_t copy_bytes(bytes_span_t destination, bytes_view_t source) noexcept {
    if (destination.size() != source.size()) return status_t::config_mismatch_k;
    if (!source.empty()) std::memcpy(destination.data(), source.data(), source.size());
    return status_t::success_k;
}

/** Rows @p first to @p first + @p count of a flat row-major @p matrix whose rows are @p row_bytes
 *  wide, or empty when the matrix holds fewer. */
template <typename bytes_type_>
bytes_type_ flat_rows(bytes_type_ matrix, std::size_t first, std::size_t count, std::size_t row_bytes) noexcept {
    std::size_t const begin = first * row_bytes, bytes = count * row_bytes;
    if (begin > matrix.size() || matrix.size() - begin < bytes) return {};
    return matrix.subspan(begin, bytes);
}

/** Whether a filtered search admits one candidate or passes over it - rejected nodes
 *  still route the walk, they just never enter the reported candidates. Mirrors
 *  @c usearch_admission_t value for value, so the two enums cannot drift. */
enum class admission_t : int {

    /** Skip the node; the walk still routes through it. */
    reject_k = usearch_reject_k,

    /** Report the node among the matches. */
    admit_k = usearch_admit_k,
};

/** Whether the operation a progress callback is watching carries on or unwinds.
 *  Mirrors @c usearch_continuation_t value for value, so the two enums cannot drift. */
enum class continuation_t : int {

    /** Carry on to the next unit of work. */
    continue_k = usearch_continue_k,

    /** Unwind: the verb publishes nothing further and reports @c cancelled_k. */
    cancel_k = usearch_cancel_k,
};

/** Whether a device verb returns when its work is done, or once it is merely queued on its stream,
 *  which is how a sharded index overlaps its devices before one drain. */
enum class completion_t : int {

    /** Return once the work is finished and its outputs are readable. */
    blocking_k,

    /** Return once the work is queued; outputs stay indeterminate until a drain. */
    deferred_k,
};

/**
 *  @brief Static, English description of @p status. Never returns @c nullptr, never allocates.
 *  @note The fall-through sits after the @c switch rather than in a @c default: label, so adding
 *      an enumerator without a text here is a @c -Wswitch warning instead of silent prose.
 */
constexpr char const* status_to_string(status_t status) noexcept {
    switch (status) {
    case status_t::success_k: return "success";
    case status_t::bad_alloc_k: return "an allocation failed";
    case status_t::shape_mismatch_k: return "dimensions, row counts, or output capacity disagree";
    case status_t::unsupported_layout_k: return "a stride or packing no traversal can walk";
    case status_t::unknown_kind_k: return "a scalar, metric, or residency tag this build does not know";
    case status_t::read_only_k: return "a mutating verb on a structure opened in place";
    case status_t::io_k: return "a file could not be opened, read, written, or parsed";
    case status_t::device_error_k: return "an accelerator runtime call failed";
    case status_t::malformed_k: return "the payload is truncated or describes something else";
    case status_t::internal_k: return "the library reported neither success nor a failure";
    case status_t::overflow_k: return "a hard design ceiling was reached";
    case status_t::uninitialized_k: return "the structure needs a successful reservation first";
    case status_t::config_mismatch_k: return "the request contradicts the frozen configuration";
    case status_t::contended_k: return "a synchronization contract was violated concurrently";
    case status_t::cancelled_k: return "a progress callback asked the operation to stop";
    case status_t::unsupported_k: return "this residency does not implement the verb or its traversal";
    case status_t::deferred_k: return "the call was submitted deferred and is not final yet";
    case status_t::invalid_argument_k: return "an argument breaks the verb's contract";
    }
    return "an unrecognized status";
}

/** A result paired with the @c status_t explaining it; the value is only meaningful
 *  on @c success_k. */
template <typename value_type_>
struct [[nodiscard]] expected {
    using value_t = value_type_;

    value_type_ value;
    status_t status;

    explicit operator bool() const noexcept { return succeeded(status); }
};

/** The borrowing face of @ref expected for reference results: there is no null
 *  reference, so the borrow is stored as an address and @c value() binds it only on
 *  @c success_k. */
template <typename value_type_>
struct [[nodiscard]] expected<value_type_&> {
    value_type_* borrowed;
    status_t status;

    value_type_& value() const noexcept { return *borrowed; }
    explicit operator bool() const noexcept { return succeeded(status); }
};

/** A half-open interval of integer identifiers, iterable like the standard views -
 *  the unit callers partition work by, one contiguous run per core. */
template <typename index_type_ = std::size_t>
struct iota_view {

    /** The first identifier of the interval. */
    index_type_ first {0};

    /** One past the last identifier. */
    index_type_ bound {0};

    constexpr std::size_t size() const noexcept { return static_cast<std::size_t>(bound - first); }
    constexpr bool empty() const noexcept { return first == bound; }

    struct iterator {
        index_type_ value;

        constexpr index_type_ operator*() const noexcept { return value; }
        constexpr iterator& operator++() noexcept { return ++value, *this; }
        constexpr bool operator==(iterator const& other) const noexcept { return value == other.value; }
        constexpr bool operator!=(iterator const& other) const noexcept { return value != other.value; }
    };
    constexpr iterator begin() const noexcept { return {first}; }
    constexpr iterator end() const noexcept { return {bound}; }
};

/** A range over an indexed accessor: @c owner[position] for every position below @p count,
 *  so a caller reads @c for(auto element : range) rather than counting positions itself. */
template <typename owner_type_>
struct indexed_range {

    /** What the positions are read from; borrowed, never owned. */
    owner_type_ const* owner = nullptr;

    /** Positions the range covers, from zero. */
    std::size_t count = 0;

    constexpr std::size_t size() const noexcept { return count; }
    constexpr bool empty() const noexcept { return !count; }
    constexpr auto operator[](std::size_t position) const noexcept { return (*owner)[position]; }

    struct iterator {
        owner_type_ const* owner = nullptr;
        std::size_t position = 0;

        constexpr auto operator*() const noexcept { return (*owner)[position]; }
        constexpr iterator& operator++() noexcept { return ++position, *this; }
        constexpr bool operator==(iterator const& other) const noexcept { return position == other.position; }
        constexpr bool operator!=(iterator const& other) const noexcept { return position != other.position; }
    };
    constexpr iterator begin() const noexcept { return {owner, 0}; }
    constexpr iterator end() const noexcept { return {owner, count}; }
};

/** Moves neighbor IDs between ID spaces: an ID inside @p source_range lands at
 *  @c id-source_range.first+shift, every other ID folds onto @p sentinel, which a walk
 *  reads as naming no node. */
template <std::unsigned_integral id_type_>
constexpr void rebase_neighbors(std::span<id_type_> neighbors, iota_view<id_type_> source_range, id_type_ shift,
                                id_type_ sentinel) noexcept {
    for (id_type_& neighbor : neighbors)
        neighbor = source_range.first <= neighbor && neighbor < source_range.bound //
                       ? static_cast<id_type_>(neighbor - source_range.first + shift)
                       : sentinel;
}

/**
 *  @brief Packs the surviving IDs of each row of @p rows to its front and pads the tail with
 *      @p sentinel, writing @p first_row + row as a self-loop for a row nothing survived in.
 *
 *  A rebase folds every foreign ID onto the sentinel wherever it sat, and a concurrent walk
 *  reads a sentinel in slot 0 as a node locked by a live writer - which a loaded graph has none
 *  of, so it would spin forever. Compacting restores what a live index maintains: slot 0 is a
 *  real ID, and the sentinel only ever ends a row.
 */
template <std::unsigned_integral id_type_>
constexpr void compact_neighbors(std::span<id_type_> rows, std::size_t connectivity, id_type_ first_row,
                                 id_type_ sentinel) noexcept {
    if (!connectivity) return;
    assert(rows.size() % connectivity == 0 && "A partial row never matches the loop's exit");
    for (std::size_t row = 0; row * connectivity != rows.size(); ++row) {
        std::span<id_type_> const slots = rows.subspan(row * connectivity, connectivity);
        std::size_t kept = 0;
        for (std::size_t slot = 0; slot != connectivity; ++slot)
            if (slots[slot] != sentinel) slots[kept++] = slots[slot];
        if (!kept) slots[kept++] = static_cast<id_type_>(first_row + row);
        for (std::size_t slot = kept; slot != connectivity; ++slot) slots[slot] = sentinel;
    }
}

/** The concurrency environment a caller certifies for one operation. */
enum class synchronization_t : u32_t {

    /** Arbitrary readers & writers may be live - the safe default everywhere. */
    concurrent_k,

    /** Readers may be live, but no mutators: neighbor lists are read as plain memory -
     *  no fences - and the locked-node retry branches compile out entirely. */
    immutable_k,

    /** Sole ownership - nothing else touches the index, readers included. */
    exclusive_k,
};

/** How a search reaches the rows it scores, mirroring @c usearch_traversal_t. */
enum class traversal_t : u32_t {

    /** Walk the graph, scoring the rows the expansion reaches - approximate, the default. */
    walk_k = usearch_traversal_walk_k,

    /** Score every published row - exact, ignoring the expansion factor. */
    scan_k = usearch_traversal_scan_k,
};

enum connectivity_t : u32_t {
    connectivity_unknown_k = 0,

    /** The minimal recommended graph connectivity level, expanding to 16x 32-bit neighbor IDs,
     *  exactly matching the ZMM register size and cache-line width on x86, but half of the width
     *  of coherency protocols of many CPUs and half the cache line width of many Arm CPUs. */
    connectivity_16_k = 16,

    /** The default recommended graph connectivity level, expanding to 32x 32-bit neighbor IDs,
     *  perfect for in-memory billion-scale search indexes. */
    connectivity_32_k = 32,

    /** The minimal recommended graph connectivity level for on-disk billion- and trillion-scale
     *  search indexes, matching the minimal 512-byte SSD sector size. */
    connectivity_128_k = 128,
};

/** The walk breadth a walk takes when its config asks for none: twice the neighbors one row
 *  holds, in every design and residency. */
constexpr std::size_t default_expansion_factor(std::size_t connectivity) noexcept { return connectivity * 2; }

/** Element types USearch speaks when exchanging payloads - NumKong's numeric dtypes
 *  plus the packed, opaque, identity and text families that interchange formats carry
 *  and a compute library has no place for. Every storage width also has a neutral
 *  spelling naming that many opaque bits, and every other spelling is an alias adding
 *  a reading to the same width; this enum is the USearch-side vocabulary every reader
 *  parses those spellings into. */
enum class dtype_t : u32_t {
    unknown_k = 0,
    /* Floats, by shrinking dynamic range. */
    f64_k,
    f32_k,
    bf16_k,
    f16_k,
    e5m2_k,
    e4m3_k,
    e3m2_k,
    e2m3_k,
    /* Signed integers, by shrinking dynamic range; the packed pair is two nibbles to the byte. */
    i64_k,
    i32_k,
    i16_k,
    i8_k,
    i4x2_k,
    /* Unsigned integers, by shrinking dynamic range; the packed families ride two or eight to the byte. */
    u64_k,
    u32_k,
    u16_k,
    u8_k,
    u4x2_k,
    u1x8_k,
    /* Opaque widths, naming bits and nothing else. */
    b128_k,
    b64_k,
    b32_k,
    b16_k,
    b8_k,
    /* A sixteen-byte identity, and bytes a reader may assume are valid text. */
    uuid_k,
    utf8_k,
};

/** The canonical spelling of @p dtype - the very strings interchange headers carry. */
constexpr char const* dtype_name(dtype_t dtype) noexcept {
    switch (dtype) {
    case dtype_t::f64_k: return "f64";
    case dtype_t::f32_k: return "f32";
    case dtype_t::bf16_k: return "bf16";
    case dtype_t::f16_k: return "f16";
    case dtype_t::e5m2_k: return "e5m2";
    case dtype_t::e4m3_k: return "e4m3";
    case dtype_t::e3m2_k: return "e3m2";
    case dtype_t::e2m3_k: return "e2m3";
    case dtype_t::i64_k: return "i64";
    case dtype_t::i32_k: return "i32";
    case dtype_t::i16_k: return "i16";
    case dtype_t::i8_k: return "i8";
    case dtype_t::i4x2_k: return "i4x2";
    case dtype_t::u64_k: return "u64";
    case dtype_t::u32_k: return "u32";
    case dtype_t::u16_k: return "u16";
    case dtype_t::u8_k: return "u8";
    case dtype_t::u4x2_k: return "u4x2";
    case dtype_t::u1x8_k: return "u1x8";
    case dtype_t::b128_k: return "b128";
    case dtype_t::b64_k: return "b64";
    case dtype_t::b32_k: return "b32";
    case dtype_t::b16_k: return "b16";
    case dtype_t::b8_k: return "b8";
    case dtype_t::uuid_k: return "uuid";
    case dtype_t::utf8_k: return "utf8";
    case dtype_t::unknown_k: return "unknown";
    }
    return "unknown";
}

/** Parses a canonical dtype spelling - the reverse of @ref dtype_name - mapping
 *  anything unrecognized onto @c unknown_k. */
constexpr dtype_t dtype_named(std::string_view name) noexcept {
    if (name == "f64") return dtype_t::f64_k;
    if (name == "f32") return dtype_t::f32_k;
    if (name == "bf16") return dtype_t::bf16_k;
    if (name == "f16") return dtype_t::f16_k;
    if (name == "e5m2") return dtype_t::e5m2_k;
    if (name == "e4m3") return dtype_t::e4m3_k;
    if (name == "e3m2") return dtype_t::e3m2_k;
    if (name == "e2m3") return dtype_t::e2m3_k;
    if (name == "i64") return dtype_t::i64_k;
    if (name == "i32") return dtype_t::i32_k;
    if (name == "i16") return dtype_t::i16_k;
    if (name == "i8") return dtype_t::i8_k;
    if (name == "i4x2") return dtype_t::i4x2_k;
    if (name == "u64") return dtype_t::u64_k;
    if (name == "u32") return dtype_t::u32_k;
    if (name == "u16") return dtype_t::u16_k;
    if (name == "u8") return dtype_t::u8_k;
    if (name == "u4x2") return dtype_t::u4x2_k;
    if (name == "u1x8") return dtype_t::u1x8_k;
    if (name == "b128") return dtype_t::b128_k;
    if (name == "b64") return dtype_t::b64_k;
    if (name == "b32") return dtype_t::b32_k;
    if (name == "b16") return dtype_t::b16_k;
    if (name == "b8") return dtype_t::b8_k;
    if (name == "uuid") return dtype_t::uuid_k;
    if (name == "utf8") return dtype_t::utf8_k;
    return dtype_t::unknown_k;
}

/** Bits one element of @p dtype occupies - eight for a byte-wide scalar, four or one
 *  for the packed families, zero for the variable-length and unknown ones. */
constexpr std::size_t dtype_bits(dtype_t dtype) noexcept {
    switch (dtype) {
    case dtype_t::f64_k:
    case dtype_t::i64_k:
    case dtype_t::u64_k:
    case dtype_t::b64_k: return 64;
    case dtype_t::f32_k:
    case dtype_t::i32_k:
    case dtype_t::u32_k:
    case dtype_t::b32_k: return 32;
    case dtype_t::bf16_k:
    case dtype_t::f16_k:
    case dtype_t::i16_k:
    case dtype_t::u16_k:
    case dtype_t::b16_k: return 16;
    case dtype_t::e5m2_k:
    case dtype_t::e4m3_k:
    case dtype_t::e3m2_k:
    case dtype_t::e2m3_k:
    case dtype_t::i8_k:
    case dtype_t::u8_k:
    case dtype_t::b8_k:
    case dtype_t::utf8_k: return 8;
    case dtype_t::i4x2_k:
    case dtype_t::u4x2_k: return 4;
    case dtype_t::u1x8_k: return 1;
    case dtype_t::uuid_k:
    case dtype_t::b128_k: return 128;
    case dtype_t::unknown_k: return 0;
    }
    return 0;
}

/** Bytes one element of @p dtype occupies; zero for the packed families, whose elements
 *  share a byte, and for the unknown one. */
constexpr std::size_t dtype_bytes(dtype_t dtype) noexcept {
    std::size_t const bits = dtype_bits(dtype);
    return bits && bits % 8 == 0 ? bits / 8 : 0;
}

/** Returns 2^ @p exponent as a value of the requested unsigned type. */
template <typename result_type_>
constexpr result_type_ power_of_two(u32_t exponent) noexcept {
    return result_type_(1) << exponent;
}

/** Returns a mask selecting the lowest @p exponent bits, i.e. 2^ @p exponent - 1. */
template <typename result_type_>
constexpr result_type_ power_of_two_mask(u32_t exponent) noexcept {
    return (result_type_(1) << exponent) - result_type_(1);
}

/** Integer division rounding towards positive infinity; @p divisor must be non-zero. */
template <typename value_type_>
constexpr value_type_ divide_round_up(value_type_ value, value_type_ divisor) noexcept {
    return (value + divisor - value_type_(1)) / divisor;
}

/** The smallest multiple of @p multiple at or above @p value; @p multiple must be non-zero. */
template <typename value_type_>
constexpr value_type_ round_up_to_multiple(value_type_ value, value_type_ multiple) noexcept {
    return divide_round_up(value, multiple) * multiple;
}

/** @p value with its two bytes reversed. */
constexpr std::uint16_t reversed_bytes(std::uint16_t value) noexcept {
    return static_cast<std::uint16_t>((value << 8) | (value >> 8));
}

/** @p value with its four bytes reversed. */
constexpr std::uint32_t reversed_bytes(std::uint32_t value) noexcept {
    return (value << 24) | ((value & 0x0000FF00u) << 8) | ((value >> 8) & 0x0000FF00u) | (value >> 24);
}

/** @p value with its eight bytes reversed, as the two halves swapped and each reversed. */
constexpr std::uint64_t reversed_bytes(std::uint64_t value) noexcept {
    return (static_cast<std::uint64_t>(reversed_bytes(static_cast<std::uint32_t>(value))) << 32) |
           reversed_bytes(static_cast<std::uint32_t>(value >> 32));
}

/** @p value with its bytes in little-endian order, and the same call to read one back - the
 *  conversion is its own inverse, so one name serves both directions. A no-op where the host is
 *  already little-endian, and a @ref reversed_bytes where it is not. */
template <typename value_type_>
constexpr value_type_ little_endian(value_type_ value) noexcept {
    static_assert(std::is_unsigned_v<value_type_>, "Byte order is only defined for unsigned integers");
    if constexpr (std::endian::native == std::endian::little) return value;
    else return reversed_bytes(value);
}

/** The smaller of @p first & @p second - @c std::min without the <algorithm> tax,
 *  named to also dodge the @c min macro of Windows headers. */
template <typename value_type_>
constexpr value_type_ min_of(value_type_ first, value_type_ second) noexcept {
    return second < first ? second : first;
}

/** The larger of @p first & @p second - the @c max twin of @ref min_of. */
template <typename value_type_>
constexpr value_type_ max_of(value_type_ first, value_type_ second) noexcept {
    return first < second ? second : first;
}

/** Rows whose neighbors fill a walk's @p expansion candidates at @p connectivity neighbors per row,
 *  none while an index holds no geometry yet. */
template <typename count_type_>
constexpr count_type_ filling_rows(count_type_ expansion, count_type_ connectivity) noexcept {
    return connectivity ? divide_round_up(expansion, connectivity) : count_type_ {0};
}

/** Rows a walk expands per hop: @p concurrent_walks, the walks the hardware runs at once, divided
 *  among the walks that need them, at least one and at most @p filling_rows. While a batch has more
 *  walks waiting than the hardware runs, every walk expands one row per hop. */
template <typename count_type_>
constexpr count_type_ walk_width(count_type_ concurrent_walks, count_type_ live_walks, count_type_ unstarted_walks,
                                 count_type_ filling_rows) noexcept {
    count_type_ const walks = max_of(live_walks, min_of(unstarted_walks, concurrent_walks));
    return min_of(filling_rows, max_of<count_type_>(concurrent_walks / walks, 1));
}

/** Outcome of one reachability audit, the same on every tier, layer and residency. */
struct health_result_t {

    /** Nodes published when the audit started - the population it judges; later insertions are not
     *  counted. */
    std::size_t audited_nodes = 0;

    /** Of those, the nodes a search reaches from its entry points - the searcher's view. */
    std::size_t reachable_nodes = 0;

    /** Level-synchronous sweeps to convergence. */
    std::size_t sweeps = 0;

    /** Levels the sweep back toward the entry point took, where an audit sweeps both ways. */
    std::size_t back_sweeps = 0;
};

/** Folds another shard's audit in: populations add, and the depth is the deepest shard's, since a
 *  searcher waits for the last one. */
inline health_result_t& operator+=(health_result_t& into, health_result_t const& other) noexcept {
    into.audited_nodes += other.audited_nodes;
    into.reachable_nodes += other.reachable_nodes;
    into.sweeps = max_of(into.sweeps, other.sweeps);
    into.back_sweeps = max_of(into.back_sweeps, other.back_sweeps);
    return into;
}

/** The tail slots of a full row a heal may give up to islands, which spread over them by ID so
 *  islands sharing a row land apart: the farther half, keeping the nearer half that carries search
 *  and slot 0, and nothing in a single-neighbor row. */
template <typename size_type_>
constexpr size_type_ heal_spread(size_type_ connectivity) noexcept {
    return connectivity / 2;
}

/** The rows that @p node_id walks across first in round @p round of merging @p parts parts, whose
 *  ascending @p part_bases start below @p rows: the neighboring group of `2^round` parts, or none
 *  when no group neighbors its own. */
template <typename node_id_type_>
constexpr iota_view<node_id_type_> part_across(node_id_type_ const* part_bases, std::size_t parts, std::size_t round,
                                               node_id_type_ rows, node_id_type_ node_id) noexcept {
    std::size_t part = 0;
    while (part + 1 < parts && part_bases[part + 1] <= node_id) ++part;
    std::size_t const first = ((part >> round) ^ 1u) << round, bound = first + (std::size_t {1} << round);
    if (first >= parts) return {};
    return {part_bases[first], bound < parts ? part_bases[bound] : rows};
}

/** The universal matrix-engine row count: Intel AMX tiles, Arm SME's ZA at SVL-512,
 *  and NVIDIA mma fragments all speak 16 rows natively - Hopper's warpgroups speak 4×16.
 *  Tiled paths across every index design derive their shapes from it, never tune it. */
inline constexpr std::size_t tile_span_k = 16;

/**
 *  @brief The first position in the @p values, ascending under @p comparator, not ordered before
 *      @p value - @c std::lower_bound re-spelled in ten lines, so headers don't drag in
 *      @c <algorithm>, one of the heaviest includes on the compile-health charts.
 *  @see https://artificial-mind.net/projects/compile-health/
 */
template <typename value_type_, typename comparator_type_ = std::less<void>>
constexpr value_type_* lower_bound(std::span<value_type_> values, std::type_identity_t<value_type_> const& value,
                                   comparator_type_ comparator = {}) noexcept {
    value_type_* first = values.data();
    std::size_t count = values.size();
    while (count) {
        std::size_t const half = count / 2;
        value_type_* const middle = first + half;
        bool const is_before = comparator(*middle, value);
        first = is_before ? middle + 1 : first;
        count = is_before ? count - half - 1 : half;
    }
    return first;
}

/**
 *  @brief Converts any contiguous vector-like container - NumKong views & spans included -
 *      into the STL hand-off shape, preserving the element's constness.
 *
 *  Two NumKong properties make a blind conversion unsafe, so both are guarded without this
 *  header ever including NumKong: packed sub-byte scalars report their @b logical dimension
 *  count through @c size(), exceeding their storage count, and are rejected at compile time
 *  through their own @c component_t reflection; strided views violate the contiguity that
 *  @c std::span assumes, and are rejected by the assertion.
 */
template <typename vector_type_>
    requires requires(vector_type_ const& vector) {
        vector.data();
        vector.size();
    }
constexpr auto to_span(vector_type_ const& vector) noexcept {
    using value_t = std::remove_pointer_t<decltype(vector.data())>;
    using scalar_t = std::remove_const_t<value_t>;
    if constexpr (requires { typename scalar_t::component_t; })
        static_assert(                                                //
            std::is_same_v<scalar_t, typename scalar_t::component_t>, //
            "Packed sub-byte scalars size themselves in logical "     //
            "dimensions, not storage values, so they cannot form a span");
    if constexpr (requires { vector.stride_bytes(); })
        assert(static_cast<std::size_t>(vector.stride_bytes()) == sizeof(value_t));
    return std::span<value_t> {vector.data(), vector.size()};
}

/**
 *  @brief The writable sibling of @ref to_span, for a run the caller owns and fills; named
 *      apart rather than overloaded, since an overload would silently hand a mutable span to
 *      every existing caller holding a non-const run.
 *  @see @c std::as_bytes and @c std::as_writable_bytes, which separate the same pair.
 */
template <typename vector_type_>
    requires requires(vector_type_& vector) {
        vector.data();
        vector.size();
    }
constexpr auto to_writable_span(vector_type_& vector) noexcept {
    using value_t = std::remove_pointer_t<decltype(vector.data())>;
    return std::span<value_t> {vector.data(), vector.size()};
}

/** The character sibling of @ref to_span: any contiguous run of @c char as a view. */
template <typename characters_type_>
    requires requires(characters_type_ const& characters) {
        { characters.data() } -> std::convertible_to<char const*>;
        characters.size();
    }
constexpr std::string_view to_string_view(characters_type_ const& characters) noexcept {
    return std::string_view {characters.data(), characters.size()};
}

/** Any entry pairing a graph-local @c node_id with a @c distance - the shape that
 *  search results share across index designs. */
template <typename entry_type_>
concept scored_node = requires(entry_type_ entry) {
    entry.node_id;
    entry.distance;
};

/** @brief A two-stage save or load in flight: move-only and never assigned over, its steps
 *      handed to threads one at a time, spent by @c finish. */
template <typename plan_type_>
concept index_plan = std::move_constructible<plan_type_> && !std::copy_constructible<plan_type_> &&
                     !std::is_copy_assignable_v<plan_type_> && !std::is_move_assignable_v<plan_type_> &&
                     requires(plan_type_ plan) {
                         plan.steps().size();
                         { plan.finish() } noexcept -> std::same_as<status_t>;
                     };

/** @brief Rows taken against a ceiling: move-only, released by its own death unless consumed. */
template <typename claim_type_>
concept index_claim = std::move_constructible<claim_type_> && !std::copy_constructible<claim_type_> &&
                      requires(claim_type_ claim) {
                          { claim.rows() } noexcept -> std::same_as<std::size_t>;
                          { claim.consume() } noexcept;
                      };

/** @brief The protocol every dense index answers, at one shard and across many: the verbs a
 *      sharded layer drives its shards by, and the plans they hand out. */
template <typename index_type_>
concept index_family = index_plan<typename index_type_::save_plan_t> && index_plan<typename index_type_::load_plan_t> &&
                       requires(index_type_& index, index_type_ const& constant) {
                           { constant.size() } noexcept -> std::same_as<std::size_t>;
                           { constant.capacity() } noexcept -> std::same_as<std::size_t>;
                           {
                               index.reserve(typename index_type_::reserve_capacity_t {})
                           } noexcept -> std::same_as<expected<std::size_t>>;
                       };

/**
 *  @brief Merges the ascending @p results of one search into the caller's bounded ranking,
 *      keeping it ascending by distance.
 *
 *  Indexes store no keys, so IDs are translated into the caller's key space on the way in -
 *  letting several sub-indexes with colliding ID spaces, like one per memory domain, gather
 *  into one shared buffer. The parallel spans' common length is the ranking's capacity.
 *
 *  @param[in] results One index's search results, ascending by distance.
 *  @param[in] keys_of_ids Any ID-to-key mapping with @c operator[] - a flat array, a raw
 *      pointer, or a custom type.
 *  @param[inout] keys Keys of the ranking, already holding @p old_count hits from other indexes.
 *  @param[inout] distances Distances of the same ranking, parallel to @p keys.
 *  @param[in] old_count Hits already in the ranking before this merge.
 *  @return The merged hit count, never exceeding the capacity.
 */
template <scored_node entry_type_, typename keys_of_ids_type_, typename key_type_, typename distance_type_>
std::size_t merge_search_results(std::span<entry_type_ const> results, keys_of_ids_type_ const& keys_of_ids,
                                 std::span<key_type_> keys, std::span<distance_type_> distances,
                                 std::size_t old_count) noexcept {
    static_assert(std::is_trivially_copyable_v<key_type_> && std::is_trivially_copyable_v<distance_type_>,
                  "The ranking is shifted with `memmove`, so both spans must hold trivial types");
    std::size_t const max_count = keys.size() < distances.size() ? keys.size() : distances.size();
    std::size_t merged_count = old_count;
    for (std::size_t hit = 0; hit != results.size(); ++hit) {
        entry_type_ const& scored = results[hit];
        distance_type_ const distance = static_cast<distance_type_>(scored.distance);
        std::size_t const offset = lower_bound(distances.subspan(0, merged_count), distance) - distances.data();
        if (offset == max_count) break; // Results are ascending - every later hit is worse.

        std::size_t const count_worse = merged_count - offset - (max_count == merged_count);
        std::memmove(keys.data() + offset + 1, keys.data() + offset, count_worse * sizeof(key_type_));
        std::memmove(distances.data() + offset + 1, distances.data() + offset, count_worse * sizeof(distance_type_));
        keys[offset] = keys_of_ids[scored.node_id];
        distances[offset] = distance;
        merged_count += merged_count != max_count;
    }
    return merged_count;
}

/** Tag: a capped monitored wait - re-checks on its own after a bounded pause. */
struct wait_capped_t {};

/**
 *  @brief Portable spin-wait baseline with the same call shape as ForkUnion's hardware wait
 *      monitors, like @c fu::x86_tpause_t, @c fu::arm64_wfet_t, and @c fu::risc5_wrs_t.
 *
 *  Ignores the watched word and returns at once, so the loop re-checks it - correctness never
 *  depends on the waiter, only latency and power do. The first overload watches a @c std::atomic
 *  object; the second watches an in-place word owned through @c std::atomic_ref, like the slot-0
 *  locks living inside neighbor rows. The waiting bound (capped or uncapped) arrives as an
 *  arbitrary tag type, so this header stays independent of ForkUnion.
 *
 *  The pointer overload exists because of pre-C++26 STL limitations: @c std::atomic_ref cannot
 *  bind const-qualified objects until P3323, @c std::atomic::wait offers no timed variant, and
 *  neither API exposes the address-armed hardware waits - @c UMWAIT, @c WFET, @c WRS - that
 *  ForkUnion reaches directly, so the raw address plus the observed bit pattern is the portable
 *  currency.
 */
struct dummy_waiter_t {
    template <typename value_type_, typename thread_index_type_, typename bound_type_ = wait_capped_t>
    void operator()(std::atomic<value_type_> const&, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {
    }
    template <typename value_type_, typename thread_index_type_, typename bound_type_ = wait_capped_t>
    void operator()(value_type_ const*, value_type_, thread_index_type_, bound_type_ = {}) const noexcept {}
};

/** The compiled-out cache hints, the default everywhere: a prefetcher over them hints nothing. */
struct dummy_cache_hints_t {};

/** The line tags a cache-hints type takes, and whether it emits any hint; the ForkUnion bridge
 *  specializes it for ForkUnion's hints. */
template <typename cache_hints_type_>
struct cache_line_tags;

template <>
struct cache_line_tags<dummy_cache_hints_t> {
    inline static constexpr bool prefetches_k = false;
};

/**
 *  @brief Streams whole objects toward the core or away from it through @p cache_hints_type_, a
 *      line at a time.
 *
 *  Engines build it from the raw hints @c src/ picks and the line bytes it reads once per machine.
 *  Over hints that emit nothing, @ref prefetches_k is false, so a walk skips the loads that only
 *  fed it.
 */
template <typename cache_hints_type_>
struct object_prefetcher {
    using tags_t = cache_line_tags<cache_hints_type_>;
    inline static constexpr bool prefetches_k = tags_t::prefetches_k;

    std::size_t line_bytes = default_alignment_k;

    /** Pulls @p bytes from @p address into this core's L2 with read intent. */
    void prefetch(void const* address, std::size_t bytes) const noexcept {
        if constexpr (prefetches_k) hint_(address, bytes, tags_t::prefetch_k);
    }

    /** Pushes @p bytes from @p address away from this core, toward the last-level cache. */
    void demote(void const* address, std::size_t bytes) const noexcept {
        if constexpr (prefetches_k) hint_(address, bytes, tags_t::demote_k);
    }

    /** Pulls @p bytes from @p address toward this core with write intent. */
    void promote(void const* address, std::size_t bytes) const noexcept {
        if constexpr (prefetches_k) hint_(address, bytes, tags_t::promote_k);
    }

  private:
    template <typename tag_type_>
    void hint_(void const* address, std::size_t bytes, tag_type_ tag) const noexcept {
        for (std::size_t offset = 0; offset < bytes; offset += line_bytes)
            cache_hints_type_ {}(static_cast<byte_t const*>(address) + offset, tag);
    }
};

/**
 *  @brief The standard's @c std::atomic_ref, extended to bind a const word for loads - the one
 *      place a const reader of a plain synchronization word casts its constness away, so no owner
 *      has to declare that word @c mutable.
 *
 *  The standard binds const words only from C++26 on, per P3323; ForkUnion's references already
 *  do, and this is the analog for code that must build on the standard library alone. A const word
 *  only ever loads. The reference is held as a member, with every named verb forwarded.
 */
template <typename value_type_>
struct standard_atomic_ref {
    explicit standard_atomic_ref(value_type_& word) noexcept : word_(word) {}

    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return word_.load(order); }
    void store(value_type_ value, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        word_.store(value, order);
    }
    value_type_ exchange(value_type_ value, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.exchange(value, order);
    }
    bool compare_exchange_weak(value_type_& expected, value_type_ desired,
                               std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.compare_exchange_weak(expected, desired, order);
    }
    bool compare_exchange_weak(value_type_& expected, value_type_ desired, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return word_.compare_exchange_weak(expected, desired, success, failure);
    }
    bool compare_exchange_strong(value_type_& expected, value_type_ desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.compare_exchange_strong(expected, desired, order);
    }
    bool compare_exchange_strong(value_type_& expected, value_type_ desired, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        return word_.compare_exchange_strong(expected, desired, success, failure);
    }
    value_type_ fetch_add(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.fetch_add(operand, order);
    }
    value_type_ fetch_sub(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.fetch_sub(operand, order);
    }
    value_type_ fetch_and(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.fetch_and(operand, order);
    }
    value_type_ fetch_or(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.fetch_or(operand, order);
    }
    value_type_ fetch_xor(value_type_ operand, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return word_.fetch_xor(operand, order);
    }

  private:
    std::atomic_ref<value_type_> word_;
};

template <typename value_type_>
struct standard_atomic_ref<value_type_ const> {
    explicit standard_atomic_ref(value_type_ const& word) noexcept : word_(const_cast<value_type_&>(word)) {}
    value_type_ load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return word_.load(order); }

  private:
    std::atomic_ref<value_type_> word_;
};

/** The compiled-out default for optional callbacks, like an insertion's assigned-ID
 *  observer: accepts any arguments and does nothing. */
struct dummy_callback_t {
    template <typename... args_types_>
    void operator()(args_types_&&...) const noexcept {}
};

/** The compiled-out default filter: admits anything it is asked about. Walks key on its
 *  @b type to skip the filtered machinery whole, so passing it explicitly costs nothing and
 *  answers the same. */
struct dummy_predicate_t {
    template <typename... args_types_>
    constexpr admission_t operator()(args_types_&&...) const noexcept {
        return admission_t::admit_k;
    }

    template <typename item_type_, std::size_t extent_>
    constexpr void operator()(std::span<item_type_ const, extent_>,
                              std::span<admission_t, extent_> admissions) const noexcept {
        for (admission_t& admission : admissions) admission = admission_t::admit_k;
    }
};

/** Lanes in a quad, what a predicate hears per call: the candidates one NumKong finalize scores
 *  together, reducing four accumulators into one 128-bit vector. */
inline constexpr std::size_t quad_lanes_k = 4;

/** The lanes one quad carries, one @p value_type_ each. */
template <typename value_type_>
using quad = std::array<value_type_, quad_lanes_k>;

/** Calls @p body with @p count, at most @p limit_, as its compile-time extent, so whatever the body
 *  sizes by that extent has a static size; a zero @p count calls nothing. */
template <std::size_t limit_, typename body_type_>
inline void with_static_extent(std::size_t count, body_type_&& body) noexcept {
    [&]<std::size_t... extents_>(std::index_sequence<extents_...>) noexcept {
        [[maybe_unused]] bool const called =
            ((count == extents_ + 1 && (body.template operator()<extents_ + 1>(), true)) || ...);
    }(std::make_index_sequence<limit_>());
}

/** A predicate asked as candidates arrive, at most a quad a call, every call at a static extent: a
 *  full quad, or a round's shorter tail. @p item_type_ is a key or a node ID. */
template <typename predicate_type_, typename item_type_>
concept immediate_predicate = requires(predicate_type_& predicate, std::span<item_type_ const, quad_lanes_k> items,
                                       std::span<admission_t, quad_lanes_k> admissions) {
    predicate(items, admissions);
};

/** A predicate that pays per call, so it gathers candidates beside the engine's @p payload_type_
 *  and answers them together: a drain hands each admitted payload on and leaves it empty. */
template <typename predicate_type_, typename item_type_, typename payload_type_>
concept aggregating_predicate = requires(predicate_type_& predicate, std::span<item_type_ const, quad_lanes_k> items,
                                         std::span<payload_type_ const, quad_lanes_k> payloads) {
    predicate.push(items, payloads);
    predicate.drain([](payload_type_ const&) noexcept {});
};

/** A family handing each executor thread a predicate of its own, as an engine driving its own
 *  threads needs of an aggregating one. */
template <typename predicate_type_>
concept per_thread_predicates = requires(predicate_type_& predicate, std::size_t thread) {
    { predicate.of_thread(thread) } -> std::same_as<typename predicate_type_::thread_predicate_t&>;
};

/** Asks @p predicate about a quad of @p items, handing @p on_admitted the payload of each it
 *  admits. */
template <typename predicate_type_, typename item_type_, typename payload_type_, std::size_t extent_,
          typename on_admitted_type_>
inline void ask_quad(predicate_type_& predicate, std::span<item_type_ const, extent_> items,
                     std::span<payload_type_ const, extent_> payloads, on_admitted_type_&& on_admitted) noexcept {
    std::array<admission_t, extent_> admissions;
    predicate(items, std::span<admission_t, extent_>(admissions));
    for (std::size_t lane = 0; lane != extent_; ++lane)
        if (admissions[lane] == admission_t::admit_k) on_admitted(payloads[lane]);
}

/** An aggregating @p predicate only gathers the quad, answering it at @ref settle_predicate. */
template <typename predicate_type_, typename item_type_, typename payload_type_, std::size_t extent_,
          typename on_admitted_type_>
    requires aggregating_predicate<predicate_type_, item_type_, payload_type_>
inline void ask_quad(predicate_type_& predicate, std::span<item_type_ const, extent_> items,
                     std::span<payload_type_ const, extent_> payloads, on_admitted_type_&&) noexcept {
    predicate.push(items, payloads);
}

/** Ends a round for @p predicate: an immediate one answered every quad already. */
template <typename predicate_type_, typename on_admitted_type_>
inline void settle_predicate(predicate_type_&, on_admitted_type_&&) noexcept {}

/** Answers everything an aggregating @p predicate gathered, handing @p on_admitted each admitted
 *  payload. */
template <typename predicate_type_, typename on_admitted_type_>
    requires requires(predicate_type_& predicate, on_admitted_type_&& on_admitted) { predicate.drain(on_admitted); }
inline void settle_predicate(predicate_type_& predicate, on_admitted_type_&& on_admitted) noexcept {
    predicate.drain(on_admitted);
}

/** Reserves nothing for an immediate @p predicate, which gathers nothing. */
template <typename predicate_type_>
inline status_t reserve_predicate(predicate_type_&, std::size_t) noexcept {
    return status_t::success_k;
}

/** Reserves an aggregating @p predicate for @p candidates between two settles, on whatever arena
 *  its engine rewound last. */
template <typename predicate_type_>
    requires requires(predicate_type_& predicate, std::size_t candidates) {
        { predicate.reserve(candidates) } -> std::same_as<status_t>;
    }
inline status_t reserve_predicate(predicate_type_& predicate, std::size_t candidates) noexcept {
    return predicate.reserve(candidates);
}

/** Opens nothing for a predicate every thread shares. */
template <typename predicate_type_, typename executor_type_>
inline status_t open_predicate(predicate_type_&, executor_type_&) noexcept {
    return status_t::success_k;
}

/** Opens a per-thread family on @p executor 's scratch, right after the call's rewind. */
template <typename predicate_type_, typename executor_type_>
    requires requires(predicate_type_& predicate, executor_type_& executor) {
        { predicate.open(executor) } -> std::same_as<status_t>;
    }
inline status_t open_predicate(predicate_type_& predicate, executor_type_& executor) noexcept {
    return predicate.open(executor);
}

/** The predicate executor thread @p thread asks: @p predicate itself, which every thread shares. */
template <typename predicate_type_>
inline predicate_type_& thread_predicate(predicate_type_& predicate, std::size_t) noexcept {
    return predicate;
}

/** A family keeping one aggregating predicate per thread hands each thread its own, so an engine
 *  driving its own threads settles each thread's rounds apart. */
template <per_thread_predicates predicate_type_>
inline typename predicate_type_::thread_predicate_t& thread_predicate(predicate_type_& predicate,
                                                                      std::size_t thread) noexcept {
    return predicate.of_thread(thread);
}

/** Ends a round for @p fold, letting a fold behind an aggregating predicate answer it. */
template <typename fold_type_>
inline void settle_fold(fold_type_&) noexcept {}

template <typename fold_type_>
    requires requires(fold_type_& fold) { fold.settle(); }
inline void settle_fold(fold_type_& fold) noexcept {
    fold.settle();
}

/**
 *  @brief A predicate over keys asked about node IDs: @p key_of_type_ maps a node ID to its key, a
 *      key column or an index's lookup, and the keys go on at the extent the IDs came in.
 *
 *  An aggregating predicate is forwarded the same way, its payloads untouched, so the engines fold
 *  over node IDs alone whichever shape the caller's predicate has.
 */
template <typename predicate_type_, typename key_of_type_>
struct node_predicate {
    predicate_type_& predicate;
    key_of_type_ key_of;

    template <typename node_id_type_, std::size_t extent_,
              typename key_type_ = std::invoke_result_t<key_of_type_ const&, node_id_type_>>
        requires immediate_predicate<predicate_type_, key_type_>
    void operator()(std::span<node_id_type_ const, extent_> nodes,
                    std::span<admission_t, extent_> admissions) noexcept {
        std::array<key_type_, extent_> keys;
        for (std::size_t lane = 0; lane != extent_; ++lane) keys[lane] = key_of(nodes[lane]);
        predicate(std::span<key_type_ const, extent_>(keys), admissions);
    }

    template <typename node_id_type_, typename payload_type_, std::size_t extent_,
              typename key_type_ = std::invoke_result_t<key_of_type_ const&, node_id_type_>>
        requires aggregating_predicate<predicate_type_, key_type_, payload_type_>
    void push(std::span<node_id_type_ const, extent_> nodes,
              std::span<payload_type_ const, extent_> payloads) noexcept {
        std::array<key_type_, extent_> keys;
        for (std::size_t lane = 0; lane != extent_; ++lane) keys[lane] = key_of(nodes[lane]);
        predicate.push(std::span<key_type_ const, extent_>(keys), payloads);
    }

    template <typename on_admitted_type_>
        requires requires(predicate_type_& predicate, on_admitted_type_&& on_admitted) { predicate.drain(on_admitted); }
    void drain(on_admitted_type_&& on_admitted) noexcept {
        predicate.drain(on_admitted);
    }
};

/** The compiled-out default progress watcher: always carries on, so an unwatched long
 *  verb runs to completion with no cancellation checks surviving inlining. */
struct dummy_progress_t {
    template <typename... args_types_>
    constexpr continuation_t operator()(args_types_&&...) const noexcept {
        return continuation_t::continue_k;
    }
};

/** The CPU families USearch ships tuned variants for - our own vocabulary, never an
 *  alias of a sister library's feature bits. Each class bundles the compute features,
 *  wait instruction, and atomics its silicon generation guarantees; explicit helpers
 *  lower a class onto NumKong, ForkUnion & StringZilla capability masks where those
 *  libraries are linked. */
enum class cpu_class_t : u32_t {
    unknown_k = 0,

    /** The portable floor: no SIMD assumptions, standard yields. */
    serial_k,

    /** Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    intel_haswell_k,

    /** Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI & its INT8 / IFMA / NE-CONVERT
     *  successors without AVX-512, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_sierra_k,

    /** Skylake-class x86: the AVX-512 foundation tier. */
    intel_skylake_k,

    /** Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    intel_icelake_k,

    /** Genoa-class x86 - Zen 4 and newer: native @c bf16 dot products over AVX-512. */
    amd_genoa_k,

    /** Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 &
     *  AMX-INT8 tiles. */
    intel_sapphire_k,

    /** Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles. */
    intel_granite_k,

    /** Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions and
     *  the Sierra vector bundle, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_diamond_k,

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; pre-Armv8.6 phones land
     *  here too. */
    aws_g2_k,

    /** Armv8.6+ NEON: dot products, @c bf16 & @c f16 fused multiplies. Named for
     *  Apple M2; Cortex-A710 & X2 - Snapdragon 8 Gen 1, AmpereOne - carry the same. */
    apple_m2_k,

    /** Apple M4-class Arm: the @c apple_m2_k bundle plus the SME outer-product tiles,
     *  @c BMOPA among them, without plain SVE. */
    apple_m4_k,

    /** Graviton5-class Arm - Neoverse V3, Armv9.2: the @c apple_m2_k bundle with
     *  @c FEAT_WFxT guaranteed, so waits monitor instead of spinning. */
    aws_g5_k,

    /** Vera-class Arm - NVIDIA's Armv9.2+ cores with @c FEAT_FP8 dot products. */
    nvidia_vera_k,

    /** Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23: the vector extension with
     *  @c Zvbb, @c Zawrs monitored waits, @c Zacas single-instruction compare-and-swap. */
    alibaba_c930_k,

    /** WebAssembly every engine loads: 32-bit, one caller-inclusive thread, the fixed-width @c simd128
     *  tier, so it runs on WebKit and in a page served without cross-origin isolation. */
    wasm32_v128_k,

    /** WebAssembly with relaxed SIMD, a shared memory for Web Worker pools and 64-bit addressing,
     *  bundled because every engine that ships one ships the other two. */
    wasm64_extended_k,
};

/** The GPU vendors USearch drives, each through a runtime loaded on first use, so one binary
 *  addresses every vendor's cards at once and loads where none is installed. */
enum class gpu_vendor_t : u32_t {
    unknown_k = 0,

    /** NVIDIA, through the static CUDA runtime and driver entry points it resolves. */
    nvidia_k,

    /** AMD, through the HIP runtime opened at first use and embedded code objects. */
    amd_k,

    /** Apple, through Metal and kernels compiled from embedded source at first use. */
    apple_k,
};

/** The GPU families USearch ships tuned variants for, the device twin of @ref cpu_class_t: one
 *  class per generation of kernel kits, each compiled for its own architecture alone. A class
 *  implies its vendor, and within a vendor an unknown newer device takes the newest class it
 *  satisfies. */
enum class gpu_class_t : u32_t {
    unknown_k = 0,

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30. */
    nvidia_ampere_k,

    /** Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions. */
    nvidia_ada_k,

    /** Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    nvidia_hopper_k,

    /** Blackwell-class NVIDIA - B200, B300 & GB300: TMA @c gather4 scattered fetches
     *  and @c tcgen05 tensor-core GEMMs. */
    nvidia_blackwell_k,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark: Blackwell's conversions
     *  without @c tcgen05. */
    nvidia_blackwellrtx_k,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    nvidia_rubin_k,

    /** CDNA3 AMD - MI300X & MI325X: gfx942, 64-wide waves, FNUZ fp8. */
    amd_cdna3_k,

    /** CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8. */
    amd_cdna4_k,

    /** CDNA5 AMD - MI400: gfx1250, 32-wide waves. */
    amd_cdna5_k,

    /** Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2. */
    apple_metal7_k,

    /** Apple9-family GPUs - M3 & M4. */
    apple_metal9_k,

    /** Apple10-family GPUs - M5: the Neural Accelerator matrix units. */
    apple_metal10_k,
};

/** One GPU as USearch numbers it - the vendor, that vendor's own ordinal and its class - with what
 *  a banner or a benchmark row reports, filled once by the vendor's listing, so no harness calls a
 *  vendor SDK. */
struct gpu_device_t {
    gpu_vendor_t vendor = gpu_vendor_t::unknown_k;
    u32_t vendor_ordinal = 0;
    gpu_class_t gpu_class = gpu_class_t::unknown_k;

    /** Streaming multiprocessors, compute units or GPU cores. */
    u32_t compute_units = 0;

    /** Device memory, or the working set Metal recommends for unified memory. */
    u64_t memory_bytes = 0;

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64] = {};
};

/** Devices one vendor's listing holds at most, so the process-wide listing needs no heap. */
inline constexpr std::size_t gpu_devices_capacity_k = 64;

/** Feature bits an I/O engine answers through @c capabilities() once open, since a refused
 *  registration turns @c fixed_buffers_k off at run time, so callers, tests, and docs introspect
 *  variants uniformly instead of by type name. */
enum class io_capability_t : u32_t {
    none_k = 0,

    /** Staging buffers are pre-registered with the kernel/ring - no per-read
     *  page pinning; roughly an 11% throughput gain at NVMe rates. */
    fixed_buffers_k = 1u << 0,

    /** Completions are polled from device queues instead of interrupt-driven;
     *  needs poll-capable queues & direct I/O; ~21% on top of the fixed path. */
    polled_k = 1u << 1,

    /** Requests address raw LBAs on a namespace, not file offsets - the
     *  filesystem & block layer are bypassed; ~20-30% over the block path. */
    lba_addressing_k = 1u << 2,
};

inline constexpr io_capability_t operator|(io_capability_t first, io_capability_t second) noexcept {
    return static_cast<io_capability_t>(static_cast<u32_t>(first) | static_cast<u32_t>(second));
}

/** Per-vector header carrying L2 norm and row-sum (used by DPBUSD compensation).
 *  Both fields share width @c dot_result_type_ - the lane width every downstream finalize
 *  and through-from-dot helper consumes. */
template <typename dot_result_type_>
struct vector_norms {
    dot_result_type_ sum;
    dot_result_type_ sumsq;
};

/**
 *  @brief Picks the first of @p kits_ this build compiled and @p caps_ runs, else @p floor_ - a
 *      ladder's one closing @c >::type, with no nested @c std::conditional_t pile.
 *
 *  Every kit states its own needs: @c capability_k, the capability it runs under, and
 *  @c compiled_k, whether this build carries its body. A kit whose target this build cannot carry
 *  is a stub of those two members alone, so naming it in a ladder never needs its body. @p caps_
 *  is a NumKong or a StringZilla capability set, whichever the kits state theirs in.
 */
template <auto caps_, typename floor_, typename... kits_>
struct highest_kit {
    using type = floor_;
};

template <auto caps_, typename floor_, typename head_, typename... tail_>
struct highest_kit<caps_, floor_, head_, tail_...> {
    using type = std::conditional_t<head_::compiled_k && (caps_ & head_::capability_k) != 0, head_,
                                    typename highest_kit<caps_, floor_, tail_...>::type>;
};

/**
 *  @brief The high half of the full @p first x @p second product - the multiply-shift
 *      kernel mapping a uniform word onto a bounded range without division.
 *
 *  Computed in half-width limbs, so no double-width type is ever needed: one definition
 *  serves every unsigned width on every compiler, @c constexpr and device-callable alike.
 *  Schoolbook four-limb form from Henry S. Warren Jr., "Hacker's Delight", 2nd ed., §8-2.
 */
template <typename word_type_>
constexpr word_type_ multiply_high(word_type_ first, word_type_ second) noexcept {
    static_assert(std::is_integral_v<word_type_> && std::is_unsigned_v<word_type_>);
    constexpr unsigned half_bits_k = sizeof(word_type_) * 4u;
    constexpr word_type_ half_mask_k = (word_type_(1) << half_bits_k) - word_type_(1);
    word_type_ const first_low = first & half_mask_k, first_high = first >> half_bits_k;
    word_type_ const second_low = second & half_mask_k, second_high = second >> half_bits_k;
    word_type_ const cross = (first_low * second_low >> half_bits_k) + first_high * second_low;
    word_type_ const cross_with_carry = (cross & half_mask_k) + first_low * second_high;
    return first_high * second_high + (cross >> half_bits_k) + (cross_with_carry >> half_bits_k);
}

/** 2^64 and 2^32 divided by the golden ratio, rounded to odd: SplitMix64's step and Fibonacci
 *  hashing's multipliers, each a bijection that spreads consecutive integers apart. */
inline constexpr u64_t golden_ratio_u64_k = 0x9E3779B97F4A7C15ull;
inline constexpr u32_t golden_ratio_u32_k = 0x9E3779B9u;

/** Avalanches all 64 bits of @p word - the SplitMix64 finalizer, the one bit-mixing
 *  primitive behind the PRNG, the UUID hash, and the open-addressing probe starts. */
constexpr u64_t mix_bits(u64_t word) noexcept {
    word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
    word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
    return word ^ (word >> 31);
}

/** How far @p candidate sits past @p query on the Fibonacci-hashed ring, the hashes' difference
 *  modulo 2^32. Equal distances order by it, so each member of a duplicate class prefers its own
 *  successor on the ring, never the class's lowest ID. */
constexpr u32_t ring_offset(u32_t candidate, u32_t query) noexcept { return golden_ratio_u32_k * (candidate - query); }

/** The image of @p position under a bijection over `[0, count)` keyed by @p seed: two odd
 *  multiplies around an xor-shift over the next power of two, its cycle walked until the image
 *  lands below @p count, so the first few positions name distinct rows in constant expected time,
 *  never a scan. Positions at or past @p count have no image. */
constexpr u64_t seeded_bijection(u64_t position, u64_t count, u64_t seed) noexcept {
    if (count < 2) return 0;
    std::size_t const bits = static_cast<std::size_t>(std::bit_width(count - 1));
    u64_t const mask = bits == 64 ? ~u64_t {0} : (u64_t {1} << bits) - 1;
    u64_t const first_multiplier = mix_bits(seed) | 1u, second_multiplier = mix_bits(seed + 1) | 1u;
    u64_t const offset = mix_bits(seed + 2);
    u64_t image = position;
    do {
        image = (image * first_multiplier + offset) & mask;
        image ^= image >> (bits / 2 + 1);
        image = (image * second_multiplier) & mask;
    } while (image >= count);
    return image;
}

/**
 *  @brief Minimal 8-byte SplitMix64 engine for stochastic NSW start-point selection - the
 *      cheap-to-construct sibling of the standard engines, paired with @ref uniform_below
 *      the way the standard pairs engines with distributions.
 *
 *  Picked over @c std::mt19937 (5 KB of per-thread state on libstdc++) and @c std::minstd_rand
 *  (a per-call 64-bit modulo). Quality is ample for spreading entry points; each traversal
 *  context owns one, seeded distinctly per thread or warp lane so the streams decorrelate.
 *  Every member is @c constexpr, so CUDA toolchains compiling with relaxed-constexpr call
 *  it from device code through this one definition. Satisfies the standard
 *  @c uniform_random_bit_generator concept, so the standard distributions accept it too.
 */
struct splitmix64_t {
    using result_type = u64_t;

    u64_t state = 0;

    static constexpr u64_t min() noexcept { return 0; }
    static constexpr u64_t max() noexcept { return ~u64_t(0); }

    /** One SplitMix64 step: 64 fresh uniform bits. */
    constexpr u64_t operator()() noexcept { return mix_bits(state += golden_ratio_u64_k); }
};

/** A SplitMix64 stream for @p key, decorrelated from the streams of its neighboring keys. */
constexpr splitmix64_t seeded_stream(u64_t key) noexcept { return splitmix64_t {mix_bits(key)}; }

/**
 *  @brief Uniform index distribution over [0, @c bound) - the cheap sibling of
 *      @c std::uniform_int_distribution: an aggregate holding one word, drawing through a single
 *      width-appropriate multiply-shift, no rejection loop.
 *
 *  Bias is about @c bound / 2^w for the index's bit width w, negligible for entry-point selection.
 *
 *  The random operand narrows to the index width @b before the multiply: wider surviving
 *  engine bits would land in the kept half and escape the bound.
 */
template <typename index_type_>
struct uniform_below;

template <>
struct uniform_below<u32_t> {
    u32_t bound = 0;

    /** One draw, through plain 64-bit arithmetic. */
    template <typename generator_type_>
    constexpr u32_t operator()(generator_type_& generator) const noexcept {
        return static_cast<u32_t>(static_cast<u64_t>(static_cast<u32_t>(generator())) * bound >> 32u);
    }
};

template <>
struct uniform_below<u64_t> {
    u64_t bound = 0;

    /** One draw, through the high half of the 128-bit product. @ref multiply_high */
    template <typename generator_type_>
    constexpr u64_t operator()(generator_type_& generator) const noexcept {
        return multiply_high(static_cast<u64_t>(generator()), bound);
    }
};

/** Graph node identifier within one sub-index - 32 bits, matching the 4-billion ceiling. */
using node_id_t = u32_t;

/** Distance representation shared by all metrics, regardless of the stored dtype. */
using distance_t = f32_t;

/** Reserved all-ones value, never a valid node ID within a sub-index. Doubles as the
 *  lock & empty-slot sentinel in neighbor lists of every index design. */
inline constexpr node_id_t node_id_unknown_k = 0xFFFFFFFFu;

/** One traversal candidate: a node ID paired with its distance to the query. */
struct scored_node_t {

    /** Identifier of the scored node within its sub-index. */
    node_id_t node_id;

    /** Distance to the query, in the instantiated kernel's metric. */
    distance_t distance;
};

/** Warps per thread block across every kernel of the VRAM design - four schedulers per
 *  multiprocessor on every modern GPU, one lane group each. */
inline constexpr u32_t vram_warps_per_block_k = 4;

/** Ceiling on concurrently in-flight candidate-vector gathers per lane - the memory-parallelism
 *  window over the @b candidate axis; the NVIDIA index also caps the lanes sharing one row at it,
 *  so a wide row takes more steps rather than more reductions. */
inline constexpr u32_t vram_max_gathers_in_flight_k = 8;

/** IDs the first insert wave over an empty shard admits at once, a measured width: every GPU design
 *  ramps its waves up from it. */
inline constexpr u32_t vram_admission_ramp_floor_k = 32;

/** Shares of the rows already below it an insert wave admits, as negative powers of two, both
 *  measured: a dense wave as wide as those rows, which its back-links splice in before the next
 *  wave walks, and a basic wave a quarter of them, the widest at which re-selected waves still
 *  reach the recall of rows inserted one at a time, since a basic wave's rows link one another
 *  only through the walk after the merge. */
inline constexpr u32_t vram_dense_admission_share_log2_k = 0;
inline constexpr u32_t vram_basic_admission_share_log2_k = 2;

/** The end of the insert wave admitting from @p wave_first, its share @p share_log2 of the rows
 *  below it and never under @c vram_admission_ramp_floor_k, short of @p end_id. */
constexpr u64_t vram_admission_wave_end(u64_t wave_first, u64_t end_id, u32_t share_log2) noexcept {
    return min_of(end_id, wave_first + max_of(wave_first >> share_log2, u64_t {vram_admission_ramp_floor_k}));
}

/** Slots of the widest register rung on the walks' rung ladder,
 *  @c vram_sorted_candidates<16,warp_t>: as many ranks per lane. A walk expanding past that rung's
 *  width - this times the warp's lanes, 512 on 32-lane warps - takes the arena frontier instead, so
 *  no expansion or count is capped by it. */
inline constexpr u32_t vram_widest_candidate_slots_k = 16;

/** Device-resident telemetry counters, one block per driving context, zeroed at each submission
 *  and read after its drain. */
struct vram_telemetry_block_t {
    u64_t hops, scored, lock_attempts, lock_skips, reselections, restarts, locked_drops;

    /** Scored candidates split by storage residency; zero unless the view names a resident
     *  range. */
    u64_t local_scored, remote_scored;

    /** Frontier tails compacted to what can still rank, entries tied at that radius a compacted
     *  tail still could not fit, and visited-set claims out of probes, which re-scored. */
    u64_t tail_compactions, tail_dropped, visited_saturations;

    /** Nodes whose neighbors a refinement merge landed in a different sequence than they held. */
    u64_t changed_nodes;
};

/**
 *  @brief An index's claim word: past the watermark by the rows of the one batch or load in
 *      flight, claimed from the watermark by one compare-exchange, or the sentinel while a caller
 *      holds it closed, so a second claimant refuses rather than waits.
 *
 *  The watermark, the rows visible, is the index's own, and every call takes it as @p visible.
 */
struct claim_word_t {

    /** The word's value while a caller holds it closed, past any row count. */
    static constexpr std::size_t closed_k = std::numeric_limits<std::size_t>::max();

    std::size_t claimed_rows = 0;

    /** Whether a caller holds the word closed. */
    bool closed() const noexcept {
        return standard_atomic_ref<std::size_t const>(claimed_rows).load(std::memory_order_acquire) == closed_k;
    }

    /** Rows claimed by batches and loads, visible or not; a closed word holds none past
     *  @p visible. */
    std::size_t claimed(std::size_t visible) const noexcept {
        std::size_t const claimed =
            standard_atomic_ref<std::size_t const>(claimed_rows).load(std::memory_order_acquire);
        return claimed == closed_k ? visible : claimed;
    }

    /** Claims @p rows from @p visible: one compare-exchange that succeeds only on an open word
     *  with no batch or load in flight, so its holder alone grows, launches and publishes them. */
    expected<std::size_t> claim(std::size_t visible, std::size_t rows, std::size_t max_rows) noexcept {
        std::size_t first = visible;
        if (rows > max_rows - first) return {{}, status_t::overflow_k};
        if (!standard_atomic_ref<std::size_t>(claimed_rows)
                 .compare_exchange_strong(first, first + rows, std::memory_order_acq_rel, std::memory_order_relaxed))
            return {{}, status_t::contended_k};
        return {first, status_t::success_k};
    }

    /** Gives back the claim over @p rows from @p first, which launched nothing; a word that moved
     *  on, closed by a caller around a load, stays as it is. */
    void unclaim(std::size_t first, std::size_t rows) noexcept {
        std::size_t claimed = first + rows;
        standard_atomic_ref<std::size_t>(claimed_rows)
            .compare_exchange_strong(claimed, first, std::memory_order_release, std::memory_order_relaxed);
    }

    /** Closes the word at @p visible, so every claim refuses until @ref reopen; refused while a
     *  batch or load is in flight. */
    status_t close(std::size_t visible) noexcept {
        return standard_atomic_ref<std::size_t>(claimed_rows)
                       .compare_exchange_strong(visible, closed_k, std::memory_order_acq_rel, std::memory_order_relaxed)
                   ? status_t::success_k
                   : status_t::contended_k;
    }

    /** Reopens a closed word at @p visible, past every row a load published meanwhile; an open
     *  one stays as it is. */
    void reopen(std::size_t visible) noexcept {
        std::size_t closed = closed_k;
        standard_atomic_ref<std::size_t>(claimed_rows)
            .compare_exchange_strong(closed, visible, std::memory_order_acq_rel, std::memory_order_relaxed);
    }
};

} // namespace unum::usearch::utilities

/** Makes @c uuid_t a first-class key of standard-hash-keyed containers: fold one mixed
 *  half into the other, then mix again - @ref unum::usearch::utilities::mix_bits twice. */
template <>
struct std::hash<unum::usearch::utilities::uuid_t> {
    std::size_t operator()(unum::usearch::utilities::uuid_t const& key) const noexcept {
        namespace utilities = unum::usearch::utilities;
        struct halves_t {
            utilities::u64_t low, high;
        };
        auto const halves = std::bit_cast<halves_t>(key);
        return static_cast<std::size_t>(utilities::mix_bits(halves.low ^ utilities::mix_bits(halves.high)));
    }
};
