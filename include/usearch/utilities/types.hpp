/**
 *  @brief Vector Search helper structures and types
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/types.hpp
 *  @date July 17, 2026
 */
#pragma once
#include <cassert> // `assert` on the `to_span` contiguity contract
#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memmove`, `std::memcmp`, `std::memcpy`

#include <atomic>      // `std::atomic`
#include <bit>         // `std::bit_cast` - hashing `uuid_t` as two words
#include <concepts>    // `std::convertible_to` - the string-view conversion's constraint
#include <functional>  // `std::hash` - specialized for `uuid_t` below
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::type_identity_t`
#include <thread>      // `std::this_thread::yield`

#include "usearch/types.h" // `usearch_status_t`, so the C++ mirror cannot drift from the ABI

/** @brief Pre-announced sister-library namespaces, so the project-wide shortcuts below are
 *      legal before any of their headers is included. */
namespace ashvardanian {
namespace numkong {}
namespace stringzilla {}
namespace forkunion {}
namespace smashtable {}
} // namespace ashvardanian

namespace unum {
namespace usearch {

namespace nk = ashvardanian::numkong;
namespace fu = ashvardanian::forkunion;
namespace sz = ashvardanian::stringzilla;
namespace st = ashvardanian::smashtable;

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

/** @brief "usearch3" read as a little-endian word - the one magic fronting every USearch
 *      payload; the word right after it discriminates the layout. */
inline constexpr u64_t usearch_magic_k = 0x3368637261657375ull;

/** @brief The 128-bit key default - the SafeGraphs @c uuid column, one UUID per node. */
struct uuid_t {
    std::byte bytes[16];

    bool operator==(uuid_t const& other) const noexcept { return std::memcmp(bytes, other.bytes, sizeof(bytes)) == 0; }
};
static_assert(std::is_trivially_copyable_v<uuid_t> && sizeof(uuid_t) == 16);

/** @brief Dense node identifier within one sub-index - 32 bits, matching the 4-billion ceiling. */
using dense_node_id_t = u32_t;
/** @brief Distance representation shared by all metrics, regardless of the stored dtype. */
using dense_distance_t = f32_t;

/** @brief Reserved all-ones value, never a valid node id within a sub-index. Doubles as the
 *      lock & empty-slot sentinel in neighbor lists of every index design. */
inline constexpr dense_node_id_t dense_node_id_unknown_k = 0xFFFFFFFFu;

/** @brief The single-byte scalar every allocator deals in - the unit of raw storage. */
using byte_t = std::byte;

/** @brief A mutable run of raw bytes - the staging & carve currency of writers. */
using bytes_span_t = std::span<byte_t>;

/** @brief A read-only run of raw bytes - the currency of readers & mapped payloads. */
using bytes_view_t = std::span<byte_t const>;

/** @brief The bytes behind @p count typed elements at @p data, read-only. */
template <typename value_type_>
bytes_view_t bytes_of(value_type_ const* data, std::size_t count) noexcept {
    return std::as_bytes(std::span {data, count});
}

/** @brief The bytes behind @p count typed elements at @p data, writable. */
template <typename value_type_>
bytes_span_t bytes_of(value_type_* data, std::size_t count) noexcept {
    return std::as_writable_bytes(std::span {data, count});
}

/** @brief The process-wide allocator, named rather than defaulted: a container takes this when
 *      no arena, memory domain or device owns its bytes, so every such placement is one grep
 *      away instead of hiding in an omitted argument. */
struct global_allocator_t {};
inline constexpr global_allocator_t global_allocator {};

/** @brief The one project-wide default alignment: 128 bytes covers x86's adjacent-line
 *      prefetcher, the 128-byte cache lines of Apple's and other Arm cores, and a CUDA
 *      coalesced transaction alike, where a 64-byte guess serves only one of them.
 *  @see @c fu::default_alignment_k in ForkUnion's @c types.hpp, which settles on the same
 *      value and documents why @c std::hardware_destructive_interference_size cannot. */
inline constexpr std::size_t default_alignment_k = 128;

/** @brief One traversal candidate: a node id paired with its distance to the query. */
struct dense_scored_node_t {
    /** @brief Identifier of the scored node within its sub-index. */
    dense_node_id_t node_id;
    /** @brief Distance to the query, in the instantiated kernel's metric. */
    dense_distance_t distance;
};

/** @brief Orders scored nodes nearest-first; the comparator of the sorted results pool. */
struct dense_closer_scored_node_t {
    inline bool operator()(dense_scored_node_t first, dense_scored_node_t second) const noexcept {
        return first.distance < second.distance;
    }
};

/** @brief Orders scored nodes furthest-first; the comparator of the max-heap frontier. */
struct dense_further_scored_node_t {
    inline bool operator()(dense_scored_node_t first, dense_scored_node_t second) const noexcept {
        return first.distance > second.distance;
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
    /** @brief The call completed and any paired value is meaningful. */
    success_k = usearch_success_k,
    /** @brief The underlying allocator failed to provide memory; retrying a smaller batch may succeed. */
    bad_alloc_k = usearch_error_out_of_memory_k,
    /** @brief Dimensionality, row counts, or output capacity disagree. */
    shape_mismatch_k = usearch_error_shape_mismatch_k,
    /** @brief A stride or packing no traversal can walk. */
    unsupported_layout_k = usearch_error_unsupported_layout_k,
    /** @brief A scalar, metric, or residency tag this build does not know. */
    unknown_kind_k = usearch_error_unknown_kind_k,
    /** @brief A mutating verb on a structure that opens its checkpoint in place. */
    read_only_k = usearch_error_read_only_k,
    /** @brief A file could not be opened, read, written, or parsed. */
    io_k = usearch_error_io_k,
    /** @brief An accelerator runtime call failed - the device state should be treated as
     *      poisoned and the structure rebuilt. */
    device_error_k = usearch_error_device_k,
    /** @brief The serialized payload is truncated or does not describe this structure. */
    malformed_k = usearch_error_malformed_input_k,
    /** @brief Neither success nor a named failure was reported - a defect, not a state. */
    internal_k = usearch_error_internal_k,
    /** @brief The structure reached a hard design ceiling, like the 2^32 id space; retrying can't help. */
    overflow_k = usearch_error_overflow_k,
    /** @brief The operation requires a prior successful @c reserve() -like initialization call. */
    uninitialized_k = usearch_error_uninitialized_k,
    /** @brief The requested configuration contradicts an already-frozen one, like re-reserving
     *      with a different geometry; pass matching parameters or start a fresh structure. */
    config_mismatch_k = usearch_error_config_mismatch_k,
    /** @brief A certified synchronization contract was violated by a concurrent operation;
     *      discard any produced output and retry in a quieter phase. */
    contended_k = usearch_error_contended_k,
    /** @brief A progress callback asked the operation to stop; it unwound without publishing,
     *      so the structure carries whatever it did before the call. */
    cancelled_k = usearch_error_cancelled_k,
};

/** @brief Whether @p status reports success. */
constexpr bool succeeded(status_t status) noexcept { return status == status_t::success_k; }

/** @brief Whether @p status reports failure. */
constexpr bool failed(status_t status) noexcept { return status != status_t::success_k; }

/** @brief Copies @p source over @p destination; the sizes must agree. */
inline status_t copy_bytes(bytes_span_t destination, bytes_view_t source) noexcept {
    if (destination.size() != source.size()) return status_t::config_mismatch_k;
    if (!source.empty()) std::memcpy(destination.data(), source.data(), source.size());
    return status_t::success_k;
}

/** @brief Whether a filtered search admits one candidate or passes over it - rejected nodes
 *      still route the walk, they just never enter the reported pool. Mirrors
 *      @c usearch_admission_t value for value, so the two enums cannot drift. */
enum class admission_t : int {
    /** @brief Skip the node; the walk still routes through it. */
    reject_k = usearch_reject_k,
    /** @brief Report the node among the matches. */
    admit_k = usearch_admit_k,
};

/** @brief Whether the operation a progress callback is watching carries on or unwinds.
 *      Mirrors @c usearch_continuation_t value for value, so the two enums cannot drift. */
enum class continuation_t : int {
    /** @brief Carry on to the next unit of work. */
    continue_k = usearch_continue_k,
    /** @brief Unwind: the verb publishes nothing further and reports @c cancelled_k. */
    cancel_k = usearch_cancel_k,
};

/** @brief Whether a verb returns when its work is done, or once it is merely queued on the
 *      executor. Mirrors @c usearch_completion_t value for value, so the two enums cannot drift. */
enum class completion_t : int {
    /** @brief Return once the work is finished and its outputs are readable. */
    blocking_k = usearch_completion_blocking_k,
    /** @brief Return once the work is queued; outputs stay indeterminate until a drain. */
    deferred_k = usearch_completion_deferred_k,
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
    }
    return "an unrecognized status";
}

/** @brief A result paired with the @c status_t explaining it; the value is only meaningful
 *      on @c success_k. */
template <typename value_type_>
struct [[nodiscard]] expected {
    value_type_ value;
    status_t status;

    explicit operator bool() const noexcept { return succeeded(status); }
};

/** @brief The borrowing face of @ref expected for reference results: there is no null
 *      reference, so the borrow is stored as an address and @c value() binds it only on
 *      @c success_k. */
template <typename value_type_>
struct [[nodiscard]] expected<value_type_&> {
    value_type_* borrowed;
    status_t status;

    value_type_& value() const noexcept { return *borrowed; }
    explicit operator bool() const noexcept { return succeeded(status); }
};

/** @brief A half-open interval of integer identifiers, iterable like the standard views -
 *      the unit callers partition work by, one contiguous run per core. */
template <typename index_type_ = std::size_t>
struct iota_view {
    /** @brief The first identifier of the interval. */
    index_type_ first {0};
    /** @brief One past the last identifier. */
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

/**
 *  @brief A range over an indexed accessor: @c owner[position] for every position below @p count,
 *      so a caller reads @c for(auto element : range) rather than counting positions itself.
 */
template <typename owner_type_>
struct indexed_range {
    /** @brief What the positions are read from; borrowed, never owned. */
    owner_type_ const* owner = nullptr;
    /** @brief Positions the range covers, from zero. */
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

/** @brief Moves neighbor IDs between id spaces: an ID inside @p source_range lands at
 *      @c id-source_range.first+shift, every other ID folds onto @p sentinel, which a walk
 *      reads as naming no node. */
template <std::unsigned_integral id_type_>
constexpr void rebase_neighbors(std::span<id_type_> neighbors, iota_view<id_type_> source_range, id_type_ shift,
                                id_type_ sentinel) noexcept {
    for (id_type_& neighbor : neighbors)
        neighbor = source_range.first <= neighbor && neighbor < source_range.bound //
                       ? static_cast<id_type_>(neighbor - source_range.first + shift)
                       : sentinel;
}

/**
 *  @brief Packs the surviving ids of each row of @p rows to its front and pads the tail with
 *      @p sentinel, writing @p first_row + row as a self-loop for a row nothing survived in.
 *
 *  A rebase folds every foreign id onto the sentinel wherever it sat, and a concurrent walk
 *  reads a sentinel in slot 0 as a node locked by a live writer - which a loaded graph has none
 *  of, so it would spin forever. Compacting restores what a live index maintains: slot 0 is a
 *  real id, and the sentinel only ever ends a row.
 */
template <std::unsigned_integral id_type_>
constexpr void compact_neighbors(std::span<id_type_> rows, std::size_t connectivity, id_type_ first_row,
                                 id_type_ sentinel) noexcept {
    if (!connectivity) return;
    for (std::size_t row = 0; row * connectivity != rows.size(); ++row) {
        std::span<id_type_> const slots = rows.subspan(row * connectivity, connectivity);
        std::size_t kept = 0;
        for (std::size_t slot = 0; slot != connectivity; ++slot)
            if (slots[slot] != sentinel) slots[kept++] = slots[slot];
        if (!kept) slots[kept++] = static_cast<id_type_>(first_row + row);
        for (std::size_t slot = kept; slot != connectivity; ++slot) slots[slot] = sentinel;
    }
}

/** @brief The concurrency environment a caller certifies for one operation. */
enum class synchronization_t : u32_t {
    /** @brief Arbitrary readers & writers may be live - the safe default everywhere. */
    concurrent_k,
    /** @brief Readers may be live, but no mutators: neighbor lists are read as plain memory -
     *      no fences - and the locked-node retry branches compile out entirely. */
    immutable_k,
    /** @brief Sole ownership - nothing else touches the index, readers included. */
    exclusive_k,
};

enum connectivity_t : u32_t {
    connectivity_unknown_k = 0,
    /**
     *  @brief The minimal recommended graph connectivity level, expanding to 16x 32-bit neighbor IDs,
     *      exactly matching the ZMM register size and cache-line width on x86, but half of the width
     *      of coherency protocols of many CPUs and half the cache line width of many Arm CPUs.
     */
    connectivity_16_k = 16,
    /**
     *  @brief The default recommended graph connectivity level, expanding to 32x 32-bit neighbor IDs,
     *      perfect for in-memory billion-scale search indexes.
     */
    connectivity_32_k = 32,
    /**
     *  @brief The minimal recommended graph connectivity level for on-disk billion- and trillion-scale
     *      search indexes, matching the minimal 512-byte SSD sector size.
     */
    connectivity_128_k = 128,
};

/**
 *  @brief Element types USearch speaks when exchanging payloads - NumKong's numeric dtypes
 *      plus the packed, opaque, identity and text families that interchange formats carry
 *      and a compute library has no place for. Every storage width also has a neutral
 *      spelling naming that many opaque bits, and every other spelling is an alias adding
 *      a reading to the same width; this enum is the USearch-side vocabulary every reader
 *      parses those spellings into.
 */
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

/** @brief The canonical spelling of @p dtype - the very strings interchange headers carry. */
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

/** @brief Parses a canonical dtype spelling - the reverse of @ref dtype_name - mapping
 *      anything unrecognized onto @c unknown_k. */
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

/** @brief Bits one element of @p dtype occupies - eight for a byte-wide scalar, four or one
 *      for the packed families, zero for the variable-length and unknown ones. */
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

/** @brief Bytes one element of @p dtype occupies; zero for the packed families, whose elements
 *      share a byte, and for the unknown one. */
constexpr std::size_t dtype_bytes(dtype_t dtype) noexcept {
    std::size_t const bits = dtype_bits(dtype);
    return bits && bits % 8 == 0 ? bits / 8 : 0;
}

/**
 *  @brief Fixed 32-byte descriptor fronting every serialized dense index payload - volatile
 *      in-RAM snapshots, GPU-resident uploads, and persisted on-NVMe layouts alike. It describes
 *      the graph & its frozen geometry, never the residency: what sections follow and how they
 *      are laid out is the per-storage contract versioned by @c version_k.
 */
struct dense_index_header_t {
    /** @brief The expected @c magic contents: "usearch3" read as a little-endian word. */
    inline static constexpr u64_t magic_k = usearch_magic_k;
    /** @brief The layout revision this implementation writes and understands. */
    inline static constexpr u32_t version_k = 1;

    /** @brief Identifies USearch dense index payloads; must equal @c magic_k. */
    u64_t magic;
    /** @brief Layout revision of the sections that follow. */
    u32_t version;
    /** @brief The stored @c nk_dtype_t; must match the loading instantiation. */
    u32_t dtype;
    /** @brief The @c nk_kernel_kind_t metric; must match the loading instantiation. */
    u32_t kernel;
    /** @brief Scalar dimensions per vector - one half of the frozen geometry. */
    u32_t dimensions;
    /** @brief Neighbor ids per node - the other half of the frozen geometry. */
    u32_t connectivity;
    /** @brief Published nodes in the payload; the chunk layout re-derives from it. */
    u32_t node_count;
};
static_assert(sizeof(dense_index_header_t) == 32 && std::is_trivially_copyable_v<dense_index_header_t>);

/** @brief Returns 2^ @p exponent as a value of the requested unsigned type. */
template <typename result_type_>
constexpr result_type_ power_of_two(u32_t exponent) noexcept {
    return result_type_(1) << exponent;
}

/** @brief Returns a mask selecting the lowest @p exponent bits, i.e. 2^ @p exponent - 1. */
template <typename result_type_>
constexpr result_type_ power_of_two_mask(u32_t exponent) noexcept {
    return (result_type_(1) << exponent) - result_type_(1);
}

/** @brief Integer division rounding towards positive infinity; @p divisor must be non-zero. */
template <typename value_type_>
constexpr value_type_ divide_round_up(value_type_ value, value_type_ divisor) noexcept {
    return (value + divisor - value_type_(1)) / divisor;
}

/** @brief The smallest multiple of @p multiple at or above @p value; @p multiple must be non-zero. */
template <typename value_type_>
constexpr value_type_ round_up_to_multiple(value_type_ value, value_type_ multiple) noexcept {
    return divide_round_up(value, multiple) * multiple;
}

/** @brief @p value with its two bytes reversed. */
constexpr std::uint16_t reversed_bytes(std::uint16_t value) noexcept {
    return static_cast<std::uint16_t>((value << 8) | (value >> 8));
}

/** @brief @p value with its four bytes reversed. */
constexpr std::uint32_t reversed_bytes(std::uint32_t value) noexcept {
    return (value << 24) | ((value & 0x0000FF00u) << 8) | ((value >> 8) & 0x0000FF00u) | (value >> 24);
}

/** @brief @p value with its eight bytes reversed, as the two halves swapped and each reversed. */
constexpr std::uint64_t reversed_bytes(std::uint64_t value) noexcept {
    return (static_cast<std::uint64_t>(reversed_bytes(static_cast<std::uint32_t>(value))) << 32) |
           reversed_bytes(static_cast<std::uint32_t>(value >> 32));
}

/** @brief @p value with its bytes in little-endian order, and the same call to read one back - the
 *      conversion is its own inverse, so one name serves both directions. A no-op where the host is
 *      already little-endian, and a @ref reversed_bytes where it is not. */
template <typename value_type_>
constexpr value_type_ little_endian(value_type_ value) noexcept {
    static_assert(std::is_unsigned_v<value_type_>, "Byte order is only defined for unsigned integers");
    if constexpr (std::endian::native == std::endian::little) return value;
    else return reversed_bytes(value);
}

/** @brief The smaller of @p first & @p second - @c std::min without the <algorithm> tax,
 *      named to also dodge the @c min macro of Windows headers. */
template <typename value_type_>
constexpr value_type_ min_of(value_type_ first, value_type_ second) noexcept {
    return second < first ? second : first;
}

/** @brief The larger of @p first & @p second - the @c max twin of @ref min_of. */
template <typename value_type_>
constexpr value_type_ max_of(value_type_ first, value_type_ second) noexcept {
    return first < second ? second : first;
}

/** @brief The universal matrix-engine row count: Intel AMX tiles, Arm SME's ZA at SVL-512,
 *      and NVIDIA mma fragments all speak 16 rows natively - Hopper's warpgroups speak 4×16.
 *      Tiled paths across every index design derive their shapes from it, never tune it. */
inline constexpr std::size_t tile_span_k = 16;

/**
 *  @brief The first position in the ascending @p values not ordered before @p value -
 *      @c std::lower_bound re-spelled in ten lines, so headers don't drag in
 *      @c <algorithm>, one of the heaviest includes on the compile-health charts.
 *  @see https://artificial-mind.net/projects/compile-health/
 */
template <typename value_type_>
constexpr value_type_* lower_bound(std::span<value_type_> values,
                                   std::type_identity_t<value_type_> const& value) noexcept {
    value_type_* first = values.data();
    std::size_t count = values.size();
    while (count) {
        std::size_t const half = count / 2;
        value_type_* const middle = first + half;
        bool const is_before = *middle < value;
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

/** @brief The writable sibling of @ref to_span, for a run the caller owns and fills; named
 *      apart rather than overloaded, since an overload would silently hand a mutable span to
 *      every existing caller holding a non-const run.
 *  @see @c std::as_bytes and @c std::as_writable_bytes, which separate the same pair. */
template <typename vector_type_>
    requires requires(vector_type_& vector) {
        vector.data();
        vector.size();
    }
constexpr auto to_writable_span(vector_type_& vector) noexcept {
    using value_t = std::remove_pointer_t<decltype(vector.data())>;
    return std::span<value_t> {vector.data(), vector.size()};
}

/** @brief The character sibling of @ref to_span: any contiguous run of @c char as a view. */
template <typename characters_type_>
    requires requires(characters_type_ const& characters) {
        { characters.data() } -> std::convertible_to<char const*>;
        characters.size();
    }
constexpr std::string_view to_string_view(characters_type_ const& characters) noexcept {
    return std::string_view {characters.data(), characters.size()};
}

/** @brief Any entry pairing a graph-local @c node_id with a @c distance - the shape that
 *      search results share across index designs. */
template <typename entry_type_>
concept scored_node = requires(entry_type_ entry) {
    entry.node_id;
    entry.distance;
};

/**
 *  @brief Merges the ascending @p results of one search into the caller's bounded ranking,
 *      keeping it ascending by distance.
 *
 *  Indexes store no keys, so ids are translated into the caller's key space on the way in -
 *  letting several sub-indexes with colliding id spaces, like one per memory domain, gather
 *  into one shared buffer. The parallel spans' common length is the ranking's capacity.
 *
 *  @param[in] results One index's search results, ascending by distance.
 *  @param[in] keys_of_ids Any id-to-key mapping with @c operator[] - a flat array, a raw
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

/** @brief Tag: an uncapped monitored wait - the watched word is the only wake source. */
struct wait_uncapped_t {};
/** @brief Tag: a capped monitored wait - re-checks on its own after a bounded pause. */
struct wait_capped_t {};

/**
 *  @brief Portable spin-wait baseline with the same call shape as ForkUnion's hardware wait
 *      monitors, like @c fu::x86_tpause_t, @c fu::arm64_wfet_t, and @c fu::risc5_wrs_t.
 *
 *  Ignores the watched word and simply yields the timeslice - correctness never depends on the
 *  waiter, only latency and power do. The first overload watches a @c std::atomic object; the
 *  second watches an in-place word owned through @c std::atomic_ref, like the slot-0 locks living
 *  inside neighbor rows. The waiting bound (capped or uncapped) arrives as an arbitrary tag type,
 *  so this header stays independent of ForkUnion.
 *
 *  The pointer overload exists because of pre-C++26 STL limitations: @c std::atomic_ref cannot
 *  bind const-qualified objects until P3323, @c std::atomic::wait offers no timed variant, and
 *  neither API exposes the address-armed hardware waits - @c UMWAIT, @c WFET, @c WRS - that
 *  ForkUnion reaches directly, so the raw address plus the observed bit pattern is the portable
 *  currency.
 */
struct dummy_waiter_t {
    template <typename value_type_, typename thread_index_type_, typename bound_type_>
    void operator()(std::atomic<value_type_> const&, value_type_, thread_index_type_, bound_type_) const noexcept {
        std::this_thread::yield();
    }
    template <typename value_type_, typename thread_index_type_, typename bound_type_>
    void operator()(value_type_ const*, value_type_, thread_index_type_, bound_type_) const noexcept {
        std::this_thread::yield();
    }
};

/** @brief The compiled-out default for optional callbacks, like an insertion's assigned-id
 *      observer: accepts any arguments and does nothing. */
struct dummy_callback_t {
    template <typename... args_types_>
    void operator()(args_types_&&...) const noexcept {}
};

/** @brief The compiled-out default filter: admits anything it is asked about. Walks key on its
 *      @b type to skip the filtered machinery whole, so passing it explicitly costs nothing and
 *      answers the same. */
struct dummy_predicate_t {
    template <typename... args_types_>
    constexpr admission_t operator()(args_types_&&...) const noexcept {
        return admission_t::admit_k;
    }
};

/** @brief The compiled-out default progress watcher: always carries on, so an unwatched long
 *      verb runs to completion with no cancellation checks surviving inlining. */
struct dummy_progress_t {
    template <typename... args_types_>
    constexpr continuation_t operator()(args_types_&&...) const noexcept {
        return continuation_t::continue_k;
    }
};

/**
 *  @brief The CPU families USearch ships tuned variants for - our own vocabulary, never an
 *      alias of a sister library's feature bits. Each class bundles the compute features,
 *      wait instruction, and atomics its silicon generation guarantees; explicit helpers
 *      lower a class onto NumKong, ForkUnion & StringZilla capability masks where those
 *      libraries are linked.
 */
enum class cpu_class_t : u32_t {
    unknown_k = 0,
    /** @brief The portable floor: no SIMD assumptions, standard yields. */
    serial_k,
    /** @brief Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    intel_haswell_k,
    /** @brief Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI & its INT8 / IFMA / NE-CONVERT
     *      successors without AVX-512, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_sierra_k,
    /** @brief Skylake-class x86: the AVX-512 foundation tier. */
    intel_skylake_k,
    /** @brief Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    intel_icelake_k,
    /** @brief Genoa-class x86 - Zen 4 and newer: native @c bf16 dot products over AVX-512. */
    amd_genoa_k,
    /** @brief Diamond Rapids-class x86: the AVX-512 lineage plus AVX10.2's FP8 conversions and
     *      the Sierra vector bundle, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_diamond_k,
    /** @brief Graviton2-class Arm - Neoverse N1, the bare NEON floor; pre-v8.6 phones land
     *      here too. */
    aws_g2_k,
    /** @brief Armv8.6+ NEON: dot products, @c bf16 & @c f16 fused multiplies. Named for
     *      Apple M2; Cortex-A710 & X2 - Snapdragon 8 Gen 1, AmpereOne - carry the same. */
    apple_m2_k,
    /** @brief Graviton5-class Arm - Neoverse V3, Armv9.2: the @c apple_m2_k bundle with
     *      @c FEAT_WFxT guaranteed, so waits monitor instead of spinning; M4-class too. */
    aws_g5_k,
    /** @brief Vera-class Arm - NVIDIA's v9.2+ cores with @c FEAT_FP8 dot products. */
    nvidia_vera_k,
    /** @brief Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23: the vector extension with
     *      @c Zvbb, @c Zawrs monitored waits, @c Zacas single-instruction compare-and-swap. */
    alibaba_c930_k,
};

/**
 *  @brief The GPU families USearch ships tuned variants for, the device twin of
 *      @ref cpu_class_t: one class per generation of kernel kits, lowered onto CUDA
 *      architecture ceilings through explicit helpers where the CUDA SDK is linked.
 */
enum class gpu_class_t : u32_t {
    unknown_k = 0,
    /** @brief Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    nvidia_hopper_k,
    /** @brief Blackwell-class NVIDIA - B200 & GB200: TMA @c gather4 scattered fetches
     *      and @c tcgen05 tensor-core GEMMs. */
    nvidia_blackwell_k,
    /** @brief Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    nvidia_rubin_k,
};

/** @brief Feature bits an I/O engine publishes through its @c capabilities_k constant, so
 *      callers, tests, and docs introspect variants uniformly instead of by type name. */
enum class io_capability_t : u32_t {
    none_k = 0,
    /** @brief Staging buffers are pre-registered with the kernel/ring - no per-read
     *      page pinning; roughly an 11% throughput gain at NVMe rates. */
    fixed_buffers_k = 1u << 0,
    /** @brief Completions are polled from device queues instead of interrupt-driven;
     *      needs poll-capable queues & direct I/O; ~21% on top of the fixed path. */
    polled_k = 1u << 1,
    /** @brief Requests address raw LBAs on a namespace, not file offsets - the
     *      filesystem & block layer are bypassed; ~20-30% over the block path. */
    lba_addressing_k = 1u << 2,
};

inline constexpr io_capability_t operator|(io_capability_t first, io_capability_t second) noexcept {
    return static_cast<io_capability_t>(static_cast<u32_t>(first) | static_cast<u32_t>(second));
}

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

/** @brief Avalanches all 64 bits of @p word - the SplitMix64 finalizer, the one bit-mixing
 *      primitive behind the PRNG, the UUID hash, and the open-addressing probe starts. */
constexpr u64_t mix_bits(u64_t word) noexcept {
    word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
    word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
    return word ^ (word >> 31);
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

    /** @brief One SplitMix64 step: 64 fresh uniform bits. */
    constexpr u64_t operator()() noexcept { return mix_bits(state += 0x9E3779B97F4A7C15ull); }
};

/**
 *  @brief Uniform index distribution over [0, @c bound) - the cheap sibling of
 *      @c std::uniform_int_distribution: an aggregate holding one word, drawing through a
 *      single width-appropriate multiply-shift, no rejection loop. Bias is about
 *      @c bound / 2^w for the index's bit width w, negligible for entry-point selection.
 *
 *  The random operand narrows to the index width @b before the multiply: wider surviving
 *  engine bits would land in the kept half and escape the bound.
 */
template <typename index_type_>
struct uniform_below;

template <>
struct uniform_below<u32_t> {
    u32_t bound = 0;

    /** @brief One draw, through plain 64-bit arithmetic. */
    template <typename generator_type_>
    constexpr u32_t operator()(generator_type_& generator) const noexcept {
        return static_cast<u32_t>(static_cast<u64_t>(static_cast<u32_t>(generator())) * bound >> 32u);
    }
};

template <>
struct uniform_below<u64_t> {
    u64_t bound = 0;

    /** @brief One draw, through the high half of the 128-bit product. @ref multiply_high */
    template <typename generator_type_>
    constexpr u64_t operator()(generator_type_& generator) const noexcept {
        return multiply_high(static_cast<u64_t>(generator()), bound);
    }
};

} // namespace usearch
} // namespace unum

/** @brief Makes @c uuid_t a first-class key of standard-hash-keyed containers: fold one
 *      mixed half into the other, then mix again - @ref unum::usearch::mix_bits twice. */
template <>
struct std::hash<unum::usearch::uuid_t> {
    std::size_t operator()(unum::usearch::uuid_t const& key) const noexcept {
        struct halves_t {
            unum::usearch::u64_t low, high;
        };
        auto const halves = std::bit_cast<halves_t>(key);
        return static_cast<std::size_t>(unum::usearch::mix_bits(halves.low ^ unum::usearch::mix_bits(halves.high)));
    }
};
