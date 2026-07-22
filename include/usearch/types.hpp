/**
 *  @brief Vector Search helper structures and types
 *  @author Ash Vardanian
 *  @file include/usearch/types.hpp
 *  @date July 17, 2026
 */
#pragma once
#include <cassert> // `assert` on the `to_span` contiguity contract
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memmove`

#include <atomic>      // `std::atomic`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::type_identity_t`
#include <thread>      // `std::this_thread::yield`

/** @brief Pre-announced sister-library namespaces, so the project-wide shortcuts below are
 *      legal before any of their headers is included. */
namespace ashvardanian {
namespace numkong {}
namespace forkunion {}
namespace stringzilla {}
} // namespace ashvardanian

namespace unum {
namespace usearch {

namespace nk = ashvardanian::numkong;
namespace fu = ashvardanian::forkunion;
namespace sz = ashvardanian::stringzilla;

using u32_t = std::uint32_t;
using u64_t = std::uint64_t;
using f32_t = float;

/** @brief Dense node identifier within one sub-index - 32 bits, matching the 4-billion ceiling. */
using inner_node_id_t = u32_t;
/** @brief Distance representation shared by all metrics, regardless of the stored dtype. */
using inner_distance_t = f32_t;

/** @brief Reserved all-ones value, never a valid node id within a sub-index. Doubles as the
 *      lock & empty-slot sentinel in neighbor lists of every index design. */
inline constexpr inner_node_id_t inner_node_id_unknown_k = 0xFFFFFFFFu;

/** @brief One traversal candidate: a node id paired with its distance to the query. */
struct inner_scored_node_t {
    /** @brief Identifier of the scored node within its sub-index. */
    inner_node_id_t node_id;
    /** @brief Distance to the query, in the instantiated kernel's metric. */
    inner_distance_t distance;
};

/** @brief Orders scored nodes nearest-first; the comparator of the sorted results pool.
 *      The flag publishes that this order is the ascending order of each packed element's
 *      high 32 bits read as @c f32 - the contract SIMD container siblings key on. */
struct inner_closer_scored_node_t {
    static constexpr bool ascends_by_high_f32_bits_k = true;
    inline bool operator()(inner_scored_node_t a, inner_scored_node_t b) const noexcept {
        return a.distance < b.distance;
    }
};

/** @brief Orders scored nodes furthest-first; the comparator of the max-heap frontier.
 *      The flag mirrors @c inner_closer_scored_node_t for the descending direction. */
struct inner_further_scored_node_t {
    static constexpr bool descends_by_high_f32_bits_k = true;
    inline bool operator()(inner_scored_node_t a, inner_scored_node_t b) const noexcept {
        return a.distance > b.distance;
    }
};

/**
 *  @brief Minimal 8-byte SplitMix64 PRNG for stochastic NSW start-point selection.
 *
 *  Picked over @c std::mt19937 (5 KB of per-thread state on libstdc++) and @c std::minstd_rand
 *  (a per-call 64-bit modulo). Quality is ample for spreading entry points; each traversal
 *  context owns one, seeded distinctly per thread or warp lane so the streams decorrelate.
 *  Every member is @c constexpr, so CUDA toolchains compiling with relaxed-constexpr call
 *  it from device code through this one definition.
 */
struct splitmix_u32_t {
    using seed_t = u64_t;
    seed_t state_;

    constexpr explicit splitmix_u32_t(seed_t seed = 0) noexcept : state_(seed) {}

    /** @brief One SplitMix64 step. */
    constexpr seed_t unbounded() noexcept {
        seed_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /** @brief Index in [0, @p bound) via Lemire's multiply-shift - one multiply, no division.
     *      Bias is about @p bound / 2^32, negligible for entry-point selection. The random operand
     *      must be narrowed to 32 bits @b before widening: multiplying the full 64-bit state
     *      would let its high bits survive the shift and escape the bound. */
    constexpr u32_t bounded(u32_t bound) noexcept {
        return static_cast<u32_t>(static_cast<u64_t>(static_cast<u32_t>(unbounded())) * bound >> 32);
    }
};

enum class status_t {
    success_k,
    /** @brief The underlying allocator failed to provide memory; retrying a smaller batch may succeed. */
    bad_alloc_k,
    /** @brief The structure reached a hard design ceiling, like the 2^32 id space; retrying can't help. */
    overflow_k,
    /** @brief The operation requires a prior successful @c reserve() -like initialization call. */
    uninitialized_k,
    /** @brief The requested configuration contradicts an already-frozen one, like re-reserving
     *      with a different geometry; pass matching parameters or start a fresh structure. */
    config_mismatch_k,
    /** @brief An accelerator runtime call failed - the device state should be treated as
     *      poisoned and the structure rebuilt. */
    device_error_k,
    /** @brief A certified synchronization contract was violated by a concurrent operation;
     *      discard any produced output and retry in a quieter phase. */
    contended_k,
    /** @brief The serialized payload is truncated or does not describe this structure. */
    malformed_k,
};

/** @brief A result paired with the @c status_t explaining it; the value is only meaningful
 *      on @c success_k. */
template <typename value_type_>
struct expected {
    value_type_ value;
    status_t status;
};

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

enum connectivity_t : std::uint32_t {
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
 *      plus the @c bin16_k identities and @c utf8_k strings that interchange formats carry
 *      and a compute library has no place for. Interchange formats themselves only name
 *      types as strings over raw blobs; this enum is the USearch-side vocabulary every
 *      reader parses those spellings into.
 */
enum class dtype_t : u32_t {
    unknown_k = 0,
    // Integers, by shrinking dynamic range.
    u64_k,
    i64_k,
    u32_k,
    i32_k,
    u16_k,
    i16_k,
    u8_k,
    i8_k,
    // Floats, by shrinking dynamic range.
    f64_k,
    f32_k,
    bf16_k,
    f16_k,
    e5m2_k,
    e4m3_k,
    // Non-numeric payloads interchange formats carry.
    bin16_k,
    utf8_k,
};

/** @brief The canonical spelling of @p dtype - the very strings interchange headers carry. */
constexpr char const* dtype_name(dtype_t dtype) noexcept {
    switch (dtype) {
    case dtype_t::u64_k: return "u64";
    case dtype_t::i64_k: return "i64";
    case dtype_t::u32_k: return "u32";
    case dtype_t::i32_k: return "i32";
    case dtype_t::u16_k: return "u16";
    case dtype_t::i16_k: return "i16";
    case dtype_t::u8_k: return "u8";
    case dtype_t::i8_k: return "i8";
    case dtype_t::f64_k: return "f64";
    case dtype_t::f32_k: return "f32";
    case dtype_t::bf16_k: return "bf16";
    case dtype_t::f16_k: return "f16";
    case dtype_t::e5m2_k: return "e5m2";
    case dtype_t::e4m3_k: return "e4m3";
    case dtype_t::bin16_k: return "bin16";
    case dtype_t::utf8_k: return "utf8";
    case dtype_t::unknown_k: return "unknown";
    }
    return "unknown";
}

/** @brief Parses a canonical dtype spelling - the reverse of @ref dtype_name - mapping
 *      anything unrecognized onto @c unknown_k. */
constexpr dtype_t dtype_named(std::string_view name) noexcept {
    if (name == "u64") return dtype_t::u64_k;
    if (name == "i64") return dtype_t::i64_k;
    if (name == "u32") return dtype_t::u32_k;
    if (name == "i32") return dtype_t::i32_k;
    if (name == "u16") return dtype_t::u16_k;
    if (name == "i16") return dtype_t::i16_k;
    if (name == "u8") return dtype_t::u8_k;
    if (name == "i8") return dtype_t::i8_k;
    if (name == "f64") return dtype_t::f64_k;
    if (name == "f32") return dtype_t::f32_k;
    if (name == "bf16") return dtype_t::bf16_k;
    if (name == "f16") return dtype_t::f16_k;
    if (name == "e5m2") return dtype_t::e5m2_k;
    if (name == "e4m3") return dtype_t::e4m3_k;
    if (name == "bin16") return dtype_t::bin16_k;
    if (name == "utf8") return dtype_t::utf8_k;
    return dtype_t::unknown_k;
}

/**
 *  @brief Fixed 32-byte descriptor fronting every serialized inner index payload - volatile
 *      in-RAM snapshots, GPU-resident uploads, and persisted on-NVMe layouts alike. It describes
 *      the graph & its frozen geometry, never the residency: what sections follow and how they
 *      are laid out is the per-storage contract versioned by @c version_k.
 */
struct inner_index_header_t {
    /** @brief The expected @c magic contents: "usearch3" read as a little-endian word. */
    inline static constexpr u64_t magic_k = 0x3368637261657375ull;
    /** @brief The layout revision this implementation writes and understands. */
    inline static constexpr u32_t version_k = 1;

    /** @brief Identifies USearch inner index payloads; must equal @c magic_k. */
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
static_assert(sizeof(inner_index_header_t) == 32 && std::is_trivially_copyable_v<inner_index_header_t>);

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

} // namespace usearch
} // namespace unum
