/**
 *  @brief Vector Search index for a single uniform latency memory space on H100 & H200 GPUs.
 *  @author Ash Vardanian
 *  @file include/usearch/inner_hopper_index.cuh
 *  @date July 18, 2026
 */
#pragma once

#include <bit>     // `std::bit_width` for hash-prefix widths
#include <cstring> // `std::memcpy`

#include <cuda.h>         // Driver API: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`
#include <cuda_runtime.h> // Runtime API: streams, events, `cudaMemcpyAsync`
#if defined(__CUDACC__)
#include <cuda/atomic> // `cuda::atomic_ref` - the device twin of `std::atomic_ref`
#endif

#include <forkunion/types.hpp> // `fu::round_up_to_multiple` & friends

#include <numkong/capabilities.h> // `nk_kernel_kind_t` & the kernel-kind constants
#include <numkong/matrix.hpp>
#include <numkong/types.h>
#include <numkong/vector.hpp>

#include "usearch/types.hpp"

namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;

namespace unum {
namespace usearch {

/** @brief Warps per thread block across every kernel of this design - four schedulers per
 *      Hopper SM, one warp each. */
inline constexpr u32_t inner_hopper_warps_per_block_k = 4;
/** @brief Rows of the per-warp shared candidate cache; also bounds the diversity-selection
 *      pool prefix and the staging arrays. */
inline constexpr u32_t inner_hopper_cache_rows_k = 64;
/** @brief Floor & jitter of the @c __nanosleep backoff while retrying a held slot-0 lock -
 *      measured to keep hub convoys tolerable on young graphs. */
inline constexpr u32_t lock_backoff_floor_ns_k = 512;
inline constexpr u32_t lock_backoff_jitter_mask_k = 1023;
/** @brief Acquire attempts before a traversal drops a transiently locked candidate. */
inline constexpr u32_t traversal_peek_retries_k = 16;
/** @brief Acquire attempts before a back-linker abandons one reverse edge - the bounded-wait
 *      policy; abandonments are telemetered and measured below one in a million. */
inline constexpr u32_t back_link_lock_attempts_k = 256;
/** @brief Admission-ramp floor: the in-flight window starts here and doubles with the
 *      published set, bounding bootstrap lock contention. */
inline constexpr u32_t admission_ramp_floor_k = 32;

#if defined(__CUDACC__)

/**
 *  @brief One traversal candidate packed into eight bytes, ordered by distance as raw
 *      unsigned words: bits [63:33] hold the top 31 bits of the sign-flipped @c f32
 *      distance, bit 32 carries the "already expanded" flag, bits [31:0] the full node id.
 *
 *  The all-ones pattern is the empty slot: its distance prefix is the flipped NaN that
 *  sorts last, it reads as expanded, and its id is @c inner_node_id_unknown_k - so
 *  traversal loops need no emptiness guards. Dropping the distance LSB costs one ulp of
 *  ordering precision: pools are "sorted by @c distance_bits(), tie order unspecified",
 *  which is the practical semantic of approximate search either way, and in exchange ids
 *  keep all 32 bits - the id ceiling matches the CPU-resident index exactly.
 */
struct inner_hopper_scored_node_t {
    u64_t bits_;

    inline static constexpr u64_t expanded_bit_k = u64_t(1) << 32;
    /** @brief Dispensable ids: everything but the reserved all-ones sentinel. */
    inline static constexpr std::size_t id_capacity_k = (u64_t(1) << 32) - 1u;

    static __device__ __forceinline__ inner_hopper_scored_node_t make(inner_distance_t distance,
                                                                      inner_node_id_t node_id) noexcept {
        u32_t flipped = __float_as_uint(distance);
        flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return {(static_cast<u64_t>(flipped >> 1) << 33) | node_id};
    }
    __device__ static constexpr inner_hopper_scored_node_t sentinel() noexcept { return {~u64_t(0)}; }

    /** @brief The flag-free comparison currency: 31 ordering bits of the flipped distance. */
    __device__ __forceinline__ u32_t distance_bits() const noexcept { return static_cast<u32_t>(bits_ >> 33); }
    /** @brief Decoded distance, with the truncated ordering LSB reconstructed as zero. */
    __device__ __forceinline__ inner_distance_t distance() const noexcept {
        u32_t flipped = distance_bits() << 1;
        flipped ^= (flipped & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
        return __uint_as_float(flipped);
    }
    __device__ __forceinline__ inner_node_id_t id() const noexcept { return static_cast<inner_node_id_t>(bits_); }
    __device__ __forceinline__ bool expanded() const noexcept { return bits_ & expanded_bit_k; }
    /** @brief Lazily deletes the entry from the frontier; a no-op on the sentinel. */
    __device__ __forceinline__ void mark_expanded() noexcept { bits_ |= expanded_bit_k; }
    __device__ __forceinline__ bool operator<(inner_hopper_scored_node_t other) const noexcept {
        return bits_ < other.bits_;
    }
    __device__ __forceinline__ bool operator==(inner_hopper_scored_node_t other) const noexcept {
        return bits_ == other.bits_;
    }

    /** @brief Warp shuffles have no user-type overloads - the entry shuffles its raw word. */
    __device__ __forceinline__ inner_hopper_scored_node_t shuffle_xor(int lane_mask) const noexcept {
        return {__shfl_xor_sync(0xFFFFFFFFu, bits_, lane_mask)};
    }
    __device__ __forceinline__ inner_hopper_scored_node_t shuffle_from(int source_lane) const noexcept {
        return {__shfl_sync(0xFFFFFFFFu, bits_, source_lane)};
    }
    __device__ __forceinline__ inner_hopper_scored_node_t shuffle_up(int delta) const noexcept {
        return {__shfl_up_sync(0xFFFFFFFFu, bits_, delta)};
    }
};
static_assert(sizeof(inner_hopper_scored_node_t) == 8, "Entries shuffle as one 64-bit word");
static_assert(std::is_trivially_copyable_v<inner_hopper_scored_node_t>);

/**
 *  @brief Full bitonic sort of one 32-lane entry vector, ascending by lane - fifteen
 *      shuffle-exchange stages, all in registers.
 */
__device__ __forceinline__ inner_hopper_scored_node_t inner_hopper_sort32(inner_hopper_scored_node_t value,
                                                                          int lane) noexcept {
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
        for (int j = k >> 1; j; j >>= 1) {
            inner_hopper_scored_node_t const partner = value.shuffle_xor(j);
            bool const up = (lane & k) == 0;
            bool const lower = (lane & j) == 0;
            inner_hopper_scored_node_t const nearer = value < partner ? value : partner;
            inner_hopper_scored_node_t const further = value < partner ? partner : value;
            value = (up == lower) ? nearer : further;
        }
    return value;
}

/**
 *  @brief Warp-collective ascending pool of 32x @c slots_ scored entries - the fused
 *      results-and-frontier of one traversal, striped as rank @b r at (slot r/32,
 *      lane r%32), always sorted by @c distance_bits() with unspecified tie order.
 *
 *  Subsumes both host-side traversal containers: the sorted results buffer (the radius
 *  lives at the last rank) and the frontier heap (the nearest unexpanded entry plays
 *  @c top(), the expanded bit plays lazy deletion). Two register-residency invariants are
 *  part of the contract: every slot index is compile-time after unrolling - a single
 *  runtime-indexed access spills the whole pool to local memory - and batch insertion
 *  goes through the fold-merge network below, never a full re-sort.
 */
template <u32_t slots_>
struct inner_hopper_sorted_pool {
    inner_hopper_scored_node_t entries_[slots_];

    inline static constexpr u32_t capacity_k = slots_ * 32;
    static_assert((slots_ & (slots_ - 1)) == 0, "The fold-merge network needs a power-of-two slot count");

    __device__ __forceinline__ void clear() noexcept {
#pragma unroll
        for (u32_t slot = 0; slot != slots_; ++slot) entries_[slot] = inner_hopper_scored_node_t::sentinel();
    }

    /**
     *  @brief Merges a 32-lane staged batch into the pool, keeping the 32x @c slots_
     *      nearest. [ascending pool | sentinel pad | descending batch] is one bitonic
     *      sequence: a single fold absorbs the batch, log2 of @c slots_ in-lane stages plus
     *      five shuffle stages restore ascending order.
     */
    __device__ __forceinline__ void insert_batch(inner_hopper_scored_node_t staged, int lane) noexcept {
        inner_hopper_scored_node_t const descending = inner_hopper_sort32(staged, lane).shuffle_xor(31);
        entries_[slots_ - 1] = entries_[slots_ - 1] < descending ? entries_[slots_ - 1] : descending;
#pragma unroll
        for (u32_t stride = slots_ / 2; stride >= 1; stride >>= 1)
#pragma unroll
            for (u32_t base = 0; base < slots_; ++base)
                if ((base & stride) == 0 && (base | stride) < slots_) {
                    inner_hopper_scored_node_t const a = entries_[base], b = entries_[base | stride];
                    entries_[base] = a < b ? a : b;
                    entries_[base | stride] = a < b ? b : a;
                }
#pragma unroll
        for (int j = 16; j; j >>= 1) {
            bool const lower = (lane & j) == 0;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot) {
                inner_hopper_scored_node_t const partner = entries_[slot].shuffle_xor(j);
                entries_[slot] = lower == (entries_[slot] < partner) ? entries_[slot] : partner;
            }
        }
    }

    /** @brief The nearest entry not yet expanded - the frontier's @c top() - or the sentinel. */
    __device__ __forceinline__ inner_hopper_scored_node_t nearest_unexpanded(int lane) const noexcept {
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot) {
            unsigned const mask = __ballot_sync(0xFFFFFFFFu, !entries_[slot].expanded());
            if (mask) return entries_[slot].shuffle_from(__ffs(mask) - 1);
        }
        return inner_hopper_scored_node_t::sentinel();
    }

    /** @brief Lazily deletes @p entry from the frontier - the unrolled-compare spelling
     *      keeps every slot index compile-time. Duplicates cannot arise: staged batches
     *      are deduplicated against the pool before every merge. */
    __device__ __forceinline__ void mark_expanded_at(inner_hopper_scored_node_t entry) noexcept {
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (entries_[slot] == entry) entries_[slot].mark_expanded();
    }

    /** @brief The search radius: ordering bits of the furthest kept entry. */
    __device__ __forceinline__ u32_t radius() const noexcept {
        return entries_[slots_ - 1].shuffle_from(31).distance_bits();
    }
    /** @brief True once every rank holds a real candidate - the radius becomes binding. */
    __device__ __forceinline__ bool full() const noexcept {
        return !(entries_[slots_ - 1].shuffle_from(31) == inner_hopper_scored_node_t::sentinel());
    }

    /**
     *  @brief Kills every staged entry whose id already resides in the pool. The visited
     *      tags yield false negatives on eviction, and a re-admitted duplicate would
     *      silently displace genuine tail candidates - ruining diversity selection while
     *      leaving top-10 metrics intact - so this exact probe runs before every merge.
     */
    __device__ __forceinline__ inner_hopper_scored_node_t deduplicate(inner_hopper_scored_node_t staged,
                                                                      int lane) const noexcept {
        inner_node_id_t const staged_id = staged.id();
        bool resident = false;
#pragma unroll
        for (int probe = 0; probe < 32; ++probe) {
            inner_node_id_t const probe_id = __shfl_sync(0xFFFFFFFFu, staged_id, probe);
            bool holds = false;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot)
                holds |= !(entries_[slot] == inner_hopper_scored_node_t::sentinel()) && entries_[slot].id() == probe_id;
            if (__any_sync(0xFFFFFFFFu, holds) && lane == probe) resident = true;
        }
        return resident ? inner_hopper_scored_node_t::sentinel() : staged;
    }

    /** @brief Entry at @p rank, valid on every lane. */
    __device__ __forceinline__ inner_hopper_scored_node_t rank_entry(u32_t rank) const noexcept {
        inner_hopper_scored_node_t result = inner_hopper_scored_node_t::sentinel();
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (slot == rank / 32) result = entries_[slot];
        return result.shuffle_from(rank % 32);
    }
};

/**
 *  @brief Per-warp direct-mapped visited filter over caller-provided shared memory.
 *
 *  Yields false @b negatives only: an evicted tag costs a re-gather, never a skipped
 *  candidate, so the error polarity is bandwidth-loss, not recall-loss. The compensating
 *  exact check is @c inner_hopper_sorted_pool::deduplicate - together they play the role
 *  of the host-side visited hash-set.
 */
template <u32_t tags_count_>
struct inner_hopper_visited_tags {
    u32_t (&tags_)[tags_count_];

    static_assert((tags_count_ & (tags_count_ - 1)) == 0, "Tags are indexed by a hash prefix");
    inline static constexpr u32_t tag_shift_k = 32 - std::bit_width(tags_count_ - 1);

    __device__ __forceinline__ void reset(int lane) noexcept {
        for (u32_t tag = lane; tag < tags_count_; tag += 32) tags_[tag] = inner_node_id_unknown_k;
    }

    /**
     *  @brief Marks @p node_id visited; @c true when it was fresh. Lanes carrying the same
     *      id in one call are collapsed to their lowest lane via hardware value matching,
     *      so a batch can never stage intra-batch duplicates.
     */
    __device__ __forceinline__ bool test_and_set(inner_node_id_t node_id, int lane, bool active) noexcept {
        unsigned const peers = __match_any_sync(0xFFFFFFFFu, active ? node_id : 0xFFFFFFFFu ^ lane);
        bool fresh = active && (peers & ((1u << lane) - 1)) == 0;
        u32_t const tag_slot = (node_id * 0x9E3779B9u) >> tag_shift_k;
        if (fresh) fresh = tags_[tag_slot] != node_id;
        __syncwarp();
        if (fresh) tags_[tag_slot] = node_id;
        __syncwarp();
        return fresh;
    }
};

/**
 *  @brief One warp's shared-memory scratch with two disjoint lifetimes, made type-visible:
 *      the visited tags during traversal, the candidate-vector cache during diversity
 *      selection & back-linking. The union is the hand-off - @c search_around receives the
 *      tags arm, @c select_diverse and @c back_link_neighbors the cache arm.
 */
template <u32_t tags_count_, typename fragment_type_, u32_t cache_rows_, u32_t row_fragments_>
union inner_hopper_scratch {
    u32_t tags[tags_count_];
    fragment_type_ cache[cache_rows_][row_fragments_];
    static_assert(sizeof(cache) <= sizeof(tags),
                  "The candidate cache lives inside the tag arm - wider rows need a larger tag count");
};

/**
 *  @section inner_hopper_kits Math Kits
 *
 *  A kit binds one storage dtype to one metric with warp-shaped geometry, mirroring the
 *  CPU metric kits: fragments are per-lane register tiles, @c lanes_per_row_k lanes span
 *  one vector, and metrics with precomputable norms finalize "through the dot". The
 *  cross-lane reduction is deliberately not a kit member - one shuffle tree of width
 *  @c lanes_per_row_k serves every kit. Symmetric candidate-to-candidate distances reuse
 *  @c distance(dot, make_query_state(norms_a), norms_b).
 */
template <typename kit_>
concept inner_hopper_kit = requires(typename kit_::fragment_t fragment, typename kit_::accumulator_t accumulator,
                                    typename kit_::norms_accumulator_t norms_accumulator, typename kit_::norms_t norms,
                                    typename kit_::query_state_t query_state, typename kit_::dot_t dot) {
    typename kit_::scalar_t;
    typename kit_::fragment_t;
    typename kit_::accumulator_t;
    typename kit_::dot_t;
    typename kit_::norms_t;
    typename kit_::norms_accumulator_t;
    typename kit_::query_state_t;
    { kit_::fragment_bytes_k } -> std::convertible_to<u32_t>;
    { kit_::row_fragments_k } -> std::convertible_to<u32_t>;
    { kit_::lanes_per_row_k } -> std::convertible_to<u32_t>;
    { kit_::rows_per_step_k } -> std::convertible_to<u32_t>;
    { kit_::max_waves_in_flight_k } -> std::convertible_to<u32_t>;
    { kit_::has_norms_k } -> std::convertible_to<bool>;
    { kit_::zero_accumulator() } -> std::same_as<typename kit_::accumulator_t>;
    { kit_::accumulate_dot(fragment, fragment, accumulator) } -> std::same_as<typename kit_::accumulator_t>;
    { kit_::partial_to_dot(accumulator) } -> std::same_as<typename kit_::dot_t>;
    { kit_::zero_norms_accumulator() } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::accumulate_norms(fragment, norms_accumulator) } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::partials_to_norms(norms_accumulator) } -> std::same_as<typename kit_::norms_t>;
    { kit_::make_query_state(norms) } -> std::same_as<typename kit_::query_state_t>;
    { kit_::distance(dot, query_state, norms) } -> std::same_as<inner_distance_t>;
};

/** @brief Shared @c i8 geometry & dot arithmetic - every @c i8 metric kit inherits it. */
template <u32_t dimensions_>
struct inner_hopper_i8_dot_kit {
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using accumulator_t = int;
    using dot_t = int;
    /** @brief Layout-identical to the CPU planes' @b {sum,sumsq} norms record. */
    struct norms_t {
        int sum, sumsq;
    };
    using norms_accumulator_t = int2;
    struct query_state_t {
        f32_t rsqrt;
        int sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t row_fragments_k = dimensions_ / 16;
    inline static constexpr u32_t lanes_per_row_k = row_fragments_k;
    inline static constexpr u32_t rows_per_step_k = 32 / lanes_per_row_k;
    inline static constexpr u32_t max_waves_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;
    static_assert(dimensions_ % 16 == 0 && (lanes_per_row_k & (lanes_per_row_k - 1)) == 0);
    static_assert(lanes_per_row_k <= 32, "One vector row never spans more than a warp");

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        accumulator = __dp4a(static_cast<int>(a.x), static_cast<int>(b.x), accumulator);
        accumulator = __dp4a(static_cast<int>(a.y), static_cast<int>(b.y), accumulator);
        accumulator = __dp4a(static_cast<int>(a.z), static_cast<int>(b.z), accumulator);
        return __dp4a(static_cast<int>(a.w), static_cast<int>(b.w), accumulator);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0, 0}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        accumulator.x = __dp4a(static_cast<int>(a.x), 0x01010101, accumulator.x);
        accumulator.x = __dp4a(static_cast<int>(a.y), 0x01010101, accumulator.x);
        accumulator.x = __dp4a(static_cast<int>(a.z), 0x01010101, accumulator.x);
        accumulator.x = __dp4a(static_cast<int>(a.w), 0x01010101, accumulator.x);
        accumulator.y = __dp4a(static_cast<int>(a.x), static_cast<int>(a.x), accumulator.y);
        accumulator.y = __dp4a(static_cast<int>(a.y), static_cast<int>(a.y), accumulator.y);
        accumulator.y = __dp4a(static_cast<int>(a.z), static_cast<int>(a.z), accumulator.y);
        accumulator.y = __dp4a(static_cast<int>(a.w), static_cast<int>(a.w), accumulator.y);
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f, norms.sumsq};
    }
};

template <u32_t dimensions_>
struct inner_hopper_i8_angular_kit : inner_hopper_i8_dot_kit<dimensions_> {
    using base_t = inner_hopper_i8_dot_kit<dimensions_>;
    static __device__ __forceinline__ inner_distance_t distance(int dot, typename base_t::query_state_t query_state,
                                                                typename base_t::norms_t norms) noexcept {
        return 1.0f - static_cast<f32_t>(dot) * query_state.rsqrt * rsqrtf(static_cast<f32_t>(norms.sumsq));
    }
};

template <u32_t dimensions_>
struct inner_hopper_i8_sqeuclidean_kit : inner_hopper_i8_dot_kit<dimensions_> {
    using base_t = inner_hopper_i8_dot_kit<dimensions_>;
    static __device__ __forceinline__ inner_distance_t distance(int dot, typename base_t::query_state_t query_state,
                                                                typename base_t::norms_t norms) noexcept {
        return static_cast<f32_t>(query_state.sumsq) + static_cast<f32_t>(norms.sumsq) - 2.0f * static_cast<f32_t>(dot);
    }
};

/** @brief Maps a (dtype, metric) pair onto its device kit; unsupported pairs fail the
 *      index instantiation with a one-line constraint error. */
template <nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, u32_t dimensions_>
struct inner_hopper_kit_for;

template <u32_t dimensions_>
struct inner_hopper_kit_for<nk_i8_k, nk_kernel_angular_k, dimensions_> {
    using kit_t = inner_hopper_i8_angular_kit<dimensions_>;
};
template <u32_t dimensions_>
struct inner_hopper_kit_for<nk_i8_k, nk_kernel_sqeuclidean_k, dimensions_> {
    using kit_t = inner_hopper_i8_sqeuclidean_kit<dimensions_>;
};

/** @brief Generic cross-lane reduction: folds sub-warp partials of width @p lanes_per_row_
 *      down to the leader lane. One tree serves every kit. */
template <u32_t lanes_per_row_, typename dot_type_>
__device__ __forceinline__ dot_type_ inner_hopper_reduce(dot_type_ partial) noexcept {
#pragma unroll
    for (u32_t offset = lanes_per_row_ / 2; offset; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFFu, partial, offset, lanes_per_row_);
    return partial;
}

#endif // defined(__CUDACC__)

/** @brief Device-resident counter block - all-RMW, no compare-and-swap anywhere.
 *      @ref inner_hopper_index_concurrency */
struct inner_hopper_counters_t {
    /** @brief Ticket dispenser and work-queue head fused into one word: a warp's fetch_add
     *      result @b is its node id. Only the host may reset it, between kernel launches,
     *      to the published count - over-claimed tickets die with their launch. */
    u32_t next_ticket;
    /** @brief Published-node count, advanced by a fire-and-forget reduction at each slot-0
     *      release. Advisory: only the admission ramp and @c size() reporting consume it -
     *      unlike the CPU watermark, it implies nothing about @b which ids are published. */
    u32_t published;
    /** @brief Traversal & mutation telemetry, aggregated once per inserted node. */
    u64_t hops, scored, lock_attempts, lock_skips, full_lists, restarts, locked_drops;
};

#if defined(__CUDACC__)

/**
 *  @brief Typed views over the three device planes, @b [rows,*] each: packed vectors,
 *      per-row norms, and fixed-capacity neighbor lists. Under the flat virtual-memory
 *      arena the node id @b is the location - one multiply, no chunk directory.
 */
template <inner_hopper_kit kit_, u32_t connectivity_>
struct inner_hopper_planes_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** @brief Vector plane, @b [rows,row_fragments] fragments, one aligned row per node. */
    fragment_t* vectors;
    /** @brief Norms plane, @b [rows] records, written once pre-publication. */
    norms_t* norms;
    /** @brief Neighbor plane, @b [rows,connectivity] ids; slot 0 doubles as the lock word. */
    u32_t* neighbors;

    /** @brief Lane's fragment of the vector stored at @p node_id. */
    __device__ __forceinline__ fragment_t vector_for(inner_node_id_t node_id, u32_t fragment_index) const noexcept {
        return vectors[static_cast<std::size_t>(node_id) * kit_::row_fragments_k + fragment_index];
    }
    __device__ __forceinline__ fragment_t* vector_span_for(inner_node_id_t node_id) const noexcept {
        return vectors + static_cast<std::size_t>(node_id) * kit_::row_fragments_k;
    }
    __device__ __forceinline__ norms_t norm_for(inner_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ norms_t& norm_ref_for(inner_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ u32_t* neighbors_for(inner_node_id_t node_id) const noexcept {
        return neighbors + static_cast<std::size_t>(node_id) * connectivity_;
    }
};

/**
 *  @brief Typed view over one node's neighbor row - the only legal way to touch a list
 *      that concurrent warps may be reading, mirroring the CPU's @c neighbors_list_ref_t
 *      method-for-method over @c cuda::atomic_ref at device scope.
 *
 *  Slot 0 is the combined lock & first-neighbor word: the all-ones sentinel while
 *  unpublished or locked, the node's own id when published empty, otherwise the first
 *  neighbor. One deliberate divergence from the CPU: the lock is @c try_lock with a
 *  bounded attempt budget - unbounded spinning against a descheduled owner warp is a
 *  livelock hazard unique to the GPU, and skipped reverse edges are the accepted,
 *  telemetered cost.
 */
struct inner_hopper_neighbors_ref_t {
    u32_t* row_; // Non-const even for readers: `cuda::atomic_ref` cannot bind const words pre-P3323.
    u32_t capacity_;

    using word_ref_t = cuda::atomic_ref<u32_t, cuda::thread_scope_device>;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }

    /** @brief Relaxed atomic read of slot @p index; slot 0 belongs to @c peek_first / @c try_lock. */
    __device__ __forceinline__ inner_node_id_t load_slot(u32_t index) const noexcept {
        return word_ref_t(row_[index]).load(cuda::memory_order_relaxed);
    }
    /** @brief Relaxed atomic write of slot @p index - only for the unpublished owner or the
     *      current lock holder. */
    __device__ __forceinline__ void store_slot(u32_t index, inner_node_id_t value) const noexcept {
        word_ref_t(row_[index]).store(value, cuda::memory_order_relaxed);
    }
    /** @brief Acquire-load of the lock word: the sentinel means unpublished or locked; any
     *      other value is the first neighbor id, with everything its last (un)locker wrote
     *      now visible to us. */
    __device__ __forceinline__ inner_node_id_t peek_first() const noexcept {
        return word_ref_t(row_[0]).load(cuda::memory_order_acquire);
    }
    /** @brief Bounded lock acquisition: exchanges the sentinel in with acquire order, up to
     *      @p max_attempts times with jittered @c __nanosleep backoff. An exchange that
     *      receives the sentinel back did @b not acquire the lock - that rule lives here,
     *      in one place. @return The displaced first-neighbor value, or the sentinel. */
    __device__ __forceinline__ inner_node_id_t try_lock(u32_t max_attempts,
                                                        splitmix_u32_t& random_generator) const noexcept {
        for (u32_t attempt = 0; attempt != max_attempts; ++attempt) {
            inner_node_id_t const first_value = word_ref_t(row_[0]).exchange( //
                inner_node_id_unknown_k, cuda::memory_order_acquire);
            if (first_value != inner_node_id_unknown_k) return first_value;
            __nanosleep(static_cast<unsigned>(lock_backoff_floor_ns_k + //
                                              (random_generator.unbounded() & lock_backoff_jitter_mask_k)));
        }
        return inner_node_id_unknown_k;
    }
    /** @brief Release-stores @p first_value into slot 0 - unlocking and simultaneously
     *      (re)publishing the edited list to every future acquiring reader. */
    __device__ __forceinline__ void unlock(inner_node_id_t first_value) const noexcept {
        word_ref_t(row_[0]).store(first_value, cuda::memory_order_release);
    }
    /** @brief Warp-collective row read for @b unlocked rows: the leader lane acquires
     *      slot 0, the others load their slots relaxed - one ordered load per row, not
     *      thirty-two. The sentinel in the returned lane-0 value means the row is
     *      transiently locked or unpublished. A lock holder must @b never call this -
     *      slot 0 holds the sentinel while locked; pass the displaced first value from
     *      @c try_lock through @c locked_row instead. */
    __device__ __forceinline__ inner_node_id_t load_row(int lane) const noexcept {
        return lane ? load_slot(lane) : peek_first();
    }
    /** @brief Warp-collective row read for the @b lock holder: lane 0 reports the
     *      displaced @p first_value, the others load their slots relaxed. */
    __device__ __forceinline__ inner_node_id_t locked_row(int lane, inner_node_id_t first_value) const noexcept {
        return lane ? load_slot(lane) : first_value;
    }
};

/** @brief Per-warp register telemetry, folded into the global counters once per node. */
struct inner_hopper_telemetry_t {
    u64_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, full_lists = 0, restarts = 0, locked_drops = 0;

    __device__ __forceinline__ void flush(inner_hopper_counters_t* counters, int lane) noexcept {
        // @c atomicAdd spells 64-bit as `unsigned long long`; @c u64_t is layout-identical.
        static_assert(sizeof(u64_t) == sizeof(unsigned long long));
        if (lane != 0) return;
        using wide_t = unsigned long long;
        atomicAdd(reinterpret_cast<wide_t*>(&counters->hops), static_cast<wide_t>(hops));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->scored), static_cast<wide_t>(scored));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->lock_attempts), static_cast<wide_t>(lock_attempts));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->lock_skips), static_cast<wide_t>(lock_skips));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->full_lists), static_cast<wide_t>(full_lists));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->restarts), static_cast<wide_t>(restarts));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->locked_drops), static_cast<wide_t>(locked_drops));
    }
};

/**
 *  @brief Each warp performing scans over the graph owns one context like this - the
 *      device twin of the CPU's per-thread traversal context: the candidate pool, the
 *      lane's PRNG stream, and the running telemetry.
 */
template <u32_t pool_slots_>
struct inner_hopper_traversal_context {
    inner_hopper_sorted_pool<pool_slots_> pool;
    splitmix_u32_t random_generator;
    inner_hopper_telemetry_t telemetry;
};

/**
 *  @brief Scores one staged batch of up to 32 candidate rows as a single flattened load
 *      volley - word w*32+lane of the @b [batch,row_fragments] fragment grid, so every
 *      lane issues up to @c row_fragments_k independent loads - then reduces sub-warp
 *      partials and packs scored entries onto their staging lanes.
 *
 *  Mutable-plane discipline: vector & norm loads here are plain loads, made safe by the
 *  acquire chain - every staged id was discovered through an acquiring slot-0 read that
 *  synchronizes with its publisher's release. @return This lane's scored entry, or the
 *  sentinel on lanes past @p batch_count.
 */
template <inner_hopper_kit kit_, u32_t connectivity_>
__device__ __forceinline__ inner_hopper_scored_node_t inner_hopper_score_batch( //
    inner_hopper_planes_view<kit_, connectivity_> const& planes, typename kit_::fragment_t query_fragment,
    typename kit_::query_state_t query_state, u32_t const (&staged_ids)[inner_hopper_cache_rows_k],
    inner_hopper_scored_node_t (&staged_scored)[32], u32_t batch_count, int lane) noexcept {

    using norms_t = typename kit_::norms_t;
    constexpr u32_t row_fragments_k = kit_::row_fragments_k;
    constexpr u32_t lanes_per_row_k = kit_::lanes_per_row_k;
    constexpr u32_t rows_per_step_k = kit_::rows_per_step_k;
    constexpr u32_t waves_per_pass_k = row_fragments_k < kit_::max_waves_in_flight_k ? row_fragments_k
                                                                                     : kit_::max_waves_in_flight_k;
    bool const leader = (lane % lanes_per_row_k) == 0;

    // Waves are independent row groups; the pass loop tiles them so at most
    // `max_waves_in_flight_k` loads - the volley register budget - coexist per lane.
#pragma unroll 1
    for (u32_t base = 0; base < row_fragments_k; base += waves_per_pass_k) {
        norms_t batch_norms[waves_per_pass_k];
#pragma unroll
        for (u32_t wave = 0; wave < waves_per_pass_k; ++wave) {
            u32_t const row = (base + wave) * rows_per_step_k + lane / lanes_per_row_k;
            if (kit_::has_norms_k && leader && row < batch_count) batch_norms[wave] = planes.norm_for(staged_ids[row]);
        }
        typename kit_::fragment_t loaded[waves_per_pass_k];
#pragma unroll
        for (u32_t wave = 0; wave < waves_per_pass_k; ++wave) {
            u32_t const row = ((base + wave) * 32 + lane) / lanes_per_row_k;
            if (row < batch_count) loaded[wave] = planes.vector_for(staged_ids[row], lane % lanes_per_row_k);
        }
#pragma unroll
        for (u32_t wave = 0; wave < waves_per_pass_k; ++wave) {
            u32_t const row = ((base + wave) * 32 + lane) / lanes_per_row_k;
            typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
            if (row < batch_count) accumulator = kit_::accumulate_dot(loaded[wave], query_fragment, accumulator);
            typename kit_::dot_t const dot = inner_hopper_reduce<lanes_per_row_k>(kit_::partial_to_dot(accumulator));
            u32_t const leader_row = (base + wave) * rows_per_step_k + lane / lanes_per_row_k;
            if (leader && leader_row < batch_count) {
                inner_distance_t const distance = kit_::distance(dot, query_state, batch_norms[wave]);
                staged_scored[leader_row] = inner_hopper_scored_node_t::make(distance, staged_ids[leader_row]);
            }
        }
    }
    __syncwarp();
    return lane < static_cast<int>(batch_count) ? staged_scored[lane] : inner_hopper_scored_node_t::sentinel();
}

/**
 *  @brief Samples one 32-wide batch of distinct, published landmarks below @p sampling_bound
 *      into @p staged_ids - uniform draws gated on the slot-0 publication word, probing
 *      forward on unpublished hits so out-of-order publication only costs retries.
 *  @return The number of staged landmarks; zero when the visited set saturated the bound.
 */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t tags_count_>
__device__ __forceinline__ u32_t inner_hopper_sample_landmarks( //
    inner_hopper_planes_view<kit_, connectivity_> const& planes, inner_node_id_t sampling_bound,
    inner_hopper_visited_tags<tags_count_>& visited, splitmix_u32_t& random_generator,
    u32_t (&staged_ids)[inner_hopper_cache_rows_k], int lane) noexcept {

    inner_node_id_t draw = random_generator.bounded(sampling_bound);
    bool published = false;
    for (int attempt = 0; attempt != 4 && !published; ++attempt) {
        inner_hopper_neighbors_ref_t const row {planes.neighbors_for(draw), connectivity_};
        published = row.peek_first() != inner_node_id_unknown_k;
        if (!published) draw = draw + 1 == sampling_bound ? 0 : draw + 1;
    }
    bool const fresh = visited.test_and_set(draw, lane, published);
    unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
    u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
    if (fresh) staged_ids[ordinal] = draw;
    __syncwarp();
    return __popc(fresh_mask);
}

/**
 *  @brief Greedy best-first NSW walk from random published landmarks toward the query -
 *      one warp, one traversal. Read-only on the graph: takes no locks, drops transiently
 *      locked candidates after a bounded retry, and leaves up to @c pool.capacity_k nearest
 *      published candidates ascending in @c context.pool.
 */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t tags_count_, u32_t pool_slots_>
__device__ __forceinline__ void inner_hopper_search_around( //
    inner_hopper_planes_view<kit_, connectivity_> const& planes, typename kit_::fragment_t query_fragment,
    typename kit_::query_state_t query_state, inner_node_id_t sampling_bound,
    inner_hopper_visited_tags<tags_count_>& visited, u32_t (&staged_ids)[inner_hopper_cache_rows_k],
    inner_hopper_scored_node_t (&staged_scored)[32], inner_hopper_traversal_context<pool_slots_>& context,
    int lane) noexcept {

    constexpr u32_t hops_cap_k = inner_hopper_sorted_pool<pool_slots_>::capacity_k * 4;
    constexpr u32_t restarts_cap_k = 8;
    inner_hopper_sorted_pool<pool_slots_>& pool = context.pool;
    inner_hopper_telemetry_t& telemetry = context.telemetry;
    visited.reset(lane);
    pool.clear();

    u32_t batch_count = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            batch_count = inner_hopper_sample_landmarks<kit_, connectivity_>( //
                planes, sampling_bound, visited, context.random_generator, staged_ids, lane);
        }
        else {
            inner_hopper_scored_node_t const candidate = pool.nearest_unexpanded(lane);
            if (candidate == inner_hopper_scored_node_t::sentinel()) {
                // Starvation restart: a drained frontier before a full pool means the walk
                // exhausted everything reachable from its seeds - re-seed and continue.
                if (pool.full() || telemetry.restarts >= restarts_cap_k || telemetry.hops >= hops_cap_k) break;
                seeding = true;
                ++telemetry.restarts;
                continue;
            }
            // Once the pool is full and the nearest unexpanded entry lies beyond its
            // radius, nothing deeper can improve the candidate set.
            if (pool.full() && candidate.distance_bits() > pool.radius()) break;
            pool.mark_expanded_at(candidate);
            if (telemetry.hops >= hops_cap_k) break;
            ++telemetry.hops;

            // Read the candidate's neighbor row; a transiently locked row is retried
            // briefly, then dropped rather than convoying behind the re-linker. Only the
            // leader lane polls the racing lock word: per-lane polls can observe different
            // values of a concurrent unlock, and a value-dependent `continue` would fork
            // the warp across loop iterations - fatal for every collective downstream.
            inner_hopper_neighbors_ref_t const row {planes.neighbors_for(candidate.id()), connectivity_};
            inner_node_id_t first_value = inner_node_id_unknown_k;
            if (lane == 0) {
                first_value = row.peek_first();
                for (u32_t attempt = 0; attempt != traversal_peek_retries_k && first_value == inner_node_id_unknown_k;
                     ++attempt) {
                    __nanosleep(512);
                    first_value = row.peek_first();
                }
            }
            first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
            if (first_value == inner_node_id_unknown_k) {
                ++telemetry.locked_drops;
                continue;
            }
            inner_node_id_t const neighbor_id = lane ? row.load_slot(lane) : first_value;
            bool const usable = neighbor_id != inner_node_id_unknown_k && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(neighbor_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) staged_ids[ordinal] = neighbor_id;
            __syncwarp();
            batch_count = __popc(fresh_mask);
        }
        if (!batch_count) {
            seeding = false;
            continue;
        }
        telemetry.scored += batch_count;
        inner_hopper_scored_node_t staged = inner_hopper_score_batch<kit_, connectivity_>( //
            planes, query_fragment, query_state, staged_ids, staged_scored, batch_count, lane);
        staged = pool.deduplicate(staged, lane);
        pool.insert_batch(staged, lane);
        seeding = false;
    }
}

/**
 *  @brief Greedy diversity selection over the ascending pool prefix - the classic NSW
 *      pruning heuristic: a candidate is kept only when it is closer to the query than to
 *      every already-kept link, so links spread across directions. Candidate vectors are
 *      cached in the warp's shared scratch; lane @b j owns kept link @b j.
 *  @return The kept-link count; @p kept_id and @p kept_cache_row are per-lane outputs.
 */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t cache_rows_, u32_t pool_slots_>
__device__ __forceinline__ u32_t inner_hopper_select_diverse( //
    inner_hopper_planes_view<kit_, connectivity_> const& planes,
    typename kit_::fragment_t (&cache)[cache_rows_][kit_::row_fragments_k],
    typename kit_::norms_t (&cache_norms)[cache_rows_], u32_t (&staged_ids)[inner_hopper_cache_rows_k],
    inner_hopper_traversal_context<pool_slots_>& context, inner_node_id_t& kept_id, int lane) noexcept {

    constexpr u32_t row_fragments_k = kit_::row_fragments_k;
    constexpr u32_t selectable_k = min_of(cache_rows_, inner_hopper_cache_rows_k);
    inner_hopper_sorted_pool<pool_slots_>& pool = context.pool;

    // Stage the ascending prefix ids & cache their vectors and norms.
    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < selectable_k; base += 32) {
        inner_hopper_scored_node_t const entry = base + lane < 64 ? pool.rank_entry(base + lane)
                                                                  : inner_hopper_scored_node_t::sentinel();
        unsigned const live = __ballot_sync(0xFFFFFFFFu, !(entry == inner_hopper_scored_node_t::sentinel()));
        if (!(entry == inner_hopper_scored_node_t::sentinel())) staged_ids[base + lane] = entry.id();
        candidates_count += __popc(live);
        if (live != 0xFFFFFFFFu) break;
    }
    __syncwarp();
    for (u32_t word = lane; word < candidates_count * row_fragments_k; word += 32)
        cache[word / row_fragments_k][word % row_fragments_k] = planes.vector_for(staged_ids[word / row_fragments_k],
                                                                                  word % row_fragments_k);
    if (kit_::has_norms_k)
        for (u32_t row = lane; row < candidates_count; row += 32) cache_norms[row] = planes.norm_for(staged_ids[row]);
    __syncwarp();

    kept_id = inner_node_id_unknown_k;
    u32_t kept_cache_row = 0;
    u32_t kept_count = 0;
    for (u32_t candidate_index = 0; candidate_index < candidates_count && kept_count < connectivity_;
         ++candidate_index) {
        f32_t const to_query = pool.rank_entry(candidate_index).distance();
        bool diverse = true;
        if (static_cast<u32_t>(lane) < kept_count) {
            typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
#pragma unroll
            for (u32_t fragment = 0; fragment != row_fragments_k; ++fragment)
                accumulator = kit_::accumulate_dot(cache[candidate_index][fragment], cache[kept_cache_row][fragment],
                                                   accumulator);
            f32_t const between = kit_::distance(kit_::partial_to_dot(accumulator),
                                                 kit_::make_query_state(cache_norms[candidate_index]),
                                                 cache_norms[kept_cache_row]);
            diverse = between >= to_query;
            ++context.telemetry.scored;
        }
        if (__all_sync(0xFFFFFFFFu, diverse)) {
            if (static_cast<u32_t>(lane) == kept_count) {
                kept_id = staged_ids[candidate_index];
                kept_cache_row = candidate_index;
            }
            ++kept_count;
        }
    }
    return kept_count;
}

/**
 *  @brief Wires the reverse edges: locks each selected neighbor's row via the bounded
 *      slot-0 exchange, appends when there is room, and re-runs the diversity heuristic
 *      over the full list plus the newcomer when there is none - rewriting the row under
 *      the lock. Cheaper replacement policies measurably degrade the graph.
 *
 *  Locking discipline: at most one neighbor lock is held at any moment, only after this
 *  node's own publication, and never while waiting on anything else - lock holders are
 *  always runnable. A lock that cannot be acquired within the attempt budget skips the
 *  edge; approximate search tolerates the missing reverse link by construction.
 */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t cache_rows_>
__device__ __forceinline__ void inner_hopper_back_link_neighbors( //
    inner_hopper_planes_view<kit_, connectivity_> const& planes, inner_node_id_t new_node_id,
    typename kit_::norms_t new_norms, inner_node_id_t kept_id, u32_t kept_count,
    typename kit_::fragment_t (&cache)[cache_rows_][kit_::row_fragments_k],
    typename kit_::norms_t (&cache_norms)[cache_rows_], u32_t (&staged_ids)[inner_hopper_cache_rows_k],
    inner_hopper_scored_node_t (&staged_scored)[32], splitmix_u32_t& random_generator,
    inner_hopper_telemetry_t& telemetry, int lane) noexcept {

    constexpr u32_t row_fragments_k = kit_::row_fragments_k;
    static_assert(cache_rows_ >= connectivity_ + 1, "Re-selection caches every link plus the newcomer");

    for (u32_t link = 0; link < kept_count; ++link) {
        inner_node_id_t const target = __shfl_sync(0xFFFFFFFFu, kept_id, link);
        inner_hopper_neighbors_ref_t const row {planes.neighbors_for(target), connectivity_};
        inner_node_id_t first_value = inner_node_id_unknown_k;
        if (lane == 0) first_value = row.try_lock(back_link_lock_attempts_k, random_generator);
        first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
        ++telemetry.lock_attempts;
        if (first_value == inner_node_id_unknown_k) { // Bounded-wait policy: skip the edge.
            ++telemetry.lock_skips;
            continue;
        }
        if (first_value == target) { // Published-empty self-loop: the newcomer is the first neighbor.
            if (lane == 0) row.unlock(new_node_id);
            continue;
        }

        inner_node_id_t const slot_value = row.locked_row(lane, first_value);
        unsigned const empty_mask = __ballot_sync(0xFFFFFFFFu, slot_value == inner_node_id_unknown_k);
        if (empty_mask) { // Fast path: append into the first empty slot, unlock unchanged.
            if (lane == 0) row.store_slot(__ffs(empty_mask) - 1, new_node_id);
            __syncwarp();
            if (lane == 0) row.unlock(first_value);
            continue;
        }

        // Full list: score every link & the newcomer against the target itself, then walk
        // all of them ascending, re-running the diversity heuristic, and rewrite the row.
        ++telemetry.full_lists;
        staged_ids[lane] = slot_value;
        __syncwarp();
        typename kit_::fragment_t const target_fragment = planes.vector_for(target, lane % kit_::lanes_per_row_k);
        typename kit_::query_state_t const target_state = kit_::make_query_state(planes.norm_for(target));

        // The newcomer's distance to the target, from its freshly-owned vector row.
        typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
        accumulator = kit_::accumulate_dot(planes.vector_for(new_node_id, lane % kit_::lanes_per_row_k),
                                           target_fragment, accumulator);
        typename kit_::dot_t const self_dot = inner_hopper_reduce<kit_::lanes_per_row_k>(
            kit_::partial_to_dot(accumulator));
        f32_t const newcomer_distance = kit_::distance(__shfl_sync(0xFFFFFFFFu, self_dot, 0), target_state, new_norms);
        telemetry.scored += connectivity_ + 1;

        inner_hopper_scored_node_t const scored = inner_hopper_score_batch<kit_, connectivity_>( //
            planes, target_fragment, target_state, staged_ids, staged_scored, connectivity_, lane);
        inner_hopper_scored_node_t const sorted_links = inner_hopper_sort32(
            lane < static_cast<int>(connectivity_)
                ? inner_hopper_scored_node_t::make(scored.distance(), lane) // Id = slot index.
                : inner_hopper_scored_node_t::sentinel(),
            lane);

        // Cache the links' vectors (their norms ride the staging) plus the newcomer at the
        // tail row, then merge-walk all `connectivity_ + 1` ascending by distance-to-target.
        for (u32_t word = lane; word < connectivity_ * row_fragments_k; word += 32)
            cache[word / row_fragments_k][word % row_fragments_k] = planes.vector_for(
                staged_ids[word / row_fragments_k], word % row_fragments_k);
        for (u32_t word = lane; word < row_fragments_k; word += 32)
            cache[cache_rows_ - 1][word] = planes.vector_for(new_node_id, word);
        for (u32_t cache_row = lane; cache_row < connectivity_; cache_row += 32)
            cache_norms[cache_row] = planes.norm_for(staged_ids[cache_row]);
        if (lane == 0) cache_norms[cache_rows_ - 1] = new_norms;
        __syncwarp();

        u32_t existing_index = 0;
        bool newcomer_pending = true;
        u32_t refined_cache_row = cache_rows_ - 1;
        inner_node_id_t refined_id = inner_node_id_unknown_k;
        u32_t refined_count = 0;
        for (u32_t step = 0; step != connectivity_ + 1 && refined_count != connectivity_; ++step) {
            inner_hopper_scored_node_t const next_link = existing_index < connectivity_
                                                             ? sorted_links.shuffle_from(existing_index)
                                                             : inner_hopper_scored_node_t::sentinel();
            bool const take_newcomer = newcomer_pending && newcomer_distance < next_link.distance();
            f32_t const candidate_distance = take_newcomer ? newcomer_distance : next_link.distance();
            u32_t const candidate_row = take_newcomer ? cache_rows_ - 1 : next_link.id();
            if (!take_newcomer && existing_index >= connectivity_) break;
            if (take_newcomer) newcomer_pending = false;
            else ++existing_index;

            bool diverse = true;
            if (static_cast<u32_t>(lane) < refined_count) {
                typename kit_::accumulator_t pair_accumulator = kit_::zero_accumulator();
#pragma unroll
                for (u32_t fragment = 0; fragment != row_fragments_k; ++fragment)
                    pair_accumulator = kit_::accumulate_dot(cache[candidate_row][fragment],
                                                            cache[refined_cache_row][fragment], pair_accumulator);
                f32_t const between = kit_::distance(kit_::partial_to_dot(pair_accumulator),
                                                     kit_::make_query_state(cache_norms[candidate_row]),
                                                     cache_norms[refined_cache_row]);
                diverse = between >= candidate_distance;
                ++telemetry.scored;
            }
            if (__all_sync(0xFFFFFFFFu, diverse)) {
                if (static_cast<u32_t>(lane) == refined_count) {
                    refined_cache_row = candidate_row;
                    refined_id = candidate_row == cache_rows_ - 1 ? new_node_id : staged_ids[candidate_row];
                }
                ++refined_count;
            }
        }

        // Rewrite the row: tail slots relaxed, shrink sentinel-terminated, head unlocks.
        if (lane > 0)
            row.store_slot(lane, static_cast<u32_t>(lane) < refined_count ? refined_id : inner_node_id_unknown_k);
        __syncwarp();
        inner_node_id_t const refined_head = __shfl_sync(0xFFFFFFFFu, refined_id, 0);
        if (lane == 0) row.unlock(refined_head);
    }
}

/** @brief Sub-warp reduction for the norms accumulators of the current kits. Only the
 *      sub-warp leader holds the complete sums afterwards - always follow with
 *      @c inner_hopper_broadcast_norms before any lane-divergent consumer. */
template <u32_t lanes_per_row_>
__device__ __forceinline__ int2 inner_hopper_reduce_norms(int2 partial) noexcept {
#pragma unroll
    for (u32_t offset = lanes_per_row_ / 2; offset; offset >>= 1) {
        partial.x += __shfl_down_sync(0xFFFFFFFFu, partial.x, offset, lanes_per_row_);
        partial.y += __shfl_down_sync(0xFFFFFFFFu, partial.y, offset, lanes_per_row_);
    }
    return partial;
}
/** @brief Rebroadcasts the sub-warp leader's reduced norms to every lane - values that
 *      feed lane-divergent control flow must be warp-uniform. */
template <u32_t lanes_per_row_>
__device__ __forceinline__ int2 inner_hopper_broadcast_norms(int2 reduced) noexcept {
    reduced.x = __shfl_sync(0xFFFFFFFFu, reduced.x, 0, lanes_per_row_);
    reduced.y = __shfl_sync(0xFFFFFFFFu, reduced.y, 0, lanes_per_row_);
    return reduced;
}

/** @brief Geometry & plane pointers one insertion launch operates on. Every plane row a
 *      dispensable ticket can touch is mapped and sentinel-filled ahead of the launch, so
 *      growth hazards cannot reach the kernel. */
template <inner_hopper_kit kit_, u32_t connectivity_>
struct inner_hopper_build_view {
    inner_hopper_planes_view<kit_, connectivity_> planes;
    /** @brief Incoming vectors, @b [rows,row_fragments] in ticket order, device-resident. */
    typename kit_::fragment_t const* staging;
    /** @brief Tickets strictly below this bound may be claimed & inserted by this launch. */
    u32_t rows_count;
    inner_hopper_counters_t* counters;
};

/** @brief Inputs & outputs of one search launch: a batch of queries in, the nearest ids
 *      and - when requested - their distances out, both ascending per query. */
template <inner_hopper_kit kit_, u32_t connectivity_>
struct inner_hopper_search_view {
    inner_hopper_planes_view<kit_, connectivity_> planes;
    typename kit_::fragment_t const* queries; // @b [queries,row_fragments]
    inner_node_id_t* result_ids;              // @b [queries,count]
    inner_distance_t* result_distances;       // @b [queries,count]; null skips the writes
    u32_t queries_count;
    u32_t count;
    u32_t rows_count;
    inner_hopper_counters_t* counters;
};

/**
 *  @brief Inserts the node identified by @p ticket: lands the payload & norms, walks the
 *      published graph, selects diverse links, publishes with one release store, then
 *      wires the reverse edges - the whole CPU insertion flow, one warp per node.
 */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t tags_count_, u32_t pool_slots_, u32_t cache_rows_>
__device__ __forceinline__ void inner_hopper_insert_one( //
    inner_hopper_build_view<kit_, connectivity_> const& view, inner_node_id_t ticket,
    inner_hopper_scratch<tags_count_, typename kit_::fragment_t, cache_rows_, kit_::row_fragments_k>& scratch,
    typename kit_::norms_t (&cache_norms)[cache_rows_], u32_t (&staged_ids)[inner_hopper_cache_rows_k],
    inner_hopper_scored_node_t (&staged_scored)[32], inner_hopper_traversal_context<pool_slots_>& context,
    int lane) noexcept {

    constexpr u32_t row_fragments_k = kit_::row_fragments_k;
    inner_hopper_planes_view<kit_, connectivity_> const& planes = view.planes;

    // Land the payload: copy the vector fragments, derive both norms from the owned bytes.
    typename kit_::fragment_t const* staging_row = view.staging + static_cast<std::size_t>(ticket) * row_fragments_k;
    typename kit_::fragment_t* owned_row = planes.vector_span_for(ticket);
    for (u32_t fragment = lane; fragment < row_fragments_k; fragment += 32) owned_row[fragment] = staging_row[fragment];
    typename kit_::fragment_t const query_fragment = staging_row[lane % kit_::lanes_per_row_k];
    typename kit_::norms_accumulator_t norms_accumulator = kit_::zero_norms_accumulator();
    norms_accumulator = kit_::accumulate_norms(query_fragment, norms_accumulator);
    norms_accumulator = inner_hopper_reduce_norms<kit_::lanes_per_row_k>(norms_accumulator);
    norms_accumulator = inner_hopper_broadcast_norms<kit_::lanes_per_row_k>(norms_accumulator);
    typename kit_::norms_t const new_norms = kit_::partials_to_norms(norms_accumulator);
    if (lane == 0) planes.norm_ref_for(ticket) = new_norms;
    typename kit_::query_state_t const query_state = kit_::make_query_state(new_norms);

    // Walk the published prefix - a virgin index publishes an empty node instead.
    if (ticket != 0) {
        inner_hopper_visited_tags<tags_count_> visited {scratch.tags};
        inner_hopper_search_around<kit_, connectivity_>( //
            planes, query_fragment, query_state, ticket, visited, staged_ids, staged_scored, context, lane);
    }
    else context.pool.clear();

    // Select diverse links & write the own row - no locks: no reader arrives before the
    // slot-0 release below flips it from the sentinel.
    inner_node_id_t kept_id = inner_node_id_unknown_k;
    u32_t const kept_count = inner_hopper_select_diverse<kit_, connectivity_>( //
        planes, scratch.cache, cache_norms, staged_ids, context, kept_id, lane);
    inner_hopper_neighbors_ref_t const own_row {planes.neighbors_for(ticket), connectivity_};
    if (lane > 0 && static_cast<u32_t>(lane) < kept_count) own_row.store_slot(lane, kept_id);
    __syncwarp();
    inner_node_id_t const first_link = __shfl_sync(0xFFFFFFFFu, kept_id, 0);
    if (lane == 0) {
        own_row.unlock(kept_count ? first_link : ticket); // Self-loop encodes published-empty.
        atomicAdd(&view.counters->published, 1u);         // Fire-and-forget reduction.
    }

    // Only now - published, no locks held - wire the reverse edges.
    inner_hopper_back_link_neighbors<kit_, connectivity_>( //
        planes, ticket, new_norms, kept_id, kept_count, scratch.cache, cache_norms, staged_ids, staged_scored,
        context.random_generator, context.telemetry, lane);
}

/**
 *  @section inner_hopper_index_concurrency Concurrency & Publication
 *
 *  One warp inserts one node, and no two warps ever synchronize beyond per-neighbor
 *  slot-0 locks: there is no watermark, no publication order, and no compare-and-swap.
 *  The ticket dispenser doubles as the work queue - a warp's fetch_add result @b is its
 *  node id, so aborted launches leave no capacity to re-credit and the host restart rule
 *  (reset the dispenser to the published count between launches) is the only bookkeeping.
 *  A node is usable exactly when its slot 0 acquire-loads non-sentinel; a self-loop id
 *  there means "published, zero neighbors". Landmark sampling draws below the claimed
 *  ticket and probes past unpublished hits, so out-of-order publication costs retries,
 *  never correctness. The admission ramp - ticket @b t waits until
 *  t < published + max(32, published) - bounds early-graph lock contention and keeps
 *  the sampler's hit rate above one half; past bootstrap, warps never wait on each other.
 *  Every give-up path still publishes (a self-loop tombstone), so dispensed ids never
 *  stall anything, and a post-mortem sweep can tombstone the stragglers of a killed
 *  launch precisely because publication order is free.
 */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t tags_count_, u32_t pool_slots_>
__global__ void __launch_bounds__(inner_hopper_warps_per_block_k * 32)
    inner_hopper_insert_kernel(inner_hopper_build_view<kit_, connectivity_> view) {
    using scratch_t =
        inner_hopper_scratch<tags_count_, typename kit_::fragment_t, inner_hopper_cache_rows_k, kit_::row_fragments_k>;
    __shared__ scratch_t scratch[inner_hopper_warps_per_block_k];
    __shared__ typename kit_::norms_t cache_norms[inner_hopper_warps_per_block_k][inner_hopper_cache_rows_k];
    __shared__ u32_t staged_ids[inner_hopper_warps_per_block_k][inner_hopper_cache_rows_k];
    __shared__ inner_hopper_scored_node_t staged_scored[inner_hopper_warps_per_block_k][32];

    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    while (true) {
        u32_t ticket;
        if (lane == 0) ticket = atomicAdd(&view.counters->next_ticket, 1u);
        ticket = __shfl_sync(0xFFFFFFFFu, ticket, 0);
        if (ticket >= view.rows_count) return;

        while (true) { // Admission ramp - the only bootstrap throttle.
            u32_t published;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(published) : "l"(&view.counters->published));
            u32_t const window = max_of(published, admission_ramp_floor_k);
            if (ticket < published + window) break;
            __nanosleep(2048);
        }

        inner_hopper_traversal_context<pool_slots_> context;
        context.random_generator = splitmix_u32_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(ticket) * 33 + lane)};
        inner_hopper_insert_one<kit_, connectivity_>( //
            view, ticket, scratch[warp], cache_norms[warp], staged_ids[warp], staged_scored[warp], context, lane);
        context.telemetry.flush(view.counters, lane);
    }
}

/** @brief One warp per query: the same traversal as insertion, reporting the nearest
 *      @c count published ids ascending by distance. Safe against concurrent insertions:
 *      the walk gates every row on its slot-0 word and never writes to the graph. */
template <inner_hopper_kit kit_, u32_t connectivity_, u32_t tags_count_, u32_t pool_slots_>
__global__ void __launch_bounds__(inner_hopper_warps_per_block_k * 32)
    inner_hopper_search_kernel(inner_hopper_search_view<kit_, connectivity_> view) {
    using scratch_t =
        inner_hopper_scratch<tags_count_, typename kit_::fragment_t, inner_hopper_cache_rows_k, kit_::row_fragments_k>;
    __shared__ scratch_t scratch[inner_hopper_warps_per_block_k];
    __shared__ u32_t staged_ids[inner_hopper_warps_per_block_k][inner_hopper_cache_rows_k];
    __shared__ inner_hopper_scored_node_t staged_scored[inner_hopper_warps_per_block_k][32];

    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (u32_t query_index = blockIdx.x * inner_hopper_warps_per_block_k + warp; query_index < view.queries_count;
         query_index += gridDim.x * inner_hopper_warps_per_block_k) {

        typename kit_::fragment_t const* query_row = view.queries +
                                                     static_cast<std::size_t>(query_index) * kit_::row_fragments_k;
        typename kit_::fragment_t const query_fragment = query_row[lane % kit_::lanes_per_row_k];
        typename kit_::norms_accumulator_t norms_accumulator = kit_::zero_norms_accumulator();
        norms_accumulator = kit_::accumulate_norms(query_fragment, norms_accumulator);
        norms_accumulator = inner_hopper_reduce_norms<kit_::lanes_per_row_k>(norms_accumulator);
        norms_accumulator = inner_hopper_broadcast_norms<kit_::lanes_per_row_k>(norms_accumulator);
        typename kit_::norms_t const query_norms = kit_::partials_to_norms(norms_accumulator);
        typename kit_::query_state_t const query_state = kit_::make_query_state(query_norms);

        inner_hopper_traversal_context<pool_slots_> context;
        context.random_generator = splitmix_u32_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(query_index) * 33 + lane)};
        inner_hopper_visited_tags<tags_count_> visited {scratch[warp].tags};
        inner_hopper_search_around<kit_, connectivity_>( //
            view.planes, query_fragment, query_state, view.rows_count, visited, staged_ids[warp], staged_scored[warp],
            context, lane);
        // The rank extraction shuffles - every lane participates, only in-range ones store.
        // Ranks past the pool's live prefix surface as sentinel ids, marking absent matches.
        for (u32_t rank_base = 0; rank_base < view.count; rank_base += 32) {
            u32_t const rank = rank_base + lane;
            inner_hopper_scored_node_t const ranked_entry = context.pool.rank_entry(rank);
            if (rank < view.count) {
                std::size_t const slot = static_cast<std::size_t>(query_index) * view.count + rank;
                view.result_ids[slot] = ranked_entry.id();
                if (view.result_distances) view.result_distances[slot] = ranked_entry.distance();
            }
        }
        context.telemetry.flush(view.counters, lane);
    }
}

/** @brief Post-drain structural audit of the neighbor plane: every row published, every id
 *      in range, self-loop rows childless, no duplicate edges within a row. One warp scans
 *      one row; each bad row bumps @p violations once. */
template <u32_t connectivity_>
__global__ void inner_hopper_invariant_kernel(u32_t const* neighbors, u32_t rows_count, u64_t* violations) {
    static_assert(connectivity_ == 32, "One warp audits one neighbor row");
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        u32_t const value = neighbors[row * connectivity_ + lane];
        u32_t const first = __shfl_sync(0xFFFFFFFFu, value, 0);
        bool bad = false;
        bad |= first == inner_node_id_unknown_k;
        bad |= value != inner_node_id_unknown_k && value >= rows_count;
        bad |= first == static_cast<u32_t>(row) && lane > 0 && value != inner_node_id_unknown_k;
#pragma unroll
        for (int shift = 1; shift < 32; ++shift) { // Uniform trip count: duplicate-edge scan.
            u32_t const other = __shfl_sync(0xFFFFFFFFu, value, (lane + shift) & 31);
            bad |= value != inner_node_id_unknown_k && other == value && ((lane + shift) & 31) > lane;
        }
        if (__ballot_sync(0xFFFFFFFFu, bad) && lane == 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(violations), 1ull);
    }
}

#endif // defined(__CUDACC__)

/**
 *  @brief One growable device-memory plane: a fixed virtual-address reservation backed by
 *      physical slabs mapped on demand, so the base pointer never moves and node addressing
 *      stays one multiply. Mapping is host-driven and always completes ahead of any kernel
 *      that could touch the fresh range - growth hazards cannot reach device code.
 */
struct inner_hopper_arena_t {

    /** @brief Preferred physical slab, rounded to the driver's granularity at reservation -
     *      large enough to amortize mapping calls, small enough for incremental growth. */
    inline static constexpr std::size_t preferred_slab_bytes_k = 128u << 20;

    CUdeviceptr base_ = 0;
    std::size_t va_bytes_ = 0;
    std::size_t mapped_bytes_ = 0;
    std::size_t slab_bytes_ = 0;
    int device_ = 0;

    inner_hopper_arena_t() = default;
    inner_hopper_arena_t(inner_hopper_arena_t const&) = delete;
    inner_hopper_arena_t& operator=(inner_hopper_arena_t const&) = delete;

    /** @brief Reserves @p va_bytes of virtual address space - no physical memory yet. */
    status_t reserve(std::size_t va_bytes, int device) noexcept {
        CUmemAllocationProp properties = allocation_properties(device);
        std::size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
            return status_t::device_error_k;
        slab_bytes_ = max_of(granularity, preferred_slab_bytes_k / granularity * granularity);
        va_bytes_ = fu::round_up_to_multiple(va_bytes, slab_bytes_);
        device_ = device;
        if (cuMemAddressReserve(&base_, va_bytes_, 0, 0, 0) != CUDA_SUCCESS) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /**
     *  @brief Maps physical slabs until at least @p bytes are usable; never unmaps early.
     *
     *  Each allocation handle is released right after a successful mapping: the driver
     *  reference-counts them, so the mapping alone keeps the physical memory alive and no
     *  handle registry is needed - the destructor's unmap is the final release.
     */
    status_t grow_to(std::size_t bytes) noexcept {
        if (bytes > va_bytes_) return status_t::overflow_k;
        CUmemAllocationProp properties = allocation_properties(device_);
        CUmemAccessDesc access = {};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (mapped_bytes_ < bytes) {
            CUmemGenericAllocationHandle handle;
            if (cuMemCreate(&handle, slab_bytes_, &properties, 0) != CUDA_SUCCESS) return status_t::bad_alloc_k;
            bool const mapped = cuMemMap(base_ + mapped_bytes_, slab_bytes_, 0, handle, 0) == CUDA_SUCCESS &&
                                cuMemSetAccess(base_ + mapped_bytes_, slab_bytes_, &access, 1) == CUDA_SUCCESS;
            cuMemRelease(handle);
            if (!mapped) return status_t::device_error_k;
            mapped_bytes_ += slab_bytes_;
        }
        return status_t::success_k;
    }

    ~inner_hopper_arena_t() noexcept {
        if (mapped_bytes_) cuMemUnmap(base_, mapped_bytes_);
        if (base_) cuMemAddressFree(base_, va_bytes_);
    }

  private:
    static CUmemAllocationProp allocation_properties(int device) noexcept {
        CUmemAllocationProp properties = {};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        return properties;
    }
};

#if defined(__CUDACC__)

/**
 *  @brief Vector Search index for a single uniform latency memory space on Hopper GPUs.
 *
 *  Mirrors the CPU-resident volatile index verb-for-verb - @c reserve, @c insert,
 *  @c search - with three deliberate divergences, each documented where it lives:
 *  geometry is compile-time (register tiles are sized by it), a batch's assigned ids are
 *  the contiguous ticket range (no per-vector id span), and publication is unordered
 *  (@ref inner_hopper_index_concurrency). @section inner_hopper_index_growth Growth
 *  The first successful @c reserve() freezes the virtual-address ceiling; physical slabs
 *  map on demand ahead of every launch, and fresh neighbor rows are sentinel-filled
 *  before any ticket covering them can be dispensed.
 */
template <                                               //
    nk_dtype_t vector_dtype_ = nk_i8_k,                  //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k, //
    u32_t dimensions_ = 128,                             //
    connectivity_t connectivity_ = connectivity_32_k>    //
struct inner_hopper_index {

    using kit_t = typename inner_hopper_kit_for<vector_dtype_, kernel_kind_, dimensions_>::kit_t;
    using scalar_t = typename kit_t::scalar_t;
    using fragment_t = typename kit_t::fragment_t;
    using norms_t = typename kit_t::norms_t;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using planes_view_t = inner_hopper_planes_view<kit_t, connectivity_>;

    inline static constexpr u32_t dimensions_k = dimensions_;
    inline static constexpr u32_t connectivity_k = connectivity_;
    inline static constexpr std::size_t max_node_count_k = inner_hopper_scored_node_t::id_capacity_k;
    inline static constexpr std::size_t row_bytes_k = static_cast<std::size_t>(kit_t::row_fragments_k) *
                                                      kit_t::fragment_bytes_k;
    /** @brief The visited-tag arm must cover the 64-row candidate cache. */
    inline static constexpr u32_t tags_count_k = sizeof(fragment_t) * kit_t::row_fragments_k * 64 <= 2048 * 4 ? 2048
                                                                                                              : 4096;

    /** @brief Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;
    };
    /** @brief Outcome of a @c reserve() call: the capacity & connectivity actually in effect. */
    struct reserve_result_t {
        std::size_t new_capacity = 0;
        connectivity_t connectivity = connectivity_;
        status_t status = status_t::success_k;
    };
    /** @brief Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {
        /** @brief Breadth of the construction candidate pool; rounded up to a multiple of 32,
         *      clamped to 512. Larger pools trade build speed for graph quality. */
        u32_t expansion_factor = 0;
    };
    /** @brief Outcome & telemetry of one @c insert() batch. Ids are the contiguous range
     *      @b [first_id,first_id+count) - the ticket order of the incoming rows. */
    struct insert_result_t {
        status_t status = status_t::success_k;
        std::size_t new_size = 0;
        inner_node_id_t first_id = inner_node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        /** @brief Reverse edges abandoned after the bounded lock budget - GPU-only telemetry. */
        std::size_t abandoned_back_links = 0;
        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };
    struct search_config_t {
        /** @brief Breadth of the candidate pool; defaults to twice the connectivity, and is
         *      raised automatically when @c count needs a wider pool. */
        u32_t expansion_factor = 0;
        /** @brief Matches reported per query, up to the widest supported pool of 512. */
        u32_t count = 10;
    };
    struct search_result_t {
        status_t status = status_t::success_k;
        std::size_t count = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };

    inner_hopper_index(int device = 0) noexcept : device_(device) {}
    inner_hopper_index(inner_hopper_index const&) = delete;
    inner_hopper_index& operator=(inner_hopper_index const&) = delete;
    ~inner_hopper_index() noexcept {
        if (counters_) cudaFree(counters_);
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t connectivity() const noexcept { return connectivity_; }
    std::size_t dimensions() const noexcept { return dimensions_; }
    /** @brief Device-side plane views - the audit surface for invariant checkers & tests. */
    planes_view_t planes() const noexcept {
        return {reinterpret_cast<fragment_t*>(vectors_.base_), reinterpret_cast<norms_t*>(norms_.base_),
                reinterpret_cast<u32_t*>(neighbors_.base_)};
    }

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries. The first
     *      successful call freezes the geometry and the virtual-address ceiling; later
     *      calls with matching geometry only map further physical slabs.
     */
    reserve_result_t reserve(reserve_capacity_t config) noexcept {
        if (config.dimensions != dimensions_ || config.connectivity != connectivity_)
            return {mapped_rows_, static_cast<connectivity_t>(connectivity_), status_t::config_mismatch_k};
        if (config.capacity > max_node_count_k)
            return {mapped_rows_, static_cast<connectivity_t>(connectivity_), status_t::overflow_k};
        if (!va_ceiling_rows_) { // First reservation: freeze the ceiling, allocate the counters.
            // Bind the runtime's primary context before any driver-API mapping - the VMM
            // planes must live in the same context the kernels launch into.
            if (cudaSetDevice(device_) != cudaSuccess || cudaFree(nullptr) != cudaSuccess ||
                cudaDeviceGetAttribute(&device_sm_count_, cudaDevAttrMultiProcessorCount, device_) != cudaSuccess)
                return {0, static_cast<connectivity_t>(connectivity_), status_t::device_error_k};
            cuInit(0);
            va_ceiling_rows_ = config.capacity;
            if (vectors_.reserve(va_ceiling_rows_ * row_bytes_k, device_) != status_t::success_k ||
                norms_.reserve(va_ceiling_rows_ * sizeof(norms_t), device_) != status_t::success_k ||
                neighbors_.reserve(va_ceiling_rows_ * connectivity_ * sizeof(u32_t), device_) != status_t::success_k ||
                staging_.reserve(va_ceiling_rows_ * row_bytes_k, device_) != status_t::success_k)
                return {0, static_cast<connectivity_t>(connectivity_), status_t::bad_alloc_k};
            if (cudaMalloc(&counters_, sizeof(inner_hopper_counters_t)) != cudaSuccess ||
                cudaMemset(counters_, 0, sizeof(inner_hopper_counters_t)) != cudaSuccess)
                return {0, static_cast<connectivity_t>(connectivity_), status_t::device_error_k};
        }
        status_t const grown = grow_rows(config.capacity, nullptr);
        return {mapped_rows_, static_cast<connectivity_t>(connectivity_), grown};
    }

    /**
     *  @brief Inserts a batch of @p new_vectors, assigning them the contiguous id range
     *      @b [size(),size()+rows) in row order. Blocks until the batch is fully
     *      published; @p stream orders the transfers & the launch.
     */
    insert_result_t insert(vectors_view_t new_vectors, insert_config_t config = {},
                           cudaStream_t stream = nullptr) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        if (!rows) return {status_t::success_k, size_, static_cast<inner_node_id_t>(size_)};
        if (!va_ceiling_rows_) return {status_t::uninitialized_k};
        if (new_vectors.extent(1) != dimensions_) return {status_t::config_mismatch_k};
        std::size_t const total_rows = size_ + rows;
        if (status_t grown = grow_rows(total_rows, stream); grown != status_t::success_k) return {grown};

        // Stage the incoming vectors at their ticket offsets & snapshot the telemetry.
        if (cudaMemcpyAsync(reinterpret_cast<void*>(staging_.base_ + size_ * row_bytes_k), new_vectors.byte_data(),
                            rows * row_bytes_k, cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return {status_t::device_error_k};
        inner_hopper_counters_t const before = counters_snapshot();

        inner_hopper_build_view<kit_t, connectivity_> view {planes(), //
                                                            reinterpret_cast<fragment_t const*>(staging_.base_),
                                                            static_cast<u32_t>(total_rows), counters_};
        launch_insert(view, pool_slots_for(config.expansion_factor, 4), stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};

        // The restart rule: over-claimed tickets died with the launch - rewind the dispenser.
        inner_hopper_counters_t after = counters_snapshot();
        if (after.published != total_rows) return {status_t::device_error_k};
        after.next_ticket = after.published;
        cudaMemcpy(counters_, &after, sizeof(u32_t) * 2, cudaMemcpyHostToDevice);

        insert_result_t result;
        result.first_id = static_cast<inner_node_id_t>(size_);
        size_ = total_rows;
        result.new_size = total_rows;
        result.visited_members = after.hops - before.hops;
        result.computed_distances = after.scored - before.scored;
        result.restarts = after.restarts - before.restarts;
        result.abandoned_back_links = after.lock_skips - before.lock_skips;
        return result;
    }

    /**
     *  @brief Finds the nearest published neighbors of every query row, writing
     *      @c config.count ids per query into @p result_ids - and their distances into
     *      @p result_distances when it is non-empty - ascending by distance, with sentinel
     *      ids marking absent matches. Safe to overlap with insertions on other streams.
     */
    search_result_t search(vectors_view_t queries, nk::matrix_span<inner_node_id_t> result_ids,
                           nk::matrix_span<inner_distance_t> result_distances = {}, search_config_t config = {},
                           cudaStream_t stream = nullptr) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (!queries_count) return {status_t::success_k};
        if (!va_ceiling_rows_) return {status_t::uninitialized_k};
        bool const wants_distances = result_distances.extent(0) != 0;
        if (queries.extent(1) != dimensions_ || result_ids.extent(0) != queries_count ||
            result_ids.extent(1) < config.count || config.count > 512 ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {status_t::config_mismatch_k};

        void* device_queries = nullptr;
        inner_node_id_t* device_ids = nullptr;
        inner_distance_t* device_distances = nullptr;
        std::size_t const results_count = queries_count * config.count;
        if (cudaMallocAsync(&device_queries, queries_count * row_bytes_k, stream) != cudaSuccess ||
            cudaMallocAsync(reinterpret_cast<void**>(&device_ids), results_count * sizeof(inner_node_id_t), stream) !=
                cudaSuccess ||
            (wants_distances && cudaMallocAsync(reinterpret_cast<void**>(&device_distances),
                                                results_count * sizeof(inner_distance_t), stream) != cudaSuccess))
            return {status_t::bad_alloc_k};
        cudaMemcpyAsync(device_queries, queries.byte_data(), queries_count * row_bytes_k, cudaMemcpyHostToDevice,
                        stream);
        inner_hopper_counters_t const before = counters_snapshot();

        inner_hopper_search_view<kit_t, connectivity_> view {planes(),
                                                             reinterpret_cast<fragment_t const*>(device_queries),
                                                             device_ids,
                                                             device_distances,
                                                             static_cast<u32_t>(queries_count),
                                                             config.count,
                                                             static_cast<u32_t>(size_),
                                                             counters_};
        // The pool must cover the requested count, whatever the expansion asked for.
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const expansion = config.expansion_factor > config.count ? config.expansion_factor : config.count * 2;
        launch_search(view, pool_slots_for(expansion, count_slots > 2 ? count_slots : 2), stream);
        cudaMemcpyAsync(result_ids.data(), device_ids, results_count * sizeof(inner_node_id_t), cudaMemcpyDeviceToHost,
                        stream);
        if (wants_distances)
            cudaMemcpyAsync(result_distances.data(), device_distances, results_count * sizeof(inner_distance_t),
                            cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(device_queries, stream);
        cudaFreeAsync(device_ids, stream);
        if (wants_distances) cudaFreeAsync(device_distances, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};

        inner_hopper_counters_t const after = counters_snapshot();
        search_result_t result;
        result.count = config.count;
        result.visited_members = after.hops - before.hops;
        result.computed_distances = after.scored - before.scored;
        return result;
    }

  private:
    inner_hopper_arena_t vectors_, norms_, neighbors_, staging_;
    inner_hopper_counters_t* counters_ = nullptr;
    std::size_t va_ceiling_rows_ = 0;
    std::size_t mapped_rows_ = 0;
    std::size_t size_ = 0;
    int device_ = 0;
    int device_sm_count_ = 0;

    /** @brief Maps every plane up to @p rows and sentinel-fills the fresh neighbor rows -
     *      always ahead of any launch that could dispense tickets into them. */
    status_t grow_rows(std::size_t rows, cudaStream_t stream) noexcept {
        if (rows <= mapped_rows_) return status_t::success_k;
        std::size_t const neighbors_before = neighbors_.mapped_bytes_;
        if (status_t status = vectors_.grow_to(rows * row_bytes_k); status != status_t::success_k) return status;
        if (status_t status = norms_.grow_to(rows * sizeof(norms_t)); status != status_t::success_k) return status;
        if (status_t status = neighbors_.grow_to(rows * connectivity_ * sizeof(u32_t)); status != status_t::success_k)
            return status;
        if (status_t status = staging_.grow_to(rows * row_bytes_k); status != status_t::success_k) return status;
        std::size_t const fresh_bytes = neighbors_.mapped_bytes_ - neighbors_before;
        if (fresh_bytes && cuMemsetD32Async(neighbors_.base_ + neighbors_before, inner_node_id_unknown_k,
                                            fresh_bytes / 4, stream) != CUDA_SUCCESS)
            return status_t::device_error_k;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return status_t::device_error_k;
        mapped_rows_ = neighbors_.mapped_bytes_ / (connectivity_ * sizeof(u32_t));
        return status_t::success_k;
    }

    inner_hopper_counters_t counters_snapshot() const noexcept {
        inner_hopper_counters_t snapshot = {};
        cudaMemcpy(&snapshot, counters_, sizeof(snapshot), cudaMemcpyDeviceToHost);
        return snapshot;
    }

    /** @brief Rounds the requested expansion to the closed pool ladder: 64/128/256/512. */
    static u32_t pool_slots_for(u32_t expansion_factor, u32_t default_slots) noexcept {
        if (!expansion_factor) return default_slots;
        u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion_factor, 32));
        return min_of(max_of(slots, 2u), 16u);
    }

    /** @brief Blocks per grid: three resident blocks per SM saturate the insertion path
     *      at its register budget; oversubscribing eight per SM keeps enough independent
     *      queries in flight to reach the memory-level-parallelism knee while searching. */
    inline static constexpr u32_t insert_blocks_per_sm_k = 3;
    inline static constexpr u32_t search_blocks_per_sm_k = 8;
    inline static constexpr u32_t block_threads_k = inner_hopper_warps_per_block_k * 32;

    void launch_insert(inner_hopper_build_view<kit_t, connectivity_> view, u32_t pool_slots,
                       cudaStream_t stream) noexcept {
        u32_t const blocks = device_sm_count_ * insert_blocks_per_sm_k;
        switch (pool_slots) {
        case 2:
            inner_hopper_insert_kernel<kit_t, connectivity_, tags_count_k, 2>
                <<<blocks, block_threads_k, 0, stream>>>(view);
            break;
        case 4:
            inner_hopper_insert_kernel<kit_t, connectivity_, tags_count_k, 4>
                <<<blocks, block_threads_k, 0, stream>>>(view);
            break;
        case 8:
            inner_hopper_insert_kernel<kit_t, connectivity_, tags_count_k, 8>
                <<<blocks, block_threads_k, 0, stream>>>(view);
            break;
        default:
            inner_hopper_insert_kernel<kit_t, connectivity_, tags_count_k, 16>
                <<<blocks, block_threads_k, 0, stream>>>(view);
        }
    }
    void launch_search(inner_hopper_search_view<kit_t, connectivity_> view, u32_t pool_slots,
                       cudaStream_t stream) noexcept {
        u32_t const blocks = device_sm_count_ * search_blocks_per_sm_k;
        switch (pool_slots) {
        case 2:
            inner_hopper_search_kernel<kit_t, connectivity_, tags_count_k, 2>
                <<<blocks, block_threads_k, 0, stream>>>(view);
            break;
        case 4:
            inner_hopper_search_kernel<kit_t, connectivity_, tags_count_k, 4>
                <<<blocks, block_threads_k, 0, stream>>>(view);
            break;
        case 8:
            inner_hopper_search_kernel<kit_t, connectivity_, tags_count_k, 8>
                <<<blocks, block_threads_k, 0, stream>>>(view);
            break;
        default:
            inner_hopper_search_kernel<kit_t, connectivity_, tags_count_k, 16>
                <<<blocks, block_threads_k, 0, stream>>>(view);
        }
    }
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
