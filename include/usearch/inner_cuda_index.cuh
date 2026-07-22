/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *  @author Ash Vardanian
 *  @file include/usearch/inner_cuda_index.cuh
 *  @date July 18, 2026
 */
#pragma once

#include <atomic>  // `std::atomic` row claiming across concurrent inserting contexts
#include <thread>  // `std::this_thread::yield` while another context grows the planes
#include <bit>     // `std::bit_width` for hash-prefix widths
#include <cstddef> // `std::byte` for raw scratch & arena spans
#include <cstring> // `std::memcpy`

#include <cuda.h>         // Driver API: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`
#include <cuda_runtime.h> // Runtime API: streams, events, `cudaMemcpyAsync`
#if defined(__CUDACC__)
#include <cuda/atomic>     // `cuda::atomic_ref` - the device twin of `std::atomic_ref`
#include <cuda_pipeline.h> // `__pipeline_memcpy_async` - register-free global-to-shared copies
#endif

#include <forkunion/types.hpp> // `fu::round_up_to_multiple` & friends

#include <numkong/capabilities.h> // `nk_kernel_kind_t` & the kernel-kind constants
#include <numkong/matrix.hpp>
#include <numkong/types.h>
#include <numkong/vector.hpp>

#include "usearch/cuda_device.hpp"
#include "usearch/inner_metric.cuh"
#include "usearch/types.hpp"

namespace unum {
namespace usearch {

/** @brief Warps per thread block across every kernel of this design - four schedulers per
 *      SM on every modern NVIDIA architecture, one warp each. */
inline constexpr u32_t inner_cuda_warps_per_block_k = 4;
/** @brief Resident blocks per SM - a residency contract shared by the launch grids and the
 *      kernels' launch bounds, so the register allocator sizes to it instead of trading
 *      spills for occupancy the launches never ask for. Search kernels with small pools fit
 *      a four-block contract spill-free and earn a third more resident warps; the wider
 *      pool ladder and every insert kernel stay at three. */
inline constexpr u32_t inner_cuda_resident_blocks_per_sm_k = 3;
/** @brief Rows of the per-warp shared candidate cache; also bounds the diversity-selection
 *      pool prefix and the staging arrays. */
inline constexpr u32_t inner_cuda_cache_rows_k = 64;
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
struct inner_cuda_scored_node_t {
    u64_t bits_;

    inline static constexpr u64_t expanded_bit_k = u64_t(1) << 32;
    /** @brief Dispensable ids: everything but the reserved all-ones sentinel. */
    inline static constexpr std::size_t id_capacity_k = (u64_t(1) << 32) - 1u;

    static __device__ __forceinline__ inner_cuda_scored_node_t make(inner_distance_t distance,
                                                                    inner_node_id_t node_id) noexcept {
        u32_t flipped = __float_as_uint(distance);
        flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return {(static_cast<u64_t>(flipped >> 1) << 33) | node_id};
    }
    __device__ static constexpr inner_cuda_scored_node_t sentinel() noexcept { return {~u64_t(0)}; }

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
    __device__ __forceinline__ bool operator<(inner_cuda_scored_node_t other) const noexcept {
        return bits_ < other.bits_;
    }
    __device__ __forceinline__ bool operator==(inner_cuda_scored_node_t other) const noexcept {
        return bits_ == other.bits_;
    }

    /** @brief Warp shuffles have no user-type overloads - the entry shuffles its raw word. */
    __device__ __forceinline__ inner_cuda_scored_node_t shuffle_xor(int lane_mask) const noexcept {
        return {__shfl_xor_sync(0xFFFFFFFFu, bits_, lane_mask)};
    }
    __device__ __forceinline__ inner_cuda_scored_node_t shuffle_from(int source_lane) const noexcept {
        return {__shfl_sync(0xFFFFFFFFu, bits_, source_lane)};
    }
    __device__ __forceinline__ inner_cuda_scored_node_t shuffle_up(int delta) const noexcept {
        return {__shfl_up_sync(0xFFFFFFFFu, bits_, delta)};
    }
};
static_assert(sizeof(inner_cuda_scored_node_t) == 8, "Entries shuffle as one 64-bit word");
static_assert(std::is_trivially_copyable_v<inner_cuda_scored_node_t>);

/**
 *  @brief Full bitonic sort of one 32-lane entry vector, ascending by lane - fifteen
 *      shuffle-exchange stages, all in registers.
 */
__device__ __forceinline__ inner_cuda_scored_node_t inner_cuda_sort32(inner_cuda_scored_node_t value,
                                                                      int lane) noexcept {
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
        for (int j = k >> 1; j; j >>= 1) {
            inner_cuda_scored_node_t const partner = value.shuffle_xor(j);
            bool const up = (lane & k) == 0;
            bool const lower = (lane & j) == 0;
            inner_cuda_scored_node_t const nearer = value < partner ? value : partner;
            inner_cuda_scored_node_t const further = value < partner ? partner : value;
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
struct inner_cuda_sorted_pool {
    inner_cuda_scored_node_t entries_[slots_];

    inline static constexpr u32_t capacity_k = slots_ * 32;
    static_assert((slots_ & (slots_ - 1)) == 0, "The fold-merge network needs a power-of-two slot count");

    __device__ __forceinline__ void clear() noexcept {
#pragma unroll
        for (u32_t slot = 0; slot != slots_; ++slot) entries_[slot] = inner_cuda_scored_node_t::sentinel();
    }

    /**
     *  @brief Merges a 32-lane staged batch into the pool, keeping the 32x @c slots_
     *      nearest. [ascending pool | sentinel pad | descending batch] is one bitonic
     *      sequence: a single fold absorbs the batch, log2 of @c slots_ in-lane stages plus
     *      five shuffle stages restore ascending order.
     */
    __device__ __forceinline__ void insert_batch(inner_cuda_scored_node_t staged, int lane) noexcept {
        (void)insert_batch_spilling(staged, lane);
    }

    /**
     *  @brief The same fold-merge, reporting what fell out: at the fold boundary the
     *      element-wise @b max of the last slot and the descending batch is exactly the
     *      union's worst 32 - sentinels while the pool still has room. @return This
     *      lane's displaced entry, for the caller to spill onward or drop.
     */
    __device__ __forceinline__ inner_cuda_scored_node_t insert_batch_spilling(inner_cuda_scored_node_t staged,
                                                                              int lane) noexcept {
        inner_cuda_scored_node_t const descending = inner_cuda_sort32(staged, lane).shuffle_xor(31);
        inner_cuda_scored_node_t const displaced = entries_[slots_ - 1] < descending ? descending
                                                                                     : entries_[slots_ - 1];
        entries_[slots_ - 1] = entries_[slots_ - 1] < descending ? entries_[slots_ - 1] : descending;
#pragma unroll
        for (u32_t stride = slots_ / 2; stride >= 1; stride >>= 1)
#pragma unroll
            for (u32_t base = 0; base < slots_; ++base)
                if ((base & stride) == 0 && (base | stride) < slots_) {
                    inner_cuda_scored_node_t const a = entries_[base], b = entries_[base | stride];
                    entries_[base] = a < b ? a : b;
                    entries_[base | stride] = a < b ? b : a;
                }
#pragma unroll
        for (int j = 16; j; j >>= 1) {
            bool const lower = (lane & j) == 0;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot) {
                inner_cuda_scored_node_t const partner = entries_[slot].shuffle_xor(j);
                entries_[slot] = lower == (entries_[slot] < partner) ? entries_[slot] : partner;
            }
        }
        return displaced;
    }

    /** @brief The rank capacity - also the base of the traversal's hop budget. */
    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_k; }

    /** @brief The nearest entry not yet expanded - the frontier's @c top() - or the sentinel. */
    __device__ __forceinline__ inner_cuda_scored_node_t nearest_unexpanded(int lane) const noexcept {
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot) {
            unsigned const mask = __ballot_sync(0xFFFFFFFFu, !entries_[slot].expanded());
            if (mask) return entries_[slot].shuffle_from(__ffs(mask) - 1);
        }
        return inner_cuda_scored_node_t::sentinel();
    }

    /** @brief Lazily deletes @p entry from the frontier - the unrolled-compare spelling
     *      keeps every slot index compile-time. Duplicates cannot arise: the exact
     *      visited set admits every id into the walk exactly once. */
    __device__ __forceinline__ void mark_expanded_at(inner_cuda_scored_node_t entry) noexcept {
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
        return !(entries_[slots_ - 1].shuffle_from(31) == inner_cuda_scored_node_t::sentinel());
    }

    /**
     *  @brief Kills every staged entry whose id already resides in the pool. The walk
     *      itself never needs this - the exact visited set stages every id once - but the
     *      arena frontier's final fold does: a node promoted out of the tail and later
     *      snapshotted back leaves two arena copies of one id.
     */
    __device__ __forceinline__ inner_cuda_scored_node_t deduplicate(inner_cuda_scored_node_t staged,
                                                                    int lane) const noexcept {
        inner_node_id_t const staged_id = staged.id();
        bool resident = false;
#pragma unroll
        for (int probe = 0; probe < 32; ++probe) {
            inner_node_id_t const probe_id = __shfl_sync(0xFFFFFFFFu, staged_id, probe);
            bool holds = false;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot)
                holds |= !(entries_[slot] == inner_cuda_scored_node_t::sentinel()) && entries_[slot].id() == probe_id;
            if (__any_sync(0xFFFFFFFFu, holds) && lane == probe) resident = true;
        }
        return resident ? inner_cuda_scored_node_t::sentinel() : staged;
    }

    /** @brief Entry at @p rank, valid on every lane. */
    __device__ __forceinline__ inner_cuda_scored_node_t rank_entry(u32_t rank) const noexcept {
        inner_cuda_scored_node_t result = inner_cuda_scored_node_t::sentinel();
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (slot == rank / 32) result = entries_[slot];
        return result.shuffle_from(rank % 32);
    }
};

/** @brief One warp's slice of the global traversal arena: an append-only span of scored
 *      entries spilled from the register head, consumed by refills on frontier drain.
 *      Appends past @p capacity_ are counted, never written - the nearest candidates
 *      always reach the head first, so drops only ever shed the farthest tail. */
struct inner_cuda_tail_pool_t {
    inner_cuda_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_ = 0;
    u32_t dropped_ = 0;

    __device__ __forceinline__ void append(inner_cuda_scored_node_t value, int lane) noexcept {
        bool const live = !(value == inner_cuda_scored_node_t::sentinel());
        unsigned const live_mask = __ballot_sync(0xFFFFFFFFu, live);
        u32_t const ordinal = __popc(live_mask & ((1u << lane) - 1));
        if (live && size_ + ordinal < capacity_) entries_[size_ + ordinal] = value;
        u32_t const grown = min_of(size_ + __popc(live_mask), capacity_);
        dropped_ += size_ + __popc(live_mask) - grown;
        size_ = grown;
        __syncwarp();
    }
};

/**
 *  @brief The register frontier: the fused results-and-frontier of a bounded walk, a thin
 *      verb layer over one register pool. The CPU keeps its results buffer and frontier
 *      heap separate; registers fuse them - the expanded bit plays frontier deletion, the
 *      last rank plays the results radius - and this wrapper names the traversal verbs so
 *      one walk body serves this and the arena twin below.
 */
template <u32_t slots_>
struct inner_cuda_register_frontier {
    inner_cuda_sorted_pool<slots_> ranks_;

    __device__ __forceinline__ void reset(int) noexcept { ranks_.clear(); }
    __device__ __forceinline__ u32_t hops_budget() const noexcept { return ranks_.capacity() * 4; }
    __device__ __forceinline__ void push_batch(inner_cuda_scored_node_t staged, int lane) noexcept {
        // A full pool keeps the nearest of the union, so a batch with no entry inside the
        // radius is a guaranteed no-op - one ballot skips the whole bitonic merge. Only
        // the narrow pools take the gate: past them the extra live radius dependency
        // costs the 3-block kernels their zero-spill budget.
        if constexpr (slots_ <= 4)
            if (ranks_.full() && !__ballot_sync(0xFFFFFFFFu, staged.distance_bits() < ranks_.radius())) return;
        ranks_.insert_batch(staged, lane);
    }
    /** @brief The nearest undiscovered-from candidate, already marked expanded - or the
     *      sentinel when every rank is spent. */
    __device__ __forceinline__ inner_cuda_scored_node_t pop_nearest(int lane) noexcept {
        inner_cuda_scored_node_t const nearest = ranks_.nearest_unexpanded(lane);
        if (!(nearest == inner_cuda_scored_node_t::sentinel())) ranks_.mark_expanded_at(nearest);
        return nearest;
    }
    /** @brief After a sentinel pop: full ranks mean the walk is done, not just starved. */
    __device__ __forceinline__ bool exhausted() const noexcept { return ranks_.full(); }
    /** @brief The classic radius test: full ranks and a candidate beyond the worst kept. */
    __device__ __forceinline__ bool beyond_radius(inner_cuda_scored_node_t candidate) const noexcept {
        return ranks_.full() && candidate.distance_bits() > ranks_.radius();
    }
    /** @brief Ranked results are the container itself - nothing to fold. */
    __device__ __forceinline__ void finalize_ranks(int) noexcept {}
    __device__ __forceinline__ inner_cuda_scored_node_t rank_entry(u32_t rank) const noexcept {
        return ranks_.rank_entry(rank);
    }
};

/**
 *  @brief The arena frontier: the de-capped twin - a register head (the exact nearest 64,
 *      absorbing every per-hop merge at constant cost) plus an arena tail taking the
 *      head's spill. The frontier refills from the tail when the head's unexpanded
 *      entries drain, so expansion follows discovery order past the head's width, and
 *      the walk retires after @c expansion_target_ expansions instead of on a radius test - the
 *      expansion budget IS the ef semantic and the horizon.
 */
struct inner_cuda_arena_frontier {
    inner_cuda_sorted_pool<2> head_;
    inner_cuda_tail_pool_t tail_;
    u32_t expansion_target_ = 0;
    u32_t expansions_ = 0;

    __device__ __forceinline__ void reset(int) noexcept {
        head_.clear();
        tail_.size_ = 0;
        expansions_ = 0;
    }
    __device__ __forceinline__ u32_t hops_budget() const noexcept { return expansion_target_ * 4; }
    __device__ __forceinline__ void push_batch(inner_cuda_scored_node_t staged, int lane) noexcept {
        // When every staged entry sits at or past the head's radius the merge displaces
        // nothing - the batch goes straight to the tail, which by construction only holds
        // entries farther than everything head-resident, and staged >= radius keeps that.
        if (head_.full() && !__ballot_sync(0xFFFFFFFFu, staged.distance_bits() < head_.radius()))
            return tail_.append(staged, lane);
        tail_.append(head_.insert_batch_spilling(staged, lane), lane);
    }
    __device__ __forceinline__ inner_cuda_scored_node_t pop_nearest(int lane) noexcept {
        // A consumed expansion budget IS the frontier's edge: reporting starvation here
        // routes the walk into its drained-frontier exit, where exhausted() seals it.
        if (exhausted()) return inner_cuda_scored_node_t::sentinel();
        inner_cuda_scored_node_t nearest = head_.nearest_unexpanded(lane);
        if (nearest == inner_cuda_scored_node_t::sentinel()) {
            refill(lane);
            nearest = head_.nearest_unexpanded(lane);
        }
        if (!(nearest == inner_cuda_scored_node_t::sentinel())) {
            head_.mark_expanded_at(nearest);
            ++expansions_;
        }
        return nearest;
    }
    __device__ __forceinline__ bool exhausted() const noexcept { return expansions_ >= expansion_target_; }
    __device__ __forceinline__ bool beyond_radius(inner_cuda_scored_node_t) const noexcept { return false; }
    __device__ __forceinline__ inner_cuda_scored_node_t rank_entry(u32_t rank) const noexcept {
        return head_.rank_entry(rank);
    }

    /** @brief Promotes the nearest unexpanded tail entries into the head. Every tail
     *      entry is farther than everything head-resident - that is the spill invariant -
     *      so a merge could never admit them: instead the drained head, all expanded by
     *      now, spills wholesale to the tail for the final compaction, and the champions
     *      take it over as the fresh frontier. Lane champions are not the global top 32,
     *      but refills recur on every drain, so promotion order only affects scheduling,
     *      never membership. */
    __device__ __forceinline__ void refill(int lane) noexcept {
        inner_cuda_scored_node_t best = inner_cuda_scored_node_t::sentinel();
        u32_t best_index = 0;
        for (u32_t index = lane; index < tail_.size_; index += 32) {
            inner_cuda_scored_node_t const entry = tail_.entries_[index];
            if (!entry.expanded() && entry < best) best = entry, best_index = index;
        }
        __syncwarp();
        if (!__ballot_sync(0xFFFFFFFFu, !(best == inner_cuda_scored_node_t::sentinel()))) return;
        // Promotion tombstones the source slot, so every id lives in exactly one place:
        // promoted ids exist only in the head until the head itself re-spills them - once.
        // Each lane's best comes from its own stride residues, so the stores never clash.
        if (!(best == inner_cuda_scored_node_t::sentinel()))
            tail_.entries_[best_index] = inner_cuda_scored_node_t::sentinel();
        tail_.append(head_.entries_[0], lane);
        tail_.append(head_.entries_[1], lane);
        head_.clear();
        head_.insert_batch(best, lane);
    }

    /** @brief Rebuilds the exact nearest 64 from everything the walk has seen - the last
     *      head generation plus every tail entry - and parks them in the head for ranked
     *      extraction. Single residency by construction: promotion tombstones its tail
     *      slot, so no id ever holds two live entries and the fold needs no dedup pass.
     *      Call once, after the walk. */
    __device__ __forceinline__ void finalize_ranks(int lane) noexcept {
        inner_cuda_sorted_pool<2> results;
        results.clear();
        results.insert_batch(head_.entries_[0], lane);
        results.insert_batch(head_.entries_[1], lane);
        for (u32_t base = 0; base < tail_.size_; base += 32) {
            u32_t const index = base + lane;
            inner_cuda_scored_node_t const entry = index < tail_.size_ ? tail_.entries_[index]
                                                                       : inner_cuda_scored_node_t::sentinel();
            results.insert_batch(entry, lane);
        }
        head_ = results;
    }
};

/**
 *  @brief Exact visited set over a warp-private arena span: all-ones empty sentinel,
 *      multiplicative hash, linear probing - the CPU growing hash set's discipline, with
 *      the capacity sized ahead by the host so it never grows. The @b only kind of
 *      membership structure this codebase allows: lossy variants are banned outright,
 *      their false negatives silently invert the recall-vs-expansion curve.
 *
 *  Lanes carrying the same id in one call collapse to their lowest lane via hardware
 *  value matching, so a batch can never stage intra-batch duplicates; the
 *  compare-exchange only arbitrates two distinct ids landing on one empty slot -
 *  scratch bookkeeping, not part of the graph's all-RMW protocol. A saturated set - the
 *  probe budget exhausted on a full neighborhood - reports ids as already visited, so an
 *  under-provisioned walk starves and retires deterministically instead of looping.
 */
struct inner_cuda_visited_set_t {
    /** @brief Packed @b {epoch,id} slots: an entry is live only when its epoch matches
     *      this walk's, so "resetting" is bumping one register - the host zeroes the
     *      arena once per launch batch, and epoch zero marks never-used. */
    u64_t* slots_;
    /** @brief One field carries the whole shape: the slot count is a power of two by
     *      construction in @c arena_shape_for, so the probe mask derives as
     *      @b 0xFFFFFFFF>>hash_shift_ and no capacity is stored. */
    u32_t hash_shift_;
    u32_t epoch_;

    inline static constexpr u32_t probe_budget_k = 64;

    /** @brief Marks @p node_id visited; @c true when it was fresh. */
    __device__ __forceinline__ bool test_and_set(inner_node_id_t node_id, int lane, bool active) noexcept {
        unsigned const peers = __match_any_sync(0xFFFFFFFFu, active ? node_id : 0xFFFFFFFFu ^ lane);
        bool fresh = active && (peers & ((1u << lane) - 1)) == 0;
        bool probing = fresh;
        u64_t const claim = (static_cast<u64_t>(epoch_) << 32) | node_id;
        u32_t const mask = 0xFFFFFFFFu >> hash_shift_;
        u32_t slot = (node_id * 0x9E3779B9u) >> hash_shift_;
        // The slice is warp-private, so the only contention is two lanes of THIS warp
        // landing on one empty slot in the same round - arbitrated in registers, no
        // atomics: the lowest rival lane stores, the warp barrier publishes the store,
        // and losers re-read the slot, see a foreign epoch-current id, and probe on.
        for (u32_t round = 0; __any_sync(0xFFFFFFFFu, probing) && round != probe_budget_k; ++round) {
            u64_t const observed = probing ? slots_[slot] : 0;
            bool claiming = false;
            if (probing) {
                if (observed == claim) fresh = probing = false; // Already visited this walk.
                else if (static_cast<u32_t>(observed >> 32) == epoch_) slot = (slot + 1) & mask;
                else claiming = true; // Stale or never used.
            }
            unsigned const rivals = __match_any_sync(0xFFFFFFFFu, claiming ? slot : 0xFFFFFFFFu ^ lane);
            if (claiming && (rivals & ((1u << lane) - 1)) == 0) slots_[slot] = claim, probing = false;
            __syncwarp();
        }
        if (probing) fresh = false; // Saturation: claim "visited", starve, retire.
        __syncwarp();
        return fresh;
    }
};

#endif // defined(__CUDACC__)

/** @brief Device-resident index control block - the state warps consult across SMs and
 *      across launches. One per index, allocated through the traversals allocator; every
 *      driving context polls it, none owns it. All fields are all-RMW, no compare-and-swap.
 *      @ref inner_cuda_index_concurrency */
struct inner_cuda_control_t {
    /** @brief Published-node count: advanced by a fire-and-forget @c atomicAdd at each
     *      slot-0 release, polled by the admission ramp. Advisory - it implies nothing
     *      about @b which ids are published. */
    u32_t published_count;
};

/** @brief Device-resident telemetry sink, aggregated once per traversal. One per driving
 *      context, never shared - two concurrent launches diffing one block would
 *      cross-contaminate their before/after snapshots. */
struct inner_cuda_telemetry_block_t {
    u64_t hops, scored, lock_attempts, lock_skips, full_lists, restarts, locked_drops;
};

#if defined(__CUDACC__)

/**
 *  @brief Typed views over the three device planes, @b [rows,*] each: packed vectors,
 *      per-row norms, and fixed-capacity neighbor lists. Under the flat virtual-memory
 *      arena the node id @b is the location - one multiply, no chunk directory.
 */
template <inner_cuda_kit kit_>
struct inner_cuda_planes_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** @brief Vector plane, @b [rows,row_fragments] fragments, one aligned row per node. */
    fragment_t* vectors;
    /** @brief Norms plane, @b [rows] records, written once pre-publication. */
    norms_t* norms;
    /** @brief Neighbor plane, @b [rows,connectivity] ids; slot 0 doubles as the lock word. */
    u32_t* neighbors;
    /** @brief Neighbor-row width: the stride of @c neighbors and the link budget per node. */
    u32_t connectivity;
    /** @brief Fragments per vector row - the stride of @c vectors; the row's byte length is
     *      the dimension count rounded up to whole fragments, tail bytes zeroed. */
    u32_t row_fragments;
    /** @brief Lanes cooperating on one row while scoring: the smallest power of two
     *      covering @c row_fragments, capped at the warp width. */
    u32_t lanes_per_row;
    /** @brief Scoring steps each of those lanes takes to cover one full row. */
    u32_t steps_per_row;
    /** @brief Fixed-point reciprocal of @c row_fragments - `ceil(2^31 / row_fragments)` -
     *      so flattened [row,fragment] word indices split without hardware division. The
     *      2^31 base fits any fragment width in 32 bits, and exactness holds whenever
     *      word x row_fragments stays under 2^31 - true for every cache-fill index here. */
    u32_t row_fragments_magic;

    /** @brief Splits a flattened word index over a @b [rows,row_fragments] grid - one
     *      32x32 widening multiply and a shift. */
    __device__ __forceinline__ u32_t row_of_word(u32_t word) const noexcept {
        return static_cast<u32_t>((static_cast<u64_t>(word) * row_fragments_magic) >> 31);
    }
    __device__ __forceinline__ u32_t fragment_of_word(u32_t word, u32_t row) const noexcept {
        return word - row * row_fragments;
    }

    /** @brief Lane's fragment of the vector stored at @p node_id. */
    __device__ __forceinline__ fragment_t vector_for(inner_node_id_t node_id, u32_t fragment_index) const noexcept {
        return vectors[static_cast<std::size_t>(node_id) * row_fragments + fragment_index];
    }
    __device__ __forceinline__ fragment_t* vector_span_for(inner_node_id_t node_id) const noexcept {
        return vectors + static_cast<std::size_t>(node_id) * row_fragments;
    }
    __device__ __forceinline__ norms_t norm_for(inner_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ norms_t& norm_ref_for(inner_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ u32_t* neighbors_for(inner_node_id_t node_id) const noexcept {
        return neighbors + static_cast<std::size_t>(node_id) * connectivity;
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
struct inner_cuda_neighbors_ref_t {
    u32_t* row_; // Non-const even for readers: `cuda::atomic_ref` cannot bind const words pre-P3323.
    u32_t capacity_;

    using word_ref_t = cuda::atomic_ref<u32_t, cuda::thread_scope_device>;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }
    /** @brief Warp-ballot mask of the lanes whose index is a valid slot of this row -
     *      collective row reads report the sentinel on the lanes outside it. */
    __device__ __forceinline__ unsigned lane_mask() const noexcept {
        return capacity_ >= 32 ? 0xFFFFFFFFu : (1u << capacity_) - 1u;
    }

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
        if (static_cast<u32_t>(lane) >= capacity_) return inner_node_id_unknown_k;
        return lane ? load_slot(lane) : peek_first();
    }
    /** @brief Warp-collective row read for the @b lock holder: lane 0 reports the
     *      displaced @p first_value, the others load their slots relaxed. */
    __device__ __forceinline__ inner_node_id_t locked_row(int lane, inner_node_id_t first_value) const noexcept {
        if (static_cast<u32_t>(lane) >= capacity_) return inner_node_id_unknown_k;
        return lane ? load_slot(lane) : first_value;
    }
};

/** @brief Per-warp register telemetry, folded into the global counters once per node.
 *      Counters are 32-bit in registers - one walk's hops cap at a few thousand and every
 *      other counter is bounded by hops times the connectivity - and widen at the flush. */
struct inner_cuda_telemetry_t {
    u32_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, full_lists = 0, restarts = 0, locked_drops = 0;

    /** @brief Folds the live counters into the device block; the search path never touches
     *      the lock or list counters, so its instantiation skips four dead atomics. */
    template <bool inserting_>
    __device__ __forceinline__ void flush(inner_cuda_telemetry_block_t* counters, int lane) noexcept {
        // @c atomicAdd spells 64-bit as `unsigned long long`; @c u64_t is layout-identical.
        static_assert(sizeof(u64_t) == sizeof(unsigned long long));
        if (lane != 0) return;
        using wide_t = unsigned long long;
        atomicAdd(reinterpret_cast<wide_t*>(&counters->hops), static_cast<wide_t>(hops));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->scored), static_cast<wide_t>(scored));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->restarts), static_cast<wide_t>(restarts));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->locked_drops), static_cast<wide_t>(locked_drops));
        if constexpr (!inserting_) return;
        atomicAdd(reinterpret_cast<wide_t*>(&counters->lock_attempts), static_cast<wide_t>(lock_attempts));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->lock_skips), static_cast<wide_t>(lock_skips));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->full_lists), static_cast<wide_t>(full_lists));
    }
};

/**
 *  @brief Each warp performing scans over the graph owns one context like this - the
 *      device twin of the CPU's per-thread traversal context: the candidate pool, the
 *      lane's PRNG stream, and the running telemetry.
 */
template <typename frontier_>
struct inner_cuda_warp_context {
    frontier_ frontier;
    splitmix_u32_t random_generator;
    inner_cuda_telemetry_t telemetry;
};

/**
 *  @brief Scores one staged batch of up to 32 candidate rows as a single flattened load
 *      gather batch. Rows are gathered in groups: a sub-warp of @c lanes_per_row lanes owns
 *      one row, stepping @c steps_per_row times across its fragments, while the group tile
 *      keeps up to @c max_gathers_in_flight_k independent loads per lane. The query lives in
 *      a shared-memory span and is re-read by broadcast each step, so no per-lane query
 *      registers cap the dimension count.
 *
 *  Mutable-plane discipline: vector & norm loads here are plain loads, made safe by the
 *  acquire chain - every staged id was discovered through an acquiring slot-0 read that
 *  synchronizes with its publisher's release. @return This lane's scored entry, or the
 *  sentinel on lanes past @p gathered_count.
 */
template <inner_cuda_kit kit_, u32_t resident_blocks_ = inner_cuda_resident_blocks_per_sm_k>
__device__ __forceinline__ inner_cuda_scored_node_t inner_cuda_score_batch( //
    inner_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, u32_t const (&gathered_ids)[inner_cuda_cache_rows_k],
    inner_cuda_scored_node_t (&gathered_scored)[32], u32_t gathered_count, int lane, u32_t first_row = 0) noexcept {

    using norms_t = typename kit_::norms_t;
    // The gather window targets the SM's outstanding-request knee: 128 concurrent random
    // gathers saturate the HBM path, so window = 128 / resident warps. On the 4-block
    // kernels this knowingly oversubscribes the register file by a few registers - ptxas
    // spills a handful of COLD loop-carried bytes to keep every HOT gather slot live.
    constexpr u32_t gathers_in_flight_k = min_of(kit_::max_gathers_in_flight_k,
                                                 128u / (inner_cuda_warps_per_block_k * resident_blocks_));
    u32_t const lanes_log2 = 31 - __clz(planes.lanes_per_row);
    u32_t const rows_per_gather = 32u >> lanes_log2;
    u32_t const lane_in_row = static_cast<u32_t>(lane) & (planes.lanes_per_row - 1);
    u32_t const subwarp = static_cast<u32_t>(lane) >> lanes_log2;
    bool const leader = lane_in_row == 0;

    // Row groups are independent; the pass loop tiles them so at most `gathers_in_flight_k` loads -
    // the gather register budget - coexist per lane.
#pragma unroll 1
    for (u32_t group_base = 0; group_base < planes.lanes_per_row; group_base += gathers_in_flight_k) {
        norms_t batch_norms[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (kit_::has_norms_k && leader && group_base + slot < planes.lanes_per_row && row < gathered_count)
                batch_norms[slot] = planes.norm_for(gathered_ids[first_row + row]);
        }
        typename kit_::accumulator_t accumulators[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) accumulators[slot] = kit_::zero_accumulator();
#pragma unroll 1
        for (u32_t step = 0; step != planes.steps_per_row; ++step) {
            u32_t const fragment_index = lane_in_row + (step << lanes_log2);
            bool const in_row = fragment_index < planes.row_fragments;
            typename kit_::fragment_t const query_value = in_row ? query_span[fragment_index]
                                                                 : typename kit_::fragment_t {};
            typename kit_::fragment_t loaded[gathers_in_flight_k];
#pragma unroll
            for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
                u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
                if (in_row && group_base + slot < planes.lanes_per_row && row < gathered_count)
                    loaded[slot] = planes.vector_for(gathered_ids[first_row + row], fragment_index);
            }
#pragma unroll
            for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
                u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
                if (in_row && group_base + slot < planes.lanes_per_row && row < gathered_count)
                    accumulators[slot] = kit_::accumulate_dot(loaded[slot], query_value, accumulators[slot]);
            }
        }
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            typename kit_::dot_t const dot = inner_cuda_reduce(kit_::partial_to_dot(accumulators[slot]),
                                                               planes.lanes_per_row);
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (leader && group_base + slot < planes.lanes_per_row && row < gathered_count) {
                inner_distance_t const distance = kit_::distance(dot, query_state, batch_norms[slot]);
                gathered_scored[row] = inner_cuda_scored_node_t::make(distance, gathered_ids[first_row + row]);
            }
        }
    }
    __syncwarp();
    return lane < static_cast<int>(gathered_count) ? gathered_scored[lane] : inner_cuda_scored_node_t::sentinel();
}

/** @brief Scores one gathered batch against the query and folds it into the pool,
 *      dropping the entries the pool already holds. */
template <inner_cuda_kit kit_, u32_t resident_blocks_ = inner_cuda_resident_blocks_per_sm_k, typename frontier_>
__device__ __forceinline__ void inner_cuda_score_into_frontier( //
    inner_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, u32_t const (&gathered_ids)[inner_cuda_cache_rows_k],
    inner_cuda_scored_node_t (&gathered_scored)[32], u32_t gathered_count, inner_cuda_warp_context<frontier_>& context,
    int lane) noexcept {
    context.telemetry.scored += gathered_count;
    inner_cuda_scored_node_t const staged = inner_cuda_score_batch<kit_, resident_blocks_>( //
        planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, lane);
    // No dedup: the exact visited set upstream guarantees every id stages once.
    context.frontier.push_batch(staged, lane);
}

/**
 *  @brief Samples one 32-wide batch of distinct, published landmarks below @p sampling_bound
 *      into @p gathered_ids - uniform draws gated on the slot-0 publication word, probing
 *      forward on unpublished hits so out-of-order publication only costs retries.
 *  @return The number of staged landmarks; zero when the visited set saturated the bound.
 */
template <inner_cuda_kit kit_>
__device__ __forceinline__ u32_t inner_cuda_sample_landmarks( //
    inner_cuda_planes_view<kit_> const& planes, inner_node_id_t sampling_bound, inner_cuda_visited_set_t visited,
    splitmix_u32_t& random_generator, u32_t (&gathered_ids)[inner_cuda_cache_rows_k], int lane) noexcept {

    inner_node_id_t draw = random_generator.bounded(sampling_bound);
    bool published = false;
    for (int attempt = 0; attempt != 4 && !published; ++attempt) {
        inner_cuda_neighbors_ref_t const row {planes.neighbors_for(draw), planes.connectivity};
        published = row.peek_first() != inner_node_id_unknown_k;
        if (!published) draw = draw + 1 == sampling_bound ? 0 : draw + 1;
    }
    bool const fresh = visited.test_and_set(draw, lane, published);
    unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
    u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
    if (fresh) gathered_ids[ordinal] = draw;
    __syncwarp();
    return __popc(fresh_mask);
}

/**
 *  @brief Greedy best-first NSW walk from random published landmarks toward the query -
 *      one warp, one traversal. Read-only on the graph: takes no locks, drops transiently
 *      locked candidates after a bounded retry, and leaves the nearest published
 *      candidates ranked in @c context.frontier.
 */
template <inner_cuda_kit kit_, u32_t resident_blocks_ = inner_cuda_resident_blocks_per_sm_k, typename frontier_>
__device__ __forceinline__ void inner_cuda_search_around( //
    inner_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, inner_node_id_t sampling_bound, inner_cuda_visited_set_t visited,
    u32_t (&gathered_ids)[inner_cuda_cache_rows_k], inner_cuda_scored_node_t (&gathered_scored)[32],
    inner_cuda_warp_context<frontier_>& context, int lane) noexcept {

    constexpr u32_t restarts_cap_k = 8;
    frontier_& frontier = context.frontier;
    inner_cuda_telemetry_t& telemetry = context.telemetry;
    u32_t const hops_cap = frontier.hops_budget();
    frontier.reset(lane);

    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = inner_cuda_sample_landmarks<kit_>( //
                planes, sampling_bound, visited, context.random_generator, gathered_ids, lane);
            if (gathered_count)
                inner_cuda_score_into_frontier<kit_, resident_blocks_>( //
                    planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
            seeding = false;
        }
        else {
            inner_cuda_scored_node_t const candidate = frontier.pop_nearest(lane);
            if (candidate == inner_cuda_scored_node_t::sentinel()) {
                // Starvation restart: a drained frontier before an exhausted walk means
                // nothing reachable from its seeds remains - re-seed and continue.
                if (frontier.exhausted() || telemetry.restarts >= restarts_cap_k || telemetry.hops >= hops_cap) break;
                seeding = true;
                ++telemetry.restarts;
                continue;
            }
            // Once the nearest remaining candidate lies beyond the frontier's horizon,
            // nothing deeper can improve the result set.
            if (frontier.beyond_radius(candidate)) break;
            if (telemetry.hops >= hops_cap) break;
            ++telemetry.hops;

            // Read the candidate's neighbor row; a transiently locked row is retried
            // briefly, then dropped rather than convoying behind the re-linker. Only the
            // leader lane polls the racing lock word: per-lane polls can observe different
            // values of a concurrent unlock, and a value-dependent `continue` would fork
            // the warp across loop iterations - fatal for every collective downstream.
            inner_cuda_neighbors_ref_t const row {planes.neighbors_for(candidate.id()), planes.connectivity};
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
            // The first warp-width of slots - the whole row when the connectivity fits one
            // warp. Keeping this sweep loop-free keeps the hot traversal path compact.
            inner_node_id_t const neighbor_id = lane == 0                                   ? first_value
                                                : static_cast<u32_t>(lane) < row.capacity() ? row.load_slot(lane)
                                                                                            : inner_node_id_unknown_k;
            bool const usable = neighbor_id != inner_node_id_unknown_k && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(neighbor_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) gathered_ids[ordinal] = neighbor_id;
            __syncwarp();
            u32_t const gathered_count = __popc(fresh_mask);
            if (gathered_count)
                inner_cuda_score_into_frontier<kit_, resident_blocks_>( //
                    planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
            // Rows wider than a warp continue a warp-width of slots at a time; the bound is
            // warp-uniform, so every collective executes at one program point on every lane.
            for (u32_t slot_base = 32; slot_base < planes.connectivity; slot_base += 32) {
                u32_t const slot = slot_base + lane;
                inner_node_id_t const wide_id = slot < planes.connectivity ? row.load_slot(slot)
                                                                           : inner_node_id_unknown_k;
                bool const wide_usable = wide_id != inner_node_id_unknown_k && wide_id != candidate.id();
                bool const wide_fresh = visited.test_and_set(wide_id, lane, wide_usable);
                unsigned const wide_mask = __ballot_sync(0xFFFFFFFFu, wide_fresh);
                u32_t const wide_ordinal = __popc(wide_mask & ((1u << lane) - 1));
                if (wide_fresh) gathered_ids[wide_ordinal] = wide_id;
                __syncwarp();
                u32_t const wide_count = __popc(wide_mask);
                if (wide_count)
                    inner_cuda_score_into_frontier<kit_, resident_blocks_>( //
                        planes, query_span, query_state, gathered_ids, gathered_scored, wide_count, context, lane);
            }
        }
    }
}

/** @brief Runtime-strided view over the warp's shared vector cache - rows of
 *      @c row_fragments_ fragments each, aliasing the scratch slice. */
template <typename fragment_type_>
struct inner_cuda_vector_cache {
    fragment_type_* data_;
    u32_t row_fragments_;

    __device__ __forceinline__ fragment_type_* row(u32_t row_index) const noexcept {
        return data_ + static_cast<std::size_t>(row_index) * row_fragments_;
    }
    __device__ __forceinline__ fragment_type_& at(u32_t row_index, u32_t fragment_index) const noexcept {
        return row(row_index)[fragment_index];
    }
};

/**
 *  @brief Warp-cooperative ascending buffer of scored nodes over a shared-memory span -
 *      the device twin of the CPU's sorted buffer. Chunks arrive pre-sorted from the
 *      32-wide bitonic network and merge in place: every lane ranks its incoming entry
 *      against the whole span, existing entries count the incoming values ahead of them,
 *      and one barrier separates the staged reads from the scattered writes. Entries
 *      ranked past @c capacity_ are dropped - the insert-with-limit contract.
 */
struct inner_cuda_sorted_buffer {
    inner_cuda_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_;

    __device__ __forceinline__ inner_cuda_scored_node_t at(u32_t rank) const noexcept { return entries_[rank]; }

    __device__ __forceinline__ void insert_batch(inner_cuda_scored_node_t value, u32_t count, int lane) noexcept {
        constexpr u32_t span_batches_k = divide_round_up(inner_cuda_cache_rows_k, 32u);
        bool const incoming = static_cast<u32_t>(lane) < count;
        u32_t incoming_rank = incoming ? static_cast<u32_t>(lane) : 0;
        for (u32_t i = 0; i != size_; ++i) // Warp-uniform bound: every lane walks the span.
            incoming_rank += incoming && entries_[i].distance_bits() <= value.distance_bits();
        // Existing entries shift by the number of incoming values strictly ahead of them;
        // ties resolve the other way than above, so no two final positions collide.
        inner_cuda_scored_node_t moved[span_batches_k];
        u32_t moved_to[span_batches_k];
        u32_t batch = 0;
        for (u32_t base = 0; base < size_; base += 32, ++batch) {
            u32_t const index = base + lane;
            bool const active = index < size_;
            moved[batch] = active ? entries_[index] : inner_cuda_scored_node_t::sentinel();
            u32_t shift = 0;
            for (u32_t source = 0; source != count; ++source) // Warp-uniform: shuffles inside.
                shift += active &&
                         value.shuffle_from(static_cast<int>(source)).distance_bits() < moved[batch].distance_bits();
            moved_to[batch] = index + shift;
        }
        __syncwarp(); // Every displaced entry is staged in registers before any write lands.
        batch = 0;
        for (u32_t base = 0; base < size_; base += 32, ++batch)
            if (base + lane < size_ && moved_to[batch] < capacity_) entries_[moved_to[batch]] = moved[batch];
        if (incoming && incoming_rank < capacity_) entries_[incoming_rank] = value;
        __syncwarp();
        size_ = min_of(size_ + count, capacity_);
    }
};

/** @brief Shared-memory spans backing one warp's link re-selection, carved after the
 *      vector cache: the selected target ids, the ascending per-link scores, and the
 *      refined cache-row indices. */
struct inner_cuda_relink_scratch {
    u32_t* selected_ids;
    inner_cuda_scored_node_t* link_entries;
    u32_t* refined_rows;

    /** @brief Carves the three spans past the vector cache's @p cache_bytes: the ascending
     *      link scores first - the widest alignment - then the selected target ids, then
     *      the refined cache-row indices. The layout rule lives here, with the type. */
    static __device__ __forceinline__ inner_cuda_relink_scratch carve(std::byte* scratch, u32_t connectivity,
                                                                      std::size_t cache_bytes) noexcept {
        u32_t const slots = min_of(connectivity, static_cast<u32_t>(inner_cuda_cache_rows_k));
        std::byte* const base = scratch + cache_bytes;
        return {reinterpret_cast<u32_t*>(base + sizeof(inner_cuda_scored_node_t) * slots),
                reinterpret_cast<inner_cuda_scored_node_t*>(base),
                reinterpret_cast<u32_t*>(base + (sizeof(inner_cuda_scored_node_t) + sizeof(u32_t)) * slots)};
    }
};

/**
 *  @brief Greedy diversity selection over the ascending pool prefix - the classic NSW
 *      pruning heuristic: a candidate is kept only when it is closer to the query than to
 *      every already-kept link, so links spread across directions. Candidate vectors are
 *      cached in the warp's shared scratch; the kept list lives in the shared selected
 *      span, ascending by construction since candidates arrive ascending.
 *  @return The kept-link count; @p selected_ids holds that many node ids on return.
 */
template <inner_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ u32_t inner_cuda_select_diverse( //
    inner_cuda_planes_view<kit_> const& planes, inner_cuda_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[inner_cuda_cache_rows_k], u32_t (&gathered_ids)[inner_cuda_cache_rows_k],
    u32_t* selected_ids, inner_cuda_warp_context<frontier_>& context, int lane) noexcept {

    frontier_& frontier = context.frontier;

    // Stage the ascending prefix ids & cache their vectors and norms.
    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < inner_cuda_cache_rows_k; base += 32) {
        inner_cuda_scored_node_t const entry = base + lane < 64 ? frontier.rank_entry(base + lane)
                                                                : inner_cuda_scored_node_t::sentinel();
        unsigned const live = __ballot_sync(0xFFFFFFFFu, !(entry == inner_cuda_scored_node_t::sentinel()));
        if (!(entry == inner_cuda_scored_node_t::sentinel())) gathered_ids[base + lane] = entry.id();
        candidates_count += __popc(live);
        if (live != 0xFFFFFFFFu) break;
    }
    __syncwarp();
    for (u32_t word = lane; word < candidates_count * planes.row_fragments; word += 32) {
        u32_t const row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, row);
        __pipeline_memcpy_async(&cache.at(row, fragment), planes.vector_span_for(gathered_ids[row]) + fragment,
                                sizeof(typename kit_::fragment_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    if (kit_::has_norms_k)
        for (u32_t row = lane; row < candidates_count; row += 32) cache_norms[row] = planes.norm_for(gathered_ids[row]);
    __syncwarp();

    u32_t selected_count = 0;
    for (u32_t candidate_index = 0; candidate_index < candidates_count && selected_count < planes.connectivity;
         ++candidate_index) {
        f32_t const to_query = frontier.rank_entry(candidate_index).distance();
        bool diverse = true;
        for (u32_t base = 0; base < selected_count && diverse; base += 32) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < selected_count) {
                u32_t const kept_row = selected_ids[base + lane];
                typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    accumulator = kit_::accumulate_dot(cache.at(candidate_index, fragment),
                                                       cache.at(kept_row, fragment), accumulator);
                f32_t const between = kit_::distance(kit_::partial_to_dot(accumulator),
                                                     kit_::make_query_state(cache_norms[candidate_index]),
                                                     cache_norms[kept_row]);
                lane_diverse = between >= to_query;
                ++context.telemetry.scored;
            }
            diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
        }
        if (diverse) {
            if (lane == 0) selected_ids[selected_count] = candidate_index;
            ++selected_count;
            __syncwarp();
        }
    }
    // Rewrite the kept rows as node ids in place - the caller links to nodes, not rows.
    __syncwarp();
    for (u32_t slot = lane; slot < selected_count; slot += 32) selected_ids[slot] = gathered_ids[selected_ids[slot]];
    __syncwarp();
    return selected_count;
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
template <inner_cuda_kit kit_>
__device__ __forceinline__ void inner_cuda_back_link_neighbors( //
    inner_cuda_planes_view<kit_> const& planes, inner_node_id_t new_node_id, typename kit_::norms_t new_norms,
    inner_cuda_relink_scratch relink, u32_t selected_count, inner_cuda_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[inner_cuda_cache_rows_k], u32_t (&gathered_ids)[inner_cuda_cache_rows_k],
    inner_cuda_scored_node_t (&gathered_scored)[32], splitmix_u32_t& random_generator,
    inner_cuda_telemetry_t& telemetry, int lane) noexcept {

    // The cache hosts every existing link, the newcomer on the last row, and the re-linked
    // target itself on the row before it - the scored batch reads the target as its query.
    constexpr u32_t newcomer_row_k = inner_cuda_cache_rows_k - 1;
    constexpr u32_t target_row_k = inner_cuda_cache_rows_k - 2;
    // Those two reserved rows also bound how many links one warp can re-select in shared
    // memory; wider rows keep sentinel tails until the arena lifts the cache ceiling.
    u32_t const connectivity = min_of(planes.connectivity, inner_cuda_cache_rows_k - 2u);

    for (u32_t link = 0; link < selected_count; ++link) {
        inner_node_id_t const target = relink.selected_ids[link];
        inner_cuda_neighbors_ref_t const row {planes.neighbors_for(target), connectivity};
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

        // Read the whole locked row a warp-width at a time, remembering the first empty
        // slot; the same sweep stages every live link id for the potential re-selection.
        u32_t empty_slot = connectivity;
        for (u32_t base = 0; base < connectivity; base += 32) {
            u32_t const slot = base + lane;
            inner_node_id_t const slot_value = slot == 0             ? first_value
                                               : slot < connectivity ? row.load_slot(slot)
                                                                     : inner_node_id_unknown_k;
            if (slot < connectivity) gathered_ids[slot] = slot_value;
            unsigned const empty_mask = __ballot_sync(0xFFFFFFFFu,
                                                      slot < connectivity && slot_value == inner_node_id_unknown_k);
            if (empty_mask && empty_slot == connectivity) empty_slot = base + __ffs(empty_mask) - 1;
        }
        __syncwarp();
        if (empty_slot != connectivity) { // Fast path: append into that slot, unlock unchanged.
            if (lane == 0) row.store_slot(empty_slot, new_node_id);
            __syncwarp();
            if (lane == 0) row.unlock(first_value);
            continue;
        }

        // Full list: score every link & the newcomer against the target itself, then walk
        // all of them ascending, re-running the diversity heuristic, and rewrite the row.
        ++telemetry.full_lists;
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            cache.at(target_row_k, word) = planes.vector_for(target, word);
        __syncwarp();
        typename kit_::query_state_t const target_state = kit_::make_query_state(planes.norm_for(target));

        // The newcomer's distance to the target, from its freshly-owned vector row.
        typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            accumulator = kit_::accumulate_dot(planes.vector_for(new_node_id, word), cache.at(target_row_k, word),
                                               accumulator);
        typename kit_::dot_t const self_dot = inner_cuda_reduce(kit_::partial_to_dot(accumulator), 32u);
        f32_t const newcomer_distance = kit_::distance(__shfl_sync(0xFFFFFFFFu, self_dot, 0), target_state, new_norms);
        telemetry.scored += connectivity + 1;

        // Score the links a warp-width at a time, pre-sort each batch with the bitonic
        // network, and merge every batch into the ascending sorted buffer - the ported
        // insert-with-limit, so any link count the cache can hold walks in order.
        inner_cuda_sorted_buffer links_sorted {relink.link_entries, 0, connectivity};
        for (u32_t base = 0; base < connectivity; base += 32) {
            u32_t const batch_count = min_of(connectivity - base, 32u);
            inner_cuda_scored_node_t const scored = inner_cuda_score_batch<kit_>( //
                planes, cache.row(target_row_k), target_state, gathered_ids, gathered_scored, batch_count, lane, base);
            inner_cuda_scored_node_t const sorted_batch = inner_cuda_sort32(
                static_cast<u32_t>(lane) < batch_count
                    ? inner_cuda_scored_node_t::make(scored.distance(), base + lane) // Id = slot index.
                    : inner_cuda_scored_node_t::sentinel(),
                lane);
            links_sorted.insert_batch(sorted_batch, batch_count, lane);
        }

        // Cache the links' vectors (their norms ride the staging) plus the newcomer at the
        // tail row, then merge-walk all `connectivity + 1` ascending by distance-to-target.
        // The fill rides the async-copy pipeline: every fragment is a global-to-shared
        // `cp.async` that holds no registers, so the runtime word count costs no pressure.
        u32_t const link_words = connectivity * planes.row_fragments;
        for (u32_t word = lane; word < link_words; word += 32) {
            u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
            __pipeline_memcpy_async(&cache.at(cache_row, fragment),
                                    planes.vector_span_for(gathered_ids[cache_row]) + fragment,
                                    sizeof(typename kit_::fragment_t));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            cache.at(newcomer_row_k, word) = planes.vector_for(new_node_id, word);
        for (u32_t cache_row = lane; cache_row < connectivity; cache_row += 32)
            cache_norms[cache_row] = planes.norm_for(gathered_ids[cache_row]);
        if (lane == 0) cache_norms[newcomer_row_k] = new_norms;
        __syncwarp();

        u32_t existing_index = 0;
        bool newcomer_pending = true;
        u32_t refined_count = 0;
        for (u32_t merge_index = 0; merge_index != connectivity + 1 && refined_count != connectivity; ++merge_index) {
            inner_cuda_scored_node_t const next_link = existing_index < links_sorted.size_
                                                           ? links_sorted.at(existing_index)
                                                           : inner_cuda_scored_node_t::sentinel();
            bool const take_newcomer = newcomer_pending && newcomer_distance < next_link.distance();
            f32_t const candidate_distance = take_newcomer ? newcomer_distance : next_link.distance();
            u32_t const candidate_row = take_newcomer ? newcomer_row_k : next_link.id();
            if (!take_newcomer && existing_index >= links_sorted.size_) break;
            if (take_newcomer) newcomer_pending = false;
            else ++existing_index;

            bool diverse = true;
            for (u32_t base = 0; base < refined_count && diverse; base += 32) { // Warp-uniform bounds.
                bool lane_diverse = true;
                if (base + lane < refined_count) {
                    u32_t const kept_row = relink.refined_rows[base + lane];
                    typename kit_::accumulator_t pair_accumulator = kit_::zero_accumulator();
                    for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                        pair_accumulator = kit_::accumulate_dot(cache.at(candidate_row, fragment),
                                                                cache.at(kept_row, fragment), pair_accumulator);
                    f32_t const between = kit_::distance(kit_::partial_to_dot(pair_accumulator),
                                                         kit_::make_query_state(cache_norms[candidate_row]),
                                                         cache_norms[kept_row]);
                    lane_diverse = between >= candidate_distance;
                    ++telemetry.scored;
                }
                diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
            }
            if (diverse) {
                if (lane == 0) relink.refined_rows[refined_count] = candidate_row;
                ++refined_count;
                __syncwarp();
            }
        }

        // Rewrite the row: tail slots relaxed, shrink sentinel-terminated, head unlocks.
        __syncwarp();
        for (u32_t slot = lane; slot < connectivity; slot += 32) {
            if (slot == 0) continue;
            inner_node_id_t linked = inner_node_id_unknown_k;
            if (slot < refined_count) {
                u32_t const kept_row = relink.refined_rows[slot];
                linked = kept_row == newcomer_row_k ? new_node_id : gathered_ids[kept_row];
            }
            row.store_slot(slot, linked);
        }
        __syncwarp();
        if (lane == 0) {
            u32_t const head_row = relink.refined_rows[0];
            row.unlock(head_row == newcomer_row_k ? new_node_id : gathered_ids[head_row]);
        }
    }
}

/**
 *  @brief Layout of one warp's slice of the global traversal arena - the single owner of
 *      that arithmetic: the host computes the shape, the views carry it, and the kernels
 *      ask it for typed spans. Every slice holds the exact visited set; wide walks add
 *      the frontier's spill tail after it.
 */
struct inner_cuda_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    constexpr std::size_t slice_bytes() const noexcept {
        return static_cast<std::size_t>(visited_slots) * sizeof(u64_t) +
               static_cast<std::size_t>(tail_entries) * sizeof(inner_cuda_scored_node_t);
    }
    __device__ __forceinline__ u64_t* visited_span(std::byte* base, std::size_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(base + warp_index * slice_bytes());
    }
    __device__ __forceinline__ inner_cuda_scored_node_t* tail_span(std::byte* base,
                                                                   std::size_t warp_index) const noexcept {
        return reinterpret_cast<inner_cuda_scored_node_t*>( //
            base + warp_index * slice_bytes() + static_cast<std::size_t>(visited_slots) * sizeof(u64_t));
    }
};

/** @brief One warp's launch-uniform slice - its coordinates, its dynamic-shared scratch,
 *      and its arena visited span: the shared prologue of every traversal kernel, all
 *      affine arithmetic compiling into the exact SASS it replaces. */
struct inner_cuda_warp_slice_t {
    int lane, warp;
    std::size_t warp_index;
    std::byte* scratch;
    u64_t* visited_slots;
};

template <typename view_>
__device__ __forceinline__ inner_cuda_warp_slice_t inner_cuda_carve_slice(view_ const& view,
                                                                          std::byte* dynamic_scratch) noexcept {
    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    std::size_t const warp_index = blockIdx.x * static_cast<std::size_t>(inner_cuda_warps_per_block_k) + warp;
    std::byte* const scratch = dynamic_scratch + static_cast<std::size_t>(warp) * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch, view.arena_shape.visited_span(view.arena, warp_index)};
}

/** @brief Geometry & plane pointers one insertion launch operates on. Every plane row a
 *      dispensable new_node_id can touch is mapped and sentinel-filled ahead of the launch, so
 *      growth hazards cannot reach the kernel. */
template <inner_cuda_kit kit_>
struct inner_cuda_build_view {
    inner_cuda_planes_view<kit_> planes;
    /** @brief Incoming vectors, @b [rows,row_fragments] in dispensed-id order, device-resident. */
    typename kit_::fragment_t const* staging;
    /** @brief Ids strictly below this bound may be claimed & inserted by this launch. */
    u32_t end_id;
    /** @brief Bytes of the per-warp dynamic-shared slice: the query span, grown to fit
     *      the candidate cache and the re-selection spans. */
    u32_t scratch_bytes_per_warp;
    /** @brief The batch's base id: staging row 0 lands there, and the kernel's per-launch
     *      cursor hands out offsets within @b [first_id,end_id). */
    inner_node_id_t first_id;
    /** @brief The traversal arena and each warp's slice of it - the exact visited set. */
    std::byte* arena;
    inner_cuda_arena_shape_t arena_shape;
    /** @brief The context's monotone epoch floor at launch: batch-local epochs ride above
     *      it, so stale arena slots from earlier launches always compare older and the
     *      host never re-zeroes a warm arena. */
    u32_t epoch_base;
    /** @brief The index's shared control block - the admission ramp's cross-batch signal. */
    inner_cuda_control_t* control;
    inner_cuda_telemetry_block_t* telemetry;
    /** @brief This launch's work-queue head: offsets within the batch, zeroed per launch. */
    u32_t* batch_cursor;
};

/** @brief Inputs & outputs of one search launch: a batch of queries in, the nearest ids
 *      and - when requested - their distances out, both ascending per query. */
template <inner_cuda_kit kit_>
struct inner_cuda_search_view {
    inner_cuda_planes_view<kit_> planes;
    typename kit_::fragment_t const* queries; // @b [queries,row_fragments]
    inner_node_id_t* result_ids;              // @b [queries,count]
    inner_distance_t* result_distances;       // @b [queries,count]; null skips the writes
    u32_t queries_count;
    u32_t count;
    /** @brief Published-node ceiling for landmark sampling - the host's snapshot of
     *      @c size() at launch; the walk itself gates on slot-0 publication words. */
    u32_t sampling_bound;
    /** @brief Bytes of the per-warp dynamic-shared slice - the query span. */
    u32_t scratch_bytes_per_warp;
    /** @brief Requested expansion; past the register ladder it becomes the wide walk's
     *      expansion budget and sizes the arena's spill tail. */
    u32_t expansion;
    /** @brief The traversal arena and each warp's slice of it - the exact visited set,
     *      then the wide frontier's spill tail. */
    std::byte* arena;
    inner_cuda_arena_shape_t arena_shape;
    /** @brief The context's monotone epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    inner_cuda_telemetry_block_t* telemetry;
};

/**
 *  @brief Inserts the node identified by @p new_node_id: lands the payload & norms, walks the
 *      published graph, selects diverse links, publishes with one release store, then
 *      wires the reverse edges - the whole CPU insertion flow, one warp per node.
 */
template <inner_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void inner_cuda_insert_one( //
    inner_cuda_build_view<kit_> const& view, inner_node_id_t new_node_id, inner_cuda_visited_set_t visited,
    typename kit_::fragment_t* query_span, inner_cuda_vector_cache<typename kit_::fragment_t> cache,
    inner_cuda_relink_scratch relink, typename kit_::norms_t (&cache_norms)[inner_cuda_cache_rows_k],
    u32_t (&gathered_ids)[inner_cuda_cache_rows_k], inner_cuda_scored_node_t (&gathered_scored)[32],
    inner_cuda_warp_context<frontier_>& context, int lane) noexcept {

    inner_cuda_planes_view<kit_> const& planes = view.planes;

    // Land the payload: copy the vector fragments, derive both norms from the owned bytes.
    // The same pass parks the query in its shared span for the traversal to re-read.
    typename kit_::fragment_t const* staging_row =
        view.staging + static_cast<std::size_t>(new_node_id - view.first_id) * planes.row_fragments;
    typename kit_::fragment_t* owned_row = planes.vector_span_for(new_node_id);
    typename kit_::norms_accumulator_t norms_accumulator = kit_::zero_norms_accumulator();
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32) {
        typename kit_::fragment_t const value = staging_row[fragment];
        owned_row[fragment] = value;
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = inner_cuda_reduce_norms(norms_accumulator);
    norms_accumulator = inner_cuda_broadcast_norms(norms_accumulator);
    typename kit_::norms_t const new_norms = kit_::partials_to_norms(norms_accumulator);
    if (lane == 0) planes.norm_ref_for(new_node_id) = new_norms;
    typename kit_::query_state_t const query_state = kit_::make_query_state(new_norms);
    __syncwarp();

    // Walk the published prefix - a virgin index publishes an empty node instead.
    if (new_node_id != 0)
        inner_cuda_search_around<kit_>( //
            planes, query_span, query_state, new_node_id, visited, gathered_ids, gathered_scored, context, lane);
    else context.frontier.reset(lane);

    // Select diverse links & write the own row - no locks: no reader arrives before the
    // slot-0 release below flips it from the sentinel.
    u32_t const selected_count = inner_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered_ids, relink.selected_ids, context, lane);
    inner_cuda_neighbors_ref_t const own_row {planes.neighbors_for(new_node_id), planes.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += 32)
        if (slot > 0) own_row.store_slot(slot, relink.selected_ids[slot]);
    __syncwarp();
    inner_node_id_t const first_link = selected_count ? relink.selected_ids[0] : new_node_id;
    if (lane == 0) {
        own_row.unlock(first_link);                    // A self-loop encodes published-empty.
        atomicAdd(&view.control->published_count, 1u); // Fire-and-forget reduction.
    }

    // Only now - published, no locks held - wire the reverse edges.
    inner_cuda_back_link_neighbors<kit_>( //
        planes, new_node_id, new_norms, relink, selected_count, cache, cache_norms, gathered_ids, gathered_scored,
        context.random_generator, context.telemetry, lane);
}

/**
 *
 *  @section inner_cuda_index_concurrency Concurrency & Publication
 *
 *  One warp inserts one node, and no two warps ever synchronize beyond per-neighbor
 *  slot-0 locks: there is no watermark, no publication order, and no compare-and-swap.
 *  Id dispensing is batch-local: the host pre-claims a dense range under its growth
 *  latch, and each launch's per-context cursor hands out offsets within it - a warp's
 *  fetch_add result @b is its node id, sibling batches can never cross-claim, and an
 *  aborted launch's over-dispensed offsets die with its cursor, no re-crediting anywhere.
 *  A node is usable exactly when its slot 0 acquire-loads non-sentinel; a self-loop id
 *  there means "published, zero neighbors". Landmark sampling draws below the claimed
 *  new_node_id and probes past unpublished hits, so out-of-order publication costs retries,
 *  never correctness. The admission ramp - new_node_id @b t waits until
 *  t < published + max(32, published) - bounds early-graph lock contention and keeps
 *  the sampler's hit rate above one half; past bootstrap, warps never wait on each other.
 *  Every give-up path still publishes (a self-loop tombstone), so dispensed ids never
 *  stall anything, and a post-mortem sweep can tombstone the stragglers of a killed
 *  launch precisely because publication order is free.
 */
template <inner_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(inner_cuda_warps_per_block_k * 32, inner_cuda_resident_blocks_per_sm_k)
    inner_cuda_insert_kernel(inner_cuda_build_view<kit_> view) {
    // One launch-sized dynamic slice per warp hosts two disjoint lifetimes over the same
    // bytes: the query span while traversing, then the candidate-vector cache while
    // selecting & back-linking - the host sizes the slice to the larger shape. The exact
    // visited set lives in the warp's arena slice, not in shared memory.
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[inner_cuda_warps_per_block_k][inner_cuda_cache_rows_k];
    __shared__ u32_t gathered_ids[inner_cuda_warps_per_block_k][inner_cuda_cache_rows_k];
    __shared__ inner_cuda_scored_node_t gathered_scored[inner_cuda_warps_per_block_k][32];

    inner_cuda_warp_slice_t const slice = inner_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    inner_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    inner_cuda_relink_scratch const relink = inner_cuda_relink_scratch::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            inner_cuda_cache_rows_k);
    while (true) {
        u32_t new_node_id;
        // The work queue is batch-local: the host pre-claims [first_id,end_id) before
        // the launch, so concurrent batches from sibling contexts can never cross-claim,
        // and an aborted launch's over-dispensed offsets die with its cursor.
        if (lane == 0) new_node_id = view.first_id + atomicAdd(view.batch_cursor, 1u);
        new_node_id = __shfl_sync(0xFFFFFFFFu, new_node_id, 0);
        if (new_node_id >= view.end_id) return;

        while (true) { // Admission ramp - the only bootstrap throttle.
            u32_t published;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(published) : "l"(&view.control->published_count));
            u32_t const window = max_of(published, admission_ramp_floor_k);
            if (new_node_id < published + window) break;
            __nanosleep(2048);
        }

        inner_cuda_warp_context<inner_cuda_register_frontier<pool_slots_>> context;
        context.random_generator = splitmix_u32_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(new_node_id) * 33 + lane)};
        inner_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + (new_node_id - view.first_id) + 1};
        inner_cuda_insert_one<kit_>( //
            view, new_node_id, visited, query_span, cache, relink, cache_norms[warp], gathered_ids[warp],
            gathered_scored[warp], context, lane);
        context.telemetry.flush<true>(view.telemetry, lane);
    }
}

/**
 *  @brief One query's full search - stages the query into its shared span, walks the
 *      graph, folds the ranks, and stores the nearest @c count ids (and distances)
 *      ascending. The kernels below are thin drivers: they differ only in the frontier
 *      they hand in. Safe against concurrent insertions: the walk gates every row on its
 *      slot-0 word and never writes to the graph.
 */
template <inner_cuda_kit kit_, u32_t resident_blocks_, typename frontier_>
__device__ __forceinline__ void inner_cuda_search_one( //
    inner_cuda_search_view<kit_> const& view, u32_t query_index, typename kit_::fragment_t* query_span,
    inner_cuda_visited_set_t visited, frontier_ frontier, u32_t (&gathered_ids)[inner_cuda_cache_rows_k],
    inner_cuda_scored_node_t (&gathered_scored)[32], int lane) noexcept {

    typename kit_::fragment_t const* query_row = view.queries +
                                                 static_cast<std::size_t>(query_index) * view.planes.row_fragments;
    typename kit_::norms_accumulator_t norms_accumulator = kit_::zero_norms_accumulator();
    for (u32_t fragment = lane; fragment < view.planes.row_fragments; fragment += 32) {
        typename kit_::fragment_t const value = query_row[fragment];
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = inner_cuda_reduce_norms(norms_accumulator);
    norms_accumulator = inner_cuda_broadcast_norms(norms_accumulator);
    typename kit_::norms_t const query_norms = kit_::partials_to_norms(norms_accumulator);
    typename kit_::query_state_t const query_state = kit_::make_query_state(query_norms);
    __syncwarp();

    inner_cuda_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = splitmix_u32_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(query_index) * 33 + lane)};
    inner_cuda_search_around<kit_, resident_blocks_>( //
        view.planes, query_span, query_state, view.sampling_bound, visited, gathered_ids, gathered_scored, context,
        lane);
    // The rank extraction shuffles - every lane participates, only in-range ones store.
    // Ranks past the live prefix surface as sentinel ids, marking absent matches.
    context.frontier.finalize_ranks(lane);
    for (u32_t rank_base = 0; rank_base < view.count; rank_base += 32) {
        u32_t const rank = rank_base + lane;
        inner_cuda_scored_node_t const ranked_entry = context.frontier.rank_entry(rank);
        if (rank < view.count) {
            std::size_t const slot = static_cast<std::size_t>(query_index) * view.count + rank;
            view.result_ids[slot] = ranked_entry.id();
            if (view.result_distances) view.result_distances[slot] = kit_::convert_to_distance(ranked_entry.distance());
        }
    }
    context.telemetry.flush<false>(view.telemetry, lane);
}

/** @brief One warp per query over the register frontier - expansions within the ladder. */
template <inner_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(inner_cuda_warps_per_block_k * 32, pool_slots_ <= 4 ? 4 : 3)
    inner_cuda_search_kernel(inner_cuda_search_view<kit_> view) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ u32_t gathered_ids[inner_cuda_warps_per_block_k][inner_cuda_cache_rows_k];
    __shared__ inner_cuda_scored_node_t gathered_scored[inner_cuda_warps_per_block_k][32];

    inner_cuda_warp_slice_t const slice = inner_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * inner_cuda_warps_per_block_k) {
        inner_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + query_index + 1};
        inner_cuda_search_one<kit_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, query_index, query_span, visited, inner_cuda_register_frontier<pool_slots_> {}, gathered_ids[warp],
            gathered_scored[warp], lane);
    }
}

/** @brief One warp per query past the register ladder's width: the same walk over the
 *      arena frontier, whose expansion budget @b is the requested expansion factor -
 *      unbounded by any register shape. Results cover the frontier's head, the exact
 *      nearest 64, so the facade admits this path only for counts within it. */
template <inner_cuda_kit kit_>
__global__ void __launch_bounds__(inner_cuda_warps_per_block_k * 32, 4)
    inner_cuda_search_wide_kernel(inner_cuda_search_view<kit_> view) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ u32_t gathered_ids[inner_cuda_warps_per_block_k][inner_cuda_cache_rows_k];
    __shared__ inner_cuda_scored_node_t gathered_scored[inner_cuda_warps_per_block_k][32];

    inner_cuda_warp_slice_t const slice = inner_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    inner_cuda_arena_frontier frontier;
    frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * inner_cuda_warps_per_block_k) {
        inner_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + query_index + 1};
        inner_cuda_search_one<kit_, 4u>(view, query_index, query_span, visited, frontier, gathered_ids[warp],
                                        gathered_scored[warp], lane);
    }
}

/** @brief Rebuilds the norms plane from the vector plane, one warp per row - norms are
 *      derived state and never serialized, so every load re-computes them where the
 *      vectors already are, instead of shipping redundant bytes. */
template <inner_cuda_kit kit_>
__global__ void inner_cuda_derive_norms_kernel(inner_cuda_planes_view<kit_> planes, u32_t rows_count) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        typename kit_::norms_accumulator_t accumulator = kit_::zero_norms_accumulator();
        for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
            accumulator = kit_::accumulate_norms(planes.vector_for(static_cast<inner_node_id_t>(row), fragment),
                                                 accumulator);
        accumulator = inner_cuda_reduce_norms(accumulator);
        if (lane == 0) planes.norm_ref_for(static_cast<inner_node_id_t>(row)) = kit_::partials_to_norms(accumulator);
    }
}

/** @brief Post-drain structural audit of the neighbor plane: every row published, every id
 *      in range, self-loop rows childless, no duplicate edges within a row. One warp scans
 *      one row, a warp-width of slots at a time; each bad row bumps @p violations once. */
static __global__ void inner_cuda_invariant_kernel(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                   u64_t* violations) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        u32_t const first = neighbors[row * connectivity];
        bool bad = first == inner_node_id_unknown_k;
        for (u32_t base = 0; base < connectivity; base += 32) { // Warp-uniform trip count.
            u32_t const slot = base + lane;
            u32_t const value = slot < connectivity ? neighbors[row * connectivity + slot] : inner_node_id_unknown_k;
            bad |= value != inner_node_id_unknown_k && value >= rows_count;
            bad |= first == static_cast<u32_t>(row) && slot > 0 && value != inner_node_id_unknown_k;
            // Duplicate-edge scan: rotate every not-earlier batch past this lane's value.
            for (u32_t other_base = base; other_base < connectivity; other_base += 32) {
                u32_t const other_own = other_base + lane;
                u32_t const other_value = other_base == base         ? value
                                          : other_own < connectivity ? neighbors[row * connectivity + other_own]
                                                                     : inner_node_id_unknown_k;
                for (int shift = 0; shift < 32; ++shift) {
                    int const source = (lane + shift) & 31;
                    u32_t const other = __shfl_sync(0xFFFFFFFFu, other_value, source);
                    u32_t const other_slot = other_base + static_cast<u32_t>(source);
                    bad |= value != inner_node_id_unknown_k && other == value && other_slot > slot &&
                           other_slot < connectivity;
                }
            }
        }
        if (__ballot_sync(0xFFFFFFFFu, bad) && lane == 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(violations), 1ull);
    }
}

#endif // defined(__CUDACC__)

#if defined(__CUDACC__)

/**
 *  @brief One agent's device-side scratch for driving a CUDA index - the GPU twin of the
 *      CPU's caller-owned @c inner_traversal_context. The CPU context is a thread's
 *      NUMA-local scratch; this is a @b stream's pool-local scratch: it owns the ordering
 *      stream, the traversal arena, the batch staging, and the telemetry sink, all drawn
 *      from one allocator and grown once to the widest launch, so the index's verbs issue
 *      launches and never per-call allocations.
 *
 *  Cardinality inverts against the CPU: a context is a concurrent submission lane, two to
 *  four per index - one per stream - never one per task. Two contexts on one index share
 *  the read-only planes and the published counter, and nothing else: their arenas, staging,
 *  and telemetry are disjoint by construction, so an insert stream and a search stream
 *  never alias scratch.
 */
template <nk_dtype_t, nk_kernel_kind_t, cuda_arch_t, typename, typename, typename>
struct inner_cuda_index;

template <cuda_allocator allocator_type_ = cuda_device_allocator<std::byte>>
struct inner_cuda_traversal_context {
    using allocator_t = allocator_type_;

    /** @brief The single-query verb lands this many nearest matches at most - the arena
     *      frontier's exact head width, and far past any interactive top-k. */
    inline static constexpr u32_t results_capacity_k = 64;

    /** @brief The single-query verb's landing: ascending @c inner_scored_node_t matches,
     *      the CPU context's @c results in miniature - inline storage, no allocation. */
    struct results_t {
        inner_scored_node_t entries_[results_capacity_k];
        std::size_t count_ = 0;

        inner_scored_node_t* data() noexcept { return entries_; }
        inner_scored_node_t const* data() const noexcept { return entries_; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** @brief Creates the context's ordering stream, binds a working allocator to it on
     *      @p allocator's device - the argument carries placement only - and allocates the
     *      fixed-size telemetry sink; the sized buffers wait for the first verb, so
     *      construction is cheap and @c noexcept. */
    inner_cuda_traversal_context(allocator_t allocator = {}) noexcept
        : allocator_ {stream_, allocator.device()},
          telemetry_ {cuda_make_unique(telemetry_allocator_t {allocator_}, 1)},
          batch_cursor_ {cuda_make_unique(cursor_allocator_t {allocator_}, 1)} {
        if (telemetry_) cudaMemsetAsync(telemetry_.get(), 0, sizeof(inner_cuda_telemetry_block_t), stream_);
    }
    inner_cuda_traversal_context(inner_cuda_traversal_context const&) = delete;
    inner_cuda_traversal_context& operator=(inner_cuda_traversal_context const&) = delete;
    /** @brief Every owned resource is a stream handle or a @c cuda_unique_ptr, so moves are
     *      member-wise steals and destruction is declaration-driven: the owning pointers -
     *      declared after @c stream_ - free onto the still-live stream, then the stream
     *      wrapper drains and destroys it. */
    inner_cuda_traversal_context(inner_cuda_traversal_context&&) noexcept = default;
    ~inner_cuda_traversal_context() noexcept = default;

    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return allocator_.device(); }

    /** @brief Blocking host copy of this context's telemetry sink. */
    inner_cuda_telemetry_block_t telemetry_snapshot() const noexcept {
        inner_cuda_telemetry_block_t snapshot = {};
        cudaMemcpyAsync(&snapshot, telemetry_.get(), sizeof(snapshot), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        return snapshot;
    }

  private:
    /** @brief Only the index's verbs reach the raw scratch - callers observe a context
     *      through @c results and the snapshots, never through its buffers. */
    template <nk_dtype_t, nk_kernel_kind_t, cuda_arch_t, typename, typename, typename>
    friend struct inner_cuda_index;

    using telemetry_allocator_t = typename allocator_t::template rebind<inner_cuda_telemetry_block_t>::other;
    using cursor_allocator_t = typename allocator_t::template rebind<u32_t>::other;
    using bytes_ptr_t = cuda_unique_ptr<std::byte, allocator_t>;

    cuda_stream_t stream_;
    allocator_t allocator_;
    /** @brief Monotone epoch floor over the arena's visited slots: every launch's epochs
     *      ride above it, so the arena is zeroed only when freshly allocated or when the
     *      32-bit epoch space wraps - never per launch. */
    u64_t epoch_floor_ = 0;
    cuda_unique_ptr<inner_cuda_telemetry_block_t, telemetry_allocator_t> telemetry_;
    cuda_unique_ptr<u32_t, cursor_allocator_t> batch_cursor_;
    bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** @brief Grow-only traversal arena for @p bytes - the exact visited sets and spill
     *      tails of every resident warp; kept across calls, like the CPU context keeps
     *      its larger link scratches. @return The base, or null on exhaustion. */
    std::byte* ensure_arena(std::size_t bytes) noexcept { return ensure_(arena_, arena_bytes_, bytes); }
    /** @brief Grow-only staging for one batch's device transport - an insert's
     *      fragment-padded vectors, or a search's queries & result landing. */
    std::byte* ensure_staging(std::size_t bytes) noexcept { return ensure_(staging_, staging_bytes_, bytes); }

    std::byte* ensure_(bytes_ptr_t& buffer, std::size_t& capacity, std::size_t bytes) noexcept {
        if (bytes <= capacity) return buffer.get();
        buffer = cuda_make_unique(allocator_, bytes);
        capacity = buffer ? bytes : 0;
        return buffer.get();
    }
};

/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  Mirrors the CPU-resident volatile index verb-for-verb - @c reserve, @c insert,
 *  @c search - with two deliberate divergences, each documented where it lives: a batch's
 *  assigned ids are the contiguous new_node_id range (no per-vector id span), and
 *  publication is unordered - see @ref inner_cuda_index_concurrency.
 *
 *  @section inner_cuda_index_growth Growth
 *
 *  The first successful @c reserve() freezes the virtual-address ceiling; physical slabs
 *  map on demand ahead of every launch, and fresh neighbor rows are sentinel-filled
 *  before any new_node_id covering them can be dispensed.
 */
template <                                                                  //
    nk_dtype_t vector_dtype_ = nk_i8_k,                                     //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                    //
    cuda_arch_t arch_ceiling_ = cuda_arch_blackwell_k,                      //
    typename vectors_allocator_type_ = cuda_vmm_allocator<std::byte>,       //
    typename neighbors_allocator_type_ = cuda_vmm_allocator<std::byte>,     //
    typename traversals_allocator_type_ = cuda_device_allocator<std::byte>> //
struct inner_cuda_index {

    using kit_chain_t = inner_cuda_kit_chain<vector_dtype_, kernel_kind_, arch_ceiling_>;
    using kit_t = typename kit_chain_t::kit_t;
    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using scalar_t = typename kit_t::scalar_t;
    using fragment_t = typename kit_t::fragment_t;
    using norms_t = typename kit_t::norms_t;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_view_t = nk::vector_view<scalar_t>;
    using node_id_t = inner_node_id_t;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using distance_t = inner_distance_t;
    using small_size_t = std::uint32_t;
    using header_t = inner_index_header_t;
    using planes_view_t = inner_cuda_planes_view<kit_t>;
    using inner_traversal_context_t = inner_cuda_traversal_context<traversals_allocator_t>;

    /** @brief The strongest @c synchronization_t contract this design honors - purely
     *      documentation, nothing routes on it. One graph, many concurrent agents: batches
     *      claim disjoint dense id ranges under the admission latch, dispense offsets from
     *      per-launch cursors, and the all-RMW publication protocol makes overlapping
     *      searches safe. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    /** @brief Capability labels for the shared reports: the GPU picks no CPU SIMD tier. */
    inline static constexpr nk_capability_t picked_quads_capability_k = nk_cap_serial_k;
    /** @brief The generation row the kit chain selected under this instantiation's ceiling -
     *      the GPU twin of @c picked_quads_capability_k, surfaced in the bench reports. */
    inline static constexpr cuda_arch_t picked_arch_k = kit_chain_t::picked_arch_k;
    inline static constexpr bool has_tiles_k = false;
    inline static constexpr nk_capability_t picked_tiles_capability_k = nk_cap_serial_k;

    inline static constexpr std::size_t max_node_count_k = inner_cuda_scored_node_t::id_capacity_k;
    /** @brief Default visited-set slot count: a runtime knob now, this is only the
     *      heuristic floor - the scratch slice must also fit the candidate cache. */
    inline static constexpr u32_t default_visited_capacity_k = 2048;

    /** @brief The once-frozen geometry word's type, the CPU index's exact packing:
     *      dimensions in the high half, connectivity in the low, zero meaning "not yet
     *      reserved" - so one atomic word answers both "what shape" and "is it frozen". */
    using packed_geometry_id_t = u64_t;
    inline static constexpr packed_geometry_id_t pack_geometry(std::size_t dimensions,
                                                               std::size_t connectivity) noexcept {
        return (static_cast<packed_geometry_id_t>(dimensions) << 32u) | static_cast<packed_geometry_id_t>(connectivity);
    }
    inline static constexpr u32_t geometry_dimensions(packed_geometry_id_t geometry) noexcept {
        return static_cast<u32_t>(geometry >> 32u);
    }
    inline static constexpr u32_t geometry_connectivity(packed_geometry_id_t geometry) noexcept {
        return static_cast<u32_t>(geometry);
    }

    /** @brief Bytes of one packed logical row, counted bit-exactly through NumKong's own
     *      reflection - sub-byte types pack several dimensions per storage value, and
     *      @c bits_per_dimension already knows every packing. */
    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }
    /** @brief Fragments per vector row - derived from the frozen dimensions, like every
     *      row measure below: the geometry word is the only stored source of truth. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), kit_t::fragment_bytes_k);
    }
    /** @brief The vector row's byte length: the dimension count rounded up to whole
     *      fragments, tail bytes kept zero so every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept {
        return static_cast<std::size_t>(row_fragments()) * kit_t::fragment_bytes_k;
    }
    /** @brief Runtime scoring geometry: lanes cooperating on one row (the smallest power
     *      of two covering its fragments, at most a warp), and their steps across it. */
    u32_t lanes_per_row() const noexcept { return min_of(std::bit_ceil(row_fragments()), 32u); }
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), lanes_per_row()); }

    /** @brief Bytes of one warp's dynamic-shared slice on the insertion path: the query
     *      span while traversing, then the candidate cache & re-selection spans. */
    u32_t insert_scratch_bytes_per_warp() const noexcept {
        u32_t const relink_slots = min_of(static_cast<u32_t>(connectivity()), //
                                          static_cast<u32_t>(inner_cuda_cache_rows_k));
        u32_t const relink_bytes = relink_slots * static_cast<u32_t>(sizeof(inner_cuda_scored_node_t) + //
                                                                     2 * sizeof(u32_t));
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * inner_cuda_cache_rows_k;
        return max_of(static_cast<u32_t>(row_bytes()), cache_bytes + relink_bytes);
    }

    /** @brief Which frontier a warp's arena slice serves: the register ladder needs only
     *      the exact visited set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** @brief One warp's arena slice for a walk expanding @p expansion candidates: the
     *      exact visited set - sized well past the expected visit count, so linear
     *      probing stays short - plus the spill tail when @p frontier_kind outgrows
     *      registers. */
    static inner_cuda_arena_shape_t arena_shape_for(u32_t expansion, frontier_kind_t frontier_kind) noexcept {
        u32_t const visited_slots = std::bit_ceil(64 * expansion);
        return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)),
                frontier_kind == frontier_kind_t::wide_arena_k ? 32 * expansion : 0};
    }

    /** @brief Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;
    };
    /** @brief Outcome of a @c reserve() call: the capacity & connectivity actually in effect. */
    struct reserve_result_t {
        std::size_t new_capacity = 0;
        connectivity_t connectivity = connectivity_unknown_k;
        status_t status = status_t::success_k;
    };
    /** @brief Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {
        /** @brief Breadth of the construction candidate pool; rounded up to a multiple of 32,
         *      clamped to 512. Larger pools trade build speed for graph quality. */
        small_size_t expansion_factor = 0;
    };
    /** @brief Outcome & telemetry of one @c insert() batch. Ids are the contiguous range
     *      @b [first_id,first_id+count) - the dispensed order of the incoming rows. */
    struct insert_result_t {
        status_t status = status_t::success_k;
        std::size_t new_size = 0;
        inner_node_id_t first_id = inner_node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        /** @brief Reverse edges abandoned after the bounded lock budget - GPU-only telemetry. */
        std::size_t abandoned_back_links = 0;
        /** @brief The CPU splits refine-side distance work out of @c computed_distances;
         *      the GPU folds it all into one counter, so these report zero. */
        std::size_t computed_distances_in_refines = 0;
        std::size_t computed_distances_in_reverse_refines = 0;
        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };
    struct search_config_t {
        /** @brief Breadth of the candidate pool; defaults to twice the connectivity, and is
         *      raised automatically when @c count needs a wider pool. */
        small_size_t expansion_factor = 0;
        /** @brief Matches reported per query, up to the widest supported pool of 512. */
        small_size_t count = 10;
        /** @brief The caller's mutation contract. The GPU walk is insert-safe by protocol
         *      - every row gates on its slot-0 acquire - so all three levels share one
         *      code path and the field only documents intent. */
        synchronization_t synchronization = synchronization_t::concurrent_k;
    };
    struct search_result_t {
        status_t status = status_t::success_k;
        std::size_t count = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };

    /** @brief Mirrors the CPU index's construction: one allocator per plane concern, each
     *      carrying its device placement; the index pins itself to the vectors allocator's
     *      GPU and demands the neighbors allocator agree. Allocates nothing here. */
    inner_cuda_index(vectors_allocator_t const& vectors_allocator = {},
                     neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, vectors_plane_(vectors_allocator), norms_plane_(vectors_allocator),
          neighbors_plane_(neighbors_allocator) {}
    inner_cuda_index(inner_cuda_index const&) = delete;
    inner_cuda_index& operator=(inner_cuda_index const&) = delete;

    /** @brief Claimed rows - the id frontier. The CPU's @c size() reads its publication
     *      watermark instead; GPU publication is unordered, so the claim frontier is the
     *      only dense bound, and the sampler probes past unpublished ids by protocol. */
    std::size_t size() const noexcept { return admission_.claimed_rows.load(std::memory_order_acquire); }
    std::size_t connectivity() const noexcept {
        return geometry_connectivity(geometry_.load(std::memory_order_acquire));
    }
    std::size_t dimensions() const noexcept { return geometry_dimensions(geometry_.load(std::memory_order_acquire)); }
    /** @brief Device-side plane views - the audit surface for invariant checkers & tests. */
    planes_view_t planes() const noexcept {
        u32_t const fragments = row_fragments();
        return {reinterpret_cast<fragment_t*>(vectors_plane_.base()),
                reinterpret_cast<norms_t*>(norms_plane_.base()),
                reinterpret_cast<u32_t*>(neighbors_plane_.base()),
                static_cast<u32_t>(connectivity()),
                fragments,
                lanes_per_row(),
                steps_per_row(),
                static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments)};
    }

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries. The first
     *      successful call freezes the geometry and the virtual-address ceiling; later
     *      calls with matching geometry only map further physical slabs.
     */
    reserve_result_t reserve(reserve_capacity_t config) noexcept {
        // Both plane allocators must place their memory on the index's pinned device.
        if (neighbors_plane_.device() != device_.ordinal)
            return {0, connectivity_unknown_k, status_t::config_mismatch_k};
        if (!config.dimensions || !config.connectivity) return {0, connectivity_unknown_k, status_t::config_mismatch_k};
        // The type's only compare-and-swap, on this cold path: the first success freezes
        // both flexible axes at once, exactly like the CPU index's geometry word; later
        // calls must agree with whatever they observe.
        packed_geometry_id_t const requested = pack_geometry(config.dimensions,
                                                             static_cast<std::size_t>(config.connectivity));
        packed_geometry_id_t observed = 0;
        if (!geometry_.compare_exchange_strong(observed, requested, std::memory_order_acq_rel) && observed != requested)
            return {mapped_rows(), static_cast<connectivity_t>(geometry_connectivity(observed)),
                    status_t::config_mismatch_k};
        connectivity_t const connectivity = config.connectivity;
        if (config.capacity > max_node_count_k) return {mapped_rows(), connectivity, status_t::overflow_k};
        if (!vectors_plane_.reserved_bytes()) { // First reservation: bind the device, size the ceiling.
            // Bind the runtime's primary context before any driver-API mapping - the VMM
            // planes must live in the same context the kernels launch into - and read the
            // device specs the launch grids and the ceiling policy consult.
            if (cudaSetDevice(device_.ordinal) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
                return {0, connectivity, status_t::device_error_k};
            expected<cuda_device_specs_t> const specs = cuda_device_specs_t::try_query(device_.ordinal);
            if (specs.status != status_t::success_k) return {0, connectivity, specs.status};
            device_ = specs.value;
            cuInit(0);
            // Virtual address space is free, so the ceiling is generous and independent of
            // the first requested capacity: geometry freezes here, row capacity stays
            // grow-only up to physical memory - a small first reserve() must not cap growth.
            // The reservation itself is the stored ceiling; no cached row count shadows it.
            std::size_t const bytes_per_row = row_bytes() + sizeof(norms_t) + connectivity * sizeof(u32_t);
            std::size_t const ceiling_rows = min_of(
                max_node_count_k, max_of<std::size_t>(config.capacity, 4 * device_.vram_bytes / bytes_per_row));
            if (vectors_plane_.reserve_range(ceiling_rows * row_bytes(), device_.ordinal) != status_t::success_k ||
                norms_plane_.reserve_range(ceiling_rows * sizeof(norms_t), device_.ordinal) != status_t::success_k ||
                neighbors_plane_.reserve_range(ceiling_rows * connectivity * sizeof(u32_t), device_.ordinal) !=
                    status_t::success_k)
                return {0, connectivity, status_t::bad_alloc_k};
            control_allocator_t control_allocator {nullptr, device_.ordinal};
            control_ = cuda_make_unique(control_allocator, 1);
            if (!control_ || cudaMemset(control_.get(), 0, sizeof(inner_cuda_control_t)) != cudaSuccess)
                return {0, connectivity, status_t::device_error_k};
        }
        status_t const grown = grow_rows(config.capacity, nullptr);
        return {mapped_rows(), connectivity, grown};
    }

    /**
     *  @brief Inserts a batch of @p new_vectors, assigning them the contiguous id range
     *      @b [size(),size()+rows) in row order. Blocks until the batch is fully
     *      published; @p stream orders the transfers & the launch.
     */
    insert_result_t insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids, inner_traversal_context_t& context,
                           insert_config_t config = {}) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        if (!rows) return {status_t::success_k, size(), static_cast<inner_node_id_t>(size())};
        if (!vectors_plane_.reserved_bytes()) return {status_t::uninitialized_k};
        if (new_vectors.extent(1) != dimensions()) return {status_t::config_mismatch_k};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context.batch_cursor_ || !context.telemetry_)
            return {status_t::config_mismatch_k};
        cudaStream_t const stream = context.stream();
        // Every fallible preparation precedes the id claim: a claimed range that never
        // publishes would starve every later batch's admission ramp forever, so nothing
        // that can fail is allowed to run between the claim and the launch. The staged
        // batch and the arena depend only on batch-local offsets, never on first_id.
        std::byte* const staging = context.ensure_staging(rows * row_bytes());
        if (!staging) return {status_t::bad_alloc_k};
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (scalar_row_bytes != row_bytes()) cudaMemsetAsync(staging, 0, rows * row_bytes(), stream);
        if (cudaMemcpy2DAsync(staging, row_bytes(), new_vectors.byte_data(), scalar_row_bytes, scalar_row_bytes, rows,
                              cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return {status_t::device_error_k};
        inner_cuda_telemetry_block_t const before = context.telemetry_snapshot();

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const pool_slots = pool_slots_for(config.expansion_factor, 4);
        inner_cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32,
                                                                     frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.sm_count) * insert_blocks_per_sm_k *
                                        inner_cuda_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        std::byte* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {status_t::bad_alloc_k};

        // Grow-then-claim under the cold admission latch: a sibling batch must never launch
        // into unmapped rows, and each batch claims a disjoint dense id range.
        while (admission_.latch.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
        std::size_t const first_id = admission_.claimed_rows.load(std::memory_order_relaxed);
        std::size_t const total_rows = first_id + rows;
        status_t const grown = grow_rows(total_rows, stream);
        if (grown == status_t::success_k) admission_.claimed_rows.store(total_rows, std::memory_order_release);
        admission_.latch.clear(std::memory_order_release);
        if (grown != status_t::success_k) return {grown};
        // Epoch discipline: zero marks never-used slots, and this launch's epochs ride
        // above the context's monotone floor - so only a fresh allocation or a wrapped
        // 32-bit epoch space warrants zeroing, never a warm arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + rows > 0xFFFFFFFEull) {
            cudaMemsetAsync(arena, 0, context.arena_bytes_, stream);
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += rows;
        cudaMemsetAsync(context.batch_cursor_.get(), 0, sizeof(u32_t), stream);
        inner_cuda_build_view<kit_t> view {planes(), //
                                           reinterpret_cast<fragment_t const*>(staging),
                                           static_cast<u32_t>(total_rows),
                                           insert_scratch_bytes_per_warp(),
                                           static_cast<inner_node_id_t>(first_id),
                                           arena,
                                           arena_shape,
                                           epoch_base,
                                           control_.get(),
                                           context.telemetry_.get(),
                                           context.batch_cursor_.get()};
        launch_insert(view, pool_slots, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};

        // The batch cursor is the per-launch completion oracle: every offset dispensed
        // means every claimed id published - give-up paths publish tombstones. Per-batch
        // `published` equality checks do not compose once sibling batches overlap.
        u32_t dispensed = 0;
        if (cudaMemcpy(&dispensed, context.batch_cursor_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
            dispensed < rows)
            return {status_t::device_error_k};
        inner_cuda_telemetry_block_t const after = context.telemetry_snapshot();

        insert_result_t result;
        result.first_id = static_cast<inner_node_id_t>(first_id);
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<inner_node_id_t>(first_id + row);
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
                           nk::matrix_span<inner_distance_t> result_distances, inner_traversal_context_t& context,
                           search_config_t config = {}) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (!queries_count) return {status_t::success_k};
        if (!vectors_plane_.reserved_bytes()) return {status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context.telemetry_) return {status_t::config_mismatch_k};
        cudaStream_t const stream = context.stream();
        bool const wants_distances = result_distances.extent(0) != 0;
        if (queries.extent(1) != dimensions() || result_ids.extent(0) != queries_count ||
            result_ids.extent(1) < config.count || config.count > 512 ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {status_t::config_mismatch_k};

        // The register ladder serves expansions up to its widest pool; anything past it
        // walks the arena-backed pool, whose results cover the head - the nearest 64.
        u32_t const expansion = config.expansion_factor > config.count ? config.expansion_factor : config.count * 2;
        bool const wide = expansion > inner_cuda_sorted_pool<16>::capacity_k;
        if (wide && config.count > inner_cuda_sorted_pool<2>::capacity_k) return {status_t::config_mismatch_k};

        // The pool must cover the requested count, whatever the expansion asked for.
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for(expansion, count_slots > 2 ? count_slots : 2);
        inner_cuda_arena_shape_t const arena_shape = arena_shape_for(
            wide ? expansion : pool_slots * 32,
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.sm_count) * search_blocks_per_sm_k *
                                        inner_cuda_warps_per_block_k;
        std::size_t const results_count = queries_count * config.count;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        std::byte* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {status_t::bad_alloc_k};
        // The batch transport rides the context's grow-only staging: queries first - they
        // carry the widest alignment - then the result ids and distances; the verb drains
        // the stream before returning, so one buffer serves every call on this lane.
        std::size_t const queries_bytes = queries_count * row_bytes();
        std::size_t const ids_bytes = results_count * sizeof(inner_node_id_t);
        std::byte* const staging = context.ensure_staging( //
            queries_bytes + ids_bytes + (wants_distances ? results_count * sizeof(inner_distance_t) : 0));
        if (!staging) return {status_t::bad_alloc_k};
        void* const device_queries = staging;
        inner_node_id_t* const device_ids = reinterpret_cast<inner_node_id_t*>(staging + queries_bytes);
        inner_distance_t* const device_distances =
            wants_distances ? reinterpret_cast<inner_distance_t*>(staging + queries_bytes + ids_bytes) : nullptr;
        // Epoch discipline: zero marks never-used slots, and this launch's epochs ride
        // above the context's monotone floor - zeroing happens only on fresh allocations
        // and 32-bit epoch wrap, never on a warm arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + queries_count > 0xFFFFFFFEull) {
            cudaMemsetAsync(arena, 0, context.arena_bytes_, stream);
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += queries_count;
        // Pitched upload: host rows are packed scalars, device rows pad to whole fragments,
        // and the pad bytes must be zero for the zero-tail distance identity to hold.
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (scalar_row_bytes != row_bytes()) cudaMemsetAsync(device_queries, 0, queries_count * row_bytes(), stream);
        cudaMemcpy2DAsync(device_queries, row_bytes(), queries.byte_data(), scalar_row_bytes, scalar_row_bytes,
                          queries_count, cudaMemcpyHostToDevice, stream);
        inner_cuda_telemetry_block_t const before = context.telemetry_snapshot();

        inner_cuda_search_view<kit_t> view {planes(),
                                            reinterpret_cast<fragment_t const*>(device_queries),
                                            device_ids,
                                            device_distances,
                                            static_cast<u32_t>(queries_count),
                                            config.count,
                                            static_cast<u32_t>(size()),
                                            static_cast<u32_t>(row_bytes()),
                                            expansion,
                                            arena,
                                            arena_shape,
                                            epoch_base,
                                            context.telemetry_.get()};
        if (wide) launch_search_wide(view, stream);
        else launch_search(view, pool_slots, stream);
        cudaMemcpyAsync(result_ids.data(), device_ids, results_count * sizeof(inner_node_id_t), cudaMemcpyDeviceToHost,
                        stream);
        if (wants_distances)
            cudaMemcpyAsync(result_distances.data(), device_distances, results_count * sizeof(inner_distance_t),
                            cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};

        inner_cuda_telemetry_block_t const after = context.telemetry_snapshot();
        search_result_t result;
        result.count = config.count;
        result.visited_members = after.hops - before.hops;
        result.computed_distances = after.scored - before.scored;
        result.restarts = after.restarts - before.restarts;
        return result;
    }

    /**
     *  @brief One query in, the nearest matches ascending in @c context.results - the CPU
     *      verb's exact shape, so the shared scenarios & phases drive the GPU unmodified.
     *      A one-query grid is the correctness path, not the throughput path: batch
     *      queries through the overload above to fill the machine.
     */
    search_result_t search(vector_view_t query_vector, inner_traversal_context_t& context,
                           search_config_t config = {}) noexcept {
        context.results.count_ = 0;
        if (!size()) return {status_t::success_k}; // An empty index matches nothing, touches nothing.
        // Matches reported follow the CPU semantic: up to the expansion factor, defaulting
        // to twice the connectivity, clipped to the results landing and the graph.
        u32_t const expansion = config.expansion_factor ? config.expansion_factor
                                                        : static_cast<u32_t>(connectivity()) * 2;
        u32_t const count = min_of<u32_t>(min_of<u32_t>(inner_traversal_context_t::results_capacity_k, expansion),
                                          static_cast<u32_t>(size()));
        inner_node_id_t staged_ids[inner_traversal_context_t::results_capacity_k];
        inner_distance_t staged_distances[inner_traversal_context_t::results_capacity_k];
        vectors_view_t const one_query {query_vector.data(), {1, dimensions()}};
        config.count = count;
        search_result_t result = search(one_query, nk::matrix_span<inner_node_id_t> {staged_ids, {1, count}},
                                        nk::matrix_span<inner_distance_t> {staged_distances, {1, count}}, context,
                                        config);
        if (!result) return result;
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != inner_node_id_unknown_k; ++live)
            context.results.entries_[live] = inner_scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        result.count = live;
        return result;
    }

    /** @brief The serialized extents: published nodes x logical dimensions & connectivity -
     *      the fragment padding is a storage detail that never reaches a file. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /**
     *  @brief Fills the caller's typed matrices with the two serialized planes: vectors as
     *      logical @b [nodes,dimensions] rows - the fragment padding stripped by a strided
     *      copy - and neighbors verbatim, already speaking the positional sentinel
     *      convention (an id at or past the node count names no edge, a slot-0 self-loop
     *      is published-empty). Norms are derived state and never serialized.
     */
    status_t save(nk::matrix_span<scalar_t> vectors, nk::matrix_span<node_id_t> neighbors,
                  synchronization_t synchronization) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return status_t::config_mismatch_k;
        saved_shape_t const shape = saved_shape();
        if (vectors.extent(0) != shape.nodes || vectors.extent(1) != shape.dimensions ||
            neighbors.extent(0) != shape.nodes || neighbors.extent(1) != shape.connectivity)
            return status_t::config_mismatch_k;
        if (!shape.nodes) return status_t::success_k;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (cudaMemcpy2D(vectors.data(), scalar_row_bytes, vectors_plane_.base(), row_bytes(), scalar_row_bytes,
                         shape.nodes, cudaMemcpyDeviceToHost) != cudaSuccess)
            return status_t::device_error_k;
        if (cudaMemcpy(neighbors.data(), neighbors_plane_.base(), shape.nodes * shape.connectivity * sizeof(node_id_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /**
     *  @brief Rebuilds the index from serialized planes: demands an exclusive caller and a
     *      virgin, geometry-frozen index; re-pads the vectors into fragment rows - freshly
     *      committed plane memory arrives zeroed, so the pad tails hold the identity -
     *      re-derives the norms on the device, and publishes the counters last.
     */
    status_t load(nk::matrix_view<scalar_t> vectors, nk::matrix_view<node_id_t> neighbors,
                  synchronization_t synchronization) noexcept {
        if (synchronization != synchronization_t::exclusive_k) return status_t::config_mismatch_k;
        if (!vectors_plane_.reserved_bytes()) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        std::size_t const nodes = vectors.extent(0);
        if (vectors.extent(1) != dimensions() || neighbors.extent(0) != nodes ||
            neighbors.extent(1) != connectivity() || nodes > max_node_count_k)
            return status_t::config_mismatch_k;
        if (!nodes) return status_t::success_k;
        if (status_t grown = grow_rows(nodes, nullptr); grown != status_t::success_k) return grown;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (cudaMemcpy2D(vectors_plane_.base(), row_bytes(), vectors.data(), scalar_row_bytes, scalar_row_bytes, nodes,
                         cudaMemcpyHostToDevice) != cudaSuccess)
            return status_t::device_error_k;
        if (cudaMemcpy(neighbors_plane_.base(), neighbors.data(), nodes * connectivity() * sizeof(node_id_t),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return status_t::device_error_k;
        u32_t const blocks = static_cast<u32_t>(device_.sm_count) * search_blocks_per_sm_k;
        inner_cuda_derive_norms_kernel<kit_t><<<blocks, block_threads_k>>>(planes(), static_cast<u32_t>(nodes));
        // Publish the control block last, once every plane byte is in place.
        inner_cuda_control_t const restored {static_cast<u32_t>(nodes)};
        if (cudaMemcpy(control_.get(), &restored, sizeof(restored), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
            return status_t::device_error_k;
        admission_.claimed_rows.store(nodes, std::memory_order_release);
        return status_t::success_k;
    }

  private:
    using vectors_plane_allocator_t = typename vectors_allocator_t::template rebind<std::byte>::other;
    using neighbors_plane_allocator_t = typename neighbors_allocator_t::template rebind<std::byte>::other;
    using control_allocator_t = typename traversals_allocator_t::template rebind<inner_cuda_control_t>::other;

    /**
     *
     *  @section inner_cuda_index_host_state Host state & its concurrency regimes
     *
     *  Four regimes cover everything below, mirroring the CPU index member-for-member:
     *  @c device_ and @c geometry_ are @b frozen - the specs at first reserve, the geometry
     *  by the type's only compare-and-swap - and read lock-free everywhere after. The three
     *  @b planes append-only grow under the admission latch; each reservable allocator IS
     *  its plane, owning the base pointer and the committed extent, so ceilings and mapped
     *  row counts are derived observers, never cached. The @b control block is device
     *  memory no host thread dereferences. The @b admission struct is the only mutable host
     *  state: every field is written solely between @c latch acquire and clear.
     *
     *  Counter correspondence to the CPU: its @c last_allocated_id_ is our derived
     *  @c mapped_rows(), its @c next_visible_id_ our @c admission_.claimed_rows, and its
     *  @c last_visible_id_ our device-side @c control_->published - except GPU publication
     *  is unordered, so @c size() reports the dense claim frontier, and landmark sampling
     *  probes past unpublished ids by protocol instead of trusting a watermark.
     */

    /** @brief Hardware identity: the ordinal comes from the vectors allocator at
     *      construction; the remaining specs fill at the first @c reserve(). */
    cuda_device_specs_t device_;
    /** @brief The frozen geometry word - dims high, connectivity low, zero = unreserved. */
    std::atomic<packed_geometry_id_t> geometry_ {0};
    /** @brief The planes: norms share the vectors allocator's placement. */
    vectors_plane_allocator_t vectors_plane_, norms_plane_;
    neighbors_plane_allocator_t neighbors_plane_;
    /** @brief The device-resident control block, allocated by the first @c reserve(). */
    cuda_unique_ptr<inner_cuda_control_t, control_allocator_t> control_;
    /** @brief Grow-then-claim state: @c latch serializes siblings, @c claimed_rows
     *      release-publishes each batch's dense id range - what @c size() reports. */
    struct admission_t {
        std::atomic_flag latch = ATOMIC_FLAG_INIT;
        std::atomic<std::size_t> claimed_rows = 0;
    } admission_;

    /** @brief Rows physically mapped across all three planes, derived from the neighbors
     *      plane - it commits last in @c grow_rows, so its extent is the safe minimum even
     *      after a partially failed grow. */
    std::size_t mapped_rows() const noexcept {
        std::size_t const stride = connectivity() * sizeof(u32_t);
        return stride ? neighbors_plane_.committed_bytes() / stride : 0;
    }

    /** @brief Maps every plane up to @p rows and sentinel-fills the fresh neighbor rows -
     *      always ahead of any launch that could dispense new_node_ids into them. Callers
     *      hold the admission latch, or run exclusively like @c reserve and @c load. */
    status_t grow_rows(std::size_t rows, cudaStream_t stream) noexcept {
        if (rows <= mapped_rows()) return status_t::success_k;
        std::size_t const neighbors_before = neighbors_plane_.committed_bytes();
        if (status_t status = vectors_plane_.commit(rows * row_bytes()); status != status_t::success_k) return status;
        if (status_t status = norms_plane_.commit(rows * sizeof(norms_t)); status != status_t::success_k) return status;
        if (status_t status = neighbors_plane_.commit(rows * connectivity() * sizeof(u32_t));
            status != status_t::success_k)
            return status;
        std::size_t const fresh_bytes = neighbors_plane_.committed_bytes() - neighbors_before;
        if (fresh_bytes && cuMemsetD32Async(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>( //
                                                neighbors_plane_.base() + neighbors_before)),
                                            inner_node_id_unknown_k, fresh_bytes / 4, stream) != CUDA_SUCCESS)
            return status_t::device_error_k;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return status_t::device_error_k;
        return status_t::success_k;
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
    inline static constexpr u32_t insert_blocks_per_sm_k = inner_cuda_resident_blocks_per_sm_k;
    inline static constexpr u32_t search_blocks_per_sm_k = 8;
    inline static constexpr u32_t block_threads_k = inner_cuda_warps_per_block_k * 32;

    /** @brief Default dynamic-shared-memory opt-in threshold: launches asking for more
     *      than this must raise the kernel's attribute before the first launch. */
    inline static constexpr u32_t static_scratch_limit_k = 48 * 1024;

    template <u32_t pool_slots_>
    void launch_insert_with(inner_cuda_build_view<kit_t> view, cudaStream_t stream) noexcept {
        u32_t const scratch_bytes = inner_cuda_warps_per_block_k * view.scratch_bytes_per_warp;
        if (scratch_bytes > static_scratch_limit_k)
            cudaFuncSetAttribute(inner_cuda_insert_kernel<kit_t, pool_slots_>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, scratch_bytes);
        inner_cuda_insert_kernel<kit_t, pool_slots_>
            <<<device_.sm_count * insert_blocks_per_sm_k, block_threads_k, scratch_bytes, stream>>>(view);
    }
    void launch_insert(inner_cuda_build_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        switch (pool_slots) {
        case 2: launch_insert_with<2>(view, stream); break;
        case 4: launch_insert_with<4>(view, stream); break;
        case 8: launch_insert_with<8>(view, stream); break;
        default: launch_insert_with<16>(view, stream);
        }
    }
    template <u32_t pool_slots_>
    void launch_search_with(inner_cuda_search_view<kit_t> view, cudaStream_t stream) noexcept {
        u32_t const scratch_bytes = inner_cuda_warps_per_block_k * view.scratch_bytes_per_warp;
        if (scratch_bytes > static_scratch_limit_k)
            cudaFuncSetAttribute(inner_cuda_search_kernel<kit_t, pool_slots_>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, scratch_bytes);
        inner_cuda_search_kernel<kit_t, pool_slots_>
            <<<device_.sm_count * search_blocks_per_sm_k, block_threads_k, scratch_bytes, stream>>>(view);
    }
    void launch_search(inner_cuda_search_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        switch (pool_slots) {
        case 2: launch_search_with<2>(view, stream); break;
        case 4: launch_search_with<4>(view, stream); break;
        case 8: launch_search_with<8>(view, stream); break;
        default: launch_search_with<16>(view, stream);
        }
    }
    void launch_search_wide(inner_cuda_search_view<kit_t> view, cudaStream_t stream) noexcept {
        u32_t const scratch_bytes = inner_cuda_warps_per_block_k * view.scratch_bytes_per_warp;
        if (scratch_bytes > static_scratch_limit_k)
            cudaFuncSetAttribute(inner_cuda_search_wide_kernel<kit_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 scratch_bytes);
        inner_cuda_search_wide_kernel<kit_t>
            <<<device_.sm_count * search_blocks_per_sm_k, block_threads_k, scratch_bytes, stream>>>(view);
    }
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
