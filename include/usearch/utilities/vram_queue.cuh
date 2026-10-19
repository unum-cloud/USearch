/**
 *  @file include/usearch/utilities/vram_queue.cuh
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Device-side dynamic data-structures for Graph traversals on GPUs.
 *
 *  Standalone header providing per-warp traversal structures over registers, shared
 *  memory, and global arenas, like:
 *
 *  - Packed scored-node @b candidates ordered by raw distance bits
 *  - Register-resident and arena-backed @b frontiers with append-only tail candidates
 *  - Exact epoch-tagged @b visited-sets probed warp-cooperatively
 *  - Locked @b neighbor rows, sorted @b buffers, @b telemetry and the @b arena slices every
 *    index's walks carve
 *
 *  Compiled by NVCC and HIP-Clang alike: every collective goes through a warp policy each vendor
 *  defines beside its device vocabulary, so lane masks, rank striping and the bitonic stages all
 *  derive from its @c width_k, 32 or 64 lanes. A policy names @c mask_t, @c width_k, the
 *  @c shuffle, @c shuffle_xor, @c shuffle_up and @c shuffle_down moves, the @c ballot, @c any,
 *  @c all and @c match_any votes, @c sync, the @c count, @c first and @c below mask arithmetic,
 *  the relaxed, acquire and release atomics, and the @c fence_release fence. Lanes are @c u32_t
 *  throughout. Staging candidate rows into shared memory is the vendor's schedule, not the warp
 *  policy, and no policy sleeps: contended rows are revisited after other work instead.
 */
#pragma once

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_walks.hpp" // `gpu_arena_shape_t`, the host twin of the arena shape

namespace unum::usearch::utilities {

/** Row @p row of a @b [rows,stride] matrix at @p base - the one place a row offset widens past 32
 *  bits, since node IDs times row strides outgrow them. */
template <typename value_type_>
__device__ __forceinline__ value_type_* vram_row(value_type_* base, std::size_t row, u32_t stride) noexcept {
    return base + row * stride;
}

/** The random stream of @p lane walking for @p stream_key - the node it inserts or refines, or the
 *  query it answers - seeded through @ref mix_bits as the host seeds its streams, so no two lanes
 *  or keys share one. */
template <typename warp_type_>
__device__ __forceinline__ splitmix64_t vram_lane_generator(u64_t stream_key, u32_t lane) noexcept {
    return splitmix64_t {mix_bits(stream_key * warp_type_::width_k + lane + 1)};
}

/** The calling warp's place in a launch: its grid-wide index, its lane, and the grid's warps. */
struct vram_grid_warp_t {
    u32_t index, lane, count;
};

template <typename warp_type_>
__device__ __forceinline__ vram_grid_warp_t vram_grid_warp() noexcept {
    return {(blockIdx.x * blockDim.x + threadIdx.x) / warp_type_::width_k, threadIdx.x % warp_type_::width_k,
            gridDim.x * blockDim.x / warp_type_::width_k};
}

/** Hands @p callback every neighbor of @p row the calling lane owns - its lane-strided slots, past
 *  sentinels, self-loops and IDs at or past @p rows_count. */
template <typename warp_type_, typename node_id_type_, typename callback_type_>
__device__ __forceinline__ void vram_for_each_link(node_id_type_ const* neighbors, u32_t connectivity,
                                                   node_id_type_ rows_count, node_id_type_ row, u32_t lane,
                                                   callback_type_&& callback) noexcept {
    node_id_type_ const* const slots = vram_row(neighbors, row, connectivity);
    for (u32_t slot = lane; slot < connectivity; slot += warp_type_::width_k)
        if (node_id_type_ const target = slots[slot];
            target != std::numeric_limits<node_id_type_>::max() && target != row && target < rows_count)
            callback(target);
}

/** One traversal candidate, ordered by distance as raw unsigned words; @c u32_t and @c u64_t IDs
 *  each specialize it. */
template <typename node_id_type_>
struct vram_scored_node;

/**
 *  @brief One traversal candidate packed into eight bytes, ordered by distance as raw
 *      unsigned words: bits [63:33] hold the top 31 bits of the sign-flipped @c f32
 *      distance, bit 32 carries the "already expanded" flag, bits [31:0] the full node ID.
 *
 *  The all-ones pattern is the empty slot: its distance prefix is the flipped NaN that
 *  sorts last, it reads as expanded, and its ID is @c node_id_unknown_k - so
 *  traversal loops need no emptiness guards. Dropping the distance LSB costs one ulp of
 *  ordering precision: candidates are "sorted by @c distance_bits(), tie order unspecified",
 *  which is the practical semantic of approximate search either way, and in exchange IDs
 *  keep all 32 bits - the ID ceiling matches the CPU-resident index exactly.
 */
template <>
struct vram_scored_node<u32_t> {
    u64_t bits_;

    inline static constexpr u64_t expanded_bit_k = u64_t(1) << 32;

    /** Dispensable IDs: everything but the reserved all-ones sentinel. */
    inline static constexpr std::size_t id_capacity_k = (u64_t(1) << 32) - 1u;

    static __device__ __forceinline__ vram_scored_node make(distance_t distance, u32_t node_id) noexcept {
        u32_t flipped = __float_as_uint(distance);
        flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return {(static_cast<u64_t>(flipped >> 1) << 33) | node_id};
    }
    __device__ static constexpr vram_scored_node sentinel() noexcept { return {~u64_t(0)}; }

    /** The flag-free comparison currency: 31 ordering bits of the flipped distance. */
    __device__ __forceinline__ u32_t distance_bits() const noexcept { return static_cast<u32_t>(bits_ >> 33); }

    /** Decoded distance, with the truncated ordering LSB reconstructed as zero. */
    __device__ __forceinline__ distance_t distance() const noexcept {
        u32_t flipped = distance_bits() << 1;
        flipped ^= (flipped & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
        return __uint_as_float(flipped);
    }
    __device__ __forceinline__ u32_t id() const noexcept { return static_cast<u32_t>(bits_); }
    __device__ __forceinline__ bool expanded() const noexcept { return bits_ & expanded_bit_k; }

    /** Lazily deletes the entry from the frontier; a no-op on the sentinel. */
    __device__ __forceinline__ void mark_expanded() noexcept { bits_ |= expanded_bit_k; }
    __device__ __forceinline__ bool operator<(vram_scored_node other) const noexcept { return bits_ < other.bits_; }
    __device__ __forceinline__ bool operator==(vram_scored_node other) const noexcept { return bits_ == other.bits_; }
};
static_assert(sizeof(vram_scored_node<u32_t>) == gpu_scored_node_bytes<u32_t>, "Entries shuffle as one 64-bit word");
static_assert(std::is_trivially_copyable_v<vram_scored_node<u32_t>>);

/**
 *  @brief One traversal candidate with a 64-bit ID, in two words ordered by distance, then ID:
 *      bits [63:32] of the first hold the whole sign-flipped @c f32 distance, bit 0 carries the
 *      "already expanded" flag, and the second word is the full node ID.
 *
 *  The all-ones pair is the empty slot, as in the 32-bit twin: its distance is the flipped NaN that
 *  sorts last, it reads as expanded, and its ID is the all-ones sentinel. The ID has a word of its
 *  own, so the distance keeps every bit.
 */
template <>
struct vram_scored_node<u64_t> {
    u64_t bits_;
    u64_t id_;

    inline static constexpr u64_t expanded_bit_k = 1;

    /** Dispensable IDs: everything but the reserved all-ones sentinel. */
    inline static constexpr std::size_t id_capacity_k = ~u64_t(0);

    static __device__ __forceinline__ vram_scored_node make(distance_t distance, u64_t node_id) noexcept {
        u32_t flipped = __float_as_uint(distance);
        flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return {static_cast<u64_t>(flipped) << 32, node_id};
    }
    __device__ static constexpr vram_scored_node sentinel() noexcept { return {~u64_t(0), ~u64_t(0)}; }

    /** The flag-free comparison currency: all 32 ordering bits of the flipped distance. */
    __device__ __forceinline__ u32_t distance_bits() const noexcept { return static_cast<u32_t>(bits_ >> 32); }

    __device__ __forceinline__ distance_t distance() const noexcept {
        u32_t flipped = distance_bits();
        flipped ^= (flipped & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
        return __uint_as_float(flipped);
    }
    __device__ __forceinline__ u64_t id() const noexcept { return id_; }
    __device__ __forceinline__ bool expanded() const noexcept { return bits_ & expanded_bit_k; }

    /** Lazily deletes the entry from the frontier; a no-op on the sentinel. */
    __device__ __forceinline__ void mark_expanded() noexcept { bits_ |= expanded_bit_k; }
    __device__ __forceinline__ bool operator<(vram_scored_node other) const noexcept {
        return bits_ < other.bits_ || (bits_ == other.bits_ && id_ < other.id_);
    }
    __device__ __forceinline__ bool operator==(vram_scored_node other) const noexcept {
        return bits_ == other.bits_ && id_ == other.id_;
    }
};
static_assert(sizeof(vram_scored_node<u64_t>) == gpu_scored_node_bytes<u64_t>, "Entries shuffle as two words");
static_assert(std::is_trivially_copyable_v<vram_scored_node<u64_t>>);

/** Lane shuffles have no user-type overloads - an entry shuffles its raw words. */
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ vram_scored_node<node_id_type_> shuffle_xor(vram_scored_node<node_id_type_> value,
                                                                       u32_t lane_mask) noexcept {
    if constexpr (std::is_same_v<node_id_type_, u32_t>) return {warp_type_::shuffle_xor(value.bits_, lane_mask)};
    else return {warp_type_::shuffle_xor(value.bits_, lane_mask), warp_type_::shuffle_xor(value.id_, lane_mask)};
}
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ vram_scored_node<node_id_type_> shuffle_from(vram_scored_node<node_id_type_> value,
                                                                        u32_t source_lane) noexcept {
    if constexpr (std::is_same_v<node_id_type_, u32_t>) return {warp_type_::shuffle(value.bits_, source_lane)};
    else return {warp_type_::shuffle(value.bits_, source_lane), warp_type_::shuffle(value.id_, source_lane)};
}
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ vram_scored_node<node_id_type_> shuffle_up(vram_scored_node<node_id_type_> value,
                                                                      u32_t delta) noexcept {
    if constexpr (std::is_same_v<node_id_type_, u32_t>) return {warp_type_::shuffle_up(value.bits_, delta)};
    else return {warp_type_::shuffle_up(value.bits_, delta), warp_type_::shuffle_up(value.id_, delta)};
}

/** Full bitonic sort of one lane-wide entry vector, ascending by lane - fifteen shuffle-exchange
 *  stages on 32 lanes, twenty-one on 64, all in registers. */
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ vram_scored_node<node_id_type_> sort_lanes(vram_scored_node<node_id_type_> value,
                                                                      u32_t lane) noexcept {
    using scored_node_t = vram_scored_node<node_id_type_>;
#pragma unroll
    for (u32_t stage_width = 2; stage_width <= warp_type_::width_k; stage_width <<= 1)
#pragma unroll
        for (u32_t partner_stride = stage_width >> 1; partner_stride; partner_stride >>= 1) {
            scored_node_t const partner = shuffle_xor<warp_type_>(value, partner_stride);
            bool const ascending = (lane & stage_width) == 0;
            bool const lower = (lane & partner_stride) == 0;
            scored_node_t const nearer = value < partner ? value : partner;
            scored_node_t const further = value < partner ? partner : value;
            value = (ascending == lower) ? nearer : further;
        }
    return value;
}

/**
 *  @brief Warp-collective ascending candidates of @c width_k x @c slots_ scored entries - the fused
 *      results-and-frontier of one traversal, striped as rank @b r at (slot r/width, lane
 *      r%width), always sorted by @c distance_bits() with unspecified tie order.
 *
 *  Subsumes both host-side traversal containers: the sorted results buffer (the radius
 *  lives at the last rank) and the frontier heap (the nearest unexpanded entry plays
 *  @c top(), the expanded bit plays lazy deletion). Two register-residency invariants are
 *  part of the contract: every slot index is compile-time after unrolling - a single
 *  runtime-indexed access spills all of its candidates to local memory - and batch insertion
 *  goes through the fold-merge network below, never a full re-sort.
 */
template <u32_t slots_, typename warp_type_, typename node_id_type_>
struct vram_sorted_candidates {
    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    scored_node_t entries_[slots_];

    inline static constexpr u32_t width_k = warp_t::width_k;
    inline static constexpr u32_t capacity_k = slots_ * width_k;
    static_assert((slots_ & (slots_ - 1)) == 0, "The fold-merge network needs a power-of-two slot count");

    __device__ __forceinline__ void clear() noexcept {
#pragma unroll
        for (u32_t slot = 0; slot != slots_; ++slot) entries_[slot] = scored_node_t::sentinel();
    }

    /** Merges a lane-wide staged batch into the candidates, keeping the @c capacity_k nearest.
     *  [ascending candidates | sentinel pad | descending batch] is one bitonic sequence: a single
     *  fold absorbs the batch, log2 of @c slots_ in-lane stages plus log2 of @c width_k shuffle
     *  stages restore ascending order. */
    __device__ __forceinline__ void insert_batch(scored_node_t staged, u32_t lane) noexcept {
        (void)insert_batch_spilling(staged, lane);
    }

    /** The same fold-merge, reporting what fell out: at the fold boundary the element-wise @b max
     *  of the last slot and the descending batch is exactly the union's worst lane-width -
     *  sentinels while the candidates still have room. @return This lane's displaced entry, for the
     *  caller to spill onward or drop. */
    __device__ __forceinline__ scored_node_t insert_batch_spilling(scored_node_t staged, u32_t lane) noexcept {
        scored_node_t const descending = shuffle_xor<warp_t>(sort_lanes<warp_t>(staged, lane), width_k - 1);
        scored_node_t const displaced = entries_[slots_ - 1] < descending ? descending : entries_[slots_ - 1];
        entries_[slots_ - 1] = entries_[slots_ - 1] < descending ? entries_[slots_ - 1] : descending;
#pragma unroll
        for (u32_t stride = slots_ / 2; stride >= 1; stride >>= 1)
#pragma unroll
            for (u32_t base = 0; base < slots_; ++base)
                if ((base & stride) == 0 && (base | stride) < slots_) {
                    scored_node_t const lower_entry = entries_[base];
                    scored_node_t const upper_entry = entries_[base | stride];
                    entries_[base] = lower_entry < upper_entry ? lower_entry : upper_entry;
                    entries_[base | stride] = lower_entry < upper_entry ? upper_entry : lower_entry;
                }
        // A rolled stride keeps the network's code small, while the slot loop below stays unrolled.
#pragma unroll 1
        for (u32_t partner_stride = width_k / 2; partner_stride; partner_stride >>= 1) {
            bool const lower = (lane & partner_stride) == 0;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot) {
                scored_node_t const partner = shuffle_xor<warp_t>(entries_[slot], partner_stride);
                entries_[slot] = lower == (entries_[slot] < partner) ? entries_[slot] : partner;
            }
        }
        return displaced;
    }

    /** The fold-merge's result for a batch of which only the @p admitted lanes can enter: each
     *  takes its rank, ranks at and past it move up one, and the last rank drops. Raw entries never
     *  tie, since the IDs of one walk are distinct, so the strict rank is exact; every slot index
     *  stays compile-time, and both guards are warp-uniform. */
    __device__ __forceinline__ void insert_each(scored_node_t staged, typename warp_t::mask_t admitted,
                                                u32_t lane) noexcept {
        while (admitted) {
            scored_node_t const value = shuffle_from<warp_t>(staged, warp_t::first(admitted));
            admitted &= admitted - 1;
            u32_t rank = 0;
#pragma unroll
            for (u32_t slot = 0; slot != slots_; ++slot) rank += warp_t::count(warp_t::ballot(entries_[slot] < value));
            if (rank == capacity_k) continue; // An earlier entry of this batch already pushed the worst below it.
            // Highest slot first, so each slot's old last lane carries up before that slot
            // is rewritten.
#pragma unroll
            for (u32_t slot = slots_; slot-- != 0;) {
                if ((slot + 1) * width_k <= rank) break; // Lower slots keep their ranks.
                scored_node_t const carried = slot ? shuffle_from<warp_t>(entries_[slot - 1], width_k - 1)
                                                   : scored_node_t::sentinel();
                scored_node_t const shifted = shuffle_up<warp_t>(entries_[slot], 1);
                u32_t const position = slot * width_k + lane;
                if (position == rank) entries_[slot] = value;
                else if (position > rank) entries_[slot] = lane ? shifted : carried;
            }
        }
    }

    /** The rank capacity. */
    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_k; }

    /** The nearest entry not yet expanded - the frontier's @c top() - or the sentinel. */
    __device__ __forceinline__ scored_node_t nearest_unexpanded() const noexcept {
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot) {
            typename warp_t::mask_t const mask = warp_t::ballot(!entries_[slot].expanded());
            if (mask) return shuffle_from<warp_t>(entries_[slot], warp_t::first(mask));
        }
        return scored_node_t::sentinel();
    }

    /** Lazily deletes @p entry from the frontier - the unrolled-compare spelling
     *  keeps every slot index compile-time. Duplicates cannot arise: the exact
     *  visited set admits every ID into the walk exactly once. */
    __device__ __forceinline__ void mark_expanded_at(scored_node_t entry) noexcept {
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (entries_[slot] == entry) entries_[slot].mark_expanded();
    }

    /** The search radius: ordering bits of the furthest kept entry. */
    __device__ __forceinline__ u32_t radius() const noexcept {
        return shuffle_from<warp_t>(entries_[slots_ - 1], width_k - 1).distance_bits();
    }

    /** True once every rank holds a real candidate - the radius becomes binding. */
    __device__ __forceinline__ bool full() const noexcept {
        return !(shuffle_from<warp_t>(entries_[slots_ - 1], width_k - 1) == scored_node_t::sentinel());
    }

    /** Kills every staged entry whose ID already resides in the candidates. The exact visited set
     *  stages every ID once until it saturates; past that, the walk and the arena frontier's final
     *  fold keep one copy of a re-scored ID. */
    __device__ __forceinline__ scored_node_t deduplicate(scored_node_t staged, u32_t lane) const noexcept {
        node_id_type_ const staged_id = staged.id();
        bool resident = false;
#pragma unroll
        for (u32_t probe = 0; probe < width_k; ++probe) {
            node_id_type_ const probe_id = warp_t::shuffle(staged_id, probe);
            bool holds = false;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot)
                holds |= !(entries_[slot] == scored_node_t::sentinel()) && entries_[slot].id() == probe_id;
            if (warp_t::any(holds) && lane == probe) resident = true;
        }
        return resident ? scored_node_t::sentinel() : staged;
    }

    /** Entry at @p rank, valid on every lane. */
    __device__ __forceinline__ scored_node_t rank_entry(u32_t rank) const noexcept {
        scored_node_t result = scored_node_t::sentinel();
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (slot == rank / width_k) result = entries_[slot];
        return shuffle_from<warp_t>(result, rank % width_k);
    }
};

/** One warp's slice of the global traversal arena: an append-only span of scored
 *  entries spilled from the register head, consumed by refills on frontier drain.
 *  Sized ahead by the host and never grown; its frontier compacts it before it fills. */
template <typename warp_type_, typename node_id_type_>
struct vram_tail_candidates {
    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_ = 0;

    /** Appends a lane-width, answering how many entries found no room, every lane alike. */
    __device__ __forceinline__ u32_t append(scored_node_t value, u32_t lane) noexcept {
        bool const live = !(value == scored_node_t::sentinel());
        typename warp_t::mask_t const live_mask = warp_t::ballot(live);
        u32_t const ordinal = warp_t::count(live_mask & warp_t::below(lane));
        if (live && size_ + ordinal < capacity_) entries_[size_ + ordinal] = value;
        u32_t const grown = min_of(size_ + warp_t::count(live_mask), capacity_);
        u32_t const dropped = size_ + warp_t::count(live_mask) - grown;
        size_ = grown;
        warp_t::sync();
        return dropped;
    }
};

/** The register frontier: the fused results-and-frontier of a bounded walk, a thin verb layer over
 *  one set of register candidates. The CPU keeps its results buffer and frontier heap separate;
 *  registers fuse them - the expanded bit plays frontier deletion, the last rank plays the results
 *  radius - and this wrapper names the traversal verbs so one walk body serves this and the arena
 *  twin below. */
template <u32_t slots_, typename warp_type_, typename node_id_type_>
struct vram_register_frontier {
    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    vram_sorted_candidates<slots_, warp_t, node_id_type_> ranks_;

    __device__ __forceinline__ void clear(u32_t) noexcept { ranks_.clear(); }

    /** A walk over full ranks ends at their radius, so a further walk needs no budget renewed. */
    __device__ __forceinline__ void renew() noexcept {}

    /** Nothing to compact or drop: the ranks keep the nearest of every union by
     *  construction. */
    __device__ __forceinline__ u32_t compactions() const noexcept { return 0; }
    __device__ __forceinline__ u32_t dropped() const noexcept { return 0; }

    /** Whether the ranks already hold the ID @p staged_id names on this lane, every lane taking
     *  part. */
    __device__ __forceinline__ bool holds(node_id_type_ staged_id, u32_t lane) const noexcept {
        return ranks_.deduplicate(scored_node_t::make(0, staged_id), lane) == scored_node_t::sentinel();
    }

    /** The furthest kept rank, which nothing at or past enters; the sentinel until every rank
     *  fills. */
    __device__ __forceinline__ scored_node_t worst() const noexcept {
        return shuffle_from<warp_t>(ranks_.entries_[slots_ - 1], warp_t::width_k - 1);
    }
    __device__ __forceinline__ void insert_batch(scored_node_t staged, u32_t lane) noexcept {
        // Only entries below the worst kept one can enter:
        // a few take their ranks, a crowd takes the fold-merge.
        constexpr u32_t few_k = 3;
        scored_node_t const worst = shuffle_from<warp_t>(ranks_.entries_[slots_ - 1], warp_t::width_k - 1);
        typename warp_t::mask_t const admitted = warp_t::ballot(staged < worst);
        if (warp_t::count(admitted) <= few_k) ranks_.insert_each(staged, admitted, lane);
        else ranks_.insert_batch(staged, lane);
    }

    /** The nearest undiscovered-from candidate, already marked expanded - or the
     *  sentinel when every rank is spent. */
    __device__ __forceinline__ scored_node_t pop_nearest(u32_t lane) noexcept {
        scored_node_t const nearest = ranks_.nearest_unexpanded();
        if (!(nearest == scored_node_t::sentinel())) ranks_.mark_expanded_at(nearest);
        return nearest;
    }

    /** After a sentinel pop: full ranks mean the walk is done, not just starved. */
    __device__ __forceinline__ bool exhausted() const noexcept { return ranks_.full(); }

    /** The classic radius test: full ranks and a candidate beyond the worst kept. */
    __device__ __forceinline__ bool beyond_radius(scored_node_t candidate) const noexcept {
        return ranks_.full() && candidate.distance_bits() > ranks_.radius();
    }

    /** Hands @p emit the nearest @p count entries ascending, a warp-width per call, rank
     *  @b base+lane on each lane; ranks past the candidates are sentinels. The ranks are the
     *  container itself, so nothing is folded. */
    template <typename emit_type_>
    __device__ __forceinline__ void for_each_rank(u32_t count, u32_t lane, emit_type_&& emit) const noexcept {
        // Slot by slot at compile time keeps every rung's candidates in registers, which a rolled
        // loop over runtime ranks spills to local memory.
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (slot * warp_t::width_k < count) emit(slot * warp_t::width_k + lane, ranks_.entries_[slot]);
        for (u32_t base = slots_ * warp_t::width_k; base < count; base += warp_t::width_k)
            emit(base + lane, scored_node_t::sentinel());
    }
};

/** The arena frontier: the de-capped twin - a register head (the exact nearest two lane-widths,
 *  absorbing every per-hop merge at constant cost) plus an arena tail taking the
 *  head's spill. The frontier refills from the tail when the head's unexpanded
 *  entries drain, so expansion follows discovery order past the head's width, and
 *  the walk retires after @c expansion_target_ expansions instead of on a radius test - the
 *  expansion budget IS the ef semantic and the horizon. */
template <typename warp_type_, typename node_id_type_>
struct vram_arena_frontier {
    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    vram_sorted_candidates<2, warp_t, node_id_type_> head_;
    vram_tail_candidates<warp_t, node_id_type_> tail_;
    u32_t expansion_target_ = 0;
    u32_t expansions_ = 0;

    /** Times a full tail kept only what can still rank among the expansion target. */
    u32_t compactions_ = 0;

    /** Entries tied at that radius a compacted tail still could not fit. */
    u32_t dropped_ = 0;

    __device__ __forceinline__ void clear(u32_t) noexcept {
        head_.clear();
        tail_.size_ = 0;
        expansions_ = 0;
    }

    /** Grants a further walk over what this frontier holds its own expansion budget. */
    __device__ __forceinline__ void renew() noexcept { expansions_ = 0; }

    __device__ __forceinline__ u32_t compactions() const noexcept { return compactions_; }
    __device__ __forceinline__ u32_t dropped() const noexcept { return dropped_; }

    /** Whether the head already holds the ID @p staged_id names on this lane, every lane taking
     *  part. */
    __device__ __forceinline__ bool holds(node_id_type_ staged_id, u32_t lane) const noexcept {
        return head_.deduplicate(scored_node_t::make(0, staged_id), lane) == scored_node_t::sentinel();
    }
    __device__ __forceinline__ void insert_batch(scored_node_t staged, u32_t lane) noexcept {
        // When every staged entry sits at or past the head's radius the merge displaces
        // nothing - the batch goes straight to the tail, which by construction only holds
        // entries farther than everything head-resident, and staged >= radius keeps that.
        scored_node_t const spilled = head_.full() && !warp_t::ballot(staged.distance_bits() < head_.radius())
                                          ? staged
                                          : head_.insert_batch_spilling(staged, lane);
        spill(spilled, lane);
    }

    /** Appends a lane-width to the tail; a full tail first keeps what can still rank among the
     *  walk's expansion target - every result and every entry its remaining expansions can take -
     *  counted, as are the entries it still cannot fit. */
    __device__ __forceinline__ void spill(scored_node_t value, u32_t lane) noexcept {
        bool const full = tail_.size_ + warp_t::width_k > tail_.capacity_;
        if (full) keep_nearest(expansion_target_, lane);
        compactions_ += full;
        dropped_ += tail_.append(value, lane);
    }
    __device__ __forceinline__ scored_node_t pop_nearest(u32_t lane) noexcept {
        // A consumed expansion budget IS the frontier's edge: reporting starvation here
        // routes the walk into its drained-frontier exit, where exhausted() seals it.
        if (exhausted()) return scored_node_t::sentinel();
        scored_node_t nearest = head_.nearest_unexpanded();
        if (nearest == scored_node_t::sentinel()) {
            refill(lane);
            nearest = head_.nearest_unexpanded();
        }
        if (!(nearest == scored_node_t::sentinel())) {
            head_.mark_expanded_at(nearest);
            ++expansions_;
        }
        return nearest;
    }
    __device__ __forceinline__ bool exhausted() const noexcept { return expansions_ >= expansion_target_; }
    __device__ __forceinline__ bool beyond_radius(scored_node_t) const noexcept { return false; }

    /** The tail keeps every entry the walk scores, so no rank bounds what enters. */
    __device__ __forceinline__ scored_node_t worst() const noexcept { return scored_node_t::sentinel(); }

    /** Promotes the nearest unexpanded tail entries into the head. Every tail entry is farther
     *  than everything head-resident - that is the spill invariant - so a merge could never admit
     *  them: instead the drained head, all expanded by now, spills wholesale to the tail for the
     *  final compaction, and the champions take it over as the fresh frontier. Lane champions are
     *  not the global top lane-width, but refills recur on every drain, so promotion order only
     *  affects scheduling, never membership. */
    __device__ __forceinline__ void refill(u32_t lane) noexcept {
        scored_node_t best = scored_node_t::sentinel();
        u32_t best_index = 0;
        for (u32_t index = lane; index < tail_.size_; index += warp_t::width_k) {
            scored_node_t const entry = tail_.entries_[index];
            if (!entry.expanded() && entry < best) best = entry, best_index = index;
        }
        warp_t::sync();
        if (!warp_t::ballot(!(best == scored_node_t::sentinel()))) return;
        // Promotion tombstones the source slot, so every ID lives in exactly one place:
        // promoted IDs exist only in the head until the head itself re-spills them - once.
        // Each lane's best comes from its own stride residues, so the stores never clash.
        if (!(best == scored_node_t::sentinel())) tail_.entries_[best_index] = scored_node_t::sentinel();
        spill(head_.entries_[0], lane);
        spill(head_.entries_[1], lane);
        head_.clear();
        head_.insert_batch(best, lane);
    }

    /** Drops the tail entries that cannot rank among the nearest @p count of head and tail: a radix
     *  select finds the count-th smallest distance key, eight bits a pass from the top, and every
     *  tail entry at or below it stays, ties included, so the rounds still order them by full word.
     *  Each lane tallies the digits whose low bits are its own index, by ballots, so the histogram
     *  never leaves registers. */
    __device__ __forceinline__ void keep_nearest(u32_t count, u32_t lane) noexcept {
        constexpr u32_t lane_bits_k = warp_t::width_k == 64 ? 6 : 5;
        constexpr u32_t high_bits_k = 8 - lane_bits_k;
        constexpr u32_t bins_k = 1u << high_bits_k;
        u32_t prefix = 0, prefix_mask = 0, rank = count;
        for (u32_t pass = 0; pass != 4; ++pass) {
            u32_t const shift = 24 - 8 * pass;
            u32_t bins[bins_k] = {};
            // The head's two slots tally as the first two batches,
            // then the tail a warp-width at a time.
            for (u32_t base = 0; base < 2 * warp_t::width_k + tail_.size_; base += warp_t::width_k) {
                u32_t const index = base + lane - 2 * warp_t::width_k;
                scored_node_t const entry = base == 0                 ? head_.entries_[0]
                                            : base == warp_t::width_k ? head_.entries_[1]
                                            : index < tail_.size_     ? tail_.entries_[index]
                                                                      : scored_node_t::sentinel();
                u32_t const key = entry.distance_bits(), digit = (key >> shift) & 0xFFu;
                typename warp_t::mask_t own = warp_t::ballot(!(entry == scored_node_t::sentinel()) &&
                                                             (key & prefix_mask) == prefix);
#pragma unroll
                for (u32_t bit = 0; bit != lane_bits_k; ++bit) {
                    typename warp_t::mask_t const set = warp_t::ballot((digit >> bit) & 1u);
                    own &= (lane >> bit) & 1u ? set : ~set;
                }
                typename warp_t::mask_t high[high_bits_k];
#pragma unroll
                for (u32_t bit = 0; bit != high_bits_k; ++bit)
                    high[bit] = warp_t::ballot((digit >> (lane_bits_k + bit)) & 1u);
#pragma unroll
                for (u32_t bin = 0; bin != bins_k; ++bin) {
                    typename warp_t::mask_t matching = own;
#pragma unroll
                    for (u32_t bit = 0; bit != high_bits_k; ++bit)
                        matching &= (bin >> bit) & 1u ? high[bit] : ~high[bit];
                    bins[bin] += warp_t::count(matching);
                }
            }

            // Digits ascend bin-major, lane-minor;
            // the warp-uniform guard keeps every vote full-mask.
            u32_t below = 0;
            bool found = false;
#pragma unroll
            for (u32_t bin = 0; bin != bins_k; ++bin) {
                u32_t inclusive = bins[bin];
#pragma unroll
                for (u32_t stride = 1; stride != warp_t::width_k; stride <<= 1) {
                    u32_t const lower = warp_t::shuffle_up(inclusive, stride);
                    if (lane >= stride) inclusive += lower;
                }
                u32_t const total = warp_t::shuffle(inclusive, warp_t::width_k - 1);
                if (!found && below + total >= rank) {
                    u32_t const owner = warp_t::first(warp_t::ballot(below + inclusive >= rank));
                    below += warp_t::shuffle(inclusive - bins[bin], owner);
                    prefix |= ((bin << lane_bits_k) | owner) << shift;
                    found = true;
                }
                if (!found) below += total;
            }
            if (!found) return; // The walk saw fewer than count entries, so every one ranks.
            prefix_mask |= 0xFFu << shift;
            rank -= below;
        }

        // In place: a kept entry lands at or before its own slot,
        // after the ballot over its batch's loads.
        u32_t kept = 0;
        for (u32_t base = 0; base < tail_.size_; base += warp_t::width_k) {
            u32_t const index = base + lane;
            scored_node_t const entry = index < tail_.size_ ? tail_.entries_[index] : scored_node_t::sentinel();
            bool const keep = !(entry == scored_node_t::sentinel()) && entry.distance_bits() <= prefix;
            typename warp_t::mask_t const keeping = warp_t::ballot(keep);
            if (keep) tail_.entries_[kept + warp_t::count(keeping & warp_t::below(lane))] = entry;
            kept += warp_t::count(keeping);
        }
        tail_.size_ = kept;
        warp_t::sync();
    }

    /** Hands @p emit the nearest @p count entries of everything the walk has seen - the last head
     *  generation plus every tail entry - ascending, a warp-width per call, rank @b base+lane on
     *  each lane; ranks past what the walk saw are sentinels. Each round folds every entry ranked
     *  after the previous round's last into a fresh two-slot candidates; past one round the tail
     *  first keeps only the entries that can rank, so the rounds scale with @p count, not the walk.
     *  Promotion tombstones its tail slot, but an ID re-scored past a visited-set saturation can
     *  sit in the tail twice, so each round keeps one copy against itself and the round before.
     *  Call once, after the walk. */
    template <typename emit_type_>
    __device__ __forceinline__ void for_each_rank(u32_t count, u32_t lane, emit_type_&& emit) noexcept {
        using round_t = vram_sorted_candidates<2, warp_t, node_id_type_>;
        if (count > round_t::capacity_k) keep_nearest(count, lane);
        scored_node_t last = scored_node_t::sentinel();
        round_t previous;
        previous.clear();
        for (u32_t base = 0; base < count; base += round_t::capacity_k) {
            round_t round;
            round.clear();
            auto const fold = [&](scored_node_t entry) noexcept {
                entry = base == 0 || last < entry ? entry : scored_node_t::sentinel();
                round.insert_batch(round.deduplicate(previous.deduplicate(entry, lane), lane), lane);
            };
            // A short round means the walk saw nothing more: every later rank is a sentinel.
            if (base == 0 || !(last == scored_node_t::sentinel())) {
                fold(head_.entries_[0]);
                fold(head_.entries_[1]);
                for (u32_t tail_base = 0; tail_base < tail_.size_; tail_base += warp_t::width_k) {
                    u32_t const index = tail_base + lane;
                    fold(index < tail_.size_ ? tail_.entries_[index] : scored_node_t::sentinel());
                }
            }
            emit(base + lane, round.entries_[0]);
            emit(base + warp_t::width_k + lane, round.entries_[1]);
            last = round.rank_entry(round_t::capacity_k - 1);
            previous = round;
        }
    }
};

/**
 *  @brief Exact visited set over a warp-private arena span: all-ones empty sentinel, multiplicative
 *      hash, linear probing - the CPU growing hash set's discipline, with the capacity sized ahead
 *      by the host so it never grows.
 *
 *  The @b only kind of membership structure this codebase allows: lossy variants are banned
 *  outright, their false negatives silently invert the recall-vs-expansion curve.
 *
 *  Lanes carrying the same ID in one call collapse to their lowest lane via value matching, so a
 *  batch can never stage intra-batch duplicates; the compare-exchange only arbitrates two distinct
 *  IDs landing on one empty slot - scratch bookkeeping, not part of the graph's all-RMW protocol. A
 *  saturated set - the probe budget exhausted on a full neighborhood - reports the ID fresh, so it
 *  re-scores rather than starving the walk, counts every such claim as the receipt the telemetry
 *  reports, and leaves the walk to skip IDs its frontier already holds.
 */
template <typename warp_type_, typename node_id_type_>
struct vram_visited_set {
    using warp_t = warp_type_;

    /** Packed @b {epoch,ID} slots: an entry is live only when its epoch matches this walk's, so
     *  "resetting" is bumping one register - the host zeroes the arena once per launch batch, and
     *  epoch zero marks never-used. 16-byte aligned, since a probe loads the aligned pair of slots
     *  it lands in at once. A @c u64_t ID takes a slot of two words, the epoch and then the ID, and
     *  a probe loads that one slot. */
    u64_t* slots_;

    /** One field carries the whole shape: the slot count is a power of two by
     *  construction in @c arena_shape_for, so the probe mask derives as
     *  @b 0xFFFFFFFF>>hash_shift_ and no capacity is stored. */
    u32_t hash_shift_;
    u32_t epoch_;

    /** Claims that ran out of probes and re-scored; warp-uniform, so lane zero reports it. */
    u32_t saturations_ = 0;

    /** Probe rounds before a claim counts as saturated: log2 of the slots, the expected longest
     *  probe of linear probing at the half load the host sizes the set for; a round reads an
     *  aligned pair of slots, so a claim scans up to twice as many. */
    __device__ __forceinline__ u32_t probe_budget() const noexcept { return 32 - hash_shift_; }

    /** Marks @p node_id visited; @c true when it was fresh. A lane with nothing to claim passes the
     *  sentinel, which no walk visits. */
    __device__ __forceinline__ bool test_and_set(node_id_type_ node_id, u32_t lane) noexcept {
        constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
        bool const active = node_id != unknown_k;
        // Warp-uniform: an idle call skips the matching votes, a ballot per distinct value on HIP.
        if (!warp_t::any(active)) return false;
        typename warp_t::mask_t const peers = warp_t::match_any(active ? node_id : unknown_k ^ lane);
        bool fresh = active && (peers & warp_t::below(lane)) == 0;
        bool probing = fresh;
        u32_t const mask = 0xFFFFFFFFu >> hash_shift_;
        // The slice is warp-private, so the only contention is two lanes of THIS warp
        // landing on one empty slot in the same round - arbitrated in registers, no
        // atomics: the lowest rival lane stores, the warp barrier publishes the store,
        // and losers re-read the slot, see a foreign epoch-current ID, and probe on.
        if constexpr (std::is_same_v<node_id_type_, u64_t>) {
            u32_t slot = static_cast<u32_t>((node_id * 0x9E3779B97F4A7C15ull) >> 32) >> hash_shift_;
            ulonglong2* const pairs = reinterpret_cast<ulonglong2*>(slots_);
            for (u32_t round = 0, budget = probe_budget(); warp_t::any(probing) && round != budget; ++round) {
                // One 16-byte load covers the probed slot, its epoch word and its ID word.
                ulonglong2 const pair = probing ? pairs[slot] : ulonglong2 {};
                bool claiming = false;
                if (probing) {
                    if (pair.x != epoch_) claiming = true;
                    else if (pair.y == node_id) fresh = probing = false;
                    else slot = (slot + 1) & mask;
                }
                typename warp_t::mask_t const rivals = warp_t::match_any(claiming ? slot : 0xFFFFFFFFu ^ lane);
                if (claiming && (rivals & warp_t::below(lane)) == 0)
                    pairs[slot] = make_ulonglong2(epoch_, node_id), probing = false;
                warp_t::sync();
            }
        }
        else {
            u64_t const claim = (static_cast<u64_t>(epoch_) << 32) | node_id;
            u32_t slot = (node_id * 0x9E3779B9u) >> hash_shift_;
            ulonglong2 const* const pairs = reinterpret_cast<ulonglong2 const*>(slots_);
            for (u32_t round = 0, budget = probe_budget(); warp_t::any(probing) && round != budget; ++round) {
                // One 16-byte load covers the probed slot's aligned pair, so an even-slot collision
                // resolves this round.
                ulonglong2 const pair = probing ? pairs[slot >> 1] : ulonglong2 {};
                u64_t const observed = slot & 1 ? pair.y : pair.x;
                bool claiming = false;
                if (probing) {
                    // Visited this walk, or a stale or never-used slot to claim, then the same for
                    // the pair's odd slot.
                    if (observed == claim) fresh = probing = false;
                    else if ((observed >> 32) != epoch_) claiming = true;
                    else if (slot & 1) slot = (slot + 1) & mask;
                    else if (pair.y == claim) fresh = probing = false;
                    else if ((pair.y >> 32) != epoch_) slot |= 1, claiming = true;
                    else slot = (slot + 2) & mask;
                }
                typename warp_t::mask_t const rivals = warp_t::match_any(claiming ? slot : 0xFFFFFFFFu ^ lane);
                if (claiming && (rivals & warp_t::below(lane)) == 0) slots_[slot] = claim, probing = false;
                warp_t::sync();
            }
        }
        // Saturation: the ID re-scores as fresh, counted; the frontier keeps one copy of it.
        saturations_ += warp_t::count(warp_t::ballot(probing));
        warp_t::sync();
        return fresh;
    }
};

/**
 *  @brief Typed view over one node's neighbors - the only legal way to touch slots that
 *      concurrent warps may be reading, over the warp policy's device-scope atomics.
 *
 *  Slot 0 is the combined lock and first-neighbor word: the all-ones sentinel while unpublished
 *  or locked, the node's own ID when published empty, otherwise the first neighbor. Acquisition
 *  is one exchange that never waits - spinning against a descheduled owner warp would livelock -
 *  so a held row is revisited after other work, and an unlinked neighbor is the telemetered cost.
 */
template <typename warp_type_, typename node_id_type_>
struct vram_neighbors_ref {
    using warp_t = warp_type_;
    node_id_type_* row_; // Non-const even for readers: device atomic references cannot bind const words pre-P3323.
    u32_t capacity_;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }

    /** Acquire read of slot @p index, pairing with the lock holder's release fence so an ID read
     *  here comes with its node's payload; slot 0 belongs to @c peek_first. */
    __device__ __forceinline__ node_id_type_ load_slot(u32_t index) const noexcept {
        return warp_t::load_acquire(row_[index]);
    }

    /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current lock
     *  holder. */
    __device__ __forceinline__ void store_slot(u32_t index, node_id_type_ value) const noexcept {
        warp_t::store_relaxed(row_[index], value);
    }

    /** Acquire-load of the lock word: the sentinel means unpublished or locked, any other value is
     *  the first neighbor ID with the last unlocker's writes visible. */
    __device__ __forceinline__ node_id_type_ peek_first() const noexcept { return warp_t::load_acquire(row_[0]); }

    /** One acquisition attempt: exchanges the sentinel in with acquire order; an exchange that
     *  receives the sentinel back did @b not acquire. @return The displaced first-neighbor value,
     *  or the sentinel. */
    __device__ __forceinline__ node_id_type_ lock_if_free() const noexcept {
        return warp_t::exchange_acquire(row_[0], std::numeric_limits<node_id_type_>::max());
    }

    /** Release-stores @p first_value into slot 0 - unlocking and simultaneously (re)publishing the
     *  edited neighbors to every future acquiring reader. */
    __device__ __forceinline__ void unlock(node_id_type_ first_value) const noexcept {
        warp_t::store_release(row_[0], first_value);
    }
};

/** Which traversals a telemetry flush folds: inserting ones also own the lock and reselection
 *  counters. */
enum class vram_traversal_t : bool { searching_k, inserting_k };

/** Per-warp register telemetry, folded into the device counters once per node; 32-bit in registers,
 *  since every counter is bounded by hops times connectivity, widened at the flush. */
struct vram_telemetry_t {
    u32_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, reselections = 0, restarts = 0, locked_drops = 0;
    u32_t local_scored = 0, remote_scored = 0;
    u32_t tail_compactions = 0, tail_dropped = 0, visited_saturations = 0;
    u32_t changed_nodes = 0;

    /** Folds the live counters into the device block; the search instantiation skips the lock and
     *  reselection atomics it never touches. */
    template <typename warp_type_, vram_traversal_t traversal_>
    __device__ __forceinline__ void flush(vram_telemetry_block_t* counters, u32_t lane) noexcept {
        if (lane != 0) return;
        // Tallies alone: the drain after the launch publishes them, so every add is relaxed.
        auto const add = [](u64_t& counter, u32_t value) noexcept {
            warp_type_::fetch_add_relaxed(counter, static_cast<u64_t>(value));
        };
        add(counters->hops, hops);
        add(counters->scored, scored);
        add(counters->restarts, restarts);
        add(counters->locked_drops, locked_drops);
        if (changed_nodes) add(counters->changed_nodes, changed_nodes);
        if (local_scored | remote_scored) {
            add(counters->local_scored, local_scored);
            add(counters->remote_scored, remote_scored);
        }
        if (tail_compactions | tail_dropped | visited_saturations) {
            add(counters->tail_compactions, tail_compactions);
            add(counters->tail_dropped, tail_dropped);
            add(counters->visited_saturations, visited_saturations);
        }
        if constexpr (traversal_ == vram_traversal_t::searching_k) return;
        add(counters->lock_attempts, lock_attempts);
        add(counters->lock_skips, lock_skips);
        add(counters->reselections, reselections);
    }
};

/** What a walk asks of its frontier, the register ladder's or the arena's: start and renew a walk,
 *  take a scored batch, hand back the nearest unexpanded entry, tell a staged ID it already holds,
 *  and count what its tail compacted and dropped. */
template <typename frontier_>
concept vram_frontier = requires(frontier_& frontier, frontier_ const& held, typename frontier_::scored_node_t entry,
                                 u32_t lane) {
    frontier.clear(lane);
    frontier.renew();
    frontier.insert_batch(entry, lane);
    { frontier.pop_nearest(lane) } -> std::same_as<typename frontier_::scored_node_t>;
    { held.holds(entry.id(), lane) } -> std::same_as<bool>;
    { held.exhausted() } -> std::same_as<bool>;
    { held.beyond_radius(entry) } -> std::same_as<bool>;
    { held.compactions() } -> std::same_as<u32_t>;
    { held.dropped() } -> std::same_as<u32_t>;
};

/** One walking warp's state: the candidates, the random stream, the running telemetry, and the
 *  shared scratch past its query span, where a rescoring hands its fold what the fold keeps. */
template <vram_frontier frontier_>
struct vram_warp_context {
    frontier_ frontier;
    splitmix64_t random_generator;
    vram_telemetry_t telemetry;
    byte_t* scratch = nullptr;

    /** Rows each round expands, from @ref walk_width; one, the plain greedy walk, for inserts and
     *  refines. */
    u32_t walk_width = 1;
};

/**
 *  @brief Expands @p first, whose lock word @p first_value the walk already read, with up to
 *      @p width - 1 further candidates inside the radius, as one round.
 *
 *  Every row's slots load in one window, the rows claim in pop order, and every @p round_rows of
 *  fresh IDs, a warp's width or more, reach @p score through the @p staged buffer, which holds
 *  @p round_rows and a warp's width more, and whose @p pending entries past them carry across
 *  rounds.
 *
 *  Lane @b r reads row @b r 's lock word once and the warp shuffles it, so no two lanes branch on
 *  racing reads; a locked further row is held when the walk holds none, else counted dropped.
 *  The rows' slot loads fit @c vram_max_gathers_in_flight_k per lane, which bounds the round, and a
 *  row wider than that window loads in several.
 *
 *  @return @c sated_k when a scored batch sated the fold.
 */
template <typename warp_type_, typename node_id_type_, vram_frontier frontier_, typename storage_,
          typename score_type_>
__device__ __forceinline__ walk_demand_t vram_expand_round( //
    frontier_& frontier, vram_visited_set<warp_type_, node_id_type_>& visited, vram_telemetry_t& telemetry,
    storage_ const& storage, vram_scored_node<node_id_type_> first, node_id_type_ first_value, u32_t width,
    node_id_type_ first_invisible_id, vram_scored_node<node_id_type_>& held, u32_t& held_at_hop, node_id_type_* staged,
    u32_t& pending, u32_t round_rows, u32_t lane, score_type_&& score) noexcept {

    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    using row_t = vram_neighbors_ref<warp_t, node_id_type_>;
    constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
    constexpr u32_t window_k = vram_max_gathers_in_flight_k;
    u32_t const connectivity = storage.connectivity;

    // Lane `r` keeps the round's row `r`; the radius only tightens, so a popped row past it ends
    // the round.
    u32_t const rows_cap = max_of(1u, min_of(width, window_k / divide_round_up(connectivity, warp_t::width_k)));
    scored_node_t row_entry = lane == 0 ? first : scored_node_t::sentinel();
    u32_t rows = 1;
    while (rows < rows_cap) {
        scored_node_t const next = frontier.pop_nearest(lane);
        if (next == scored_node_t::sentinel() || frontier.beyond_radius(next)) break;
        if (lane == rows) row_entry = next;
        ++telemetry.hops, ++rows;
    }
    node_id_type_ const row_value = lane == 0 ? first_value
                                    : lane < rows
                                        ? row_t {storage.neighbors_for(row_entry.id()), connectivity}.peek_first()
                                        : unknown_k;
    // The lane barrier orders every lane's slot loads after the readers' acquires.
    warp_t::sync();
    typename warp_t::mask_t const locked = warp_t::ballot(lane < rows && row_value == unknown_k);
    if (locked) {
        bool const holding = held == scored_node_t::sentinel();
        if (holding) held = shuffle_from<warp_t>(row_entry, warp_t::first(locked)), held_at_hop = telemetry.hops;
        telemetry.locked_drops += warp_t::count(locked) - holding;
    }

    // Live rows after the first, walked lowest lane first; every bound and step is warp-uniform.
    typename warp_t::mask_t rows_left = warp_t::ballot(lane < rows && row_value != unknown_k) &
                                        ~typename warp_t::mask_t {1};
    u32_t row = 0, slot_base = 0;
    bool more = true;
    while (more) {
        node_id_type_ neighbor_ids[window_k], owner_ids[window_k];
        u32_t loads = 0;
#pragma unroll
        for (u32_t load = 0; load != window_k; ++load) {
            if (!more) break;
            node_id_type_ const owner = warp_t::shuffle(row_entry.id(), row);
            node_id_type_ const word = warp_t::shuffle(row_value, row);
            u32_t const slot = slot_base + lane;
            neighbor_ids[load] = slot == 0 ? word
                                 : slot < connectivity
                                     ? row_t {storage.neighbors_for(owner), connectivity}.load_slot(slot)
                                     : unknown_k;
            owner_ids[load] = owner;
            ++loads;
            slot_base += warp_t::width_k;
            if (slot_base < connectivity) continue;
            more = rows_left != 0;
            if (more) row = warp_t::first(rows_left), rows_left &= rows_left - 1;
            slot_base = 0;
        }
        // Rolled, so the claim and the scoring it may call inline once per walk, not once per load.
#pragma unroll 1
        for (u32_t load = 0; load != window_k; ++load) {
            if (load == loads) break;
            node_id_type_ const neighbor_id = neighbor_ids[load];
            // The bound never exceeds the ID space, so it rejects the sentinel too.
            bool const usable = neighbor_id < first_invisible_id && neighbor_id != owner_ids[load];
            bool const claimed = visited.test_and_set(usable ? neighbor_id : unknown_k, lane);
            // Past a saturation a claim re-scores, so an ID the frontier or this staging already
            // holds stages no second copy; every lane takes part in the frontier's collective scan.
            bool repeated = visited.saturations_ && frontier.holds(neighbor_id, lane);
            for (u32_t index = 0; visited.saturations_ && index != pending; ++index)
                repeated |= staged[index] == neighbor_id;
            bool const fresh = claimed && !repeated;
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
            if (fresh) staged[pending + warp_t::count(fresh_mask & warp_t::below(lane))] = neighbor_id;
            warp_t::sync();
            pending += warp_t::count(fresh_mask);
            if (pending < round_rows) continue;
            if (score(round_rows) == walk_demand_t::sated_k) return walk_demand_t::sated_k;
            // Every lane read the scored IDs before the carry moves down over them.
            warp_t::sync();
            pending -= round_rows;
            node_id_type_ const carried = lane < pending ? staged[round_rows + lane] : unknown_k;
            warp_t::sync();
            if (lane < pending) staged[lane] = carried;
            warp_t::sync();
        }
    }
    return walk_demand_t::continue_k;
}

/** Runtime-strided view over the warp's vector cache - rows of @c row_fragments_ fragments each,
 *  aliasing the scratch slice, in shared memory or the warp's arena slice. In the arena the cache
 *  is by reference: a staged row is the storage row its ID names, read in place, not a copy. */
template <typename fragment_type_>
struct vram_vector_cache {
    fragment_type_* data_;
    u32_t row_fragments_;

    /** The staged IDs a by-reference cache resolves its rows through, or null when it copies them. */
    u32_t* ids_ = nullptr;

    __device__ __forceinline__ fragment_type_* row(u32_t row_index) const noexcept {
        return data_ + row_index * row_fragments_;
    }
    __device__ __forceinline__ fragment_type_& at(u32_t row_index, u32_t fragment_index) const noexcept {
        return row(row_index)[fragment_index];
    }

    /** Row @p row_index as a staging or @ref place_row left it. */
    template <typename storage_>
    __device__ __forceinline__ fragment_type_ const* staged_row(storage_ const& storage,
                                                                u32_t row_index) const noexcept {
        return ids_ ? storage.vector_span_for(ids_[row_index]) : row(row_index);
    }

    /** Puts the storage row of @p node_id at @p row_index for one warp, which syncs before reading it. */
    template <typename warp_, typename storage_>
    __device__ __forceinline__ void place_row(storage_ const& storage, u32_t row_index, u32_t node_id,
                                              u32_t lane) const noexcept {
        if (ids_) {
            if (lane == 0) ids_[row_index] = node_id;
            return;
        }
        for (u32_t fragment = lane; fragment < row_fragments_; fragment += warp_::width_k)
            at(row_index, fragment) = storage.vector_for(node_id, fragment);
    }
};

/** Warp-cooperative ascending buffer of scored nodes over a shared-memory span: chunks arrive
 *  pre-sorted from the lane-wide bitonic network and merge in place, entries past @c capacity_
 *  drop. */
template <typename warp_type_, typename node_id_type_>
struct vram_sorted_buffer {
    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_;

    __device__ __forceinline__ scored_node_t at(u32_t rank) const noexcept { return entries_[rank]; }

    __device__ __forceinline__ void insert_batch(scored_node_t value, u32_t count, u32_t lane) noexcept {
        bool const incoming = lane < count;
        u32_t incoming_rank = incoming ? lane : 0;
        for (u32_t i = 0; i != size_; ++i) // Warp-uniform bound: every lane walks the span.
            incoming_rank += incoming && entries_[i].distance_bits() <= value.distance_bits();
        // Existing entries shift by the number of incoming values strictly ahead of them;
        // ties resolve the other way than above, so no two final positions collide. Shifts only
        // move entries up, so moving the last lane-width first never overwrites an unmoved one.
        for (u32_t batches = divide_round_up(size_, warp_t::width_k); batches; --batches) {
            u32_t const index = (batches - 1) * warp_t::width_k + lane;
            bool const active = index < size_;
            scored_node_t const moved = active ? entries_[index] : scored_node_t::sentinel();
            u32_t shift = 0;
            for (u32_t source = 0; source != count; ++source) // Warp-uniform: shuffles inside.
                shift += (shuffle_from<warp_t>(value, source).distance_bits() < moved.distance_bits()) && active;
            warp_t::sync(); // The whole lane-width is in registers before any of its writes lands.
            if (active && index + shift < capacity_) entries_[index + shift] = moved;
            warp_t::sync();
        }
        if (incoming && incoming_rank < capacity_) entries_[incoming_rank] = value;
        warp_t::sync();
        size_ = min_of(size_ + count, capacity_);
    }
};

/** Spans backing one warp's neighbor re-selection, carved after the vector cache: the selected
 *  target IDs, the ascending per-candidate scores, the kept cache positions, and the cached
 *  candidates' IDs and distances. */
template <typename node_id_type_>
struct vram_selection_scratch {
    using scored_node_t = vram_scored_node<node_id_type_>;
    node_id_type_* selected_ids;
    scored_node_t* scored_entries;
    u32_t* kept_positions;

    /** Each cached candidate's distance to the query, by cache row, staged once so the selection
     *  loop reads memory, not register candidates it would unroll to index. */
    f32_t* candidate_distances;

    /** The node ID each cache row holds, by cache row, which a by-reference cache resolves its rows
     *  through. */
    node_id_type_* candidate_ids = nullptr;

    /** Carves the five spans of @p rows entries each past the last of @p cache's @p rows rows. */
    template <typename fragment_type_>
    static __device__ __forceinline__ vram_selection_scratch carve(vram_vector_cache<fragment_type_> cache,
                                                                   u32_t rows) noexcept {
        return carve(reinterpret_cast<byte_t*>(cache.row(rows)), rows);
    }

    /** Carves the five spans of @p rows entries each from @p first on, 16-byte aligned, widest
     *  alignment first: the ascending candidate scores, the two node-ID spans, then the kept
     *  positions and the staged distances. */
    static __device__ __forceinline__ vram_selection_scratch carve(byte_t* first, u32_t rows) noexcept {
        scored_node_t* const scored_entries = reinterpret_cast<scored_node_t*>(first);
        node_id_type_* const selected_ids = reinterpret_cast<node_id_type_*>(scored_entries + rows);
        node_id_type_* const candidate_ids = selected_ids + rows;
        u32_t* const kept_positions = reinterpret_cast<u32_t*>(candidate_ids + rows);
        return {selected_ids, scored_entries, kept_positions, reinterpret_cast<f32_t*>(kept_positions + rows),
                candidate_ids};
    }

    /** The span past the five, where a design keeps its per-candidate state. */
    __device__ __forceinline__ byte_t* end(u32_t rows) const noexcept {
        return reinterpret_cast<byte_t*>(candidate_distances + rows);
    }
};

/** Layout of one warp's slice of the traversal arena, computed on the host and carried by the
 *  views: the exact visited set, the wide frontier's spill tail when it has one, then a builder's
 *  scratch when shared memory cannot fit its rows. Its spans are as wide as node IDs. */
struct vram_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    /** Bytes of a builder's scratch - query span, candidate cache and re-selection spans - kept
     *  here instead of dynamic shared memory, or zero when shared memory fits it. */
    u32_t scratch_bytes;

    /** Bytes of one warp's slice, a multiple of 16 so every visited span starts on the boundary its
     *  paired probes load from; the slices together may pass 4 GB, one alone never does. */
    template <typename node_id_type_>
    __device__ __forceinline__ u32_t slice_bytes() const noexcept {
        u32_t const bytes = visited_slots * gpu_visited_slot_bytes<node_id_type_> +
                            tail_entries * sizeof(vram_scored_node<node_id_type_>) + scratch_bytes;
        return (bytes + 15u) & ~15u;
    }
    template <typename node_id_type_>
    __device__ __forceinline__ u64_t* visited_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(vram_row(base, warp_index, slice_bytes<node_id_type_>()));
    }
    template <typename node_id_type_>
    __device__ __forceinline__ vram_scored_node<node_id_type_>* tail_span(byte_t* base,
                                                                          u32_t warp_index) const noexcept {
        return reinterpret_cast<vram_scored_node<node_id_type_>*>(
            visited_span<node_id_type_>(base, warp_index) +
            visited_slots * (gpu_visited_slot_bytes<node_id_type_> / sizeof(u64_t)));
    }
    template <typename node_id_type_>
    __device__ __forceinline__ byte_t* scratch_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<byte_t*>(tail_span<node_id_type_>(base, warp_index) + tail_entries);
    }
};

static_assert(sizeof(gpu_arena_shape_t) == sizeof(vram_arena_shape_t));

/** The device twin of the host's arena shape, field by field. */
constexpr vram_arena_shape_t vram_view(gpu_arena_shape_t shape) noexcept {
    return {shape.visited_slots, shape.visited_hash_shift, shape.tail_entries, shape.scratch_bytes};
}

/** One warp's slice of a launch: its coordinates, its scratch - dynamic shared memory, or its arena
 *  slice's cache span when shared memory cannot fit it - and its arena visited span, the shared
 *  prologue of every traversal kernel. */
struct vram_warp_slice_t {
    u32_t lane, warp, warp_index;
    byte_t* scratch;
    u64_t* visited_slots;
};

/** Carves the calling warp's slice out of @p view and the block's @p dynamic_scratch; the arena
 *  base is allocation-aligned and every slice a multiple of 16 bytes, so each visited span takes
 *  paired 16-byte probes. */
template <typename warp_type_, typename node_id_type_, typename view_>
__device__ __forceinline__ vram_warp_slice_t vram_carve_slice(view_ const& view, byte_t* dynamic_scratch) noexcept {
    u32_t const warp = threadIdx.x / warp_type_::width_k, lane = threadIdx.x % warp_type_::width_k;
    u32_t const warp_index = blockIdx.x * vram_warps_per_block_k + warp;
    byte_t* const scratch = view.arena_shape.scratch_bytes
                                ? view.arena_shape.template scratch_span<node_id_type_>(view.arena, warp_index)
                                : dynamic_scratch + warp * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch,
            view.arena_shape.template visited_span<node_id_type_>(view.arena, warp_index)};
}

} // namespace unum::usearch::utilities
