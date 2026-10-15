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
 *  - Register-resident and arena-backed @b frontiers with append-only tail pools
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

namespace unum::usearch::utilities {

/** Row @p row of a @b [rows,stride] matrix at @p base - the one place a row offset widens past 32
 *  bits, since node IDs times row strides outgrow them. */
template <typename value_type_>
__device__ __forceinline__ value_type_* vram_row(value_type_* base, u32_t row, u32_t stride) noexcept {
    return base + static_cast<std::size_t>(row) * stride;
}

/** The random stream of @p lane walking for @p stream_key - the node it inserts or refines, or the
 *  query it answers - seeded through @ref mix_bits as the host seeds its streams, so no two lanes
 *  or keys share one. */
template <typename warp_type_>
__device__ __forceinline__ splitmix64_t vram_lane_generator(u32_t stream_key, u32_t lane) noexcept {
    return splitmix64_t {mix_bits(static_cast<u64_t>(stream_key) * warp_type_::width_k + lane + 1)};
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
template <typename warp_type_, typename callback_type_>
__device__ __forceinline__ void vram_for_each_link(u32_t const* neighbors, u32_t connectivity, u32_t rows_count,
                                                   u32_t row, u32_t lane, callback_type_&& callback) noexcept {
    u32_t const* const slots = vram_row(neighbors, row, connectivity);
    for (u32_t slot = lane; slot < connectivity; slot += warp_type_::width_k)
        if (u32_t const target = slots[slot]; target != node_id_unknown_k && target != row && target < rows_count)
            callback(target);
}

/**
 *  @brief One traversal candidate packed into eight bytes, ordered by distance as raw
 *      unsigned words: bits [63:33] hold the top 31 bits of the sign-flipped @c f32
 *      distance, bit 32 carries the "already expanded" flag, bits [31:0] the full node ID.
 *
 *  The all-ones pattern is the empty slot: its distance prefix is the flipped NaN that
 *  sorts last, it reads as expanded, and its ID is @c node_id_unknown_k - so
 *  traversal loops need no emptiness guards. Dropping the distance LSB costs one ulp of
 *  ordering precision: pools are "sorted by @c distance_bits(), tie order unspecified",
 *  which is the practical semantic of approximate search either way, and in exchange IDs
 *  keep all 32 bits - the ID ceiling matches the CPU-resident index exactly.
 */
struct vram_scored_node_t {
    u64_t bits_;

    inline static constexpr u64_t expanded_bit_k = u64_t(1) << 32;

    /** Dispensable IDs: everything but the reserved all-ones sentinel. */
    inline static constexpr std::size_t id_capacity_k = (u64_t(1) << 32) - 1u;

    static __device__ __forceinline__ vram_scored_node_t make(distance_t distance, node_id_t node_id) noexcept {
        u32_t flipped = __float_as_uint(distance);
        flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return {(static_cast<u64_t>(flipped >> 1) << 33) | node_id};
    }
    __device__ static constexpr vram_scored_node_t sentinel() noexcept { return {~u64_t(0)}; }

    /** The flag-free comparison currency: 31 ordering bits of the flipped distance. */
    __device__ __forceinline__ u32_t distance_bits() const noexcept { return static_cast<u32_t>(bits_ >> 33); }

    /** Decoded distance, with the truncated ordering LSB reconstructed as zero. */
    __device__ __forceinline__ distance_t distance() const noexcept {
        u32_t flipped = distance_bits() << 1;
        flipped ^= (flipped & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
        return __uint_as_float(flipped);
    }
    __device__ __forceinline__ node_id_t id() const noexcept { return static_cast<node_id_t>(bits_); }
    __device__ __forceinline__ bool expanded() const noexcept { return bits_ & expanded_bit_k; }

    /** Lazily deletes the entry from the frontier; a no-op on the sentinel. */
    __device__ __forceinline__ void mark_expanded() noexcept { bits_ |= expanded_bit_k; }
    __device__ __forceinline__ bool operator<(vram_scored_node_t other) const noexcept { return bits_ < other.bits_; }
    __device__ __forceinline__ bool operator==(vram_scored_node_t other) const noexcept { return bits_ == other.bits_; }
};
static_assert(sizeof(vram_scored_node_t) == 8, "Entries shuffle as one 64-bit word");
static_assert(std::is_trivially_copyable_v<vram_scored_node_t>);

/** Lane shuffles have no user-type overloads - an entry shuffles its raw word. */
template <typename warp_type_>
__device__ __forceinline__ vram_scored_node_t shuffle_xor(vram_scored_node_t value, u32_t lane_mask) noexcept {
    return {warp_type_::shuffle_xor(value.bits_, lane_mask)};
}
template <typename warp_type_>
__device__ __forceinline__ vram_scored_node_t shuffle_from(vram_scored_node_t value, u32_t source_lane) noexcept {
    return {warp_type_::shuffle(value.bits_, source_lane)};
}
template <typename warp_type_>
__device__ __forceinline__ vram_scored_node_t shuffle_up(vram_scored_node_t value, u32_t delta) noexcept {
    return {warp_type_::shuffle_up(value.bits_, delta)};
}

/** Full bitonic sort of one lane-wide entry vector, ascending by lane - fifteen shuffle-exchange
 *  stages on 32 lanes, twenty-one on 64, all in registers. */
template <typename warp_type_>
__device__ __forceinline__ vram_scored_node_t sort_lanes(vram_scored_node_t value, u32_t lane) noexcept {
#pragma unroll
    for (u32_t stage_width = 2; stage_width <= warp_type_::width_k; stage_width <<= 1)
#pragma unroll
        for (u32_t partner_stride = stage_width >> 1; partner_stride; partner_stride >>= 1) {
            vram_scored_node_t const partner = shuffle_xor<warp_type_>(value, partner_stride);
            bool const ascending = (lane & stage_width) == 0;
            bool const lower = (lane & partner_stride) == 0;
            vram_scored_node_t const nearer = value < partner ? value : partner;
            vram_scored_node_t const further = value < partner ? partner : value;
            value = (ascending == lower) ? nearer : further;
        }
    return value;
}

/**
 *  @brief Warp-collective ascending pool of @c width_k x @c slots_ scored entries - the fused
 *      results-and-frontier of one traversal, striped as rank @b r at (slot r/width, lane
 *      r%width), always sorted by @c distance_bits() with unspecified tie order.
 *
 *  Subsumes both host-side traversal containers: the sorted results buffer (the radius
 *  lives at the last rank) and the frontier heap (the nearest unexpanded entry plays
 *  @c top(), the expanded bit plays lazy deletion). Two register-residency invariants are
 *  part of the contract: every slot index is compile-time after unrolling - a single
 *  runtime-indexed access spills the whole pool to local memory - and batch insertion
 *  goes through the fold-merge network below, never a full re-sort.
 */
template <u32_t slots_, typename warp_type_>
struct vram_sorted_pool {
    using warp_t = warp_type_;
    vram_scored_node_t entries_[slots_];

    inline static constexpr u32_t width_k = warp_t::width_k;
    inline static constexpr u32_t capacity_k = slots_ * width_k;
    static_assert((slots_ & (slots_ - 1)) == 0, "The fold-merge network needs a power-of-two slot count");

    __device__ __forceinline__ void clear() noexcept {
#pragma unroll
        for (u32_t slot = 0; slot != slots_; ++slot) entries_[slot] = vram_scored_node_t::sentinel();
    }

    /** Merges a lane-wide staged batch into the pool, keeping the @c capacity_k nearest.
     *  [ascending pool | sentinel pad | descending batch] is one bitonic sequence: a single fold
     *  absorbs the batch, log2 of @c slots_ in-lane stages plus log2 of @c width_k shuffle stages
     *  restore ascending order. */
    __device__ __forceinline__ void insert_batch(vram_scored_node_t staged, u32_t lane) noexcept {
        (void)insert_batch_spilling(staged, lane);
    }

    /** The same fold-merge, reporting what fell out: at the fold boundary the
     *  element-wise @b max of the last slot and the descending batch is exactly the
     *  union's worst lane-width - sentinels while the pool still has room. @return This
     *  lane's displaced entry, for the caller to spill onward or drop. */
    __device__ __forceinline__ vram_scored_node_t insert_batch_spilling(vram_scored_node_t staged,
                                                                        u32_t lane) noexcept {
        vram_scored_node_t const descending = shuffle_xor<warp_t>(sort_lanes<warp_t>(staged, lane), width_k - 1);
        vram_scored_node_t const displaced = entries_[slots_ - 1] < descending ? descending : entries_[slots_ - 1];
        entries_[slots_ - 1] = entries_[slots_ - 1] < descending ? entries_[slots_ - 1] : descending;
#pragma unroll
        for (u32_t stride = slots_ / 2; stride >= 1; stride >>= 1)
#pragma unroll
            for (u32_t base = 0; base < slots_; ++base)
                if ((base & stride) == 0 && (base | stride) < slots_) {
                    vram_scored_node_t const lower_entry = entries_[base];
                    vram_scored_node_t const upper_entry = entries_[base | stride];
                    entries_[base] = lower_entry < upper_entry ? lower_entry : upper_entry;
                    entries_[base | stride] = lower_entry < upper_entry ? upper_entry : lower_entry;
                }
#pragma unroll
        for (u32_t partner_stride = width_k / 2; partner_stride; partner_stride >>= 1) {
            bool const lower = (lane & partner_stride) == 0;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot) {
                vram_scored_node_t const partner = shuffle_xor<warp_t>(entries_[slot], partner_stride);
                entries_[slot] = lower == (entries_[slot] < partner) ? entries_[slot] : partner;
            }
        }
        return displaced;
    }

    /** The fold-merge's result for a batch of which only the @p admitted lanes can enter: each
     *  takes its rank, ranks at and past it move up one, and the last rank drops. Raw entries never
     *  tie, since the IDs of one walk are distinct, so the strict rank is exact; every slot index
     *  stays compile-time, and both guards are warp-uniform. */
    __device__ __forceinline__ void insert_each(vram_scored_node_t staged, typename warp_t::mask_t admitted,
                                                u32_t lane) noexcept {
        while (admitted) {
            vram_scored_node_t const value = shuffle_from<warp_t>(staged, warp_t::first(admitted));
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
                vram_scored_node_t const carried = slot ? shuffle_from<warp_t>(entries_[slot - 1], width_k - 1)
                                                        : vram_scored_node_t::sentinel();
                vram_scored_node_t const shifted = shuffle_up<warp_t>(entries_[slot], 1);
                u32_t const position = slot * width_k + lane;
                if (position == rank) entries_[slot] = value;
                else if (position > rank) entries_[slot] = lane ? shifted : carried;
            }
        }
    }

    /** The rank capacity - also the base of the traversal's hop budget. */
    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_k; }

    /** The nearest entry not yet expanded - the frontier's @c top() - or the sentinel. */
    __device__ __forceinline__ vram_scored_node_t nearest_unexpanded() const noexcept {
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot) {
            typename warp_t::mask_t const mask = warp_t::ballot(!entries_[slot].expanded());
            if (mask) return shuffle_from<warp_t>(entries_[slot], warp_t::first(mask));
        }
        return vram_scored_node_t::sentinel();
    }

    /** Lazily deletes @p entry from the frontier - the unrolled-compare spelling
     *  keeps every slot index compile-time. Duplicates cannot arise: the exact
     *  visited set admits every ID into the walk exactly once. */
    __device__ __forceinline__ void mark_expanded_at(vram_scored_node_t entry) noexcept {
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
        return !(shuffle_from<warp_t>(entries_[slots_ - 1], width_k - 1) == vram_scored_node_t::sentinel());
    }

    /** Kills every staged entry whose ID already resides in the pool. The walk
     *  itself never needs this - the exact visited set stages every ID once - but the
     *  arena frontier's final fold does: a node promoted out of the tail and later
     *  snapshotted back leaves two arena copies of one ID. */
    __device__ __forceinline__ vram_scored_node_t deduplicate(vram_scored_node_t staged, u32_t lane) const noexcept {
        node_id_t const staged_id = staged.id();
        bool resident = false;
#pragma unroll
        for (u32_t probe = 0; probe < width_k; ++probe) {
            node_id_t const probe_id = warp_t::shuffle(staged_id, probe);
            bool holds = false;
#pragma unroll
            for (u32_t slot = 0; slot < slots_; ++slot)
                holds |= !(entries_[slot] == vram_scored_node_t::sentinel()) && entries_[slot].id() == probe_id;
            if (warp_t::any(holds) && lane == probe) resident = true;
        }
        return resident ? vram_scored_node_t::sentinel() : staged;
    }

    /** Entry at @p rank, valid on every lane. */
    __device__ __forceinline__ vram_scored_node_t rank_entry(u32_t rank) const noexcept {
        vram_scored_node_t result = vram_scored_node_t::sentinel();
#pragma unroll
        for (u32_t slot = 0; slot < slots_; ++slot)
            if (slot == rank / width_k) result = entries_[slot];
        return shuffle_from<warp_t>(result, rank % width_k);
    }
};

/** One warp's slice of the global traversal arena: an append-only span of scored
 *  entries spilled from the register head, consumed by refills on frontier drain.
 *  Sized ahead by the host and never grown: appends past @p capacity_ are counted,
 *  never written - the nearest candidates always reach the head first, so drops only
 *  ever shed the farthest tail - and the count is the receipt the telemetry reports. */
template <typename warp_type_>
struct vram_tail_pool {
    using warp_t = warp_type_;
    vram_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_ = 0;
    u32_t dropped_ = 0;

    __device__ __forceinline__ void append(vram_scored_node_t value, u32_t lane) noexcept {
        bool const live = !(value == vram_scored_node_t::sentinel());
        typename warp_t::mask_t const live_mask = warp_t::ballot(live);
        u32_t const ordinal = warp_t::count(live_mask & warp_t::below(lane));
        if (live && size_ + ordinal < capacity_) entries_[size_ + ordinal] = value;
        u32_t const grown = min_of(size_ + warp_t::count(live_mask), capacity_);
        dropped_ += size_ + warp_t::count(live_mask) - grown;
        size_ = grown;
        warp_t::sync();
    }
};

/** The register frontier: the fused results-and-frontier of a bounded walk, a thin
 *  verb layer over one register pool. The CPU keeps its results buffer and frontier
 *  heap separate; registers fuse them - the expanded bit plays frontier deletion, the
 *  last rank plays the results radius - and this wrapper names the traversal verbs so
 *  one walk body serves this and the arena twin below. */
template <u32_t slots_, typename warp_type_>
struct vram_register_frontier {
    using warp_t = warp_type_;
    vram_sorted_pool<slots_, warp_t> ranks_;

    __device__ __forceinline__ void reset(u32_t) noexcept { ranks_.clear(); }

    /** A walk over full ranks ends at their radius, so a further walk needs no budget renewed. */
    __device__ __forceinline__ void renew() noexcept {}
    __device__ __forceinline__ u32_t hops_budget() const noexcept { return ranks_.capacity() * 4; }

    /** Nothing to shed: the pool keeps the nearest of every union by construction. */
    __device__ __forceinline__ u32_t dropped() const noexcept { return 0; }

    /** The furthest kept rank, which nothing at or past enters; the sentinel until every rank
     *  fills. */
    __device__ __forceinline__ vram_scored_node_t worst() const noexcept {
        return shuffle_from<warp_t>(ranks_.entries_[slots_ - 1], warp_t::width_k - 1);
    }
    __device__ __forceinline__ void push_batch(vram_scored_node_t staged, u32_t lane) noexcept {
        // Only entries below the worst kept one can enter:
        // a few take their ranks, a crowd takes the fold-merge.
        constexpr u32_t few_k = 3;
        vram_scored_node_t const worst = shuffle_from<warp_t>(ranks_.entries_[slots_ - 1], warp_t::width_k - 1);
        typename warp_t::mask_t const admitted = warp_t::ballot(staged < worst);
        if (warp_t::count(admitted) <= few_k) ranks_.insert_each(staged, admitted, lane);
        else ranks_.insert_batch(staged, lane);
    }

    /** The nearest undiscovered-from candidate, already marked expanded - or the
     *  sentinel when every rank is spent. */
    __device__ __forceinline__ vram_scored_node_t pop_nearest(u32_t lane) noexcept {
        vram_scored_node_t const nearest = ranks_.nearest_unexpanded();
        if (!(nearest == vram_scored_node_t::sentinel())) ranks_.mark_expanded_at(nearest);
        return nearest;
    }

    /** After a sentinel pop: full ranks mean the walk is done, not just starved. */
    __device__ __forceinline__ bool exhausted() const noexcept { return ranks_.full(); }

    /** The classic radius test: full ranks and a candidate beyond the worst kept. */
    __device__ __forceinline__ bool beyond_radius(vram_scored_node_t candidate) const noexcept {
        return ranks_.full() && candidate.distance_bits() > ranks_.radius();
    }

    __device__ __forceinline__ vram_scored_node_t rank_entry(u32_t rank) const noexcept {
        return ranks_.rank_entry(rank);
    }

    /** Hands @p emit the nearest @p count entries ascending, a warp-width per call, rank
     *  @b base+lane on each lane; ranks past the pool are sentinels. The ranks are the container
     *  itself, so nothing is folded. */
    template <typename emit_type_>
    __device__ __forceinline__ void for_each_rank(u32_t count, u32_t lane, emit_type_&& emit) const noexcept {
        for (u32_t base = 0; base < count; base += warp_t::width_k) emit(base + lane, ranks_.rank_entry(base + lane));
    }

    /** The ranks are the container itself, so nothing settles before a selection reads them. */
    __device__ __forceinline__ void settle(u32_t) noexcept {}
};

/** The arena frontier: the de-capped twin - a register head (the exact nearest two lane-widths,
 *  absorbing every per-hop merge at constant cost) plus an arena tail taking the
 *  head's spill. The frontier refills from the tail when the head's unexpanded
 *  entries drain, so expansion follows discovery order past the head's width, and
 *  the walk retires after @c expansion_target_ expansions instead of on a radius test - the
 *  expansion budget IS the ef semantic and the horizon. */
template <typename warp_type_>
struct vram_arena_frontier {
    using warp_t = warp_type_;
    vram_sorted_pool<2, warp_t> head_;
    vram_tail_pool<warp_t> tail_;
    u32_t expansion_target_ = 0;
    u32_t expansions_ = 0;

    __device__ __forceinline__ void reset(u32_t) noexcept {
        head_.clear();
        tail_.size_ = 0;
        expansions_ = 0;
    }

    /** Grants a further walk over what this frontier holds its own expansion budget. */
    __device__ __forceinline__ void renew() noexcept { expansions_ = 0; }
    __device__ __forceinline__ u32_t hops_budget() const noexcept { return expansion_target_ * 4; }

    /** Tail entries shed past the slice's capacity - the walk's one lossy step, receipted. */
    __device__ __forceinline__ u32_t dropped() const noexcept { return tail_.dropped_; }
    __device__ __forceinline__ void push_batch(vram_scored_node_t staged, u32_t lane) noexcept {
        // When every staged entry sits at or past the head's radius the merge displaces
        // nothing - the batch goes straight to the tail, which by construction only holds
        // entries farther than everything head-resident, and staged >= radius keeps that.
        if (head_.full() && !warp_t::ballot(staged.distance_bits() < head_.radius())) return tail_.append(staged, lane);
        tail_.append(head_.insert_batch_spilling(staged, lane), lane);
    }
    __device__ __forceinline__ vram_scored_node_t pop_nearest(u32_t lane) noexcept {
        // A consumed expansion budget IS the frontier's edge: reporting starvation here
        // routes the walk into its drained-frontier exit, where exhausted() seals it.
        if (exhausted()) return vram_scored_node_t::sentinel();
        vram_scored_node_t nearest = head_.nearest_unexpanded();
        if (nearest == vram_scored_node_t::sentinel()) {
            refill(lane);
            nearest = head_.nearest_unexpanded();
        }
        if (!(nearest == vram_scored_node_t::sentinel())) {
            head_.mark_expanded_at(nearest);
            ++expansions_;
        }
        return nearest;
    }
    __device__ __forceinline__ bool exhausted() const noexcept { return expansions_ >= expansion_target_; }
    __device__ __forceinline__ bool beyond_radius(vram_scored_node_t) const noexcept { return false; }

    /** The tail keeps every entry the walk scores, so no rank bounds what enters. */
    __device__ __forceinline__ vram_scored_node_t worst() const noexcept { return vram_scored_node_t::sentinel(); }

    /** Folds the nearest two lane-widths of everything the walk saw into the head, which then
     *  answers @c rank_entry for a selection; call once, after the walk, since the tail keeps its
     *  copies. */
    __device__ __forceinline__ void settle(u32_t lane) noexcept {
        for_each_rank(head_.capacity_k, lane, [&](u32_t rank, vram_scored_node_t entry) noexcept {
            if (rank < warp_t::width_k) head_.entries_[0] = entry;
            else head_.entries_[1] = entry;
        });
    }
    __device__ __forceinline__ vram_scored_node_t rank_entry(u32_t rank) const noexcept {
        return head_.rank_entry(rank);
    }

    /** Promotes the nearest unexpanded tail entries into the head. Every tail entry is farther
     *  than everything head-resident - that is the spill invariant - so a merge could never admit
     *  them: instead the drained head, all expanded by now, spills wholesale to the tail for the
     *  final compaction, and the champions take it over as the fresh frontier. Lane champions are
     *  not the global top lane-width, but refills recur on every drain, so promotion order only
     *  affects scheduling, never membership. */
    __device__ __forceinline__ void refill(u32_t lane) noexcept {
        vram_scored_node_t best = vram_scored_node_t::sentinel();
        u32_t best_index = 0;
        for (u32_t index = lane; index < tail_.size_; index += warp_t::width_k) {
            vram_scored_node_t const entry = tail_.entries_[index];
            if (!entry.expanded() && entry < best) best = entry, best_index = index;
        }
        warp_t::sync();
        if (!warp_t::ballot(!(best == vram_scored_node_t::sentinel()))) return;
        // Promotion tombstones the source slot, so every ID lives in exactly one place:
        // promoted IDs exist only in the head until the head itself re-spills them - once.
        // Each lane's best comes from its own stride residues, so the stores never clash.
        if (!(best == vram_scored_node_t::sentinel())) tail_.entries_[best_index] = vram_scored_node_t::sentinel();
        tail_.append(head_.entries_[0], lane);
        tail_.append(head_.entries_[1], lane);
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
                vram_scored_node_t const entry = base == 0                 ? head_.entries_[0]
                                                 : base == warp_t::width_k ? head_.entries_[1]
                                                 : index < tail_.size_     ? tail_.entries_[index]
                                                                           : vram_scored_node_t::sentinel();
                u32_t const key = entry.distance_bits(), digit = (key >> shift) & 0xFFu;
                typename warp_t::mask_t own = warp_t::ballot(!(entry == vram_scored_node_t::sentinel()) &&
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
            vram_scored_node_t const entry = index < tail_.size_ ? tail_.entries_[index]
                                                                 : vram_scored_node_t::sentinel();
            bool const keep = !(entry == vram_scored_node_t::sentinel()) && entry.distance_bits() <= prefix;
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
     *  after the previous round's last into a fresh two-slot pool; past one round the tail first
     *  keeps only the entries that can rank, so the rounds scale with @p count, not the walk.
     *  Single residency by construction: promotion tombstones its tail slot, so the entries' words
     *  are distinct and every one ranks exactly once. Call once, after the walk. */
    template <typename emit_type_>
    __device__ __forceinline__ void for_each_rank(u32_t count, u32_t lane, emit_type_&& emit) noexcept {
        using round_t = vram_sorted_pool<2, warp_t>;
        if (count > round_t::capacity_k) keep_nearest(count, lane);
        vram_scored_node_t last = vram_scored_node_t::sentinel();
        for (u32_t base = 0; base < count; base += round_t::capacity_k) {
            auto const unranked = [&](vram_scored_node_t entry) noexcept {
                return base == 0 || last < entry ? entry : vram_scored_node_t::sentinel();
            };
            round_t round;
            round.clear();
            // A short round means the walk saw nothing more: every later rank is a sentinel.
            if (base == 0 || !(last == vram_scored_node_t::sentinel())) {
                round.insert_batch(unranked(head_.entries_[0]), lane);
                round.insert_batch(unranked(head_.entries_[1]), lane);
                for (u32_t tail_base = 0; tail_base < tail_.size_; tail_base += warp_t::width_k) {
                    u32_t const index = tail_base + lane;
                    round.insert_batch(
                        index < tail_.size_ ? unranked(tail_.entries_[index]) : vram_scored_node_t::sentinel(), lane);
                }
            }
            emit(base + lane, round.entries_[0]);
            emit(base + warp_t::width_k + lane, round.entries_[1]);
            last = round.rank_entry(round_t::capacity_k - 1);
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
 *  saturated set - the probe budget exhausted on a full neighborhood - reports IDs as already
 *  visited, so an under-provisioned walk starves and retires deterministically instead of looping,
 *  and counts every such claim as the receipt the telemetry reports.
 */
template <typename warp_type_>
struct vram_visited_set {
    using warp_t = warp_type_;

    /** Packed @b {epoch,ID} slots: an entry is live only when its epoch matches
     *  this walk's, so "resetting" is bumping one register - the host zeroes the
     *  arena once per launch batch, and epoch zero marks never-used. 16-byte aligned,
     *  since a probe loads the aligned pair of slots it lands in at once. */
    u64_t* slots_;

    /** One field carries the whole shape: the slot count is a power of two by
     *  construction in @c arena_shape_for, so the probe mask derives as
     *  @b 0xFFFFFFFF>>hash_shift_ and no capacity is stored. */
    u32_t hash_shift_;
    u32_t epoch_;

    /** Claims that ran out of probes and starved their walk; warp-uniform, so lane zero
     *  reports it. */
    u32_t saturations_ = 0;

    /** Probe rounds a claim takes before it starves: each round reads one aligned pair of slots, so
     *  a claim scans up to two slots a round, and saturation needs a fuller table than one slot a
     *  round. */
    inline static constexpr u32_t probe_budget_k = 64;

    /** Marks @p node_id visited; @c true when it was fresh. A lane with nothing to claim passes the
     *  sentinel, which no walk visits. */
    __device__ __forceinline__ bool test_and_set(node_id_t node_id, u32_t lane) noexcept {
        bool const active = node_id != node_id_unknown_k;
        // Warp-uniform: an idle call skips the matching votes, a ballot per distinct value on HIP.
        if (!warp_t::any(active)) return false;
        typename warp_t::mask_t const peers = warp_t::match_any(active ? node_id : 0xFFFFFFFFu ^ lane);
        bool fresh = active && (peers & warp_t::below(lane)) == 0;
        bool probing = fresh;
        u64_t const claim = (static_cast<u64_t>(epoch_) << 32) | node_id;
        u32_t const mask = 0xFFFFFFFFu >> hash_shift_;
        u32_t slot = (node_id * 0x9E3779B9u) >> hash_shift_;
        ulonglong2 const* const pairs = reinterpret_cast<ulonglong2 const*>(slots_);
        // The slice is warp-private, so the only contention is two lanes of THIS warp
        // landing on one empty slot in the same round - arbitrated in registers, no
        // atomics: the lowest rival lane stores, the warp barrier publishes the store,
        // and losers re-read the slot, see a foreign epoch-current ID, and probe on.
        for (u32_t round = 0; warp_t::any(probing) && round != probe_budget_k; ++round) {
            // One 16-byte load covers the probed slot's aligned pair, so an even-slot collision
            // resolves this round.
            ulonglong2 const pair = probing ? pairs[slot >> 1] : ulonglong2 {};
            u64_t const observed = slot & 1 ? pair.y : pair.x;
            bool claiming = false;
            if (probing) {
                // Visited this walk, or a stale or never-used slot to claim, then the same for the
                // pair's odd slot.
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
        // Saturation: claim "visited", starve, retire - and count it, warp-uniformly.
        saturations_ += warp_t::count(warp_t::ballot(probing));
        if (probing) fresh = false;
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
template <typename warp_type_>
struct vram_neighbors_ref {
    using warp_t = warp_type_;
    u32_t* row_; // Non-const even for readers: device atomic references cannot bind const words pre-P3323.
    u32_t capacity_;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }

    /** Acquire read of slot @p index, pairing with the lock holder's release fence so an ID read
     *  here comes with its node's payload; slot 0 belongs to @c peek_first. */
    __device__ __forceinline__ node_id_t load_slot(u32_t index) const noexcept {
        return warp_t::load_acquire(row_[index]);
    }

    /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current lock
     *  holder. */
    __device__ __forceinline__ void store_slot(u32_t index, node_id_t value) const noexcept {
        warp_t::store_relaxed(row_[index], value);
    }

    /** Acquire-load of the lock word: the sentinel means unpublished or locked, any other value is
     *  the first neighbor ID with the last unlocker's writes visible. */
    __device__ __forceinline__ node_id_t peek_first() const noexcept { return warp_t::load_acquire(row_[0]); }

    /** One acquisition attempt: exchanges the sentinel in with acquire order; an exchange that
     *  receives the sentinel back did @b not acquire. @return The displaced first-neighbor value,
     *  or the sentinel. */
    __device__ __forceinline__ node_id_t lock_if_free() const noexcept {
        return warp_t::exchange_acquire(row_[0], node_id_unknown_k);
    }

    /** Release-stores @p first_value into slot 0 - unlocking and simultaneously (re)publishing the
     *  edited neighbors to every future acquiring reader. */
    __device__ __forceinline__ void unlock(node_id_t first_value) const noexcept {
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
    u32_t tail_drops = 0, visited_saturations = 0;
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
        if (tail_drops | visited_saturations) {
            add(counters->tail_drops, tail_drops);
            add(counters->visited_saturations, visited_saturations);
        }
        if constexpr (traversal_ == vram_traversal_t::searching_k) return;
        add(counters->lock_attempts, lock_attempts);
        add(counters->lock_skips, lock_skips);
        add(counters->reselections, reselections);
    }
};

/** One walking warp's state: the candidate pool, the random stream, the running telemetry, and the shared scratch
 *  past its query span, where a rescoring hands its fold what the fold keeps. */
template <typename frontier_>
struct vram_warp_context {
    frontier_ frontier;
    splitmix64_t random_generator;
    vram_telemetry_t telemetry;
    byte_t* scratch = nullptr;
};

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
template <typename warp_type_>
struct vram_sorted_buffer {
    using warp_t = warp_type_;
    vram_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_;

    __device__ __forceinline__ vram_scored_node_t at(u32_t rank) const noexcept { return entries_[rank]; }

    __device__ __forceinline__ void insert_batch(vram_scored_node_t value, u32_t count, u32_t lane) noexcept {
        constexpr u32_t span_batches_k = divide_round_up(vram_cache_rows_k, warp_t::width_k);
        bool const incoming = lane < count;
        u32_t incoming_rank = incoming ? lane : 0;
        for (u32_t i = 0; i != size_; ++i) // Warp-uniform bound: every lane walks the span.
            incoming_rank += incoming && entries_[i].distance_bits() <= value.distance_bits();
        // Existing entries shift by the number of incoming values strictly ahead of them;
        // ties resolve the other way than above, so no two final positions collide.
        vram_scored_node_t moved[span_batches_k];
        u32_t moved_to[span_batches_k];
        u32_t batch = 0;
        for (u32_t base = 0; base < size_; base += warp_t::width_k, ++batch) {
            u32_t const index = base + lane;
            bool const active = index < size_;
            moved[batch] = active ? entries_[index] : vram_scored_node_t::sentinel();
            u32_t shift = 0;
            for (u32_t source = 0; source != count; ++source) // Warp-uniform: shuffles inside.
                shift += (shuffle_from<warp_t>(value, source).distance_bits() < moved[batch].distance_bits()) && active;
            moved_to[batch] = index + shift;
        }
        warp_t::sync(); // Every displaced entry is staged in registers before any write lands.
        batch = 0;
        for (u32_t base = 0; base < size_; base += warp_t::width_k, ++batch)
            if (base + lane < size_ && moved_to[batch] < capacity_) entries_[moved_to[batch]] = moved[batch];
        if (incoming && incoming_rank < capacity_) entries_[incoming_rank] = value;
        warp_t::sync();
        size_ = min_of(size_ + count, capacity_);
    }
};

/** Shared-memory spans backing one warp's neighbor re-selection, carved after the vector cache:
 *  the selected target IDs, the ascending per-candidate scores, and the kept cache positions. */
struct vram_selection_scratch_t {
    u32_t* selected_ids;
    vram_scored_node_t* scored_entries;
    u32_t* kept_positions;

    /** Each cached candidate's distance to the query, by cache row, staged once so the selection
     *  loop reads memory, not a register pool it would unroll to index. */
    f32_t* candidate_distances;

    /** Carves the four spans past the last row of @p cache: the ascending candidate scores first,
     *  since they carry the widest alignment, then the two ID spans and the staged distances. */
    template <typename fragment_type_>
    static __device__ __forceinline__ vram_selection_scratch_t carve(vram_vector_cache<fragment_type_> cache,
                                                                     u32_t connectivity) noexcept {
        u32_t const slots = min_of(connectivity, vram_cache_rows_k);
        vram_scored_node_t* const scored_entries = reinterpret_cast<vram_scored_node_t*>(cache.row(vram_cache_rows_k));
        u32_t* const selected_ids = reinterpret_cast<u32_t*>(scored_entries + slots);
        u32_t* const kept_positions = selected_ids + slots;
        return {selected_ids, scored_entries, kept_positions, reinterpret_cast<f32_t*>(kept_positions + slots)};
    }
};

/** Layout of one warp's slice of the traversal arena, computed on the host and carried by the
 *  views: the exact visited set, the wide frontier's spill tail when it has one, then a builder's
 *  scratch when shared memory cannot seat its rows. */
struct vram_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    /** Bytes of a builder's scratch - query span, candidate cache and re-selection spans - kept
     *  here instead of dynamic shared memory, or zero when shared memory seats it. */
    u32_t cache_bytes;

    /** Bytes of one warp's slice, a multiple of 16 so every visited span starts on the boundary its
     *  paired probes load from; the slices together may pass 4 GB, one alone never does. */
    __device__ __forceinline__ u32_t slice_bytes() const noexcept {
        u32_t const bytes = visited_slots * sizeof(u64_t) + tail_entries * sizeof(vram_scored_node_t) + cache_bytes;
        return (bytes + 15u) & ~15u;
    }
    __device__ __forceinline__ u64_t* visited_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(vram_row(base, warp_index, slice_bytes()));
    }
    __device__ __forceinline__ vram_scored_node_t* tail_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<vram_scored_node_t*>(visited_span(base, warp_index) + visited_slots);
    }
    __device__ __forceinline__ byte_t* cache_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<byte_t*>(tail_span(base, warp_index) + tail_entries);
    }
};

/** One warp's slice of a launch: its coordinates, its scratch - dynamic shared memory, or its arena
 *  slice's cache span when shared memory cannot seat it - and its arena visited span, the shared
 *  prologue of every traversal kernel. */
struct vram_warp_slice_t {
    u32_t lane, warp, warp_index;
    byte_t* scratch;
    u64_t* visited_slots;
};

/** Carves the calling warp's slice out of @p view and the block's @p dynamic_scratch; the arena
 *  base is allocation-aligned and every slice a multiple of 16 bytes, so each visited span takes
 *  paired 16-byte probes. */
template <typename warp_type_, typename view_>
__device__ __forceinline__ vram_warp_slice_t vram_carve_slice(view_ const& view, byte_t* dynamic_scratch) noexcept {
    u32_t const warp = threadIdx.x / warp_type_::width_k, lane = threadIdx.x % warp_type_::width_k;
    u32_t const warp_index = blockIdx.x * vram_warps_per_block_k + warp;
    byte_t* const scratch = view.arena_shape.cache_bytes ? view.arena_shape.cache_span(view.arena, warp_index)
                                                         : dynamic_scratch + warp * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch, view.arena_shape.visited_span(view.arena, warp_index)};
}

} // namespace unum::usearch::utilities
