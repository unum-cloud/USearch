/**
 *  @file include/usearch/dense/vram_queue.cuh
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
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

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
    __device__ __forceinline__ u32_t hops_budget() const noexcept { return ranks_.capacity() * 4; }

    /** Nothing to shed: the pool keeps the nearest of every union by construction. */
    __device__ __forceinline__ u32_t dropped() const noexcept { return 0; }
    __device__ __forceinline__ void push_batch(vram_scored_node_t staged, u32_t lane) noexcept {
        // A full pool keeps the nearest of the union, so a batch with no entry inside the
        // radius is a guaranteed no-op - one ballot skips the whole bitonic merge. Only
        // the narrow pools take the gate: past them the extra live radius dependency
        // costs the 3-block kernels their zero-spill budget.
        if constexpr (slots_ <= 4)
            if (ranks_.full() && !warp_t::ballot(staged.distance_bits() < ranks_.radius())) return;
        ranks_.insert_batch(staged, lane);
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

    /** Hands @p emit the nearest @p count entries of everything the walk has seen - the last head
     *  generation plus every tail entry - ascending, a warp-width per call, rank @b base+lane on
     *  each lane; ranks past what the walk saw are sentinels. Each round folds every entry ranked
     *  after the previous round's last into a fresh two-slot pool, so the ranks stream out with no
     *  scratch and only as many rounds as @p count asks for. Single residency by construction:
     *  promotion tombstones its tail slot, so the entries' words are distinct and every one ranks
     *  exactly once. Call once, after the walk. */
    template <typename emit_type_>
    __device__ __forceinline__ void for_each_rank(u32_t count, u32_t lane, emit_type_&& emit) const noexcept {
        using round_t = vram_sorted_pool<2, warp_t>;
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
     *  arena once per launch batch, and epoch zero marks never-used. */
    u64_t* slots_;

    /** One field carries the whole shape: the slot count is a power of two by
     *  construction in @c arena_shape_for, so the probe mask derives as
     *  @b 0xFFFFFFFF>>hash_shift_ and no capacity is stored. */
    u32_t hash_shift_;
    u32_t epoch_;

    /** Claims that ran out of probes and starved their walk; warp-uniform, so lane zero
     *  reports it. */
    u32_t saturations_ = 0;

    inline static constexpr u32_t probe_budget_k = 64;

    /** Marks @p node_id visited; @c true when it was fresh. A lane with nothing to claim passes the
     *  sentinel, which no walk visits. */
    __device__ __forceinline__ bool test_and_set(node_id_t node_id, u32_t lane) noexcept {
        bool const active = node_id != node_id_unknown_k;
        typename warp_t::mask_t const peers = warp_t::match_any(active ? node_id : 0xFFFFFFFFu ^ lane);
        bool fresh = active && (peers & warp_t::below(lane)) == 0;
        bool probing = fresh;
        u64_t const claim = (static_cast<u64_t>(epoch_) << 32) | node_id;
        u32_t const mask = 0xFFFFFFFFu >> hash_shift_;
        u32_t slot = (node_id * 0x9E3779B9u) >> hash_shift_;
        // The slice is warp-private, so the only contention is two lanes of THIS warp
        // landing on one empty slot in the same round - arbitrated in registers, no
        // atomics: the lowest rival lane stores, the warp barrier publishes the store,
        // and losers re-read the slot, see a foreign epoch-current ID, and probe on.
        for (u32_t round = 0; warp_t::any(probing) && round != probe_budget_k; ++round) {
            u64_t const observed = probing ? slots_[slot] : 0;
            bool claiming = false;
            if (probing) {
                if (observed == claim) fresh = probing = false; // Already visited this walk.
                else if (static_cast<u32_t>(observed >> 32) == epoch_) slot = (slot + 1) & mask;
                else claiming = true; // Stale or never used.
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

} // namespace unum::usearch::dense
