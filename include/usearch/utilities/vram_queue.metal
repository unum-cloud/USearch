/**
 *  @file include/usearch/utilities/vram_queue.metal
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The walk vocabulary every Apple index's kernels share - the MSL twin of
 *      @c vram_queue.cuh: the simdgroup policy, the packed scored nodes, the register candidates,
 *      the arena frontier, the visited set and the greedy walk's expansion round.
 *
 *  Embedded right after @c vram_apple_shared.hpp at the head of every Apple MSL source, as
 *  @c metal_context_source_k assembles them, so the dense and basic kernels walk over one
 *  definition. A frontier empties for any walk naming its spill tail, the tail's capacity and the
 *  expansion target it retires at, and a walk scores what an expansion round stages through a
 *  callable of its own.
 */
#include <metal_stdlib>
using namespace metal;

namespace unum::usearch::utilities {

/**
 *  @brief The Apple warp policy the kernels are written against: 32-lane simdgroups with 32-bit
 *      lane masks and device-scope atomics.
 *
 *  Metal's atomics are relaxed, so acquire and release are spelled as a relaxed access beside a
 *  device-scope fence, and it has no value matching, so lanes collapse equal values by shuffling
 *  every lane's value past every other.
 */
struct metal_warp_t {
    using mask_t = uint;
    static constexpr constant uint width_k = metal_simdgroup_width_k;

    static mask_t ballot(bool has_vote) { return mask_t(simd_vote::vote_t(simd_ballot(has_vote))); }

    /** The lanes strictly below @p lane. */
    static mask_t below(uint lane) { return (1u << lane) - 1u; }
    static void sync() { simdgroup_barrier(mem_flags::mem_device); }

    /** Lanes are 32-bit everywhere, as on the CUDA and HIP policies; Metal takes 16-bit ones. */
    template <typename value_type_>
    static value_type_ shuffle(value_type_ value, uint source_lane) {
        return simd_shuffle(value, ushort(source_lane));
    }
    template <typename value_type_>
    static value_type_ shuffle_xor(value_type_ value, uint lane_mask) {
        return simd_shuffle_xor(value, ushort(lane_mask));
    }

    /** Metal shuffles no 64-bit word, so a scored node moves as its two halves. */
    static ulong shuffle(ulong value, uint source_lane) {
        return as_type<ulong>(simd_shuffle(as_type<uint2>(value), ushort(source_lane)));
    }
    static ulong shuffle_xor(ulong value, uint lane_mask) {
        return as_type<ulong>(simd_shuffle_xor(as_type<uint2>(value), ushort(lane_mask)));
    }

    /** The lanes holding the same @p value as this one: every lane meets every other. */
    static mask_t match_any(uint value) {
        mask_t peers = 0;
        for (uint lane = 0; lane != width_k; ++lane) peers |= mask_t(shuffle(value, lane) == value) << lane;
        return peers;
    }

    static void fence() { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); }
    static uint load_acquire(device uint* word) {
        uint const value = atomic_load_explicit(atomic_(word), memory_order_relaxed);
        fence();
        return value;
    }
    static void store_relaxed(device uint* word, uint value) {
        atomic_store_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static void store_release(device uint* word, uint value) {
        fence();
        atomic_store_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static uint exchange_acquire(device uint* word, uint value) {
        uint const previous = atomic_exchange_explicit(atomic_(word), value, memory_order_relaxed);
        fence();
        return previous;
    }
    static uint load_relaxed(device uint* word) { return atomic_load_explicit(atomic_(word), memory_order_relaxed); }

    /** Byte @p index of @p column, which other lanes write byte by byte; Metal has no byte atomics,
     *  so it loads the word holding it, which needs the column on a word boundary. */
    static uint load_relaxed(device uchar* column, uint index) {
        uint const word = atomic_load_explicit(atomic_((device uint*)column + (index >> 2)), memory_order_relaxed);
        return (word >> ((index & 3u) * 8u)) & 0xFFu;
    }
    static uint exchange_relaxed(device uint* word, uint value) {
        return atomic_exchange_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static uint fetch_add_relaxed(device uint* word, uint value) {
        return atomic_fetch_add_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static uint fetch_min_relaxed(device uint* word, uint value) {
        return atomic_fetch_min_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static uint fetch_or_relaxed(device uint* word, uint value) {
        return atomic_fetch_or_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static uint fetch_and_relaxed(device uint* word, uint value) {
        return atomic_fetch_and_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static bool compare_exchange_relaxed(device uint* word, uint expected, uint desired) {
        return atomic_compare_exchange_weak_explicit(atomic_(word), &expected, desired, memory_order_relaxed,
                                                     memory_order_relaxed);
    }

    static uint reduce_add(uint value) { return simd_sum(value); }
    static uint reduce_min(uint value) { return simd_min(value); }
    static ulong reduce_min(ulong value) {
        for (uint mask = width_k / 2; mask; mask >>= 1) {
            ulong const other = shuffle_xor(value, mask);
            value = other < value ? other : value;
        }
        return value;
    }

  private:
    static device atomic_uint* atomic_(device uint* word) { return (device atomic_uint*)word; }
};

/** The ID no node takes, every bit of a scored node's low half set; the walks key 32-bit IDs, the
 *  half a scored node holds, and the hosts refuse wider ones on Apple. */
constant constexpr uint metal_unknown_k = ~0u;

/** The empty slot, every bit of a scored node set, and the bit marking a node expanded. */
constant constexpr ulong metal_sentinel_k = ~0ul;
constant constexpr ulong metal_expanded_bit_k = 1ul << 32;

#pragma region Scored Nodes

/** The chunks of @p divisor that cover @p value, the last one partial. */
inline constexpr uint divide_round_up(uint value, uint divisor) { return value / divisor + uint(value % divisor != 0); }

/** One candidate in eight bytes, ordered as a raw word: the top 31 bits of the sign-flipped
 *  distance, the expanded bit, then the full node ID; the all-ones word is the empty slot. */
inline ulong metal_make(float distance, uint node_id) {
    uint flipped = as_type<uint>(distance);
    flipped ^= (flipped & 0x80000000u) ? ~0u : 0x80000000u;
    return (ulong(flipped >> 1) << 33) | node_id;
}
inline uint metal_distance_bits(ulong node) { return uint(node >> 33); }
inline float metal_distance_of(ulong node) {
    uint flipped = metal_distance_bits(node) << 1;
    flipped ^= (flipped & 0x80000000u) ? 0x80000000u : ~0u;
    return as_type<float>(flipped);
}
inline uint metal_id_of(ulong node) { return uint(node); }
inline bool metal_expanded(ulong node) { return (node & metal_expanded_bit_k) != 0; }

/** Full bitonic sort of one 32-lane vector of nodes, ascending by lane. */
inline ulong metal_sort_lanes(ulong value, uint lane) {
    for (uint stage = 2; stage <= metal_warp_t::width_k; stage <<= 1)
        for (uint stride = stage >> 1; stride; stride >>= 1) {
            ulong const partner = metal_warp_t::shuffle_xor(value, stride);
            bool const ascending = (lane & stage) == 0, lower = (lane & stride) == 0;
            ulong const nearer = value < partner ? value : partner, further = value < partner ? partner : value;
            value = ascending == lower ? nearer : further;
        }
    return value;
}

/** The register candidates of @c 32 x @p slots_ nodes: rank r at slot r/32 of lane r%32, always
 *  ascending. */
template <uint slots_>
struct metal_sorted_candidates {
    ulong entries[slots_];

    void clear() {
        for (uint slot = 0; slot != slots_; ++slot) entries[slot] = metal_sentinel_k;
    }

    /** Folds a staged 32-lane batch in, keeping the nearest; the batch descends against the
     *  ascending candidates. */
    void insert_batch(ulong staged, uint lane) { insert_batch_spilling(staged, lane); }

    /** The same fold, answering this lane's entry the fold displaced: sentinels while the
     *  candidates have room. */
    ulong insert_batch_spilling(ulong staged, uint lane) {
        ulong const descending = metal_warp_t::shuffle_xor(metal_sort_lanes(staged, lane), metal_warp_t::width_k - 1);
        ulong const last = entries[slots_ - 1];
        ulong const displaced = last < descending ? descending : last;
        entries[slots_ - 1] = last < descending ? last : descending;
        for (uint stride = slots_ / 2; stride >= 1; stride >>= 1)
            for (uint base = 0; base < slots_; ++base)
                if ((base & stride) == 0 && (base | stride) < slots_) {
                    ulong const lower_entry = entries[base], upper_entry = entries[base | stride];
                    entries[base] = lower_entry < upper_entry ? lower_entry : upper_entry;
                    entries[base | stride] = lower_entry < upper_entry ? upper_entry : lower_entry;
                }
        for (uint stride = metal_warp_t::width_k / 2; stride; stride >>= 1) {
            bool const lower = (lane & stride) == 0;
            for (uint slot = 0; slot < slots_; ++slot) {
                ulong const partner = metal_warp_t::shuffle_xor(entries[slot], stride);
                entries[slot] = lower == (entries[slot] < partner) ? entries[slot] : partner;
            }
        }
        return displaced;
    }

    ulong nearest_unexpanded() {
        for (uint slot = 0; slot < slots_; ++slot) {
            uint const mask = metal_warp_t::ballot(!metal_expanded(entries[slot]));
            if (mask) return metal_warp_t::shuffle(entries[slot], ctz(mask));
        }
        return metal_sentinel_k;
    }
    void mark_expanded_at(ulong entry) {
        for (uint slot = 0; slot < slots_; ++slot)
            if (entries[slot] == entry) entries[slot] |= metal_expanded_bit_k;
    }
    uint radius() { return metal_distance_bits(metal_warp_t::shuffle(entries[slots_ - 1], metal_warp_t::width_k - 1)); }
    bool full() { return metal_warp_t::shuffle(entries[slots_ - 1], metal_warp_t::width_k - 1) != metal_sentinel_k; }
    ulong rank_entry(uint rank) {
        ulong result = metal_sentinel_k;
        for (uint slot = 0; slot < slots_; ++slot)
            if (slot == rank / metal_warp_t::width_k) result = entries[slot];
        return metal_warp_t::shuffle(result, rank % metal_warp_t::width_k);
    }

    /** Empties the candidates for a walk, which they read nothing of, as the CUDA register
     *  frontier's @c clear does. */
    template <typename walk_type_>
    void clear(thread walk_type_&) {
        clear();
    }

    /** Nothing to compact or drop: the ranks keep the nearest of every union by construction. */
    uint compactions() { return 0; }
    uint dropped() { return 0; }

    /** Kills every staged entry whose ID already resides in the candidates, as the CUDA
     *  candidates' @c deduplicate does, so a re-scored ID past a saturation keeps one copy. */
    ulong deduplicate(ulong staged, uint lane) {
        uint const staged_id = metal_id_of(staged);
        bool resident = false;
        for (uint probe = 0; probe < metal_warp_t::width_k; ++probe) {
            uint const probe_id = metal_warp_t::shuffle(staged_id, probe);
            bool holds = false;
            for (uint slot = 0; slot < slots_; ++slot)
                holds |= entries[slot] != metal_sentinel_k && metal_id_of(entries[slot]) == probe_id;
            if (metal_warp_t::ballot(holds) && lane == probe) resident = true;
        }
        return resident ? metal_sentinel_k : staged;
    }

    /** Whether the candidates already hold the ID @p staged_id names on this lane, every lane
     *  taking part. */
    bool holds(uint staged_id, uint lane) { return deduplicate(metal_make(0, staged_id), lane) == metal_sentinel_k; }

    /** The fold's result for a batch of which only the @p admitted lanes can enter, as the CUDA
     *  candidates' @c insert_each writes it: each takes its rank, ranks at and past it move up one,
     *  and the last rank drops. */
    void insert_each(ulong staged, uint admitted, uint lane) {
        while (admitted) {
            ulong const value = metal_warp_t::shuffle(staged, ctz(admitted));
            admitted &= admitted - 1;
            uint rank = 0;
            for (uint slot = 0; slot != slots_; ++slot) rank += popcount(metal_warp_t::ballot(entries[slot] < value));
            // An earlier entry already pushed the worst below it.
            if (rank == slots_ * metal_warp_t::width_k) continue;
            // Highest slot first, so each slot's old last lane carries up before anything
            // rewrites the slot.
            for (uint slot = slots_; slot-- != 0;) {
                // Lower slots keep their ranks.
                if ((slot + 1) * metal_warp_t::width_k <= rank) break;
                ulong const carried = slot ? metal_warp_t::shuffle(entries[slot - 1], metal_warp_t::width_k - 1)
                                           : metal_sentinel_k;
                ulong const shifted = metal_warp_t::shuffle(entries[slot], lane ? lane - 1 : 0);
                uint const position = slot * metal_warp_t::width_k + lane;
                if (position == rank) entries[slot] = value;
                else if (position > rank) entries[slot] = lane ? shifted : carried;
            }
        }
    }
    void push_batch(ulong staged, uint lane) {
        // Only entries strictly nearer than the worst kept one enter, as the CUDA frontier's do: a
        // few take their ranks, a crowd takes the fold.
        constexpr uint few_k = 3;
        ulong const worst = metal_warp_t::shuffle(entries[slots_ - 1], metal_warp_t::width_k - 1);
        bool const nearer = metal_distance_bits(staged) < metal_distance_bits(worst);
        uint const admitted = metal_warp_t::ballot(nearer);
        if (popcount(admitted) <= few_k) insert_each(staged, admitted, lane);
        else insert_batch(nearer ? staged : metal_sentinel_k, lane);
    }
    ulong pop_nearest(uint) {
        ulong const nearest = nearest_unexpanded();
        if (nearest != metal_sentinel_k) mark_expanded_at(nearest);
        return nearest;
    }
    bool exhausted() { return full(); }
    bool beyond_radius(ulong candidate) { return full() && metal_distance_bits(candidate) > radius(); }

    /** Hands @p emit the nearest @p count entries ascending, a warp-width per call, rank
     *  @b base+lane on each lane. */
    template <typename emit_type_>
    void for_each_rank(uint count, uint lane, thread emit_type_& emit) {
        for (uint base = 0; base < count; base += metal_warp_t::width_k) emit(base + lane, rank_entry(base + lane));
    }
};

/** An ascending span of scored nodes holding at most @c capacity, the CUDA re-selection's
 *  @c vram_sorted_buffer. */
struct metal_sorted_buffer {
    device ulong* entries;
    uint size, capacity;

    ulong at(uint rank) { return entries[rank]; }

    /** Merges a pre-sorted batch of @p count nodes in, keeping @c capacity. */
    void insert_batch(ulong value, uint count, uint lane) {
        bool const incoming = lane < count;
        uint incoming_rank = incoming ? lane : 0;
        // Simdgroup-uniform bound: every lane walks the span.
        for (uint index = 0; index != size; ++index)
            incoming_rank += incoming && metal_distance_bits(entries[index]) <= metal_distance_bits(value);
        // Existing entries shift by the number of incoming values strictly ahead of them; ties
        // resolve the other way than above, so no two final positions collide. Shifts only move
        // entries up, so moving the last lane-width first never overwrites an unmoved one.
        for (uint batches = divide_round_up(size, metal_warp_t::width_k); batches; --batches) {
            uint const index = (batches - 1) * metal_warp_t::width_k + lane;
            bool const active = index < size;
            ulong const moved = active ? entries[index] : metal_sentinel_k;
            uint shift = 0;
            // Simdgroup-uniform: shuffles inside.
            for (uint source = 0; source != count; ++source) {
                ulong const appended_node = metal_warp_t::shuffle(value, source);
                shift += active && metal_distance_bits(appended_node) < metal_distance_bits(moved);
            }
            // The whole lane-width is in registers before any of its writes reaches memory.
            metal_warp_t::sync();
            if (active && index + shift < capacity) entries[index + shift] = moved;
            metal_warp_t::sync();
        }
        if (incoming && incoming_rank < capacity) entries[incoming_rank] = value;
        metal_warp_t::sync();
        size = min(size + count, capacity);
    }
};

#pragma endregion Scored Nodes

#pragma region Frontiers

/** The lanes among a tally's votes whose digit's high bits spell @p bin: of the lanes @p own
 *  holds, those each of the @p high bit votes agrees with. */
inline uint metal_lanes_in_bin(uint own, thread uint const* high, uint high_bits, uint bin) {
    for (uint bit = 0; bit != high_bits; ++bit) own &= (bin >> bit) & 1u ? high[bit] : ~high[bit];
    return own;
}

/** The arena frontier, the CUDA one line for line: a two-slot register front plus a spill tail in
 *  the simdgroup's slice, refilled from the tail when the front drains, retiring after
 *  @c expansion_target expansions, and ranking everything it kept at the end, a front's width a
 *  round at a time. */
struct metal_arena_frontier {
    metal_sorted_candidates<2> front;
    device ulong* tail;
    uint tail_size, tail_capacity;

    /** Times a full tail kept only what can still rank among the expansion target, and the entries
     *  tied at that radius a compacted tail still could not fit. */
    uint tail_compactions, tail_dropped;
    uint expansion_target, expansions;

    /** Empties the frontier for a walk, which names the tail it spills to, the tail's capacity and
     *  the expansion target. */
    template <typename walk_type_>
    void clear(thread walk_type_& walk) {
        front.clear();
        tail = walk.tail, tail_size = 0, tail_capacity = walk.tail_entries, tail_compactions = 0, tail_dropped = 0;
        expansion_target = walk.expansion, expansions = 0;
    }

    uint compactions() { return tail_compactions; }
    uint dropped() { return tail_dropped; }
    bool holds(uint staged_id, uint lane) { return front.holds(staged_id, lane); }

    /** Appends a lane-wide batch to the tail; a full tail first keeps what can still rank among the
     *  walk's expansion target, as the CUDA arena frontier does, counted, as are the entries it
     *  still cannot fit. */
    void spill(ulong value, uint lane) {
        bool const full = tail_size + metal_warp_t::width_k > tail_capacity;
        if (full) keep_nearest(expansion_target, lane);
        tail_compactions += full;
        bool const live = value != metal_sentinel_k;
        uint const live_mask = metal_warp_t::ballot(live);
        uint const ordinal = popcount(live_mask & metal_warp_t::below(lane));
        if (live && tail_size + ordinal < tail_capacity) tail[tail_size + ordinal] = value;
        uint const grown = min(tail_size + popcount(live_mask), tail_capacity);
        tail_dropped += tail_size + popcount(live_mask) - grown;
        tail_size = grown;
        metal_warp_t::sync();
    }

    /** A batch at or past the front's radius goes straight to the tail, which only holds entries
     *  farther than the front's; otherwise the front's spill does. */
    void push_batch(ulong staged, uint lane) {
        ulong const spilled = front.full() && !metal_warp_t::ballot(metal_distance_bits(staged) < front.radius())
                                  ? staged
                                  : front.insert_batch_spilling(staged, lane);
        spill(spilled, lane);
    }

    /** Promotes the nearest unexpanded tail entries into a drained front, tombstoning their slots,
     *  so every ID lives in one place; the spent front spills wholesale to the tail. */
    void refill(uint lane) {
        ulong best = metal_sentinel_k;
        uint best_index = 0;
        for (uint index = lane; index < tail_size; index += metal_warp_t::width_k) {
            ulong const entry = tail[index];
            if (!metal_expanded(entry) && entry < best) best = entry, best_index = index;
        }
        metal_warp_t::sync();
        if (!metal_warp_t::ballot(best != metal_sentinel_k)) return;
        if (best != metal_sentinel_k) tail[best_index] = metal_sentinel_k;
        spill(front.entries[0], lane);
        spill(front.entries[1], lane);
        front.clear();
        front.insert_batch(best, lane);
    }

    ulong pop_nearest(uint lane) {
        // A consumed budget is the frontier's edge: the walk's drained-frontier exit seals it.
        if (exhausted()) return metal_sentinel_k;
        ulong nearest = front.nearest_unexpanded();
        if (nearest == metal_sentinel_k) {
            refill(lane);
            nearest = front.nearest_unexpanded();
        }
        if (nearest != metal_sentinel_k) {
            front.mark_expanded_at(nearest);
            ++expansions;
        }
        return nearest;
    }
    bool exhausted() { return expansions >= expansion_target; }
    bool beyond_radius(ulong) { return false; }

    /** Drops the tail entries that cannot rank among the nearest @p count, as the CUDA arena
     *  frontier's @c keep_nearest does: a radix select of the count-th smallest distance key, eight
     *  bits a pass from the top, ties kept, each lane tallying by ballots the digits whose low five
     *  bits are its own index. */
    void keep_nearest(uint count, uint lane) {
        constexpr uint lane_bits_k = 5, high_bits_k = 3, bins_k = 8;
        uint prefix = 0, prefix_mask = 0, rank = count;
        for (uint pass = 0; pass != 4; ++pass) {
            uint const shift = 24 - 8 * pass;
            uint bins[bins_k] = {};
            // The front's two slots tally as the first two batches, then the tail a simdgroup-width
            // at a time.
            for (uint base = 0; base < 2 * metal_warp_t::width_k + tail_size; base += metal_warp_t::width_k) {
                uint const index = base + lane - 2 * metal_warp_t::width_k;
                ulong const entry = base == 0                       ? front.entries[0]
                                    : base == metal_warp_t::width_k ? front.entries[1]
                                    : index < tail_size             ? tail[index]
                                                                    : metal_sentinel_k;
                uint const key = metal_distance_bits(entry), digit = (key >> shift) & 0xFFu;
                uint own = metal_warp_t::ballot(entry != metal_sentinel_k && (key & prefix_mask) == prefix);
                for (uint bit = 0; bit != lane_bits_k; ++bit) {
                    uint const set = metal_warp_t::ballot(((digit >> bit) & 1u) != 0);
                    own &= (lane >> bit) & 1u ? set : ~set;
                }
                uint high[high_bits_k];
                for (uint bit = 0; bit != high_bits_k; ++bit)
                    high[bit] = metal_warp_t::ballot(((digit >> (lane_bits_k + bit)) & 1u) != 0);
                for (uint bin = 0; bin != bins_k; ++bin)
                    bins[bin] += popcount(metal_lanes_in_bin(own, high, high_bits_k, bin));
            }
            // Digits ascend bin-major, lane-minor; the uniform guard keeps every vote across the
            // whole simdgroup, all lanes at once.
            uint below = 0;
            bool found = false;
            for (uint bin = 0; bin != bins_k; ++bin) {
                uint const inclusive = simd_prefix_inclusive_sum(bins[bin]);
                uint const total = metal_warp_t::shuffle(inclusive, metal_warp_t::width_k - 1);
                if (!found && below + total >= rank) {
                    uint const owner = ctz(metal_warp_t::ballot(below + inclusive >= rank));
                    below += metal_warp_t::shuffle(inclusive - bins[bin], owner);
                    prefix |= ((bin << lane_bits_k) | owner) << shift;
                    found = true;
                }
                if (!found) below += total;
            }
            // The walk saw fewer than count entries, so every one ranks.
            if (!found) return;
            prefix_mask |= 0xFFu << shift;
            rank -= below;
        }
        // In place: a kept entry moves to its own slot or an earlier one, after the ballot over the
        // loads of its batch.
        uint kept = 0;
        for (uint base = 0; base < tail_size; base += metal_warp_t::width_k) {
            uint const index = base + lane;
            ulong const entry = index < tail_size ? tail[index] : metal_sentinel_k;
            bool const keep = entry != metal_sentinel_k && metal_distance_bits(entry) <= prefix;
            uint const keeping = metal_warp_t::ballot(keep);
            if (keep) tail[kept + popcount(keeping & metal_warp_t::below(lane))] = entry;
            kept += popcount(keeping);
        }
        tail_size = kept;
        metal_warp_t::sync();
    }

    /** Hands @p emit the nearest @p count entries of the front and the tail ascending, as the CUDA
     *  arena frontier does: each round folds every entry ranked after the previous round's last
     *  into fresh two-slot candidates, and past one round the tail first keeps only the entries
     *  that can rank. */
    template <typename emit_type_>
    void for_each_rank(uint count, uint lane, thread emit_type_& emit) {
        constexpr uint round_ranks_k = 2 * metal_warp_t::width_k;
        if (count > round_ranks_k) keep_nearest(count, lane);
        ulong last = metal_sentinel_k;
        metal_sorted_candidates<2> previous;
        previous.clear();
        for (uint base = 0; base < count; base += round_ranks_k) {
            metal_sorted_candidates<2> round;
            round.clear();
            // An ID re-scored past a visited-set saturation can sit in the tail twice, so a round
            // keeps one copy of it.
            auto const fold = [&](ulong entry) {
                entry = base == 0 || last < entry ? entry : metal_sentinel_k;
                round.insert_batch(round.deduplicate(previous.deduplicate(entry, lane), lane), lane);
            };
            // Once a round comes up short the walk saw nothing more, and every later rank is a
            // sentinel of its own.
            if (base == 0 || last != metal_sentinel_k) {
                fold(front.entries[0]);
                fold(front.entries[1]);
                for (uint tail_base = 0; tail_base < tail_size; tail_base += metal_warp_t::width_k) {
                    uint const index = tail_base + lane;
                    fold(index < tail_size ? tail[index] : metal_sentinel_k);
                }
            }
            emit(base + lane, round.entries[0]);
            emit(base + metal_warp_t::width_k + lane, round.entries[1]);
            last = round.rank_entry(round_ranks_k - 1);
            previous = round;
        }
    }
};

/** The frontier of a walk on the candidate ladder's rung of @p slots_ slots; slot zero is the arena
 *  frontier. */
template <uint slots_>
struct metal_frontier_of {
    using type = metal_sorted_candidates<slots_>;
};
template <>
struct metal_frontier_of<0> {
    using type = metal_arena_frontier;
};

#pragma endregion Frontiers

#pragma region Walks

/** Entry @p index of a matrix of @p stride columns at @p base: the one place an offset widens past
 *  32 bits. */
template <typename value_type_>
inline device value_type_* metal_row(device value_type_* base, uint index, uint stride) {
    return base + ulong(index) * stride;
}

/** Avalanches all 64 bits of @p word - the host's @c mix_bits, the SplitMix64 finalizer. */
inline ulong metal_mix_bits(ulong word) {
    word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ul;
    word = (word ^ (word >> 27)) * 0x94D049BB133111EBul;
    return word ^ (word >> 31);
}

/** The random stream of @p lane walking for @p stream_key, seeded as @c vram_lane_generator seeds
 *  the CUDA and HIP ones. */
inline ulong metal_lane_generator(uint stream_key, uint lane) {
    return metal_mix_bits(ulong(stream_key) * metal_warp_t::width_k + lane + 1);
}

/*  The walk verbs below take any simdgroup's walk naming what every walk keeps, whatever it scores:
 *  the @c neighbors matrix of @c connectivity slots, the @c counters words, its slice's @c visited
 *  set of @c hash_shift and @c epoch, the two-width @c ids span, its @c lane, the @c random stream,
 *  the @c walk_width it expands at, and the @c saturations, @c hops and @c locked_drops tallies
 *  beside the rest of the telemetry @ref metal_flush folds. The dense and basic walks name them
 *  alike, and each adds how it scores. */

/** One SplitMix64 step of @p walk 's stream, as @c splitmix64_t takes it, by
 *  @c golden_ratio_u64_k. */
template <typename walk_type_>
inline ulong metal_next_random(thread walk_type_& walk) {
    return metal_mix_bits(walk.random += 0x9E3779B97F4A7C15ul);
}

/** Marks @p node_id visited in @p walk 's set, answering whether it was fresh; equal IDs collapse
 *  to their lowest lane, and a claim out of probes, log2 of the slots, the expected longest probe
 *  of linear probing at the half load the host sizes the set for, counts as a saturation and
 *  re-scores as fresh, as the CUDA set's @c probe_budget derives it. A lane with nothing to claim
 *  passes the sentinel, which no walk visits. A probe loads an aligned pair, so the span is aligned
 *  to 16 bytes. */
template <typename walk_type_>
inline bool metal_test_and_set(thread walk_type_& walk, uint node_id) {
    bool const active = node_id != metal_unknown_k;
    // Simdgroup-uniform: an idle call skips the matching votes, a shuffle per lane on Metal.
    if (!simd_any(active)) return false;
    uint const lane = walk.lane, epoch = walk.epoch, hash_shift = walk.hash_shift;
    uint const peers = metal_warp_t::match_any(active ? node_id : ~lane);
    bool fresh = active && (peers & metal_warp_t::below(lane)) == 0;
    bool probing = fresh;
    ulong const claim = (ulong(epoch) << 32) | node_id;
    uint const mask = ~0u >> hash_shift;
    uint slot = (node_id * 0x9E3779B9u) >> hash_shift;
    device ulong2 const* const pairs = (device ulong2 const*)walk.visited;
    for (uint round = 0, budget = 32 - hash_shift; simd_any(probing) && round != budget; ++round) {
        // One 16-byte load covers the probed slot's aligned pair, so an even-slot collision
        // resolves this round.
        ulong2 const pair = probing ? pairs[slot >> 1] : ulong2(0);
        ulong const observed = slot & 1 ? pair.y : pair.x;
        bool claiming = false;
        if (probing) {
            if (observed == claim) fresh = probing = false;
            else if (uint(observed >> 32) != epoch) claiming = true;
            else if (slot & 1) slot = (slot + 1) & mask;
            else if (pair.y == claim) fresh = probing = false;
            else if (uint(pair.y >> 32) != epoch) slot |= 1, claiming = true;
            else slot = (slot + 2) & mask;
        }
        uint const rivals = metal_warp_t::match_any(claiming ? slot : ~lane);
        if (claiming && (rivals & metal_warp_t::below(lane)) == 0) walk.visited[slot] = claim, probing = false;
        metal_warp_t::sync();
    }
    walk.saturations += popcount(metal_warp_t::ballot(probing));
    metal_warp_t::sync();
    return fresh;
}

/** Stages the fresh IDs of one lane-wide read into @p walk 's @c ids, answering how many; lanes
 *  pass the sentinel for none. Past a saturation a claim re-scores, so an ID @p frontier holds
 *  stages no second copy of itself. */
template <typename walk_type_, typename frontier_>
inline uint metal_stage_fresh(thread walk_type_& walk, uint node_id, thread frontier_& frontier) {
    // Every lane takes part in the frontier's collective scan.
    bool const claimed = metal_test_and_set(walk, node_id);
    bool const fresh = !(walk.saturations && frontier.holds(node_id, walk.lane)) && claimed;
    uint const fresh_mask = metal_warp_t::ballot(fresh);
    if (fresh) walk.ids[popcount(fresh_mask & metal_warp_t::below(walk.lane))] = node_id;
    metal_warp_t::sync();
    return popcount(fresh_mask);
}

/** Draws one lane-wide batch of distinct, published landmarks from @p sampling_first up to
 *  @p sampling_bound into @p walk 's @c ids: a lane meeting an unpublished node draws again, which
 *  ends since a wave admits at most a share of the nodes below it, and a lane meeting a visited one
 *  steps forward, for one lap of the range at most, as the CUDA and HIP samplers do; answers how
 *  many, zero once every published node of the range is visited. @p frontier is asked whether it
 *  holds a draw once a saturated visited set re-scores it. */
template <typename walk_type_, typename frontier_>
inline uint metal_sample_landmarks(thread walk_type_& walk, uint sampling_first, uint sampling_bound,
                                   thread frontier_& frontier) {
    uint const span = sampling_bound - sampling_first, lane = walk.lane;
    uint draw = sampling_first + uint((ulong(uint(metal_next_random(walk))) * span) >> 32);
    uint steps = 0;
    bool fresh = false;
    uint const saturated = walk.saturations;
    // The trip count is simdgroup-uniform, since every lane takes part in each collective claim,
    // and the lap ends at the set's first saturation, whose claims re-score, skipping draws the
    // frontier already holds.
    while (!simd_all(fresh || steps == span) && walk.saturations == saturated) {
        bool const searching = !fresh && steps != span;
        device uint* const links = metal_row(walk.neighbors, draw, walk.connectivity);
        bool const published = searching && metal_warp_t::load_acquire(links) != metal_unknown_k;
        bool const drawn = metal_test_and_set(walk, published ? draw : metal_unknown_k);
        bool const claimed = !(walk.saturations != saturated && frontier.holds(draw, lane)) && drawn;
        // An unpublished node draws again; a visited one steps forward, one lap at most.
        bool const stepping = searching && !claimed && published;
        if (searching && !claimed && !published)
            draw = sampling_first + uint((ulong(uint(metal_next_random(walk))) * span) >> 32);
        if (stepping) draw = draw + 1 == sampling_bound ? sampling_first : draw + 1;
        steps += stepping;
        fresh = fresh || claimed;
    }
    uint const fresh_mask = metal_warp_t::ballot(fresh);
    if (fresh) walk.ids[popcount(fresh_mask & metal_warp_t::below(lane))] = draw;
    metal_warp_t::sync();
    return popcount(fresh_mask);
}

/** The window of one expansion round: each lane's slot load of a node's links, and the node they
 *  belong to. */
struct metal_gathered_links_t {
    uint neighbor_ids[metal_max_gathers_in_flight_k], owner_ids[metal_max_gathers_in_flight_k];
    uint loads;
};

/** Claims and stages the window's @c loads slot loads, every simdgroup-width of fresh IDs reaching
 *  @p score through @p walk 's two-width @c ids span, whose @p pending entries carry to the caller
 *  across every round. */
template <typename walk_type_, typename frontier_, typename score_type_>
inline void metal_stage_window(thread walk_type_& walk, thread frontier_& frontier,
                               thread metal_gathered_links_t& window, uint first_invisible, thread uint& pending,
                               thread score_type_& score) {
    uint const lane = walk.lane;
    // Rolled, so the claim and the scoring it may call inline once per walk, not once per load.
#pragma unroll 1
    for (uint load = 0; load != window.loads; ++load) {
        uint const neighbor_id = window.neighbor_ids[load];
        bool const usable = neighbor_id < first_invisible && neighbor_id != window.owner_ids[load];
        bool const claimed = metal_test_and_set(walk, usable ? neighbor_id : metal_unknown_k);
        // Past a saturation a claim re-scores, so an ID the frontier or this staging holds stages
        // no second copy of itself.
        bool repeated = walk.saturations && frontier.holds(neighbor_id, lane);
        for (uint index = 0; walk.saturations && index != pending; ++index) repeated |= walk.ids[index] == neighbor_id;
        bool const fresh = claimed && !repeated;
        uint const fresh_mask = metal_warp_t::ballot(fresh);
        if (fresh) walk.ids[pending + popcount(fresh_mask & metal_warp_t::below(lane))] = neighbor_id;
        metal_warp_t::sync();
        pending += popcount(fresh_mask);
        if (pending < metal_warp_t::width_k) continue;
        score(metal_warp_t::width_k);
        metal_warp_t::sync();
        pending -= metal_warp_t::width_k;
        uint const carried = lane < pending ? walk.ids[metal_warp_t::width_k + lane] : metal_unknown_k;
        metal_warp_t::sync();
        if (lane < pending) walk.ids[lane] = carried;
        metal_warp_t::sync();
    }
}

/** Expands @p first, whose lock word @p first_value the walk already read, with up to
 *  @c walk_width - 1 further candidates inside the radius as one round, as the CUDA and HIP
 *  @c vram_expand_round does: every node's slots load in one window, the nodes claim in pop order,
 *  and each simdgroup-width of fresh IDs reaches @p score, staged through @ref metal_stage_window.
 *  A locked further node is held when the walk holds none, else counted dropped. */
template <typename walk_type_, typename frontier_, typename score_type_>
inline void metal_expand_round(thread walk_type_& walk, thread frontier_& frontier, ulong first, uint first_value,
                               uint first_invisible, thread ulong& held, thread uint& held_at_hop, thread uint& pending,
                               thread score_type_& score) {
    uint const lane = walk.lane, connectivity = walk.connectivity;
    uint const chunks = divide_round_up(connectivity, metal_warp_t::width_k);
    uint const expanding_cap = max(1u, min(walk.walk_width, uint(metal_max_gathers_in_flight_k) / chunks));
    // Lane `e` keeps the round's expanded node `e`; the radius only tightens, so one popped past it
    // ends the round.
    ulong expanded_entry = lane == 0 ? first : metal_sentinel_k;
    uint expanding = 1;
    while (expanding < expanding_cap) {
        ulong const next = frontier.pop_nearest(lane);
        if (next == metal_sentinel_k || frontier.beyond_radius(next)) break;
        if (lane == expanding) expanded_entry = next;
        ++walk.hops, ++expanding;
    }
    device uint* const expanded_links = metal_row(walk.neighbors, metal_id_of(expanded_entry), connectivity);
    uint const lock_value = lane == 0          ? first_value
                            : lane < expanding ? metal_warp_t::load_acquire(expanded_links)
                                               : metal_unknown_k;
    metal_warp_t::fence();
    uint const locked = metal_warp_t::ballot(lane < expanding && lock_value == metal_unknown_k);
    if (locked) {
        bool const holding = held == metal_sentinel_k;
        if (holding) held = metal_warp_t::shuffle(expanded_entry, ctz(locked)), held_at_hop = walk.hops;
        walk.locked_drops += popcount(locked) - uint(holding);
    }
    uint owners_left = metal_warp_t::ballot(lane < expanding && lock_value != metal_unknown_k) & ~1u;
    uint owner_lane = 0, slot_base = 0;
    bool more = true;
    while (more) {
        metal_gathered_links_t window;
        window.loads = 0;
        for (uint load = 0; load != metal_max_gathers_in_flight_k && more; ++load) {
            uint const owner = metal_id_of(metal_warp_t::shuffle(expanded_entry, owner_lane));
            uint const word = metal_warp_t::shuffle(lock_value, owner_lane);
            uint const slot = slot_base + lane;
            device uint* const owner_links = metal_row(walk.neighbors, owner, connectivity);
            window.neighbor_ids[load] = slot == 0             ? word
                                        : slot < connectivity ? metal_warp_t::load_acquire(owner_links + slot)
                                                              : metal_unknown_k;
            window.owner_ids[load] = owner;
            ++window.loads;
            slot_base += metal_warp_t::width_k;
            if (slot_base < connectivity) continue;
            more = owners_left != 0;
            if (more) owner_lane = ctz(owners_left), owners_left &= owners_left - 1;
            slot_base = 0;
        }
        metal_stage_window(walk, frontier, window, first_invisible, pending, score);
    }
}

/** Folds @p walk 's telemetry into the device counters, from its leader; a tally carries into its
 *  high word, since one batch's scored nodes outgrow 32 bits. */
template <typename walk_type_>
inline void metal_flush(thread walk_type_& walk) {
    if (walk.lane != 0) return;
    auto const add = [&](uint counter, uint value) {
        if (!value) return;
        uint const low = atomic_fetch_add_explicit(walk.counters + counter, value, memory_order_relaxed);
        if (low + value < low) atomic_fetch_add_explicit(walk.counters + counter + 1, 1u, memory_order_relaxed);
    };
    add(metal_hops_k, walk.hops);
    add(metal_scored_k, walk.scored_count);
    add(metal_restarts_k, walk.restarts);
    add(metal_lock_attempts_k, walk.lock_attempts);
    add(metal_lock_skips_k, walk.lock_skips);
    add(metal_reselections_k, walk.reselections);
    add(metal_locked_drops_k, walk.locked_drops);
    add(metal_saturations_k, walk.saturations);
    add(metal_tail_compactions_k, walk.tail_compactions);
    add(metal_tail_dropped_k, walk.tail_dropped);
}

#pragma endregion Walks

} // namespace unum::usearch::utilities
