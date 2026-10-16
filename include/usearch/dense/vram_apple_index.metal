/**
 *  @file include/usearch/dense/vram_apple_index.metal
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Metal walk, insert, refine and search kernels of the dense Apple index - the MSL port of
 *      @c vram_index.cuh and the kernels of @c vram_nvidia_index.cuh.
 *
 *  One 32-wide simdgroup walks one query or inserts one node, with the same packed scored nodes,
 *  register pools, exact visited sets, slot-0 lock words and admission waves as the CUDA design, over
 *  @ref metal_warp_t, whose atomics and value matching Metal spells its own way. Every
 *  per-simdgroup scratch span - the query, the candidate cache, the visited set and the staged batch
 *  - lives in one device slice rather than threadgroup memory, which caps at 32 KB and would bound
 *  the dimensions. The host prepends @c vram_apple_shared.h, its prelude and
 *  @c vram_apple_metric.metal before this file, which takes its kit from @c quads_ladder.
 */

namespace unum::usearch::dense {

using kit_t = typename quads_ladder<metal_dtype_k, metal_capability_k>::kit_t;
using warp_t = metal_warp_t;
using dot_t = typename kit_t::dot_t;
using norms_t = typename kit_t::norms_t;
using norms_accumulator_t = typename kit_t::norms_accumulator_t;

constant constexpr uint metal_unknown_k = 0xFFFFFFFFu;
constant constexpr ulong metal_sentinel_k = 0xFFFFFFFFFFFFFFFFul;
constant constexpr ulong metal_expanded_bit_k = 1ul << 32;

/** Probe rounds a visited claim takes before it starves, each reading one aligned pair of slots, as the CUDA and
 *  HIP sets do. */
constant constexpr uint metal_probe_budget_k = 64;

struct metal_walk_t;

#pragma region Scored Nodes

/** One candidate in eight bytes, ordered as a raw word: the top 31 bits of the sign-flipped distance, the expanded
 *  bit, then the full node ID; the all-ones word is the empty slot. */
inline ulong metal_make(float distance, uint node_id) {
    uint flipped = as_type<uint>(distance);
    flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return (ulong(flipped >> 1) << 33) | node_id;
}
inline uint metal_distance_bits(ulong node) { return uint(node >> 33); }
inline float metal_distance_of(ulong node) {
    uint flipped = metal_distance_bits(node) << 1;
    flipped ^= (flipped & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return as_type<float>(flipped);
}
inline uint metal_id_of(ulong node) { return uint(node); }
inline bool metal_expanded(ulong node) { return (node & metal_expanded_bit_k) != 0; }

/** Full bitonic sort of one 32-lane vector of nodes, ascending by lane. */
inline ulong metal_sort_lanes(ulong value, uint lane) {
    for (uint stage = 2; stage <= warp_t::width_k; stage <<= 1)
        for (uint stride = stage >> 1; stride; stride >>= 1) {
            ulong const partner = warp_t::shuffle_xor(value, stride);
            bool const ascending = (lane & stage) == 0, lower = (lane & stride) == 0;
            ulong const nearer = value < partner ? value : partner, further = value < partner ? partner : value;
            value = ascending == lower ? nearer : further;
        }
    return value;
}

/** The register pool of @c 32 x @p slots_ nodes: rank r at (slot r/32, lane r%32), always ascending. */
template <uint slots_>
struct metal_pool {
    ulong entries[slots_];

    void clear() {
        for (uint slot = 0; slot != slots_; ++slot) entries[slot] = metal_sentinel_k;
    }

    /** Folds a staged 32-lane batch in, keeping the nearest; the batch descends against the ascending pool. */
    void insert_batch(ulong staged, uint lane) { insert_batch_spilling(staged, lane); }

    /** The same fold, answering this lane's entry the fold displaced - sentinels while the pool has room. */
    ulong insert_batch_spilling(ulong staged, uint lane) {
        ulong const descending = warp_t::shuffle_xor(metal_sort_lanes(staged, lane), warp_t::width_k - 1);
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
        for (uint stride = warp_t::width_k / 2; stride; stride >>= 1) {
            bool const lower = (lane & stride) == 0;
            for (uint slot = 0; slot < slots_; ++slot) {
                ulong const partner = warp_t::shuffle_xor(entries[slot], stride);
                entries[slot] = lower == (entries[slot] < partner) ? entries[slot] : partner;
            }
        }
        return displaced;
    }

    ulong nearest_unexpanded() {
        for (uint slot = 0; slot < slots_; ++slot) {
            uint const mask = warp_t::ballot(!metal_expanded(entries[slot]));
            if (mask) return warp_t::shuffle(entries[slot], ctz(mask));
        }
        return metal_sentinel_k;
    }
    void mark_expanded_at(ulong entry) {
        for (uint slot = 0; slot < slots_; ++slot)
            if (entries[slot] == entry) entries[slot] |= metal_expanded_bit_k;
    }
    uint radius() { return metal_distance_bits(warp_t::shuffle(entries[slots_ - 1], warp_t::width_k - 1)); }
    bool full() { return warp_t::shuffle(entries[slots_ - 1], warp_t::width_k - 1) != metal_sentinel_k; }
    ulong rank_entry(uint rank) {
        ulong result = metal_sentinel_k;
        for (uint slot = 0; slot < slots_; ++slot)
            if (slot == rank / warp_t::width_k) result = entries[slot];
        return warp_t::shuffle(result, rank % warp_t::width_k);
    }

    /** The frontier verbs of the CUDA register frontier. */
    void reset(thread metal_walk_t&) { clear(); }

    /** A walk over full ranks ends at their radius, so a further walk needs no budget renewed. */
    void renew() {}
    uint hops_budget() { return slots_ * warp_t::width_k * 4; }
    uint dropped() { return 0; }
    /** The fold's result for a batch of which only the @p admitted lanes can enter, as the CUDA pool's
     *  @c insert_each lands it: each takes its rank, ranks at and past it move up one, and the last rank drops. */
    void insert_each(ulong staged, uint admitted, uint lane) {
        while (admitted) {
            ulong const value = warp_t::shuffle(staged, ctz(admitted));
            admitted &= admitted - 1;
            uint rank = 0;
            for (uint slot = 0; slot != slots_; ++slot) rank += popcount(warp_t::ballot(entries[slot] < value));
            if (rank == slots_ * warp_t::width_k) continue; // An earlier entry already pushed the worst below it.
            // Highest slot first, so each slot's old last lane carries up before that slot is rewritten.
            for (uint slot = slots_; slot-- != 0;) {
                if ((slot + 1) * warp_t::width_k <= rank) break; // Lower slots keep their ranks.
                ulong const carried = slot ? warp_t::shuffle(entries[slot - 1], warp_t::width_k - 1) : metal_sentinel_k;
                ulong const shifted = warp_t::shuffle(entries[slot], lane ? lane - 1 : 0);
                uint const position = slot * warp_t::width_k + lane;
                if (position == rank) entries[slot] = value;
                else if (position > rank) entries[slot] = lane ? shifted : carried;
            }
        }
    }
    void push_batch(ulong staged, uint lane) {
        // Only entries below the worst kept one can enter: a few take their ranks, a crowd takes the fold.
        constexpr uint few_k = 3;
        ulong const worst = warp_t::shuffle(entries[slots_ - 1], warp_t::width_k - 1);
        uint const admitted = warp_t::ballot(staged < worst);
        if (popcount(admitted) <= few_k) insert_each(staged, admitted, lane);
        else insert_batch(staged, lane);
    }
    ulong pop_nearest(uint) {
        ulong const nearest = nearest_unexpanded();
        if (nearest != metal_sentinel_k) mark_expanded_at(nearest);
        return nearest;
    }
    bool exhausted() { return full(); }
    bool beyond_radius(ulong candidate) { return full() && metal_distance_bits(candidate) > radius(); }

    /** Hands @p emit the nearest @p count entries ascending, a warp-width per call, rank @b base+lane on each lane. */
    template <typename emit_type_>
    void for_each_rank(uint count, uint lane, thread emit_type_& emit) {
        for (uint base = 0; base < count; base += warp_t::width_k) emit(base + lane, rank_entry(base + lane));
    }

    /** The ranks are the pool itself, so nothing settles before a selection reads them. */
    void settle(uint) {}
};

/** The arena frontier, the CUDA one line for line: a two-slot register head plus a spill tail in the simdgroup's
 *  slice, refilled from the tail when the head drains, retiring after @c expansion_target expansions, and ranking
 *  everything it kept at the end, a head's width per round. */
struct metal_arena_frontier {
    metal_pool<2> head;
    device ulong* tail;
    uint tail_size, tail_capacity, tail_dropped;
    uint expansion_target, expansions;

    void reset(thread metal_walk_t& walk);

    /** Grants a further walk over what this frontier holds its own expansion budget. */
    void renew() { expansions = 0; }
    uint hops_budget() { return expansion_target * 4; }
    uint dropped() { return tail_dropped; }

    /** Appends a lane-wide batch to the tail, counting what passes its capacity instead of writing it. */
    void append(ulong value, uint lane) {
        bool const live = value != metal_sentinel_k;
        uint const live_mask = warp_t::ballot(live);
        uint const ordinal = popcount(live_mask & warp_t::below(lane));
        if (live && tail_size + ordinal < tail_capacity) tail[tail_size + ordinal] = value;
        uint const grown = min(tail_size + popcount(live_mask), tail_capacity);
        tail_dropped += tail_size + popcount(live_mask) - grown;
        tail_size = grown;
        warp_t::sync();
    }

    /** A batch at or past the head's radius goes straight to the tail, which only holds entries farther than the
     *  head's; otherwise the head's spill does. */
    void push_batch(ulong staged, uint lane) {
        if (head.full() && !warp_t::ballot(metal_distance_bits(staged) < head.radius())) return append(staged, lane);
        append(head.insert_batch_spilling(staged, lane), lane);
    }

    /** Promotes the nearest unexpanded tail entries into a drained head, tombstoning their slots, so every ID lives
     *  in one place; the spent head spills wholesale to the tail. */
    void refill(uint lane) {
        ulong best = metal_sentinel_k;
        uint best_index = 0;
        for (uint index = lane; index < tail_size; index += warp_t::width_k) {
            ulong const entry = tail[index];
            if (!metal_expanded(entry) && entry < best) best = entry, best_index = index;
        }
        warp_t::sync();
        if (!warp_t::ballot(best != metal_sentinel_k)) return;
        if (best != metal_sentinel_k) tail[best_index] = metal_sentinel_k;
        append(head.entries[0], lane);
        append(head.entries[1], lane);
        head.clear();
        head.insert_batch(best, lane);
    }

    ulong pop_nearest(uint lane) {
        // A consumed budget is the frontier's edge: the walk's drained-frontier exit seals it.
        if (exhausted()) return metal_sentinel_k;
        ulong nearest = head.nearest_unexpanded();
        if (nearest == metal_sentinel_k) {
            refill(lane);
            nearest = head.nearest_unexpanded();
        }
        if (nearest != metal_sentinel_k) {
            head.mark_expanded_at(nearest);
            ++expansions;
        }
        return nearest;
    }
    bool exhausted() { return expansions >= expansion_target; }
    bool beyond_radius(ulong) { return false; }

    /** Folds the nearest two lane-widths of everything the walk saw into the head, which then answers @c rank_entry
     *  for a selection, as the CUDA arena frontier does; call once, after the walk. */
    void settle(uint lane) {
        auto keep = [&](uint rank, ulong entry) {
            if (rank < warp_t::width_k) head.entries[0] = entry;
            else head.entries[1] = entry;
        };
        for_each_rank(2 * warp_t::width_k, lane, keep);
    }
    ulong rank_entry(uint rank) { return head.rank_entry(rank); }

    /** Drops the tail entries that cannot rank among the nearest @p count, as the CUDA arena frontier's
     *  @c keep_nearest does: a radix select of the count-th smallest distance key, eight bits a pass from the top,
     *  ties kept, each lane tallying by ballots the digits whose low five bits are its own index. */
    void keep_nearest(uint count, uint lane) {
        constexpr uint lane_bits_k = 5, high_bits_k = 3, bins_k = 8;
        uint prefix = 0, prefix_mask = 0, rank = count;
        for (uint pass = 0; pass != 4; ++pass) {
            uint const shift = 24 - 8 * pass;
            uint bins[bins_k] = {};
            // The head's two slots tally as the first two batches, then the tail a simdgroup-width at a time.
            for (uint base = 0; base < 2 * warp_t::width_k + tail_size; base += warp_t::width_k) {
                uint const index = base + lane - 2 * warp_t::width_k;
                ulong const entry = base == 0                 ? head.entries[0]
                                    : base == warp_t::width_k ? head.entries[1]
                                    : index < tail_size       ? tail[index]
                                                              : metal_sentinel_k;
                uint const key = metal_distance_bits(entry), digit = (key >> shift) & 0xFFu;
                uint own = warp_t::ballot(entry != metal_sentinel_k && (key & prefix_mask) == prefix);
                for (uint bit = 0; bit != lane_bits_k; ++bit) {
                    uint const set = warp_t::ballot(((digit >> bit) & 1u) != 0);
                    own &= (lane >> bit) & 1u ? set : ~set;
                }
                uint high[high_bits_k];
                for (uint bit = 0; bit != high_bits_k; ++bit)
                    high[bit] = warp_t::ballot(((digit >> (lane_bits_k + bit)) & 1u) != 0);
                for (uint bin = 0; bin != bins_k; ++bin) {
                    uint matching = own;
                    for (uint bit = 0; bit != high_bits_k; ++bit)
                        matching &= (bin >> bit) & 1u ? high[bit] : ~high[bit];
                    bins[bin] += popcount(matching);
                }
            }
            // Digits ascend bin-major, lane-minor; the uniform guard keeps every vote simdgroup-wide.
            uint below = 0;
            bool found = false;
            for (uint bin = 0; bin != bins_k; ++bin) {
                uint const inclusive = simd_prefix_inclusive_sum(bins[bin]);
                uint const total = warp_t::shuffle(inclusive, warp_t::width_k - 1);
                if (!found && below + total >= rank) {
                    uint const owner = ctz(warp_t::ballot(below + inclusive >= rank));
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
        // In place: a kept entry lands at or before its own slot, after the ballot over its batch's loads.
        uint kept = 0;
        for (uint base = 0; base < tail_size; base += warp_t::width_k) {
            uint const index = base + lane;
            ulong const entry = index < tail_size ? tail[index] : metal_sentinel_k;
            bool const keep = entry != metal_sentinel_k && metal_distance_bits(entry) <= prefix;
            uint const keeping = warp_t::ballot(keep);
            if (keep) tail[kept + popcount(keeping & warp_t::below(lane))] = entry;
            kept += popcount(keeping);
        }
        tail_size = kept;
        warp_t::sync();
    }

    /** Hands @p emit the nearest @p count entries of the head and the tail ascending, as the CUDA arena frontier
     *  does: each round folds every entry ranked after the previous round's last into a fresh two-slot pool, and
     *  past one round the tail first keeps only the entries that can rank. */
    template <typename emit_type_>
    void for_each_rank(uint count, uint lane, thread emit_type_& emit) {
        constexpr uint round_ranks_k = 2 * warp_t::width_k;
        if (count > round_ranks_k) keep_nearest(count, lane);
        ulong last = metal_sentinel_k;
        for (uint base = 0; base < count; base += round_ranks_k) {
            auto const unranked = [&](ulong entry) { return base == 0 || last < entry ? entry : metal_sentinel_k; };
            metal_pool<2> round;
            round.clear();
            // Once a round comes up short the walk saw nothing more, and every later rank is a sentinel.
            if (base == 0 || last != metal_sentinel_k) {
                round.insert_batch(unranked(head.entries[0]), lane);
                round.insert_batch(unranked(head.entries[1]), lane);
                for (uint tail_base = 0; tail_base < tail_size; tail_base += warp_t::width_k) {
                    uint const index = tail_base + lane;
                    round.insert_batch(index < tail_size ? unranked(tail[index]) : metal_sentinel_k, lane);
                }
            }
            emit(base + lane, round.entries[0]);
            emit(base + warp_t::width_k + lane, round.entries[1]);
            last = round.rank_entry(round_ranks_k - 1);
        }
    }
};

/** The frontier of a walk on the pool ladder's rung of @p slots_ slots; slot zero is the arena frontier. */
template <uint slots_>
struct metal_frontier_of {
    using type = metal_pool<slots_>;
};
template <>
struct metal_frontier_of<0> {
    using type = metal_arena_frontier;
};

#pragma endregion Scored Nodes

#pragma region Walk

/** Row @p row of a @b [rows,stride] matrix at @p base - the one place a row offset widens past 32 bits. */
template <typename value_type_>
inline device value_type_* metal_row(device value_type_* base, uint row, uint stride) {
    return base + ulong(row) * stride;
}

/** Avalanches all 64 bits of @p word - the host's @c mix_bits, the SplitMix64 finalizer. */
inline ulong metal_mix_bits(ulong word) {
    word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ul;
    word = (word ^ (word >> 27)) * 0x94D049BB133111EBul;
    return word ^ (word >> 31);
}

/** The random stream of @p lane walking for @p stream_key, seeded as @c vram_lane_generator seeds the CUDA and HIP
 *  ones. */
inline ulong metal_lane_generator(uint stream_key, uint lane) {
    return metal_mix_bits(ulong(stream_key) * warp_t::width_k + lane + 1);
}

/** One simdgroup's walk: the flat storage, its slice of scratch, the visited set, the random stream
 *  and the telemetry. */
struct metal_walk_t {
    device uint4* vectors;
    device norms_t* norms;
    device uint* neighbors;
    device atomic_uint* counters;

    device uint4* query;
    device uint4* cache;
    device ulong* visited;
    device ulong* scored;
    device ulong* sorted;
    device norms_t* cache_norms;
    device uint* ids;
    device dot_t* dots;
    device uint* selected_ids;
    device uint* kept_positions;

    /** Each cached candidate's distance to the query, staged once so the selection reads memory, not the pool. */
    device float* distances;
    device ulong* tail;

    uint row_fragments, lanes_per_row, steps_per_row, connectivity, metric, row_fragments_magic;
    uint tail_entries, expansion;
    bool has_norms;
    uint lane;

    uint hash_shift, epoch, saturations;
    ulong random;
    uint hops, scored_count, restarts, lock_attempts, lock_skips, reselections, locked_drops, tail_drops;

    norms_t norm_for(uint node_id) { return has_norms ? norms[node_id] : norms_t(0); }
    device uint* neighbors_for(uint node_id) { return metal_row(neighbors, node_id, connectivity); }
    device uint4* vector_for(uint node_id) { return metal_row(vectors, node_id, row_fragments); }
    device uint4* cache_row(uint row) { return cache + row * row_fragments; }

    /** The row holding flattened @p word, exact for every word under 2^31: the reciprocal never undershoots, and
     *  overshoots by at most one there. */
    uint row_of_word(uint word) {
        uint const row = uint((ulong(word) * row_fragments_magic) >> 31);
        return row - uint(row * row_fragments > word);
    }

    /** One SplitMix64 step, as @c splitmix64_t takes it. */
    ulong next_random() { return metal_mix_bits(random += 0x9E3779B97F4A7C15ul); }

    /** Marks @p node_id visited, answering whether it was fresh; equal IDs collapse to their lowest lane, and a
     *  saturated set reports a claim as visited, so the walk starves rather than loops. A lane with nothing to claim
     *  passes the sentinel, which no walk visits. The span is 16-byte aligned, since a probe loads an aligned pair. */
    bool test_and_set(uint node_id) {
        bool const active = node_id != metal_unknown_k;
        // Simdgroup-uniform: an idle call skips the matching votes, a shuffle per lane on Metal.
        if (!simd_any(active)) return false;
        uint const peers = warp_t::match_any(active ? node_id : (0xFFFFFFFFu ^ lane));
        bool fresh = active && (peers & warp_t::below(lane)) == 0;
        bool probing = fresh;
        ulong const claim = (ulong(epoch) << 32) | node_id;
        uint const mask = 0xFFFFFFFFu >> hash_shift;
        uint slot = (node_id * 0x9E3779B9u) >> hash_shift;
        for (uint round = 0; simd_any(probing) && round != metal_probe_budget_k; ++round) {
            // One 16-byte load covers the probed slot's aligned pair, so an even-slot collision resolves this round.
            ulong2 const pair = probing ? ((device ulong2 const*)visited)[slot >> 1] : ulong2(0);
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
            uint const rivals = warp_t::match_any(claiming ? slot : (0xFFFFFFFFu ^ lane));
            if (claiming && (rivals & warp_t::below(lane)) == 0) visited[slot] = claim, probing = false;
            warp_t::sync();
        }
        saturations += popcount(warp_t::ballot(probing));
        if (probing) fresh = false;
        warp_t::sync();
        return fresh;
    }

    /** Scores @p count rows of @p batch_ids against @p query, a sub-group of @c lanes_per_row lanes per row, landing
     *  each row's node in @c scored and its dot in @c dots; answers this lane's node, or the sentinel past @p count.
     *  A lane whose row lies past the batch re-reads the batch's first row instead of taking a guard, and each lane
     *  finalizes its own row, so the metric's epilogue runs once per lane. */
    ulong score_batch(device uint4 const* query_row, norms_t query_norms, device uint const* batch_ids, uint count) {
        uint const lanes_log2 = ctz(lanes_per_row);
        uint const rows_per_round = warp_t::width_k >> lanes_log2;
        uint const lane_in_row = lane & (lanes_per_row - 1);
        uint const subgroup = lane >> lanes_log2;
        bool const live = lane < count;
        // Requested before the rows, so the two round trips overlap.
        norms_t const target_norms = live ? norm_for(batch_ids[lane]) : norms_t(0);
        device uint4 const* const query = query_row + lane_in_row;
        // Simdgroup-uniform, so the reduction's shuffles stay legal: rounds past the batch own no row.
        for (uint round = 0; round != lanes_per_row && round * rows_per_round < count; ++round) {
            uint const row = round * rows_per_round + subgroup;
            device uint4 const* const stored = vector_for(batch_ids[row < count ? row : 0]) + lane_in_row;
            dot_t accumulator = dot_t(0);
            for (uint step = 0; step != steps_per_row; ++step) {
                uint const offset = step << lanes_log2;
                if (lane_in_row + offset < row_fragments)
                    accumulator = kit_t::accumulate_dot(stored[offset], query[offset], accumulator);
            }
            for (uint offset = lanes_per_row >> 1; offset; offset >>= 1)
                accumulator += warp_t::shuffle_xor(accumulator, offset);
            if (row < count && lane_in_row == 0) dots[row] = accumulator;
        }
        warp_t::sync();
        if (!live) return metal_sentinel_k;
        ulong const staged = metal_make(kit_t::distance(metric, dots[lane], query_norms, target_norms),
                                        batch_ids[lane]);
        scored[lane] = staged;
        return staged;
    }

    /** Stages the fresh IDs of one lane-wide read into @c ids, answering how many; lanes pass the sentinel for none. */
    uint stage_fresh(uint node_id) {
        bool const fresh = test_and_set(node_id);
        uint const fresh_mask = warp_t::ballot(fresh);
        if (fresh) ids[popcount(fresh_mask & warp_t::below(lane))] = node_id;
        warp_t::sync();
        return popcount(fresh_mask);
    }

    /** Draws one lane-wide batch of distinct, published landmarks in @b [first,bound) into @c ids: over the first
     *  few probes a lane meeting an unpublished row draws again, then each lane probes forward past unpublished and
     *  visited rows for one lap of the range, as the CUDA and HIP samplers do; answers how many, zero once every
     *  published row of the range is visited. */
    uint sample_landmarks(uint first, uint bound) {
        // Rows still inserting gather at the top of the range, which a redraw leaves far sooner than a lap.
        constexpr uint redraws_k = 16;
        uint const span = bound - first;
        uint draw = first + uint((ulong(uint(next_random())) * span) >> 32);
        bool fresh = false;
        uint const saturated = saturations;
        // The trip count is simdgroup-uniform, since every lane takes part in each collective claim, and a saturated
        // set reports every claim visited, so the lap ends at its first saturation instead of probing the whole range.
        uint const redraws = min(span, redraws_k);
        for (ulong probes = 0; probes != ulong(redraws) + span && !simd_all(fresh) && saturations == saturated;
             ++probes) {
            bool const published = !fresh && warp_t::load_acquire(neighbors_for(draw)) != metal_unknown_k;
            bool const claimed = test_and_set(published ? draw : metal_unknown_k);
            if (!fresh && !claimed)
                draw = !published && probes < redraws ? first + uint((ulong(uint(next_random())) * span) >> 32)
                       : draw + 1 == bound            ? first
                                                      : draw + 1;
            fresh = fresh || claimed;
        }
        uint const fresh_mask = warp_t::ballot(fresh);
        if (fresh) ids[popcount(fresh_mask & warp_t::below(lane))] = draw;
        warp_t::sync();
        return popcount(fresh_mask);
    }
};

/** What a walk does with every scored batch beside folding it into the pool. */
struct metal_fold_t {
    device uint* ids;
    device dot_t* products;
    uint wanted, written, scored;
    bool keeps;
    device uint4 const* heads;
    device float const* weights;
    uint heads_count;

    void absorb(thread metal_walk_t& walk, uint count) {
        if (!keeps) return;
        uint const writable = min(count, wanted - written);
        if (walk.lane < writable) {
            ids[written + walk.lane] = walk.ids[walk.lane];
            products[written + walk.lane] = walk.dots[walk.lane];
        }
        written += writable;
        scored += count;
    }
};

/** Scores one gathered batch, rescores it under indexer heads when the fold names any, folds it into the pool, and
 *  hands it to the fold. */
template <typename frontier_>
inline void metal_score_into(thread metal_walk_t& walk, thread frontier_& pool, norms_t query_norms, uint count,
                             thread metal_fold_t& fold) {
    walk.scored_count += count;
    ulong staged = walk.score_batch(walk.query, query_norms, walk.ids, count);
    if (fold.heads_count) {
        bool const live = walk.lane < count;
        float indexed = 0;
        for (uint head = 0; head != fold.heads_count; ++head) {
            walk.score_batch(fold.heads + head * walk.row_fragments, norms_t(0), walk.ids, count);
            if (live) indexed += fold.weights[head] * max(float(walk.dots[walk.lane]), 0.0f);
            warp_t::sync();
        }
        if (live) {
            staged = metal_make(1 - indexed, walk.ids[walk.lane]);
            walk.scored[walk.lane] = staged;
        }
    }
    pool.push_batch(staged, walk.lane);
    fold.absorb(walk, count);
}

/** Continues a greedy best-first walk over whatever @p pool holds, seeding from landmarks in @b [first,bound) and
 *  never following a neighbor at or past @p first_invisible; read-only on the graph, a transiently locked candidate
 *  held through one other expansion and dropped if still locked then, as the CUDA and HIP walks do. */
template <typename frontier_>
inline void metal_search_ahead(thread metal_walk_t& walk, thread frontier_& pool, norms_t query_norms, uint first,
                               uint bound, uint first_invisible, thread metal_fold_t& fold) {
    // The cap bounds one walk, while the telemetry adds up every walk the simdgroup takes.
    uint const hops_cap = pool.hops_budget();
    uint hops = 0;
    // The one candidate held for its second look, and the hop it was held at; a drained pool moves that hop out of
    // reach, so the next iteration looks at once.
    ulong held = metal_sentinel_k;
    uint held_at_hop = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            uint const count = walk.sample_landmarks(first, bound);
            if (!count) break; // Every published row below the bound is already visited.
            metal_score_into(walk, pool, query_norms, count, fold);
            seeding = false;
            continue;
        }
        bool const revisiting = held != metal_sentinel_k && hops != held_at_hop;
        ulong const candidate = revisiting ? held : pool.pop_nearest(walk.lane);
        if (revisiting) held = metal_sentinel_k;
        if (candidate == metal_sentinel_k) {
            if (held != metal_sentinel_k) {
                held_at_hop = ~0u;
                continue;
            }
            // A drained frontier before an exhausted walk re-seeds at one hop of the budget, so a walk whose full tail
            // drops every landmark still ends on the cap.
            if (pool.exhausted() || hops >= hops_cap) break;
            seeding = true;
            ++walk.restarts, ++hops;
            continue;
        }
        if (pool.beyond_radius(candidate) || hops >= hops_cap) break;
        ++hops, ++walk.hops;
        uint const candidate_id = metal_id_of(candidate);
        device uint* const row = walk.neighbors_for(candidate_id);
        uint const first_value = warp_t::shuffle(walk.lane == 0 ? warp_t::load_acquire(row) : metal_unknown_k, 0);
        // Every lane reads the candidate's payload, so every lane fences behind the leader's acquire.
        warp_t::fence();
        if (first_value == metal_unknown_k) {
            // Held through one other expansion instead of waited on; a second refusal drops it.
            if (revisiting || held != metal_sentinel_k) ++walk.locked_drops;
            else held = candidate, held_at_hop = hops;
            continue;
        }
        for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += warp_t::width_k) {
            uint const slot = slot_base + walk.lane;
            uint const neighbor_id = slot == 0                  ? first_value
                                     : slot < walk.connectivity ? warp_t::load_acquire(row + slot)
                                                                : metal_unknown_k;
            bool const usable = neighbor_id < first_invisible && neighbor_id != candidate_id;
            uint const count = walk.stage_fresh(usable ? neighbor_id : metal_unknown_k);
            if (count) metal_score_into(walk, pool, query_norms, count, fold);
        }
    }
    walk.locked_drops += held != metal_sentinel_k;
}

/** Greedy diversity selection over the pool's ascending prefix into @c selected_ids, backfilled with the nearest
 *  rejected candidates as the CUDA selection does; answers the kept count. */
template <typename frontier_>
inline uint metal_select_diverse(thread metal_walk_t& walk, thread frontier_& pool) {
    pool.settle(walk.lane);
    uint candidates = 0;
    for (uint base = 0; base < metal_cache_rows_k; base += warp_t::width_k) {
        ulong const entry = pool.rank_entry(base + walk.lane);
        uint const live = warp_t::ballot(entry != metal_sentinel_k);
        if (entry != metal_sentinel_k) {
            walk.ids[base + walk.lane] = metal_id_of(entry);
            walk.distances[base + walk.lane] = metal_distance_of(entry);
        }
        candidates += popcount(live);
        if (live != 0xFFFFFFFFu) break;
    }
    warp_t::sync();
    for (uint word = walk.lane; word < candidates * walk.row_fragments; word += warp_t::width_k) {
        uint const row = walk.row_of_word(word), fragment_index = word - row * walk.row_fragments;
        walk.cache_row(row)[fragment_index] = walk.vector_for(walk.ids[row])[fragment_index];
    }
    for (uint row = walk.lane; row < candidates; row += warp_t::width_k)
        walk.cache_norms[row] = walk.norm_for(walk.ids[row]);
    warp_t::sync();

    uint selected = 0;
    for (uint candidate = 0; candidate < candidates && selected < walk.connectivity; ++candidate) {
        float const to_query = walk.distances[candidate];
        bool diverse = true;
        for (uint base = 0; base < selected && diverse; base += warp_t::width_k) {
            bool lane_diverse = true;
            if (base + walk.lane < selected) {
                uint const kept = walk.selected_ids[base + walk.lane];
                dot_t accumulator = dot_t(0);
                for (uint fragment_index = 0; fragment_index != walk.row_fragments; ++fragment_index)
                    accumulator = kit_t::accumulate_dot(walk.cache_row(candidate)[fragment_index],
                                                        walk.cache_row(kept)[fragment_index], accumulator);
                lane_diverse = kit_t::distance(walk.metric, accumulator, walk.cache_norms[candidate],
                                               walk.cache_norms[kept]) >= to_query;
                ++walk.scored_count;
            }
            diverse = simd_all(lane_diverse);
        }
        if (diverse) {
            if (walk.lane == 0) walk.selected_ids[selected] = candidate;
            ++selected;
            warp_t::sync();
        }
    }
    uint const diverse_count = selected;
    for (uint candidate = 0, matched = 0; candidate < candidates && selected < walk.connectivity; ++candidate) {
        if (matched != diverse_count && walk.selected_ids[matched] == candidate) {
            ++matched;
            continue;
        }
        if (walk.lane == 0) walk.selected_ids[selected] = candidate;
        ++selected;
        warp_t::sync();
    }
    warp_t::sync();
    for (uint slot = walk.lane; slot < selected; slot += warp_t::width_k)
        walk.selected_ids[slot] = walk.ids[walk.selected_ids[slot]];
    warp_t::sync();
    return selected;
}

/** Merges a pre-sorted batch of @p count nodes into the ascending span @c sorted of @p size, keeping @p capacity. */
inline uint metal_merge_sorted(thread metal_walk_t& walk, ulong value, uint count, uint size, uint capacity) {
    bool const incoming = walk.lane < count;
    uint incoming_rank = incoming ? walk.lane : 0;
    for (uint index = 0; index != size; ++index)
        incoming_rank += incoming && metal_distance_bits(walk.sorted[index]) <= metal_distance_bits(value);
    ulong moved[2];
    uint moved_to[2];
    for (uint base = 0, batch = 0; base < size; base += warp_t::width_k, ++batch) {
        uint const index = base + walk.lane;
        bool const active = index < size;
        moved[batch] = active ? walk.sorted[index] : metal_sentinel_k;
        uint shift = 0;
        for (uint source = 0; source != count; ++source)
            shift += active && metal_distance_bits(warp_t::shuffle(value, source)) < metal_distance_bits(moved[batch]);
        moved_to[batch] = index + shift;
    }
    warp_t::sync();
    for (uint base = 0, batch = 0; base < size; base += warp_t::width_k, ++batch)
        if (base + walk.lane < size && moved_to[batch] < capacity) walk.sorted[moved_to[batch]] = moved[batch];
    if (incoming && incoming_rank < capacity) walk.sorted[incoming_rank] = value;
    warp_t::sync();
    return min(size + count, capacity);
}

/** Back-links @p node_id, which a wave's newcomers picked, chasing their edges from @p link_heads through
 *  @p link_next: appended into its free slots when they all fit, and otherwise re-selected over its live neighbors and
 *  every newcomer, ascending by distance to the node, keeping the diverse picks alone as the CPU's reverse refine does:
 *  a backfilled row stays full, so on a dot graph every later newcomer competes with the hubs and the small-norm nodes
 *  lose their last in-links. Edge @b e is newcomer @p first_id + e / connectivity; newcomers past the cache stay
 *  unlinked and are counted. The launch hands each node to one simdgroup, the only writer of its row. */
inline void metal_back_link_one(thread metal_walk_t& walk, uint node_id, device uint const* link_heads,
                                device uint const* link_next, uint first_id) {
    uint const node_row = metal_cache_rows_k - 1;
    uint const connectivity = min(walk.connectivity, node_row);
    device uint* const row = walk.neighbors_for(node_id);

    // The live neighbors first, a self-loop in slot 0 meaning none, then the newcomers.
    uint live_count = 0;
    for (uint base = 0; base < connectivity; base += warp_t::width_k) {
        uint const slot = base + walk.lane;
        uint const live_id = slot < connectivity ? warp_t::load_acquire(row + slot) : metal_unknown_k;
        bool const usable = live_id != metal_unknown_k && live_id != node_id;
        uint const usable_mask = warp_t::ballot(usable);
        if (usable) walk.ids[live_count + popcount(usable_mask & warp_t::below(walk.lane))] = live_id;
        live_count += popcount(usable_mask);
        warp_t::sync();
    }
    // A chain is serial, so one lane follows it.
    uint count = live_count, unlinked = 0;
    if (walk.lane == 0)
        for (uint edge = link_heads[node_id]; edge != metal_unknown_k; edge = link_next[edge]) {
            if (count < node_row) walk.ids[count++] = first_id + edge / walk.connectivity;
            else ++unlinked;
        }
    count = warp_t::shuffle(count, 0);
    walk.lock_skips += warp_t::shuffle(unlinked, 0);
    warp_t::sync();

    if (count <= connectivity) {
        for (uint slot = max(live_count, 1u) + walk.lane; slot < count; slot += warp_t::width_k)
            warp_t::store_relaxed(row + slot, walk.ids[slot]);
        warp_t::fence();
        if (walk.lane == 0 && !live_count && count) warp_t::store_release(row, walk.ids[0]);
        return;
    }

    ++walk.reselections;
    for (uint word = walk.lane; word < walk.row_fragments; word += warp_t::width_k)
        walk.cache_row(node_row)[word] = walk.vector_for(node_id)[word];
    warp_t::sync();
    norms_t const node_norms = walk.norm_for(node_id);
    walk.scored_count += count;

    uint sorted_size = 0;
    for (uint base = 0; base < count; base += warp_t::width_k) {
        uint const batch = min(count - base, warp_t::width_k);
        ulong const scored = walk.score_batch(walk.cache_row(node_row), node_norms, walk.ids + base, batch);
        ulong const sorted = metal_sort_lanes(
            walk.lane < batch ? metal_make(metal_distance_of(scored), base + walk.lane) : metal_sentinel_k, walk.lane);
        sorted_size = metal_merge_sorted(walk, sorted, batch, sorted_size, count);
    }
    for (uint word = walk.lane; word < count * walk.row_fragments; word += warp_t::width_k) {
        uint const cache_row = walk.row_of_word(word), fragment_index = word - cache_row * walk.row_fragments;
        walk.cache_row(cache_row)[fragment_index] = walk.vector_for(walk.ids[cache_row])[fragment_index];
    }
    for (uint cache_row = walk.lane; cache_row < count; cache_row += warp_t::width_k)
        walk.cache_norms[cache_row] = walk.norm_for(walk.ids[cache_row]);
    warp_t::sync();

    // The merge walk over the union, ascending, as cache rows.
    uint refined = 0;
    for (uint existing = 0; existing != sorted_size && refined != connectivity; ++existing) {
        float const candidate_distance = metal_distance_of(walk.sorted[existing]);
        uint const position = metal_id_of(walk.sorted[existing]);
        bool diverse = true;
        for (uint base = 0; base < refined && diverse; base += warp_t::width_k) {
            bool lane_diverse = true;
            if (base + walk.lane < refined) {
                uint const kept = walk.kept_positions[base + walk.lane];
                dot_t pair = dot_t(0);
                for (uint fragment_index = 0; fragment_index != walk.row_fragments; ++fragment_index)
                    pair = kit_t::accumulate_dot(walk.cache_row(position)[fragment_index],
                                                 walk.cache_row(kept)[fragment_index], pair);
                lane_diverse = kit_t::distance(walk.metric, pair, walk.cache_norms[position], walk.cache_norms[kept]) >=
                               candidate_distance;
                ++walk.scored_count;
            }
            diverse = simd_all(lane_diverse);
        }
        if (!diverse) continue;
        if (walk.lane == 0) walk.kept_positions[refined] = position;
        ++refined;
        warp_t::sync();
    }
    warp_t::sync();
    for (uint slot = walk.lane; slot < connectivity; slot += warp_t::width_k)
        if (slot)
            warp_t::store_relaxed(row + slot, slot < refined ? walk.ids[walk.kept_positions[slot]] : metal_unknown_k);
    warp_t::fence();
    if (walk.lane == 0) warp_t::store_release(row, walk.ids[walk.kept_positions[0]]);
}

/** Folds the walk's telemetry into the device counters, from its leader; a tally carries into its high word, since
 *  one batch's scored nodes outgrow 32 bits. */
inline void metal_flush(thread metal_walk_t& walk) {
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
    add(metal_tail_drops_k, walk.tail_drops);
}

/** Opens the walk of the simdgroup at @p slice over its scratch, with fresh telemetry. */
inline metal_walk_t metal_open(constant metal_arguments_t& arguments, device uint4* vectors, device norms_t* norms,
                               device uint* neighbors, device uchar* arena, device atomic_uint* counters, uint slice,
                               uint lane) {
    metal_walk_t walk;
    device uchar* cursor = metal_row(arena, slice, arguments.slice_bytes);
    auto const carve = [&](uint bytes) {
        device uchar* const carved = cursor;
        cursor += bytes;
        return carved;
    };
    uint const fragment_bytes = arguments.row_fragments * metal_fragment_bytes_k;
    walk.vectors = vectors, walk.norms = norms, walk.neighbors = neighbors, walk.counters = counters;
    walk.query = (device uint4*)carve(fragment_bytes);
    walk.cache = (device uint4*)carve(fragment_bytes * arguments.cache_rows);
    walk.visited = (device ulong*)carve(arguments.visited_slots * 8);
    walk.scored = (device ulong*)carve(metal_scored_span_bytes_k);
    walk.sorted = (device ulong*)carve(metal_sorted_span_bytes_k);
    walk.cache_norms = (device norms_t*)carve(metal_cache_norms_span_bytes_k);
    walk.ids = (device uint*)carve(metal_ids_span_bytes_k);
    walk.dots = (device dot_t*)carve(metal_dots_span_bytes_k);
    walk.selected_ids = (device uint*)carve(metal_selected_ids_span_bytes_k);
    walk.kept_positions = (device uint*)carve(metal_kept_positions_span_bytes_k);
    walk.distances = (device float*)carve(metal_distances_span_bytes_k);
    walk.tail = (device ulong*)carve(arguments.tail_entries * 8);
    walk.tail_entries = arguments.tail_entries, walk.expansion = arguments.expansion;
    walk.row_fragments = arguments.row_fragments, walk.lanes_per_row = arguments.lanes_per_row;
    walk.steps_per_row = arguments.steps_per_row, walk.connectivity = arguments.connectivity;
    walk.metric = arguments.metric, walk.row_fragments_magic = arguments.row_fragments_magic;
    walk.has_norms = arguments.has_norms != 0, walk.lane = lane;
    walk.hash_shift = arguments.hash_shift, walk.epoch = 0, walk.saturations = 0, walk.random = 0;
    walk.hops = 0, walk.scored_count = 0, walk.restarts = 0, walk.lock_attempts = 0, walk.lock_skips = 0;
    walk.reselections = 0, walk.locked_drops = 0, walk.tail_drops = 0;
    return walk;
}

void metal_arena_frontier::reset(thread metal_walk_t& walk) {
    head.clear();
    tail = walk.tail, tail_size = 0, tail_capacity = walk.tail_entries, tail_dropped = 0;
    expansion_target = walk.expansion, expansions = 0;
}

/** Stages @p row into the walk's query span and answers its norms. */
inline norms_t metal_stage_query(thread metal_walk_t& walk, device uint4 const* row) {
    norms_accumulator_t moments = norms_accumulator_t(0);
    for (uint fragment_index = walk.lane; fragment_index < walk.row_fragments; fragment_index += warp_t::width_k) {
        uint4 const value = row[fragment_index];
        walk.query[fragment_index] = value;
        moments = kit_t::accumulate_norms(value, moments);
    }
    warp_t::sync();
    return kit_t::partials_to_norms(simd_sum(moments));
}

#pragma endregion Walk

#pragma region Kernels

/** The chains a wave's newcomers thread their picks onto for the back-link launch behind them: each node's last edge,
 *  unknown for none, the edge each was threaded before, and the nodes any newcomer picked with their count. */
struct metal_links_t {
    device uint* heads;
    device uint* next;
    device uint* touched_ids;
    device atomic_uint* touched_count;
};

/** One simdgroup inserts one node at a time, assigning itself IDs from the batch counter, and threads its picks onto
 *  @p links. */
template <uint slots_>
inline void metal_insert(constant metal_arguments_t& arguments, device uint4* vectors, device norms_t* norms,
                         device uint* neighbors, device uint4 const* staging, device uchar* arena,
                         device atomic_uint* counters, metal_links_t links, uint slice, uint lane) {
    metal_walk_t walk = metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    while (true) {
        uint new_node_id = lane == 0 ? atomic_fetch_add_explicit(counters + metal_assigned_k, 1u, memory_order_relaxed)
                                     : 0;
        new_node_id = arguments.first_id + warp_t::shuffle(new_node_id, 0);
        if (new_node_id >= arguments.end_id) break;
        walk.epoch = arguments.epoch_base + (new_node_id - arguments.first_id) + 1;
        walk.random = metal_lane_generator(new_node_id, lane);

        // Land the payload, derive the norms, and park the query in one pass.
        device uint4 const* const staged = metal_row(staging, new_node_id - arguments.first_id, walk.row_fragments);
        device uint4* const owned = walk.vector_for(new_node_id);
        for (uint fragment_index = lane; fragment_index < walk.row_fragments; fragment_index += warp_t::width_k)
            owned[fragment_index] = staged[fragment_index];
        norms_t const new_norms = metal_stage_query(walk, staged);
        if (lane == 0 && walk.has_norms) norms[new_node_id] = new_norms;
        warp_t::sync();

        typename metal_frontier_of<slots_>::type pool;
        pool.reset(walk);
        metal_fold_t fold {};
        if (new_node_id != 0) metal_search_ahead(walk, pool, new_norms, 0, new_node_id, arguments.end_id, fold);
        walk.tail_drops += pool.dropped();
        uint const selected = metal_select_diverse(walk, pool);
        device uint* const own = walk.neighbors_for(new_node_id);
        for (uint slot = lane; slot < selected; slot += warp_t::width_k)
            if (slot > 0) warp_t::store_relaxed(own + slot, walk.selected_ids[slot]);
        // Every lane wrote the payload or the slots, so every lane fences ahead of the leader's release.
        warp_t::fence();
        if (lane == 0) {
            warp_t::store_release(own, selected ? walk.selected_ids[0] : new_node_id);
            atomic_fetch_add_explicit(counters + metal_published_k, 1u, memory_order_relaxed);
        }
        for (uint slot = lane; slot < selected; slot += warp_t::width_k) {
            uint const target = walk.selected_ids[slot];
            uint const edge = (new_node_id - arguments.first_id) * walk.connectivity + slot;
            uint const previous = warp_t::exchange_acquire(links.heads + target, edge);
            links.next[edge] = previous;
            if (previous == metal_unknown_k)
                links.touched_ids[atomic_fetch_add_explicit(links.touched_count, 1u, memory_order_relaxed)] = target;
        }
        warp_t::sync();
    }
    metal_flush(walk);
}

/** One simdgroup back-links one node the wave before it touched at a time, assigning itself the touched list's offsets
 *  from the launch counter, and resets the node's chain for the next wave. */
inline void metal_back_link(constant metal_arguments_t& arguments, device uint4* vectors, device norms_t* norms,
                            device uint* neighbors, device uchar* arena, device atomic_uint* counters,
                            metal_links_t links, uint slice, uint lane) {
    metal_walk_t walk = metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    uint const touched = atomic_load_explicit(links.touched_count, memory_order_relaxed);
    while (true) {
        uint offset = lane == 0 ? atomic_fetch_add_explicit(counters + metal_assigned_k, 1u, memory_order_relaxed) : 0;
        offset = warp_t::shuffle(offset, 0);
        if (offset >= touched) break;
        uint const node_id = links.touched_ids[offset];
        metal_back_link_one(walk, node_id, links.heads, links.next, arguments.first_id);
        if (lane == 0) links.heads[node_id] = metal_unknown_k;
    }
    metal_flush(walk);
}

/** One simdgroup re-selects one node at a time into the shadow matrix, read-only on the live one. */
template <uint slots_>
inline void metal_refine(constant metal_arguments_t& arguments, device uint4* vectors, device norms_t* norms,
                         device uint* neighbors, device uint const* node_ids, device uchar* arena,
                         device atomic_uint* counters, device uint* shadow, device uint const* part_bases, uint slice,
                         uint lane) {
    metal_walk_t walk = metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    while (true) {
        uint offset = lane == 0 ? atomic_fetch_add_explicit(counters + metal_assigned_k, 1u, memory_order_relaxed) : 0;
        offset = warp_t::shuffle(offset, 0);
        if (arguments.first_id + offset >= arguments.end_id) break;
        uint const node_id = arguments.use_node_ids ? node_ids[offset] : arguments.first_id + offset;
        walk.epoch = arguments.epoch_base + offset + 1;
        walk.random = metal_lane_generator(node_id, lane);
        metal_stage_query(walk, walk.vector_for(node_id));
        norms_t const query_norms = walk.norm_for(node_id);

        typename metal_frontier_of<slots_>::type pool;
        pool.reset(walk);
        metal_fold_t fold {};
        walk.test_and_set(node_id);
        // The walk across runs first, from the landmarks of the round's neighboring group of parts alone, the
        // rows `part_across` names to the CUDA and HIP refines, and both walks share one call site, so the walk
        // inlines once.
        uint sampling_first = 0, sampling_bound = 0;
        if (arguments.parts > 1) {
            uint part = 0;
            while (part + 1 < arguments.parts && part_bases[part + 1] <= node_id) ++part;
            uint const first = ((part >> arguments.round) ^ 1u) << arguments.round;
            uint const bound = first + (1u << arguments.round);
            if (first < arguments.parts) {
                sampling_first = part_bases[first];
                sampling_bound = bound < arguments.parts ? part_bases[bound] : arguments.sampling_bound;
            }
        }
#pragma unroll 1
        for (uint walk_number = sampling_first < sampling_bound ? 0 : 1; walk_number != 2; ++walk_number) {
            if (walk_number) {
                pool.renew();
                sampling_first = 0, sampling_bound = arguments.sampling_bound;
                device uint* const own = walk.neighbors_for(node_id);
                uint const first_value = warp_t::shuffle(lane == 0 ? warp_t::load_acquire(own) : 0u, 0);
                if (first_value != metal_unknown_k && first_value != node_id)
                    for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += warp_t::width_k) {
                        uint const slot = slot_base + lane;
                        uint const seed = slot == 0                  ? first_value
                                          : slot < walk.connectivity ? warp_t::load_acquire(own + slot)
                                                                     : metal_unknown_k;
                        uint const count = walk.stage_fresh(seed != node_id ? seed : metal_unknown_k);
                        if (count) metal_score_into(walk, pool, query_norms, count, fold);
                    }
            }
            metal_search_ahead(walk, pool, query_norms, sampling_first, sampling_bound, arguments.sampling_bound, fold);
        }
        walk.tail_drops += pool.dropped();
        uint const selected = metal_select_diverse(walk, pool);
        device uint* const shadow_row = metal_row(shadow, node_id, walk.connectivity);
        for (uint slot = lane; slot < walk.connectivity; slot += warp_t::width_k)
            shadow_row[slot] = slot < selected ? walk.selected_ids[slot] : slot == 0 ? node_id : metal_unknown_k;
        warp_t::sync();
    }
    metal_flush(walk);
}

/** One simdgroup walks one query at a time, assigning itself queries from the batch counter, landing its nearest IDs
 *  and distances, or under @p fold.keeps the pairs it scored and their count, ranked under indexer heads when the
 *  fold names any. Slot zero walks the arena frontier and ranks everything it kept. */
template <uint slots_>
inline void metal_search(constant metal_arguments_t& arguments, device uint4* vectors, device norms_t* norms,
                         device uint* neighbors, device uint4 const* queries, device uchar* arena,
                         device atomic_uint* counters, device uint* result_ids, device float* result_distances,
                         device dot_t* products, device uint* scored_counts, device uint4 const* heads,
                         device float const* weights, bool keeps, uint slice, uint lane) {
    metal_walk_t walk = metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    while (true) {
        uint query = lane == 0 ? atomic_fetch_add_explicit(counters + metal_assigned_k, 1u, memory_order_relaxed) : 0;
        query = warp_t::shuffle(query, 0);
        if (query >= arguments.queries_count) break;
        walk.epoch = arguments.epoch_base + query + 1;
        walk.random = metal_lane_generator(query, lane);
        norms_t const query_norms = metal_stage_query(walk, metal_row(queries, query, walk.row_fragments));
        typename metal_frontier_of<slots_>::type pool;
        pool.reset(walk);
        metal_fold_t fold {};
        if (keeps) {
            fold.keeps = true;
            fold.ids = metal_row(result_ids, query, arguments.count);
            fold.products = metal_row(products, query, arguments.count);
            fold.wanted = arguments.count;
        }
        if (arguments.heads_count) {
            fold.heads = metal_row(heads, query, arguments.heads_count * walk.row_fragments);
            fold.weights = metal_row(weights, query, arguments.heads_count);
            fold.heads_count = arguments.heads_count;
        }
        metal_search_ahead(walk, pool, query_norms, 0, arguments.sampling_bound, arguments.sampling_bound, fold);
        walk.tail_drops += pool.dropped();
        if (keeps) {
            for (uint slot = fold.written + lane; slot < arguments.count; slot += warp_t::width_k)
                fold.ids[slot] = metal_unknown_k;
            if (lane == 0) scored_counts[query] = fold.scored;
            continue;
        }
        device uint* const ids_row = metal_row(result_ids, query, arguments.count);
        device float* const distances_row = metal_row(result_distances, query, arguments.count);
        uint const metric = walk.metric, count = arguments.count;
        auto land = [&](uint rank, ulong entry) {
            if (rank >= count) return;
            ids_row[rank] = metal_id_of(entry);
            distances_row[rank] = kit_t::convert_to_distance(metric, metal_distance_of(entry));
        };
        pool.for_each_rank(count, lane, land);
    }
    metal_flush(walk);
}

/** Rebuilds the norms of rows from @c first_id to @c end_id, one simdgroup per row. */
[[host_name("metal_derive_norms_kernel")]] kernel void metal_derive_norms_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint4* vectors [[buffer(metal_vectors_buffer_k)]], device norms_t* norms [[buffer(metal_norms_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint const slices = groups * metal_simdgroups_k;
    for (uint row = arguments.first_id + group * metal_simdgroups_k + simdgroup; row < arguments.end_id;
         row += slices) {
        device uint4 const* const stored = metal_row(vectors, row, arguments.row_fragments);
        norms_accumulator_t moments = norms_accumulator_t(0);
        for (uint fragment_index = lane; fragment_index < arguments.row_fragments; fragment_index += warp_t::width_k)
            moments = kit_t::accumulate_norms(stored[fragment_index], moments);
        moments = simd_sum(moments);
        if (lane == 0) norms[row] = kit_t::partials_to_norms(moments);
    }
}

/*  The entry points, one instantiation per rung the host's ladder names: 2, 4, 8 and 16 slots of 32 lanes, and
 *  slot zero's arena frontier past the widest pool, each under the name the host looks it up by. */

template <uint slots_>
kernel void metal_insert_kernel(constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
                                device uint4* vectors [[buffer(metal_vectors_buffer_k)]],
                                device norms_t* norms [[buffer(metal_norms_buffer_k)]],
                                device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
                                device uint4 const* staging [[buffer(metal_staging_buffer_k)]],
                                device uchar* arena [[buffer(metal_arena_buffer_k)]],
                                device atomic_uint* counters [[buffer(metal_counters_buffer_k)]],
                                device uint* link_heads [[buffer(metal_link_heads_buffer_k)]],
                                device uint* link_next [[buffer(metal_link_next_buffer_k)]],
                                device uint* touched_ids [[buffer(metal_touched_ids_buffer_k)]],
                                device atomic_uint* touched_count [[buffer(metal_touched_count_buffer_k)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    metal_insert<slots_>(arguments, vectors, norms, neighbors, staging, arena, counters,
                         {link_heads, link_next, touched_ids, touched_count}, group * metal_simdgroups_k + simdgroup,
                         lane);
}

template [[host_name("metal_insert_kernel_0")]] kernel void metal_insert_kernel<0>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("metal_insert_kernel_2")]] kernel void metal_insert_kernel<2>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("metal_insert_kernel_4")]] kernel void metal_insert_kernel<4>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("metal_insert_kernel_8")]] kernel void metal_insert_kernel<8>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("metal_insert_kernel_16")]] kernel void metal_insert_kernel<16>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);

/** One back-link launch per wave, over the nodes its insert launch touched. */
[[host_name("metal_back_link_kernel")]] kernel void metal_back_link_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint4* vectors [[buffer(metal_vectors_buffer_k)]], device norms_t* norms [[buffer(metal_norms_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]], device uchar* arena [[buffer(metal_arena_buffer_k)]],
    device atomic_uint* counters [[buffer(metal_counters_buffer_k)]],
    device uint* link_heads [[buffer(metal_link_heads_buffer_k)]],
    device uint* link_next [[buffer(metal_link_next_buffer_k)]],
    device uint* touched_ids [[buffer(metal_touched_ids_buffer_k)]],
    device atomic_uint* touched_count [[buffer(metal_touched_count_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    metal_back_link(arguments, vectors, norms, neighbors, arena, counters,
                    {link_heads, link_next, touched_ids, touched_count}, group * metal_simdgroups_k + simdgroup, lane);
}

template <uint slots_>
kernel void metal_refine_kernel(constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
                                device uint4* vectors [[buffer(metal_vectors_buffer_k)]],
                                device norms_t* norms [[buffer(metal_norms_buffer_k)]],
                                device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
                                device uint const* node_ids [[buffer(metal_staging_buffer_k)]],
                                device uchar* arena [[buffer(metal_arena_buffer_k)]],
                                device atomic_uint* counters [[buffer(metal_counters_buffer_k)]],
                                device uint* shadow [[buffer(metal_shadow_buffer_k)]],
                                device uint const* part_bases [[buffer(metal_part_bases_buffer_k)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    metal_refine<slots_>(arguments, vectors, norms, neighbors, node_ids, arena, counters, shadow, part_bases,
                         group * metal_simdgroups_k + simdgroup, lane);
}

template [[host_name("metal_refine_kernel_0")]] kernel void metal_refine_kernel<0>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, device uint const*, uint, uint, uint);
template [[host_name("metal_refine_kernel_2")]] kernel void metal_refine_kernel<2>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, device uint const*, uint, uint, uint);
template [[host_name("metal_refine_kernel_4")]] kernel void metal_refine_kernel<4>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, device uint const*, uint, uint, uint);
template [[host_name("metal_refine_kernel_8")]] kernel void metal_refine_kernel<8>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, device uint const*, uint, uint, uint);
template [[host_name("metal_refine_kernel_16")]] kernel void metal_refine_kernel<16>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, device uint const*, uint, uint, uint);

/*  Searches and graph products share one entry point per width, plus slot zero's arena frontier past the widest
 *  pool: the word at `metal_keeps_buffer_k` says which the launch lands. */

template <uint slots_>
kernel void metal_search_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint4* vectors [[buffer(metal_vectors_buffer_k)]], device norms_t* norms [[buffer(metal_norms_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint4 const* queries [[buffer(metal_staging_buffer_k)]],
    device uchar* arena [[buffer(metal_arena_buffer_k)]],
    device atomic_uint* counters [[buffer(metal_counters_buffer_k)]], device uint* ids [[buffer(metal_ids_buffer_k)]],
    device float* distances [[buffer(metal_distances_buffer_k)]],
    device dot_t* products [[buffer(metal_products_buffer_k)]],
    device uint* scored [[buffer(metal_scored_counts_buffer_k)]],
    device uint4 const* heads [[buffer(metal_heads_buffer_k)]],
    device float const* weights [[buffer(metal_weights_buffer_k)]],
    constant uint& keeps [[buffer(metal_keeps_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    metal_search<slots_>(arguments, vectors, norms, neighbors, queries, arena, counters, ids, distances, products,
                         scored, heads, weights, keeps != 0, group * metal_simdgroups_k + simdgroup, lane);
}

template [[host_name("metal_search_kernel_0")]] kernel void metal_search_kernel<0>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, uint);
template [[host_name("metal_search_kernel_2")]] kernel void metal_search_kernel<2>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, uint);
template [[host_name("metal_search_kernel_4")]] kernel void metal_search_kernel<4>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, uint);
template [[host_name("metal_search_kernel_8")]] kernel void metal_search_kernel<8>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, uint);
template [[host_name("metal_search_kernel_16")]] kernel void metal_search_kernel<16>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, uint);

#pragma endregion Kernels

} // namespace unum::usearch::dense
