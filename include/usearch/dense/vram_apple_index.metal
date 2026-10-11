/**
 *  @file include/usearch/dense/vram_apple_index.metal
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Metal walk, insert, refine and search kernels of the dense Apple index - the MSL port of
 *      @c vram_index.cuh and the kernels of @c vram_nvidia_index.cuh.
 *
 *  One 32-wide simdgroup walks one query or inserts one node, with the same packed scored nodes,
 *  register pools, exact visited sets, slot-0 lock words and admission ramp as the CUDA design.
 *  Metal differs in three places. Its atomics are relaxed, so acquire and release are spelled as a
 *  relaxed access beside a device-scope fence. It has no value matching, so lanes collapse equal IDs
 *  through a shuffle loop. And every per-simdgroup scratch span - the query, the candidate cache, the
 *  visited set and the staged batch - lives in one device slice rather than threadgroup memory,
 *  which caps at 32 KB and would bound the dimensions. The host prepends @c vram_apple_metric.metal
 *  and a @c dense_metal_kit_t alias before this file.
 */

namespace unum {
namespace usearch {

using kit_t = dense_metal_kit_t;
using dot_t = typename kit_t::dot_t;
using norms_t = typename kit_t::norms_t;

constant constexpr uint dense_metal_lanes_k = 32;
constant constexpr uint dense_metal_simdgroups_k = 4;
constant constexpr uint dense_metal_cache_rows_k = 64;
constant constexpr uint dense_metal_unknown_k = 0xFFFFFFFFu;
constant constexpr ulong dense_metal_sentinel_k = 0xFFFFFFFFFFFFFFFFul;
constant constexpr ulong dense_metal_expanded_bit_k = 1ul << 32;
constant constexpr uint dense_metal_probe_budget_k = 64;
constant constexpr uint dense_metal_restarts_cap_k = 8;
constant constexpr uint dense_metal_peek_retries_k = 16;
constant constexpr uint dense_metal_lock_budget_k = 4096;
constant constexpr uint dense_metal_admission_floor_k = 32;

/** The counters words: the batch's assigned rows, the index's published rows, then the telemetry. */
enum dense_metal_counter_t : uint {
    dense_metal_assigned_k = 0,
    dense_metal_published_k,
    dense_metal_hops_k,
    dense_metal_scored_k,
    dense_metal_restarts_k,
    dense_metal_lock_attempts_k,
    dense_metal_lock_skips_k,
    dense_metal_reselections_k,
    dense_metal_locked_drops_k,
    dense_metal_saturations_k,
    dense_metal_changed_k,
};

/** Every scalar a launch carries, mirrored by @c dense_metal_arguments_t on the host word for word. */
struct dense_metal_arguments_t {
    uint row_fragments, lanes_per_row, steps_per_row, connectivity;
    uint metric, has_norms, first_id, end_id;
    uint sampling_bound, queries_count, count, epoch_base;
    uint visited_slots, hash_shift, cache_rows, slice_bytes;
    uint heads_count, row_fragments_magic, use_node_ids, padding;
};

#pragma region Lanes

inline uint dense_metal_ballot(bool predicate) { return uint(simd_vote::vote_t(simd_ballot(predicate))); }
inline uint dense_metal_below(ushort lane) { return (1u << lane) - 1u; }
inline void dense_metal_sync() { simdgroup_barrier(mem_flags::mem_device); }

inline ulong dense_metal_shuffle(ulong value, ushort source_lane) {
    return as_type<ulong>(simd_shuffle(as_type<uint2>(value), source_lane));
}
inline ulong dense_metal_shuffle_xor(ulong value, ushort lane_mask) {
    return as_type<ulong>(simd_shuffle_xor(as_type<uint2>(value), lane_mask));
}

/** The lanes holding the same @p value as this one: Metal has no value matching, so every lane meets every other. */
inline uint dense_metal_match_any(uint value) {
    uint peers = 0;
    for (ushort lane = 0; lane != dense_metal_lanes_k; ++lane)
        peers |= uint(simd_shuffle(value, lane) == value) << lane;
    return peers;
}

inline device atomic_uint* dense_metal_atomic(device uint* word) { return (device atomic_uint*)word; }
inline void dense_metal_fence() {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
inline uint dense_metal_load_acquire(device uint* word) {
    uint const value = atomic_load_explicit(dense_metal_atomic(word), memory_order_relaxed);
    dense_metal_fence();
    return value;
}
inline void dense_metal_store_relaxed(device uint* word, uint value) {
    atomic_store_explicit(dense_metal_atomic(word), value, memory_order_relaxed);
}
inline void dense_metal_store_release(device uint* word, uint value) {
    dense_metal_fence();
    atomic_store_explicit(dense_metal_atomic(word), value, memory_order_relaxed);
}
inline uint dense_metal_exchange_acquire(device uint* word, uint value) {
    uint const previous = atomic_exchange_explicit(dense_metal_atomic(word), value, memory_order_relaxed);
    dense_metal_fence();
    return previous;
}

#pragma endregion Lanes

#pragma region Scored Nodes

/** One candidate in eight bytes, ordered as a raw word: the top 31 bits of the sign-flipped distance, the expanded
 *  bit, then the full node ID; the all-ones word is the empty slot. */
inline ulong dense_metal_make(float distance, uint node_id) {
    uint flipped = as_type<uint>(distance);
    flipped ^= (flipped & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return (ulong(flipped >> 1) << 33) | node_id;
}
inline uint dense_metal_distance_bits(ulong node) { return uint(node >> 33); }
inline float dense_metal_distance_of(ulong node) {
    uint flipped = dense_metal_distance_bits(node) << 1;
    flipped ^= (flipped & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return as_type<float>(flipped);
}
inline uint dense_metal_id_of(ulong node) { return uint(node); }
inline bool dense_metal_expanded(ulong node) { return (node & dense_metal_expanded_bit_k) != 0; }

/** Full bitonic sort of one 32-lane vector of nodes, ascending by lane. */
inline ulong dense_metal_sort_lanes(ulong value, ushort lane) {
    for (uint stage = 2; stage <= dense_metal_lanes_k; stage <<= 1)
        for (uint stride = stage >> 1; stride; stride >>= 1) {
            ulong const partner = dense_metal_shuffle_xor(value, ushort(stride));
            bool const ascending = (lane & stage) == 0, lower = (lane & stride) == 0;
            ulong const nearer = value < partner ? value : partner, further = value < partner ? partner : value;
            value = ascending == lower ? nearer : further;
        }
    return value;
}

/** The register pool of @c 32 x @p slots_ nodes: rank r at (slot r/32, lane r%32), always ascending. */
template <uint slots_>
struct dense_metal_pool {
    ulong entries[slots_];

    void clear() {
        for (uint slot = 0; slot != slots_; ++slot) entries[slot] = dense_metal_sentinel_k;
    }

    /** Folds a staged 32-lane batch in, keeping the nearest; the batch descends against the ascending pool. */
    void insert_batch(ulong staged, ushort lane) {
        ulong const descending = dense_metal_shuffle_xor(dense_metal_sort_lanes(staged, lane), dense_metal_lanes_k - 1);
        ulong const last = entries[slots_ - 1];
        entries[slots_ - 1] = last < descending ? last : descending;
        for (uint stride = slots_ / 2; stride >= 1; stride >>= 1)
            for (uint base = 0; base < slots_; ++base)
                if ((base & stride) == 0 && (base | stride) < slots_) {
                    ulong const lower_entry = entries[base], upper_entry = entries[base | stride];
                    entries[base] = lower_entry < upper_entry ? lower_entry : upper_entry;
                    entries[base | stride] = lower_entry < upper_entry ? upper_entry : lower_entry;
                }
        for (uint stride = dense_metal_lanes_k / 2; stride; stride >>= 1) {
            bool const lower = (lane & stride) == 0;
            for (uint slot = 0; slot < slots_; ++slot) {
                ulong const partner = dense_metal_shuffle_xor(entries[slot], ushort(stride));
                entries[slot] = lower == (entries[slot] < partner) ? entries[slot] : partner;
            }
        }
    }

    ulong nearest_unexpanded() {
        for (uint slot = 0; slot < slots_; ++slot) {
            uint const mask = dense_metal_ballot(!dense_metal_expanded(entries[slot]));
            if (mask) return dense_metal_shuffle(entries[slot], ushort(ctz(mask)));
        }
        return dense_metal_sentinel_k;
    }
    void mark_expanded_at(ulong entry) {
        for (uint slot = 0; slot < slots_; ++slot)
            if (entries[slot] == entry) entries[slot] |= dense_metal_expanded_bit_k;
    }
    uint radius() {
        return dense_metal_distance_bits(dense_metal_shuffle(entries[slots_ - 1], dense_metal_lanes_k - 1));
    }
    bool full() { return dense_metal_shuffle(entries[slots_ - 1], dense_metal_lanes_k - 1) != dense_metal_sentinel_k; }
    ulong rank_entry(uint rank) {
        ulong result = dense_metal_sentinel_k;
        for (uint slot = 0; slot < slots_; ++slot)
            if (slot == rank / dense_metal_lanes_k) result = entries[slot];
        return dense_metal_shuffle(result, ushort(rank % dense_metal_lanes_k));
    }

    /** The frontier verbs of the CUDA register frontier. */
    void push_batch(ulong staged, ushort lane) {
        if (slots_ <= 4 && full() && !dense_metal_ballot(dense_metal_distance_bits(staged) < radius())) return;
        insert_batch(staged, lane);
    }
    ulong pop_nearest() {
        ulong const nearest = nearest_unexpanded();
        if (nearest != dense_metal_sentinel_k) mark_expanded_at(nearest);
        return nearest;
    }
    bool beyond_radius(ulong candidate) { return full() && dense_metal_distance_bits(candidate) > radius(); }
};

#pragma endregion Scored Nodes

#pragma region Walk

/** One simdgroup's walk: the planes, its slice of scratch, the visited set, the random stream and the telemetry. */
struct dense_metal_walk_t {
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

    uint row_fragments, lanes_per_row, steps_per_row, connectivity, metric, row_fragments_magic;
    bool has_norms;
    ushort lane;

    uint hash_shift, epoch, saturations;
    ulong random;
    uint hops, scored_count, restarts, lock_attempts, lock_skips, reselections, locked_drops;

    norms_t norm_for(uint node_id) { return has_norms ? norms[node_id] : norms_t(0); }
    device uint* neighbors_for(uint node_id) { return neighbors + ulong(node_id) * connectivity; }
    device uint4* vector_for(uint node_id) { return vectors + ulong(node_id) * row_fragments; }
    device uint4* cache_row(uint row) { return cache + ulong(row) * row_fragments; }
    uint row_of_word(uint word) { return uint((ulong(word) * row_fragments_magic) >> 31); }

    ulong next_random() {
        ulong mixed = (random += 0x9E3779B97F4A7C15ul);
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ul;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBul;
        return mixed ^ (mixed >> 31);
    }

    /** Marks @p node_id visited, answering whether it was fresh; equal IDs collapse to their lowest lane, and a
     *  saturated set reports a claim as visited, so the walk starves rather than loops. */
    bool test_and_set(uint node_id, bool active) {
        uint const peers = dense_metal_match_any(active ? node_id : (0xFFFFFFFFu ^ lane));
        bool fresh = active && (peers & dense_metal_below(lane)) == 0;
        bool probing = fresh;
        ulong const claim = (ulong(epoch) << 32) | node_id;
        uint const mask = 0xFFFFFFFFu >> hash_shift;
        uint slot = (node_id * 0x9E3779B9u) >> hash_shift;
        for (uint round = 0; simd_any(probing) && round != dense_metal_probe_budget_k; ++round) {
            ulong const observed = probing ? visited[slot] : 0;
            bool claiming = false;
            if (probing) {
                if (observed == claim) fresh = probing = false;
                else if (uint(observed >> 32) == epoch) slot = (slot + 1) & mask;
                else claiming = true;
            }
            uint const rivals = dense_metal_match_any(claiming ? slot : (0xFFFFFFFFu ^ lane));
            if (claiming && (rivals & dense_metal_below(lane)) == 0) visited[slot] = claim, probing = false;
            dense_metal_sync();
        }
        saturations += popcount(dense_metal_ballot(probing));
        if (probing) fresh = false;
        dense_metal_sync();
        return fresh;
    }

    /** Scores @p count rows of @p batch_ids against @p query, a sub-group of @c lanes_per_row lanes per row, landing
     *  each row's node in @c scored and its dot in @c dots; answers this lane's node, or the sentinel past @p count. */
    ulong score_batch(device uint4 const* query_row, norms_t query_norms, device uint const* batch_ids, uint count) {
        uint const lanes_log2 = ctz(lanes_per_row);
        uint const rows_per_round = dense_metal_lanes_k >> lanes_log2;
        uint const lane_in_row = lane & (lanes_per_row - 1);
        uint const subgroup = lane >> lanes_log2;
        for (uint round = 0; round != lanes_per_row; ++round) {
            uint const row = round * rows_per_round + subgroup;
            bool const live = row < count;
            uint const node_id = live ? batch_ids[row] : 0;
            device uint4 const* const stored = vector_for(node_id);
            dot_t accumulator = dot_t(0);
            for (uint step = 0; step != steps_per_row; ++step) {
                uint const fragment_index = lane_in_row + (step << lanes_log2);
                if (live && fragment_index < row_fragments)
                    accumulator = kit_t::dot(stored[fragment_index], query_row[fragment_index], accumulator);
            }
            for (uint offset = lanes_per_row >> 1; offset; offset >>= 1)
                accumulator += simd_shuffle_xor(accumulator, ushort(offset));
            if (live && lane_in_row == 0) {
                scored[row] = dense_metal_make(kit_t::distance(metric, accumulator, query_norms, norm_for(node_id)),
                                               node_id);
                dots[row] = accumulator;
            }
        }
        dense_metal_sync();
        return lane < count ? scored[lane] : dense_metal_sentinel_k;
    }

    /** Stages the fresh IDs of one lane-wide read into @c ids, answering how many. */
    uint stage_fresh(uint node_id, bool usable) {
        bool const fresh = test_and_set(node_id, usable);
        uint const fresh_mask = dense_metal_ballot(fresh);
        if (fresh) ids[popcount(fresh_mask & dense_metal_below(lane))] = node_id;
        dense_metal_sync();
        return popcount(fresh_mask);
    }

    /** Draws one lane-wide batch of distinct, published landmarks below @p bound into @c ids, answering how many. */
    uint sample_landmarks(uint bound) {
        uint draw = uint((ulong(uint(next_random())) * bound) >> 32);
        bool published = false;
        for (uint attempt = 0; attempt != 4 && !published; ++attempt) {
            published = dense_metal_load_acquire(neighbors_for(draw)) != dense_metal_unknown_k;
            if (!published) draw = draw + 1 >= bound ? 0 : draw + 1;
        }
        return stage_fresh(draw, published);
    }
};

/** What a walk does with every scored batch beside folding it into the pool. */
struct dense_metal_fold_t {
    device uint* ids;
    device dot_t* products;
    uint wanted, written, scored;
    bool keeps;
    device uint4 const* heads;
    device float const* weights;
    uint heads_count;

    void absorb(thread dense_metal_walk_t& walk, uint count) {
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
template <uint slots_>
inline void dense_metal_score_into(thread dense_metal_walk_t& walk, thread dense_metal_pool<slots_>& pool,
                                   norms_t query_norms, uint count, thread dense_metal_fold_t& fold) {
    walk.scored_count += count;
    ulong staged = walk.score_batch(walk.query, query_norms, walk.ids, count);
    if (fold.heads_count) {
        bool const live = walk.lane < count;
        float indexed = 0;
        for (uint head = 0; head != fold.heads_count; ++head) {
            walk.score_batch(fold.heads + ulong(head) * walk.row_fragments, norms_t(0), walk.ids, count);
            if (live) indexed += fold.weights[head] * max(float(walk.dots[walk.lane]), 0.0f);
            dense_metal_sync();
        }
        if (live) {
            staged = dense_metal_make(1 - indexed, walk.ids[walk.lane]);
            walk.scored[walk.lane] = staged;
        }
    }
    pool.push_batch(staged, walk.lane);
    fold.absorb(walk, count);
}

/** Continues a greedy best-first walk over whatever @p pool holds, seeding from landmarks below @p bound and never
 *  following a neighbor at or past @p first_invisible; read-only on the graph. */
template <uint slots_>
inline void dense_metal_search_ahead(thread dense_metal_walk_t& walk, thread dense_metal_pool<slots_>& pool,
                                     norms_t query_norms, uint bound, uint first_invisible,
                                     thread dense_metal_fold_t& fold) {
    // The caps bound one walk, while the telemetry adds up every walk the simdgroup takes.
    uint const hops_cap = slots_ * dense_metal_lanes_k * 4;
    uint hops = 0, restarts = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            uint const count = walk.sample_landmarks(bound);
            if (count) dense_metal_score_into(walk, pool, query_norms, count, fold);
            seeding = false;
            continue;
        }
        ulong const candidate = pool.pop_nearest();
        if (candidate == dense_metal_sentinel_k) {
            if (pool.full() || restarts >= dense_metal_restarts_cap_k || hops >= hops_cap) break;
            seeding = true;
            ++restarts, ++walk.restarts;
            continue;
        }
        if (pool.beyond_radius(candidate) || hops >= hops_cap) break;
        ++hops, ++walk.hops;
        uint const candidate_id = dense_metal_id_of(candidate);
        device uint* const row = walk.neighbors_for(candidate_id);
        uint first_value = dense_metal_unknown_k;
        if (walk.lane == 0) {
            first_value = dense_metal_load_acquire(row);
            for (uint attempt = 0; attempt != dense_metal_peek_retries_k && first_value == dense_metal_unknown_k;
                 ++attempt)
                first_value = dense_metal_load_acquire(row);
        }
        first_value = simd_shuffle(first_value, ushort(0));
        // Every lane reads the candidate's payload, so every lane fences behind the leader's acquire.
        dense_metal_fence();
        if (first_value == dense_metal_unknown_k) {
            ++walk.locked_drops;
            continue;
        }
        for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += dense_metal_lanes_k) {
            uint const slot = slot_base + walk.lane;
            uint const neighbor_id = slot == 0                  ? first_value
                                     : slot < walk.connectivity ? dense_metal_load_acquire(row + slot)
                                                                : dense_metal_unknown_k;
            bool const usable = neighbor_id < first_invisible && neighbor_id != candidate_id;
            uint const count = walk.stage_fresh(neighbor_id, usable);
            if (count) dense_metal_score_into(walk, pool, query_norms, count, fold);
        }
    }
}

/** Greedy diversity selection over the pool's ascending prefix into @c selected_ids, backfilled with the nearest
 *  rejected candidates as the CUDA selection does; answers the kept count. */
template <uint slots_>
inline uint dense_metal_select_diverse(thread dense_metal_walk_t& walk, thread dense_metal_pool<slots_>& pool) {
    uint candidates = 0;
    for (uint base = 0; base < dense_metal_cache_rows_k; base += dense_metal_lanes_k) {
        ulong const entry = pool.rank_entry(base + walk.lane);
        uint const live = dense_metal_ballot(entry != dense_metal_sentinel_k);
        if (entry != dense_metal_sentinel_k) walk.ids[base + walk.lane] = dense_metal_id_of(entry);
        candidates += popcount(live);
        if (live != 0xFFFFFFFFu) break;
    }
    dense_metal_sync();
    for (uint word = walk.lane; word < candidates * walk.row_fragments; word += dense_metal_lanes_k) {
        uint const row = walk.row_of_word(word), fragment_index = word - row * walk.row_fragments;
        walk.cache_row(row)[fragment_index] = walk.vector_for(walk.ids[row])[fragment_index];
    }
    for (uint row = walk.lane; row < candidates; row += dense_metal_lanes_k)
        walk.cache_norms[row] = walk.norm_for(walk.ids[row]);
    dense_metal_sync();

    uint selected = 0;
    for (uint candidate = 0; candidate < candidates && selected < walk.connectivity; ++candidate) {
        float const to_query = dense_metal_distance_of(pool.rank_entry(candidate));
        bool diverse = true;
        for (uint base = 0; base < selected && diverse; base += dense_metal_lanes_k) {
            bool lane_diverse = true;
            if (base + walk.lane < selected) {
                uint const kept = walk.selected_ids[base + walk.lane];
                dot_t accumulator = dot_t(0);
                for (uint fragment_index = 0; fragment_index != walk.row_fragments; ++fragment_index)
                    accumulator = kit_t::dot(walk.cache_row(candidate)[fragment_index],
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
            dense_metal_sync();
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
        dense_metal_sync();
    }
    dense_metal_sync();
    for (uint slot = walk.lane; slot < selected; slot += dense_metal_lanes_k)
        walk.selected_ids[slot] = walk.ids[walk.selected_ids[slot]];
    dense_metal_sync();
    return selected;
}

/** Merges a pre-sorted batch of @p count nodes into the ascending span @c sorted of @p size, keeping @p capacity. */
inline uint dense_metal_merge_sorted(thread dense_metal_walk_t& walk, ulong value, uint count, uint size,
                                     uint capacity) {
    bool const incoming = walk.lane < count;
    uint incoming_rank = incoming ? walk.lane : 0;
    for (uint index = 0; index != size; ++index)
        incoming_rank += incoming && dense_metal_distance_bits(walk.sorted[index]) <= dense_metal_distance_bits(value);
    ulong moved[2];
    uint moved_to[2];
    for (uint base = 0, batch = 0; base < size; base += dense_metal_lanes_k, ++batch) {
        uint const index = base + walk.lane;
        bool const active = index < size;
        moved[batch] = active ? walk.sorted[index] : dense_metal_sentinel_k;
        uint shift = 0;
        for (uint source = 0; source != count; ++source)
            shift += active && dense_metal_distance_bits(dense_metal_shuffle(value, ushort(source))) <
                                   dense_metal_distance_bits(moved[batch]);
        moved_to[batch] = index + shift;
    }
    dense_metal_sync();
    for (uint base = 0, batch = 0; base < size; base += dense_metal_lanes_k, ++batch)
        if (base + walk.lane < size && moved_to[batch] < capacity) walk.sorted[moved_to[batch]] = moved[batch];
    if (incoming && incoming_rank < capacity) walk.sorted[incoming_rank] = value;
    dense_metal_sync();
    return min(size + count, capacity);
}

/** Back-links @p new_node_id into each selected neighbor through the bounded slot-0 exchange: an append when a slot is
 *  free, else a diversity re-selection over the neighbors and the newcomer, ascending by distance to the target. */
inline void dense_metal_back_link(thread dense_metal_walk_t& walk, uint new_node_id, norms_t new_norms, uint selected) {
    uint const newcomer_row = dense_metal_cache_rows_k - 1, target_row = dense_metal_cache_rows_k - 2;
    uint const connectivity = min(walk.connectivity, dense_metal_cache_rows_k - 2);
    for (uint selected_index = 0; selected_index < selected; ++selected_index) {
        uint const target = walk.selected_ids[selected_index];
        device uint* const row = walk.neighbors_for(target);
        uint first_value = dense_metal_unknown_k;
        if (walk.lane == 0)
            for (uint attempt = 0; attempt != dense_metal_lock_budget_k && first_value == dense_metal_unknown_k;
                 ++attempt)
                first_value = dense_metal_exchange_acquire(row, dense_metal_unknown_k);
        first_value = simd_shuffle(first_value, ushort(0));
        dense_metal_fence();
        ++walk.lock_attempts;
        if (first_value == dense_metal_unknown_k) {
            ++walk.lock_skips;
            continue;
        }
        if (first_value == target) {
            if (walk.lane == 0) dense_metal_store_release(row, new_node_id);
            continue;
        }
        uint empty_slot = connectivity;
        for (uint base = 0; base < connectivity; base += dense_metal_lanes_k) {
            uint const slot = base + walk.lane;
            uint const slot_value = slot == 0             ? first_value
                                    : slot < connectivity ? dense_metal_load_acquire(row + slot)
                                                          : dense_metal_unknown_k;
            if (slot < connectivity) walk.ids[slot] = slot_value;
            uint const empty_mask = dense_metal_ballot(slot < connectivity && slot_value == dense_metal_unknown_k);
            if (empty_mask && empty_slot == connectivity) empty_slot = base + ctz(empty_mask);
        }
        dense_metal_sync();
        dense_metal_fence();
        if (empty_slot != connectivity) {
            if (walk.lane == 0) dense_metal_store_relaxed(row + empty_slot, new_node_id);
            dense_metal_fence();
            if (walk.lane == 0) dense_metal_store_release(row, first_value);
            continue;
        }

        ++walk.reselections;
        for (uint word = walk.lane; word < walk.row_fragments; word += dense_metal_lanes_k) {
            walk.cache_row(target_row)[word] = walk.vector_for(target)[word];
            walk.cache_row(newcomer_row)[word] = walk.vector_for(new_node_id)[word];
        }
        dense_metal_sync();
        norms_t const target_norms = walk.norm_for(target);
        dot_t self_dot = dot_t(0);
        for (uint word = walk.lane; word < walk.row_fragments; word += dense_metal_lanes_k)
            self_dot = kit_t::dot(walk.cache_row(newcomer_row)[word], walk.cache_row(target_row)[word], self_dot);
        self_dot = simd_sum(self_dot);
        float const newcomer_distance = kit_t::distance(walk.metric, self_dot, target_norms, new_norms);
        walk.scored_count += connectivity + 1;

        uint sorted_size = 0;
        for (uint base = 0; base < connectivity; base += dense_metal_lanes_k) {
            uint const count = min(connectivity - base, dense_metal_lanes_k);
            ulong const scored = walk.score_batch(walk.cache_row(target_row), target_norms, walk.ids + base, count);
            ulong const sorted = dense_metal_sort_lanes(
                walk.lane < count ? dense_metal_make(dense_metal_distance_of(scored), base + walk.lane)
                                  : dense_metal_sentinel_k,
                walk.lane);
            sorted_size = dense_metal_merge_sorted(walk, sorted, count, sorted_size, connectivity);
        }
        for (uint word = walk.lane; word < connectivity * walk.row_fragments; word += dense_metal_lanes_k) {
            uint const cache_row = walk.row_of_word(word), fragment_index = word - cache_row * walk.row_fragments;
            walk.cache_row(cache_row)[fragment_index] = walk.vector_for(walk.ids[cache_row])[fragment_index];
        }
        for (uint cache_row = walk.lane; cache_row < connectivity; cache_row += dense_metal_lanes_k)
            walk.cache_norms[cache_row] = walk.norm_for(walk.ids[cache_row]);
        if (walk.lane == 0) walk.cache_norms[newcomer_row] = new_norms;
        dense_metal_sync();

        // The merge walk over the sorted neighbors and the newcomer, as cache rows, keeping the diverse picks
        // alone as the CPU's reverse refine does: a backfilled row stays full, so on a dot graph every later
        // newcomer competes with the hubs and the small-norm nodes lose their last in-links.
        uint refined = 0, existing = 0;
        bool newcomer_pending = true;
        while (refined != connectivity) {
            ulong const next_link = existing < sorted_size ? walk.sorted[existing] : dense_metal_sentinel_k;
            bool const take_newcomer = newcomer_pending && newcomer_distance < dense_metal_distance_of(next_link);
            if (!take_newcomer && existing >= sorted_size) break;
            float const candidate_distance = take_newcomer ? newcomer_distance : dense_metal_distance_of(next_link);
            uint const position = take_newcomer ? newcomer_row : dense_metal_id_of(next_link);
            if (take_newcomer) newcomer_pending = false;
            else ++existing;
            bool diverse = true;
            for (uint base = 0; base < refined && diverse; base += dense_metal_lanes_k) {
                bool lane_diverse = true;
                if (base + walk.lane < refined) {
                    uint const kept = walk.kept_positions[base + walk.lane];
                    dot_t pair = dot_t(0);
                    for (uint fragment_index = 0; fragment_index != walk.row_fragments; ++fragment_index)
                        pair = kit_t::dot(walk.cache_row(position)[fragment_index],
                                          walk.cache_row(kept)[fragment_index], pair);
                    lane_diverse = kit_t::distance(walk.metric, pair, walk.cache_norms[position],
                                                   walk.cache_norms[kept]) >= candidate_distance;
                    ++walk.scored_count;
                }
                diverse = simd_all(lane_diverse);
            }
            if (!diverse) continue;
            if (walk.lane == 0) walk.kept_positions[refined] = position;
            ++refined;
            dense_metal_sync();
        }
        dense_metal_sync();
        for (uint slot = walk.lane; slot < connectivity; slot += dense_metal_lanes_k) {
            if (slot == 0) continue;
            uint linked = dense_metal_unknown_k;
            if (slot < refined) {
                uint const kept = walk.kept_positions[slot];
                linked = kept == newcomer_row ? new_node_id : walk.ids[kept];
            }
            dense_metal_store_relaxed(row + slot, linked);
        }
        dense_metal_fence();
        if (walk.lane == 0) {
            uint const head = walk.kept_positions[0];
            dense_metal_store_release(row, head == newcomer_row ? new_node_id : walk.ids[head]);
        }
    }
}

/** Folds the walk's telemetry into the device counters, from its leader. */
inline void dense_metal_flush(thread dense_metal_walk_t& walk) {
    if (walk.lane != 0) return;
    auto const add = [&](uint counter, uint value) {
        if (value) atomic_fetch_add_explicit(walk.counters + counter, value, memory_order_relaxed);
    };
    add(dense_metal_hops_k, walk.hops);
    add(dense_metal_scored_k, walk.scored_count);
    add(dense_metal_restarts_k, walk.restarts);
    add(dense_metal_lock_attempts_k, walk.lock_attempts);
    add(dense_metal_lock_skips_k, walk.lock_skips);
    add(dense_metal_reselections_k, walk.reselections);
    add(dense_metal_locked_drops_k, walk.locked_drops);
    add(dense_metal_saturations_k, walk.saturations);
}

/** Opens the walk of the simdgroup at @p slice over its scratch, with fresh telemetry. */
inline dense_metal_walk_t dense_metal_open(constant dense_metal_arguments_t& arguments, device uint4* vectors,
                                           device norms_t* norms, device uint* neighbors, device uchar* arena,
                                           device atomic_uint* counters, uint slice, ushort lane) {
    dense_metal_walk_t walk;
    device uchar* cursor = arena + ulong(slice) * arguments.slice_bytes;
    auto const carve = [&](uint bytes) {
        device uchar* const carved = cursor;
        cursor += bytes;
        return carved;
    };
    uint const fragment_bytes = arguments.row_fragments * 16;
    walk.vectors = vectors, walk.norms = norms, walk.neighbors = neighbors, walk.counters = counters;
    walk.query = (device uint4*)carve(fragment_bytes);
    walk.cache = (device uint4*)carve(fragment_bytes * arguments.cache_rows);
    walk.visited = (device ulong*)carve(arguments.visited_slots * 8);
    walk.scored = (device ulong*)carve(dense_metal_lanes_k * 8);
    walk.sorted = (device ulong*)carve(dense_metal_cache_rows_k * 8);
    walk.cache_norms = (device norms_t*)carve(dense_metal_cache_rows_k * 8);
    walk.ids = (device uint*)carve(dense_metal_cache_rows_k * 4);
    walk.dots = (device dot_t*)carve(dense_metal_lanes_k * 4);
    walk.selected_ids = (device uint*)carve(dense_metal_cache_rows_k * 4);
    walk.kept_positions = (device uint*)carve(dense_metal_cache_rows_k * 4);
    walk.row_fragments = arguments.row_fragments, walk.lanes_per_row = arguments.lanes_per_row;
    walk.steps_per_row = arguments.steps_per_row, walk.connectivity = arguments.connectivity;
    walk.metric = arguments.metric, walk.row_fragments_magic = arguments.row_fragments_magic;
    walk.has_norms = arguments.has_norms != 0, walk.lane = lane;
    walk.hash_shift = arguments.hash_shift, walk.epoch = 0, walk.saturations = 0, walk.random = 0;
    walk.hops = 0, walk.scored_count = 0, walk.restarts = 0, walk.lock_attempts = 0, walk.lock_skips = 0;
    walk.reselections = 0, walk.locked_drops = 0;
    return walk;
}

/** Stages @p row into the walk's query span and answers its norms. */
inline norms_t dense_metal_stage_query(thread dense_metal_walk_t& walk, device uint4 const* row) {
    norms_t moments = norms_t(0);
    for (uint fragment_index = walk.lane; fragment_index < walk.row_fragments; fragment_index += dense_metal_lanes_k) {
        uint4 const value = row[fragment_index];
        walk.query[fragment_index] = value;
        moments = kit_t::moments(value, moments);
    }
    dense_metal_sync();
    return kit_t::norms(simd_sum(moments));
}

#pragma endregion Walk

#pragma region Kernels

/** One simdgroup inserts one node at a time, assigning itself IDs from the batch counter. */
template <uint slots_>
inline void dense_metal_insert(constant dense_metal_arguments_t& arguments, device uint4* vectors,
                               device norms_t* norms, device uint* neighbors, device uint4 const* staging,
                               device uchar* arena, device atomic_uint* counters, uint slice, ushort lane) {
    dense_metal_walk_t walk = dense_metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    while (true) {
        uint new_node_id = 0;
        if (lane == 0)
            new_node_id = arguments.first_id +
                          atomic_fetch_add_explicit(counters + dense_metal_assigned_k, 1u, memory_order_relaxed);
        new_node_id = simd_shuffle(new_node_id, ushort(0));
        if (new_node_id >= arguments.end_id) break;
        while (true) {
            uint const published = atomic_load_explicit(counters + dense_metal_published_k, memory_order_relaxed);
            if (new_node_id < published + max(published, dense_metal_admission_floor_k)) break;
        }
        walk.epoch = arguments.epoch_base + (new_node_id - arguments.first_id) + 1;
        walk.random = 0xA5A5A5A5ul ^ (ulong(new_node_id) * 33 + lane);

        // Land the payload, derive the norms, and park the query in one pass.
        device uint4 const* const staged = staging + ulong(new_node_id - arguments.first_id) * walk.row_fragments;
        device uint4* const owned = walk.vector_for(new_node_id);
        for (uint fragment_index = lane; fragment_index < walk.row_fragments; fragment_index += dense_metal_lanes_k)
            owned[fragment_index] = staged[fragment_index];
        norms_t const new_norms = dense_metal_stage_query(walk, staged);
        if (lane == 0 && walk.has_norms) norms[new_node_id] = new_norms;
        dense_metal_sync();

        dense_metal_pool<slots_> pool;
        pool.clear();
        dense_metal_fold_t fold {};
        if (new_node_id != 0) dense_metal_search_ahead(walk, pool, new_norms, new_node_id, arguments.end_id, fold);
        uint const selected = dense_metal_select_diverse(walk, pool);
        device uint* const own = walk.neighbors_for(new_node_id);
        for (uint slot = lane; slot < selected; slot += dense_metal_lanes_k)
            if (slot > 0) dense_metal_store_relaxed(own + slot, walk.selected_ids[slot]);
        // Every lane wrote the payload or the slots, so every lane fences ahead of the leader's release.
        dense_metal_fence();
        if (lane == 0) {
            dense_metal_store_release(own, selected ? walk.selected_ids[0] : new_node_id);
            atomic_fetch_add_explicit(counters + dense_metal_published_k, 1u, memory_order_relaxed);
        }
        dense_metal_back_link(walk, new_node_id, new_norms, selected);
    }
    dense_metal_flush(walk);
}

/** One simdgroup re-selects one node at a time into the shadow plane, read-only on the live one. */
template <uint slots_>
inline void dense_metal_refine(constant dense_metal_arguments_t& arguments, device uint4* vectors,
                               device norms_t* norms, device uint* neighbors, device uint const* node_ids,
                               device uchar* arena, device atomic_uint* counters, device uint* shadow, uint slice,
                               ushort lane) {
    dense_metal_walk_t walk = dense_metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    while (true) {
        uint offset = 0;
        if (lane == 0) offset = atomic_fetch_add_explicit(counters + dense_metal_assigned_k, 1u, memory_order_relaxed);
        offset = simd_shuffle(offset, ushort(0));
        if (arguments.first_id + offset >= arguments.end_id) break;
        uint const node_id = arguments.use_node_ids ? node_ids[offset] : arguments.first_id + offset;
        walk.epoch = arguments.epoch_base + offset + 1;
        walk.random = 0xA5A5A5A5ul ^ (ulong(node_id) * 33 + lane);
        dense_metal_stage_query(walk, walk.vector_for(node_id));
        norms_t const query_norms = walk.norm_for(node_id);

        dense_metal_pool<slots_> pool;
        pool.clear();
        dense_metal_fold_t fold {};
        walk.test_and_set(node_id, true);
        device uint* const own = walk.neighbors_for(node_id);
        uint const first_value = simd_shuffle(lane == 0 ? dense_metal_load_acquire(own) : 0u, ushort(0));
        if (first_value != dense_metal_unknown_k && first_value != node_id)
            for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += dense_metal_lanes_k) {
                uint const slot = slot_base + lane;
                uint const seed = slot == 0                  ? first_value
                                  : slot < walk.connectivity ? dense_metal_load_acquire(own + slot)
                                                             : dense_metal_unknown_k;
                uint const count = walk.stage_fresh(seed, seed != dense_metal_unknown_k && seed != node_id);
                if (count) dense_metal_score_into(walk, pool, query_norms, count, fold);
            }
        dense_metal_search_ahead(walk, pool, query_norms, arguments.sampling_bound, arguments.sampling_bound, fold);
        uint const selected = dense_metal_select_diverse(walk, pool);
        device uint* const shadow_row = shadow + ulong(node_id) * walk.connectivity;
        for (uint slot = lane; slot < walk.connectivity; slot += dense_metal_lanes_k)
            shadow_row[slot] = slot < selected ? walk.selected_ids[slot] : slot == 0 ? node_id : dense_metal_unknown_k;
        dense_metal_sync();
    }
    dense_metal_flush(walk);
}

/** One simdgroup walks one query at a time, landing its nearest IDs and distances, or under @p fold.keeps the pairs
 *  it scored and their count, ranked under indexer heads when the fold names any. */
template <uint slots_>
inline void dense_metal_search(constant dense_metal_arguments_t& arguments, device uint4* vectors,
                               device norms_t* norms, device uint* neighbors, device uint4 const* queries,
                               device uchar* arena, device atomic_uint* counters, device uint* result_ids,
                               device float* result_distances, device dot_t* products, device uint* scored_counts,
                               device uint4 const* heads, device float const* weights, bool keeps, uint slice,
                               uint slices, ushort lane) {
    dense_metal_walk_t walk = dense_metal_open(arguments, vectors, norms, neighbors, arena, counters, slice, lane);
    for (uint query = slice; query < arguments.queries_count; query += slices) {
        walk.epoch = arguments.epoch_base + query + 1;
        walk.random = 0xA5A5A5A5ul ^ (ulong(query) * 33 + lane);
        norms_t const query_norms = dense_metal_stage_query(walk, queries + ulong(query) * walk.row_fragments);
        dense_metal_pool<slots_> pool;
        pool.clear();
        dense_metal_fold_t fold {};
        if (keeps) {
            fold.keeps = true;
            fold.ids = result_ids + ulong(query) * arguments.count;
            fold.products = products + ulong(query) * arguments.count;
            fold.wanted = arguments.count;
        }
        if (arguments.heads_count) {
            fold.heads = heads + ulong(query) * arguments.heads_count * walk.row_fragments;
            fold.weights = weights + ulong(query) * arguments.heads_count;
            fold.heads_count = arguments.heads_count;
        }
        dense_metal_search_ahead(walk, pool, query_norms, arguments.sampling_bound, arguments.sampling_bound, fold);
        if (keeps) {
            for (uint slot = fold.written + lane; slot < arguments.count; slot += dense_metal_lanes_k)
                fold.ids[slot] = dense_metal_unknown_k;
            if (lane == 0) scored_counts[query] = fold.scored;
            continue;
        }
        for (uint rank_base = 0; rank_base < arguments.count; rank_base += dense_metal_lanes_k) {
            uint const rank = rank_base + lane;
            ulong const entry = pool.rank_entry(rank);
            if (rank >= arguments.count) continue;
            ulong const slot = ulong(query) * arguments.count + rank;
            result_ids[slot] = dense_metal_id_of(entry);
            result_distances[slot] = dense_metal_convert(walk.metric, dense_metal_distance_of(entry));
        }
    }
    dense_metal_flush(walk);
}

/** Rebuilds the norms of rows from @c first_id to @c end_id, one simdgroup per row. */
[[host_name("dense_metal_derive_norms_kernel")]] kernel void dense_metal_derive_norms_kernel(
    constant dense_metal_arguments_t& arguments [[buffer(0)]], device uint4* vectors [[buffer(1)]],
    device norms_t* norms [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint const slices = groups * dense_metal_simdgroups_k;
    for (uint row = arguments.first_id + group * dense_metal_simdgroups_k + simdgroup; row < arguments.end_id;
         row += slices) {
        norms_t moments = norms_t(0);
        for (uint fragment_index = lane; fragment_index < arguments.row_fragments;
             fragment_index += dense_metal_lanes_k)
            moments = kit_t::moments(vectors[ulong(row) * arguments.row_fragments + fragment_index], moments);
        moments = simd_sum(moments);
        if (lane == 0) norms[row] = kit_t::norms(moments);
    }
}

/*  The entry points, one instantiation per pool width the host's ladder names: 2, 4, 8 and 16 slots of
 *  32 lanes, each under the name the host looks it up by. */

template <uint slots_>
kernel void dense_metal_insert_kernel(constant dense_metal_arguments_t& arguments [[buffer(0)]],
                                      device uint4* vectors [[buffer(1)]], device norms_t* norms [[buffer(2)]],
                                      device uint* neighbors [[buffer(3)]], device uint4 const* staging [[buffer(4)]],
                                      device uchar* arena [[buffer(5)]], device atomic_uint* counters [[buffer(6)]],
                                      uint group [[threadgroup_position_in_grid]],
                                      ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
    dense_metal_insert<slots_>(arguments, vectors, norms, neighbors, staging, arena, counters,
                               group * dense_metal_simdgroups_k + simdgroup, lane);
}

template [[host_name("dense_metal_insert_kernel_2")]] kernel void dense_metal_insert_kernel<2>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, uint, ushort, ushort);
template [[host_name("dense_metal_insert_kernel_4")]] kernel void dense_metal_insert_kernel<4>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, uint, ushort, ushort);
template [[host_name("dense_metal_insert_kernel_8")]] kernel void dense_metal_insert_kernel<8>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, uint, ushort, ushort);
template [[host_name("dense_metal_insert_kernel_16")]] kernel void dense_metal_insert_kernel<16>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, uint, ushort, ushort);

template <uint slots_>
kernel void dense_metal_refine_kernel(constant dense_metal_arguments_t& arguments [[buffer(0)]],
                                      device uint4* vectors [[buffer(1)]], device norms_t* norms [[buffer(2)]],
                                      device uint* neighbors [[buffer(3)]], device uint const* node_ids [[buffer(4)]],
                                      device uchar* arena [[buffer(5)]], device atomic_uint* counters [[buffer(6)]],
                                      device uint* shadow [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                                      ushort simdgroup [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
    dense_metal_refine<slots_>(arguments, vectors, norms, neighbors, node_ids, arena, counters, shadow,
                               group * dense_metal_simdgroups_k + simdgroup, lane);
}

template [[host_name("dense_metal_refine_kernel_2")]] kernel void dense_metal_refine_kernel<2>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, uint, ushort, ushort);
template [[host_name("dense_metal_refine_kernel_4")]] kernel void dense_metal_refine_kernel<4>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, uint, ushort, ushort);
template [[host_name("dense_metal_refine_kernel_8")]] kernel void dense_metal_refine_kernel<8>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, uint, ushort, ushort);
template [[host_name("dense_metal_refine_kernel_16")]] kernel void dense_metal_refine_kernel<16>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint const*, device uchar*,
    device atomic_uint*, device uint*, uint, ushort, ushort);

/*  Searches and graph products share one entry point per width: buffer 13 says which the launch lands. */

template <uint slots_>
kernel void dense_metal_search_kernel(
    constant dense_metal_arguments_t& arguments [[buffer(0)]], device uint4* vectors [[buffer(1)]],
    device norms_t* norms [[buffer(2)]], device uint* neighbors [[buffer(3)]],
    device uint4 const* queries [[buffer(4)]], device uchar* arena [[buffer(5)]],
    device atomic_uint* counters [[buffer(6)]], device uint* ids [[buffer(7)]], device float* distances [[buffer(8)]],
    device dot_t* products [[buffer(9)]], device uint* scored [[buffer(10)]], device uint4 const* heads [[buffer(11)]],
    device float const* weights [[buffer(12)]], constant uint& keeps [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    dense_metal_search<slots_>(arguments, vectors, norms, neighbors, queries, arena, counters, ids, distances,
                               products, scored, heads, weights, keeps != 0,
                               group * dense_metal_simdgroups_k + simdgroup, groups * dense_metal_simdgroups_k, lane);
}

template [[host_name("dense_metal_search_kernel_2")]] kernel void dense_metal_search_kernel<2>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, ushort, ushort);
template [[host_name("dense_metal_search_kernel_4")]] kernel void dense_metal_search_kernel<4>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, ushort, ushort);
template [[host_name("dense_metal_search_kernel_8")]] kernel void dense_metal_search_kernel<8>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, ushort, ushort);
template [[host_name("dense_metal_search_kernel_16")]] kernel void dense_metal_search_kernel<16>(
    constant dense_metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device float*, device dot_t*, device uint*, device uint4 const*,
    device float const*, constant uint&, uint, uint, ushort, ushort);

#pragma endregion Kernels

} // namespace usearch
} // namespace unum
