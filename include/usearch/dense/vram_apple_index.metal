/**
 *  @file include/usearch/dense/vram_apple_index.metal
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Metal walk, insert, search and round kernels of the dense Apple index - the MSL port
 *      of @c vram_index.cuh and the kernels of @c vram_nvidia_index.cuh.
 *
 *  One 32-wide simdgroup walks one query or inserts one node, with the same packed scored nodes,
 *  register candidates, exact visited sets, slot-0 lock words and admission waves as the CUDA design, over
 *  @ref metal_warp_t, whose atomics and value matching Metal spells its own way. Every
 *  per-simdgroup scratch span, the query, the candidate cache, the visited set and the staged
 *  batch, lives in one device slice rather than threadgroup memory, which caps at 32 KB and would
 *  bound the dimensions. The host prepends @c vram_apple_shared.hpp, its prelude and
 *  @c vram_apple_metric.metal before this file, which takes its kit from @c quads_ladder, all after
 *  the contexts' source, whose @c vram_queue.metal holds the walk's verbs and queue.
 */

namespace unum::usearch::dense {

using kit_t = typename quads_ladder<metal_dtype_k, metal_capability_k>::kit_t;
using warp_t = metal_warp_t;
using dot_t = typename kit_t::dot_t;
using norms_t = typename kit_t::norms_t;
using norms_accumulator_t = typename kit_t::norms_accumulator_t;

#pragma region Walk

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
    device ulong* tail;
    device ulong* scored;
    device uint* ids;
    device dot_t* dots;
    device ulong* scored_entries;
    device uint* selected_ids;
    device uint* candidate_ids;
    device uint* kept_positions;

    /** Each cached candidate's distance to the query, staged once so the selection reads memory, not the candidates. */
    device float* candidate_distances;
    device norms_t* candidate_norms;

    uint row_fragments, lanes_per_row, steps_per_row, connectivity, metric, row_fragments_magic;
    uint tail_entries, expansion, selection_nodes, walk_width;
    bool has_norms;
    uint lane;

    uint hash_shift, epoch, saturations;
    ulong random;
    uint hops, scored_count, restarts, lock_attempts, lock_skips, reselections, locked_drops, tail_compactions;
    uint tail_dropped;

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

    /** Scores @p count rows of @p batch_ids against @p query, a sub-group of @c lanes_per_row lanes
     *  per row, writing each row's node in @c scored and its dot in @c dots; answers this lane's
     *  node, or the sentinel past @p count. A lane whose row lies past the batch re-reads the
     *  batch's first row instead of taking a guard, and each lane finalizes its own row, so the
     *  metric's epilogue runs once per lane. */
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
};

/** What a walk does with every scored batch beside folding it into the candidates. */
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

/** Scores one gathered batch, rescores it under indexer heads when the fold names any, folds it into the candidates,
 * and hands it to the fold. */
template <typename frontier_>
inline void metal_score_into_frontier(thread metal_walk_t& walk, thread frontier_& frontier, norms_t query_norms,
                                      uint count, thread metal_fold_t& fold) {
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
    frontier.push_batch(staged, walk.lane);
    fold.absorb(walk, count);
}

/** Continues a greedy best-first walk over whatever @p frontier holds, seeding from landmarks in @b [first,bound) and
 *  never following a neighbor at or past @p first_invisible; pre-staged entries compete with them and must already
 *  be visited, and each call walks to convergence. Read-only on the graph, a transiently locked candidate held
 *  through one other expansion and dropped if still locked then, as the CUDA and HIP walks do. */
template <typename frontier_>
inline void metal_search_ahead(thread metal_walk_t& walk, thread frontier_& frontier, norms_t query_norms, uint first,
                               uint bound, uint first_invisible, thread metal_fold_t& fold) {
    // The one candidate held for its second look, and the hop it was held at; drained candidates move that hop out of
    // reach, so the next iteration looks at once.
    ulong held = metal_sentinel_k;
    uint held_at_hop = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            uint const count = metal_sample_landmarks(walk, first, bound, frontier);
            if (!count) break; // Every published row below the bound is already visited.
            metal_score_into_frontier(walk, frontier, query_norms, count, fold);
            seeding = false;
            continue;
        }
        bool const revisiting = held != metal_sentinel_k && walk.hops != held_at_hop;
        ulong const candidate = revisiting ? held : frontier.pop_nearest(walk.lane);
        if (revisiting) held = metal_sentinel_k;
        if (candidate == metal_sentinel_k) {
            if (held != metal_sentinel_k) {
                held_at_hop = ~0u;
                continue;
            }
            // A drained frontier before an exhausted walk re-seeds, until no published row of the range is left
            // unvisited.
            if (frontier.exhausted()) break;
            seeding = true;
            ++walk.restarts;
            continue;
        }
        if (frontier.beyond_radius(candidate)) break;
        ++walk.hops;
        uint const candidate_id = metal_id_of(candidate);
        device uint* const row = walk.neighbors_for(candidate_id);
        uint const first_value = warp_t::shuffle(walk.lane == 0 ? warp_t::load_acquire(row) : metal_unknown_k, 0);
        // Every lane reads the candidate's payload, so every lane fences behind the leader's acquire.
        warp_t::fence();
        if (first_value == metal_unknown_k) {
            // Held through one other expansion instead of waited on; a second refusal drops it.
            if (revisiting || held != metal_sentinel_k) ++walk.locked_drops;
            else held = candidate, held_at_hop = walk.hops;
            continue;
        }
        uint pending = 0;
        auto score = [&](uint count) { metal_score_into_frontier(walk, frontier, query_norms, count, fold); };
        metal_expand_round(walk, frontier, candidate, first_value, first_invisible, held, held_at_hop, pending, score);
        // A dense kit pays per score call, not per idle lane, so the round scores what it staged.
        if (pending) metal_score_into_frontier(walk, frontier, query_norms, pending, fold);
    }
    walk.locked_drops += held != metal_sentinel_k;
}

/** Greedy diversity selection over all of the walk's @c selection_nodes candidates, ascending, into @c selected_ids,
 *  backfilled with the nearest rejected candidates as the CUDA selection does; answers the kept count. */
template <typename frontier_>
inline uint metal_select_diverse(thread metal_walk_t& walk, thread frontier_& frontier) {
    uint candidates = 0;
    auto stage = [&](uint rank, ulong entry) {
        bool const live = rank < walk.selection_nodes && entry != metal_sentinel_k;
        if (live) {
            walk.candidate_ids[rank] = metal_id_of(entry);
            walk.candidate_distances[rank] = metal_distance_of(entry);
        }
        candidates += popcount(warp_t::ballot(live));
    };
    frontier.for_each_rank(walk.selection_nodes, walk.lane, stage);
    warp_t::sync();
    for (uint word = walk.lane; word < candidates * walk.row_fragments; word += warp_t::width_k) {
        uint const row = walk.row_of_word(word), fragment_index = word - row * walk.row_fragments;
        walk.cache_row(row)[fragment_index] = walk.vector_for(walk.candidate_ids[row])[fragment_index];
    }
    for (uint row = walk.lane; row < candidates; row += warp_t::width_k)
        walk.candidate_norms[row] = walk.norm_for(walk.candidate_ids[row]);
    warp_t::sync();

    uint selected = 0;
    for (uint candidate = 0; candidate < candidates && selected < walk.connectivity; ++candidate) {
        float const to_query = walk.candidate_distances[candidate];
        bool diverse = true;
        for (uint base = 0; base < selected && diverse; base += warp_t::width_k) {
            bool lane_diverse = true;
            if (base + walk.lane < selected) {
                uint const kept = walk.selected_ids[base + walk.lane];
                dot_t accumulator = dot_t(0);
                for (uint fragment_index = 0; fragment_index != walk.row_fragments; ++fragment_index)
                    accumulator = kit_t::accumulate_dot(walk.cache_row(candidate)[fragment_index],
                                                        walk.cache_row(kept)[fragment_index], accumulator);
                lane_diverse = kit_t::distance(walk.metric, accumulator, walk.candidate_norms[candidate],
                                               walk.candidate_norms[kept]) >= to_query;
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
        walk.selected_ids[slot] = walk.candidate_ids[walk.selected_ids[slot]];
    warp_t::sync();
    return selected;
}

/** Re-selects one node's neighbors from @c candidate_ids, its @p candidates_count distinct staged
 *  candidates, the diversity heuristic over them ascending by distance to the node, and writes them
 *  in its live row slot by slot, slot zero released last, so a concurrent walk reads every slot
 *  whole. The node's own row is the cache's @p query_row, its last. It keeps the diverse picks
 *  alone, as the CPU's reverse refine does: a backfilled row stays full, so on a dot graph every
 *  later appended node competes with the hubs and the small-norm nodes lose their last in-links,
 *  one by one. */
inline void metal_reselect_staged(thread metal_walk_t& walk, uint node_id, uint candidates_count, uint query_row) {
    device uint* const candidate_ids = walk.candidate_ids;
    device uint* const kept_positions = walk.kept_positions;
    norms_t const query_state = walk.norm_for(node_id);

    // Cache the candidate vectors & norms for the pairwise diversity checks, and the node's own row.
    for (uint word = walk.lane; word < candidates_count * walk.row_fragments; word += warp_t::width_k) {
        uint const cache_row = walk.row_of_word(word), fragment_index = word - cache_row * walk.row_fragments;
        walk.cache_row(cache_row)[fragment_index] = walk.vector_for(candidate_ids[cache_row])[fragment_index];
    }
    for (uint fragment_index = walk.lane; fragment_index < walk.row_fragments; fragment_index += warp_t::width_k)
        walk.cache_row(query_row)[fragment_index] = walk.vector_for(node_id)[fragment_index];
    for (uint cache_row = walk.lane; cache_row < candidates_count; cache_row += warp_t::width_k)
        walk.candidate_norms[cache_row] = walk.norm_for(candidate_ids[cache_row]);
    warp_t::sync();
    device uint4 const* const node_span = walk.cache_row(query_row);

    // Score every candidate against the node & sort ascending; IDs are cache-row indices.
    walk.scored_count += candidates_count;
    metal_sorted_buffer sorted_candidates {walk.scored_entries, 0, query_row};
    for (uint base = 0; base < candidates_count; base += warp_t::width_k) {
        uint const batch_count = min(candidates_count - base, warp_t::width_k);
        ulong const scored = walk.score_batch(node_span, query_state, candidate_ids + base, batch_count);
        ulong const sorted_batch = metal_sort_lanes(
            walk.lane < batch_count ? metal_make(metal_distance_of(scored), base + walk.lane) : metal_sentinel_k,
            walk.lane);
        sorted_candidates.insert_batch(sorted_batch, batch_count, walk.lane);
    }

    // The greedy diversity walk, ascending by distance to the node.
    uint refined_count = 0;
    for (uint merge_index = 0; merge_index != sorted_candidates.size && refined_count != walk.connectivity;
         ++merge_index) {
        ulong const entry = sorted_candidates.at(merge_index);
        float const to_query = metal_distance_of(entry);
        uint const candidate_position = metal_id_of(entry);
        bool diverse = true;
        for (uint base = 0; base < refined_count && diverse; base += warp_t::width_k) { // Simdgroup-uniform bounds.
            bool lane_diverse = true;
            if (base + walk.lane < refined_count) {
                uint const kept_position = kept_positions[base + walk.lane];
                dot_t pair_accumulator = dot_t(0);
                for (uint fragment_index = 0; fragment_index != walk.row_fragments; ++fragment_index)
                    pair_accumulator = kit_t::accumulate_dot(walk.cache_row(candidate_position)[fragment_index],
                                                             walk.cache_row(kept_position)[fragment_index],
                                                             pair_accumulator);
                float const between = kit_t::distance(walk.metric, pair_accumulator,
                                                      walk.candidate_norms[candidate_position],
                                                      walk.candidate_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++walk.scored_count;
            }
            diverse = simd_all(lane_diverse);
        }
        if (diverse) {
            if (walk.lane == 0) kept_positions[refined_count] = candidate_position;
            ++refined_count;
            warp_t::sync();
        }
    }

    // Write the final neighbors in the live row - the insertion's slot convention.
    warp_t::sync();
    device uint* const live_row = walk.neighbors_for(node_id);
    for (uint slot = walk.lane; slot < walk.connectivity; slot += warp_t::width_k)
        if (slot)
            warp_t::store_relaxed(live_row + slot,
                                  slot < refined_count ? candidate_ids[kept_positions[slot]] : metal_unknown_k);
    warp_t::fence();
    if (walk.lane == 0) warp_t::store_release(live_row, candidate_ids[kept_positions[0]]);
}

/** Back-links one node a wave of appended nodes picked, chasing their picks from @p link_heads
 *  through @p link_next in rounds: each round caches the live row and as many appended nodes as fit
 *  before the node's own row at @p query_row, appends them into the free slots when they all fit,
 *  and otherwise re-selects over both through @ref metal_reselect_staged; the next round starts
 *  from the row the last one wrote, so no appended node is dropped. Pick @b p is appended node
 *  <tt>first_id + p / connectivity</tt>. The launch hands each node to one simdgroup, the sole writer of
 *  its own row. */
inline void metal_back_link_one(thread metal_walk_t& walk, uint node_id, device uint const* link_heads,
                                device uint const* link_next, uint first_id, uint query_row) {
    device uint* const candidate_ids = walk.candidate_ids;
    device uint* const live_row = walk.neighbors_for(node_id);

    // A chain is serial, so one lane follows it, and the rounds resume where it stopped; no lane writes the head
    // before the call returns, so every lane reads it.
    uint pick = link_heads[node_id];
    while (pick != metal_unknown_k) {
        // The live neighbors first, a self-loop in slot 0 meaning none, then the appended nodes.
        uint live_count = 0;
        for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += warp_t::width_k) {
            uint const slot = slot_base + walk.lane;
            uint const live_id = slot < walk.connectivity ? warp_t::load_acquire(live_row + slot) : metal_unknown_k;
            bool const usable = live_id != metal_unknown_k && live_id != node_id;
            uint const usable_mask = warp_t::ballot(usable);
            if (usable) candidate_ids[live_count + popcount(usable_mask & warp_t::below(walk.lane))] = live_id;
            live_count += popcount(usable_mask);
            warp_t::sync();
        }
        uint candidates_count = live_count;
        if (walk.lane == 0)
            for (; pick != metal_unknown_k && candidates_count != query_row; pick = link_next[pick])
                candidate_ids[candidates_count++] = first_id + pick / walk.connectivity;
        candidates_count = warp_t::shuffle(candidates_count, 0);
        pick = warp_t::shuffle(pick, 0);
        warp_t::sync();

        // Room for every appended node left: append them past the live neighbors, a published-empty
        // row's self-loop giving way to the first, released last, and nothing to re-judge.
        if (candidates_count <= walk.connectivity) {
            for (uint slot = max(live_count, 1u) + walk.lane; slot < candidates_count; slot += warp_t::width_k)
                warp_t::store_relaxed(live_row + slot, candidate_ids[slot]);
            warp_t::fence();
            if (walk.lane == 0 && !live_count && candidates_count) warp_t::store_release(live_row, candidate_ids[0]);
            return;
        }
        ++walk.reselections;
        metal_reselect_staged(walk, node_id, candidates_count, query_row);
    }
}

/** Opens the walk of the simdgroup at @p slice over its scratch, with fresh telemetry. */
inline metal_walk_t metal_open(constant metal_arguments_t& arguments, device uint4* vectors, device norms_t* norms,
                               device uint* neighbors, device uchar* arena, device atomic_uint* counters, uint slice,
                               uint lane) {
    metal_walk_t walk;
    device uchar* const slice_base = metal_row(arena, slice, arguments.slice_bytes);
    uint offset = 0;
    auto const carve = [&](uint bytes) {
        device uchar* const carved = slice_base + offset;
        offset += bytes;
        return carved;
    };
    uint const fragment_bytes = arguments.row_fragments * metal_fragment_bytes_k;
    uint const selection_nodes = arguments.selection_nodes;
    walk.vectors = vectors, walk.norms = norms, walk.neighbors = neighbors, walk.counters = counters;
    // The visited set, the spill tail and the scratch, as the CUDA slice lays them out; the scratch starts on a
    // fragment, which an odd tail would miss, and the host's rounding of the slice covers the step.
    walk.visited = (device ulong*)carve(arguments.visited_slots * 8);
    walk.tail = (device ulong*)carve(arguments.tail_entries * 8);
    offset = (offset + metal_fragment_bytes_k - 1) / metal_fragment_bytes_k * metal_fragment_bytes_k;
    walk.scored = (device ulong*)carve(metal_scored_span_bytes_k);
    walk.ids = (device uint*)carve(metal_ids_span_bytes_k);
    walk.dots = (device dot_t*)carve(metal_dots_span_bytes_k);
    // The query span while walking, then the candidate cache over it, and past the cache's rows each cached
    // candidate's spans, in the order `vram_selection_scratch` carves them.
    walk.query = walk.cache = (device uint4*)carve(fragment_bytes * selection_nodes);
    walk.scored_entries = (device ulong*)carve(selection_nodes * 8);
    walk.selected_ids = (device uint*)carve(selection_nodes * 4);
    walk.candidate_ids = (device uint*)carve(selection_nodes * 4);
    walk.kept_positions = (device uint*)carve(selection_nodes * 4);
    walk.candidate_distances = (device float*)carve(selection_nodes * 4);
    walk.candidate_norms = (device norms_t*)carve(selection_nodes * 8);
    walk.tail_entries = arguments.tail_entries, walk.expansion = arguments.expansion;
    walk.selection_nodes = selection_nodes, walk.walk_width = 1;
    walk.row_fragments = arguments.row_fragments, walk.lanes_per_row = arguments.lanes_per_row;
    walk.steps_per_row = arguments.steps_per_row, walk.connectivity = arguments.connectivity;
    walk.metric = arguments.metric, walk.row_fragments_magic = arguments.row_fragments_magic;
    walk.has_norms = arguments.has_norms != 0, walk.lane = lane;
    walk.hash_shift = arguments.hash_shift, walk.epoch = 0, walk.saturations = 0, walk.random = 0;
    walk.hops = 0, walk.scored_count = 0, walk.restarts = 0, walk.lock_attempts = 0, walk.lock_skips = 0;
    walk.reselections = 0, walk.locked_drops = 0, walk.tail_compactions = 0, walk.tail_dropped = 0;
    return walk;
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

/** The chains a wave's appended nodes thread their picks onto for the back-link launch behind them:
 *  each node's last pick, unknown for none, the pick each was threaded before, and the nodes any
 *  appended node picked with their count. */
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

        // Write the payload, derive the norms, and park the query in one pass.
        device uint4 const* const staged = metal_row(staging, new_node_id - arguments.first_id, walk.row_fragments);
        device uint4* const owned = walk.vector_for(new_node_id);
        for (uint fragment_index = lane; fragment_index < walk.row_fragments; fragment_index += warp_t::width_k)
            owned[fragment_index] = staged[fragment_index];
        norms_t const new_norms = metal_stage_query(walk, staged);
        if (lane == 0 && walk.has_norms) norms[new_node_id] = new_norms;
        warp_t::sync();

        typename metal_frontier_of<slots_>::type frontier;
        frontier.clear(walk);
        metal_fold_t fold {};
        if (new_node_id != 0) metal_search_ahead(walk, frontier, new_norms, 0, new_node_id, arguments.end_id, fold);
        walk.tail_compactions += frontier.compactions();
        walk.tail_dropped += frontier.dropped();
        uint const selected = metal_select_diverse(walk, frontier);
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
            uint const pick = (new_node_id - arguments.first_id) * walk.connectivity + slot;
            uint const previous = warp_t::exchange_acquire(links.heads + target, pick);
            links.next[pick] = previous;
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
        metal_back_link_one(walk, node_id, links.heads, links.next, arguments.first_id, walk.selection_nodes - 1);
        if (lane == 0) links.heads[node_id] = metal_unknown_k;
    }
    metal_flush(walk);
}

/** One simdgroup walks one query at a time, assigning itself queries from the batch counter,
 *  writing its nearest IDs and distances, or under @p fold.keeps the pairs it scored and their
 *  count, ranked under indexer heads when the fold names any. Slot zero walks the arena frontier
 *  and ranks everything it kept. */
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
        // Simdgroups no other walk holds widen this one, never past the rows that fill its candidates.
        uint live = lane == 0 ? atomic_fetch_add_explicit(counters + metal_live_k, 1u, memory_order_relaxed) + 1 : 0;
        live = warp_t::shuffle(live, 0);
        walk.walk_width = walk_width(arguments.concurrent_walks, live, arguments.queries_count - query,
                                     arguments.filling_nodes);
        walk.epoch = arguments.epoch_base + query + 1;
        walk.random = metal_lane_generator(query, lane);
        norms_t const query_norms = metal_stage_query(walk, metal_row(queries, query, walk.row_fragments));
        typename metal_frontier_of<slots_>::type frontier;
        frontier.clear(walk);
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
        metal_search_ahead(walk, frontier, query_norms, 0, arguments.sampling_bound, arguments.sampling_bound, fold);
        if (lane == 0) atomic_fetch_sub_explicit(counters + metal_live_k, 1u, memory_order_relaxed);
        walk.tail_compactions += frontier.compactions();
        walk.tail_dropped += frontier.dropped();
        if (keeps) {
            for (uint slot = fold.written + lane; slot < arguments.count; slot += warp_t::width_k)
                fold.ids[slot] = metal_unknown_k;
            if (lane == 0) scored_counts[query] = fold.scored;
            continue;
        }
        device uint* const ids_row = metal_row(result_ids, query, arguments.count);
        device float* const distances_row = metal_row(result_distances, query, arguments.count);
        uint const metric = walk.metric, count = arguments.count;
        auto write_rank = [&](uint rank, ulong entry) {
            if (rank >= count) return;
            ids_row[rank] = metal_id_of(entry);
            distances_row[rank] = kit_t::convert_to_distance(metric, metal_distance_of(entry));
        };
        frontier.for_each_rank(count, lane, write_rank);
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
 *  slot zero's arena frontier past the widest rung, each under the name the host looks it up by. */

template <uint slots_>
kernel void usearch_dense_insert_kernel(constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
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

template [[host_name("usearch_dense_insert_kernel_0")]] kernel void usearch_dense_insert_kernel<0>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("usearch_dense_insert_kernel_2")]] kernel void usearch_dense_insert_kernel<2>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("usearch_dense_insert_kernel_4")]] kernel void usearch_dense_insert_kernel<4>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("usearch_dense_insert_kernel_8")]] kernel void usearch_dense_insert_kernel<8>(
    constant metal_arguments_t&, device uint4*, device norms_t*, device uint*, device uint4 const*, device uchar*,
    device atomic_uint*, device uint*, device uint*, device uint*, device atomic_uint*, uint, uint, uint);
template [[host_name("usearch_dense_insert_kernel_16")]] kernel void usearch_dense_insert_kernel<16>(
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

/*  Searches and graph products share one entry point per width, plus slot zero's arena frontier past the widest
 *  rung: the word at @c metal_keeps_buffer_k says which the launch writes. */

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

#pragma region Gathers and Splits

/** Copies the packed row of every listed node into its output row, one simdgroup per entry, as the
 *  stored scalar: a lookup's gather into memory the host reads in place. */
[[host_name("metal_gather_vectors_kernel")]] kernel void metal_gather_vectors_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint4 const* vectors [[buffer(metal_vectors_buffer_k)]],
    device uint2 const* rows [[buffer(metal_gathered_rows_buffer_k)]],
    device uchar* outputs [[buffer(metal_gathered_outputs_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint const slices = groups * metal_simdgroups_k;
    for (uint entry = group * metal_simdgroups_k + simdgroup; entry < arguments.queries_count; entry += slices) {
        // The node first, then its output row, as the host's eight-byte entries lay them.
        uint2 const row = rows[entry];
        device uchar const* const from = (device uchar const*)metal_row(vectors, row.x, arguments.row_fragments);
        device uchar* const into = outputs + ulong(row.y) * arguments.packed_bytes;
        for (uint byte = lane; byte < arguments.packed_bytes; byte += metal_simdgroup_width_k) into[byte] = from[byte];
    }
}

/** Sums a batch's per-query counts into the counters' matched words on the grid's first simdgroup
 *  alone, so the host reads one word instead of every query's count. */
[[host_name("metal_sum_counts_kernel")]] kernel void metal_sum_counts_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device ulong const* counts [[buffer(metal_counts_buffer_k)]],
    device uint* counters [[buffer(metal_counters_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (group != 0 || simdgroup != 0) return;
    ulong sum = 0;
    for (uint query = lane; query < arguments.queries_count; query += metal_simdgroup_width_k) sum += counts[query];
    for (uint mask = metal_simdgroup_width_k / 2; mask; mask >>= 1) sum += warp_t::shuffle_xor(sum, mask);
    if (lane != 0) return;
    counters[metal_matched_k] = uint(sum);
    counters[metal_matched_k + 1] = uint(sum >> 32);
}

/** Whether bit @p row of @p words is set. */
inline bool metal_bitset_contains(device uint const* words, uint row) { return (words[row / 32] >> (row % 32)) & 1u; }

/** Sets bit @p row of @p words, which other lanes set beside it. */
inline void metal_bitset_insert(device uint* words, uint row) {
    atomic_fetch_or_explicit((device atomic_uint*)(words + row / 32), 1u << (row % 32), memory_order_relaxed);
}

/** Whether the round before the bound one placed nothing, which ends the growth: a round queued
 *  past convergence leaves at entry, as every later pass of it does; a first round, having none
 *  before it, never ends the growth. */
inline bool metal_converged(constant metal_arguments_t& arguments, device uint const* placed) {
    return arguments.round > 1 && placed[(arguments.round & 1u) ^ 1u] == 0;
}

/** A growth round's first pass, one simdgroup per frontier row: a row of a part with room proposes
 *  itself to every unplaced row it links to, and each keeps its lowest-ID proposer, so no
 *  simdgroup order sways the round, as the CUDA pass does. */
[[host_name("metal_partition_propose_kernel")]] kernel void metal_partition_propose_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint const* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint const* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device atomic_uint* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint const* frontier [[buffer(metal_frontier_buffer_k)]],
    device uint const* part_sizes [[buffer(metal_part_sizes_buffer_k)]],
    device uint const* placed [[buffer(metal_placed_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (metal_converged(arguments, placed)) return;
    uint const slices = groups * metal_simdgroups_k, rows = arguments.end_id;
    for (uint row = group * metal_simdgroups_k + simdgroup; row < rows; row += slices) {
        if (!metal_bitset_contains(frontier, row) || part_sizes[node_parts[row]] >= arguments.part_capacity) continue;
        device uint const* const links = metal_row(neighbors, row, arguments.connectivity);
        for (uint slot = lane; slot < arguments.connectivity; slot += metal_simdgroup_width_k) {
            uint const target = links[slot];
            if (target == metal_unknown_k || target == row || target >= rows || node_parts[target] != metal_unknown_k)
                continue;
            uint held = atomic_load_explicit(proposals + target, memory_order_relaxed);
            while (row < held && !atomic_compare_exchange_weak_explicit(proposals + target, &held, row,
                                                                        memory_order_relaxed, memory_order_relaxed)) {}
        }
    }
}

/** A growth round's second pass, one thread per row: counts the rows proposed to each part, and
 *  empties the next frontier and the round's placed word for the passes after it, since the host
 *  writes nothing between the round's command buffers. */
[[host_name("metal_partition_count_kernel")]] kernel void metal_partition_count_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint const* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device uint const* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint* next_frontier [[buffer(metal_next_frontier_buffer_k)]],
    device atomic_uint* round_counts [[buffer(metal_round_counts_buffer_k)]],
    device uint* placed [[buffer(metal_placed_buffer_k)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    if (metal_converged(arguments, placed)) return;
    for (uint word = position; word < (arguments.end_id + 31) / 32; word += threads) next_frontier[word] = 0;
    if (position == 0) placed[arguments.round & 1u] = 0;
    for (uint row = position; row < arguments.end_id; row += threads) {
        uint const proposal = proposals[row];
        if (proposal != metal_unknown_k)
            atomic_fetch_add_explicit(round_counts + node_parts[proposal], 1u, memory_order_relaxed);
    }
}

/** A growth round's third pass, one thread per row: places every row proposed to a part with room
 *  for the round's whole count in the next frontier, counting them in the round's placed word; the
 *  rows a round count would overflow stay proposed for the fill. Proposers were placed in earlier
 *  rounds, so no thread rewrites a part another one reads. */
[[host_name("metal_partition_accept_kernel")]] kernel void metal_partition_accept_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device uint* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint* next_frontier [[buffer(metal_next_frontier_buffer_k)]],
    device uint const* part_sizes [[buffer(metal_part_sizes_buffer_k)]],
    device uint const* round_counts [[buffer(metal_round_counts_buffer_k)]],
    device uint* placed [[buffer(metal_placed_buffer_k)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    if (metal_converged(arguments, placed)) return;
    for (uint row = position; row < arguments.end_id; row += threads) {
        uint const proposal = proposals[row];
        if (proposal == metal_unknown_k) continue;
        uint const part = node_parts[proposal];
        if (round_counts[part] > arguments.part_capacity - part_sizes[part]) continue;
        node_parts[row] = part;
        proposals[row] = metal_unknown_k;
        metal_bitset_insert(next_frontier, row);
        atomic_fetch_add_explicit((device atomic_uint*)(placed + (arguments.round & 1u)), 1u, memory_order_relaxed);
    }
}

/** A growth round's last pass, one threadgroup per part: a part whose round count fit takes it into
 *  its size, and one the round count overflows places its lowest-ID proposed rows up to its room in
 *  the next frontier, and fills up. Either way the part's proposals and round count clear for the
 *  round that follows. */
[[host_name("metal_partition_fill_kernel")]] kernel void metal_partition_fill_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device uint* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint* next_frontier [[buffer(metal_next_frontier_buffer_k)]],
    device uint* part_sizes [[buffer(metal_part_sizes_buffer_k)]],
    device uint* round_counts [[buffer(metal_round_counts_buffer_k)]],
    device uint* placed [[buffer(metal_placed_buffer_k)]], uint part [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]], uint group_threads [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simdgroup_counts[metal_simdgroups_k];
    if (metal_converged(arguments, placed)) return;
    uint const wanted = round_counts[part], room = arguments.part_capacity - part_sizes[part];
    threadgroup_barrier(mem_flags::mem_device);
    if (wanted > room)
        for (uint first = 0, taken = 0; first < arguments.end_id; first += group_threads) {
            uint const row = first + thread_index;
            uint const proposal = row < arguments.end_id ? proposals[row] : metal_unknown_k;
            bool const proposed_here = proposal != metal_unknown_k && node_parts[proposal] == part;
            uint const voted = warp_t::ballot(proposed_here);
            if (lane == 0) simdgroup_counts[simdgroup] = popcount(voted);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint rank = taken + popcount(voted & warp_t::below(lane)), total = 0;
            for (uint other = 0; other != metal_simdgroups_k; ++other) {
                if (other < simdgroup) rank += simdgroup_counts[other];
                total += simdgroup_counts[other];
            }
            if (proposed_here) {
                if (rank < room) {
                    node_parts[row] = part;
                    metal_bitset_insert(next_frontier, row);
                }
                proposals[row] = metal_unknown_k;
            }
            taken += total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    if (thread_index != 0) return;
    part_sizes[part] += wanted > room ? room : wanted;
    round_counts[part] = 0;
    if (wanted > room)
        atomic_fetch_add_explicit((device atomic_uint*)(placed + (arguments.round & 1u)), room, memory_order_relaxed);
}

#pragma endregion Gathers and Splits

/*  The refine round on the device, the Metal twins of the CUDA and HIP kernels: a health check
 *  from one root with exact depths both ways, island components labeled in parallel, the heal
 *  linking them with no distance, a gather filing every entry under its nearest hub, and a
 *  restitch of each neighborhood all against all. Rows and hubs stage in threadgroup memory while
 *  they fit its 32 KB, and are read in place past it, as the walks read theirs. */

#pragma region Rounds

/** Whether bit @p row of @p words is set, read through the word other lanes set bits in. */
inline bool metal_bit(device uint* words, uint row) {
    return (warp_t::load_relaxed(words + row / 32) >> (row % 32)) & 1u;
}

/** Sets bit @p row of @p words, answering whether it was set already. */
inline bool metal_bitset_claim(device uint* words, uint row) {
    uint const bit = 1u << (row % 32);
    return (warp_t::fetch_or_relaxed(words + row / 32, bit) & bit) != 0;
}

/** Clears bit @p row of @p words, answering whether it was set. */
inline bool metal_bitset_release(device uint* words, uint row) {
    uint const bit = 1u << (row % 32);
    return (warp_t::fetch_and_relaxed(words + row / 32, ~bit) & bit) != 0;
}

/** Adds @p value to the tally at @p counter, carrying into its high word. */
inline void metal_tally_add(device uint* counters, uint counter, uint value) {
    if (!value) return;
    uint const low = warp_t::fetch_add_relaxed(counters + counter, value);
    if (low + value < low) warp_t::fetch_add_relaxed(counters + counter + 1, 1u);
}

/** The dot of two rows of @p fragments, each staged in threadgroup memory or read in place. */
template <typename left_type_, typename right_type_>
inline dot_t metal_rows_dot(left_type_ left, right_type_ right, uint fragments) {
    dot_t sum = dot_t(0);
    for (uint fragment_index = 0; fragment_index != fragments; ++fragment_index)
        sum = kit_t::accumulate_dot(left[fragment_index], right[fragment_index], sum);
    return sum;
}

/** The columns of one health check, as the RAM audit names them, beside the round's counters:
 *  depths both ways a byte each, the exact in-link and shallower in-link counts, and the reached,
 *  reaching-root and two-ways bits. */
struct metal_health_t {
    device uint* neighbors;
    device uint* counters;
    device uchar* depths_from_root;
    device uchar* depths_to_root;
    device uint* in_links;
    device uint* shallower_in_links;
    device uint* reached;
    device uint* reaches_root;
    device uint* two_ways_to_root;
    uint rows_count, connectivity;

    device uint* row(uint node) { return metal_row(neighbors, node, connectivity); }
    uint depth_from_root(uint node) { return warp_t::load_relaxed(depths_from_root, node); }
    uint depth_to_root(uint node) { return warp_t::load_relaxed(depths_to_root, node); }
    bool in_component(uint node) { return metal_bit(reached, node) && metal_bit(reaches_root, node); }

    /** The in-links of @p node read saturated at 255, so every decision matches the RAM audit's
     *  byte. */
    uint in_links_of(uint node) { return min(warp_t::load_relaxed(in_links + node), 255u); }

    /** Marks @p node 's way to the root one past @p lowest, and its two ways when two links met it;
     *  answers whether it marked. */
    uint mark_reaches_root(uint node, uint lowest, uint at_lowest) {
        if (lowest == metal_unknown_depth_k) return 0;
        uint const depth = min(lowest + 1u, uint(metal_deepest_depth_k));
        depths_to_root[node] = uchar(depth);
        metal_bitset_claim(reaches_root, node);
        if (at_lowest > 1 && lowest < depth) metal_bitset_claim(two_ways_to_root, node);
        return 1;
    }

    /** Takes one of @p node 's links from the depth before it, answering whether another
     *  remains. */
    bool take_shallower_in_link_if_spare(uint node) {
        uint observed = warp_t::load_relaxed(shallower_in_links + node);
        while (observed > 1 && !warp_t::compare_exchange_relaxed(shallower_in_links + node, observed, observed - 1))
            observed = warp_t::load_relaxed(shallower_in_links + node);
        return observed > 1;
    }

    /** Takes one of @p node 's in-links for a swap, answering whether another remains. */
    bool take_in_link_if_spare(uint node) {
        uint observed = warp_t::load_relaxed(in_links + node);
        while (observed && !warp_t::compare_exchange_relaxed(in_links + node, observed, observed - 1))
            observed = warp_t::load_relaxed(in_links + node);
        return observed > 1;
    }

    /** Takes the link from @p owner to @p resident for a swap: one not from the depth before the
     *  resident goes freely, a shallower in-link only while another remains. */
    bool take_link_for_swap_if_spare(uint owner, uint resident) {
        uint const owner_depth = depth_from_root(owner);
        uint const child = owner_depth < metal_deepest_depth_k ? owner_depth + 1 : uint(metal_unknown_depth_k);
        return child == metal_unknown_depth_k || depth_from_root(resident) != child ||
               take_shallower_in_link_if_spare(resident);
    }

    /** Takes that link when the reach rule releases it: never the resident's last shallower
     *  in-link, and a link toward the root only while the owner keeps another as close to it. */
    bool take_link_under_lock_if_spare(uint owner, uint resident) {
        bool const way_to_root = metal_bit(reaches_root, owner) && depth_to_root(owner) != 0 &&
                                 depth_to_root(resident) < depth_to_root(owner);
        if (way_to_root && !metal_bit(two_ways_to_root, owner)) return false;
        if (!take_link_for_swap_if_spare(owner, resident)) return false;
        if (way_to_root) metal_bitset_release(two_ways_to_root, owner);
        return true;
    }

    /** Files a heal's link from @p owner to island entry @p member in the columns the reach rule
     *  keeps, as the RAM audit's @c record_heal_link does. */
    void record_heal_link(uint owner, uint member) {
        uint const owner_depth = depth_from_root(owner);
        if (depth_from_root(member) == metal_unknown_depth_k && owner_depth < metal_deepest_depth_k) {
            depths_from_root[member] = uchar(owner_depth + 1);
            shallower_in_links[member] = 1;
        }
        uint const member_depth_to_root = depth_to_root(member);
        if (depth_to_root(owner) == metal_unknown_depth_k && member_depth_to_root < metal_deepest_depth_k)
            depths_to_root[owner] = uchar(member_depth_to_root + 1);
    }
};

/** The lowest depth to root among the links one simdgroup read, and how many links met it. */
struct metal_way_to_root_t {
    uint lowest = metal_unknown_depth_k, at_lowest = 0;

    void note(uint depth) {
        if (depth < lowest) lowest = depth, at_lowest = 1;
        else at_lowest += depth == lowest && depth != metal_unknown_depth_k;
    }

    /** The lowest depth every lane met, and the links meeting it, on every lane. */
    metal_way_to_root_t reduced() const {
        uint const least = warp_t::reduce_min(lowest);
        metal_way_to_root_t way;
        way.lowest = least, way.at_lowest = warp_t::reduce_add(lowest == least ? at_lowest : 0u);
        return way;
    }
};

/** Calls @p visit on the lanes holding a link of @p node 's row below the rows but its own, the
 *  links past the first sentinel ignored; a row whose slot zero is the sentinel or the node itself
 *  holds none. Every lane calls it; @p visit runs on some. */
template <typename visit_type_>
inline void metal_for_each_link(thread metal_health_t& health, uint node, uint lane, visit_type_ visit) {
    device uint* const row = health.row(node);
    uint const first = warp_t::shuffle(lane == 0 ? warp_t::load_acquire(row) : 0u, 0);
    if (first == metal_unknown_k || first == node) return;
    for (uint base = 0; base < health.connectivity; base += warp_t::width_k) {
        uint const slot = base + lane;
        uint const target = slot >= health.connectivity ? metal_unknown_k
                            : slot                      ? warp_t::load_relaxed(row + slot)
                                                        : first;
        uint const ends = warp_t::ballot(target == metal_unknown_k);
        bool const listed = !ends || lane < ctz(ends);
        if (listed && target < health.rows_count && target != node) visit(target);
        if (ends) return;
    }
}

/** The rank of @p node among the @p islands ascending island nodes, or @p islands when none. */
inline uint metal_island_rank(device uint* island_nodes, uint islands, uint node) {
    uint low = 0, high = islands;
    while (low < high) {
        uint const middle = (low + high) / 2;
        if (island_nodes[middle] < node) low = middle + 1;
        else high = middle;
    }
    return low < islands && island_nodes[low] == node ? low : islands;
}

/** Calls @p visit with the rank of every island node that island node @p node links to, the islands
 *  as ordered, so a way to the root marked meanwhile changes no component's links. */
template <typename visit_type_>
inline void metal_for_each_island_link(thread metal_health_t& health, device uint* island_nodes, uint islands,
                                       uint node, uint lane, visit_type_ visit) {
    metal_for_each_link(health, node, lane, [&](uint target) {
        uint const rank = metal_island_rank(island_nodes, islands, target);
        if (rank != islands) visit(rank);
    });
}

/** Files @p key of island member @p rank under its @p component in two 32-bit minimums, the key's
 *  in the high word on the first pass and the least rank holding it in the low word on the second:
 *  64-bit atomic minimums exist only from Apple GPU family 9, and one source serves them all. */
inline void metal_claim_entry(device uint* entries, uint component, uint key) {
    warp_t::fetch_min_relaxed(entries + 2 * component + 1, key);
}
inline void metal_settle_entry(device uint* entries, uint component, uint key, uint rank) {
    if (entries[2 * component + 1] == key) warp_t::fetch_min_relaxed(entries + 2 * component, rank);
}

/** The key an island member stands for its component by, the least first: a member a link crosses
 *  through, then the fewest in-links, ties by a keyed draw, as the RAM audit orders them. */
inline uint metal_entry_key(thread metal_health_t& health, uint node, uint parent, uint roles) {
    bool const crosses = parent != metal_unknown_k || (roles & metal_links_to_component_k) != 0;
    return (crosses ? 0u : 1u << 31) | health.in_links_of(node) << 23 | uint(metal_mix_bits(node) >> 41);
}

/** The root of @p rank 's set among the union-find @p parents, halving its path. */
inline uint metal_find(device uint* parents, uint rank) {
    for (;;) {
        uint const parent = warp_t::load_relaxed(parents + rank);
        if (parent == rank) return rank;
        uint const grand = warp_t::load_relaxed(parents + parent);
        if (grand != parent) warp_t::compare_exchange_relaxed(parents + rank, parent, grand);
        rank = grand;
    }
}

/** Joins the sets of @p first and @p second, the higher root hooked under the lower one. */
inline void metal_unite(device uint* parents, uint first, uint second) {
    for (;;) {
        first = metal_find(parents, first), second = metal_find(parents, second);
        if (first == second) return;
        if (warp_t::compare_exchange_relaxed(parents + max(first, second), max(first, second), min(first, second)))
            return;
    }
}

/** The health columns bound at their slots, every check's kernel taking them alike. */
inline metal_health_t metal_health(constant metal_arguments_t& arguments, device uint* neighbors, device uint* counters,
                                   device uchar* depths_from_root, device uchar* depths_to_root, device uint* in_links,
                                   device uint* shallower_in_links, device uint* reached, device uint* reaches_root,
                                   device uint* two_ways_to_root) {
    metal_health_t health;
    health.neighbors = neighbors, health.counters = counters;
    health.depths_from_root = depths_from_root, health.depths_to_root = depths_to_root;
    health.in_links = in_links, health.shallower_in_links = shallower_in_links;
    health.reached = reached, health.reaches_root = reaches_root, health.two_ways_to_root = two_ways_to_root;
    health.rows_count = arguments.end_id, health.connectivity = arguments.connectivity;
    return health;
}

/** One depth of a health check from its root, one simdgroup an entry: every entry at @c depth
 *  counts an in-link into each target, reaches the targets no entry reached before one depth
 *  deeper, counts a shallower in-link into every target at that depth or still being stored, and
 *  marks its own way to the root in the same read, as the RAM audit's @c expand_ does. */
[[host_name("metal_health_expand_kernel")]] kernel void metal_health_expand_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint const depth = arguments.depth;
    bool const saturated = depth >= metal_deepest_depth_k;
    uint const own = min(depth, uint(metal_deepest_depth_k)), child = min(depth + 1u, uint(metal_deepest_depth_k));
    uint reached_count = 0, marked_count = 0;
    for (uint row = group * metal_simdgroups_k + simdgroup; row < health.rows_count;
         row += groups * metal_simdgroups_k) {
        bool expands = lane == 0 && metal_bit(reached, row) && health.depth_from_root(row) == own;
        // Past the deepest recorded depth an entry marks its expansion in its shallower in-links.
        if (expands && saturated) expands = !warp_t::exchange_relaxed(shallower_in_links + row, 1u);
        if (!warp_t::shuffle(uint(expands), 0)) continue;
        metal_way_to_root_t way;
        metal_for_each_link(health, row, lane, [&](uint target) {
            if (!warp_t::fetch_add_relaxed(in_links + target, 1u) && !metal_bitset_claim(reached, target)) {
                depths_from_root[target] = uchar(child);
                ++reached_count;
            }
            // A depth still unknown is one another simdgroup reached this depth and is storing.
            uint const target_depth = health.depth_from_root(target);
            if (child != metal_deepest_depth_k && (target_depth == child || target_depth == metal_unknown_depth_k))
                warp_t::fetch_add_relaxed(shallower_in_links + target, 1u);
            way.note(health.depth_to_root(target));
        });
        way = way.reduced();
        if (lane == 0 && health.depth_to_root(row) == metal_unknown_depth_k)
            marked_count += health.mark_reaches_root(row, way.lowest, way.at_lowest);
    }
    reached_count = warp_t::reduce_add(reached_count);
    if (lane == 0 && reached_count) warp_t::fetch_add_relaxed(counters + metal_round_reached_k, reached_count);
    if (lane == 0 && marked_count) warp_t::fetch_add_relaxed(counters + metal_round_marked_k, marked_count);
}

/** Re-reads the reached entries not yet marked, one simdgroup an entry, marking each that now links
 *  to an entry reaching the root, as the RAM audit's @c recheck_reaches_root_ does. */
[[host_name("metal_health_recheck_kernel")]] kernel void metal_health_recheck_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint marked_count = 0;
    for (uint row = group * metal_simdgroups_k + simdgroup; row < health.rows_count;
         row += groups * metal_simdgroups_k) {
        bool const rechecks = lane == 0 && metal_bit(reached, row) && !metal_bit(reaches_root, row);
        if (!warp_t::shuffle(uint(rechecks), 0)) continue;
        metal_way_to_root_t way;
        metal_for_each_link(health, row, lane, [&](uint target) { way.note(health.depth_to_root(target)); });
        way = way.reduced();
        if (lane == 0) marked_count += health.mark_reaches_root(row, way.lowest, way.at_lowest);
    }
    if (lane == 0 && marked_count) warp_t::fetch_add_relaxed(counters + metal_round_marked_k, marked_count);
}

/** Writes the entries outside the root's component in ascending order on one threadgroup, a word of
 *  entries a thread and a threadgroup's words at a time, so the ranks match the RAM audit's. */
[[host_name("metal_order_islands_kernel")]] kernel void metal_order_islands_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    uint thread_index [[thread_position_in_threadgroup]], uint group_threads [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simdgroup_counts[metal_simdgroups_k];
    uint const rows = arguments.end_id, words = (rows + 31) / 32;
    uint written = 0;
    for (uint first = 0; first < words; first += group_threads) {
        uint const word = first + thread_index;
        uint lanes = 0;
        if (word < words) {
            uint const visible = rows - word * 32;
            lanes = ~(reached[word] & reaches_root[word]) & (visible >= 32 ? 0xFFFFFFFFu : (1u << visible) - 1u);
        }
        uint const count = popcount(lanes), before = simd_prefix_exclusive_sum(count);
        if (lane == warp_t::width_k - 1) simdgroup_counts[simdgroup] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint offset = written + before, total = 0;
        for (uint other = 0; other != metal_simdgroups_k; ++other) {
            if (other < simdgroup) offset += simdgroup_counts[other];
            total += simdgroup_counts[other];
        }
        for (; lanes; lanes &= lanes - 1) island_nodes[offset++] = word * 32 + ctz(lanes);
        written += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0) counters[metal_round_islands_k] = written;
}

/** Opens every island's columns, one simdgroup an island: marks those linking into the root's
 *  component and their ways to the root, and starts its labels from its own rank, as the RAM
 *  audit's @c gather_islands_ does. */
[[host_name("metal_health_islands_kernel")]] kernel void metal_health_islands_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_labels [[buffer(metal_island_labels_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]],
    device uint* island_roles [[buffer(metal_island_roles_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    for (uint rank = group * metal_simdgroups_k + simdgroup; rank < arguments.islands;
         rank += groups * metal_simdgroups_k) {
        uint const node = island_nodes[rank];
        bool links = false;
        metal_way_to_root_t way;
        metal_for_each_link(health, node, lane, [&](uint target) {
            if (!health.in_component(target)) return;
            links = true;
            way.note(health.depth_to_root(target));
        });
        links = simd_any(links);
        way = way.reduced();
        if (lane != 0) continue;
        island_roles[rank] = links ? uint(metal_links_to_component_k) : 0u;
        island_parents[rank] = metal_unknown_k;
        island_components[rank] = metal_unknown_k;
        island_labels[2 * rank] = island_labels[2 * rank + 1] = rank;
        // The members a component counts while labeled, until the labels pass files its entry here.
        island_entries[2 * rank] = island_entries[2 * rank + 1] = 0;
        // A reached island never links into the component, or a recheck marked it.
        health.mark_reaches_root(node, way.lowest, way.at_lowest);
    }
}

/** One pass of the island components, the strong components among the islands, each named by its
 *  least rank as the RAM audit's Tarjan walk names it. Pass zero lowers, one simdgroup an island,
 *  the forward label of every unlabeled island it links to and its own backward label to theirs,
 *  and marks its way to the root through the islands it links to; pass one settles, a thread an
 *  island, every one whose labels meet, as both name the least rank it reaches and is reached
 *  from, counting its members, and restarts the rest from their own ranks. */
[[host_name("metal_island_components_kernel")]] kernel void metal_island_components_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_labels [[buffer(metal_island_labels_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint const islands = arguments.islands;
    uint changed = 0;
    if (arguments.pass == 1) {
        for (uint rank = position; rank < islands; rank += threads) {
            if (island_components[rank] != metal_unknown_k) continue;
            uint const forward = island_labels[2 * rank], backward = island_labels[2 * rank + 1];
            if (forward == backward) {
                island_components[rank] = forward;
                warp_t::fetch_add_relaxed(island_entries + 2 * forward, 1u);
                ++changed;
            }
            else island_labels[2 * rank] = island_labels[2 * rank + 1] = rank;
        }
    }
    else
        for (uint rank = group * metal_simdgroups_k + simdgroup; rank < islands; rank += groups * metal_simdgroups_k) {
            uint const node = island_nodes[rank];
            bool const open = island_components[rank] == metal_unknown_k;
            bool const unmarked = warp_t::shuffle(
                uint(lane == 0 && health.depth_to_root(node) == metal_unknown_depth_k), 0);
            if (!open && !unmarked) continue;
            uint const forward = open ? warp_t::load_relaxed(island_labels + 2 * rank) : 0u;
            uint lowered = 0;
            metal_way_to_root_t way;
            metal_for_each_island_link(health, island_nodes, islands, node, lane, [&](uint next) {
                if (unmarked) way.note(health.depth_to_root(island_nodes[next]));
                if (!open || island_components[next] != metal_unknown_k) return;
                lowered += warp_t::fetch_min_relaxed(island_labels + 2 * next, forward) > forward;
                uint const backward = warp_t::load_relaxed(island_labels + 2 * next + 1);
                lowered += warp_t::fetch_min_relaxed(island_labels + 2 * rank + 1, backward) > backward;
            });
            way = way.reduced();
            lowered = warp_t::reduce_add(lowered);
            if (lane == 0 && unmarked) lowered += health.mark_reaches_root(node, way.lowest, way.at_lowest);
            if (lane == 0) changed += lowered;
        }
    changed = warp_t::reduce_add(changed);
    if (lane == 0 && changed) warp_t::fetch_add_relaxed(counters + metal_round_relabeled_k, changed);
}

/** Counts the entries no reached entry links to and those linked to by fewer entries than they link
 *  to, one simdgroup an entry, and files the least entry of the root's component linking to each
 *  island as its parent, as the RAM audit's @c count_in_and_out_links_ does. */
[[host_name("metal_health_count_links_kernel")]] kernel void metal_health_count_links_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint unlinked = 0, underlinked = 0;
    for (uint row = group * metal_simdgroups_k + simdgroup; row < health.rows_count;
         row += groups * metal_simdgroups_k) {
        bool const member = warp_t::shuffle(uint(lane == 0 && health.in_component(row)), 0);
        uint out_links = 0;
        metal_for_each_link(health, row, lane, [&](uint target) {
            ++out_links;
            if (!member || health.in_component(target)) return;
            uint const rank = metal_island_rank(island_nodes, arguments.islands, target);
            if (rank != arguments.islands) warp_t::fetch_min_relaxed(island_parents + rank, row);
        });
        out_links = warp_t::reduce_add(out_links);
        uint const linked_by = health.in_links_of(row);
        unlinked += linked_by == 0, underlinked += linked_by < out_links;
    }
    if (lane != 0) return;
    if (unlinked) warp_t::fetch_add_relaxed(counters + metal_round_unlinked_k, unlinked);
    if (underlinked) warp_t::fetch_add_relaxed(counters + metal_round_underlinked_k, underlinked);
}

/** One pass of labeling the islands, as the RAM audit's @c label_islands_ does. Pass zero, a thread
 *  an island, marks every component first unreached and stranded, empties the entries and starts
 *  the weak components' union-find, the root's component in the last slot. Pass one, a simdgroup an
 *  island, clears the roles its links disprove, files its entry key and joins the weak components
 *  its links join. Pass two, a thread an island, settles the entries, spreads each component's
 *  roles to its members and counts the unreached and stranded ones. Pass three counts the weak
 *  components, the union-find's roots. */
[[host_name("metal_label_islands_kernel")]] kernel void metal_label_islands_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_labels [[buffer(metal_island_labels_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]],
    device uint* island_roles [[buffer(metal_island_roles_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint const islands = arguments.islands;
    // The forward and backward labels served the components; the union-find takes their slots.
    device uint* const weak = island_labels;
    uint const component_roles = metal_unreached_island_k | metal_stranded_island_k;
    if (arguments.pass == 0) {
        for (uint rank = position; rank <= islands; rank += threads) {
            weak[rank] = rank;
            if (rank == islands) continue;
            if (island_components[rank] == rank) warp_t::fetch_or_relaxed(island_roles + rank, component_roles);
            island_entries[2 * rank] = island_entries[2 * rank + 1] = metal_unknown_k;
        }
        return;
    }
    if (arguments.pass == 1) {
        for (uint rank = group * metal_simdgroups_k + simdgroup; rank < islands; rank += groups * metal_simdgroups_k) {
            uint const node = island_nodes[rank], component = island_components[rank];
            uint const parent = island_parents[rank], roles = warp_t::load_relaxed(island_roles + rank);
            metal_for_each_island_link(health, island_nodes, islands, node, lane, [&](uint linked) {
                metal_unite(weak, rank, linked);
                uint const next = island_components[linked];
                if (next == component) return;
                warp_t::fetch_and_relaxed(island_roles + component, ~uint(metal_stranded_island_k));
                warp_t::fetch_and_relaxed(island_roles + next, ~uint(metal_unreached_island_k));
            });
            if (lane != 0) continue;
            if (parent != metal_unknown_k) {
                warp_t::fetch_or_relaxed(island_roles + rank, metal_linked_from_component_k);
                warp_t::fetch_and_relaxed(island_roles + component, ~uint(metal_unreached_island_k));
            }
            if (roles & metal_links_to_component_k)
                warp_t::fetch_and_relaxed(island_roles + component, ~uint(metal_stranded_island_k));
            if (parent != metal_unknown_k || (roles & metal_links_to_component_k)) metal_unite(weak, rank, islands);
            metal_claim_entry(island_entries, component, metal_entry_key(health, node, parent, roles));
        }
        return;
    }
    uint counted[2] = {0, 0};
    for (uint rank = position; rank <= islands && arguments.pass == 3; rank += threads)
        counted[0] += metal_find(weak, rank) == rank;
    for (uint rank = position; rank < islands && arguments.pass == 2; rank += threads) {
        uint const node = island_nodes[rank], component = island_components[rank];
        uint const roles = warp_t::load_relaxed(island_roles + rank);
        metal_settle_entry(island_entries, component, metal_entry_key(health, node, island_parents[rank], roles), rank);
        uint const roles_of_component = warp_t::load_relaxed(island_roles + component) & component_roles;
        if (component != rank) warp_t::fetch_or_relaxed(island_roles + rank, roles_of_component);
        else {
            counted[0] += (roles_of_component & metal_unreached_island_k) != 0;
            counted[1] += (roles_of_component & metal_stranded_island_k) != 0;
        }
    }
    counted[0] = warp_t::reduce_add(counted[0]), counted[1] = warp_t::reduce_add(counted[1]);
    if (lane != 0) return;
    if (arguments.pass == 3 && counted[0]) warp_t::fetch_add_relaxed(counters + metal_round_components_k, counted[0]);
    if (arguments.pass == 2 && counted[0])
        warp_t::fetch_add_relaxed(counters + metal_round_unreached_islands_k, counted[0]);
    if (arguments.pass == 2 && counted[1])
        warp_t::fetch_add_relaxed(counters + metal_round_stranded_islands_k, counted[1]);
}

/** Links @p member from @p owner 's row, as the RAM round's @c add_or_swap_link_ does keeping none:
 *  a free slot first; else, inside the root's component, the slot whose resident is linked to by
 *  the most entries, when it stays linked to by more than @p member and the reach rule releases it.
 *  The heal runs alone, so it writes rows with plain stores; answers whether it wrote. */
inline bool metal_add_or_swap_link(thread metal_health_t& health, device uint* changed, uint owner, uint member,
                                   thread uint& changed_rows) {
    if (owner == member || owner >= health.rows_count || member >= health.rows_count) return false;
    device uint* const row = health.row(owner);
    uint const connectivity = health.connectivity, first = row[0];
    bool written = false;
    if (first == owner) row[0] = member, written = true;
    else {
        bool present = first == member;
        uint free_slot = connectivity, richest = connectivity;
        for (uint slot = 1; slot != connectivity && !present; ++slot) {
            uint const resident = row[slot];
            present = resident == member;
            if (resident == metal_unknown_k) {
                free_slot = slot;
                break;
            }
        }
        uint richest_links = health.in_links_of(member);
        for (uint slot = 0; !present && free_slot == connectivity && slot != connectivity && health.in_component(owner);
             ++slot) {
            uint const resident = slot ? row[slot] : first;
            if (resident >= health.rows_count || health.in_links_of(resident) <= richest_links + 1u) continue;
            richest = slot, richest_links = health.in_links_of(resident);
        }
        uint const displaced = richest == connectivity ? metal_unknown_k : richest ? row[richest] : first;
        if (present) {}
        else if (free_slot != connectivity) row[free_slot] = member, written = true;
        else if (richest != connectivity && health.take_link_under_lock_if_spare(owner, displaced)) {
            row[richest] = member;
            health.take_in_link_if_spare(displaced);
            written = true;
        }
    }
    if (!written) return false;
    warp_t::fetch_add_relaxed(health.in_links + member, 1u);
    changed_rows += !metal_bitset_claim(changed, owner);
    return true;
}

/** Links every unreached or stranded island component through the root's component with no
 *  distance, in rank order on one lane, as the RAM round's @c heal_ does: an unreached one's entry
 *  from the first of its own links inside the component, a stranded one's entry to the entry of the
 *  component linking to it. */
[[host_name("metal_heal_islands_kernel")]] kernel void metal_heal_islands_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* changed [[buffer(metal_changed_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]],
    device uint* island_roles [[buffer(metal_island_roles_buffer_k)]], uint position [[thread_position_in_grid]]) {
    if (position != 0) return;
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint healed = 0, changed_rows = 0;
    for (uint rank = 0; rank != arguments.islands; ++rank) {
        if (island_components[rank] != rank) continue;
        uint const roles = island_roles[rank], member = island_entries[2 * rank], entry = island_nodes[member];
        if (roles & metal_unreached_island_k) {
            device uint* const row = health.row(entry);
            uint owner = metal_unknown_k;
            for (uint slot = 0; slot != health.connectivity && owner == metal_unknown_k; ++slot) {
                uint const link = row[slot];
                if (link == metal_unknown_k) break;
                if (link < health.rows_count && link != entry && health.in_component(link)) owner = link;
            }
            if (owner != metal_unknown_k && metal_add_or_swap_link(health, changed, owner, entry, changed_rows))
                health.record_heal_link(owner, entry), ++healed;
        }
        uint const parent = island_parents[member];
        if ((roles & metal_stranded_island_k) && parent != metal_unknown_k &&
            metal_add_or_swap_link(health, changed, entry, parent, changed_rows))
            health.record_heal_link(entry, parent), ++healed;
    }
    counters[metal_round_healed_k] += healed;
    counters[metal_round_changed_k] += changed_rows;
}

/** Files every entry of a gather's runs under its nearest hub, the lower one among equals, one item
 *  a threadgroup: the item's run stages its hub rows and norms in threadgroup memory while they
 *  fit, or reads them in place, and each simdgroup takes an entry, one hub a lane, whole dots, so
 *  only the minimum crosses lanes. A first gather's run spans IDs; a later one positions among the
 *  entries in neighborhood order. An entry no distance reached joins the first neighborhood of its
 *  run, as on RAM. */
[[host_name("metal_gather_neighborhoods_kernel")]] kernel void metal_gather_neighborhoods_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint4* vectors [[buffer(metal_vectors_buffer_k)]], device norms_t* norms [[buffer(metal_norms_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device ushort* entry_neighborhoods [[buffer(metal_entry_neighborhoods_buffer_k)]],
    device uint* neighborhood_entries [[buffer(metal_neighborhood_entries_buffer_k)]],
    device uint* hub_nodes [[buffer(metal_hub_nodes_buffer_k)]], device uint4* runs [[buffer(metal_runs_buffer_k)]],
    device uint* run_items [[buffer(metal_run_items_buffer_k)]], uint thread_index [[thread_position_in_threadgroup]],
    uint group_threads [[threads_per_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint4 staged[uint(metal_round_staged_bytes_k) / metal_fragment_bytes_k];
    threadgroup uint claimed;
    uint const fragments = arguments.row_fragments, items = run_items[arguments.runs_count];
    bool const has_norms = arguments.has_norms != 0;
    for (;;) {
        if (thread_index == 0) claimed = warp_t::fetch_add_relaxed(counters + metal_round_claimed_k, 1u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint const item = claimed;
        if (item >= items) return;
        uint low = 0, high = arguments.runs_count;
        while (high - low > 1) {
            uint const middle = (low + high) / 2;
            if (run_items[middle] <= item) low = middle;
            else high = middle;
        }
        // A run is its first entry, its width, its first hub and its hubs.
        uint4 const run = runs[low];
        uint const first = (item - run_items[low]) * run.w, bound = min(first + run.w, run.y);
        bool const staging = run.w * (fragments * metal_fragment_bytes_k + 8) <= metal_round_staged_bytes_k;
        threadgroup norms_t* const hub_norms = (threadgroup norms_t*)(staged + run.w * fragments);
        for (uint at = thread_index; staging && at < run.w * fragments; at += group_threads) {
            device uint4 const* const hub_row = metal_row(vectors, hub_nodes[run.z + at / fragments], fragments);
            staged[at] = hub_row[at % fragments];
        }
        for (uint hub = thread_index; staging && hub < run.w; hub += group_threads)
            hub_norms[hub] = has_norms ? norms[hub_nodes[run.z + hub]] : norms_t(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint position = first + simdgroup; position < bound; position += metal_simdgroups_k) {
            uint const node = arguments.pass ? neighborhood_entries[run.x + position] : run.x + position;
            device uint4 const* const entry = metal_row(vectors, node, fragments);
            norms_t const entry_norms = has_norms ? norms[node] : norms_t(0);
            ulong nearest = metal_sentinel_k;
            for (uint hub = lane; hub < run.w; hub += warp_t::width_k) {
                uint const hub_node = hub_nodes[run.z + hub];
                float const distance =
                    staging
                        ? kit_t::distance(arguments.metric, metal_rows_dot(entry, staged + hub * fragments, fragments),
                                          entry_norms, hub_norms[hub])
                        : kit_t::distance(arguments.metric,
                                          metal_rows_dot(entry, metal_row(vectors, hub_node, fragments), fragments),
                                          entry_norms, has_norms ? norms[hub_node] : norms_t(0));
                ulong const scored = isnan(distance) ? metal_sentinel_k : metal_make(distance, hub);
                nearest = scored < nearest ? scored : nearest;
            }
            nearest = warp_t::reduce_min(nearest);
            if (lane == 0) entry_neighborhoods[node] = ushort(nearest == metal_sentinel_k ? 0u : metal_id_of(nearest));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/** One simdgroup's slice of a restitch's scratch: its candidates as scored nodes of their
 *  positions, ascending, two rows' worth; the entry's nearest neighborhood-mates, a row's worth;
 *  the candidates' IDs, the live links first, and distances, two rows' worth each; and the
 *  positions it keeps, a row's worth. */
struct metal_stitch_t {
    device ulong* sorted;
    device ulong* mates;
    device uint* ids;
    device float* distances;
    device uint* kept;
};

/** Puts back what the kept candidates dropped of the @p live_count live links that reachability
 *  needs, exactly as the RAM index's @c keep_reach_ does, on one lane: a member whose last
 *  shallower in-link @p node is, and, when no pick stands closer to the root, the nearest live
 *  member that does. Answers the kept count. */
inline uint metal_keep_reach(thread metal_health_t& health, uint node, metal_stitch_t stitch, uint live_count,
                             uint kept_count, uint connectivity) {
    uint const depth = health.depth_from_root(node), depth_to_root = health.depth_to_root(node);
    // The saturated depth counts no shallower in-links, so its children are never released.
    uint const child = depth < metal_deepest_depth_k ? depth + 1 : uint(metal_unknown_depth_k);
    auto holds = [&](uint candidate) {
        for (uint rank = 0; rank != kept_count; ++rank)
            if (stitch.kept[rank] == candidate) return true;
        return false;
    };
    auto releasable = [&](uint candidate) {
        uint const member = stitch.ids[candidate];
        return child == metal_unknown_depth_k || health.depth_from_root(member) != child || candidate >= live_count ||
               health.take_shallower_in_link_if_spare(member);
    };
    // The kept members gather at the tail, ahead of which the farthest releasable pick gives way.
    uint tail_kept = 0;
    auto keep = [&](uint candidate) {
        if (holds(candidate)) return;
        if (kept_count < connectivity) {
            stitch.kept[kept_count++] = candidate;
            ++tail_kept;
            return;
        }
        uint const tail = kept_count - tail_kept;
        uint at = tail;
        while (at && !releasable(stitch.kept[at - 1])) --at;
        if (!at) return;
        stitch.kept[at - 1] = stitch.kept[tail - 1];
        stitch.kept[tail - 1] = candidate;
        ++tail_kept;
    };
    if (child != metal_unknown_depth_k)
        for (uint candidate = 0; candidate != live_count; ++candidate) {
            uint const member = stitch.ids[candidate];
            if (health.depth_from_root(member) != child || holds(candidate)) continue;
            if (!health.take_shallower_in_link_if_spare(member)) keep(candidate);
        }
    if (depth_to_root != metal_unknown_depth_k && depth_to_root != 0) {
        uint standing = 0;
        while (standing != kept_count && health.depth_to_root(stitch.ids[stitch.kept[standing]]) >= depth_to_root)
            ++standing;
        uint way = standing != kept_count ? stitch.kept[standing] : metal_unknown_k;
        for (uint candidate = 0; candidate != live_count && standing == kept_count; ++candidate)
            if (health.depth_to_root(stitch.ids[candidate]) < depth_to_root &&
                (way == metal_unknown_k || stitch.distances[candidate] < stitch.distances[way]))
                way = candidate;
        if (way != metal_unknown_k) keep(way);
    }
    return kept_count;
}

/** Re-stitches each neighborhood, one threadgroup a neighborhood claimed from a counter: its rows
 *  and norms stage in threadgroup memory while they fit, or are read in place; each simdgroup takes
 *  an entry, keeps its nearest neighborhood-mates by position in its own slice, both directions of
 *  every pair and never itself, so no atomic merges a pair, then lets its live links, scored
 *  against it, and its mates no live link names compete, ascending. Only the diverse ones stay,
 *  with no refill, then the reach rule, and a row is rewritten only where its link set changed,
 *  slot zero released last, counted once in @c changed. */
[[host_name("metal_restitch_kernel")]] kernel void metal_restitch_kernel(
    constant metal_arguments_t& arguments [[buffer(metal_arguments_buffer_k)]],
    device uint4* vectors [[buffer(metal_vectors_buffer_k)]], device norms_t* norms [[buffer(metal_norms_buffer_k)]],
    device uint* neighbors [[buffer(metal_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* changed [[buffer(metal_changed_buffer_k)]],
    device uint* neighborhood_entries [[buffer(metal_neighborhood_entries_buffer_k)]],
    device uint* neighborhood_offsets [[buffer(metal_neighborhood_offsets_buffer_k)]],
    device uchar* stitch_scratch [[buffer(metal_stitch_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]], uint group_threads [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint4 staged[uint(metal_round_staged_bytes_k) / metal_fragment_bytes_k];
    threadgroup uint claimed;
    metal_health_t health = metal_health(arguments, neighbors, counters, depths_from_root, depths_to_root, in_links,
                                         shallower_in_links, reached, reaches_root, two_ways_to_root);
    uint const fragments = arguments.row_fragments, connectivity = arguments.connectivity, metric = arguments.metric;
    bool const has_norms = arguments.has_norms != 0;
    device uchar* const slice = metal_row(stitch_scratch, group * metal_simdgroups_k + simdgroup,
                                          arguments.slice_bytes);
    metal_stitch_t stitch;
    stitch.sorted = (device ulong*)slice;
    stitch.mates = (device ulong*)(slice + 16 * connectivity);
    stitch.ids = (device uint*)(slice + 24 * connectivity);
    stitch.distances = (device float*)(slice + 32 * connectivity);
    stitch.kept = (device uint*)(slice + 40 * connectivity);
    uint added = 0, dropped = 0, changed_rows = 0;
    for (;;) {
        if (thread_index == 0) claimed = warp_t::fetch_add_relaxed(counters + metal_round_claimed_k, 1u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint const neighborhood = claimed;
        if (neighborhood >= arguments.neighborhoods) break;
        uint const first = neighborhood_offsets[neighborhood], count = neighborhood_offsets[neighborhood + 1] - first;
        device uint* const entries = neighborhood_entries + first;
        bool const staging = count * (fragments * metal_fragment_bytes_k + 8) <= metal_round_staged_bytes_k;
        threadgroup norms_t* const staged_norms = (threadgroup norms_t*)(staged + count * fragments);
        for (uint at = thread_index; staging && at < count * fragments; at += group_threads) {
            device uint4 const* const member_row = metal_row(vectors, entries[at / fragments], fragments);
            staged[at] = member_row[at % fragments];
        }
        for (uint member = thread_index; staging && member < count; member += group_threads)
            staged_norms[member] = has_norms ? norms[entries[member]] : norms_t(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto norm_of = [&](uint id) { return has_norms ? norms[id] : norms_t(0); };
        auto member_norms = [&](uint member) { return staging ? staged_norms[member] : norm_of(entries[member]); };
        auto mate_distance = [&](uint position, uint member) {
            dot_t const dot = staging ? metal_rows_dot(staged + position * fragments, staged + member * fragments,
                                                       fragments)
                                      : metal_rows_dot(metal_row(vectors, entries[position], fragments),
                                                       metal_row(vectors, entries[member], fragments), fragments);
            return kit_t::distance(metric, dot, member_norms(position), member_norms(member));
        };
        for (uint position = simdgroup; position < count; position += metal_simdgroups_k) {
            uint const node_id = entries[position];
            norms_t const node_norms = member_norms(position);

            // The nearest neighborhood-mates, by position, never the entry itself.
            metal_sorted_buffer nearest {stitch.mates, 0, connectivity};
            for (uint base = 0; base < count; base += warp_t::width_k) {
                uint const member = base + lane;
                bool const other = member < count && member != position;
                float const distance = other ? mate_distance(position, member) : 0.0f;
                ulong const scored = other && !isnan(distance) ? metal_make(distance, member) : metal_sentinel_k;
                nearest.insert_batch(metal_sort_lanes(scored, lane),
                                     popcount(warp_t::ballot(scored != metal_sentinel_k)), lane);
            }

            // The live links first, scored against the entry; a self-loop in slot zero means none.
            device uint* const row = health.row(node_id);
            uint const first_value = warp_t::shuffle(lane == 0 ? warp_t::load_acquire(row) : 0u, 0);
            uint live_count = 0;
            for (uint base = 0; base < connectivity; base += warp_t::width_k) {
                uint const slot = base + lane;
                uint const target = first_value == metal_unknown_k || first_value == node_id ? metal_unknown_k
                                    : slot >= connectivity                                   ? metal_unknown_k
                                    : slot ? warp_t::load_relaxed(row + slot)
                                           : first_value;
                bool const usable = target != metal_unknown_k && target != node_id && target < health.rows_count;
                uint const usable_mask = warp_t::ballot(usable);
                if (usable) stitch.ids[live_count + popcount(usable_mask & warp_t::below(lane))] = target;
                live_count += popcount(usable_mask);
                warp_t::sync();
            }
            for (uint candidate = lane; candidate < live_count; candidate += warp_t::width_k) {
                uint const target = stitch.ids[candidate];
                dot_t const dot = staging ? metal_rows_dot(staged + position * fragments,
                                                           metal_row(vectors, target, fragments), fragments)
                                          : metal_rows_dot(metal_row(vectors, node_id, fragments),
                                                           metal_row(vectors, target, fragments), fragments);
                stitch.distances[candidate] = kit_t::distance(metric, dot, node_norms, norm_of(target));
            }
            warp_t::sync();

            // Then the mates no live link names.
            uint candidates = live_count;
            for (uint rank = 0; rank != nearest.size; ++rank) {
                ulong const mate = nearest.at(rank);
                uint const mate_id = entries[metal_id_of(mate)];
                bool held = false;
                for (uint candidate = lane; candidate < live_count; candidate += warp_t::width_k)
                    held = held || stitch.ids[candidate] == mate_id;
                if (simd_any(held)) continue;
                if (lane == 0) stitch.ids[candidates] = mate_id, stitch.distances[candidates] = metal_distance_of(mate);
                ++candidates;
                warp_t::sync();
            }

            // All of them ascending, then the greedy diverse prefix with no refill.
            metal_sorted_buffer sorted {stitch.sorted, 0, 2 * connectivity};
            for (uint base = 0; base < candidates; base += warp_t::width_k) {
                uint const candidate = base + lane;
                ulong const scored = candidate < candidates ? metal_make(stitch.distances[candidate], candidate)
                                                            : metal_sentinel_k;
                sorted.insert_batch(metal_sort_lanes(scored, lane), min(candidates - base, warp_t::width_k), lane);
            }
            uint kept = 0;
            for (uint rank = 0; rank != sorted.size && kept != connectivity; ++rank) {
                uint const candidate = metal_id_of(sorted.at(rank));
                uint const candidate_id = stitch.ids[candidate];
                float const to_node = stitch.distances[candidate];
                bool diverse = true;
                for (uint base = 0; base < kept && diverse; base += warp_t::width_k) {
                    bool lane_diverse = true;
                    if (base + lane < kept) {
                        uint const other_id = stitch.ids[stitch.kept[base + lane]];
                        dot_t const dot = metal_rows_dot(metal_row(vectors, candidate_id, fragments),
                                                         metal_row(vectors, other_id, fragments), fragments);
                        lane_diverse = kit_t::distance(metric, dot, norm_of(candidate_id), norm_of(other_id)) >=
                                       to_node;
                    }
                    diverse = simd_all(lane_diverse);
                }
                if (!diverse) continue;
                if (lane == 0) stitch.kept[kept] = candidate;
                ++kept;
                warp_t::sync();
            }
            kept = warp_t::shuffle(
                lane == 0 ? metal_keep_reach(health, node_id, stitch, live_count, kept, connectivity) : 0u, 0);
            warp_t::sync();

            // Equal link sets write nothing, else the row takes the kept links, slot zero last.
            uint in_live = 0;
            for (uint rank = lane; rank < kept; rank += warp_t::width_k) in_live += stitch.kept[rank] < live_count;
            in_live = warp_t::reduce_add(in_live);
            if (in_live == kept && in_live == live_count) continue;
            for (uint slot = lane; slot < connectivity; slot += warp_t::width_k)
                if (slot)
                    warp_t::store_relaxed(row + slot, slot < kept ? stitch.ids[stitch.kept[slot]] : metal_unknown_k);
            warp_t::fence();
            if (lane != 0) continue;
            warp_t::store_release(row, kept ? stitch.ids[stitch.kept[0]] : node_id);
            added += kept - in_live, dropped += live_count - in_live;
            changed_rows += !metal_bitset_claim(changed, node_id);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane != 0) return;
    metal_tally_add(counters, metal_round_added_k, added);
    metal_tally_add(counters, metal_round_dropped_k, dropped);
    if (changed_rows) warp_t::fetch_add_relaxed(counters + metal_round_changed_k, changed_rows);
}

#pragma endregion Rounds

} // namespace unum::usearch::dense
