/**
 *  @brief The core tier's proof: the flat headers below the modalities compile with the
 *      standard library alone - no submodule include paths, no third-party dependencies.
 *  @author Ash Vardanian
 *  @file test/core.cpp
 *  @date July 23, 2026
 *
 *  This translation unit builds with only @c -Iinclude, so any dependency creeping into
 *  @c types.hpp, @c ram_queue.hpp, or @c utilities/ram_hints.hpp fails loudly in CI instead
 *  of silently widening the core tier. The checks are deliberately minimal - instantiate the
 *  traversal structures, push & pop a few entries - existence is the assertion.
 */
#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_queue.hpp"
#include "usearch/utilities/ram_hints.hpp"

using namespace unum::usearch;

int main() {
    max_heap<dense_scored_node_t, dense_further_scored_node_t> frontier;
    if (!frontier.reserve(64)) return 1;
    frontier.insert({dense_node_id_t {1}, dense_distance_t {0.5f}});
    frontier.insert({dense_node_id_t {2}, dense_distance_t {0.25f}});
    if (frontier.top().node_id != 2) return 1; // The furthest-first comparator surfaces the nearest.
    frontier.pop();

    sorted_buffer<dense_scored_node_t, dense_closer_scored_node_t> results;
    if (!results.reserve(4)) return 1;
    results.insert({dense_node_id_t {3}, dense_distance_t {0.75f}}, 4);
    if (results.top().node_id != 3) return 1;

    growing_hash_set<dense_node_id_t> visited;
    if (!visited.reserve(64)) return 1;
    visited.set(42);
    if (!visited.test(42) || visited.test(43)) return 1;

    splitmix_u32_t random_generator {7};
    [[maybe_unused]] u32_t const bounded = random_generator.bounded(10);

    prefetch_query_sse_t const prefetch;
    prefetch(&random_generator);
    return 0;
}
