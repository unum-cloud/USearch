/**
 *  @brief The type-punning facade over any (modality, residency) shards - the single
 *      dynamically-typed surface every language binding consumes.
 *  @author Ash Vardanian
 *  @file include/usearch/router.hpp
 *  @date July 23, 2026
 *
 *  The router sits above the modality, residency, and capability axes rather than on
 *  them: it owns shards of any one (modality, residency) coordinate, merges their results
 *  through @ref merge_search_results, and exposes one dynamically-typed surface keyed by
 *  runtime (dtype, kernel) pairs. Bindings link the router's pre-instantiated library -
 *  the @c cpp/dense_indexes.cpp trampoline and its per-dtype shards - and never
 *  instantiate index templates themselves, so a binding rebuild never touches the
 *  instantiation grids.
 */
#pragma once

#include "usearch/types.hpp"

namespace unum {
namespace usearch {

/** @brief Owns type-punned shards of one (modality, residency) coordinate and merges
 *      their rankings; defined beside the pre-instantiated shards it dispatches onto. */
struct router_t;

} // namespace usearch
} // namespace unum
