/**
 *  @file include/usearch/utilities/ram_walks.hpp
 *  @author Ash Vardanian
 *  @date October 30, 2026
 *  @brief The breadth-first walk every sharded RAM index's cursor shares: the nodes it visited,
 *      one bit each, and the frontier it still has to expand.
 *
 *  A cursor owns one walk by value and drives it: @ref breadth_first_walk::open sizes the bits to
 *  the shards' watermarks as the cursor snapshotted them, @ref breadth_first_walk::visit admits a
 *  node once, and @ref breadth_first_walk::expand hands the frontier back in meeting order. The
 *  walk names no shard type: the cursor seeks its shard cursors and resolves keys itself.
 */
#pragma once

#include <limits> // `std::numeric_limits` - the lanes one word of visited bits carries
#include <span>   // `std::span` - the shards' visible counts at open

#include "usearch/utilities/types.hpp"       // `shard_node`, `u1x64_t`, `divide_round_up`
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the bits and the frontier
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator_t` - where they live

namespace unum::usearch::utilities {

/** How a cursor orders the entries it yields. */
enum class cursor_order_t : u32_t {

    /** Every entry as stored, shard after shard. */
    stored_k,

    /** The start entries first, then the entries they link to, breadth-first. */
    breadth_first_k,
};

/**
 *  @brief A breadth-first walk over the shards of one sharded index: the nodes it visited, one bit
 *      each, and the frontier it still has to expand, in the order it visited them.
 *
 *  The bits span each shard's watermark as it stood at @ref open; a node at or past it is never
 *  visited, so the cursor driving the walk keeps its shard cursors' snapshots unrefreshed.
 */
template <typename node_id_type_>
struct breadth_first_walk {
    using node_id_t = node_id_type_;
    using shard_node_t = shard_node<node_id_type_>;

    /** Lanes one word of visited bits carries. */
    inline static constexpr std::size_t lanes_k = std::numeric_limits<u1x64_t>::digits;

    /** One bit per node of every shard, the shards' runs back to back from @c visited_bases_. */
    growing_buffer<u1x64_t, forkunion_allocator_t> visited_;

    /** The first word of each shard's run in @c visited_, one more than the shards. */
    growing_buffer<std::size_t, forkunion_allocator_t> visited_bases_;

    /** The nodes visited and not yet expanded, in visiting order; @c expanded_ of them are behind
     *  the walk. */
    growing_buffer<shard_node_t, forkunion_allocator_t> frontier_;
    std::size_t expanded_ = 0;

    explicit breadth_first_walk(forkunion_allocator_t allocator) noexcept
        : visited_(allocator), visited_bases_(allocator), frontier_(allocator) {}

    /**
     *  @brief Opens the walk over the shards whose watermarks @p visible_per_shard names, every bit
     *      clear and the frontier empty.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The bits or their bases found no memory.
     */
    status_t open(std::span<std::size_t const> visible_per_shard) noexcept {
        frontier_.clear();
        expanded_ = 0;
        if (failed(visited_bases_.resize(visible_per_shard.size() + 1))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != visible_per_shard.size(); ++shard)
            visited_bases_[shard + 1] = visited_bases_[shard] +
                                        divide_round_up<std::size_t>(visible_per_shard[shard], lanes_k);
        return visited_.resize(visited_bases_.back());
    }

    /**
     *  @brief Marks @p node of @p shard visited and appends it to the frontier; a node visited
     *      before stays where it is.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The frontier could not grow; the node stays unvisited.
     */
    status_t visit(u32_t shard, node_id_t node) noexcept {
        u1x64_t& word = visited_[visited_bases_[shard] + node / lanes_k];
        u1x64_t const bit = u1x64_t {1} << (node % lanes_k);
        if (word & bit) return status_t::success_k;
        if (status_t const queued = frontier_.push_back({shard, node}); failed(queued)) return queued;
        word |= bit;
        return status_t::success_k;
    }

    /** Takes the next unexpanded frontier node into @p next, answering false once none remain. */
    bool expand(shard_node_t& next) noexcept {
        if (expanded_ == frontier_.size()) return false;
        next = frontier_[expanded_++];
        return true;
    }
};

} // namespace unum::usearch::utilities
