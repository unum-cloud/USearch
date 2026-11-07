/**
 *  @file include/usearch/utilities/ram_walks.hpp
 *  @author Ash Vardanian
 *  @date October 30, 2026
 *  @brief The walks every sharded index's cursor shares: the breadth-first one, with the nodes it
 *      visited, one bit each, and the frontier it still has to expand, and the stored order.
 *
 *  A cursor owns one walk by value and drives it: @c breadth_first_walk::open sizes the bits to
 *  the shards' watermarks as the cursor snapshotted them, @c breadth_first_walk::visit admits a
 *  node once, and @c breadth_first_walk::expand hands the frontier back in meeting order.
 *  @c cursor_walk picks between that walk and the stored order. Neither names a shard type: the
 *  cursor seeks its shard cursors and resolves keys itself.
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
        return open(visible_per_shard.size(), [&](std::size_t shard) noexcept { return visible_per_shard[shard]; });
    }

    /** Opens the walk over @p shards_count shards, shard @c s visiting the @p visible_of(s) nodes
     *  below its watermark, as @ref open over their counts does. */
    template <typename visible_of_type_>
    status_t open(std::size_t shards_count, visible_of_type_&& visible_of) noexcept {
        frontier_.clear();
        expanded_ = 0;
        if (failed(visited_bases_.resize(shards_count + 1))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_count; ++shard)
            visited_bases_[shard + 1] = visited_bases_[shard] +
                                        divide_round_up<std::size_t>(visible_of(shard), lanes_k);
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

/**
 *  @brief Where a cursor stands in its walk over the shards of one sharded index: the
 *      breadth-first walk, whose opening picks that order, or the next stored entry.
 *
 *  The cursor names the shards and how to enter one; a stored walk enters each shard once, as it
 *  reaches the shard's first node, so a cursor over live shards refreshes a shard's watermark
 *  then, and the walk yields the nodes below the count entering answered.
 */
template <typename node_id_type_>
struct cursor_walk {
    using node_id_t = node_id_type_;
    using shard_node_t = shard_node<node_id_type_>;

    explicit cursor_walk(forkunion_allocator_t allocator) noexcept : breadth_first_(allocator) {}

    /**
     *  @brief Turns the walk breadth-first over @p shards_count shards, shard @c s visiting the
     *      @p visible_of(s) nodes below its watermark; the starts follow through @ref visit.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The visited bits or their bases found no memory.
     */
    template <typename visible_of_type_>
    status_t open_breadth_first(std::size_t shards_count, visible_of_type_&& visible_of) noexcept {
        return breadth_first_.open(shards_count, visible_of);
    }

    /** Whether the walk runs breadth-first, so every link it meets is a node to visit: opening
     *  sized the visited bits' bases, which a stored walk never does. */
    bool breadth_first() const noexcept { return !breadth_first_.visited_bases_.empty(); }

    /** Visits @p node of @p shard in a breadth-first walk; see @c breadth_first_walk::visit. */
    status_t visit(u32_t shard, node_id_t node) noexcept { return breadth_first_.visit(shard, node); }

    /**
     *  @brief Takes the next entry into @p entry, answering false once the walk ran out.
     *
     *  A stored walk runs over @p shards_count shards, entering each through @p enter, as
     *  <tt>enter(shard) noexcept → nodes</tt>, once, as it reaches it.
     */
    template <typename enter_type_>
    bool next(shard_node_t& entry, std::size_t shards_count, enter_type_&& enter) noexcept {
        if (breadth_first()) return breadth_first_.expand(entry);
        for (; next_stored_.shard != shards_count; next_stored_ = {next_stored_.shard + 1, 0}) {
            // The walk only moves forward, so it stands on a shard's first node once, on arriving.
            if (!next_stored_.node) stored_nodes_ = enter(next_stored_.shard);
            if (next_stored_.node == stored_nodes_) continue;
            entry = next_stored_;
            ++next_stored_.node;
            return true;
        }
        return false;
    }

  private:
    /** The nodes visited and the frontier of a breadth-first walk; untouched by a stored one. */
    breadth_first_walk<node_id_t> breadth_first_;

    /** As stored: the entry the next step takes, and the nodes its shard held when entered. */
    shard_node_t next_stored_ {};
    std::size_t stored_nodes_ = 0;
};

} // namespace unum::usearch::utilities
