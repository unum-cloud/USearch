/**
 *  @file include/usearch/dense/sharded_indexes.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Same-geometry dense shards viewed as one: the component every dense tower owns, holding
 *      the shards, the keys of their rows, the rotation that routes new rows, and the commit word
 *      a reader spanning the shards snapshots by.
 *
 *  A tower keeps its own verbs and its own residency's logic, and reaches its shards through this
 *  set: placement and reservation, the key locators, the claim words closed and reopened in shard
 *  order, and the snapshot of every shard's watermark at one instant no commit straddles.
 */
#pragma once

#include <memory>  // `std::unique_ptr` - an index holds atomics and never moves
#include <span>    // `std::span` - the watermarks a snapshot fills
#include <utility> // `std::move`, `std::swap`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer`, `chunked_column` - the shards & their key columns
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator` - the shard table's placement
#include "usearch/utilities/ram_tables.hpp"  // `chunked_multimap` - the locator map
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

/**
 *  @brief Same-geometry shards viewed as one, the keys of their rows, and the words that keep a
 *      reader spanning them consistent.
 *
 *  It owns its shards outright, so a tower's destructor, consuming merge and clear are
 *  expressible. Shards arrive one at a time through @ref emplace, so a tower grows as its
 *  placements are engaged rather than being declared whole.
 *
 *  @tparam index_type_ The shard type.
 *  @tparam key_type_ The caller-visible identity, opaque to every shard.
 *  @tparam key_hash_type_ The locator map's hash of a key.
 *  @tparam atomic_ref_template_ The atomic reference owning the plain synchronization words.
 *  @tparam wait_type_ What a snapshot waits through while a commit is in flight.
 */
template <typename index_type_, typename key_type_, typename key_hash_type_,
          template <typename> class atomic_ref_template_, typename wait_type_>
struct shard_set {

    /** The shard type - geometry parameters in the shard's own order, so the two never disagree. */
    using index_t = index_type_;

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;

    /** Where one entry lives: the shard, and its shard-local ID - a tower's only address for an
     *  entry. */
    struct key_locator_t {

        /** The shard holding the entry, positional in placement order. */
        std::size_t shard = 0;

        /** The entry's ID inside that shard's own space. */
        node_id_t node = 0;
    };

    /** The atomic reference owning the plain synchronization words - the shards' and the set's
     *  own routing tally alike. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Key to every entry stored under it; insert-only, wait-free for readers, growing in
     *  chunks. */
    using locators_map_t =
        chunked_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t, atomic_ref_template_>;

    /** Keys of one shard's rows, indexed by that shard's own node IDs; host-resident, growing in
     *  the chunks the CPU shards grow in. */
    using keys_column_t = chunked_column<key_t, chunk_schedule<10, 20>, index_t::max_node_count_k,
                                         forkunion_allocator_t, atomic_ref_template_>;

    using reserve_capacity_t = typename index_t::reserve_capacity_t;

    /** One shard: the graph, and the keys of its rows. Placement is not repeated: the shard
     *  carries its own. */
    struct shard_t {

        /** The graph, growing on demand; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;

        /** The key of each of this shard's rows, indexed by node ID; every claimer covers the rows
         *  it may write before writing them. */
        keys_column_t keys;
    };

    /** An empty set of shards scoring by @p kernel, its shard table, locator map and checkpoint
     *  catalogs on @p allocator. */
    explicit shard_set(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : shards_(allocator), locators_of_key_(allocator), allocator_(allocator), kernel_(kernel) {}
    shard_set(shard_set&& other) noexcept : shard_set(other.kernel_, other.allocator_) { swap(other); }
    shard_set& operator=(shard_set&& other) noexcept { return swap(other), *this; }
    shard_set(shard_set const&) = delete;
    shard_set& operator=(shard_set const&) = delete;

    /** Exchanges two sets whole; a move implies the caller holds both exclusively. */
    void swap(shard_set& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::swap(kernel_, other.kernel_);
        std::size_t const routed_here = routed();
        reset_routing(other.routed());
        other.reset_routing(routed_here);
    }

#pragma region Shards

    /** Shards in the set, in placement order. */
    std::size_t size() const noexcept { return shards_.size(); }
    bool empty() const noexcept { return shards_.empty(); }

    shard_t& operator[](std::size_t shard) noexcept { return shards_[shard]; }
    shard_t const& operator[](std::size_t shard) const noexcept { return shards_[shard]; }
    shard_t* begin() noexcept { return shards_.data(); }
    shard_t* end() noexcept { return shards_.data() + shards_.size(); }
    shard_t const* begin() const noexcept { return shards_.data(); }
    shard_t const* end() const noexcept { return shards_.data() + shards_.size(); }

    /** The metric every shard scores by and every checkpoint names, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** The host allocator behind the shard table, the locator map and every checkpoint catalog;
     *  the shards carry their own. */
    forkunion_allocator_t allocator() const noexcept { return allocator_; }

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->capacity();
        return total;
    }

    /** Bytes every shard's storage, its key column and the shared locator map occupy. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = locators_of_key_.memory_usage();
        for (shard_t const& one : shards_) bytes += one.index->memory_usage() + one.keys.memory_usage();
        return bytes;
    }

    /** Appends one shard the caller already built, its key column on @p keys_allocator; refuses
     *  one walking by another kernel, one whose frozen geometry differs from the shards already
     *  here, and one already reserved. Several shards may share a placement, each a slice of the
     *  graph rather than a copy. @ref reserve closes the set. */
    status_t emplace(std::unique_ptr<index_t>&& shard, forkunion_allocator_t keys_allocator) noexcept {
        if (!shard || shard->kernel() != kernel_) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (shard->capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        return shards_.push_back(shard_t {std::move(shard), keys_column_t {keys_allocator}});
    }

    /**
     *  @brief Grows every shard ahead of @p config.capacity rows, split evenly over them - a hint,
     *      since the shards, their key columns and the locator map all grow on demand; the first
     *      call freezes the geometry.
     *
     *  Callable alongside insertions and searches on the host residency: nothing it grows ever
     *  moves.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (shards_.empty()) return {0, status_t::uninitialized_k};
        reserve_capacity_t per_shard = config;
        per_shard.capacity = divide_round_up(config.capacity, shards_.size());
        for (shard_t& one : shards_) {
            auto const [shard_capacity, status] = one.index->reserve(per_shard);
            if (failed(status)) return {capacity(), status};
            if (status_t const covered = one.keys.cover(shard_capacity); failed(covered)) return {capacity(), covered};
        }
        if (status_t const indexed = locators_of_key_.reserve(config.capacity); failed(indexed))
            return {capacity(), indexed};
        return {capacity(), status_t::success_k};
    }

    /** Closes every shard's claim word in shard order, or none: a shard with a batch or load in
     *  flight reopens the ones before it and answers @c contended_k. A batch claims in the same
     *  order, so one refused by a closed shard has claimed nothing on the shards before it. */
    status_t close() noexcept {
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (status_t const closed = shards_[shard].index->close_(); failed(closed)) {
                while (shard) shards_[--shard].index->reopen_();
                return closed;
            }
        return status_t::success_k;
    }

    /** Reopens every closed shard's claim word at its watermark, the last shard first, so a batch
     *  taking the shards in order finds shard zero closed until every later one reopened. */
    void reopen() noexcept {
        for (std::size_t shard = shards_.size(); shard;) shards_[--shard].index->reopen_();
    }

#pragma endregion Shards

#pragma region Keys and Routing

    /** Every key's entries. Duplicates coexist as separate slots, which is what several values
     *  under one key means, and costs one slot per row rather than per key. */
    locators_map_t& locators() noexcept { return locators_of_key_; }
    locators_map_t const& locators() const noexcept { return locators_of_key_; }

    /** Drops every key's locators and restarts the routing, once every shard emptied. */
    void forget_rows() noexcept {
        locators_of_key_.clear();
        reset_routing(0);
    }

    /** Routes @p rows more rows, answering the rows routed before them - the next batch's rotation
     *  base, so a stream of small batches never drifts onto shard zero. */
    std::size_t route(std::size_t rows) noexcept {
        return atomic_ref<std::size_t>(routed_rows_).fetch_add(rows, std::memory_order_relaxed);
    }

    /** Rows routed so far - what traversal contexts seed apart by, and apart across calls. */
    std::size_t routed() const noexcept {
        return atomic_ref<std::size_t const>(routed_rows_).load(std::memory_order_relaxed);
    }

    /** Restarts the routing as if @p rows rows had been routed, as a restored checkpoint does. */
    void reset_routing(std::size_t rows) noexcept {
        atomic_ref<std::size_t>(routed_rows_).store(rows, std::memory_order_relaxed);
    }

    /** The first row of a batch that @p shard takes under @p rotation - a bijection over the
     *  shards, so the residue classes it names partition the batch exactly. */
    static std::size_t first_row_of_shard(std::size_t shard, std::size_t shards, std::size_t rotation) noexcept {
        return (shard + shards - rotation) % shards;
    }

    /** How many of @p rows a shard starting at @p first takes, stepping by @p shards. */
    static std::size_t rows_from(std::size_t rows, std::size_t first, std::size_t shards) noexcept {
        return first < rows ? divide_round_up(rows - first, shards) : 0;
    }

    /** Writes into @p part_bases, one run per shard as @p part_offsets cuts it, a part for every
     *  non-empty shard of @p donor its load @p plans appended, after the resident part and any
     *  earlier donor's; a shard empty beforehand keeps the first part landing in it as its own. */
    template <typename plans_type_>
    void record_parts(shard_set const& donor, plans_type_ const& plans, node_id_t* part_bases,
                      std::span<std::size_t const> part_offsets) const noexcept {
        std::size_t const shards = shards_.size(), donors = donor.shards_.size();
        for (std::size_t into = 0; into != shards; ++into) {
            node_id_t* const parts = part_bases + part_offsets[into];
            std::size_t const room = part_offsets[into + 1] - part_offsets[into];
            std::size_t count = 0;
            while (count != room && parts[count] != node_id_unknown_k) ++count;
            node_id_t base = plans[into].base;
            for (std::size_t from = into; from < donors; from += shards) {
                node_id_t const rows = static_cast<node_id_t>(donor.shards_[from].index->size());
                if (!rows) continue;
                if (!count && base) parts[count++] = 0;
                parts[count++] = base;
                base += rows;
            }
        }
    }

#pragma endregion Keys and Routing

#pragma region Commits and Snapshots

    /** Opens a commit spanning several shards: its watermarks may now move one shard at a time. */
    void begin_commit() noexcept { atomic_ref<u64_t>(commit_sequence_).add(1, std::memory_order_release); }

    /** Closes the commit @ref begin_commit opened; closing carries into the high half, so a reader
     *  sees the word move even when two commits overlap. */
    void end_commit() noexcept {
        atomic_ref<u64_t>(commit_sequence_).add((u64_t {1} << 32) - 1, std::memory_order_release);
    }

    /** Takes every shard's watermark into @p watermarks, unless it is empty, at one instant no
     *  commit spanning shards publishes through, so a walk of each shard below its own sees every
     *  batch whole or not at all; answers their sum. */
    std::size_t snapshot(std::span<node_id_t> watermarks) const noexcept {
        atomic_ref<u64_t const> const sequence {commit_sequence_};
        while (true) {
            u64_t const before = sequence.load(std::memory_order_acquire);
            // A set low half is a commit between its shards' stores.
            if (static_cast<u32_t>(before)) {
                wait_type_ {}(&commit_sequence_, before, 0u);
                continue;
            }
            std::size_t total = 0;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                std::size_t const visible = shards_[shard].index->size();
                if (!watermarks.empty()) watermarks[shard] = static_cast<node_id_t>(visible);
                total += visible;
            }
            // The watermarks' acquires keep this re-read after them.
            if (sequence.load(std::memory_order_relaxed) == before) return total;
        }
    }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t rows() const noexcept { return snapshot({}); }

#pragma endregion Commits and Snapshots

  private:

    /** The owned shards, graph and keys together. A shard's key column grows only ahead of the rows
     *  a search can read, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;

    /** Every key's entries, as @ref locators exposes them. */
    locators_map_t locators_of_key_;

    /** The host allocator behind the shard table, the locator map and every checkpoint catalog. */
    forkunion_allocator_t allocator_;

    /** The metric every shard is built with and every checkpoint names. */
    nk_kernel_kind_t kernel_;

    /** Rows routed so far - the next batch's rotation base. */
    std::size_t routed_rows_ = 0;

    /** Commits in flight in the low half, commits landed in the high half: a reader spanning the
     *  shards retries its snapshot while one is in flight or one landed under it. Every search
     *  polls it, so it keeps off the line every insertion bumps. */
    alignas(default_alignment_k) u64_t commit_sequence_ = 0;
};

} // namespace unum::usearch::dense
