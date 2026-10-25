/**
 *  @file include/usearch/utilities/ram_shards.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Shards viewed as one: the component every plural tower owns, holding the shards, the keys
 *      of their rows, the rotation that routes new rows, and the commit word a reader spanning the
 *      shards snapshots by.
 *
 *  A tower keeps its own verbs and its own residency's logic, and reaches its shards through this
 *  set: placement and reservation, the key locators, the claim words closed and reopened in shard
 *  order, the snapshot of every shard's watermark at one instant no commit straddles, and the
 *  epoch its readers bracket themselves with against the tower's destructive windows. The tower
 *  builds and reserves every shard and checks its shape, so the set never reads one.
 */
#pragma once

#include <array>       // `std::array` - the shard table's chunks
#include <atomic>      // `std::atomic_thread_fence` - the epoch's reader bracket
#include <bit>         // `std::bit_width` - a shard's chunk
#include <limits>      // `std::numeric_limits` - the chunk count & the opening bit
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <new>         // placement `new` - a shard lands in its chunk
#include <span>        // `std::span` - the watermarks a snapshot fills
#include <type_traits> // `std::invoke_result_t`, `std::is_same_v` - the bracketed readers' answers
#include <utility>     // `std::move`, `std::swap`, `std::exchange`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `chunked_column` - the shards' key columns
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator` - the shard table's placement
#include "usearch/utilities/ram_tables.hpp"  // `chunked_multimap` - the locator map

namespace unum::usearch::utilities {

/** Keys of one shard's rows, indexed by that shard's own node IDs; host-resident, growing in the
 *  chunks the CPU shards grow in. A tower's per-shard key storage holds one. */
template <typename index_type_, typename key_type_, template <typename> class atomic_ref_template_>
using shard_keys_column = chunked_column<key_type_, chunk_schedule<10, 20>, index_type_::max_node_count_k,
                                         forkunion_allocator_t, atomic_ref_template_>;

/** The host key column of a shard whose key storage is that column alone. */
template <typename key_type_, typename schedule_type_, std::size_t max_rows_, typename allocator_type_,
          template <typename> class atomic_ref_template_>
chunked_column<key_type_, schedule_type_, max_rows_, allocator_type_, atomic_ref_template_>& host_keys(
    chunked_column<key_type_, schedule_type_, max_rows_, allocator_type_, atomic_ref_template_>& keys) noexcept {
    return keys;
}

/** Where one source shard's rows land: shard @c from of the sources' concatenated shard lists,
 *  into shard @c into of the target. Sixteen bytes. */
struct landing_t {
    std::size_t from = 0;
    std::size_t into = 0;
};

static_assert(sizeof(landing_t) == 16);

/** Where one row of a split batch lands: the shard, counted from the open generation's first, and
 *  its position among the rows that shard takes. Sixteen bytes. */
struct row_in_shard_t {
    std::size_t shard = 0;
    std::size_t position_in_shard = 0;
};

static_assert(sizeof(row_in_shard_t) == 16);

/** Whether a target shard shares the source shard's placement, which every rank prefers. */
enum class placement_match_t : bool { other_k, same_k };

/** Ranks a target shard with room for a compaction, the lowest first: any on the source shard's
 *  placement before any other, the earliest on ties, so each shard fills before the next takes. */
inline std::size_t rank_first_fit(std::size_t, placement_match_t match) noexcept {
    return match == placement_match_t::same_k ? 0 : 1;
}

/** Ranks a target shard with room for an absorb, the lowest first: the fewest rows, then one on the
 *  source shard's placement, so the rows spread over the shards as they stand. */
inline std::size_t rank_fewest_rows(std::size_t rows, placement_match_t match) noexcept {
    return 2 * rows + (match == placement_match_t::same_k ? 0 : 1);
}

/**
 *  @brief Shards viewed as one, the keys of their rows, and the words that keep a reader spanning
 *      them consistent.
 *
 *  It owns its shards outright, so a tower's destructor, consuming merge and clear are expressible.
 *  Shards arrive one at a time through @ref emplace, so a tower grows as its placements are engaged
 *  rather than being declared whole. Those placements are the first generation; once its shards
 *  reach @ref shard_capacity, @ref open_generation_ appends another shard per placement, and only
 *  the newest generation takes insertions. A tower whose readers may hold a shard's values retires
 *  its shards rather than emptying them, through @ref retire_older_generations_:
 *  they stay in place, closed and empty to every reader, until the set dies. The set reads no
 *  shard's shape: the tower checks it on @ref emplace and reserves every shard a generation opens
 *  with.
 *
 *  @tparam index_type_ The shard type.
 *  @tparam shard_keys_type_ One shard's key storage, the tower's own: a @ref shard_keys_column
 *      alone, or one beside the device twins it feeds; @ref host_keys reaches its host column.
 *  @tparam key_type_ The caller-visible identity, opaque to every shard.
 *  @tparam key_hash_type_ The locator map's hash of a key.
 *  @tparam atomic_ref_template_ The atomic reference owning the plain synchronization words.
 *  @tparam wait_type_ What a snapshot waits through while a commit is in flight.
 */
template <typename index_type_, typename shard_keys_type_, typename key_type_, typename key_hash_type_,
          template <typename> class atomic_ref_template_, typename wait_type_>
struct shard_set {

    /** The shard type, built and shaped by the tower. */
    using index_t = index_type_;

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;

    /** Where one entry lives: the shard, and its shard-local ID - a tower's only address for an
     *  entry. Eight bytes on an eight-byte boundary, so a locator slot is sixteen with its key and
     *  one atomic store swaps it whole. */
    struct alignas(sizeof(u64_t)) key_locator_t {

        /** The shard holding the entry, positional in placement order. */
        u32_t shard = 0;

        /** The entry's ID inside that shard's own space. */
        node_id_t node = 0;

        bool operator==(key_locator_t const&) const noexcept = default;
    };
    static_assert(sizeof(key_locator_t) == 8);
    static_assert(std::is_same_v<typename index_t::node_id_t, node_id_t>,
                  "A locator names a row in 32 bits, so plural shards hold u32 nodes");

    /** The atomic reference owning the plain synchronization words - the shards' and the set's
     *  own routing tally alike. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Key to every entry stored under it; insert-only, wait-free for readers, growing in
     *  chunks. */
    using locators_map_t =
        chunked_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t, atomic_ref_template_>;

    using reserve_capacity_t = typename index_t::reserve_capacity_t;

    /** One shard: the graph, and the keys of its rows. Placement is not repeated: the shard
     *  carries its own. */
    struct shard_t {

        /** The graph, growing on demand; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;

        /** The key of each of this shard's rows, indexed by node ID in its host column; every
         *  claimer covers the rows it may write before writing them. */
        shard_keys_type_ keys;
    };

    /** One generation of shards: @c width shards from @c first, one per placement of the first. */
    struct generation_t {
        std::size_t first = 0;
        std::size_t width = 0;
    };

    /** One batch split across the open generation and the fresh ones it opened: rows are split
     *  across the shards round-robin from the starting shard on, the open generation taking the
     *  rows below @c rows_in_open_generation and every fresh one as many as the bound lets its
     *  shards hold. */
    struct batch_split_t {

        /** The generation open when the batch was split, the first the batch lands in. */
        generation_t open_generation {};

        /** Rows in the batch. */
        std::size_t batch_rows = 0;

        /** The shard of each generation that row zero of its share lands on. */
        std::size_t starting_shard = 0;

        /** Rows that land in the open generation; the rest land in fresh ones. */
        std::size_t rows_in_open_generation = 0;

        /** Rows a fresh generation takes before the next one opens: its width times the bound. */
        std::size_t rows_per_fresh_generation = 0;

        /** Fresh generations the batch fills past the open one. */
        std::size_t fresh_generations() const noexcept {
            return divide_round_up(batch_rows - rows_in_open_generation, rows_per_fresh_generation);
        }

        /** Shards the batch spans from the open generation's first, in placement order. */
        std::size_t shards_spanned() const noexcept { return open_generation.width * (1 + fresh_generations()); }

        /** Rows shard @p shard takes, counted from the open generation's first. */
        std::size_t rows_for_shard(std::size_t shard) const noexcept {
            std::size_t const width = open_generation.width;
            std::size_t const generation = shard / width, offset = shard % width;
            std::size_t const share = !generation ? rows_in_open_generation
                                                  : min_of(rows_per_fresh_generation,
                                                           batch_rows - rows_in_open_generation -
                                                               (generation - 1) * rows_per_fresh_generation);
            return rows_from(share, first_row_of_shard(offset, width, starting_shard), width);
        }

        /** Where row @p row of the batch lands: its shard from the open generation's first, and its
         *  position among that shard's rows. */
        row_in_shard_t shard_and_position_of_row(std::size_t row) const noexcept {
            std::size_t const width = open_generation.width;
            std::size_t generation = 0, row_in_generation = row;
            if (row >= rows_in_open_generation) {
                generation = 1 + (row - rows_in_open_generation) / rows_per_fresh_generation;
                row_in_generation = (row - rows_in_open_generation) % rows_per_fresh_generation;
            }
            return {generation * width + (starting_shard + row_in_generation) % width, row_in_generation / width};
        }
    };

    /** Walks the shards in placement order; the chunks never move, so a shard it reached keeps its
     *  address while a generation opens. */
    template <typename shard_type_>
    struct shard_iterator {
        shard_set const* set = nullptr;
        std::size_t shard = 0;

        shard_type_& operator*() const noexcept { return set->at_(shard); }
        shard_type_* operator->() const noexcept { return &set->at_(shard); }
        shard_iterator& operator++() noexcept { return ++shard, *this; }
        bool operator==(shard_iterator const& other) const noexcept { return shard == other.shard; }
    };

    /** An empty set of shards, its shard table, locator map and checkpoint catalogs on
     *  @p allocator. */
    explicit shard_set(forkunion_allocator_t allocator) noexcept : locators_of_key_(allocator), allocator_(allocator) {}
    shard_set(shard_set&& other) noexcept : shard_set(other.allocator_) { swap(other); }
    shard_set& operator=(shard_set&& other) noexcept { return swap(other), *this; }
    shard_set(shard_set const&) = delete;
    shard_set& operator=(shard_set const&) = delete;

    /** Destroys every shard, idle and retired ones included, and frees the chunks they sat in. */
    ~shard_set() noexcept {
        for (std::size_t shard = 0; shard != built_shards_; ++shard) at_(shard).~shard_t();
        for (std::size_t chunk = 0; chunk != shard_chunks_.size(); ++chunk)
            if (shard_t* const chunk_shards = std::exchange(shard_chunks_[chunk], nullptr))
                forkunion_allocator<shard_t> {allocator_}.deallocate(chunk_shards, std::size_t {1} << chunk);
    }

    /** Exchanges two sets whole; a move implies the caller holds both exclusively. */
    void swap(shard_set& other) noexcept {
        std::swap(shard_chunks_, other.shard_chunks_);
        std::swap(shards_count_, other.shards_count_);
        std::swap(built_shards_, other.built_shards_);
        std::swap(retired_shards_, other.retired_shards_);
        std::swap(generation_width_, other.generation_width_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::swap(shard_capacity_, other.shard_capacity_);
        std::size_t const routed_here = routed();
        reset_routing(other.routed());
        other.reset_routing(routed_here);
    }

#pragma region Shards

    /** Shards published in the set, in placement order, every generation's and the retired ones;
     *  read with acquire, so every shard it counts is whole. */
    std::size_t size() const noexcept {
        return atomic_ref<std::size_t const>(shards_count_).load(std::memory_order_acquire) & ~(opening_k | sealed_k);
    }
    bool empty() const noexcept { return !size(); }

    /** Shards @ref retire_older_generations_ retired, the first ones in placement order: each
     *  counts no rows and stays closed for good. Iteration starts past them, while positions keep
     *  counting them. */
    std::size_t retired_shards() const noexcept {
        return atomic_ref<std::size_t const>(retired_shards_).load(std::memory_order_acquire);
    }

    /** Whether @ref close sealed the set: no claim may start and no generation may open but the
     *  closer's own, until @ref reopen. Stable for a caller holding the tower's shared side. */
    bool sealed() const noexcept {
        return atomic_ref<std::size_t const>(shards_count_).load(std::memory_order_acquire) & sealed_k;
    }

    shard_t& operator[](std::size_t shard) noexcept { return at_(shard); }
    shard_t const& operator[](std::size_t shard) const noexcept { return at_(shard); }
    shard_iterator<shard_t> begin() noexcept { return {this, retired_shards()}; }
    shard_iterator<shard_t> end() noexcept { return {this, size()}; }
    shard_iterator<shard_t const> begin() const noexcept { return {this, retired_shards()}; }
    shard_iterator<shard_t const> end() const noexcept { return {this, size()}; }

    /** Walks the indexes of the shards from @c first to @c bound in placement order. */
    template <typename index_reference_type_>
    struct indexes_range {
        struct iterator {
            shard_set const* set = nullptr;
            std::size_t shard = 0;

            index_reference_type_& operator*() const noexcept { return *set->at_(shard).index; }
            iterator& operator++() noexcept { return ++shard, *this; }
            bool operator==(iterator const& other) const noexcept { return shard == other.shard; }
        };

        shard_set const* set = nullptr;
        std::size_t first = 0;
        std::size_t bound = 0;

        iterator begin() const noexcept { return {set, first}; }
        iterator end() const noexcept { return {set, bound}; }
    };

    /** The indexes of the shards past the retired ones, in placement order. */
    indexes_range<index_t> live_indexes() noexcept { return {this, retired_shards(), size()}; }
    indexes_range<index_t const> live_indexes() const noexcept { return {this, retired_shards(), size()}; }

    /** The host allocator behind the shard table, the locator map and every checkpoint catalog;
     *  the shards carry their own. */
    forkunion_allocator_t allocator() const noexcept { return allocator_; }

    /** Rows the shards past the retired ones host without growing again; they grow past it on
     *  demand. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : *this) total += one.index->capacity();
        return total;
    }

    /** Bytes every shard's storage, its key storage and the shared locator map occupy, idle and
     *  retired shards included. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = locators_of_key_.memory_usage();
        std::size_t const built = atomic_ref<std::size_t const>(built_shards_).load(std::memory_order_acquire);
        for (std::size_t shard = 0; shard != built; ++shard)
            bytes += at_(shard).index->memory_usage() + at_(shard).keys.memory_usage();
        return bytes;
    }

    /** Bounds every shard at @p rows, clamped to the ID space; zero means the ID space. Fixed once
     *  storage exists, as the placement is. */
    void bound_shards(std::size_t rows) noexcept {
        if (capacity() || retired_shards()) return;
        shard_capacity_ = rows && rows < index_t::max_node_count_k ? rows : index_t::max_node_count_k;
    }

    /** Rows one shard may hold. */
    std::size_t shard_capacity() const noexcept { return shard_capacity_; }

    /** Appends one shard the caller already built beside its empty key storage @p keys; refuses a
     *  null one and one already reserved, the tower having checked its shape. Several shards may
     *  share a placement, each a slice of the graph rather than a copy. @ref reserve closes the
     *  set. */
    status_t emplace(std::unique_ptr<index_t>&& shard, shard_keys_type_&& keys) noexcept {
        if (!shard || shard->capacity()) return status_t::config_mismatch_k;
        if (capacity() || retired_shards() || built_shards_ != size()) return status_t::config_mismatch_k;
        std::size_t const shards = size();
        if (status_t const placed = construct_(shards, shard_t {std::move(shard), std::move(keys)}); failed(placed))
            return placed;
        atomic_ref<std::size_t>(built_shards_).store(shards + 1, std::memory_order_release);
        atomic_ref<std::size_t>(shards_count_).store(shards + 1, std::memory_order_release);
        generation_width_ = shards + 1;
        return status_t::success_k;
    }

    /**
     *  @brief Grows every shard past the retired ones and its key column ahead of
     *      @p config.capacity rows, split evenly over them - a hint, since the shards, their key
     *      columns and the locator map all grow on demand; the first call freezes the geometry.
     *
     *  Callable alongside insertions and searches on the host residency: nothing it grows ever
     *  moves. The locator map grows as rows claim their keys, so a set nobody looks keys up in
     *  never allocates one.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (empty()) return {0, status_t::uninitialized_k};
        reserve_capacity_t per_shard = config;
        per_shard.capacity = min_of(divide_round_up(config.capacity, size() - retired_shards()), shard_capacity_);
        per_shard.ceiling = shard_capacity_;
        for (shard_t& one : *this) {
            auto const [shard_capacity, status] = one.index->reserve(per_shard);
            if (failed(status)) return {capacity(), status};
            if (status_t const covered = host_keys(one.keys).cover(shard_capacity); failed(covered))
                return {capacity(), covered};
        }
        return {capacity(), status_t::success_k};
    }

    /**
     *  @brief Closes every shard's claim word from @p first on in shard order, or none, and seals
     *      the set: a shard with a batch or load in flight, or holding claims it never dispensed,
     *      reopens the ones before it and answers @c contended_k. Retired shards are closed
     *      already, and stay so.
     *
     *  The caller holds the tower's unique side, so no claim starts or moves meanwhile. A batch
     *  claims in the same order, so one refused by a closed shard has claimed nothing on the shards
     *  before it.
     */
    status_t close(std::size_t first = 0) noexcept {
        std::size_t const live = max_of(first, retired_shards());
        for (std::size_t shard = live, shards = size(); shard != shards; ++shard) {
            index_t& one = *at_(shard).index;
            status_t closed = one.close_();
            // A dense RAM shard closes only its dispenser, so undispensed claims refuse here.
            if (succeeded(closed) && one.claimed_() != one.size()) one.reopen_(), closed = status_t::contended_k;
            if (failed(closed)) {
                while (shard != live) at_(--shard).index->reopen_();
                return closed;
            }
        }
        // The unique side orders the seal for every later claimer, which takes the shared side.
        atomic_ref<std::size_t>(shards_count_).fetch_or(sealed_k, std::memory_order_relaxed);
        return status_t::success_k;
    }

    /** Reopens every closed shard's claim word at its watermark, the last shard first, so a batch
     *  taking the shards in order finds the first one closed until every later one reopened, then
     *  lifts the seal. Idle shards past the count and retired ones before it stay closed. */
    void reopen() noexcept {
        for (std::size_t shard = size(), live = retired_shards(); shard != live;) at_(--shard).index->reopen_();
        atomic_ref<std::size_t>(shards_count_).fetch_and(~sealed_k, std::memory_order_release);
    }

#pragma endregion Shards

#pragma region Generations

    /** The newest generation, the only one insertions land in; the older ones are sealed. */
    generation_t open_generation() const noexcept {
        std::size_t const shards = size();
        return {shards - generation_width_, generation_width_};
    }

    /** Rows a batch split over @p generation from @p starting_shard lands before any of its shards
     *  passes the bound: the shard taking rows from @c f on, with @c l rows of room left, takes up
     *  to row @c f + l * width. */
    std::size_t room(generation_t generation, std::size_t starting_shard) const noexcept {
        std::size_t rows = std::numeric_limits<std::size_t>::max();
        for (std::size_t offset = 0; offset != generation.width; ++offset) {
            std::size_t const claimed = at_(generation.first + offset).index->claimed_();
            std::size_t const left = claimed < shard_capacity_ ? shard_capacity_ - claimed : 0;
            rows = min_of(rows, first_row_of_shard(offset, generation.width, starting_shard) + left * generation.width);
        }
        return rows;
    }

    /**
     *  @brief Splits a batch of @p rows across the open generation's shards, opening through
     *      @p build as many fresh generations as the rows past the bound fill, as
     *      @ref open_generation_ does.
     *
     *  The starting shard comes from the rows routed so far, which keeps a stream of small batches
     *  from drifting onto shard zero; a racing batch taking the same room first leaves this one's
     *  claims refused with @c overflow_k, and a failed generation answers its status.
     */
    template <typename build_type_>
    expected<batch_split_t> split_batch_into_shards(std::size_t rows, build_type_&& build) noexcept {
        batch_split_t split;
        split.open_generation = open_generation();
        std::size_t const width = split.open_generation.width;
        split.batch_rows = rows;
        split.starting_shard = route(rows) % width;
        split.rows_in_open_generation = min_of(rows, room(split.open_generation, split.starting_shard));
        split.rows_per_fresh_generation = width * shard_capacity_;
        for (std::size_t opened = 0, fresh = split.fresh_generations(); opened != fresh; ++opened)
            if (status_t const generated = open_generation_({split.open_generation.first + opened * width, width},
                                                            build);
                failed(generated))
                return {split, generated};
        return {split, status_t::success_k};
    }

    /**
     *  @brief Opens the generation after @p full: one shard per placement, built and reserved
     *      under the bound by @p build from that placement's shard of @p full, or the idle shards
     *      a @ref forget_rows kept there, all published at once.
     *
     *  One caller opens it while the others wait on the count word, and a caller finding it opened
     *  already answers success untouched; a sealed set answers @c contended_k, since only its
     *  closer may grow it. The new shards stay invisible to every reader until the release store
     *  that counts them, and a failed build publishes none. @p build is called as
     *  `build(like) → shard_t`, its index reserved with @ref shard_capacity as the ceiling, a null
     *  index refusing with @c bad_alloc_k.
     */
    template <typename build_type_>
    status_t open_generation_(generation_t full, build_type_&& build) noexcept {
        return open_generation_under_(full, 0, build);
    }

    /** The closer's @ref open_generation_ of a set it sealed: the new shards are born closed like
     *  the rest, before any reader or claimer can see them, so a packer or a load grows the set it
     *  holds closed without a window in which a batch could claim there. */
    template <typename build_type_>
    status_t open_generation_past_seal_(generation_t full, build_type_&& build) noexcept {
        return open_generation_under_(full, sealed_k, build);
    }

#pragma endregion Generations

#pragma region Keys and Routing

    /** Every key's entries. Duplicates coexist as separate slots, which is what several values
     *  under one key means, and costs one slot per row rather than per key. */
    locators_map_t& locators() noexcept { return locators_of_key_; }
    locators_map_t const& locators() const noexcept { return locators_of_key_; }

    /** Drops every key's locators, keeps every generation past the first as idle shards for the
     *  next @ref open_generation_ to reuse, and restarts the routing; inside the tower's window,
     *  once every shard forgot its rows in place, so a reader still holding one reads retained
     *  memory and retries on the epoch. Idle shards keep their capacity and stay closed. A set
     *  that retires its shards never forgets them. */
    void forget_rows() noexcept {
        atomic_ref<std::size_t> const count {shards_count_};
        std::size_t const seen = count.load(std::memory_order_relaxed);
        count.store((seen & sealed_k) | min_of(seen & ~sealed_k, generation_width_), std::memory_order_release);
        locators_of_key_.clear();
        reset_routing(0);
    }

    /**
     *  @brief Retires every shard before the open generation, inside the tower's window with every
     *      shard closed: each keeps every record and value a reader may still hold while its
     *      watermark drops to zero, stays closed for good, and frees only when the set dies.
     *
     *  The count of published shards stays, so a reader that entered a retired shard keeps walking
     *  memory nothing frees until its epoch fails, and no generation ever refills one. Drops every
     *  key's locators and restarts the routing. Each shard answers `retire_()`.
     */
    void retire_older_generations_() noexcept {
        std::size_t const first = open_generation().first;
        for (std::size_t shard = retired_shards(); shard != first; ++shard) at_(shard).index->retire_();
        atomic_ref<std::size_t>(retired_shards_).store(first, std::memory_order_release);
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

#pragma endregion Keys and Routing

#pragma region Landings

    /**
     *  @brief Packs every source shard into a shard of this set under @p bound rows, largest first,
     *      each into the shard with room that @p rank scores lowest, and opens a generation through
     *      @p open when none has room.
     *
     *  The source shards number in landing order, the @p own shards of this set first when it packs
     *  in place. In place, an own shard no kept shard takes stays where it is, a shard whose own
     *  rows are still pending takes nothing, and one they moved out of takes only other sources'
     *  shards; otherwise the rows already here count against the bound.
     *
     *  @param[out] landings One per source shard, @c from naming it in landing order.
     *  @param[in] source_shard Called as `source_shard(from) → shard_t const&`.
     *  @param[in] placement_of Called as `placement_of(shard) → placement`, compared for equality.
     *  @param[in] rank Called as `rank(rows, match) → std::size_t` for a shard with room holding
     *      @c rows, @c match telling whether it shares the source shard's placement:
     *      @ref rank_first_fit or @ref rank_fewest_rows.
     *  @param[in] open Called as `open() → status_t`, appending a generation closed like the rest.
     *  @retval unsupported_k A source shard past @p bound, which would have to be cut first.
     */
    template <typename source_shard_type_, typename placement_type_, typename rank_type_, typename open_type_>
    status_t pack(std::span<landing_t> landings, std::size_t own, std::size_t bound, source_shard_type_&& source_shard,
                  placement_type_&& placement_of, rank_type_&& rank, open_type_&& open) const noexcept {
        constexpr std::size_t unplaced = std::numeric_limits<std::size_t>::max();
        std::size_t const count = landings.size();
        for (std::size_t from = 0; from != count; ++from) landings[from] = {from, unplaced};
        auto load_of = [&](std::size_t into) noexcept {
            std::size_t rows = own ? 0 : at_(into).index->size();
            for (landing_t const& landing : landings)
                if (landing.into == into) rows += source_shard(landing.from).index->size();
            return rows;
        };
        for (std::size_t placed = 0; placed != count; ++placed) {
            // The largest shard left; own shards number first, so a tie keeps one in place.
            std::size_t from = unplaced, rows = 0;
            for (std::size_t candidate = 0; candidate != count; ++candidate) {
                if (landings[candidate].into != unplaced) continue;
                std::size_t const candidate_rows = source_shard(candidate).index->size();
                if (from == unplaced || candidate_rows > rows) from = candidate, rows = candidate_rows;
            }
            if (rows > bound) return status_t::unsupported_k;
            auto best = [&]() noexcept {
                std::size_t into = unplaced, best_rank = unplaced;
                for (std::size_t bin = retired_shards(), shards = size(); bin != shards; ++bin) {
                    std::size_t const kept = bin < own ? landings[bin].into : bin;
                    if (kept != bin && (kept == unplaced || from < own)) continue;
                    std::size_t const load = load_of(bin);
                    if (load + rows > bound) continue;
                    placement_match_t const match = placement_of(at_(bin)) == placement_of(source_shard(from))
                                                        ? placement_match_t::same_k
                                                        : placement_match_t::other_k;
                    std::size_t const ranked = rank(load, match);
                    if (ranked < best_rank) into = bin, best_rank = ranked;
                }
                return into;
            };
            std::size_t into = best();
            // An own shard nothing takes stays put; another source's opens a generation.
            if (into == unplaced && from < own) into = from;
            else if (into == unplaced) {
                if (status_t const opened = open(); failed(opened)) return opened;
                into = best();
                if (into == unplaced) return status_t::overflow_k;
            }
            landings[from].into = into;
        }
        return status_t::success_k;
    }

    /** Sizes the part runs of @p shards target shards to @p landings: a resident part, and one per
     *  source shard landing in the shard; @p part_offsets cuts @p part_bases, every base unset. */
    template <typename offsets_type_, typename bases_type_>
    static status_t size_parts(std::span<landing_t const> landings, std::size_t shards, offsets_type_& part_offsets,
                               bases_type_& part_bases) noexcept {
        if (failed(part_offsets.resize(shards + 1))) return status_t::bad_alloc_k;
        std::size_t parts = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            part_offsets[into] = parts++;
            for (landing_t const& landing : landings)
                if (landing.into == into) ++parts;
        }
        part_offsets[shards] = parts;
        if (failed(part_bases.resize(parts))) return status_t::bad_alloc_k;
        for (node_id_t& base : part_bases) base = node_id_unknown_k;
        return status_t::success_k;
    }

    /** Writes into @p part_bases, one run per shard as @p part_offsets cuts it, a part for every
     *  non-empty shard of @p donor its load @p plans appended where @p landings, one per donor
     *  shard, put it, after the resident part and any earlier donor's; a shard empty beforehand
     *  keeps the first part landing in it as its own, and in place a shard landing in itself stays
     *  resident. The offsets count the shards the plans span, never a generation opened since; the
     *  donor may sit on another residency, as a compaction's device sources do. */
    template <typename donor_type_, typename plans_type_>
    void record_parts(donor_type_ const& donor, std::span<landing_t const> landings, plans_type_ const& plans,
                      node_id_t* part_bases, std::span<std::size_t const> part_offsets) const noexcept {
        std::size_t const shards = part_offsets.size() - 1, donors = landings.size();
        for (std::size_t into = 0; into != shards; ++into) {
            node_id_t* const parts = part_bases + part_offsets[into];
            std::size_t const room = part_offsets[into + 1] - part_offsets[into];
            std::size_t count = 0;
            while (count != room && parts[count] != node_id_unknown_k) ++count;
            node_id_t base = plans[into].base;
            for (std::size_t from = 0; from != donors; ++from) {
                if (landings[from].into != into || (static_cast<void const*>(&donor) == this && from == into)) continue;
                node_id_t const rows = static_cast<node_id_t>(donor[from].index->size());
                if (!rows) continue;
                if (!count && base) parts[count++] = 0;
                parts[count++] = base;
                base += rows;
            }
        }
    }

#pragma endregion Landings

#pragma region Commits and Snapshots

    /** Opens a commit spanning several shards: its watermarks may now move one shard at a time. The
     *  no-return adds go through @c atomic_add, so the standard reference serves as well as
     *  ForkUnion's. */
    void begin_commit() noexcept {
        atomic_add(atomic_ref<u64_t>(commit_sequence_), u64_t {1}, std::memory_order_release);
    }

    /** Closes the commit @ref begin_commit opened; closing carries into the high half, so a reader
     *  sees the word move even when two commits overlap. */
    void end_commit() noexcept {
        atomic_add(atomic_ref<u64_t>(commit_sequence_), (u64_t {1} << 32) - 1, std::memory_order_release);
    }

    /** Takes every shard's watermark into @p watermarks, unless it is empty, at one instant no
     *  commit spanning shards publishes through, so a walk of each shard below its own sees every
     *  batch whole or not at all; answers their sum. A non-empty @p watermarks holds one per shard,
     *  and a generation opened since it was sized answers @c shape_mismatch_k, to size it again. */
    expected<std::size_t> snapshot(std::span<node_id_t> watermarks) const noexcept {
        atomic_ref<u64_t const> const sequence {commit_sequence_};
        while (true) {
            u64_t const before = sequence.load(std::memory_order_acquire);
            // A set low half is a commit between its shards' stores.
            if (static_cast<u32_t>(before)) {
                wait_type_ {}(&commit_sequence_, before, 0u);
                continue;
            }
            // Read past the sequence, so a commit it saw counts every generation it landed in.
            std::size_t const shards = size();
            if (!watermarks.empty() && watermarks.size() != shards) return {0, status_t::shape_mismatch_k};
            std::size_t total = 0;
            for (std::size_t shard = 0; shard != shards; ++shard) {
                std::size_t const visible = at_(shard).index->size();
                if (!watermarks.empty()) watermarks[shard] = static_cast<node_id_t>(visible);
                total += visible;
            }
            // The watermarks' acquires keep this re-read after them.
            if (sequence.load(std::memory_order_relaxed) == before) return {total, status_t::success_k};
        }
    }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t rows() const noexcept { return snapshot({}).value; }

    /** The epoch a reader starts under: an even one, waiting out a window in progress through the
     *  waiter. The reader takes no read-modify-write, so readers never starve a window. */
    u64_t read_epoch() const noexcept {
        atomic_ref<u64_t const> const epoch {epoch_};
        u64_t observed = epoch.load(std::memory_order_acquire);
        for (; observed & 1; observed = epoch.load(std::memory_order_acquire)) wait_type_ {}(&epoch_, observed, 0u);
        return observed;
    }

    /** Whether no window began since @p epoch, so everything read under it stands; the fence keeps
     *  the re-read after every read of the bracketed call. */
    bool still_valid(u64_t epoch) const noexcept {
        std::atomic_thread_fence(std::memory_order_acquire);
        return atomic_ref<u64_t const>(epoch_).load(std::memory_order_relaxed) == epoch;
    }

    /** Opens a destructive window under the tower's unique side: the epoch turns odd before any
     *  store of the window, which the fence keeps after it. */
    void begin_window() noexcept {
        atomic_ref<u64_t>(epoch_).fetch_add(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }

    /** Closes the window @ref begin_window opened, releasing its stores to the next reader. */
    void end_window() noexcept { atomic_ref<u64_t>(epoch_).fetch_add(1, std::memory_order_release); }

    /** Runs the reader @p read under the epoch, again once if a destructive window overlapped it,
     *  and answers @c contended_k if one overlapped that too, so whatever it returns was read whole
     *  between windows; @p read answers an @c expected or a @c status_t. */
    template <typename read_type_>
    std::invoke_result_t<read_type_&> read_retrying_once(read_type_&& read) const noexcept {
        using read_result_t = std::invoke_result_t<read_type_&>;
        for (std::size_t attempt = 0; attempt != 2; ++attempt) {
            u64_t const epoch = read_epoch();
            read_result_t result = read();
            if (still_valid(epoch)) return result;
        }
        if constexpr (std::is_same_v<read_result_t, status_t>) return status_t::contended_k;
        else return {{}, status_t::contended_k};
    }

    /** Runs the reader @p read until no destructive window overlapped it, for an answer with no
     *  room for @c contended_k; each retry waits out a window still open through the waiter, and
     *  every window ends without waiting on a reader. */
    template <typename read_type_>
    std::invoke_result_t<read_type_&> read_until_settled(read_type_&& read) const noexcept {
        while (true) {
            u64_t const epoch = read_epoch();
            std::invoke_result_t<read_type_&> result = read();
            if (still_valid(epoch)) return result;
        }
    }

#pragma endregion Commits and Snapshots

  private:

    /** The high bit of @c shards_count_, set while one caller opens a generation past the count. */
    inline static constexpr std::size_t opening_k = std::size_t {1} << (std::numeric_limits<std::size_t>::digits - 1);

    /** The bit below @ref opening_k, set while @ref close holds the set closed. */
    inline static constexpr std::size_t sealed_k = opening_k >> 1;

    /** The one body of @ref open_generation_ and @ref open_generation_past_seal_: @p seal is the
     *  seal bit the caller expects on the count word, and the one every new shard is born under. */
    template <typename build_type_>
    status_t open_generation_under_(generation_t full, std::size_t seal, build_type_& build) noexcept {
        atomic_ref<std::size_t> const count {shards_count_};
        std::size_t const first = full.first + full.width;
        while (true) {
            std::size_t seen = first | seal;
            if (count.compare_exchange_strong(seen, first | seal | opening_k, std::memory_order_acquire,
                                              std::memory_order_acquire))
                break;
            if ((seen & sealed_k) != seal) return status_t::contended_k;
            if ((seen & ~(opening_k | sealed_k)) != first) return status_t::success_k;
            wait_type_ {}(&shards_count_, seen, 0u);
        }
        // Idle shards a forget kept are whole generations, closed, and already reserved.
        if (first + full.width <= built_shards_) {
            if (!seal)
                for (std::size_t shard = first; shard != first + full.width; ++shard) at_(shard).index->reopen_();
            count.store((first + full.width) | seal, std::memory_order_release);
            return status_t::success_k;
        }
        status_t built = status_t::success_k;
        std::size_t constructed = 0;
        for (; constructed != full.width; ++constructed) {
            shard_t shard = build(at_(full.first + constructed));
            if (!shard.index) built = status_t::bad_alloc_k;
            else built = construct_(first + constructed, std::move(shard));
            if (failed(built)) break;
            if (seal) [[maybe_unused]]
                status_t const closed = at_(first + constructed).index->close_();
        }
        if (failed(built)) {
            for (std::size_t shard = first; shard != first + constructed; ++shard) at_(shard).~shard_t();
            count.store(first | seal, std::memory_order_release);
            return built;
        }
        atomic_ref<std::size_t>(built_shards_).store(first + full.width, std::memory_order_release);
        count.store((first + full.width) | seal, std::memory_order_release);
        return status_t::success_k;
    }

    /** The shard at @p shard: chunk @c c holds the 2^c shards from 2^c - 1 on. */
    shard_t& at_(std::size_t shard) const noexcept {
        std::size_t const biased = shard + 1;
        std::size_t const chunk = static_cast<std::size_t>(std::bit_width(biased)) - 1;
        return shard_chunks_[chunk][biased - (std::size_t {1} << chunk)];
    }

    /** Moves @p shard into slot @p position, allocating its chunk at the chunk's first slot; the
     *  count is the caller's to publish. */
    status_t construct_(std::size_t position, shard_t&& shard) noexcept {
        std::size_t const biased = position + 1;
        std::size_t const chunk = static_cast<std::size_t>(std::bit_width(biased)) - 1;
        if (!shard_chunks_[chunk]) {
            shard_t* const chunk_shards = forkunion_allocator<shard_t> {allocator_}.allocate(std::size_t {1} << chunk);
            if (!chunk_shards) return status_t::bad_alloc_k;
            shard_chunks_[chunk] = chunk_shards;
        }
        new (shard_chunks_[chunk] + (biased - (std::size_t {1} << chunk))) shard_t {std::move(shard)};
        return status_t::success_k;
    }

    /** The owned shards, graph and keys together, in chunks that double and never move, so a search
     *  reading shard @c s while a generation opens keeps its address; at most as many chunks as a
     *  count has bits. A shard's key column grows only ahead of the rows a search can read, since a
     *  search reads it while rows are still landing. */
    std::array<shard_t*, std::numeric_limits<std::size_t>::digits> shard_chunks_ {};

    /** Shards published, read with acquire by @ref size; a generation's shards exist before they
     *  count. The high bit marks a generation being opened, the next one a sealed set. */
    std::size_t shards_count_ = 0;

    /** Shards constructed in the chunks: the published ones, then the idle ones a @ref forget_rows
     *  kept past the count; only the caller opening a generation grows it. */
    std::size_t built_shards_ = 0;

    /** Shards @ref retire_older_generations_ retired, a prefix of the published ones; read with
     *  acquire by @ref retired. */
    std::size_t retired_shards_ = 0;

    /** Placements of the first generation, which every later one repeats. */
    std::size_t generation_width_ = 0;

    /** Every key's entries, as @ref locators exposes them. */
    locators_map_t locators_of_key_;

    /** The host allocator behind the shard table, the locator map and every checkpoint catalog. */
    forkunion_allocator_t allocator_;

    /** Rows routed so far - the next batch's rotation base. */
    std::size_t routed_rows_ = 0;

    /** Rows one shard may hold, set by @ref bound_shards. */
    std::size_t shard_capacity_ = index_t::max_node_count_k;

    /** Commits in flight in the low half, commits landed in the high half: a reader spanning the
     *  shards retries its snapshot while one is in flight or one landed under it. Every search
     *  polls it, so it keeps off the line every insertion bumps. */
    alignas(default_alignment_k) u64_t commit_sequence_ = 0;

    /** Destructive windows in progress in its low bit, windows closed above it: a reader brackets
     *  its call with @ref read_epoch and @ref still_valid, as basic's certified probes do. On the
     *  commit word's line, which every reader polls already. */
    u64_t epoch_ = 0;
};

/**
 *  @brief A reader's position over the published rows of a @ref shard_set, shard by shard in
 *      placement order: the component every plural's cursor owns.
 *
 *  Each shard's watermark is snapshotted as the position enters it, so a row once passed is never
 *  revisited, while a later shard may carry rows published after the walk began. Every call names
 *  the set it walks, which must be the one it opened on. The position holds only under the epoch
 *  it opened in, which @ref still_valid re-checks after every read.
 */
template <typename shards_type_>
struct shard_cursor {
    using shards_t = shards_type_;
    using node_id_t = typename shards_t::index_t::node_id_t;

    /** Enters the first shard of @p shards holding rows, under the epoch it reads now. */
    explicit shard_cursor(shards_t const& shards) noexcept : epoch_(shards.read_epoch()) { enter_(shards); }

    /** Whether every row of @p shards was passed. */
    bool exhausted(shards_t const& shards) const noexcept { return shard_ >= shards.size(); }

    /** The shard holding the current row. */
    std::size_t shard() const noexcept { return shard_; }

    /** The current row inside that shard. */
    node_id_t node() const noexcept { return node_; }

    /** Steps past the current row, entering the next shard holding rows once this one ran out. */
    void advance(shards_t const& shards) noexcept {
        if (++node_ != visible_) return;
        ++shard_;
        enter_(shards);
    }

    /** Whether no destructive window began since the position opened, so every row read stands. */
    bool still_valid(shards_t const& shards) const noexcept { return shards.still_valid(epoch_); }

  private:

    /** Snapshots the current shard's watermark, skipping shards with nothing published. */
    void enter_(shards_t const& shards) noexcept {
        for (node_ = 0; shard_ < shards.size(); ++shard_)
            if ((visible_ = static_cast<node_id_t>(shards[shard_].index->size())) != 0) return;
    }

    /** The shard being read, positional in placement order; the shard count past the end. */
    std::size_t shard_ = 0;

    /** The current row inside that shard. */
    node_id_t node_ = 0;

    /** That shard's watermark snapshot: rows strictly below it are readable. */
    node_id_t visible_ = 0;

    /** The epoch the position opened in. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::utilities
