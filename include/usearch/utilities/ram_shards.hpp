/**
 *  @file include/usearch/utilities/ram_shards.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Shards viewed as one: the component every sharded index owns, holding the shards, the
 *      rotation that routes new entries, and the commit word a reader spanning the shards
 *      snapshots by.
 *
 *  A tower keeps its own verbs and its own residency's logic, and reaches its shards through this
 *  set: placement and reservation, the key reads spanning every shard, the claim words closed and
 *  reopened in shard order, the snapshot of every shard's watermark at one instant no commit
 *  straddles, and the epoch its readers bracket themselves with against the tower's destructive
 *  windows. The tower builds and reserves every shard and checks its shape, so the set never reads
 *  one.
 */
#pragma once

#include <array>       // `std::array` - the shard table's chunks
#include <atomic>      // `std::atomic_thread_fence` - the epoch's reader bracket
#include <bit>         // `std::bit_width` - a shard's chunk
#include <limits>      // `std::numeric_limits` - the chunk count & the opening bit
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <new>         // placement `new` - a shard is built in its chunk
#include <span>        // `std::span` - the watermarks a snapshot fills
#include <type_traits> // `std::invoke_result_t`, `std::is_same_v` - the bracketed readers' answers
#include <utility>     // `std::move`, `std::swap`, `std::exchange`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator` - the shard table's placement

namespace unum::usearch::utilities {

/** Where one source shard's entries move: shard @c from of the sources' concatenated shard lists,
 *  into shard @c into of the target; two words, sixteen bytes, or eight on 32-bit addresses. */
struct shard_move_t {
    std::size_t from = 0;
    std::size_t into = 0;
};

static_assert(sizeof(shard_move_t) == 2 * sizeof(std::size_t));

/** Where one entry of a split batch goes: the shard, counted from the open shard generation's
 *  first, and its offset among the entries that shard takes. Two words, as @ref shard_move_t. */
struct entry_in_shard_t {
    std::size_t shard = 0;
    std::size_t offset = 0;
};

static_assert(sizeof(entry_in_shard_t) == 2 * sizeof(std::size_t));

/** Whether a target shard shares the source shard's placement, which every rank prefers. */
enum class placement_match_t : bool { other_k, same_k };

/** Ranks a target shard with room for a compaction, the lowest first: any on the source shard's
 *  placement before any other, the earliest on ties, so each shard fills before the next takes. */
inline std::size_t rank_first_fit(std::size_t, placement_match_t match) noexcept {
    return match == placement_match_t::same_k ? 0 : 1;
}

/** Ranks a target shard with room for an absorb, the lowest first: the fewest entries, then one on
 *  the source shard's placement, so the entries spread over the shards as they stand. */
inline std::size_t rank_fewest_entries(std::size_t entries, placement_match_t match) noexcept {
    return 2 * entries + (match == placement_match_t::same_k ? 0 : 1);
}

/**
 *  @brief Shards viewed as one, and the words that keep a reader spanning them consistent.
 *
 *  It owns its shards outright, so a tower's destructor, consuming merge and clear are expressible.
 *  Shards arrive one at a time through @ref emplace, so a tower grows as its placements are engaged
 *  rather than being declared whole. Those placements are the first shard generation; once its
 *  shards reach @ref shard_capacity, @c open_shard_generation_ appends another shard per placement,
 *  and only the newest shard generation takes insertions. A tower whose readers may hold a shard's
 *  values retires its shards rather than emptying them, through @c retire_older_shard_generations_:
 *  they stay in place, closed and empty to every reader, until the set dies. The set reads no
 *  shard's shape: the tower checks it on @ref emplace and reserves every shard a new shard
 *  generation opens with.
 *
 *  @tparam index_type_ The shard type.
 *  @tparam atomic_ref_template_ The atomic reference owning the plain synchronization words.
 *  @tparam wait_type_ What a snapshot waits through while a commit is in flight.
 */
template <typename index_type_, template <typename> class atomic_ref_template_, typename wait_type_>
struct shard_set {

    /** The shard type, built and shaped by the tower. */
    using index_t = index_type_;

    /** The caller-visible identity of every entry, which each shard files beside its own nodes. */
    using key_t = typename index_t::key_t;

    /** A node inside one shard, as wide as the shard names it. */
    using node_id_t = typename index_t::node_id_t;

    /** The atomic reference owning the plain synchronization words - the shards' and the set's
     *  own routing tally alike. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    using reserve_capacity_t = typename index_t::reserve_capacity_t;

    /** One shard: the graph, which owns its keys and carries its own placement. */
    struct shard_t {

        /** The graph, growing on demand; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;
    };

    /** One shard generation: one new shard per placement, opened when the current shards reach
     *  their bound, as @c width shards from @c first. */
    struct shard_generation_t {
        std::size_t first = 0;
        std::size_t width = 0;
    };

    /** One batch split across the open shard generation and the fresh ones it opened: entries are
     *  split across the shards round-robin from the starting shard on, the open shard generation
     *  taking the entries below @c entries_in_open_shard_generation and every fresh one as many as
     *  the bound lets its shards hold. */
    struct batch_split_t {

        /** The shard generation open when the batch was split, the first the batch goes into. */
        shard_generation_t open_shard_generation {};

        /** Entries in the batch. */
        std::size_t batch_entries = 0;

        /** The shard of each shard generation that entry zero of its share goes to. */
        std::size_t starting_shard = 0;

        /** Entries that go into the open shard generation; the rest go into fresh ones. */
        std::size_t entries_in_open_shard_generation = 0;

        /** Entries a fresh shard generation takes before the next one opens: its width times the
         *  bound. */
        std::size_t entries_per_fresh_shard_generation = 0;

        /** Fresh shard generations the batch fills past the open one. */
        std::size_t fresh_shard_generations() const noexcept {
            return divide_round_up(batch_entries - entries_in_open_shard_generation,
                                   entries_per_fresh_shard_generation);
        }

        /** Shards the batch spans from the open shard generation's first, in placement order. */
        std::size_t shards_spanned() const noexcept {
            return open_shard_generation.width * (1 + fresh_shard_generations());
        }

        /** Entries shard @p shard takes, counted from the open shard generation's first. */
        std::size_t entries_for_shard(std::size_t shard) const noexcept {
            std::size_t const width = open_shard_generation.width;
            std::size_t const shard_generation = shard / width, offset = shard % width;
            std::size_t const share = !shard_generation
                                          ? entries_in_open_shard_generation
                                          : min_of(entries_per_fresh_shard_generation,
                                                   batch_entries - entries_in_open_shard_generation -
                                                       (shard_generation - 1) * entries_per_fresh_shard_generation);
            return entries_from(share, first_entry_of_shard(offset, width, starting_shard), width);
        }

        /** Where entry @p entry of the batch goes: its shard from the open shard generation's
         *  first, and its offset among that shard's entries. */
        entry_in_shard_t shard_and_offset_of_entry(std::size_t entry) const noexcept {
            std::size_t const width = open_shard_generation.width;
            std::size_t shard_generation = 0, entry_in_shard_generation = entry;
            if (entry >= entries_in_open_shard_generation) {
                shard_generation = 1 + (entry - entries_in_open_shard_generation) / entries_per_fresh_shard_generation;
                entry_in_shard_generation = (entry - entries_in_open_shard_generation) %
                                            entries_per_fresh_shard_generation;
            }
            return {shard_generation * width + (starting_shard + entry_in_shard_generation) % width,
                    entry_in_shard_generation / width};
        }
    };

    /** Walks the shards in placement order; the chunks never move, so a shard it reached keeps its
     *  address while a shard generation opens. */
    template <typename shard_type_>
    struct shard_iterator {
        shard_set const* set = nullptr;
        std::size_t shard = 0;

        shard_type_& operator*() const noexcept { return set->at_(shard); }
        shard_type_* operator->() const noexcept { return &set->at_(shard); }
        shard_iterator& operator++() noexcept { return ++shard, *this; }
        bool operator==(shard_iterator const& other) const noexcept { return shard == other.shard; }
    };

    /** An empty set of shards, its shard table and checkpoint catalogs on @p allocator. */
    explicit shard_set(forkunion_allocator_t allocator) noexcept : allocator_(allocator) {}
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
        std::swap(shard_generation_width_, other.shard_generation_width_);
        std::swap(allocator_, other.allocator_);
        std::swap(shard_capacity_, other.shard_capacity_);
        std::size_t const routed_here = routed();
        reset_routing(other.routed());
        other.reset_routing(routed_here);
    }

#pragma region Shards

    /** Shards published in the set, in placement order, every shard generation's and the retired
     *  ones; read with acquire, so every shard it counts is whole. */
    std::size_t size() const noexcept {
        return atomic_ref<std::size_t const>(shards_count_).load(std::memory_order_acquire) & ~(opening_k | sealed_k);
    }
    bool empty() const noexcept { return !size(); }

    /** Shards @c retire_older_shard_generations_ retired, the first ones in placement order: each
     *  counts no entries and stays closed for good. Iteration starts past them, while positions
     *  keep counting them. */
    std::size_t retired_shards() const noexcept {
        return atomic_ref<std::size_t const>(retired_shards_).load(std::memory_order_acquire);
    }

    /** Whether @ref close sealed the set: no claim may start and no shard generation may open but
     *  the closer's own, until @ref reopen. Stable for a caller holding the tower's shared side. */
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

    /** The host allocator behind the shard table and every checkpoint catalog; the shards carry
     *  their own. */
    forkunion_allocator_t allocator() const noexcept { return allocator_; }

    /** Entries the shards past the retired ones host without growing again; they grow past it on
     *  demand. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& shard : *this) total += shard.index->capacity();
        return total;
    }

    /** Bytes every shard's storage occupies, idle and retired shards included. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = 0;
        std::size_t const built = atomic_ref<std::size_t const>(built_shards_).load(std::memory_order_acquire);
        for (std::size_t shard = 0; shard != built; ++shard) bytes += at_(shard).index->memory_usage();
        return bytes;
    }

    /** Bounds every shard at @p entries, clamped to the ID space; zero means the ID space. Fixed
     *  once storage exists, as the placement is. */
    void bound_shards(std::size_t entries) noexcept {
        if (capacity() || retired_shards()) return;
        shard_capacity_ = entries && entries < index_t::max_node_count_k ? entries : index_t::max_node_count_k;
    }

    /** Entries one shard may hold. */
    std::size_t shard_capacity() const noexcept { return shard_capacity_; }

    /** Appends one shard the caller already built; refuses a null one and one already reserved, the
     *  tower having checked its shape. Several shards may share a placement, each a slice of the
     *  graph rather than a copy. @ref reserve closes the set.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p shard is null or already reserved.
     *  - @c config_mismatch_k The set is reserved, retired shards, or has a shard generation in
     *    flight.
     */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard || shard->capacity()) return status_t::config_mismatch_k;
        if (capacity() || retired_shards() || built_shards_ != size()) return status_t::config_mismatch_k;
        std::size_t const shards = size();
        if (status_t const placed = construct_(shards, shard_t {std::move(shard)}); failed(placed)) return placed;
        atomic_ref<std::size_t>(built_shards_).store(shards + 1, std::memory_order_release);
        atomic_ref<std::size_t>(shards_count_).store(shards + 1, std::memory_order_release);
        shard_generation_width_ = shards + 1;
        return status_t::success_k;
    }

    /**
     *  @brief Grows every shard past the retired ones ahead of @p config.capacity entries, split
     *      evenly over them - a hint, since the shards grow on demand.
     *
     *  Callable alongside insertions and searches on the host residency: nothing it grows ever
     *  moves.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c unknown_kind_k A dense shard's kits cannot score its metric.
     *  - @c config_mismatch_k A shard's geometry is malformed, or on NVIDIA its neighbors live on
     *    another device.
     *  - @c overflow_k A shard's share passes its ID space, its ceiling, its chunk directory or its
     *    key column.
     *  - @c unsupported_k On Apple, a pipeline is narrower than the kernels need.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (empty()) return {0, status_t::uninitialized_k};
        reserve_capacity_t per_shard = config;
        per_shard.capacity = min_of(divide_round_up(config.capacity, size() - retired_shards()), shard_capacity_);
        per_shard.ceiling = shard_capacity_;
        for (shard_t& shard : *this)
            if (status_t const reserved = shard.index->reserve(per_shard).status; failed(reserved))
                return {capacity(), reserved};
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
     *
     *  @par Statuses:
     *  - @c contended_k A shard has a batch or load in flight, or holds claims it never dispensed.
     */
    status_t close(std::size_t first = 0) noexcept {
        std::size_t const live = max_of(first, retired_shards());
        for (std::size_t shard = live, shards = size(); shard != shards; ++shard) {
            index_t& shard_index = *at_(shard).index;
            status_t closed = shard_index.close_();
            // A dense RAM shard closes only its dispenser, so undispensed claims refuse here.
            if (succeeded(closed) && shard_index.claimed_() != shard_index.size())
                shard_index.reopen_(), closed = status_t::contended_k;
            if (failed(closed)) {
                while (shard != live) at_(--shard).index->reopen_();
                return closed;
            }
        }
        // The unique side orders the seal for every later claimer, which takes the shared side.
        atomic_ref<std::size_t>(shards_count_).set_bits(sealed_k, std::memory_order_relaxed);
        return status_t::success_k;
    }

    /** Reopens every closed shard's claim word at its watermark, the last shard first, so a batch
     *  taking the shards in order finds the first one closed until every later one reopened, then
     *  lifts the seal. Idle shards past the count and retired ones before it stay closed. */
    void reopen() noexcept {
        for (std::size_t shard = size(), live = retired_shards(); shard != live;) at_(--shard).index->reopen_();
        atomic_ref<std::size_t>(shards_count_).clear_bits(sealed_k, std::memory_order_release);
    }

#pragma endregion Shards

#pragma region Generations

    /** The newest shard generation, the only one insertions go into; the older ones are sealed. */
    shard_generation_t open_shard_generation() const noexcept {
        std::size_t const shards = size();
        return {shards - shard_generation_width_, shard_generation_width_};
    }

    /** Entries a batch split over @p shard generation from @p starting_shard places before any of
     *  its shards passes the bound: the shard taking entries from @c f on, with @c l entries of
     *  room left, takes up to entry @c f + l * width. */
    std::size_t room(shard_generation_t shard_generation, std::size_t starting_shard) const noexcept {
        std::size_t entries = (std::numeric_limits<std::size_t>::max)();
        for (std::size_t offset = 0; offset != shard_generation.width; ++offset) {
            std::size_t const claimed = at_(shard_generation.first + offset).index->claimed_();
            std::size_t const left = claimed < shard_capacity_ ? shard_capacity_ - claimed : 0;
            entries = min_of(entries, first_entry_of_shard(offset, shard_generation.width, starting_shard) +
                                          left * shard_generation.width);
        }
        return entries;
    }

    /**
     *  @brief Splits a batch of @p entries across the open shard generation's shards, opening
     *      through @p build as many fresh shard generations as the entries past the bound fill, as
     *      @c open_shard_generation_ does.
     *
     *  The starting shard comes from the entries routed so far, which keeps a stream of small
     *  batches from drifting onto shard zero; a racing batch taking the same room first leaves this
     *  one's claims refused with @c overflow_k, and a failed shard generation answers its status.
     *
     *  @par Statuses:
     *  - @c contended_k The set is sealed while a fresh shard generation must open.
     *  - @c bad_alloc_k @p build answers a null index.
     */
    template <typename build_type_>
    expected<batch_split_t> split_batch_into_shards(std::size_t entries, build_type_&& build) noexcept {
        batch_split_t split;
        split.open_shard_generation = open_shard_generation();
        std::size_t const width = split.open_shard_generation.width;
        split.batch_entries = entries;
        split.starting_shard = route(entries) % width;
        split.entries_in_open_shard_generation = min_of(entries,
                                                        room(split.open_shard_generation, split.starting_shard));
        split.entries_per_fresh_shard_generation = width * shard_capacity_;
        for (std::size_t opened = 0, fresh = split.fresh_shard_generations(); opened != fresh; ++opened)
            if (status_t const generated = open_shard_generation_(
                    {split.open_shard_generation.first + opened * width, width}, build);
                failed(generated))
                return {split, generated};
        return {split, status_t::success_k};
    }

    /**
     *  @brief Opens the shard generation after @p full: one shard per placement, built and reserved
     *      under the bound by @p build from that placement's shard of @p full, or the idle shards a
     *      @ref forget_entries kept there, all published at once.
     *
     *  One caller opens it while the others wait on the count word, and a caller finding it opened
     *  already answers success untouched; a sealed set answers @c contended_k, since only its
     *  closer may grow it. The new shards stay invisible to every reader until the release store
     *  that counts them, and a failed build publishes none. @p build is called as <tt>build(like) →
     *  shard_t</tt>, its index reserved with @ref shard_capacity as the ceiling, a null index
     *  refusing with @c bad_alloc_k.
     */
    template <typename build_type_>
    status_t open_shard_generation_(shard_generation_t full, build_type_&& build) noexcept {
        return open_shard_generation_under_(full, 0, build);
    }

    /** The closer's @ref open_shard_generation_ of a set it sealed: the new shards are born closed
     *  like the rest, before any reader or claimer can see them, so a packer or a load grows the
     *  set it holds closed without a window in which a batch could claim there. */
    template <typename build_type_>
    status_t open_shard_generation_past_seal_(shard_generation_t full, build_type_&& build) noexcept {
        return open_shard_generation_under_(full, sealed_k, build);
    }

#pragma endregion Generations

#pragma region Keys and Routing

    /** Entries under @p key across every shard, exact: read until settled, so a batch spanning
     *  shards counts whole or not at all. */
    std::size_t count(typename index_t::key_t const& key) const noexcept {
        return read_until_settled([&]() noexcept {
            std::size_t total = 0;
            for (shard_t const& shard : *this) total += shard.index->count(key);
            return total;
        });
    }

    /** Whether any shard holds an entry under @p key, read as @ref count reads, stopping at the
     *  first. */
    bool contains(typename index_t::key_t const& key) const noexcept {
        return read_until_settled([&]() noexcept {
            for (shard_t const& shard : *this)
                if (shard.index->contains(key)) return true;
            return false;
        });
    }

    /** Invokes @p callback as <tt>callback(index, node)</tt> for every entry under @p key in every
     *  shard, each below the watermark its shard acquires. Never retried, so the callback hears
     *  every entry once; a caller needing a consistent view across shards reads a snapshot. */
    template <typename callback_type_>
    void find_each(typename index_t::key_t const& key, callback_type_&& callback) const noexcept {
        for (shard_t const& shard : *this) {
            index_t const& index = *shard.index;
            index.find_each(key, [&](node_id_t node) noexcept { callback(index, node); });
        }
    }

    /** Invokes @p callback as <tt>callback(index, node)</tt> for every entry under @p key below the
     *  snapshot @p visible, one watermark per shard from the first. */
    template <typename callback_type_>
    void find_each(typename index_t::key_t const& key, std::span<node_id_t const> visible,
                   callback_type_&& callback) const noexcept {
        for (std::size_t shard = 0; shard != visible.size(); ++shard) {
            index_t const& index = *at_(shard).index;
            index.keys().find_each(key, visible[shard], [&](node_id_t node) noexcept { callback(index, node); });
        }
    }

    /** Keeps every shard generation past the first as idle shards for the next
     *  @c open_shard_generation_ to reuse, and restarts the routing; inside the tower's window,
     *  once every shard forgot its entries in place, so a reader still holding one reads retained
     *  memory and retries on the epoch. Idle shards keep their capacity and stay closed. A set that
     *  retires its shards never forgets them. */
    void forget_entries() noexcept {
        atomic_ref<std::size_t> const count {shards_count_};
        std::size_t const seen = count.load(std::memory_order_relaxed);
        count.store((seen & sealed_k) | min_of(seen & ~sealed_k, shard_generation_width_), std::memory_order_release);
        reset_routing(0);
    }

    /**
     *  @brief Retires every shard before the open shard generation, inside the tower's window with
     *      every shard closed: each keeps every record and value a reader may still hold while its
     *      watermark drops to zero, stays closed for good, and frees only when the set dies.
     *
     *  The count of published shards stays, so a reader that entered a retired shard keeps walking
     *  memory nothing frees until its epoch fails, and no shard generation ever refills one.
     *  Restarts the routing. Each shard answers <tt>retire_()</tt>.
     */
    void retire_older_shard_generations_() noexcept {
        std::size_t const first = open_shard_generation().first;
        for (std::size_t shard = retired_shards(); shard != first; ++shard) at_(shard).index->retire_();
        atomic_ref<std::size_t>(retired_shards_).store(first, std::memory_order_release);
        reset_routing(0);
    }

    /** Routes @p entries more entries, answering the entries routed before them - the next batch's
     *  rotation base, so a stream of small batches never drifts onto shard zero. */
    std::size_t route(std::size_t entries) noexcept {
        return atomic_ref<std::size_t>(routed_entries_).fetch_add(entries, std::memory_order_relaxed);
    }

    /** Entries routed so far - what traversal contexts seed apart by, and apart across calls. */
    std::size_t routed() const noexcept {
        return atomic_ref<std::size_t const>(routed_entries_).load(std::memory_order_relaxed);
    }

    /** Restarts the routing as if @p entries entries had been routed, as a restored checkpoint
     *  does. */
    void reset_routing(std::size_t entries) noexcept {
        atomic_ref<std::size_t>(routed_entries_).store(entries, std::memory_order_relaxed);
    }

    /** The first entry of a batch that @p shard takes under @p rotation - a bijection over the
     *  shards, so the residue classes it names partition the batch exactly. */
    static std::size_t first_entry_of_shard(std::size_t shard, std::size_t shards, std::size_t rotation) noexcept {
        return (shard + shards - rotation) % shards;
    }

    /** How many of @p entries a shard starting at @p first takes, stepping by @p shards. */
    static std::size_t entries_from(std::size_t entries, std::size_t first, std::size_t shards) noexcept {
        return first < entries ? divide_round_up(entries - first, shards) : 0;
    }

#pragma endregion Keys and Routing

#pragma region Shard Moves

    /**
     *  @brief Packs every source shard into a shard of this set under @p bound entries, largest
     *      first, each into the shard with room that @p rank scores lowest, and opens a shard
     *      generation through @p open when none has room.
     *
     *  The source shards number in move order, the @p own shards of this set first when it packs in
     *  place. In place, an own shard no kept shard takes stays where it is, a shard whose own
     *  entries are still pending takes nothing, and one they moved out of takes only other sources'
     *  shards; otherwise the entries already here count against the bound.
     *
     *  @param[out] moves One per source shard, @c from naming it in move order.
     *  @param[in] own How many shards of this set are sources themselves, packing in place.
     *  @param[in] bound The most entries a shard may hold.
     *  @param[in] source_shard Called as <tt>source_shard(from) → shard_t const&amp;</tt>.
     *  @param[in] placement_of Called as <tt>placement_of(shard) → placement</tt>, compared for
     *      equality.
     *  @param[in] rank Called as <tt>rank(entries, match) → std::size_t</tt> for a shard with room
     *      holding @c entries, @c match telling whether it shares the source shard's placement:
     *      @ref rank_first_fit or @ref rank_fewest_entries.
     *  @param[in] open Called as <tt>open() → status_t</tt>, appending a shard generation
     *      closed like the rest.
     *
     *  @par Statuses:
     *  - @c unsupported_k A source shard past @p bound, which would have to be cut first.
     *  - @c overflow_k A freshly opened shard generation still has no room for a source shard.
     */
    template <typename source_shard_type_, typename placement_type_, typename rank_type_, typename open_type_>
    status_t pack(std::span<shard_move_t> moves, std::size_t own, std::size_t bound, source_shard_type_&& source_shard,
                  placement_type_&& placement_of, rank_type_&& rank, open_type_&& open) const noexcept {
        constexpr std::size_t unplaced = (std::numeric_limits<std::size_t>::max)();
        std::size_t const count = moves.size();
        for (std::size_t from = 0; from != count; ++from) moves[from] = {from, unplaced};
        auto load_of = [&](std::size_t into) noexcept {
            std::size_t entries = own ? 0 : at_(into).index->size();
            for (shard_move_t const& move : moves)
                if (move.into == into) entries += source_shard(move.from).index->size();
            return entries;
        };
        for (std::size_t placed = 0; placed != count; ++placed) {
            // The largest shard left; own shards number first, so a tie keeps one in place.
            std::size_t from = unplaced, entries = 0;
            for (std::size_t candidate = 0; candidate != count; ++candidate) {
                if (moves[candidate].into != unplaced) continue;
                std::size_t const candidate_entries = source_shard(candidate).index->size();
                if (from == unplaced || candidate_entries > entries) from = candidate, entries = candidate_entries;
            }
            if (entries > bound) return status_t::unsupported_k;
            auto best = [&]() noexcept {
                std::size_t into = unplaced, best_rank = unplaced;
                for (std::size_t bin = retired_shards(), shards = size(); bin != shards; ++bin) {
                    std::size_t const kept = bin < own ? moves[bin].into : bin;
                    if (kept != bin && (kept == unplaced || from < own)) continue;
                    std::size_t const load = load_of(bin);
                    if (load + entries > bound) continue;
                    placement_match_t const match = placement_of(at_(bin)) == placement_of(source_shard(from))
                                                        ? placement_match_t::same_k
                                                        : placement_match_t::other_k;
                    std::size_t const ranked = rank(load, match);
                    if (ranked < best_rank) into = bin, best_rank = ranked;
                }
                return into;
            };
            std::size_t into = best();
            // An own shard nothing takes stays put; another source's opens a shard generation.
            if (into == unplaced && from < own) into = from;
            else if (into == unplaced) {
                if (status_t const opened = open(); failed(opened)) return opened;
                into = best();
                if (into == unplaced) return status_t::overflow_k;
            }
            moves[from].into = into;
        }
        return status_t::success_k;
    }

    /** Sizes the part runs of @p shards target shards to @p moves: a resident part, and one per
     *  source shard moving into the shard; @p part_offsets cuts @p part_bases, every base unset. */
    template <typename offsets_type_, typename bases_type_>
    static status_t size_parts(std::span<shard_move_t const> moves, std::size_t shards, offsets_type_& part_offsets,
                               bases_type_& part_bases) noexcept {
        if (failed(part_offsets.resize(shards + 1))) return status_t::bad_alloc_k;
        std::size_t parts = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            part_offsets[into] = parts++;
            for (shard_move_t const& move : moves)
                if (move.into == into) ++parts;
        }
        part_offsets[shards] = parts;
        if (failed(part_bases.resize(parts))) return status_t::bad_alloc_k;
        for (node_id_t& base : part_bases) base = node_id_limits<node_id_t>::unknown_k;
        return status_t::success_k;
    }

    /** Writes into @p part_bases, one run per shard as @p part_offsets cuts it, a part for every
     *  non-empty shard of @p donor its load @p plans appended where @p moves, one per donor shard,
     *  put it, after the resident part and any earlier donor's; a shard empty beforehand keeps the
     *  first part moving into it as its own, and in place a shard moving into itself stays
     *  resident. The offsets count the shards the plans span, never a shard generation opened
     *  since; the donor may sit on another residency, as a compaction's device sources do. */
    template <typename donor_type_, typename plans_type_>
    void record_parts(donor_type_ const& donor, std::span<shard_move_t const> moves, plans_type_ const& plans,
                      node_id_t* part_bases, std::span<std::size_t const> part_offsets) const noexcept {
        std::size_t const shards = part_offsets.size() - 1, donors = moves.size();
        for (std::size_t into = 0; into != shards; ++into) {
            node_id_t* const parts = part_bases + part_offsets[into];
            std::size_t const room = part_offsets[into + 1] - part_offsets[into];
            std::size_t count = 0;
            while (count != room && parts[count] != node_id_limits<node_id_t>::unknown_k) ++count;
            node_id_t base = plans[into].base;
            for (std::size_t from = 0; from != donors; ++from) {
                if (moves[from].into != into || (static_cast<void const*>(&donor) == this && from == into)) continue;
                node_id_t const entries = static_cast<node_id_t>(donor[from].index->size());
                if (!entries) continue;
                if (!count && base) parts[count++] = 0;
                parts[count++] = base;
                base += entries;
            }
        }
    }

#pragma endregion Shard Moves

#pragma region Commits and Snapshots

    /** Opens a commit spanning several shards: its watermarks may now move one shard at a time. */
    void begin_commit() noexcept {
        atomic_ref<u64_t>(commit_sequence_).store_add(u64_t {1}, std::memory_order_release);
    }

    /** Closes the commit @ref begin_commit opened; closing carries into the high half, so a reader
     *  sees the word move even when two commits overlap. */
    void end_commit() noexcept {
        atomic_ref<u64_t>(commit_sequence_).store_add((u64_t {1} << 32) - 1, std::memory_order_release);
    }

    /** Takes every shard's watermark into @p watermarks, unless it is empty, at one instant no
     *  commit spanning shards publishes through, so a walk of each shard below its own sees every
     *  batch whole or not at all; answers their sum. A non-empty @p watermarks holds one per shard,
     *  and a shard generation opened since it was sized answers @c shape_mismatch_k, to size it
     *  again.
     *
     *  @par Statuses:
     *  - @c shape_mismatch_k A non-empty @p watermarks holds other than one slot per shard.
     */
    expected<std::size_t> snapshot(std::span<node_id_t> watermarks) const noexcept {
        return read_committed([&]() noexcept -> expected<std::size_t> {
            // Read past the sequence, so a commit it saw counts every generation it published into.
            std::size_t const shards = size();
            if (!watermarks.empty() && watermarks.size() != shards) return {0, status_t::shape_mismatch_k};
            std::size_t total = 0;
            for (std::size_t shard = 0; shard != shards; ++shard) {
                std::size_t const visible = at_(shard).index->size();
                if (!watermarks.empty()) watermarks[shard] = static_cast<node_id_t>(visible);
                total += visible;
            }
            return {total, status_t::success_k};
        });
    }

    /** Runs the reader @p read at one instant no commit spanning shards publishes through, again
     *  while one published under it, so it sees every batch whole or not at all; @p read loads
     *  through acquires, which keep the sequence re-read after them. */
    template <typename read_type_>
    std::invoke_result_t<read_type_&> read_committed(read_type_&& read) const noexcept {
        atomic_ref<u64_t const> const sequence {commit_sequence_};
        while (true) {
            u64_t const before = sequence.load(std::memory_order_acquire);
            // A set low half is a commit between its shards' stores.
            if (static_cast<u32_t>(before)) {
                wait_type_ {}(&commit_sequence_, before, 0u);
                continue;
            }
            std::invoke_result_t<read_type_&> result = read();
            if (sequence.load(std::memory_order_relaxed) == before) return result;
        }
    }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t entries() const noexcept { return snapshot({}).value; }

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
        atomic_ref<u64_t>(epoch_).store_add(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }

    /** Closes the window @ref begin_window opened, releasing its stores to the next reader. */
    void end_window() noexcept { atomic_ref<u64_t>(epoch_).store_add(1, std::memory_order_release); }

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

    /** Runs the reader @p read until neither a destructive window nor a commit spanning
     *  shards overlapped it, for an exact answer with no room for @c contended_k; each
     *  retry waits out a window or a commit still in flight through the waiter, and
     *  neither ever waits on a reader. */
    template <typename read_type_>
    std::invoke_result_t<read_type_&> read_until_settled(read_type_&& read) const noexcept {
        while (true) {
            u64_t const epoch = read_epoch();
            std::invoke_result_t<read_type_&> result = read_committed(read);
            if (still_valid(epoch)) return result;
        }
    }

#pragma endregion Commits and Snapshots

  private:

    /** The high bit of @c shards_count_, set while one caller opens a shard generation past the
     *  count. */
    inline static constexpr std::size_t opening_k = std::size_t {1} << (std::numeric_limits<std::size_t>::digits - 1);

    /** The bit below @c opening_k, set while @ref close holds the set closed. */
    inline static constexpr std::size_t sealed_k = opening_k >> 1;

    /** The one body of @ref open_shard_generation_ and @c open_shard_generation_past_seal_:
     *  @p seal is the seal bit the caller expects on the count word, and the one every new shard is
     *  born under. */
    template <typename build_type_>
    status_t open_shard_generation_under_(shard_generation_t full, std::size_t seal, build_type_& build) noexcept {
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
        // Idle shards a forget kept are whole shard generations, closed, and already reserved.
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

    /** The owned shards, in chunks that double and never move, so a search reading shard @c s while
     *  a shard generation opens keeps its address; at most as many chunks as a count has bits. */
    std::array<shard_t*, std::numeric_limits<std::size_t>::digits> shard_chunks_ {};

    /** Shards published, read with acquire by @ref size; a shard generation's shards exist before
     *  they count. Its high bit marks a shard generation being opened, its next a sealed set. */
    std::size_t shards_count_ = 0;

    /** Shards constructed in the chunks: the published ones, then the idle ones a
     *  @ref forget_entries kept past the count; only the caller opening a shard generation grows
     *  it. */
    std::size_t built_shards_ = 0;

    /** Shards @ref retire_older_shard_generations_ retired, a prefix of the published ones; read
     *  with acquire by @c retired. */
    std::size_t retired_shards_ = 0;

    /** Placements of the first shard generation, which every later one repeats. */
    std::size_t shard_generation_width_ = 0;

    /** The host allocator behind the shard table and every checkpoint catalog. */
    forkunion_allocator_t allocator_;

    /** Entries routed so far - the next batch's rotation base. */
    std::size_t routed_entries_ = 0;

    /** Entries one shard may hold, set by @ref bound_shards. */
    std::size_t shard_capacity_ = index_t::max_node_count_k;

    /** Commits in flight in the low half, commits published in the high half: a reader spanning the
     *  shards retries its snapshot while one is in flight or one published under it. Every search
     *  polls it, so it keeps off the line every insertion bumps. */
    alignas(default_alignment_k) u64_t commit_sequence_ = 0;

    /** Destructive windows in progress in its low bit, windows closed above it: a reader brackets
     *  its call with @ref read_epoch and @ref still_valid, as basic's certified probes do. On the
     *  commit word's line, which every reader polls already. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::utilities
