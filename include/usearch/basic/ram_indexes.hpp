/**
 *  @file include/usearch/basic/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Many basic RAM shards viewed as one, so a basic index grows past one shard's ID space.
 *
 *  The sharded index over @c basic::ram_index, shaped after it: contexts are the caller's, one per
 *  thread, metrics come with every call, and the caller drives its own threads. Shards share one
 *  bound, and an insertion crossing it opens another shard generation on the same placements. Keys
 *  stay in the shards' records, so lookups fan out over every shard, and a search walks each shard
 *  below one snapshot of the watermarks and merges their rankings into the caller's keys.
 *
 *  Mutators hold one gate's shared side for each call, and the destructive windows - a clear, and
 *  an absorb emptying its donors - take its unique side or answer @c contended_k. Readers take no
 *  lock: they bracket each call with the shard set's epoch, retry once when a window overlapped
 *  them, and answer @c contended_k when a second one did. A record's value owns heap memory a
 *  reader may still score, so a clear frees nothing: it retires the shards whole and fresh ones
 *  take the next entries, and retired shards keep their memory until the index dies.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - the sample scratch's alignment

#include <array>        // `std::array` - a sample distance round
#include <functional>   // `std::hash` - the shard set's key hash
#include <memory>       // `std::unique_ptr` - an index holds atomics and never moves
#include <mutex>        // `std::unique_lock` - the windows' hold on the gate
#include <new>          // `std::nothrow` - a shard built on a memory domain
#include <shared_mutex> // `std::shared_lock` - the mutators' hold on the gate
#include <span>         // `std::span` - the caller's watermarks, keys & distances
#include <string_view>  // `std::string_view` - the checkpoint paths
#include <type_traits>  // `std::is_same_v` - the unpredicated walk
#include <utility>      // `std::move`, `std::exchange`

#include <forkunion/forkunion.h> // `fu_memory_domain_id_t`

#include "usearch/utilities/types.hpp"       // `shared_mutex` - the gate
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the plans' shard plans
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator_t` - the shard table's placement
#include "usearch/utilities/ram_shards.hpp"  // `shard_set` - the shard table
#include "usearch/utilities/ram_walks.hpp"   // `cursor_walk` - the cursor's walk
#include "usearch/utilities/sample.hpp"      // `sample_plan` - the SAMPLE verb's roots, paths & regions
#include "usearch/basic/ram_index.hpp"
#include "usearch/basic/shard_checkpoints.hpp" // `save_shards`, `load_shards` - the checkpoints of every live shard

namespace unum::usearch::basic {

template <typename indexes_type_>
struct ram_indexes_cursor;

/**
 *  @brief Basic RAM shards viewed as one: insertions routed over the shard generations, searches
 *      merged across them, lookups fanned out.
 *
 *  @tparam index_type_ The shard, a @c basic::ram_index whose allocator a memory domain builds. Its
 *      waiter and atomic reference serve the shard table too, so the defaults hold: the standard
 *      reference posts what ForkUnion's spell natively through @c atomic_add. Its node IDs are 32
 *      bits wide, as the shard table locates every node in 8 bytes; a lone @c basic::ram_index
 *      takes either width.
 */
template <typename index_type_>
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct ram_indexes {
    friend struct ram_indexes_cursor<ram_indexes>;

    /** A reader's walk over every live shard's published entries: as stored in placement order,
     *  or breadth-first from the entries it opens on, each with the keys it links to. */
    using cursor_t = ram_indexes_cursor<ram_indexes>;

    using index_t = index_type_;

    /* Entries and scores */
    using key_t = typename index_t::key_t;
    using value_t = typename index_t::value_t;
    using node_id_t = typename index_t::node_id_t;
    using distance_t = typename index_t::distance_t;
    using scored_node_t = typename index_t::scored_node_t;

    /* Memory and waits */
    using allocator_t = typename index_t::allocator_t;
    using wait_t = typename index_t::wait_t;
    using traversal_context_t = typename index_t::traversal_context_t;

    /* Verb configs and results */
    using insert_config_t = typename index_t::insert_config_t;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using split_config_t = typename index_t::split_config_t;

    template <typename context_allocator_type_>
    using traversal_context_over = typename index_t::template traversal_context_over<context_allocator_type_>;

    template <typename word_type_>
    using atomic_ref = typename index_t::template atomic_ref<word_type_>;

    /** The shards and the rotation that routes new entries; the keys stay in their records. */
    using shards_t = shard_set<index_t, index_t::template atomic_ref, wait_t>;
    using shard_t = typename shards_t::shard_t;
    using batch_split_t = typename shards_t::batch_split_t;

  private:
    using shared_mutex_t = shared_mutex<wait_t, index_t::template atomic_ref>;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;
    using shard_insert_plan_t = typename index_t::insert_plan_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;
    using shard_refine_plan_t = typename index_t::refine_plan_t;

    /** Resolves a node ID walked in one shard to the key its record holds, as a predicate and a
     *  merge both read it. */
    struct key_of_node_t {
        index_t const* shard = nullptr;

        key_t operator()(node_id_t node_id) const noexcept { return shard->key_of(node_id).value; }
        key_t operator[](node_id_t node_id) const noexcept { return shard->key_of(node_id).value; }
    };

    /** The engine relation of a walk that brings none, for every shard alike. */
    struct unrelated_t {
        dummy_predicate_t operator()(index_t const&) const noexcept { return {}; }
    };

    /** Every shard as SAMPLE reads it below the watermarks of @c watermarks_snapshot, scored by the
     *  caller's metric; see @c sample_source. */
    template <typename metric_type_, typename context_type_, typename coordinates_type_>
    struct basic_sample_source {
        using key_t = typename index_t::key_t;
        using node_id_t = typename index_t::node_id_t;
        using shard_node_t = shard_node<node_id_t>;

        ram_indexes const& indexes;
        std::span<node_id_t const> watermarks_snapshot;
        metric_type_& metric;
        context_type_& context;
        coordinates_type_ const& fixed_coordinates_of;

        value_t const& value_(shard_node_t node) const noexcept {
            return indexes.shards_[node.shard].index->published_value_of(node.node);
        }

        std::span<node_id_t const> watermarks() const noexcept { return watermarks_snapshot; }

        key_t key_of(shard_node_t node) const noexcept {
            return indexes.shards_[node.shard].index->key_of(node.node).value;
        }

        template <typename callback_type_>
        void nodes_of(key_t const& key, callback_type_&& callback) const noexcept {
            for (std::size_t shard = 0; shard != watermarks_snapshot.size(); ++shard)
                indexes.shards_[shard].index->key_map_.find_each(
                    key, watermarks_snapshot[shard],
                    [&](node_id_t node) noexcept { callback(shard_node_t {static_cast<u32_t>(shard), node}); });
        }

        /** Copies the list under the node's lock into @c staged_ids, then calls back unlocked. */
        template <typename callback_type_>
        void neighbors(shard_node_t node, callback_type_&& callback) const noexcept {
            typename index_t::node_t& record = indexes.shards_[node.shard].index->nodes_[node.node];
            record.lock();
            std::size_t const degree = record.neighbors.size();
            context.staged_ids.clear();
            status_t const staged = context.staged_ids.reserve(degree);
            if (succeeded(staged))
                for (std::size_t link = 0; link != degree; ++link)
                    context.staged_ids.push_back_reserved(record.neighbors[link]);
            record.unlock();
            for (node_id_t const next : context.staged_ids)
                if (next < watermarks_snapshot[node.shard]) callback(shard_node_t {node.shard, next});
        }

        /** Scores in rounds as long as the metric stages, each reopened on the same query. */
        void distance(shard_node_t from, std::span<shard_node_t const> nodes,
                      std::span<f32_t> distances) const noexcept {
            value_t const& query = value_(from);
            std::array<distance_t, 64> drained;
            std::size_t staged = 0, first = 0;
            auto const drain = [&]() noexcept {
                if (!staged) return;
                metric.drain(std::span<distance_t>(drained.data(), staged), unbounded_radius<distance_t>());
                for (std::size_t lane = 0; lane != staged; ++lane)
                    distances[first + lane] = static_cast<f32_t>(drained[lane]);
                first += staged, staged = 0;
            };
            for (shard_node_t const node : nodes) {
                if (staged == drained.size()) drain();
                if (!staged) metric.reset(query);
                if (metric.push(value_(node)) == metric_staging_t::refused_k) {
                    drain();
                    metric.reset(query);
                    metric.push(value_(node));
                }
                ++staged;
            }
            drain();
        }

        std::size_t walk(shard_node_t from, shard_node_t toward, std::span<shard_node_t> chain) const noexcept {
            return sample_greedy_walk(*this, from, toward, chain);
        }

        void coordinates_of(shard_node_t node, std::span<f32_t, 3> xyz) const noexcept
            requires fixed_coordinates<coordinates_type_>
        {
            fixed_coordinates_of(value_(node), xyz);
        }
    };

  public:
#pragma region Placement and Measures

    /** Every index names the host allocator its shard table sits on and the line its shards
     *  prefetch by; nothing is placed by default. */
    explicit ram_indexes(forkunion_allocator_t allocator, std::size_t line_bytes = default_alignment_k) noexcept
        : shards_(allocator), line_bytes_(line_bytes) {}
    ram_indexes(ram_indexes const&) = delete;
    ram_indexes& operator=(ram_indexes const&) = delete;
    ram_indexes(ram_indexes&&) = delete;
    ram_indexes& operator=(ram_indexes&&) = delete;

    /** Bounds every shard at @p nodes, clamped to the ID space; zero means the ID space. Past it,
     *  an insertion opens another shard generation on the same placements. Ignored once storage
     *  exists. */
    void bound_shards(std::size_t nodes) noexcept { shards_.bound_shards(nodes); }

    /**
     *  Appends one empty shard placed on @p memory_domain; refused once storage exists.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index already holds storage, or has a shard generation in
     *    flight.
     */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(allocator_t {memory_domain});
        if (!shard) return status_t::bad_alloc_k;
        return shards_.emplace(std::move(shard));
    }

    /**
     *  Covers @p capacity entries split evenly over the shards, each under the bound - a hint,
     *  since the shards grow on demand; callable alongside every other verb.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed yet.
     *  - @c overflow_k A shard's share passes its node IDs.
     */
    expected<std::size_t> reserve(std::size_t capacity) noexcept { return shards_.reserve({capacity}); }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.entries(); }

    /** The epoch a reader starts under, waiting out a destructive window in progress. */
    u64_t read_epoch() const noexcept { return shards_.read_epoch(); }

    /** Whether no destructive window began since @p epoch, so everything read under it stands. */
    bool still_valid(u64_t epoch) const noexcept { return shards_.still_valid(epoch); }

    /** Shards in the index, every shard generation's and the retired ones, in placement order: the
     *  positions a snapshot fills. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** Shards past the retired ones, which every walk and checkpoint spans. */
    std::size_t live_shards_count() const noexcept { return shards_.size() - shards_.retired_shards(); }

    /** Entries the live shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** Bytes every shard's records and key map occupy, retired shards included. */
    std::size_t memory_usage() const noexcept { return shards_.memory_usage(); }

    /** The allocator a caller builds the values it inserts with: the first shard's, which moves
     *  them into a shard on the same domain without a copy; asked only once a shard is placed. */
    allocator_t get_allocator() const noexcept { return shards_[0].index->get_allocator(); }

    /** The shards past the retired ones, in placement order. */
    typename shards_t::template indexes_range<index_t> live_shards() noexcept { return shards_.live_indexes(); }
    typename shards_t::template indexes_range<index_t const> live_shards() const noexcept {
        return shards_.live_indexes();
    }

    /** One consistent watermark per shard into @p watermarks, the caller's buffer, so the searches
     *  below it see every batch whole or not at all; answers their sum. A buffer sized before a
     *  shard generation opened answers @c shape_mismatch_k, to be sized again.
     *
     *  @par Statuses:
     *  - @c shape_mismatch_k A non-empty @p watermarks holds other than one slot per shard.
     */
    expected<std::size_t> snapshot(std::span<node_id_t> watermarks) const noexcept {
        return shards_.snapshot(watermarks);
    }

    /**
     *  @brief Empties the index without freeing anything a reader may hold: a fresh shard
     *      generation on the same placements takes the next entries, and every shard before it
     *      retires whole.
     *
     *  Takes the gate alone and closes every shard first, so a call in flight, a batch holding
     *  claims or an absorb holding the index refuses it with @c contended_k, and an index holding
     *  no nodes stays as it is. The fresh shards open before any retires, so a refused build
     *  answers @c bad_alloc_k with every node in place. A search it overlaps reads values nothing
     *  frees, and retries.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, or a batch, refine or absorb holds a shard's claims.
     */
    status_t clear() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        if (status_t const closed = shards_.close(); failed(closed)) return closed;
        if (shards_.entries()) {
            if (status_t const opened = shards_.open_shard_generation_past_seal_(
                    shards_.open_shard_generation(),
                    [this](shard_t const& like) noexcept { return build_shard_(like); });
                failed(opened))
                return shards_.reopen(), opened;
            shards_.begin_window();
            shards_.retire_older_shard_generations_();
            shards_.end_window();
        }
        shards_.reopen();
        return status_t::success_k;
    }

#pragma endregion Placement and Measures

#pragma region Checkpoints

    /** What a checkpoint does on these shards' memory domains, for @ref save_shards and
     *  @c load_shards: every shard's own plans on the calling thread, its keys in its records,
     *  which those plans write and read, and a load's parts past the placed shards on shard
     *  generations built like them. */
    struct checkpoint_residency_t {
        ram_indexes* owner = nullptr;

        /**
         *  The counts a save of shard @p shard needs, as the shard's own @c columns_extents takes
         *  them.
         *
         *  @par Statuses:
         *  - @c contended_k A mutator or an open plan holds the shard, or a dispensed batch has yet
         *    to publish.
         */
        template <typename codec_type_>
        static expected<columns_extents_t> columns_extents(shards_t const& shards, std::size_t shard,
                                                           codec_type_ const& codec) noexcept {
            return shards[shard].index->columns_extents(codec);
        }

        /**
         *  Opens the save plan of shard @p shard into @p columns, the shard's own @c plan_save.
         *
         *  @par Statuses:
         *  - @c contended_k A mutator or an open plan holds the shard, or a dispensed batch has yet
         *    to publish.
         *  - @c overflow_k The key or a prefix-sum column is shorter than its count, or the
         *    neighbor IDs or value bytes are shorter than the sums need.
         *  - @c config_mismatch_k The key or a prefix-sum column is longer than its count.
         */
        template <typename codec_type_>
        static expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                       typename index_t::columns_t columns,
                                                                       codec_type_ const& codec) noexcept {
            return shards[shard].index->plan_save(columns, codec);
        }

        /**
         *  Opens the load plan of shard @p shard from @p columns, the shard's own @c plan_load.
         *
         *  @par Statuses:
         *  - @c contended_k A mutator or an open plan holds the shard.
         *  - @c config_mismatch_k The shard already publishes nodes.
         *  - @c malformed_k A prefix-sum column is not one longer than the keys, does not rise from
         *    zero to its element column's length, or a neighbor ID names no node.
         *  - @c overflow_k The nodes pass the shard's node IDs, or a node's capacity wraps.
         */
        static expected<typename index_t::load_plan_t> open_shard_load(
            shards_t& shards, std::size_t shard, typename index_t::columns_view_t columns) noexcept {
            return shards[shard].index->plan_load(columns);
        }

        /**
         *  Covers @p entries in shard @p shard under the set's shard bound.
         *
         *  @par Statuses:
         *  - @c overflow_k @p entries passes the shard's node IDs.
         */
        static expected<std::size_t> reserve(shards_t& shards, std::size_t shard, std::size_t entries) noexcept {
            return shards[shard].index->reserve({entries, shards.shard_capacity()});
        }

        /**
         *  Leaves part @p part the live shard it names, a placed one, or opens shard generations on
         *  the placed shards' memory domains until it exists.
         *
         *  @par Statuses:
         *  - @c contended_k The set is sealed while a shard generation must open.
         */
        status_t open_shard_for_part(shards_t& shards, std::size_t part) noexcept {
            while (shards.size() <= shards.retired_shards() + part)
                if (status_t const opened = shards.open_shard_generation_(
                        shards.open_shard_generation(),
                        [this](shard_t const& like) noexcept { return owner->build_shard_(like); });
                    failed(opened))
                    return opened;
            return status_t::success_k;
        }
    };

    /** Writes the live shards under @p path through @c save_shards: a lone live shard's monolith,
     *  or one part per live shard beside the manifest @p path names. Takes the gate alone, so a
     *  call in flight refuses it with @c contended_k. A refusal of @p codec passes through.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, a mutator or an open plan holds a shard, or the
     *    pool of @p executor runs another dispatch.
     *  - @c uninitialized_k No shard is live, or the pool of @p executor has no workers.
     *  - @c config_mismatch_k @p config does not validate, an element has no interchange dtype, a
     *    value disagrees with its shape, or a segment is misaligned.
     *  - @c invalid_argument_k A part or manifest path does not parse.
     *  - @c overflow_k A device ends before a part or the manifest, or a column is shorter than
     *    the layout needs.
     *  - @c malformed_k A written part's header does not parse back.
     */
    template <typename codec_type_, typename executor_type_, typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, codec_type_ const& codec, executor_type_& executor,
                  save_config_t const& config, std::string_view metric, progress_type_&& progress = {}) noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        return save_shards(shards_, checkpoint_residency_t {this}, path, codec, executor, config, metric,
                           std::forward<progress_type_>(progress));
    }

    /** Reads the monolith or manifest at @p path into the placed, empty shards, part @c p into live
     *  shard @c p, opened past the placed ones over their memory domains in turn. Takes the gate
     *  alone, as @ref save does. A refusal of @p codec passes through.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, the set is sealed while a shard generation must
     *    open, a mutator or plan holds a shard, or the pool of @p executor runs another dispatch.
     *  - @c uninitialized_k No shard is placed, or the pool of @p executor has no workers.
     *  - @c config_mismatch_k The shards hold entries, a column's dtype, shape or key width
     *    differs from what the shards and @p codec take, or a segment is misaligned.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, or a part, its header, columns or sums do
     *    not parse.
     *  - @c overflow_k The nodes pass a shard's node IDs or key multimap.
     */
    template <typename codec_type_, typename executor_type_, typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, codec_type_ const& codec, executor_type_& executor,
                  load_config_t const& config, progress_type_&& progress = {}) noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        return load_shards(shards_, checkpoint_residency_t {this}, path, codec, executor, config,
                           std::forward<progress_type_>(progress));
    }

    /**
     *  @brief Bytes a buffer @ref save under @p config writes: a lone live shard's monolith, or the
     *      manifest and every live shard's part it lists.
     *
     *  Sizing goes through @c shard_storage_requirements and opens no shard, so the residency it
     *  takes needs no owner.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is live.
     *  - @c contended_k A mutator or an open plan holds a shard.
     *  - @c config_mismatch_k @p config does not validate, or an element has no interchange dtype.
     */
    template <typename codec_type_>
    expected<std::size_t> storage_requirements(codec_type_ const& codec, save_config_t const& config,
                                               std::string_view metric) const noexcept {
        return shard_storage_requirements(shards_, checkpoint_residency_t {}, codec, config, metric);
    }

    /**
     *  @brief Writes the live shards into @p buffer through @c save_shards, sized by
     *      @ref storage_requirements, taking the gate as the path form does.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, a mutator or an open plan holds a shard, or the
     *    pool of @p executor runs another dispatch.
     *  - @c uninitialized_k No shard is live, or the pool of @p executor has no workers.
     *  - @c config_mismatch_k @p config does not validate, an element has no interchange dtype, a
     *    value disagrees with its shape, or a segment is misaligned.
     *  - @c overflow_k @p buffer is shorter than the checkpoint.
     */
    template <typename codec_type_, typename executor_type_, typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t buffer, codec_type_ const& codec, executor_type_& executor, save_config_t const& config,
                  std::string_view metric, progress_type_&& progress = {}) noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        return save_shards(shards_, checkpoint_residency_t {this}, buffer, codec, executor, config, metric,
                           std::forward<progress_type_>(progress));
    }

    /**
     *  @brief Reads the monolith or manifest-led checkpoint in @p buffer into the placed, empty
     *      shards through @c load_shards, as the path form does.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, the set is sealed while a shard generation must
     *    open, a mutator or plan holds a shard, or the pool of @p executor runs another dispatch.
     *  - @c uninitialized_k No shard is placed, or the pool of @p executor has no workers.
     *  - @c config_mismatch_k The shards hold entries, a column's dtype, shape or key width
     *    differs from what the shards and @p codec take, or a segment is misaligned.
     *  - @c malformed_k The buffer lists no part, or a part, its header, columns or sums do not
     *    parse.
     *  - @c overflow_k The nodes pass a shard's node IDs or key multimap.
     */
    template <typename codec_type_, typename executor_type_, typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t buffer, codec_type_ const& codec, executor_type_& executor, load_config_t const& config,
                  progress_type_&& progress = {}) noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        return load_shards(shards_, checkpoint_residency_t {this}, buffer, codec, executor, config,
                           std::forward<progress_type_>(progress));
    }

#pragma endregion Checkpoints

#pragma region Insertion

    /**
     *  @brief One batch in flight across the shards: one shard plan per shard of the open shard
     *      generation and of every shard generation the batch opened, so its entries can only be
     *      placed, and @ref finish publishes them all at once.
     *
     *  Entries are split across the open shard generation's shards round-robin from the starting
     *  shard on, and entries past what fits under the bound fill fresh shard generations in turn,
     *  split the same way. Every entry must be placed before @ref finish, since each shard plan
     *  holds its shard's later turns until then. Each step and @ref finish hold the gate's shared
     *  side for their own call.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : shard_plans_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : shard_plans_(std::move(other.shard_plans_)), batch_split_(other.batch_split_),
              owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /**
         *  @brief Places entries @p first onward, one per value of @p values under its key in
         *      @p keys, each in the shard its position falls to, as the shard plan's own step does.
         *
         *  Any number of threads step one plan at once over disjoint entries, each with its own
         *  @p metric and a @p context that @c index_t::reserve_traversal sized. Entries outside the
         *  plan or keys of another length place nothing, and @ref finish then refuses.
         */
        template <synchronization_t synchronization_ = synchronization_t::concurrent_k, typename metric_type_,
                  typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
        void insert(std::size_t first, std::span<value_t> values, std::span<key_t const> keys, metric_type_& metric,
                    traversal_context_over<context_allocator_type_>& context, selector_type_ selector = {},
                    insert_config_t config = {}) noexcept {
            std::size_t const count = values.size();
            if (!owner_ || keys.size() != count || first > batch_split_.batch_entries ||
                count > batch_split_.batch_entries - first)
                return;
            shared_lock_t const held {owner_->gate_};
            for (std::size_t position = 0; position != count; ++position) {
                entry_in_shard_t const target = batch_split_.shard_and_offset_of_entry(first + position);
                shard_plans_[target.shard].template insert<synchronization_>(target.offset, values.subspan(position, 1),
                                                                             keys.subspan(position, 1), metric, context,
                                                                             selector, config);
            }
        }

        /**
         *  @brief Publishes every entry at once: @c config_mismatch_k while any has yet to be
         *      placed, leaving the plan whole; then every shard's turn inside one commit, so a
         *      snapshot sees the batch whole or not at all.
         *
         *  The plan is spent either way, a shard refusing its turn answering the first refusal
         *  after every other shard published, so a second call answers @c uninitialized_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k An entry has yet to be placed.
         */
        status_t finish() noexcept {
            if (!owner_) return status_t::uninitialized_k;
            std::size_t touched = 0;
            for (shard_insert_plan_t& shard_plan : shard_plans_) {
                if (atomic_ref<std::size_t>(shard_plan.placed).load(std::memory_order_acquire) !=
                    shard_plan.range.size())
                    return status_t::config_mismatch_k;
                touched += !shard_plan.range.empty();
            }
            shared_lock_t const held {owner_->gate_};
            // Shards publish in order, so a turn waited for here is one an earlier shard of another
            // batch already published, and no two batches wait on each other.
            status_t status = status_t::success_k;
            if (touched > 1) owner_->shards_.begin_commit();
            for (shard_insert_plan_t& shard_plan : shard_plans_)
                if (!shard_plan.range.empty())
                    if (status_t const published = shard_plan.finish(); failed(published) && succeeded(status))
                        status = published;
            if (touched > 1) owner_->shards_.end_commit();
            owner_ = nullptr;
            return status;
        }

        /** Invokes @p callback as <tt>callback(shard, range)</tt> for every shard the plan places entries
         *  in, with the IDs dispensed there, in placement order; before @ref finish, so a caller
         *  can keep what it follows the published nodes up with, as long links. */
        template <typename callback_type_>
        void for_each_range(callback_type_&& callback) const noexcept {
            if (!owner_) return;
            for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                if (!shard_plans_[shard].range.empty())
                    callback(*owner_->shards_[batch_split_.open_shard_generation.first + shard].index,
                             shard_plans_[shard].range);
        }

      private:
        friend ram_indexes;

        /** One plan per shard from the open shard generation's first, in placement order; an empty
         *  one where no entry goes. */
        growing_buffer<shard_insert_plan_t, forkunion_allocator_t> shard_plans_;

        /** The batch split across the shard generation open at planning, which the shard plans
         *  start at, and the fresh ones it opened. */
        batch_split_t batch_split_ {};

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once @c finish published. */
        ram_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens a batch of @p count entries: routes them over the open shard generation and as
     *      many fresh ones as the bound makes them fill, then opens one shard plan per shard taking
     *      entries, in placement order.
     *
     *  The rotation comes from the entries routed so far, which keeps a stream of small batches
     *  from drifting onto shard zero. Every shard claims its entries before any dispenses, so a
     *  refusal gives back every claim and strands no ID. Everything fallible past the shard plans
     *  belongs before this call, as for a shard's own plan: building the values and sizing each
     *  stepping thread's context.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard placed yet.
     *  - @c contended_k An absorb holds the index closed.
     *  - @c overflow_k A racing batch took the room this one was routed into.
     */
    expected<insert_plan_t> plan_insert(std::size_t count) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        shared_lock_t const held {gate_};
        if (shards_.sealed()) return {{}, status_t::contended_k};
        insert_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        if (!count) return {std::move(plan), status_t::success_k};
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            count, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.batch_split_ = batch_split;

        // Every shard claims before any dispenses, so a refusal gives back the claims before it as
        // they die, and no shard is left holding IDs that would never publish.
        std::size_t const shards = batch_split.shards_spanned();
        growing_buffer<shard_capacity_claim_t, forkunion_allocator_t> claims {shards_.allocator()};
        if (failed(claims.reserve_exactly(shards)) || failed(plan.shard_plans_.reserve_exactly(shards)))
            return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const shard_entries = batch_split.entries_for_shard(shard);
            if (!shard_entries) {
                claims.push_back_reserved(shard_capacity_claim_t {});
                continue;
            }
            auto [claim,
                  status] = shards_[batch_split.open_shard_generation.first + shard].index->claim_(shard_entries);
            if (failed(status)) return {{}, status};
            claims.push_back_reserved(std::move(claim));
        }
        for (std::size_t shard = 0; shard != shards; ++shard)
            plan.shard_plans_.push_back_reserved(
                shards_[batch_split.open_shard_generation.first + shard].index->dispense_(std::move(claims[shard])));
        return {std::move(plan), status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Search and Lookups

    /**
     *  @brief One query's nearest keys over every shard @p watermarks counts, each walked below its
     *      watermark as @c index_t::search_if walks, merged ascending into @p keys and
     *      @p distances.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[in] watermarks One per shard from @ref snapshot.
     *  @param[out] keys The merged ranking's keys, as many as it holds.
     *  @param[out] distances The same ranking's distances, parallel to @p keys.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from each shard's connectivity".
     *  @param[in] relate Builds the engine's own relation over one shard's node IDs, called as
     *      <tt>relate(shard) → relation</tt>, or @c dummy_predicate_t where it relates every node.
     *  @param[in,out] predicate The caller's predicate over keys, asked about each shard's walked
     *      nodes through their records, or @c dummy_predicate_t for an unpredicated walk.
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry over any allocator.
     *  @return The matches merged.
     *
     *  @par Statuses:
     *  - @c contended_k Two windows overlapped the call or one outran the snapshot.
     */
    template <typename query_type_, typename metric_type_, typename relate_type_, typename predicate_type_,
              typename context_allocator_type_>
    expected<std::size_t> search_if(query_type_ const& query, std::span<node_id_t const> watermarks,
                                    std::span<key_t> keys, std::span<distance_t> distances, search_config_t config,
                                    relate_type_ const& relate, predicate_type_& predicate, metric_type_& metric,
                                    traversal_context_over<context_allocator_type_>& context) noexcept {
        using relation_t = std::invoke_result_t<relate_type_ const&, index_t const&>;
        return merge_shards_(
            watermarks, keys, distances,
            [&](index_t& one, node_id_t watermark, std::size_t) noexcept -> expected<std::span<scored_node_t const>> {
                if constexpr (std::is_same_v<relation_t, dummy_predicate_t> &&
                              std::is_same_v<predicate_type_, dummy_predicate_t>) {
                    status_t const walked = one.template fold_walk_below_<synchronization_t::concurrent_k>(
                                                   query, metric, dummy_fold_t {}, context.results, context, config,
                                                   watermark)
                                                .status;
                    return {ranking_of_(context.results), walked};
                }
                else {
                    relation_t relation = relate(one);
                    node_predicate admits {predicate, key_of_node_t {&one}};
                    context.admitted.clear();
                    predicate_fold fold {relation, admits, context.admitted, one.non_zero_expansion_factor(config)};
                    status_t const walked = one.template fold_walk_below_<synchronization_t::concurrent_k>(
                                                   query, metric, fold, context.admitted, context, config, watermark)
                                                .status;
                    return {ranking_of_(context.admitted), walked};
                }
            });
    }

    /**
     *  @ref search_if with no engine relation of its own.
     *
     *  @par Statuses:
     *  - @c contended_k Two windows overlapped the call or one outran the snapshot.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     */
    template <typename query_type_, typename metric_type_, typename predicate_type_, typename context_allocator_type_>
    expected<std::size_t> search_if(query_type_ const& query, std::span<node_id_t const> watermarks,
                                    std::span<key_t> keys, std::span<distance_t> distances, search_config_t config,
                                    predicate_type_& predicate, metric_type_& metric,
                                    traversal_context_over<context_allocator_type_>& context) noexcept {
        return search_if(query, watermarks, keys, distances, config, unrelated_t {}, predicate, metric, context);
    }

    /** The exact sibling of @c search_if: every entry below each shard's watermark scored as
     *  @c index_t::scan scores, the nearest @p keys can hold kept, as @p relate and @p predicate
     *  admit them. Each shard's rounds drain within the merged ranking's worst match once it is
     *  full, so later shards score fewer entries.
     *
     *  @par Statuses:
     *  - @c contended_k Two windows overlapped the call or one outran the snapshot.
     */
    template <typename query_type_, typename metric_type_, typename relate_type_, typename predicate_type_,
              typename context_allocator_type_>
    expected<std::size_t> scan(query_type_ const& query, std::span<node_id_t const> watermarks, std::span<key_t> keys,
                               std::span<distance_t> distances, relate_type_ const& relate, predicate_type_& predicate,
                               metric_type_& metric,
                               traversal_context_over<context_allocator_type_>& context) noexcept {
        using relation_t = std::invoke_result_t<relate_type_ const&, index_t const&>;
        std::size_t const wanted = min_of(keys.size(), distances.size());
        if (!wanted) return {0, status_t::success_k};
        return merge_shards_(watermarks, keys, distances,
                             [&](index_t& one, node_id_t watermark,
                                 std::size_t merged) noexcept -> expected<std::span<scored_node_t const>> {
                                 context.results.clear();
                                 relation_t relation = relate(one);
                                 node_predicate admits {predicate, key_of_node_t {&one}};
                                 predicate_fold fold {relation, admits, context.results, wanted};
                                 auto const radius = [&]() noexcept {
                                     distance_t bound = merged == wanted ? distances[wanted - 1]
                                                                         : unbounded_radius<distance_t>();
                                     if (context.results.size() == wanted)
                                         bound = min_of(bound, context.results.top().distance);
                                     return bound;
                                 };
                                 status_t const scanned =
                                     one.scan_below_(query, metric, radius, fold, context, wanted, watermark).status;
                                 return {ranking_of_(context.results), scanned};
                             });
    }

    /**
     *  @ref scan with no engine relation of its own.
     *
     *  @par Statuses:
     *  - @c contended_k Two windows overlapped the call or one outran the snapshot.
     */
    template <typename query_type_, typename metric_type_, typename predicate_type_, typename context_allocator_type_>
    expected<std::size_t> scan(query_type_ const& query, std::span<node_id_t const> watermarks, std::span<key_t> keys,
                               std::span<distance_t> distances, predicate_type_& predicate, metric_type_& metric,
                               traversal_context_over<context_allocator_type_>& context) noexcept {
        return scan(query, watermarks, keys, distances, unrelated_t {}, predicate, metric, context);
    }

    /** Entries under @p key across every live shard, read between destructive windows and between
     *  commits, so a batch spanning shards counts whole or not at all. */
    std::size_t count(key_t const& key) const noexcept { return shards_.count(key); }

    /** Invokes @p callback as <tt>callback(shard, node_id)</tt> for every published entry under
     *  @p key in every live shard; an ID a clear retired meanwhile is refused by every accessor of
     *  its shard. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        shards_.find_each(key, callback);
    }

    /**
     *  @brief Draws up to @p outputs' length entries around the focus keys, or across every shard
     *      @p watermarks counts without any, and assigns them three coordinates among the shown
     *      points of @p inputs.
     *
     *  The roots, paths and regions of @c sample_plan, scored by @p metric: each draw names a key
     *  and its ordinal among that key's entries as @ref find_each orders them, the shown point or
     *  earlier draw it hangs from, and its coordinates; a shorter @p outputs draws a prefix of a
     *  longer one. Without shown points the draws take @p coordinates_of where it is given, else a
     *  classical scaling among themselves. The scratch is one uninitialized buffer on @p context 's
     *  allocator, and neighbor lists stage in its @c staged_ids.
     *
     *  @param[in] inputs The shown points, their coordinates and the focus keys.
     *  @param[in] watermarks One per shard from @ref snapshot.
     *  @param[out] outputs The draws, as many as its keys hold.
     *  @param[in] config The seed and the regions' breadth.
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch over any allocator.
     *  @param[in] coordinates_of Writes a stored value's fixed coordinates, as
     *      <tt>void(value_t const&, std::span&lt;f32_t, 3&gt;)</tt>, or @c no_fixed_coordinates_t
     *      to solve them from distances.
     *
     *  @par Statuses:
     *  - @c shape_mismatch_k An output's or the shown coordinates' length disagrees.
     *  - @c bad_alloc_k The scratch cannot be allocated.
     *  - @c contended_k Two windows overlapped the call or one outran the snapshot.
     */
    template <typename metric_type_, typename context_allocator_type_,
              typename coordinates_type_ = no_fixed_coordinates_t>
    expected<sample_result_t> sample(sample_inputs<key_t> inputs, std::span<node_id_t const> watermarks,
                                     sample_outputs<key_t> outputs, sample_config_t config, metric_type_& metric,
                                     traversal_context_over<context_allocator_type_>& context,
                                     coordinates_type_ const& coordinates_of = {}) noexcept {
        using source_t =
            basic_sample_source<metric_type_, traversal_context_over<context_allocator_type_>, coordinates_type_>;
        using plan_t = sample_plan<source_t>;
        std::size_t const scratch_bytes = plan_t::scratch_bytes(outputs.keys.size(), inputs.shown.size(), config);
        growing_buffer<u64_t, context_allocator_type_> scratch {context.allocator};
        if (failed(scratch.resize_uninitialized(divide_round_up(scratch_bytes + default_alignment_k, sizeof(u64_t)))))
            return {{}, status_t::bad_alloc_k};
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(scratch.data());
        byte_t* const aligned = reinterpret_cast<byte_t*>(
            round_up_to_multiple<std::uintptr_t>(address, default_alignment_k));
        return shards_.read_retrying_once([&]() noexcept -> expected<sample_result_t> {
            if (watermarks.size() > shards_.size()) return {{}, status_t::contended_k};
            for (std::size_t shard = 0; shard != watermarks.size(); ++shard)
                if (watermarks[shard] > shards_[shard].index->size()) return {{}, status_t::contended_k};
            source_t const source {*this, watermarks, metric, context, coordinates_of};
            plan_t plan {source, inputs, outputs, config, {aligned, scratch_bytes}};
            status_t status = plan.draw();
            if (succeeded(status)) status = plan.assign_ordinals();
            if (succeeded(status)) status = plan.assign_coordinates();
            return {plan.result(), status};
        });
    }

#pragma endregion Search and Lookups

#pragma region Refinement

    /**
     *  @brief One absorb or split in flight: every donor shard moves into a shard of this index
     *      through that shard's own plan, or every shard of this index splits into the same shard
     *      of every receiver, and each shard plan then re-selects across what moved in and heals.
     *
     *  Steps of a phase are (shard, position) pairs over the shard plans' own steps, so any number
     *  of threads may @ref run them at once; @ref advance is the single-caller barrier, advancing
     *  every shard plan together. While any shard plan copies, nodes move out of the donors, or out
     *  of this index in a split, so their readers wait out a window spanning the copy phases, and
     *  the receiving index publishes every shard's nodes inside one commit. The plan holds every
     *  donor's unique side, or this index's and every receiver's in a split, and this index sealed
     *  for its whole life, and its death reopens this index, every receiver and every donor not
     *  consumed: one whose every shard moved in stays empty, closed and sealed.
     */
    struct [[nodiscard]] refine_plan_t {

        /** One step: the shard whose plan runs it, and its position among that plan's steps.
         *  Two words. */
        struct step_t {
            std::size_t shard = 0;
            std::size_t position = 0;
        };
        static_assert(sizeof(step_t) == 2 * sizeof(std::size_t));

        /** What the shard plans' audits judge, summed: the published nodes. */
        struct audit_t {
            std::size_t visible = 0;
        };

        /** Steps of the current phase a driver already ran, so a cancelled call resumes at the next
         *  one. */
        std::size_t taken = 0;

        /** Nodes the planned phases re-selected so far, which a driver's progress reports. */
        std::size_t ticked = 0;

        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An unbound plan over @p allocator; only @ref plan_absorb and @ref plan_split fill
         *  it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : shard_plans_(allocator), donor_slots_(allocator), donor_holds_(allocator), receiver_shards_(allocator),
              receiver_holds_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : taken(other.taken), ticked(other.ticked), shard_plans_(std::move(other.shard_plans_)),
              donor_slots_(std::move(other.donor_slots_)), donor_holds_(std::move(other.donor_holds_)),
              donors_(other.donors_), receiver_shards_(std::move(other.receiver_shards_)),
              receiver_holds_(std::move(other.receiver_holds_)), receivers_(other.receivers_),
              owner_hold_(std::move(other.owner_hold_)), owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;
        ~refine_plan_t() noexcept { release_(); }

        /** Step @p position of the current phase; past the last shard beyond the end. */
        step_t operator[](std::size_t position) const noexcept {
            for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                std::size_t const steps = shard_plans_[shard].steps().size();
                if (position < steps) return {shard, position};
                position -= steps;
            }
            return {shard_plans_.size(), 0};
        }

        /** The current phase's steps over every shard plan, positionally. */
        indexed_range<refine_plan_t> steps() const noexcept {
            std::size_t total = 0;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) total += shard_plan.steps().size();
            return {this, total};
        }

        /**
         *  Runs @p step on the calling thread through its shard's plan, as that plan's own
         *  @c run does; a step past the end runs nothing.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The step's shard plan is bound to no index.
         */
        template <typename metric_type_, typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
        status_t run(step_t const& step, metric_type_& metric, traversal_context_over<context_allocator_type_>& context,
                     selector_type_ const& selector = {}) noexcept {
            if (step.shard >= shard_plans_.size()) return status_t::success_k;
            shard_refine_plan_t& shard_plan = shard_plans_[step.shard];
            return shard_plan.run(shard_plan[step.position], metric, context, selector);
        }

        /** The nodes @p step re-selects or sweeps, which a driver's progress counts. */
        std::size_t nodes_in(step_t const& step) const noexcept {
            if (step.shard >= shard_plans_.size()) return 0;
            return shard_plans_[step.shard][step.position].nodes.size();
        }

        /** The barrier between phases, one caller after every step of the current one ran: advances
         *  every shard plan, publishing what moved inside one commit of every receiving index, and
         *  answers whether any has another phase. */
        bool advance() noexcept {
            if (!owner_) return false;
            std::size_t const moving_before = moving_();
            if (moving_before) for_each_receiving_([](shards_t& shards) noexcept { shards.begin_commit(); });
            std::size_t follows = 0;
            for (shard_refine_plan_t& shard_plan : shard_plans_) follows += shard_plan.advance();
            if (moving_before) for_each_receiving_([](shards_t& shards) noexcept { shards.end_commit(); });
            if (moving_before && !moving_()) {
                end_windows_();
                release_consumed_donors_();
            }
            return follows != 0;
        }

        /** Stops every shard plan's re-selection where its steps ran, so the next @ref advance
         *  turns to the heal; a no-op for a plan still moving. */
        void stop() noexcept {
            for (shard_refine_plan_t& shard_plan : shard_plans_) shard_plan.stop();
            taken = steps().size();
        }

        /** The published nodes every shard plan's audit judges. */
        audit_t audit() const noexcept {
            audit_t total;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) total.visible += shard_plan.audit.visible;
            return total;
        }

        /** Nodes every shard plan's re-selection rounds step over, summed: the total a driver's
         *  progress counts against. */
        std::size_t planned_nodes() const noexcept {
            std::size_t total = 0;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) total += shard_plan.planned_nodes();
            return total;
        }

        /** The budget planning reserved every neighbor list for, which every shard shares. */
        std::size_t connectivity() const noexcept {
            std::size_t widest = 0;
            for (shard_refine_plan_t const& shard_plan : shard_plans_)
                widest = max_of(widest, shard_plan.connectivity());
            return widest;
        }

        /** The walks' knobs every shard plan shares: the widest breadth any of them walks. */
        insert_config_t walk() const noexcept {
            insert_config_t widest {};
            for (shard_refine_plan_t const& shard_plan : shard_plans_)
                widest.expansion_factor = max_of(widest.expansion_factor, shard_plan.walk().expansion_factor);
            return widest;
        }

        /** Whether any shard plan re-selects its planned range, which a caller's progress
         *  counts. */
        bool reselecting() const noexcept {
            for (shard_refine_plan_t const& shard_plan : shard_plans_)
                if (shard_plan.reselecting()) return true;
            return false;
        }

        /** Whether any shard plan still copies a donor, which no stop skips. */
        bool moving() const noexcept { return moving_() != 0; }

        /** Whether every live shard of the donor at @p donor moved in, consuming it: once the copy
         *  phases end, the plan drops its hold and nulls its slot, so its owner may free it at
         *  once; it stays closed and sealed for good. */
        bool consumed(std::size_t donor) const noexcept { return donor < donors_.size() && !donors_[donor]; }

        /**
         *  @brief Invokes @p visit as <tt>visit(shard, nodes)</tt> for every node range the plan
         *      rewrote: every node of every shard a shard plan refined, or every node of every
         *      receiver shard that a split filled.
         *
         *  Called once the plan finished and before it dies, while the index stays sealed, so a
         *  rule the re-selection skips, like a modality's long links, is drawn again over them.
         */
        template <typename visit_type_>
        void for_each_shard_range(visit_type_&& visit) const noexcept {
            if (!owner_) return;
            for (index_t* receiver_shard : receiver_shards_)
                visit(*receiver_shard, iota_view<node_id_t> {0, static_cast<node_id_t>(receiver_shard->size())});
            if (!receivers_.empty()) return;
            for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                // A shard no donor moved into, and a retired one, has no plan of its own.
                if (shard_plans_[shard].finish().status == status_t::uninitialized_k) continue;
                index_t& one = *owner_->shards_[shard].index;
                visit(one, iota_view<node_id_t> {0, static_cast<node_id_t>(one.size())});
            }
        }

        /**
         *  The pass so far, summed over the shard plans: complete once @ref advance answered false,
         *  otherwise @c cancelled_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         *  - @c cancelled_k A shard plan stopped before @ref advance answered false.
         */
        expected<refine_report_t> finish() const noexcept {
            refine_report_t total;
            status_t status = owner_ ? status_t::success_k : status_t::uninitialized_k;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) {
                auto const [tally, tally_status] = shard_plan.finish();
                // A shard no donor moved into has no plan of its own.
                if (tally_status == status_t::uninitialized_k) continue;
                if (failed(tally_status) && succeeded(status)) status = tally_status;
                total += tally;
            }
            return {total, status};
        }

      private:
        friend ram_indexes;

        /** Shard plans in their copy phases; the donors' windows stay open while any is. */
        std::size_t moving_() const noexcept {
            std::size_t moving = 0;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) moving += shard_plan.moving();
            return moving;
        }

        /** Calls @p visit with the shard set of every index the moves publish into: this one
         *  in an absorb, every receiver in a split. */
        template <typename visit_type_>
        void for_each_receiving_(visit_type_&& visit) noexcept {
            if (receivers_.empty()) visit(owner_->shards_);
            for (ram_indexes* receiver : receivers_) visit(receiver->shards_);
        }

        /** Opens the windows of every index the moves empty: the donors, or this one in a
         *  split. */
        void begin_windows_() noexcept {
            for (std::size_t donor = 0; donor != donor_holds_.size(); ++donor) donors_[donor]->shards_.begin_window();
            if (!receivers_.empty()) owner_->shards_.begin_window();
        }

        void end_windows_() noexcept {
            for (std::size_t donor = 0; donor != donor_holds_.size(); ++donor) donors_[donor]->shards_.end_window();
            if (!receivers_.empty()) owner_->shards_.end_window();
        }

        /** Closes any copy window, drops the shard plans, which reopen their shards and release
         *  the donor and receiver shards, reopens every donor not consumed, every receiver closed,
         *  then this index, and drops every hold; a consumed donor stays closed and sealed. */
        void release_() noexcept {
            if (moving_()) end_windows_();
            shard_plans_.clear();
            std::size_t const donor_shards_count = donor_shards_count_();
            for (std::size_t held = 0; held != donor_holds_.size(); ++held)
                if (donors_[held] && !consumed_(*donors_[held], donor_shards_count)) donors_[held]->shards_.reopen();
            donor_holds_.clear();
            for (std::size_t held = 0; held != receiver_holds_.size(); ++held) receivers_[held]->shards_.reopen();
            receiver_holds_.clear();
            if (owner_) std::exchange(owner_, nullptr)->shards_.reopen();
            owner_hold_ = {};
        }

        /** Once the copy phases end, drops the hold of every donor whose shards all moved in and
         *  nulls its slot; consumption is judged for every donor before any slot nulls. */
        void release_consumed_donors_() noexcept {
            std::size_t const donor_shards_count = donor_shards_count_();
            for (std::size_t held = 0; held != donor_holds_.size(); ++held)
                if (consumed_(*donors_[held], donor_shards_count)) {
                    donor_holds_[held].unlock();
                    donors_[held] = nullptr;
                }
        }

        /** The live shards of the donors the plan still holds. */
        std::size_t donor_shards_count_() const noexcept {
            std::size_t donor_shards_count = 0;
            for (std::size_t held = 0; held != donor_holds_.size(); ++held)
                if (donors_[held])
                    donor_shards_count += donors_[held]->shards_.size() - donors_[held]->shards_.retired_shards();
            return donor_shards_count;
        }

        /** Whether every live shard of @p donor moved in: planning slotted every one of the
         *  @p donor_shards_count donor shards, and each of this donor's slots nulled as its shard
         *  moved in. A planning refused midway slotted only some, and moved none, and a donor never
         *  placed has nothing to move. */
        bool consumed_(ram_indexes const& donor, std::size_t donor_shards_count) const noexcept {
            if (donor.shards_.empty() || donor_slots_.size() != donor_shards_count) return false;
            for (shard_t const& one : donor.shards_)
                for (index_t const* slot : donor_slots_)
                    if (slot == one.index.get()) return false;
            return true;
        }

        /** One plan per shard of this index, unbound where no donor shard moves in. */
        growing_buffer<shard_refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** Every donor shard grouped by the shard it moves into, each group a shard plan's donors,
         *  nulled as they move in; sized once, so the groups never move. */
        growing_buffer<index_t*, forkunion_allocator_t> donor_slots_;

        /** Every donor's unique side, held and closed in donor order. */
        growing_buffer<unique_lock_t, forkunion_allocator_t> donor_holds_;

        /** The caller's donors, which must outlive the plan. */
        std::span<ram_indexes*> donors_ {};

        /** Per live shard of this index, the same live shard of every receiver in receiver order:
         *  what that shard's split scatters into; sized once, so the groups never move. */
        growing_buffer<index_t*, forkunion_allocator_t> receiver_shards_;

        /** Every receiver's unique side, held and closed in receiver order. */
        growing_buffer<unique_lock_t, forkunion_allocator_t> receiver_holds_;

        /** The caller's receivers of a split, which must outlive the plan; empty otherwise. */
        std::span<ram_indexes*> receivers_ {};

        /** This index's unique side, held for a split's whole life. */
        unique_lock_t owner_hold_;

        /** The index absorbing or splitting, sealed while the plan lives; null in an unbound
         *  plan. */
        ram_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens an absorb of @p donors, moving every donor shard into a shard of this index
     *      through that shard's own absorb.
     *
     *  Closes every donor under its unique side, which the plan keeps, and this index under its
     *  own, then packs every live donor shard into a live shard of this index under the bound,
     *  opening shard generations as they fill. A donor every shard of which moved in is consumed:
     *  it stays closed and sealed once the plan dies, refusing insertions and clears with
     *  @c contended_k.
     *
     *  @param[in,out] donors The indexes giving their nodes up; the slots must outlive the plan.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     *  @param[in] selector Carries the budget every neighbor list is reserved for.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the shard
     *      plans.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A null donor, or this index itself.
     *  - @c contended_k A call or batch in flight on this index or a donor, or a repeated donor.
     *  - @c unsupported_k A donor shard past the bound, which would have to be cut first.
     */
    template <typename selector_type_>
    expected<refine_plan_t> plan_absorb(std::span<ram_indexes*> donors, refine_config_t config,
                                        selector_type_ const& selector, traversal_context_t& context) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        for (ram_indexes const* donor : donors)
            if (!donor || donor == this) return {{}, status_t::config_mismatch_k};
        refine_plan_t plan {shards_.allocator()};
        if (failed(plan.donor_holds_.reserve_exactly(donors.size()))) return {{}, status_t::bad_alloc_k};
        plan.donors_ = donors;
        std::size_t donor_shards_count = 0;
        for (ram_indexes* donor : donors) {
            unique_lock_t hold {donor->gate_, std::try_to_lock};
            if (!hold) return {{}, status_t::contended_k};
            if (status_t const closed = donor->shards_.close(); failed(closed)) return {{}, closed};
            plan.donor_holds_.push_back_reserved(std::move(hold));
            donor_shards_count += donor->shards_.size() - donor->shards_.retired_shards();
        }

        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return {{}, status_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_ = this;

        growing_buffer<shard_t const*, forkunion_allocator_t> donor_shards {shards_.allocator()};
        growing_buffer<shard_move_t, forkunion_allocator_t> moves {shards_.allocator()};
        if (failed(donor_shards.reserve_exactly(donor_shards_count)) || failed(moves.resize(donor_shards_count)) ||
            failed(plan.donor_slots_.reserve_exactly(donor_shards_count)))
            return {{}, status_t::bad_alloc_k};
        for (ram_indexes const* donor : donors)
            for (shard_t const& one : donor->shards_) donor_shards.push_back_reserved(&one);
        auto const donor_shard = [&](std::size_t from) noexcept -> shard_t const& { return *donor_shards[from]; };
        auto const placement_of = [](shard_t const& one) noexcept { return one.index->get_allocator(); };
        auto const open = [this]() noexcept {
            return shards_.open_shard_generation_past_seal_(
                shards_.open_shard_generation(), [this](shard_t const& like) noexcept { return build_shard_(like); });
        };
        if (status_t const packed = shards_.pack(to_writable_span(moves), 0, shards_.shard_capacity(), donor_shard,
                                                 placement_of, rank_fewest_entries, open);
            failed(packed))
            return {{}, packed};

        std::size_t const shards = shards_.size();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t const first = plan.donor_slots_.size();
            for (shard_move_t const& move : moves)
                if (move.into == into) plan.donor_slots_.push_back_reserved(donor_shards[move.from]->index.get());
            index_t& one = *shards_[into].index;
            if (plan.donor_slots_.size() == first) {
                plan.shard_plans_.push_back_reserved(shard_refine_plan_t {one.get_allocator()});
                continue;
            }
            // The set closed every shard, and the shard's own plan closes it again for its life.
            one.reopen_();
            auto [shard_plan, status] = one.plan_absorb(
                std::span<index_t*>(plan.donor_slots_.data() + first, plan.donor_slots_.size() - first), config,
                selector, context);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            // Dropped, the shard plans move nothing, so a refused plan closes no window it opened.
            if (failed(status)) return plan.shard_plans_.clear(), expected<refine_plan_t> {{}, status};
        }
        if (plan.moving_()) plan.begin_windows_();
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Opens a refinement of every live shard in place, each through its own plan over all
     *      its nodes, which re-selects, sweeps and heals as one pass.
     *
     *  Closes this index under its unique side and keeps it sealed for the plan's life, as an
     *  absorb does, while searches run throughout.
     *
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     *  @param[in] selector Carries the budget every neighbor list is reserved for.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the shard
     *      plans.
     *
     *  @par Statuses:
     *  - @c contended_k A call or batch in flight on this index.
     */
    template <typename selector_type_>
    expected<refine_plan_t> plan_refine(refine_config_t config, selector_type_ const& selector,
                                        traversal_context_t& context) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        refine_plan_t plan {shards_.allocator()};
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return {{}, status_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_ = this;
        std::size_t const shards = shards_.size(), retired = shards_.retired_shards();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (shard < retired) {
                plan.shard_plans_.push_back_reserved(shard_refine_plan_t {one.get_allocator()});
                continue;
            }
            // The set closed every shard, and the shard's own plan closes it again for its life.
            one.reopen_();
            auto [shard_plan, status] = one.plan_refine(iota_view<node_id_t> {0, static_cast<node_id_t>(one.size())},
                                                        config, selector, context);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            if (failed(status)) return {{}, status};
        }
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @ref plan_absorb.
     *
     *  Live shard @c s splits into live shard @c s of every receiver through its own
     *  @c index_t::plan_split, one graph-local part per receiver with its keys beside its nodes,
     *  and this index empties.
     *
     *  Every fallible step happens here: this index's unique side taken and its shards closed,
     *  then every receiver's, all held until the plan dies, and every shard's split planned. The
     *  copy phases open this index's window, so its readers wait them out, and the barrier
     *  publishing the parts shows a search of each receiver every shard's nodes or none. Every
     *  receiver shard then re-selects its picked nodes and heals; a stop pauses only that walk.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two, all distinct, each
     *      with as many live shards as this index; the slots must outlive the plan.
     *  @param[in] config Optional tuning knobs, zeros meaning "derive from the selector's budget",
     *      and the seed picking every shard's part roots: equal seeds split equal graphs alike.
     *  @param[in] selector Carries the budget every neighbor list is reserved for.
     *  @param[in,out] context The planning thread's scratch; its allocator places the shard plans.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, another live shard count, or a zero budget.
     *  - @c contended_k A call or batch in flight on this index or a receiver, or another refine,
     *    absorb or split holding either.
     *  - @c overflow_k A part past a receiver shard's ceiling.
     */
    template <typename selector_type_>
    expected<refine_plan_t> plan_split(std::span<ram_indexes*> receivers, split_config_t config,
                                       selector_type_ const& selector, traversal_context_t& context) noexcept {
        std::size_t const parts = receivers.size();
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (parts < 2) return {{}, status_t::config_mismatch_k};
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            if (!receivers[receiver] || receivers[receiver] == this) return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == receivers[receiver]) return {{}, status_t::config_mismatch_k};
        }
        refine_plan_t plan {shards_.allocator()};
        if (failed(plan.receiver_holds_.reserve_exactly(parts))) return {{}, status_t::bad_alloc_k};
        unique_lock_t hold {gate_, std::try_to_lock};
        if (!hold) return {{}, status_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_hold_ = std::move(hold);
        plan.owner_ = this;
        plan.receivers_ = receivers;
        std::size_t const shards = live_shards_count(), retired = shards_.retired_shards();
        // A closed receiver takes no entry, so its emptiness and shards hold for the plan's life.
        for (ram_indexes* into : receivers) {
            unique_lock_t receiver_hold {into->gate_, std::try_to_lock};
            if (!receiver_hold) return {{}, status_t::contended_k};
            if (status_t const closed = into->shards_.close(); failed(closed)) return {{}, closed};
            plan.receiver_holds_.push_back_reserved(std::move(receiver_hold));
            if (into->live_shards_count() != shards || into->size()) return {{}, status_t::config_mismatch_k};
        }
        if (failed(plan.shard_plans_.reserve_exactly(shards)) ||
            failed(plan.receiver_shards_.reserve_exactly(shards * parts)))
            return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard)
            for (ram_indexes* into : receivers)
                plan.receiver_shards_.push_back_reserved(
                    into->shards_[into->shards_.retired_shards() + shard].index.get());
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::span<index_t*> const into {plan.receiver_shards_.data() + shard * parts, parts};
            // The sets closed every receiver shard; its own plan now closes it again for its life.
            for (index_t* receiver_shard : into) receiver_shard->reopen_();
            auto [shard_plan, status] = shards_[retired + shard].index->plan_split(
                into, {config.refine, config.seed + shard}, selector, context);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            if (failed(status)) return plan.shard_plans_.clear(), expected<refine_plan_t> {{}, status};
        }
        plan.begin_windows_();
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Moves every entry of this index into one graph-local part per receiver and links
     *      each in: @ref plan_split, every phase drained on the calling thread with @p metric.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, another live shard count, or a zero budget.
     *  - @c contended_k A call or plan holds this index or a receiver.
     *  - @c overflow_k A part past a receiver shard's ceiling.
     */
    template <typename metric_type_, typename selector_type_>
    expected<refine_report_t> split(std::span<ram_indexes*> receivers, split_config_t config,
                                    selector_type_ const& selector, metric_type_& metric,
                                    traversal_context_t& context) noexcept {
        auto [plan, status] = plan_split(receivers, config, selector, context);
        if (failed(status)) return {{}, status};
        do
            for (typename refine_plan_t::step_t const step : plan.steps()) [[maybe_unused]]
                status_t const ran = plan.run(step, metric, context, selector);
        while (plan.advance());
        return plan.finish();
    }

    /**
     *  @brief Reachability of every live shard, each audited by its own @c plan_health from node
     *      zero both ways on the calling thread, summed as @c reachability_t folds.
     *
     *  Read-only, so concurrent insertions leave fresh nodes uncounted; the audit runs again once
     *  if a destructive window overlapped it, and each shard's bitsets come from the allocator
     *  @p context carries.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c bad_alloc_k A shard's bitsets could not be placed.
     *  - @c contended_k Two destructive windows overlapped the audit.
     */
    template <typename context_allocator_type_>
    expected<reachability_t> health(traversal_context_over<context_allocator_type_>& context) noexcept {
        using audit_t = typename index_t::template health_plan<context_allocator_type_>;
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        return shards_.read_retrying_once([&]() noexcept -> expected<reachability_t> {
            reachability_t total;
            for (index_t& one : shards_.live_indexes()) {
                auto [audit, audit_status] = one.plan_health(context);
                if (failed(audit_status)) return {total, audit_status};
                do
                    for (typename audit_t::step_t const step : audit.steps()) [[maybe_unused]]
                        std::size_t const expanded = audit.sweep(step);
                while (audit.advance_level());
                auto const [shard_health, finish_status] = audit.finish();
                if (failed(finish_status)) return {total, finish_status};
                total += shard_health;
            }
            return {total, status_t::success_k};
        });
    }

#pragma endregion Refinement

  private:

    /** The ascending matches a walk or a scan left in @p ranking. */
    template <typename ranking_type_>
    static std::span<scored_node_t const> ranking_of_(ranking_type_ const& ranking) noexcept {
        return {ranking.begin(), ranking.size()};
    }

    /**
     *  @brief Walks every shard @p watermarks counts through @p walk_shard, each below its own
     *      watermark, and merges each ranking into @p keys and @p distances through its records.
     *
     *  The whole merge runs again once if a destructive window overlapped it, then answers
     *  @c contended_k; watermarks a window outran since their snapshot answer @c contended_k, so
     *  the caller snapshots again. @p walk_shard is called as <tt>walk_shard(shard, watermark,
     *  merged) → expected<std::span<scored_node_t const>></tt>.
     */
    template <typename walk_shard_type_>
    expected<std::size_t> merge_shards_(std::span<node_id_t const> watermarks, std::span<key_t> keys,
                                        std::span<distance_t> distances, walk_shard_type_&& walk_shard) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<std::size_t> {
            if (watermarks.size() > shards_.size()) return {0, status_t::contended_k};
            std::size_t merged = 0;
            status_t status = status_t::success_k;
            // Retired shards and shards published empty hold a zero watermark, and walk nothing.
            for (std::size_t shard = 0; shard != watermarks.size() && succeeded(status); ++shard) {
                if (!watermarks[shard]) continue;
                index_t& one = *shards_[shard].index;
                if (watermarks[shard] > one.size()) return {0, status_t::contended_k};
                auto const [ranking, walk_status] = walk_shard(one, watermarks[shard], merged);
                status = walk_status;
                merged = merge_search_results(ranking, key_of_node_t {&one}, keys, distances, merged);
            }
            return {merged, status};
        });
    }

    /** One empty shard on @p allocator; null when the allocation fails. */
    std::unique_ptr<index_t> build_(allocator_t const& allocator) const noexcept {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(allocator, line_bytes_));
    }

    /** One shard of a new shard generation on the placement @p like sits on, reserved under the
     *  bound; a null index when the allocation fails. */
    shard_t build_shard_(shard_t const& like) const noexcept {
        std::unique_ptr<index_t> index = build_(like.index->get_allocator());
        if (index && failed(index->reserve({0, shards_.shard_capacity()}).status)) index.reset();
        return shard_t {std::move(index)};
    }

    /** The shards and their routing, the commit word every snapshot reads by and the epoch every
     *  reader brackets itself with. */
    shards_t shards_;

    /** Shared by every mutator for one call, alone by every destructive window; never swapped. */
    alignas(default_alignment_k) shared_mutex_t gate_;

    /** The line every shard this index builds prefetches by. */
    std::size_t line_bytes_;
};

/**
 *  @brief A reader's walk over the entries of one @c ram_indexes as its live shards stood at
 *      open: every entry as stored, shard after shard, or breadth-first from the start entries,
 *      each entry once, each with the keys it links to.
 *
 *  Every call names the index it walks, which must be the one the cursor opened on. One shard
 *  cursor per live shard stages each entry's neighbors under that node's lock, so the walk never
 *  blocks a writer for longer than one record, and links stay inside a shard, so each resolves to
 *  its key through the shard it leaves. A breadth-first walk reaches the nodes below each shard
 *  cursor's snapshot at open; a stored walk refreshes each shard's watermark as it enters that
 *  shard. The walk holds only under the epoch the cursor opened in: once a destructive window moved
 *  it, every pull answers @c contended_k.
 */
template <typename indexes_type_>
struct ram_indexes_cursor {
    using indexes_t = indexes_type_;
    using index_t = typename indexes_t::index_t;
    using key_t = typename indexes_t::key_t;
    using node_id_t = typename indexes_t::node_id_t;
    using walk_t = cursor_walk<node_id_t>;
    using shard_node_t = typename walk_t::shard_node_t;
    using shard_cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;

    /** Where a pull gathers its entries' links, in the caller's arena: the linked keys end to end,
     *  and the offset where each entry's run starts. */
    using links_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using links_keys_t = growing_buffer<key_t, links_allocator_t>;
    using links_offsets_t = growing_buffer<std::size_t, links_allocator_t>;

    explicit ram_indexes_cursor(forkunion_allocator_t allocator) noexcept
        : shard_cursors_(allocator), walk_(allocator), allocator_(allocator) {}

    /**
     *  @brief Opens the walk over every live shard of @p indexes under the epoch it reads now; any
     *      @p starts make it breadth-first, each key starting from its lowest node in the lowest
     *      shard holding it, and a key held nowhere starting nothing.
     *
     *  The shards are taken mutable, as a shard cursor locks each record it stages.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The shard cursors, the visited bits or the frontier found no memory.
     */
    status_t open(indexes_t& indexes, std::span<key_t const> starts) noexcept {
        epoch_ = indexes.read_epoch();
        first_shard_ = indexes.shards_.retired_shards();
        std::size_t const count = indexes.shards_count() - first_shard_;
        if (failed(shard_cursors_.reserve_exactly(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard)
            shard_cursors_.push_back_reserved(
                shard_cursor_t {*indexes.shards_[first_shard_ + shard].index, allocator_});
        if (starts.empty()) return status_t::success_k;
        if (status_t const opened = walk_.open_breadth_first(
                count, [&](std::size_t shard) noexcept { return shard_cursors_[shard].visible_count(); });
            failed(opened))
            return opened;
        for (key_t const& key : starts)
            for (std::size_t shard = 0; shard != count; ++shard) {
                node_id_t const visible_count = shard_cursors_[shard].visible_count();
                node_id_t lowest = visible_count;
                indexes.shards_[first_shard_ + shard].index->find_each(
                    key, [&](node_id_t node) noexcept { lowest = min_of(lowest, node); });
                if (lowest == visible_count) continue;
                if (status_t const visited = walk_.visit(static_cast<u32_t>(shard), lowest); failed(visited))
                    return visited;
                break;
            }
        return status_t::success_k;
    }

    /**
     *  @brief Writes up to @p keys.size() next entries' keys into @p keys, gathering each one's
     *      links into @p links_keys and @p links_offsets when both are named, and answers the
     *      number of keys written.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k A link or a newly visited entry found no memory; the entries written stand.
     *  - @c contended_k A destructive window of @p indexes opened since the walk did; none stand.
     */
    expected<std::size_t> pull(indexes_t const& indexes, std::span<key_t> keys, links_keys_t* links_keys,
                               links_offsets_t* links_offsets) noexcept {
        auto enter = [&](u32_t shard) noexcept {
            shard_cursors_[shard].refresh();
            return shard_cursors_[shard].visible_count();
        };
        std::size_t written = 0;
        shard_node_t entry;
        while (written != keys.size() && walk_.next(entry, shard_cursors_.size(), enter)) {
            shard_cursor_t& cursor = shard_cursors_[entry.shard];
            index_t const& shard = *indexes.shards_[first_shard_ + entry.shard].index;
            if (status_t const sought = cursor.seek(entry.node); failed(sought)) return {written, sought};
            std::span<node_id_t const> const neighbors = cursor.neighbors();
            if (links_keys && (failed(links_keys->reserve(links_keys->size() + neighbors.size())) ||
                               failed(links_offsets->reserve(links_offsets->size() + 1))))
                return {written, status_t::bad_alloc_k};
            for (node_id_t const neighbor : neighbors) {
                // Breadth-first never refreshes, so the bits span the shard cursors' snapshots.
                if (walk_.breadth_first() && neighbor < cursor.visible_count())
                    if (status_t const visited = walk_.visit(entry.shard, neighbor); failed(visited))
                        return {written, visited};
                if (!links_keys) continue;
                expected<key_t> const linked = shard.key_of(neighbor);
                if (succeeded(linked.status)) links_keys->push_back_reserved(linked.value);
            }
            if (links_keys) links_offsets->push_back_reserved(links_keys->size());
            keys[written++] = shard.key_of(entry.node).value;
        }
        // Retired shards keep their records, so a pull that ran past a window read nothing freed.
        if (!indexes.still_valid(epoch_)) return {0, status_t::contended_k};
        return {written, status_t::success_k};
    }

  private:
    /** One per live shard at open, in placement order, each over the snapshot it took then. */
    growing_buffer<shard_cursor_t, forkunion_allocator_t> shard_cursors_;

    /** The order of the walk and where it stands. */
    walk_t walk_;

    forkunion_allocator_t allocator_;

    /** The position of the first shard walked: the first live one at open, which a clear since
     *  retires in place. */
    std::size_t first_shard_ = 0;

    /** The epoch the walk opened under; a pull after a window since answers contended. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::basic
