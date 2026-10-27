/**
 *  @file include/usearch/basic/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Many basic RAM shards viewed as one, so a basic index grows past one shard's ID space.
 *
 *  The plural over @c basic::ram_index, shaped after it: contexts are the caller's, one per thread,
 *  metrics come with every call, and the caller drives its own threads. Shards share one bound, and
 *  an insertion crossing it opens another shard generation on the same placements. Keys stay in the
 *  shards' records, so lookups fan out over every shard, and a search walks each shard below one
 *  snapshot of the watermarks and merges their rankings into the caller's keys.
 *
 *  Mutators hold one gate's shared side for each call, and the destructive windows - a clear, and
 *  an absorb emptying its donors - take its unique side or answer @c contended_k. Readers take no
 *  lock: they bracket each call with the shard set's epoch, retry once when a window overlapped
 *  them, and answer @c contended_k when a second one did. A record's value owns heap memory a
 *  reader may still score, so a clear frees nothing: it retires the shards whole and fresh ones
 *  take the next rows, and retired shards keep their memory until the index dies.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <functional>   // `std::hash` - the shard set's key hash
#include <memory>       // `std::unique_ptr` - an index holds atomics and never moves
#include <mutex>        // `std::unique_lock` - the windows' hold on the gate
#include <new>          // `std::nothrow` - a shard built on a memory domain
#include <shared_mutex> // `std::shared_lock` - the mutators' hold on the gate
#include <span>         // `std::span` - the caller's watermarks, keys & distances
#include <string_view>  // `std::string_view` - the checkpoint paths
#include <type_traits>  // `std::is_same_v` - the unfiltered walk
#include <utility>      // `std::move`, `std::exchange`

#include <forkunion.h> // `fu_memory_domain_id_t`

#include "usearch/utilities/types.hpp"       // `shared_mutex` - the gate
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the plans' shard plans
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator_t` - the shard table's placement
#include "usearch/utilities/ram_shards.hpp"  // `shard_set` - the shard table
#include "usearch/basic/ram_index.hpp"
#include "usearch/basic/shard_checkpoints.hpp" // `save_shards`, `load_shards` - the checkpoints of every live shard

namespace unum::usearch::basic {

template <typename indexes_type_>
struct ram_indexes_cursor;

/** Keys live in the basic records, so the plural stores none and resolves them per shard. */
struct record_keys_t {
    status_t cover(std::size_t) const noexcept { return status_t::success_k; }
    std::size_t memory_usage() const noexcept { return 0; }
    friend record_keys_t& host_keys(record_keys_t& keys) noexcept { return keys; }
};

/**
 *  @brief Basic RAM shards viewed as one: insertions routed over the shard generations, searches
 *      merged across them, lookups fanned out.
 *
 *  @tparam index_type_ The shard, a @c basic::ram_index whose allocator a memory domain builds. Its
 *      waiter and atomic reference serve the shard table too, so the defaults hold: the standard
 *      reference posts what ForkUnion's spell natively through @c atomic_add. Its node IDs are 32
 *      bits wide, as the shard table locates every row in 8 bytes; a lone @c basic::ram_index takes
 *      either width.
 */
template <typename index_type_>
    requires std::same_as<typename index_type_::node_id_t, utilities::node_id_t>
struct ram_indexes {
    friend struct ram_indexes_cursor<ram_indexes>;

    /** A reader's walk over the keys of every shard's published rows in placement order. */
    using cursor_t = ram_indexes_cursor<ram_indexes>;

    using index_t = index_type_;
    using key_t = typename index_t::key_t;
    using value_t = typename index_t::value_t;
    using node_id_t = typename index_t::node_id_t;
    using distance_t = typename index_t::distance_t;
    using allocator_t = typename index_t::allocator_t;
    using wait_t = typename index_t::wait_t;
    using scored_node_t = typename index_t::scored_node_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using insert_config_t = typename index_t::insert_config_t;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;

    template <typename context_allocator_type_>
    using traversal_context_over = typename index_t::template traversal_context_over<context_allocator_type_>;

    template <typename word_type_>
    using atomic_ref = typename index_t::template atomic_ref<word_type_>;

    /** The shards and the rotation that routes new rows; the keys stay in their records. */
    using shards_t = shard_set<index_t, record_keys_t, key_t, std::hash<key_t>, index_t::template atomic_ref, wait_t>;
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

    /** The engine filter of a walk that brings none, for every shard alike. */
    struct unrelated_t {
        dummy_predicate_t operator()(index_t const&) const noexcept { return {}; }
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

    /** Bounds every shard at @p rows, clamped to the ID space; zero means the ID space. Past it, an
     *  insertion opens another shard generation on the same placements. Ignored once storage
     *  exists. */
    void bound_shards(std::size_t rows) noexcept { shards_.bound_shards(rows); }

    /** Appends one empty shard placed on @p memory_domain; refused once storage exists. */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(allocator_t {memory_domain});
        if (!shard) return status_t::bad_alloc_k;
        return shards_.emplace(std::move(shard), record_keys_t {});
    }

    /** Covers @p capacity rows split evenly over the shards, each under the bound - a hint, since
     *  the shards grow on demand; callable alongside every other verb. */
    expected<std::size_t> reserve(std::size_t capacity) noexcept { return shards_.reserve({capacity}); }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.entries(); }

    /** Shards in the index, every shard generation's and the retired ones, in placement order: the
     *  positions a snapshot fills. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** Shards past the retired ones, which every walk and checkpoint spans. */
    std::size_t live_shards_count() const noexcept { return shards_.size() - shards_.retired_shards(); }

    /** Rows the live shards host without growing again; they grow past it on demand. */
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
     *  shard generation opened answers @c shape_mismatch_k, to be sized again. */
    expected<std::size_t> snapshot(std::span<node_id_t> watermarks) const noexcept {
        return shards_.snapshot(watermarks);
    }

    /**
     *  @brief Empties the index without freeing anything a reader may hold: a fresh shard
     *      generation on the same placements takes the next rows, and every shard before it retires
     *      whole.
     *
     *  Takes the gate alone and closes every shard first, so a call in flight, a batch holding
     *  claims or an absorb holding the index refuses it with @c contended_k, and an index holding
     *  no rows stays as it is. The fresh shards open before any retires, so a refused build answers
     *  @c bad_alloc_k with every row in place. A search it lands under reads values nothing frees,
     *  and retries.
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
     *  @ref load_shards: every shard's own plans on the calling thread, its keys in its records,
     *  which those plans write and read, so no key column or locator lands beside them, and a
     *  load's parts past the placed shards on shard generations built like them. */
    struct checkpoint_residency_t {
        using seats_t = typename shards_t::locators_map_t::seats_t;

        ram_indexes* owner = nullptr;

        template <typename codec_type_>
        static expected<columns_extents_t> columns_extents(shards_t& shards, std::size_t shard,
                                                           codec_type_ const& codec) noexcept {
            return shards[shard].index->columns_extents(codec);
        }
        template <typename codec_type_>
        static expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                       typename index_t::columns_t columns,
                                                                       codec_type_ const& codec) noexcept {
            return shards[shard].index->plan_save(columns, codec);
        }
        static expected<typename index_t::load_plan_t> open_shard_load(
            shards_t& shards, std::size_t shard, typename index_t::columns_view_t columns) noexcept {
            return shards[shard].index->plan_load(columns);
        }
        static expected<std::size_t> reserve(shards_t& shards, std::size_t shard, std::size_t rows) noexcept {
            return shards[shard].index->reserve({rows, shards.shard_capacity()});
        }

        /** The shard's own save plan writes its records' keys, so nothing lands here. */
        static void save_keys(shards_t&, std::size_t, std::span<key_t>) noexcept {}

        /** The shard's own load plan lands the keys in its records, so no seat is claimed. */
        static expected<seats_t> claim_seats(shards_t&, std::size_t) noexcept { return {{}, status_t::success_k}; }
        static status_t land_keys(shards_t&, std::size_t, std::size_t, std::span<key_t const>, seats_t) noexcept {
            return status_t::success_k;
        }

        /** Leaves part @p part the live shard it names, a placed one, or opens shard generations on
         *  the placed shards' memory domains until it exists. */
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

    /** Writes the live shards under @p path through @ref save_shards: a lone live shard's monolith,
     *  or one part per live shard beside the manifest @p path names. Takes the gate alone, so a
     *  call in flight refuses it with @c contended_k. */
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
     *  alone, as @ref save does. */
    template <typename codec_type_, typename executor_type_, typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, codec_type_ const& codec, executor_type_& executor,
                  load_config_t const& config, progress_type_&& progress = {}) noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        return load_shards(shards_, checkpoint_residency_t {this}, path, codec, executor, config,
                           std::forward<progress_type_>(progress));
    }

#pragma endregion Checkpoints

#pragma region Insertion

    /**
     *  @brief One batch in flight across the shards: one shard plan per shard of the open shard
     *      generation and of every shard generation the batch opened, so its rows can only land,
     *      and @ref finish publishes them all at once.
     *
     *  Rows are split across the open shard generation's shards round-robin from the starting shard
     *  on, and rows past what fits under the bound fill fresh shard generations in turn, split the
     *  same way. Every row must land before @ref finish, since each shard plan holds its shard's
     *  later turns until then. Each step and @ref finish hold the gate's shared side for their own
     *  call.
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
         *  @brief Lands rows @p first onward, one per value of @p values under its key in @p keys,
         *      each in the shard its position falls to, as the shard plan's own step does.
         *
         *  Any number of threads step one plan at once over disjoint rows, each with its own
         *  @p metric and a @p context that @c index_t::reserve_traversal sized. Rows outside the
         *  plan or keys of another length land nothing, and @ref finish then refuses.
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
                entry_in_shard_t const landing = batch_split_.shard_and_offset_of_entry(first + position);
                shard_plans_[landing.shard].template insert<synchronization_>(
                    landing.offset, values.subspan(position, 1), keys.subspan(position, 1), metric, context, selector,
                    config);
            }
        }

        /** Publishes every row at once: @c config_mismatch_k while any has yet to land, leaving
         *  the plan whole; then every shard's turn inside one commit, so a snapshot sees the batch
         *  whole or not at all. The plan is spent either way, a shard refusing its turn answering
         *  the first refusal after every other shard published, so a second call answers
         *  @c uninitialized_k. */
        status_t finish() noexcept {
            if (!owner_) return status_t::uninitialized_k;
            std::size_t touched = 0;
            for (shard_insert_plan_t& shard_plan : shard_plans_) {
                if (atomic_ref<std::size_t>(shard_plan.landed).load(std::memory_order_acquire) !=
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

        /** Invokes @p callback as `callback(shard, range)` for every shard the plan lands rows in,
         *  with the IDs dispensed there, in placement order; before @ref finish, so a caller can
         *  keep what it follows the published rows up with, as long links. */
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
         *  one where no row lands. */
        growing_buffer<shard_insert_plan_t, forkunion_allocator_t> shard_plans_;

        /** The batch split across the shard generation open at planning, which the shard plans
         *  start at, and the fresh ones it opened. */
        batch_split_t batch_split_ {};

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once @c finish published. */
        ram_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens a batch of @p count rows: routes them over the open shard generation and as
     *      many fresh ones as the bound makes them fill, then opens one shard plan per shard taking
     *      rows, in placement order.
     *
     *  The rotation comes from the rows routed so far, which keeps a stream of small batches from
     *  drifting onto shard zero. Every shard claims its rows before any dispenses, so a refusal
     *  gives back every claim and strands no ID. Everything fallible past the shard plans belongs
     *  before this call, as for a shard's own plan: building the values and sizing each stepping
     *  thread's context.
     *
     *  @retval uninitialized_k No shard placed yet.
     *  @retval contended_k An absorb holds the index closed.
     *  @retval overflow_k A racing batch took the room this one was routed into.
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
     *  @param[inout] metric This thread's scoring instance.
     *  @param[in] relate Builds the engine's own filter over one shard's node IDs, called as
     *      `relate(shard) → relation`, answering @c dummy_predicate_t where it filters nothing.
     *  @param[inout] predicate The caller's filter over keys, asked about each shard's walked nodes
     *      through their records, or @c dummy_predicate_t for an unfiltered walk.
     *  @param[inout] context The calling thread's scratch & telemetry over any allocator.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from each shard's connectivity".
     *  @param[in] watermarks One per shard from @ref snapshot.
     *  @param[out] keys The merged ranking's keys, as many as it holds.
     *  @param[out] distances The same ranking's distances, parallel to @p keys.
     *  @return The matches merged, or @c contended_k when two windows overlapped the call or one
     *      outran the snapshot.
     */
    template <typename query_type_, typename metric_type_, typename relate_type_, typename predicate_type_,
              typename context_allocator_type_>
    expected<std::size_t> search_if(query_type_ const& query, metric_type_& metric, relate_type_ const& relate,
                                    predicate_type_& predicate,
                                    traversal_context_over<context_allocator_type_>& context, search_config_t config,
                                    std::span<node_id_t const> watermarks, std::span<key_t> keys,
                                    std::span<distance_t> distances) noexcept {
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

    /** @ref search_if with no engine filter of its own. */
    template <typename query_type_, typename metric_type_, typename predicate_type_, typename context_allocator_type_>
    expected<std::size_t> search_if(query_type_ const& query, metric_type_& metric, predicate_type_& predicate,
                                    traversal_context_over<context_allocator_type_>& context, search_config_t config,
                                    std::span<node_id_t const> watermarks, std::span<key_t> keys,
                                    std::span<distance_t> distances) noexcept {
        return search_if(query, metric, unrelated_t {}, predicate, context, config, watermarks, keys, distances);
    }

    /** The exact sibling of @ref search_if: every row below each shard's watermark scored as
     *  @c index_t::scan scores, the nearest @p keys can hold kept, as @p relate and @p predicate
     *  admit them. Each shard's rounds drain within the merged ranking's worst match once it is
     *  full, so later shards score fewer rows. */
    template <typename query_type_, typename metric_type_, typename relate_type_, typename predicate_type_,
              typename context_allocator_type_>
    expected<std::size_t> scan(query_type_ const& query, metric_type_& metric, relate_type_ const& relate,
                               predicate_type_& predicate, traversal_context_over<context_allocator_type_>& context,
                               std::span<node_id_t const> watermarks, std::span<key_t> keys,
                               std::span<distance_t> distances) noexcept {
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

    /** @ref scan with no engine filter of its own. */
    template <typename query_type_, typename metric_type_, typename predicate_type_, typename context_allocator_type_>
    expected<std::size_t> scan(query_type_ const& query, metric_type_& metric, predicate_type_& predicate,
                               traversal_context_over<context_allocator_type_>& context,
                               std::span<node_id_t const> watermarks, std::span<key_t> keys,
                               std::span<distance_t> distances) noexcept {
        return scan(query, metric, unrelated_t {}, predicate, context, watermarks, keys, distances);
    }

    /** Entries under @p key across every live shard, read between destructive windows. */
    std::size_t count(key_t const& key) const noexcept {
        return shards_.read_until_settled([&]() noexcept {
            std::size_t total = 0;
            for (shard_t const& one : shards_) total += one.index->count(key);
            return total;
        });
    }

    /** Invokes @p callback as `callback(shard, node_id)` for every published entry under @p key in
     *  every live shard; an ID a clear retired meanwhile is refused by every accessor of its
     *  shard. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        for (shard_t const& one : shards_) {
            index_t const& shard = *one.index;
            shard.find_each(key, [&](node_id_t node_id) noexcept { callback(shard, node_id); });
        }
    }

#pragma endregion Search and Lookups

#pragma region Refinement

    /**
     *  @brief One absorb in flight: every donor shard lands in a shard of this index through that
     *      shard's own plan, which then re-selects across what landed and heals.
     *
     *  Steps of a phase are (shard, position) pairs over the shard plans' own steps, so any number
     *  of threads may @ref run them at once; @ref advance is the single-caller barrier, advancing
     *  every shard plan together. While any shard plan lands, its rows move out of the donors, so
     *  the donors' readers wait out a window spanning the landing phases. The plan holds every
     *  donor's unique side and this index sealed for its whole life, and its death reopens this
     *  index and every donor not consumed: one whose every shard landed stays empty, closed and
     *  sealed.
     */
    struct [[nodiscard]] refine_plan_t {

        /** One step: the shard whose plan runs it, and its position among that plan's steps.
         *  Two words. */
        struct step_t {
            std::size_t shard = 0;
            std::size_t position = 0;
        };
        static_assert(sizeof(step_t) == 2 * sizeof(std::size_t));

        /** What the shard plans' audits judge, summed: the published rows a driver's progress
         *  counts against. */
        struct audit_t {
            std::size_t visible = 0;
        };

        /** Steps of the current phase a driver already ran, so a cancelled call resumes at the next
         *  one. */
        std::size_t taken = 0;

        /** Nodes the planned phases re-selected so far, which a driver's progress reports. */
        std::size_t ticked = 0;

        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An unbound plan over @p allocator; only @ref plan_absorb ever fills it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : shard_plans_(allocator), donor_slots_(allocator), donor_holds_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : taken(other.taken), ticked(other.ticked), shard_plans_(std::move(other.shard_plans_)),
              donor_slots_(std::move(other.donor_slots_)), donor_holds_(std::move(other.donor_holds_)),
              donors_(other.donors_), owner_(std::exchange(other.owner_, nullptr)) {}
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

        /** Runs @p step on the calling thread through its shard's plan, as that plan's own
         *  @c run does; a step past the end runs nothing. */
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
         *  every shard plan, publishing what landed inside one commit, and answers whether any has
         *  another phase. */
        bool advance() noexcept {
            if (!owner_) return false;
            std::size_t const landing_before = landing_();
            if (landing_before) owner_->shards_.begin_commit();
            std::size_t follows = 0;
            for (shard_refine_plan_t& shard_plan : shard_plans_) follows += shard_plan.advance();
            if (landing_before) owner_->shards_.end_commit();
            if (landing_before && !landing_()) {
                end_donor_windows_();
                release_consumed_donors_();
            }
            return follows != 0;
        }

        /** Stops every shard plan's re-selection where its steps ran, so the next @ref advance
         *  turns to the heal; a no-op for a plan still landing. */
        void stop() noexcept {
            for (shard_refine_plan_t& shard_plan : shard_plans_) shard_plan.stop();
            taken = steps().size();
        }

        /** The published rows every shard plan's audit judges. */
        audit_t audit() const noexcept {
            audit_t total;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) total.visible += shard_plan.audit.visible;
            return total;
        }

        /** The budget planning reserved every row for, which every shard shares. */
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

        /** Whether any shard plan still lands a donor, which no stop skips. */
        bool landing() const noexcept { return landing_() != 0; }

        /** Whether every live shard of the donor at @p donor landed, consuming it: once the landing
         *  phases end, the plan drops its hold and nulls its slot, so its owner may free it at
         *  once; it stays closed and sealed for good. */
        bool consumed(std::size_t donor) const noexcept { return donor < donors_.size() && !donors_[donor]; }

        /**
         *  @brief Invokes @p visit as `visit(shard, rows)` for every row range the plan rewrote:
         *      every row of every shard a shard plan refined.
         *
         *  Called once the plan finished and before it dies, while the index stays sealed, so a
         *  rule the re-selection skips, like a modality's long links, is drawn again over them.
         */
        template <typename visit_type_>
        void for_each_shard_range(visit_type_&& visit) const noexcept {
            if (!owner_) return;
            for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                // A shard no donor landed in, and a retired one, has no plan of its own.
                if (shard_plans_[shard].finish().status == status_t::uninitialized_k) continue;
                index_t& one = *owner_->shards_[shard].index;
                visit(one, iota_view<node_id_t> {0, static_cast<node_id_t>(one.size())});
            }
        }

        /** The pass so far, summed over the shard plans: complete once @ref advance answered false,
         *  otherwise @c cancelled_k. */
        expected<refine_result_t> finish() const noexcept {
            refine_result_t total;
            status_t status = owner_ ? status_t::success_k : status_t::uninitialized_k;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) {
                auto const [tally, tally_status] = shard_plan.finish();
                // A shard no donor landed in has no plan of its own.
                if (tally_status == status_t::uninitialized_k) continue;
                if (failed(tally_status) && succeeded(status)) status = tally_status;
                total += tally;
            }
            return {total, status};
        }

      private:
        friend ram_indexes;

        /** Shard plans in their landing phases; the donors' windows stay open while any is. */
        std::size_t landing_() const noexcept {
            std::size_t landing = 0;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) landing += shard_plan.landing();
            return landing;
        }

        void begin_donor_windows_() noexcept {
            for (std::size_t donor = 0; donor != donor_holds_.size(); ++donor) donors_[donor]->shards_.begin_window();
        }

        void end_donor_windows_() noexcept {
            for (std::size_t donor = 0; donor != donor_holds_.size(); ++donor) donors_[donor]->shards_.end_window();
        }

        /** Closes any landing window, drops the shard plans, which reopen their shards and release
         *  the donor shards, reopens every donor not consumed, then this index, and drops every
         *  hold; a consumed donor stays closed and sealed. */
        void release_() noexcept {
            if (landing_()) end_donor_windows_();
            shard_plans_.clear();
            std::size_t const sources = sources_();
            for (std::size_t held = 0; held != donor_holds_.size(); ++held)
                if (donors_[held] && !consumed_(*donors_[held], sources)) donors_[held]->shards_.reopen();
            donor_holds_.clear();
            if (owner_) std::exchange(owner_, nullptr)->shards_.reopen();
        }

        /** Once the landing phases end, drops the hold of every donor whose shards all landed and
         *  nulls its slot; consumption is judged for every donor before any slot nulls. */
        void release_consumed_donors_() noexcept {
            std::size_t const sources = sources_();
            for (std::size_t held = 0; held != donor_holds_.size(); ++held)
                if (consumed_(*donors_[held], sources)) {
                    donor_holds_[held].unlock();
                    donors_[held] = nullptr;
                }
        }

        /** The live shards of the donors the plan still holds. */
        std::size_t sources_() const noexcept {
            std::size_t sources = 0;
            for (std::size_t held = 0; held != donor_holds_.size(); ++held)
                if (donors_[held]) sources += donors_[held]->shards_.size() - donors_[held]->shards_.retired_shards();
            return sources;
        }

        /** Whether every live shard of @p donor landed: planning slotted every one of the
         *  @p sources donor shards, and each of this donor's slots nulled as its shard landed. A
         *  planning refused midway slotted only some, and landed none, and a donor never placed has
         *  nothing to land. */
        bool consumed_(ram_indexes const& donor, std::size_t sources) const noexcept {
            if (donor.shards_.empty() || donor_slots_.size() != sources) return false;
            for (shard_t const& one : donor.shards_)
                for (index_t const* slot : donor_slots_)
                    if (slot == one.index.get()) return false;
            return true;
        }

        /** One plan per shard of this index, unbound where no donor shard lands. */
        growing_buffer<shard_refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** Every donor shard grouped by the shard it lands in, each group a shard plan's donors,
         *  nulled as they land; sized once, so the groups never move. */
        growing_buffer<index_t*, forkunion_allocator_t> donor_slots_;

        /** Every donor's unique side, held and closed in donor order. */
        growing_buffer<unique_lock_t, forkunion_allocator_t> donor_holds_;

        /** The caller's donors, which must outlive the plan. */
        std::span<ram_indexes*> donors_ {};

        /** The index absorbing, sealed while the plan lives; null in an unbound plan. */
        ram_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens an absorb of @p donors, landing every donor shard in a shard of this index
     *      through that shard's own absorb.
     *
     *  Closes every donor under its unique side, which the plan keeps, and this index under its
     *  own, then packs every live donor shard into a live shard of this index under the bound,
     *  opening shard generations as they fill. A donor every shard of which landed is consumed: it
     *  stays closed and sealed once the plan dies, refusing insertions and clears with
     *  @c contended_k.
     *
     *  @param[inout] donors The indexes giving their rows up; the slots must outlive the plan.
     *  @param[inout] context The planning thread's scratch, whose allocator places the shard plans.
     *  @param[in] selector Carries the budget every row is reserved for.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     *  @retval config_mismatch_k A null donor, or this index itself.
     *  @retval contended_k A call or batch in flight on this index or a donor, or a repeated donor.
     *  @retval unsupported_k A donor shard past the bound, which would have to be cut first.
     */
    template <typename selector_type_ = keep_diverse_t>
    expected<refine_plan_t> plan_absorb(std::span<ram_indexes*> donors, traversal_context_t& context,
                                        selector_type_ const& selector = {}, refine_config_t config = {}) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        for (ram_indexes const* donor : donors)
            if (!donor || donor == this) return {{}, status_t::config_mismatch_k};
        refine_plan_t plan {shards_.allocator()};
        if (failed(plan.donor_holds_.reserve_exactly(donors.size()))) return {{}, status_t::bad_alloc_k};
        plan.donors_ = donors;
        std::size_t sources = 0;
        for (ram_indexes* donor : donors) {
            unique_lock_t hold {donor->gate_, std::try_to_lock};
            if (!hold) return {{}, status_t::contended_k};
            if (status_t const closed = donor->shards_.close(); failed(closed)) return {{}, closed};
            plan.donor_holds_.push_back_reserved(std::move(hold));
            sources += donor->shards_.size() - donor->shards_.retired_shards();
        }

        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return {{}, status_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_ = this;

        growing_buffer<shard_t const*, forkunion_allocator_t> source_shards {shards_.allocator()};
        growing_buffer<landing_t, forkunion_allocator_t> landings {shards_.allocator()};
        if (failed(source_shards.reserve_exactly(sources)) || failed(landings.resize(sources)) ||
            failed(plan.donor_slots_.reserve_exactly(sources)))
            return {{}, status_t::bad_alloc_k};
        for (ram_indexes const* donor : donors)
            for (shard_t const& one : donor->shards_) source_shards.push_back_reserved(&one);
        auto const source_shard = [&](std::size_t from) noexcept -> shard_t const& { return *source_shards[from]; };
        auto const placement_of = [](shard_t const& one) noexcept { return one.index->get_allocator(); };
        auto const open = [this]() noexcept {
            return shards_.open_shard_generation_past_seal_(
                shards_.open_shard_generation(), [this](shard_t const& like) noexcept { return build_shard_(like); });
        };
        if (status_t const packed = shards_.pack(to_writable_span(landings), 0, shards_.shard_capacity(), source_shard,
                                                 placement_of, rank_fewest_entries, open);
            failed(packed))
            return {{}, packed};

        std::size_t const shards = shards_.size();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t const first = plan.donor_slots_.size();
            for (landing_t const& landing : landings)
                if (landing.into == into)
                    plan.donor_slots_.push_back_reserved(source_shards[landing.from]->index.get());
            index_t& one = *shards_[into].index;
            if (plan.donor_slots_.size() == first) {
                plan.shard_plans_.push_back_reserved(shard_refine_plan_t {one.get_allocator()});
                continue;
            }
            // The set closed every shard, and the shard's own plan closes it again for its life.
            one.reopen_();
            auto [shard_plan, status] = one.plan_absorb(
                std::span<index_t*>(plan.donor_slots_.data() + first, plan.donor_slots_.size() - first), context,
                selector, config);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            if (failed(status)) return {{}, status};
        }
        if (plan.landing_()) plan.begin_donor_windows_();
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Opens a refinement of every live shard in place, each through its own plan over all
     *      its rows, which re-selects, sweeps and heals as one pass.
     *
     *  Closes this index under its unique side and keeps it sealed for the plan's life, as an
     *  absorb does, while searches run throughout.
     *
     *  @param[inout] context The planning thread's scratch, whose allocator places the shard plans.
     *  @param[in] selector Carries the budget every row is reserved for.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     *  @retval contended_k A call or batch in flight on this index.
     */
    template <typename selector_type_ = keep_diverse_t>
    expected<refine_plan_t> plan_refine(traversal_context_t& context, selector_type_ const& selector = {},
                                        refine_config_t config = {}) noexcept {
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
                                                        context, selector, config);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            if (failed(status)) return {{}, status};
        }
        return {std::move(plan), status_t::success_k};
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
     *  the caller snapshots again. @p walk_shard is called as
     *  `walk_shard(shard, watermark, merged) → expected<std::span<scored_node_t const>>`.
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
        return shard_t {std::move(index), record_keys_t {}};
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
 *  @brief A reader's walk over the keys of every shard's published rows, in placement order.
 *
 *  Each shard's watermark is snapshotted as the walk enters it, so a row once passed is never
 *  revisited, while a later shard may carry rows published after the walk began. The position
 *  holds only under the epoch the cursor opened in: once a destructive window moved it, every read
 *  answers @c contended_k.
 */
template <typename indexes_type_>
struct ram_indexes_cursor {
    using indexes_t = indexes_type_;
    using key_t = typename indexes_t::key_t;

    /** Enters the first shard of @p indexes, the one every later @ref read must name, under the
     *  epoch it reads now. */
    explicit ram_indexes_cursor(indexes_t const& indexes) noexcept : position_(indexes.shards_) {}

    /** Copies the keys of up to @p keys.size() rows of @p indexes into @p keys, advancing past them
     *  and answering the rows written. A destructive window since the cursor opened answers zero
     *  rows with @c contended_k: retired shards keep their records, so a read that ran past one
     *  touched nothing freed, and the check at its end discards it. */
    expected<std::size_t> read(indexes_t const& indexes, std::span<key_t> keys) noexcept {
        std::size_t written = 0;
        for (; written != keys.size() && !position_.exhausted(indexes.shards_); ++written) {
            keys[written] = indexes.shards_[position_.shard()].index->key_of(position_.node()).value;
            position_.advance(indexes.shards_);
        }
        if (!position_.still_valid(indexes.shards_)) return {0, status_t::contended_k};
        return {written, status_t::success_k};
    }

  private:

    /** The shard and row the next read starts at, under the epoch the cursor opened in. */
    shard_cursor<typename indexes_t::shards_t> position_;
};

} // namespace unum::usearch::basic
