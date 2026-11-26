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
 *  a reshard emptying its sources - take its unique side or answer @c contended_k. Readers take no
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
#include <new>          // `std::nothrow` - a shard built on its allocator
#include <shared_mutex> // `std::shared_lock` - the mutators' hold on the gate
#include <span>         // `std::span` - the caller's watermarks, keys & distances
#include <type_traits>  // `std::is_same_v` - the unpredicated walk
#include <utility>      // `std::move`, `std::exchange`

#include "usearch/utilities/types.hpp"       // `shared_mutex` - the gate
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the plans' shard plans
#include "usearch/utilities/ram_shards.hpp"  // `shard_set` - the shard table
#include "usearch/utilities/ram_walks.hpp"   // `cursor_walk` - the cursor's walk
#include "usearch/utilities/sample.hpp"      // `sample_plan` - the SAMPLE verb's roots, paths & regions
#include "usearch/basic/ram_index.hpp"

namespace unum::usearch::basic {

template <typename indexes_type_>
struct ram_indexes_cursor;

template <typename indexes_type_>
struct ram_checkpoint_residency;

/**
 *  @brief Basic RAM shards viewed as one: insertions routed over the shard generations, searches
 *      merged across them, lookups fanned out.
 *
 *  @tparam index_type_ The shard, a @c basic::ram_index with 32-bit node IDs, whose allocator
 *      places it and the shard table alike.
 *
 *  The shard's waiter and atomic reference serve the shard table too, so the defaults hold: the
 *  standard reference posts what ForkUnion's spell natively through @c atomic_add. Node IDs are
 *  32 bits wide, as the shard table locates every node in 8 bytes; a lone @c basic::ram_index
 *  takes either width.
 */
template <typename index_type_>
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct ram_indexes {
    friend struct ram_indexes_cursor<ram_indexes>;
    friend struct ram_checkpoint_residency<ram_indexes>;

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
    using reshard_config_t = typename index_t::reshard_config_t;

    template <typename context_allocator_type_>
    using traversal_context_over = typename index_t::template traversal_context_over<context_allocator_type_>;

    template <typename word_type_>
    using atomic_ref = typename index_t::template atomic_ref<word_type_>;

    /** The shards and the rotation that routes new entries; the keys stay in their records. */
    using shards_t = shard_set<index_t, index_t::template atomic_ref, wait_t, allocator_t>;
    using shard_t = typename shards_t::shard_t;
    using batch_split_t = typename shards_t::batch_split_t;

  private:
    using shared_mutex_t = shared_mutex<wait_t, index_t::template atomic_ref>;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;
    using shard_insert_plan_t = typename index_t::insert_plan_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;
    using shard_refine_plan_t = typename index_t::refine_plan_t;
    using shard_reshard_plan_t = typename index_t::reshard_plan_t;

    /** Resolves a node ID walked in one shard to the key its record holds, as a predicate and a
     *  merge both read it. */
    struct key_of_node_t {
        index_t const* shard = nullptr;

        key_t operator()(node_id_t node_id) const noexcept { return shard->nodes_[node_id].key; }
        key_t operator[](node_id_t node_id) const noexcept { return (*this)(node_id); }
    };

    /** The engine relation of a walk that brings none, for every shard alike. */
    struct unrelated_t {
        dummy_predicate_t operator()(index_t const&) const noexcept { return {}; }
    };

    /** Every shard as SAMPLE reads it below the watermarks of @c watermarks_snapshot, scored by the
     *  caller's metric; see @c a_sample_source. */
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
            return indexes.shards_[node.shard].index->nodes_[node.node].key;
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
            requires a_coordinates_provider<coordinates_type_>
        {
            fixed_coordinates_of(value_(node), xyz);
        }
    };

  public:
#pragma region Placement and Measures

    /** Every index names the host allocator its shard table sits on and the line its shards
     *  prefetch by; nothing is placed by default. */
    explicit ram_indexes(allocator_t allocator, std::size_t line_bytes = default_alignment_k) noexcept
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
     *  @brief Appends one empty shard on @p allocator; refused once storage exists.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index already holds storage, or has a shard generation in
     *    flight.
     */
    status_t emplace(allocator_t const& allocator) noexcept {
        std::unique_ptr<index_t> shard = build_(allocator);
        if (!shard) return status_code_t::out_of_memory_k;
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
     *  claims or a reshard holding the index refuses it with @c contended_k, and an index holding
     *  no nodes stays as it is. The fresh shards open before any retires, so a refused build
     *  answers @c out_of_memory_k with every node in place. A search it overlaps reads values
     *  nothing frees, and retries.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, or a batch, a refine or a reshard holds the claims
     *    of a shard it names.
     */
    status_t clear() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_code_t::contended_k;
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
        return status_code_t::success_k;
    }

#pragma endregion Placement and Measures

#pragma region Insertion

    /**
     *  @brief One batch in flight across the shards: one shard plan per shard of the open shard
     *      generation and of every shard generation the batch opened, so its entries can only be
     *      placed, and @ref finish publishes them all at once.
     *
     *  Entries are split across the open shard generation's shards round-robin from the starting
     *  shard on, and entries past what fits under the bound fill fresh shard generations in turn,
     *  split the same way. Every entry must be placed before @ref finish, since each shard plan
     *  holds its shard's later turns until then. Each task and @ref finish hold the gate's shared
     *  side for their own call.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(allocator_t allocator) noexcept : shard_plans_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : shard_plans_(std::move(other.shard_plans_)), batch_split_(other.batch_split_),
              owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /**
         *  @brief Places entries @p first onward, one per value of @p values under its key in
         *      @p keys, each in the shard its position falls to, as the shard plan's own task does.
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
            if (!owner_) return status_code_t::uninitialized_k;
            std::size_t touched = 0;
            for (shard_insert_plan_t& shard_plan : shard_plans_) {
                if (atomic_ref<std::size_t>(shard_plan.placed).load(std::memory_order_acquire) !=
                    shard_plan.range.size())
                    return status_code_t::config_mismatch_k;
                touched += !shard_plan.range.empty();
            }
            shared_lock_t const held {owner_->gate_};
            // Shards publish in order, so a turn waited for here is one an earlier shard of another
            // batch already published, and no two batches wait on each other.
            status_t status = status_code_t::success_k;
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
        growing_buffer<shard_insert_plan_t, allocator_t> shard_plans_;

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
     *  - @c contended_k A reshard holds the index closed.
     *  - @c overflow_k A racing batch took the room this one was routed into.
     */
    expected<insert_plan_t> plan_insert(std::size_t count) noexcept {
        if (shards_.empty()) return {{}, status_code_t::uninitialized_k};
        shared_lock_t const held {gate_};
        if (shards_.sealed()) return {{}, status_code_t::contended_k};
        insert_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        if (!count) return {std::move(plan), status_code_t::success_k};
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            count, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.batch_split_ = batch_split;

        // Every shard claims before any dispenses, so a refusal gives back the claims before it as
        // they die, and no shard is left holding IDs that would never publish.
        std::size_t const shards = batch_split.shards_spanned();
        growing_buffer<shard_capacity_claim_t, allocator_t> claims {shards_.allocator()};
        if (failed(claims.reserve_exactly(shards)) || failed(plan.shard_plans_.reserve_exactly(shards)))
            return {{}, status_code_t::out_of_memory_k};
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
        return {std::move(plan), status_code_t::success_k};
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
     *  @param[in] hints Optional keys whose nodes every shard's walk takes as landmarks ahead of
     *      its random draws, in hint order; a key no shard holds counts as a missing hint.
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
                                    traversal_context_over<context_allocator_type_>& context,
                                    std::span<key_t const> hints = {}) noexcept {
        using relation_t = std::invoke_result_t<relate_type_ const&, index_t const&>;
        // A query takes a landmark list's worth of keys, as the context reserved; every shard's
        // staging marks the keys it held, and the keys none marked are missing.
        std::span<key_t const> const taken = hints.first(min_of(hints.size(), context.hinted_held.capacity()));
        if (failed(context.hinted_held.resize(taken.size()))) return {0, status_code_t::out_of_memory_k};
        auto const [merged, status] = merge_shards_(
            watermarks, keys, distances,
            [&](index_t& one, node_id_t watermark, std::size_t) noexcept -> expected<std::span<scored_node_t const>> {
                std::span<node_id_t const> const hinted = one.stage_hints_(taken, watermark, context);
                if constexpr (std::is_same_v<relation_t, dummy_predicate_t> &&
                              std::is_same_v<predicate_type_, dummy_predicate_t>) {
                    status_t const walked = one.template fold_walk_below_<synchronization_t::concurrent_k>(
                                                   query, metric, dummy_fold_t {}, context.results, context, config,
                                                   watermark, hinted)
                                                .status;
                    return {ranking_of_(context.results), walked};
                }
                else {
                    relation_t relation = relate(one);
                    node_predicate admits {predicate, key_of_node_t {&one}};
                    context.admitted.clear();
                    predicate_fold fold {relation, admits, context.admitted, one.non_zero_expansion_factor(config)};
                    status_t const walked = one.template fold_walk_below_<synchronization_t::concurrent_k>(
                                                   query, metric, fold, context.admitted, context, config, watermark,
                                                   hinted)
                                                .status;
                    return {ranking_of_(context.admitted), walked};
                }
            });
        for (u8_t const held : context.hinted_held) context.telemetry.missing_hints += !held;
        return {merged, status};
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
                                    traversal_context_over<context_allocator_type_>& context,
                                    std::span<key_t const> hints = {}) noexcept {
        return search_if(query, watermarks, keys, distances, config, unrelated_t {}, predicate, metric, context, hints);
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
        if (!wanted) return {0, status_code_t::success_k};
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
     *  - @c out_of_memory_k The scratch cannot be allocated.
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
            return {{}, status_code_t::out_of_memory_k};
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(scratch.data());
        byte_t* const aligned = reinterpret_cast<byte_t*>(
            round_up_to_multiple<std::uintptr_t>(address, default_alignment_k));
        return shards_.read_retrying_once([&]() noexcept -> expected<sample_result_t> {
            if (watermarks.size() > shards_.size()) return {{}, status_code_t::contended_k};
            for (std::size_t shard = 0; shard != watermarks.size(); ++shard)
                if (watermarks[shard] > shards_[shard].index->size()) return {{}, status_code_t::contended_k};
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
     *  @brief One refine in flight: every live shard runs one round through its own plan.
     *
     *  Tasks of a phase are (shard, position) pairs over the shard plans' own tasks, so any number
     *  of threads may @ref run them at once; @ref advance is the single-caller barrier, advancing
     *  every shard plan together. The plan keeps this index sealed for its whole life, and its
     *  death reopens it.
     */
    struct [[nodiscard]] refine_plan_t {

        /** One task: the shard whose plan runs it, and its position among that plan's tasks.
         *  Two words. */
        struct task_t {
            std::size_t shard = 0;
            std::size_t position = 0;
        };
        static_assert(sizeof(task_t) == 2 * sizeof(std::size_t));

        /** What the shard plans' audits judge, summed: the published nodes. */
        struct health_t {
            std::size_t visible = 0;
        };

        /** Tasks of the current phase a driver already ran, so a cancelled call resumes at the next
         *  one. */
        std::size_t taken = 0;

        refine_plan_t() noexcept : refine_plan_t(allocator_t {}) {}

        /** An unbound plan over @p allocator; only @ref plan_refine fills it. */
        explicit refine_plan_t(allocator_t allocator) noexcept : shard_plans_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : taken(other.taken), shard_plans_(std::move(other.shard_plans_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;
        ~refine_plan_t() noexcept { release_(); }

        /** Task @p position of the current phase; past the last shard beyond the end. */
        task_t operator[](std::size_t position) const noexcept { return task_of(shard_plans_, position); }

        /** The current phase's tasks over every shard plan, positionally. */
        indexed_range<refine_plan_t> tasks() const noexcept { return {this, tasks_count(shard_plans_)}; }

        /**
         *  @brief Runs @p task on the calling thread through its shard's plan, as that plan's own
         *      @c run does; a task past the end runs nothing.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The task's shard plan is bound to no index.
         */
        template <typename metric_type_, typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
        status_t run(task_t const& task, metric_type_& metric, traversal_context_over<context_allocator_type_>& context,
                     selector_type_ const& selector = {}) noexcept {
            if (task.shard >= shard_plans_.size()) return status_code_t::success_k;
            shard_refine_plan_t& shard_plan = shard_plans_[task.shard];
            return shard_plan.run(shard_plan[task.position], metric, context, selector);
        }

        /** The barrier between phases, one caller after every task of the current one ran: advances
         *  every shard plan and answers whether any has another phase. */
        bool advance() noexcept {
            if (!owner_) return false;
            std::size_t follows = 0;
            for (shard_refine_plan_t& shard_plan : shard_plans_) follows += shard_plan.advance();
            return follows != 0;
        }

        /** Stops every shard plan's round at the barrier it stands at, its closing health check
         *  running on. */
        void stop() noexcept {
            for (shard_refine_plan_t& shard_plan : shard_plans_) shard_plan.stop();
            taken = 0;
        }

        /** The published nodes every shard plan's audit judges. */
        health_t health() const noexcept {
            health_t total;
            for (shard_refine_plan_t const& shard_plan : shard_plans_) total.visible += shard_plan.health.visible;
            return total;
        }

        /** The budget planning reserved every neighbor list for, which every shard shares. */
        std::size_t connectivity() const noexcept { return widest_connectivity(shard_plans_); }

        /** Units of the phase the first rounding shard plan is in, done and in all, summed over the
         *  shards in it: what a driver's progress hears at every barrier. */
        std::size_t completed() const noexcept { return lead_phase_units(shard_plans_).completed; }
        std::size_t total() const noexcept { return lead_phase_units(shard_plans_).total; }

        /**
         *  @brief Grows @p context to what any shard plan's gather or restitch task touches, which
         *      a driver calls before the plan's first task.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k @p context cannot grow to a round's scratch.
         */
        template <typename context_allocator_type_>
        status_t reserve(traversal_context_over<context_allocator_type_>& context) const noexcept {
            return reserve_contexts(shard_plans_, context);
        }

        /** The phase the shards' rounds are in: the first shard still rounding names it. */
        phase_t phase() const noexcept { return phase_of(shard_plans_); }

        /** Where that phase sits among a round's six; zero once every shard's round ended. */
        std::size_t ordinal() const noexcept { return ordinal_of(shard_plans_); }

        /** Opens every shard's next round once each one's ended, from the roots their closing
         *  health checks kept; a no-op while any shard is mid-round. The index stays sealed. */
        void next_round() noexcept {
            if (next_rounds(shard_plans_)) taken = 0;
        }

        /** Invokes @p visit with every shard and node range the rounds rewrote, once the plan
         *  finished and before it dies, so a rule the round skips, such as a modality's long links,
         *  is drawn again over them. */
        template <typename visit_type_>
        void for_each_shard_range(visit_type_&& visit) const noexcept {
            if (owner_) visit_rounded_shards(shard_plans_, *owner_, visit);
        }

        /**
         *  @brief The pass so far, summed over the shard plans: complete once @ref advance answered
         *      false, otherwise @c cancelled_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         *  - @c cancelled_k A shard plan stopped before @ref advance answered false.
         */
        expected<refine_report_t> finish() const noexcept { return finish_rounds(shard_plans_, owner_); }

      private:
        friend ram_indexes;

        /** Drops the shard plans, which reopen their shards, then reopens this index. */
        void release_() noexcept {
            shard_plans_.clear();
            if (owner_) std::exchange(owner_, nullptr)->shards_.reopen();
        }

        /** One plan per shard of this index, unbound where it runs no round. */
        growing_buffer<shard_refine_plan_t, allocator_t> shard_plans_;

        /** The index refined, sealed while the plan lives; null in an unbound plan. */
        ram_indexes* owner_ = nullptr;
    };

    /**
     *  @brief One reshard in flight: every source shard moves into a shard of this index, or every
     *      shard of this index splits into the same shard of every target, and then every target
     *      shard runs one round through its own plan.
     *
     *  Tasks, barriers and progress run as @ref refine_plan_t's do. While any shard plan moves,
     *  nodes move out of the sources, or out of this index in a split or when its own shards
     *  repack, so their readers wait out a window spanning the move phases, and every receiving
     *  index publishes every shard's nodes inside one commit. The plan owns its source and target
     *  lists, copied at planning, and holds every source's unique side, or this index's and every
     *  target's in a split, and this index sealed for its whole life; its death reopens this index,
     *  every target and every source, an emptied one empty and live.
     */
    struct [[nodiscard]] reshard_plan_t {
        using task_t = typename refine_plan_t::task_t;
        using health_t = typename refine_plan_t::health_t;

        /** Tasks of the current phase a driver already ran, so a cancelled call resumes at the next
         *  one. */
        std::size_t taken = 0;

        reshard_plan_t() noexcept : reshard_plan_t(allocator_t {}) {}

        /** An unbound plan over @p allocator; only @ref plan_reshard fills it. */
        explicit reshard_plan_t(allocator_t allocator) noexcept
            : shard_plans_(allocator), sources_(allocator), source_holds_(allocator), targets_(allocator),
              target_holds_(allocator) {}
        reshard_plan_t(reshard_plan_t&& other) noexcept
            : taken(other.taken), shard_plans_(std::move(other.shard_plans_)), sources_(std::move(other.sources_)),
              source_holds_(std::move(other.source_holds_)), targets_(std::move(other.targets_)),
              target_holds_(std::move(other.target_holds_)), owner_hold_(std::move(other.owner_hold_)),
              own_moves_(other.own_moves_), owner_(std::exchange(other.owner_, nullptr)) {}
        reshard_plan_t& operator=(reshard_plan_t&&) = delete;
        reshard_plan_t(reshard_plan_t const&) = delete;
        reshard_plan_t& operator=(reshard_plan_t const&) = delete;
        ~reshard_plan_t() noexcept { release_(); }

        /** Task @p position of the current phase; past the last shard beyond the end. */
        task_t operator[](std::size_t position) const noexcept { return task_of(shard_plans_, position); }

        /** The current phase's tasks over every shard plan, positionally. */
        indexed_range<reshard_plan_t> tasks() const noexcept { return {this, tasks_count(shard_plans_)}; }

        /**
         *  @brief Runs @p task on the calling thread through its shard's plan, as that plan's own
         *      @c run does; a task past the end runs nothing.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The task's shard plan is bound to no index.
         */
        template <typename metric_type_, typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
        status_t run(task_t const& task, metric_type_& metric, traversal_context_over<context_allocator_type_>& context,
                     selector_type_ const& selector = {}) noexcept {
            if (task.shard >= shard_plans_.size()) return status_code_t::success_k;
            shard_reshard_plan_t& shard_plan = shard_plans_[task.shard];
            return shard_plan.run(shard_plan[task.position], metric, context, selector);
        }

        /** The barrier between phases, one caller after every task of the current one ran: advances
         *  every shard plan, publishing what moved inside one commit of every receiving index, and
         *  answers whether any has another phase. */
        bool advance() noexcept {
            if (!owner_) return false;
            std::size_t const moving_before = moving_();
            if (moving_before) for_each_receiving_([](shards_t& shards) noexcept { shards.begin_commit(); });
            std::size_t follows = 0;
            for (shard_reshard_plan_t& shard_plan : shard_plans_) follows += shard_plan.advance();
            if (moving_before) for_each_receiving_([](shards_t& shards) noexcept { shards.end_commit(); });
            if (moving_before && !moving_()) {
                if (emptying_owner_()) owner_->shards_.end_window();
                release_emptied_sources_();
            }
            return follows != 0;
        }

        /** A no-op: the rounds after moves run to their end, since only their restitch links in a
         *  moved part nothing links into. */
        void stop() noexcept {}

        /** The published nodes every shard plan's audit judges. */
        health_t health() const noexcept {
            health_t total;
            for (shard_reshard_plan_t const& shard_plan : shard_plans_) total.visible += shard_plan.health().visible;
            return total;
        }

        /** The budget planning reserved every neighbor list for, which every shard shares. */
        std::size_t connectivity() const noexcept { return widest_connectivity(shard_plans_); }

        /** Units of the phase the first rounding shard plan is in, done and in all, summed over the
         *  shards in it; none in the moves. */
        std::size_t completed() const noexcept { return lead_phase_units(shard_plans_).completed; }
        std::size_t total() const noexcept { return lead_phase_units(shard_plans_).total; }

        /** Grows @p context to what any shard plan's round task touches. */
        template <typename context_allocator_type_>
        status_t reserve(traversal_context_over<context_allocator_type_>& context) const noexcept {
            return reserve_contexts(shard_plans_, context);
        }

        /** Whether any shard plan still moves nodes, which no stop skips. */
        bool moving() const noexcept { return moving_() != 0; }

        /** The phase the plan is in: the first shard still moving or rounding names it. */
        phase_t phase() const noexcept {
            for (shard_reshard_plan_t const& shard_plan : shard_plans_)
                if (shard_plan.moving()) return shard_plan.phase();
            return phase_of(shard_plans_);
        }

        /** Where the rounds' phase sits among a round's six; zero in the moves and once every
         *  shard's round ended. */
        std::size_t ordinal() const noexcept { return ordinal_of(shard_plans_); }

        /** Opens every target shard's next round once each one's ended; a no-op in the moves and
         *  while any shard is mid-round. */
        void next_round() noexcept {
            if (!moving_() && next_rounds(shard_plans_)) taken = 0;
        }

        /** Invokes @p visit with every shard and node range the plan rewrote, once it finished and
         *  before it dies: every node of every target shard that one of its rounds ran over. */
        template <typename visit_type_>
        void for_each_shard_range(visit_type_&& visit) const noexcept {
            if (!owner_) return;
            if (targets_.size() == 1) return visit_rounded_shards(shard_plans_, *owner_, visit);
            for (ram_indexes* target : targets_)
                for (index_t& one : target->shards_.live_indexes())
                    visit(one, iota_view<node_id_t> {0, static_cast<node_id_t>(one.size())});
        }

        /** Whether every live shard of @p source moved out and emptied it: once the move phases
         *  end, the plan drops its hold, and its death reopens the source empty and live. */
        bool moved(ram_indexes const* source) const noexcept {
            for (ram_indexes const* listed : sources_)
                if (listed == source) return emptied_(*source);
            return false;
        }

        /**
         *  @brief The pass so far, summed over the shard plans: complete once @ref advance answered
         *      false, otherwise @c cancelled_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         *  - @c cancelled_k A shard plan stopped before @ref advance answered false.
         */
        expected<refine_report_t> finish() const noexcept { return finish_rounds(shard_plans_, owner_); }

      private:
        friend ram_indexes;

        /** Shard plans in their move phases; the sources' windows stay open while any is. */
        std::size_t moving_() const noexcept {
            std::size_t moving = 0;
            for (shard_reshard_plan_t const& shard_plan : shard_plans_) moving += shard_plan.moving();
            return moving;
        }

        /** Whether the moves empty shards of this index: a split's, or a pack's whose own shards
         *  repack. */
        bool emptying_owner_() const noexcept { return targets_.size() > 1 || own_moves_ != 0; }

        /** Calls @p visit with the shard set of every index the moves publish into: this one in a
         *  pack or a split gathering others first, and every target in a split. */
        template <typename visit_type_>
        void for_each_receiving_(visit_type_&& visit) noexcept {
            if (targets_.size() == 1 || !sources_.empty()) visit(owner_->shards_);
            if (targets_.size() == 1) return;
            for (ram_indexes* target : targets_) visit(target->shards_);
        }

        /** Opens the windows of every index the moves empty: the sources, and this one in a split
         *  or a repack. */
        void begin_windows_() noexcept {
            for (std::size_t source = 0; source != source_holds_.size(); ++source)
                sources_[source]->shards_.begin_window();
            if (emptying_owner_()) owner_->shards_.begin_window();
        }

        /** Closes the window of every source still held and this index's own, drops the shard
         *  plans, which reopen their shards and release the source and target shards, reopens
         *  every source, an emptied one once the shard plans dropped, every target closed, then
         *  this index, and drops every hold. */
        void release_() noexcept {
            for (std::size_t held = 0; held != source_holds_.size(); ++held)
                if (source_holds_[held]) sources_[held]->shards_.end_window();
            if (owner_ && emptying_owner_() && moving_()) owner_->shards_.end_window();
            for (std::size_t held = 0; held != source_holds_.size(); ++held)
                if (source_holds_[held] && emptied_(*sources_[held])) source_holds_[held].unlock();
                else if (source_holds_[held]) sources_[held]->shards_.reopen();
            shard_plans_.clear();
            // An emptied source reopens once no shard plan reads its shards, empty and live.
            for (std::size_t held = 0; held != source_holds_.size(); ++held)
                if (!source_holds_[held]) sources_[held]->shards_.reopen();
            source_holds_.clear();
            for (std::size_t held = 0; held != target_holds_.size(); ++held) targets_[held]->shards_.reopen();
            target_holds_.clear();
            if (owner_) std::exchange(owner_, nullptr)->shards_.reopen();
            owner_hold_ = {};
        }

        /** Once the move phases end, closes the window and drops the hold of every source whose
         *  shards all moved in; a source still to move keeps its window open. */
        void release_emptied_sources_() noexcept {
            for (std::size_t held = 0; held != source_holds_.size(); ++held)
                if (source_holds_[held] && emptied_(*sources_[held])) {
                    sources_[held]->shards_.end_window();
                    source_holds_[held].unlock();
                }
        }

        /** Whether every live shard of @p source moved out through some shard plan; a source never
         *  placed has nothing to move. */
        bool emptied_(ram_indexes const& source) const noexcept {
            if (source.shards_.empty() || shard_plans_.empty()) return false;
            for (index_t const& one : source.shards_.live_indexes()) {
                bool moved = false;
                for (shard_reshard_plan_t const& shard_plan : shard_plans_) moved = moved || shard_plan.moved(&one);
                if (!moved) return false;
            }
            return true;
        }

        /** One plan per shard of this index, unbound where it runs no round. */
        growing_buffer<shard_reshard_plan_t, allocator_t> shard_plans_;

        /** The sources other than this index, copied at planning: a pack's, or those a split
         *  gathers into this index first. */
        growing_buffer<ram_indexes*, allocator_t> sources_;

        /** Every source's unique side, held and closed in source order, each dropped once the
         *  source is emptied. */
        growing_buffer<unique_lock_t, allocator_t> source_holds_;

        /** This index for a pack, or every split target, copied at planning. */
        growing_buffer<ram_indexes*, allocator_t> targets_;

        /** Every split target's unique side, held and closed in target order. */
        growing_buffer<unique_lock_t, allocator_t> target_holds_;

        /** This index's unique side, held for the plan's life while the moves empty its shards. */
        unique_lock_t owner_hold_;

        /** This index's own shards that move into a sibling. */
        std::size_t own_moves_ = 0;

        /** The index packed into or split, sealed while the plan lives; null in an unbound plan. */
        ram_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens one refinement round of every live shard in place, each through its own
     *      @c index_t::plan_refine.
     *
     *  Closes this index under its unique side and keeps it sealed for the plan's life, while
     *  searches run throughout.
     *
     *  @param[in] config The cache a core keeps, which sizes every shard's round.
     *  @param[in] selector Holds the budget every neighbor list is reserved for.
     *  @param[in] codec Measures the stored values.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the shard
     *      plans.
     *
     *  @par Statuses:
     *  - @c contended_k A call or batch in flight on this index.
     */
    template <typename selector_type_, typename codec_type_>
    expected<refine_plan_t> plan_refine(refine_config_t config, selector_type_ const& selector,
                                        codec_type_ const& codec, traversal_context_t& context) noexcept {
        if (shards_.empty()) return {{}, status_code_t::uninitialized_k};
        refine_plan_t plan {shards_.allocator()};
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return {{}, status_code_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_ = this;
        std::size_t const shards = shards_.size(), retired = shards_.retired_shards();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_code_t::out_of_memory_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (shard < retired) {
                plan.shard_plans_.push_back_reserved(shard_refine_plan_t {one.get_allocator()});
                continue;
            }
            // The set closed every shard, and the shard's own plan closes it again for its life.
            one.reopen_();
            auto [shard_plan, status] = one.plan_refine(config, selector, codec, context);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            if (failed(status)) return {{}, status};
        }
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief Opens a reshard of @p sources into @p targets: into this index alone, or out of this
     *      index, the first source, into several targets.
     *
     *  Into this index, every live source shard packs into a live shard of this index under the
     *  bound, the shards with the fewest entries first, opening shard generations as they fill, and
     *  moves in through that shard's own reshard; a zero @p config.shard_capacity keeps this
     *  index's own shards where they are, while a capacity repacks them too when this index is
     *  among the sources, a shard packed into itself staying. Every live shard that keeps its nodes
     *  then runs one round, so the finish audits the whole index. A source every shard of which
     *  moved in is emptied, and reopens empty and live once the plan dies.
     *
     *  Out of this index, every other source first packs into its live shards as into one target
     *  under its own bound, with no round; then live shard @c s splits into live shard @c s of
     *  every target through its own reshard, one graph-local part per target with its keys beside
     *  its nodes, and every source empties; every target shard then runs one round.
     *
     *  Every fallible step happens here: every index the plan reads or moves closed under its
     *  unique side, the lists copied, and every shard's plan opened. The move phases open the
     *  emptied indexes' windows, so their readers wait them out, and the barrier publishing the
     *  moves shows a search of each receiving index every shard's nodes or none.
     *
     *  @param[in] sources The indexes giving their nodes up: any for a pack into this index, or
     *      this index first for a split.
     *  @param[in] targets This index alone, or at least two empty indexes, no source among them,
     *      each with as many live shards as this one.
     *  @param[in] config The cache a core keeps, which sizes every round, the most entries a shard
     *      holds once packed, and the seed picking every shard's part roots.
     *  @param[in] selector Holds the budget every neighbor list is reserved for.
     *  @param[in] codec Measures the stored values.
     *  @param[in,out] context The planning thread's scratch; its allocator places the shard plans.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k No source, no target, or a null or repeated index in either list.
     *  - @c unsupported_k A source shard past the bound, which would have to be cut first.
     *  - @c uninitialized_k No shard is placed.
     *  - @c config_mismatch_k Neither this index the one target nor the first source of several
     *    targets, a split target that is a source, holds entries or another live shard count, or a
     *    zero budget.
     *  - @c contended_k A call or batch in flight on any index named, or another plan holding one.
     *  - @c overflow_k A part past a target shard's ceiling.
     */
    template <typename selector_type_, typename codec_type_>
    expected<reshard_plan_t> plan_reshard(std::span<ram_indexes* const> sources, std::span<ram_indexes* const> targets,
                                          reshard_config_t config, selector_type_ const& selector,
                                          codec_type_ const& codec, traversal_context_t& context) noexcept {
        if (sources.empty() || targets.empty()) return {{}, status_code_t::invalid_argument_k};
        for (std::span<ram_indexes* const> const listed : {sources, targets})
            if (status_t const refused = listed_refusal(listed); failed(refused)) return {{}, refused};
        if (shards_.empty()) return {{}, status_code_t::uninitialized_k};
        bool const packing = targets.size() == 1 && targets[0] == this;
        bool const splitting = sources[0] == this && targets.size() > 1;
        if (!packing && !splitting) return {{}, status_code_t::config_mismatch_k};
        if (splitting) return plan_split_(sources.subspan(1), targets, config, selector, codec, context);
        return plan_pack_(sources, config, selector, codec, context);
    }

    /**
     *  @brief Moves @p sources into @p targets and links them in: @ref plan_reshard, every phase
     *      drained on the calling thread with @p metric.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k As @ref plan_reshard refuses.
     *  - @c unsupported_k As @ref plan_reshard refuses.
     *  - @c uninitialized_k No shard is placed.
     *  - @c config_mismatch_k As @ref plan_reshard refuses.
     *  - @c contended_k A call or plan holds any index named.
     *  - @c overflow_k A part past a target shard's ceiling.
     */
    template <typename metric_type_, typename selector_type_, typename codec_type_>
    expected<refine_report_t> reshard(std::span<ram_indexes* const> sources, std::span<ram_indexes* const> targets,
                                      reshard_config_t config, selector_type_ const& selector, codec_type_ const& codec,
                                      metric_type_& metric, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_reshard(sources, targets, config, selector, codec, context);
        if (failed(status)) return {{}, status};
        if (status_t const sized = plan.reserve(context); failed(sized)) {
            [[maybe_unused]] auto const [report, report_status] = plan.finish();
            return {report, sized};
        }
        do
            for (typename reshard_plan_t::task_t const task : plan.tasks()) [[maybe_unused]]
                status_t const ran = plan.run(task, metric, context, selector);
        while (plan.advance());
        return plan.finish();
    }

    /**
     *  @brief Reachability of every live shard, each audited by its own @c plan_health from node
     *      zero both ways on the calling thread, summed as @c reachability_t folds.
     *
     *  Read-only, so concurrent insertions leave fresh nodes uncounted; the audit runs again once
     *  if a destructive window overlapped it, and each shard's bitsets come from the allocator
     *  @p context holds.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c out_of_memory_k A shard's bitsets could not be placed.
     *  - @c contended_k Two destructive windows overlapped the audit.
     */
    template <typename context_allocator_type_>
    expected<reachability_t> health(traversal_context_over<context_allocator_type_>& context) noexcept {
        using health_plan_t = typename index_t::template health_plan<context_allocator_type_>;
        if (shards_.empty()) return {{}, status_code_t::uninitialized_k};
        return shards_.read_retrying_once([&]() noexcept -> expected<reachability_t> {
            reachability_t total;
            for (index_t& one : shards_.live_indexes()) {
                auto [plan, plan_status] = one.plan_health(context);
                if (failed(plan_status)) return {total, plan_status};
                do
                    for (typename health_plan_t::task_t const task : plan.tasks()) [[maybe_unused]]
                        std::size_t const expanded = plan.sweep(task);
                while (plan.advance_depth());
                auto const [shard_health, finish_status] = plan.finish();
                if (failed(finish_status)) return {total, finish_status};
                total += shard_health;
            }
            return {total, status_code_t::success_k};
        });
    }

#pragma endregion Refinement

  private:

    /** Task @p position among every plan of @p plans in turn, or one past the last shard. */
    template <typename plan_type_>
    static typename refine_plan_t::task_t task_of(growing_buffer<plan_type_, allocator_t> const& plans,
                                                  std::size_t position) noexcept {
        for (std::size_t shard = 0; shard != plans.size(); ++shard) {
            std::size_t const tasks = plans[shard].tasks().size();
            if (position < tasks) return {shard, position};
            position -= tasks;
        }
        return {plans.size(), 0};
    }

    /** Tasks of the current phase over every plan of @p plans. */
    template <typename plan_type_>
    static std::size_t tasks_count(growing_buffer<plan_type_, allocator_t> const& plans) noexcept {
        std::size_t total = 0;
        for (plan_type_ const& plan : plans) total += plan.tasks().size();
        return total;
    }

    /** The widest budget any plan of @p plans reserved its neighbor lists for. */
    template <typename plan_type_>
    static std::size_t widest_connectivity(growing_buffer<plan_type_, allocator_t> const& plans) noexcept {
        std::size_t widest = 0;
        for (plan_type_ const& plan : plans) widest = max_of(widest, plan.connectivity());
        return widest;
    }

    /** Grows @p context to what any plan of @p plans touches. */
    template <typename plan_type_, typename context_allocator_type_>
    static status_t reserve_contexts(growing_buffer<plan_type_, allocator_t> const& plans,
                                     traversal_context_over<context_allocator_type_>& context) noexcept {
        for (plan_type_ const& plan : plans)
            if (status_t const sized = plan.reserve(context); failed(sized)) return sized;
        return status_code_t::success_k;
    }

    /** The phase the first plan of @p plans still rounding is in, or the first plan's once every
     *  round ended. */
    template <typename plan_type_>
    static phase_t phase_of(growing_buffer<plan_type_, allocator_t> const& plans) noexcept {
        for (plan_type_ const& plan : plans)
            if (plan.rounding()) return plan.phase();
        return plans.empty() ? phase_t::none_k : plans[0].phase();
    }

    /** Where the phase of the first plan of @p plans still rounding sits; zero once every round
     *  ended. */
    template <typename plan_type_>
    static std::size_t ordinal_of(growing_buffer<plan_type_, allocator_t> const& plans) noexcept {
        for (plan_type_ const& plan : plans)
            if (plan.rounding()) return plan.ordinal();
        return 0;
    }

    /** Opens every plan's next round once each one's ended, answering whether they opened. */
    template <typename plan_type_>
    static bool next_rounds(growing_buffer<plan_type_, allocator_t>& plans) noexcept {
        for (plan_type_ const& plan : plans)
            if (plan.rounding()) return false;
        for (plan_type_& plan : plans) plan.next_round();
        return true;
    }

    /** Invokes @p visit with every node of every shard of @p owner whose plan in @p plans ran a
     *  round; a retired shard, and one a repack emptied, have no plan of their own. */
    template <typename plan_type_, typename visit_type_>
    static void visit_rounded_shards(growing_buffer<plan_type_, allocator_t> const& plans, ram_indexes const& owner,
                                     visit_type_& visit) noexcept {
        for (std::size_t shard = 0; shard != plans.size(); ++shard) {
            if (plans[shard].finish().status == status_code_t::uninitialized_k) continue;
            index_t& one = *owner.shards_[shard].index;
            visit(one, iota_view<node_id_t> {0, static_cast<node_id_t>(one.size())});
        }
    }

    /** The rounds of @p plans over @p owner so far, summed, as both plans' @c finish report them;
     *  a plan bound to no index has no round of its own. */
    template <typename plan_type_>
    static expected<refine_report_t> finish_rounds(growing_buffer<plan_type_, allocator_t> const& plans,
                                                   ram_indexes const* owner) noexcept {
        refine_report_t total;
        status_t status = owner ? status_code_t::success_k : status_code_t::uninitialized_k;
        for (plan_type_ const& plan : plans) {
            auto const [tally, tally_status] = plan.finish();
            if (tally_status == status_code_t::uninitialized_k) continue;
            if (failed(tally_status) && succeeded(status)) status = tally_status;
            total += tally;
        }
        return {total, status};
    }

    /** What a pack's @ref plan_reshard does: closes every other source under its unique side, and
     *  this index, then packs the moving shards into live shards of this index and opens every
     *  live shard's own reshard. */
    template <typename selector_type_, typename codec_type_>
    expected<reshard_plan_t> plan_pack_(std::span<ram_indexes* const> sources, reshard_config_t config,
                                        selector_type_ const& selector, codec_type_ const& codec,
                                        traversal_context_t& context) noexcept {
        reshard_plan_t plan {shards_.allocator()};
        bool const repacking = config.shard_capacity != 0;
        std::size_t own = 0;
        if (failed(plan.sources_.reserve_exactly(sources.size())) ||
            failed(plan.source_holds_.reserve_exactly(sources.size())) || failed(plan.targets_.reserve_exactly(1)))
            return {{}, status_code_t::out_of_memory_k};
        plan.targets_.push_back_reserved(this);
        std::size_t source_shards = 0;
        for (ram_indexes* source : sources) {
            if (source == this) {
                if (repacking) own = shards_.size();
                continue;
            }
            unique_lock_t hold {source->gate_, std::try_to_lock};
            if (!hold) return {{}, status_code_t::contended_k};
            if (status_t const closed = source->shards_.close(); failed(closed)) return {{}, closed};
            plan.sources_.push_back_reserved(source);
            plan.source_holds_.push_back_reserved(std::move(hold));
            source_shards += source->shards_.size() - source->shards_.retired_shards();
        }

        unique_lock_t held {gate_, std::try_to_lock};
        if (!held) return {{}, status_code_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_ = this;

        // This index's own shards lead the moving ones when they repack, then every source's.
        std::size_t const moving = own + source_shards;
        growing_buffer<shard_t const*, allocator_t> moving_shards {shards_.allocator()};
        growing_buffer<shard_move_t, allocator_t> moves {shards_.allocator()};
        growing_buffer<index_t*, allocator_t> slots {shards_.allocator()};
        if (failed(moving_shards.reserve_exactly(moving)) || failed(moves.resize(moving)) ||
            failed(slots.reserve_exactly(moving)))
            return {{}, status_code_t::out_of_memory_k};
        for (std::size_t shard = 0; shard != own; ++shard) moving_shards.push_back_reserved(&shards_[shard]);
        for (ram_indexes const* source : plan.sources_)
            for (shard_t const& one : source->shards_) moving_shards.push_back_reserved(&one);
        auto const moving_shard = [&](std::size_t from) noexcept -> shard_t const& { return *moving_shards[from]; };
        auto const placement_of = [](shard_t const& one) noexcept { return one.index->get_allocator(); };
        auto const open = [this]() noexcept {
            return shards_.open_shard_generation_past_seal_(
                shards_.open_shard_generation(), [this](shard_t const& like) noexcept { return build_shard_(like); });
        };
        std::size_t const bound = repacking ? min_of(config.shard_capacity, shards_.shard_capacity())
                                            : shards_.shard_capacity();
        if (status_t const packed = shards_.pack(to_writable_span(moves), own, bound, moving_shard, placement_of,
                                                 rank_fewest_entries, open);
            failed(packed))
            return {{}, packed};
        for (std::size_t from = 0; from != own; ++from) plan.own_moves_ += moves[from].into != from;
        // Moving its own shards empties some of this index, which then stays held for the plan.
        if (plan.own_moves_) plan.owner_hold_ = std::move(held);

        std::size_t const shards = shards_.size(), retired = shards_.retired_shards();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_code_t::out_of_memory_k};
        for (std::size_t into = 0; into != shards; ++into) {
            index_t& one = *shards_[into].index;
            // A retired shard, and one whose nodes move into a sibling, runs no round of its own.
            if (into < retired || (into < own && moves[into].into != into)) {
                plan.shard_plans_.push_back_reserved(shard_reshard_plan_t {one.get_allocator()});
                continue;
            }
            slots.clear();
            for (std::size_t from = 0; from != moving; ++from)
                if (moves[from].into == into && (from >= own || from != into))
                    slots.push_back_reserved(moving_shards[from]->index.get());
            index_t* const target = &one;
            // The set closed every shard, and the shard's own plan closes it again for its life.
            one.reopen_();
            auto [shard_plan, status] = one.plan_reshard(to_span(slots), std::span<index_t* const> {&target, 1}, config,
                                                         selector, codec, context);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            // Dropped, the shard plans move nothing, so a refused plan closes no window it opened.
            if (failed(status)) return plan.shard_plans_.clear(), expected<reshard_plan_t> {{}, status};
        }
        if (plan.moving_()) plan.begin_windows_();
        return {std::move(plan), status_code_t::success_k};
    }

    /** What a split's @ref plan_reshard does: closes this index, every one of @p gathered and every
     *  target under their unique sides, all held until the plan dies, packs the gathered shards
     *  into this index's live shards, and opens every live shard's split, with the shards packed
     *  into it, into the same live shard of every target. */
    template <typename selector_type_, typename codec_type_>
    expected<reshard_plan_t> plan_split_(std::span<ram_indexes* const> gathered, std::span<ram_indexes* const> targets,
                                         reshard_config_t config, selector_type_ const& selector,
                                         codec_type_ const& codec, traversal_context_t& context) noexcept {
        std::size_t const parts = targets.size();
        std::array<ram_indexes const*, 1> const own {this};
        if (lists_share(own, targets) || lists_share(gathered, targets)) return {{}, status_code_t::config_mismatch_k};
        reshard_plan_t plan {shards_.allocator()};
        if (failed(plan.targets_.reserve_exactly(parts)) || failed(plan.target_holds_.reserve_exactly(parts)) ||
            failed(plan.sources_.reserve_exactly(gathered.size())) ||
            failed(plan.source_holds_.reserve_exactly(gathered.size())))
            return {{}, status_code_t::out_of_memory_k};
        for (ram_indexes* into : targets) plan.targets_.push_back_reserved(into);
        std::size_t gathered_shards = 0;
        for (ram_indexes* from : gathered) {
            unique_lock_t hold {from->gate_, std::try_to_lock};
            if (!hold) return {{}, status_code_t::contended_k};
            if (status_t const closed = from->shards_.close(); failed(closed)) return {{}, closed};
            plan.sources_.push_back_reserved(from);
            plan.source_holds_.push_back_reserved(std::move(hold));
            gathered_shards += from->shards_.size() - from->shards_.retired_shards();
        }
        unique_lock_t hold {gate_, std::try_to_lock};
        if (!hold) return {{}, status_code_t::contended_k};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        plan.owner_hold_ = std::move(hold);
        plan.owner_ = this;

        // The gathered shards pack beside this index's own nodes, which stay where they are.
        growing_buffer<shard_t const*, allocator_t> moving_shards {shards_.allocator()};
        growing_buffer<shard_move_t, allocator_t> moves {shards_.allocator()};
        if (failed(moving_shards.reserve_exactly(gathered_shards)) || failed(moves.resize(gathered_shards)))
            return {{}, status_code_t::out_of_memory_k};
        for (ram_indexes const* from : plan.sources_)
            for (shard_t const& one : from->shards_) moving_shards.push_back_reserved(&one);
        auto const moving_shard = [&](std::size_t from) noexcept -> shard_t const& { return *moving_shards[from]; };
        auto const placement_of = [](shard_t const& one) noexcept { return one.index->get_allocator(); };
        auto const open = [this]() noexcept {
            return shards_.open_shard_generation_past_seal_(
                shards_.open_shard_generation(), [this](shard_t const& like) noexcept { return build_shard_(like); });
        };
        if (status_t const packed = shards_.pack(to_writable_span(moves), 0, shards_.shard_capacity(), moving_shard,
                                                 placement_of, rank_fewest_entries, open);
            failed(packed))
            return {{}, packed};

        std::size_t const shards = live_shards_count(), retired = shards_.retired_shards();
        // A closed target takes no entry, so its emptiness and shards hold for the plan's life.
        for (ram_indexes* into : targets) {
            unique_lock_t target_hold {into->gate_, std::try_to_lock};
            if (!target_hold) return {{}, status_code_t::contended_k};
            if (status_t const closed = into->shards_.close(); failed(closed)) return {{}, closed};
            plan.target_holds_.push_back_reserved(std::move(target_hold));
            if (into->live_shards_count() != shards || into->size()) return {{}, status_code_t::config_mismatch_k};
        }
        growing_buffer<index_t*, allocator_t> into_shards {shards_.allocator()};
        growing_buffer<index_t*, allocator_t> from_shards {shards_.allocator()};
        if (failed(plan.shard_plans_.reserve_exactly(shards)) || failed(into_shards.reserve_exactly(parts)) ||
            failed(from_shards.reserve_exactly(1 + gathered_shards)))
            return {{}, status_code_t::out_of_memory_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            into_shards.clear();
            for (ram_indexes* into : targets) {
                index_t* const target_shard = into->shards_[into->shards_.retired_shards() + shard].index.get();
                // The sets closed every target shard; its own plan closes it again for its life.
                target_shard->reopen_();
                into_shards.push_back_reserved(target_shard);
            }
            from_shards.clear();
            from_shards.push_back_reserved(shards_[retired + shard].index.get());
            for (std::size_t from = 0; from != gathered_shards; ++from)
                if (moves[from].into == retired + shard)
                    from_shards.push_back_reserved(moving_shards[from]->index.get());
            reshard_config_t shard_config = config;
            shard_config.seed = config.seed + shard;
            auto [shard_plan, status] = from_shards[0]->plan_reshard(to_span(from_shards), to_span(into_shards),
                                                                     shard_config, selector, codec, context);
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
            if (failed(status)) return plan.shard_plans_.clear(), expected<reshard_plan_t> {{}, status};
        }
        plan.begin_windows_();
        return {std::move(plan), status_code_t::success_k};
    }

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
            if (watermarks.size() > shards_.size()) return {0, status_code_t::contended_k};
            std::size_t merged = 0;
            status_t status = status_code_t::success_k;
            // Retired shards and shards published empty hold a zero watermark, and walk nothing.
            for (std::size_t shard = 0; shard != watermarks.size() && succeeded(status); ++shard) {
                if (!watermarks[shard]) continue;
                index_t& one = *shards_[shard].index;
                if (watermarks[shard] > one.size()) return {0, status_code_t::contended_k};
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
    using allocator_t = typename indexes_t::allocator_t;
    using walk_t = cursor_walk<node_id_t, allocator_t>;
    using shard_node_t = typename walk_t::shard_node_t;
    using shard_cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, allocator_t>;

    /** Where a pull gathers its entries' links, in the caller's arena: the linked keys end to end,
     *  and the offset where each entry's run starts. */
    using links_allocator_t = arena_allocator<byte_t, allocator_t>;
    using links_keys_t = growing_buffer<key_t, links_allocator_t>;
    using links_offsets_t = growing_buffer<std::size_t, links_allocator_t>;

    explicit ram_indexes_cursor(allocator_t allocator) noexcept
        : shard_cursors_(allocator), walk_(allocator), allocator_(allocator) {}

    /**
     *  @brief Opens the walk over every live shard of @p indexes under the epoch it reads now; any
     *      @p starts make it breadth-first, each key starting from its lowest node in the lowest
     *      shard holding it, and a key held nowhere starting nothing.
     *
     *  The shards are taken mutable, as a shard cursor locks each record it stages.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The shard cursors, the visited bits or the frontier found no memory.
     */
    status_t open(indexes_t& indexes, std::span<key_t const> starts) noexcept {
        epoch_ = indexes.read_epoch();
        first_shard_ = indexes.shards_.retired_shards();
        std::size_t const count = indexes.shards_count() - first_shard_;
        if (failed(shard_cursors_.reserve_exactly(count))) return status_code_t::out_of_memory_k;
        for (std::size_t shard = 0; shard != count; ++shard)
            shard_cursors_.push_back_reserved(
                shard_cursor_t {*indexes.shards_[first_shard_ + shard].index, allocator_});
        if (starts.empty()) return status_code_t::success_k;
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
        return status_code_t::success_k;
    }

    /**
     *  @brief Writes up to @p keys.size() next entries' keys into @p keys, gathering each one's
     *      links into @p links_keys and @p links_offsets when both are named, and answers the
     *      number of keys written.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k No memory for a link or a newly visited entry; written entries stand.
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
                return {written, status_code_t::out_of_memory_k};
            for (node_id_t const neighbor : neighbors) {
                // Breadth-first never refreshes, so the bits span the shard cursors' snapshots.
                if (walk_.breadth_first() && neighbor < cursor.visible_count())
                    if (status_t const visited = walk_.visit(entry.shard, neighbor); failed(visited))
                        return {written, visited};
                if (!links_keys) continue;
                auto const [linked, linked_status] = shard.key_of(neighbor);
                if (succeeded(linked_status)) links_keys->push_back_reserved(linked);
            }
            if (links_keys) links_offsets->push_back_reserved(links_keys->size());
            auto const [key, key_status] = shard.key_of(entry.node);
            if (failed(key_status)) return {written, key_status};
            keys[written++] = key;
        }
        // Retired shards keep their records, so a pull that ran past a window read nothing freed.
        if (!indexes.still_valid(epoch_)) return {0, status_code_t::contended_k};
        return {written, status_code_t::success_k};
    }

  private:
    /** One per live shard at open, in placement order, each over the snapshot it took then. */
    growing_buffer<shard_cursor_t, allocator_t> shard_cursors_;

    /** The order of the walk and where it stands. */
    walk_t walk_;

    [[no_unique_address]] allocator_t allocator_;

    /** The position of the first shard walked: the first live one at open, which a clear since
     *  retires in place. */
    std::size_t first_shard_ = 0;

    /** The epoch the walk opened under; a pull after a window since answers contended. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::basic
