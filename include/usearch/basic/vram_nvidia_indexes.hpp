/**
 *  @file include/usearch/basic/vram_nvidia_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Many basic GPU shards on NVIDIA devices viewed as one, so a modality's graph grows past
 *      one device and one shard's ID space.
 *
 *  The plural over a modality's face of @c basic::vram_ragged_graph: shards share one bound, and a
 *  batch crossing it opens another shard generation on the same devices. Every shard keeps its own
 *  keys and key map on the host, so a lookup probes every shard. A search ranks every query on
 *  every shard below one snapshot of their watermarks and merges the shards' matches through their
 *  keys; a batch launches shard by shard, files its keys in every shard, and publishes on every
 *  shard inside one commit, so a search or a count sees it whole or not at all.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <functional>  // `std::hash` - the shard set's key hash
#include <limits>      // `std::numeric_limits` - the node-ID type's sentinel
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the caller's keys & counts
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move`, `std::exchange`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the shard set's words
#include <numkong/tensor.hpp>    // `nk::matrix_span` - the merged rankings

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"        // `growing_buffer` - the plans' parts & the landings
#include "usearch/utilities/ram_device.hpp"         // `forkunion_allocator_t` - the shard table's placement
#include "usearch/utilities/vram_nvidia_device.hpp" // `cuda_executor_t`, `cuda_vendor_t`
#include "usearch/utilities/ram_shards.hpp"         // `shard_set` - the shard table
#include "usearch/utilities/vram_shards.hpp"        // `vram_shard_fanout` - the contexts & pairing
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/basic/shard_checkpoints.hpp" // `save_shards`, `load_shards`

namespace unum::usearch::basic {

/**
 *  @brief Basic GPU shards viewed as one: batches routed over the shard generations, searches
 *      merged across them, lookups probing every shard.
 *
 *  @tparam index_type_ The shard, a modality's NVIDIA face over @c basic::vram_ragged_graph, with
 *      32-bit node IDs, as the shard table locates every node in 8 bytes.
 *  @tparam key_type_ The caller-visible identity, the shards' own.
 *  @tparam build_type_ Builds one unreserved shard, configured as every other, on a device:
 *      <tt>build(device) → std::unique_ptr<index_type_></tt>, null when the allocation fails.
 *  @tparam landed_score_type_ The scores a shard's search lands on the host before the merge, the
 *      shard's own distance unless a modality's search lands another.
 *  @tparam wait_type_ What a reader spanning the shards waits through while a commit is in flight.
 */
template <                                                          //
    typename index_type_,                                           //
    typename key_type_,                                             //
    typename build_type_,                                           //
    typename landed_score_type_ = typename index_type_::distance_t, //
    typename wait_type_ = dummy_waiter_t                            //
    >
    requires std::same_as<typename index_type_::node_id_t, utilities::node_id_t>
struct vram_nvidia_indexes {
    static_assert(std::same_as<typename index_type_::key_t, key_type_>, "Every shard keeps the plural's keys");
    using index_t = index_type_;
    using key_t = key_type_;
    using node_id_t = typename index_t::node_id_t;
    using distance_t = typename index_t::distance_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using insert_config_t = typename index_t::insert_config_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;
    using scored_node_t = scored_node<node_id_t, distance_t>;

    /** The shards and the rotation that routes new entries; the keys stay in the shards. */
    using shards_t = shard_set<index_t, fu::standard_atomic_ref, wait_type_>;
    using shard_t = typename shards_t::shard_t;

    /** The streams and their scratch every batch verb runs on, paired with the shards by device. */
    using executor_t = cuda_executor_t;

  private:
    using fanout_t = utilities::vram_shard_fanout<shards_t, cuda_vendor_t, landed_score_type_>;
    using traversals_t = typename fanout_t::traversals_t;
    using shard_insert_plan_t = typename index_t::insert_plan_t;

  public:
#pragma region Placement and Measures

    /** Every index names the host allocator its shard table sits on, and the builder of its
     *  shards; nothing is placed by default. */
    vram_nvidia_indexes(forkunion_allocator_t allocator, build_type_ build) noexcept
        : shards_(allocator), build_(std::move(build)) {}
    vram_nvidia_indexes(vram_nvidia_indexes const&) = delete;
    vram_nvidia_indexes& operator=(vram_nvidia_indexes const&) = delete;

    /** Bounds every shard at @p nodes, clamped to the ID space, zero meaning the ID space; past it,
     *  a batch opens another shard generation on the same devices. Ignored once storage exists. */
    void bound_shards(std::size_t nodes) noexcept { shards_.bound_shards(nodes); }

    /**
     *  Appends one shard built on CUDA device @p device.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Storage exists already: the set is reserved, retired shards, or has
     *    a shard generation in flight.
     */
    status_t emplace(int device) noexcept {
        std::unique_ptr<index_t> shard = build_(device);
        if (!shard) return status_t::bad_alloc_k;
        return shards_.emplace(std::move(shard));
    }

    /**
     *  Grows every shard ahead of @p config.capacity entries and @p config.bytes value bytes, both
     *  split evenly over the shards, since each shard's tape holds only its own nodes' values; the
     *  first call freezes the connectivity.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or the face's kits take parameters and none were
     *    set.
     *  - @c config_mismatch_k The connectivity is not a power of two, or differs from the one a
     *    shard froze.
     *  - @c overflow_k A shard's share passes its ID space, its reserved range or its host keys,
     *    or its value bytes the reserved tape.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        config.bytes = divide_round_up(config.bytes, max_of<std::size_t>(shards_.size(), 1));
        return shards_.reserve(config);
    }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed. */
    std::size_t size() const noexcept { return shards_.entries(); }

    /** Shards in the index, every shard generation's, in placement order. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** Shards past the retired ones, which every walk and checkpoint spans. */
    std::size_t live_shards_count() const noexcept { return shards_.size() - shards_.retired_shards(); }

    /** Entries the shards host without growing again. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** Bytes every shard and its keys occupy, on the devices and host. */
    std::size_t memory_usage() const noexcept { return shards_.memory_usage(); }

    /** The device the first shard sits on, where a call landing matches on a device lands them; an
     *  empty index answers the device it will be placed on, zero. */
    int first_device() const noexcept { return shards_.empty() ? 0 : shards_[0].index->device_specs().ordinal; }

    /** The shards past the retired ones, in placement order. */
    typename shards_t::template indexes_range<index_t> live_shards() noexcept { return shards_.live_indexes(); }
    typename shards_t::template indexes_range<index_t const> live_shards() const noexcept {
        return shards_.live_indexes();
    }

    /**
     *  @brief Empties the index between batches: retires every shard, which keeps what a reader
     *      still walks until the index dies, behind a fresh shard generation on the same devices.
     *
     *  Refused with @c contended_k while a batch, load or save holds a shard; a retired shard stays
     *  closed for good, and the fresh shard generation takes every later batch.
     *
     *  @par Statuses:
     *  - @c contended_k A shard has a batch or load in flight, or holds claims it never dispensed.
     */
    status_t clear() noexcept {
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

    /** Invokes @p callback as <tt>callback(device)</tt> for the device of every shard, in placement
     *  order, so a caller builds an executor driving them all. */
    template <typename callback_type_>
    void for_each_device(callback_type_&& callback) const noexcept {
        for (shard_t const& one : shards_) callback(one.index->device_specs().ordinal);
    }

    /** The first shard's kept failure, or success; asked of the shards, never cached. */
    status_t status() const noexcept {
        for (shard_t const& one : shards_)
            if (status_t const kept = one.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }

#pragma endregion Placement and Measures

#pragma region Insertion

    /**
     *  @brief One batch across the shards: every shard's claim over the entries it takes, so the
     *      one @ref insert only lands.
     *
     *  Each shard of a shard generation takes as many entries as splitting the batch across the
     *  shards round-robin from the starting shard would give it, as one contiguous run of the
     *  batch, since a shard's entries cross as one slice of the caller's tape. A plan that dies
     *  before @ref insert gives every claim back.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : shard_plans_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : shard_plans_(std::move(other.shard_plans_)), streams_(std::move(other.streams_)),
              executor_(other.executor_), config_(other.config_), first_(other.first_), entries_(other.entries_),
              owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /**
         *  @brief Lands every entry of @p tape under its key in @p keys, shard by shard, and spends
         *      the plan whatever it answers.
         *
         *  Each shard's run assigns its keys and launches and drains on its device as the shard's
         *  own plan does, leaving its nodes unpublished; every drained run then files its keys in
         *  its shard's map, the claim making it the one writer, and publishes inside one commit, so
         *  a search or a count snapshotting the shards sees the batch on all of them or none.
         *
         *  A refusal stops at its shard: the runs before it still publish with their keys, and the
         *  claims after it are given back.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or a shard was never reserved.
         *  - @c config_mismatch_k The @p keys or @p tape hold other than the planned entries.
         *  - @c unsupported_k An entry past every rung of a shard's kits.
         *  - @c overflow_k A shard's entries pass its reserved range, or their values its reserved
         *    tape.
         */
        expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape) noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (keys.size() != entries_ || tape.size() != entries_) return {{}, status_t::config_mismatch_k};
            vram_nvidia_indexes& owner = *std::exchange(owner_, nullptr);
            shards_t& shards = owner.shards_;
            insert_result_t result;
            if (!entries_) return {result, status_t::success_k};
            traversals_t traversals {shards.allocator()};
            if (status_t const readied = fanout_t::reserve_traversals(shards, traversals, *executor_, tape, config_);
                failed(readied))
                return {result, readied};
            status_t status = status_t::success_k;
            std::size_t first_offset = 0, drained = 0, launched = 0;
            for (; drained != shard_plans_.size(); ++drained) {
                shard_insert_plan_t& shard_plan = shard_plans_[drained];
                std::size_t const placed = first_ + drained, entries = shard_plan.entries;
                if (!entries) continue;
                if (status = fanout_t::enter(shards, placed); failed(status)) break;
                expected<insert_result_t> const inserted = shards[placed].index->graph_.launch_(
                    shard_plan, keys.subspan(first_offset, entries), tape.subview(first_offset, entries),
                    traversals[placed], config_);
                if (status = inserted.status; failed(status)) break;
                result += inserted.value;
                first_offset += entries;
                ++launched;
            }

            // Each drained shard's claim admits this batch alone, so no publish waits in commit,
            // and the keys file ahead of it, keeping the commit as short as the watermark stores.
            for (std::size_t plan = 0; plan != drained; ++plan)
                if (shard_insert_plan_t const& shard_plan = shard_plans_[plan]; shard_plan.entries)
                    shards[first_ + plan].index->graph_.publish_keys_(
                        {shard_plan.first_id, static_cast<node_id_t>(shard_plan.first_id + shard_plan.entries)});
            if (launched > 1) shards.begin_commit();
            for (std::size_t plan = 0; plan != drained; ++plan)
                if (shard_insert_plan_t const& shard_plan = shard_plans_[plan]; shard_plan.entries)
                    shards[first_ + plan].index->graph_.publish_(shard_plan.first_id + shard_plan.entries);
            if (launched > 1) shards.end_commit();
            result.new_size = owner.size();
            return {result, status};
        }

      private:
        friend vram_nvidia_indexes;

        /** One claim per shard from the open shard generation's first, in placement order; an empty
         *  one where no entry lands. */
        growing_buffer<shard_insert_plan_t, forkunion_allocator_t> shard_plans_;

        /** The executor's hold, kept for the plan's whole life. */
        typename executor_t::claim_t streams_;

        /** The executor the runs launch on. */
        executor_t* executor_ = nullptr;

        /** The per-shard insertion configuration, applied uniformly. */
        insert_config_t config_ {};

        /** The open shard generation's first shard at planning, which the claims start at. */
        std::size_t first_ = 0;

        /** Entries in the batch. */
        std::size_t entries_ = 0;

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once spent. */
        vram_nvidia_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens a batch of @p entries: takes the executor, routes the entries over the open
     *      shard generation and as many fresh ones as the bound makes them fill, and claims every
     *      shard's run.
     *
     *  A shard with a batch, load, refine or save in flight refuses its claim, and the whole plan
     *  with @c contended_k, giving back the claims before it.
     *
     *  @param[in] entries Entries in the batch.
     *  @param[in,out] executor The streams and their scratch, held until the plan dies.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device.
     *  - @c contended_k Another call holds the @p executor, the index is sealed, or a shard has a
     *    batch or load in flight or its claim word closed.
     *  - @c overflow_k A shard's entries pass its ceiling.
     */
    expected<insert_plan_t> plan_insert(std::size_t entries, executor_t& executor,
                                        insert_config_t config = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        insert_plan_t plan {shards_.allocator()};
        plan.entries_ = entries;
        plan.config_ = config;
        plan.executor_ = &executor;
        if (!entries) {
            plan.owner_ = this;
            return {std::move(plan), status_t::success_k};
        }
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            entries, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.first_ = batch_split.open_shard_generation.first;

        std::size_t const shards = batch_split.shards_spanned();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const shard_entries = batch_split.entries_for_shard(shard);
            if (!shard_entries) {
                plan.shard_plans_.push_back_reserved(shard_insert_plan_t {});
                continue;
            }
            auto [shard_plan,
                  status] = shards_[batch_split.open_shard_generation.first + shard].index->plan_insert(shard_entries);
            if (failed(status)) return {{}, status};
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
        }
        plan.owner_ = this;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Inserts every entry of @p tape under its key in @p keys: @ref plan_insert, then its one
     *  @c insert.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, or a shard was never reserved.
     *  - @c config_mismatch_k The @p keys are not one per entry, or the @p executor drives no
     *    stream on some shard's device.
     *  - @c contended_k Another call holds the @p executor, the index is sealed, or a shard has a
     *    batch or load in flight or its claim word closed.
     *  - @c unsupported_k An entry past every rung of a shard's kits.
     *  - @c overflow_k A shard's entries pass its ceiling, its reserved range or its host keys, or
     *    their values its reserved tape.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (keys.size() != tape.size()) return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(tape.size(), executor, config);
        if (failed(status)) return {{}, status};
        return plan.insert(keys, tape);
    }

#pragma endregion Insertion

#pragma region Checkpoints

    /** What a checkpoint does on these shards' devices, for @ref save_shards and @c load_shards:
     *  every shard's plans on its own stream of @c executor, which carry its keys too, and a load's
     *  reservation at the connectivity the checkpoint froze. */
    struct checkpoint_residency_t {
        vram_nvidia_indexes* owner = nullptr;
        executor_t* executor = nullptr;
        reserve_capacity_t reservation {};

        template <typename codec_type_>
        expected<columns_extents_t> columns_extents(shards_t& shards, std::size_t shard, codec_type_ const&) noexcept {
            typename executor_t::stream_t* const stream = stream_(shards, shard);
            if (!stream) return {{}, status_t::config_mismatch_k};
            traversal_context_t context {typename traversal_context_t::allocator_t {stream->arena, *stream}};
            return shards[shard].index->columns_extents(context);
        }
        template <typename codec_type_>
        expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                columns<key_t, node_id_t, byte_t, u64_t> columns,
                                                                codec_type_ const&) noexcept {
            typename executor_t::stream_t* const stream = stream_(shards, shard);
            if (!stream) return {{}, status_t::config_mismatch_k};
            traversal_context_t context {typename traversal_context_t::allocator_t {stream->arena, *stream}};
            return shards[shard].index->plan_save(columns, context);
        }
        expected<typename index_t::load_plan_t> open_shard_load(
            shards_t& shards, std::size_t shard,
            columns<key_t const, node_id_t const, byte_t const, u64_t const> columns) noexcept {
            typename executor_t::stream_t* const stream = stream_(shards, shard);
            if (!stream) return {{}, status_t::config_mismatch_k};
            traversal_context_t context {typename traversal_context_t::allocator_t {stream->arena, *stream}};
            return shards[shard].index->plan_load(columns, context);
        }
        expected<std::size_t> reserve(shards_t& shards, std::size_t shard, std::size_t entries) noexcept {
            reserve_capacity_t config = reservation;
            config.capacity = entries;
            config.ceiling = shards.shard_capacity();
            return shards[shard].index->reserve(config);
        }

        /**
         *  Leaves part @p part the live shard it names, a placed one, or opens shard generations on
         *  the placed shards' devices until it exists, each placed shard first frozen at the
         *  checkpoint's connectivity, which the shard generations take after it.
         *
         *  @par Statuses:
         *  - @c config_mismatch_k The checkpoint's connectivity is not a power of two, or differs
         *    from the one a placed shard froze.
         */
        status_t open_shard_for_part(shards_t& shards, std::size_t part) noexcept {
            if (shards.retired_shards() + part < shards.size()) return status_t::success_k;
            for (std::size_t shard = shards.retired_shards(); shard != shards.size(); ++shard)
                if (status_t const reserved = reserve(shards, shard, 0).status; failed(reserved)) return reserved;
            while (shards.size() <= shards.retired_shards() + part)
                if (status_t const opened = shards.open_shard_generation_(
                        shards.open_shard_generation(),
                        [this](shard_t const& like) noexcept { return owner->build_shard_(like); });
                    failed(opened))
                    return opened;
            return status_t::success_k;
        }

      private:

        /** The stream on shard @p shard 's device, made current; null when the executor drives
         *  none there. */
        typename executor_t::stream_t* stream_(shards_t const& shards, std::size_t shard) noexcept {
            std::size_t const stream = fanout_t::stream_of(shards, *executor, shard);
            if (stream == executor->size() || failed(fanout_t::enter(shards, shard))) return nullptr;
            return &executor->stream(stream);
        }
    };

    /**
     *  Writes the live shards under @p path through @c save_shards: a lone live shard's monolith,
     *  or one per live shard beside a manifest; each shard's nodes cross on its own stream of
     *  @p executor, in order.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, or none is live.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p config
     *    does not validate, an element has no interchange dtype, a value disagrees with its
     *    shape, or a segment is misaligned.
     *  - @c contended_k Another call holds the @p executor, or a mutator, a batch or an open plan
     *    holds a shard.
     *  - @c invalid_argument_k A part or manifest path does not parse.
     *  - @c overflow_k A device ends before a part or the manifest, or a column is shorter than
     *    the layout needs.
     *  - @c malformed_k A written part's header does not parse back.
     */
    template <typename codec_type_, typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, codec_type_ const& codec, executor_t& executor, save_config_t const& config,
                  std::string_view metric, progress_type_&& progress = {}) noexcept {
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        serial_executor_t in_order;
        return save_shards(shards_, checkpoint_residency_t {this, &executor}, path, codec, in_order, config, metric,
                           std::forward<progress_type_>(progress));
    }

    /**
     *  Reads the checkpoint at @p path into the placed, empty shards through @ref load_shards, one
     *  part per shard, opened past the placed ones where the parts outnumber them, reserving each
     *  at @p connectivity, which the checkpoint froze.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, or no reserve froze the connectivity.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, the shards
     *    hold entries, the @p connectivity differs from the one a shard froze, a column's dtype,
     *    shape or key width differs from what the shards and @p codec take, or a segment is
     *    misaligned.
     *  - @c contended_k Another call holds the @p executor, the set is sealed while a shard
     *    generation must open, or a mutator, batch, load or plan holds a shard.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, or a part, its header, columns or sums do
     *    not parse.
     *  - @c overflow_k The nodes pass a shard's node IDs or host keys.
     *  - @c unsupported_k An entry past every rung of the face's kits.
     */
    template <typename codec_type_, typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, codec_type_ const& codec, executor_t& executor, load_config_t const& config,
                  connectivity_t connectivity, progress_type_&& progress = {}) noexcept {
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        checkpoint_residency_t residency {this, &executor};
        residency.reservation.connectivity = connectivity;
        serial_executor_t in_order;
        return load_shards(shards_, residency, path, codec, in_order, config, std::forward<progress_type_>(progress));
    }

#pragma endregion Checkpoints

#pragma region Search and Lookups

    /**
     *  @brief Ranks every query of @p queries on every shard and merges each shard's matches into
     *      the caller's rankings through its keys, ascending.
     *
     *  @param[in] queries The queries, one tape entry of bytes each.
     *  @param[in,out] executor The streams and their scratch, claimed for the call.
     *  @param[in] config The walks' breadth; the matches per query are as many as @p keys is wide.
     *  @param[out] keys The merged keys of each query.
     *  @param[out] distances The same rankings' distances, in any type a distance converts to.
     *  @param[out] counts The matches each query holds.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, or the
     *    @p executor has no streams.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p counts
     *    is not one per query, or the row counts or widths of @p keys and @p distances disagree.
     *  - @c unsupported_layout_k A row of @p keys or @p distances is not contiguous.
     *  - @c contended_k Another call holds the @p executor.
     *  - @c unsupported_k A query past every rung of a shard's kits.
     */
    template <typename score_type_>
    expected<search_result_t> search(tape_view<byte_t const> queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts) noexcept {
        config.count = static_cast<small_size_t>(keys.extent(1));
        auto search_shard = [&](index_t& shard, nk::matrix_span<node_id_t> ids, nk::matrix_span<distance_t> scores,
                                std::span<std::size_t>, traversal_context_t& traversal, std::size_t visible) noexcept {
            return shard.graph_.search_below_(queries, ids, scores, traversal, config, visible).status;
        };
        return search_every_shard(queries.size(), executor, search_shard, keys, distances, counts, queries, config);
    }

    /**
     *  @brief Ranks @p queries_count queries on every live shard through @p search_shard, a
     *      modality's own search, and merges the shards' rankings by score into @p keys,
     *      @p scores and @p counts, as many matches per query as @p keys is wide.
     *
     *  @p search_shard is called as <tt>search_shard(index, node_ids, scores, counts, traversal,
     *  visible) → status_t</tt> on each shard holding visible nodes, landing its queries' matches
     *  in the host landing in @p landed_score_type_ scores; a query's matches end at its count or
     *  at its first sentinel ID. One snapshot bounds every shard's walk, so the merged rankings see
     *  each batch on every shard or on none, and @p usage, the arguments of a shard's
     *  @c traversal_memory_usage, sizes every context.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, or the @p executor has no streams.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p counts
     *    is not one per query, or the row counts or widths of @p keys and @p scores disagree.
     *  - @c unsupported_layout_k A row of @p keys or @p scores is not contiguous.
     *  - @c contended_k Another call holds the @p executor.
     */
    template <typename search_shard_type_, typename score_type_, typename... usage_types_>
    expected<search_result_t> search_every_shard(std::size_t queries_count, executor_t& executor,
                                                 search_shard_type_&& search_shard, nk::matrix_span<key_t> keys,
                                                 nk::matrix_span<score_type_> scores, std::span<std::size_t> counts,
                                                 usage_types_ const&... usage) noexcept {
        if (status_t const refused = fanout_t::refusal(shards_, executor, queries_count, keys, scores, counts);
            failed(refused))
            return {{}, refused};
        for (std::size_t& count : counts) count = 0;
        if (!queries_count || !keys.extent(1)) return {{}, status_t::success_k};
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(usage...); failed(begun)) return {{}, begun};
        auto search_one = [&](std::size_t shard, auto ids, auto shard_scores, std::span<std::size_t> shard_counts,
                              traversal_context_t& traversal, std::size_t visible) noexcept {
            return search_shard(*shards_[shard].index, ids, shard_scores, shard_counts, traversal, visible);
        };
        return fanout.search_every_shard(queries_count, keys.extent(1), search_one, keys, scores, counts);
    }

    /** Entries under @p key across every shard, read between destructive windows and between
     *  commits, so a batch spanning shards counts whole or not at all. */
    std::size_t count(key_t const& key) const noexcept { return shards_.count(key); }

    /** Invokes @p callback as <tt>callback(shard, node_id)</tt> for every published entry under
     *  @p key in every shard, each below the watermark it acquired. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        shards_.find_each(key, callback);
    }

#pragma endregion Search and Lookups

  private:

    /** One shard of a new shard generation on the device @p like sits on, reserved under the bound
     *  with its connectivity; a null index when the build or reservation fails. */
    shard_t build_shard_(shard_t const& like) const noexcept {
        std::unique_ptr<index_t> index = build_(like.index->device_specs().ordinal);
        reserve_capacity_t reserved {};
        reserved.connectivity = static_cast<connectivity_t>(like.index->connectivity());
        reserved.ceiling = shards_.shard_capacity();
        if (index && failed(index->reserve(reserved).status)) index.reset();
        return shard_t {std::move(index)};
    }

    /** The shards, their routing and the commit word. */
    shards_t shards_;

    /** Builds every shard this index places, alike. */
    build_type_ build_;
};

} // namespace unum::usearch::basic
