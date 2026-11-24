/**
 *  @file include/usearch/basic/vram_nvidia_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Many basic GPU shards on NVIDIA devices viewed as one, so a modality's graph grows past
 *      one device and one shard's ID space.
 *
 *  The sharded index over a modality's face of @c basic::vram_ragged_graph: shards share one bound,
 *  and a batch crossing it opens another shard generation on the same devices. Every shard keeps
 *  its own keys and key map on the host, so a lookup probes every shard. A search ranks every query
 *  on every shard below one snapshot of their watermarks and merges the shards' matches through
 *  their keys; a batch launches shard by shard, files its keys in every shard, and publishes on
 *  every shard inside one commit, so a search or a count sees it whole or not at all.
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
#include "usearch/utilities/ram_buffers.hpp"        // `growing_buffer` - the plans' parts & the moves
#include "usearch/utilities/ram_device.hpp"         // `forkunion_allocator_t` - the shard table's placement
#include "usearch/utilities/vram_nvidia_device.hpp" // `cuda_executor_t`, `cuda_vendor_t`
#include "usearch/utilities/ram_shards.hpp"         // `shard_set` - the shard table
#include "usearch/utilities/ram_walks.hpp"          // `breadth_first_walk`, `cursor_order_t` - the cursor's walk
#include "usearch/utilities/vram_shards.hpp"        // `vram_shard_fanout` - the contexts & pairing
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/basic/shard_checkpoints.hpp" // `save_shards`, `load_shards`

namespace unum::usearch::basic {

template <typename indexes_type_>
struct vram_nvidia_indexes_cursor;

/**
 *  @brief Basic GPU shards viewed as one: batches routed over the shard generations, searches
 *      merged across them, lookups probing every shard.
 *
 *  @tparam index_type_ The shard, a modality's NVIDIA face over @c basic::vram_ragged_graph, with
 *      32-bit node IDs, as the shard table locates every node in 8 bytes.
 *  @tparam key_type_ The caller-visible identity, the shards' own.
 *  @tparam build_type_ Builds one unreserved shard, configured as every other, on a device:
 *      <tt>build(device) → std::unique_ptr<index_type_></tt>, null when the allocation fails.
 *  @tparam shard_score_type_ The scores a shard's search copies to the host before the merge, the
 *      shard's own distance unless a modality's search copies another.
 *  @tparam wait_type_ What a reader spanning the shards waits through while a commit is in flight.
 */
template <                                                         //
    typename index_type_,                                          //
    typename key_type_,                                            //
    typename build_type_,                                          //
    typename shard_score_type_ = typename index_type_::distance_t, //
    typename wait_type_ = dummy_waiter_t                           //
    >
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct vram_nvidia_indexes {
    static_assert(std::same_as<typename index_type_::key_t, key_type_>, "Every shard keeps the sharded index's keys");
    friend struct vram_nvidia_indexes_cursor<vram_nvidia_indexes>;

    /** A reader's walk over every live shard's published entries: as stored in placement order,
     *  or breadth-first from the entries it opens on, each with the keys it links to. */
    using cursor_t = vram_nvidia_indexes_cursor<vram_nvidia_indexes>;

    using index_t = index_type_;
    using key_t = key_type_;
    using node_id_t = typename index_t::node_id_t;
    using distance_t = typename index_t::distance_t;
    using candidates_count_t = typename index_t::candidates_count_t;
    using traversal_context_t = typename index_t::traversal_context_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using insert_config_t = typename index_t::insert_config_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_config_t = typename index_t::search_config_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using refine_report_t = typename index_t::refine_report_t;
    using reachability_t = typename index_t::reachability_t;
    using scored_node_t = scored_node<node_id_t, distance_t>;

    /** The shards and the rotation that routes new entries; the keys stay in the shards. */
    using shards_t = shard_set<index_t, fu::standard_atomic_ref, wait_type_>;
    using shard_t = typename shards_t::shard_t;

    /** The streams and their scratch every batch verb runs on, paired with the shards by device. */
    using executor_t = cuda_executor_t;

  private:
    using fanout_t = utilities::vram_shard_fanout<shards_t, cuda_vendor_t, shard_score_type_>;
    using traversals_t = typename fanout_t::traversals_t;
    using shard_insert_plan_t = typename index_t::insert_plan_t;
    using graph_t = typename index_t::graph_t;

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
        if (!shard) return status_code_t::out_of_memory_k;
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

    /** How the last load from a path read its parts onto the devices. */
    vram_load_path_t loaded_through() const noexcept { return loaded_through_; }

    /** Shards past the retired ones, which every walk and checkpoint spans. */
    std::size_t live_shards_count() const noexcept { return shards_.size() - shards_.retired_shards(); }

    /** Entries the shards host without growing again. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** Bytes every shard and its keys occupy, on the devices and host. */
    std::size_t memory_usage() const noexcept { return shards_.memory_usage(); }

    /** The device the first shard sits on, where a call writing matches on a device writes them; an
     *  empty index answers the device it will be placed on, zero. */
    int first_device() const noexcept { return shards_.empty() ? 0 : shards_[0].index->device_specs().ordinal; }

    /** The shards past the retired ones, in placement order. */
    typename shards_t::template indexes_range<index_t> live_shards() noexcept { return shards_.live_indexes(); }
    typename shards_t::template indexes_range<index_t const> live_shards() const noexcept {
        return shards_.live_indexes();
    }

    /** The epoch a cursor opens under; @ref still_valid tells whether a destructive window opened
     *  since. */
    u64_t read_epoch() const noexcept { return shards_.read_epoch(); }

    /** Whether no destructive window opened since @p epoch, so a cursor's reads still stand. */
    bool still_valid(u64_t epoch) const noexcept { return shards_.still_valid(epoch); }

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
        return status_code_t::success_k;
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
        return status_code_t::success_k;
    }

#pragma endregion Placement and Measures

#pragma region Insertion

    /**
     *  @brief One batch across the shards: every shard's claim over the entries it takes, so the
     *      one @ref insert only appends.
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
         *  @brief Writes every entry of @p tape under its key in @p keys, shard by shard, and
         *      spends the plan whatever it answers.
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
            if (!owner_) return {{}, status_code_t::uninitialized_k};
            if (keys.size() != entries_ || tape.size() != entries_) return {{}, status_code_t::config_mismatch_k};
            vram_nvidia_indexes& owner = *std::exchange(owner_, nullptr);
            shards_t& shards = owner.shards_;
            insert_result_t result;
            if (!entries_) return {result, status_code_t::success_k};
            traversals_t traversals {shards.allocator()};
            if (status_t const readied = fanout_t::reserve_traversals(shards, traversals, *executor_, tape, config_);
                failed(readied))
                return {result, readied};
            status_t status = status_code_t::success_k;
            std::size_t first_offset = 0, drained = 0, launched = 0;
            for (; drained != shard_plans_.size(); ++drained) {
                shard_insert_plan_t& shard_plan = shard_plans_[drained];
                std::size_t const placed = first_ + drained, entries = shard_plan.entries;
                if (!entries) continue;
                if (status = fanout_t::enter(shards, placed); failed(status)) break;
                auto const [inserted, inserted_status] = shards[placed].index->graph_.launch_(
                    shard_plan, keys.subspan(first_offset, entries), tape.subview(first_offset, entries),
                    traversals[placed], config_);
                if (status = inserted_status; failed(status)) break;
                result += inserted;
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
         *  one where no entry goes. */
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
            return {std::move(plan), status_code_t::success_k};
        }
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            entries, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.first_ = batch_split.open_shard_generation.first;

        std::size_t const shards = batch_split.shards_spanned();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_code_t::out_of_memory_k};
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
        return {std::move(plan), status_code_t::success_k};
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
        if (keys.size() != tape.size()) return {{}, status_code_t::config_mismatch_k};
        auto [plan, status] = plan_insert(tape.size(), executor, config);
        if (failed(status)) return {{}, status};
        return plan.insert(keys, tape);
    }

#pragma endregion Insertion

#pragma region Checkpoints

    /** What a checkpoint does on these shards' devices, for @ref save_shards and @c load_shards:
     *  every shard's plans on its own stream of @c executor, which hold its keys too, and a load's
     *  reservation at the connectivity the checkpoint froze. */
    struct checkpoint_residency_t {
        vram_nvidia_indexes* owner = nullptr;
        executor_t* executor = nullptr;
        reserve_capacity_t reservation {};

        /** One source per mapped part of a load from a path, in part order; a file cuFile refuses
         *  leaves its source copying. */
        growing_buffer<cuda_part_source_t, forkunion_allocator_t> sources {forkunion_allocator_t {}};

        /** Opens a source per mapped part, cuFile-registered where the file takes O_DIRECT. */
        status_t open_sources(std::span<mapped_file_t> mappings) noexcept {
            if (failed(sources.resize(mappings.size()))) return status_code_t::out_of_memory_k;
            bool every_part_registered = !mappings.empty();
            for (std::size_t part = 0; part != mappings.size(); ++part) {
                sources[part] = {linux_cufile_file_t::open(executor->api(), mappings[part]), mappings[part].view()};
                every_part_registered = every_part_registered && sources[part].file;
            }
            owner->loaded_through_ = every_part_registered ? vram_load_path_t::cufile_k : vram_load_path_t::host_copy_k;
            return status_code_t::success_k;
        }

        template <typename codec_type_>
        expected<columns_extents_t> columns_extents(shards_t& shards, std::size_t shard, codec_type_ const&) noexcept {
            typename executor_t::stream_t* const stream = stream_(shards, shard);
            if (!stream) return {{}, status_code_t::config_mismatch_k};
            traversal_context_t context {typename traversal_context_t::allocator_t {stream->arena, *stream}};
            return shards[shard].index->columns_extents(context);
        }
        template <typename codec_type_>
        expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                columns<key_t, node_id_t, byte_t, u64_t> columns,
                                                                codec_type_ const&) noexcept {
            typename executor_t::stream_t* const stream = stream_(shards, shard);
            if (!stream) return {{}, status_code_t::config_mismatch_k};
            traversal_context_t context {typename traversal_context_t::allocator_t {stream->arena, *stream}};
            return shards[shard].index->plan_save(columns, context);
        }
        expected<typename index_t::load_plan_t> open_shard_load(
            shards_t& shards, std::size_t shard, safegraphs_saved_columns<node_id_t> const& saved) noexcept {
            typename executor_t::stream_t* const stream = stream_(shards, shard);
            if (!stream) return {{}, status_code_t::config_mismatch_k};
            traversal_context_t context {typename traversal_context_t::allocator_t {stream->arena, *stream}};
            return shards[shard].index->plan_load(saved, context, to_span(sources));
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
            if (shards.retired_shards() + part < shards.size()) return status_code_t::success_k;
            for (std::size_t shard = shards.retired_shards(); shard != shards.size(); ++shard)
                if (status_t const reserved = reserve(shards, shard, 0).status; failed(reserved)) return reserved;
            while (shards.size() <= shards.retired_shards() + part)
                if (status_t const opened = shards.open_shard_generation_(
                        shards.open_shard_generation(),
                        [this](shard_t const& like) noexcept { return owner->build_shard_(like); });
                    failed(opened))
                    return opened;
            return status_code_t::success_k;
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
     *  @brief Reads @p checkpoint, a path or the bytes of one in host memory, into the placed,
     *      empty shards through @ref load_shards, one part per shard, opened past the placed ones
     *      where parts outnumber them, reserving each at the @p connectivity the checkpoint froze.
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
     *  - @c io_k A read through cuFile ended short.
     */
    template <typename checkpoint_type_, typename codec_type_, typename progress_type_ = dummy_progress_t>
    status_t load(checkpoint_type_ checkpoint, codec_type_ const& codec, executor_t& executor,
                  load_config_t const& config, connectivity_t connectivity, progress_type_&& progress = {}) noexcept {
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        checkpoint_residency_t residency {this, &executor};
        residency.reservation.connectivity = connectivity;
        serial_executor_t in_order;
        return load_shards(shards_, std::move(residency), checkpoint, codec, in_order, config,
                           std::forward<progress_type_>(progress));
    }

#pragma endregion Checkpoints

#pragma region Search and Lookups

    /**
     *  @brief Ranks every query of @p queries on every shard and merges each shard's matches into
     *      the caller's rankings through its keys, ascending.
     *
     *  @param[in] queries The queries, one tape entry of bytes each.
     *  @param[out] keys The merged keys of each query.
     *  @param[out] distances The same rankings' distances, in any type a distance converts to.
     *  @param[out] counts The matches each query holds.
     *  @param[in] config The walks' breadth; the matches per query are as many as @p keys is wide.
     *  @param[in,out] executor The streams and their scratch, claimed for the call.
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
    expected<search_result_t> search(tape_view<byte_t const> queries, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<score_type_> distances, std::span<std::size_t> counts,
                                     search_config_t config, executor_t& executor) noexcept {
        config.count = static_cast<candidates_count_t>(keys.extent(1));
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
     *  @p search_shard is called as
     *  <tt>search_shard(index, node_ids, scores, counts, traversal, visible) → status_t</tt> on each shard
     *  holding visible nodes and copying its queries' matches into the host staging, their scores
     *  in @p shard_score_type_; a query's matches end at its count or at its first sentinel ID. One
     *  snapshot bounds every shard's walk, so the merged rankings see each batch on every shard or
     *  on none, and @p usage, the arguments of a shard's @c traversal_memory_usage, sizes every
     *  context of the call.
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
        if (!queries_count || !keys.extent(1)) return {{}, status_code_t::success_k};
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(queries_count, keys.extent(1), usage...); failed(begun))
            return {{}, begun};
        auto search_one = [&](std::size_t shard, auto ids, auto shard_scores, std::span<std::size_t> shard_counts,
                              traversal_context_t& traversal, std::size_t visible) noexcept {
            return search_shard(*shards_[shard].index, ids, shard_scores, shard_counts, traversal, visible);
        };
        return fanout.search_every_shard(queries_count, search_one, keys, scores, counts);
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

#pragma region Refinement

    /**
     *  @brief Checks the health of every shard, every shard's batch of passes queued on its own
     *      device before any waits: each shard's root component, depths and islands, as its own
     *      @c health finds them, folded over the shards.
     *
     *  Each shard checks the nodes it published when its check began. Read-only, and bracketed by
     *  the epoch as every reader is.
     *
     *  @param[in,out] executor The streams and their scratch, one per shard's device.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, or the @p executor has no streams.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k The shared kernels lack the node-ID type.
     *  - @c out_of_memory_k A shard's columns or counters cannot be placed.
     */
    expected<reachability_t> health(executor_t& executor) noexcept {
        using health_plan_t = typename graph_t::health_plan_t;
        using round_shape_t = typename graph_t::round_shape_t;
        return shards_.read_retrying_once([&]() noexcept -> expected<reachability_t> {
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};
            std::size_t widest = 0;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard)
                widest = max_of(widest, shards_[shard].index->size());
            traversals_t traversals {shards_.allocator()};
            if (status_t const ready = fanout_t::reserve_traversals(shards_, traversals, executor, refine_config_t {},
                                                                    widest);
                failed(ready))
                return {{}, ready};
            growing_buffer<health_plan_t, forkunion_allocator_t> checks {shards_.allocator()};
            if (failed(checks.reserve_exactly(traversals.size()))) return {{}, status_code_t::out_of_memory_k};
            for (std::size_t shard = 0; shard != traversals.size(); ++shard) {
                if (status_t const entered = fanout_t::enter(shards_, shard); failed(entered)) return {{}, entered};
                graph_t& graph = shards_[shard].index->graph_;
                traversal_context_t& context = traversals[shard];
                checks.push_back_reserved(health_plan_t {graph_t::empty_round_scratch_(context)});
                std::size_t const entries = graph.size();
                if (!entries) continue;
                if (status_t const refused = graph.graph_kernels_(context).status; failed(refused))
                    return {{}, refused};
                if (status_t const carved = graph.carve_health_(checks.back(), context, round_shape_t {entries});
                    failed(carved))
                    return {{}, carved};
                checks.back().rebind(graph, static_cast<node_id_t>(entries));
            }
            // Every shard's batch queues before any waits, so the devices check at once.
            for (bool running = true; running;) {
                for (std::size_t shard = 0; shard != checks.size(); ++shard) {
                    if (!checks[shard].running()) continue;
                    if (status_t const entered = fanout_t::enter(shards_, shard); failed(entered)) return {{}, entered};
                    if (status_t const queued = checks[shard].queue(traversals[shard]); failed(queued))
                        return {{}, queued};
                }
                running = false;
                for (std::size_t shard = 0; shard != checks.size(); ++shard) {
                    if (!checks[shard].running()) continue;
                    if (status_t const drained = checks[shard].wait(traversals[shard]); failed(drained))
                        return {{}, drained};
                    running |= checks[shard].advance();
                }
            }
            reachability_t result;
            for (health_plan_t const& check : checks) {
                auto const [checked, status] = check.finish();
                if (failed(status)) return {{}, status};
                result += checked;
            }
            return {result, status_code_t::success_k};
        });
    }

    /** A reshard's knobs, as every shard's graph names them. */
    using reshard_config_t = typename index_t::reshard_config_t;

    struct reshard_plan_t;

    /**
     *  @brief A refine across every live shard: one shard round each, opened on its device's stream
     *      over a run of the plan's own, and driven in lockstep, every shard's phase queued before
     *      any is awaited. Every claim word the shard plans closed reopens when they die.
     *
     *  @ref run continues it after a cancel; @ref next_round opens every shard's next round once
     *  each one's ended; @ref finish stops the rounds, closes what they leave, and reports.
     */
    struct [[nodiscard]] refine_plan_t {

        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : arenas_(allocator), traversals_(allocator), shard_plans_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : arenas_(std::move(other.arenas_)), traversals_(std::move(other.traversals_)),
              shard_plans_(std::move(other.shard_plans_)), owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        ~refine_plan_t() noexcept {
            if (owner_) release_();
        }

        /**
         *  @brief Queues every shard's next phase, then awaits them all, until no shard advances; a
         *      @c cancel_k answer from @p on_visited, asked at every barrier of a round with the
         *      units its phase did so far and in all, summed over the shards, stops before the next
         *      phase and answers @c cancelled_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or finished.
         *  - @c config_mismatch_k The @p executor drives no stream on some shard's device.
         *  - @c contended_k Another call holds the @p executor.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_code_t::uninitialized_k;
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired)) return paired;
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            return owner_->drive_shard_plans_(shard_plans_, traversals_, on_visited);
        }

        /**
         *  @brief Stops every round, runs its closing health check, sums every shard's report, and
         *      releases the plan.
         *
         *  @par Statuses:
         *  - As @ref run, and the first shard plan's failure.
         */
        expected<refine_report_t> finish(executor_t& executor) noexcept {
            if (!owner_) return {{}, status_code_t::uninitialized_k};
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired))
                return {{}, paired};
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};
            auto const [healed, healed_status] = owner_->close_shard_plans_(shard_plans_, traversals_);
            release_();
            return {healed, healed_status};
        }

        /** The reports every shard plan holds so far, folded. */
        refine_report_t running() const noexcept { return fold_shard_plans_(shard_plans_, traversals_); }

        /** The phase the shards are in, where it sits among a round's six, and its units done and
         *  in all, summed over the shards. */
        phase_t phase() const noexcept { return shard_plans_phase_(shard_plans_); }
        std::size_t ordinal() const noexcept { return shard_plans_ordinal_(shard_plans_); }
        std::size_t completed() const noexcept { return shard_plans_completed_(shard_plans_); }
        std::size_t total() const noexcept { return shard_plans_total_(shard_plans_); }

        /** Opens every shard's next round once each one's ended; a no-op while any is mid-round. */
        void next_round() noexcept { next_shard_rounds_(shard_plans_); }

      private:
        friend vram_nvidia_indexes;

        /** Drops every shard plan, whose death reopens the claim words, unbinding the plan. */
        void release_() noexcept {
            shard_plans_.clear();
            owner_ = nullptr;
        }

        /** One run per live shard, which the plan's contexts carve from, so no other call's
         *  rewind of the executor's runs reaches them. */
        growing_buffer<typename executor_t::arena_t, forkunion_allocator_t> arenas_;
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> shard_plans_;
        vram_nvidia_indexes* owner_ = nullptr;
    };

    /**
     *  @brief A reshard across every live shard, as @ref plan_reshard opens it: one shard reshard
     *      each, driven in lockstep as @ref refine_plan_t drives its rounds.
     *
     *  A gather of sources into this index appends every other source's shards into the first live
     *  shard of this index on their device while planning, so every one is consumed once the plan
     *  exists. A cut of this index moves every live shard's nodes into the matching live shard of
     *  every target while its shard plans run, and this index forgets its entries once the moves
     *  finished. Every target shard then runs its rounds.
     */
    struct [[nodiscard]] reshard_plan_t {

        explicit reshard_plan_t(forkunion_allocator_t allocator) noexcept
            : arenas_(allocator), traversals_(allocator), shard_plans_(allocator), sources_(allocator),
              targets_(allocator), slots_(allocator) {}
        reshard_plan_t(reshard_plan_t&& other) noexcept
            : arenas_(std::move(other.arenas_)), traversals_(std::move(other.traversals_)),
              shard_plans_(std::move(other.shard_plans_)), sources_(std::move(other.sources_)),
              targets_(std::move(other.targets_)), slots_(std::move(other.slots_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        reshard_plan_t& operator=(reshard_plan_t&&) = delete;
        reshard_plan_t(reshard_plan_t const&) = delete;
        reshard_plan_t& operator=(reshard_plan_t const&) = delete;

        ~reshard_plan_t() noexcept {
            if (owner_) release_();
        }

        /** Drives the shard plans, as @ref refine_plan_t::run does. */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_code_t::uninitialized_k;
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired)) return paired;
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            return owner_->drive_shard_plans_(shard_plans_, traversals_, on_visited);
        }

        /**
         *  @brief Stops every round, runs its closing health check, sums every shard's report, and
         *      releases the plan; a cut's source forgets its entries here once its moves finished.
         *
         *  @par Statuses:
         *  - As @ref refine_plan_t::finish.
         */
        expected<refine_report_t> finish(executor_t& executor) noexcept {
            if (!owner_) return {{}, status_code_t::uninitialized_k};
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired))
                return {{}, paired};
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};
            auto const [healed, healed_status] = owner_->close_shard_plans_(shard_plans_, traversals_);
            release_();
            return {healed, healed_status};
        }

        /** The reports every shard plan holds so far, folded. */
        refine_report_t running() const noexcept { return fold_shard_plans_(shard_plans_, traversals_); }

        /** Whether a cut still moves nodes into its targets. */
        bool moving() const noexcept {
            for (typename index_t::reshard_plan_t const& plan : shard_plans_)
                if (plan.moving()) return true;
            return false;
        }

        /** The shards' phase, its place among a round's six, and its units, as @ref refine_plan_t
         *  reports them. */
        phase_t phase() const noexcept { return shard_plans_phase_(shard_plans_); }
        std::size_t ordinal() const noexcept { return shard_plans_ordinal_(shard_plans_); }
        std::size_t completed() const noexcept { return shard_plans_completed_(shard_plans_); }
        std::size_t total() const noexcept { return shard_plans_total_(shard_plans_); }

        /** Opens every target shard's next round once each one's ended; a no-op mid-round. */
        void next_round() noexcept { next_shard_rounds_(shard_plans_); }

        /** Whether the nodes of @p index moved out and left it empty, consumed: a source other than
         *  the gathering index, each emptied while planning, or a cut's first source once moved. */
        bool moved(vram_nvidia_indexes const* index) const noexcept {
            bool const cutting = targets_.size() > 1;
            if (cutting && index == sources_[0]) return !shard_plans_.empty() && !moving();
            for (vram_nvidia_indexes const* source : sources_)
                if (source == index) return cutting || index != targets_[0];
            return false;
        }

      private:
        friend vram_nvidia_indexes;

        /** Drops every shard plan, whose death reopens the claim words, a finished cut's source
         *  forgetting its entries after; the plan is unbound after. */
        void release_() noexcept {
            vram_nvidia_indexes& owner = *std::exchange(owner_, nullptr);
            bool const moved = targets_.size() > 1 && !shard_plans_.empty() && !moving();
            shard_plans_.clear();
            if (moved) {
                owner.shards_.begin_window();
                owner.shards_.forget_entries();
                owner.shards_.end_window();
            }
        }

        growing_buffer<typename executor_t::arena_t, forkunion_allocator_t> arenas_;
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;
        growing_buffer<typename index_t::reshard_plan_t, forkunion_allocator_t> shard_plans_;

        /** Both lists, as the caller named them. */
        growing_buffer<vram_nvidia_indexes*, forkunion_allocator_t> sources_;
        growing_buffer<vram_nvidia_indexes*, forkunion_allocator_t> targets_;

        /** A gather's source shards grouped by the live shard taking them in, or a cut's target
         *  shards grouped by the live shard parting into them, which the shard plans point at. */
        growing_buffer<index_t*, forkunion_allocator_t> slots_;
        vram_nvidia_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens one round on every live shard, each shard's plan on its device's stream, which
     *      closes its claim word for the plan's life.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no live shards.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device.
     *  - @c contended_k A batch, a load or a plan in flight on a shard.
     *  - Those of the shards' @c plan_refine.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, executor_t& executor) noexcept {
        refine_plan_t plan {shards_.allocator()};
        status_t const opened = open_plan_(plan, config, executor, 0, 0,
                                           [&](index_t& shard, traversal_context_t& context, std::size_t) noexcept {
                                               return shard.plan_refine(config, context);
                                           });
        return {std::move(plan), opened};
    }

    /**
     *  @brief Opens a reshard of @p sources into @p targets: this index the one target of a gather
     *      of several sources, or the first source of a cut into several targets, every other
     *      source gathered into it first, its rounds dropped.
     *
     *  A gather takes every live shard of every other source into the first live shard of this
     *  index on its device, which appends its nodes and keys, empties it, and runs a round as
     *  @ref plan_refine does; every other source is consumed once the plan exists, and nothing may
     *  search one meanwhile. A cut parts every live shard's nodes among the matching live shard of
     *  every target, which then run a round each; this index forgets its entries once the moves
     *  have all finished. The plan copies both lists, and every index either names must outlive it.
     *
     *  @par Statuses:
     *  - As @ref plan_refine.
     *  - @c invalid_argument_k An empty list, a null or a repeated index within one.
     *  - @c config_mismatch_k A source among several targets, neither a gather into this index nor
     *    a cut of it, a source shard on a device none of this index's live shards sits on, or a
     *    cut target holding entries or unlike this index's live shards in count or device.
     *  - @c unsupported_k A nonzero @p config.shard_capacity, as no device reshard places shards
     *    anew.
     *  - @c overflow_k Nodes past a shard's reserved ceiling.
     */
    expected<reshard_plan_t> plan_reshard(std::span<vram_nvidia_indexes* const> sources,
                                          std::span<vram_nvidia_indexes* const> targets, reshard_config_t config,
                                          executor_t& executor) noexcept {
        reshard_plan_t plan {shards_.allocator()};
        if (sources.empty() || targets.empty()) return {std::move(plan), status_code_t::invalid_argument_k};
        for (std::span<vram_nvidia_indexes* const> const listed : {sources, targets})
            if (status_t const refused = listed_refusal(listed); failed(refused)) return {std::move(plan), refused};
        if (targets.size() > 1 && lists_share(sources, targets))
            return {std::move(plan), status_code_t::config_mismatch_k};
        if (config.shard_capacity) return {std::move(plan), status_code_t::unsupported_k};
        bool const gathering = targets.size() == 1 && targets[0] == this;
        bool const cutting = sources[0] == this && targets.size() > 1;
        if (!gathering && !cutting) return {std::move(plan), status_code_t::config_mismatch_k};
        if (failed(plan.sources_.reserve_exactly(sources.size())) ||
            failed(plan.targets_.reserve_exactly(targets.size())))
            return {std::move(plan), status_code_t::out_of_memory_k};
        for (vram_nvidia_indexes* source : sources) plan.sources_.push_back_reserved(source);
        for (vram_nvidia_indexes* target : targets) plan.targets_.push_back_reserved(target);
        if (gathering) return {std::move(plan), open_gather_(plan, config, executor)};
        // Every target answers before the gather empties a source, so a refused one moves nothing.
        if (status_t const refused = refuses_cut_targets_(targets); failed(refused)) return {std::move(plan), refused};
        if (status_t const gathered = gather_before_cut_(sources, config, executor); failed(gathered))
            return {std::move(plan), gathered};
        return {std::move(plan), open_cut_(plan, config, executor)};
    }

    /** Moves @p sources into @p targets and links them in: @ref plan_reshard, then @c run and
     *  @c finish. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> reshard(std::span<vram_nvidia_indexes* const> sources,
                                      std::span<vram_nvidia_indexes* const> targets, reshard_config_t config,
                                      executor_t& executor, progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_reshard(sources, targets, config, executor);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(ran) ? ran : finish_status};
    }

    /** Runs one round on every live shard: @ref plan_refine, then @c run and @c finish. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> refine(refine_config_t config, executor_t& executor,
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_refine(config, executor);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(ran) ? ran : finish_status};
    }

#pragma endregion Refinement

  private:

    /** The live shard at @p live, counted past the retired ones. */
    index_t& live_shard_(std::size_t live) noexcept { return *shards_[shards_.retired_shards() + live].index; }
    index_t const& live_shard_(std::size_t live) const noexcept {
        return *shards_[shards_.retired_shards() + live].index;
    }

    /** The first live shard on @p device, or the shard count when none sits there. */
    std::size_t first_live_on_(int device) const noexcept {
        for (std::size_t shard = shards_.retired_shards(); shard != shards_.size(); ++shard)
            if (shards_[shard].index->device_specs().ordinal == device) return shard;
        return shards_.size();
    }

    /** Drives every shard plan of @p plans on its context of @p traversals until none follows or
     *  @p on_visited cancels, every phase queued on every shard's device before any waits; a failed
     *  phase keeps its status in its plan, and the first one answers. Shards whose rounds end early
     *  sit idle, since a finished plan queues nothing. */
    template <typename plans_type_, typename progress_type_>
    status_t drive_shard_plans_(plans_type_& plans,
                                growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals,
                                progress_type_& on_visited) noexcept {
        status_t outcome = status_code_t::success_k;
        std::size_t const first = shards_.retired_shards();
        for (bool advancing = true; advancing;) {
            for (std::size_t shard = 0; shard != plans.size(); ++shard) {
                status_t queued = fanout_t::enter(shards_, first + shard);
                if (succeeded(queued)) queued = plans[shard].queue(traversals[shard]);
                if (failed(queued) && succeeded(outcome)) outcome = queued;
            }
            for (std::size_t shard = 0; shard != plans.size(); ++shard)
                if (status_t const waited = plans[shard].wait(traversals[shard]); failed(waited) && succeeded(outcome))
                    outcome = waited;
            // A round's every barrier is heard in its phase's own units, as the RAM drain does.
            std::size_t completed = 0, total = 0, advanced = 0;
            bool rounding = false;
            for (auto& plan : plans) {
                rounding |= plan.rounding();
                completed += plan.completed(), total += plan.total();
                advanced += plan.advance();
            }
            advancing = advanced != 0;
            if (advancing && rounding && on_visited(completed, total) == continuation_t::cancel_k)
                return status_code_t::cancelled_k;
        }
        return outcome;
    }

    /** Stops every round of @p plans, drives what they leave to their closing checks under the
     *  caller's claim of the executor, and folds every shard's report, the first failure kept. */
    template <typename plans_type_>
    expected<refine_report_t> close_shard_plans_(
        plans_type_& plans, growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals) noexcept {
        for (auto& plan : plans) plan.stop();
        dummy_progress_t closing_alone;
        status_t closed = drive_shard_plans_(plans, traversals, closing_alone);
        for (std::size_t shard = 0; shard != plans.size(); ++shard)
            if (status_t const kept = plans[shard].finish(traversals[shard]).status; failed(kept) && succeeded(closed))
                closed = kept;
        return {fold_shard_plans_(plans, traversals), closed};
    }

    /** The reports of @p plans so far, folded, the first taken whole. */
    template <typename plans_type_>
    static refine_report_t fold_shard_plans_(
        plans_type_ const& plans,
        growing_buffer<traversal_context_t, forkunion_allocator_t> const& traversals) noexcept {
        refine_report_t result;
        for (std::size_t shard = 0; shard != plans.size(); ++shard) {
            // A shard mid-round answers cancelled_k, so the fold reads its report alone.
            [[maybe_unused]] auto const [report, report_status] = plans[shard].finish(traversals[shard]);
            if (shard) result += report;
            else result = report;
        }
        return result;
    }

    /** The phase of the first of @p plans not done, or the first's once all are. */
    template <typename plans_type_>
    static phase_t shard_plans_phase_(plans_type_ const& plans) noexcept {
        for (auto const& plan : plans)
            if (plan.phase() != phase_t::done_k) return plan.phase();
        return plans.empty() ? phase_t::none_k : plans[0].phase();
    }

    /** Where the first rounding plan's phase sits among a round's six; zero once none rounds. */
    template <typename plans_type_>
    static std::size_t shard_plans_ordinal_(plans_type_ const& plans) noexcept {
        for (auto const& plan : plans)
            if (plan.rounding()) return plan.ordinal();
        return 0;
    }

    /** Units of the phases in hand done and in all, summed over @p plans. */
    template <typename plans_type_>
    static std::size_t shard_plans_completed_(plans_type_ const& plans) noexcept {
        std::size_t sum = 0;
        for (auto const& plan : plans) sum += plan.completed();
        return sum;
    }
    template <typename plans_type_>
    static std::size_t shard_plans_total_(plans_type_ const& plans) noexcept {
        std::size_t sum = 0;
        for (auto const& plan : plans) sum += plan.total();
        return sum;
    }

    /** Opens every next round of @p plans once each one's ended; a no-op while any is not done. */
    template <typename plans_type_>
    static void next_shard_rounds_(plans_type_& plans) noexcept {
        for (auto const& plan : plans)
            if (plan.phase() != phase_t::done_k) return;
        for (auto& plan : plans) plan.next_round();
    }

    /** Opens one shard plan per live shard of @p plan through @p open_shard, called as
     *  <tt>open_shard(shard, context, live)</tt>, over a context and a run of the plan's own on the
     *  shard's device's stream, sized for the widest shard grown by @p appended entries, or cut
     *  into @p targets parts when two or more. */
    template <typename plan_type_, typename open_shard_type_>
    status_t open_plan_(plan_type_& plan, refine_config_t config, executor_t& executor, std::size_t appended,
                        std::size_t targets, open_shard_type_&& open_shard) noexcept {
        std::size_t const live = live_shards_count(), first = shards_.retired_shards();
        if (!live) return status_code_t::uninitialized_k;
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        std::size_t widest = 0;
        for (index_t const& shard : live_shards()) widest = max_of(widest, shard.size());
        std::size_t walk_bytes = 0;
        for (index_t const& shard : live_shards())
            walk_bytes = max_of(walk_bytes, targets ? shard.traversal_memory_usage(config, widest, targets)
                                                    : shard.traversal_memory_usage(config, widest + appended));
        if (failed(plan.arenas_.reserve_exactly(live)) || failed(plan.traversals_.reserve_exactly(live)) ||
            failed(plan.shard_plans_.reserve_exactly(live)))
            return status_code_t::out_of_memory_k;
        plan.owner_ = this;
        for (std::size_t shard = 0; shard != live; ++shard) {
            // The run grows on the current device's stream, so that device is bound first.
            if (status_t const entered = fanout_t::enter(shards_, first + shard); failed(entered)) return entered;
            typename executor_t::stream_t& stream = executor.stream(
                fanout_t::stream_of(shards_, executor, first + shard));
            plan.arenas_.push_back_reserved(typename executor_t::arena_t {stream.arena.upstream()});
            if (failed(plan.arenas_[shard].reserve_exactly(walk_bytes))) return status_code_t::out_of_memory_k;
            plan.traversals_.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {plan.arenas_[shard], stream}});
            if (!plan.traversals_.back()) return status_code_t::out_of_memory_k;
            auto [shard_plan, status] = open_shard(*shards_[first + shard].index, plan.traversals_[shard], shard);
            if (failed(status)) return status;
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
        }
        return status_code_t::success_k;
    }

    /** Opens a gather of the plan's other sources into this index: each source shard enters the
     *  first live shard of this index on its device, and every other source forgets its entries
     *  once every shard plan opened. A live shard taking no source shard gathers only itself. */
    status_t open_gather_(reshard_plan_t& plan, reshard_config_t const& config, executor_t& executor) noexcept {
        std::size_t source_shards = 0, source_entries = 0;
        for (vram_nvidia_indexes* const source : plan.sources_) {
            if (source == this) continue;
            source_shards += source->live_shards_count();
            for (index_t const& shard : source->live_shards()) {
                if (first_live_on_(shard.device_specs().ordinal) == shards_.size())
                    return status_code_t::config_mismatch_k;
                source_entries += shard.size();
            }
        }
        if (failed(plan.slots_.reserve_exactly(source_shards + live_shards_count())))
            return status_code_t::out_of_memory_k;
        std::size_t const first = shards_.retired_shards();
        typename index_t::reshard_config_t const shard_config {config.refine, 0, config.seed};
        status_t const opened = open_plan_(
            plan, config.refine, executor, source_entries, 0,
            [&](index_t& shard, traversal_context_t& context, std::size_t live) noexcept {
                // The first live shard on a device takes every source shard there.
                std::size_t const from = plan.slots_.size();
                int const device = shard.device_specs().ordinal;
                bool const first_there = first_live_on_(device) == first + live;
                plan.slots_.push_back_reserved(&shard);
                for (vram_nvidia_indexes* const source : plan.sources_)
                    if (source != this)
                        for (index_t& given : source->live_shards())
                            if (first_there && given.device_specs().ordinal == device)
                                plan.slots_.push_back_reserved(&given);
                std::span<index_t* const> const taken {plan.slots_.data() + from, plan.slots_.size() - from};
                return shard.plan_reshard(taken, taken.first(1), shard_config, context);
            });
        if (failed(opened)) return opened;
        for (vram_nvidia_indexes* const source : plan.sources_) {
            if (source == this) continue;
            source->shards_.begin_window();
            source->shards_.forget_entries();
            source->shards_.end_window();
        }
        return status_code_t::success_k;
    }

    /** Gathers every other one of @p sources into this index before a cut, as a gather's planning
     *  appends them, its rounds dropped unrun; nothing to do with this index the only source. */
    status_t gather_before_cut_(std::span<vram_nvidia_indexes* const> sources, reshard_config_t const& config,
                                executor_t& executor) noexcept {
        if (sources.size() == 1) return status_code_t::success_k;
        reshard_plan_t gathering {shards_.allocator()};
        if (failed(gathering.sources_.reserve_exactly(sources.size())) || failed(gathering.targets_.reserve_exactly(1)))
            return status_code_t::out_of_memory_k;
        for (vram_nvidia_indexes* source : sources) gathering.sources_.push_back_reserved(source);
        gathering.targets_.push_back_reserved(this);
        return open_gather_(gathering, config, executor);
    }

    /** Whether @p targets can take a cut of this index: each empty, with as many live shards as
     *  this index, shard @c s on the device this index's shard @c s sits on; a gather into this
     *  index keeps its live shards, so the answer holds once it lands. */
    status_t refuses_cut_targets_(std::span<vram_nvidia_indexes* const> targets) const noexcept {
        std::size_t const live = live_shards_count();
        for (vram_nvidia_indexes const* const target : targets) {
            if (target->size() || target->live_shards_count() != live) return status_code_t::config_mismatch_k;
            for (std::size_t shard = 0; shard != live; ++shard)
                if (target->live_shard_(shard).device_specs().ordinal != live_shard_(shard).device_specs().ordinal)
                    return status_code_t::config_mismatch_k;
        }
        return status_code_t::success_k;
    }

    /** Opens a cut of this index into the plan's targets, which @ref refuses_cut_targets_ admitted:
     *  shard @c s parts its nodes among shard @c s of every target. */
    status_t open_cut_(reshard_plan_t& plan, reshard_config_t const& config, executor_t& executor) noexcept {
        std::size_t const live = live_shards_count(), targets = plan.targets_.size();
        if (failed(plan.slots_.reserve_exactly(live * (targets + 1)))) return status_code_t::out_of_memory_k;
        typename index_t::reshard_config_t const shard_config {config.refine, 0, config.seed};
        return open_plan_(plan, config.refine, executor, 0, targets,
                          [&](index_t& shard, traversal_context_t& context, std::size_t at) noexcept {
                              std::size_t const from = plan.slots_.size();
                              plan.slots_.push_back_reserved(&shard);
                              for (vram_nvidia_indexes* const target : plan.targets_)
                                  plan.slots_.push_back_reserved(&target->live_shard_(at));
                              std::span<index_t* const> const listed {plan.slots_.data() + from, targets + 1};
                              return shard.plan_reshard(listed.first(1), listed.subspan(1), shard_config, context);
                          });
    }

    /** One shard of a new shard generation on the device @p like sits on, reserved under the bound
     *  with its connectivity; a null index when the build or reservation fails. */
    shard_t build_shard_(shard_t const& like) noexcept {
        std::unique_ptr<index_t> index = build_(like.index->device_specs().ordinal);
        reserve_capacity_t reserved {};
        reserved.connectivity = static_cast<connectivity_t>(like.index->connectivity());
        reserved.ceiling = shards_.shard_capacity();
        if (index && failed(index->reserve(reserved).status)) index.reset();
        return shard_t {std::move(index)};
    }

    /** The shards, their routing and the commit word. */
    shards_t shards_;

    /** How the last load from a path read its parts, which @c describe reports. */
    vram_load_path_t loaded_through_ = vram_load_path_t::none_k;

    /** Builds every shard this index places, alike. */
    build_type_ build_;
};

/**
 *  @brief A reader's walk over the entries of one @c vram_nvidia_indexes as its live shards stood
 *      at open: every entry as stored, shard after shard, or breadth-first from the start entries,
 *      each entry once, each with the keys it links to.
 *
 *  Every call names the index it walks, which must be the one the cursor opened on. One shard
 *  cursor per live shard copies the neighbor rows of one block of nodes to the host whenever a seek
 *  leaves the block it holds, so the walk never blocks a writer, and links stay inside a shard, so
 *  each resolves to its key through the shard it leaves. A breadth-first walk reaches the nodes
 *  below each shard cursor's snapshot at open; a stored walk refreshes each shard's watermark as it
 *  enters that shard. The walk holds only under the epoch the cursor opened in: once a destructive
 *  window moved it, every pull answers @c contended_k.
 */
template <typename indexes_type_>
struct vram_nvidia_indexes_cursor {
    using indexes_t = indexes_type_;
    using index_t = typename indexes_t::index_t;
    using key_t = typename indexes_t::key_t;
    using node_id_t = typename indexes_t::node_id_t;
    using walk_t = breadth_first_walk<node_id_t>;
    using shard_node_t = typename walk_t::shard_node_t;

    /** Where a pull gathers its entries' links, in the caller's arena: the linked keys end to end,
     *  and the offset where each entry's run starts. */
    using links_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using links_keys_t = growing_buffer<key_t, links_allocator_t>;
    using links_offsets_t = growing_buffer<std::size_t, links_allocator_t>;

    /** One shard as the walk reads it: its watermark as the walk last took it, and the host copy of
     *  the neighbor rows of one block of nodes, as many as one word of visited bits spans. */
    struct shard_cursor_t {
        explicit shard_cursor_t(forkunion_allocator_t allocator) noexcept : staged_rows_(allocator) {}

        /**
         *  @brief Takes the watermark of @p shard and sizes the staging for one block of its rows.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k The staging found no memory.
         */
        status_t open(index_t const& shard) noexcept {
            refresh(shard);
            return staged_rows_.resize_uninitialized(walk_t::lanes_k * shard.connectivity());
        }

        /** Nodes visible to this cursor; IDs strictly below are walked. */
        node_id_t visible_count() const noexcept { return visible_count_; }

        /** Widens the snapshot to the nodes @p shard published since. */
        void refresh(index_t const& shard) noexcept { visible_count_ = static_cast<node_id_t>(shard.size()); }

        /** The row of the node the last seek stood on, raw as stored. */
        std::span<node_id_t const> neighbors() const noexcept { return sought_row_; }

        /**
         *  @brief Stands on @p node of @p shard, copying the block of rows it falls in unless the
         *      staged block holds it.
         *
         *  @par Statuses:
         *  - @c malformed_k The @p node is at or past the snapshot.
         *  - @c config_mismatch_k The shard holds fewer rows than the snapshot, as a clear left it.
         *  - @c device_k The copy failed.
         */
        status_t seek(index_t const& shard, node_id_t node) noexcept {
            if (node >= visible_count_) return status_code_t::malformed_k;
            std::size_t const row_ids = shard.connectivity();
            if (node < staged_first_ || node >= staged_first_ + staged_count_) {
                node_id_t const first = static_cast<node_id_t>(node - node % walk_t::lanes_k);
                node_id_t const count = min_of<node_id_t>(walk_t::lanes_k, visible_count_ - first);
                staged_count_ = 0;
                std::span<node_id_t> const block = to_writable_span(staged_rows_).first(count * row_ids);
                if (status_t const staged = shard.neighbors(first, block); failed(staged)) return staged;
                staged_first_ = first;
                staged_count_ = count;
            }
            sought_row_ = to_span(staged_rows_).subspan((node - staged_first_) * row_ids, row_ids);
            return status_code_t::success_k;
        }

      private:

        /** The neighbor rows of the staged block, @c connectivity IDs per node, raw as stored. */
        growing_buffer<node_id_t, forkunion_allocator_t> staged_rows_;

        /** The first node of the staged block. */
        node_id_t staged_first_ = 0;

        /** Nodes in the staged block: none before the first seek, nor after a failed copy. */
        node_id_t staged_count_ = 0;

        /** The row of the node the last seek stood on, inside @c staged_rows_. */
        std::span<node_id_t const> sought_row_ {};

        /** The shard's watermark as of open or the last refresh: IDs below it are walked. */
        node_id_t visible_count_ = 0;
    };

    explicit vram_nvidia_indexes_cursor(forkunion_allocator_t allocator) noexcept
        : shard_cursors_(allocator), walk_(allocator), allocator_(allocator) {}

    /**
     *  @brief Opens the walk over every live shard of @p indexes under the epoch it reads now; any
     *      @p starts make it breadth-first, each key starting from its lowest node in the lowest
     *      shard holding it, and a key held nowhere starting nothing.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k A cursor, its staging, the visited bits or the frontier got no memory.
     */
    status_t open(indexes_t const& indexes, std::span<key_t const> starts) noexcept {
        epoch_ = indexes.read_epoch();
        first_shard_ = indexes.shards_.retired_shards();
        std::size_t const count = indexes.shards_count() - first_shard_;
        if (failed(shard_cursors_.reserve_exactly(count))) return status_code_t::out_of_memory_k;
        for (std::size_t shard = 0; shard != count; ++shard) {
            shard_cursors_.push_back_reserved(shard_cursor_t {allocator_});
            if (status_t const opened = shard_cursors_.back().open(shard_(indexes, shard)); failed(opened))
                return opened;
        }
        if (starts.empty()) return status_code_t::success_k;
        order_ = cursor_order_t::breadth_first_k;
        growing_buffer<std::size_t, forkunion_allocator_t> visible {allocator_};
        if (failed(visible.resize_uninitialized(count))) return status_code_t::out_of_memory_k;
        for (std::size_t shard = 0; shard != count; ++shard) visible[shard] = shard_cursors_[shard].visible_count();
        if (status_t const opened = walk_.open(to_span(visible)); failed(opened)) return opened;
        for (key_t const& key : starts)
            for (std::size_t shard = 0; shard != count; ++shard) {
                node_id_t const visible_count = shard_cursors_[shard].visible_count();
                node_id_t lowest = visible_count;
                shard_(indexes, shard).find_each(key, [&](node_id_t node) noexcept { lowest = min_of(lowest, node); });
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
     *  - @c device_k A shard's rows did not copy; written entries stand.
     *  - @c contended_k A destructive window of @p indexes opened since the walk did; none stand.
     */
    expected<std::size_t> pull(indexes_t const& indexes, std::span<key_t> keys, links_keys_t* links_keys,
                               links_offsets_t* links_offsets) noexcept {
        std::size_t written = 0;
        shard_node_t entry;
        while (written != keys.size() && next_(indexes, entry)) {
            shard_cursor_t& cursor = shard_cursors_[entry.shard];
            index_t const& shard = shard_(indexes, entry.shard);
            if (status_t const sought = cursor.seek(shard, entry.node); failed(sought))
                return indexes.still_valid(epoch_)
                           ? expected<std::size_t> {written, sought}
                           : expected<std::size_t> {0,
                                                    {status_code_t::contended_k, "A window moved the cursor's shard"}};
            std::span<node_id_t const> const neighbors = cursor.neighbors();
            if (links_keys && (failed(links_keys->reserve(links_keys->size() + neighbors.size())) ||
                               failed(links_offsets->reserve(links_offsets->size() + 1))))
                return {written, status_code_t::out_of_memory_k};
            for (node_id_t const neighbor : neighbors) {
                // Rows come raw: a self-loop links nowhere, and sentinels fail both bounds below.
                if (neighbor == entry.node) continue;
                // Breadth-first never refreshes, so the bits span the shard cursors' snapshots.
                if (order_ == cursor_order_t::breadth_first_k && neighbor < cursor.visible_count())
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
        // Retired shards keep their storage, so a pull that ran past a window read nothing freed.
        if (!indexes.still_valid(epoch_)) return {0, {status_code_t::contended_k, "A window passed the cursor's pull"}};
        return {written, status_code_t::success_k};
    }

  private:

    /** The live shard at @p live as the walk opened, counted past the shards retired then. */
    index_t const& shard_(indexes_t const& indexes, std::size_t live) const noexcept {
        return *indexes.shards_[first_shard_ + live].index;
    }

    /** Takes the next entry into @p entry, answering false once the walk ran out. */
    bool next_(indexes_t const& indexes, shard_node_t& entry) noexcept {
        if (order_ == cursor_order_t::breadth_first_k) return walk_.expand(entry);
        while (next_stored_.shard != shard_cursors_.size()) {
            if (next_stored_.node < shard_cursors_[next_stored_.shard].visible_count()) {
                entry = next_stored_;
                ++next_stored_.node;
                return true;
            }
            next_stored_ = {next_stored_.shard + 1, 0};
            if (next_stored_.shard != shard_cursors_.size())
                shard_cursors_[next_stored_.shard].refresh(shard_(indexes, next_stored_.shard));
        }
        return false;
    }

    /** One per live shard at open, in placement order, each over the snapshot it took then. */
    growing_buffer<shard_cursor_t, forkunion_allocator_t> shard_cursors_;

    /** The nodes visited and the frontier of a breadth-first walk; untouched by a stored one. */
    walk_t walk_;

    forkunion_allocator_t allocator_;
    cursor_order_t order_ = cursor_order_t::stored_k;

    /** The position of the first shard walked: the first live one at open, which a clear since
     *  retires in place. */
    std::size_t first_shard_ = 0;

    /** As stored: the entry the next pull starts at. */
    shard_node_t next_stored_ {};

    /** The epoch the walk opened under; a pull after a window since answers contended. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::basic
