/**
 *  @file include/usearch/basic/vram_nvidia_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Many basic GPU shards on NVIDIA devices viewed as one, so a modality's graph grows past
 *      one device and one shard's ID space.
 *
 *  The plural over a modality's face of @c basic::vram_ragged_graph: shards share one bound, and a
 *  batch crossing it opens another shard generation on the same devices. GPU graphs store no keys,
 *  so each shard keeps a host key column beside it and the shard set's locator map finds every
 *  entry of a key. A search ranks every query on every shard below one snapshot of their watermarks
 *  and merges the shards' matches through their key columns; a batch launches shard by shard and
 *  publishes on every shard inside one commit, so a search sees it whole or not at all.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <functional>  // `std::hash` - the locator map's key hash
#include <limits>      // `std::numeric_limits` - the node-ID type's sentinel
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the caller's keys & counts
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move`, `std::exchange`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the shard set's words
#include <numkong/tensor.hpp>    // `nk::matrix_span` - the merged rankings

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"        // `growing_buffer` - the plans' parts & the landings
#include "usearch/utilities/ram_device.hpp"         // `forkunion_allocator_t` - the host columns' placement
#include "usearch/utilities/vram_nvidia_device.hpp" // `cuda_executor_t`, `cuda_vendor_t`
#include "usearch/utilities/ram_shards.hpp"         // `shard_set` - the shard table
#include "usearch/utilities/vram_shard_fanout.hpp"  // `dense::vram_shard_fanout` - the contexts & pairing
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/basic/shard_checkpoints.hpp" // `save_shards`, `load_shards` - the checkpoints

namespace unum::usearch::basic {

/**
 *  @brief Basic GPU shards viewed as one: batches routed over the shard generations, searches
 *      merged across them, lookups through the shared locator map.
 *
 *  @tparam index_type_ The shard, a modality's NVIDIA face over @c basic::vram_ragged_graph, with
 *      32-bit node IDs, as the shard table locates every row in 8 bytes.
 *  @tparam key_type_ The caller-visible identity, opaque to every shard.
 *  @tparam build_type_ Builds one unreserved shard, configured as every other, on a device:
 *      `build(device) → std::unique_ptr<index_type_>`, null when the allocation fails.
 *  @tparam wait_type_ What a reader spanning the shards waits through while a commit is in flight.
 *  @tparam key_hash_type_ The locator map's hash of a key.
 */
template <                                         //
    typename index_type_,                          //
    typename key_type_,                            //
    typename build_type_,                          //
    typename wait_type_ = dummy_waiter_t,          //
    typename key_hash_type_ = std::hash<key_type_> //
    >
    requires std::same_as<typename index_type_::node_id_t, utilities::node_id_t>
struct vram_nvidia_indexes {
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

    /** Every shard's keys, indexed by its node IDs on the host, where the merges resolve them. */
    using keys_column_t = shard_keys_column<index_t, key_t, fu::standard_atomic_ref>;

    /** The shards, their key columns, the locator map and the rotation that routes new rows. */
    using shards_t = shard_set<index_t, keys_column_t, key_t, key_hash_type_, fu::standard_atomic_ref, wait_type_>;
    using shard_t = typename shards_t::shard_t;
    using key_locator_t = typename shards_t::key_locator_t;

    /** The streams and their scratch every batch verb runs on, paired with the shards by device. */
    using executor_t = cuda_executor_t;

  private:
    using fanout_t = dense::vram_shard_fanout<shards_t, cuda_vendor_t>;
    using traversals_t = typename fanout_t::traversals_t;
    using shard_insert_plan_t = typename index_t::insert_plan_t;

  public:
#pragma region Placement and Measures

    /** Every index names the host allocator its shard table, key columns and locator map sit on,
     *  and the builder of its shards; nothing is placed by default. */
    vram_nvidia_indexes(forkunion_allocator_t allocator, build_type_ build) noexcept
        : shards_(allocator), build_(std::move(build)) {}
    vram_nvidia_indexes(vram_nvidia_indexes const&) = delete;
    vram_nvidia_indexes& operator=(vram_nvidia_indexes const&) = delete;

    /** Bounds every shard at @p rows, clamped to the ID space, zero meaning the ID space; past it,
     *  a batch opens another shard generation on the same devices. Ignored once storage exists. */
    void bound_shards(std::size_t rows) noexcept { shards_.bound_shards(rows); }

    /** Appends one shard built on CUDA device @p device; refused once storage exists. */
    status_t emplace(int device) noexcept {
        std::unique_ptr<index_t> shard = build_(device);
        if (!shard) return status_t::bad_alloc_k;
        return shards_.emplace(std::move(shard), keys_column_t {shards_.allocator()});
    }

    /** Grows every shard ahead of @p config.capacity rows and @p config.bytes value bytes, both
     *  split evenly over the shards, since each shard's tape holds only its own rows' values; the
     *  first call freezes the connectivity. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        config.bytes = divide_round_up(config.bytes, max_of<std::size_t>(shards_.size(), 1));
        return shards_.reserve(config);
    }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed. */
    std::size_t size() const noexcept { return shards_.rows(); }

    /** Shards in the index, every shard generation's, in placement order. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** Shards past the retired ones, which every walk and checkpoint spans. */
    std::size_t live_shards_count() const noexcept { return shards_.size() - shards_.retired_shards(); }

    /** Rows the shards host without growing again. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** Bytes every shard, its key column and the locator map occupy, on the devices and host. */
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
     */
    status_t clear() noexcept {
        if (status_t const closed = shards_.close(); failed(closed)) return closed;
        if (shards_.rows()) {
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

    /** Invokes @p callback as `callback(device)` for the device of every shard, in placement order,
     *  so a caller builds an executor driving them all. */
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
     *  @brief One batch across the shards: every shard's claim over the rows it takes, its key
     *      column covered to them, and the locator seats, so the one @ref insert only lands.
     *
     *  Each shard of a shard generation takes as many rows as splitting the batch across the shards
     *  round-robin from the starting shard would give it, as one contiguous run of the batch, since
     *  a shard's rows cross as one slice of the caller's tape. A plan that dies before @ref insert
     *  gives every claim back.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : shard_plans_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : shard_plans_(std::move(other.shard_plans_)), streams_(std::move(other.streams_)),
              executor_(other.executor_), seats_(other.seats_), config_(other.config_), first_(other.first_),
              rows_(other.rows_), owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /**
         *  @brief Lands every row of @p tape under its key in @p keys, shard by shard, and spends
         *      the plan whatever it answers.
         *
         *  Each shard's run writes its keys and launches and drains on its device as the shard's
         *  own plan does, leaving its rows unpublished; every drained run then publishes inside one
         *  commit, so a search snapshotting the shards sees the batch on all of them or none, and
         *  their keys enter the locator map last.
         *
         *  A refusal stops at its shard: the runs before it still publish together with their keys,
         *  and the claims after it are given back.
         */
        expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape) noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (keys.size() != rows_ || tape.size() != rows_) return {{}, status_t::config_mismatch_k};
            vram_nvidia_indexes& owner = *std::exchange(owner_, nullptr);
            shards_t& shards = owner.shards_;
            insert_result_t result;
            if (!rows_) return {result, status_t::success_k};
            traversals_t traversals {shards.allocator()};
            if (status_t const readied = fanout_t::reserve_traversals(shards, traversals, *executor_, tape, config_);
                failed(readied))
                return {result, readied};
            status_t status = status_t::success_k;
            std::size_t first_row = 0, drained = 0, launched = 0;
            for (; drained != shard_plans_.size(); ++drained) {
                shard_insert_plan_t& shard_plan = shard_plans_[drained];
                std::size_t const placed = first_ + drained, rows = shard_plan.rows;
                if (!rows) continue;
                shard_t& one = shards[placed];
                // Keys land before their rows publish, so a search resolving a row reads one.
                for (std::size_t row = 0; row != rows; ++row)
                    one.keys[static_cast<node_id_t>(shard_plan.first_id + row)] = keys[first_row + row];
                if (status = fanout_t::enter(shards, placed); failed(status)) break;
                expected<insert_result_t> const inserted = one.index->graph_.launch_(
                    shard_plan, tape.subview(first_row, rows), traversals[placed], config_);
                if (status = inserted.status; failed(status)) break;
                result += inserted.value;
                first_row += rows;
                ++launched;
            }

            // Each drained shard's claim admits this batch alone, so no publish waits in commit.
            if (launched > 1) shards.begin_commit();
            for (std::size_t plan = 0; plan != drained; ++plan)
                if (shard_insert_plan_t const& shard_plan = shard_plans_[plan]; shard_plan.rows)
                    shards[first_ + plan].index->graph_.publish_(shard_plan.first_id + shard_plan.rows);
            if (launched > 1) shards.end_commit();
            for (std::size_t plan = 0; plan != drained; ++plan) {
                shard_insert_plan_t const& shard_plan = shard_plans_[plan];
                std::size_t const placed = first_ + plan;
                for (std::size_t row = 0; row != shard_plan.rows; ++row) {
                    node_id_t const node = static_cast<node_id_t>(shard_plan.first_id + row);
                    shards.locators().insert(seats_, shards[placed].keys[node],
                                             key_locator_t {static_cast<u32_t>(placed), node});
                }
            }
            result.new_size = owner.size();
            return {result, status};
        }

      private:
        friend vram_nvidia_indexes;

        /** One claim per shard from the open shard generation's first, in placement order; an empty
         *  one where no row lands. */
        growing_buffer<shard_insert_plan_t, forkunion_allocator_t> shard_plans_;

        /** The executor's hold, kept for the plan's whole life. */
        typename executor_t::claim_t streams_;

        /** The executor the runs launch on. */
        executor_t* executor_ = nullptr;

        /** One locator seat per row. */
        typename shards_t::locators_map_t::seats_t seats_ {};

        /** The per-shard insertion configuration, applied uniformly. */
        insert_config_t config_ {};

        /** The open shard generation's first shard at planning, which the claims start at. */
        std::size_t first_ = 0;

        /** Rows in the batch. */
        std::size_t rows_ = 0;

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once spent. */
        vram_nvidia_indexes* owner_ = nullptr;
    };

    /**
     *  @brief Opens a batch of @p rows: takes the executor, routes the rows over the open shard
     *      generation and as many fresh ones as the bound makes them fill, claims every shard's run
     *      and covers its key column, and claims the locator seats.
     *
     *  A shard with a batch, load, refine or save in flight refuses its claim, and the whole plan
     *  with @c contended_k, giving back the claims before it.
     *
     *  @param[in] rows Rows in the batch.
     *  @param[inout] executor The streams and their scratch, held until the plan dies.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        insert_plan_t plan {shards_.allocator()};
        plan.rows_ = rows;
        plan.config_ = config;
        plan.executor_ = &executor;
        if (!rows) {
            plan.owner_ = this;
            return {std::move(plan), status_t::success_k};
        }
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            rows, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.first_ = batch_split.open_shard_generation.first;

        std::size_t const shards = batch_split.shards_spanned();
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const rows_for_shard = batch_split.rows_for_shard(shard);
            if (!rows_for_shard) {
                plan.shard_plans_.push_back_reserved(shard_insert_plan_t {});
                continue;
            }
            shard_t& one = shards_[batch_split.open_shard_generation.first + shard];
            auto [shard_plan, status] = one.index->plan_insert(rows_for_shard);
            if (failed(status)) return {{}, status};
            if (status_t const covered = one.keys.cover(shard_plan.first_id + rows_for_shard); failed(covered))
                return {{}, covered};
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
        }
        auto [seats, seat_status] = shards_.locators().claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        plan.owner_ = this;
        return {std::move(plan), status_t::success_k};
    }

    /** Inserts every row of @p tape under its key in @p keys: @ref plan_insert, then its one
     *  @c insert. */
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (keys.size() != tape.size()) return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(tape.size(), executor, config);
        if (failed(status)) return {{}, status};
        return plan.insert(keys, tape);
    }

#pragma endregion Insertion

#pragma region Checkpoints

    /** What a checkpoint does on these shards' devices, for @ref save_shards and @ref load_shards:
     *  every shard's plans on its own stream of @c executor, its keys through the host key column
     *  and the locator map, and a load's reservation at the connectivity the checkpoint froze. */
    struct checkpoint_residency_t {
        using seats_t = typename shards_t::locators_map_t::seats_t;

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
        expected<std::size_t> reserve(shards_t& shards, std::size_t shard, std::size_t rows) noexcept {
            reserve_capacity_t config = reservation;
            config.capacity = rows;
            config.ceiling = shards.shard_capacity();
            return shards[shard].index->reserve(config);
        }

        /** Copies shard @p shard 's host key column into @p keys, one per row a save writes. */
        static void save_keys(shards_t& shards, std::size_t shard, std::span<key_t> keys) noexcept {
            std::size_t landed = 0;
            host_keys(shards[shard].keys).for_each_span({0, keys.size()}, [&](std::span<key_t const> run) noexcept {
                for (key_t const& key : run) keys[landed++] = key;
            });
        }

        /** Claims the locator seats of @p rows keys a load lands. */
        static expected<seats_t> claim_seats(shards_t& shards, std::size_t rows) noexcept {
            if (status_t const indexed = shards.locators().reserve(shards.rows() + rows); failed(indexed))
                return {{}, indexed};
            return shards.locators().claim(rows);
        }

        /** Lands @p keys from row @p first_row of shard @p shard into its host key column and the
         *  locator map, on the seats @ref claim_seats claimed. */
        static status_t land_keys(shards_t& shards, std::size_t shard, std::size_t first_row,
                                  std::span<key_t const> keys, seats_t seats) noexcept {
            auto& column = host_keys(shards[shard].keys);
            if (status_t const covered = column.cover(first_row + keys.size()); failed(covered)) return covered;
            for (std::size_t row = 0; row != keys.size(); ++row) {
                column[static_cast<node_id_t>(first_row + row)] = keys[row];
                shards.locators().insert(
                    seats, keys[row],
                    key_locator_t {static_cast<u32_t>(shard), static_cast<node_id_t>(first_row + row)});
            }
            return status_t::success_k;
        }

        /** Leaves part @p part the live shard it names, a placed one, or opens shard generations on
         *  the placed shards' devices until it exists, each placed shard first frozen at the
         *  checkpoint's connectivity, which the shard generations take after it. */
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

    /** Writes the live shards under @p path through @ref save_shards: a lone live shard's monolith,
     *  or one per live shard beside a manifest; each shard's rows cross on its own stream of
     *  @p executor, in order. */
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

    /** Reads the checkpoint at @p path into the placed, empty shards through @ref load_shards, one
     *  part per shard, opened past the placed ones where the parts outnumber them, reserving each
     *  at @p connectivity, which the checkpoint froze. */
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
     *      the caller's rankings through its key column, ascending.
     *
     *  @param[in] queries The queries, one tape entry of bytes each.
     *  @param[inout] executor The streams and their scratch, claimed for the call.
     *  @param[in] config The walks' breadth; the matches per query are as many as @p keys is wide.
     *  @param[out] keys One row of merged keys per query.
     *  @param[out] distances The same rankings' distances.
     *  @param[out] counts The matches each query's row holds.
     */
    expected<search_result_t> search(tape_view<byte_t const> queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept {
        std::size_t const queries_count = queries.size();
        if (status_t const refused = fanout_t::refusal(shards_, executor, queries_count, keys, distances, counts);
            failed(refused))
            return {{}, refused};
        for (std::size_t& count : counts) count = 0;
        if (!queries_count || !keys.extent(1)) return {{}, status_t::success_k};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        config.count = static_cast<small_size_t>(keys.extent(1));

        // One snapshot bounds every shard's walk, sized again for a shard generation opened
        // meanwhile, so the merged rankings see each batch on every shard or on none.
        traversals_t traversals {shards_.allocator()};
        growing_buffer<node_id_t, forkunion_allocator_t> watermarks {shards_.allocator()};
        while (true) {
            traversals.clear();
            if (status_t const readied = fanout_t::reserve_traversals(shards_, traversals, executor, queries, config);
                failed(readied))
                return {{}, readied};
            if (failed(watermarks.resize(traversals.size()))) return {{}, status_t::bad_alloc_k};
            if (shards_.snapshot(to_writable_span(watermarks)).status != status_t::shape_mismatch_k) break;
        }
        std::size_t const landing_entries = queries_count * config.count;
        growing_buffer<node_id_t, forkunion_allocator_t> landed_ids {shards_.allocator()};
        growing_buffer<distance_t, forkunion_allocator_t> landed_distances {shards_.allocator()};
        growing_buffer<scored_node_t, forkunion_allocator_t> frame {shards_.allocator()};
        if (failed(landed_ids.reserve_exactly(landing_entries)) ||
            failed(landed_distances.reserve_exactly(landing_entries)) || failed(frame.reserve_exactly(config.count)))
            return {{}, status_t::bad_alloc_k};
        landed_ids.extend_uninitialized(landing_entries);
        landed_distances.extend_uninitialized(landing_entries);
        frame.extend_uninitialized(config.count);
        nk::matrix_span<node_id_t> const shard_ids {landed_ids.data(), queries_count, config.count};
        nk::matrix_span<distance_t> const shard_distances {landed_distances.data(), queries_count, config.count};

        // The snapshot counted the shards once, so a shard generation opened since takes no part.
        search_result_t merged;
        for (std::size_t shard = 0; shard != traversals.size(); ++shard) {
            if (!watermarks[shard]) continue;
            if (status_t const entered = fanout_t::enter(shards_, shard); failed(entered)) return {merged, entered};
            expected<search_result_t> const found = shards_[shard].index->graph_.search_below_(
                queries, shard_ids, shard_distances, traversals[shard], config, watermarks[shard]);
            if (failed(found.status)) return {merged, found.status};
            merged += found.value;
            for (std::size_t query = 0; query != queries_count; ++query) {
                node_id_t const* const ids = landed_ids.data() + query * config.count;
                distance_t const* const scores = landed_distances.data() + query * config.count;
                // An absent match ends the row, and must never index a key column.
                std::size_t live = 0;
                for (; live != config.count && ids[live] != std::numeric_limits<node_id_t>::max(); ++live)
                    frame[live] = scored_node_t {ids[live], scores[live]};
                counts[query] = merge_search_results(std::span<scored_node_t const> {frame.data(), live},
                                                     shards_[shard].keys, to_span(keys.row(query).as_vector()),
                                                     to_span(distances.row(query).as_vector()), counts[query]);
            }
        }
        // Each shard counted its own matches, which the merged rows replace.
        merged.count = 0;
        for (std::size_t const count : counts) merged.count += count;
        return {merged, status_t::success_k};
    }

    /** Entries under @p key across every shard. */
    std::size_t count(key_t const& key) const noexcept { return shards_.locators().count(key); }

    /** Invokes @p callback as `callback(shard, node_id)` for every published entry under @p key. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        shards_.locators().find_each(
            key, [&](key_locator_t locator) noexcept { callback(*shards_[locator.shard].index, locator.node); });
    }

    /** One shard's matches as a judge lands them on the host: @c wanted node IDs and distances per
     *  query, row by row, and how many of each row hold matches. */
    struct landing_span_t {
        std::span<node_id_t> ids;
        std::span<f64_t> distances;
        std::span<std::size_t> counts;
        std::size_t wanted = 0;
    };

    /**
     *  @brief Ranks the queries of a modality's own search on every live shard in turn through
     *      @p judge, and merges the shards' rankings by distance into @p keys and @p distances, row
     *      by row, @p staging.wanted wide.
     *
     *  @p judge is called as `judge(shard, staging) → expected<traversal_telemetry_t>` per live
     *  shard holding rows, landing that shard's matches in @p staging, the caller's host buffers,
     *  which this call merges before the next shard reuses them; a refusal stops at its shard.
     *
     *  @param[in] staging One shard's landing, as wide as the rows this call fills.
     *  @param[out] keys The merged rankings' keys, @c staging.wanted per query.
     *  @param[out] distances The same rankings' distances, ascending per query.
     *  @param[out] counts The matches each query's row holds.
     */
    template <typename judge_type_>
    expected<traversal_telemetry_t> search_shards(landing_span_t staging, std::span<key_t> keys,
                                                  std::span<f64_t> distances, std::span<std::size_t> counts,
                                                  judge_type_&& judge) noexcept {
        using entry_t = scored_node<node_id_t, f64_t>;
        std::size_t const wanted = staging.wanted;
        for (std::size_t& count : counts) count = 0;
        traversal_telemetry_t total;
        std::size_t const first = shards_.retired_shards(), shards = shards_.size();
        for (std::size_t position = first; position != shards; ++position) {
            shard_t& one = shards_[position];
            if (!one.index->size()) continue;
            auto const [walked, status] = judge(*one.index, staging);
            if (failed(status)) return {total, status};
            total += walked;
            for (std::size_t row = 0; row != counts.size(); ++row)
                for (std::size_t match = 0; match != staging.counts[row]; ++match) {
                    entry_t const entry {staging.ids[row * wanted + match], staging.distances[row * wanted + match]};
                    counts[row] = merge_search_results(std::span<entry_t const> {&entry, 1}, one.keys,
                                                       keys.subspan(row * wanted, wanted),
                                                       distances.subspan(row * wanted, wanted), counts[row]);
                }
        }
        return {total, status_t::success_k};
    }

#pragma endregion Search and Lookups

  private:

    /** One shard of a new shard generation on the device @p like sits on, reserved under the bound
     *  with its connectivity, and its key column; a null index when the build or reservation
     *  fails. */
    shard_t build_shard_(shard_t const& like) const noexcept {
        std::unique_ptr<index_t> index = build_(like.index->device_specs().ordinal);
        reserve_capacity_t reserved {};
        reserved.connectivity = static_cast<connectivity_t>(like.index->connectivity());
        reserved.ceiling = shards_.shard_capacity();
        if (index && failed(index->reserve(reserved).status)) index.reset();
        return shard_t {std::move(index), keys_column_t {shards_.allocator()}};
    }

    /** The shards, their key columns, the locator map, their routing and the commit word. */
    shards_t shards_;

    /** Builds every shard this index places, alike. */
    build_type_ build_;
};

} // namespace unum::usearch::basic
