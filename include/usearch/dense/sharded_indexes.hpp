/**
 *  @file include/usearch/dense/sharded_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The base every dense plural shares: same-geometry shards viewed as one, the keys of their
 *      rows, rotation-balanced routing, and SafeGraphs checkpoints.
 *
 *  @c ram_indexes, @c vram_nvidia_indexes, @c vram_amd_indexes and @c vram_apple_indexes derive
 *  from it, CRTP style, and keep only what their residency does differently, through the hooks the
 *  protected section lists. The device verbs the three GPU plurals share live here as well; the
 *  host plural declares its own of every one of them, which hides these, so none is instantiated
 *  for it.
 *
 *  Checkpoints ride the SafeGraphs interchange - one consolidated monolith, or one per shard beside
 *  a manifest - written positionally with every header last, so a failed save reads as absent.
 */
#pragma once

#include <cstring> // `std::memcmp` - the monolith's magic

#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the caller's keys & the merged ranking
#include <string_view> // `std::string_view` - the checkpoint paths
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::move`, `std::exchange`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `chunked_column` - the key columns
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the host columns' placement
#include "usearch/utilities/ram_tables.hpp"        // `chunked_multimap` - the locator map
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths & manifests - the checkpoints
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

/**
 *  @brief Same-geometry shards viewed as one: the CRTP base of every dense plural.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are expressible.
 *  Shards arrive one at a time through @ref emplace, so the index grows as its placements are
 *  engaged rather than being declared whole. The shard's configs, views and results are
 *  re-exported verbatim, so a batched call and a single one read the same.
 *
 *  @tparam derived_ The plural itself, which answers the hooks.
 *  @tparam index_type_ The shard type.
 *  @tparam executor_type_ The threads or streams every bulk verb runs on; it names no index, so one
 *      executor serves every instantiation.
 *  @tparam key_type_ The caller-visible identity, opaque to every shard.
 *  @tparam key_hash_type_ The locator map's hash of a key.
 *  @tparam atomic_ref_template_ The atomic reference owning the plain synchronization words.
 */
template <                                                               //
    typename derived_,                                                   //
    typename index_type_,                                                //
    typename executor_type_,                                             //
    typename key_type_,                                                  //
    typename key_hash_type_,                                             //
    template <typename> class atomic_ref_template_ = standard_atomic_ref //
    >
struct sharded_indexes {

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;

    /** Where one entry lives: the shard, and its shard-local ID - the index's only address for an
     *  entry. */
    struct key_locator_t {

        /** The shard holding the entry, positional in placement order. */
        std::size_t shard = 0;

        /** The entry's ID inside that shard's own space. */
        node_id_t node = 0;
    };

    /** The atomic reference owning the plain synchronization words - the shards' and this index's
     *  own routing tally alike. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Key to every entry stored under it; insert-only, wait-free for readers, growing in
     *  chunks. */
    using locators_map_t =
        chunked_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t, atomic_ref_template_>;

    /** The shard type - geometry parameters in the shard's own order, so the two never disagree. */
    using index_t = index_type_;

    /** Keys of one shard's rows, indexed by that shard's own node IDs; host-resident, growing in
     *  the chunks the CPU shards grow in. */
    using keys_column_t = chunked_column<key_t, chunk_schedule<10, 20>, index_t::max_node_count_k,
                                         forkunion_allocator_t, atomic_ref_template_>;

    using executor_t = executor_type_;
    using traversal_context_t = typename index_t::traversal_context_t;
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using refine_result_t = typename index_t::refine_result_t;
    using health_result_t = typename index_t::health_result_t;

    /** One shard: the graph, and the keys of its rows. Placement is not repeated: the shard
     *  carries its own. */
    struct shard_t {

        /** The graph, growing on demand; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;

        /** The key of each of this shard's rows, indexed by node ID; every claimer covers the rows
         *  it may write before writing them. */
        keys_column_t keys;
    };

#pragma region Placement

    sharded_indexes(sharded_indexes const&) = delete;
    sharded_indexes& operator=(sharded_indexes const&) = delete;

    /** Exchanges two indexes whole; a move implies the caller holds both exclusively. */
    void swap(derived_& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::swap(kernel_, other.kernel_);
        std::size_t const routed = atomic_ref<std::size_t>(routed_rows_).load(std::memory_order_relaxed);
        atomic_ref<std::size_t>(routed_rows_)
            .store(atomic_ref<std::size_t>(other.routed_rows_).load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
        atomic_ref<std::size_t>(other.routed_rows_).store(routed, std::memory_order_relaxed);
    }

    /** Appends one shard the caller already built, consuming it; refuses any walking by another
     *  kernel, any whose frozen geometry differs from the shards already here, and any already
     *  reserved. Several shards may share a placement, each a slice of the graph rather than a
     *  copy. @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard || shard->kernel() != kernel_) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (shard->capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        return derived().place_(std::move(shard));
    }

    /** Empties every shard in place, keeping its placement, geometry and capacity; a kept failure
     *  drops with the rows. Demands the same quiescence as @ref reserve. */
    status_t clear() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = derived().rebuild_(*one.index);
            if (!fresh) return status_t::bad_alloc_k;
            one.index = std::move(fresh);
        }
        locators_of_key_.clear();
        atomic_ref<std::size_t>(routed_rows_).store(0, std::memory_order_relaxed);
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

#pragma endregion Placement

#pragma region Measures and Reservation

    /** The metric every shard scores by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Shards in the index, in placement order. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }

    /** The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return derived().snapshot_({}); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->capacity();
        return total;
    }

    /** The first shard's kept failure, or success; asked of the shards, never cached. */
    status_t status() const noexcept
        requires requires(index_t const& one) { one.status(); }
    {
        for (shard_t const& one : shards_)
            if (status_t const kept = one.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }

    /** Bytes the index occupies - every shard's storage, its key column, and the shared locator
     *  map; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(derived_) + locators_of_key_.memory_usage();
        for (shard_t const& one : shards_) bytes += one.index->memory_usage() + one.keys.memory_usage();
        return bytes;
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
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            auto const [shard_capacity, status] = shards_[shard].index->reserve(per_shard);
            if (failed(status)) return {capacity(), status};
            if (status_t const covered = derived().cover_keys_(shard, shard_capacity); failed(covered))
                return {capacity(), covered};
        }
        if (status_t const indexed = locators_of_key_.reserve(config.capacity); failed(indexed))
            return {capacity(), indexed};
        return {capacity(), status_t::success_k};
    }

#pragma endregion Measures and Reservation

#pragma region Insertion

    /**
     *  @brief One batch in flight across the device shards.
     *
     *  The executor and every shard's claim over the rows it takes are held for the plan's whole
     *  life, one context per shard opened for the whole batch, every shard's storage and key
     *  columns grown to the rows it takes, and the locator seats claimed, so a step fails only on
     *  its devices.
     *
     *  Row @c t of the batch lands on shard @c (rotation+t) mod S. Steps run one after another on
     *  one host thread, each writing its rows' keys and launching the rows as deferred inserts;
     *  @ref finish settles them and turns every shard's watermark inside one commit, so a reader
     *  spanning the shards sees the batch whole or not at all. A failed launch or drain publishes
     *  nothing, and neither does a plan dying unfinished: every shard holding launched rows keeps
     *  the failure, so its rows stay invisible and later mutations are refused.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : traversals_(allocator), shares_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : traversals_(std::move(other.traversals_)), shares_(std::move(other.shares_)), seats_(other.seats_),
              config_(other.config_), rows_(other.rows_), rotation_(other.rotation_), result_(other.result_),
              status_(other.status_), streams_(std::move(other.streams_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /** An unfinished plan drains what it launched, keeps @c cancelled_k on every shard holding
         *  it, and gives back every claim that launched nothing. */
        ~insert_plan_t() noexcept {
            if (owner_) [[maybe_unused]]
                status_t const abandoned = abandon_(status_t::cancelled_k);
        }

        /** Launches rows @b [first_row,first_row+rows) of the plan under @p keys on their shards;
         *  the rows and the keys stay live until @ref finish. Mismatched arguments launch nothing,
         *  any later refusal fails the plan. */
        status_t insert(std::size_t first_row, nk::vector_view<key_t> keys, rows_view_t rows) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            std::size_t const count = rows.rows;
            if (keys.size() != count || first_row > rows_ || count > rows_ - first_row ||
                (count && rows.columns != owner_->dimensions()))
                return status_t::config_mismatch_k;
            if (!count) return status_t::success_k;
            std::size_t const shards = shares_.size();
            // Each shard takes one slab per window at most, so no launch stages more.
            std::size_t const window_rows = derived_::window_rows_k * shards;
            for (std::size_t first = 0; first < count && succeeded(status_); first += window_rows) {
                std::size_t const taking = min_of(window_rows, count - first);
                std::size_t const rotation = (rotation_ + first_row + first) % shards;
                for (std::size_t shard = 0; shard != shards && succeeded(status_); ++shard)
                    status_ = launch_(shard, first_row_of_shard_(shard, shards, rotation),
                                      keys[nk::range(first, first + taking)], rows.slice(first, taking));
            }
            return status_;
        }

        /** Settles every launch, publishes every row at once, and reports the batch's telemetry. A
         *  plan short of rows refuses and stays whole, so the missing steps can still land; any
         *  other call spends it, and a failed launch or drain publishes nothing and keeps its
         *  failure on every shard holding launched rows. */
        expected<insert_result_t> finish() noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            std::size_t launched = 0;
            for (share_t const& share : shares_) launched += share.launched;
            if (succeeded(status_) && launched != rows_) return {{owner_->size()}, status_t::config_mismatch_k};
            insert_result_t result = result_;
            status_t status = owner_->settle_insert_(traversals_, status_, result);
            derived_& owner = *owner_;
            if (failed(status)) {
                result.new_size = owner.size();
                return {result, abandon_(status)};
            }

            // Every shard's claim admits this batch alone, so no turn waits inside the commit.
            standard_atomic_ref<u64_t>(owner.commit_sequence_).fetch_add(1, std::memory_order_release);
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (share_t const& share = shares_[shard]; share.launched)
                    if (status_t const published = owner.shards_[shard].index->publish_(share.first_id, share.launched);
                        failed(published) && succeeded(status))
                        status = published;
            standard_atomic_ref<u64_t>(owner.commit_sequence_)
                .fetch_add((u64_t {1} << 32) - 1, std::memory_order_release);

            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                for (std::size_t row = 0; row != shares_[shard].launched; ++row) {
                    node_id_t const node = static_cast<node_id_t>(shares_[shard].first_id + row);
                    owner.locators_of_key_.insert(seats_, owner.shards_[shard].keys[node], key_locator_t {shard, node});
                }
            owner_ = nullptr;
            result.new_size = owner.size();
            return {result, status};
        }

      private:

        /** One shard's part of the batch. */
        struct share_t {

            /** The ID the part starts at: the shard's claim at planning. */
            node_id_t first_id = 0;

            /** Rows of the batch the shard takes. */
            std::size_t rows = 0;

            /** Rows launched on the shard so far, contiguous from @c first_id. */
            std::size_t launched = 0;
        };

        /** Writes the keys of the step's rows shard @p shard takes, from @p first on in strides of
         *  the shard count, then launches those rows there. */
        status_t launch_(std::size_t shard, std::size_t first, nk::vector_view<key_t> keys, rows_view_t rows) noexcept {
            std::size_t const shards = shares_.size(), count = rows.rows;
            std::size_t const taking = rows_from_(count, first, shards);
            share_t& share = shares_[shard];
            if (!taking) return status_t::success_k;
            // Overlapping steps would write keys past the cover the plan grew.
            if (share.launched + taking > share.rows) return status_t::config_mismatch_k;
            shard_t& one = owner_->shards_[shard];
            traversal_context_t& traversal = traversals_[shard];
            if (status_t const entered = owner_->enter_(shard); failed(entered)) return entered;
            node_id_t const first_node = static_cast<node_id_t>(share.first_id + share.launched);
            for (std::size_t taken = 0; taken != taking; ++taken)
                one.keys[first_node + static_cast<node_id_t>(taken)] = keys[first + taken * shards];
            // On the launch's own stream, so the device keys land before the rows they name
            // publish.
            if (status_t const uploaded = owner_->upload_keys_(shard, {first_node, first_node + taking},
                                                               traversal.stream());
                failed(uploaded))
                return uploaded;
            expected<insert_result_t> const inserted = one.index->submit_insert_(
                {rows.row(first), taking, rows.columns, rows.stride * shards, rows.dtype}, first_node, {}, traversal,
                config_);
            if (!inserted) return inserted.status;
            share.launched += taking;
            result_.visited_members += inserted.value.visited_members;
            result_.computed_distances += inserted.value.computed_distances;
            result_.restarts += inserted.value.restarts;
            result_.unlinked_neighbors += inserted.value.unlinked_neighbors;
            result_.tail_drops += inserted.value.tail_drops;
            result_.visited_saturations += inserted.value.visited_saturations;
            return status_t::success_k;
        }

        /** Drains every context, keeps @p failure on every shard holding launched rows, and gives
         *  back every claim that launched none, spending the plan. */
        status_t abandon_(status_t failure) noexcept {
            for (traversal_context_t& traversal : traversals_) [[maybe_unused]]
                status_t const drained = traversal.drain();
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (shares_[shard].launched) [[maybe_unused]]
                    status_t const kept = owner_->shards_[shard].index->fail_(failure);
                else if (shares_[shard].rows)
                    owner_->shards_[shard].index->unclaim_(shares_[shard].first_id, shares_[shard].rows);
            owner_ = nullptr;
            return failure;
        }

        /** One context per shard, opened for the whole batch. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

        /** One part per shard, in placement order. */
        growing_buffer<share_t, forkunion_allocator_t> shares_;

        /** The locator seats every row's key takes in @c finish. */
        typename locators_map_t::seats_t seats_ {};

        /** The per-shard insertion configuration, applied uniformly. */
        insert_config_t config_ {};

        /** Rows in the batch. */
        std::size_t rows_ = 0;

        /** The shard row zero of the batch lands on. */
        std::size_t rotation_ = 0;

        /** The telemetry the launches reported as they returned, summed. */
        insert_result_t result_ {};

        /** The first refusal past a step's arguments, which leaves the whole batch unpublished. */
        status_t status_ = status_t::success_k;

        /** The executor's hold, kept for the plan's whole life, one caller at a time. */
        typename executor_t::claim_t streams_;

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once spent. */
        derived_* owner_ = nullptr;

        friend sharded_indexes;
    };

    /**
     *  @brief Opens a batch of @p rows: takes the executor, opens one context per shard, claims
     *      every shard's rows from its watermark, grows every shard's storage and key columns to
     *      them, and claims the locator seats.
     *
     *  A shard with a batch, load, refine, absorb or save in flight refuses its claim, and the
     *  whole plan with @c contended_k.
     *
     *  @param[in] rows Rows in the batch; the plan's steps launch each exactly once, in any order.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        insert_plan_t plan {allocator_};
        plan.rows_ = rows;
        plan.config_ = config;
        if (!rows) {
            plan.owner_ = &derived();
            return {std::move(plan), status_t::success_k};
        }
        // The executor's hold guards the scratch; the shards' claims keep other mutations off.
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);

        std::size_t const shards = shards_.size();
        if (status_t const readied = derived().reserve_traversals_(
                plan.traversals_, executor, min_of(divide_round_up(rows, shards), derived_::window_rows_k), config);
            failed(readied))
            return {{}, readied};
        if (status_t const admitted = derived().admits_insert_(); failed(admitted)) return {{}, admitted};
        auto [seats, seat_status] = locators_of_key_.claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        // Rotating by rows routed keeps a stream of small batches from drifting onto shard zero.
        plan.rotation_ = standard_atomic_ref<std::size_t>(routed_rows_).fetch_add(rows, std::memory_order_relaxed) %
                         shards;
        if (failed(plan.shares_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const taking = rows_from_(rows, first_row_of_shard_(shard, shards, plan.rotation_), shards);
            node_id_t first_id = 0;
            if (taking) {
                auto const [claimed_id, claim_status] = shards_[shard].index->claim_(taking);
                if (failed(claim_status)) {
                    for (std::size_t earlier = 0; earlier != shard; ++earlier)
                        if (plan.shares_[earlier].rows)
                            shards_[earlier].index->unclaim_(plan.shares_[earlier].first_id,
                                                             plan.shares_[earlier].rows);
                    return {{}, claim_status};
                }
                first_id = claimed_id;
            }
            plan.shares_.push_back_reserved({first_id, taking, 0});
        }
        // Every claim taken, a refusal from here on dies with the plan, which gives them back.
        plan.owner_ = &derived();
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const claimed_rows = plan.shares_[shard].first_id + plan.shares_[shard].rows;
            if (!plan.shares_[shard].rows) continue;
            if (status_t const entered = derived().enter_(shard); failed(entered)) return {{}, entered};
            reserve_capacity_t const grown {claimed_rows, static_cast<connectivity_t>(connectivity()), dimensions()};
            if (status_t const reserved = shards_[shard].index->reserve(grown).status; failed(reserved))
                return {{}, reserved};
            if (status_t const covered = derived().cover_keys_(shard, claimed_rows); failed(covered))
                return {{}, covered};
        }
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Inserts a batch under @p keys, dealing rows round-robin across the shards from a
     *      rotating start, so they fill evenly and each sees the whole distribution:
     *      @ref plan_insert, one step over every row, and its @c finish.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row landed in is
     *  the index's business. The keys are what the caller gets its rows back by.
     *
     *  A shard failing past its claim keeps that failure; a shard with a batch, load, refine,
     *  absorb or save in flight refuses the call as @c contended_k.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_rows The batch, one row per entry, in any scalar NumKong casts, imported on
     *      each shard's device.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, rows_view_t new_rows, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (keys.size() != new_rows.rows) return {{size()}, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(new_rows.rows, executor, config);
        if (failed(status)) return {{size()}, status};
        // A refused step leaves rows short, and `finish` reports the refusal it kept or the
        // shortfall.
        [[maybe_unused]] status_t const launched = plan.insert(0, keys, new_rows);
        return plan.finish();
    }

    /** Inserts one row under @p key into whichever shard the rotation names next - the one-row
     *  sibling of the batched form, priced identically: one launch on one device. */
    expected<insert_result_t> insert(key_t const& key, vector_view_t new_vector, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (!new_vector.is_contiguous()) return {{}, status_t::unsupported_layout_k};
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        return insert(std::span<key_t const> {&key, 1u}, row, executor, config);
    }

#pragma endregion Insertion

#pragma region Refinement

    /**
     *  @brief One refine in flight over every device shard, holding every claim word closed
     *      meanwhile.
     *
     *  It owns its walk contexts, their runs and its part starts, so nothing about it lives in the
     *  index; @ref run continues it after a cancel, and @ref finish heals what a stopped pass
     *  leaves and reopens. Every shard's @c refine_plan_t is driven phase by phase from this
     *  thread: each phase queues on every shard's device first, then waits once per shard, so the
     *  devices work side by side. An absorb's plan first lands its donors, one after another, and
     *  only then opens the shards' passes over the parts they appended. The executor's streams
     *  must outlive the plan.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb ever fills it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : arenas_(allocator), traversals_(allocator), shard_plans_(allocator), part_bases_(allocator),
              part_offsets_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : arenas_(std::move(other.arenas_)), traversals_(std::move(other.traversals_)),
              shard_plans_(std::move(other.shard_plans_)), part_bases_(std::move(other.part_bases_)),
              part_offsets_(std::move(other.part_offsets_)), donors_(other.donors_),
              landed_donors_(other.landed_donors_), config_(other.config_), population_(other.population_),
              owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** An unfinished plan heals nothing: it reopens, and the graph is as the last phase left
         *  it. */
        ~refine_plan_t() noexcept {
            if (!owner_) return;
            shard_plans_.clear();
            owner_->reopen_shards_();
        }

        /**
         *  @brief Drives the pending phases on @p executor until the pass ends or @p on_visited
         *      cancels; @c cancelled_k means a later call continues where this one stopped.
         *
         *  Lands the donors still pending first, as @ref plan_absorb describes; a failing one stays
         *  pending, with every later one, for the next call. The executor is held only for the
         *  call.
         *
         *  @param[inout] executor The executor the plan was opened over.
         *  @param[in] on_visited Fires per re-selecting phase with the nodes visited so far and the
         *      population; a @c cancel_k answer stops before the next phase.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (status_t const paired = owner_->pairs_with_(executor); failed(paired)) return paired;
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; landed_donors_ != donors_.size(); ++landed_donors_) {
                derived_*& donor = donors_[landed_donors_];
                if (status_t const closed = donor->close_shards_(); failed(closed)) return closed;
                status_t const landed = owner_->absorb_(*donor, executor, *this);
                // Emptied donor shards are fresh and open, so only a failure leaves them closed.
                donor->reopen_shards_();
                if (failed(landed)) return landed;
                donor = nullptr;
            }
            if (shard_plans_.empty())
                if (status_t const opened = open_shards_(); failed(opened)) return opened;
            return drive_(on_visited);
        }

        /** Stops any re-selection, audits and heals, reopens the claim words, and reports. A plan
         *  whose donors have not all landed heals nothing and reports nothing. */
        expected<refine_result_t> finish(executor_t& executor) noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (status_t const paired = owner_->pairs_with_(executor); failed(paired)) return {{}, paired};
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};
            status_t healed = status_t::success_k;
            if (!shard_plans_.empty()) {
                for (typename index_t::refine_plan_t& plan : shard_plans_) plan.stop();
                dummy_progress_t heal_alone;
                healed = drive_(heal_alone);
                for (typename index_t::refine_plan_t const& plan : shard_plans_)
                    if (status_t const kept = plan.finish().status; failed(kept) && succeeded(healed)) healed = kept;
            }
            refine_result_t const result = running();
            shard_plans_.clear();
            std::exchange(owner_, nullptr)->reopen_shards_();
            return {result, healed};
        }

        /** The totals so far, summed over the shards' passes from the counters and telemetry the
         *  last phase landed: what @ref finish reports once it healed, and zeros while donors are
         *  pending. */
        refine_result_t running() const noexcept {
            refine_result_t result;
            for (typename index_t::refine_plan_t const& plan : shard_plans_) result += plan.finish().value;
            return result;
        }

      private:
        friend sharded_indexes;
        friend derived_;

        /** Opens every shard's pass over its parts on its own context, once no donor is pending. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != owner_->shards_.size(); ++shard) {
                std::span<node_id_t const> parts;
                if (!part_offsets_.empty())
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
                if (status_t const entered = owner_->enter_(shard); failed(entered))
                    return shard_plans_.clear(), entered;
                auto [plan, status] = owner_->shards_[shard].index->plan_refine({0, node_id_unknown_k},
                                                                                traversals_[shard], config_, parts);
                if (failed(status)) return shard_plans_.clear(), status;
                shard_plans_.push_back_reserved(std::move(plan));
            }
            population_ = owner_->size();
            return status_t::success_k;
        }

        /** Drives every shard's pending phases until none follows or @p on_visited cancels; a
         *  failed phase keeps its status in its plan, and the first one answers. */
        template <typename progress_type_>
        status_t drive_(progress_type_& on_visited) noexcept {
            status_t outcome = status_t::success_k;
            std::size_t advancing = 0;
            do {
                bool reselected = false;
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                    reselected |= shard_plans_[shard].reselecting();
                    if (succeeded(owner_->enter_(shard))) static_cast<void>(shard_plans_[shard].queue());
                }
                for (typename index_t::refine_plan_t& plan : shard_plans_)
                    if (status_t const waited = plan.wait(); failed(waited) && succeeded(outcome)) outcome = waited;
                advancing = 0;
                std::size_t visited = 0;
                for (typename index_t::refine_plan_t& plan : shard_plans_) {
                    advancing += plan.advance();
                    visited += plan.visited();
                }
                if (advancing && reselected && on_visited(visited, population_) == continuation_t::cancel_k)
                    return status_t::cancelled_k;
            } while (advancing);
            return outcome;
        }

        /** One run per shard, on its stream, so no other call on the executor rewinds what the plan
         *  keeps between its calls; empty where the contexts own their buffers. */
        growing_buffer<typename executor_t::arena_t, forkunion_allocator_t> arenas_;

        /** One walk context per shard, over its run of the plan. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

        /** Every shard's own pass in placement order; none until the donors landed. */
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** The parts the landings appended, ascending per shard, @c node_id_unknown_k past the
         *  last. */
        growing_buffer<node_id_t, forkunion_allocator_t> part_bases_;

        /** Where each shard's run of @ref part_bases_ starts, and one past the last; empty with no
         *  donors. */
        growing_buffer<std::size_t, forkunion_allocator_t> part_offsets_;

        /** The caller's donor slots, each nulled once landed. */
        std::span<derived_*> donors_;

        /** Donors landed so far. */
        std::size_t landed_donors_ = 0;

        /** The passes' knobs. */
        refine_config_t config_ {};

        /** The rows published once the donors landed - the population @c on_visited reports. */
        std::size_t population_ = 0;

        /** The index refined, closed while the plan lives; null in an empty plan and once
         *  finished. */
        derived_* owner_ = nullptr;
    };

    /** Opens a refine of every shard: closes the claim words and builds everything the plan runs
     *  on, as @ref plan_absorb does with no donors, then opens the shards' passes. */
    expected<refine_plan_t> plan_refine(executor_t& executor, refine_config_t config = {}) noexcept {
        return plan_absorb({}, executor, config);
    }

    /**
     *  @brief Re-selects the neighbors of every stored node and heals whatever that strands:
     *      @ref plan_refine, then @c run and @c finish.
     *
     *  A stopped pass leaves the graph correct and merely less improved. Every shard's claim word
     *  stays closed for the pass, so a batch, load or save in flight refuses it with
     *  @c contended_k, and one arriving meanwhile is refused; whether searches may overlap is the
     *  residency's @c plan_refine to say.
     *
     *  @param[inout] executor The threads or streams and their scratch.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] on_visited Fires per re-selecting phase with the nodes visited so far and the
     *      population; a @c cancel_k answer skips the rounds left, still heals, and reports
     *      @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = derived().plan_refine(executor, config);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every row of each into
     *      this index, one donor after another: donor shard @c s appends to shard @c s modulo the
     *      shard count with its edges rebased and its keys beside it, as a part of its own.
     *
     *  Each donor lands through one plan that does every fallible thing first: a load claiming all
     *  the rows each receiving shard takes, the key slots, the locator seats and the donor's fresh
     *  empty shards. A donor shard then crosses one column at a time on the receiving shard's
     *  stream, its keys beside them; every part lands before any shard publishes, and only then do
     *  the keys resolve. The donor empties into its fresh shards, which keep their devices but no
     *  capacity, its slot nulls, and its shards join the receiving shards' parts in the plan, so at
     *  most one donor's rows are ever held twice.
     *
     *  Once the last donor landed, the plan's re-selection walks every shard across the parts the
     *  landings appended, round by round; a cancel pauses only that walk. This index's claim words
     *  stay closed while the plan lives and the donor's while it lands, so a batch, load or save in
     *  flight on either refuses it with @c contended_k, and an insertion into this index arriving
     *  meanwhile is refused. Searches may not overlap on either index, and nothing may start an
     *  insertion into a donor. An index never reserved takes the first donor's geometry.
     *
     *  @param[inout] donors The indexes giving their rows up, each nulled once consumed; the slots
     *      must outlive the landings.
     *  @param[inout] executor The streams every @c run and @c finish names, driving every device of
     *      every index.
     *  @param[in] config The refine's knobs; zeros mean "derive from connectivity".
     *  @retval config_mismatch_k A null donor, another metric or geometry, a device the executor
     *      does not drive, or this index itself.
     *  @retval contended_k A batch, load or save in flight on this index, or another refine or
     *      absorb.
     */
    expected<refine_plan_t> plan_absorb(std::span<derived_*> donors, executor_t& executor,
                                        refine_config_t config = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        derived_ const* shaped = capacity() ? &derived() : nullptr;
        for (derived_* donor : donors) {
            if (!donor || donor == &derived() || donor->kernel_ != kernel_) return {{}, status_t::config_mismatch_k};
            if (status_t const kept = donor->status(); failed(kept)) return {{}, kept};
            if (status_t const paired = donor->pairs_with_(executor); failed(paired)) return {{}, paired};
            if (!donor->capacity()) continue;
            if (!shaped) shaped = donor;
            else if (donor->dimensions() != shaped->dimensions() || donor->connectivity() != shaped->connectivity())
                return {{}, status_t::config_mismatch_k};
        }
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        if (status_t const closed = close_shards_(); failed(closed)) return {{}, closed};
        refine_plan_t plan {allocator_};
        plan.owner_ = &derived();
        plan.donors_ = donors;
        plan.config_ = config;
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};

        // A shard's parts: its resident one, and one per donor shard landing in it.
        if (!donors.empty()) {
            if (failed(plan.part_offsets_.resize(shards + 1))) return {{}, status_t::bad_alloc_k};
            std::size_t parts = 0;
            for (std::size_t into = 0; into != shards; ++into) {
                plan.part_offsets_[into] = parts++;
                for (derived_ const* donor : donors)
                    for (std::size_t from = into; from < donor->shards_.size(); from += shards) ++parts;
            }
            plan.part_offsets_[shards] = parts;
            if (failed(plan.part_bases_.resize(parts))) return {{}, status_t::bad_alloc_k};
            for (node_id_t& base : plan.part_bases_) base = node_id_unknown_k;
        }
        if (status_t const opened = derived().open_walks_(plan, executor, shaped ? *shaped : derived(),
                                                          donors.empty() ? 1u : 2u);
            failed(opened))
            return {{}, opened};
        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Moves every row of each of @p donors into this index and links them in: @ref plan_absorb,
     *  then @c run and @c finish. @p on_visited answers as in @ref refine, for the re-selection
     *  alone; a cancel still heals and reports @c cancelled_k. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> absorb(std::span<derived_*> donors, executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = derived().plan_absorb(donors, executor, config);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /** Moves every row of @p donor into this index, as the plural @ref absorb does with one
     *  donor. */
    expected<refine_result_t> absorb(derived_& donor, executor_t& executor, refine_config_t config = {}) noexcept {
        derived_* donors[1] = {&donor};
        return derived().absorb(std::span<derived_*>(donors), executor, config);
    }

    /** Splits this index into receivers, the inverse of @ref plan_absorb: a gap on the device
     *  residencies, which answer @c unsupported_k, while the host residency defines its own. */
    expected<refine_plan_t> plan_split(std::span<derived_*>, executor_t&, refine_config_t = {}, u64_t = 0) noexcept {
        return {{}, status_t::unsupported_k};
    }

    /** Moves every row of this index into one graph-local part per receiver and links each in:
     *  @c plan_split, then @c run and @c finish. @p on_visited answers as in @ref refine, for the
     *  re-selection alone; a cancel still heals and reports @c cancelled_k. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> split(std::span<derived_*> receivers, executor_t& executor, refine_config_t config = {},
                                    u64_t seed = 0, progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = derived().plan_split(receivers, executor, config, seed);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /**
     *  @brief Measures how much of every shard a search can reach, shard by shard on its own
     *      device: reachable nodes summed, depth the deepest shard's. Read-only.
     *  @param[inout] executor The streams and their scratch, one per shard.
     */
    expected<health_result_t> health(executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};

        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = derived().reserve_traversals_(traversals, executor, 1, insert_config_t {});
            failed(ready))
            return {{}, ready};

        health_result_t result;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            if (status_t const entered = derived().enter_(shard); failed(entered)) return {{}, entered};
            auto const [shard_health, status] = shards_[shard].index->health(traversals[shard]);
            if (failed(status)) return {{}, status};
            result += shard_health;
        }
        return {result, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Search

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending by
     *      distance - the end-to-end multi-shard lookup.
     *
     *  Takes the executor rather than one context: a context is pinned to the device its stream was
     *  opened on, so no single one can walk every shard.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[inout] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' key columns, so colliding ID spaces still land in one ranking.
     *  @param[inout] distances Distances of the same ranking, parallel to @p keys.
     *
     *  The batched search over one row, so every shard walks at once and the rankings merge as
     *  they land; each shard walks as broad as its expansion factor and keeps at most that many
     *  matches. A failing shard keeps whatever the others already merged, so the reported count
     *  and the ranking always agree.
     */
    expected<search_result_t> search(vector_view_t query_vector, executor_t& executor, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!query_vector.is_contiguous()) return {{}, status_t::unsupported_layout_k};
        std::size_t const expansion = shards_[0].index->non_zero_expansion_factor(config);
        std::size_t const width = min_of(min_of(keys.size(), distances.size()), expansion);
        if (!width) return {{}, pairs_with_(executor)};
        config.expansion_factor = static_cast<typename index_t::small_size_t>(expansion);
        config.count = static_cast<typename index_t::small_size_t>(width);
        batch_t batch {executor, allocator_};
        if (status_t const begun = begin_batch_(batch, 1, config); failed(begun)) return {{}, begun};
        std::size_t count = 0;
        expected<search_result_t> found = search_(batch, vectors_view_t {query_vector.data(), 1u, query_vector.size()},
                                                  config, nk::matrix_span<key_t> {keys.data(), 1u, width},
                                                  nk::matrix_span<distance_t> {distances.data(), 1u, width},
                                                  std::span<std::size_t> {&count, 1u});
        found.value.count = count;
        return found;
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard and gathers the rankings into one
     *      per query.
     *
     *  The executor-driven sibling of @c search, one launch per shard and window rather than one
     *  thread per query: the queries cross @c window_rows_k at a time, every shard walks each
     *  window, lands its matches in its own columns of one host landing, and the landed columns
     *  merge query by query through the key columns.
     *
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts, imported on
     *      each shard's device.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly; the width of
     *      @p keys sets its @c count.
     *  @param[out] keys Merged rankings, one query per row, as many matches as its width.
     *  @param[out] distances The same rankings' distances in any score type, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *  @return The merged hits summed over queries, and the telemetry summed over shards.
     *
     *  A refused shard stops the fan-out but keeps every shard and window already landed, so the
     *  rankings and the counts always agree; every queued stream is awaited before returning, since
     *  the caller's queries and the call's own landing are DMA targets until then. Rankings scored
     *  in f32 or f64 that all live on the first shard's device may merge there instead, into the
     *  same rankings, and land only once every shard did.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts) noexcept {
        if (status_t const refusal = batch_refusal_(executor, queries.rows, keys, distances, counts);
            failed(refusal) || !queries.rows)
            return {{}, refusal};
        config.count = static_cast<typename index_t::small_size_t>(keys.extent(1));
        batch_t batch {executor, allocator_};
        if (status_t const begun = begin_batch_(batch, min_of(queries.rows, derived_::window_rows_k), config);
            failed(begun))
            return {{}, begun};
        auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
            nk::range_t const rows = nk::range(first, first + taking);
            return search_(batch, queries.slice(first, taking), config, keys(rows, nk::slice),
                           distances(rows, nk::slice), counts.subspan(first, taking));
        };
        return for_each_window<search_result_t>(queries.rows, derived_::window_rows_k, search_window);
    }

    /** The batched @c search ranked by a lightning indexer: every shard walks the queries under the
     *  same @p heads, passed to it as is, so whichever shard scored them, the merged distances are:
     *  `1 − Σₕ wₕ · ReLU(qₕ · k)`.
     *  A dot verb: any other metric refuses with @c config_mismatch_k. */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = batch_refusal_(executor, queries.rows, keys, distances, counts);
            failed(refusal) || !queries.rows)
            return {{}, refusal};
        std::size_t const heads_count = heads.weights.extent(1);
        if (heads.weights.extent(0) != queries.rows || heads.heads.extent(0) != queries.rows * heads_count)
            return {{}, status_t::config_mismatch_k};
        config.count = static_cast<typename index_t::small_size_t>(keys.extent(1));
        batch_t batch {executor, allocator_};
        if (status_t const begun = begin_batch_(batch, min_of(queries.rows, derived_::window_rows_k), config, heads);
            failed(begun))
            return {{}, begun};
        auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
            nk::range_t const rows = nk::range(first, first + taking);
            indexer_heads<scalar_t> const window_heads {
                heads.heads(nk::range(first * heads_count, (first + taking) * heads_count), nk::slice),
                heads.weights(rows, nk::slice)};
            return search_(batch, queries.slice(first, taking), config, keys(rows, nk::slice),
                           distances(rows, nk::slice), counts.subspan(first, taking), window_heads);
        };
        return for_each_window<search_result_t>(queries.rows, derived_::window_rows_k, search_window);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row
     *      keeps the pairs its walks scored, keys resolved and dots at their native width.
     *
     *  @param[in] left The left-hand rows, one walk per row, in the shards' stored scalar type, as
     *      in @c search.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[out] keys The key of each kept pair, one left-hand row per row, as many pairs as its
     *      width.
     *  @param[out] products The dot of each kept pair, parallel to @p keys, in any product type.
     *  @param[out] counts Pairs scored per row across every shard, above the row width when
     *      truncated.
     *
     *  A full row keeps whichever pairs landed first. The rows cross @c window_rows_k at a time,
     *  and a refused shard stops the fan-out but keeps every shard and window already landed, so
     *  the pairs and the counts always agree; the first refusal is what the merged result reports.
     *  A metric other than dot refuses with @c config_mismatch_k.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                       std::span<std::size_t> counts) noexcept {
        std::size_t const rows_count = left.rows, wanted = keys.extent(1);
        std::size_t const shards = shards_.size();
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(rows_count, keys, products); failed(refusal))
            return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};
        if (left.columns != dimensions()) return {{}, status_t::config_mismatch_k};

        // Every shard lands its own columns of one host landing, as tall as a window and reused by
        // every one, and its context is sized for that width.
        config.count = static_cast<typename index_t::small_size_t>(wanted);
        std::size_t const landing_rows = min_of(rows_count, derived_::window_rows_k);
        std::size_t const landing_entries = landing_rows * shards * wanted;
        growing_buffer<node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<dot_result_t, forkunion_allocator_t> products_landing {allocator_};
        growing_buffer<u32_t, forkunion_allocator_t> counts_landing {allocator_};
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(products_landing.resize_uninitialized(landing_entries)) ||
            failed(counts_landing.resize_uninitialized(shards * landing_rows)))
            return {{}, status_t::bad_alloc_k};
        batch_t batch {executor, allocator_};
        if (status_t const begun = begin_batch_(batch, landing_rows, config); failed(begun)) return {{}, begun};

        auto shard_columns = [&](std::size_t shard) noexcept {
            return nk::range(shard * wanted, (shard + 1) * wanted);
        };
        auto multiply_window = [&](std::size_t first, std::size_t taking) noexcept {
            // Shards add to these as they land, on the host.
            for (std::size_t& count : counts.subspan(first, taking)) count = 0;
            rows_view_t const window_left = left.slice(first, taking);
            nk::matrix_span<node_id_t> const ids_wide {ids_landing.data(), taking, shards * wanted};
            nk::matrix_span<dot_result_t> const products_wide {products_landing.data(), taking, shards * wanted};
            auto shard_counts = [&](std::size_t shard) noexcept {
                return to_writable_span(counts_landing).subspan(shard * taking, taking);
            };
            auto multiply_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
                return derived().multiply_shard_(shard, window_left, ids_wide(nk::all, shard_columns(shard), nk::slice),
                                                 products_wide(nk::all, shard_columns(shard), nk::slice),
                                                 shard_counts(shard), traversal, config, visible);
            };
            // Shards append as they land until a row fills, so a full row keeps whichever pairs
            // came first.
            auto merge_shard = [&](std::size_t shard, traversal_context_t const& traversal) noexcept {
                nk::matrix_span<node_id_t> const shard_ids = ids_wide(nk::all, shard_columns(shard), nk::slice);
                nk::matrix_span<dot_result_t> const shard_products = products_wide(nk::all, shard_columns(shard),
                                                                                   nk::slice);
                std::span<u32_t> const scored_counts = shard_counts(shard);
                derived().land_products_(shard, traversal, shard_ids, shard_products, scored_counts);
                for (std::size_t row = 0; row != taking; ++row) {
                    std::size_t const kept = min_of(counts[first + row], wanted);
                    std::size_t const scored = scored_counts[row];
                    for (std::size_t pair = 0; pair != min_of(scored, wanted - kept); ++pair) {
                        keys(first + row, kept + pair) = shards_[shard].keys[shard_ids(row, pair)];
                        products(first + row, kept + pair) = shard_products(row, pair);
                    }
                    counts[first + row] += scored;
                }
            };
            return for_each_shard_(batch, multiply_shard, merge_shard, counts.subspan(first, taking));
        };
        return for_each_window<search_result_t>(rows_count, derived_::window_rows_k, multiply_window);
    }

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key among the tokens it ranked, and those maxima sum to that key's
     *      score.
     *
     *  @param[in] queries The token rows of every query, @p rows_per_query consecutive rows each.
     *  @param[in] rows_per_query Rows in every query; the row count must be a whole multiple of it.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly; each row ranks as
     *      many tokens as its expansion factor, and one wider than the device search can land
     *      refuses.
     *  @param[out] keys The best-scoring keys of each query, one query per row, as many keys as its
     *      width.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, at most the width of @p keys.
     *
     *  A key some row never ranked misses that row's maximum, so with non-negative per-row maxima a
     *  score lands below its exact value, never above. Whole queries cross @c window_rows_k rows at
     *  a time, or one query when it outgrows that, and a refused window keeps every one already
     *  landed. A metric other than dot refuses with @c config_mismatch_k.
     */
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys, nk::matrix_span<f64_t> scores,
                                     std::span<std::size_t> counts) noexcept {
        std::size_t const rows_count = queries.rows;
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
        std::size_t const groups_count = rows_count / rows_per_query;
        if (counts.size() != groups_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(groups_count, keys, scores); failed(refusal))
            return {{}, refusal};
        if (!groups_count) return {{}, status_t::success_k};

        // Every token row keeps its whole ranked pool, its keys resolved and merged across the
        // shards; a window holds whole queries, so none splits across two.
        std::size_t const width = shards_[0].index->non_zero_expansion_factor(config);
        std::size_t const window_rows = max_of<std::size_t>(derived_::window_rows_k / rows_per_query, 1) *
                                        rows_per_query;
        std::size_t const landing_rows = min_of(rows_count, window_rows);
        growing_buffer<key_t, forkunion_allocator_t> row_keys {allocator_};
        growing_buffer<distance_t, forkunion_allocator_t> row_distances {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> row_counts {allocator_};
        maxsim_reducer<key_t, distance_t, forkunion_allocator_t> reduce {allocator_};
        if (failed(row_keys.resize_uninitialized(landing_rows * width)) ||
            failed(row_distances.resize_uninitialized(landing_rows * width)) ||
            failed(row_counts.resize_uninitialized(landing_rows)) || failed(reduce.reserve(rows_per_query, width)))
            return {{}, status_t::bad_alloc_k};
        config.count = static_cast<typename index_t::small_size_t>(width);
        batch_t batch {executor, allocator_};
        if (status_t const begun = begin_batch_(batch, landing_rows, config); failed(begun)) return {{}, begun};
        auto maxsim_window = [&](std::size_t first, std::size_t taking) noexcept {
            nk::matrix_span<key_t> const ranked_keys {row_keys.data(), taking, width};
            nk::matrix_span<distance_t> const ranked_distances {row_distances.data(), taking, width};
            std::span<std::size_t> const ranked_counts = to_writable_span(row_counts).first(taking);
            expected<search_result_t> ranked = search_(batch, queries.slice(first, taking), config, ranked_keys,
                                                       ranked_distances, ranked_counts);
            if (!ranked) return ranked;
            std::size_t const first_group = first / rows_per_query, groups = taking / rows_per_query;
            nk::range_t const window_groups = nk::range(first_group, first_group + groups);
            ranked.value.count = reduce(ranked_keys, ranked_distances, ranked_counts, keys(window_groups, nk::slice),
                                        scores(window_groups, nk::slice), counts.subspan(first_group, groups));
            return ranked;
        };
        return for_each_window<search_result_t>(rows_count, window_rows, maxsim_window);
    }

#pragma endregion Search

#pragma region Checkpoints

    /** Bytes a @ref save under @p config writes: one monolith, or every part file and the manifest
     *  measured over a placeholder stem; zero for an invalid config or an unplaced index. The row
     *  count is read now, so a buffer sized here fits a save no row lands before. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        if (shards_.empty() || !dimensions()) return {0, status_t::uninitialized_k};
        if (status_t const valid = config.validate(); failed(valid)) return {0, valid};
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {allocator_};
        if (status_t const named = custom_fields_(fields, config); failed(named)) return {0, named};
        std::span<safegraphs_custom_field_t const> const fields_span = to_span(fields);

        // A header that cannot be formatted is an allocation failure the caller must see, never a
        // zero it would size a file by.
        status_t framing = status_t::success_k;
        auto part_bytes = [&](std::size_t nodes) noexcept {
            auto const [header, status] = monolith_header_(nodes, config, fields_span);
            if (failed(status)) framing = status;
            return failed(status) ? 0 : header.monolith_bytes();
        };
        if (config.layout == checkpoint_layout_t::monolith_k) {
            std::size_t const monolith = part_bytes(size());
            return {monolith, framing};
        }
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += part_bytes(one.index->size());
        growing_buffer<char, forkunion_allocator_t> entry_path {allocator_};
        auto part_at = [&](std::size_t part) noexcept {
            std::size_t const nodes = shards_[part].index->size();
            auto const [name, status] = safegraphs_part_path(entry_path, "index", part, shards_.size());
            if (failed(status)) framing = status;
            return safegraphs_part_t {name, part_bytes(nodes), nodes};
        };
        auto const [manifest, status] = safegraphs_manifest_bytes(shards_.size(), part_at);
        if (failed(status)) return {0, status};
        if (failed(framing)) return {0, framing};
        return {total + manifest, status_t::success_k};
    }

    /** One monolith of a checkpoint: where it is, what its header says, and the tokens its metadata
     *  carries. */
    struct part_t {

        /** The relative path's characters inside the paths arena; empty names the checkpoint path
         *  itself. */
        iota_view<std::size_t> path;

        /** The part's whole bytes - a mapping's view, or the caller's own buffer. */
        bytes_view_t monolith;

        /** Rows the monolith holds, per its own header and columns. */
        std::size_t nodes = 0;

        /** The monolith's header. */
        safegraphs_header_t header;

        /** The metric token the monolith recorded; empty when it omitted one. */
        safegraphs_name_t metric;
    };

    /**
     *  @brief One save in flight across every shard: one shard plan each, the parts they write
     *      into, and the tail @ref finish commits - the keys, the headers, and the manifest.
     *
     *  Steps are flat and each names its shard, so a caller walks them from one thread or many;
     *  @ref steps_of names one shard's own run. A device save closes every shard's claim word from
     *  planning until the plan finishes or dies, so no batch, load, refine or other save tears the
     *  rows it copies; a host save settles instead. A dying plan drops its mappings and its shard
     *  plans, leaving parts no header commits.
     */
    struct [[nodiscard]] save_plan_t {

        /** One step of one shard: its shard, and that shard's own step. */
        struct step_t {

            /** The shard the rows come from. */
            std::size_t shard = 0;

            /** The shard plan's own step. */
            typename index_t::save_plan_t::step_t inner;
        };

        save_plan_t() noexcept : save_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_save ever fills it. */
        explicit save_plan_t(forkunion_allocator_t allocator) noexcept
            : plans_(allocator), headers_(allocator), mappings_(allocator), fields_(allocator), path_(allocator) {}
        save_plan_t(save_plan_t&& other) noexcept
            : layout_(other.layout_), monolith_(other.monolith_), plans_(std::move(other.plans_)),
              headers_(std::move(other.headers_)), mappings_(std::move(other.mappings_)),
              fields_(std::move(other.fields_)), path_(std::move(other.path_)), streams_(std::move(other.streams_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** The mappings, the shard plans and the executor's hold die with the plan; an unfinished
         *  one ends the save once its shard plans are gone. */
        ~save_plan_t() noexcept {
            if (!owner_) return;
            plans_.clear();
            owner_->end_save_();
        }

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t shard = 0;
            std::size_t first = 0;
            for (; first + plans_[shard].steps().size() <= position; ++shard) first += plans_[shard].steps().size();
            step.shard = shard;
            step.inner = plans_[shard][position - first];
            return step;
        }

        /** The steps the save spans - every shard's own, in placement order; none once spent. */
        indexed_range<save_plan_t> steps() const noexcept {
            std::size_t count = 0;
            if (owner_)
                for (typename index_t::save_plan_t const& plan : plans_) count += plan.steps().size();
            return {this, count};
        }

        /** The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
        iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
            if (!owner_ || shard >= plans_.size()) return {};
            std::size_t first = 0;
            for (std::size_t earlier = 0; earlier != shard; ++earlier) first += plans_[earlier].steps().size();
            return {first, first + plans_[shard].steps().size()};
        }

        /** Lands step @p step through its own shard's plan, rebasing through @p scratch on the host
         *  residency; a device shard leaves it untouched. */
        status_t save(step_t const& step, nk::matrix_span<node_id_t> scratch) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (step.shard >= plans_.size()) return status_t::config_mismatch_k;
            return plans_[step.shard].save(step.inner, scratch);
        }

        /**
         *  @brief Pronounces the checkpoint complete: every shard plan closes, its keys land at its
         *      base, the headers follow, and a distributed save commits with its manifest - the
         *      commit record, so a failed save leaves parts and no checkpoint.
         *
         *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
         *      the remaining shards and reports @c cancelled_k.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t finish(progress_type_&& on_saved = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            bool const distributed = layout_ == checkpoint_layout_t::distributed_k;
            std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
            for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
                if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
                safegraphs_header_t const& header = headers_[distributed ? shard : 0];
                bytes_span_t const target = distributed ? mappings_[shard].span() : monolith_;
                std::size_t const nodes = plans_[shard].visible;
                std::size_t const base = plans_[shard].id_shift;
                // The key column is chunked, so its rows land run by run.
                bytes_span_t const key_rows = header.rows_of(target, header.key, base, nodes);
                std::size_t written = 0;
                status_t wrote = status_t::success_k;
                owner_->shards_[shard].keys.for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
                    if (succeeded(wrote))
                        wrote = copy_bytes(key_rows.subspan(written, run.size_bytes()),
                                           bytes_of(run.data(), run.size()));
                    written += run.size_bytes();
                });
                if (failed(wrote)) return wrote;
                if (distributed) {
                    if (status_t const framed = safegraphs_write_header(target, header, fields); failed(framed))
                        return framed;
                    if (status_t const flushed = mappings_[shard].sync(); failed(flushed)) return flushed;
                }
                if (on_saved(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
            }
            if (distributed) {
                if (status_t const committed = write_manifest_(); failed(committed)) return committed;
            }
            else {
                if (status_t const framed = safegraphs_write_header(monolith_, headers_[0], fields); failed(framed))
                    return framed;
                if (!mappings_.empty())
                    if (status_t const flushed = mappings_[0].sync(); failed(flushed)) return flushed;
            }
            std::exchange(owner_, nullptr)->end_save_();
            return status_t::success_k;
        }

      private:

        /** The manifest that commits a distributed save; entries name their parts relative to
         *  it. */
        status_t write_manifest_() noexcept {
            std::size_t const shards = plans_.size();
            std::string_view const path = to_string_view(path_);
            std::string_view const base = safegraphs_basename(safegraphs_stem(path));
            growing_buffer<char, forkunion_allocator_t> entry_path {owner_->allocator_};
            auto part_at = [&](std::size_t shard) noexcept {
                auto const [name, status] = safegraphs_part_path(entry_path, base, shard, shards);
                return safegraphs_part_t {name, headers_[shard].monolith_bytes(), plans_[shard].visible};
            };
            return safegraphs_write_manifest(path, shards, part_at);
        }

        /** The layout the parts were framed in. */
        checkpoint_layout_t layout_ = checkpoint_layout_t::monolith_k;

        /** The one graph every shard writes into; empty for a distributed save. */
        bytes_span_t monolith_;

        /** One plan per shard, in placement order. */
        growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans_;

        /** One header per part - one in all for a monolith. */
        growing_buffer<safegraphs_header_t, forkunion_allocator_t> headers_;

        /** The files the parts were mapped into; empty when the caller owns the bytes. */
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;

        /** The custom fields every header carries, outliving the plan's steps. */
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields_;

        /** The checkpoint's own path, copied so the plan outlives the caller's view. */
        growing_buffer<char, forkunion_allocator_t> path_;

        /** The executor's hold, kept for the plan's whole life on a device residency - one caller
         *  at a time; empty on the host one, whose steps claim it as they fan out. */
        typename executor_t::claim_t streams_;

        /** The index being checkpointed; null in the empty plan a refused @ref plan_save returns,
         *  and once @c finish committed. */
        derived_* owner_ = nullptr;

        friend sharded_indexes;
        friend derived_;
    };

    /**
     *  @brief Opens a save under @p path in the layout @p config names: one monolith of every
     *      shard's rows in placement order with IDs rebased, so it reads as one graph, or one
     *      monolith per shard beside a manifest.
     *
     *  Every part is framed, sized and mapped here, and every header lands in @c finish, so a
     *  failed or cancelled save reads as absent.
     *
     *  @param[in] path The monolith's path, or the manifest's - its parts land beside it as
     *      @c <stem>-00001-of-0000S.safegraphs.
     *  @param[in] config The layout, alignment, certified contract and provenance; the device
     *      storage cannot be settlement-probed, so @c concurrent_k refuses.
     *  @param[inout] executor The streams the exports ride, held until the plan dies.
     */
    expected<save_plan_t> plan_save(std::string_view path, save_config_t const& config, executor_t& executor) noexcept {
        return plan_save_(path, config, &executor);
    }

    /** Opens a save into @p monolith as one graph - the in-memory sibling of the path form; a
     *  manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size the bytes by
     *  @ref storage_requirements; fewer answer @c overflow_k. */
    expected<save_plan_t> plan_save(bytes_span_t monolith, save_config_t const& config, executor_t& executor) noexcept {
        return plan_save_(monolith, config, &executor);
    }

    /**
     *  @brief One load in flight across every shard: the mapped parts, one shard plan each, and
     *      the flat steps their rows arrive in - one per part's share of one chunk or batch.
     *
     *  A shard takes its parts in order, so a plan appends them at one base and @ref finish
     *  publishes every shard at once. Every shard's claim word stays closed from planning until the
     *  plan finishes or dies, so no insertion lands beside it; a device load also holds the
     *  executor. A dying plan releases every claim and publishes nothing.
     */
    struct [[nodiscard]] load_plan_t {

        /** One step of one shard: where it lands, which part it reads, and that part's own ID space
         *  alongside the shard plan's step. */
        struct step_t {

            /** The shard the rows land in. */
            std::size_t shard = 0;

            /** The part the rows come from. */
            std::size_t part = 0;

            /** The rows of that part the shard takes - the ID space they speak. */
            iota_view<node_id_t> source_range;

            /** The shard plan's own step, cut to one part's share of one chunk or batch. */
            typename index_t::load_plan_t::step_t inner;
        };

        load_plan_t() noexcept : load_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_load ever fills it. */
        explicit load_plan_t(forkunion_allocator_t allocator) noexcept
            : mappings_(allocator), parts_(allocator), landings_(allocator), plans_(allocator), steps_(allocator) {}
        load_plan_t(load_plan_t&& other) noexcept
            : mappings_(std::move(other.mappings_)), parts_(std::move(other.parts_)),
              landings_(std::move(other.landings_)), plans_(std::move(other.plans_)), steps_(std::move(other.steps_)),
              streams_(std::move(other.streams_)), owner_(std::exchange(other.owner_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** The mappings and the executor's hold die with the plan; an unfinished one rolls its
         *  shard plans back under the closed claim words, then reopens them. */
        ~load_plan_t() noexcept {
            if (!owner_) return;
            plans_.clear();
            owner_->reopen_shards_();
        }

        /** Rows the checkpoint lands across every shard - what its headers said. */
        std::size_t rows() const noexcept {
            std::size_t landing_rows = 0;
            for (typename index_t::load_plan_t const& plan : plans_) landing_rows += plan.rows;
            return landing_rows;
        }

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            if (position >= steps().size()) return {};
            return steps_[position];
        }

        /** The steps the load spans - every shard's own, in placement order; none once spent. */
        indexed_range<load_plan_t> steps() const noexcept { return {this, owner_ ? steps_.size() : 0}; }

        /** The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
        iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
            if (!owner_) return {};
            std::size_t first = 0;
            while (first != steps_.size() && steps_[first].shard != shard) ++first;
            std::size_t bound = first;
            while (bound != steps_.size() && steps_[bound].shard == shard) ++bound;
            return {first, bound};
        }

        /** Lands step @p step: its part's own columns, under that part's ID space, through the
         *  shard's own plan. */
        status_t load(step_t const& step) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (step.shard >= plans_.size() || step.part >= parts_.size()) return status_t::config_mismatch_k;
            part_t const& part = parts_[step.part];
            typename index_t::columns_view_t const columns {
                part.header.data_of(part.monolith, part.header.value),
                part.header.data_of(part.monolith, part.header.neighbors),
                part.header.norms_of(part.monolith, owner_->norms_column_dtype_())};
            return plans_[step.shard].load(step.inner, columns, step.source_range);
        }

        /**
         *  @brief Publishes the restored graph: every shard's keys land at its base and enter the
         *      locator map, then every shard plan publishes - the order an insert keeps, so a
         *      search never reaches a published row without its key.
         *
         *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k reports
         *      @c cancelled_k.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t finish(progress_type_&& on_loaded = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
                if (status_t const entered = owner_->enter_(shard); failed(entered)) return entered;
                std::size_t first = plans_[shard].base;
                for (checkpoint_assignment_t const& landing : landings_) {
                    std::size_t const rows = landing.rows.size();
                    if (landing.shard != shard || !rows) continue;
                    part_t const& part = parts_[landing.part];
                    std::size_t const stored_bytes = part.header.key.row_bytes();
                    bytes_view_t const part_keys = part.header.rows_of(part.monolith, part.header.key,
                                                                       landing.rows.first, rows);
                    if (part_keys.size() != rows * stored_bytes) return status_t::malformed_k;
                    auto const key_of = [&](std::size_t row) noexcept {
                        return safegraphs_widened_key<key_t>(part_keys.data() + row * stored_bytes,
                                                             part.header.key.dtype);
                    };
                    if (status_t const landed = owner_->land_rows_(shard, first, rows, plans_[shard], key_of);
                        failed(landed))
                        return landed;
                    first += rows;
                }
                // Every plan finishes, an empty one included, since finishing releases the claim.
                if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
            }
            for (std::size_t shard = 0; shard != plans_.size(); ++shard)
                if (on_loaded(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
            atomic_ref<std::size_t>(owner_->routed_rows_).store(rows(), std::memory_order_relaxed);
            std::exchange(owner_, nullptr)->reopen_shards_();
            return status_t::success_k;
        }

      private:

        /** The files the parts were mapped from; every part's rows are read out of them. */
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;

        /** The checkpoint's parts, vetted at planning. */
        growing_buffer<part_t, forkunion_allocator_t> parts_;

        /** Which part lands where, in the order each shard appends them - the sequence @c finish
         *  walks to place the keys. */
        growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> landings_;

        /** One plan per shard, in placement order. */
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans_;

        /** The flat steps, grouped by shard. */
        growing_buffer<step_t, forkunion_allocator_t> steps_;

        /** The executor's hold, kept for the plan's whole life on a device residency - one caller
         *  at a time; empty on the host one. */
        typename executor_t::claim_t streams_;

        /** The index being restored, its claim words closed so no batch interleaves with the
         *  restore; null in the empty plan a refused @ref plan_load returns, and once @c finish
         *  published. */
        derived_* owner_ = nullptr;

        friend sharded_indexes;
        friend derived_;
    };

    /**
     *  @brief Opens a load of the checkpoint at @p path: the content says monolith or manifest,
     *      and every part's header is vetted before a byte lands.
     *
     *  The reservation comes from @p config, and the parts land per its placement - as written,
     *  part @c p on shard @c p mod S, or cut into one stripe per shard. The index must be empty;
     *  the plan reports the rows it will land through @c rows().
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] config The placement and the rows to reserve; zero fits the checkpoint.
     *  @param[inout] executor The streams the imports ride, held until the plan dies.
     */
    expected<load_plan_t> plan_load(std::string_view path, load_config_t const& config, executor_t& executor) noexcept {
        return plan_load_(path, config, &executor);
    }

    /** Opens a load of one monolith held in @p monolith - the implementation the path form maps
     *  for. */
    expected<load_plan_t> plan_load(bytes_view_t monolith, load_config_t const& config, executor_t& executor) noexcept {
        return plan_load_(monolith, config, &executor);
    }

    /**
     *  @brief Serializes the index under @p path - the one-shot over @ref plan_save, its steps
     *      walked by the residency's own drain.
     *
     *  @param[in] path The monolith's path, or the manifest's.
     *  @param[in] config The layout, alignment, synchronization contract and provenance.
     *  @param[inout] executor The threads or streams the shard writes ride.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save_(path, config, &executor);
        if (failed(status)) return status;
        if (status_t const landed = derived().drain_steps_(plan, executor); failed(landed)) return landed;
        return plan.finish(on_saved);
    }

    /** Serializes the index into @p monolith as one graph - the one-shot over the bytes form of
     *  @ref plan_save. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save_(monolith, config, &executor);
        if (failed(status)) return status;
        if (status_t const landed = derived().drain_steps_(plan, executor); failed(landed)) return landed;
        return plan.finish(on_saved);
    }

    /**
     *  @brief Restores the checkpoint at @p path into the placed shards - the one-shot over
     *      @ref plan_load, its steps walked by the residency's own drain.
     *
     *  The index must be empty, and a batch in flight refuses the load as @c contended_k, as the
     *  load refuses a batch. A cancelled or failed load clears it and reports why, so nothing is
     *  published and the next load starts clean.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load_(path, config, &executor);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = derived().drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        // Nothing is published by a failed or cancelled load: the shards go back to empty, once
        // the plan has died and released whatever it claimed.
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = derived().clear();
        return settled;
    }

    /** Restores one monolith held in @p monolith - the one-shot over the bytes form of
     *  @ref plan_load. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t monolith, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load_(monolith, config, &executor);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = derived().drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = derived().clear();
        return settled;
    }

    /** The checkpoint's metadata without loading it - @ref safegraphs_read_metadata. */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
    }

    /** The metadata of one monolith held in memory. */
    static expected<safegraphs_metadata_t> metadata(bytes_view_t buffer) noexcept {
        return safegraphs_read_metadata(buffer);
    }

#pragma endregion Checkpoints

#pragma region Lookups

    /** Entries stored under @p key; a concurrent insertion can only make it under-report. */
    std::size_t count(key_t const& key) const noexcept { return locators_of_key_.count(key); }

    /** Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return locators_of_key_.contains(key); }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  Safe against concurrent insertions, which only ever add; every row rides the stream of
     *  @p executor on its shard's device without claiming it, since no scratch is carved, and one
     *  drain settles them all. A device fault, or a shard on a device @p executor drives no stream
     *  on, leaves rows unwritten with the total honest.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted, executor_t& executor) const noexcept {
        std::size_t const total = queue_rows_(key, vectors, wanted, executor);
        if (total) [[maybe_unused]]
            status_t const drained = executor.drain();
        return total;
    }

    /**
     *  @brief The batched @ref count: the entries stored under every key of @p keys.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[inout] executor Unused on a device residency, since the locator map lives on the
     *      host; it rides in the signature so a caller drives every residency alike.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        for (std::size_t position = 0; position != keys.size(); ++position) entries[position] = count(keys[position]);
        return status_t::success_k;
    }

    /** The batched @ref contains: one in @p entries where a key of @p keys holds any entry, zero
     *  where it holds none; the executor rides in the signature as in the batched @ref count. */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        for (std::size_t position = 0; position != keys.size(); ++position)
            entries[position] = contains(keys[position]) ? 1 : 0;
        return status_t::success_k;
    }

    /**
     *  @brief The batched @ref get: up to @p wanted rows of every key of @p keys, and every key's
     *      true total, every row queued on its shard's stream and the executor drained once.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] rows @b [keys·wanted,dimensions]: row q · @p wanted + i is the i-th row of key
     *      q; refused as @ref rows_refusal says, and with @c unsupported_k in any scalar but the
     *      stored one.
     *  @param[in] wanted Rows kept per key.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted
     *      when its rows truncated, zero when absent.
     *  @param[inout] executor The streams the copies ride, paired positionally with the shards.
     */
    status_t get(std::span<key_t const> keys, rows_span_t rows, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (!rows.holds<scalar_t>()) return status_t::unsupported_k;
        vectors_span_t const vectors = rows.as<scalar_t>();
        if (status_t const refusal = rows_refusal(keys.size() * wanted, vectors, row_values_()); failed(refusal))
            return refusal;
        if (status_t const paired = pairs_with_(executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        for (std::size_t position = 0; position != keys.size(); ++position) {
            std::size_t const first = position * wanted;
            entries[position] = queue_rows_(keys[position], vectors(nk::range(first, first + wanted), nk::slice),
                                            wanted, executor);
        }
        return executor.drain();
    }

#pragma endregion Lookups

  protected:

    /** Every plural names the @p kernel its shards score by and the host allocator its shard table,
     *  key columns, locator map and checkpoint catalogs sit on; nothing is placed by default. */
    sharded_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : shards_(allocator), locators_of_key_(allocator), allocator_(allocator), kernel_(kernel) {}

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~sharded_indexes() noexcept = default;

    derived_& derived() noexcept { return static_cast<derived_&>(*this); }
    derived_ const& derived() const noexcept { return static_cast<derived_ const&>(*this); }

    /** What every window of one batched read shares, taken once by @ref begin_batch_: the executor
     *  and its claim, one context per shard sized for the widest window, and one snapshot of the
     *  shards' visible rows, so every window reads the same rows. */
    struct batch_t {
        executor_t& executor;
        typename executor_t::claim_t claim;
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals;
        growing_buffer<std::size_t, forkunion_allocator_t> visible;
        growing_buffer<std::size_t, forkunion_allocator_t> pending;

        batch_t(executor_t& executor, forkunion_allocator_t allocator) noexcept
            : executor(executor), traversals(allocator), visible(allocator), pending(allocator) {}
    };

#pragma region Hooks

    /** Appends @p shard, vetted by @ref emplace, with an empty key column on the host allocator. */
    status_t place_(std::unique_ptr<index_t>&& shard) noexcept {
        return shards_.push_back(shard_t {std::move(shard), keys_column_t {allocator_}});
    }

    /** Covers rows below @p rows in shard @p shard 's key column. */
    status_t cover_keys_(std::size_t shard, std::size_t rows) noexcept { return shards_[shard].keys.cover(rows); }

    /** Makes shard @p shard 's device current for the launches that follow; nothing to do where
     *  launches name their device. */
    status_t enter_(std::size_t) const noexcept { return status_t::success_k; }

    /** Copies @p rows of a shard's host key column to its device twin; nothing to do where walks
     *  read the host column in place. */
    template <typename on_type_>
    static status_t upload_keys_(std::size_t, iota_view<std::size_t>, on_type_ const&) noexcept {
        return status_t::success_k;
    }

    /** Marks a context's submission done queuing, so its landing can be polled; nothing to do where
     *  a submission marks itself. */
    static status_t record_(traversal_context_t&) noexcept { return status_t::success_k; }

    /** Whether a batch may open on every shard; every shard may, unless the residency says not. */
    status_t admits_insert_() const noexcept { return status_t::success_k; }

    /** Drains a batch's @p traversals and adds what each spent to @p result, where the launches
     *  report their telemetry only once landed; answers @p status, or the first failed drain. */
    static status_t settle_insert_(growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals,
                                   status_t status, insert_result_t& result) noexcept {
        for (traversal_context_t& traversal : traversals)
            if (status_t const drained = traversal.drain(); failed(drained) && succeeded(status)) status = drained;
        // Every context was born zeroed for this plan alone, so its sink holds every launch on
        // its shard.
        for (traversal_context_t const& traversal : traversals) {
            vram_telemetry_block_t const spent = traversal.spent();
            result.visited_members += spent.hops;
            result.computed_distances += spent.scored;
            result.restarts += spent.restarts;
            result.unlinked_neighbors += spent.lock_skips;
            result.tail_drops += spent.tail_drops;
            result.visited_saturations += spent.visited_saturations;
        }
        return status;
    }

    /** Builds one traversal context per shard over its device's stream and run, so a fan-out never
     *  allocates and never shares scratch; @p usage, the arguments of a shard's
     *  @c traversal_memory_usage, sizes them for the widest shard, and every stream holds a context
     *  for each shard it serves. */
    template <typename... usage_types_>
    status_t reserve_traversals_(growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals,
                                 executor_t& executor, usage_types_ const&... usage) noexcept {
        std::size_t bytes_per_shard = 0, shards_per_stream = 0;
        for (shard_t const& one : shards_)
            bytes_per_shard = max_of(bytes_per_shard, one.index->traversal_memory_usage(usage...));
        for (std::size_t stream = 0; stream != executor.size(); ++stream) {
            std::size_t served = 0;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard)
                served += stream_of_(executor, shard) == stream;
            shards_per_stream = max_of(shards_per_stream, served);
        }
        if (status_t const sized = executor.reserve(bytes_per_shard * shards_per_stream); failed(sized)) return sized;
        if (failed(traversals.reserve_exactly(shards_.size()))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            typename executor_t::stream_t& stream = executor.stream(stream_of_(executor, shard));
            traversals.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {stream.arena, stream}});
            if (!traversals.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Opens one run per shard on its stream and one walk context over each, seeded for the widest
     *  shard of @p shaped as it stands, @p walks walks into one visited set; landed rows grow
     *  them. */
    status_t open_walks_(refine_plan_t& plan, executor_t& executor, derived_ const& shaped, unsigned walks) noexcept {
        std::size_t const shards = shards_.size();
        if (failed(plan.arenas_.reserve_exactly(shards)) || failed(plan.traversals_.reserve_exactly(shards)))
            return status_t::bad_alloc_k;
        std::size_t walk_bytes = 0;
        for (shard_t const& one : shaped.shards_)
            walk_bytes = max_of(walk_bytes, one.index->traversal_memory_usage(plan.config_, walks));
        for (std::size_t shard = 0; shard != shards; ++shard) {
            typename executor_t::stream_t& stream = executor.stream(stream_of_(executor, shard));
            plan.arenas_.push_back_reserved(typename executor_t::arena_t {stream.arena.upstream()});
            if (failed(plan.arenas_[shard].reserve_exactly(walk_bytes))) return status_t::bad_alloc_k;
            plan.traversals_.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {plan.arenas_[shard], stream}});
            if (!plan.traversals_.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Lands donor shard @p from 's rows by @p step of shard @p into 's load @p plan, its keys
     *  already in the host column; one host copy per column where both sit in shared memory. */
    static status_t land_part_(std::size_t, typename index_t::load_plan_t& plan,
                               typename index_t::load_plan_t::step_t const& step, derived_ const& donor,
                               std::size_t from, executor_t&) noexcept {
        return plan.load(step, *donor.shards_[from].index, step.source_rows);
    }

    /** Opens shard @p shard 's own save over @p columns, its IDs rebased by @p first_row. */
    expected<typename index_t::save_plan_t> plan_shard_save_(std::size_t shard,
                                                             typename index_t::columns_t const& columns,
                                                             synchronization_t synchronization, std::size_t first_row,
                                                             executor_t*) noexcept {
        return shards_[shard].index->plan_save(columns, synchronization, static_cast<node_id_t>(first_row));
    }

    /** Opens shard @p shard 's own load of @p rows rows at its watermark. */
    expected<typename index_t::load_plan_t> plan_shard_load_(std::size_t shard, std::size_t rows,
                                                             executor_t*) noexcept {
        return shards_[shard].index->plan_load(rows);
    }

    /** Whether the index can write @p config through @p executor: placed, reserved, tokenized,
     *  legal, paired, unbroken, and certified quiescent. */
    status_t saveable_(save_config_t const& config, executor_t* executor) const noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        if (config.synchronization == synchronization_t::concurrent_k) return status_t::config_mismatch_k;
        if (status_t const valid = config.validate(); failed(valid)) return valid;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(*executor);
    }

    /** Takes the executor's streams into @p plan and closes every shard's claim word, which the
     *  device storage needs, since it cannot be settlement-probed. */
    status_t gate_save_(save_plan_t& plan, executor_t* executor) noexcept {
        auto [claim, claim_status] = executor->claim();
        if (failed(claim_status)) return claim_status;
        plan.streams_ = std::move(claim);
        return close_shards_();
    }

    /** Ends a save @ref gate_save_ opened: reopens every claim word. */
    void end_save_() noexcept { reopen_shards_(); }

    /** Whether the index can take a checkpoint through @p executor - placed, empty, unbroken and
     *  paired - then takes the executor's streams and closes every shard's claim word for @p plan,
     *  whose death reopens them, and checks emptiness beneath them, where no batch can land before
     *  the load does. */
    status_t gate_load_(load_plan_t& plan, executor_t* executor) noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        if (status_t const kept = status(); failed(kept)) return kept;
        if (status_t const paired = pairs_with_(*executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor->claim();
        if (failed(claim_status)) return claim_status;
        plan.streams_ = std::move(claim);
        if (status_t const closed = close_shards_(); failed(closed)) return closed;
        plan.owner_ = &derived();
        return size() ? status_t::config_mismatch_k : status_t::success_k;
    }

    /** Walks @p plan 's steps from this one host thread, taking step @c s of every shard before
     *  step @c s+1, so a device copies one step while the host stages another's, and the first
     *  failing step ends it. A save lands each step, a load reads it straight out of its part.
     *  The executor is the plan's already, so it rides unused. */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan, executor_t&) noexcept {
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step,
                                           nk::matrix_span<node_id_t> scratch) { planned.save(step, scratch); };
        std::size_t const shards = shards_.size();
        auto const steps = plan.steps();
        std::size_t deepest = 0;
        for (std::size_t shard = 0; shard != shards; ++shard) deepest = max_of(deepest, plan.steps_of(shard).size());
        for (std::size_t position = 0; position != deepest; ++position)
            for (std::size_t shard = 0; shard != shards; ++shard) {
                iota_view<std::size_t> const own = plan.steps_of(shard);
                if (position >= own.size()) continue;
                status_t landed = status_t::success_k;
                if constexpr (saving_k) landed = plan.save(steps[own.first + position], {});
                else landed = plan.load(steps[own.first + position]);
                if (failed(landed)) return landed;
            }
        return status_t::success_k;
    }

    /** The body of the batched @c search on the first shard's device, where the residency has one;
     *  @c deferred_k sends the batch through the host landing. */
    template <typename score_type_, typename... heads_types_>
    expected<search_result_t> search_on_device_(batch_t&, rows_view_t, search_config_t, nk::matrix_span<key_t>,
                                                nk::matrix_span<score_type_>, std::span<std::size_t>,
                                                heads_types_ const&...) noexcept {
        return {{}, status_t::deferred_k};
    }

    /** Submits shard @p shard 's walks of @p queries, landing its matches in @p ids and
     *  @p distances. */
    template <typename... heads_types_>
    status_t search_shard_(std::size_t shard, rows_view_t queries, nk::matrix_span<node_id_t> ids,
                           nk::matrix_span<distance_t> distances, traversal_context_t& traversal,
                           search_config_t const& config, std::size_t visible, heads_types_ const&... heads) noexcept {
        return shards_[shard]
            .index->submit_search_(queries, ids, distances, traversal, config, visible, heads...)
            .status;
    }

    /** Lands shard @p shard 's matches into @p ids and @p distances once its walks landed; the
     *  submission already named them where the walks write there directly. */
    static void land_search_(std::size_t, traversal_context_t const&, nk::matrix_span<node_id_t>,
                             nk::matrix_span<distance_t>) noexcept {}

    /** Submits shard @p shard 's products of @p left, landing its pairs in @p ids, @p products and
     *  @p counts. */
    status_t multiply_shard_(std::size_t shard, rows_view_t left, nk::matrix_span<node_id_t> ids,
                             nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                             traversal_context_t& traversal, search_config_t const& config,
                             std::size_t visible) noexcept {
        return shards_[shard].index->submit_multiply_(left, ids, products, counts, traversal, config, visible).status;
    }

    /** Lands shard @p shard 's pairs once its walks landed; the submission already named them where
     *  the walks write there directly. */
    static void land_products_(std::size_t, traversal_context_t const&, nk::matrix_span<node_id_t>,
                               nk::matrix_span<dot_result_t>, std::span<u32_t>) noexcept {}

#pragma endregion Hooks

    /** Closes every shard's claim word in shard order, or none: a shard with a batch or load in
     *  flight reopens the ones before it and answers @c contended_k. A batch claims in the same
     *  order, so one refused by a closed shard has claimed nothing on the shards before it. */
    status_t close_shards_() noexcept {
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (status_t const closed = shards_[shard].index->close_(); failed(closed)) {
                while (shard) shards_[--shard].index->reopen_();
                return closed;
            }
        return status_t::success_k;
    }

    /** Reopens every closed shard's claim word at its watermark, the last shard first, so a batch
     *  taking the shards in order finds shard zero closed until every later one reopened. */
    void reopen_shards_() noexcept {
        for (std::size_t shard = shards_.size(); shard;) shards_[--shard].index->reopen_();
    }

    /** The first row of a batch that @p shard takes under @p rotation - a bijection over the
     *  shards, so the residue classes it names partition the batch exactly. */
    static std::size_t first_row_of_shard_(std::size_t shard, std::size_t shards, std::size_t rotation) noexcept {
        return (shard + shards - rotation) % shards;
    }

    /** How many of @p rows a shard starting at @p first takes, stepping by @p shards. */
    static std::size_t rows_from_(std::size_t rows, std::size_t first, std::size_t shards) noexcept {
        return first < rows ? divide_round_up(rows - first, shards) : 0;
    }

    /** Lands the keys of @p rows rows appended at @p first on shard @p shard by its load @p plan,
     *  row @c r taking @p key_of (r): written to the host column, copied to any device twin on the
     *  plan's stream, which the append's own publication drains first, and only then entered into
     *  the locator map, so a failed copy leaves no key resolving to rows a later insert reuses. */
    template <typename key_of_type_>
    status_t land_rows_(std::size_t shard, std::size_t first, std::size_t rows,
                        typename index_t::load_plan_t const& plan, key_of_type_&& key_of) noexcept {
        shard_t& one = shards_[shard];
        if (status_t const covered = derived().cover_keys_(shard, first + rows); failed(covered)) return covered;
        auto [seats, seat_status] = locators_of_key_.claim(rows);
        if (failed(seat_status)) return seat_status;
        std::size_t written = 0;
        one.keys.for_each_span({first, first + rows}, [&](std::span<key_t> run) noexcept {
            for (key_t& key : run) key = key_of(written++);
        });
        if (status_t const uploaded = derived().upload_keys_(shard, {first, first + rows}, plan); failed(uploaded))
            return uploaded;
        for (std::size_t row = first; row != first + rows; ++row)
            locators_of_key_.insert(seats, one.keys[row], key_locator_t {shard, static_cast<node_id_t>(row)});
        return status_t::success_k;
    }

    /** Writes into @p part_bases, one run per shard as @p part_offsets cuts it, a part for every
     *  non-empty shard of @p donor its load @p plans appended, after the resident part and any
     *  earlier donor's; a shard empty beforehand keeps the first part landing in it as its own. */
    template <typename plans_type_>
    void record_parts_(derived_ const& donor, plans_type_ const& plans, node_id_t* part_bases,
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

    /** Whether @p executor drives a stream on every shard's device - shards sharing a device share
     *  its stream, and queue on it in order. */
    status_t pairs_with_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (stream_of_(executor, shard) == executor.size()) return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** The position of the stream on shard @p shard 's device in @p executor, or its size when it
     *  drives none there. */
    std::size_t stream_of_(executor_t const& executor, std::size_t shard) const noexcept {
        int const device = shards_[shard].index->device_specs().ordinal;
        std::size_t stream = 0;
        while (stream != executor.size() && executor.device(stream) != device) ++stream;
        return stream;
    }

    /** Queues the copies of up to @p wanted rows stored under @p key into @p vectors, each on the
     *  stream of @p executor on its shard's device, skipping a shard it drives no stream for, and
     *  answers the true total; the caller drains once for every row it queued. */
    std::size_t queue_rows_(key_t const& key, vectors_span_t vectors, std::size_t wanted,
                            executor_t& executor) const noexcept {
        std::size_t total = 0;
        locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            index_t const& shard = *shards_[locator.shard].index;
            std::size_t const stream = stream_of_(executor, locator.shard);
            if (total < wanted && total < vectors.extent(0) && stream != executor.size() &&
                succeeded(derived().enter_(locator.shard))) [[maybe_unused]]
                status_t const queued = shard.vector(locator.node, vectors.row(total).as_vector(),
                                                     executor.stream(stream).handle);
            ++total;
        });
        return total;
    }

    /** Lands every row of @p donor and consumes it, under both indexes' closed claim words and the
     *  executor's hold @p refining keeps: every fallible step first, then the copies, one drain,
     *  and the publication, and the donor's shards join the plan's parts. */
    status_t absorb_(derived_& donor, executor_t& executor, refine_plan_t& refining) noexcept {
        std::size_t const shards = shards_.size();
        if (!donor.size()) return status_t::success_k;
        if (!capacity())
            if (status_t const reserved = reserve({donor.size(), static_cast<connectivity_t>(donor.connectivity()),
                                                   donor.dimensions()})
                                              .status;
                failed(reserved))
                return reserved;

        // Every shard claims all its rows first, and the donor's fresh shards exist before any
        // lands, so a failure before publication releases them all.
        std::size_t const donors = donor.shards_.size();
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {allocator_};
        growing_buffer<std::unique_ptr<index_t>, forkunion_allocator_t> emptied {allocator_};
        if (failed(plans.reserve_exactly(shards)) || failed(emptied.reserve_exactly(donors)))
            return status_t::bad_alloc_k;
        std::size_t landing = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = into; from < donors; from += shards) rows += donor.shards_[from].index->size();
            auto [plan, status] = derived().plan_shard_load_(into, rows, &executor);
            if (failed(status)) return status;
            // Only a donor on another device crosses through the host, so only it pins a bounce.
            if constexpr (requires { plan.pin_bounce(); }) {
                bool crossing = false;
                for (std::size_t from = into; from < donors; from += shards)
                    crossing |= donor.shards_[from].index->device_specs().ordinal !=
                                shards_[into].index->device_specs().ordinal;
                if (crossing)
                    if (status_t const pinned = plan.pin_bounce(); failed(pinned)) return pinned;
            }
            plans.push_back_reserved(std::move(plan));
            if (status_t const covered = derived().cover_keys_(into, plans[into].base + rows); failed(covered))
                return covered;
            landing += rows;
        }
        auto [seats, seat_status] = locators_of_key_.claim(landing);
        if (failed(seat_status)) return seat_status;
        for (shard_t const& part : donor.shards_) {
            emptied.push_back_reserved(donor.rebuild_(*part.index));
            if (!emptied.back()) return status_t::bad_alloc_k;
        }

        // A refused copy still waits for those queued before it, whose rows go to the next claim.
        status_t queued = status_t::success_k;
        for (std::size_t into = 0; into != shards && succeeded(queued); ++into) {
            shard_t& one = shards_[into];
            typename index_t::load_plan_t& plan = plans[into];
            node_id_t base = plan.base;
            for (std::size_t from = into; from < donors && succeeded(queued); from += shards) {
                shard_t const& part = donor.shards_[from];
                node_id_t const part_rows = static_cast<node_id_t>(part.index->size());
                if (!part_rows) continue;
                typename index_t::load_plan_t::step_t step;
                step.index_rows = {base, base + part_rows};
                step.source_rows = {0, part_rows};
                for (node_id_t row : step.source_rows) one.keys[base + row] = part.keys[row];
                queued = derived().land_part_(into, plan, step, donor, from, executor);
                base += part_rows;
            }
        }
        if (status_t const drained = executor.drain(); failed(queued) || failed(drained))
            return failed(queued) ? queued : drained;
        for (typename index_t::load_plan_t& plan : plans)
            if (status_t const published = plan.finish(); failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                locators_of_key_.insert(seats, shards_[into].keys[row], key_locator_t {into, row});

        // Each donor shard becomes a part of its own in the plan, which sized every shard's run.
        record_parts_(donor, plans, refining.part_bases_.data(), to_span(refining.part_offsets_));
        for (std::size_t from = 0; from != donors; ++from) {
            shard_t& part = donor.shards_[from];
            part.index = std::move(emptied[from]);
            part.keys = keys_column_t {donor.allocator_};
        }
        donor.locators_of_key_.clear();
        atomic_ref<std::size_t>(donor.routed_rows_).store(0, std::memory_order_relaxed);
        return status_t::success_k;
    }

    /** Takes every shard's visible rows into @p visible, unless it is empty, at one instant no
     *  commit straddles; answers their sum. */
    std::size_t snapshot_(std::span<std::size_t> visible) const noexcept {
        while (true) {
            u64_t const before = standard_atomic_ref<u64_t const>(commit_sequence_).load(std::memory_order_acquire);
            if (static_cast<u32_t>(before)) {
                fu::standard_yield_t {}(&commit_sequence_, before, 0u, fu::wait_uncapped_t {});
                continue;
            }
            std::size_t total = 0;
            for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
                std::size_t const rows = shards_[shard].index->size();
                if (!visible.empty()) visible[shard] = rows;
                total += rows;
            }
            // The watermarks' acquires keep this re-read after them.
            if (standard_atomic_ref<u64_t const>(commit_sequence_).load(std::memory_order_relaxed) == before)
                return total;
        }
    }

    /** Zips one (query, shard) pair of parallel landing rows into the scored shape
     *  @c merge_search_results consumes, stopping at the first sentinel ID - an absent match must
     *  never index a key column. @return The live matches repacked. */
    static std::size_t repack_landed_matches_(nk::vector_view<node_id_t> landed_ids,
                                              nk::vector_view<distance_t> landed_distances,
                                              std::span<scored_node_t> frame) noexcept {
        std::size_t live = 0;
        for (; live != landed_ids.size() && landed_ids[live] != node_id_unknown_k; ++live)
            frame[live] = scored_node_t {landed_ids[live], landed_distances[live]};
        return live;
    }

    /** Why a batch of @p rows cannot rank into @p keys, @p scores and @p counts on @p executor: the
     *  checks @c search_ opens with, run once over the whole batch before any window slices it. */
    template <typename score_type_>
    status_t batch_refusal_(executor_t const& executor, std::size_t rows, nk::matrix_span<key_t> keys,
                            nk::matrix_span<score_type_> scores, std::span<std::size_t const> counts) const noexcept {
        if (status_t const paired = pairs_with_(executor); failed(paired)) return paired;
        if (counts.size() != rows) return status_t::config_mismatch_k;
        return rankings_refusal(rows, keys, scores);
    }

    /** The body of the batched @c search over one window of @p batch, passing any indexer @p heads
     *  to every shard as is; @p config counts as many matches as @p keys is wide. */
    template <typename score_type_, typename... heads_types_>
    expected<search_result_t> search_(batch_t& batch, rows_view_t queries, search_config_t config,
                                      nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                      std::span<std::size_t> counts, heads_types_ const&... heads) noexcept {
        std::size_t const queries_count = queries.rows;
        std::size_t const shards = shards_.size();
        if (status_t const paired = pairs_with_(batch.executor); paired != status_t::success_k) return {{}, paired};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};
        // Vetted before anything is carved; the shards would refuse only after the landing and the
        // runs had been sized for a doomed call.
        if (queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        if constexpr (std::is_same_v<score_type_, f32_t> || std::is_same_v<score_type_, f64_t>) {
            expected<search_result_t> found = derived().search_on_device_(batch, queries, config, keys, distances,
                                                                          counts, heads...);
            if (found.status != status_t::deferred_k) return found;
        }

        // The host landing: every shard's matches, one wide matrix each, shard columns side by
        // side; never zeroed, since only columns a landed shard wrote are ever read. One row of
        // them repacks into the merge's frame at a time.
        std::size_t const landing_entries = queries_count * shards * config.count;
        growing_buffer<node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<distance_t, forkunion_allocator_t> distances_landing {allocator_};
        growing_buffer<scored_node_t, forkunion_allocator_t> frame {allocator_};
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(distances_landing.resize_uninitialized(landing_entries)) ||
            failed(frame.resize_uninitialized(config.count)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<node_id_t> const ids_wide {ids_landing.data(), queries_count, shards * config.count};
        nk::matrix_span<distance_t> const distances_wide {distances_landing.data(), queries_count,
                                                          shards * config.count};

        auto shard_columns = [&](std::size_t shard) noexcept {
            return nk::range(shard * config.count, (shard + 1) * config.count);
        };
        auto search_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            return derived().search_shard_(shard, queries, ids_wide(nk::all, shard_columns(shard), nk::slice),
                                           distances_wide(nk::all, shard_columns(shard), nk::slice), traversal, config,
                                           visible, heads...);
        };
        auto merge_shard = [&](std::size_t shard, traversal_context_t const& traversal) noexcept {
            nk::matrix_span<node_id_t> const shard_ids = ids_wide(nk::all, shard_columns(shard), nk::slice);
            nk::matrix_span<distance_t> const shard_distances = distances_wide(nk::all, shard_columns(shard),
                                                                               nk::slice);
            derived().land_search_(shard, traversal, shard_ids, shard_distances);
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const live = repack_landed_matches_(
                    shard_ids.row(query).as_vector(), shard_distances.row(query).as_vector(), to_writable_span(frame));
                counts[query] = merge_search_results(std::span<scored_node_t const> {frame.data(), live},
                                                     shards_[shard].keys, to_span(keys.row(query).as_vector()),
                                                     to_span(distances.row(query).as_vector()), counts[query]);
            }
        };
        for (std::size_t& count : counts) count = 0;
        return for_each_shard_(batch, search_shard, merge_shard, counts);
    }

    /** Claims the executor of @p batch, sizes one context per shard for @p window_rows rows under
     *  @p config and any indexer @p heads, and snapshots the shards' visible rows. */
    template <typename traversal_config_type_, typename... heads_types_>
    status_t begin_batch_(batch_t& batch, std::size_t window_rows, traversal_config_type_ config,
                          heads_types_ const&... heads) noexcept {
        // The executor serializes, never the index: a second caller with its own executor searches
        // concurrently.
        auto [claim, claim_status] = batch.executor.claim();
        if (failed(claim_status)) return claim_status;
        batch.claim = std::move(claim);
        if (status_t const readied = derived().reserve_traversals_(batch.traversals, batch.executor, window_rows,
                                                                   config, heads...);
            failed(readied))
            return readied;
        if (failed(batch.visible.resize(shards_.size())) || failed(batch.pending.reserve_exactly(shards_.size())))
            return status_t::bad_alloc_k;
        snapshot_(to_writable_span(batch.visible));
        return status_t::success_k;
    }

    /** Queues @p submit under the snapshot of @p batch on every shard before awaiting any, hands
     *  each shard and its context to @p merge as its stream lands - whichever lands first, so one
     *  merge overlaps the other walks - and sums the telemetry & the rows' @p counts; the first
     *  refusal is the one kept. */
    template <typename submit_type_, typename merge_type_>
    expected<search_result_t> for_each_shard_(batch_t& batch, submit_type_&& submit, merge_type_&& merge,
                                              std::span<std::size_t const> counts) noexcept {
        std::size_t const shards = shards_.size();
        growing_buffer<traversal_context_t, forkunion_allocator_t>& traversals = batch.traversals;
        growing_buffer<std::size_t, forkunion_allocator_t>& pending = batch.pending;

        // Queue every shard before awaiting any, so the batch costs the slowest device rather than
        // the sum. From here to the last landing there must be no return: the caller's rows and
        // the landing are DMA targets on every queued stream - the refuser's included, since its
        // upload may already be reading the caller's rows.
        status_t first_refusal = status_t::success_k;
        std::size_t queued = 0;
        for (; queued != shards; ++queued) {
            pending.push_back_reserved(queued);
            // A launch fails on a stream of any device but the current one; a refuser is awaited
            // too.
            status_t submitted = succeeded(derived().enter_(queued))
                                     ? submit(queued, traversals[queued], batch.visible[queued])
                                     : status_t::device_error_k;
            if (status_t const recorded = derived().record_(traversals[queued]); succeeded(submitted))
                submitted = recorded;
            if (failed(submitted)) {
                first_refusal = submitted;
                break;
            }
        }

        search_result_t merged;
        while (!pending.empty())
            for (std::size_t position = 0; position < pending.size();) {
                std::size_t const shard = pending[position];
                status_t const landed = traversals[shard].landed();
                if (landed == status_t::deferred_k) {
                    ++position;
                    continue;
                }
                pending[position] = pending.back();
                pending.pop_back();
                if (failed(landed)) {
                    if (succeeded(first_refusal)) first_refusal = landed;
                    continue; // An unlanded landing is unreadable - this shard merges nothing.
                }
                // A landed context's sink holds this call's spend on its shard.
                vram_telemetry_block_t const spent = traversals[shard].spent();
                merged.visited_members += spent.hops;
                merged.computed_distances += spent.scored;
                merged.restarts += spent.restarts;
                merged.tail_drops += spent.tail_drops;
                merged.visited_saturations += spent.visited_saturations;
                if (shard != queued) merge(shard, traversals[shard]); // The refuser landed nothing to merge.
            }
        for (std::size_t const count : counts) merged.count += count;
        return {merged, first_refusal};
    }

    /** The interchange token of one key - the monolith's key column dtype. */
    inline static constexpr dtype_t key_dtype_k = safegraphs_key_dtype<key_t>();

    /** The value column's dtype - the stored scalar's own token, so a packed word's suffix carries
     *  the packing and the column's shape counts logical dimensions. */
    inline static constexpr dtype_t value_column_dtype_k = dtype_of(scalar_t::dtype());

    /** The norms column's dtype: the dot product's own token where the shards keep norms, and
     *  @c unknown_k where they keep none, so no column is written and none is read. */
    dtype_t norms_column_dtype_() const noexcept {
        return !shards_.empty() && shards_[0].index->stores_norms() ? dtype_of(dot_result_t::dtype())
                                                                    : dtype_t::unknown_k;
    }

    /** Storage values per stored row - the copy width of every column. */
    std::size_t row_values_() const noexcept {
        return divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** The custom fields every monolith carries: the metric and the caller's provenance; the value
     *  column's own dtype names the stored scalar. */
    status_t custom_fields_(growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields,
                            save_config_t const& config) const noexcept {
        if (failed(fields.reserve_exactly(1 + config.provenance.size()))) return status_t::bad_alloc_k;
        fields.push_back_reserved({"metric", nk_kernel_name(kernel_)});
        for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
        return status_t::success_k;
    }

    /** The monolith header for @p nodes rows under @p config 's alignment. */
    expected<safegraphs_header_t> monolith_header_(std::size_t nodes, save_config_t const& config,
                                                   std::span<safegraphs_custom_field_t const> fields) const noexcept {
        safegraphs_regular_shape_t const shape {nodes,          key_dtype_k,    value_column_dtype_k, dimensions(),
                                                dtype_t::u32_k, connectivity(), norms_column_dtype_()};
        return safegraphs_regular_header(shape, config.alignment, fields);
    }

    /** Copies @p path into @p buffer, so a plan outlives the caller's view of it. */
    static status_t keep_path_(growing_buffer<char, forkunion_allocator_t>& buffer, std::string_view path) noexcept {
        if (failed(buffer.reserve_exactly(path.size()))) return status_t::bad_alloc_k;
        for (char const character : path) buffer.push_back_reserved(character);
        return status_t::success_k;
    }

    /** Opens a save under @p path; @p executor is null where the plan holds no executor. */
    expected<save_plan_t> plan_save_(std::string_view path, save_config_t const& config,
                                     executor_t* executor) noexcept {
        if (status_t const ready = derived().saveable_(config, executor); failed(ready)) return {{}, ready};
        save_plan_t plan {allocator_};
        if (status_t const gated = derived().gate_save_(plan, executor); failed(gated)) return {{}, gated};
        plan.owner_ = &derived();
        if (status_t const kept = keep_path_(plan.path_, path); failed(kept)) return {{}, kept};
        if (config.layout == checkpoint_layout_t::distributed_k) {
            if (status_t const opened = open_save_parts_(plan, config, executor); failed(opened)) return {{}, opened};
            return {std::move(plan), status_t::success_k};
        }
        auto const [bytes, bytes_status] = storage_requirements(config);
        if (failed(bytes_status)) return {{}, bytes_status};
        auto [mapping, mapping_status] = mapped_file_t::write(path, bytes);
        if (failed(mapping_status)) return {{}, mapping_status};
        if (failed(plan.mappings_.push_back(std::move(mapping)))) return {{}, status_t::bad_alloc_k};
        if (status_t const opened = open_save_monolith_(plan, plan.mappings_[0].span(), config, executor);
            failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a save into @p monolith; @p executor is null where the plan holds no executor. */
    expected<save_plan_t> plan_save_(bytes_span_t monolith, save_config_t const& config,
                                     executor_t* executor) noexcept {
        if (status_t const ready = derived().saveable_(config, executor); failed(ready)) return {{}, ready};
        if (config.layout != checkpoint_layout_t::monolith_k) return {{}, status_t::config_mismatch_k};
        save_plan_t plan {allocator_};
        if (status_t const gated = derived().gate_save_(plan, executor); failed(gated)) return {{}, gated};
        plan.owner_ = &derived();
        if (status_t const opened = open_save_monolith_(plan, monolith, config, executor); failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a load of the checkpoint at @p path; @p executor is null where the plan holds no
     *  executor. */
    expected<load_plan_t> plan_load_(std::string_view path, load_config_t const& config,
                                     executor_t* executor) noexcept {
        load_plan_t plan {allocator_};
        if (status_t const gated = derived().gate_load_(plan, executor); failed(gated)) return {{}, gated};
        if (status_t const listed = catalog_(path, plan.mappings_, plan.parts_); failed(listed)) return {{}, listed};
        if (status_t const opened = open_load_(plan, config, executor); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a load of one monolith held in @p monolith; @p executor is null where the plan holds
     *  no executor. */
    expected<load_plan_t> plan_load_(bytes_view_t monolith, load_config_t const& config,
                                     executor_t* executor) noexcept {
        load_plan_t plan {allocator_};
        if (status_t const gated = derived().gate_load_(plan, executor); failed(gated)) return {{}, gated};
        if (status_t const listed = catalog_monolith_(monolith, plan.parts_); failed(listed)) return {{}, listed};
        if (status_t const opened = open_load_(plan, config, executor); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens shard @p shard 's own save over the @p nodes rows it writes at @p first_row of
     *  @p target; the plan's own @c id_shift is where its keys land in @c finish. */
    status_t open_save_shard_(save_plan_t& plan, std::size_t shard, bytes_span_t target,
                              safegraphs_header_t const& header, std::size_t first_row, std::size_t nodes,
                              save_config_t const& config, executor_t* executor) noexcept {
        typename index_t::columns_t const columns {header.rows_of(target, header.value, first_row, nodes),
                                                   header.rows_of(target, header.neighbors, first_row, nodes),
                                                   header.rows_of(target, header.norms, first_row, nodes)};
        auto [inner, status] = derived().plan_shard_save_(shard, columns, config.synchronization, first_row, executor);
        if (failed(status)) return status;
        plan.plans_.push_back_reserved(std::move(inner));
        return status_t::success_k;
    }

    /** Frames @p monolith for one graph, every shard writing at its own base with its IDs rebased
     *  onto it, and opens one save plan per shard over the rows it owns. */
    status_t open_save_monolith_(save_plan_t& plan, bytes_span_t monolith, save_config_t const& config,
                                 executor_t* executor) noexcept {
        std::size_t const shards = shards_.size();
        plan.layout_ = checkpoint_layout_t::monolith_k;
        plan.monolith_ = monolith;
        if (status_t const named = custom_fields_(plan.fields_, config); failed(named)) return named;
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {allocator_};
        if (failed(plan.plans_.reserve_exactly(shards)) || failed(plan.headers_.reserve_exactly(1)) ||
            failed(shard_rows.reserve_exactly(shards)))
            return status_t::bad_alloc_k;

        // The rows are read once, so a shard that grows between here and its watermark is refused
        // there.
        std::size_t rows = 0;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_rows.push_back_reserved(shards_[shard].index->size());
            rows += shard_rows[shard];
        }
        auto const [header, status] = monolith_header_(rows, config, to_span(plan.fields_));
        if (failed(status)) return status;
        if (monolith.size() < header.monolith_bytes()) return status_t::overflow_k;
        plan.headers_.push_back_reserved(header);
        for (std::size_t shard = 0, base = 0; shard != shards; base += shard_rows[shard], ++shard)
            if (status_t const opened = open_save_shard_(plan, shard, monolith, plan.headers_[0], base,
                                                         shard_rows[shard], config, executor);
                failed(opened))
                return opened;
        return status_t::success_k;
    }

    /** Frames, sizes and maps one part per shard beside the plan's path, and opens one save plan
     *  over each - a part's IDs stay shard-local, so its rows travel verbatim. */
    status_t open_save_parts_(save_plan_t& plan, save_config_t const& config, executor_t* executor) noexcept {
        std::size_t const shards = shards_.size();
        plan.layout_ = checkpoint_layout_t::distributed_k;
        if (status_t const named = custom_fields_(plan.fields_, config); failed(named)) return named;
        if (failed(plan.plans_.reserve_exactly(shards)) || failed(plan.headers_.reserve_exactly(shards)) ||
            failed(plan.mappings_.reserve_exactly(shards)))
            return status_t::bad_alloc_k;

        std::span<safegraphs_custom_field_t const> const fields = to_span(plan.fields_);
        std::string_view const stem = safegraphs_stem(to_string_view(plan.path_));
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const nodes = shards_[shard].index->size();
            auto const [header, header_status] = monolith_header_(nodes, config, fields);
            if (failed(header_status)) return header_status;
            auto const [file_path, path_status] = safegraphs_part_path(part_path, stem, shard, shards);
            if (failed(path_status)) return path_status;
            auto [mapping, mapping_status] = mapped_file_t::write(file_path, header.monolith_bytes());
            if (failed(mapping_status)) return mapping_status;
            plan.headers_.push_back_reserved(header);
            plan.mappings_.push_back_reserved(std::move(mapping));
            if (status_t const opened = open_save_shard_(plan, shard, plan.mappings_[shard].span(),
                                                         plan.headers_[shard], 0, nodes, config, executor);
                failed(opened))
                return opened;
        }
        return status_t::success_k;
    }

    /** Reads one part's header and tokens out of the bytes it carries. */
    status_t describe_part_(part_t& part) const noexcept {
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
        };
        auto const [header, status] = safegraphs_read_header(part.monolith, allocator_, per_custom_field);
        if (failed(status)) return status;
        if (header.monolith_bytes() > part.monolith.size()) return status_t::malformed_k;
        part.header = header;
        part.nodes = header.nodes();
        return status_t::success_k;
    }

    /** Lists @p monolith as the checkpoint's only part. */
    status_t catalog_monolith_(bytes_view_t monolith,
                               growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        part_t part;
        part.monolith = monolith;
        if (status_t const described = describe_part_(part); failed(described)) return described;
        return parts.push_back(part);
    }

    /** Lists the parts at @p path: the magic makes a lone monolith, anything else a manifest whose
     *  every part is mapped and its header read now, so no entry is taken on trust. The mappings
     *  land in @p mappings, which must outlive the load. */
    status_t catalog_(std::string_view path, growing_buffer<mapped_file_t, forkunion_allocator_t>& mappings,
                      growing_buffer<part_t, forkunion_allocator_t>& parts) const noexcept {
        auto [mapping, mapping_status] = mapped_file_t::read(path);
        if (failed(mapping_status)) return mapping_status;
        bytes_view_t const checkpoint = mapping.view();
        if (checkpoint.size() >= safegraphs_prefix_t::magic_k.size() &&
            std::memcmp(checkpoint.data(), safegraphs_prefix_t::magic_k.data(), safegraphs_prefix_t::magic_k.size()) ==
                0) {
            if (status_t const listed = catalog_monolith_(checkpoint, parts); failed(listed)) return listed;
            return mappings.push_back(std::move(mapping));
        }

        growing_buffer<char, forkunion_allocator_t> paths_arena {allocator_};
        status_t kept = status_t::success_k;
        auto per_part = [&](safegraphs_part_t entry) noexcept {
            std::size_t const offset = paths_arena.size();
            for (char const character : entry.path)
                if (failed(paths_arena.push_back(character))) return kept = status_t::bad_alloc_k, void();
            part_t part;
            part.path = {offset, paths_arena.size()};
            if (failed(parts.push_back(part))) kept = status_t::bad_alloc_k;
        };
        auto const [summary, summary_status] = safegraphs_read_manifest(path, allocator_, per_part);
        if (failed(summary_status)) return summary_status;
        if (failed(kept)) return kept;
        if (parts.empty()) return status_t::malformed_k;
        if (failed(mappings.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
        growing_buffer<char, forkunion_allocator_t> part_path {allocator_};
        for (part_t& part : parts) {
            auto const [sibling, sibling_status] = safegraphs_sibling_path(
                part_path, path, to_string_view(paths_arena).substr(part.path.first, part.path.size()));
            if (failed(sibling_status)) return sibling_status;
            auto [part_mapping, part_mapping_status] = mapped_file_t::read(sibling);
            if (failed(part_mapping_status)) return part_mapping_status;
            part.monolith = part_mapping.view();
            if (status_t const described = describe_part_(part); failed(described)) return described;
            mappings.push_back_reserved(std::move(part_mapping));
        }
        return status_t::success_k;
    }

    /** Whether @p part can land here: three regular columns of the index's own dtypes, its keys no
     *  wider than the index's, the metric it names agreeing with this index, the geometry @p first
     *  froze, and every column describing the same rows. */
    status_t vetted_part_(part_t const& part, safegraphs_header_t const& first) const noexcept {
        safegraphs_header_t const& header = part.header;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (!safegraphs_key_widens(header.key.dtype, key_dtype_k) || header.value.dtype != value_column_dtype_k ||
            header.neighbors.dtype != dtype_t::u32_k)
            return status_t::config_mismatch_k;
        if (header.key.shape != column_shape_t {scalar_shape_t {}} || header.value.shape != first.value.shape ||
            header.neighbors.shape != first.neighbors.shape)
            return status_t::config_mismatch_k;
        if (!part.metric.empty() && nk_kernel_named(part.metric.characters, part.metric.view().size()) != kernel_)
            return status_t::config_mismatch_k;
        if (header.key.nodes() != part.nodes || header.value.nodes() != part.nodes ||
            header.neighbors.nodes() != part.nodes)
            return status_t::malformed_k;
        return status_t::success_k;
    }

    /**
     *  @brief Vets the plan's parts, assigns them onto the shards per @p config 's placement,
     *      reserves from it, and opens one load plan per shard over the rows it ends up with.
     *
     *  A shard takes its parts in order and appends them at one base, so the flat steps are the
     *  intersections of each part's rows with the chunks or batches they land in - a part that
     *  spans two answers two steps, and two parts sharing one answer one each.
     */
    status_t open_load_(load_plan_t& plan, load_config_t const& config, executor_t* executor) noexcept {
        std::size_t const shards = shards_.size();
        if (plan.parts_.empty()) return status_t::malformed_k;
        safegraphs_header_t const& first = plan.parts_[0].header;
        for (part_t const& part : plan.parts_)
            if (status_t const vetted = vetted_part_(part, first); failed(vetted)) return vetted;
        connectivity_t const connectivity = static_cast<connectivity_t>(
            regular_width(first.neighbors.shape).value_or(0));
        if (connectivity != connectivity_16_k && connectivity != connectivity_32_k &&
            connectivity != connectivity_128_k)
            return status_t::malformed_k;
        std::size_t const dimensions = regular_width(first.value.shape).value_or(0);
        if (!dimensions) return status_t::malformed_k;

        // Parts onto shards per the placement, and the rows each shard ends up with.
        growing_buffer<std::size_t, forkunion_allocator_t> part_rows {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {allocator_};
        if (failed(part_rows.reserve_exactly(plan.parts_.size())) || failed(shard_rows.resize(shards)))
            return status_t::bad_alloc_k;
        for (part_t const& part : plan.parts_) part_rows.push_back_reserved(part.nodes);
        status_t kept = status_t::success_k;
        assign_parts(config.placement, to_span(part_rows), shards, [&](checkpoint_assignment_t assignment) noexcept {
            if (failed(plan.landings_.push_back(assignment))) kept = status_t::bad_alloc_k;
            shard_rows[assignment.shard] += assignment.rows.size();
        });
        if (failed(kept)) return kept;
        std::size_t widest = 0;
        for (std::size_t const rows : shard_rows) widest = max_of(widest, rows);

        // Growth ahead of the rows the parts carry - the config's or the widest shard times the
        // shard count - and the frozen geometry, when there is one, must match the parts'.
        std::size_t const wanted = max_of(config.capacity, widest * shards);
        if (capacity() && (dimensions != this->dimensions() || connectivity != this->connectivity()))
            return status_t::config_mismatch_k;
        if (!capacity() && !wanted) return status_t::success_k;
        if (status_t const reserved = reserve({wanted, connectivity, dimensions}).status; failed(reserved))
            return reserved;

        if (failed(plan.plans_.reserve_exactly(shards))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [inner, status] = derived().plan_shard_load_(shard, shard_rows[shard], executor);
            if (failed(status)) return status;
            // The parts' columns cross from host memory, through a bounce where a shard keeps one.
            if constexpr (requires { inner.pin_bounce(); })
                if (status_t const pinned = inner.pin_bounce(); failed(pinned)) return pinned;
            plan.plans_.push_back_reserved(std::move(inner));
            typename index_t::load_plan_t const& opened = plan.plans_[shard];
            std::size_t first_row = opened.base;
            for (checkpoint_assignment_t const& assignment : plan.landings_) {
                if (assignment.shard != shard) continue;
                std::size_t const bound_row = first_row + assignment.rows.size();
                for (auto const batch : opened.steps()) {
                    std::size_t const low = max_of<std::size_t>(batch.index_rows.first, first_row);
                    std::size_t const high = min_of<std::size_t>(batch.index_rows.bound, bound_row);
                    if (low >= high) continue;
                    typename load_plan_t::step_t step;
                    step.shard = shard;
                    step.part = assignment.part;
                    step.source_range = {static_cast<node_id_t>(assignment.rows.first),
                                         static_cast<node_id_t>(assignment.rows.bound)};
                    step.inner.position = batch.position;
                    step.inner.index_rows = {static_cast<node_id_t>(low), static_cast<node_id_t>(high)};
                    step.inner.source_rows = {static_cast<node_id_t>(assignment.rows.first + low - first_row),
                                              static_cast<node_id_t>(assignment.rows.first + high - first_row)};
                    if (failed(plan.steps_.push_back(step))) return status_t::bad_alloc_k;
                }
                first_row = bound_row;
            }
        }
        return status_t::success_k;
    }

    /** The owned shards, graph and keys together. A shard's key column grows only ahead of the rows
     *  a search can read, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;

    /** Every key's entries. Duplicates coexist as separate slots, which is what several values
     *  under one key means, and costs one slot per row rather than per key. */
    locators_map_t locators_of_key_;

    /** The host allocator behind the shard table, the key columns, the locator map and every
     *  checkpoint catalog; the shards carry their own. */
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
