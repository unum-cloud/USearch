/**
 *  @file include/usearch/dense/vram_apple_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The sharded index over Apple-resident dense shards - any number per device - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The Apple twin of @c vram_nvidia_indexes: same-geometry shards, each owned and pinned to one
 *  Apple GPU, viewed as one index. Insertions rotate across the shards so they fill evenly and each
 *  holds a representative sample; searches submit to every shard's stream before awaiting any and
 *  gather through @ref merge_search_results as each lands, resolving matches into the caller's
 *  keys. Memory is unified, so the keys live on the host alone, an absorb crosses as host copies,
 *  and a submission lands its own results. The shards and their keys are a @c shard_set and
 *  checkpoints ride @c shard_save_plan and @c shard_load_plan; the fan-out and the drivers are this
 *  tower's own. Attention refuses with @c unsupported_k, as the shards do.
 *
 *  Beyond the executor's claim, each shard's claim word admits one batch or load at a time and
 *  closes around a refine, an absorb, a checkpoint load or a save, which refuse with
 *  @c contended_k while a batch is in flight; a shard's kept failure refuses every later mutation,
 *  and the commit sequence word lets a reader spanning the shards see each batch whole or not at
 *  all.
 */
#pragma once

#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <span>        // `std::span` - the caller's keys & the merged ranking
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move`, `std::forward`, `std::exchange`, `std::swap`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the shard set's words

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the plans' state & the landings
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the host columns' placement
#include "usearch/utilities/safegraphs_format.hpp" // `save_config_t`, `load_config_t`, `safegraphs_read_metadata`
#include "usearch/dense/types.hpp"
#include "usearch/dense/sharded_indexes.hpp"
#include "usearch/dense/shard_checkpoints.hpp"
#include "usearch/dense/vram_apple_index.hpp"

namespace unum::usearch::dense {

/**
 *  @brief The sharded index: same-geometry Apple-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are expressible.
 *  Each shard's keys live on the host alone: unified memory lets a walk resolve them in place, so
 *  no device twin is kept.
 *
 *  @tparam wait_type_ What a reader spanning the shards waits through while a commit is in flight.
 */
template <                                        //
    typename key_type_ = uuid_t,                  //
    nk_dtype_t vector_dtype_ = nk_i8_k,           //
    nk_capability_t capability_ = nk_cap_metal_k, //
    typename wait_type_ = dummy_waiter_t,         //
    typename key_hash_type_ = std::hash<key_type_>>
struct vram_apple_indexes {

    /** The shard type - geometry parameters in the shard's own order, so the two never disagree. */
    using index_t = vram_apple_index<vector_dtype_, capability_>;

    /** The shards, the keys of their rows, the rotation that routes new rows, and the commit word a
     *  reader spanning the shards snapshots by. */
    using shards_t = shard_set<index_t, key_type_, key_hash_type_, fu::standard_atomic_ref, wait_type_>;

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;
    using key_locator_t = typename shards_t::key_locator_t;
    using keys_column_t = typename shards_t::keys_column_t;
    using shard_t = typename shards_t::shard_t;

    /** The streams and their scratch every bulk verb runs on, paired positionally with the
     *  shards. */
    using executor_t = metal_executor_t;
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
    static_assert(index_family<index_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The Apple capabilities the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

    /** The vendor whose devices the shards sit on, beside the vendors' twins of this index. */
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::apple_k;

    /** Rows one window of a batch verb stages on a shard at most: one slab, the granularity
     *  checkpoints move by, so a batch of any size crosses in bounded staging. */
    inline static constexpr std::size_t window_rows_k = power_of_two<std::size_t>(index_t::batch_rows_log2_k);

#pragma region Placement

    /** Every index names the metric its shards score by and the host allocator its shard table, key
     *  columns, locator map and checkpoint catalogs sit on; nothing is placed by default. */
    vram_apple_indexes() = delete;
    explicit vram_apple_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : shards_(kernel, allocator) {}
    vram_apple_indexes(vram_apple_indexes&& other) noexcept : shards_(std::move(other.shards_)) {}
    vram_apple_indexes& operator=(vram_apple_indexes&& other) noexcept { return swap(other), *this; }
    vram_apple_indexes(vram_apple_indexes const&) = delete;
    vram_apple_indexes& operator=(vram_apple_indexes const&) = delete;

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~vram_apple_indexes() noexcept = default;

    /** Exchanges two indexes whole; a move implies the caller holds both exclusively. */
    void swap(vram_apple_indexes& other) noexcept { shards_.swap(other.shards_); }

    /** Appends one shard built here on the device @p stream names, for callers with none to hand
     *  over. */
    status_t emplace(typename executor_t::stream_t const& stream) noexcept {
        std::unique_ptr<index_t> shard = build_(stream.handle.device);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** Appends one shard the caller already built, consuming it, with its key column on the host;
     *  refuses any walking by another kernel, any whose frozen geometry differs from the shards
     *  already here, and any already reserved. @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        return shards_.emplace(std::move(shard), shards_.allocator());
    }

    /** Empties every shard in place, keeping its placement, geometry and capacity; a kept failure
     *  drops with the rows. Demands the same quiescence as @ref reserve. */
    status_t clear() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = rebuild_(*one.index);
            if (!fresh) return status_t::bad_alloc_k;
            one.index = std::move(fresh);
        }
        shards_.forget_rows();
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

#pragma endregion Placement

#pragma region Measures and Reservation

    /** The metric every shard scores by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return shards_.kernel(); }

    /** Shards in the index, in placement order. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }

    /** The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.connectivity(); }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.rows(); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** The first shard's kept failure, or success; asked of the shards, never cached. */
    status_t status() const noexcept {
        for (shard_t const& one : shards_)
            if (status_t const kept = one.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }

    /** Bytes the index occupies - every shard's storage, its key column, and the shared locator
     *  map; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept { return sizeof(vram_apple_indexes) + shards_.memory_usage(); }

    /** Grows every shard ahead of @p config.capacity rows, split evenly over them - a hint, since
     *  the shards, their key columns and the locator map all grow on demand; the first call freezes
     *  the geometry. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return shards_.reserve(config); }

#pragma endregion Measures and Reservation

#pragma region Insertion

    /**
     *  @brief One batch in flight across the device shards.
     *
     *  The executor and every shard's claim over the rows it takes are held for the plan's whole
     *  life, one context per shard opened for the whole batch, every shard's storage and key column
     *  grown to the rows it takes, and the locator seats claimed, so a step fails only on its
     *  devices.
     *
     *  Row @c t of the batch lands on shard @c (rotation+t) mod S. Steps run one after another on
     *  one host thread, each writing its rows' keys and launching the rows, every launch landing
     *  before it answers; @ref finish turns every shard's watermark inside one commit, so a reader
     *  spanning the shards sees the batch whole or not at all. A failed launch publishes nothing,
     *  and neither does a plan dying unfinished: every shard holding launched rows keeps the
     *  failure, so its rows stay invisible and later mutations are refused.
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

        /** An unfinished plan keeps @c cancelled_k on every shard holding launched rows, and gives
         *  back every claim that launched nothing. */
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
            std::size_t const window_rows = window_rows_k * shards;
            for (std::size_t first = 0; first < count && succeeded(status_); first += window_rows) {
                std::size_t const taking = min_of(window_rows, count - first);
                std::size_t const rotation = (rotation_ + first_row + first) % shards;
                for (std::size_t shard = 0; shard != shards && succeeded(status_); ++shard)
                    status_ = launch_(shard, shards_t::first_row_of_shard(shard, shards, rotation),
                                      keys[nk::range(first, first + taking)], rows.slice(first, taking));
            }
            return status_;
        }

        /** Publishes every row at once and reports the batch's telemetry. A plan short of rows
         *  refuses and stays whole, so the missing steps can still land; any other call spends it,
         *  and a failed launch publishes nothing and keeps its failure on every shard holding
         *  launched rows. */
        expected<insert_result_t> finish() noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            std::size_t launched = 0;
            for (share_t const& share : shares_) launched += share.launched;
            if (succeeded(status_) && launched != rows_) return {{owner_->size()}, status_t::config_mismatch_k};
            // Every launch landed before answering and reported its telemetry then.
            insert_result_t result = result_;
            status_t status = status_;
            vram_apple_indexes& owner = *owner_;
            if (failed(status)) {
                result.new_size = owner.size();
                return {result, abandon_(status)};
            }

            // Every shard's claim admits this batch alone, so no turn waits inside the commit.
            owner.shards_.begin_commit();
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (share_t const& share = shares_[shard]; share.launched)
                    if (status_t const published = owner.shards_[shard].index->publish_(share.first_id, share.launched);
                        failed(published) && succeeded(status))
                        status = published;
            owner.shards_.end_commit();

            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                for (std::size_t row = 0; row != shares_[shard].launched; ++row) {
                    node_id_t const node = static_cast<node_id_t>(shares_[shard].first_id + row);
                    owner.shards_.locators().insert(seats_, owner.shards_[shard].keys[node],
                                                    key_locator_t {shard, node});
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
            std::size_t const taking = shards_t::rows_from(count, first, shards);
            share_t& share = shares_[shard];
            if (!taking) return status_t::success_k;
            // Overlapping steps would write keys past the cover the plan grew.
            if (share.launched + taking > share.rows) return status_t::config_mismatch_k;
            shard_t& one = owner_->shards_[shard];
            node_id_t const first_node = static_cast<node_id_t>(share.first_id + share.launched);
            for (std::size_t taken = 0; taken != taking; ++taken)
                one.keys[first_node + static_cast<node_id_t>(taken)] = keys[first + taken * shards];
            expected<insert_result_t> const inserted = one.index->submit_insert_(
                {rows.row(first), taking, rows.columns, rows.stride * shards, rows.dtype}, first_node, {},
                traversals_[shard], config_);
            if (!inserted) return inserted.status;
            share.launched += taking;
            result_.visited_members += inserted.value.visited_members;
            result_.computed_distances += inserted.value.computed_distances;
            result_.restarts += inserted.value.restarts;
            result_.unlinked_neighbors += inserted.value.unlinked_neighbors;
            result_.tail_compactions += inserted.value.tail_compactions;
            result_.tail_dropped += inserted.value.tail_dropped;
            result_.visited_saturations += inserted.value.visited_saturations;
            return status_t::success_k;
        }

        /** Drains every context, keeps @p failure on every shard holding launched rows, and gives
         *  back every claim that launched none, spending the plan. */
        status_t abandon_(status_t failure) noexcept {
            // A launch refused after it queued may still be running on its stream.
            for (traversal_context_t& traversal : traversals_) [[maybe_unused]]
                status_t const drained = traversal.wait();
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (shares_[shard].launched) [[maybe_unused]]
                    status_t const kept = owner_->shards_[shard].index->fail_(failure);
                else if (shares_[shard].rows)
                    owner_->shards_[shard].index->unclaim_(shares_[shard].first_id, shares_[shard].rows);
            owner_ = nullptr;
            return failure;
        }

        /** One context per shard: the batch's launches stay in flight on it until @ref finish. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

        /** One part per shard, in placement order. */
        growing_buffer<share_t, forkunion_allocator_t> shares_;

        /** The locator seats every row's key takes in @c finish. */
        typename shards_t::locators_map_t::seats_t seats_ {};

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
        vram_apple_indexes* owner_ = nullptr;

        friend vram_apple_indexes;
    };

    /**
     *  @brief Opens a batch of @p rows: takes the executor, opens one context per shard, claims
     *      every shard's rows from its watermark, grows every shard's storage and key column to
     *      them, and claims the locator seats.
     *
     *  A shard with a batch, load, refine, absorb or save in flight refuses its claim, and the
     *  whole plan with @c contended_k. Every shard must fit its build launch before any row is
     *  admitted, or a mixed fleet inserts partially, so a shard that cannot refuses the whole plan
     *  with @c config_mismatch_k.
     *
     *  @param[in] rows Rows in the batch; the plan's steps launch each exactly once, in any order.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        insert_plan_t plan {shards_.allocator()};
        plan.rows_ = rows;
        plan.config_ = config;
        if (!rows) {
            plan.owner_ = this;
            return {std::move(plan), status_t::success_k};
        }
        // The executor's hold guards the scratch; the shards' claims keep other mutations off.
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);

        std::size_t const shards = shards_.size();
        if (status_t const readied = fanout_t::reserve_traversals(shards_, plan.traversals_, executor); failed(readied))
            return {{}, readied};
        for (shard_t const& one : shards_)
            if (!one.index->build_scratch_fits()) return {{}, status_t::config_mismatch_k};
        auto [seats, seat_status] = shards_.locators().claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        // Rotating by rows routed keeps a stream of small batches from drifting onto shard zero.
        plan.rotation_ = shards_.route(rows) % shards;
        if (failed(plan.shares_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const taking = shards_t::rows_from(
                rows, shards_t::first_row_of_shard(shard, shards, plan.rotation_), shards);
            node_id_t first_id = 0;
            if (taking) {
                auto const [claimed_id, shard_claim_status] = shards_[shard].index->claim_(taking);
                if (failed(shard_claim_status)) {
                    for (std::size_t earlier = 0; earlier != shard; ++earlier)
                        if (plan.shares_[earlier].rows)
                            shards_[earlier].index->unclaim_(plan.shares_[earlier].first_id,
                                                             plan.shares_[earlier].rows);
                    return {{}, shard_claim_status};
                }
                first_id = claimed_id;
            }
            plan.shares_.push_back_reserved({first_id, taking, 0});
        }
        // Every claim taken, a refusal from here on dies with the plan, which gives them back.
        plan.owner_ = this;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const claimed_rows = plan.shares_[shard].first_id + plan.shares_[shard].rows;
            if (!plan.shares_[shard].rows) continue;
            reserve_capacity_t const grown {claimed_rows, static_cast<connectivity_t>(connectivity()), dimensions()};
            if (status_t const reserved = shards_[shard].index->reserve(grown).status; failed(reserved))
                return {{}, reserved};
            if (status_t const covered = shards_[shard].keys.cover(claimed_rows); failed(covered)) return {{}, covered};
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
     *  @param[in] new_rows The batch, one row per entry, in any scalar NumKong casts.
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
     *  It owns its walk contexts, which own their buffers, and its part starts, so nothing about it
     *  lives in the index; @ref run continues it after a cancel, and @ref finish heals what a
     *  stopped pass leaves and reopens. Every shard's @c refine_plan_t is driven phase by phase
     *  from this thread: each phase queues on every shard's device first, then waits once per
     *  shard, so the devices work side by side. An absorb's plan first lands its donors, one after
     *  another, and only then opens the shards' passes over the parts they appended.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb ever fills it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : traversals_(allocator), shard_plans_(allocator), part_bases_(allocator), part_offsets_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : traversals_(std::move(other.traversals_)), shard_plans_(std::move(other.shard_plans_)),
              part_bases_(std::move(other.part_bases_)), part_offsets_(std::move(other.part_offsets_)),
              donors_(other.donors_), landed_donors_(other.landed_donors_), config_(other.config_),
              population_(other.population_), owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** An unfinished plan heals nothing: it reopens, and the graph is as the last phase left
         *  it. */
        ~refine_plan_t() noexcept {
            if (!owner_) return;
            shard_plans_.clear();
            owner_->shards_.reopen();
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
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired)) return paired;
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; landed_donors_ != donors_.size(); ++landed_donors_) {
                vram_apple_indexes*& donor = donors_[landed_donors_];
                if (status_t const closed = donor->shards_.close(); failed(closed)) return closed;
                status_t const landed = owner_->absorb_(*donor, executor, *this);
                // Emptied donor shards are fresh and open, so only a failure leaves them closed.
                donor->shards_.reopen();
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
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired))
                return {{}, paired};
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
            std::exchange(owner_, nullptr)->shards_.reopen();
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
        friend vram_apple_indexes;

        /** Opens every shard's pass over its parts on its own context, once no donor is pending. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != owner_->shards_.size(); ++shard) {
                std::span<node_id_t const> parts;
                if (!part_offsets_.empty())
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
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
                    [[maybe_unused]] status_t const queued = shard_plans_[shard].queue(traversals_[shard], config_);
                }
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                    if (status_t const waited = shard_plans_[shard].wait(traversals_[shard]);
                        failed(waited) && succeeded(outcome))
                        outcome = waited;
                advancing = 0;
                std::size_t visited = 0;
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                    advancing += shard_plans_[shard].advance(traversals_[shard]);
                    visited += shard_plans_[shard].visited();
                }
                if (advancing && reselected && on_visited(visited, population_) == continuation_t::cancel_k)
                    return status_t::cancelled_k;
            } while (advancing);
            return outcome;
        }

        /** One walk context per shard over its stream, each pass queued on it across phases. */
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
        std::span<vram_apple_indexes*> donors_;

        /** Donors landed so far. */
        std::size_t landed_donors_ = 0;

        /** The passes' knobs. */
        refine_config_t config_ {};

        /** The rows published once the donors landed - the population @c on_visited reports. */
        std::size_t population_ = 0;

        /** The index refined, closed while the plan lives; null in an empty plan and once
         *  finished. */
        vram_apple_indexes* owner_ = nullptr;
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
     *  @c contended_k, and one arriving meanwhile is refused.
     *
     *  @param[inout] executor The streams and their scratch.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] on_visited Fires per re-selecting phase with the nodes visited so far and the
     *      population; a @c cancel_k answer skips the rounds left, still heals, and reports
     *      @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_refine(executor, config);
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
     *  empty shards. A donor shard then crosses as one host copy per column, its keys beside them;
     *  every part lands before any shard publishes, and only then do the keys resolve. The donor
     *  empties into its fresh shards, which keep their devices but no capacity, its slot nulls, and
     *  its shards join the receiving shards' parts in the plan, so at most one donor's rows are
     *  ever held twice.
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
    expected<refine_plan_t> plan_absorb(std::span<vram_apple_indexes*> donors, executor_t& executor,
                                        refine_config_t config = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        for (vram_apple_indexes* donor : donors) {
            if (!donor || donor == this || donor->kernel() != kernel()) return {{}, status_t::config_mismatch_k};
            if (status_t const kept = donor->status(); failed(kept)) return {{}, kept};
            if (status_t const paired = fanout_t::pairs_with(donor->shards_, executor); failed(paired))
                return {{}, paired};
        }
        vram_apple_indexes const* shaped = capacity() ? this : nullptr;
        for (vram_apple_indexes* donor : donors) {
            if (!donor->capacity()) continue;
            if (!shaped) shaped = donor;
            else if (donor->dimensions() != shaped->dimensions() || donor->connectivity() != shaped->connectivity())
                return {{}, status_t::config_mismatch_k};
        }
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.donors_ = donors;
        plan.config_ = config;
        if (failed(plan.shard_plans_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};

        // A shard's parts: its resident one, and one per donor shard landing in it.
        if (!donors.empty()) {
            if (failed(plan.part_offsets_.resize(shards + 1))) return {{}, status_t::bad_alloc_k};
            std::size_t parts = 0;
            for (std::size_t into = 0; into != shards; ++into) {
                plan.part_offsets_[into] = parts++;
                for (vram_apple_indexes const* donor : donors)
                    for (std::size_t from = into; from < donor->shards_.size(); from += shards) ++parts;
            }
            plan.part_offsets_[shards] = parts;
            if (failed(plan.part_bases_.resize(parts))) return {{}, status_t::bad_alloc_k};
            for (node_id_t& base : plan.part_bases_) base = node_id_unknown_k;
        }
        // A refine's walk contexts own their buffers, so they open as any verb's do.
        if (status_t const opened = fanout_t::reserve_traversals(shards_, plan.traversals_, executor); failed(opened))
            return {{}, opened};
        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Moves every row of each of @p donors into this index and links them in: @ref plan_absorb,
     *  then @c run and @c finish. @p on_visited answers as in @ref refine, for the re-selection
     *  alone; a cancel still heals and reports @c cancelled_k. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> absorb(std::span<vram_apple_indexes*> donors, executor_t& executor,
                                     refine_config_t config = {}, progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_absorb(donors, executor, config);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /** Moves every row of @p donor into this index, as the plural @ref absorb does with one
     *  donor. */
    expected<refine_result_t> absorb(vram_apple_indexes& donor, executor_t& executor,
                                     refine_config_t config = {}) noexcept {
        vram_apple_indexes* donors[1] = {&donor};
        return absorb(std::span<vram_apple_indexes*>(donors), executor, config);
    }

    /** Splits this index into receivers, the inverse of @ref plan_absorb: a gap on the device
     *  residencies, which answer @c unsupported_k, while the host residency defines its own. */
    expected<refine_plan_t> plan_split(std::span<vram_apple_indexes*>, executor_t&, refine_config_t = {},
                                       u64_t = 0) noexcept {
        return {{}, status_t::unsupported_k};
    }

    /** Moves every row of this index into one graph-local part per receiver and links each in:
     *  @ref plan_split, then @c run and @c finish. @p on_visited answers as in @ref refine, for the
     *  re-selection alone; a cancel still heals and reports @c cancelled_k. */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> split(std::span<vram_apple_indexes*> receivers, executor_t& executor,
                                    refine_config_t config = {}, u64_t seed = 0,
                                    progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_split(receivers, executor, config, seed);
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
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};

        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals {shards_.allocator()};
        if (status_t const ready = fanout_t::reserve_traversals(shards_, traversals, executor); failed(ready))
            return {{}, ready};

        health_result_t result;
        for (std::size_t shard = 0; shard != shards; ++shard) {
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
     *  they land. A failing shard keeps whatever the others already merged, so the reported count
     *  and the ranking always agree.
     */
    expected<search_result_t> search(vector_view_t query_vector, executor_t& executor, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!query_vector.is_contiguous()) return {{}, status_t::unsupported_layout_k};
        std::size_t const width = min_of(keys.size(), distances.size());
        if (!width) return {{}, fanout_t::pairs_with(shards_, executor)};
        config.expansion_factor = static_cast<typename index_t::small_size_t>(
            shards_[0].index->non_zero_expansion_factor(config));
        config.count = static_cast<typename index_t::small_size_t>(width);
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(); failed(begun)) return {{}, begun};
        std::size_t count = 0;
        expected<search_result_t> found = search_window_(
            fanout, vectors_view_t {query_vector.data(), 1u, query_vector.size()}, config,
            nk::matrix_span<key_t> {keys.data(), 1u, width}, nk::matrix_span<distance_t> {distances.data(), 1u, width},
            std::span<std::size_t> {&count, 1u});
        found.value.count = count;
        return found;
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard and gathers the rankings into one
     *      per query.
     *
     *  The queries cross @c window_rows_k at a time, every shard walks each window, its matches
     *  land in its own columns of one host landing, and the landed columns merge query by query
     *  through the key columns.
     *
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly; the width of
     *      @p keys sets its @c count.
     *  @param[out] keys Merged rankings, one query per row, as many matches as its width.
     *  @param[out] distances The same rankings' distances in any score type, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *  @return The merged hits summed over queries, and the telemetry summed over shards.
     *
     *  A refused shard stops the fan-out but keeps every shard and window already landed, so the
     *  rankings and the counts always agree.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts) noexcept {
        if (status_t const refusal = fanout_t::refusal(shards_, executor, queries.rows, keys, distances, counts);
            failed(refusal) || !queries.rows)
            return {{}, refusal};
        config.count = static_cast<typename index_t::small_size_t>(keys.extent(1));
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(); failed(begun)) return {{}, begun};
        auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
            nk::range_t const rows = nk::range(first, first + taking);
            return search_window_(fanout, queries.slice(first, taking), config, keys(rows, nk::slice),
                                  distances(rows, nk::slice), counts.subspan(first, taking));
        };
        return for_each_window<search_result_t>(queries.rows, window_rows_k, search_window);
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
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = fanout_t::refusal(shards_, executor, queries.rows, keys, distances, counts);
            failed(refusal) || !queries.rows)
            return {{}, refusal};
        std::size_t const heads_count = heads.weights.extent(1);
        if (heads.weights.extent(0) != queries.rows || heads.heads.extent(0) != queries.rows * heads_count)
            return {{}, status_t::config_mismatch_k};
        config.count = static_cast<typename index_t::small_size_t>(keys.extent(1));
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(); failed(begun)) return {{}, begun};
        auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
            nk::range_t const rows = nk::range(first, first + taking);
            indexer_heads<scalar_t> const window_heads {
                heads.heads(nk::range(first * heads_count, (first + taking) * heads_count), nk::slice),
                heads.weights(rows, nk::slice)};
            return search_window_(fanout, queries.slice(first, taking), config, keys(rows, nk::slice),
                                  distances(rows, nk::slice), counts.subspan(first, taking), window_heads);
        };
        return for_each_window<search_result_t>(queries.rows, window_rows_k, search_window);
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
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
            return {{}, paired};
        if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(rows_count, keys, products); failed(refusal))
            return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};
        if (left.columns != dimensions()) return {{}, status_t::config_mismatch_k};

        // Every shard lands its own columns of one host landing, as tall as a window and reused by
        // every one.
        config.count = static_cast<typename index_t::small_size_t>(wanted);
        std::size_t const landing_rows = min_of(rows_count, window_rows_k);
        std::size_t const landing_entries = landing_rows * shards * wanted;
        growing_buffer<node_id_t, forkunion_allocator_t> ids_landing {shards_.allocator()};
        growing_buffer<dot_result_t, forkunion_allocator_t> products_landing {shards_.allocator()};
        growing_buffer<u32_t, forkunion_allocator_t> counts_landing {shards_.allocator()};
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(products_landing.resize_uninitialized(landing_entries)) ||
            failed(counts_landing.resize_uninitialized(shards * landing_rows)))
            return {{}, status_t::bad_alloc_k};
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(); failed(begun)) return {{}, begun};

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
                return shards_[shard].index->submit_multiply_(window_left, wanted, traversal, config, visible).status;
            };
            // Shards append as they land until a row fills, so a full row keeps whichever pairs
            // came first.
            auto merge_shard = [&](std::size_t shard, traversal_context_t const& traversal) noexcept {
                nk::matrix_span<node_id_t> const shard_ids = ids_wide(nk::all, shard_columns(shard), nk::slice);
                nk::matrix_span<dot_result_t> const shard_products = products_wide(nk::all, shard_columns(shard),
                                                                                   nk::slice);
                std::span<u32_t> const scored_counts = shard_counts(shard);
                [[maybe_unused]] std::size_t const landed = shards_[shard].index->land_multiply_(
                    traversal, shard_ids, shard_products, scored_counts);
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
            return fanout.for_each_shard(multiply_shard, merge_shard, counts.subspan(first, taking));
        };
        return for_each_window<search_result_t>(rows_count, window_rows_k, multiply_window);
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
     *      many tokens as its expansion factor.
     *  @param[out] keys The best-scoring keys of each query, one query per row, as many keys as its
     *      width.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, at most the width of @p keys.
     *  @param[out] scored Keys scored per query, before the width of @p keys cut any.
     *
     *  A key some row never ranked misses that row's maximum, so with non-negative per-row maxima a
     *  score lands below its exact value, never above. Whole queries cross @c window_rows_k rows at
     *  a time, or one query when it outgrows that, and a refused window keeps every one already
     *  landed. A metric other than dot refuses with @c config_mismatch_k.
     */
    template <typename score_type_>
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<score_type_> scores, std::span<std::size_t> counts,
                                     std::span<std::size_t> scored) noexcept {
        std::size_t const rows_count = queries.rows;
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
            return {{}, paired};
        if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
        std::size_t const groups_count = rows_count / rows_per_query;
        if (counts.size() != groups_count || scored.size() != groups_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(groups_count, keys, scores); failed(refusal))
            return {{}, refusal};
        if (!groups_count) return {{}, status_t::success_k};

        // Every token row keeps all of its ranked candidates, its keys resolved and merged across
        // the shards; a window holds whole queries, so none splits across two.
        std::size_t const width = shards_[0].index->non_zero_expansion_factor(config);
        std::size_t const window_rows = max_of<std::size_t>(window_rows_k / rows_per_query, 1) * rows_per_query;
        std::size_t const landing_rows = min_of(rows_count, window_rows);
        growing_buffer<key_t, forkunion_allocator_t> row_keys {shards_.allocator()};
        growing_buffer<distance_t, forkunion_allocator_t> row_distances {shards_.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> row_counts {shards_.allocator()};
        maxsim_reducer<key_t, distance_t, forkunion_allocator_t> reduce {shards_.allocator()};
        if (failed(row_keys.resize_uninitialized(landing_rows * width)) ||
            failed(row_distances.resize_uninitialized(landing_rows * width)) ||
            failed(row_counts.resize_uninitialized(landing_rows)) || failed(reduce.reserve(rows_per_query, width)))
            return {{}, status_t::bad_alloc_k};
        config.count = static_cast<typename index_t::small_size_t>(width);
        fanout_t fanout {shards_, executor};
        if (status_t const begun = fanout.begin(); failed(begun)) return {{}, begun};
        auto maxsim_window = [&](std::size_t first, std::size_t taking) noexcept {
            nk::matrix_span<key_t> const ranked_keys {row_keys.data(), taking, width};
            nk::matrix_span<distance_t> const ranked_distances {row_distances.data(), taking, width};
            std::span<std::size_t> const ranked_counts = to_writable_span(row_counts).first(taking);
            expected<search_result_t> ranked = search_window_(fanout, queries.slice(first, taking), config, ranked_keys,
                                                              ranked_distances, ranked_counts);
            if (!ranked) return ranked;
            std::size_t const first_group = first / rows_per_query, groups = taking / rows_per_query;
            nk::range_t const window_groups = nk::range(first_group, first_group + groups);
            ranked.value.count = reduce(ranked_keys, ranked_distances, ranked_counts, keys(window_groups, nk::slice),
                                        scores(window_groups, nk::slice), counts.subspan(first_group, groups),
                                        scored.subspan(first_group, groups));
            return ranked;
        };
        return for_each_window<search_result_t>(rows_count, window_rows, maxsim_window);
    }

    /** Grouped-query attention of every token over every shard, vetted as on NVIDIA and AMD, then
     *  answered by the first shard's face on its own stream; @c config_mismatch_k unless the shards
     *  score by dot. */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, search_config_t config = {}) noexcept {
        std::size_t const rows_count = queries.rows;
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
            return {{}, paired};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.rows != rows_count || outputs.columns != dimensions() ||
            (!carry.empty() && carry.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        traversal_context_t traversal {executor.stream(fanout_t::stream_of(shards_, executor, 0)).handle.share()};
        return shards_[0].index->attend(queries, head_count, std::forward<value_lookup_type_>(value_lookup),
                                        score_scale, outputs, carry, traversal, config);
    }

#pragma endregion Search

#pragma region Checkpoints

    /** What a device checkpoint does beyond the shared plans: it holds the executor's streams for
     *  the plan's whole life, every shard opens its own plan, the keys live in the host columns
     *  alone, and a save closes the claim words, since device storage cannot be
     *  settlement-probed. */
    struct checkpoint_residency_t {

        /** The executor's hold, kept for the plan's whole life, one caller at a time. */
        typename executor_t::claim_t streams;

        static expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                       typename index_t::columns_t const& columns,
                                                                       synchronization_t synchronization,
                                                                       std::size_t first_row) noexcept {
            return shards[shard].index->plan_save(columns, synchronization, static_cast<node_id_t>(first_row));
        }
        static expected<typename index_t::load_plan_t> open_shard_load(shards_t& shards, std::size_t shard,
                                                                       std::size_t rows) noexcept {
            return shards[shard].index->plan_load(rows);
        }
        static status_t land_keys(shards_t&, std::size_t, iota_view<std::size_t>,
                                  typename index_t::load_plan_t const&) noexcept {
            return status_t::success_k;
        }
        static void end_save(shards_t& shards) noexcept { shards.reopen(); }
    };

    /** One save in flight across every shard: one shard plan each, the parts they write into, and
     *  the tail @c finish commits - the keys, the headers, and the manifest. */
    using save_plan_t = shard_save_plan<shards_t, checkpoint_residency_t>;

    /** One load in flight across every shard: the mapped parts, one shard plan each, and the flat
     *  steps their rows arrive in - one per part's share of one batch. */
    using load_plan_t = shard_load_plan<shards_t, checkpoint_residency_t>;

    /** Bytes a @ref save under @p config writes: one monolith, or every part file and the manifest
     *  measured over a placeholder stem; zero for an invalid config or an unplaced index. The row
     *  count is read now, so a buffer sized here fits a save no row lands before. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        return shard_storage_requirements(shards_, config);
    }

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
        auto [residency, status] = gate_save_(config, executor);
        if (failed(status)) return {{}, status};
        return save_plan_t::open(shards_, std::move(residency), path, config);
    }

    /** Opens a save into @p monolith as one graph - the in-memory sibling of the path form; a
     *  manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size the bytes by
     *  @ref storage_requirements; fewer answer @c overflow_k. */
    expected<save_plan_t> plan_save(bytes_span_t monolith, save_config_t const& config, executor_t& executor) noexcept {
        auto [residency, status] = gate_save_(config, executor);
        if (failed(status)) return {{}, status};
        return save_plan_t::open(shards_, std::move(residency), monolith, config);
    }

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
        auto [residency, status] = gate_load_(executor);
        if (failed(status)) return {{}, status};
        return load_plan_t::open(shards_, std::move(residency), path, config);
    }

    /** Opens a load of one monolith held in @p monolith - the implementation the path form maps
     *  for. */
    expected<load_plan_t> plan_load(bytes_view_t monolith, load_config_t const& config, executor_t& executor) noexcept {
        auto [residency, status] = gate_load_(executor);
        if (failed(status)) return {{}, status};
        return load_plan_t::open(shards_, std::move(residency), monolith, config);
    }

    /**
     *  @brief Serializes the index under @p path - the one-shot over @ref plan_save, its steps
     *      walked from this one host thread.
     *
     *  @param[in] path The monolith's path, or the manifest's.
     *  @param[in] config The layout, alignment, synchronization contract and provenance.
     *  @param[inout] executor The streams the shard writes ride.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(path, config, executor);
        if (failed(status)) return status;
        if (status_t const landed = drain_steps_(plan); failed(landed)) return landed;
        return plan.finish(on_saved);
    }

    /** Serializes the index into @p monolith as one graph - the one-shot over the bytes form of
     *  @ref plan_save. */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t monolith, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(monolith, config, executor);
        if (failed(status)) return status;
        if (status_t const landed = drain_steps_(plan); failed(landed)) return landed;
        return plan.finish(on_saved);
    }

    /**
     *  @brief Restores the checkpoint at @p path into the placed shards - the one-shot over
     *      @ref plan_load, its steps walked from this one host thread.
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
            auto [plan, status] = plan_load(path, config, executor);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        // Nothing is published by a failed or cancelled load: the shards go back to empty, once
        // the plan has died and released whatever it claimed.
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /** Restores one monolith held in @p monolith - the one-shot over the bytes form of
     *  @ref plan_load. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t monolith, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(monolith, config, executor);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
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
    std::size_t count(key_t const& key) const noexcept { return shards_.locators().count(key); }

    /** Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return shards_.locators().contains(key); }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  Safe against concurrent insertions, which only ever add; each row is read in place from
     *  unified memory, so the executor rides in the signature unused, as in the batched @ref count.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted, executor_t&) const noexcept {
        std::size_t total = 0;
        shards_.locators().find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < vectors.extent(0)) [[maybe_unused]]
                status_t const copied = shards_[locator.shard].index->vector(locator.node,
                                                                             vectors.row(total).as_vector());
            ++total;
        });
        return total;
    }

    /**
     *  @brief The batched @ref count: the entries stored under every key of @p keys.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[inout] executor Unused, since the locator map lives on the host; it rides in the
     *      signature so a caller drives every residency alike.
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

    /** The batched @ref get: up to @p wanted rows of every key of @p keys, row q · @p wanted + i
     *  being the i-th row of key q, and every key's true total in @p entries; rows in any scalar
     *  but the stored one refuse with @c unsupported_k. */
    status_t get(std::span<key_t const> keys, rows_span_t rows, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (!rows.holds<scalar_t>()) return status_t::unsupported_k;
        vectors_span_t const vectors = rows.as<scalar_t>();
        std::size_t const row_values = divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
        if (status_t const refusal = rows_refusal(keys.size() * wanted, vectors, row_values); failed(refusal))
            return refusal;
        for (std::size_t position = 0; position != keys.size(); ++position) {
            std::size_t const first = position * wanted;
            entries[position] = get(keys[position], vectors(nk::range(first, first + wanted), nk::slice), wanted,
                                    executor);
        }
        return status_t::success_k;
    }

#pragma endregion Lookups

  private:

    /**
     *  @brief One batched read over every shard: the executor's claim, one context per shard over
     *      its device's stream, and one snapshot of the shards' visible rows, so every window reads
     *      the same rows.
     *
     *  Each context grows its buffers at its first verb, and each submission lands its own
     *  results, so nothing is sized ahead and no landing is copied back from a device.
     */
    struct fanout_t {
        using traversals_t = growing_buffer<traversal_context_t, forkunion_allocator_t>;

        /** A fan-out of @p shards over @p executor, claimed by @ref begin. */
        fanout_t(shards_t const& shards, executor_t& executor) noexcept
            : shards_(shards), executor_(executor), traversals_(shards.allocator()), visible_(shards.allocator()),
              pending_(shards.allocator()), landed_ids_(shards.allocator()), landed_distances_(shards.allocator()),
              frame_(shards.allocator()) {}

        /** Whether @p executor drives a stream on every shard's device - shards sharing a device
         *  share its stream, and queue on it in order. */
        static status_t pairs_with(shards_t const& shards, executor_t const& executor) noexcept {
            if (shards.empty()) return status_t::uninitialized_k;
            for (std::size_t shard = 0; shard != shards.size(); ++shard)
                if (stream_of(shards, executor, shard) == executor.size()) return status_t::config_mismatch_k;
            return status_t::success_k;
        }

        /** The position of the stream on shard @p shard 's device in @p executor, or its size when
         *  it drives none there. */
        static std::size_t stream_of(shards_t const& shards, executor_t const& executor, std::size_t shard) noexcept {
            int const device = shards[shard].index->device_specs().ordinal;
            std::size_t stream = 0;
            while (stream != executor.size() && executor.device(stream) != device) ++stream;
            return stream;
        }

        /** Opens one traversal context per shard over its device's stream, so a fan-out never
         *  shares scratch; each grows its buffers at its first verb. */
        static status_t reserve_traversals(shards_t const& shards, traversals_t& traversals,
                                           executor_t& executor) noexcept {
            if (failed(traversals.reserve_exactly(shards.size()))) return status_t::bad_alloc_k;
            for (std::size_t shard = 0; shard != shards.size(); ++shard) {
                traversals.push_back_reserved(
                    traversal_context_t {executor.stream(stream_of(shards, executor, shard)).handle.share()});
                if (!traversals.back()) return status_t::bad_alloc_k;
            }
            return status_t::success_k;
        }

        /** Why a batch of @p rows cannot rank into @p keys, @p scores and @p counts on @p executor:
         *  the checks every window repeats, run once over the whole batch. */
        template <typename score_type_>
        static status_t refusal(shards_t const& shards, executor_t const& executor, std::size_t rows,
                                nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> scores,
                                std::span<std::size_t const> counts) noexcept {
            if (status_t const paired = pairs_with(shards, executor); failed(paired)) return paired;
            if (counts.size() != rows) return status_t::config_mismatch_k;
            return rankings_refusal(rows, keys, scores);
        }

        /** The executor every window queues on. */
        executor_t& executor() noexcept { return executor_; }

        /** Claims the executor, opens one context per shard, and snapshots the shards' visible
         *  rows. */
        status_t begin() noexcept {
            // The executor serializes, never the index: a second caller with its own executor
            // searches concurrently.
            auto [claim, claim_status] = executor_.claim();
            if (failed(claim_status)) return claim_status;
            claim_ = std::move(claim);
            if (status_t const readied = reserve_traversals(shards_, traversals_, executor_); failed(readied))
                return readied;
            if (failed(visible_.resize(shards_.size())) || failed(pending_.reserve_exactly(shards_.size())))
                return status_t::bad_alloc_k;
            shards_.snapshot(to_writable_span(visible_));
            return status_t::success_k;
        }

        /** Submits @p submit under the snapshot on every shard before awaiting any, hands each
         *  shard and its context to @p merge as it lands, and sums the telemetry and the rows'
         *  @p counts; the first refusal is the one kept. */
        template <typename submit_type_, typename merge_type_>
        expected<search_result_t> for_each_shard(submit_type_&& submit, merge_type_&& merge,
                                                 std::span<std::size_t const> counts) noexcept {
            std::size_t const shards = shards_.size();
            status_t first_refusal = status_t::success_k;
            std::size_t queued = 0;
            for (; queued != shards; ++queued) {
                pending_.push_back_reserved(queued);
                if (status_t const submitted = submit(queued, traversals_[queued], visible_[queued]);
                    failed(submitted)) {
                    first_refusal = submitted;
                    break;
                }
            }

            search_result_t merged;
            while (!pending_.empty())
                for (std::size_t position = 0; position < pending_.size();) {
                    std::size_t const shard = pending_[position];
                    status_t const landed = traversals_[shard].landed();
                    if (landed == status_t::deferred_k) {
                        ++position;
                        continue;
                    }
                    pending_[position] = pending_.back();
                    pending_.pop_back();
                    if (failed(landed)) {
                        if (succeeded(first_refusal)) first_refusal = landed;
                        continue; // An unlanded landing is unreadable - this shard merges nothing.
                    }
                    // A landed context's sink holds this call's spend on its shard.
                    vram_telemetry_block_t const spent = traversals_[shard].spent();
                    merged.visited_members += spent.hops;
                    merged.computed_distances += spent.scored;
                    merged.restarts += spent.restarts;
                    merged.tail_compactions += spent.tail_compactions;
                    merged.tail_dropped += spent.tail_dropped;
                    merged.visited_saturations += spent.visited_saturations;
                    if (shard != queued) merge(shard, traversals_[shard]); // The refuser landed nothing to merge.
                }
            for (std::size_t const count : counts) merged.count += count;
            return {merged, first_refusal};
        }

        /** Ranks one window of @p queries: every shard submits its walks, and each one's matches
         *  land into its own columns of one wide landing and merge query by query through its key
         *  column as it lands. Passes any indexer @p heads to every shard as is. */
        template <typename score_type_, typename... heads_types_>
        expected<search_result_t> search_landed(rows_view_t queries, search_config_t config,
                                                nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                                std::span<std::size_t> counts, heads_types_ const&... heads) noexcept {
            std::size_t const queries_count = queries.rows, shards = shards_.size();
            std::size_t const landing_entries = queries_count * shards * config.count;
            frame_.clear();
            if (failed(landed_ids_.reserve_exactly(landing_entries)) ||
                failed(landed_distances_.reserve_exactly(landing_entries)) ||
                failed(frame_.reserve_exactly(config.count)))
                return {{}, status_t::bad_alloc_k};
            frame_.extend_uninitialized(config.count);
            nk::matrix_span<node_id_t> const ids_wide {landed_ids_.data(), queries_count, shards * config.count};
            nk::matrix_span<distance_t> const distances_wide {landed_distances_.data(), queries_count,
                                                              shards * config.count};

            auto shard_columns = [&](std::size_t shard) noexcept {
                return nk::range(shard * config.count, (shard + 1) * config.count);
            };
            auto search_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
                return shards_[shard].index->submit_search_(queries, traversal, config, visible, heads...).status;
            };
            auto merge_shard = [&](std::size_t shard, traversal_context_t const& traversal) noexcept {
                nk::matrix_span<node_id_t> const shard_ids = ids_wide(nk::all, shard_columns(shard), nk::slice);
                nk::matrix_span<distance_t> const shard_distances = distances_wide(nk::all, shard_columns(shard),
                                                                                   nk::slice);
                shards_[shard].index->land_search_(traversal, shard_ids, shard_distances);
                for (std::size_t query = 0; query != queries_count; ++query) {
                    std::size_t live = 0;
                    nk::vector_view<node_id_t> const landed_ids = shard_ids.row(query).as_vector();
                    nk::vector_view<distance_t> const landed_distances = shard_distances.row(query).as_vector();
                    // An absent match must never index a key column.
                    for (; live != landed_ids.size() && landed_ids[live] != node_id_unknown_k; ++live)
                        frame_[live] = scored_node_t {landed_ids[live], landed_distances[live]};
                    counts[query] = merge_search_results(std::span<scored_node_t const> {frame_.data(), live},
                                                         shards_[shard].keys, to_span(keys.row(query).as_vector()),
                                                         to_span(distances.row(query).as_vector()), counts[query]);
                }
            };
            for (std::size_t& count : counts) count = 0;
            return for_each_shard(search_shard, merge_shard, counts);
        }

      private:

        /** The shards the batch reads; borrowed from the tower for the call. */
        shards_t const& shards_;

        /** The executor every window queues on. */
        executor_t& executor_;

        /** The executor's hold, from @ref begin until the fan-out dies. */
        typename executor_t::claim_t claim_;

        /** One context per shard. */
        traversals_t traversals_;

        /** Every shard's visible rows in the one snapshot every window reads below. */
        growing_buffer<node_id_t, forkunion_allocator_t> visible_;

        /** The shards submitted and not landed yet. */
        growing_buffer<std::size_t, forkunion_allocator_t> pending_;

        /** The host landing every window's matches land in, and the frame one row of them
         *  repacks into, grown to the widest window and reused by the rest. */
        growing_buffer<node_id_t, forkunion_allocator_t> landed_ids_;
        growing_buffer<distance_t, forkunion_allocator_t> landed_distances_;
        growing_buffer<scored_node_t, forkunion_allocator_t> frame_;
    };

    /** One unreserved shard placed on @p device; null when the allocation fails. */
    std::unique_ptr<index_t> build_(int device) const noexcept {
        metal_shared_allocator_t const allocator {device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel(), allocator, allocator));
    }

    /** One unreserved shard on the device @p like sits on. */
    std::unique_ptr<index_t> rebuild_(index_t const& like) const noexcept {
        return build_(like.device_specs().ordinal);
    }

    /** Lands every row of @p donor and consumes it, under both indexes' closed claim words and the
     *  executor's hold @p refining keeps: every fallible step first, then one host copy per column,
     *  one drain, and the publication, and the donor's shards join the plan's parts. */
    status_t absorb_(vram_apple_indexes& donor, executor_t& executor, refine_plan_t& refining) noexcept {
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
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        growing_buffer<std::unique_ptr<index_t>, forkunion_allocator_t> emptied {shards_.allocator()};
        if (failed(plans.reserve_exactly(shards)) || failed(emptied.reserve_exactly(donors)))
            return status_t::bad_alloc_k;
        std::size_t landing = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = into; from < donors; from += shards) rows += donor.shards_[from].index->size();
            auto [plan, status] = shards_[into].index->plan_load(rows);
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
            if (status_t const covered = shards_[into].keys.cover(plans[into].base + rows); failed(covered))
                return covered;
            landing += rows;
        }
        auto [seats, seat_status] = shards_.locators().claim(landing);
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
                queued = plan.load(step, *part.index, step.source_rows);
                base += part_rows;
            }
        }
        if (status_t const drained = executor.wait(); failed(queued) || failed(drained))
            return failed(queued) ? queued : drained;
        for (typename index_t::load_plan_t& plan : plans)
            if (status_t const published = plan.finish(); failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                shards_.locators().insert(seats, shards_[into].keys[row], key_locator_t {into, row});

        // Each donor shard becomes a part of its own in the plan, which sized every shard's run.
        shards_.record_parts(donor.shards_, plans, refining.part_bases_.data(), to_span(refining.part_offsets_));
        for (std::size_t from = 0; from != donors; ++from) {
            shard_t& part = donor.shards_[from];
            part.index = std::move(emptied[from]);
            part.keys = keys_column_t {donor.shards_.allocator()};
        }
        donor.shards_.forget_rows();
        return status_t::success_k;
    }

    /** Whether the index can write @p config through @p executor - placed, reserved, legal, paired,
     *  unbroken and certified quiescent - then takes the executor's streams and closes every
     *  shard's claim word, which the device storage needs, since it cannot be settlement-probed. */
    expected<checkpoint_residency_t> gate_save_(save_config_t const& config, executor_t& executor) noexcept {
        if (config.synchronization == synchronization_t::concurrent_k) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (status_t const closed = shards_.close(); failed(closed)) return {{}, closed};
        return {checkpoint_residency_t {std::move(claim)}, status_t::success_k};
    }

    /** Whether the index can take a checkpoint through @p executor - placed, empty, unbroken and
     *  paired - then takes the executor's streams; the load plan closes the claim words. */
    expected<checkpoint_residency_t> gate_load_(executor_t& executor) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (size()) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        return {checkpoint_residency_t {std::move(claim)}, status_t::success_k};
    }

    /** Walks @p plan 's steps from this one host thread, taking step @c s of every shard before
     *  step @c s+1, and the first failing step ends it. A save lands each step, a load reads it
     *  straight out of its part. */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan) noexcept {
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

    /** One window of the batched @c search, ranked through the host landing. */
    template <typename score_type_, typename... heads_types_>
    expected<search_result_t> search_window_(fanout_t& fanout, rows_view_t queries, search_config_t config,
                                             nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                             std::span<std::size_t> counts, heads_types_ const&... heads) noexcept {
        std::size_t const queries_count = queries.rows;
        if (status_t const paired = fanout_t::pairs_with(shards_, fanout.executor()); paired != status_t::success_k)
            return {{}, paired};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};
        // Vetted before anything is carved; the shards would refuse only after the landing had
        // been sized for a doomed call.
        if (queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        return fanout.search_landed(queries, config, keys, distances, counts, heads...);
    }

    /** The shards, the keys of their rows, their routing and the commit word every reader
     *  snapshots by. */
    shards_t shards_;
};

} // namespace unum::usearch::dense
