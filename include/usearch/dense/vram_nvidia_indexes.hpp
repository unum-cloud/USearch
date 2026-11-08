/**
 *  @file include/usearch/dense/vram_nvidia_indexes.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The sharded index over GPU-resident dense shards - any number per device - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The multi-shard layer over @c vram_nvidia_index: same-geometry shards, each owned and pinned to
 *  one CUDA device, viewed as one index. Insertions rotate across the shards so they fill evenly
 *  and each holds a representative sample; searches scatter over every shard and gather through
 *  @ref unum::usearch::utilities::merge_search_results, resolving matches into the caller's keys,
 *  so shards with colliding ID spaces merge into one shared ranking. The shards and their keys are
 *  a @c shard_set, checkpoints ride @c shard_save_plan and @c shard_load_plan, and every batched
 *  read fans out through a @c vram_shard_fanout; the device keys, the CUDA calls and the drivers
 *  are this tower's own.
 *
 *  Beyond the executor's claim, each shard's claim word admits one batch or load at a time and
 *  closes around a refine, an absorb, a checkpoint load or a save, which refuse with
 *  @c contended_k while a batch is in flight; a shard's kept failure refuses every later mutation,
 *  and the commit sequence word lets a reader spanning the shards see each batch whole or not at
 *  all. Mutators hold one gate's shared side per call, and the destructive windows - a clear, each
 *  close, and an absorb emptying its donor in place - take its unique side or answer
 *  @c contended_k. Readers bracket each call with the shard set's epoch, retry once when a window
 *  overlapped them, and answer @c contended_k when a second one did; emptied shards keep their
 *  storage until the index dies.
 *
 *  Plain host C++ like the shard's own face: the attend gather launches through the shard's
 *  declarations, which the kernel side defines.
 */
#pragma once

#include <memory>       // `std::unique_ptr` - an index holds atomics and never moves
#include <mutex>        // `std::unique_lock` - the windows' hold on the gate
#include <shared_mutex> // `std::shared_lock` - the mutators' hold on the gate
#include <span>         // `std::span` - the caller's keys & the merged ranking
#include <string_view>  // `std::string_view` - the checkpoint paths
#include <type_traits>  // `std::is_same_v`, `std::remove_cvref_t`, `std::invoke_result_t`
#include <utility>      // `std::move`, `std::exchange`, `std::swap`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the shard set's words

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the plans' state & the key staging
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the host columns' placement
#include "usearch/utilities/safegraphs_format.hpp" // `save_config_t`, `load_config_t`, `safegraphs_read_metadata`
#include "usearch/utilities/ram_shards.hpp"        // `shard_set` - the shards, their keys & routing
#include "usearch/utilities/ram_walks.hpp"         // `breadth_first_walk`, `cursor_order_t` - the cursor's walk
#include "usearch/utilities/vram_shards.hpp"       // `vram_shard_fanout` - one batched read over the shards
#include "usearch/dense/types.hpp"
#include "usearch/dense/shard_checkpoints.hpp"
#include "usearch/dense/vram_nvidia_index.hpp"

namespace unum::usearch::dense {

template <typename vram_indexes_type_, typename ram_indexes_type_>
struct vram_nvidia_compact_plan;

template <typename indexes_type_>
struct vram_nvidia_indexes_cursor;

/**
 *  @brief The sharded index: same-geometry GPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are expressible.
 *  Each shard owns its keys, twice: on the host, where the merges resolve them and every lookup
 *  probes them, and on its device, where a walk resolves its nodes and the on-device merge reads
 *  them. A key lookup probes every shard; no map spans them.
 *
 *  @tparam wait_type_ What a reader spanning the shards waits through while a commit is in flight.
 */
template <                                            //
    typename key_type_,                               //
    nk_dtype_t vector_dtype_,                         //
    typename kernels_type_,                           //
    typename cache_hints_type_ = dummy_cache_hints_t, //
    typename wait_type_ = dummy_waiter_t,             //
    typename key_hash_type_ = std::hash<key_type_>    //
    >
struct vram_nvidia_indexes {
    template <typename, typename>
    friend struct vram_nvidia_compact_plan;

    /** A reader's walk over every shard's published entries: as stored in placement order, or
     *  breadth-first from the entries it opens on, each with the keys it links to. */
    using cursor_t = vram_nvidia_indexes_cursor<vram_nvidia_indexes>;

    /** The shard type - geometry parameters in the shard's own order, so the two never disagree. */
    using index_t = vram_nvidia_index<vector_dtype_, kernels_type_, wait_type_, cache_hints_type_,
                                      cuda_virtual_memory_allocator<byte_t>, cuda_virtual_memory_allocator<byte_t>,
                                      cuda_arena_allocator_t, key_type_, key_hash_type_>;

    /** The caller-visible identity of every entry, which each shard files beside its own nodes. */
    using key_t = key_type_;

    /** The shards, the rotation that routes new rows, and the commit word a reader spanning the
     *  shards snapshots by. */
    using shards_t = shard_set<index_t, fu::standard_atomic_ref, wait_type_>;

    /** One batched read's fan-out over every shard's device. */
    using fanout_t = utilities::vram_shard_fanout<shards_t, cuda_vendor_t, typename index_t::distance_t>;
    using shard_t = typename shards_t::shard_t;
    using shard_generation_t = typename shards_t::shard_generation_t;

    /** The streams and their scratch every bulk verb runs on, paired positionally with the
     *  shards. */
    using executor_t = cuda_executor_t;
    using traversal_context_t = typename index_t::traversal_context_t;

    /* Scalars and Views */
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vectors_span_t = nk::matrix_span<scalar_t>;

    /* Configs and Results */
    using search_config_t = typename index_t::search_config_t;
    using insert_config_t = typename index_t::insert_config_t;
    using reserve_capacity_t = typename index_t::reserve_capacity_t;
    using geometry_t = typename index_t::geometry_t;
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_config_t = typename index_t::refine_config_t;
    using split_config_t = typename index_t::split_config_t;

    /** A MaxSim's walk knobs and the keys each query keeps, zero keeping every key it scored. */
    struct maxsim_config_t {
        search_config_t search {};
        std::size_t wanted = 0;
    };
    using refine_report_t = typename index_t::refine_report_t;
    using reachability_t = typename index_t::reachability_t;
    static_assert(index_family<index_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The NVIDIA capabilities the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = kernels_type_::capability_k;

    /** The vendor whose devices the shards sit on, beside the vendors' twins of this index. */
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;

    /** Rows one window of a batch verb stages on a shard at most: one slab, the granularity
     *  checkpoints move by, so a batch of any size crosses in bounded staging. */
    inline static constexpr std::size_t window_rows_k = power_of_two<std::size_t>(index_t::batch_rows_log2_k);

#pragma region Placement

    /** Every index names the metric its shards score by, the geometry every shard keeps, and the
     *  host allocator its shard table, its shards' keys and checkpoint catalogs sit on; nothing is
     *  placed by default. */
    vram_nvidia_indexes() = delete;
    explicit vram_nvidia_indexes(nk_kernel_kind_t kernel, geometry_t geometry, forkunion_allocator_t allocator) noexcept
        : shards_(allocator), kernel_(kernel), geometry_(geometry) {}
    vram_nvidia_indexes(vram_nvidia_indexes&& other) noexcept
        : shards_(other.shards_.allocator()), kernel_(other.kernel_), geometry_(other.geometry_) {
        swap(other);
    }
    vram_nvidia_indexes& operator=(vram_nvidia_indexes&& other) noexcept { return swap(other), *this; }
    vram_nvidia_indexes(vram_nvidia_indexes const&) = delete;
    vram_nvidia_indexes& operator=(vram_nvidia_indexes const&) = delete;

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~vram_nvidia_indexes() noexcept = default;

    /** Exchanges two indexes whole, every shard's keys with it. */
    void swap(vram_nvidia_indexes& other) noexcept {
        shards_.swap(other.shards_);
        std::swap(kernel_, other.kernel_);
        std::swap(geometry_, other.geometry_);
    }

    /** Bounds every shard at @p rows, clamped to the ID space; zero means the ID space. Past it, an
     *  insertion opens another shard generation on the same placements. Ignored once storage
     *  exists. */
    void bound_shards(std::size_t rows) noexcept { shards_.bound_shards(rows); }

    /** Rows one shard may hold. */
    std::size_t shard_capacity() const noexcept { return shards_.shard_capacity(); }

    /**
     *  Appends one shard built here on the device @p stream names, for callers with none to hand
     *  over.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The set is reserved, retired shards, or has a shard generation in
     *    flight.
     */
    status_t emplace(typename executor_t::stream_t const& stream) noexcept {
        std::unique_ptr<index_t> shard = build_(*stream.api, stream.device);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /**
     *  Appends one shard the caller already built, consuming it, with its keys inside it.
     *  @ref reserve closes the shard set.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p shard is null, walks by another kernel or keeps another
     *    geometry, or is already reserved; or the set is reserved, retired shards, or has a shard
     *    generation in flight.
     */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard || shard->kernel() != kernel_ || !(shard->geometry() == geometry_))
            return status_t::config_mismatch_k;
        int const device = shard->device_specs().ordinal;
        cuda_api_t const& api = shard->api();
        if (cuda_set_device(api, device) != cuda_success_k) return status_t::device_error_k;
        if (status_t const placed = shards_.emplace(std::move(shard)); failed(placed)) return placed;
        // Every shard is still unreserved, so each maps its slabs for every device placed so far.
        std::bitset<gpu_devices_capacity_k> const devices = devices_();
        for (shard_t& shard : shards_) shard.index->share_with_(devices);
        return status_t::success_k;
    }

    /**
     *  @brief Empties every shard in place, keeping its placement, geometry and capacity, and every
     *      shard generation past the first as idle shards the next ones reuse; a kept failure drops
     *      with the rows.
     *
     *  Takes the gate alone and closes every shard first; a search it overlaps is retried once.
     *
     *  @par Statuses:
     *  - @c contended_k A call in flight, a batch or load holding claims, or a plan holding the
     *    index closed.
     */
    status_t clear() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        if (status_t const closed = shards_.close(); failed(closed)) return closed;
        shards_.begin_window();
        clear_();
        shards_.end_window();
        shards_.reopen();
        return status_t::success_k;
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

    /** The epoch a cursor opens under; @ref still_valid tells whether a destructive window opened
     *  since. */
    u64_t read_epoch() const noexcept { return shards_.read_epoch(); }

    /** Whether no destructive window opened since @p epoch, so a cursor's reads still stand. */
    bool still_valid(u64_t epoch) const noexcept { return shards_.still_valid(epoch); }

    /** The geometry every shard keeps, fixed at construction - @ref emplace refused any other. */
    geometry_t geometry() const noexcept { return geometry_; }
    std::size_t dimensions() const noexcept { return geometry_.dimensions; }

    /** Connections per node, fixed with the geometry. */
    std::size_t connectivity() const noexcept { return geometry_.connectivity; }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.entries(); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** The first shard's kept failure, or success; asked of the shards, never cached. It is the
     *  failure an insertion refused with after rows launched on a shard, kept until @ref clear. */
    status_t status() const noexcept {
        for (shard_t const& shard : shards_)
            if (status_t const kept = shard.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }

    /** Bytes the index occupies - every shard's storage and its host and device keys; traversal
     *  scratch is the executor's. */
    std::size_t memory_usage() const noexcept { return sizeof(vram_nvidia_indexes) + shards_.memory_usage(); }

    /**
     *  Grows every shard ahead of @p config.capacity rows, split evenly over them, its keys with
     *  it - a hint, since all of them grow on demand.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c unknown_kind_k A shard's kits cannot score its metric.
     *  - @c config_mismatch_k A shard's neighbors live on another device, the dimensions are zero,
     *    the connectivity is not a power of two, @p config names another geometry, or no walk fits
     *    it.
     *  - @c contended_k A shard has a batch or load in flight.
     *  - @c overflow_k A shard's share passes its ID space or its reserved range.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return shards_.reserve(config); }

#pragma endregion Measures and Reservation

#pragma region Insertion

    /**
     *  @brief One batch in flight across the device shards.
     *
     *  The executor and every shard's claim over the rows it takes are held for the plan's whole
     *  life, one context per shard opened for the whole batch, and every shard's storage and keys
     *  grown to the rows it takes, so a step fails only on its devices.
     *
     *  Row @c t of the batch goes to shard @c (rotation+t) mod S. Steps run one after another on
     *  one host thread, each writing its rows' keys and launching the rows as deferred inserts;
     *  @ref finish settles them, files every shard's keys into its map, and turns every shard's
     *  watermark inside one commit, so a reader spanning the shards sees the batch whole or not at
     *  all. A failed launch or drain publishes nothing, and neither does a plan dying unfinished:
     *  every shard holding launched rows keeps the failure, so its rows stay invisible and later
     *  mutations are refused.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : traversals_(allocator), shares_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : traversals_(std::move(other.traversals_)), shares_(std::move(other.shares_)), config_(other.config_),
              batch_split_(other.batch_split_), result_(other.result_), status_(other.status_),
              streams_(std::move(other.streams_)), owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /** An unfinished plan drains what it launched, keeps @c cancelled_k on every shard holding
         *  it, and gives back every claim that launched nothing. */
        ~insert_plan_t() noexcept {
            if (owner_) [[maybe_unused]]
                status_t const abandoned = abandon_(status_t::cancelled_k);
        }

        /**
         *  Launches rows @b [first_row,first_row+rows) of the plan under @p keys on their shards;
         *  the rows and the keys stay live until @ref finish. Mismatched arguments launch nothing,
         *  any later refusal fails the plan.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or @ref status keeps it.
         *  - @c config_mismatch_k The @p keys are not one per row, the rows pass the batch or are
         *    not as wide as the index, a step overlaps rows already launched, or @ref status keeps
         *    it.
         *  - @c unsupported_k The rows' scalar has no device cast into the stored one, or
         *    @ref status keeps it.
         *  - @c unsupported_layout_k Rows sit closer than one packed row apart, or @ref status
         *    keeps it.
         *  - @c overflow_k A shard's rows pass its reserved range, or @ref status keeps it.
         *  - @c cancelled_k The index keeps it, as @ref status says.
         */
        status_t insert(std::size_t first_row, nk::vector_view<key_t> keys, rows_view_t rows) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            shared_lock_t const held {owner_->gate_};
            std::size_t const count = rows.rows;
            std::size_t const batch_entries = batch_split_.batch_entries;
            if (keys.size() != count || first_row > batch_entries || count > batch_entries - first_row ||
                (count && rows.columns != owner_->dimensions()))
                return status_t::config_mismatch_k;
            if (!count) return status_t::success_k;
            std::size_t const width = batch_split_.open_shard_generation.width;
            std::size_t const in_open = batch_split_.entries_in_open_shard_generation;
            std::size_t const per_fresh = batch_split_.entries_per_fresh_shard_generation;
            // Each shard takes one slab per window at most, so no launch stages more.
            std::size_t const window_rows = window_rows_k * width;
            for (std::size_t first = 0, taking = 0; first < count && succeeded(status_); first += taking) {
                // Entries below the open shard generation's share are split across its shards, the
                // rest across fresh ones in turn, and no window crosses from one shard generation
                // into the next.
                std::size_t const entry = first_row + first;
                std::size_t shard_generation = 0, entry_in_shard_generation = entry, left = in_open - entry;
                if (entry >= in_open) {
                    shard_generation = 1 + (entry - in_open) / per_fresh;
                    entry_in_shard_generation = (entry - in_open) % per_fresh;
                    left = per_fresh - entry_in_shard_generation;
                }
                taking = min_of(min_of(window_rows, count - first), left);
                std::size_t const rotation = (batch_split_.starting_shard + entry_in_shard_generation) % width;
                for (std::size_t offset = 0; offset != width && succeeded(status_); ++offset)
                    status_ = launch_(shard_generation * width + offset,
                                      shards_t::first_entry_of_shard(offset, width, rotation),
                                      keys[nk::range(first, first + taking)], rows.slice(first, taking));
            }
            return status_;
        }

        /**
         *  @brief Settles every launch, publishes every row at once, and reports the batch's
         *      telemetry. A plan short of rows refuses and stays whole, so the missing steps can
         *      still be copied.
         *
         *  Any other call spends the plan, and a failed launch or drain publishes nothing and
         *  keeps its failure on every shard holding launched rows.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent, or a step or @ref status keeps it.
         *  - @c config_mismatch_k The steps launched fewer rows than the batch holds, or a step or
         *    @ref status keeps it.
         *  - @c unsupported_k, @c unsupported_layout_k or @c overflow_k A step or @ref status keeps
         *    it.
         *  - @c cancelled_k The index keeps it, as @ref status says.
         */
        expected<insert_result_t> finish() noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            std::size_t launched = 0;
            for (share_t const& share : shares_) launched += share.launched;
            if (succeeded(status_) && launched != batch_split_.batch_entries)
                return {{owner_->size()}, status_t::config_mismatch_k};
            shared_lock_t const held {owner_->gate_};
            insert_result_t result = result_;
            status_t status = settle_(result);
            vram_nvidia_indexes& owner = *owner_;
            if (failed(status)) {
                result.new_size = owner.size();
                return {result, abandon_(status)};
            }

            // Each shard's claim makes its holder the one writer of its map, and the keys file
            // ahead of the commit, so the bracket stays as short as the watermark stores.
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (share_t const& share = shares_[shard]; share.launched)
                    owner.shards_[batch_split_.open_shard_generation.first + shard].index->publish_keys_(
                        {share.first_id, static_cast<node_id_t>(share.first_id + share.launched)});

            // Every shard's claim admits this batch alone, so no turn waits inside the commit.
            owner.shards_.begin_commit();
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (share_t const& share = shares_[shard]; share.launched)
                    if (status_t const published =
                            owner.shards_[batch_split_.open_shard_generation.first + shard].index->publish_(
                                share.first_id, share.launched);
                        failed(published) && succeeded(status))
                        status = published;
            owner.shards_.end_commit();
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

        /** Launches the step's rows the plan's shard @p shard takes, from @p first on in strides of
         *  the shard generation's width, under their keys. */
        status_t launch_(std::size_t shard, std::size_t first, nk::vector_view<key_t> keys, rows_view_t rows) noexcept {
            std::size_t const width = batch_split_.open_shard_generation.width, count = rows.rows;
            std::size_t const taking = shards_t::entries_from(count, first, width);
            share_t& share = shares_[shard];
            if (!taking) return status_t::success_k;
            // Overlapping steps would write keys past the cover the plan grew.
            if (share.launched + taking > share.rows) return status_t::config_mismatch_k;
            std::size_t const placed = batch_split_.open_shard_generation.first + shard;
            shard_t& slot = owner_->shards_[placed];
            traversal_context_t& traversal = traversals_[placed];
            if (status_t const entered = fanout_t::enter(owner_->shards_, placed); failed(entered)) return entered;
            node_id_t const first_node = static_cast<node_id_t>(share.first_id + share.launched);
            expected<insert_result_t> const inserted = slot.index->submit_insert_(
                keys[nk::range(first, count, width)],
                {rows.row(first), taking, rows.columns, rows.stride * width, rows.dtype}, first_node, {}, traversal,
                config_);
            if (!inserted) return inserted.status;
            share.launched += taking;
            result_ += inserted.value;
            return status_t::success_k;
        }

        /** Drains every context and adds what each spent to @p result, since the launches report
         *  their telemetry only once finished; answers the plan's status, or the first failed
         *  drain. */
        status_t settle_(insert_result_t& result) noexcept {
            status_t status = status_;
            for (traversal_context_t& traversal : traversals_)
                if (status_t const drained = traversal.wait(); failed(drained) && succeeded(status)) status = drained;
            // Every context was born zeroed for this plan alone, so its sink holds every launch on
            // its shard.
            for (traversal_context_t const& traversal : traversals_) {
                vram_telemetry_block_t const spent = traversal.spent();
                result.visited_members += spent.hops;
                result.computed_distances += spent.scored;
                result.restarts += spent.restarts;
                result.unlinked_neighbors += spent.lock_skips;
                result.tail_compactions += spent.tail_compactions;
                result.tail_dropped += spent.tail_dropped;
                result.visited_saturations += spent.visited_saturations;
            }
            return status;
        }

        /** Drains every context, keeps @p failure on every shard holding launched rows, and gives
         *  back every claim that launched none, spending the plan. */
        status_t abandon_(status_t failure) noexcept {
            for (traversal_context_t& traversal : traversals_) [[maybe_unused]]
                status_t const drained = traversal.wait();
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (shares_[shard].launched) [[maybe_unused]]
                    status_t const kept =
                        owner_->shards_[batch_split_.open_shard_generation.first + shard].index->fail_(failure);
                else if (shares_[shard].rows)
                    owner_->shards_[batch_split_.open_shard_generation.first + shard].index->unclaim_(
                        shares_[shard].first_id, shares_[shard].rows);
            owner_ = nullptr;
            return failure;
        }

        /** One context per shard of the index: the batch's launches stay in flight on it until
         *  @ref finish. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

        /** One part per shard from the open shard generation's first, in placement order. */
        growing_buffer<share_t, forkunion_allocator_t> shares_;

        /** The per-shard insertion configuration, applied uniformly. */
        insert_config_t config_ {};

        /** The batch split across the shard generation open at planning, which the parts start
         *  at, and the fresh ones it opened. */
        typename shards_t::batch_split_t batch_split_ {};

        /** The telemetry the launches reported as they returned, summed. */
        insert_result_t result_ {};

        /** The first refusal past a step's arguments, which leaves the whole batch unpublished. */
        status_t status_ = status_t::success_k;

        /** The executor's hold, kept for the plan's whole life, one caller at a time. */
        typename executor_t::claim_t streams_;

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once spent. */
        vram_nvidia_indexes* owner_ = nullptr;

        friend vram_nvidia_indexes;
    };

    /**
     *  @brief Opens a batch of @p rows: takes the executor, opens one context per shard, claims
     *      every shard's rows from its watermark, and grows every shard's storage and keys to
     *      them.
     *
     *  A batch crossing the bound tops the open shard generation up and opens as many fresh shard
     *  generations as its other rows fill, each shard built with its keys on the device its
     *  placement names, so it goes in whole; a racing batch taking the same room first leaves this
     *  one's claim refused with @c overflow_k. A shard with a batch, load, refine, absorb or save
     *  in flight refuses its claim, and the whole plan with @c contended_k.
     *
     *  @param[in] rows Rows in the batch; the plan's steps launch each exactly once, in any order.
     *  @param[in,out] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, the @p executor has no streams, or
     *    @ref status keeps it.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, a shard
     *    refuses to grow, as its @c reserve says, or @ref status keeps it.
     *  - @c contended_k The index is sealed, another call holds the @p executor, or a shard has a
     *    batch or load in flight or its claim word closed.
     *  - @c overflow_k A shard's rows pass its ceiling or its reserved range, or @ref status keeps
     *    it.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        shared_lock_t const held {gate_};
        if (shards_.sealed()) return {{}, status_t::contended_k};
        insert_plan_t plan {shards_.allocator()};
        plan.config_ = config;
        if (!rows) {
            plan.owner_ = this;
            return {std::move(plan), status_t::success_k};
        }
        // The executor's hold guards the scratch; the shards' claims keep other mutations off.
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);

        // The open shard generation takes what fits under the bound, fresh ones the rest, each
        // filled to it; rotating by entries routed keeps small batches from drifting onto shard
        // zero.
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            rows, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.batch_split_ = batch_split;
        shard_generation_t const open = batch_split.open_shard_generation;

        std::size_t const shards = batch_split.shards_spanned();
        if (failed(plan.shares_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::size_t const taking = batch_split.entries_for_shard(shard);
            node_id_t first_id = 0;
            if (taking) {
                auto const [claimed_id, shard_claim_status] = shards_[open.first + shard].index->claim_(taking);
                if (failed(shard_claim_status)) {
                    for (std::size_t earlier = 0; earlier != shard; ++earlier)
                        if (plan.shares_[earlier].rows)
                            shards_[open.first + earlier].index->unclaim_(plan.shares_[earlier].first_id,
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
            std::size_t const placed = open.first + shard;
            std::size_t const claimed_entries = plan.shares_[shard].first_id + plan.shares_[shard].rows;
            if (!plan.shares_[shard].rows) continue;
            if (status_t const entered = fanout_t::enter(shards_, placed); failed(entered)) return {{}, entered};
            // The claim makes the plan the shard's one grower, binding it on its first growth.
            if (status_t const grown = shards_[placed].index->map_nodes_(claimed_entries); failed(grown))
                return {{}, grown};
        }
        // Every shard is bound now, so the contexts size for what its walks hold.
        if (status_t const readied = fanout_t::reserve_traversals(
                shards_, plan.traversals_, executor, min_of(divide_round_up(rows, open.width), window_rows_k), config);
            failed(readied))
            return {{}, readied};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Inserts a batch under @p keys, splitting rows across the shards round-robin from a
     *      rotating start, so they fill evenly and each sees the whole distribution:
     *      @ref plan_insert, one step over every row, and its @c finish.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row went to is
     *  the index's business. The keys are what the caller gets its rows back by.
     *
     *  A shard failing past its claim keeps that failure; a shard with a batch, load, refine,
     *  absorb or save in flight refuses the call as @c contended_k.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_rows The batch, one row per entry, in any scalar NumKong casts, imported on
     *      each shard's device.
     *  @param[in,out] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p keys are not one per row, the rows are not as wide as the
     *    index, the @p executor drives no stream on some shard's device, a shard refuses to grow,
     *    or @ref status keeps it.
     *  - @c uninitialized_k The index has no shards, the @p executor has no streams, or
     *    @ref status keeps it.
     *  - @c contended_k The index is sealed, another call holds the @p executor, or a shard has a
     *    batch or load in flight or its claim word closed.
     *  - @c overflow_k A shard's rows pass its ceiling or its reserved range, or @ref status keeps
     *    it.
     *  - @c unsupported_k The rows' scalar has no device cast into the stored one, or @ref status
     *    keeps it.
     *  - @c unsupported_layout_k Rows sit closer than one packed row apart, or @ref status keeps
     *    it.
     *  - @c cancelled_k The index keeps it, as @ref status says.
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

    /**
     *  Inserts one row under @p key into whichever shard the rotation names next - the one-row
     *  sibling of the batched form, priced identically: one launch on one device.
     *
     *  @par Statuses:
     *  - @c unsupported_layout_k The @p new_vector is not contiguous, or @ref status keeps it.
     *  - @c config_mismatch_k The row is not as wide as the index, the @p executor drives no
     *    stream on some shard's device, a shard refuses to grow, or @ref status keeps it.
     *  - @c uninitialized_k The index has no shards, the @p executor has no streams, or
     *    @ref status keeps it.
     *  - @c contended_k The index is sealed, another call holds the @p executor, or a shard has a
     *    batch or load in flight or its claim word closed.
     *  - @c overflow_k A shard's rows pass its ceiling or its reserved range, or @ref status keeps
     *    it.
     *  - @c unsupported_k or @c cancelled_k The index keeps it, as @ref status says.
     */
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
     *  devices work side by side. An absorb's plan first appends its donors, one after another, and
     *  only then opens the shards' passes over the parts they appended; a split's opens every
     *  shard's split into the receivers' shards at planning and keeps the receivers closed too. The
     *  executor's streams must outlive the plan.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb ever fills it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : arenas_(allocator), traversals_(allocator), shard_plans_(allocator), part_bases_(allocator),
              part_offsets_(allocator), moves_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : arenas_(std::move(other.arenas_)), traversals_(std::move(other.traversals_)),
              shard_plans_(std::move(other.shard_plans_)), part_bases_(std::move(other.part_bases_)),
              part_offsets_(std::move(other.part_offsets_)), moves_(std::move(other.moves_)),
              donors_(std::exchange(other.donors_, {})), receivers_(std::exchange(other.receivers_, {})),
              absorbed_donors_(other.absorbed_donors_), config_(other.config_), population_(other.population_),
              owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** An unfinished plan heals nothing: it reopens, and the graph is as the last phase left
         *  it. */
        ~refine_plan_t() noexcept {
            if (!owner_) return;
            shard_plans_.clear();
            release_();
        }

        /**
         *  @brief Drives the pending phases on @p executor until the pass ends or @p on_visited
         *      cancels; @c cancelled_k means a later call continues where this one stopped.
         *
         *  Appends the donors still pending first, as @ref plan_absorb describes; a failing one
         *  stays pending and closed, with every later one, for the next call, and an appended one
         *  stays empty, closed and sealed. The executor is held only for the call.
         *
         *  @param[in,out] executor The executor the plan was opened over.
         *  @param[in] on_visited Fires per re-selecting phase with the nodes visited so far and the
         *      population; a @c cancel_k answer stops before the next phase.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or finished, or @ref status keeps it.
         *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, a
         *    donor's nodes are never copied, or @ref status keeps it.
         *  - @c contended_k Another call holds the @p executor or a donor's gate, or a shard has a
         *    batch or load in flight.
         *  - @c overflow_k The appended rows pass a shard's ceiling or reserved range, or @ref
         *    status keeps it.
         *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired)) return paired;
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; absorbed_donors_ != donors_.size(); ++absorbed_donors_) {
                vram_nvidia_indexes*& donor = donors_[absorbed_donors_];
                if (status_t const absorbed = owner_->absorb_(*donor, moves_of_(), executor, *this); failed(absorbed))
                    return absorbed;
                donor = nullptr;
            }
            if (shard_plans_.empty())
                if (status_t const opened = open_shards_(); failed(opened)) return opened;
            return drive_(on_visited);
        }

        /**
         *  @brief Stops any re-selection, audits and heals, reopens the claim words, and reports. A
         *      plan whose donors have not all been appended heals nothing and reports nothing.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or finished, or a pass found @ref status keeping
         *    it.
         *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, or a
         *    pass found @ref status keeping it.
         *  - @c contended_k Another call holds the @p executor.
         *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k
         *    A pass found the index keeping it, as @ref status says.
         */
        expected<refine_report_t> finish(executor_t& executor) noexcept {
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
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                    if (status_t const kept = shard_plans_[shard].finish(traversals_[shard]).status;
                        failed(kept) && succeeded(healed))
                        healed = kept;
            }
            refine_report_t const result = running();
            shard_plans_.clear();
            release_();
            return {result, healed};
        }

        /** The totals so far, summed over the shards' passes from the counters and telemetry the
         *  last phase finished: what @ref finish reports once it healed, and zeros while donors are
         *  pending. */
        refine_report_t running() const noexcept {
            refine_report_t result;
            for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                result += shard_plans_[shard].finish(traversals_[shard]).value;
            return result;
        }

      private:
        friend vram_nvidia_indexes;

        /** Reopens this index, every donor still pending and every receiver; the plan is unbound
         *  after. A split that emptied this index drops its routing inside a window. */
        void release_() noexcept {
            vram_nvidia_indexes& owner = *std::exchange(owner_, nullptr);
            if (!receivers_.empty() && !owner.size()) {
                owner.shards_.begin_window();
                owner.shards_.forget_entries();
                owner.shards_.end_window();
            }
            owner.shards_.reopen();
            for (vram_nvidia_indexes* pending : donors_.subspan(absorbed_donors_)) pending->shards_.reopen();
            for (vram_nvidia_indexes* receiver : receivers_) receiver->shards_.reopen();
            donors_ = {}, receivers_ = {};
        }

        /** The moves of the first pending donor: donors are appended in order, so the shards of
         *  every later one close the moves. */
        std::span<shard_move_t const> moves_of_() const noexcept {
            std::size_t later = 0;
            for (std::size_t donor = absorbed_donors_ + 1; donor != donors_.size(); ++donor)
                later += donors_[donor]->shards_.size();
            std::size_t const count = donors_[absorbed_donors_]->shards_.size();
            return to_span(moves_).subspan(moves_.size() - later - count, count);
        }

        /** Opens every shard's pass over its parts on its own context, once no donor is pending. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != traversals_.size(); ++shard) {
                std::span<node_id_t const> parts;
                // A refine re-selects every row; an absorb only the rows appended after the
                // resident part.
                iota_view<node_id_t> nodes {0, node_id_unknown_k};
                if (!part_offsets_.empty()) {
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
                    nodes.first = parts.size() > 1 ? parts[1] : node_id_unknown_k;
                }
                if (status_t const entered = fanout_t::enter(owner_->shards_, shard); failed(entered))
                    return shard_plans_.clear(), entered;
                auto [plan, status] = owner_->shards_[shard].index->plan_refine(nodes, parts, config_,
                                                                                traversals_[shard]);
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
                    if (succeeded(fanout_t::enter(owner_->shards_, shard))) [[maybe_unused]]
                        status_t const queued = shard_plans_[shard].queue(traversals_[shard], config_);
                }
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                    if (status_t const waited = shard_plans_[shard].wait(traversals_[shard]);
                        failed(waited) && succeeded(outcome))
                        outcome = waited;
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

        /** One run per shard on its stream, holding the device scratch in flight across phases. */
        growing_buffer<typename executor_t::arena_t, forkunion_allocator_t> arenas_;

        /** One walk context per shard over its run, each pass queued on it across phases. */
        growing_buffer<traversal_context_t, forkunion_allocator_t> traversals_;

        /** Every shard's own pass in placement order; none until the donors were appended. */
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** The parts the moves appended, ascending per shard, @c node_id_unknown_k past the
         *  last. */
        growing_buffer<node_id_t, forkunion_allocator_t> part_bases_;

        /** Where each shard's run of @c part_bases_ starts, and one past the last; empty with no
         *  donors. */
        growing_buffer<std::size_t, forkunion_allocator_t> part_offsets_;

        /** One move per donor shard, in move order, as @ref plan_absorb packed them; empty
         *  with no donors. */
        growing_buffer<shard_move_t, forkunion_allocator_t> moves_;

        /** The caller's donor slots, each closed at planning and nulled once appended. */
        std::span<vram_nvidia_indexes*> donors_;

        /** A split's receivers, closed while the plan lives; empty for a refine or an absorb. */
        std::span<vram_nvidia_indexes*> receivers_;

        /** Donors appended so far. */
        std::size_t absorbed_donors_ = 0;

        /** The passes' knobs. */
        refine_config_t config_ {};

        /** The population @c on_visited reports: the rows published once donors were appended. */
        std::size_t population_ = 0;

        /** The index refined, closed while the plan lives; null in an empty plan and once
         *  finished. */
        vram_nvidia_indexes* owner_ = nullptr;
    };

    /**
     *  Opens a refine of every shard: closes the claim words and builds everything the plan runs
     *  on, as @ref plan_absorb does with no donors, then opens the shards' passes.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or was never reserved, or @ref status keeps
     *    it.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, or
     *    @ref status keeps it.
     *  - @c contended_k A call, batch, load or save in flight, or another refine or absorb holding
     *    the index.
     *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index
     *    keeps it, as @ref status says.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, executor_t& executor) noexcept {
        return plan_absorb({}, config, executor);
    }

    /**
     *  @brief Re-selects the neighbors of every stored node and heals the islands that leaves:
     *      @ref plan_refine, then @c run and @c finish.
     *
     *  A stopped pass leaves the graph correct and merely less improved. Every shard's claim word
     *  stays closed for the pass, so a batch, load or save in flight refuses it with
     *  @c contended_k, and one arriving meanwhile is refused.
     *
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] executor The streams and their scratch.
     *  @param[in] on_visited Fires per re-selecting phase with the nodes visited so far and the
     *      population; a @c cancel_k answer skips the rounds left, still heals, and reports
     *      @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or was never reserved, or @ref status keeps
     *    it.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, or
     *    @ref status keeps it.
     *  - @c contended_k A call, batch, load or save in flight, another refine or absorb holding
     *    the index, or another call holding the @p executor.
     *  - @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index keeps it, as @ref
     *    status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> refine(refine_config_t config, executor_t& executor,
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_refine(config, executor);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every row of each into
     *      this index, one donor after another: donor shards, largest first, each append to the
     *      shard with the fewest rows that keeps it under @ref shard_capacity, one on its own
     *      device on ties, with its edges rebased and its keys beside it, as a part of its own;
     *      when none has room, a shard generation opens on the same devices.
     *
     *  Each donor is appended through one plan that does every fallible thing first: the donor's
     *  gate, and a load claiming all the rows each receiving shard takes. Every donor's committed
     *  storage maps for this index's devices at planning, so a donor shard then crosses device to
     *  device, one column at a time on the receiving shard's stream, its keys beside them; every
     *  part goes before any shard publishes, and each shard's load files its keys just before its
     *  watermark turns. The donor then forgets its rows in place inside its window, keeping its
     *  shards and their storage, its slot nulls, it stays closed and sealed, consumed, and its
     *  shards go among the receiving shards' parts in the plan, so at most one donor's nodes are
     *  ever held twice.
     *
     *  Once the last donor was appended, the plan's re-selection walks every shard across the parts
     *  the moves appended, round by round; a cancel pauses only that walk. This index and every
     *  donor close here, each under its own gate, so the packing never goes stale: this index stays
     *  closed while the plan lives and each donor until it is appended, a batch, load or save in
     *  flight on any of them refuses the plan with @c contended_k, and an insertion into any of
     *  them arriving meanwhile is refused. Searches of this index may not overlap the plan's
     *  re-selection, whose merge rewrites neighbors in place; searches of a donor run throughout,
     *  retrying once around the window that empties it.
     *
     *  @param[in,out] donors The indexes giving their rows up, each nulled once consumed; the slots
     *      must outlive the moves.
     *  @param[in] config The refine's knobs; zeros mean "derive from connectivity".
     *  @param[in,out] executor The streams every @c run and @c finish names, driving every device
     *      of every index.
     *
     *  @par Statuses:
     *  - @c uninitialized_k This index has no shards, or it or a donor keeps it, as its
     *    @ref status says.
     *  - @c config_mismatch_k A null donor, another metric or geometry, a device the executor
     *    does not drive, a donor device one of this index's devices cannot reach, this index
     *    itself, or a geometry or walk context a shard refuses, as its @c reserve and
     *    @c plan_refine say; or this index or a donor keeps it.
     *  - @c contended_k A call, batch, load or save in flight on this index or a donor, a
     *    repeated donor, or another refine or absorb holding either.
     *  - @c unsupported_k A donor shard past @ref shard_capacity, which would have to be cut, or
     *    this index or a donor keeps it.
     *  - @c overflow_k A fresh shard generation still has no room for a donor shard, or this
     *    index or a donor keeps it.
     *  - @c cancelled_k or @c unsupported_layout_k This index or a donor keeps it,
     *    as its @ref status says.
     */
    expected<refine_plan_t> plan_absorb(std::span<vram_nvidia_indexes*> donors, refine_config_t config,
                                        executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        for (vram_nvidia_indexes* donor : donors) {
            if (!donor || donor == this || donor->kernel() != kernel() || !(donor->geometry() == geometry()))
                return {{}, status_t::config_mismatch_k};
            if (status_t const kept = donor->status(); failed(kept)) return {{}, kept};
            if (status_t const paired = fanout_t::pairs_with(donor->shards_, executor); failed(paired))
                return {{}, paired};
        }
        // The walks size for the first index whose shards grew, this one or a donor.
        vram_nvidia_indexes const* shaped = this;
        for (vram_nvidia_indexes const* donor : donors)
            if (!shaped->capacity()) shaped = donor;
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        if (status_t const closed = close_gated_(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.config_ = config;
        for (std::size_t closing = 0; closing != donors.size(); ++closing)
            if (status_t const closed = donors[closing]->close_gated_(); failed(closed)) {
                while (closing) donors[--closing]->shards_.reopen();
                return {{}, closed};
            }
        plan.donors_ = donors;
        // A closed donor commits no slab meanwhile, so this index's devices add to its mapped ones.
        std::bitset<gpu_devices_capacity_k> const devices = devices_();
        for (vram_nvidia_indexes* donor : donors)
            for (shard_t& shard : donor->shards_)
                if (status_t const granted = shard.index->grant_(devices); failed(granted)) return {{}, granted};
        if (!donors.empty()) {
            std::size_t sources = 0;
            for (vram_nvidia_indexes const* donor : donors) sources += donor->shards_.size();
            if (failed(plan.moves_.resize(sources))) return {{}, status_t::bad_alloc_k};
            auto source_shard = [&](std::size_t from) noexcept -> shard_t const& {
                std::size_t donor = 0;
                for (; from >= donors[donor]->shards_.size(); ++donor) from -= donors[donor]->shards_.size();
                return donors[donor]->shards_[from];
            };
            auto device_of = [](shard_t const& shard) noexcept { return shard.index->device_specs().ordinal; };
            auto open = [&]() noexcept {
                // This index is sealed, so the shard generation opens closed like the rest, past
                // the seal.
                return shards_.open_shard_generation_past_seal_(
                    shards_.open_shard_generation(),
                    [this](shard_t const& built) noexcept { return build_shard_(built); });
            };
            if (status_t const packed = shards_.pack(to_writable_span(plan.moves_), 0, shards_.shard_capacity(),
                                                     source_shard, device_of, rank_fewest_entries, open);
                failed(packed))
                return {{}, packed};
            if (status_t const sized = shards_t::size_parts(to_span(plan.moves_), shards_.size(), plan.part_offsets_,
                                                            plan.part_bases_);
                failed(sized))
                return {{}, sized};
        }
        // The moves may have opened a shard generation, whose shards take part like the others.
        std::size_t const planned_shards = donors.empty() ? shards : plan.part_offsets_.size() - 1;
        if (failed(plan.shard_plans_.reserve_exactly(planned_shards))) return {{}, status_t::bad_alloc_k};
        if (status_t const opened = open_walks_(plan, executor, *shaped, planned_shards, donors.empty() ? 1u : 2u, 0);
            failed(opened))
            return {{}, opened};
        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Moves every row of each of @p donors into this index and links them in:
     *      @ref plan_absorb, then @c run and @c finish. @p on_visited answers as in @ref refine,
     *      for the re-selection alone; a cancel still heals and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k This index has no shards, or it or a donor keeps it, as its
     *    @ref status says.
     *  - @c config_mismatch_k A null donor, another metric or geometry, a device the @p executor
     *    does not drive, a donor device this index cannot reach, this index itself, a geometry or
     *    walk context a shard refuses, donor rows never copied, or this index or a donor keeps it.
     *  - @c contended_k A call, batch, load or save in flight on this index or a donor, a
     *    repeated donor, another refine or absorb holding either, or another call holding the
     *    @p executor.
     *  - @c unsupported_k A donor shard past @ref shard_capacity, or this index or a donor keeps
     *    it.
     *  - @c overflow_k A fresh shard generation still has no room for a donor shard, the appended
     *    rows pass a shard's ceiling or reserved range, or this index or a donor keeps it.
     *  - @c unsupported_layout_k This index or a donor keeps it, as its @ref status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> absorb(std::span<vram_nvidia_indexes*> donors, refine_config_t config,
                                     executor_t& executor, progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_absorb(donors, config, executor);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /**
     *  @brief Moves every row of @p donor into this index, as the many-donor @ref absorb does with
     *      a single donor.
     *
     *  @par Statuses:
     *  - @c uninitialized_k This index has no shards, or it or @p donor keeps it, as its
     *    @ref status says.
     *  - @c config_mismatch_k The @p donor is this index, has another metric or geometry, or sits
     *    on a device the @p executor does not drive or this index cannot reach; a shard refuses its
     *    geometry or walk context, its nodes are never copied, or this index or @p donor keeps it.
     *  - @c contended_k A call, batch, load or save in flight on this index or @p donor, another
     *    refine or absorb holding either, or another call holding the @p executor.
     *  - @c unsupported_k A shard of @p donor past @ref shard_capacity, or this index or
     *    @p donor keeps it.
     *  - @c overflow_k A fresh shard generation still has no room for a shard of @p donor, the
     *    appended rows pass a shard's ceiling or reserved range, or this index or @p donor keeps
     *    it.
     *  - @c unsupported_layout_k This index or @p donor keeps it, as its @ref status says.
     */
    expected<refine_report_t> absorb(vram_nvidia_indexes& donor, refine_config_t config,
                                     executor_t& executor) noexcept {
        vram_nvidia_indexes* donors[1] = {&donor};
        return absorb(std::span<vram_nvidia_indexes*>(donors), config, executor);
    }

    /**
     *  @brief Opens a split of this index into @p receivers, the inverse of @ref plan_absorb: each
     *      shard partitions its own graph breadth-first into one part per receiver, and part @c r
     *      of shard @c s goes to shard @c s of receiver @c r, on the same device.
     *
     *  Every shard's split opens at planning, queuing its partition, as the shard's @c plan_split
     *  describes; the plan's phases then scatter every shard's nodes, keys included, publish the
     *  receivers' shards, and re-select, audit and heal each of them over its picked nodes. This
     *  index and every receiver close here, each under its own gate, and stay closed while the plan
     *  lives; nothing may search this index meanwhile, and once the scatter finished it is empty,
     *  keeping its shards and their storage.
     *
     *  @param[in,out] receivers The empty indexes taking the parts, at least two, each with as
     *      many shards as this one, shard by shard on the same devices.
     *  @param[in] config The refinement knobs, zeros meaning "derive from connectivity", and the
     *      seed picking the parts' roots, so equal seeds over equal graphs split alike.
     *  @param[in,out] executor The streams every @c run and @c finish names.
     *
     *  @par Statuses:
     *  - @c uninitialized_k This index has no shards, or it or a receiver keeps it, as its
     *    @ref status says.
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, another metric, geometry or shard count, a shard on another device, a
     *    device the @p executor does not drive, or a walk context a shard refuses.
     *  - @c contended_k A call, batch, load or save in flight on this index or a receiver, or
     *    another refine or absorb holding either.
     *  - @c overflow_k A shard's part slots pass the 32-bit IDs.
     */
    expected<refine_plan_t> plan_split(std::span<vram_nvidia_indexes*> receivers, split_config_t config,
                                       executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (receivers.size() < 2) return {{}, status_t::config_mismatch_k};
        for (std::size_t receiver = 0; receiver != receivers.size(); ++receiver) {
            vram_nvidia_indexes* const into = receivers[receiver];
            if (!into || into == this || into->kernel() != kernel() || !(into->geometry() == geometry()) ||
                into->shards_.size() != shards || into->size())
                return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == into) return {{}, status_t::config_mismatch_k};
            for (std::size_t shard = 0; shard != shards; ++shard)
                if (into->shards_[shard].index->device_specs().ordinal != shards_[shard].index->device_specs().ordinal)
                    return {{}, status_t::config_mismatch_k};
            if (status_t const kept = into->status(); failed(kept)) return {{}, kept};
        }
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
        if (status_t const closed = close_gated_(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.config_ = config.refine;
        for (std::size_t closing = 0; closing != receivers.size(); ++closing)
            if (status_t const closed = receivers[closing]->close_gated_(); failed(closed)) {
                while (closing) receivers[--closing]->shards_.reopen();
                return {{}, closed};
            }
        plan.receivers_ = receivers;
        growing_buffer<index_t*, forkunion_allocator_t> parts {shards_.allocator()};
        if (failed(parts.resize(receivers.size())) || failed(plan.shard_plans_.reserve_exactly(shards)))
            return {{}, status_t::bad_alloc_k};
        if (status_t const opened = open_walks_(plan, executor, *this, shards, 1, receivers.size()); failed(opened))
            return {{}, opened};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            for (std::size_t receiver = 0; receiver != receivers.size(); ++receiver)
                parts[receiver] = receivers[receiver]->shards_[shard].index.get();
            if (status_t const entered = fanout_t::enter(shards_, shard); failed(entered)) return {{}, entered};
            auto [shard_plan, status] = shards_[shard].index->plan_split(to_writable_span(parts), config,
                                                                         plan.traversals_[shard]);
            if (failed(status)) return {{}, status};
            plan.shard_plans_.push_back_reserved(std::move(shard_plan));
        }
        plan.population_ = size();
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Moves every row of this index into one graph-local part per receiver and links
     *      each in: @ref plan_split, then @c run and @c finish. @p on_visited answers as in @ref
     *      refine, for the re-selection alone; a cancel still heals and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k, @c config_mismatch_k, @c contended_k or @c overflow_k As
     *    @ref plan_split refuses, or a pass found a shard keeping it.
     *  - @c cancelled_k @p on_visited cancelled; the receivers still healed.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> split(std::span<vram_nvidia_indexes*> receivers, split_config_t config,
                                    executor_t& executor, progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_split(receivers, config, executor);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /**
     *  @brief Measures how much of every shard a search can reach, every shard's batch of sweeps
     *      queued on its own device before any waits: reachable nodes summed, depth the deepest
     *      shard's.
     *
     *  Read-only, and bracketed by the epoch as every reader is.
     *
     *  @param[in,out] executor The streams and their scratch, one per shard.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards, the @p executor has no streams, or
     *    @ref status keeps it.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, or
     *    @ref status keeps it.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index
     *    keeps it, as @ref status says.
     */
    expected<reachability_t> health(executor_t& executor) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<reachability_t> {
            std::size_t const shards = shards_.size();
            if (!shards) return {{}, status_t::uninitialized_k};
            if (status_t const kept = status(); failed(kept)) return {{}, kept};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};

            growing_buffer<traversal_context_t, forkunion_allocator_t> traversals {shards_.allocator()};
            if (status_t const ready = fanout_t::reserve_traversals(shards_, traversals, executor, 1,
                                                                    insert_config_t {});
                failed(ready))
                return {{}, ready};

            using health_plan_t = typename index_t::health_plan_t;
            growing_buffer<health_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
            if (failed(plans.reserve_exactly(traversals.size()))) return {{}, status_t::bad_alloc_k};
            for (std::size_t shard = 0; shard != traversals.size(); ++shard) {
                if (status_t const entered = fanout_t::enter(shards_, shard); failed(entered)) return {{}, entered};
                auto [plan, status] = shards_[shard].index->plan_health(traversals[shard]);
                if (failed(status)) return {{}, status};
                plans.push_back_reserved(std::move(plan));
            }
            // Every shard's batch queues before any waits, so the devices sweep at once.
            bool running = true;
            while (running) {
                running = false;
                for (std::size_t shard = 0; shard != plans.size(); ++shard) {
                    if (!plans[shard].running()) continue;
                    if (status_t const entered = fanout_t::enter(shards_, shard); failed(entered)) return {{}, entered};
                    if (status_t const queued = plans[shard].queue(traversals[shard]); failed(queued))
                        return {{}, queued};
                }
                for (std::size_t shard = 0; shard != plans.size(); ++shard) {
                    if (status_t const waited = plans[shard].wait(traversals[shard]); failed(waited))
                        return {{}, waited};
                    running |= plans[shard].running();
                }
            }
            reachability_t result;
            for (health_plan_t const& plan : plans) {
                auto const [shard_health, status] = plan.finish();
                if (failed(status)) return {{}, status};
                result += shard_health;
            }
            return {result, status_t::success_k};
        });
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
     *  @param[in,out] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' key columns, so colliding ID spaces still merge into one ranking.
     *  @param[in,out] distances Distances of the same ranking, parallel to @p keys.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in,out] executor The streams and their scratch, paired positionally with the shards.
     *
     *  The batched search over one row, so every shard walks at once and the rankings merge as they
     *  finish; each shard keeps as many matches as the ranking is wide, walking as broad as its
     *  expansion factor, or broader where the shard raises it to cover that width, as the batched
     *  search does. A failing shard keeps whatever the others already merged, so the reported count
     *  and the ranking always agree.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c unsupported_layout_k The @p query_vector is not contiguous.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, or the
     *    query is not as wide as the index.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     */
    expected<search_result_t> search(vector_view_t query_vector, std::span<key_t> keys, std::span<distance_t> distances,
                                     search_config_t config, executor_t& executor) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            if (shards_.empty()) return {{}, status_t::uninitialized_k};
            if (!query_vector.is_contiguous()) return {{}, status_t::unsupported_layout_k};
            std::size_t const width = min_of(keys.size(), distances.size());
            if (!width) return {{}, fanout_t::pairs_with(shards_, executor)};
            config.expansion_factor = static_cast<typename index_t::candidates_count_t>(
                shards_[0].index->non_zero_expansion_factor(config));
            config.count = static_cast<typename index_t::candidates_count_t>(width);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(1, width, 1, config); failed(begun)) return {{}, begun};
            std::size_t count = 0;
            expected<search_result_t> found = search_window_(
                fanout, vectors_view_t {query_vector.data(), 1u, query_vector.size()}, config,
                nk::matrix_span<key_t> {keys.data(), 1u, width},
                nk::matrix_span<distance_t> {distances.data(), 1u, width}, std::span<std::size_t> {&count, 1u});
            found.value.count = count;
            return found;
        });
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard and gathers the rankings into one
     *      per query.
     *
     *  The executor-driven sibling of @c search, one launch per shard and window rather than one
     *  thread per query: the queries cross @c window_rows_k at a time, every shard walks each
     *  window, copies its matches into its own columns of one host staging, and the copied columns
     *  merge query by query through the key columns.
     *
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts, imported on
     *      each shard's device.
     *  @param[out] matches Merged rankings, one query per row, as many matches as its width: the
     *      keys, their distances in any score type, and per-query merged hit counts.
     *  @param[in] config The per-shard search configuration, applied uniformly; the width of
     *      @p matches sets its @c count.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
     *  @return The merged hits summed over queries, and the telemetry summed over shards.
     *
     *  A refused shard stops the fan-out, and its window merges nothing while every earlier window
     *  stays merged, so the rankings and the counts always agree; every queued stream is awaited
     *  before returning, since the caller's queries and the call's own staging are DMA targets
     *  until then. Rankings scored in f32 or f64 that all live on the first shard's device may
     *  merge there instead, into the same rankings, and are written only once every shard finished.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p matches
     *    holds other than one count per query, or its row counts or widths disagree, or the queries
     *    are not as wide as the index.
     *  - @c unsupported_layout_k A row of @p matches is not contiguous, or query rows sit closer
     *    than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped the
     *    read twice.
     *  - @c unsupported_k The queries' scalar has no device cast into the stored one.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                     search_config_t config, executor_t& executor) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            if (status_t const refusal = fanout_t::refusal(shards_, executor, queries.rows, matches.keys,
                                                           matches.scores, matches.counts);
                failed(refusal) || !queries.rows)
                return {{}, refusal};
            config.count = static_cast<typename index_t::candidates_count_t>(matches.keys.extent(1));
            std::size_t const widest = min_of(queries.rows, window_rows_k);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(widest, config.count, widest, config); failed(begun))
                return {{}, begun};
            auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
                nk::range_t const rows = nk::range(first, first + taking);
                return search_window_(fanout, queries.slice(first, taking), config, matches.keys(rows, nk::slice),
                                      matches.scores(rows, nk::slice), matches.counts.subspan(first, taking));
            };
            return for_each_window<search_result_t>(queries.rows, window_rows_k, search_window);
        });
    }

    /**
     *  @brief The batched @c search ranked by a lightning indexer: every shard walks the queries
     *      under the same @p heads, passed to it as is, so whichever shard scored them, the
     *      merged distances are: <tt>1 − Σₕ wₕ · ReLU(qₕ · k)</tt>.
     *
     *  A dot verb.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the @p executor drives
     *    no stream on some shard's device, or the queries are not as wide as the index.
     *  - @c config_mismatch_k The @p matches hold other than one count per query, their row counts
     *    or widths disagree, or the heads or weights disagree with the queries.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c unsupported_layout_k A row of @p matches is not contiguous, or query or
     *    head rows sit closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k A scan is asked, or the queries' or heads' scalar has no device cast
     *    into the stored one.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     ranked_matches_span<key_t, distance_t> matches, search_config_t config,
                                     executor_t& executor) noexcept
        requires floating_scalar<scalar_t>
    {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = fanout_t::refusal(shards_, executor, queries.rows, matches.keys,
                                                           matches.scores, matches.counts);
                failed(refusal) || !queries.rows)
                return {{}, refusal};
            std::size_t const heads_count = heads.weights.extent(1);
            if (heads.weights.extent(0) != queries.rows || heads.heads.extent(0) != queries.rows * heads_count)
                return {{}, status_t::config_mismatch_k};
            config.count = static_cast<typename index_t::candidates_count_t>(matches.keys.extent(1));
            std::size_t const widest = min_of(queries.rows, window_rows_k);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(widest, config.count, widest, config, heads); failed(begun))
                return {{}, begun};
            auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
                nk::range_t const rows = nk::range(first, first + taking);
                indexer_heads<scalar_t> const window_heads {
                    heads.heads(nk::range(first * heads_count, (first + taking) * heads_count), nk::slice),
                    heads.weights(rows, nk::slice)};
                return search_window_(fanout, queries.slice(first, taking), config, matches.keys(rows, nk::slice),
                                      matches.scores(rows, nk::slice), matches.counts.subspan(first, taking),
                                      window_heads);
            };
            return for_each_window<search_result_t>(queries.rows, window_rows_k, search_window);
        });
    }

    /**
     *  @brief Graph products packed on the first shard's device: every row keeps up to
     *      @c config.count of the pairs its walks scored, the first shards' pairs first, keys
     *      resolved through each shard's device key column, back to back in @p tape.
     *
     *  @param[in] left The left-hand rows, one walk per row, as in @c search.
     *  @param[out] products Row q's pairs from @c offsets[q]: keys, then products in
     *      @p product_type_, the dot's own type or @c f64_t, up to the tape's stride per row; the
     *      offsets hold one more than the rows, the first the base the caller zeroed. All on the
     *      first shard's device, where every window appends from its first offset.
     *  @param[out] scored The pairs each row's walks scored across every shard, above the stride
     *      when truncated; host memory, copied to the host with each window.
     *  @param[in] config The per-shard walk configuration.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
     *  @return The pairs scored, summed over rows on the device, and the telemetry over shards.
     *
     *  The rows cross @c window_rows_k at a time, each window appending to the tape where the last
     *  ended, with one sleeping wait per window and only the 8 B sum and the scored counts
     *  crossing to the host. A refused shard stops the fan-out, and its window appends nothing.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the @p executor drives
     *    no stream on some shard's device, the tape is not on the first shard's device, @p scored
     *    or the tape disagree with the rows, or the rows are not as wide as the index.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c unsupported_layout_k Rows sit closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k A scan is asked, or the rows' scalar has no device cast into the stored
     *    one.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, products_tape<key_t, product_type_> products,
                                       std::span<std::size_t> scored, search_config_t config,
                                       executor_t& executor) noexcept {
        static_assert(std::is_same_v<product_type_, dot_result_t> || std::is_same_v<product_type_, f64_t>,
                      "Products move at the dot's own width or widen to f64");
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            std::size_t const rows_count = left.rows, stride = products.stride;
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
            if (status_t const refusal = products.refusal(rows_count, scored); failed(refusal)) return {{}, refusal};
            if (!rows_count) return {{}, status_t::success_k};
            if (left.columns != dimensions()) return {{}, status_t::config_mismatch_k};
            int const first_device = shards_[0].index->device_specs().ordinal;
            cuda_api_t const& api = shards_[0].index->api();
            if (cuda_device_of(api, products.offsets.data()) != first_device ||
                (stride && (cuda_device_of(api, products.keys.data()) != first_device ||
                            cuda_device_of(api, products.scores.data()) != first_device)))
                return {{}, status_t::config_mismatch_k};
            config.count = static_cast<typename index_t::candidates_count_t>(stride);
            std::size_t const widest = min_of(rows_count, window_rows_k);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(0, 0, widest, config); failed(begun)) return {{}, begun};
            auto multiply_window = [&](std::size_t first, std::size_t taking) noexcept {
                products_tape<key_t, product_type_> const window {products.keys, products.scores,
                                                                  products.offsets.subspan(first, taking + 1), stride};
                return multiply_window_(fanout, left.slice(first, taking), window, scored.subspan(first, taking),
                                        config);
            };
            return for_each_window<search_result_t>(rows_count, window_rows_k, multiply_window);
        });
    }

    /**
     *  @brief Exact late-interaction scores on the devices: every token row ranks its candidates,
     *      the keys any row of a query ranked become that query's nominees, and each nominee's
     *      score sums, over the query's rows, its best dot among @b all of its entries.
     *
     *  @param[in] queries The token rows of every query, back to back.
     *  @param[in] query_offsets Where each query's rows start, one more than the queries, the
     *      first zero and the last the row count; host memory.
     *  @param[out] scored The nominees each query scored, before @c config.wanted cut any; host
     *      memory, copied to the host with each window.
     *  @param[out] matches Each query's kept keys and their products' widened sums, largest first,
     *      from @c offsets[q], up to @c config.wanted of them and every nominee for zero, at most
     *      the tape's stride; all on the first shard's device.
     *  @param[in] config The walks' configuration, each row ranking as many tokens as its expansion
     *      factor, and the keys kept per query, zero keeping every nominee.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
     *  @return The nominees scored, summed on the device, and the walks' telemetry.
     *
     *  Per window of whole queries: every shard walks and resolves its rows' rankings, which the
     *  first shard turns into each query's distinct nominees; their keys cross to the host once,
     *  16 B each, and every shard's own map lists their entries, 8 B each going down; every shard
     *  takes each nominee's best dot per row over its entries, and the first shard folds the
     *  maxima across shards, sums them per row and keeps the best into the tape. Three sleeping
     *  waits per window, no polling.
     *
     *  The @p matches offsets hold one more than the queries, the first the base the caller zeroed,
     *  and every window appends from its first offset.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the @p executor drives
     *    no stream on some shard's device, or the rows are not as wide as the index.
     *  - @c config_mismatch_k The offsets disagree with the queries or the rows, @p scored or the
     *    tape disagree with the queries, the tape is not on the first shard's device, or a query
     *    keeps more than the tape's stride.
     *  - @c uninitialized_k, @c contended_k, @c unsupported_k As the batched @c search says.
     */
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     std::span<std::size_t> scored, products_tape<key_t, f64_t> matches,
                                     maxsim_config_t config, executor_t& executor) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            std::size_t const queries_count = scored.size(), stride = matches.stride;
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return {{}, paired};
            if (query_offsets.size() != queries_count + 1 || query_offsets.front() != 0 ||
                query_offsets.back() != queries.rows || config.wanted > stride)
                return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = matches.refusal(queries_count, scored); failed(refusal)) return {{}, refusal};
            if (!queries_count) return {{}, status_t::success_k};
            if (queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
            int const first_device = shards_[0].index->device_specs().ordinal;
            cuda_api_t const& api = shards_[0].index->api();
            if (cuda_device_of(api, matches.offsets.data()) != first_device ||
                cuda_device_of(api, matches.keys.data()) != first_device ||
                cuda_device_of(api, matches.scores.data()) != first_device)
                return {{}, status_t::config_mismatch_k};

            // A window holds whole queries, as many as the longest one fits into the window's rows.
            std::size_t longest = 0;
            for (std::size_t query = 0; query != queries_count; ++query)
                longest = max_of(longest, query_offsets[query + 1] - query_offsets[query]);
            std::size_t const window_queries = max_of<std::size_t>(window_rows_k / max_of<std::size_t>(longest, 1), 1);
            std::size_t const widest_rows = min_of(queries.rows, window_queries * longest);
            std::size_t const width = shards_[0].index->non_zero_expansion_factor(config.search);
            // Without a bound, a query keeps every nominee: every row's breadth on every shard.
            if (!config.wanted && longest * shards_.size() * width > stride) return {{}, status_t::config_mismatch_k};
            config.search.count = static_cast<typename index_t::candidates_count_t>(width);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(0, 0, widest_rows, config.search); failed(begun))
                return {{}, begun};

            // The host's share of a window: its row offsets, the union of its nominees and every
            // shard's entries of them, reused by every window.
            growing_buffer<u32_t, forkunion_allocator_t> row_offsets {shards_.allocator()};
            growing_buffer<std::size_t, forkunion_allocator_t> nominee_offsets {shards_.allocator()};
            growing_buffer<key_t, forkunion_allocator_t> nominee_keys {shards_.allocator()};
            growing_buffer<vram_gathered_row_t, forkunion_allocator_t> entries {shards_.allocator()};
            growing_buffer<std::size_t, forkunion_allocator_t> entries_ends {shards_.allocator()};
            if (failed(row_offsets.resize(window_queries + 1)) || failed(nominee_offsets.resize(window_queries + 1)) ||
                failed(entries_ends.resize(fanout.shards())))
                return {{}, status_t::bad_alloc_k};
            auto maxsim_window = [&](std::size_t first_query, std::size_t taking) noexcept {
                std::size_t const first_row = query_offsets[first_query];
                for (std::size_t query = 0; query <= taking; ++query)
                    row_offsets[query] = static_cast<u32_t>(query_offsets[first_query + query] - first_row);
                products_tape<key_t, f64_t> const window {matches.keys, matches.scores,
                                                          matches.offsets.subspan(first_query, taking + 1), stride};
                return maxsim_window_(fanout, queries.slice(first_row, row_offsets[taking]),
                                      to_span(row_offsets).first(taking + 1), scored.subspan(first_query, taking),
                                      window, config, nominee_offsets, nominee_keys, entries, entries_ends);
            };
            return for_each_window<search_result_t>(queries_count, window_queries, maxsim_window);
        });
    }

    /**
     *  @brief Grouped-query attention of every token over every shard: each shard folds the nodes
     *      its walks score into per-head partial rows and carries on its own device, and the first
     *      shard's device merges them by the flash rule, exact over the union.
     *
     *  @param[in] queries The query heads, in the shards' stored scalar type as in @c search; row t
     *      · @p head_count + h is head h of token t, and each token walks every shard once.
     *  @param[in] head_count The query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Device-callable: <tt>value_lookup(key)</tt> points at the key's
     *      value row on a 16-byte boundary, as wide as the keys and stored like them, in memory
     *      every shard's device reads; a null row refuses the call with @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys, in host or
     *      device memory; rows in any scalar but f32 cast on the first shard's device first, and
     *      must fill whole bytes.
     *  @param[out] carry Each row's softmax state over every shard, one per row, in host or device
     *      memory; an empty span skips them.
     *  @param[in] config The per-shard walk configuration, applied uniformly; @c count plays no
     *      part.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
     *  @return The query rows attended, and the traversal telemetry every shard spent.
     *
     *  Whole tokens cross @c window_rows_k rows at a time, and a refused window keeps every window
     *  that was already merged.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The shards score by another metric than dot, the @p executor drives
     *    no stream on some shard's device, @p outputs or @p carry disagree with the rows, the
     *    queries are not as wide as the index, or the heads' scratch fits no walk.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c shape_mismatch_k The @p head_count is zero or does not divide the rows, or a walk
     *    visited a node with no value row.
     *  - @c unsupported_k A scan is asked, or the queries' or outputs' scalar has no device cast.
     *  - @c unsupported_layout_k Output or query rows sit closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     search_config_t config, executor_t& executor) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            using shard_lookup_t = vram_keyed_lookup<key_t, std::remove_cvref_t<value_lookup_type_>>;
            std::size_t const rows_count = queries.rows;
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
                return {{}, paired};
            if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
            if (status_t const refusal = shards_[0].index->attend_refusal_(rows_count, outputs, carry); failed(refusal))
                return {{}, refusal};
            if (!rows_count) return {{}, status_t::success_k};
            std::size_t const dimensions = this->dimensions();
            if (queries.columns != dimensions) return {{}, status_t::config_mismatch_k};

            // A window holds whole tokens, so none splits its heads across two.
            std::size_t const window_rows = max_of<std::size_t>(window_rows_k / head_count, 1) * head_count;
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(0, 0, min_of(rows_count, window_rows), config, head_count,
                                                    shards_.size());
                failed(begun))
                return {{}, begun};
            std::size_t const shards = fanout.shards();
            auto attend_window = [&](std::size_t first, std::size_t taking) noexcept -> expected<search_result_t> {
                rows_view_t const window_queries = queries.slice(first, taking);
                rows_span_t const window_outputs = outputs.slice(first, taking);
                std::span<softmax_state_t> const window_carry = carry.empty() ? carry : carry.subspan(first, taking);
                // Shard 0 stages one block of rows and carries per shard, its own first and the
                // others copied in.
                std::size_t const outputs_bytes = taking * dimensions * sizeof(f32_t);
                std::size_t const block_bytes = outputs_bytes + taking * sizeof(softmax_state_t);
                byte_t* blocks = nullptr;
                traversal_context_t* first_traversal = nullptr;
                auto attend_shard = [&](std::size_t shard, traversal_context_t& traversal,
                                        std::size_t visible) noexcept {
                    shard_lookup_t const lookup {shards_[shard].index->device_keys_(), value_lookup};
                    expected<f32_t*> const submitted = shards_[shard].index->submit_attend_(
                        window_queries, head_count, lookup, score_scale, traversal, config, shard ? 1 : shards,
                        shard ? 0 : shards_[0].index->exported_bytes_(window_outputs), visible);
                    if (!submitted) return submitted.status;
                    if (!shard) {
                        blocks = reinterpret_cast<byte_t*>(submitted.value);
                        first_traversal = &traversal;
                        return status_t::success_k;
                    }
                    // On the shard's own stream, so the copy follows its walks; unified addressing
                    // tells the two devices apart. Shard 0's staging may have grown on its stream,
                    // which its context recorded when it queued, so following that mark orders the
                    // copy after it.
                    return succeeded(traversal.follow(*first_traversal)) &&
                                   cuda_copy_async(traversal.api(), blocks + shard * block_bytes, submitted.value,
                                                   block_bytes, traversal.stream()) == cuda_success_k
                               ? status_t::success_k
                               : status_t::device_error_k;
                };
                // The first shard's stream waits on every other shard's event before this, so the
                // merge reads every block, and the output copy follows it.
                auto merge = [&](traversal_context_t& first) noexcept {
                    if (status_t const merging = shards_[0].index->launch_merge_attention_(
                            blocks, static_cast<u32_t>(taking), static_cast<u32_t>(dimensions),
                            static_cast<u32_t>(shards), first);
                        failed(merging))
                        return merging;
                    return shards_[0].index->copy_attended_(reinterpret_cast<f32_t*>(blocks), shards, window_outputs,
                                                            window_carry, first);
                };
                expected<search_result_t> attended = fanout.await_every_shard(0, attend_shard, merge);
                if (!attended) return attended;
                // A shard whose walk visited a node without its value row folded a wrong softmax.
                for (std::size_t shard = 0; shard != shards; ++shard)
                    if (fanout.traversal(shard).missed()) return {{}, status_t::shape_mismatch_k};
                attended.value.count = taking;
                return attended;
            };
            return for_each_window<search_result_t>(rows_count, window_rows, attend_window);
        });
    }

#pragma endregion Search

#pragma region Checkpoints

    /** What a device checkpoint does beyond the shared plans: it holds the executor's streams for
     *  the plan's whole life, every shard opens its own plan on its device's stream, a shard's own
     *  load plan uploads its keys, and a save closes the claim words, since device storage cannot
     *  be settlement-probed. */
    struct checkpoint_residency_t {

        /** The index whose shards a load fills, and whose shard generations it opens. */
        vram_nvidia_indexes* owner = nullptr;

        /** The streams every shard plan queues on. */
        executor_t* executor = nullptr;

        /** The executor's hold, kept for the plan's whole life, one caller at a time. */
        typename executor_t::claim_t streams;

        /** One source per mapped part of a load from a path, in part order; a file cuFile refuses
         *  leaves its source copying. */
        growing_buffer<cuda_part_source_t, forkunion_allocator_t> sources {forkunion_allocator_t {}};

        /** Opens a source per mapped part, cuFile-registered where the file takes O_DIRECT. */
        status_t open_sources(std::span<mapped_file_t> mappings) noexcept {
            if (failed(sources.resize(mappings.size()))) return status_t::bad_alloc_k;
            for (std::size_t part = 0; part != mappings.size(); ++part)
                sources[part] = {linux_cufile_file_t::open(executor->api(), mappings[part]), mappings[part].view()};
            return status_t::success_k;
        }

        /**
         *  Opens shard @p shard 's save into @p columns on the stream of its device.
         *
         *  @par Statuses:
         *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, or @p columns are too
         *    short for the shard's rows.
         *  - @c uninitialized_k The shard was never reserved.
         */
        expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                typename index_t::columns_t const& columns,
                                                                synchronization_t synchronization) noexcept {
            if (status_t const entered = fanout_t::enter(shards, shard); failed(entered)) return {{}, entered};
            return shards[shard].index->plan_save(columns, synchronization,
                                                  executor->stream(fanout_t::stream_of(shards, *executor, shard)));
        }

        /**
         *  @brief Opens shard @p shard 's load of @p rows on the stream of its device, whose steps
         *      copy the parts' columns from memory, or read their registered files through cuFile.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The shard was never reserved, or its @c status keeps it.
         *  - @c overflow_k The rows pass the shard's ceiling or reserved range, or its @c status
         *    keeps it.
         *  - @c contended_k A batch or load is in flight on the shard.
         *  - @c cancelled_k, @c config_mismatch_k, @c unsupported_k or @c unsupported_layout_k
         *    The shard keeps it, as its @c status says.
         */
        expected<typename index_t::load_plan_t> open_shard_load(shards_t& shards, std::size_t shard,
                                                                std::size_t rows) noexcept {
            if (status_t const entered = fanout_t::enter(shards, shard); failed(entered)) return {{}, entered};
            return shards[shard].index->plan_load(rows, executor->stream(fanout_t::stream_of(shards, *executor, shard)),
                                                  to_span(sources));
        }

        void end_save(shards_t& shards) noexcept { shards.reopen(); }

        /**
         *  Closes every shard of the loading index under its unique side.
         *
         *  @par Statuses:
         *  - @c contended_k A call in flight, a claim outstanding, or a plan holding the index
         *    closed.
         */
        status_t close(shards_t&) noexcept { return owner->close_gated_(); }

        /** Leaves part @p part the live shard it names, a placed one, or opens shard generations on
         *  the placed shards' devices until it exists. */
        status_t open_shard_for_part(shards_t& shards, std::size_t part) noexcept {
            while (shards.size() <= shards.retired_shards() + part)
                if (status_t const opened = shards.open_shard_generation_past_seal_(
                        shards.open_shard_generation(),
                        [this](shard_t const& like) noexcept { return owner->build_shard_(like); });
                    failed(opened))
                    return opened;
            return status_t::success_k;
        }
    };

    /** One save in flight across every shard: one shard plan each, the parts they write into, and
     *  the tail @c finish commits - the keys, the headers, and the manifest. */
    using save_plan_t = shard_save_plan<shards_t, checkpoint_residency_t>;

    /** One load in flight across every shard: the mapped parts, one shard plan each, and the flat
     *  steps their rows arrive in - one per part's share of one batch. */
    using load_plan_t = shard_load_plan<shards_t, checkpoint_residency_t>;

    /** Bytes a buffer @ref save under @p config writes: a lone live shard's monolith, or the
     *  manifest and every live shard's part it lists; zero for an invalid config or an unplaced
     *  index. The row counts are read now, so a buffer sized here fits a save no row enters
     *  before. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        return shard_storage_requirements(shards_, config);
    }

    /**
     *  @brief Opens a save under @p path: a lone live shard's monolith at the path itself, or one
     *      monolith per live shard beside the manifest the path names.
     *
     *  Every part is framed, sized and mapped here, and every header is written in @c finish, so a
     *  failed or cancelled save reads as absent.
     *
     *  @param[in] path The monolith's path, or the manifest's - its parts go beside it as
     *      @c -00001-of-0000S.safegraphs after the stem.
     *  @param[in] config The alignment, certified contract and provenance; the device
     *      storage cannot be settlement-probed, so @c concurrent_k refuses.
     *  @param[in,out] executor The streams the exports ride, held until the plan dies.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p config asks for @c concurrent_k or does not validate, the
     *    @p executor drives no stream on some shard's device, a shard's columns cannot hold its
     *    rows, or @ref status keeps it.
     *  - @c uninitialized_k The index has no shards or was never reserved, or @ref status keeps
     *    it.
     *  - @c contended_k Another call holds the @p executor, or a call, batch, load or plan holds
     *    the index.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c invalid_argument_k A part path does not parse.
     *  - @c overflow_k A device ends before a part does, or @ref status keeps it.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    expected<save_plan_t> plan_save(std::string_view path, save_config_t const& config, executor_t& executor) noexcept {
        auto [residency, status] = gate_save_(config, executor);
        if (failed(status)) return {{}, status};
        return save_plan_t::open(shards_, std::move(residency), path, config);
    }

    /**
     *  Opens a save into @p buffer - the in-memory sibling of the path form: a lone live shard's
     *  monolith, or the manifest leading the buffer and the parts it lists. Size the bytes by
     *  @ref storage_requirements.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p config asks for @c concurrent_k or does not validate, the
     *    @p executor drives no stream on some shard's device, a shard's columns cannot hold its
     *    rows, or @ref status keeps it.
     *  - @c uninitialized_k The index has no shards or was never reserved, or @ref status keeps
     *    it.
     *  - @c contended_k Another call holds the @p executor, or a call, batch, load or plan holds
     *    the index.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c overflow_k The @p buffer is shorter than the checkpoint, or @ref status keeps it.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    expected<save_plan_t> plan_save(bytes_span_t buffer, save_config_t const& config, executor_t& executor) noexcept {
        auto [residency, status] = gate_save_(config, executor);
        if (failed(status)) return {{}, status};
        return save_plan_t::open(shards_, std::move(residency), buffer, config);
    }

    /**
     *  @brief Opens a load of the checkpoint at @p path: the content says monolith or manifest,
     *      and every part's header is vetted before a byte is written.
     *
     *  The reservation comes from @p config, and part @c p goes to live shard @c p, opened past
     *  the placed ones where the parts outnumber them. The index must be empty;
     *  the plan reports the rows it will load through @c rows().
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] config The rows to reserve; zero fits the checkpoint.
     *  @param[in,out] executor The streams the imports ride, held until the plan dies.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, or
     *    @ref status keeps it.
     *  - @c config_mismatch_k The index holds entries, the @p executor drives no stream on some
     *    shard's device, the checkpoint's geometry, dtypes or metric differ from the shards', or
     *    @ref status keeps it.
     *  - @c contended_k Another call holds the @p executor or the gate, or a batch or plan holds
     *    a shard's claims.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, a part or its header does not parse, or its
     *    connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound or a key column, or
     *    @ref status keeps it.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    expected<load_plan_t> plan_load(std::string_view path, load_config_t const& config, executor_t& executor) noexcept {
        auto [residency, status] = gate_load_(executor);
        if (failed(status)) return {{}, status};
        return load_plan_t::open(shards_, std::move(residency), path, config);
    }

    /**
     *  Opens a load of the checkpoint held in @p buffer: a lone monolith, or the manifest leading
     *  the buffer and the parts it lists.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, or
     *    @ref status keeps it.
     *  - @c config_mismatch_k The index holds entries, the @p executor drives no stream on some
     *    shard's device, the checkpoint's geometry, dtypes or metric differ from the shards', or
     *    @ref status keeps it.
     *  - @c contended_k Another call holds the @p executor or the gate, or a batch or plan holds
     *    a shard's claims.
     *  - @c malformed_k The manifest lists no part, a part or its header does not parse, or its
     *    connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound or a key column, or
     *    @ref status keeps it.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    expected<load_plan_t> plan_load(bytes_view_t buffer, load_config_t const& config, executor_t& executor) noexcept {
        return plan_load(checkpoint_source_t {buffer}, config, executor);
    }

    /**
     *  @brief Opens a load of the checkpoint @p source holds, in host memory or on a device.
     *
     *  The manifest, the headers and the key columns cross to the host through a synchronous
     *  copy, and a device's columns then copy device to device on each shard's stream, with no
     *  bounce through host memory.
     *
     *  @par Statuses:
     *  - Those of the buffer form.
     *  - @c device_error_k A copy out of @p source 's device failed.
     */
    expected<load_plan_t> plan_load(checkpoint_source_t source, load_config_t const& config,
                                    executor_t& executor) noexcept {
        auto [residency, status] = gate_load_(executor);
        if (failed(status)) return {{}, status};
        cuda_api_t const& api = shards_[0].index->api();
        auto copy = [&api, device = source.device](bytes_span_t host, bytes_view_t bytes) noexcept {
            return cuda_set_device(api, device) == cuda_success_k &&
                           cuda_copy(api, host.data(), bytes.data(), bytes.size()) == cuda_success_k
                       ? status_t::success_k
                       : status_t::device_error_k;
        };
        return load_plan_t::open(shards_, std::move(residency), source, config, copy);
    }

    /**
     *  @brief Serializes the index under @p path - the one-shot over @ref plan_save, its steps
     *      walked from this one host thread.
     *
     *  @param[in] path The monolith's path, or the manifest's.
     *  @param[in] config The alignment, synchronization contract and provenance.
     *  @param[in,out] executor The streams the shard writes ride.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p config asks for @c concurrent_k or does not validate, the
     *    @p executor drives no stream on some shard's device, a shard's columns cannot hold its
     *    rows, a shard's steps never copied all its rows, a header does not fit before its data,
     *    or @ref status keeps it.
     *  - @c uninitialized_k The index has no shards or was never reserved, or @ref status keeps
     *    it.
     *  - @c contended_k Another call holds the @p executor, or a call, batch, load or plan holds
     *    the index.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c invalid_argument_k A part path or the manifest path does not parse.
     *  - @c overflow_k A device ends before a part does, a part is shorter than its header, the
     *    manifest does not fit, or @ref status keeps it.
     *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(path, config, executor);
        if (failed(status)) return status;
        if (status_t const drained = drain_steps_(plan); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /**
     *  @brief Serializes the index into @p buffer.
     *
     *  The one-shot over the bytes form of @ref plan_save.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p config asks for @c concurrent_k or does not validate, the
     *    @p executor drives no stream on some shard's device, a shard's columns cannot hold its
     *    rows, a shard's steps never copied all its rows, a header does not fit before its data,
     *    or @ref status keeps it.
     *  - @c uninitialized_k The index has no shards or was never reserved, or @ref status keeps
     *    it.
     *  - @c contended_k Another call holds the @p executor, or a call, batch, load or plan holds
     *    the index.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c overflow_k The @p buffer is shorter than the checkpoint, a part is shorter than its
     *    header, the manifest does not fit, or @ref status keeps it.
     *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t buffer, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(buffer, config, executor);
        if (failed(status)) return status;
        if (status_t const drained = drain_steps_(plan); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /**
     *  @brief Restores the checkpoint at @p path into the placed shards - the one-shot over
     *      @ref plan_load, its steps walked from this one host thread.
     *
     *  The index must be empty, and a batch in flight refuses the load as @c contended_k, as the
     *  load refuses a batch. A cancelled or failed load clears it and reports why, so nothing is
     *  published and the next load starts clean. Where cuFile takes a part's file, its columns read
     *  into device staging; every other part copies from the mapping.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, a bounce was
     *    never pinned, or @ref status keeps it.
     *  - @c config_mismatch_k The index holds entries, the @p executor drives no stream on some
     *    shard's device, the checkpoint's geometry, dtypes or metric differ from the shards', a
     *    step cannot write its rows, a shard's steps never copied them, or @ref status keeps it.
     *  - @c contended_k Another call holds the @p executor or the gate, or a batch or plan holds
     *    a shard's claims.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, a part, its header or its key column does
     *    not parse, or its connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound or a key column, or
     *    @ref status keeps it.
     *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
     *  - @c io_k A read through cuFile failed.
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

    /**
     *  @brief Restores the checkpoint held in @p buffer.
     *
     *  The one-shot over the bytes form of @ref plan_load.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, a bounce was
     *    never pinned, or @ref status keeps it.
     *  - @c config_mismatch_k The index holds entries, the @p executor drives no stream on some
     *    shard's device, the checkpoint's geometry, dtypes or metric differ from the shards', a
     *    step cannot write its rows, a shard's steps never copied them, or @ref status keeps it.
     *  - @c contended_k Another call holds the @p executor or the gate, or a batch or plan holds
     *    a shard's claims.
     *  - @c malformed_k The manifest lists no part, a part, its header or its key column does not
     *    parse, or its connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound or a key column, or
     *    @ref status keeps it.
     *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t buffer, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        return load(checkpoint_source_t {buffer}, config, executor, std::forward<progress_type_>(on_loaded));
    }

    /** Restores the checkpoint @p source holds, in host memory or on a device - the one-shot over
     *  the source form of @ref plan_load, with the statuses of both. */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(checkpoint_source_t source, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(source, config, executor);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /**
     *  The checkpoint's metadata without loading it - @ref safegraphs_read_metadata.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k A path does not parse.
     *  - @c malformed_k The file is empty, or its manifest or header does not parse.
     */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
    }

    /**
     *  The metadata of one monolith held in memory.
     *
     *  @par Statuses:
     *  - @c malformed_k The manifest or the header does not parse.
     */
    static expected<safegraphs_metadata_t> metadata(bytes_view_t buffer) noexcept {
        return safegraphs_read_metadata(buffer);
    }

#pragma endregion Checkpoints

#pragma region Lookups

    /** Entries stored under @p key across every shard, exact: the epoch covers the windows and the
     *  commit sequence covers batches spanning shards, so a batch counts whole or not at all. */
    std::size_t count(key_t const& key) const noexcept { return shards_.count(key); }

    /** Whether at least one entry is stored under @p key, under the same reads as @ref count. */
    bool contains(key_t const& key) const noexcept { return shards_.contains(key); }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  The batched @c get for one key: safe against concurrent insertions, which only ever add. A
     *  device fault, a held @p executor, or a shard on a device @p executor drives no stream on
     *  leaves rows unwritten with the total honest.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted, executor_t& executor) const noexcept {
        std::size_t total = 0;
        std::size_t const kept = min_of(wanted, vectors.extent(0));
        if (failed(get({&key, 1}, rows_span_t {vectors}.slice(0, kept), {&total, 1}, kept, executor)))
            return count(key);
        return total;
    }

    /**
     *  @brief The batched @c count: the entries stored under every key of @p keys.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *
     *  The executor is unused, since every shard's key map lives on the host; it rides in the
     *  signature so a caller drives every residency alike.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p entries are not as many as the @p keys.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        for (std::size_t position = 0; position != keys.size(); ++position) entries[position] = count(keys[position]);
        return status_t::success_k;
    }

    /**
     *  The batched @c contains: one in @p entries where a key of @p keys holds any entry, zero
     *  where it holds none; the executor rides in the signature as in the batched @ref count.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p entries are not as many as the @p keys.
     */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t&) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        for (std::size_t position = 0; position != keys.size(); ++position)
            entries[position] = contains(keys[position]) ? 1 : 0;
        return status_t::success_k;
    }

    /**
     *  @brief The batched @c get: up to @p wanted rows of every key of @p keys, and every key's
     *      true total, gathered on the devices and cast into the rows' scalar there.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] rows @b [keys·wanted,dimensions] in any scalar NumKong casts the stored one
     *      into: row q · @p wanted + i is the i-th row of key q, and rows past a key's total are
     *      left unspecified; wider rows keep their tails.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted
     *      when its rows truncated, zero when absent.
     *  @param[in] wanted Rows kept per key.
     *  @param[in,out] executor The streams the gathers ride, paired positionally with the shards.
     *
     *  Every shard's own key map is probed on the host under one commit sequence, and only the node
     *  lists cross down, 8 B per row. Rows in device memory are gathered in place by the shards on
     *  their device; host rows are gathered into the first shard's staging and copied to the host
     *  once. A shard on another device gathers into its own staging, then copies each row across.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p entries are not as many as the @p keys, @p rows hold other
     *    than @p keys · @p wanted rows or narrower ones, or @p executor drives no stream on some
     *    shard's device.
     *  - @c unsupported_k NumKong has no device cast from the stored scalar into the rows'.
     *  - @c uninitialized_k The index has no shards.
     *  - @c contended_k Another call holds @p executor, or a destructive window overlapped the
     *    read twice.
     */
    status_t get(std::span<key_t const> keys, rows_span_t rows, std::span<std::size_t> entries, std::size_t wanted,
                 executor_t& executor) const noexcept {
        std::size_t const rows_count = keys.size() * wanted;
        if (entries.size() != keys.size() || rows.rows != rows_count || (rows_count && rows.columns < dimensions()))
            return status_t::config_mismatch_k;
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return paired;
        return shards_.read_retrying_once([&]() noexcept -> status_t {
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(0, 0); failed(begun)) return begun;
            std::size_t const shards = fanout.shards();
            growing_buffer<vram_gathered_row_t, forkunion_allocator_t> listed {shards_.allocator()};
            growing_buffer<std::size_t, forkunion_allocator_t> listed_ends {shards_.allocator()};
            if (failed(listed.reserve_exactly(rows_count)) || failed(listed_ends.resize(shards)))
                return status_t::bad_alloc_k;
            // Shard by shard, so each shard's list is one run, under one commit sequence, so a
            // batch spanning shards counts whole or not at all.
            shards_.read_committed([&]() noexcept {
                listed.clear();
                for (std::size_t& total : entries) total = 0;
                for (std::size_t shard = 0; shard != shards; ++shard) {
                    for (std::size_t position = 0; position != keys.size(); ++position)
                        shards_[shard].index->find_each(keys[position], [&](node_id_t node) noexcept {
                            if (entries[position] < wanted)
                                listed.push_back_reserved(
                                    vram_gathered_row_t {static_cast<u32_t>(node),
                                                         static_cast<u32_t>(position * wanted + entries[position])});
                            ++entries[position];
                        });
                    listed_ends[shard] = listed.size();
                }
                return listed.size();
            });

            cuda_api_t const& api = shards_[0].index->api();
            int const rows_device = cuda_device_of(api, rows.data);
            int const target_device = rows_device >= 0 ? rows_device : shards_[0].index->device_specs().ordinal;
            std::size_t const row_bytes = packed_bytes(rows.dtype, dimensions());
            std::size_t const hosted_bytes =
                rows_device >= 0 ? 0 : round_up_to_multiple<std::size_t>(rows_count * row_bytes, default_alignment_k);
            rows_span_t gathered {rows.data, rows_count, dimensions(), rows.stride, rows.dtype};
            // The most one shard carves: the first's hosted rows, every listed row, and off the
            // rows' device a staging as tall as the rows.
            std::size_t const widest_outputs_bytes =
                hosted_bytes +
                round_up_to_multiple<std::size_t>(rows_count * sizeof(vram_gathered_row_t), default_alignment_k) +
                rows_count * row_bytes;
            traversal_context_t* first_traversal = nullptr;
            auto gather_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t) noexcept {
                std::size_t const first = shard ? listed_ends[shard - 1] : 0, count = listed_ends[shard] - first;
                bool const in_place = traversal.device() == target_device;
                std::size_t const own_hosted_bytes = shard ? 0 : hosted_bytes;
                std::size_t const list_bytes = round_up_to_multiple<std::size_t>(count * sizeof(vram_gathered_row_t),
                                                                                 default_alignment_k);
                std::size_t const outputs_bytes = own_hosted_bytes + list_bytes +
                                                  (in_place ? 0 : rows_count * row_bytes);
                byte_t* const outputs = traversal.allocator().allocate(outputs_bytes);
                if (!outputs && outputs_bytes) return status_t::bad_alloc_k;
                if (!shard) {
                    first_traversal = &traversal;
                    if (hosted_bytes) gathered = {outputs, rows_count, dimensions(), row_bytes, rows.dtype};
                }
                // The first shard's staging may have grown on its stream, so later shards follow
                // its mark before writing into it.
                else if (failed(traversal.follow(*first_traversal))) return status_t::device_error_k;
                if (!count) return status_t::success_k;
                vram_gathered_row_t* const list = reinterpret_cast<vram_gathered_row_t*>(outputs + own_hosted_bytes);
                std::span<vram_gathered_row_t const> const shard_list {list, count};
                if (in_place)
                    return fanout_t::vendor_t::copy(api, list, listed.data() + first,
                                                    count * sizeof(vram_gathered_row_t),
                                                    traversal.stream()) == status_t::success_k
                               ? shards_[shard].index->launch_gather_vectors_(shard_list, gathered, traversal)
                               : status_t::device_error_k;
                // Off the rows' device the rows gather into a staging laid out as the rows, then
                // cross one by one.
                rows_span_t const staged {outputs + own_hosted_bytes + list_bytes, rows_count, dimensions(), row_bytes,
                                          rows.dtype};
                status_t status = fanout_t::vendor_t::copy(api, list, listed.data() + first,
                                                           count * sizeof(vram_gathered_row_t), traversal.stream());
                if (succeeded(status))
                    status = shards_[shard].index->launch_gather_vectors_(shard_list, staged, traversal);
                for (std::size_t row = first; row != first + count && succeeded(status); ++row) {
                    std::size_t const output_row = listed[row].output_row;
                    status = fanout_t::vendor_t::copy(api, gathered.row(output_row), staged.row(output_row), row_bytes,
                                                      traversal.stream());
                }
                return status;
            };
            auto bring_to_host = [&](traversal_context_t& first) noexcept {
                if (!hosted_bytes) return status_t::success_k;
                return fanout_t::vendor_t::copy_2d(api, rows.data, rows.stride, gathered.data, row_bytes, row_bytes,
                                                   rows_count, first.stream());
            };
            return fanout.await_every_shard(widest_outputs_bytes, gather_shard, bring_to_host).status;
        });
    }

#pragma endregion Lookups

  private:

    /** Every device a shard sits on, which each shard maps its storage for, so a load reads a shard
     *  on another device directly. */
    std::bitset<gpu_devices_capacity_k> devices_() const noexcept {
        std::bitset<gpu_devices_capacity_k> devices;
        for (shard_t const& shard : shards_)
            devices[static_cast<std::size_t>(shard.index->device_specs().ordinal)] = true;
        return devices;
    }

    /** One unreserved shard placed on @p device; null when the allocation fails. */
    std::unique_ptr<index_t> build_(cuda_api_t const& api, int device) const noexcept {
        typename index_t::vectors_allocator_t const allocator {api, device};
        return std::unique_ptr<index_t>(new (std::nothrow)
                                            index_t(kernel(), geometry_, allocator, allocator, shards_.allocator()));
    }

    /** Forgets every published shard's rows and keys in place, each keeping its storage, keeping
     *  the shard generations past the first as idle shards: what a clear and an emptied donor
     *  become, inside their own window with every shard closed. */
    void clear_() noexcept {
        for (shard_t& shard : shards_) shard.index->forget_();
        shards_.forget_entries();
    }

    /** Closes every shard under the unique side, refusing a call in flight, a claim outstanding or
     *  a plan holding the index closed with @c contended_k; closing moves nothing a reader sees, so
     *  it moves no epoch. */
    status_t close_gated_() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        return held ? shards_.close() : status_t::contended_k;
    }

    /** One shard of a new shard generation, on the device @p like sits on, current while its
     *  events are born, mapped for every device placed and bound under the bound; a null index
     *  when the device, the allocation or the binding fails. */
    shard_t build_shard_(shard_t const& like) noexcept {
        int const device = like.index->device_specs().ordinal;
        cuda_api_t const& api = like.index->api();
        std::unique_ptr<index_t> index = cuda_set_device(api, device) == cuda_success_k ? build_(api, device) : nullptr;
        if (index) index->share_with_(devices_());
        reserve_capacity_t bounded {};
        bounded.ceiling = shards_.shard_capacity();
        if (index && failed(index->reserve(bounded).status)) index.reset();
        return shard_t {std::move(index)};
    }

    /** Opens one run on its stream and one walk context over each of the first @p shards shards,
     *  the ones the plan spans, seeded for the widest shard of @p shaped as it stands, @p walks
     *  walks into one visited set, and a split's partition into @p receivers parts when two or
     *  more; appended rows grow them. */
    status_t open_walks_(refine_plan_t& plan, executor_t& executor, vram_nvidia_indexes const& shaped,
                         std::size_t shards, unsigned walks, std::size_t receivers) noexcept {
        if (failed(plan.arenas_.reserve_exactly(shards)) || failed(plan.traversals_.reserve_exactly(shards)))
            return status_t::bad_alloc_k;
        std::size_t walk_bytes = 0;
        for (shard_t const& shard : shaped.shards_)
            walk_bytes = max_of(walk_bytes, shard.index->traversal_memory_usage(plan.config_, walks, receivers));
        for (std::size_t shard = 0; shard != shards; ++shard) {
            typename executor_t::stream_t& stream = executor.stream(fanout_t::stream_of(shards_, executor, shard));
            plan.arenas_.push_back_reserved(typename executor_t::arena_t {stream.arena.upstream()});
            if (failed(plan.arenas_[shard].reserve_exactly(walk_bytes))) return status_t::bad_alloc_k;
            plan.traversals_.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {plan.arenas_[shard], stream}});
            if (!plan.traversals_.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Appends every row of @p donor where @p moves, one per donor shard, put it and consumes it,
     *  under both indexes' closed claim words and the executor's hold @p refining keeps: the
     *  donor's gate and every other fallible step first, then the copies, one drain, and the
     *  publication, and the donor's shards become the plan's parts as it then forgets all its rows
     *  in its own window. */
    status_t absorb_(vram_nvidia_indexes& donor, std::span<shard_move_t const> moves, executor_t& executor,
                     refine_plan_t& refining) noexcept {
        std::size_t const shards = refining.traversals_.size();
        if (!donor.size()) return status_t::success_k;
        // The donor empties once its rows are copied, so its gate comes before anything that a
        // refusal could undo.
        unique_lock_t const emptying {donor.gate_, std::try_to_lock};
        if (!emptying) return status_t::contended_k;

        // Every shard claims all its rows first, so a failure before publication releases them all.
        std::size_t const donors = moves.size();
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        if (failed(plans.reserve_exactly(shards))) return status_t::bad_alloc_k;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = 0; from != donors; ++from)
                if (moves[from].into == into) rows += donor.shards_[from].index->size();
            if (status_t const entered = fanout_t::enter(shards_, into); failed(entered)) return entered;
            auto [plan, status] = shards_[into].index->plan_load(
                rows, executor.stream(fanout_t::stream_of(shards_, executor, into)));
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
        }

        // A refused copy still waits for those queued before it, whose rows go to the next claim.
        status_t queued = status_t::success_k;
        for (std::size_t into = 0; into != shards && succeeded(queued); ++into) {
            typename index_t::load_plan_t& plan = plans[into];
            node_id_t base = plan.base;
            for (std::size_t from = 0; from != donors && succeeded(queued); ++from) {
                if (moves[from].into != into) continue;
                shard_t const& donor_slot = donor.shards_[from];
                node_id_t const part_rows = static_cast<node_id_t>(donor_slot.index->size());
                if (!part_rows) continue;
                typename index_t::load_plan_t::step_t step;
                step.index_nodes = {base, base + part_rows};
                step.source_nodes = {0, part_rows};
                queued = plan.load(step, *donor_slot.index, step.source_nodes);
                base += part_rows;
            }
        }
        if (status_t const drained = executor.wait(); failed(queued) || failed(drained))
            return failed(queued) ? queued : drained;
        for (typename index_t::load_plan_t& plan : plans) {
            plan.publish_keys();
            if (status_t const published = plan.finish(); failed(published)) return published;
        }

        // Each donor shard becomes a part of its own in the plan, which sized every shard's run.
        shards_.record_parts(donor.shards_, moves, plans, refining.part_bases_.data(), to_span(refining.part_offsets_));
        donor.shards_.begin_window();
        donor.clear_();
        donor.shards_.end_window();
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
        if (status_t const closed = close_gated_(); failed(closed)) return {{}, closed};
        return {checkpoint_residency_t {this, &executor, std::move(claim)}, status_t::success_k};
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
        return {checkpoint_residency_t {this, &executor, std::move(claim)}, status_t::success_k};
    }

    /** Walks @p plan 's steps from this one host thread, taking step @c s of every shard before
     *  step @c s+1, so a device copies one step while the host stages another's, and the first
     *  failing step ends it. A save writes each step, a load reads it straight out of its part. */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan) noexcept {
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step) {
            planned.save(step);
        };
        std::size_t const shards = shards_.size();
        auto const steps = plan.steps();
        std::size_t deepest = 0;
        for (std::size_t shard = 0; shard != shards; ++shard) deepest = max_of(deepest, plan.steps_of(shard).size());
        for (std::size_t position = 0; position != deepest; ++position)
            for (std::size_t shard = 0; shard != shards; ++shard) {
                iota_view<std::size_t> const own = plan.steps_of(shard);
                if (position >= own.size()) continue;
                status_t stepped = status_t::success_k;
                if constexpr (saving_k) stepped = plan.save(steps[own.first + position]);
                else stepped = plan.load(steps[own.first + position]);
                if (failed(stepped)) return stepped;
            }
        return status_t::success_k;
    }

    /** One window of the batched @c search: on the first shard's device when the rankings live
     *  there and score in @c f32 or @c f64, through the host staging otherwise. */
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
        // Vetted before anything is carved; the shards would refuse only after the walk outputs and
        // the runs had been sized for a doomed call.
        if (queries.columns != dimensions()) return {{}, status_t::config_mismatch_k};
        if constexpr (std::is_same_v<score_type_, f32_t> || std::is_same_v<score_type_, f64_t>) {
            expected<search_result_t> found = search_on_device_(fanout, queries, config, keys, distances, counts,
                                                                heads...);
            if (found.status != status_t::deferred_k) return found;
        }
        auto search_shard = [&](std::size_t shard, nk::matrix_span<node_id_t> ids, nk::matrix_span<distance_t> scores,
                                std::span<std::size_t>, traversal_context_t& traversal, std::size_t visible) noexcept {
            return shards_[shard]
                .index->submit_search_(queries, ids, scores, traversal, config, visible, heads...)
                .status;
        };
        return fanout.search_every_shard(queries_count, search_shard, keys, distances, counts);
    }

    /** One window of @c multiply: every shard walks and resolves its products into a ranking on
     *  its own device, every further one copies it to the first, and one compaction there appends
     *  every row's kept pairs to @p products and counts the pairs it scored, which cross to the
     *  host's @p scored with their sum. */
    template <typename product_type_>
    expected<search_result_t> multiply_window_(fanout_t& fanout, rows_view_t left,
                                               products_tape<key_t, product_type_> const& products,
                                               std::span<std::size_t> scored, search_config_t config) noexcept {
        std::size_t const rows_count = left.rows, shards = fanout.shards(), wanted = config.count;
        auto aligned = [](std::size_t bytes) noexcept {
            return round_up_to_multiple<std::size_t>(bytes, default_alignment_k);
        };
        std::size_t const ids_bytes = aligned(rows_count * wanted * sizeof(node_id_t));
        std::size_t const products_bytes = aligned(rows_count * wanted * sizeof(dot_result_t));
        std::size_t const walked_bytes = ids_bytes + products_bytes + aligned(rows_count * sizeof(u32_t));
        std::size_t const counts_bytes = rows_count * sizeof(std::size_t),
                          keys_bytes = rows_count * wanted * sizeof(key_t);
        std::size_t const block_bytes = aligned(counts_bytes + keys_bytes +
                                                rows_count * wanted * sizeof(product_type_));
        auto ranking_of = [&](byte_t* block) noexcept {
            return vram_ranking_t {block + counts_bytes,
                                   block + counts_bytes + keys_bytes,
                                   reinterpret_cast<std::size_t*>(block),
                                   wanted * sizeof(key_t),
                                   wanted * sizeof(product_type_),
                                   static_cast<u32_t>(wanted),
                                   sizeof(key_t),
                                   sizeof(product_type_)};
        };

        // The first shard's staging holds every shard's ranking, then the scored counts and the
        // word they sum into.
        std::size_t const scored_bytes = aligned(rows_count * sizeof(std::size_t));
        std::size_t const first_outputs_bytes = walked_bytes + shards * block_bytes + scored_bytes +
                                                sizeof(std::size_t);
        byte_t* first_outputs = nullptr;
        traversal_context_t* first_traversal = nullptr;
        auto multiply_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            byte_t* const outputs = traversal.allocator().allocate(shard ? walked_bytes + block_bytes
                                                                         : first_outputs_bytes);
            if (!outputs) return status_t::bad_alloc_k;
            if (!shard) first_outputs = outputs, first_traversal = &traversal;
            nk::matrix_span<node_id_t> const ids {reinterpret_cast<node_id_t*>(outputs), rows_count, wanted};
            nk::matrix_span<dot_result_t> const products {reinterpret_cast<dot_result_t*>(outputs + ids_bytes),
                                                          rows_count, wanted};
            std::span<u32_t> const counts {reinterpret_cast<u32_t*>(outputs + ids_bytes + products_bytes), rows_count};
            expected<search_result_t> const submitted = shards_[shard].index->submit_multiply_(
                left, ids, products, counts, traversal, config, visible);
            if (!submitted) return submitted.status;
            byte_t* const block = outputs + walked_bytes;
            if (status_t const resolved = shards_[shard].index->launch_resolve_products_(ids, products, counts,
                                                                                         ranking_of(block), traversal);
                failed(resolved) || !shard)
                return resolved;
            // The first shard's staging may have grown on its stream, which its context recorded
            // when it queued, so following that mark orders the copy after the growth.
            if (failed(traversal.follow(*first_traversal))) return status_t::device_error_k;
            return fanout_t::vendor_t::copy(traversal.api(), first_outputs + walked_bytes + shard * block_bytes, block,
                                            block_bytes, traversal.stream());
        };
        // The first shard's stream waits on every other shard's event before this, so the
        // compaction reads every ranking.
        auto merge = [&](traversal_context_t& first) noexcept {
            std::span<std::size_t> const scored_on_device {
                reinterpret_cast<std::size_t*>(first_outputs + walked_bytes + shards * block_bytes), rows_count};
            std::size_t* const total = reinterpret_cast<std::size_t*>(first_outputs + walked_bytes +
                                                                      shards * block_bytes + scored_bytes);
            vram_products_tape_t const tape {products.offsets.data(), reinterpret_cast<byte_t*>(products.keys.data()),
                                             reinterpret_cast<byte_t*>(products.scores.data())};
            if (status_t const compacted = shards_[0].index->launch_compact_products_(
                    ranking_of(first_outputs + walked_bytes), block_bytes, static_cast<u32_t>(shards),
                    static_cast<u32_t>(rows_count), tape, scored_on_device, first);
                failed(compacted))
                return compacted;
            if (status_t const summed = shards_[0].index->launch_sum_counts_(scored_on_device, total, first);
                failed(summed))
                return summed;
            if (status_t const hosted = fanout_t::vendor_t::copy(first.api(), scored.data(), scored_on_device.data(),
                                                                 scored.size_bytes(), first.stream());
                failed(hosted))
                return hosted;
            return first.copy_matched(total);
        };
        expected<search_result_t> found = fanout.await_every_shard(first_outputs_bytes, multiply_shard, merge);
        if (!found) return found;
        found.value.count = fanout.traversal(0).matched();
        return found;
    }

    /** One window of @c maxsim over @p queries, whose @p row_offsets are relative: the walks and
     *  the nominees, the nominees' keys up to the host and their entries down, then the rescoring,
     *  the fold and the selection into @p matches, each step one await, the last copying each
     *  query's scored count to the host's @p scored. Only the walks' telemetry counts, since the
     *  later awaits copy the same counters again. */
    expected<search_result_t> maxsim_window_(
        fanout_t& fanout, rows_view_t queries, std::span<u32_t const> row_offsets, std::span<std::size_t> scored,
        products_tape<key_t, f64_t> const& matches, maxsim_config_t config,
        growing_buffer<std::size_t, forkunion_allocator_t>& nominee_offsets,
        growing_buffer<key_t, forkunion_allocator_t>& nominee_keys,
        growing_buffer<vram_gathered_row_t, forkunion_allocator_t>& entries,
        growing_buffer<std::size_t, forkunion_allocator_t>& entries_ends) noexcept {
        std::size_t const rows = queries.rows, shards = fanout.shards(), queries_count = row_offsets.size() - 1;
        std::size_t const width = config.search.count;
        std::size_t longest = 0;
        for (std::size_t query = 0; query != queries_count; ++query)
            longest = max_of<std::size_t>(longest, row_offsets[query + 1] - row_offsets[query]);
        std::size_t const capacity = longest * shards * width;
        auto aligned = [](std::size_t bytes) noexcept {
            return round_up_to_multiple<std::size_t>(bytes, default_alignment_k);
        };

        // Walks: every shard's rows ranked and resolved into a block the first shard gathers.
        std::size_t const ids_bytes = aligned(rows * width * sizeof(node_id_t));
        std::size_t const walked_bytes = ids_bytes + aligned(rows * width * sizeof(distance_t));
        std::size_t const counts_bytes = rows * sizeof(std::size_t), scores_bytes = rows * width * sizeof(distance_t);
        std::size_t const block_bytes = aligned(counts_bytes + scores_bytes + rows * width * sizeof(key_t));
        auto ranking_of = [&](byte_t* block) noexcept {
            return vram_ranking_t {block + counts_bytes + scores_bytes,
                                   block + counts_bytes,
                                   reinterpret_cast<std::size_t*>(block),
                                   width * sizeof(key_t),
                                   width * sizeof(distance_t),
                                   static_cast<u32_t>(width),
                                   sizeof(key_t),
                                   sizeof(distance_t)};
        };
        // The first shard's nominees past the blocks: the device row offsets, each query's distinct
        // keys at full capacity, then the union packed from offset zero.
        std::size_t const row_offsets_bytes = aligned(row_offsets.size_bytes());
        std::size_t const nominees_bytes = aligned(queries_count * sizeof(std::size_t)) +
                                           aligned(queries_count * capacity * sizeof(key_t));
        std::size_t const union_offsets_bytes = aligned((queries_count + 1) * sizeof(std::size_t));
        std::size_t const first_walk_bytes = walked_bytes + shards * block_bytes + row_offsets_bytes + nominees_bytes +
                                             union_offsets_bytes + queries_count * capacity * sizeof(key_t);
        byte_t* first_outputs = nullptr;
        traversal_context_t* first_traversal = nullptr;
        auto walk_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            byte_t* const outputs = traversal.allocator().allocate(shard ? walked_bytes + block_bytes
                                                                         : first_walk_bytes);
            if (!outputs) return status_t::bad_alloc_k;
            if (!shard) first_outputs = outputs, first_traversal = &traversal;
            node_id_t* const ids = reinterpret_cast<node_id_t*>(outputs);
            distance_t* const distances = reinterpret_cast<distance_t*>(outputs + ids_bytes);
            expected<search_result_t> const found = shards_[shard].index->submit_search_(
                queries, nk::matrix_span<node_id_t> {ids, rows, width},
                nk::matrix_span<distance_t> {distances, rows, width}, traversal, config.search, visible);
            if (!found) return found.status;
            byte_t* const block = outputs + walked_bytes;
            if (status_t const resolved = shards_[shard].index->launch_resolve_matches_(
                    ids, distances, static_cast<u32_t>(width),
                    reinterpret_cast<byte_t const*>(shards_[shard].index->device_keys_()), ranking_of(block),
                    static_cast<u32_t>(rows), traversal);
                failed(resolved))
                return resolved;
            if (!shard)
                return fanout_t::vendor_t::copy(traversal.api(), block + shards * block_bytes, row_offsets.data(),
                                                row_offsets.size_bytes(), traversal.stream());
            // The first shard's staging may have grown on its stream, which its context recorded
            // when it queued, so following that mark orders the copy after the growth.
            if (failed(traversal.follow(*first_traversal))) return status_t::device_error_k;
            return fanout_t::vendor_t::copy(traversal.api(), first_outputs + walked_bytes + shard * block_bytes, block,
                                            block_bytes, traversal.stream());
        };
        auto nominate = [&](traversal_context_t& first) noexcept {
            byte_t* const offsets_on_device = first_outputs + walked_bytes + shards * block_bytes;
            byte_t* const nominees = offsets_on_device + row_offsets_bytes;
            byte_t* const union_offsets = nominees + nominees_bytes;
            byte_t* const union_keys = union_offsets + union_offsets_bytes;
            std::span<u32_t const> const device_row_offsets {reinterpret_cast<u32_t const*>(offsets_on_device),
                                                             row_offsets.size()};
            vram_ranking_t const distinct {nominees + aligned(queries_count * sizeof(std::size_t)),
                                           nullptr,
                                           reinterpret_cast<std::size_t*>(nominees),
                                           capacity * sizeof(key_t),
                                           0,
                                           static_cast<u32_t>(capacity),
                                           sizeof(key_t),
                                           0};
            vram_products_tape_t const nominated {reinterpret_cast<std::size_t*>(union_offsets), union_keys, nullptr};
            status_t status = shards_[0].index->launch_nominate_keys_(ranking_of(first_outputs + walked_bytes),
                                                                      block_bytes, static_cast<u32_t>(shards),
                                                                      device_row_offsets, distinct, first);
            if (succeeded(status))
                status = fanout_t::vendor_t::fill(first.api(), union_offsets, 0, sizeof(std::size_t), first.stream());
            if (succeeded(status))
                status = shards_[0].index->launch_compact_products_(distinct, 0, 1, static_cast<u32_t>(queries_count),
                                                                    nominated, {}, first);
            if (succeeded(status))
                status = fanout_t::vendor_t::copy(first.api(), nominee_offsets.data(), union_offsets,
                                                  (queries_count + 1) * sizeof(std::size_t), first.stream());
            return status;
        };
        expected<search_result_t> walked = fanout.await_every_shard(first_walk_bytes, walk_shard, nominate);
        if (!walked) return walked;

        // The nominees' keys up to the host, 16 B each, then every shard's entries of them through
        // its map, under one commit sequence.
        std::size_t const nominees_count = nominee_offsets[queries_count];
        if (failed(nominee_keys.resize(nominees_count))) return {walked.value, status_t::bad_alloc_k};
        auto keys_to_host = [&](traversal_context_t& first) noexcept {
            byte_t* const union_keys = first_outputs + walked_bytes + shards * block_bytes + row_offsets_bytes +
                                       nominees_bytes + union_offsets_bytes;
            return fanout_t::vendor_t::copy(first.api(), nominee_keys.data(), union_keys,
                                            nominees_count * sizeof(key_t), first.stream());
        };
        auto queue_nothing = [](std::size_t, traversal_context_t&, std::size_t) noexcept {
            return status_t::success_k;
        };
        if (status_t const hosted = fanout.await_every_shard(0, queue_nothing, keys_to_host).status; failed(hosted))
            return {walked.value, hosted};
        status_t listed = status_t::success_k;
        shards_.read_committed([&]() noexcept {
            entries.clear();
            listed = status_t::success_k;
            for (std::size_t shard = 0; shard != shards; ++shard) {
                for (std::size_t nominee = 0; nominee != nominees_count; ++nominee)
                    shards_[shard].index->find_each(nominee_keys[nominee], [&](node_id_t node) noexcept {
                        if (failed(entries.push_back(
                                vram_gathered_row_t {static_cast<u32_t>(node), static_cast<u32_t>(nominee)})))
                            listed = status_t::bad_alloc_k;
                    });
                entries_ends[shard] = entries.size();
            }
            return entries.size();
        });
        if (failed(listed)) return {walked.value, listed};

        // Rescoring: every shard's best dot per nominee and row over its own entries, gathered into
        // the first shard's staging, folded and selected there.
        std::size_t const nominee_offsets_bytes = aligned((queries_count + 1) * sizeof(std::size_t));
        std::size_t const partials_bytes = aligned(nominees_count * longest * sizeof(dot_result_t));
        std::size_t const keys_bytes = aligned(nominees_count * sizeof(key_t));
        // The most one shard carves: every shard's entries, and on the first every shard's
        // partials, the nominees' keys and scores, the scored counts and the word they sum into.
        std::size_t const scores_bytes_of_nominees = aligned(nominees_count * sizeof(f64_t));
        std::size_t const scored_bytes = aligned(queries_count * sizeof(std::size_t));
        std::size_t const first_own_bytes = shards * partials_bytes + keys_bytes + scores_bytes_of_nominees +
                                            scored_bytes + sizeof(std::size_t);
        std::size_t const widest_rescore_bytes = row_offsets_bytes + nominee_offsets_bytes +
                                                 aligned(entries.size() * sizeof(vram_gathered_row_t)) +
                                                 first_own_bytes;
        auto rescore_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t) noexcept {
            std::size_t const first = shard ? entries_ends[shard - 1] : 0, count = entries_ends[shard] - first;
            std::size_t const entries_bytes = aligned(count * sizeof(vram_gathered_row_t));
            std::size_t const shared_bytes = row_offsets_bytes + nominee_offsets_bytes + entries_bytes;
            std::size_t const own_bytes = shard ? partials_bytes : first_own_bytes;
            byte_t* const outputs = traversal.allocator().allocate(shared_bytes + own_bytes);
            if (!outputs) return status_t::bad_alloc_k;
            if (!shard) first_outputs = outputs, first_traversal = &traversal;
            byte_t* const device_nominee_offsets = outputs + row_offsets_bytes;
            byte_t* const device_entries = device_nominee_offsets + nominee_offsets_bytes;
            byte_t* const partials = device_entries + entries_bytes;
            typename fanout_t::vendor_t::stream_handle_t const stream = traversal.stream();
            typename fanout_t::vendor_t::api_t const& shard_api = traversal.api();
            status_t status = fanout_t::vendor_t::copy(shard_api, outputs, row_offsets.data(), row_offsets.size_bytes(),
                                                       stream);
            if (succeeded(status))
                status = fanout_t::vendor_t::copy(shard_api, device_nominee_offsets, nominee_offsets.data(),
                                                  (queries_count + 1) * sizeof(std::size_t), stream);
            if (succeeded(status) && count)
                status = fanout_t::vendor_t::copy(shard_api, device_entries, entries.data() + first,
                                                  count * sizeof(vram_gathered_row_t), stream);
            if (!shard && succeeded(status) && nominees_count)
                status = fanout_t::vendor_t::copy(shard_api, partials + shards * partials_bytes, nominee_keys.data(),
                                                  nominees_count * sizeof(key_t), stream);
            if (failed(status) || !nominees_count) return status;
            status = shards_[shard].index->launch_maxsim_rescore_(
                shards_[shard].index->staged_queries_(traversal, rows),
                {reinterpret_cast<u32_t const*>(outputs), row_offsets.size()},
                {reinterpret_cast<std::size_t const*>(device_nominee_offsets), queries_count + 1},
                {reinterpret_cast<vram_gathered_row_t const*>(device_entries), count},
                nk::matrix_span<dot_result_t> {reinterpret_cast<dot_result_t*>(partials), nominees_count, longest},
                traversal);
            if (failed(status) || !shard) return status;
            if (failed(traversal.follow(*first_traversal))) return status_t::device_error_k;
            byte_t* const first_partials = first_outputs + row_offsets_bytes + nominee_offsets_bytes +
                                           aligned(entries_ends[0] * sizeof(vram_gathered_row_t));
            return fanout_t::vendor_t::copy(shard_api, first_partials + shard * partials_bytes, partials,
                                            partials_bytes, stream);
        };
        // The first shard's stream waits on every other shard's event before this, so the fold
        // reads every shard's maxima, and only the sum of the scored counts crosses back.
        auto select = [&](traversal_context_t& first) noexcept {
            byte_t* const device_row_offsets = first_outputs;
            byte_t* const device_nominee_offsets = first_outputs + row_offsets_bytes;
            byte_t* const partials = device_nominee_offsets + nominee_offsets_bytes +
                                     aligned(entries_ends[0] * sizeof(vram_gathered_row_t));
            byte_t* const keys = partials + shards * partials_bytes;
            f64_t* const scores = reinterpret_cast<f64_t*>(keys + keys_bytes);
            std::span<std::size_t> const scored_on_device {
                reinterpret_cast<std::size_t*>(keys + keys_bytes + scores_bytes_of_nominees), queries_count};
            std::size_t* const total = reinterpret_cast<std::size_t*>(keys + keys_bytes + scores_bytes_of_nominees +
                                                                      scored_bytes);
            std::span<u32_t const> const offsets_of_rows {reinterpret_cast<u32_t const*>(device_row_offsets),
                                                          row_offsets.size()};
            // The nominee offsets stay writable: the top-k tape takes them as its own.
            std::span<std::size_t> const offsets_of_nominees {reinterpret_cast<std::size_t*>(device_nominee_offsets),
                                                              queries_count + 1};
            status_t status = status_t::success_k;
            if (nominees_count)
                status = shards_[0].index->launch_maxsim_fold_(
                    nk::matrix_span<dot_result_t> {reinterpret_cast<dot_result_t*>(partials), nominees_count, longest},
                    partials_bytes, static_cast<u32_t>(shards), offsets_of_rows, offsets_of_nominees,
                    {scores, nominees_count}, first);
            vram_products_tape_t const nominated {offsets_of_nominees.data(), keys, nullptr};
            vram_products_tape_t const tape {matches.offsets.data(), reinterpret_cast<byte_t*>(matches.keys.data()),
                                             reinterpret_cast<byte_t*>(matches.scores.data())};
            if (succeeded(status))
                status = shards_[0].index->launch_top_k_({scores, nominees_count}, nominated,
                                                         config.wanted ? config.wanted : capacity, tape,
                                                         scored_on_device, first);
            if (succeeded(status)) status = shards_[0].index->launch_sum_counts_(scored_on_device, total, first);
            if (succeeded(status))
                status = fanout_t::vendor_t::copy(first.api(), scored.data(), scored_on_device.data(),
                                                  scored.size_bytes(), first.stream());
            return succeeded(status) ? first.copy_matched(total) : status;
        };
        if (status_t const selected = fanout.await_every_shard(widest_rescore_bytes, rescore_shard, select).status;
            failed(selected))
            return {walked.value, selected};
        walked.value.count = fanout.traversal(0).matched();
        return walked;
    }

    /** The body of the batched @c search into rankings on the first shard's device, when all three
     *  live there: every shard resolves its walks into keys on its own device, every further one
     *  copies its ranking to the first, and one merge there writes the caller's, rank for rank as
     *  the host merge would. @c deferred_k when they live elsewhere. */
    template <typename score_type_, typename... heads_types_>
    expected<search_result_t> search_on_device_(fanout_t& fanout, rows_view_t queries, search_config_t config,
                                                nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                                std::span<std::size_t> counts, heads_types_ const&... heads) noexcept {
        int const first_device = shards_[0].index->device_specs().ordinal;
        cuda_api_t const& api = shards_[0].index->api();
        if (cuda_device_of(api, keys.data()) != first_device || cuda_device_of(api, distances.data()) != first_device ||
            cuda_device_of(api, counts.data()) != first_device)
            return {{}, status_t::deferred_k};
        std::size_t const queries_count = queries.rows, shards = fanout.shards();
        // Every caller of `search_window_` counts as many matches as the keys are wide.
        u32_t const width = config.count;
        std::size_t const ids_bytes = queries_count * width * sizeof(node_id_t);
        std::size_t const walked_bytes = ids_bytes + queries_count * width * sizeof(distance_t);
        std::size_t const counts_bytes = queries_count * sizeof(std::size_t);
        std::size_t const scores_bytes = queries_count * width * sizeof(score_type_);
        std::size_t const block_bytes = round_up_to_multiple<std::size_t>(
            counts_bytes + scores_bytes + queries_count * width * sizeof(key_t), sizeof(std::size_t));
        // The first shard's staging holds every shard's ranking, none when a single shard resolves
        // straight into the caller's, and ends with the word the counts sum into.
        std::size_t const first_blocks = shards == 1 ? 0 : shards;
        std::size_t const first_outputs_bytes = walked_bytes + first_blocks * block_bytes + sizeof(std::size_t);
        auto ranking_of = [&](byte_t* block) noexcept {
            return vram_ranking_t {block + counts_bytes + scores_bytes,
                                   block + counts_bytes,
                                   reinterpret_cast<std::size_t*>(block),
                                   width * sizeof(key_t),
                                   width * sizeof(score_type_),
                                   width,
                                   sizeof(key_t),
                                   sizeof(score_type_)};
        };
        vram_ranking_t const merged {reinterpret_cast<byte_t*>(keys.data()),
                                     reinterpret_cast<byte_t*>(distances.data()),
                                     counts.data(),
                                     static_cast<std::size_t>(keys.stride_bytes(0)),
                                     static_cast<std::size_t>(distances.stride_bytes(0)),
                                     width,
                                     sizeof(key_t),
                                     sizeof(score_type_)};

        // Every shard writes its walks, then its own ranking, into the outputs it carves from its
        // stream's run, the first shard every shard's ranking; a single shard resolves straight
        // into the caller's.
        byte_t* first_outputs = nullptr;
        traversal_context_t* first_traversal = nullptr;
        auto search_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            byte_t* const outputs = traversal.allocator().allocate(shard ? walked_bytes + block_bytes
                                                                         : first_outputs_bytes);
            if (!outputs) return status_t::bad_alloc_k;
            if (!shard) first_outputs = outputs, first_traversal = &traversal;
            node_id_t* const ids = reinterpret_cast<node_id_t*>(outputs);
            distance_t* const walked_distances = reinterpret_cast<distance_t*>(outputs + ids_bytes);
            expected<search_result_t> const found = shards_[shard].index->submit_search_(
                queries, nk::matrix_span<node_id_t> {ids, queries_count, width},
                nk::matrix_span<distance_t> {walked_distances, queries_count, width}, traversal, config, visible,
                heads...);
            if (!found) return found.status;
            byte_t* const block = outputs + walked_bytes;
            if (status_t const resolved = shards_[shard].index->launch_resolve_matches_(
                    ids, walked_distances, width, reinterpret_cast<byte_t const*>(shards_[shard].index->device_keys_()),
                    shards == 1 ? merged : ranking_of(block), static_cast<u32_t>(queries_count), traversal);
                failed(resolved) || !shard)
                return resolved;
            // On the shard's own stream, so the copy follows its resolution; unified addressing
            // tells the two devices apart. The first shard's staging may have grown on its stream,
            // which its context recorded when it queued, so following that mark orders the copy
            // after the growth without the host waiting.
            return succeeded(traversal.follow(*first_traversal)) &&
                           cuda_copy_async(api, first_outputs + walked_bytes + shard * block_bytes, block, block_bytes,
                                           traversal.stream()) == cuda_success_k
                       ? status_t::success_k
                       : status_t::device_error_k;
        };
        // The first shard's stream waits on every other shard's event before this, so the merge
        // reads every ranking, and only the counts' sum crosses back, 8 B into a pinned slot.
        auto merge = [&](traversal_context_t& first) noexcept {
            if (shards != 1)
                if (status_t const merging = shards_[0].index->launch_merge_matches_(
                        ranking_of(first_outputs + walked_bytes), block_bytes, static_cast<u32_t>(shards), merged,
                        static_cast<u32_t>(queries_count), first);
                    failed(merging))
                    return merging;
            std::size_t* const total = reinterpret_cast<std::size_t*>(first_outputs + walked_bytes +
                                                                      first_blocks * block_bytes);
            if (status_t const summed = shards_[0].index->launch_sum_counts_(counts, total, first); failed(summed))
                return summed;
            return first.copy_matched(total);
        };
        expected<search_result_t> found = fanout.await_every_shard(first_outputs_bytes, search_shard, merge);
        if (!found) return found;
        found.value.count = fanout.traversal(0).matched();
        return found;
    }

    /** The gate every mutator holds shared for one call and every destructive window takes alone,
     *  through @c try_lock, so a window never waits; readers never touch it. */
    using shared_mutex_t = shared_mutex<wait_type_, fu::standard_atomic_ref>;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    /** The shards, the keys of their rows, their routing, the commit word every reader snapshots by
     *  and the epoch every reader brackets itself with. */
    shards_t shards_;

    /** The metric every shard this index builds or accepts scores by. */
    nk_kernel_kind_t kernel_;

    /** The dimensions and connectivity every shard keeps, fixed at construction. */
    geometry_t geometry_ {};

    /** Shared by every mutator for one call, alone by every destructive window; never swapped. */
    alignas(default_alignment_k) shared_mutex_t gate_;
};

/**
 *  @brief A reader's walk over the entries of one @c vram_nvidia_indexes as its shards stood at
 *      open: every entry as stored, shard after shard, or breadth-first from the start entries,
 *      each entry once, each with the keys it links to.
 *
 *  Every call names the index it walks, which must be the one the cursor opened on. One shard
 *  cursor per shard copies the neighbor rows of one block of nodes to the host whenever a seek
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
    using node_id_t = typename index_t::node_id_t;
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
         *  - @c bad_alloc_k The staging found no memory.
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
         *  - @c device_error_k The copy failed.
         */
        status_t seek(index_t const& shard, node_id_t node) noexcept {
            if (node >= visible_count_) return status_t::malformed_k;
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
            return status_t::success_k;
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

        /** The shard's watermark at open or the last refresh: IDs strictly below it are walked. */
        node_id_t visible_count_ = 0;
    };

    explicit vram_nvidia_indexes_cursor(forkunion_allocator_t allocator) noexcept
        : shard_cursors_(allocator), walk_(allocator), allocator_(allocator) {}

    /**
     *  @brief Opens the walk over every shard of @p indexes under the epoch it reads now; any
     *      @p starts make it breadth-first, each key starting from its lowest node in the lowest
     *      shard holding it, and a key held nowhere starting nothing.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The cursors, their staging, the visited bits or the frontier got no memory.
     */
    status_t open(indexes_t const& indexes, std::span<key_t const> starts) noexcept {
        epoch_ = indexes.read_epoch();
        std::size_t const count = indexes.shards_count();
        if (failed(shard_cursors_.reserve_exactly(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard) {
            shard_cursors_.push_back_reserved(shard_cursor_t {allocator_});
            if (status_t const opened = shard_cursors_.back().open(indexes.shard(shard)); failed(opened)) return opened;
        }
        if (starts.empty()) return status_t::success_k;
        order_ = cursor_order_t::breadth_first_k;
        growing_buffer<std::size_t, forkunion_allocator_t> visible {allocator_};
        if (failed(visible.resize_uninitialized(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard) visible[shard] = shard_cursors_[shard].visible_count();
        if (status_t const opened = walk_.open(to_span(visible)); failed(opened)) return opened;
        for (key_t const& key : starts)
            for (std::size_t shard = 0; shard != count; ++shard) {
                node_id_t const visible_count = shard_cursors_[shard].visible_count();
                node_id_t lowest = visible_count;
                indexes.shard(shard).find_each(key, [&](node_id_t node) noexcept { lowest = min_of(lowest, node); });
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
     *  - @c device_error_k A shard's rows did not copy; the entries written stand.
     *  - @c contended_k A destructive window of @p indexes opened since the walk did; none stand.
     */
    expected<std::size_t> pull(indexes_t const& indexes, std::span<key_t> keys, links_keys_t* links_keys,
                               links_offsets_t* links_offsets) noexcept {
        std::size_t written = 0;
        shard_node_t entry;
        while (written != keys.size() && next_(indexes, entry)) {
            shard_cursor_t& cursor = shard_cursors_[entry.shard];
            index_t const& shard = indexes.shard(entry.shard);
            if (status_t const sought = cursor.seek(shard, entry.node); failed(sought))
                return indexes.still_valid(epoch_) ? expected<std::size_t> {written, sought}
                                                   : expected<std::size_t> {0, status_t::contended_k};
            std::span<node_id_t const> const neighbors = cursor.neighbors();
            if (links_keys && (failed(links_keys->reserve(links_keys->size() + neighbors.size())) ||
                               failed(links_offsets->reserve(links_offsets->size() + 1))))
                return {written, status_t::bad_alloc_k};
            for (node_id_t const neighbor : neighbors) {
                // Rows come raw: a self-loop links nowhere, the sentinels fail both bounds below.
                if (neighbor == entry.node) continue;
                // Breadth-first never refreshes, so the bits span the shard cursors' snapshots.
                if (order_ == cursor_order_t::breadth_first_k && neighbor < cursor.visible_count())
                    if (status_t const visited = walk_.visit(entry.shard, neighbor); failed(visited))
                        return {written, visited};
                if (!links_keys) continue;
                expected<key_t> const linked = shard.key_of(neighbor);
                if (succeeded(linked.status)) links_keys->push_back_reserved(linked.value);
            }
            if (links_keys) links_offsets->push_back_reserved(links_keys->size());
            keys[written++] = shard.key_of(entry.node).value;
        }
        // Emptied shards keep their storage, so a pull that ran past a window read nothing freed.
        if (!indexes.still_valid(epoch_)) return {0, status_t::contended_k};
        return {written, status_t::success_k};
    }

  private:

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
                shard_cursors_[next_stored_.shard].refresh(indexes.shard(next_stored_.shard));
        }
        return false;
    }

    /** One per shard, in placement order, each over the snapshot it took at open. */
    growing_buffer<shard_cursor_t, forkunion_allocator_t> shard_cursors_;

    /** The nodes visited and the frontier of a breadth-first walk; untouched by a stored one. */
    walk_t walk_;

    forkunion_allocator_t allocator_;
    cursor_order_t order_ = cursor_order_t::stored_k;

    /** As stored: the entry the next pull starts at. */
    shard_node_t next_stored_ {};

    /** The epoch the walk opened under; a pull after a window since answers contended. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::dense
