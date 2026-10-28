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
 *  so shards with colliding ID spaces land in one shared ranking. The shards and their keys are a
 *  @c shard_set, checkpoints ride @c shard_save_plan and @c shard_load_plan, and every batched read
 *  fans out through a @c vram_shard_fanout; the device keys, the CUDA calls and the drivers are
 *  this tower's own.
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

#include <bit>          // `std::bit_ceil` - the key staging's halves
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
#include "usearch/utilities/vram_shards.hpp"       // `vram_shard_fanout` - one batched read over the shards
#include "usearch/dense/types.hpp"
#include "usearch/dense/shard_checkpoints.hpp"
#include "usearch/dense/vram_nvidia_index.hpp"

namespace unum::usearch::dense {

template <typename vram_indexes_type_, typename ram_indexes_type_>
struct vram_nvidia_compact_plan;

/**
 *  @brief The sharded index: same-geometry GPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are expressible.
 *  Each shard's keys live twice: on the host, where the merges resolve them, and on its device,
 *  where a walk resolves its nodes and the on-device merge reads them.
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

    /** The shard type - geometry parameters in the shard's own order, so the two never disagree. */
    using index_t = vram_nvidia_index<vector_dtype_, kernels_type_, cache_hints_type_>;

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;

    /** Every shard's host key column, where the merges resolve its nodes. */
    using keys_column_t = shard_keys_column<index_t, key_t, fu::standard_atomic_ref>;

    /** One shard's page-locked ring between its two key columns: two halves of keys, each refilled
     *  only once the copy out of it landed. Sized as plans cover rows and freed with the shard. */
    struct key_staging_t {
        growing_buffer<key_t, cuda_pinned_allocator_t> keys;

        /** Marks the last copy out of each half. */
        cuda_event_t crossed[2];

        /** Pins through @p api and births both events on the current device. */
        explicit key_staging_t(cuda_api_t const& api) noexcept
            : keys {cuda_pinned_allocator_t {api}}, crossed {cuda_event_t {api}, cuda_event_t {api}} {}

        /** Halves filled so far, alternating between the two. */
        std::size_t chunks = 0;
    };

    /** One shard's keys, held with the shard so a shard generation opens them beside it and no
     *  search reads a base that moves: the host column, its device twin, and the staging between
     *  them. */
    struct device_shard_keys_t {

        /** The keys on the host, where the merges resolve them. */
        keys_column_t host;

        /** The same keys on the shard's device, so a walk resolves its nodes to keys; reserved for
         *  every ID a shard can hold, since address space is free and the base never moves under a
         *  walk, and mapped as it grows. */
        cuda_virtual_memory_allocator<key_t> device;

        /** The ring the host column crosses into the device twin through, its events born on the
         *  shard's device. */
        key_staging_t staging;

        /** Bytes the host column and the device twin's mapped rows occupy. */
        std::size_t memory_usage() const noexcept { return host.memory_usage() + device.committed_bytes(); }

        /** The host column, which the shared set and checkpoint code reach. */
        friend keys_column_t& host_keys(device_shard_keys_t& keys) noexcept { return keys.host; }
        friend keys_column_t const& host_keys(device_shard_keys_t const& keys) noexcept { return keys.host; }
    };

    /** The shards, the keys of their rows, the rotation that routes new rows, and the commit word a
     *  reader spanning the shards snapshots by. */
    using shards_t =
        shard_set<index_t, device_shard_keys_t, key_t, key_hash_type_, fu::standard_atomic_ref, wait_type_>;

    /** One batched read's fan-out over every shard's device. */
    using fanout_t = utilities::vram_shard_fanout<shards_t, cuda_vendor_t, typename index_t::distance_t>;
    using key_locator_t = typename shards_t::key_locator_t;
    using shard_t = typename shards_t::shard_t;
    using shard_generation_t = typename shards_t::shard_generation_t;

    /** The streams and their scratch every bulk verb runs on, paired positionally with the
     *  shards. */
    using executor_t = cuda_executor_t;
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

    /** The NVIDIA capabilities the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = kernels_type_::capability_k;

    /** The vendor whose devices the shards sit on, beside the vendors' twins of this index. */
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;

    /** Rows one window of a batch verb stages on a shard at most: one slab, the granularity
     *  checkpoints move by, so a batch of any size crosses in bounded staging. */
    inline static constexpr std::size_t window_rows_k = power_of_two<std::size_t>(index_t::batch_rows_log2_k);

#pragma region Placement

    /** Every index names the metric its shards score by and the host allocator its shard table, key
     *  columns, locator map and checkpoint catalogs sit on; nothing is placed by default. */
    vram_nvidia_indexes() = delete;
    explicit vram_nvidia_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : shards_(allocator), kernel_(kernel) {}
    vram_nvidia_indexes(vram_nvidia_indexes&& other) noexcept
        : shards_(other.shards_.allocator()), kernel_(other.kernel_) {
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
     *  Appends one shard the caller already built, consuming it, with its host key column, a device
     *  twin and the staging between them on its device, where the staging's events are born.
     *  @ref reserve closes the shard set.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p shard is null, walks by another kernel, has a frozen geometry
     *    differing from the shards already here, or is already reserved; or the set is reserved,
     *    retired shards, or has a shard generation in flight.
     */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard || shard->kernel() != kernel_ ||
            (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity())))
            return status_t::config_mismatch_k;
        int const device = shard->device_specs().ordinal;
        cuda_api_t const& api = shard->api();
        if (cuda_set_device(api, device) != cuda_success_k) return status_t::device_error_k;
        if (status_t const placed = shards_.emplace(
                std::move(shard),
                device_shard_keys_t {keys_column_t {shards_.allocator()},
                                     cuda_virtual_memory_allocator<key_t> {api, device}, key_staging_t {api}});
            failed(placed))
            return placed;
        // Every shard is still unreserved, so each maps its slabs for every device placed so far.
        std::bitset<gpu_devices_capacity_k> const devices = devices_();
        for (shard_t& one : shards_) one.index->share_with_(devices);
        return status_t::success_k;
    }

    /**
     *  Empties every shard in place, keeping its placement, geometry and capacity, and every shard
     *  generation past the first as idle shards the next ones reuse; a kept failure drops with the
     *  rows. Takes the gate alone and closes every shard first; a search it lands under retries.
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

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.entries(); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** The first shard's kept failure, or success; asked of the shards, never cached. It is the
     *  failure an insertion refused with after rows launched on a shard, kept until @ref clear. */
    status_t status() const noexcept {
        for (shard_t const& one : shards_)
            if (status_t const kept = one.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }

    /** Bytes the index occupies - every shard's storage, its host and device key columns, and the
     *  shared locator map; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept { return sizeof(vram_nvidia_indexes) + shards_.memory_usage(); }

    /**
     *  Grows every shard ahead of @p config.capacity rows, split evenly over them, and both key
     *  columns of each with it - a hint, since all of them grow on demand; the first call freezes
     *  the geometry.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c unknown_kind_k A shard's kits cannot score its metric.
     *  - @c config_mismatch_k A shard's neighbors live on another device, its dimensions are zero,
     *    its connectivity is not a power of two, its geometry differs from the frozen one, or no
     *    walk fits it.
     *  - @c overflow_k A shard's share passes its ID space, its reserved range or its host key
     *    column.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        auto const [reserved, status] = shards_.reserve(config);
        if (failed(status)) return {reserved, status};
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (status_t const covered = cover_device_keys_(shard, shards_[shard].index->capacity()); failed(covered))
                return {reserved, covered};
        return {reserved, status_t::success_k};
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
              config_(other.config_), batch_split_(other.batch_split_), result_(other.result_), status_(other.status_),
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
         *  Settles every launch, publishes every row at once, and reports the batch's telemetry. A
         *  plan short of rows refuses and stays whole, so the missing steps can still land; any
         *  other call spends it, and a failed launch or drain publishes nothing and keeps its
         *  failure on every shard holding launched rows.
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

            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                for (std::size_t row = 0; row != shares_[shard].launched; ++row) {
                    std::size_t const placed = batch_split_.open_shard_generation.first + shard;
                    node_id_t const node = static_cast<node_id_t>(shares_[shard].first_id + row);
                    owner.shards_.locators().insert(seats_, owner.shards_[placed].keys.host[node],
                                                    key_locator_t {static_cast<u32_t>(placed), node});
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

        /** Writes the keys of the step's rows the plan's shard @p shard takes, from @p first on in
         *  strides of the shard generation's width, then launches those rows there. */
        status_t launch_(std::size_t shard, std::size_t first, nk::vector_view<key_t> keys, rows_view_t rows) noexcept {
            std::size_t const width = batch_split_.open_shard_generation.width, count = rows.rows;
            std::size_t const taking = shards_t::entries_from(count, first, width);
            share_t& share = shares_[shard];
            if (!taking) return status_t::success_k;
            // Overlapping steps would write keys past the cover the plan grew.
            if (share.launched + taking > share.rows) return status_t::config_mismatch_k;
            std::size_t const placed = batch_split_.open_shard_generation.first + shard;
            shard_t& one = owner_->shards_[placed];
            traversal_context_t& traversal = traversals_[placed];
            if (status_t const entered = fanout_t::enter(owner_->shards_, placed); failed(entered)) return entered;
            node_id_t const first_node = static_cast<node_id_t>(share.first_id + share.launched);
            for (std::size_t taken = 0; taken != taking; ++taken)
                one.keys.host[first_node + static_cast<node_id_t>(taken)] = keys[first + taken * width];
            // On the launch's own stream, so the device keys land before the rows they name
            // publish.
            if (status_t const uploaded = owner_->upload_keys_(placed, {first_node, first_node + taking},
                                                               traversal.stream());
                failed(uploaded))
                return uploaded;
            expected<insert_result_t> const inserted = one.index->submit_insert_(
                {rows.row(first), taking, rows.columns, rows.stride * width, rows.dtype}, first_node, {}, traversal,
                config_);
            if (!inserted) return inserted.status;
            share.launched += taking;
            result_ += inserted.value;
            return status_t::success_k;
        }

        /** Drains every context and adds what each spent to @p result, since the launches report
         *  their telemetry only once landed; answers the plan's status, or the first failed
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

        /** The locator seats every row's key takes in @c finish. */
        typename shards_t::locators_map_t::seats_t seats_ {};

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
     *      every shard's rows from its watermark, grows every shard's storage and key columns to
     *      them, and claims the locator seats.
     *
     *  A batch crossing the bound tops the open shard generation up and opens as many fresh shard
     *  generations as its other rows fill, each shard built with its keys on the device its
     *  placement names, so it lands whole; a racing batch taking the same room first leaves this
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
     *  - @c overflow_k The seats pass the locator chunks, a shard's rows pass its ceiling, its
     *    reserved range or its key columns, or @ref status keeps it.
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

        if (status_t const readied = fanout_t::reserve_traversals(
                shards_, plan.traversals_, executor, min_of(divide_round_up(rows, open.width), window_rows_k), config);
            failed(readied))
            return {{}, readied};
        auto [seats, seat_status] = shards_.locators().claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
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
            std::size_t const claimed_rows = plan.shares_[shard].first_id + plan.shares_[shard].rows;
            if (!plan.shares_[shard].rows) continue;
            if (status_t const entered = fanout_t::enter(shards_, placed); failed(entered)) return {{}, entered};
            reserve_capacity_t const grown {claimed_rows, static_cast<connectivity_t>(connectivity()), dimensions()};
            if (status_t const reserved = shards_[placed].index->reserve(grown).status; failed(reserved))
                return {{}, reserved};
            if (status_t const covered = shards_[placed].keys.host.cover(claimed_rows); failed(covered))
                return {{}, covered};
            if (status_t const covered = cover_device_keys_(placed, claimed_rows); failed(covered))
                return {{}, covered};
        }
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Inserts a batch under @p keys, splitting rows across the shards round-robin from a
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
     *  - @c overflow_k The seats pass the locator chunks, a shard's rows pass its ceiling, its
     *    reserved range or its key columns, or @ref status keeps it.
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
     *  - @c overflow_k The seat passes the locator chunks, a shard's rows pass its ceiling, its
     *    reserved range or its key columns, or @ref status keeps it.
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
     *  devices work side by side. An absorb's plan first lands its donors, one after another, and
     *  only then opens the shards' passes over the parts they appended. The executor's streams
     *  must outlive the plan.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb ever fills it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : arenas_(allocator), traversals_(allocator), shard_plans_(allocator), part_bases_(allocator),
              part_offsets_(allocator), landings_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : arenas_(std::move(other.arenas_)), traversals_(std::move(other.traversals_)),
              shard_plans_(std::move(other.shard_plans_)), part_bases_(std::move(other.part_bases_)),
              part_offsets_(std::move(other.part_offsets_)), landings_(std::move(other.landings_)),
              donors_(std::exchange(other.donors_, {})), landed_donors_(other.landed_donors_), config_(other.config_),
              population_(other.population_), owner_(std::exchange(other.owner_, nullptr)) {}
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
         *  Lands the donors still pending first, as @ref plan_absorb describes; a failing one stays
         *  pending and closed, with every later one, for the next call, and a landed one stays
         *  empty, closed and sealed. The executor is held only for the call.
         *
         *  @param[in,out] executor The executor the plan was opened over.
         *  @param[in] on_visited Fires per re-selecting phase with the nodes visited so far and the
         *      population; a @c cancel_k answer stops before the next phase.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or finished, or @ref status keeps it.
         *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, a
         *    donor's rows never land, or @ref status keeps it.
         *  - @c contended_k Another call holds the @p executor or a donor's gate, or a shard has a
         *    batch or load in flight.
         *  - @c overflow_k The landed rows pass a shard's ceiling, reserved range, key columns or
         *    the locator chunks, or @ref status keeps it.
         *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (status_t const paired = fanout_t::pairs_with(owner_->shards_, executor); failed(paired)) return paired;
            auto [claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; landed_donors_ != donors_.size(); ++landed_donors_) {
                vram_nvidia_indexes*& donor = donors_[landed_donors_];
                if (status_t const landed = owner_->absorb_(*donor, landings_of_(), executor, *this); failed(landed))
                    return landed;
                donor = nullptr;
            }
            if (shard_plans_.empty())
                if (status_t const opened = open_shards_(); failed(opened)) return opened;
            return drive_(on_visited);
        }

        /**
         *  Stops any re-selection, audits and heals, reopens the claim words, and reports. A plan
         *  whose donors have not all landed heals nothing and reports nothing.
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
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                    if (status_t const kept = shard_plans_[shard].finish(traversals_[shard]).status;
                        failed(kept) && succeeded(healed))
                        healed = kept;
            }
            refine_result_t const result = running();
            shard_plans_.clear();
            release_();
            return {result, healed};
        }

        /** The totals so far, summed over the shards' passes from the counters and telemetry the
         *  last phase landed: what @ref finish reports once it healed, and zeros while donors are
         *  pending. */
        refine_result_t running() const noexcept {
            refine_result_t result;
            for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard)
                result += shard_plans_[shard].finish(traversals_[shard]).value;
            return result;
        }

      private:
        friend vram_nvidia_indexes;

        /** Reopens this index and every donor still pending; the plan is unbound after. */
        void release_() noexcept {
            std::exchange(owner_, nullptr)->shards_.reopen();
            for (vram_nvidia_indexes* pending : donors_.subspan(landed_donors_)) pending->shards_.reopen();
            donors_ = {};
        }

        /** The landings of the first pending donor: donors land in order, so the shards of every
         *  later one close the landings. */
        std::span<landing_t const> landings_of_() const noexcept {
            std::size_t later = 0;
            for (std::size_t donor = landed_donors_ + 1; donor != donors_.size(); ++donor)
                later += donors_[donor]->shards_.size();
            std::size_t const count = donors_[landed_donors_]->shards_.size();
            return to_span(landings_).subspan(landings_.size() - later - count, count);
        }

        /** Opens every shard's pass over its parts on its own context, once no donor is pending. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != traversals_.size(); ++shard) {
                std::span<node_id_t const> parts;
                // A refine re-selects every row; a landing only the rows appended after the
                // resident part.
                iota_view<node_id_t> nodes {0, node_id_unknown_k};
                if (!part_offsets_.empty()) {
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
                    nodes.first = parts.size() > 1 ? parts[1] : node_id_unknown_k;
                }
                if (status_t const entered = fanout_t::enter(owner_->shards_, shard); failed(entered))
                    return shard_plans_.clear(), entered;
                auto [plan, status] = owner_->shards_[shard].index->plan_refine(nodes, traversals_[shard], config_,
                                                                                parts);
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

        /** Every shard's own pass in placement order; none until the donors landed. */
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** The parts the landings appended, ascending per shard, @c node_id_unknown_k past the
         *  last. */
        growing_buffer<node_id_t, forkunion_allocator_t> part_bases_;

        /** Where each shard's run of @c part_bases_ starts, and one past the last; empty with no
         *  donors. */
        growing_buffer<std::size_t, forkunion_allocator_t> part_offsets_;

        /** One landing per donor shard, in landing order, as @ref plan_absorb packed them; empty
         *  with no donors. */
        growing_buffer<landing_t, forkunion_allocator_t> landings_;

        /** The caller's donor slots, each closed at planning and nulled once landed. */
        std::span<vram_nvidia_indexes*> donors_;

        /** Donors landed so far. */
        std::size_t landed_donors_ = 0;

        /** The passes' knobs. */
        refine_config_t config_ {};

        /** The rows published once the donors landed - the population @c on_visited reports. */
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
     *  @param[in,out] executor The streams and their scratch.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
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
     *      this index, one donor after another: donor shards, largest first, each append to the
     *      shard with the fewest rows that keeps it under @ref shard_capacity, one on its own
     *      device on ties, with its edges rebased and its keys beside it, as a part of its own;
     *      when none has room, a shard generation opens on the same devices.
     *
     *  Each donor lands through one plan that does every fallible thing first: the donor's gate, a
     *  load claiming all the rows each receiving shard takes, the key slots and the locator seats.
     *  Every donor's committed storage maps for this index's devices at planning, so a donor shard
     *  then crosses device to device, one column at a time on the receiving shard's stream, its
     *  keys beside them; every part lands before any shard publishes, and only then do the keys
     *  resolve. The donor then forgets its rows in place inside its window, keeping its shards and
     *  their storage, its slot nulls, it stays closed and sealed, consumed, and its shards land
     *  among the receiving shards' parts in the plan, so at most one donor's rows are ever held
     *  twice.
     *
     *  Once the last donor landed, the plan's re-selection walks every shard across the parts the
     *  landings appended, round by round; a cancel pauses only that walk. This index and every
     *  donor close here, each under its own gate, so the packing never goes stale: this index stays
     *  closed while the plan lives and each donor until it lands, a batch, load or save in flight
     *  on any of them refuses the plan with @c contended_k, and an insertion into any of them
     *  arriving meanwhile is refused. Searches of this index may not overlap the plan's
     *  re-selection, whose merge rewrites neighbors in place; searches of a donor run throughout,
     *  retrying once around the window that empties it. An index never reserved takes the first
     *  donor's geometry.
     *
     *  @param[in,out] donors The indexes giving their rows up, each nulled once consumed; the slots
     *      must outlive the landings.
     *  @param[in,out] executor The streams every @c run and @c finish names, driving every device
     *      of every index.
     *  @param[in] config The refine's knobs; zeros mean "derive from connectivity".
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
    expected<refine_plan_t> plan_absorb(std::span<vram_nvidia_indexes*> donors, executor_t& executor,
                                        refine_config_t config = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        vram_nvidia_indexes const* shaped = capacity() ? this : nullptr;
        for (vram_nvidia_indexes* donor : donors) {
            if (!donor || donor == this || donor->kernel() != kernel()) return {{}, status_t::config_mismatch_k};
            if (status_t const kept = donor->status(); failed(kept)) return {{}, kept};
            if (status_t const paired = fanout_t::pairs_with(donor->shards_, executor); failed(paired))
                return {{}, paired};
            if (!donor->capacity()) continue;
            if (!shaped) shaped = donor;
            else if (donor->dimensions() != shaped->dimensions() || donor->connectivity() != shaped->connectivity())
                return {{}, status_t::config_mismatch_k};
        }
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
        // Closed, a donor commits no slab meanwhile, so this index's devices join its mapped ones.
        std::bitset<gpu_devices_capacity_k> const devices = devices_();
        for (vram_nvidia_indexes* donor : donors)
            for (shard_t& one : donor->shards_)
                if (status_t const granted = one.index->grant_(devices); failed(granted)) return {{}, granted};
        if (!donors.empty()) {
            std::size_t sources = 0;
            for (vram_nvidia_indexes const* donor : donors) sources += donor->shards_.size();
            if (failed(plan.landings_.resize(sources))) return {{}, status_t::bad_alloc_k};
            auto source_shard = [&](std::size_t from) noexcept -> shard_t const& {
                std::size_t donor = 0;
                for (; from >= donors[donor]->shards_.size(); ++donor) from -= donors[donor]->shards_.size();
                return donors[donor]->shards_[from];
            };
            auto device_of = [](shard_t const& shard) noexcept { return shard.index->device_specs().ordinal; };
            // Only a donor with rows needs a shard generation, so the geometry to take exists.
            auto open = [&]() noexcept {
                vram_nvidia_indexes const& like = shaped ? *shaped : *this;
                if (!capacity())
                    if (status_t const reserved =
                            reserve({0, static_cast<connectivity_t>(like.connectivity()), like.dimensions()}).status;
                        failed(reserved))
                        return reserved;
                // This index is sealed, so the shard generation opens closed like the rest, past
                // the seal.
                return shards_.open_shard_generation_past_seal_(
                    shards_.open_shard_generation(),
                    [this](shard_t const& built) noexcept { return build_shard_(built); });
            };
            if (status_t const packed = shards_.pack(to_writable_span(plan.landings_), 0, shards_.shard_capacity(),
                                                     source_shard, device_of, rank_fewest_entries, open);
                failed(packed))
                return {{}, packed};
            if (status_t const sized = shards_t::size_parts(to_span(plan.landings_), shards_.size(), plan.part_offsets_,
                                                            plan.part_bases_);
                failed(sized))
                return {{}, sized};
        }
        // The landings may have opened a shard generation, whose shards take part like the others.
        std::size_t const landed_shards = donors.empty() ? shards : plan.part_offsets_.size() - 1;
        if (failed(plan.shard_plans_.reserve_exactly(landed_shards))) return {{}, status_t::bad_alloc_k};
        if (status_t const opened = open_walks_(plan, executor, shaped ? *shaped : *this, landed_shards,
                                                donors.empty() ? 1u : 2u);
            failed(opened))
            return {{}, opened};
        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  Moves every row of each of @p donors into this index and links them in: @ref plan_absorb,
     *  then @c run and @c finish. @p on_visited answers as in @ref refine, for the re-selection
     *  alone; a cancel still heals and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k This index has no shards, or it or a donor keeps it, as its
     *    @ref status says.
     *  - @c config_mismatch_k A null donor, another metric or geometry, a device the @p executor
     *    does not drive, a donor device one of this index's devices cannot reach, this index
     *    itself, a geometry or walk context a shard refuses, or donor rows that never land; or
     *    this index or a donor keeps it.
     *  - @c contended_k A call, batch, load or save in flight on this index or a donor, a
     *    repeated donor, another refine or absorb holding either, or another call holding the
     *    @p executor.
     *  - @c unsupported_k A donor shard past @ref shard_capacity, or this index or a donor keeps
     *    it.
     *  - @c overflow_k A fresh shard generation still has no room for a donor shard, the landed
     *    rows pass a shard's ceiling, reserved range or key columns or the locator chunks, or
     *    this index or a donor keeps it.
     *  - @c unsupported_layout_k This index or a donor keeps it, as its @ref status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> absorb(std::span<vram_nvidia_indexes*> donors, executor_t& executor,
                                     refine_config_t config = {}, progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_absorb(donors, executor, config);
        if (failed(status)) return {{}, status};
        status_t const ran = plan.run(executor, on_visited);
        auto const [result, finish_status] = plan.finish(executor);
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /**
     *  Moves every row of @p donor into this index, as the plural @ref absorb does with one
     *  donor.
     *
     *  @par Statuses:
     *  - @c uninitialized_k This index has no shards, or it or @p donor keeps it, as its
     *    @ref status says.
     *  - @c config_mismatch_k The @p donor is this index, has another metric or geometry, or
     *    sits on a device the @p executor does not drive or one of this index's devices cannot
     *    reach; a shard refuses the geometry or its walk context, the donor's rows never land,
     *    or this index or @p donor keeps it.
     *  - @c contended_k A call, batch, load or save in flight on this index or @p donor, another
     *    refine or absorb holding either, or another call holding the @p executor.
     *  - @c unsupported_k A shard of @p donor past @ref shard_capacity, or this index or
     *    @p donor keeps it.
     *  - @c overflow_k A fresh shard generation still has no room for a shard of @p donor, the
     *    landed rows pass a shard's ceiling, reserved range or key columns or the locator
     *    chunks, or this index or @p donor keeps it.
     *  - @c unsupported_layout_k This index or @p donor keeps it, as its @ref status says.
     */
    expected<refine_result_t> absorb(vram_nvidia_indexes& donor, executor_t& executor,
                                     refine_config_t config = {}) noexcept {
        vram_nvidia_indexes* donors[1] = {&donor};
        return absorb(std::span<vram_nvidia_indexes*>(donors), executor, config);
    }

    /**
     *  Splits this index into receivers, the inverse of @c plan_absorb: a gap on the device
     *  residencies, while the host residency defines its own.
     *
     *  @par Statuses:
     *  - @c unsupported_k Always, as the device residencies split nothing.
     */
    expected<refine_plan_t> plan_split(std::span<vram_nvidia_indexes*>, executor_t&, refine_config_t = {},
                                       u64_t = 0) noexcept {
        return {{}, status_t::unsupported_k};
    }

    /**
     *  Moves every row of this index into one graph-local part per receiver and links each in:
     *  @ref plan_split, then @c run and @c finish. @p on_visited answers as in @ref refine, for the
     *  re-selection alone; a cancel still heals and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c unsupported_k Always, as @ref plan_split refuses.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> split(std::span<vram_nvidia_indexes*> receivers, executor_t& executor,
                                    refine_config_t config = {}, u64_t seed = 0,
                                    progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_split(receivers, executor, config, seed);
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
    expected<health_result_t> health(executor_t& executor) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<health_result_t> {
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
                    if (status_t const landed = plans[shard].wait(traversals[shard]); failed(landed))
                        return {{}, landed};
                    running |= plans[shard].running();
                }
            }
            health_result_t result;
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
     *  @param[in,out] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in,out] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' key columns, so colliding ID spaces still land in one ranking.
     *  @param[in,out] distances Distances of the same ranking, parallel to @p keys.
     *
     *  The batched search over one row, so every shard walks at once and the rankings merge as
     *  they land; each shard keeps as many matches as the ranking is wide, walking as broad as its
     *  expansion factor, or broader where the shard raises it to cover that width, as the batched
     *  search does. A failing shard keeps whatever the others already merged, so the reported
     *  count and the ranking always agree.
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
    expected<search_result_t> search(vector_view_t query_vector, executor_t& executor, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            if (shards_.empty()) return {{}, status_t::uninitialized_k};
            if (!query_vector.is_contiguous()) return {{}, status_t::unsupported_layout_k};
            std::size_t const width = min_of(keys.size(), distances.size());
            if (!width) return {{}, fanout_t::pairs_with(shards_, executor)};
            config.expansion_factor = static_cast<typename index_t::small_size_t>(
                shards_[0].index->non_zero_expansion_factor(config));
            config.count = static_cast<typename index_t::small_size_t>(width);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(1, config); failed(begun)) return {{}, begun};
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
     *  window, lands its matches in its own columns of one host landing, and the landed columns
     *  merge query by query through the key columns.
     *
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts, imported on
     *      each shard's device.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
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
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c config_mismatch_k The @p executor drives no stream on some shard's device, @p counts
     *    is not one per query, the row counts or widths of @p keys and @p distances disagree, or
     *    the queries are not as wide as the index.
     *  - @c unsupported_layout_k A row of @p keys or @p distances is not contiguous, or query rows
     *    sit closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k The queries' scalar has no device cast into the stored one.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                     std::span<std::size_t> counts) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            if (status_t const refusal = fanout_t::refusal(shards_, executor, queries.rows, keys, distances, counts);
                failed(refusal) || !queries.rows)
                return {{}, refusal};
            config.count = static_cast<typename index_t::small_size_t>(keys.extent(1));
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(min_of(queries.rows, window_rows_k), config); failed(begun))
                return {{}, begun};
            auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
                nk::range_t const rows = nk::range(first, first + taking);
                return search_window_(fanout, queries.slice(first, taking), config, keys(rows, nk::slice),
                                      distances(rows, nk::slice), counts.subspan(first, taking));
            };
            return for_each_window<search_result_t>(queries.rows, window_rows_k, search_window);
        });
    }

    /**
     *  The batched @c search ranked by a lightning indexer: every shard walks the queries under the
     *  same @p heads, passed to it as is, so whichever shard scored them, the merged distances are:
     *  <tt>1 − Σₕ wₕ · ReLU(qₕ · k)</tt>.
     *  A dot verb.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the @p executor drives
     *    no stream on some shard's device, @p counts is not one per query, the row counts or
     *    widths of @p keys and @p distances disagree, the heads or weights disagree with the
     *    queries, or the queries are not as wide as the index.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c unsupported_layout_k A row of @p keys or @p distances is not contiguous, or query or
     *    head rows sit closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k A scan is asked, or the queries' or heads' scalar has no device cast
     *    into the stored one.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept
        requires floating_scalar<scalar_t>
    {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = fanout_t::refusal(shards_, executor, queries.rows, keys, distances, counts);
                failed(refusal) || !queries.rows)
                return {{}, refusal};
            std::size_t const heads_count = heads.weights.extent(1);
            if (heads.weights.extent(0) != queries.rows || heads.heads.extent(0) != queries.rows * heads_count)
                return {{}, status_t::config_mismatch_k};
            config.count = static_cast<typename index_t::small_size_t>(keys.extent(1));
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(min_of(queries.rows, window_rows_k), config, heads); failed(begun))
                return {{}, begun};
            auto search_window = [&](std::size_t first, std::size_t taking) noexcept {
                nk::range_t const rows = nk::range(first, first + taking);
                indexer_heads<scalar_t> const window_heads {
                    heads.heads(nk::range(first * heads_count, (first + taking) * heads_count), nk::slice),
                    heads.weights(rows, nk::slice)};
                return search_window_(fanout, queries.slice(first, taking), config, keys(rows, nk::slice),
                                      distances(rows, nk::slice), counts.subspan(first, taking), window_heads);
            };
            return for_each_window<search_result_t>(queries.rows, window_rows_k, search_window);
        });
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row
     *      keeps the pairs its walks scored, keys resolved and dots at their native width.
     *
     *  @param[in] left The left-hand rows, one walk per row, in the shards' stored scalar type, as
     *      in @c search.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
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
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the @p executor drives
     *    no stream on some shard's device, @p counts is not one per row, the row counts or widths
     *    of @p keys and @p products disagree, or the rows are not as wide as the index.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c unsupported_layout_k A row of @p keys or @p products is not contiguous, or rows sit
     *    closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k A scan is asked, or the rows' scalar has no device cast into the stored
     *    one.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<product_type_> products,
                                       std::span<std::size_t> counts) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            std::size_t const rows_count = left.rows, wanted = keys.extent(1);
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
                return {{}, paired};
            if (counts.size() != rows_count) return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = rankings_refusal(rows_count, keys, products); failed(refusal))
                return {{}, refusal};
            if (!rows_count) return {{}, status_t::success_k};
            if (left.columns != dimensions()) return {{}, status_t::config_mismatch_k};

            // Every shard lands its own columns of one host landing, as tall as a window and reused
            // by every one, and its context is sized for that width.
            config.count = static_cast<typename index_t::small_size_t>(wanted);
            std::size_t const landing_rows = min_of(rows_count, window_rows_k);
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(landing_rows, config); failed(begun)) return {{}, begun};
            std::size_t const shards = fanout.shards(), landing_entries = landing_rows * shards * wanted;
            growing_buffer<node_id_t, forkunion_allocator_t> ids_landing {shards_.allocator()};
            growing_buffer<dot_result_t, forkunion_allocator_t> products_landing {shards_.allocator()};
            growing_buffer<u32_t, forkunion_allocator_t> counts_landing {shards_.allocator()};
            if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
                failed(products_landing.resize_uninitialized(landing_entries)) ||
                failed(counts_landing.resize_uninitialized(shards * landing_rows)))
                return {{}, status_t::bad_alloc_k};

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
                auto multiply_shard = [&](std::size_t shard, traversal_context_t& traversal,
                                          std::size_t visible) noexcept {
                    return shards_[shard]
                        .index
                        ->submit_multiply_(window_left, ids_wide(nk::all, shard_columns(shard), nk::slice),
                                           products_wide(nk::all, shard_columns(shard), nk::slice), shard_counts(shard),
                                           traversal, config, visible)
                        .status;
                };
                // Shards append as they land until a row fills, so a full row keeps whichever pairs
                // came first.
                auto merge_shard = [&](std::size_t shard, traversal_context_t const&) noexcept {
                    nk::matrix_span<node_id_t> const shard_ids = ids_wide(nk::all, shard_columns(shard), nk::slice);
                    nk::matrix_span<dot_result_t> const shard_products = products_wide(nk::all, shard_columns(shard),
                                                                                       nk::slice);
                    std::span<u32_t> const scored_counts = shard_counts(shard);
                    for (std::size_t row = 0; row != taking; ++row) {
                        std::size_t const kept = min_of(counts[first + row], wanted);
                        std::size_t const scored = scored_counts[row];
                        for (std::size_t pair = 0; pair != min_of(scored, wanted - kept); ++pair) {
                            keys(first + row, kept + pair) = shards_[shard].keys.host[shard_ids(row, pair)];
                            products(first + row, kept + pair) = shard_products(row, pair);
                        }
                        counts[first + row] += scored;
                    }
                };
                return fanout.for_each_shard(multiply_shard, merge_shard, counts.subspan(first, taking));
            };
            return for_each_window<search_result_t>(rows_count, window_rows_k, multiply_window);
        });
    }

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key among the tokens it ranked, and those maxima sum to that key's
     *      score.
     *
     *  @param[in] queries The token rows of every query, @p rows_per_query consecutive rows each.
     *  @param[in] rows_per_query Rows in every query; the row count must be a whole multiple of it.
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly; each row ranks as
     *      many tokens as its expansion factor, and one wider than the device search can land
     *      refuses.
     *  @param[out] keys The best-scoring keys of each query, one query per row, as many keys as its
     *      width.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, at most the width of @p keys.
     *  @param[out] scored Keys scored per query, before the width of @p keys cut any.
     *
     *  A key some row never ranked misses that row's maximum, so with non-negative per-row maxima a
     *  score lands below its exact value, never above. Whole queries cross @c window_rows_k rows at
     *  a time, or one query when it outgrows that, and a refused window keeps every one already
     *  landed.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the @p executor drives
     *    no stream on some shard's device, @p counts or @p scored is not one per query, the row
     *    counts or widths of @p keys and @p scores disagree, or the rows are not as wide as the
     *    index.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c shape_mismatch_k The @p rows_per_query is zero or does not divide the rows.
     *  - @c unsupported_layout_k A row of @p keys or @p scores is not contiguous, or rows sit
     *    closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     *  - @c unsupported_k The rows' scalar has no device cast into the stored one.
     */
    template <typename score_type_>
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<score_type_> scores, std::span<std::size_t> counts,
                                     std::span<std::size_t> scored) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            std::size_t const rows_count = queries.rows;
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
                return {{}, paired};
            if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
            std::size_t const groups_count = rows_count / rows_per_query;
            if (counts.size() != groups_count || scored.size() != groups_count)
                return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = rankings_refusal(groups_count, keys, scores); failed(refusal))
                return {{}, refusal};
            if (!groups_count) return {{}, status_t::success_k};

            // Every token row keeps all of its ranked candidates, its keys resolved and merged
            // across the shards; a window holds whole queries, so none splits across two.
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
            if (status_t const begun = fanout.begin(landing_rows, config); failed(begun)) return {{}, begun};
            auto maxsim_window = [&](std::size_t first, std::size_t taking) noexcept {
                nk::matrix_span<key_t> const ranked_keys {row_keys.data(), taking, width};
                nk::matrix_span<distance_t> const ranked_distances {row_distances.data(), taking, width};
                std::span<std::size_t> const ranked_counts = to_writable_span(row_counts).first(taking);
                expected<search_result_t> ranked = search_window_(fanout, queries.slice(first, taking), config,
                                                                  ranked_keys, ranked_distances, ranked_counts);
                if (!ranked) return ranked;
                std::size_t const first_group = first / rows_per_query, groups = taking / rows_per_query;
                nk::range_t const window_groups = nk::range(first_group, first_group + groups);
                ranked.value.count = reduce(ranked_keys, ranked_distances, ranked_counts,
                                            keys(window_groups, nk::slice), scores(window_groups, nk::slice),
                                            counts.subspan(first_group, groups), scored.subspan(first_group, groups));
                return ranked;
            };
            return for_each_window<search_result_t>(rows_count, window_rows, maxsim_window);
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
     *  @param[in,out] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly; @c count plays no
     *      part.
     *  @return The query rows attended, and the traversal telemetry every shard spent.
     *
     *  Whole tokens cross @c window_rows_k rows at a time, and a refused window keeps every one
     *  already landed.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The shards score by another metric than dot, the @p executor drives
     *    no stream on some shard's device, @p outputs or @p carry disagree with the rows, the
     *    queries are not as wide as the index, or the heads' scratch fits no walk.
     *  - @c uninitialized_k The index has no shards or was never reserved, or the @p executor has
     *    no streams.
     *  - @c shape_mismatch_k The @p head_count is zero or does not divide the rows, or a walk met
     *    a node with no value row.
     *  - @c unsupported_k A scan is asked, or the queries' or outputs' scalar has no device cast.
     *  - @c unsupported_layout_k Output or query rows sit closer than one packed row apart.
     *  - @c contended_k Another call holds the @p executor, or a destructive window overlapped
     *    the read twice.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, search_config_t config = {}) noexcept {
        return shards_.read_retrying_once([&]() noexcept -> expected<search_result_t> {
            using shard_lookup_t = vram_keyed_lookup<key_t, std::remove_cvref_t<value_lookup_type_>>;
            std::size_t const rows_count = queries.rows;
            if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
            if (status_t const paired = fanout_t::pairs_with(shards_, executor); paired != status_t::success_k)
                return {{}, paired};
            if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
            if (status_t const refusal = shards_[0].index->landing_refusal_(rows_count, outputs, carry);
                failed(refusal))
                return {{}, refusal};
            if (!rows_count) return {{}, status_t::success_k};
            std::size_t const dimensions = this->dimensions();
            if (queries.columns != dimensions) return {{}, status_t::config_mismatch_k};

            // A window holds whole tokens, so none splits its heads across two.
            std::size_t const window_rows = max_of<std::size_t>(window_rows_k / head_count, 1) * head_count;
            fanout_t fanout {shards_, executor};
            if (status_t const begun = fanout.begin(min_of(rows_count, window_rows), config, head_count,
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
                    shard_lookup_t const lookup {shards_[shard].keys.device.base(), value_lookup};
                    expected<f32_t*> const landed = shards_[shard].index->submit_attend_(
                        window_queries, head_count, lookup, score_scale, traversal, config, shard ? 1 : shards,
                        shard ? 0 : shards_[0].index->exported_bytes_(window_outputs), visible);
                    if (!landed) return landed.status;
                    if (!shard) {
                        blocks = reinterpret_cast<byte_t*>(landed.value);
                        first_traversal = &traversal;
                        return status_t::success_k;
                    }
                    // On the shard's own stream, so the copy follows its walks; unified addressing
                    // tells the two devices apart. Shard 0's staging may have grown on its stream,
                    // which its context marked when it queued, so following that mark orders the
                    // copy after it.
                    return succeeded(traversal.follow(*first_traversal)) &&
                                   cuda_copy_async(traversal.api(), blocks + shard * block_bytes, landed.value,
                                                   block_bytes, traversal.stream()) == cuda_success_k
                               ? status_t::success_k
                               : status_t::device_error_k;
                };
                status_t merged = status_t::success_k;
                std::size_t landed = 0;
                auto merge_shard = [&](std::size_t, traversal_context_t const& traversal) noexcept {
                    // A shard whose walk met a node without its value row folded a wrong softmax.
                    if (traversal.missed()) merged = status_t::shape_mismatch_k;
                    if (++landed != shards || failed(merged)) return;
                    // The last shard landed, so every block did; the merge and the landing ride
                    // shard 0's stream, which its context's event drains.
                    cuda_stream_handle_t const stream = first_traversal->stream();
                    u32_t const rows = static_cast<u32_t>(taking);
                    if (cuda_set_device(first_traversal->api(), first_traversal->device()) != cuda_success_k ||
                        failed(shards_[0].index->launch_merge_attention_(blocks, rows, static_cast<u32_t>(dimensions),
                                                                         static_cast<u32_t>(shards), stream)) ||
                        failed(shards_[0].index->land_attended_(reinterpret_cast<f32_t*>(blocks), shards,
                                                                window_outputs, window_carry, stream)) ||
                        failed(first_traversal->wait()))
                        merged = status_t::device_error_k;
                };
                expected<search_result_t> attended = fanout.for_each_shard(attend_shard, merge_shard, {});
                if (!attended) return attended;
                if (failed(merged)) return {{}, merged};
                attended.value.count = taking;
                return attended;
            };
            return for_each_window<search_result_t>(rows_count, window_rows, attend_window);
        });
    }

#pragma endregion Search

#pragma region Checkpoints

    /** What a device checkpoint does beyond the shared plans: it holds the executor's streams for
     *  the plan's whole life, every shard opens its own plan on its device's stream, a load's keys
     *  follow to the device twins, and a save closes the claim words, since device storage cannot
     *  be settlement-probed. */
    struct checkpoint_residency_t {

        /** The index whose device keys a load fills, and whose shard generations it opens. */
        vram_nvidia_indexes* owner = nullptr;

        /** The streams every shard plan queues on. */
        executor_t* executor = nullptr;

        /** The executor's hold, kept for the plan's whole life, one caller at a time. */
        typename executor_t::claim_t streams;

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
         *  Opens shard @p shard 's load of @p rows on the stream of its device and pins its bounce,
         *  since the parts' columns cross from host memory.
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
            auto [plan, status] = shards[shard].index->plan_load(
                rows, executor->stream(fanout_t::stream_of(shards, *executor, shard)));
            if (failed(status)) return {std::move(plan), status};
            // The parts' columns cross from host memory, through a bounce.
            status_t const pinned = plan.pin_bounce();
            return {std::move(plan), pinned};
        }

        /**
         *  Covers @p rows in shard @p shard 's device key column and uploads them from its host
         *  column on the stream of @p plan.
         *
         *  @par Statuses:
         *  - @c overflow_k The rows pass the device key column's range.
         */
        status_t land_keys(shards_t& shards, std::size_t shard, iota_view<std::size_t> rows,
                           typename index_t::load_plan_t const& plan) noexcept {
            if (status_t const entered = fanout_t::enter(shards, shard); failed(entered)) return entered;
            if (status_t const covered = owner->cover_device_keys_(shard, rows.bound); failed(covered)) return covered;
            return owner->upload_keys_(shard, rows, plan.stream);
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
     *  index. The row counts are read now, so a buffer sized here fits a save no row lands
     *  before. */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        return shard_storage_requirements(shards_, config);
    }

    /**
     *  @brief Opens a save under @p path: a lone live shard's monolith at the path itself, or one
     *      monolith per live shard beside the manifest the path names.
     *
     *  Every part is framed, sized and mapped here, and every header lands in @c finish, so a
     *  failed or cancelled save reads as absent.
     *
     *  @param[in] path The monolith's path, or the manifest's - its parts land beside it as
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
     *      and every part's header is vetted before a byte lands.
     *
     *  The reservation comes from @p config, and part @c p lands on live shard @c p, opened past
     *  the placed ones where the parts outnumber them. The index must be empty;
     *  the plan reports the rows it will land through @c rows().
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
     *  - @c overflow_k A part passes a shard's bound, a key column or the locator map, or
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
     *  - @c overflow_k A part passes a shard's bound, a key column or the locator map, or
     *    @ref status keeps it.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref
     *    status says.
     */
    expected<load_plan_t> plan_load(bytes_view_t buffer, load_config_t const& config, executor_t& executor) noexcept {
        auto [residency, status] = gate_load_(executor);
        if (failed(status)) return {{}, status};
        return load_plan_t::open(shards_, std::move(residency), buffer, config);
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
     *    rows, a shard's steps never landed all its rows, a header does not fit before its data,
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
        if (status_t const landed = drain_steps_(plan); failed(landed)) return landed;
        return plan.finish(on_saved);
    }

    /**
     *  Serializes the index into @p buffer - the one-shot over the bytes form of
     *  @ref plan_save.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p config asks for @c concurrent_k or does not validate, the
     *    @p executor drives no stream on some shard's device, a shard's columns cannot hold its
     *    rows, a shard's steps never landed all its rows, a header does not fit before its data,
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
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, a bounce was
     *    never pinned, or @ref status keeps it.
     *  - @c config_mismatch_k The index holds entries, the @p executor drives no stream on some
     *    shard's device, the checkpoint's geometry, dtypes or metric differ from the shards', a
     *    step cannot land its rows, a shard's steps never landed all its rows, or @ref status
     *    keeps it.
     *  - @c contended_k Another call holds the @p executor or the gate, or a batch or plan holds
     *    a shard's claims.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, a part, its header or its key column does
     *    not parse, or its connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound, a key column or the locator map, or
     *    @ref status keeps it.
     *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
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
     *  Restores the checkpoint held in @p buffer - the one-shot over the bytes form of
     *  @ref plan_load.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index has no shards or a shard was never reserved, a bounce was
     *    never pinned, or @ref status keeps it.
     *  - @c config_mismatch_k The index holds entries, the @p executor drives no stream on some
     *    shard's device, the checkpoint's geometry, dtypes or metric differ from the shards', a
     *    step cannot land its rows, a shard's steps never landed all its rows, or @ref status
     *    keeps it.
     *  - @c contended_k Another call holds the @p executor or the gate, or a batch or plan holds
     *    a shard's claims.
     *  - @c malformed_k The manifest lists no part, a part, its header or its key column does not
     *    parse, or its connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound, a key column or the locator map, or
     *    @ref status keeps it.
     *  - @c unsupported_k or @c unsupported_layout_k The index keeps it, as @ref status says.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t buffer, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(buffer, config, executor);
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

    /** Entries stored under @p key; a concurrent insertion can only make it under-report. Exact
     *  without an epoch: a compaction swaps each locator whole, and a clear empties the map inside
     *  its window. */
    std::size_t count(key_t const& key) const noexcept { return shards_.locators().count(key); }

    /** Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return shards_.locators().contains(key); }

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
        return shards_.read_until_settled([&]() noexcept {
            std::size_t const total = queue_rows_(key, vectors, wanted, executor);
            if (total) [[maybe_unused]]
                status_t const drained = executor.wait();
            return total;
        });
    }

    /**
     *  @brief The batched @c count: the entries stored under every key of @p keys.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *
     *  The executor is unused, since the locator map lives on the host; it rides in the
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
     *      true total, every row queued on its shard's stream and the executor drained once.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] rows @b [keys·wanted,dimensions]: row q · @p wanted + i is the i-th row of key
     *      q; refused as @ref rows_refusal says, and with @c unsupported_k in any scalar but the
     *      stored one.
     *  @param[in] wanted Rows kept per key.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted
     *      when its rows truncated, zero when absent.
     *  @param[in,out] executor The streams the copies ride, paired positionally with the shards.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p entries are not as many as the @p keys, @p rows hold other
     *    than @p keys · @p wanted rows or narrower ones, or @p executor drives no stream on some
     *    shard's device.
     *  - @c unsupported_k The @p rows are in another scalar than the stored one.
     *  - @c unsupported_layout_k A row's values are not packed.
     *  - @c uninitialized_k The index has no shards.
     *  - @c contended_k Another call holds @p executor, or a destructive window overlapped the
     *    read twice.
     */
    status_t get(std::span<key_t const> keys, rows_span_t rows, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (!rows.holds<scalar_t>()) return status_t::unsupported_k;
        vectors_span_t const vectors = rows.as<scalar_t>();
        std::size_t const row_values = divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
        if (status_t const refusal = rows_refusal(keys.size() * wanted, vectors, row_values); failed(refusal))
            return refusal;
        if (status_t const paired = fanout_t::pairs_with(shards_, executor); failed(paired)) return paired;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return shards_.read_retrying_once([&]() noexcept {
            for (std::size_t position = 0; position != keys.size(); ++position) {
                std::size_t const first = position * wanted;
                entries[position] = queue_rows_(keys[position], vectors(nk::range(first, first + wanted), nk::slice),
                                                wanted, executor);
            }
            return executor.wait();
        });
    }

#pragma endregion Lookups

  private:

    /** Every device a shard sits on, which each shard maps its storage for, so a load reads a shard
     *  on another device directly. */
    std::bitset<gpu_devices_capacity_k> devices_() const noexcept {
        std::bitset<gpu_devices_capacity_k> devices;
        for (shard_t const& one : shards_) devices[static_cast<std::size_t>(one.index->device_specs().ordinal)] = true;
        return devices;
    }

    /** One unreserved shard placed on @p device; null when the allocation fails. */
    std::unique_ptr<index_t> build_(cuda_api_t const& api, int device) const noexcept {
        typename index_t::vectors_allocator_t const allocator {api, device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(kernel(), allocator, allocator));
    }

    /** Forgets every published shard's rows in place, each keeping its storage, then every key's
     *  locators, keeping the shard generations past the first as idle shards: what a clear and an
     *  emptied donor become, inside their own window with every shard closed. */
    void clear_() noexcept {
        for (shard_t& one : shards_) one.index->forget_();
        shards_.forget_entries();
    }

    /** Closes every shard under the unique side, refusing a call in flight, a claim outstanding or
     *  a plan holding the index closed with @c contended_k; closing moves nothing a reader sees, so
     *  it moves no epoch. */
    status_t close_gated_() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        return held ? shards_.close() : status_t::contended_k;
    }

    /** One shard of a new shard generation and its keys, on the device @p like sits on, current
     *  while the staging's events are born, mapped for every device placed and reserved in the
     *  frozen geometry under the bound; a null index when the device, the allocation or the
     *  reservation fails. */
    shard_t build_shard_(shard_t const& like) const noexcept {
        int const device = like.index->device_specs().ordinal;
        cuda_api_t const& api = like.index->api();
        std::unique_ptr<index_t> index = cuda_set_device(api, device) == cuda_success_k ? build_(api, device) : nullptr;
        if (index) index->share_with_(devices_());
        reserve_capacity_t const geometry {0, static_cast<connectivity_t>(connectivity()), dimensions(),
                                           shards_.shard_capacity()};
        if (index && failed(index->reserve(geometry).status)) index.reset();
        return shard_t {std::move(index),
                        device_shard_keys_t {keys_column_t {shards_.allocator()},
                                             cuda_virtual_memory_allocator<key_t> {api, device}, key_staging_t {api}}};
    }

    /** Covers rows below @p rows in shard @p shard 's device key column, reserving it at its first
     *  cover, and widens the staging between the two columns to a power of two of rows, up to a
     *  window per half. */
    status_t cover_device_keys_(std::size_t shard, std::size_t rows) noexcept {
        cuda_virtual_memory_allocator<key_t>& device_keys = shards_[shard].keys.device;
        if (!device_keys.reserved_bytes())
            if (status_t const reserved = device_keys.reserve_range(index_t::max_node_count_k * sizeof(key_t));
                failed(reserved))
                return reserved;
        key_staging_t& staging = shards_[shard].keys.staging;
        std::size_t const half_keys = min_of(window_rows_k, std::bit_ceil(rows));
        if (rows && staging.keys.size() < 2 * half_keys) {
            // Pinning needs a current context, and the copies out of the old halves must land.
            if (failed(fanout_t::enter(shards_, shard)) || failed(staging.crossed[0].wait()) ||
                failed(staging.crossed[1].wait()))
                return status_t::device_error_k;
            staging.keys.reset();
            if (failed(staging.keys.resize_uninitialized(2 * half_keys))) return status_t::bad_alloc_k;
        }
        return device_keys.commit(rows * sizeof(key_t));
    }

    /** Copies @p rows of shard @p shard 's host key column into its device twin on @p stream,
     *  through the shard's staging half by half, since a pageable source would copy synchronously;
     *  the host refills a half only once the copy out of it landed. */
    status_t upload_keys_(std::size_t shard, iota_view<std::size_t> rows, cuda_stream_handle_t stream) noexcept {
        key_staging_t& staging = shards_[shard].keys.staging;
        std::size_t const half_keys = staging.keys.size() / 2;
        if (rows.empty()) return status_t::success_k;
        if (!half_keys) return status_t::uninitialized_k;
        key_t* const device_keys = shards_[shard].keys.device.base();
        for (std::size_t first = rows.first; first < rows.bound; first += half_keys) {
            std::size_t const bound = min_of(first + half_keys, rows.bound);
            std::size_t const half = staging.chunks++ % 2;
            key_t* staged = staging.keys.data() + half * half_keys;
            if (staging.chunks > 2 && failed(staging.crossed[half].wait())) return status_t::device_error_k;
            shards_[shard].keys.host.for_each_span({first, bound}, [&](std::span<key_t> run) noexcept {
                for (key_t const& key : run) *staged++ = key;
            });
            if (cuda_copy_async(shards_[shard].keys.device.api(), device_keys + first,
                                staging.keys.data() + half * half_keys, (bound - first) * sizeof(key_t),
                                stream) != cuda_success_k ||
                failed(staging.crossed[half].record(stream)))
                return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** Opens one run on its stream and one walk context over each of the first @p shards shards,
     *  the ones the plan spans, seeded for the widest shard of @p shaped as it stands, @p walks
     *  walks into one visited set; landed rows grow them. */
    status_t open_walks_(refine_plan_t& plan, executor_t& executor, vram_nvidia_indexes const& shaped,
                         std::size_t shards, unsigned walks) noexcept {
        if (failed(plan.arenas_.reserve_exactly(shards)) || failed(plan.traversals_.reserve_exactly(shards)))
            return status_t::bad_alloc_k;
        std::size_t walk_bytes = 0;
        for (shard_t const& one : shaped.shards_)
            walk_bytes = max_of(walk_bytes, one.index->traversal_memory_usage(plan.config_, walks));
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

    /** Lands every row of @p donor where @p landings, one per donor shard, put it and consumes it,
     *  under both indexes' closed claim words and the executor's hold @p refining keeps: the
     *  donor's gate and every other fallible step first, then the copies, one drain, and the
     *  publication, and the donor's shards land among the plan's parts as it forgets its rows in
     *  its window. */
    status_t absorb_(vram_nvidia_indexes& donor, std::span<landing_t const> landings, executor_t& executor,
                     refine_plan_t& refining) noexcept {
        std::size_t const shards = refining.traversals_.size();
        if (!donor.size()) return status_t::success_k;
        // The donor empties once its rows land, so its gate comes before anything a refusal undoes.
        unique_lock_t const emptying {donor.gate_, std::try_to_lock};
        if (!emptying) return status_t::contended_k;
        if (!capacity())
            if (status_t const reserved = reserve({donor.size(), static_cast<connectivity_t>(donor.connectivity()),
                                                   donor.dimensions()})
                                              .status;
                failed(reserved))
                return reserved;

        // Every shard claims all its rows first, so a failure before publication releases them all.
        std::size_t const donors = landings.size();
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        if (failed(plans.reserve_exactly(shards))) return status_t::bad_alloc_k;
        std::size_t landing = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = 0; from != donors; ++from)
                if (landings[from].into == into) rows += donor.shards_[from].index->size();
            if (status_t const entered = fanout_t::enter(shards_, into); failed(entered)) return entered;
            auto [plan, status] = shards_[into].index->plan_load(
                rows, executor.stream(fanout_t::stream_of(shards_, executor, into)));
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
            if (status_t const covered = shards_[into].keys.host.cover(plans[into].base + rows); failed(covered))
                return covered;
            if (status_t const covered = cover_device_keys_(into, plans[into].base + rows); failed(covered))
                return covered;
            landing += rows;
        }
        auto [seats, seat_status] = shards_.locators().claim(landing);
        if (failed(seat_status)) return seat_status;

        // A refused copy still waits for those queued before it, whose rows go to the next claim.
        status_t queued = status_t::success_k;
        for (std::size_t into = 0; into != shards && succeeded(queued); ++into) {
            shard_t& one = shards_[into];
            typename index_t::load_plan_t& plan = plans[into];
            node_id_t base = plan.base;
            for (std::size_t from = 0; from != donors && succeeded(queued); ++from) {
                if (landings[from].into != into) continue;
                shard_t const& part = donor.shards_[from];
                node_id_t const part_rows = static_cast<node_id_t>(part.index->size());
                if (!part_rows) continue;
                typename index_t::load_plan_t::step_t step;
                step.index_rows = {base, base + part_rows};
                step.source_rows = {0, part_rows};
                for (node_id_t row : step.source_rows) one.keys.host[base + row] = part.keys.host[row];
                queued = land_part_(into, plan, step, donor, from);
                base += part_rows;
            }
        }
        if (status_t const drained = executor.wait(); failed(queued) || failed(drained))
            return failed(queued) ? queued : drained;
        for (typename index_t::load_plan_t& plan : plans)
            if (status_t const published = plan.finish(); failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                shards_.locators().insert(seats, shards_[into].keys.host[row],
                                          key_locator_t {static_cast<u32_t>(into), row});

        // Each donor shard becomes a part of its own in the plan, which sized every shard's run.
        shards_.record_parts(donor.shards_, landings, plans, refining.part_bases_.data(),
                             to_span(refining.part_offsets_));
        donor.shards_.begin_window();
        donor.clear_();
        donor.shards_.end_window();
        return status_t::success_k;
    }

    /** Queues donor shard @p from 's columns by @p step of shard @p into 's load @p plan, then its
     *  device keys, all on the plan's stream: another device's keys may not map here, so they cross
     *  from the host column. */
    status_t land_part_(std::size_t into, typename index_t::load_plan_t& plan,
                        typename index_t::load_plan_t::step_t const& step, vram_nvidia_indexes const& donor,
                        std::size_t from) noexcept {
        index_t const& part = *donor.shards_[from].index;
        if (status_t const queued = plan.load(step, part, step.source_rows); failed(queued)) return queued;
        node_id_t const base = step.index_rows.first, rows = step.source_rows.bound;
        if (part.device_specs().ordinal != shards_[into].index->device_specs().ordinal)
            return upload_keys_(into, {base, base + rows}, plan.stream);
        return cuda_copy_async(shards_[into].keys.device.api(), shards_[into].keys.device.base() + base,
                               donor.shards_[from].keys.device.base(), rows * sizeof(key_t),
                               plan.stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Queues the copies of up to @p wanted rows stored under @p key into @p vectors, each on the
     *  stream of @p executor on its shard's device, skipping a shard it drives no stream for, and
     *  answers the true total; the caller drains once for every row it queued. */
    std::size_t queue_rows_(key_t const& key, vectors_span_t vectors, std::size_t wanted,
                            executor_t& executor) const noexcept {
        std::size_t total = 0;
        shards_.locators().find_each(key, [&](key_locator_t locator) noexcept {
            index_t const& shard = *shards_[locator.shard].index;
            std::size_t const stream = fanout_t::stream_of(shards_, executor, locator.shard);
            if (total < wanted && total < vectors.extent(0) && stream != executor.size() &&
                succeeded(fanout_t::enter(shards_, locator.shard))) [[maybe_unused]]
                status_t const queued = shard.vector(locator.node, vectors.row(total).as_vector(),
                                                     executor.stream(stream).handle);
            ++total;
        });
        return total;
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
     *  failing step ends it. A save lands each step, a load reads it straight out of its part. */
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

    /** One window of the batched @c search: on the first shard's device when the rankings live
     *  there and score in @c f32 or @c f64, through the host landing otherwise. */
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
        // Vetted before anything is carved; the shards would refuse only after the landing and the
        // runs had been sized for a doomed call.
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
        return fanout.search_every_shard(queries_count, config.count, search_shard, keys, distances, counts);
    }

    /** The body of the batched @c search into rankings on the first shard's device, when all three
     *  live there: every shard resolves its walks into keys on its own device, every further one
     *  copies its ranking to the first, and one merge there lands the caller's, rank for rank as
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

        // Every shard lands its walks, then its own ranking, in device memory of the call on its
        // own stream, the first shard every shard's ranking; a single shard resolves straight
        // into the caller's.
        growing_buffer<cuda_unique_ptr<byte_t, cuda_device_allocator_t>, forkunion_allocator_t> landings {
            shards_.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> landed_counts {shards_.allocator()};
        if (failed(landings.reserve_exactly(shards)) || failed(landed_counts.resize(queries_count)))
            return {{}, status_t::bad_alloc_k};
        traversal_context_t* first_traversal = nullptr;
        auto search_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            std::size_t const blocks = shards == 1 ? 0 : shard ? 1 : shards;
            landings.push_back_reserved(cuda_make_unique(cuda_device_allocator_t {fanout.executor().stream(
                                                             fanout_t::stream_of(shards_, fanout.executor(), shard))},
                                                         walked_bytes + blocks * block_bytes));
            byte_t* const landing = landings.back().get();
            if (!landing) return status_t::bad_alloc_k;
            if (!shard) first_traversal = &traversal;
            node_id_t* const ids = reinterpret_cast<node_id_t*>(landing);
            distance_t* const walked_distances = reinterpret_cast<distance_t*>(landing + ids_bytes);
            expected<search_result_t> const found = shards_[shard].index->submit_search_(
                queries, nk::matrix_span<node_id_t> {ids, queries_count, width},
                nk::matrix_span<distance_t> {walked_distances, queries_count, width}, traversal, config, visible,
                heads...);
            if (!found) return found.status;
            byte_t* const block = landing + walked_bytes;
            if (status_t const resolved = shards_[shard].index->launch_resolve_matches_(
                    ids, walked_distances, width, reinterpret_cast<byte_t const*>(shards_[shard].keys.device.base()),
                    shards == 1 ? merged : ranking_of(block), static_cast<u32_t>(queries_count), traversal.stream());
                failed(resolved) || blocks != 1)
                return resolved;
            // On the shard's own stream, so the copy follows its resolution; unified addressing
            // tells the two devices apart. The first shard's landing was allocated on its stream,
            // which its context marked when it queued, so following that mark orders the copy after
            // the allocation without the host waiting.
            return succeeded(traversal.follow(*first_traversal)) &&
                           cuda_copy_async(api, landings[0].get() + walked_bytes + shard * block_bytes, block,
                                           block_bytes, traversal.stream()) == cuda_success_k
                       ? status_t::success_k
                       : status_t::device_error_k;
        };
        status_t merged_status = status_t::success_k;
        std::size_t landed = 0;
        auto merge_shard = [&](std::size_t, traversal_context_t const&) noexcept {
            if (++landed != shards) return;
            // The last shard landed, so every ranking did; the merge rides shard 0's stream, which
            // its context's event drains, and the counts come back for the sum.
            cuda_stream_handle_t const stream = first_traversal->stream();
            if (cuda_set_device(api, first_traversal->device()) != cuda_success_k ||
                (shards != 1 && failed(shards_[0].index->launch_merge_matches_(
                                    ranking_of(landings[0].get() + walked_bytes), block_bytes,
                                    static_cast<u32_t>(shards), merged, static_cast<u32_t>(queries_count), stream))) ||
                cuda_copy_async(api, landed_counts.data(), counts.data(), counts.size_bytes(), stream) !=
                    cuda_success_k ||
                failed(first_traversal->wait()))
                merged_status = status_t::device_error_k;
        };
        expected<search_result_t> found = fanout.for_each_shard(search_shard, merge_shard, to_span(landed_counts));
        if (!found) return found;
        if (failed(merged_status)) return {{}, merged_status};
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

    /** Shared by every mutator for one call, alone by every destructive window; never swapped. */
    alignas(default_alignment_k) shared_mutex_t gate_;
};

} // namespace unum::usearch::dense
