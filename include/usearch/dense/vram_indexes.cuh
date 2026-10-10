/**
 *  @file include/usearch/dense/vram_indexes.cuh
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The sharded index over GPU-resident dense shards - one per device - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The multi-shard layer over @c dense_vram_index: same-geometry shards, each owned and pinned
 *  to one CUDA device, viewed as one index. Insertions rotate across the shards so they fill
 *  evenly and each holds a representative sample; searches scatter over every shard and gather
 *  through @ref merge_search_results, resolving matches into the caller's keys, so shards with
 *  colliding ID spaces land in one shared ranking. Checkpoints ride the SafeGraphs interchange -
 *  one consolidated monolith, or one per shard beside a manifest - written positionally with
 *  every header last, so a failed save reads as absent.
 *
 *  Beyond the executor's claim, the insertion gate admits one mutating batch at a time, a shard's
 *  kept failure refuses every later mutation, and the commit sequence word lets a reader spanning
 *  the shards see each batch whole or not at all.
 */
#pragma once

#include <cstring> // `std::memcmp` - the monolith's magic

#include <array>       // `std::array` - the monolith magic
#include <atomic>      // `std::atomic` - the routing tally
#include <memory>      // `std::unique_ptr` - an index holds atomics and never moves
#include <mutex>       // `std::lock_guard`, `std::adopt_lock` - the one-batch-at-a-time hold
#include <numbers>     // `std::numbers::log2e_v` - the base-2 flash merge
#include <span>        // `std::span` - the caller's keys & the merged ranking
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move` - handing a mapping to the buffer that outlives the load

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the shard table & key columns
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the host columns' placement
#include "usearch/utilities/ram_tables.hpp"        // `chunked_multimap` - the locator map
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths & manifests - the checkpoints
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_index.cuh"

namespace unum {
namespace usearch {

#if defined(__CUDACC__)

/** A key-level value lookup seen through one shard's device key column, so the shard's walks resolve node IDs. */
template <typename key_type_, typename value_lookup_type_>
struct dense_cuda_keyed_lookup {

    /** The shard's keys on its device, by node ID. */
    key_type_ const* keys;

    /** Resolves a key to its value row. */
    value_lookup_type_ value_lookup;

    __device__ __forceinline__ auto operator()(dense_node_id_t node_id) const noexcept {
        return value_lookup(keys[node_id]);
    }
};

/** Merges the normalized attended rows and carries of @p shards walks into the first's by the flash rule, a warp per
 *  row: each rescales to the joint maximum and weighs by its share of the joint partition. Every shard holds a block
 *  of @b [rows,columns] rows, then @b [rows] carries, the blocks lying back to back from @p blocks. */
static __global__ void dense_cuda_merge_attention_kernel(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards) {
    u32_t const row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    std::size_t const outputs_bytes = std::size_t {rows} * columns * sizeof(f32_t);
    std::size_t const block_bytes = outputs_bytes + rows * sizeof(dense_softmax_state_t);
    auto const carry_of = [&](u32_t shard) noexcept -> dense_softmax_state_t& {
        auto* const carries = reinterpret_cast<dense_softmax_state_t*>(blocks + shard * block_bytes + outputs_bytes);
        return carries[row];
    };
    auto const row_of = [&](u32_t shard) noexcept {
        return reinterpret_cast<f32_t*>(blocks + shard * block_bytes) + std::size_t {row} * columns;
    };
    f32_t maximum = carry_of(0).maximum_score, partition = 0;
    for (u32_t shard = 1; shard != shards; ++shard) maximum = fmaxf(maximum, carry_of(shard).maximum_score);
    // The carries keep natural-log maxima, so their gaps turn base-2 for the exponent.
    auto const weight_of = [&](u32_t shard) noexcept {
        dense_softmax_state_t const carry = carry_of(shard);
        return carry.partition * exp2f((carry.maximum_score - maximum) * std::numbers::log2e_v<f32_t>);
    };
    for (u32_t shard = 0; shard != shards; ++shard) partition += weight_of(shard);
    f32_t* const merged = row_of(0);
    for (u32_t shard = 0; shard != shards; ++shard) {
        f32_t const share = partition > 0 ? weight_of(shard) / partition : 0;
        f32_t const* const partial = row_of(shard);
        for (u32_t column = lane; column < columns; column += 32)
            merged[column] = (shard ? merged[column] : 0) + share * partial[column];
    }
    // Every lane reads the first carry above before the leader replaces it.
    __syncwarp();
    if (lane == 0) carry_of(0) = {maximum, partition};
}

/**
 *  @brief The sharded index: same-geometry GPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are
 *  expressible. Shards arrive one at a time through @ref emplace, so the index grows as devices
 *  are engaged rather than being declared whole. The shard's configs, views and results are
 *  re-exported verbatim, so a batched call and a single one read the same.
 */
template <                                               //
    typename key_type_ = uuid_t,                         //
    nk_dtype_t vector_dtype_ = nk_i8_k,                  //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k, //
    cuda_arch_t capability_ = cuda_arch_ampere_k,        //
    typename key_hash_type_ = std::hash<key_type_>       //
    >
struct dense_vram_indexes {

    /** The caller-visible identity, opaque to every shard - indexes store no keys, so the whole
     *  key vocabulary begins here. */
    using key_t = key_type_;

    /** Where one entry lives: the shard, and its shard-local ID - the index's only address for an
     *  entry. */
    struct key_locator_t {

        /** The shard holding the entry, positional in placement order. */
        std::size_t shard = 0;

        /** The entry's ID inside that shard's own space. */
        dense_node_id_t node = 0;
    };

    /** Key to every entry stored under it; insert-only, wait-free for readers, growing in chunks. */
    using locators_map_t = chunked_multimap<key_t, key_locator_t, key_hash_type_, forkunion_allocator_t>;

    /** The shard type - geometry parameters in the shard's own order, so the two never disagree. */
    using index_t = dense_vram_index<vector_dtype_, kernel_kind_, capability_>;
    static_assert(index_family<index_t>);

    /** Keys of one shard's rows, indexed by that shard's own node IDs; host-resident, growing in the chunks the CPU
     *  shards grow in. */
    using keys_column_t =
        chunked_column<key_t, chunk_schedule<10, 20>, index_t::max_node_count_k, forkunion_allocator_t>;

    /** The streams and their scratch every bulk verb runs on; it names no index, so one executor
     *  serves every instantiation. */
    using executor_t = cuda_executor_t;
    using traversal_t = typename index_t::dense_traversal_context_t;
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

    /** One shard: the graph on its device, and the keys of its rows on the host and on the device. Placement is not
     *  repeated: the index reports it through @c device_specs. */
    struct shard_t {

        /** The graph; an index holds atomics and never moves. */
        std::unique_ptr<index_t> index;

        /** The key of each of this shard's rows, indexed by node ID; a batch covers its rows before writing them. */
        keys_column_t keys;

        /** The same keys on the shard's device, so a walk resolves its nodes to keys; reserved for every ID a shard
         *  can hold, since address space is free and the base never moves under a walk, and mapped as it grows. */
        cuda_virtual_memory_allocator<key_t> device_keys;
    };

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The distance kernel this instantiation walks with. */
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;

    /** The compute-capability ceiling the kernels compiled under. */
    inline static constexpr cuda_arch_t capability_k = capability_;

#pragma region Placement

    /** Every index names the host allocator its shard table, key columns, locator map and
     *  checkpoint catalogs sit on; nothing is placed by default. */
    dense_vram_indexes() = delete;
    explicit dense_vram_indexes(forkunion_allocator_t allocator) noexcept
        : shards_(allocator), locators_of_key_(allocator), allocator_(allocator) {}
    dense_vram_indexes(dense_vram_indexes&& other) noexcept
        : shards_(other.allocator_), locators_of_key_(other.allocator_), allocator_(other.allocator_) {
        swap(other);
    }
    dense_vram_indexes& operator=(dense_vram_indexes&& other) noexcept { return swap(other), *this; }
    dense_vram_indexes(dense_vram_indexes const&) = delete;
    dense_vram_indexes& operator=(dense_vram_indexes const&) = delete;

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~dense_vram_indexes() noexcept = default;

    /** Exchanges two indexes whole; a move implies the caller holds both exclusively, so the
     *  insertion gate is not swapped along with the state it guards. */
    void swap(dense_vram_indexes& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(locators_of_key_, other.locators_of_key_);
        std::swap(allocator_, other.allocator_);
        std::swap(routed_rows_, other.routed_rows_);
    }

    /** Appends one shard built here on @p device, for callers with none to hand over. */
    status_t emplace(int device) noexcept {
        std::unique_ptr<index_t> shard = build_(device);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** Appends one shard the caller already built, consuming it; refuses any whose frozen geometry
     *  differs from the shards already here, any already reserved, and any whose device already
     *  holds one - the index shards, never replicates. @ref reserve closes the shard set. */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard) return status_t::config_mismatch_k;
        if (capacity()) return status_t::config_mismatch_k;
        if (shard->capacity()) return status_t::config_mismatch_k;
        if (!shards_.empty() && (shard->dimensions() != dimensions() || shard->connectivity() != connectivity()))
            return status_t::config_mismatch_k;
        for (shard_t const& one : shards_)
            if (one.index->device_specs().ordinal == shard->device_specs().ordinal) return status_t::config_mismatch_k;
        return shards_.push_back(shard_t {std::move(shard), keys_column_t {allocator_}, {}});
    }

    /** Empties every shard in place, keeping its device, geometry and capacity; a kept failure
     *  drops with the rows. Demands the same quiescence as @ref reserve. */
    status_t clear() noexcept {
        reserve_capacity_t const geometry {capacity(), static_cast<connectivity_t>(connectivity()), dimensions()};
        for (shard_t& one : shards_) {
            std::unique_ptr<index_t> fresh = build_(one.index->device_specs().ordinal);
            if (!fresh) return status_t::bad_alloc_k;
            one.index = std::move(fresh);
        }
        locators_of_key_.clear();
        routed_rows_ = 0;
        return geometry.capacity ? reserve(geometry).status : status_t::success_k;
    }

#pragma endregion Placement

#pragma region Measures and Reservation

    /** Shards in the index, one per engaged device. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }

    /** The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** The frozen geometry, taken from the first shard - @ref emplace refused any that differ. */
    std::size_t dimensions() const noexcept { return shards_.empty() ? 0 : shards_[0].index->dimensions(); }

    /** Connections per node, frozen with the geometry. */
    std::size_t connectivity() const noexcept { return shards_.empty() ? 0 : shards_[0].index->connectivity(); }

    /** Entries across every shard: each shard's own @c size, summed over one snapshot no batch straddles. */
    std::size_t size() const noexcept { return snapshot_({}); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept {
        std::size_t total = 0;
        for (shard_t const& one : shards_) total += one.index->capacity();
        return total;
    }

    /** The first shard's kept failure, or success; asked of the shards, never cached. */
    status_t status() const noexcept {
        for (shard_t const& one : shards_)
            if (status_t const kept = one.index->status(); failed(kept)) return kept;
        return status_t::success_k;
    }

    /** Bytes the index occupies - every shard's storage, its key columns, and the shared locator
     *  map; traversal scratch is the executor's. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(dense_vram_indexes) + locators_of_key_.memory_usage();
        for (shard_t const& one : shards_)
            bytes += one.index->memory_usage() + one.keys.memory_usage() + one.device_keys.committed_bytes();
        return bytes;
    }

    /** Grows every shard ahead of @p config.capacity rows, split evenly over them - a hint, since the shards, their
     *  key columns and the locator map all grow on demand; the first call freezes the geometry. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (shards_.empty()) return {0, status_t::uninitialized_k};
        reserve_capacity_t per_shard = config;
        per_shard.capacity = divide_round_up(config.capacity, shards_.size());
        for (shard_t& one : shards_) {
            auto const [shard_capacity, status] = one.index->reserve(per_shard);
            if (failed(status)) return {capacity(), status};
            if (status_t const covered = cover_keys_(one, shard_capacity); failed(covered))
                return {capacity(), covered};
        }
        if (status_t const indexed = locators_of_key_.reserve(config.capacity); failed(indexed))
            return {capacity(), indexed};
        return {capacity(), status_t::success_k};
    }

#pragma endregion Measures and Reservation

#pragma region Insertion

    /**
     *  @brief One batch in flight across the shards: the executor and the insertion gate held for the plan's whole
     *      life, one context per shard carved for the whole batch, every shard's planes and key columns grown to the
     *      rows it takes, and the locator seats claimed, so a step fails only on its devices.
     *
     *  Row @c t of the batch lands on shard @c (rotation+t) mod S. Steps run one after another on one host thread,
     *  each writing its rows' keys and launching the rows as deferred inserts; @ref finish drains them and turns
     *  every shard's watermark inside one commit, so a reader spanning the shards sees the batch whole or not at all.
     *  A failed launch or drain publishes nothing, and neither does a plan dying unfinished: every shard holding
     *  launched rows keeps the failure, so its rows stay invisible and later mutations are refused.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : traversals_(allocator), shares_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : traversals_(std::move(other.traversals_)), shares_(std::move(other.shares_)), seats_(other.seats_),
              config_(other.config_), rows_(other.rows_), rotation_(other.rotation_), status_(other.status_),
              streams_(std::move(other.streams_)), mutating_(std::move(other.mutating_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /** An unfinished plan drains what it launched and keeps @c cancelled_k on every shard holding it. */
        ~insert_plan_t() noexcept {
            if (owner_) [[maybe_unused]]
                status_t const abandoned = abandon_(status_t::cancelled_k);
        }

        /** Launches rows @b [first_row,first_row+rows) of the plan under @p keys on their shards; the rows and the
         *  keys stay live until @ref finish. Mismatched arguments launch nothing, any later refusal fails the plan. */
        status_t insert(std::size_t first_row, std::span<key_t const> keys, vectors_view_t rows) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            std::size_t const count = rows.extent(0);
            if (keys.size() != count || first_row > rows_ || count > rows_ - first_row ||
                (count && rows.extent(1) != owner_->dimensions()))
                return status_t::config_mismatch_k;
            if (!count) return status_t::success_k;
            std::size_t const shards = shares_.size();
            std::size_t const rotation = (rotation_ + first_row) % shards;
            for (std::size_t shard = 0; shard != shards && succeeded(status_); ++shard)
                status_ = launch_(shard, first_row_of_shard_(shard, shards, rotation), keys, rows);
            return status_;
        }

        /** Drains every launch, publishes every row at once, and reports the batch's telemetry. A plan short of rows
         *  refuses and stays whole, so the missing steps can still land; any other call spends it, and a failed
         *  launch or drain publishes nothing and keeps its failure on every shard holding launched rows. */
        expected<insert_result_t> finish() noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            std::size_t launched = 0;
            for (share_t const& share : shares_) launched += share.launched;
            if (succeeded(status_) && launched != rows_) return {{owner_->size()}, status_t::config_mismatch_k};
            status_t status = status_;
            for (traversal_t& traversal : traversals_)
                if (status_t const drained = traversal.drain(); failed(drained) && succeeded(status)) status = drained;
            // Every context was born zeroed for this plan alone, so its sink holds every launch on its shard.
            insert_result_t result;
            for (traversal_t const& traversal : traversals_) {
                dense_cuda_telemetry_block_t const spent = traversal.spent();
                result.visited_members += spent.hops;
                result.computed_distances += spent.scored;
                result.restarts += spent.restarts;
                result.unlinked_neighbors += spent.lock_skips;
                result.tail_drops += spent.tail_drops;
                result.visited_saturations += spent.visited_saturations;
            }
            dense_vram_indexes& owner = *owner_;
            if (failed(status)) {
                result.new_size = owner.size();
                return {result, abandon_(status)};
            }

            // Batches run one at a time under the gate, so no turn waits inside the commit.
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
                    dense_node_id_t const node = static_cast<dense_node_id_t>(shares_[shard].first_id + row);
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
            dense_node_id_t first_id = 0;

            /** Rows of the batch the shard takes. */
            std::size_t rows = 0;

            /** Rows launched on the shard so far, contiguous from @c first_id. */
            std::size_t launched = 0;
        };

        /** Writes the keys of the step's rows shard @p shard takes, from @p first on in strides of the shard count,
         *  then launches those rows there. */
        status_t launch_(std::size_t shard, std::size_t first, std::span<key_t const> keys,
                         vectors_view_t rows) noexcept {
            std::size_t const shards = shares_.size(), count = rows.extent(0);
            std::size_t const taking = rows_from_(count, first, shards);
            share_t& share = shares_[shard];
            if (!taking) return status_t::success_k;
            // Overlapping steps would write keys past the cover the plan grew.
            if (share.launched + taking > share.rows) return status_t::config_mismatch_k;
            shard_t& one = owner_->shards_[shard];
            traversal_t& traversal = traversals_[shard];
            if (cudaSetDevice(traversal.device()) != cudaSuccess) return status_t::device_error_k;
            dense_node_id_t const first_node = static_cast<dense_node_id_t>(share.first_id + share.launched);
            for (std::size_t taken = 0; taken != taking; ++taken)
                one.keys[first_node + static_cast<dense_node_id_t>(taken)] = keys[first + taken * shards];
            // On the launch's own stream, so the device keys land before the rows they name publish.
            if (status_t const uploaded = upload_keys_(one, {first_node, first_node + taking}, traversal.stream());
                failed(uploaded))
                return uploaded;
            expected<insert_result_t> const inserted = one.index->submit_insert_(
                rows(nk::range(first, count, shards), nk::slice), {}, traversal, config_);
            if (!inserted) return inserted.status;
            share.launched += taking;
            // A first ID other than the part's next one names a claimer outside the gate.
            return inserted.value.first_id == first_node ? status_t::success_k : status_t::contended_k;
        }

        /** Drains every context and keeps @p failure on every shard holding launched rows, spending the plan. */
        status_t abandon_(status_t failure) noexcept {
            for (traversal_t& traversal : traversals_) [[maybe_unused]]
                status_t const drained = traversal.drain();
            for (std::size_t shard = 0; shard != shares_.size(); ++shard)
                if (shares_[shard].launched) [[maybe_unused]]
                    status_t const kept = owner_->shards_[shard].index->fail_(failure);
            owner_ = nullptr;
            return failure;
        }

        /** One context per shard, carved for the whole batch. */
        growing_buffer<traversal_t, forkunion_allocator_t> traversals_;

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

        /** The first refusal past a step's arguments, which leaves the whole batch unpublished. */
        status_t status_ = status_t::success_k;

        /** The executor's hold, kept for the plan's whole life, one caller at a time. */
        typename executor_t::claim_t streams_;

        /** The insertion gate, held so batches never interleave their claims. */
        std::unique_lock<fu::spin_mutex_t> mutating_;

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and once spent. */
        dense_vram_indexes* owner_ = nullptr;

        friend dense_vram_indexes;
    };

    /**
     *  @brief Opens a batch of @p rows: takes the executor and the insertion gate, carves one context per shard,
     *      grows every shard's planes and key columns to the rows it takes, and claims the locator seats.
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
            plan.owner_ = this;
            return {std::move(plan), status_t::success_k};
        }
        // Two holds: the executor's guards the scratch, the index's keeps batches from interleaving claims.
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
        if (!mutating_.lock_if_free()) return {{}, status_t::contended_k};
        plan.mutating_ = std::unique_lock<fu::spin_mutex_t> {mutating_, std::adopt_lock};

        std::size_t const shards = shards_.size();
        if (status_t const readied = reserve_traversals_(plan.traversals_, executor, divide_round_up(rows, shards),
                                                         config);
            failed(readied))
            return {{}, readied};
        // Every shard must seat its build launch before any row is admitted, or a mixed fleet inserts partially.
        for (shard_t const& one : shards_)
            if (!one.index->build_scratch_fits()) return {{}, status_t::config_mismatch_k};
        auto [seats, seat_status] = locators_of_key_.claim(rows);
        if (failed(seat_status)) return {{}, seat_status};
        plan.seats_ = seats;
        // Rotating by rows routed keeps a stream of small batches from drifting onto shard zero.
        plan.rotation_ = standard_atomic_ref<std::size_t>(routed_rows_).fetch_add(rows, std::memory_order_relaxed) %
                         shards;
        if (failed(plan.shares_.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_t& one = shards_[shard];
            std::size_t const first_id = one.index->claimed_();
            std::size_t const taking = rows_from_(rows, first_row_of_shard_(shard, shards, plan.rotation_), shards);
            plan.shares_.push_back_reserved({static_cast<dense_node_id_t>(first_id), taking, 0});
            if (!taking) continue;
            if (cudaSetDevice(one.index->device_specs().ordinal) != cudaSuccess) return {{}, status_t::device_error_k};
            reserve_capacity_t const grown {first_id + taking, static_cast<connectivity_t>(connectivity()),
                                            dimensions()};
            if (status_t const reserved = one.index->reserve(grown).status; failed(reserved)) return {{}, reserved};
            if (status_t const covered = cover_keys_(one, first_id + taking); failed(covered)) return {{}, covered};
        }
        plan.owner_ = this;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Inserts a batch under @p keys, dealing rows round-robin across the shards from a
     *      rotating start, so they fill evenly and each sees the whole distribution: @ref plan_insert,
     *      one step over every row, and its @c finish.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row landed in
     *  is the index's business. The keys are what the caller gets its rows back by.
     *
     *  A shard failing past its claim keeps that failure; a first ID other than the predicted one
     *  names a claimer outside this index and refuses the call as @c contended_k.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_vectors The batch, one row per entry, in the shards' stored scalar type.
     *  @param[inout] executor The streams and their scratch, paired positionally with the shards.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, vectors_view_t new_vectors, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        if (keys.size() != new_vectors.extent(0)) return {{size()}, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(new_vectors.extent(0), executor, config);
        if (failed(status)) return {{size()}, status};
        // A refused step leaves rows short, and `finish` reports the refusal it kept or the shortfall.
        [[maybe_unused]] status_t const launched = plan.insert(0, keys, new_vectors);
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
     *  @brief Re-selects the neighbors of every stored node and heals whatever that strands, one shard at a time.
     *
     *  A shard's pass owns its device and drains its stream before returning, so the shards run one
     *  after another. A stopped pass leaves the graph correct and merely less improved, and the next
     *  call sweeps again toward the same fixed point. Insertions and searches may not overlap.
     *
     *  @param[inout] executor The streams and their scratch, one per shard.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] on_visited Fires per shard with the nodes swept so far and the population; a
     *      @c cancel_k answer stops before the next shard, reported as @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config = {},
                                     progress_type_&& on_visited = {}) noexcept {
        std::size_t const shards = shards_.size();
        if (!shards) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const paired = pairs_with_(executor); failed(paired)) return {{}, paired};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (!mutating_.lock_if_free()) return {{}, status_t::contended_k};
        std::lock_guard const mutating {mutating_, std::adopt_lock};

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, 1,
                                                       insert_config_t {config.expansion_factor});
            failed(ready))
            return {{}, ready};

        refine_result_t result;
        std::size_t const published = size();
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return {{}, status_t::device_error_k};
            auto const [part, status] = one.refine({0, dense_node_id_unknown_k}, traversals[shard], config);
            combine_refinement_(result, part);
            if (failed(status)) return {result, status};
            if (on_visited(result.visited_nodes, published) == continuation_t::cancel_k)
                return {result, status_t::cancelled_k};
        }
        return {result, status_t::success_k};
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

        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const ready = reserve_traversals_(traversals, executor, 1, insert_config_t {}); failed(ready))
            return {{}, ready};

        health_result_t result;
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return {{}, status_t::device_error_k};
            auto const [shard_health, status] = one.health(traversals[shard]);
            if (failed(status)) return {{}, status};
            combine_health_(result, shard_health);
        }
        return {result, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Search

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending
     *      by distance - the end-to-end multi-shard lookup.
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
     *  A failing shard stops the scatter but keeps whatever earlier shards already merged, so
     *  the reported count and the ranking always agree.
     */
    expected<search_result_t> search(vector_view_t query_vector, executor_t& executor, search_config_t config,
                                     std::span<key_t> keys, std::span<distance_t> distances) noexcept {
        search_result_t merged;
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {merged, paired};
        // The executor serializes, never the index: a second caller with its own executor searches concurrently.
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {merged, claim_status};
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const readied = reserve_traversals_(traversals, executor, 1, config); failed(readied))
            return {merged, readied};
        growing_buffer<std::size_t, forkunion_allocator_t> visible {allocator_};
        if (failed(visible.resize(shards_.size()))) return {merged, status_t::bad_alloc_k};
        snapshot_(to_writable_span(visible));
        status_t status = status_t::success_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            traversal_t& traversal = traversals[shard];
            if (cudaSetDevice(traversal.device()) != cudaSuccess) return {merged, status_t::device_error_k};
            auto const [search_result, search_status] = shards_[shard].index->search_(query_vector, traversal, config,
                                                                                      visible[shard]);
            if (failed(search_status)) {
                status = search_status;
                break;
            }
            merged.visited_members += search_result.visited_members;
            merged.computed_distances += search_result.computed_distances;
            merged.restarts += search_result.restarts;
            merged.tail_drops += search_result.tail_drops;
            merged.visited_saturations += search_result.visited_saturations;
            merged.count = merge_search_results(to_span(traversal.results), shards_[shard].keys, keys, distances,
                                                merged.count);
        }
        return {merged, status};
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard and gathers the rankings into one
     *      per query.
     *
     *  The executor-driven sibling of @c search, one launch per shard rather than one thread per
     *  query: every shard walks the full batch, lands its matches in its own columns of one host
     *  landing, and the drained columns merge query by query through the key columns.
     *
     *  @param[in] queries The batch, one query per row, in the shards' stored scalar type.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in] wanted Matches wanted per query - the merged rankings' width.
     *  @param[out] keys Merged rankings, @b [queries,wanted] row-major.
     *  @param[out] distances Distances of the same rankings, parallel to @p keys.
     *  @param[out] counts Per-query merged hit counts, one per query row.
     *  @return The merged hits summed over queries, and the telemetry summed over shards.
     *
     *  A refused shard stops the fan-out but keeps every shard already landed, so the rankings and
     *  the counts always agree; every queued stream is drained before returning, since the caller's
     *  queries and the call's own landing are DMA targets until then.
     */
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     std::size_t wanted, std::span<key_t> keys, std::span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept {
        return search_(queries, executor, config, wanted, keys, distances, counts);
    }

    /** The batched @c search ranked by a lightning indexer: every shard walks the queries under the same @p heads,
     *  passed to it as is, so the merged distances are 1 − Σₕ wₕ · ReLU(qₕ · k) whichever shard scored them. */
    expected<search_result_t> search(vectors_view_t queries, dense_indexer_heads<scalar_t> const& heads,
                                     executor_t& executor, search_config_t config, std::size_t wanted,
                                     std::span<key_t> keys, std::span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k && !nk::is_integral_dtype<scalar_t>())
    {
        return search_(queries, executor, config, wanted, keys, distances, counts, heads);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row keeps
     *      the pairs its walks scored, keys resolved and dots at their native width.
     *
     *  @param[in] left The left-hand rows, one walk per row, in the shards' stored scalar type.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] wanted Pairs kept per row - the row width of @p keys and @p products.
     *  @param[out] keys The key of each kept pair, @b [rows,wanted] row-major.
     *  @param[out] products The dot of each kept pair, parallel to @p keys.
     *  @param[out] counts Pairs scored per row across every shard, above @p wanted when truncated.
     *
     *  A full row keeps an arbitrary subset of its pairs. A refused shard stops the fan-out but keeps
     *  every shard already landed, so the pairs and the counts always agree; the first refusal is what
     *  the merged result reports.
     */
    expected<search_result_t> multiply(vectors_view_t left, executor_t& executor, search_config_t config,
                                       std::size_t wanted, std::span<key_t> keys, std::span<dot_result_t> products,
                                       std::span<std::size_t> counts) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        std::size_t const rows_count = left.extent(0);
        std::size_t const shards = shards_.size();
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (counts.size() != rows_count || keys.size() < rows_count * wanted || products.size() < rows_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (left.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};

        // Every shard lands its own columns of one host landing, and its context is sized for that width.
        config.count = static_cast<typename index_t::small_size_t>(wanted);
        std::size_t const landing_entries = rows_count * shards * wanted;
        growing_buffer<dense_node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<dot_result_t, forkunion_allocator_t> products_landing {allocator_};
        growing_buffer<u32_t, forkunion_allocator_t> counts_landing {allocator_};
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(products_landing.resize_uninitialized(landing_entries)) ||
            failed(counts_landing.resize_uninitialized(shards * rows_count)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<dense_node_id_t> const ids_wide {ids_landing.data(), rows_count, shards * wanted};
        nk::matrix_span<dot_result_t> const products_wide {products_landing.data(), rows_count, shards * wanted};

        auto multiply_shard = [&](std::size_t shard, traversal_t& traversal, std::size_t visible) noexcept {
            nk::matrix_span<dense_node_id_t> const shard_ids = ids_wide(
                nk::all, nk::range(shard * wanted, (shard + 1) * wanted), nk::slice);
            nk::matrix_span<dot_result_t> const shard_products = products_wide(
                nk::all, nk::range(shard * wanted, (shard + 1) * wanted), nk::slice);
            std::span<u32_t> const shard_counts {counts_landing.data() + shard * rows_count, rows_count};
            expected<search_result_t> const multiplied = shards_[shard].index->submit_multiply_(
                left, shard_ids, shard_products, shard_counts, traversal, config, visible);
            return multiplied.status;
        };
        // Shards append in placement order until a row fills, so rows truncate exactly as on the host.
        auto merge_shard = [&](std::size_t shard, traversal_t const&) noexcept {
            for (std::size_t row = 0; row != rows_count; ++row) {
                std::size_t const landed_offset = (row * shards + shard) * wanted;
                std::size_t const kept = min_of(counts[row], wanted);
                std::size_t const scored = counts_landing[shard * rows_count + row];
                for (std::size_t pair = 0; pair != min_of(scored, wanted - kept); ++pair) {
                    keys[row * wanted + kept + pair] = shards_[shard].keys[ids_landing[landed_offset + pair]];
                    products[row * wanted + kept + pair] = products_landing[landed_offset + pair];
                }
                counts[row] += scored;
            }
        };
        for (std::size_t& count : counts) count = 0;
        return for_each_shard_(executor, rows_count, config, multiply_shard, merge_shard, counts);
    }

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key among the tokens it ranked, and those maxima sum to that key's score.
     *
     *  @param[in] queries The token rows of every query, @p rows_per_query consecutive rows each.
     *  @param[in] rows_per_query Rows in every query; the row count must be a whole multiple of it.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly; each row ranks as many
     *      tokens as its expansion factor, and one wider than the device search can land refuses.
     *  @param[in] wanted Keys kept per query - the row width of @p keys and @p scores.
     *  @param[out] keys The best-scoring keys of each query, @b [queries,wanted] row-major.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, at most @p wanted.
     *
     *  A key some row never ranked misses that row's maximum, so with non-negative per-row maxima
     *  a score lands below its exact value, never above.
     */
    expected<search_result_t> maxsim(vectors_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, std::size_t wanted, std::span<key_t> keys,
                                     std::span<f64_t> scores, std::span<std::size_t> counts) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        std::size_t const rows_count = queries.extent(0);
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (!rows_per_query || rows_count % rows_per_query) return {{}, status_t::shape_mismatch_k};
        std::size_t const groups_count = rows_count / rows_per_query;
        if (counts.size() != groups_count || keys.size() < groups_count * wanted ||
            scores.size() < groups_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!groups_count) return {{}, status_t::success_k};

        // Every token row keeps its whole ranked pool, its keys resolved and merged across the shards.
        std::size_t const width = shards_[0].index->non_zero_expansion_factor(config);
        config.count = static_cast<typename index_t::small_size_t>(width);
        growing_buffer<key_t, forkunion_allocator_t> row_keys {allocator_};
        growing_buffer<distance_t, forkunion_allocator_t> row_distances {allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> row_counts {allocator_};
        dense_maxsim_reducer<key_t, forkunion_allocator_t> reduce {allocator_};
        if (failed(row_keys.resize_uninitialized(rows_count * width)) ||
            failed(row_distances.resize_uninitialized(rows_count * width)) ||
            failed(row_counts.resize_uninitialized(rows_count)) ||
            failed(reduce.reserve(rows_per_query, width, wanted)))
            return {{}, status_t::bad_alloc_k};
        expected<search_result_t> ranked = search(queries, executor, config, width, to_writable_span(row_keys),
                                                  to_writable_span(row_distances), to_writable_span(row_counts));
        if (!ranked) return ranked;
        ranked.value.count = reduce(to_span(row_keys), to_span(row_distances), to_span(row_counts), keys, scores,
                                    counts);
        return ranked;
    }

    /**
     *  @brief Grouped-query attention of every token over every shard: each shard folds the nodes its walks score
     *      into per-head partial rows and carries on its own device, and the first shard's device merges them by
     *      the flash rule, exact over the union.
     *
     *  @param[in] queries The query heads, in the shards' stored scalar type; row t · @p head_count + h is head h
     *      of token t, and each token walks every shard once.
     *  @param[in] head_count The query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Device-callable: `value_lookup(key)` points at the key's value row on a 16-byte
     *      boundary, as wide as the keys and stored like them, in memory every shard's device reads; a null row
     *      refuses the call with @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, @b [rows,dimensions] row-major, in host or device memory.
     *  @param[out] carry Each row's softmax state over every shard, one per row, in host or device memory.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly; @c count plays no part.
     *  @return The query rows attended, and the traversal telemetry every shard spent.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, std::span<f32_t> outputs,
                                     std::span<dense_softmax_state_t> carry, executor_t& executor,
                                     search_config_t config = {}) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        using shard_lookup_t = dense_cuda_keyed_lookup<key_t, std::remove_cvref_t<value_lookup_type_>>;
        std::size_t const rows_count = queries.extent(0), shards = shards_.size();
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.size() != rows_count * dimensions() || carry.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (queries.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};

        // Shard 0 stages one block of rows and carries per shard, its own first and the others copied in.
        std::size_t const block_bytes = outputs.size_bytes() + carry.size_bytes();
        byte_t* blocks = nullptr;
        auto attend_shard = [&](std::size_t shard, traversal_t& traversal, std::size_t visible) noexcept {
            shard_lookup_t const lookup {shards_[shard].device_keys.base(), value_lookup};
            expected<f32_t*> const landed = shards_[shard].index->submit_attend_(
                queries, head_count, lookup, score_scale, traversal, config, shard ? 1 : shards, visible);
            if (!landed) return landed.status;
            if (!shard) {
                blocks = reinterpret_cast<byte_t*>(landed.value);
                return status_t::success_k;
            }
            // On the shard's own stream, so the copy follows its walks.
            return cudaMemcpyPeerAsync(blocks + shard * block_bytes, executor.device(0), landed.value,
                                       executor.device(shard), block_bytes, traversal.stream()) == cudaSuccess
                       ? status_t::success_k
                       : status_t::device_error_k;
        };
        status_t merged = status_t::success_k;
        auto merge_shard = [&](std::size_t shard, traversal_t const& traversal) noexcept {
            // A shard whose walk met a node without its value row folded a wrong softmax.
            if (traversal.missed()) merged = status_t::shape_mismatch_k;
            if (shard + 1 != shards || failed(merged)) return;
            // Every shard drained, so every block landed; the merge and the landing ride shard 0's stream.
            cudaStream_t const stream = executor.stream(0).handle;
            u32_t const rows = static_cast<u32_t>(rows_count);
            if (cudaSetDevice(executor.device(0)) != cudaSuccess ||
                failed(cuda_launch(dense_cuda_merge_attention_kernel,
                                   divide_round_up<u32_t>(rows, dense_cuda_warps_per_block_k),
                                   dense_cuda_warps_per_block_k * 32, 0, stream, blocks, rows,
                                   static_cast<u32_t>(dimensions()), static_cast<u32_t>(shards))) ||
                cudaMemcpyAsync(outputs.data(), blocks, outputs.size_bytes(), cudaMemcpyDefault, stream) !=
                    cudaSuccess ||
                cudaMemcpyAsync(carry.data(), blocks + outputs.size_bytes(), carry.size_bytes(), cudaMemcpyDefault,
                                stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess)
                merged = status_t::device_error_k;
        };
        expected<search_result_t> attended = for_each_shard_(executor, rows_count, config, attend_shard, merge_shard,
                                                             {}, head_count, shards);
        if (!attended) return attended;
        if (failed(merged)) return {{}, merged};
        attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** Bytes a @ref save under @p config writes: one monolith, or every part file and the manifest
     *  measured over a placeholder stem; zero for an invalid config or an unplaced index. */
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

    /** @brief One monolith of a checkpoint: where it is, what its header says, and the tokens
     *      its metadata carries. */
    struct part_t {
        /** @brief The relative path's characters inside the paths arena; empty names the
         *      checkpoint path itself. */
        iota_view<std::size_t> path;
        /** @brief The part's whole bytes - a mapping's view, or the caller's own buffer. */
        bytes_view_t monolith;
        /** @brief Rows the monolith holds, per its own header and columns. */
        std::size_t nodes = 0;
        /** @brief The monolith's header. */
        safegraphs_header_t header;
        /** @brief The metric token the monolith recorded; empty when it omitted one. */
        safegraphs_name_t metric;
        /** @brief The stored scalar the monolith recorded; empty when it omitted one. */
        safegraphs_name_t quantization;
    };

    /**
     *  @brief One save in flight across every shard: one shard plan each on its own device's
     *      stream, the parts they write into, and the tail @ref finish commits - the keys, the
     *      headers, and the manifest.
     *
     *  Steps are flat and each names its shard, so a caller walks them round-robin from one host
     *  thread; @ref steps_of names one shard's own run. A dying plan drops its mappings and its
     *  shard plans, leaving parts no header commits.
     */
    struct [[nodiscard]] save_plan_t {
        /** @brief One step of one shard: its shard, and that shard's own step. */
        struct step_t {
            /** @brief The shard the rows come from. */
            std::size_t shard = 0;
            /** @brief The shard plan's own step. */
            typename index_t::save_plan_t::step_t inner;
        };

        save_plan_t() noexcept : save_plan_t(forkunion_allocator_t {}) {}
        /** @brief An empty plan over @p allocator; only @ref plan_save ever fills it. */
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
        /** @brief Every member is RAII: the mappings, the shard plans and the executor's hold all
         *      die with the plan. */
        ~save_plan_t() noexcept = default;

        /** @brief Step @p position of the plan; an empty one past the last. */
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

        /** @brief The steps the save spans - every shard's own, in placement order; none once spent. */
        indexed_range<save_plan_t> steps() const noexcept {
            std::size_t count = 0;
            if (owner_)
                for (typename index_t::save_plan_t const& plan : plans_) count += plan.steps().size();
            return {this, count};
        }

        /** @brief The contiguous run of steps shard @p shard owns - what a round-robin walk takes. */
        iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
            if (!owner_ || shard >= plans_.size()) return {};
            std::size_t first = 0;
            for (std::size_t earlier = 0; earlier != shard; ++earlier) first += plans_[earlier].steps().size();
            return {first, first + plans_[shard].steps().size()};
        }

        /** @brief Lands step @p step through its own shard's plan; @p scratch rides along for the
         *      host residency's sake and is untouched here. */
        status_t save(step_t const& step, nk::matrix_span<dense_node_id_t> scratch) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (step.shard >= plans_.size()) return status_t::config_mismatch_k;
            return plans_[step.shard].save(step.inner, scratch);
        }

        /**
         *  @brief Pronounces the checkpoint complete: every shard plan closes, its keys land at
         *      its base, the headers follow, and a distributed save commits with its manifest -
         *      the commit record, so a failed save leaves parts and no checkpoint.
         *
         *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k
         *      stops the remaining shards and reports @c cancelled_k.
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
            owner_ = nullptr;
            return status_t::success_k;
        }

      private:
        /** @brief The manifest that commits a distributed save; entries name their parts
         *      relative to it. */
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

        /** @brief The layout the parts were framed in. */
        checkpoint_layout_t layout_ = checkpoint_layout_t::monolith_k;
        /** @brief The one graph every shard writes into; empty for a distributed save. */
        bytes_span_t monolith_;
        /** @brief One plan per shard, in placement order. */
        growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans_;
        /** @brief One header per part - one in all for a monolith. */
        growing_buffer<safegraphs_header_t, forkunion_allocator_t> headers_;
        /** @brief The files the parts were mapped into; empty when the caller owns the bytes. */
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;
        /** @brief The custom fields every header carries, outliving the plan's steps. */
        growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields_;
        /** @brief The checkpoint's own path, copied so the plan outlives the caller's view. */
        growing_buffer<char, forkunion_allocator_t> path_;
        /** @brief The executor's hold, kept for the plan's whole life - one caller at a time. */
        typename executor_t::claim_t streams_;
        /** @brief The index being checkpointed; null in the empty plan a refused @ref plan_save
         *      returns, and once @c finish committed. */
        dense_vram_indexes* owner_ = nullptr;

        friend dense_vram_indexes;
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
     *      planes cannot be settlement-probed, so @c concurrent_k refuses.
     *  @param[inout] executor The streams the exports ride, held until the plan dies.
     */
    expected<save_plan_t> plan_save(std::string_view path, save_config_t const& config, executor_t& executor) noexcept {
        if (status_t const ready = saveable_(config, executor); failed(ready)) return {{}, ready};
        save_plan_t plan {allocator_};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
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

    /** @brief Opens a save into @p monolith as one graph - the in-memory sibling of the path
     *      form; a manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size
     *      the bytes by @ref storage_requirements; fewer answer @c overflow_k. */
    expected<save_plan_t> plan_save(bytes_span_t monolith, save_config_t const& config, executor_t& executor) noexcept {
        if (status_t const ready = saveable_(config, executor); failed(ready)) return {{}, ready};
        if (config.layout != checkpoint_layout_t::monolith_k) return {{}, status_t::config_mismatch_k};
        save_plan_t plan {allocator_};
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.streams_ = std::move(claim);
        if (status_t const opened = open_save_monolith_(plan, monolith, config, executor); failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief One load in flight across every shard: the mapped parts, one shard plan each on its
     *      own device's stream, and the flat steps their rows arrive in - one per part's share of
     *      one batch.
     *
     *  Holds the insertion gate and the executor for its whole life, so a batch in flight refuses
     *  the load and the load refuses a batch. A dying plan releases every claim and publishes
     *  nothing.
     */
    struct [[nodiscard]] load_plan_t {
        /** @brief One step of one shard: where it lands, which part it reads, and that part's own
         *      ID space alongside the shard plan's step. */
        struct step_t {
            /** @brief The shard the rows land in. */
            std::size_t shard = 0;
            /** @brief The part the rows come from. */
            std::size_t part = 0;
            /** @brief The rows of that part the shard takes - the ID space they speak. */
            iota_view<dense_node_id_t> source_range;
            /** @brief The shard plan's own step, cut to one part's share of one batch. */
            typename index_t::load_plan_t::step_t inner;
        };

        load_plan_t() noexcept : load_plan_t(forkunion_allocator_t {}) {}
        /** @brief An empty plan over @p allocator; only @ref plan_load ever fills it. */
        explicit load_plan_t(forkunion_allocator_t allocator) noexcept
            : mappings_(allocator), parts_(allocator), landings_(allocator), plans_(allocator), steps_(allocator) {}
        load_plan_t(load_plan_t&& other) noexcept
            : mappings_(std::move(other.mappings_)), parts_(std::move(other.parts_)),
              landings_(std::move(other.landings_)), plans_(std::move(other.plans_)), steps_(std::move(other.steps_)),
              streams_(std::move(other.streams_)), mutating_(std::move(other.mutating_)),
              owner_(std::exchange(other.owner_, nullptr)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;
        /** @brief Every member is RAII: the mappings and both holds die with the plan, and the
         *      shard plans release the rows they claimed unless @c finish published them. */
        ~load_plan_t() noexcept = default;

        /** @brief Rows the checkpoint lands across every shard - what its headers said. */
        std::size_t rows() const noexcept {
            std::size_t landing_rows = 0;
            for (typename index_t::load_plan_t const& plan : plans_) landing_rows += plan.rows;
            return landing_rows;
        }

        /** @brief Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            if (position >= steps().size()) return {};
            return steps_[position];
        }

        /** @brief The steps the load spans - every shard's own, in placement order; none once spent. */
        indexed_range<load_plan_t> steps() const noexcept { return {this, owner_ ? steps_.size() : 0}; }

        /** @brief The contiguous run of steps shard @p shard owns - what a round-robin walk takes. */
        iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
            if (!owner_) return {};
            std::size_t first = 0;
            while (first != steps_.size() && steps_[first].shard != shard) ++first;
            std::size_t bound = first;
            while (bound != steps_.size() && steps_[bound].shard == shard) ++bound;
            return {first, bound};
        }

        /** @brief Lands step @p step: its part's own columns, under that part's ID space, through
         *      the shard's own plan. */
        status_t load(step_t const& step) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (step.shard >= plans_.size() || step.part >= parts_.size()) return status_t::config_mismatch_k;
            part_t const& part = parts_[step.part];
            typename index_t::columns_view_t const columns {part.header.data_of(part.monolith, part.header.value),
                                                            part.header.data_of(part.monolith, part.header.neighbors)};
            return plans_[step.shard].load(step.inner, columns, step.source_range);
        }

        /**
         *  @brief Publishes the restored graph: every shard's keys land at its base and enter the
         *      locator map, then every shard plan publishes on its own device - the order an
         *      insert keeps, so a search never reaches a published row without its key.
         *
         *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k reports
         *      @c cancelled_k.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t finish(progress_type_&& on_loaded = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
                std::size_t first = plans_[shard].base;
                for (checkpoint_assignment_t const& landing : landings_) {
                    std::size_t const rows = landing.rows.size();
                    if (landing.shard != shard || !rows) continue;
                    part_t const& part = parts_[landing.part];
                    keys_column_t& keys = owner_->shards_[shard].keys;
                    if (status_t const covered = cover_keys_(owner_->shards_[shard], first + rows); failed(covered))
                        return covered;
                    auto [seats, seat_status] = owner_->locators_of_key_.claim(rows);
                    if (failed(seat_status)) return seat_status;
                    // The key column is chunked, so the part's keys land run by run.
                    bytes_view_t const part_keys = part.header.rows_of(part.monolith, part.header.key,
                                                                       landing.rows.first, rows);
                    std::size_t read_bytes = 0;
                    status_t read = status_t::success_k;
                    keys.for_each_span({first, first + rows}, [&](std::span<key_t> run) noexcept {
                        if (succeeded(read))
                            read = copy_bytes(bytes_of(run.data(), run.size()),
                                              part_keys.subspan(read_bytes, run.size_bytes()));
                        read_bytes += run.size_bytes();
                    });
                    if (failed(read)) return read;
                    for (std::size_t row = 0; row != rows; ++row)
                        owner_->locators_of_key_.insert(
                            seats, keys[first + row], key_locator_t {shard, static_cast<dense_node_id_t>(first + row)});
                    first += rows;
                }
                // Every plan finishes, an empty one included, since finishing releases the claim.
                if (cudaSetDevice(owner_->shards_[shard].index->device_specs().ordinal) != cudaSuccess)
                    return status_t::device_error_k;
                if (status_t const uploaded = upload_keys_(owner_->shards_[shard], {plans_[shard].base, first},
                                                           nullptr);
                    failed(uploaded))
                    return uploaded;
                if (cudaStreamSynchronize(nullptr) != cudaSuccess) return status_t::device_error_k;
                if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
            }
            for (std::size_t shard = 0; shard != plans_.size(); ++shard)
                if (on_loaded(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
            owner_->routed_rows_ = rows();
            owner_ = nullptr;
            return status_t::success_k;
        }

      private:
        /** @brief The files the parts were mapped from; every part's rows are read out of them. */
        growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;
        /** @brief The checkpoint's parts, vetted at planning. */
        growing_buffer<part_t, forkunion_allocator_t> parts_;
        /** @brief Which part lands where, in the order each shard appends them - the sequence
         *      @c finish walks to place the keys. */
        growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> landings_;
        /** @brief One plan per shard, in placement order. */
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans_;
        /** @brief The flat steps, grouped by shard. */
        growing_buffer<step_t, forkunion_allocator_t> steps_;
        /** @brief The executor's hold, kept for the plan's whole life - one caller at a time. */
        typename executor_t::claim_t streams_;
        /** @brief The insertion gate, held so no batch interleaves with the restore. */
        std::unique_lock<fu::spin_mutex_t> mutating_;
        /** @brief The index being restored; null in the empty plan a refused @ref plan_load
         *      returns, and once @c finish published. */
        dense_vram_indexes* owner_ = nullptr;

        friend dense_vram_indexes;
    };

    /**
     *  @brief Opens a load of the checkpoint at @p path: the content says monolith or manifest,
     *      every part's header is vetted before a byte lands, the reservation comes from
     *      @p config, and the parts land per its placement - as written, part @c p on shard
     *      @c p mod S, or cut into one stripe per shard.
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] config The placement and the rows to reserve; zero fits the checkpoint.
     *  @param[inout] executor The streams the imports ride, held until the plan dies.
     *
     *  The index must be empty; the plan reports the rows it will land through @c rows().
     */
    expected<load_plan_t> plan_load(std::string_view path, load_config_t const& config, executor_t& executor) noexcept {
        if (status_t const ready = loadable_(executor); failed(ready)) return {{}, ready};
        load_plan_t plan {allocator_};
        if (status_t const gated = gate_load_(plan, executor); failed(gated)) return {{}, gated};
        if (status_t const listed = catalog_(path, plan.mappings_, plan.parts_); failed(listed)) return {{}, listed};
        if (status_t const opened = open_load_(plan, config, executor); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** @brief Opens a load of one monolith held in @p monolith - the implementation the path form
     *      maps for. */
    expected<load_plan_t> plan_load(bytes_view_t monolith, load_config_t const& config, executor_t& executor) noexcept {
        if (status_t const ready = loadable_(executor); failed(ready)) return {{}, ready};
        load_plan_t plan {allocator_};
        if (status_t const gated = gate_load_(plan, executor); failed(gated)) return {{}, gated};
        if (status_t const listed = catalog_monolith_(monolith, plan.parts_); failed(listed)) return {{}, listed};
        if (status_t const opened = open_load_(plan, config, executor); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Serializes the index under @p path - the one-shot over @ref plan_save, its steps
     *      walked round-robin from this one host thread.
     *
     *  @param[in] path The monolith's path, or the manifest's.
     *  @param[in] config The layout, alignment, certified contract and provenance.
     *  @param[inout] executor The streams the exports ride.
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
     *      @ref plan_load, its steps walked round-robin over the devices.
     *
     *  The index must be empty, and it takes the hold every batch takes, so a batch in flight
     *  refuses the load as @c contended_k and the load refuses a batch the same way. A cancelled
     *  or failed load clears it and reports why, so nothing is published and the next load
     *  starts clean.
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

    /** Entries stored under @p key; a concurrent insertion can only make it under-report. */
    std::size_t count(key_t const& key) const noexcept { return locators_of_key_.count(key); }

    /** Whether at least one entry is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return locators_of_key_.contains(key); }

    /**
     *  @brief Copies up to @p wanted rows stored under @p key into @p vectors and reports the
     *      @b true total under that key - a return above @p wanted means the copy truncated, so
     *      the caller resizes and asks again; zero means absent.
     *
     *  Safe against concurrent insertions, which only ever add; each row rides one blocking
     *  device-to-host copy, and a device fault leaves that row unwritten with the total honest.
     */
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept {
        std::size_t total = 0;
        locators_of_key_.find_each(key, [&](key_locator_t locator) noexcept {
            if (total < wanted && total < vectors.extent(0)) [[maybe_unused]]
                status_t const copied = shards_[locator.shard].index->vector(locator.node,
                                                                             vectors.row(total).as_vector());
            ++total;
        });
        return total;
    }

  private:

    /** One unreserved shard placed on @p device; null when the allocation fails. */
    static std::unique_ptr<index_t> build_(int device) noexcept {
        typename index_t::vectors_allocator_t const allocator {device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(allocator, allocator));
    }

    /** Whether @p executor drives exactly this index's devices, in this index's order - the pairing
     *  every fan-out indexes by, checked once rather than searched per shard. */
    status_t pairs_with_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (executor.size() != shards_.size()) return status_t::config_mismatch_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard)
            if (executor.device(shard) != shards_[shard].index->device_specs().ordinal)
                return status_t::config_mismatch_k;
        return status_t::success_k;
    }

#pragma region Insertion

    /** The first row of a batch that @p shard takes under @p rotation - a bijection over the
     *  shards, so the residue classes it names partition the batch exactly. */
    static std::size_t first_row_of_shard_(std::size_t shard, std::size_t shards, std::size_t rotation) noexcept {
        return (shard + shards - rotation) % shards;
    }

    /** How many of @p rows a shard starting at @p first takes, stepping by @p shards. */
    static std::size_t rows_from_(std::size_t rows, std::size_t first, std::size_t shards) noexcept {
        return first < rows ? divide_round_up(rows - first, shards) : 0;
    }

    /** Builds one traversal context per shard over its own stream and run, so a fan-out never
     *  allocates and never shares scratch; @p rows, and any indexer @p heads a search stages, size them for the
     *  widest walk. */
    template <typename traversal_config_type_, typename... heads_types_>
    status_t reserve_traversals_(growing_buffer<traversal_t, forkunion_allocator_t>& traversals, executor_t& executor,
                                 std::size_t rows, traversal_config_type_ config,
                                 heads_types_ const&... heads) noexcept {
        std::size_t bytes_per_device = 0;
        for (shard_t const& one : shards_)
            bytes_per_device = max_of(bytes_per_device, one.index->traversal_memory_usage(rows, config, heads...));
        if (status_t const sized = executor.reserve(bytes_per_device); failed(sized)) return sized;
        if (failed(traversals.reserve_exactly(shards_.size()))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_.size(); ++shard) {
            typename executor_t::stream_t& stream = executor.stream(shard);
            traversals.push_back_reserved(
                traversal_t {typename traversal_t::allocator_t {stream.arena, stream.handle, stream.device}});
            if (!traversals.back()) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Covers rows below @p rows in both of @p one's key columns, reserving the device one at its first cover. */
    static status_t cover_keys_(shard_t& one, std::size_t rows) noexcept {
        if (status_t const covered = one.keys.cover(rows); failed(covered)) return covered;
        if (!one.device_keys.reserved_bytes())
            if (status_t const reserved = one.device_keys.reserve_range(index_t::max_node_count_k * sizeof(key_t),
                                                                        one.index->device_specs().ordinal);
                failed(reserved))
                return reserved;
        return one.device_keys.commit(rows * sizeof(key_t));
    }

    /** Copies @p rows of @p one's host key column into its device twin on @p stream, run by run, since the host
     *  column is chunked. */
    static status_t upload_keys_(shard_t& one, iota_view<std::size_t> rows, cudaStream_t stream) noexcept {
        status_t uploaded = status_t::success_k;
        std::size_t row = rows.first;
        one.keys.for_each_span(rows, [&](std::span<key_t> run) noexcept {
            if (succeeded(uploaded) && cudaMemcpyAsync(one.device_keys.base() + row, run.data(), run.size_bytes(),
                                                       cudaMemcpyHostToDevice, stream) != cudaSuccess)
                uploaded = status_t::device_error_k;
            row += run.size();
        });
        return uploaded;
    }

#pragma endregion Insertion

    /** Folds one shard's sweep into the group's; every counter is a population or a spend, so all
     *  of them add. */
    static void combine_refinement_(refine_result_t& into, refine_result_t const& one) noexcept {
        into.visited_nodes += one.visited_nodes;
        into.changed_nodes += one.changed_nodes;
        into.islands_found += one.islands_found;
        into.islands_reconnected += one.islands_reconnected;
        into.reachable_nodes += one.reachable_nodes;
        into.sweeps = max_of(into.sweeps, one.sweeps);
        into.visited_members += one.visited_members;
        into.computed_distances += one.computed_distances;
        into.computed_distances_in_refines += one.computed_distances_in_refines;
        into.computed_distances_in_reverse_refines += one.computed_distances_in_reverse_refines;
        into.restarts += one.restarts;
        into.saturated_walks += one.saturated_walks;
    }

    /** Folds one shard's audit into the group's: populations add, and the depth is the deepest
     *  shard's, since a searcher waits for the last one. */
    static void combine_health_(health_result_t& into, health_result_t const& one) noexcept {
        into.audited_nodes += one.audited_nodes;
        into.reachable_nodes += one.reachable_nodes;
        into.sweeps = max_of(into.sweeps, one.sweeps);
    }

#pragma region Search

    /** Takes every shard's visible rows into @p visible, unless it is empty, at one instant no commit straddles;
     *  answers their sum. */
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
     *  @c merge_search_results consumes, stopping at the first sentinel ID - an absent
     *  match must never index a key column. @return The live matches repacked. */
    static std::size_t repack_landed_matches_(dense_node_id_t const* landed_ids,
                                              dense_distance_t const* landed_distances, std::size_t count,
                                              dense_scored_node_t* frame) noexcept {
        std::size_t live = 0;
        for (; live != count && landed_ids[live] != dense_node_id_unknown_k; ++live)
            frame[live] = dense_scored_node_t {landed_ids[live], landed_distances[live]};
        return live;
    }

    /** The body of the batched @c search, passing any indexer @p heads to every shard as is. */
    template <typename... heads_types_>
    expected<search_result_t> search_(vectors_view_t queries, executor_t& executor, search_config_t config,
                                      std::size_t wanted, std::span<key_t> keys, std::span<distance_t> distances,
                                      std::span<std::size_t> counts, heads_types_ const&... heads) noexcept {
        std::size_t const queries_count = queries.extent(0);
        std::size_t const shards = shards_.size();
        if (status_t const paired = pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (counts.size() != queries_count || keys.size() < queries_count * wanted ||
            distances.size() < queries_count * wanted)
            return {{}, status_t::config_mismatch_k};
        if (!queries_count) return {{}, status_t::success_k};
        // Vetted before anything is claimed or carved; the shards would refuse only after the
        // landing and the runs had been sized for a doomed call.
        if (queries.extent(1) != dimensions() || config.count > dense_cuda_sorted_pool<16>::capacity_k)
            return {{}, status_t::config_mismatch_k};

        // The host landing: every shard's matches, one wide matrix each, shard columns side by
        // side; never zeroed, since only columns a drained shard wrote are ever read.
        std::size_t const landing_entries = queries_count * shards * config.count;
        growing_buffer<dense_node_id_t, forkunion_allocator_t> ids_landing {allocator_};
        growing_buffer<dense_distance_t, forkunion_allocator_t> distances_landing {allocator_};
        if (failed(ids_landing.resize_uninitialized(landing_entries)) ||
            failed(distances_landing.resize_uninitialized(landing_entries)))
            return {{}, status_t::bad_alloc_k};
        nk::matrix_span<dense_node_id_t> const ids_wide {ids_landing.data(), queries_count, shards * config.count};
        nk::matrix_span<dense_distance_t> const distances_wide {distances_landing.data(), queries_count,
                                                                shards * config.count};

        auto search_shard = [&](std::size_t shard, traversal_t& traversal, std::size_t visible) noexcept {
            nk::matrix_span<dense_node_id_t> const shard_ids = ids_wide(
                nk::all, nk::range(shard * config.count, (shard + 1) * config.count), nk::slice);
            nk::matrix_span<dense_distance_t> const shard_distances = distances_wide(
                nk::all, nk::range(shard * config.count, (shard + 1) * config.count), nk::slice);
            expected<search_result_t> const found = shards_[shard].index->submit_search_(
                queries, shard_ids, shard_distances, traversal, config, visible, heads...);
            return found.status;
        };
        auto merge_shard = [&](std::size_t shard, traversal_t const&) noexcept {
            dense_scored_node_t frame[dense_cuda_sorted_pool<16>::capacity_k];
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const landed_offset = (query * shards + shard) * config.count;
                std::size_t const live = repack_landed_matches_(
                    ids_landing.data() + landed_offset, distances_landing.data() + landed_offset, config.count, frame);
                std::span<key_t> const keys_row {keys.data() + query * wanted, wanted};
                std::span<distance_t> const distances_row {distances.data() + query * wanted, wanted};
                counts[query] = merge_search_results(std::span<dense_scored_node_t const> {frame, live},
                                                     shards_[shard].keys, keys_row, distances_row, counts[query]);
            }
        };
        for (std::size_t& count : counts) count = 0;
        return for_each_shard_(executor, queries_count, config, search_shard, merge_shard, counts, heads...);
    }

    /** Claims @p executor, sizes one context per shard for @p rows_count rows under @p config and any indexer
     *  @p heads, snapshots the shards' visible rows, queues @p submit under its snapshot on every shard before
     *  awaiting any, drains every queued stream, hands each drained shard and its context to @p merge, and sums the
     *  telemetry & the rows' @p counts; the first refusal is the one kept. */
    template <typename submit_type_, typename merge_type_, typename... heads_types_>
    expected<search_result_t> for_each_shard_(executor_t& executor, std::size_t rows_count, search_config_t config,
                                              submit_type_&& submit, merge_type_&& merge,
                                              std::span<std::size_t const> counts,
                                              heads_types_ const&... heads) noexcept {
        // The executor serializes, never the index: a second caller with its own executor searches concurrently.
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        growing_buffer<traversal_t, forkunion_allocator_t> traversals {allocator_};
        if (status_t const readied = reserve_traversals_(traversals, executor, rows_count, config, heads...);
            failed(readied))
            return {{}, readied};
        std::size_t const shards = shards_.size();
        growing_buffer<std::size_t, forkunion_allocator_t> visible {allocator_};
        if (failed(visible.resize(shards))) return {{}, status_t::bad_alloc_k};
        snapshot_(to_writable_span(visible));

        // Queue every shard before awaiting any, so the batch costs the slowest device rather than
        // the sum. From here to the drain there must be no return: the caller's rows and the landing
        // are DMA targets on every queued stream.
        status_t first_refusal = status_t::success_k;
        std::size_t queued = 0;
        // A launch fails on a stream of any device but the current one.
        for (; queued != shards; ++queued)
            if (status_t const submitted = cudaSetDevice(traversals[queued].device()) == cudaSuccess
                                               ? submit(queued, traversals[queued], visible[queued])
                                               : status_t::device_error_k;
                failed(submitted)) {
                first_refusal = submitted;
                break;
            }

        // Await every stream that may carry queued transfers - the refuser included, since its
        // upload may already be reading the caller's rows - and merge each drained shard's columns.
        for (std::size_t shard = 0; shard != shards && shard <= queued; ++shard) {
            if (status_t const drained = traversals[shard].drain(); failed(drained)) {
                if (succeeded(first_refusal)) first_refusal = drained;
                continue; // An undrained landing is unreadable - this shard merges nothing.
            }
            if (shard == queued) continue; // The refuser drained for the caller's sake, landed nothing.
            merge(shard, traversals[shard]);
        }

        // Every drained context's sink holds this call's spend on its shard.
        search_result_t merged;
        for (std::size_t shard = 0; shard != shards && shard <= queued; ++shard) {
            dense_cuda_telemetry_block_t const spent = traversals[shard].spent();
            merged.visited_members += spent.hops;
            merged.computed_distances += spent.scored;
            merged.restarts += spent.restarts;
            merged.tail_drops += spent.tail_drops;
            merged.visited_saturations += spent.visited_saturations;
        }
        for (std::size_t const count : counts) merged.count += count;
        return {merged, first_refusal};
    }

#pragma endregion Search

#pragma region Checkpoints

    /** The interchange token of one key - the monolith's key column dtype. */
    inline static constexpr dtype_t key_dtype_k = safegraphs_key_dtype<key_t>();

    /** The value column's dtype - the stored scalar's own token, so a packed word's suffix carries
     *  the packing and the column's shape counts logical dimensions. */
    inline static constexpr dtype_t value_column_dtype_k = safegraphs_value_dtype<scalar_t>();

    /** Whether the index can write @p config through @p executor: placed, reserved, tokenized,
     *  legal, paired, unbroken, and certified quiescent. */
    status_t saveable_(save_config_t const& config, executor_t const& executor) const noexcept {
        static_assert(key_dtype_k != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards_.empty() || !dimensions()) return status_t::uninitialized_k;
        if (value_column_dtype_k == dtype_t::unknown_k) return status_t::unknown_kind_k;
        if (config.synchronization == synchronization_t::concurrent_k) return status_t::config_mismatch_k;
        if (status_t const valid = config.validate(); failed(valid)) return valid;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(executor);
    }

    /** Whether the index can take a checkpoint through @p executor: placed, empty, unbroken, and
     *  paired. */
    status_t loadable_(executor_t const& executor) const noexcept {
        if (shards_.empty()) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        if (status_t const kept = status(); failed(kept)) return kept;
        return pairs_with_(executor);
    }

    /** Storage values per stored row - the copy width of every column. */
    std::size_t row_values_() const noexcept {
        return divide_round_up<std::size_t>(dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** The custom fields every monolith carries: the metric, the stored scalar by its own name -
     *  what a packed @c u8 column cannot say - and the caller's provenance. */
    status_t custom_fields_(growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields,
                            save_config_t const& config) const noexcept {
        if (failed(fields.reserve_exactly(2 + config.provenance.size()))) return status_t::bad_alloc_k;
        fields.push_back_reserved({"metric", nk_kernel_name(kernel_kind_)});
        fields.push_back_reserved({"quantization", nk_dtype_name(vector_dtype_)});
        for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
        return status_t::success_k;
    }

    /** The monolith header for @p nodes rows under @p config 's alignment. */
    expected<safegraphs_header_t> monolith_header_(std::size_t nodes, save_config_t const& config,
                                                   std::span<safegraphs_custom_field_t const> fields) const noexcept {
        safegraphs_regular_shape_t const shape {nodes,        key_dtype_k,    value_column_dtype_k,
                                                dimensions(), dtype_t::u32_k, connectivity()};
        return safegraphs_regular_header(shape, config.alignment, fields);
    }

    /** @brief Walks @p plan 's steps from this one host thread, taking step @c s of every shard
     *      before step @c s+1; every step drains its stream before the next, so the walk is
     *      serial across the devices, and the first failing step ends it. A save lands each
     *      step, a load reads it straight out of its part.
     */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan) noexcept {
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step,
                                           nk::matrix_span<dense_node_id_t> scratch) { planned.save(step, scratch); };
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

    /** Copies @p path into @p buffer, so a plan outlives the caller's view of it. */
    static status_t keep_path_(growing_buffer<char, forkunion_allocator_t>& buffer, std::string_view path) noexcept {
        if (failed(buffer.reserve_exactly(path.size()))) return status_t::bad_alloc_k;
        for (char const character : path) buffer.push_back_reserved(character);
        return status_t::success_k;
    }

    /** @brief Opens shard @p shard 's own save over the @p nodes rows it writes at @p first_row of
     *      @p target, on its own device's stream; the plan's own @c id_shift is where its keys land
     *      in @c finish. */
    status_t open_save_shard_(save_plan_t& plan, std::size_t shard, bytes_span_t target,
                              safegraphs_header_t const& header, std::size_t first_row, std::size_t nodes,
                              save_config_t const& config, executor_t& executor) noexcept {
        index_t& one = *shards_[shard].index;
        if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
        typename index_t::columns_t const columns {header.rows_of(target, header.value, first_row, nodes),
                                                   header.rows_of(target, header.neighbors, first_row, nodes)};
        auto [inner, status] = one.plan_save(columns, config.synchronization, executor.stream(shard).handle,
                                             static_cast<dense_node_id_t>(first_row));
        if (failed(status)) return status;
        plan.plans_.push_back_reserved(std::move(inner));
        return status_t::success_k;
    }

    /** @brief Frames @p monolith for one graph, every shard writing at its own base with its IDs
     *      rebased onto it, and opens one save plan per shard over the rows it owns. */
    status_t open_save_monolith_(save_plan_t& plan, bytes_span_t monolith, save_config_t const& config,
                                 executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        plan.layout_ = checkpoint_layout_t::monolith_k;
        plan.monolith_ = monolith;
        if (status_t const named = custom_fields_(plan.fields_, config); failed(named)) return named;
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {allocator_};
        if (failed(plan.plans_.reserve_exactly(shards)) || failed(plan.headers_.reserve_exactly(1)) ||
            failed(shard_rows.reserve_exactly(shards)))
            return status_t::bad_alloc_k;

        // The rows are read once, so a shard that grows between here and its watermark is refused there.
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
        plan.owner_ = this;
        return status_t::success_k;
    }

    /** Frames, sizes and maps one part per shard beside the plan's path, and opens one save plan
     *  over each - a part's IDs stay shard-local, so its rows travel verbatim. */
    status_t open_save_parts_(save_plan_t& plan, save_config_t const& config, executor_t& executor) noexcept {
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
        plan.owner_ = this;
        return status_t::success_k;
    }

    /** Reads one part's header and tokens out of the bytes it carries. */
    status_t describe_part_(part_t& part) const noexcept {
        auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
            if (field.key == "metric") part.metric = safegraphs_name_t {field.value};
            if (field.key == "quantization") part.quantization = safegraphs_name_t {field.value};
        };
        auto const [header, status] = safegraphs_read_header(part.monolith, allocator_, per_custom_field);
        if (failed(status)) return status;
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
                part_path, path, std::string_view {paths_arena.data() + part.path.first, part.path.size()});
            if (failed(sibling_status)) return sibling_status;
            auto [part_mapping, part_mapping_status] = mapped_file_t::read(sibling);
            if (failed(part_mapping_status)) return part_mapping_status;
            part.monolith = part_mapping.view();
            if (status_t const described = describe_part_(part); failed(described)) return described;
            mappings.push_back_reserved(std::move(part_mapping));
        }
        return status_t::success_k;
    }

    /** Whether @p part can land here: three regular columns of the index's own dtypes, the tokens
     *  it names agreeing with this instantiation, the geometry @p first froze, and every column
     *  describing the same rows. */
    status_t vetted_part_(part_t const& part, safegraphs_header_t const& first) const noexcept {
        safegraphs_header_t const& header = part.header;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (header.key.dtype != key_dtype_k || header.value.dtype != value_column_dtype_k ||
            header.neighbors.dtype != dtype_t::u32_k)
            return status_t::config_mismatch_k;
        if (header.key.shape != column_shape_t {scalar_shape_t {}} || header.value.shape != first.value.shape ||
            header.neighbors.shape != first.neighbors.shape)
            return status_t::config_mismatch_k;
        if (!part.metric.empty() && nk_kernel_named(part.metric.characters, part.metric.view().size()) != kernel_kind_)
            return status_t::config_mismatch_k;
        if (!part.quantization.empty() &&
            nk_dtype_named(part.quantization.characters, part.quantization.view().size()) != vector_dtype_)
            return status_t::config_mismatch_k;
        if (header.key.nodes() != part.nodes || header.value.nodes() != part.nodes ||
            header.neighbors.nodes() != part.nodes)
            return status_t::malformed_k;
        return status_t::success_k;
    }

    /** Takes the two holds a load runs under - the executor's streams and the insertion
     *  gate - and checks emptiness beneath them, where no batch can land before the load does. */
    status_t gate_load_(load_plan_t& plan, executor_t& executor) noexcept {
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        plan.streams_ = std::move(claim);
        if (!mutating_.lock_if_free()) return status_t::contended_k;
        plan.mutating_ = std::unique_lock<fu::spin_mutex_t> {mutating_, std::adopt_lock};
        return size() ? status_t::config_mismatch_k : status_t::success_k;
    }

    /**
     *  @brief Vets the plan's parts, assigns them onto the shards per @p config 's placement,
     *      reserves from it, and opens one load plan per shard over the rows it ends up with.
     *
     *  A shard takes its parts in order and appends them at one base, so the flat steps are the
     *  intersections of each part's rows with the batches they land in - a part that spans two
     *  batches answers two steps, and two parts sharing a batch answer one each.
     */
    status_t open_load_(load_plan_t& plan, load_config_t const& config, executor_t& executor) noexcept {
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

        // Growth ahead of the rows the parts carry - the config's or the widest shard times the shard count - and
        // the frozen geometry, when there is one, must match the parts'.
        std::size_t const wanted = max_of(config.capacity, widest * shards);
        if (capacity() && (dimensions != this->dimensions() || connectivity != this->connectivity()))
            return status_t::config_mismatch_k;
        if (!capacity() && !wanted) {
            plan.owner_ = this;
            return status_t::success_k;
        }
        if (status_t const reserved = reserve({wanted, connectivity, dimensions}).status; failed(reserved))
            return reserved;

        if (failed(plan.plans_.reserve_exactly(shards))) return status_t::bad_alloc_k;
        std::size_t const vectors_row_bytes = row_values_() * sizeof(scalar_t);
        std::size_t const neighbors_row_bytes = connectivity * sizeof(dense_node_id_t);
        for (std::size_t shard = 0; shard != shards; ++shard) {
            index_t& one = *shards_[shard].index;
            if (cudaSetDevice(one.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
            auto [inner, status] = one.plan_load(shard_rows[shard], executor.stream(shard).handle);
            if (failed(status)) return status;
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
                    step.source_range = {static_cast<dense_node_id_t>(assignment.rows.first),
                                         static_cast<dense_node_id_t>(assignment.rows.bound)};
                    step.inner.position = batch.position;
                    step.inner.index_rows = {static_cast<dense_node_id_t>(low), static_cast<dense_node_id_t>(high)};
                    step.inner.source_rows = {static_cast<dense_node_id_t>(assignment.rows.first + low - first_row),
                                              static_cast<dense_node_id_t>(assignment.rows.first + high - first_row)};
                    step.inner.vectors_bytes = (high - low) * vectors_row_bytes;
                    step.inner.neighbors_bytes = (high - low) * neighbors_row_bytes;
                    if (failed(plan.steps_.push_back(step))) return status_t::bad_alloc_k;
                }
                first_row = bound_row;
            }
        }
        plan.owner_ = this;
        return status_t::success_k;
    }

#pragma endregion Checkpoints

    /** The owned shards, graph and keys together. A shard's key column is sized at @c reserve and
     *  never grown, since a search reads it while rows are still landing. */
    growing_buffer<shard_t, forkunion_allocator_t> shards_;

    /** Every key's entries. Duplicates coexist as separate slots, which is what several values
     *  under one key means, and costs one slot per row rather than per key. */
    locators_map_t locators_of_key_;

    /** The host allocator behind the shard table, the key columns, the locator map and every
     *  checkpoint catalog; the shards' planes live on their devices. */
    forkunion_allocator_t allocator_;

    /** Rows routed so far - the next batch's rotation base. */
    std::size_t routed_rows_ = 0;

    /** Held by the one batch of mutating work at a time; a refused hold is @c contended_k. */
    fu::spin_mutex_t mutating_;

    /** Commits in flight in the low half, commits landed in the high half: a reader spanning the shards retries its
     *  snapshot while one is in flight or one landed under it. */
    u64_t commit_sequence_ = 0;
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
