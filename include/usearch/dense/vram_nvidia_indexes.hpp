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
 *  @ref merge_search_results, resolving matches into the caller's keys, so shards with colliding ID
 *  spaces land in one shared ranking. Everything but the device keys, the CUDA calls and the
 *  attention gather is @c sharded_indexes.
 *
 *  Beyond the executor's claim, each shard's claim word admits one batch or load at a time and
 *  closes around a refine, an absorb, a checkpoint load or a save, which refuse with
 *  @c contended_k while a batch is in flight; a shard's kept failure refuses every later mutation,
 *  and the commit sequence word lets a reader spanning the shards see each batch whole or not at
 *  all.
 *
 *  Plain host C++ like the shard's own face: the attend gather launches through the shard's
 *  declarations, which the kernel side defines.
 */
#pragma once

#include <memory>  // `std::unique_ptr` - an index holds atomics and never moves
#include <span>    // `std::span` - the caller's keys & the merged ranking
#include <utility> // `std::move`

#include "usearch/dense/sharded_indexes.hpp"
#include "usearch/dense/vram_nvidia_index.hpp"

namespace unum::usearch::dense {

/**
 *  @brief The sharded index: same-geometry GPU-resident shards viewed as one.
 *
 *  Each shard's keys live twice: on the host, where the merges resolve them, and on its device,
 *  where a walk resolves its nodes and the on-device merge reads them.
 */
template <                                         //
    typename key_type_,                            //
    nk_dtype_t vector_dtype_,                      //
    nk_capability_t capability_,                   //
    typename kernels_type_,                        //
    typename key_hash_type_ = std::hash<key_type_> //
    >
struct vram_nvidia_indexes
    : sharded_indexes<vram_nvidia_indexes<key_type_, vector_dtype_, capability_, kernels_type_, key_hash_type_>, //
                      vram_nvidia_index<vector_dtype_, capability_, kernels_type_>,                              //
                      cuda_executor_t, key_type_, key_hash_type_> {

    using base_t = sharded_indexes<vram_nvidia_indexes, vram_nvidia_index<vector_dtype_, capability_, kernels_type_>,
                                   cuda_executor_t, key_type_, key_hash_type_>;
    using typename base_t::batch_t;
    using typename base_t::executor_t;
    using typename base_t::index_t;
    using typename base_t::key_t;
    using typename base_t::search_config_t;
    using typename base_t::search_result_t;
    using typename base_t::traversal_context_t;
    static_assert(index_family<index_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    /** The NVIDIA capabilities the kernels compiled under. */
    inline static constexpr nk_capability_t capability_k = capability_;

    /** The vendor whose devices the shards sit on, beside the vendors' twins of this index. */
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;

    /** Rows one window of a batch verb stages on a shard at most: one slab, the granularity
     *  checkpoints move by, so a batch of any size crosses in bounded staging. */
    inline static constexpr std::size_t window_rows_k = power_of_two<std::size_t>(index_t::batch_rows_log2_k);

    vram_nvidia_indexes() = delete;
    explicit vram_nvidia_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : base_t(kernel, allocator), device_keys_(allocator), key_stagings_(allocator) {}
    vram_nvidia_indexes(vram_nvidia_indexes&& other) noexcept
        : base_t(other.kernel_, other.allocator_), device_keys_(other.allocator_), key_stagings_(other.allocator_) {
        swap(other);
    }
    vram_nvidia_indexes& operator=(vram_nvidia_indexes&& other) noexcept { return swap(other), *this; }

    /** Exchanges two indexes whole, device keys and their staging included. */
    void swap(vram_nvidia_indexes& other) noexcept {
        base_t::swap(other);
        std::swap(device_keys_, other.device_keys_);
        std::swap(key_stagings_, other.key_stagings_);
    }

    using base_t::emplace;

    /** Appends one shard built here on the device @p stream names, for callers with none to hand
     *  over. */
    status_t emplace(typename executor_t::stream_t const& stream) noexcept {
        std::unique_ptr<index_t> shard = build_(stream.device);
        if (!shard) return status_t::bad_alloc_k;
        return this->emplace(std::move(shard));
    }

    /** Bytes the index occupies - the shared measure plus every shard's device keys. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = base_t::memory_usage();
        for (cuda_virtual_memory_allocator<key_t> const& keys : device_keys_) bytes += keys.committed_bytes();
        return bytes;
    }

    /**
     *  @brief Grouped-query attention of every token over every shard: each shard folds the nodes
     *      its walks score into per-head partial rows and carries on its own device, and the first
     *      shard's device merges them by the flash rule, exact over the union.
     *
     *  @param[in] queries The query heads, in the shards' stored scalar type as in @c search; row t
     *      · @p head_count + h is head h of token t, and each token walks every shard once.
     *  @param[in] head_count The query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Device-callable: `value_lookup(key)` points at the key's value row
     *      on a 16-byte boundary, as wide as the keys and stored like them, in memory every shard's
     *      device reads; a null row refuses the call with @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys, in host or
     *      device memory; rows in any scalar but f32 cast on the first shard's device first, and
     *      must fill whole bytes.
     *  @param[out] carry Each row's softmax state over every shard, one per row, in host or device
     *      memory; an empty span skips them.
     *  @param[inout] executor The contexts and streams, paired positionally with the shards.
     *  @param[in] config The per-shard walk configuration, applied uniformly; @c count plays no
     *      part.
     *  @return The query rows attended, and the traversal telemetry every shard spent;
     *      @c config_mismatch_k unless the shards score by dot.
     *
     *  Whole tokens cross @c window_rows_k rows at a time, and a refused window keeps every one
     *  already landed.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     executor_t& executor, search_config_t config = {}) noexcept {
        using shard_lookup_t = vram_keyed_lookup<key_t, std::remove_cvref_t<value_lookup_type_>>;
        std::size_t const rows_count = queries.rows, shards = shards_.size();
        if (this->kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (status_t const paired = this->pairs_with_(executor); paired != status_t::success_k) return {{}, paired};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (status_t const refusal = shards_[0].index->landing_refusal_(rows_count, outputs, carry); failed(refusal))
            return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};
        std::size_t const dimensions = this->dimensions();
        if (queries.columns != dimensions) return {{}, status_t::config_mismatch_k};

        // A window holds whole tokens, so none splits its heads across two.
        std::size_t const window_rows = max_of<std::size_t>(window_rows_k / head_count, 1) * head_count;
        batch_t batch {executor, this->allocator_};
        if (status_t const begun = this->begin_batch_(batch, min_of(rows_count, window_rows), config, head_count,
                                                      shards);
            failed(begun))
            return {{}, begun};
        auto attend_window = [&](std::size_t first, std::size_t taking) noexcept -> expected<search_result_t> {
            rows_view_t const window_queries = queries.slice(first, taking);
            rows_span_t const window_outputs = outputs.slice(first, taking);
            std::span<softmax_state_t> const window_carry = carry.empty() ? carry : carry.subspan(first, taking);
            // Shard 0 stages one block of rows and carries per shard, its own first and the others
            // copied in.
            std::size_t const outputs_bytes = taking * dimensions * sizeof(f32_t);
            std::size_t const block_bytes = outputs_bytes + taking * sizeof(softmax_state_t);
            byte_t* blocks = nullptr;
            traversal_context_t* first_traversal = nullptr;
            auto attend_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
                shard_lookup_t const lookup {device_keys_[shard].base(), value_lookup};
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
                               cuda_copy_async(blocks + shard * block_bytes, landed.value, block_bytes,
                                               traversal.stream()) == cuda_success_k
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
                if (cuda_set_device(first_traversal->device()) != cuda_success_k ||
                    failed(shards_[0].index->launch_merge_attention_(blocks, rows, static_cast<u32_t>(dimensions),
                                                                     static_cast<u32_t>(shards), stream)) ||
                    failed(shards_[0].index->land_attended_(reinterpret_cast<f32_t*>(blocks), shards, window_outputs,
                                                            window_carry, stream)) ||
                    failed(first_traversal->wait()))
                    merged = status_t::device_error_k;
            };
            expected<search_result_t> attended = this->for_each_shard_(batch, attend_shard, merge_shard, {});
            if (!attended) return attended;
            if (failed(merged)) return {{}, merged};
            attended.value.count = taking;
            return attended;
        };
        return for_each_window<search_result_t>(rows_count, window_rows, attend_window);
    }

  private:
    friend base_t;
    using base_t::shards_;

    /** One unreserved shard placed on @p device; null when the allocation fails. */
    std::unique_ptr<index_t> build_(int device) const noexcept {
        typename index_t::vectors_allocator_t const allocator {device};
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(this->kernel_, allocator, allocator));
    }

    /** One unreserved shard on the device @p like sits on. */
    std::unique_ptr<index_t> rebuild_(index_t const& like) const noexcept {
        return build_(like.device_specs().ordinal);
    }

    /** Appends @p shard with its host key column, a device twin and the staging between them on its
     *  device, where the staging's events are born. */
    status_t place_(std::unique_ptr<index_t>&& shard) noexcept {
        int const device = shard->device_specs().ordinal;
        if (cuda_set_device(device) != cuda_success_k) return status_t::device_error_k;
        if (failed(device_keys_.push_back(cuda_virtual_memory_allocator<key_t> {device}))) return status_t::bad_alloc_k;
        if (failed(key_stagings_.push_back(key_staging_t {}))) return device_keys_.pop_back(), status_t::bad_alloc_k;
        status_t const placed = base_t::place_(std::move(shard));
        if (failed(placed)) device_keys_.pop_back(), key_stagings_.pop_back();
        return placed;
    }

    /** Covers rows below @p rows in both of shard @p shard 's key columns, reserving the device one
     *  at its first cover, and widens the staging between them to a power of two of rows, up to a
     *  window per half. */
    status_t cover_keys_(std::size_t shard, std::size_t rows) noexcept {
        if (status_t const covered = shards_[shard].keys.cover(rows); failed(covered)) return covered;
        cuda_virtual_memory_allocator<key_t>& device_keys = device_keys_[shard];
        if (!device_keys.reserved_bytes())
            if (status_t const reserved = device_keys.reserve_range(index_t::max_node_count_k * sizeof(key_t));
                failed(reserved))
                return reserved;
        key_staging_t& staging = key_stagings_[shard];
        std::size_t const half_keys = min_of(window_rows_k, std::bit_ceil(rows));
        if (rows && staging.keys.size() < 2 * half_keys) {
            // Pinning needs a current context, and the copies out of the old halves must land.
            if (failed(enter_(shard)) || failed(staging.crossed[0].wait()) || failed(staging.crossed[1].wait()))
                return status_t::device_error_k;
            staging.keys.reset();
            if (failed(staging.keys.resize_uninitialized(2 * half_keys))) return status_t::bad_alloc_k;
        }
        return device_keys.commit(rows * sizeof(key_t));
    }

    /** Makes shard @p shard 's device current, since a launch fails on a stream of any other. */
    status_t enter_(std::size_t shard) const noexcept {
        return cuda_set_device(shards_[shard].index->device_specs().ordinal) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Copies @p rows of shard @p shard 's host key column into its device twin on @p stream,
     *  through the shard's staging half by half, since a pageable source would copy synchronously;
     *  the host refills a half only once the copy out of it landed. */
    status_t upload_keys_(std::size_t shard, iota_view<std::size_t> rows, cuda_stream_handle_t stream) noexcept {
        key_staging_t& staging = key_stagings_[shard];
        std::size_t const half_keys = staging.keys.size() / 2;
        if (rows.empty()) return status_t::success_k;
        if (!half_keys) return status_t::uninitialized_k;
        key_t* const device_keys = device_keys_[shard].base();
        for (std::size_t first = rows.first; first < rows.bound; first += half_keys) {
            std::size_t const bound = min_of(first + half_keys, rows.bound);
            std::size_t const half = staging.chunks++ % 2;
            key_t* staged = staging.keys.data() + half * half_keys;
            if (staging.chunks > 2 && failed(staging.crossed[half].wait())) return status_t::device_error_k;
            shards_[shard].keys.for_each_span({first, bound}, [&](std::span<key_t> run) noexcept {
                for (key_t const& key : run) *staged++ = key;
            });
            if (cuda_copy_async(device_keys + first, staging.keys.data() + half * half_keys,
                                (bound - first) * sizeof(key_t), stream) != cuda_success_k ||
                failed(staging.crossed[half].record(stream)))
                return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** The upload of a load's keys rides the load's own stream. */
    status_t upload_keys_(std::size_t shard, iota_view<std::size_t> rows,
                          typename index_t::load_plan_t const& plan) noexcept {
        return upload_keys_(shard, rows, plan.stream);
    }

    /** Queues donor shard @p from 's columns by @p step of shard @p into 's load @p plan on the
     *  donor's stream, then its device keys on the plan's: another device's keys may not map here,
     *  so they cross from the host column. */
    status_t land_part_(std::size_t into, typename index_t::load_plan_t& plan,
                        typename index_t::load_plan_t::step_t const& step, vram_nvidia_indexes const& donor,
                        std::size_t from, executor_t& executor) noexcept {
        index_t const& part = *donor.shards_[from].index;
        if (status_t const queued = plan.load(step, part, step.source_rows,
                                              executor.stream(donor.stream_of_(executor, from)));
            failed(queued))
            return queued;
        node_id_t const base = step.index_rows.first, rows = step.source_rows.bound;
        if (part.device_specs().ordinal != shards_[into].index->device_specs().ordinal)
            return upload_keys_(into, {base, base + rows}, plan.stream);
        return cuda_copy_async(device_keys_[into].base() + base, donor.device_keys_[from].base(), rows * sizeof(key_t),
                               plan.stream) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Marks a context's submission done queuing on its stream. */
    static status_t record_(traversal_context_t& traversal) noexcept { return traversal.record(); }

    /** Opens shard @p shard 's save on its device's stream in @p executor. */
    expected<typename index_t::save_plan_t> plan_shard_save_(std::size_t shard,
                                                             typename index_t::columns_t const& columns,
                                                             synchronization_t synchronization, std::size_t first_row,
                                                             executor_t* executor) noexcept {
        if (status_t const entered = enter_(shard); failed(entered)) return {{}, entered};
        return shards_[shard].index->plan_save(columns, synchronization,
                                               executor->stream(this->stream_of_(*executor, shard)),
                                               static_cast<node_id_t>(first_row));
    }

    /** Opens shard @p shard 's load on its device's stream in @p executor. */
    expected<typename index_t::load_plan_t> plan_shard_load_(std::size_t shard, std::size_t rows,
                                                             executor_t* executor) noexcept {
        if (status_t const entered = enter_(shard); failed(entered)) return {{}, entered};
        return shards_[shard].index->plan_load(rows, executor->stream(this->stream_of_(*executor, shard)));
    }

    /** The body of the batched @c search into rankings on the first shard's device, when all three
     *  live there: every shard resolves its walks into keys on its own device, every further one
     *  copies its ranking to the first, and one merge there lands the caller's, rank for rank as
     *  the host merge would. @c deferred_k when they live elsewhere. */
    template <typename score_type_, typename... heads_types_>
    expected<search_result_t> search_on_device_(batch_t& batch, rows_view_t queries, search_config_t config,
                                                nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                                std::span<std::size_t> counts, heads_types_ const&... heads) noexcept {
        using distance_t = typename base_t::distance_t;
        int const first_device = shards_[0].index->device_specs().ordinal;
        if (cuda_device_of(keys.data()) != first_device || cuda_device_of(distances.data()) != first_device ||
            cuda_device_of(counts.data()) != first_device)
            return {{}, status_t::deferred_k};
        std::size_t const queries_count = queries.rows, shards = shards_.size();
        // Every caller of `search_` counts as many matches as the keys are wide.
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
            this->allocator_};
        growing_buffer<std::size_t, forkunion_allocator_t> landed_counts {this->allocator_};
        if (failed(landings.reserve_exactly(shards)) || failed(landed_counts.resize(queries_count)))
            return {{}, status_t::bad_alloc_k};
        traversal_context_t* first_traversal = nullptr;
        auto search_shard = [&](std::size_t shard, traversal_context_t& traversal, std::size_t visible) noexcept {
            std::size_t const blocks = shards == 1 ? 0 : shard ? 1 : shards;
            landings.push_back_reserved(cuda_make_unique(
                cuda_device_allocator_t {batch.executor.stream(this->stream_of_(batch.executor, shard))},
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
                    ids, walked_distances, width, reinterpret_cast<byte_t const*>(device_keys_[shard].base()),
                    shards == 1 ? merged : ranking_of(block), static_cast<u32_t>(queries_count), traversal.stream());
                failed(resolved) || blocks != 1)
                return resolved;
            // On the shard's own stream, so the copy follows its resolution; unified addressing
            // tells the two devices apart. The first shard's landing was allocated on its stream,
            // which its context marked when it queued, so following that mark orders the copy after
            // the allocation without the host waiting.
            return succeeded(traversal.follow(*first_traversal)) &&
                           cuda_copy_async(landings[0].get() + walked_bytes + shard * block_bytes, block, block_bytes,
                                           traversal.stream()) == cuda_success_k
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
            if (cuda_set_device(first_traversal->device()) != cuda_success_k ||
                (shards != 1 && failed(shards_[0].index->launch_merge_matches_(
                                    ranking_of(landings[0].get() + walked_bytes), block_bytes,
                                    static_cast<u32_t>(shards), merged, static_cast<u32_t>(queries_count), stream))) ||
                cuda_copy_async(landed_counts.data(), counts.data(), counts.size_bytes(), stream) != cuda_success_k ||
                failed(first_traversal->wait()))
                merged_status = status_t::device_error_k;
        };
        expected<search_result_t> found = this->for_each_shard_(batch, search_shard, merge_shard,
                                                                to_span(landed_counts));
        if (!found) return found;
        if (failed(merged_status)) return {{}, merged_status};
        return found;
    }

    /** The same keys as every shard's host column on its device, so a walk resolves its nodes to
     *  keys; reserved for every ID a shard can hold, since address space is free and the base never
     *  moves under a walk, and mapped as it grows. Parallel to the shards. */
    growing_buffer<cuda_virtual_memory_allocator<key_t>, forkunion_allocator_t> device_keys_;

    /** One shard's page-locked ring between its two key columns: two halves of keys, each refilled
     *  only once the copy out of it landed. Sized as plans cover rows and freed with the index. */
    struct key_staging_t {
        growing_buffer<key_t, cuda_pinned_allocator_t> keys {cuda_pinned_allocator_t {}};

        /** Marks the last copy out of each half. */
        cuda_event_t crossed[2];

        /** Halves filled so far, alternating between the two. */
        std::size_t chunks = 0;
    };

    /** Every shard's key staging, parallel to the shards. */
    growing_buffer<key_staging_t, forkunion_allocator_t> key_stagings_;
};

} // namespace unum::usearch::dense
