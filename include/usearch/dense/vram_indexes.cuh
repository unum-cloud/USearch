/**
 *  @brief Typed multi-shard views over GPU-resident dense indexes - one shard per memory
 *      domain, stitched into one traversable graph, hosting the merger machinery too.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/vram_indexes.cuh
 *  @date July 23, 2026
 */
#pragma once

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the shard, base & stream tables
#include "usearch/dense/vram_index.cuh"

namespace unum {
namespace usearch {

#if defined(__CUDACC__)

/** @brief Globalizes one shard's neighbor plane in place: every valid id gains the shard's
 *      global base, sentinels stay sentinels - self-loop publication markers translate
 *      correctly because a row's own id shifts by the very same base. */
inline __global__ void dense_cuda_globalize_kernel(u32_t* words, std::size_t words_count, u32_t base) {
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t word = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; word < words_count;
         word += stride)
        if (words[word] != dense_node_id_unknown_k) words[word] += base;
}

/**
 *  @brief Re-links one already-resident node through the stitched union: search from its
 *      own vector, diverse-select, rewrite the row under its local lock - staling the old
 *      component-local links - then back-link into every reachable component. The
 *      insertion flow minus payload landing and publication accounting.
 *
 *  The in-place sibling of @ref dense_cuda_refine_one: the refinement passes own the
 *  graph exclusively and land shadow rows, while an index-group merge runs against components
 *  other devices may be re-linking concurrently, so it publishes through the same slot-0
 *  lock protocol the insertion path uses. Uniting the two on the shadow-plane convention
 *  is the natural next consolidation once the reverse scatter spans devices.
 */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_cuda_merge_one( //
    dense_cuda_build_view<kit_> const& view, u32_t sampling_bound, dense_node_id_t node_id,
    dense_cuda_visited_set_t visited, typename kit_::fragment_t* query_span,
    dense_cuda_vector_cache<typename kit_::fragment_t> cache, dense_cuda_relink_scratch_t relink,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], u32_t (&gathered_ids)[dense_cuda_cache_rows_k],
    dense_cuda_scored_node_t (&gathered_scored)[32], dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;
    // The query is the node's own resident payload; its norms are long derived.
    typename kit_::fragment_t const* own_vector = planes.vector_span_for(node_id);
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
        query_span[fragment] = own_vector[fragment];
    typename kit_::norms_t const own_norms = planes.norm_for(node_id);
    typename kit_::query_state_t const query_state = kit_::make_query_state(own_norms);
    __syncwarp();
    visited.test_and_set(node_id, lane, lane == 0); // Never selects itself as a link.

    dense_cuda_search_around<kit_>( //
        planes, query_span, query_state, sampling_bound, visited, gathered_ids, gathered_scored, context, lane);
    u32_t const selected_count = dense_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered_ids, relink.selected_ids, context, lane);

    // Rewrite the own row under its lock: fresh links in, stale component-local tail out.
    dense_cuda_neighbors_ref_t const own_row {planes.neighbors_for(node_id), planes.connectivity};
    dense_node_id_t displaced = dense_node_id_unknown_k;
    if (lane == 0) displaced = own_row.lock_if_free(1u << 20, context.random_generator);
    displaced = __shfl_sync(0xFFFFFFFFu, displaced, 0);
    if (displaced == dense_node_id_unknown_k) return; // Beyond the budget - keep the old links.
    for (u32_t slot = lane; slot < planes.connectivity; slot += 32)
        if (slot > 0)
            own_row.store_slot(slot, slot < selected_count ? relink.selected_ids[slot] : dense_node_id_unknown_k);
    __syncwarp();
    if (lane == 0) own_row.unlock(selected_count ? relink.selected_ids[0] : node_id);

    dense_cuda_back_link_neighbors<kit_>( //
        planes, node_id, own_norms, relink, selected_count, cache, cache_norms, gathered_ids, gathered_scored,
        context.random_generator, context.telemetry, lane);
}

/** @brief The index-group merge driver: dispenses node ids @b [first_id,end_id) off the batch
 *      cursor, one warp per node - the insert kernel's frame without the admission ramp,
 *      since every id in range is long published. */
template <dense_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_sm_k)
    dense_cuda_merge_shards_kernel(dense_cuda_build_view<kit_> view, u32_t sampling_bound) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_cuda_relink_scratch_t const relink = dense_cuda_relink_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_cuda_cache_rows_k);
    while (true) {
        u32_t node_id;
        if (lane == 0) node_id = view.first_id + atomicAdd(view.batch_cursor, 1u);
        node_id = __shfl_sync(0xFFFFFFFFu, node_id, 0);
        if (node_id >= view.end_id) return;

        dense_cuda_warp_context<dense_cuda_register_frontier<pool_slots_>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(node_id) * 33 + lane)};
        dense_cuda_visited_set_t const visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + (node_id - view.first_id) + 1};
        dense_cuda_merge_one<kit_>( //
            view, sampling_bound, node_id, visited, query_span, cache, relink, cache_norms[warp], gathered_ids[warp],
            gathered_scored[warp], context, lane);
        context.telemetry.flush<true>(view.telemetry, lane);
    }
}

/**
 *  @brief A group of same-geometry indexes - one shard per memory domain - stitched into
 *      one traversable graph over a shared address space: the multi-GPU capacity tier.
 *
 *  @b make adopts shards whose planes already share a transparent address space and
 *  whose neighbor ids are already global - the multi-node fabric-handle adoption path.
 *  @b merge takes independently built shards on peer devices and does the whole
 *  progressive unification: globalize neighbor ids in place, alias every physical slab
 *  into the shared windows via @c cuMemRetainAllocationHandle - zero copies - replicate
 *  the norms per device, then merge each source shard into the target graph. Serial
 *  ordering runs the merges over a coherent global-id prefix for the last point of
 *  recall; concurrent ordering launches every source on its own device stream - ONE host
 *  thread, no host threading, wall clock of the largest source.
 *
 *  @section dense_vram_indexes_constraints Constraints the measurements dictated
 *
 *  Shards must be vmm-backed - managed planes cannot be aliased by handle, and a managed
 *  norms mirror measured a 40x latency collapse under two concurrent readers, so every
 *  device owns a full LOCAL norms replica instead. Every shard but the last must hold a
 *  whole-slab row quota so global ids stay dense and landmark sampling never probes an
 *  unmapped hole. Merged shards are no longer standalone until their neighbor ids are
 *  localized again by subtracting the bases.
 */
template <nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, cuda_arch_t arch_ceiling_ = cuda_arch_blackwell_k,
          cuda_reservable_allocator vectors_allocator_type_ = cuda_virtual_memory_allocator<std::byte>,
          cuda_reservable_allocator neighbors_allocator_type_ = cuda_virtual_memory_allocator<std::byte>>
struct dense_vram_indexes {
    using index_t = dense_vram_index<vector_dtype_, kernel_kind_, arch_ceiling_, vectors_allocator_type_,
                                     neighbors_allocator_type_>;
    using kit_t = typename index_t::kit_t;
    using scalar_t = typename index_t::scalar_t;
    using fragment_t = typename index_t::fragment_t;
    using norms_t = typename index_t::norms_t;
    using node_id_t = typename index_t::node_id_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using planes_view_t = typename index_t::planes_view_t;
    using small_size_t = typename index_t::small_size_t;

    /** @brief How source shards enter the target graph: serial ordering keeps landmark
     *      sampling over a coherent global-id prefix - the last point of recall;
     *      concurrent ordering launches every source on its own device stream - the
     *      wall-clock floor. */
    enum class merge_order_t : u32_t { serial_k, concurrent_k };
    /** @brief One merge's tuning: the insertion pool each source re-links with - the
     *      quality-vs-wall knob, ef 128 halves the wall for under a recall point - and
     *      the ordering of the source merges. */
    struct merge_config_t {
        small_size_t expansion_factor = 256;
        merge_order_t order = merge_order_t::serial_k;
    };

    dense_vram_indexes() = default;
    dense_vram_indexes(dense_vram_indexes&& other) noexcept { swap(other); }
    dense_vram_indexes& operator=(dense_vram_indexes&& other) noexcept { return swap(other), *this; }
    dense_vram_indexes(dense_vram_indexes const&) = delete;
    dense_vram_indexes& operator=(dense_vram_indexes const&) = delete;
    ~dense_vram_indexes() noexcept { release(); }

    void swap(dense_vram_indexes& other) noexcept {
        std::swap(shards_, other.shards_);
        std::swap(bases_, other.bases_);
        std::swap(stitched_planes_, other.stitched_planes_);
        std::swap(vectors_window_, other.vectors_window_);
        std::swap(neighbors_window_, other.neighbors_window_);
        std::swap(vectors_window_bytes_, other.vectors_window_bytes_);
        std::swap(neighbors_window_bytes_, other.neighbors_window_bytes_);
        std::swap(norms_replicas_, other.norms_replicas_);
        std::swap(streams_, other.streams_);
        std::swap(merged_, other.merged_);
    }

    std::size_t size() const noexcept { return bases_.size() ? bases_[shards_count()] : 0; }
    std::size_t shards_count() const noexcept { return shards_.size(); }
    /** @brief The shard owning global @p id and its local row within it. */
    std::pair<std::size_t, node_id_t> locate(node_id_t id) const noexcept {
        std::size_t shard = shards_count();
        while (shard != 0 && bases_[shard - 1] > id) --shard;
        return {shard - 1, id - static_cast<node_id_t>(bases_[shard - 1])};
    }

    /** @brief Adopts shards whose planes already live in one transparent address space with
     *      global neighbor ids - validation and registry only, no re-mapping. */
    [[nodiscard]] static expected<dense_vram_indexes> make(std::span<index_t*> shards) noexcept {
        dense_vram_indexes indexes;
        status_t const status = indexes.adopt(shards);
        return {std::move(indexes), status};
    }

    /**
     *  @brief Unifies independently built shards into one navigable graph: globalize,
     *      zero-copy aliasing, replicate norms, then one source merge per non-first shard.
     *      Synchronous and exclusive: no other work may touch the shards while it runs.
     */
    [[nodiscard]] static expected<dense_vram_indexes> merge(std::span<index_t*> shards,
                                                            merge_config_t config = {}) noexcept {
        dense_vram_indexes indexes;
        if (status_t status = indexes.adopt(shards); status != status_t::success_k) return {std::move(indexes), status};
        if (status_t status = indexes.globalize(); status != status_t::success_k) return {std::move(indexes), status};
        if (status_t status = indexes.stitch(); status != status_t::success_k) return {std::move(indexes), status};
        // Staged sampling needs a coherent id prefix, so its phases serialize; free
        // ordering launches every source at once on its own device stream.
        std::size_t const sources = indexes.shards_count() - 1;
        growing_buffer<merge_scratch_t> scratch;
        if (sources)
            if (status_t const sized = scratch.resize(sources); failed(sized)) return {std::move(indexes), sized};
        status_t status = status_t::success_k;
        if (config.order == merge_order_t::serial_k) {
            for (std::size_t slot = 1; slot != indexes.shards_count() && status == status_t::success_k; ++slot) {
                status = indexes.merge_begin(slot, config, scratch[slot - 1]);
                if (status == status_t::success_k) status = indexes.merge_end(slot, scratch[slot - 1]);
            }
        }
        else {
            for (std::size_t slot = 1; slot != indexes.shards_count() && status == status_t::success_k; ++slot)
                status = indexes.merge_begin(slot, config, scratch[slot - 1]);
            for (std::size_t slot = 1; slot != indexes.shards_count(); ++slot)
                if (status_t ended = indexes.merge_end(slot, scratch[slot - 1]); ended != status_t::success_k)
                    status = ended;
        }
        if (status != status_t::success_k) return {std::move(indexes), status};
        indexes.merged_ = true;
        return {std::move(indexes), status_t::success_k};
    }

    /** @brief The stitched planes for single-walk searches - shard 0's view with the three
     *      pointers swapped to the shared windows; feed the stock search launch with
     *      @c sampling_bound @c = @c size(). Valid only after a successful @c merge. */
    planes_view_t planes() const noexcept { return stitched_planes_; }
    /** @brief The stitched planes as seen from group slot @p slot: identical windows, but
     *      the norms pointer is that device's LOCAL replica and the resident range feeds
     *      the locality telemetry - the view every kernel on that device must use. */
    planes_view_t planes_for(std::size_t slot) const noexcept {
        planes_view_t planes = stitched_planes_;
        planes.norms = norms_replicas_[slot];
        planes.resident_begin = static_cast<u32_t>(bases_[slot]);
        planes.resident_end = static_cast<u32_t>(bases_[slot + 1]);
        return planes;
    }
    /** @brief The group slot's orchestration stream - all group launches on that device
     *      ride it, so ONE host thread drives the whole group asynchronously. */
    cudaStream_t stream_for(std::size_t slot) const noexcept { return streams_[slot]; }

  private:
    /** @brief The adopted shards, never owned; slot @b i owns the global rows
     *      @b [bases_[i],bases_[i+1]). */
    growing_buffer<index_t*> shards_;
    /** @brief Prefix row sums, one per shard plus the trailing group total. */
    growing_buffer<std::size_t> bases_;
    planes_view_t stitched_planes_ = {};
    CUdeviceptr vectors_window_ = 0, neighbors_window_ = 0;
    std::size_t vectors_window_bytes_ = 0, neighbors_window_bytes_ = 0;
    /** @brief One full norms replica per group device: every scored candidate reads a
     *      norm, and any shared placement collapses under concurrent readers. */
    growing_buffer<norms_t*> norms_replicas_;
    /** @brief One non-blocking stream per group device - the group's whole orchestration
     *      plane; kernel launches are asynchronous, so no host threading exists anywhere. */
    growing_buffer<cudaStream_t> streams_;
    bool merged_ = false;

    /** @brief Device scratch one merge phase owns for its lifetime. */
    struct merge_scratch_t {
        std::byte* arena = nullptr;
        dense_cuda_telemetry_block_t* telemetry = nullptr;
        u32_t* cursor = nullptr;
    };

    void release() noexcept {
        if (vectors_window_)
            cuMemUnmap(vectors_window_, vectors_window_bytes_),
                cuMemAddressFree(vectors_window_, vectors_window_bytes_);
        if (neighbors_window_)
            cuMemUnmap(neighbors_window_, neighbors_window_bytes_),
                cuMemAddressFree(neighbors_window_, neighbors_window_bytes_);
        for (std::size_t slot = 0; slot != norms_replicas_.size(); ++slot)
            if (norms_replicas_[slot]) cudaFree(norms_replicas_[slot]);
        for (std::size_t slot = 0; slot != streams_.size(); ++slot)
            if (streams_[slot]) cudaStreamDestroy(streams_[slot]);
        vectors_window_ = neighbors_window_ = 0;
    }

    /** @brief Registers the shards and validates the group invariants: uniform frozen
     *      geometry, and whole-slab row quotas on every shard but the last, so global ids
     *      continue densely across shard boundaries. Creates the per-device streams. */
    status_t adopt(std::span<index_t*> shards) noexcept {
        if (shards.empty()) return status_t::config_mismatch_k;
        std::size_t const dimensions = shards.front()->dimensions();
        std::size_t const connectivity = shards.front()->connectivity();
        if (!dimensions) return status_t::uninitialized_k;
        if (failed(shards_.resize(shards.size())) || failed(bases_.resize(shards.size() + 1)) ||
            failed(streams_.resize(shards.size())))
            return status_t::bad_alloc_k;
        std::size_t rows = 0;
        for (std::size_t slot = 0; slot != shards.size(); ++slot) {
            index_t* shard = shards[slot];
            if (shard->dimensions() != dimensions || shard->connectivity() != connectivity)
                return status_t::config_mismatch_k;
            shards_[slot] = shard;
            bases_[slot] = rows;
            rows += shard->size();
        }
        bases_[shards.size()] = rows;
        std::size_t const slab_rows = quota_rows_of(*shards.front());
        for (std::size_t slot = 0; slot + 1 != shards_count(); ++slot)
            if (shards_[slot]->size() % slab_rows) return status_t::config_mismatch_k;
        for (std::size_t slot = 0; slot != shards_count(); ++slot) {
            if (cudaSetDevice(shards_[slot]->device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
            if (cudaStreamCreateWithFlags(&streams_[slot], cudaStreamNonBlocking) != cudaSuccess)
                return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** @brief Rows per physical slab under the wider of the two re-mapped record sizes -
     *      the quota every non-last shard must be a multiple of. */
    static std::size_t quota_rows_of(index_t const& shard) noexcept {
        std::size_t const record = max_of(shard.row_bytes(), shard.connectivity() * sizeof(u32_t));
        return vectors_allocator_type_::preferred_slab_bytes_k / record;
    }

    status_t globalize() noexcept {
        for (std::size_t slot = 1; slot != shards_count(); ++slot) {
            index_t& shard = *shards_[slot];
            if (cudaSetDevice(shard.device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
            dense_cuda_globalize_kernel<<<1024, 256>>>(shard.planes().neighbors, shard.size() * shard.connectivity(),
                                                       static_cast<u32_t>(bases_[slot]));
            if (cudaDeviceSynchronize() != cudaSuccess) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** @brief Aliases every shard's physical slabs into the shared windows by retained
     *      handle and grants the whole device group access; every device also gathers a
     *      full local norms replica - 4-byte records defy slab quotas, and replicas keep
     *      every scored read local. */
    status_t stitch() noexcept {
        index_t const& first = *shards_[0];
        std::size_t const row_bytes = first.row_bytes();
        std::size_t const neighbors_row_bytes = first.connectivity() * sizeof(u32_t);
        std::size_t const slab_bytes = vectors_allocator_type_::preferred_slab_bytes_k;
        for (std::size_t slot = 0; slot != shards_count(); ++slot) {
            vectors_window_bytes_ += fu::round_up_to_multiple(shards_[slot]->size() * row_bytes, slab_bytes);
            neighbors_window_bytes_ += fu::round_up_to_multiple(shards_[slot]->size() * neighbors_row_bytes,
                                                                slab_bytes);
        }
        if (cuMemAddressReserve(&vectors_window_, vectors_window_bytes_, 0, 0, 0) != CUDA_SUCCESS ||
            cuMemAddressReserve(&neighbors_window_, neighbors_window_bytes_, 0, 0, 0) != CUDA_SUCCESS)
            return status_t::bad_alloc_k;
        if (status_t const sized = norms_replicas_.resize(shards_count()); failed(sized)) return sized;
        for (std::size_t slot = 0; slot != shards_count(); ++slot) {
            if (cudaSetDevice(shards_[slot]->device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
            if (cudaMalloc(&norms_replicas_[slot], size() * sizeof(norms_t)) != cudaSuccess)
                return status_t::bad_alloc_k;
        }

        growing_buffer<CUmemAccessDesc> access;
        if (status_t const sized = access.resize(shards_count()); failed(sized)) return sized;
        for (std::size_t slot = 0; slot != shards_count(); ++slot) {
            access[slot].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access[slot].location.id = shards_[slot]->device_specs().ordinal;
            access[slot].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        for (std::size_t slot = 0; slot != shards_count(); ++slot) {
            index_t& shard = *shards_[slot];
            planes_view_t const planes = shard.planes();
            if (status_t status = alias(vectors_window_ + bases_[slot] * row_bytes, planes.vectors,
                                        fu::round_up_to_multiple(shard.size() * row_bytes, slab_bytes), slab_bytes,
                                        access);
                status != status_t::success_k)
                return status;
            if (status_t status = alias(neighbors_window_ + bases_[slot] * neighbors_row_bytes, planes.neighbors,
                                        fu::round_up_to_multiple(shard.size() * neighbors_row_bytes, slab_bytes),
                                        slab_bytes, access);
                status != status_t::success_k)
                return status;
            for (std::size_t replica = 0; replica != shards_count(); ++replica)
                if (cudaMemcpyPeer(norms_replicas_[replica] + bases_[slot], shards_[replica]->device_specs().ordinal,
                                   planes.norms, shard.device_specs().ordinal,
                                   shard.size() * sizeof(norms_t)) != cudaSuccess)
                    return status_t::device_error_k;
        }
        stitched_planes_ = first.planes();
        stitched_planes_.vectors = reinterpret_cast<fragment_t*>(vectors_window_);
        stitched_planes_.norms = norms_replicas_[0];
        stitched_planes_.neighbors = reinterpret_cast<u32_t*>(neighbors_window_);
        return status_t::success_k;
    }

    static status_t alias(CUdeviceptr window, void* plane_base, std::size_t committed_bytes, std::size_t slab_bytes,
                          growing_buffer<CUmemAccessDesc> const& access) noexcept {
        for (std::size_t offset = 0; offset < committed_bytes; offset += slab_bytes) {
            CUmemGenericAllocationHandle handle;
            if (cuMemRetainAllocationHandle(&handle, reinterpret_cast<std::byte*>(plane_base) + offset) != CUDA_SUCCESS)
                return status_t::device_error_k;
            bool const mapped = cuMemMap(window + offset, slab_bytes, 0, handle, 0) == CUDA_SUCCESS;
            cuMemRelease(handle);
            if (!mapped) return status_t::device_error_k;
        }
        if (cuMemSetAccess(window, committed_bytes, access.data(), access.size()) != CUDA_SUCCESS)
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /** @brief Queues one source merge on its device's own stream and returns without
     *      waiting: every own-row lock is local, only back-links cross the fabric, and the
     *      norms come from the source's local replica. Pair with @c merge_end. */
    status_t merge_begin(std::size_t slot, merge_config_t config, merge_scratch_t& scratch) noexcept {
        index_t& source = *shards_[slot];
        cuda_device_specs_t const& specs = source.device_specs();
        if (cudaSetDevice(specs.ordinal) != cudaSuccess) return status_t::device_error_k;
        u32_t const pool_slots = min_of(max_of(std::bit_ceil(divide_round_up<u32_t>(config.expansion_factor, 32)), 2u),
                                        16u);
        auto const arena_shape = index_t::arena_shape_for(pool_slots * 32, index_t::frontier_kind_t::register_ladder_k);
        u32_t const blocks = static_cast<u32_t>(specs.sm_count) * dense_cuda_resident_blocks_per_sm_k;
        std::size_t const arena_bytes = static_cast<std::size_t>(blocks) * dense_cuda_warps_per_block_k *
                                        arena_shape.slice_bytes();
        cudaStream_t const stream = streams_[slot];
        if (cudaMalloc(&scratch.arena, arena_bytes) != cudaSuccess ||
            cudaMemsetAsync(scratch.arena, 0, arena_bytes, stream) != cudaSuccess ||
            cudaMalloc(&scratch.telemetry, sizeof(dense_cuda_telemetry_block_t)) != cudaSuccess ||
            cudaMemsetAsync(scratch.telemetry, 0, sizeof(dense_cuda_telemetry_block_t), stream) != cudaSuccess ||
            cudaMalloc(&scratch.cursor, sizeof(u32_t)) != cudaSuccess ||
            cudaMemsetAsync(scratch.cursor, 0, sizeof(u32_t), stream) != cudaSuccess)
            return status_t::bad_alloc_k;

        dense_cuda_build_view<kit_t> view {planes_for(slot),
                                           nullptr, // Payloads are long resident.
                                           static_cast<u32_t>(bases_[slot + 1]),
                                           source.insert_scratch_bytes_per_warp(),
                                           static_cast<node_id_t>(bases_[slot]),
                                           scratch.arena,
                                           arena_shape,
                                           0,
                                           nullptr, // No admission ramp, no publication accounting.
                                           scratch.telemetry,
                                           scratch.cursor};
        u32_t const sampling_bound = static_cast<u32_t>(
            config.order == merge_order_t::serial_k ? bases_[slot + 1] : bases_[shards_count()]);
        dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
            dense_cuda_launch<dense_cuda_merge_shards_kernel<kit_t, slots.value>>(view, blocks, stream, sampling_bound);
        });
        return status_t::success_k;
    }

    /** @brief Waits for the phase's stream and releases its scratch. */
    status_t merge_end(std::size_t slot, merge_scratch_t& scratch) noexcept {
        if (cudaSetDevice(shards_[slot]->device_specs().ordinal) != cudaSuccess) return status_t::device_error_k;
        status_t const status = cudaStreamSynchronize(streams_[slot]) == cudaSuccess ? status_t::success_k
                                                                                     : status_t::device_error_k;
        if (scratch.cursor) cudaFree(scratch.cursor);
        if (scratch.telemetry) cudaFree(scratch.telemetry);
        if (scratch.arena) cudaFree(scratch.arena);
        scratch = {};
        return status;
    }
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
