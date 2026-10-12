/**
 *  @file include/usearch/dense/vram_apple_index.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Vector Search index for the unified memory of one Apple GPU.
 *
 *  The Apple twin of @c vram_nvidia_index: the same flat storage, the same walk, insert and refine
 *  kernels ported to Metal in @c vram_apple_index.metal, and the same verbs. Memory is unified, so
 *  every matrix and column is a shared buffer the host reads and writes in place: checkpoints copy
 *  without a bounce, and the reachability audit and the island reconnection run on the host over
 *  the storage the device just wrote. A matrix grows by allocating a wider buffer, under an
 *  exclusive hold that every walk shares while its command buffer runs.
 *
 *  Metal promises no forward progress between command buffers, while an insert's admission ramp
 *  waits on rows earlier batches publish, so batches run one at a time; searches overlap them
 *  freely. The kernels compile from the embedded source at the first @c reserve, once per storage
 *  dtype and device.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <atomic>       // `std::memory_order`
#include <bit>          // `std::bit_ceil`, `std::bit_width`, `std::countr_zero`, `std::has_single_bit`
#include <mutex>        // `std::unique_lock` - the batch hold
#include <shared_mutex> // `std::shared_mutex` - walks share the storage, growth takes it alone
#include <span>         // `std::span`
#include <utility>      // `std::exchange`

#include <forkunion/types.hpp> // `fu::standard_yield_t` - the watermark's turn

#include <numkong/capabilities.h> // `nk_kernel_kind_t`
#include <numkong/tensor.hpp>     // `nk::matrix_span`, `nk::range`
#include <numkong/types.hpp>      // `nk::type_for`
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_apple_device.hpp"
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL kits, embedded so the kernels compile on the device without a Metal toolchain. */
inline constexpr char metal_metric_source_k[] = {
#embed "vram_apple_metric.metal" suffix(, 0)
};

/** The MSL walk, insert, refine and search kernels, compiled after the kits. */
inline constexpr char metal_index_source_k[] = {
#embed "vram_apple_index.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** The kit name @c vram_apple_metric.metal gives @p dtype, or null for a dtype it carries none
 *  for. */
constexpr char const* metal_kit_name(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_bf16_k: return "bf16_metal_quads_kit_t";
    case nk_f16_k: return "f16_metal_quads_kit_t";
    case nk_e4m3_k: return "e4m3_metal_quads_kit_t";
    case nk_e5m2_k: return "e5m2_metal_quads_kit_t";
    case nk_e3m2_k: return "e3m2_metal_quads_kit_t";
    case nk_e2m3_k: return "e2m3_metal_quads_kit_t";
    case nk_i8_k: return "i8_metal_quads_kit_t";
    case nk_u8_k: return "u8_metal_quads_kit_t";
    case nk_i4_k: return "i4x2_metal_quads_kit_t";
    case nk_u4_k: return "u4x2_metal_quads_kit_t";
    case nk_u1_k: return "u1x8_metal_quads_kit_t";
    default: return nullptr;
    }
}

/** The whole MSL source for @p dtype_ - the kits, the alias naming its kit, then the kernels -
 *  assembled once per process; its address keys the pipelines built from it. */
template <nk_dtype_t dtype_>
char const* metal_source() noexcept {
    constexpr std::size_t alias_capacity_k = 128;
    static char source[sizeof(metal_metric_source_k) + sizeof(metal_index_source_k) + alias_capacity_k];
    static std::once_flag assembled;
    std::call_once(assembled, [] {
        std::size_t length = sizeof(metal_metric_source_k) - 1;
        std::memcpy(source, metal_metric_source_k, length);
        length += static_cast<std::size_t>(
            std::snprintf(source + length, alias_capacity_k,
                          "\nnamespace unum { namespace usearch { namespace dense {\nusing metal_kit_t = %s;\n}}}\n",
                          metal_kit_name(dtype_)));
        std::memcpy(source + length, metal_index_source_k, sizeof(metal_index_source_k));
    });
    return source;
}

/** Matches the widest register pool holds; wider walks clamp to it. */
inline constexpr u32_t metal_widest_pool_k = 512;

/** The walking kernels @c vram_apple_index.metal instantiates once per pool width. */
enum class metal_kernel_t : u32_t {
    insert_k,
    refine_k,
    search_k,
};

/** Every scalar a launch carries, word for word the @c metal_arguments_t of the kernels. */
struct metal_arguments_t {
    u32_t row_fragments = 0, lanes_per_row = 0, steps_per_row = 0, connectivity = 0;
    u32_t metric = 0, has_norms = 0, first_id = 0, end_id = 0;
    u32_t sampling_bound = 0, queries_count = 0, count = 0, epoch_base = 0;
    u32_t visited_slots = 0, hash_shift = 0, cache_rows = 0, slice_bytes = 0;
    u32_t heads_count = 0, row_fragments_magic = 0, use_node_ids = 0, padding = 0;
};

/** The counter words a launch binds: the batch's assigned rows, the published rows, then the walks'
 *  telemetry, and the word an attend raises when a node has no value row. */
enum class metal_counter_t : u32_t {
    assigned_k = 0,
    published_k,
    hops_k,
    scored_k,
    restarts_k,
    lock_attempts_k,
    lock_skips_k,
    reselections_k,
    locked_drops_k,
    saturations_k,
    changed_k,
    missed_k,
    count_k = 16,
};

/** What a search launch lands: ranked distances, or every scored pair of a product. */
enum class metal_output_t : bool { distances_k, products_k };

/** One caller's scratch for driving an Apple index: the walks' arena, the staging of one batch, the
 *  telemetry counters and the single-query landing. Two contexts on one index share only the
 *  storage. */
template <typename buffers_allocator_type_ = metal_shared_allocator_t>
struct metal_traversal_context {

    /** The queue the context submits to, named an allocator because the residency-level harnesses
     *  build every context from its @c allocator_t. */
    using allocator_t = metal_queue_t;

    /** The backend of the arena, the staging and the counters, on the queue's device. */
    using buffers_allocator_t = buffers_allocator_type_;

    /** Matches the single-query verb lands at most. */
    inline static constexpr u32_t results_capacity_k = 64;

    /** The single-query verb's landing: ascending matches in inline storage. */
    struct results_t {
        scored_node_t entries_[results_capacity_k];
        std::size_t count_ = 0;

        scored_node_t* data() noexcept { return entries_; }
        scored_node_t const* data() const noexcept { return entries_; }
        scored_node_t& operator[](std::size_t rank) noexcept { return entries_[rank]; }
        scored_node_t const& operator[](std::size_t rank) const noexcept { return entries_[rank]; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** Binds the context to @p queue's device; buffers wait for the first verb. */
    metal_traversal_context(allocator_t queue = {}) noexcept
        : queue_(std::move(queue)), context_(metal_context(queue_.device)), buffers_allocator_(queue_.device) {
        if (context_) counters_ = buffers_allocator_.allocate(static_cast<std::size_t>(metal_counter_t::count_k) * 4);
    }
    metal_traversal_context(metal_traversal_context&&) noexcept = default;
    metal_traversal_context(metal_traversal_context const&) = delete;
    metal_traversal_context& operator=(metal_traversal_context const&) = delete;

    int device() const noexcept { return queue_.device; }
    void* queue() const noexcept { return queue_.handle() ? queue_.handle() : context_->queue; }
    explicit operator bool() const noexcept { return context_ && counters_; }

    /** Waits for the last command buffer this context committed - one queue's completion, the
     *  executor's @c drain being the fleet's. */
    status_t drain() noexcept { return committed_.drain(); }

    /** Polls whether the last command buffer this context committed finished - @c success_k once
     *  it has, @c deferred_k while it runs - without blocking, so one host thread awaits whichever
     *  of several queues lands first. */
    status_t landed() const noexcept { return committed_.landed(); }

    /** The telemetry of the last submission, zeroed when it was staged; read once it landed. */
    vram_telemetry_block_t spent() const noexcept {
        vram_telemetry_block_t block {};
        block.hops = word_(metal_counter_t::hops_k);
        block.scored = word_(metal_counter_t::scored_k);
        block.restarts = word_(metal_counter_t::restarts_k);
        block.lock_attempts = word_(metal_counter_t::lock_attempts_k);
        block.lock_skips = word_(metal_counter_t::lock_skips_k);
        block.reselections = word_(metal_counter_t::reselections_k);
        block.locked_drops = word_(metal_counter_t::locked_drops_k);
        block.visited_saturations = word_(metal_counter_t::saturations_k);
        block.changed_nodes = word_(metal_counter_t::changed_k);
        return block;
    }

    /** Whether the last attend met a node its value lookup had no row for, which refuses it; read
     *  once it landed. */
    bool missed() const noexcept { return word_(metal_counter_t::missed_k) != 0; }

  private:
    template <nk_dtype_t, nk_capability_t, typename, typename, typename>
    friend struct vram_apple_index;

    /** Where the last walk left its outputs in the staging, read back once the context lands. */
    struct landing_t {
        std::size_t rows = 0, count = 0;
        std::size_t first_offset = 0, second_offset = 0, third_offset = 0;
    };

    allocator_t queue_;
    metal_context_t* context_ = nullptr;
    buffers_allocator_t buffers_allocator_;
    metal_buffer_t counters_;
    metal_buffer_t arena_;
    metal_buffer_t staging_;
    landing_t landing_;

    /** The last command buffer submitted through this context; its queue runs them in commit order,
     *  so this one landing means every earlier one did. */
    metal_launch_t committed_;

    /** The counter word @p counter, read in place from shared memory. */
    u32_t word_(metal_counter_t counter) const noexcept {
        u32_t const* const words = reinterpret_cast<u32_t const*>(counters_.host);
        return words[static_cast<u32_t>(counter)];
    }

    /** Monotone epoch floor over the arena's visited slots, so the arena zeroes only when fresh or
     *  wrapped. */
    u64_t epoch_floor_ = 0;

    /** Dispatches @p launch over @p groups threadgroups of @p threads and keeps it as the command
     *  buffer the context awaits. */
    status_t submit_(metal_launch_t& launch, std::size_t groups, std::size_t threads) noexcept {
        if (status_t const submitted = launch.submit(groups, threads); failed(submitted)) return submitted;
        committed_ = std::move(launch);
        return status_t::success_k;
    }

    /** Grows @p buffer to at least @p bytes; answers whether it was reallocated, zeroed by
     *  Metal. */
    status_t ensure_(metal_buffer_t& buffer, std::size_t bytes, bool& fresh) noexcept {
        fresh = false;
        if (bytes <= buffer.bytes) return status_t::success_k;
        buffer = buffers_allocator_.allocate(bytes);
        fresh = true;
        return buffer ? status_t::success_k : status_t::bad_alloc_k;
    }
};

/**
 *  @brief Vector Search index for the unified memory of one Apple GPU.
 *
 *  Speaks @c vram_nvidia_index 's verbs: a batch's IDs are one contiguous range turning visible a
 *  whole batch at a time, and the first successful @c reserve freezes the geometry. Attention is
 *  refused with @c unsupported_k until its value-lookup ABI is designed for Metal, and walks past
 *  the widest register pool of 512 clamp to it.
 */
template <                                                          //
    nk_dtype_t vector_dtype_ = nk_i8_k,                             //
    nk_capability_t capability_ = nk_cap_metal_k,                   //
    typename vectors_allocator_type_ = metal_shared_allocator_t,    //
    typename neighbors_allocator_type_ = metal_shared_allocator_t,  //
    typename traversals_allocator_type_ = metal_shared_allocator_t> //
struct vram_apple_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    static_assert(metal_kit_name(vector_dtype_) != nullptr, "No Metal kit stores this dtype");

    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;
    using dot_t = typename dot_result_t::raw_t;
    using norms_t = vector_norms<dot_t>;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_view_t = nk::vector_view<scalar_t>;
    using node_id_t = dense::node_id_t;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using distance_t = dense::distance_t;
    using small_size_t = u32_t;
    using traversal_context_t = metal_traversal_context<traversals_allocator_t>;
    using refine_result_t = dense::refine_result_t;
    using health_result_t = dense::health_result_t;
    static_assert(sizeof(norms_t) == 8, "The kernels address norms as two 32-bit words");

    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;
    inline static constexpr std::size_t max_node_count_k = 0xFFFFFFFEu;

    /** Fragment bytes - the kits read rows sixteen bytes at a time, and rows pad to whole
     *  fragments. */
    inline static constexpr std::size_t fragment_bytes_k = 16;

    /** Simdgroups per threadgroup and threadgroups per launch, at most - 256 walks in flight. */
    inline static constexpr std::size_t simdgroups_per_group_k = 4;
    inline static constexpr std::size_t groups_per_launch_k = 64;

    /** Rows of the per-simdgroup candidate cache, as on CUDA. */
    inline static constexpr std::size_t cache_rows_k = 64;

    /** The @p kernel every walk scores by, checked against the dtype at the first @c reserve, and
     *  one allocator per storage concern; the index pins itself to the vectors allocator's device
     *  and demands the neighbors allocator agree. */
    explicit vram_apple_index(nk_kernel_kind_t kernel, vectors_allocator_t const& vectors_allocator = {},
                              neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, kernel_ {kernel}, vectors_allocator_ {vectors_allocator},
          neighbors_allocator_ {neighbors_allocator} {}
    vram_apple_index(vram_apple_index const&) = delete;
    vram_apple_index& operator=(vram_apple_index const&) = delete;
    ~vram_apple_index() noexcept {
        vectors_allocator_.deallocate(vectors_matrix_);
        vectors_allocator_.deallocate(norms_column_);
        neighbors_allocator_.deallocate(neighbors_matrix_);
    }

#pragma region Measures

    /** Rows visible to searches. */
    std::size_t size() const noexcept {
        return standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed rows unpublished, kept for the index's lifetime. */
    status_t status() const noexcept {
        return standard_atomic_ref<status_t const>(status_).load(std::memory_order_acquire);
    }

    /** Rows the storage holds without growing. */
    std::size_t capacity() const noexcept { return capacity_rows_; }
    std::size_t connectivity() const noexcept { return connectivity_; }
    std::size_t dimensions() const noexcept { return dimensions_; }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    std::size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : connectivity() * 2;
    }

    /** The device's identity and limits, queried at the first @c reserve. */
    metal_device_specs_t const& device_specs() const noexcept { return device_; }
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }
    bool stores_norms() const noexcept { return kernel_ != nk_kernel_dot_k; }

    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }
    u32_t row_fragments() const noexcept {
        return static_cast<u32_t>(divide_round_up<std::size_t>(packed_row_bytes(), fragment_bytes_k));
    }
    std::size_t row_bytes() const noexcept { return row_fragments() * fragment_bytes_k; }

    /** Layout of one simdgroup's slice of the traversal arena. */
    struct arena_shape_t {
        u32_t visited_slots = 0;
        u32_t visited_hash_shift = 0;
        std::size_t slice_bytes = 0;
    };

    /** One simdgroup's arena slice for a walk expanding @p expansion candidates beside
     *  @p cache_rows cached rows: the exact visited set, sized well past the expected visit count
     *  so probing stays short, the frontier and batch spans, then the query and cache rows - in
     *  device memory, since threadgroup memory would bound the row width. */
    arena_shape_t arena_shape_for(u32_t expansion, std::size_t cache_rows) const noexcept {
        u32_t const visited_slots = std::bit_ceil(64 * expansion);
        std::size_t const spans_bytes = 32 * 8 + 64 * 8 + 64 * 8 + 64 * 4 + 32 * 4 + 64 * 4 + 64 * 4;
        std::size_t const rows_bytes = row_bytes() * (1 + cache_rows);
        return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)),
                round_up_to_multiple<std::size_t>(rows_bytes + visited_slots * 8 + spans_bytes, 16)};
    }

    /** Whether a search or multiply launch seats on this device: every simdgroup in flight takes
     *  the widest pool's slice and the @p fold_bytes an attend keeps past it, within one buffer. */
    bool walk_scratch_fits(std::size_t fold_bytes = 0) const noexcept {
        return groups_per_launch_k * simdgroups_per_group_k *
                   (arena_shape_for(metal_widest_pool_k, 0).slice_bytes + fold_bytes) <=
               device_.max_buffer_bytes;
    }

    /** Whether an insert or refine launch seats on this device: every simdgroup in flight takes
     *  the widest pool's slice beside its candidate cache, within one buffer. */
    bool build_scratch_fits() const noexcept {
        return groups_per_launch_k * simdgroups_per_group_k *
                   arena_shape_for(metal_widest_pool_k, cache_rows_k).slice_bytes <=
               device_.max_buffer_bytes;
    }

    /** Bytes this index occupies: its storage and its counters. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_apple_index) + vectors_matrix_.bytes + norms_column_.bytes + neighbors_matrix_.bytes +
               counters_.bytes;
    }

    /** Copies the stored row of @p node into @p destination with the fragment padding stripped. */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination) const noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        std::shared_lock storage {storage_};
        std::memcpy(destination.byte_data(), vectors_matrix_.host + static_cast<std::size_t>(node) * row_bytes(),
                    packed_row_bytes());
        return status_t::success_k;
    }

    /** Whether every claimed ID is published; call it with no batch in flight. */
    status_t validate() const noexcept {
        return published_() == claimed_rows_ ? status_t::success_k : status_t::device_error_k;
    }

    /** The serialized extents. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /** Where a save lands: row-major host columns the caller sized from @ref saved_shape. */
    struct columns_t {
        bytes_span_t vectors;
        bytes_span_t neighbors;
        bytes_span_t norms;
    };

    /** Where a load reads from - the read-only twin of @ref columns_t. */
    struct columns_view_t {
        bytes_view_t vectors;
        bytes_view_t neighbors;
        bytes_view_t norms;
    };

#pragma endregion Measures

#pragma region Reservation

    /** Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;
    };

    /** Ensures the index can host at least @p config.capacity entries; the first success freezes
     *  the geometry. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        bool const scored = vector_dtype_ == nk_u1_k
                                ? kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_hamming_k ||
                                      kernel_ == nk_kernel_jaccard_k
                                : kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_angular_k ||
                                      kernel_ == nk_kernel_sqeuclidean_k || kernel_ == nk_kernel_euclidean_k;
        if (!scored) return {0, status_t::unknown_kind_k};
        if (!config.dimensions || !std::has_single_bit(static_cast<u32_t>(config.connectivity)))
            return {0, status_t::config_mismatch_k};
        std::unique_lock batch {batch_};
        if (dimensions_ &&
            (dimensions_ != config.dimensions || connectivity_ != static_cast<std::size_t>(config.connectivity)))
            return {capacity(), status_t::config_mismatch_k};
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        if (!dimensions_) {
            if (neighbors_allocator_.device() != device_.ordinal) return {0, status_t::config_mismatch_k};
            auto const [specs, status] = metal_device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            metal_context_t* const context = metal_context(device_.ordinal);
            if (!context) return {0, status_t::device_error_k};
            if (status_t const loaded = load_pipelines_(); failed(loaded)) return {0, loaded};
            counters_ = metal_allocate(*context, static_cast<std::size_t>(metal_counter_t::count_k) * 4);
            if (!counters_) return {0, status_t::bad_alloc_k};
            dimensions_ = config.dimensions;
            connectivity_ = static_cast<std::size_t>(config.connectivity);
        }
        status_t const grown = grow_rows_(config.capacity);
        return {capacity(), grown};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {
        small_size_t expansion_factor = 0;
    };

    /** The bytes a context grows to before it can insert @p rows rows: its counters, one arena
     *  slice per simdgroup in flight, and the staged rows; zero before @c reserve freezes the
     *  geometry. */
    std::size_t traversal_memory_usage(std::size_t rows, insert_config_t config = {}) const noexcept {
        if (!dimensions_) return 0;
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        return static_cast<std::size_t>(metal_counter_t::count_k) * sizeof(u32_t) +
               groups_for_(rows) * simdgroups_per_group_k * arena_shape_for(pool_slots * 32, cache_rows_k).slice_bytes +
               rows * row_bytes();
    }

    /** Telemetry of one @c insert() batch over the contiguous IDs @b [first_id,first_id+count). */
    struct [[nodiscard]] insert_result_t {
        std::size_t new_size = 0;
        node_id_t first_id = node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t unlinked_neighbors = 0;
        std::size_t tail_drops = 0;
        std::size_t visited_saturations = 0;
        std::size_t computed_distances_in_refines = 0;
        std::size_t computed_distances_in_reverse_refines = 0;
    };

    /** Inserts @p new_vectors under the next contiguous IDs, waits for the device, publishes them
     *  once every earlier batch has, and reports telemetry. */
    expected<insert_result_t> insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        expected<insert_result_t> inserted = submit_insert_(new_vectors, assigned_ids, context, config);
        if (!inserted || !new_vectors.extent(0)) return inserted;
        if (status_t const published = publish_(inserted.value.first_id, new_vectors.extent(0)); failed(published))
            return {{inserted.value.new_size, inserted.value.first_id}, published};
        return inserted;
    }

#pragma endregion Insertion

#pragma region Search

    /** Optional tuning knobs of @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {
        small_size_t expansion_factor = 0;
        small_size_t count = 10;
        synchronization_t synchronization = synchronization_t::concurrent_k;
    };

    /** Telemetry of one @c search() batch. */
    struct [[nodiscard]] search_result_t {
        std::size_t count = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t tail_drops = 0;
        std::size_t visited_saturations = 0;
    };

    /** The bytes a context grows to before it can search or multiply @p rows rows under @p config,
     *  landing @c config.count columns per row: its counters, one arena slice per simdgroup in
     *  flight, and the staged rows beside their outputs. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        if (!dimensions_) return 0;
        u32_t const expansion = max_of<u32_t>(
            config.expansion_factor > config.count ? config.expansion_factor : config.count * 2,
            static_cast<u32_t>(non_zero_expansion_factor(config)));
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        return static_cast<std::size_t>(metal_counter_t::count_k) * sizeof(u32_t) +
               groups_for_(rows) * simdgroups_per_group_k * arena_shape_for(pool_slots * 32, 0).slice_bytes +
               rows * (row_bytes() + config.count * (sizeof(node_id_t) + sizeof(distance_t)) + sizeof(u32_t));
    }

    /** The bytes a search under the indexer @p heads needs: the plain walk's, plus the staging of
     *  the heads and their weights. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config,
                                       indexer_heads<scalar_t> const& heads) const noexcept {
        return traversal_memory_usage(rows, config) + heads.heads.extent(0) * (row_bytes() + sizeof(f32_t));
    }

    /** The nearest published neighbors of every query row, ascending, with sentinel IDs past the
     *  live matches. */
    expected<search_result_t> search(vectors_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {}, completion_t = completion_t::blocking_k) noexcept {
        return search_(queries, result_ids, result_distances, context, config, size(), dummy_heads_t {});
    }

    /** The batched @c search ranked by a lightning indexer; a dot verb, refusing every other
     *  metric. */
    expected<search_result_t> search(vectors_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {}, completion_t = completion_t::blocking_k) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return search_(queries, result_ids, result_distances, context, config, size(), heads);
    }

    /** One query in, the nearest matches ascending in @c context.results. */
    expected<search_result_t> search(vector_view_t query_vector, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return search_one_(query_vector, context, config, size());
    }

    /** Graph products of every left-hand row: each keeps the pairs its walk scored, until its
     *  output row fills. */
    expected<search_result_t> multiply(vectors_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       traversal_context_t& context, search_config_t config = {},
                                       completion_t = completion_t::blocking_k) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return multiply_(left, node_ids, products, counts, context, config, size());
    }

    /** Device attention is NVIDIA's alone until its value lookup has a Metal ABI. */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t, std::size_t, value_lookup_type_&&, f32_t, nk::matrix_span<f32_t>,
                                     std::span<softmax_state_t>, traversal_context_t&, search_config_t = {},
                                     completion_t = completion_t::blocking_k) noexcept {
        return {{}, status_t::unsupported_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** Optional tuning knobs of @c refine(). */
    struct refine_config_t {
        small_size_t expansion_factor = 0;
    };

    /**
     *  @brief Re-selects the neighbors of every node in @p nodes from a fresh walk seeded by the
     *      ones it holds, then heals whatever no search can reach. An empty range heals.
     *
     *  The walk lands the re-selection in a shadow matrix that replaces the walked rows once the
     *  device finishes; unlike the CUDA pass no reverse merge follows, since the heal's audit and
     *  reconnection, run on the host over unified memory, restore whatever a re-selection strands.
     *  Demands an exclusive caller.
     */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
        std::unique_lock batch {batch_};
        std::size_t const rows = size();
        refine_result_t result;
        auto add_walks = [&](refine_result_t const& walked) noexcept {
            result.visited_nodes += walked.visited_nodes;
            result.changed_nodes += walked.changed_nodes;
            result.visited_members += walked.visited_members;
            result.computed_distances += walked.computed_distances;
            result.restarts += walked.restarts;
        };
        node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, rows));
        if (nodes.first < bound) {
            auto const [walked, status] = refine_pass_(context, config, {nodes.first, bound}, {});
            if (failed(status)) return {result, status};
            add_walks(walked);
        }
        if (!rows) return {result, status_t::success_k};

        audit_t audit;
        if (failed(audit.reached.resize(rows)) || failed(audit.frontier.resize(rows)) ||
            failed(audit.next_frontier.resize(rows)) || failed(audit.islands.reserve_exactly(rows)))
            return {result, status_t::bad_alloc_k};
        audit_(audit);
        result.sweeps += audit.sweeps;
        result.islands_found = audit.islands.size();
        for (std::size_t islands_before = audit.islands.size(); !audit.islands.empty();) {
            auto const [healed, healed_status] = refine_pass_(
                context, config, {0, static_cast<node_id_t>(audit.islands.size())}, to_span(audit.islands));
            if (failed(healed_status)) return {result, healed_status};
            add_walks(healed);
            audit_(audit);
            result.sweeps += audit.sweeps;
            if (audit.islands.empty()) break;
            result.islands_reconnected += reconnect_(audit);
            audit_(audit);
            result.sweeps += audit.sweeps;
            // A round that frees no island frees none on the next either, since the picks it needs
            // are unreached.
            if (audit.islands.size() >= islands_before) break;
            islands_before = audit.islands.size();
        }
        result.reachable_nodes = rows - audit.islands.size();
        return {result, status_t::success_k};
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws, on
     *  the host. */
    expected<health_result_t> health(traversal_context_t&) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_t::success_k};
        audit_t audit;
        if (failed(audit.reached.resize(rows)) || failed(audit.frontier.resize(rows)) ||
            failed(audit.next_frontier.resize(rows)) || failed(audit.islands.reserve_exactly(rows)))
            return {{}, status_t::bad_alloc_k};
        std::shared_lock storage {storage_};
        audit_(audit);
        return {{rows, rows - audit.islands.size(), audit.sweeps}, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /** One save in flight: the watermark it took and its steps, each copying rows straight out of
     *  the storage. */
    struct [[nodiscard]] save_plan_t {
        columns_t columns {};
        node_id_t id_shift = 0;
        node_id_t visible = 0;
        std::size_t batch_rows = 0;
        std::size_t covered_vectors = 0;
        std::size_t covered_neighbors = 0;

        save_plan_t() noexcept = default;
        save_plan_t(vram_apple_index const& index, columns_t columns, node_id_t id_shift, node_id_t visible) noexcept
            : columns(columns), id_shift(id_shift), visible(visible),
              batch_rows(min_of(static_cast<std::size_t>(visible), power_of_two<std::size_t>(20))), index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), id_shift(other.id_shift), visible(other.visible), batch_rows(other.batch_rows),
              covered_vectors(other.covered_vectors), covered_neighbors(other.covered_neighbors),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;
        ~save_plan_t() noexcept = default;

        struct step_t {
            std::size_t position = 0;
            iota_view<node_id_t> index_rows;
            iota_view<node_id_t> source_rows;
        };

        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * batch_rows;
            std::size_t const bound = min_of(first + batch_rows, static_cast<std::size_t>(visible));
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
            return step;
        }
        indexed_range<save_plan_t> steps() const noexcept {
            return {this, index_ && batch_rows ? divide_round_up<std::size_t>(visible, batch_rows) : 0};
        }

        /** Copies the step's rows out of the storage: vectors unpadded, neighbors rebased onto
         *  @c id_shift with every ID past the watermark folded to the sentinel, and norms where the
         *  columns take them. */
        status_t save(step_t const& step, nk::matrix_span<node_id_t>) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const rows = step.index_rows;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const first = step.source_rows.first;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, first, rows.size(), scalar_row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, first, rows.size(), neighbors_row_bytes);
            bytes_span_t const norms_rows = flat_rows(columns.norms, first, rows.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            std::shared_lock storage {index_->storage_};
            for (std::size_t row = 0; row != rows.size(); ++row)
                std::memcpy(vectors_rows.data() + row * scalar_row_bytes,
                            index_->vectors_matrix_.host + (rows.first + row) * index_->row_bytes(), scalar_row_bytes);
            if (!norms_rows.empty())
                std::memcpy(norms_rows.data(), index_->norms_column_.host + rows.first * sizeof(norms_t),
                            norms_rows.size());
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(rows.size(), std::memory_order_acq_rel);
            node_id_t const* const source = reinterpret_cast<node_id_t const*>(index_->neighbors_matrix_.host) +
                                            static_cast<std::size_t>(rows.first) * connectivity;
            node_id_t* const landed = reinterpret_cast<node_id_t*>(neighbors_rows.data());
            for (std::size_t slot = 0; slot != rows.size() * connectivity; ++slot)
                landed[slot] = source[slot] < visible ? static_cast<node_id_t>(source[slot] + id_shift)
                                                      : node_id_unknown_k;
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != nodes ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            index_ = nullptr;
            return status_t::success_k;
        }

      private:
        vram_apple_index const* index_ = nullptr;
    };

    /** Opens a save into @p columns; @c concurrent_k refuses, since the storage cannot be probed
     *  for settlement. */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    node_id_t id_shift = 0) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return {{}, status_t::config_mismatch_k};
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        std::size_t const rows = size();
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        return {save_plan_t {*this, columns, id_shift, static_cast<node_id_t>(rows)}, status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread. */
    status_t save(columns_t columns, synchronization_t synchronization,
                  nk::matrix_span<node_id_t> scratch = {}) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /** One load in flight: the rows it appends past every claimed one, holding the batch hold for
     *  its whole life; @ref finish derives the norms no step carried and publishes. */
    struct [[nodiscard]] load_plan_t {
        columns_view_t columns {};
        iota_view<node_id_t> source_range {};
        node_id_t base = 0;
        std::size_t rows = 0;
        std::size_t batch_rows = 0;
        std::size_t covered_vectors = 0;
        std::size_t covered_neighbors = 0;
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;
        load_plan_t(vram_apple_index& index, std::unique_lock<std::mutex>&& batch, node_id_t base,
                    std::size_t rows) noexcept
            : base(base), rows(rows), batch_rows(min_of(rows, power_of_two<std::size_t>(20))), index_(&index),
              batch_(std::move(batch)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), source_range(other.source_range), base(other.base),
              rows(std::exchange(other.rows, 0)), batch_rows(other.batch_rows), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), covered_norms(other.covered_norms),
              index_(std::exchange(other.index_, nullptr)), batch_(std::move(other.batch_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;
        ~load_plan_t() noexcept = default;

        struct step_t {
            std::size_t position = 0;
            iota_view<node_id_t> index_rows;
            iota_view<node_id_t> source_rows;
        };

        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = static_cast<std::size_t>(base) + position * batch_rows;
            std::size_t const bound = min_of(first + batch_rows, static_cast<std::size_t>(base) + rows);
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            return step;
        }
        indexed_range<load_plan_t> steps() const noexcept {
            return {this, index_ && batch_rows ? divide_round_up(rows, batch_rows) : 0};
        }

        status_t load(step_t const& step) noexcept { return load(step, columns, source_range); }

        /** Copies the step's rows out of @p columns into the storage: vectors padded, norms where
         *  carried, neighbors rebased from @p source_range onto the appended range and
         *  compacted. */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const first = step.source_rows.first;
            bytes_view_t const vectors_rows = flat_rows(columns.vectors, first, range.size(), scalar_row_bytes);
            bytes_view_t const neighbors_rows = flat_rows(columns.neighbors, first, range.size(), neighbors_row_bytes);
            bytes_view_t const norms_rows = flat_rows(columns.norms, first, range.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes();
            for (std::size_t row = 0; row != range.size(); ++row) {
                byte_t* const landing = index_->vectors_matrix_.host + (range.first + row) * row_bytes;
                std::memcpy(landing, vectors_rows.data() + row * scalar_row_bytes, scalar_row_bytes);
                std::memset(landing + scalar_row_bytes, 0, row_bytes - scalar_row_bytes);
            }
            if (index_->stores_norms() && !norms_rows.empty()) {
                std::memcpy(index_->norms_column_.host + range.first * sizeof(norms_t), norms_rows.data(),
                            norms_rows.size());
                standard_atomic_ref<std::size_t> {covered_norms}.fetch_add(range.size(), std::memory_order_acq_rel);
            }
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(range.size(), std::memory_order_acq_rel);
            std::span<node_id_t> const landed {reinterpret_cast<node_id_t*>(index_->neighbors_matrix_.host) +
                                                   static_cast<std::size_t>(range.first) * connectivity,
                                               range.size() * connectivity};
            std::memcpy(landed.data(), neighbors_rows.data(), neighbors_rows.size());
            rebase_neighbors(landed, source_range, shift, node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, node_id_unknown_k);
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Derives the norms no step carried on the device, lifts the hold, and publishes the
         *  appended rows. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != rows ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                if (index_->stores_norms() &&
                    standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows)
                    if (status_t const derived = index_->derive_norms_(base, rows); failed(derived))
                        return index_->fail_(derived);
                index_->claimed_rows_ = base + rows;
                u32_t* const counters = reinterpret_cast<u32_t*>(index_->counters_.host);
                counters[static_cast<u32_t>(metal_counter_t::published_k)] = static_cast<u32_t>(base + rows);
                batch_.unlock();
            }
            vram_apple_index* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_t::success_k;
        }

      private:
        vram_apple_index* index_ = nullptr;
        std::unique_lock<std::mutex> batch_;
    };

    /** Opens a load of @p rows from @p columns past every claimed one. */
    expected<load_plan_t> plan_load(columns_view_t columns, std::size_t rows,
                                    iota_view<node_id_t> source_range) noexcept {
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_load(rows);
        if (failed(status)) return {{}, status};
        plan.columns = columns;
        plan.source_range = source_range;
        return {std::move(plan), status};
    }

    /** Opens a load of @p rows the caller sources per step, as the sharded index drives its
     *  parts. */
    expected<load_plan_t> plan_load(std::size_t rows) noexcept {
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (!rows) return {load_plan_t {*this, {}, static_cast<node_id_t>(claimed_rows_), 0}, status_t::success_k};
        std::unique_lock batch {batch_, std::try_to_lock};
        if (!batch.owns_lock()) return {{}, status_t::contended_k};
        node_id_t const base = static_cast<node_id_t>(claimed_rows_);
        if (rows > max_node_count_k - base) return {{}, status_t::overflow_k};
        if (status_t const grown = grow_rows_(base + rows); failed(grown)) return {{}, grown};
        return {load_plan_t {*this, std::move(batch), base, rows}, status_t::success_k};
    }

    /** Appends the whole source on the calling thread. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range) noexcept {
        auto [plan, status] = plan_load(columns, rows, source_range);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

    /** The sharded twin publishes and fails its shard's batches, and walks under its own
     *  snapshot. */
    template <typename, nk_dtype_t, nk_capability_t, typename>
    friend struct vram_apple_indexes;

  private:

    /** The reachability audit's host state, reused across a heal's rounds. */
    struct audit_t {
        growing_buffer<u8_t> reached {global_allocator};
        growing_buffer<u8_t> frontier {global_allocator};
        growing_buffer<u8_t> next_frontier {global_allocator};
        growing_buffer<u32_t> islands {global_allocator};
        std::size_t sweeps = 0;
    };

#pragma region Batches

    /** The device half of @c insert: claims the next rows, stages and runs the batch, and returns
     *  with the IDs, the rows invisible until the caller publishes them - which the sharded twin
     *  does for every step at once. */
    expected<insert_result_t> submit_insert_(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                             traversal_context_t& context, insert_config_t config) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        if (!rows) return {{claimed_rows_, static_cast<node_id_t>(claimed_rows_)}, status_t::success_k};
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (new_vectors.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (new_vectors.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
            return {{}, status_t::unsupported_layout_k};

        std::unique_lock batch {batch_};
        std::size_t const first_id = claimed_rows_;
        if (rows > max_node_count_k - first_id) return {{}, status_t::overflow_k};
        if (status_t const grown = grow_rows_(first_id + rows); failed(grown)) return {{}, grown};
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, rows * row_bytes(), fresh); failed(staged))
            return {{}, staged};
        stage_rows_(context.staging_.host, new_vectors);
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(context, arguments, pool_slots * 32, cache_rows_k, rows); failed(sized))
            return {{}, sized};
        arguments.first_id = static_cast<u32_t>(first_id);
        arguments.end_id = static_cast<u32_t>(first_id + rows);
        reset_counters_(counters_);
        claimed_rows_ = first_id + rows;

        // Batches run one at a time, so the batch drains under its hold before the next claims.
        status_t launched = status_t::device_error_k;
        {
            std::shared_lock storage {storage_};
            metal_launch_t launch {context.queue(), pipeline_(metal_kernel_t::insert_k, pool_slots)};
            if (launch) {
                bind_storage_(launch, arguments);
                launch.bind(context.staging_, 4);
                launch.bind(context.arena_, 5);
                launch.bind(counters_, 6);
                launched = context.submit_(launch, groups_for_(rows), simdgroups_per_group_k * 32);
                if (succeeded(launched)) launched = context.drain();
            }
        }
        if (failed(launched)) return {{first_id, static_cast<node_id_t>(first_id)}, fail_(launched)};

        insert_result_t result;
        result.first_id = static_cast<node_id_t>(first_id);
        result.new_size = first_id + rows;
        result.visited_members = counter_(counters_, metal_counter_t::hops_k);
        result.computed_distances = counter_(counters_, metal_counter_t::scored_k);
        result.restarts = counter_(counters_, metal_counter_t::restarts_k);
        result.unlinked_neighbors = counter_(counters_, metal_counter_t::lock_skips_k);
        result.visited_saturations = counter_(counters_, metal_counter_t::saturations_k);
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<node_id_t>(first_id + row);
        return {result, status_t::success_k};
    }

#pragma endregion Batches

#pragma region Launches

    /** The scalars every launch shares. */
    metal_arguments_t arguments_() const noexcept {
        u32_t const fragments = row_fragments();
        u32_t const lanes_per_row = min_of(std::bit_ceil(fragments), 32u);
        metal_arguments_t arguments;
        arguments.row_fragments = fragments;
        arguments.lanes_per_row = lanes_per_row;
        arguments.steps_per_row = divide_round_up(fragments, lanes_per_row);
        arguments.connectivity = static_cast<u32_t>(connectivity());
        arguments.metric = metric_();
        arguments.has_norms = stores_norms() ? 1 : 0;
        arguments.row_fragments_magic = static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments);
        return arguments;
    }

    /** The kernel kind lowered onto the metric the kits name. */
    u32_t metric_() const noexcept {
        switch (kernel_) {
        case nk_kernel_dot_k: return 0;
        case nk_kernel_angular_k: return 1;
        case nk_kernel_sqeuclidean_k: return 2;
        case nk_kernel_euclidean_k: return 3;
        case nk_kernel_hamming_k: return 4;
        default: return 5;
        }
    }

    /** Threadgroups for @p work walks, at most @c groups_per_launch_k. */
    static std::size_t groups_for_(std::size_t work) noexcept {
        return min_of(groups_per_launch_k, max_of<std::size_t>(1, divide_round_up(work, simdgroups_per_group_k)));
    }

    /** Sizes @p context 's arena for @p work walks over pools of @p expansion with @p cache_rows
     *  cache rows, zeroing it when fresh or when the epochs wrap, and fills the slice shape and the
     *  epoch base into @p arguments. */
    status_t shape_arena_(traversal_context_t& context, metal_arguments_t& arguments, u32_t expansion,
                          std::size_t cache_rows, std::size_t work) noexcept {
        arena_shape_t const shape = arena_shape_for(expansion, cache_rows);
        std::size_t const slices = groups_for_(work) * simdgroups_per_group_k;
        bool fresh = false;
        if (status_t const sized = context.ensure_(context.arena_, slices * shape.slice_bytes, fresh); failed(sized))
            return sized;
        if (fresh || context.epoch_floor_ + work > 0xFFFFFFFEull) {
            if (!fresh) std::memset(context.arena_.host, 0, context.arena_.bytes);
            context.epoch_floor_ = 0;
        }
        arguments.visited_slots = shape.visited_slots;
        arguments.hash_shift = shape.visited_hash_shift;
        arguments.cache_rows = static_cast<u32_t>(cache_rows);
        arguments.slice_bytes = static_cast<u32_t>(shape.slice_bytes);
        arguments.epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += work;
        return status_t::success_k;
    }

    /** Copies @p rows into @p staging at the vectors matrix pitch, the fragment padding zeroed. */
    void stage_rows_(byte_t* staging, vectors_view_t rows) const noexcept {
        std::size_t const scalar_row_bytes = packed_row_bytes();
        for (std::size_t row = 0; row != rows.extent(0); ++row) {
            byte_t* const landing = staging + row * row_bytes();
            std::memcpy(landing, rows.byte_data() + row * rows.stride_bytes(0), scalar_row_bytes);
            std::memset(landing + scalar_row_bytes, 0, row_bytes() - scalar_row_bytes);
        }
    }

    /** Zeroes @p counters' assigned row and telemetry, keeping the published row. */
    static void reset_counters_(metal_buffer_t& counters) noexcept {
        u32_t* const words = reinterpret_cast<u32_t*>(counters.host);
        u32_t const published = words[static_cast<u32_t>(metal_counter_t::published_k)];
        std::memset(words, 0, counters.bytes);
        words[static_cast<u32_t>(metal_counter_t::published_k)] = published;
    }
    static u32_t counter_(metal_buffer_t const& counters, metal_counter_t counter) noexcept {
        u32_t const* const words = reinterpret_cast<u32_t const*>(counters.host);
        return words[static_cast<u32_t>(counter)];
    }

    /** Binds the arguments, the two matrices and the norms column, the vectors standing in for an
     *  empty norms column. */
    void bind_storage_(metal_launch_t& launch, metal_arguments_t const& arguments) const noexcept {
        launch.bind_bytes(&arguments, sizeof(arguments), 0);
        launch.bind(vectors_matrix_, 1);
        launch.bind(stores_norms() ? norms_column_ : vectors_matrix_, 2);
        launch.bind(neighbors_matrix_, 3);
    }

    /** The pipeline of @p kernel at @p pool_slots slots, built at the first @c reserve. */
    void* pipeline_(metal_kernel_t kernel, u32_t pool_slots) const noexcept {
        return pipelines_[static_cast<u32_t>(kernel)][std::countr_zero(pool_slots) - 1];
    }

    /** Builds every pipeline this dtype's kernels need, once at the first @c reserve, so no launch
     *  waits on the device's registry; @c contended_k while another index builds there. */
    status_t load_pipelines_() noexcept {
        metal_context_t* const context = metal_context(device_.ordinal);
        if (!context) return status_t::device_error_k;
        char const* const source = metal_source<vector_dtype_>();
        for (u32_t kernel = 0; kernel != 3; ++kernel)
            for (u32_t rung = 0; rung != 4; ++rung) {
                auto const [pipeline, status] = context->pipeline(
                    source, name_of_(static_cast<metal_kernel_t>(kernel), 2u << rung));
                if (failed(status)) return status;
                pipelines_[kernel][rung] = pipeline;
            }
        auto const [pipeline, status] = context->pipeline(source, "metal_derive_norms_kernel");
        derive_norms_pipeline_ = pipeline;
        return status;
    }

    /** The name of @p kernel's instantiation at @p pool_slots slots, a power of two up to 16. */
    static char const* name_of_(metal_kernel_t kernel, u32_t pool_slots) noexcept {
        static constexpr char const* names_k[][4] = {
            {"metal_insert_kernel_2", "metal_insert_kernel_4", "metal_insert_kernel_8", "metal_insert_kernel_16"},
            {"metal_refine_kernel_2", "metal_refine_kernel_4", "metal_refine_kernel_8", "metal_refine_kernel_16"},
            {"metal_search_kernel_2", "metal_search_kernel_4", "metal_search_kernel_8", "metal_search_kernel_16"},
        };
        return names_k[static_cast<u32_t>(kernel)][std::countr_zero(pool_slots) - 1];
    }

    /** Rounds the requested expansion to the pool ladder of 64, 128, 256 and 512 ranks. */
    static u32_t pool_slots_for_(u32_t expansion_factor, u32_t default_slots) noexcept {
        if (!expansion_factor) return default_slots;
        u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion_factor, 32));
        return min_of(max_of(slots, 2u), 16u);
    }

    /** Recomputes the norms of rows @b [first,first+rows) on the device. */
    status_t derive_norms_(std::size_t first, std::size_t rows) noexcept {
        metal_arguments_t arguments = arguments_();
        arguments.first_id = static_cast<u32_t>(first);
        arguments.end_id = static_cast<u32_t>(first + rows);
        metal_context_t* const context = metal_context(device_.ordinal);
        if (!context) return status_t::device_error_k;
        std::shared_lock storage {storage_};
        metal_launch_t launch {context->queue, derive_norms_pipeline_};
        if (!launch) return status_t::device_error_k;
        launch.bind_bytes(&arguments, sizeof(arguments), 0);
        launch.bind(vectors_matrix_, 1);
        launch.bind(norms_column_, 2);
        if (status_t const submitted = launch.submit(groups_for_(rows), simdgroups_per_group_k * 32); failed(submitted))
            return submitted;
        return launch.drain();
    }

#pragma endregion Launches

#pragma region Walks

    /** The batched search over the @p visible rows, under any indexer @p heads: submitted, awaited
     *  and read back. */
    template <typename heads_type_>
    expected<search_result_t> search_(vectors_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                      nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                      search_config_t config, std::size_t visible, heads_type_ const& heads) noexcept {
        std::size_t const queries_count = queries.extent(0);
        bool const wants_distances = result_distances.extent(0) != 0;
        if (!queries_count) return {{}, status_t::success_k};
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {{}, status_t::config_mismatch_k};
        if (expected<search_result_t> const submitted = submit_search_(queries, context, config, visible, heads);
            !submitted)
            return submitted;
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        land_search_(context, result_ids, result_distances);
        return {spent_(context, config.count), status_t::success_k};
    }

    /** The submission half of @c search: stages the queries and any indexer @p heads and launches
     *  the walks over the @p visible rows; until the context lands, its staging holds the matches
     *  @ref land_search_ reads back. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(vectors_view_t queries, traversal_context_t& context,
                                             search_config_t config, std::size_t visible,
                                             heads_type_ const& heads = {}) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        std::size_t const queries_count = queries.extent(0);
        context.landing_ = {};
        if (!queries_count) return {{}, status_t::success_k};
        if (config.count > metal_widest_pool_k) return {{}, status_t::config_mismatch_k};
        std::size_t heads_count = 0;
        if constexpr (indexer_k) {
            heads_count = heads.weights.extent(1);
            if (heads.weights.extent(0) != queries_count || heads.heads.extent(0) != queries_count * heads_count ||
                heads.heads.extent(1) != dimensions())
                return {{}, status_t::config_mismatch_k};
        }
        u32_t const expansion = config.expansion_factor > config.count ? config.expansion_factor : config.count * 2;
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        std::size_t const results = queries_count * config.count;
        std::size_t const rows_bytes = (queries_count + queries_count * heads_count) * row_bytes();
        std::size_t const weights_bytes = queries_count * heads_count * sizeof(f32_t);
        std::size_t const outputs_bytes = results * (sizeof(node_id_t) + sizeof(distance_t));
        auto staged = stage_walk_(queries, context, pool_slots, rows_bytes + weights_bytes + outputs_bytes, visible);
        if (!staged) return {{}, staged.status};
        metal_arguments_t arguments = staged.value;
        arguments.count = config.count;
        arguments.heads_count = static_cast<u32_t>(heads_count);
        byte_t* const staging = context.staging_.host;
        if constexpr (indexer_k) {
            stage_rows_(staging + queries_count * row_bytes(), heads.heads);
            for (std::size_t query = 0; query != queries_count; ++query)
                std::memcpy(staging + rows_bytes + query * heads_count * sizeof(f32_t),
                            heads.weights.byte_data() + query * heads.weights.stride_bytes(0),
                            heads_count * sizeof(f32_t));
        }
        std::size_t const ids_offset = rows_bytes + weights_bytes;
        std::size_t const distances_offset = ids_offset + results * sizeof(node_id_t);
        context.landing_ = {queries_count, config.count, ids_offset, distances_offset, 0};
        if (status_t const walked = launch_walk_(context, arguments, pool_slots, ids_offset, distances_offset,
                                                 distances_offset, queries_count * row_bytes(), rows_bytes,
                                                 metal_output_t::distances_k);
            failed(walked))
            return {{}, walked};
        return {{config.count}, status_t::success_k};
    }

    /** Copies the matches the last search on @p context left in its staging into @p result_ids
     *  and, unless it is empty, @p result_distances; call it once the context landed. */
    void land_search_(traversal_context_t const& context, nk::matrix_span<node_id_t> result_ids,
                      nk::matrix_span<distance_t> result_distances) const noexcept {
        auto const& landing = context.landing_;
        bool const wants_distances = result_distances.extent(0) != 0;
        auto const* const ids = reinterpret_cast<node_id_t const*>(context.staging_.host + landing.first_offset);
        auto const* const distances = reinterpret_cast<distance_t const*>(context.staging_.host +
                                                                          landing.second_offset);
        for (std::size_t query = 0; query != landing.rows; ++query)
            for (std::size_t rank = 0; rank != landing.count; ++rank) {
                result_ids(query, rank) = ids[query * landing.count + rank];
                if (wants_distances) result_distances(query, rank) = distances[query * landing.count + rank];
            }
    }

    /** The graph products over the @p visible rows: submitted, awaited and read back. */
    expected<search_result_t> multiply_(vectors_view_t left, nk::matrix_span<node_id_t> node_ids,
                                        nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                        traversal_context_t& context, search_config_t config,
                                        std::size_t visible) noexcept {
        std::size_t const rows_count = left.extent(0), wanted = node_ids.extent(1);
        if (node_ids.extent(0) != rows_count || products.extent(0) != rows_count || products.extent(1) != wanted ||
            counts.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (expected<search_result_t> const submitted = submit_multiply_(left, wanted, context, config, visible);
            !submitted)
            return submitted;
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        search_result_t result = spent_(context, 0);
        result.count = land_multiply_(context, node_ids, products, counts);
        return {result, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows and launches walks keeping up to
     *  @p wanted pairs each over the @p visible rows; until the context lands, its staging holds
     *  the pairs @ref land_multiply_ reads back. */
    expected<search_result_t> submit_multiply_(vectors_view_t left, std::size_t wanted, traversal_context_t& context,
                                               search_config_t config, std::size_t visible) noexcept {
        std::size_t const rows_count = left.extent(0);
        context.landing_ = {};
        if (!rows_count) return {{}, status_t::success_k};
        u32_t const pool_slots = pool_slots_for_(static_cast<u32_t>(non_zero_expansion_factor(config)), 2);
        std::size_t const rows_bytes = rows_count * row_bytes();
        std::size_t const ids_bytes = rows_count * wanted * sizeof(node_id_t);
        std::size_t const products_bytes = rows_count * wanted * sizeof(dot_t);
        auto staged = stage_walk_(left, context, pool_slots, rows_bytes + ids_bytes + products_bytes + rows_count * 4,
                                  visible);
        if (!staged) return {{}, staged.status};
        metal_arguments_t arguments = staged.value;
        arguments.count = static_cast<u32_t>(wanted);
        std::size_t const products_offset = rows_bytes + ids_bytes;
        std::size_t const counts_offset = products_offset + products_bytes;
        context.landing_ = {rows_count, wanted, rows_bytes, products_offset, counts_offset};
        if (status_t const walked = launch_walk_(context, arguments, pool_slots, rows_bytes, products_offset,
                                                 counts_offset, 0, 0, metal_output_t::products_k);
            failed(walked))
            return {{}, walked};
        return {{}, status_t::success_k};
    }

    /** Copies the pairs and the scored counts the last multiply on @p context left in its staging
     *  into @p node_ids, @p products and @p counts, and answers the pairs scored in all; call it
     *  once the context landed. */
    std::size_t land_multiply_(traversal_context_t const& context, nk::matrix_span<node_id_t> node_ids,
                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts) const noexcept {
        auto const& landing = context.landing_;
        byte_t const* const staging = context.staging_.host;
        auto const* const ids = reinterpret_cast<node_id_t const*>(staging + landing.first_offset);
        auto const* const dots = reinterpret_cast<dot_t const*>(staging + landing.second_offset);
        auto const* const scored = reinterpret_cast<u32_t const*>(staging + landing.third_offset);
        std::size_t total = 0;
        for (std::size_t row = 0; row != landing.rows; ++row) {
            for (std::size_t pair = 0; pair != landing.count; ++pair) {
                node_ids(row, pair) = ids[row * landing.count + pair];
                products(row, pair) = dot_result_t::from_raw(dots[row * landing.count + pair]);
            }
            counts[row] = scored[row];
            total += scored[row];
        }
        return total;
    }

    /** The single-query search over the @p visible rows. */
    expected<search_result_t> search_one_(vector_view_t query_vector, traversal_context_t& context,
                                          search_config_t config, std::size_t visible) noexcept {
        context.results.count_ = 0;
        if (!visible) return {{}, status_t::success_k};
        if (query_vector.size() != dimensions()) return {{}, status_t::config_mismatch_k};
        u32_t const expansion = static_cast<u32_t>(non_zero_expansion_factor(config));
        u32_t const count = min_of<u32_t>(min_of<u32_t>(traversal_context_t::results_capacity_k, expansion),
                                          static_cast<u32_t>(visible));
        node_id_t staged_ids[traversal_context_t::results_capacity_k];
        distance_t staged_distances[traversal_context_t::results_capacity_k];
        vectors_view_t const one_query {query_vector.data(), {1, query_vector.size()}};
        config.count = count;
        expected<search_result_t> result = search_(one_query, nk::matrix_span<node_id_t> {staged_ids, {1, count}},
                                                   nk::matrix_span<distance_t> {staged_distances, {1, count}}, context,
                                                   config, visible, dummy_heads_t {});
        if (!result) return result;
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != node_id_unknown_k; ++live)
            context.results.entries_[live] = scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        result.value.count = live;
        return result;
    }

    /** Validates @p queries, sizes the arena and the staging, and stages the query rows: what every
     *  walking verb does before its kernel. */
    expected<metal_arguments_t> stage_walk_(vectors_view_t queries, traversal_context_t& context, u32_t pool_slots,
                                            std::size_t staging_bytes, std::size_t visible) noexcept {
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (queries.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        if (queries.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
            return {{}, status_t::unsupported_layout_k};
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(context, arguments, pool_slots * 32, 0, queries.extent(0));
            failed(sized))
            return {{}, sized};
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, staging_bytes, fresh); failed(staged))
            return {{}, staged};
        stage_rows_(context.staging_.host, queries);
        arguments.sampling_bound = static_cast<u32_t>(visible);
        arguments.queries_count = static_cast<u32_t>(queries.extent(0));
        std::memset(context.counters_.host, 0, context.counters_.bytes);
        return {arguments, status_t::success_k};
    }

    /** Submits the search kernel over the staged queries, its outputs at the given staging offsets;
     *  an empty index lands its sentinels at once. */
    status_t launch_walk_(traversal_context_t& context, metal_arguments_t const& arguments, u32_t pool_slots,
                          std::size_t ids_offset, std::size_t second_offset, std::size_t third_offset,
                          std::size_t heads_offset, std::size_t weights_offset, metal_output_t output) noexcept {
        bool const keeps = output == metal_output_t::products_k;
        if (!arguments.sampling_bound) {
            // An empty index matches nothing: every rank is a sentinel and every row scored none.
            auto* const ids = reinterpret_cast<node_id_t*>(context.staging_.host + ids_offset);
            for (std::size_t slot = 0; slot != std::size_t {arguments.queries_count} * arguments.count; ++slot)
                ids[slot] = node_id_unknown_k;
            if (keeps) std::memset(context.staging_.host + third_offset, 0, arguments.queries_count * sizeof(u32_t));
            return status_t::success_k;
        }
        u32_t const keeps_word = keeps ? 1 : 0;
        std::shared_lock storage {storage_};
        metal_launch_t launch {context.queue(), pipeline_(metal_kernel_t::search_k, pool_slots)};
        if (!launch) return status_t::device_error_k;
        bind_storage_(launch, arguments);
        launch.bind(context.staging_, 4);
        launch.bind(context.arena_, 5);
        launch.bind(context.counters_, 6);
        launch.bind(context.staging_, 7, ids_offset);
        launch.bind(context.staging_, 8, keeps ? 0 : second_offset);
        launch.bind(context.staging_, 9, keeps ? second_offset : 0);
        launch.bind(context.staging_, 10, keeps ? third_offset : 0);
        launch.bind(context.staging_, 11, heads_offset);
        launch.bind(context.staging_, 12, weights_offset);
        launch.bind_bytes(&keeps_word, sizeof(keeps_word), 13);
        return context.submit_(launch, groups_for_(arguments.queries_count), simdgroups_per_group_k * 32);
    }

    /** The telemetry the last walk on @p context spent, with @p count matches per query. */
    static search_result_t spent_(traversal_context_t const& context, std::size_t count) noexcept {
        vram_telemetry_block_t const spent = context.spent();
        search_result_t result;
        result.count = count;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.visited_saturations = spent.visited_saturations;
        return result;
    }

#pragma endregion Walks

#pragma region Refinement Passes

    /** Re-selects @p nodes, or the IDs @p node_ids lists with @p nodes spanning its offsets, into a
     *  shadow matrix on the device, then lands every walked row in the live matrix and counts the
     *  rows that changed. */
    expected<refine_result_t> refine_pass_(traversal_context_t& context, refine_config_t config,
                                           iota_view<node_id_t> nodes, std::span<u32_t const> node_ids) noexcept {
        std::size_t const rows = size();
        if (!rows || nodes.first >= nodes.bound) return {{}, status_t::success_k};
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        std::size_t const walk_count = nodes.size();
        std::size_t const connectivity = this->connectivity();
        std::size_t const shadow_bytes = rows * connectivity * sizeof(u32_t);
        std::size_t const ids_bytes = node_ids.size_bytes();
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, ids_bytes + shadow_bytes, fresh); failed(staged))
            return {{}, staged};
        if (ids_bytes) std::memcpy(context.staging_.host, node_ids.data(), ids_bytes);
        u32_t const pool_slots = pool_slots_for_(static_cast<u32_t>(non_zero_expansion_factor(config)), 4);
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(context, arguments, pool_slots * 32, cache_rows_k, walk_count);
            failed(sized))
            return {{}, sized};
        arguments.first_id = nodes.first;
        arguments.end_id = nodes.bound;
        arguments.sampling_bound = static_cast<u32_t>(rows);
        arguments.use_node_ids = node_ids.empty() ? 0 : 1;
        reset_counters_(counters_);

        status_t launched = status_t::device_error_k;
        {
            std::shared_lock storage {storage_};
            metal_launch_t launch {context.queue(), pipeline_(metal_kernel_t::refine_k, pool_slots)};
            if (launch) {
                bind_storage_(launch, arguments);
                launch.bind(context.staging_, 4);
                launch.bind(context.arena_, 5);
                launch.bind(counters_, 6);
                launch.bind(context.staging_, 7, ids_bytes);
                launched = context.submit_(launch, groups_for_(walk_count), simdgroups_per_group_k * 32);
                if (succeeded(launched)) launched = context.drain();
            }
        }
        if (failed(launched)) return {{}, launched};

        // Exclusive pass: the walked rows land in the live matrix with plain stores.
        refine_result_t result;
        u32_t* const live = reinterpret_cast<u32_t*>(neighbors_matrix_.host);
        u32_t const* const shadow = reinterpret_cast<u32_t const*>(context.staging_.host + ids_bytes);
        for (std::size_t offset = 0; offset != walk_count; ++offset) {
            std::size_t const node = node_ids.empty() ? nodes.first + offset : node_ids[offset];
            u32_t* const row = live + node * connectivity;
            u32_t const* const picked = shadow + node * connectivity;
            if (std::memcmp(row, picked, connectivity * sizeof(u32_t)) == 0) continue;
            std::memcpy(row, picked, connectivity * sizeof(u32_t));
            ++result.changed_nodes;
        }
        result.visited_nodes = walk_count;
        result.visited_members = counter_(counters_, metal_counter_t::hops_k);
        result.computed_distances = counter_(counters_, metal_counter_t::scored_k);
        result.restarts = counter_(counters_, metal_counter_t::restarts_k);
        return {result, status_t::success_k};
    }

    /** Sweeps forward reachability on the host from as many stride-spread landmarks as one search
     *  draws, collecting the rows it never reached into @c islands. */
    void audit_(audit_t& audit) const noexcept {
        std::size_t const rows = size(), connectivity = this->connectivity();
        std::size_t const seed_count = min_of(connectivity, rows);
        u32_t const* const neighbors = reinterpret_cast<u32_t const*>(neighbors_matrix_.host);
        std::memset(audit.reached.data(), 0, rows);
        std::memset(audit.frontier.data(), 0, rows);
        for (std::size_t seed = 0; seed != seed_count; ++seed) {
            std::size_t const row = seed * rows / seed_count;
            audit.reached[row] = audit.frontier[row] = 1;
        }
        constexpr std::size_t sweeps_cap_k = 256;
        for (audit.sweeps = 1; audit.sweeps != sweeps_cap_k; ++audit.sweeps) {
            std::memset(audit.next_frontier.data(), 0, rows);
            bool expanded = false;
            for (std::size_t row = 0; row != rows; ++row) {
                if (!audit.frontier[row]) continue;
                expanded = true;
                for (std::size_t slot = 0; slot != connectivity; ++slot) {
                    u32_t const target = neighbors[row * connectivity + slot];
                    if (target == node_id_unknown_k || target == row || target >= rows || audit.reached[target])
                        continue;
                    audit.reached[target] = audit.next_frontier[target] = 1;
                }
            }
            if (!expanded) break;
            std::swap(audit.frontier, audit.next_frontier);
        }
        audit.islands.clear();
        for (std::size_t row = 0; row != rows; ++row)
            if (!audit.reached[row]) audit.islands.push_back_reserved(static_cast<u32_t>(row));
    }

    /** Reconnects each island into the first of its reached picks with room for it, or into the
     *  landmark covering its ID: a free slot when one is left, otherwise a landing spread across
     *  the tail that never evicts a sibling island nor a reached resident left with no other
     *  reached in-link. Answers the links written. */
    std::size_t reconnect_(audit_t& audit) noexcept {
        std::size_t const rows = size(), connectivity = this->connectivity();
        std::size_t const seed_count = min_of(connectivity, rows);
        std::size_t const spread = heal_spread(connectivity);
        u32_t* const neighbors = reinterpret_cast<u32_t*>(neighbors_matrix_.host);
        growing_buffer<u32_t> in_links {global_allocator};
        if (failed(in_links.resize(rows))) return 0;
        for (std::size_t row = 0; row != rows; ++row) {
            if (!audit.reached[row]) continue;
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                u32_t const target = neighbors[row * connectivity + slot];
                if (target != node_id_unknown_k && target != row && target < rows) ++in_links[target];
            }
        }
        std::size_t reconnected = 0;
        for (std::size_t island_index = 0; island_index != audit.islands.size(); ++island_index) {
            u32_t const island = audit.islands[island_index];
            auto const land = [&](u32_t target) noexcept {
                u32_t* const target_row = neighbors + static_cast<std::size_t>(target) * connectivity;
                for (std::size_t slot = 0; slot != connectivity; ++slot) {
                    if (target_row[slot] == island) return true;
                    if (target_row[slot] != node_id_unknown_k && target_row[slot] != target) continue;
                    target_row[slot] = island, ++reconnected;
                    return true;
                }
                for (std::size_t attempt = 0; attempt != spread; ++attempt) {
                    std::size_t const landing = connectivity - 1 - (island_index + attempt) % spread;
                    u32_t const resident = target_row[landing];
                    if (resident == island) return true;
                    if (resident != node_id_unknown_k && (resident >= rows || !audit.reached[resident])) continue;
                    bool const evicting = resident != node_id_unknown_k && resident != target;
                    if (evicting && in_links[resident] <= 1) continue;
                    if (evicting) --in_links[resident];
                    target_row[landing] = island, ++reconnected;
                    return true;
                }
                return false;
            };
            bool landed = false;
            u32_t const* const island_row = neighbors + static_cast<std::size_t>(island) * connectivity;
            for (std::size_t slot = 0; slot != connectivity && !landed; ++slot) {
                u32_t const pick = island_row[slot];
                if (pick != node_id_unknown_k && pick != island && pick < rows && audit.reached[pick])
                    landed = land(pick);
            }
            if (landed || !seed_count) continue;
            std::size_t const seed = static_cast<std::size_t>(island) * seed_count / rows;
            u32_t const landmark = static_cast<u32_t>(seed * rows / seed_count);
            if (landmark != island) land(landmark);
        }
        return reconnected;
    }

#pragma endregion Refinement Passes

#pragma region State

    /** Keeps @p failure as the status unless one is kept already. */
    status_t fail_(status_t failure) noexcept {
        status_t expected = status_t::success_k;
        standard_atomic_ref<status_t>(status_).compare_exchange_strong(expected, failure, std::memory_order_acq_rel,
                                                                       std::memory_order_acquire);
        return failure;
    }

    std::size_t claimed_() const noexcept { return claimed_rows_; }
    u32_t published_() const noexcept { return counter_(counters_, metal_counter_t::published_k); }

    /** Turns the watermark from @p first over @p rows once every batch claimed before has turned
     *  it. */
    status_t publish_(std::size_t first, std::size_t rows) noexcept {
        for (std::size_t watermark = size(); watermark != first; watermark = size()) {
            if (status_t const kept = status(); failed(kept)) return kept;
            fu::standard_yield_t {}(&last_visible_id_, watermark, 0u, fu::wait_capped_t {});
        }
        standard_atomic_ref<std::size_t>(last_visible_id_).store(first + rows, std::memory_order_release);
        return status_t::success_k;
    }

    /** Grows every matrix and column to hold @p rows, doubling, the fresh neighbor rows
     *  sentinel-filled; the caller holds the batch hold, and the copy waits for every walk sharing
     *  the storage. */
    status_t grow_rows_(std::size_t rows) noexcept {
        if (rows <= capacity_rows_) return status_t::success_k;
        std::size_t const grown = max_of(rows, capacity_rows_ * 2);
        std::size_t const neighbors_row_bytes = connectivity() * sizeof(u32_t);
        metal_buffer_t vectors = vectors_allocator_.allocate(grown * row_bytes());
        metal_buffer_t norms = stores_norms() ? vectors_allocator_.allocate(grown * sizeof(norms_t))
                                              : metal_buffer_t {};
        metal_buffer_t neighbors = neighbors_allocator_.allocate(grown * neighbors_row_bytes);
        if (!vectors || !neighbors || (stores_norms() && !norms)) {
            vectors_allocator_.deallocate(vectors), vectors_allocator_.deallocate(norms);
            neighbors_allocator_.deallocate(neighbors);
            return status_t::bad_alloc_k;
        }
        std::unique_lock storage {storage_};
        if (capacity_rows_) {
            std::memcpy(vectors.host, vectors_matrix_.host, capacity_rows_ * row_bytes());
            if (stores_norms()) std::memcpy(norms.host, norms_column_.host, capacity_rows_ * sizeof(norms_t));
            std::memcpy(neighbors.host, neighbors_matrix_.host, capacity_rows_ * neighbors_row_bytes);
        }
        std::memset(neighbors.host + capacity_rows_ * neighbors_row_bytes, 0xFF,
                    (grown - capacity_rows_) * neighbors_row_bytes);
        vectors_allocator_.deallocate(vectors_matrix_), vectors_allocator_.deallocate(norms_column_);
        neighbors_allocator_.deallocate(neighbors_matrix_);
        vectors_matrix_ = std::move(vectors), norms_column_ = std::move(norms),
        neighbors_matrix_ = std::move(neighbors);
        capacity_rows_ = grown;
        return status_t::success_k;
    }

    metal_device_specs_t device_;
    nk_kernel_kind_t kernel_;
    std::size_t dimensions_ = 0, connectivity_ = 0, capacity_rows_ = 0;

    /** The vectors matrix and norms column backend. */
    vectors_allocator_t vectors_allocator_;

    /** The neighbors matrix backend. */
    neighbors_allocator_t neighbors_allocator_;

    /** The vectors matrix and norms column, each a shared buffer replaced wholesale on growth. */
    metal_buffer_t vectors_matrix_, norms_column_;

    /** The neighbors matrix, on the same terms. */
    metal_buffer_t neighbors_matrix_;

    /** The batches' counters: the assigned and published rows and the telemetry of the last insert
     *  or refine. */
    metal_buffer_t counters_;

    /** Every walking kernel's pipeline per pool rung, and the norms derivation's, owned by the
     *  device's registry. */
    void* pipelines_[3][4] = {};
    void* derive_norms_pipeline_ = nullptr;

    /** Shared by every walk while its command buffer runs, taken alone by growth. */
    mutable std::shared_mutex storage_;

    /** Held by the one batch, load or refinement at a time. */
    std::mutex batch_;

    std::size_t claimed_rows_ = 0;
    std::size_t last_visible_id_ = 0;
    status_t status_ = status_t::success_k;

#pragma endregion State
};

} // namespace unum::usearch::dense
