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
 *  waits on rows earlier batches publish, so the claim word admits one batch or load at a time and
 *  refuses another with @c contended_k; searches overlap them freely. The kernels compile from the
 *  embedded source at the first @c reserve, once per storage dtype and Apple class on each device.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::snprintf` - the source's prelude
#include <cstring> // `std::memcpy`, `std::memset`

#include <atomic>       // `std::memory_order`
#include <bit>          // `std::bit_ceil`, `std::bit_width`, `std::countr_zero`, `std::has_single_bit`
#include <mutex>        // `std::unique_lock` - the growth hold
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
#include "usearch/utilities/vram_device.hpp" // `vram_kernel_symbol_t` - a kernel's name
#include "usearch/utilities/vram_walks.hpp"  // `gpu_arena_shape`, `gpu_walk_visits`, `gpu_candidate_slots`
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_metric.hpp" // `import_row` - rows cross into the shared staging in the stored scalar
#include "usearch/utilities/vram_apple_shared.h" // `metal_counter_t`, `metal_extent_t` - read before the dense words
#include "usearch/dense/vram_apple_shared.h"     // `metal_arguments_t`, `metal_buffer_index_t`, `metal_metric_t`

namespace unum::usearch::dense {

static_assert(metal_max_gathers_in_flight_k == vram_max_gathers_in_flight_k, "The Metal round window is the CUDA one");

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

#if USEARCH_ARCH_METAL_

/** The vocabulary the dense index and its kernels share, spliced after the contexts'. */
inline constexpr char metal_shared_source_k[] = {
#embed "vram_apple_shared.h" suffix(, 0)
};

/** The MSL kits, embedded so the kernels compile on the device without a Metal toolchain. */
inline constexpr char metal_metric_source_k[] = {
#embed "vram_apple_metric.metal" suffix(, 0)
};

/** The MSL walk, insert, refine and search kernels, compiled after the kits. */
inline constexpr char metal_index_source_k[] = {
#embed "vram_apple_index.metal" suffix(, 0)
};
#else
// A unit without Metal compiles no pipeline, and NVCC's front end knows no `#embed`.
inline constexpr char metal_shared_source_k[] = {0};
inline constexpr char metal_metric_source_k[] = {0};
inline constexpr char metal_index_source_k[] = {0};
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** Bytes the prelude of one MSL source reaches, its terminator included. */
inline constexpr std::size_t metal_prelude_capacity_k = 1024;

/** The MSL source for @p dtype_ under @p capability_, assembled once per process: the shared
 *  vocabularies of the contexts and of the dense index, a prelude spelling NumKong's dtypes and
 *  capabilities from the host's own definitions, since MSL cannot include NumKong's headers, with
 *  the dtype and capability set this library compiles for, then the kits and the kernels. Its
 *  address keys the library built from it, so every dtype and Apple class compiles its own. */
template <nk_dtype_t dtype_, nk_capability_t capability_>
char const* metal_source() noexcept {
    static char source[sizeof(metal_context_source_k) + sizeof(metal_shared_source_k) + metal_prelude_capacity_k +
                       sizeof(metal_metric_source_k) + sizeof(metal_index_source_k)];
    static std::once_flag assembled;
    std::call_once(assembled, [] {
        std::size_t length = sizeof(metal_context_source_k) - 1;
        std::memcpy(source, metal_context_source_k, length);
        std::memcpy(source + length, metal_shared_source_k, sizeof(metal_shared_source_k) - 1);
        length += sizeof(metal_shared_source_k) - 1;
        length += static_cast<std::size_t>(std::snprintf( //
            source + length, metal_prelude_capacity_k,
            "\nenum nk_dtype_t : uint {\n"                                                                         //
            "    nk_bf16_k = %u, nk_f16_k = %u, nk_e5m2_k = %u, nk_e4m3_k = %u, nk_e3m2_k = %u, nk_e2m3_k = %u,\n" //
            "    nk_i8_k = %u, nk_i4_k = %u, nk_u8_k = %u, nk_u4_k = %u, nk_u1_k = %u,\n"                          //
            "};\n"                                                                                                 //
            "using nk_capability_t = ulong;\n"                                                                     //
            "constant constexpr nk_capability_t nk_cap_metal_k = %lluul, nk_cap_apple9_k = %lluul,\n"              //
            "                                   nk_cap_apple10_k = %lluul;\n"                                      //
            "namespace unum { namespace usearch { namespace dense {\n"                                             //
            "constant constexpr nk_dtype_t metal_dtype_k = nk_dtype_t(%u);\n"                                      //
            "constant constexpr nk_capability_t metal_capability_k = %lluul;\n"                                    //
            "} } }\n",                                                                                             //
            unsigned {nk_bf16_k}, unsigned {nk_f16_k}, unsigned {nk_e5m2_k}, unsigned {nk_e4m3_k}, unsigned {nk_e3m2_k},
            unsigned {nk_e2m3_k}, unsigned {nk_i8_k}, unsigned {nk_i4_k}, unsigned {nk_u8_k}, unsigned {nk_u4_k},
            unsigned {nk_u1_k}, static_cast<unsigned long long>(nk_cap_metal_k),
            static_cast<unsigned long long>(nk_cap_apple9_k), static_cast<unsigned long long>(nk_cap_apple10_k),
            unsigned {dtype_}, static_cast<unsigned long long>(capability_)));
        std::memcpy(source + length, metal_metric_source_k, sizeof(metal_metric_source_k) - 1);
        length += sizeof(metal_metric_source_k) - 1;
        std::memcpy(source + length, metal_index_source_k, sizeof(metal_index_source_k));
    });
    return source;
}

/** One exported kernel: the kernel and its register rung, zero where it keeps no ladder or walks
 *  the arena frontier. */
struct metal_rung_t {
    vram_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation @c vram_apple_index.metal exports, which the host builds a pipeline for at
 *  the first @c reserve. */
inline constexpr metal_rung_t metal_rungs[] = {
    {vram_kernel_t::insert_k, 0},  {vram_kernel_t::insert_k, 2},       {vram_kernel_t::insert_k, 4},
    {vram_kernel_t::insert_k, 8},  {vram_kernel_t::insert_k, 16},      {vram_kernel_t::back_link_k, 0},
    {vram_kernel_t::refine_k, 0},  {vram_kernel_t::refine_k, 2},       {vram_kernel_t::refine_k, 4},
    {vram_kernel_t::refine_k, 8},  {vram_kernel_t::refine_k, 16},      {vram_kernel_t::search_k, 0},
    {vram_kernel_t::search_k, 2},  {vram_kernel_t::search_k, 4},       {vram_kernel_t::search_k, 8},
    {vram_kernel_t::search_k, 16}, {vram_kernel_t::derive_norms_k, 0},
};

/** Spells the @c host_name @c vram_apple_index.metal exports @p rung under: the kernel's name, then
 *  the candidate slots of a walking kernel's instantiation. */
inline void metal_kernel_symbol(vram_kernel_symbol_t& symbol, metal_rung_t rung) noexcept {
    symbol = {};
    switch (rung.kernel) {
    case vram_kernel_t::insert_k: symbol.append("metal_insert_kernel_"); break;
    case vram_kernel_t::refine_k: symbol.append("metal_refine_kernel_"); break;
    case vram_kernel_t::search_k: symbol.append("metal_search_kernel_"); break;
    case vram_kernel_t::back_link_k: symbol.append("metal_back_link_kernel"); return;
    case vram_kernel_t::derive_norms_k: symbol.append("metal_derive_norms_kernel"); return;
    default: return;
    }
    symbol.append_value(rung.slots);
}

/** What a search launch lands: ranked distances, or every scored pair of a product. */
enum class metal_output_t : bool { distances_k, products_k };

/**
 *  @brief Vector Search index for the unified memory of one Apple GPU.
 *
 *  Speaks @c vram_nvidia_index 's verbs: a batch's IDs are one contiguous range turning visible a
 *  whole batch at a time, and the first successful @c reserve freezes the geometry. Attention is
 *  refused with @c unsupported_k until its value-lookup ABI is designed for Metal, and walks past
 *  the widest register rung take the arena frontier, as on NVIDIA and AMD.
 */
template <                                                          //
    nk_dtype_t vector_dtype_ = nk_i8_k,                             //
    nk_capability_t capability_ = nk_cap_metal_k,                   //
    typename vectors_allocator_type_ = metal_shared_allocator_t,    //
    typename neighbors_allocator_type_ = metal_shared_allocator_t,  //
    typename traversals_allocator_type_ = metal_shared_allocator_t> //
struct vram_apple_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

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

    /** The 32-bit ID space, less the sentinel, which also marks a closed claim word. */
    inline static constexpr std::size_t max_node_count_k = 0xFFFFFFFEu;

    /** Fragment bytes - the kits read rows sixteen bytes at a time, and rows pad to whole
     *  fragments. */
    inline static constexpr std::size_t fragment_bytes_k = metal_fragment_bytes_k;

    /** Rows one checkpoint batch carries between a matrix and the caller: one slab, the row
     *  granularity the RAM residency grows by. */
    inline static constexpr std::size_t batch_rows_log2_k = 20;

    /** The host capability rows import into the shared staging at: the variant's own names only the
     *  GPU, and every Apple silicon host has NEON. */
    inline static constexpr nk_capability_t import_capability_k = nk_cap_serial_k | nk_cap_neon_k;

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
    small_size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor
                                       : static_cast<small_size_t>(default_expansion_factor(connectivity()));
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

    /** Lanes per simdgroup, as every pipeline built for the device reports them; zero before the
     *  first @c reserve. */
    u32_t wave_lanes() const noexcept { return device_.wave_lanes; }

    /** Bytes of one builder simdgroup's slice over @p selection_rows cached candidates: one
     *  batch's scored nodes, IDs and dots, which CUDA keeps in shared memory, then the query span
     *  while walking, then the candidate cache, and per cached candidate its score, three ID
     *  spans, its distance and its norms. */
    u32_t build_scratch_bytes_per_warp(u32_t selection_rows) const noexcept {
        u32_t const candidate_bytes = static_cast<u32_t>(row_bytes() + gpu_scored_node_bytes<node_id_t> +
                                                         3 * sizeof(u32_t) + sizeof(f32_t) + sizeof(norms_t));
        return metal_batch_spans_bytes_k + max_of(static_cast<u32_t>(row_bytes()), selection_rows * candidate_bytes);
    }

    /** Which frontier a simdgroup's arena slice serves: the register ladder needs only the exact
     *  visited set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** One simdgroup's arena slice for @p walks walks sharing one visited set over
     *  @p candidate_rows candidates each: the exact visited set over every ID they claim, plus the
     *  spill tail when @p frontier_kind outgrows registers, what their expansions stage. */
    gpu_arena_shape_t arena_shape_for(u32_t candidate_rows, frontier_kind_t frontier_kind,
                                      u32_t walks = 1) const noexcept {
        u32_t const connectivity = static_cast<u32_t>(this->connectivity()), lanes = wave_lanes();
        u32_t const visits = walks * gpu_walk_visits(candidate_rows, connectivity, lanes);
        u32_t const tail = frontier_kind == frontier_kind_t::wide_arena_k
                               ? walks * gpu_walk_tail_entries(candidate_rows, connectivity, lanes)
                               : 0;
        return gpu_arena_shape(visits, tail, 0);
    }

    /** Whether a search or multiply launch fits on this device: every simdgroup the widest
     *  register rung's grid keeps in flight takes that rung's slice and the @p fold_bytes an
     *  attend keeps past it, within one buffer; a wider walk sizes its arena tail per call. */
    bool walk_scratch_fits(std::size_t fold_bytes = 0) const noexcept {
        std::size_t const simdgroups = std::size_t {device_.multiprocessor_count} *
                                       resident_blocks_(vram_kernel_t::search_k, vram_widest_candidate_slots_k) *
                                       metal_simdgroups_k;
        return simdgroups * (walk_arena_shape_(0, vram_widest_candidate_slots_k).template slice_bytes<node_id_t>() +
                             fold_bytes) <=
               device_.max_buffer_bytes;
    }

    /** Whether an insert or refine launch fits on this device: every simdgroup the widest
     *  register rung's grid keeps in flight takes that rung's slice beside its builder scratch,
     *  within one buffer, since threadgroup memory would bound the row width. */
    bool build_scratch_fits() const noexcept {
        std::size_t const simdgroups = std::size_t {device_.multiprocessor_count} *
                                       resident_blocks_(vram_kernel_t::insert_k, vram_widest_candidate_slots_k) *
                                       metal_simdgroups_k;
        return simdgroups * build_arena_shape_(0, vram_widest_candidate_slots_k, 1).template slice_bytes<node_id_t>() <=
               device_.max_buffer_bytes;
    }

    /** Bytes this index occupies: its storage and its counters. */
    std::size_t memory_usage() const noexcept {
        return sizeof(vram_apple_index) + vectors_matrix_.bytes + norms_column_.bytes + neighbors_matrix_.bytes +
               counters_.bytes;
    }

    /** Copies the stored row of @p node into @p destination with the fragment padding stripped. */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination) noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        std::shared_lock storage {storage_};
        std::memcpy(destination.byte_data(), vectors_matrix_.host + node * row_bytes(), packed_row_bytes());
        return status_t::success_k;
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

        /** Rows the index may ever host, clamped to the ID space; frozen with the geometry, past it
         *  every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;
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
            metal_context_t* const context = metal_context(device_.ordinal);
            if (!context) return {0, status_t::device_error_k};
            // The pipelines report the simdgroup width the specification carries, so they build
            // first.
            if (status_t const loaded = load_pipelines_(); failed(loaded)) return {0, loaded};
            auto const [specs, status] = metal_device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            counters_ = metal_allocate(*context, counters_bytes_k);
            if (!counters_) return {0, status_t::bad_alloc_k};
            dimensions_ = config.dimensions;
            connectivity_ = static_cast<std::size_t>(config.connectivity);
            ceiling_rows_ = min_of(config.ceiling, max_node_count_k);
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
     *  slice per simdgroup in flight, and the staged rows beside the row an import widens through
     *  and the back-link chains; zero before @c reserve freezes the geometry. */
    std::size_t traversal_memory_usage(std::size_t rows, insert_config_t config = {}) const noexcept {
        if (!dimensions_ || !device_.multiprocessor_count) return 0;
        u32_t const expansion = static_cast<u32_t>(vram_insert_expansion(config.expansion_factor));
        u32_t const candidate_slots = candidate_slots_for_(expansion, 0);
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, 1);
        std::size_t const arena_warps = std::size_t {device_.multiprocessor_count} *
                                        resident_blocks_(vram_kernel_t::insert_k, candidate_slots) * metal_simdgroups_k;
        return counters_bytes_k + arena_warps * arena_shape.template slice_bytes<node_id_t>() + rows * row_bytes() +
               widened_bytes_() + links_bytes_(rows, size() + rows, 1);
    }

    /** Telemetry of one @c insert() batch over the contiguous IDs @b [first_id,first_id+count). */
    struct [[nodiscard]] insert_result_t {
        std::size_t new_size = 0;
        node_id_t first_id = node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t unlinked_neighbors = 0;
        std::size_t tail_compactions = 0;
        std::size_t tail_dropped = 0;
        std::size_t visited_saturations = 0;
        std::size_t computed_distances_in_refines = 0;
        std::size_t computed_distances_in_reverse_refines = 0;
    };

    /** Inserts @p new_rows under the contiguous IDs claimed from the watermark, waits for the
     *  device, publishes them, and reports telemetry. One batch or load holds the claim word at a
     *  time, so another one in flight, or a closed word, answers @c contended_k. */
    expected<insert_result_t> insert(rows_view_t new_rows, node_ids_span_t assigned_ids, traversal_context_t& context,
                                     insert_config_t config = {}) noexcept {
        if (!new_rows.rows) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_t::success_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        auto const [first_id, claim_status] = claim_(new_rows.rows);
        if (failed(claim_status)) return {{}, claim_status};
        expected<insert_result_t> inserted = submit_insert_(new_rows, first_id, assigned_ids, context, config);
        if (!inserted) {
            // A refusal that kept no failure launched nothing.
            if (succeeded(status())) unclaim_(first_id, new_rows.rows);
            return inserted;
        }
        if (status_t const published = publish_(inserted.value.first_id, new_rows.rows); failed(published))
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
        traversal_t traversal = traversal_t::walk_k;
    };

    /** Telemetry of one @c search() batch. */
    struct [[nodiscard]] search_result_t {
        std::size_t count = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t tail_compactions = 0;
        std::size_t tail_dropped = 0;
        std::size_t visited_saturations = 0;
    };

    /** The bytes a context grows to before it can search or multiply @p rows rows under @p config,
     *  landing @c config.count columns per row: its counters, one arena slice per simdgroup in
     *  flight, and the staged rows beside their outputs. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        if (!dimensions_ || !device_.multiprocessor_count) return 0;
        // Search widens the candidates to cover its count while multiply walks the bare factor, so
        // the wider one sizes.
        u32_t const expansion = max_of<u32_t>(gpu_walk_breadth(config.expansion_factor, config.count),
                                              non_zero_expansion_factor(config));
        u32_t const count_slots = divide_round_up<u32_t>(config.count, wave_lanes());
        u32_t const candidate_slots = candidate_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        std::size_t const arena_warps = std::size_t {walk_blocks_(
                                            rows, resident_blocks_(vram_kernel_t::search_k, candidate_slots))} *
                                        metal_simdgroups_k;
        return counters_bytes_k +
               arena_warps * walk_arena_shape_(expansion, candidate_slots).template slice_bytes<node_id_t>() +
               rows * (row_bytes() + config.count * (sizeof(node_id_t) + sizeof(distance_t)) + sizeof(u32_t)) +
               widened_bytes_();
    }

    /** The bytes a search under the indexer @p heads needs: the plain walk's, plus the staging of
     *  the heads and their weights. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config,
                                       indexer_heads<scalar_t> const& heads) const noexcept {
        return traversal_memory_usage(rows, config) + heads.heads.extent(0) * (row_bytes() + sizeof(f32_t));
    }

    /** The nearest published neighbors of every query row, ascending, with sentinel IDs past the
     *  live matches. */
    expected<search_result_t> search(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {}, completion_t = completion_t::blocking_k) noexcept {
        return search_(queries, result_ids, result_distances, context, config, size(), dummy_heads_t {});
    }

    /** The batched @c search ranked by a lightning indexer; a dot verb, refusing every other
     *  metric. */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
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

    /** Filtered search answers @c device_error_k: a host predicate cannot run inside the walk, and
     *  a device-resident admission bitmap is a design of its own. */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t, predicate_type_&&, traversal_context_t&,
                                        search_config_t = {}) noexcept {
        return {{}, status_t::device_error_k};
    }

    /** Graph products of every left-hand row: each keeps the pairs its walk scored, until its
     *  output row fills. */
    expected<search_result_t> multiply(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       traversal_context_t& context, search_config_t config = {},
                                       completion_t = completion_t::blocking_k) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return multiply_(left, node_ids, products, counts, context, config, size());
    }

    /** Attention is refused with @c unsupported_k: a Metal kernel cannot call the caller's value
     *  lookup, just as it cannot run a predicate. */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t, std::size_t, value_lookup_type_&&, f32_t, rows_span_t,
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
     *  @brief One refinement in flight on this index's device, in the shape of the CUDA and HIP
     *      plans so one host thread drives every shard: @ref queue a phase, @ref wait for it,
     *      @ref advance past it, and @ref finish once no phase follows.
     *
     *  A re-selecting phase walks every node of the planned range, once per round of walks across
     *  while the shard holds several parts, into a shadow matrix that replaces the walked rows once
     *  the device finishes; unlike the CUDA pass no back-link merge follows, since the heal's audit
     *  and reconnection, run on the host over unified memory in @ref advance, restore whatever a
     *  re-selection strands. @ref plan_refine sized every buffer, so no phase allocates. Every
     *  phase takes the context the plan was opened on back from its caller. Holds the index's batch
     *  lock from plan to death, so demands an exclusive caller.
     */
    struct refine_plan_t {
        refine_plan_t() noexcept = default;
        refine_plan_t(refine_plan_t&&) noexcept = default;
        refine_plan_t& operator=(refine_plan_t&&) noexcept = default;

        /** Submits the current phase's pass on @p context 's stream under @p config; the audit, on
         *  the host, waits for @ref advance. */
        status_t queue(traversal_context_t& context, refine_config_t config) noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (failed(status_)) return status_;
            if (phase_ == phase_t::reselect_k)
                status_ = index_->submit_pass_(context, config, walked_(), {}, parts_, round_);
            else if (phase_ == phase_t::heal_k)
                status_ = index_->submit_pass_(context, config, walked_(), to_span(audit_.islands));
            return status_;
        }

        /** Waits on @p context for the submitted pass, if one was. */
        status_t wait(traversal_context_t& context) noexcept {
            if (index_ && succeeded(status_) && !walked_().empty()) status_ = context.wait();
            return status_;
        }

        /** Lands the pass from @p context on the host, audits after the rounds and every heal,
         *  picks the next phase, and answers whether one follows. */
        bool advance(traversal_context_t& context) noexcept {
            if (!index_ || failed(status_) || phase_ == phase_t::done_k) return false;
            metal_audit_t& audit = audit_;
            if (iota_view<node_id_t> const walked = walked_(); !walked.empty())
                tally_ += index_->land_pass_(
                    context, walked, phase_ == phase_t::heal_k ? to_span(audit.islands) : std::span<u32_t const> {});
            if (phase_ == phase_t::reselect_k) {
                if (++round_ < rounds_) return true;
                phase_ = phase_t::audit_k;
                return true;
            }
            std::size_t const islands_before = audit.islands.size();
            if (phase_ == phase_t::heal_k) tally_.islands_reconnected += index_->reconnect_(audit);
            status_t const audited = index_->audit_(audit);
            std::size_t const rows = index_->size(), left = audit.islands.size();
            tally_.health = {rows, rows - left, audit.sweeps};
            // A refused audit judged no row, so nothing heals.
            if (failed(audited)) {
                status_ = audited;
                phase_ = phase_t::done_k;
                return false;
            }
            if (phase_ == phase_t::audit_k) tally_.islands_found = left;
            // A round that frees no island frees none on the next either, since the picks it needs
            // are unreached.
            bool const healing = left && (phase_ == phase_t::audit_k || left < islands_before);
            phase_ = healing ? phase_t::heal_k : phase_t::done_k;
            return healing;
        }

        /** Stops the re-selection after the round that landed last, so the next phase audits and
         *  heals; a no-op in any other phase. */
        void stop() noexcept {
            if (phase_ == phase_t::reselect_k) rounds_ = 0, phase_ = phase_t::audit_k;
        }

        /** Whether the current phase re-selects the planned range - the phase a caller's
         *  progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Nodes re-selected so far. */
        std::size_t visited() const noexcept { return tally_.visited_nodes; }

        /** The pass, complete once @ref advance answered false, otherwise @c cancelled_k; a failure
         *  a phase kept answers instead. */
        expected<refine_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (failed(status_)) return {tally_, status_};
            return {tally_, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend struct vram_apple_index;
        enum class phase_t : u8_t { reselect_k, audit_k, heal_k, done_k };

        /** The rows the current phase's pass walks: the round's nodes while re-selecting, every
         *  island while healing, and none while auditing. */
        iota_view<node_id_t> walked_() const noexcept {
            if (phase_ == phase_t::reselect_k) return round_ ? iota_view<node_id_t> {0, nodes_.bound} : nodes_;
            if (phase_ == phase_t::heal_k) return {0, static_cast<node_id_t>(audit_.islands.size())};
            return {};
        }

        std::unique_lock<std::mutex> batch_;
        vram_apple_index* index_ = nullptr;

        /** The audit whose reached rows and islands the heal walks and reconnects, from one
         *  @ref advance to the next. */
        metal_audit_t audit_ {};

        /** The IDs the first round re-selects. */
        iota_view<node_id_t> nodes_;

        /** Parts the rounds merge, the rounds, and the round queued or, once they
         *  ended, finished. */
        u32_t parts_ = 0;
        u32_t rounds_ = 1;
        u32_t round_ = 0;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_t::success_k;
        refine_result_t tally_;
    };

    /**
     *  @brief Opens a refinement of the nodes in @p nodes, clamped to the published IDs: takes the
     *      batch lock, sizes the staging, the arena and the host audit for the widest phase, and
     *      copies @p part_bases into the staging, so no phase allocates. An empty range heals.
     *
     *  Two or more @p part_bases below the published IDs split the rows into parts, as a shard
     *  that absorbed others' rows holds them: each node then first walks across the neighboring
     *  group of parts alone, round after round, from empty candidates and landmarks drawn there,
     *  before its own neighbors and the whole graph, and re-selects from the union. The live graph
     *  stays as it was through a round's walks, so a walk across never leaves its group, and the
     *  heal's passes walk no part first.
     */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                        refine_config_t config = {},
                                        std::span<node_id_t const> part_bases = {}) noexcept {
        refine_plan_t plan;
        plan.batch_ = std::unique_lock {batch_};
        std::size_t const rows = size();
        std::size_t parts = 0;
        while (parts != part_bases.size() && part_bases[parts] < rows) ++parts;
        if (parts < 2) parts = 0;
        if (rows) {
            if (!dimensions_) return {std::move(plan), status_t::uninitialized_k};
            if (context.device() != device_.ordinal || !context) return {std::move(plan), status_t::config_mismatch_k};
            bool fresh = false;
            if (status_t const staged = context.ensure_(context.staging_, staging_bytes_(parts), fresh); failed(staged))
                return {std::move(plan), staged};
            std::memcpy(context.staging_.host + bases_offset_(), part_bases.data(), parts * sizeof(u32_t));
            // The widest walk's arena now, so no pass grows it.
            u32_t const expansion = non_zero_expansion_factor(config);
            u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
            u32_t const walks = parts ? 2 : 1;
            metal_arguments_t arguments = arguments_();
            if (status_t const sized = shape_arena_(
                    context, arguments, build_arena_shape_(expansion, candidate_slots, walks),
                    device_.multiprocessor_count * resident_blocks_(vram_kernel_t::refine_k, candidate_slots), rows);
                failed(sized))
                return {std::move(plan), sized};
            metal_audit_t& audit = plan.audit_;
            for (growing_buffer<u8_t>* marks : {&audit.reached, &audit.frontier, &audit.next_frontier})
                if (failed(marks->reserve_exactly(rows))) return {std::move(plan), status_t::bad_alloc_k};
            if (failed(audit.islands.reserve_exactly(rows)) || failed(audit.in_links.reserve_exactly(rows)))
                return {std::move(plan), status_t::bad_alloc_k};
        }
        node_id_t const bound = static_cast<node_id_t>(min_of<std::size_t>(nodes.bound, rows));
        plan.index_ = this;
        plan.nodes_ = {min_of(nodes.first, bound), bound};
        plan.parts_ = static_cast<u32_t>(parts);
        plan.rounds_ = parts ? static_cast<u32_t>(std::bit_width(parts - 1)) : 1u;
        plan.phase_ = plan.nodes_.empty() && plan.rounds_ == 1 ? refine_plan_t::phase_t::audit_k
                                                               : refine_plan_t::phase_t::reselect_k;
        if (!rows) plan.phase_ = refine_plan_t::phase_t::done_k;
        return {std::move(plan), status_t::success_k};
    }

    /** One @ref refine_plan_t driven to its end on the calling thread, waiting once per phase. */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                     refine_config_t config = {}, std::span<node_id_t const> part_bases = {}) noexcept {
        auto [plan, status] = plan_refine(nodes, context, config, part_bases);
        if (failed(status)) return {{}, status};
        do
            if (status_t const landed = plan.queue(context, config); failed(landed) || failed(plan.wait(context)))
                break;
        while (plan.advance(context));
        return plan.finish();
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws, on
     *  the host, sweeping until no row expands. */
    expected<health_result_t> health(traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_t::success_k};
        metal_audit_t& audit = context.audit_;
        std::shared_lock storage {storage_};
        if (status_t const audited = audit_(audit); failed(audited)) return {{}, audited};
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
        std::size_t covered = 0;

        save_plan_t() noexcept = default;
        save_plan_t(vram_apple_index& index, columns_t columns, node_id_t id_shift, node_id_t visible) noexcept
            : columns(columns), id_shift(id_shift), visible(visible), index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), id_shift(other.id_shift), visible(other.visible), covered(other.covered),
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
            std::size_t const first = position * batch_rows();
            std::size_t const bound = min_of(first + batch_rows(), static_cast<std::size_t>(visible));
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
            return step;
        }
        indexed_range<save_plan_t> steps() const noexcept {
            return {this, index_ && visible ? divide_round_up<std::size_t>(visible, batch_rows()) : 0};
        }
        std::size_t batch_rows() const noexcept {
            return min_of(static_cast<std::size_t>(visible), power_of_two<std::size_t>(batch_rows_log2_k));
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
            node_id_t const* const source = index_->neighbor_rows_() + rows.first * connectivity;
            node_id_t* const landed = reinterpret_cast<node_id_t*>(neighbors_rows.data());
            for (std::size_t slot = 0; slot != rows.size() * connectivity; ++slot)
                landed[slot] = source[slot] < visible ? static_cast<node_id_t>(source[slot] + id_shift)
                                                      : node_id_unknown_k;
            standard_atomic_ref<std::size_t> {covered}.fetch_add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            index_ = nullptr;
            return status_t::success_k;
        }

      private:
        vram_apple_index* index_ = nullptr;
    };

    /** Opens a save into @p columns; @c concurrent_k refuses, since the storage cannot be probed
     *  for settlement. */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    node_id_t id_shift = 0) noexcept {
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
                  nk::matrix_span<node_id_t> scratch = {}) noexcept {
        auto [plan, status] = plan_save(columns, synchronization);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /** One load in flight: the rows it appends at the watermark, under its own claim or a claim
     *  word its caller closed, holding the growth hold for its whole life, since growth would
     *  relocate the storage its steps write; @ref finish derives the norms no step carried and
     *  publishes. */
    struct [[nodiscard]] load_plan_t {
        node_id_t base = 0;
        std::size_t rows = 0;
        std::size_t covered = 0;
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;
        load_plan_t(vram_apple_index& index, std::unique_lock<std::mutex>&& batch, node_id_t base,
                    std::size_t rows) noexcept
            : base(base), rows(rows), index_(&index), batch_(std::move(batch)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), rows(std::exchange(other.rows, 0)), covered(other.covered),
              covered_norms(other.covered_norms), index_(std::exchange(other.index_, nullptr)),
              batch_(std::move(other.batch_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan turns its rows back to sentinels under the hold, since the next claim
         *  lands on them without growing, and gives its claim back. */
        ~load_plan_t() noexcept {
            if (!index_ || !rows) return;
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(u32_t);
            std::memset(index_->neighbors_matrix_.host + base * neighbors_row_bytes, 0xFF, rows * neighbors_row_bytes);
            index_->unclaim_(base, rows);
        }

        struct step_t {
            std::size_t position = 0;
            iota_view<node_id_t> index_rows;
            iota_view<node_id_t> source_rows;
        };

        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = static_cast<std::size_t>(base) + position * batch_rows();
            std::size_t const bound = min_of(first + batch_rows(), static_cast<std::size_t>(base) + rows);
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            return step;
        }
        indexed_range<load_plan_t> steps() const noexcept {
            return {this, index_ && rows ? divide_round_up(rows, batch_rows()) : 0};
        }
        std::size_t batch_rows() const noexcept { return min_of(rows, power_of_two<std::size_t>(batch_rows_log2_k)); }

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
            std::span<node_id_t> const landed {index_->neighbor_rows_() + range.first * connectivity,
                                               range.size() * connectivity};
            std::memcpy(landed.data(), neighbors_rows.data(), neighbors_rows.size());
            rebase_neighbors(landed, source_range, shift, node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, node_id_unknown_k);
            standard_atomic_ref<std::size_t> {covered}.fetch_add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Copies the step's rows out of @p source 's own storage rather than columns, as an absorb
         *  drives it: both keep padded rows of one geometry in shared buffers, so every column
         *  moves whole on the host, and the neighbors rebase from @p source_range onto the appended
         *  range and compact. The source takes no rows until the copy returns. */
        status_t load(step_t const& step, vram_apple_index& source, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty() || range.size() != step.source_rows.size() || step.source_rows.bound > source.size() ||
                !same_geometry(source, *index_))
                return status_t::config_mismatch_k;
            node_id_t const shift = range.first - step.source_rows.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes(), connectivity = index_->connectivity();
            std::size_t const into = range.first, from = step.source_rows.first, count = range.size();
            std::shared_lock storage {source.storage_};
            std::memcpy(index_->vectors_matrix_.host + into * row_bytes, source.vectors_matrix_.host + from * row_bytes,
                        count * row_bytes);
            if (index_->stores_norms()) {
                std::memcpy(index_->norms_column_.host + into * sizeof(norms_t),
                            source.norms_column_.host + from * sizeof(norms_t), count * sizeof(norms_t));
                standard_atomic_ref<std::size_t> {covered_norms}.fetch_add(count, std::memory_order_acq_rel);
            }
            std::span<node_id_t> const landed {index_->neighbor_rows_() + into * connectivity, count * connectivity};
            std::memcpy(landed.data(), source.neighbor_rows_() + from * connectivity, landed.size_bytes());
            rebase_neighbors(landed, source_range, shift, node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, node_id_unknown_k);
            standard_atomic_ref<std::size_t> {covered}.fetch_add(count, std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Derives the norms no step carried on the device, lifts the hold, and publishes the
         *  appended rows. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                if (index_->stores_norms() &&
                    standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows)
                    if (status_t const derived = index_->derive_norms_(base, rows); failed(derived))
                        return index_->fail_(derived);
                u32_t* const words = metal_words(index_->counters_);
                words[metal_published_k] = static_cast<u32_t>(base + rows);
                batch_.unlock();
            }
            vram_apple_index* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_t::success_k;
        }

      private:
        vram_apple_index* index_ = nullptr;
        std::unique_lock<std::mutex> batch_;
    };

    /** Opens a load of @p rows at the watermark, sourced per step, as the sharded index drives its
     *  parts: claims them, or lands them under the claim word its caller closed; a batch or load in
     *  flight refuses it. */
    expected<load_plan_t> plan_load(std::size_t rows) noexcept {
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (!rows) return {load_plan_t {*this, {}, static_cast<node_id_t>(claimed_()), 0}, status_t::success_k};
        std::unique_lock batch {batch_, std::try_to_lock};
        if (!batch.owns_lock()) return {{}, status_t::contended_k};
        // A closed word is the caller's, who refines or absorbs around the load; an open one is
        // claimed like a batch's, so no batch lands beside it.
        bool const claiming = !claims_.closed();
        node_id_t base = static_cast<node_id_t>(size());
        if (claiming) {
            auto const [first_id, claim_status] = claim_(rows);
            if (failed(claim_status)) return {{}, claim_status};
            base = first_id;
        }
        else if (rows > ceiling_rows_ - base) return {{}, status_t::overflow_k};
        if (status_t const grown = grow_rows_(base + rows); failed(grown)) {
            unclaim_(base, rows);
            return {{}, grown};
        }
        return {load_plan_t {*this, std::move(batch), base, rows}, status_t::success_k};
    }

    /** Appends the whole source on the calling thread; columns shorter than @p rows are refused
     *  before a row is claimed. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range) noexcept {
        if (!dimensions_) return status_t::uninitialized_k;
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return status_t::config_mismatch_k;
        auto [plan, status] = plan_load(rows);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step, columns, source_range); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

    /** The sharded twin publishes and fails its shard's batches, and walks under its own
     *  snapshot. */
    template <typename, nk_dtype_t, nk_capability_t, typename, typename>
    friend struct vram_apple_indexes;
    template <typename, typename, typename, typename, template <typename> class, typename>
    friend struct utilities::shard_set;

  private:

#pragma region Batches

    /** The device half of @c insert: stages and runs @p new_rows at @p first_id, and returns with
     *  their IDs, the rows invisible until the caller publishes them - which the sharded twin does
     *  for every step at once. The caller holds the claim over them; a refusal before the launch
     *  keeps no failure, one after it keeps the failure. */
    expected<insert_result_t> submit_insert_(rows_view_t new_rows, node_id_t first_id, node_ids_span_t assigned_ids,
                                             traversal_context_t& context, insert_config_t config) noexcept {
        std::size_t const rows = new_rows.rows;
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const refusal = rows_refusal_(new_rows); failed(refusal)) return {{}, refusal};

        // Growth relocates the storage, so it runs under the hold a reserve takes too.
        {
            std::unique_lock batch {batch_};
            if (status_t const grown = grow_rows_(first_id + rows); failed(grown)) return {{}, grown};
        }
        // Past the staged rows, every wave's insert and back-link count into their own words,
        // zeroed here, then the row an import widens through, then the chains, every head unknown.
        std::size_t const rows_bytes = rows * row_bytes();
        std::size_t const waves = insert_waves_(first_id, first_id + rows);
        std::size_t const waves_bytes = waves * 2 * counters_bytes_k;
        std::size_t const links_offset = rows_bytes + waves_bytes + widened_bytes_();
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_,
                                                    links_offset + links_bytes_(rows, first_id + rows, waves), fresh);
            failed(staged))
            return {{}, staged};
        stage_rows_(context.staging_.host, new_rows, context.staging_.host + rows_bytes + waves_bytes);
        std::memset(context.staging_.host + rows_bytes, 0, waves_bytes);
        std::memset(context.staging_.host + links_offset, 0xFF, (first_id + rows) * sizeof(u32_t));
        std::memset(context.staging_.host + links_offset + links_bytes_(rows, first_id + rows, 0), 0,
                    waves * sizeof(u32_t));
        // Every inserting simdgroup owns an arena slice for its exact visited set and its scratch.
        u32_t const expansion = static_cast<u32_t>(vram_insert_expansion(config.expansion_factor));
        u32_t const candidate_slots = candidate_slots_for_(expansion, 0);
        u32_t const blocks = device_.multiprocessor_count * resident_blocks_(vram_kernel_t::insert_k, candidate_slots);
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(context, arguments, build_arena_shape_(expansion, candidate_slots, 1),
                                                blocks, rows);
            failed(sized))
            return {{}, sized};
        arguments.first_id = static_cast<u32_t>(first_id);
        arguments.end_id = static_cast<u32_t>(first_id + rows);
        arguments.selection_rows = walk_selection_rows_(expansion, candidate_slots);
        arguments.expansion = expansion;
        reset_counters_(counters_);

        // The claim admits one batch at a time, so the batch drains before the next claims.
        status_t launched = status_t::success_k;
        {
            std::shared_lock storage {storage_};
            launched = launch_insert_(arguments, candidate_slots, blocks, links_offset, context);
            if (succeeded(launched)) launched = context.wait();
        }
        if (failed(launched)) return {{first_id, static_cast<node_id_t>(first_id)}, fail_(launched)};
        // Every launch's published rows and tallies fold into the batch's counters; their assigned
        // words are spent.
        u32_t* const counters = metal_words(counters_);
        for (std::size_t offset = rows_bytes; offset != rows_bytes + waves_bytes; offset += counters_bytes_k) {
            u32_t const* const wave = reinterpret_cast<u32_t const*>(context.staging_.host + offset);
            counters[metal_published_k] += wave[metal_published_k];
            for (u32_t word = metal_hops_k; word + 1 < metal_counter_words_k; word += 2) {
                u64_t const total = metal_tally(counters, word) + metal_tally(wave, word);
                counters[word] = static_cast<u32_t>(total), counters[word + 1] = static_cast<u32_t>(total >> 32);
            }
        }

        insert_result_t result;
        result.first_id = first_id;
        result.new_size = first_id + rows;
        result.visited_members = metal_tally(counters, metal_hops_k);
        result.computed_distances = metal_tally(counters, metal_scored_k);
        result.restarts = metal_tally(counters, metal_restarts_k);
        result.unlinked_neighbors = metal_tally(counters, metal_lock_skips_k);
        result.visited_saturations = metal_tally(counters, metal_saturations_k);
        result.tail_compactions = metal_tally(counters, metal_tail_compactions_k);
        result.tail_dropped = metal_tally(counters, metal_tail_dropped_k);
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<node_id_t>(first_id + row);
        return {result, status_t::success_k};
    }

#pragma endregion Batches

#pragma region Launches

    /** The scalars every launch shares. */
    metal_arguments_t arguments_() const noexcept {
        u32_t const fragments = row_fragments();
        u32_t const lanes_per_row = min_of(std::bit_ceil(fragments), wave_lanes());
        metal_arguments_t arguments {};
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
    metal_metric_t metric_() const noexcept {
        switch (kernel_) {
        case nk_kernel_dot_k: return metal_dot_k;
        case nk_kernel_angular_k: return metal_angular_k;
        case nk_kernel_sqeuclidean_k: return metal_sqeuclidean_k;
        case nk_kernel_euclidean_k: return metal_euclidean_k;
        case nk_kernel_hamming_k: return metal_hamming_k;
        default: return metal_jaccard_k;
        }
    }

    /** Threadgroups of @p kernel on its @p slots rung resident per GPU core: as many as one
     *  threadgroup of its pipeline holds threads, its @c maxTotalThreadsPerThreadgroup, which its
     *  register use bounds, since a threadgroup runs on one core and Metal reports no occupancy;
     *  every grid and the arena behind it are sized by it, as CUDA's launch bounds size its. Zero
     *  before the first @c reserve. */
    u32_t resident_blocks_(vram_kernel_t kernel, u32_t slots) const noexcept {
        u32_t const threads = threadgroup_threads_();
        return threads ? pipeline_(kernel, slots).max_threads / threads : 0;
    }

    /** Blocks a walk over @p queries launches at @p resident_blocks threadgroups per GPU core: a
     *  simdgroup per query up to the grid the device runs at once, so the arena carves a slice only
     *  for a simdgroup that can take a query. */
    u32_t walk_blocks_(std::size_t queries, u32_t resident_blocks) const noexcept {
        return static_cast<u32_t>(min_of<std::size_t>(divide_round_up<std::size_t>(queries, metal_simdgroups_k),
                                                      std::size_t {device_.multiprocessor_count} * resident_blocks));
    }

    /** Sizes @p context 's arena for @p blocks threadgroups of slices shaped @p shape, taking
     *  epochs for @p work walks and zeroing it when fresh or when the epochs wrap, and fills the
     *  slice shape and the epoch base into @p arguments. */
    status_t shape_arena_(traversal_context_t& context, metal_arguments_t& arguments, gpu_arena_shape_t shape,
                          std::size_t blocks, std::size_t work) noexcept {
        std::size_t const slices = blocks * metal_simdgroups_k;
        bool fresh = false;
        if (status_t const sized = context.ensure_(context.arena_, slices * shape.template slice_bytes<node_id_t>(),
                                                   fresh);
            failed(sized))
            return sized;
        if (fresh || context.epoch_floor_ + work > 0xFFFFFFFEull) {
            if (!fresh) std::memset(context.arena_.host, 0, context.arena_.bytes);
            context.epoch_floor_ = 0;
        }
        arguments.visited_slots = shape.visited_slots;
        arguments.hash_shift = shape.visited_hash_shift;
        arguments.tail_entries = shape.tail_entries;
        arguments.slice_bytes = static_cast<u32_t>(shape.template slice_bytes<node_id_t>());
        arguments.epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += work;
        return status_t::success_k;
    }

    /** Bytes of the f32 row an import widens f64 rows through before narrowing them at unit RMS. */
    std::size_t widened_bytes_() const noexcept { return dimensions() * sizeof(f32_t); }

    /** Bytes of the back-link chains of @p rows rows that leave @p total_rows behind, in @p waves:
     *  a head per row, a next link per pick, the touched list as long, and each wave's count. */
    std::size_t links_bytes_(std::size_t rows, std::size_t total_rows, std::size_t waves) const noexcept {
        return (total_rows + 2 * rows * connectivity_ + waves) * sizeof(u32_t);
    }

    /** Why @p rows cannot cross into the staging: a width other than the index's, or rows closer
     *  than one packed row apart, a zero stride repeating the first. */
    status_t rows_refusal_(rows_view_t rows) const noexcept {
        if (rows.columns != dimensions()) return status_t::config_mismatch_k;
        if (rows.stride && rows.stride < packed_bytes(rows.dtype, rows.columns)) return status_t::unsupported_layout_k;
        return status_t::success_k;
    }

    /** Imports @p rows into @p staging at the vectors matrix pitch in the stored scalar, the
     *  fragment padding zeroed, widening through the f32 row at @p widened where they narrow at
     *  unit RMS from f64. */
    void stage_rows_(byte_t* staging, rows_view_t rows, byte_t* widened) const noexcept {
        nk::vector_span<f32_t> const widened_row {reinterpret_cast<f32_t*>(widened), dimensions()};
        for (std::size_t row = 0; row != rows.rows; ++row) {
            byte_t* const landing = staging + row * row_bytes();
            // Zeroed whole, not past the packed bytes, as packed casts write whole bytes only.
            std::memset(landing, 0, row_bytes());
            import_row<typename elementwise_ladder<import_capability_k>::kit_t,
                       typename rmscast_ladder<scalar_t, import_capability_k>::kit_t>(
                rows, row, kernel_, nk::vector_span<scalar_t> {reinterpret_cast<scalar_t*>(landing), dimensions()},
                widened_row);
        }
    }

    /** Zeroes @p counters' assigned row and telemetry, keeping the published row. */
    static void reset_counters_(metal_buffer_t& counters) noexcept {
        u32_t* const words = metal_words(counters);
        u32_t const published = words[metal_published_k];
        std::memset(words, 0, counters.bytes);
        words[metal_published_k] = published;
    }

    /** Binds the arguments, the two matrices and the norms column, the vectors standing in for an
     *  empty norms column. */
    void bind_storage_(metal_launch_t& launch, metal_arguments_t const& arguments) const noexcept {
        launch.bind_bytes(&arguments, sizeof(arguments), metal_arguments_buffer_k);
        launch.bind(vectors_matrix_, metal_vectors_buffer_k);
        launch.bind(stores_norms() ? norms_column_ : vectors_matrix_, metal_norms_buffer_k);
        launch.bind(neighbors_matrix_, metal_neighbors_buffer_k);
    }

    /** The pipeline of @p kernel at @p candidate_slots slots, built at the first @c reserve. */
    metal_pipeline_state_t pipeline_(vram_kernel_t kernel, u32_t candidate_slots) const noexcept {
        return pipelines_[static_cast<u32_t>(kernel)][gpu_rung_index(candidate_slots)];
    }

    /** Builds every pipeline this dtype's kernels need, once at the first @c reserve, so no launch
     *  waits on the device's registry, the arena frontier's rung included; a pipeline too
     *  register-heavy to hold one threadgroup of four simdgroups refuses with @c unsupported_k. */
    status_t load_pipelines_() noexcept {
        metal_context_t* const context = metal_context(device_.ordinal);
        if (!context) return status_t::device_error_k;
        char const* const source = metal_source<vector_dtype_, capability_>();
        vram_kernel_symbol_t symbol;
        for (metal_rung_t const rung : metal_rungs) {
            metal_kernel_symbol(symbol, rung);
            auto const [state, status] = context->pipeline(source, symbol.text);
            if (failed(status)) return status;
            if (state.max_threads < metal_simdgroups_k * context->wave_lanes) return status_t::unsupported_k;
            pipelines_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)] = state;
        }
        return status_t::success_k;
    }

    /** One searching simdgroup's arena slice on the rung of @p candidate_slots, the arena
     *  frontier's over @p expansion at zero, then the batch spans and the query span CUDA keeps in
     *  shared memory. */
    gpu_arena_shape_t walk_arena_shape_(u32_t expansion, u32_t candidate_slots) const noexcept {
        gpu_arena_shape_t shape = arena_shape_for(
            walk_selection_rows_(expansion, candidate_slots),
            candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k);
        shape.scratch_bytes = build_scratch_bytes_per_warp(0);
        return shape;
    }

    /** Rounds @p expansion_factor to a register rung of @p default_slots when it asks for none, in
     *  ranks of @ref wave_lanes. */
    u32_t candidate_slots_for_(u32_t expansion_factor, u32_t default_slots) const noexcept {
        return gpu_candidate_slots(expansion_factor, default_slots, wave_lanes());
    }

    /** Candidates a walk's diversity selection caches: all of the walk's candidates, as the CPU's
     *  selection judges, @p candidate_slots ranks per lane on the register ladder, or the arena
     *  frontier's @p expansion past it. */
    u32_t walk_selection_rows_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return candidate_slots ? candidate_slots * wave_lanes() : expansion;
    }

    /** Candidates a re-selection caches, a back-link round's: a row of the node's own neighbors, a
     *  row's worth of incoming newcomers, and the node's own row last. */
    u32_t reselection_rows_() const noexcept { return 2 * static_cast<u32_t>(connectivity()) + 1; }

    /** Bytes of one builder simdgroup's slice, its walk's or its re-selection's, whichever caches
     *  more: an insert and its back-link share one arena shape. */
    u32_t builder_scratch_bytes_per_warp_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return build_scratch_bytes_per_warp(
            max_of(walk_selection_rows_(expansion, candidate_slots), reselection_rows_()));
    }

    /** One building simdgroup's arena slice for @p walks walks on the rung of @p candidate_slots,
     *  the arena frontier's over @p expansion at zero, then the builders' scratch, which the slice
     *  always keeps, since threadgroup memory would bound the row width. */
    gpu_arena_shape_t build_arena_shape_(u32_t expansion, u32_t candidate_slots, u32_t walks) const noexcept {
        gpu_arena_shape_t shape = arena_shape_for(
            walk_selection_rows_(expansion, candidate_slots),
            candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k, walks);
        shape.scratch_bytes = builder_scratch_bytes_per_warp_(expansion, candidate_slots);
        return shape;
    }

    /** Bytes of one launch's counter words. */
    inline static constexpr std::size_t counters_bytes_k = metal_counter_words_k * sizeof(u32_t);

    /** Threads per threadgroup, four simdgroups of @ref wave_lanes. */
    u32_t threadgroup_threads_() const noexcept { return metal_simdgroups_k * wave_lanes(); }

    /** The ID a wave starting at @p wave_first stops before: as wide as the IDs below it, never
     *  narrower than @c vram_admission_ramp_floor_k, and never past @p end_id. */
    static u32_t wave_end_(u32_t wave_first, u32_t end_id) noexcept {
        return static_cast<u32_t>(vram_admission_wave_end(wave_first, end_id, vram_dense_admission_share_log2_k));
    }

    /** Waves an insert of IDs @b [first_id,end_id) launches in. */
    static std::size_t insert_waves_(std::size_t first_id, std::size_t end_id) noexcept {
        std::size_t waves = 0;
        for (u32_t wave_first = static_cast<u32_t>(first_id); wave_first != end_id; ++waves)
            wave_first = wave_end_(wave_first, static_cast<u32_t>(end_id));
        return waves;
    }

    /** Submits the rows @p arguments names in waves, each committed behind the last on the
     *  context's stream, so every ID a wave admits walks rows already published. Each wave's insert
     *  threads its picks onto the chains at @p links_offset in the staging, and a back-link launch
     *  behind it merges them, one simdgroup per touched node, so no row is locked; the back-link
     *  rides the insert's @p blocks threadgroups and arena, caching a re-selection's rows. Each
     *  launch counts into its own words past the staged rows, since nothing on the stream zeroes
     *  one between waves. */
    status_t launch_insert_(metal_arguments_t const& arguments, u32_t candidate_slots, u32_t blocks,
                            std::size_t links_offset, traversal_context_t& context) noexcept {
        std::size_t const rows = arguments.end_id - arguments.first_id;
        std::size_t const rows_bytes = rows * row_bytes();
        std::size_t const next_offset = links_offset + std::size_t {arguments.end_id} * sizeof(u32_t);
        std::size_t const touched_offset = next_offset + rows * connectivity_ * sizeof(u32_t);
        std::size_t const counts_offset = touched_offset + rows * connectivity_ * sizeof(u32_t);
        auto const bind_links = [&](metal_launch_t& launch, std::size_t wave) noexcept {
            launch.bind(context.staging_, metal_link_heads_buffer_k, links_offset);
            launch.bind(context.staging_, metal_link_next_buffer_k, next_offset);
            launch.bind(context.staging_, metal_touched_ids_buffer_k, touched_offset);
            launch.bind(context.staging_, metal_touched_count_buffer_k, counts_offset + wave * sizeof(u32_t));
        };
        std::size_t wave = 0;
        for (u32_t wave_first = arguments.first_id; wave_first != arguments.end_id; ++wave) {
            u32_t const wave_end = wave_end_(wave_first, arguments.end_id);
            u32_t const offset = wave_first - arguments.first_id;
            metal_arguments_t wave_arguments = arguments;
            wave_arguments.first_id = wave_first;
            wave_arguments.end_id = wave_end;
            wave_arguments.epoch_base = arguments.epoch_base + offset;
            metal_launch_t launch {context.stream(), pipeline_(vram_kernel_t::insert_k, candidate_slots).pipeline};
            if (!launch) return status_t::device_error_k;
            bind_storage_(launch, wave_arguments);
            launch.bind(context.staging_, metal_staging_buffer_k, std::size_t {offset} * row_bytes());
            launch.bind(context.arena_, metal_arena_buffer_k);
            launch.bind(context.staging_, metal_counters_buffer_k, rows_bytes + 2 * wave * counters_bytes_k);
            bind_links(launch, wave);
            if (status_t const submitted = context.submit_(launch, blocks, threadgroup_threads_()); failed(submitted))
                return submitted;
            metal_launch_t back_link {context.stream(), pipeline_(vram_kernel_t::back_link_k, 0).pipeline};
            if (!back_link) return status_t::device_error_k;
            wave_arguments.selection_rows = reselection_rows_();
            bind_storage_(back_link, wave_arguments);
            back_link.bind(context.arena_, metal_arena_buffer_k);
            back_link.bind(context.staging_, metal_counters_buffer_k, rows_bytes + (2 * wave + 1) * counters_bytes_k);
            bind_links(back_link, wave);
            if (status_t const submitted = context.submit_(back_link, blocks, threadgroup_threads_());
                failed(submitted))
                return submitted;
            wave_first = wave_end;
        }
        return status_t::success_k;
    }

    /** Recomputes the norms of rows @b [first,first+rows) on the device. */
    status_t derive_norms_(std::size_t first, std::size_t rows) noexcept {
        metal_arguments_t arguments = arguments_();
        arguments.first_id = static_cast<u32_t>(first);
        arguments.end_id = static_cast<u32_t>(first + rows);
        metal_context_t* const context = metal_context(device_.ordinal);
        if (!context) return status_t::device_error_k;
        std::shared_lock storage {storage_};
        metal_launch_t launch {context->queue, pipeline_(vram_kernel_t::derive_norms_k, 0).pipeline};
        if (!launch) return status_t::device_error_k;
        launch.bind_bytes(&arguments, sizeof(arguments), metal_arguments_buffer_k);
        launch.bind(vectors_matrix_, metal_vectors_buffer_k);
        launch.bind(norms_column_, metal_norms_buffer_k);
        if (status_t const submitted = launch.submit(
                walk_blocks_(rows, resident_blocks_(vram_kernel_t::derive_norms_k, 0)), threadgroup_threads_());
            failed(submitted))
            return submitted;
        return launch.wait();
    }

#pragma endregion Launches

#pragma region Walks

    /** The batched search over the @p visible rows, under any indexer @p heads: submitted, awaited
     *  and read back. */
    template <typename heads_type_>
    expected<search_result_t> search_(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                      nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                      search_config_t config, std::size_t visible, heads_type_ const& heads) noexcept {
        std::size_t const queries_count = queries.rows;
        bool const wants_distances = result_distances.extent(0) != 0;
        if (!queries_count) return {{}, status_t::success_k};
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {{}, status_t::config_mismatch_k};
        if (expected<search_result_t> const submitted = submit_search_(queries, context, config, visible, heads);
            !submitted)
            return submitted;
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        land_search_(context, result_ids, result_distances);
        return {spent_(context, config.count), status_t::success_k};
    }

    /** The submission half of @c search: stages the queries and any indexer @p heads and launches
     *  the walks over the @p visible rows; until the context lands, its staging holds the matches
     *  @ref land_search_ reads back. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(rows_view_t queries, traversal_context_t& context, search_config_t config,
                                             std::size_t visible, heads_type_ const& heads = {}) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        std::size_t const queries_count = queries.rows;
        context.landing_ = {};
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (!queries_count) return {{}, status_t::success_k};
        std::size_t heads_count = 0;
        if constexpr (indexer_k) {
            heads_count = heads.weights.extent(1);
            if (heads.weights.extent(0) != queries_count || heads.heads.extent(0) != queries_count * heads_count ||
                heads.heads.extent(1) != dimensions())
                return {{}, status_t::config_mismatch_k};
        }
        // The expansion covers the count, so the ladder's candidates hold every rank, and past it
        // the arena frontier ranks everything it saw.
        u32_t const expansion = gpu_walk_breadth(config.expansion_factor, config.count);
        u32_t const count_slots = divide_round_up<u32_t>(config.count, wave_lanes());
        u32_t const candidate_slots = candidate_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        std::size_t const results = queries_count * config.count;
        std::size_t const rows_bytes = (queries_count + queries_count * heads_count) * row_bytes();
        std::size_t const weights_bytes = queries_count * heads_count * sizeof(f32_t);
        std::size_t const outputs_bytes = results * (sizeof(node_id_t) + sizeof(distance_t));
        auto staged = stage_walk_(queries, context, expansion, candidate_slots,
                                  rows_bytes + weights_bytes + outputs_bytes, visible);
        if (!staged) return {{}, staged.status};
        metal_arguments_t arguments = staged.value;
        arguments.count = config.count;
        arguments.heads_count = static_cast<u32_t>(heads_count);
        byte_t* const staging = context.staging_.host;
        if constexpr (indexer_k) {
            stage_rows_(staging + queries_count * row_bytes(), heads.heads,
                        staging + rows_bytes + weights_bytes + outputs_bytes);
            for (std::size_t query = 0; query != queries_count; ++query)
                std::memcpy(staging + rows_bytes + query * heads_count * sizeof(f32_t),
                            heads.weights.byte_data() + query * heads.weights.stride_bytes(0),
                            heads_count * sizeof(f32_t));
        }
        std::size_t const ids_offset = rows_bytes + weights_bytes;
        std::size_t const distances_offset = ids_offset + results * sizeof(node_id_t);
        context.landing_ = {queries_count, config.count, ids_offset, distances_offset, 0};
        if (status_t const walked = launch_walk_(context, arguments, candidate_slots, ids_offset, distances_offset,
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
    expected<search_result_t> multiply_(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                        nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                        traversal_context_t& context, search_config_t config,
                                        std::size_t visible) noexcept {
        std::size_t const rows_count = left.rows, wanted = node_ids.extent(1);
        if (node_ids.extent(0) != rows_count || products.extent(0) != rows_count || products.extent(1) != wanted ||
            counts.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (expected<search_result_t> const submitted = submit_multiply_(left, wanted, context, config, visible);
            !submitted)
            return submitted;
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        search_result_t result = spent_(context, 0);
        result.count = land_multiply_(context, node_ids, products, counts);
        return {result, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows and launches walks keeping up to
     *  @p wanted pairs each over the @p visible rows; until the context lands, its staging holds
     *  the pairs @ref land_multiply_ reads back. */
    expected<search_result_t> submit_multiply_(rows_view_t left, std::size_t wanted, traversal_context_t& context,
                                               search_config_t config, std::size_t visible) noexcept {
        std::size_t const rows_count = left.rows;
        context.landing_ = {};
        if (!rows_count) return {{}, status_t::success_k};
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 2);
        std::size_t const rows_bytes = rows_count * row_bytes();
        std::size_t const ids_bytes = rows_count * wanted * sizeof(node_id_t);
        std::size_t const products_bytes = rows_count * wanted * sizeof(dot_t);
        std::size_t const counts_bytes = rows_count * sizeof(u32_t);
        auto staged = stage_walk_(left, context, expansion, candidate_slots,
                                  rows_bytes + ids_bytes + products_bytes + counts_bytes, visible);
        if (!staged) return {{}, staged.status};
        metal_arguments_t arguments = staged.value;
        arguments.count = static_cast<u32_t>(wanted);
        std::size_t const products_offset = rows_bytes + ids_bytes;
        std::size_t const counts_offset = products_offset + products_bytes;
        context.landing_ = {rows_count, wanted, rows_bytes, products_offset, counts_offset};
        if (status_t const walked = launch_walk_(context, arguments, candidate_slots, rows_bytes, products_offset,
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
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const count = min_of<u32_t>(expansion, visible);
        if (failed(context.results.entries_.reserve(std::size_t {count} * 2))) return {{}, status_t::bad_alloc_k};
        context.results.entries_.extend_uninitialized(std::size_t {count} * 2);
        node_id_t* const staged_ids = reinterpret_cast<node_id_t*>(context.results.entries_.data() + count);
        distance_t* const staged_distances = reinterpret_cast<distance_t*>(staged_ids + count);
        vectors_view_t const one_query {query_vector.data(), {1, query_vector.size()}};
        config.expansion_factor = expansion;
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
     *  walking verb does before its kernel, on the rung of @p candidate_slots for a walk expanding
     *  @p expansion; the row an import widens through follows the @p staging_bytes the verb asks
     *  for. */
    expected<metal_arguments_t> stage_walk_(rows_view_t queries, traversal_context_t& context, u32_t expansion,
                                            u32_t candidate_slots, std::size_t staging_bytes,
                                            std::size_t visible) noexcept {
        if (!dimensions_) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rows_refusal_(queries); failed(refusal)) return {{}, refusal};
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(
                context, arguments, walk_arena_shape_(expansion, candidate_slots),
                walk_blocks_(queries.rows, resident_blocks_(vram_kernel_t::search_k, candidate_slots)), queries.rows);
            failed(sized))
            return {{}, sized};
        arguments.expansion = expansion;
        arguments.concurrent_walks = device_.multiprocessor_count *
                                     resident_blocks_(vram_kernel_t::search_k, candidate_slots) * metal_simdgroups_k;
        arguments.filling_rows = filling_rows<u32_t>(expansion, static_cast<u32_t>(connectivity()));
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, staging_bytes + widened_bytes_(), fresh);
            failed(staged))
            return {{}, staged};
        stage_rows_(context.staging_.host, queries, context.staging_.host + staging_bytes);
        arguments.sampling_bound = static_cast<u32_t>(visible);
        arguments.queries_count = static_cast<u32_t>(queries.rows);
        std::memset(context.counters_.host, 0, context.counters_.bytes);
        return {arguments, status_t::success_k};
    }

    /** Submits the search kernel over the staged queries, its outputs at the given staging offsets;
     *  an empty index lands its sentinels at once. */
    status_t launch_walk_(traversal_context_t& context, metal_arguments_t const& arguments, u32_t candidate_slots,
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
        metal_launch_t launch {context.stream(), pipeline_(vram_kernel_t::search_k, candidate_slots).pipeline};
        if (!launch) return status_t::device_error_k;
        bind_storage_(launch, arguments);
        launch.bind(context.staging_, metal_staging_buffer_k);
        launch.bind(context.arena_, metal_arena_buffer_k);
        launch.bind(context.counters_, metal_counters_buffer_k);
        launch.bind(context.staging_, metal_ids_buffer_k, ids_offset);
        launch.bind(context.staging_, metal_distances_buffer_k, keeps ? 0 : second_offset);
        launch.bind(context.staging_, metal_products_buffer_k, keeps ? second_offset : 0);
        launch.bind(context.staging_, metal_scored_counts_buffer_k, keeps ? third_offset : 0);
        launch.bind(context.staging_, metal_heads_buffer_k, heads_offset);
        launch.bind(context.staging_, metal_weights_buffer_k, weights_offset);
        launch.bind_bytes(&keeps_word, sizeof(keeps_word), metal_keeps_buffer_k);
        return context.submit_(
            launch, walk_blocks_(arguments.queries_count, resident_blocks_(vram_kernel_t::search_k, candidate_slots)),
            threadgroup_threads_());
    }

    /** The telemetry the last walk on @p context spent, with @p count matches per query. */
    static search_result_t spent_(traversal_context_t const& context, std::size_t count) noexcept {
        vram_telemetry_block_t const spent = context.spent();
        search_result_t result;
        result.count = count;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        result.tail_compactions = spent.tail_compactions;
        result.tail_dropped = spent.tail_dropped;
        result.visited_saturations = spent.visited_saturations;
        return result;
    }

#pragma endregion Walks

#pragma region Refinement Passes

    /** Where a refine's staging keeps the walked rows' shadow neighbors: past room for every row's
     *  ID, which the heal's passes list. */
    std::size_t shadow_offset_() const noexcept {
        return round_up_to_multiple<std::size_t>(size() * sizeof(u32_t), default_alignment_k);
    }

    /** Where a refine's staging keeps the parts' bases: past the shadow neighbors. */
    std::size_t bases_offset_() const noexcept { return shadow_offset_() + size() * connectivity() * sizeof(u32_t); }

    /** Bytes of a refine's staging over the rows published now, with room for @p parts bases. */
    std::size_t staging_bytes_(std::size_t parts) const noexcept { return bases_offset_() + parts * sizeof(u32_t); }

    /** Submits the re-selection of @p nodes, or of the IDs @p node_ids lists with @p nodes spanning
     *  its offsets, into the shadow matrix of the staging @ref plan_refine sized, each node first
     *  walking across the neighboring group in round @p round when @p parts parts are given.
     *  Nothing waits; @ref land_pass_ reads the pass once it landed. */
    status_t submit_pass_(traversal_context_t& context, refine_config_t config, iota_view<node_id_t> nodes,
                          std::span<u32_t const> node_ids, u32_t parts = 0, u32_t round = 0) noexcept {
        std::size_t const rows = size();
        if (!rows || nodes.first >= nodes.bound) return status_t::success_k;
        if (status_t const kept = status(); failed(kept)) return kept;
        std::size_t const walk_count = nodes.size();
        if (!node_ids.empty()) std::memcpy(context.staging_.host, node_ids.data(), node_ids.size_bytes());
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
        // Walking across makes two walks into one visited set, so the slice is shaped for both.
        u32_t const walks = parts > 1 ? 2 : 1;
        u32_t const blocks = device_.multiprocessor_count * resident_blocks_(vram_kernel_t::refine_k, candidate_slots);
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(
                context, arguments, build_arena_shape_(expansion, candidate_slots, walks), blocks, walk_count);
            failed(sized))
            return sized;
        arguments.first_id = nodes.first;
        arguments.selection_rows = walk_selection_rows_(expansion, candidate_slots);
        arguments.expansion = expansion;
        arguments.end_id = nodes.bound;
        arguments.sampling_bound = static_cast<u32_t>(rows);
        arguments.use_node_ids = node_ids.empty() ? 0 : 1;
        arguments.parts = parts > 1 ? parts : 0;
        arguments.round = round;
        reset_counters_(counters_);

        std::shared_lock storage {storage_};
        metal_launch_t launch {context.stream(), pipeline_(vram_kernel_t::refine_k, candidate_slots).pipeline};
        if (!launch) return status_t::device_error_k;
        bind_storage_(launch, arguments);
        launch.bind(context.staging_, metal_staging_buffer_k);
        launch.bind(context.arena_, metal_arena_buffer_k);
        launch.bind(counters_, metal_counters_buffer_k);
        launch.bind(context.staging_, metal_shadow_buffer_k, shadow_offset_());
        launch.bind(context.staging_, metal_part_bases_buffer_k, bases_offset_());
        return context.submit_(launch, blocks, threadgroup_threads_());
    }

    /** Lands the pass @ref submit_pass_ submitted over @p nodes or @p node_ids, once it finished:
     *  every walked row replaces the live one, and the rows that changed are counted. */
    refine_result_t land_pass_(traversal_context_t& context, iota_view<node_id_t> nodes,
                               std::span<u32_t const> node_ids) noexcept {
        std::size_t const walk_count = nodes.size();
        std::size_t const connectivity = this->connectivity();
        // Exclusive pass: the walked rows land in the live matrix with plain stores.
        refine_result_t result;
        u32_t* const live = neighbor_rows_();
        u32_t const* const shadow = reinterpret_cast<u32_t const*>(context.staging_.host + shadow_offset_());
        for (std::size_t offset = 0; offset != walk_count; ++offset) {
            std::size_t const node = node_ids.empty() ? nodes.first + offset : node_ids[offset];
            u32_t* const row = live + node * connectivity;
            u32_t const* const picked = shadow + node * connectivity;
            if (std::memcmp(row, picked, connectivity * sizeof(u32_t)) == 0) continue;
            std::memcpy(row, picked, connectivity * sizeof(u32_t));
            ++result.changed_nodes;
        }
        u32_t const* const counters = metal_words(counters_);
        result.visited_nodes = walk_count;
        result.visited_members = metal_tally(counters, metal_hops_k);
        result.computed_distances = metal_tally(counters, metal_scored_k);
        result.restarts = metal_tally(counters, metal_restarts_k);
        return result;
    }

    /** Sweeps forward reachability on the host from as many stride-spread landmarks as one search
     *  draws, collecting the rows it never reached into @c islands; @p audit grows to the rows
     *  published now and keeps its capacity for later rounds and calls. */
    status_t audit_(metal_audit_t& audit) const noexcept {
        std::size_t const rows = size(), connectivity = this->connectivity();
        std::size_t const seed_count = min_of(connectivity, rows);
        for (growing_buffer<u8_t>* marks : {&audit.reached, &audit.frontier, &audit.next_frontier}) {
            if (failed(marks->reserve_exactly(rows))) return status_t::bad_alloc_k;
            marks->clear();
            marks->extend_uninitialized(rows);
        }
        if (failed(audit.islands.reserve_exactly(rows))) return status_t::bad_alloc_k;
        u32_t const* const neighbors = neighbor_rows_();
        std::memset(audit.reached.data(), 0, rows);
        std::memset(audit.frontier.data(), 0, rows);
        for (std::size_t seed = 0; seed != seed_count; ++seed) {
            std::size_t const row = seed * rows / seed_count;
            audit.reached[row] = audit.frontier[row] = 1;
        }
        // Each sweep's frontier holds only rows the one before reached first, so the sweeps end
        // once every reachable row was a frontier.
        for (audit.sweeps = 1;; ++audit.sweeps) {
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
        return status_t::success_k;
    }

    /** Reconnects each island into the first of its reached picks with room for it, or into the
     *  landmark covering its ID: a free slot when one is left, otherwise a landing spread across
     *  the tail by its ID that never evicts a sibling island nor a reached resident left with no
     *  other reached in-link. Answers the links written. */
    std::size_t reconnect_(metal_audit_t& audit) noexcept {
        std::size_t const rows = size(), connectivity = this->connectivity();
        std::size_t const seed_count = min_of(connectivity, rows);
        std::size_t const spread = heal_spread(connectivity);
        u32_t* const neighbors = neighbor_rows_();
        growing_buffer<u32_t>& in_links = audit.in_links;
        if (failed(in_links.reserve_exactly(rows))) return 0;
        in_links.clear();
        in_links.extend_uninitialized(rows);
        std::memset(in_links.data(), 0, rows * sizeof(u32_t));
        for (std::size_t row = 0; row != rows; ++row) {
            if (!audit.reached[row]) continue;
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                u32_t const target = neighbors[row * connectivity + slot];
                if (target != node_id_unknown_k && target != row && target < rows) ++in_links[target];
            }
        }
        std::size_t reconnected = 0;
        for (u32_t const island : audit.islands) {
            auto const land = [&](u32_t target) noexcept {
                u32_t* const target_row = neighbors + static_cast<std::size_t>(target) * connectivity;
                for (std::size_t slot = 0; slot != connectivity; ++slot) {
                    if (target_row[slot] == island) return true;
                    if (target_row[slot] != node_id_unknown_k && target_row[slot] != target) continue;
                    target_row[slot] = island, ++reconnected;
                    return true;
                }
                for (std::size_t attempt = 0; attempt != spread; ++attempt) {
                    std::size_t const landing = connectivity - 1 - (island + attempt) % spread;
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

    /** Rows claimed by batches and loads, visible or not; a closed word holds none past the
     *  watermark. */
    std::size_t claimed_() const noexcept { return claims_.claimed(size()); }

    /** Claims @p rows from the watermark: one compare-exchange that succeeds only on an open word
     *  with no batch or load in flight, so its holder alone runs and publishes them. */
    expected<node_id_t> claim_(std::size_t rows) noexcept {
        auto const [first_id, claim_status] = claims_.claim(size(), rows, ceiling_rows_);
        return {static_cast<node_id_t>(first_id), claim_status};
    }

    /** Gives back the claim over @p rows from @p first, which launched nothing; a word that moved
     *  on, closed by a caller around a load, stays as it is. */
    void unclaim_(std::size_t first, std::size_t rows) noexcept { claims_.unclaim(first, rows); }

    /** Closes the claim word at the watermark, so every claim refuses until @ref reopen_; refused
     *  while a batch or load is in flight. */
    status_t close_() noexcept { return claims_.close(size()); }

    /** Reopens a closed claim word at the watermark, past every row a load published meanwhile; an
     *  open one stays as it is. */
    void reopen_() noexcept { claims_.reopen(size()); }

    /** Forgets every row under the closed claim word: the published neighbor rows refill with the
     *  sentinel once the walks sharing the storage drained, and the watermark drops to zero, a kept
     *  failure with it, so IDs restart from zero on the storage already grown, which nothing frees.
     *  */
    void forget_() noexcept {
        std::unique_lock storage {storage_};
        if (std::size_t const rows = size())
            std::memset(neighbors_matrix_.host, 0xFF, rows * connectivity() * sizeof(u32_t));
        standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
        standard_atomic_ref<status_t>(status_).store(status_t::success_k, std::memory_order_release);
    }

    /** Row zero of the neighbors matrix, as the IDs its rows hold. */
    node_id_t* neighbor_rows_() const noexcept { return reinterpret_cast<node_id_t*>(neighbors_matrix_.host); }

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

    /** Rows the index may ever host, fixed by the first @c reserve. */
    std::size_t ceiling_rows_ = max_node_count_k;

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

    /** Every kernel's pipeline per candidate rung, owned by the device's registry. */
    metal_pipeline_state_t pipelines_[static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};

    /** Shared by every walk while its command buffer runs, taken alone by growth. */
    std::shared_mutex storage_;

    /** Held around every growth, a load's whole life and a refinement, whose host writes into the
     *  storage a growth would relocate. */
    std::mutex batch_;

    /** The claim word, closed while a refine, an absorb, a load or a save of the sharded index
     *  holds it. */
    claim_word_t claims_;
    std::size_t last_visible_id_ = 0;
    status_t status_ = status_t::success_k;

#pragma endregion State
};

} // namespace unum::usearch::dense
