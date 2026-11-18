/**
 *  @file include/usearch/dense/vram_apple_index.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Vector Search index for the unified memory of one Apple GPU.
 *
 *  The Apple twin of @c vram_nvidia_index: the same flat storage, the same walk, insert, health and
 *  round kernels ported to Metal in @c vram_apple_index.metal, and the same verbs. Memory is
 *  unified, so every matrix and column is a shared buffer the host reads and writes in place:
 *  checkpoints copy without a bounce, and a round's barriers draw hubs and order entries by
 *  neighborhood over the columns the device just wrote. A matrix grows by allocating a wider
 *  buffer, under an exclusive hold that every walk shares while its command buffer runs.
 *
 *  Metal promises no forward progress between command buffers, while an insert's admission ramp
 *  waits on rows earlier batches publish, so the claim word admits one batch or load at a time and
 *  refuses another with @c contended_k; searches overlap them freely. The kernels compile from the
 *  embedded source at the first @c reserve, once per storage dtype and Apple class on each device.
 */
#pragma once

#include <array>   // `std::array` - a health check's entries per depth
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::snprintf` - the source's prelude
#include <cstring> // `std::memcpy`, `std::memset`

#include <functional> // `std::hash` - the default hash of a key

#include <atomic>       // `std::memory_order`
#include <bit>          // `std::bit_ceil`, `std::bit_width`, `std::countr_zero`, `std::has_single_bit`
#include <mutex>        // `std::mutex` - the growth hold
#include <shared_mutex> // `std::shared_mutex` - walks share the storage, growth takes it alone
#include <span>         // `std::span`
#include <utility>      // `std::exchange`, `std::forward`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the published words
#include <forkunion/types.hpp>   // `fu::wait_capped_t` - the watermark's turn

#include <numkong/capabilities.h> // `nk_kernel_kind_t`
#include <numkong/tensor.hpp>     // `nk::matrix_span`, `nk::range`
#include <numkong/types.hpp>      // `nk::type_for`
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator_t` - the key column's placement
#include "usearch/utilities/ram_sorts.hpp"  // `order_by_neighborhood`, `split_wide_neighborhoods` - a round's barriers
#include "usearch/utilities/ram_tables.hpp" // `keyed_nodes` - every node's key and every key's nodes
#include "usearch/utilities/vram_apple_device.hpp"
#include "usearch/utilities/vram_device.hpp" // `vram_kernel_symbol_t` - a kernel's name
#include "usearch/utilities/vram_walks.hpp"  // `gpu_arena_shape`, `gpu_walk_visits`, `gpu_candidate_slots`
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_metric.hpp" // `import_row` - rows cross into the shared staging in the stored scalar
#include "usearch/utilities/vram_apple_shared.hpp" // `metal_counter_t`, `metal_extent_t` - read before the dense words
#include "usearch/dense/vram_apple_shared.hpp"     // `metal_arguments_t`, `metal_buffer_index_t`, `metal_metric_t`

namespace unum::usearch::dense {

static_assert(metal_max_gathers_in_flight_k == vram_max_gathers_in_flight_k, "The Metal round window is the CUDA one");

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

#if USEARCH_ARCH_METAL_

/** The vocabulary the dense index and its kernels share, spliced after the contexts'. */
inline constexpr char metal_shared_source_k[] = {
#embed "vram_apple_shared.hpp" suffix(, 0)
};

/** The MSL kits, embedded so the kernels compile on the device without a Metal toolchain. */
inline constexpr char metal_metric_source_k[] = {
#embed "vram_apple_metric.metal" suffix(, 0)
};

/** The MSL walk, insert, search, health and round kernels, compiled after the kits. */
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

/** One exported kernel of @p kernel_enum_type_: the kernel and its register rung, zero where it
 *  keeps no ladder or walks the arena frontier. */
template <typename kernel_enum_type_>
struct metal_rung {
    kernel_enum_type_ kernel;
    u32_t slots;
};

/** Every instantiation of a shared walk @c vram_apple_index.metal exports, which the host builds a
 *  pipeline for at the first @c reserve. */
inline constexpr metal_rung<vram_kernel_t> metal_rungs[] = {
    {vram_kernel_t::insert_k, 0},
    {vram_kernel_t::insert_k, 2},
    {vram_kernel_t::insert_k, 4},
    {vram_kernel_t::insert_k, 8},
    {vram_kernel_t::insert_k, 16},
    {vram_kernel_t::back_link_k, 0},
    {vram_kernel_t::search_k, 0},
    {vram_kernel_t::search_k, 2},
    {vram_kernel_t::search_k, 4},
    {vram_kernel_t::search_k, 8},
    {vram_kernel_t::search_k, 16},
    {vram_kernel_t::partition_propose_k, 0},
    {vram_kernel_t::partition_count_k, 0},
    {vram_kernel_t::partition_accept_k, 0},
    {vram_kernel_t::partition_fill_k, 0},
    {vram_kernel_t::gather_vectors_k, 0},
    {vram_kernel_t::sum_counts_k, 0},
    {vram_kernel_t::gather_neighborhoods_k, 0},
    {vram_kernel_t::restitch_k, 0},
    {vram_kernel_t::health_expand_k, 0},
    {vram_kernel_t::health_recheck_k, 0},
    {vram_kernel_t::order_islands_k, 0},
    {vram_kernel_t::health_islands_k, 0},
    {vram_kernel_t::island_components_k, 0},
    {vram_kernel_t::health_count_links_k, 0},
    {vram_kernel_t::label_islands_k, 0},
    {vram_kernel_t::heal_islands_k, 0},
};

/** Every dense-only instantiation @c vram_apple_index.metal exports, built beside the shared
 *  ones. */
inline constexpr metal_rung<vram_dense_kernel_t> metal_dense_rungs[] = {
    {vram_dense_kernel_t::derive_norms_k, 0},
};

/** Spells the @c host_name @c vram_apple_index.metal exports @p rung under: the kernel's name, then
 *  the candidate slots of a walking kernel's instantiation. */
inline void metal_kernel_symbol(vram_kernel_symbol_t& symbol, metal_rung<vram_kernel_t> rung) noexcept {
    symbol = {};
    switch (rung.kernel) {
    case vram_kernel_t::insert_k: symbol.append("usearch_dense_insert_kernel_"); break;
    case vram_kernel_t::search_k: symbol.append("metal_search_kernel_"); break;
    case vram_kernel_t::back_link_k: symbol.append("metal_back_link_kernel"); return;
    case vram_kernel_t::partition_propose_k: symbol.append("metal_partition_propose_kernel"); return;
    case vram_kernel_t::partition_count_k: symbol.append("metal_partition_count_kernel"); return;
    case vram_kernel_t::partition_accept_k: symbol.append("metal_partition_accept_kernel"); return;
    case vram_kernel_t::partition_fill_k: symbol.append("metal_partition_fill_kernel"); return;
    case vram_kernel_t::gather_vectors_k: symbol.append("metal_gather_vectors_kernel"); return;
    case vram_kernel_t::sum_counts_k: symbol.append("metal_sum_counts_kernel"); return;
    case vram_kernel_t::gather_neighborhoods_k: symbol.append("metal_gather_neighborhoods_kernel"); return;
    case vram_kernel_t::restitch_k: symbol.append("metal_restitch_kernel"); return;
    case vram_kernel_t::health_expand_k: symbol.append("metal_health_expand_kernel"); return;
    case vram_kernel_t::health_recheck_k: symbol.append("metal_health_recheck_kernel"); return;
    case vram_kernel_t::order_islands_k: symbol.append("metal_order_islands_kernel"); return;
    case vram_kernel_t::health_islands_k: symbol.append("metal_health_islands_kernel"); return;
    case vram_kernel_t::island_components_k: symbol.append("metal_island_components_kernel"); return;
    case vram_kernel_t::health_count_links_k: symbol.append("metal_health_count_links_kernel"); return;
    case vram_kernel_t::label_islands_k: symbol.append("metal_label_islands_kernel"); return;
    case vram_kernel_t::heal_islands_k: symbol.append("metal_heal_islands_kernel"); return;
    default: return;
    }
    symbol.append_value(rung.slots);
}
inline void metal_kernel_symbol(vram_kernel_symbol_t& symbol, metal_rung<vram_dense_kernel_t> rung) noexcept {
    symbol = {};
    if (rung.kernel == vram_dense_kernel_t::derive_norms_k) symbol.append("metal_derive_norms_kernel");
}

/** What a search launch writes: ranked distances, or every scored pair of a product. */
enum class metal_output_t : bool { distances_k, products_k };

/**
 *  @brief Vector Search index for the unified memory of one Apple GPU.
 *
 *  Speaks @c vram_nvidia_index 's verbs: a batch's IDs are one contiguous range turning visible a
 *  whole batch at a time, the geometry is fixed at construction, and the first growth builds the
 *  pipelines. Every node's key lives in a host @c keyed_nodes the claim holder writes and files
 *  before the watermark turns, as the storage is shared memory the walks read in place. Attention
 *  is refused with @c unsupported_k until its value-lookup ABI is designed for Metal, and walks
 *  past the widest register rung take the arena frontier, as on NVIDIA and AMD.
 *  A batch waits through @p wait_type_ for the batches before it to publish.
 */
template <                                                          //
    nk_dtype_t vector_dtype_ = nk_i8_k,                             //
    nk_capability_t capability_ = nk_cap_metal_k,                   //
    typename wait_type_ = dummy_waiter_t,                           //
    typename vectors_allocator_type_ = metal_shared_allocator_t,    //
    typename neighbors_allocator_type_ = metal_shared_allocator_t,  //
    typename traversals_allocator_type_ = metal_shared_allocator_t, //
    typename key_type_ = uuid128_t,                                 //
    typename key_hash_type_ = std::hash<key_type_>                  //
    >
struct vram_apple_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using wait_t = wait_type_;
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
    using candidates_count_t = u32_t;
    using traversal_context_t = metal_traversal_context<traversals_allocator_t>;
    using refine_report_t = dense::refine_report_t;
    using reachability_t = dense::reachability_t;
    using shared_lock_t = std::shared_lock<std::shared_mutex>;
    using unique_lock_t = std::unique_lock<std::shared_mutex>;
    using batch_lock_t = std::unique_lock<std::mutex>;
    static_assert(sizeof(norms_t) == 8, "The kernels address norms as two 32-bit words");

    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    /** The 32-bit ID space, less the sentinel, which also marks a closed claim word. */
    inline static constexpr std::size_t max_node_count_k = 0xFFFFFFFEu;

    /** Fragment bytes - the kits read rows sixteen bytes at a time, and rows pad to whole
     *  fragments. */
    inline static constexpr std::size_t fragment_bytes_k = metal_fragment_bytes_k;

    /** The host capability rows import into the shared staging at: the variant's own names only the
     *  GPU, and every Apple silicon host has NEON. */
    inline static constexpr nk_capability_t import_capability_k = nk_cap_serial_k | nk_cap_neon_k;

    /** The caller-visible identity of every node. */
    using key_t = key_type_;

    /** The keys of a batch, one per row. */
    using keys_view_t = nk::vector_view<key_t>;

    /** Every node's key and every key's nodes, growing in the RAM residency's chunks. */
    using keyed_t = keyed_nodes<key_t, node_id_t, chunk_schedule<10, 20>, max_node_count_k, forkunion_allocator_t,
                                fu::standard_atomic_ref, key_hash_type_>;

    /** The dimensions and connectivity every row keeps, fixed at construction. */
    using geometry_t = dense::geometry_t;

    /** The @p kernel every walk scores by and the @p geometry every row keeps, both checked at the
     *  first growth, and one allocator per storage concern; the index pins itself to the vectors
     *  allocator's device, demands the neighbors allocator agree, and keeps its keys and key map on
     *  @p keys_allocator. */
    explicit vram_apple_index(nk_kernel_kind_t kernel, geometry_t geometry,
                              vectors_allocator_t const& vectors_allocator = {},
                              neighbors_allocator_t const& neighbors_allocator = {},
                              forkunion_allocator_t const& keys_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, kernel_ {kernel}, geometry_ {geometry},
          vectors_allocator_ {vectors_allocator}, neighbors_allocator_ {neighbors_allocator}, keyed_ {keys_allocator} {}
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
        return fu::standard_atomic_ref<std::size_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed rows unpublished, kept for the index's lifetime. */
    status_t status() const noexcept {
        status_code_t const code =
            fu::standard_atomic_ref<status_code_t const>(status_.code).load(std::memory_order_acquire);
        return {code, fu::standard_atomic_ref<char const* const>(status_.note).load(std::memory_order_acquire)};
    }

    /** Rows the storage holds without growing. */
    std::size_t capacity() const noexcept { return capacity_rows_; }
    std::size_t connectivity() const noexcept { return geometry_.connectivity; }
    std::size_t dimensions() const noexcept { return geometry_.dimensions; }

    /** The dimensions & connectivity fixed at construction. */
    geometry_t geometry() const noexcept { return geometry_; }

    /** Every node's key, and every key's nodes; readers pass the watermark they acquired. */
    keyed_t const& keys() const noexcept { return keyed_; }

    /**
     *  The key of the visible @p node.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible.
     */
    expected<key_t> key_of(node_id_t node) const noexcept {
        if (node >= size()) return {{}, status_code_t::config_mismatch_k};
        return {keyed_[node], status_code_t::success_k};
    }

    /** Invokes @p callback for every published node stored under @p key, as of one acquire of the
     *  watermark - wait-free. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        keyed_.find_each(key, static_cast<node_id_t>(size()), callback);
    }

    /** The visible nodes stored under @p key, as of one acquire of the watermark. */
    std::size_t count(key_t const& key) const noexcept { return keyed_.count(key, static_cast<node_id_t>(size())); }

    /** Whether any visible node is stored under @p key, as of one acquire of the watermark. */
    bool contains(key_t const& key) const noexcept { return keyed_.contains(key, static_cast<node_id_t>(size())); }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    candidates_count_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor
                                       : static_cast<candidates_count_t>(default_expansion_factor(connectivity()));
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

    /** Bytes of one builder simdgroup's slice over @p selection_nodes cached candidates: one
     *  batch's scored nodes, IDs and dots, which CUDA keeps in shared memory, then the query span
     *  while walking, then the candidate cache, and per cached candidate its score, three ID
     *  spans, its distance and its norms. */
    u32_t build_scratch_bytes_per_warp(u32_t selection_nodes) const noexcept {
        u32_t const candidate_bytes = static_cast<u32_t>(row_bytes() + gpu_scored_node_bytes<node_id_t> +
                                                         3 * sizeof(u32_t) + sizeof(f32_t) + sizeof(norms_t));
        return metal_batch_spans_bytes_k + max_of(static_cast<u32_t>(row_bytes()), selection_nodes * candidate_bytes);
    }

    /** Which frontier a simdgroup's arena slice serves: the register ladder needs only the exact
     *  visited set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** One simdgroup's arena slice for @p walks walks sharing one visited set over
     *  @p candidate_nodes candidates each: the exact visited set over every ID they claim, plus the
     *  spill tail when @p frontier_kind outgrows registers, what their expansions stage. */
    gpu_arena_shape_t arena_shape_for(u32_t candidate_nodes, frontier_kind_t frontier_kind,
                                      u32_t walks = 1) const noexcept {
        u32_t const connectivity = static_cast<u32_t>(this->connectivity()), lanes = wave_lanes();
        u32_t const visits = walks * gpu_walk_visits(candidate_nodes, connectivity, lanes);
        u32_t const tail = frontier_kind == frontier_kind_t::wide_arena_k
                               ? walks * gpu_walk_tail_entries(candidate_nodes, connectivity, lanes)
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

    /** Whether an insert launch fits on this device: every simdgroup the widest register rung's
     *  grid keeps in flight takes that rung's slice beside its builder scratch, within one buffer,
     *  since threadgroup memory would bound the row width. */
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

    /**
     *  Copies the stored row of @p node into @p destination with the fragment padding stripped.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible, or @p destination is not as wide as a
     *    row.
     *  - @c unsupported_layout_k The @p destination is not contiguous.
     */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination) noexcept {
        if (node >= size()) return status_code_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_code_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_code_t::unsupported_layout_k;
        shared_lock_t storage {storage_};
        std::memcpy(destination.byte_data(), vectors_matrix_.host + node * row_bytes(), packed_row_bytes());
        return status_code_t::success_k;
    }

    /** Writes up to @p wanted rows stored under @p key into @p vectors in their scalar, any NumKong
     *  casts the stored one into, read in place from unified memory, and answers the key's true
     *  total; the storage lock keeps a growth from freeing the rows mid-copy. */
    std::size_t get(key_t const& key, rows_span_t vectors, std::size_t wanted) noexcept {
        rows_span_t const target {vectors.data, vectors.rows, dimensions(), vectors.stride, vectors.dtype};
        shared_lock_t storage {storage_};
        std::size_t total = 0;
        find_each(key, [&](node_id_t node) noexcept {
            if (total < wanted && total < target.rows)
                export_row<typename elementwise_ladder<import_capability_k>::kit_t>(
                    vectors_matrix_.host + node * row_bytes(), vector_dtype_k, target, total);
            ++total;
        });
        return total;
    }

    /**
     *  @brief Copies the neighbor rows of the nodes from @p first on, @c connectivity IDs each,
     *      into the host @p rows, raw as stored, sentinels and self-loops left in; the storage
     *      lock keeps a growth from freeing the rows mid-copy.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p rows hold no whole number of rows, or the rows pass @c size.
     */
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) noexcept {
        std::size_t const row_ids = connectivity();
        if (!row_ids || rows.size() % row_ids || std::size_t {first} + rows.size() / row_ids > size())
            return status_code_t::config_mismatch_k;
        shared_lock_t storage {storage_};
        std::memcpy(rows.data(), neighbor_rows_() + std::size_t {first} * row_ids, rows.size_bytes());
        return status_code_t::success_k;
    }

    /** The serialized extents. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /** Where a save writes: row-major host columns the caller sized from @c saved_shape. */
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

        /** Neighbor IDs per row of @c neighbors, at most the index's connectivity, a load padding
         *  each row with empty slots for a refine to fill; zero reads the index's own. */
        std::size_t connectivity = 0;
    };

#pragma endregion Measures

#pragma region Reservation

    /** Requested capacity for a @c reserve() call. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;

        /** Rows the index may ever host, clamped to the ID space; fixed by the first growth a
         *  @c reserve makes, past it every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;

        /** A request for @p entries; the ceiling is set by name only, so no positional field binds
         *  on it. */
        reserve_capacity_t(std::size_t entries = 0) noexcept : capacity(entries) {}
    };

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries - a hint, since every
     *      claim grows the storage it needs; the first growth builds the pipelines.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k The metric is not one the kits score for this scalar.
     *  - @c config_mismatch_k The dimensions are zero, the connectivity is not a power of two, or
     *    the neighbors live on another device.
     *  - @c overflow_k The capacity passes the ID space.
     *  - @c device_k The device is not listed.
     *  - @c unsupported_k A pipeline's simdgroups or threadgroups are narrower than the kernels
     *    need, or the device's registry lists no core count.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (config.capacity > max_node_count_k) return {capacity(), status_code_t::overflow_k};
        batch_lock_t batch {batch_};
        if (!counters_) ceiling_nodes_ = min_of(config.ceiling, max_node_count_k);
        if (status_t const bound = bind_(); failed(bound)) return {capacity(), bound};
        status_t const grown = grow_nodes_(config.capacity);
        return {capacity(), grown};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {
        candidates_count_t expansion_factor = 0;
    };

    /** The bytes a context grows to before it can insert @p rows rows: its counters, one arena
     *  slice per simdgroup in flight, and the staged rows beside the row an import widens through
     *  and the back-link chains; zero before the first growth. */
    std::size_t traversal_memory_usage(std::size_t rows, insert_config_t config = {}) const noexcept {
        if (!geometry_.dimensions || !device_.multiprocessor_count) return 0;
        u32_t const expansion = static_cast<u32_t>(vram_insert_expansion(config.expansion_factor));
        u32_t const candidate_slots = candidate_slots_for_(expansion);
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

    /**
     *  Inserts @p new_rows under @p keys and the contiguous IDs claimed from the watermark, growing
     *  the storage on demand, waits for the device, files the keys, publishes the rows, and reports
     *  telemetry. One batch or load holds the claim word at a time, so another one in flight, or a
     *  closed word, answers @c contended_k.
     *
     *  @par Statuses:
     *  - @c cancelled_k The index keeps it, as @c status says.
     *  - @c config_mismatch_k The @p keys are not one per row, the rows are not as wide as the
     *    index, @p assigned_ids are not one per row, @p context rides another device or cannot
     *    walk, or @c status keeps it.
     *  - @c unsupported_layout_k Rows sit closer than one packed row apart, or @c status keeps it.
     *  - @c overflow_k The rows pass the ceiling.
     *  - @c contended_k Another batch or load is in flight, or the claim word is closed.
     */
    expected<insert_result_t> insert(keys_view_t keys, rows_view_t new_rows, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        if (keys.size() != new_rows.rows) return {{}, status_code_t::config_mismatch_k};
        if (!new_rows.rows) return {{claimed_(), static_cast<node_id_t>(claimed_())}, status_code_t::success_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        auto const [first_id, claim_status] = claim_(new_rows.rows);
        if (failed(claim_status)) return {{}, claim_status};
        auto [inserted, insert_status] = submit_insert_(keys, new_rows, first_id, assigned_ids, context, config);
        if (failed(insert_status)) {
            // A refusal that kept no failure launched nothing.
            if (succeeded(status())) unclaim_(first_id, new_rows.rows);
            return {inserted, insert_status};
        }
        publish_keys_({first_id, static_cast<node_id_t>(first_id + new_rows.rows)});
        if (status_t const published = publish_(inserted.first_id, new_rows.rows); failed(published))
            return {{inserted.new_size, inserted.first_id}, published};
        return {inserted, insert_status};
    }

#pragma endregion Insertion

#pragma region Search

    /** Optional tuning knobs of @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {
        candidates_count_t expansion_factor = 0;
        candidates_count_t count = 10;
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
     *  writing @c config.count columns per row: its counters, one arena slice per simdgroup in
     *  flight, and the staged rows beside their outputs. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        if (!geometry_.dimensions || !device_.multiprocessor_count) return 0;
        // Search widens the candidates to cover its count while multiply walks the bare factor, so
        // the wider one sizes.
        u32_t const expansion = max_of<u32_t>(gpu_walk_breadth(config.expansion_factor, config.count),
                                              non_zero_expansion_factor(config));
        u32_t const candidate_slots = candidate_slots_for_(expansion);
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

    /**
     *  The nearest published neighbors of every query row, ascending, with sentinel IDs past the
     *  live matches.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The outputs hold other than one row per query or fewer than
     *    @c count columns, the queries are not as wide as the index, or @p context rides another
     *    device or cannot walk.
     *  - @c unsupported_k A scan is asked, which this vendor has none of.
     *  - @c unsupported_layout_k Query rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context, completion_t = completion_t::blocking_k) noexcept {
        return search_(queries, result_ids, result_distances, context, config, size(), dummy_heads_t {});
    }

    /**
     *  The batched @c search ranked by a lightning indexer; a dot verb, refusing every other
     *  metric.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the outputs hold other
     *    than one row per query or fewer than @c count columns, the heads or weights disagree with
     *    the queries, the queries are not as wide as the index, or @p context rides another device
     *    or cannot walk.
     *  - @c unsupported_k A scan is asked, which this vendor has none of.
     *  - @c unsupported_layout_k Query rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context, completion_t = completion_t::blocking_k) noexcept
        requires a_floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_code_t::config_mismatch_k};
        return search_(queries, result_ids, result_distances, context, config, size(), heads);
    }

    /**
     *  One query in, the nearest matches ascending in @c context.results.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The query is not as wide as the index, or @p context rides another
     *    device or cannot walk.
     *  - @c unsupported_k A scan is asked, which this vendor has none of.
     */
    expected<search_result_t> search(vector_view_t query_vector, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return search_one_(query_vector, context, config, size());
    }

    /**
     *  @brief Filtered search is refused: a host predicate cannot run inside the walk, and a
     *      device-resident admission bitmap is a design of its own.
     *
     *  @par Statuses:
     *  - @c device_k Every call.
     */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t, search_config_t, predicate_type_&&,
                                        traversal_context_t&) noexcept {
        return {{}, status_code_t::device_k};
    }

    /**
     *  Graph products of every left-hand row: each keeps the pairs its walk scored, until its
     *  output row fills.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the outputs or
     *    @p counts disagree with the rows, the rows are not as wide as the index, or @p context
     *    rides another device or cannot walk.
     *  - @c unsupported_layout_k Rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> multiply(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       search_config_t config, traversal_context_t& context,
                                       completion_t = completion_t::blocking_k) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_code_t::config_mismatch_k};
        return multiply_(left, node_ids, products, counts, context, config, size());
    }

    /**
     *  Attention is refused with @c unsupported_k: a Metal kernel cannot call the caller's value
     *  lookup, just as it cannot run a predicate.
     *
     *  @par Statuses:
     *  - @c unsupported_k Every call.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t, std::size_t, value_lookup_type_&&, f32_t, rows_span_t,
                                     std::span<softmax_state_t>, search_config_t, traversal_context_t&,
                                     completion_t = completion_t::blocking_k) noexcept {
        return {{}, status_code_t::unsupported_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** Optional tuning knobs of @c refine(): the dense refine config, as nothing here differs. */
    using refine_config_t = dense::refine_config_t;

    /** A reshard's knobs: its rounds', the shard capacity a mono-index refuses, and the seed
     *  that orders a split's growth. */
    using reshard_config_t = dense::reshard_config_t;

    struct reshard_plan_t;

    /** One target's part of a split as the host assigns it: its nodes, and those given slots so
     *  far. */
    struct split_part_t {
        std::size_t entries = 0, assigned_nodes = 0;
    };

    /** A split's state beside the rounds it turns into. */
    struct split_t {

        /** The index the split empties, until the scatter finished. */
        vram_apple_index* source = nullptr;

        /** The targets, one part each, and the one the round is bound to. */
        growing_buffer<vram_apple_index*> targets {global_allocator};
        growing_buffer<split_part_t> parts {global_allocator};
        std::size_t bound_target = 0;

        /** The source's nodes at planning, and the most nodes one part takes: those over the
         *  targets, rounded up. */
        std::size_t entries = 0, part_capacity = 0;

        /** The partition's shared buffer, laid out as @c split_layout_ places it, which the host
         *  reads in place between the phases. */
        metal_buffer_t device_bytes;

        /** The growth round the next batch queues from, zero once the growth converged. */
        u32_t next_round = 0;

        /** The round every target runs. */
        refine_config_t config {};

        /** The rounds of the targets settled so far, folded, and how many settled. */
        refine_report_t report {};
        std::size_t settled = 0;
    };

    /** Where a health check and a round keep each column in a context's round buffer, in bytes: the
     *  counters; the depths both ways, the in-link and shallower in-link counts, and the reached,
     *  reaching-root, two-ways and changed bits; then the islands' columns, which the round's
     *  alias, since only a check and the heal read the islands: each entry's neighborhood, the
     *  entries in neighborhood order, their starts with room for splits, a gather's hubs, its runs
     *  and the items before each run, and one selection's scratch per simdgroup. */
    struct round_layout_t {
        std::size_t counters = 0, depths_from_root = 0, depths_to_root = 0, in_links = 0, shallower_in_links = 0;
        std::size_t reached = 0, reaches_root = 0, two_ways_to_root = 0, changed = 0;
        std::size_t island_nodes = 0, island_components = 0, island_labels = 0, island_parents = 0;
        std::size_t island_entries = 0, island_roles = 0;
        std::size_t entry_neighborhoods = 0, neighborhood_entries = 0, neighborhood_offsets = 0, hub_nodes = 0;
        std::size_t runs = 0, run_items = 0, stitch = 0, bytes = 0;
    };

    /** What a round's columns are sized for, zeros sizing a lone check: the hubs one gather draws
     *  at most and the most entries a neighborhood holds, as the RAM round names them, the hubs and
     *  runs a gather may write, and the threadgroups a restitch runs. */
    struct round_sizing_t {
        std::size_t hubs_per_draw = 0, capacity = 0, hubs = 0, runs = 0, groups = 0;
    };

    /**
     *  @brief One health check on the device, the twin of the RAM audit: one root, exact depths
     *      both ways, in-link counts, and the islands' components, roles, entries and parents.
     *
     *  The islands' strong components are labeled in parallel, and the weak components counted by
     *  union-find. Its columns live in a context's round buffer, which its planner sized, so no
     *  pass allocates. @ref queue submits the current pass, the caller waits on the context, and
     *  @ref advance reads the counters over unified memory and picks the next pass, answering
     *  whether one follows.
     */
    struct health_plan_t {

        /** The watermark snapshot the check covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** Where the check's columns, and the round's beside them, sit in the round buffer. */
        round_layout_t layout {};

        /** Turns the check onto @p index 's IDs below @p visible, its columns at @p layout in the
         *  round buffer whose host bytes start at @p columns, so the next sweep draws its root. */
        void bind(vram_apple_index& index, node_id_t visible, round_layout_t const& layout, byte_t* columns) noexcept {
            index_ = &index;
            this->visible = visible;
            this->layout = layout;
            columns_ = columns;
            root_ = node_id_unknown_k;
        }

        /** Clears every column and starts from the root an earlier sweep kept, or else from a
         *  seeded draw, marked both ways at depth zero: the next pass expands depth one. */
        void restart_sweep() noexcept {
            // Opening and closing checks share their root, so no other draw can read as progress.
            bool const kept = root_ < visible;
            node_id_t const root = kept      ? root_
                                   : visible ? static_cast<node_id_t>(seeded_bijection(0, visible, 0))
                                             : node_id_unknown_k;
            drawn_root_ = kept ? node_id_unknown_k : root;
            start_from_(root);
        }

        /**
         *  @brief Submits the current pass on @p context 's stream, none once the check converged.
         *
         *  @par Statuses:
         *  - @c device_k The pass could not be encoded.
         */
        status_t queue(traversal_context_t& context) noexcept {
            metal_arguments_t arguments = index_->arguments_();
            arguments.end_id = visible;
            arguments.depth = static_cast<u32_t>(depth_);
            arguments.islands = static_cast<u32_t>(islands_);
            arguments.pass = pass_;
            u32_t* const words = counters();
            vram_apple_index& index = *index_;
            switch (step_) {
            case step_t::expand_k:
                words[metal_round_reached_k] = words[metal_round_marked_k] = 0;
                return index.launch_round_(vram_kernel_t::health_expand_k, arguments, layout, visible, context);
            case step_t::recheck_k:
                words[metal_round_marked_k] = 0;
                return index.launch_round_(vram_kernel_t::health_recheck_k, arguments, layout, visible, context);
            case step_t::order_islands_k:
                return index.launch_round_(vram_kernel_t::order_islands_k, arguments, layout, 0, context);
            case step_t::islands_k:
                return index.launch_round_(vram_kernel_t::health_islands_k, arguments, layout, islands_, context);
            case step_t::components_k:
                words[metal_round_relabeled_k] = 0;
                return index.launch_round_(vram_kernel_t::island_components_k, arguments, layout, islands_, context);
            case step_t::count_links_k:
                words[metal_round_unlinked_k] = words[metal_round_underlinked_k] = 0;
                return index.launch_round_(vram_kernel_t::health_count_links_k, arguments, layout, visible, context);
            case step_t::labels_k:
                words[metal_round_unreached_islands_k] = words[metal_round_stranded_islands_k] = 0;
                words[metal_round_components_k] = 0;
                return index.launch_round_(vram_kernel_t::label_islands_k, arguments, layout, islands_ + 1, context);
            default: return status_code_t::success_k;
            }
        }

        /** The barrier between passes, once the last one finished: files the entries a depth
         *  reached, turns to the rechecks once a depth reaches nothing, orders and labels the
         *  islands once a recheck marks nothing, re-roots once to the largest island when it
         *  outgrows the drawn root's component, then counts the links and labels the islands'
         *  roles; answers whether another pass follows. */
        bool advance() noexcept {
            u32_t const* const words = counters();
            switch (step_) {
            case step_t::expand_k: {
                std::size_t const reached = words[metal_round_reached_k];
                swept_ += reached + words[metal_round_marked_k];
                if (!reached) {
                    step_ = step_t::recheck_k;
                    return true;
                }
                depth_entries_[min_of<std::size_t>(++depth_, metal_deepest_depth_k)] += reached;
                ++depth_from_root_;
                return true;
            }
            case step_t::recheck_k:
                if (std::size_t const marked = words[metal_round_marked_k]; marked) {
                    swept_ += marked, ++depth_to_root_;
                    return true;
                }
                component_ = component_entries_();
                step_ = component_ == visible ? step_t::count_links_k : step_t::order_islands_k;
                return true;
            case step_t::order_islands_k:
                islands_ = words[metal_round_islands_k];
                step_ = step_t::islands_k;
                return true;
            case step_t::islands_k:
                unsettled_ = islands_, pass_ = 0;
                step_ = step_t::components_k;
                return true;
            case step_t::components_k:
                // Labels lower until none moves, then the islands whose labels meet settle.
                if (pass_ == 0) {
                    pass_ = words[metal_round_relabeled_k] ? 0 : 1;
                    return true;
                }
                unsettled_ -= words[metal_round_relabeled_k], pass_ = 0;
                return unsettled_ ? true : choose_root_();
            case step_t::count_links_k:
                unlinked_ = words[metal_round_unlinked_k], underlinked_ = words[metal_round_underlinked_k];
                if (!islands_) {
                    components_ = 1;
                    step_ = step_t::done_k;
                    return false;
                }
                pass_ = 0;
                step_ = step_t::labels_k;
                return true;
            case step_t::labels_k:
                if (pass_ == 2) {
                    unreached_islands_ = words[metal_round_unreached_islands_k];
                    stranded_islands_ = words[metal_round_stranded_islands_k];
                }
                if (pass_ != 3) {
                    ++pass_;
                    return true;
                }
                components_ = words[metal_round_components_k];
                step_ = step_t::done_k;
                return false;
            default: return false;
            }
        }

        /** The report of a converged check: the entries of the root's component, the depths each
         *  direction took, the in-link counts, and the islands' components. */
        reachability_t finish() const noexcept {
            reachability_t result;
            result.audited_nodes = visible;
            result.reachable_nodes = component_entries_();
            result.depth_from_root = depth_from_root_;
            result.depth_to_root = depth_to_root_;
            result.unlinked = unlinked_;
            result.underlinked = underlinked_;
            std::size_t most_populated = 0;
            for (std::size_t depth = 1; depth != depth_entries_.size(); ++depth)
                if (depth_entries_[depth] > depth_entries_[most_populated]) most_populated = depth;
            for (std::size_t depth = most_populated + 1; depth < depth_entries_.size(); ++depth)
                result.beyond_most_populated_depth += depth_entries_[depth];
            result.unreached_islands = unreached_islands_;
            result.stranded_islands = stranded_islands_;
            result.components = components_;
            return result;
        }

        /** Entries the sweeps reached plus entries they marked a way to the root for, of twice the
         *  checked ones: what a caller's progress counts. */
        std::size_t swept() const noexcept { return swept_; }

        /** Entries outside the root's component, the islands' nodes. */
        std::size_t islands() const noexcept { return islands_; }

        /** The counter words the passes count into, read over unified memory once a pass
         *  finished. */
        u32_t* counters() const noexcept { return reinterpret_cast<u32_t*>(columns_ + layout.counters); }

        /** The column at @p offset of the round buffer as @p value_type_ values. */
        template <typename value_type_>
        value_type_* column(std::size_t offset) const noexcept {
            return reinterpret_cast<value_type_*>(columns_ + offset);
        }

      private:
        /** The passes, as the RAM audit names its own, plus those labeling the islands on the
         *  device, which RAM runs on one thread once its rechecks end. */
        enum class step_t : u8_t {
            expand_k,
            recheck_k,
            order_islands_k,
            islands_k,
            components_k,
            count_links_k,
            labels_k,
            done_k
        };

        /** Entries reached from the root and reaching it, counted over the bitsets on the host. */
        std::size_t component_entries_() const noexcept {
            u32_t const* const reached = column<u32_t>(layout.reached);
            u32_t const* const reaches_root = column<u32_t>(layout.reaches_root);
            std::size_t entries = 0;
            for (std::size_t word = 0; word != divide_round_up<std::size_t>(visible, 32); ++word)
                entries += static_cast<std::size_t>(std::popcount(reached[word] & reaches_root[word]));
            return entries;
        }

        /** Closes one root's sweep once the island components settled: sweeps again from the first
         *  entry of the largest when it outgrows the drawn root's component, else counts links. */
        bool choose_root_() noexcept {
            u32_t const* const components = column<u32_t>(layout.island_components);
            u32_t const* const members = column<u32_t>(layout.island_entries);
            std::size_t largest = 0, largest_entries = 0;
            for (std::size_t rank = 0; rank != islands_; ++rank)
                if (components[rank] == rank && members[2 * rank] > largest_entries)
                    largest = rank, largest_entries = members[2 * rank];
            if (root_ == drawn_root_ && largest_entries > component_) {
                drawn_root_ = node_id_unknown_k;
                node_id_t const* const island_nodes = column<node_id_t>(layout.island_nodes);
                start_from_(island_nodes[largest]);
                return true;
            }
            step_ = step_t::count_links_k;
            return true;
        }

        /** Clears what one root's sweep fills, the round's changed bits aside, and marks @p root
         *  both ways at depth zero; with no root the check is over. */
        void start_from_(node_id_t root) noexcept {
            std::size_t const words = divide_round_up<std::size_t>(visible, 32) * sizeof(u32_t);
            std::memset(columns_ + layout.counters, 0, metal_round_counter_words_k * sizeof(u32_t));
            std::memset(columns_ + layout.depths_from_root, metal_unknown_depth_k, visible);
            std::memset(columns_ + layout.depths_to_root, metal_unknown_depth_k, visible);
            std::memset(columns_ + layout.in_links, 0, visible * sizeof(u32_t));
            std::memset(columns_ + layout.shallower_in_links, 0, visible * sizeof(u32_t));
            for (std::size_t bitset : {layout.reached, layout.reaches_root, layout.two_ways_to_root})
                std::memset(columns_ + bitset, 0, words);
            depth_entries_.fill(0);
            swept_ = root == node_id_unknown_k ? 0 : 2;
            component_ = islands_ = unsettled_ = 0;
            depth_ = depth_from_root_ = depth_to_root_ = 0;
            unlinked_ = underlinked_ = unreached_islands_ = stranded_islands_ = components_ = 0;
            pass_ = 0;
            root_ = root;
            step_ = root == node_id_unknown_k ? step_t::done_k : step_t::expand_k;
            if (root == node_id_unknown_k) return;
            u32_t* const reached = column<u32_t>(layout.reached);
            u32_t* const reaches_root = column<u32_t>(layout.reaches_root);
            u8_t* const depths_from_root = column<u8_t>(layout.depths_from_root);
            u8_t* const depths_to_root = column<u8_t>(layout.depths_to_root);
            reached[root / 32] |= 1u << (root % 32), reaches_root[root / 32] |= 1u << (root % 32);
            depths_from_root[root] = depths_to_root[root] = 0;
            depth_entries_[0] = 1;
        }

        /** The checked index; null on an unbound plan. */
        vram_apple_index* index_ = nullptr;

        /** The round buffer's host bytes, which planning sized and no pass moves. */
        byte_t* columns_ = nullptr;

        /** The node the sweeps start from, a seeded draw or the first entry of a larger island, and
         *  the root this check drew, which may give way once, or none for a kept root. */
        node_id_t root_ = node_id_unknown_k;
        node_id_t drawn_root_ = node_id_unknown_k;

        /** Entries each depth reached; the fullest is the depth the deep stratum draws past. */
        std::array<std::size_t, metal_unknown_depth_k> depth_entries_ {};

        /** Entries the sweeps reached and marked so far, of the root's component before the islands
         *  were gathered, outside it, and outside every settled island component. */
        std::size_t swept_ = 0, component_ = 0, islands_ = 0, unsettled_ = 0;

        /** The depth the next expansion reaches from, and the depths each direction took. */
        std::size_t depth_ = 0, depth_from_root_ = 0, depth_to_root_ = 0;

        /** The population counts the report holds, as @c reachability_t names them. */
        std::size_t unlinked_ = 0, underlinked_ = 0, unreached_islands_ = 0, stranded_islands_ = 0, components_ = 0;

        /** The pass within the components or labels the next batch queues. */
        u32_t pass_ = 0;
        step_t step_ = step_t::done_k;
    };

    /**
     *  @brief One refinement round in flight on this index's device, in the shape of the CUDA and
     *      HIP plans so one host thread drives every shard: @ref queue a phase, @ref wait for it,
     *      @ref advance past it, and @ref finish once no phase follows.
     *
     *  The round is the RAM one on Metal: the opening health check, the islands it named healed
     *  without a distance, a seeded draw of hubs, a gather filing every entry under its nearest
     *  hub, and again inside every neighborhood over its capacity, a restitch of each neighborhood
     *  in one core's threadgroup memory, every entry keeping the diverse links among its own and
     *  its nearest neighborhood-mates and no others, and the closing health check from the same
     *  root, which decides the verdict. Hubs are drawn and entries ordered by neighborhood on the
     *  host over unified memory at the barriers. @ref plan_refine sized every column, so no phase
     *  allocates. Every phase takes the context the plan was opened on back from its caller. Holds
     *  the index's batch lock from plan to death, so demands an exclusive caller.
     */
    struct refine_plan_t {
        refine_plan_t() noexcept = default;
        refine_plan_t(refine_plan_t&&) noexcept = default;
        refine_plan_t& operator=(refine_plan_t&&) noexcept = default;

        /** The health check the round opens and closes with, whose columns the reach rule reads. */
        health_plan_t health;

        /**
         *  @brief Submits the current phase on @p context 's stream; a hubs draw submits nothing,
         *      as the barrier closing it draws them on the host.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c device_k A phase could not be encoded.
         */
        status_t queue(traversal_context_t& context, refine_config_t) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (failed(status_)) return status_;
            switch (phase_) {
            case phase_t::health_k: status_ = health.queue(context); break;
            case phase_t::heal_k: status_ = index_->queue_heal_(health, context); break;
            case phase_t::gather_k: status_ = queue_gather_(context); break;
            case phase_t::restitch_k: status_ = queue_restitch_(context); break;
            default: break;
            }
            return status_;
        }

        /**
         *  @brief Waits on @p context for the submitted phase.
         *
         *  @par Statuses:
         *  - @c device_k The device failed the phase.
         */
        status_t wait(traversal_context_t& context) noexcept {
            if (index_ && succeeded(status_) && phase_ != phase_t::done_k) status_ = context.wait();
            return status_;
        }

        /** The barrier after a phase, one caller once it finished, answering whether one follows:
         *  takes a converged check in, draws the hubs, orders the entries by neighborhood or
         *  gathers the wide ones again, or reads the restitch's counts. */
        bool advance(traversal_context_t&) noexcept {
            if (!index_ || failed(status_) || phase_ == phase_t::done_k) return false;
            if (phase_ == phase_t::health_k) return health.advance() || converge_();
            u32_t const* const words = health.counters();
            switch (phase_) {
            case phase_t::heal_k:
                tally_.islands_reconnected += words[metal_round_healed_k];
                tally_.changed_nodes = words[metal_round_changed_k];
                // A round bound after a reshard's stop heals and verifies alone.
                if (tally_.verdict == refine_verdict_t::stopped_k) verify_(phase_t::restitch_k);
                else phase_ = phase_t::hubs_k;
                break;
            case phase_t::hubs_k:
                draw_hubs_();
                phase_ = phase_t::gather_k;
                break;
            // Neighborhoods over capacity gather again among their own entries.
            case phase_t::gather_k:
                if (!order_neighborhoods_()) phase_ = phase_t::restitch_k;
                break;
            case phase_t::restitch_k:
                tally_.reselected_nodes += health.visible;
                tally_.changed_nodes = words[metal_round_changed_k];
                tally_.added_links += metal_tally(words, metal_round_added_k);
                tally_.dropped_links += metal_tally(words, metal_round_dropped_k);
                verify_(phase_t::restitch_k);
                break;
            default: break;
            }
            return succeeded(status_);
        }

        /** Skips what is left of the round's phase in hand, as the RAM round's @c stop does: an
         *  opening health check, or a heal not yet run, ends the round with nothing written; a hubs
         *  draw, a gather or a restitch sends the round to the closing health check; the closing
         *  one runs on, and the verdict reads stopped. */
        void stop() noexcept {
            if (!rounding()) return;
            tally_.verdict = refine_verdict_t::stopped_k;
            switch (phase_) {
            case phase_t::health_k:
                if (swept_after_ == phase_t::none_k) phase_ = phase_t::done_k;
                break;
            case phase_t::heal_k:
                tally_.reachability = tally_.opening;
                phase_ = phase_t::done_k;
                break;
            case phase_t::hubs_k:
            case phase_t::gather_k:
            case phase_t::restitch_k: verify_(phase_t::restitch_k); break;
            default: break;
            }
        }

        /** Whether the plan runs a round's phase, whose every barrier a caller's progress hears. */
        bool rounding() const noexcept {
            if (phase_ == phase_t::health_k)
                return swept_after_ == phase_t::none_k || swept_after_ == phase_t::restitch_k;
            return phase_ >= phase_t::heal_k && phase_ <= phase_t::restitch_k;
        }

        /** The phase the plan is in. */
        phase_t phase() const noexcept { return phase_; }

        /** Whether the plan moves entries, which a refine never does. */
        bool moving() const noexcept { return false; }

        /** The position of the round's phase in hand among its six, from the opening health check
         *  to the closing one; zero outside a round's phases. */
        std::size_t ordinal() const noexcept { return rounding() ? round_ordinal(phase_, swept_after_) : 0; }

        /** Units of the round's phase in hand done so far, at a barrier before it advances: entries
         *  swept and marked by a check, and otherwise all of the phase's. */
        std::size_t completed() const noexcept { return phase_ == phase_t::health_k ? health.swept() : total(); }

        /** Units of the round's phase in hand in all: islands healed, hubs drawn, entries one
         *  gather files, or entries re-stitched. */
        std::size_t total() const noexcept {
            switch (phase_) {
            case phase_t::health_k: return 2 * std::size_t {health.visible};
            case phase_t::heal_k: return health.islands();
            case phase_t::hubs_k: return first_hubs_(health.visible);
            case phase_t::gather_k: return gathered_;
            case phase_t::restitch_k: return health.visible;
            default: return 0;
            }
        }

        /** Opens the next round once one ended, from the root the closing check kept: the count and
         *  the changed marks start over, and the opening check sweeps again. A no-op mid-round. */
        void next_round() noexcept {
            if (!index_ || phase_ != phase_t::done_k || failed(status_)) return;
            if (swept_after_ != phase_t::none_k && swept_after_ != phase_t::restitch_k) return;
            tally_ = {};
            verify_(phase_t::none_k);
        }

        /**
         *  @brief The round so far: complete once @ref advance answered false, otherwise
         *      @c cancelled_k; a failure a phase kept answers instead.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c cancelled_k A phase still follows, or the round was stopped.
         *  - @c device_k The device failed a phase.
         */
        expected<refine_report_t> finish() const noexcept {
            if (!index_) return {{}, status_code_t::uninitialized_k};
            if (failed(status_)) return {tally_, status_};
            bool const done = phase_ == phase_t::done_k && tally_.verdict != refine_verdict_t::stopped_k;
            return {tally_, done ? status_code_t::success_k : status_code_t::cancelled_k};
        }

      private:
        friend struct vram_apple_index;
        friend reshard_plan_t;

        /** Opens the check following @p phase from the root the last one kept. */
        void verify_(phase_t phase) noexcept {
            health.restart_sweep();
            swept_after_ = phase;
            phase_ = phase_t::health_k;
        }

        /** Takes in a converged check: opens the round after the opening one, concludes it after
         *  the closing one; answers whether a phase follows. */
        bool converge_() noexcept {
            reachability_t const reachability = health.finish();
            tally_.reachability = reachability;
            if (swept_after_ == phase_t::none_k) open_round_(reachability);
            else phase_ = phase_t::done_k, tally_.verdict = concluded_verdict(tally_);
            return phase_ != phase_t::done_k;
        }

        /** Takes in the opening check: seeds the round's draws from what it measured, so a round
         *  over a changed graph draws anew while the ID order decides nothing, starts the changed
         *  marks over, and opens the heal. */
        void open_round_(reachability_t const& opening) noexcept {
            tally_.opening = opening;
            tally_.islands_found = opening.audited_nodes - opening.reachable_nodes;
            round_seed_ = mix_bits(
                opening.audited_nodes ^
                mix_bits(opening.reachable_nodes ^
                         mix_bits(opening.unlinked ^ mix_bits(opening.underlinked ^ opening.depth_from_root))));
            std::memset(health.template column<byte_t>(health.layout.changed), 0,
                        divide_round_up<std::size_t>(health.visible, 32) * sizeof(u32_t));
            phase_ = health.visible ? phase_t::heal_k : phase_t::done_k;
        }

        /** Hubs one gather over @p entries draws, as every tower draws them. */
        std::size_t hubs_for_(std::size_t entries) const noexcept {
            return round_hubs(entries, sizing_.hubs_per_draw, sizing_.capacity);
        }

        /** Hubs the round's first gather over @p entries draws, as @ref round_first_hubs counts. */
        std::size_t first_hubs_(std::size_t entries) const noexcept {
            return round_first_hubs(entries, sizing_.hubs_per_draw, sizing_.capacity);
        }

        /** Draws the first gather's hubs by seeded bijection over the visible entries, ascending,
         *  into its one run over every ID. */
        void draw_hubs_() noexcept {
            std::size_t const visible = health.visible, hubs = first_hubs_(visible);
            std::span<node_id_t> const hub_nodes {health.template column<node_id_t>(health.layout.hub_nodes), hubs};
            for (std::size_t rank = 0; rank != hubs; ++rank)
                hub_nodes[rank] = static_cast<node_id_t>(seeded_bijection(rank, visible, round_seed_));
            partial_sort(hub_nodes, hubs);
            u32_t* const runs = health.template column<u32_t>(health.layout.runs);
            u32_t* const run_items = health.template column<u32_t>(health.layout.run_items);
            runs[0] = 0, runs[1] = static_cast<u32_t>(visible), runs[2] = 0, runs[3] = static_cast<u32_t>(hubs);
            run_items[0] = 0, run_items[1] = static_cast<u32_t>(divide_round_up(visible, hubs));
            runs_ = 1, neighborhoods_ = 0, gathered_ = visible;
        }

        /** Sorts the entries by the neighborhood the gather gave each, after the first gather by
         *  @ref order_by_neighborhood and after a later one by @ref split_wide_neighborhoods, in
         *  the starts planning reserved; then writes, for every neighborhood still over capacity,
         *  its run and its hubs, drawn among its own entries as the RAM round's @c gather_run_
         *  draws them, and answers whether a gather runs once more over those alone. */
        bool order_neighborhoods_() noexcept {
            round_layout_t const& layout = health.layout;
            std::size_t const visible = health.visible;
            std::span<u16_t const> const tags {health.template column<u16_t>(layout.entry_neighborhoods), visible};
            std::span<node_id_t> const entries {health.template column<node_id_t>(layout.neighborhood_entries),
                                                visible};
            std::span<node_id_t> const offsets {health.template column<node_id_t>(layout.neighborhood_offsets),
                                                split_room(visible, sizing_.hubs_per_draw)};
            u32_t const* const runs = health.template column<u32_t>(layout.runs);
            neighborhood_order_t const order = neighborhoods_
                                                   ? split_wide_neighborhoods(tags, entries, offsets, neighborhoods_,
                                                                              sizing_.capacity, sizing_.hubs_per_draw)
                                                   : order_by_neighborhood(tags, runs[3], entries, offsets);
            neighborhoods_ = order.neighborhoods;
            tally_.neighborhoods = order.neighborhoods, tally_.widest_neighborhood = order.widest;
            if (order.widest <= sizing_.capacity) return false;
            node_id_t* const hub_nodes = health.template column<node_id_t>(layout.hub_nodes);
            u32_t* const run_words = health.template column<u32_t>(layout.runs);
            u32_t* const run_items = health.template column<u32_t>(layout.run_items);
            std::size_t hubs = 0, items = 0;
            runs_ = gathered_ = 0;
            for (std::size_t neighborhood = 0; neighborhood != order.neighborhoods; ++neighborhood) {
                node_id_t const first = offsets[neighborhood];
                std::size_t const width = offsets[neighborhood + 1] - first;
                if (width <= sizing_.capacity) continue;
                std::size_t const drawn = hubs_for_(width);
                u64_t const seed = mix_bits(round_seed_ ^ first);
                for (std::size_t rank = 0; rank != drawn; ++rank)
                    hub_nodes[hubs + rank] = entries[first + seeded_bijection(rank, width, seed)];
                u32_t* const run = run_words + 4 * runs_;
                run[0] = first, run[1] = static_cast<u32_t>(width), run[2] = static_cast<u32_t>(hubs);
                run[3] = static_cast<u32_t>(drawn);
                run_items[runs_++] = static_cast<u32_t>(items);
                items += divide_round_up(width, drawn), hubs += drawn, gathered_ += width;
            }
            run_items[runs_] = static_cast<u32_t>(items);
            return true;
        }

        /** Commits the gather over the runs in hand, a first one's over IDs. */
        status_t queue_gather_(traversal_context_t& context) noexcept {
            metal_arguments_t arguments = index_->arguments_();
            arguments.end_id = health.visible;
            arguments.runs_count = static_cast<u32_t>(runs_);
            arguments.pass = neighborhoods_ ? 1u : 0u;
            u32_t* const words = health.counters();
            words[metal_round_claimed_k] = 0;
            return index_->launch_neighborhoods_(vram_kernel_t::gather_neighborhoods_k, arguments, health.layout,
                                                 sizing_.groups, context);
        }

        /** Commits the restitch of every neighborhood the order holds. */
        status_t queue_restitch_(traversal_context_t& context) noexcept {
            metal_arguments_t arguments = index_->arguments_();
            arguments.end_id = health.visible;
            arguments.neighborhoods = static_cast<u32_t>(neighborhoods_);
            arguments.slice_bytes = static_cast<u32_t>(index_->stitch_slice_bytes_());
            u32_t* const words = health.counters();
            words[metal_round_claimed_k] = 0;
            for (u32_t const tally : {metal_round_added_k, metal_round_dropped_k}) words[tally] = words[tally + 1] = 0;
            return index_->launch_neighborhoods_(vram_kernel_t::restitch_k, arguments, health.layout, sizing_.groups,
                                                 context);
        }

        batch_lock_t batch_;
        vram_apple_index* index_ = nullptr;

        /** What the round's columns are sized for, which the gather and the restitch read. */
        round_sizing_t sizing_ {};

        /** Neighborhoods the order by neighborhood holds, none before the first gather was ordered;
         *  the runs the next gather files; and the entries those runs span. */
        std::size_t neighborhoods_ = 0, runs_ = 0, gathered_ = 0;

        /** What seeds every draw of the round. */
        u64_t round_seed_ = 0;

        /** The phase the current check follows, which decides the phase after it converges. */
        phase_t swept_after_ = phase_t::none_k;
        phase_t phase_ = phase_t::done_k;
        status_t status_ = status_code_t::success_k;
        refine_report_t tally_;
    };

    /**
     *  @brief One reshard in flight from this index into its targets, in the shape of the CUDA and
     *      HIP plans: @ref queue a phase, @ref wait for it, @ref advance past it, and @ref finish
     *      once no phase follows.
     *
     *  The partition grows on the device; the barrier after its last batch scatters the parts into
     *  the targets on the host over unified memory and publishes them; each target then runs one
     *  round, as @ref refine_plan_t does, in turn, the reports folding over the targets.
     *  @ref next_round runs another round on every target from the first. Holds the source's batch
     *  lock through the moves and the bound target's through its round.
     */
    struct reshard_plan_t {
        reshard_plan_t() noexcept = default;
        reshard_plan_t(reshard_plan_t&&) noexcept = default;
        reshard_plan_t& operator=(reshard_plan_t&&) noexcept = default;

        /**
         *  @brief Submits the current phase on @p context 's stream: a partition batch, or the
         *      bound target's round's phase, after a @ref next_round binding the first target.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c device_k A phase could not be encoded.
         *  - Those of @ref plan_refine, binding a target.
         */
        status_t queue(traversal_context_t& context, refine_config_t config) noexcept {
            if (!source_) return status_code_t::uninitialized_k;
            if (failed(status_)) return status_;
            if (moving()) return status_ = source_->queue_rounds_(split_, context);
            if (unbound_()) bind_target_(context);
            if (failed(status_) || ended_()) return status_;
            return rounds_.queue(context, config);
        }

        /**
         *  @brief Waits on @p context for the submitted phase.
         *
         *  @par Statuses:
         *  - @c device_k The device failed the phase.
         */
        status_t wait(traversal_context_t& context) noexcept {
            if (!source_ || failed(status_)) return status_;
            if (moving()) return status_ = context.wait();
            return rounds_.index_ ? rounds_.wait(context) : status_;
        }

        /** The barrier after a phase, one caller once it finished, answering whether one follows:
         *  grows the partition on or ends it in the moves, advances the bound target's round, or
         *  folds its ended round and binds the next target. Once the last target settled, it
         *  settles nothing more. */
        bool advance(traversal_context_t& context) noexcept {
            if (!source_ || failed(status_)) return false;
            if (moving()) return advance_partition_(context);
            if (unbound_()) bind_target_(context);
            if (failed(status_) || ended_()) return false;
            if (rounds_.advance(context)) return true;
            if (failed(rounds_.status_)) return false;
            return settle_target_(context);
        }

        /** Stops the bound target's round, as @ref refine_plan_t::stop does, and every later
         *  target heals and verifies alone; the partition runs on. */
        void stop() noexcept {
            if (!moving()) rounds_.stop();
        }

        /** Whether the plan runs a round's phase, whose every barrier a caller's progress hears. */
        bool rounding() const noexcept { return !moving() && !ended_() && rounds_.rounding(); }

        /** The phase the plan is in: the partition through the moves, then the bound target's
         *  round's; none while @ref next_round left the first target to bind. */
        phase_t phase() const noexcept {
            if (moving()) return moves_;
            if (ended_()) return phase_t::done_k;
            return rounds_.index_ ? rounds_.phase() : phase_t::none_k;
        }

        /** Whether the plan still moves entries: its partition, whose last barrier scatters. */
        bool moving() const noexcept { return moves_ != phase_t::done_k; }

        /** Where the bound round's phase sits among its six, and its units done and in all; zero
         *  in the moves. */
        std::size_t ordinal() const noexcept { return rounding() ? rounds_.ordinal() : 0; }
        std::size_t completed() const noexcept { return moving() || ended_() ? 0 : rounds_.completed(); }
        std::size_t total() const noexcept { return moving() || ended_() ? 0 : rounds_.total(); }

        /** Runs another round on every target once every target's ended, from the first; the
         *  report starts over. A no-op in the moves and mid-round. */
        void next_round() noexcept {
            if (!source_ || moving() || failed(status_) || !ended_()) return;
            // The first target binds on the next call taking a context, as `plan_refine` needs one.
            rounds_ = refine_plan_t {};
            split_.bound_target = 0;
            split_.settled = 0;
            split_.report = {};
        }

        /**
         *  @brief The rounds so far, folded over the targets: complete once @ref advance answered
         *      false, otherwise @c cancelled_k; a failure a phase kept answers instead.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is unbound, as a refused one answers.
         *  - @c cancelled_k A phase still follows, or a round was stopped.
         *  - @c device_k The device failed a phase.
         *  - Those of @ref plan_refine, binding a target.
         */
        expected<refine_report_t> finish() const noexcept {
            if (!source_) return {{}, status_code_t::uninitialized_k};
            if (moving()) return {{}, failed(status_) ? status_ : status_code_t::cancelled_k};
            refine_report_t result = split_.report;
            status_t outcome = status_;
            if (!ended_() && rounds_.index_) {
                auto const [bound, bound_status] = rounds_.finish();
                if (split_.settled) result += bound;
                else result = bound;
                if (succeeded(outcome)) outcome = bound_status;
            }
            if (succeeded(outcome) && (!ended_() || result.verdict == refine_verdict_t::stopped_k))
                outcome = status_code_t::cancelled_k;
            return {result, outcome};
        }

      private:
        friend struct vram_apple_index;

        /** Whether every target's round ended and settled. */
        bool ended_() const noexcept { return !moving() && split_.bound_target >= split_.targets.size(); }

        /** Whether @ref next_round left the first target to bind. */
        bool unbound_() const noexcept { return !moving() && !ended_() && !rounds_.index_; }

        /** The barrier after a partition batch: grows on while a batch still placed nodes, and once
         *  one placed none places the nodes no batch reached, assigns every node its slot, scatters
         *  them on the host over unified memory, files their keys, publishes every target, lets the
         *  source go and binds the first target holding entries. */
        bool advance_partition_(traversal_context_t& context) noexcept {
            vram_apple_index& source = *source_;
            if (source.placing_(split_)) return true;
            source.place_islands_(split_);
            split_.next_round = 0;
            assign_slots_();
            source.scatter_(split_);
            publish_parts_();
            moves_ = phase_t::done_k;
            split_.bound_target = 0;
            bind_target_(context);
            return succeeded(status_) && !ended_();
        }

        /** Turns every source node's part into its slot in place, in ID order within each part. */
        void assign_slots_() noexcept {
            split_layout_t const layout = split_layout_(split_.entries, split_.targets.size());
            u32_t* const slots = reinterpret_cast<u32_t*>(split_.device_bytes.host + layout.node_parts);
            for (split_part_t& part : split_.parts) part = {};
            for (std::size_t node = 0; node != split_.entries; ++node) {
                std::size_t const part = slots[node];
                ++split_.parts[part].entries;
                slots[node] = static_cast<u32_t>(part * split_.part_capacity + split_.parts[part].assigned_nodes++);
            }
        }

        /** Files every source key under its slot in its target, publishes every target's nodes at
         *  once under the closed claim words, and empties the source. */
        void publish_parts_() noexcept {
            vram_apple_index& source = *std::exchange(split_.source, nullptr);
            split_layout_t const layout = split_layout_(split_.entries, split_.targets.size());
            u32_t const* const slots = reinterpret_cast<u32_t const*>(split_.device_bytes.host + layout.node_parts);
            for (std::size_t node = 0; node != split_.entries; ++node)
                split_.targets[slots[node] / split_.part_capacity]->keyed_.assign(
                    static_cast<node_id_t>(slots[node] % split_.part_capacity),
                    source.keyed_[static_cast<node_id_t>(node)]);
            for (std::size_t index = 0; index != split_.targets.size(); ++index) {
                vram_apple_index& target = *split_.targets[index];
                std::size_t const entries = split_.parts[index].entries;
                u32_t* const words = metal_words(target.counters_);
                words[metal_published_k] = static_cast<u32_t>(entries);
                target.publish_keys_({0, static_cast<node_id_t>(entries)});
                if (status_t const published = target.publish_(0, entries); failed(published)) status_ = published;
            }
            source.forget_();
        }

        /** Binds the round to the bound target or the next one holding nodes, under its own batch
         *  hold and over the round buffer @ref plan_reshard sized for a whole part; a target after
         *  a stop heals and verifies alone. Past the last, the plan has ended. */
        void bind_target_(traversal_context_t& context) noexcept {
            while (split_.bound_target != split_.targets.size() && !split_.parts[split_.bound_target].entries)
                ++split_.bound_target;
            if (ended_()) return;
            auto [bound, status] = split_.targets[split_.bound_target]->plan_refine(split_.config, context);
            if (failed(status)) {
                status_ = status;
                return;
            }
            bool const stopped = split_.settled && split_.report.verdict == refine_verdict_t::stopped_k;
            rounds_ = std::move(bound);
            if (stopped) rounds_.tally_.verdict = refine_verdict_t::stopped_k;
        }

        /** Folds the bound target's ended round into the plan's report and binds the next target,
         *  answering whether a phase follows. */
        bool settle_target_(traversal_context_t& context) noexcept {
            if (split_.settled++) split_.report += rounds_.tally_;
            else split_.report = rounds_.tally_;
            ++split_.bound_target;
            bind_target_(context);
            return succeeded(status_) && !ended_();
        }

        /** The bound target's round, unbound in the moves and after @ref next_round. */
        refine_plan_t rounds_;

        /** The split's state: its targets, their parts and the folded reports. */
        split_t split_;

        /** The index the entries move out of; null in a refused plan. */
        vram_apple_index* source_ = nullptr;

        /** The phase of the moves still pending; done once the parts published. */
        phase_t moves_ = phase_t::done_k;
        status_t status_ = status_code_t::success_k;
    };

    /**
     *  @brief Opens one refinement round over the published nodes on @p context 's stream, as the
     *      RAM @c plan_refine does, so no phase allocates.
     *
     *  Takes the batch lock, grows the context's round buffer to the check's columns and the
     *  round's, and sizes the hubs and the neighborhood capacity from the threadgroup memory one
     *  GPU core keeps, never more than the device grants.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p context rides another device or cannot submit.
     *  - @c out_of_memory_k The round buffer cannot grow.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        refine_plan_t plan;
        plan.batch_ = batch_lock_t {batch_};
        std::size_t const rows = size();
        if (rows) {
            if (!counters_) return {std::move(plan), status_code_t::uninitialized_k};
            if (context.device() != device_.ordinal || !context)
                return {std::move(plan), status_code_t::config_mismatch_k};
            plan.sizing_ = round_sizing_(rows, config.cache_bytes);
            if (status_t const carved = carve_round_(plan.health, rows, plan.sizing_, context); failed(carved))
                return {std::move(plan), carved};
            plan.health.restart_sweep();
        }
        plan.index_ = this;
        plan.swept_after_ = phase_t::none_k;
        plan.phase_ = rows ? phase_t::health_k : phase_t::done_k;
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief One @ref plan_refine driven to its end on the calling thread, waiting once per pass.
     *
     *  @par Statuses:
     *  - Those of @ref plan_refine and of the plan's phases.
     */
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(config, context);
        if (failed(status)) return {{}, status};
        do
            if (status_t const queued = plan.queue(context, config); failed(queued) || failed(plan.wait(context)))
                break;
        while (plan.advance(context));
        return plan.finish();
    }

    /**
     *  @brief Opens a reshard of @p sources into @p targets, the CUDA and HIP plan's twin: this
     *      index, the one source, splits into one graph-local part per target, the partition
     *      growing on the device and the moves running on the host over unified memory.
     *
     *  One root per part through @ref seeded_bijection under @p config.seed, then rounds of growth
     *  over the links on the device, each part capped at the nodes over the targets rounded up,
     *  and each node a round reaches kept by its lowest-ID proposer, so equal seeds over equal
     *  graphs split alike, as on NVIDIA and AMD. The nodes no round reached enter the smallest
     *  parts in ID order on the host, which then assigns the slots in ID order within each part,
     *  moves every node's vector, norms and neighbors into its target, the links inside its part
     *  rebased and those across dropped, files the keys, publishes every target at once and
     *  empties this index. Each target then runs one round, as @ref plan_refine does, the moved
     *  part's islands healed and its neighborhoods re-stitched, and the reports fold.
     *
     *  The caller keeps this index's and every target's claim word closed for the plan's life, and
     *  nothing searches this index meanwhile. Targets must be empty, distinct, and share this
     *  index's device, metric and geometry.
     *
     *  @param[in] sources This index alone: an Apple mono-index splits one source into its targets.
     *  @param[in] targets The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] config The rounds' knobs and the seed picking the parts' roots; a shard capacity
     *      is refused, as a mono-index has no shards to cap.
     *  @param[in,out] context The driving context, whose round buffer takes a whole part's round.
     *  @param[in] rows The rows this index holds once the appends planned ahead of the split land,
     *      as a gather into it before its cut plans them; the rows it holds now otherwise.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k No sources or no targets, a null one, or an index named twice
     *    within a list.
     *  - @c unsupported_k A nonzero @c shard_capacity, or any shape but this index alone into two
     *    or more targets: a pack, a recut or several sources into several targets.
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k This index among the targets, a non-empty target, one of another
     *    device, metric or geometry, or a @p context of another device or one that cannot submit.
     *  - @c contended_k This index's or a target's claim word is open.
     *  - @c overflow_k Part slots past the 32-bit IDs.
     *  - @c out_of_memory_k The partition's buffer or the round buffer cannot grow.
     */
    expected<reshard_plan_t> plan_reshard(std::span<vram_apple_index* const> sources,
                                          std::span<vram_apple_index* const> targets, reshard_config_t config,
                                          traversal_context_t& context, std::size_t rows) noexcept {
        reshard_plan_t plan;
        for (std::span<vram_apple_index* const> const listed : {sources, targets}) {
            if (listed.empty()) return {std::move(plan), status_code_t::invalid_argument_k};
            if (status_t const refused = listed_refusal(listed); failed(refused)) return {std::move(plan), refused};
        }
        if (config.shard_capacity) return {std::move(plan), status_code_t::unsupported_k};
        if (sources.size() != 1 || sources[0] != this || targets.size() < 2)
            return {std::move(plan), status_code_t::unsupported_k};
        std::size_t const parts = targets.size();
        if (!counters_) return {std::move(plan), status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {std::move(plan), status_code_t::config_mismatch_k};
        for (vram_apple_index const* into : targets)
            if (into == this || into->device_.ordinal != device_.ordinal || into->kernel_ != kernel_ ||
                !(into->geometry_ == geometry_) || into->size())
                return {std::move(plan), status_code_t::config_mismatch_k};
        if (!claims_.closed())
            return {std::move(plan), {status_code_t::contended_k, "The source's claim word is open"}};
        for (vram_apple_index const* into : targets)
            if (!into->claims_.closed())
                return {std::move(plan), {status_code_t::contended_k, "A target's claim word is open"}};
        split_t& split = plan.split_;
        if (failed(split.targets.reserve_exactly(parts)) || failed(split.parts.resize(parts)))
            return {std::move(plan), status_code_t::out_of_memory_k};
        for (vram_apple_index* const into : targets) split.targets.push_back_reserved(into);
        split.source = this;
        split.entries = rows;
        split.part_capacity = divide_round_up<std::size_t>(rows, parts);
        split.config = config.refine;
        if (parts * split.part_capacity >= (std::numeric_limits<u32_t>::max)())
            return {std::move(plan), status_code_t::overflow_k};
        // Every target holds a full part's rows, sentinel-filled, before the partition runs.
        for (vram_apple_index* const into : split.targets) {
            batch_lock_t held {into->batch_};
            if (status_t const bound = into->bind_(); failed(bound)) return {std::move(plan), bound};
            if (status_t const grown = into->grow_nodes_(split.part_capacity); failed(grown))
                return {std::move(plan), grown};
        }
        if (rows) {
            split_layout_t const layout = split_layout_(rows, parts);
            split.device_bytes = context.buffers_allocator_.allocate(layout.device_bytes);
            if (!split.device_bytes) return {std::move(plan), status_code_t::out_of_memory_k};
            // The round buffer takes a whole part's round now, so no target grows it later.
            round_layout_t const part_layout = round_layout_(
                split.part_capacity, round_sizing_(split.part_capacity, split.config.cache_bytes));
            bool fresh = false;
            if (status_t const grown = context.ensure_(context.round_, part_layout.bytes, fresh); failed(grown))
                return {std::move(plan), grown};
            seed_partition_(split, config.seed);
        }
        else split.bound_target = parts;
        plan.source_ = this;
        plan.moves_ = rows ? phase_t::partition_k : phase_t::done_k;
        return {std::move(plan), status_code_t::success_k};
    }

    /** @ref plan_reshard of the rows this index holds now. */
    expected<reshard_plan_t> plan_reshard(std::span<vram_apple_index* const> sources,
                                          std::span<vram_apple_index* const> targets, reshard_config_t config,
                                          traversal_context_t& context) noexcept {
        return plan_reshard(sources, targets, config, context, size());
    }

    /**
     *  @brief One @ref plan_reshard driven to its end on the calling thread, waiting once per pass.
     *
     *  @par Statuses:
     *  - Those of @ref plan_reshard and of the plan's phases.
     */
    expected<refine_report_t> reshard(std::span<vram_apple_index* const> sources,
                                      std::span<vram_apple_index* const> targets, reshard_config_t config,
                                      traversal_context_t& context) noexcept {
        auto [plan, status] = plan_reshard(sources, targets, config, context);
        if (failed(status)) return {{}, status};
        do
            if (status_t const queued = plan.queue(context, config.refine);
                failed(queued) || failed(plan.wait(context)))
                break;
        while (plan.advance(context));
        return plan.finish();
    }

    /**
     *  @brief Measures how much of the graph a search reaches from one seeded root, on the device:
     *      depths both ways, in-link counts, and the islands' components, as the RAM audit
     *      reports them. Grows the context's round buffer to the check's columns once.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p context rides another device or cannot submit.
     *  - @c out_of_memory_k The round buffer cannot grow.
     *  - @c device_k The device failed a pass.
     */
    expected<reachability_t> health(traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_code_t::success_k};
        if (!counters_) return {{}, status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_code_t::config_mismatch_k};
        health_plan_t check;
        if (status_t const carved = carve_round_(check, rows, {}, context); failed(carved)) return {{}, carved};
        check.restart_sweep();
        do {
            if (status_t const queued = check.queue(context); failed(queued)) return {{}, queued};
            if (status_t const waited = context.wait(); failed(waited)) return {{}, waited};
        } while (check.advance());
        return {check.finish(), status_code_t::success_k};
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /** One save in flight: the watermark it took and its tasks, each copying rows straight out of
     *  the storage. */
    struct [[nodiscard]] save_plan_t {
        columns_t columns {};
        node_id_t visible = 0;
        std::size_t covered = 0;

        save_plan_t() noexcept = default;
        save_plan_t(vram_apple_index& index, columns_t columns, node_id_t visible) noexcept
            : columns(columns), visible(visible), index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), visible(other.visible), covered(other.covered),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;
        ~save_plan_t() noexcept = default;

        struct task_t {
            std::size_t position = 0;
            iota_view<node_id_t> index_nodes;
            iota_view<node_id_t> source_nodes;
        };

        task_t operator[](std::size_t position) const noexcept {
            task_t task;
            if (position >= tasks().size()) return task;
            std::size_t const first = position * batch_rows();
            std::size_t const bound = min_of(first + batch_rows(), static_cast<std::size_t>(visible));
            task.position = position;
            task.index_nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            task.source_nodes = task.index_nodes;
            return task;
        }
        indexed_range<save_plan_t> tasks() const noexcept {
            return {this, index_ && visible ? divide_round_up<std::size_t>(visible, batch_rows()) : 0};
        }
        std::size_t batch_rows() const noexcept {
            return min_of(static_cast<std::size_t>(visible), checkpoint_batch_rows(index_->row_bytes()));
        }

        /**
         *  @brief Copies the task's rows out of the storage: vectors unpadded, neighbors with every
         *      ID past the watermark folded to the sentinel, and norms where the columns take them.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The task is empty, or the columns cannot hold its rows.
         */
        status_t save(task_t const& task) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            iota_view<node_id_t> const rows = task.index_nodes;
            if (rows.empty()) return status_code_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const first = task.source_nodes.first;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, first, rows.size(), scalar_row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, first, rows.size(), neighbors_row_bytes);
            bytes_span_t const norms_rows = flat_rows(columns.norms, first, rows.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_code_t::config_mismatch_k;
            shared_lock_t storage {index_->storage_};
            for (std::size_t row = 0; row != rows.size(); ++row)
                std::memcpy(vectors_rows.data() + row * scalar_row_bytes,
                            index_->vectors_matrix_.host + (rows.first + row) * index_->row_bytes(), scalar_row_bytes);
            if (!norms_rows.empty())
                std::memcpy(norms_rows.data(), index_->norms_column_.host + rows.first * sizeof(norms_t),
                            norms_rows.size());
            node_id_t const* const source = index_->neighbor_rows_() + rows.first * connectivity;
            node_id_t* const neighbor_slots = reinterpret_cast<node_id_t*>(neighbors_rows.data());
            for (std::size_t slot = 0; slot != rows.size() * connectivity; ++slot)
                neighbor_slots[slot] = source[slot] < visible ? source[slot] : node_id_unknown_k;
            fu::standard_atomic_ref<std::size_t> {covered}.store_add(rows.size(), std::memory_order_acq_rel);
            return status_code_t::success_k;
        }

        /**
         *  @brief Pronounces the snapshot complete once every task's rows were copied to the
         *      columns, spending the plan.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some task's rows were never copied.
         */
        status_t finish() noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != nodes)
                return status_code_t::config_mismatch_k;
            index_ = nullptr;
            return status_code_t::success_k;
        }

      private:
        vram_apple_index* index_ = nullptr;
    };

    /**
     *  Opens a save into @p columns.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, since the storage cannot
     *    be probed for settlement, or @p columns are too short for the rows.
     *  - @c uninitialized_k The index never grew.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization) noexcept {
        if (synchronization == synchronization_t::concurrent_k) return {{}, status_code_t::config_mismatch_k};
        if (!counters_) return {{}, status_code_t::uninitialized_k};
        std::size_t const rows = size();
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return {{}, status_code_t::config_mismatch_k};
        return {save_plan_t {*this, columns, static_cast<node_id_t>(rows)}, status_code_t::success_k};
    }

    /**
     *  Serializes every row under the watermark into @p columns on the calling thread.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, or @p columns are too
     *    short for the rows.
     *  - @c uninitialized_k The index never grew.
     */
    status_t save(columns_t columns, synchronization_t synchronization) noexcept {
        auto [plan, status] = plan_save(columns, synchronization);
        if (failed(status)) return status;
        for (typename save_plan_t::task_t const task : plan.tasks())
            if (status_t const saved = plan.save(task); failed(saved)) return saved;
        return plan.finish();
    }

    /** One load in flight: the rows it appends at the watermark, under its own claim or a claim
     *  word its caller closed, holding the growth hold for its whole life, since growth would
     *  relocate the storage its tasks write; @ref finish derives the norms no task supplied and
     *  publishes. */
    struct [[nodiscard]] load_plan_t {
        node_id_t base = 0;
        std::size_t rows = 0;
        std::size_t covered = 0;
        std::size_t covered_norms = 0;

        load_plan_t() noexcept = default;
        load_plan_t(vram_apple_index& index, batch_lock_t&& batch, node_id_t base, std::size_t rows) noexcept
            : base(base), rows(rows), index_(&index), batch_(std::move(batch)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), rows(std::exchange(other.rows, 0)), covered(other.covered),
              covered_norms(other.covered_norms), index_(std::exchange(other.index_, nullptr)),
              batch_(std::move(other.batch_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan turns its rows back to sentinels under the hold, since the next claim
         *  goes onto them without growing, and gives its claim back. */
        ~load_plan_t() noexcept {
            if (!index_ || !rows) return;
            std::size_t const neighbors_row_bytes = index_->connectivity() * sizeof(u32_t);
            std::memset(index_->neighbors_matrix_.host + base * neighbors_row_bytes, 0xFF, rows * neighbors_row_bytes);
            index_->unclaim_(base, rows);
        }

        struct task_t {
            std::size_t position = 0;
            iota_view<node_id_t> index_nodes;
            iota_view<node_id_t> source_nodes;
        };

        task_t operator[](std::size_t position) const noexcept {
            task_t task;
            if (position >= tasks().size()) return task;
            std::size_t const first = static_cast<std::size_t>(base) + position * batch_rows();
            std::size_t const bound = min_of(first + batch_rows(), static_cast<std::size_t>(base) + rows);
            task.position = position;
            task.index_nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            task.source_nodes = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            return task;
        }
        indexed_range<load_plan_t> tasks() const noexcept {
            return {this, index_ && rows ? divide_round_up(rows, batch_rows()) : 0};
        }
        std::size_t batch_rows() const noexcept { return min_of(rows, checkpoint_batch_rows(index_->row_bytes())); }

        /**
         *  @brief Copies the task's rows out of @p columns into the storage: vectors padded, norms
         *      where supplied, neighbors rebased from @p source_range onto the appended range and
         *      compacted, and row @c r of the task's source nodes keyed <tt>key_of(r)</tt>.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k The task is empty, @p columns cannot hold its rows or hold rows
         *    wider than the index's, or the rebased IDs pass the plan.
         */
        template <typename key_of_type_>
        status_t load(task_t const& task, columns_view_t columns, iota_view<node_id_t> source_range,
                      key_of_type_&& key_of) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            iota_view<node_id_t> const range = task.index_nodes;
            if (range.empty()) return status_code_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const column_connectivity = columns.connectivity ? columns.connectivity : connectivity;
            if (column_connectivity > connectivity)
                return {status_code_t::config_mismatch_k, "Neighbor rows wider than the index"};
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const column_row_bytes = column_connectivity * sizeof(node_id_t);
            std::size_t const first = task.source_nodes.first;
            bytes_view_t const vectors_rows = flat_rows(columns.vectors, first, range.size(), scalar_row_bytes);
            bytes_view_t const neighbors_rows = flat_rows(columns.neighbors, first, range.size(), column_row_bytes);
            bytes_view_t const norms_rows = flat_rows(columns.norms, first, range.size(), sizeof(norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_code_t::config_mismatch_k;
            node_id_t const shift = static_cast<node_id_t>(range.first - task.source_nodes.first + source_range.first);
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_code_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes();
            for (std::size_t row = 0; row != range.size(); ++row) {
                byte_t* const vector_row = index_->vectors_matrix_.host + (range.first + row) * row_bytes;
                std::memcpy(vector_row, vectors_rows.data() + row * scalar_row_bytes, scalar_row_bytes);
                std::memset(vector_row + scalar_row_bytes, 0, row_bytes - scalar_row_bytes);
            }
            if (index_->stores_norms() && !norms_rows.empty()) {
                std::memcpy(index_->norms_column_.host + range.first * sizeof(norms_t), norms_rows.data(),
                            norms_rows.size());
                fu::standard_atomic_ref<std::size_t> {covered_norms}.store_add(range.size(), std::memory_order_acq_rel);
            }
            // The rebase reads whole rows, so a widened row's tail must hold the all-ones sentinel.
            for (std::size_t row = 0; row != range.size(); ++row) {
                byte_t* const neighbor_row = index_->neighbors_matrix_.host + (range.first + row) * neighbors_row_bytes;
                std::memcpy(neighbor_row, neighbors_rows.data() + row * column_row_bytes, column_row_bytes);
                std::memset(neighbor_row + column_row_bytes, 0xFF, neighbors_row_bytes - column_row_bytes);
            }
            std::span<node_id_t> const copied_rows {index_->neighbor_rows_() + range.first * connectivity,
                                                    range.size() * connectivity};
            rebase_neighbors(copied_rows, source_range, shift, node_id_unknown_k);
            compact_neighbors(copied_rows, connectivity, range.first, node_id_unknown_k);
            for (node_id_t node = range.first; node != range.bound; ++node)
                index_->keyed_.assign(node, key_of(static_cast<node_id_t>(first + node - range.first)));
            fu::standard_atomic_ref<std::size_t> {covered}.store_add(range.size(), std::memory_order_acq_rel);
            return status_code_t::success_k;
        }

        /** Files every appended row under the key its task copied, which @ref finish then makes
         *  visible - the one writer of the key map, once every task was copied and before the
         *  finish; with a task missing it files nothing. */
        void publish_keys() noexcept {
            if (!index_ || fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return;
            index_->publish_keys_({base, static_cast<node_id_t>(base + rows)});
        }

        /**
         *  @brief Copies the task's rows out of @p source 's own storage rather than columns, as a
         *      reshard packing into one target drives it.
         *
         *  Both keep padded rows of one geometry in shared buffers, so every column moves whole on
         *  the host with its keys, and the neighbors rebase from @p source_range onto the appended
         *  range and compact. The source takes no rows until the copy returns.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Another geometry, or rows outside the plan or the source.
         */
        status_t load(task_t const& task, vram_apple_index& source, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            iota_view<node_id_t> const range = task.index_nodes;
            if (range.empty() || range.size() != task.source_nodes.size() || task.source_nodes.bound > source.size() ||
                !same_geometry(source, *index_))
                return status_code_t::config_mismatch_k;
            node_id_t const shift = range.first - task.source_nodes.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_code_t::config_mismatch_k;
            std::size_t const row_bytes = index_->row_bytes(), connectivity = index_->connectivity();
            std::size_t const into = range.first, from = task.source_nodes.first, count = range.size();
            shared_lock_t storage {source.storage_};
            std::memcpy(index_->vectors_matrix_.host + into * row_bytes, source.vectors_matrix_.host + from * row_bytes,
                        count * row_bytes);
            if (index_->stores_norms()) {
                std::memcpy(index_->norms_column_.host + into * sizeof(norms_t),
                            source.norms_column_.host + from * sizeof(norms_t), count * sizeof(norms_t));
                fu::standard_atomic_ref<std::size_t> {covered_norms}.store_add(count, std::memory_order_acq_rel);
            }
            std::span<node_id_t> const copied_rows {index_->neighbor_rows_() + into * connectivity,
                                                    count * connectivity};
            std::memcpy(copied_rows.data(), source.neighbor_rows_() + from * connectivity, copied_rows.size_bytes());
            rebase_neighbors(copied_rows, source_range, shift, node_id_unknown_k);
            compact_neighbors(copied_rows, connectivity, range.first, node_id_unknown_k);
            for (std::size_t row = 0; row != count; ++row)
                index_->keyed_.assign(static_cast<node_id_t>(into + row),
                                      source.keyed_[static_cast<node_id_t>(from + row)]);
            fu::standard_atomic_ref<std::size_t> {covered}.store_add(count, std::memory_order_acq_rel);
            return status_code_t::success_k;
        }

        /**
         *  @brief Derives the norms no task supplied on the device, lifts the hold, and publishes
         *      the appended nodes with the keys @ref publish_keys filed.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or spent.
         *  - @c config_mismatch_k Some task's rows were never copied, or the index keeps it, as its
         *    @c status says.
         *  - @c cancelled_k or @c unsupported_layout_k The index keeps it, as its @c status says.
         */
        status_t finish() noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (fu::standard_atomic_ref<std::size_t> {covered}.load(std::memory_order_acquire) != rows)
                return status_code_t::config_mismatch_k;
            if (rows) {
                if (index_->stores_norms() &&
                    fu::standard_atomic_ref<std::size_t> {covered_norms}.load(std::memory_order_acquire) != rows)
                    if (status_t const derived = index_->derive_norms_(base, rows); failed(derived))
                        return index_->fail_(derived);
                u32_t* const words = metal_words(index_->counters_);
                words[metal_published_k] = static_cast<u32_t>(base + rows);
                batch_.unlock();
            }
            vram_apple_index* const index = std::exchange(index_, nullptr);
            return rows ? index->publish_(base, rows) : status_code_t::success_k;
        }

      private:
        vram_apple_index* index_ = nullptr;
        batch_lock_t batch_;
    };

    /**
     *  @brief Opens a load of @p rows at the watermark, sourced per task, as the sharded index
     *      drives its parts: claims them, or places them under the claim word its caller closed; a
     *      batch or load in flight refuses it.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c contended_k Another batch or load is in flight.
     *  - @c overflow_k The rows pass the ceiling.
     *  - @c cancelled_k, @c config_mismatch_k or @c unsupported_layout_k The index keeps it, as
     *    @c status says.
     */
    expected<load_plan_t> plan_load(std::size_t rows) noexcept {
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (!rows) return {load_plan_t {*this, {}, static_cast<node_id_t>(claimed_()), 0}, status_code_t::success_k};
        batch_lock_t batch {batch_, std::try_to_lock};
        if (!batch.owns_lock()) return {{}, {status_code_t::contended_k, "Another batch or load is in flight"}};
        if (status_t const bound = bind_(); failed(bound)) return {{}, bound};
        // A closed word is the caller's, who refines or reshards around the load; an open one is
        // claimed like a batch's, so no batch goes beside it.
        bool const claiming = !claims_.closed();
        node_id_t base = static_cast<node_id_t>(size());
        if (claiming) {
            auto const [first_id, claim_status] = claim_(rows);
            if (failed(claim_status)) return {{}, claim_status};
            base = first_id;
        }
        else if (rows > ceiling_nodes_ - base) return {{}, status_code_t::overflow_k};
        if (status_t const grown = grow_nodes_(base + rows); failed(grown)) {
            unclaim_(base, rows);
            return {{}, grown};
        }
        return {load_plan_t {*this, std::move(batch), base, rows}, status_code_t::success_k};
    }

    /**
     *  @brief Appends the whole source on the calling thread; columns shorter than @p rows are
     *      refused before a row is claimed.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p columns are shorter than @p rows, hold neighbor rows wider
     *    than the index's or norms it does not store, or @c status keeps it.
     *  - @c contended_k Another batch or load is in flight.
     *  - @c overflow_k The rows pass the ceiling.
     *  - @c cancelled_k or @c unsupported_layout_k The index keeps it, as @c status says.
     */
    status_t load(columns_view_t columns, keys_view_t keys, std::size_t rows,
                  iota_view<node_id_t> source_range) noexcept {
        std::size_t const column_connectivity = columns.connectivity ? columns.connectivity : connectivity();
        if (keys.size() < rows || columns.vectors.size() < rows * packed_row_bytes() ||
            column_connectivity > connectivity() ||
            columns.neighbors.size() < rows * column_connectivity * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(norms_t))))
            return status_code_t::config_mismatch_k;
        auto [plan, status] = plan_load(rows);
        if (failed(status)) return status;
        for (typename load_plan_t::task_t const task : plan.tasks())
            if (status_t const loaded = plan.load(task, columns, source_range,
                                                  [&](node_id_t row) noexcept { return keys[row]; });
                failed(loaded))
                return loaded;
        plan.publish_keys();
        return plan.finish();
    }

#pragma endregion Checkpoints

    /** The sharded twin publishes and fails its shard's batches, and walks under its own
     *  snapshot. */
    template <typename, nk_dtype_t, nk_capability_t, typename, typename>
    friend struct vram_apple_indexes;
    template <typename, template <typename> class shards_atomic_ref_template_, typename, typename>
    friend struct util::shard_set;

  private:

#pragma region Batches

    /** The device half of @c insert: writes @p keys, stages and runs @p new_rows at @p first_id,
     *  and returns with their IDs, the rows invisible until the caller publishes them - which the
     *  sharded twin does for every task at once. The caller holds the claim over them; a refusal
     *  before the launch keeps no failure, one after it keeps the failure. */
    expected<insert_result_t> submit_insert_(keys_view_t keys, rows_view_t new_rows, node_id_t first_id,
                                             node_ids_span_t assigned_ids, traversal_context_t& context,
                                             insert_config_t config) noexcept {
        std::size_t const rows = new_rows.rows;
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_code_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_code_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (status_t const refusal = rows_refusal_(new_rows); failed(refusal)) return {{}, refusal};

        // Growth relocates the storage, so it runs under the hold a reserve takes too, and the
        // first one builds the pipelines every size below reads.
        {
            batch_lock_t batch {batch_};
            if (status_t const bound = bind_(); failed(bound)) return {{}, bound};
            if (status_t const grown = grow_nodes_(first_id + rows); failed(grown)) return {{}, grown};
        }
        // The growth covered the keys, and the claim makes this thread the map's one writer.
        for (std::size_t row = 0; row != rows; ++row) keyed_.assign(first_id + static_cast<node_id_t>(row), keys[row]);
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
        u32_t const candidate_slots = candidate_slots_for_(expansion);
        u32_t const blocks = device_.multiprocessor_count * resident_blocks_(vram_kernel_t::insert_k, candidate_slots);
        metal_arguments_t arguments = arguments_();
        if (status_t const sized = shape_arena_(context, arguments, build_arena_shape_(expansion, candidate_slots, 1),
                                                blocks, rows);
            failed(sized))
            return {{}, sized};
        arguments.first_id = static_cast<u32_t>(first_id);
        arguments.end_id = static_cast<u32_t>(first_id + rows);
        arguments.selection_nodes = walk_selection_nodes_(expansion, candidate_slots);
        arguments.expansion = expansion;
        reset_counters_(counters_);

        // The claim admits one batch at a time, so the batch drains before the next claims.
        status_t launched = status_code_t::success_k;
        {
            shared_lock_t storage {storage_};
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
        return {result, status_code_t::success_k};
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
    template <typename kernel_enum_type_>
    u32_t resident_blocks_(kernel_enum_type_ kernel, u32_t slots) const noexcept {
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
        return status_code_t::success_k;
    }

    /** Bytes of the f32 row an import widens f64 rows through before narrowing them at unit RMS. */
    std::size_t widened_bytes_() const noexcept { return dimensions() * sizeof(f32_t); }

    /** Bytes of the back-link chains of @p rows rows that leave @p total_nodes behind, in @p waves:
     *  a head per row, a next link per pick, the touched list as long, and each wave's count. */
    std::size_t links_bytes_(std::size_t rows, std::size_t total_nodes, std::size_t waves) const noexcept {
        return (total_nodes + 2 * rows * geometry_.connectivity + waves) * sizeof(u32_t);
    }

    /** Why @p rows cannot cross into the staging: a width other than the index's, or rows closer
     *  than one packed row apart, a zero stride repeating the first. */
    status_t rows_refusal_(rows_view_t rows) const noexcept {
        if (rows.columns != dimensions()) return status_code_t::config_mismatch_k;
        if (rows.stride && rows.stride < packed_bytes(rows.dtype, rows.columns))
            return status_code_t::unsupported_layout_k;
        return status_code_t::success_k;
    }

    /** Imports @p rows into @p staging at the vectors matrix pitch in the stored scalar, the
     *  fragment padding zeroed, widening through the f32 row at @p widened where they narrow at
     *  unit RMS from f64. */
    void stage_rows_(byte_t* staging, rows_view_t rows, byte_t* widened) const noexcept {
        nk::vector_span<f32_t> const widened_row {reinterpret_cast<f32_t*>(widened), dimensions()};
        for (std::size_t row = 0; row != rows.rows; ++row) {
            byte_t* const staged_row = staging + row * row_bytes();
            // Zeroed whole, not past the packed bytes, as packed casts write whole bytes only.
            std::memset(staged_row, 0, row_bytes());
            import_row<typename elementwise_ladder<import_capability_k>::kit_t,
                       typename rmscast_ladder<scalar_t, import_capability_k>::kit_t>(
                rows, row, kernel_, nk::vector_span<scalar_t> {reinterpret_cast<scalar_t*>(staged_row), dimensions()},
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
    metal_pipeline_state_t pipeline_(vram_dense_kernel_t kernel, u32_t candidate_slots) const noexcept {
        return dense_pipelines_[static_cast<u32_t>(kernel)][gpu_rung_index(candidate_slots)];
    }

    /** Builds every pipeline this dtype's kernels need, once at the first @c reserve, so no launch
     *  waits on the device's registry, the arena frontier's rung included; a pipeline too
     *  register-heavy to hold one threadgroup of four simdgroups refuses with @c unsupported_k. */
    status_t load_pipelines_() noexcept {
        metal_context_t* const context = vectors_allocator_.context();
        kernel_tables<metal_module_t>* const kernels = vectors_allocator_.kernels();
        if (!context || !kernels) return status_code_t::device_k;
        char const* const source = metal_source<vector_dtype_, capability_>();
        vram_kernel_symbol_t symbol;
        for (metal_rung<vram_kernel_t> const rung : metal_rungs) {
            metal_kernel_symbol(symbol, rung);
            auto const [state, status] = kernels->pipeline(*context->api, context->device, source, symbol.text);
            if (failed(status)) return status;
            if (state.max_threads < metal_simdgroups_k * metal_simdgroup_width_k) return status_code_t::unsupported_k;
            pipelines_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)] = state;
        }
        for (metal_rung<vram_dense_kernel_t> const rung : metal_dense_rungs) {
            metal_kernel_symbol(symbol, rung);
            auto const [state, status] = kernels->pipeline(*context->api, context->device, source, symbol.text);
            if (failed(status)) return status;
            if (state.max_threads < metal_simdgroups_k * metal_simdgroup_width_k) return status_code_t::unsupported_k;
            dense_pipelines_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)] = state;
        }
        return status_code_t::success_k;
    }

    /** One searching simdgroup's arena slice on the rung of @p candidate_slots, the arena
     *  frontier's over @p expansion at zero, then the batch spans and the query span CUDA keeps in
     *  shared memory. */
    gpu_arena_shape_t walk_arena_shape_(u32_t expansion, u32_t candidate_slots) const noexcept {
        gpu_arena_shape_t shape = arena_shape_for(
            walk_selection_nodes_(expansion, candidate_slots),
            candidate_slots ? frontier_kind_t::register_ladder_k : frontier_kind_t::wide_arena_k);
        shape.scratch_bytes = build_scratch_bytes_per_warp(0);
        return shape;
    }

    /** Rounds @p expansion_factor to a register rung in ranks of @ref wave_lanes. */
    u32_t candidate_slots_for_(u32_t expansion_factor) const noexcept {
        return gpu_candidate_slots(expansion_factor, wave_lanes());
    }

    /** Candidates a walk's diversity selection caches: all of the walk's candidates, as the CPU's
     *  selection judges, @p candidate_slots ranks per lane on the register ladder, or the arena
     *  frontier's @p expansion past it. */
    u32_t walk_selection_nodes_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return candidate_slots ? candidate_slots * wave_lanes() : expansion;
    }

    /** Candidates a re-selection caches, a back-link round's: a row of the node's own neighbors, a
     *  row's worth of incoming appended nodes, and the node's own row last. */
    u32_t reselection_nodes_() const noexcept { return 2 * static_cast<u32_t>(connectivity()) + 1; }

    /** Bytes of one builder simdgroup's slice, its walk's or its re-selection's, whichever caches
     *  more: an insert and its back-link share one arena shape. */
    u32_t builder_scratch_bytes_per_warp_(u32_t expansion, u32_t candidate_slots) const noexcept {
        return build_scratch_bytes_per_warp(
            max_of(walk_selection_nodes_(expansion, candidate_slots), reselection_nodes_()));
    }

    /** One building simdgroup's arena slice for @p walks walks on the rung of @p candidate_slots,
     *  the arena frontier's over @p expansion at zero, then the builders' scratch, which the slice
     *  always keeps, since threadgroup memory would bound the row width. */
    gpu_arena_shape_t build_arena_shape_(u32_t expansion, u32_t candidate_slots, u32_t walks) const noexcept {
        gpu_arena_shape_t shape = arena_shape_for(
            walk_selection_nodes_(expansion, candidate_slots),
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
        std::size_t const touched_offset = next_offset + rows * geometry_.connectivity * sizeof(u32_t);
        std::size_t const counts_offset = touched_offset + rows * geometry_.connectivity * sizeof(u32_t);
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
            metal_launch_t launch {context.api(), context.stream(),
                                   pipeline_(vram_kernel_t::insert_k, candidate_slots).pipeline};
            if (!launch) return status_code_t::device_k;
            bind_storage_(launch, wave_arguments);
            launch.bind(context.staging_, metal_staging_buffer_k, std::size_t {offset} * row_bytes());
            launch.bind(context.arena_, metal_arena_buffer_k);
            launch.bind(context.staging_, metal_counters_buffer_k, rows_bytes + 2 * wave * counters_bytes_k);
            bind_links(launch, wave);
            if (status_t const submitted = context.submit_(launch, blocks, threadgroup_threads_()); failed(submitted))
                return submitted;
            metal_launch_t back_link {context.api(), context.stream(),
                                      pipeline_(vram_kernel_t::back_link_k, 0).pipeline};
            if (!back_link) return status_code_t::device_k;
            wave_arguments.selection_nodes = reselection_nodes_();
            bind_storage_(back_link, wave_arguments);
            back_link.bind(context.arena_, metal_arena_buffer_k);
            back_link.bind(context.staging_, metal_counters_buffer_k, rows_bytes + (2 * wave + 1) * counters_bytes_k);
            bind_links(back_link, wave);
            if (status_t const submitted = context.submit_(back_link, blocks, threadgroup_threads_());
                failed(submitted))
                return submitted;
            wave_first = wave_end;
        }
        return status_code_t::success_k;
    }

    /** Recomputes the norms of rows @b [first,first+rows) on the device. */
    status_t derive_norms_(std::size_t first, std::size_t rows) noexcept {
        metal_arguments_t arguments = arguments_();
        arguments.first_id = static_cast<u32_t>(first);
        arguments.end_id = static_cast<u32_t>(first + rows);
        metal_context_t* const context = vectors_allocator_.context();
        if (!context) return status_code_t::device_k;
        shared_lock_t storage {storage_};
        metal_launch_t launch {*context->api, context->queue,
                               pipeline_(vram_dense_kernel_t::derive_norms_k, 0).pipeline};
        if (!launch) return status_code_t::device_k;
        launch.bind_bytes(&arguments, sizeof(arguments), metal_arguments_buffer_k);
        launch.bind(vectors_matrix_, metal_vectors_buffer_k);
        launch.bind(norms_column_, metal_norms_buffer_k);
        if (status_t const submitted = launch.submit(
                walk_blocks_(rows, resident_blocks_(vram_dense_kernel_t::derive_norms_k, 0)), threadgroup_threads_());
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
        if (!queries_count) return {{}, status_code_t::success_k};
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {{}, status_code_t::config_mismatch_k};
        if (expected<search_result_t> const submitted = submit_search_(queries, context, config, visible, heads);
            !submitted)
            return submitted;
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        read_search_(context, result_ids, result_distances);
        return {spent_(context, config.count), status_code_t::success_k};
    }

    /** The submission half of @c search: stages the queries and any indexer @p heads and launches
     *  the walks over the @p visible rows; until the context finishes, its staging holds the
     *  matches @c read_search_ reads back. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(rows_view_t queries, traversal_context_t& context, search_config_t config,
                                             std::size_t visible, heads_type_ const& heads = {}) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        std::size_t const queries_count = queries.rows;
        context.walk_output_offsets_ = {};
        if (config.traversal != traversal_t::walk_k) return {{}, status_code_t::unsupported_k};
        if (!queries_count) return {{}, status_code_t::success_k};
        std::size_t heads_count = 0;
        if constexpr (indexer_k) {
            heads_count = heads.weights.extent(1);
            if (heads.weights.extent(0) != queries_count || heads.heads.extent(0) != queries_count * heads_count ||
                heads.heads.extent(1) != dimensions())
                return {{}, status_code_t::config_mismatch_k};
        }
        // The expansion covers the count, so the ladder's candidates hold every rank, and past it
        // the arena frontier ranks everything it saw.
        u32_t const expansion = gpu_walk_breadth(config.expansion_factor, config.count);
        u32_t const candidate_slots = candidate_slots_for_(expansion);
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
        context.walk_output_offsets_ = {queries_count, config.count, ids_offset, distances_offset, 0};
        if (status_t const walked = launch_walk_(context, arguments, candidate_slots, ids_offset, distances_offset,
                                                 distances_offset, queries_count * row_bytes(), rows_bytes,
                                                 metal_output_t::distances_k);
            failed(walked))
            return {{}, walked};
        return {{config.count}, status_code_t::success_k};
    }

    /** Copies the matches the last search on @p context left in its staging into @p result_ids
     *  and, unless it is empty, @p result_distances; call it once the context finished. */
    void read_search_(traversal_context_t const& context, nk::matrix_span<node_id_t> result_ids,
                      nk::matrix_span<distance_t> result_distances) const noexcept {
        auto const& offsets = context.walk_output_offsets_;
        bool const wants_distances = result_distances.extent(0) != 0;
        auto const* const ids = reinterpret_cast<node_id_t const*>(context.staging_.host + offsets.first_offset);
        auto const* const distances = reinterpret_cast<distance_t const*>(context.staging_.host +
                                                                          offsets.second_offset);
        for (std::size_t query = 0; query != offsets.queries; ++query)
            for (std::size_t rank = 0; rank != offsets.count; ++rank) {
                result_ids(query, rank) = ids[query * offsets.count + rank];
                if (wants_distances) result_distances(query, rank) = distances[query * offsets.count + rank];
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
            return {{}, status_code_t::config_mismatch_k};
        if (!rows_count) return {{}, status_code_t::success_k};
        if (expected<search_result_t> const submitted = submit_multiply_(left, wanted, context, config, visible);
            !submitted)
            return submitted;
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        search_result_t result = spent_(context, 0);
        result.count = read_multiply_(context, node_ids, products, counts);
        return {result, status_code_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows and launches walks keeping up to
     *  @p wanted pairs each over the @p visible rows; until the context finishes, its staging holds
     *  the pairs @c read_multiply_ reads back. */
    expected<search_result_t> submit_multiply_(rows_view_t left, std::size_t wanted, traversal_context_t& context,
                                               search_config_t config, std::size_t visible) noexcept {
        std::size_t const rows_count = left.rows;
        context.walk_output_offsets_ = {};
        if (!rows_count) return {{}, status_code_t::success_k};
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion);
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
        context.walk_output_offsets_ = {rows_count, wanted, rows_bytes, products_offset, counts_offset};
        if (status_t const walked = launch_walk_(context, arguments, candidate_slots, rows_bytes, products_offset,
                                                 counts_offset, 0, 0, metal_output_t::products_k);
            failed(walked))
            return {{}, walked};
        return {{}, status_code_t::success_k};
    }

    /** Copies the pairs and the scored counts the last multiply on @p context left in its staging
     *  into @p node_ids, @p products and @p counts, and answers the pairs scored in all; call it
     *  once the context finished. */
    std::size_t read_multiply_(traversal_context_t const& context, nk::matrix_span<node_id_t> node_ids,
                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts) const noexcept {
        auto const& offsets = context.walk_output_offsets_;
        byte_t const* const staging = context.staging_.host;
        auto const* const ids = reinterpret_cast<node_id_t const*>(staging + offsets.first_offset);
        auto const* const dots = reinterpret_cast<dot_t const*>(staging + offsets.second_offset);
        auto const* const scored = reinterpret_cast<u32_t const*>(staging + offsets.third_offset);
        std::size_t total = 0;
        for (std::size_t row = 0; row != offsets.queries; ++row) {
            for (std::size_t pair = 0; pair != offsets.count; ++pair) {
                node_ids(row, pair) = ids[row * offsets.count + pair];
                products(row, pair) = dot_result_t::from_raw(dots[row * offsets.count + pair]);
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
        if (!visible) return {{}, status_code_t::success_k};
        if (query_vector.size() != dimensions()) return {{}, status_code_t::config_mismatch_k};
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const count = min_of<u32_t>(expansion, visible);
        if (failed(context.results.entries_.reserve(std::size_t {count} * 2)))
            return {{}, status_code_t::out_of_memory_k};
        context.results.entries_.extend_uninitialized(std::size_t {count} * 2);
        node_id_t* const staged_ids = reinterpret_cast<node_id_t*>(context.results.entries_.data() + count);
        distance_t* const staged_distances = reinterpret_cast<distance_t*>(staged_ids + count);
        vectors_view_t const one_query {query_vector.data(), {1, query_vector.size()}};
        config.expansion_factor = expansion;
        config.count = count;
        auto [found, search_status] = search_(one_query, nk::matrix_span<node_id_t> {staged_ids, {1, count}},
                                              nk::matrix_span<distance_t> {staged_distances, {1, count}}, context,
                                              config, visible, dummy_heads_t {});
        if (failed(search_status)) return {found, search_status};
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != node_id_unknown_k; ++live)
            context.results.entries_[live] = scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        found.count = live;
        return {found, search_status};
    }

    /** Validates @p queries, sizes the arena and the staging, and stages the query rows: what every
     *  walking verb does before its kernel, on the rung of @p candidate_slots for a walk expanding
     *  @p expansion; the row an import widens through follows the @p staging_bytes the verb asks
     *  for. */
    expected<metal_arguments_t> stage_walk_(rows_view_t queries, traversal_context_t& context, u32_t expansion,
                                            u32_t candidate_slots, std::size_t staging_bytes,
                                            std::size_t visible) noexcept {
        if (!geometry_.dimensions) return {{}, status_code_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_code_t::config_mismatch_k};
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
        arguments.filling_nodes = filling_nodes<u32_t>(expansion, static_cast<u32_t>(connectivity()));
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, staging_bytes + widened_bytes_(), fresh);
            failed(staged))
            return {{}, staged};
        stage_rows_(context.staging_.host, queries, context.staging_.host + staging_bytes);
        arguments.sampling_bound = static_cast<u32_t>(visible);
        arguments.queries_count = static_cast<u32_t>(queries.rows);
        std::memset(context.counters_.host, 0, context.counters_.bytes);
        return {arguments, status_code_t::success_k};
    }

    /** Submits the search kernel over the staged queries, its outputs at the given staging offsets;
     *  an empty index writes its sentinels at once. */
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
            return status_code_t::success_k;
        }
        u32_t const keeps_word = keeps ? 1 : 0;
        shared_lock_t storage {storage_};
        metal_launch_t launch {context.api(), context.stream(),
                               pipeline_(vram_kernel_t::search_k, candidate_slots).pipeline};
        if (!launch) return status_code_t::device_k;
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

#pragma region Round Launches

    /** What a round over @p rows rows keeps under @p cache_bytes of threadgroup memory a core, the
     *  device's own figure for zero and never more: half of it in hub rows, never below a quad, the
     *  rest in one neighborhood's rows, never below a row of mates and the entry itself, which a
     *  restitch needs to fill a row from mates alone; past what threadgroup memory holds, the rows
     *  are read in place. Then the most hubs and runs the gathers of a round write, and one
     *  restitch threadgroup per GPU core, which its staged rows fill. */
    round_sizing_t round_sizing_(std::size_t rows, std::size_t cache_bytes) const noexcept {
        std::size_t const device_bytes = device_.cache_bytes_per_core();
        std::size_t const budget = cache_bytes ? min_of(cache_bytes, device_bytes) : device_bytes;
        round_sizing_t sizing;
        sizing.hubs_per_draw = max_of<std::size_t>(budget / (2 * row_bytes()), quad_lanes_k);
        std::size_t const rest = budget - min_of(budget, sizing.hubs_per_draw * row_bytes());
        sizing.capacity = max_of(rest / row_bytes(), connectivity() + 1);
        // A later gather draws at most two hubs per capacity of a wide neighborhood, plus one each.
        sizing.hubs = max_of(round_hubs(rows, sizing.hubs_per_draw, sizing.capacity), 3 * rows / sizing.capacity + 1);
        sizing.runs = rows / sizing.capacity + 1;
        sizing.groups = device_.multiprocessor_count;
        return sizing;
    }

    /** Bytes of one restitch simdgroup's scratch, as @c metal_restitch_kernel carves it: two rows
     *  of candidates as scored nodes, a row of mates, two rows of candidate IDs and distances, and
     *  a row of kept positions. */
    std::size_t stitch_slice_bytes_() const noexcept {
        std::size_t const row_ids = connectivity();
        std::size_t const link_bytes = 3 * sizeof(u64_t) + 2 * (sizeof(u32_t) + sizeof(f32_t)) + sizeof(u32_t);
        return round_up_to_multiple<std::size_t>(row_ids * link_bytes, metal_fragment_bytes_k);
    }

    /** The round buffer's layout over @p rows rows for a round sized as @p sizing, or for a lone
     *  check's zeros, every column starting on a fragment. */
    round_layout_t round_layout_(std::size_t rows, round_sizing_t const& sizing) const noexcept {
        std::size_t offset = 0;
        auto carve = [&](std::size_t bytes) noexcept {
            std::size_t const at = offset;
            offset = round_up_to_multiple<std::size_t>(offset + bytes, metal_fragment_bytes_k);
            return at;
        };
        std::size_t const ids = rows * sizeof(u32_t), words = divide_round_up<std::size_t>(rows, 32) * sizeof(u32_t);
        round_layout_t layout;
        layout.counters = carve(metal_round_counter_words_k * sizeof(u32_t));
        layout.depths_from_root = carve(rows);
        layout.depths_to_root = carve(rows);
        layout.in_links = carve(ids);
        layout.shallower_in_links = carve(ids);
        layout.reached = carve(words);
        layout.reaches_root = carve(words);
        layout.two_ways_to_root = carve(words);
        layout.changed = carve(words);
        std::size_t const aliased = offset;
        layout.island_nodes = carve(ids);
        layout.island_components = carve(ids);
        // Two labels per island, whose slots serve the weak components' union-find, root's last.
        layout.island_labels = carve(2 * ids);
        layout.island_parents = carve(ids);
        layout.island_entries = carve(2 * ids);
        layout.island_roles = carve(ids);
        std::size_t const islands_bytes = offset;
        offset = aliased;
        layout.entry_neighborhoods = carve(rows * sizeof(u16_t));
        layout.neighborhood_entries = carve(ids);
        layout.neighborhood_offsets = carve(split_room(rows, sizing.hubs_per_draw) * sizeof(u32_t));
        layout.hub_nodes = carve(sizing.hubs * sizeof(u32_t));
        layout.runs = carve(sizing.runs * 4 * sizeof(u32_t));
        layout.run_items = carve((sizing.runs + 1) * sizeof(u32_t));
        layout.stitch = carve(sizing.groups * metal_simdgroups_k * stitch_slice_bytes_());
        layout.bytes = max_of(offset, islands_bytes);
        return layout;
    }

    /** Grows @p context 's round buffer to a round over @p rows rows sized as @p sizing, a lone
     *  check's for zeros, and binds @p health onto it. */
    status_t carve_round_(health_plan_t& health, std::size_t rows, round_sizing_t const& sizing,
                          traversal_context_t& context) noexcept {
        round_layout_t const layout = round_layout_(rows, sizing);
        bool fresh = false;
        if (status_t const grown = context.ensure_(context.round_, layout.bytes, fresh); failed(grown)) return grown;
        health.bind(*this, static_cast<node_id_t>(rows), layout, context.round_.host);
        return status_code_t::success_k;
    }

    /** Threadgroups a round pass of @p kernel over @p items spreads over, a simdgroup an item up to
     *  the grid the device runs at once, never none. */
    u32_t spread_(vram_kernel_t kernel, std::size_t items) const noexcept {
        return max_of(walk_blocks_(items, max_of(resident_blocks_(kernel, 0), 1u)), 1u);
    }

    /** One column of the round buffer and the slot it binds at. */
    struct round_column_t {
        metal_buffer_index_t slot;
        std::size_t offset;
    };

    /** Binds the storage, the round's counters and the check's columns of @p layout. */
    void bind_health_(metal_launch_t& launch, metal_arguments_t const& arguments, round_layout_t const& layout,
                      traversal_context_t const& context) const noexcept {
        bind_storage_(launch, arguments);
        round_column_t const columns[] = {
            {metal_round_counters_buffer_k, layout.counters},
            {metal_depths_from_root_buffer_k, layout.depths_from_root},
            {metal_depths_to_root_buffer_k, layout.depths_to_root},
            {metal_in_links_buffer_k, layout.in_links},
            {metal_shallower_in_links_buffer_k, layout.shallower_in_links},
            {metal_reached_buffer_k, layout.reached},
            {metal_reaches_root_buffer_k, layout.reaches_root},
            {metal_two_ways_to_root_buffer_k, layout.two_ways_to_root},
            {metal_changed_buffer_k, layout.changed},
        };
        for (auto const& [slot, offset] : columns) launch.bind(context.round_, slot, offset);
    }

    /** Commits one pass of a health check's or the heal's @p kernel over @p items on @p context 's
     *  stream, the check's columns and the islands' bound. */
    status_t launch_round_(vram_kernel_t kernel, metal_arguments_t const& arguments, round_layout_t const& layout,
                           std::size_t items, traversal_context_t& context) noexcept {
        shared_lock_t storage {storage_};
        metal_launch_t launch {context.api(), context.stream(), pipeline_(kernel, 0).pipeline};
        if (!launch) return status_code_t::device_k;
        bind_health_(launch, arguments, layout, context);
        round_column_t const columns[] = {
            {metal_island_nodes_buffer_k, layout.island_nodes},
            {metal_island_components_buffer_k, layout.island_components},
            {metal_island_labels_buffer_k, layout.island_labels},
            {metal_island_parents_buffer_k, layout.island_parents},
            {metal_island_entries_buffer_k, layout.island_entries},
            {metal_island_roles_buffer_k, layout.island_roles},
        };
        for (auto const& [slot, offset] : columns) launch.bind(context.round_, slot, offset);
        // The islands' order is one threadgroup's walk over the words, and the heal one lane's.
        bool const alone = kernel == vram_kernel_t::order_islands_k || kernel == vram_kernel_t::heal_islands_k;
        return context.submit_(launch, alone ? 1u : spread_(kernel, items), threadgroup_threads_());
    }

    /** Commits a gather's or a restitch's @p kernel over @p groups threadgroups on @p context 's
     *  stream, the check's columns and the round's bound. */
    status_t launch_neighborhoods_(vram_kernel_t kernel, metal_arguments_t const& arguments,
                                   round_layout_t const& layout, std::size_t groups,
                                   traversal_context_t& context) noexcept {
        shared_lock_t storage {storage_};
        metal_launch_t launch {context.api(), context.stream(), pipeline_(kernel, 0).pipeline};
        if (!launch) return status_code_t::device_k;
        bind_health_(launch, arguments, layout, context);
        round_column_t const columns[] = {
            {metal_entry_neighborhoods_buffer_k, layout.entry_neighborhoods},
            {metal_neighborhood_entries_buffer_k, layout.neighborhood_entries},
            {metal_neighborhood_offsets_buffer_k, layout.neighborhood_offsets},
            {metal_hub_nodes_buffer_k, layout.hub_nodes},
            {metal_runs_buffer_k, layout.runs},
            {metal_run_items_buffer_k, layout.run_items},
            {metal_stitch_buffer_k, layout.stitch},
        };
        for (auto const& [slot, offset] : columns) launch.bind(context.round_, slot, offset);
        return context.submit_(launch, groups, threadgroup_threads_());
    }

    /** Commits the heal of the islands @p health named, none when it named none. */
    status_t queue_heal_(health_plan_t& health, traversal_context_t& context) noexcept {
        if (!health.islands()) return status_code_t::success_k;
        metal_arguments_t arguments = arguments_();
        arguments.end_id = health.visible;
        arguments.islands = static_cast<u32_t>(health.islands());
        return launch_round_(vram_kernel_t::heal_islands_k, arguments, health.layout, 1, context);
    }

    /** Where a split's shared buffer keeps each array of @p nodes nodes into @p parts parts, in
     *  bytes: each node's part, which the host turns into its slot, the round's proposals, the two
     *  frontiers, one bit per node each, then per part its size and its round count, and the nodes
     *  the last two rounds placed, by the round number's parity. */
    struct split_layout_t {
        std::size_t node_parts = 0, proposals = 0, frontier = 0, next_frontier = 0;
        std::size_t part_sizes = 0, round_counts = 0, placed = 0, device_bytes = 0;
    };
    static split_layout_t split_layout_(std::size_t nodes, std::size_t parts) noexcept {
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        split_layout_t layout;
        layout.proposals = nodes * sizeof(u32_t);
        layout.frontier = layout.proposals + nodes * sizeof(u32_t);
        layout.next_frontier = layout.frontier + words * sizeof(u32_t);
        layout.part_sizes = layout.next_frontier + words * sizeof(u32_t);
        layout.round_counts = layout.part_sizes + parts * sizeof(u32_t);
        layout.placed = layout.round_counts + parts * sizeof(u32_t);
        layout.device_bytes = layout.placed + 2 * sizeof(u32_t);
        return layout;
    }

    /** Starts @p split 's partition on the host, the buffer still idle: every node unplaced and
     *  unproposed, the bitsets and counters empty, and the roots @p seed picks placed alone into
     *  the frontier of round one. */
    void seed_partition_(split_t& split, u64_t seed) noexcept {
        std::size_t const entries = split.entries, parts = split.targets.size();
        split_layout_t const layout = split_layout_(entries, parts);
        byte_t* const scratch = split.device_bytes.host;
        std::memset(scratch, 0xFF, layout.frontier);
        std::memset(scratch + layout.frontier, 0, layout.device_bytes - layout.frontier);
        u32_t* const node_parts = reinterpret_cast<u32_t*>(scratch + layout.node_parts);
        u32_t* const frontier = reinterpret_cast<u32_t*>(scratch + layout.frontier);
        u32_t* const part_sizes = reinterpret_cast<u32_t*>(scratch + layout.part_sizes);
        for (std::size_t part = 0; part != min_of(parts, entries); ++part) {
            std::size_t const root = seeded_bijection(part, entries, seed);
            node_parts[root] = static_cast<u32_t>(part);
            frontier[root / 32] |= 1u << (root % 32);
            part_sizes[part] = 1;
        }
        split.next_round = 1;
    }

    /** The scalars and the scratch slots every partition pass of @p split binds, its frontiers
     *  swapped on even rounds. */
    void bind_partition_(metal_launch_t& launch, metal_arguments_t const& arguments,
                         split_t const& split) const noexcept {
        split_layout_t const layout = split_layout_(split.entries, split.targets.size());
        bool const odd = arguments.round & 1u;
        launch.bind_bytes(&arguments, sizeof(arguments), metal_arguments_buffer_k);
        launch.bind(neighbors_matrix_, metal_neighbors_buffer_k);
        launch.bind(split.device_bytes, metal_node_parts_buffer_k, layout.node_parts);
        launch.bind(split.device_bytes, metal_proposals_buffer_k, layout.proposals);
        launch.bind(split.device_bytes, metal_frontier_buffer_k, odd ? layout.frontier : layout.next_frontier);
        launch.bind(split.device_bytes, metal_next_frontier_buffer_k, odd ? layout.next_frontier : layout.frontier);
        launch.bind(split.device_bytes, metal_part_sizes_buffer_k, layout.part_sizes);
        launch.bind(split.device_bytes, metal_round_counts_buffer_k, layout.round_counts);
        launch.bind(split.device_bytes, metal_placed_buffer_k, layout.placed);
    }

    /** Commits @p split 's growth rounds from its next one up to twice it on @p context 's stream,
     *  four passes each, so the batches double until one ends past convergence; a round queued past
     *  it leaves at entry. Every node is placed once at most, so the nodes bound the rounds. */
    status_t queue_rounds_(split_t& split, traversal_context_t& context) noexcept {
        metal_arguments_t arguments = arguments_();
        arguments.end_id = static_cast<u32_t>(split.entries);
        arguments.part_capacity = static_cast<u32_t>(split.part_capacity);
        u32_t const first_round = split.next_round, parts = static_cast<u32_t>(split.targets.size());
        shared_lock_t storage {storage_};
        for (u32_t round = first_round; round != 2 * first_round; ++round) {
            arguments.round = round;
            for (vram_kernel_t const pass : {vram_kernel_t::partition_propose_k, vram_kernel_t::partition_count_k,
                                             vram_kernel_t::partition_accept_k, vram_kernel_t::partition_fill_k}) {
                metal_launch_t launch {context.api(), context.stream(), pipeline_(pass, 0).pipeline};
                if (!launch) return status_code_t::device_k;
                bind_partition_(launch, arguments, split);
                u32_t const groups = pass == vram_kernel_t::partition_fill_k
                                         ? parts
                                         : device_.multiprocessor_count * max_of(resident_blocks_(pass, 0), 1u);
                if (status_t const submitted = context.submit_(launch, groups, threadgroup_threads_());
                    failed(submitted))
                    return submitted;
            }
        }
        split.next_round = 2 * first_round;
        return status_code_t::success_k;
    }

    /** Whether the last growth round of @p split placed anything, so more rounds follow; read once
     *  its batch finished. */
    static bool placing_(split_t const& split) noexcept {
        split_layout_t const layout = split_layout_(split.entries, split.targets.size());
        u32_t const* const placed = reinterpret_cast<u32_t const*>(split.device_bytes.host + layout.placed);
        return placed[(split.next_round - 1) & 1u] != 0;
    }

    /** Places every node no growth round reached on the host, in ID order, each entering the part
     *  smallest at its turn, the lowest-indexed among equals, as the CUDA pass does on a block. */
    static void place_islands_(split_t& split) noexcept {
        split_layout_t const layout = split_layout_(split.entries, split.targets.size());
        u32_t* const node_parts = reinterpret_cast<u32_t*>(split.device_bytes.host + layout.node_parts);
        u32_t* const part_sizes = reinterpret_cast<u32_t*>(split.device_bytes.host + layout.part_sizes);
        std::size_t const parts = split.targets.size();
        for (std::size_t node = 0; node != split.entries; ++node) {
            if (node_parts[node] != node_id_unknown_k) continue;
            std::size_t smallest = 0;
            for (std::size_t part = 1; part != parts; ++part)
                if (part_sizes[part] < part_sizes[smallest]) smallest = part;
            node_parts[node] = static_cast<u32_t>(smallest);
            ++part_sizes[smallest];
        }
    }

    /** Moves every node of @p split into its target at the slot it was assigned, on the host over
     *  unified memory: its vector and norms as they are, and its neighbors with the links inside
     *  its part rebased through their own slots and packed to the front, the links across dropped,
     *  and a node nothing survived in pointing at itself. */
    void scatter_(split_t const& split) noexcept {
        split_layout_t const layout = split_layout_(split.entries, split.targets.size());
        u32_t const* const slots = reinterpret_cast<u32_t const*>(split.device_bytes.host + layout.node_parts);
        std::size_t const capacity = split.part_capacity, connectivity = this->connectivity();
        shared_lock_t storage {storage_};
        for (std::size_t node = 0; node != split.entries; ++node) {
            u32_t const part = static_cast<u32_t>(slots[node] / capacity);
            std::size_t const position = slots[node] % capacity;
            vram_apple_index& target = *split.targets[part];
            std::memcpy(target.vectors_matrix_.host + position * row_bytes(), vectors_matrix_.host + node * row_bytes(),
                        row_bytes());
            if (stores_norms())
                std::memcpy(target.norms_column_.host + position * sizeof(norms_t),
                            norms_column_.host + node * sizeof(norms_t), sizeof(norms_t));
            node_id_t const* const from = neighbor_rows_() + node * connectivity;
            node_id_t* const into = target.neighbor_rows_() + position * connectivity;
            std::size_t kept = 0;
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                node_id_t const id = from[slot];
                if (id < split.entries && id != node && slots[id] / capacity == part)
                    into[kept++] = static_cast<node_id_t>(slots[id] % capacity);
            }
            for (std::size_t slot = kept; slot != connectivity; ++slot) into[slot] = node_id_unknown_k;
            if (!kept) into[0] = static_cast<node_id_t>(position);
        }
    }

    /**
     *  @brief Copies on @p context 's stream the packed row of every entry of @p rows, as the
     *      stored scalar, into the staging past the entries, @p output_rows rows packed back to
     *      back: a lookup's gather, which the host reads in place once the context finished.
     *
     *  @return Where the gathered rows start in the staging, or null on a refusal.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device or cannot walk, or an entry
     *    names a node not visible or an output row past @p output_rows.
     */
    expected<byte_t const*> launch_gather_vectors_(std::span<vram_gathered_row_t const> rows, std::size_t output_rows,
                                                   traversal_context_t& context) noexcept {
        if (context.device() != device_.ordinal || !context) return {nullptr, status_code_t::config_mismatch_k};
        std::size_t const visible = size();
        for (vram_gathered_row_t const& row : rows)
            if (row.node >= visible || row.output_row >= output_rows)
                return {nullptr, status_code_t::config_mismatch_k};
        std::size_t const outputs_offset = round_up_to_multiple(rows.size_bytes(), default_alignment_k);
        bool fresh = false;
        if (status_t const staged = context.ensure_(context.staging_, outputs_offset + output_rows * packed_row_bytes(),
                                                    fresh);
            failed(staged))
            return {nullptr, staged};
        std::memcpy(context.staging_.host, rows.data(), rows.size_bytes());
        if (rows.empty()) return {context.staging_.host + outputs_offset, status_code_t::success_k};
        metal_arguments_t arguments = arguments_();
        arguments.queries_count = static_cast<u32_t>(rows.size());
        arguments.packed_bytes = static_cast<u32_t>(packed_row_bytes());
        shared_lock_t storage {storage_};
        metal_launch_t launch {context.api(), context.stream(), pipeline_(vram_kernel_t::gather_vectors_k, 0).pipeline};
        if (!launch) return {nullptr, status_code_t::device_k};
        launch.bind_bytes(&arguments, sizeof(arguments), metal_arguments_buffer_k);
        launch.bind(vectors_matrix_, metal_vectors_buffer_k);
        launch.bind(context.staging_, metal_gathered_rows_buffer_k);
        launch.bind(context.staging_, metal_gathered_outputs_buffer_k, outputs_offset);
        status_t const submitted = context.submit_(
            launch, walk_blocks_(rows.size(), max_of(resident_blocks_(vram_kernel_t::gather_vectors_k, 0), 1u)),
            threadgroup_threads_());
        return {failed(submitted) ? nullptr : context.staging_.host + outputs_offset, submitted};
    }

    /**
     *  @brief Sums on @p context 's stream the @p queries_count per-query counts its staging holds
     *      from @p counts_offset bytes in, as a walk left them, into the one word that the
     *      context's @c matched reads once it finished.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The counts pass the staging, or the @p context rides another device
     *    or cannot walk.
     */
    status_t launch_sum_counts_(std::size_t counts_offset, u32_t queries_count,
                                traversal_context_t& context) const noexcept {
        if (context.device() != device_.ordinal || !context ||
            counts_offset + std::size_t {queries_count} * sizeof(u64_t) > context.staging_.bytes)
            return status_code_t::config_mismatch_k;
        metal_arguments_t arguments = arguments_();
        arguments.queries_count = queries_count;
        metal_launch_t launch {context.api(), context.stream(), pipeline_(vram_kernel_t::sum_counts_k, 0).pipeline};
        if (!launch) return status_code_t::device_k;
        launch.bind_bytes(&arguments, sizeof(arguments), metal_arguments_buffer_k);
        launch.bind(context.staging_, metal_counts_buffer_k, counts_offset);
        launch.bind(context.counters_, metal_counters_buffer_k);
        return context.submit_(launch, 1, threadgroup_threads_());
    }

#pragma endregion Round Launches

#pragma region State

    /** Keeps @p failure as the status unless one is kept already. */
    status_t fail_(status_t failure) noexcept {
        keep_first_failure<fu::standard_atomic_ref>(status_, failure);
        return failure;
    }

    /** Rows claimed by batches and loads, visible or not; a closed word holds none past the
     *  watermark. */
    std::size_t claimed_() const noexcept { return claims_.claimed(size()); }

    /** Claims @p rows from the watermark: one compare-exchange that succeeds only on an open word
     *  with no batch or load in flight, so its holder alone runs and publishes them. */
    expected<node_id_t> claim_(std::size_t rows) noexcept {
        auto const [first_id, claim_status] = claims_.claim(size(), rows, ceiling_nodes_);
        return {static_cast<node_id_t>(first_id), claim_status};
    }

    /** Gives back the claim over @p rows from @p first, which launched nothing; a word that moved
     *  on, closed by a caller around a load, stays as it is. */
    void unclaim_(std::size_t first, std::size_t rows) noexcept { claims_.unclaim(first, rows); }

    /** Closes the claim word at the watermark, so every claim refuses until @c reopen_; refused
     *  while a batch or load is in flight. */
    status_t close_() noexcept { return claims_.close(size()); }

    /** Reopens a closed claim word at the watermark, past every row a load published meanwhile; an
     *  open one stays as it is. */
    void reopen_() noexcept { claims_.reopen(size()); }

    /** Forgets every row under the closed claim word: the published neighbor rows refill with the
     *  sentinel once the walks sharing the storage drained, and the watermark drops to zero, a kept
     *  failure with it, so IDs restart from zero on the storage already grown, which nothing frees.
     *  The key map empties with them. */
    void forget_() noexcept {
        unique_lock_t storage {storage_};
        if (std::size_t const rows = size())
            std::memset(neighbors_matrix_.host, 0xFF, rows * connectivity() * sizeof(u32_t));
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(0, std::memory_order_release);
        fu::standard_atomic_ref<char const*>(status_.note).store(nullptr, std::memory_order_relaxed);
        fu::standard_atomic_ref<status_code_t>(status_.code).store(status_code_t::success_k, std::memory_order_release);
        keyed_.forget();
    }

    /** Files every row of @p range under its key ahead of its watermark: the claim holder, or the
     *  closer of the word, is the map's one writer. */
    void publish_keys_(iota_view<node_id_t> range) noexcept { keyed_.publish(range); }

    /** Row zero of the neighbors matrix, as the IDs its rows hold. */
    node_id_t* neighbor_rows_() const noexcept { return reinterpret_cast<node_id_t*>(neighbors_matrix_.host); }

    /** Turns the watermark from @p first over @p rows once every batch claimed before has turned
     *  it. */
    status_t publish_(std::size_t first, std::size_t rows) noexcept {
        for (std::size_t watermark = size(); watermark != first; watermark = size()) {
            if (status_t const kept = status(); failed(kept)) return kept;
            wait_t {}(&last_visible_id_, watermark, 0u, fu::wait_capped_t {});
        }
        fu::standard_atomic_ref<std::size_t>(last_visible_id_).store(first + rows, std::memory_order_release);
        return status_code_t::success_k;
    }

    /**
     *  @brief Binds the index at its first growth: checks the metric and the geometry, builds the
     *      pipelines, queries the device and allocates the counters.
     *
     *  Later calls answer success untouched. The caller holds the batch hold.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k The metric is not one the kits score for this scalar.
     *  - @c config_mismatch_k The dimensions are zero, the connectivity is not a power of two, or
     *    the neighbors live on another device.
     *  - @c device_k The device is not listed.
     */
    status_t bind_() noexcept {
        if (counters_) return status_code_t::success_k;
        bool const scored = vector_dtype_ == nk_u1_k
                                ? kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_hamming_k ||
                                      kernel_ == nk_kernel_jaccard_k
                                : kernel_ == nk_kernel_dot_k || kernel_ == nk_kernel_angular_k ||
                                      kernel_ == nk_kernel_sqeuclidean_k || kernel_ == nk_kernel_euclidean_k;
        if (!scored) return status_code_t::unknown_kind_k;
        if (!geometry_.dimensions || !std::has_single_bit(static_cast<u32_t>(geometry_.connectivity)))
            return status_code_t::config_mismatch_k;
        if (neighbors_allocator_.device() != device_.ordinal) return status_code_t::config_mismatch_k;
        metal_context_t* const context = vectors_allocator_.context();
        if (!context) return status_code_t::device_k;
        // The pipelines report the simdgroup width the specification names, so they build first.
        if (status_t const loaded = load_pipelines_(); failed(loaded)) return loaded;
        auto const [specs, status] = metal_device_specs_t::query(*context, device_.ordinal);
        if (failed(status)) return status;
        device_ = specs;
        counters_ = metal_allocate(*context, counters_bytes_k);
        return counters_ ? status_code_t::success_k : status_code_t::out_of_memory_k;
    }

    /** Grows every matrix and column to hold @p rows, doubling, the fresh neighbor rows
     *  sentinel-filled; the caller holds the batch hold, and the copy waits for every walk sharing
     *  the storage. */
    status_t grow_nodes_(std::size_t rows) noexcept {
        if (rows > ceiling_nodes_) return status_code_t::overflow_k;
        if (status_t const covered = keyed_.cover(rows); failed(covered)) return covered;
        if (rows <= capacity_rows_) return status_code_t::success_k;
        std::size_t const grown = max_of(rows, capacity_rows_ * 2);
        std::size_t const neighbors_row_bytes = connectivity() * sizeof(u32_t);
        metal_buffer_t vectors = vectors_allocator_.allocate(grown * row_bytes());
        metal_buffer_t norms = stores_norms() ? vectors_allocator_.allocate(grown * sizeof(norms_t))
                                              : metal_buffer_t {};
        metal_buffer_t neighbors = neighbors_allocator_.allocate(grown * neighbors_row_bytes);
        if (!vectors || !neighbors || (stores_norms() && !norms)) {
            vectors_allocator_.deallocate(vectors), vectors_allocator_.deallocate(norms);
            neighbors_allocator_.deallocate(neighbors);
            return status_code_t::out_of_memory_k;
        }
        unique_lock_t storage {storage_};
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
        return status_code_t::success_k;
    }

    metal_device_specs_t device_;
    nk_kernel_kind_t kernel_;
    geometry_t geometry_;
    std::size_t capacity_rows_ = 0;

    /** Rows the index may ever host, fixed by the first growth a @c reserve makes. */
    std::size_t ceiling_nodes_ = max_node_count_k;

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

    /** Every shared and every dense kernel's pipeline per candidate rung, owned by the device's
     *  registry. */
    metal_pipeline_state_t pipelines_[static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
    metal_pipeline_state_t dense_pipelines_[static_cast<u32_t>(vram_dense_kernel_t::count_k)][gpu_rungs_per_kernel_k] =
        {};

    /** Shared by every walk while its command buffer runs, taken alone by growth. */
    std::shared_mutex storage_;

    /** Held around every growth, a load's whole life and a refinement, whose host writes into the
     *  storage a growth would relocate. */
    std::mutex batch_;

    /** The claim word, closed while a refine, a reshard, a load or a save of the sharded index
     *  holds it. */
    claim_word_t claims_;
    std::size_t last_visible_id_ = 0;
    status_t status_ = status_code_t::success_k;

    /** Every node's key and every key's nodes: assigned and filed by the claim's holder once its
     *  rows ran, before the watermark turns. */
    keyed_t keyed_;

#pragma endregion State
};

} // namespace unum::usearch::dense
